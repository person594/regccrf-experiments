import os
import re
from collections import namedtuple

Token = namedtuple('Token', ['word', 'prop', 'roleset', 'args'])

class ConllLoader:
    def __init__(self, text_col, prop_col, roleset_col, arg_col, last_arg_col=None):
        self.text_col = text_col
        self.prop_col = prop_col
        self.roleset_col = roleset_col
        self.arg_col = arg_col
        self.last_arg_col = last_arg_col

    def load(self, path):
        data = []
        for filename in sorted(os.listdir(path)):
            filepath = os.path.join(path, filename)
            if os.path.isdir(filepath):
                data += self.load(filepath)
            elif filepath.endswith(".gold_conll"):
                data += self.load_file(filepath)
        return data
        

    def load_file(self, path):
        with open(path) as f:
            documents = []
            document = []
            for line in f:
                if line.startswith("#"):
                    continue
                if line.strip() == "":
                    if len(document) > 0:
                        documents.append(document)
                        document = []
                    continue
                fields = line.strip().split()
                token = Token(
                    fields[self.text_col],
                    fields[self.prop_col],
                    fields[self.roleset_col],
                    fields[self.arg_col:self.last_arg_col]
                )
                document.append(token)
        return documents


    def write_props(self, path, data, corpus, pred=None):
        
        sentence_sexps = []
        sentences = []

        corpus_index = 0

        for sentence in data:
            n_props = len(sentence[0].args)
            sentence_sexps = []
            for prop in range(n_props):
                gold_sexp = [t.args[prop] for t in sentence]
                text, bios, old_to_new = corpus[corpus_index]
                if pred is not None:
                    bios = pred[corpus_index]
        
                old_bios = [bios[i] for i in old_to_new]
                sexp = bios_to_sexp(old_bios)
                assert len(sexp) == len(gold_sexp)
                
                sentence_sexps.append(sexp)
                corpus_index += 1
            sentences.append(sentence_sexps)

        with open(path, 'w') as f:
            for data_sentence, sentence_sexps in zip(data, sentences):
                lines = []
                for token in data_sentence:
                    if token.prop != '-' and token.roleset != '-':
                        lines.append([token.prop])
                    else:
                        lines.append(['-'])
                for sexp in sentence_sexps:
                    for i, token in enumerate(sexp):
                        lines[i].append(token)
                for line in lines:
                    f.write('\t'.join(line) + '\n')
                f.write('\n')            

    # re-implemented as simple as possible, to ensure no
    # data leakage or mistakes
    def write_gold_props(self, path, data):
        with open(path, 'w') as f:
            for sentence in data:
                lines = []
                for token in sentence:
                    if token.prop != '-' and token.roleset != '-':
                        lines.append([token.prop])
                    else:
                        lines.append(['-'])

                n_args = len(sentence[0].args)
                for i in range(n_args):
                    for j, token in enumerate(sentence):
                        lines[j].append(token.args[i])
                for line in lines:
                    f.write('\t'.join(line) + '\n')
                f.write('\n')


                
def bios_to_sexp(bios):
    sexp = []
    open_type = None
    for bio in bios:
        if open_type is not None and bio != 'I-' + open_type:
            open_type = None
            sexp[-1] += ')'
        if bio.startswith('B-'):
            open_type = bio[2:]
            sexp.append('(%s*' % open_type)
        else:
            sexp.append('*')
    if open_type is not None:
        sexp[-1] += ')'
    return sexp


def to_bios(corpus):
    roles = {}
    text_bios = []
    for document in corpus:
        text = [token.word for token in document]
        n_args = len(document[0].args)
        for i in range(n_args):
            bios = []
            labels = [token.args[i] for token in document]
            open_role = None
            for label in labels:
                match = re.match(r"\(([^\s]+)\*", label)
                if match:
                    role = match.group(1)
                    if role not in roles:
                        roles[role] = 0
                    roles[role] += 1
                    assert open_role is None
                    bios.append('B-' + role)
                    open_role = role
                else:
                    if open_role is not None:
                        bios.append('I-' + open_role)
                    else:
                        bios.append('O')

                if re.match(r".*\)", label):
                    open_role = None

            text_bios.append((text, bios))
    #print(roles)
    #breakpoint()
    return text_bios
