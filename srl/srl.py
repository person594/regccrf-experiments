"""srl

Usage:
  srl.py train [--crf] <model-name> <train-path> <dev-path>
  srl.py predict <model-name> <corpus-path> <out-path>

"""


import random
from itertools import count
import logging
from datetime import datetime

from docopt import docopt

import torch
import torch.nn as nn
import torch.optim as optim

from automic import *
from ccrf import  RegCCRF
from conll_loader import ConllLoader, to_bios

from bert_wrapper import BertWrapper

model_name = None
def log(*args):
    assert model_name is not None
    timestamp = datetime.now().replace(microsecond=0).isoformat(' ') + '| '
    filler = '\n' + ' ' * (len(timestamp) - 2) + '| '
    new_args = []
    for arg in args:
        arg = str(arg)
        arg = arg.replace('\n', filler)
        new_args.append(arg)

    if len(new_args) == 0:
        new_args = (filler[1:],)
    else:
        new_args[0] = timestamp + new_args[0]

    with open('logs/%s.log' % model_name, 'a') as f:
        print(*new_args, file=f)
    print(*new_args)


def free_automaton():
    """
    Mimic a normal CRF, by allowing any string
    """
    # all roles in the training set (excluding V)
    roles = [
        'ARG1', 'ARG0', 'ARG2', 'ARGM-TMP', 'ARGM-MOD', 'ARGM-ADV', 'ARGM-DIS', 'ARGM-MNR', 'ARGM-LOC', 'ARGM-NEG', 'R-ARG0', 'R-ARG1', 'ARG3', 'ARGM-PRP', 'ARG4', 'ARGM-CAU', 'ARGM-DIR', 'ARGM-PRD', 'C-ARG1', 'ARGM-ADJ', 'ARGM-EXT', 
        'ARGM-GOL', 'ARGM-PNC', 'R-ARGM-LOC', 'ARGM-LVB', 'R-ARGM-TMP', 'ARGM-COM', 'R-ARG2', 'C-ARG2', 'C-ARG0', 'ARGM-REC', 'ARG5', 
        'R-ARGM-MNR',  'R-ARG3',  'R-ARGM-CAU',  'C-ARGM-MNR',  'ARGA',  'C-ARGM-ADV',  'C-ARG4',  'R-ARGM-ADV',  'C-ARGM-EXT',  'C-ARGM-TMP',  'R-ARGM-DIR',  'R-ARG4',  'C-ARGM-LOC',  'C-ARG3',  'ARGM-PRR',  'ARGM-PRX',  'R-ARGM-PRP',  'R-ARGM-EXT',  'R-ARGM-GOL',  'R-ARGM-COM',  'ARGM-DSP',  'C-ARGM-DSP',  'R-ARGM-PNC',  'C-ARGM-CAU',  'R-ARG5',  'C-ARGM-PRP',  'C-ARGM-DIS',  'C-ARGM-ADJ',  'C-ARGM-MOD',  'R-ARGM-PRD',  'R-ARGM-MOD',  'C-ARGM-DIR',  'C-ARGM-NEG',  'C-ARGM-COM'
    ]

    labels = ["O"] + ["B-" + role for role in roles] + ["I-" + role for role in roles]

    # make a free automaton where everything goes to the same state
    A = Automaton(1, {0})
    for label in labels:
        A.transitions[0][label] |= {0}

    return A

    
def build_automaton():
    # set up the automaton -- do it manually,
    # since we can't efficiently create one compositionally

    # 6 optional arguments and one required verb

    log("Constructing Automaton...")
    core = ["ARG0", "ARG1", "ARG2", "ARG3", "ARG4"] # exclude ARG5 -- only 79 times in training set

    n_core = len(core)

    noncore = [
        'ARGM-TMP', 'ARGM-MOD', 'ARGM-ADV', 'ARGM-DIS', 'ARGM-MNR',
        'ARGM-LOC','ARGM-NEG', 'ARGM-PRP', 'ARGM-CAU', 'ARGM-DIR',
        'ARGM-PRD', 'ARGM-ADJ', 'ARGM-EXT', 'ARGM-GOL', 'ARGM-PNC',
        'R-ARG0', 'R-ARG1'
    ] # everything else only 500 times or less; exclude
    
    continuation = ['C-ARG1']

    A = Automaton(2**n_core, set())
    for waiting_state in range(2**n_core):
        A.transitions[waiting_state]['O'].add(waiting_state)
        A.accepting.add(waiting_state)
        for i, noncore_role in enumerate(noncore):
            successor = A.n_states
            A.add_state()
            A.transitions[waiting_state]['B-'+noncore_role] |= {successor, waiting_state}
            A.transitions[successor]['I-'+noncore_role] |= {successor, waiting_state}
        for i, core_role in enumerate(core):
            if not waiting_state & (1<<i):
                successor = A.n_states
                A.add_state()
                next_waiting = waiting_state | (1<<i)
                A.transitions[waiting_state]['B-'+core_role] |= {successor, next_waiting}
                A.transitions[successor]['I-'+core_role] |= {successor, next_waiting}
            elif 'C-' + core_role in continuation:
                successor = A.n_states
                A.add_state()
                A.transitions[waiting_state]['B-C-'+core_role] |= {successor, waiting_state}
                A.transitions[successor]['I-C-'+core_role] |= {successor, waiting_state}

    log("Trimming Automaton...")
    A = trim(A)
    log("Automaton has %d states" % A.n_states)
    return A


def retokenize(xys, tokenizer):
    """
    retokenizes according to the tokenizer
    """
    processed = []
    for text, bios in xys:
        new_tokens = [tokenizer.cls_token_id]
        new_bios = ['O']
        old_to_new = []

        # first word of the sentence has no preceeding space
        preceeding_space = False

        for word, label in zip(text, bios):
            if word.startswith('/') and len(word) > 1:
                word = word[1:]

            if word[0] in {"'", '.', ',', '-'}:
                preceeding_space = False
            
            if preceeding_space:
                word = ' ' + word

            preceeding_space = True
            
            old_to_new.append(len(new_tokens))
            token_ids = tokenizer(word)['input_ids'][1:-1]
            if label == 'B-V':
                # use a special token to mark the predicate, (but don't put it in the label sequence)
                token_ids = [tokenizer.unk_token_id] + token_ids
            new_tokens.extend(token_ids)

            if label in  {'B-V', 'I-V'}:
                label = 'O'

            if label.startswith('B-'):
                ilabel = 'I-' + label[2:]
                new_bios.extend([label] + [ilabel for _ in token_ids[1:]])
            else:
                new_bios.extend([label] * len(token_ids))

        new_tokens.append(tokenizer.eos_token_id)
        new_bios.append('O')
        processed.append((new_tokens, new_bios, old_to_new))
    return processed

def prepare_corpus(corpus, tokenizer):
    xys = to_bios(corpus)
    xys = retokenize(xys, tokenizer)

    return xys

def filter_xys(xys, A, max_len=120):
    # filter our training data to remove instances not in A
    filtered_xys = []

    
    A_labels = set()
    for transition in A.transitions:
        A_labels.update(transition.keys())

    n_fixed_bios = 0
        
    log("Filtering irregular strings")
    for text, bios, old_to_new in xys:
        fixed_bios = []
        for bio in bios:
            if bio not in A_labels:
                fixed_bios.append('O')
            else:
                fixed_bios.append(bio)
        if bios != fixed_bios:
            n_fixed_bios += 1
            bios = fixed_bios
                
        if len(A.paths(bios)) == 1:
            filtered_xys.append((text, bios, old_to_new))

    log("Stripped rare tags from %f%%" % (100 * n_fixed_bios / len(xys)))
    log("Filtered out %f%%" % (100 * (1 - len(filtered_xys) / len(xys))))

    xys = filtered_xys

    log("Filtering long strings...")
    filtered_xys = []
    for text, bios, old_to_new in xys:
        if len(text) < 120:
            filtered_xys.append((text, bios, old_to_new))

    log("Filtered out %f%%" % (100* (1 - len(filtered_xys) / len(xys))))

    return filtered_xys


def prepare_batch(batch, prepare_gold=True):
    max_len = max(len(bi[0]) for bi in batch)
    x = []
    y = []
    for i, (tokens, bios, _) in enumerate(batch):
        d = max_len - len(tokens)
        tokens = tokens + [tokenizer.pad_token_id] * d
        x.append(tokens)
        if prepare_gold:
            bios = bios + ['O'] * d
            labels = ccrf.encode(bios)
            y.append(labels)

    
    x = torch.tensor(x, dtype=torch.int64).to(device)
    if prepare_gold:
        y = torch.tensor(y, dtype=torch.int64).to(device)
        return (x, y)
    else:
        return x


def predict(corpus, bert, ccrf):
    bert.eval()
    with torch.no_grad():
        predictions = []
        # without gradients, we can fit way bigger batches in memory
        p_batch_size = 50
        for i in range(0, len(corpus), p_batch_size):
            print("Predicting %.2f%%         " % (100*i/len(corpus)), end='\r')
            batch = corpus[i:i+p_batch_size]
            x = prepare_batch(batch, False)
            encoded = bert(x)
            encoded = projection(encoded)
            paths = ccrf(encoded)
            for path in paths:
                predictions.append(ccrf.decode(path))
    print('                                        ', end='\r')
    bert.train()
    return predictions

def evaluate(pred_bios, gold_bios):

    def spans(bios):
        open_span_type = None
        span_start = None
        spans = set()
        for i, label in enumerate(bios + ['O']):
            if open_span_type is not None and label != 'I-' + open_span_type:
                spans.add((open_span_type, span_start, i))
                open_span_type = None
                span_start = None
            if label.startswith("B-"):
                open_span_type = label[2:]
                span_start = i
        return spans
    
    tp = 0
    fp = 0
    fn = 0
    
    for pred, gold in zip(pred_bios, gold_bios):
        pred_spans = spans(pred)
        gold_spans = spans(gold)
        # filter out verbs for evaluation
        pred_spans = {s for s in pred_spans if s[0] != 'V'}
        gold_spans = {s for s in gold_spans if s[0] != 'V'}
        tp += len(pred_spans & gold_spans)
        fp += len(pred_spans - gold_spans)
        fn += len(gold_spans - pred_spans)
    if tp == 0:
        precision = 0
        recall = 0
        f1 = 0
    else:
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = 2/(1/precision + 1/recall)
    log("tp", tp)
    log("fp", fp)
    log("fn", fn)
    log("precision", precision)
    log("recall", recall)
    log("f1", f1)
    return f1
        
if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    arguments = docopt(__doc__)

    if arguments['predict']:
        model_name = arguments['<model-name>']
        model_path = 'models/%s.pt' % model_name
        corpus_path = arguments['<corpus-path>']
        out_path = arguments['<out-path>']
        with open(model_path, 'rb') as f:
            bert, projection, ccrf = torch.load(f)

        loader = ConllLoader(3, 6, 7, 11, -1)
        tokenizer = bert.tokenizer
        log("Loading corpus...")
        corpus = loader.load(corpus_path)
        log("Preparing corpus...")
        xys = prepare_corpus(corpus, tokenizer)
        log("Predicting...")
        bios = predict(xys, bert, ccrf)

        loader.write_props(out_path, corpus, xys, bios)
            
    
    if arguments['train']:
        
        train_path = arguments['<train-path>']
        dev_path = arguments['<dev-path>']
        model_name = arguments['<model-name>']
        
        if arguments['--crf']:
            log("Using unconstrained CRF -- building free automaton")
            A = free_automaton()
        else:
            A = build_automaton()

        log("Instantiating Roberta...")
        bert = BertWrapper(roberta=True).to(device)
        
        tokenizer = bert.tokenizer

        loader = ConllLoader(3, 6, 7, 11, -1)

        log("Loading training corpus...")
        train_corpus = loader.load(train_path)
        log("Preparing training corpus...")
        train_xys = prepare_corpus(train_corpus, tokenizer)
        log("Filtering training data...")
        train_xys = filter_xys(train_xys, A)

        log("Loading dev corpus...")
        dev_corpus = loader.load(dev_path)
        log("Preparing dev corpus...")
        dev_xys = prepare_corpus(dev_corpus, tokenizer)

        # keep a small dev corpus around for fast evaluation
        tinydev_xys = dev_xys[:]
        random.shuffle(tinydev_xys)
        tinydev_xys = tinydev_xys[:len(tinydev_xys)//20]

        log("Instantiating CCRF...")
        ccrf = RegCCRF(A).to(device)

        log("CCRF has %d tags" % ccrf.n_tags)

        projection = nn.Linear(768, ccrf.n_labels).to(device)

        # begin training
        log("Starting Training...")

        batch_size = 2

        params = list(bert.parameters()) + list(ccrf.parameters()) + list(projection.parameters())
        optimizer = optim.Adam(params, lr=1e-5)

        chunk_size = 5000

        chunk_loss = 0

        minichunk_size = 200
        minichunk_loss = 0

        grad_agg = 4

        step = 0

        best_f1 = -1

        stagnation = 0

        max_stagnation = 50

        for epoch in count(1):
            random.shuffle(train_xys)

            for i in range(0, len(train_xys), batch_size):
                step += 1
                batch = train_xys[i:i+batch_size]
                x, y = prepare_batch(batch)
                bert.train()
                encoded = bert(x)
                # encoded: float32[batch, sequence, bert.n_units]
                encoded = projection(encoded)
                # encoded: float32[batch, sequence, ccrf.n_labels]
                loss = ccrf.loss(encoded, y)
                chunk_loss += float(loss)
                minichunk_loss += float(loss)

                if step % minichunk_size == 0:
                    log(epoch, step, minichunk_loss / minichunk_size)
                    minichunk_loss = 0

                loss = loss / grad_agg
                loss.backward()

                if step % grad_agg == 0:
                    optimizer.step()
                    optimizer.zero_grad()

                if step % chunk_size == 0:
                    log("Average loss:", chunk_loss / chunk_size)

                    log("Evaluating on dev corpus...")
                    pred_bios = predict(tinydev_xys, bert, ccrf)
                    gold_bios = [c[1] for c in tinydev_xys]
                    f1 = evaluate(pred_bios, gold_bios)
                    if f1 > best_f1:
                        best_f1 = f1
                        stagnation = 0
                        log("Best f1 so far; saving everything")
                        with open('models/%s.pt'%model_name, 'wb') as f:
                            torch.save((bert, projection, ccrf), f)
                    else:
                        stagnation += 1
                        log("stagnation: %d / %d" % (stagnation, max_stagnation))
                        if stagnation >= max_stagnation:
                            log("terminating.")
                            exit()
                    log()

                    chunk_loss = 0
