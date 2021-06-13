import sys

import conll_loader

corpus_dir = sys.argv[1]
outfile = sys.argv[2]

loader = conll_loader.ConllLoader(3, 6, 7, 11, -1)

data = loader.load(corpus_dir)

loader.write_gold_props(outfile, data)
