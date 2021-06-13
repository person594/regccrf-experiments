"""
Purpose:
  A CRF can be constrained at training time, or also only at inference time.
  This experiment is to determine whether the probability distributions produced
  by these two procedures are identical.  I hypothesize  that they are not.
"""


from ccrf import RegCCRF, CRF
from automic import literal

import torch
import torch.nn as nn
import torch.optim as optim

import random

batch_size = 50


space = literal(["-"])
a = literal(["a"])
b = literal(["b"])
c = literal(["c"])
d = literal(["d"])

constrained_language = (a + space + c | a + space + d | b + space + d)
unconstrained_language = (a | b | c | d | space)[:]

constrained = RegCCRF(constrained_language)
unconstrained = RegCCRF(unconstrained_language)

posthoc = RegCCRF(constrained_language)

# label indices should be consistent between constrained and posthoc: ensure that
assert constrained.labels == posthoc.labels

#strings = list(constrained_language)
strings = [['a', '-', 'c']]*4 + [['a', '-', 'd']]*3 + [['b', '-', 'd']]*3

length = len(strings[0])

constrained_logits = nn.Parameter(torch.zeros(length, constrained.n_labels, dtype=torch.float32))
unconstrained_logits = nn.Parameter(torch.zeros(length, unconstrained.n_labels, dtype=torch.float32)) 


constrained_params = list(constrained.parameters()) + [constrained_logits]
unconstrained_params = list(unconstrained.parameters()) + [unconstrained_logits]

constrained_optimizer = optim.SGD(constrained_params, lr=.1, weight_decay=0e-4)
unconstrained_optimizer = optim.SGD(unconstrained_params, lr=.1, weight_decay=0e-4)

# train the two models
for iteration in range(5000):
    batch_constrained_logits = torch.stack([constrained_logits]*batch_size)
    batch_unconstrained_logits = torch.stack([unconstrained_logits]*batch_size)

    batch_strings = [random.choice(strings) for _ in range(batch_size)]
    constrained_y = torch.tensor([constrained.encode(s) for s in batch_strings], dtype=torch.int64)
    unconstrained_y = torch.tensor([unconstrained.encode(s) for s in batch_strings], dtype=torch.int64)

    constrained_loss = constrained.loss(batch_constrained_logits, constrained_y)
    constrained_optimizer.zero_grad()
    constrained_loss.backward()
    constrained_optimizer.step()

    unconstrained_loss = unconstrained.loss(batch_unconstrained_logits, unconstrained_y)
    unconstrained_optimizer.zero_grad()
    unconstrained_loss.backward()
    unconstrained_optimizer.step()


    if iteration % 100 == 0:
        # LR decay
        for g in constrained_optimizer.param_groups:
            g['lr'] *= .9
        for g in unconstrained_optimizer.param_groups:
            g['lr'] *= .9
        print("lr", g['lr'])
        # monitor
        with torch.no_grad():
            for p_u, p_ph in zip(unconstrained.parameters(), posthoc.parameters()):
                p_ph.copy_(p_u)

            y = torch.tensor([constrained.encode(s) for s in strings])
            batch_constrained_logits = torch.stack([constrained_logits]*len(strings))
            batch_unconstrained_logits = torch.stack([unconstrained_logits]*len(strings))


            constrained_H = constrained.loss(batch_constrained_logits, y)
            constrained_lps = constrained.log_p(batch_constrained_logits, y)


            posthoc_H = posthoc.loss(batch_unconstrained_logits, y) 
            posthoc_lps = posthoc.log_p(batch_unconstrained_logits, y)


            print("Constrained p: ", torch.exp(constrained_lps))
            print("Post-hoc p: ", torch.exp(posthoc_lps))
            print("Constrained H:", float(constrained_H))
            print("Posthoc H:", float(posthoc_H))
            print()


