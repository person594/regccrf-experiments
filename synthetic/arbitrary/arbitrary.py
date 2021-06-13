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
import pickle

batch_size = 50


space = literal(["-"])
a = literal(["a"])
b = literal(["b"])

constrained_language = (a + space)[:] | (b + space)[:]
unconstrained_language = (a | b | space)[:]

constrained = RegCCRF(constrained_language)
unconstrained = RegCCRF(unconstrained_language)

posthoc = RegCCRF(constrained_language)

# label indices should be consistent between constrained and posthoc: ensure that
assert constrained.labels == posthoc.labels


constrained_lps = []
constrained_Hs = []

posthoc_lps = []
posthoc_Hs = []

for k in range(1, 21):


    #strings = list(constrained_language)
    strings = [['a', '-' ]*k]*3 + [['b', '-']*k]*1
    length = 2*k


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
            # get some samples, and make sure that both crfs have reached their equilibrium distributions
            with torch.no_grad():

                for p_u, p_ph in zip(unconstrained.parameters(), posthoc.parameters()):
                    p_ph.copy_(p_u)

                y = torch.tensor([constrained.encode(s) for s in strings])
                batch_constrained_logits = torch.stack([constrained_logits]*len(strings))
                batch_unconstrained_logits = torch.stack([unconstrained_logits]*len(strings))


                constrained_H = constrained.loss(batch_constrained_logits, y)
                constrained_lp = constrained.log_p(batch_constrained_logits, y)[0]


                posthoc_H = posthoc.loss(batch_unconstrained_logits, y) 
                posthoc_lp = posthoc.log_p(batch_unconstrained_logits, y)[0]


                print("k=%d"% k)
                print("Constrained p: ", float(torch.exp(constrained_lp)))
                print("Post-hoc p: ", float(torch.exp(posthoc_lp)))
                print("Constrained H:", float(constrained_H))
                print("Posthoc H:", float(posthoc_H))
                print()

    constrained_lps.append(float(constrained_lp))
    constrained_Hs.append(float(constrained_H))
    posthoc_lps.append(float(posthoc_lp))
    posthoc_Hs.append(float(posthoc_H))
    with open('out.pkl', 'wb') as f:
        pickle.dump((constrained_lps, constrained_Hs, posthoc_lps, posthoc_Hs), f)
