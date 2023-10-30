import math
import torch.nn.functional as F
import torch.nn as nn
from torch import tensor
import torch



Qs = tensor([[36., 49., 29., 15., 53., 27.]], dtype=torch.float64)*100
Ps = tensor([[-0.0765,  0.1868,  0.1787, -0.7810, -0.0509,  0.3807]], dtype=torch.float64)


def kl_div(a1, a2):
    # the individual terms of the KL divergence can be calculated like this
    manual_kl = (a2.softmax(1) * (a2.log_softmax(1) - a1.log_softmax(1)))

    # applying necessary transformations
    a1ready = a1.log_softmax(1)
    a2ready = a2.softmax(1)

    print('\nSums')
    print(manual_kl.sum())
    print(F.kl_div(a1ready, a2ready, reduction='none').sum())
    print(F.kl_div(a1ready, a2ready, reduction='sum'))

    print('\nMeans')
    print(manual_kl.mean())
    print(F.kl_div(a1ready, a2ready, reduction='none').mean())
    print(F.kl_div(a1ready, a2ready, reduction='mean'))

    print('\nBatchmean')
    print(manual_kl.mean(0).sum())
    print(F.kl_div(a1ready, a2ready, reduction='batchmean'))


kl_div(Qs, Ps)

#loss = nn.BCEWithLogitsLoss()
#input = torch.randn(3, requires_grad=True)
#target = torch.empty(3).random_(2)
#output = loss(input, target)
#output.backward()