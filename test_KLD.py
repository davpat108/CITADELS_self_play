import math
import torch.nn.functional as F
from torch import tensor
import torch



Qs = tensor([[0., 0., 0., 0., 1., 0.]])
Ps = tensor([[0., 0., 0., 0., 1., 0.]])


def kl_div(a1, a2):
    # the individual terms of the KL divergence can be calculated like this
    a1 = a1/torch.sum(a1)*3
    a2 = a2/torch.sum(a2)*3
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

#loss_terms = [Q * (math.log(Q) - math.log(P)) for Q, P in zip(Qs, Ps)]
#loss = sum(loss_terms) / len(Qs)
#
#print(loss)
