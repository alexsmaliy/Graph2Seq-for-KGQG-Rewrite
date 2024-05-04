import torch
from torch import nn

def dropout(t: torch.Tensor, drop_prob: float, /, shared_axes=None, training = False):
    if shared_axes is None:
        shared_axes = []
    if drop_prob == 0.0 or not training:
        return t
    size = list(t.size())
    for i in shared_axes:
        size[i] = 1
    mask = (
        t.new(*size)
         .bernoulli_(1.0 - drop_prob)
         .div_(1.0 - drop_prob)
         .expand_as(t)
    )
    return t * mask
