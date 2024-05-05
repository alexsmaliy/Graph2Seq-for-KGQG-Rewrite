import numpy as np
import torch
from torch import nn

def create_mask(x, N, device=None):
    x = x.data
    mask = np.zeros((x.size(0), N))
    for i in range(x.size(0)):
        mask[i, :x[i]] = 1
    return send_to_device(torch.Tensor(mask), device)

def dropout(t: torch.Tensor, drop_prob: float, *, shared_axes=None, training=False):
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

def send_to_device(tensor: torch.Tensor, device: torch.device):
    tensor.to(device)
    return tensor
