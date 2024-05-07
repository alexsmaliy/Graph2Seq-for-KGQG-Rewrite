import numpy as np
import torch

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

def pad_2d_vals(in_vals, dim1_size, dim2_size, fills=0, dtype=np.int32):
    out_val = np.ones((dim1_size, dim2_size), dtype=dtype) * fills
    if dim1_size > len(in_vals): dim1_size = len(in_vals)
    for i in range(dim1_size):
        cur_in_vals = in_vals[i]
        cur_dim2_size = dim2_size
        if cur_dim2_size > len(cur_in_vals): cur_dim2_size = len(cur_in_vals)
        out_val[i,:cur_dim2_size] = cur_in_vals[:cur_dim2_size]
    return out_val

def pad_2d_vals_no_size(in_vals, fills=0, dtype=np.int32):
    size1 = len(in_vals)
    size2 = np.max([len(x) for x in in_vals])
    return pad_2d_vals(in_vals, size1, size2, fills=fills, dtype=dtype)

def pad_3d_vals(in_vals, dim1_size, dim2_size, dim3_size, fills=0, dtype=np.int32):
    out_val = np.ones((dim1_size, dim2_size, dim3_size), dtype=dtype) * fills
    if dim1_size > len(in_vals): dim1_size = len(in_vals)
    for i in range(dim1_size):
        in_vals_i = in_vals[i]
        cur_dim2_size = dim2_size
        if cur_dim2_size > len(in_vals_i): cur_dim2_size = len(in_vals_i)
        for j in range(cur_dim2_size):
            in_vals_ij = in_vals_i[j]
            cur_dim3_size = dim3_size
            if cur_dim3_size > len(in_vals_ij): cur_dim3_size = len(in_vals_ij)
            out_val[i, j, :cur_dim3_size] = in_vals_ij[:cur_dim3_size]
    return out_val

def pad_3d_vals_no_size(in_vals, fills=0, dtype=np.int32):
    size1 = len(in_vals)
    size2 = np.max([len(x) for x in in_vals])
    size3 = 0
    for val in in_vals:
        if len(val) > 0:
            cur_size3 = np.max([len(x) for x in val])
            if size3<cur_size3: size3 = cur_size3
    return pad_3d_vals(in_vals, size1, size2, size3, fills=fills, dtype=dtype)
