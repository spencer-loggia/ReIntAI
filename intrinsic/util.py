import torch
import numpy as np


def conv_identity_params(in_spatial, desired_kernel, stride=1):
    """
    finds convolution parameters that will maintain the same output shape
    :param in_spatial: spatial dimension to maintain
    :param desired_kernel: desired kernel, actual kernel may be smaller
    :param stride: desired stride. desired output shape scaled by stride / ((stride -1) * kernel)
    :return:

    """
    if desired_kernel == 1:
        return 1, 0
    pad = .1
    in_spatial /= stride
    kernel = min(desired_kernel, in_spatial)
    while round(pad) != pad or pad >= kernel:
        # compute padding that will maintain spatial dims during actual conv
        # pad = (((in_spatial - 1) * stride) - in_spatial + kernel) / 2
        if stride == 1:
            out = in_spatial
        else:
            out = stride * in_spatial / ((stride - 1) * kernel)
        if out.is_integer():
            pad = (stride * (out - 1) - in_spatial + kernel) / 2
        if kernel < 2:
            raise RuntimeError("Could not find kernel pad combination to maintain dimensionality")
        kernel = max(kernel - 1, 0)

    # print("Using kernel", kernel + 1, " and pad", int(pad))
    return int(kernel + 1), int(pad)


def unfold_nd(input_tensor: torch.Tensor, kernel_size: int, padding: int, spatial_dims: int, stride=1):
    """
    Unfolds an input tensor with an arbitrary number of spatial dimensions using an even kernel.
    :param input_tensor: (batch, channel, spatial_0, ... , spatial_n)
    :param kernel_size: int < spatial
    :param padding: int < kernel
    :param spatial_dims: number of spatial dimensions n
    :param stride: kernel stride length. defualt = 1
    :return: unfolded tensor. approx (batch, channels * kernel^n, spatial^n) up to padding discrepancies with stride 1
    """
    pad = [padding] * (2 * spatial_dims)
    batch_size = input_tensor.shape[0]
    channel_size = input_tensor.shape[1]
    padded = torch.nn.functional.pad(input_tensor, pad, "constant", 0)
    for i in range(spatial_dims):
        padded = padded.unfold(dimension=2 + i, size=kernel_size, step=stride)
    kernel_channel_dim = channel_size
    spatial_flat_dim = 1
    for i in range(spatial_dims):
        padded = padded.transpose(2 + i, 2 + spatial_dims + i)
        kernel_channel_dim *= padded.shape[2 + i]
        spatial_flat_dim *= padded.shape[2 + spatial_dims + i]
    # conform with Unfold modules output formatting.
    padded = padded.reshape(batch_size, kernel_channel_dim, spatial_flat_dim)
    return padded


def triu_to_square(triu_vector, n, includes_diag=False):
    """
    Converts an upper triangle vector to a full (redundant) symmetrical square matrix.
    :param tri_vector: data point vector
    :param n: size of resulting square
    :param includes_diag: whether the main diagonal is included in triu_vector
    :return: a symmetric square tensor
    """
    if includes_diag:
        offset = 0
    else:
        offset = 1
    adj = torch.zeros((n, n), dtype=torch.float)
    ind = torch.triu_indices(n, n, offset=offset)
    adj[ind[0], ind[1]] = triu_vector
    adj = (adj.T + adj)
    if includes_diag:
        adj = adj - torch.diag(torch.diagonal(adj) / 2)
    return adj
