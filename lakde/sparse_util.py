import itertools

import torch

from lakde.kernels import ind2ptr


def graph_to_col(graph, col_offset=0):
    Ng, K = graph.shape
    row = graph.flatten()
    col = torch.arange(
        col_offset, col_offset + Ng, dtype=graph.dtype, device=graph.device
    )
    col = col[:, None].expand(-1, K).flatten()

    indices = torch.stack((row, col), dim=0)
    return indices[:, torch.argsort(indices[0])]


def graph_col_to_slices(indices, N, block_size):
    slices = []
    rowptr = ind2ptr(indices[0], N)

    # split the column
    for bi, i in enumerate(range(0, N, block_size)):
        row_start = rowptr[i]
        row_end = rowptr[min(i + block_size, N)]
        inds = indices[:, row_start:row_end].clone()
        inds[0] -= i  # adjust row so the slices have top left (0, 0)
        slices.append(inds)

    return slices


def cat_slices_and_sort(slices):
    slices = torch.cat(slices, dim=1)
    return slices[:, torch.argsort(slices[0])]


def graph_to_slices(graph, block_size):
    N, K = graph.shape
    row_slices = [[] for _ in range(0, N, block_size)]

    for j in range(0, N, block_size):
        col_slices = graph_col_to_slices(
            graph_to_col(graph[j : j + block_size], j), N, block_size
        )
        for bi, s in enumerate(col_slices):
            row_slices[bi].append(s)

    for bi, i in enumerate(range(0, N, block_size)):
        row_slices[bi] = cat_slices_and_sort(row_slices[bi])

    return row_slices


def cat_sparse(all_indices, all_values, shifts, dim=0):
    offset = 0
    shifted_indices = []
    if isinstance(shifts, int):
        shifts = itertools.repeat(shifts)
    for inds, shift in zip(all_indices, shifts):
        inds = inds.clone()
        inds[dim] += offset
        shifted_indices.append(inds)
        offset += shift

    if all_values is not None:
        return torch.cat(shifted_indices, dim=1), torch.cat(all_values)
    else:
        return torch.cat(shifted_indices, dim=1)
