"""
A blocked matrix multiplication.

Uses three program id axes. Block size `B2` represent the batches to process out of `N2`.
Block size `B0` represent the rows of `x` to process out of `N0`. Block size `B1` represent the cols of `y` to process out of `N1`. The middle shape is `MID`.

$$z_{i, j, k} = \sum_{l} x_{i,j, l} \times y_{i, l, k} \text{ for } i = 1\ldots N_2, j = 1\ldots N_0, k = 1\ldots N_1$$

You are allowed to use `tl.dot` which computes a smaller mat mul.

Hint: the main trick is that you can split a matmul into smaller parts.

$$z_{i, j, k} = \sum_{l=1}^{L/2} x_{i,j, l} \times y_{i, l, k} +  \sum_{l=L/2}^{L} x_{i,j, l} \times y_{i, l, k} $$
"""

import torch
import triton
from torch import Tensor
import triton.language as tl
import jaxtyping
from jaxtyping import Float32, Int32

def dot_spec(x: Float32[Tensor, "4 32 32"], y: Float32[Tensor, "4 32 32"]) -> Float32[Tensor, "4 32 32"]:
    return x @ y

@triton.jit
def dot_kernel(x_ptr, y_ptr, z_ptr, N0, N1, N2, MID, B0: tl.constexpr, B1: tl.constexpr, B2: tl.constexpr, B_MID: tl.constexpr):
    block_id_j = tl.program_id(0)
    block_id_k = tl.program_id(1)
    block_id_i = tl.program_id(2)

    off_i = block_id_i * B2 + tl.arange(0, B2)
    off_j = block_id_j * B0 + tl.arange(0, B0)
    off_k = block_id_k * B1 + tl.arange(0, B1)

    mask_i = off_i < N2
    mask_j = off_j < N0
    mask_k = off_k < N1

    z = tl.zeros((B2, B0, B1), dtype=tl.float32)
    off_z = (
        off_i[:, None, None] * N0 * N1
        + off_j[None, :, None] * N1
        + off_k[None, None, :]
    )
    mask_z = mask_i[:, None, None] & mask_j[None, :, None] & mask_k[None, None, :]

    for l in tl.range(0, MID, B_MID):
        off_l = l + tl.arange(0, B_MID)
        mask_l = off_l < MID
        off_x = (
            off_i[:, None, None] * N0 * MID
            + off_j[None, :, None] * MID
            + off_l[None, None, :]
        )
        off_y = (
            off_i[:, None, None] * MID * N1
            + off_l[None, :, None] * N1
            + off_k[None, None, :]
        )
        mask_x = mask_i[:, None, None] & mask_j[None, :, None] & mask_l[None, None, :]
        mask_y = mask_i[:, None, None] & mask_l[None, :, None] & mask_k[None, None, :]
        x = tl.load(x_ptr + off_x, mask=mask_x)
        y = tl.load(y_ptr + off_y, mask=mask_y)
        z += tl.dot(x, y)

    tl.store(z_ptr + off_z, z, mask=mask_z)