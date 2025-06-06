"""
A batched 2D convolution.

Uses one program id axis. Block size `B0` represent the batches to process out of `N0`.
Image `x` is size is `H` by `W` with only 1 channel, and kernel `k` is size `KH` by `KW`.

$$z_{i, j, k} = \sum_{oj, ok} k_{oj,ok} \times x_{i,j + oj, k + ok} \text{ for } i = 1\ldots N_0$$
"""

import torch
import triton
from torch import Tensor
import triton.language as tl
import jaxtyping
from jaxtyping import Float32, Int32

def conv2d_spec(x: Float32[Tensor, "4 8 8"], k: Float32[Tensor, "4 4"]) -> Float32[Tensor, "4 8 8"]:
    z = torch.zeros(4, 8, 8)
    x = torch.nn.functional.pad(x, (0, 4, 0, 4, 0, 0), value=0.0)
    print(x.shape, k.shape)
    for i in range(8):
        for j in range(8):
            z[:, i, j] = (k[None, :, :] * x[:, i: i+4, j: j + 4]).sum(1).sum(1)
    return z


@triton.jit
def conv2d_kernel(x_ptr, k_ptr, z_ptr, N0, H, W, KH: tl.constexpr, KW: tl.constexpr, B0: tl.constexpr):
    block_id_i = tl.program_id(0)
    # Finish me!
    off_i = block_id_i * B0 + tl.arange(0, B0)
    mask_i = off_i < N0

    off_h = tl.arange(0, KH)
    off_w = tl.arange(0, KW)
    off_hw = off_h[:, None] * KW + off_w[None, :]

    k = tl.load(k_ptr + off_hw)

    for j in tl.range(0, H):
        for l in tl.range(0, W):
            off_j_oj = j + off_h[None, :, None]
            off_l_ol = l + off_w[None, None, :]
            off_x = off_i * H * W + off_j_oj * W + off_l_ol
            mask_x = (off_j_oj < H) & (off_l_ol < W)
            x = tl.load(x_ptr + off_x, mask=mask_x)

            z = tl.sum(x * k[None, :])
            off_z = off_i * H * W + j * W + l
            tl.store(z_ptr + off_z, z)

    return