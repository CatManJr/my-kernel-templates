"""
A scalar version of FlashAttention.

Uses zero programs. Block size `B0` represents `k` of length `N0`.
Block size `B0` represents `q` of length `N0`. Block size `B0` represents `v` of length `N0`.
Sequence length is `T`. Process it `B1 < T` elements at a time.  

$$z_{i} = \sum_{j} \text{softmax}(q_1 k_1, \ldots, q_T k_T)_j v_{j} \text{ for } i = 1\ldots N_0$$

This can be done in 1 loop using online softmax.
"""

import torch
import triton
from torch import Tensor
import triton.language as tl
import jaxtyping
from jaxtyping import Float32, Int32

def flashatt_spec(q: Float32[Tensor, "200"], k: Float32[Tensor, "200"], v: Float32[Tensor, "200"]) -> Float32[Tensor, "200"]:
    x = q[:, None] * k[None, :]
    x_max = x.max(1, keepdim=True)[0]
    x = x - x_max
    x_exp = x.exp()
    soft =  x_exp  / x_exp.sum(1, keepdim=True)
    return (v[None, :] * soft).sum(1)

@triton.jit
def flashatt_kernel(
    q_ptr, k_ptr, v_ptr, z_ptr, N0, T, B0: tl.constexpr, B1: tl.constexpr
):
    block_id_i = tl.program_id(0)
    log2_e = 1.44269504
    # myexp = lambda x: tl.exp2(log2_e * x)
    # Finish me!

    off_i = block_id_i * B0 + tl.arange(0, B0)
    mask_i = off_i < N0
    inf = 1.0e6

    # Need `other`!!!
    q = tl.load(q_ptr + off_i, mask=mask_i)

    # The variable names of Triton's offcial FlashAttention tutorial
    # is attached here for reference.
    # Our variable names are consistent with Puzzle 8.

    # l_i
    exp_sum = tl.zeros((B0,), dtype=tl.float32)
    # m_i
    qk_max = tl.full((B0,), -inf, dtype=tl.float32)
    z = tl.zeros((B0,), dtype=tl.float32)

    for id_j in tl.range(0, T, B1):
        off_j = id_j + tl.arange(0, B1)
        mask_j = off_j < T
        mask_ij = mask_i[:, None] & mask_j[None, :]

        k = tl.load(k_ptr + off_j, mask=mask_j)
        qk = q[:, None] * k[None, :] + tl.where(mask_ij, 0, -1.0e6)
        # print(qk.shape)

        # m_ij
        new_max = tl.maximum(tl.max(qk, axis=1), qk_max)
        qk_exp = tl.exp2(log2_e * (qk - new_max[:, None]))
        # alpha
        factor = tl.exp2(log2_e * (qk_max - new_max))
        # l_ij
        new_exp_sum = exp_sum * factor + tl.sum(qk_exp, axis=1)
        v = tl.load(v_ptr + off_j, mask=mask_j, other=0.0)
        z = z * factor + tl.sum(qk_exp * v[None, :], axis=1)

        qk_max = new_max
        exp_sum = new_exp_sum

    z = z / exp_sum
    tl.store(z_ptr + off_i, z, mask=mask_i)
    return