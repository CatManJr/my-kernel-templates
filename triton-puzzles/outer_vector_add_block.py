import torch
import triton
from torch import Tensor
import triton.language as tl
import jaxtyping
from jaxtyping import Float32, Int32

def add_vec_block_spec(x: Float32[Tensor, "100"], y: Float32[Tensor, "90"]) -> Float32[Tensor, "90 100"]:
    return x[None, :] + y[:, None]

@triton.jit
def add_vec_block_kernel(x_ptr, y_ptr, z_ptr, N0, N1, B0: tl.constexpr, B1: tl.constexpr):
    blockid_0 = tl.program_id(0)
    blockid_1 = tl.program_id(1)

    off_x = blockid_0 * B0 + tl.arange(0, B0)
    off_y = blockid_1 * B1 + tl.arange(0, B1)
    off_z = off_y[:, None] * N0 + off_x[None, :]

    mask_x = off_x < N0
    mask_y = off_y < N1
    mask_z = mask_y[:, None] & mask_x[None, :]

    x = tl.load(x_ptr + off_x, mask=mask_x)
    y = tl.load(y_ptr + off_y, mask=mask_y)

    z = add_vec_block_spec(x, y)
    tl.store(z_ptr + off_z, z, mask = mask_z)
    return