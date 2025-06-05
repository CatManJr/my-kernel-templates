import torch
import triton
from torch import Tensor
import triton.language as tl
import jaxtyping
from jaxtyping import Float32, Int32

def add_vec_spec(x: Float32[Tensor, "32"], y: Float32[Tensor, "32"]) -> Float32[Tensor, "32 32"]:
    return x[None, :] + y[:, None]

@triton.jit
def add_vec_kernel(x_ptr, y_ptr, z_ptr, N0, N1, B0: tl.constexpr, B1: tl.constexpr):
    blockid = tl.program_id(0)
    off_x = blockid * B0 + tl.arange(0, B0)
    off_y = blockid * B1 + tl.arange(0, B1)
    off_z = off_y[:, None] * B0 + off_x[None, :]
    x = tl.load(x_ptr + off_x)
    y = tl.load(y_ptr + off_y)
    x = tl.load(x_ptr + off_x)
    y = tl.load(y_ptr + off_y)
    z = add_vec_spec(x, y)
    tl.store(z_ptr + off_z, z)
    return