import torch
import triton
from torch import Tensor
import triton.language as tl
import jaxtyping
from jaxtyping import Float32, Int32

def add2_spec(x: Float32[Tensor, "200"]) -> Float32[Tensor, "200"]:
    return x + 10.

@triton.jit
def add_mask2_kernel(x_ptr, z_ptr, N0, B0: tl.constexpr):
    blockid = tl.program_id(0)
    range = blockid * B0 + tl.arange(0, B0)
    mask = range < N0
    x = tl.load(x_ptr + range, mask=mask)
    x_plus_10 = x + 10.
    tl.store(z_ptr + range, x_plus_10, mask=mask)