"""
When doing matrix multiplication with quantized neural networks a common strategy is to store the weight matrix in lower precision, with a shift and scale term.

For this problem our `weight` will be stored in 4 bits. We can store `FPINT` of these in a 32 bit integer. In addition for every `group` weights in order we will store 1 `scale` float value and 1 `shift` 4 bit value. We store these for the column of weight. The `activation`s are stored separately in standard floats.

Mathematically it looks like.

$$z_{j, k} = \sum_{l} sc_{j, \frac{l}{g}} (w_{j, l} - sh_{j, \frac{l}{g}}) \times y_{l, k} \text{ for } i = 1\ldots N_2, j = 1\ldots N_0, k = 1\ldots N_1$$

However, it is a bit more complex since we need to also extract the 4-bit values into floats to begin.
"""

import torch
import triton
from torch import Tensor
import triton.language as tl
import jaxtyping
from jaxtyping import Float32, Int32

FPINT = 32 // 4
GROUP = 8

def quant_dot_spec(scale : Float32[Tensor, "32 8"],
                   offset : Int32[Tensor, "32"],
                   weight: Int32[Tensor, "32 8"],
                   activation: Float32[Tensor, "64 32"]) -> Float32[Tensor, "32 32"]:
    offset = offset.view(32, 1)
    def extract(x):
        over = torch.arange(8) * 4
        mask = 2**4 - 1
        return (x[..., None] >> over) & mask
    scale = scale[..., None].expand(-1, 8, GROUP).contiguous().view(-1, 64)
    offset = extract(offset)[..., None].expand(-1, 1, 8, GROUP).contiguous().view(-1, 64)
    return ( scale * (extract(weight).view(-1, 64) - offset))  @ activation

@triton.jit
def quant_dot_kernel(scale_ptr, offset_ptr, weight_ptr, activation_ptr,
                     z_ptr, N0, N1, MID, B0: tl.constexpr, B1: tl.constexpr, B_MID: tl.constexpr):
    block_id_j = tl.program_id(0)
    block_id_k = tl.program_id(1)
    # Finish me!
    off_j = block_id_j * B0 + tl.arange(0, B0)
    off_k = block_id_k * B1 + tl.arange(0, B1)

    mask_j = off_j < N0
    mask_k = off_k < N1

    z = tl.zeros((B0, B1), dtype=tl.float32)
    off_z = off_j[:, None] * N1 + off_k[None, :]
    mask_z = mask_j[:, None] & mask_k[None, :]

    for l in tl.range(0, MID, B_MID):
        # load scale
        off_l_div_g = tl.arange(0, B_MID // GROUP) + (l // GROUP)
        mask_l_div_g = off_l_div_g < (MID // GROUP)
        off_scale = off_j[:, None] * (MID // GROUP) + off_l_div_g[None, :]
        # print(off_scale.shape)
        mask_scale = mask_j[:, None] & mask_l_div_g[None, :]
        scale = tl.load(scale_ptr + off_scale, mask=mask_scale)

        # load shift (offset)
        # (32,), each 32bits integer store FPINT(8)*4 shifts
        shift = tl.load(offset_ptr + off_j, mask=mask_j)

        # load weight
        # note: our weight will be stored in 4bits.
        off_weight_l = l + tl.arange(0, B_MID // FPINT)
        mask_weight_l = off_weight_l < (MID // FPINT)
        off_weight = off_j[:, None] * (MID // FPINT) + off_weight_l[None, :]
        mask_weight = mask_j[:, None] & mask_weight_l[None, :]
        weight = tl.load(weight_ptr + off_weight, mask=mask_weight)

        # load activation as normal float
        off_l = l + tl.arange(0, B_MID)
        mask_l = off_l < MID
        off_activation = off_l[:, None] * N1 + off_k[None, :]
        mask_activation = mask_l[:, None] & mask_k[None, :]
        activation = tl.load(activation_ptr + off_activation, mask=mask_activation)

        # unpack weight and shift
        BITS = 32 // FPINT
        unpack_offs = tl.arange(0, FPINT) * BITS
        unpack_upperbound_mask = (1 << BITS) - 1
        unpacked_shift = (shift[:, None] >> unpack_offs) & unpack_upperbound_mask
        unpacked_weight = (weight[:, :, None] >> unpack_offs) & unpack_upperbound_mask
        # quant transform
        # [BLOCK_J, 8, 1] * ([BLOCK_J, 8, 8] - [BLOCK_J, 8, 1])
        transformed_weight = scale[:, :, None] * (
            unpacked_weight - unpacked_shift[:, :, None]
        )
        # shape: [*, 64]
        transformed_weight = transformed_weight.reshape(
            unpacked_shift.shape[0], unpacked_shift.shape[-1] * FPINT
        )

        # compute
        z += tl.dot(transformed_weight, activation)

    tl.store(z_ptr + off_z, z, mask=mask_z)

    return