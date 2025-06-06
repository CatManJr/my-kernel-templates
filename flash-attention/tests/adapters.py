from __future__ import annotations

from typing import Type

import torch

# Import our FlashAttention implementations
from flash_attention2 import (
    FlashAttentionPyTorchFunction, 
    FlashAttentionTritonFunction,
    TRITON_AVAILABLE
)

# Import FlashAttention-1 implementation
from flash_attention import (
    FlashAttentionV1Function,
    TRITON_AVAILABLE as TRITON_V1_AVAILABLE
)


def get_flashattention_autograd_function_pytorch() -> Type:
    """
    Returns a torch.autograd.Function subclass that implements FlashAttention-2.
    The expectation is that this class will implement FlashAttention-2
    using standard PyTorch operations.

    Returns:
        A class object (not an instance of the class)
    """
    return FlashAttentionPyTorchFunction


def get_flashattention_autograd_function_triton() -> Type:
    """
    Returns a torch.autograd.Function subclass that implements FlashAttention-2
    using Triton kernels.
    The expectation is that this class will implement the same operations
    as the class you return in get_flashattention_autograd_function_pytorch(),
    but it should do so by invoking custom Triton kernels in the forward
    and backward passes.

    Returns:
        A class object (not an instance of the class)
    """
    return FlashAttentionTritonFunction


def get_flashattention_v1_autograd_function() -> Type:
    """
    Returns a torch.autograd.Function subclass that implements FlashAttention-1
    using Triton kernels.
    
    FlashAttention-1 uses 4D tensors (batch, heads, seq_len, head_dim) and
    has different parallelization strategy compared to FlashAttention-2.

    Returns:
        A class object (not an instance of the class)
    """
    return FlashAttentionV1Function
