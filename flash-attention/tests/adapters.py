from __future__ import annotations

from typing import Type

import torch

# Import our FlashAttention implementations
from flash_attention2 import (
    FlashAttentionPyTorchFunction, 
    FlashAttentionTritonFunction,
    TRITON_AVAILABLE
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
