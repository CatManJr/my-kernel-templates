from __future__ import annotations

from typing import Type

import torch

# Import our FlashAttention implementations
from flash_attention2 import (
    FlashAttentionPyTorchFunction, 
    FlashAttentionTritonFunction,
    FlashAttentionTritonAll,
    TRITON_AVAILABLE
)

# Import FlashAttention-1 implementation
from flash_attention import (
    FlashAttention1TritonFunction,
    TRITON_AVAILABLE as TRITON_V1_AVAILABLE
)

# Import FlashAttention-3 implementation
from flash_attention3 import (
    FlashAttention3TritonFunction,
    TRITON_AVAILABLE as TRITON_V3_AVAILABLE
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

def get_flashattention_autograd_function_triton_all() -> Type:
    """
    Returns a torch.autograd.Function subclass that implements FlashAttention-2
    using Triton kernels for both fwd & bwd.

    Returns:
        A class object (not an instance of the class)
    """
    return FlashAttentionTritonAll


def get_flashattention_v1_autograd_function() -> Type:
    """
    Returns a torch.autograd.Function subclass that implements FlashAttention-1
    using Triton kernel for forward pass and PyTorch for backward pass.
    
    This version uses Triton kernel for the forward pass (for performance) but
    falls back to PyTorch implementation for the backward pass (for simplicity).

    Returns:
        A class object (not an instance of the class)
    """
    return FlashAttention1TritonFunction  # Pure PyTorch fallback


def get_flashattention_v3_autograd_function() -> Type:
    """
    Returns a torch.autograd.Function subclass that implements FlashAttention-3
    using Triton kernels with advanced optimizations.
    
    FlashAttention-3 includes:
    1. Asynchronous data movement with computation overlap
    2. Warp specialization for better parallelism
    3. Enhanced numerical stability with mixed precision
    4. Optimized memory access patterns
    5. Optional low-rank approximation for very long sequences
    6. Adaptive tile sizing based on sequence length and hardware

    Returns:
        A class object (not an instance of the class)
    """
    return FlashAttention3TritonFunction
