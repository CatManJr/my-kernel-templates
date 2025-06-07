#!/usr/bin/env python3
"""
Official FlashAttention-2 Leaderboard Benchmark Script

This script implements the exact test case from the CS336 Assignment 2 leaderboard
specification for testing FlashAttention-2 performance on H100 GPUs.

Test Configuration:
- Batch size: 1
- Sequence length: 16,384
- Number of heads: 16
- Head dimension: 64 (dmodel=1024, dmodel/n_heads=64)
- Data type: BF16
- Causal masking: True
- Benchmark: Forward + Backward pass timing
"""

import torch
import triton
import triton.testing
import time
from typing import Optional, Dict, Any
import argparse

# Import your FlashAttention-2 implementation
try:
    # Import from the local flash_attention2.py file
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    from flash_attention2 import FlashAttentionTritonFunction, TRITON_AVAILABLE
    print("✅ Successfully imported FlashAttention-2 implementations")
    print(f"   - Triton available: {TRITON_AVAILABLE}")
except ImportError as e:
    print(f"❌ Failed to import FlashAttention-2: {e}")
    TRITON_AVAILABLE = False


class FlashAttention2:
    """
    Official leaderboard wrapper for FlashAttention-2.
    
    This class should wrap your best performing FlashAttention-2 implementation.
    You can choose between different implementations by modifying the apply method.
    """
    
    @staticmethod
    def apply(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, is_causal: bool = True) -> torch.Tensor:
        """
        Apply FlashAttention-2 with the signature expected by the leaderboard.
        
        Args:
            q: Query tensor (n_heads, sequence_length, d_head)
            k: Key tensor (n_heads, sequence_length, d_head)  
            v: Value tensor (n_heads, sequence_length, d_head)
            is_causal: Whether to apply causal masking
            
        Returns:
            Output tensor (n_heads, sequence_length, d_head)
        """
        if not TRITON_AVAILABLE:
            raise RuntimeError("Triton FlashAttention-2 is not available")
        
        # Choose your best implementation here:
        # Option 1: Use the standard Triton implementation
        return FlashAttentionTritonFunction.apply(q, k, v, is_causal)
        
        # Option 2: Use the enhanced Triton implementation with full backward kernel
        # return FlashAttentionTritonAll.apply(q, k, v, is_causal)


def create_test_tensors(n_heads: int = 16, 
                       sequence_length: int = 16384, 
                       d_head: int = 64,
                       device: str = 'cuda',
                       dtype: torch.dtype = torch.bfloat16) -> tuple:
    """
    Create test tensors matching the official leaderboard specification.
    
    Args:
        n_heads: Number of attention heads (16)
        sequence_length: Sequence length (16,384)
        d_head: Head dimension (64, derived from dmodel=1024/16)
        device: Device to create tensors on
        dtype: Data type (bfloat16)
        
    Returns:
        Tuple of (q, k, v) tensors with requires_grad=True
    """
    shape = (n_heads, sequence_length, d_head)
    
    # Create tensors with the exact specification
    q = torch.randn(shape, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn(shape, device=device, dtype=dtype, requires_grad=True)  
    v = torch.randn(shape, device=device, dtype=dtype, requires_grad=True)
    
    return q, k, v


def test_timing_flash_forward_backward():
    """
    Official leaderboard timing test function.
    
    This is the exact test that will be run on the leaderboard to measure
    your FlashAttention-2 implementation performance.
    """
    print("🚀 Running Official FlashAttention-2 Leaderboard Benchmark")
    print("=" * 70)
    
    # Leaderboard test configuration
    n_heads = 16
    d_head = 64
    sequence_length = 16384
    
    print(f"Configuration:")
    print(f"  - n_heads: {n_heads}")
    print(f"  - d_head: {d_head}")
    print(f"  - sequence_length: {sequence_length}")
    print(f"  - dmodel: {n_heads * d_head}")
    print(f"  - dtype: torch.bfloat16")
    print(f"  - is_causal: True")
    print(f"  - device: cuda")
    
    # Create test tensors
    q, k, v = create_test_tensors(n_heads, sequence_length, d_head)
    
    print(f"\nTensor shapes:")
    print(f"  - Q: {q.shape}")
    print(f"  - K: {k.shape}")
    print(f"  - V: {v.shape}")
    
    # Compile the FlashAttention function (as specified in the test)
    flash = torch.compile(FlashAttention2.apply)
    
    def flash_forward_backward():
        """Forward and backward pass to be benchmarked."""
        # Clear gradients
        if q.grad is not None:
            q.grad.zero_()
        if k.grad is not None:
            k.grad.zero_()
        if v.grad is not None:
            v.grad.zero_()
        
        # Forward pass
        o = flash(q, k, v, True)  # is_causal=True
        
        # Backward pass
        loss = o.sum()
        loss.backward()
        
        # Ensure computation is complete
        torch.cuda.synchronize()
    
    print(f"\n⏱️  Running benchmark...")
    print(f"  - Warmup iterations: 1000")
    print(f"  - Benchmark iterations: 10000")
    
    # Run the official benchmark (matching the leaderboard specification exactly)
    try:
        results = triton.testing.do_bench(
            flash_forward_backward, 
            rep=10000,      # 10,000 repetitions as specified
            warmup=1000     # 1,000 warmup iterations as specified
        )
        
        print(f"\n✅ Benchmark completed successfully!")
        print(f"📊 Results:")
        print(f"  - Average time per iteration: {results:.4f} ms")
        print(f"  - Throughput: {1000/results:.2f} iterations/second")
        
        # Calculate additional metrics
        total_operations = n_heads * sequence_length * d_head
        throughput_ops = total_operations * (1000 / results)  # ops per second
        
        print(f"  - Total operations per iteration: {total_operations:,}")
        print(f"  - Throughput: {throughput_ops/1e9:.2f} GOPS")
        
        return results
        
    except Exception as e:
        print(f"❌ Benchmark failed: {e}")
        print(f"   This could be due to:")
        print(f"   - CUDA OOM (try reducing sequence length)")
        print(f"   - Implementation errors")
        print(f"   - Triton compilation issues")
        return None


def main():
    """Main function to run the leaderboard benchmark."""
    parser = argparse.ArgumentParser(description='FlashAttention-2 Leaderboard Benchmark')
    parser.add_argument('--skip-verification', action='store_true',
                       help='Skip correctness verification (deprecated - always skipped)')
    
    args = parser.parse_args()
    
    print("FlashAttention-2 Official Leaderboard Benchmark")
    print("=" * 50)
    print("📝 Note: Correctness verification skipped - tests completed in pytest")
    
    # Check environment
    if not torch.cuda.is_available():
        print("❌ CUDA is not available. This benchmark requires a CUDA GPU.")
        return
    
    if not TRITON_AVAILABLE:
        print("❌ Triton FlashAttention-2 is not available.")
        return
    
    print(f"🖥️  Device: {torch.cuda.get_device_name()}")
    print(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Run the official benchmark directly
    print("\n" + "="*70)
    result = test_timing_flash_forward_backward()
    
    if result is not None:
        print(f"\n🎯 Final Result: {result:.4f} ms")
        print(f"   (This is the number that will be submitted to the leaderboard)")
    else:
        print(f"\n❌ Benchmark failed to complete")


if __name__ == "__main__":
    main()