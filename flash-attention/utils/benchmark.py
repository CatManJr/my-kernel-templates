"""
FlashAttention Benchmarking Utilities

This module provides benchmarking utilities for comparing different FlashAttention implementations
with standard PyTorch attention mechanisms. Designed to be reusable for FlashAttention-1, 2, and 3.
"""

import math
import torch
import triton
import triton.testing
from typing import Dict, List, Tuple, Callable, Any, Optional
import pandas as pd
from dataclasses import dataclass
import time


@dataclass
class BenchmarkConfig:
    """Configuration for benchmarking experiments."""
    sequence_lengths: List[int]
    head_dims: List[int]
    dtypes: List[torch.dtype]
    batch_size: int = 1
    num_heads: int = 1  # Added for FlashAttention-1 which expects 4D tensors
    is_causal: bool = True
    num_warmup: int = 10
    num_iterations: int = 100
    device: str = 'cuda'


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""
    seq_len: int
    head_dim: int
    dtype: str
    implementation: str
    forward_ms: float
    backward_ms: float
    end_to_end_ms: float
    memory_mb: Optional[float] = None


def standard_pytorch_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, 
                             is_causal: bool = False) -> torch.Tensor:
    """
    Standard PyTorch attention implementation for comparison.
    
    Args:
        Q: Query tensor (batch, seq_len, head_dim) or (batch, heads, seq_len, head_dim)
        K: Key tensor (batch, seq_len, head_dim) or (batch, heads, seq_len, head_dim)
        V: Value tensor (batch, seq_len, head_dim) or (batch, heads, seq_len, head_dim)
        is_causal: Whether to apply causal masking
        
    Returns:
        Output tensor with same shape as Q
    """
    # Handle both 3D and 4D tensors
    if Q.dim() == 4:
        batch_size, num_heads, seq_len, head_dim = Q.shape
        scale = 1.0 / math.sqrt(head_dim)
        
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) * scale  # (batch, heads, seq_len, seq_len)
        
        # Apply causal mask if needed
        if is_causal:
            mask = torch.triu(torch.ones(seq_len, seq_len, device=Q.device), diagonal=1).bool()
            scores.masked_fill_(mask, float('-inf'))
        
        # Apply softmax
        attn_weights = torch.softmax(scores, dim=-1)
        
        # Apply attention to values
        output = torch.matmul(attn_weights, V)
    else:
        # 3D case (existing implementation)
        batch_size, seq_len, head_dim = Q.shape
        scale = 1.0 / math.sqrt(head_dim)
        
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) * scale  # (batch, seq_len, seq_len)
        
        # Apply causal mask if needed
        if is_causal:
            mask = torch.triu(torch.ones(seq_len, seq_len, device=Q.device), diagonal=1).bool()
            scores.masked_fill_(mask, float('-inf'))
        
        # Apply softmax
        attn_weights = torch.softmax(scores, dim=-1)
        
        # Apply attention to values
        output = torch.matmul(attn_weights, V)
    
    return output


class AttentionBenchmarker:
    """
    Benchmarking class for comparing different attention implementations.
    """
    
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.results: List[BenchmarkResult] = []
        
    def create_inputs_3d(self, seq_len: int, head_dim: int, dtype: torch.dtype) -> Tuple[torch.Tensor, ...]:
        """Create random 3D inputs for FlashAttention-2 and PyTorch standard."""
        batch_size = self.config.batch_size
        device = self.config.device
        
        # Create random inputs with requires_grad for backward pass
        Q = torch.randn(batch_size, seq_len, head_dim, dtype=dtype, device=device, requires_grad=True)
        K = torch.randn(batch_size, seq_len, head_dim, dtype=dtype, device=device, requires_grad=True)
        V = torch.randn(batch_size, seq_len, head_dim, dtype=dtype, device=device, requires_grad=True)
        
        # Create gradient tensor for backward pass
        grad_output = torch.randn(batch_size, seq_len, head_dim, dtype=dtype, device=device)
        
        return Q, K, V, grad_output
    
    def create_inputs_4d(self, seq_len: int, head_dim: int, dtype: torch.dtype) -> Tuple[torch.Tensor, ...]:
        """Create random 4D inputs for FlashAttention-1."""
        batch_size = self.config.batch_size
        num_heads = self.config.num_heads
        device = self.config.device
        
        # Create random inputs with requires_grad for backward pass
        Q = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=dtype, device=device, requires_grad=True)
        K = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=dtype, device=device, requires_grad=True)
        V = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=dtype, device=device, requires_grad=True)
        
        # Create gradient tensor for backward pass
        grad_output = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=dtype, device=device)
        
        return Q, K, V, grad_output
    
    def benchmark_forward_accurate(self, attention_fn: Callable, Q: torch.Tensor, K: torch.Tensor, 
                                  V: torch.Tensor) -> float:
        """More accurate benchmark for forward pass using CUDA events."""
        # Ensure CUDA synchronization
        torch.cuda.synchronize()
        
        # Warmup
        for _ in range(self.config.num_warmup):
            _ = attention_fn(Q, K, V, self.config.is_causal)
        
        torch.cuda.synchronize()
        
        # Use CUDA events for precise timing
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        times = []
        for _ in range(self.config.num_iterations):
            start_event.record()
            output = attention_fn(Q, K, V, self.config.is_causal)
            end_event.record()
            torch.cuda.synchronize()
            times.append(start_event.elapsed_time(end_event))
        
        return sum(times) / len(times)
    
    def benchmark_backward_accurate(self, attention_fn: Callable, Q: torch.Tensor, K: torch.Tensor, 
                                   V: torch.Tensor, grad_output: torch.Tensor) -> float:
        """More accurate benchmark for backward pass using CUDA events."""
        # Ensure CUDA synchronization
        torch.cuda.synchronize()
        
        # Warmup - use fresh tensors each time to avoid graph accumulation
        for _ in range(self.config.num_warmup):
            # Create fresh input tensors for warmup
            Q_warm = Q.detach().clone().requires_grad_(True)
            K_warm = K.detach().clone().requires_grad_(True)
            V_warm = V.detach().clone().requires_grad_(True)
            
            output = attention_fn(Q_warm, K_warm, V_warm, self.config.is_causal)
            output.backward(grad_output)
            
            # Clean up
            del Q_warm, K_warm, V_warm, output
        
        torch.cuda.synchronize()
        
        # Use CUDA events for precise timing
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        times = []
        for _ in range(self.config.num_iterations):
            # Create fresh input tensors for each iteration
            Q_test = Q.detach().clone().requires_grad_(True)
            K_test = K.detach().clone().requires_grad_(True)
            V_test = V.detach().clone().requires_grad_(True)
            
            # Forward pass (not timed)
            output = attention_fn(Q_test, K_test, V_test, self.config.is_causal)
            
            # Time only the backward pass
            start_event.record()
            output.backward(grad_output)
            end_event.record()
            torch.cuda.synchronize()
            
            times.append(start_event.elapsed_time(end_event))
            
            # Clean up
            del Q_test, K_test, V_test, output
        
        return sum(times) / len(times)
    
    def benchmark_end_to_end_accurate(self, attention_fn: Callable, Q: torch.Tensor, K: torch.Tensor, 
                                     V: torch.Tensor, grad_output: torch.Tensor) -> float:
        """More accurate benchmark for end-to-end pass using CUDA events."""
        # Ensure CUDA synchronization
        torch.cuda.synchronize()
        
        # Warmup
        for _ in range(self.config.num_warmup):
            Q_warm = Q.detach().clone().requires_grad_(True)
            K_warm = K.detach().clone().requires_grad_(True)
            V_warm = V.detach().clone().requires_grad_(True)
            
            output = attention_fn(Q_warm, K_warm, V_warm, self.config.is_causal)
            output.backward(grad_output)
            
            del Q_warm, K_warm, V_warm, output
        
        torch.cuda.synchronize()
        
        # Use CUDA events for precise timing
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        times = []
        for _ in range(self.config.num_iterations):
            # Create fresh input tensors
            Q_test = Q.detach().clone().requires_grad_(True)
            K_test = K.detach().clone().requires_grad_(True)
            V_test = V.detach().clone().requires_grad_(True)
            
            # Time both forward and backward
            start_event.record()
            output = attention_fn(Q_test, K_test, V_test, self.config.is_causal)
            output.backward(grad_output)
            end_event.record()
            torch.cuda.synchronize()
            
            times.append(start_event.elapsed_time(end_event))
            
            # Clean up
            del Q_test, K_test, V_test, output
        
        return sum(times) / len(times)

    def benchmark_forward(self, attention_fn: Callable, Q: torch.Tensor, K: torch.Tensor, 
                         V: torch.Tensor) -> float:
        """Benchmark forward pass - use accurate timing for PyTorch functions."""
        # Use accurate timing for PyTorch/compiled functions, triton timing for others
        if hasattr(attention_fn, '__name__') and 'triton' in attention_fn.__name__.lower():
            # Use triton timing for triton functions
            def forward_fn():
                return attention_fn(Q, K, V, self.config.is_causal)
            
            return triton.testing.do_bench(
                forward_fn,
                warmup=self.config.num_warmup,
                rep=self.config.num_iterations
            )
        else:
            # Use accurate CUDA event timing for PyTorch functions
            return self.benchmark_forward_accurate(attention_fn, Q, K, V)
    
    def benchmark_backward(self, attention_fn: Callable, Q: torch.Tensor, K: torch.Tensor, 
                          V: torch.Tensor, grad_output: torch.Tensor) -> float:
        """Benchmark backward pass - always use accurate timing."""
        return self.benchmark_backward_accurate(attention_fn, Q, K, V, grad_output)
    
    def benchmark_end_to_end(self, attention_fn: Callable, Q: torch.Tensor, K: torch.Tensor, 
                           V: torch.Tensor, grad_output: torch.Tensor) -> float:
        """Benchmark complete forward + backward pass - always use accurate timing."""
        return self.benchmark_end_to_end_accurate(attention_fn, Q, K, V, grad_output)

    def benchmark_single_config(self, seq_len: int, head_dim: int, dtype: torch.dtype,
                              implementations: Dict[str, Callable]) -> List[BenchmarkResult]:
        """Benchmark a single configuration across all implementations."""
        print(f"Benchmarking seq_len={seq_len}, head_dim={head_dim}, dtype={dtype}")
        
        results = []
        
        for impl_name, attention_fn in implementations.items():
            try:
                # Create appropriate inputs based on implementation
                if 'FlashAttention1' in impl_name or 'V1' in impl_name:
                    Q, K, V, grad_output = self.create_inputs_4d(seq_len, head_dim, dtype)
                else:
                    Q, K, V, grad_output = self.create_inputs_3d(seq_len, head_dim, dtype)
                
                # Benchmark forward pass
                forward_ms = self.benchmark_forward(attention_fn, Q, K, V)
                
                # Benchmark backward pass  
                backward_ms = self.benchmark_backward(attention_fn, Q, K, V, grad_output)
                
                # Benchmark end-to-end
                end_to_end_ms = self.benchmark_end_to_end(attention_fn, Q, K, V, grad_output)
                
                # Record results
                result = BenchmarkResult(
                    seq_len=seq_len,
                    head_dim=head_dim,
                    dtype=str(dtype).split('.')[-1],  # e.g., 'float32'
                    implementation=impl_name,
                    forward_ms=forward_ms,
                    backward_ms=backward_ms,
                    end_to_end_ms=end_to_end_ms
                )
                results.append(result)
                
                print(f"  {impl_name}: Forward={forward_ms:.3f}ms, Backward={backward_ms:.3f}ms, E2E={end_to_end_ms:.3f}ms")
                
            except Exception as e:
                print(f"  {impl_name}: FAILED - {str(e)}")
                # Still record a failed result
                result = BenchmarkResult(
                    seq_len=seq_len,
                    head_dim=head_dim,
                    dtype=str(dtype).split('.')[-1],
                    implementation=impl_name,
                    forward_ms=float('inf'),
                    backward_ms=float('inf'),
                    end_to_end_ms=float('inf')
                )
                results.append(result)
        
        return results
    
    def run_full_benchmark(self, implementations: Dict[str, Callable]) -> pd.DataFrame:
        """Run complete benchmark across all configurations."""
        print("Starting FlashAttention Comprehensive Benchmark")
        print("=" * 60)
        
        all_results = []
        
        for seq_len in self.config.sequence_lengths:
            for head_dim in self.config.head_dims:
                for dtype in self.config.dtypes:
                    # Clear CUDA cache before each configuration
                    torch.cuda.empty_cache()
                    
                    config_results = self.benchmark_single_config(
                        seq_len, head_dim, dtype, implementations
                    )
                    all_results.extend(config_results)
        
        # Convert to DataFrame for easy analysis
        df = pd.DataFrame([
            {
                'seq_len': r.seq_len,
                'head_dim': r.head_dim,
                'dtype': r.dtype,
                'implementation': r.implementation,
                'forward_ms': r.forward_ms,
                'backward_ms': r.backward_ms,
                'end_to_end_ms': r.end_to_end_ms
            }
            for r in all_results
        ])
        
        self.results = all_results
        return df
    
    def save_results(self, df: pd.DataFrame, filename: str):
        """Save benchmark results to CSV."""
        df.to_csv(filename, index=False)
        print(f"Results saved to {filename}")
    
    def print_summary_table(self, df: pd.DataFrame):
        """Print a formatted summary table."""
        print("\n" + "=" * 140)
        print("BENCHMARK RESULTS SUMMARY")
        print("=" * 140)
        
        # Group by configuration and create pivot table
        for dtype in df['dtype'].unique():
            print(f"\nDatatype: {dtype}")
            print("-" * 120)
            
            dtype_df = df[df['dtype'] == dtype]
            
            # Create pivot table for each metric
            for metric in ['forward_ms', 'backward_ms', 'end_to_end_ms']:
                print(f"\n{metric.replace('_', ' ').title()}:")
                
                pivot = dtype_df.pivot_table(
                    values=metric,
                    index=['seq_len', 'head_dim'],
                    columns='implementation',
                    fill_value=float('inf')
                )
                
                # Format the table nicely
                formatted_pivot = pivot.copy()
                for col in formatted_pivot.columns:
                    formatted_pivot[col] = formatted_pivot[col].apply(
                        lambda x: f"{x:.3f}" if x != float('inf') else "FAIL"
                    )
                
                print(formatted_pivot.to_string())
                print()


def create_default_config() -> BenchmarkConfig:
    """Create default benchmark configuration."""
    return BenchmarkConfig(
        sequence_lengths=[2**i for i in range(7, 15)],  # 128 to 16384 (reduced for faster benchmarking)
        head_dims=[2**i for i in range(4, 8)],          # 16 to 128
        dtypes=[torch.bfloat16, torch.float32],
        batch_size=1,
        num_heads=1,
        is_causal=True,
        num_warmup=10,
        num_iterations=100,
        device='cuda'
    )


def create_quick_config() -> BenchmarkConfig:
    """Create quick benchmark configuration for testing."""
    return BenchmarkConfig(
        sequence_lengths=[128, 256, 512, 1024],
        head_dims=[32, 64],
        dtypes=[torch.float32],
        batch_size=1,
        num_heads=1,
        is_causal=True,
        num_warmup=5,
        num_iterations=20,
        device='cuda'
    )


def run_comprehensive_flashattention_benchmark():
    """
    Main function to run comprehensive FlashAttention benchmark across all versions.
    """
    # Import our implementations
    try:
        from flash_attention2 import flash_attention_triton as flash_attention2_triton, flash_attention_all_triton
        from flash_attention import flash_attention_v1
        # TODO: Uncomment when FlashAttention-3 is implemented
        # from flash_attention3 import flash_attention_v3
    except ImportError:
        import sys
        import os
        sys.path.append(os.path.dirname(os.path.dirname(__file__)))
        from flash_attention2 import flash_attention_triton as flash_attention2_triton, flash_attentions_all_triton
        from flash_attention import flash_attention_v1
        # TODO: Uncomment when FlashAttention-3 is implemented
        # from flash_attention3 import flash_attention_v3
    
    # Create benchmark configuration
    config = create_default_config()
    benchmarker = AttentionBenchmarker(config)
    
    # Define implementations to compare (removed slow PyTorch FlashAttention-2)
    implementations = {
        'PyTorch_Standard': standard_pytorch_attention,
        'FlashAttention1_Triton': flash_attention_v1,
        'FlashAttention2_Triton': flash_attention2_triton,
        'FlashAttention2_AllTriton': flash_attention_all_triton,
        # TODO: Uncomment when FlashAttention-3 is implemented
        # 'FlashAttention3_Triton': flash_attention_v3,
    }
    
    # Run benchmark
    print("Comprehensive FlashAttention Benchmark (All Versions)")
    print(f"Configurations: {len(config.sequence_lengths)} seq_lens × {len(config.head_dims)} head_dims × {len(config.dtypes)} dtypes")
    print(f"Total configurations: {len(config.sequence_lengths) * len(config.head_dims) * len(config.dtypes)}")
    print(f"Implementations: {list(implementations.keys())}")
    print()
    
    df = benchmarker.run_full_benchmark(implementations)
    
    # Print results
    benchmarker.print_summary_table(df)
    
    # Save results
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"flashattention_comprehensive_benchmark_{timestamp}.csv"
    benchmarker.save_results(df, filename)
    
    return df, benchmarker


def run_quick_flashattention_benchmark():
    """
    Quick benchmark for testing all FlashAttention versions.
    """
    # Import our implementations
    try:
        from flash_attention2 import flash_attention_triton as flash_attention2_triton, flash_attention_all_triton
        from flash_attention import flash_attention_v1
        # TODO: Uncomment when FlashAttention-3 is implemented
        # from flash_attention3 import flash_attention_v3
    except ImportError:
        import sys
        import os
        sys.path.append(os.path.dirname(os.path.dirname(__file__)))
        from flash_attention2 import flash_attention_triton as flash_attention2_triton, flash_attention_all_triton
        from flash_attention import flash_attention_v1
        # TODO: Uncomment when FlashAttention-3 is implemented
        # from flash_attention3 import flash_attention_v3
    
    # Create quick benchmark configuration
    config = create_quick_config()
    benchmarker = AttentionBenchmarker(config)
    
    # Define implementations to compare
    implementations = {
        'PyTorch_Standard': standard_pytorch_attention,
        'FlashAttention1_Triton': flash_attention_v1,
        'FlashAttention2_Triton': flash_attention2_triton,
        'FlashAttention2_AllTriton': flash_attention_all_triton,
        # TODO: Uncomment when FlashAttention-3 is implemented
        # 'FlashAttention3_Triton': flash_attention_v3,
    }
    
    # Run benchmark
    print("Quick FlashAttention Benchmark (All Versions)")
    print(f"Configurations: {len(config.sequence_lengths)} seq_lens × {len(config.head_dims)} head_dims × {len(config.dtypes)} dtypes")
    print(f"Total configurations: {len(config.sequence_lengths) * len(config.head_dims) * len(config.dtypes)}")
    print(f"Implementations: {list(implementations.keys())}")
    print()
    
    df = benchmarker.run_full_benchmark(implementations)
    
    # Print results
    benchmarker.print_summary_table(df)
    
    # Save results
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"flashattention_quick_benchmark_{timestamp}.csv"
    benchmarker.save_results(df, filename)
    
    return df, benchmarker


# Legacy function for backward compatibility
def run_flashattention2_benchmark():
    """Legacy function - redirects to comprehensive benchmark."""
    return run_comprehensive_flashattention_benchmark()


if __name__ == "__main__":
    # Check if CUDA is available
    if not torch.cuda.is_available():
        print("CUDA is not available. This benchmark requires a CUDA-enabled GPU.")
        exit(1)
    
    # Run the comprehensive benchmark
    df, benchmarker = run_comprehensive_flashattention_benchmark()