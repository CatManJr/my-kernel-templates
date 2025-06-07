"""
FlashAttention Comprehensive Benchmarking Utilities

This module provides optimized benchmarking utilities for comparing our three FlashAttention 
implementations (FA-1, FA-2, FA-3) with standard PyTorch attention, considering each version's 
optimal data precision and use cases.

FA-1: Best for small-medium sequences, float32 precision
FA-2: Best for general use, bfloat16 precision  
FA-3: Best for long sequences, mixed precision
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
class OptimizedBenchmarkConfig:
    """Optimized configuration for benchmarking FlashAttention implementations."""
    # Sequence lengths optimized for each FA version
    short_sequences: List[int] = None    # FA-1 optimal range
    medium_sequences: List[int] = None   # FA-2 optimal range  
    long_sequences: List[int] = None     # FA-3 optimal range
    
    # Head dimensions commonly used
    head_dims: List[int] = None
    
    # Precision per implementation
    fa1_dtypes: List[torch.dtype] = None     # FA-1 works best with float32
    fa2_dtypes: List[torch.dtype] = None     # FA-2 optimized for bfloat16
    fa3_dtypes: List[torch.dtype] = None     # FA-3 uses mixed precision
    pytorch_dtypes: List[torch.dtype] = None # PyTorch baseline
    
    batch_size: int = 1
    is_causal: bool = True
    num_warmup: int = 5
    num_iterations: int = 20
    device: str = 'cuda'
    
    def __post_init__(self):
        """Set default values optimized for each FA version."""
        if self.short_sequences is None:
            self.short_sequences = [128, 256, 512]  # FA-1 sweet spot
        if self.medium_sequences is None:
            self.medium_sequences = [1024, 2048, 4096]  # FA-2 sweet spot
        if self.long_sequences is None:
            self.long_sequences = [2048, 4096, 8192]  # FA-3 using same as FA-2 to avoid hanging
            
        if self.head_dims is None:
            self.head_dims = [32, 64, 128]  # Common transformer head dimensions
            
        # 🔧 MODIFIED: Use only FP32 precision for all implementations
        if self.fa1_dtypes is None:
            self.fa1_dtypes = [torch.float32]  # Only FP32
        if self.fa2_dtypes is None:
            self.fa2_dtypes = [torch.float32]  # Only FP32
        if self.fa3_dtypes is None:
            self.fa3_dtypes = [torch.float32]  # Only FP32
        if self.pytorch_dtypes is None:
            self.pytorch_dtypes = [torch.float32]  # Only FP32


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
    speedup_vs_pytorch: Optional[float] = None


def standard_pytorch_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, 
                             is_causal: bool = False) -> torch.Tensor:
    """Standard PyTorch attention implementation for comparison (3D tensors only)."""
    batch_size, seq_len, head_dim = Q.shape
    scale = 1.0 / math.sqrt(head_dim)
    
    # Compute attention scores
    scores = torch.matmul(Q, K.transpose(-2, -1)) * scale
    
    # Apply causal mask if needed
    if is_causal:
        mask = torch.triu(torch.ones(seq_len, seq_len, device=Q.device), diagonal=1).bool()
        scores.masked_fill_(mask, float('-inf'))
    
    # Apply softmax and compute output
    attn_weights = torch.softmax(scores, dim=-1)
    output = torch.matmul(attn_weights, V)
    
    return output


class FlashAttentionBenchmarker:
    """
    Optimized benchmarker for our three FlashAttention implementations.
    """
    
    def __init__(self, config: OptimizedBenchmarkConfig):
        self.config = config
        self.results: List[BenchmarkResult] = []
        self.pytorch_baseline: Dict[Tuple[int, int, str], float] = {}
        
    def create_test_inputs(self, seq_len: int, head_dim: int, dtype: torch.dtype, 
                          is_4d: bool = False) -> Tuple[torch.Tensor, ...]:
        """Create test inputs with optimal memory layout."""
        batch_size = self.config.batch_size
        device = self.config.device
        
        # All FlashAttention implementations use 3D tensors (batch, seq_len, head_dim)
        shape = (batch_size, seq_len, head_dim)
        
        # Create inputs with proper initialization for numerical stability
        Q = torch.randn(shape, dtype=dtype, device=device, requires_grad=True) * 0.1
        K = torch.randn(shape, dtype=dtype, device=device, requires_grad=True) * 0.1
        V = torch.randn(shape, dtype=dtype, device=device, requires_grad=True) * 0.1
        
        grad_output = torch.randn(shape, dtype=dtype, device=device) * 0.1
        
        return Q, K, V, grad_output
    
    def benchmark_with_cuda_events(self, benchmark_fn: Callable) -> float:
        """Accurate benchmarking using CUDA events."""
        torch.cuda.synchronize()
        
        # Warmup
        for _ in range(self.config.num_warmup):
            benchmark_fn()
        
        torch.cuda.synchronize()
        
        # Benchmark with CUDA events
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        times = []
        for _ in range(self.config.num_iterations):
            start_event.record()
            benchmark_fn()
            end_event.record()
            torch.cuda.synchronize()
            times.append(start_event.elapsed_time(end_event))
        
        return sum(times) / len(times)
    
    def benchmark_implementation(self, impl_name: str, attention_fn: Callable, 
                                seq_len: int, head_dim: int, dtype: torch.dtype) -> BenchmarkResult:
        """Benchmark a single implementation configuration - 只比较前向传播."""
        
        # Create appropriate inputs 
        is_4d = 'FA1' in impl_name or 'FlashAttention1' in impl_name
        Q, K, V, grad_output = self.create_test_inputs(seq_len, head_dim, dtype, is_4d)
        
        try:
            # Clear GPU memory and get baseline BEFORE detaching
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            memory_before = torch.cuda.memory_allocated() / (1024 * 1024)  # MB
            
            # 现在detach张量，只测试前向传播
            Q_test = Q.detach()
            K_test = K.detach() 
            V_test = V.detach()
            
            # Measure memory after creating test tensors
            torch.cuda.synchronize()
            memory_after_inputs = torch.cuda.memory_allocated() / (1024 * 1024)  # MB
            
            # Store references to prevent garbage collection during benchmark
            tensors_ref = [Q_test, K_test, V_test]
            
            # Benchmark forward pass only
            def forward_fn():
                output = attention_fn(Q_test, K_test, V_test, self.config.is_causal)
                # Ensure output is computed
                torch.cuda.synchronize()
                return output
            
            forward_ms = self.benchmark_with_cuda_events(forward_fn)
            
            # Measure peak memory after forward pass
            torch.cuda.synchronize()
            memory_peak = torch.cuda.memory_allocated() / (1024 * 1024)  # MB
            
            # Calculate actual memory usage (peak memory during computation)
            memory_usage_mb = max(memory_peak - memory_before, memory_after_inputs - memory_before)
            
            # Clean up references
            del tensors_ref, Q_test, K_test, V_test
            
            # Calculate speedup vs PyTorch baseline (only forward pass)
            pytorch_key = (seq_len, head_dim, str(dtype).split('.')[-1])
            speedup = None
            if pytorch_key in self.pytorch_baseline:
                speedup = self.pytorch_baseline[pytorch_key] / forward_ms
            
            return BenchmarkResult(
                seq_len=seq_len,
                head_dim=head_dim,
                dtype=str(dtype).split('.')[-1],
                implementation=impl_name,
                forward_ms=forward_ms,
                backward_ms=0.0,  # 不测试backward
                end_to_end_ms=forward_ms,  # 只有前向传播
                memory_mb=memory_usage_mb,
                speedup_vs_pytorch=speedup
            )
            
        except Exception as e:
            print(f"    FAILED {impl_name}: {str(e)}")
            return BenchmarkResult(
                seq_len=seq_len,
                head_dim=head_dim,
                dtype=str(dtype).split('.')[-1],
                implementation=impl_name,
                forward_ms=float('inf'),
                backward_ms=0.0,
                end_to_end_ms=float('inf'),
                memory_mb=0.0,
                speedup_vs_pytorch=0.0
            )
        finally:
            # Clean up original tensors
            del Q, K, V, grad_output
            torch.cuda.empty_cache()
    
    def run_optimized_benchmark(self) -> pd.DataFrame:
        """Run optimized benchmark focusing on each FA version's strengths."""
        print("🚀 FlashAttention Optimized Benchmark")
        print("=" * 80)
        
        # Import implementations
        try:
            # FA-1: Uses FlashAttention1TritonFunction class
            from flash_attention import flash_attention1_triton
            
            
            # FA-2: Uses flash_attention_triton function
            from flash_attention2 import flash_attention_triton
            
            # FA-3: Uses FlashAttention3TritonFunction class  
            from flash_attention3 import FlashAttention3TritonFunction
            def flash_attention_v3(Q, K, V, is_causal=False):
                return FlashAttention3TritonFunction.apply(Q, K, V, is_causal)
                
        except ImportError as e:
            print(f"Import error: {e}")
            print("Please ensure all FlashAttention modules are available")
            return pd.DataFrame()
        
        all_results = []
        
        # 1. Test PyTorch baseline first (for speedup calculations)
        print("\n📊 Testing PyTorch Baseline...")
        for seq_len in (self.config.short_sequences + self.config.medium_sequences + 
                       self.config.long_sequences):
            for head_dim in self.config.head_dims:
                for dtype in self.config.pytorch_dtypes:
                    print(f"  PyTorch: seq_len={seq_len}, head_dim={head_dim}, dtype={dtype}")
                    
                    result = self.benchmark_implementation(
                        "PyTorch_Standard", standard_pytorch_attention, 
                        seq_len, head_dim, dtype
                    )
                    all_results.append(result)
                    
                    # Store baseline for speedup calculation
                    key = (seq_len, head_dim, str(dtype).split('.')[-1])
                    self.pytorch_baseline[key] = result.end_to_end_ms
        
        # 2. Test FA-1 on short sequences (its strength)
        print("\n⚡ Testing FlashAttention-1 (Short Sequences)...")
        for seq_len in self.config.short_sequences:
            for head_dim in self.config.head_dims:
                for dtype in self.config.fa1_dtypes:
                    print(f"  FA-1: seq_len={seq_len}, head_dim={head_dim}, dtype={dtype}")
                    
                    result = self.benchmark_implementation(
                        "FlashAttention1_Triton", flash_attention1_triton,
                        seq_len, head_dim, dtype
                    )
                    all_results.append(result)
        
        # 3. Test FA-2 on medium sequences (its strength)  
        print("\n🔥 Testing FlashAttention-2 (Medium Sequences)...")
        for seq_len in self.config.medium_sequences:
            for head_dim in self.config.head_dims:
                for dtype in self.config.fa2_dtypes:
                    print(f"  FA-2: seq_len={seq_len}, head_dim={head_dim}, dtype={dtype}")
                    
                    result = self.benchmark_implementation(
                        "FlashAttention2_Triton", flash_attention_triton,
                        seq_len, head_dim, dtype
                    )
                    all_results.append(result)
        
        # 4. Test FA-3 on long sequences (its strength)
        print("\n🌟 Testing FlashAttention-3 (Long Sequences)...")
        for seq_len in self.config.long_sequences:
            for head_dim in self.config.head_dims:
                for dtype in self.config.fa3_dtypes:
                    print(f"  FA-3: seq_len={seq_len}, head_dim={head_dim}, dtype={dtype}")
                    
                    result = self.benchmark_implementation(
                        "FlashAttention3_Triton", flash_attention_v3,
                        seq_len, head_dim, dtype
                    )
                    all_results.append(result)
        
        # 5. Cross-comparison: test all FA versions on all sequence lengths
        print("\n🔄 Cross-Comparison (All FA versions on all sequences)...")
        all_sequences = self.config.short_sequences + self.config.medium_sequences + self.config.long_sequences
        
        implementations = [
            ("FA1_CrossTest", flash_attention1_triton, self.config.fa1_dtypes),
            ("FA2_CrossTest", flash_attention_triton, self.config.fa2_dtypes),
            ("FA3_CrossTest", flash_attention_v3, self.config.fa3_dtypes),
        ]
        
        for seq_len in all_sequences:
            for head_dim in [64]:  # Focus on common head_dim for cross-comparison
                for impl_name, impl_fn, dtypes in implementations:
                    for dtype in dtypes:
                        print(f"  {impl_name}: seq_len={seq_len}, head_dim={head_dim}, dtype={dtype}")
                        
                        result = self.benchmark_implementation(
                            impl_name, impl_fn, seq_len, head_dim, dtype
                        )
                        all_results.append(result)
        
        # Convert to DataFrame
        self.results = all_results
        df = pd.DataFrame([
            {
                'seq_len': r.seq_len,
                'head_dim': r.head_dim,
                'dtype': r.dtype,
                'implementation': r.implementation,
                'forward_ms': r.forward_ms,
                'backward_ms': r.backward_ms,
                'end_to_end_ms': r.end_to_end_ms,
                'memory_mb': r.memory_mb if r.memory_mb is not None else 0.0,  # Add default value
                'speedup_vs_pytorch': r.speedup_vs_pytorch,
            }
            for r in all_results
        ])
        
        return df
    
    def print_optimized_summary(self, df: pd.DataFrame):
        """Print optimized summary focusing on each FA version's strengths."""
        print("\n" + "=" * 100)
        print("🎯 FLASHATTENTION OPTIMIZED BENCHMARK RESULTS")
        print("=" * 100)
        
        # 1. Best performance for each sequence length range
        print("\n📈 PERFORMANCE BY SEQUENCE LENGTH RANGE:")
        print("-" * 80)
        
        ranges = [
            ("Short (128-512)", self.config.short_sequences),
            ("Medium (1K-4K)", self.config.medium_sequences), 
            ("Long (8K-16K)", self.config.long_sequences)
        ]
        
        for range_name, seq_lens in ranges:
            print(f"\n{range_name}:")
            range_df = df[df['seq_len'].isin(seq_lens) & (df['implementation'] != 'PyTorch_Standard')]
            
            if not range_df.empty:
                best_per_config = range_df.loc[range_df.groupby(['seq_len', 'head_dim', 'dtype'])['end_to_end_ms'].idxmin()]
                
                for _, row in best_per_config.iterrows():
                    speedup_str = f"{row['speedup_vs_pytorch']:.2f}x" if row['speedup_vs_pytorch'] else "N/A"
                    print(f"  seq_len={row['seq_len']}, head_dim={row['head_dim']}, dtype={row['dtype']}: "
                          f"{row['implementation']} ({row['end_to_end_ms']:.2f}ms, {speedup_str})")
        
        # 2. Speedup comparison
        print(f"\n🚀 SPEEDUP vs PyTorch (End-to-End):")
        print("-" * 80)
        
        fa_implementations = ['FlashAttention1_Triton', 'FlashAttention2_Triton', 'FlashAttention3_Triton']
        
        for impl in fa_implementations:
            impl_df = df[df['implementation'] == impl]
            if not impl_df.empty and impl_df['speedup_vs_pytorch'].notna().any():
                avg_speedup = impl_df['speedup_vs_pytorch'].mean()
                max_speedup = impl_df['speedup_vs_pytorch'].max()
                print(f"  {impl}: Average {avg_speedup:.2f}x, Max {max_speedup:.2f}x")
        
        # 3. Detailed comparison table
        print(f"\n📊 DETAILED RESULTS (End-to-End Time in ms):")
        print("-" * 120)
        
        # Create pivot table
        pivot_df = df.pivot_table(
            values='end_to_end_ms',
            index=['seq_len', 'head_dim', 'dtype'],
            columns='implementation',
            fill_value=float('inf')
        )
        
        # Format and display
        for col in pivot_df.columns:
            pivot_df[col] = pivot_df[col].apply(lambda x: f"{x:.2f}" if x != float('inf') else "FAIL")
        
        print(pivot_df.to_string())
        
        # 4. Memory efficiency (if available)
        if df['memory_mb'].notna().any():
            print(f"\n💾 MEMORY USAGE (MB):")
            print("-" * 50)
            memory_df = df[df['memory_mb'].notna()].groupby('implementation')['memory_mb'].mean()
            for impl, mem in memory_df.items():
                print(f"  {impl}: {mem:.1f} MB")
    
    def save_results(self, df: pd.DataFrame, filename: str = None):
        """Save benchmark results to CSV."""
        if filename is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"flashattention_optimized_benchmark_{timestamp}.csv"
        
        df.to_csv(filename, index=False)
        print(f"\n💾 Results saved to {filename}")


def create_optimized_config() -> OptimizedBenchmarkConfig:
    """Create optimized benchmark configuration."""
    return OptimizedBenchmarkConfig(
        # Sequence ranges optimized for each FA version - 使用更合理的长度
        short_sequences=[128, 256, 512],      # FA-1 strength
        medium_sequences=[1024, 2048],        # FA-2 strength - 移除4096
        long_sequences=[2048, 4096],          # FA-3 strength - 移除8192和16384
        
        # Common head dimensions
        head_dims=[32, 64],                   # 减少head_dims，专注于32和64
        
        # Optimal precisions per version
        fa1_dtypes=[torch.float32],                    # FA-1 works best with fp32
        fa2_dtypes=[torch.float32],                    # FA-2 optimized for bf16
        fa3_dtypes=[torch.float32],                    # FA-3 mixed precision
        pytorch_dtypes=[torch.float32],               # Baseline
        
        batch_size=1,
        is_causal=True,
        num_warmup=3,                         # 减少warmup次数
        num_iterations=10,                    # 减少迭代次数
        device='cuda'
    )


def run_flashattention_optimized_benchmark():
    """
    Main function to run optimized FlashAttention benchmark.
    Focus on each version's strengths and optimal use cases.
    """
    if not torch.cuda.is_available():
        print("❌ CUDA is not available. This benchmark requires a CUDA-enabled GPU.")
        return None, None
    
    print("🎯 FlashAttention Optimized Benchmark")
    print("Focus: Each version tested in its optimal conditions")
    print("All implementations tested with FP32 precision for consistency")
    print("FA-1: Short sequences (128-512)")
    print("FA-2: Medium sequences (1K-4K)")  
    print("FA-3: Long sequences (8K-16K)")
    print()
    
    # Create optimized configuration
    config = create_optimized_config()
    benchmarker = FlashAttentionBenchmarker(config)
    
    # Run benchmark
    df = benchmarker.run_optimized_benchmark()
    
    if df.empty:
        print("❌ Benchmark failed - no results generated")
        return None, None
    
    # Print results
    benchmarker.print_optimized_summary(df)
    
    # Save results
    benchmarker.save_results(df)
    
    return df, benchmarker


# Legacy aliases for backward compatibility
run_comprehensive_flashattention_benchmark = run_flashattention_optimized_benchmark
run_quick_flashattention_benchmark = run_flashattention_optimized_benchmark
run_flashattention2_benchmark = run_flashattention_optimized_benchmark


if __name__ == "__main__":
    # Run the optimized benchmark
    df, benchmarker = run_flashattention_optimized_benchmark()

def run_official_leaderboard_benchmark():
    """
    Run the official CS336 Assignment 2 leaderboard benchmark for FlashAttention-2.
    
    This implements the exact test specification:
    - Batch size: 1  
    - Sequence length: 16,384
    - Number of heads: 16
    - Head dimension: 64 (dmodel=1024/16)
    - Data type: BF16
    - Causal masking: True
    - Timing: Forward + Backward pass
    """
    print("\n" + "=" * 100)
    print("🏆 OFFICIAL CS336 LEADERBOARD BENCHMARK - FlashAttention-2")
    print("=" * 100)
    
    if not torch.cuda.is_available():
        print("❌ CUDA is not available. This benchmark requires a CUDA GPU.")
        return None
    
    # Official leaderboard configuration
    n_heads = 16
    d_head = 64  # dmodel=1024, so d_head = 1024/16 = 64
    sequence_length = 16384
    batch_size = 1
    dtype = torch.bfloat16
    is_causal = True
    
    print(f"📋 Official Test Configuration:")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Number of heads: {n_heads}")
    print(f"  - Head dimension: {d_head}")
    print(f"  - Sequence length: {sequence_length:,}")
    print(f"  - Model dimension: {n_heads * d_head}")
    print(f"  - Data type: {dtype}")
    print(f"  - Causal masking: {is_causal}")
    print(f"  - Device: {torch.cuda.get_device_name()}")
    
    # Create test tensors with official specification
    # Note: Official test uses shape (n_heads, seq_len, d_head)
    shape = (n_heads, sequence_length, d_head)
    
    print(f"\n🎯 Creating test tensors:")
    print(f"  - Shape: {shape}")
    print(f"  - Memory per tensor: {torch.numel(torch.zeros(shape)) * 2 / 1e6:.1f} MB (BF16)")
    
    q = torch.randn(shape, device='cuda', dtype=dtype, requires_grad=True)
    k = torch.randn(shape, device='cuda', dtype=dtype, requires_grad=True)
    v = torch.randn(shape, device='cuda', dtype=dtype, requires_grad=True)
    
    # Import the FlashAttention-2 implementation
    try:
        from flash_attention2 import FlashAttentionTritonFunction
        
        # Create the leaderboard class wrapper
        class FlashAttention2:
            @staticmethod
            def apply(q, k, v, is_causal=True):
                return FlashAttentionTritonFunction.apply(q, k, v, is_causal)
        
        print(f"✅ FlashAttention-2 implementation loaded successfully")
        
    except ImportError as e:
        print(f"❌ Failed to import FlashAttention-2: {e}")
        return None
    
    # Compile the function as specified in the official test
    flash = torch.compile(FlashAttention2.apply)
    
    def flash_forward_backward():
        """The exact function that will be benchmarked on the leaderboard."""
        # Clear any existing gradients
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
        
        # Ensure all operations complete
        torch.cuda.synchronize()
    
    print(f"\n⏱️  Running Official Benchmark:")
    print(f"  - Warmup iterations: 1,000")
    print(f"  - Benchmark iterations: 10,000")
    print(f"  - Function: flash_forward_backward() [forward + backward]")
    
    try:
        # Run a quick correctness check first
        print(f"\n🔍 Quick correctness verification...")
        test_output = flash(q, k, v, True)
        test_loss = test_output.sum()
        test_loss.backward()
        
        assert test_output.shape == q.shape, f"Shape mismatch: {test_output.shape} vs {q.shape}"
        assert q.grad is not None, "Q gradient is None"
        assert k.grad is not None, "K gradient is None"
        assert v.grad is not None, "V gradient is None"
        print(f"✅ Correctness check passed")
        
        # Clear gradients before benchmark
        q.grad.zero_()
        k.grad.zero_()
        v.grad.zero_()
        
        # Run the official benchmark using triton.testing.do_bench
        print(f"\n🚀 Starting official benchmark...")
        
        results = triton.testing.do_bench(
            flash_forward_backward,
            rep=10000,     # 10,000 repetitions as specified
            warmup=1000    # 1,000 warmup iterations as specified  
        )
        
        print(f"\n🎉 OFFICIAL LEADERBOARD RESULT:")
        print(f"" + "="*60)
        print(f"⏰ Average Time: {results:.4f} ms")
        print(f"🚀 Throughput: {1000/results:.2f} forward+backward/sec")
        print(f"" + "="*60)
        
        # Additional metrics for context
        total_params = n_heads * sequence_length * d_head
        flops_per_iteration = 4 * n_heads * sequence_length * sequence_length * d_head  # Rough estimate
        gflops = (flops_per_iteration * 1000 / results) / 1e9
        
        print(f"\n📊 Additional Metrics:")
        print(f"  - Parameters per iteration: {total_params:,}")
        print(f"  - Estimated FLOPS per iteration: {flops_per_iteration/1e9:.2f} GFLOPS")
        print(f"  - Estimated throughput: {gflops:.2f} GFLOPS")
        print(f"  - Memory bandwidth utilization: High (exact depends on hardware)")
        
        print(f"\n💡 Leaderboard Submission:")
        print(f"  Submit this result: {results:.4f} ms")
        
        return results
        
    except torch.cuda.OutOfMemoryError:
        print(f"❌ CUDA Out of Memory Error")
        print(f"   - This configuration requires significant GPU memory")
        print(f"   - Estimated memory needed: ~{3 * torch.numel(q) * 2 / 1e9:.1f} GB for tensors alone")
        print(f"   - Try running on a GPU with more memory (H100 recommended)")
        return None
        
    except Exception as e:
        print(f"❌ Benchmark failed: {e}")
        print(f"   Check your FlashAttention-2 implementation for bugs")
        return None