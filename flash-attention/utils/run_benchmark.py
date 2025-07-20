#!/usr/bin/env python3
"""
Optimized FlashAttention Benchmark Runner

Run optimized benchmarks comparing our three FlashAttention implementations
(FA-1, FA-2, FA-3) with standard PyTorch attention, focusing on each version's
optimal conditions and data precisions.

Usage: uv run utils/run_benchmark.py
"""

import sys
import os

# Add the parent directory to the path so we can import flash_attention modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import dependencies (managed by uv)
import pandas as pd
import torch
import triton
from utils.benchmark import (
    run_flashattention_optimized_benchmark,
    create_optimized_config,
    FlashAttentionBenchmarker,
    OptimizedBenchmarkConfig,
    standard_pytorch_attention
)


def run_quick_optimized_benchmark():
    """Run a quick optimized benchmark with reduced configurations."""
    # Create quick config with fewer test cases
    config = OptimizedBenchmarkConfig(
        short_sequences=[128, 256],          # FA-1 strength (reduced)
        medium_sequences=[256, 512],         # FA-2 strength (reduced)  
        long_sequences=[1024, 2048],         # FA-3 using same as FA-2 to avoid hanging
        head_dims=[64],                      # Focus on common head_dim
        fa1_dtypes=[torch.float32],          # Only FP32
        fa2_dtypes=[torch.float32],          # Only FP32
        fa3_dtypes=[torch.float32],          # Only FP32
        pytorch_dtypes=[torch.float32],      # Only FP32
        batch_size=1,
        is_causal=True,
        num_warmup=5,
        num_iterations=10,                   # Fewer iterations for speed
        device='cuda'
    )
    
    benchmarker = FlashAttentionBenchmarker(config)
    
    print("Quick Optimized FlashAttention Benchmark")
    print("Focus: Each version tested in its sweet spot")
    print("Reduced configurations for faster execution")
    print()
    
    df = benchmarker.run_optimized_benchmark()
    
    if df.empty:
        print("Quick benchmark failed")
        return None, None
    
    benchmarker.print_optimized_summary(df)
    return df, benchmarker


def run_custom_optimized_benchmark():
    """Run a custom benchmark with user-specified optimizations."""
    print("Custom Optimized Benchmark Configuration")
    print("=" * 50)
    
    # Get user preferences for sequence ranges
    print("Configure sequence length ranges for each FA version:")
    
    # FA-1 sequences
    fa1_input = input("FA-1 sequences (comma-separated, default: 128,256,512): ").strip()
    if fa1_input:
        try:
            short_sequences = [int(x.strip()) for x in fa1_input.split(',')]
        except ValueError:
            print("Invalid input, using default")
            short_sequences = [128, 256, 512]
    else:
        short_sequences = [128, 256, 512]
    
    # FA-2 sequences
    fa2_input = input("FA-2 sequences (comma-separated, default: 1024,2048,4096): ").strip()
    if fa2_input:
        try:
            medium_sequences = [int(x.strip()) for x in fa2_input.split(',')]
        except ValueError:
            print("Invalid input, using default")
            medium_sequences = [1024, 2048, 4096]
    else:
        medium_sequences = [1024, 2048, 4096]
    
    # FA-3 sequences
    fa3_input = input("FA-3 sequences (comma-separated, default: 8192,16384): ").strip()
    if fa3_input:
        try:
            long_sequences = [int(x.strip()) for x in fa3_input.split(',')]
        except ValueError:
            print("Invalid input, using default")
            long_sequences = [4096, 8192]
    else:
        long_sequences = [4096, 8192]
    
    # Head dimensions
    head_input = input("Head dimensions (comma-separated, default: 32,64,128): ").strip()
    if head_input:
        try:
            head_dims = [int(x.strip()) for x in head_input.split(',')]
        except ValueError:
            print("Invalid input, using default")
            head_dims = [4, 16, 64]
    else:
        head_dims = [4, 16, 64]
    
    # Precision selection
    print("\nPrecision configuration:")
    print("FA-1 works best with float32")
    print("FA-2 is optimized for bfloat16") 
    print("FA-3 uses mixed precision (bfloat16)")
    
    use_optimal = input("Use optimal precisions for each version? (Y/n): ").strip().lower()
    if use_optimal in ['n', 'no']:
        # Let user choose precisions
        dtype_map = {"float32": torch.float32, "bfloat16": torch.bfloat16}
        
        fa1_dtype_input = input("FA-1 dtypes (float32,bfloat16, default: float32): ").strip()
        fa1_dtypes = [dtype_map.get(x.strip(), torch.float32) for x in fa1_dtype_input.split(',') if x.strip()]
        if not fa1_dtypes:
            fa1_dtypes = [torch.float32]
        
        fa2_dtype_input = input("FA-2 dtypes (float32,bfloat16, default: bfloat16): ").strip()
        fa2_dtypes = [dtype_map.get(x.strip(), torch.bfloat16) for x in fa2_dtype_input.split(',') if x.strip()]
        if not fa2_dtypes:
            fa2_dtypes = [torch.bfloat16]
        
        fa3_dtype_input = input("FA-3 dtypes (float32,bfloat16, default: bfloat16): ").strip()
        fa3_dtypes = [dtype_map.get(x.strip(), torch.bfloat16) for x in fa3_dtype_input.split(',') if x.strip()]
        if not fa3_dtypes:
            fa3_dtypes = [torch.bfloat16]
    else:
        # Use optimal precisions
        fa1_dtypes = [torch.float32]
        fa2_dtypes = [torch.bfloat16]
        fa3_dtypes = [torch.bfloat16]
    
    # Create custom config
    config = OptimizedBenchmarkConfig(
        short_sequences=short_sequences,
        medium_sequences=medium_sequences,
        long_sequences=long_sequences,
        head_dims=head_dims,
        fa1_dtypes=fa1_dtypes,
        fa2_dtypes=fa2_dtypes,
        fa3_dtypes=fa3_dtypes,
        pytorch_dtypes=[torch.float32],
        batch_size=1,
        is_causal=True,
        num_warmup=5,
        num_iterations=20,
        device='cuda'
    )
    
    benchmarker = FlashAttentionBenchmarker(config)
    
    print(f"\nRunning Custom Optimized FlashAttention Benchmark")
    print(f"FA-1: {len(short_sequences)} sequences × {len(head_dims)} head_dims × {len(fa1_dtypes)} dtypes")
    print(f"FA-2: {len(medium_sequences)} sequences × {len(head_dims)} head_dims × {len(fa2_dtypes)} dtypes") 
    print(f"FA-3: {len(long_sequences)} sequences × {len(head_dims)} head_dims × {len(fa3_dtypes)} dtypes")
    print("=" * 80)
    
    df = benchmarker.run_optimized_benchmark()
    
    if df.empty:
        print("Custom benchmark failed")
        return None, None
        
    benchmarker.print_optimized_summary(df)
    return df, benchmarker


def run_version_comparison():
    """Run a focused comparison between FA versions at their optimal conditions."""
    print("FlashAttention Version Comparison")
    print("Testing each version at its optimal sequence length and precision")
    print("=" * 70)
    
    # Import implementations
    try:
        # FA-1: Uses flash_attention1_triton function
        from flash_attention import flash_attention1_triton
        
        # FA-2: Uses flash_attention_triton function
        from flash_attention2 import flash_attention_triton
        
        # FA-3: Uses FlashAttention3TritonFunction class  
        from flash_attention3 import FlashAttention3TritonFunction
        def flash_attention_v3(Q, K, V, is_causal=False):
            return FlashAttention3TritonFunction.apply(Q, K, V, is_causal)
            
    except ImportError as e:
        print(f"Failed to import FlashAttention implementations: {e}")
        return None, None
    
    # Create focused comparison config
    config = OptimizedBenchmarkConfig(
        short_sequences=[512],        # FA-1 optimal
        medium_sequences=[2048],      # FA-2 optimal
        long_sequences=[8192],        # FA-3 optimal
        head_dims=[64, 128],          # Common dimensions
        fa1_dtypes=[torch.float32],   # Only FP32
        fa2_dtypes=[torch.float32],   # Only FP32
        fa3_dtypes=[torch.float32],   # Only FP32
        pytorch_dtypes=[torch.float32],
        batch_size=1,
        is_causal=True,
        num_warmup=10,
        num_iterations=30,            # More iterations for precision
        device='cuda'
    )
    
    benchmarker = FlashAttentionBenchmarker(config)
    df = benchmarker.run_optimized_benchmark()
    
    if df.empty:
        print("Version comparison failed")
        return None, None
    
    # Custom analysis for version comparison
    print("\n" + "=" * 100)
    print("FLASHATTENTION VERSION COMPARISON ANALYSIS")
    print("=" * 100)
    
    # Analyze each version at its optimal condition
    print("\nOPTIMAL PERFORMANCE ANALYSIS:")
    print("-" * 60)
    
    optimal_conditions = [
        ("FA-1", 512, torch.float32, "FlashAttention1_Triton"),
        ("FA-2", 2048, torch.bfloat16, "FlashAttention2_Triton"), 
        ("FA-3", 8192, torch.bfloat16, "FlashAttention3_Triton")
    ]
    
    for version, seq_len, dtype, impl_name in optimal_conditions:
        dtype_str = str(dtype).split('.')[-1]
        
        # Get results for this optimal condition
        condition_df = df[
            (df['seq_len'] == seq_len) & 
            (df['dtype'] == dtype_str) & 
            (df['implementation'] == impl_name)
        ]
        
        if not condition_df.empty:
            result = condition_df.iloc[0]
            pytorch_df = df[
                (df['seq_len'] == seq_len) & 
                (df['dtype'] == 'float32') &  # PyTorch baseline is always float32
                (df['implementation'] == 'PyTorch_Standard')
            ]
            
            if not pytorch_df.empty:
                pytorch_time = pytorch_df.iloc[0]['end_to_end_ms']
                speedup = pytorch_time / result['end_to_end_ms'] if result['end_to_end_ms'] != float('inf') else 0
                
                print(f"{version} (seq_len={seq_len}, {dtype_str}):")
                print(f"  Forward: {result['forward_ms']:.2f}ms")
                print(f"  Backward: {result['backward_ms']:.2f}ms") 
                print(f"  End-to-End: {result['end_to_end_ms']:.2f}ms")
                print(f"  Speedup vs PyTorch: {speedup:.2f}x")
                print()
        else:
            print(f"{version}: FAILED or no data")
            print()
    
    benchmarker.print_optimized_summary(df)
    return df, benchmarker


def run_precision_analysis():
    """Analyze the impact of different precisions on each FA version."""
    print("FlashAttention Precision Analysis")
    print("Testing impact of float32 vs bfloat16 on each version")
    print("=" * 65)
    
    # Test all versions with both precisions
    config = OptimizedBenchmarkConfig(
        short_sequences=[512],                                # Representative for FA-1
        medium_sequences=[2048],                              # Representative for FA-2
        long_sequences=[8192],                                # Representative for FA-3
        head_dims=[64],                                       # Focus on common head_dim
        fa1_dtypes=[torch.float32, torch.bfloat16],          # Test both for FA-1
        fa2_dtypes=[torch.float32, torch.bfloat16],          # Test both for FA-2
        fa3_dtypes=[torch.float32, torch.bfloat16],          # Test both for FA-3
        pytorch_dtypes=[torch.float32],
        batch_size=1,
        is_causal=True,
        num_warmup=5,
        num_iterations=15,
        device='cuda'
    )
    
    benchmarker = FlashAttentionBenchmarker(config)
    df = benchmarker.run_optimized_benchmark()
    
    if df.empty:
        print("Precision analysis failed")
        return None, None
    
    # Custom precision analysis
    print("\n" + "=" * 100)
    print("PRECISION IMPACT ANALYSIS")
    print("=" * 100)
    
    versions = [
        ("FA-1", 512, "FlashAttention1_Triton"),
        ("FA-2", 2048, "FlashAttention2_Triton"),
        ("FA-3", 8192, "FlashAttention3_Triton")
    ]
    
    for version, seq_len, impl_name in versions:
        print(f"\n{version} (seq_len={seq_len}):")
        print("-" * 40)
        
        for dtype_str in ['float32', 'bfloat16']:
            result_df = df[
                (df['seq_len'] == seq_len) & 
                (df['dtype'] == dtype_str) & 
                (df['implementation'] == impl_name)
            ]
            
            if not result_df.empty:
                result = result_df.iloc[0]
                print(f"  {dtype_str}: {result['end_to_end_ms']:.2f}ms "
                      f"(speedup: {result['speedup_vs_pytorch']:.2f}x)")
            else:
                print(f"  {dtype_str}: FAILED or no data")
    
    benchmarker.print_optimized_summary(df)
    return df, benchmarker


def main():
    """Main optimized benchmark runner with multiple options."""
    if not torch.cuda.is_available():
        print("CUDA is not available. This benchmark requires a CUDA-enabled GPU.")
        return
    
    print("FlashAttention Optimized Benchmark Suite")
    print("Focus: Each version tested in its optimal conditions")
    print("=" * 60)
    print("1. Quick Optimized Benchmark (fast, each version's strengths)")
    print("2. Full Optimized Benchmark (comprehensive, all optimal conditions)")
    print("3. Version Comparison (head-to-head at optimal settings)")
    print("4. Precision Analysis (float32 vs bfloat16 impact)")
    print("5. Custom Optimized Benchmark (user-specified configs)")
    print()
    
    choice = input("Choose benchmark type (1-5): ").strip()
    
    try:
        if choice == "1":
            print("\nRunning Quick Optimized Benchmark...")
            df, benchmarker = run_quick_optimized_benchmark()
        elif choice == "2":
            print("\nRunning Full Optimized Benchmark...")
            df, benchmarker = run_flashattention_optimized_benchmark()
        elif choice == "3":
            print("\nRunning Version Comparison...")
            df, benchmarker = run_version_comparison()
        elif choice == "4":
            print("\nRunning Precision Analysis...")
            df, benchmarker = run_precision_analysis()
        elif choice == "5":
            print("\nRunning Custom Optimized Benchmark...")
            df, benchmarker = run_custom_optimized_benchmark()
        else:
            print("Invalid choice. Running Quick Optimized Benchmark by default.")
            df, benchmarker = run_quick_optimized_benchmark()
    except Exception as e:
        print(f"Benchmark failed with error: {e}")
        import traceback
        traceback.print_exc()
        return
    
    if df is None or df.empty:
        print("No benchmark results generated")
        return
    
    print("\nBenchmark completed successfully!")
    
    # Option to save detailed results
    save = input("\nSave detailed results to CSV? (y/N): ").strip().lower()
    if save in ['y', 'yes']:
        import time
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"flashattention_optimized_benchmark_{timestamp}.csv"
        benchmarker.save_results(df, filename)
    
    # Option to show best configurations
    best = input("Show best configuration for each sequence length? (y/N): ").strip().lower()
    if best in ['y', 'yes']:
        print("\n" + "=" * 100)
        print("BEST CONFIGURATION SUMMARY")
        print("=" * 100)
        
        # Group by sequence length and find best implementation
        unique_seq_lens = sorted(df['seq_len'].unique())
        
        for seq_len in unique_seq_lens:
            seq_df = df[
                (df['seq_len'] == seq_len) & 
                (df['implementation'] != 'PyTorch_Standard') &
                (df['end_to_end_ms'] != float('inf'))
            ]
            
            if not seq_df.empty:
                best_idx = seq_df['end_to_end_ms'].idxmin()
                best_result = seq_df.loc[best_idx]
                
                speedup_str = f"{best_result['speedup_vs_pytorch']:.2f}x" if best_result['speedup_vs_pytorch'] else "N/A"
                print(f"seq_len={seq_len}: {best_result['implementation']} "
                      f"({best_result['dtype']}) - {best_result['end_to_end_ms']:.2f}ms ({speedup_str})")


if __name__ == "__main__":
    main()