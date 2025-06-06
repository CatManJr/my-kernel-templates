#!/usr/bin/env python3
"""
Comprehensive FlashAttention Benchmark Runner

Run comprehensive benchmarks comparing all FlashAttention implementations
(v1, v2, and v3 when available) with standard PyTorch attention.
"""

import sys
import os

# Add the parent directory to the path so we can import flash_attention modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Check dependencies
try:
    import pandas as pd
except ImportError:
    print("Installing pandas for benchmark results...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pandas"])
    import pandas as pd

import torch
import triton
from utils.benchmark import (
    run_comprehensive_flashattention_benchmark, 
    run_quick_flashattention_benchmark,
    create_default_config, 
    create_quick_config,
    AttentionBenchmarker
)


def run_custom_benchmark():
    """Run a custom benchmark with user-specified configurations."""
    from utils.benchmark import BenchmarkConfig, standard_pytorch_attention
    
    try:
        from flash_attention2 import flash_attention_triton as flash_attention2_triton, flash_attention_all_triton
        from flash_attention import flash_attention_v1
        # TODO: Uncomment when FlashAttention-3 is implemented
        # from flash_attention3 import flash_attention_v3
    except ImportError as e:
        print(f"Failed to import FlashAttention implementations: {e}")
        return
    
    print("Custom Benchmark Configuration")
    print("=" * 40)
    
    # Get user preferences
    print("Available sequence lengths: 128, 256, 512, 1024, 2048, 4096, 8192, 16384")
    seq_input = input("Enter sequence lengths (comma-separated, or press Enter for default): ").strip()
    if seq_input:
        try:
            sequence_lengths = [int(x.strip()) for x in seq_input.split(',')]
        except ValueError:
            print("Invalid input, using default sequence lengths")
            sequence_lengths = [128, 512, 1024, 2048]
    else:
        sequence_lengths = [128, 512, 1024, 2048]
    
    print("Available head dimensions: 16, 32, 64, 128")
    head_input = input("Enter head dimensions (comma-separated, or press Enter for default): ").strip()
    if head_input:
        try:
            head_dims = [int(x.strip()) for x in head_input.split(',')]
        except ValueError:
            print("Invalid input, using default head dimensions")
            head_dims = [32, 64]
    else:
        head_dims = [32, 64]
    
    print("Available dtypes: float32, bfloat16")
    dtype_input = input("Enter dtypes (comma-separated, or press Enter for default): ").strip()
    if dtype_input:
        dtype_map = {"float32": torch.float32, "bfloat16": torch.bfloat16}
        try:
            dtypes = [dtype_map[x.strip()] for x in dtype_input.split(',') if x.strip() in dtype_map]
            if not dtypes:
                raise ValueError("No valid dtypes")
        except (ValueError, KeyError):
            print("Invalid input, using default dtypes")
            dtypes = [torch.float32]
    else:
        dtypes = [torch.float32]
    
    # Custom config
    config = BenchmarkConfig(
        sequence_lengths=sequence_lengths,
        head_dims=head_dims,
        dtypes=dtypes,
        batch_size=1,
        num_heads=1,
        is_causal=True,
        num_warmup=5,
        num_iterations=20,
        device='cuda'
    )
    
    benchmarker = AttentionBenchmarker(config)
    
    # Ask which implementations to include
    print("\nAvailable implementations:")
    print("1. PyTorch Standard")
    print("2. FlashAttention v1 (Triton)")
    print("3. FlashAttention v2 (Triton)")
    print("4. FlashAttention v2 All-Triton (Full Triton forward+backward)")
    # print("5. FlashAttention v3 (Triton)")  # TODO: Uncomment when v3 is available
    
    impl_input = input("Select implementations to benchmark (e.g., '1,2,3,4' or press Enter for all): ").strip()
    
    all_implementations = {
        '1': ('PyTorch_Standard', standard_pytorch_attention),
        '2': ('FlashAttention1_Triton', flash_attention_v1),
        '3': ('FlashAttention2_Triton', flash_attention2_triton),
        '4': ('FlashAttention2_AllTriton', flash_attention2_all_triton),
        # '5': ('FlashAttention3_Triton', flash_attention_v3),  # TODO: Uncomment when v3 is available
    }
    
    if impl_input:
        try:
            selected = impl_input.split(',')
            implementations = {name: func for idx, (name, func) in all_implementations.items() if idx in selected}
        except:
            print("Invalid selection, using all implementations")
            implementations = {name: func for name, func in all_implementations.values()}
    else:
        implementations = {name: func for name, func in all_implementations.values()}
    
    if not implementations:
        print("No valid implementations selected, using all")
        implementations = {name: func for name, func in all_implementations.values()}
    
    print(f"\nRunning Custom FlashAttention Benchmark")
    print(f"Configurations: {len(sequence_lengths)} seq_lens × {len(head_dims)} head_dims × {len(dtypes)} dtypes")
    print(f"Implementations: {list(implementations.keys())}")
    print("=" * 60)
    
    df = benchmarker.run_full_benchmark(implementations)
    benchmarker.print_summary_table(df)
    
    return df, benchmarker


def run_performance_comparison():
    """Run a focused performance comparison between FlashAttention versions."""
    from utils.benchmark import BenchmarkConfig, standard_pytorch_attention
    
    try:
        from flash_attention2 import flash_attention_triton as flash_attention2_triton, flash_attention2_all_triton
        from flash_attention import flash_attention_v1
        # TODO: Uncomment when FlashAttention-3 is implemented
        # from flash_attention3 import flash_attention_v3
    except ImportError as e:
        print(f"Failed to import FlashAttention implementations: {e}")
        return
    
    # Performance-focused config - test larger sequences where differences are more apparent
    config = BenchmarkConfig(
        sequence_lengths=[1024, 2048, 4096, 8192],  # Focus on larger sequences
        head_dims=[64, 128],  # Common head dimensions
        dtypes=[torch.bfloat16],  # Use bfloat16 for better performance
        batch_size=1,
        num_heads=1,
        is_causal=True,
        num_warmup=10,
        num_iterations=50,  # More iterations for stable measurements
        device='cuda'
    )
    
    benchmarker = AttentionBenchmarker(config)
    
    implementations = {
        'PyTorch_Standard': standard_pytorch_attention,
        'FlashAttention1_Triton': flash_attention_v1,
        'FlashAttention2_Triton': flash_attention2_triton,
        'FlashAttention2_AllTriton': flash_attention2_all_triton,
        # TODO: Uncomment when FlashAttention-3 is implemented
        # 'FlashAttention3_Triton': flash_attention_v3,
    }
    
    print("Performance-Focused FlashAttention Comparison")
    print("Focus: Large sequences, bfloat16 precision")
    print(f"Configurations: {len(config.sequence_lengths)} seq_lens × {len(config.head_dims)} head_dims")
    print(f"Implementations: {list(implementations.keys())}")
    print("=" * 60)
    
    df = benchmarker.run_full_benchmark(implementations)
    benchmarker.print_summary_table(df)
    
    # Performance analysis
    print("\n" + "=" * 80)
    print("PERFORMANCE ANALYSIS")
    print("=" * 80)
    
    for seq_len in config.sequence_lengths:
        seq_data = df[df['seq_len'] == seq_len]
        if not seq_data.empty:
            print(f"\nSequence Length {seq_len}:")
            pytorch_time = seq_data[seq_data['implementation'] == 'PyTorch_Standard']['end_to_end_ms'].iloc[0]
            
            for impl in ['FlashAttention1_Triton', 'FlashAttention2_Triton', 'FlashAttention2_AllTriton']:
                if impl in seq_data['implementation'].values:
                    flash_time = seq_data[seq_data['implementation'] == impl]['end_to_end_ms'].iloc[0]
                    if flash_time != float('inf') and pytorch_time != float('inf'):
                        speedup = pytorch_time / flash_time
                        print(f"  {impl}: {speedup:.2f}x speedup over PyTorch")
                    else:
                        print(f"  {impl}: FAILED or insufficient data")
    
    return df, benchmarker


def main():
    """Main benchmark runner with multiple options."""
    if not torch.cuda.is_available():
        print("CUDA is not available. This benchmark requires a CUDA-enabled GPU.")
        return
    
    print("FlashAttention Comprehensive Benchmark Suite")
    print("=" * 50)
    print("1. Quick Benchmark (fast, limited configs)")
    print("2. Comprehensive Benchmark (all versions, full configs)")
    print("3. Performance Comparison (focus on speedup analysis)")
    print("4. Custom Benchmark (user-specified configs)")
    print()
    
    choice = input("Choose benchmark type (1-4): ").strip()
    
    try:
        if choice == "1":
            print("\nRunning Quick Benchmark...")
            df, benchmarker = run_quick_flashattention_benchmark()
        elif choice == "2":
            print("\nRunning Comprehensive Benchmark...")
            df, benchmarker = run_comprehensive_flashattention_benchmark()
        elif choice == "3":
            print("\nRunning Performance Comparison...")
            df, benchmarker = run_performance_comparison()
        elif choice == "4":
            print("\nRunning Custom Benchmark...")
            df, benchmarker = run_custom_benchmark()
        else:
            print("Invalid choice. Running Quick Benchmark by default.")
            df, benchmarker = run_quick_flashattention_benchmark()
    except Exception as e:
        print(f"Benchmark failed with error: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\nBenchmark completed successfully!")
    
    # Option to save detailed results
    save = input("\nSave detailed results to CSV? (y/N): ").strip().lower()
    if save in ['y', 'yes']:
        import time
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"flashattention_benchmark_{timestamp}.csv"
        benchmarker.save_results(df, filename)
    
    # Option to show performance summary
    summary = input("Show performance summary? (y/N): ").strip().lower()
    if summary in ['y', 'yes']:
        print("\n" + "=" * 80)
        print("PERFORMANCE SUMMARY")
        print("=" * 80)
        
        # Show best performing implementation for each config
        grouped = df.groupby(['seq_len', 'head_dim', 'dtype'])
        for (seq_len, head_dim, dtype), group in grouped:
            if len(group) > 1:
                best = group.loc[group['end_to_end_ms'].idxmin()]
                print(f"seq_len={seq_len}, head_dim={head_dim}, dtype={dtype}: "
                      f"Best = {best['implementation']} ({best['end_to_end_ms']:.3f}ms)")


if __name__ == "__main__":
    main()