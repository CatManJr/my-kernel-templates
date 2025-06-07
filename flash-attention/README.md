# Flash-attention Triton Kernels
Inspired by the CS336 course at Stanford, I am working on implementing Flash Attention kernels using Triton. This repository will contain my implementations of Flash Attention versions 1, 2, and 3.
##  Strongly Recommended: Using UV to solve the environments
We use `uv` to manage dependencies. That is a fabulous toolkit built with Rust. Fisrt， install it with pip:
```bash
pip install uv
```
Then, you need to create a configuration file `pyproject.toml` in the root directory of the project. You can refer to the one in this repository.
Then you can run any code with 
```bash
uv run <your_script.py>
``` 
You can test the the algo `flash_attention2.py` by running:
```bash
uv run python pytest
```
the test is modified from the original one in the course, I only kept the test for Flash Attention 2. If you need more libraries, use `uv add <library>` to add them, and then run the test again.

## Benchmark
You can run the benchmark with:
```bash
uv run utils/run_benchmark.py
```
## Different Implementations of Flash Attention 2

FA-2 and FA-2 FULL share the same forward kernel implementation, but FA-2 FULL incorporates a Triton backward kernel for enhanced performance.

### Backward Pass Performance vs PyTorch

| Implementation | Average Speedup | Max Speedup | Best Configuration |
|---|---|---|---|
| **FlashAttention2_Triton** | 0.00x | 0.00x | seq_len=1024, head_dim=64 |
| **FlashAttention2_FULL** | 0.41x | 0.48x | seq_len=1024, head_dim=64 |
| **FA2_CrossTest** | 0.03x | 0.15x | seq_len=128, head_dim=64 |
| **FA2_FULL_CrossTest** | 0.43x | 0.68x | seq_len=256, head_dim=64 |

### FA-2 vs FA-2-FULL Backward Pass Comparison

| Configuration | FA-2 | FA-2-FULL | Winner | Speedup |
|---|---|---|---|---|
| seq_len=1024, head_dim=32 | 169.97ms | 1.17ms | **FA-2-FULL** | **144.80x** |
| seq_len=1024, head_dim=64 | 162.84ms | 1.39ms | **FA-2-FULL** | **117.21x** |
| seq_len=2048, head_dim=32 | 645.34ms | 1.35ms | **FA-2-FULL** | **477.32x** |
| seq_len=2048, head_dim=64 | 626.05ms | 1.67ms | **FA-2-FULL** | **375.88x** |

> **Key Insight**: The results demonstrate the significant performance benefits of implementing custom Triton kernels for backward passes. However, forward pass performance still lags behind PyTorch implementations, highlighting the continued importance of CUDA for fine-grained GPU operations.