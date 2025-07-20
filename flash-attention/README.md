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

### Backward Pass Performance Comparison (vs PyTorch)

| Configuration | FA-2 Backward | FA-2-FULL Backward | PyTorch Backward | FA-2 vs PyTorch | FA-2-FULL vs PyTorch |
|---|---|---|---|---|---|
| seq_len=1024, head_dim=32 | 162.99ms | 1.33ms | 0.63ms | 0.004x | 0.47x |
| seq_len=1024, head_dim=64 | 160.83ms | 1.71ms | 0.68ms | 0.004x | 0.40x |
| seq_len=2048, head_dim=32 | 647.28ms | 1.38ms | 0.50ms | 0.001x | 0.36x |
| seq_len=2048, head_dim=64 | 629.93ms | 1.75ms | 0.54ms | 0.001x | 0.31x |

### FA-2 vs FA-2-FULL Backward Pass Comparison

| Configuration | FA-2 | FA-2-FULL | Winner | Speedup |
|---|---|---|---|---|
| seq_len=1024, head_dim=32 | 162.99ms | 1.33ms | **FA-2-FULL** | **122.69x** |
| seq_len=1024, head_dim=64 | 160.83ms | 1.71ms | **FA-2-FULL** | **94.28x** |
| seq_len=2048, head_dim=32 | 647.28ms | 1.38ms | **FA-2-FULL** | **468.33x** |
| seq_len=2048, head_dim=64 | 629.93ms | 1.75ms | **FA-2-FULL** | **359.30x** |

> **Summary**: The results demonstrate the significant performance benefits of implementing custom Triton kernels for backward passes. FA-2-FULL achieves 94x-468x speedups in backward pass compared to FA-2's PyTorch backward implementation. I used more operations and storage to fit in DaoAI's backward kernel, so there still exist large performance gaps to PyTorch API.

### Forward Pass Performance vs PyTorch

| Configuration | FA-2 | FA-2-FULL | FA-2 Speedup | FA-2-FULL Speedup |
|---|---|---|---|---|
| seq_len=1024, head_dim=32 | 0.41ms | 0.65ms | 1.12x | 0.71x |
| seq_len=1024, head_dim=64 | 0.47ms | 1.00ms | 0.94x | 0.44x |
| seq_len=2048, head_dim=32 | 0.46ms | 1.78ms | 1.16x | 0.30x |
| seq_len=2048, head_dim=64 | 0.61ms | 0.92ms | 0.94x | 0.63x |

> **Summary**: FA-2-FULL's forward pass shows problems from pure Triton implementation. Because I used more operations and storage to fit in DaoAI's backward kernel, FA-2's Triton forward offers better forward pass performance although they both use the same forward kernel.