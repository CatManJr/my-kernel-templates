# My-Kernel-Templates :floppy_disk:
This repo contains some typical triton kernels (maybe cuda as well) I wrote or used. I will regularly review and modify them to make my future kernel developing more easier.
## Update Memo :calling:
### Update June 6, 2025: from triton puzzles :gift_heart:
Lately, I walked through the fabulous work [![triton-puzzles](https://badgen.net/badge/Srush/Triton%20Puzzles/blue?icon=github)](https://github.com/srush/Triton-Puzzles/) to overcome the challenges I am facing in CS336 2025Spring. I passed the 12 tests and modified them to separate python files. They are very simple kernels that serve as cheat sheets for some classic algorithms, as well as how to load and store data from and to accelerators. 

### Update June 7, 2025: Flash Attentions from CS336 :rocket:
Inspired by [CS336 in Stanford](https://stanford-cs336.github.io/spring2025/), I reproduced the Flash Attention in this repo too. The course required a FlashAttention 2, but I also implemented version 1 and 3.
They are implemented with a Triton forward kernel and a torch backward function. For FlashAttention-2, I also implemented full pytorch version and full Triton verison (backward kernel adopted from Dao AILab) [![Dao AILab](https://badgen.net/badge/Dao-AILab/flash-attention/blue?icon=github)](https://github.com/Dao-AILab/flash-attention)
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

### Long-term project (from June 3, 2025): LeetGPU :hourglass_flowing_sand:
I am also solving challenges on [`LeetGPU`](https://leetgpu.com/challenges). I plan to gradually move my solutions here as learning notes. 
