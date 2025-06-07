# my-triton-templates :floppy_disk:
This repo contains some typical triton kernel I wrote or used. I will regularly review and modify them to make my future kernel developing more easier.
## Update Memo :calling:
### Update June 6, 2025: from triton puzzles :gift_heart:
Lately, I walked through the fabulous work [![triton-puzzles](https://badgen.net/badge/Srush/Triton%20Puzzles/blue?icon=github)](https://github.com/srush/Triton-Puzzles/) to overcome the challenges I am facing in CS336 2025Spring. I passed the 12 tests and modified them to separate python files. They are very simple kernels that serve as cheat sheets for some classic algorithms, as well as how to load and store data from and to accelerators. 

### Update June 7, 2025: Flash Attentions from CS336 :rocket:
Inspired by [CS336 in Stanford](https://stanford-cs336.github.io/spring2025/), I reproduced the Flash Attention in this repo too. The course required a FlashAttention 2, but I also implemented version 1 and 3.
They are implemented with a Triton forward kernel and a torch backward function. For FlashAttention-2, I also implemented full pytorch version and full Triton verison (backward kernel adopted from Dao AILab) [![Dao AILab](https://badgen.net/badge/Dao-AILab/flash-attention/blue?icon=github)](https://github.com/Dao-AILab/flash-attention)
#### Different Implementations of Flash Attention 2

- My FA-2 and FA-2 FULL share the same forward kernel implementation, but FA-2 FULL incorporates a Triton backward kernel for enhanced performance.
> **Key Insight**: The results demonstrate the significant performance benefits of implementing custom Triton kernels for both custom forward & backward. However, the triton backward kernel performance still lags behind PyTorch existing implementations, highlighting the continued importance of CUDA for fine-grained GPU operations.💡 
#### Forward Performance vs PyTorch
| Configuration | FA-2 | FA-2-FULL | Speedup(FA-2) | Speedup(FA-2-FULL) |
|---|---|---|---|---|
| seq_len=1024, head_dim=32 | 0.41ms | 0.59ms | 1.48x | 1.01x |
| seq_len=1024, head_dim=64 | 0.54ms | 0.78ms | 0.92x | 0.63x |
| seq_len=2048, head_dim=32 | 0.45ms | 0.82ms | 1.42x | 0.78x |
| seq_len=2048, head_dim=64 | 0.53ms | 0.83ms | 1.26x | 0.81x |

#### Backward Performance vs PyTorch

| Implementation | Average Speedup | Max Speedup | Configuration |
|---|---|---|---|
| **FlashAttention2_Triton** | 0.00x | 0.00x | seq_len=1024, head_dim=64 |
| **FlashAttention2_FULL** | 0.41x | 0.48x | seq_len=1024, head_dim=64 |
| **FA2_CrossTest** | 0.03x | 0.15x | seq_len=128, head_dim=64 |
| **FA2_FULL_CrossTest** | 0.43x | 0.68x | seq_len=256, head_dim=64 |

#### FA-2 vs FA-2-FULL Backward Pass Comparison

| Configuration | FA-2 | FA-2-FULL | Winner | Speedup |
|---|---|---|---|---|
| seq_len=1024, head_dim=32 | 169.97ms | 1.17ms | **FA-2-FULL** | **144.80x** |
| seq_len=1024, head_dim=64 | 162.84ms | 1.39ms | **FA-2-FULL** | **117.21x** |
| seq_len=2048, head_dim=32 | 645.34ms | 1.35ms | **FA-2-FULL** | **477.32x** |
| seq_len=2048, head_dim=64 | 626.05ms | 1.67ms | **FA-2-FULL** | **375.88x** |

> **Conclusion**: The FlashAttention-2 FULL with Triton backward pass is much faster than FlashAttention-2's manual PyTorch backward pass. However, it's slower than the backward pass in PyTorch's built-in functions. Also, the manually implemented FlashAttention-2 forward propagation operator is faster than PyTorch's native multi-head attention. 🌟

### Long-term project (from June 3, 2025): LeetGPU :hourglass_flowing_sand:
I am also solving challenges on [`LeetGPU`](https://leetgpu.com/challenges). I plan to gradually move my solutions here as learning notes. 
