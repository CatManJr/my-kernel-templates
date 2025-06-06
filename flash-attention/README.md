# Flash-attention Triton Kernels
Inspired by the CS336 course at Stanford, I am working on implementing Flash Attention kernels using Triton. This repository will contain my implementations of Flash Attention versions 1, 2, and 3.
#  Strongly Recommended: Using UV to solve the environments
We use `uv` to manage dependencies. That is a fabulous toolkit built with Rust. You can test the the code `flash_attention2.py` by running:
```bash
uv run python pytest
```
the test is modified from the original one in the course, I only kept the test for Flash Attention 2. If you need more libraries, use `uv add <library>` to add them, and then run the test again.