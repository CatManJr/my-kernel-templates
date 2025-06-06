"""
FlashAttention-1 implementation in Triton.

This module implements the original FlashAttention algorithm for efficient attention computation
with reduced memory usage through tiling and online softmax computation.

Key differences from FlashAttention-2:
1. Different parallelization strategy (sequence-level vs batch-level)
2. Simpler tiling approach
3. Different memory access patterns
"""

import math
import torch
from typing import Optional, Tuple

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False
    raise ImportError("Triton is required for FlashAttention-1 implementation")


@triton.jit
def flash_attention_v1_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr, L_ptr,
    stride_qb, stride_qh, stride_qm, stride_qk,
    stride_kb, stride_kh, stride_kn, stride_kk, 
    stride_vb, stride_vh, stride_vn, stride_vk,
    stride_ob, stride_oh, stride_om, stride_ok,
    stride_lb, stride_lh, stride_lm,
    B, H, M, N, K,
    scale,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    is_causal: tl.constexpr,
):
    """
    FlashAttention-1 Triton kernel with improved dtype handling and numerical stability.
    """
    # Program IDs - FlashAttention-1 uses different parallelization
    pid_m = tl.program_id(0)
    pid_b = tl.program_id(1) 
    pid_h = tl.program_id(2)
    
    # Number of blocks
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    
    # Reorder program IDs for better cache locality (FlashAttention-1 specific)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid_m // GROUP_SIZE_M
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid_m % group_size_m)
    
    # Compute offsets
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    
    # Initialize pointers for Q
    q_ptrs = (Q_ptr + 
              pid_b * stride_qb + 
              pid_h * stride_qh +
              offs_m[:, None] * stride_qm + 
              offs_k[None, :] * stride_qk)
    
    # Initialize accumulators - use float32 for numerical stability
    acc = tl.zeros([BLOCK_M, BLOCK_K], dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    m_i = tl.full([BLOCK_M], value=float('-inf'), dtype=tl.float32)
    
    # Load Q block and convert to float32 for computation
    q_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
    q = tl.load(q_ptrs, mask=q_mask, other=0.0)
    q = q.to(tl.float32)  # Ensure float32 for computation
    
    # Loop over K, V blocks
    for start_n in range(0, N, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        
        # Compute the actual block size for this iteration
        curr_n = tl.minimum(BLOCK_N, N - start_n)
        offs_n_curr = start_n + tl.arange(0, BLOCK_N)
        
        # Initialize pointers for K and V
        k_ptrs = (K_ptr + 
                  pid_b * stride_kb + 
                  pid_h * stride_kh +
                  offs_n_curr[:, None] * stride_kn + 
                  offs_k[None, :] * stride_kk)
        
        v_ptrs = (V_ptr + 
                  pid_b * stride_vb + 
                  pid_h * stride_vh +
                  offs_n_curr[:, None] * stride_vn + 
                  offs_k[None, :] * stride_vk)
        
        # Load K and V blocks
        k_mask = (offs_n_curr[:, None] < N) & (offs_k[None, :] < K)
        v_mask = (offs_n_curr[:, None] < N) & (offs_k[None, :] < K)
        
        k = tl.load(k_ptrs, mask=k_mask, other=0.0)
        v = tl.load(v_ptrs, mask=v_mask, other=0.0)
        
        # Convert to float32 for computation
        k = k.to(tl.float32)
        v = v.to(tl.float32)
        
        # Compute attention scores S_ij = Q @ K^T
        s_ij = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        s_ij = tl.dot(q, tl.trans(k), s_ij) * scale
        
        # Apply causal mask if needed
        if is_causal:
            causal_mask = (offs_m[:, None] >= (start_n + offs_n_curr[None, :]))
            s_ij = tl.where(causal_mask, s_ij, float('-inf'))
        
        # Mask out invalid positions
        valid_mask = (offs_m[:, None] < M) & ((start_n + offs_n_curr[None, :]) < N)
        s_ij = tl.where(valid_mask, s_ij, float('-inf'))
        
        # Online softmax computation (FlashAttention-1 algorithm)
        m_ij = tl.max(s_ij, 1)  # Row-wise max
        m_i_new = tl.maximum(m_i, m_ij)
        
        # Compute P_ij = exp(S_ij - m_i_new)
        p_ij = tl.exp(s_ij - m_i_new[:, None])
        
        # Update normalization factor
        l_i_new = tl.exp(m_i - m_i_new) * l_i + tl.sum(p_ij, 1)
        
        # Update output
        alpha = tl.exp(m_i - m_i_new)
        acc = acc * alpha[:, None] + tl.dot(p_ij, v)
        
        # Update running statistics
        l_i = l_i_new
        m_i = m_i_new
    
    # Final normalization with safety check
    l_i_safe = tl.maximum(l_i, 1e-20)  # Prevent division by zero
    acc = acc / l_i_safe[:, None]
    
    # Compute final log-sum-exp for backward pass
    l_final = m_i + tl.log(l_i_safe)
    
    # Store output (convert back to original dtype)
    o_ptrs = (O_ptr + 
              pid_b * stride_ob + 
              pid_h * stride_oh +
              offs_m[:, None] * stride_om + 
              offs_k[None, :] * stride_ok)
    
    l_ptrs = (L_ptr +
              pid_b * stride_lb +
              pid_h * stride_lh + 
              offs_m * stride_lm)
    
    o_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
    l_mask = offs_m < M
    
    # Store with original dtype
    tl.store(o_ptrs, acc.to(O_ptr.dtype.element_ty), mask=o_mask)
    tl.store(l_ptrs, l_final, mask=l_mask)


@triton.jit  
def flash_attention_v1_backward_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr, DO_ptr, L_ptr,
    DQ_ptr, DK_ptr, DV_ptr, D_ptr,
    stride_qb, stride_qh, stride_qm, stride_qk,
    stride_kb, stride_kh, stride_kn, stride_kk,
    stride_vb, stride_vh, stride_vn, stride_vk,
    stride_ob, stride_oh, stride_om, stride_ok,
    stride_dob, stride_doh, stride_dom, stride_dok,
    stride_lb, stride_lh, stride_lm,
    stride_dqb, stride_dqh, stride_dqm, stride_dqk,
    stride_dkb, stride_dkh, stride_dkn, stride_dkk,
    stride_dvb, stride_dvh, stride_dvn, stride_dvk,
    stride_db, stride_dh, stride_dm,
    B, H, M, N, K,
    scale,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    is_causal: tl.constexpr,
):
    """FlashAttention-1 backward kernel with proper gradient accumulation."""
    pid_n = tl.program_id(0)  # 改为按N维度并行化以避免竞争条件
    pid_b = tl.program_id(1)
    pid_h = tl.program_id(2)
    
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_m = tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, BLOCK_K)
    
    # 初始化dK和dV累积器
    dk_acc = tl.zeros([BLOCK_N, BLOCK_K], dtype=tl.float32)
    dv_acc = tl.zeros([BLOCK_N, BLOCK_K], dtype=tl.float32)
    
    # 加载K和V块
    k_ptrs = (K_ptr + pid_b * stride_kb + pid_h * stride_kh +
              offs_n[:, None] * stride_kn + offs_k[None, :] * stride_kk)
    v_ptrs = (V_ptr + pid_b * stride_vb + pid_h * stride_vh +
              offs_n[:, None] * stride_vn + offs_k[None, :] * stride_vk)
    
    k_mask = (offs_n[:, None] < N) & (offs_k[None, :] < K)
    v_mask = (offs_n[:, None] < N) & (offs_k[None, :] < K)
    
    k = tl.load(k_ptrs, mask=k_mask, other=0.0).to(tl.float32)
    v = tl.load(v_ptrs, mask=v_mask, other=0.0).to(tl.float32)
    
    # 循环处理所有Q块
    for start_m in range(0, M, BLOCK_M):
        offs_m_curr = start_m + tl.arange(0, BLOCK_M)
        
        # 加载Q, O, DO, L, D
        q_ptrs = (Q_ptr + pid_b * stride_qb + pid_h * stride_qh +
                  offs_m_curr[:, None] * stride_qm + offs_k[None, :] * stride_qk)
        o_ptrs = (O_ptr + pid_b * stride_ob + pid_h * stride_oh +
                  offs_m_curr[:, None] * stride_om + offs_k[None, :] * stride_ok)
        do_ptrs = (DO_ptr + pid_b * stride_dob + pid_h * stride_doh +
                   offs_m_curr[:, None] * stride_dom + offs_k[None, :] * stride_dok)
        l_ptrs = (L_ptr + pid_b * stride_lb + pid_h * stride_lh + offs_m_curr * stride_lm)
        d_ptrs = (D_ptr + pid_b * stride_db + pid_h * stride_dh + offs_m_curr * stride_dm)
        
        q_mask = (offs_m_curr[:, None] < M) & (offs_k[None, :] < K)
        l_mask = offs_m_curr < M
        
        q = tl.load(q_ptrs, mask=q_mask, other=0.0).to(tl.float32)
        o = tl.load(o_ptrs, mask=q_mask, other=0.0).to(tl.float32)
        do = tl.load(do_ptrs, mask=q_mask, other=0.0).to(tl.float32)
        l = tl.load(l_ptrs, mask=l_mask, other=0.0)
        d = tl.load(d_ptrs, mask=l_mask, other=0.0)
        
        # 重计算注意力分数
        s_ij = tl.dot(q, tl.trans(k)) * scale
        
        # 应用因果掩码和有效掩码
        valid_mask = (offs_m_curr[:, None] < M) & (offs_n[None, :] < N)
        if is_causal:
            causal_mask = (offs_m_curr[:, None] >= offs_n[None, :])
            valid_mask = valid_mask & causal_mask
            s_ij = tl.where(causal_mask, s_ij, float('-inf'))
        
        s_ij = tl.where(valid_mask, s_ij, float('-inf'))
        
        # 计算softmax概率
        p_ij = tl.exp(s_ij - l[:, None])
        p_ij = tl.where(valid_mask, p_ij, 0.0)
        
        # 计算梯度
        dp_ij = tl.dot(do, tl.trans(v))
        ds_ij = p_ij * (dp_ij - d[:, None]) * scale
        ds_ij = tl.where(valid_mask, ds_ij, 0.0)
        
        # 累积dK和dV梯度
        dk_acc += tl.dot(tl.trans(ds_ij), q)
        dv_acc += tl.dot(tl.trans(p_ij), do)
        
        # 计算并存储dQ（每个M块独立）
        dq = tl.dot(ds_ij, k)
        dq_ptrs = (DQ_ptr + pid_b * stride_dqb + pid_h * stride_dqh +
                   offs_m_curr[:, None] * stride_dqm + offs_k[None, :] * stride_dqk)
        tl.store(dq_ptrs, dq.to(DQ_ptr.dtype.element_ty), mask=q_mask)
    
    # store dK and dV gradients
    dk_ptrs = (DK_ptr + pid_b * stride_dkb + pid_h * stride_dkh +
               offs_n[:, None] * stride_dkn + offs_k[None, :] * stride_dkk)
    dv_ptrs = (DV_ptr + pid_b * stride_dvb + pid_h * stride_dvh +
               offs_n[:, None] * stride_dvn + offs_k[None, :] * stride_dvk)
    
    tl.store(dk_ptrs, dk_acc.to(DK_ptr.dtype.element_ty), mask=k_mask)
    tl.store(dv_ptrs, dv_acc.to(DV_ptr.dtype.element_ty), mask=v_mask)
    


class FlashAttentionV1Function(torch.autograd.Function):
    """
    FlashAttention-1 implementation using Triton.
    
    This is the original FlashAttention algorithm with:
    - Sequence-level parallelization
    - Simpler tiling strategy
    - Different memory access patterns compared to V2
    """
    
    @staticmethod
    def forward(ctx, q, k, v, is_causal=False):
        # Input validation
        assert q.dim() == k.dim() == v.dim() == 4, "Expected 4D tensors (batch, heads, seq_len, head_dim)"
        B, H, M, K = q.shape
        _, _, N, _ = k.shape
        assert k.shape == v.shape, "K and V must have the same shape"
        assert q.shape[-1] == k.shape[-1], "Q and K must have the same head dimension"
        
        # Ensure contiguous memory layout
        q = q.contiguous()
        k = k.contiguous() 
        v = v.contiguous()
        
        # Initialize output tensors
        o = torch.empty_like(q)
        l = torch.empty((B, H, M), device=q.device, dtype=torch.float32)
        
        # FlashAttention-1 specific hyperparameters
        scale = 1.0 / math.sqrt(K)
        
        # Block sizes - FlashAttention-1 uses smaller, more conservative blocks
        BLOCK_M = 64  # Query block size
        BLOCK_N = 64  # Key/Value block size  
        BLOCK_K = K   # Head dimension (no blocking on this dim)
        GROUP_SIZE_M = 8
        
        # Launch kernel
        grid = (triton.cdiv(M, BLOCK_M), B, H)
        
        flash_attention_v1_kernel[grid](
            q, k, v, o, l,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            o.stride(0), o.stride(1), o.stride(2), o.stride(3),
            l.stride(0), l.stride(1), l.stride(2),
            B, H, M, N, K,
            scale,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            GROUP_SIZE_M=GROUP_SIZE_M,
            is_causal=is_causal,
            num_warps=4,
            num_stages=2,
        )
        
        # Save tensors for backward pass
        ctx.save_for_backward(q, k, v, o, l)
        ctx.is_causal = is_causal
        ctx.scale = scale
        ctx.BLOCK_M = BLOCK_M
        ctx.BLOCK_N = BLOCK_N
        ctx.BLOCK_K = BLOCK_K
        
        return o
    
    @staticmethod
    def backward(ctx, grad_output):
        q, k, v, o, l = ctx.saved_tensors
        is_causal = ctx.is_causal
        scale = ctx.scale
        BLOCK_M = ctx.BLOCK_M
        BLOCK_N = ctx.BLOCK_N  
        BLOCK_K = ctx.BLOCK_K
        
        B, H, M, K = q.shape
        N = k.shape[2]
        
        # Compute D = sum(dO * O, dim=-1)
        d = torch.sum(grad_output * o, dim=-1)
        
        # Initialize gradient tensors
        dq = torch.zeros_like(q)
        dk = torch.zeros_like(k)
        dv = torch.zeros_like(v)
        
        # Launch backward kernel with N-dimension parallelization
        grid = (triton.cdiv(N, BLOCK_N), B, H)
        
        flash_attention_v1_backward_kernel[grid](
            q, k, v, o, grad_output, l,
            dq, dk, dv, d,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            o.stride(0), o.stride(1), o.stride(2), o.stride(3),
            grad_output.stride(0), grad_output.stride(1), grad_output.stride(2), grad_output.stride(3),
            l.stride(0), l.stride(1), l.stride(2),
            dq.stride(0), dq.stride(1), dq.stride(2), dq.stride(3),
            dk.stride(0), dk.stride(1), dk.stride(2), dk.stride(3),
            dv.stride(0), dv.stride(1), dv.stride(2), dv.stride(3),
            d.stride(0), d.stride(1), d.stride(2),
            B, H, M, N, K,
            scale,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            is_causal=is_causal,
            num_warps=4,
            num_stages=2,
        )
        
        return dq, dk, dv, None


def flash_attention_v1(q, k, v, is_causal=False):
    """
    FlashAttention-1 implementation.
    
    Args:
        q: Query tensor (batch, heads, seq_len, head_dim)
        k: Key tensor (batch, heads, seq_len, head_dim)  
        v: Value tensor (batch, heads, seq_len, head_dim)
        is_causal: Whether to apply causal masking
        
    Returns:
        Output tensor with same shape as q
    """
    if not TRITON_AVAILABLE:
        raise RuntimeError("Triton is not available for FlashAttention-1")
    
    return FlashAttentionV1Function.apply(q, k, v, is_causal)


# Compatibility aliases
flash_attention = flash_attention_v1
flash_attention_triton = flash_attention_v1