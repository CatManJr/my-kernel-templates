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


if TRITON_AVAILABLE:
    @triton.jit
    def flash_fwd_kernel_v1_correct(
        Q_ptr, K_ptr, V_ptr,
        O_ptr, L_ptr, M_ptr,
        stride_qb, stride_qq, stride_qd,
        stride_kb, stride_kk, stride_kd,
        stride_vb, stride_vk, stride_vd,
        stride_ob, stride_oq, stride_od,
        stride_lb, stride_lq,
        stride_mb, stride_mq,
        N_QUERIES, N_KEYS,
        scale,
        D: tl.constexpr,
        BR: tl.constexpr,  # Block size for queries (rows)
        BC: tl.constexpr,  # Block size for keys (columns)
        is_causal: tl.constexpr,
    ):
        """
        Correct FlashAttention-1 kernel following Algorithm 1 from the original paper.
        
        Key implementation details:
        - Outer loop over K/V blocks (j), inner loop over Q blocks (i)
        - Each thread block processes one K/V block against all Q blocks
        - Follows the exact memory access pattern from Algorithm 1
        """
        # Program indices - CRITICAL: each thread block handles one K/V block
        kv_block_idx = tl.program_id(0)  # Which K/V block (j in Algorithm 1)
        batch_idx = tl.program_id(1)     # Which batch
        
        # Calculate number of blocks (Algorithm 1, line 3-4)
        Tr = tl.cdiv(N_QUERIES, BR)  # Number of Q blocks
        Tc = tl.cdiv(N_KEYS, BC)      # Number of K/V blocks
        
        # Early exit if this thread block is out of bounds
        if kv_block_idx >= Tc:
            return
            
        # Calculate K/V block boundaries (Algorithm 1, line 5)
        kv_start = kv_block_idx * BC
        kv_end = tl.minimum(kv_start + BC, N_KEYS)
        
        # Load K_j and V_j blocks from HBM to SRAM (Algorithm 1, line 6)
        K_block_ptr = tl.make_block_ptr(
            K_ptr + batch_idx * stride_kb,
            shape=(N_KEYS, D),
            strides=(stride_kk, stride_kd),
            offsets=(kv_start, 0),
            block_shape=(BC, D),
            order=(1, 0),
        )
        
        V_block_ptr = tl.make_block_ptr(
            V_ptr + batch_idx * stride_vb,
            shape=(N_KEYS, D),
            strides=(stride_vk, stride_vd),
            offsets=(kv_start, 0),
            block_shape=(BC, D),
            order=(1, 0),
        )
        
        # Load K_j and V_j with boundary checking - STAY IN SRAM
        Kj = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero")
        Vj = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero")
        
        # Inner loop over Q blocks (Algorithm 1, line 7)
        for q_block_idx in range(Tr):
            q_start = q_block_idx * BR
            q_end = tl.minimum(q_start + BR, N_QUERIES)
            
            # Load Q_i, O_i, l_i, m_i from HBM to SRAM (Algorithm 1, line 8)
            Q_block_ptr = tl.make_block_ptr(
                Q_ptr + batch_idx * stride_qb,
                shape=(N_QUERIES, D),
                strides=(stride_qq, stride_qd),
                offsets=(q_start, 0),
                block_shape=(BR, D),
                order=(1, 0),
            )
            
            O_block_ptr = tl.make_block_ptr(
                O_ptr + batch_idx * stride_ob,
                shape=(N_QUERIES, D),
                strides=(stride_oq, stride_od),
                offsets=(q_start, 0),
                block_shape=(BR, D),
                order=(1, 0),
            )
            
            L_block_ptr = tl.make_block_ptr(
                L_ptr + batch_idx * stride_lb,
                shape=(N_QUERIES,),
                strides=(stride_lq,),
                offsets=(q_start,),
                block_shape=(BR,),
                order=(0,),
            )
            
            M_block_ptr = tl.make_block_ptr(
                M_ptr + batch_idx * stride_mb,
                shape=(N_QUERIES,),
                strides=(stride_mq,),
                offsets=(q_start,),
                block_shape=(BR,),
                order=(0,),
            )
            
            # Load current state
            Qi = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero")
            Oi = tl.load(O_block_ptr, boundary_check=(0, 1), padding_option="zero")
            li = tl.load(L_block_ptr, boundary_check=(0,), padding_option="zero")
            mi = tl.load(M_block_ptr, boundary_check=(0,), padding_option="zero")
            
            # Compute S_ij = Q_i @ K_j^T (Algorithm 1, line 9)
            Sij = tl.dot(Qi, tl.trans(Kj)) * scale
            
            # Apply causal mask if needed
            if is_causal:
                q_indices = q_start + tl.arange(0, BR)
                k_indices = kv_start + tl.arange(0, BC)
                causal_mask = q_indices[:, None] >= k_indices[None, :]
                Sij = tl.where(causal_mask, Sij, float('-inf'))
            
            # Compute row-wise max and softmax (Algorithm 1, line 10)
            m_tilde_ij = tl.max(Sij, axis=1)  # rowmax(S_ij)
            P_tilde_ij = tl.exp(Sij - m_tilde_ij[:, None])  # exp(S_ij - m_tilde_ij)
            l_tilde_ij = tl.sum(P_tilde_ij, axis=1)  # rowsum(P_tilde_ij)
            
            # Update max and normalization (Algorithm 1, line 11)
            m_new_i = tl.maximum(mi, m_tilde_ij)  # max(m_i, m_tilde_ij)
            
            # Compute exponential terms for updating l_i
            exp_mi = tl.exp(mi - m_new_i)
            exp_m_tilde = tl.exp(m_tilde_ij - m_new_i)
            l_new_i = exp_mi * li + exp_m_tilde * l_tilde_ij
            
            # Update output O_i (Algorithm 1, line 12)
            # O_i <- diag(l_new_i)^(-1) * (diag(l_i) * exp(m_i - m_new_i) * O_i + exp(m_tilde_ij - m_new_i) * P_tilde_ij * V_j)
            
            # First term: diag(l_i) * exp(m_i - m_new_i) * O_i
            scaling_old = (li * exp_mi) / l_new_i
            Oi_scaled = Oi * scaling_old[:, None]
            
            # Second term: exp(m_tilde_ij - m_new_i) * P_tilde_ij * V_j
            P_scaled = P_tilde_ij * (exp_m_tilde / l_new_i)[:, None]
            Oi_new = tl.dot(P_scaled.to(Vj.dtype), Vj)
            
            # Combine terms
            Oi_final = Oi_scaled + Oi_new
            
            # Store updated values back to HBM (Algorithm 1, line 12-13)
            tl.store(O_block_ptr, Oi_final.to(O_block_ptr.type.element_ty), boundary_check=(0, 1))
            tl.store(L_block_ptr, l_new_i, boundary_check=(0,))
            tl.store(M_block_ptr, m_new_i, boundary_check=(0,))

    class FlashAttention1TritonFunction(torch.autograd.Function):
        """
        Correct implementation of original FlashAttention-1 following Algorithm 1.
        
        This implementation strictly follows the algorithm from the original paper:
        - Outer loop over K/V blocks (j from 1 to Tc)  
        - Inner loop over Q blocks (i from 1 to Tr)
        - Block sizes: Bc = M/(4d), Br = min(M/(4d), d)
        - Each thread block processes one K/V block against all Q blocks
        """
        
        @staticmethod
        def forward(ctx, Q, K, V, is_causal=False):
            batch_size, seq_len, head_dim = Q.shape
            device = Q.device
            dtype = Q.dtype
            
            # Ensure contiguous tensors
            Q = Q.contiguous()
            K = K.contiguous() 
            V = V.contiguous()
            
            scale = 1.0 / math.sqrt(head_dim)
            
            # Block sizes following Algorithm 1 exactly
            # Assume SRAM size M = 128KB for block size calculation
            M = 128 * 1024  # 128KB SRAM
            BC = min(128, M // (4 * head_dim))  # Bc = M/(4d)
            BR = min(BC, head_dim)              # Br = min(M/(4d), d)
            
            # Ensure minimum block sizes and power of 2
            BC = max(16, BC)
            BR = max(16, BR)
            BC = 1 << (BC.bit_length() - 1) if BC > 0 else 16
            BR = 1 << (BR.bit_length() - 1) if BR > 0 else 16
            
            # Initialize outputs (Algorithm 1, line 2)
            O = torch.zeros_like(Q)  # O = (0)_N×d
            L = torch.zeros(batch_size, seq_len, device=device, dtype=torch.float32)  # ℓ = (0)_N
            M = torch.full((batch_size, seq_len), float('-inf'), device=device, dtype=torch.float32)  # m = (-∞)_N
            
            # Calculate number of blocks (Algorithm 1, line 3-4)
            Tr = triton.cdiv(seq_len, BR)  # Number of Q blocks
            Tc = triton.cdiv(seq_len, BC)  # Number of K/V blocks
            
            # Launch kernel - CRITICAL: grid size based on K/V blocks, not Q blocks
            grid = (Tc, batch_size)  # Each thread block handles one K/V block
            
            flash_fwd_kernel_v1_correct[grid](
                Q, K, V, O, L, M,
                Q.stride(0), Q.stride(1), Q.stride(2),
                K.stride(0), K.stride(1), K.stride(2),
                V.stride(0), V.stride(1), V.stride(2),
                O.stride(0), O.stride(1), O.stride(2),
                L.stride(0), L.stride(1),
                M.stride(0), M.stride(1),
                seq_len, seq_len, scale,
                D=head_dim,
                BR=BR,
                BC=BC,
                is_causal=is_causal,
                num_warps=4,
                num_stages=2,
            )
            
            # Convert L back to log space for backward pass
            L_log = M + torch.log(L)
            
            ctx.save_for_backward(Q, K, V, O, L_log)
            ctx.is_causal = is_causal
            ctx.scale = scale
            
            return O
        
        @staticmethod
        def backward(ctx, grad_output):
            """Backward pass using torch.compile and recomputation."""
            Q, K, V, O, L = ctx.saved_tensors
            is_causal = ctx.is_causal
            scale = ctx.scale
            
            @torch.compile(mode='default', fullgraph=False)
            def compute_gradients(Q, K, V, O, L, grad_output, is_causal, scale):
                # Standard recomputation-based backward pass
                scores = torch.matmul(Q, K.transpose(-2, -1)) * scale
                
                if is_causal:
                    mask = torch.triu(torch.ones_like(scores, dtype=torch.bool), diagonal=1)
                    scores.masked_fill_(mask, float('-inf'))
                
                attn_probs = torch.exp(scores - L.unsqueeze(-1))
                
                # Compute gradients
                grad_V = torch.matmul(attn_probs.transpose(-2, -1), grad_output)
                grad_attn_probs = torch.matmul(grad_output, V.transpose(-2, -1))
                
                D = torch.sum(grad_output * O, dim=-1, keepdim=True)
                grad_scores = attn_probs * (grad_attn_probs - D) * scale
                
                if is_causal:
                    grad_scores.masked_fill_(mask, 0.0)
                
                grad_Q = torch.matmul(grad_scores, K)
                grad_K = torch.matmul(grad_scores.transpose(-2, -1), Q)
                
                return grad_Q, grad_K, grad_V
            
            return compute_gradients(Q, K, V, O, L, grad_output, is_causal, scale) + (None,)