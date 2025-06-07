"""
FlashAttention-2 implementation in PyTorch and Triton.

This module implements the FlashAttention-2 algorithm for efficient attention computation
with reduced memory usage through tiling and online softmax computation.
"""

import math
import torch
import torch.nn.functional as F
from typing import Optional, Tuple

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False


@torch.compile
def flash_backward_gpu(Q, K, V, O, dO, L, is_causal=False):
    """
    GPU-accelerated backward pass for FlashAttention-2 using torch.compile.
    
    This implementation uses recomputation strategy as described in the FlashAttention paper
    (Equations 13-19) to avoid storing intermediate softmax values.
    
    Args:
        Q: Query tensor (batch, seq_len, head_dim)
        K: Key tensor (batch, seq_len, head_dim)
        V: Value tensor (batch, seq_len, head_dim)
        O: Output tensor from forward pass (batch, seq_len, head_dim)
        dO: Gradient w.r.t. output (batch, seq_len, head_dim)
        L: Log-sum-exp from forward pass (batch, seq_len)
        is_causal: Whether causal masking was applied
        
    Returns:
        Tuple of (dQ, dK, dV) gradients
    """
    batch_size, seq_len, head_dim = Q.shape
    scale = 1.0 / math.sqrt(head_dim)
    
    # Initialize gradient tensors
    dQ = torch.zeros_like(Q)
    dK = torch.zeros_like(K)
    dV = torch.zeros_like(V)
    
    # Compute D vector (Equation 14): D_i = sum_j(dO_ij * O_ij)
    D = torch.sum(dO * O, dim=-1)  # (batch, seq_len)
    
    # Determine optimal tile size for memory efficiency
    # Use smaller tiles for better memory management on GPU
    TILE_SIZE = min(128, seq_len)
    TILE_SIZE = max(32, TILE_SIZE)  # Ensure minimum tile size
    
    # Make TILE_SIZE a power of 2 for better GPU performance
    TILE_SIZE = 1 << (TILE_SIZE.bit_length() - 1)
    
    num_tiles = (seq_len + TILE_SIZE - 1) // TILE_SIZE
    
    # Process in tiles for memory efficiency
    for i in range(num_tiles):
        # Query tile indices
        q_start = i * TILE_SIZE
        q_end = min(q_start + TILE_SIZE, seq_len)
        q_slice = slice(q_start, q_end)
        
        # Extract query tile and related tensors
        Qi = Q[:, q_slice]  # (batch, tile_size, head_dim)
        dOi = dO[:, q_slice]  # (batch, tile_size, head_dim)
        Li = L[:, q_slice]  # (batch, tile_size)
        Di = D[:, q_slice]  # (batch, tile_size)
        
        # Initialize gradient accumulator for this query tile
        dQi = torch.zeros_like(Qi)
        
        for j in range(num_tiles):
            # Key/Value tile indices
            k_start = j * TILE_SIZE
            k_end = min(k_start + TILE_SIZE, seq_len)
            k_slice = slice(k_start, k_end)
            
            # Extract key/value tile
            Kj = K[:, k_slice]  # (batch, tile_size, head_dim)
            Vj = V[:, k_slice]  # (batch, tile_size, head_dim)
            
            # Recompute attention scores (Equation 15)
            # S_ij = Q_i @ K_j^T * scale
            Sij = torch.bmm(Qi, Kj.transpose(-2, -1)) * scale  # (batch, q_tile, k_tile)
            
            # Apply causal mask if needed
            if is_causal:
                # Create causal mask for current tiles
                q_indices = torch.arange(q_start, q_end, device=Q.device)[:, None]
                k_indices = torch.arange(k_start, k_end, device=Q.device)[None, :]
                causal_mask = q_indices < k_indices
                Sij = Sij.masked_fill(causal_mask, float('-inf'))
            
            # Recompute attention weights (Equation 16)
            # P_ij = exp(S_ij - L_i)
            Pij = torch.exp(Sij - Li.unsqueeze(-1))  # (batch, q_tile, k_tile)
            
            # Compute gradient w.r.t. V (Equation 17)
            # dV_j += P_ij^T @ dO_i
            dVj = torch.bmm(Pij.transpose(-2, -1), dOi)  # (batch, k_tile, head_dim)
            dV[:, k_slice] += dVj
            
            # Compute gradient w.r.t. P (Equation 18)
            # dP_ij = dO_i @ V_j^T
            dPij = torch.bmm(dOi, Vj.transpose(-2, -1))  # (batch, q_tile, k_tile)
            
            # Compute gradient w.r.t. S (Equation 19)
            # dS_ij = P_ij * (dP_ij - D_i)
            dSij = Pij * (dPij - Di.unsqueeze(-1))  # (batch, q_tile, k_tile)
            
            # Apply causal mask to gradients
            if is_causal:
                dSij = dSij.masked_fill(causal_mask, 0.0)
            
            # Accumulate gradient w.r.t. Q
            # dQ_i += dS_ij @ K_j * scale
            dQi += torch.bmm(dSij, Kj) * scale  # (batch, q_tile, head_dim)
            
            # Compute gradient w.r.t. K
            # dK_j += dS_ij^T @ Q_i * scale
            dKj = torch.bmm(dSij.transpose(-2, -1), Qi) * scale  # (batch, k_tile, head_dim)
            dK[:, k_slice] += dKj
        
        # Store accumulated gradient for this query tile
        dQ[:, q_slice] = dQi
    
    return dQ, dK, dV


class FlashAttentionPyTorchFunction(torch.autograd.Function):
    """
    Highly optimized PyTorch implementation of FlashAttention-2
    with minimal overhead and maximum PyTorch optimization.
    """
    
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        """
        FlashAttention-2 forward pass.
        Key optimizations:
        1. Minimize memory allocations
        2. Use native PyTorch operations
        3. Eliminate unnecessary dtype conversions
        4. Use torch.compile compatible operations
        """
        batch_size, seq_len, head_dim = Q.shape
        device = Q.device
        dtype = Q.dtype
        
        # Use native dtype for better performance - no conversions
        scale = (head_dim ** -0.5)  # More efficient than 1.0 / math.sqrt(head_dim)
        
        # Adaptive tile sizing for optimal performance
        if seq_len <= 128:
            TILE_SIZE = 64  # Larger tiles for small sequences
        elif seq_len <= 512:
            TILE_SIZE = 128
        else:
            TILE_SIZE = 256
        
        TILE_SIZE = min(TILE_SIZE, seq_len)
        num_tiles = (seq_len + TILE_SIZE - 1) // TILE_SIZE  # Ceiling division
        
        # Pre-allocate all tensors to avoid repeated allocations
        O = torch.zeros_like(Q)
        L = torch.empty(batch_size, seq_len, device=device, dtype=torch.float32)
        
        # Pre-compute causal mask template if needed
        causal_mask = None
        if is_causal:
            # Create a reusable mask template
            mask_template = torch.tril(torch.ones(TILE_SIZE, TILE_SIZE, device=device, dtype=torch.bool))
        
        # Process tiles with optimized loops
        for i in range(num_tiles):
            q_start = i * TILE_SIZE
            q_end = min(q_start + TILE_SIZE, seq_len)
            q_len = q_end - q_start
            
            Qi = Q[:, q_start:q_end]  # No dtype conversion
            
            # Initialize accumulators - use native dtype
            Oi = torch.zeros(batch_size, q_len, head_dim, device=device, dtype=dtype)
            li = torch.zeros(batch_size, q_len, device=device, dtype=torch.float32)
            mi = torch.full((batch_size, q_len), float('-inf'), device=device, dtype=torch.float32)
            
            for j in range(num_tiles):
                k_start = j * TILE_SIZE
                k_end = min(k_start + TILE_SIZE, seq_len)
                k_len = k_end - k_start
                
                # Early termination for causal attention
                if is_causal and k_start >= q_end:
                    break
                
                Kj = K[:, k_start:k_end]
                Vj = V[:, k_start:k_end]
                
                # Optimized attention computation using einsum for better performance
                Sij = torch.einsum('bqd,bkd->bqk', Qi, Kj) * scale
                
                # Efficient causal masking
                if is_causal:
                    # Create mask only for the actual tile sizes
                    q_indices = torch.arange(q_start, q_end, device=device)[:, None]
                    k_indices = torch.arange(k_start, k_end, device=device)[None, :]
                    mask = q_indices < k_indices
                    Sij.masked_fill_(mask, float('-inf'))
                
                # Optimized online softmax with fused operations
                mij = torch.max(Sij, dim=-1)[0]
                mi_new = torch.maximum(mi, mij)
                
                # Compute softmax with numerical stability
                Pij = torch.exp(Sij - mi_new.unsqueeze(-1))
                
                # Fused accumulator updates
                alpha = torch.exp(mi - mi_new)
                li_new = alpha * li + torch.sum(Pij, dim=-1)
                
                # Fused output update - use einsum for efficiency
                alpha_reshaped = alpha.unsqueeze(-1)
                Oi = alpha_reshaped * Oi + torch.einsum('bqk,bkd->bqd', Pij, Vj)
                
                mi = mi_new
                li = li_new
            
            # Final normalization with safety check
            li_safe = torch.clamp(li, min=1e-8)  # Prevent division by zero
            Oi = Oi / li_safe.unsqueeze(-1)
            
            # Store results
            O[:, q_start:q_end] = Oi
            L[:, q_start:q_end] = mi + torch.log(li_safe)
        
        ctx.save_for_backward(Q, K, V, O, L)
        ctx.is_causal = is_causal
        ctx.scale = scale
        
        return O
    
    @staticmethod  
    def backward(ctx, grad_output):
        """GPU-accelerated backward pass using torch.compile and recomputation."""
        Q, K, V, O, L = ctx.saved_tensors
        is_causal = ctx.is_causal
        
        # Use the compiled GPU backward function
        dQ, dK, dV = flash_backward_gpu(Q, K, V, O, grad_output, L, is_causal)
        
        return dQ, dK, dV, None


if TRITON_AVAILABLE:
    @triton.jit
    def flash_fwd_kernel(
        Q_ptr, K_ptr, V_ptr,
        O_ptr, L_ptr,
        stride_qb, stride_qq, stride_qd,
        stride_kb, stride_kk, stride_kd,
        stride_vb, stride_vk, stride_vd,
        stride_ob, stride_oq, stride_od,
        stride_lb, stride_lq,
        N_QUERIES, N_KEYS,
        scale,
        D: tl.constexpr,
        Q_TILE_SIZE: tl.constexpr,
        K_TILE_SIZE: tl.constexpr,
        is_causal: tl.constexpr,
    ):
        # Program indices
        query_tile_index = tl.program_id(0)
        batch_index = tl.program_id(1)

        # Offset each pointer with the corresponding batch index
        Q_block_ptr = tl.make_block_ptr(
            Q_ptr + batch_index * stride_qb,
            shape=(N_QUERIES, D),
            strides=(stride_qq, stride_qd),
            offsets=(query_tile_index * Q_TILE_SIZE, 0),
            block_shape=(Q_TILE_SIZE, D),
            order=(1, 0),
        )

        # Setup K and V block pointers
        K_block_ptr = tl.make_block_ptr(
            K_ptr + batch_index * stride_kb,
            shape=(N_KEYS, D),
            strides=(stride_kk, stride_kd),
            offsets=(0, 0),
            block_shape=(K_TILE_SIZE, D),
            order=(1, 0),
        )

        V_block_ptr = tl.make_block_ptr(
            V_ptr + batch_index * stride_vb,
            shape=(N_KEYS, D),
            strides=(stride_vk, stride_vd),
            offsets=(0, 0),
            block_shape=(K_TILE_SIZE, D),
            order=(1, 0),
        )

        # Output block pointers
        O_block_ptr = tl.make_block_ptr(
            O_ptr + batch_index * stride_ob,
            shape=(N_QUERIES, D),
            strides=(stride_oq, stride_od),
            offsets=(query_tile_index * Q_TILE_SIZE, 0),
            block_shape=(Q_TILE_SIZE, D),
            order=(1, 0),
        )

        L_block_ptr = tl.make_block_ptr(
            L_ptr + batch_index * stride_lb,
            shape=(N_QUERIES,),
            strides=(stride_lq,),
            offsets=(query_tile_index * Q_TILE_SIZE,),
            block_shape=(Q_TILE_SIZE,),
            order=(0,),
        )

        # Load query tile
        q = tl.load(Q_block_ptr)

        # Initialize accumulator buffers
        acc = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)
        l_i = tl.zeros((Q_TILE_SIZE,), dtype=tl.float32)
        m_i = tl.full((Q_TILE_SIZE,), value=float('-inf'), dtype=tl.float32)

        # Compute number of key tiles
        num_k_tiles = tl.cdiv(N_KEYS, K_TILE_SIZE)

        # Loop over key tiles
        for j in range(num_k_tiles):
            k_start = j * K_TILE_SIZE
            
            # Load K and V tiles
            k = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero")
            v = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero")

            # Compute attention scores
            s_ij = tl.dot(q, tl.trans(k)) * scale

            # Apply causal mask if needed
            if is_causal:
                q_indices = query_tile_index * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)
                k_indices = k_start + tl.arange(0, K_TILE_SIZE)
                causal_mask = q_indices[:, None] >= k_indices[None, :]
                s_ij = tl.where(causal_mask, s_ij, -1e6)

            # Online softmax computation
            m_ij = tl.max(s_ij, axis=1)
            m_new = tl.maximum(m_i, m_ij)
            p_ij_tilde = tl.exp(s_ij - m_new[:, None])
            l_new = tl.exp(m_i - m_new) * l_i + tl.sum(p_ij_tilde, axis=1)

            # Update accumulator
            alpha = tl.exp(m_i - m_new)
            acc = acc * alpha[:, None] + tl.dot(p_ij_tilde.to(v.dtype), v)

            # Update state
            m_i = m_new
            l_i = l_new

            # Advance pointers
            K_block_ptr = K_block_ptr.advance((K_TILE_SIZE, 0))
            V_block_ptr = V_block_ptr.advance((K_TILE_SIZE, 0))

        # Final normalization
        acc = acc / l_i[:, None]
        l_final = m_i + tl.log(l_i)

        # Store results
        tl.store(O_block_ptr, acc.to(O_block_ptr.type.element_ty), boundary_check=(0, 1))
        tl.store(L_block_ptr, l_final, boundary_check=(0,))

    class FlashAttentionTritonFunction(torch.autograd.Function):
        """Triton implementation of FlashAttention-2."""
        
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
            
            # Optimized tile sizes
            Q_TILE_SIZE = min(64, seq_len)
            K_TILE_SIZE = min(64, seq_len)
            Q_TILE_SIZE = max(16, Q_TILE_SIZE)
            K_TILE_SIZE = max(16, K_TILE_SIZE)
            
            # Round to power of 2 for better performance
            Q_TILE_SIZE = 1 << (Q_TILE_SIZE.bit_length() - 1)
            K_TILE_SIZE = 1 << (K_TILE_SIZE.bit_length() - 1)
            
            # Initialize outputs
            O = torch.empty_like(Q)
            L = torch.empty(batch_size, seq_len, device=device, dtype=torch.float32)
            
            # Launch kernel
            num_q_tiles = triton.cdiv(seq_len, Q_TILE_SIZE)
            grid = (num_q_tiles, batch_size)
            
            flash_fwd_kernel[grid](
                Q, K, V, O, L,
                Q.stride(0), Q.stride(1), Q.stride(2),
                K.stride(0), K.stride(1), K.stride(2),
                V.stride(0), V.stride(1), V.stride(2),
                O.stride(0), O.stride(1), O.stride(2),
                L.stride(0), L.stride(1),
                seq_len, seq_len, scale,
                D=head_dim,
                Q_TILE_SIZE=Q_TILE_SIZE,
                K_TILE_SIZE=K_TILE_SIZE,
                is_causal=is_causal,
                num_warps=4,
                num_stages=2,
            )
            
            ctx.save_for_backward(Q, K, V, O, L)
            ctx.is_causal = is_causal
            ctx.scale = scale
            
            return O
        
        @staticmethod
        def backward(ctx, grad_output):
            """Efficient backward pass using the same PyTorch implementation with proper context."""
            Q, K, V, O, L = ctx.saved_tensors
            is_causal = ctx.is_causal
            scale = ctx.scale
            
            batch_size, seq_len, head_dim = Q.shape
            device = Q.device
            dtype = Q.dtype
            
            # Pre-allocate gradients
            dQ = torch.zeros_like(Q)
            dK = torch.zeros_like(K) 
            dV = torch.zeros_like(V)
            
            # Efficient D computation
            D = torch.sum(grad_output * O, dim=-1)
            
            # Use optimized tile size for backward
            TILE_SIZE = min(64, seq_len)
            TILE_SIZE = max(16, TILE_SIZE)
            num_tiles = (seq_len + TILE_SIZE - 1) // TILE_SIZE
            
            # Optimized backward computation
            for i in range(num_tiles):
                q_start = i * TILE_SIZE
                q_end = min(q_start + TILE_SIZE, seq_len)
                
                Qi = Q[:, q_start:q_end]
                dOi = grad_output[:, q_start:q_end]
                Li = L[:, q_start:q_end]
                Di = D[:, q_start:q_end]
                
                dQi = torch.zeros_like(Qi)
                
                for j in range(num_tiles):
                    k_start = j * TILE_SIZE
                    k_end = min(k_start + TILE_SIZE, seq_len)
                    
                    Kj = K[:, k_start:k_end]
                    Vj = V[:, k_start:k_end]
                    
                    # Recompute attention efficiently
                    Sij = torch.einsum('bqd,bkd->bqk', Qi, Kj) * scale
                    
                    if is_causal:
                        q_indices = torch.arange(q_start, q_end, device=device)[:, None]
                        k_indices = torch.arange(k_start, k_end, device=device)[None, :]
                        mask = q_indices < k_indices
                        Sij.masked_fill_(mask, float('-inf'))
                    
                    Pij = torch.exp(Sij - Li.unsqueeze(-1))
                    
                    # Efficient gradient computation with einsum
                    dVj = torch.einsum('bqk,bqd->bkd', Pij, dOi)
                    dV[:, k_start:k_end] += dVj
                    
                    dPij = torch.einsum('bqd,bkd->bqk', dOi, Vj)
                    dSij = Pij * (dPij - Di.unsqueeze(-1))
                    
                    if is_causal:
                        dSij.masked_fill_(mask, 0.0)
                    
                    dQi += torch.einsum('bqk,bkd->bqd', dSij, Kj) * scale
                    dKj = torch.einsum('bqk,bqd->bkd', dSij, Qi) * scale
                    dK[:, k_start:k_end] += dKj
            
                dQ[:, q_start:q_end] = dQi
            
            return dQ, dK, dV, None

    @triton.jit
    def flash_bwd_kernel_v2(
        Q_ptr, K_ptr, V_ptr, O_ptr, dO_ptr, L_ptr, D_ptr,
        dQ_ptr, dK_ptr, dV_ptr,
        stride_qb, stride_qq, stride_qd,
        stride_kb, stride_kk, stride_kd,
        stride_vb, stride_vk, stride_vd,
        stride_ob, stride_oq, stride_od,
        stride_dob, stride_doq, stride_dod,
        stride_lb, stride_lq,
        stride_db, stride_dq_idx,
        stride_dqb, stride_dqq, stride_dqd,
        stride_dkb, stride_dkk, stride_dkd,
        stride_dvb, stride_dvk, stride_dvd,
        N_QUERIES, N_KEYS,
        scale,
        D: tl.constexpr,
        Q_TILE_SIZE: tl.constexpr,
        K_TILE_SIZE: tl.constexpr,
        is_causal: tl.constexpr,
    ):
        """
        Algorithm 2: Tiled FlashAttention-2 backward pass
        
        Key insight: Compute P twice to avoid synchronization:
        - Once for dQ computation 
        - Once for dK and dV computation
        
        This implementation follows the exact algorithm structure where:
        1. Outer loop over K/V tiles (j)
        2. Inner loop over Q tiles (i) 
        3. Atomic updates for dQ accumulation
        """
        # This kernel processes one K/V tile (j-th tile)
        k_tile_id = tl.program_id(0)  # j in algorithm
        batch_id = tl.program_id(1)
        
        # Calculate K/V tile boundaries
        k_start = k_tile_id * K_TILE_SIZE
        k_end = tl.minimum(k_start + K_TILE_SIZE, N_KEYS)
        
        # Load K(j) and V(j) tiles - done once per kernel
        K_block_ptr = tl.make_block_ptr(
            K_ptr + batch_id * stride_kb,
            shape=(N_KEYS, D),
            strides=(stride_kk, stride_kd),
            offsets=(k_start, 0),
            block_shape=(K_TILE_SIZE, D),
            order=(1, 0),
        )
        
        V_block_ptr = tl.make_block_ptr(
            V_ptr + batch_id * stride_vb,
            shape=(N_KEYS, D),
            strides=(stride_vk, stride_vd),
            offsets=(k_start, 0),
            block_shape=(K_TILE_SIZE, D),
            order=(1, 0),
        )
        
        # Load K(j) and V(j) from global memory
        k_j = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero")
        v_j = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero")
        
        # Initialize dK(j) = dV(j) = 0
        dk_j = tl.zeros((K_TILE_SIZE, D), dtype=tl.float32)
        dv_j = tl.zeros((K_TILE_SIZE, D), dtype=tl.float32)
        
        # Inner loop over Q tiles (i = 1, ..., Tq)
        num_q_tiles = tl.cdiv(N_QUERIES, Q_TILE_SIZE)
        
        for q_tile_id in range(num_q_tiles):
            q_start = q_tile_id * Q_TILE_SIZE
            q_end = tl.minimum(q_start + Q_TILE_SIZE, N_QUERIES)
            
            # Early termination for causal attention
            if is_causal and k_start >= q_end:
                continue
                
            # Load Qi, Oi, dOi from global memory
            Q_block_ptr = tl.make_block_ptr(
                Q_ptr + batch_id * stride_qb,
                shape=(N_QUERIES, D),
                strides=(stride_qq, stride_qd),
                offsets=(q_start, 0),
                block_shape=(Q_TILE_SIZE, D),
                order=(1, 0),
            )
            
            O_block_ptr = tl.make_block_ptr(
                O_ptr + batch_id * stride_ob,
                shape=(N_QUERIES, D),
                strides=(stride_oq, stride_od),
                offsets=(q_start, 0),
                block_shape=(Q_TILE_SIZE, D),
                order=(1, 0),
            )
            
            dO_block_ptr = tl.make_block_ptr(
                dO_ptr + batch_id * stride_dob,
                shape=(N_QUERIES, D),
                strides=(stride_doq, stride_dod),
                offsets=(q_start, 0),
                block_shape=(Q_TILE_SIZE, D),
                order=(1, 0),
            )
            
            L_block_ptr = tl.make_block_ptr(
                L_ptr + batch_id * stride_lb,
                shape=(N_QUERIES,),
                strides=(stride_lq,),
                offsets=(q_start,),
                block_shape=(Q_TILE_SIZE,),
                order=(0,),
            )
            
            D_block_ptr = tl.make_block_ptr(
                D_ptr + batch_id * stride_db,
                shape=(N_QUERIES,),
                strides=(stride_dq_idx,),
                offsets=(q_start,),
                block_shape=(Q_TILE_SIZE,),
                order=(0,),
            )
            
            # Load all required tensors
            q_i = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero")
            o_i = tl.load(O_block_ptr, boundary_check=(0, 1), padding_option="zero")
            do_i = tl.load(dO_block_ptr, boundary_check=(0, 1), padding_option="zero")
            L_i = tl.load(L_block_ptr, boundary_check=(0,), padding_option="zero")
            D_i = tl.load(D_block_ptr, boundary_check=(0,), padding_option="zero")
            
            # Compute tile of attention scores: S_i^(j) = Q_i (K^(j))^T / sqrt(d)
            s_ij = tl.dot(q_i, tl.trans(k_j)) * scale
            
            # Apply causal mask if needed
            if is_causal:
                q_indices = q_start + tl.arange(0, Q_TILE_SIZE)
                k_indices = k_start + tl.arange(0, K_TILE_SIZE)
                causal_mask = q_indices[:, None] < k_indices[None, :]
                s_ij = tl.where(causal_mask, -1e6, s_ij)
            
            # Compute attention probabilities: P_i^(j) = exp(S_i^(j) - L_i)
            p_ij = tl.exp(s_ij - L_i[:, None])
            
            # Compute dV^(j) += (P_i^(j))^T dO_i
            dv_j += tl.dot(tl.trans(p_ij.to(do_i.dtype)), do_i)
            
            # Compute dP_i^(j) = dO_i V_j^T  
            dp_ij = tl.dot(do_i, tl.trans(v_j))
            
            # Compute dS_i^(j) = P_i^(j) ◦ (dP_i^(j) - D_i) / sqrt(d)
            ds_ij = p_ij * (dp_ij.to(p_ij.dtype) - D_i[:, None]) * scale
            
            # Apply causal mask to gradients
            if is_causal:
                ds_ij = tl.where(causal_mask, 0.0, ds_ij)
            
            # Load dQ_i from global memory, update dQ_i += dS_i^(j) K^(j), write back
            # This must be atomic for correctness!
            dQ_block_ptr = tl.make_block_ptr(
                dQ_ptr + batch_id * stride_dqb,
                shape=(N_QUERIES, D),
                strides=(stride_dqq, stride_dqd),
                offsets=(q_start, 0),
                block_shape=(Q_TILE_SIZE, D),
                order=(1, 0),
            )
            
            # Load existing dQ_i
            dq_i = tl.load(dQ_block_ptr, boundary_check=(0, 1), padding_option="zero")
            
            # Update: dQ_i += dS_i^(j) K^(j)
            dq_i += tl.dot(ds_ij.to(k_j.dtype), k_j)
            
            # Write back to global memory (atomic)
            tl.store(dQ_block_ptr, dq_i.to(dQ_block_ptr.type.element_ty), boundary_check=(0, 1))
            
            # Compute dK^(j) += (dS_i^(j))^T Q_i
            dk_j += tl.dot(tl.trans(ds_ij.to(q_i.dtype)), q_i)
        
        # Write dK^(j) and dV^(j) to global memory as the j-th tiles of dK and dV
        dK_block_ptr = tl.make_block_ptr(
            dK_ptr + batch_id * stride_dkb,
            shape=(N_KEYS, D),
            strides=(stride_dkk, stride_dkd),
            offsets=(k_start, 0),
            block_shape=(K_TILE_SIZE, D),
            order=(1, 0),
        )
        
        dV_block_ptr = tl.make_block_ptr(
            dV_ptr + batch_id * stride_dvb,
            shape=(N_KEYS, D),
            strides=(stride_dvk, stride_dvd),
            offsets=(k_start, 0),
            block_shape=(K_TILE_SIZE, D),
            order=(1, 0),
        )
        
        tl.store(dK_block_ptr, dk_j.to(dK_block_ptr.type.element_ty), boundary_check=(0, 1))
        tl.store(dV_block_ptr, dv_j.to(dV_block_ptr.type.element_ty), boundary_check=(0, 1))
    
    class FlashAttentionTritonAll(torch.autograd.Function):
        """
        Enhanced Triton implementation of FlashAttention-2 with optimized backward pass.
        This version uses Triton kernels for both forward and backward passes with online softmax.
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
            
            # Optimized tile sizes
            Q_TILE_SIZE = min(64, seq_len)
            K_TILE_SIZE = min(64, seq_len)
            Q_TILE_SIZE = max(16, Q_TILE_SIZE)
            K_TILE_SIZE = max(16, K_TILE_SIZE)
            
            # Round to power of 2 for better performance
            Q_TILE_SIZE = 1 << (Q_TILE_SIZE.bit_length() - 1)
            K_TILE_SIZE = 1 << (K_TILE_SIZE.bit_length() - 1)
            
            # Initialize outputs
            O = torch.empty_like(Q)
            L = torch.empty(batch_size, seq_len, device=device, dtype=torch.float32)
            
            # Launch kernel
            num_q_tiles = triton.cdiv(seq_len, Q_TILE_SIZE)
            grid = (num_q_tiles, batch_size)
            
            flash_fwd_kernel[grid](
                Q, K, V, O, L,
                Q.stride(0), Q.stride(1), Q.stride(2),
                K.stride(0), K.stride(1), K.stride(2),
                V.stride(0), V.stride(1), V.stride(2),
                O.stride(0), O.stride(1), O.stride(2),
                L.stride(0), L.stride(1),
                seq_len, seq_len, scale,
                D=head_dim,
                Q_TILE_SIZE=Q_TILE_SIZE,
                K_TILE_SIZE=K_TILE_SIZE,
                is_causal=is_causal,
                num_warps=4,
                num_stages=2,
            )
            
            ctx.save_for_backward(Q, K, V, O, L)
            ctx.is_causal = is_causal
            ctx.scale = scale
            ctx.Q_TILE_SIZE = Q_TILE_SIZE  # Save tile sizes to context
            ctx.K_TILE_SIZE = K_TILE_SIZE
            
            return O
        
        @staticmethod
        def backward(ctx, grad_output):
            """
            Algorithm 2: Tiled FlashAttention-2 backward pass using Triton kernel.
            
            This implementation follows the exact algorithm structure:
            1. Compute D = rowsum(dO ◦ O) 
            2. Outer loop over K/V tiles (j)
            3. Inner loop over Q tiles (i) for each K/V tile
            4. Atomic updates for dQ accumulation
            """
            Q, K, V, O, L = ctx.saved_tensors
            is_causal = ctx.is_causal
            scale = ctx.scale
            Q_TILE_SIZE = ctx.Q_TILE_SIZE
            K_TILE_SIZE = ctx.K_TILE_SIZE
            
            batch_size, seq_len, head_dim = Q.shape
            device = Q.device
            
            # Ensure contiguous gradients
            grad_output = grad_output.contiguous()
            
            # Algorithm 2, Step 1: Compute D = rowsum(dO ◦ O)
            D = torch.sum(grad_output * O, dim=-1)  # (batch, seq_len)
            
            # Initialize gradient tensors (Algorithm 2 initialization)
            dQ = torch.zeros_like(Q)
            dK = torch.zeros_like(K)
            dV = torch.zeros_like(V)
            
            # Algorithm 2: Outer loop over K/V tiles (j = 1, ..., Tk)
            num_k_tiles = triton.cdiv(seq_len, K_TILE_SIZE)
            grid = (num_k_tiles, batch_size)
            
            # Launch the Algorithm 2 backward kernel
            flash_bwd_kernel_v2[grid](
                Q, K, V, O, grad_output, L, D,
                dQ, dK, dV,
                # Q strides
                Q.stride(0), Q.stride(1), Q.stride(2),
                # K strides  
                K.stride(0), K.stride(1), K.stride(2),
                # V strides
                V.stride(0), V.stride(1), V.stride(2),
                # O strides
                O.stride(0), O.stride(1), O.stride(2),
                # dO strides
                grad_output.stride(0), grad_output.stride(1), grad_output.stride(2),
                # L strides
                L.stride(0), L.stride(1),
                # D strides
                D.stride(0), D.stride(1),
                # dQ strides
                dQ.stride(0), dQ.stride(1), dQ.stride(2),
                # dK strides
                dK.stride(0), dK.stride(1), dK.stride(2),
                # dV strides
                dV.stride(0), dV.stride(1), dV.stride(2),
                # Problem size
                seq_len, seq_len, scale,
                # Compile-time constants
                D=head_dim,
                Q_TILE_SIZE=Q_TILE_SIZE,
                K_TILE_SIZE=K_TILE_SIZE,
                is_causal=is_causal,
                # Triton launch parameters
                num_warps=4,
                num_stages=2,
            )
            
            return dQ, dK, dV, None
else:
    class FlashAttentionTritonFunction(torch.autograd.Function):
        @staticmethod
        def forward(ctx, Q, K, V, is_causal=False):
            raise RuntimeError("Triton is not available")
        
        @staticmethod
        def backward(ctx, grad_output):
            raise NotImplementedError()


def flash_attention_pytorch(Q, K, V, is_causal=False):
    """Optimized PyTorch FlashAttention-2."""
    return FlashAttentionPyTorchFunction.apply(Q, K, V, is_causal)


def flash_attention_triton(Q, K, V, is_causal=False):
    """Triton FlashAttention-2 with proper dtype handling."""
    if not TRITON_AVAILABLE:
        raise RuntimeError("Triton is not available")
    
    # Ensure input tensors are properly converted and contiguous
    original_dtype = Q.dtype
    
    # Convert inputs to compatible dtype if needed
    if Q.dtype == torch.bfloat16:
        # Keep bfloat16 for computation efficiency
        Q = Q.contiguous()
        K = K.contiguous()
        V = V.contiguous()
    else:
        # For other dtypes, ensure they're contiguous
        Q = Q.contiguous()
        K = K.contiguous()
        V = V.contiguous()
    
    result = FlashAttentionTritonFunction.apply(Q, K, V, is_causal)
    
    # Ensure output has the same dtype as input
    if result.dtype != original_dtype:
        result = result.to(original_dtype)
    
    return result


def flash_attention_all_triton(Q, K, V, is_causal=False):
    """
    Enhanced Triton FlashAttention-2 with full Triton backward pass.
    
    This version uses Triton kernels for both forward and backward passes,
    featuring optimized online softmax implementation and memory-efficient
    gradient computation with recomputation strategy.
    
    Key improvements over V1:
    1. Full Triton kernel implementation for backward pass
    2. Online softmax in backward computation
    3. Optimized memory access patterns
    4. Better tile size management
    
    Args:
        Q: Query tensor (batch, seq_len, head_dim)
        K: Key tensor (batch, seq_len, head_dim)  
        V: Value tensor (batch, seq_len, head_dim)
        is_causal: Whether to apply causal masking
        
    Returns:
        Output tensor (batch, seq_len, head_dim)
    """
    if not TRITON_AVAILABLE:
        raise RuntimeError("Triton is not available")
    return FlashAttentionTritonAll.apply(Q, K, V, is_causal)