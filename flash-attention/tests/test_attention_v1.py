import pytest
import torch
from einops import einsum, rearrange

from .adapters import get_flashattention_v1_autograd_function


def _attention_and_lse_4d(q, k, v, is_causal=False):
    """
    Reference attention implementation for 4D tensors (batch, heads, seq_len, head_dim).
    This is used as ground truth for FlashAttention-1 testing.
    """
    batch_size, n_heads, n_queries, d = q.shape
    _, _, n_keys, _ = k.shape
    scale = 1 / (d ** 0.5)
    
    # Compute attention scores: S = Q @ K^T * scale
    S = torch.einsum('bhqd,bhkd->bhqk', q, k) * scale
    
    if is_causal:
        # Apply causal mask
        mask = torch.triu(torch.ones(n_queries, n_keys, device=S.device), diagonal=1).bool()
        S = S.masked_fill(mask, float('-inf'))
    
    # Compute softmax
    P = torch.softmax(S, dim=-1)
    
    # Compute output: O = P @ V
    o = torch.einsum('bhqk,bhkd->bhqd', P, v)
    
    # Compute log-sum-exp for each query position (always in float32 for numerical stability)
    L = torch.logsumexp(S, dim=-1).to(torch.float32)
    
    return o, L


def _make_attn_inputs_4d(device=None, dtype=torch.float32):
    """
    Create 4D attention inputs for FlashAttention-1 testing.
    Shape: (batch, heads, seq_len, head_dim)
    """
    torch.random.manual_seed(42)  # Different seed from v2 tests for independence
    batch_size = 2
    n_heads = 4
    n_queries = 64
    n_keys = 64
    head_dim = 32
    
    q = torch.randn(batch_size, n_heads, n_queries, head_dim, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn(batch_size, n_heads, n_keys, head_dim, device=device, dtype=dtype, requires_grad=True)
    v = torch.randn(batch_size, n_heads, n_keys, head_dim, device=device, dtype=dtype, requires_grad=True)
    do = torch.randn(batch_size, n_heads, n_queries, head_dim, device=device, dtype=dtype)

    return q, k, v, do


def _test_flash_v1_forward_pass(impl, device="cuda", is_causal=False, dtype=torch.float32):
    """Test FlashAttention-1 forward pass against reference implementation."""
    q, k, v, _do = _make_attn_inputs_4d(device, dtype)
    
    # Run FlashAttention-1
    o = impl(q, k, v, is_causal)

    # Extract L from the saved tensors
    assert o.grad_fn.saved_tensors is not None, "No saved tensors found in the output tensor. Make sure your autograd forward is saving them using ctx.save_for_backward."
    
    # For FlashAttention-1, L should have shape (batch, heads, seq_len)
    expected_l_shape = (q.shape[0], q.shape[1], q.shape[2])
    maybe_ls = [t for t in o.grad_fn.saved_tensors if t.shape == expected_l_shape]

    assert len(maybe_ls) == 1, f"Expected one tensor of shape {expected_l_shape} in saved tensors, but found {len(maybe_ls)}. The tests require you to save exactly one tensor of this shape, corresponding to the log-sum-exp of the attention scores."
    l = maybe_ls[0]

    # Compute reference
    o_ref, l_ref = _attention_and_lse_4d(q, k, v, is_causal)

    # Compare outputs
    torch.testing.assert_close(o, o_ref, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(l, l_ref, rtol=1e-2, atol=1e-2)


def _test_flash_v1_backward_pass(impl, device="cuda", is_causal=False, dtype=torch.float32):
    """Test FlashAttention-1 backward pass against reference implementation."""
    # Compute reference gradients
    q_ref, k_ref, v_ref, do = _make_attn_inputs_4d(device, dtype)
    o_ref, _ = _attention_and_lse_4d(q_ref, k_ref, v_ref, is_causal)
    o_ref.backward(do)
    dq_ref, dk_ref, dv_ref = q_ref.grad, k_ref.grad, v_ref.grad

    # Compute FlashAttention-1 gradients
    q, k, v, do = _make_attn_inputs_4d(device, dtype)
    o = impl(q, k, v, is_causal)
    o.backward(do)

    # Compare gradients with appropriate tolerances for different dtypes
    if dtype == torch.bfloat16:
        # More relaxed tolerances for bfloat16 due to lower precision
        rtol, atol = 5e-2, 5e-2
    else:
        rtol, atol = 1e-2, 1e-2

    torch.testing.assert_close(q.grad, dq_ref, rtol=rtol, atol=atol)
    torch.testing.assert_close(k.grad, dk_ref, rtol=rtol, atol=atol)
    torch.testing.assert_close(v.grad, dv_ref, rtol=rtol, atol=atol)


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="A GPU must be available to run Triton kernels",
)
@pytest.mark.parametrize("is_causal", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_flash_v1_forward_pass_triton(is_causal, dtype):
    """Test FlashAttention-1 forward pass with different causal settings and dtypes."""
    _test_flash_v1_forward_pass(
        get_flashattention_v1_autograd_function().apply, 
        device="cuda", 
        is_causal=is_causal,
        dtype=dtype
    )


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="A GPU must be available to run Triton kernels",
)
@pytest.mark.parametrize("is_causal", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_flash_v1_backward_pass_triton(is_causal, dtype):
    """Test FlashAttention-1 backward pass with different causal settings and dtypes."""
    _test_flash_v1_backward_pass(
        get_flashattention_v1_autograd_function().apply, 
        device="cuda", 
        is_causal=is_causal,
        dtype=dtype
    )


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="A GPU must be available to run Triton kernels",
)
@pytest.mark.parametrize("seq_len", [32, 64, 128, 256])
def test_flash_v1_different_sequence_lengths(seq_len):
    """Test FlashAttention-1 with different sequence lengths."""
    torch.random.manual_seed(42)
    batch_size = 1
    n_heads = 2
    head_dim = 32
    device = "cuda"
    
    q = torch.randn(batch_size, n_heads, seq_len, head_dim, device=device, requires_grad=True)
    k = torch.randn(batch_size, n_heads, seq_len, head_dim, device=device, requires_grad=True)
    v = torch.randn(batch_size, n_heads, seq_len, head_dim, device=device, requires_grad=True)
    do = torch.randn(batch_size, n_heads, seq_len, head_dim, device=device)

    # Test forward pass with adjusted tolerances for larger sequences
    impl = get_flashattention_v1_autograd_function().apply
    o = impl(q, k, v, False)
    o_ref, _ = _attention_and_lse_4d(q, k, v, False)
    
    # Use more relaxed tolerances for larger sequences due to accumulated numerical errors
    if seq_len >= 128:
        rtol, atol = 5e-2, 5e-2
    else:
        rtol, atol = 1e-2, 1e-2
    
    torch.testing.assert_close(o, o_ref, rtol=rtol, atol=atol)

    # Test backward pass
    o.backward(do)
    # Just check that gradients are computed without error
    assert q.grad is not None
    assert k.grad is not None
    assert v.grad is not None


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="A GPU must be available to run Triton kernels",
)
def test_flash_v1_numerical_stability():
    """Test FlashAttention-1 numerical stability with extreme values."""
    torch.random.manual_seed(42)
    batch_size = 1
    n_heads = 1
    seq_len = 32
    head_dim = 16
    device = "cuda"
    
    # Test with large values that could cause overflow
    q = torch.randn(batch_size, n_heads, seq_len, head_dim, device=device) * 10
    k = torch.randn(batch_size, n_heads, seq_len, head_dim, device=device) * 10
    v = torch.randn(batch_size, n_heads, seq_len, head_dim, device=device)
    
    q.requires_grad_(True)
    k.requires_grad_(True)
    v.requires_grad_(True)
    
    impl = get_flashattention_v1_autograd_function().apply
    
    # Should not produce NaN or Inf
    o = impl(q, k, v, False)
    assert not torch.isnan(o).any(), "Output contains NaN values"
    assert not torch.isinf(o).any(), "Output contains Inf values"
    
    # Test backward pass
    do = torch.randn_like(o)
    o.backward(do)
    
    assert not torch.isnan(q.grad).any(), "Query gradients contain NaN values"
    assert not torch.isnan(k.grad).any(), "Key gradients contain NaN values"
    assert not torch.isnan(v.grad).any(), "Value gradients contain NaN values"


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="A GPU must be available to run Triton kernels",
)
def test_flash_v1_gradient_consistency():
    """Test that FlashAttention-1 gradients are consistent across multiple runs."""
    torch.random.manual_seed(42)
    
    def run_backward():
        q, k, v, do = _make_attn_inputs_4d("cuda")
        impl = get_flashattention_v1_autograd_function().apply
        o = impl(q, k, v, True)  # Use causal attention
        o.backward(do)
        return q.grad.clone(), k.grad.clone(), v.grad.clone()
    
    # Run multiple times
    dq1, dk1, dv1 = run_backward()
    dq2, dk2, dv2 = run_backward()
    
    # Results should be identical
    torch.testing.assert_close(dq1, dq2, rtol=0, atol=0)
    torch.testing.assert_close(dk1, dk2, rtol=0, atol=0)
    torch.testing.assert_close(dv1, dv2, rtol=0, atol=0)