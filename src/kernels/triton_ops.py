"""
SPARSA-LM OpenAI Triton Custom Kernels
High-performance GPU kernels for transformer operations
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple

# Check if Triton is available
try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False


# =============================================================================
# TRITON KERNEL: RMSNorm
# =============================================================================

if TRITON_AVAILABLE:
    @triton.jit
    def _rmsnorm_fwd_kernel(
        X,  # Input tensor pointer
        Y,  # Output tensor pointer
        W,  # Weight tensor pointer
        stride_x,  # Stride for batch dimension
        N,  # Hidden dimension
        eps,  # Epsilon for numerical stability
        BLOCK_SIZE: tl.constexpr,
    ):
        """
        Forward kernel for RMS Layer Normalization.

        RMSNorm(x) = x / sqrt(mean(x^2) + eps) * weight
        """
        # Get row index
        row = tl.program_id(0)

        # Compute pointers
        X += row * stride_x
        Y += row * stride_x

        # Load input and compute mean of squares
        _sum = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
        for off in range(0, N, BLOCK_SIZE):
            cols = off + tl.arange(0, BLOCK_SIZE)
            mask = cols < N
            x = tl.load(X + cols, mask=mask, other=0.0).to(tl.float32)
            _sum += x * x

        # Compute RMS
        mean_sq = tl.sum(_sum, axis=0) / N
        rms = tl.rsqrt(mean_sq + eps)

        # Normalize and scale
        for off in range(0, N, BLOCK_SIZE):
            cols = off + tl.arange(0, BLOCK_SIZE)
            mask = cols < N
            x = tl.load(X + cols, mask=mask, other=0.0).to(tl.float32)
            w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
            y = x * rms * w
            tl.store(Y + cols, y, mask=mask)


    @triton.jit
    def _rmsnorm_bwd_kernel(
        DY,  # Gradient of output
        X,   # Input tensor
        W,   # Weight tensor
        DX,  # Gradient of input (output)
        DW,  # Gradient of weight (output)
        stride,
        N,
        eps,
        BLOCK_SIZE: tl.constexpr,
    ):
        """Backward kernel for RMSNorm."""
        row = tl.program_id(0)

        DY += row * stride
        X += row * stride
        DX += row * stride

        # Compute forward values
        _sum = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
        for off in range(0, N, BLOCK_SIZE):
            cols = off + tl.arange(0, BLOCK_SIZE)
            mask = cols < N
            x = tl.load(X + cols, mask=mask, other=0.0).to(tl.float32)
            _sum += x * x

        mean_sq = tl.sum(_sum, axis=0) / N
        rms = tl.rsqrt(mean_sq + eps)

        # Compute gradients
        _dot = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
        for off in range(0, N, BLOCK_SIZE):
            cols = off + tl.arange(0, BLOCK_SIZE)
            mask = cols < N
            x = tl.load(X + cols, mask=mask, other=0.0).to(tl.float32)
            dy = tl.load(DY + cols, mask=mask, other=0.0).to(tl.float32)
            w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
            _dot += dy * x * w

        dot = tl.sum(_dot, axis=0)

        for off in range(0, N, BLOCK_SIZE):
            cols = off + tl.arange(0, BLOCK_SIZE)
            mask = cols < N
            x = tl.load(X + cols, mask=mask, other=0.0).to(tl.float32)
            dy = tl.load(DY + cols, mask=mask, other=0.0).to(tl.float32)
            w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)

            # dx = w * rms * (dy - x * dot / N * rms^2)
            dx = w * rms * (dy - x * dot * rms * rms / N)
            tl.store(DX + cols, dx, mask=mask)


def triton_rmsnorm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    RMSNorm using Triton kernel.

    Args:
        x: Input tensor of shape (batch, seq_len, hidden_size)
        weight: Weight tensor of shape (hidden_size,)
        eps: Epsilon for numerical stability

    Returns:
        Normalized tensor
    """
    if not TRITON_AVAILABLE:
        # Fallback to PyTorch
        variance = x.pow(2).mean(-1, keepdim=True)
        x_normed = x * torch.rsqrt(variance + eps)
        return x_normed * weight

    # Reshape for kernel
    original_shape = x.shape
    x = x.view(-1, x.shape[-1])
    y = torch.empty_like(x)

    # Kernel parameters
    N = x.shape[-1]
    BLOCK_SIZE = triton.next_power_of_2(N)
    BLOCK_SIZE = min(BLOCK_SIZE, 4096)

    # Launch kernel
    num_rows = x.shape[0]
    _rmsnorm_fwd_kernel[(num_rows,)](
        x, y, weight,
        x.stride(0),
        N, eps,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return y.view(original_shape)


# =============================================================================
# TRITON KERNEL: SwiGLU Activation
# =============================================================================

if TRITON_AVAILABLE:
    @triton.jit
    def _swiglu_fwd_kernel(
        X,  # Input tensor (gate)
        Y,  # Input tensor (up)
        OUT,  # Output tensor
        N,  # Number of elements
        BLOCK_SIZE: tl.constexpr,
    ):
        """
        Forward kernel for SwiGLU: silu(gate) * up

        SiLU(x) = x * sigmoid(x)
        """
        pid = tl.program_id(0)
        offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < N

        x = tl.load(X + offs, mask=mask, other=0.0).to(tl.float32)
        y = tl.load(Y + offs, mask=mask, other=0.0).to(tl.float32)

        # SiLU activation: x * sigmoid(x)
        sigmoid_x = tl.sigmoid(x)
        silu_x = x * sigmoid_x

        # Output: silu(gate) * up
        out = silu_x * y
        tl.store(OUT + offs, out, mask=mask)


    @triton.jit
    def _swiglu_bwd_kernel(
        DOUT,  # Gradient of output
        X,     # Gate input
        Y,     # Up input
        DX,    # Gradient of gate (output)
        DY,    # Gradient of up (output)
        N,
        BLOCK_SIZE: tl.constexpr,
    ):
        """Backward kernel for SwiGLU."""
        pid = tl.program_id(0)
        offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < N

        dout = tl.load(DOUT + offs, mask=mask, other=0.0).to(tl.float32)
        x = tl.load(X + offs, mask=mask, other=0.0).to(tl.float32)
        y = tl.load(Y + offs, mask=mask, other=0.0).to(tl.float32)

        # Forward: out = silu(x) * y = x * sigmoid(x) * y
        sigmoid_x = tl.sigmoid(x)
        silu_x = x * sigmoid_x

        # Gradient of y: dout * silu(x)
        dy = dout * silu_x
        tl.store(DY + offs, dy, mask=mask)

        # Gradient of x: dout * y * (sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x)))
        #              = dout * y * sigmoid(x) * (1 + x * (1 - sigmoid(x)))
        dsilu = sigmoid_x * (1.0 + x * (1.0 - sigmoid_x))
        dx = dout * y * dsilu
        tl.store(DX + offs, dx, mask=mask)


def triton_swiglu(
    gate: torch.Tensor,
    up: torch.Tensor,
) -> torch.Tensor:
    """
    SwiGLU activation using Triton kernel.

    SwiGLU(gate, up) = SiLU(gate) * up
                     = gate * sigmoid(gate) * up

    Args:
        gate: Gate projection output
        up: Up projection output

    Returns:
        Activated tensor
    """
    if not TRITON_AVAILABLE:
        # Fallback to PyTorch
        return torch.nn.functional.silu(gate) * up

    assert gate.shape == up.shape
    output = torch.empty_like(gate)

    N = gate.numel()
    BLOCK_SIZE = 1024

    grid = (triton.cdiv(N, BLOCK_SIZE),)
    _swiglu_fwd_kernel[grid](
        gate.view(-1), up.view(-1), output.view(-1),
        N,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return output


# =============================================================================
# TRITON KERNEL: Rotary Position Embedding
# =============================================================================

if TRITON_AVAILABLE:
    @triton.jit
    def _rotary_embedding_kernel(
        Q,      # Query tensor
        K,      # Key tensor
        COS,    # Cosine embeddings
        SIN,    # Sine embeddings
        Q_OUT,  # Output query
        K_OUT,  # Output key
        seq_len,
        head_dim,
        stride_qb, stride_qh, stride_qs, stride_qd,
        stride_kb, stride_kh, stride_ks, stride_kd,
        BLOCK_SIZE: tl.constexpr,
    ):
        """
        Rotary position embedding kernel.

        Applies rotation to query and key tensors.
        """
        # Get indices
        batch = tl.program_id(0)
        head = tl.program_id(1)
        seq = tl.program_id(2)

        # Compute base offsets
        q_base = batch * stride_qb + head * stride_qh + seq * stride_qs
        k_base = batch * stride_kb + head * stride_kh + seq * stride_ks

        # Load cos and sin for this position
        half_dim = head_dim // 2

        for d in range(0, half_dim, BLOCK_SIZE):
            cols = d + tl.arange(0, BLOCK_SIZE)
            mask = cols < half_dim

            # Load Q values
            q1 = tl.load(Q + q_base + cols, mask=mask, other=0.0)
            q2 = tl.load(Q + q_base + cols + half_dim, mask=mask, other=0.0)

            # Load K values
            k1 = tl.load(K + k_base + cols, mask=mask, other=0.0)
            k2 = tl.load(K + k_base + cols + half_dim, mask=mask, other=0.0)

            # Load cos and sin
            cos = tl.load(COS + seq * half_dim + cols, mask=mask, other=0.0)
            sin = tl.load(SIN + seq * half_dim + cols, mask=mask, other=0.0)

            # Apply rotation to Q
            q1_rot = q1 * cos - q2 * sin
            q2_rot = q1 * sin + q2 * cos
            tl.store(Q_OUT + q_base + cols, q1_rot, mask=mask)
            tl.store(Q_OUT + q_base + cols + half_dim, q2_rot, mask=mask)

            # Apply rotation to K
            k1_rot = k1 * cos - k2 * sin
            k2_rot = k1 * sin + k2 * cos
            tl.store(K_OUT + k_base + cols, k1_rot, mask=mask)
            tl.store(K_OUT + k_base + cols + half_dim, k2_rot, mask=mask)


def triton_rotary_embedding(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary position embeddings using Triton kernel.

    Args:
        q: Query tensor (batch, heads, seq_len, head_dim)
        k: Key tensor (batch, heads, seq_len, head_dim)
        cos: Cosine embeddings (seq_len, head_dim // 2)
        sin: Sine embeddings (seq_len, head_dim // 2)

    Returns:
        Tuple of rotated (q, k)
    """
    if not TRITON_AVAILABLE:
        # Fallback to PyTorch
        q1, q2 = q.chunk(2, dim=-1)
        k1, k2 = k.chunk(2, dim=-1)

        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)

        q_rot = torch.cat([q1 * cos - q2 * sin, q1 * sin + q2 * cos], dim=-1)
        k_rot = torch.cat([k1 * cos - k2 * sin, k1 * sin + k2 * cos], dim=-1)

        return q_rot, k_rot

    batch, heads, seq_len, head_dim = q.shape
    q_out = torch.empty_like(q)
    k_out = torch.empty_like(k)

    BLOCK_SIZE = min(32, head_dim // 2)

    grid = (batch, heads, seq_len)
    _rotary_embedding_kernel[grid](
        q, k, cos, sin, q_out, k_out,
        seq_len, head_dim,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return q_out, k_out


# =============================================================================
# TRITON KERNEL: Fused Attention
# =============================================================================

if TRITON_AVAILABLE:
    @triton.jit
    def _fused_attention_kernel(
        Q, K, V, O,
        stride_qb, stride_qh, stride_qm, stride_qk,
        stride_kb, stride_kh, stride_kn, stride_kk,
        stride_vb, stride_vh, stride_vn, stride_vk,
        stride_ob, stride_oh, stride_om, stride_ok,
        N_CTX,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        scale,
    ):
        """
        Fused attention kernel (simplified Flash Attention).

        Computes: softmax(Q @ K.T / sqrt(d)) @ V
        """
        # Get program ID
        off_b = tl.program_id(0)
        off_h = tl.program_id(1)
        off_m = tl.program_id(2)

        # Compute offsets
        offs_m = off_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)

        # Initialize pointers
        q_ptrs = Q + off_b * stride_qb + off_h * stride_qh + (offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qk)
        k_ptrs = K + off_b * stride_kb + off_h * stride_kh + (offs_n[:, None] * stride_kn + offs_k[None, :] * stride_kk)
        v_ptrs = V + off_b * stride_vb + off_h * stride_vh + (offs_n[:, None] * stride_vn + offs_k[None, :] * stride_vk)

        # Load Q block
        q = tl.load(q_ptrs, mask=(offs_m[:, None] < N_CTX) & (offs_k[None, :] < BLOCK_K), other=0.0)

        # Initialize accumulators
        m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float('inf')
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
        acc = tl.zeros([BLOCK_M, BLOCK_K], dtype=tl.float32)

        # Loop over K, V blocks
        for start_n in range(0, N_CTX, BLOCK_N):
            start_n = tl.multiple_of(start_n, BLOCK_N)
            offs_n_curr = start_n + tl.arange(0, BLOCK_N)

            # Load K block
            k = tl.load(k_ptrs, mask=(offs_n_curr[:, None] < N_CTX) & (offs_k[None, :] < BLOCK_K), other=0.0)

            # Compute QK^T
            qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
            qk += tl.dot(q, tl.trans(k)) * scale

            # Apply causal mask
            causal_mask = offs_m[:, None] >= offs_n_curr[None, :]
            qk = tl.where(causal_mask, qk, float('-inf'))

            # Online softmax
            m_ij = tl.max(qk, 1)
            m_i_new = tl.maximum(m_i, m_ij)
            alpha = tl.exp(m_i - m_i_new)
            p = tl.exp(qk - m_i_new[:, None])

            # Update accumulators
            l_i = alpha * l_i + tl.sum(p, 1)
            acc = acc * alpha[:, None]

            # Load V and accumulate
            v = tl.load(v_ptrs, mask=(offs_n_curr[:, None] < N_CTX) & (offs_k[None, :] < BLOCK_K), other=0.0)
            acc += tl.dot(p.to(v.dtype), v)

            m_i = m_i_new
            k_ptrs += BLOCK_N * stride_kn
            v_ptrs += BLOCK_N * stride_vn

        # Finalize output
        acc = acc / l_i[:, None]

        # Store output
        o_ptrs = O + off_b * stride_ob + off_h * stride_oh + (offs_m[:, None] * stride_om + offs_k[None, :] * stride_ok)
        tl.store(o_ptrs, acc, mask=(offs_m[:, None] < N_CTX) & (offs_k[None, :] < BLOCK_K))


def triton_fused_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: Optional[float] = None,
) -> torch.Tensor:
    """
    Fused attention using Triton kernel.

    Args:
        q: Query tensor (batch, heads, seq_len, head_dim)
        k: Key tensor (batch, heads, seq_len, head_dim)
        v: Value tensor (batch, heads, seq_len, head_dim)
        scale: Attention scale (default: 1/sqrt(head_dim))

    Returns:
        Attention output
    """
    if not TRITON_AVAILABLE:
        # Fallback to PyTorch
        if scale is None:
            scale = q.shape[-1] ** -0.5

        attn = torch.matmul(q, k.transpose(-2, -1)) * scale

        # Causal mask
        seq_len = q.shape[2]
        mask = torch.triu(torch.ones(seq_len, seq_len, device=q.device, dtype=torch.bool), diagonal=1)
        attn = attn.masked_fill(mask, float('-inf'))

        attn = torch.softmax(attn, dim=-1)
        return torch.matmul(attn, v)

    batch, heads, seq_len, head_dim = q.shape
    if scale is None:
        scale = head_dim ** -0.5

    output = torch.empty_like(q)

    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = head_dim

    grid = (batch, heads, triton.cdiv(seq_len, BLOCK_M))

    _fused_attention_kernel[grid](
        q, k, v, output,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        output.stride(0), output.stride(1), output.stride(2), output.stride(3),
        seq_len,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        scale=scale,
    )

    return output


# =============================================================================
# TORCH.NN MODULES WITH TRITON BACKEND
# =============================================================================

class TritonRMSNorm(nn.Module):
    """RMSNorm with Triton acceleration."""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_rmsnorm(x, self.weight, self.eps)


class TritonSwiGLU(nn.Module):
    """SwiGLU with Triton acceleration."""

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        activated = triton_swiglu(gate, up)
        return self.down_proj(activated)


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def check_triton_available() -> bool:
    """Check if Triton is available and working."""
    if not TRITON_AVAILABLE:
        return False

    try:
        # Test with a small tensor
        x = torch.randn(2, 4, 8, device='cuda')
        w = torch.ones(8, device='cuda')
        _ = triton_rmsnorm(x, w)
        return True
    except Exception:
        return False
