"""
SPARSA-LM Core Layers
Implements fundamental building blocks for the AutoRegressive model
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.

    More efficient than LayerNorm as it doesn't require computing mean.
    Used in modern LLMs like LLaMA and Mistral.
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply RMS normalization."""
        input_dtype = x.dtype
        x = x.to(torch.float32)
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x.to(input_dtype)


class RotaryEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE).

    Encodes positional information through rotation of query and key vectors.
    Enables better length generalization and relative position awareness.
    """

    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 2048,
        base: float = 10000.0,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base

        # Compute inverse frequencies
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, device=device).float() / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Build cosine and sine cache
        self._set_cos_sin_cache(max_position_embeddings, device)

    def _set_cos_sin_cache(self, seq_len: int, device: Optional[torch.device] = None):
        """Precompute cosine and sine values for rotary embeddings."""
        self.max_seq_len_cached = seq_len
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, x: torch.Tensor, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return cosine and sine embeddings for the given sequence length."""
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len, x.device)
        return (
            self.cos_cached[:seq_len].to(x.dtype),
            self.sin_cached[:seq_len].to(x.dtype),
        )


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary position embeddings to query and key tensors."""
    if position_ids is not None:
        cos = cos[position_ids].unsqueeze(1)
        sin = sin[position_ids].unsqueeze(1)
    else:
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class KVCache:
    """
    Key-Value cache for efficient autoregressive generation.

    Stores computed key and value tensors to avoid recomputation
    during incremental decoding.
    """

    def __init__(
        self,
        batch_size: int,
        max_seq_len: int,
        num_kv_heads: int,
        head_dim: int,
        device: torch.device,
        dtype: torch.dtype = torch.float16,
    ):
        self.max_seq_len = max_seq_len
        self.current_len = 0

        # Pre-allocate cache
        cache_shape = (batch_size, num_kv_heads, max_seq_len, head_dim)
        self.k_cache = torch.zeros(cache_shape, device=device, dtype=dtype)
        self.v_cache = torch.zeros(cache_shape, device=device, dtype=dtype)

    def update(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        layer_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Update cache with new key-value pairs and return full cache."""
        seq_len = k.shape[2]
        self.k_cache[:, :, self.current_len : self.current_len + seq_len] = k
        self.v_cache[:, :, self.current_len : self.current_len + seq_len] = v

        return (
            self.k_cache[:, :, : self.current_len + seq_len],
            self.v_cache[:, :, : self.current_len + seq_len],
        )

    def increment(self, length: int = 1):
        """Increment the current cache length."""
        self.current_len += length

    def reset(self):
        """Reset the cache."""
        self.current_len = 0
        self.k_cache.zero_()
        self.v_cache.zero_()


class GroupedQueryAttention(nn.Module):
    """
    Grouped Query Attention (GQA).

    Reduces memory and computation by using fewer key-value heads
    than query heads. Each group of query heads shares the same
    key-value head.
    """

    def __init__(self, config: ModelConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.num_kv_groups = config.num_kv_groups

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_bias)

        self.rotary_emb = RotaryEmbedding(
            self.head_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta,
        )

        self.sliding_window = config.sliding_window
        self.attention_dropout = config.attention_dropout

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Forward pass for grouped query attention.

        Args:
            hidden_states: Input tensor of shape (batch, seq_len, hidden_size)
            attention_mask: Optional attention mask
            position_ids: Optional position IDs
            past_key_value: Optional cached key-value pairs
            use_cache: Whether to return updated cache

        Returns:
            Tuple of (output, optional cached key-value pairs)
        """
        batch_size, seq_len, _ = hidden_states.shape

        # Project to queries, keys, values
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # Apply rotary embeddings
        cos, sin = self.rotary_emb(q, seq_len)
        q, k = apply_rotary_pos_emb(q, k, cos, sin, position_ids)

        # Handle KV cache
        if past_key_value is not None:
            k = torch.cat([past_key_value[0], k], dim=2)
            v = torch.cat([past_key_value[1], v], dim=2)

        past_key_value = (k, v) if use_cache else None

        # Repeat KV heads for GQA
        k = k.repeat_interleave(self.num_kv_groups, dim=1)
        v = v.repeat_interleave(self.num_kv_groups, dim=1)

        # Compute attention
        attn_output = self._attention(q, k, v, attention_mask)

        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        attn_output = self.o_proj(attn_output)

        return attn_output, past_key_value

    def _attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute scaled dot-product attention."""
        # Try Flash Attention if available
        if self.config.use_flash_attention and hasattr(F, 'scaled_dot_product_attention'):
            return F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attention_mask,
                dropout_p=self.attention_dropout if self.training else 0.0,
                is_causal=attention_mask is None,
            )

        # Standard attention
        scale = 1.0 / math.sqrt(self.head_dim)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale

        # Apply sliding window if configured
        if self.sliding_window is not None:
            seq_len = q.shape[2]
            kv_len = k.shape[2]
            indices = torch.arange(seq_len, device=q.device)[:, None]
            kv_indices = torch.arange(kv_len, device=q.device)[None, :]
            window_mask = (indices - kv_indices).abs() <= self.sliding_window
            attn_weights = attn_weights.masked_fill(~window_mask, float('-inf'))

        # Apply causal mask
        if attention_mask is None:
            seq_len = q.shape[2]
            kv_len = k.shape[2]
            causal_mask = torch.triu(
                torch.ones(seq_len, kv_len, dtype=torch.bool, device=q.device),
                diagonal=kv_len - seq_len + 1
            )
            attn_weights = attn_weights.masked_fill(causal_mask, float('-inf'))
        else:
            attn_weights = attn_weights + attention_mask

        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
        attn_weights = F.dropout(attn_weights, p=self.attention_dropout, training=self.training)

        return torch.matmul(attn_weights, v)


class SwiGLU(nn.Module):
    """
    SwiGLU Activation Function.

    Combines Swish activation with Gated Linear Unit for improved
    performance in transformer feed-forward networks.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size

        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply SwiGLU activation."""
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class TransformerBlock(nn.Module):
    """
    Single Transformer Block with Pre-Norm architecture.

    Consists of:
    1. RMS Normalization
    2. Grouped Query Attention
    3. RMS Normalization
    4. SwiGLU Feed-Forward Network
    """

    def __init__(self, config: ModelConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx

        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = GroupedQueryAttention(config, layer_idx)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = SwiGLU(config)

        self.hidden_dropout = config.hidden_dropout

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Forward pass for transformer block.

        Args:
            hidden_states: Input tensor
            attention_mask: Optional attention mask
            position_ids: Optional position IDs
            past_key_value: Optional cached key-value pairs
            use_cache: Whether to return updated cache

        Returns:
            Tuple of (output, optional cached key-value pairs)
        """
        # Self-attention with residual connection
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, present_key_value = self.self_attn(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            use_cache=use_cache,
        )
        hidden_states = F.dropout(hidden_states, p=self.hidden_dropout, training=self.training)
        hidden_states = residual + hidden_states

        # Feed-forward with residual connection
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = F.dropout(hidden_states, p=self.hidden_dropout, training=self.training)
        hidden_states = residual + hidden_states

        return hidden_states, present_key_value
