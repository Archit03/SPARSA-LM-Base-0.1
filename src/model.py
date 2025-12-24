"""
SPARSA-LM 360M - Modern LLaMA-Style Transformer Architecture

Features:
- RMSNorm (instead of LayerNorm)
- SwiGLU activation (instead of GELU/ReLU)
- Rotary Position Embeddings (RoPE)
- Grouped Query Attention (GQA)
- Flash Attention 2 support
- Sliding Window Attention
- KV-Cache for efficient inference

This code belongs to EllanorAI and is licensed under the EllanorAI Proprietary License.
"""

import math
import logging
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict, Any, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

try:
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.bert_padding import unpad_input, pad_input
    FLASH_ATTN_AVAILABLE = True
except ImportError:
    FLASH_ATTN_AVAILABLE = False
    logging.warning("Flash Attention not available. Using standard attention.")

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False


@dataclass
class SPARSAConfig:
    """Configuration for SPARSA-LM 360M Model."""

    # Model dimensions
    vocab_size: int = 32000
    hidden_dim: int = 1024
    num_layers: int = 28
    num_heads: int = 16
    num_kv_heads: int = 8  # For GQA
    ff_dim: int = 4096
    max_seq_len: int = 2048

    # Normalization
    rms_norm_eps: float = 1e-6

    # Attention
    use_flash_attention: bool = True
    use_sliding_window: bool = True
    sliding_window_size: int = 512
    attention_dropout: float = 0.0
    attention_bias: bool = False

    # Position embeddings
    rope_theta: float = 10000.0
    rope_scaling: Optional[Dict[str, Any]] = None

    # Activation
    use_swiglu: bool = True
    hidden_act: str = "silu"

    # Training
    dropout: float = 0.0
    use_checkpointing: bool = True
    tie_embeddings: bool = False
    initializer_range: float = 0.02

    # Inference
    use_cache: bool = True

    # Device
    device: Optional[torch.device] = None
    dtype: torch.dtype = torch.bfloat16

    # Special tokens
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2

    def __post_init__(self):
        assert self.hidden_dim % self.num_heads == 0, "hidden_dim must be divisible by num_heads"
        assert self.num_heads % self.num_kv_heads == 0, "num_heads must be divisible by num_kv_heads"
        self.head_dim = self.hidden_dim // self.num_heads
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        if self.device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        x = x.float()
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return (self.weight * x).to(input_dtype)


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE) with optional scaling."""

    def __init__(
        self,
        dim: int,
        max_seq_len: int = 2048,
        base: float = 10000.0,
        scaling_factor: float = 1.0,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        self.scaling_factor = scaling_factor

        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.float32) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._set_cos_sin_cache(max_seq_len, device or torch.device("cpu"))

    def _set_cos_sin_cache(self, seq_len: int, device: torch.device):
        self.max_seq_len_cached = seq_len
        t = torch.arange(seq_len, device=device, dtype=torch.float32) / self.scaling_factor
        freqs = torch.outer(t, self.inv_freq.to(device))
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, x: torch.Tensor, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len, x.device)
        return (
            self.cos_cached[:seq_len].to(x.dtype),
            self.sin_cached[:seq_len].to(x.dtype),
        )


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dims of the input."""
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
    """Key-Value cache for efficient autoregressive generation."""

    def __init__(self):
        self.key_cache: List[torch.Tensor] = []
        self.value_cache: List[torch.Tensor] = []

    def update(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        layer_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if layer_idx >= len(self.key_cache):
            self.key_cache.append(key)
            self.value_cache.append(value)
        else:
            self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key], dim=2)
            self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value], dim=2)

        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def get_seq_len(self, layer_idx: int = 0) -> int:
        if len(self.key_cache) <= layer_idx:
            return 0
        return self.key_cache[layer_idx].shape[2]

    def reset(self):
        self.key_cache = []
        self.value_cache = []


class GroupedQueryAttention(nn.Module):
    """Grouped Query Attention (GQA) with Flash Attention support."""

    def __init__(self, config: SPARSAConfig, layer_idx: int = 0):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.hidden_dim = config.hidden_dim
        self.num_heads = config.num_heads
        self.num_kv_heads = config.num_kv_heads
        self.head_dim = config.head_dim
        self.num_kv_groups = config.num_kv_groups

        self.q_proj = nn.Linear(self.hidden_dim, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.hidden_dim, self.num_kv_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_dim, self.num_kv_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_dim, bias=config.attention_bias)

        self.rotary_emb = RotaryEmbedding(
            self.head_dim,
            max_seq_len=config.max_seq_len,
            base=config.rope_theta,
        )

        self.use_flash_attention = config.use_flash_attention and FLASH_ATTN_AVAILABLE
        self.sliding_window = config.sliding_window_size if config.use_sliding_window else None
        self.attention_dropout = config.attention_dropout

    def _repeat_kv(self, hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
        """Repeat KV heads for grouped query attention."""
        batch, num_kv_heads, seq_len, head_dim = hidden_states.shape
        if n_rep == 1:
            return hidden_states
        hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_kv_heads, n_rep, seq_len, head_dim)
        return hidden_states.reshape(batch, num_kv_heads * n_rep, seq_len, head_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[KVCache] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[KVCache]]:
        batch_size, seq_len, _ = hidden_states.shape

        # Project to Q, K, V
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Reshape for multi-head attention
        query_states = query_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # Get position embeddings and apply RoPE
        cos, sin = self.rotary_emb(query_states, seq_len=seq_len + (past_key_value.get_seq_len(self.layer_idx) if past_key_value else 0))
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        # Update KV cache
        if past_key_value is not None:
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx)

        # Repeat KV for GQA
        key_states = self._repeat_kv(key_states, self.num_kv_groups)
        value_states = self._repeat_kv(value_states, self.num_kv_groups)

        # Compute attention
        if self.use_flash_attention:
            attn_output = self._flash_attention(query_states, key_states, value_states, attention_mask)
        else:
            attn_output = self._standard_attention(query_states, key_states, value_states, attention_mask)

        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_dim)
        attn_output = self.o_proj(attn_output)

        return attn_output, past_key_value

    def _flash_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Flash Attention 2 implementation."""
        batch_size, num_heads, seq_len, head_dim = query.shape

        # Reshape for flash attention: (batch, seq, heads, head_dim)
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        # Flash attention with optional sliding window
        attn_output = flash_attn_func(
            query,
            key,
            value,
            dropout_p=self.attention_dropout if self.training else 0.0,
            causal=True,
            window_size=(self.sliding_window, self.sliding_window) if self.sliding_window else (-1, -1),
        )

        return attn_output.transpose(1, 2)

    def _standard_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Standard scaled dot-product attention."""
        batch_size, num_heads, seq_len, head_dim = query.shape
        kv_seq_len = key.shape[2]

        # Compute attention scores
        attn_weights = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(head_dim)

        # Apply causal mask
        causal_mask = torch.triu(
            torch.ones(seq_len, kv_seq_len, device=query.device, dtype=torch.bool),
            diagonal=kv_seq_len - seq_len + 1
        )
        attn_weights = attn_weights.masked_fill(causal_mask, float('-inf'))

        # Apply sliding window mask if enabled
        if self.sliding_window is not None:
            window_mask = torch.ones(seq_len, kv_seq_len, device=query.device, dtype=torch.bool)
            for i in range(seq_len):
                start = max(0, kv_seq_len - seq_len + i - self.sliding_window)
                end = kv_seq_len - seq_len + i + 1
                window_mask[i, start:end] = False
            attn_weights = attn_weights.masked_fill(window_mask, float('-inf'))

        # Apply attention mask
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        # Softmax and dropout
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
        attn_weights = F.dropout(attn_weights, p=self.attention_dropout, training=self.training)

        # Compute output
        attn_output = torch.matmul(attn_weights, value)

        return attn_output


class SwiGLU(nn.Module):
    """SwiGLU activation function (GLU variant with SiLU/Swish)."""

    def __init__(self, hidden_dim: int, ff_dim: int, bias: bool = False):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_dim, ff_dim, bias=bias)
        self.up_proj = nn.Linear(hidden_dim, ff_dim, bias=bias)
        self.down_proj = nn.Linear(ff_dim, hidden_dim, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class TransformerBlock(nn.Module):
    """Single transformer block with pre-norm architecture."""

    def __init__(self, config: SPARSAConfig, layer_idx: int = 0):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.input_layernorm = RMSNorm(config.hidden_dim, eps=config.rms_norm_eps)
        self.self_attn = GroupedQueryAttention(config, layer_idx)
        self.post_attention_layernorm = RMSNorm(config.hidden_dim, eps=config.rms_norm_eps)

        if config.use_swiglu:
            self.mlp = SwiGLU(config.hidden_dim, config.ff_dim)
        else:
            self.mlp = nn.Sequential(
                nn.Linear(config.hidden_dim, config.ff_dim),
                nn.SiLU(),
                nn.Linear(config.ff_dim, config.hidden_dim),
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[KVCache] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[KVCache]]:
        # Self-attention with residual
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, past_key_value = self.self_attn(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            use_cache=use_cache,
        )
        hidden_states = residual + hidden_states

        # MLP with residual
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states, past_key_value


class SPARSALM(nn.Module):
    """
    SPARSA-LM 360M: Modern LLaMA-Style Causal Language Model.

    A decoder-only transformer with:
    - RMSNorm for layer normalization
    - SwiGLU activation in the MLP
    - Rotary Position Embeddings (RoPE)
    - Grouped Query Attention (GQA)
    - Flash Attention 2 support
    - Sliding window attention
    """

    def __init__(self, config: SPARSAConfig):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.hidden_dim = config.hidden_dim

        # Token embeddings
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_dim)

        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerBlock(config, layer_idx=i) for i in range(config.num_layers)
        ])

        # Final layer norm
        self.norm = RMSNorm(config.hidden_dim, eps=config.rms_norm_eps)

        # Output projection (language modeling head)
        self.lm_head = nn.Linear(config.hidden_dim, config.vocab_size, bias=False)

        # Weight tying (optional)
        if config.tie_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

        # Initialize weights
        self.apply(self._init_weights)

        # Gradient checkpointing flag
        self.gradient_checkpointing = config.use_checkpointing

    def _init_weights(self, module: nn.Module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embed_tokens

    def set_input_embeddings(self, value: nn.Embedding):
        self.embed_tokens = value

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[KVCache] = None,
        use_cache: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        batch_size, seq_len = input_ids.shape

        # Get embeddings
        hidden_states = self.embed_tokens(input_ids)

        # Prepare position ids
        if position_ids is None:
            past_len = past_key_values.get_seq_len() if past_key_values else 0
            position_ids = torch.arange(
                past_len, past_len + seq_len, device=input_ids.device
            ).unsqueeze(0).expand(batch_size, -1)

        # Prepare attention mask for Flash Attention
        if attention_mask is not None and not (self.config.use_flash_attention and FLASH_ATTN_AVAILABLE):
            attention_mask = self._prepare_attention_mask(attention_mask, seq_len)

        # Initialize KV cache if needed
        if use_cache and past_key_values is None:
            past_key_values = KVCache()

        # Store hidden states if requested
        all_hidden_states = () if output_hidden_states else None

        # Forward through transformer layers
        for layer in self.layers:
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.gradient_checkpointing and self.training:
                hidden_states, past_key_values = checkpoint(
                    layer,
                    hidden_states,
                    attention_mask,
                    position_ids,
                    past_key_values,
                    use_cache,
                    use_reentrant=False,
                )
            else:
                hidden_states, past_key_values = layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    use_cache=use_cache,
                )

        # Final layer norm
        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        # Compute logits
        logits = self.lm_head(hidden_states)

        if return_dict:
            return {
                "logits": logits,
                "past_key_values": past_key_values,
                "hidden_states": all_hidden_states,
            }

        if use_cache:
            return logits, past_key_values
        return logits

    def _prepare_attention_mask(
        self,
        attention_mask: torch.Tensor,
        seq_len: int,
    ) -> torch.Tensor:
        """Prepare 4D attention mask for standard attention."""
        # Expand mask to 4D: (batch, 1, seq, seq)
        extended_mask = attention_mask[:, None, None, :]
        extended_mask = extended_mask.expand(-1, -1, seq_len, -1)

        # Create causal mask
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=attention_mask.device),
            diagonal=1
        ).bool()

        # Combine masks
        extended_mask = extended_mask.masked_fill(causal_mask, 0)
        extended_mask = (1.0 - extended_mask) * torch.finfo(torch.float32).min

        return extended_mask.to(attention_mask.dtype)

    def compute_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        reduction: str = "mean",
    ) -> torch.Tensor:
        """Compute cross-entropy loss for language modeling."""
        # Shift for next-token prediction
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        # Flatten for loss computation
        loss_fct = nn.CrossEntropyLoss(
            ignore_index=self.config.pad_token_id,
            reduction=reduction,
        )

        loss = loss_fct(
            shift_logits.view(-1, self.vocab_size),
            shift_labels.view(-1),
        )

        return loss

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 128,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
        repetition_penalty: float = 1.0,
        do_sample: bool = True,
        eos_token_id: Optional[int] = None,
        pad_token_id: Optional[int] = None,
    ) -> torch.Tensor:
        """Generate text autoregressively with KV caching."""
        self.eval()
        eos_token_id = eos_token_id or self.config.eos_token_id
        pad_token_id = pad_token_id or self.config.pad_token_id

        batch_size = input_ids.shape[0]
        kv_cache = KVCache()

        # Track which sequences have finished
        finished = torch.zeros(batch_size, dtype=torch.bool, device=input_ids.device)

        for _ in range(max_new_tokens):
            # Forward pass with KV cache
            outputs = self.forward(
                input_ids[:, -1:] if kv_cache.get_seq_len() > 0 else input_ids,
                past_key_values=kv_cache,
                use_cache=True,
            )

            logits = outputs["logits"][:, -1, :]
            kv_cache = outputs["past_key_values"]

            # Apply temperature
            if temperature != 1.0:
                logits = logits / temperature

            # Apply repetition penalty
            if repetition_penalty != 1.0:
                for i in range(batch_size):
                    for token_id in set(input_ids[i].tolist()):
                        if logits[i, token_id] > 0:
                            logits[i, token_id] /= repetition_penalty
                        else:
                            logits[i, token_id] *= repetition_penalty

            # Apply top-k filtering
            if top_k > 0:
                indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                logits[indices_to_remove] = float('-inf')

            # Apply top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0

                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = float('-inf')

            # Sample or greedy decode
            if do_sample:
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)

            # Update finished sequences
            finished = finished | (next_token.squeeze(-1) == eos_token_id)

            # Replace finished tokens with pad
            next_token = torch.where(
                finished.unsqueeze(-1),
                torch.full_like(next_token, pad_token_id),
                next_token,
            )

            # Append to sequence
            input_ids = torch.cat([input_ids, next_token], dim=-1)

            # Stop if all sequences finished
            if finished.all():
                break

        return input_ids

    def save_pretrained(self, save_directory: str):
        """Save model and config to directory (HuggingFace Hub compatible)."""
        import os
        import json

        os.makedirs(save_directory, exist_ok=True)

        # Save model weights
        torch.save(self.state_dict(), os.path.join(save_directory, "pytorch_model.bin"))

        # Save config
        config_dict = {
            "architecture": "SPARSALM",
            "vocab_size": self.config.vocab_size,
            "hidden_dim": self.config.hidden_dim,
            "num_layers": self.config.num_layers,
            "num_heads": self.config.num_heads,
            "num_kv_heads": self.config.num_kv_heads,
            "ff_dim": self.config.ff_dim,
            "max_seq_len": self.config.max_seq_len,
            "rms_norm_eps": self.config.rms_norm_eps,
            "rope_theta": self.config.rope_theta,
            "use_flash_attention": self.config.use_flash_attention,
            "use_sliding_window": self.config.use_sliding_window,
            "sliding_window_size": self.config.sliding_window_size,
            "use_swiglu": self.config.use_swiglu,
            "tie_embeddings": self.config.tie_embeddings,
        }

        with open(os.path.join(save_directory, "config.json"), "w") as f:
            json.dump(config_dict, f, indent=2)

    @classmethod
    def from_pretrained(cls, model_path: str, device: Optional[torch.device] = None):
        """Load model from pretrained weights."""
        import os
        import json

        # Load config
        with open(os.path.join(model_path, "config.json"), "r") as f:
            config_dict = json.load(f)

        config = SPARSAConfig(
            vocab_size=config_dict["vocab_size"],
            hidden_dim=config_dict["hidden_dim"],
            num_layers=config_dict["num_layers"],
            num_heads=config_dict["num_heads"],
            num_kv_heads=config_dict.get("num_kv_heads", config_dict["num_heads"]),
            ff_dim=config_dict["ff_dim"],
            max_seq_len=config_dict["max_seq_len"],
            rms_norm_eps=config_dict.get("rms_norm_eps", 1e-6),
            rope_theta=config_dict.get("rope_theta", 10000.0),
            use_flash_attention=config_dict.get("use_flash_attention", True),
            use_sliding_window=config_dict.get("use_sliding_window", True),
            sliding_window_size=config_dict.get("sliding_window_size", 512),
            use_swiglu=config_dict.get("use_swiglu", True),
            tie_embeddings=config_dict.get("tie_embeddings", False),
            device=device,
        )

        # Create model
        model = cls(config)

        # Load weights
        state_dict = torch.load(
            os.path.join(model_path, "pytorch_model.bin"),
            map_location=device or "cpu",
        )
        model.load_state_dict(state_dict)

        if device:
            model = model.to(device)

        return model

    def num_parameters(self, trainable_only: bool = False) -> int:
        """Count model parameters."""
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())


# Backward compatibility alias
Transformer = SPARSALM
TransformerConfig = SPARSAConfig


def create_model(config: Dict[str, Any]) -> SPARSALM:
    """Factory function to create SPARSA-LM model from config dict."""
    model_config = SPARSAConfig(
        vocab_size=config.get("vocab_size", 32000),
        hidden_dim=config.get("hidden_dim", 1024),
        num_layers=config.get("num_layers", 28),
        num_heads=config.get("num_heads", 16),
        num_kv_heads=config.get("num_kv_heads", 8),
        ff_dim=config.get("ff_dim", 4096),
        max_seq_len=config.get("max_seq_len", 2048),
        rms_norm_eps=config.get("rms_norm_eps", 1e-6),
        use_flash_attention=config.get("use_flash_attention", True),
        use_sliding_window=config.get("use_sliding_window", True),
        sliding_window_size=config.get("sliding_window_size", 512),
        use_swiglu=config.get("use_swiglu", True),
        dropout=config.get("dropout", 0.0),
        use_checkpointing=config.get("use_checkpointing", True),
        tie_embeddings=config.get("tie_embeddings", False),
        initializer_range=config.get("initializer_range", 0.02),
    )

    return SPARSALM(model_config)
