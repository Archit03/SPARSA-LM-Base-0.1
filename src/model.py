import math
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
import transformers
from typing import Dict, Optional, List, Tuple, Union
from torch.optim.lr_scheduler import LambdaLR
import numpy as np
import random

"""This code belongs to EllanorAI and is licensed under the EllanorAI Proprietary License."""

########################################################################################
# TransformerConfig
########################################################################################

class TransformerConfig:
    def __init__(
        self,
        # Model Architecture
        vocab_size: int,
        d_model: int,  # hidden_dim in config
        num_layers: int,
        num_heads: int,
        d_ff: int,  # ff_dim in config
        max_seq_len: int,
        dropout: float,

        # Noise insertions
        noise_type: str = "mask",
        noise_prob: float = 0.0,
        
        # Attention Mechanisms
        use_rope: bool = True,
        window_size: int = 4,
        global_tokens: int = 0,
        
        # Architecture Options
        prenorm: bool = True,
        tie_embeddings: bool = False,
        
        # Training Features
        use_checkpointing: bool = True,
        use_regularization: bool = True,
        use_mixed_precision: bool = True,
        label_smoothing: float = 0.1,
        l2_reg: float = 0.01,
        max_grad_norm: float = 1.0,
        
        # Optimization
        learning_rate: float = 1e-4,
        weight_decay: float = 0.1,
        warmup_ratio: float = 0.1,
        scheduler_type: str = "cosine_with_min_lr",
        lr_scheduler_kwargs: dict = None,

        init_scale: float = 0.02,
        layer_norm_eps: float = 1e-5,
        
        # Special Tokens
        pad_token_id: int = 0,
        
        # Model Behavior
        activation: str = "gelu",  # Options: "gelu", "relu", "silu"
        
        # Device
        device: Optional[torch.device] = None,
        
        # Added parameter
        use_reentrant: bool = True
    ):
        """Initialize transformer configuration with validation."""
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        assert activation in ["gelu", "relu", "silu"], f"Unsupported activation: {activation}"
        
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.max_seq_len = max_seq_len
        self.dropout = dropout
        self.init_scale = init_scale
        self.layer_norm_eps = layer_norm_eps
        
        self.use_rope = use_rope
        self.window_size = window_size
        self.global_tokens = global_tokens
        
        self.prenorm = prenorm
        self.tie_embeddings = tie_embeddings
        
        self.use_checkpointing = use_checkpointing
        self.use_regularization = use_regularization
        self.use_mixed_precision = use_mixed_precision
        self.label_smoothing = label_smoothing
        self.l2_reg = l2_reg
        self.max_grad_norm = max_grad_norm
        
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_ratio = warmup_ratio
        self.scheduler_type = scheduler_type
        self.lr_scheduler_kwargs = lr_scheduler_kwargs or {"min_lr": 1e-6}
        
        self.pad_token_id = pad_token_id
        
        self.activation = activation

        self.noise_type = noise_type
        self.noise_prob = noise_prob

        assert window_size > 0, "Window size must be positive"
        assert window_size <= max_seq_len, "Window size cannot exceed max sequence length"
        assert global_tokens >= 0, "Number of global tokens cannot be negative"
        assert global_tokens <= max_seq_len, "Number of global tokens cannot exceed max sequence length"

        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.use_reentrant = use_reentrant

########################################################################################
# Core Components - Positional Encodings
########################################################################################

class RotaryPositionalEmbedding(nn.Module):
    """Rotary positional embedding with caching and vectorized computation."""
    def __init__(self, dim: int, max_seq_len: int = 4096):
        super().__init__()
        inv_freq = 1. / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        self.max_seq_len = max_seq_len
        self.seq_len_cached = -1
        self.cos_cached = None
        self.sin_cached = None

    def forward(self, x: torch.Tensor, seq_len: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        if seq_len is None:
            seq_len = x.size(1)
        if seq_len > self.max_seq_len:
            raise ValueError(f"Sequence length {seq_len} exceeds maximum allowed length {self.max_seq_len}")
        if seq_len > self.seq_len_cached:
            self.seq_len_cached = seq_len
            t = torch.arange(seq_len, device=x.device).unsqueeze(1)  # shape: [seq_len, 1]
            # Use broadcasting to compute frequencies without explicit loops
            freqs = t * self.inv_freq.unsqueeze(0)  # shape: [seq_len, dim/2]
            emb = torch.cat((freqs, freqs), dim=-1)   # shape: [seq_len, dim]
            emb = emb.unsqueeze(0).unsqueeze(0)  # shape: [1, 1, seq_len, dim]
            self.cos_cached = emb.cos()
            self.sin_cached = emb.sin()
        return self.cos_cached[:, :, :seq_len, :], self.sin_cached[:, :, :seq_len, :]

class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding with dropout."""
    def __init__(self, d_model: int, max_len: int = 4096, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        return self.dropout(x + self.pe[:, :seq_len])

########################################################################################
# Sparse Multi-Head Attention with Optional KV Caching
########################################################################################

class SparseMultiHeadAttention(nn.Module):
    """
    Multi-Head Attention with Sparse Local Windowing and Optional Global Tokens.
    Minor improvement: vectorized mask building can be implemented in future.
    """
    def __init__(self, config, is_causal=False):
        super().__init__()
        self.num_heads = config.num_heads
        self.d_model = config.d_model
        self.d_k = config.d_model // config.num_heads
        self.window_size = config.window_size
        self.global_tokens = config.global_tokens
        self.is_causal = is_causal

        self.query = nn.Linear(config.d_model, config.d_model)
        self.key = nn.Linear(config.d_model, config.d_model)
        self.value = nn.Linear(config.d_model, config.d_model)
        self.output = nn.Linear(config.d_model, config.d_model)

        self.dropout = nn.Dropout(config.dropout)
        self.softmax = nn.Softmax(dim=-1)

        for layer in [self.query, self.key, self.value, self.output]:
            nn.init.xavier_uniform_(layer.weight, gain=1.0 / math.sqrt(self.d_k))
            nn.init.zeros_(layer.bias)

    def build_local_window_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        # Minor tweak: a more vectorized approach can replace this loop in the future.
        mask = torch.zeros(seq_len, seq_len, dtype=torch.float, device=device)
        for i in range(seq_len):
            start = max(0, i - self.window_size)
            end = min(seq_len, i + self.window_size + 1)
            mask[i, start:end] = 1
        if self.global_tokens > 0:
            mask[:self.global_tokens, :] = 1
            mask[:, :self.global_tokens] = 1
        return mask

    def forward(self, query, key, value, attn_mask=None, past_key_value=None, use_cache=False):
        batch_size, new_seq_len, _ = query.size()
        Q = self.query(query).view(batch_size, new_seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.key(key).view(batch_size, new_seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.value(value).view(batch_size, new_seq_len, self.num_heads, self.d_k).transpose(1, 2)

        if past_key_value is not None:
            past_K, past_V = past_key_value
            K = torch.cat([past_K, K], dim=2)
            V = torch.cat([past_V, V], dim=2)
        else:
            past_K = torch.zeros((batch_size, self.num_heads, 0, self.d_k), device=K.device, dtype=K.dtype)
            past_V = torch.zeros((batch_size, self.num_heads, 0, self.d_k), device=V.device, dtype=V.dtype)

        next_key_value = (K, V) if use_cache else None
        total_seq_len = K.size(2)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        local_mask = self.build_local_window_mask(total_seq_len, device=query.device)
        start = total_seq_len - new_seq_len
        local_mask_slice = local_mask[start:start+new_seq_len, :]
        scores = scores.masked_fill(local_mask_slice.unsqueeze(0).unsqueeze(0) == 0, float('-inf'))

        if self.is_causal:
            full_causal = torch.triu(torch.ones(total_seq_len, total_seq_len, device=query.device), diagonal=1)
            causal_slice = full_causal[start:start+new_seq_len, :]
            scores = scores.masked_fill(causal_slice.unsqueeze(0).unsqueeze(0) == 1, float('-inf'))

        scores = self.softmax(scores)
        scores = self.dropout(scores)
        output = torch.matmul(scores, V)
        output = output.transpose(1, 2).contiguous().view(batch_size, new_seq_len, self.d_model)
        output = self.output(output)
        return output, next_key_value

########################################################################################
# Local Window Mask with Optional Global Tokens (Alternate Implementation)
########################################################################################

def build_local_window_mask(seq_len: int, window_size: int, global_tokens: int = 0, is_causal: bool = False, device: Optional[torch.device] = None) -> torch.Tensor:
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mask = torch.zeros(seq_len, seq_len, dtype=torch.float, device=device)
    for i in range(seq_len):
        for j in range(seq_len):
            if i < global_tokens or j < global_tokens:
                continue
            if abs(i - j) > window_size:
                mask[i, j] = float('-inf')
            if is_causal and j > i:
                mask[i, j] = float('-inf')
    return mask

########################################################################################
# Residual + LayerNorm with Optional Gradient Checkpointing
########################################################################################

class ResidualConnection(nn.Module):
    """
    Implements y = x + dropout(sublayer(LN(x))) with additional error checking.
    """
    def __init__(self, d_model, dropout=0.1, use_checkpointing=False, use_reentrant=False, layer_norm_eps=1e-5):
        super().__init__()
        self.norm = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout = nn.Dropout(dropout)
        self.use_checkpointing = use_checkpointing
        self.use_reentrant = use_reentrant

    def forward(self, x, sublayer):
        def forward_sublayer(normed_x):
            return sublayer(normed_x)
        
        if not x.isfinite().all():
            logging.warning("Non-finite values in residual input; replacing with numerical safe values.")
            x = torch.nan_to_num(x, nan=0.0, posinf=1e4, neginf=-1e4)
        
        normed = self.norm(x)
        if self.use_checkpointing and normed.requires_grad:
            sublayer_output = checkpoint(forward_sublayer, normed, use_reentrant=self.use_reentrant)
        else:
            sublayer_output = sublayer(normed)
        
        if not sublayer_output.isfinite().all():
            logging.warning("Non-finite values in sublayer output; replacing with numerical safe values.")
            sublayer_output = torch.nan_to_num(sublayer_output, nan=0.0, posinf=1e4, neginf=-1e4)
        
        out = x + self.dropout(sublayer_output)
        if not out.isfinite().all():
            logging.warning("Non-finite values in residual output; replacing with numerical safe values.")
            out = torch.nan_to_num(out, nan=0.0, posinf=1e4, neginf=-1e4)
        return out

########################################################################################
# Feed Forward
########################################################################################

class FeedForward(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.linear1 = nn.Linear(config.d_model, config.d_ff)
        self.linear2 = nn.Linear(config.d_ff, config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        self.activation = {
            'relu': F.relu,
            'gelu': F.gelu,
            'silu': F.silu
        }[config.activation.lower()]

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x

########################################################################################
# Encoder Block with Sparse Attention, Global Tokens, and Checkpointing
########################################################################################

class EncoderBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self_attn = SparseMultiHeadAttention(config)
        self.feed_forward = FeedForward(config)
        self.self_attn_res = ResidualConnection(config.d_model, config.dropout)
        self.ff_res = ResidualConnection(config.d_model, config.dropout)

    def forward(self, x, src_mask=None, past_key_value=None, use_cache=False):
        next_kv = None
        def sa_sublayer(normed_x):
            nonlocal next_kv
            out, new_kv = self.self_attn(normed_x, normed_x, normed_x, attn_mask=src_mask, past_key_value=past_key_value, use_cache=use_cache)
            next_kv = new_kv
            return out
        x = self.self_attn_res(x, sa_sublayer)
        x = self.ff_res(x, self.feed_forward)
        return x, next_kv

########################################################################################
# Decoder Block with Sparse Self-Attention, Cross-Attention, Checkpointing, and Cache
########################################################################################

class DecoderBlock(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.self_attn_res = ResidualConnection(config.d_model, config.dropout, config.use_checkpointing, config.use_reentrant)
        self.self_attn = SparseMultiHeadAttention(config, is_causal=True)
        self.cross_attn_res = ResidualConnection(config.d_model, config.dropout, config.use_checkpointing, config.use_reentrant)
        self.cross_attn = SparseMultiHeadAttention(config)
        self.ff_res = ResidualConnection(config.d_model, config.dropout, config.use_checkpointing, config.use_reentrant)
        self.feed_forward = FeedForward(config)

    def forward(self, x, encoder_output, tgt_mask=None, src_mask=None, past_key_value=None, use_cache=False):
        past_self_k = past_self_v = past_cross_k = past_cross_v = None
        if past_key_value is not None:
            if len(past_key_value) == 2:
                past_self_k, past_self_v = past_key_value
            elif len(past_key_value) == 4:
                past_self_k, past_self_v, past_cross_k, past_cross_v = past_key_value
        next_self_kv = None
        def self_attn_sublayer(normed_x):
            nonlocal next_self_kv
            out, new_self_kv = self.self_attn(normed_x, normed_x, normed_x,
                                               attn_mask=tgt_mask,
                                               past_key_value=(past_self_k, past_self_v),
                                               use_cache=use_cache)
            next_self_kv = new_self_kv
            return out
        x = self.self_attn_res(x, self_attn_sublayer)
        next_cross_kv = None
        def cross_attn_sublayer(normed_x):
            nonlocal next_cross_kv
            out, new_cross_kv = self.cross_attn(normed_x, encoder_output, encoder_output,
                                                 attn_mask=src_mask,
                                                 past_key_value=(past_cross_k, past_cross_v),
                                                 use_cache=use_cache)
            next_cross_kv = new_cross_kv
            return out
        x = self.cross_attn_res(x, cross_attn_sublayer)
        x = self.ff_res(x, self.feed_forward)
        next_key_value = None
        if use_cache:
            next_key_value = (
                next_self_kv[0] if next_self_kv else None,
                next_self_kv[1] if next_self_kv else None,
                next_cross_kv[0] if next_cross_kv else None,
                next_cross_kv[1] if next_cross_kv else None
            )
        return x, next_key_value

########################################################################################
# Full Encoder
########################################################################################

class Encoder(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.layers = nn.ModuleList([EncoderBlock(config) for _ in range(config.num_layers)])
        self.norm = nn.LayerNorm(config.d_model)
    
    def forward(self, x, src_mask=None, past_key_values=None, use_cache=False):
        next_past_key_values = []
        batch_size = x.size(0)
        for i, layer in enumerate(self.layers):
            if past_key_values is not None and i < len(past_key_values):
                past_kv = past_key_values[i]
            else:
                num_heads = layer.self_attn.num_heads
                d_k = layer.self_attn.d_k
                past_kv = (
                    torch.zeros(batch_size, num_heads, 0, d_k, device=x.device, dtype=x.dtype),
                    torch.zeros(batch_size, num_heads, 0, d_k, device=x.device, dtype=x.dtype)
                )
            x, new_kv = layer(x, src_mask=src_mask, past_key_value=past_kv, use_cache=use_cache)
            next_past_key_values.append(new_kv)
        x = self.norm(x)
        return x, next_past_key_values if use_cache else None

########################################################################################
# Full Decoder
########################################################################################

class Decoder(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.layers = nn.ModuleList([DecoderBlock(config) for _ in range(config.num_layers)])
        self.norm = nn.LayerNorm(config.d_model)

    def forward(self, x, encoder_output, tgt_mask=None, src_mask=None, past_key_values=None, use_cache=False):
        next_past_key_values = []
        batch_size = x.size(0)
        for i, layer in enumerate(self.layers):
            if past_key_values is not None and i < len(past_key_values):
                past_kv = past_key_values[i]
            else:
                num_heads = layer.self_attn.num_heads
                d_k = layer.self_attn.d_k
                self_past = (
                    torch.zeros(batch_size, num_heads, 0, d_k, device=x.device, dtype=x.dtype),
                    torch.zeros(batch_size, num_heads, 0, d_k, device=x.device, dtype=x.dtype)
                )
                cross_past = (
                    torch.zeros(batch_size, num_heads, 0, d_k, device=x.device, dtype=x.dtype),
                    torch.zeros(batch_size, num_heads, 0, d_k, device=x.device, dtype=x.dtype)
                )
                past_kv = (self_past[0], self_past[1], cross_past[0], cross_past[1])
            x, new_kv = layer(x, encoder_output, tgt_mask=tgt_mask, src_mask=src_mask, past_key_value=past_kv, use_cache=use_cache)
            next_past_key_values.append(new_kv)
        x = self.norm(x)
        return x, next_past_key_values if use_cache else None

########################################################################################
# Noise Injection Function
########################################################################################

def add_noise_to_input(tokens: torch.Tensor, noise_type="mask", mask_token_id=4, prob=0.0, vocab_size=None, pad_token_id=0) -> torch.Tensor:
    """
    Apply noise directly on GPU.
    For 'delete' noise, tokens are re-padded per row.
    """
    noisy_tokens = tokens.clone()
    batch_size, seq_len = noisy_tokens.shape

    if noise_type == "mask":
        mask = torch.rand_like(noisy_tokens.float()) < prob
        noisy_tokens = torch.where(mask, torch.tensor(mask_token_id, device=tokens.device), noisy_tokens)
    elif noise_type == "delete":
        keep_mask = (torch.rand_like(noisy_tokens.float()) > prob)
        new_noisy = torch.full_like(noisy_tokens, pad_token_id)
        for i in range(batch_size):
            row = noisy_tokens[i]
            row_keep = keep_mask[i]
            row_kept = row[row_keep]
            length_to_copy = min(seq_len, row_kept.size(0))
            new_noisy[i, :length_to_copy] = row_kept[:length_to_copy]
        noisy_tokens = new_noisy
    elif noise_type == "permute":
        perm = torch.rand(batch_size, seq_len, device=tokens.device).argsort(dim=1)
        noisy_tokens = noisy_tokens.gather(1, perm)
    elif noise_type == "substitute":
        if vocab_size is None:
            raise ValueError("Must provide vocab_size for 'substitute' noise.")
        mask = torch.rand_like(noisy_tokens.float()) < prob
        random_tokens = torch.randint(low=0, high=vocab_size, size=noisy_tokens.shape, device=noisy_tokens.device)
        noisy_tokens = torch.where(mask, random_tokens, noisy_tokens)
    return noisy_tokens

########################################################################################
# Full Encoder–Decoder Transformer Model
########################################################################################

class Transformer(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        self.device = config.device
        
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        
        if not config.use_rope:
            self.pos_encoding = PositionalEncoding(config.d_model, config.max_seq_len, config.dropout)
        
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)
        
        self.generator = nn.Linear(config.d_model, config.vocab_size)
        if config.tie_embeddings:
            self.generator.weight = self.embedding.weight
        
        self._reset_parameters()
        
        self.scaler = torch.amp.GradScaler(device="cuda" if torch.cuda.is_available() else "cpu",
                                            enabled=config.use_mixed_precision)
        self.metrics = {
            'train': {'loss': [], 'accuracy': [], 'perplexity': []},
            'val': {'loss': [], 'accuracy': [], 'perplexity': []}
        }
        self.best_val_loss = float('inf')

    def _reset_parameters(self):
        for name, p in self.named_parameters():
            if p.dim() > 1:
                if 'embedding' in name:
                    nn.init.normal_(p, mean=0.0, std=self.config.init_scale)
                elif 'linear' in name or 'w_' in name:
                    nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                elif 'norm' in name:
                    if 'weight' in name:
                        nn.init.ones_(p)
                    elif 'bias' in name:
                        nn.init.zeros_(p)
            else:
                nn.init.zeros_(p)

    def forward(self, src: torch.Tensor, tgt: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None, tgt_mask: Optional[torch.Tensor] = None,
                past_key_values: Optional[List[Tuple]] = None, use_cache: bool = False
               ) -> Union[torch.Tensor, Tuple[torch.Tensor, Tuple]]:
        """
        Forward pass:
          - Encoder receives a noisy version of src during training.
          - Decoder receives clean tgt tokens to learn reconstruction.
        """
        if src.dtype != torch.long:
            print(f"🚨 WARNING: Model received `src` with dtype {src.dtype}, forcing conversion to long.")
            src = src.to(torch.long)
        if tgt is not None and tgt.dtype != torch.long:
            print(f"🚨 WARNING: Model received `tgt` with dtype {tgt.dtype}, forcing conversion to long.")
            tgt = tgt.to(torch.long)

        if not src.isfinite().all():
            src = torch.nan_to_num(src)
            print("Warning: Non-finite values in source tokens")
        
        # --- Encoder: Apply noise to source tokens only during training ---
        if self.training and getattr(self.config, 'use_noise_injection', False):
            src_noisy = add_noise_to_input(
                src,
                noise_type=self.config.noise_type,
                mask_token_id=self.config.pad_token_id,
                prob=self.config.noise_prob
            )
        else:
            src_noisy = src
        
        with torch.amp.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu",
                                enabled=self.config.use_mixed_precision):
            enc_inp = self.embedding(src_noisy)
            if not self.config.use_rope:
                enc_inp = self.pos_encoding(enc_inp)
            if self.training:
                enc_inp = enc_inp + torch.randn_like(enc_inp) * 1e-5
            
            encoder_past = None if past_key_values is None else past_key_values[0]
            encoder_output, enc_past = self.encoder(enc_inp, src_mask=attention_mask, past_key_values=encoder_past, use_cache=use_cache)
            if not encoder_output.isfinite().all():
                print("Warning: Non-finite values in encoder output")
                encoder_output = torch.nan_to_num(encoder_output, nan=0.0, posinf=1e4, neginf=-1e4)
            
            # --- Decoder: Always use clean target tokens ---
            if tgt is not None:
                dec_inp = self.embedding(tgt)
                if not self.config.use_rope:
                    dec_inp = self.pos_encoding(dec_inp)
                decoder_past = None if past_key_values is None else past_key_values[1]
                decoder_output, dec_past = self.decoder(dec_inp, encoder_output, tgt_mask=tgt_mask, src_mask=attention_mask, past_key_values=decoder_past, use_cache=use_cache)
                logits = self.generator(decoder_output) / math.sqrt(self.config.d_model)
                if use_cache:
                    return logits, (enc_past, dec_past)
                return logits
            return encoder_output

    def configure_optimizer(self, config: TransformerConfig):
        decay_params = []
        no_decay_params = []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if 'bias' in name or 'norm' in name or 'embedding' in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)
        optim_groups = [
            {'params': decay_params, 'weight_decay': config.weight_decay},
            {'params': no_decay_params, 'weight_decay': 0.0}
        ]
        optimizer = torch.optim.AdamW(
            optim_groups,
            lr=config.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        return optimizer
    
    def training_step(self, batch, optimizer) -> Dict[str, float]:
        self.train()
        metrics = {}
        with torch.amp.autocast(enabled=self.config.use_mixed_precision):
            outputs = self(
                src=batch['encoder_input_ids'],
                tgt=batch['decoder_input_ids'],
                attention_mask=batch.get('encoder_attention_mask'),
                tgt_mask=batch.get('decoder_attention_mask')
            )
            loss = self.compute_loss(outputs, batch['labels'])
            metrics['loss'] = loss.item()
            if self.config.use_regularization:
                reg_loss = self.compute_regularization()
                loss = loss + reg_loss
                metrics['reg_loss'] = reg_loss.item()
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(self.parameters(), self.config.max_grad_norm)
        metrics['grad_norm'] = grad_norm.item()
        self.scaler.step(optimizer)
        self.scaler.update()
        optimizer.zero_grad(set_to_none=True)
        return metrics

    def validation_step(self, batch) -> Dict[str, float]:
        self.eval()
        metrics = {}
        with torch.no_grad():
            outputs = self(
                src=batch['input_ids'],
                tgt=batch['labels'],
                src_mask=batch.get('attention_mask'),
                tgt_mask=self._generate_square_subsequent_mask(batch['labels'].size(1))
            )
            if outputs is None:
                print("Validation Step - Model output is None! Skipping this batch...")
                return {"loss": float("inf"), "accuracy": 0.0}
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            metrics['loss'] = self.compute_loss(outputs, batch['labels']).item()
            if outputs is not None:
                metrics.update(self.compute_metrics(outputs, batch['labels']))
        return metrics

    def compute_regularization(self) -> torch.Tensor:
        reg_loss = 0.0
        if self.config.l2_reg > 0:
            for param in self.parameters():
                reg_loss += torch.norm(param, p=2)
            reg_loss *= self.config.l2_reg
        return reg_loss

    def compute_metrics(self, outputs, labels) -> Dict[str, float]:
        metrics = {}
        predictions = outputs.argmax(dim=-1)
        mask = labels != self.config.pad_token_id
        correct = (predictions == labels) & mask
        denominator = mask.sum().float().clamp(min=1)
        metrics['accuracy'] = correct.sum().float() / denominator
        metrics['perplexity'] = torch.exp(self.compute_loss(outputs, labels))
        return metrics

    def compute_loss(self, outputs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        outputs = outputs.float() / math.sqrt(self.config.d_model)
        log_probs = F.log_softmax(outputs, dim=-1, dtype=torch.float32)
        if self.config.label_smoothing > 0:
            smooth_target = torch.full_like(log_probs, self.config.label_smoothing / (self.config.vocab_size - 1))
            smooth_target.scatter_(-1, labels.unsqueeze(-1), 1.0 - self.config.label_smoothing)
            pad_mask = (labels != self.config.pad_token_id).float().unsqueeze(-1)
            loss = -(smooth_target * log_probs * pad_mask).sum(dim=-1)
        else:
            loss = F.nll_loss(
                log_probs.view(-1, self.config.vocab_size),
                labels.view(-1),
                ignore_index=self.config.pad_token_id,
                reduction='none'
            )
        pad_mask = (labels != self.config.pad_token_id).float()
        return (loss * pad_mask).sum() / pad_mask.sum().clamp(min=1)
    
    @staticmethod
    def get_scheduler(optimizer, config: TransformerConfig, num_training_steps: int):
        num_warmup_steps = int(num_training_steps * config.warmup_ratio)
        if config.scheduler_type in ["cosine_warmup", "cosine_with_min_lr"]:
            default_min_lr = 0.0 if config.scheduler_type == "cosine_warmup" else 1e-6
            min_lr = config.lr_scheduler_kwargs.get("min_lr", default_min_lr)
            def lr_lambda(step):
                if step < num_warmup_steps:
                    return float(step) / float(max(1, num_warmup_steps))
                else:
                    progress = float(step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
                    return min_lr + (config.learning_rate - min_lr) * 0.5 * (1.0 + np.cos(np.pi * progress))
            return LambdaLR(optimizer, lr_lambda)
        elif config.scheduler_type == "linear_warmup":
            return transformers.get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps
            )
        else:
            raise ValueError(f"Unknown scheduler type: {config.scheduler_type}")

    @staticmethod
    def _generate_square_subsequent_mask(sz: int) -> torch.Tensor:
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
