import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
import transformers
from typing import Dict, Optional, List, Tuple, Union

"""This code belongs to EllanorAI and is licensed under the EllanorAI Proprietary License."""

########################################################################################
#TransformerConfig
########################################################################################

class TransformerConfig:
    def __init__(
        self,
        # Model Architecture
        vocab_size: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int,
        dropout: float,
        
        # Attention Mechanisms
        use_rope: bool,
        window_size: int,
        global_tokens: int,
        
        # Architecture Options
        prenorm: bool = True,
        tie_embeddings: bool = True,
        
        # Training Features
        use_checkpointing: bool = False,
        use_regularization: bool = False,
        use_mixed_precision: bool = False,
        label_smoothing: float = 0.0,
        l2_reg: float = 0.0,
        max_grad_norm: float = 1.0,
        
        # Optimization
        learning_rate: float = 3e-4,
        weight_decay: float = 0.01,
        warmup_ratio: float = 0.1,
        scheduler_type: str = "linear_warmup",
        
        # Special Tokens
        pad_token_id: int = 0,
        
        # Model Behavior
        activation: str = "gelu"  # Options: "gelu", "relu", "silu"
    ):
        """Initialize transformer configuration with validation."""
        # Validate core architecture parameters
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        assert activation in ["gelu", "relu", "silu"], f"Unsupported activation: {activation}"
        
        # Model Architecture
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.max_seq_len = max_seq_len
        self.dropout = dropout
        
        # Attention Mechanisms
        self.use_rope = use_rope
        self.window_size = window_size
        self.global_tokens = global_tokens
        
        # Architecture Options
        self.prenorm = prenorm
        self.tie_embeddings = tie_embeddings
        
        # Training Features
        self.use_checkpointing = use_checkpointing
        self.use_regularization = use_regularization
        self.use_mixed_precision = use_mixed_precision
        self.label_smoothing = label_smoothing
        self.l2_reg = l2_reg
        self.max_grad_norm = max_grad_norm
        
        # Optimization
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_ratio = warmup_ratio
        self.scheduler_type = scheduler_type
        
        # Special Tokens
        self.pad_token_id = pad_token_id
        
        # Model Behavior
        self.activation = activation

    @staticmethod
    def get_2M_config():
        """Returns config for ~2M parameter model with 6k vocab size"""
        return TransformerConfig(
            # Model Architecture
            vocab_size=6000,
            d_model=256,
            num_layers=6,
            num_heads=8,
            d_ff=1024,
            max_seq_len=2048,
            dropout=0.1,
            
            # Attention Mechanisms
            use_rope=True,
            window_size=256,
            global_tokens=8,
            
            # Architecture Options
            prenorm=True,
            tie_embeddings=True,
            
            # Default training features
            use_checkpointing=False,
            use_mixed_precision=True,
            activation="gelu"
        )

###############################################################################
# Core Components - Positional Encodings
###############################################################################
class RotaryPositionalEmbedding(nn.Module):
    """Rotary positional embedding for enhanced position encoding."""
    def __init__(self, dim: int, max_seq_len: int = 4096):
        super().__init__()
        inv_freq = 1. / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        self.max_seq_len = max_seq_len
        self.seq_len_cached = -1
        self.cos_cached = None
        self.sin_cached = None

    def forward(self, x: torch.Tensor, seq_len: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute rotary position embeddings.
        
        Args:
            x: Input tensor to get device information
            seq_len: Sequence length for position embeddings
            
        Returns:
            Tuple of (cos, sin) tensors for rotary embeddings
        """
        if seq_len > self.seq_len_cached:
            self.seq_len_cached = seq_len
            t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
            freqs = torch.einsum('i,j->ij', t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
            self.cos_cached = emb.cos()[None, None, :, :]
            self.sin_cached = emb.sin()[None, None, :, :]
        return self.cos_cached[:, :, :seq_len, :], self.sin_cached[:, :, :seq_len, :]

class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding."""
    def __init__(self, d_model: int, max_len: int = 4096, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input tensor.
        
        Args:
            x: Input tensor (batch_size, seq_len, d_model)
            
        Returns:
            Tensor with positional encoding added
        """
        seq_len = x.size(1)
        return self.dropout(x + self.pe[:, :seq_len])

###############################################################################
# Sparse Multi-Head Attention with Optional KV Caching
###############################################################################
class SparseMultiHeadAttention(nn.Module):
    """
    Multi-Head Attention using a local-window sparse mechanism.
    Optionally supports caching for incremental decoding.
    """
    def __init__(
        self, 
        config: TransformerConfig,
        is_causal: bool = False
    ):
        super().__init__()
        self.config = config
        self.is_causal = is_causal
        
        self.d_k = config.d_model // config.num_heads
        self.w_q = nn.Linear(config.d_model, config.d_model)
        self.w_k = nn.Linear(config.d_model, config.d_model)
        self.w_v = nn.Linear(config.d_model, config.d_model)
        self.out_proj = nn.Linear(config.d_model, config.d_model)
        
        if config.use_rope:
            self.rope = RotaryPositionalEmbedding(self.d_k, config.max_seq_len)
        
        self.dropout = nn.Dropout(config.dropout)

    def _apply_rope(self, x, seq_len):
        # x: (batch, heads, seq_len, head_dim)
        if not self.config.use_rope:
            return x
            
        cos, sin = self.rope(x, seq_len)
        # Apply RoPE rotation
        x_rope = torch.cat([-x[..., 1::2], x[..., ::2]], dim=-1)
        return x * cos + x_rope * sin

    def attn_function(self, Q, K, V, local_mask, attn_mask):
        """
        Perform scaled dot-product attention with support for sparse local window masking 
        and optional additive attention masking.


    Args:
        Q (torch.Tensor): Query tensor of shape (batch_size, num_heads, seq_len, head_dim).
        K (torch.Tensor): Key tensor of shape (batch_size, num_heads, seq_len, head_dim).
        V (torch.Tensor): Value tensor of shape (batch_size, num_heads, seq_len, head_dim).
        local_mask (torch.Tensor): Local window mask tensor of shape (seq_len, seq_len).
                                   Indicates positions to mask out based on a local window.
        attn_mask (Optional[torch.Tensor]): Additive attention mask of shape 
                                            (batch_size, seq_len) or 
                                            (batch_size, seq_len, seq_len).
                                            Used to mask certain tokens or positions 
                                            in the attention computation.
        
        Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            - output (torch.Tensor): The resulting tensor after applying attention,
              of shape (batch_size, num_heads, seq_len, head_dim).
            - scores (torch.Tensor): The normalized attention scores of shape 
              (batch_size, num_heads, seq_len, seq_len).
        
        Notes:
        - This function uses scaled dot-product attention.
        - Handles broadcasting and alignment of attention masks to match `Q`, `K`, and `V`.
        - Supports masking out-of-window positions using `local_mask` and
          masking specific tokens or positions using `attn_mask`.
          
        """
        print(f"Q shape: {Q.shape}, K shape: {K.shape}, V shape: {V.shape}, attn_mask shape: {attn_mask.shape}")
        batch_size, num_heads, seq_len, _ = Q.size()
        local_mask = local_mask.unsqueeze(0).unsqueeze(1).expand(batch_size, num_heads, seq_len, seq_len)  # (1, 1, seq_len, seq_len)

        if attn_mask is not None:
            attn_mask = attn_mask[:, None, None, :].expand(batch_size, num_heads, seq_len, seq_len)
        
        #Calculate attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        if attn_mask is not None:
            scores += attn_mask
        scores = scores + local_mask
        scores = scores.softmax(dim=-1)

        #Compute final attn output
        output = torch.matmul(scores, V)

        return output, scores


    def forward(
        self, 
        q, 
        k, 
        v, 
        attn_mask=None, 
        past_key_value=None, 
        use_cache=False
    ):
        """
        q, k, v: (batch_size, seq_len, d_model)
        attn_mask: e.g. (batch_size, 1, seq_len, seq_len), or broadcastable
        past_key_value: (past_k, past_v) each with shape:
                        (batch_size, num_heads, past_seq_len, d_k)
        use_cache: If True, will store/return new (k, v).

        Returns:
            output: (batch_size, seq_len, d_model)
            next_key_value: Optional[Tuple[Tensor, Tensor]]
        """
        batch_size, seq_len, _ = q.size()
        
        # 1) Project Q, K, V
        Q = self.w_q(q)
        K_ = self.w_k(k)
        V_ = self.w_v(v)

        # 2) Reshape
        Q = Q.view(batch_size, seq_len, self.config.num_heads, self.d_k).transpose(1, 2)
        K_ = K_.view(batch_size, seq_len, self.config.num_heads, self.d_k).transpose(1, 2)
        V_ = V_.view(batch_size, seq_len, self.config.num_heads, self.d_k).transpose(1, 2)

        # 3) If caching, concatenate new K/V with past K/V
        if past_key_value is not None:
            (past_k, past_v) = past_key_value
            # K_ and V_ might correspond to only the "new" tokens
            K_ = torch.cat([past_k, K_], dim=2)  # seq_len dimension
            V_ = torch.cat([past_v, V_], dim=2)
        
        next_key_value = None
        if use_cache:
            # We'll return the entire K_, V_ for next step
            next_key_value = (K_, V_)

        # 4) Build or reuse local window mask
        # The effective sequence length is K_.size(2)
        full_seq_len = K_.size(2)
        local_mask = build_local_window_mask(full_seq_len, self.config.window_size, self.config.global_tokens)

        # 5) Compute attention
        # We'll define a function for checkpoint if needed
        def fn_attention(Q_, K__, V__):
            return self.attn_function(Q_, K__, V__, local_mask, attn_mask)
        
        if self.config.use_checkpointing and Q.requires_grad:
            # For memory efficiency, checkpoint the attention function
            output = checkpoint(fn_attention, Q, K_, V_)
        else:
            output = fn_attention(Q, K_, V_)

        # 6) Reshape back
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.config.d_model)

        # 7) Final linear projection
        output = self.out_proj(output)

        return output, next_key_value

###############################################################################
# Local Window Mask with Optional Global Tokens
###############################################################################
def build_local_window_mask(seq_len: int, window_size: int, global_tokens: int = 0, device: Optional[torch.device] = None) -> torch.Tensor:
    """Build local window mask for attention.
    
    Args:
        seq_len: Length of sequence
        window_size: Size of local attention window
        global_tokens: Number of global tokens that can attend to all positions
        device: Device to create tensor on (defaults to CPU)
    """
    mask = torch.zeros(seq_len, seq_len, dtype=torch.float, device=device)
    for i in range(seq_len):
        for j in range(seq_len):
            if i < global_tokens or j < global_tokens:
                continue
            if abs(i - j) > window_size:
                mask[i, j] = float('-inf')
    return mask

###############################################################################
# Residual + LayerNorm with Optional Gradient Checkpointing
###############################################################################
class ResidualConnection(nn.Module):
    """
    y = x + dropout(sublayer(LN(x)))
    If gradient checkpointing is enabled, wraps the sublayer call in checkpoint().
    """
    def __init__(self, d_model, dropout=0.1, use_checkpointing=False):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.use_checkpointing = use_checkpointing

    def forward(self, x, sublayer):
        """
        sublayer: a callable that takes (normalized_x) -> output
        """
        def forward_sublayer(normed_x):
            return sublayer(normed_x)
        
        normed = self.norm(x)

        if self.use_checkpointing and normed.requires_grad:
            out = x + self.dropout(checkpoint(forward_sublayer, normed))
        else:
            out = x + self.dropout(sublayer(normed))
        return out

###############################################################################
# Feed Forward
###############################################################################
class FeedForward(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.linear1 = nn.Linear(config.d_model, config.d_ff)
        self.linear2 = nn.Linear(config.d_ff, config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        
        # Support different activation functions
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

###############################################################################
# Encoder Block with Sparse Attention + Global Option + Checkpointing
###############################################################################
class EncoderBlock(nn.Module):
    def __init__(
        self, 
        config: TransformerConfig
    ):
        super().__init__()
        self.self_attn_res = ResidualConnection(
            config.d_model, 
            config.dropout, 
            config.use_checkpointing
        )
        self.ff_res = ResidualConnection(
            config.d_model, 
            config.dropout, 
            config.use_checkpointing
        )

        self.self_attn = SparseMultiHeadAttention(config)
        self.feed_forward = FeedForward(config)

    def forward(self, x, src_mask=None, past_key_value=None, use_cache=False):
        """
        x: (batch_size, seq_len, d_model)
        src_mask: optional attention mask
        past_key_value: optional caching
        use_cache: bool
        """
        # Self-attention
        def sa_fn(_x):
            out, next_kv = self.self_attn(_x, _x, _x, attn_mask=src_mask,
                                          past_key_value=past_key_value,
                                          use_cache=use_cache)
            return out, next_kv
        
        # We wrap the call so the residual connection only sees the "output" part
        sa_out, next_kv = sa_fn(x)
        x = self.self_attn_res(x, lambda _x: sa_out)  
        
        # Feed-forward
        x = self.ff_res(x, self.feed_forward)

        return x, next_kv

###############################################################################
# Decoder Block with Sparse Self-Attn + Cross-Attn + Checkpointing + Cache
###############################################################################
class DecoderBlock(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        # 1) Decoder self-attention
        self.self_attn_res = ResidualConnection(
            config.d_model, 
            config.dropout, 
            config.use_checkpointing
        )
        self.self_attn = SparseMultiHeadAttention(config, is_causal=True)

        # 2) Cross-attention
        self.cross_attn_res = ResidualConnection(
            config.d_model, 
            config.dropout, 
            config.use_checkpointing
        )
        self.cross_attn = SparseMultiHeadAttention(config)

        # 3) Feed-forward
        self.ff_res = ResidualConnection(
            config.d_model, 
            config.dropout, 
            config.use_checkpointing
        )
        self.feed_forward = FeedForward(config)

    def forward(
        self, 
        x, 
        encoder_output, 
        tgt_mask=None, 
        src_mask=None, 
        past_key_value=None, 
        use_cache=False
    ):
        """
        x: (batch_size, tgt_seq_len, d_model)
        encoder_output: (batch_size, src_seq_len, d_model)
        tgt_mask: optional mask for decoder self-attention
        src_mask: optional mask for cross-attention
        past_key_value: tuple containing:
           (past_self_k, past_self_v, past_cross_k, past_cross_v) or similar
        use_cache: bool
        """
        # Split the past key/values if given
        (past_self_k, past_self_v, past_cross_k, past_cross_v) = (None, None, None, None)
        if past_key_value is not None:
            past_self_k, past_self_v, past_cross_k, past_cross_v = past_key_value

        # 1) Decoder self-attention
        def self_attn_fn(_x):
            out, next_self_kv = self.self_attn(
                _x, _x, _x,
                attn_mask=tgt_mask,
                past_key_value=(past_self_k, past_self_v),
                use_cache=use_cache
            )
            return out, next_self_kv

        self_sa_out, next_self_kv = self_attn_fn(x)
        x = self.self_attn_res(x, lambda _x: self_sa_out)

        # 2) Cross-attention
        def cross_attn_fn(_x):
            out, next_cross_kv = self.cross_attn(
                _x,
                encoder_output,
                encoder_output,
                attn_mask=src_mask,
                past_key_value=(past_cross_k, past_cross_v),
                use_cache=use_cache
            )
            return out, next_cross_kv

        cross_out, next_cross_kv = cross_attn_fn(x)
        x = self.cross_attn_res(x, lambda _x: cross_out)

        # 3) Feed-forward
        x = self.ff_res(x, self.feed_forward)

        next_key_value = None
        if use_cache:
            next_key_value = (next_self_kv[0], next_self_kv[1],
                              next_cross_kv[0], next_cross_kv[1])

        return x, next_key_value

###############################################################################
# Full Encoder
###############################################################################
class Encoder(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.layers = nn.ModuleList([
            EncoderBlock(config)
            for _ in range(config.num_layers)
        ])
        self.norm = nn.LayerNorm(config.d_model)

    def forward(self, x, src_mask=None, past_key_values=None, use_cache=False):
        """
        x: (batch_size, src_seq_len, d_model)
        src_mask: optional mask
        past_key_values: optional list of (k,v) for each layer
        use_cache: bool
        """
        next_past_key_values = []

        for i, layer in enumerate(self.layers):
            past_kv = past_key_values[i] if (past_key_values is not None) else None
            x, new_kv = layer(x, src_mask=src_mask, past_key_value=past_kv, use_cache=use_cache)
            next_past_key_values.append(new_kv)
        
        x = self.norm(x)
        return x, next_past_key_values if use_cache else None

###############################################################################
# Full Decoder
###############################################################################
class Decoder(nn.Module):
    def __init__(
        self,
        d_model,
        num_heads,
        window_size,
        global_tokens,
        d_ff,
        num_layers,
        dropout=0.1,
        use_checkpointing=False
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            DecoderBlock(
                d_model=d_model, 
                num_heads=num_heads, 
                window_size=window_size, 
                global_tokens=global_tokens, 
                d_ff=d_ff, 
                dropout=dropout,
                use_checkpointing=use_checkpointing
            )
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self, 
        x, 
        encoder_output, 
        tgt_mask=None, 
        src_mask=None, 
        past_key_values=None, 
        use_cache=False
    ):
        """
        x: (batch_size, tgt_seq_len, d_model)
        encoder_output: (batch_size, src_seq_len, d_model)
        tgt_mask: optional mask
        src_mask: optional mask for cross-attention
        past_key_values: optional list of (self_k,v,cross_k,v) for each layer
        use_cache: bool
        """
        next_past_key_values = []
        
        for i, layer in enumerate(self.layers):
            past_kv = past_key_values[i] if (past_key_values is not None) else None
            x, new_kv = layer(
                x, 
                encoder_output, 
                tgt_mask=tgt_mask, 
                src_mask=src_mask,
                past_key_value=past_kv,
                use_cache=use_cache
            )
            next_past_key_values.append(new_kv)

        x = self.norm(x)
        return x, next_past_key_values if use_cache else None

###############################################################################
# Full Encoder–Decoder Transformer with Advanced Training Features
###############################################################################

class Transformer(nn.Module):
    def __init__(
        self,
        config: TransformerConfig
    ):
        super().__init__()
        self.config = config
        
        # Embeddings
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        
        # Positional encoding fallback
        if not config.use_rope:
            self.pos_encoding = PositionalEncoding(
                config.d_model, 
                config.max_seq_len, 
                config.dropout
            )
        
        # Create encoder
        self.encoder = Encoder(config)
        
        # Output projection
        self.generator = nn.Linear(config.d_model, config.vocab_size)
        
        self._reset_parameters()
        
        # Training features
        self.scaler = torch.amp.GradScaler(device ='cuda', enabled=config.use_mixed_precision)
        self.metrics = {'train': {}, 'val': {}}
        self.best_val_loss = float('inf')

    def _reset_parameters(self):
        """Initialize parameters."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        src: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]]]]:
        """
        Forward pass of the transformer.
        
        Args:
            src: Input tensor of shape (batch_size, seq_len)
            attention_mask: Optional attention mask
            past_key_values: Optional cached key/value states for each layer
            use_cache: Whether to return cached key/value states
            
        Returns:
            If use_cache=False: output logits
            If use_cache=True: tuple of (output logits, list of cached key/value states)
        """
        # Embedding + positional encoding
        x = self.embedding(src)
        if not self.config.use_rope:
            x = self.pos_encoding(x)
            
        # Encoder
        encoder_output, next_cache = self.encoder(
            x, 
            attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache
        )
        
        # Generate output
        output = self.generator(encoder_output)
        
        if use_cache:
            return output, next_cache
        return output

    def configure_optimizer(self, config: TransformerConfig):
        """Configure optimizer with weight decay"""
        # Separate weight decay and non-weight decay params
        decay_params = []
        no_decay_params = []
        
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            # Don't apply weight decay to biases and layer norms
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
        """Single training step with mixed precision"""
        self.train()
        metrics = {}
        
        with torch.amp.autocast(enabled=self.config.use_mixed_precision):
            # Forward pass
            outputs = self(
                src=batch['input_ids'],
                tgt=batch['labels'],
                src_mask=batch.get('attention_mask'),
                tgt_mask=self._generate_square_subsequent_mask(batch['labels'].size(1))
            )
            
            loss = self.compute_loss(outputs, batch['labels'])
            metrics['loss'] = loss.item()
            
            # Add regularization if configured
            if self.config.use_regularization:
                reg_loss = self.compute_regularization()
                loss += reg_loss
                metrics['reg_loss'] = reg_loss.item()
        
        # Backward pass with gradient scaling
        self.scaler.scale(loss).backward()
        
        # Gradient clipping
        if self.config.max_grad_norm > 0:
            self.scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(self.parameters(), self.config.max_grad_norm)
        
        self.scaler.step(optimizer)
        self.scaler.update()
        optimizer.zero_grad()
        
        return metrics

    def validation_step(self, batch) -> Dict[str, float]:
        """Validation step with metrics collection"""
        self.eval()
        metrics = {}
        
        with torch.no_grad():
            outputs = self(
                src=batch['input_ids'],
                tgt=batch['labels'],
                src_mask=batch.get('attention_mask'),
                tgt_mask=self._generate_square_subsequent_mask(batch['labels'].size(1))
            )
            
            metrics['loss'] = self.compute_loss(outputs, batch['labels']).item()
            metrics.update(self.compute_metrics(outputs, batch['labels']))
            
        return metrics

    def compute_regularization(self) -> torch.Tensor:
        """Compute regularization loss"""
        reg_loss = 0.0
        
        if self.config.l2_reg > 0:
            for param in self.parameters():
                reg_loss += torch.norm(param, p=2)
            reg_loss *= self.config.l2_reg
            
        return reg_loss

    def compute_metrics(self, outputs, labels) -> Dict[str, float]:
        """Compute evaluation metrics"""
        metrics = {}
        predictions = outputs.argmax(dim=-1)
        
        # Accuracy
        mask = labels != self.config.pad_token_id
        correct = (predictions == labels) & mask
        metrics['accuracy'] = correct.sum().float() / mask.sum()
        
        # Perplexity
        metrics['perplexity'] = torch.exp(self.compute_loss(outputs, labels))
        
        return metrics

    def compute_loss(self, outputs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute cross entropy loss with optional label smoothing.
        
        Args:
            outputs: Model output logits (batch_size, seq_len, vocab_size)
            labels: Target labels (batch_size, seq_len)
            
        Returns:
            Loss tensor
        """
        if self.config.label_smoothing > 0:
            return F.cross_entropy(
                outputs.view(-1, outputs.size(-1)),
                labels.view(-1),
                ignore_index=self.config.pad_token_id,
                label_smoothing=self.config.label_smoothing
            )
        else:
            return F.cross_entropy(
                outputs.view(-1, outputs.size(-1)),
                labels.view(-1),
                ignore_index=self.config.pad_token_id
            )

    @staticmethod
    def get_scheduler(optimizer, config: TransformerConfig, num_training_steps: int):
        """Enhanced learning rate scheduler with warmup"""
        num_warmup_steps = int(num_training_steps * config.warmup_ratio)
        
        if config.scheduler_type == "cosine_warmup":
            return transformers.get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps
            )
        elif config.scheduler_type == "linear_warmup":
            return transformers.get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps
            )
