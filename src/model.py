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
        d_model: int,  # hidden_dim in config
        num_layers: int,
        num_heads: int,
        d_ff: int,  # ff_dim in config
        max_seq_len: int,
        dropout: float,
        
        # Attention Mechanisms
        use_rope: bool = True,
        window_size: int = 4,
        global_tokens: int = 0,
        
        # Architecture Options
        prenorm: bool = True,
        tie_embeddings: bool = False,
        
        # Training Features
        use_checkpointing: bool = False,
        use_regularization: bool = False,
        use_mixed_precision: bool = True,
        label_smoothing: float = 0.0,
        l2_reg: float = 0.0,
        max_grad_norm: float = 1.0,
        
        # Optimization
        learning_rate: float = 3e-4,
        weight_decay: float = 0.01,
        warmup_ratio: float = 0.1,
        scheduler_type: str = "linear",
        
        # Special Tokens
        pad_token_id: int = 0,
        
        # Model Behavior
        activation: str = "gelu",  # Options: "gelu", "relu", "silu"
        
        # Device
        device: Optional[torch.device] = None,
        
        # Added this parameter
        use_reentrant: bool = False
    ):
        """Initialize transformer configuration with validation."""
        # Validate core architecture parameters
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        assert activation in ["gelu", "relu", "silu"], f"Unsupported activation: {activation}"
        
        # Model Architecture
        self.vocab_size = vocab_size
        self.d_model = d_model  # This is hidden_dim from config
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_ff = d_ff  # This is ff_dim from config
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

        # Validate window size
        assert window_size > 0, "Window size must be positive"
        assert window_size <= max_seq_len, "Window size cannot exceed max sequence length"
        assert global_tokens >= 0, "Number of global tokens cannot be negative"
        assert global_tokens <= max_seq_len, "Number of global tokens cannot exceed max sequence length"

        # Device
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Added this line
        self.use_reentrant = use_reentrant

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
        if seq_len > self.max_seq_len:
            raise ValueError(f"Sequence length {seq_len} exceeds maximum allowed length {self.max_seq_len}")
        
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
        """
        batch_size, num_heads, seq_len, _ = Q.size()
        
        # Ensure local_mask is on the correct device and expand
        local_mask = local_mask.to(Q.device)
        local_mask = local_mask.unsqueeze(0).unsqueeze(1).expand(batch_size, num_heads, seq_len, seq_len)

        # Calculate attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        # Handle attention mask if provided
        if attn_mask is not None:
            attn_mask = attn_mask.to(Q.device)
            # Handle different input mask dimensions
            if attn_mask.dim() == 2:  # [batch_size, seq_len]
                attn_mask = attn_mask.unsqueeze(1).unsqueeze(2)
            elif attn_mask.dim() == 3:  # [batch_size, 1, seq_len]
                attn_mask = attn_mask.unsqueeze(1)
            # Expand to [batch_size, num_heads, seq_len, seq_len]
            attn_mask = attn_mask.expand(batch_size, num_heads, seq_len, seq_len)
            # Apply mask by setting masked positions to -inf
            scores = scores.masked_fill(~attn_mask.bool(), float('-inf'))

        # Apply local window mask
        scores = scores + local_mask
        
        # Apply softmax
        scores = scores.softmax(dim=-1)

        # Compute final attn output
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
        # Validate input shapes
        batch_size, seq_len, d_model = q.size()
        assert d_model == self.config.d_model, f"Input dimension {d_model} doesn't match config dimension {self.config.d_model}"
        assert k.size(0) == batch_size and v.size(0) == batch_size, "Batch sizes must match"
        assert k.size(-1) == d_model and v.size(-1) == d_model, "Hidden dimension must match"
        
        # 1) Project Q, K, V
        Q = self.w_q(q)
        K_ = self.w_k(k)
        V_ = self.w_v(v)

        # 2) Reshape
        Q = Q.view(batch_size, seq_len, self.config.num_heads, self.d_k).transpose(1, 2)
        K_ = K_.view(batch_size, seq_len, self.config.num_heads, self.d_k).transpose(1, 2)
        V_ = V_.view(batch_size, seq_len, self.config.num_heads, self.d_k).transpose(1, 2)

        # Apply RoPE if configured
        if self.config.use_rope:
            Q = self._apply_rope(Q, seq_len)
            K_ = self._apply_rope(K_, K_.size(2))

        # 3) Handle caching with validation
        if past_key_value is not None:
            try:
                past_k, past_v = past_key_value
                # Validate cache shapes
                assert past_k.size(0) == batch_size, "Cache batch size must match"
                assert past_k.size(1) == self.config.num_heads, "Cache heads must match"
                assert past_k.size(-1) == self.d_k, "Cache dimension must match"
                assert past_k.size() == past_v.size(), "Cache K,V shapes must match"
                
                K_ = torch.cat([past_k, K_], dim=2)  # seq_len dimension
                V_ = torch.cat([past_v, V_], dim=2)
            except Exception as e:
                raise RuntimeError(f"Error processing cached KV: {str(e)}")
        
        next_key_value = None
        if use_cache:
            next_key_value = (K_, V_)

        # 4) Build or reuse local window mask
        full_seq_len = K_.size(2)
        local_mask = build_local_window_mask(full_seq_len, self.config.window_size, self.config.global_tokens, self.is_causal, self.config.device)

        # 5) Compute attention with proper error handling
        def fn_attention(Q_, K__, V__):
            try:
                output, _ = self.attn_function(Q_, K__, V__, local_mask, attn_mask)
                return output
            except RuntimeError as e:
                if "out of memory" in str(e):
                    torch.cuda.empty_cache()
                    raise RuntimeError("GPU out of memory. Try reducing batch size or sequence length.")
                raise e
        
        if self.config.use_checkpointing and Q.requires_grad:
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
def build_local_window_mask(
    seq_len: int, 
    window_size: int, 
    global_tokens: int = 0, 
    is_causal: bool = False,
    device: Optional[torch.device] = None
) -> torch.Tensor:
    """Build local window mask for attention."""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mask = torch.zeros(seq_len, seq_len, dtype=torch.float, device=device)
    
    for i in range(seq_len):
        for j in range(seq_len):
            if i < global_tokens or j < global_tokens:
                continue
            if abs(i - j) > window_size:
                mask[i, j] = float('-inf')
            # Add causal masking if needed
            if is_causal and j > i:
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
    def __init__(self, d_model, dropout=0.1, use_checkpointing=False, use_reentrant=False):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.use_checkpointing = use_checkpointing
        self.use_reentrant = use_reentrant

    def forward(self, x, sublayer):
        """
        sublayer: a callable that takes (normalized_x) -> output
        """
        def forward_sublayer(normed_x):
            return sublayer(normed_x)
        
        normed = self.norm(x)

        if self.use_checkpointing and normed.requires_grad:
            out = x + self.dropout(checkpoint(
                forward_sublayer, 
                normed,
                use_reentrant=self.use_reentrant
            ))
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
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.self_attn_res = ResidualConnection(
            config.d_model, 
            config.dropout, 
            config.use_checkpointing,
            config.use_reentrant
        )
        self.ff_res = ResidualConnection(
            config.d_model, 
            config.dropout, 
            config.use_checkpointing,
            config.use_reentrant
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
        next_kv = None

        def sa_sublayer(normed_x):
            nonlocal next_kv
            out, new_kv = self.self_attn(
                normed_x, normed_x, normed_x, 
                attn_mask = src_mask,
                past_key_value=past_key_value,
                use_cache=use_cache

            )
            next_kv = new_kv
            return out
        
        x = self.self_attn_res(x, sa_sublayer)

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
            config.use_checkpointing,
            config.use_reentrant
        )
        self.self_attn = SparseMultiHeadAttention(config, is_causal=True)

        # 2) Cross-attention
        self.cross_attn_res = ResidualConnection(
            config.d_model, 
            config.dropout, 
            config.use_checkpointing,
            config.use_reentrant
        )
        self.cross_attn = SparseMultiHeadAttention(config)

        # 3) Feed-forward
        self.ff_res = ResidualConnection(
            config.d_model, 
            config.dropout, 
            config.use_checkpointing,
            config.use_reentrant
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
        (past_self_k, past_self_v, past_cross_k, past_cross_v) = (None, None, None, None)
        if past_key_value is not None:
            past_self_k, past_self_v, past_cross_k, past_cross_v = past_key_value
        
        # 1) Decoder self_attn (on normed_x)
        next_self_kv = None
        def self_attn_sublayer(normed_x):
            nonlocal next_self_kv
            out, new_self_kv = self.self_attn(
                    normed_x, normed_x, normed_x,
                    attn_mask = tgt_mask,
                    past_key_value = (past_self_k, past_self_v),
                    use_cache=use_cache
            )
            next_self_kv = new_self_kv
            return out
        
        x = self.self_attn_res(x, self_attn_sublayer)
        # 2) Cross-attention (on normed_X)
        next_cross_kv = None
        def cross_attn_sublayer(normed_x):
            nonlocal next_cross_kv
            out, new_cross_kv = self.cross_attn(
                normed_x,
                encoder_output,
                encoder_output,
                attn_mask = src_mask,
                past_key_value = (past_cross_k, past_cross_v),
                use_cache = use_cache
            )
            next_cross_kv = new_cross_kv
            return out
        x = self.cross_attn_res(x, cross_attn_sublayer)

        # 3) Feed-forward
        x = self.ff_res(x, self.feed_forward)

        next_key_value = None
        if use_cache:
            next_key_value = (
                next_self_kv[0], next_self_kv[1],
                next_cross_kv[0], next_cross_kv[1]
            )
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
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.layers = nn.ModuleList([
            DecoderBlock(config)
            for _ in range(config.num_layers)
        ])
        self.norm = nn.LayerNorm(config.d_model)

    def forward(
        self, 
        x, 
        encoder_output, 
        tgt_mask=None, 
        src_mask=None, 
        past_key_values = None,
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
        self.decoder = Decoder(config)
        
        # Output projection
        self.generator = nn.Linear(config.d_model, config.vocab_size)
        if config.tie_embeddings:
            self.generator.weight = self.embedding.weight
        
        self._reset_parameters()
        
        # Training features
        self.scaler = torch.amp.GradScaler(
            enabled=config.use_mixed_precision
        )
        self.metrics = {
            'train': {'loss': [], 'accuracy': [], 'perplexity': []},
            'val': {'loss': [], 'accuracy': [], 'perplexity': []}
        }
        self.best_val_loss = float('inf')

    def _reset_parameters(self):
        """Initialize parameters."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
            self, 
            src: torch.Tensor,
            tgt: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            tgt_mask: Optional[torch.Tensor] = None,
            past_key_values: Optional[List[Tuple]] = None,
            use_cache: bool = False
    ):
        """
          Args:
            src: shape (batch_size, src_seq_len)
            tgt: shape (batch_size, tgt_seq_len), optional
            attention_mask: Typically a padding mask for the encoder (batch_size, src_seq_len)
            tgt_mask: Typically a causal mask for the decoder (batch_size, tgt_seq_len, tgt_seq_len)
            past_key_values: Optional list of cached key/value states for each layer
            use_cache: If True, return new cache
        """

        # 1) Encoder the source (bidirectional, like BERT)
        enc_inp = self.embedding(src)
        if not self.config.use_rope:
            enc_inp = self.pos_encoding(enc_inp)
        
        #Pass the appropriate "past_key_values" to the encoder if using caching.
        encoder_output, enc_past = self.encoder(
            enc_inp,
            src_mask = attention_mask,
            past_key_values= past_key_values if past_key_values else None,
            use_cache=use_cache
        )

        # 2) If tgt is provided, run the decoder (causal, like GPT)
        if tgt is not None:
            dec_inp = self.embedding(tgt)
            if not self.config.use_rope:
                dec_inp = self.pos_encoding(dec_inp)
            
            # Pass the decoder portion of past_key_values if you want caching in the decoder
            decoder_outputs, dec_past = self.decoder(
                dec_inp,
                encoder_output,
                tgt_mask = tgt_mask,
                src_mask = attention_mask,
                past_key_values=(past_key_values) if past_key_values else None,
                use_cache=use_cache
            )

            logits = self.generator(decoder_outputs)
            if logits.size(-1) != self.config.vocab_size:
                self.logger.warning(
                    f"Logits last dimension {logits.size(-1)} does not match config.vocab_size {self.config.vocab_size}. Slicing logits."
                )
                logits = logits[..., :self.config.vocab_size]
            logits = torch.clamp(logits, min=-1e9, max=1e9)


            #If using cache, return both the logits and the new cache
            if use_cache:
                return logits, (enc_past, dec_past)
            return logits
        
        #3) If no tgt is provided, return something for encoder-only usage.
        return encoder_output

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
                tgt_mask=self._generate_square_subsequent_mask(batch['labels'].size(1)))
            
            loss = self.compute_loss(outputs, batch['labels'])
            metrics['loss'] = loss.item()
            
            # Add regularization if configured
            if self.config.use_regularization:
                reg_loss = self.compute_regularization()
                loss += reg_loss
                metrics['reg_loss'] = reg_loss.item()
        
        # Backward pass with gradient scaling
        self.scaler.scale(loss).backward()
        
        self.scaler.unscale_(optimizer) #Unscale gradients before clipping.
        # Gradient clipping
        if self.config.max_grad_norm > 0:
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
        
        denominator = mask.sum().float().clamp(min=1)  # Prevent divide-by-zero
        metrics['accuracy'] = correct.sum().float() / denominator
        
        # Perplexity
        metrics['perplexity'] = torch.exp(self.compute_loss(outputs, labels))
        
        return metrics

    def compute_loss(self, outputs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute cross entropy loss with input validation."""
        # Validate shapes and values
        if outputs.dim() != 3:
            raise ValueError(f"ERROR: Expected outputs shape (batch_size, seq_len, vocab_size), but got {outputs.shape}")  
        if labels.dim() != 2:
            raise ValueError(f"ERROR: Expected labels shape (batch_size, seq_len), but got {labels.shape}")
        
        # Ensure labels are within valid range
        if torch.any(labels >= self.config.vocab_size):
            self.logger.warning(f"Labels Exceeded vocab size! Clamping values")
            labels = torch.clamp(labels, 0, self.config.vocab_size -1)
       
        # Ensure labels do not contain negative values (except -100 for ignore_index)
        if torch.any((labels < 0) & (labels != -100)):
            raise ValueError(f" ERROR: Found invalid negative label values! Min label: {labels.min().item()}")
        
        # Flatten tensors for loss calculation
        flatten_outputs = outputs.view(-1, outputs.size(-1))
        flat_labels = labels.view(-1)

        # Retrieve label smoothing parameter (default to 0 if missing)
        label_smoothing = getattr(self.config, "label_smoothing", 0.0)

        #Compute Loss with optional label smoothing
        loss = F.cross_entropy(
            flatten_outputs,
            flat_labels,
            ignore_index=self.config.pad_token_id,
            label_smoothing=label_smoothing
        )
        return loss
        
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

    @staticmethod
    def _generate_square_subsequent_mask(sz: int) -> torch.Tensor:
        """Generate a square mask for the sequence. The masked positions are filled with float('-inf').
        Unmasked positions are filled with float(0.0).
        """
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
