import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

###############################################################################
# Local Window Mask with Optional Global Tokens
###############################################################################
def build_local_window_mask(
    seq_len: int, 
    window_size: int, 
    global_tokens: int = 0
) -> torch.Tensor:
    """
    Create a (seq_len, seq_len) mask with float('-inf') outside a fixed window.
    Also supports 'global_tokens': for i < global_tokens OR j < global_tokens,
    we allow full attention (i.e., no -inf for those positions).

    If i is the query index and j is the key index:
      - If either i < global_tokens or j < global_tokens, no -inf (global).
      - Else, we allow attention if |i - j| <= window_size.
      - Otherwise, fill with -inf to block attention.

    Returns:
        mask: (seq_len, seq_len) with 0 where allowed, -inf where blocked.
    """
    mask = torch.zeros(seq_len, seq_len, dtype=torch.float)
    for i in range(seq_len):
        for j in range(seq_len):
            # If in global region, let attention be free
            if i < global_tokens or j < global_tokens:
                continue  # no -inf
            # Else, local window
            if abs(i - j) > window_size:
                mask[i, j] = float('-inf')
    return mask

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
        d_model: int, 
        num_heads: int, 
        window_size: int, 
        global_tokens: int = 0,
        dropout: float = 0.1,
        use_checkpointing: bool = False
    ):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.window_size = window_size
        self.global_tokens = global_tokens
        self.dropout = nn.Dropout(dropout)
        self.use_checkpointing = use_checkpointing

        # Projections for Q, K, V
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)

        # Output projection
        self.out_proj = nn.Linear(d_model, d_model)

    def _attn_function(
        self, 
        Q, K, V, 
        local_mask: torch.Tensor, 
        attn_mask: torch.Tensor = None
    ):
        """
        Core attention function used by forward().
        (batch_size, num_heads, seq_len, d_k)
        """
        # Compute scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        # Add local-window mask
        # local_mask: (seq_len, seq_len) => expand to (1,1,seq_len,seq_len)
        scores = scores + local_mask.unsqueeze(0).unsqueeze(0).to(scores.device)

        # Additional mask (e.g., causal or padding)
        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask == 0, float('-inf'))

        # Softmax + dropout
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Weighted sum over V
        output = torch.matmul(attn_weights, V)
        return output

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
        Q = Q.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K_ = K_.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V_ = V_.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

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
        local_mask = build_local_window_mask(full_seq_len, self.window_size, self.global_tokens)

        # 5) Compute attention
        # We'll define a function for checkpoint if needed
        def fn_attention(Q_, K__, V__):
            return self._attn_function(Q_, K__, V__, local_mask, attn_mask)
        
        if self.use_checkpointing and Q.requires_grad:
            # For memory efficiency, checkpoint the attention function
            output = checkpoint(fn_attention, Q, K_, V_)
        else:
            output = fn_attention(Q, K_, V_)

        # 6) Reshape back
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)

        # 7) Final linear projection
        output = self.out_proj(output)

        return output, next_key_value

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
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        x = self.linear1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x

###############################################################################
# Encoder Block with Sparse Attention + Global Option + Checkpointing
###############################################################################
class EncoderBlock(nn.Module):
    def __init__(
        self, 
        d_model, 
        num_heads, 
        window_size, 
        global_tokens=0, 
        d_ff=2048, 
        dropout=0.1, 
        use_checkpointing=False
    ):
        super().__init__()
        self.self_attn_res = ResidualConnection(d_model, dropout, use_checkpointing)
        self.ff_res = ResidualConnection(d_model, dropout, use_checkpointing)

        self.self_attn = SparseMultiHeadAttention(
            d_model=d_model,
            num_heads=num_heads,
            window_size=window_size,
            global_tokens=global_tokens,
            dropout=dropout,
            use_checkpointing=use_checkpointing
        )
        self.feed_forward = FeedForward(d_model, d_ff, dropout)

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
    def __init__(
        self, 
        d_model, 
        num_heads, 
        window_size, 
        global_tokens=0, 
        d_ff=2048, 
        dropout=0.1, 
        use_checkpointing=False
    ):
        super().__init__()
        # 1) Decoder self-attention
        self.self_attn_res = ResidualConnection(d_model, dropout, use_checkpointing)
        self.self_attn = SparseMultiHeadAttention(
            d_model=d_model,
            num_heads=num_heads,
            window_size=window_size,
            global_tokens=global_tokens,
            dropout=dropout,
            use_checkpointing=use_checkpointing
        )

        # 2) Cross-attention
        self.cross_attn_res = ResidualConnection(d_model, dropout, use_checkpointing)
        self.cross_attn = SparseMultiHeadAttention(
            d_model=d_model,
            num_heads=num_heads,
            window_size=window_size,
            global_tokens=global_tokens,
            dropout=dropout,
            use_checkpointing=use_checkpointing
        )

        # 3) Feed-forward
        self.ff_res = ResidualConnection(d_model, dropout, use_checkpointing)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)

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
            EncoderBlock(
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
# Positional Encoding
###############################################################################
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=4096, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * 
                             (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)  # shape: (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        x: (batch_size, seq_len, d_model)
        """
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :].to(x.device)
        return self.dropout(x)

###############################################################################
# Full Encoder–Decoder Transformer with Sparse Attention + Checkpointing + Cache
###############################################################################
class Transformer(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        tgt_vocab_size,
        d_model=512,
        num_heads=8,
        window_size=4,
        global_tokens=0,
        d_ff=2048,
        num_layers=6,
        max_seq_len=1024,
        dropout=0.1,
        use_checkpointing=False
    ):
        super().__init__()
        self.d_model = d_model

        # Embeddings
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)

        # Positional encodings
        self.src_pos = PositionalEncoding(d_model, max_seq_len, dropout)
        self.tgt_pos = PositionalEncoding(d_model, max_seq_len, dropout)
        
        # Encoder / Decoder
        self.encoder = Encoder(
            d_model=d_model,
            num_heads=num_heads,
            window_size=window_size,
            global_tokens=global_tokens,
            d_ff=d_ff,
            num_layers=num_layers,
            dropout=dropout,
            use_checkpointing=use_checkpointing
        )
        self.decoder = Decoder(
            d_model=d_model,
            num_heads=num_heads,
            window_size=window_size,
            global_tokens=global_tokens,
            d_ff=d_ff,
            num_layers=num_layers,
            dropout=dropout,
            use_checkpointing=use_checkpointing
        )
        
        # Final projection
        self.generator = nn.Linear(d_model, tgt_vocab_size)

        # Initialize
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def encode(self, src, src_mask=None, past_key_values=None, use_cache=False):
        # (batch_size, src_seq_len) => embed => position => encoder
        x = self.src_embedding(src) * math.sqrt(self.d_model)
        x = self.src_pos(x)
        out, next_keys = self.encoder(x, src_mask, past_key_values, use_cache)
        return out, next_keys

    def decode(
        self, 
        tgt, 
        memory, 
        tgt_mask=None, 
        src_mask=None, 
        past_key_values=None, 
        use_cache=False
    ):
        # (batch_size, tgt_seq_len)
        x = self.tgt_embedding(tgt) * math.sqrt(self.d_model)
        x = self.tgt_pos(x)
        out, next_keys = self.decoder(x, memory, tgt_mask, src_mask, past_key_values, use_cache)
        return out, next_keys

    def forward(
        self, 
        src, 
        tgt, 
        src_mask=None, 
        tgt_mask=None, 
        past_key_values_enc=None,
        past_key_values_dec=None,
        use_cache=False
    ):
        """
        src, tgt: (batch_size, seq_len)
        src_mask, tgt_mask: optional attention masks
        past_key_values_enc/dec: lists of key-value pairs for caching
        use_cache: bool -> whether to return new key-values
        """
        # 1) Encode
        memory, enc_next = self.encode(
            src, 
            src_mask=src_mask, 
            past_key_values=past_key_values_enc, 
            use_cache=use_cache
        )
        # 2) Decode
        hidden, dec_next = self.decode(
            tgt, 
            memory,
            tgt_mask=tgt_mask, 
            src_mask=src_mask,
            past_key_values=past_key_values_dec,
            use_cache=use_cache
        )
        # 3) Project to vocab
        logits = self.generator(hidden)
        return logits, enc_next, dec_next
