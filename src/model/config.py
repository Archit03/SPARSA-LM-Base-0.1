"""
SPARSA-LM Model Configuration
LLaMA-like AutoRegressive Language Model Architecture Specification
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict
import json
import math


@dataclass
class ModelConfig:
    """
    Configuration for the AutoRegressive Language Model.

    Architecture: LLaMA-like Decoder-Only Transformer
    ================================================

    Key Components:
    1. Token Embeddings - vocab_size x hidden_size
    2. Transformer Blocks (num_hidden_layers)
       - Pre-RMSNorm
       - Grouped Query Attention (GQA) with RoPE
       - Residual Connection
       - Pre-RMSNorm
       - SwiGLU Feed-Forward (gate_proj, up_proj, down_proj)
       - Residual Connection
    3. Final RMSNorm
    4. LM Head (tied with embeddings by default)

    Parameter Calculation:
    =====================
    - Embeddings: vocab_size * hidden_size
    - Per Layer:
      - Attention: 4 * hidden_size^2 (q, k, v, o projections with GQA)
      - FFN: 3 * hidden_size * intermediate_size (gate, up, down)
      - Norms: 2 * hidden_size
    - Total per layer: ~12 * hidden_size^2 (for typical 4x FFN expansion)
    """

    # ==========================================================================
    # Core Model Dimensions
    # ==========================================================================

    vocab_size: int = 32000
    hidden_size: int = 1024           # d_model
    intermediate_size: int = 4096     # FFN hidden dim (typically 4x hidden_size)
    num_hidden_layers: int = 28       # Number of transformer blocks
    num_attention_heads: int = 16     # Query heads
    num_key_value_heads: int = 8      # KV heads for GQA (< num_attention_heads)

    # ==========================================================================
    # Sequence Configuration
    # ==========================================================================

    max_position_embeddings: int = 2048  # Maximum sequence length
    sliding_window: Optional[int] = None  # Sliding window attention (None = full)

    # ==========================================================================
    # Normalization
    # ==========================================================================

    rms_norm_eps: float = 1e-6        # RMSNorm epsilon
    norm_type: str = "rmsnorm"        # "rmsnorm" or "layernorm"

    # ==========================================================================
    # Activation Function
    # ==========================================================================

    hidden_act: str = "silu"          # SiLU (Swish) for SwiGLU
    mlp_type: str = "swiglu"          # "swiglu", "gelu", or "glu"

    # ==========================================================================
    # Attention Configuration
    # ==========================================================================

    attention_dropout: float = 0.0    # Attention dropout (0 for modern LLMs)
    attention_bias: bool = False      # Whether to use bias in attention projections

    # ==========================================================================
    # Rotary Position Embeddings (RoPE)
    # ==========================================================================

    rope_theta: float = 10000.0       # RoPE base frequency
    rope_scaling: Optional[Dict] = None  # For extended context (e.g., {"type": "linear", "factor": 2.0})
    rope_partial_factor: float = 1.0  # Fraction of head_dim to apply RoPE to

    # ==========================================================================
    # Initialization
    # ==========================================================================

    initializer_range: float = 0.02   # Std for weight initialization
    residual_scale: float = 1.0       # Scale factor for residual connections

    # ==========================================================================
    # Regularization
    # ==========================================================================

    hidden_dropout: float = 0.0       # Hidden state dropout
    embedding_dropout: float = 0.0    # Embedding dropout

    # ==========================================================================
    # Optimization Features
    # ==========================================================================

    use_flash_attention: bool = True         # Use Flash Attention 2
    use_triton_kernels: bool = True          # Use custom Triton kernels
    use_gradient_checkpointing: bool = False  # Activation checkpointing
    use_memory_efficient_attention: bool = True

    # ==========================================================================
    # Weight Tying
    # ==========================================================================

    tie_word_embeddings: bool = True  # Tie input/output embeddings

    # ==========================================================================
    # Special Token IDs
    # ==========================================================================

    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2
    unk_token_id: int = 3

    # ==========================================================================
    # Model Identification
    # ==========================================================================

    model_type: str = "sparsa_lm"
    architectures: List[str] = field(default_factory=lambda: ["AutoRegressiveLM"])

    def __post_init__(self):
        """Validate configuration after initialization."""
        # Validate head dimensions
        assert self.hidden_size % self.num_attention_heads == 0, \
            f"hidden_size ({self.hidden_size}) must be divisible by num_attention_heads ({self.num_attention_heads})"
        assert self.num_attention_heads % self.num_key_value_heads == 0, \
            f"num_attention_heads ({self.num_attention_heads}) must be divisible by num_key_value_heads ({self.num_key_value_heads})"

        # Validate FFN dimension (typically 8/3 * hidden_size for SwiGLU)
        if self.intermediate_size == 4096 and self.hidden_size != 1024:
            pass  # Allow override

    # ==========================================================================
    # Computed Properties
    # ==========================================================================

    @property
    def head_dim(self) -> int:
        """Dimension of each attention head."""
        return self.hidden_size // self.num_attention_heads

    @property
    def num_kv_groups(self) -> int:
        """Number of key-value groups for GQA."""
        return self.num_attention_heads // self.num_key_value_heads

    @property
    def num_parameters(self) -> int:
        """Estimate total number of parameters."""
        # Embeddings (possibly tied)
        embed_params = self.vocab_size * self.hidden_size

        # Per-layer parameters
        # Attention: Q + K + V + O projections
        # With GQA: Q has full heads, K/V have reduced heads
        q_params = self.hidden_size * self.hidden_size
        kv_params = 2 * self.hidden_size * (self.num_key_value_heads * self.head_dim)
        o_params = self.hidden_size * self.hidden_size
        attn_params = q_params + kv_params + o_params

        # FFN: gate, up, down projections for SwiGLU
        ffn_params = 3 * self.hidden_size * self.intermediate_size

        # Layer norms: 2 per layer
        norm_params = 2 * self.hidden_size

        # Total per layer
        layer_params = attn_params + ffn_params + norm_params

        # Final norm + LM head (if not tied)
        final_params = self.hidden_size  # Final RMSNorm
        if not self.tie_word_embeddings:
            final_params += self.vocab_size * self.hidden_size

        total = embed_params + (self.num_hidden_layers * layer_params) + final_params
        return total

    @property
    def num_parameters_millions(self) -> float:
        """Parameters in millions."""
        return self.num_parameters / 1_000_000

    @property
    def flops_per_token(self) -> int:
        """Approximate FLOPs per token for forward pass."""
        # Attention: 4 * seq_len * hidden_size^2 per layer
        # FFN: 8 * hidden_size * intermediate_size per layer (SwiGLU has 3 projections but gating)
        attn_flops = 4 * self.hidden_size * self.hidden_size
        ffn_flops = 8 * self.hidden_size * self.intermediate_size
        layer_flops = attn_flops + ffn_flops
        total_flops = self.num_hidden_layers * layer_flops
        return total_flops

    # ==========================================================================
    # Serialization
    # ==========================================================================

    def to_dict(self) -> dict:
        """Convert configuration to dictionary."""
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}

    def save(self, path: str) -> None:
        """Save configuration to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_dict(cls, config_dict: dict) -> "ModelConfig":
        """Create configuration from dictionary."""
        return cls(**{k: v for k, v in config_dict.items() if k in cls.__dataclass_fields__})

    @classmethod
    def load(cls, path: str) -> "ModelConfig":
        """Load configuration from JSON file."""
        with open(path, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)

    # ==========================================================================
    # Preset Configurations
    # ==========================================================================

    @classmethod
    def smollm_135m(cls) -> "ModelConfig":
        """SmolLM-135M configuration."""
        return cls(
            vocab_size=49152,
            hidden_size=576,
            intermediate_size=1536,
            num_hidden_layers=30,
            num_attention_heads=9,
            num_key_value_heads=3,
            max_position_embeddings=2048,
        )

    @classmethod
    def smollm_360m(cls) -> "ModelConfig":
        """SmolLM-360M configuration."""
        return cls(
            vocab_size=49152,
            hidden_size=960,
            intermediate_size=2560,
            num_hidden_layers=32,
            num_attention_heads=15,
            num_key_value_heads=5,
            max_position_embeddings=2048,
        )

    @classmethod
    def smollm_1_7b(cls) -> "ModelConfig":
        """SmolLM-1.7B configuration."""
        return cls(
            vocab_size=49152,
            hidden_size=2048,
            intermediate_size=8192,
            num_hidden_layers=24,
            num_attention_heads=32,
            num_key_value_heads=32,
            max_position_embeddings=2048,
        )

    @classmethod
    def llama_7b(cls) -> "ModelConfig":
        """LLaMA-7B configuration."""
        return cls(
            vocab_size=32000,
            hidden_size=4096,
            intermediate_size=11008,
            num_hidden_layers=32,
            num_attention_heads=32,
            num_key_value_heads=32,
            max_position_embeddings=2048,
        )

    @classmethod
    def llama2_7b(cls) -> "ModelConfig":
        """LLaMA-2-7B configuration (with GQA)."""
        return cls(
            vocab_size=32000,
            hidden_size=4096,
            intermediate_size=11008,
            num_hidden_layers=32,
            num_attention_heads=32,
            num_key_value_heads=32,  # LLaMA-2 7B uses MHA, not GQA
            max_position_embeddings=4096,
        )

    @classmethod
    def llama3_8b(cls) -> "ModelConfig":
        """LLaMA-3-8B configuration."""
        return cls(
            vocab_size=128256,
            hidden_size=4096,
            intermediate_size=14336,
            num_hidden_layers=32,
            num_attention_heads=32,
            num_key_value_heads=8,  # GQA with 8 KV heads
            max_position_embeddings=8192,
            rope_theta=500000.0,
        )

    @classmethod
    def mistral_7b(cls) -> "ModelConfig":
        """Mistral-7B configuration."""
        return cls(
            vocab_size=32000,
            hidden_size=4096,
            intermediate_size=14336,
            num_hidden_layers=32,
            num_attention_heads=32,
            num_key_value_heads=8,  # GQA
            max_position_embeddings=32768,
            sliding_window=4096,
            rope_theta=10000.0,
        )

    @classmethod
    def small(cls) -> "ModelConfig":
        """Small model configuration (~125M parameters)."""
        return cls(
            vocab_size=32000,
            hidden_size=768,
            intermediate_size=2048,
            num_hidden_layers=12,
            num_attention_heads=12,
            num_key_value_heads=4,
            max_position_embeddings=2048,
        )

    @classmethod
    def base(cls) -> "ModelConfig":
        """Base model configuration (~360M parameters)."""
        return cls(
            vocab_size=32000,
            hidden_size=1024,
            intermediate_size=4096,
            num_hidden_layers=28,
            num_attention_heads=16,
            num_key_value_heads=8,
            max_position_embeddings=2048,
        )

    @classmethod
    def large(cls) -> "ModelConfig":
        """Large model configuration (~760M parameters)."""
        return cls(
            vocab_size=32000,
            hidden_size=1536,
            intermediate_size=6144,
            num_hidden_layers=32,
            num_attention_heads=24,
            num_key_value_heads=8,
            max_position_embeddings=2048,
        )

    @classmethod
    def xl(cls) -> "ModelConfig":
        """XL model configuration (~1.5B parameters)."""
        return cls(
            vocab_size=32000,
            hidden_size=2048,
            intermediate_size=8192,
            num_hidden_layers=36,
            num_attention_heads=32,
            num_key_value_heads=8,
            max_position_embeddings=4096,
        )

    def print_architecture(self):
        """Print detailed architecture specification."""
        print("=" * 70)
        print("SPARSA-LM ARCHITECTURE SPECIFICATION")
        print("=" * 70)
        print(f"\nModel Type: {self.model_type}")
        print(f"Estimated Parameters: {self.num_parameters_millions:.1f}M")
        print(f"FLOPs per Token: {self.flops_per_token:,}")

        print(f"\n{'─' * 70}")
        print("DIMENSIONS")
        print(f"{'─' * 70}")
        print(f"  Vocabulary Size:     {self.vocab_size:,}")
        print(f"  Hidden Size:         {self.hidden_size}")
        print(f"  Intermediate Size:   {self.intermediate_size}")
        print(f"  Num Layers:          {self.num_hidden_layers}")
        print(f"  Num Attention Heads: {self.num_attention_heads}")
        print(f"  Num KV Heads (GQA):  {self.num_key_value_heads}")
        print(f"  Head Dimension:      {self.head_dim}")
        print(f"  KV Groups:           {self.num_kv_groups}")

        print(f"\n{'─' * 70}")
        print("SEQUENCE")
        print(f"{'─' * 70}")
        print(f"  Max Position Embeddings: {self.max_position_embeddings}")
        print(f"  Sliding Window:          {self.sliding_window or 'None (full attention)'}")

        print(f"\n{'─' * 70}")
        print("COMPONENTS")
        print(f"{'─' * 70}")
        print(f"  Normalization:       {self.norm_type} (eps={self.rms_norm_eps})")
        print(f"  Activation:          {self.hidden_act} ({self.mlp_type})")
        print(f"  Position Encoding:   RoPE (theta={self.rope_theta})")
        print(f"  Attention Bias:      {self.attention_bias}")
        print(f"  Tie Embeddings:      {self.tie_word_embeddings}")

        print(f"\n{'─' * 70}")
        print("OPTIMIZATION")
        print(f"{'─' * 70}")
        print(f"  Flash Attention:     {self.use_flash_attention}")
        print(f"  Triton Kernels:      {self.use_triton_kernels}")
        print(f"  Gradient Checkpoint: {self.use_gradient_checkpointing}")

        print(f"\n{'─' * 70}")
        print("LAYER BREAKDOWN")
        print(f"{'─' * 70}")
        print("  Each Transformer Block:")
        print(f"    1. RMSNorm (hidden_size={self.hidden_size})")
        print(f"    2. Grouped Query Attention")
        print(f"       - Q: {self.hidden_size} -> {self.num_attention_heads} x {self.head_dim}")
        print(f"       - K: {self.hidden_size} -> {self.num_key_value_heads} x {self.head_dim}")
        print(f"       - V: {self.hidden_size} -> {self.num_key_value_heads} x {self.head_dim}")
        print(f"       - O: {self.num_attention_heads * self.head_dim} -> {self.hidden_size}")
        print(f"       - RoPE applied to Q, K")
        print(f"    3. Residual Connection")
        print(f"    4. RMSNorm (hidden_size={self.hidden_size})")
        print(f"    5. SwiGLU Feed-Forward")
        print(f"       - gate_proj: {self.hidden_size} -> {self.intermediate_size}")
        print(f"       - up_proj:   {self.hidden_size} -> {self.intermediate_size}")
        print(f"       - down_proj: {self.intermediate_size} -> {self.hidden_size}")
        print(f"       - output = down(silu(gate) * up)")
        print(f"    6. Residual Connection")
        print("=" * 70)


# =============================================================================
# MODEL ARCHITECTURE COMPARISON TABLE
# =============================================================================

MODEL_COMPARISON = """
╔══════════════════╦═══════════╦════════════╦═══════════╦═════════╦══════════╦══════════╗
║ Model            ║ Params    ║ Hidden     ║ Layers    ║ Heads   ║ KV Heads ║ Context  ║
╠══════════════════╬═══════════╬════════════╬═══════════╬═════════╬══════════╬══════════╣
║ SmolLM-135M      ║ 135M      ║ 576        ║ 30        ║ 9       ║ 3        ║ 2048     ║
║ SmolLM-360M      ║ 360M      ║ 960        ║ 32        ║ 15      ║ 5        ║ 2048     ║
║ SmolLM-1.7B      ║ 1.7B      ║ 2048       ║ 24        ║ 32      ║ 32       ║ 2048     ║
║ SPARSA-Small     ║ 125M      ║ 768        ║ 12        ║ 12      ║ 4        ║ 2048     ║
║ SPARSA-Base      ║ 360M      ║ 1024       ║ 28        ║ 16      ║ 8        ║ 2048     ║
║ SPARSA-Large     ║ 760M      ║ 1536       ║ 32        ║ 24      ║ 8        ║ 2048     ║
║ SPARSA-XL        ║ 1.5B      ║ 2048       ║ 36        ║ 32      ║ 8        ║ 4096     ║
║ LLaMA-7B         ║ 7B        ║ 4096       ║ 32        ║ 32      ║ 32       ║ 2048     ║
║ LLaMA-2-7B       ║ 7B        ║ 4096       ║ 32        ║ 32      ║ 32       ║ 4096     ║
║ LLaMA-3-8B       ║ 8B        ║ 4096       ║ 32        ║ 32      ║ 8        ║ 8192     ║
║ Mistral-7B       ║ 7B        ║ 4096       ║ 32        ║ 32      ║ 8        ║ 32768    ║
╚══════════════════╩═══════════╩════════════╩═══════════╩═════════╩══════════╩══════════╝
"""


if __name__ == "__main__":
    # Print architecture comparison
    print(MODEL_COMPARISON)

    # Print detailed architecture for base model
    config = ModelConfig.base()
    config.print_architecture()
