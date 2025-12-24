"""
SPARSA-LM Model Configuration
Defines the configuration for the AutoRegressive Language Model
"""

from dataclasses import dataclass, field
from typing import Optional, List
import json


@dataclass
class ModelConfig:
    """
    Configuration for the AutoRegressive Language Model.

    This configuration defines all hyperparameters for the model architecture
    including attention mechanisms, normalization, and positional embeddings.
    """

    # Model dimensions
    vocab_size: int = 32000
    hidden_size: int = 1024
    intermediate_size: int = 4096
    num_hidden_layers: int = 28
    num_attention_heads: int = 16
    num_key_value_heads: int = 8  # For Grouped Query Attention

    # Sequence configuration
    max_position_embeddings: int = 2048
    sliding_window: Optional[int] = 512

    # Normalization
    rms_norm_eps: float = 1e-6

    # Activation
    hidden_act: str = "silu"

    # Attention configuration
    attention_dropout: float = 0.0
    attention_bias: bool = False

    # RoPE configuration
    rope_theta: float = 10000.0
    rope_scaling: Optional[dict] = None

    # Initialization
    initializer_range: float = 0.02

    # Dropout
    hidden_dropout: float = 0.0

    # Flash Attention
    use_flash_attention: bool = True

    # Gradient checkpointing
    use_gradient_checkpointing: bool = False

    # Tie word embeddings
    tie_word_embeddings: bool = True

    # Special tokens
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2

    # Model type identifier
    model_type: str = "sparsa_lm"

    def __post_init__(self):
        """Validate configuration after initialization."""
        assert self.hidden_size % self.num_attention_heads == 0, \
            f"hidden_size ({self.hidden_size}) must be divisible by num_attention_heads ({self.num_attention_heads})"
        assert self.num_attention_heads % self.num_key_value_heads == 0, \
            f"num_attention_heads ({self.num_attention_heads}) must be divisible by num_key_value_heads ({self.num_key_value_heads})"

    @property
    def head_dim(self) -> int:
        """Dimension of each attention head."""
        return self.hidden_size // self.num_attention_heads

    @property
    def num_kv_groups(self) -> int:
        """Number of key-value groups for GQA."""
        return self.num_attention_heads // self.num_key_value_heads

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

    @classmethod
    def small(cls) -> "ModelConfig":
        """Small model configuration (~125M parameters)."""
        return cls(
            hidden_size=768,
            intermediate_size=3072,
            num_hidden_layers=12,
            num_attention_heads=12,
            num_key_value_heads=4,
        )

    @classmethod
    def base(cls) -> "ModelConfig":
        """Base model configuration (~360M parameters)."""
        return cls()  # Default configuration

    @classmethod
    def large(cls) -> "ModelConfig":
        """Large model configuration (~760M parameters)."""
        return cls(
            hidden_size=1536,
            intermediate_size=6144,
            num_hidden_layers=32,
            num_attention_heads=24,
            num_key_value_heads=8,
        )

    @classmethod
    def xl(cls) -> "ModelConfig":
        """XL model configuration (~1.5B parameters)."""
        return cls(
            hidden_size=2048,
            intermediate_size=8192,
            num_hidden_layers=36,
            num_attention_heads=32,
            num_key_value_heads=8,
            max_position_embeddings=4096,
        )
