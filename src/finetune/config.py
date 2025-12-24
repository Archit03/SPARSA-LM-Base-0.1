"""
SPARSA-LM Finetuning Configuration
Configuration for DAPO/VAPO RL-based finetuning
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict
import json


@dataclass
class FinetuneConfig:
    """
    Configuration for RL-based finetuning.

    Supports DAPO (Decoupled Clip and Dynamic Sampling Policy Optimization)
    and VAPO (Value-model Augmented Proximal Policy Optimization).
    """

    # Output paths
    output_dir: str = "outputs/finetune"
    checkpoint_dir: str = "checkpoints"
    logging_dir: str = "logs"

    # Training hyperparameters
    num_epochs: int = 3
    max_steps: int = -1
    per_device_batch_size: int = 4
    gradient_accumulation_steps: int = 4

    # Optimizer
    learning_rate: float = 1e-5
    weight_decay: float = 0.01
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1.0

    # Learning rate schedule
    warmup_steps: int = 100
    lr_scheduler: str = "cosine"

    # PPO hyperparameters (base for DAPO/VAPO)
    ppo_epochs: int = 4
    gamma: float = 1.0
    lam: float = 0.95
    kl_penalty: str = "kl"  # "kl" or "abs" or "mse"
    init_kl_coef: float = 0.1
    target_kl: float = 0.1
    adaptive_kl: bool = True

    # DAPO-specific hyperparameters
    dapo_enabled: bool = True
    clip_range_upper: float = 0.2  # Decoupled upper clip
    clip_range_lower: float = 0.1  # Decoupled lower clip
    dynamic_sampling: bool = True
    sampling_temperature_init: float = 1.0
    sampling_temperature_decay: float = 0.995
    entropy_coef: float = 0.01
    entropy_target: float = 0.1  # Prevents entropy collapse

    # VAPO-specific hyperparameters
    vapo_enabled: bool = False
    value_model_coef: float = 0.5
    reward_smoothing: float = 0.1
    dense_reward: bool = True
    value_clip_range: float = 0.2

    # Generation parameters
    max_new_tokens: int = 256
    generation_temperature: float = 0.7
    generation_top_p: float = 0.9
    generation_top_k: int = 50
    num_return_sequences: int = 1

    # Reward model
    reward_model_path: Optional[str] = None
    use_external_reward: bool = False
    reward_baseline: str = "moving_average"  # "none", "moving_average", "critic"

    # Reference model
    use_reference_model: bool = True
    reference_model_path: Optional[str] = None

    # Mixed precision
    bf16: bool = True
    fp16: bool = False

    # DeepSpeed
    use_deepspeed: bool = True
    deepspeed_stage: int = 2

    # Checkpointing
    save_steps: int = 500
    save_total_limit: int = 3

    # Logging
    logging_steps: int = 10
    report_to: List[str] = field(default_factory=lambda: ["wandb"])
    wandb_project: str = "sparsa-lm-finetune"

    # Reproducibility
    seed: int = 42

    def __post_init__(self):
        """Validate configuration."""
        if self.dapo_enabled and self.vapo_enabled:
            raise ValueError("Cannot enable both DAPO and VAPO simultaneously")

        if self.clip_range_lower > self.clip_range_upper:
            raise ValueError("clip_range_lower must be <= clip_range_upper")

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}

    def save(self, path: str) -> None:
        """Save configuration to JSON."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_dict(cls, config_dict: dict) -> "FinetuneConfig":
        """Create from dictionary."""
        return cls(**{k: v for k, v in config_dict.items() if k in cls.__dataclass_fields__})

    @classmethod
    def load(cls, path: str) -> "FinetuneConfig":
        """Load from JSON file."""
        with open(path, 'r') as f:
            return cls.from_dict(json.load(f))

    @classmethod
    def dapo_default(cls) -> "FinetuneConfig":
        """Default DAPO configuration."""
        return cls(
            dapo_enabled=True,
            vapo_enabled=False,
            clip_range_upper=0.2,
            clip_range_lower=0.1,
            dynamic_sampling=True,
            entropy_coef=0.01,
        )

    @classmethod
    def vapo_default(cls) -> "FinetuneConfig":
        """Default VAPO configuration."""
        return cls(
            dapo_enabled=False,
            vapo_enabled=True,
            value_model_coef=0.5,
            reward_smoothing=0.1,
            dense_reward=True,
        )
