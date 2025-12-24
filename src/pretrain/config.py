"""
SPARSA-LM Pretraining Configuration
Configuration for distributed pretraining
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict
import json


@dataclass
class PretrainConfig:
    """
    Configuration for pretraining the AutoRegressive model.

    Supports distributed training with DeepSpeed ZeRO optimization.
    """

    # Output paths
    output_dir: str = "outputs/pretrain"
    checkpoint_dir: str = "checkpoints"
    logging_dir: str = "logs"

    # Training hyperparameters
    num_epochs: int = 3
    max_steps: int = -1  # -1 means use epochs
    per_device_batch_size: int = 4
    gradient_accumulation_steps: int = 8
    effective_batch_size: int = 256  # Will be computed if not set

    # Optimizer
    optimizer: str = "adamw"
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1.0

    # Learning rate schedule
    lr_scheduler: str = "cosine"
    warmup_steps: int = 2000
    warmup_ratio: float = 0.0  # Alternative to warmup_steps
    min_lr_ratio: float = 0.1  # Minimum LR as ratio of max

    # Mixed precision
    mixed_precision: str = "bf16"  # "fp16", "bf16", or "fp32"
    bf16: bool = True
    fp16: bool = False

    # DeepSpeed
    use_deepspeed: bool = True
    deepspeed_stage: int = 2  # ZeRO stage (0, 1, 2, or 3)
    offload_optimizer: bool = False
    offload_param: bool = False

    # Gradient checkpointing
    gradient_checkpointing: bool = True

    # Checkpointing
    save_steps: int = 1000
    save_total_limit: int = 5
    resume_from_checkpoint: Optional[str] = None

    # Logging
    logging_steps: int = 10
    log_level: str = "info"
    report_to: List[str] = field(default_factory=lambda: ["wandb"])

    # Weights & Biases
    wandb_project: str = "sparsa-lm-pretrain"
    wandb_run_name: Optional[str] = None
    wandb_entity: Optional[str] = None

    # Data
    dataloader_num_workers: int = 4
    dataloader_pin_memory: bool = True
    dataloader_prefetch_factor: int = 2

    # Distributed training
    local_rank: int = -1
    world_size: int = 1

    # Reproducibility
    seed: int = 42

    # Evaluation
    eval_steps: int = 500
    eval_accumulation_steps: int = 1

    def __post_init__(self):
        """Validate and compute derived values."""
        if self.bf16 and self.fp16:
            raise ValueError("Cannot use both bf16 and fp16")

        if self.bf16:
            self.mixed_precision = "bf16"
        elif self.fp16:
            self.mixed_precision = "fp16"

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}

    def save(self, path: str) -> None:
        """Save configuration to JSON."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_dict(cls, config_dict: dict) -> "PretrainConfig":
        """Create from dictionary."""
        return cls(**{k: v for k, v in config_dict.items() if k in cls.__dataclass_fields__})

    @classmethod
    def load(cls, path: str) -> "PretrainConfig":
        """Load from JSON file."""
        with open(path, 'r') as f:
            return cls.from_dict(json.load(f))

    def get_deepspeed_config(self) -> dict:
        """Generate DeepSpeed configuration."""
        config = {
            "train_batch_size": self.effective_batch_size,
            "train_micro_batch_size_per_gpu": self.per_device_batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "gradient_clipping": self.max_grad_norm,
            "steps_per_print": self.logging_steps,
            "optimizer": {
                "type": "AdamW",
                "params": {
                    "lr": self.learning_rate,
                    "betas": [self.adam_beta1, self.adam_beta2],
                    "eps": self.adam_epsilon,
                    "weight_decay": self.weight_decay,
                }
            },
            "scheduler": {
                "type": "WarmupDecayLR",
                "params": {
                    "warmup_min_lr": 0,
                    "warmup_max_lr": self.learning_rate,
                    "warmup_num_steps": self.warmup_steps,
                    "total_num_steps": self.max_steps if self.max_steps > 0 else 100000,
                }
            },
            "fp16": {
                "enabled": self.fp16,
                "loss_scale": 0,
                "loss_scale_window": 1000,
                "initial_scale_power": 16,
                "hysteresis": 2,
                "min_loss_scale": 1,
            },
            "bf16": {
                "enabled": self.bf16,
            },
            "zero_optimization": self._get_zero_config(),
            "activation_checkpointing": {
                "partition_activations": self.gradient_checkpointing,
                "cpu_checkpointing": False,
                "contiguous_memory_optimization": True,
                "number_checkpoints": None,
                "synchronize_checkpoint_boundary": False,
                "profile": False,
            },
            "wall_clock_breakdown": False,
        }

        return config

    def _get_zero_config(self) -> dict:
        """Get ZeRO optimization configuration."""
        if self.deepspeed_stage == 0:
            return {"stage": 0}

        config = {
            "stage": self.deepspeed_stage,
            "allgather_partitions": True,
            "allgather_bucket_size": 2e8,
            "overlap_comm": True,
            "reduce_scatter": True,
            "reduce_bucket_size": 2e8,
            "contiguous_gradients": True,
        }

        if self.deepspeed_stage >= 2:
            config["round_robin_gradients"] = True

        if self.deepspeed_stage == 3:
            config["stage3_max_live_parameters"] = 1e9
            config["stage3_max_reuse_distance"] = 1e9
            config["stage3_prefetch_bucket_size"] = 5e7
            config["stage3_param_persistence_threshold"] = 1e6
            config["sub_group_size"] = 1e12
            config["elastic_checkpoint"] = True

        if self.offload_optimizer:
            config["offload_optimizer"] = {
                "device": "cpu",
                "pin_memory": True,
            }

        if self.offload_param and self.deepspeed_stage == 3:
            config["offload_param"] = {
                "device": "cpu",
                "pin_memory": True,
            }

        return config
