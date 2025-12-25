"""
SPARSA-LM Experiment Tracking
Weights & Biases integration and logging utilities
"""

from .wandb_config import (
    WandbConfig,
    WandbTracker,
    init_wandb,
    log_metrics,
    log_model,
    finish_run,
)

from .metrics import (
    MetricLogger,
    TrainingMetrics,
    EvaluationMetrics,
    compute_perplexity,
    compute_accuracy,
)

__all__ = [
    # W&B
    "WandbConfig",
    "WandbTracker",
    "init_wandb",
    "log_metrics",
    "log_model",
    "finish_run",
    # Metrics
    "MetricLogger",
    "TrainingMetrics",
    "EvaluationMetrics",
    "compute_perplexity",
    "compute_accuracy",
]
