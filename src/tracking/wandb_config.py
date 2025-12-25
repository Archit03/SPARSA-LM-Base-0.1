"""
SPARSA-LM Weights & Biases Configuration and Tracking
Comprehensive experiment tracking for training pipelines
"""

import os
import json
import time
import logging
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any, List, Union
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# Check if wandb is available
try:
    import wandb
    from wandb import AlertLevel
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None
    AlertLevel = None


@dataclass
class WandbConfig:
    """
    Configuration for Weights & Biases experiment tracking.

    This configures all aspects of W&B integration including:
    - Project and run settings
    - Logging configuration
    - Artifact tracking
    - Alert settings
    """

    # ==========================================================================
    # Project Configuration
    # ==========================================================================

    project: str = "sparsa-lm"
    entity: Optional[str] = None  # W&B team/username
    name: Optional[str] = None  # Run name (auto-generated if None)
    group: Optional[str] = None  # Group related runs (e.g., "hyperparameter-sweep")
    job_type: Optional[str] = None  # Type of run: "train", "eval", "preprocess"
    tags: List[str] = field(default_factory=list)  # Run tags for filtering
    notes: Optional[str] = None  # Run notes/description

    # ==========================================================================
    # Logging Configuration
    # ==========================================================================

    log_freq: int = 10  # Log metrics every N steps
    log_code: bool = True  # Log code to W&B
    log_model: bool = True  # Log model checkpoints as artifacts
    log_gradients: bool = False  # Log gradient histograms (expensive)
    log_parameters: bool = False  # Log parameter histograms
    gradient_log_freq: int = 100  # How often to log gradients

    # ==========================================================================
    # Model Watching
    # ==========================================================================

    watch_model: bool = False  # Enable model watching
    watch_log: str = "gradients"  # "gradients", "parameters", "all", or None
    watch_log_freq: int = 100  # How often to log when watching

    # ==========================================================================
    # Checkpoint and Artifact Settings
    # ==========================================================================

    save_checkpoints: bool = True
    checkpoint_freq: int = 1000  # Save checkpoint every N steps
    checkpoint_artifact_type: str = "model"
    keep_checkpoints: int = 3  # Number of checkpoints to keep

    # ==========================================================================
    # Run Settings
    # ==========================================================================

    mode: str = "online"  # "online", "offline", or "disabled"
    resume: Optional[str] = None  # Resume run: "allow", "must", "never", or run_id
    reinit: bool = False  # Allow multiple wandb.init() calls
    anonymous: Optional[str] = None  # "allow", "must", or "never"

    # ==========================================================================
    # Alerts
    # ==========================================================================

    enable_alerts: bool = True
    alert_on_nan: bool = True
    alert_on_training_complete: bool = True
    alert_on_eval_improvement: bool = True

    # ==========================================================================
    # Sweep Configuration (Hyperparameter Optimization)
    # ==========================================================================

    sweep_id: Optional[str] = None
    sweep_count: Optional[int] = None

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return asdict(self)

    def get_init_kwargs(self) -> dict:
        """Get kwargs for wandb.init()."""
        return {
            "project": self.project,
            "entity": self.entity,
            "name": self.name,
            "group": self.group,
            "job_type": self.job_type,
            "tags": self.tags,
            "notes": self.notes,
            "mode": self.mode,
            "resume": self.resume,
            "reinit": self.reinit,
            "anonymous": self.anonymous,
        }

    @classmethod
    def for_pretraining(cls, name: Optional[str] = None) -> "WandbConfig":
        """Configuration for pretraining runs."""
        return cls(
            project="sparsa-lm-pretrain",
            name=name or f"pretrain-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
            job_type="pretrain",
            tags=["pretrain", "foundation"],
            log_freq=10,
            log_model=True,
            checkpoint_freq=5000,
            watch_model=False,  # Too expensive for pretraining
        )

    @classmethod
    def for_sft(cls, name: Optional[str] = None) -> "WandbConfig":
        """Configuration for SFT runs."""
        return cls(
            project="sparsa-lm-sft",
            name=name or f"sft-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
            job_type="sft",
            tags=["sft", "instruction-tuning"],
            log_freq=10,
            log_model=True,
            checkpoint_freq=500,
            alert_on_eval_improvement=True,
        )

    @classmethod
    def for_rlhf(cls, name: Optional[str] = None, method: str = "dapo") -> "WandbConfig":
        """Configuration for RLHF runs."""
        return cls(
            project="sparsa-lm-rlhf",
            name=name or f"{method}-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
            job_type="rlhf",
            tags=["rlhf", method],
            log_freq=5,  # More frequent logging for RL
            log_model=True,
            checkpoint_freq=200,
            alert_on_eval_improvement=True,
            log_gradients=True,  # Important for RL debugging
            gradient_log_freq=50,
        )

    @classmethod
    def for_evaluation(cls, name: Optional[str] = None) -> "WandbConfig":
        """Configuration for evaluation runs."""
        return cls(
            project="sparsa-lm-eval",
            name=name or f"eval-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
            job_type="eval",
            tags=["evaluation", "benchmark"],
            log_freq=1,
            log_model=False,
            save_checkpoints=False,
        )


class WandbTracker:
    """
    Weights & Biases experiment tracker.

    Provides high-level interface for:
    - Run initialization and configuration
    - Metric logging with automatic batching
    - Model checkpoint artifacts
    - Training alerts
    - Hyperparameter sweeps
    """

    def __init__(
        self,
        config: WandbConfig,
        training_config: Optional[Dict[str, Any]] = None,
        model_config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize W&B tracker.

        Args:
            config: W&B configuration
            training_config: Training hyperparameters to log
            model_config: Model architecture config to log
        """
        self.config = config
        self.training_config = training_config or {}
        self.model_config = model_config or {}
        self.run = None
        self.step = 0
        self.best_metric = float('inf')
        self.start_time = None
        self._metric_buffer = {}
        self._initialized = False

    def init(self) -> Optional['wandb.sdk.wandb_run.Run']:
        """Initialize W&B run."""
        if not WANDB_AVAILABLE:
            logger.warning("W&B not available. Install with: pip install wandb")
            return None

        if self._initialized:
            return self.run

        try:
            # Combine configs
            combined_config = {
                **self.training_config,
                "model": self.model_config,
                "wandb": self.config.to_dict(),
            }

            # Initialize run
            self.run = wandb.init(
                config=combined_config,
                **self.config.get_init_kwargs(),
            )

            # Log code if enabled
            if self.config.log_code:
                self._log_code()

            self._initialized = True
            self.start_time = time.time()

            logger.info(f"W&B run initialized: {self.run.url}")
            return self.run

        except Exception as e:
            logger.error(f"Failed to initialize W&B: {e}")
            return None

    def _log_code(self):
        """Log code to W&B."""
        if not self.run:
            return

        try:
            # Log Python files in src directory
            self.run.log_code(
                root=".",
                include_fn=lambda path: (
                    path.endswith(".py") and
                    not any(x in path for x in ["__pycache__", ".git", "venv", "env"])
                ),
            )
        except Exception as e:
            logger.warning(f"Failed to log code: {e}")

    def watch_model(self, model: nn.Module):
        """
        Watch model for gradient/parameter logging.

        Args:
            model: PyTorch model to watch
        """
        if not self.run or not self.config.watch_model:
            return

        try:
            wandb.watch(
                model,
                log=self.config.watch_log,
                log_freq=self.config.watch_log_freq,
                log_graph=True,
            )
            logger.info("Model watching enabled")
        except Exception as e:
            logger.warning(f"Failed to watch model: {e}")

    def log(
        self,
        metrics: Dict[str, Any],
        step: Optional[int] = None,
        commit: bool = True,
    ):
        """
        Log metrics to W&B.

        Args:
            metrics: Dictionary of metrics to log
            step: Step number (uses internal counter if None)
            commit: Whether to commit the log
        """
        if not self.run:
            return

        if step is not None:
            self.step = step
        else:
            self.step += 1

        # Add metrics to buffer
        self._metric_buffer.update(metrics)

        # Log if it's time
        if commit and self.step % self.config.log_freq == 0:
            self._flush_metrics()

    def _flush_metrics(self):
        """Flush buffered metrics to W&B."""
        if not self.run or not self._metric_buffer:
            return

        try:
            # Add timing information
            if self.start_time:
                self._metric_buffer["time/elapsed_hours"] = (time.time() - self.start_time) / 3600
                self._metric_buffer["time/step"] = self.step

            wandb.log(self._metric_buffer, step=self.step)
            self._metric_buffer = {}

        except Exception as e:
            logger.warning(f"Failed to log metrics: {e}")

    def log_summary(self, metrics: Dict[str, Any]):
        """
        Log summary metrics (shown in run overview).

        Args:
            metrics: Summary metrics
        """
        if not self.run:
            return

        try:
            for key, value in metrics.items():
                wandb.run.summary[key] = value
        except Exception as e:
            logger.warning(f"Failed to log summary: {e}")

    def log_table(
        self,
        name: str,
        data: List[Dict[str, Any]],
        columns: Optional[List[str]] = None,
    ):
        """
        Log a table to W&B.

        Args:
            name: Table name
            data: List of row dictionaries
            columns: Column names (inferred from data if None)
        """
        if not self.run:
            return

        try:
            if columns is None and data:
                columns = list(data[0].keys())

            table = wandb.Table(columns=columns)
            for row in data:
                table.add_data(*[row.get(col) for col in columns])

            wandb.log({name: table}, step=self.step)
        except Exception as e:
            logger.warning(f"Failed to log table: {e}")

    def log_histogram(
        self,
        name: str,
        values: Union[torch.Tensor, List[float]],
        num_bins: int = 64,
    ):
        """
        Log a histogram to W&B.

        Args:
            name: Histogram name
            values: Values to plot
            num_bins: Number of bins
        """
        if not self.run:
            return

        try:
            if isinstance(values, torch.Tensor):
                values = values.detach().cpu().numpy().flatten()

            wandb.log({name: wandb.Histogram(values, num_bins=num_bins)}, step=self.step)
        except Exception as e:
            logger.warning(f"Failed to log histogram: {e}")

    def log_image(
        self,
        name: str,
        image: Any,
        caption: Optional[str] = None,
    ):
        """
        Log an image to W&B.

        Args:
            name: Image name
            image: Image data (numpy array, PIL, or path)
            caption: Image caption
        """
        if not self.run:
            return

        try:
            wandb.log({name: wandb.Image(image, caption=caption)}, step=self.step)
        except Exception as e:
            logger.warning(f"Failed to log image: {e}")

    def save_checkpoint(
        self,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        epoch: int = 0,
        metrics: Optional[Dict[str, float]] = None,
        name: Optional[str] = None,
    ):
        """
        Save model checkpoint as W&B artifact.

        Args:
            model: Model to save
            optimizer: Optimizer state
            scheduler: LR scheduler state
            epoch: Current epoch
            metrics: Metrics to include in artifact metadata
            name: Artifact name
        """
        if not self.run or not self.config.save_checkpoints:
            return

        try:
            # Create checkpoint directory
            checkpoint_dir = Path(f"checkpoints/step_{self.step}")
            checkpoint_dir.mkdir(parents=True, exist_ok=True)

            # Save model
            model_to_save = model.module if hasattr(model, 'module') else model
            torch.save(model_to_save.state_dict(), checkpoint_dir / "model.pt")

            # Save optimizer
            if optimizer:
                torch.save(optimizer.state_dict(), checkpoint_dir / "optimizer.pt")

            # Save scheduler
            if scheduler:
                torch.save(scheduler.state_dict(), checkpoint_dir / "scheduler.pt")

            # Save training state
            state = {
                "step": self.step,
                "epoch": epoch,
                "metrics": metrics or {},
                "config": self.training_config,
            }
            with open(checkpoint_dir / "training_state.json", 'w') as f:
                json.dump(state, f, indent=2)

            # Create artifact
            artifact_name = name or f"model-step-{self.step}"
            artifact = wandb.Artifact(
                name=artifact_name,
                type=self.config.checkpoint_artifact_type,
                metadata={
                    "step": self.step,
                    "epoch": epoch,
                    **(metrics or {}),
                },
            )
            artifact.add_dir(str(checkpoint_dir))

            # Log artifact
            self.run.log_artifact(artifact)
            logger.info(f"Saved checkpoint artifact: {artifact_name}")

        except Exception as e:
            logger.warning(f"Failed to save checkpoint: {e}")

    def alert(
        self,
        title: str,
        text: str,
        level: str = "INFO",
        wait_duration: int = 300,
    ):
        """
        Send an alert.

        Args:
            title: Alert title
            text: Alert body
            level: Alert level ("INFO", "WARN", "ERROR")
            wait_duration: Seconds to wait before sending another alert
        """
        if not self.run or not self.config.enable_alerts or not WANDB_AVAILABLE:
            return

        try:
            level_map = {
                "INFO": AlertLevel.INFO,
                "WARN": AlertLevel.WARN,
                "ERROR": AlertLevel.ERROR,
            }
            wandb.alert(
                title=title,
                text=text,
                level=level_map.get(level, AlertLevel.INFO),
                wait_duration=wait_duration,
            )
        except Exception as e:
            logger.warning(f"Failed to send alert: {e}")

    def check_for_nan(self, metrics: Dict[str, float]) -> bool:
        """
        Check for NaN values and alert if found.

        Args:
            metrics: Metrics to check

        Returns:
            True if NaN found
        """
        import math

        nan_found = False
        for key, value in metrics.items():
            if isinstance(value, float) and math.isnan(value):
                nan_found = True
                if self.config.alert_on_nan:
                    self.alert(
                        title="NaN Detected",
                        text=f"NaN value detected in metric: {key} at step {self.step}",
                        level="ERROR",
                    )
                break

        return nan_found

    def check_improvement(
        self,
        metric: float,
        lower_is_better: bool = True,
    ) -> bool:
        """
        Check if metric improved and alert if enabled.

        Args:
            metric: Current metric value
            lower_is_better: Whether lower values are better

        Returns:
            True if improved
        """
        improved = (lower_is_better and metric < self.best_metric) or \
                   (not lower_is_better and metric > self.best_metric)

        if improved:
            old_best = self.best_metric
            self.best_metric = metric

            if self.config.alert_on_eval_improvement:
                self.alert(
                    title="New Best Model",
                    text=f"Metric improved from {old_best:.4f} to {metric:.4f} at step {self.step}",
                    level="INFO",
                )

        return improved

    def finish(self, quiet: bool = False):
        """
        Finish the W&B run.

        Args:
            quiet: Whether to suppress output
        """
        if not self.run:
            return

        try:
            # Flush remaining metrics
            self._flush_metrics()

            # Log final summary
            if self.start_time:
                self.log_summary({
                    "total_steps": self.step,
                    "total_time_hours": (time.time() - self.start_time) / 3600,
                    "best_metric": self.best_metric,
                })

            # Send completion alert
            if self.config.alert_on_training_complete:
                self.alert(
                    title="Training Complete",
                    text=f"Run finished at step {self.step}. Best metric: {self.best_metric:.4f}",
                    level="INFO",
                )

            wandb.finish(quiet=quiet)
            logger.info("W&B run finished")

        except Exception as e:
            logger.warning(f"Error finishing W&B run: {e}")

        self._initialized = False
        self.run = None


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

_active_tracker: Optional[WandbTracker] = None


def init_wandb(
    config: Optional[WandbConfig] = None,
    training_config: Optional[Dict[str, Any]] = None,
    model_config: Optional[Dict[str, Any]] = None,
    **kwargs,
) -> Optional[WandbTracker]:
    """
    Initialize global W&B tracker.

    Args:
        config: W&B configuration
        training_config: Training hyperparameters
        model_config: Model configuration
        **kwargs: Override config values

    Returns:
        Initialized tracker or None if unavailable
    """
    global _active_tracker

    if config is None:
        config = WandbConfig(**kwargs)
    elif kwargs:
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)

    _active_tracker = WandbTracker(config, training_config, model_config)
    _active_tracker.init()
    return _active_tracker


def log_metrics(
    metrics: Dict[str, Any],
    step: Optional[int] = None,
    commit: bool = True,
):
    """
    Log metrics to global tracker.

    Args:
        metrics: Metrics to log
        step: Step number
        commit: Whether to commit
    """
    if _active_tracker:
        _active_tracker.log(metrics, step, commit)


def log_model(
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    epoch: int = 0,
    metrics: Optional[Dict[str, float]] = None,
    name: Optional[str] = None,
):
    """
    Save model checkpoint to global tracker.

    Args:
        model: Model to save
        optimizer: Optimizer state
        scheduler: Scheduler state
        epoch: Current epoch
        metrics: Metrics metadata
        name: Artifact name
    """
    if _active_tracker:
        _active_tracker.save_checkpoint(model, optimizer, scheduler, epoch, metrics, name)


def finish_run(quiet: bool = False):
    """
    Finish the global W&B run.

    Args:
        quiet: Suppress output
    """
    global _active_tracker
    if _active_tracker:
        _active_tracker.finish(quiet)
        _active_tracker = None


# =============================================================================
# SWEEP CONFIGURATION
# =============================================================================

def create_sweep_config(
    method: str = "bayes",
    metric_name: str = "eval/loss",
    metric_goal: str = "minimize",
    parameters: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Create a W&B sweep configuration.

    Args:
        method: Sweep method ("bayes", "grid", "random")
        metric_name: Metric to optimize
        metric_goal: "minimize" or "maximize"
        parameters: Parameter search space

    Returns:
        Sweep configuration dictionary
    """
    if parameters is None:
        # Default hyperparameter search space
        parameters = {
            "learning_rate": {
                "distribution": "log_uniform_values",
                "min": 1e-5,
                "max": 1e-3,
            },
            "weight_decay": {
                "distribution": "log_uniform_values",
                "min": 1e-4,
                "max": 1e-1,
            },
            "warmup_ratio": {
                "distribution": "uniform",
                "min": 0.01,
                "max": 0.1,
            },
            "batch_size": {
                "values": [8, 16, 32, 64],
            },
            "gradient_accumulation_steps": {
                "values": [1, 2, 4, 8],
            },
        }

    return {
        "method": method,
        "metric": {
            "name": metric_name,
            "goal": metric_goal,
        },
        "parameters": parameters,
        "early_terminate": {
            "type": "hyperband",
            "min_iter": 3,
            "eta": 3,
            "s": 2,
        },
    }


def init_sweep(
    sweep_config: Dict[str, Any],
    project: str = "sparsa-lm-sweeps",
    entity: Optional[str] = None,
) -> Optional[str]:
    """
    Initialize a W&B sweep.

    Args:
        sweep_config: Sweep configuration
        project: W&B project name
        entity: W&B entity

    Returns:
        Sweep ID or None if unavailable
    """
    if not WANDB_AVAILABLE:
        logger.warning("W&B not available for sweeps")
        return None

    try:
        sweep_id = wandb.sweep(sweep_config, project=project, entity=entity)
        logger.info(f"Created sweep: {sweep_id}")
        return sweep_id
    except Exception as e:
        logger.error(f"Failed to create sweep: {e}")
        return None


def run_sweep_agent(
    sweep_id: str,
    train_function: callable,
    count: Optional[int] = None,
    project: str = "sparsa-lm-sweeps",
    entity: Optional[str] = None,
):
    """
    Run a W&B sweep agent.

    Args:
        sweep_id: Sweep ID to join
        train_function: Training function to call
        count: Number of runs to execute
        project: W&B project name
        entity: W&B entity
    """
    if not WANDB_AVAILABLE:
        logger.warning("W&B not available for sweeps")
        return

    try:
        wandb.agent(
            sweep_id,
            function=train_function,
            count=count,
            project=project,
            entity=entity,
        )
    except Exception as e:
        logger.error(f"Sweep agent failed: {e}")
