"""
SPARSA-LM Supervised Fine-Tuning Trainer
Instruction tuning with evaluation
"""

import os
import time
import math
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Callable
from pathlib import Path

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler

logger = logging.getLogger(__name__)


@dataclass
class SFTConfig:
    """
    Configuration for Supervised Fine-Tuning.

    Training Schedule:
    - Epochs: 2-3 for instruction tuning
    - Learning rate: 1e-5 to 5e-5 (lower than pretraining)
    - Batch size: 4-16 per device
    - Warmup: 3-10% of total steps
    """

    # Output paths
    output_dir: str = "outputs/sft"
    checkpoint_dir: str = "checkpoints"
    logging_dir: str = "logs"

    # Training hyperparameters
    num_epochs: int = 3
    max_steps: int = -1
    per_device_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    eval_batch_size: int = 8

    # Optimizer
    optimizer: str = "adamw"
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1.0

    # Learning rate schedule
    lr_scheduler: str = "cosine"
    warmup_ratio: float = 0.03
    warmup_steps: int = 0  # If 0, use warmup_ratio
    min_lr_ratio: float = 0.0

    # Mixed precision
    bf16: bool = True
    fp16: bool = False

    # DeepSpeed
    use_deepspeed: bool = True
    deepspeed_stage: int = 2

    # Gradient checkpointing
    gradient_checkpointing: bool = True

    # Data
    max_seq_length: int = 2048
    mask_prompt: bool = True
    packing: bool = False  # Sequence packing for efficiency

    # Checkpointing
    save_steps: int = 500
    save_total_limit: int = 3
    eval_steps: int = 500
    resume_from_checkpoint: Optional[str] = None

    # Logging
    logging_steps: int = 10
    report_to: List[str] = field(default_factory=lambda: ["wandb"])
    wandb_project: str = "sparsa-lm-sft"
    wandb_run_name: Optional[str] = None

    # Evaluation
    eval_strategy: str = "steps"  # "steps", "epoch", or "no"
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False

    # Reproducibility
    seed: int = 42

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}

    def save(self, path: str):
        import json
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "SFTConfig":
        import json
        with open(path, 'r') as f:
            return cls(**json.load(f))


class SFTTrainer:
    """
    Trainer for Supervised Fine-Tuning.

    Features:
    - Instruction tuning with prompt masking
    - Multi-dataset training
    - Evaluation on benchmarks
    - LoRA/PEFT support (optional)
    """

    def __init__(
        self,
        model: nn.Module,
        config: SFTConfig,
        train_dataset: torch.utils.data.Dataset,
        eval_dataset: Optional[torch.utils.data.Dataset] = None,
        tokenizer: Any = None,
        compute_metrics: Optional[Callable] = None,
        data_collator: Optional[Callable] = None,
    ):
        self.model = model
        self.config = config
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.tokenizer = tokenizer
        self.compute_metrics = compute_metrics
        self.data_collator = data_collator

        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_metric = float('inf') if not config.greater_is_better else float('-inf')

        # Setup
        self._setup_distributed()
        self._setup_logging()
        self._setup_directories()

    def _setup_distributed(self):
        """Initialize distributed training."""
        self.is_distributed = dist.is_initialized()
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.world_size = dist.get_world_size() if self.is_distributed else 1
        self.is_main_process = self.local_rank == 0

        if torch.cuda.is_available():
            self.device = torch.device(f"cuda:{self.local_rank}")
            torch.cuda.set_device(self.device)
        else:
            self.device = torch.device("cpu")

        self.model = self.model.to(self.device)

    def _setup_logging(self):
        """Setup logging and experiment tracking."""
        if self.is_main_process:
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            )

            if "wandb" in self.config.report_to:
                try:
                    import wandb
                    wandb.init(
                        project=self.config.wandb_project,
                        name=self.config.wandb_run_name,
                        config=self.config.to_dict(),
                    )
                    self.wandb = wandb
                except ImportError:
                    logger.warning("wandb not installed")
                    self.wandb = None
            else:
                self.wandb = None

    def _setup_directories(self):
        """Create output directories."""
        if self.is_main_process:
            Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)
            Path(self.config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
            Path(self.config.logging_dir).mkdir(parents=True, exist_ok=True)

    def train(self) -> Dict[str, float]:
        """Run supervised fine-tuning."""
        # Create optimizer and scheduler
        optimizer = self._create_optimizer()
        num_training_steps = self._get_num_training_steps()
        scheduler = self._create_scheduler(optimizer, num_training_steps)

        # Setup gradient checkpointing
        if self.config.gradient_checkpointing:
            self.model.config.use_gradient_checkpointing = True

        # Setup DeepSpeed or DDP
        if self.config.use_deepspeed:
            self.model, optimizer, _, scheduler = self._setup_deepspeed(
                optimizer, scheduler, num_training_steps
            )
        elif self.is_distributed:
            self.model = nn.parallel.DistributedDataParallel(
                self.model,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
            )

        # Create dataloaders
        train_loader = self._create_dataloader(self.train_dataset, shuffle=True)
        eval_loader = None
        if self.eval_dataset:
            eval_loader = self._create_dataloader(self.eval_dataset, shuffle=False, eval=True)

        # Resume from checkpoint
        if self.config.resume_from_checkpoint:
            self._load_checkpoint(self.config.resume_from_checkpoint, optimizer, scheduler)

        # Training loop
        logger.info("=" * 60)
        logger.info("SUPERVISED FINE-TUNING")
        logger.info("=" * 60)
        logger.info(f"  Num examples = {len(self.train_dataset):,}")
        logger.info(f"  Num epochs = {self.config.num_epochs}")
        logger.info(f"  Per-device batch size = {self.config.per_device_batch_size}")
        logger.info(f"  Gradient accumulation = {self.config.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {num_training_steps:,}")
        logger.info(f"  Learning rate = {self.config.learning_rate}")
        logger.info("=" * 60)

        total_loss = 0.0
        start_time = time.time()

        for epoch in range(self.epoch, self.config.num_epochs):
            self.epoch = epoch
            self.model.train()

            if hasattr(train_loader.sampler, 'set_epoch'):
                train_loader.sampler.set_epoch(epoch)

            epoch_loss = 0.0
            num_batches = 0

            for step, batch in enumerate(train_loader):
                # Move batch to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                         for k, v in batch.items()}

                # Forward pass
                outputs = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch.get("attention_mask"),
                    labels=batch["labels"],
                )

                loss = outputs["loss"] if isinstance(outputs, dict) else outputs.loss

                # Scale loss for gradient accumulation
                if self.config.gradient_accumulation_steps > 1:
                    loss = loss / self.config.gradient_accumulation_steps

                # Backward pass
                if self.config.use_deepspeed:
                    self.model.backward(loss)
                else:
                    loss.backward()

                epoch_loss += loss.item() * self.config.gradient_accumulation_steps
                num_batches += 1

                # Optimizer step
                if (step + 1) % self.config.gradient_accumulation_steps == 0:
                    if self.config.use_deepspeed:
                        self.model.step()
                    else:
                        if self.config.max_grad_norm > 0:
                            torch.nn.utils.clip_grad_norm_(
                                self.model.parameters(),
                                self.config.max_grad_norm
                            )
                        optimizer.step()
                        scheduler.step()
                        optimizer.zero_grad()

                    self.global_step += 1
                    total_loss += loss.item() * self.config.gradient_accumulation_steps

                    # Logging
                    if self.global_step % self.config.logging_steps == 0:
                        avg_loss = epoch_loss / num_batches
                        lr = scheduler.get_last_lr()[0] if scheduler else self.config.learning_rate
                        self._log_metrics({
                            "train/loss": avg_loss,
                            "train/learning_rate": lr,
                            "train/epoch": epoch + step / len(train_loader),
                            "train/global_step": self.global_step,
                        })

                    # Evaluation
                    if (self.config.eval_strategy == "steps" and
                        self.global_step % self.config.eval_steps == 0 and
                        eval_loader is not None):
                        eval_metrics = self.evaluate(eval_loader)
                        self._log_metrics(eval_metrics)
                        self._maybe_save_best(eval_metrics, optimizer, scheduler)

                    # Checkpointing
                    if self.global_step % self.config.save_steps == 0:
                        self._save_checkpoint(optimizer, scheduler)

                    # Max steps check
                    if self.config.max_steps > 0 and self.global_step >= self.config.max_steps:
                        break

            # End of epoch evaluation
            if self.config.eval_strategy == "epoch" and eval_loader is not None:
                eval_metrics = self.evaluate(eval_loader)
                self._log_metrics(eval_metrics)
                self._maybe_save_best(eval_metrics, optimizer, scheduler)

            if self.config.max_steps > 0 and self.global_step >= self.config.max_steps:
                break

        # Final save
        self._save_checkpoint(optimizer, scheduler, final=True)

        training_time = time.time() - start_time
        metrics = {
            "train/final_loss": total_loss / max(self.global_step, 1),
            "train/total_steps": self.global_step,
            "train/training_time_hours": training_time / 3600,
        }

        logger.info(f"Training completed in {training_time / 3600:.2f} hours")

        return metrics

    @torch.no_grad()
    def evaluate(self, eval_loader: DataLoader) -> Dict[str, float]:
        """Evaluate model on validation set."""
        self.model.eval()

        total_loss = 0.0
        total_tokens = 0

        for batch in eval_loader:
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}

            outputs = self.model(
                input_ids=batch["input_ids"],
                attention_mask=batch.get("attention_mask"),
                labels=batch["labels"],
            )

            loss = outputs["loss"] if isinstance(outputs, dict) else outputs.loss
            num_tokens = (batch["labels"] != -100).sum().item()

            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens

        self.model.train()

        avg_loss = total_loss / max(total_tokens, 1)
        perplexity = math.exp(min(avg_loss, 20))

        return {
            "eval/loss": avg_loss,
            "eval/perplexity": perplexity,
        }

    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer with proper weight decay."""
        decay_params = []
        no_decay_params = []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if any(nd in name.lower() for nd in ["bias", "layernorm", "rmsnorm", "norm"]):
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        optimizer_groups = [
            {"params": decay_params, "weight_decay": self.config.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]

        return torch.optim.AdamW(
            optimizer_groups,
            lr=self.config.learning_rate,
            betas=(self.config.adam_beta1, self.config.adam_beta2),
            eps=self.config.adam_epsilon,
        )

    def _create_scheduler(self, optimizer, num_training_steps):
        """Create learning rate scheduler."""
        from torch.optim.lr_scheduler import LambdaLR

        warmup_steps = self.config.warmup_steps
        if warmup_steps == 0:
            warmup_steps = int(num_training_steps * self.config.warmup_ratio)

        def lr_lambda(current_step: int) -> float:
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))

            progress = float(current_step - warmup_steps) / float(
                max(1, num_training_steps - warmup_steps)
            )
            return max(
                self.config.min_lr_ratio,
                0.5 * (1.0 + math.cos(math.pi * progress))
            )

        return LambdaLR(optimizer, lr_lambda)

    def _get_num_training_steps(self) -> int:
        """Calculate total training steps."""
        if self.config.max_steps > 0:
            return self.config.max_steps

        num_batches = len(self.train_dataset) // self.config.per_device_batch_size
        num_batches = num_batches // self.world_size
        steps_per_epoch = num_batches // self.config.gradient_accumulation_steps
        return steps_per_epoch * self.config.num_epochs

    def _setup_deepspeed(self, optimizer, scheduler, num_training_steps):
        """Setup DeepSpeed."""
        try:
            import deepspeed
        except ImportError:
            raise ImportError("Please install deepspeed")

        ds_config = {
            "train_batch_size": self.config.per_device_batch_size * self.config.gradient_accumulation_steps * self.world_size,
            "train_micro_batch_size_per_gpu": self.config.per_device_batch_size,
            "gradient_accumulation_steps": self.config.gradient_accumulation_steps,
            "gradient_clipping": self.config.max_grad_norm,
            "bf16": {"enabled": self.config.bf16},
            "fp16": {"enabled": self.config.fp16},
            "zero_optimization": {
                "stage": self.config.deepspeed_stage,
                "overlap_comm": True,
                "contiguous_gradients": True,
            },
        }

        model_engine, optimizer, _, scheduler = deepspeed.initialize(
            model=self.model,
            optimizer=optimizer,
            lr_scheduler=scheduler,
            config=ds_config,
        )

        return model_engine, optimizer, None, scheduler

    def _create_dataloader(
        self,
        dataset,
        shuffle: bool = True,
        eval: bool = False,
    ) -> DataLoader:
        """Create data loader."""
        batch_size = self.config.eval_batch_size if eval else self.config.per_device_batch_size

        sampler = None
        if self.is_distributed and not eval:
            sampler = DistributedSampler(dataset, shuffle=shuffle)
            shuffle = False

        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle and sampler is None,
            sampler=sampler,
            num_workers=4,
            pin_memory=True,
            collate_fn=self.data_collator,
        )

    def _maybe_save_best(self, eval_metrics: Dict, optimizer, scheduler):
        """Save checkpoint if this is the best model."""
        metric_value = eval_metrics.get(f"eval/{self.config.metric_for_best_model.replace('eval_', '')}")
        if metric_value is None:
            return

        is_better = (
            (self.config.greater_is_better and metric_value > self.best_metric) or
            (not self.config.greater_is_better and metric_value < self.best_metric)
        )

        if is_better:
            self.best_metric = metric_value
            self._save_checkpoint(optimizer, scheduler, is_best=True)
            logger.info(f"New best model! {self.config.metric_for_best_model}: {metric_value:.4f}")

    def _save_checkpoint(self, optimizer, scheduler, final: bool = False, is_best: bool = False):
        """Save training checkpoint."""
        if not self.is_main_process:
            return

        if is_best:
            checkpoint_name = "best"
        elif final:
            checkpoint_name = "final"
        else:
            checkpoint_name = f"step_{self.global_step}"

        checkpoint_path = Path(self.config.checkpoint_dir) / checkpoint_name
        checkpoint_path.mkdir(parents=True, exist_ok=True)

        # Save model
        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
        if hasattr(model_to_save, 'save_pretrained'):
            model_to_save.save_pretrained(str(checkpoint_path))
        else:
            torch.save(model_to_save.state_dict(), checkpoint_path / "model.pt")

        # Save tokenizer
        if self.tokenizer:
            self.tokenizer.save_pretrained(str(checkpoint_path))

        # Save training state
        state = {
            "global_step": self.global_step,
            "epoch": self.epoch,
            "best_metric": self.best_metric,
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
            "config": self.config.to_dict(),
        }
        torch.save(state, checkpoint_path / "training_state.pt")

        logger.info(f"Saved checkpoint to {checkpoint_path}")

    def _load_checkpoint(self, checkpoint_path: str, optimizer, scheduler):
        """Load training checkpoint."""
        checkpoint_path = Path(checkpoint_path)

        state_path = checkpoint_path / "training_state.pt"
        if state_path.exists():
            state = torch.load(state_path, map_location=self.device)
            self.global_step = state["global_step"]
            self.epoch = state["epoch"]
            self.best_metric = state.get("best_metric", float('inf'))
            optimizer.load_state_dict(state["optimizer_state_dict"])
            if scheduler and state.get("scheduler_state_dict"):
                scheduler.load_state_dict(state["scheduler_state_dict"])

        logger.info(f"Resumed from {checkpoint_path} at step {self.global_step}")

    def _log_metrics(self, metrics: Dict[str, float]):
        """Log metrics."""
        if not self.is_main_process:
            return

        metrics_str = " | ".join([
            f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}"
            for k, v in metrics.items()
        ])
        logger.info(f"Step {self.global_step} | {metrics_str}")

        if self.wandb:
            self.wandb.log(metrics, step=self.global_step)
