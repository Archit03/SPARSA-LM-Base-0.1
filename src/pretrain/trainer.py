"""
SPARSA-LM Pretraining Trainer
Distributed training with DeepSpeed
"""

import os
import time
import math
import logging
from typing import Optional, Dict, Any, Callable
from pathlib import Path

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler

from .config import PretrainConfig
from ..model import AutoRegressiveLM, ModelConfig
from ..data import PretrainDataset, PretrainCollator, DatasetConfig

logger = logging.getLogger(__name__)


class PretrainTrainer:
    """
    Trainer for pretraining the AutoRegressive model.

    Features:
    - DeepSpeed ZeRO optimization (stages 1, 2, 3)
    - Mixed precision training (BF16/FP16)
    - Gradient checkpointing
    - Distributed data parallel training
    - Automatic checkpointing and resumption
    - Weights & Biases logging
    """

    def __init__(
        self,
        model: AutoRegressiveLM,
        config: PretrainConfig,
        train_dataset: PretrainDataset,
        eval_dataset: Optional[PretrainDataset] = None,
        tokenizer: Any = None,
        compute_metrics: Optional[Callable] = None,
    ):
        self.model = model
        self.config = config
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.tokenizer = tokenizer
        self.compute_metrics = compute_metrics

        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_loss = float('inf')

        # Setup distributed training
        self._setup_distributed()

        # Setup logging
        self._setup_logging()

        # Setup output directories
        self._setup_directories()

    def _setup_distributed(self):
        """Initialize distributed training."""
        self.is_distributed = dist.is_initialized()
        self.local_rank = self.config.local_rank if self.is_distributed else 0
        self.world_size = dist.get_world_size() if self.is_distributed else 1
        self.is_main_process = self.local_rank == 0

        if self.is_distributed:
            self.device = torch.device(f"cuda:{self.local_rank}")
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Move model to device
        self.model = self.model.to(self.device)

    def _setup_logging(self):
        """Setup logging and experiment tracking."""
        if self.is_main_process:
            logging.basicConfig(
                level=getattr(logging, self.config.log_level.upper()),
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            )

            # Setup W&B
            if "wandb" in self.config.report_to:
                try:
                    import wandb
                    wandb.init(
                        project=self.config.wandb_project,
                        name=self.config.wandb_run_name,
                        entity=self.config.wandb_entity,
                        config=self.config.to_dict(),
                    )
                    self.wandb = wandb
                except ImportError:
                    logger.warning("wandb not installed, skipping logging")
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
        """
        Run pretraining.

        Returns:
            Dictionary with training metrics
        """
        # Setup optimizer and scheduler
        optimizer = self._create_optimizer()
        scheduler = self._create_scheduler(optimizer)

        # Setup DeepSpeed if enabled
        if self.config.use_deepspeed:
            self.model, optimizer, _, scheduler = self._setup_deepspeed(optimizer, scheduler)
        else:
            # Standard DDP
            if self.is_distributed:
                self.model = nn.parallel.DistributedDataParallel(
                    self.model,
                    device_ids=[self.local_rank],
                    output_device=self.local_rank,
                )

        # Create data loader
        train_loader = self._create_dataloader(self.train_dataset, shuffle=True)

        # Resume from checkpoint if specified
        if self.config.resume_from_checkpoint:
            self._load_checkpoint(self.config.resume_from_checkpoint, optimizer, scheduler)

        # Training loop
        logger.info("Starting pretraining...")
        logger.info(f"  Num epochs = {self.config.num_epochs}")
        logger.info(f"  Per-device batch size = {self.config.per_device_batch_size}")
        logger.info(f"  Gradient accumulation steps = {self.config.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {self._get_total_steps(train_loader)}")

        total_loss = 0.0
        step_loss = 0.0
        start_time = time.time()

        for epoch in range(self.epoch, self.config.num_epochs):
            self.epoch = epoch
            self.model.train()

            if hasattr(train_loader.sampler, 'set_epoch'):
                train_loader.sampler.set_epoch(epoch)

            for step, batch in enumerate(train_loader):
                # Move batch to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

                # Forward pass
                outputs = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch.get("attention_mask"),
                    labels=batch["labels"],
                )

                loss = outputs["loss"]

                # Scale loss for gradient accumulation
                if self.config.gradient_accumulation_steps > 1:
                    loss = loss / self.config.gradient_accumulation_steps

                # Backward pass
                if self.config.use_deepspeed:
                    self.model.backward(loss)
                else:
                    loss.backward()

                step_loss += loss.item()

                # Optimizer step
                if (step + 1) % self.config.gradient_accumulation_steps == 0:
                    if self.config.use_deepspeed:
                        self.model.step()
                    else:
                        # Gradient clipping
                        if self.config.max_grad_norm > 0:
                            torch.nn.utils.clip_grad_norm_(
                                self.model.parameters(),
                                self.config.max_grad_norm
                            )
                        optimizer.step()
                        scheduler.step()
                        optimizer.zero_grad()

                    self.global_step += 1
                    total_loss += step_loss
                    avg_loss = step_loss * self.config.gradient_accumulation_steps

                    # Logging
                    if self.global_step % self.config.logging_steps == 0:
                        self._log_metrics({
                            "train/loss": avg_loss,
                            "train/learning_rate": scheduler.get_last_lr()[0],
                            "train/epoch": epoch + step / len(train_loader),
                            "train/global_step": self.global_step,
                        })

                    step_loss = 0.0

                    # Save checkpoint
                    if self.global_step % self.config.save_steps == 0:
                        self._save_checkpoint(optimizer, scheduler)

                    # Evaluation
                    if self.eval_dataset and self.global_step % self.config.eval_steps == 0:
                        eval_metrics = self.evaluate()
                        self._log_metrics(eval_metrics)

                    # Check max steps
                    if self.config.max_steps > 0 and self.global_step >= self.config.max_steps:
                        break

            if self.config.max_steps > 0 and self.global_step >= self.config.max_steps:
                break

        # Final save
        self._save_checkpoint(optimizer, scheduler, final=True)

        training_time = time.time() - start_time
        metrics = {
            "train/total_loss": total_loss / self.global_step,
            "train/total_steps": self.global_step,
            "train/training_time_hours": training_time / 3600,
        }

        logger.info(f"Training completed in {training_time / 3600:.2f} hours")

        return metrics

    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        """Run evaluation."""
        if self.eval_dataset is None:
            return {}

        self.model.eval()
        eval_loader = self._create_dataloader(self.eval_dataset, shuffle=False)

        total_loss = 0.0
        total_steps = 0

        for batch in eval_loader:
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

            outputs = self.model(
                input_ids=batch["input_ids"],
                attention_mask=batch.get("attention_mask"),
                labels=batch["labels"],
            )

            total_loss += outputs["loss"].item()
            total_steps += 1

        avg_loss = total_loss / total_steps
        perplexity = math.exp(min(avg_loss, 20))  # Cap at e^20

        self.model.train()

        return {
            "eval/loss": avg_loss,
            "eval/perplexity": perplexity,
        }

    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer with weight decay."""
        # Separate parameters that should and shouldn't have weight decay
        decay_params = []
        no_decay_params = []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if any(nd in name for nd in ["bias", "layernorm", "layer_norm", "norm"]):
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        optimizer_groups = [
            {"params": decay_params, "weight_decay": self.config.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]

        optimizer = torch.optim.AdamW(
            optimizer_groups,
            lr=self.config.learning_rate,
            betas=(self.config.adam_beta1, self.config.adam_beta2),
            eps=self.config.adam_epsilon,
        )

        return optimizer

    def _create_scheduler(self, optimizer: torch.optim.Optimizer):
        """Create learning rate scheduler."""
        from torch.optim.lr_scheduler import LambdaLR

        def lr_lambda(current_step: int) -> float:
            # Warmup
            if current_step < self.config.warmup_steps:
                return float(current_step) / float(max(1, self.config.warmup_steps))

            # Cosine decay
            progress = float(current_step - self.config.warmup_steps) / float(
                max(1, self.config.max_steps - self.config.warmup_steps)
            )
            return max(
                self.config.min_lr_ratio,
                0.5 * (1.0 + math.cos(math.pi * progress))
            )

        return LambdaLR(optimizer, lr_lambda)

    def _setup_deepspeed(self, optimizer, scheduler):
        """Initialize DeepSpeed engine."""
        try:
            import deepspeed
        except ImportError:
            raise ImportError("DeepSpeed is required for distributed training. Install with: pip install deepspeed")

        ds_config = self.config.get_deepspeed_config()

        model_engine, optimizer, _, scheduler = deepspeed.initialize(
            model=self.model,
            optimizer=optimizer,
            lr_scheduler=scheduler,
            config=ds_config,
        )

        return model_engine, optimizer, None, scheduler

    def _create_dataloader(self, dataset, shuffle: bool = True) -> DataLoader:
        """Create data loader with optional distributed sampler."""
        collator = PretrainCollator(
            pad_token_id=self.tokenizer.pad_token_id if self.tokenizer else 0,
        )

        sampler = None
        if self.is_distributed and not isinstance(dataset, torch.utils.data.IterableDataset):
            sampler = DistributedSampler(dataset, shuffle=shuffle)
            shuffle = False

        return DataLoader(
            dataset,
            batch_size=self.config.per_device_batch_size,
            shuffle=shuffle and sampler is None,
            sampler=sampler,
            num_workers=self.config.dataloader_num_workers,
            pin_memory=self.config.dataloader_pin_memory,
            collate_fn=collator,
            prefetch_factor=self.config.dataloader_prefetch_factor if self.config.dataloader_num_workers > 0 else None,
        )

    def _get_total_steps(self, train_loader: DataLoader) -> int:
        """Calculate total training steps."""
        if self.config.max_steps > 0:
            return self.config.max_steps

        num_batches = len(train_loader) if hasattr(train_loader, '__len__') else 1000000
        steps_per_epoch = num_batches // self.config.gradient_accumulation_steps
        return steps_per_epoch * self.config.num_epochs

    def _save_checkpoint(self, optimizer, scheduler, final: bool = False):
        """Save training checkpoint."""
        if not self.is_main_process:
            return

        checkpoint_name = "final" if final else f"step_{self.global_step}"
        checkpoint_path = Path(self.config.checkpoint_dir) / checkpoint_name

        # Save model
        if self.config.use_deepspeed:
            self.model.save_checkpoint(str(checkpoint_path))
        else:
            model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
            model_to_save.save_pretrained(str(checkpoint_path))

        # Save training state
        state = {
            "global_step": self.global_step,
            "epoch": self.epoch,
            "best_loss": self.best_loss,
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
            "config": self.config.to_dict(),
        }
        torch.save(state, checkpoint_path / "training_state.pt")

        logger.info(f"Saved checkpoint to {checkpoint_path}")

        # Clean up old checkpoints
        self._cleanup_checkpoints()

    def _load_checkpoint(self, checkpoint_path: str, optimizer, scheduler):
        """Load training checkpoint."""
        checkpoint_path = Path(checkpoint_path)

        # Load training state
        state_path = checkpoint_path / "training_state.pt"
        if state_path.exists():
            state = torch.load(state_path, map_location=self.device)
            self.global_step = state["global_step"]
            self.epoch = state["epoch"]
            self.best_loss = state.get("best_loss", float('inf'))
            optimizer.load_state_dict(state["optimizer_state_dict"])
            if scheduler and state.get("scheduler_state_dict"):
                scheduler.load_state_dict(state["scheduler_state_dict"])

        # Load model
        if self.config.use_deepspeed:
            self.model.load_checkpoint(str(checkpoint_path))
        else:
            model_to_load = self.model.module if hasattr(self.model, 'module') else self.model
            model_to_load = AutoRegressiveLM.from_pretrained(str(checkpoint_path), device=str(self.device))

        logger.info(f"Resumed from checkpoint {checkpoint_path} at step {self.global_step}")

    def _cleanup_checkpoints(self):
        """Remove old checkpoints beyond save_total_limit."""
        if self.config.save_total_limit <= 0:
            return

        checkpoint_dir = Path(self.config.checkpoint_dir)
        checkpoints = sorted(
            [d for d in checkpoint_dir.iterdir() if d.is_dir() and d.name.startswith("step_")],
            key=lambda x: int(x.name.split("_")[1])
        )

        while len(checkpoints) > self.config.save_total_limit:
            old_checkpoint = checkpoints.pop(0)
            import shutil
            shutil.rmtree(old_checkpoint)
            logger.info(f"Removed old checkpoint: {old_checkpoint}")

    def _log_metrics(self, metrics: Dict[str, float]):
        """Log metrics to console and W&B."""
        if not self.is_main_process:
            return

        # Log to console
        metrics_str = " | ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        logger.info(f"Step {self.global_step} | {metrics_str}")

        # Log to W&B
        if self.wandb:
            self.wandb.log(metrics, step=self.global_step)
