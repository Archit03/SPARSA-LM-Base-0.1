"""
SPARSA-LM Multi-GPU Training with DeepSpeed ZeRO

Features:
- DeepSpeed ZeRO-2/ZeRO-3 optimization
- Multi-GPU distributed training (4x L4 24GB)
- Mixed precision training (BF16)
- Gradient checkpointing
- Automatic batch size scaling
- WandB integration

This code belongs to EllanorAI and is licensed under the EllanorAI Proprietary License.
"""

import os
import sys
import yaml
import json
import logging
import time
import math
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple, List
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, DistributedSampler
import numpy as np

# Optional DeepSpeed import
try:
    import deepspeed
    from deepspeed import DeepSpeedEngine
    from deepspeed.runtime.zero.stage3 import estimate_zero3_model_states_mem_needs_all_live
    DEEPSPEED_AVAILABLE = True
except ImportError:
    DEEPSPEED_AVAILABLE = False
    logging.warning("DeepSpeed not available. Install with: pip install deepspeed")

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

try:
    from transformers import AutoTokenizer, get_scheduler
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False

from tqdm import tqdm

# Import local modules
from model import SPARSALM, SPARSAConfig, create_model
from dataset import (
    StreamingPretrainDataset,
    InstructionDataset,
    DatasetConfig,
    create_dataloader,
    get_tokenizer,
)


@dataclass
class TrainingConfig:
    """Training configuration."""

    # Basic settings
    epochs: int = 100
    batch_size: int = 4
    gradient_accumulation_steps: int = 8
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    max_grad_norm: float = 1.0
    warmup_ratio: float = 0.03

    # Model settings
    model_path: Optional[str] = None
    tokenizer_path: str = "tokenizer"
    max_seq_len: int = 2048

    # DeepSpeed settings
    use_deepspeed: bool = True
    deepspeed_config: str = "config/deepspeed_config.json"
    zero_stage: int = 2

    # Hardware settings
    device: str = "cuda"
    bf16: bool = True
    fp16: bool = False

    # Checkpointing
    checkpoint_dir: str = "checkpoints/sparsa-360m"
    save_steps: int = 1000
    eval_steps: int = 500
    logging_steps: int = 10

    # Distributed settings
    local_rank: int = -1
    world_size: int = 1

    # Logging
    use_wandb: bool = True
    wandb_project: str = "SPARSA-LM-360M"

    # Early stopping
    early_stopping_patience: int = 5

    # Seed
    seed: int = 42


def setup_logging(log_dir: str, rank: int = 0) -> logging.Logger:
    """Setup logging with file and console handlers."""
    os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger("sparsa_training")
    logger.setLevel(logging.INFO if rank == 0 else logging.WARNING)

    # File handler
    fh = logging.FileHandler(os.path.join(log_dir, f"training_rank{rank}.log"))
    fh.setLevel(logging.DEBUG)

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO if rank == 0 else logging.WARNING)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_optimizer_grouped_params(
    model: nn.Module,
    weight_decay: float,
    learning_rate: float,
) -> List[Dict]:
    """Get optimizer param groups with weight decay separation."""
    no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight", "norm.weight"]

    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters()
                      if not any(nd in n for nd in no_decay) and p.requires_grad],
            "weight_decay": weight_decay,
            "lr": learning_rate,
        },
        {
            "params": [p for n, p in model.named_parameters()
                      if any(nd in n for nd in no_decay) and p.requires_grad],
            "weight_decay": 0.0,
            "lr": learning_rate,
        },
    ]

    return optimizer_grouped_parameters


class DeepSpeedTrainer:
    """Trainer with DeepSpeed integration for multi-GPU training."""

    def __init__(
        self,
        config: TrainingConfig,
        model_config: Dict[str, Any],
        data_config: Optional[DatasetConfig] = None,
    ):
        self.config = config
        self.model_config = model_config
        self.data_config = data_config or DatasetConfig(max_seq_len=config.max_seq_len)

        # Setup distributed training
        self._setup_distributed()

        # Setup logging
        self.logger = setup_logging(
            os.path.join(config.checkpoint_dir, "logs"),
            rank=self.local_rank,
        )

        # Set seed
        set_seed(config.seed + self.local_rank)

        # Initialize tokenizer
        self.tokenizer = self._setup_tokenizer()

        # Initialize model
        self.model = self._setup_model()

        # Initialize datasets
        self.train_dataset, self.val_dataset = self._setup_datasets()

        # Initialize DeepSpeed engine
        self.model_engine, self.optimizer, self.train_loader, self.scheduler = self._setup_deepspeed()

        # Training state
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.patience_counter = 0

        # WandB
        if config.use_wandb and WANDB_AVAILABLE and self.local_rank == 0:
            wandb.init(
                project=config.wandb_project,
                config={**vars(config), **model_config},
            )
            self.use_wandb = True
        else:
            self.use_wandb = False

    def _setup_distributed(self):
        """Setup distributed training environment."""
        if self.config.use_deepspeed and DEEPSPEED_AVAILABLE:
            deepspeed.init_distributed()
            self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
            self.world_size = int(os.environ.get("WORLD_SIZE", 1))
        else:
            self.local_rank = 0
            self.world_size = 1

        if torch.cuda.is_available():
            torch.cuda.set_device(self.local_rank)
            self.device = torch.device("cuda", self.local_rank)
        else:
            self.device = torch.device("cpu")

        if self.local_rank == 0:
            print(f"Distributed training: world_size={self.world_size}, local_rank={self.local_rank}")

    def _setup_tokenizer(self):
        """Initialize tokenizer."""
        tokenizer = get_tokenizer(
            self.config.tokenizer_path,
            vocab_size=self.model_config.get("vocab_size", 32000),
        )

        # Resize model vocab if needed
        self.model_config["vocab_size"] = len(tokenizer)

        return tokenizer

    def _setup_model(self) -> SPARSALM:
        """Initialize or load model."""
        if self.config.model_path and os.path.exists(self.config.model_path):
            self.logger.info(f"Loading model from {self.config.model_path}")
            model = SPARSALM.from_pretrained(self.config.model_path)
        else:
            self.logger.info("Creating new model")
            model = create_model(self.model_config)

        # Log model size
        num_params = model.num_parameters()
        self.logger.info(f"Model parameters: {num_params:,} ({num_params/1e6:.1f}M)")

        return model

    def _setup_datasets(self):
        """Initialize datasets."""
        self.logger.info("Setting up datasets...")

        train_dataset = StreamingPretrainDataset(
            config=self.data_config,
            tokenizer=self.tokenizer,
            split="train",
        )

        val_dataset = StreamingPretrainDataset(
            config=self.data_config,
            tokenizer=self.tokenizer,
            split="validation",
        )

        return train_dataset, val_dataset

    def _setup_deepspeed(self):
        """Initialize DeepSpeed engine."""
        # Load DeepSpeed config
        if os.path.exists(self.config.deepspeed_config):
            with open(self.config.deepspeed_config) as f:
                ds_config = json.load(f)
        else:
            # Default config
            ds_config = self._get_default_deepspeed_config()

        # Update config with training params
        ds_config["train_micro_batch_size_per_gpu"] = self.config.batch_size
        ds_config["gradient_accumulation_steps"] = self.config.gradient_accumulation_steps

        # Setup optimizer params
        optimizer_params = get_optimizer_grouped_params(
            self.model,
            self.config.weight_decay,
            self.config.learning_rate,
        )

        # Create data loader
        train_loader = create_dataloader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=4,
        )

        # Calculate training steps
        num_training_steps = len(train_loader) * self.config.epochs
        num_warmup_steps = int(num_training_steps * self.config.warmup_ratio)

        if self.config.use_deepspeed and DEEPSPEED_AVAILABLE:
            # Initialize DeepSpeed
            model_engine, optimizer, _, scheduler = deepspeed.initialize(
                model=self.model,
                model_parameters=optimizer_params,
                config=ds_config,
            )

            self.logger.info(f"DeepSpeed initialized with ZeRO stage {ds_config.get('zero_optimization', {}).get('stage', 0)}")
        else:
            # Fallback to standard PyTorch
            model_engine = self.model.to(self.device)

            optimizer = torch.optim.AdamW(
                optimizer_params,
                lr=self.config.learning_rate,
                betas=(0.9, 0.95),
                eps=1e-8,
            )

            scheduler = get_scheduler(
                "cosine",
                optimizer=optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps,
            )

        return model_engine, optimizer, train_loader, scheduler

    def _get_default_deepspeed_config(self) -> Dict:
        """Get default DeepSpeed configuration."""
        return {
            "train_batch_size": "auto",
            "train_micro_batch_size_per_gpu": self.config.batch_size,
            "gradient_accumulation_steps": self.config.gradient_accumulation_steps,
            "gradient_clipping": self.config.max_grad_norm,

            "zero_optimization": {
                "stage": self.config.zero_stage,
                "overlap_comm": True,
                "contiguous_gradients": True,
                "reduce_bucket_size": 5e8,
                "stage3_prefetch_bucket_size": 5e8,
                "stage3_param_persistence_threshold": 1e6,
            },

            "bf16": {
                "enabled": self.config.bf16,
            },

            "fp16": {
                "enabled": self.config.fp16,
                "loss_scale": 0,
                "initial_scale_power": 16,
            },

            "optimizer": {
                "type": "AdamW",
                "params": {
                    "lr": self.config.learning_rate,
                    "betas": [0.9, 0.95],
                    "eps": 1e-8,
                    "weight_decay": self.config.weight_decay,
                }
            },

            "scheduler": {
                "type": "WarmupDecayLR",
                "params": {
                    "warmup_min_lr": 0,
                    "warmup_max_lr": self.config.learning_rate,
                    "warmup_num_steps": "auto",
                    "total_num_steps": "auto",
                }
            },

            "activation_checkpointing": {
                "partition_activations": True,
                "contiguous_memory_optimization": True,
            },

            "wall_clock_breakdown": False,
        }

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Execute single training step."""
        self.model_engine.train()

        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        labels = batch["labels"].to(self.device)

        # Forward pass
        outputs = self.model_engine(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )

        # Compute loss
        logits = outputs["logits"]
        loss = self.model_engine.module.compute_loss(logits, labels) if hasattr(self.model_engine, 'module') else self.model.compute_loss(logits, labels)

        # Backward pass
        if self.config.use_deepspeed and DEEPSPEED_AVAILABLE:
            self.model_engine.backward(loss)
            self.model_engine.step()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()

        # Compute metrics
        with torch.no_grad():
            perplexity = torch.exp(loss).item()

        return {
            "loss": loss.item(),
            "perplexity": perplexity,
        }

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Run validation."""
        self.model_engine.eval()

        val_loader = create_dataloader(
            self.val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=2,
        )

        total_loss = 0.0
        total_steps = 0

        for batch in tqdm(val_loader, desc="Validating", disable=self.local_rank != 0):
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)

            outputs = self.model_engine(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True,
            )

            logits = outputs["logits"]
            loss = self.model_engine.module.compute_loss(logits, labels) if hasattr(self.model_engine, 'module') else self.model.compute_loss(logits, labels)

            total_loss += loss.item()
            total_steps += 1

            if total_steps >= 100:  # Limit validation steps
                break

        avg_loss = total_loss / max(total_steps, 1)
        perplexity = math.exp(avg_loss)

        return {
            "val_loss": avg_loss,
            "val_perplexity": perplexity,
        }

    def save_checkpoint(self, tag: str = "latest"):
        """Save model checkpoint."""
        if self.local_rank != 0:
            return

        checkpoint_path = os.path.join(self.config.checkpoint_dir, tag)
        os.makedirs(checkpoint_path, exist_ok=True)

        if self.config.use_deepspeed and DEEPSPEED_AVAILABLE:
            self.model_engine.save_checkpoint(checkpoint_path)
        else:
            # Save model
            model_to_save = self.model_engine.module if hasattr(self.model_engine, 'module') else self.model_engine
            model_to_save.save_pretrained(checkpoint_path)

            # Save optimizer and scheduler
            torch.save({
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
                "global_step": self.global_step,
                "best_val_loss": self.best_val_loss,
            }, os.path.join(checkpoint_path, "training_state.pt"))

        # Save tokenizer
        self.tokenizer.save_pretrained(checkpoint_path)

        self.logger.info(f"Saved checkpoint to {checkpoint_path}")

    def train(self):
        """Main training loop."""
        self.logger.info("Starting training...")
        self.logger.info(f"Epochs: {self.config.epochs}")
        self.logger.info(f"Batch size: {self.config.batch_size}")
        self.logger.info(f"Gradient accumulation: {self.config.gradient_accumulation_steps}")
        self.logger.info(f"Effective batch size: {self.config.batch_size * self.config.gradient_accumulation_steps * self.world_size}")

        for epoch in range(self.config.epochs):
            self.logger.info(f"\n{'='*50}")
            self.logger.info(f"Epoch {epoch + 1}/{self.config.epochs}")

            epoch_loss = 0.0
            epoch_steps = 0

            progress_bar = tqdm(
                self.train_loader,
                desc=f"Epoch {epoch + 1}",
                disable=self.local_rank != 0,
            )

            for step, batch in enumerate(progress_bar):
                metrics = self.train_step(batch)

                epoch_loss += metrics["loss"]
                epoch_steps += 1
                self.global_step += 1

                # Update progress bar
                progress_bar.set_postfix({
                    "loss": f"{metrics['loss']:.4f}",
                    "ppl": f"{metrics['perplexity']:.2f}",
                })

                # Logging
                if self.global_step % self.config.logging_steps == 0:
                    if self.use_wandb:
                        wandb.log({
                            "train/loss": metrics["loss"],
                            "train/perplexity": metrics["perplexity"],
                            "train/learning_rate": self.scheduler.get_last_lr()[0] if hasattr(self.scheduler, 'get_last_lr') else self.config.learning_rate,
                            "train/global_step": self.global_step,
                        })

                # Evaluation
                if self.global_step % self.config.eval_steps == 0:
                    val_metrics = self.validate()
                    self.logger.info(f"Step {self.global_step} - Val Loss: {val_metrics['val_loss']:.4f}, Val PPL: {val_metrics['val_perplexity']:.2f}")

                    if self.use_wandb:
                        wandb.log({
                            "val/loss": val_metrics["val_loss"],
                            "val/perplexity": val_metrics["val_perplexity"],
                        })

                    # Early stopping check
                    if val_metrics["val_loss"] < self.best_val_loss:
                        self.best_val_loss = val_metrics["val_loss"]
                        self.patience_counter = 0
                        self.save_checkpoint("best")
                    else:
                        self.patience_counter += 1

                    if self.patience_counter >= self.config.early_stopping_patience:
                        self.logger.info("Early stopping triggered")
                        break

                # Save checkpoint
                if self.global_step % self.config.save_steps == 0:
                    self.save_checkpoint(f"step_{self.global_step}")

            # Epoch summary
            avg_epoch_loss = epoch_loss / max(epoch_steps, 1)
            self.logger.info(f"Epoch {epoch + 1} - Avg Loss: {avg_epoch_loss:.4f}")

            # Save epoch checkpoint
            self.save_checkpoint(f"epoch_{epoch + 1}")

            if self.patience_counter >= self.config.early_stopping_patience:
                break

        # Final save
        self.save_checkpoint("final")
        self.logger.info("Training completed!")

        if self.use_wandb:
            wandb.finish()


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    """Main entry point for training."""
    import argparse

    parser = argparse.ArgumentParser(description="SPARSA-LM Training")
    parser.add_argument("--config", type=str, default="config/training_config.yaml")
    parser.add_argument("--deepspeed_config", type=str, default="config/deepspeed_config.json")
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    args = parser.parse_args()

    # Load config
    config_dict = load_config(args.config)

    # Create training config
    training_config = TrainingConfig(
        epochs=args.epochs or config_dict["training"].get("epochs", 100),
        batch_size=args.batch_size or config_dict["training"].get("batch_size", 4),
        gradient_accumulation_steps=config_dict["training"].get("gradient_accumulation_steps", 8),
        learning_rate=args.learning_rate or config_dict["training"].get("learning_rate", 3e-4),
        weight_decay=config_dict["training"].get("weight_decay", 0.1),
        max_grad_norm=config_dict["training"].get("max_grad_norm", 1.0),
        warmup_ratio=config_dict["training"].get("warmup_ratio", 0.03),
        tokenizer_path=config_dict["tokenizer"].get("path", "tokenizer"),
        max_seq_len=config_dict["model"].get("max_seq_len", 2048),
        use_deepspeed=config_dict["training"].get("use_deepspeed", True),
        deepspeed_config=args.deepspeed_config,
        zero_stage=2,
        bf16=config_dict["training"].get("bf16", True),
        checkpoint_dir=config_dict["training"].get("checkpoint_dir", "checkpoints/sparsa-360m"),
        save_steps=config_dict["training"].get("checkpoint_save_frequency", 1000),
        eval_steps=500,
        logging_steps=config_dict["training"].get("log_every_n_steps", 10),
        use_wandb=config_dict["logging"].get("use_wandb", True),
        wandb_project=config_dict["logging"].get("wandb_project", "SPARSA-LM-360M"),
        early_stopping_patience=config_dict["training"].get("early_stopping_patience", 5),
        seed=config_dict["training"].get("seed", 42),
        local_rank=args.local_rank,
    )

    # Model config
    model_config = config_dict["model"]

    # Initialize trainer
    trainer = DeepSpeedTrainer(
        config=training_config,
        model_config=model_config,
    )

    # Start training
    trainer.train()


if __name__ == "__main__":
    main()
