import os
import yaml
import logging
import wandb
import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
from transformers import PreTrainedTokenizerFast, get_scheduler
from typing import Dict, Any, Optional
from src.dataset import DatasetProcessor
from src.model import Transformer
from src.utils import setup_logger, save_checkpoint, load_checkpoint, set_seed, MemoryMonitor

class ConfigurationError(Exception):
    """Custom exception for configuration-related errors."""
    pass

class Trainer:
    def __init__(self, config_path: str):
        try:
            # Load configuration
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
                self._validate_config(self.config)

            # Setup logging
            self.logger = setup_logger(self.config['logging']['log_dir'], name='trainer')

            # Initialize W&B
            if self.config['logging'].get('use_wandb', True):
                wandb.init(
                    project=self.config['logging'].get('wandb_project', 'transformer-training'),
                    config=self.config
                )
                self.use_wandb = True
            else:
                self.use_wandb = False

            # Set random seeds for reproducibility
            set_seed(self.config['training']['seed'])

            # Load tokenizer
            self.tokenizer = PreTrainedTokenizerFast.from_pretrained(self.config['tokenizer']['path'])

            # Prepare datasets dynamically
            self.dataset_processor = DatasetProcessor(self.tokenizer, self.config['dataset'])
            self.train_dataset = self.dataset_processor.get_train_dataset()
            self.val_dataset = self.dataset_processor.get_val_dataset()

            self.train_loader = DataLoader(
                self.train_dataset,
                batch_size=self.config['training']['batch_size'],
                shuffle=True,
                num_workers=self.config['training'].get('num_workers', 4),
                pin_memory=True
            )

            self.val_loader = DataLoader(
                self.val_dataset,
                batch_size=self.config['training']['batch_size'],
                shuffle=False,
                num_workers=self.config['training'].get('num_workers', 4),
                pin_memory=True
            )

            # Initialize model
            self.model = Transformer(self.config['model']).to(self.config['training']['device'])

            # Setup optimizer, scheduler, and loss function
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.config['training']['learning_rate'],
                weight_decay=self.config['training'].get('weight_decay', 0.01)
            )

            num_training_steps = len(self.train_loader) * self.config['training']['epochs']
            self.scheduler = get_scheduler(
                name=self.config['training'].get('scheduler_type', 'linear'),
                optimizer=self.optimizer,
                num_warmup_steps=int(num_training_steps * self.config['training'].get('warmup_ratio', 0.1)),
                num_training_steps=num_training_steps
            )

            self.criterion = torch.nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)

            # Initialize memory monitor
            self.memory_monitor = MemoryMonitor()

            # Load checkpoint if specified
            self.start_epoch = 0
            if self.config['training']['resume_from_checkpoint']:
                self.start_epoch = load_checkpoint(
                    self.config['training']['resume_from_checkpoint'],
                    self.model,
                    self.optimizer,
                    self.scheduler
                )

            # Gradient accumulation setup
            self.gradient_accumulation_steps = self.config['training'].get('gradient_accumulation_steps', 1)

        except Exception as e:
            raise RuntimeError(f"Initialization failed: {e}")

    def _validate_config(self, config: dict):
        required_keys = ['logging', 'training', 'tokenizer', 'dataset', 'model']
        for key in required_keys:
            if key not in config:
                raise ConfigurationError(f"Missing required config key: {key}")
        if config['training']['batch_size'] <= 0:
            raise ConfigurationError("Batch size must be greater than 0.")

    def train_one_epoch(self, epoch: int):
        self.model.train()
        total_loss = 0
        progress_bar = tqdm(self.train_loader, desc=f"Training Epoch {epoch + 1}")

        for step, batch in enumerate(progress_bar):
            inputs, attention_mask, labels = batch
            inputs, attention_mask, labels = (
                inputs.to(self.config['training']['device']),
                attention_mask.to(self.config['training']['device']),
                labels.to(self.config['training']['device'])
            )

            with torch.cuda.amp.autocast(enabled=self.config['training'].get('use_mixed_precision', False)):
                outputs = self.model(inputs, attention_mask=attention_mask)
                loss = self.criterion(outputs.view(-1, outputs.size(-1)), labels.view(-1))
                loss = loss / self.gradient_accumulation_steps

            total_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

            # Backward pass
            loss.backward()

            # Gradient accumulation step
            if (step + 1) % self.gradient_accumulation_steps == 0 or (step + 1) == len(self.train_loader):
                clip_grad_norm_(self.model.parameters(), self.config['training'].get('max_grad_norm', 1.0))
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()

            # Log to W&B
            if self.use_wandb:
                wandb.log({
                    "train_loss": loss.item(),
                    "lr": self.scheduler.get_last_lr()[0]
                })

        avg_loss = total_loss / len(self.train_loader)
        self.logger.info(f"Epoch {epoch + 1} Training Loss: {avg_loss:.4f}")
        return avg_loss

    @torch.no_grad()
    def validate(self, epoch: int):
        self.model.eval()
        total_loss = 0
        progress_bar = tqdm(self.val_loader, desc=f"Validating Epoch {epoch + 1}")

        for batch in progress_bar:
            inputs, attention_mask, labels = batch
            inputs, attention_mask, labels = (
                inputs.to(self.config['training']['device']),
                attention_mask.to(self.config['training']['device']),
                labels.to(self.config['training']['device'])
            )

            outputs = self.model(inputs, attention_mask=attention_mask)
            loss = self.criterion(outputs.view(-1, outputs.size(-1)), labels.view(-1))
            total_loss += loss.item()

        avg_loss = total_loss / len(self.val_loader)
        self.logger.info(f"Epoch {epoch + 1} Validation Loss: {avg_loss:.4f}")

        # Log to W&B
        if self.use_wandb:
            wandb.log({"val_loss": avg_loss})

        return avg_loss

    def train(self):
        best_val_loss = float('inf')

        for epoch in range(self.start_epoch, self.config['training']['epochs']):
            self.logger.info(f"Starting Epoch {epoch + 1}/{self.config['training']['epochs']}")
            train_loss = self.train_one_epoch(epoch)
            val_loss = self.validate(epoch)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(
                    self.config['training']['checkpoint_dir'],
                    self.model,
                    self.optimizer,
                    self.scheduler,
                    epoch
                )
                self.logger.info(f"Saved best model checkpoint at epoch {epoch + 1}")

            self.logger.info(f"Epoch {epoch + 1} Summary: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")

            # Log memory usage
            memory_usage = self.memory_monitor.log_memory_usage()
            self.logger.info(f"Memory Usage: {memory_usage}")

            # Log to W&B
            if self.use_wandb:
                wandb.log({
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "memory_usage": memory_usage
                })

        self.logger.info("Training completed.")
