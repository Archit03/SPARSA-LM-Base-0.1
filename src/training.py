import os
import yaml
import logging
import wandb
import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
from transformers import PreTrainedTokenizerFast, get_scheduler, get_linear_schedule_with_warmup
from typing import Dict, Any, Optional, Tuple
import transformers
import traceback

# ----------------------------------------------------------------------
# Import your dataset/module references:
# ----------------------------------------------------------------------
from dataset import DatasetProcessor, TextDataset, add_noise_to_input
from model import Transformer, TransformerConfig
from utils import setup_logging, save_checkpoint, load_checkpoint, set_seed, MemoryMonitor
from torch.amp import GradScaler
import optuna
from sklearn.model_selection import train_test_split


# ----------------------------------------------------------------------
# Debugging Utilities
# ----------------------------------------------------------------------
def debug_tensor_values(tensor: torch.Tensor, name: str, logger: logging.Logger) -> bool:
    """
    Debug helper to analyze tensor values, shape, and stats.
    """
    try:
        if tensor is None:
            logger.error(f"{name} is None")
            return False
        
        stats = {
            "shape": tensor.shape,
            "dtype": tensor.dtype,
            "device": str(tensor.device),
            "min": tensor.min().item(),
            "max": tensor.max().item(),
            "mean": tensor.float().mean().item(),
            "num_nan": torch.isnan(tensor).sum().item(),
            "num_inf": torch.isinf(tensor).sum().item()
        }
        
        logger.info(f"{name} stats: {stats}")
        return True
    except Exception as e:
        logger.error(f"Error analyzing {name}: {e}")
        return False

def validate_model_outputs(outputs: torch.Tensor,
                           labels: torch.Tensor,
                           vocab_size: int,
                           logger: logging.Logger) -> bool:
    """
    Validate model outputs and labels before loss calculation.
    """
    try:
        # 1. Dimension check
        if outputs.dim() != labels.dim() + 1:
            logger.error(f"Dimension mismatch: outputs={outputs.shape}, labels={labels.shape}")
            return False
        
        # 2. Check overall stats for outputs & labels
        if not debug_tensor_values(outputs, "outputs", logger):
            return False
        if not debug_tensor_values(labels, "labels", logger):
            return False
        
        # 3. Validate label values against vocab_size
        flat_labels = labels.view(-1)
        invalid_labels = (flat_labels >= vocab_size) & (flat_labels != -100)
        if invalid_labels.any():
            invalid_indices = torch.where(invalid_labels)[0]
            invalid_values = flat_labels[invalid_labels]
            logger.error(f"Invalid label values (>= vocab_size): {invalid_values.cpu().tolist()}")
            logger.error(f"Positions with invalid values: {invalid_indices.cpu().tolist()}")
            logger.error(f"Vocabulary size: {vocab_size}")
            return False
        
        # 4. Check for negative labels other than -100
        negative_labels = (flat_labels < 0) & (flat_labels != -100)
        if negative_labels.any():
            neg_indices = torch.where(negative_labels)[0]
            neg_values = flat_labels[negative_labels]
            logger.error(f"Negative label values (not -100): {neg_values.cpu().tolist()}")
            logger.error(f"Positions with negative values: {neg_indices.cpu().tolist()}")
            return False
        
        return True
    except Exception as e:
        logger.error(f"Validation error: {e}")
        return False

def debug_loss_inputs(outputs: torch.Tensor,
                      labels: torch.Tensor,
                      vocab_size: int,
                      logger: logging.Logger) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Prepare and validate inputs for loss calculation by flattening
    and checking dimension / range correctness.
    """
    try:
        # Flatten the outputs and labels
        flat_outputs = outputs.view(-1, outputs.size(-1))
        flat_labels = labels.view(-1)
        
        # Basic shape checks
        if flat_outputs.size(0) != flat_labels.size(0):
            logger.error(
                f"Size mismatch after flattening: "
                f"outputs={flat_outputs.shape}, labels={flat_labels.shape}"
            )
            return None, None
        
        if flat_outputs.size(1) != vocab_size:
            logger.error(
                f"Output size mismatch: got {flat_outputs.size(1)}, "
                f"expected {vocab_size} (check model vocab_size or tokenizer vocab_size)."
            )
            return None, None
        
        # Validate the combined shapes and value ranges
        if not validate_model_outputs(flat_outputs, flat_labels, vocab_size, logger):
            return None, None
        
        return flat_outputs, flat_labels
    except Exception as e:
        logger.error(f"Error in debug_loss_inputs: {e}")
        return None, None

# ----------------------------------------------------------------------
# Additional Classes/Exceptions
# ----------------------------------------------------------------------
class ConfigurationError(Exception):
    """Custom exception for configuration-related errors."""
    pass

# ----------------------------------------------------------------------
# Trainer Class
# ----------------------------------------------------------------------
class Trainer:
    def __init__(self, training_config: Dict, data_config: Dict):
        try:
            # Validate configuration first
            self._validate_config(training_config)

            # Assign configurations
            self.config = training_config
            self.data_config = data_config

            # Early stopping setup
            self.best_metric = float('inf')
            self.stopping_counter = 0
            self.early_stopping_patience = self.config['training'].get('early_stopping_patience', 3)

            # Setup logging
            self.logger = setup_logging(self.config['logging']['log_dir'], name='LuminaLM_trainer')

            # Initialize W&B
            if self.config['logging'].get('use_wandb', True):
                wandb.init(
                    project=self.config['logging'].get('wandb_project', 'LuminaLM-training'),
                    config=self.config
                )
                self.use_wandb = True
            else:
                self.use_wandb = False

            # Set random seed
            set_seed(self.config['training']['seed'])

            # Load tokenizer
            tokenizer_path = self.config['tokenizer']['path']
            self.tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)
            self.logger.info(f"Tokenizer loaded with vocab size: {self.tokenizer.vocab_size}")

            if self.tokenizer.pad_token is None or self.tokenizer.pad_token_id is None:
                self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            else:
                self.logger.info(f"PAD token already exists: {self.tokenizer.pad_token} (ID: {self.tokenizer.pad_token_id})")

            # Load dataset configuration
            datasets = self.data_config.get('datasets', [])
            if not isinstance(datasets, list):
                raise ValueError("'datasets' must be a list.")

            source_dir = next(
                (d.get('config', {}).get('source_dir') for d in datasets if d.get('name') == self.config['dataset']['train_dataset']),
                None
            )
            if not source_dir or not os.path.exists(source_dir):
                raise ValueError(f"Invalid source directory: {source_dir}")

            # Dataset Processor
            split_config = self.config['dataset'].get('split', {'test_size': 0.2, 'random_state': 42})
            preprocessing_config = self.config['dataset'].get('preprocessing', {})

            self.dataset_processor = DatasetProcessor(
                source_dir=source_dir,
                split=split_config,
                preprocessing_config=preprocessing_config
            )

            # Load datasets
            max_length = self.config['dataset']['max_seq_len']
            self.train_dataset = self.dataset_processor.get_train_dataset(tokenizer=self.tokenizer, max_length=max_length)
            self.val_dataset = self.dataset_processor.get_val_dataset(tokenizer=self.tokenizer, max_length=max_length)

            # Initialize device
            self.device = torch.device(self.config['training'].get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))

            # Data Loaders
            self.train_loader = DataLoader(
                dataset=self.train_dataset,
                batch_size=self.config['training']['batch_size'],
                shuffle=True,
                num_workers=self.config['training'].get('num_workers', 1),
                pin_memory=self.device.type == "cuda",
                collate_fn=TextDataset.collate_fn,
                drop_last=self.config['training'].get('drop_last', False)
            )
            self.val_loader = DataLoader(
                dataset=self.val_dataset,
                batch_size=self.config['training']['batch_size'],
                shuffle=False,
                num_workers=self.config['training'].get('num_workers', 1),
                pin_memory=self.device.type == "cuda",
                collate_fn=TextDataset.collate_fn,
                drop_last=self.config['training'].get('drop_last', False)
            )

            # Scheduler Setup
            scheduler_type = self.config["training"].get("scheduler_type", "linear")
            lr_scheduler_kwargs = self.config["training"].get("lr_scheduler_kwargs", {})
            if scheduler_type == "cosine_with_min_lr":
                if 'min_lr' not in lr_scheduler_kwargs and 'min_lr_rate' not in lr_scheduler_kwargs:
                    self.logger.warning("`min_lr` not found in `lr_scheduler_kwargs`, using default 1e-6")
                    lr_scheduler_kwargs['min_lr'] = 1e-6

            if scheduler_type == "cosine_with_min_lr" and 'min_lr' not in lr_scheduler_kwargs and 'min_lr_rate' not in lr_scheduler_kwargs:
                raise ValueError("`lr_scheduler_kwargs` must contain either `min_lr` or `min_lr_rate`.")

            # Gradient Accumulation
            self.enable_gradient_accumulation = self.config['training'].get('enable_gradient_accumulation', True)
            self.gradient_accumulation_steps = self.config['training'].get('gradient_accumulation_steps', 1) if self.enable_gradient_accumulation else 1

            # Model Configuration
            checkpointing_params = self.config['model'].get('checkpointing_params', {})

            model_config = TransformerConfig(
                d_model=self.config['model']['hidden_dim'],
                num_heads=self.config['model']['num_heads'],
                window_size=self.config['model'].get('window_size', 4),
                global_tokens=self.config['model'].get('global_tokens', 0),
                d_ff=self.config['model']['ff_dim'],
                num_layers=self.config['model']['num_layers'],
                dropout=self.config['model']['dropout'],
                max_seq_len=self.config['model']['max_seq_len'],
                activation=self.config['model'].get('activation', 'gelu'),
                use_rope=self.config['model'].get('use_rope', True),
                prenorm=self.config['model'].get('prenorm', True),
                vocab_size=self.config['model']['vocab_size'],
                tie_embeddings=self.config['model'].get('tie_embeddings', False),
                scheduler_type=scheduler_type,
                learning_rate=self.config['training']['learning_rate'],
                weight_decay=self.config['training']['weight_decay'],
                warmup_ratio=self.config['training'].get('warmup_ratio', 0.1),
                use_mixed_precision=self.config['training'].get('use_mixed_precision', True),
                max_grad_norm=self.config['training'].get('max_grad_norm', 1.0),
                pad_token_id=self.tokenizer.pad_token_id,
                l2_reg=self.config['training'].get('l2_reg', 0.0),
                use_checkpointing=self.config['model'].get('use_checkpointing', False),
                use_reentrant=checkpointing_params.get('use_reentrant', False),
                noise_type=self.config['training'].get('noise_type', 'mask'),
                noise_prob=self.config['training'].get('noise_prob', 0.3)
            )

            # Initialize Model
            self.model = Transformer(model_config).to(self.device)

            def _init_weights(module):
                if isinstance(module, (torch.nn.Linear, torch.nn.Embedding)):
                    module.weight.data.normal_(mean=0.0, std=0.02)
                    if isinstance(module, torch.nn.Linear) and module.bias is not None:
                        module.bias.data.zero_()
                elif isinstance(module, torch.nn.LayerNorm):
                    module.bias.data.zero_()
                    module.weight.data.fill_(1.0)

            self.model.apply(_init_weights) 

            # Optimizer Setup
            self.optimizer = self._configure_optimizer()

            # Scheduler Initialization
            num_training_steps = len(self.train_loader) * self.config['training']['epochs']
            self.scheduler = self._configure_scheduler(self.optimizer, self.config['training'], num_training_steps)

            # Loss Function
            self.criterion = torch.nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id, label_smoothing=0.00)

            # Memory Monitor
            self.memory_monitor = MemoryMonitor()

            # Checkpoints
            self.start_epoch = 0
            if self.config['training'].get('resume_from_checkpoint'):
                self.start_epoch = load_checkpoint(
                    self.config['training']['resume_from_checkpoint'],
                    self.model,
                    self.optimizer,
                    self.scheduler
                )

            # Mixed Precision
            self.scaler = GradScaler(enabled=self.config['training'].get('use_mixed_precision', True))

        except Exception as e:
            raise RuntimeError(f"LuminaLM Initialization failed: {e}")

    def _setup_distributed_training(self):
        """
        Enable distributed training if multiple GPUs are available.
        """
        if torch.cuda.device_count() > 1:
            self.model = torch.nn.DataParallel(self.model)
            self.logger.info(f"Using {torch.cuda.device_count()} GPUs for LuminaLM")
    
    # ----------------------------------------------------------------------
    # Configure Scheduler
    # ----------------------------------------------------------------------
    def _configure_scheduler(self, optimizer, config, num_training_steps):
        """
        Configure the learning rate scheduler with proper handling of cosine with min LR.
        """
        num_warmup_steps = int(num_training_steps * config.get('warmup_ratio', 0.1))
        
        scheduler_type = config.get("scheduler_type", "linear")
        lr_scheduler_kwargs = config.get("lr_scheduler_kwargs", {})

        if scheduler_type == "cosine_with_min_lr":
            # Ensure min_lr is a valid float
            min_lr = float(lr_scheduler_kwargs.get('min_lr', 1e-6))
            
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=num_training_steps - num_warmup_steps,
                eta_min=min_lr
            )

        elif scheduler_type == "linear_warmup":
            return transformers.get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps
            )

        else:
            raise ValueError(f"Unknown scheduler type: {scheduler_type}")

    # ----------------------------------------------------------------------
    # Configure Optimizer
    # ----------------------------------------------------------------------
    def _configure_optimizer(self):
        """
        Modified optimizer configuration with improved numerical stability
        """
        decay_params = []
        no_decay_params = []

        # Initialize parameter groups with proper scaling
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue

            # Scale initial values if they're too large
            if param.abs().max() > 1.0:
                with torch.no_grad():
                    param.data.div_(param.abs().max())

            if any(nd in name for nd in ['bias', 'LayerNorm.weight']):
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        optimizer_grouped_parameters = [
            {
                'params': decay_params,
                'weight_decay': self.config['training']['weight_decay']
            },
            {
                'params': no_decay_params,
                'weight_decay': 0.0
            }
        ]

        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=self.config['training']['learning_rate'],
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=self.config['training']['weight_decay'],
            amsgrad=True
        )

        return optimizer

    def _validate_config(self, config: Dict):
        """
        Comprehensive configuration validation to ensure required keys exist.
        """
        try:
            required_nested_keys = {
                'training': ['device', 'epochs', 'batch_size', 'learning_rate'],
                'model': ['num_layers', 'num_heads', 'hidden_dim'],
                'logging': ['log_dir']
            }

            for section, keys in required_nested_keys.items():
                if section not in config:
                    raise ConfigurationError(f"Missing required section: {section}")
                for key in keys:
                    if key not in config[section]:
                        raise ConfigurationError(f"Missing required key '{key}' in {section} section")

            # Validate and convert learning rate
            try:
                config['training']['learning_rate'] = float(config['training']['learning_rate'])
            except (ValueError, TypeError):
                raise ConfigurationError("Invalid learning rate value")

            # Validate device configuration
            if 'cuda' in str(config['training']['device']) and not torch.cuda.is_available():
                raise ConfigurationError("CUDA device specified but not available")

        except Exception as e:
            raise ConfigurationError(f"Configuration validation failed: {str(e)}")

    def _calculate_perplexity(self, loss: float) -> float:
        """
        Calculate perplexity from a scalar loss value.
        """
        return float(torch.exp(torch.tensor(loss)))

    def _early_stopping(self, val_metric: float) -> bool:
        """
        Enhanced early stopping with best model preservation.
        """
        improved = val_metric < self.best_metric
        if improved:
            self.best_metric = val_metric
            self.stopping_counter = 0
            self._save_best_model()
            return False
        else:
            self.stopping_counter += 1
        
        if self.stopping_counter >= self.early_stopping_patience:
            self.logger.info(
                f"Early stopping triggered after {self.stopping_counter} epochs without improvement"
            )
            self._restore_best_model()
            return True
        
        return False

    def _save_best_model(self):
        """
        Save the best model state to disk.
        """
        best_model_path = os.path.join(self.config['training']['checkpoint_dir'], 'best_model.pt')
        torch.save({
            'epoch': self.start_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_metric': self.best_metric,
            'config': self.config
        }, best_model_path)
        self.logger.info(f"Saved best model to {best_model_path}")

    def _restore_best_model(self):
        """
        Restore the best model state from disk.
        """
        best_model_path = os.path.join(self.config['training']['checkpoint_dir'], 'best_model.pt')
        if os.path.exists(best_model_path):
            checkpoint = torch.load(best_model_path, map_location=str(self.device), weights_only=True)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            self.logger.info("Restored best model state")
        else:
            self.logger.warning("No best model checkpoint found to restore")

    def _log_training_metrics(self, loss: float, step: int, epoch: int):
        """
        Structured logging of training metrics to W&B.
        """
        if not self.use_wandb:
            return
        
        metrics = {
            "train/loss": loss,
            "train/learning_rate": self.scheduler.get_last_lr()[0],
            "train/epoch": epoch,
            "train/step": step,
        }

        if torch.cuda.is_available():
            metrics.update({
                "system/gpu_memory_allocated": torch.cuda.memory_allocated(),
                "system/gpu_memory_reserved": torch.cuda.memory_reserved()
            })

        wandb.log(metrics)

    def _compute_gradient_norm(self) -> float:
        """
        Compute the total gradient norm for debugging or logging.
        """
        total_norm = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2).item()
                total_norm += param_norm ** 2
        return total_norm ** 0.5

    def _handle_oom_error(self):
        """
        Handle out of memory errors by clearing cache and reducing batch size if possible.
        """
        if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()
        self.logger.warning("OOM detected. Attempting to recover...")

        if self.config['training']['batch_size'] > 1:
            self.config['training']['batch_size'] //= 2
            self.logger.warning(f"Reduced batch size to {self.config['training']['batch_size']}")
            self._setup_dataloaders()

    def _setup_dataloaders(self):
        """
        Re-initialize dataloaders if batch size changes on OOM.
        """
        self.train_loader = DataLoader(
            dataset=self.train_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=True,
            num_workers=self.config['training'].get('num_workers', 1),
            pin_memory=self.config['training'].get('pin_memory', True),
            collate_fn=TextDataset.collate_fn,
            drop_last=self.config['training'].get('drop_last', False)
        )
        self.val_loader = DataLoader(
            dataset=self.val_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=False,
            num_workers=self.config['training'].get('num_workers', 1),
            pin_memory=self.config['training'].get('pin_memory', True),
            collate_fn=TextDataset.collate_fn,
            drop_last=self.config['training'].get('drop_last', False)
        )

    def train_one_epoch(self, epoch: int) -> Tuple[float, float]:
        """
        Training loop with single noise injection (only from the dataset),
        plus standard gradient accumulation and mixed precision.
        Ensures unscale_() is called exactly once per update to avoid
        'unscale_() has already been called...' errors.
        """
        self.model.train()
        total_loss = 0.0
        step_in_epoch = 0
        accumulation_counter = 0
        accum_loss = 0.0

        # Re-init scaler if needed
        if not hasattr(self, 'scaler') or self.scaler is None:
            self.scaler = GradScaler(
                enabled=self.config['training'].get('use_mixed_precision', True),
                init_scale=2**10,
                growth_factor=1.5,
                backoff_factor=0.5,
                growth_interval=100
            )

        progress_bar = tqdm(self.train_loader, desc=f"Training Epoch {epoch + 1}")
        self.optimizer.zero_grad(set_to_none=True)

        for step, batch in enumerate(progress_bar):
            try:
                # -------------------------------
                # 1) Move batch to device
                # -------------------------------
                encoder_input_ids = batch["encoder_input_ids"].to(self.device).long()
                encoder_attention_mask = batch["encoder_attention_mask"].to(self.device)
                decoder_input_ids = batch["decoder_input_ids"].to(self.device).long()
                decoder_attention_mask = batch["decoder_attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device).long()

                # -------------------------------
                # 2) Forward pass (autocast)
                # -------------------------------
                with torch.amp.autocast(
                    device_type="cuda" if torch.cuda.is_available() else "cpu",
                    enabled=self.config['training'].get('use_mixed_precision', True)
                ):
                    outputs = self.model(
                        src=encoder_input_ids,
                        tgt=decoder_input_ids,
                        attention_mask=encoder_attention_mask,
                        tgt_mask=decoder_attention_mask
                    )

                    # If outputs contain NaN/Inf, skip
                    if not torch.isfinite(outputs).all():
                        self.logger.warning(f"Step {step} - Non-finite values in model output, skipping batch")
                        continue

                    # Flatten for CE loss
                    flat_outputs = outputs.view(-1, outputs.size(-1))
                    flat_labels = labels.view(-1)

                    # Avoid log(0) issues by adding a tiny epsilon
                    eps = 1e-8
                    flat_outputs = flat_outputs + eps

                    loss = self.criterion(flat_outputs, flat_labels)

                    # If the loss is NaN/Inf, skip
                    if not torch.isfinite(loss):
                        self.logger.warning(f"Step {step} - Non-finite loss detected, skipping batch")
                        continue

                    # Scale loss if gradient accumulation
                    loss = loss / self.gradient_accumulation_steps

                # -------------------------------
                # 3) Backward (scaled)
                # -------------------------------
                self.scaler.scale(loss).backward()

                accumulation_counter += 1
                accum_loss += loss.item()

                # -------------------------------
                # 4) Update step if accumulated
                # -------------------------------
                if accumulation_counter == self.gradient_accumulation_steps:
                    # unscale_() => only once here
                    self.scaler.unscale_(self.optimizer)

                    # Check gradients after unscale_
                    valid_gradients = True
                    grad_norm = 0.0
                    max_grad_value = 0.0

                    for name, param in self.model.named_parameters():
                        if param.grad is not None:
                            grad_value = param.grad.abs().max().item()
                            max_grad_value = max(max_grad_value, grad_value)

                            if not torch.isfinite(param.grad).all():
                                self.logger.warning(f"NaN/Inf detected in gradients of {name}")
                                valid_gradients = False
                                break

                            grad_norm += param.grad.norm().item() ** 2

                    grad_norm = grad_norm ** 0.5

                    # If gradient is too large, also invalid
                    if max_grad_value > self.config['training'].get('max_grad_value', 100.0):
                        self.logger.warning(f"Step {step} - Gradient value too large: {max_grad_value}")
                        valid_gradients = False

                    if valid_gradients:
                        # If everything is okay, we do gradient clipping + optimizer step
                        clip_grad_norm_(
                            self.model.parameters(),
                            self.config['training'].get('max_grad_norm', 1.0)
                        )

                        self.scaler.step(self.optimizer)
                        self.scaler.update()

                        if self.scheduler is not None:
                            self.scheduler.step()

                        total_loss += accum_loss
                        step_in_epoch += 1
                    else:
                        # If gradients are invalid, skip the step
                        self.logger.warning(f"Step {step} - Invalid gradients, skipping optimizer step.")
                        self.scaler.update()  # still update the scaler so it can adjust scale

                    # Reset for next accumulation
                    self.optimizer.zero_grad(set_to_none=True)
                    accumulation_counter = 0
                    accum_loss = 0.0

                # -------------------------------
                # 5) Progress bar
                # -------------------------------
                current_lr = (
                    self.scheduler.get_last_lr()[0]
                    if self.scheduler else self.config['training']['learning_rate']
                )
                progress_bar.set_postfix({
                    'loss': loss.item(),
                    'lr': f'{current_lr:.2e}',
                    'grad_norm': f'{grad_norm:.2e}' if 'grad_norm' in locals() else 'N/A'
                })

            except RuntimeError as e:
                if "CUDA out of memory" in str(e):
                    self._handle_oom_error()
                    continue
                raise e

            # Optional W&B logging
            if self.use_wandb:
                wandb.log({
                    "train/loss": loss.item(),
                    "train/learning_rate": self.scheduler.get_last_lr()[0]
                })

        # Summarize
        avg_loss = total_loss / step_in_epoch if step_in_epoch > 0 else float('inf')
        return avg_loss, self._calculate_perplexity(avg_loss)

    @torch.no_grad()
    def validate(self, epoch: int) -> Tuple[float, float]:
        """
        Validation loop with a single round of noise (or no noise) from the dataset.
        Typically, you'd keep validation data clean. But if your dataset
        is already noising in __getitem__(), that applies here too.
        """
        self.model.eval()
        total_loss = 0.0
        valid_steps = 0
        progress_bar = tqdm(self.val_loader, desc=f"LuminaLM Validating Epoch {epoch + 1}")

        for step, batch in enumerate(progress_bar):
            try:
                encoder_input_ids = batch.get("encoder_input_ids", None)
                encoder_attention_mask = batch.get("encoder_attention_mask", None)
                decoder_input_ids = batch.get("decoder_input_ids", None)
                decoder_attention_mask = batch.get("decoder_attention_mask", None)
                labels = batch.get("labels", None)

                if not all(t is not None for t in [encoder_input_ids, encoder_attention_mask, decoder_input_ids, decoder_attention_mask, labels]):
                    self.logger.error(f"Step {step} - Missing keys in batch. Skipping...")
                    continue

                encoder_input_ids = encoder_input_ids.to(self.device).long()
                encoder_attention_mask = encoder_attention_mask.to(self.device)
                decoder_input_ids = decoder_input_ids.to(self.device).long()
                decoder_attention_mask = decoder_attention_mask.to(self.device)
                labels = labels.to(self.device).long()

                with torch.amp.autocast(
                    device_type="cuda" if torch.cuda.is_available() else "cpu",
                    enabled=self.config['training'].get('use_mixed_precision', True)
                ):
                    outputs = self.model(
                        src=encoder_input_ids,
                        tgt=decoder_input_ids,
                        attention_mask=encoder_attention_mask,
                        tgt_mask=decoder_attention_mask
                    )

                if torch.isnan(outputs).any() or torch.isinf(outputs).any():
                    self.logger.error(f"Step {step} - Model output contains NaN/Inf! Skipping...")
                    continue

                flat_outputs = outputs.view(-1, outputs.size(-1))
                flat_labels = labels.view(-1)
                loss = self.criterion(flat_outputs, flat_labels)
                if torch.isnan(loss) or torch.isinf(loss):
                    self.logger.error(f"Step {step} - NaN/Inf detected in loss! Skipping batch.")
                    continue

                total_loss += loss.item()
                valid_steps += 1
                self.logger.info(f"Step {step} - Validation Loss: {loss.item():.4f}")

                if step % self.config['memory_monitor'].get('log_frequency', 100) == 0:
                    if self.memory_monitor:
                        self.memory_monitor.log_memory()

            except Exception as e:
                self.logger.error(f"Validation Error at Step {step}: {e}")

        avg_loss = total_loss / valid_steps if valid_steps > 0 else float('inf')
        val_perplexity = self._calculate_perplexity(avg_loss)

        self.logger.info(
            f"LuminaLM Epoch {epoch + 1} Validation Loss: {avg_loss:.4f}, "
            f"Perplexity: {val_perplexity:.2f}"
        )

        if self.use_wandb:
            wandb.log({
                "model": "LuminaLM",
                "val_loss": avg_loss,
                "val_perplexity": val_perplexity
            })

        return avg_loss, val_perplexity

    def train(self) -> str:
        """
        Main training entry.
        """
        try:
            best_val_loss = float('inf')
            best_model_path = os.path.join(
                self.config['training']['checkpoint_dir'],
                'best_LuminaLM_model.pt'
            )

            for epoch in range(self.start_epoch, self.config['training']['epochs']):
                self.logger.info(f"\n{'='*50}")
                self.logger.info(f"Starting LuminaLM Epoch {epoch + 1}/{self.config['training']['epochs']}")

                if torch.cuda.is_available():
                    self.logger.info(f"GPU Memory before epoch: {torch.cuda.memory_allocated()/1024**2:.2f}MB")

                train_loss, train_ppl = self.train_one_epoch(epoch)
                self.logger.info(f"Epoch {epoch + 1} Training - Loss: {train_loss:.4f}, Perplexity: {train_ppl:.2f}")

                val_loss, val_ppl = self.validate(epoch)
                self.logger.info(f"Epoch {epoch + 1} Validation - Loss: {val_loss:.4f}, Perplexity: {val_ppl:.2f}")

                self.logger.info(f"Current val_loss: {val_loss:.4f}, Best val_loss: {best_val_loss:.4f}")
                self.logger.info(f"Early stopping counter: {self.stopping_counter}/{self.early_stopping_patience}")

                if self._early_stopping(val_loss):
                    self.logger.warning(f"Early stopping triggered at epoch {epoch + 1}")
                    self.logger.warning(f"Final early stopping counter: {self.stopping_counter}")
                    self.logger.warning(f"Best validation loss achieved: {self.best_metric:.4f}")
                    break

                # Save best model
                if val_loss < best_val_loss:
                    self.logger.info(f"Validation loss improved from {best_val_loss:.4f} to {val_loss:.4f}")
                    best_val_loss = val_loss
                    torch.save({
                        'model_name': 'LuminaLM',
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'scheduler_state_dict': self.scheduler.state_dict(),
                        'loss': best_val_loss,
                        'stopping_counter': self.stopping_counter,
                        'best_metric': self.best_metric
                    }, best_model_path)
                    self.logger.info(f"Saved best model checkpoint at epoch {epoch + 1}")
                else:
                    self.logger.info(f"Validation loss did not improve from {best_val_loss:.4f}")

                # Save latest checkpoint each epoch
                latest_checkpoint_path = os.path.join(
                    self.config['training']['checkpoint_dir'],
                    'latest_LuminaLM_model_checkpoint.pt'
                )
                torch.save({
                    'model_name': 'LuminaLM',
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'loss': val_loss,
                    'stopping_counter': self.stopping_counter,
                    'best_metric': self.best_metric
                }, latest_checkpoint_path)

                self.logger.info(f"Saved latest model checkpoint after epoch {epoch + 1}")

                current_lr = self.scheduler.get_last_lr()[0]
                self.logger.info(f"Current learning rate: {current_lr:.2e}")
                self.logger.info(f"{'='*50}\n")

            self.logger.info("LuminaLM training completed.")
            return best_model_path

        except Exception as e:
            self.logger.error(f"Training failed with error: {str(e)}")
            self.logger.error(f"Error traceback: {traceback.format_exc()}")
            raise


# ----------------------------------------------------------------------
# Main function
# ----------------------------------------------------------------------
def main():
    """
    Main entry point for training. Loads config, sets up trainer, and starts training.
    """
    # Paths to the configuration files
    training_config_path = 'config/training_config.yaml'
    data_config_path = 'config/training_data_config.yaml'

    try:
        with open(training_config_path, 'r') as f:
            training_config = yaml.safe_load(f)

        with open(data_config_path, 'r') as f:
            data_config = yaml.safe_load(f)

        model_dir = 'model'
        os.makedirs(model_dir, exist_ok=True)

        # Update training configuration to set model saving directory
        training_config['training']['checkpoint_dir'] = model_dir

        # Initialize trainer
        trainer = Trainer(training_config=training_config, data_config=data_config)

        # Start training
        best_model_path = trainer.train()
        print(f"Training completed. Best model saved at: {best_model_path}")

    except Exception as e:
        logging.error(f"Training failed: {e}")
        raise


if __name__ == "__main__":
    main()
