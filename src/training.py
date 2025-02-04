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
from typing import Dict, Any, Optional, Tuple

# ----------------------------------------------------------------------
# Import your dataset/module references:
# ----------------------------------------------------------------------
from dataset import DatasetProcessor, TextDataset
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
        # 1. Dimension check: outputs should be [batch_size, seq_len, vocab_size]
        #    labels should be [batch_size, seq_len] if not flattened yet.
        #    Alternatively, if using flattened shapes, you must adjust accordingly.
        
        # If you're flattening them after you get them from the model,
        # check the dimension difference (outputs dim = labels dim + 1).
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
        # Typically, -100 is used for ignore_index in CrossEntropyLoss
        # So the only valid range is: [0, vocab_size - 1] or exactly -100
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
        # Flatten the outputs to [batch_size*seq_len, vocab_size]
        # Flatten labels to [batch_size*seq_len]
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

            # Early stopping initialization
            self.best_metric = float('inf')
            self.stopping_counter = 0
            self.early_stopping_patience = self.config['training'].get('early_stopping_patience', 3)

            # Setup logging
            self.logger = setup_logging(self.config['logging']['log_dir'], name='LuminaLM_trainer')

            # Initialize Weights & Biases (W&B)
            if self.config['logging'].get('use_wandb', True):
                wandb.init(
                    project=self.config['logging'].get('wandb_project', 'LuminaLM-training'),
                    config=self.config
                )
                self.use_wandb = True
            else:
                self.use_wandb = False

            # Set random seeds for reproducibility
            set_seed(self.config['training']['seed'])

            # Load tokenizer
            tokenizer_path = self.config['tokenizer']['path']
            self.tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)
            self.logger.info(f"Tokenizer loaded with vocab size: {self.tokenizer.vocab_size}")
            if self.tokenizer.pad_token is None or self.tokenizer.pad_token_id is None:
                self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            else:
                self.logger.info(f"PAD token already exists: {self.tokenizer.pad_token} (ID: {self.tokenizer.pad_token_id})")

            # Load dataset configuration and initialize DatasetProcessor
            datasets = self.data_config.get('datasets', [])
            if not isinstance(datasets, list):
                raise ConfigurationError("'datasets' must be a list of dataset configurations.")
            
            source_dir = next(
                (d.get('config', {}).get('source_dir') for d in datasets
                 if d.get('name') == self.config['dataset']['train_dataset']),
                None
            )
            if not source_dir or not os.path.exists(source_dir):
                raise ConfigurationError(f"Invalid source directory: {source_dir}")

            split_config = self.config['dataset'].get('split', {'test_size': 0.2, 'random_state': 42})
            preprocessing_config = self.config['dataset'].get('preprocessing', {})

            self.dataset_processor = DatasetProcessor(
                source_dir=source_dir,
                split=split_config,
                preprocessing_config=preprocessing_config
            )

            # Prepare train and validation datasets
            max_length = self.config['dataset']['max_seq_len']
            self.train_dataset = self.dataset_processor.get_train_dataset(
                tokenizer=self.tokenizer,
                max_length=max_length
            )
            self.val_dataset = self.dataset_processor.get_val_dataset(
                tokenizer=self.tokenizer,
                max_length=max_length
            )

            # Initialize data loaders
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

            # Model configuration and initialization
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
                scheduler_type=self.config['training']['scheduler_type'],
                learning_rate=self.config['training']['learning_rate'],
                weight_decay=self.config['training']['weight_decay'],
                warmup_ratio=self.config['training'].get('warmup_ratio', 0.1),
                use_mixed_precision=self.config['training'].get('use_mixed_precision', True),
                max_grad_norm=self.config['training'].get('max_grad_norm', 1.0),
                pad_token_id=self.tokenizer.pad_token_id,
                l2_reg=self.config['training'].get('l2_reg', 0.0),
                use_checkpointing=self.config['model'].get('use_checkpointing', False),
                use_reentrant=checkpointing_params.get('use_reentrant', False)
            )
            self.model = Transformer(model_config).to(self.config['training']['device'])

            # Optimizer, scheduler, and loss
            self.optimizer = self._configure_optimizer()
            num_training_steps = len(self.train_loader) * self.config['training']['epochs']
            self.scheduler = get_scheduler(
                name=self.config['training']['scheduler_type'],
                optimizer=self.optimizer,
                num_warmup_steps=int(num_training_steps * self.config['training'].get('warmup_ratio')),
                num_training_steps=num_training_steps
            )
            self.criterion = torch.nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)

            # Memory monitor
            self.memory_monitor = MemoryMonitor()

            # Checkpoints
            self.start_epoch = 0
            if self.config['training']['resume_from_checkpoint']:
                self.start_epoch = load_checkpoint(
                    self.config['training']['resume_from_checkpoint'],
                    self.model,
                    self.optimizer,
                    self.scheduler
                )

            # Gradient accumulation and mixed precision
            self.gradient_accumulation_steps = self.config['training']['gradient_accumulation_steps']
            self.scaler = GradScaler(enabled=self.config['training'].get('use_mixed_precision', True))

            # Handy short references
            self.device = self.config['training']['device']
            
        except Exception as e:
            raise RuntimeError(f"LuminaLM Initialization failed: {e}")
        
    def _setup_distributed_training(self):
        """
        Enable distributed training if multiple GPUs are available.
        """
        if torch.cuda.device_count() > 1:
            self.model = torch.nn.DataParallel(self.model)
            self.logger.info(f"Using {torch.cuda.device_count()} GPUs for LuminaLM")

    def _configure_optimizer(self) -> torch.optim.Optimizer:
        """
        Configure optimizer with weight decay for certain parameter groups.
        """
        decay_params = []
        no_decay_params = []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if 'bias' in name or 'norm' in name or 'embedding' in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        optim_groups = [
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
            optim_groups,
            lr=self.config['training']['learning_rate'],
            betas=(0.9, 0.999),
            eps=1e-8
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
        # Because exp(loss) can overflow, we wrap it in float().
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
            checkpoint = torch.load(best_model_path, map_location=self.device)
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

        # Log memory stats if on GPU
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

        # Potentially reduce batch size or other strategies
        if self.config['training']['batch_size'] > 1:
            self.config['training']['batch_size'] //= 2
            self.logger.warning(f"Reduced batch size to {self.config['training']['batch_size']}")
            # Recreate data loaders with new batch size
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
        Train the model for one epoch with gradient accumulation and debugging checks.

        Args: 
            epoch (int): Current epoch number.
        Returns:
            Tuple[float, float]: Average training loss and perplexity for each epoch.
        """
        self.model.train()
        total_loss = 0.0
        step_in_epoch = 0
        accumulation_counter = 0
        accum_loss = 0.0

        progress_bar = tqdm(self.train_loader, desc=f"LuminaLM Training Epoch {epoch + 1}")

        for step, batch in enumerate(progress_bar):
            try:
                if "encoder_input_ids" in batch:
                    encoder_input_ids = batch["encoder_input_ids"].to(self.device)
                    encoder_attention_mask = batch["encoder_attention_mask"].to(self.device)
                    decoder_input_ids = batch["decoder_input_ids"].to(self.device)
                    decoder_attention_mask = batch["decoder_attention_mask"].to(self.device)
                    labels = batch["labels"].to(self.device)
                else:
                    self.logger.info(f"\nStep {step} - Tensor states")
                    debug_tensor_values(encoder_input_ids, "encoder_input_ids", self.logger)
                    debug_tensor_values(encoder_attention_mask, "encoder_attention_mask", self.logger)
                    debug_tensor_values(labels, "labels", self.logger)

                    if decoder_input_ids is not None:
                        outputs = self.model(
                            input_ids=encoder_input_ids,
                            attention_mask=encoder_attention_mask,
                            decoder_input_ids=decoder_input_ids,
                            decoder_attention_mask=decoder_attention_mask,
                        )
                    else:
                        outputs = self.model(
                            input_ids=encoder_input_ids,
                            attention_mask=encoder_attention_mask,
                        )
                    
                    flat_outputs, flat_labels = debug_loss_inputs(
                        outputs,
                        labels,
                        self.model.config.vocab_size,
                        self.logger
                    )

                    if flat_outputs is None or flat_labels is None:
                        raise ValueError("Invalid outputs/labels for loss calculation (debug checks failed).")
                    
                    debug_tensor_values(flat_outputs, "flat_outputs", self.logger)
                    debug_tensor_values(flat_labels, "flat_labels", self.logger)
 
                    # Ensure labels are within the correct range
                    if torch.any(flat_labels < -100) or torch.any(flat_labels >= self.model.config.vocab_size):
                        self.logger.error(f"Labels out of bounds! Min: {flat_labels.min().item()}, Max: {flat_labels.max().item()}")
                        raise ValueError("Detected invalid label values before loss calculation.")
                   
                    loss = self.criterion(flat_outputs, flat_labels)
                    if torch.isnan(loss) or torch.isinf(loss):
                        self.logger.error("NaN or Inf detected in loss! Skipping this step.")
                        continue
                    accum_loss += loss.item()
                    loss = loss / self.gradient_accumulation_steps # Gradient accumulation

                    # Mixed precision training, backpropagation, and gradient scaling
                    self.scaler.scale(loss).backward()
                    accumulation_counter += 1

                    if accumulation_counter ==  self.gradient_accumulation_steps:
                        #Unscale the gradients
                        self.scaler.unscale_(self.optimizer)

                        # Clip gradients for stability
                        clip_grad_norm_(
                            self.model.parameters(), 
                            self.config['training'].get('max_grad_norm', 1.0)
                            )
                        
                        #Optimizer step
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        self.scheduler.step()

                        # Zero gradients properly
                        self.optimizer.zero_grad(set_to_none=True)
                        total_loss += accum_loss
                        step_in_epoch += 1

                        # Log training metrics
                        current_loss = accum_loss / self.gradient_accumulation_steps
                        self._log_training_metrics(current_loss, step_in_epoch, epoch + 1)

                        #Reset accumulation counter and loss
                        accum_loss  = 0.0
                        accumulation_counter = 0

                        progress_bar.set_postfix({
                            "loss": current_loss,
                            "lr" : self.scheduler.get_last_lr()[0]
                        })

            except RuntimeError as re:
                if "CUDA out of memory" in str(re):
                    self._handle_oom_error()
                    continue
                else:
                    self.logger.error(f"Runtime error in training loop: {re}")
                    raise re
            except Exception as e:
                self.logger.error(f"Error in training loop: {e}")
                raise e
        avg_loss = total_loss / step_in_epoch if step_in_epoch > 0 else 0.0
        return avg_loss, self._calculate_perplexity(avg_loss)
    
    @torch.no_grad()
    def validate(self, epoch: int) -> Tuple[float, float]:
        """
        Validate (evaluate) the model on the validation set.
        """
        self.model.eval()
        total_loss = 0.0
        progress_bar = tqdm(self.val_loader, desc=f"LuminaLM Validating Epoch {epoch + 1}")

        for batch in progress_bar:
            try:
                inputs = batch.get("encoder_input_ids", None)
                attention_mask = batch.get("encoder_attention_mask", None)
                labels = batch.get("labels", None)  # FIX: Assign labels BEFORE checking invalid values

                if inputs is None or attention_mask is None or labels is None:
                    self.logger.error("Missing keys in batch. Skipping this batch...")
                    continue

                inputs = inputs.to(self.device)
                attention_mask = attention_mask.to(self.device)
                labels = labels.to(self.device)

                #FIX: Assign `labels` before using in `invalid_labels`
                invalid_labels = (labels < 0) | (labels >= self.model.config.vocab_size)
                if invalid_labels.any():
                    self.logger.error(f"Invalid label values detected! {labels[invalid_labels].tolist()}")
                    raise ValueError("Invalid labels found in validation step!")
                
                # Forward pass
                outputs = self.model(inputs, attention_mask=attention_mask)
                 
                # Flatten for cross-entropy loss
                flat_outputs = outputs.view(-1, outputs.size(-1))
                flat_labels = labels.view(-1)

                # Compute loss
                loss = self.criterion(flat_outputs, flat_labels)
                total_loss += loss.item()
            except Exception as e:
                self.logger.error(f"Validation: {e}")
        
        avg_loss = total_loss / len(self.val_loader) if len(self.val_loader) > 0 else float('inf')
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
        Main training loop which runs across all epochs:
          - train_one_epoch
          - validate
          - early stopping
          - checkpointing
        """
        try:
            best_val_loss = float('inf')
            best_model_path = os.path.join(
                self.config['training']['checkpoint_dir'],
                'best_LuminaLM_model.pt'
            )

            for epoch in range(self.start_epoch, self.config['training']['epochs']):
                self.logger.info(f"Starting LuminaLM Epoch {epoch + 1}/{self.config['training']['epochs']}")
                
                train_loss, train_ppl = self.train_one_epoch(epoch)
                val_loss, val_ppl = self.validate(epoch)

                # Early stopping check
                if self._early_stopping(val_loss):
                    self.logger.info(f"Early stopping triggered for LuminaLM at epoch {epoch}")
                    break

                # Save "best" checkpoint
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save({
                        'model_name': 'LuminaLM',
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'scheduler_state_dict': self.scheduler.state_dict(),
                        'loss': best_val_loss
                    }, best_model_path)

                    save_checkpoint(
                        self.config['training']['checkpoint_dir'],
                        self.model,
                        self.optimizer,
                        self.scheduler,
                        epoch,
                        model_name='LuminaLM'
                    )
                    self.logger.info(
                        f"Saved best LuminaLM model checkpoint at epoch {epoch + 1}"
                    )

                # Log metrics to W&B
                if self.use_wandb:
                    wandb.log({
                        "model": "LuminaLM",
                        "epoch": epoch + 1,
                        "train_loss": train_loss,
                        "train_perplexity": train_ppl,
                        "val_loss": val_loss,
                        "val_perplexity": val_ppl
                    })

            self.logger.info("LuminaLM training completed.")
            return best_model_path

        except Exception as e:
            self.logger.error(f"Training failed: {e}")
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
        # Load training config
        with open(training_config_path, 'r') as f:
            training_config = yaml.safe_load(f)

        # Load data config
        with open(data_config_path, 'r') as f:
            data_config = yaml.safe_load(f)

        # Ensure model directory exists
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

# ----------------------------------------------------------------------
# Entry Point
# ----------------------------------------------------------------------
if __name__ == "__main__":
    main()
