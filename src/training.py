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
from dataset import DatasetProcessor, TextDataset
from model import Transformer, TransformerConfig
from utils import setup_logging, save_checkpoint, load_checkpoint, set_seed, MemoryMonitor
from torch.amp import GradScaler, autocast
import optuna

from sklearn.model_selection import train_test_split

class ConfigurationError(Exception):
    """Custom exception for configuration-related errors."""
    pass

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
            if self.tokenizer.pad_token is None:
                self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
                self.logger.info(f"Added `pad_token`: {self.tokenizer.pad_token}")

            # Load dataset configuration and initialize DatasetProcessor
            datasets = self.data_config.get('datasets', [])
            if not isinstance(datasets, list):
                raise ConfigurationError("'datasets' must be a list of dataset configurations.")
            source_dir = next(
                (d.get('config', {}).get('source_dir') for d in datasets if d.get('name') == self.config['dataset']['train_dataset']),
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

        except Exception as e:
            raise RuntimeError(f"LuminaLM Initialization failed: {e}")
        
    def _setup_distributed_training(self):
        """Enable distributed training if multiple GPUs available."""
        if torch.cuda.device_count() > 1:
            self.model = torch.nn.DataParallel(self.model)
            self.logger.info(f"Using {torch.cuda.device_count()} GPUs for LuminaLM")

    def _configure_optimizer(self):
        """Configure optimizer with weight decay."""
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
            {'params': decay_params, 'weight_decay': self.config['training']['weight_decay']},
            {'params': no_decay_params, 'weight_decay': 0.0}
        ]

        return torch.optim.AdamW(
            optim_groups,
            lr=self.config['training']['learning_rate'],
            betas=(0.9, 0.999),
            eps=1e-8
        )
    
    def _validate_config(self, config: Dict):
        """Comprehensive configuration validation."""
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

    def _calculate_perplexity(self, loss):
        """Calculate perplexity from loss."""
        return torch.exp(torch.tensor(loss)).item()

    def _early_stopping(self, val_metric: float) -> bool:
        """Enhanced early stopping with best model preservation."""
        improved = val_metric < self.best_metric
        if improved:
            self.best_metric = val_metric
            self.stopping_counter = 0
            self._save_best_model()
            return False
        
        self.stopping_counter += 1
        if self.stopping_counter >= self.early_stopping_patience:
            self.logger.info(f"Early stopping triggered after {self.stopping_counter} epochs without improvement")
            self._restore_best_model()
            return True
        
        return False

    def _save_best_model(self):
        """Save the best model state."""
        best_model_path = os.path.join(self.config['training']['checkpoint_dir'], 'best_model.pt')
        torch.save({
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_metric': self.best_metric,
            'config': self.config
        }, best_model_path)
        self.logger.info(f"Saved best model to {best_model_path}")

    def _restore_best_model(self):
        """Restore the best model state."""
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
        """Structured logging of training metrics."""
        if not self.use_wandb:
            return

        metrics = {
            "train/loss": loss,
            "train/learning_rate": self.scheduler.get_last_lr()[0],
            "train/epoch": epoch,
            "train/step": step,
        }

        # Log memory stats if available
        if torch.cuda.is_available():
            metrics.update({
                "system/gpu_memory_allocated": torch.cuda.memory_allocated(),
                "system/gpu_memory_reserved": torch.cuda.memory_reserved()
            })

        # Log gradients at specified intervals
        if step % self.config['logging'].get('gradient_logging_frequency', 1000) == 0:
            grad_norm = self._compute_gradient_norm()
            metrics["train/gradient_norm"] = grad_norm

        wandb.log(metrics)

    def _compute_gradient_norm(self) -> float:
        """Compute the total gradient norm."""
        total_norm = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        return total_norm ** 0.5

    def train_one_epoch(self, epoch: int) -> Tuple[float, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        total_samples = 0
        
        progress_bar = tqdm(self.train_loader, desc=f"LuminaLM Training Epoch {epoch}")
        
        for batch in progress_bar:
            try:
                # Move batch to device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Validate inputs
                if torch.any(labels >= self.model.config.vocab_size):
                    labels = torch.clamp(labels, 0, self.model.config.vocab_size - 1)
                
                # Forward pass
                outputs = self.model(
                    src=input_ids,
                    attention_mask=attention_mask
                )
                
                # Compute loss
                loss = self.criterion(outputs.view(-1, outputs.size(-1)), labels.view(-1))
                
                # Accumulate loss for logging
                total_loss += loss.item()
                total_samples += 1

                # Backward pass
                self.scaler.scale(loss).backward()

                # Update weights when gradient accumulation is complete
                if total_samples == self.gradient_accumulation_steps:
                    try:
                        # Unscale gradients
                        self.scaler.unscale_(self.optimizer)
                        
                        # Clip gradients
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), 
                            self.config['training'].get('max_grad_norm', 1.0)
                        )
                        
                        # Optimizer step
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        self.scheduler.step()
                        
                        # Zero gradients
                        self.optimizer.zero_grad(set_to_none=True)

                        # Update metrics
                        total_loss += loss.item()
                        
                        # Log metrics
                        if self.use_wandb:
                            self._log_training_metrics(
                                loss=loss.item(),
                                step=total_samples,
                                epoch=epoch
                            )

                        # Reset accumulation
                        total_loss = 0
                        total_samples = 0

                        # Update progress bar
                        progress_bar.set_postfix({
                            'loss': total_loss / total_samples,
                            'lr': self.scheduler.get_last_lr()[0]
                        })

                    except RuntimeError as e:
                        if "out of memory" in str(e):
                            self._handle_oom_error()
                            continue
                        raise e

            except Exception as e:
                self.logger.error(f"Error in training loop: {str(e)}")
                raise

        avg_loss = total_loss / total_samples
        return avg_loss, self._calculate_perplexity(avg_loss)

    def _get_device_type(self) -> str:
        """Determine the correct device type for autocast."""
        if torch.cuda.is_available() and 'cuda' in str(self.config['training']['device']):
            return 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return 'mps'
        return 'cpu'

    def _handle_oom_error(self):
        """Handle out of memory errors gracefully."""
        if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()
        self.logger.warning("OOM detected. Attempting to recover...")
        
        # Reduce batch size if possible
        if self.config['training']['batch_size'] > 1:
            self.config['training']['batch_size'] //= 2
            self.logger.warning(f"Reduced batch size to {self.config['training']['batch_size']}")
            
            # Recreate data loaders with new batch size
            self._setup_dataloaders()

    @torch.no_grad()
    def validate(self, epoch: int):
        self.model.eval()
        total_loss = 0
        progress_bar = tqdm(self.val_loader, desc=f"LuminaLM Validating Epoch {epoch + 1}")

        for batch in progress_bar:
            inputs = batch["input_ids"].to(self.config['training']['device'])
            attention_mask = batch["attention_mask"].to(self.config['training']['device'])
            labels = batch["labels"].to(self.config['training']['device'])

            outputs = self.model(inputs, attention_mask=attention_mask)
            loss = self.criterion(outputs.view(-1, outputs.size(-1)), labels.view(-1))
            total_loss += loss.item()

        avg_loss = total_loss / len(self.val_loader)
        val_perplexity = self._calculate_perplexity(avg_loss)
        self.logger.info(f"LuminaLM Epoch {epoch + 1} Validation Loss: {avg_loss:.4f}, Perplexity: {val_perplexity:.2f}")

        if self.use_wandb:
            wandb.log({
                "model": "LuminaLM",
                "val_loss": avg_loss,
                "val_perplexity": val_perplexity
            })

        return avg_loss, val_perplexity

    def train(self):
        try:
            best_val_loss = float('inf')
            best_model_path = os.path.join(
                self.config['training']['checkpoint_dir'], 
                'best_LuminaLM_model.pt'
            )

            for epoch in range(self.start_epoch, self.config['training']['epochs']):
                try:
                    self.logger.info(f"Starting LuminaLM Epoch {epoch + 1}/{self.config['training']['epochs']}")
                    train_loss, train_perplexity = self.train_one_epoch(epoch)
                    val_loss, val_perplexity = self.validate(epoch)

                    # Early stopping check
                    if self._early_stopping(val_loss):
                        self.logger.info(f"Early stopping triggered for LuminaLM at epoch {epoch}")
                        break

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
                        self.logger.info(f"Saved best LuminaLM model checkpoint at epoch {epoch + 1}")

                    if self.use_wandb:
                        wandb.log({
                            "model": "LuminaLM",
                            "epoch": epoch + 1,
                            "train_loss": train_loss,
                            "train_perplexity": train_perplexity,
                            "val_loss": val_loss,
                            "val_perplexity": val_perplexity
                        })

                except Exception as epoch_error:
                    self.logger.error(f"Error in epoch {epoch}: {epoch_error}")
                    break

            self.logger.info("LuminaLM training completed.")
            return best_model_path

        except Exception as e:
            self.logger.error(f"Training failed: {e}")
            raise
    
    
def main():
    """Main function to run the training process."""
    # Paths to the configuration files
    training_config_path = 'config/training_config.yaml'
    data_config_path = 'config/training_data_config.yaml'

    try:
        # Load training config
        with open(training_config_path, 'r') as f:
            training_config = yaml.safe_load(f)
            print(f"Loaded data_config: {data_config_path}") # for debugging
            print(f"Learning rate: {training_config['training']['learning_rate']}, Type: {type(training_config['training']['learning_rate'])}") # for debugging

        # Load data config
        with open(data_config_path, 'r') as f:
            data_config = yaml.safe_load(f)
            print(f"Loaded data_config: {data_config_path} successfully.") # for debugging
            print(f"data_config: {data_config}, Type: {type(data_config)}")

        # Ensure model directory exists
        model_dir = 'model'
        os.makedirs(model_dir, exist_ok=True)

        # Update training configuration to set model saving directory
        training_config['training']['checkpoint_dir'] = model_dir

        # Initialize trainer with the loaded configurations
        trainer = Trainer(training_config=training_config, data_config=data_config)

        # Start training
        best_model_path = trainer.train()
        print(f"Training completed. Best model saved at: {best_model_path}")

    except Exception as e:
        logging.error(f"Training failed: {e}")
        raise


if __name__ == "__main__":
    main()
