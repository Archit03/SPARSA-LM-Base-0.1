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
            # Assign configurations
            self.config = training_config
            self.data_config = data_config

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
            checkpointing_params = self.config['model'].get('checkpointing', {})
            # Model configuration and initialization
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
                use_reentrant = checkpointing_params.get('use_reentrant', True)
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
        required_keys = ['logging', 'training', 'tokenizer', 'datasets', 'model']
        for key in required_keys:
           if key not in config:
                raise ConfigurationError(f"Missing required config key: {key}")

    # Convert learning_rate to float if it's a string
        if isinstance(config['training']['learning_rate'], str):
            try:
                config['training']['learning_rate'] = float(config['training']['learning_rate'])
            except ValueError:
                raise ConfigurationError("'learning_rate' must be a valid float value.")

        if not isinstance(config['training']['learning_rate'], (int, float)):
            raise ConfigurationError("'learning_rate' must be a float.")
        if not isinstance(config['training']['batch_size'], int) or config['training']['batch_size'] <= 0:
            raise ConfigurationError("'batch_size' must be a positive integer.")

    def _calculate_perplexity(self, loss):
        """Calculate perplexity from loss."""
        return torch.exp(torch.tensor(loss)).item()

    def _early_stopping(self, metric):
        """Implement early stopping mechanism."""
        if metric < self.best_metric:
            self.best_metric = metric
            self.stopping_counter = 0
        else:
            self.stopping_counter += 1

        return self.stopping_counter >= self.early_stopping_patience

    def _log_gradients_and_activations(self):
        """Log gradients and activations for debugging."""
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                wandb.log({f"gradient_norm/{name}": param.grad.norm().item()})
            wandb.log({f"parameter_norm/{name}": param.norm().item()})
            

    def train_one_epoch(self, epoch: int):
        self.model.train()
        total_loss = 0
        progress_bar = tqdm(self.train_loader, desc=f"LuminaLM Training Epoch {epoch + 1}")

        for step, batch in enumerate(progress_bar):
            inputs = batch["input_ids"].to(self.config['training']['device'])
            attention_mask = batch["attention_mask"].to(self.config['training']['device'])
            labels = batch["labels"].to(self.config['training']['device'])

            # Specify the device type for autocast
            device_type = 'cuda' if self.config['training']['device'] == 'cuda' else 'cpu'

        with autocast(device_type=device_type, enabled=self.config['training'].get('use_mixed_precision', True)):
            outputs = self.model(inputs, attention_mask=attention_mask)
            loss = self.criterion(outputs.view(-1, outputs.size(-1)), labels.view(-1))
            loss = loss / self.gradient_accumulation_steps

        total_loss += loss.item()
        progress_bar.set_postfix(loss=loss.item())

        # Backward pass
        self.scaler.scale(loss).backward()

        # Gradient accumulation step
        if (step + 1) % self.gradient_accumulation_steps == 0 or (step + 1) == len(self.train_loader):
            self.scaler.unscale_(self.optimizer)
            clip_grad_norm_(self.model.parameters(), self.config['training'].get('max_grad_norm', 1.0))
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()
            self.optimizer.zero_grad()

        # Log to W&B
        if self.use_wandb:
            self._log_gradients_and_activations()
            wandb.log({
                "model": "LuminaLM",
                "train_loss": loss.item(),
                "lr": self.scheduler.get_last_lr()[0]
            })

        avg_loss = total_loss / len(self.train_loader)
        train_perplexity = self._calculate_perplexity(avg_loss)
        self.logger.info(f"LuminaLM Epoch {epoch + 1} Training Loss: {avg_loss:.4f}, Perplexity: {train_perplexity:.2f}")
        return avg_loss, train_perplexity


    @torch.no_grad()
    def validate(self, epoch: int):
        self.model.eval()
        total_loss = 0
        progress_bar = tqdm(self.val_loader, desc=f"LuminaLM Validating Epoch {epoch + 1}")

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
        val_perplexity = self._calculate_perplexity(avg_loss)
        self.logger.info(f"LuminaLM Epoch {epoch + 1} Validation Loss: {avg_loss:.4f}, Perplexity: {val_perplexity:.2f}")

        # Log to W&B
        if self.use_wandb:
            wandb.log({
                "model": "LuminaLM",
                "val_loss": avg_loss,
                "val_perplexity": val_perplexity
            })

        return avg_loss, val_perplexity

    def train(self):
        best_val_loss = float('inf')
        best_model_path = os.path.join(
            self.config['training']['checkpoint_dir'], 
            'best_LuminaLM_model.pt'
        )

        for epoch in range(self.start_epoch, self.config['training']['epochs']):
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

            # Log to W&B
            if self.use_wandb:
                wandb.log({
                    "model": "LuminaLM",
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "train_perplexity": train_perplexity,
                    "val_loss": val_loss,
                    "val_perplexity": val_perplexity
                })

        self.logger.info("LuminaLM training completed.")
        return best_model_path
    
    
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
