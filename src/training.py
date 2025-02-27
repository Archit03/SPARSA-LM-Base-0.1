import os
import yaml
import logging
import wandb
import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
from transformers import PreTrainedTokenizerFast, get_linear_schedule_with_warmup
from typing import Dict, Any, Optional, Tuple
import transformers
import traceback
from dataset import DatasetProcessor, TextDataset
from model import Transformer, TransformerConfig
from utils import setup_logging, save_checkpoint, load_checkpoint, set_seed, MemoryMonitor
from torch.amp import GradScaler
import optuna
from sklearn.model_selection import train_test_split
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

# ----------------------------------------------------------------------
# Debugging Utilities (unchanged)
# ----------------------------------------------------------------------
def debug_tensor_values(tensor: torch.Tensor, name: str, logger: logging.Logger) -> bool:
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

def validate_model_outputs(outputs: torch.Tensor, labels: torch.Tensor, vocab_size: int, logger: logging.Logger) -> bool:
    try:
        if outputs.dim() != labels.dim() + 1:
            logger.error(f"Dimension mismatch: outputs={outputs.shape}, labels={labels.shape}")
            return False
        if not debug_tensor_values(outputs, "outputs", logger):
            return False
        if not debug_tensor_values(labels, "labels", logger):
            return False
        flat_labels = labels.view(-1)
        invalid_labels = (flat_labels >= vocab_size) & (flat_labels != -100)
        if invalid_labels.any():
            invalid_indices = torch.where(invalid_labels)[0]
            invalid_values = flat_labels[invalid_labels]
            logger.error(f"Invalid label values (>= vocab_size): {invalid_values.cpu().tolist()}")
            logger.error(f"Positions with invalid values: {invalid_indices.cpu().tolist()}")
            logger.error(f"Vocabulary size: {vocab_size}")
            return False
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

def debug_loss_inputs(outputs: torch.Tensor, labels: torch.Tensor, vocab_size: int, logger: logging.Logger) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    try:
        flat_outputs = outputs.view(-1, outputs.size(-1))
        flat_labels = labels.view(-1)
        if flat_outputs.size(0) != flat_labels.size(0):
            logger.error(f"Size mismatch after flattening: outputs={flat_outputs.shape}, labels={flat_labels.shape}")
            return None, None
        if flat_outputs.size(1) != vocab_size:
            logger.error(f"Output size mismatch: got {flat_outputs.size(1)}, expected {vocab_size} (check model vocab_size or tokenizer vocab_size).")
            return None, None
        if not validate_model_outputs(flat_outputs, flat_labels, vocab_size, logger):
            return None, None
        return flat_outputs, flat_labels
    except Exception as e:
        logger.error(f"Error in debug_loss_inputs: {e}")
        return None, None

# ----------------------------------------------------------------------
# Additional Classes/Exceptions (unchanged)
# ----------------------------------------------------------------------
class ConfigurationError(Exception):
    """Custom exception for configuration-related errors."""
    pass

# ----------------------------------------------------------------------
# Trainer Class (updated)
# ----------------------------------------------------------------------
class Trainer:
    def __init__(self, training_config: Dict, data_config: Dict):
        try:
            self._validate_config(training_config)
            self.config = training_config
            self.data_config = data_config

            # Set deterministic algorithms and anomaly detection if enabled
            if self.config.get("extras", {}).get("deterministic_algorithms", False):
                torch.use_deterministic_algorithms(True)
                torch.backends.cudnn.deterministic = True
            if self.config.get("extras", {}).get("detect_anomaly", False):
                torch.autograd.set_detect_anomaly(True)

            # Early stopping setup
            self.best_metric = float('inf')
            self.stopping_counter = 0
            self.early_stopping_patience = self.config['training'].get('early_stopping_patience', 3)

            # Setup logging
            self.logger = setup_logging(self.config['logging']['log_dir'], name='LuminaLM_trainer')

            # Initialize W&B if enabled
            if self.config['logging'].get('use_wandb', True):
                wandb.init(project=self.config['logging'].get('wandb_project', 'LuminaLM-training'),
                           config=self.config)
                self.use_wandb = True
            else:
                self.use_wandb = False

            set_seed(self.config['training']['seed'])

            # --- Tokenizer Loading & Special Tokens Setup ---
            tokenizer_path = self.config['tokenizer']['path']
            if os.path.isfile(tokenizer_path):
                tokenizer_dir = os.path.dirname(tokenizer_path)
            else:
                tokenizer_dir = tokenizer_path

            self.tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_dir)
            self.logger.info(f"Tokenizer loaded with vocab size: {self.tokenizer.vocab_size}")

            expected_special_tokens = {
                "pad_token": "[PAD]",
                "unk_token": "[UNK]",
                "cls_token": "[CLS]",
                "sep_token": "[SEP]",
                "mask_token": "[MASK]",
                "bos_token": "[BOS]",
                "eos_token": "[EOS]"
            }

            special_mapping = {}
            for key, token in expected_special_tokens.items():
                if token in self.tokenizer.all_special_tokens:
                    token_id = self.tokenizer.convert_tokens_to_ids(token)
                else:
                    token_id = None
                special_mapping[key] = (token, token_id)
            self.logger.info(f"Special tokens mapping (derived from vocabulary): {special_mapping}")

            missing_tokens = {key: token for key, (token, token_id) in special_mapping.items() if token_id is None}
            if missing_tokens:
                self.logger.info(f"Missing special tokens detected: {missing_tokens}. Adding them...")
                self.tokenizer.add_special_tokens(missing_tokens)
                vocab = self.tokenizer.get_vocab()
                for key, token in expected_special_tokens.items():
                    token_id = vocab.get(token)
                    special_mapping[key] = (token, token_id)
                self.logger.info(f"Special tokens mapping (after adding missing tokens): {special_mapping}")
            else:
                self.logger.info("All special tokens are present with IDs.")

            self.tokenizer.pad_token = expected_special_tokens["pad_token"]
            self.tokenizer.unk_token = expected_special_tokens["unk_token"]
            self.tokenizer.cls_token = expected_special_tokens["cls_token"]
            self.tokenizer.sep_token = expected_special_tokens["sep_token"]
            self.tokenizer.mask_token = expected_special_tokens["mask_token"]
            self.tokenizer.bos_token = expected_special_tokens["bos_token"]
            self.tokenizer.eos_token = expected_special_tokens["eos_token"]

            self.tokenizer.save_pretrained(tokenizer_dir)
            # ------------------------------------------------------------------

            # Load dataset configuration
            datasets = self.data_config.get('datasets', [])
            if not isinstance(datasets, list):
                raise ValueError("'datasets' must be a list.")
            source_dir = next((d.get('config', {}).get('source_dir') for d in datasets 
                               if d.get('name') == self.config['dataset']['train_dataset']), None)
            if not source_dir or not os.path.exists(source_dir):
                raise ValueError(f"Invalid source directory: {source_dir}")

            split_config = self.config['dataset'].get('split', {'test_size': 0.2, 'random_state': 42})
            preprocessing_config = self.config['dataset'].get('preprocessing', {})

            self.dataset_processor = DatasetProcessor(source_dir=source_dir, split=split_config, preprocessing_config=preprocessing_config)

            max_length = self.config['dataset']['max_seq_len']
            self.train_dataset = self.dataset_processor.get_train_dataset(tokenizer=self.tokenizer, max_length=max_length)
            self.val_dataset = self.dataset_processor.get_val_dataset(tokenizer=self.tokenizer, max_length=max_length)

            self.device = torch.device(self.config['training'].get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))

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

            scheduler_type = self.config["training"].get("scheduler_type", "linear")
            lr_scheduler_kwargs = self.config["training"].get("lr_scheduler_kwargs", {})
            if scheduler_type in ["cosine_with_min_lr", "cosine_warmup"]:
                if 'min_lr' not in lr_scheduler_kwargs and 'min_lr_rate' not in lr_scheduler_kwargs:
                    self.logger.warning("`min_lr` not found in `lr_scheduler_kwargs`, using default 1e-6")
                    lr_scheduler_kwargs['min_lr'] = 1e-6

            self.enable_gradient_accumulation = self.config['training'].get('enable_gradient_accumulation', True)
            self.gradient_accumulation_steps = self.config['training'].get('gradient_accumulation_steps', 1) if self.enable_gradient_accumulation else 1

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
                normalize_before=self.config['model'].get('normalize_before', True),  # Use normalize_before from config
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
                use_reentrant=self.config['model'].get('use_reentrant', False),
                noise_type=self.config['training'].get('noise_type', 'mask'),
                noise_prob=self.config['training'].get('noise_prob', 0.3)
            )

            self.model = Transformer(model_config).to(self.device)

            def _init_weights(module):
                if isinstance(module, (torch.nn.Linear, torch.nn.Embedding)):
                    torch.nn.init.normal_(module.weight, mean=0.0, std=self.config['model'].get('initializer_range', 0.02))
                    if isinstance(module, torch.nn.Linear) and module.bias is not None:
                        torch.nn.init.zeros_(module.bias)
                elif isinstance(module, torch.nn.LayerNorm):
                    torch.nn.init.ones_(module.weight)
                    torch.nn.init.zeros_(module.bias)

            self.model.apply(_init_weights)

            self.optimizer = self._configure_optimizer()
            num_training_steps = len(self.train_loader) * self.config['training']['epochs']
            self.scheduler = self._configure_scheduler(self.optimizer, self.config['training'], num_training_steps)
            self.criterion = torch.nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id, label_smoothing=0.00)
            self.memory_monitor = MemoryMonitor()
            self.start_epoch = 0
            if self.config['training'].get('resume_from_checkpoint'):
                self.start_epoch = load_checkpoint(
                    self.config['training']['resume_from_checkpoint'],
                    self.model,
                    self.optimizer,
                    self.scheduler
                )

            self.scaler = GradScaler(enabled=self.config['training'].get('use_mixed_precision', True))

            # --- Extras: Set up empty CUDA cache and anomaly detection ---
            self.empty_cuda_cache_freq = self.config.get("extras", {}).get("empty_cuda_cache_freq", 10)
            # -------------------------------------------------------------------

            # --- Optional: Test tokenizer with a simple encode/decode ---
            test_sentence = "Hello, how are you?"
            encoded_ids = self.tokenizer.encode(test_sentence)
            decoded_sentence = self.tokenizer.decode(encoded_ids)
            self.logger.info(f"Test sentence: {test_sentence}")
            self.logger.info(f"Encoded IDs: {encoded_ids}")
            self.logger.info(f"Decoded sentence: {decoded_sentence}")
            # -------------------------------------------------------------------

        except Exception as e:
            raise RuntimeError(f"LuminaLM Initialization failed: {e}")

    def _setup_distributed_training(self):
        if torch.cuda.device_count() > 1:
            self.model = torch.nn.DataParallel(self.model)
            self.logger.info(f"Using {torch.cuda.device_count()} GPUs for LuminaLM")

    def _configure_scheduler(self, optimizer, config, num_training_steps):
        num_warmup_steps = int(num_training_steps * config.get('warmup_ratio', 0.1))
        scheduler_type = config.get("scheduler_type", "linear")
        lr_scheduler_kwargs = config.get("lr_scheduler_kwargs", {})

        if scheduler_type in ["cosine_warmup", "cosine_with_min_lr"]:
            default_min_lr = 0.0 if scheduler_type == "cosine_warmup" else 1e-6
            min_lr = float(lr_scheduler_kwargs.get("min_lr", default_min_lr))
            def lr_lambda(step):
                if step < num_warmup_steps:
                    return float(step) / float(max(1, num_warmup_steps))
                else:
                    progress = float(step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
                    return min_lr + (self.config['training']['learning_rate'] - min_lr) * 0.5 * (1.0 + np.cos(np.pi * progress))
            return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        elif scheduler_type == "linear_warmup":
            return transformers.get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps
            )
        else:
            raise ValueError(f"Unknown scheduler type: {scheduler_type}")

    def _configure_optimizer(self):
        decay_params = []
        no_decay_params = []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if param.abs().max() > 1.0:
                with torch.no_grad():
                    param.data.div_(param.abs().max())
            if any(nd in name for nd in ['bias', 'LayerNorm.weight']):
                no_decay_params.append(param)
            else:
                decay_params.append(param)
        optimizer_grouped_parameters = [
            {'params': decay_params, 'weight_decay': self.config['training']['weight_decay']},
            {'params': no_decay_params, 'weight_decay': 0.0}
        ]
        # Use parameters from the optimizer block if provided
        optimizer_params = self.config.get("optimizer", {})
        eps = float(optimizer_params.get("eps", 1e-8))
        betas = tuple(optimizer_params.get("betas", [0.9, 0.999]))
        fused = optimizer_params.get("fused", False)
        # Note: gradient centralization is not implemented here; add custom logic if needed.
        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=self.config['training']['learning_rate'],
            betas=betas,
            eps=eps,
            weight_decay=self.config['training']['weight_decay'],
            amsgrad=True
        )
        return optimizer

    def _validate_config(self, config: Dict):
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
            try:
                config['training']['learning_rate'] = float(config['training']['learning_rate'])
            except (ValueError, TypeError):
                raise ConfigurationError("Invalid learning rate value")
            if 'cuda' in str(config['training']['device']) and not torch.cuda.is_available():
                raise ConfigurationError("CUDA device specified but not available")
        except Exception as e:
            raise ConfigurationError(f"Configuration validation failed: {str(e)}")

    def _calculate_perplexity(self, loss: float) -> float:
        return float(torch.exp(torch.tensor(loss)))

    def _early_stopping(self, val_metric: float) -> bool:
        improved = val_metric < self.best_metric
        if improved:
            self.best_metric = val_metric
            self.stopping_counter = 0
            self._save_best_model()
            return False
        else:
            self.stopping_counter += 1
        if self.stopping_counter >= self.early_stopping_patience:
            self.logger.info(f"Early stopping triggered after {self.stopping_counter} epochs without improvement")
            self._restore_best_model()
            return True
        return False

    def _save_best_model(self):
        best_model_path = os.path.join(self.config['training']['checkpoint_dir'], 'best_LuminaLM_model.pt')
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
        best_model_path = os.path.join(self.config['training']['checkpoint_dir'], 'best_LuminaLM_model.pt')
        if os.path.exists(best_model_path):
            checkpoint = torch.load(best_model_path, map_location=str(self.device), weights_only=True)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            self.logger.info("Restored best model state")
        else:
            self.logger.warning("No best model checkpoint found to restore")

    def _log_training_metrics(self, loss: float, step: int, epoch: int):
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
        total_norm = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2).item()
                total_norm += param_norm ** 2
        return total_norm ** 0.5

    def _handle_oom_error(self):
        if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()
        self.logger.warning("OOM detected. Attempting to recover...")
        if self.config['training']['batch_size'] > 1:
            self.config['training']['batch_size'] //= 2
            self.logger.warning(f"Reduced batch size to {self.config['training']['batch_size']}")
            self._setup_dataloaders()

    def _setup_dataloaders(self):
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
        self.model.train()
        total_loss = 0.0
        step_in_epoch = 0
        accumulation_counter = 0
        accum_loss = 0.0

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
                encoder_input_ids = batch["encoder_input_ids"].to(self.device).long()
                encoder_attention_mask = batch["encoder_attention_mask"].to(self.device)
                decoder_input_ids = batch["decoder_input_ids"].to(self.device).long()
                decoder_attention_mask = batch["decoder_attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device).long()

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
                    if not torch.isfinite(outputs).all():
                        self.logger.warning(f"Step {step} - Non-finite values in model output, skipping batch")
                        continue

                    flat_outputs = outputs.view(-1, outputs.size(-1))
                    flat_labels = labels.view(-1)
                    eps = 1e-8
                    flat_outputs = flat_outputs + eps
                    loss = self.criterion(flat_outputs, flat_labels)
                    if not torch.isfinite(loss):
                        self.logger.warning(f"Step {step} - Non-finite loss detected, skipping batch")
                        continue
                    loss = loss / self.gradient_accumulation_steps

                self.scaler.scale(loss).backward()
                accumulation_counter += 1
                accum_loss += loss.item()

                if accumulation_counter == self.gradient_accumulation_steps:
                    self.scaler.unscale_(self.optimizer)
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
                    if max_grad_value > self.config['training'].get('max_grad_value', 100.0):
                        self.logger.warning(f"Step {step} - Gradient value too large: {max_grad_value}")
                        valid_gradients = False

                    if valid_gradients:
                        clip_grad_norm_(self.model.parameters(), self.config['training'].get('max_grad_norm', 1.0))
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        if self.scheduler is not None:
                            self.scheduler.step()
                        total_loss += accum_loss
                        step_in_epoch += 1
                    else:
                        self.logger.warning(f"Step {step} - Invalid gradients, skipping optimizer step.")
                        self.scaler.update()
                    self.optimizer.zero_grad(set_to_none=True)
                    accumulation_counter = 0
                    accum_loss = 0.0

                current_lr = self.scheduler.get_last_lr()[0] if self.scheduler else self.config['training']['learning_rate']
                progress_bar.set_postfix({'loss': loss.item(), 'lr': f'{current_lr:.2e}', 'grad_norm': f'{grad_norm:.2e}' if 'grad_norm' in locals() else 'N/A'})

                # Periodically empty CUDA cache if specified in extras
                if step % self.config.get("extras", {}).get("empty_cuda_cache_freq", 10) == 0:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

            except RuntimeError as e:
                if "CUDA out of memory" in str(e):
                    self._handle_oom_error()
                    continue
                raise e

            if self.use_wandb:
                wandb.log({"train/loss": loss.item(), "train/learning_rate": self.scheduler.get_last_lr()[0]})

        avg_loss = total_loss / step_in_epoch if step_in_epoch > 0 else float('inf')
        return avg_loss, self._calculate_perplexity(avg_loss)

    @torch.no_grad()
    def validate(self, epoch: int) -> Tuple[float, float]:
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
        self.logger.info(f"LuminaLM Epoch {epoch + 1} Validation Loss: {avg_loss:.4f}, Perplexity: {val_perplexity:.2f}")
        if self.use_wandb:
            wandb.log({"model": "LuminaLM", "val_loss": avg_loss, "val_perplexity": val_perplexity})
        return avg_loss, val_perplexity

    def train(self) -> str:
        try:
            best_val_loss = float('inf')
            best_model_path = os.path.join(self.config['training']['checkpoint_dir'], 'best_LuminaLM_model.pt')
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
                latest_checkpoint_path = os.path.join(self.config['training']['checkpoint_dir'], 'latest_LuminaLM_model_checkpoint.pt')
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
    training_config_path = 'config/training_config.yaml'
    data_config_path = 'config/training_data_config.yaml'
    try:
        with open(training_config_path, 'r') as f:
            training_config = yaml.safe_load(f)
        with open(data_config_path, 'r') as f:
            data_config = yaml.safe_load(f)
        model_dir = 'model'
        os.makedirs(model_dir, exist_ok=True)
        training_config['training']['checkpoint_dir'] = model_dir
        trainer = Trainer(training_config=training_config, data_config=data_config)
        best_model_path = trainer.train()
        print(f"Training completed. Best model saved at: {best_model_path}")
    except Exception as e:
        logging.error(f"Training failed: {e}")
        raise

if __name__ == "__main__":
    main()
