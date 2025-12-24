"""
SPARSA-LM RL Finetuning Trainer
Unified trainer for DAPO and VAPO RL-based finetuning
"""

import os
import time
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List, Callable, Union
import copy

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler

from .config import FinetuneConfig
from .dapo import DAPOTrainer
from .vapo import VAPOTrainer, ValueModel
from ..model import AutoRegressiveLM, ModelConfig
from ..data import RLDataset, RLCollator

logger = logging.getLogger(__name__)


class RLTrainer:
    """
    Unified RL Trainer for DAPO and VAPO finetuning.

    Provides a high-level interface for running RL-based finetuning
    with either DAPO or VAPO algorithms.

    Features:
    - Automatic algorithm selection based on config
    - Distributed training support
    - Reward model integration
    - Checkpoint management
    - Experiment tracking
    """

    def __init__(
        self,
        model: AutoRegressiveLM,
        config: FinetuneConfig,
        train_dataset: RLDataset,
        tokenizer: Any,
        reward_fn: Optional[Callable] = None,
        reward_model: Optional[nn.Module] = None,
    ):
        self.model = model
        self.config = config
        self.train_dataset = train_dataset
        self.tokenizer = tokenizer
        self.reward_fn = reward_fn
        self.reward_model = reward_model

        # Training state
        self.global_step = 0
        self.epoch = 0

        # Setup distributed
        self._setup_distributed()

        # Setup logging
        self._setup_logging()

        # Create output directories
        self._setup_directories()

        # Initialize RL trainer
        self._setup_rl_trainer()

    def _setup_distributed(self):
        """Initialize distributed training."""
        self.is_distributed = dist.is_initialized()
        self.local_rank = self.config.local_rank if hasattr(self.config, 'local_rank') else 0
        self.world_size = dist.get_world_size() if self.is_distributed else 1
        self.is_main_process = self.local_rank == 0

        if self.is_distributed:
            self.device = torch.device(f"cuda:{self.local_rank}")
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    def _setup_rl_trainer(self):
        """Initialize the appropriate RL trainer (DAPO or VAPO)."""
        # Create reference model (frozen copy of policy)
        if self.config.use_reference_model:
            self.reference_model = copy.deepcopy(self.model)
            self.reference_model.eval()
            for param in self.reference_model.parameters():
                param.requires_grad = False
        else:
            self.reference_model = None

        # Create value model for VAPO
        if self.config.vapo_enabled:
            self.value_model = ValueModel(
                base_model=copy.deepcopy(self.model),
                hidden_size=self.model.config.hidden_size,
            )
        else:
            self.value_model = None

        # Initialize trainer
        if self.config.dapo_enabled:
            self.rl_trainer = DAPOTrainer(
                policy_model=self.model,
                reference_model=self.reference_model,
                value_model=self.value_model,
                tokenizer=self.tokenizer,
                clip_range_upper=self.config.clip_range_upper,
                clip_range_lower=self.config.clip_range_lower,
                dynamic_sampling=self.config.dynamic_sampling,
                sampling_temperature_init=self.config.sampling_temperature_init,
                sampling_temperature_decay=self.config.sampling_temperature_decay,
                entropy_coef=self.config.entropy_coef,
                entropy_target=self.config.entropy_target,
                kl_coef=self.config.init_kl_coef,
                target_kl=self.config.target_kl,
                gamma=self.config.gamma,
                lam=self.config.lam,
                device=str(self.device),
            )
            logger.info("Initialized DAPO trainer")
        elif self.config.vapo_enabled:
            self.rl_trainer = VAPOTrainer(
                policy_model=self.model,
                value_model=self.value_model,
                reference_model=self.reference_model,
                tokenizer=self.tokenizer,
                clip_range=self.config.clip_range_upper,
                value_clip_range=self.config.value_clip_range,
                value_coef=self.config.value_model_coef,
                dense_reward=self.config.dense_reward,
                reward_smoothing=self.config.reward_smoothing,
                kl_coef=self.config.init_kl_coef,
                target_kl=self.config.target_kl,
                gamma=self.config.gamma,
                lam=self.config.lam,
                device=str(self.device),
            )
            logger.info("Initialized VAPO trainer")
        else:
            raise ValueError("Either dapo_enabled or vapo_enabled must be True")

    def train(self) -> Dict[str, float]:
        """
        Run RL finetuning.

        Returns:
            Dictionary with training metrics
        """
        # Create optimizer
        optimizer = self._create_optimizer()

        # Create data loader
        train_loader = self._create_dataloader()

        # Resume from checkpoint if specified
        if self.config.resume_from_checkpoint:
            self._load_checkpoint(self.config.resume_from_checkpoint, optimizer)

        # Training loop
        logger.info("Starting RL finetuning...")
        logger.info(f"  Algorithm: {'DAPO' if self.config.dapo_enabled else 'VAPO'}")
        logger.info(f"  Num epochs: {self.config.num_epochs}")
        logger.info(f"  Batch size: {self.config.per_device_batch_size}")

        total_metrics = {}
        start_time = time.time()

        for epoch in range(self.epoch, self.config.num_epochs):
            self.epoch = epoch
            epoch_metrics = self._train_epoch(train_loader, optimizer)

            # Log epoch metrics
            for key, value in epoch_metrics.items():
                if key not in total_metrics:
                    total_metrics[key] = []
                total_metrics[key].append(value)

            self._log_metrics({"epoch": epoch, **epoch_metrics})

            # Save checkpoint
            self._save_checkpoint(optimizer)

            # Update dynamic parameters
            if self.config.dapo_enabled:
                self.rl_trainer.update_sampling_temperature()

            # Check early stopping based on KL
            if self.config.adaptive_kl:
                avg_kl = epoch_metrics.get("approx_kl", 0)
                self.rl_trainer.update_kl_coef(avg_kl)

        # Final save
        self._save_checkpoint(optimizer, final=True)

        training_time = time.time() - start_time
        final_metrics = {
            "train/total_epochs": self.epoch + 1,
            "train/total_steps": self.global_step,
            "train/training_time_hours": training_time / 3600,
        }

        # Average metrics across epochs
        for key, values in total_metrics.items():
            final_metrics[f"train/avg_{key}"] = sum(values) / len(values)

        logger.info(f"Training completed in {training_time / 3600:.2f} hours")

        return final_metrics

    def _train_epoch(
        self,
        train_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
    ) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()

        epoch_stats = {
            "policy_loss": [],
            "value_loss": [],
            "approx_kl": [],
            "clip_fraction": [],
        }

        for batch_idx, batch in enumerate(train_loader):
            # Move batch to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

            # Rollout: generate responses
            with torch.no_grad():
                if self.config.vapo_enabled:
                    sequences, log_probs, ref_log_probs, values = self.rl_trainer.generate_responses(
                        prompts=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        max_new_tokens=self.config.max_new_tokens,
                        temperature=self.config.generation_temperature,
                        top_p=self.config.generation_top_p,
                        top_k=self.config.generation_top_k,
                    )
                else:
                    sequences, log_probs, ref_log_probs = self.rl_trainer.generate_responses(
                        prompts=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        max_new_tokens=self.config.max_new_tokens,
                        top_p=self.config.generation_top_p,
                        top_k=self.config.generation_top_k,
                    )
                    values = None

            # Compute rewards
            scores = self._compute_rewards(sequences, batch)

            # Compute advantages
            prompt_lengths = batch.get("prompt_lengths", batch["input_ids"].shape[1] * torch.ones(sequences.shape[0], device=self.device))

            if self.config.vapo_enabled:
                rewards = self.rl_trainer.compute_dense_rewards(
                    sequences=sequences,
                    final_scores=scores,
                    prompt_lengths=prompt_lengths.long(),
                    values=values,
                    log_probs=log_probs,
                    ref_log_probs=ref_log_probs,
                )
            else:
                rewards, _ = self.rl_trainer.compute_rewards(
                    sequences=sequences,
                    scores=scores,
                    prompt_lengths=prompt_lengths.long(),
                    log_probs=log_probs,
                    ref_log_probs=ref_log_probs,
                )

            # Get values for advantage computation
            if values is None:
                with torch.no_grad():
                    outputs = self.model(input_ids=sequences)
                    # Use logits as value proxy
                    values = outputs["logits"].mean(dim=-1)[:, batch["input_ids"].shape[1] - 1:-1]

            # Create attention mask for generated sequence
            gen_attention_mask = torch.ones(sequences.shape[0], rewards.shape[1], device=self.device)

            advantages, returns = self.rl_trainer.compute_advantages(
                rewards=rewards,
                values=values,
                masks=gen_attention_mask,
            )

            # PPO training epochs
            for ppo_epoch in range(self.config.ppo_epochs):
                # Shuffle minibatches
                indices = torch.randperm(sequences.shape[0])

                for start in range(0, sequences.shape[0], self.config.per_device_batch_size):
                    end = start + self.config.per_device_batch_size
                    mb_indices = indices[start:end]

                    # Get minibatch
                    mb_sequences = sequences[mb_indices]
                    mb_attention_mask = torch.ones_like(mb_sequences)
                    mb_old_log_probs = log_probs[mb_indices]
                    mb_advantages = advantages[mb_indices]
                    mb_returns = returns[mb_indices]
                    mb_prompt_lengths = prompt_lengths[mb_indices]

                    if self.config.vapo_enabled:
                        mb_old_values = values[mb_indices]
                        output = self.rl_trainer.train_step(
                            sequences=mb_sequences,
                            attention_mask=mb_attention_mask,
                            old_log_probs=mb_old_log_probs,
                            old_values=mb_old_values,
                            advantages=mb_advantages,
                            returns=mb_returns,
                            prompt_lengths=mb_prompt_lengths.long(),
                        )
                    else:
                        output = self.rl_trainer.train_step(
                            sequences=mb_sequences,
                            attention_mask=mb_attention_mask,
                            old_log_probs=mb_old_log_probs,
                            advantages=mb_advantages,
                            returns=mb_returns,
                            prompt_lengths=mb_prompt_lengths.long(),
                        )

                    # Backward pass
                    optimizer.zero_grad()
                    output.loss.backward()

                    # Gradient clipping
                    if self.config.max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            self.config.max_grad_norm
                        )

                    optimizer.step()

                    # Track stats
                    epoch_stats["policy_loss"].append(output.policy_loss.item())
                    epoch_stats["value_loss"].append(output.value_loss.item())
                    epoch_stats["approx_kl"].append(output.approx_kl)
                    epoch_stats["clip_fraction"].append(output.clip_fraction)

                    self.global_step += 1

                    # Logging
                    if self.global_step % self.config.logging_steps == 0:
                        self._log_step_metrics(output)

                    # Check KL early stopping
                    if output.approx_kl > self.config.target_kl * 2:
                        logger.warning(f"KL divergence too high ({output.approx_kl:.4f}), stopping epoch early")
                        break

        # Reset RL trainer stats
        self.rl_trainer.reset_stats()

        # Average epoch stats
        return {k: sum(v) / len(v) if v else 0 for k, v in epoch_stats.items()}

    def _compute_rewards(
        self,
        sequences: torch.Tensor,
        batch: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Compute rewards for generated sequences."""
        if self.reward_fn is not None:
            # Use custom reward function
            texts = self.tokenizer.batch_decode(sequences, skip_special_tokens=True)
            prompts = self.tokenizer.batch_decode(batch["input_ids"], skip_special_tokens=True)
            rewards = self.reward_fn(prompts, texts)
            return torch.tensor(rewards, device=self.device)

        if self.reward_model is not None:
            # Use reward model
            with torch.no_grad():
                outputs = self.reward_model(input_ids=sequences)
                rewards = outputs["rewards"] if "rewards" in outputs else outputs["logits"].mean(dim=-1)
            return rewards.squeeze(-1)

        # Default: return zeros (should be overridden)
        logger.warning("No reward function or model provided, using zero rewards")
        return torch.zeros(sequences.shape[0], device=self.device)

    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer for RL training."""
        # Collect parameters from all trainable models
        params = list(self.model.parameters())
        if self.value_model is not None:
            params.extend(self.value_model.parameters())

        optimizer = torch.optim.AdamW(
            params,
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            betas=(self.config.adam_beta1, self.config.adam_beta2),
            eps=self.config.adam_epsilon,
        )

        return optimizer

    def _create_dataloader(self) -> DataLoader:
        """Create data loader for training."""
        collator = RLCollator(
            pad_token_id=self.tokenizer.pad_token_id,
            padding_side="left",  # Left padding for generation
        )

        sampler = None
        shuffle = True
        if self.is_distributed:
            sampler = DistributedSampler(self.train_dataset, shuffle=True)
            shuffle = False

        return DataLoader(
            self.train_dataset,
            batch_size=self.config.per_device_batch_size,
            shuffle=shuffle,
            sampler=sampler,
            collate_fn=collator,
            num_workers=4,
            pin_memory=True,
        )

    def _save_checkpoint(self, optimizer: torch.optim.Optimizer, final: bool = False):
        """Save training checkpoint."""
        if not self.is_main_process:
            return

        checkpoint_name = "final" if final else f"step_{self.global_step}"
        checkpoint_path = Path(self.config.checkpoint_dir) / checkpoint_name
        checkpoint_path.mkdir(parents=True, exist_ok=True)

        # Save model
        self.model.save_pretrained(str(checkpoint_path))

        # Save value model if present
        if self.value_model is not None:
            torch.save(
                self.value_model.state_dict(),
                checkpoint_path / "value_model.pt"
            )

        # Save training state
        state = {
            "global_step": self.global_step,
            "epoch": self.epoch,
            "optimizer_state_dict": optimizer.state_dict(),
            "config": self.config.to_dict(),
        }
        torch.save(state, checkpoint_path / "training_state.pt")

        logger.info(f"Saved checkpoint to {checkpoint_path}")

    def _load_checkpoint(self, checkpoint_path: str, optimizer: torch.optim.Optimizer):
        """Load training checkpoint."""
        checkpoint_path = Path(checkpoint_path)

        # Load training state
        state_path = checkpoint_path / "training_state.pt"
        if state_path.exists():
            state = torch.load(state_path, map_location=self.device)
            self.global_step = state["global_step"]
            self.epoch = state["epoch"]
            optimizer.load_state_dict(state["optimizer_state_dict"])

        # Load model
        self.model = AutoRegressiveLM.from_pretrained(str(checkpoint_path), device=str(self.device))

        # Load value model if present
        value_model_path = checkpoint_path / "value_model.pt"
        if self.value_model is not None and value_model_path.exists():
            self.value_model.load_state_dict(
                torch.load(value_model_path, map_location=self.device)
            )

        logger.info(f"Resumed from checkpoint {checkpoint_path} at step {self.global_step}")

    def _log_step_metrics(self, output):
        """Log step-level metrics."""
        metrics = {
            "step/policy_loss": output.policy_loss.item(),
            "step/value_loss": output.value_loss.item(),
            "step/approx_kl": output.approx_kl,
            "step/clip_fraction": output.clip_fraction,
            "step/global_step": self.global_step,
        }
        self._log_metrics(metrics)

    def _log_metrics(self, metrics: Dict[str, float]):
        """Log metrics to console and W&B."""
        if not self.is_main_process:
            return

        # Console logging
        metrics_str = " | ".join([f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}" for k, v in metrics.items()])
        logger.info(f"Step {self.global_step} | {metrics_str}")

        # W&B logging
        if self.wandb:
            self.wandb.log(metrics, step=self.global_step)
