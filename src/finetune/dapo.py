"""
SPARSA-LM DAPO (Decoupled Clip and Dynamic Sampling Policy Optimization)
Advanced PPO variant for language model finetuning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List, Any
from dataclasses import dataclass
import math


@dataclass
class DAPOOutput:
    """Output container for DAPO training step."""
    loss: torch.Tensor
    policy_loss: torch.Tensor
    value_loss: torch.Tensor
    entropy_loss: torch.Tensor
    kl_divergence: torch.Tensor
    clip_fraction: float
    approx_kl: float
    entropy: float


class DAPOTrainer:
    """
    DAPO (Decoupled Clip and Dynamic Sampling Policy Optimization) Trainer.

    Key innovations:
    1. Decoupled Clipping: Separate upper and lower clip bounds for more
       fine-grained control over policy updates.
    2. Dynamic Sampling: Adaptive temperature scheduling during generation
       to prevent entropy collapse.
    3. Entropy Target: Maintains minimum policy entropy for exploration.
    """

    def __init__(
        self,
        policy_model: nn.Module,
        reference_model: Optional[nn.Module] = None,
        value_model: Optional[nn.Module] = None,
        tokenizer: Any = None,
        clip_range_upper: float = 0.2,
        clip_range_lower: float = 0.1,
        dynamic_sampling: bool = True,
        sampling_temperature_init: float = 1.0,
        sampling_temperature_decay: float = 0.995,
        entropy_coef: float = 0.01,
        entropy_target: float = 0.1,
        value_coef: float = 0.5,
        kl_coef: float = 0.1,
        target_kl: float = 0.1,
        gamma: float = 1.0,
        lam: float = 0.95,
        max_grad_norm: float = 1.0,
        device: str = "cuda",
    ):
        self.policy_model = policy_model
        self.reference_model = reference_model
        self.value_model = value_model
        self.tokenizer = tokenizer

        # DAPO-specific parameters
        self.clip_range_upper = clip_range_upper
        self.clip_range_lower = clip_range_lower
        self.dynamic_sampling = dynamic_sampling
        self.sampling_temperature = sampling_temperature_init
        self.sampling_temperature_decay = sampling_temperature_decay
        self.entropy_coef = entropy_coef
        self.entropy_target = entropy_target

        # Standard PPO parameters
        self.value_coef = value_coef
        self.kl_coef = kl_coef
        self.target_kl = target_kl
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm

        self.device = torch.device(device)

        # Move models to device
        self.policy_model = self.policy_model.to(self.device)
        if self.reference_model is not None:
            self.reference_model = self.reference_model.to(self.device)
            self.reference_model.eval()
        if self.value_model is not None:
            self.value_model = self.value_model.to(self.device)

        # Statistics tracking
        self.stats = {
            "approx_kl": [],
            "clip_fraction": [],
            "entropy": [],
            "policy_loss": [],
            "value_loss": [],
        }

    @torch.no_grad()
    def generate_responses(
        self,
        prompts: torch.Tensor,
        attention_mask: torch.Tensor,
        max_new_tokens: int = 256,
        top_p: float = 0.9,
        top_k: int = 50,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generate responses using dynamic sampling temperature.

        Returns:
            Tuple of (sequences, log_probs, ref_log_probs)
        """
        self.policy_model.eval()

        batch_size = prompts.shape[0]
        prompt_length = prompts.shape[1]

        # Use dynamic temperature
        temperature = self.sampling_temperature if self.dynamic_sampling else 1.0

        # Generate sequences
        generated = self.policy_model.generate(
            input_ids=prompts,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            do_sample=True,
        )

        # Compute log probabilities
        log_probs = self._compute_log_probs(generated, prompts.shape[1])

        # Compute reference log probabilities
        if self.reference_model is not None:
            ref_log_probs = self._compute_log_probs(
                generated, prompts.shape[1], use_reference=True
            )
        else:
            ref_log_probs = torch.zeros_like(log_probs)

        self.policy_model.train()

        return generated, log_probs, ref_log_probs

    def _compute_log_probs(
        self,
        sequences: torch.Tensor,
        prompt_length: int,
        use_reference: bool = False,
    ) -> torch.Tensor:
        """Compute log probabilities for generated tokens."""
        model = self.reference_model if use_reference else self.policy_model

        with torch.no_grad() if use_reference else torch.enable_grad():
            outputs = model(input_ids=sequences)
            logits = outputs["logits"]

        # Get log probs for generated tokens only
        log_probs = F.log_softmax(logits, dim=-1)
        generated_tokens = sequences[:, prompt_length:]
        token_log_probs = torch.gather(
            log_probs[:, prompt_length - 1:-1],
            dim=-1,
            index=generated_tokens.unsqueeze(-1)
        ).squeeze(-1)

        return token_log_probs

    def compute_rewards(
        self,
        sequences: torch.Tensor,
        scores: torch.Tensor,
        prompt_lengths: torch.Tensor,
        log_probs: torch.Tensor,
        ref_log_probs: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute rewards with KL penalty.

        Args:
            sequences: Generated sequences
            scores: Reward model scores
            prompt_lengths: Length of each prompt
            log_probs: Policy log probabilities
            ref_log_probs: Reference log probabilities

        Returns:
            Tuple of (rewards, non_score_rewards)
        """
        # KL divergence penalty
        kl = log_probs - ref_log_probs
        non_score_rewards = -self.kl_coef * kl

        # Add reward at the last token
        rewards = non_score_rewards.clone()
        for i in range(sequences.shape[0]):
            response_length = sequences.shape[1] - prompt_lengths[i]
            if response_length > 0:
                rewards[i, response_length - 1] += scores[i]

        return rewards, non_score_rewards

    def compute_advantages(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        masks: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute GAE (Generalized Advantage Estimation).

        Args:
            rewards: Per-token rewards
            values: Value estimates
            masks: Attention masks for valid tokens

        Returns:
            Tuple of (advantages, returns)
        """
        batch_size, seq_len = rewards.shape
        advantages = torch.zeros_like(rewards)
        returns = torch.zeros_like(rewards)

        lastgaelam = 0
        for t in reversed(range(seq_len)):
            if t == seq_len - 1:
                next_value = 0
                next_mask = 0
            else:
                next_value = values[:, t + 1]
                next_mask = masks[:, t + 1]

            delta = rewards[:, t] + self.gamma * next_value * next_mask - values[:, t]
            advantages[:, t] = lastgaelam = delta + self.gamma * self.lam * next_mask * lastgaelam

        returns = advantages + values

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return advantages, returns

    def train_step(
        self,
        sequences: torch.Tensor,
        attention_mask: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        returns: torch.Tensor,
        prompt_lengths: torch.Tensor,
    ) -> DAPOOutput:
        """
        Perform single DAPO training step with decoupled clipping.

        Args:
            sequences: Generated sequences
            attention_mask: Attention mask
            old_log_probs: Log probs from rollout
            advantages: Computed advantages
            returns: Computed returns
            prompt_lengths: Length of each prompt

        Returns:
            DAPOOutput containing losses and metrics
        """
        # Forward pass
        outputs = self.policy_model(input_ids=sequences, attention_mask=attention_mask)
        logits = outputs["logits"]

        # Compute new log probs
        log_probs = F.log_softmax(logits, dim=-1)

        # Create response mask
        batch_size, seq_len, vocab_size = logits.shape
        response_mask = torch.zeros(batch_size, seq_len - 1, device=self.device)
        for i in range(batch_size):
            prompt_len = prompt_lengths[i]
            response_mask[i, prompt_len - 1:] = attention_mask[i, prompt_len:]

        # Get token log probs
        generated_tokens = sequences[:, 1:]
        new_log_probs = torch.gather(
            log_probs[:, :-1],
            dim=-1,
            index=generated_tokens.unsqueeze(-1)
        ).squeeze(-1)

        # Compute ratio
        ratio = torch.exp(new_log_probs - old_log_probs)

        # DAPO: Decoupled clipping
        # Upper bound: limits how much we can increase probability
        # Lower bound: limits how much we can decrease probability
        advantages_positive = (advantages > 0).float()
        advantages_negative = 1 - advantages_positive

        # Apply different clips based on advantage sign
        ratio_clipped_upper = torch.clamp(ratio, 1 - self.clip_range_lower, 1 + self.clip_range_upper)
        ratio_clipped_lower = torch.clamp(ratio, 1 - self.clip_range_upper, 1 + self.clip_range_lower)

        ratio_clipped = (
            advantages_positive * ratio_clipped_upper +
            advantages_negative * ratio_clipped_lower
        )

        # Policy loss with decoupled clipping
        pg_loss1 = -advantages * ratio
        pg_loss2 = -advantages * ratio_clipped
        policy_loss = torch.max(pg_loss1, pg_loss2)
        policy_loss = (policy_loss * response_mask).sum() / response_mask.sum()

        # Value loss (if using value model)
        if self.value_model is not None:
            value_outputs = self.value_model(input_ids=sequences, attention_mask=attention_mask)
            values = value_outputs["logits"].squeeze(-1)[:, :-1]
            value_loss = F.mse_loss(values * response_mask, returns * response_mask)
        else:
            value_loss = torch.tensor(0.0, device=self.device)

        # Entropy loss with target
        probs = F.softmax(logits[:, :-1], dim=-1)
        log_probs_all = F.log_softmax(logits[:, :-1], dim=-1)
        entropy = -(probs * log_probs_all).sum(dim=-1)
        entropy = (entropy * response_mask).sum() / response_mask.sum()

        # DAPO: Entropy targeting to prevent collapse
        entropy_loss = self.entropy_coef * torch.max(
            torch.tensor(0.0, device=self.device),
            self.entropy_target - entropy
        )

        # Total loss
        loss = policy_loss + self.value_coef * value_loss + entropy_loss

        # Compute metrics
        with torch.no_grad():
            approx_kl = ((ratio - 1) - (ratio.log())).mean().item()
            clip_fraction = ((ratio - 1).abs() > self.clip_range_upper).float().mean().item()

        # Update stats
        self.stats["approx_kl"].append(approx_kl)
        self.stats["clip_fraction"].append(clip_fraction)
        self.stats["entropy"].append(entropy.item())
        self.stats["policy_loss"].append(policy_loss.item())
        self.stats["value_loss"].append(value_loss.item())

        # KL divergence for adaptive KL
        kl_divergence = (new_log_probs - old_log_probs).mean()

        return DAPOOutput(
            loss=loss,
            policy_loss=policy_loss,
            value_loss=value_loss,
            entropy_loss=entropy_loss,
            kl_divergence=kl_divergence,
            clip_fraction=clip_fraction,
            approx_kl=approx_kl,
            entropy=entropy.item(),
        )

    def update_sampling_temperature(self):
        """Update dynamic sampling temperature."""
        if self.dynamic_sampling:
            self.sampling_temperature *= self.sampling_temperature_decay
            # Clamp to reasonable bounds
            self.sampling_temperature = max(0.5, min(2.0, self.sampling_temperature))

    def update_kl_coef(self, kl: float):
        """Adaptively update KL coefficient."""
        if kl > self.target_kl * 1.5:
            self.kl_coef *= 1.5
        elif kl < self.target_kl / 1.5:
            self.kl_coef /= 1.5
        # Clamp to reasonable bounds
        self.kl_coef = max(0.01, min(1.0, self.kl_coef))

    def get_stats(self) -> Dict[str, float]:
        """Get training statistics."""
        stats = {}
        for key, values in self.stats.items():
            if values:
                stats[f"dapo/{key}"] = sum(values) / len(values)
        stats["dapo/sampling_temperature"] = self.sampling_temperature
        stats["dapo/kl_coef"] = self.kl_coef
        return stats

    def reset_stats(self):
        """Reset statistics for new epoch."""
        for key in self.stats:
            self.stats[key] = []
