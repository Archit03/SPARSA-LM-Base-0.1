"""
SPARSA-LM VAPO (Value-model Augmented Proximal Policy Optimization)
PPO variant with dense reward signals from value model
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List, Any
from dataclasses import dataclass


@dataclass
class VAPOOutput:
    """Output container for VAPO training step."""
    loss: torch.Tensor
    policy_loss: torch.Tensor
    value_loss: torch.Tensor
    dense_reward_loss: torch.Tensor
    kl_divergence: torch.Tensor
    clip_fraction: float
    approx_kl: float
    value_accuracy: float


class ValueModel(nn.Module):
    """
    Value model for VAPO dense reward estimation.

    Shares architecture with policy model but outputs scalar values
    for each token position.
    """

    def __init__(self, base_model: nn.Module, hidden_size: int):
        super().__init__()
        self.base_model = base_model
        self.value_head = nn.Linear(hidden_size, 1)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass returning per-token values.

        Args:
            input_ids: Input token IDs
            attention_mask: Optional attention mask

        Returns:
            Dictionary with 'values' tensor
        """
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_hidden_states=True,
        )

        # Use last hidden state for value prediction
        hidden_states = outputs["hidden_states"][-1]
        values = self.value_head(hidden_states).squeeze(-1)

        return {"values": values}


class VAPOTrainer:
    """
    VAPO (Value-model Augmented Proximal Policy Optimization) Trainer.

    Key innovations:
    1. Dense Rewards: Per-token reward signals from value model instead
       of sparse end-of-sequence rewards.
    2. Reward Smoothing: Temporal smoothing of dense rewards for stability.
    3. Value Clipping: Separate clipping for value function updates.
    4. Dual Value Training: Jointly train policy and value model.
    """

    def __init__(
        self,
        policy_model: nn.Module,
        value_model: nn.Module,
        reference_model: Optional[nn.Module] = None,
        tokenizer: Any = None,
        clip_range: float = 0.2,
        value_clip_range: float = 0.2,
        value_coef: float = 0.5,
        dense_reward: bool = True,
        reward_smoothing: float = 0.1,
        kl_coef: float = 0.1,
        target_kl: float = 0.1,
        gamma: float = 1.0,
        lam: float = 0.95,
        max_grad_norm: float = 1.0,
        device: str = "cuda",
    ):
        self.policy_model = policy_model
        self.value_model = value_model
        self.reference_model = reference_model
        self.tokenizer = tokenizer

        # VAPO-specific parameters
        self.dense_reward = dense_reward
        self.reward_smoothing = reward_smoothing
        self.value_clip_range = value_clip_range

        # Standard PPO parameters
        self.clip_range = clip_range
        self.value_coef = value_coef
        self.kl_coef = kl_coef
        self.target_kl = target_kl
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm

        self.device = torch.device(device)

        # Move models to device
        self.policy_model = self.policy_model.to(self.device)
        self.value_model = self.value_model.to(self.device)
        if self.reference_model is not None:
            self.reference_model = self.reference_model.to(self.device)
            self.reference_model.eval()

        # Running statistics for reward normalization
        self.reward_mean = 0.0
        self.reward_std = 1.0
        self.reward_count = 0

        # Statistics tracking
        self.stats = {
            "approx_kl": [],
            "clip_fraction": [],
            "value_accuracy": [],
            "policy_loss": [],
            "value_loss": [],
            "dense_reward": [],
        }

    @torch.no_grad()
    def generate_responses(
        self,
        prompts: torch.Tensor,
        attention_mask: torch.Tensor,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generate responses with value estimates.

        Returns:
            Tuple of (sequences, log_probs, ref_log_probs, values)
        """
        self.policy_model.eval()
        self.value_model.eval()

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

        # Compute values
        value_outputs = self.value_model(input_ids=generated)
        values = value_outputs["values"][:, prompts.shape[1] - 1:-1]

        self.policy_model.train()
        self.value_model.train()

        return generated, log_probs, ref_log_probs, values

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

        log_probs = F.log_softmax(logits, dim=-1)
        generated_tokens = sequences[:, prompt_length:]
        token_log_probs = torch.gather(
            log_probs[:, prompt_length - 1:-1],
            dim=-1,
            index=generated_tokens.unsqueeze(-1)
        ).squeeze(-1)

        return token_log_probs

    def compute_dense_rewards(
        self,
        sequences: torch.Tensor,
        final_scores: torch.Tensor,
        prompt_lengths: torch.Tensor,
        values: torch.Tensor,
        log_probs: torch.Tensor,
        ref_log_probs: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute dense per-token rewards using value model.

        The key idea is to distribute the final reward across tokens
        based on value function differences.

        Args:
            sequences: Generated sequences
            final_scores: End-of-sequence reward scores
            prompt_lengths: Length of each prompt
            values: Per-token value estimates
            log_probs: Policy log probabilities
            ref_log_probs: Reference log probabilities

        Returns:
            Dense per-token rewards
        """
        batch_size, seq_len = values.shape

        if not self.dense_reward:
            # Sparse reward: only at last token
            rewards = torch.zeros_like(values)
            kl = log_probs - ref_log_probs
            non_score_rewards = -self.kl_coef * kl

            for i in range(batch_size):
                response_length = sequences.shape[1] - prompt_lengths[i]
                if response_length > 0:
                    rewards[i, :response_length] = non_score_rewards[i, :response_length]
                    rewards[i, response_length - 1] += final_scores[i]
            return rewards

        # Dense reward computation
        # Use value differences to distribute reward
        value_diffs = torch.zeros_like(values)
        value_diffs[:, :-1] = values[:, 1:] - values[:, :-1]
        value_diffs[:, -1] = final_scores - values[:, -1]

        # Apply temporal smoothing
        if self.reward_smoothing > 0:
            smoothed_diffs = torch.zeros_like(value_diffs)
            for t in range(seq_len):
                if t == 0:
                    smoothed_diffs[:, t] = value_diffs[:, t]
                else:
                    smoothed_diffs[:, t] = (
                        (1 - self.reward_smoothing) * value_diffs[:, t] +
                        self.reward_smoothing * smoothed_diffs[:, t - 1]
                    )
            value_diffs = smoothed_diffs

        # Add KL penalty
        kl = log_probs - ref_log_probs
        kl_penalty = -self.kl_coef * kl

        # Combine dense rewards with KL penalty
        dense_rewards = value_diffs + kl_penalty

        # Normalize rewards
        self._update_reward_stats(dense_rewards)
        dense_rewards = (dense_rewards - self.reward_mean) / (self.reward_std + 1e-8)

        return dense_rewards

    def _update_reward_stats(self, rewards: torch.Tensor):
        """Update running reward statistics for normalization."""
        batch_mean = rewards.mean().item()
        batch_std = rewards.std().item()
        batch_count = rewards.numel()

        # Welford's online algorithm for running mean and variance
        total_count = self.reward_count + batch_count
        delta = batch_mean - self.reward_mean
        self.reward_mean += delta * batch_count / total_count

        # Update variance using parallel algorithm
        if self.reward_count > 0:
            m_a = self.reward_std ** 2 * self.reward_count
            m_b = batch_std ** 2 * batch_count
            m2 = m_a + m_b + delta ** 2 * self.reward_count * batch_count / total_count
            self.reward_std = (m2 / total_count) ** 0.5

        self.reward_count = total_count

    def compute_advantages(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        masks: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute GAE with dense rewards.

        Args:
            rewards: Dense per-token rewards
            values: Value estimates
            masks: Attention masks for valid tokens

        Returns:
            Tuple of (advantages, returns)
        """
        batch_size, seq_len = rewards.shape
        advantages = torch.zeros_like(rewards)

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
        old_values: torch.Tensor,
        advantages: torch.Tensor,
        returns: torch.Tensor,
        prompt_lengths: torch.Tensor,
    ) -> VAPOOutput:
        """
        Perform single VAPO training step.

        Args:
            sequences: Generated sequences
            attention_mask: Attention mask
            old_log_probs: Log probs from rollout
            old_values: Values from rollout
            advantages: Computed advantages
            returns: Computed returns
            prompt_lengths: Length of each prompt

        Returns:
            VAPOOutput containing losses and metrics
        """
        # Forward pass - policy
        policy_outputs = self.policy_model(input_ids=sequences, attention_mask=attention_mask)
        logits = policy_outputs["logits"]

        # Forward pass - value
        value_outputs = self.value_model(input_ids=sequences, attention_mask=attention_mask)
        values = value_outputs["values"][:, :-1]

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

        # Policy loss with clipping
        ratio = torch.exp(new_log_probs - old_log_probs)
        pg_loss1 = -advantages * ratio
        pg_loss2 = -advantages * torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range)
        policy_loss = torch.max(pg_loss1, pg_loss2)
        policy_loss = (policy_loss * response_mask).sum() / response_mask.sum()

        # VAPO: Value loss with clipping
        value_pred = values
        value_pred_clipped = old_values + torch.clamp(
            value_pred - old_values,
            -self.value_clip_range,
            self.value_clip_range
        )

        value_loss1 = (value_pred - returns) ** 2
        value_loss2 = (value_pred_clipped - returns) ** 2
        value_loss = 0.5 * torch.max(value_loss1, value_loss2)
        value_loss = (value_loss * response_mask).sum() / response_mask.sum()

        # Dense reward auxiliary loss (encourage value model accuracy)
        dense_reward_loss = F.mse_loss(
            values[:, :-1] * response_mask[:, :-1],
            values[:, 1:].detach() * response_mask[:, :-1]
        )

        # Total loss
        loss = policy_loss + self.value_coef * value_loss + 0.1 * dense_reward_loss

        # Compute metrics
        with torch.no_grad():
            approx_kl = ((ratio - 1) - (ratio.log())).mean().item()
            clip_fraction = ((ratio - 1).abs() > self.clip_range).float().mean().item()

            # Value accuracy (how well value predicts returns)
            value_error = ((values - returns).abs() * response_mask).sum() / response_mask.sum()
            value_accuracy = 1.0 - min(1.0, value_error.item())

        # Update stats
        self.stats["approx_kl"].append(approx_kl)
        self.stats["clip_fraction"].append(clip_fraction)
        self.stats["value_accuracy"].append(value_accuracy)
        self.stats["policy_loss"].append(policy_loss.item())
        self.stats["value_loss"].append(value_loss.item())
        self.stats["dense_reward"].append(dense_reward_loss.item())

        # KL divergence
        kl_divergence = (new_log_probs - old_log_probs).mean()

        return VAPOOutput(
            loss=loss,
            policy_loss=policy_loss,
            value_loss=value_loss,
            dense_reward_loss=dense_reward_loss,
            kl_divergence=kl_divergence,
            clip_fraction=clip_fraction,
            approx_kl=approx_kl,
            value_accuracy=value_accuracy,
        )

    def update_kl_coef(self, kl: float):
        """Adaptively update KL coefficient."""
        if kl > self.target_kl * 1.5:
            self.kl_coef *= 1.5
        elif kl < self.target_kl / 1.5:
            self.kl_coef /= 1.5
        self.kl_coef = max(0.01, min(1.0, self.kl_coef))

    def get_stats(self) -> Dict[str, float]:
        """Get training statistics."""
        stats = {}
        for key, values in self.stats.items():
            if values:
                stats[f"vapo/{key}"] = sum(values) / len(values)
        stats["vapo/kl_coef"] = self.kl_coef
        stats["vapo/reward_mean"] = self.reward_mean
        stats["vapo/reward_std"] = self.reward_std
        return stats

    def reset_stats(self):
        """Reset statistics for new epoch."""
        for key in self.stats:
            self.stats[key] = []
