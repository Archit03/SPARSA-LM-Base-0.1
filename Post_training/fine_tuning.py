"""
SPARSA-LM RLAIF Fine-Tuning Pipeline with DAPO and VAPO

Implements:
- DAPO: Decoupled Clip and Dynamic Sampling Policy Optimization
- VAPO: Value-model Augmented Proximal Policy Optimization
- RLAIF: Reinforcement Learning from AI Feedback

Features:
- Decoupled clipping for policy gradient (upper/lower bounds)
- Dynamic sampling based on performance metrics
- Value model for dense reward signals
- External AI feedback reward model (LLM-as-judge)

This code belongs to EllanorAI and is licensed under the EllanorAI Proprietary License.
"""

import os
import math
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from abc import ABC, abstractmethod
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False

try:
    import deepspeed
    DEEPSPEED_AVAILABLE = True
except ImportError:
    DEEPSPEED_AVAILABLE = False

# Import local modules
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.model import SPARSALM, SPARSAConfig


@dataclass
class RLAIFConfig:
    """Configuration for RLAIF training."""

    # Training settings
    learning_rate: float = 1e-5
    num_epochs: int = 3
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    max_grad_norm: float = 1.0

    # Generation settings
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50

    # PPO/DAPO settings
    gamma: float = 1.0
    gae_lambda: float = 0.95
    ppo_epochs: int = 4
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.01

    # DAPO-specific settings
    use_dapo: bool = True
    clip_eps_upper: float = 0.2  # Upper bound for policy clip
    clip_eps_lower: float = 0.1  # Lower bound (prevent entropy collapse)
    dynamic_sampling: bool = True
    sampling_temperature_decay: float = 0.995
    min_sampling_temperature: float = 0.3

    # VAPO-specific settings
    use_vapo: bool = True
    value_model_coef: float = 0.3
    use_dense_rewards: bool = True
    reward_smoothing: float = 0.1

    # Reward model settings
    reward_model_type: str = "ai_feedback"  # "ai_feedback", "learned", "hybrid"
    judge_model: str = "meta-llama/Llama-3-70B-Instruct"  # External judge for RLAIF
    reward_baseline: str = "moving_average"

    # KL divergence settings
    kl_coef: float = 0.1
    target_kl: float = 0.01
    adaptive_kl: bool = True

    # Device settings
    device: str = "cuda"
    use_bf16: bool = True


class RewardModel(nn.Module):
    """Learned reward model for RLHF."""

    def __init__(self, base_model: SPARSALM):
        super().__init__()
        self.backbone = base_model
        self.reward_head = nn.Linear(base_model.hidden_dim, 1)

        # Initialize reward head
        nn.init.zeros_(self.reward_head.weight)
        nn.init.zeros_(self.reward_head.bias)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute reward scores for sequences."""
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )

        # Use last hidden state at the last non-padding position
        hidden_states = outputs["hidden_states"][-1]

        if attention_mask is not None:
            # Find last non-padding position
            seq_lengths = attention_mask.sum(dim=1) - 1
            batch_indices = torch.arange(hidden_states.size(0), device=hidden_states.device)
            last_hidden = hidden_states[batch_indices, seq_lengths]
        else:
            last_hidden = hidden_states[:, -1, :]

        rewards = self.reward_head(last_hidden).squeeze(-1)
        return rewards


class ValueModel(nn.Module):
    """Value model for estimating state values (used in VAPO)."""

    def __init__(self, base_model: SPARSALM):
        super().__init__()
        self.backbone = base_model
        self.value_head = nn.Sequential(
            nn.Linear(base_model.hidden_dim, base_model.hidden_dim),
            nn.GELU(),
            nn.Linear(base_model.hidden_dim, 1),
        )

        # Initialize value head
        for layer in self.value_head:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=0.01)
                nn.init.zeros_(layer.bias)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute value estimates for each position."""
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )

        hidden_states = outputs["hidden_states"][-1]
        values = self.value_head(hidden_states).squeeze(-1)

        return values


class AIFeedbackRewardModel:
    """
    Reward model using AI feedback (LLM-as-judge).
    Uses an external large model to score responses.
    """

    def __init__(
        self,
        judge_model: str = "meta-llama/Llama-3-70B-Instruct",
        device: str = "cuda",
    ):
        self.judge_model_name = judge_model
        self.device = device
        self.judge_model = None
        self.judge_tokenizer = None

        # Reward prompts for different criteria
        self.reward_prompts = {
            "helpfulness": """Rate the helpfulness of the following response on a scale of 1-10.
Response: {response}
Rating (just the number):""",

            "accuracy": """Rate the factual accuracy of the following response on a scale of 1-10.
Response: {response}
Rating (just the number):""",

            "coherence": """Rate the coherence and clarity of the following response on a scale of 1-10.
Response: {response}
Rating (just the number):""",

            "overall": """You are evaluating an AI assistant's response.
Question: {prompt}
Response: {response}

Rate the overall quality of this response on a scale of 1-10, considering:
- Helpfulness and relevance
- Accuracy and correctness
- Clarity and coherence
- Completeness

Rating (just the number):""",
        }

    def load_judge(self):
        """Lazy load the judge model."""
        if self.judge_model is None and HF_AVAILABLE:
            try:
                self.judge_tokenizer = AutoTokenizer.from_pretrained(self.judge_model_name)
                self.judge_model = AutoModelForCausalLM.from_pretrained(
                    self.judge_model_name,
                    torch_dtype=torch.bfloat16,
                    device_map="auto",
                    trust_remote_code=True,
                )
            except Exception as e:
                logging.warning(f"Failed to load judge model: {e}")
                self.judge_model = None

    def compute_reward(
        self,
        prompts: List[str],
        responses: List[str],
        criterion: str = "overall",
    ) -> torch.Tensor:
        """Compute rewards using AI feedback."""
        self.load_judge()

        rewards = []
        for prompt, response in zip(prompts, responses):
            reward = self._score_single(prompt, response, criterion)
            rewards.append(reward)

        return torch.tensor(rewards, dtype=torch.float32)

    def _score_single(self, prompt: str, response: str, criterion: str) -> float:
        """Score a single response."""
        if self.judge_model is None:
            # Fallback: use simple heuristics
            return self._heuristic_score(response)

        try:
            eval_prompt = self.reward_prompts[criterion].format(
                prompt=prompt,
                response=response,
            )

            inputs = self.judge_tokenizer(
                eval_prompt,
                return_tensors="pt",
                truncation=True,
                max_length=2048,
            ).to(self.device)

            with torch.no_grad():
                outputs = self.judge_model.generate(
                    **inputs,
                    max_new_tokens=5,
                    temperature=0.1,
                    do_sample=False,
                )

            generated = self.judge_tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True,
            )

            # Extract numeric rating
            try:
                rating = float(generated.strip().split()[0])
                rating = max(1.0, min(10.0, rating))  # Clamp to valid range
            except (ValueError, IndexError):
                rating = 5.0  # Default to middle

            # Normalize to [-1, 1] range
            return (rating - 5.5) / 4.5

        except Exception as e:
            logging.warning(f"Error in AI scoring: {e}")
            return 0.0

    def _heuristic_score(self, response: str) -> float:
        """Simple heuristic scoring when judge model is unavailable."""
        score = 0.0

        # Length penalty/bonus
        words = len(response.split())
        if words < 10:
            score -= 0.3
        elif words > 500:
            score -= 0.1
        elif 50 < words < 200:
            score += 0.2

        # Coherence heuristics
        if response.strip().endswith(('.', '!', '?')):
            score += 0.1

        # Avoid empty or very short responses
        if len(response.strip()) < 20:
            score -= 0.5

        return max(-1.0, min(1.0, score))


class DAPOTrainer:
    """
    DAPO: Decoupled Clip and Dynamic Sampling Policy Optimization

    Key features:
    1. Decoupled Clipping: Separate upper/lower clip thresholds to prevent entropy collapse
    2. Dynamic Sampling: Filter prompt groups based on performance metrics
    3. Token-level rewards for fine-grained optimization
    """

    def __init__(
        self,
        model: SPARSALM,
        ref_model: SPARSALM,
        tokenizer,
        config: RLAIFConfig,
        reward_model: Optional[RewardModel] = None,
        value_model: Optional[ValueModel] = None,
    ):
        self.model = model
        self.ref_model = ref_model
        self.tokenizer = tokenizer
        self.config = config
        self.reward_model = reward_model
        self.value_model = value_model

        # AI feedback reward
        self.ai_reward = AIFeedbackRewardModel(
            judge_model=config.judge_model,
            device=config.device,
        )

        # Optimizer
        self.optimizer = AdamW(
            model.parameters(),
            lr=config.learning_rate,
            betas=(0.9, 0.95),
            weight_decay=0.01,
        )

        # Statistics for dynamic sampling
        self.prompt_stats: Dict[str, Dict[str, float]] = {}
        self.running_reward_mean = 0.0
        self.running_reward_std = 1.0
        self.update_count = 0

        # Adaptive KL coefficient
        self.kl_coef = config.kl_coef

        # Freeze reference model
        for param in self.ref_model.parameters():
            param.requires_grad = False

    def compute_advantages(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        masks: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute GAE advantages and returns."""
        advantages = torch.zeros_like(rewards)
        returns = torch.zeros_like(rewards)

        gae = 0.0
        for t in reversed(range(rewards.size(1))):
            if t == rewards.size(1) - 1:
                next_value = 0.0
            else:
                next_value = values[:, t + 1]

            delta = rewards[:, t] + self.config.gamma * next_value * masks[:, t] - values[:, t]
            gae = delta + self.config.gamma * self.config.gae_lambda * masks[:, t] * gae
            advantages[:, t] = gae
            returns[:, t] = gae + values[:, t]

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return advantages, returns

    def compute_policy_loss_dapo(
        self,
        log_probs: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        masks: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute DAPO policy loss with decoupled clipping.

        Uses different clip thresholds for upper and lower bounds:
        - Upper clip (eps_upper): Prevents policy from moving too far in positive direction
        - Lower clip (eps_lower): Allows policy to reduce probability of bad actions more freely
        """
        ratio = torch.exp(log_probs - old_log_probs)

        # Decoupled clipping
        eps_upper = self.config.clip_eps_upper
        eps_lower = self.config.clip_eps_lower

        # Clip with asymmetric bounds
        clipped_ratio = torch.where(
            advantages > 0,
            torch.clamp(ratio, 1.0 - eps_lower, 1.0 + eps_upper),
            torch.clamp(ratio, 1.0 - eps_upper, 1.0 + eps_lower),
        )

        # Compute surrogate losses
        surr1 = ratio * advantages
        surr2 = clipped_ratio * advantages

        # Take minimum (pessimistic bound)
        policy_loss = -torch.min(surr1, surr2)

        # Apply mask
        policy_loss = (policy_loss * masks).sum() / masks.sum()

        return policy_loss

    def compute_value_loss(
        self,
        values: torch.Tensor,
        old_values: torch.Tensor,
        returns: torch.Tensor,
        masks: torch.Tensor,
    ) -> torch.Tensor:
        """Compute clipped value loss."""
        # Value clipping
        value_clipped = old_values + torch.clamp(
            values - old_values,
            -self.config.clip_eps_upper,
            self.config.clip_eps_upper,
        )

        # Compute both losses
        value_loss1 = F.mse_loss(values, returns, reduction='none')
        value_loss2 = F.mse_loss(value_clipped, returns, reduction='none')

        # Take maximum
        value_loss = torch.max(value_loss1, value_loss2)
        value_loss = (value_loss * masks).sum() / masks.sum()

        return value_loss

    def compute_kl_divergence(
        self,
        log_probs: torch.Tensor,
        ref_log_probs: torch.Tensor,
        masks: torch.Tensor,
    ) -> torch.Tensor:
        """Compute KL divergence from reference model."""
        kl = torch.exp(ref_log_probs) * (ref_log_probs - log_probs)
        kl = (kl * masks).sum() / masks.sum()
        return kl

    def dynamic_sampling_filter(
        self,
        prompts: List[str],
        prompt_rewards: Dict[str, List[float]],
    ) -> List[str]:
        """
        Filter prompts based on dynamic sampling criteria.
        Removes prompts that are too easy or too hard for stable training.
        """
        if not self.config.dynamic_sampling:
            return prompts

        filtered_prompts = []
        for prompt in prompts:
            if prompt not in prompt_rewards:
                filtered_prompts.append(prompt)
                continue

            rewards = prompt_rewards[prompt]
            if len(rewards) < 3:
                filtered_prompts.append(prompt)
                continue

            # Compute statistics
            mean_reward = sum(rewards) / len(rewards)
            variance = sum((r - mean_reward) ** 2 for r in rewards) / len(rewards)
            std_reward = variance ** 0.5

            # Filter based on performance consistency
            # Keep prompts with moderate difficulty (not too easy, not too hard)
            if -0.8 < mean_reward < 0.8 and std_reward > 0.1:
                filtered_prompts.append(prompt)
            elif len(rewards) < 10:
                # Keep for more exploration
                filtered_prompts.append(prompt)

        return filtered_prompts if filtered_prompts else prompts

    @torch.no_grad()
    def generate_responses(
        self,
        prompts: List[str],
    ) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
        """Generate responses for prompts."""
        self.model.eval()

        # Tokenize prompts
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config.max_new_tokens,
        ).to(self.config.device)

        # Generate
        outputs = self.model.generate(
            input_ids=inputs["input_ids"],
            max_new_tokens=self.config.max_new_tokens,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            top_k=self.config.top_k,
            do_sample=True,
        )

        # Decode responses
        responses = self.tokenizer.batch_decode(
            outputs[:, inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )

        return inputs["input_ids"], outputs, responses

    def train_step(
        self,
        prompts: List[str],
    ) -> Dict[str, float]:
        """Perform a single training step."""
        self.model.train()

        # Generate responses
        prompt_ids, response_ids, responses = self.generate_responses(prompts)

        # Compute rewards
        rewards = self.ai_reward.compute_reward(prompts, responses)
        rewards = rewards.to(self.config.device)

        # Update running statistics
        self.running_reward_mean = 0.99 * self.running_reward_mean + 0.01 * rewards.mean().item()
        self.running_reward_std = 0.99 * self.running_reward_std + 0.01 * rewards.std().item()

        # Normalize rewards
        rewards = (rewards - self.running_reward_mean) / (self.running_reward_std + 1e-8)

        # Compute log probabilities and values
        attention_mask = (response_ids != self.tokenizer.pad_token_id).long()

        # Forward pass through policy model
        outputs = self.model(
            input_ids=response_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )

        # Compute log probs
        logits = outputs["logits"][:, :-1, :]
        labels = response_ids[:, 1:]
        log_probs = F.log_softmax(logits, dim=-1)
        log_probs = log_probs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)

        # Reference model log probs
        with torch.no_grad():
            ref_outputs = self.ref_model(
                input_ids=response_ids,
                attention_mask=attention_mask,
                return_dict=True,
            )
            ref_logits = ref_outputs["logits"][:, :-1, :]
            ref_log_probs = F.log_softmax(ref_logits, dim=-1)
            ref_log_probs = ref_log_probs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)

        # Value model predictions
        if self.value_model is not None and self.config.use_vapo:
            values = self.value_model(response_ids, attention_mask)[:, :-1]
        else:
            values = torch.zeros_like(log_probs)

        # Create token-level rewards (dense rewards for VAPO)
        token_rewards = self._compute_dense_rewards(rewards, response_ids.shape[1] - 1)

        # Compute masks
        masks = attention_mask[:, 1:].float()

        # Compute advantages
        advantages, returns = self.compute_advantages(token_rewards, values, masks)

        # Store old values for PPO updates
        old_log_probs = log_probs.detach()
        old_values = values.detach()

        # PPO/DAPO optimization loop
        total_loss = 0.0
        policy_losses = []
        value_losses = []
        kl_divs = []

        for _ in range(self.config.ppo_epochs):
            # Recompute predictions
            outputs = self.model(
                input_ids=response_ids,
                attention_mask=attention_mask,
                return_dict=True,
            )
            logits = outputs["logits"][:, :-1, :]
            log_probs = F.log_softmax(logits, dim=-1)
            log_probs = log_probs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)

            if self.value_model is not None and self.config.use_vapo:
                values = self.value_model(response_ids, attention_mask)[:, :-1]
            else:
                values = torch.zeros_like(log_probs)

            # Compute losses
            policy_loss = self.compute_policy_loss_dapo(
                log_probs, old_log_probs, advantages, masks
            )

            value_loss = self.compute_value_loss(
                values, old_values, returns, masks
            )

            kl_div = self.compute_kl_divergence(log_probs, ref_log_probs, masks)

            # Entropy bonus
            probs = F.softmax(logits, dim=-1)
            entropy = -(probs * log_probs.unsqueeze(-1).exp() * F.log_softmax(logits, dim=-1)).sum(-1)
            entropy = (entropy * masks).sum() / masks.sum()

            # Total loss
            loss = (
                policy_loss
                + self.config.value_loss_coef * value_loss
                + self.kl_coef * kl_div
                - self.config.entropy_coef * entropy
            )

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
            self.optimizer.step()

            total_loss += loss.item()
            policy_losses.append(policy_loss.item())
            value_losses.append(value_loss.item())
            kl_divs.append(kl_div.item())

        # Adaptive KL coefficient
        if self.config.adaptive_kl:
            mean_kl = sum(kl_divs) / len(kl_divs)
            if mean_kl > self.config.target_kl * 1.5:
                self.kl_coef *= 1.5
            elif mean_kl < self.config.target_kl * 0.5:
                self.kl_coef /= 1.5
            self.kl_coef = max(0.001, min(1.0, self.kl_coef))

        self.update_count += 1

        return {
            "loss": total_loss / self.config.ppo_epochs,
            "policy_loss": sum(policy_losses) / len(policy_losses),
            "value_loss": sum(value_losses) / len(value_losses),
            "kl_div": sum(kl_divs) / len(kl_divs),
            "mean_reward": rewards.mean().item(),
            "kl_coef": self.kl_coef,
        }

    def _compute_dense_rewards(
        self,
        final_rewards: torch.Tensor,
        seq_len: int,
    ) -> torch.Tensor:
        """Compute dense token-level rewards (for VAPO)."""
        batch_size = final_rewards.size(0)
        token_rewards = torch.zeros(batch_size, seq_len, device=self.config.device)

        if self.config.use_vapo and self.config.use_dense_rewards:
            # Distribute rewards with exponential decay from end
            decay = 0.9
            for t in range(seq_len - 1, -1, -1):
                if t == seq_len - 1:
                    token_rewards[:, t] = final_rewards
                else:
                    token_rewards[:, t] = decay * token_rewards[:, t + 1]

            # Smooth rewards
            if self.config.reward_smoothing > 0:
                smoothed = torch.zeros_like(token_rewards)
                window = 5
                for t in range(seq_len):
                    start = max(0, t - window // 2)
                    end = min(seq_len, t + window // 2 + 1)
                    smoothed[:, t] = token_rewards[:, start:end].mean(dim=1)
                token_rewards = (
                    self.config.reward_smoothing * smoothed
                    + (1 - self.config.reward_smoothing) * token_rewards
                )
        else:
            # Sparse reward: only at the end
            token_rewards[:, -1] = final_rewards

        return token_rewards


class VAPOTrainer(DAPOTrainer):
    """
    VAPO: Value-model Augmented Proximal Policy Optimization

    Extends DAPO with:
    1. Dedicated value model for dense reward signals
    2. Long chain-of-thought optimization
    3. Better credit assignment through value bootstrapping
    """

    def __init__(
        self,
        model: SPARSALM,
        ref_model: SPARSALM,
        tokenizer,
        config: RLAIFConfig,
        reward_model: Optional[RewardModel] = None,
    ):
        # Create value model
        value_config = SPARSAConfig(
            vocab_size=model.config.vocab_size,
            hidden_dim=model.config.hidden_dim,
            num_layers=model.config.num_layers // 2,  # Smaller for efficiency
            num_heads=model.config.num_heads,
            num_kv_heads=model.config.num_kv_heads,
            ff_dim=model.config.ff_dim,
            max_seq_len=model.config.max_seq_len,
        )
        value_base = SPARSALM(value_config)
        value_model = ValueModel(value_base).to(config.device)

        super().__init__(
            model=model,
            ref_model=ref_model,
            tokenizer=tokenizer,
            config=config,
            reward_model=reward_model,
            value_model=value_model,
        )

        # Separate optimizer for value model
        self.value_optimizer = AdamW(
            self.value_model.parameters(),
            lr=config.learning_rate * 3,  # Higher LR for value model
            betas=(0.9, 0.95),
            weight_decay=0.01,
        )

    def train_step(self, prompts: List[str]) -> Dict[str, float]:
        """VAPO training step with value model updates."""
        metrics = super().train_step(prompts)

        # Additional value model training
        if self.value_model is not None:
            self.value_model.train()

            # Generate additional samples for value learning
            _, response_ids, _ = self.generate_responses(prompts)
            attention_mask = (response_ids != self.tokenizer.pad_token_id).long()

            values = self.value_model(response_ids, attention_mask)

            # Value target from reward model
            with torch.no_grad():
                rewards = self.ai_reward.compute_reward(
                    prompts,
                    self.tokenizer.batch_decode(response_ids, skip_special_tokens=True),
                )
                rewards = rewards.to(self.config.device)
                rewards = (rewards - self.running_reward_mean) / (self.running_reward_std + 1e-8)

            # Compute value loss
            target_values = rewards.unsqueeze(1).expand(-1, values.size(1))
            value_loss = F.mse_loss(values * attention_mask, target_values * attention_mask)

            self.value_optimizer.zero_grad()
            value_loss.backward()
            self.value_optimizer.step()

            metrics["vapo_value_loss"] = value_loss.item()

        return metrics


class RLAIFPipeline:
    """
    Complete RLAIF fine-tuning pipeline.

    Supports:
    - DAPO (Decoupled Clip and Dynamic Sampling)
    - VAPO (Value-model Augmented PPO)
    - AI Feedback reward models
    """

    def __init__(
        self,
        model_path: str,
        tokenizer_path: str,
        config: Optional[RLAIFConfig] = None,
        output_dir: str = "checkpoints/rlaif",
    ):
        self.config = config or RLAIFConfig()
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Load model and tokenizer
        self.model = SPARSALM.from_pretrained(model_path, device=self.config.device)
        self.ref_model = SPARSALM.from_pretrained(model_path, device=self.config.device)

        if HF_AVAILABLE:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        else:
            raise ImportError("HuggingFace transformers required for tokenizer")

        # Initialize trainer based on config
        if self.config.use_vapo:
            self.trainer = VAPOTrainer(
                model=self.model,
                ref_model=self.ref_model,
                tokenizer=self.tokenizer,
                config=self.config,
            )
            logging.info("Initialized VAPO trainer")
        else:
            self.trainer = DAPOTrainer(
                model=self.model,
                ref_model=self.ref_model,
                tokenizer=self.tokenizer,
                config=self.config,
            )
            logging.info("Initialized DAPO trainer")

    def train(
        self,
        train_prompts: List[str],
        eval_prompts: Optional[List[str]] = None,
    ):
        """Run RLAIF training."""
        logging.info(f"Starting RLAIF training with {len(train_prompts)} prompts")

        for epoch in range(self.config.num_epochs):
            epoch_metrics = []

            # Shuffle prompts
            import random
            random.shuffle(train_prompts)

            # Apply dynamic sampling filter
            if hasattr(self.trainer, 'dynamic_sampling_filter'):
                filtered_prompts = self.trainer.dynamic_sampling_filter(
                    train_prompts,
                    self.trainer.prompt_stats,
                )
            else:
                filtered_prompts = train_prompts

            # Training loop
            for i in range(0, len(filtered_prompts), self.config.batch_size):
                batch_prompts = filtered_prompts[i:i + self.config.batch_size]
                if len(batch_prompts) < self.config.batch_size:
                    continue

                metrics = self.trainer.train_step(batch_prompts)
                epoch_metrics.append(metrics)

                if (i // self.config.batch_size) % 10 == 0:
                    avg_loss = sum(m["loss"] for m in epoch_metrics[-10:]) / min(10, len(epoch_metrics))
                    avg_reward = sum(m["mean_reward"] for m in epoch_metrics[-10:]) / min(10, len(epoch_metrics))
                    logging.info(
                        f"Epoch {epoch + 1}, Step {i // self.config.batch_size}: "
                        f"Loss={avg_loss:.4f}, Reward={avg_reward:.4f}"
                    )

            # Epoch summary
            avg_metrics = {
                k: sum(m[k] for m in epoch_metrics) / len(epoch_metrics)
                for k in epoch_metrics[0].keys()
            }
            logging.info(f"Epoch {epoch + 1} complete: {avg_metrics}")

            # Save checkpoint
            self.save_checkpoint(epoch)

            # Evaluation
            if eval_prompts:
                eval_metrics = self.evaluate(eval_prompts)
                logging.info(f"Evaluation metrics: {eval_metrics}")

    def evaluate(self, prompts: List[str]) -> Dict[str, float]:
        """Evaluate the model on a set of prompts."""
        self.model.eval()

        all_rewards = []
        for i in range(0, len(prompts), self.config.batch_size):
            batch_prompts = prompts[i:i + self.config.batch_size]

            with torch.no_grad():
                _, _, responses = self.trainer.generate_responses(batch_prompts)

            rewards = self.trainer.ai_reward.compute_reward(batch_prompts, responses)
            all_rewards.extend(rewards.tolist())

        return {
            "mean_reward": sum(all_rewards) / len(all_rewards),
            "max_reward": max(all_rewards),
            "min_reward": min(all_rewards),
        }

    def save_checkpoint(self, epoch: int):
        """Save model checkpoint."""
        checkpoint_path = os.path.join(self.output_dir, f"checkpoint-epoch-{epoch}")
        self.model.save_pretrained(checkpoint_path)

        # Save training state
        state = {
            "epoch": epoch,
            "kl_coef": self.trainer.kl_coef,
            "running_reward_mean": self.trainer.running_reward_mean,
            "running_reward_std": self.trainer.running_reward_std,
        }
        with open(os.path.join(checkpoint_path, "training_state.json"), "w") as f:
            json.dump(state, f)

        logging.info(f"Saved checkpoint to {checkpoint_path}")

    def push_to_hub(self, repo_id: str, token: Optional[str] = None):
        """Push model to HuggingFace Hub."""
        if not HF_AVAILABLE:
            raise ImportError("HuggingFace transformers required for Hub upload")

        from huggingface_hub import HfApi

        api = HfApi()
        api.upload_folder(
            folder_path=self.output_dir,
            repo_id=repo_id,
            repo_type="model",
            token=token,
        )
        logging.info(f"Pushed model to {repo_id}")


def main():
    """Example usage of RLAIF pipeline."""
    import argparse

    parser = argparse.ArgumentParser(description="SPARSA-LM RLAIF Training")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--tokenizer_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="checkpoints/rlaif")
    parser.add_argument("--use_vapo", action="store_true")
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    args = parser.parse_args()

    # Configure
    config = RLAIFConfig(
        use_vapo=args.use_vapo,
        use_dapo=not args.use_vapo,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
    )

    # Initialize pipeline
    pipeline = RLAIFPipeline(
        model_path=args.model_path,
        tokenizer_path=args.tokenizer_path,
        config=config,
        output_dir=args.output_dir,
    )

    # Example training prompts
    train_prompts = [
        "Explain the concept of machine learning in simple terms.",
        "Write a Python function to calculate fibonacci numbers.",
        "What are the health benefits of regular exercise?",
        "Solve: If x + 5 = 12, what is x?",
        "Describe the process of photosynthesis.",
    ]

    # Train
    pipeline.train(train_prompts)


if __name__ == "__main__":
    main()
