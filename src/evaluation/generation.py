"""
SPARSA-LM Generation Utilities
Text generation for evaluation and inference
"""

import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Union, Tuple

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


@dataclass
class GenerationConfig:
    """Configuration for text generation."""

    # Length control
    max_new_tokens: int = 256
    min_new_tokens: int = 0

    # Sampling parameters
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 0.9
    typical_p: float = 1.0

    # Repetition control
    repetition_penalty: float = 1.0
    no_repeat_ngram_size: int = 0
    encoder_repetition_penalty: float = 1.0

    # Beam search (disabled by default)
    num_beams: int = 1
    num_beam_groups: int = 1
    diversity_penalty: float = 0.0
    length_penalty: float = 1.0
    early_stopping: bool = False

    # Sampling control
    do_sample: bool = True

    # Stop conditions
    eos_token_id: Optional[int] = None
    pad_token_id: Optional[int] = None
    stop_sequences: List[str] = field(default_factory=list)

    # Output control
    return_logprobs: bool = False
    return_dict_in_generate: bool = True

    # Batch generation
    num_return_sequences: int = 1

    @classmethod
    def for_greedy(cls) -> "GenerationConfig":
        """Configuration for greedy decoding."""
        return cls(
            temperature=1.0,
            do_sample=False,
            top_k=0,
            top_p=1.0,
        )

    @classmethod
    def for_sampling(
        cls,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> "GenerationConfig":
        """Configuration for sampling."""
        return cls(
            temperature=temperature,
            do_sample=True,
            top_k=0,
            top_p=top_p,
        )

    @classmethod
    def for_beam_search(
        cls,
        num_beams: int = 4,
    ) -> "GenerationConfig":
        """Configuration for beam search."""
        return cls(
            do_sample=False,
            num_beams=num_beams,
            early_stopping=True,
        )

    @classmethod
    def for_code_generation(cls) -> "GenerationConfig":
        """Configuration optimized for code generation."""
        return cls(
            temperature=0.2,
            do_sample=True,
            top_p=0.95,
            repetition_penalty=1.1,
            max_new_tokens=512,
        )


class TextGenerator:
    """
    Text generation utility for evaluation.

    Supports:
    - Various decoding strategies
    - Batched generation
    - Log probability tracking
    - Stop sequence handling
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        config: Optional[GenerationConfig] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config or GenerationConfig()

        # Setup special tokens
        if self.config.eos_token_id is None:
            self.config.eos_token_id = tokenizer.eos_token_id
        if self.config.pad_token_id is None:
            self.config.pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    @torch.no_grad()
    def generate(
        self,
        prompts: Union[str, List[str]],
        config: Optional[GenerationConfig] = None,
    ) -> List[Dict[str, Any]]:
        """
        Generate text from prompts.

        Args:
            prompts: Single prompt or list of prompts
            config: Optional override configuration

        Returns:
            List of generation results
        """
        if isinstance(prompts, str):
            prompts = [prompts]

        config = config or self.config

        # Tokenize prompts
        encodings = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            add_special_tokens=True,
        )

        input_ids = encodings["input_ids"].to(self.model.device)
        attention_mask = encodings["attention_mask"].to(self.model.device)

        # Generate
        if config.num_beams > 1:
            outputs = self._generate_beam_search(input_ids, attention_mask, config)
        elif config.do_sample:
            outputs = self._generate_sampling(input_ids, attention_mask, config)
        else:
            outputs = self._generate_greedy(input_ids, attention_mask, config)

        # Decode and format results
        results = []
        for i, prompt in enumerate(prompts):
            prompt_len = encodings["input_ids"][i].ne(self.config.pad_token_id).sum()

            batch_outputs = outputs[i * config.num_return_sequences : (i + 1) * config.num_return_sequences]

            for output in batch_outputs:
                generated_ids = output[prompt_len:]
                generated_text = self.tokenizer.decode(
                    generated_ids,
                    skip_special_tokens=True,
                )

                # Handle stop sequences
                for stop_seq in config.stop_sequences:
                    if stop_seq in generated_text:
                        generated_text = generated_text[:generated_text.index(stop_seq)]

                results.append({
                    "prompt": prompt,
                    "generated_text": generated_text,
                    "full_text": prompt + generated_text,
                    "generated_ids": generated_ids.tolist(),
                })

        return results

    def _generate_greedy(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        config: GenerationConfig,
    ) -> torch.Tensor:
        """Greedy decoding."""
        batch_size = input_ids.shape[0]
        generated = input_ids

        for _ in range(config.max_new_tokens):
            outputs = self.model(
                input_ids=generated,
                attention_mask=torch.ones_like(generated),
            )
            logits = outputs["logits"][:, -1, :]

            # Apply repetition penalty
            if config.repetition_penalty != 1.0:
                logits = self._apply_repetition_penalty(logits, generated, config.repetition_penalty)

            # Greedy selection
            next_tokens = logits.argmax(dim=-1, keepdim=True)

            # Append
            generated = torch.cat([generated, next_tokens], dim=-1)

            # Check for EOS
            if (next_tokens == config.eos_token_id).all():
                break

        return generated

    def _generate_sampling(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        config: GenerationConfig,
    ) -> torch.Tensor:
        """Sampling-based generation."""
        batch_size = input_ids.shape[0]
        device = input_ids.device

        # Expand for multiple return sequences
        if config.num_return_sequences > 1:
            input_ids = input_ids.repeat_interleave(config.num_return_sequences, dim=0)
            attention_mask = attention_mask.repeat_interleave(config.num_return_sequences, dim=0)
            batch_size *= config.num_return_sequences

        generated = input_ids
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        past_key_values = None

        for _ in range(config.max_new_tokens):
            # Forward pass
            outputs = self.model(
                input_ids=generated if past_key_values is None else generated[:, -1:],
                attention_mask=attention_mask if past_key_values is None else None,
                past_key_values=past_key_values,
                use_cache=True,
            )

            logits = outputs["logits"][:, -1, :]
            past_key_values = outputs.get("past_key_values")

            # Apply repetition penalty
            if config.repetition_penalty != 1.0:
                logits = self._apply_repetition_penalty(logits, generated, config.repetition_penalty)

            # Apply temperature
            if config.temperature != 1.0:
                logits = logits / config.temperature

            # Apply top-k filtering
            if config.top_k > 0:
                logits = self._apply_top_k(logits, config.top_k)

            # Apply top-p (nucleus) filtering
            if config.top_p < 1.0:
                logits = self._apply_top_p(logits, config.top_p)

            # Apply typical sampling
            if config.typical_p < 1.0:
                logits = self._apply_typical_p(logits, config.typical_p)

            # Sample
            probs = F.softmax(logits, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1)

            # Replace finished sequences with pad
            next_tokens = torch.where(
                finished.unsqueeze(-1),
                torch.full_like(next_tokens, config.pad_token_id),
                next_tokens,
            )

            # Append
            generated = torch.cat([generated, next_tokens], dim=-1)

            # Update finished
            finished = finished | (next_tokens.squeeze(-1) == config.eos_token_id)

            if finished.all():
                break

        return generated

    def _generate_beam_search(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        config: GenerationConfig,
    ) -> torch.Tensor:
        """Beam search generation."""
        batch_size = input_ids.shape[0]
        device = input_ids.device
        num_beams = config.num_beams

        # Expand inputs for beam search
        input_ids = input_ids.unsqueeze(1).expand(-1, num_beams, -1).reshape(batch_size * num_beams, -1)

        # Initialize beam scores
        beam_scores = torch.zeros((batch_size, num_beams), device=device)
        beam_scores[:, 1:] = -float('inf')  # Only first beam is active initially
        beam_scores = beam_scores.view(-1)

        generated = input_ids
        done = torch.zeros(batch_size, dtype=torch.bool, device=device)

        for step in range(config.max_new_tokens):
            outputs = self.model(
                input_ids=generated,
                attention_mask=torch.ones_like(generated),
            )
            logits = outputs["logits"][:, -1, :]

            # Apply length penalty
            if config.length_penalty != 1.0:
                length = step + 1
                logits = logits / (length ** config.length_penalty)

            vocab_size = logits.shape[-1]
            log_probs = F.log_softmax(logits, dim=-1)

            # Add to beam scores
            next_scores = beam_scores.unsqueeze(-1) + log_probs
            next_scores = next_scores.view(batch_size, -1)

            # Get top-k beams
            next_scores, next_tokens = torch.topk(
                next_scores, 2 * num_beams, dim=1, largest=True, sorted=True
            )

            next_beam_indices = next_tokens // vocab_size
            next_tokens = next_tokens % vocab_size

            # Select top beams
            beam_scores = next_scores[:, :num_beams].view(-1)
            beam_tokens = next_tokens[:, :num_beams].view(-1, 1)
            beam_indices = next_beam_indices[:, :num_beams].view(-1)

            # Reorder generated sequences
            batch_beam_indices = beam_indices + torch.arange(
                batch_size, device=device
            ).unsqueeze(1).repeat(1, num_beams).view(-1) * num_beams

            generated = torch.cat([
                generated[batch_beam_indices],
                beam_tokens,
            ], dim=-1)

            # Check for completion
            if config.early_stopping:
                is_eos = beam_tokens.squeeze(-1) == config.eos_token_id
                done = done | is_eos.view(batch_size, num_beams).any(dim=1)

                if done.all():
                    break

        # Return best beam for each batch item
        generated = generated.view(batch_size, num_beams, -1)
        best_beams = beam_scores.view(batch_size, num_beams).argmax(dim=1)
        best_sequences = generated[torch.arange(batch_size), best_beams]

        return best_sequences

    def _apply_repetition_penalty(
        self,
        logits: torch.Tensor,
        generated: torch.Tensor,
        penalty: float,
    ) -> torch.Tensor:
        """Apply repetition penalty to logits."""
        for i in range(logits.shape[0]):
            for token_id in set(generated[i].tolist()):
                if logits[i, token_id] > 0:
                    logits[i, token_id] /= penalty
                else:
                    logits[i, token_id] *= penalty
        return logits

    def _apply_top_k(
        self,
        logits: torch.Tensor,
        k: int,
    ) -> torch.Tensor:
        """Apply top-k filtering."""
        indices_to_remove = logits < torch.topk(logits, k)[0][..., -1, None]
        logits = logits.masked_fill(indices_to_remove, float('-inf'))
        return logits

    def _apply_top_p(
        self,
        logits: torch.Tensor,
        p: float,
    ) -> torch.Tensor:
        """Apply nucleus (top-p) filtering."""
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above threshold
        sorted_indices_to_remove = cumulative_probs > p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = False

        # Scatter back
        indices_to_remove = sorted_indices_to_remove.scatter(
            1, sorted_indices, sorted_indices_to_remove
        )
        logits = logits.masked_fill(indices_to_remove, float('-inf'))

        return logits

    def _apply_typical_p(
        self,
        logits: torch.Tensor,
        p: float,
    ) -> torch.Tensor:
        """Apply typical sampling."""
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)

        # Compute entropy
        entropy = -(probs * log_probs).sum(dim=-1, keepdim=True)

        # Compute deviation from entropy
        deviation = torch.abs(-log_probs - entropy)

        # Sort by deviation
        sorted_deviation, sorted_indices = torch.sort(deviation, descending=False)
        sorted_probs = probs.gather(-1, sorted_indices)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

        # Remove tokens with high deviation
        sorted_indices_to_remove = cumulative_probs > p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = False

        indices_to_remove = sorted_indices_to_remove.scatter(
            1, sorted_indices, sorted_indices_to_remove
        )
        logits = logits.masked_fill(indices_to_remove, float('-inf'))

        return logits


def generate_samples(
    model: Any,
    tokenizer: Any,
    prompts: Union[str, List[str]],
    config: Optional[GenerationConfig] = None,
    **kwargs,
) -> List[Dict[str, Any]]:
    """
    Generate samples from prompts.

    Args:
        model: Language model
        tokenizer: Tokenizer
        prompts: Input prompts
        config: Generation configuration
        **kwargs: Override config parameters

    Returns:
        List of generation results
    """
    if config is None:
        config = GenerationConfig()

    # Apply kwargs
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)

    generator = TextGenerator(model, tokenizer, config)
    return generator.generate(prompts)


def batch_generate(
    model: Any,
    tokenizer: Any,
    prompts: List[str],
    batch_size: int = 8,
    config: Optional[GenerationConfig] = None,
    show_progress: bool = True,
) -> List[Dict[str, Any]]:
    """
    Generate samples in batches.

    Args:
        model: Language model
        tokenizer: Tokenizer
        prompts: List of prompts
        batch_size: Batch size for generation
        config: Generation configuration
        show_progress: Show progress bar

    Returns:
        List of generation results
    """
    generator = TextGenerator(model, tokenizer, config)
    results = []

    if show_progress:
        try:
            from tqdm import tqdm
            iterator = tqdm(range(0, len(prompts), batch_size), desc="Generating")
        except ImportError:
            iterator = range(0, len(prompts), batch_size)
    else:
        iterator = range(0, len(prompts), batch_size)

    for i in iterator:
        batch_prompts = prompts[i:i + batch_size]
        batch_results = generator.generate(batch_prompts)
        results.extend(batch_results)

    return results
