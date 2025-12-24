"""
SPARSA-LM Text Generation Utilities
Advanced generation strategies and configurations
"""

from dataclasses import dataclass, field
from typing import Optional, List, Callable, Union
import torch
import torch.nn.functional as F


@dataclass
class GenerationConfig:
    """
    Configuration for text generation.

    Controls all aspects of the generation process including
    sampling strategies, stopping criteria, and output formatting.
    """

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

    # Beam search (if not sampling)
    num_beams: int = 1
    num_beam_groups: int = 1
    diversity_penalty: float = 0.0
    length_penalty: float = 1.0
    early_stopping: bool = False

    # Sampling mode
    do_sample: bool = True

    # Special tokens
    pad_token_id: Optional[int] = 0
    bos_token_id: Optional[int] = 1
    eos_token_id: Optional[int] = 2

    # Output control
    return_dict_in_generate: bool = False
    output_scores: bool = False
    output_logits: bool = False

    def validate(self):
        """Validate configuration parameters."""
        if self.temperature <= 0:
            raise ValueError("temperature must be positive")
        if self.top_k < 0:
            raise ValueError("top_k must be non-negative")
        if not 0.0 < self.top_p <= 1.0:
            raise ValueError("top_p must be in (0, 1]")
        if self.repetition_penalty < 1.0:
            raise ValueError("repetition_penalty must be >= 1.0")
        if self.num_beams < 1:
            raise ValueError("num_beams must be >= 1")


class GenerationMixin:
    """
    Mixin class providing advanced generation capabilities.

    Can be inherited by model classes to add sophisticated
    generation methods beyond basic autoregressive sampling.
    """

    @torch.no_grad()
    def generate_advanced(
        self,
        input_ids: torch.Tensor,
        generation_config: Optional[GenerationConfig] = None,
        stopping_criteria: Optional[List[Callable]] = None,
        logits_processor: Optional[List[Callable]] = None,
        streamer: Optional[Callable] = None,
        **kwargs,
    ) -> Union[torch.Tensor, dict]:
        """
        Advanced generation with full configurability.

        Args:
            input_ids: Input token IDs
            generation_config: Generation configuration
            stopping_criteria: List of stopping criteria functions
            logits_processor: List of logits processing functions
            streamer: Optional streaming callback
            **kwargs: Override generation config parameters

        Returns:
            Generated token IDs or dictionary with additional outputs
        """
        # Merge config with kwargs
        config = generation_config or GenerationConfig()
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
        config.validate()

        # Route to appropriate generation method
        if config.num_beams > 1:
            return self._beam_search(input_ids, config, stopping_criteria, logits_processor)
        elif config.do_sample:
            return self._sample(input_ids, config, stopping_criteria, logits_processor, streamer)
        else:
            return self._greedy_decode(input_ids, config, stopping_criteria, logits_processor, streamer)

    def _sample(
        self,
        input_ids: torch.Tensor,
        config: GenerationConfig,
        stopping_criteria: Optional[List[Callable]] = None,
        logits_processor: Optional[List[Callable]] = None,
        streamer: Optional[Callable] = None,
    ) -> torch.Tensor:
        """Sampling-based generation."""
        batch_size = input_ids.shape[0]
        device = input_ids.device

        generated_ids = input_ids.clone()
        past_key_values = None
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        all_scores = [] if config.output_scores else None

        for step in range(config.max_new_tokens):
            # Forward pass
            outputs = self.forward(
                input_ids=generated_ids if past_key_values is None else generated_ids[:, -1:],
                past_key_values=past_key_values,
                use_cache=True,
            )

            logits = outputs["logits"][:, -1, :]
            past_key_values = outputs["past_key_values"]

            # Apply logits processors
            if logits_processor:
                for processor in logits_processor:
                    logits = processor(generated_ids, logits)

            # Apply repetition penalty
            if config.repetition_penalty != 1.0:
                logits = self._apply_repetition_penalty(logits, generated_ids, config.repetition_penalty)

            # Apply no-repeat n-gram
            if config.no_repeat_ngram_size > 0:
                logits = self._apply_no_repeat_ngram(logits, generated_ids, config.no_repeat_ngram_size)

            # Apply temperature
            if config.temperature != 1.0:
                logits = logits / config.temperature

            # Apply top-k
            if config.top_k > 0:
                logits = self._top_k_filtering(logits, config.top_k)

            # Apply top-p
            if config.top_p < 1.0:
                logits = self._top_p_filtering(logits, config.top_p)

            # Apply typical-p
            if config.typical_p < 1.0:
                logits = self._typical_filtering(logits, config.typical_p)

            # Store scores
            if all_scores is not None:
                all_scores.append(logits.clone())

            # Sample
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # Stream token
            if streamer is not None:
                streamer(next_token)

            # Update finished
            finished = finished | (next_token.squeeze(-1) == config.eos_token_id)

            # Replace finished with pad
            next_token = torch.where(
                finished.unsqueeze(-1),
                torch.full_like(next_token, config.pad_token_id),
                next_token,
            )

            generated_ids = torch.cat([generated_ids, next_token], dim=-1)

            # Check stopping criteria
            if stopping_criteria:
                should_stop = any(criterion(generated_ids, None) for criterion in stopping_criteria)
                if should_stop:
                    break

            # Check if all finished
            if finished.all():
                break

            # Check min length
            if step < config.min_new_tokens:
                continue

        if config.return_dict_in_generate:
            return {
                "sequences": generated_ids,
                "scores": all_scores,
            }
        return generated_ids

    def _greedy_decode(
        self,
        input_ids: torch.Tensor,
        config: GenerationConfig,
        stopping_criteria: Optional[List[Callable]] = None,
        logits_processor: Optional[List[Callable]] = None,
        streamer: Optional[Callable] = None,
    ) -> torch.Tensor:
        """Greedy decoding generation."""
        batch_size = input_ids.shape[0]
        device = input_ids.device

        generated_ids = input_ids.clone()
        past_key_values = None
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        for step in range(config.max_new_tokens):
            outputs = self.forward(
                input_ids=generated_ids if past_key_values is None else generated_ids[:, -1:],
                past_key_values=past_key_values,
                use_cache=True,
            )

            logits = outputs["logits"][:, -1, :]
            past_key_values = outputs["past_key_values"]

            if logits_processor:
                for processor in logits_processor:
                    logits = processor(generated_ids, logits)

            if config.repetition_penalty != 1.0:
                logits = self._apply_repetition_penalty(logits, generated_ids, config.repetition_penalty)

            next_token = logits.argmax(dim=-1, keepdim=True)

            if streamer is not None:
                streamer(next_token)

            finished = finished | (next_token.squeeze(-1) == config.eos_token_id)

            next_token = torch.where(
                finished.unsqueeze(-1),
                torch.full_like(next_token, config.pad_token_id),
                next_token,
            )

            generated_ids = torch.cat([generated_ids, next_token], dim=-1)

            if stopping_criteria:
                should_stop = any(criterion(generated_ids, None) for criterion in stopping_criteria)
                if should_stop:
                    break

            if finished.all():
                break

        return generated_ids

    def _beam_search(
        self,
        input_ids: torch.Tensor,
        config: GenerationConfig,
        stopping_criteria: Optional[List[Callable]] = None,
        logits_processor: Optional[List[Callable]] = None,
    ) -> torch.Tensor:
        """Beam search generation."""
        batch_size = input_ids.shape[0]
        device = input_ids.device
        num_beams = config.num_beams

        # Expand input for beams
        input_ids = input_ids.unsqueeze(1).expand(-1, num_beams, -1)
        input_ids = input_ids.reshape(batch_size * num_beams, -1)

        # Initialize beam scores
        beam_scores = torch.zeros(batch_size, num_beams, device=device)
        beam_scores[:, 1:] = -1e9  # Only first beam active initially

        generated_ids = input_ids.clone()
        past_key_values = None

        for step in range(config.max_new_tokens):
            outputs = self.forward(
                input_ids=generated_ids if past_key_values is None else generated_ids[:, -1:],
                past_key_values=past_key_values,
                use_cache=True,
            )

            logits = outputs["logits"][:, -1, :]
            past_key_values = outputs["past_key_values"]

            if logits_processor:
                for processor in logits_processor:
                    logits = processor(generated_ids, logits)

            vocab_size = logits.shape[-1]
            log_probs = F.log_softmax(logits, dim=-1)

            # Add beam scores
            next_scores = log_probs + beam_scores.view(-1, 1)
            next_scores = next_scores.view(batch_size, num_beams * vocab_size)

            # Get top-k for each batch
            next_scores, next_tokens = torch.topk(next_scores, num_beams, dim=-1)

            # Compute beam and token indices
            beam_indices = next_tokens // vocab_size
            token_indices = next_tokens % vocab_size

            # Update beam scores
            beam_scores = next_scores

            # Reorder past key values
            beam_idx = beam_indices + torch.arange(batch_size, device=device).unsqueeze(-1) * num_beams
            beam_idx = beam_idx.view(-1)

            if past_key_values is not None:
                past_key_values = [
                    (kv[0].index_select(0, beam_idx), kv[1].index_select(0, beam_idx))
                    for kv in past_key_values
                ]

            # Reorder generated_ids and append new tokens
            generated_ids = generated_ids.index_select(0, beam_idx)
            generated_ids = torch.cat([generated_ids, token_indices.view(-1, 1)], dim=-1)

            # Check for EOS
            eos_mask = token_indices == config.eos_token_id
            if eos_mask.all():
                break

            # Apply length penalty
            if config.length_penalty != 1.0:
                length = generated_ids.shape[1]
                beam_scores = beam_scores / (length ** config.length_penalty)

        # Select best beam for each batch
        best_beams = beam_scores.argmax(dim=-1)
        batch_indices = torch.arange(batch_size, device=device) * num_beams + best_beams
        generated_ids = generated_ids.index_select(0, batch_indices)

        return generated_ids

    def _apply_repetition_penalty(
        self,
        logits: torch.Tensor,
        input_ids: torch.Tensor,
        penalty: float,
    ) -> torch.Tensor:
        """Apply repetition penalty to logits."""
        score = torch.gather(logits, 1, input_ids)
        score = torch.where(score < 0, score * penalty, score / penalty)
        logits.scatter_(1, input_ids, score)
        return logits

    def _apply_no_repeat_ngram(
        self,
        logits: torch.Tensor,
        input_ids: torch.Tensor,
        ngram_size: int,
    ) -> torch.Tensor:
        """Prevent repeated n-grams."""
        batch_size, seq_len = input_ids.shape

        if seq_len < ngram_size:
            return logits

        for batch_idx in range(batch_size):
            generated = input_ids[batch_idx].tolist()
            ngrams = set()

            for i in range(len(generated) - ngram_size + 1):
                ngram = tuple(generated[i:i + ngram_size])
                ngrams.add(ngram)

            # Ban tokens that would complete a repeated n-gram
            prefix = tuple(generated[-(ngram_size - 1):])
            for ngram in ngrams:
                if ngram[:-1] == prefix:
                    logits[batch_idx, ngram[-1]] = float('-inf')

        return logits

    def _top_k_filtering(self, logits: torch.Tensor, top_k: int) -> torch.Tensor:
        """Apply top-k filtering."""
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        return logits.masked_fill(indices_to_remove, float('-inf'))

    def _top_p_filtering(self, logits: torch.Tensor, top_p: float) -> torch.Tensor:
        """Apply nucleus (top-p) filtering."""
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        return logits.masked_fill(indices_to_remove, float('-inf'))

    def _typical_filtering(self, logits: torch.Tensor, typical_p: float) -> torch.Tensor:
        """Apply typical sampling filtering."""
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)

        # Compute entropy
        entropy = -torch.sum(probs * log_probs, dim=-1, keepdim=True)

        # Compute deviation from entropy
        deviation = torch.abs(-log_probs - entropy)

        # Sort by deviation
        sorted_deviation, sorted_indices = torch.sort(deviation, dim=-1)
        sorted_probs = probs.gather(-1, sorted_indices)

        # Compute cumulative probability
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

        # Create mask
        sorted_indices_to_remove = cumulative_probs > typical_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        return logits.masked_fill(indices_to_remove, float('-inf'))


class StoppingCriteria:
    """Base class for stopping criteria."""

    def __call__(self, input_ids: torch.Tensor, scores: Optional[torch.Tensor]) -> bool:
        raise NotImplementedError


class MaxLengthCriteria(StoppingCriteria):
    """Stop when reaching maximum length."""

    def __init__(self, max_length: int):
        self.max_length = max_length

    def __call__(self, input_ids: torch.Tensor, scores: Optional[torch.Tensor]) -> bool:
        return input_ids.shape[1] >= self.max_length


class EosTokenCriteria(StoppingCriteria):
    """Stop when all sequences have generated EOS."""

    def __init__(self, eos_token_id: int):
        self.eos_token_id = eos_token_id

    def __call__(self, input_ids: torch.Tensor, scores: Optional[torch.Tensor]) -> bool:
        return (input_ids[:, -1] == self.eos_token_id).all()


class StopStringCriteria(StoppingCriteria):
    """Stop when a specific string is generated."""

    def __init__(self, tokenizer, stop_strings: List[str]):
        self.tokenizer = tokenizer
        self.stop_strings = stop_strings

    def __call__(self, input_ids: torch.Tensor, scores: Optional[torch.Tensor]) -> bool:
        for seq in input_ids:
            text = self.tokenizer.decode(seq)
            for stop_string in self.stop_strings:
                if stop_string in text:
                    return True
        return False
