"""
SPARSA-LM AutoRegressive Language Model
Main model architecture implementation
"""

from typing import Optional, Tuple, List, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .config import ModelConfig
from .layers import RMSNorm, TransformerBlock


class AutoRegressiveLM(nn.Module):
    """
    AutoRegressive Language Model.

    A decoder-only transformer model for causal language modeling
    with modern architectural improvements including:
    - RMSNorm for efficient normalization
    - Rotary Position Embeddings (RoPE)
    - Grouped Query Attention (GQA)
    - SwiGLU activation
    - Flash Attention support
    - KV-Cache for efficient generation
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # Token embeddings
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)

        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerBlock(config, layer_idx)
            for layer_idx in range(config.num_hidden_layers)
        ])

        # Final layer norm
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # Language model head
        if config.tie_word_embeddings:
            self.lm_head = None
        else:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights
        self.apply(self._init_weights)

        # Gradient checkpointing flag
        self.gradient_checkpointing = config.use_gradient_checkpointing

    def _init_weights(self, module: nn.Module):
        """Initialize model weights."""
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    def get_input_embeddings(self) -> nn.Embedding:
        """Get input embedding layer."""
        return self.embed_tokens

    def set_input_embeddings(self, value: nn.Embedding):
        """Set input embedding layer."""
        self.embed_tokens = value

    def get_output_embeddings(self) -> nn.Linear:
        """Get output embedding layer (LM head)."""
        if self.lm_head is not None:
            return self.lm_head
        return self.embed_tokens

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
        return_hidden_states: bool = False,
        labels: Optional[torch.Tensor] = None,
    ) -> dict:
        """
        Forward pass of the model.

        Args:
            input_ids: Input token IDs of shape (batch_size, seq_len)
            attention_mask: Optional attention mask
            position_ids: Optional position IDs
            past_key_values: Optional cached key-value pairs for each layer
            use_cache: Whether to return updated cache
            return_hidden_states: Whether to return all hidden states
            labels: Optional labels for language modeling loss

        Returns:
            Dictionary containing:
                - logits: Output logits of shape (batch_size, seq_len, vocab_size)
                - loss: Optional language modeling loss
                - past_key_values: Optional updated cache
                - hidden_states: Optional list of hidden states
        """
        batch_size, seq_len = input_ids.shape

        # Get embeddings
        hidden_states = self.embed_tokens(input_ids)

        # Prepare position IDs
        if position_ids is None:
            past_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0
            position_ids = torch.arange(
                past_length, past_length + seq_len,
                device=input_ids.device
            ).unsqueeze(0).expand(batch_size, -1)

        # Prepare attention mask
        if attention_mask is not None:
            attention_mask = self._prepare_attention_mask(attention_mask, seq_len, hidden_states.dtype)

        # Store hidden states if requested
        all_hidden_states = [] if return_hidden_states else None

        # Forward through transformer layers
        present_key_values = [] if use_cache else None

        for idx, layer in enumerate(self.layers):
            if return_hidden_states:
                all_hidden_states.append(hidden_states)

            past_key_value = past_key_values[idx] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:
                hidden_states, present_key_value = checkpoint(
                    layer,
                    hidden_states,
                    attention_mask,
                    position_ids,
                    past_key_value,
                    use_cache,
                    use_reentrant=False,
                )
            else:
                hidden_states, present_key_value = layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    use_cache=use_cache,
                )

            if use_cache:
                present_key_values.append(present_key_value)

        # Final layer norm
        hidden_states = self.norm(hidden_states)

        if return_hidden_states:
            all_hidden_states.append(hidden_states)

        # Compute logits
        if self.lm_head is not None:
            logits = self.lm_head(hidden_states)
        else:
            logits = F.linear(hidden_states, self.embed_tokens.weight)

        # Compute loss if labels provided
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        return {
            "logits": logits,
            "loss": loss,
            "past_key_values": present_key_values,
            "hidden_states": all_hidden_states,
        }

    def _prepare_attention_mask(
        self,
        attention_mask: torch.Tensor,
        seq_len: int,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Prepare attention mask for scaled dot-product attention."""
        # Expand mask from (batch, seq) to (batch, 1, 1, seq)
        expanded_mask = attention_mask[:, None, None, :].expand(-1, 1, seq_len, -1)
        # Invert: 0 -> -inf, 1 -> 0
        inverted_mask = (1.0 - expanded_mask.to(dtype)) * torch.finfo(dtype).min
        return inverted_mask

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
        repetition_penalty: float = 1.0,
        do_sample: bool = True,
        eos_token_id: Optional[int] = None,
        pad_token_id: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Generate text autoregressively.

        Args:
            input_ids: Input token IDs
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_k: Top-k filtering parameter
            top_p: Nucleus sampling parameter
            repetition_penalty: Penalty for repeated tokens
            do_sample: Whether to sample or use greedy decoding
            eos_token_id: End of sequence token ID
            pad_token_id: Padding token ID

        Returns:
            Generated token IDs
        """
        eos_token_id = eos_token_id or self.config.eos_token_id
        pad_token_id = pad_token_id or self.config.pad_token_id

        batch_size = input_ids.shape[0]
        past_key_values = None
        generated_ids = input_ids.clone()

        # Track which sequences are finished
        finished = torch.zeros(batch_size, dtype=torch.bool, device=input_ids.device)

        for _ in range(max_new_tokens):
            # Forward pass
            outputs = self.forward(
                input_ids=generated_ids if past_key_values is None else generated_ids[:, -1:],
                past_key_values=past_key_values,
                use_cache=True,
            )

            logits = outputs["logits"][:, -1, :]
            past_key_values = outputs["past_key_values"]

            # Apply repetition penalty
            if repetition_penalty != 1.0:
                for i in range(batch_size):
                    for token_id in set(generated_ids[i].tolist()):
                        if logits[i, token_id] > 0:
                            logits[i, token_id] /= repetition_penalty
                        else:
                            logits[i, token_id] *= repetition_penalty

            # Apply temperature
            if temperature != 1.0:
                logits = logits / temperature

            if do_sample:
                # Top-k filtering
                if top_k > 0:
                    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                    logits = logits.masked_fill(indices_to_remove, float('-inf'))

                # Top-p (nucleus) filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    logits = logits.masked_fill(indices_to_remove, float('-inf'))

                # Sample
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                # Greedy decoding
                next_token = logits.argmax(dim=-1, keepdim=True)

            # Update finished sequences
            finished = finished | (next_token.squeeze(-1) == eos_token_id)

            # Replace tokens in finished sequences with pad
            next_token = torch.where(
                finished.unsqueeze(-1),
                torch.full_like(next_token, pad_token_id),
                next_token,
            )

            generated_ids = torch.cat([generated_ids, next_token], dim=-1)

            # Stop if all sequences are finished
            if finished.all():
                break

        return generated_ids

    def count_parameters(self, trainable_only: bool = True) -> int:
        """Count the number of parameters in the model."""
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())

    def save_pretrained(self, path: str):
        """Save model and configuration."""
        import os
        os.makedirs(path, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(path, "model.pt"))
        self.config.save(os.path.join(path, "config.json"))

    @classmethod
    def from_pretrained(cls, path: str, device: str = "cpu") -> "AutoRegressiveLM":
        """Load model from pretrained checkpoint."""
        import os
        config = ModelConfig.load(os.path.join(path, "config.json"))
        model = cls(config)
        state_dict = torch.load(os.path.join(path, "model.pt"), map_location=device)
        model.load_state_dict(state_dict)
        return model
