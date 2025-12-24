"""
SPARSA-LM Data Collators
Batch collation utilities for different training scenarios
"""

from typing import List, Dict, Any, Optional
import torch


class DataCollator:
    """
    Base data collator for batching examples.

    Handles padding and stacking of tensors for batch processing.
    """

    def __init__(
        self,
        pad_token_id: int = 0,
        label_pad_token_id: int = -100,
        max_length: Optional[int] = None,
        padding_side: str = "right",
    ):
        self.pad_token_id = pad_token_id
        self.label_pad_token_id = label_pad_token_id
        self.max_length = max_length
        self.padding_side = padding_side

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Collate a list of features into a batch."""
        if not features:
            return {}

        # Get all keys from first example
        keys = features[0].keys()
        batch = {}

        for key in keys:
            values = [f[key] for f in features]

            # Handle tensor values
            if isinstance(values[0], torch.Tensor):
                # Check if all same length
                lengths = [v.shape[0] for v in values]
                max_len = max(lengths)

                if self.max_length is not None:
                    max_len = min(max_len, self.max_length)

                # Pad if necessary
                if not all(l == max_len for l in lengths):
                    padded_values = []
                    for v in values:
                        if len(v) < max_len:
                            pad_value = self.label_pad_token_id if "label" in key else self.pad_token_id
                            padding = torch.full((max_len - len(v),), pad_value, dtype=v.dtype)
                            if self.padding_side == "right":
                                v = torch.cat([v, padding])
                            else:
                                v = torch.cat([padding, v])
                        elif len(v) > max_len:
                            v = v[:max_len]
                        padded_values.append(v)
                    values = padded_values

                batch[key] = torch.stack(values)
            else:
                # Keep non-tensor values as list
                batch[key] = values

        return batch


class PretrainCollator(DataCollator):
    """
    Data collator for pretraining.

    Handles input_ids, labels, and attention masks for
    causal language modeling.
    """

    def __init__(
        self,
        pad_token_id: int = 0,
        label_pad_token_id: int = -100,
        max_length: Optional[int] = None,
        mlm: bool = False,
        mlm_probability: float = 0.15,
    ):
        super().__init__(pad_token_id, label_pad_token_id, max_length)
        self.mlm = mlm
        self.mlm_probability = mlm_probability

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Collate pretraining batch."""
        batch = super().__call__(features)

        # For causal LM, labels are shifted input_ids
        if "labels" not in batch and "input_ids" in batch:
            batch["labels"] = batch["input_ids"].clone()
            # Mask padding tokens in labels
            if "attention_mask" in batch:
                batch["labels"] = batch["labels"].masked_fill(
                    batch["attention_mask"] == 0,
                    self.label_pad_token_id
                )

        return batch


class FinetuneCollator(DataCollator):
    """
    Data collator for instruction finetuning.

    Handles instruction-response formatting with proper
    label masking for the prompt portion.
    """

    def __init__(
        self,
        pad_token_id: int = 0,
        label_pad_token_id: int = -100,
        max_length: Optional[int] = None,
        mask_prompt: bool = True,
    ):
        super().__init__(pad_token_id, label_pad_token_id, max_length)
        self.mask_prompt = mask_prompt

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Collate finetuning batch."""
        batch = super().__call__(features)

        # Mask prompt tokens in labels if requested
        if self.mask_prompt and "prompt_length" in features[0]:
            for i, feature in enumerate(features):
                prompt_len = feature["prompt_length"]
                batch["labels"][i, :prompt_len] = self.label_pad_token_id

        return batch


class RLCollator(DataCollator):
    """
    Data collator for reinforcement learning.

    Handles prompt batching for policy generation.
    """

    def __init__(
        self,
        pad_token_id: int = 0,
        max_length: Optional[int] = None,
        padding_side: str = "left",  # Left padding for generation
    ):
        super().__init__(pad_token_id, -100, max_length, padding_side)

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Collate RL batch with left padding."""
        batch = super().__call__(features)

        # Store prompt lengths for later use
        if "prompt_length" in features[0]:
            batch["prompt_lengths"] = torch.tensor([f["prompt_length"] for f in features])

        return batch


class PreferenceCollator(DataCollator):
    """
    Data collator for preference learning.

    Handles chosen/rejected pairs for DPO or RLHF training.
    """

    def __init__(
        self,
        pad_token_id: int = 0,
        label_pad_token_id: int = -100,
        max_length: Optional[int] = None,
    ):
        super().__init__(pad_token_id, label_pad_token_id, max_length)

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Collate preference pairs."""
        # Separate chosen and rejected
        chosen_features = []
        rejected_features = []

        for f in features:
            chosen_features.append({
                "input_ids": f["chosen_input_ids"],
                "attention_mask": f["chosen_attention_mask"],
            })
            rejected_features.append({
                "input_ids": f["rejected_input_ids"],
                "attention_mask": f["rejected_attention_mask"],
            })

        # Collate separately
        chosen_batch = super().__call__(chosen_features)
        rejected_batch = super().__call__(rejected_features)

        # Combine
        return {
            "chosen_input_ids": chosen_batch["input_ids"],
            "chosen_attention_mask": chosen_batch["attention_mask"],
            "rejected_input_ids": rejected_batch["input_ids"],
            "rejected_attention_mask": rejected_batch["attention_mask"],
        }


class PackedCollator(DataCollator):
    """
    Data collator with sequence packing.

    Packs multiple sequences into single training examples
    to maximize GPU utilization.
    """

    def __init__(
        self,
        pad_token_id: int = 0,
        label_pad_token_id: int = -100,
        max_length: int = 2048,
        pack_sequences: bool = True,
    ):
        super().__init__(pad_token_id, label_pad_token_id, max_length)
        self.pack_sequences = pack_sequences

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Collate with optional packing."""
        if not self.pack_sequences:
            return super().__call__(features)

        # Pack sequences
        packed_input_ids = []
        packed_labels = []
        packed_attention_mask = []

        current_input_ids = []
        current_labels = []
        current_attention_mask = []
        current_length = 0

        for feature in features:
            input_ids = feature["input_ids"]
            labels = feature.get("labels", input_ids)
            attention_mask = feature.get("attention_mask", torch.ones_like(input_ids))

            # Remove padding
            valid_length = attention_mask.sum().item()
            input_ids = input_ids[:valid_length]
            labels = labels[:valid_length]
            attention_mask = attention_mask[:valid_length]

            # Check if we can add to current pack
            if current_length + len(input_ids) <= self.max_length:
                current_input_ids.append(input_ids)
                current_labels.append(labels)
                current_attention_mask.append(attention_mask)
                current_length += len(input_ids)
            else:
                # Save current pack and start new one
                if current_input_ids:
                    packed = self._finalize_pack(
                        current_input_ids, current_labels, current_attention_mask
                    )
                    packed_input_ids.append(packed["input_ids"])
                    packed_labels.append(packed["labels"])
                    packed_attention_mask.append(packed["attention_mask"])

                current_input_ids = [input_ids]
                current_labels = [labels]
                current_attention_mask = [attention_mask]
                current_length = len(input_ids)

        # Don't forget the last pack
        if current_input_ids:
            packed = self._finalize_pack(
                current_input_ids, current_labels, current_attention_mask
            )
            packed_input_ids.append(packed["input_ids"])
            packed_labels.append(packed["labels"])
            packed_attention_mask.append(packed["attention_mask"])

        return {
            "input_ids": torch.stack(packed_input_ids),
            "labels": torch.stack(packed_labels),
            "attention_mask": torch.stack(packed_attention_mask),
        }

    def _finalize_pack(
        self,
        input_ids_list: List[torch.Tensor],
        labels_list: List[torch.Tensor],
        attention_mask_list: List[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Finalize a pack with padding."""
        input_ids = torch.cat(input_ids_list)
        labels = torch.cat(labels_list)
        attention_mask = torch.cat(attention_mask_list)

        # Pad to max_length
        pad_length = self.max_length - len(input_ids)
        if pad_length > 0:
            input_ids = torch.cat([
                input_ids,
                torch.full((pad_length,), self.pad_token_id, dtype=input_ids.dtype)
            ])
            labels = torch.cat([
                labels,
                torch.full((pad_length,), self.label_pad_token_id, dtype=labels.dtype)
            ])
            attention_mask = torch.cat([
                attention_mask,
                torch.zeros(pad_length, dtype=attention_mask.dtype)
            ])

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        }
