"""
SPARSA-LM Dataset Module
Dataset classes for pretraining and finetuning
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Iterator, Any, Union
import random

import torch
from torch.utils.data import Dataset, IterableDataset


@dataclass
class DatasetConfig:
    """Configuration for datasets."""

    # Tokenizer
    tokenizer_path: str = "tokenizer"

    # Sequence length
    max_seq_length: int = 2048

    # Pretrain data sources
    pretrain_datasets: List[str] = field(default_factory=lambda: [
        "c4",
        "wikipedia",
        "openwebtext",
    ])

    # Domain weights for pretraining
    domain_weights: Dict[str, float] = field(default_factory=lambda: {
        "language": 0.35,
        "code": 0.25,
        "math": 0.15,
        "reasoning": 0.15,
        "other": 0.10,
    })

    # Finetune data format
    finetune_format: str = "instruction"  # or "conversation"

    # Data processing
    num_workers: int = 4
    prefetch_factor: int = 2

    # Streaming
    streaming: bool = True
    shuffle_buffer_size: int = 10000


class PretrainDataset(IterableDataset):
    """
    Streaming dataset for pretraining.

    Supports multiple data sources with domain weighting
    and efficient streaming from HuggingFace datasets.
    """

    def __init__(
        self,
        config: DatasetConfig,
        tokenizer: Any,
        split: str = "train",
    ):
        self.config = config
        self.tokenizer = tokenizer
        self.split = split
        self.max_seq_length = config.max_seq_length

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        """Iterate over the dataset."""
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("Please install datasets: pip install datasets")

        # Load and interleave datasets
        datasets = []
        for dataset_name in self.config.pretrain_datasets:
            try:
                ds = load_dataset(dataset_name, split=self.split, streaming=self.config.streaming)
                datasets.append(ds)
            except Exception as e:
                print(f"Warning: Could not load {dataset_name}: {e}")
                continue

        if not datasets:
            raise ValueError("No datasets could be loaded")

        # Interleave datasets with shuffling
        for ds in datasets:
            if self.config.streaming:
                ds = ds.shuffle(buffer_size=self.config.shuffle_buffer_size)

            for example in ds:
                # Extract text from different dataset formats
                text = self._extract_text(example)
                if text is None:
                    continue

                # Tokenize
                tokens = self.tokenizer.encode(text, add_special_tokens=True)

                # Chunk into sequences
                for i in range(0, len(tokens) - 1, self.max_seq_length):
                    chunk = tokens[i:i + self.max_seq_length + 1]
                    if len(chunk) < 2:
                        continue

                    input_ids = torch.tensor(chunk[:-1], dtype=torch.long)
                    labels = torch.tensor(chunk[1:], dtype=torch.long)

                    # Pad if necessary
                    if len(input_ids) < self.max_seq_length:
                        pad_length = self.max_seq_length - len(input_ids)
                        input_ids = torch.cat([
                            input_ids,
                            torch.full((pad_length,), self.tokenizer.pad_token_id, dtype=torch.long)
                        ])
                        labels = torch.cat([
                            labels,
                            torch.full((pad_length,), -100, dtype=torch.long)  # Ignore in loss
                        ])

                    yield {
                        "input_ids": input_ids,
                        "labels": labels,
                        "attention_mask": (input_ids != self.tokenizer.pad_token_id).long(),
                    }

    def _extract_text(self, example: Dict) -> Optional[str]:
        """Extract text from various dataset formats."""
        # Common text fields
        for field in ["text", "content", "document", "passage", "article"]:
            if field in example and example[field]:
                return example[field]

        # Instruction format
        if "instruction" in example:
            instruction = example.get("instruction", "")
            input_text = example.get("input", "")
            output = example.get("output", "")
            return f"{instruction}\n{input_text}\n{output}".strip()

        # Conversation format
        if "messages" in example:
            messages = example["messages"]
            return "\n".join([m.get("content", "") for m in messages])

        return None


class FinetuneDataset(Dataset):
    """
    Dataset for instruction finetuning.

    Supports instruction-following and conversation formats
    with proper prompt templating.
    """

    def __init__(
        self,
        config: DatasetConfig,
        tokenizer: Any,
        data: Union[List[Dict], str],
        split: str = "train",
    ):
        self.config = config
        self.tokenizer = tokenizer
        self.max_seq_length = config.max_seq_length

        # Load data
        if isinstance(data, str):
            self.data = self._load_data(data, split)
        else:
            self.data = data

    def _load_data(self, path: str, split: str) -> List[Dict]:
        """Load data from file or HuggingFace."""
        import json

        if path.endswith(".json") or path.endswith(".jsonl"):
            with open(path, 'r') as f:
                if path.endswith(".jsonl"):
                    return [json.loads(line) for line in f]
                return json.load(f)
        else:
            # Try loading from HuggingFace
            try:
                from datasets import load_dataset
                ds = load_dataset(path, split=split)
                return list(ds)
            except Exception:
                raise ValueError(f"Could not load data from {path}")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single example."""
        example = self.data[idx]

        # Format based on data type
        if self.config.finetune_format == "instruction":
            text = self._format_instruction(example)
        else:
            text = self._format_conversation(example)

        # Tokenize
        tokens = self.tokenizer.encode(text, add_special_tokens=True)

        # Truncate if necessary
        if len(tokens) > self.max_seq_length + 1:
            tokens = tokens[:self.max_seq_length + 1]

        input_ids = torch.tensor(tokens[:-1], dtype=torch.long)
        labels = torch.tensor(tokens[1:], dtype=torch.long)

        # Pad if necessary
        if len(input_ids) < self.max_seq_length:
            pad_length = self.max_seq_length - len(input_ids)
            input_ids = torch.cat([
                input_ids,
                torch.full((pad_length,), self.tokenizer.pad_token_id, dtype=torch.long)
            ])
            labels = torch.cat([
                labels,
                torch.full((pad_length,), -100, dtype=torch.long)
            ])

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": (input_ids != self.tokenizer.pad_token_id).long(),
        }

    def _format_instruction(self, example: Dict) -> str:
        """Format instruction-following example."""
        instruction = example.get("instruction", "")
        input_text = example.get("input", "")
        output = example.get("output", example.get("response", ""))

        if input_text:
            prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n{output}"
        else:
            prompt = f"### Instruction:\n{instruction}\n\n### Response:\n{output}"

        return prompt

    def _format_conversation(self, example: Dict) -> str:
        """Format conversation example."""
        messages = example.get("messages", example.get("conversation", []))

        formatted = []
        for msg in messages:
            role = msg.get("role", msg.get("from", "user"))
            content = msg.get("content", msg.get("value", ""))

            if role in ["user", "human"]:
                formatted.append(f"User: {content}")
            elif role in ["assistant", "gpt", "model"]:
                formatted.append(f"Assistant: {content}")
            elif role == "system":
                formatted.append(f"System: {content}")

        return "\n\n".join(formatted)


class RLDataset(Dataset):
    """
    Dataset for reinforcement learning finetuning.

    Provides prompts for policy generation during RL training.
    """

    def __init__(
        self,
        config: DatasetConfig,
        tokenizer: Any,
        prompts: List[str],
    ):
        self.config = config
        self.tokenizer = tokenizer
        self.prompts = prompts
        self.max_seq_length = config.max_seq_length

    def __len__(self) -> int:
        return len(self.prompts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a prompt for generation."""
        prompt = self.prompts[idx]

        # Tokenize prompt
        tokens = self.tokenizer.encode(prompt, add_special_tokens=True)

        # Truncate if necessary
        if len(tokens) > self.max_seq_length:
            tokens = tokens[:self.max_seq_length]

        input_ids = torch.tensor(tokens, dtype=torch.long)

        # Pad if necessary
        if len(input_ids) < self.max_seq_length:
            pad_length = self.max_seq_length - len(input_ids)
            attention_mask = torch.ones(len(input_ids), dtype=torch.long)
            input_ids = torch.cat([
                input_ids,
                torch.full((pad_length,), self.tokenizer.pad_token_id, dtype=torch.long)
            ])
            attention_mask = torch.cat([
                attention_mask,
                torch.zeros(pad_length, dtype=torch.long)
            ])
        else:
            attention_mask = torch.ones(len(input_ids), dtype=torch.long)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "prompt_length": len(tokens),
        }


class PreferenceDataset(Dataset):
    """
    Dataset for preference learning (DPO, RLHF).

    Contains pairs of chosen and rejected responses.
    """

    def __init__(
        self,
        config: DatasetConfig,
        tokenizer: Any,
        data: List[Dict],
    ):
        self.config = config
        self.tokenizer = tokenizer
        self.data = data
        self.max_seq_length = config.max_seq_length

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a preference pair."""
        example = self.data[idx]

        prompt = example.get("prompt", "")
        chosen = example.get("chosen", "")
        rejected = example.get("rejected", "")

        # Tokenize chosen
        chosen_text = f"{prompt}{chosen}"
        chosen_tokens = self.tokenizer.encode(chosen_text, add_special_tokens=True)

        # Tokenize rejected
        rejected_text = f"{prompt}{rejected}"
        rejected_tokens = self.tokenizer.encode(rejected_text, add_special_tokens=True)

        # Truncate
        chosen_tokens = chosen_tokens[:self.max_seq_length]
        rejected_tokens = rejected_tokens[:self.max_seq_length]

        # Pad chosen
        chosen_input_ids = torch.tensor(chosen_tokens, dtype=torch.long)
        if len(chosen_input_ids) < self.max_seq_length:
            pad_length = self.max_seq_length - len(chosen_input_ids)
            chosen_attention_mask = torch.ones(len(chosen_input_ids), dtype=torch.long)
            chosen_input_ids = torch.cat([
                chosen_input_ids,
                torch.full((pad_length,), self.tokenizer.pad_token_id, dtype=torch.long)
            ])
            chosen_attention_mask = torch.cat([
                chosen_attention_mask,
                torch.zeros(pad_length, dtype=torch.long)
            ])
        else:
            chosen_attention_mask = torch.ones(len(chosen_input_ids), dtype=torch.long)

        # Pad rejected
        rejected_input_ids = torch.tensor(rejected_tokens, dtype=torch.long)
        if len(rejected_input_ids) < self.max_seq_length:
            pad_length = self.max_seq_length - len(rejected_input_ids)
            rejected_attention_mask = torch.ones(len(rejected_input_ids), dtype=torch.long)
            rejected_input_ids = torch.cat([
                rejected_input_ids,
                torch.full((pad_length,), self.tokenizer.pad_token_id, dtype=torch.long)
            ])
            rejected_attention_mask = torch.cat([
                rejected_attention_mask,
                torch.zeros(pad_length, dtype=torch.long)
            ])
        else:
            rejected_attention_mask = torch.ones(len(rejected_input_ids), dtype=torch.long)

        return {
            "chosen_input_ids": chosen_input_ids,
            "chosen_attention_mask": chosen_attention_mask,
            "rejected_input_ids": rejected_input_ids,
            "rejected_attention_mask": rejected_attention_mask,
        }
