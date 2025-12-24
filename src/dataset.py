"""
SPARSA-LM Dataset Pipeline - Modern HuggingFace Streaming Integration

Features:
- Streaming datasets from HuggingFace Hub
- Multi-domain weighted mixing (code, math, language, medical, reasoning)
- Instruction tuning datasets (MMLU, Competition Math, etc.)
- Efficient tokenization with packing
- Memory-efficient data loading

This code belongs to EllanorAI and is licensed under the EllanorAI Proprietary License.
"""

import os
import logging
import random
from typing import Dict, List, Any, Optional, Iterator, Union, Callable
from dataclasses import dataclass, field
from itertools import cycle, islice

import torch
from torch.utils.data import Dataset, IterableDataset, DataLoader

try:
    from datasets import load_dataset, interleave_datasets, IterableDataset as HFIterableDataset
    from datasets import Dataset as HFDataset
    HF_DATASETS_AVAILABLE = True
except ImportError:
    HF_DATASETS_AVAILABLE = False
    logging.warning("HuggingFace datasets not available. Install with: pip install datasets")

try:
    from transformers import AutoTokenizer, PreTrainedTokenizerFast
    HF_TOKENIZERS_AVAILABLE = True
except ImportError:
    HF_TOKENIZERS_AVAILABLE = False
    logging.warning("HuggingFace transformers not available. Install with: pip install transformers")


@dataclass
class DatasetConfig:
    """Configuration for dataset loading and processing."""

    # General settings
    max_seq_len: int = 2048
    seed: int = 42
    num_workers: int = 4

    # Streaming settings
    streaming: bool = True
    buffer_size: int = 10000
    shuffle_buffer: int = 10000

    # Packing settings (pack multiple sequences into one)
    use_packing: bool = True
    pack_sequences: bool = True

    # Domain weights for pre-training mixture
    domain_weights: Dict[str, float] = field(default_factory=lambda: {
        "language": 0.35,
        "code": 0.25,
        "math": 0.15,
        "reasoning": 0.15,
        "medical": 0.10,
    })

    # Pre-training datasets by domain
    pretrain_datasets: Dict[str, List[str]] = field(default_factory=lambda: {
        "language": [
            "allenai/c4",
            "wikipedia",
            "bookcorpus",
        ],
        "code": [
            "bigcode/starcoderdata",
            "codeparrot/github-code",
        ],
        "math": [
            "open-web-math/open-web-math",
            "camel-ai/math",
        ],
        "reasoning": [
            "allenai/ai2_arc",
            "gsm8k",
        ],
        "medical": [
            "medmcqa",
            "pubmed_qa",
        ],
    })

    # Instruction tuning datasets
    instruction_datasets: List[str] = field(default_factory=lambda: [
        "cais/mmlu",
        "hendrycks/competition_math",
        "bigcode/the-stack-smol",
        "openai/gsm8k",
        "allenai/ai2_arc",
    ])


class StreamingPretrainDataset(IterableDataset):
    """Streaming dataset for pre-training with weighted domain mixing."""

    def __init__(
        self,
        config: DatasetConfig,
        tokenizer: PreTrainedTokenizerFast,
        split: str = "train",
    ):
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer
        self.split = split
        self.max_seq_len = config.max_seq_len

        # Set up special token IDs
        self.pad_token_id = tokenizer.pad_token_id or 0
        self.bos_token_id = tokenizer.bos_token_id or 1
        self.eos_token_id = tokenizer.eos_token_id or 2

        # Load and interleave datasets
        self.datasets = self._load_domain_datasets()

    def _load_domain_datasets(self) -> List[HFIterableDataset]:
        """Load datasets for each domain with streaming."""
        domain_datasets = []
        domain_weights = []

        for domain, weight in self.config.domain_weights.items():
            if domain not in self.config.pretrain_datasets:
                continue

            for dataset_name in self.config.pretrain_datasets[domain]:
                try:
                    # Load with streaming
                    ds = self._load_single_dataset(dataset_name)
                    if ds is not None:
                        domain_datasets.append(ds)
                        domain_weights.append(weight / len(self.config.pretrain_datasets[domain]))
                        logging.info(f"Loaded {dataset_name} for domain {domain}")
                except Exception as e:
                    logging.warning(f"Failed to load {dataset_name}: {e}")

        # Normalize weights
        if domain_weights:
            total = sum(domain_weights)
            domain_weights = [w / total for w in domain_weights]

        return domain_datasets, domain_weights

    def _load_single_dataset(self, dataset_name: str) -> Optional[HFIterableDataset]:
        """Load a single dataset with appropriate configuration."""
        try:
            # Handle different dataset formats
            if dataset_name == "allenai/c4":
                ds = load_dataset(
                    "allenai/c4",
                    "en",
                    split=self.split,
                    streaming=self.config.streaming,
                    trust_remote_code=True,
                )
            elif dataset_name == "wikipedia":
                ds = load_dataset(
                    "wikipedia",
                    "20220301.en",
                    split=self.split,
                    streaming=self.config.streaming,
                    trust_remote_code=True,
                )
            elif dataset_name == "bigcode/starcoderdata":
                ds = load_dataset(
                    "bigcode/starcoderdata",
                    split=self.split,
                    streaming=self.config.streaming,
                    trust_remote_code=True,
                )
            else:
                ds = load_dataset(
                    dataset_name,
                    split=self.split,
                    streaming=self.config.streaming,
                    trust_remote_code=True,
                )

            # Shuffle if streaming
            if self.config.streaming and hasattr(ds, 'shuffle'):
                ds = ds.shuffle(seed=self.config.seed, buffer_size=self.config.shuffle_buffer)

            return ds

        except Exception as e:
            logging.warning(f"Could not load {dataset_name}: {e}")
            return None

    def _get_text_from_example(self, example: Dict[str, Any]) -> str:
        """Extract text from various dataset formats."""
        # Try common text field names
        text_fields = ['text', 'content', 'article', 'passage', 'context', 'question', 'answer']

        for field in text_fields:
            if field in example and example[field]:
                return str(example[field])

        # Concatenate all string fields if no specific field found
        texts = []
        for key, value in example.items():
            if isinstance(value, str) and value.strip():
                texts.append(value)

        return " ".join(texts)

    def _tokenize_and_pack(self, texts: List[str]) -> Dict[str, torch.Tensor]:
        """Tokenize and pack multiple texts into a single sequence."""
        all_tokens = []

        for text in texts:
            tokens = self.tokenizer.encode(
                text,
                add_special_tokens=False,
                truncation=False,
            )
            all_tokens.extend(tokens)
            all_tokens.append(self.eos_token_id)

        # Create packed sequences
        sequences = []
        current_seq = [self.bos_token_id]

        for token in all_tokens:
            if len(current_seq) >= self.max_seq_len - 1:
                current_seq.append(self.eos_token_id)
                sequences.append(current_seq[:self.max_seq_len])
                current_seq = [self.bos_token_id]
            current_seq.append(token)

        # Handle remaining tokens
        if len(current_seq) > 1:
            current_seq.append(self.eos_token_id)
            # Pad to max_seq_len
            while len(current_seq) < self.max_seq_len:
                current_seq.append(self.pad_token_id)
            sequences.append(current_seq[:self.max_seq_len])

        return sequences

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        """Iterate over the dataset with weighted sampling."""
        datasets, weights = self.datasets

        if not datasets:
            logging.warning("No datasets available for iteration")
            return

        # Create weighted iterators
        iterators = [iter(ds) for ds in datasets]
        text_buffer = []
        buffer_size = 10 if self.config.use_packing else 1

        while True:
            try:
                # Sample a domain based on weights
                domain_idx = random.choices(range(len(weights)), weights=weights, k=1)[0]
                iterator = iterators[domain_idx]

                try:
                    example = next(iterator)
                    text = self._get_text_from_example(example)

                    if text.strip():
                        text_buffer.append(text)

                    # Process buffer when full
                    if len(text_buffer) >= buffer_size:
                        if self.config.use_packing:
                            sequences = self._tokenize_and_pack(text_buffer)
                            for seq in sequences:
                                input_ids = torch.tensor(seq, dtype=torch.long)
                                attention_mask = (input_ids != self.pad_token_id).long()
                                labels = input_ids.clone()
                                labels[labels == self.pad_token_id] = -100

                                yield {
                                    "input_ids": input_ids,
                                    "attention_mask": attention_mask,
                                    "labels": labels,
                                }
                        else:
                            for text in text_buffer:
                                encoding = self.tokenizer(
                                    text,
                                    max_length=self.max_seq_len,
                                    padding="max_length",
                                    truncation=True,
                                    return_tensors="pt",
                                )
                                input_ids = encoding["input_ids"].squeeze(0)
                                attention_mask = encoding["attention_mask"].squeeze(0)
                                labels = input_ids.clone()
                                labels[attention_mask == 0] = -100

                                yield {
                                    "input_ids": input_ids,
                                    "attention_mask": attention_mask,
                                    "labels": labels,
                                }

                        text_buffer = []

                except StopIteration:
                    # Refresh iterator for this domain
                    iterators[domain_idx] = iter(datasets[domain_idx])

            except Exception as e:
                logging.warning(f"Error during iteration: {e}")
                continue


class InstructionDataset(Dataset):
    """Dataset for instruction tuning with various formats."""

    def __init__(
        self,
        config: DatasetConfig,
        tokenizer: PreTrainedTokenizerFast,
        split: str = "train",
    ):
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer
        self.split = split
        self.max_seq_len = config.max_seq_len

        # Load instruction datasets
        self.examples = self._load_instruction_data()

    def _load_instruction_data(self) -> List[Dict[str, str]]:
        """Load and format instruction tuning data."""
        all_examples = []

        for dataset_name in self.config.instruction_datasets:
            try:
                examples = self._load_and_format_dataset(dataset_name)
                all_examples.extend(examples)
                logging.info(f"Loaded {len(examples)} examples from {dataset_name}")
            except Exception as e:
                logging.warning(f"Failed to load {dataset_name}: {e}")

        random.shuffle(all_examples)
        return all_examples

    def _load_and_format_dataset(self, dataset_name: str) -> List[Dict[str, str]]:
        """Load and format a specific instruction dataset."""
        examples = []

        try:
            if dataset_name == "cais/mmlu":
                ds = load_dataset("cais/mmlu", "all", split=self.split, trust_remote_code=True)
                for item in ds:
                    question = item.get("question", "")
                    choices = item.get("choices", [])
                    answer_idx = item.get("answer", 0)

                    if choices:
                        formatted_choices = "\n".join([f"{chr(65+i)}. {c}" for i, c in enumerate(choices)])
                        instruction = f"Question: {question}\n\n{formatted_choices}"
                        response = f"The answer is {chr(65+answer_idx)}: {choices[answer_idx]}"
                        examples.append({"instruction": instruction, "response": response})

            elif dataset_name == "hendrycks/competition_math":
                ds = load_dataset("hendrycks/competition_math", split=self.split, trust_remote_code=True)
                for item in ds:
                    problem = item.get("problem", "")
                    solution = item.get("solution", "")
                    examples.append({
                        "instruction": f"Solve the following math problem:\n{problem}",
                        "response": solution,
                    })

            elif dataset_name == "bigcode/the-stack-smol":
                ds = load_dataset(
                    "bigcode/the-stack-smol",
                    "data/python",
                    split=self.split,
                    streaming=True,
                    trust_remote_code=True,
                )
                # Take a subset for instruction tuning
                for i, item in enumerate(ds):
                    if i >= 10000:  # Limit for instruction dataset
                        break
                    content = item.get("content", "")
                    if len(content) > 100 and len(content) < 5000:
                        # Create code explanation task
                        examples.append({
                            "instruction": "Explain what the following Python code does:\n```python\n" + content[:1000] + "\n```",
                            "response": "This code " + content[:200] + "...",
                        })

            elif dataset_name == "openai/gsm8k":
                ds = load_dataset("openai/gsm8k", "main", split=self.split, trust_remote_code=True)
                for item in ds:
                    question = item.get("question", "")
                    answer = item.get("answer", "")
                    examples.append({
                        "instruction": f"Solve this math word problem:\n{question}",
                        "response": answer,
                    })

            elif dataset_name == "allenai/ai2_arc":
                ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split=self.split, trust_remote_code=True)
                for item in ds:
                    question = item.get("question", "")
                    choices = item.get("choices", {})
                    answer_key = item.get("answerKey", "")

                    if choices:
                        labels = choices.get("label", [])
                        texts = choices.get("text", [])
                        formatted = "\n".join([f"{l}. {t}" for l, t in zip(labels, texts)])
                        examples.append({
                            "instruction": f"Question: {question}\n{formatted}",
                            "response": f"The correct answer is {answer_key}.",
                        })

        except Exception as e:
            logging.warning(f"Error loading {dataset_name}: {e}")

        return examples

    def _format_chat(self, instruction: str, response: str) -> str:
        """Format instruction-response pair as chat template."""
        return f"<|user|>\n{instruction}\n<|assistant|>\n{response}<|end|>"

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        example = self.examples[idx]

        # Format as chat
        text = self._format_chat(example["instruction"], example["response"])

        # Tokenize
        encoding = self.tokenizer(
            text,
            max_length=self.max_seq_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)

        # Create labels (mask padding and instruction part)
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100

        # Find where assistant response starts and mask everything before
        assistant_token = self.tokenizer.encode("<|assistant|>", add_special_tokens=False)
        if assistant_token:
            input_list = input_ids.tolist()
            for i in range(len(input_list) - len(assistant_token)):
                if input_list[i:i+len(assistant_token)] == assistant_token:
                    labels[:i+len(assistant_token)] = -100
                    break

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


class PreferenceDataset(Dataset):
    """Dataset for preference learning (RLHF/DPO)."""

    def __init__(
        self,
        config: DatasetConfig,
        tokenizer: PreTrainedTokenizerFast,
        split: str = "train",
    ):
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer
        self.split = split
        self.max_seq_len = config.max_seq_len

        # Load preference datasets
        self.examples = self._load_preference_data()

    def _load_preference_data(self) -> List[Dict[str, str]]:
        """Load preference pairs (chosen vs rejected)."""
        examples = []

        preference_datasets = [
            "Anthropic/hh-rlhf",
            "stanfordnlp/SHP",
        ]

        for dataset_name in preference_datasets:
            try:
                ds = load_dataset(dataset_name, split=self.split, trust_remote_code=True)
                for item in ds:
                    prompt = item.get("prompt", item.get("history", ""))
                    chosen = item.get("chosen", item.get("human_ref_A", ""))
                    rejected = item.get("rejected", item.get("human_ref_B", ""))

                    if prompt and chosen and rejected:
                        examples.append({
                            "prompt": prompt,
                            "chosen": chosen,
                            "rejected": rejected,
                        })

            except Exception as e:
                logging.warning(f"Failed to load {dataset_name}: {e}")

        return examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        example = self.examples[idx]

        # Tokenize prompt + chosen
        chosen_text = f"{example['prompt']}\n{example['chosen']}"
        chosen_encoding = self.tokenizer(
            chosen_text,
            max_length=self.max_seq_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # Tokenize prompt + rejected
        rejected_text = f"{example['prompt']}\n{example['rejected']}"
        rejected_encoding = self.tokenizer(
            rejected_text,
            max_length=self.max_seq_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "chosen_input_ids": chosen_encoding["input_ids"].squeeze(0),
            "chosen_attention_mask": chosen_encoding["attention_mask"].squeeze(0),
            "rejected_input_ids": rejected_encoding["input_ids"].squeeze(0),
            "rejected_attention_mask": rejected_encoding["attention_mask"].squeeze(0),
        }


def create_dataloader(
    dataset: Union[Dataset, IterableDataset],
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
    drop_last: bool = True,
) -> DataLoader:
    """Create a DataLoader with optimal settings."""

    def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        return {
            key: torch.stack([item[key] for item in batch])
            for key in batch[0].keys()
        }

    # IterableDataset doesn't support shuffle
    if isinstance(dataset, IterableDataset):
        shuffle = False

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        collate_fn=collate_fn,
    )


def get_tokenizer(
    tokenizer_path: str = "tokenizer",
    vocab_size: int = 32000,
) -> PreTrainedTokenizerFast:
    """Load or create a modern tokenizer."""

    if os.path.exists(tokenizer_path):
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    else:
        # Use LLaMA tokenizer as base
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                "meta-llama/Llama-2-7b-hf",
                use_fast=True,
            )
        except Exception:
            # Fallback to a simple tokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                "gpt2",
                use_fast=True,
            )

    # Ensure special tokens are set
    special_tokens = {
        "pad_token": "<|pad|>",
        "bos_token": "<|begin|>",
        "eos_token": "<|end|>",
        "unk_token": "<|unk|>",
    }

    # Add chat tokens
    additional_tokens = [
        "<|user|>",
        "<|assistant|>",
        "<|system|>",
    ]

    tokenizer.add_special_tokens(special_tokens)
    tokenizer.add_tokens(additional_tokens)

    return tokenizer


# Backward compatibility
class DatasetProcessor:
    """Backward compatible wrapper for dataset processing."""

    def __init__(self, source_dir: str = None, split: Dict = None, preprocessing_config: Dict = None):
        self.config = DatasetConfig()
        self.source_dir = source_dir

    def get_train_dataset(self, tokenizer, max_length: int = 2048):
        self.config.max_seq_len = max_length
        return StreamingPretrainDataset(self.config, tokenizer, split="train")

    def get_val_dataset(self, tokenizer, max_length: int = 2048):
        self.config.max_seq_len = max_length
        return StreamingPretrainDataset(self.config, tokenizer, split="validation")


class TextDataset(Dataset):
    """Simple text dataset for backward compatibility."""

    def __init__(self, data: List[str], tokenizer, max_length: int):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data[idx]
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    @staticmethod
    def collate_fn(batch):
        return {
            key: torch.stack([item[key] for item in batch])
            for key in batch[0].keys()
        }
