"""
SPARSA-LM SFT Data Processing
Production-quality instruction dataset handling with sequence packing
"""

import json
import logging
import random
import hashlib
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Iterator, Union, Tuple, Callable
from pathlib import Path
from collections import defaultdict
from abc import ABC, abstractmethod
import math

import torch
from torch.utils.data import Dataset, IterableDataset, get_worker_info

logger = logging.getLogger(__name__)


# =============================================================================
# PROMPT TEMPLATES - COMPREHENSIVE COLLECTION
# =============================================================================

@dataclass
class PromptTemplate:
    """Configurable prompt template."""
    name: str
    system_template: Optional[str] = None
    user_template: str = "{content}"
    assistant_template: str = "{content}"
    system_token: Optional[str] = None
    user_token: Optional[str] = None
    assistant_token: Optional[str] = None
    end_token: str = ""
    bos_token: str = ""
    eos_token: str = ""

    # For token-level loss masking
    response_template: Optional[str] = None  # Marker for where response starts


PROMPT_TEMPLATES: Dict[str, PromptTemplate] = {
    "alpaca": PromptTemplate(
        name="alpaca",
        system_template=None,
        user_template=(
            "Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n{content}\n\n"
            "### Response:\n"
        ),
        assistant_template="{content}",
        response_template="### Response:\n",
        eos_token="</s>",
    ),
    "alpaca_input": PromptTemplate(
        name="alpaca_input",
        system_template=None,
        user_template=(
            "Below is an instruction that describes a task, paired with an input that provides further context. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n{instruction}\n\n"
            "### Input:\n{input}\n\n"
            "### Response:\n"
        ),
        assistant_template="{content}",
        response_template="### Response:\n",
        eos_token="</s>",
    ),
    "chatml": PromptTemplate(
        name="chatml",
        system_template="<|im_start|>system\n{content}<|im_end|>\n",
        user_template="<|im_start|>user\n{content}<|im_end|>\n",
        assistant_template="<|im_start|>assistant\n{content}<|im_end|>\n",
        response_template="<|im_start|>assistant\n",
        bos_token="<|im_start|>",
        eos_token="<|im_end|>",
    ),
    "llama2": PromptTemplate(
        name="llama2",
        system_template="<<SYS>>\n{content}\n<</SYS>>\n\n",
        user_template="[INST] {content} [/INST]",
        assistant_template=" {content} </s>",
        response_template="[/INST]",
        bos_token="<s>",
        eos_token="</s>",
    ),
    "llama3": PromptTemplate(
        name="llama3",
        system_template="<|start_header_id|>system<|end_header_id|>\n\n{content}<|eot_id|>",
        user_template="<|start_header_id|>user<|end_header_id|>\n\n{content}<|eot_id|>",
        assistant_template="<|start_header_id|>assistant<|end_header_id|>\n\n{content}<|eot_id|>",
        response_template="<|start_header_id|>assistant<|end_header_id|>\n\n",
        bos_token="<|begin_of_text|>",
        eos_token="<|eot_id|>",
    ),
    "zephyr": PromptTemplate(
        name="zephyr",
        system_template="<|system|>\n{content}</s>\n",
        user_template="<|user|>\n{content}</s>\n",
        assistant_template="<|assistant|>\n{content}</s>\n",
        response_template="<|assistant|>\n",
        eos_token="</s>",
    ),
    "vicuna": PromptTemplate(
        name="vicuna",
        system_template="A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.\n\n",
        user_template="USER: {content}\n",
        assistant_template="ASSISTANT: {content}</s>\n",
        response_template="ASSISTANT:",
        eos_token="</s>",
    ),
    "mistral": PromptTemplate(
        name="mistral",
        system_template=None,
        user_template="[INST] {content} [/INST]",
        assistant_template=" {content}</s>",
        response_template="[/INST]",
        bos_token="<s>",
        eos_token="</s>",
    ),
    "phi3": PromptTemplate(
        name="phi3",
        system_template="<|system|>\n{content}<|end|>\n",
        user_template="<|user|>\n{content}<|end|>\n",
        assistant_template="<|assistant|>\n{content}<|end|>\n",
        response_template="<|assistant|>\n",
        eos_token="<|end|>",
    ),
    "gemma": PromptTemplate(
        name="gemma",
        system_template=None,
        user_template="<start_of_turn>user\n{content}<end_of_turn>\n",
        assistant_template="<start_of_turn>model\n{content}<end_of_turn>\n",
        response_template="<start_of_turn>model\n",
        bos_token="<bos>",
        eos_token="<eos>",
    ),
}


# =============================================================================
# CONVERSATION FORMATTER
# =============================================================================

class ConversationFormatter:
    """
    Robust conversation formatter with token-level loss masking support.

    Handles:
    - Multi-turn conversations
    - System messages
    - Input fields (for Alpaca-style)
    - Token position tracking for loss masking
    """

    def __init__(
        self,
        template_name: str = "chatml",
        tokenizer: Any = None,
        max_seq_length: int = 2048,
        add_generation_prompt: bool = False,
    ):
        if template_name not in PROMPT_TEMPLATES:
            raise ValueError(f"Unknown template: {template_name}. Available: {list(PROMPT_TEMPLATES.keys())}")

        self.template = PROMPT_TEMPLATES[template_name]
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.add_generation_prompt = add_generation_prompt

        # Cache response template tokens for loss masking
        self._response_template_tokens = None
        if tokenizer and self.template.response_template:
            self._response_template_tokens = tokenizer.encode(
                self.template.response_template,
                add_special_tokens=False
            )

    def format_messages(
        self,
        messages: List[Dict[str, str]],
        system_message: Optional[str] = None,
    ) -> Tuple[str, List[Tuple[int, int]]]:
        """
        Format messages into a single string with response position tracking.

        Args:
            messages: List of message dicts with 'role' and 'content'
            system_message: Optional system message to prepend

        Returns:
            Tuple of (formatted_text, response_positions)
            where response_positions is list of (start, end) character positions
        """
        formatted = ""
        response_positions = []

        # Add BOS if template has it
        if self.template.bos_token:
            formatted += self.template.bos_token

        # Add system message
        if system_message and self.template.system_template:
            formatted += self.template.system_template.format(content=system_message)

        # Format each message
        for msg in messages:
            role = msg.get("role", "user").lower()
            content = msg.get("content", "")

            # Normalize role names
            if role in ["human", "user", "input"]:
                role = "user"
            elif role in ["gpt", "assistant", "model", "output", "bot"]:
                role = "assistant"
            elif role == "system":
                if self.template.system_template:
                    formatted += self.template.system_template.format(content=content)
                continue

            if role == "user":
                # Handle Alpaca-style input
                if "input" in msg and msg["input"]:
                    if self.template.name == "alpaca_input":
                        formatted += self.template.user_template.format(
                            instruction=content,
                            input=msg["input"]
                        )
                    else:
                        formatted += self.template.user_template.format(
                            content=f"{content}\n\nInput: {msg['input']}"
                        )
                else:
                    formatted += self.template.user_template.format(content=content)

            elif role == "assistant":
                # Track response position for loss masking
                start_pos = len(formatted)
                formatted += self.template.assistant_template.format(content=content)
                end_pos = len(formatted)
                response_positions.append((start_pos, end_pos))

        # Add generation prompt if requested (for inference)
        if self.add_generation_prompt:
            formatted += self.template.response_template or ""

        return formatted, response_positions

    def format_instruction(
        self,
        instruction: str,
        input_text: str = "",
        output: str = "",
    ) -> Tuple[str, List[Tuple[int, int]]]:
        """Format Alpaca-style instruction."""
        messages = [
            {"role": "user", "content": instruction, "input": input_text},
        ]
        if output:
            messages.append({"role": "assistant", "content": output})

        return self.format_messages(messages)

    def tokenize_with_labels(
        self,
        text: str,
        response_positions: List[Tuple[int, int]],
        mask_prompt: bool = True,
    ) -> Tuple[List[int], List[int]]:
        """
        Tokenize text and create labels with proper masking.

        Args:
            text: Formatted conversation text
            response_positions: Character positions of responses
            mask_prompt: Whether to mask prompt tokens with -100

        Returns:
            Tuple of (input_ids, labels)
        """
        # Tokenize with offset mapping
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_seq_length,
            truncation=True,
            return_offsets_mapping=True,
        )

        input_ids = encoding["input_ids"]
        offset_mapping = encoding.get("offset_mapping", [])

        if not mask_prompt or not response_positions:
            # Train on all tokens
            labels = input_ids.copy()
        else:
            # Mask prompt tokens
            labels = [-100] * len(input_ids)

            for token_idx, (start, end) in enumerate(offset_mapping):
                if start == end:
                    continue

                # Check if this token is within any response region
                for resp_start, resp_end in response_positions:
                    if start >= resp_start and end <= resp_end:
                        labels[token_idx] = input_ids[token_idx]
                        break

        # Shift labels for next-token prediction
        labels = labels[1:] + [-100]
        input_ids = input_ids[:-1]

        return input_ids, labels


# =============================================================================
# DATA QUALITY FILTERS
# =============================================================================

class DataFilter(ABC):
    """Abstract base class for data filters."""

    @abstractmethod
    def __call__(self, example: Dict[str, Any]) -> bool:
        """Return True if example passes the filter."""
        pass


class LengthFilter(DataFilter):
    """Filter by text length."""

    def __init__(self, min_length: int = 10, max_length: int = 100000):
        self.min_length = min_length
        self.max_length = max_length

    def __call__(self, example: Dict[str, Any]) -> bool:
        text = self._get_text(example)
        return self.min_length <= len(text) <= self.max_length

    def _get_text(self, example: Dict[str, Any]) -> str:
        if "text" in example:
            return example["text"]
        if "messages" in example:
            return " ".join(m.get("content", "") for m in example["messages"])
        if "conversations" in example:
            return " ".join(m.get("value", m.get("content", "")) for m in example["conversations"])
        return str(example)


class LanguageFilter(DataFilter):
    """Filter by language (using simple heuristics or fasttext)."""

    def __init__(self, allowed_languages: List[str] = None, min_confidence: float = 0.8):
        self.allowed_languages = allowed_languages or ["en"]
        self.min_confidence = min_confidence
        self._detector = None

    def __call__(self, example: Dict[str, Any]) -> bool:
        # Simple heuristic: check for high proportion of ASCII
        text = self._get_text(example)
        if not text:
            return False
        ascii_ratio = sum(c.isascii() for c in text) / len(text)
        return ascii_ratio > 0.9  # Rough English filter

    def _get_text(self, example: Dict[str, Any]) -> str:
        if "text" in example:
            return example["text"][:1000]
        if "messages" in example:
            return " ".join(m.get("content", "") for m in example["messages"][:3])[:1000]
        return ""


class QualityFilter(DataFilter):
    """Filter by content quality heuristics."""

    def __init__(
        self,
        min_words: int = 5,
        max_repetition_ratio: float = 0.3,
        min_unique_words_ratio: float = 0.2,
    ):
        self.min_words = min_words
        self.max_repetition_ratio = max_repetition_ratio
        self.min_unique_words_ratio = min_unique_words_ratio

    def __call__(self, example: Dict[str, Any]) -> bool:
        text = self._get_text(example)
        words = text.lower().split()

        if len(words) < self.min_words:
            return False

        # Check for excessive repetition
        word_counts = defaultdict(int)
        for word in words:
            word_counts[word] += 1

        max_count = max(word_counts.values()) if word_counts else 0
        if max_count / len(words) > self.max_repetition_ratio:
            return False

        # Check vocabulary diversity
        unique_ratio = len(word_counts) / len(words)
        if unique_ratio < self.min_unique_words_ratio:
            return False

        return True

    def _get_text(self, example: Dict[str, Any]) -> str:
        if "text" in example:
            return example["text"]
        if "messages" in example:
            return " ".join(m.get("content", "") for m in example["messages"])
        return str(example)


class DuplicateFilter(DataFilter):
    """Filter duplicate examples using MinHash or exact matching."""

    def __init__(self, num_perm: int = 128, threshold: float = 0.8):
        self.seen_hashes = set()
        self.num_perm = num_perm
        self.threshold = threshold

    def __call__(self, example: Dict[str, Any]) -> bool:
        text = self._get_text(example)
        text_hash = hashlib.md5(text.encode()).hexdigest()

        if text_hash in self.seen_hashes:
            return False

        self.seen_hashes.add(text_hash)
        return True

    def _get_text(self, example: Dict[str, Any]) -> str:
        if "text" in example:
            return example["text"]
        if "messages" in example:
            return " ".join(m.get("content", "") for m in example["messages"])
        return str(example)

    def reset(self):
        """Reset seen hashes."""
        self.seen_hashes.clear()


# =============================================================================
# SEQUENCE PACKING
# =============================================================================

class SequencePacker:
    """
    Pack multiple sequences into single examples for efficient training.

    This significantly improves GPU utilization by reducing padding.
    Uses First-Fit-Decreasing bin packing algorithm.
    """

    def __init__(
        self,
        max_seq_length: int = 2048,
        pad_token_id: int = 0,
        add_eos_between: bool = True,
        eos_token_id: int = 2,
        shuffle_before_packing: bool = True,
        seed: int = 42,
    ):
        self.max_seq_length = max_seq_length
        self.pad_token_id = pad_token_id
        self.add_eos_between = add_eos_between
        self.eos_token_id = eos_token_id
        self.shuffle_before_packing = shuffle_before_packing
        self.seed = seed

    def pack_sequences(
        self,
        sequences: List[Dict[str, torch.Tensor]],
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Pack sequences into bins of max_seq_length.

        Args:
            sequences: List of dicts with 'input_ids', 'attention_mask', 'labels'

        Returns:
            Packed sequences
        """
        if self.shuffle_before_packing:
            random.seed(self.seed)
            sequences = sequences.copy()
            random.shuffle(sequences)

        # Extract lengths and sort by length (descending)
        seq_with_len = [(len(s["input_ids"]), i, s) for i, s in enumerate(sequences)]
        seq_with_len.sort(key=lambda x: -x[0])

        # Bin packing using First-Fit-Decreasing
        bins: List[List[int]] = []  # List of sequence indices in each bin
        bin_lengths: List[int] = []  # Current length of each bin

        for length, idx, _ in seq_with_len:
            sep_len = 1 if self.add_eos_between else 0

            # Find first bin that can fit this sequence
            placed = False
            for bin_idx, bin_len in enumerate(bin_lengths):
                if bin_len + length + sep_len <= self.max_seq_length:
                    bins[bin_idx].append(idx)
                    bin_lengths[bin_idx] += length + sep_len
                    placed = True
                    break

            # Create new bin if needed
            if not placed:
                bins.append([idx])
                bin_lengths.append(length)

        # Create packed sequences
        packed_sequences = []
        for bin_indices in bins:
            packed = self._pack_bin([sequences[i] for i in bin_indices])
            packed_sequences.append(packed)

        logger.info(f"Packed {len(sequences)} sequences into {len(packed_sequences)} bins "
                   f"({100 * len(packed_sequences) / len(sequences):.1f}% of original)")

        return packed_sequences

    def _pack_bin(
        self,
        sequences: List[Dict[str, torch.Tensor]],
    ) -> Dict[str, torch.Tensor]:
        """Pack a bin of sequences into a single sequence."""
        input_ids = []
        attention_mask = []
        labels = []
        position_ids = []

        current_pos = 0

        for i, seq in enumerate(sequences):
            seq_len = len(seq["input_ids"])

            # Add separator between sequences
            if i > 0 and self.add_eos_between:
                input_ids.append(self.eos_token_id)
                attention_mask.append(1)
                labels.append(-100)  # Don't train on separator
                position_ids.append(0)  # Reset position for new sequence
                current_pos = 0

            # Add sequence
            input_ids.extend(seq["input_ids"].tolist() if isinstance(seq["input_ids"], torch.Tensor) else seq["input_ids"])
            attention_mask.extend(seq["attention_mask"].tolist() if isinstance(seq["attention_mask"], torch.Tensor) else seq["attention_mask"])
            labels.extend(seq["labels"].tolist() if isinstance(seq["labels"], torch.Tensor) else seq["labels"])
            position_ids.extend(range(current_pos, current_pos + seq_len))
            current_pos += seq_len

        # Pad to max_seq_length
        pad_len = self.max_seq_length - len(input_ids)
        if pad_len > 0:
            input_ids.extend([self.pad_token_id] * pad_len)
            attention_mask.extend([0] * pad_len)
            labels.extend([-100] * pad_len)
            position_ids.extend([0] * pad_len)

        # Truncate if needed (shouldn't happen with proper bin packing)
        input_ids = input_ids[:self.max_seq_length]
        attention_mask = attention_mask[:self.max_seq_length]
        labels = labels[:self.max_seq_length]
        position_ids = position_ids[:self.max_seq_length]

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "position_ids": torch.tensor(position_ids, dtype=torch.long),
        }


# =============================================================================
# DATASETS
# =============================================================================

@dataclass
class SFTDataConfig:
    """Configuration for SFT data processing."""

    # Data sources
    datasets: List[str] = field(default_factory=lambda: [
        "teknium/OpenHermes-2.5",
    ])

    # Processing
    max_seq_length: int = 2048
    template: str = "chatml"
    mask_prompt: bool = True

    # Sequence packing
    use_packing: bool = True
    pack_add_eos: bool = True

    # Data mixing
    dataset_weights: Optional[Dict[str, float]] = None
    max_samples_per_dataset: Optional[int] = None

    # Filtering
    min_length: int = 10
    max_length: int = 100000
    filter_duplicates: bool = True
    filter_quality: bool = True

    # Streaming
    streaming: bool = False
    buffer_size: int = 10000

    # Validation
    validation_split: float = 0.05
    seed: int = 42

    # Preprocessing workers
    num_proc: int = 8

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}


class InstructionDataset(Dataset):
    """
    Production-quality instruction fine-tuning dataset.

    Features:
    - Multiple format support (Alpaca, ShareGPT, OASST, etc.)
    - Token-level loss masking
    - Optional sequence packing
    - Data quality filtering
    - Efficient preprocessing
    """

    def __init__(
        self,
        data: Union[List[Dict], str],
        tokenizer: Any,
        config: SFTDataConfig,
        split: str = "train",
    ):
        self.tokenizer = tokenizer
        self.config = config
        self.split = split

        # Initialize formatter
        self.formatter = ConversationFormatter(
            template_name=config.template,
            tokenizer=tokenizer,
            max_seq_length=config.max_seq_length,
        )

        # Initialize filters
        self.filters = []
        if config.min_length > 0 or config.max_length < 100000:
            self.filters.append(LengthFilter(config.min_length, config.max_length))
        if config.filter_quality:
            self.filters.append(QualityFilter())
        if config.filter_duplicates:
            self.filters.append(DuplicateFilter())

        # Load and preprocess data
        raw_data = self._load_data(data) if isinstance(data, str) else data
        filtered_data = self._filter_data(raw_data)
        self.examples = self._preprocess_data(filtered_data)

        # Apply sequence packing if enabled
        if config.use_packing and split == "train":
            packer = SequencePacker(
                max_seq_length=config.max_seq_length,
                pad_token_id=tokenizer.pad_token_id,
                add_eos_between=config.pack_add_eos,
                eos_token_id=tokenizer.eos_token_id,
                seed=config.seed,
            )
            self.examples = packer.pack_sequences(self.examples)

        logger.info(f"Created {split} dataset with {len(self.examples)} examples")

    def _load_data(self, path: str) -> List[Dict]:
        """Load data from file or HuggingFace."""
        path_obj = Path(path)

        if path_obj.exists():
            if path_obj.suffix == ".jsonl":
                with open(path_obj, 'r') as f:
                    return [json.loads(line) for line in f]
            elif path_obj.suffix == ".json":
                with open(path_obj, 'r') as f:
                    data = json.load(f)
                    return data if isinstance(data, list) else [data]
            elif path_obj.is_dir():
                # Load all JSON/JSONL files in directory
                data = []
                for file in path_obj.glob("*.json*"):
                    data.extend(self._load_data(str(file)))
                return data
        else:
            # Try HuggingFace
            try:
                from datasets import load_dataset
                ds = load_dataset(path, split="train")
                return list(ds)
            except Exception as e:
                raise ValueError(f"Could not load data from {path}: {e}")

        return []

    def _filter_data(self, data: List[Dict]) -> List[Dict]:
        """Apply data quality filters."""
        if not self.filters:
            return data

        filtered = []
        for example in data:
            if all(f(example) for f in self.filters):
                filtered.append(example)

        logger.info(f"Filtered {len(data)} -> {len(filtered)} examples "
                   f"({100 * len(filtered) / max(len(data), 1):.1f}% kept)")

        return filtered

    def _preprocess_data(self, data: List[Dict]) -> List[Dict[str, torch.Tensor]]:
        """Preprocess all examples."""
        examples = []

        for raw_example in data:
            try:
                processed = self._process_example(raw_example)
                if processed is not None:
                    examples.append(processed)
            except Exception as e:
                logger.debug(f"Failed to process example: {e}")
                continue

        return examples

    def _process_example(self, example: Dict) -> Optional[Dict[str, torch.Tensor]]:
        """Process a single example."""
        # Detect format and extract messages
        messages = self._extract_messages(example)

        if not messages:
            return None

        # Format conversation
        text, response_positions = self.formatter.format_messages(messages)

        # Tokenize with labels
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.config.max_seq_length,
            truncation=True,
            padding=False,
            return_tensors=None,
        )

        input_ids = encoding["input_ids"]

        # Create labels
        if self.config.mask_prompt and response_positions:
            labels = self._create_masked_labels(text, input_ids, response_positions)
        else:
            labels = input_ids.copy()

        # Shift for next-token prediction
        input_ids = input_ids[:-1]
        labels = labels[1:]

        attention_mask = [1] * len(input_ids)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

    def _extract_messages(self, example: Dict) -> List[Dict[str, str]]:
        """Extract messages from various formats."""
        # ShareGPT / OpenHermes format
        if "conversations" in example:
            messages = []
            for turn in example["conversations"]:
                role = turn.get("from", turn.get("role", "user"))
                content = turn.get("value", turn.get("content", ""))
                messages.append({"role": role, "content": content})
            return messages

        # Standard messages format
        if "messages" in example:
            return example["messages"]

        # Alpaca format
        if "instruction" in example:
            messages = [
                {
                    "role": "user",
                    "content": example.get("instruction", ""),
                    "input": example.get("input", ""),
                }
            ]
            if "output" in example or "response" in example:
                messages.append({
                    "role": "assistant",
                    "content": example.get("output", example.get("response", "")),
                })
            return messages

        # Prompt-response format
        if "prompt" in example and ("response" in example or "completion" in example):
            return [
                {"role": "user", "content": example["prompt"]},
                {"role": "assistant", "content": example.get("response", example.get("completion", ""))},
            ]

        # Text-only format
        if "text" in example:
            return [{"role": "assistant", "content": example["text"]}]

        return []

    def _create_masked_labels(
        self,
        text: str,
        input_ids: List[int],
        response_positions: List[Tuple[int, int]],
    ) -> List[int]:
        """Create labels with prompt tokens masked."""
        # Get character to token mapping
        char_to_token = []
        current_pos = 0

        for token_idx, token_id in enumerate(input_ids):
            token_str = self.tokenizer.decode([token_id])
            token_len = len(token_str)
            for _ in range(token_len):
                char_to_token.append(token_idx)
            current_pos += token_len

        # Fill in remaining positions
        while len(char_to_token) < len(text):
            char_to_token.append(len(input_ids) - 1)

        # Mark response tokens
        labels = [-100] * len(input_ids)

        for resp_start, resp_end in response_positions:
            # Find token range for this response
            start_token = char_to_token[min(resp_start, len(char_to_token) - 1)]
            end_token = char_to_token[min(resp_end - 1, len(char_to_token) - 1)]

            for token_idx in range(start_token, min(end_token + 1, len(labels))):
                labels[token_idx] = input_ids[token_idx]

        return labels

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        example = self.examples[idx]

        # Pad if not using packing
        if not self.config.use_packing:
            example = self._pad_example(example)

        return example

    def _pad_example(self, example: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Pad example to max_seq_length."""
        seq_len = len(example["input_ids"])
        pad_len = self.config.max_seq_length - 1 - seq_len  # -1 for shift

        if pad_len <= 0:
            return {
                "input_ids": example["input_ids"][:self.config.max_seq_length - 1],
                "attention_mask": example["attention_mask"][:self.config.max_seq_length - 1],
                "labels": example["labels"][:self.config.max_seq_length - 1],
            }

        return {
            "input_ids": torch.cat([
                example["input_ids"],
                torch.full((pad_len,), self.tokenizer.pad_token_id, dtype=torch.long)
            ]),
            "attention_mask": torch.cat([
                example["attention_mask"],
                torch.zeros(pad_len, dtype=torch.long)
            ]),
            "labels": torch.cat([
                example["labels"],
                torch.full((pad_len,), -100, dtype=torch.long)
            ]),
        }


class StreamingInstructionDataset(IterableDataset):
    """
    Streaming instruction dataset for large-scale training.

    Features:
    - Memory-efficient streaming
    - Multi-worker support with proper sharding
    - On-the-fly preprocessing
    - Shuffling buffer
    """

    def __init__(
        self,
        dataset_paths: List[str],
        tokenizer: Any,
        config: SFTDataConfig,
        shuffle: bool = True,
    ):
        self.dataset_paths = dataset_paths
        self.tokenizer = tokenizer
        self.config = config
        self.shuffle = shuffle

        self.formatter = ConversationFormatter(
            template_name=config.template,
            tokenizer=tokenizer,
            max_seq_length=config.max_seq_length,
        )

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        worker_info = get_worker_info()

        if worker_info is None:
            # Single-process loading
            worker_id = 0
            num_workers = 1
        else:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers

        # Shard datasets across workers
        worker_datasets = self.dataset_paths[worker_id::num_workers]

        # Create shuffling buffer
        buffer = []

        for dataset_path in worker_datasets:
            for example in self._stream_dataset(dataset_path):
                processed = self._process_example(example)
                if processed is not None:
                    if self.shuffle:
                        buffer.append(processed)
                        if len(buffer) >= self.config.buffer_size:
                            random.shuffle(buffer)
                            while len(buffer) > self.config.buffer_size // 2:
                                yield buffer.pop()
                    else:
                        yield processed

        # Yield remaining examples
        if self.shuffle:
            random.shuffle(buffer)
        for example in buffer:
            yield example

    def _stream_dataset(self, path: str) -> Iterator[Dict]:
        """Stream examples from a dataset."""
        try:
            from datasets import load_dataset
            ds = load_dataset(path, split="train", streaming=True)
            for example in ds:
                yield example
        except Exception:
            # Try local file
            path_obj = Path(path)
            if path_obj.exists():
                if path_obj.suffix == ".jsonl":
                    with open(path_obj, 'r') as f:
                        for line in f:
                            yield json.loads(line)

    def _process_example(self, example: Dict) -> Optional[Dict[str, torch.Tensor]]:
        """Process a single example."""
        # Similar to InstructionDataset._process_example
        messages = self._extract_messages(example)

        if not messages:
            return None

        text, response_positions = self.formatter.format_messages(messages)

        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.config.max_seq_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)

        # Create labels (mask prompt)
        labels = input_ids.clone()
        if self.config.mask_prompt:
            # Simplified: mask all non-pad tokens up to response
            labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": input_ids[:-1],
            "attention_mask": attention_mask[:-1],
            "labels": labels[1:],
        }

    def _extract_messages(self, example: Dict) -> List[Dict[str, str]]:
        """Extract messages from example."""
        if "conversations" in example:
            return [
                {"role": t.get("from", "user"), "content": t.get("value", "")}
                for t in example["conversations"]
            ]
        if "messages" in example:
            return example["messages"]
        if "instruction" in example:
            msgs = [{"role": "user", "content": example.get("instruction", "")}]
            if "output" in example:
                msgs.append({"role": "assistant", "content": example["output"]})
            return msgs
        return []


# =============================================================================
# DATA COLLATOR
# =============================================================================

class SFTDataCollator:
    """
    Data collator for SFT with proper padding and batching.

    Handles:
    - Dynamic padding to batch max length
    - Position IDs for packed sequences
    - Attention mask creation
    """

    def __init__(
        self,
        tokenizer: Any,
        pad_to_multiple_of: int = 8,
        max_seq_length: Optional[int] = None,
    ):
        self.tokenizer = tokenizer
        self.pad_to_multiple_of = pad_to_multiple_of
        self.max_seq_length = max_seq_length
        self.pad_token_id = tokenizer.pad_token_id

    def __call__(
        self,
        batch: List[Dict[str, torch.Tensor]],
    ) -> Dict[str, torch.Tensor]:
        """Collate batch of examples."""
        # Find max length in batch
        max_len = max(len(example["input_ids"]) for example in batch)

        # Round up to multiple
        if self.pad_to_multiple_of:
            max_len = ((max_len + self.pad_to_multiple_of - 1)
                       // self.pad_to_multiple_of * self.pad_to_multiple_of)

        # Optionally cap at max_seq_length
        if self.max_seq_length:
            max_len = min(max_len, self.max_seq_length)

        # Pad and stack
        input_ids = []
        attention_mask = []
        labels = []
        position_ids = []

        for example in batch:
            seq_len = len(example["input_ids"])
            pad_len = max_len - seq_len

            input_ids.append(torch.cat([
                example["input_ids"][:max_len],
                torch.full((max(pad_len, 0),), self.pad_token_id, dtype=torch.long)
            ])[:max_len])

            attention_mask.append(torch.cat([
                example["attention_mask"][:max_len],
                torch.zeros(max(pad_len, 0), dtype=torch.long)
            ])[:max_len])

            labels.append(torch.cat([
                example["labels"][:max_len],
                torch.full((max(pad_len, 0),), -100, dtype=torch.long)
            ])[:max_len])

            # Handle position IDs if present (for packed sequences)
            if "position_ids" in example:
                position_ids.append(torch.cat([
                    example["position_ids"][:max_len],
                    torch.zeros(max(pad_len, 0), dtype=torch.long)
                ])[:max_len])

        result = {
            "input_ids": torch.stack(input_ids),
            "attention_mask": torch.stack(attention_mask),
            "labels": torch.stack(labels),
        }

        if position_ids:
            result["position_ids"] = torch.stack(position_ids)

        return result


# =============================================================================
# DATASET CREATION UTILITIES
# =============================================================================

def create_sft_datasets(
    config: SFTDataConfig,
    tokenizer: Any,
) -> Tuple[Dataset, Optional[Dataset]]:
    """
    Create train and validation SFT datasets.

    Args:
        config: SFT data configuration
        tokenizer: Tokenizer instance

    Returns:
        Tuple of (train_dataset, eval_dataset)
    """
    all_data = []

    for dataset_name in config.datasets:
        try:
            if config.streaming:
                # Use streaming dataset
                train_ds = StreamingInstructionDataset(
                    [dataset_name],
                    tokenizer,
                    config,
                )
                return train_ds, None
            else:
                data = _load_dataset(dataset_name, config)

                # Apply weight
                if config.dataset_weights:
                    weight = config.dataset_weights.get(dataset_name, 1.0)
                    if weight < 1.0:
                        n_samples = int(len(data) * weight)
                        random.seed(config.seed)
                        data = random.sample(data, n_samples)

                # Apply sample limit
                if config.max_samples_per_dataset:
                    data = data[:config.max_samples_per_dataset]

                all_data.extend(data)
                logger.info(f"Loaded {len(data)} examples from {dataset_name}")

        except Exception as e:
            logger.warning(f"Failed to load {dataset_name}: {e}")

    # Split into train/val
    random.seed(config.seed)
    random.shuffle(all_data)

    val_size = int(len(all_data) * config.validation_split)
    train_data = all_data[val_size:]
    val_data = all_data[:val_size] if val_size > 0 else None

    train_config = SFTDataConfig(**config.to_dict())
    val_config = SFTDataConfig(**config.to_dict())
    val_config.use_packing = False  # Don't pack validation

    train_ds = InstructionDataset(train_data, tokenizer, train_config, split="train")
    val_ds = InstructionDataset(val_data, tokenizer, val_config, split="val") if val_data else None

    return train_ds, val_ds


def _load_dataset(dataset_name: str, config: SFTDataConfig) -> List[Dict]:
    """Load a dataset by name."""
    try:
        from datasets import load_dataset
        ds = load_dataset(dataset_name, split="train")
        return list(ds)
    except Exception as e:
        raise ValueError(f"Could not load {dataset_name}: {e}")


# =============================================================================
# SFT EVALUATION BENCHMARKS
# =============================================================================

SFT_EVAL_BENCHMARKS = {
    "mt_bench": {
        "description": "Multi-turn conversation benchmark",
        "hf_path": "lmsys/mt_bench_human_judgments",
        "metric": "rating",
        "categories": ["writing", "roleplay", "reasoning", "math", "coding", "extraction", "stem", "humanities"],
    },
    "alpaca_eval": {
        "description": "Alpaca evaluation for instruction following",
        "hf_path": "tatsu-lab/alpaca_eval",
        "metric": "win_rate",
    },
    "ifeval": {
        "description": "Instruction Following Evaluation",
        "hf_path": "google/IFEval",
        "metric": "accuracy",
        "constraint_types": ["length", "format", "content", "count"],
    },
    "arena_hard": {
        "description": "Arena Hard benchmark from LMSYS",
        "hf_path": "lmsys/arena-hard-v0.1",
        "metric": "win_rate",
    },
}
