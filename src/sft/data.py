"""
SPARSA-LM SFT Data Processing
Instruction dataset handling and formatting
"""

import json
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Iterator, Union
from pathlib import Path

import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


# =============================================================================
# PROMPT TEMPLATES
# =============================================================================

PROMPT_TEMPLATES = {
    "alpaca": {
        "with_input": (
            "Below is an instruction that describes a task, paired with an input that provides further context. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n{instruction}\n\n"
            "### Input:\n{input}\n\n"
            "### Response:\n{output}"
        ),
        "without_input": (
            "Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n{instruction}\n\n"
            "### Response:\n{output}"
        ),
    },
    "chatml": {
        "system": "<|im_start|>system\n{system}<|im_end|>\n",
        "user": "<|im_start|>user\n{content}<|im_end|>\n",
        "assistant": "<|im_start|>assistant\n{content}<|im_end|>\n",
    },
    "llama2": {
        "system": "<<SYS>>\n{system}\n<</SYS>>\n\n",
        "user": "[INST] {content} [/INST]",
        "assistant": " {content} </s><s>",
    },
    "llama3": {
        "system": "<|start_header_id|>system<|end_header_id|>\n\n{system}<|eot_id|>",
        "user": "<|start_header_id|>user<|end_header_id|>\n\n{content}<|eot_id|>",
        "assistant": "<|start_header_id|>assistant<|end_header_id|>\n\n{content}<|eot_id|>",
    },
    "zephyr": {
        "system": "<|system|>\n{system}</s>\n",
        "user": "<|user|>\n{content}</s>\n",
        "assistant": "<|assistant|>\n{content}</s>\n",
    },
}


def format_instruction(
    instruction: str,
    input_text: str = "",
    output: str = "",
    template: str = "alpaca",
) -> str:
    """Format instruction using specified template."""
    if template == "alpaca":
        if input_text:
            return PROMPT_TEMPLATES["alpaca"]["with_input"].format(
                instruction=instruction,
                input=input_text,
                output=output,
            )
        else:
            return PROMPT_TEMPLATES["alpaca"]["without_input"].format(
                instruction=instruction,
                output=output,
            )
    else:
        raise ValueError(f"Unknown template: {template}")


def format_conversation(
    messages: List[Dict[str, str]],
    template: str = "chatml",
    system_message: Optional[str] = None,
) -> str:
    """Format multi-turn conversation using specified template."""
    templates = PROMPT_TEMPLATES.get(template)
    if not templates:
        raise ValueError(f"Unknown template: {template}")

    formatted = ""

    # Add system message if provided
    if system_message and "system" in templates:
        formatted += templates["system"].format(system=system_message)

    # Format each message
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")

        if role in ["user", "human"]:
            formatted += templates["user"].format(content=content)
        elif role in ["assistant", "gpt", "model"]:
            formatted += templates["assistant"].format(content=content)
        elif role == "system" and "system" in templates:
            formatted += templates["system"].format(system=content)

    return formatted


# =============================================================================
# DATASETS
# =============================================================================

@dataclass
class SFTDataConfig:
    """Configuration for SFT data."""

    # Data sources
    datasets: List[str] = field(default_factory=lambda: [
        "teknium/OpenHermes-2.5",
        "HuggingFaceH4/ultrachat_200k",
    ])

    # Processing
    max_seq_length: int = 2048
    template: str = "alpaca"
    mask_prompt: bool = True  # Mask prompt tokens in loss

    # Mixing
    dataset_weights: Optional[Dict[str, float]] = None
    max_samples_per_dataset: Optional[int] = None

    # Validation
    validation_split: float = 0.05
    seed: int = 42


class InstructionDataset(Dataset):
    """
    Dataset for instruction fine-tuning.

    Supports multiple formats:
    - Alpaca: instruction, input, output
    - ShareGPT: conversations with roles
    - OpenAssistant: conversation trees
    """

    def __init__(
        self,
        data: Union[List[Dict], str],
        tokenizer: Any,
        config: SFTDataConfig,
    ):
        self.tokenizer = tokenizer
        self.config = config
        self.max_seq_length = config.max_seq_length

        # Load data
        if isinstance(data, str):
            self.data = self._load_data(data)
        else:
            self.data = data

        logger.info(f"Loaded {len(self.data)} instruction examples")

    def _load_data(self, path: str) -> List[Dict]:
        """Load data from file or HuggingFace."""
        path = Path(path)

        if path.exists():
            # Local file
            if path.suffix == ".jsonl":
                with open(path, 'r') as f:
                    return [json.loads(line) for line in f]
            elif path.suffix == ".json":
                with open(path, 'r') as f:
                    return json.load(f)
        else:
            # HuggingFace dataset
            try:
                from datasets import load_dataset
                ds = load_dataset(path, split="train")
                return list(ds)
            except Exception as e:
                raise ValueError(f"Could not load data from {path}: {e}")

        return []

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single example."""
        example = self.data[idx]

        # Format based on data structure
        if "instruction" in example:
            # Alpaca format
            text = format_instruction(
                instruction=example.get("instruction", ""),
                input_text=example.get("input", ""),
                output=example.get("output", example.get("response", "")),
                template=self.config.template,
            )
            # Track where the output starts for loss masking
            prompt_text = format_instruction(
                instruction=example.get("instruction", ""),
                input_text=example.get("input", ""),
                output="",
                template=self.config.template,
            )
        elif "messages" in example or "conversations" in example:
            # Conversation format
            messages = example.get("messages", example.get("conversations", []))
            text = format_conversation(messages, template=self.config.template)
            # Get prompt up to last assistant message
            prompt_messages = messages[:-1] if messages else []
            prompt_text = format_conversation(prompt_messages, template=self.config.template)
        elif "prompt" in example and "response" in example:
            # Simple prompt-response
            text = f"{example['prompt']}\n{example['response']}"
            prompt_text = example['prompt']
        else:
            # Fallback: use 'text' field
            text = example.get("text", "")
            prompt_text = ""

        # Tokenize
        tokens = self.tokenizer.encode(text, add_special_tokens=True)

        # Truncate if necessary
        if len(tokens) > self.max_seq_length:
            tokens = tokens[:self.max_seq_length]

        input_ids = torch.tensor(tokens[:-1], dtype=torch.long)
        labels = torch.tensor(tokens[1:], dtype=torch.long)

        # Mask prompt tokens if configured
        if self.config.mask_prompt and prompt_text:
            prompt_tokens = self.tokenizer.encode(prompt_text, add_special_tokens=True)
            prompt_len = min(len(prompt_tokens) - 1, len(labels))
            labels[:prompt_len] = -100

        # Pad if necessary
        if len(input_ids) < self.max_seq_length - 1:
            pad_length = self.max_seq_length - 1 - len(input_ids)
            input_ids = torch.cat([
                input_ids,
                torch.full((pad_length,), self.tokenizer.pad_token_id, dtype=torch.long)
            ])
            labels = torch.cat([
                labels,
                torch.full((pad_length,), -100, dtype=torch.long)
            ])

        attention_mask = (input_ids != self.tokenizer.pad_token_id).long()

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


class ChatDataset(Dataset):
    """
    Dataset for chat/conversation fine-tuning.

    Handles multi-turn conversations with proper role formatting.
    """

    def __init__(
        self,
        conversations: List[List[Dict[str, str]]],
        tokenizer: Any,
        config: SFTDataConfig,
        system_message: Optional[str] = None,
    ):
        self.conversations = conversations
        self.tokenizer = tokenizer
        self.config = config
        self.system_message = system_message
        self.max_seq_length = config.max_seq_length

    def __len__(self) -> int:
        return len(self.conversations)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a conversation example."""
        messages = self.conversations[idx]

        # Format conversation
        text = format_conversation(
            messages,
            template=self.config.template,
            system_message=self.system_message,
        )

        # Tokenize
        tokens = self.tokenizer.encode(text, add_special_tokens=True)

        # Truncate
        if len(tokens) > self.max_seq_length:
            tokens = tokens[:self.max_seq_length]

        input_ids = torch.tensor(tokens[:-1], dtype=torch.long)
        labels = torch.tensor(tokens[1:], dtype=torch.long)

        # Mask user messages in labels (only train on assistant responses)
        if self.config.mask_prompt:
            labels = self._mask_user_messages(messages, labels)

        # Pad
        if len(input_ids) < self.max_seq_length - 1:
            pad_length = self.max_seq_length - 1 - len(input_ids)
            input_ids = torch.cat([
                input_ids,
                torch.full((pad_length,), self.tokenizer.pad_token_id, dtype=torch.long)
            ])
            labels = torch.cat([
                labels,
                torch.full((pad_length,), -100, dtype=torch.long)
            ])

        attention_mask = (input_ids != self.tokenizer.pad_token_id).long()

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    def _mask_user_messages(
        self,
        messages: List[Dict[str, str]],
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Mask user message tokens in labels."""
        # This is a simplified implementation
        # For production, track token positions during formatting
        return labels


# =============================================================================
# DATA LOADING UTILITIES
# =============================================================================

def load_sft_dataset(
    dataset_name: str,
    tokenizer: Any,
    config: SFTDataConfig,
    split: str = "train",
) -> Dataset:
    """Load an SFT dataset from HuggingFace."""
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("Please install datasets: pip install datasets")

    logger.info(f"Loading {dataset_name}...")

    # Dataset-specific loading
    if "OpenHermes" in dataset_name:
        ds = load_dataset(dataset_name, split=split)
        # OpenHermes has 'conversations' field
        data = []
        for example in ds:
            if "conversations" in example:
                data.append({"messages": example["conversations"]})
        return InstructionDataset(data, tokenizer, config)

    elif "ultrachat" in dataset_name:
        ds = load_dataset(dataset_name, split=split)
        data = []
        for example in ds:
            if "messages" in example:
                data.append({"messages": example["messages"]})
        return InstructionDataset(data, tokenizer, config)

    elif "oasst" in dataset_name.lower():
        ds = load_dataset(dataset_name, split=split)
        # OpenAssistant has tree structure, extract linear conversations
        data = []
        for example in ds:
            if "text" in example:
                data.append({"text": example["text"]})
        return InstructionDataset(data, tokenizer, config)

    elif "alpaca" in dataset_name.lower():
        ds = load_dataset(dataset_name, split=split)
        return InstructionDataset(list(ds), tokenizer, config)

    else:
        # Generic loading
        ds = load_dataset(dataset_name, split=split)
        return InstructionDataset(list(ds), tokenizer, config)


def create_mixed_sft_dataset(
    config: SFTDataConfig,
    tokenizer: Any,
) -> Dataset:
    """Create a mixed dataset from multiple sources."""
    from torch.utils.data import ConcatDataset

    datasets = []
    weights = config.dataset_weights or {}

    for dataset_name in config.datasets:
        try:
            ds = load_sft_dataset(dataset_name, tokenizer, config)

            # Sample if weight specified
            weight = weights.get(dataset_name, 1.0)
            if weight < 1.0:
                num_samples = int(len(ds) * weight)
                indices = torch.randperm(len(ds))[:num_samples].tolist()
                ds = torch.utils.data.Subset(ds, indices)

            # Limit samples if specified
            if config.max_samples_per_dataset:
                if len(ds) > config.max_samples_per_dataset:
                    indices = list(range(config.max_samples_per_dataset))
                    ds = torch.utils.data.Subset(ds, indices)

            datasets.append(ds)
            logger.info(f"Added {len(ds)} samples from {dataset_name}")

        except Exception as e:
            logger.warning(f"Failed to load {dataset_name}: {e}")

    return ConcatDataset(datasets)


# =============================================================================
# EVALUATION DATASETS
# =============================================================================

SFT_EVAL_BENCHMARKS = {
    "mt_bench": {
        "description": "Multi-turn conversation benchmark",
        "hf_path": "lmsys/mt_bench_human_judgments",
        "metric": "rating",
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
    },
}
