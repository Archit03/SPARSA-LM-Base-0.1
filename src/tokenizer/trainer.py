"""
SPARSA-LM Tokenizer Training
BPE and SentencePiece tokenizer training with evaluation metrics
"""

import os
import json
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Iterator, Any, Union
from pathlib import Path
from collections import Counter
import math

logger = logging.getLogger(__name__)


@dataclass
class TokenizerConfig:
    """
    Configuration for tokenizer training.

    Architecture Options:
    - bpe: Byte-Pair Encoding (HuggingFace tokenizers)
    - unigram: SentencePiece Unigram
    - bpe_sp: SentencePiece BPE

    SmolLM-360M uses BPE with 49,152 vocab size.
    LLaMA uses SentencePiece BPE with 32,000 vocab size.
    """

    # Architecture
    architecture: str = "bpe"  # "bpe", "unigram", "bpe_sp"

    # Vocabulary
    vocab_size: int = 32000
    min_frequency: int = 2

    # Special Tokens
    special_tokens: List[str] = field(default_factory=lambda: [
        "<pad>",    # Padding token
        "<s>",      # Beginning of sequence
        "</s>",     # End of sequence
        "<unk>",    # Unknown token
        "<mask>",   # Mask token (optional)
    ])

    # Token IDs
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2
    unk_token_id: int = 3

    # BPE-specific
    byte_fallback: bool = True
    add_prefix_space: bool = False
    trim_offsets: bool = True

    # SentencePiece-specific
    character_coverage: float = 0.9999
    model_type: str = "bpe"  # "bpe", "unigram", "char", "word"
    normalization_rule_name: str = "identity"
    add_dummy_prefix: bool = True
    remove_extra_whitespaces: bool = True
    split_by_unicode_script: bool = True
    split_by_whitespace: bool = True
    split_by_number: bool = True
    split_digits: bool = True
    treat_whitespace_as_suffix: bool = False

    # Training
    max_input_chars_per_word: int = 100
    continuing_subword_prefix: str = ""
    end_of_word_suffix: str = ""

    # Dataset configuration
    training_datasets: List[str] = field(default_factory=lambda: [
        "wikipedia:20220301.en",
        "Skylion007/openwebtext",
        "bookcorpusopen",
        "bigcode/the-stack-v2:python",
        "arxiv_dataset",
    ])
    max_samples_per_dataset: int = 1_000_000
    text_column: str = "text"

    # Output
    output_dir: str = "tokenizer"
    save_pretrained_format: bool = True

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}

    def save(self, path: str):
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "TokenizerConfig":
        with open(path, 'r') as f:
            return cls(**json.load(f))

    @classmethod
    def smollm(cls) -> "TokenizerConfig":
        """SmolLM-style tokenizer configuration."""
        return cls(
            architecture="bpe",
            vocab_size=49152,
            special_tokens=["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<pad>"],
        )

    @classmethod
    def llama(cls) -> "TokenizerConfig":
        """LLaMA-style tokenizer configuration."""
        return cls(
            architecture="bpe_sp",
            vocab_size=32000,
            character_coverage=0.9999,
            model_type="bpe",
        )


class TokenizerTrainer:
    """
    Tokenizer trainer supporting BPE and SentencePiece.

    Training Pipeline:
    1. Load and prepare training data from multiple sources
    2. Train tokenizer with specified architecture
    3. Evaluate tokenizer quality
    4. Save in HuggingFace format
    """

    def __init__(self, config: TokenizerConfig):
        self.config = config
        self.tokenizer = None

    def prepare_training_data(self) -> Iterator[str]:
        """Load and prepare training data from configured datasets."""
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("Please install datasets: pip install datasets")

        for dataset_spec in self.config.training_datasets:
            # Parse dataset specification (name:subset format)
            if ":" in dataset_spec:
                parts = dataset_spec.split(":")
                dataset_name = parts[0]
                subset = parts[1] if len(parts) > 1 else None
            else:
                dataset_name = dataset_spec
                subset = None

            logger.info(f"Loading dataset: {dataset_name}" + (f" ({subset})" if subset else ""))

            try:
                if subset:
                    ds = load_dataset(dataset_name, subset, split="train", streaming=True)
                else:
                    ds = load_dataset(dataset_name, split="train", streaming=True)

                count = 0
                for example in ds:
                    # Extract text from various column names
                    text = None
                    for col in [self.config.text_column, "text", "content", "document"]:
                        if col in example and example[col]:
                            text = example[col]
                            break

                    if text and len(text) > 10:
                        yield text
                        count += 1

                        if count >= self.config.max_samples_per_dataset:
                            break

                logger.info(f"  Loaded {count} samples from {dataset_name}")

            except Exception as e:
                logger.warning(f"Failed to load {dataset_name}: {e}")
                continue

    def train_bpe(self) -> Any:
        """Train BPE tokenizer using HuggingFace tokenizers."""
        try:
            from tokenizers import Tokenizer, models, trainers, pre_tokenizers, processors, decoders
            from tokenizers.normalizers import NFKC, Sequence, Lowercase, Strip
        except ImportError:
            raise ImportError("Please install tokenizers: pip install tokenizers")

        logger.info("Training BPE tokenizer...")

        # Initialize tokenizer
        tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))

        # Normalizer
        tokenizer.normalizer = Sequence([NFKC(), Strip()])

        # Pre-tokenizer
        if self.config.byte_fallback:
            tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=self.config.add_prefix_space)
        else:
            tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

        # Trainer
        trainer = trainers.BpeTrainer(
            vocab_size=self.config.vocab_size,
            min_frequency=self.config.min_frequency,
            special_tokens=self.config.special_tokens,
            show_progress=True,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet() if self.config.byte_fallback else [],
        )

        # Train
        def data_iterator():
            for text in self.prepare_training_data():
                yield text

        tokenizer.train_from_iterator(data_iterator(), trainer=trainer)

        # Post-processor
        tokenizer.post_processor = processors.ByteLevel(trim_offsets=self.config.trim_offsets)

        # Decoder
        tokenizer.decoder = decoders.ByteLevel()

        self.tokenizer = tokenizer
        return tokenizer

    def train_sentencepiece(self) -> Any:
        """Train SentencePiece tokenizer."""
        try:
            import sentencepiece as spm
            import tempfile
        except ImportError:
            raise ImportError("Please install sentencepiece: pip install sentencepiece")

        logger.info("Training SentencePiece tokenizer...")

        # Write training data to temp file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            temp_path = f.name
            for text in self.prepare_training_data():
                f.write(text + "\n")

        # Train SentencePiece
        output_prefix = os.path.join(self.config.output_dir, "tokenizer")
        os.makedirs(self.config.output_dir, exist_ok=True)

        spm.SentencePieceTrainer.train(
            input=temp_path,
            model_prefix=output_prefix,
            vocab_size=self.config.vocab_size,
            character_coverage=self.config.character_coverage,
            model_type=self.config.model_type,
            pad_id=self.config.pad_token_id,
            bos_id=self.config.bos_token_id,
            eos_id=self.config.eos_token_id,
            unk_id=self.config.unk_token_id,
            normalization_rule_name=self.config.normalization_rule_name,
            add_dummy_prefix=self.config.add_dummy_prefix,
            remove_extra_whitespaces=self.config.remove_extra_whitespaces,
            split_by_unicode_script=self.config.split_by_unicode_script,
            split_by_whitespace=self.config.split_by_whitespace,
            split_by_number=self.config.split_by_number,
            split_digits=self.config.split_digits,
            treat_whitespace_as_suffix=self.config.treat_whitespace_as_suffix,
        )

        # Clean up
        os.unlink(temp_path)

        # Load trained model
        sp = spm.SentencePieceProcessor()
        sp.Load(f"{output_prefix}.model")

        self.tokenizer = sp
        return sp

    def train(self) -> Any:
        """Train tokenizer based on configured architecture."""
        if self.config.architecture == "bpe":
            return self.train_bpe()
        elif self.config.architecture in ["unigram", "bpe_sp"]:
            return self.train_sentencepiece()
        else:
            raise ValueError(f"Unknown architecture: {self.config.architecture}")

    def save(self, path: Optional[str] = None):
        """Save tokenizer to disk."""
        path = path or self.config.output_dir
        os.makedirs(path, exist_ok=True)

        if self.config.architecture == "bpe":
            self._save_bpe(path)
        else:
            self._save_sentencepiece(path)

        # Save config
        self.config.save(os.path.join(path, "tokenizer_config.json"))
        logger.info(f"Saved tokenizer to {path}")

    def _save_bpe(self, path: str):
        """Save BPE tokenizer in HuggingFace format."""
        # Save tokenizer.json
        self.tokenizer.save(os.path.join(path, "tokenizer.json"))

        # Create tokenizer_config.json for HuggingFace compatibility
        hf_config = {
            "tokenizer_class": "PreTrainedTokenizerFast",
            "bos_token": self.config.special_tokens[1] if len(self.config.special_tokens) > 1 else "<s>",
            "eos_token": self.config.special_tokens[2] if len(self.config.special_tokens) > 2 else "</s>",
            "unk_token": self.config.special_tokens[3] if len(self.config.special_tokens) > 3 else "<unk>",
            "pad_token": self.config.special_tokens[0] if len(self.config.special_tokens) > 0 else "<pad>",
            "model_max_length": 2048,
            "clean_up_tokenization_spaces": True,
        }
        with open(os.path.join(path, "special_tokens_map.json"), 'w') as f:
            json.dump(hf_config, f, indent=2)

    def _save_sentencepiece(self, path: str):
        """Save SentencePiece tokenizer."""
        # SentencePiece model is already saved during training
        # Create HuggingFace compatibility files
        pass


class TokenizerEvaluator:
    """
    Tokenizer evaluation metrics.

    Metrics:
    1. Fertility: Average tokens per word
    2. Coverage: Percentage of words covered without UNK
    3. Compression ratio: Characters per token
    4. Vocabulary utilization: Used vocab / total vocab
    5. Subword regularity: Consistency of tokenization
    """

    def __init__(self, tokenizer: Any, config: TokenizerConfig):
        self.tokenizer = tokenizer
        self.config = config
        self.is_bpe = config.architecture == "bpe"

    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs."""
        if self.is_bpe:
            return self.tokenizer.encode(text).ids
        else:
            return self.tokenizer.encode(text)

    def decode(self, ids: List[int]) -> str:
        """Decode token IDs to text."""
        if self.is_bpe:
            return self.tokenizer.decode(ids)
        else:
            return self.tokenizer.decode(ids)

    def evaluate(self, eval_texts: List[str]) -> Dict[str, float]:
        """
        Evaluate tokenizer on a set of texts.

        Returns:
            Dictionary with evaluation metrics
        """
        metrics = {
            "fertility": 0.0,
            "coverage": 0.0,
            "compression_ratio": 0.0,
            "avg_token_length": 0.0,
            "unk_rate": 0.0,
            "vocab_utilization": 0.0,
        }

        total_tokens = 0
        total_words = 0
        total_chars = 0
        total_unk = 0
        used_tokens = set()

        unk_id = self.config.unk_token_id

        for text in eval_texts:
            # Tokenize
            token_ids = self.encode(text)

            # Count statistics
            total_tokens += len(token_ids)
            total_words += len(text.split())
            total_chars += len(text)

            # Track UNK tokens
            total_unk += sum(1 for t in token_ids if t == unk_id)

            # Track vocabulary usage
            used_tokens.update(token_ids)

        # Compute metrics
        if total_words > 0:
            metrics["fertility"] = total_tokens / total_words

        if total_tokens > 0:
            metrics["unk_rate"] = total_unk / total_tokens
            metrics["coverage"] = 1.0 - metrics["unk_rate"]

        if total_tokens > 0:
            metrics["compression_ratio"] = total_chars / total_tokens
            metrics["avg_token_length"] = total_chars / total_tokens

        metrics["vocab_utilization"] = len(used_tokens) / self.config.vocab_size

        return metrics

    def evaluate_on_domains(self, domain_texts: Dict[str, List[str]]) -> Dict[str, Dict[str, float]]:
        """
        Evaluate tokenizer across different domains.

        Args:
            domain_texts: Dictionary mapping domain names to text lists

        Returns:
            Dictionary mapping domain names to metrics
        """
        results = {}
        for domain, texts in domain_texts.items():
            results[domain] = self.evaluate(texts)
        return results

    def compute_subword_regularity(self, words: List[str]) -> float:
        """
        Compute subword regularity: consistency of tokenization.

        Higher values indicate more consistent tokenization patterns.
        """
        tokenizations = {}
        for word in words:
            tokens = tuple(self.encode(word))
            if tokens not in tokenizations:
                tokenizations[tokens] = 0
            tokenizations[tokens] += 1

        # Compute entropy
        total = sum(tokenizations.values())
        entropy = -sum(
            (c / total) * math.log2(c / total)
            for c in tokenizations.values()
            if c > 0
        )

        # Normalize by max entropy
        max_entropy = math.log2(len(words)) if len(words) > 1 else 1
        regularity = 1.0 - (entropy / max_entropy) if max_entropy > 0 else 1.0

        return regularity

    def print_report(self, eval_texts: List[str]):
        """Print formatted evaluation report."""
        metrics = self.evaluate(eval_texts)

        print("=" * 60)
        print("TOKENIZER EVALUATION REPORT")
        print("=" * 60)
        print(f"\nConfiguration:")
        print(f"  Architecture: {self.config.architecture}")
        print(f"  Vocabulary Size: {self.config.vocab_size:,}")
        print(f"\nMetrics:")
        print(f"  Fertility (tokens/word): {metrics['fertility']:.2f}")
        print(f"  Coverage (1 - UNK rate): {metrics['coverage']:.4f}")
        print(f"  Compression (chars/token): {metrics['compression_ratio']:.2f}")
        print(f"  Avg Token Length: {metrics['avg_token_length']:.2f}")
        print(f"  UNK Rate: {metrics['unk_rate']:.6f}")
        print(f"  Vocabulary Utilization: {metrics['vocab_utilization']:.4f}")
        print("=" * 60)


# =============================================================================
# TOKENIZER TRAINING HYPERPARAMETERS
# =============================================================================

TOKENIZER_TRAINING_CONFIG = {
    "smollm_360m": {
        "architecture": "bpe",
        "vocab_size": 49152,
        "min_frequency": 2,
        "byte_fallback": True,
        "training_datasets": [
            "HuggingFaceTB/cosmopedia-v2",
            "HuggingFaceFW/fineweb-edu:sample-10BT",
            "bigcode/the-stack-v2:python",
        ],
        "max_samples_per_dataset": 2_000_000,
        "target_fertility": 1.5,  # Target tokens per word
        "target_compression": 4.0,  # Target chars per token
    },
    "llama_style": {
        "architecture": "bpe_sp",
        "vocab_size": 32000,
        "character_coverage": 0.9999,
        "model_type": "bpe",
        "training_datasets": [
            "wikipedia:20220301.en",
            "Skylion007/openwebtext",
            "bookcorpusopen",
            "allenai/c4:en",
        ],
        "max_samples_per_dataset": 1_000_000,
        "target_fertility": 1.3,
        "target_compression": 4.5,
    },
    "code_focused": {
        "architecture": "bpe",
        "vocab_size": 32768,
        "min_frequency": 3,
        "byte_fallback": True,
        "training_datasets": [
            "bigcode/the-stack-v2:python",
            "bigcode/the-stack-v2:javascript",
            "bigcode/the-stack-v2:java",
            "bigcode/the-stack-v2:cpp",
            "bigcode/the-stack-v2:go",
        ],
        "max_samples_per_dataset": 500_000,
        "target_fertility": 2.0,  # Code has more tokens per word
        "target_compression": 3.5,
    },
}


# =============================================================================
# EVALUATION METRICS TARGETS
# =============================================================================

TOKENIZER_QUALITY_TARGETS = {
    "fertility": {
        "excellent": (1.0, 1.5),
        "good": (1.5, 2.0),
        "acceptable": (2.0, 2.5),
        "poor": (2.5, float('inf')),
    },
    "coverage": {
        "excellent": (0.999, 1.0),
        "good": (0.995, 0.999),
        "acceptable": (0.99, 0.995),
        "poor": (0.0, 0.99),
    },
    "compression_ratio": {
        "excellent": (4.0, 5.0),
        "good": (3.5, 4.0),
        "acceptable": (3.0, 3.5),
        "poor": (0.0, 3.0),
    },
    "vocab_utilization": {
        "excellent": (0.8, 1.0),
        "good": (0.6, 0.8),
        "acceptable": (0.4, 0.6),
        "poor": (0.0, 0.4),
    },
}


def grade_metric(metric_name: str, value: float) -> str:
    """Grade a metric value."""
    if metric_name not in TOKENIZER_QUALITY_TARGETS:
        return "unknown"

    targets = TOKENIZER_QUALITY_TARGETS[metric_name]
    for grade, (low, high) in targets.items():
        if low <= value < high:
            return grade
    return "unknown"


if __name__ == "__main__":
    # Example usage
    config = TokenizerConfig.llama()
    trainer = TokenizerTrainer(config)

    print("Tokenizer Training Configuration:")
    print(json.dumps(config.to_dict(), indent=2))
