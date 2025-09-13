#!/usr/bin/env python3
"""
SPARSA-LM BPE Tokenizer
Production BPE tokenizer for HuggingFace deployment.
"""

import os
import yaml
import logging
import argparse
from pathlib import Path
from typing import Iterator
from tqdm import tqdm

from datasets import load_dataset
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers, processors
from transformers import PreTrainedTokenizerFast

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class BPETokenizer:
    """BPE tokenizer trainer for SPARSA-LM."""
    
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        self.vocab_size = config['tokenizer']['vocab_size']
        self.min_frequency = config['tokenizer']['min_frequency']
        self.special_tokens = config['tokenizer']['special_tokens']
        self.datasets = config['datasets']
        self.output_dir = Path(config['output_dir'])
        self.cache_dir = config.get('cache_dir', '.cache')
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize BPE tokenizer
        self.tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
        self.tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
        self.tokenizer.decoder = decoders.ByteLevel()
        
        logger.info(f"Initialized BPE tokenizer (vocab_size: {self.vocab_size})")
    
    def load_text_data(self) -> Iterator[str]:
        """Load text data from configured datasets."""
        total_samples = 0
        
        for dataset_config in self.datasets:
            name = dataset_config['name']
            subset = dataset_config.get('subset')
            split = dataset_config.get('split', 'train')
            text_column = dataset_config.get('text_column', 'text')
            max_samples = dataset_config.get('max_samples', 50000)
            
            logger.info(f"Loading {name}" + (f"/{subset}" if subset else ""))
            
            try:
                if subset:
                    dataset = load_dataset(name, subset, split=split, cache_dir=self.cache_dir)
                else:
                    dataset = load_dataset(name, split=split, cache_dir=self.cache_dir)
                
                samples_processed = 0
                for item in tqdm(dataset, desc=f"Processing {name}", unit="samples"):
                    if samples_processed >= max_samples:
                        break
                    
                    if isinstance(item, dict):
                        text = item.get(text_column, "")
                    else:
                        text = str(item)
                    
                    if text and len(text.strip()) > 10:
                        yield text.strip()
                        samples_processed += 1
                        total_samples += 1
                
                logger.info(f"Processed {samples_processed} samples from {name}")
                
            except Exception as e:
                logger.error(f"Failed to load {name}: {e}")
                continue
        
        logger.info(f"Total samples: {total_samples}")
    
    def train(self) -> None:
        """Train the BPE tokenizer."""
        logger.info("Starting BPE training...")
        
        trainer = trainers.BpeTrainer(
            vocab_size=self.vocab_size,
            min_frequency=self.min_frequency,
            special_tokens=self.special_tokens,
            show_progress=True,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet()
        )
        
        text_iterator = self.load_text_data()
        self.tokenizer.train_from_iterator(text_iterator, trainer=trainer)
        
        self.tokenizer.post_processor = processors.TemplateProcessing(
            single="[BOS] $A [EOS]",
            pair="[BOS] $A [SEP] $B [EOS]",
            special_tokens=[
                ("[BOS]", self.tokenizer.token_to_id("[BOS]")),
                ("[SEP]", self.tokenizer.token_to_id("[SEP]")),
                ("[EOS]", self.tokenizer.token_to_id("[EOS]")),
            ],
        )
        
        logger.info("BPE training completed")
    
    def save(self) -> None:
        """Save tokenizer in HuggingFace format."""
        logger.info("Saving tokenizer...")
        
        # Save tokenizer.json
        tokenizer_path = self.output_dir / "tokenizer.json"
        self.tokenizer.save(str(tokenizer_path))
        
        # Create HuggingFace tokenizer
        hf_tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=self.tokenizer,
            model_max_length=512,
            padding_side="right",
        )
        
        # Set special tokens
        hf_tokenizer.pad_token = "[PAD]"
        hf_tokenizer.unk_token = "[UNK]"
        hf_tokenizer.cls_token = "[CLS]"
        hf_tokenizer.sep_token = "[SEP]"
        hf_tokenizer.mask_token = "[MASK]"
        hf_tokenizer.bos_token = "[BOS]"
        hf_tokenizer.eos_token = "[EOS]"
        
        # Save HuggingFace format
        hf_tokenizer.save_pretrained(str(self.output_dir))
        
        # Create model card
        model_card = f"""---
license: mit
library_name: transformers
tags:
- tokenizer
- bpe
- sparsa-lm
language:
- en
---

# SPARSA-LM BPE Tokenizer

BPE tokenizer for SPARSA-LM (Sparse Attention Lumina Language Model).

## Details

- **Vocabulary Size**: {self.vocab_size:,} tokens
- **Algorithm**: Byte-Pair Encoding (BPE)
- **Special Tokens**: {len(self.special_tokens)}

## Usage

```python
from transformers import PreTrainedTokenizerFast

tokenizer = PreTrainedTokenizerFast.from_pretrained("path/to/tokenizer")

text = "Hello, how are you?"
tokens = tokenizer.encode(text)
decoded = tokenizer.decode(tokens)
```

## Special Tokens

- `[PAD]` - Padding
- `[UNK]` - Unknown
- `[CLS]` - Classification
- `[SEP]` - Separator
- `[MASK]` - Mask for MLM
- `[BOS]` - Beginning of sequence
- `[EOS]` - End of sequence
"""
        
        with open(self.output_dir / "README.md", 'w') as f:
            f.write(model_card)
        
        logger.info(f"Tokenizer saved to: {self.output_dir}")
    
    def test(self) -> None:
        """Test the tokenizer."""
        logger.info("Testing tokenizer...")
        
        test_texts = [
            "Hello, how are you?",
            "[BOS] This is a test. [EOS]",
            "The quick brown fox jumps over the lazy dog.",
            "[MASK] language modeling example."
        ]
        
        for text in test_texts:
            encoded = self.tokenizer.encode(text)
            decoded = self.tokenizer.decode(encoded.ids)
            
            print(f"Original: {text}")
            print(f"Tokens: {encoded.tokens}")
            print(f"Decoded: {decoded}")
            print("-" * 40)


def main():
    """Main training function."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config/tokenizer_config.yaml')
    parser.add_argument('--test-only', action='store_true')
    args = parser.parse_args()
    
    tokenizer_trainer = BPETokenizer(args.config)
    
    if args.test_only:
        tokenizer_path = tokenizer_trainer.output_dir / "tokenizer.json"
        if tokenizer_path.exists():
            tokenizer_trainer.tokenizer = Tokenizer.from_file(str(tokenizer_path))
            tokenizer_trainer.test()
        else:
            logger.error(f"No tokenizer found at {tokenizer_path}")
    else:
        tokenizer_trainer.train()
        tokenizer_trainer.save()
        tokenizer_trainer.test()


if __name__ == "__main__":
    main()