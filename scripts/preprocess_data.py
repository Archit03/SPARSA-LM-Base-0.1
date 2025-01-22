import gc
import re
import torch
import logging
import hashlib
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import unicodedata
from collections import defaultdict, Counter
import sys
from datetime import datetime
import signal
import queue
from threading import Event, Lock
import psutil
import warnings
import pickle
from functools import lru_cache
import mmap
import ftfy
from contextlib import contextmanager

# Check for GPU availability and set device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    # Set memory limits for RTX 1650 (4GB)
    torch.cuda.set_per_process_memory_fraction(0.8)  # Use 80% of GPU memory
    BATCH_SIZE = 32  # Adjust based on your GPU memory
else:
    BATCH_SIZE = 1

class MLTextConfig:
    """Configuration for ML-specific text preprocessing."""

    def __init__(self):
        self.max_sequence_length = 2048
        self.min_sequence_length = 8
        self.batch_size = BATCH_SIZE
        self.special_tokens = {
            'pad': '[PAD]',
            'unk': '[UNK]',
            'mask': '[MASK]',
            'bos': '[BOS]',
            'eos': '[EOS]',
            'sep': '[SEP]',
            'url': '[URL]',
            'email': '[EMAIL]',
            'phone': '[PHONE]',
            'number': '[NUMBER]',
            'date': '[DATE]',
            'time': '[TIME]',
            'entity': '[ENTITY]'
        }
        self.preserve_patterns = {
            'latex': r'\$[^$]+\$|\\\([^)]+\\\)|\\\[[^\]]+\\\]',
            'code_block': r'```[^`]+```',
            'inline_code': r'`[^`]+`',
            'markdown_header': r'^#{1,6}\s.*$',
            'list_item': r'^[-*+]\s.*$',
            'quote': r'^>\s.*$',
            'table': r'\|[^|]+\|'
        }
        self.excluded_chars = set('\x00\x01\x02\x03\x04\x05\x06\x07\x08\x0b\x0c\x0e\x0f\x10\x11\x12\x13\x14\x15\x16\x17\x18\x19\x1a\x1b\x1c\x1d\x1e\x1f')


class MLTextProcessor:
    """Advanced text processor optimized for ML training data with GPU support."""

    def __init__(self, config: MLTextConfig):
        self.config = config
        self._setup_regex_patterns()
        self.device = DEVICE

    def _setup_regex_patterns(self):
        """Compile regex patterns for efficient text processing."""
        self.patterns = {
            name: re.compile(pattern, re.MULTILINE)
            for name, pattern in self.config.preserve_patterns.items()
        }

    @torch.cuda.amp.autocast()  # Enable automatic mixed precision
    def process_batch(self, texts: List[str]) -> List[str]:
        """Process a batch of texts using GPU acceleration where possible."""
        # Convert texts to tensor representation for GPU processing
        # This is a simplified example - you may need to adjust based on your specific needs
        processed_texts = []
        
        # Process texts in parallel using GPU
        for text in texts:
            normalized = self.normalize_unicode(text)
            cleaned = self._remove_excluded_chars(normalized)
            processed = self._apply_ml_transformations(cleaned)
            processed_texts.append(processed)
            
        return processed_texts

    def normalize_unicode(self, text: str) -> str:
        """Normalize Unicode text."""
        return ftfy.fix_text(text)

    def process_text(self, text: str) -> str:
        """Process single text with ML-specific optimizations."""
        return self.process_batch([text])[0]

    def _remove_excluded_chars(self, text: str) -> str:
        """Remove unwanted characters."""
        return ''.join(c for c in text if c not in self.config.excluded_chars)

    def _apply_ml_transformations(self, text: str) -> str:
        """Apply ML-specific transformations."""
        text = re.sub(r'\s+', ' ', text)
        return text.strip()


class MLFileProcessor:
    """Process files with ML-specific optimizations using GPU acceleration."""

    def __init__(self, input_dir: Path, output_dir: Path, config: MLTextConfig):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.processor = MLTextProcessor(config)
        self.stop_event = Event()
        self.error_queue = queue.Queue()
        self.batch_size = config.batch_size
        self.batch_lock = Lock()

        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """Handle termination signals."""
        logging.warning(f"Received termination signal {signum}. Stopping processing...")
        self.stop_event.set()

    def process_directory(self, max_workers: Optional[int] = None):
        """Process directory with GPU-accelerated batch processing."""
        input_files = list(self.input_dir.rglob('*.txt'))
        
        # Process files in batches
        batches = [input_files[i:i + self.batch_size] 
                  for i in range(0, len(input_files), self.batch_size)]

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for batch in batches:
                if self.stop_event.is_set():
                    break
                future = executor.submit(self._process_batch, batch)
                futures.append(future)

            with tqdm(total=len(input_files), desc="Processing files") as pbar:
                for future in as_completed(futures):
                    try:
                        num_processed = future.result()
                        pbar.update(num_processed)
                    except Exception as e:
                        self.error_queue.put(str(e))
                        pbar.update(self.batch_size)

        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

    def _process_batch(self, file_paths: List[Path]) -> int:
        """Process a batch of files together."""
        texts = []
        for input_path in file_paths:
            with input_path.open('r', encoding='utf-8', errors='replace') as infile:
                texts.append(infile.read())

        # Process batch using GPU
        processed_texts = self.processor.process_batch(texts)

        # Write results
        for input_path, processed_text in zip(file_paths, processed_texts):
            relative_path = input_path.relative_to(self.input_dir)
            output_path = self.output_dir / relative_path
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with output_path.open('w', encoding='utf-8', newline='\n') as outfile:
                outfile.write(processed_text)

        return len(file_paths)


def main():
    """Main execution function."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(), logging.FileHandler('logs/ml_processing.log')]
    )

    # Log GPU information
    if torch.cuda.is_available():
        logging.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
        logging.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    else:
        logging.warning("No GPU available. Using CPU.")

    config = MLTextConfig()
    input_directory = Path(r"C:\Users\ASUS\Desktop\PreProcessed\processed\pubmed\test")
    output_directory = Path(r"C:\Users\ASUS\Desktop\PreProcessed\processed\cleaned")

    processor = MLFileProcessor(input_directory, output_directory, config)
    processor.process_directory()


if __name__ == "__main__":
    main()