import gc
import re
import torch
import logging
import hashlib
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union
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


class MLTextConfig:
    """Configuration for ML-specific text preprocessing."""

    def __init__(self):
        self.max_sequence_length = 2048
        self.min_sequence_length = 8
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
    """Advanced text processor optimized for ML training data."""

    def __init__(self, config: MLTextConfig):
        self.config = config
        self._setup_regex_patterns()

    def _setup_regex_patterns(self):
        """Compile regex patterns for efficient text processing."""
        self.patterns = {
            name: re.compile(pattern, re.MULTILINE)
            for name, pattern in self.config.preserve_patterns.items()
        }

    def normalize_unicode(self, text: str) -> str:
        """Normalize Unicode text."""
        return ftfy.fix_text(text)

    def process_text(self, text: str) -> str:
        """Process text with ML-specific optimizations."""
        text = self.normalize_unicode(text)
        text = self._remove_excluded_chars(text)
        text = self._apply_ml_transformations(text)
        return text

    def _remove_excluded_chars(self, text: str) -> str:
        """Remove unwanted characters."""
        return ''.join(c for c in text if c not in self.config.excluded_chars)

    def _apply_ml_transformations(self, text: str) -> str:
        """Apply ML-specific transformations."""
        text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
        return text.strip()


class MLFileProcessor:
    """Process files with ML-specific optimizations."""

    def __init__(self, input_dir: Path, output_dir: Path, config: MLTextConfig):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.processor = MLTextProcessor(config)
        self.stop_event = Event()
        self.error_queue = queue.Queue()

        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """Handle termination signals."""
        logging.warning(f"Received termination signal {signum}. Stopping processing...")
        self.stop_event.set()

    def process_directory(self, max_workers: Optional[int] = None):
        """Process directory with ML optimizations."""
        input_files = list(self.input_dir.rglob('*.txt'))

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for file in input_files:
                if self.stop_event.is_set():
                    break
                future = executor.submit(self._process_file, file)
                futures.append(future)

            with tqdm(total=len(futures), desc="Processing files") as pbar:
                for future in as_completed(futures):
                    try:
                        future.result()
                        pbar.update(1)
                    except Exception as e:
                        self.error_queue.put(str(e))
                        pbar.update(1)

    def _process_file(self, input_path: Path) -> None:
        """Process individual file."""
        relative_path = input_path.relative_to(self.input_dir)
        output_path = self.output_dir / relative_path

        with input_path.open('r', encoding='utf-8', errors='replace') as infile:
            text = infile.read()

        processed_text = self.processor.process_text(text)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open('w', encoding='utf-8', newline='\n') as outfile:
            outfile.write(processed_text)


def main():
    """Main execution function."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(), logging.FileHandler('ml_processing.log')]
    )

    config = MLTextConfig()
    input_directory = Path(r"C:\Users\ASUS\Desktop\PreProcessed\processed\pubmed")
    output_directory = Path(r"C:\Users\ASUS\Desktop\PreProcessed\processed\cleaned")

    processor = MLFileProcessor(input_directory, output_directory, config)
    processor.process_directory()


if __name__ == "__main__":
    main()
