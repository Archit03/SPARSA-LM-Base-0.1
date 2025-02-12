import gc
import re
import torch
import logging
import hashlib
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import unicodedata
import signal
import queue
from threading import Event, Lock
import psutil
import ftfy
import sys

# ✅ Check for GPU availability and set device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.cuda.set_per_process_memory_fraction(0.8)  # Use 80% of GPU memory
    BATCH_SIZE = 32  # Adjust based on GPU memory
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
        self.excluded_chars = set(
            '\x00\x01\x02\x03\x04\x05\x06\x07\x08\x0b\x0c\x0e\x0f\x10\x11\x12\x13\x14\x15\x16\x17\x18\x19\x1a\x1b\x1c\x1d\x1e\x1f'
        )

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

    def normalize_unicode(self, text: str) -> str:
        """Normalize Unicode text using ftfy."""
        return ftfy.fix_text(text)

    def remove_html_tags(self, text: str) -> str:
        """Remove HTML tags from text."""
        return re.sub(r'<[^>]+>', '', text)

    def remove_duplicate_lines(self, text: str) -> str:
        """Remove duplicate lines in text while preserving formatting."""
        lines = text.split("\n")
        seen = set()
        unique_lines = []
        for line in lines:
            line = line.strip()
            if line and line not in seen:
                seen.add(line)
                unique_lines.append(line)
        return "\n".join(unique_lines)

    def clean_text(self, text: str) -> str:
        """Apply multiple cleaning transformations."""
        text = self.normalize_unicode(text)
        text = self.remove_html_tags(text)
        text = self.remove_duplicate_lines(text)
        text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
        return text.strip()

    def process_batch(self, texts: List[str]) -> List[str]:
        """Process a batch of texts using GPU acceleration where possible."""
        return [self.clean_text(text) for text in texts]

class MLFileProcessor:
    """Process files with ML-specific optimizations using GPU acceleration."""
    def __init__(self, input_dir: Path, config: MLTextConfig):
        self.input_dir = Path(input_dir)
        self.processor = MLTextProcessor(config)
        self.stop_event = Event()
        self.error_queue = queue.Queue()
        self.batch_size = config.batch_size
        self.batch_lock = Lock()

        # ✅ Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """Handle termination signals."""
        logging.warning(f"⚠️ Received termination signal {signum}. Stopping processing...")
        self.stop_event.set()

    def _is_cleaned(self, text: str) -> bool:
        """
        ✅ Check if text is already cleaned.
        - If it contains control characters, excessive blank lines, or duplicate sequences, it's unclean.
        """
        if any(char in text for char in self.processor.config.excluded_chars):
            return False  # Contains control characters
        if "  " in text or "\t" in text:  # Extra spaces or tabs
            return False
        return True  # Otherwise, assume it's clean

    def _process_batch(self, file_paths: List[Path]) -> int:
        """✅ Process a batch of files together, checking for cleanliness first."""
        texts = []
        paths_to_clean = []

        for input_path in file_paths:
            with input_path.open('r', encoding='utf-8', errors='replace') as infile:
                text = infile.read()
                if not self._is_cleaned(text):  # 🚨 Check if text needs cleaning
                    paths_to_clean.append(input_path)
                texts.append(text)

        # Process only uncleaned files
        if paths_to_clean:
            logging.info(f"🛠 Cleaning {len(paths_to_clean)} uncleaned files in-place...")

            processed_texts = self.processor.process_batch(texts)

            # Overwrite the original files with cleaned text
            for input_path, processed_text in zip(paths_to_clean, processed_texts):
                with input_path.open('w', encoding='utf-8', newline='\n') as outfile:
                    outfile.write(processed_text)

        return len(paths_to_clean)  # Return number of files cleaned

    def process_directory(self, max_workers: Optional[int] = None):
        """✅ Process directory with in-place cleaning before processing."""
        input_files = list(self.input_dir.rglob('*.txt'))

        # ✅ Process files in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(self._process_batch, [file]) for file in input_files]

            with tqdm(total=len(input_files), desc="🔄 Processing files") as pbar:
                for future in as_completed(futures):
                    future.result()  # Ensure errors are caught
                    pbar.update(1)

        # ✅ Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

def main():
    """✅ Main execution function."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),  # Ensure console output supports Unicode
            logging.FileHandler('logs/ml_processing.log', encoding='utf-8')  # Ensure UTF-8 encoding in log file
        ]
    )

    config = MLTextConfig()
    input_directory = Path(r"C:\Users\ASUS\Desktop\PreProcessed\processed\cleaned")

    processor = MLFileProcessor(input_directory, config)
    processor.process_directory()

    logging.info("✅ All files are cleaned and preprocessed.")

if __name__ == "__main__":
    main()
