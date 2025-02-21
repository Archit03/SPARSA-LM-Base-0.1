#!/usr/bin/env python
import gc
import re
import sys
import ftfy
import queue
import signal
import psutil
import logging
import torch
import html
import unicodedata

from bs4 import BeautifulSoup, FeatureNotFound
from pathlib import Path
from tqdm import tqdm
from threading import Event, Lock
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, List

# GPU Setup
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.cuda.set_per_process_memory_fraction(0.8)  # Use 80% of GPU memory
    BATCH_SIZE = 32  # Adjust based on available VRAM
else:
    BATCH_SIZE = 1  # CPU mode

class MLTextConfig:
    """
    Configuration for text preprocessing.
    
    Options include:
      - Removing HTML markup, URLs, emails, and code snippets.
      - Converting text to lowercase if desired.
      - Removing all non-English characters (i.e. non-ASCII).
      - Normalizing text using Unicode NFKC.
    """
    def __init__(self) -> None:
        self.batch_size: int = BATCH_SIZE
        self.do_lowercase: bool = False          # Set to True to force lowercase conversion
        self.min_text_length: int = 5             # Minimum allowed text length
        self.remove_html: bool = True             # Remove HTML markup
        self.remove_urls: bool = True             # Remove URLs
        self.remove_emails: bool = True           # Remove email addresses
        self.remove_code_snippets: bool = True    # Remove inline and block code
        self.remove_non_english: bool = True      # Remove all non-English characters (non-ASCII)
        self.normalize_unicode: bool = True       # Normalize Unicode using NFKC

class MLTextProcessor:
    """
    Processes and cleans text for training.
    
    The cleaning pipeline includes:
      1. Fixing broken Unicode with ftfy.
      2. Removing HTML markup.
      3. Replacing problematic Unicode characters.
      4. Removing URLs and email addresses.
      5. Removing code snippets (both block and inline).
      6. Removing all non-English characters (keeping only ASCII).
      7. Normalizing text with NFKC.
    """
    def __init__(self, config: MLTextConfig) -> None:
        self.config = config

    def remove_html_markup(self, text: str) -> str:
        """Removes HTML, CSS, and JavaScript markup from text."""
        try:
            soup = BeautifulSoup(text, "lxml")
            return soup.get_text(separator=" ")
        except FeatureNotFound:
            logging.warning("lxml not installed. Falling back to html.parser.")
            try:
                soup = BeautifulSoup(text, "html.parser")
                return soup.get_text(separator=" ")
            except Exception as e:
                logging.error(f"Failed to parse HTML: {e}. Using regex fallback.")
                return re.sub(r'<[^>]+>', '', text)

    def remove_problematic_unicode(self, text: str) -> str:
        """
        Replaces problematic Unicode characters with ASCII equivalents.
        Replacements include common quotation marks and dashes.
        """
        unicode_replacements = {
            "\u2013": "-",  # EN DASH
            "\u2014": "-",  # EM DASH
            "\u2018": "'",  # LEFT SINGLE QUOTATION MARK
            "\u2019": "'",  # RIGHT SINGLE QUOTATION MARK
            "\u201C": '"',  # LEFT DOUBLE QUOTATION MARK
            "\u201D": '"',  # RIGHT DOUBLE QUOTATION MARK
        }
        for char, replacement in unicode_replacements.items():
            text = text.replace(char, replacement)
        return text

    def remove_urls_and_emails(self, text: str) -> str:
        """
        Removes URLs, email addresses, and bare domain names.
        """
        text = re.sub(r'https?://\S+|www\.\S+', ' ', text)
        text = re.sub(r'\b[\w\.-]+@[\w\.-]+\.\w+\b', ' ', text)
        text = re.sub(r'\b(?:[a-z0-9-]+\.)+(?:com|org|net|edu|gov|mil|biz|info|mobi|name|aero|asia|jobs|museum)\b',
                      ' ', text, flags=re.IGNORECASE)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def remove_code_snippets(self, text: str) -> str:
        """
        Removes code snippets:
         - Multi-line code blocks enclosed in triple backticks.
         - Inline code enclosed in single backticks.
         - Lines starting with a comment symbol.
        """
        text = re.sub(r"```.*?```", "", text, flags=re.DOTALL)
        text = re.sub(r"`[^`]+`", "", text)
        text = re.sub(r"#.*", "", text)
        return text

    def remove_non_english_text(self, text: str) -> str:
        """
        Removes all non-English characters by retaining only ASCII characters.
        This effectively removes characters outside the U+0000 to U+007F range.
        """
        return re.sub(r'[^\x00-\x7F]+', ' ', text).strip()

    def normalize_text(self, text: str) -> str:
        """
        Applies the full cleaning pipeline:
          1. Fix broken Unicode.
          2. Remove HTML markup.
          3. Replace problematic Unicode.
          4. Remove URLs and email addresses.
          5. Remove code snippets.
          6. Convert to lowercase (if enabled).
          7. Remove all non-English characters (if enabled).
          8. Normalize Unicode with NFKC.
        """
        text = ftfy.fix_text(text)
        if self.config.remove_html:
            text = self.remove_html_markup(text)
        text = self.remove_problematic_unicode(text)
        if self.config.remove_urls or self.config.remove_emails:
            text = self.remove_urls_and_emails(text)
        if self.config.remove_code_snippets:
            text = self.remove_code_snippets(text)
        if self.config.do_lowercase:
            text = text.lower()
        if self.config.remove_non_english:
            text = self.remove_non_english_text(text)
        if self.config.normalize_unicode:
            text = unicodedata.normalize('NFKC', text)
        return text.strip()

    def process_batch(self, texts: List[str]) -> List[str]:
        """Processes a list of text strings."""
        return [self.normalize_text(t) for t in texts]

# -------------------- File Processor --------------------
class MLFileProcessor:
    """
    Processes text files in a directory using MLTextProcessor.
    Recursively finds all .txt files, cleans them, and writes the cleaned text back.
    """
    def __init__(self, input_dir: Path, config: MLTextConfig) -> None:
        self.input_dir = input_dir
        self.processor = MLTextProcessor(config)
        self.stop_event = Event()
        self.error_queue: "queue.Queue[Exception]" = queue.Queue()
        self.batch_size: int = config.batch_size
        self.batch_lock = Lock()
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame) -> None:
        logging.warning(f"Received termination signal {signum}. Stopping processing...")
        self.stop_event.set()

    def _process_file(self, input_path: Path) -> None:
        if self.stop_event.is_set():
            return
        try:
            logging.info(f"Processing file: {input_path}")
            with input_path.open('r', encoding='utf-8', errors='replace') as infile:
                text = infile.read()
            cleaned_text = self.processor.normalize_text(text)
            with input_path.open('w', encoding='utf-8', newline='\n', errors="replace") as outfile:
                outfile.write(cleaned_text + "\n")
        except Exception as e:
            self.error_queue.put(e)
            logging.error(f"Error processing file {input_path}: {e}")

    def process_directory(self, max_workers: Optional[int] = None) -> None:
        input_files = list(self.input_dir.rglob('*.txt'))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(self._process_file, f) for f in input_files]
            for future in tqdm(as_completed(futures), total=len(input_files), desc="Processing"):
                try:
                    future.result()
                except Exception as e:
                    logging.error(f"Error in future processing: {e}")

# -------------------- Main Test --------------------
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # --- Test cleaning on a sample text ---
    test_text = """
    Hello, how are you? Visit our website at https://example.com and contact us at test@example.com.
    Here is some code: ```def hello(): print("Hello World!")```
    And some HTML: <p>This is a paragraph.</p>
    Non-English: こんにちは、世界！ Привет, мир!
    """
    
    config = MLTextConfig()
    # For perfect cleaning: remove HTML, URLs, emails, code, and remove all non-English characters.
    config.do_lowercase = True
    config.remove_html = True
    config.remove_urls = True
    config.remove_emails = True
    config.remove_code_snippets = True
    config.remove_non_english = True
    config.normalize_unicode = True

    processor = MLTextProcessor(config)
    
    cleaned = processor.normalize_text(test_text)
    logging.info("=== Cleaning Test ===")
    logging.info(f"Original Text:\n{test_text}")
    logging.info(f"Cleaned Text:\n{cleaned}")
    
    # --- Process a directory if provided ---
    if len(sys.argv) > 1:
        input_directory = Path(sys.argv[1])
    else:
        # Default directory (change as needed)
        input_directory = Path(r"C:\Users\ASUS\Desktop\PreProcessed\processed\cleaned")
    
    file_processor = MLFileProcessor(input_directory, config)
    file_processor.process_directory(max_workers=4)
    
    # Force garbage collection after processing
    gc.collect()
