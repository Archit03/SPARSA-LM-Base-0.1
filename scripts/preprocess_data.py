#!/usr/bin/env python
import gc
import re
import sys
import ftfy
import queue
import signal
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
      - Removing HTML tags and decoding HTML entities.
      - Normalizing Unicode (NFKC) and removing unwanted control characters (preserving newlines).
      - Normalizing whitespace (collapsing spaces, controlling consecutive newlines, stripping line ends,
        and filtering out lines shorter than a minimum length).
      - Removing URLs, email addresses, and bare domain names.
      - Optionally converting text to lowercase.
      - Optionally removing non-English characters (i.e., non-ASCII).
    """
    def __init__(self) -> None:
        self.batch_size: int = BATCH_SIZE
        self.do_lowercase: bool = False           # Convert text to lowercase if True.
        self.min_line_length: int = 5              # Drop lines shorter than this length.
        self.remove_html: bool = True              # Remove HTML markup.
        self.remove_urls: bool = True              # Remove URLs.
        self.remove_emails: bool = True            # Remove email addresses.
        self.remove_code_snippets: bool = True     # Remove code snippets.
        self.remove_non_english: bool = True       # Remove non-English (non-ASCII) characters.
        self.normalize_unicode: bool = True        # Normalize Unicode using NFKC.

class MLTextProcessor:
    """
    Processes and cleans text using a series of normalization steps:
    
      1. Fix broken Unicode using ftfy.
      2. Decode HTML entities.
      3. Remove HTML markup.
      4. Remove control characters (except newlines).
      5. Replace problematic Unicode characters (e.g., smart quotes, dashes).
      6. Remove URLs and email addresses.
      7. Optionally remove code snippets.
      8. Normalize whitespace (collapse spaces, trim lines, control newlines, filter short lines).
      9. Optionally convert to lowercase.
      10. Optionally remove non-English characters.
      11. Normalize Unicode using NFKC.
    """
    def __init__(self, config: MLTextConfig) -> None:
        self.config = config

    def decode_html_entities(self, text: str) -> str:
        """Decode HTML entities (e.g., &amp; to &, &lt; to <)."""
        return html.unescape(text)

    def remove_html_markup(self, text: str) -> str:
        """Strip HTML, CSS, and JavaScript markup from text."""
        try:
            soup = BeautifulSoup(text, "lxml")
            return soup.get_text(separator=" ")
        except FeatureNotFound:
            logging.warning("lxml not installed. Falling back to html.parser.")
            try:
                soup = BeautifulSoup(text, "html.parser")
                return soup.get_text(separator=" ")
            except Exception as e:
                logging.error(f"HTML parsing failed: {e}. Using regex fallback.")
                return re.sub(r'<[^>]+>', '', text)

    def remove_control_characters(self, text: str) -> str:
        """
        Remove control characters (non-printable) except newline characters.
        Removes characters in ranges: U+0000-U+0008, U+000B-U+001F, and U+007F.
        """
        return re.sub(r'[\x00-\x08\x0B-\x1F\x7F]', '', text)

    def remove_problematic_unicode(self, text: str) -> str:
        """
        Replace problematic Unicode characters with ASCII equivalents.
        Examples: smart quotes and dashes.
        """
        replacements = {
            "\u2013": "-",  # EN DASH
            "\u2014": "-",  # EM DASH
            "\u2018": "'",  # LEFT SINGLE QUOTATION MARK
            "\u2019": "'",  # RIGHT SINGLE QUOTATION MARK
            "\u201C": '"',  # LEFT DOUBLE QUOTATION MARK
            "\u201D": '"',  # RIGHT DOUBLE QUOTATION MARK
        }
        for char, repl in replacements.items():
            text = text.replace(char, repl)
        return text

    def remove_urls_and_emails(self, text: str) -> str:
        """
        Remove URLs, email addresses, and bare domain names.
        """
        text = re.sub(r'https?://\S+|www\.\S+', ' ', text)
        text = re.sub(r'\b[\w\.-]+@[\w\.-]+\.\w+\b', ' ', text)
        text = re.sub(
            r'\b(?:[a-z0-9-]+\.)+(?:com|org|net|edu|gov|mil|biz|info|mobi|name|aero|asia|jobs|museum)\b',
            ' ', text, flags=re.IGNORECASE
        )
        return re.sub(r'\s+', ' ', text).strip()

    def remove_code_snippets(self, text: str) -> str:
        """
        Remove code snippets:
          - Multi-line blocks enclosed in triple backticks.
          - Inline code enclosed in single backticks.
          - Lines starting with a comment symbol.
        """
        text = re.sub(r"```.*?```", "", text, flags=re.DOTALL)
        text = re.sub(r"`[^`]+`", "", text)
        text = re.sub(r"#.*", "", text)
        return text

    def remove_non_english_text(self, text: str) -> str:
        """
        Remove all non-English characters by retaining only ASCII characters.
        """
        return re.sub(r'[^\x00-\x7F]+', ' ', text).strip()

    def normalize_whitespace(self, text: str) -> str:
        """
        Normalize whitespace:
          - Strip spaces from beginning and end of each line.
          - Collapse multiple spaces to one.
          - Collapse multiple newlines to a single newline.
          - Optionally drop lines shorter than a minimum length.
        """
        lines = text.splitlines()
        normalized = []
        for line in lines:
            line_clean = re.sub(r'\s+', ' ', line).strip()
            if len(line_clean) >= self.config.min_line_length:
                normalized.append(line_clean)
        return "\n".join(normalized)

    def normalize_text(self, text: str) -> str:
        """
        Execute the complete cleaning pipeline.
        """
        # 1. Fix broken Unicode
        text = ftfy.fix_text(text)
        # 2. Decode HTML entities
        text = self.decode_html_entities(text)
        # 3. Remove HTML markup (if enabled)
        if self.config.remove_html:
            text = self.remove_html_markup(text)
        # 4. Remove control characters (except newlines)
        text = self.remove_control_characters(text)
        # 5. Replace problematic Unicode characters
        text = self.remove_problematic_unicode(text)
        # 6. Remove URLs and email addresses (if enabled)
        if self.config.remove_urls or self.config.remove_emails:
            text = self.remove_urls_and_emails(text)
        # 7. Optionally remove code snippets
        if self.config.remove_code_snippets:
            text = self.remove_code_snippets(text)
        # 8. Normalize whitespace (collapse spaces/newlines and trim lines)
        text = self.normalize_whitespace(text)
        # 9. Convert to lowercase if enabled
        if self.config.do_lowercase:
            text = text.lower()
        # 10. Optionally remove non-English characters
        if self.config.remove_non_english:
            text = self.remove_non_english_text(text)
        # 11. Normalize Unicode using NFKC if enabled
        if self.config.normalize_unicode:
            text = unicodedata.normalize('NFKC', text)
        return text.strip()

    def process_batch(self, texts: List[str]) -> List[str]:
        """Process a list of text strings."""
        return [self.normalize_text(t) for t in texts]

# -------------------- File Processor --------------------
class MLFileProcessor:
    """
    Processes text files in a directory:
      - Recursively finds all .txt files.
      - Cleans them using MLTextProcessor.
      - Writes the cleaned text back to the same file.
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
    
    # Configure cleaning settings.
    config = MLTextConfig()
    config.do_lowercase = True
    config.remove_html = True
    config.remove_urls = True
    config.remove_emails = True
    config.remove_code_snippets = True
    config.remove_non_english = True
    config.normalize_unicode = True
    config.min_line_length = 5  # Lines shorter than 5 characters will be dropped

    processor = MLTextProcessor(config)
    cleaned = processor.normalize_text(test_text)
    
    logging.info("=== Cleaning Test ===")
    logging.info(f"Original Text:\n{test_text}")
    logging.info(f"Cleaned Text:\n{cleaned}")
    
    # --- Process a directory if provided via command line ---
    if len(sys.argv) > 1:
        input_directory = Path(sys.argv[1])
    else:
        # Default directory (change as needed)
        input_directory = Path(r"C:\Users\ASUS\Desktop\PreProcessed\processed\cleaned\txt")
    
    file_processor = MLFileProcessor(input_directory, config)
    file_processor.process_directory(max_workers=4)
    
    # Force garbage collection after processing
    gc.collect()
