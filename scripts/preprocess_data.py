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

    This configuration class holds various options for processing text,
    including removal flags for HTML, ads, metadata, URLs, emails, and code snippets.

    **Attributes:**
    
    - ``batch_size (int)``: Batch size for processing text.
    - ``do_lowercase (bool)``: Whether to convert text to lowercase.
    - ``min_text_length (int)``: Minimum allowed text length.
    - ``remove_non_english (bool)``: Remove non-English text.
    - ``remove_japanese (bool)``: Remove Japanese text.
    - ``remove_duplicates (bool)``: Whether to remove duplicate content.
    - ``remove_html (bool)``: Remove HTML markup.
    - ``remove_ads (bool)``: Remove advertisements.
    - ``remove_boilerplate (bool)``: Remove boilerplate text.
    - ``remove_metadata (bool)``: Remove metadata.
    - ``remove_code_snippets (bool)``: Remove code snippets.
    - ``remove_urls (bool)``: Remove URLs.
    - ``remove_emails (bool)``: Remove email addresses.
    - ``keep_lists_and_bullets (bool)``: Preserve lists and bullet formatting.
    - ``remove_special_chars (bool)``: Remove special characters.
    """
    def __init__(self) -> None:
        self.batch_size: int = BATCH_SIZE
        self.do_lowercase: bool = False  # Preserve case for now
        self.min_text_length: int = 5  # Keep shorter texts if needed
        self.remove_non_english: bool = True
        self.remove_japanese: bool = True
        self.remove_duplicates: bool = False  # Retain meaningful repetitions
        self.remove_html: bool = True
        self.remove_ads: bool = True
        self.remove_boilerplate: bool = True
        self.remove_metadata: bool = True
        self.remove_code_snippets: bool = True
        self.remove_urls: bool = True  # Remove URLs
        self.remove_emails: bool = True  # Remove Emails
        self.keep_lists_and_bullets: bool = True  # Preserve structured text
        self.remove_special_chars: bool = False  # Retain special formatting


class MLTextProcessor:
    """
    Processes and cleans text for training a transformer language model.

    **Methods:**

    - ``remove_html_markup(text)``: Strips HTML, CSS, and JavaScript.
    - ``remove_problematic_unicode(text)``: Replaces problematic Unicode characters.
    - ``remove_urls_and_emails(text)``: Removes URLs, email addresses, and domains.
    - ``remove_code_snippets(text)``: Removes inline and block code snippets.
    - ``normalize_text(text)``: Runs all processing steps on the input text.
    - ``process_batch(texts)``: Processes a list of text strings.
    """
    def __init__(self, config: MLTextConfig) -> None:
        self.config = config
        self.device = DEVICE

    def remove_html_markup(self, text: str) -> str:
        """
        Removes HTML, CSS, and JavaScript markup from text.

        Attempts to use ``lxml`` via BeautifulSoup; falls back to ``html.parser``
        or regex if necessary.

        :param text: Input text containing HTML markup.
        :return: Text with HTML markup removed.
        """
        try:
            soup = BeautifulSoup(text, "lxml")  # Use 'lxml' for better handling
            return soup.get_text(separator=" ")
        except FeatureNotFound:
            logging.warning("⚠️ lxml not installed. Falling back to html.parser.")
            try:
                soup = BeautifulSoup(text, "html.parser")
                return soup.get_text(separator=" ")
            except Exception as e:
                logging.error(f"⚠️ Failed to parse HTML using BeautifulSoup: {e}. Using regex fallback.")
                return re.sub(r'<[^>]+>', '', text)

    def remove_problematic_unicode(self, text: str) -> str:
        """
        Replaces problematic Unicode characters with ASCII equivalents.

        **Replacements include:**

        - EN DASH (``\u2013``) to ``-``
        - EM DASH (``\u2014``) to ``-``
        - LEFT SINGLE QUOTATION MARK (``\u2018``) to ``'``
        - RIGHT SINGLE QUOTATION MARK (``\u2019``) to ``'``
        - LEFT DOUBLE QUOTATION MARK (``\u201C``) to ``"``
        - RIGHT DOUBLE QUOTATION MARK (``\u201D``) to ``"``

        :param text: Input text.
        :return: Text with problematic Unicode characters replaced.
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
        Removes URLs, email addresses, and domain names from the text.

        **Removal steps:**
        
        - Remove URLs beginning with ``http://``, ``https://``, or ``www.``
        - Remove email addresses using a simple regex.
        - Remove bare domain names (e.g., ``example.com``) with common TLDs.

        :param text: Input text.
        :return: Text with URLs, emails, and domains removed.
        """
        # Remove URLs (http://, https://, or www.)
        text = re.sub(r'https?://\S+|www\.\S+', ' ', text)
        # Remove email addresses (a simple but effective pattern)
        text = re.sub(r'\b[\w\.-]+@[\w\.-]+\.\w+\b', ' ', text)
        # Remove domain names that are not part of an email or URL (e.g., example.com)
        text = re.sub(r'\b(?:[a-z0-9-]+\.)+(?:com|org|net|edu|gov|mil|biz|info|mobi|name|aero|asia|jobs|museum)\b',
                      ' ', text, flags=re.IGNORECASE)
        # Clean up any extra whitespace created during removal
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def remove_code_snippets(self, text: str) -> str:
        """
        Removes inline and block code snippets while keeping normal text.

        **Removals include:**

        - Multi-line code blocks enclosed in triple backticks.
        - Inline code enclosed in single backticks.
        - Lines starting with a comment symbol (``#``).

        :param text: Input text containing code.
        :return: Text with code snippets removed.
        """
        text = re.sub(r"```.*?```", "", text, flags=re.DOTALL)  # Remove multi-line code blocks
        text = re.sub(r"`[^`]+`", "", text)  # Remove inline code
        text = re.sub(r"#.*", "", text)  # Remove commented-out lines
        return text

    def normalize_text(self, text: str) -> str:
        """
        Applies all preprocessing steps to the text.

        **Steps include:**

        - Fixing text with ``ftfy``.
        - Removing HTML markup.
        - Replacing problematic Unicode characters.
        - Removing URLs and email addresses.
        - Removing code snippets.

        :param text: Input text.
        :return: Fully normalized and cleaned text.
        """
        text = ftfy.fix_text(text)
        if self.config.remove_html:
            text = self.remove_html_markup(text)
        text = self.remove_problematic_unicode(text)
        if self.config.remove_urls or self.config.remove_emails:
            text = self.remove_urls_and_emails(text)  # Fully remove URLs & emails
        if self.config.remove_code_snippets:
            text = self.remove_code_snippets(text)  # Remove code snippets
        return text.strip()

    def process_batch(self, texts: List[str]) -> List[str]:
        """
        Processes a batch of texts.

        :param texts: List of text strings to process.
        :return: List of normalized and cleaned text strings.
        """
        return [self.normalize_text(t) for t in texts]


class MLFileProcessor:
    """
    Processes text files using the MLTextProcessor.

    This class recursively searches for all ``*.txt`` files in the given
    directory, processes each file, and writes the cleaned text back to the file.

    **Methods:**

    - ``_process_file(input_path)``: Processes an individual file.
    - ``process_directory(max_workers)``: Processes all text files in a directory.
    """
    def __init__(self, input_dir: Path, config: MLTextConfig) -> None:
        """
        Initializes the file processor.

        :param input_dir: Path to the directory containing text files.
        :param config: An instance of MLTextConfig with processing settings.
        """
        self.input_dir = input_dir
        self.processor = MLTextProcessor(config)
        self.stop_event = Event()
        self.error_queue: "queue.Queue[Exception]" = queue.Queue()
        self.batch_size: int = config.batch_size
        self.batch_lock = Lock()
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame) -> None:
        """
        Handles termination signals gracefully.

        :param signum: The signal number.
        :param frame: The current stack frame.
        """
        logging.warning(f"⚠️ Received termination signal {signum}. Stopping processing...")
        self.stop_event.set()

    def _process_file(self, input_path: Path) -> None:
        """
        Processes a single file.

        Reads the file, normalizes its text, and writes the cleaned text back.

        :param input_path: Path to the text file to be processed.
        """
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
        """
        Processes all text files in the directory.

        Uses a ThreadPoolExecutor to process files in parallel.

        :param max_workers: Maximum number of threads to use.
        """
        input_files = list(self.input_dir.rglob('*.txt'))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(self._process_file, f) for f in input_files]
            for future in tqdm(as_completed(futures), total=len(input_files), desc="Processing"):
                try:
                    future.result()
                except Exception as e:
                    logging.error(f"⚠️ Error in future processing: {e}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # Test the URL and email removal
    test_text = """
    Check out our website at https://example.com or www.test.com!
    Contact us at test@example.com or complex.email+label@sub.domain.com
    Here's a tricky URL: http://test.com/path?param=value#fragment
    And a hidden email: user.name123@subdomain.company.co.uk
    Also check example.com and test.org for more info.
    """
    
    config = MLTextConfig()
    processor = MLTextProcessor(config)
    
    print("Testing URL and email removal:")
    print("\nOriginal text:")
    print(test_text)
    print("\nCleaned text:")
    print(processor.remove_urls_and_emails(test_text))
    
    # Process directory if path is provided via command line, else use default
    if len(sys.argv) > 1:
        input_directory = Path(sys.argv[1])
    else:
        input_directory = Path(r"C:\Users\ASUS\Desktop\PreProcessed\processed\cleaned\test")
    
    file_processor = MLFileProcessor(input_directory, config)
    file_processor.process_directory(max_workers=4)
