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

from pathlib import Path
from tqdm import tqdm
from threading import Event, Lock
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, List

# ✅ GPU Setup
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.cuda.set_per_process_memory_fraction(0.8)  # Use 80% of GPU memory
    BATCH_SIZE = 32  # Adjust based on available VRAM
else:
    BATCH_SIZE = 1  # CPU mode


class MLTextConfig:
    """
    Configuration for text preprocessing for training a transformer language model.
    """
    def __init__(self) -> None:
        self.max_sequence_length: int = 2048
        self.min_sequence_length: int = 8
        self.batch_size: int = BATCH_SIZE
        self.do_lowercase: bool = True  # Whether to lowercase the text.
        self.special_tokens: dict = {
            'pad': '[PAD]',
            'unk': '[UNK]',
            'mask': '[MASK]',
            'bos': '[BOS]',
            'eos': '[EOS]',
            'sep': '[SEP]',
            'url': '[URL]',
            'email': '[EMAIL]',
            'phone': '[PHONE]',
            'date': '[DATE]',
            'time': '[TIME]',
            'entity': '[ENTITY]'
        }
        self.preserve_patterns: dict = {
            'latex': r'\$[^$]+\$|\\\([^\)]+\\\)|\\\[[^\]]+\\\]',
            'code_block': r'```[^`]+```',
            'inline_code': r'`[^`]+`',
            'markdown_header': r'^#{1,6}\s.*$',
            'list_item': r'^[-*+]\s.*$',
            'quote': r'^>\s.*$',
            'table': r'\|[^|]+\|'
        }
        # Excluded control characters (do not remove newlines)
        self.excluded_chars: set = set(
            '\x00\x01\x02\x03\x04\x05\x06\x07\x08\x0b\x0c\x0e\x0f'
            '\x10\x11\x12\x13\x14\x15\x16\x17\x18\x19\x1a\x1b\x1c'
            '\x1d\x1e\x1f'
        )


class MLTextProcessor:
    """
    Processes and cleans text for training a transformer LM.
    
    In addition to the core cleaning pipeline used in clean_text(),
    this class now includes 50 distinct cleaning functions (methods)
    that perform various cleaning operations.
    """
    def __init__(self, config: MLTextConfig) -> None:
        self.config = config
        self._setup_regex_patterns()
        self.device = DEVICE

    def _setup_regex_patterns(self) -> None:
        """Compile regex patterns for efficient text processing."""
        # Patterns to preserve (e.g., LaTeX, code blocks, etc.)
        self.patterns = {name: re.compile(pattern, re.MULTILINE)
                         for name, pattern in self.config.preserve_patterns.items()}
        self.url_pattern = re.compile(r'https?://\S+|www\.\S+', re.IGNORECASE)
        self.email_pattern = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
        self.special_chars_pattern = re.compile(r"[^a-zA-Z0-9\s.,!?;:\-\'\"()]")
        self.repeated_punct_pattern = re.compile(r'([.,!?;:\-])\1+')
        self.repeated_token_pattern = re.compile(r'(\[\w+\])(\1)+')
        self.number_pattern = re.compile(r'\[NUMBER\]', re.IGNORECASE)
        self.literal_number_pattern = re.compile(r'\bnumber\b', re.IGNORECASE)
        self.html_pattern = re.compile(r'<[^>]+>')
        self.repeated_words_pattern = re.compile(r'\b(\S+)(\s+\1)+\b', re.IGNORECASE)

    def normalize_unicode(self, text: str) -> str:
        """Normalize Unicode text using ftfy."""
        return ftfy.fix_text(text)

    def remove_excluded_chars(self, text: str) -> str:
        """Remove any characters defined in the excluded_chars set."""
        return ''.join(ch for ch in text if ch not in self.config.excluded_chars)

    def remove_html_tags(self, text: str) -> str:
        """Remove HTML tags from the text."""
        return self.html_pattern.sub('', text)

    def normalize_whitespace(self, text: str) -> str:
        """
        Normalize whitespace on a per-line basis.
        Collapses multiple spaces/tabs while preserving line breaks.
        """
        lines = text.splitlines()
        normalized_lines = [re.sub(r'[ \t]+', ' ', line).strip() for line in lines if line.strip() != ""]
        return "\n".join(normalized_lines)

    def remove_special_chars(self, text: str) -> str:
        """
        Remove unwanted symbols by only allowing letters, digits, whitespace,
        and the specified punctuation.
        """
        return self.special_chars_pattern.sub('', text)

    def remove_urls_and_emails(self, text: str) -> str:
        """Replace URLs and emails with placeholders."""
        text = self.url_pattern.sub('[URL]', text)
        text = self.email_pattern.sub('[EMAIL]', text)
        return text

    def remove_duplicate_lines(self, text: str) -> str:
        """Remove duplicate lines while preserving formatting."""
        lines = text.splitlines()
        seen = set()
        unique_lines = []
        for line in lines:
            if line not in seen:
                seen.add(line)
                unique_lines.append(line)
        return "\n".join(unique_lines)

    def remove_repeated_punct(self, text: str) -> str:
        """Collapse repeated punctuation (e.g., '!!!' becomes '!')."""
        return self.repeated_punct_pattern.sub(lambda m: m.group(1), text)

    def remove_repeated_tokens(self, text: str) -> str:
        """Collapse repeated tokens (e.g., '[MASK][MASK]' becomes '[MASK]')."""
        return self.repeated_token_pattern.sub(lambda m: m.group(1), text)

    def remove_number_tokens(self, text: str) -> str:
        """Remove all occurrences of [NUMBER] placeholders."""
        return self.number_pattern.sub('', text)

    def remove_literal_number(self, text: str) -> str:
        """Remove any standalone occurrences of the word 'number'."""
        return self.literal_number_pattern.sub('', text)

    def remove_consecutive_repeated_words(self, text: str) -> str:
        """
        Remove any consecutive repeated words using regex.
        For example, "this is is a test test" becomes "this a".
        """
        prev_text = None
        while prev_text != text:
            prev_text = text
            text = self.repeated_words_pattern.sub('', text)
        return text

    def remove_dash_dot_dash(self, text: str) -> str:
        """Remove all occurrences of the .-. pattern."""
        return text.replace('.-.', '')

    def remove_extra_newlines(self, text: str) -> str:
        """Collapse multiple newlines into a single newline."""
        return re.sub(r'\n+', '\n', text)

    def remove_non_ascii(self, text: str) -> str:
        """Remove non-ASCII characters from the text."""
        return ''.join(ch for ch in text if ord(ch) < 128)

    def remove_diacritics(self, text: str) -> str:
        """Remove diacritical marks from letters."""
        normalized = unicodedata.normalize('NFD', text)
        return ''.join(ch for ch in normalized if unicodedata.category(ch) != 'Mn')

    def expand_contractions(self, text: str) -> str:
        """Expand common contractions (e.g., can't -> cannot)."""
        contractions = {
            "can't": "cannot",
            "won't": "will not",
            "n't": " not",
            "'re": " are",
            "'s": " is",
            "'d": " would",
            "'ll": " will",
            "'ve": " have",
            "'m": " am"
        }
        for contraction, expansion in contractions.items():
            text = text.replace(contraction, expansion)
        return text

    def remove_stopwords(self, text: str) -> str:
        """Remove common stopwords from the text."""
        stopwords = {"the", "and", "is", "in", "at", "of", "a", "an"}
        return " ".join(word for word in text.split() if word not in stopwords)

    def remove_emojis(self, text: str) -> str:
        """Remove emoji characters from the text."""
        emoji_pattern = re.compile("["
            u"\U0001F600-\U0001F64F"  # emoticons
            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
            u"\U0001F680-\U0001F6FF"  # transport & map symbols
            u"\U0001F1E0-\U0001F1FF"  # flags
            "]+", flags=re.UNICODE)
        return emoji_pattern.sub(r'', text)

    def remove_mentions(self, text: str) -> str:
        """Remove @username mentions."""
        return re.sub(r'@\w+', '', text)

    def remove_hashtags(self, text: str) -> str:
        """Remove hashtags (e.g., #hashtag)."""
        return re.sub(r'#\w+', '', text)

    def remove_numbers(self, text: str) -> str:
        """Remove all digit sequences from the text."""
        return re.sub(r'\d+', '', text)

    def remove_brackets(self, text: str) -> str:
        """Remove text within square, round, or curly brackets."""
        return re.sub(r'\[.*?\]|\(.*?\)|\{.*?\}', '', text)

    def remove_underscores(self, text: str) -> str:
        """Remove underscore characters."""
        return text.replace('_', '')

    def collapse_commas(self, text: str) -> str:
        """Collapse multiple commas into a single comma."""
        return re.sub(r',,+', ',', text)

    def collapse_periods(self, text: str) -> str:
        """Collapse multiple periods into a single period."""
        return re.sub(r'\.\.+', '.', text)

    def collapse_question_marks(self, text: str) -> str:
        """Collapse multiple question marks into one."""
        return re.sub(r'\?+', '?', text)

    def collapse_exclamation_marks(self, text: str) -> str:
        """Collapse multiple exclamation marks into one."""
        return re.sub(r'\!+', '!', text)

    def trim_whitespace_around_punctuation(self, text: str) -> str:
        """Remove extra spaces around punctuation."""
        return re.sub(r'\s*([.,!?;:\-])\s*', r'\1 ', text).strip()

    def fix_spacing_after_punctuation(self, text: str) -> str:
        """Ensure a single space follows punctuation marks."""
        return re.sub(r'([.,!?;:\-])([^\s])', r'\1 \2', text)

    def remove_control_characters(self, text: str) -> str:
        """Remove control characters (ASCII 0-31) from the text."""
        return re.sub(r'[\x00-\x1F]+', ' ', text)

    def remove_html_entities(self, text: str) -> str:
        """Convert HTML entities to their corresponding characters."""
        return html.unescape(text)

    def remove_non_printable(self, text: str) -> str:
        """Remove non-printable characters."""
        return ''.join(ch for ch in text if ch.isprintable())

    def remove_latex_commands(self, text: str) -> str:
        """Remove LaTeX commands (e.g., \command)."""
        return re.sub(r'\\[a-zA-Z]+', '', text)

    def remove_markdown_syntax(self, text: str) -> str:
        """Remove common markdown syntax characters (e.g., *, _, `)."""
        return re.sub(r'[*_`]', '', text)

    def remove_code_blocks(self, text: str) -> str:
        """Remove multiline code blocks (delimited by ```)."""
        return re.sub(r'```.*?```', '', text, flags=re.DOTALL)

    def remove_inline_code(self, text: str) -> str:
        """Remove inline code enclosed in backticks."""
        return re.sub(r'`[^`]+`', '', text)

    def remove_emoji_codes(self, text: str) -> str:
        """Remove emoji codes (alias for remove_emojis)."""
        return self.remove_emojis(text)

    def remove_spam_symbols(self, text: str) -> str:
        """Remove a set of spammy symbols (e.g., #, $, %, etc.)."""
        return re.sub(r'[#$%^&*+=<>~]', '', text)

    def collapse_line_breaks(self, text: str) -> str:
        """Collapse multiple consecutive line breaks into one."""
        return re.sub(r'\n+', '\n', text)

    def remove_tabs(self, text: str) -> str:
        """Replace tab characters with a single space."""
        return text.replace('\t', ' ')

    def remove_duplicate_words(self, text: str) -> str:
        """Remove duplicate words (not only consecutive)."""
        words = text.split()
        seen = set()
        result = []
        for word in words:
            if word not in seen:
                seen.add(word)
                result.append(word)
        return " ".join(result)

    def remove_punctuation_duplicates(self, text: str) -> str:
        """Collapse duplicated punctuation using a generic pattern."""
        return re.sub(r'([^\w\s])\1+', r'\1', text)

    def remove_multiple_whitespaces(self, text: str) -> str:
        """Replace multiple whitespace characters with a single space."""
        return re.sub(r'\s+', ' ', text)

    def convert_to_ascii(self, text: str) -> str:
        """Convert text to ASCII, ignoring non-ASCII characters."""
        return text.encode('ascii', errors='ignore').decode('ascii')

    def strip_leading_trailing_whitespace(self, text: str) -> str:
        """Strip leading and trailing whitespace from the text."""
        return text.strip()

    def lowercase_text(self, text: str) -> str:
        """Convert the text to lowercase."""
        return text.lower()

    def custom_cleaning_function(self, text: str, pattern: str, replacement: str) -> str:
        """
        Apply a custom regex cleaning function.
        Replace all occurrences of 'pattern' with 'replacement'.
        """
        return re.sub(pattern, replacement, text)

    def remove_urls_only(self, text: str) -> str:
        """Remove URLs from the text (without replacing with a placeholder)."""
        return self.url_pattern.sub('', text)

    def remove_emails_only(self, text: str) -> str:
        """Remove email addresses from the text (without replacing with a placeholder)."""
        return self.email_pattern.sub('', text)

    def clean_text(self, text: str) -> str:
        """
        Clean and normalize text through a series of transformation steps.
        This pipeline uses a subset of the cleaning functions defined above.
        """
        text = self.normalize_unicode(text)
        if self.config.do_lowercase:
            text = self.lowercase_text(text)
        text = self.remove_excluded_chars(text)
        text = self.remove_html_tags(text)
        text = self.remove_urls_and_emails(text)
        text = self.remove_special_chars(text)
        text = self.remove_repeated_punct(text)
        text = self.remove_dash_dot_dash(text)
        text = self.normalize_whitespace(text)
        text = self.remove_duplicate_lines(text)
        text = self.remove_repeated_tokens(text)
        text = self.remove_number_tokens(text)
        text = self.remove_literal_number(text)
        text = self.remove_consecutive_repeated_words(text)
        text = self.remove_extra_newlines(text)
        text = self.remove_non_ascii(text)
        text = self.remove_diacritics(text)
        text = self.expand_contractions(text)
        text = self.remove_stopwords(text)
        text = self.remove_emojis(text)
        text = self.remove_mentions(text)
        text = self.remove_hashtags(text)
        text = self.remove_numbers(text)
        text = self.remove_brackets(text)
        text = self.remove_underscores(text)
        text = self.collapse_commas(text)
        text = self.collapse_periods(text)
        text = self.collapse_question_marks(text)
        text = self.collapse_exclamation_marks(text)
        text = self.trim_whitespace_around_punctuation(text)
        text = self.fix_spacing_after_punctuation(text)
        text = self.remove_control_characters(text)
        text = self.remove_html_entities(text)
        text = self.remove_non_printable(text)
        text = self.remove_latex_commands(text)
        text = self.remove_markdown_syntax(text)
        text = self.remove_code_blocks(text)
        text = self.remove_inline_code(text)
        text = self.remove_emoji_codes(text)
        text = self.remove_spam_symbols(text)
        text = self.collapse_line_breaks(text)
        text = self.remove_tabs(text)
        text = self.remove_duplicate_words(text)
        text = self.remove_punctuation_duplicates(text)
        text = self.remove_multiple_whitespaces(text)
        text = self.convert_to_ascii(text)
        text = self.strip_leading_trailing_whitespace(text)
        text = self.normalize_whitespace(text)
        return text.strip()

    def process_batch(self, texts: List[str]) -> List[str]:
        """Process a batch of texts efficiently."""
        return [self.clean_text(t) for t in texts]


class MLFileProcessor:
    """
    Processes text files by cleaning their contents using MLTextProcessor.
    Files are processed in parallel using threads.
    """
    def __init__(self, input_dir: Path, config: MLTextConfig) -> None:
        self.input_dir = input_dir
        self.processor = MLTextProcessor(config)
        self.stop_event = Event()
        self.error_queue: "queue.Queue[Exception]" = queue.Queue()
        self.batch_size: int = config.batch_size
        self.batch_lock = Lock()
        # Setup signal handlers for graceful shutdown.
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame) -> None:
        """Handle termination signals gracefully."""
        logging.warning(f"⚠️ Received termination signal {signum}. Stopping processing...")
        self.stop_event.set()

    def _log_memory_usage(self, prefix: str = "Memory usage") -> None:
        """Log current memory usage (optional debugging)."""
        process = psutil.Process()
        mem_info = process.memory_info().rss / (1024 * 1024)
        logging.info(f"{prefix}: {mem_info:.2f} MB")

    def _process_file(self, input_path: Path) -> None:
        """Clean a single file."""
        if self.stop_event.is_set():
            return
        try:
            logging.info(f"Processing file: {input_path}")
            with input_path.open('r', encoding='utf-8', errors='replace') as infile:
                text = infile.read()
            cleaned_text = self.processor.clean_text(text)
            with input_path.open('w', encoding='utf-8', newline='\n') as outfile:
                outfile.write(cleaned_text + "\n")
        except Exception as e:
            self.error_queue.put(e)
            logging.error(f"Error processing file {input_path}: {e}")

    def process_directory(self, max_workers: Optional[int] = None) -> None:
        """
        Process all .txt files in the specified directory using parallel threads.
        """
        input_files = list(self.input_dir.rglob('*.txt'))
        if not input_files:
            logging.warning("No .txt files found in the directory. Aborting.")
            return

        self._log_memory_usage("Before processing")
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(self._process_file, f) for f in input_files]
            with tqdm(total=len(input_files), desc="🔄 Cleaning files") as pbar:
                for future in as_completed(futures):
                    if self.stop_event.is_set():
                        for f in futures:
                            f.cancel()
                        break
                    try:
                        future.result()
                    except Exception as e:
                        logging.error(f"Error in future: {e}")
                    pbar.update(1)
        self._log_memory_usage("After processing")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()


def main() -> None:
    """Main execution function."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('logs/ml_processing.log', encoding='utf-8')
        ]
    )
    config = MLTextConfig()
    
    # Use the original directory.
    input_directory = Path(r"C:\Users\ASUS\Desktop\PreProcessed\processed\cleaned")
    file_processor = MLFileProcessor(input_directory, config)
    
    # Process files in the directory with 4 workers.
    file_processor.process_directory(max_workers=4)
    
    logging.info("✅ All files are cleaned and preprocessed.")

if __name__ == "__main__":
    main()
