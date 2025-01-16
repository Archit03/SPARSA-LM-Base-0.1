import os
import re
import logging
from pathlib import Path
from pqdm.processes import pqdm
import torch
import unicodedata
import time
from typing import List

class GPUAcademicTextCleanerOptimized:
    def __init__(self, memory_fraction: float = 0.8):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.cuda.set_per_process_memory_fraction(memory_fraction)
        self.setup_patterns()

    def setup_patterns(self):
        """Compile patterns for text cleaning."""
        self.patterns = {
            "urls": re.compile(r'http[s]?://\S+'),
            "emails": re.compile(r'[\w.-]+@[\w.-]+\.\w+'),
            "html_tags": re.compile(r'<.*?>'),
            "non_ascii": re.compile(r'[\x80-\xFF]+'),
            "references": re.compile(r'\[\d+(?:,\s*\d+)*\]'),
            "multiple_spaces": re.compile(r' {2,}'),
            "multiple_newlines": re.compile(r'\n{3,}')
        }

    def clean_text(self, text: str) -> str:
        """Clean a single text string."""
        text = unicodedata.normalize('NFKC', text)
        text = self.patterns["urls"].sub('', text)
        text = self.patterns["emails"].sub('', text)
        text = self.patterns["html_tags"].sub('', text)
        text = self.patterns["non_ascii"].sub('', text)
        text = self.patterns["references"].sub('', text)
        text = self.patterns["multiple_spaces"].sub(' ', text)
        text = self.patterns["multiple_newlines"].sub('\n\n', text)
        return text.strip()

class GPUAcademicTextPreprocessor:
    def __init__(self, input_dir: str, output_dir: str, max_workers: int = 4):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.max_workers = max_workers
        self.cleaner = GPUAcademicTextCleanerOptimized()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.setup_logging()

    def setup_logging(self):
        """Set up detailed logging."""
        logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler('logs/c4_text_preprocessing.log')
            ]
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info("Logger initialized for GPUAcademicTextPreprocessor")

    def read_file(self, file_path: Path) -> str:
        """Read a .txt file."""
        try:
            with open(file_path, mode='r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            self.logger.error(f"Error reading file {file_path}: {e}")
            return ""

    def write_file(self, file_path: Path, content: str):
        """Write content to a file."""
        try:
            with open(file_path, mode='w', encoding='utf-8') as f:
                f.write(content)
        except Exception as e:
            self.logger.error(f"Error writing file {file_path}: {e}")

    def process_file(self, file_path: Path):
        """Process a single .txt file with detailed logging."""
        self.logger.info(f"Starting to process file: {file_path.name}")
        try:
            text = self.read_file(file_path)
            cleaned_text = self.cleaner.clean_text(text)
            output_path = self.output_dir / f"{file_path.stem}_cleaned.txt"
            self.write_file(output_path, cleaned_text)
            self.logger.info(f"Successfully processed file: {file_path.name}")
        except Exception as e:
            self.logger.error(f"Error processing file {file_path.name}: {e}")

    def process_all_files(self):
        """Process all .txt files using pqdm for parallel processing."""
        files = list(self.input_dir.glob("*.txt"))
        self.logger.info(f"Found {len(files)} .txt files to process.")

        def process_wrapper(file_path):
            self.process_file(file_path)

        pqdm(files, process_wrapper, n_jobs=self.max_workers, desc="Processing files")

# Usage
if __name__ == "__main__":
    processor = GPUAcademicTextPreprocessor(
        input_dir=r"C:\\Users\\ASUS\\Desktop\\C4Dataset",  # Directory containing the downloaded C4 .txt files
        output_dir=r"C:\\Users\\ASUS\\Desktop\\C4Dataset\\cleaned",  # Directory for cleaned output
        max_workers=8  # Adjust number of workers as per system capability
    )
    processor.process_all_files()
