import os
import re
import logging
from pathlib import Path
from tqdm import tqdm
import torch
from concurrent.futures import ThreadPoolExecutor
import aiofiles
import asyncio
import json
import pandas as pd
import unicodedata
from typing import Dict, Set, Optional, List
from dataclasses import dataclass
import warnings
import pdfplumber
import time

# Suppress PyPDF2 warnings
warnings.filterwarnings("ignore", category=UserWarning)

@dataclass
class CleaningRules:
    """Data class to store text cleaning rules"""
    preserve_terms: Set[str]
    sentence_end: Set[str]
    preserve_characters: Set[str]
    abbreviations: Set[str]

class GPUAcademicTextCleaner:
    def __init__(self, batch_size: int = 1024 * 1024, memory_fraction: float = 0.2):
        self.batch_size = batch_size
        self.memory_fraction = memory_fraction
        self.setup_logging()
        self.setup_gpu()
        self._load_cleaning_rules()
        self.setup_patterns()
        self.setup_gpu_tensors()

    def setup_logging(self):
        """Configure logging with proper formatting"""
        logging.basicConfig(
            level=logging.DEBUG,  # Set to DEBUG for detailed logs
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler('logs/text_cleaning_detailed.log')  # Detailed log file
            ]
        )
        self.logger = logging.getLogger(__name__)

    def setup_gpu(self):
        """Initialize GPU settings with error handling and memory management"""
        try:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if self.device.type == "cuda":
                self.max_memory = torch.cuda.get_device_properties(0).total_memory
                torch.cuda.set_per_process_memory_fraction(self.memory_fraction)
                self.logger.info(f"Using GPU with {self.max_memory / 1024**3:.2f} GB memory")
            else:
                self.logger.warning("GPU not available. Using CPU instead.")
        except Exception as e:
            self.logger.error(f"Error setting up GPU: {e}")
            self.device = torch.device("cpu")

    def _load_cleaning_rules(self):
        """Load rules from a configuration file or use defaults"""
        try:
            with open('cleaning_rules.json', 'r') as f:
                rules_data = json.load(f)
        except FileNotFoundError:
            rules_data = {
                "preserve_terms": ["p", "sd", "ci", "n", "r", "t", "mg", "g", "ml", "l", "cm", "mm", "h", "min", "sec"],
                "sentence_end": [".", "!", "?"],
                "preserve_characters": list(".-_()[]{}%"),
                "abbreviations": ["e.g.", "i.e.", "et al.", "vs.", "etc.", "fig.", "tab."]
            }

        self.rules = CleaningRules(
            preserve_terms=set(rules_data["preserve_terms"]),
            sentence_end=set(rules_data["sentence_end"]),
            preserve_characters=set(rules_data["preserve_characters"]),
            abbreviations=set(rules_data["abbreviations"])
        )

    def setup_patterns(self):
        """Compile regular expressions for academic text cleaning."""
        self.patterns = {
            "whitespace": re.compile(r'\s+'),
            "urls": re.compile(r'http[s]?://\S+'),
            "emails": re.compile(r'[\w.-]+@[\w.-]+\.\w+'),
            "references": re.compile(r'\[\d+(?:,\s*\d+)*\]'),
            "figure_refs": re.compile(r'fig\.\s*\d+', re.IGNORECASE),
            "table_refs": re.compile(r'table\s*\d+', re.IGNORECASE),
            "page_numbers": re.compile(r'\b\d+\s*$'),
            "citations": re.compile(r'\(\w+\s+et\s+al\.,\s*\d{4}\)'),
            "measurements": re.compile(r'\d+(?:\.\d+)?\s*(?:mg|g|ml|L|cm|mm|h|min|sec)\b'),
            "statistics": re.compile(r'(?:p|t|r|n)\s*(?:<|>|=|≤|≥)\s*\d+(?:\.\d+)?'),
            "multiple_spaces": re.compile(r' {2,}'),
            "multiple_newlines": re.compile(r'\n{3,}'),
            "space_before_punct": re.compile(r'\s+([.,;:)])')
        }

    def setup_gpu_tensors(self):
        """Prepare tensors for GPU operations."""
        self.char_to_idx = {chr(i): i for i in range(128)}
        self.idx_to_char = {i: chr(i) for i in range(128)}
        self.char_map = torch.tensor([self.char_to_idx.get(c, 0) for c in self.char_to_idx.keys()],
                                     device=self.device)
        self.preserve_chars_tensor = torch.tensor(
            [ord(c) for c in ''.join(self.rules.preserve_characters)], device=self.device
        )

    @torch.no_grad()
    def process_batch_gpu(self, text_tensor: torch.Tensor) -> torch.Tensor:
        """Process a batch of text on GPU."""
        try:
            preserve_mask = torch.isin(text_tensor, self.preserve_chars_tensor)
            processed = text_tensor.clone()
            mask = ~preserve_mask
            upper_mask = (processed >= 65) & (processed <= 90) & mask
            processed[upper_mask] += 32
            return processed
        except Exception as e:
            self.logger.error(f"Error in GPU batch processing: {e}")
            return text_tensor

    def process_text_gpu(self, text: str) -> str:
        """Convert text to tensor, process on GPU, and convert back."""
        text_tensor = torch.tensor([ord(c) for c in text], device=self.device)
        processed_tensor = self.process_batch_gpu(text_tensor)
        return ''.join([chr(c) for c in processed_tensor.cpu().numpy() if c != 0])

    def clean_text(self, text: str) -> str:
        """Main cleaning function for academic text using GPU acceleration."""
        if not text:
            return ""

        try:
            text = unicodedata.normalize('NFKC', text)
            text = self.preserve_important_elements(text)
            text = self.patterns["urls"].sub('', text)
            text = self.patterns["emails"].sub('', text)
            text = self.process_text_gpu(text)
            text = self.patterns["multiple_spaces"].sub(' ', text)
            text = self.patterns["multiple_newlines"].sub('\n\n', text)
            text = self.patterns["space_before_punct"].sub(r'\1', text)
            return text.strip()
        except Exception as e:
            self.logger.error(f"Error in text cleaning: {e}")
            return text

    def preserve_important_elements(self, text: str) -> str:
        """Preserve important academic elements while cleaning."""
        for pattern_name in ["measurements", "statistics", "references", "figure_refs", "table_refs", "citations"]:
            pattern = self.patterns[pattern_name]
            text = pattern.sub(lambda m: f"PRESERVE{m.group(0)}PRESERVE", text)
        return text

class GPUAcademicTextPreprocessorOptimized:
    def __init__(self, input_dir: str, output_dir: str, max_workers: int = 8):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.cleaner = GPUAcademicTextCleaner()
        self.max_workers = max_workers
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.setup_logging()  # Add logger setup

    def setup_logging(self):
        """Configure logging for the preprocessor."""
        self.logger = logging.getLogger(f"{__name__}.Preprocessor")
        if not self.logger.hasHandlers():
            handler = logging.FileHandler('preprocessor_detailed.log')
            handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.DEBUG)
        self.logger.info("Logger initialized for GPUAcademicTextPreprocessorOptimized")

    async def read_file_async(self, file_path: Path) -> str:
        """Asynchronously read a file's content."""
        try:
            if file_path.suffix.lower() == '.pdf':
                return self.read_pdf(file_path)
            async with aiofiles.open(file_path, mode='r', encoding='utf-8') as f:
                return await f.read()
        except UnicodeDecodeError:
            with open(file_path, mode='r', encoding='latin1') as f:
                return f.read()

    def read_pdf(self, file_path: Path) -> str:
        """Extract text from a PDF file using pdfplumber."""
        try:
            with pdfplumber.open(file_path) as pdf:
                return '\n'.join(page.extract_text() for page in pdf.pages if page.extract_text())
        except Exception as e:
            self.logger.error(f"Error reading PDF file {file_path}: {e}")
            return ""

    async def process_and_write_file(self, file_path: Path):
        """Asynchronously read, process, and write cleaned content with detailed logs."""
        try:
            self.logger.info(f"Starting to process file: {file_path}")

            # Read the file
            start_read = time.time()
            text = await self.read_file_async(file_path)
            end_read = time.time()
            self.logger.info(f"Finished reading {file_path.name} in {end_read - start_read:.2f} seconds.")

            # Clean the text
            start_clean = time.time()
            cleaned_text = self.cleaner.clean_text(text)
            end_clean = time.time()
            self.logger.info(f"Finished cleaning {file_path.name} in {end_clean - start_clean:.2f} seconds.")

            # Write the output
            output_path = self.output_dir / f"{file_path.stem}_cleaned.txt"
            start_write = time.time()
            async with aiofiles.open(output_path, mode='w', encoding='utf-8') as f:
                await f.write(cleaned_text)
            end_write = time.time()

            self.logger.info(f"Finished writing {file_path.name} to {output_path} in {end_write - start_write:.2f} seconds.")
            self.logger.info(f"Completed processing file: {file_path}")
        except Exception as e:
            self.logger.error(f"Error processing {file_path}: {e}")

    async def process_all_async(self):
        """Process files asynchronously in batches with detailed logging."""
        files = list(self.input_dir.glob("*"))
        batch_size = 100  # Number of files per batch
        self.logger.info(f"Starting processing of {len(files)} files in batches of {batch_size}.")

        for i in range(0, len(files), batch_size):
            batch_files = files[i:i + batch_size]
            self.logger.info(f"Processing batch {i // batch_size + 1} with {len(batch_files)} files.")

            tasks = [self.process_and_write_file(file_path) for file_path in batch_files]
            start_time = time.time()
            await asyncio.gather(*tasks)
            end_time = time.time()

            self.logger.info(f"Finished batch {i // batch_size + 1} in {end_time - start_time:.2f} seconds.")

        self.logger.info("Finished processing all files.")

    def process_all(self):
        """Wrapper to run asynchronous file processing synchronously."""
        asyncio.run(self.process_all_async())

# Usage
if __name__ == "__main__":
    processor = GPUAcademicTextPreprocessorOptimized(
        input_dir=r"C:\\Users\\ASUS\\Desktop\\Data",
        output_dir=r"C:\\Users\\ASUS\\Desktop\\PreProcessed"
    )
    processor.process_all()
