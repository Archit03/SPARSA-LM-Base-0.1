import os
import re
import logging
from pathlib import Path
import torch
import aiofiles
import asyncio
import unicodedata
import time
from typing import List
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

class GPUAcademicTextCleanerOptimized:
    def __init__(self, memory_fraction: float = 0.8):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.cuda.set_per_process_memory_fraction(memory_fraction)
        self.setup_patterns()

    def setup_patterns(self):
        """Compile patterns for academic text cleaning."""
        self.patterns = {
            "urls": re.compile(r'http[s]?://\S+'),
            "emails": re.compile(r'[\w.-]+@[\w.-]+\.\w+'),
            "references": re.compile(r'\[\d+(?:,\s*\d+)*\]'),
            "multiple_spaces": re.compile(r' {2,}'),
            "multiple_newlines": re.compile(r'\n{3,}')
        }

    def clean_text(self, text: str) -> str:
        """Clean a single text string."""
        text = unicodedata.normalize('NFKC', text)
        text = self.patterns["urls"].sub('', text)
        text = self.patterns["emails"].sub('', text)
        text = self.patterns["references"].sub('', text)
        text = self.patterns["multiple_spaces"].sub(' ', text)
        text = self.patterns["multiple_newlines"].sub('\n\n', text)
        return text.strip()

class GPUAcademicTextPreprocessor:
    def __init__(self, input_dir: str, output_dir: str, batch_size: int = 50):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.batch_size = batch_size
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
                logging.FileHandler('logs/text_preprocessing.log')
            ]
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info("Logger initialized for GPUAcademicTextPreprocessor")

    async def read_file_async(self, file_path: Path) -> str:
        """Asynchronously read a .txt file."""
        async with aiofiles.open(file_path, mode='r', encoding='utf-8') as f:
            return await f.read()

    async def write_file_async(self, file_path: Path, content: str):
        """Asynchronously write content to a file."""
        async with aiofiles.open(file_path, mode='w', encoding='utf-8') as f:
            await f.write(content)

    async def process_file(self, file_path: Path):
        """Process a single .txt file with detailed logging."""
        self.logger.info(f"Starting to process file: {file_path.name}")
        try:
            text = await self.read_file_async(file_path)
            cleaned_text = self.cleaner.clean_text(text)
            output_path = self.output_dir / f"{file_path.stem}_cleaned.txt"
            await self.write_file_async(output_path, cleaned_text)
            self.logger.info(f"Successfully processed file: {file_path.name}")
        except Exception as e:
            self.logger.error(f"Error processing file {file_path.name}: {e}")

    async def process_files_in_batches(self):
        """Process .txt files in parallel batches with detailed logging."""
        files = list(self.input_dir.glob("*.txt"))
        self.logger.info(f"Found {len(files)} .txt files to process.")

        for i in range(0, len(files), self.batch_size):
            batch = files[i:i + self.batch_size]
            self.logger.info(f"Processing batch {i // self.batch_size + 1} with {len(batch)} files.")
            tasks = [self.process_file(file) for file in batch]

            start_time = time.time()
            await asyncio.gather(*tasks)
            end_time = time.time()

            self.logger.info(f"Finished processing batch {i // self.batch_size + 1} in {end_time - start_time:.2f} seconds.")

    def process_all(self):
        """Process all .txt files."""
        self.logger.info("Starting .txt file processing.")
        asyncio.run(self.process_files_in_batches())
        self.logger.info("Finished processing all .txt files.")

# Usage
if __name__ == "__main__":
    processor = GPUAcademicTextPreprocessor(
        input_dir=r"C:\Users\ASUS\Desktop\PreProcessed",
        output_dir=r"C:\Users\ASUS\Desktop\PreProcessed\cleaned",
        batch_size=50  # Adjust batch size as per GPU memory
    )
    processor.process_all()
