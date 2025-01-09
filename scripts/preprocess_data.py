import os
import re
import logging
from pathlib import Path
from tqdm import tqdm
from PyPDF2 import PdfReader
from concurrent.futures import ThreadPoolExecutor, as_completed
import torch
from multiprocessing import cpu_count
from typing import Optional, Set, List, Dict
import unicodedata
import json
from collections import Counter
import numpy as np
from functools import lru_cache

class TextCleaner:
    def __init__(self, language: str = "en"):
        self.language = language
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._load_cleaning_rules()
        self.setup_patterns()
        
    def _load_cleaning_rules(self):
        self.rules = {
            "en": {
                "allowed_chars": set("abcdefghijklmnopqrstuvwxyz'-"),
                "sentence_end": set(".!?"),
                "abbreviations": {
                    "mr.", "mrs.", "ms.", "dr.", "prof.", "sr.", "jr.", "etc.",
                    "e.g.", "i.e.", "vs.", "ph.d.", "u.s.", "u.k.", "a.m.", "p.m."
                }
            },
            "hi": {
                "allowed_chars": set(chr(i) for i in range(0x0900, 0x097F)),
                "sentence_end": set("।॥.!?"),
                "abbreviations": set()
            }
        }

    def setup_patterns(self):
        self.patterns = {
            "whitespace": re.compile(r'\s+'),
            "urls": re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'),
            "emails": re.compile(r'[\w\.-]+@[\w\.-]+\.\w+'),
            "phone": re.compile(r'\+?\d{1,4}?[-.\s]?\(?\d{1,3}?\)?[-.\s]?\d{1,4}[-.\s]?\d{1,4}[-.\s]?\d{1,9}'),
            "numbers": re.compile(r'\d+'),
            "special_chars": re.compile(r'[^\w\s]'),
            "repeated_chars": re.compile(r'(.)\1{2,}'),
            "multiple_spaces": re.compile(r' {2,}'),
            "multiple_newlines": re.compile(r'\n{2,}'),
        }

    @lru_cache(maxsize=1024)
    def is_valid_char(self, char: str) -> bool:
        return char in self.rules[self.language]["allowed_chars"]

    def normalize_unicode(self, text: str) -> str:
        return unicodedata.normalize('NFKC', text)

    def remove_control_chars(self, text: str) -> str:
        return ''.join(char for char in text if unicodedata.category(char)[0] != 'C')

    def fix_sentence_boundaries(self, text: str) -> str:
        for abbrev in self.rules[self.language]["abbreviations"]:
            text = text.replace(abbrev, abbrev.replace(".", "@"))
        
        # Fix sentence endings
        text = re.sub(r'([.!?])\s*([A-Za-z])', r'\1\n\2', text)
        
        # Restore abbreviations
        text = text.replace("@", ".")
        return text

    def correct_common_errors(self, text: str) -> str:
        corrections = {
            r'\b(c|w)ud\b': lambda m: 'could' if m.group(1) == 'c' else 'would',
            r'\b(y|h)v\b': lambda m: 'have',
            r'\b(im|i m)\b': "I'm",
            r'\bdont\b': "don't",
            r'\bwont\b': "won't",
            r'\bcant\b': "can't",
            r'\bive\b': "I've",
            r'\bthats\b': "that's",
        }
        
        for pattern, replacement in corrections.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        return text

    def normalize_punctuation(self, text: str) -> str:
        # Normalize quotes
        text = re.sub(r'[''`]', "'", text)
        text = re.sub(r'["""]', '"', text)
        
        # Fix spacing around punctuation
        text = re.sub(r'\s*([,.!?;:])\s*', r'\1 ', text)
        text = re.sub(r'\s*([,.!?;:])\s*$', r'\1', text)
        
        return text.strip()

    def process_batch(self, chars: List[str]) -> str:
        char_tensor = torch.tensor([ord(c) for c in chars], dtype=torch.int32, device=self.device)
        
        # Convert to lowercase
        char_tensor = torch.where(
            (char_tensor >= 65) & (char_tensor <= 90),
            char_tensor + 32,
            char_tensor
        )
        
        return ''.join(chr(c) for c in char_tensor.tolist())

    def clean_text(self, text: str, batch_size: int = 100000) -> str:
        try:
            # Initial normalization
            text = self.normalize_unicode(text)
            text = self.remove_control_chars(text)
            
            # Remove unwanted content
            text = self.patterns["urls"].sub(' ', text)
            text = self.patterns["emails"].sub(' ', text)
            text = self.patterns["phone"].sub(' ', text)
            
            # Process text in batches
            cleaned_parts = []
            for i in range(0, len(text), batch_size):
                batch = text[i:i + batch_size]
                cleaned_parts.append(self.process_batch(batch))
            
            text = ''.join(cleaned_parts)
            
            # Apply language-specific cleaning
            text = ''.join(char for char in text if self.is_valid_char(char) or char.isspace())
            
            # Fix common issues
            text = self.correct_common_errors(text)
            text = self.fix_sentence_boundaries(text)
            text = self.normalize_punctuation(text)
            
            # Final cleanup
            text = self.patterns["multiple_spaces"].sub(' ', text)
            text = self.patterns["multiple_newlines"].sub('\n', text)
            
            return text.strip()
            
        except Exception as e:
            logging.error(f"Error in text cleaning: {e}")
            return ""

class TextPreprocessor:
    def __init__(
        self,
        download_dir: str,
        output_dir: str,
        language: str = "en",
        batch_size: int = 100000,
        supported_extensions: Set[str] = {'.pdf', '.txt'}
    ):
        self.download_dir = Path(download_dir)
        self.output_dir = Path(output_dir)
        self.batch_size = batch_size
        self.supported_extensions = supported_extensions
        self.cleaner = TextCleaner(language)
        
        self._setup_logging()
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _setup_logging(self) -> None:
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        logging.basicConfig(
            filename=log_dir / "preprocessing.log",
            filemode="a",
            level=logging.INFO,
            format="%(asctime)s - [%(levelname)s] - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )

    def extract_text(self, file_path: Path) -> Optional[str]:
        try:
            if file_path.suffix == '.pdf':
                reader = PdfReader(str(file_path))
                return "\n".join(
                    page.extract_text() for page in reader.pages 
                    if page.extract_text()
                )
            elif file_path.suffix == '.txt':
                return file_path.read_text(encoding='utf-8')
        except Exception as e:
            logging.error(f"Failed to extract text from {file_path}: {e}")
        return None

    def process_file(self, file_path: Path) -> None:
        if file_path.suffix not in self.supported_extensions:
            logging.warning(f"Skipping unsupported file: {file_path}")
            return

        try:
            output_path = self.output_dir / f"{file_path.stem}.txt"
            
            raw_text = self.extract_text(file_path)
            if not raw_text:
                return
                
            cleaned_text = self.cleaner.clean_text(raw_text)
            if cleaned_text:
                output_path.write_text(cleaned_text, encoding='utf-8')
                logging.info(f"Processed and saved: {output_path}")
                
        except Exception as e:
            logging.error(f"Error processing {file_path}: {e}")

    def process_all(self) -> None:
        files = [
            f for f in self.download_dir.iterdir()
            if f.suffix in self.supported_extensions
        ]
        
        max_workers = min(cpu_count(), len(files))
        logging.info(f"Processing {len(files)} files with {max_workers} workers")
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self.process_file, file_path): file_path 
                for file_path in files
            }
            
            for _ in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="Processing Files"
            ):
                pass

def main():
    processor = TextPreprocessor(
        download_dir=r"C:\Users\ASUS\Desktop\Data",
        output_dir=r"C:\Users\ASUS\Desktop\PreProcessed",
        language="en"
    )
    processor.process_all()

if __name__ == "__main__":
    main()