import os
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from typing import List, Dict, Any, Union, Iterable
import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from sklearn.model_selection import train_test_split


class DatasetProcessor:
    def __init__(self, source_dir: str, split: Dict[str, Any], preprocessing_config: Dict[str, Any]):
        """
        Initialize DatasetProcessor for loading, preprocessing, and splitting data.

        Args:
            source_dir: Directory containing source data.
            split: Dictionary with keys 'test_size' and 'random_state' for splitting.
            preprocessing_config: Configuration for preprocessing text data.
        """
        self.source_dir = source_dir
        self.test_size = split.get('test_size', 0.2)
        self.random_state = split.get('random_state', 42)
        self.preprocessing_config = preprocessing_config

        self.data = []
        self.train_data = []
        self.val_data = []

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[logging.StreamHandler()],
        )

        self._load_and_split_data()

    def _load_source_data(self) -> List[str]:
        """
        Load data from the source directory based on the defined patterns.

        Returns:
            A list of loaded text data.
        """
        import glob
        import pandas as pd

        self.logger.info(f"Loading data from {self.source_dir}...")
        loaded_data = []
        patterns = self.preprocessing_config.get('patterns', ['*.txt', '*.csv'])

        for pattern in patterns:
            files = glob.glob(os.path.join(self.source_dir, pattern))
            for file in tqdm(files, desc=f"Processing files matching {pattern}"):
                try:
                    if file.endswith('.csv'):
                        df = pd.read_csv(file)
                        column = self.preprocessing_config.get('csv_text_column', 'text')
                        loaded_data.extend(df[column].dropna().tolist())
                    else:
                        with open(file, 'r', encoding='utf-8') as f:
                            loaded_data.extend(f.readlines())
                except Exception as e:
                    self.logger.error(f"Failed to process file {file}: {e}")

        self.logger.info(f"Loaded {len(loaded_data)} text samples.")
        return [line.strip() for line in loaded_data if line.strip()]

    def _load_and_split_data(self):
        """
        Load and split data into training and validation sets.
        """
        self.logger.info("Loading and splitting data...")
        self.data = self._load_source_data()
        self.train_data, self.val_data = train_test_split(
            self.data, test_size=self.test_size, random_state=self.random_state
        )
        self.logger.info(f"Train dataset size: {len(self.train_data)}")
        self.logger.info(f"Validation dataset size: {len(self.val_data)}")

    def _preprocess_text(self, texts: Iterable[str]) -> List[str]:
        """
        Preprocess text based on the preprocessing configuration.

        Args:
            texts: Iterable of raw text samples.

        Returns:
            List of preprocessed text samples.
        """
        processed = []
        for text in tqdm(texts, desc="Preprocessing text data"):
            if isinstance(text, str):
                cleaned = text.strip()
                if self.preprocessing_config.get("lowercase", False):
                    cleaned = cleaned.lower()
                if cleaned and len(cleaned.split()) >= self.preprocessing_config.get("min_length", 1):
                    processed.append(cleaned)
        return processed

    def get_train_dataset(self, tokenizer: Any, max_length: int) -> Dataset:
        """
        Return a PyTorch Dataset for the training data.
        """
        return TextDataset(self.train_data, tokenizer, max_length)

    def get_val_dataset(self, tokenizer: Any, max_length: int) -> Dataset:
        """
        Return a PyTorch Dataset for the validation data.
        """
        return TextDataset(self.val_data, tokenizer, max_length)


class TextDataset(Dataset):
    def __init__(self, data: List[str], tokenizer: Any, max_length: int):
        """
        Initialize a PyTorch Dataset with tokenized data.

        Args:
            data: List of text samples.
            tokenizer: Tokenizer for encoding text.
            max_length: Maximum sequence length for tokenization.
        """
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Return a single tokenized sample as a dictionary.

        Args:
            idx: Index of the sample.

        Returns:
            Dictionary with tokenized data and attention mask.
        """
        text = self.data[idx]
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding.input_ids.squeeze(0),
            "attention_mask": encoding.attention_mask.squeeze(0),
            "labels": encoding.input_ids.squeeze(0),
        }

    @staticmethod
    def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        Collate function for batching samples in a DataLoader.

        Args:
            batch: List of encoded samples.

        Returns:
            A dictionary of stacked tensors for each field.
        """
        return {key: torch.stack([sample[key] for sample in batch]) for key in batch[0].keys()}
