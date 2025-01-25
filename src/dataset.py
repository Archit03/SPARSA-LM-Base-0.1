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
        if not os.path.exists(source_dir):
            raise ValueError(f"Source directory does not exist: {source_dir}")
        self.source_dir = source_dir
        self.split = split
        self.preprocessing_config = preprocessing_config
        self.train_data = []
        self.val_data = []
        self._load_and_split_data()

    def _load_and_split_data(self):
    # Load and preprocess files from source_dir
        source_data = self._load_and_preprocess_source()
        test_size = self.split.get('test_size', 0.2)
        random_state = self.split.get('random_state', 42)
    
        # Perform train-test split
        self.train_data, self.val_data = train_test_split(
        source_data, test_size=test_size, random_state=random_state
    )

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[logging.StreamHandler()],
        )

        # Load and preprocess data
        self.data = self._load_and_preprocess_source()
        self.train_data, self.val_data = train_test_split(
            self.data, test_size=self.test_size, random_state=self.random_state
        )
        self.logger.info(f"Train dataset size: {len(self.train_data)}")
        self.logger.info(f"Validation dataset size: {len(self.val_data)}")

    def _load_and_preprocess_source(self):
        data = []
        patterns = self.preprocessing_config.get("patterns", ["*.txt", "*.csv"])
        for pattern in patterns:
            for file_path in Path(self.source_dir).glob(pattern):
                if file_path.suffix == ".csv":
                  import pandas as pd
                  df = pd.read_csv(file_path)
                  data.extend(df[self.preprocessing_config["csv_text_column"]].dropna().tolist())
                else:
                  with open(file_path, "r", encoding="utf-8") as f:
                      data.extend(f.readlines())
        return self._preprocess_text(data)

    def _preprocess_text(self, texts: Iterable[str]) -> List[str]:
        """
        Preprocess text based on the preprocessing configuration.

        Args:
            texts: Iterable of raw text samples.

        Returns:
            List of preprocessed text samples.
        """
        processed = []
        for text in texts:
            if isinstance(text, str):
                cleaned = text.strip()
                if self.preprocessing_config.get("lowercase", False):
                    cleaned = cleaned.lower()
                if cleaned and len(cleaned.split()) >= self.preprocessing_config.get("min_length", 1):
                    processed.append(cleaned)
        return processed

    def get_train_dataset(self, tokenizer: Any, max_length: int) -> Dataset:
        """Return a PyTorch Dataset for the training data."""
        return TextDataset(self.train_data, tokenizer, max_length)

    def get_val_dataset(self, tokenizer: Any, max_length: int) -> Dataset:
        """Return a PyTorch Dataset for the validation data."""
        return TextDataset(self.val_data, tokenizer, max_length)


class TextDataset(Dataset):
    def __init__(self, data: List[str], tokenizer: Any, max_length: int):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = self.data[idx]
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding.input_ids.squeeze(0),  # Ensure tensors are not batched
            "attention_mask": encoding.attention_mask.squeeze(0),
            "labels": encoding.input_ids.squeeze(0),  # For autoregressive tasks
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
