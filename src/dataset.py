import os
import torch
from torch.utils.data import Dataset
import yaml
from datasets import load_dataset
from tqdm import tqdm
from typing import List, Dict, Any, Optional, Union, Iterable
import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np


class DatasetProcessor:
    def __init__(self, train_config: Dict[str, Any], val_config: Dict[str, Any], preprocessing_config: Dict[str, Any]):
        """
        Initialize the dataset processor with training and validation configurations.

        Args:
            train_config: Configuration for training dataset.
            val_config: Configuration for validation dataset.
            preprocessing_config: Text preprocessing options.
        """
        self.train_config = train_config
        self.val_config = val_config
        self.preprocessing_config = preprocessing_config

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[logging.StreamHandler()],
        )

        # Load and preprocess datasets
        self.train_data = self._load_and_preprocess_dataset(self.train_config, "train")
        self.val_data = self._load_and_preprocess_dataset(self.val_config, "val")

    def _load_and_preprocess_dataset(self, config: Dict[str, Any], dataset_type: str) -> List[str]:
        """
        Load and preprocess a dataset based on its configuration.

        Args:
            config: Configuration dictionary for the dataset.
            dataset_type: Type of dataset being loaded ('train' or 'val').

        Returns:
            A list of preprocessed text samples.
        """
        data: List[str] = []
        dataset_type = dataset_type.capitalize()

        try:
            if config["type"] == "local":
                data = self._process_local_dataset(config)
            elif config["type"] == "huggingface":
                data = self._process_huggingface_dataset(config)
            else:
                raise ValueError(f"Unknown dataset type: {config['type']}")
            self.logger.info(f"{dataset_type} dataset loaded with {len(data)} samples.")
        except Exception as e:
            self.logger.error(f"Error loading {dataset_type} dataset: {e}")
            raise
        return data

    def _process_local_dataset(self, config: Dict[str, Any]) -> List[str]:
        """
        Process local dataset files.

        Args:
            config: Configuration dictionary for the local dataset.

        Returns:
            A list of preprocessed text samples.
        """
        path = Path(config["config"]["path"])
        if not path.exists():
            raise FileNotFoundError(f"Local dataset path not found: {path}")

        patterns = config["config"].get("patterns", ["*.txt"])
        processed_data: List[str] = []

        def process_file(file: Path) -> List[str]:
            try:
                if file.suffix == ".csv":
                    import pandas as pd
                    df = pd.read_csv(file)
                    return df[config["config"]["csv_text_column"]].dropna().tolist()
                else:
                    text = file.read_text(encoding="utf-8")
                    return self._preprocess_text(text.splitlines())
            except Exception as e:
                self.logger.error(f"Error processing file {file}: {e}")
                return []

        for pattern in patterns:
            files = list(path.glob(pattern))
            with ThreadPoolExecutor() as executor:
                results = executor.map(process_file, files)
                for result in results:
                    processed_data.extend(result)
        return processed_data

    def _process_huggingface_dataset(self, config: Dict[str, Any]) -> List[str]:
        """
        Process Hugging Face dataset.

        Args:
            config: Configuration dictionary for the Hugging Face dataset.

        Returns:
            A list of preprocessed text samples.
        """
        hf_data = load_dataset(
            config["name"],
            split=config.get("split", "train"),
            cache_dir=config.get("cache_dir", "./cache"),
        )

        field = config.get("field", "text")
        if field not in hf_data.features:
            raise ValueError(f"Field '{field}' not found in Hugging Face dataset.")

        return self._preprocess_text(hf_data[field])

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
        """
        Return a PyTorch Dataset for the training data.
        
        Args:
            tokenizer: Tokenizer to tokenize the text data.
            max_length: Maximum sequence length for tokenization.

        Returns:
            A PyTorch Dataset instance for training.
        """
        return TextDataset(self.train_data, tokenizer, max_length)

    def get_val_dataset(self, tokenizer: Any, max_length: int) -> Dataset:
        """
        Return a PyTorch Dataset for the validation data.
        
        Args:
            tokenizer: Tokenizer to tokenize the text data.
            max_length: Maximum sequence length for tokenization.

        Returns:
            A PyTorch Dataset instance for validation.
        """
        return TextDataset(self.val_data, tokenizer, max_length)

    def save_processed_data(self, data: List[str], output_path: Union[str, Path]) -> None:
        """
        Save preprocessed data to a file.

        Args:
            data: List of processed text samples.
            output_path: Path to save the processed data.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(data, output_path)
        size_in_mb = output_path.stat().st_size / (1024 * 1024)
        self.logger.info(f"Processed data saved at {output_path}. Size: {size_in_mb:.2f} MB, Samples: {len(data)}.")


class TextDataset(Dataset):
    def __init__(self, data: List[str], tokenizer: Any, max_length: int):
        """
        Initialize the dataset with tokenized text.

        Args:
            data: List of preprocessed text samples.
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
        Tokenize and return a single data sample.

        Args:
            idx: Index of the sample.

        Returns:
            A dictionary with tokenized input IDs, attention mask, and labels.
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
            "input_ids": encoding.input_ids.squeeze(),
            "attention_mask": encoding.attention_mask.squeeze(),
            "labels": encoding.input_ids.squeeze(),
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
