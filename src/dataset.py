import os
import glob
import logging
import numpy as np
import pandas as pd
import torch

from tqdm import tqdm
from typing import List, Dict, Any, Union, Iterable
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split


class DatasetProcessor:
    def __init__(
        self,
        source_dir: str,
        split: Dict[str, Any],
        preprocessing_config: Dict[str, Any]
    ):
        """
        Initialize DatasetProcessor for loading, preprocessing, and splitting data.

        Args:
            source_dir (str): Directory containing source data.
            split (Dict[str, Any]): 
                Keys 'test_size' (float) and 'random_state' (int) for splitting.
            preprocessing_config (Dict[str, Any]): 
                Configuration dict for how we preprocess text data.
        """
        self.source_dir = source_dir
        self.test_size = split.get("test_size", 0.2)
        self.random_state = split.get("random_state", 42)
        self.preprocessing_config = preprocessing_config

        self.data = []
        self.train_data = []
        self.val_data = []

        # Basic logger setup
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[logging.StreamHandler()],
        )

        # Load data from files and split into train/val
        self._load_and_split_data()

    def _load_source_data(self) -> List[str]:
        """
        Load data from the source directory based on file patterns.

        Returns:
            List[str]: A list of loaded text lines.
        """
        self.logger.info(f"Loading data from {self.source_dir}...")
        loaded_data = []

        # e.g. patterns could be ["*.txt", "*.csv"] from config
        patterns = self.preprocessing_config.get("patterns", ["*.txt", "*.csv"])

        for pattern in patterns:
            files = glob.glob(os.path.join(self.source_dir, pattern))
            for file in tqdm(files, desc=f"Processing files matching {pattern}"):
                try:
                    if file.endswith(".csv"):
                        df = pd.read_csv(file)
                        column = self.preprocessing_config.get("csv_text_column", "text")
                        # Extend with all non-null values in the specified column
                        loaded_data.extend(df[column].dropna().tolist())
                    else:
                        with open(file, "r", encoding="utf-8") as f:
                            loaded_data.extend(f.readlines())
                except Exception as e:
                    self.logger.error(f"Failed to process file {file}: {e}")

        self.logger.info(f"Loaded {len(loaded_data)} text samples.")

        # Strip whitespace, remove empty lines
        return [line.strip() for line in loaded_data if line.strip()]

    def _load_and_split_data(self):
        """
        Load all text data, then split into train/validation sets.
        Attempt a stratified split based on sentence length distribution.
        """
        self.logger.info("Loading and splitting data...")
        self.data = self._load_source_data()

        # Create labels for stratification based on sentence length
        if len(self.data) > 0:
            sentence_lengths = [len(txt.split()) for txt in self.data]
            min_len, max_len = min(sentence_lengths), max(sentence_lengths)
            if min_len == max_len:
                # All lines have the same length -> single bin
                labels = [0] * len(sentence_lengths)
            else:
                # 10 bins between min and max length
                bins = np.linspace(min_len, max_len, 10)
                labels = np.digitize(sentence_lengths, bins)
        else:
            labels = []

        try:
            self.train_data, self.val_data = train_test_split(
                self.data,
                test_size=self.test_size,
                random_state=self.random_state,
                stratify=labels if labels else None,
            )
        except ValueError as e:
            self.logger.warning(f"Stratified split failed: {e}. Falling back to random split.")
            self.train_data, self.val_data = train_test_split(
                self.data,
                test_size=self.test_size,
                random_state=self.random_state,
            )

        self.logger.info(f"Train dataset size: {len(self.train_data)}")
        self.logger.info(f"Validation dataset size: {len(self.val_data)}")

        # Preprocess text (e.g. lowercasing, min_length checks)
        self.train_data = self._preprocess_text(self.train_data)
        self.val_data = self._preprocess_text(self.val_data)

        self.logger.info(f"Train dataset size after preprocessing: {len(self.train_data)}")
        self.logger.info(f"Validation dataset size after preprocessing: {len(self.val_data)}")

    def _preprocess_text(self, texts: Iterable[str]) -> List[str]:
        """
        Preprocess text according to config (lowercasing, min_length, etc.).

        Args:
            texts (Iterable[str]): Raw text samples.

        Returns:
            List[str]: Cleaned/preprocessed text samples.
        """
        processed = []
        for text in tqdm(texts, desc="Preprocessing text data"):
            if isinstance(text, str):
                cleaned = text.strip()
                # Apply lowercasing if specified
                if self.preprocessing_config.get("lowercase", False):
                    cleaned = cleaned.lower()

                # Filter out lines shorter than min_length tokens
                min_length = self.preprocessing_config.get("min_length", 1)
                if cleaned and len(cleaned.split()) >= min_length:
                    processed.append(cleaned)
        return processed

    def get_train_dataset(self, tokenizer: Any, max_length: int) -> Dataset:
        """
        Create a TextDataset for training split.

        Args:
            tokenizer (Any): A Hugging Face-compatible tokenizer.
            max_length (int): Maximum sequence length for tokenization.

        Returns:
            TextDataset: A PyTorch Dataset object for training data.
        """
        return TextDataset(self.train_data, tokenizer, max_length)

    def get_val_dataset(self, tokenizer: Any, max_length: int) -> Dataset:
        """
        Create a TextDataset for validation split.

        Args:
            tokenizer (Any): A Hugging Face-compatible tokenizer.
            max_length (int): Maximum sequence length for tokenization.

        Returns:
            TextDataset: A PyTorch Dataset object for validation data.
        """
        return TextDataset(self.val_data, tokenizer, max_length)

class TextDataset(Dataset):
    def __init__(self, data: List[str], tokenizer: Any, max_length: int):
        """
        A PyTorch Dataset for tokenized text data.
        
        Args:
            data (List[str]): List of text samples.
            tokenizer (Any): Hugging Face or custom tokenizer.
            max_length (int): Maximum sequence length.
        """
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

        self.vocab_size = tokenizer.vocab_size

        self.pad_id = tokenizer.convert_tokens_to_ids("[PAD]")
        self.unk_id = tokenizer.convert_tokens_to_ids("[UNK]")
        self.bos_id = tokenizer.convert_tokens_to_ids("[BOS]")
        self.eos_id = tokenizer.convert_tokens_to_ids("[EOS]")

        print(f"🔹 Special Tokens: PAD={self.pad_id}, UNK={self.unk_id}, BOS={self.bos_id}, EOS={self.eos_id}")

        # **Fix: Ensure we have correct token IDs**
        if None in {self.pad_id, self.unk_id, self.bos_id, self.eos_id}:
            raise ValueError(
                f" Missing special tokens! PAD={self.pad_id}, UNK={self.unk_id}, BOS={self.bos_id}, EOS={self.eos_id}"
            )
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Returns a dictionary with properly formatted input tensors.
        """
        text = self.data[idx]

        # Tokenize the input text
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)

        # **Fix: Check if labels are within vocab range**
        labels = input_ids.clone()
        if torch.any(labels < 0) or torch.any(labels >= self.vocab_size):
            raise ValueError(f"Invalid label values detected! Min: {labels.min().item()}, Max: {labels.max().item()}")

        # **Fix: Shift decoder input for autoregressive tasks**
        decoder_input_ids = torch.full((self.max_length,), self.pad_id, dtype=torch.long)
        decoder_input_ids[0] = self.bos_id  # Start with BOS
        decoder_input_ids[1:len(input_ids)] = input_ids[:-1]  # Shift input tokens
        
        # **Fix: Correct decoder attention mask**
        decoder_attention_mask = (decoder_input_ids != self.pad_id).long()

        return {
            "encoder_input_ids": input_ids.to(torch.long),
            "encoder_attention_mask": attention_mask,
            "decoder_input_ids": decoder_input_ids.to(torch.long),
            "decoder_attention_mask": decoder_attention_mask,
            "labels": labels.to(torch.long),
        }

    @staticmethod
    def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        Custom collate function to batch tensors properly.
        """
        batch = [sample for sample in batch if sample is not None]
        if len(batch) == 0:
            return {}

        return {
            key: torch.stack([sample[key] for sample in batch], dim=0)
            for key in batch[0].keys()
        }