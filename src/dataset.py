import os
import glob
import logging
import numpy as np
import pandas as pd
import torch
import random

from tqdm import tqdm
from typing import List, Dict, Any, Union, Iterable
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

# ----------------------------------------------------------------------
# Updated Add Noise Function with Re-Padding for "delete"
# ----------------------------------------------------------------------
def add_noise_to_input(
    tokens: List[int],
    noise_type: str = "mask",
    mask_token_id: int = 4,  # e.g., tokenizer.convert_tokens_to_ids("[MASK]")
    prob: float = 0.3,
    pad_token_id: int = 0
) -> List[int]:
    """
    Apply noise to the encoder input for denoising training, preserving length.

    - "mask": Randomly replace tokens with `mask_token_id`
    - "delete": Randomly remove tokens, then re-pad to original length with `pad_token_id`
    - "permute": Randomly swap a percentage of tokens
    - "substitute": Randomly replace tokens with random IDs up to `max(tokens)`

    Args:
        tokens (list[int]): Tokenized sentence.
        noise_type (str): Type of corruption.
        mask_token_id (int): Token ID for a real [MASK] token (default=4).
        prob (float): Probability of applying noise to each token.
        pad_token_id (int): Used to re-pad after "delete" or as needed.

    Returns:
        list[int]: Noisy input tokens with the same length as original.
    """
    noisy_tokens = tokens[:]
    original_length = len(noisy_tokens)

    if noise_type == "mask":
        # Replace each token with mask_token_id with probability=prob
        for i in range(original_length):
            if random.random() < prob:
                noisy_tokens[i] = mask_token_id

    elif noise_type == "delete":
        # Remove tokens randomly, then re-pad to the original length
        kept = [t for t in noisy_tokens if random.random() > prob]
        # Re-pad if we deleted some tokens
        new_length = len(kept)
        # If everything got deleted, we keep an empty list
        # Now re-pad
        if new_length < original_length:
            # Copy the kept tokens to the front
            noisy_tokens = kept + [pad_token_id] * (original_length - new_length)
        else:
            # If nothing was deleted, we just keep them
            noisy_tokens = kept

    elif noise_type == "permute":
        # Randomly swap a fraction of tokens
        num_swaps = int(original_length * prob)
        for _ in range(num_swaps):
            i, j = random.randint(0, original_length - 1), random.randint(0, original_length - 1)
            noisy_tokens[i], noisy_tokens[j] = noisy_tokens[j], noisy_tokens[i]

    elif noise_type == "substitute":
        # Randomly replace tokens with random IDs up to the local max
        # If you want a full vocab-based substitution, pass in a vocab_size param.
        max_id = max(noisy_tokens) if noisy_tokens else 0
        for i in range(original_length):
            if random.random() < prob and max_id > 0:
                noisy_tokens[i] = random.randint(0, max_id)

    return noisy_tokens


# ----------------------------------------------------------------------
# Dataset Processor Class
# ----------------------------------------------------------------------
class DatasetProcessor:
    def __init__(self, source_dir: str, split: Dict[str, Any], preprocessing_config: Dict[str, Any]):
        self.source_dir = source_dir
        self.test_size = split.get("test_size", 0.2)
        self.random_state = split.get("random_state", 42)
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
        self.logger.info(f"Loading data from {self.source_dir}...")
        loaded_data = []
        
        patterns = self.preprocessing_config.get("patterns", ["*.txt", "*.csv"])
        for pattern in patterns:
            files = glob.glob(os.path.join(self.source_dir, pattern))
            for file in tqdm(files, desc=f"Processing files matching {pattern}"):
                try:
                    if file.endswith(".csv"):
                        df = pd.read_csv(file)
                        column = self.preprocessing_config.get("csv_text_column", "text")
                        loaded_data.extend(df[column].dropna().tolist())
                    else:
                        with open(file, "r", encoding="utf-8") as f:
                            loaded_data.extend(f.readlines())
                except Exception as e:
                    self.logger.error(f"Failed to process file {file}: {e}")
        
        self.logger.info(f"Loaded {len(loaded_data)} text samples.")
        return [line.strip() for line in loaded_data if line.strip()]

    def _load_and_split_data(self):
        self.logger.info("Loading and splitting data...")
        self.data = self._load_source_data()
        
        if len(self.data) > 0:
            self.train_data, self.val_data = train_test_split(
                self.data,
                test_size=self.test_size,
                random_state=self.random_state,
            )

    def get_train_dataset(self, tokenizer: Any, max_length: int) -> Dataset:
        return TextDataset(self.train_data, tokenizer, max_length)

    def get_val_dataset(self, tokenizer: Any, max_length: int) -> Dataset:
        return TextDataset(self.val_data, tokenizer, max_length)


# ----------------------------------------------------------------------
# Text Dataset Class with Noisy Encoder
# ----------------------------------------------------------------------
class TextDataset(Dataset):
    def __init__(self, data: List[str], tokenizer: Any, max_length: int):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        self.vocab_size = tokenizer.vocab_size
        self.pad_id = tokenizer.convert_tokens_to_ids("[PAD]")
        self.mask_id = tokenizer.convert_tokens_to_ids("[MASK]")
        self.bos_id = tokenizer.convert_tokens_to_ids("[BOS]")
        self.eos_id = tokenizer.convert_tokens_to_ids("[EOS]")
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Returns a dictionary with properly formatted input tensors.
        """
        text = self.data[idx]

        # 1) Tokenize original text
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        input_ids = encoding["input_ids"].squeeze(0)       # shape: [max_length]
        attention_mask = encoding["attention_mask"].squeeze(0)  # shape: [max_length]

        # 2) Apply noise to encoder input
        #    - If you want to use the real [MASK] token, pass mask_token_id=self.mask_id
        #    - If you want to do "delete" noise, it will re-pad to the same length with self.pad_id
        noisy_input_ids = add_noise_to_input(
            tokens=input_ids.tolist(), 
            noise_type="mask",   # or "delete", "permute", "substitute"
            mask_token_id=self.mask_id,  # Using [MASK] ID instead of [PAD]
            prob=0.3,
            pad_token_id=self.pad_id
        )
        noisy_input_ids = torch.tensor(noisy_input_ids, dtype=torch.long)

        # 3) Shift decoder input for autoregressive learning
        decoder_input_ids = torch.full((self.max_length,), self.pad_id, dtype=torch.long)
        decoder_input_ids[0] = self.bos_id  # Start with BOS token
        # Shift left by one for the rest
        decoder_input_ids[1:len(input_ids)] = input_ids[:-1]

        decoder_attention_mask = (decoder_input_ids != self.pad_id).long()

        return {
            "encoder_input_ids": noisy_input_ids,       # Noisy input for encoder
            "encoder_attention_mask": attention_mask,
            "decoder_input_ids": decoder_input_ids,
            "decoder_attention_mask": decoder_attention_mask,
            "labels": input_ids,  # Original sentence as labels
        }

    @staticmethod
    def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        batch = [sample for sample in batch if sample is not None]
        return {
            key: torch.stack([sample[key] for sample in batch], dim=0) 
            for key in batch[0].keys()
        }
