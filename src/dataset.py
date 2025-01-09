import os
import torch
from torch.utils.data import Dataset, DataLoader
import yaml
from datasets import load_dataset
from tqdm import tqdm
from typing import List, Dict, Any, Optional, Union, Iterable
import logging
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

class DatasetProcessor:
    def __init__(self, config_path: Union[str, Path]):
        """
        Load datasets from a YAML config file and preprocess them.

        Args:
            config_path: Path to the YAML configuration file.
        """
        self.config_path = Path(config_path)
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
            
        self.logger = logging.getLogger(__name__)
        self.datasets: List[str] = []
        self.config: Dict[str, Any] = self._load_config()
        self.datasets = self._load_and_preprocess_datasets()

    def _load_config(self) -> Dict[str, Any]:
        """Load and validate the YAML configuration file."""
        try:
            with open(self.config_path, "r", encoding="utf-8") as file:
                config = yaml.safe_load(file)
            
            if not isinstance(config, dict) or 'datasets' not in config:
                raise ValueError("Config must contain a 'datasets' key")
            return config
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML format in config file: {e}")

    def _load_and_preprocess_datasets(self) -> List[str]:
        """
        Load datasets and preprocess text.

        Returns:
            A list of preprocessed text samples.
        
        Raises:
            ValueError: If dataset configuration is invalid.
        """
        data: List[str] = []
        
        for dataset in tqdm(self.config['datasets'], desc="Loading datasets"):
            if not isinstance(dataset, dict) or 'type' not in dataset:
                raise ValueError(f"Invalid dataset configuration: {dataset}")
                
            try:
                if dataset['type'] == 'local':
                    data.extend(self._process_local_dataset(dataset))
                elif dataset['type'] == 'huggingface':
                    data.extend(self._process_huggingface_dataset(dataset))
                else:
                    self.logger.warning(f"Unknown dataset type: {dataset['type']}")
            except Exception as e:
                self.logger.error(f"Error processing dataset {dataset.get('name', 'unknown')}: {e}")
                if self.config.get('strict', False):
                    raise

        return data

    def _process_local_dataset(self, dataset: Dict[str, Any]) -> List[str]:
        """Process local dataset files."""
        if 'path' not in dataset:
            raise ValueError("Local dataset must specify 'path'")
            
        path = Path(dataset['path'])
        if not path.exists():
            raise FileNotFoundError(f"Dataset path not found: {path}")
            
        patterns = dataset.get('patterns', ['*.txt'])
        processed_data: List[str] = []
        
        for pattern in patterns:
            files = list(path.glob(pattern))
            for file in tqdm(files, desc=f"Processing {pattern} files"):
                try:
                    text = file.read_text(encoding="utf-8")
                    processed_data.extend(self._preprocess_text(text.splitlines()))
                except Exception as e:
                    self.logger.error(f"Error processing file {file}: {e}")
                    if self.config.get('strict', False):
                        raise
                        
        return processed_data

    def _process_huggingface_dataset(self, dataset: Dict[str, Any]) -> List[str]:
        """Process Hugging Face dataset."""
        if 'name' not in dataset:
            raise ValueError("Hugging Face dataset must specify 'name'")
            
        hf_data = load_dataset(
            dataset['name'],
            split=dataset.get('split', 'train'),
            cache_dir=dataset.get('cache_dir', './cache')
        )
        
        field = dataset.get('field', 'text')
        if field not in hf_data.features:
            raise ValueError(f"Field '{field}' not found in dataset")
            
        return self._preprocess_text(hf_data[field])

    def _preprocess_text(self, texts: Iterable[str]) -> List[str]:
        """
        Preprocess text data by stripping whitespace and filtering empty lines.

        Args:
            texts: Iterable of raw text samples.

        Returns:
            List of preprocessed text samples.
        """
        processed = []
        for text in texts:
            if isinstance(text, str):
                cleaned = text.strip()
                if cleaned and len(cleaned) >= self.config.get('min_length', 1):
                    processed.append(cleaned)
        return processed

    def save_processed_data(self, output_path: Union[str, Path]) -> None:
        """
        Save preprocessed data to a file and log its size.

        Args:
            output_path: Path to save the processed data.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save dataset
        torch.save(self.datasets, output_path)
        size_in_mb = output_path.stat().st_size / (1024 * 1024)  # Convert to MB
        self.logger.info(f"Processed dataset saved at {output_path}. Size: {size_in_mb:.2f} MB, Samples: {len(self.datasets)}.")

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
        text = self.data[idx]
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        return {
            "input_ids": encoding.input_ids.squeeze(),
            "attention_mask": encoding.attention_mask.squeeze(),
            "labels": encoding.input_ids.squeeze()  # For autoregressive tasks
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
