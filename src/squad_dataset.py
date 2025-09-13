# src/squad_dataset.py
"""
SQuAD Dataset Handler for SPARSA-LM
Handles loading and processing of SQuAD dataset for question-answering tasks
"""

import logging
import torch
from torch.utils.data import Dataset
from typing import List, Dict, Any, Optional, Tuple
from datasets import load_dataset
from tqdm import tqdm
import numpy as np

class SQuADDataset(Dataset):
    """
    Dataset class for SQuAD question-answering tasks.
    Formats data for encoder-decoder architecture with proper attention masks.
    """
    
    def __init__(
        self,
        tokenizer: Any,
        max_length: int = 512,
        max_answer_length: int = 128,
        split: str = "train",
        version: str = "v2.0",  # v1.1 or v2.0
        cache_dir: Optional[str] = None,
        subsample: Optional[int] = None  # For testing with smaller dataset
    ):
        """
        Initialize SQuAD dataset.
        
        Args:
            tokenizer: Tokenizer instance
            max_length: Maximum sequence length for context + question
            max_answer_length: Maximum length for answer
            split: Dataset split ('train' or 'validation')
            version: SQuAD version ('v1.1' or 'v2.0')
            cache_dir: Directory to cache the dataset
            subsample: If set, only use this many examples (for testing)
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_answer_length = max_answer_length
        self.split = split
        self.version = version
        
        # Load SQuAD dataset
        dataset_name = "squad" if version == "v1.1" else "squad_v2"
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Loading SQuAD {version} dataset, split: {split}")
        
        self.dataset = load_dataset(
            dataset_name,
            split=split,
            cache_dir=cache_dir
        )
        
        if subsample:
            self.dataset = self.dataset.select(range(min(subsample, len(self.dataset))))
            self.logger.info(f"Subsampled to {len(self.dataset)} examples")
        
        # Process and cache all examples
        self.processed_examples = []
        self._process_dataset()
        
        # Get special token IDs
        self.pad_id = tokenizer.convert_tokens_to_ids("[PAD]")
        self.bos_id = tokenizer.convert_tokens_to_ids("[BOS]")
        self.eos_id = tokenizer.convert_tokens_to_ids("[EOS]")
        self.sep_id = tokenizer.convert_tokens_to_ids("[SEP]")
        
    def _process_dataset(self):
        """Process all examples in the dataset."""
        self.logger.info(f"Processing {len(self.dataset)} examples...")
        
        for idx, example in enumerate(tqdm(self.dataset, desc="Processing SQuAD")):
            processed = self._process_example(example, idx)
            if processed:
                self.processed_examples.append(processed)
        
        self.logger.info(f"Successfully processed {len(self.processed_examples)} examples")
    
    def _process_example(self, example: Dict[str, Any], idx: int) -> Optional[Dict[str, Any]]:
        """
        Process a single SQuAD example.
        
        Format for encoder-decoder:
        - Encoder input: [BOS] question [SEP] context [EOS]
        - Decoder input: [BOS] answer [EOS]
        """
        try:
            question = example['question']
            context = example['context']
            answers = example.get('answers', {})
            
            # Handle both SQuAD v1.1 and v2.0 formats
            if answers and 'text' in answers and answers['text']:
                # For multiple answers, take the first one
                answer_text = answers['text'][0] if isinstance(answers['text'], list) else answers['text']
                answer_start = answers['answer_start'][0] if isinstance(answers['answer_start'], list) else answers['answer_start']
                has_answer = True
            else:
                # For unanswerable questions in SQuAD v2.0
                answer_text = "No answer available"
                answer_start = -1
                has_answer = False
            
            # Format input for encoder: question + context
            encoder_input = f"{self.tokenizer.bos_token} {question} {self.tokenizer.sep_token} {context} {self.tokenizer.eos_token}"
            
            # Format output for decoder: answer
            decoder_output = f"{self.tokenizer.bos_token} {answer_text} {self.tokenizer.eos_token}"
            
            return {
                'question': question,
                'context': context,
                'answer': answer_text,
                'answer_start': answer_start,
                'has_answer': has_answer,
                'encoder_input': encoder_input,
                'decoder_output': decoder_output,
                'id': example.get('id', f'example_{idx}')
            }
            
        except Exception as e:
            self.logger.warning(f"Error processing example: {str(e)}")
            return None
    
    def __len__(self) -> int:
        return len(self.processed_examples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single example formatted for training/inference.
        
        Returns:
            Dictionary with:
            - encoder_input_ids: Tokenized question + context
            - encoder_attention_mask: Attention mask for encoder
            - decoder_input_ids: Shifted answer tokens for teacher forcing
            - decoder_attention_mask: Attention mask for decoder  
            - labels: Target answer tokens
            - metadata: Original text data for analysis
        """
        example = self.processed_examples[idx]
        
        # Tokenize encoder input (question + context)
        encoder_encoding = self.tokenizer(
            example['encoder_input'],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        encoder_input_ids = encoder_encoding["input_ids"].squeeze(0)
        encoder_attention_mask = encoder_encoding["attention_mask"].squeeze(0)
        
        # Tokenize decoder output (answer)
        decoder_encoding = self.tokenizer(
            example['decoder_output'],
            max_length=self.max_answer_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        labels = decoder_encoding["input_ids"].squeeze(0)
        
        # Create decoder input (shifted right for teacher forcing)
        decoder_input_ids = torch.full((self.max_answer_length,), self.pad_id, dtype=torch.long)
        decoder_input_ids[0] = self.bos_id
        decoder_input_ids[1:len(labels)] = labels[:-1]
        decoder_attention_mask = (decoder_input_ids != self.pad_id).long()
        
        return {
            "encoder_input_ids": encoder_input_ids,
            "encoder_attention_mask": encoder_attention_mask,
            "decoder_input_ids": decoder_input_ids,
            "decoder_attention_mask": decoder_attention_mask,
            "labels": labels,
            "metadata": {
                "question": example['question'],
                "answer": example['answer'],
                "has_answer": example['has_answer'],
                "id": example['id']
            }
        }
    
    @staticmethod
    def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, Any]:
        """
        Custom collate function for batching.
        Handles both tensor data and metadata.
        """
        # Separate tensor data from metadata
        tensor_keys = ["encoder_input_ids", "encoder_attention_mask", 
                      "decoder_input_ids", "decoder_attention_mask", "labels"]
        
        collated = {}
        
        # Stack tensor data
        for key in tensor_keys:
            if key in batch[0]:
                collated[key] = torch.stack([sample[key] for sample in batch])
        
        # Collect metadata
        if "metadata" in batch[0]:
            collated["metadata"] = [sample["metadata"] for sample in batch]
        
        return collated


class SQuADDatasetProcessor:
    """
    Processor for handling SQuAD dataset with train/validation splits
    and integration with the existing SPARSA-LM training pipeline.
    """
    
    def __init__(
        self,
        tokenizer: Any,
        max_length: int = 512,
        max_answer_length: int = 128,
        version: str = "v2.0",
        cache_dir: Optional[str] = "./cache",
        train_subsample: Optional[int] = None,
        val_subsample: Optional[int] = None
    ):
        """
        Initialize SQuAD dataset processor.
        
        Args:
            tokenizer: Tokenizer instance
            max_length: Maximum sequence length
            max_answer_length: Maximum answer length
            version: SQuAD version
            cache_dir: Cache directory
            train_subsample: Subsample size for training set
            val_subsample: Subsample size for validation set
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_answer_length = max_answer_length
        self.version = version
        self.cache_dir = cache_dir
        
        self.logger = logging.getLogger(__name__)
        
        # Initialize datasets
        self.train_dataset = SQuADDataset(
            tokenizer=tokenizer,
            max_length=max_length,
            max_answer_length=max_answer_length,
            split="train",
            version=version,
            cache_dir=cache_dir,
            subsample=train_subsample
        )
        
        # SQuAD uses 'validation' split instead of 'test'
        self.val_dataset = SQuADDataset(
            tokenizer=tokenizer,
            max_length=max_length,
            max_answer_length=max_answer_length,
            split="validation",
            version=version,
            cache_dir=cache_dir,
            subsample=val_subsample
        )
        
        self.logger.info(f"Initialized SQuAD processor:")
        self.logger.info(f"  Training examples: {len(self.train_dataset)}")
        self.logger.info(f"  Validation examples: {len(self.val_dataset)}")
    
    def get_train_dataset(self) -> SQuADDataset:
        """Get training dataset."""
        return self.train_dataset
    
    def get_val_dataset(self) -> SQuADDataset:
        """Get validation dataset."""
        return self.val_dataset
    
    def get_dataloaders(
        self,
        batch_size: int,
        num_workers: int = 4,
        pin_memory: bool = True
    ) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
        """
        Create DataLoaders for training and validation.
        
        Returns:
            Tuple of (train_loader, val_loader)
        """
        from torch.utils.data import DataLoader
        
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=SQuADDataset.collate_fn,
            drop_last=True
        )
        
        val_loader = DataLoader(
            self.val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=SQuADDataset.collate_fn,
            drop_last=False
        )
        
        return train_loader, val_loader


# QA-specific metrics
class QAMetrics:
    """Metrics for evaluating question-answering performance."""
    
    @staticmethod
    def compute_exact_match(prediction: str, ground_truth: str) -> float:
        """Compute exact match score."""
        return float(prediction.strip().lower() == ground_truth.strip().lower())
    
    @staticmethod
    def compute_f1(prediction: str, ground_truth: str) -> float:
        """Compute F1 score between prediction and ground truth."""
        pred_tokens = prediction.lower().split()
        truth_tokens = ground_truth.lower().split()
        
        if len(pred_tokens) == 0 or len(truth_tokens) == 0:
            return float(pred_tokens == truth_tokens)
        
        common_tokens = set(pred_tokens) & set(truth_tokens)
        
        if len(common_tokens) == 0:
            return 0.0
        
        precision = len(common_tokens) / len(pred_tokens)
        recall = len(common_tokens) / len(truth_tokens)
        
        f1 = 2 * (precision * recall) / (precision + recall)
        return f1
    
    @staticmethod
    def evaluate_batch(predictions: List[str], ground_truths: List[str]) -> Dict[str, float]:
        """Evaluate a batch of predictions."""
        exact_matches = []
        f1_scores = []
        
        for pred, truth in zip(predictions, ground_truths):
            exact_matches.append(QAMetrics.compute_exact_match(pred, truth))
            f1_scores.append(QAMetrics.compute_f1(pred, truth))
        
        return {
            "exact_match": np.mean(exact_matches),
            "f1": np.mean(f1_scores),
            "num_examples": len(predictions)
        }
    