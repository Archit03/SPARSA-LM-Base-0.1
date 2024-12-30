import os
import torch
from torch.utils.data import Dataset
import yaml
from datasets import load_dataset

class DatasetProcessor:
    def __init__(self, config_path):
        self.config_path = config_path
        self.datasets = self.load_datasets()

    def load_config(self):
        """
        Load the datasets from the YAML config file.
        """
        with open(self.config_path, "r") as file:
            config = yaml.safe_load(file)
            return config 
    
    def load_local(self, path, patterns):
        """
        Load a local dataset.
        """
        files = []
        for pattern in patterns:
            files.extend([os.path.join(path, file) for file in os.listdir(path) if file.endswith(pattern)])
        return files
    
    def load_huggingface(self, dataset_name, split, cache_dir):
        """
        Load a HuggingFace dataset.
        """
        dataset = load_dataset(dataset_name, split=split, cache_dir=cache_dir)
        return dataset
    
    def process(self, output_dir):
        """
        Process the datasets.
        """
        for dataset in self.datasets['datasets']:
            if dataset['type'] == 'local':
                files = self.load_local(dataset['config']['path'], dataset['config'].get('patterns', ['*.txt', '*.json', '*.csv', '*.tsv', '*.xml', '*.html', '*.pdf', '*.docx', '*.pptx', '*.xlsx', '*.parquet', '*.avro', '*.jsonl', '*.jsonl.gz', '*.csv.gz', '*.tsv.gz', '*.parquet.gz', '*.avro.gz']))
                print(f"Processing the local dataset {dataset['name']} with {len(files)} files.")

            elif dataset['type'] == 'huggingface':
                hf_data = self.load_huggingface(
                    dataset['config']['dataset_name'],
                    dataset['config']['split', 'train'],
                    dataset['config']['cache_dir', './cache']
                )
                print(f"Processing the HuggingFace dataset {dataset['name']} with {len(hf_data)} samples.")

        os.makedirs(output_dir, exist_ok=True)
        torch.save(self.datasets, os.path.join(output_dir, "datasets.pt"))

class TextDataset(Dataset):
    def __init__(self, data, tokenizer, max_length):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data[idx]
        encoding = self.tokenizer(text, max_length=self.max_length, padding="max_length", truncation=True, return_tensors="pt")
        return {key: encoding[key].squeeze() for key in encoding}

    @staticmethod
    def collate_fn(batch):
        return {key: torch.stack([sample[key] for sample in batch]) for key in batch[0].keys()}
