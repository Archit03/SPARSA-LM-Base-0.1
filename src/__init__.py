"""
SPARSA-LM: Sparse Attention Language Model
AutoRegressive Architecture with DAPO/VAPO RL Finetuning
"""

__version__ = "0.2.0"

from .model import AutoRegressiveLM, ModelConfig
from .data import PretrainDataset, FinetuneDataset, DataCollator
from .pretrain import PretrainConfig, PretrainTrainer
from .finetune import DAPOTrainer, VAPOTrainer, FinetuneConfig

__all__ = [
    "AutoRegressiveLM",
    "ModelConfig",
    "PretrainDataset",
    "FinetuneDataset",
    "DataCollator",
    "PretrainConfig",
    "PretrainTrainer",
    "DAPOTrainer",
    "VAPOTrainer",
    "FinetuneConfig",
]
