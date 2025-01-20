import argparse
import os
import yaml
import torch
import logging 
import time
import wandb
import numpy as np
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
from dataclasses import dataclass
from torch.utils.data import DataLoader, Dataset, random_split
from torch.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from transformers import PreTrainedTokenizerFast
from torch.nn.utils import clip_grad_norm_
from typing import Optional, Dict, Any, Tuple, List
from src.model import Transformer, TransformerConfig
from src.tokenizer import MedicalTokenizer
from src.utils import setup_logging, set_seed, save_checkpoint, load_checkpoint, count_parameters, ensure_dir

@dataclass
class TrainingState:
    """Tracks complete training state for resumability"""
    epoch: int
    global_step:int
    
