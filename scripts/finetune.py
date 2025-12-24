#!/usr/bin/env python3
"""
SPARSA-LM Finetuning Script
Entry point for DAPO/VAPO RL-based finetuning
"""

import os
import sys
import argparse
import logging
from pathlib import Path

import torch
import torch.distributed as dist

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model import AutoRegressiveLM, ModelConfig
from src.data import RLDataset, DatasetConfig
from src.finetune import FinetuneConfig, RLTrainer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="SPARSA-LM DAPO/VAPO Finetuning")

    # Model
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to pretrained model")
    parser.add_argument("--tokenizer_path", type=str, required=True,
                        help="Path to tokenizer")

    # Algorithm selection
    parser.add_argument("--algorithm", type=str, default="dapo",
                        choices=["dapo", "vapo"],
                        help="RL algorithm to use")

    # Output
    parser.add_argument("--output_dir", type=str, default="outputs/finetune",
                        help="Output directory")

    # Training configuration
    parser.add_argument("--num_epochs", type=int, default=3,
                        help="Number of training epochs")
    parser.add_argument("--max_steps", type=int, default=-1,
                        help="Maximum training steps")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Per-device batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4,
                        help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=1e-5,
                        help="Learning rate")
    parser.add_argument("--ppo_epochs", type=int, default=4,
                        help="PPO epochs per batch")

    # DAPO-specific
    parser.add_argument("--clip_range_upper", type=float, default=0.2,
                        help="DAPO upper clip range")
    parser.add_argument("--clip_range_lower", type=float, default=0.1,
                        help="DAPO lower clip range")
    parser.add_argument("--dynamic_sampling", action="store_true", default=True,
                        help="Use dynamic sampling temperature")
    parser.add_argument("--entropy_coef", type=float, default=0.01,
                        help="Entropy coefficient")

    # VAPO-specific
    parser.add_argument("--value_coef", type=float, default=0.5,
                        help="Value model coefficient")
    parser.add_argument("--dense_reward", action="store_true", default=True,
                        help="Use dense rewards")
    parser.add_argument("--reward_smoothing", type=float, default=0.1,
                        help="Reward smoothing factor")

    # KL control
    parser.add_argument("--init_kl_coef", type=float, default=0.1,
                        help="Initial KL coefficient")
    parser.add_argument("--target_kl", type=float, default=0.1,
                        help="Target KL divergence")

    # Generation
    parser.add_argument("--max_new_tokens", type=int, default=256,
                        help="Maximum new tokens to generate")
    parser.add_argument("--generation_temperature", type=float, default=0.7,
                        help="Generation temperature")

    # Reward
    parser.add_argument("--reward_model_path", type=str, default=None,
                        help="Path to reward model")

    # Data
    parser.add_argument("--train_data", type=str, required=True,
                        help="Path to training prompts (JSON/JSONL)")
    parser.add_argument("--max_seq_length", type=int, default=2048,
                        help="Maximum sequence length")

    # Mixed precision
    parser.add_argument("--bf16", action="store_true", default=True,
                        help="Use BF16 mixed precision")

    # DeepSpeed
    parser.add_argument("--deepspeed", action="store_true", default=True,
                        help="Use DeepSpeed")
    parser.add_argument("--deepspeed_stage", type=int, default=2,
                        help="DeepSpeed ZeRO stage")

    # Checkpointing
    parser.add_argument("--save_steps", type=int, default=500,
                        help="Save checkpoint every N steps")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,
                        help="Resume from checkpoint")

    # Logging
    parser.add_argument("--logging_steps", type=int, default=10,
                        help="Log every N steps")
    parser.add_argument("--wandb_project", type=str, default="sparsa-lm-finetune",
                        help="W&B project name")

    # Distributed
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="Local rank for distributed training")

    # Seed
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")

    args = parser.parse_args()
    return args


def load_prompts(path: str) -> list:
    """Load prompts from JSON or JSONL file."""
    import json

    prompts = []
    if path.endswith('.jsonl'):
        with open(path, 'r') as f:
            for line in f:
                data = json.loads(line)
                if isinstance(data, str):
                    prompts.append(data)
                elif isinstance(data, dict):
                    prompts.append(data.get('prompt', data.get('text', '')))
    else:
        with open(path, 'r') as f:
            data = json.load(f)
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, str):
                        prompts.append(item)
                    elif isinstance(item, dict):
                        prompts.append(item.get('prompt', item.get('text', '')))

    return prompts


def setup_distributed(args):
    """Initialize distributed training."""
    if args.local_rank != -1:
        torch.cuda.set_device(args.local_rank)
        dist.init_process_group(backend="nccl")
        args.world_size = dist.get_world_size()
    else:
        args.world_size = 1

    return args


def main():
    args = parse_args()

    # Setup distributed
    args = setup_distributed(args)

    # Set seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    logger.info(f"Starting {args.algorithm.upper()} finetuning with args: {args}")

    # Load tokenizer
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)

    # Load model
    logger.info(f"Loading model from {args.model_path}")
    model = AutoRegressiveLM.from_pretrained(args.model_path)

    num_params = model.count_parameters()
    logger.info(f"Model parameters: {num_params:,}")

    # Load prompts
    prompts = load_prompts(args.train_data)
    logger.info(f"Loaded {len(prompts)} prompts")

    # Create dataset config
    dataset_config = DatasetConfig(
        tokenizer_path=args.tokenizer_path,
        max_seq_length=args.max_seq_length,
    )

    # Create dataset
    train_dataset = RLDataset(
        config=dataset_config,
        tokenizer=tokenizer,
        prompts=prompts,
    )

    # Create finetuning config
    finetune_config = FinetuneConfig(
        output_dir=args.output_dir,
        checkpoint_dir=os.path.join(args.output_dir, "checkpoints"),
        logging_dir=os.path.join(args.output_dir, "logs"),
        num_epochs=args.num_epochs,
        max_steps=args.max_steps,
        per_device_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        ppo_epochs=args.ppo_epochs,
        # Algorithm selection
        dapo_enabled=(args.algorithm == "dapo"),
        vapo_enabled=(args.algorithm == "vapo"),
        # DAPO
        clip_range_upper=args.clip_range_upper,
        clip_range_lower=args.clip_range_lower,
        dynamic_sampling=args.dynamic_sampling,
        entropy_coef=args.entropy_coef,
        # VAPO
        value_model_coef=args.value_coef,
        dense_reward=args.dense_reward,
        reward_smoothing=args.reward_smoothing,
        # KL
        init_kl_coef=args.init_kl_coef,
        target_kl=args.target_kl,
        # Generation
        max_new_tokens=args.max_new_tokens,
        generation_temperature=args.generation_temperature,
        # Mixed precision
        bf16=args.bf16,
        # DeepSpeed
        use_deepspeed=args.deepspeed,
        deepspeed_stage=args.deepspeed_stage,
        # Checkpointing
        save_steps=args.save_steps,
        # Logging
        logging_steps=args.logging_steps,
        wandb_project=args.wandb_project,
        # Seed
        seed=args.seed,
    )

    # Load reward model if specified
    reward_model = None
    if args.reward_model_path:
        logger.info(f"Loading reward model from {args.reward_model_path}")
        reward_model = AutoRegressiveLM.from_pretrained(args.reward_model_path)

    # Create trainer
    trainer = RLTrainer(
        model=model,
        config=finetune_config,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        reward_model=reward_model,
    )

    # Train
    metrics = trainer.train()

    logger.info(f"Finetuning completed. Metrics: {metrics}")


if __name__ == "__main__":
    main()
