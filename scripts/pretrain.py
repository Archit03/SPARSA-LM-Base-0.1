#!/usr/bin/env python3
"""
SPARSA-LM Pretraining Script
Entry point for distributed pretraining with DeepSpeed
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
from src.data import PretrainDataset, DatasetConfig
from src.pretrain import PretrainConfig, PretrainTrainer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="SPARSA-LM Pretraining")

    # Model configuration
    parser.add_argument("--model_size", type=str, default="base",
                        choices=["small", "base", "large", "xl"],
                        help="Model size configuration")
    parser.add_argument("--model_config", type=str, default=None,
                        help="Path to model config JSON")

    # Training configuration
    parser.add_argument("--output_dir", type=str, default="outputs/pretrain",
                        help="Output directory")
    parser.add_argument("--num_epochs", type=int, default=3,
                        help="Number of training epochs")
    parser.add_argument("--max_steps", type=int, default=-1,
                        help="Maximum training steps (-1 for epochs)")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Per-device batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8,
                        help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=3e-4,
                        help="Learning rate")
    parser.add_argument("--warmup_steps", type=int, default=2000,
                        help="Warmup steps")

    # Mixed precision
    parser.add_argument("--bf16", action="store_true", default=True,
                        help="Use BF16 mixed precision")
    parser.add_argument("--fp16", action="store_true", default=False,
                        help="Use FP16 mixed precision")

    # DeepSpeed
    parser.add_argument("--deepspeed", action="store_true", default=True,
                        help="Use DeepSpeed")
    parser.add_argument("--deepspeed_stage", type=int, default=2,
                        help="DeepSpeed ZeRO stage (0, 1, 2, 3)")

    # Data
    parser.add_argument("--tokenizer_path", type=str, required=True,
                        help="Path to tokenizer")
    parser.add_argument("--max_seq_length", type=int, default=2048,
                        help="Maximum sequence length")
    parser.add_argument("--datasets", nargs="+",
                        default=["c4", "wikipedia", "openwebtext"],
                        help="Datasets to use for pretraining")

    # Checkpointing
    parser.add_argument("--save_steps", type=int, default=1000,
                        help="Save checkpoint every N steps")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,
                        help="Resume from checkpoint")

    # Logging
    parser.add_argument("--logging_steps", type=int, default=10,
                        help="Log every N steps")
    parser.add_argument("--wandb_project", type=str, default="sparsa-lm-pretrain",
                        help="W&B project name")

    # Distributed
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="Local rank for distributed training")

    # Seed
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")

    args = parser.parse_args()
    return args


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

    logger.info(f"Starting pretraining with args: {args}")

    # Load tokenizer
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)

    # Create model config
    if args.model_config:
        model_config = ModelConfig.load(args.model_config)
    else:
        if args.model_size == "small":
            model_config = ModelConfig.small()
        elif args.model_size == "base":
            model_config = ModelConfig.base()
        elif args.model_size == "large":
            model_config = ModelConfig.large()
        else:
            model_config = ModelConfig.xl()

    model_config.vocab_size = len(tokenizer)
    model_config.use_gradient_checkpointing = True

    logger.info(f"Model config: {model_config.to_dict()}")

    # Create model
    model = AutoRegressiveLM(model_config)
    num_params = model.count_parameters()
    logger.info(f"Model parameters: {num_params:,}")

    # Create dataset config
    dataset_config = DatasetConfig(
        tokenizer_path=args.tokenizer_path,
        max_seq_length=args.max_seq_length,
        pretrain_datasets=args.datasets,
        streaming=True,
    )

    # Create dataset
    train_dataset = PretrainDataset(
        config=dataset_config,
        tokenizer=tokenizer,
        split="train",
    )

    # Create training config
    train_config = PretrainConfig(
        output_dir=args.output_dir,
        checkpoint_dir=os.path.join(args.output_dir, "checkpoints"),
        logging_dir=os.path.join(args.output_dir, "logs"),
        num_epochs=args.num_epochs,
        max_steps=args.max_steps,
        per_device_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        bf16=args.bf16,
        fp16=args.fp16,
        use_deepspeed=args.deepspeed,
        deepspeed_stage=args.deepspeed_stage,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        resume_from_checkpoint=args.resume_from_checkpoint,
        wandb_project=args.wandb_project,
        local_rank=args.local_rank,
        world_size=args.world_size,
        seed=args.seed,
    )

    # Create trainer
    trainer = PretrainTrainer(
        model=model,
        config=train_config,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
    )

    # Train
    metrics = trainer.train()

    logger.info(f"Training completed. Metrics: {metrics}")


if __name__ == "__main__":
    main()
