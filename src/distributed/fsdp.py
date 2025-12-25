"""
SPARSA-LM Fully Sharded Data Parallel (FSDP) Configuration
Production-quality FSDP setup for memory-efficient distributed training
"""

import os
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Type, Set, Callable
from pathlib import Path
import functools

import torch
import torch.nn as nn
import torch.distributed as dist

logger = logging.getLogger(__name__)

# Check FSDP availability
try:
    from torch.distributed.fsdp import (
        FullyShardedDataParallel as FSDP,
        MixedPrecision,
        BackwardPrefetch,
        ShardingStrategy,
        CPUOffload,
        StateDictType,
        FullStateDictConfig,
        ShardedStateDictConfig,
        LocalStateDictConfig,
    )
    from torch.distributed.fsdp.wrap import (
        transformer_auto_wrap_policy,
        size_based_auto_wrap_policy,
        enable_wrap,
        wrap,
        _or_policy,
    )
    from torch.distributed.fsdp.api import FullOptimStateDictConfig, ShardedOptimStateDictConfig
    FSDP_AVAILABLE = True
except ImportError:
    FSDP_AVAILABLE = False
    FSDP = None


@dataclass
class FSDPConfig:
    """
    Configuration for Fully Sharded Data Parallel training.

    FSDP enables training of models that don't fit in a single GPU's memory
    by sharding model parameters, gradients, and optimizer states across GPUs.

    Sharding Strategies:
    - FULL_SHARD: Shard parameters, gradients, and optimizer states (ZeRO-3)
    - SHARD_GRAD_OP: Shard gradients and optimizer states (ZeRO-2)
    - NO_SHARD: No sharding, equivalent to DDP
    - HYBRID_SHARD: Shard within node, replicate across nodes
    """

    # Sharding strategy
    sharding_strategy: str = "FULL_SHARD"  # FULL_SHARD, SHARD_GRAD_OP, NO_SHARD, HYBRID_SHARD

    # Mixed precision
    mixed_precision: bool = True
    mixed_precision_dtype: str = "bf16"  # bf16, fp16, fp32
    reduce_dtype: str = "fp32"  # dtype for gradient reduction
    buffer_dtype: str = "fp32"  # dtype for buffers

    # CPU offloading
    cpu_offload: bool = False
    offload_params: bool = False  # Offload parameters to CPU

    # Activation checkpointing
    activation_checkpointing: bool = True
    checkpoint_impl: str = "no_reentrant"  # reentrant, no_reentrant

    # Wrapping policy
    auto_wrap_policy: str = "transformer"  # transformer, size_based, none
    min_num_params: int = 100_000_000  # For size_based policy
    transformer_layer_cls: Optional[List[str]] = None  # For transformer policy

    # Backward prefetch
    backward_prefetch: str = "BACKWARD_PRE"  # BACKWARD_PRE, BACKWARD_POST, None

    # Synchronization
    sync_module_states: bool = True  # Sync module states from rank 0
    use_orig_params: bool = True  # Use original params for compatibility

    # Forward prefetch
    forward_prefetch: bool = True

    # Limit all-gathers
    limit_all_gathers: bool = True

    # State dict settings
    state_dict_type: str = "SHARDED"  # FULL, SHARDED, LOCAL
    rank0_only: bool = True  # Only save on rank 0 for FULL state dict

    # Debugging
    verbose: bool = False

    def get_sharding_strategy(self) -> 'ShardingStrategy':
        """Get PyTorch ShardingStrategy enum."""
        if not FSDP_AVAILABLE:
            raise RuntimeError("FSDP not available")

        strategies = {
            "FULL_SHARD": ShardingStrategy.FULL_SHARD,
            "SHARD_GRAD_OP": ShardingStrategy.SHARD_GRAD_OP,
            "NO_SHARD": ShardingStrategy.NO_SHARD,
            "HYBRID_SHARD": ShardingStrategy.HYBRID_SHARD,
            "_HYBRID_SHARD_ZERO2": ShardingStrategy._HYBRID_SHARD_ZERO2,
        }
        return strategies.get(self.sharding_strategy, ShardingStrategy.FULL_SHARD)

    def get_mixed_precision(self) -> Optional['MixedPrecision']:
        """Get MixedPrecision configuration."""
        if not self.mixed_precision or not FSDP_AVAILABLE:
            return None

        dtype_map = {
            "bf16": torch.bfloat16,
            "fp16": torch.float16,
            "fp32": torch.float32,
        }

        return MixedPrecision(
            param_dtype=dtype_map.get(self.mixed_precision_dtype, torch.bfloat16),
            reduce_dtype=dtype_map.get(self.reduce_dtype, torch.float32),
            buffer_dtype=dtype_map.get(self.buffer_dtype, torch.float32),
        )

    def get_backward_prefetch(self) -> Optional['BackwardPrefetch']:
        """Get BackwardPrefetch configuration."""
        if not FSDP_AVAILABLE:
            return None

        prefetch_map = {
            "BACKWARD_PRE": BackwardPrefetch.BACKWARD_PRE,
            "BACKWARD_POST": BackwardPrefetch.BACKWARD_POST,
            None: None,
            "None": None,
        }
        return prefetch_map.get(self.backward_prefetch)

    def get_cpu_offload(self) -> Optional['CPUOffload']:
        """Get CPUOffload configuration."""
        if not self.cpu_offload or not FSDP_AVAILABLE:
            return None

        return CPUOffload(offload_params=self.offload_params)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {k: v for k, v in self.__dict__.items()}

    @classmethod
    def for_pretraining(cls, model_size: str = "base") -> "FSDPConfig":
        """Optimized configuration for pretraining."""
        configs = {
            "small": cls(
                sharding_strategy="SHARD_GRAD_OP",
                activation_checkpointing=False,
                cpu_offload=False,
            ),
            "base": cls(
                sharding_strategy="FULL_SHARD",
                activation_checkpointing=True,
                cpu_offload=False,
            ),
            "large": cls(
                sharding_strategy="FULL_SHARD",
                activation_checkpointing=True,
                cpu_offload=False,
            ),
            "xl": cls(
                sharding_strategy="FULL_SHARD",
                activation_checkpointing=True,
                cpu_offload=True,
                offload_params=False,
            ),
        }
        return configs.get(model_size, configs["base"])

    @classmethod
    def for_finetuning(cls) -> "FSDPConfig":
        """Optimized configuration for fine-tuning."""
        return cls(
            sharding_strategy="FULL_SHARD",
            activation_checkpointing=True,
            mixed_precision=True,
            mixed_precision_dtype="bf16",
            cpu_offload=False,
        )


def get_fsdp_policy(
    config: FSDPConfig,
    transformer_layer_cls: Optional[Set[Type[nn.Module]]] = None,
) -> Optional[Callable]:
    """
    Create FSDP auto wrap policy.

    Args:
        config: FSDP configuration
        transformer_layer_cls: Set of transformer layer classes to wrap

    Returns:
        Auto wrap policy function
    """
    if not FSDP_AVAILABLE:
        return None

    if config.auto_wrap_policy == "none":
        return None

    if config.auto_wrap_policy == "transformer":
        if transformer_layer_cls is None:
            # Default transformer layer classes
            from ..model.layers import TransformerBlock
            transformer_layer_cls = {TransformerBlock}

        return functools.partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls=transformer_layer_cls,
        )

    elif config.auto_wrap_policy == "size_based":
        return functools.partial(
            size_based_auto_wrap_policy,
            min_num_params=config.min_num_params,
        )

    return None


def setup_fsdp(
    model: nn.Module,
    config: FSDPConfig,
    device_id: Optional[int] = None,
    transformer_layer_cls: Optional[Set[Type[nn.Module]]] = None,
) -> 'FSDP':
    """
    Wrap model with FSDP.

    Args:
        model: Model to wrap
        config: FSDP configuration
        device_id: CUDA device ID
        transformer_layer_cls: Transformer layer classes for wrapping

    Returns:
        FSDP-wrapped model
    """
    if not FSDP_AVAILABLE:
        raise RuntimeError("FSDP not available. Please upgrade PyTorch.")

    if device_id is None:
        device_id = int(os.environ.get("LOCAL_RANK", 0))

    # Apply activation checkpointing before FSDP wrapping
    if config.activation_checkpointing:
        _apply_activation_checkpointing(model, config, transformer_layer_cls)

    # Get auto wrap policy
    auto_wrap_policy = get_fsdp_policy(config, transformer_layer_cls)

    # Create FSDP model
    fsdp_model = FSDP(
        model,
        sharding_strategy=config.get_sharding_strategy(),
        mixed_precision=config.get_mixed_precision(),
        cpu_offload=config.get_cpu_offload(),
        backward_prefetch=config.get_backward_prefetch(),
        auto_wrap_policy=auto_wrap_policy,
        device_id=device_id,
        sync_module_states=config.sync_module_states,
        use_orig_params=config.use_orig_params,
        forward_prefetch=config.forward_prefetch,
        limit_all_gathers=config.limit_all_gathers,
    )

    if config.verbose:
        logger.info(f"FSDP model created with config:\n{config.to_dict()}")
        _log_fsdp_stats(fsdp_model)

    return fsdp_model


def _apply_activation_checkpointing(
    model: nn.Module,
    config: FSDPConfig,
    transformer_layer_cls: Optional[Set[Type[nn.Module]]] = None,
):
    """Apply activation checkpointing to model."""
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
        checkpoint_wrapper,
        CheckpointImpl,
        apply_activation_checkpointing,
    )

    if transformer_layer_cls is None:
        from ..model.layers import TransformerBlock
        transformer_layer_cls = {TransformerBlock}

    check_fn = lambda submodule: isinstance(submodule, tuple(transformer_layer_cls))

    checkpoint_impl = (
        CheckpointImpl.NO_REENTRANT
        if config.checkpoint_impl == "no_reentrant"
        else CheckpointImpl.REENTRANT
    )

    apply_activation_checkpointing(
        model,
        checkpoint_wrapper_fn=functools.partial(
            checkpoint_wrapper,
            checkpoint_impl=checkpoint_impl,
        ),
        check_fn=check_fn,
    )


def _log_fsdp_stats(model: 'FSDP'):
    """Log FSDP statistics."""
    def count_params(module):
        return sum(p.numel() for p in module.parameters())

    logger.info(f"FSDP wrapped model parameters: {count_params(model):,}")


def save_fsdp_checkpoint(
    model: 'FSDP',
    optimizer: torch.optim.Optimizer,
    checkpoint_dir: str,
    config: FSDPConfig,
    epoch: int = 0,
    step: int = 0,
    metrics: Optional[Dict[str, float]] = None,
):
    """
    Save FSDP checkpoint.

    Args:
        model: FSDP model
        optimizer: Optimizer
        checkpoint_dir: Directory to save checkpoint
        config: FSDP configuration
        epoch: Current epoch
        step: Current step
        metrics: Training metrics
    """
    if not FSDP_AVAILABLE:
        raise RuntimeError("FSDP not available")

    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    rank = dist.get_rank() if dist.is_initialized() else 0

    # Configure state dict type
    if config.state_dict_type == "FULL":
        state_dict_type = StateDictType.FULL_STATE_DICT
        state_dict_config = FullStateDictConfig(
            offload_to_cpu=True,
            rank0_only=config.rank0_only,
        )
        optim_state_dict_config = FullOptimStateDictConfig(
            offload_to_cpu=True,
            rank0_only=config.rank0_only,
        )
    elif config.state_dict_type == "SHARDED":
        state_dict_type = StateDictType.SHARDED_STATE_DICT
        state_dict_config = ShardedStateDictConfig(offload_to_cpu=True)
        optim_state_dict_config = ShardedOptimStateDictConfig(offload_to_cpu=True)
    else:  # LOCAL
        state_dict_type = StateDictType.LOCAL_STATE_DICT
        state_dict_config = LocalStateDictConfig(offload_to_cpu=True)
        optim_state_dict_config = None

    # Get model state dict
    with FSDP.state_dict_type(model, state_dict_type, state_dict_config, optim_state_dict_config):
        model_state = model.state_dict()
        optim_state = FSDP.optim_state_dict(model, optimizer) if optim_state_dict_config else optimizer.state_dict()

    # Save based on state dict type
    if config.state_dict_type == "FULL":
        if rank == 0 or not config.rank0_only:
            torch.save(model_state, checkpoint_dir / "model.pt")
            torch.save(optim_state, checkpoint_dir / "optimizer.pt")
    elif config.state_dict_type == "SHARDED":
        # Each rank saves its shard
        torch.save(model_state, checkpoint_dir / f"model_rank{rank}.pt")
        torch.save(optim_state, checkpoint_dir / f"optimizer_rank{rank}.pt")
    else:  # LOCAL
        torch.save(model_state, checkpoint_dir / f"model_rank{rank}.pt")
        torch.save(optim_state, checkpoint_dir / f"optimizer_rank{rank}.pt")

    # Save metadata on rank 0
    if rank == 0:
        metadata = {
            "epoch": epoch,
            "step": step,
            "metrics": metrics or {},
            "fsdp_config": config.to_dict(),
            "world_size": dist.get_world_size() if dist.is_initialized() else 1,
        }
        torch.save(metadata, checkpoint_dir / "metadata.pt")

    # Synchronize
    if dist.is_initialized():
        dist.barrier()

    if rank == 0:
        logger.info(f"Saved FSDP checkpoint to {checkpoint_dir}")


def load_fsdp_checkpoint(
    model: 'FSDP',
    optimizer: Optional[torch.optim.Optimizer],
    checkpoint_dir: str,
    config: FSDPConfig,
) -> Dict[str, Any]:
    """
    Load FSDP checkpoint.

    Args:
        model: FSDP model
        optimizer: Optional optimizer
        checkpoint_dir: Directory containing checkpoint
        config: FSDP configuration

    Returns:
        Metadata dictionary
    """
    if not FSDP_AVAILABLE:
        raise RuntimeError("FSDP not available")

    checkpoint_dir = Path(checkpoint_dir)
    rank = dist.get_rank() if dist.is_initialized() else 0

    # Load metadata
    metadata = torch.load(checkpoint_dir / "metadata.pt", map_location="cpu")

    # Configure state dict type
    if config.state_dict_type == "FULL":
        state_dict_type = StateDictType.FULL_STATE_DICT
        state_dict_config = FullStateDictConfig(offload_to_cpu=True, rank0_only=False)
        optim_state_dict_config = FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=False)
        model_path = checkpoint_dir / "model.pt"
        optim_path = checkpoint_dir / "optimizer.pt"
    elif config.state_dict_type == "SHARDED":
        state_dict_type = StateDictType.SHARDED_STATE_DICT
        state_dict_config = ShardedStateDictConfig(offload_to_cpu=True)
        optim_state_dict_config = ShardedOptimStateDictConfig(offload_to_cpu=True)
        model_path = checkpoint_dir / f"model_rank{rank}.pt"
        optim_path = checkpoint_dir / f"optimizer_rank{rank}.pt"
    else:  # LOCAL
        state_dict_type = StateDictType.LOCAL_STATE_DICT
        state_dict_config = LocalStateDictConfig(offload_to_cpu=True)
        optim_state_dict_config = None
        model_path = checkpoint_dir / f"model_rank{rank}.pt"
        optim_path = checkpoint_dir / f"optimizer_rank{rank}.pt"

    # Load state dicts
    model_state = torch.load(model_path, map_location="cpu")

    with FSDP.state_dict_type(model, state_dict_type, state_dict_config, optim_state_dict_config):
        model.load_state_dict(model_state)

        if optimizer and optim_path.exists():
            optim_state = torch.load(optim_path, map_location="cpu")
            if config.state_dict_type in ["FULL", "SHARDED"]:
                optim_state_to_load = FSDP.optim_state_dict_to_load(model, optimizer, optim_state)
                optimizer.load_state_dict(optim_state_to_load)
            else:
                optimizer.load_state_dict(optim_state)

    # Synchronize
    if dist.is_initialized():
        dist.barrier()

    if rank == 0:
        logger.info(f"Loaded FSDP checkpoint from {checkpoint_dir}")

    return metadata


class FSDPTrainingContext:
    """
    Context manager for FSDP training.

    Handles:
    - Model wrapping
    - Gradient accumulation
    - Synchronization
    """

    def __init__(
        self,
        model: nn.Module,
        config: FSDPConfig,
        device_id: Optional[int] = None,
        transformer_layer_cls: Optional[Set[Type[nn.Module]]] = None,
    ):
        self.original_model = model
        self.config = config
        self.device_id = device_id
        self.transformer_layer_cls = transformer_layer_cls
        self.fsdp_model = None

    def __enter__(self) -> 'FSDP':
        self.fsdp_model = setup_fsdp(
            self.original_model,
            self.config,
            self.device_id,
            self.transformer_layer_cls,
        )
        return self.fsdp_model

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Cleanup if needed
        pass


def get_fsdp_memory_stats() -> Dict[str, float]:
    """Get FSDP memory statistics."""
    if not torch.cuda.is_available():
        return {}

    return {
        "allocated_gb": torch.cuda.memory_allocated() / (1024**3),
        "reserved_gb": torch.cuda.memory_reserved() / (1024**3),
        "max_allocated_gb": torch.cuda.max_memory_allocated() / (1024**3),
        "max_reserved_gb": torch.cuda.max_memory_reserved() / (1024**3),
    }


def reset_peak_memory_stats():
    """Reset peak memory statistics."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
