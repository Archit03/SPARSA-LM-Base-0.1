"""
SPARSA-LM Distributed Training Utilities
Helper functions for distributed training setup and management
"""

import os
import random
import logging
from typing import Optional, List, Any, Dict
from functools import lru_cache

import numpy as np
import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)


def get_rank() -> int:
    """Get global rank of current process."""
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


def get_world_size() -> int:
    """Get total number of processes."""
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


def get_local_rank() -> int:
    """Get local rank (within node) of current process."""
    return int(os.environ.get("LOCAL_RANK", 0))


def get_local_world_size() -> int:
    """Get local world size (processes per node)."""
    return int(os.environ.get("LOCAL_WORLD_SIZE", torch.cuda.device_count() or 1))


def is_main_process() -> bool:
    """Check if this is the main process (rank 0)."""
    return get_rank() == 0


def is_local_main_process() -> bool:
    """Check if this is the local main process (local_rank 0)."""
    return get_local_rank() == 0


@lru_cache(maxsize=1)
def get_device() -> torch.device:
    """
    Get the device for this process.

    Returns:
        torch.device for this rank
    """
    if torch.cuda.is_available():
        local_rank = get_local_rank()
        return torch.device(f"cuda:{local_rank}")
    return torch.device("cpu")


def seed_all(seed: int, deterministic: bool = False):
    """
    Set random seed for reproducibility.

    Args:
        seed: Random seed
        deterministic: If True, use deterministic algorithms (slower)
    """
    # Offset seed by rank for different random states per process
    rank = get_rank()
    seed = seed + rank

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        if hasattr(torch, 'use_deterministic_algorithms'):
            torch.use_deterministic_algorithms(True)
    else:
        torch.backends.cudnn.benchmark = True


def setup_for_distributed(is_master: bool):
    """
    Setup logging for distributed training.

    Disables logging on non-master processes.

    Args:
        is_master: Whether this is the master process
    """
    import builtins as __builtin__

    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def get_num_gpus() -> int:
    """Get number of available GPUs."""
    return torch.cuda.device_count()


def get_gpu_memory_info(device_id: Optional[int] = None) -> Dict[str, float]:
    """
    Get GPU memory information.

    Args:
        device_id: GPU device ID (default: current device)

    Returns:
        Dictionary with memory information in GB
    """
    if not torch.cuda.is_available():
        return {}

    if device_id is None:
        device_id = get_local_rank()

    with torch.cuda.device(device_id):
        return {
            "allocated_gb": torch.cuda.memory_allocated() / (1024**3),
            "reserved_gb": torch.cuda.memory_reserved() / (1024**3),
            "max_allocated_gb": torch.cuda.max_memory_allocated() / (1024**3),
            "max_reserved_gb": torch.cuda.max_memory_reserved() / (1024**3),
            "total_gb": torch.cuda.get_device_properties(device_id).total_memory / (1024**3),
        }


def reset_memory_stats(device_id: Optional[int] = None):
    """Reset peak memory statistics."""
    if torch.cuda.is_available():
        if device_id is None:
            device_id = get_local_rank()
        torch.cuda.reset_peak_memory_stats(device_id)


def empty_cache():
    """Empty CUDA cache."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def synchronize():
    """Synchronize CUDA operations."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def get_cuda_rng_state() -> Optional[torch.Tensor]:
    """Get CUDA RNG state for checkpointing."""
    if torch.cuda.is_available():
        return torch.cuda.get_rng_state()
    return None


def set_cuda_rng_state(state: torch.Tensor):
    """Set CUDA RNG state from checkpoint."""
    if torch.cuda.is_available() and state is not None:
        torch.cuda.set_rng_state(state)


def compute_num_params(model: torch.nn.Module) -> Dict[str, int]:
    """
    Compute number of parameters in model.

    Args:
        model: PyTorch model

    Returns:
        Dictionary with parameter counts
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return {
        "total": total,
        "trainable": trainable,
        "non_trainable": total - trainable,
        "total_millions": total / 1e6,
        "trainable_millions": trainable / 1e6,
    }


def get_model_memory_footprint(model: torch.nn.Module, batch_size: int = 1) -> Dict[str, float]:
    """
    Estimate model memory footprint.

    Args:
        model: PyTorch model
        batch_size: Batch size for activation estimation

    Returns:
        Dictionary with memory estimates in GB
    """
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())

    # Estimate gradient size (same as params for most optimizers)
    grad_size = param_size

    # Rough estimate: optimizer states (Adam has 2x param size)
    optimizer_size = 2 * param_size

    return {
        "parameters_gb": param_size / (1024**3),
        "buffers_gb": buffer_size / (1024**3),
        "gradients_gb": grad_size / (1024**3),
        "optimizer_gb": optimizer_size / (1024**3),
        "total_gb": (param_size + buffer_size + grad_size + optimizer_size) / (1024**3),
    }


def shard_indices(
    total_size: int,
    num_shards: int,
    shard_id: int,
) -> List[int]:
    """
    Get indices for a particular shard.

    Args:
        total_size: Total number of items
        num_shards: Number of shards
        shard_id: Current shard ID

    Returns:
        List of indices for this shard
    """
    indices_per_shard = total_size // num_shards
    remainder = total_size % num_shards

    # Distribute remainder among first `remainder` shards
    if shard_id < remainder:
        start = shard_id * (indices_per_shard + 1)
        end = start + indices_per_shard + 1
    else:
        start = shard_id * indices_per_shard + remainder
        end = start + indices_per_shard

    return list(range(start, end))


def get_effective_batch_size(
    per_device_batch_size: int,
    gradient_accumulation_steps: int = 1,
    world_size: Optional[int] = None,
) -> int:
    """
    Calculate effective batch size.

    Args:
        per_device_batch_size: Batch size per device
        gradient_accumulation_steps: Number of accumulation steps
        world_size: Number of devices (default: auto-detect)

    Returns:
        Effective batch size
    """
    if world_size is None:
        world_size = get_world_size()

    return per_device_batch_size * gradient_accumulation_steps * world_size


def get_learning_rate_for_batch_size(
    base_lr: float,
    base_batch_size: int,
    actual_batch_size: int,
    scaling: str = "linear",
) -> float:
    """
    Scale learning rate based on batch size.

    Args:
        base_lr: Base learning rate
        base_batch_size: Base batch size
        actual_batch_size: Actual batch size
        scaling: Scaling method ("linear", "sqrt", or "none")

    Returns:
        Scaled learning rate
    """
    if scaling == "linear":
        return base_lr * (actual_batch_size / base_batch_size)
    elif scaling == "sqrt":
        return base_lr * (actual_batch_size / base_batch_size) ** 0.5
    else:  # none
        return base_lr


class DistributedTimer:
    """
    Timer for distributed training with proper synchronization.
    """

    def __init__(self, sync: bool = True):
        self.sync = sync
        self.start_time = None
        self.elapsed = 0.0

    def start(self):
        """Start the timer."""
        if self.sync:
            synchronize()
        self.start_time = torch.cuda.Event(enable_timing=True)
        self.end_time = torch.cuda.Event(enable_timing=True)
        self.start_time.record()

    def stop(self) -> float:
        """
        Stop the timer and return elapsed time.

        Returns:
            Elapsed time in seconds
        """
        self.end_time.record()
        if self.sync:
            synchronize()
        self.elapsed = self.start_time.elapsed_time(self.end_time) / 1000.0
        return self.elapsed

    def reset(self):
        """Reset the timer."""
        self.start_time = None
        self.elapsed = 0.0


class ThroughputTracker:
    """
    Track training throughput (tokens/sec, samples/sec).
    """

    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.tokens = []
        self.times = []
        self.total_tokens = 0
        self.total_time = 0.0

    def update(self, num_tokens: int, elapsed_time: float):
        """
        Update throughput tracker.

        Args:
            num_tokens: Number of tokens processed
            elapsed_time: Time elapsed in seconds
        """
        self.tokens.append(num_tokens)
        self.times.append(elapsed_time)
        self.total_tokens += num_tokens
        self.total_time += elapsed_time

        # Keep window
        while len(self.tokens) > self.window_size:
            self.tokens.pop(0)
            self.times.pop(0)

    @property
    def tokens_per_second(self) -> float:
        """Get tokens per second (windowed)."""
        if not self.times or sum(self.times) == 0:
            return 0.0
        return sum(self.tokens) / sum(self.times)

    @property
    def tokens_per_second_global(self) -> float:
        """Get tokens per second (global)."""
        if self.total_time == 0:
            return 0.0
        return self.total_tokens / self.total_time

    def get_stats(self) -> Dict[str, float]:
        """Get throughput statistics."""
        return {
            "tokens_per_second": self.tokens_per_second,
            "tokens_per_second_global": self.tokens_per_second_global,
            "total_tokens": self.total_tokens,
            "total_time_hours": self.total_time / 3600,
        }


def wait_for_everyone():
    """Wait for all processes to reach this point."""
    if dist.is_initialized():
        dist.barrier()


def only_on_main_process(func):
    """Decorator to run function only on main process."""
    def wrapper(*args, **kwargs):
        if is_main_process():
            return func(*args, **kwargs)
        return None
    return wrapper


class DistributedState:
    """
    Singleton class to hold distributed training state.
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self.rank = get_rank()
        self.world_size = get_world_size()
        self.local_rank = get_local_rank()
        self.local_world_size = get_local_world_size()
        self.device = get_device()
        self.is_main_process = is_main_process()
        self.is_local_main_process = is_local_main_process()

        self._initialized = True

    def __repr__(self):
        return (
            f"DistributedState(\n"
            f"  rank={self.rank},\n"
            f"  world_size={self.world_size},\n"
            f"  local_rank={self.local_rank},\n"
            f"  local_world_size={self.local_world_size},\n"
            f"  device={self.device},\n"
            f"  is_main_process={self.is_main_process}\n"
            f")"
        )
