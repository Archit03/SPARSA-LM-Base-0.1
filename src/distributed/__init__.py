"""
SPARSA-LM Distributed Training Infrastructure
FSDP, Tensor Parallelism, and Communication Utilities
"""

from .fsdp import (
    FSDPConfig,
    setup_fsdp,
    get_fsdp_policy,
    save_fsdp_checkpoint,
    load_fsdp_checkpoint,
)

from .tensor_parallel import (
    TensorParallelConfig,
    ColumnParallelLinear,
    RowParallelLinear,
    ParallelEmbedding,
)

from .comm import (
    DistributedContext,
    init_distributed,
    cleanup_distributed,
    all_reduce_mean,
    all_gather_tensors,
    broadcast_object,
    barrier,
)

from .utils import (
    get_rank,
    get_world_size,
    get_local_rank,
    is_main_process,
    seed_all,
    get_device,
)

__all__ = [
    # FSDP
    "FSDPConfig",
    "setup_fsdp",
    "get_fsdp_policy",
    "save_fsdp_checkpoint",
    "load_fsdp_checkpoint",
    # Tensor Parallel
    "TensorParallelConfig",
    "ColumnParallelLinear",
    "RowParallelLinear",
    "ParallelEmbedding",
    # Communication
    "DistributedContext",
    "init_distributed",
    "cleanup_distributed",
    "all_reduce_mean",
    "all_gather_tensors",
    "broadcast_object",
    "barrier",
    # Utils
    "get_rank",
    "get_world_size",
    "get_local_rank",
    "is_main_process",
    "seed_all",
    "get_device",
]
