"""
SPARSA-LM Distributed Communication Utilities
Efficient communication patterns for distributed training
"""

import os
import logging
import pickle
from typing import Any, Optional, List, Dict, Tuple
from dataclasses import dataclass
from contextlib import contextmanager

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)


@dataclass
class DistributedContext:
    """
    Context for distributed training.

    Holds all relevant information about the distributed setup.
    """

    initialized: bool = False
    backend: str = "nccl"
    world_size: int = 1
    rank: int = 0
    local_rank: int = 0
    local_world_size: int = 1
    node_rank: int = 0
    num_nodes: int = 1

    # Device information
    device: Optional[torch.device] = None

    # Process groups
    world_group: Any = None
    local_group: Any = None

    @property
    def is_main_process(self) -> bool:
        return self.rank == 0

    @property
    def is_local_main_process(self) -> bool:
        return self.local_rank == 0


_distributed_context: Optional[DistributedContext] = None


def get_distributed_context() -> DistributedContext:
    """Get the distributed context."""
    global _distributed_context
    if _distributed_context is None:
        _distributed_context = DistributedContext()
    return _distributed_context


def init_distributed(
    backend: str = "nccl",
    init_method: Optional[str] = None,
    timeout_minutes: int = 30,
) -> DistributedContext:
    """
    Initialize distributed training.

    Args:
        backend: Distributed backend (nccl, gloo, mpi)
        init_method: Initialization method (env://, tcp://, file://)
        timeout_minutes: Timeout for initialization

    Returns:
        DistributedContext with setup information
    """
    global _distributed_context

    ctx = get_distributed_context()

    if ctx.initialized:
        return ctx

    # Check if distributed environment is set
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        logger.info("Distributed environment not set. Running in single-process mode.")
        ctx.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            torch.cuda.set_device(0)
        return ctx

    # Get environment variables
    ctx.rank = int(os.environ.get("RANK", 0))
    ctx.world_size = int(os.environ.get("WORLD_SIZE", 1))
    ctx.local_rank = int(os.environ.get("LOCAL_RANK", 0))
    ctx.local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", torch.cuda.device_count() or 1))
    ctx.num_nodes = ctx.world_size // ctx.local_world_size
    ctx.node_rank = ctx.rank // ctx.local_world_size
    ctx.backend = backend

    # Set device
    if torch.cuda.is_available():
        ctx.device = torch.device(f"cuda:{ctx.local_rank}")
        torch.cuda.set_device(ctx.device)
    else:
        ctx.device = torch.device("cpu")
        if backend == "nccl":
            backend = "gloo"
            ctx.backend = backend

    # Initialize process group
    if init_method is None:
        init_method = "env://"

    timeout = torch.distributed.default_pg_timeout
    if timeout_minutes > 0:
        from datetime import timedelta
        timeout = timedelta(minutes=timeout_minutes)

    dist.init_process_group(
        backend=backend,
        init_method=init_method,
        world_size=ctx.world_size,
        rank=ctx.rank,
        timeout=timeout,
    )

    ctx.world_group = dist.group.WORLD
    ctx.initialized = True

    # Create local process group (within node)
    if ctx.local_world_size > 1:
        for node in range(ctx.num_nodes):
            ranks = list(range(
                node * ctx.local_world_size,
                (node + 1) * ctx.local_world_size
            ))
            group = dist.new_group(ranks)
            if ctx.rank in ranks:
                ctx.local_group = group

    # Log initialization
    if ctx.is_main_process:
        logger.info(
            f"Distributed training initialized:\n"
            f"  Backend: {ctx.backend}\n"
            f"  World size: {ctx.world_size}\n"
            f"  Nodes: {ctx.num_nodes}\n"
            f"  GPUs per node: {ctx.local_world_size}"
        )

    # Synchronize
    barrier()

    return ctx


def cleanup_distributed():
    """Cleanup distributed training."""
    global _distributed_context

    if dist.is_initialized():
        dist.destroy_process_group()

    if _distributed_context is not None:
        _distributed_context.initialized = False


def barrier(group: Any = None):
    """
    Synchronize all processes.

    Args:
        group: Process group (default: world)
    """
    if dist.is_initialized():
        dist.barrier(group=group)


def all_reduce_mean(
    tensor: torch.Tensor,
    group: Any = None,
) -> torch.Tensor:
    """
    All-reduce tensor and compute mean across ranks.

    Args:
        tensor: Input tensor
        group: Process group

    Returns:
        Reduced tensor (mean)
    """
    if not dist.is_initialized():
        return tensor

    world_size = dist.get_world_size(group)
    if world_size == 1:
        return tensor

    dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=group)
    tensor = tensor / world_size

    return tensor


def all_reduce_sum(
    tensor: torch.Tensor,
    group: Any = None,
) -> torch.Tensor:
    """
    All-reduce tensor with sum.

    Args:
        tensor: Input tensor
        group: Process group

    Returns:
        Reduced tensor (sum)
    """
    if not dist.is_initialized():
        return tensor

    dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=group)
    return tensor


def all_gather_tensors(
    tensor: torch.Tensor,
    group: Any = None,
) -> List[torch.Tensor]:
    """
    All-gather tensors from all ranks.

    Args:
        tensor: Input tensor
        group: Process group

    Returns:
        List of tensors from all ranks
    """
    if not dist.is_initialized():
        return [tensor]

    world_size = dist.get_world_size(group)
    if world_size == 1:
        return [tensor]

    gathered = [torch.empty_like(tensor) for _ in range(world_size)]
    dist.all_gather(gathered, tensor, group=group)

    return gathered


def all_gather_into_tensor(
    output: torch.Tensor,
    input_: torch.Tensor,
    group: Any = None,
) -> torch.Tensor:
    """
    All-gather into a single tensor.

    Args:
        output: Output tensor (pre-allocated)
        input_: Input tensor
        group: Process group

    Returns:
        Gathered tensor
    """
    if not dist.is_initialized():
        output.copy_(input_)
        return output

    dist.all_gather_into_tensor(output, input_, group=group)
    return output


def reduce_scatter_tensor(
    output: torch.Tensor,
    input_: torch.Tensor,
    group: Any = None,
) -> torch.Tensor:
    """
    Reduce-scatter tensor.

    Args:
        output: Output tensor (pre-allocated)
        input_: Input tensor
        group: Process group

    Returns:
        Reduced-scattered tensor
    """
    if not dist.is_initialized():
        output.copy_(input_)
        return output

    dist.reduce_scatter_tensor(output, input_, group=group)
    return output


def broadcast_object(
    obj: Any,
    src: int = 0,
    group: Any = None,
) -> Any:
    """
    Broadcast a Python object from source rank.

    Args:
        obj: Object to broadcast (only used on source rank)
        src: Source rank
        group: Process group

    Returns:
        Broadcasted object
    """
    if not dist.is_initialized():
        return obj

    rank = dist.get_rank(group)

    if rank == src:
        # Serialize object
        buffer = pickle.dumps(obj)
        size = torch.tensor([len(buffer)], dtype=torch.long, device="cuda")
    else:
        size = torch.tensor([0], dtype=torch.long, device="cuda")

    # Broadcast size
    dist.broadcast(size, src=src, group=group)

    if rank == src:
        buffer_tensor = torch.frombuffer(bytearray(buffer), dtype=torch.uint8).cuda()
    else:
        buffer_tensor = torch.empty(size.item(), dtype=torch.uint8, device="cuda")

    # Broadcast buffer
    dist.broadcast(buffer_tensor, src=src, group=group)

    if rank != src:
        obj = pickle.loads(buffer_tensor.cpu().numpy().tobytes())

    return obj


def broadcast_tensor(
    tensor: torch.Tensor,
    src: int = 0,
    group: Any = None,
) -> torch.Tensor:
    """
    Broadcast tensor from source rank.

    Args:
        tensor: Tensor to broadcast
        src: Source rank
        group: Process group

    Returns:
        Broadcasted tensor
    """
    if not dist.is_initialized():
        return tensor

    dist.broadcast(tensor, src=src, group=group)
    return tensor


def reduce_tensor(
    tensor: torch.Tensor,
    dst: int = 0,
    op: dist.ReduceOp = dist.ReduceOp.SUM,
    group: Any = None,
) -> torch.Tensor:
    """
    Reduce tensor to destination rank.

    Args:
        tensor: Tensor to reduce
        dst: Destination rank
        op: Reduction operation
        group: Process group

    Returns:
        Reduced tensor (valid only on dst rank)
    """
    if not dist.is_initialized():
        return tensor

    dist.reduce(tensor, dst=dst, op=op, group=group)
    return tensor


def scatter_tensor(
    tensor: torch.Tensor,
    scatter_list: Optional[List[torch.Tensor]] = None,
    src: int = 0,
    group: Any = None,
) -> torch.Tensor:
    """
    Scatter tensors from source rank.

    Args:
        tensor: Output tensor
        scatter_list: List of tensors to scatter (only on src)
        src: Source rank
        group: Process group

    Returns:
        Received tensor
    """
    if not dist.is_initialized():
        if scatter_list:
            return scatter_list[0]
        return tensor

    dist.scatter(tensor, scatter_list, src=src, group=group)
    return tensor


def gather_tensors(
    tensor: torch.Tensor,
    dst: int = 0,
    group: Any = None,
) -> Optional[List[torch.Tensor]]:
    """
    Gather tensors to destination rank.

    Args:
        tensor: Tensor to gather
        dst: Destination rank
        group: Process group

    Returns:
        List of tensors on dst rank, None on other ranks
    """
    if not dist.is_initialized():
        return [tensor]

    rank = dist.get_rank(group)
    world_size = dist.get_world_size(group)

    if rank == dst:
        gather_list = [torch.empty_like(tensor) for _ in range(world_size)]
    else:
        gather_list = None

    dist.gather(tensor, gather_list, dst=dst, group=group)
    return gather_list


@contextmanager
def sync_gradients(model: torch.nn.Module, sync: bool = True):
    """
    Context manager for gradient synchronization control.

    Args:
        model: DDP model
        sync: Whether to synchronize gradients

    Yields:
        Model context
    """
    if not sync:
        # Disable gradient sync for gradient accumulation
        if hasattr(model, 'no_sync'):
            with model.no_sync():
                yield
        else:
            yield
    else:
        yield


class GradientAccumulator:
    """
    Utility for gradient accumulation with proper distributed handling.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        accumulation_steps: int = 1,
    ):
        self.model = model
        self.accumulation_steps = accumulation_steps
        self.step_count = 0

    def should_sync(self) -> bool:
        """Check if gradients should be synchronized."""
        return (self.step_count + 1) % self.accumulation_steps == 0

    def should_step(self) -> bool:
        """Check if optimizer should step."""
        return self.should_sync()

    @contextmanager
    def accumulate(self):
        """Context manager for gradient accumulation."""
        with sync_gradients(self.model, sync=self.should_sync()):
            yield

        self.step_count += 1

    def reset(self):
        """Reset step count."""
        self.step_count = 0


class AllReduceCoalescing:
    """
    Coalesce multiple all-reduce operations for efficiency.
    """

    def __init__(
        self,
        bucket_size_mb: float = 25.0,
        group: Any = None,
    ):
        self.bucket_size = int(bucket_size_mb * 1024 * 1024 / 4)  # In floats
        self.group = group
        self.buffer = []
        self.buffer_size = 0

    def add(self, tensor: torch.Tensor):
        """Add tensor to buffer."""
        self.buffer.append(tensor)
        self.buffer_size += tensor.numel()

        if self.buffer_size >= self.bucket_size:
            self.flush()

    def flush(self):
        """Flush buffer and perform all-reduce."""
        if not self.buffer:
            return

        if not dist.is_initialized():
            self.buffer = []
            self.buffer_size = 0
            return

        # Flatten tensors
        flat = torch.cat([t.view(-1) for t in self.buffer])

        # All-reduce
        dist.all_reduce(flat, group=self.group)

        # Unflatten back
        offset = 0
        for tensor in self.buffer:
            numel = tensor.numel()
            tensor.copy_(flat[offset:offset + numel].view_as(tensor))
            offset += numel

        self.buffer = []
        self.buffer_size = 0


def get_data_parallel_rank() -> int:
    """Get data parallel rank."""
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


def get_data_parallel_world_size() -> int:
    """Get data parallel world size."""
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


def print_rank_0(message: str):
    """Print message only on rank 0."""
    if not dist.is_initialized() or dist.get_rank() == 0:
        print(message)


def log_rank_0(message: str, level: int = logging.INFO):
    """Log message only on rank 0."""
    if not dist.is_initialized() or dist.get_rank() == 0:
        logger.log(level, message)
