"""
SPARSA-LM Tensor Parallelism Infrastructure
Column and Row Parallel Linear layers for model parallelism
"""

import os
import math
import logging
from dataclasses import dataclass
from typing import Optional, Tuple, List, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

logger = logging.getLogger(__name__)


@dataclass
class TensorParallelConfig:
    """
    Configuration for Tensor Parallelism.

    Tensor parallelism splits individual layers across GPUs to enable
    training of very large models that don't fit in a single GPU.
    """

    # World configuration
    tp_size: int = 1  # Tensor parallel world size
    dp_size: int = 1  # Data parallel world size
    pp_size: int = 1  # Pipeline parallel world size (not implemented)

    # Communication
    sequence_parallel: bool = False  # Enable sequence parallelism
    async_tp: bool = False  # Async tensor parallel communication

    # Initialization
    use_cpu_initialization: bool = True
    init_method: str = "xavier"  # xavier, normal, scaled_normal

    # Precision
    reduce_in_fp32: bool = True

    @property
    def world_size(self) -> int:
        return self.tp_size * self.dp_size * self.pp_size


class TensorParallelGroup:
    """
    Manages tensor parallel process groups.
    """

    _instance = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self.tp_group = None
        self.dp_group = None
        self.tp_size = 1
        self.dp_size = 1
        self.tp_rank = 0
        self.dp_rank = 0

        self._initialized = True

    def initialize(self, config: TensorParallelConfig):
        """
        Initialize tensor parallel groups.

        Args:
            config: Tensor parallel configuration
        """
        if not dist.is_initialized():
            logger.warning("Distributed not initialized. Using single process.")
            return

        world_size = dist.get_world_size()
        rank = dist.get_rank()

        self.tp_size = min(config.tp_size, world_size)
        self.dp_size = world_size // self.tp_size

        # Create tensor parallel groups
        # Each TP group contains tp_size consecutive ranks
        for i in range(self.dp_size):
            tp_ranks = list(range(i * self.tp_size, (i + 1) * self.tp_size))
            group = dist.new_group(tp_ranks)
            if rank in tp_ranks:
                self.tp_group = group
                self.tp_rank = tp_ranks.index(rank)

        # Create data parallel groups
        # Each DP group contains one rank from each TP group
        for i in range(self.tp_size):
            dp_ranks = list(range(i, world_size, self.tp_size))
            group = dist.new_group(dp_ranks)
            if rank in dp_ranks:
                self.dp_group = group
                self.dp_rank = dp_ranks.index(rank)

        logger.info(
            f"Initialized TP groups: tp_size={self.tp_size}, dp_size={self.dp_size}, "
            f"tp_rank={self.tp_rank}, dp_rank={self.dp_rank}"
        )

    @property
    def is_first_tp_rank(self) -> bool:
        return self.tp_rank == 0

    @property
    def is_last_tp_rank(self) -> bool:
        return self.tp_rank == self.tp_size - 1


def get_tp_group() -> TensorParallelGroup:
    """Get the tensor parallel group singleton."""
    return TensorParallelGroup()


def _reduce_scatter_along_first_dim(
    input_: torch.Tensor,
    group: Any,
) -> torch.Tensor:
    """Reduce-scatter tensor along first dimension."""
    world_size = dist.get_world_size(group)

    if world_size == 1:
        return input_

    dim_size = input_.size(0)
    assert dim_size % world_size == 0

    output = torch.empty(
        dim_size // world_size, *input_.shape[1:],
        dtype=input_.dtype, device=input_.device
    )

    dist.reduce_scatter_tensor(output, input_, group=group)

    return output


def _all_gather_along_first_dim(
    input_: torch.Tensor,
    group: Any,
) -> torch.Tensor:
    """All-gather tensor along first dimension."""
    world_size = dist.get_world_size(group)

    if world_size == 1:
        return input_

    output = torch.empty(
        input_.size(0) * world_size, *input_.shape[1:],
        dtype=input_.dtype, device=input_.device
    )

    dist.all_gather_into_tensor(output, input_, group=group)

    return output


class _CopyToTensorParallelRegion(torch.autograd.Function):
    """Copy input to tensor parallel region (identity forward, all-reduce backward)."""

    @staticmethod
    def forward(ctx, input_: torch.Tensor, group: Any) -> torch.Tensor:
        ctx.group = group
        return input_

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None]:
        group = ctx.group
        if dist.get_world_size(group) > 1:
            dist.all_reduce(grad_output, group=group)
        return grad_output, None


class _ReduceFromTensorParallelRegion(torch.autograd.Function):
    """Reduce input from tensor parallel region (all-reduce forward, identity backward)."""

    @staticmethod
    def forward(ctx, input_: torch.Tensor, group: Any) -> torch.Tensor:
        if dist.get_world_size(group) > 1:
            dist.all_reduce(input_, group=group)
        return input_

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None]:
        return grad_output, None


class _ScatterToTensorParallelRegion(torch.autograd.Function):
    """Scatter input to tensor parallel region."""

    @staticmethod
    def forward(ctx, input_: torch.Tensor, group: Any) -> torch.Tensor:
        ctx.group = group
        world_size = dist.get_world_size(group)

        if world_size == 1:
            return input_

        rank = dist.get_rank(group)
        dim_size = input_.size(-1)
        assert dim_size % world_size == 0

        chunk_size = dim_size // world_size
        return input_[..., rank * chunk_size : (rank + 1) * chunk_size].contiguous()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None]:
        group = ctx.group
        return _all_gather_along_first_dim(grad_output.transpose(0, -1), group).transpose(0, -1), None


class _GatherFromTensorParallelRegion(torch.autograd.Function):
    """Gather input from tensor parallel region."""

    @staticmethod
    def forward(ctx, input_: torch.Tensor, group: Any) -> torch.Tensor:
        ctx.group = group
        world_size = dist.get_world_size(group)

        if world_size == 1:
            return input_

        # All-gather along last dimension
        gathered = [torch.empty_like(input_) for _ in range(world_size)]
        dist.all_gather(gathered, input_, group=group)
        return torch.cat(gathered, dim=-1)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None]:
        group = ctx.group
        world_size = dist.get_world_size(group)
        rank = dist.get_rank(group)

        if world_size == 1:
            return grad_output, None

        dim_size = grad_output.size(-1)
        chunk_size = dim_size // world_size

        return grad_output[..., rank * chunk_size : (rank + 1) * chunk_size].contiguous(), None


def copy_to_tensor_parallel_region(input_: torch.Tensor) -> torch.Tensor:
    """Copy to TP region."""
    tp_group = get_tp_group()
    if tp_group.tp_group is None or tp_group.tp_size == 1:
        return input_
    return _CopyToTensorParallelRegion.apply(input_, tp_group.tp_group)


def reduce_from_tensor_parallel_region(input_: torch.Tensor) -> torch.Tensor:
    """Reduce from TP region."""
    tp_group = get_tp_group()
    if tp_group.tp_group is None or tp_group.tp_size == 1:
        return input_
    return _ReduceFromTensorParallelRegion.apply(input_, tp_group.tp_group)


def scatter_to_tensor_parallel_region(input_: torch.Tensor) -> torch.Tensor:
    """Scatter to TP region."""
    tp_group = get_tp_group()
    if tp_group.tp_group is None or tp_group.tp_size == 1:
        return input_
    return _ScatterToTensorParallelRegion.apply(input_, tp_group.tp_group)


def gather_from_tensor_parallel_region(input_: torch.Tensor) -> torch.Tensor:
    """Gather from TP region."""
    tp_group = get_tp_group()
    if tp_group.tp_group is None or tp_group.tp_size == 1:
        return input_
    return _GatherFromTensorParallelRegion.apply(input_, tp_group.tp_group)


class ColumnParallelLinear(nn.Module):
    """
    Linear layer with column parallelism.

    The linear layer is defined as Y = XA + b.
    A is parallelized along its second dimension (columns).

    Arguments:
        in_features: First dimension of matrix A
        out_features: Second dimension of matrix A
        bias: If true, add bias
        gather_output: If true, gather output from all ranks
        init_method: Initialization method for weights
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        gather_output: bool = True,
        init_method: str = "xavier",
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.gather_output = gather_output
        self.init_method = init_method

        tp_group = get_tp_group()
        self.tp_size = tp_group.tp_size
        self.tp_rank = tp_group.tp_rank

        # Partition output features
        assert out_features % self.tp_size == 0
        self.out_features_per_partition = out_features // self.tp_size

        # Create weight and bias
        self.weight = nn.Parameter(torch.empty(
            self.out_features_per_partition, in_features,
            device=device, dtype=dtype
        ))

        if bias:
            self.bias = nn.Parameter(torch.empty(
                self.out_features_per_partition,
                device=device, dtype=dtype
            ))
        else:
            self.register_parameter('bias', None)

        # Initialize
        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        if self.init_method == "xavier":
            nn.init.xavier_uniform_(self.weight)
        elif self.init_method == "normal":
            nn.init.normal_(self.weight, std=0.02)
        elif self.init_method == "scaled_normal":
            std = 1.0 / math.sqrt(self.in_features)
            nn.init.normal_(self.weight, std=std)

        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        # Copy input to tensor parallel region
        input_parallel = copy_to_tensor_parallel_region(input_)

        # Matrix multiply
        output_parallel = F.linear(input_parallel, self.weight, self.bias)

        # Gather output if needed
        if self.gather_output:
            output = gather_from_tensor_parallel_region(output_parallel)
        else:
            output = output_parallel

        return output


class RowParallelLinear(nn.Module):
    """
    Linear layer with row parallelism.

    The linear layer is defined as Y = XA + b.
    A is parallelized along its first dimension (rows).

    Arguments:
        in_features: First dimension of matrix A
        out_features: Second dimension of matrix A
        bias: If true, add bias. Only rank 0 will have bias.
        input_is_parallel: If true, input is already parallelized
        init_method: Initialization method for weights
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        input_is_parallel: bool = False,
        init_method: str = "xavier",
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.input_is_parallel = input_is_parallel
        self.init_method = init_method

        tp_group = get_tp_group()
        self.tp_size = tp_group.tp_size
        self.tp_rank = tp_group.tp_rank

        # Partition input features
        assert in_features % self.tp_size == 0
        self.in_features_per_partition = in_features // self.tp_size

        # Create weight
        self.weight = nn.Parameter(torch.empty(
            out_features, self.in_features_per_partition,
            device=device, dtype=dtype
        ))

        # Only first rank has bias
        if bias and self.tp_rank == 0:
            self.bias = nn.Parameter(torch.empty(
                out_features,
                device=device, dtype=dtype
            ))
        else:
            self.register_parameter('bias', None)

        # Initialize
        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        if self.init_method == "xavier":
            nn.init.xavier_uniform_(self.weight)
        elif self.init_method == "normal":
            nn.init.normal_(self.weight, std=0.02)
        elif self.init_method == "scaled_normal":
            std = 1.0 / math.sqrt(self.in_features)
            nn.init.normal_(self.weight, std=std)

        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        # Scatter input if not already parallel
        if self.input_is_parallel:
            input_parallel = input_
        else:
            input_parallel = scatter_to_tensor_parallel_region(input_)

        # Matrix multiply (no bias - bias added after reduce)
        output_parallel = F.linear(input_parallel, self.weight)

        # Reduce across tensor parallel ranks
        output = reduce_from_tensor_parallel_region(output_parallel)

        # Add bias
        if self.bias is not None:
            output = output + self.bias

        return output


class ParallelEmbedding(nn.Module):
    """
    Embedding layer parallelized along vocabulary dimension.

    Arguments:
        num_embeddings: Size of vocabulary
        embedding_dim: Dimension of embeddings
        padding_idx: Padding index
        init_method: Initialization method
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: Optional[int] = None,
        init_method: str = "normal",
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.init_method = init_method

        tp_group = get_tp_group()
        self.tp_size = tp_group.tp_size
        self.tp_rank = tp_group.tp_rank

        # Partition vocabulary
        assert num_embeddings % self.tp_size == 0
        self.num_embeddings_per_partition = num_embeddings // self.tp_size
        self.vocab_start_index = self.tp_rank * self.num_embeddings_per_partition
        self.vocab_end_index = (self.tp_rank + 1) * self.num_embeddings_per_partition

        # Create embedding table
        self.weight = nn.Parameter(torch.empty(
            self.num_embeddings_per_partition, embedding_dim,
            device=device, dtype=dtype
        ))

        # Initialize
        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        if self.init_method == "normal":
            nn.init.normal_(self.weight, std=0.02)
        elif self.init_method == "xavier":
            nn.init.xavier_uniform_(self.weight)

        # Handle padding
        if self.padding_idx is not None:
            if self.vocab_start_index <= self.padding_idx < self.vocab_end_index:
                local_idx = self.padding_idx - self.vocab_start_index
                with torch.no_grad():
                    self.weight[local_idx].fill_(0)

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        # Create mask for tokens in this partition
        mask = (input_ >= self.vocab_start_index) & (input_ < self.vocab_end_index)

        # Shift indices to local range
        local_input = input_ - self.vocab_start_index
        local_input = local_input.clamp(min=0, max=self.num_embeddings_per_partition - 1)

        # Lookup embeddings
        output_parallel = F.embedding(local_input, self.weight)

        # Mask out tokens not in this partition
        output_parallel = output_parallel * mask.unsqueeze(-1).float()

        # All-reduce to combine embeddings
        output = reduce_from_tensor_parallel_region(output_parallel)

        return output


class ParallelVocabParallelEmbedding(nn.Module):
    """
    Vocabulary parallel embedding.

    Different from ParallelEmbedding in that it uses scatter/gather
    pattern instead of all-reduce for efficiency.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: Optional[int] = None,
        init_method: str = "normal",
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx

        tp_group = get_tp_group()
        self.tp_size = tp_group.tp_size
        self.tp_rank = tp_group.tp_rank

        # Each rank handles a subset of vocabulary
        self.vocab_start_index = self.tp_rank * (num_embeddings // self.tp_size)
        self.vocab_end_index = (self.tp_rank + 1) * (num_embeddings // self.tp_size)
        self.num_embeddings_per_partition = self.vocab_end_index - self.vocab_start_index

        self.weight = nn.Parameter(torch.empty(
            self.num_embeddings_per_partition, embedding_dim,
            device=device, dtype=dtype
        ))

        # Initialize
        nn.init.normal_(self.weight, std=0.02)

        if self.padding_idx is not None:
            if self.vocab_start_index <= self.padding_idx < self.vocab_end_index:
                local_idx = self.padding_idx - self.vocab_start_index
                with torch.no_grad():
                    self.weight[local_idx].fill_(0)

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        """Forward pass with optimized scatter-gather."""
        tp_group = get_tp_group()

        if self.tp_size == 1:
            return F.embedding(input_, self.weight, self.padding_idx)

        # Determine which tokens are in this partition
        mask = (input_ >= self.vocab_start_index) & (input_ < self.vocab_end_index)
        local_input = torch.where(
            mask,
            input_ - self.vocab_start_index,
            torch.zeros_like(input_)
        )

        # Lookup embeddings
        output_parallel = F.embedding(local_input, self.weight)

        # Zero out embeddings for tokens not in this partition
        output_parallel = output_parallel * mask.unsqueeze(-1).float()

        # All-reduce to combine
        if tp_group.tp_group is not None:
            dist.all_reduce(output_parallel, group=tp_group.tp_group)

        return output_parallel
