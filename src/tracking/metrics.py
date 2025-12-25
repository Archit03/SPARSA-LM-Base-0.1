"""
SPARSA-LM Metrics Collection and Computation
Training and evaluation metrics with proper aggregation
"""

import math
import time
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union
from collections import defaultdict
from contextlib import contextmanager

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)


@dataclass
class TrainingMetrics:
    """
    Container for training metrics.

    Tracks:
    - Loss values (total, components)
    - Learning rate schedule
    - Gradient statistics
    - Throughput metrics
    """

    # Loss metrics
    loss: float = 0.0
    loss_lm: float = 0.0
    loss_aux: float = 0.0

    # Gradient metrics
    grad_norm: float = 0.0
    grad_norm_before_clip: float = 0.0

    # Learning rate
    learning_rate: float = 0.0

    # Throughput
    tokens_per_second: float = 0.0
    samples_per_second: float = 0.0
    gpu_memory_gb: float = 0.0

    # Training progress
    step: int = 0
    epoch: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        """Convert to logging dictionary."""
        return {
            "train/loss": self.loss,
            "train/loss_lm": self.loss_lm,
            "train/loss_aux": self.loss_aux,
            "train/grad_norm": self.grad_norm,
            "train/grad_norm_before_clip": self.grad_norm_before_clip,
            "train/learning_rate": self.learning_rate,
            "train/tokens_per_second": self.tokens_per_second,
            "train/samples_per_second": self.samples_per_second,
            "train/gpu_memory_gb": self.gpu_memory_gb,
            "train/step": self.step,
            "train/epoch": self.epoch,
        }


@dataclass
class EvaluationMetrics:
    """
    Container for evaluation metrics.

    Tracks:
    - Perplexity
    - Accuracy (for classification)
    - Benchmark-specific metrics
    """

    # Core metrics
    loss: float = 0.0
    perplexity: float = 0.0

    # Classification metrics
    accuracy: float = 0.0
    f1_score: float = 0.0

    # Generation metrics
    bleu_score: float = 0.0
    rouge_l: float = 0.0

    # Benchmark-specific
    benchmark_name: str = ""
    subset_metrics: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, float]:
        """Convert to logging dictionary."""
        metrics = {
            "eval/loss": self.loss,
            "eval/perplexity": self.perplexity,
            "eval/accuracy": self.accuracy,
            "eval/f1_score": self.f1_score,
            "eval/bleu_score": self.bleu_score,
            "eval/rouge_l": self.rouge_l,
        }

        # Add subset metrics
        for key, value in self.subset_metrics.items():
            metrics[f"eval/{self.benchmark_name}/{key}"] = value

        return metrics


class MetricLogger:
    """
    Metric logging utility with aggregation and distributed support.

    Features:
    - Moving average smoothing
    - Distributed reduction
    - Throughput calculation
    - GPU memory tracking
    """

    def __init__(
        self,
        window_size: int = 100,
        distributed: bool = False,
    ):
        """
        Initialize metric logger.

        Args:
            window_size: Window size for moving averages
            distributed: Whether to use distributed reduction
        """
        self.window_size = window_size
        self.distributed = distributed and dist.is_initialized()

        # Metric storage
        self._values: Dict[str, List[float]] = defaultdict(list)
        self._counts: Dict[str, int] = defaultdict(int)
        self._sums: Dict[str, float] = defaultdict(float)

        # Timing
        self._start_time = time.time()
        self._step_start_time = time.time()
        self._total_tokens = 0
        self._total_samples = 0

    def update(
        self,
        name: str,
        value: float,
        n: int = 1,
        reduce: bool = True,
    ):
        """
        Update a metric.

        Args:
            name: Metric name
            value: Metric value
            n: Number of samples for weighted average
            reduce: Whether to perform distributed reduction
        """
        if reduce and self.distributed:
            value = self._reduce_value(value)

        self._values[name].append(value)
        if len(self._values[name]) > self.window_size:
            self._values[name].pop(0)

        self._counts[name] += n
        self._sums[name] += value * n

    def _reduce_value(self, value: float) -> float:
        """Reduce value across distributed workers."""
        if not self.distributed:
            return value

        tensor = torch.tensor([value], device='cuda')
        dist.all_reduce(tensor, op=dist.ReduceOp.AVG)
        return tensor.item()

    def get(self, name: str, mode: str = "window") -> float:
        """
        Get metric value.

        Args:
            name: Metric name
            mode: "window" for moving average, "global" for global average

        Returns:
            Metric value
        """
        if mode == "window":
            values = self._values.get(name, [0.0])
            return sum(values) / len(values) if values else 0.0
        else:
            count = self._counts.get(name, 0)
            return self._sums.get(name, 0.0) / max(count, 1)

    def get_all(self, mode: str = "window") -> Dict[str, float]:
        """Get all metrics."""
        return {name: self.get(name, mode) for name in self._values.keys()}

    def update_tokens(self, num_tokens: int):
        """Update token count for throughput calculation."""
        self._total_tokens += num_tokens

    def update_samples(self, num_samples: int):
        """Update sample count for throughput calculation."""
        self._total_samples += num_samples

    def start_step(self):
        """Mark the start of a training step."""
        self._step_start_time = time.time()

    def end_step(self, num_tokens: int = 0, num_samples: int = 0):
        """
        Mark the end of a training step.

        Args:
            num_tokens: Tokens processed in this step
            num_samples: Samples processed in this step
        """
        step_time = time.time() - self._step_start_time
        self._total_tokens += num_tokens
        self._total_samples += num_samples

        if step_time > 0:
            self.update("throughput/tokens_per_second", num_tokens / step_time, reduce=False)
            self.update("throughput/samples_per_second", num_samples / step_time, reduce=False)
            self.update("throughput/step_time_ms", step_time * 1000, reduce=False)

    @property
    def elapsed_time(self) -> float:
        """Elapsed time in seconds."""
        return time.time() - self._start_time

    @property
    def tokens_per_second(self) -> float:
        """Overall tokens per second."""
        return self._total_tokens / max(self.elapsed_time, 1e-6)

    @property
    def samples_per_second(self) -> float:
        """Overall samples per second."""
        return self._total_samples / max(self.elapsed_time, 1e-6)

    def get_gpu_memory(self) -> float:
        """Get GPU memory usage in GB."""
        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / (1024 ** 3)
        return 0.0

    def reset(self):
        """Reset all metrics."""
        self._values.clear()
        self._counts.clear()
        self._sums.clear()
        self._start_time = time.time()
        self._total_tokens = 0
        self._total_samples = 0

    @contextmanager
    def timer(self, name: str):
        """Context manager for timing operations."""
        start = time.time()
        yield
        elapsed = time.time() - start
        self.update(f"time/{name}_ms", elapsed * 1000, reduce=False)


class GradientMonitor:
    """
    Monitor gradient statistics during training.

    Tracks:
    - Gradient norms (before/after clipping)
    - Gradient statistics per layer
    - Gradient flow detection
    """

    def __init__(
        self,
        model: torch.nn.Module,
        log_per_layer: bool = False,
    ):
        """
        Initialize gradient monitor.

        Args:
            model: Model to monitor
            log_per_layer: Whether to log per-layer statistics
        """
        self.model = model
        self.log_per_layer = log_per_layer
        self._grad_norms: Dict[str, float] = {}

    def compute_grad_norm(
        self,
        norm_type: float = 2.0,
    ) -> float:
        """
        Compute gradient norm.

        Args:
            norm_type: Type of norm (1, 2, or inf)

        Returns:
            Total gradient norm
        """
        parameters = [p for p in self.model.parameters() if p.grad is not None]

        if len(parameters) == 0:
            return 0.0

        device = parameters[0].grad.device

        if norm_type == float('inf'):
            total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
        else:
            total_norm = torch.norm(
                torch.stack([
                    torch.norm(p.grad.detach(), norm_type).to(device)
                    for p in parameters
                ]),
                norm_type
            )

        return total_norm.item()

    def get_layer_grad_norms(self) -> Dict[str, float]:
        """Get gradient norms per layer."""
        grad_norms = {}

        for name, param in self.model.named_parameters():
            if param.grad is not None:
                grad_norms[name] = param.grad.detach().norm().item()

        return grad_norms

    def check_gradient_flow(self) -> Dict[str, Any]:
        """
        Check for gradient flow issues.

        Returns:
            Dictionary with gradient flow diagnostics
        """
        total_params = 0
        zero_grads = 0
        nan_grads = 0
        inf_grads = 0
        very_small = 0
        very_large = 0

        for name, param in self.model.named_parameters():
            if param.grad is not None:
                total_params += 1
                grad = param.grad.detach()

                if torch.all(grad == 0):
                    zero_grads += 1
                if torch.any(torch.isnan(grad)):
                    nan_grads += 1
                if torch.any(torch.isinf(grad)):
                    inf_grads += 1
                if grad.abs().max() < 1e-7:
                    very_small += 1
                if grad.abs().max() > 1e3:
                    very_large += 1

        return {
            "total_params_with_grad": total_params,
            "zero_grad_params": zero_grads,
            "nan_grad_params": nan_grads,
            "inf_grad_params": inf_grads,
            "very_small_grad_params": very_small,
            "very_large_grad_params": very_large,
            "healthy": nan_grads == 0 and inf_grads == 0,
        }


# =============================================================================
# METRIC COMPUTATION FUNCTIONS
# =============================================================================

def compute_perplexity(loss: float, base: float = math.e) -> float:
    """
    Compute perplexity from loss.

    Args:
        loss: Cross-entropy loss
        base: Base of exponential (e for natural, 2 for bits)

    Returns:
        Perplexity value
    """
    # Clamp to avoid overflow
    loss = min(loss, 20.0)
    return base ** loss


def compute_accuracy(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = -100,
) -> float:
    """
    Compute accuracy for classification.

    Args:
        predictions: Model predictions (logits or class indices)
        labels: Ground truth labels
        ignore_index: Index to ignore in accuracy computation

    Returns:
        Accuracy as a float
    """
    if predictions.dim() > 1:
        predictions = predictions.argmax(dim=-1)

    mask = labels != ignore_index
    if mask.sum() == 0:
        return 0.0

    correct = (predictions == labels) & mask
    return correct.sum().float() / mask.sum().float()


def compute_f1(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
    average: str = "macro",
) -> float:
    """
    Compute F1 score.

    Args:
        predictions: Model predictions
        labels: Ground truth labels
        num_classes: Number of classes
        average: "micro", "macro", or "weighted"

    Returns:
        F1 score
    """
    if predictions.dim() > 1:
        predictions = predictions.argmax(dim=-1)

    predictions = predictions.view(-1).cpu()
    labels = labels.view(-1).cpu()

    # Compute per-class metrics
    f1_scores = []
    weights = []

    for c in range(num_classes):
        pred_c = predictions == c
        label_c = labels == c

        tp = (pred_c & label_c).sum().float()
        fp = (pred_c & ~label_c).sum().float()
        fn = (~pred_c & label_c).sum().float()

        precision = tp / (tp + fp + 1e-10)
        recall = tp / (tp + fn + 1e-10)
        f1 = 2 * precision * recall / (precision + recall + 1e-10)

        f1_scores.append(f1.item())
        weights.append(label_c.sum().item())

    if average == "macro":
        return sum(f1_scores) / len(f1_scores)
    elif average == "weighted":
        total_weight = sum(weights)
        return sum(f * w for f, w in zip(f1_scores, weights)) / max(total_weight, 1)
    else:  # micro
        predictions_flat = predictions.view(-1)
        labels_flat = labels.view(-1)
        correct = (predictions_flat == labels_flat).sum().float()
        return (correct / len(labels_flat)).item()


def compute_bits_per_byte(loss: float, chars_per_token: float = 4.0) -> float:
    """
    Compute bits per byte from loss.

    Args:
        loss: Cross-entropy loss (nats)
        chars_per_token: Average characters per token

    Returns:
        Bits per byte
    """
    bits_per_token = loss / math.log(2)
    return bits_per_token / chars_per_token


def compute_token_accuracy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = -100,
) -> Dict[str, float]:
    """
    Compute various token-level accuracy metrics.

    Args:
        logits: Model logits (batch, seq_len, vocab_size)
        labels: Ground truth labels (batch, seq_len)
        ignore_index: Index to ignore

    Returns:
        Dictionary of accuracy metrics
    """
    predictions = logits.argmax(dim=-1)
    mask = labels != ignore_index

    if mask.sum() == 0:
        return {"accuracy": 0.0, "top5_accuracy": 0.0}

    # Top-1 accuracy
    correct = (predictions == labels) & mask
    accuracy = correct.sum().float() / mask.sum().float()

    # Top-5 accuracy
    top5_preds = logits.topk(5, dim=-1).indices
    labels_expanded = labels.unsqueeze(-1).expand_as(top5_preds)
    top5_correct = ((top5_preds == labels_expanded).any(dim=-1)) & mask
    top5_accuracy = top5_correct.sum().float() / mask.sum().float()

    return {
        "accuracy": accuracy.item(),
        "top5_accuracy": top5_accuracy.item(),
    }


def compute_sequence_accuracy(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    pad_token_id: int = 0,
) -> float:
    """
    Compute sequence-level accuracy.

    Args:
        predictions: Predicted sequences
        labels: Ground truth sequences
        pad_token_id: Padding token ID

    Returns:
        Sequence accuracy
    """
    batch_size = predictions.shape[0]
    correct = 0

    for i in range(batch_size):
        pred_seq = predictions[i]
        label_seq = labels[i]

        # Find non-padding length
        pred_len = (pred_seq != pad_token_id).sum()
        label_len = (label_seq != pad_token_id).sum()

        if pred_len == label_len:
            if torch.all(pred_seq[:pred_len] == label_seq[:label_len]):
                correct += 1

    return correct / batch_size


# =============================================================================
# DISTRIBUTED METRICS
# =============================================================================

def all_reduce_metrics(
    metrics: Dict[str, float],
    world_size: int,
    device: torch.device,
) -> Dict[str, float]:
    """
    All-reduce metrics across distributed workers.

    Args:
        metrics: Metrics to reduce
        world_size: Number of workers
        device: Device for tensors

    Returns:
        Reduced metrics
    """
    if world_size == 1:
        return metrics

    # Stack all values
    keys = list(metrics.keys())
    values = torch.tensor([metrics[k] for k in keys], device=device)

    # All-reduce
    dist.all_reduce(values, op=dist.ReduceOp.SUM)
    values = values / world_size

    # Reconstruct dict
    return {k: v.item() for k, v in zip(keys, values)}


def gather_metrics(
    metrics: Dict[str, float],
    world_size: int,
    device: torch.device,
) -> List[Dict[str, float]]:
    """
    Gather metrics from all workers.

    Args:
        metrics: Local metrics
        world_size: Number of workers
        device: Device for tensors

    Returns:
        List of metrics from each worker
    """
    if world_size == 1:
        return [metrics]

    keys = list(metrics.keys())
    values = torch.tensor([metrics[k] for k in keys], device=device)

    # Gather from all ranks
    gathered = [torch.zeros_like(values) for _ in range(world_size)]
    dist.all_gather(gathered, values)

    # Reconstruct dicts
    return [{k: v.item() for k, v in zip(keys, g)} for g in gathered]
