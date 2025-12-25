"""
SPARSA-LM Evaluation Harness
Unified interface for running benchmark evaluations
"""

import json
import logging
import time
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import torch
import torch.distributed as dist

from .benchmarks import (
    Benchmark,
    BenchmarkConfig,
    BenchmarkResult,
    load_benchmark,
    evaluate_benchmark,
    BENCHMARK_REGISTRY,
)

logger = logging.getLogger(__name__)


@dataclass
class EvaluationConfig:
    """Configuration for evaluation harness."""

    # Benchmarks to run
    benchmarks: List[str] = field(default_factory=lambda: [
        "arc_challenge",
        "hellaswag",
        "mmlu",
        "truthfulqa",
    ])

    # Benchmark-specific settings
    benchmark_configs: Dict[str, BenchmarkConfig] = field(default_factory=dict)

    # Default settings
    num_fewshot: int = 0
    batch_size: int = 8
    max_samples: Optional[int] = None

    # Output
    output_dir: str = "./eval_results"
    save_per_sample: bool = False
    save_predictions: bool = True

    # Distributed
    use_distributed: bool = False

    # Device
    device: Optional[str] = None

    # Logging
    verbose: bool = True

    def get_benchmark_config(self, name: str) -> BenchmarkConfig:
        """Get configuration for a specific benchmark."""
        if name in self.benchmark_configs:
            return self.benchmark_configs[name]

        # Create default config
        return BenchmarkConfig(
            num_fewshot=self.num_fewshot,
            batch_size=self.batch_size,
            max_samples=self.max_samples,
        )


class EvaluationHarness:
    """
    Unified evaluation harness for running benchmarks.

    Features:
    - Multiple benchmark support
    - Distributed evaluation
    - Result aggregation and saving
    - Progress tracking
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        config: EvaluationConfig,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config

        # Setup device
        if config.device:
            self.device = torch.device(config.device)
        else:
            self.device = next(model.parameters()).device

        # Setup output
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Results storage
        self.results: Dict[str, BenchmarkResult] = {}

    def run(self) -> Dict[str, BenchmarkResult]:
        """
        Run all configured benchmarks.

        Returns:
            Dict mapping benchmark names to results
        """
        start_time = time.time()

        for benchmark_name in self.config.benchmarks:
            try:
                result = self._run_benchmark(benchmark_name)
                self.results[benchmark_name] = result

                if self.config.verbose:
                    logger.info(f"Completed {benchmark_name}: {result}")

            except Exception as e:
                logger.error(f"Failed to run {benchmark_name}: {e}")
                continue

        total_time = time.time() - start_time

        # Save aggregated results
        self._save_results(total_time)

        return self.results

    def _run_benchmark(self, name: str) -> BenchmarkResult:
        """Run a single benchmark."""
        logger.info(f"Running benchmark: {name}")

        # Get config
        benchmark_config = self.config.get_benchmark_config(name)

        # Load benchmark
        benchmark = load_benchmark(name, benchmark_config)

        # Run evaluation
        result = evaluate_benchmark(
            self.model,
            self.tokenizer,
            benchmark,
            device=self.device,
        )

        # Save individual results if configured
        if self.config.save_predictions:
            self._save_benchmark_results(name, result)

        return result

    def _save_benchmark_results(self, name: str, result: BenchmarkResult):
        """Save individual benchmark results."""
        output_path = self.output_dir / f"{name}_results.json"

        data = {
            "benchmark": result.benchmark_name,
            "num_samples": result.num_samples,
            "metrics": result.metrics,
        }

        if self.config.save_per_sample and result.per_sample_results:
            data["per_sample"] = result.per_sample_results

        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)

    def _save_results(self, total_time: float):
        """Save aggregated results."""
        output_path = self.output_dir / "all_results.json"

        aggregated = {
            "total_time_seconds": total_time,
            "benchmarks": {},
        }

        for name, result in self.results.items():
            aggregated["benchmarks"][name] = {
                "num_samples": result.num_samples,
                "metrics": result.metrics,
            }

        # Compute average scores
        all_accuracies = []
        for result in self.results.values():
            if "accuracy" in result.metrics:
                all_accuracies.append(result.metrics["accuracy"])

        if all_accuracies:
            aggregated["average_accuracy"] = sum(all_accuracies) / len(all_accuracies)

        with open(output_path, "w") as f:
            json.dump(aggregated, f, indent=2)

        logger.info(f"Saved results to {output_path}")

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of evaluation results."""
        summary = {
            "benchmarks_run": list(self.results.keys()),
            "metrics": {},
        }

        for name, result in self.results.items():
            summary["metrics"][name] = result.metrics

        # Compute aggregate metrics
        accuracies = [
            r.metrics.get("accuracy", 0.0)
            for r in self.results.values()
        ]
        if accuracies:
            summary["average_accuracy"] = sum(accuracies) / len(accuracies)

        return summary


class DistributedEvaluationHarness(EvaluationHarness):
    """
    Distributed evaluation harness.

    Shards evaluation across multiple GPUs/nodes.
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        config: EvaluationConfig,
    ):
        super().__init__(model, tokenizer, config)

        # Distributed setup
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        self.is_main = self.rank == 0

    def _run_benchmark(self, name: str) -> BenchmarkResult:
        """Run benchmark with distributed sharding."""
        logger.info(f"[Rank {self.rank}] Running benchmark: {name}")

        benchmark_config = self.config.get_benchmark_config(name)
        benchmark = load_benchmark(name, benchmark_config)

        # Shard examples across ranks
        total_examples = len(benchmark.examples)
        examples_per_rank = total_examples // self.world_size
        start_idx = self.rank * examples_per_rank
        end_idx = start_idx + examples_per_rank

        # Last rank gets remainder
        if self.rank == self.world_size - 1:
            end_idx = total_examples

        # Create sharded benchmark
        original_examples = benchmark.examples
        benchmark.examples = benchmark.examples[start_idx:end_idx]

        # Run evaluation on shard
        result = evaluate_benchmark(
            self.model,
            self.tokenizer,
            benchmark,
            device=self.device,
        )

        # Restore original examples
        benchmark.examples = original_examples

        # Gather results from all ranks
        result = self._gather_results(name, result)

        return result

    def _gather_results(self, name: str, local_result: BenchmarkResult) -> BenchmarkResult:
        """Gather results from all ranks."""
        if self.world_size == 1:
            return local_result

        # Gather metrics
        metrics_tensor = torch.tensor(
            [local_result.metrics.get("accuracy", 0.0), local_result.num_samples],
            device=self.device,
        )

        gathered = [torch.zeros_like(metrics_tensor) for _ in range(self.world_size)]
        dist.all_gather(gathered, metrics_tensor)

        # Aggregate
        total_correct = sum(g[0].item() * g[1].item() for g in gathered)
        total_samples = sum(g[1].item() for g in gathered)

        aggregated_metrics = {
            "accuracy": total_correct / total_samples if total_samples > 0 else 0.0,
        }

        # Gather per-sample results on main rank
        per_sample = None
        if self.config.save_per_sample and self.is_main:
            # Would need object gathering - simplified here
            per_sample = local_result.per_sample_results

        return BenchmarkResult(
            benchmark_name=name,
            num_samples=int(total_samples),
            metrics=aggregated_metrics,
            per_sample_results=per_sample,
            config=local_result.config,
        )


def run_evaluation(
    model: Any,
    tokenizer: Any,
    config: Optional[EvaluationConfig] = None,
    benchmarks: Optional[List[str]] = None,
    output_dir: Optional[str] = None,
    num_fewshot: int = 0,
    distributed: bool = False,
) -> Dict[str, BenchmarkResult]:
    """
    Run evaluation with simplified interface.

    Args:
        model: Language model
        tokenizer: Tokenizer
        config: Optional full configuration
        benchmarks: List of benchmark names (overrides config)
        output_dir: Output directory (overrides config)
        num_fewshot: Number of few-shot examples
        distributed: Whether to use distributed evaluation

    Returns:
        Dict of benchmark results
    """
    if config is None:
        config = EvaluationConfig()

    if benchmarks:
        config.benchmarks = benchmarks
    if output_dir:
        config.output_dir = output_dir
    config.num_fewshot = num_fewshot
    config.use_distributed = distributed

    # Choose harness based on distributed setting
    if distributed and dist.is_initialized() and dist.get_world_size() > 1:
        harness = DistributedEvaluationHarness(model, tokenizer, config)
    else:
        harness = EvaluationHarness(model, tokenizer, config)

    return harness.run()


# =============================================================================
# LEADERBOARD PRESETS
# =============================================================================

LEADERBOARD_CONFIGS = {
    "open_llm_leaderboard": EvaluationConfig(
        benchmarks=[
            "arc_challenge",
            "hellaswag",
            "mmlu",
            "truthfulqa",
            "winogrande",
            "gsm8k",
        ],
        num_fewshot=0,
    ),
    "small_eval": EvaluationConfig(
        benchmarks=["arc_challenge", "hellaswag"],
        num_fewshot=0,
        max_samples=100,
    ),
    "code_eval": EvaluationConfig(
        benchmarks=["humaneval", "mbpp"],
        num_fewshot=0,
    ),
    "reasoning": EvaluationConfig(
        benchmarks=["arc_challenge", "hellaswag", "winogrande"],
        num_fewshot=5,
    ),
}


def get_leaderboard_config(name: str) -> EvaluationConfig:
    """Get a predefined leaderboard configuration."""
    if name not in LEADERBOARD_CONFIGS:
        raise ValueError(f"Unknown leaderboard: {name}. Available: {list(LEADERBOARD_CONFIGS.keys())}")
    return LEADERBOARD_CONFIGS[name]
