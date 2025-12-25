"""
SPARSA-LM Evaluation Framework
Comprehensive benchmark evaluation and metrics computation
"""

from .benchmarks import (
    Benchmark,
    BenchmarkConfig,
    BenchmarkResult,
    ARC,
    HellaSwag,
    MMLU,
    TruthfulQA,
    GSM8K,
    HumanEval,
    MBPP,
    WinoGrande,
    load_benchmark,
    evaluate_benchmark,
)

from .harness import (
    EvaluationHarness,
    EvaluationConfig,
    run_evaluation,
)

from .metrics import (
    compute_accuracy,
    compute_perplexity,
    compute_exact_match,
    compute_f1,
    compute_pass_at_k,
    compute_rouge,
    compute_bleu,
)

from .generation import (
    GenerationConfig,
    generate_samples,
    batch_generate,
)

__all__ = [
    # Benchmarks
    "Benchmark",
    "BenchmarkConfig",
    "BenchmarkResult",
    "ARC",
    "HellaSwag",
    "MMLU",
    "TruthfulQA",
    "GSM8K",
    "HumanEval",
    "MBPP",
    "WinoGrande",
    "load_benchmark",
    "evaluate_benchmark",
    # Harness
    "EvaluationHarness",
    "EvaluationConfig",
    "run_evaluation",
    # Metrics
    "compute_accuracy",
    "compute_perplexity",
    "compute_exact_match",
    "compute_f1",
    "compute_pass_at_k",
    "compute_rouge",
    "compute_bleu",
    # Generation
    "GenerationConfig",
    "generate_samples",
    "batch_generate",
]
