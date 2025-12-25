"""
SPARSA-LM Benchmark Implementations
Standard LLM evaluation benchmarks
"""

import json
import logging
import math
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Union
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark evaluation."""

    # Dataset settings
    dataset_name: str = ""
    subset: Optional[str] = None
    split: str = "validation"

    # Evaluation settings
    num_fewshot: int = 0
    batch_size: int = 8
    max_samples: Optional[int] = None  # Limit samples for debugging

    # Generation settings (for generative tasks)
    max_new_tokens: int = 256
    temperature: float = 0.0  # Greedy by default
    top_p: float = 1.0
    stop_sequences: List[str] = field(default_factory=list)

    # Scoring
    normalize_by_length: bool = True
    use_logprobs: bool = True


@dataclass
class BenchmarkResult:
    """Result from benchmark evaluation."""

    benchmark_name: str
    num_samples: int
    metrics: Dict[str, float]
    per_sample_results: Optional[List[Dict]] = None
    config: Optional[BenchmarkConfig] = None

    def __repr__(self):
        metrics_str = ", ".join(f"{k}={v:.4f}" for k, v in self.metrics.items())
        return f"BenchmarkResult({self.benchmark_name}: {metrics_str})"


class Benchmark(ABC):
    """
    Abstract base class for benchmarks.

    Each benchmark implements:
    - Loading examples from the dataset
    - Formatting prompts with few-shot examples
    - Computing metrics from model outputs
    """

    name: str = "benchmark"
    task_type: str = "multiple_choice"  # multiple_choice, generation, classification

    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.examples: List[Dict] = []
        self.fewshot_examples: List[Dict] = []

    @abstractmethod
    def load_examples(self) -> List[Dict]:
        """Load examples from the dataset."""
        pass

    @abstractmethod
    def format_prompt(self, example: Dict, include_answer: bool = False) -> str:
        """Format a single example as a prompt."""
        pass

    @abstractmethod
    def extract_answer(self, response: str) -> str:
        """Extract the answer from model response."""
        pass

    @abstractmethod
    def compute_metrics(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        """Compute metrics from predictions."""
        pass

    def get_fewshot_prompt(self) -> str:
        """Get few-shot examples as a prompt prefix."""
        if self.config.num_fewshot == 0:
            return ""

        prompts = []
        for example in self.fewshot_examples[:self.config.num_fewshot]:
            prompts.append(self.format_prompt(example, include_answer=True))

        return "\n\n".join(prompts) + "\n\n"

    def prepare(self):
        """Prepare benchmark (load data, setup few-shot)."""
        self.examples = self.load_examples()

        if self.config.max_samples:
            self.examples = self.examples[:self.config.max_samples]

        # Load few-shot examples from train split if available
        if self.config.num_fewshot > 0:
            try:
                train_config = BenchmarkConfig(**self.config.__dict__)
                train_config.split = "train"
                train_config.max_samples = self.config.num_fewshot * 2
                original_split = self.config.split
                self.config.split = "train"
                self.fewshot_examples = self.load_examples()
                self.config.split = original_split
            except Exception:
                logger.warning(f"Could not load few-shot examples for {self.name}")


class MultipleChoiceBenchmark(Benchmark):
    """Base class for multiple choice benchmarks."""

    task_type = "multiple_choice"

    def __init__(self, config: BenchmarkConfig):
        super().__init__(config)
        self.choices = ["A", "B", "C", "D", "E"]

    def score_choices(
        self,
        model: Any,
        tokenizer: Any,
        prompt: str,
        choices: List[str],
    ) -> Tuple[int, List[float]]:
        """
        Score multiple choice options using log probabilities.

        Returns:
            Tuple of (best_choice_index, choice_logprobs)
        """
        logprobs = []

        for choice in choices:
            full_text = prompt + choice
            encoding = tokenizer(
                full_text,
                return_tensors="pt",
                add_special_tokens=True,
            ).to(model.device)

            with torch.no_grad():
                outputs = model(
                    input_ids=encoding["input_ids"],
                    attention_mask=encoding["attention_mask"],
                )
                logits = outputs["logits"]

            # Get log probability of the choice tokens
            prompt_encoding = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
            prompt_len = prompt_encoding["input_ids"].shape[1]

            choice_logits = logits[0, prompt_len - 1:-1]
            choice_ids = encoding["input_ids"][0, prompt_len:]

            log_probs = F.log_softmax(choice_logits, dim=-1)
            choice_log_prob = 0.0
            for i, token_id in enumerate(choice_ids):
                choice_log_prob += log_probs[i, token_id].item()

            # Normalize by length if configured
            if self.config.normalize_by_length:
                choice_log_prob /= len(choice_ids)

            logprobs.append(choice_log_prob)

        best_idx = max(range(len(logprobs)), key=lambda i: logprobs[i])
        return best_idx, logprobs

    def compute_metrics(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        """Compute accuracy metrics."""
        correct = sum(p == r for p, r in zip(predictions, references))
        return {
            "accuracy": correct / len(predictions),
            "num_correct": correct,
            "num_total": len(predictions),
        }


class GenerativeBenchmark(Benchmark):
    """Base class for generative benchmarks."""

    task_type = "generation"

    def extract_answer(self, response: str) -> str:
        """Default extraction - return full response stripped."""
        return response.strip()

    def compute_metrics(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        """Default metrics for generative tasks."""
        from .metrics import compute_exact_match, compute_f1

        exact_match = compute_exact_match(predictions, references)
        f1 = compute_f1(predictions, references)

        return {
            "exact_match": exact_match,
            "f1": f1,
        }


# =============================================================================
# SPECIFIC BENCHMARK IMPLEMENTATIONS
# =============================================================================

class ARC(MultipleChoiceBenchmark):
    """
    AI2 Reasoning Challenge (ARC) benchmark.

    Tests science reasoning with multiple choice questions.
    """

    name = "arc"

    def load_examples(self) -> List[Dict]:
        """Load ARC dataset."""
        try:
            from datasets import load_dataset

            subset = self.config.subset or "ARC-Challenge"
            ds = load_dataset("allenai/ai2_arc", subset, split=self.config.split)

            examples = []
            for item in ds:
                choices = item["choices"]["text"]
                labels = item["choices"]["label"]

                examples.append({
                    "question": item["question"],
                    "choices": choices,
                    "labels": labels,
                    "answer": item["answerKey"],
                })

            return examples
        except Exception as e:
            logger.error(f"Failed to load ARC: {e}")
            return []

    def format_prompt(self, example: Dict, include_answer: bool = False) -> str:
        """Format ARC prompt."""
        prompt = f"Question: {example['question']}\n\nChoices:\n"
        for label, choice in zip(example["labels"], example["choices"]):
            prompt += f"{label}. {choice}\n"
        prompt += "\nAnswer:"

        if include_answer:
            prompt += f" {example['answer']}"

        return prompt

    def extract_answer(self, response: str) -> str:
        """Extract single letter answer."""
        response = response.strip().upper()
        for char in response:
            if char in "ABCDE":
                return char
        return ""


class HellaSwag(MultipleChoiceBenchmark):
    """
    HellaSwag commonsense reasoning benchmark.

    Tests ability to complete scenarios with plausible endings.
    """

    name = "hellaswag"

    def load_examples(self) -> List[Dict]:
        """Load HellaSwag dataset."""
        try:
            from datasets import load_dataset

            ds = load_dataset("Rowan/hellaswag", split=self.config.split)

            examples = []
            for item in ds:
                examples.append({
                    "context": self._preprocess(item["ctx"]),
                    "activity_label": item["activity_label"],
                    "endings": [self._preprocess(e) for e in item["endings"]],
                    "answer": int(item["label"]),
                })

            return examples
        except Exception as e:
            logger.error(f"Failed to load HellaSwag: {e}")
            return []

    def _preprocess(self, text: str) -> str:
        """Preprocess HellaSwag text."""
        text = text.strip()
        text = text.replace(" [title]", ". ")
        text = re.sub(r"\[.*?\]", "", text)
        text = text.replace("  ", " ")
        return text

    def format_prompt(self, example: Dict, include_answer: bool = False) -> str:
        """Format HellaSwag prompt."""
        prompt = f"{example['context']}"

        if include_answer:
            prompt += f" {example['endings'][example['answer']]}"

        return prompt

    def extract_answer(self, response: str) -> str:
        """For HellaSwag, we typically use log-prob scoring."""
        return response.strip()


class MMLU(MultipleChoiceBenchmark):
    """
    Massive Multitask Language Understanding (MMLU) benchmark.

    Tests knowledge across 57 subjects.
    """

    name = "mmlu"

    SUBJECTS = [
        "abstract_algebra", "anatomy", "astronomy", "business_ethics",
        "clinical_knowledge", "college_biology", "college_chemistry",
        "college_computer_science", "college_mathematics", "college_medicine",
        "college_physics", "computer_security", "conceptual_physics",
        "econometrics", "electrical_engineering", "elementary_mathematics",
        "formal_logic", "global_facts", "high_school_biology",
        "high_school_chemistry", "high_school_computer_science",
        "high_school_european_history", "high_school_geography",
        "high_school_government_and_politics", "high_school_macroeconomics",
        "high_school_mathematics", "high_school_microeconomics",
        "high_school_physics", "high_school_psychology",
        "high_school_statistics", "high_school_us_history",
        "high_school_world_history", "human_aging", "human_sexuality",
        "international_law", "jurisprudence", "logical_fallacies",
        "machine_learning", "management", "marketing", "medical_genetics",
        "miscellaneous", "moral_disputes", "moral_scenarios", "nutrition",
        "philosophy", "prehistory", "professional_accounting",
        "professional_law", "professional_medicine", "professional_psychology",
        "public_relations", "security_studies", "sociology", "us_foreign_policy",
        "virology", "world_religions",
    ]

    def load_examples(self) -> List[Dict]:
        """Load MMLU dataset."""
        try:
            from datasets import load_dataset

            subject = self.config.subset or "all"
            split = self.config.split if self.config.split != "validation" else "dev"

            if subject == "all":
                all_examples = []
                for subj in self.SUBJECTS:
                    try:
                        ds = load_dataset("cais/mmlu", subj, split=split)
                        for item in ds:
                            all_examples.append({
                                "question": item["question"],
                                "choices": item["choices"],
                                "answer": item["answer"],
                                "subject": subj,
                            })
                    except Exception:
                        continue
                return all_examples
            else:
                ds = load_dataset("cais/mmlu", subject, split=split)
                return [
                    {
                        "question": item["question"],
                        "choices": item["choices"],
                        "answer": item["answer"],
                        "subject": subject,
                    }
                    for item in ds
                ]
        except Exception as e:
            logger.error(f"Failed to load MMLU: {e}")
            return []

    def format_prompt(self, example: Dict, include_answer: bool = False) -> str:
        """Format MMLU prompt."""
        choices_str = "\n".join(
            f"{chr(65 + i)}. {choice}"
            for i, choice in enumerate(example["choices"])
        )
        prompt = f"Question: {example['question']}\n{choices_str}\nAnswer:"

        if include_answer:
            prompt += f" {chr(65 + example['answer'])}"

        return prompt

    def extract_answer(self, response: str) -> str:
        """Extract letter answer."""
        response = response.strip().upper()
        for char in response:
            if char in "ABCD":
                return char
        return ""


class TruthfulQA(MultipleChoiceBenchmark):
    """
    TruthfulQA benchmark.

    Tests truthfulness and avoidance of false claims.
    """

    name = "truthfulqa"

    def load_examples(self) -> List[Dict]:
        """Load TruthfulQA dataset."""
        try:
            from datasets import load_dataset

            ds = load_dataset("truthful_qa", "multiple_choice", split="validation")

            examples = []
            for item in ds:
                mc = item["mc1_targets"]
                choices = mc["choices"]
                labels = mc["labels"]
                answer_idx = labels.index(1)

                examples.append({
                    "question": item["question"],
                    "choices": choices,
                    "answer": answer_idx,
                    "category": item.get("category", ""),
                })

            return examples
        except Exception as e:
            logger.error(f"Failed to load TruthfulQA: {e}")
            return []

    def format_prompt(self, example: Dict, include_answer: bool = False) -> str:
        """Format TruthfulQA prompt."""
        choices_str = "\n".join(
            f"{chr(65 + i)}. {choice}"
            for i, choice in enumerate(example["choices"])
        )
        prompt = f"Q: {example['question']}\n{choices_str}\nA:"

        if include_answer:
            prompt += f" {chr(65 + example['answer'])}"

        return prompt

    def extract_answer(self, response: str) -> str:
        """Extract letter answer."""
        response = response.strip().upper()
        for char in response:
            if char in "ABCDEFGH":
                return char
        return ""


class GSM8K(GenerativeBenchmark):
    """
    Grade School Math 8K (GSM8K) benchmark.

    Tests mathematical reasoning with word problems.
    """

    name = "gsm8k"

    def load_examples(self) -> List[Dict]:
        """Load GSM8K dataset."""
        try:
            from datasets import load_dataset

            split = self.config.split if self.config.split != "validation" else "test"
            ds = load_dataset("gsm8k", "main", split=split)

            examples = []
            for item in ds:
                # Extract final answer
                answer_match = re.search(r"####\s*(-?\d[\d,]*)", item["answer"])
                final_answer = answer_match.group(1).replace(",", "") if answer_match else ""

                examples.append({
                    "question": item["question"],
                    "solution": item["answer"],
                    "answer": final_answer,
                })

            return examples
        except Exception as e:
            logger.error(f"Failed to load GSM8K: {e}")
            return []

    def format_prompt(self, example: Dict, include_answer: bool = False) -> str:
        """Format GSM8K prompt."""
        prompt = f"Question: {example['question']}\n\nLet's solve this step by step:\n"

        if include_answer:
            prompt += f"\n{example['solution']}"

        return prompt

    def extract_answer(self, response: str) -> str:
        """Extract numerical answer."""
        # Look for #### pattern
        match = re.search(r"####\s*(-?\d[\d,]*)", response)
        if match:
            return match.group(1).replace(",", "")

        # Look for final number
        numbers = re.findall(r"-?\d[\d,]*\.?\d*", response)
        if numbers:
            return numbers[-1].replace(",", "")

        return ""

    def compute_metrics(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        """Compute GSM8K metrics."""
        correct = 0
        for pred, ref in zip(predictions, references):
            try:
                pred_num = float(pred.replace(",", ""))
                ref_num = float(ref.replace(",", ""))
                if abs(pred_num - ref_num) < 1e-5:
                    correct += 1
            except ValueError:
                continue

        return {
            "accuracy": correct / len(predictions),
            "num_correct": correct,
            "num_total": len(predictions),
        }


class HumanEval(GenerativeBenchmark):
    """
    HumanEval code generation benchmark.

    Tests Python code generation from docstrings.
    """

    name = "humaneval"

    def load_examples(self) -> List[Dict]:
        """Load HumanEval dataset."""
        try:
            from datasets import load_dataset

            ds = load_dataset("openai_humaneval", split="test")

            examples = []
            for item in ds:
                examples.append({
                    "task_id": item["task_id"],
                    "prompt": item["prompt"],
                    "canonical_solution": item["canonical_solution"],
                    "test": item["test"],
                    "entry_point": item["entry_point"],
                })

            return examples
        except Exception as e:
            logger.error(f"Failed to load HumanEval: {e}")
            return []

    def format_prompt(self, example: Dict, include_answer: bool = False) -> str:
        """Format HumanEval prompt."""
        prompt = example["prompt"]

        if include_answer:
            prompt += example["canonical_solution"]

        return prompt

    def extract_answer(self, response: str) -> str:
        """Extract code completion."""
        # Find the end of the function
        lines = response.split("\n")
        result = []
        in_function = True
        indent_level = None

        for line in lines:
            if indent_level is None and line.strip():
                # Determine initial indentation
                indent_level = len(line) - len(line.lstrip())

            if line.strip().startswith("def ") and result:
                # New function definition - stop
                break

            if line.strip() and len(line) - len(line.lstrip()) < indent_level and result:
                # Dedented - function ended
                break

            result.append(line)

        return "\n".join(result)

    def compute_metrics(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        """Compute HumanEval metrics (pass@k)."""
        from .metrics import compute_pass_at_k

        # This requires executing code which is done in the harness
        # Return placeholder metrics here
        return {
            "pass@1": 0.0,
            "pass@10": 0.0,
        }


class MBPP(GenerativeBenchmark):
    """
    Mostly Basic Python Problems (MBPP) benchmark.

    Tests basic Python programming ability.
    """

    name = "mbpp"

    def load_examples(self) -> List[Dict]:
        """Load MBPP dataset."""
        try:
            from datasets import load_dataset

            ds = load_dataset("mbpp", split="test")

            examples = []
            for item in ds:
                examples.append({
                    "task_id": item["task_id"],
                    "prompt": item["text"],
                    "code": item["code"],
                    "test_list": item["test_list"],
                })

            return examples
        except Exception as e:
            logger.error(f"Failed to load MBPP: {e}")
            return []

    def format_prompt(self, example: Dict, include_answer: bool = False) -> str:
        """Format MBPP prompt."""
        prompt = f"# {example['prompt']}\n\n"

        if include_answer:
            prompt += example["code"]

        return prompt

    def extract_answer(self, response: str) -> str:
        """Extract Python code."""
        # Remove markdown code blocks if present
        if "```python" in response:
            match = re.search(r"```python\n(.*?)```", response, re.DOTALL)
            if match:
                return match.group(1)
        if "```" in response:
            match = re.search(r"```\n(.*?)```", response, re.DOTALL)
            if match:
                return match.group(1)

        return response.strip()


class WinoGrande(MultipleChoiceBenchmark):
    """
    WinoGrande commonsense reasoning benchmark.

    Tests pronoun resolution in sentences.
    """

    name = "winogrande"

    def load_examples(self) -> List[Dict]:
        """Load WinoGrande dataset."""
        try:
            from datasets import load_dataset

            ds = load_dataset("winogrande", "winogrande_xl", split=self.config.split)

            examples = []
            for item in ds:
                examples.append({
                    "sentence": item["sentence"],
                    "option1": item["option1"],
                    "option2": item["option2"],
                    "answer": int(item["answer"]) - 1,  # Convert to 0-indexed
                })

            return examples
        except Exception as e:
            logger.error(f"Failed to load WinoGrande: {e}")
            return []

    def format_prompt(self, example: Dict, include_answer: bool = False) -> str:
        """Format WinoGrande prompt."""
        sentence = example["sentence"]
        prompt = f"Sentence: {sentence}\n\nA. {example['option1']}\nB. {example['option2']}\n\nAnswer:"

        if include_answer:
            prompt += f" {'A' if example['answer'] == 0 else 'B'}"

        return prompt

    def extract_answer(self, response: str) -> str:
        """Extract A or B answer."""
        response = response.strip().upper()
        for char in response:
            if char in "AB":
                return char
        return ""


# =============================================================================
# BENCHMARK LOADING UTILITIES
# =============================================================================

BENCHMARK_REGISTRY: Dict[str, type] = {
    "arc": ARC,
    "arc_easy": ARC,
    "arc_challenge": ARC,
    "hellaswag": HellaSwag,
    "mmlu": MMLU,
    "truthfulqa": TruthfulQA,
    "gsm8k": GSM8K,
    "humaneval": HumanEval,
    "mbpp": MBPP,
    "winogrande": WinoGrande,
}


def load_benchmark(
    name: str,
    config: Optional[BenchmarkConfig] = None,
) -> Benchmark:
    """
    Load a benchmark by name.

    Args:
        name: Benchmark name
        config: Optional configuration

    Returns:
        Benchmark instance
    """
    name = name.lower()

    if name not in BENCHMARK_REGISTRY:
        raise ValueError(f"Unknown benchmark: {name}. Available: {list(BENCHMARK_REGISTRY.keys())}")

    if config is None:
        config = BenchmarkConfig()

    benchmark_cls = BENCHMARK_REGISTRY[name]
    benchmark = benchmark_cls(config)
    benchmark.prepare()

    return benchmark


def evaluate_benchmark(
    model: Any,
    tokenizer: Any,
    benchmark: Benchmark,
    device: Optional[torch.device] = None,
) -> BenchmarkResult:
    """
    Evaluate a model on a benchmark.

    Args:
        model: Language model
        tokenizer: Tokenizer
        benchmark: Benchmark instance
        device: Device for evaluation

    Returns:
        BenchmarkResult with metrics
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    predictions = []
    references = []
    per_sample = []

    fewshot_prefix = benchmark.get_fewshot_prompt()

    with torch.no_grad():
        for example in benchmark.examples:
            prompt = fewshot_prefix + benchmark.format_prompt(example)

            if benchmark.task_type == "multiple_choice":
                # Use log-prob scoring for multiple choice
                if isinstance(benchmark, MultipleChoiceBenchmark):
                    choices = example.get("endings") or example.get("choices", [])
                    if isinstance(example.get("answer"), int):
                        ref = str(example["answer"])
                    else:
                        ref = example["answer"]

                    # Score each choice
                    best_idx, logprobs = benchmark.score_choices(model, tokenizer, prompt, choices)
                    pred = str(best_idx)

                    predictions.append(pred)
                    references.append(ref)
                    per_sample.append({
                        "prompt": prompt,
                        "prediction": pred,
                        "reference": ref,
                        "logprobs": logprobs,
                    })
            else:
                # Generative evaluation
                encoding = tokenizer(
                    prompt,
                    return_tensors="pt",
                    add_special_tokens=True,
                ).to(device)

                outputs = model.generate(
                    input_ids=encoding["input_ids"],
                    attention_mask=encoding["attention_mask"],
                    max_new_tokens=benchmark.config.max_new_tokens,
                    temperature=benchmark.config.temperature,
                    top_p=benchmark.config.top_p,
                    do_sample=benchmark.config.temperature > 0,
                )

                response = tokenizer.decode(
                    outputs[0][encoding["input_ids"].shape[1]:],
                    skip_special_tokens=True,
                )

                pred = benchmark.extract_answer(response)
                ref = example.get("answer", "")

                predictions.append(pred)
                references.append(str(ref))
                per_sample.append({
                    "prompt": prompt,
                    "response": response,
                    "prediction": pred,
                    "reference": ref,
                })

    # Compute metrics
    metrics = benchmark.compute_metrics(predictions, references)

    return BenchmarkResult(
        benchmark_name=benchmark.name,
        num_samples=len(predictions),
        metrics=metrics,
        per_sample_results=per_sample,
        config=benchmark.config,
    )
