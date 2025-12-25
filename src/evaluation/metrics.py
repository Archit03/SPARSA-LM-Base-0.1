"""
SPARSA-LM Evaluation Metrics
Comprehensive metrics for LLM evaluation
"""

import math
import re
import string
from typing import List, Dict, Optional, Any, Tuple
from collections import Counter

import torch


def compute_accuracy(
    predictions: List[Any],
    references: List[Any],
    normalize: bool = True,
) -> float:
    """
    Compute accuracy (exact match rate).

    Args:
        predictions: Model predictions
        references: Ground truth
        normalize: Whether to normalize to [0, 1]

    Returns:
        Accuracy score
    """
    if not predictions:
        return 0.0

    correct = sum(
        str(p).strip().lower() == str(r).strip().lower()
        for p, r in zip(predictions, references)
    )

    if normalize:
        return correct / len(predictions)
    return correct


def compute_perplexity(
    loss: float,
    base: float = math.e,
) -> float:
    """
    Compute perplexity from cross-entropy loss.

    Args:
        loss: Cross-entropy loss
        base: Base of exponential (e for natural, 2 for bits)

    Returns:
        Perplexity value
    """
    # Clamp to avoid overflow
    loss = min(loss, 100.0)
    return base ** loss


def compute_perplexity_from_logits(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = -100,
) -> float:
    """
    Compute perplexity from logits and labels.

    Args:
        logits: Model logits (batch, seq_len, vocab_size)
        labels: Ground truth labels
        ignore_index: Index to ignore

    Returns:
        Perplexity
    """
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    loss = torch.nn.functional.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=ignore_index,
        reduction="mean",
    )

    return compute_perplexity(loss.item())


def compute_exact_match(
    predictions: List[str],
    references: List[str],
    normalize_fn: Optional[callable] = None,
) -> float:
    """
    Compute exact match score.

    Args:
        predictions: Model predictions
        references: Ground truth
        normalize_fn: Optional normalization function

    Returns:
        Exact match score [0, 1]
    """
    if not predictions:
        return 0.0

    if normalize_fn is None:
        normalize_fn = lambda x: x.strip().lower()

    correct = sum(
        normalize_fn(str(p)) == normalize_fn(str(r))
        for p, r in zip(predictions, references)
    )

    return correct / len(predictions)


def normalize_answer(s: str) -> str:
    """
    Normalize answer string for QA evaluation.

    Handles:
    - Lowercasing
    - Removing articles
    - Removing punctuation
    - Removing extra whitespace
    """
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def compute_f1(
    predictions: List[str],
    references: List[str],
    normalize: bool = True,
) -> float:
    """
    Compute F1 score (token-level).

    Args:
        predictions: Model predictions
        references: Ground truth
        normalize: Whether to normalize answers

    Returns:
        Average F1 score
    """
    def get_tokens(s: str) -> List[str]:
        if normalize:
            s = normalize_answer(s)
        return s.split()

    def token_f1(pred: str, ref: str) -> float:
        pred_tokens = get_tokens(pred)
        ref_tokens = get_tokens(ref)

        if not pred_tokens or not ref_tokens:
            return float(pred_tokens == ref_tokens)

        common = Counter(pred_tokens) & Counter(ref_tokens)
        num_same = sum(common.values())

        if num_same == 0:
            return 0.0

        precision = num_same / len(pred_tokens)
        recall = num_same / len(ref_tokens)

        return 2 * precision * recall / (precision + recall)

    if not predictions:
        return 0.0

    scores = [token_f1(p, r) for p, r in zip(predictions, references)]
    return sum(scores) / len(scores)


def compute_pass_at_k(
    n: int,
    c: int,
    k: int,
) -> float:
    """
    Compute pass@k for code generation.

    Args:
        n: Total number of samples
        c: Number of correct samples
        k: k in pass@k

    Returns:
        pass@k probability
    """
    if n - c < k:
        return 1.0

    # Use log to avoid overflow
    return 1.0 - math.exp(
        sum(math.log(n - c - i) - math.log(n - i) for i in range(k))
    )


def compute_pass_at_k_batch(
    results: List[List[bool]],
    k_values: List[int] = [1, 10, 100],
) -> Dict[str, float]:
    """
    Compute pass@k for a batch of problems.

    Args:
        results: List of lists of pass/fail results per problem
        k_values: k values to compute

    Returns:
        Dict mapping pass@k to score
    """
    scores = {f"pass@{k}": [] for k in k_values}

    for problem_results in results:
        n = len(problem_results)
        c = sum(problem_results)

        for k in k_values:
            if k <= n:
                scores[f"pass@{k}"].append(compute_pass_at_k(n, c, k))

    return {k: sum(v) / len(v) if v else 0.0 for k, v in scores.items()}


def compute_rouge(
    predictions: List[str],
    references: List[str],
    rouge_types: List[str] = ["rouge1", "rouge2", "rougeL"],
) -> Dict[str, float]:
    """
    Compute ROUGE scores.

    Args:
        predictions: Generated summaries
        references: Reference summaries
        rouge_types: ROUGE types to compute

    Returns:
        Dict of ROUGE scores
    """
    try:
        from rouge_score import rouge_scorer
    except ImportError:
        # Simple fallback implementation
        return _compute_rouge_simple(predictions, references)

    scorer = rouge_scorer.RougeScorer(rouge_types, use_stemmer=True)

    scores = {rt: [] for rt in rouge_types}

    for pred, ref in zip(predictions, references):
        result = scorer.score(ref, pred)
        for rt in rouge_types:
            scores[rt].append(result[rt].fmeasure)

    return {rt: sum(v) / len(v) if v else 0.0 for rt, v in scores.items()}


def _compute_rouge_simple(
    predictions: List[str],
    references: List[str],
) -> Dict[str, float]:
    """Simple ROUGE-L implementation without dependencies."""
    def lcs_length(x: List[str], y: List[str]) -> int:
        m, n = len(x), len(y)
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if x[i - 1] == y[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

        return dp[m][n]

    def rouge_l(pred: str, ref: str) -> float:
        pred_tokens = pred.lower().split()
        ref_tokens = ref.lower().split()

        if not pred_tokens or not ref_tokens:
            return 0.0

        lcs = lcs_length(pred_tokens, ref_tokens)
        precision = lcs / len(pred_tokens)
        recall = lcs / len(ref_tokens)

        if precision + recall == 0:
            return 0.0

        return 2 * precision * recall / (precision + recall)

    scores = [rouge_l(p, r) for p, r in zip(predictions, references)]
    return {"rougeL": sum(scores) / len(scores) if scores else 0.0}


def compute_bleu(
    predictions: List[str],
    references: List[List[str]],
    max_n: int = 4,
    smooth: bool = True,
) -> float:
    """
    Compute BLEU score.

    Args:
        predictions: Generated texts
        references: Reference texts (can have multiple references per prediction)
        max_n: Maximum n-gram order
        smooth: Whether to apply smoothing

    Returns:
        BLEU score
    """
    try:
        from sacrebleu import corpus_bleu
        # Ensure references are in correct format
        if references and isinstance(references[0], str):
            references = [[r] for r in references]

        # Transpose for sacrebleu format
        refs_transposed = list(zip(*references))
        return corpus_bleu(predictions, refs_transposed).score / 100
    except ImportError:
        return _compute_bleu_simple(predictions, references, max_n, smooth)


def _compute_bleu_simple(
    predictions: List[str],
    references: List[List[str]],
    max_n: int = 4,
    smooth: bool = True,
) -> float:
    """Simple BLEU implementation without dependencies."""
    def get_ngrams(tokens: List[str], n: int) -> Counter:
        return Counter(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))

    def sentence_bleu(pred: str, refs: List[str]) -> Tuple[float, int]:
        pred_tokens = pred.lower().split()

        if not pred_tokens:
            return 0.0, 0

        ref_tokens_list = [r.lower().split() for r in refs]

        precisions = []
        for n in range(1, max_n + 1):
            pred_ngrams = get_ngrams(pred_tokens, n)
            max_ref_counts = Counter()

            for ref_tokens in ref_tokens_list:
                ref_ngrams = get_ngrams(ref_tokens, n)
                for ngram, count in ref_ngrams.items():
                    max_ref_counts[ngram] = max(max_ref_counts[ngram], count)

            clipped_counts = 0
            total_counts = 0

            for ngram, count in pred_ngrams.items():
                clipped_counts += min(count, max_ref_counts.get(ngram, 0))
                total_counts += count

            if total_counts == 0:
                precision = 0.0
            else:
                if smooth and clipped_counts == 0:
                    precision = 1.0 / (total_counts + 1)
                else:
                    precision = clipped_counts / total_counts

            precisions.append(precision)

        # Geometric mean of precisions
        if min(precisions) > 0:
            log_precision = sum(math.log(p) for p in precisions) / max_n
            geo_mean = math.exp(log_precision)
        else:
            geo_mean = 0.0

        # Brevity penalty
        pred_len = len(pred_tokens)
        ref_lens = [len(r) for r in ref_tokens_list]
        closest_ref_len = min(ref_lens, key=lambda x: abs(x - pred_len))

        if pred_len > closest_ref_len:
            bp = 1.0
        elif pred_len == 0:
            bp = 0.0
        else:
            bp = math.exp(1 - closest_ref_len / pred_len)

        return geo_mean * bp, pred_len

    # Handle single reference case
    if references and isinstance(references[0], str):
        references = [[r] for r in references]

    total_bleu = 0.0
    total_len = 0

    for pred, refs in zip(predictions, references):
        bleu, length = sentence_bleu(pred, refs)
        total_bleu += bleu * length
        total_len += length

    return total_bleu / total_len if total_len > 0 else 0.0


def compute_bits_per_byte(
    loss: float,
    chars_per_token: float = 4.0,
) -> float:
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


def compute_calibration_error(
    confidences: List[float],
    accuracies: List[bool],
    n_bins: int = 10,
) -> Dict[str, float]:
    """
    Compute Expected Calibration Error (ECE).

    Args:
        confidences: Model confidence scores
        accuracies: Whether each prediction was correct
        n_bins: Number of bins

    Returns:
        Dict with ECE and MCE
    """
    if not confidences:
        return {"ece": 0.0, "mce": 0.0}

    bins = [[] for _ in range(n_bins)]
    acc_bins = [[] for _ in range(n_bins)]

    for conf, acc in zip(confidences, accuracies):
        bin_idx = min(int(conf * n_bins), n_bins - 1)
        bins[bin_idx].append(conf)
        acc_bins[bin_idx].append(float(acc))

    ece = 0.0
    mce = 0.0

    for i in range(n_bins):
        if bins[i]:
            avg_conf = sum(bins[i]) / len(bins[i])
            avg_acc = sum(acc_bins[i]) / len(acc_bins[i])
            gap = abs(avg_conf - avg_acc)

            ece += len(bins[i]) / len(confidences) * gap
            mce = max(mce, gap)

    return {"ece": ece, "mce": mce}


def compute_diversity_metrics(
    texts: List[str],
) -> Dict[str, float]:
    """
    Compute diversity metrics for generated texts.

    Args:
        texts: List of generated texts

    Returns:
        Dict with diversity metrics
    """
    all_tokens = []
    all_bigrams = []
    all_trigrams = []

    for text in texts:
        tokens = text.lower().split()
        all_tokens.extend(tokens)

        bigrams = list(zip(tokens[:-1], tokens[1:]))
        all_bigrams.extend(bigrams)

        trigrams = list(zip(tokens[:-2], tokens[1:-1], tokens[2:]))
        all_trigrams.extend(trigrams)

    distinct_1 = len(set(all_tokens)) / len(all_tokens) if all_tokens else 0
    distinct_2 = len(set(all_bigrams)) / len(all_bigrams) if all_bigrams else 0
    distinct_3 = len(set(all_trigrams)) / len(all_trigrams) if all_trigrams else 0

    return {
        "distinct_1": distinct_1,
        "distinct_2": distinct_2,
        "distinct_3": distinct_3,
    }


def compute_consistency(
    responses: List[List[str]],
) -> float:
    """
    Compute consistency across multiple responses to same prompt.

    Args:
        responses: List of response lists (multiple responses per prompt)

    Returns:
        Consistency score [0, 1]
    """
    if not responses:
        return 0.0

    consistencies = []
    for response_set in responses:
        if len(response_set) < 2:
            continue

        # Pairwise consistency
        pairs = 0
        matches = 0
        for i, r1 in enumerate(response_set):
            for r2 in response_set[i + 1:]:
                pairs += 1
                if normalize_answer(r1) == normalize_answer(r2):
                    matches += 1

        if pairs > 0:
            consistencies.append(matches / pairs)

    return sum(consistencies) / len(consistencies) if consistencies else 0.0
