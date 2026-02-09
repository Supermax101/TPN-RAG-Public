"""
Paper-grade statistical utilities for benchmark reporting.
"""

from __future__ import annotations

import math
import random
from collections import Counter
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


def wilson_ci(k: int, n: int, alpha: float = 0.05) -> Dict[str, float]:
    """
    Wilson score interval for a binomial proportion.

    Returns dict with:
    - p_hat
    - ci_low
    - ci_high
    """
    if n <= 0:
        return {"p_hat": 0.0, "ci_low": 0.0, "ci_high": 0.0}
    k = max(0, min(int(k), int(n)))
    p = k / n

    # z for two-sided alpha
    try:
        from scipy.stats import norm

        z = float(norm.ppf(1 - alpha / 2))
    except Exception:
        # Good approximation for 95% CI when scipy is unavailable.
        z = 1.959963984540054

    denom = 1 + (z * z) / n
    center = (p + (z * z) / (2 * n)) / denom
    half = (z * math.sqrt((p * (1 - p) + (z * z) / (4 * n)) / n)) / denom
    return {"p_hat": float(p), "ci_low": float(max(0.0, center - half)), "ci_high": float(min(1.0, center + half))}


def mcnemar_exact(b: int, c: int) -> float:
    """
    Exact two-sided McNemar test p-value (binomial test on discordant pairs).

    b = count(no_rag=1, rag=0)
    c = count(no_rag=0, rag=1)
    """
    b = int(max(0, b))
    c = int(max(0, c))
    n = b + c
    if n == 0:
        return 1.0

    # Two-sided exact binomial p-value with p=0.5.
    k = min(b, c)
    # Compute tail probability sum_{i=0..k} C(n,i) / 2^n
    # Use math.comb to avoid scipy dependency.
    tail = 0.0
    for i in range(0, k + 1):
        tail += math.comb(n, i)
    p_one = tail / (2**n)
    p_two = min(1.0, 2.0 * p_one)
    return float(p_two)


def cohen_kappa(a: Sequence[str], b: Sequence[str]) -> float:
    """Compute Cohen's kappa for two categorical raters."""
    if len(a) != len(b) or not a:
        return 0.0

    labels = sorted(set(a) | set(b))
    label_to_idx = {label: i for i, label in enumerate(labels)}
    n = len(labels)
    conf = np.zeros((n, n), dtype=float)
    for x, y in zip(a, b):
        conf[label_to_idx[x], label_to_idx[y]] += 1

    total = conf.sum()
    po = np.trace(conf) / total if total else 0.0
    pa = conf.sum(axis=1) / total if total else np.zeros(n)
    pb = conf.sum(axis=0) / total if total else np.zeros(n)
    pe = float(np.dot(pa, pb))
    if abs(1.0 - pe) < 1e-9:
        return 0.0
    return (po - pe) / (1.0 - pe)


def fleiss_kappa(ratings: List[List[str]]) -> float:
    """
    Compute Fleiss' kappa for multiple raters.

    ratings shape:
    - outer: items
    - inner: repeated categorical labels for one item
    """
    if not ratings:
        return 0.0

    cats = sorted({label for item in ratings for label in item})
    c_idx = {c: i for i, c in enumerate(cats)}
    n_items = len(ratings)
    n_raters = len(ratings[0])
    n_cats = len(cats)
    M = np.zeros((n_items, n_cats), dtype=float)
    for i, item in enumerate(ratings):
        for label in item:
            M[i, c_idx[label]] += 1

    p = M.sum(axis=0) / (n_items * n_raters)
    P = ((M * M).sum(axis=1) - n_raters) / (n_raters * (n_raters - 1))
    Pbar = float(P.mean())
    PbarE = float((p * p).sum())
    if abs(1.0 - PbarE) < 1e-9:
        return 0.0
    return (Pbar - PbarE) / (1.0 - PbarE)


def mcnemar_bowker(a: Sequence[str], b: Sequence[str]) -> Dict[str, float]:
    """McNemar-Bowker symmetry test for paired multi-class responses."""
    if len(a) != len(b) or not a:
        return {"statistic": 0.0, "pvalue": 1.0, "df": 0}

    labels = sorted(set(a) | set(b))
    k = len(labels)
    idx = {label: i for i, label in enumerate(labels)}
    table = np.zeros((k, k), dtype=float)
    for x, y in zip(a, b):
        table[idx[x], idx[y]] += 1

    stat = 0.0
    df = 0
    for i in range(k):
        for j in range(i + 1, k):
            nij = table[i, j]
            nji = table[j, i]
            denom = nij + nji
            if denom > 0:
                stat += (nij - nji) ** 2 / denom
                df += 1

    if df == 0:
        return {"statistic": 0.0, "pvalue": 1.0, "df": 0}

    # Chi-square survival function approximation.
    try:
        from scipy.stats import chi2

        pvalue = float(chi2.sf(stat, df))
    except Exception:
        # Fallback approximation if scipy is unavailable.
        pvalue = math.exp(-0.5 * stat)
    return {"statistic": float(stat), "pvalue": pvalue, "df": int(df)}


def paired_bootstrap_ci(
    a: Sequence[float],
    b: Sequence[float],
    n_bootstrap: int = 5000,
    alpha: float = 0.05,
    seed: int = 42,
) -> Dict[str, float]:
    """Paired bootstrap confidence interval for mean delta (a - b)."""
    if len(a) != len(b) or not a:
        return {"mean_delta": 0.0, "ci_low": 0.0, "ci_high": 0.0}

    random.seed(seed)
    n = len(a)
    deltas = []
    for _ in range(n_bootstrap):
        idxs = [random.randrange(n) for _ in range(n)]
        sample = [a[i] - b[i] for i in idxs]
        deltas.append(float(np.mean(sample)))

    deltas = sorted(deltas)
    mean_delta = float(np.mean([x - y for x, y in zip(a, b)]))
    low_idx = int((alpha / 2) * len(deltas))
    high_idx = int((1 - alpha / 2) * len(deltas)) - 1
    return {
        "mean_delta": mean_delta,
        "ci_low": deltas[max(0, low_idx)],
        "ci_high": deltas[min(len(deltas) - 1, high_idx)],
    }


def cohen_d_paired(a: Sequence[float], b: Sequence[float]) -> float:
    """Effect size for paired measurements."""
    if len(a) != len(b) or not a:
        return 0.0
    diffs = np.array(a) - np.array(b)
    std = float(np.std(diffs, ddof=1)) if len(diffs) > 1 else 0.0
    if std == 0:
        return 0.0
    return float(np.mean(diffs) / std)


def holm_bonferroni(pvalues: Dict[str, float], alpha: float = 0.05) -> Dict[str, bool]:
    """Holm-Bonferroni multiple-comparison correction."""
    ordered = sorted(pvalues.items(), key=lambda x: x[1])
    m = len(ordered)
    decisions: Dict[str, bool] = {k: False for k in pvalues}
    for rank, (name, p) in enumerate(ordered, 1):
        threshold = alpha / (m - rank + 1)
        if p <= threshold:
            decisions[name] = True
        else:
            # stop rule for Holm
            break
    return decisions
