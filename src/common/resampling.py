from __future__ import annotations

from typing import Iterable

import numpy as np

from common.selective_metrics import as_binary_array


def parse_minority_majority_ratio(ratio: str | float | int) -> float:
    if isinstance(ratio, str):
        text = ratio.strip()
        if ":" in text:
            left, right = text.split(":", maxsplit=1)
            minority = float(left)
            majority = float(right)
            if minority <= 0 or majority <= 0:
                raise ValueError("ratio values must be positive")
            return majority / minority
        value = float(text)
    else:
        value = float(ratio)
    if value <= 0:
        raise ValueError("majority-per-minority ratio must be positive")
    return value


def downsample_majority_indices(
    y: Iterable[int],
    *,
    majority_per_minority: str | float | int,
    seed: int = 7,
    minority_label: int = 1,
) -> np.ndarray:
    if minority_label not in {0, 1}:
        raise ValueError("minority_label must be 0 or 1")

    labels = as_binary_array(y, "y")
    majority_label = 1 - minority_label
    minority_idx = np.flatnonzero(labels == minority_label)
    majority_idx = np.flatnonzero(labels == majority_label)
    if len(minority_idx) == 0:
        raise ValueError("cannot downsample without minority examples")
    if len(majority_idx) == 0:
        raise ValueError("cannot downsample without majority examples")

    ratio = parse_minority_majority_ratio(majority_per_minority)
    target_majority = int(round(len(minority_idx) * ratio))
    target_majority = min(target_majority, len(majority_idx))
    if target_majority <= 0:
        raise ValueError("target majority sample count must be positive")

    rng = np.random.default_rng(seed)
    sampled_majority = rng.choice(majority_idx, size=target_majority, replace=False)
    selected = np.concatenate([minority_idx, sampled_majority])
    return np.sort(selected)


def class_counts(y: Iterable[int]) -> dict[str, int]:
    labels = as_binary_array(y, "y")
    positives = int((labels == 1).sum())
    negatives = int((labels == 0).sum())
    return {"positives": positives, "negatives": negatives}

