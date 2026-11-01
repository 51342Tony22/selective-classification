import numpy as np

from common.resampling import class_counts, downsample_majority_indices, parse_minority_majority_ratio


def test_parse_minority_majority_ratio_accepts_research_notation():
    assert parse_minority_majority_ratio("1:5") == 5.0
    assert parse_minority_majority_ratio("2:10") == 5.0
    assert parse_minority_majority_ratio(20) == 20.0


def test_downsample_majority_indices_keeps_all_minority_and_targets_ratio():
    y = np.array([0] * 100 + [1] * 10)
    selected = downsample_majority_indices(y, majority_per_minority="1:5", seed=7)
    counts = class_counts(y[selected])

    assert counts == {"positives": 10, "negatives": 50}
    assert set(np.flatnonzero(y == 1)).issubset(set(selected.tolist()))


def test_downsample_majority_does_not_oversample_if_majority_is_insufficient():
    y = np.array([0] * 8 + [1] * 10)
    selected = downsample_majority_indices(y, majority_per_minority="1:5", seed=7)
    counts = class_counts(y[selected])

    assert counts == {"positives": 10, "negatives": 8}

