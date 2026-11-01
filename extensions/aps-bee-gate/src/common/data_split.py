from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedShuffleSplit

from common.selective_metrics import as_binary_array


@dataclass(frozen=True)
class SplitIndices:
    train: np.ndarray
    calibration: np.ndarray
    threshold_selection: np.ndarray
    test: np.ndarray


def temporal_month_splits(months: Iterable[int | float | str]) -> dict[str, list]:
    values = [month for month in months if not _is_missing(month)]
    ordered = sorted(set(values))
    if len(ordered) < 8:
        raise ValueError(f"Need at least 8 ordered months; found {len(ordered)}")
    selected = ordered[:8]
    return {
        "train": selected[0:5],
        "calibration": [selected[5]],
        "threshold_selection": [selected[6]],
        "test": [selected[7]],
    }


def assign_temporal_split(months: Iterable[int | float | str]) -> np.ndarray:
    month_arr = np.asarray(list(months))
    split_months = temporal_month_splits(month_arr)
    month_to_split = {
        month: split_name
        for split_name, split_values in split_months.items()
        for month in split_values
    }
    return np.asarray([month_to_split.get(month, "unused") for month in month_arr], dtype=object)


def temporal_split_indices(months: Iterable[int | float | str]) -> SplitIndices:
    assigned = assign_temporal_split(months)
    return SplitIndices(
        train=np.flatnonzero(assigned == "train"),
        calibration=np.flatnonzero(assigned == "calibration"),
        threshold_selection=np.flatnonzero(assigned == "threshold_selection"),
        test=np.flatnonzero(assigned == "test"),
    )


def stratified_train_calibration_threshold_test_split(
    y: Iterable[int],
    *,
    train_size: float = 0.60,
    calibration_size: float = 0.15,
    threshold_selection_size: float = 0.15,
    test_size: float = 0.10,
    seed: int = 7,
) -> SplitIndices:
    labels = as_binary_array(y, "y")
    sizes = np.asarray([train_size, calibration_size, threshold_selection_size, test_size], dtype=float)
    if (sizes <= 0).any():
        raise ValueError("all split sizes must be positive")
    if not np.isclose(sizes.sum(), 1.0):
        raise ValueError(f"split sizes must sum to 1.0; got {sizes.sum()}")

    indices = np.arange(len(labels))
    train_idx, rest_idx = _stratified_split(indices, labels, train_size=train_size, seed=seed)
    rest_labels = labels[rest_idx]

    cal_fraction_of_rest = calibration_size / (calibration_size + threshold_selection_size + test_size)
    cal_rel, holdout_rel = _stratified_split(
        np.arange(len(rest_idx)),
        rest_labels,
        train_size=cal_fraction_of_rest,
        seed=seed + 1,
    )
    calibration_idx = rest_idx[cal_rel]
    holdout_idx = rest_idx[holdout_rel]
    holdout_labels = labels[holdout_idx]

    threshold_fraction_of_holdout = threshold_selection_size / (threshold_selection_size + test_size)
    threshold_rel, test_rel = _stratified_split(
        np.arange(len(holdout_idx)),
        holdout_labels,
        train_size=threshold_fraction_of_holdout,
        seed=seed + 2,
    )

    split = SplitIndices(
        train=np.sort(train_idx),
        calibration=np.sort(calibration_idx),
        threshold_selection=np.sort(holdout_idx[threshold_rel]),
        test=np.sort(holdout_idx[test_rel]),
    )
    assert_disjoint_complete(split, n=len(labels))
    assert_each_split_has_both_classes(labels, split)
    return split


def stratified_predefined_test_split(
    y: Iterable[int],
    partitions: Iterable[object],
    *,
    train_label: object = "train",
    test_label: object = "test",
    train_size: float = 0.70,
    calibration_size: float = 0.15,
    threshold_selection_size: float = 0.15,
    seed: int = 7,
) -> SplitIndices:
    """Split a predefined training pool while preserving its official test set."""
    labels = as_binary_array(y, "y")
    partition_values = np.asarray(list(partitions), dtype=object)
    if len(partition_values) != len(labels):
        raise ValueError("partitions and y must have the same length")

    sizes = np.asarray([train_size, calibration_size, threshold_selection_size], dtype=float)
    if (sizes <= 0).any() or not np.isclose(sizes.sum(), 1.0):
        raise ValueError("predefined training-pool split sizes must be positive and sum to 1.0")

    train_pool = np.flatnonzero(partition_values == train_label)
    test = np.flatnonzero(partition_values == test_label)
    known = np.concatenate([train_pool, test])
    if len(np.unique(known)) != len(labels) or set(known.tolist()) != set(range(len(labels))):
        unknown = sorted(set(partition_values.tolist()) - {train_label, test_label})
        raise ValueError(f"partitions must contain only train/test labels; unknown={unknown}")

    train_rel, holdout_rel = _stratified_split(
        np.arange(len(train_pool)),
        labels[train_pool],
        train_size=train_size,
        seed=seed,
    )
    holdout_pool = train_pool[holdout_rel]
    calibration_fraction = calibration_size / (calibration_size + threshold_selection_size)
    calibration_rel, threshold_rel = _stratified_split(
        np.arange(len(holdout_pool)),
        labels[holdout_pool],
        train_size=calibration_fraction,
        seed=seed + 1,
    )

    split = SplitIndices(
        train=np.sort(train_pool[train_rel]),
        calibration=np.sort(holdout_pool[calibration_rel]),
        threshold_selection=np.sort(holdout_pool[threshold_rel]),
        test=np.sort(test),
    )
    assert_disjoint_complete(split, n=len(labels))
    assert_each_split_has_both_classes(labels, split)
    return split


def repeated_stratified_cv_indices(
    y: Iterable[int],
    *,
    n_splits: int = 5,
    n_repeats: int = 3,
    seed: int = 7,
) -> list[dict[str, np.ndarray | int]]:
    labels = as_binary_array(y, "y")
    splitter = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    rows = []
    for fold_id, (train_idx, test_idx) in enumerate(splitter.split(np.zeros(len(labels)), labels)):
        rows.append({"fold": fold_id, "train": np.sort(train_idx), "test": np.sort(test_idx)})
    return rows


def assert_disjoint_complete(split: SplitIndices, *, n: int) -> None:
    pieces = [split.train, split.calibration, split.threshold_selection, split.test]
    combined = np.concatenate(pieces)
    if len(np.unique(combined)) != len(combined):
        raise AssertionError("split indices are not disjoint")
    if set(combined.tolist()) != set(range(n)):
        raise AssertionError("split indices do not cover all rows exactly once")


def assert_each_split_has_both_classes(y: np.ndarray, split: SplitIndices) -> None:
    for name in ["train", "calibration", "threshold_selection", "test"]:
        indices = getattr(split, name)
        values = set(y[indices].tolist())
        if values != {0, 1}:
            raise AssertionError(f"{name} split must contain both classes; found {sorted(values)}")


def _stratified_split(indices: np.ndarray, labels: np.ndarray, *, train_size: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    splitter = StratifiedShuffleSplit(n_splits=1, train_size=train_size, random_state=seed)
    train_rel, test_rel = next(splitter.split(np.zeros(len(labels)), labels))
    return indices[train_rel], indices[test_rel]


def _is_missing(value: object) -> bool:
    try:
        return bool(np.isnan(value))
    except TypeError:
        return False
