import numpy as np

from common.data_split import (
    assert_disjoint_complete,
    assign_temporal_split,
    repeated_stratified_cv_indices,
    stratified_predefined_test_split,
    stratified_train_calibration_threshold_test_split,
    temporal_month_splits,
    temporal_split_indices,
)


def test_temporal_month_split_contract_matches_baf_plan():
    splits = temporal_month_splits(range(8))

    assert splits["train"] == [0, 1, 2, 3, 4]
    assert splits["calibration"] == [5]
    assert splits["threshold_selection"] == [6]
    assert splits["test"] == [7]


def test_assign_temporal_split_marks_extra_months_unused():
    assigned = assign_temporal_split([0, 1, 2, 3, 4, 5, 6, 7, 8])

    assert assigned.tolist() == [
        "train",
        "train",
        "train",
        "train",
        "train",
        "calibration",
        "threshold_selection",
        "test",
        "unused",
    ]


def test_temporal_split_indices_are_disjoint_for_first_eight_months():
    months = np.repeat(np.arange(8), 3)
    split = temporal_split_indices(months)

    assert_disjoint_complete(split, n=len(months))
    assert len(split.train) == 15
    assert len(split.calibration) == 3
    assert len(split.threshold_selection) == 3
    assert len(split.test) == 3


def test_stratified_four_way_split_is_disjoint_and_class_preserving():
    y = np.array([0] * 80 + [1] * 20)
    split = stratified_train_calibration_threshold_test_split(y, seed=7)

    assert_disjoint_complete(split, n=len(y))
    for indices in [split.train, split.calibration, split.threshold_selection, split.test]:
        assert set(y[indices].tolist()) == {0, 1}


def test_repeated_stratified_cv_returns_requested_number_of_folds():
    y = np.array([0] * 90 + [1] * 30)
    folds = repeated_stratified_cv_indices(y, n_splits=5, n_repeats=3, seed=7)

    assert len(folds) == 15
    for row in folds:
        train = row["train"]
        test = row["test"]
        assert set(train).isdisjoint(set(test))
        assert set(y[train].tolist()) == {0, 1}
        assert set(y[test].tolist()) == {0, 1}


def test_predefined_test_split_keeps_official_test_untouched():
    y = np.array([0] * 90 + [1] * 10 + [0] * 45 + [1] * 5)
    partitions = np.array(["train"] * 100 + ["test"] * 50)

    split = stratified_predefined_test_split(y, partitions, seed=7)

    assert set(split.test.tolist()) == set(range(100, 150))
    assert len(split.train) == 70
    assert len(split.calibration) == 15
    assert len(split.threshold_selection) == 15
    assert set(y[split.test].tolist()) == {0, 1}
