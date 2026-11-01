from imbalance_selective.experiment_matrix import build_evaluation_jobs, build_training_jobs, summarize_matrix


def test_build_training_jobs_counts_cv_and_controlled_ratios():
    config = {
        "project": {"seeds": [7]},
        "datasets": ["online_shoppers"],
        "models": ["logistic_regression", "lightgbm"],
        "controlled_imbalance": {"train_ratios": ["1:5", "1:10"]},
    }

    jobs = build_training_jobs(config)

    assert len(jobs) == 6
    assert set(jobs["experiment"]) == {"natural_imbalance", "controlled_imbalance"}
    assert set(jobs["model"]) == {"logistic_regression", "lightgbm"}


def test_build_evaluation_jobs_expands_calibration_and_policy_grid():
    config = {
        "project": {"seeds": [7]},
        "datasets": ["online_shoppers"],
        "models": ["logistic_regression"],
        "controlled_imbalance": {"train_ratios": []},
        "calibration": {"methods": ["none", "platt"]},
        "thresholds": {"single_baselines": ["macro_f1"], "abstention_budgets": [0.05, 0.10]},
    }
    training = build_training_jobs(config)
    evaluation = build_evaluation_jobs(config, training)

    assert len(training) == 1
    assert len(evaluation) == 6
    assert set(evaluation["policy_family"]) == {"single", "dual"}


def test_summarize_matrix_reports_training_and_evaluation_counts():
    config = {
        "project": {"seeds": [7]},
        "datasets": ["online_shoppers"],
        "models": ["logistic_regression"],
        "controlled_imbalance": {"train_ratios": []},
        "calibration": {"methods": ["none"]},
        "thresholds": {"single_baselines": ["macro_f1"], "abstention_budgets": [0.05]},
    }
    training = build_training_jobs(config)
    evaluation = build_evaluation_jobs(config, training)
    summary = summarize_matrix(training, evaluation)

    assert summary.loc[0, "training_jobs"] == 1
    assert summary.loc[0, "evaluation_jobs"] == 2

