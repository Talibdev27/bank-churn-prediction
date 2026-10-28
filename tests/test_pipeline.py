"""End-to-end: churn.train then churn.report on the synthetic data."""

import json

import numpy as np
import pandas as pd

from churn import report, train
from churn.report import gains_table


def test_train_and_report_end_to_end(data_dir, tmp_path, test_df):
    reports = tmp_path / "reports"
    submission = tmp_path / "submission.csv"

    train.main(
        [
            "--data-dir", str(data_dir),
            "--reports-dir", str(reports),
            "--submission", str(submission),
            "--folds", "3",
        ]
    )  # fmt: skip
    report.main(["--data-dir", str(data_dir), "--reports-dir", str(reports)])

    metrics = json.loads((reports / "metrics.json").read_text())
    assert set(metrics["models"]) == {"logreg", "hgb"}
    assert metrics["best_model"] in metrics["models"]

    sub = pd.read_csv(submission)
    assert sub["id"].tolist() == test_df["id"].tolist()
    assert sub["Exited"].between(0, 1).all()

    for chart in ["gains.png", "importance.png", "partial_dependence.png"]:
        assert (reports / chart).stat().st_size > 0


def test_gains_table_perfect_ranking():
    y = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    scores = np.linspace(1, 0, len(y))

    table = gains_table(y, scores, shares=(0.2, 0.5))

    assert table["churners_caught"].tolist() == [1.0, 1.0]
    assert table["precision"].tolist() == [1.0, 0.4]
    assert table["lift"].tolist() == [5.0, 2.0]
