"""Turn churn.train's outputs into charts: cumulative gains (business view), permutation
importance and partial dependence (why customers churn).

Run after churn.train: uv run python -m churn.report
"""

import argparse
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import PartialDependenceDisplay, permutation_importance

from churn.data import load_data, split_xy

PDP_FEATURES = ["NumOfProducts", "Age", "IsActiveMember", "Geography"]
PDP_CATEGORICAL = ["NumOfProducts", "IsActiveMember", "Geography"]


def cumulative_gains(y, scores):
    """Share of customers contacted (riskiest first) vs share of all churners caught."""
    caught = np.asarray(y)[np.argsort(-np.asarray(scores))].cumsum()
    contacted = np.arange(1, len(caught) + 1) / len(caught)
    return contacted, caught / caught[-1]


def gains_table(y, scores, shares=(0.1, 0.2, 0.3)):
    """For each share of riskiest customers contacted: recall, precision and lift over random."""
    y_sorted = np.asarray(y)[np.argsort(-np.asarray(scores))]
    rows = []
    for share in shares:
        top = y_sorted[: int(np.ceil(share * len(y_sorted)))]
        rows.append(
            {
                "contacted": share,
                "churners_caught": round(top.sum() / y_sorted.sum(), 3),
                "precision": round(top.mean(), 3),
                "lift": round(top.mean() / y_sorted.mean(), 2),
            }
        )
    return pd.DataFrame(rows)


def plot_gains(oof, models, path):
    fig, ax = plt.subplots(figsize=(6, 5))
    for name in models:
        contacted, caught = cumulative_gains(oof["Exited"], oof[name])
        ax.plot(contacted, caught, label=name)
    ax.plot([0, 1], [0, 1], "--", color="grey", label="random")
    ax.set_xlabel("Share of customers contacted (riskiest first)")
    ax.set_ylabel("Share of churners caught")
    ax.set_title("Cumulative gains (out-of-fold)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_importance(model, X, y, path, seed=42):
    result = permutation_importance(model, X, y, scoring="roc_auc", n_repeats=10, random_state=seed)
    importance = pd.DataFrame(
        {"mean": result.importances_mean, "std": result.importances_std}, index=X.columns
    ).sort_values("mean")
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.barh(importance.index, importance["mean"], xerr=importance["std"])
    ax.set_xlabel("Drop in AUC when the feature is shuffled")
    ax.set_title("Permutation importance (held-out fold)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return importance


def plot_partial_dependence(model, X, path):
    fig, ax = plt.subplots(figsize=(12, 3.5))
    PartialDependenceDisplay.from_estimator(
        model, X, PDP_FEATURES, categorical_features=PDP_CATEGORICAL, n_cols=4, ax=ax
    )
    fig.suptitle("Partial dependence: predicted churn probability")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Draw report charts from churn.train outputs.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        help="folder with train.csv and test.csv (default: download from Kaggle)",
    )
    parser.add_argument("--reports-dir", type=Path, default=Path("reports"))
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    reports = args.reports_dir
    metrics = json.loads((reports / "metrics.json").read_text())
    oof = pd.read_csv(reports / "oof_predictions.csv")
    best = metrics["best_model"]

    plot_gains(oof, list(metrics["models"]), reports / "gains.png")
    table = gains_table(oof["Exited"], oof[best])
    (reports / "gains.json").write_text(table.to_json(orient="records", indent=2) + "\n")
    print(f"Gains for {best}:\n{table.to_string(index=False)}\n")

    # Importance and partial dependence use the last CV fold's model on its own held-out rows.
    holdout = joblib.load(reports / "holdout_model.joblib")
    train, _ = load_data(args.data_dir)
    X, y = split_xy(train)
    X_val, y_val = X.iloc[holdout["val_idx"]], y.iloc[holdout["val_idx"]]
    importance = plot_importance(
        holdout["model"], X_val, y_val, reports / "importance.png", args.seed
    )
    plot_partial_dependence(holdout["model"], X_val, reports / "partial_dependence.png")
    print(f"Permutation importance ({holdout['name']}):\n{importance['mean'][::-1].round(4)}")


if __name__ == "__main__":
    main()
