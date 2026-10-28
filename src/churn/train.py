"""Cross-validate the models, then fit the best one and write metrics, out-of-fold predictions
and a Kaggle submission.

Run: uv run python -m churn.train
"""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from churn.data import FEATURES, load_data, split_xy
from churn.model import MODELS


def cross_validate(model, X, y, folds=5, seed=42):
    """Return per-fold AUCs, out-of-fold predictions, and the last fold's fitted model together
    with that fold's validation indices (held-out data for the report's importance plots)."""
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    oof = np.zeros(len(y))
    aucs = []
    for train_idx, val_idx in cv.split(X, y):
        fitted = clone(model).fit(X.iloc[train_idx], y.iloc[train_idx])
        oof[val_idx] = fitted.predict_proba(X.iloc[val_idx])[:, 1]
        aucs.append(roc_auc_score(y.iloc[val_idx], oof[val_idx]))
    return np.array(aucs), oof, fitted, val_idx


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Cross-validate churn models and write a submission."
    )
    parser.add_argument("--model", choices=[*MODELS, "all"], default="all")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--data-dir",
        type=Path,
        help="folder with train.csv and test.csv (default: download from Kaggle)",
    )
    parser.add_argument("--reports-dir", type=Path, default=Path("reports"))
    parser.add_argument("--submission", type=Path, default=Path("submission.csv"))
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    train, test = load_data(args.data_dir)
    X, y = split_xy(train)
    names = list(MODELS) if args.model == "all" else [args.model]

    scores = {}
    oof = pd.DataFrame({"id": train["id"], "Exited": y})
    holdouts = {}
    for name in names:
        aucs, oof[name], fold_model, val_idx = cross_validate(
            MODELS[name](), X, y, args.folds, args.seed
        )
        scores[name] = {
            "auc_mean": round(float(aucs.mean()), 4),
            "auc_std": round(float(aucs.std()), 4),
            "auc_folds": aucs.round(4).tolist(),
        }
        holdouts[name] = {"name": name, "model": fold_model, "val_idx": val_idx}
        print(f"{name}: AUC {aucs.mean():.4f} ± {aucs.std():.4f}")

    best = max(scores, key=lambda n: scores[n]["auc_mean"])
    print(f"best model: {best}")

    args.reports_dir.mkdir(parents=True, exist_ok=True)
    metrics = {"folds": args.folds, "seed": args.seed, "best_model": best, "models": scores}
    (args.reports_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    oof.to_csv(args.reports_dir / "oof_predictions.csv", index=False)
    joblib.dump(holdouts[best], args.reports_dir / "holdout_model.joblib")

    final = MODELS[best]().fit(X, y)
    pred = final.predict_proba(test[FEATURES])[:, 1]
    pd.DataFrame({"id": test["id"], "Exited": pred}).to_csv(args.submission, index=False)
    print(f"wrote {args.submission}")


if __name__ == "__main__":
    main()
