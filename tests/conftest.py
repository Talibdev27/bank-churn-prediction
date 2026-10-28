"""Synthetic stand-in for the Kaggle data, so tests never touch the network or the real files."""

import numpy as np
import pandas as pd
import pytest


def make_frame(n, seed, start_id=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "id": np.arange(start_id, start_id + n),
            "CustomerId": rng.integers(15_500_000, 15_800_000, n).astype(float),
            "Surname": rng.choice(["Smith", "Ivanov", "Rossi"], n),
            "CreditScore": rng.integers(400, 850, n).astype(float),
            "Geography": rng.choice(["France", "Germany", "Spain"], n),
            "Gender": rng.choice(["Male", "Female"], n),
            "Age": rng.integers(18, 80, n).astype(float),
            "Tenure": rng.integers(0, 11, n).astype(float),
            "Balance": np.where(rng.random(n) < 0.6, 0.0, rng.uniform(50_000, 200_000, n)),
            "NumOfProducts": rng.choice([1.0, 2.0, 3.0, 4.0], n, p=[0.45, 0.45, 0.07, 0.03]),
            "HasCrCard": rng.integers(0, 2, n).astype(float),
            "IsActiveMember": rng.integers(0, 2, n).astype(float),
            "EstimatedSalary": rng.uniform(10_000, 200_000, n),
        }
    )


@pytest.fixture
def train_df():
    df = make_frame(300, seed=0)
    churn_odds = (df["NumOfProducts"] != 2) * 1.5 + (df["Age"] > 50) * 1.0 - 1.5
    rng = np.random.default_rng(1)
    df["Exited"] = (rng.random(len(df)) < 1 / (1 + np.exp(-churn_odds))).astype(float)
    return df


@pytest.fixture
def test_df():
    return make_frame(50, seed=2, start_id=300)


@pytest.fixture
def data_dir(tmp_path, train_df, test_df):
    train_df.to_csv(tmp_path / "train.csv", index=False)
    test_df.to_csv(tmp_path / "test.csv", index=False)
    return tmp_path
