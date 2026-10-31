"""Feature engineering, motivated by the EDA in notebooks/churn.ipynb."""

import pandas as pd


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # 65% of customers have a zero balance, and they churn differently.
    df["ZeroBalance"] = (df["Balance"] == 0).astype(int)
    # Churn climbs steeply with age.
    df["IsSenior"] = (df["Age"] >= 60).astype(int)
    df["AgeBin"] = pd.cut(df["Age"], bins=[0, 30, 40, 50, 60, 200], labels=False)
    # Older customers who are inactive may be the riskiest group of all.
    df["InactiveSenior"] = ((df["Age"] >= 50) & (df["IsActiveMember"] == 0)).astype(int)
    # Churn is U-shaped in NumOfProducts, and inactive customers churn more.
    df["ActiveByProducts"] = df["IsActiveMember"] * df["NumOfProducts"]
    df["BalancePerProduct"] = df["Balance"] / df["NumOfProducts"]
    return df
