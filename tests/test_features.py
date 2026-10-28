import pandas as pd

from churn.features import add_features


def test_add_features_values():
    df = pd.DataFrame(
        {
            "Age": [25.0, 45.0, 65.0],
            "Balance": [0.0, 100_000.0, 50_000.0],
            "NumOfProducts": [1.0, 2.0, 1.0],
            "IsActiveMember": [1.0, 0.0, 1.0],
        }
    )
    out = add_features(df)

    assert out["ZeroBalance"].tolist() == [1, 0, 0]
    assert out["IsSenior"].tolist() == [0, 0, 1]
    assert out["AgeBin"].tolist() == [0, 2, 4]
    assert out["ActiveByProducts"].tolist() == [1.0, 0.0, 1.0]
    assert out["BalancePerProduct"].tolist() == [0.0, 50_000.0, 50_000.0]


def test_add_features_does_not_modify_input():
    df = pd.DataFrame(
        {"Age": [30.0], "Balance": [0.0], "NumOfProducts": [1.0], "IsActiveMember": [0.0]}
    )
    add_features(df)
    assert list(df.columns) == ["Age", "Balance", "NumOfProducts", "IsActiveMember"]
