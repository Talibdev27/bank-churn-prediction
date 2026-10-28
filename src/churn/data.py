"""Load the Kaggle competition data."""

from pathlib import Path

import kagglehub
import pandas as pd

COMPETITION = "binaryclassificationwithabankchurndataset"
TARGET = "Exited"

# Raw inputs the models use. id, CustomerId and Surname are left out: CustomerId is a synthetic
# artifact (see the notebook) and Surname is a high-cardinality name with broken encoding.
FEATURES = [
    "CreditScore",
    "Geography",
    "Gender",
    "Age",
    "Tenure",
    "Balance",
    "NumOfProducts",
    "HasCrCard",
    "IsActiveMember",
    "EstimatedSalary",
]


def load_data(data_dir: Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (train, test). Downloads from Kaggle (cached) unless data_dir is given."""
    if data_dir is None:
        data_dir = Path(kagglehub.competition_download(COMPETITION))
    data_dir = Path(data_dir)
    return pd.read_csv(data_dir / "train.csv"), pd.read_csv(data_dir / "test.csv")


def split_xy(train: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    return train[FEATURES], train[TARGET].astype(int)
