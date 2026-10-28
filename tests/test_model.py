import pytest

from churn.data import split_xy
from churn.model import MODELS


@pytest.mark.parametrize("name", list(MODELS))
def test_pipeline_fits_and_predicts(name, train_df, test_df):
    X, y = split_xy(train_df)
    model = MODELS[name]().fit(X, y)

    proba = model.predict_proba(test_df[X.columns])[:, 1]

    assert proba.shape == (len(test_df),)
    assert ((proba >= 0) & (proba <= 1)).all()


@pytest.mark.parametrize("name", list(MODELS))
def test_pipeline_handles_unseen_category(name, train_df, test_df):
    X, y = split_xy(train_df)
    model = MODELS[name]().fit(X, y)
    unseen = test_df[X.columns].copy()
    unseen["Geography"] = "Italy"

    proba = model.predict_proba(unseen)[:, 1]

    assert proba.shape == (len(unseen),)
