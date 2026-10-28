"""Model pipelines. Each takes the raw FEATURES columns and does its own feature engineering and
encoding, so encoders are fit on training folds only and train/test always get the same columns.
"""

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, OrdinalEncoder, StandardScaler

from churn.features import add_features


def build_logreg():
    # One-hot NumOfProducts so a linear model can learn its U-shaped effect.
    categorical = ["Geography", "Gender", "NumOfProducts"]
    return make_pipeline(
        FunctionTransformer(add_features),
        ColumnTransformer(
            [("cat", OneHotEncoder(handle_unknown="ignore"), categorical)],
            remainder=StandardScaler(),
        ),
        LogisticRegression(max_iter=2000),
    )


def build_hgb(random_state=42):
    # HistGB handles categoricals natively; it only needs them as integer codes.
    # Unseen categories become -1, which HistGB treats as missing.
    categorical = ["Geography", "Gender"]
    return make_pipeline(
        FunctionTransformer(add_features),
        ColumnTransformer(
            [
                (
                    "cat",
                    OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
                    categorical,
                )
            ],
            remainder="passthrough",
            verbose_feature_names_out=False,
        ).set_output(transform="pandas"),
        HistGradientBoostingClassifier(categorical_features=categorical, random_state=random_state),
    )


MODELS = {"logreg": build_logreg, "hgb": build_hgb}
