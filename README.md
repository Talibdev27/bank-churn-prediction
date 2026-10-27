# Bank Churn Prediction

Predicting which bank customers will leave (`Exited = 1`) for the Kaggle competition
[Binary Classification with a Bank Churn Dataset](https://www.kaggle.com/competitions/binaryclassificationwithabankchurndataset).
The metric is ROC-AUC.

Everything is in [`churn.ipynb`](churn.ipynb).

## Data
- 15,000 training rows and 10,000 test rows, with no missing values
- About 20% of customers churn
- Features: credit score, geography, gender, age, tenure, balance, number of products, card ownership, activity, salary

## Key findings from EDA
- **Number of products** is the strongest signal, and its effect is U-shaped: 2 products → 4% churn, 1 → 38%, 3+ → 85–100%.
- **Germany** churns at 41%, compared with about 15% for France and Spain.
- **Women** churn at 28% vs 14% for men. **Inactive members** and **older customers** churn more.
- 65% of customers have a **zero balance**.
- `CustomerId` repeats (6,306 unique ids across 15,000 rows), but repeated ids almost never share age or surname.
  This is a synthetic artifact, not real repeat customers. CV grouped by `CustomerId` gives the same score, so there is no leakage.

## Approach
- Engineered features: `ZeroBalance`, `IsSenior`, `AgeBin`, `ActiveByProducts`, `BalancePerProduct`
- All encoding happens inside scikit-learn `Pipeline`s. Each fold's encoding is fit on its training data only,
  and train and test always end up with the same columns.
- Validation: 5-fold `StratifiedKFold` (seed 42)

## Results (5-fold CV ROC-AUC)

| Model | Features | AUC |
|---|---|---|
| Logistic Regression | raw, one-hot geography/gender | 0.8829 |
| Logistic Regression | engineered, one-hot `NumOfProducts` | **0.9275 ± 0.0035** |
| HistGradientBoosting | raw | 0.9311 |
| HistGradientBoosting | engineered, native categoricals | 0.9299 ± 0.0038 |

The engineered features close most of the gap between the linear model and the boosted model.
For boosting they make no measurable difference, because trees already find these splits on their own.

## How to run
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Put your Kaggle API token in `~/.kaggle/kaggle.json` and accept the competition rules on Kaggle.
Then run `churn.ipynb`. The notebook downloads the data with `kagglehub` and writes `submission.csv`.

## Next steps
- Tune HistGB (`learning_rate`, `max_leaf_nodes`, `l2_regularization`)
- Try LightGBM and CatBoost, then blend models
- Compare CV scores against the public leaderboard
