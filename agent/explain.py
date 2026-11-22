"""Which inputs drove the model's score for one transaction.

LightGBM computes exact per-feature contributions for tree models (TreeSHAP, `pred_contrib=True`): for each row,
the contributions plus a bias term add up to the model's raw score, the log-odds of fraud. A positive value
pushes the transaction towards fraud, a negative one away from it.

The model sees category, gender and state as one-hot columns (`category_shopping_net`, ...). `by_source_feature`
adds those columns back up, so every contribution is reported under a feature name from features.py.
"""
import numpy as np
import pandas as pd

import settings  # noqa: F401  (puts ../modelling on the import path)


def contributions(model, feats):
    """Contribution of every encoded model input to the fraud log-odds, one row per transaction.

    Columns: the model's input columns plus 'bias'. Each row sums to the raw score, i.e. logit(probability).
    Only LightGBM models are supported (the main model; see modelling/artifacts/metadata.json).
    """
    if model.name != 'lightgbm':
        raise ValueError(f'contributions are implemented for the LightGBM model, not {model.name!r}')
    X = model.encoder.transform(feats)
    values = model.model.predict(X, num_iteration=model.best_iteration, pred_contrib=True)
    return pd.DataFrame(np.asarray(values), index=feats.index, columns=list(X.columns) + ['bias'])


def by_source_feature(contrib, encoder):
    """Sum one-hot columns back into their categorical feature (e.g. all category_* columns -> 'category')."""
    prefixes = {f'{c}_': c for c in encoder.onehot_}
    groups = {}
    for col in contrib.columns:
        source = next((name for prefix, name in prefixes.items() if col.startswith(prefix)), col)
        groups.setdefault(source, []).append(col)
    return pd.DataFrame({name: contrib[cols].sum(axis=1) for name, cols in groups.items()}, index=contrib.index)


def logit(p):
    p = np.clip(np.asarray(p, float), 1e-15, 1 - 1e-15)
    return np.log(p / (1 - p))
