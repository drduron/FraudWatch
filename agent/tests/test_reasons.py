import numpy as np
import pandas as pd
import pytest

import settings  # noqa: F401
import reasons
from features import CATEGORICAL_FEATURES, NUMERIC_FEATURES

CTX = {'category_fraud_rate': {'shopping_net': 0.017, 'food_dining': 0.002}, 'overall_fraud_rate': 0.0058}


def row(**values):
    """A transaction with an ordinary card history; override values to trigger reasons."""
    base = {f: 0.0 for f in NUMERIC_FEATURES}
    base.update(category='food_dining', amt=20.0, hour=14, u_amt_cnt_all=500, u_amt_mean_all=60.0,
                u_amt_mean_24h=50.0, u_amt_cnt_30=60, acct_age_days=400, amt_vs_cat=0.5, amt_vs_uall=0.4,
                amt_vs_umax=0.1, uc_cnt_all=50, uc_frac_all=0.1, um_cnt_all=3, u_night_frac_all=0.1,
                cat_amt_mean_all=40.0, last5_night_frac=0.0)
    base.update(values)
    return base


def test_every_feature_belongs_to_at_most_one_reason():
    allf = [f for r in reasons.REASONS for f in r.features]
    assert set(allf) <= set(NUMERIC_FEATURES + CATEGORICAL_FEATURES)
    assert len(allf) == len(set(allf))
    assert 5 <= len(reasons.REASONS) <= 8          # status update: 5-8 human-readable reasons


def test_ordinary_transaction_has_no_reasons():
    contrib = {f: 0.5 for f in NUMERIC_FEATURES}      # even if the model pushed up, the facts must hold
    found, _ = reasons.explain_row(row(), contrib, CTX)
    assert found == []


def test_reasons_are_true_ranked_and_quantified():
    r = row(category='shopping_net', amt=900.0, amt_vs_cat=9.0, cat_amt_mean_all=100.0, is_night=1, hour=23,
            u_odd_cnt_72h=3, u_avc_max_72h=12.0)
    contrib = {'amt_vs_cat': 2.0, 'is_night': 1.0, 'u_odd_cnt_72h': 4.0, 'category': 0.5, 'age': 0.3}
    found, other = reasons.explain_row(r, contrib, CTX)
    assert [f['code'] for f in found] == ['RECENT_SUSPICIOUS_ACTIVITY', 'AMOUNT_HIGH_FOR_CATEGORY', 'NIGHT_TIME',
                                          'HIGH_RISK_CATEGORY']
    assert '$900.00 is 9.0x the average online shopping purchase ($100)' in found[1]['text']
    assert 'made 3 large night-time purchase(s)' in found[0]['text']
    assert other == pytest.approx(0.3)


def test_reason_with_negative_weight_is_not_reported():
    r = row(is_night=1, hour=23)
    found, _ = reasons.explain_row(r, {'is_night': -0.4}, CTX)
    assert found == []


def test_new_card_without_history():
    r = row(u_amt_cnt_all=0, acct_age_days=-1, u_amt_mean_all=-1)
    found, _ = reasons.explain_row(r, {'u_amt_cnt_all': 1.0, 'amt_vs_uall': 1.0}, CTX)
    assert [f['code'] for f in found] == ['NEW_CARD']            # no "above the card's normal" without history
    assert found[0]['text'] == 'first transaction ever seen on this card'


def test_contributions_add_up_to_the_saved_model_score():
    """explain.contributions on the real saved model: per-row contributions sum to the log-odds."""
    from models import load_model
    import explain
    path = settings.MODELLING_DIR / 'artifacts' / 'lightgbm'
    if not (path / 'model.txt').exists():
        pytest.skip('saved LightGBM model not available')
    model = load_model(path)
    rng = np.random.default_rng(0)
    df = pd.DataFrame(rng.random((40, len(NUMERIC_FEATURES))) * 5, columns=NUMERIC_FEATURES)
    for c in model.encoder.onehot_:
        df[c] = rng.choice(model.encoder.levels_[c], len(df))
    df['merchant'] = 'x'
    contrib = explain.contributions(model, df)
    p = model.predict_proba(df)
    assert np.allclose(1 / (1 + np.exp(-contrib.sum(axis=1).to_numpy())), p, atol=1e-12)
    grouped = explain.by_source_feature(contrib, model.encoder)
    assert set(grouped.columns) == set(NUMERIC_FEATURES) | {'category', 'gender', 'state', 'bias'}
    assert np.allclose(grouped.sum(axis=1), contrib.sum(axis=1))
