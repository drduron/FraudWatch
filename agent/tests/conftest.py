"""A tiny synthetic data folder in the format build_cases.py writes, so tests run without the real dataset."""
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # agent/

T0 = 1_600_000_000           # 2020-09-13 12:26:40 UTC
HOUR = 3600
CARD_A, CARD_B = 4111111111111111, 4222222222222222


def _txn(trans_num, cc_num, ts, amt, category='grocery_pos', merchant='Shop A', score=None, flag=0, source='input'):
    return {'trans_num': trans_num, 'cc_num': cc_num, 'ts': ts,
            'trans_date_trans_time': pd.Timestamp(ts, unit='s').strftime('%Y-%m-%d %H:%M:%S'),
            'merchant': merchant, 'category': category, 'amt': amt, 'dist_um_km': 50.0, 'source': source,
            'fraud_probability': score, 'fraud_flag': flag}


CALIBRATION = [  # a small table in the format calibration.build_table writes
    {'score_band': '0.18-0.50', 'other_flags_on_card_72h': False, 'cases': 17, 'fraud_rate': 0.53,
     'card_compromised_rate': 0.53},
    {'score_band': '0.90-0.99', 'other_flags_on_card_72h': True, 'cases': 30, 'fraud_rate': 1.0,
     'card_compromised_rate': 1.0},
]


def _case(t, reasons=(), other_flags=0, cell=None):
    return {'other_flags_72h': other_flags, 'calibration': cell,
            'trans_num': t['trans_num'], 'cc_num': t['cc_num'], 'card': f"...{str(t['cc_num'])[-4:]}",
            'time': t['trans_date_trans_time'], 'ts': t['ts'], 'merchant': t['merchant'], 'category': t['category'],
            'category_label': t['category'], 'amount': t['amt'], 'fraud_probability': t['fraud_probability'],
            'threshold': 0.184, 'model': 'lightgbm', 'feature_version': 'v2.1', 'reasons': list(reasons),
            'other_weight': 0.0, 'top_features': [{'feature': 'amt', 'value': t['amt'], 'contribution': 2.0,
                                                   'reason': 'AMOUNT_HIGH_FOR_CATEGORY'}]}


@pytest.fixture
def data_dir(tmp_path):
    rows = []
    # card A: 30 days of normal history (unscored), then a burst of 3 flagged transactions, then a later one
    for d in range(30):
        rows.append(_txn(f'a_hist{d}', CARD_A, T0 - (30 - d) * 24 * HOUR, 40.0 + d, source='history'))
    rows += [_txn('a_burst1', CARD_A, T0 - 2 * HOUR, 900.0, 'shopping_net', 'Shop X', 0.99, 1),
             _txn('a_burst2', CARD_A, T0 - 1 * HOUR, 800.0, 'shopping_net', 'Shop X', 0.98, 1),
             _txn('a_case', CARD_A, T0, 700.0, 'misc_net', 'Shop X', 0.97, 1),
             _txn('a_later', CARD_A, T0 + 5 * HOUR, 650.0, 'misc_net', 'Shop Y', 0.95, 1)]
    # card B: normal history and one isolated flagged purchase at Shop X
    for d in range(10):
        rows.append(_txn(f'b_hist{d}', CARD_B, T0 - (10 - d) * 24 * HOUR, 20.0, 'food_dining', 'Cafe', 0.01, 0))
    rows.append(_txn('b_case', CARD_B, T0 + HOUR, 300.0, 'shopping_pos', 'Shop X', 0.30, 1))
    txns = pd.DataFrame(rows)
    txns.to_parquet(tmp_path / 'transactions.parquet')
    pd.DataFrame([
        {'cc_num': CARD_A, 'gender': 'F', 'age_first_seen': 40.0, 'job': 'Teacher', 'city': 'Springfield',
         'state': 'IL', 'city_pop': 100000, 'first_seen_ts': T0 - 30 * 24 * HOUR},
        {'cc_num': CARD_B, 'gender': 'M', 'age_first_seen': 30.0, 'job': 'Chef', 'city': 'Austin', 'state': 'TX',
         'city_pop': 900000, 'first_seen_ts': T0 - 10 * 24 * HOUR},
    ]).to_parquet(tmp_path / 'cards.parquet')
    by_id = {r['trans_num']: r for r in rows}
    reason = {'code': 'RECENT_SUSPICIOUS_ACTIVITY', 'title': 'Card behaved suspiciously just before this',
              'text': 'the card made 2 large purchases', 'weight': 5.0}
    cases = ([_case(by_id[t], [reason]) for t in ('a_burst1', 'a_burst2', 'a_later')]
             + [_case(by_id['a_case'], [reason], 2, CALIBRATION[1]), _case(by_id['b_case'], (), 0, CALIBRATION[0])])
    cases.sort(key=lambda c: c['ts'])
    with open(tmp_path / 'queue.jsonl', 'w') as f:
        for c in cases:
            f.write(json.dumps(c) + '\n')
    (tmp_path / 'context.json').write_text(json.dumps(
        {'category_fraud_rate': {'shopping_net': 0.017, 'grocery_pos': 0.014, 'food_dining': 0.002},
         'overall_fraud_rate': 0.0058}))
    pd.DataFrame({'trans_num': [c['trans_num'] for c in cases],
                  'is_fraud': [1, 1, 1, 1, 0]}).to_parquet(tmp_path / 'labels.parquet')
    return tmp_path


@pytest.fixture
def store(data_dir):
    from store import LocalStore
    return LocalStore(data_dir)
