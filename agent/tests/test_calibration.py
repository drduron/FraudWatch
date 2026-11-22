import pandas as pd

import calibration as cal

H = 3600


def test_band_label():
    assert cal.band_label(0.184, 0.184) == '0.18-0.50'
    assert cal.band_label(0.5, 0.184) == '0.50-0.90'
    assert cal.band_label(0.995, 0.184) == '0.99-1.00'
    assert cal.band_label(1.0, 0.184) == '0.99-1.00'


def test_prior_flag_counts_counts_earlier_flags_on_the_same_card_only():
    scored = pd.DataFrame({
        'trans_num': ['a1', 'a2', 'a3', 'a4', 'b1', 'a0'],
        'cc_num':    [1,    1,    1,    1,    2,    1],
        'ts':        [0,    1 * H, 2 * H, 100 * H, 1 * H, 0],
        'fraud_flag': [1,   1,    1,    1,    1,    0],      # a0 not flagged
    })
    c = cal.prior_flag_counts(scored)
    assert c.to_dict() == {'a1': 0, 'a2': 1, 'a3': 2, 'a4': 0, 'b1': 0}   # a4 is > 72 h after the others


def test_card_compromised_looks_back_72h_including_the_row():
    rows = pd.DataFrame({'trans_num': ['x', 'y', 'z'], 'cc_num': [1, 1, 2], 'ts': [10 * H, 200 * H, 10 * H]})
    labelled = pd.DataFrame({'cc_num': [1, 1, 2], 'ts': [0, 200 * H, 11 * H], 'is_fraud': [1, 0, 1]})
    c = cal.card_compromised(rows, labelled)
    assert c.to_dict() == {'x': 1, 'y': 0, 'z': 0}       # z: the fraud is after it


def test_build_table_and_lookup():
    flagged = pd.DataFrame({'fraud_probability': [0.3, 0.3, 0.3, 0.999], 'other_flags': [0, 0, 1, 3],
                            'is_fraud': [1, 0, 1, 1], 'compromised': [1, 0, 1, 1]})
    table = cal.build_table(flagged, 0.184)
    cell = cal.lookup(table, 0.25, 0, 0.184)
    assert cell == {'score_band': '0.18-0.50', 'other_flags_on_card_72h': False, 'cases': 2, 'fraud_rate': 0.5,
                    'card_compromised_rate': 0.5}
    assert cal.lookup(table, 0.95, 0, 0.184) is None
