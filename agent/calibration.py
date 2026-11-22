"""How often flagged cases like this one turned out to be fraud, measured on the validation period.

The model's score alone is a strong but uneven signal: a flagged transaction at 0.999 is almost always fraud, one
just over the threshold is close to a coin flip. Whether the card already had other flagged transactions in the
previous 72 hours matters too: a low-scoring purchase on a card that is in a fraud burst means the card is
compromised even when that one purchase is legitimate.

The table is built from the validation period (the last 15% of the training file, never used to fit the model),
so no test labels are involved. For every flagged validation transaction it records the score band, whether the
card had other flagged transactions in the 72 hours before, whether the transaction was fraud, and whether the
card was compromised (any fraud on the card in the 72 hours up to and including it).
"""
import numpy as np
import pandas as pd

WINDOW_HOURS = 72


def band_edges(threshold):
    return [threshold, 0.5, 0.9, 0.99, 1.0]


def band_label(score, threshold):
    edges = band_edges(threshold)
    for lo, hi in zip(edges[:-1], edges[1:]):
        if score < hi or hi == 1.0:
            return f'{lo:.2f}-{hi:.2f}'
    raise ValueError(score)


def _earlier_in_window(ts, event_ts, hours, inclusive):
    """For each time in ts: how many event times fall in [t - hours, t) (or [t - hours, t] if inclusive)."""
    lo = np.searchsorted(event_ts, ts - hours * 3600, side='left')
    hi = np.searchsorted(event_ts, ts, side='right' if inclusive else 'left')
    return hi - lo


def prior_flag_counts(scored, hours=WINDOW_HOURS):
    """For each flagged row of `scored` (cc_num, ts, trans_num, fraud_flag): the number of OTHER flagged rows on
    the same card in the `hours` before it (same-second rows count as earlier). Returns a Series by trans_num."""
    flagged = scored[scored.fraud_flag == 1].sort_values(['cc_num', 'ts', 'trans_num'])
    out = []
    for _, g in flagged.groupby('cc_num', sort=False):
        ts = g.ts.to_numpy()
        out.append(pd.Series(_earlier_in_window(ts, ts, hours, inclusive=True) - 1, index=g.trans_num.to_numpy()))
    return pd.concat(out) if out else pd.Series(dtype=int)


def card_compromised(rows, labelled, hours=WINDOW_HOURS):
    """For each row of `rows` (cc_num, ts, trans_num): 1 if `labelled` (cc_num, ts, is_fraud) has a fraud on the
    same card in the `hours` up to and including the row's time. Returns a Series by trans_num."""
    fraud = labelled[labelled.is_fraud == 1]
    fraud_ts = {c: np.sort(g.ts.to_numpy()) for c, g in fraud.groupby('cc_num')}
    out = {}
    for card, g in rows.groupby('cc_num'):
        ts = g.ts.to_numpy()
        f = fraud_ts.get(card)
        hits = _earlier_in_window(ts, f, hours, inclusive=True) if f is not None else np.zeros(len(ts), int)
        out.update(zip(g.trans_num, (hits > 0).astype(int)))
    return pd.Series(out)


def build_table(flagged, threshold):
    """flagged: flagged validation rows with fraud_probability, other_flags, is_fraud, compromised.
    Returns one cell per (score band, other flags yes/no) with counts and rates."""
    df = flagged.assign(band=[band_label(s, threshold) for s in flagged.fraud_probability],
                        others=flagged.other_flags > 0)
    table = []
    for (band, others), g in df.groupby(['band', 'others'], sort=True):
        table.append({'score_band': band, 'other_flags_on_card_72h': bool(others), 'cases': int(len(g)),
                      'fraud_rate': round(float(g.is_fraud.mean()), 3),
                      'card_compromised_rate': round(float(g.compromised.mean()), 3)})
    return table


def lookup(table, score, other_flags, threshold):
    """The table cell for a case, or None if validation had no similar case."""
    band, others = band_label(score, threshold), other_flags > 0
    return next((c for c in table if c['score_band'] == band and c['other_flags_on_card_72h'] == others), None)
