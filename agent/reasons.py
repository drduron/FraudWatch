"""The human-readable reasons a transaction is fraud (status update: "define 5-8 human-readable reasons ... and
associate with the model's positive predictions").

Each reason owns a group of model features. For a flagged transaction:
  1. its weight is the sum of those features' contributions to the score (explain.py), so reasons are ranked by
     how much they actually moved the model;
  2. its rule checks the feature values (e.g. "night-time purchase" needs is_night = 1), so a reason is only
     reported when it is literally true of the transaction;
  3. its text states the fact with the transaction's own numbers.

Every model feature belongs to at most one reason; the rest (age, city size, location, ...) count as 'other'.
The same rules double as the simple heuristic flags planned for the Week 10 milestone (`applies` alone).
"""
from dataclasses import dataclass
from typing import Callable

CATEGORY_LABELS = {
    'entertainment': 'entertainment', 'food_dining': 'restaurant', 'gas_transport': 'gas / transport',
    'grocery_net': 'online grocery', 'grocery_pos': 'in-store grocery', 'health_fitness': 'health / fitness',
    'home': 'home', 'kids_pets': 'kids / pets', 'misc_net': 'online miscellaneous', 'misc_pos': 'in-store miscellaneous',
    'personal_care': 'personal care', 'shopping_net': 'online shopping', 'shopping_pos': 'in-store shopping',
    'travel': 'travel',
}


def category_label(category):
    return CATEGORY_LABELS.get(category, str(category).replace('_', ' '))


def _has_history(r):
    return r['u_amt_cnt_all'] > 0


def _times(x):
    return f'{x:.1f}x' if x < 10 else f'{x:.0f}x'


# ---- descriptions: one sentence fragment each, built from the transaction's own values ------------------------
def _describe_amount_category(r, ctx):
    return (f"${r['amt']:,.2f} is {_times(r['amt_vs_cat'])} the average {category_label(r['category'])} purchase "
            f"(${r['cat_amt_mean_all']:,.0f})")


def _describe_amount_card(r, ctx):
    text = f"about {_times(r['amt_vs_uall'])} this card's average purchase (${r['u_amt_mean_all']:,.0f})"
    if r['amt_vs_umax'] > 1:
        text += ', and larger than any earlier purchase on the card'
    return text


def _describe_night(r, ctx):
    text = f"made at {int(r['hour']):02d}:00-{int(r['hour']):02d}:59, a night hour"
    if r['u_night_frac_all'] >= 0:
        text += f"; {r['u_night_frac_all']:.0%} of this card's earlier purchases were at night"
    return text


def _describe_rapid(r, ctx):
    parts = []
    if r['u_cnt_1h'] >= 2:
        parts.append(f"{int(r['u_cnt_1h'])} other transactions on this card in the previous hour")
    if r['u_cnt_6h'] >= 3:
        parts.append(f"{int(r['u_cnt_6h'])} in the previous 6 hours")
    if r['today_vs_typical'] >= 3:
        parts.append(f"{int(r['u_cnt_today'])} earlier today, {_times(r['today_vs_typical'])} its typical daily count")
    return '; '.join(parts)


def _recent_spend_high(r):
    return _has_history(r) and r['u_amt_mean_all'] > 0 and r['u_amt_mean_24h'] >= 3 * r['u_amt_mean_all']


def _describe_recent(r, ctx):
    parts = []
    if _recent_spend_high(r):
        parts.append(f"averaged ${r['u_amt_mean_24h']:,.0f} per purchase over the previous 24 hours, "
                     f"{_times(r['u_amt_mean_24h'] / r['u_amt_mean_all'])} its usual ${r['u_amt_mean_all']:,.0f}")
    if r['u_odd_cnt_72h'] >= 1:
        parts.append(f"made {int(r['u_odd_cnt_72h'])} large night-time purchase(s) (over 2x their category "
                     f"average) in the previous 72 hours")
    if r['u_avc_max_72h'] >= 3:
        parts.append(f"made a purchase of {_times(r['u_avc_max_72h'])} its category average in that window")
    if r['last5_night_frac'] >= 0.6:
        parts.append(f"made {r['last5_night_frac']:.0%} of its last 5 purchases at night")
    return 'the card ' + '; '.join(parts) if parts else 'the card behaved unusually just before this'


def _describe_unfamiliar(r, ctx):
    label = category_label(r['category'])
    if r['uc_cnt_all'] == 0:
        text = f"first {label} purchase on this card (after {int(r['u_amt_cnt_all'])} earlier purchases)"
    else:
        text = (f"only {int(r['uc_cnt_all'])} of the card's {int(r['u_amt_cnt_all'])} earlier purchases "
                f"({r['uc_frac_all']:.1%}) were {label}")
    if r['um_cnt_all'] == 0:
        text += '; first visit to this merchant'
    return text


def _describe_category(r, ctx):
    rate, overall = ctx['category_fraud_rate'][r['category']], ctx['overall_fraud_rate']
    return (f"{category_label(r['category'])} purchases are {rate:.1%} fraud in the training data, "
            f"{_times(rate / overall)} the overall rate")


def _describe_new_card(r, ctx):
    if not _has_history(r):
        return 'first transaction ever seen on this card'
    return (f"card first seen {r['acct_age_days']:.0f} days ago, with only {int(r['u_amt_cnt_all'])} "
            f"earlier transactions")


@dataclass(frozen=True)
class Reason:
    code: str
    title: str
    features: tuple
    applies: Callable        # (row, ctx) -> bool
    describe: Callable       # (row, ctx) -> str


LAGS = tuple(f'lag{k}_{s}' for k in (1, 2, 3) for s in ('amt_vs_cat', 'hour', 'amt')) + (
    'lag4_amt_vs_cat', 'lag4_hour', 'lag5_amt_vs_cat', 'lag5_hour', 'lag1_amt_z_m')

REASONS = [
    Reason('AMOUNT_HIGH_FOR_CATEGORY', 'Amount is high for this kind of purchase',
           ('amt', 'amt_vs_cat', 'cat_amt_mean_all', 'amt_vs_segcat', 'amt_z_segcat', 'amt_z_m', 'amt_vs_m_all',
            'm_amt_mean_30', 'm_amt_mean_all', 'm_amt_std_all'),
           lambda r, c: r['amt_vs_cat'] >= 2.5 or r['amt_z_segcat'] >= 3,
           _describe_amount_category),
    Reason('AMOUNT_HIGH_FOR_CARD', "Amount is far above this card's normal spending",
           ('amt_vs_uall', 'amt_vs_u30', 'amt_vs_u24h', 'amt_vs_umax', 'amt_vs_ucmax', 'amt_vs_uc',
            'u_amt_mean_3', 'u_amt_mean_7', 'u_amt_mean_30', 'u_amt_mean_all',
            'job_bp_model', 'job_bp_stat', 'amt_vs_job_bp'),
           lambda r, c: _has_history(r) and (r['amt_vs_uall'] >= 3 or r['amt_vs_umax'] > 1),
           _describe_amount_card),
    Reason('NIGHT_TIME', 'Made late at night',
           ('is_night', 'hour', 'uh_frac_all', 'uhb_frac_all', 'u_night_frac_all'),
           lambda r, c: r['is_night'] == 1,
           _describe_night),
    Reason('RAPID_ACTIVITY', 'Unusually many transactions in a short time',
           ('u_cnt_1h', 'u_cnt_6h', 'u_cnt_24h',
            'u_cnt_today', 'today_vs_typical', 'since_last_s', 'log_since_last', 'hours_since_lag2',
            'hours_since_lag3', 'hours_since_lag4', 'hours_since_lag5', 'u_amt_cnt_3', 'u_amt_cnt_7', 'u_amt_cnt_30'),
           lambda r, c: r['today_vs_typical'] >= 3 or r['u_cnt_6h'] >= 3 or r['u_cnt_1h'] >= 2,
           _describe_rapid),
    Reason('RECENT_SUSPICIOUS_ACTIVITY', 'Card behaved suspiciously just before this',
           ('u_odd_cnt_24h', 'u_odd_cnt_72h', 'u_avc_mean_24h', 'u_avc_mean_72h', 'u_avc_max_24h', 'u_avc_max_72h',
            'u_night_cnt_24h', 'u_night_cnt_72h', 'u_amt_max_24h', 'u_amt_max_72h',
            'u_amt_mean_1h', 'u_amt_mean_6h', 'u_amt_mean_24h', 'u_amt_sum_24h') + LAGS + tuple(
               f'last{n}_{s}' for n in (5, 10) for s in ('night_frac', 'avc_mean', 'avc_max')),
           lambda r, c: (r['u_odd_cnt_72h'] >= 1 or r['u_avc_max_72h'] >= 3 or r['last5_night_frac'] >= 0.6
                         or _recent_spend_high(r)),
           _describe_recent),
    Reason('UNFAMILIAR_PURCHASE', 'Unfamiliar kind of purchase for this card',
           ('uc_cnt_all', 'uc_cnt_30', 'uc_frac_all', 'lag1_same_category', 'lag1_category_code', 'lag2_category_code',
            'lag3_category_code', 'um_cnt_3', 'um_cnt_7', 'um_cnt_30', 'um_cnt_all', 'since_last_merch_s',
            'sim_mean_7', 'sim_max_7', 'sim_mean_30', 'sim_max_30'),
           lambda r, c: _has_history(r) and (r['uc_cnt_all'] == 0 or 0 <= r['uc_frac_all'] < 0.03),
           _describe_unfamiliar),
    Reason('HIGH_RISK_CATEGORY', 'Category often targeted by fraud',
           ('category',),
           lambda r, c: c['category_fraud_rate'].get(r['category'], 0) >= 2 * c['overall_fraud_rate'],
           _describe_category),
    Reason('NEW_CARD', 'New card or little history',
           ('acct_age_days', 'u_amt_cnt_all'),
           lambda r, c: not _has_history(r) or r['u_amt_cnt_all'] < 10 or 0 <= r['acct_age_days'] < 30,
           _describe_new_card),
]
BY_CODE = {r.code: r for r in REASONS}
FEATURE_TO_REASON = {f: r.code for r in REASONS for f in r.features}


def explain_row(row, contrib, ctx, min_weight=0.0):
    """Reasons for one flagged transaction, strongest first.

    row: feature values (mapping); contrib: contribution per source feature (mapping, from explain.py);
    ctx: {'category_fraud_rate': {category: rate}, 'overall_fraud_rate': rate}.
    Returns (reasons, other_weight): reasons that apply and pushed the score up by more than `min_weight`
    log-odds, each {'code', 'title', 'text', 'weight'}; other_weight = total contribution of unassigned features.
    """
    found = []
    for reason in REASONS:
        weight = float(sum(contrib.get(f, 0.0) for f in reason.features))
        if weight > min_weight and reason.applies(row, ctx):
            found.append({'code': reason.code, 'title': reason.title, 'text': reason.describe(row, ctx),
                          'weight': round(weight, 3)})
    other = float(sum(v for f, v in contrib.items() if f not in FEATURE_TO_REASON and f != 'bias'))
    return sorted(found, key=lambda r: -r['weight']), round(other, 3)
