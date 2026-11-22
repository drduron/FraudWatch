"""The investigation tools the agent can call, bound to one case.

All tools are scoped to the case under review (its card and merchant) and to its review time: they can't reach
other cards' histories or anything that happened later. Outputs are compact JSON; card numbers are shown only as
their last four digits.
"""
import json

import numpy as np
import pandas as pd

import settings
from reasons import category_label

MAX_ROWS = 40

TOOL_DEFINITIONS = [
    {
        'name': 'get_card_transactions',
        'description': (
            "List the flagged card's own transactions in the hours before the case transaction (and the case "
            "transaction itself), newest first, with the model's fraud score and flag for each. Use it to check "
            "whether this is part of a burst: fraud on a compromised card usually comes as several transactions "
            "within hours to 3 days. Transactions from before the scored period show score null."),
        'strict': True,
        'input_schema': {
            'type': 'object',
            'properties': {'hours_before': {'type': 'integer', 'description': 'Look-back window in hours, 1-720.'}},
            'required': ['hours_before'],
            'additionalProperties': False,
        },
    },
    {
        'name': 'get_card_profile',
        'description': (
            "The card's long-term profile up to the review time: cardholder (home city, age, job), how long the "
            "card has been seen, typical and maximum amounts, share of night purchases, usual categories, "
            "transactions per day, and any earlier analyst decisions on this card."),
        'strict': True,
        'input_schema': {'type': 'object', 'properties': {}, 'required': [], 'additionalProperties': False},
    },
    {
        'name': 'get_merchant_activity',
        'description': (
            "Activity at the case transaction's merchant across all cards in the hours before the review time: "
            "number of transactions and cards, how many were flagged, and the flagged ones. Several flagged cards "
            "at one merchant can point to a compromised merchant rather than one stolen card."),
        'strict': True,
        'input_schema': {
            'type': 'object',
            'properties': {'hours_before': {'type': 'integer', 'description': 'Look-back window in hours, 1-720.'}},
            'required': ['hours_before'],
            'additionalProperties': False,
        },
    },
]


class ToolError(Exception):
    """A tool call that cannot be answered; returned to the model as an error result."""


class CaseTools:
    def __init__(self, store, case, review_delay_hours=settings.REVIEW_DELAY_HOURS):
        self.store, self.case = store, case
        self.until_ts = case['ts'] + int(review_delay_hours * 3600)
        self.calls = []

    def run(self, name, tool_input):
        """Execute one tool call; returns a JSON string. Raises ToolError for unknown tools or bad input."""
        handler = {'get_card_transactions': self.card_transactions, 'get_card_profile': self.card_profile,
                   'get_merchant_activity': self.merchant_activity}.get(name)
        if handler is None:
            raise ToolError(f'unknown tool {name!r}')
        if not isinstance(tool_input, dict):
            raise ToolError('tool input must be an object')
        result = handler(**tool_input) if tool_input else handler()
        self.calls.append({'tool': name, 'input': tool_input})
        return json.dumps(result, default=_json_default)

    @staticmethod
    def _hours(hours_before):
        if not isinstance(hours_before, int) or isinstance(hours_before, bool) or not 1 <= hours_before <= 720:
            raise ToolError('hours_before must be an integer from 1 to 720')
        return hours_before

    def card_transactions(self, hours_before):
        hours = self._hours(hours_before)
        rows = self.store.card_transactions(self.case['cc_num'], self.until_ts - hours * 3600, self.until_ts)
        rows = rows.iloc[::-1]
        out = [self._txn(r) for r in rows.head(MAX_ROWS).itertuples()]
        flagged = int(rows.fraud_flag.sum())
        case_in_window = bool((rows.trans_num == self.case['trans_num']).any())
        return {
            'card': self.case['card'], 'window_hours': hours, 'review_time': _fmt(self.until_ts),
            'transactions_in_window': int(len(rows)),
            'flagged_in_window': flagged,
            'flagged_other_than_case': flagged - int(case_in_window),
            'total_amount': round(float(rows.amt.sum()), 2),
            'shown': len(out), 'transactions': out,
        }

    def card_profile(self):
        profile = self.store.card_profile(self.case['cc_num'], self.until_ts)
        profile['earlier_analyst_decisions'] = self.store.card_decisions(self.case['cc_num'], self.case['ts'])
        return profile

    def merchant_activity(self, hours_before):
        hours = self._hours(hours_before)
        rows = self.store.merchant_activity(self.case['merchant'], self.until_ts - hours * 3600, self.until_ts)
        flagged = rows[rows.fraud_flag == 1].iloc[::-1]
        return {
            'merchant': self.case['merchant'], 'window_hours': hours,
            'transactions': int(len(rows)), 'cards': int(rows.cc_num.nunique()),
            'flagged': int(len(flagged)), 'flagged_cards': int(flagged.cc_num.nunique()),
            'flagged_transactions': [dict(self._txn(r), card=f'...{str(int(r.cc_num))[-4:]}')
                                     for r in flagged.head(15).itertuples()],
        }

    def _txn(self, r):
        score = None if pd.isna(r.fraud_probability) else round(float(r.fraud_probability), 4)
        return {
            'id': r.trans_num, 'time': r.trans_date_trans_time, 'merchant': r.merchant,
            'category': category_label(r.category), 'amount': round(float(r.amt), 2), 'fraud_score': score,
            'flagged': None if score is None else bool(r.fraud_flag),
            'hours_before_case': round((self.case['ts'] - r.ts) / 3600, 2),
            'is_case_transaction': r.trans_num == self.case['trans_num'],
        }


def _fmt(ts):
    return pd.Timestamp(int(ts), unit='s').strftime('%Y-%m-%d %H:%M:%S')


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    raise TypeError(f'{type(o).__name__} is not JSON serialisable')
