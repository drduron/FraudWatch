"""Data access for the agent: the case queue, transaction history and the audit log.

Everything the agent reads or writes goes through `LocalStore`, so moving to AWS means writing one class with
the same methods on top of DynamoDB / S3 (status update, section 4). The local version reads the files
build_cases.py writes and keeps the audit log in SQLite.

Time limit. Every history query takes an `until_ts` and never returns a transaction after it. The agent passes
the case's review time (flag time + REVIEW_DELAY_HOURS), so it only sees what was known when it reviewed the case.

No labels. Nothing here reads labels.parquet; the agent never sees whether a transaction was really fraud.
Confirmed labels only enter through analysts' decisions (`record_decision`).
"""
import json
import sqlite3
import time
from pathlib import Path

import numpy as np
import pandas as pd

import settings

NIGHT_HOURS = {22, 23, 0, 1, 2, 3}

SCHEMA = """
CREATE TABLE IF NOT EXISTS reports (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    trans_num TEXT NOT NULL, cc_num INTEGER NOT NULL, created_at TEXT NOT NULL,
    mode TEXT NOT NULL, recommended_action TEXT NOT NULL, report_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS decisions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    trans_num TEXT NOT NULL, cc_num INTEGER NOT NULL, decided_at TEXT NOT NULL, analyst TEXT NOT NULL,
    label TEXT NOT NULL CHECK (label IN ('true_fraud', 'false_alarm')),
    action TEXT NOT NULL CHECK (action IN ('block_card', 'contact_customer', 'dismiss')),
    agent_action TEXT, notes TEXT
);
CREATE INDEX IF NOT EXISTS reports_card ON reports (cc_num);
CREATE INDEX IF NOT EXISTS decisions_card ON decisions (cc_num);
"""


def now_iso():
    return time.strftime('%Y-%m-%dT%H:%M:%S')


class LocalStore:
    def __init__(self, data_dir=settings.DATA_DIR, audit_db=None):
        data_dir = Path(data_dir)
        if not (data_dir / 'queue.jsonl').exists():
            raise FileNotFoundError(f'no cases in {data_dir}: run `python build_cases.py` first')
        with open(data_dir / 'queue.jsonl') as f:
            self._queue = [json.loads(line) for line in f if line.strip()]
        self._cases = {c['trans_num']: c for c in self._queue}
        self.context = json.loads((data_dir / 'context.json').read_text())
        txns = pd.read_parquet(data_dir / 'transactions.parquet')
        txns = txns.sort_values(['cc_num', 'ts', 'trans_num']).reset_index(drop=True)
        self._txns = txns
        # row range per card (rows are sorted by card), and a time-sorted view for merchant queries
        cards = txns.cc_num.to_numpy()
        starts = np.flatnonzero(np.r_[True, cards[1:] != cards[:-1]])
        ends = np.r_[starts[1:], len(cards)]
        self._card_rows = {int(cards[s]): (int(s), int(e)) for s, e in zip(starts, ends)}
        self._by_merchant = {m: g.sort_values('ts') for m, g in txns.groupby('merchant', sort=False)}
        self._cards = pd.read_parquet(data_dir / 'cards.parquet').set_index('cc_num')
        self._db_path = Path(audit_db) if audit_db else data_dir / settings.AUDIT_DB
        with self._db() as db:
            db.executescript(SCHEMA)

    # ---- cases ----------------------------------------------------------------------------------------------
    def queue(self, include_decided=False):
        """Flagged cases in time order; by default only those without an analyst decision yet."""
        if include_decided:
            return list(self._queue)
        decided = self.decided_ids()
        return [c for c in self._queue if c['trans_num'] not in decided]

    def case(self, trans_num):
        if trans_num not in self._cases:
            raise KeyError(f'no flagged case for transaction {trans_num!r}')
        return self._cases[trans_num]

    # ---- transaction history (never after until_ts) ---------------------------------------------------------
    def card_transactions(self, cc_num, since_ts, until_ts):
        """The card's transactions with since_ts <= ts <= until_ts, oldest first."""
        s, e = self._card_rows.get(int(cc_num), (0, 0))
        rows = self._txns.iloc[s:e]
        return rows[(rows.ts >= since_ts) & (rows.ts <= until_ts)]

    def card_profile(self, cc_num, until_ts):
        """Long-term profile of the card from its transactions up to until_ts."""
        hist = self.card_transactions(cc_num, -np.inf, until_ts)
        card = self._cards.loc[int(cc_num)] if int(cc_num) in self._cards.index else None
        profile = {'card': f'...{str(int(cc_num))[-4:]}', 'transactions_seen': int(len(hist))}
        if card is not None:
            years = (until_ts - card.first_seen_ts) / (365.25 * 86400)
            profile.update({'home': f'{card.city}, {card.state}', 'city_population': int(card.city_pop),
                            'cardholder_age': int(card.age_first_seen + years), 'gender': card.gender,
                            'job': card.job})
        if hist.empty:
            return profile
        amt, hours = hist.amt, pd.to_datetime(hist.ts, unit='s').dt.hour
        last30 = hist[hist.ts > until_ts - 30 * 86400]
        cats = hist.category.value_counts(normalize=True).head(5)
        profile.update({
            'first_seen': _fmt_ts(hist.ts.iloc[0]),
            'days_of_history': round((until_ts - hist.ts.iloc[0]) / 86400, 1),
            'amount_mean': round(float(amt.mean()), 2), 'amount_median': round(float(amt.median()), 2),
            'amount_p95': round(float(amt.quantile(0.95)), 2), 'amount_max': round(float(amt.max()), 2),
            'night_share': round(float(hours.isin(NIGHT_HOURS).mean()), 3),
            'transactions_per_day_last_30d': round(len(last30) / 30, 2),
            'top_categories': {c: round(float(v), 3) for c, v in cats.items()},
            'merchants_used': int(hist.merchant.nunique()),
        })
        return profile

    def merchant_activity(self, merchant, since_ts, until_ts):
        """All cards' transactions at this merchant with since_ts <= ts <= until_ts, oldest first."""
        rows = self._by_merchant.get(merchant)
        if rows is None:
            return self._txns.iloc[:0]
        return rows[(rows.ts >= since_ts) & (rows.ts <= until_ts)]

    # ---- audit log ------------------------------------------------------------------------------------------
    def _db(self):
        return sqlite3.connect(self._db_path)

    def save_report(self, report):
        with self._db() as db:
            cur = db.execute('INSERT INTO reports (trans_num, cc_num, created_at, mode, recommended_action, '
                             'report_json) VALUES (?, ?, ?, ?, ?, ?)',
                             (report['trans_num'], report['cc_num'], report['created_at'], report['mode'],
                              report['recommended_action'], json.dumps(report)))
            return cur.lastrowid

    def latest_report(self, trans_num):
        with self._db() as db:
            row = db.execute('SELECT report_json FROM reports WHERE trans_num = ? ORDER BY id DESC LIMIT 1',
                             (trans_num,)).fetchone()
        return json.loads(row[0]) if row else None

    def record_decision(self, trans_num, label, action, analyst, notes=''):
        """The analyst's final call on a case (the agent only recommends)."""
        if label not in settings.LABELS:
            raise ValueError(f'label must be one of {settings.LABELS}')
        if action not in settings.ACTIONS:
            raise ValueError(f'action must be one of {settings.ACTIONS}')
        case = self.case(trans_num)
        report = self.latest_report(trans_num)
        with self._db() as db:
            db.execute('INSERT INTO decisions (trans_num, cc_num, decided_at, analyst, label, action, agent_action, '
                       'notes) VALUES (?, ?, ?, ?, ?, ?, ?, ?)',
                       (trans_num, case['cc_num'], now_iso(), analyst, label, action,
                        report['recommended_action'] if report else None, notes))

    def decided_ids(self):
        with self._db() as db:
            return {r[0] for r in db.execute('SELECT DISTINCT trans_num FROM decisions')}

    def card_decisions(self, cc_num, before_ts):
        """Earlier analyst decisions on this card, for cases flagged before `before_ts`."""
        with self._db() as db:
            rows = db.execute('SELECT trans_num, decided_at, label, action, notes FROM decisions WHERE cc_num = ? '
                              'ORDER BY id', (int(cc_num),)).fetchall()
        out = []
        for trans_num, decided_at, label, action, notes in rows:
            case = self._cases.get(trans_num)
            if case and case['ts'] < before_ts:
                out.append({'transaction_time': case['time'], 'amount': case['amount'],
                            'merchant': case['merchant'], 'label': label, 'action': action, 'notes': notes or ''})
        return out

    def confirmed_labels(self):
        """The latest analyst label per transaction, as is_fraud (0/1), for retraining the model."""
        with self._db() as db:
            rows = db.execute('SELECT trans_num, label, analyst, decided_at FROM decisions ORDER BY id').fetchall()
        df = pd.DataFrame(rows, columns=['trans_num', 'label', 'analyst', 'decided_at'])
        df = df.drop_duplicates('trans_num', keep='last')
        df['is_fraud'] = (df.label == 'true_fraud').astype(int)
        return df[['trans_num', 'is_fraud', 'analyst', 'decided_at']].reset_index(drop=True)


def _fmt_ts(ts):
    return pd.Timestamp(int(ts), unit='s').strftime('%Y-%m-%d %H:%M:%S')
