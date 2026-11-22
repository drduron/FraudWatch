"""How good are the agent's recommendations? Compares them with the true labels of the flagged transactions.

    python evaluate.py --mode rules --all              # rule-based baseline on every flagged case (free, seconds)
    python evaluate.py --mode claude --n 20            # Claude on 20 cases: 10 real fraud + 10 false alarms
    python evaluate.py --mode claude --n 20 --natural  # 20 random cases at the queue's natural mix (~97% fraud)

The labels come from data/labels.parquet, which only this script reads: the agent never sees them.

    python evaluate.py --rescore results/eval_claude.jsonl  # recompute the scores of a saved run (no API calls)

What counts as right, scored two ways:
  transaction level (strict): was THIS transaction fraud?
    real fraud    -> block_card is best, contact_customer is acceptable, dismiss is a MISSED FRAUD
    false alarm   -> dismiss is best, contact_customer is acceptable, block_card is a WRONGLY BLOCKED customer
  card level: the actions act on the card, so was the CARD compromised at review time, i.e. did it have a
    confirmed fraud transaction in the 72 hours up to and including the case? A legitimate purchase in the
    middle of a real fraud burst is a false alarm at transaction level, but blocking the card is right.
    (Only labels of the scored file are used, so a burst that started in the history period counts only from
    the first scored transaction.)
Because 97% of flagged cases are fraud, the default sample is balanced (half false alarms): a random sample
would contain almost no false alarms, and spotting them is where the agent adds most value.
Nothing is written to the audit log; reports go to results/eval_<mode>.jsonl.
"""
import argparse
import hashlib
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

import settings
from agent import AgentError, AgentSetupError, Investigator
from store import LocalStore

OUTCOME = {  # (is_fraud, action) -> outcome
    (1, 'block_card'): 'best', (1, 'contact_customer'): 'acceptable', (1, 'dismiss'): 'missed fraud',
    (0, 'dismiss'): 'best', (0, 'contact_customer'): 'acceptable', (0, 'block_card'): 'wrongly blocked',
}


def split_of(trans_num):
    """Fixed dev / test split of the cases (by a hash of the transaction id): tune on dev, report on test."""
    return 'dev' if int(hashlib.md5(trans_num.encode()).hexdigest(), 16) % 2 == 0 else 'test'


def pick_cases(store, labels, n, natural, seed, split=None, exclude=()):
    cases = pd.DataFrame(store.queue(include_decided=True)).merge(labels, on='trans_num', how='left')
    if cases.is_fraud.isna().any():
        raise ValueError('some cases have no label in labels.parquet; rebuild with build_cases.py')
    if split:
        cases = cases[cases.trans_num.map(split_of) == split]
    cases = cases[~cases.trans_num.isin(set(exclude))]
    if n is None:
        return cases
    if natural:
        return cases.sample(n=min(n, len(cases)), random_state=seed)
    fa, fraud = cases[cases.is_fraud == 0], cases[cases.is_fraud == 1]
    k = min(n // 2, len(fa))
    return pd.concat([fa.sample(n=k, random_state=seed), fraud.sample(n=n - k, random_state=seed)])


def card_compromised(data_dir, labels, hours=72):
    """trans_num -> 1 if the card had a labelled fraud transaction in the `hours` up to and including it."""
    txns = pd.read_parquet(data_dir / 'transactions.parquet', columns=['trans_num', 'cc_num', 'ts'])
    fraud = txns.merge(labels[labels.is_fraud == 1], on='trans_num')[['cc_num', 'ts']]
    fraud_ts = {c: g.ts.to_numpy() for c, g in fraud.groupby('cc_num')}
    cases = txns.merge(labels, on='trans_num')

    def compromised(row):
        ts = fraud_ts.get(row.cc_num)
        return int(ts is not None and ((ts >= row.ts - hours * 3600) & (ts <= row.ts)).any())
    return {r.trans_num: compromised(r) for r in cases.itertuples()}


def _table(df, truth_col, names, outcome_col):
    lines = []
    for truth, name in ((1, names[0]), (0, names[1])):
        part = df[df[truth_col] == truth]
        if part.empty:
            continue
        counts = part.recommended_action.value_counts()
        lines.append(f'  {name:<16} n={len(part):<5} ' + '  '.join(
            f'{a}: {counts.get(a, 0):>4} ({counts.get(a, 0) / len(part):.0%})' for a in settings.ACTIONS))
    out = df[outcome_col].value_counts()
    lines.append('  outcomes: ' + ', '.join(f'{k} {out.get(k, 0)}' for k in
                                            ('best', 'acceptable', 'missed fraud', 'wrongly blocked')))
    lines.append(f'  best-action rate {out.get("best", 0) / len(df):.1%} | no harmful outcome '
                 f'{(out.get("best", 0) + out.get("acceptable", 0)) / len(df):.1%}')
    return lines


def summarise(results):
    df = pd.DataFrame(results)
    return '\n'.join(['transaction level (was this transaction fraud?):']
                     + _table(df, 'is_fraud', ('real fraud', 'false alarm'), 'outcome')
                     + ['card level (was the card compromised at review time?):']
                     + _table(df, 'card_compromised', ('compromised', 'clean card'), 'card_outcome'))


def score(report, is_fraud, compromised):
    report.update(is_fraud=is_fraud, outcome=OUTCOME[(is_fraud, report['recommended_action'])],
                  card_compromised=compromised, card_outcome=OUTCOME[(compromised, report['recommended_action'])])
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--mode', choices=['rules', 'claude'], default='rules')
    g = ap.add_mutually_exclusive_group()
    g.add_argument('--n', type=int, help='number of cases (default: 20 for claude)')
    g.add_argument('--all', action='store_true', help='every flagged case')
    ap.add_argument('--natural', action='store_true', help="sample at the queue's natural fraud rate, not balanced")
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--effort', default=settings.EFFORT)
    ap.add_argument('--workers', type=int, default=4, help='cases investigated in parallel (claude mode)')
    ap.add_argument('--data-dir', type=Path, default=settings.DATA_DIR)
    ap.add_argument('--results-dir', type=Path, default=settings.RESULTS_DIR)
    ap.add_argument('--rescore', type=Path, help='recompute the scores of a saved eval_*.jsonl (no agent runs)')
    ap.add_argument('--split', choices=['dev', 'test'], help='only cases of this half (tune on dev, report on test)')
    ap.add_argument('--exclude', type=Path, nargs='*', default=[],
                    help='saved eval_*.jsonl file(s) whose cases are left out (e.g. cases already inspected)')
    ap.add_argument('--tag', default='', help='suffix for the output file: results/eval_<mode><tag>.jsonl')
    args = ap.parse_args()
    n = None if args.all else (args.n or (20 if args.mode == 'claude' else None))

    labels = pd.read_parquet(args.data_dir / 'labels.parquet')
    compromised = card_compromised(args.data_dir, labels)
    if args.rescore:
        truth = labels.set_index('trans_num').is_fraud.astype(int)
        with open(args.rescore) as f:
            results = [score(r, int(truth[r['trans_num']]), compromised[r['trans_num']])
                       for r in map(json.loads, f) if r]
        print(f'{args.rescore}: {len(results)} report(s)\n{summarise(results)}')
        return

    store = LocalStore(args.data_dir)
    exclude = [json.loads(line)['trans_num'] for p in args.exclude for line in open(p) if line.strip()]
    cases = pick_cases(store, labels, n, args.natural, args.seed, args.split, exclude)
    print(f'{args.mode}: {len(cases):,} case(s), {int(cases.is_fraud.sum()):,} real fraud / '
          f'{int((cases.is_fraud == 0).sum()):,} false alarms', flush=True)
    investigator = Investigator(store, mode=args.mode, effort=args.effort)
    truth = dict(zip(cases.trans_num, cases.is_fraud.astype(int)))

    def run(trans_num):
        import anthropic
        try:
            return investigator.investigate(store.case(trans_num)), None
        except (AgentError, anthropic.APIError) as e:     # e.g. rate limit, server error, no credit left
            return None, f'{trans_num}: {type(e).__name__}: {e}'

    t0 = time.time()
    results, errors = [], []
    workers = args.workers if args.mode == 'claude' else 1
    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for i, (report, err) in enumerate(pool.map(run, cases.trans_num), 1):
                if err:
                    errors.append(err)
                    print(f'  error {err}', file=sys.stderr)
                    continue
                y = truth[report['trans_num']]
                results.append(score(report, y, compromised[report['trans_num']]))
                if args.mode == 'claude':
                    print(f"  [{i}/{len(cases)}] {'fraud' if y else 'false alarm':<11} -> "
                          f"{report['recommended_action']:<16} {report['outcome']}", flush=True)
    except AgentSetupError as e:
        sys.exit(f'cannot call the Claude API: {e}\nSet ANTHROPIC_API_KEY, or use --mode rules')
    if not results:
        sys.exit(f'no reports produced ({len(errors)} error(s); first: {errors[0] if errors else "-"})')

    args.results_dir.mkdir(parents=True, exist_ok=True)
    out = args.results_dir / f'eval_{args.mode}{args.tag}.jsonl'
    with open(out, 'w') as f:
        for r in results:
            f.write(json.dumps(r) + '\n')
    print(summarise(results))
    footer = f'{len(results)} report(s), {len(errors)} error(s), {time.time() - t0:.0f}s -> {out}'
    if args.mode == 'claude':
        footer += f" | est. cost ${sum(r['est_cost_usd'] for r in results):.2f}"
    print(footer)


if __name__ == '__main__':
    main()
