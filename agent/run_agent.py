"""Command line for the analyst agent.

    python run_agent.py queue                        # flagged cases still waiting for a decision
    python run_agent.py investigate --next 3         # the agent investigates the next 3 cases (Claude)
    python run_agent.py investigate --txn <id>       # one specific case
    python run_agent.py investigate --next 3 --rules # rule-based reports, no API key needed
    python run_agent.py show --txn <id>              # the latest saved report for a case
    python run_agent.py decide --txn <id> --label true_fraud --action block_card --analyst <name>
    python run_agent.py export-labels                # confirmed labels for retraining the model

Claude mode needs Anthropic credentials (ANTHROPIC_API_KEY). Every report is saved to the audit log
(data/audit.db) and appended to results/reports.jsonl.
"""
import argparse
import json
import sys
import textwrap

import settings
from agent import AgentError, AgentSetupError, Investigator
from store import LocalStore


def print_report(r):
    width = 100
    print('=' * width)
    print(f"{r['summary']}")
    print(f"txn {r['trans_num']} | card {r['card']} | {r['time']} | ${r['amount']:,.2f} {r['category']} at "
          f"{r['merchant']} | score {r['fraud_probability']:.3f} (threshold {r['threshold']:.3f})")
    print('-' * width)
    print(f"RECOMMENDATION: {r['recommended_action'].upper()}  (confidence: {r['confidence']}"
          f"{', burst detected' if r['burst_detected'] else ''})")
    for title, body in (('Why flagged', r['why_flagged']), ('Reasoning', r['reasoning']),
                        ('Counter-evidence', r['counter_evidence'])):
        print(textwrap.fill(f'{title}: {body}', width, subsequent_indent='  '))
    print('Findings:')
    for f in r['investigation_findings']:
        print(textwrap.fill(f'- {f}', width, subsequent_indent='  '))
    if r['related_transaction_ids']:
        print(f"Related transactions: {', '.join(r['related_transaction_ids'])}")
    meta = f"mode {r['mode']}, {len(r['tool_calls'])} tool calls, {r['elapsed_s']}s"
    if r['mode'] == 'claude':
        meta += f", model {r['model']}, effort {r['effort']}, {r['turns']} turns, ~${r['est_cost_usd']:.3f}"
    print(f'[{meta}]')


def cmd_queue(store, args):
    cases = store.queue(include_decided=args.all)
    print(f'{len(cases):,} case(s){" (including decided)" if args.all else " awaiting a decision"}')
    for c in cases[:args.limit]:
        top = c['reasons'][0]['title'] if c['reasons'] else '-'
        print(f"{c['trans_num']}  {c['time']}  {c['card']}  ${c['amount']:>9,.2f}  {c['category_label']:<22} "
              f"score {c['fraud_probability']:.3f}  {top}")


def cmd_investigate(store, args):
    if args.txn:
        cases = [store.case(t) for t in args.txn]
    else:
        cases = store.queue()[:args.next]
    if not cases:
        sys.exit('no cases to investigate')
    try:
        investigator = Investigator(store, mode='rules' if args.rules else 'claude', effort=args.effort,
                                    review_delay_hours=args.review_delay_hours)
    except Exception as e:     # e.g. the anthropic package or credentials are missing
        sys.exit(f'could not start the Claude client ({e}); set ANTHROPIC_API_KEY or use --rules')
    args.results_dir.mkdir(parents=True, exist_ok=True)
    failures = 0
    for case in cases:
        try:
            report = investigator.investigate(case)
        except AgentError as e:
            failures += 1
            print(f"[{case['trans_num']}] agent failed: {e}", file=sys.stderr)
            continue
        except AgentSetupError as e:
            sys.exit(f'cannot call the Claude API: {e}\nSet ANTHROPIC_API_KEY, or use --rules for rule-based reports')
        except Exception as e:
            import anthropic
            if isinstance(e, anthropic.AuthenticationError):
                sys.exit('Anthropic API key missing or invalid: set ANTHROPIC_API_KEY, or use --rules')
            if isinstance(e, anthropic.APIError):
                failures += 1
                print(f"[{case['trans_num']}] API error: {e}", file=sys.stderr)
                continue
            raise
        store.save_report(report)
        with open(args.results_dir / 'reports.jsonl', 'a') as f:
            f.write(json.dumps(report) + '\n')
        print_report(report)
    if failures:
        sys.exit(f'{failures} of {len(cases)} case(s) failed')


def cmd_show(store, args):
    report = store.latest_report(args.txn)
    if report is None:
        sys.exit(f'no saved report for {args.txn}; run `python run_agent.py investigate --txn {args.txn}`')
    print_report(report)


def cmd_decide(store, args):
    store.record_decision(args.txn, args.label, args.action, args.analyst, args.notes)
    report = store.latest_report(args.txn)
    agreed = report and report['recommended_action'] == args.action
    print(f'recorded: {args.txn} -> {args.label}, {args.action}'
          + ('' if report is None else f" (agent recommended {report['recommended_action']}"
                                       f"{', agreed' if agreed else ''})"))


def cmd_export_labels(store, args):
    labels = store.confirmed_labels()
    args.out = args.out or args.results_dir / 'confirmed_labels.csv'
    args.out.parent.mkdir(parents=True, exist_ok=True)
    labels.to_csv(args.out, index=False)
    print(f'{len(labels):,} confirmed label(s) ({labels.is_fraud.sum():,} fraud) -> {args.out}')


def main():
    from pathlib import Path
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data-dir', type=Path, default=settings.DATA_DIR)
    ap.add_argument('--results-dir', type=Path, default=settings.RESULTS_DIR)
    sub =ap.add_subparsers(dest='command', required=True)

    p = sub.add_parser('queue', help='list flagged cases awaiting a decision')
    p.add_argument('--limit', type=int, default=20)
    p.add_argument('--all', action='store_true', help='include cases that already have a decision')

    p = sub.add_parser('investigate', help='run the agent on cases')
    g = p.add_mutually_exclusive_group()
    g.add_argument('--txn', nargs='+', help='transaction id(s) of flagged cases')
    g.add_argument('--next', type=int, default=1, help='the next N undecided cases in time order')
    p.add_argument('--rules', action='store_true', help='rule-based reports instead of Claude (no API key)')
    p.add_argument('--effort', choices=['low', 'medium', 'high', 'xhigh', 'max'], default=settings.EFFORT)
    p.add_argument('--review-delay-hours', type=float, default=settings.REVIEW_DELAY_HOURS,
                   help='review the case this many hours after the flag (the agent sees transactions up to then)')

    p = sub.add_parser('show', help='print the latest saved report for a case')
    p.add_argument('--txn', required=True)

    p = sub.add_parser('decide', help="record the analyst's decision on a case")
    p.add_argument('--txn', required=True)
    p.add_argument('--label', required=True, choices=settings.LABELS)
    p.add_argument('--action', required=True, choices=settings.ACTIONS)
    p.add_argument('--analyst', required=True)
    p.add_argument('--notes', default='')

    p = sub.add_parser('export-labels', help='write analyst-confirmed labels for retraining')
    p.add_argument('--out', type=Path, help='default: <results-dir>/confirmed_labels.csv')

    args = ap.parse_args()
    store = LocalStore(args.data_dir)
    {'queue': cmd_queue, 'investigate': cmd_investigate, 'show': cmd_show, 'decide': cmd_decide,
     'export-labels': cmd_export_labels}[args.command](store, args)


if __name__ == '__main__':
    main()
