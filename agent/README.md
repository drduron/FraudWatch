# FraudWatch agent: an AI assistant for fraud analysts

The model flags transactions; this agent prepares each flagged one for the analyst. It does what the status
update's "Planned Addition: An AI Agent for Analysts" lists:

| Planned | Where |
|---|---|
| Explain in plain language why a transaction was flagged, building on the 5-8 human-readable reasons | `reasons.py` (the 8 reasons) + `explain.py` (which features drove the score) |
| First round of investigation: the card's recent activity, other suspicious transactions around it (bursts) | `tools.py`: `get_card_transactions`, `get_card_profile`, `get_merchant_activity` |
| Suggest block card / contact customer / dismiss, with reasoning; the analyst makes the final call | `agent.py` (Claude, or fixed rules), `run_agent.py decide` |
| All the evidence in one place | one structured case report per transaction |
| A record of each case and decision: audit trail and confirmed labels for retraining | `store.py` (SQLite audit log), `run_agent.py export-labels` |

---

## 1. Quick start
From the repository root, after following `modelling/README.md` (Python 3.11 environment, `download_data.py`):

```bash
pip install -r agent/requirements.txt      # adds the Anthropic SDK and pytest to the modelling libraries
cd agent
python build_cases.py                      # score the test file, build the case queue (~1 min; memory as for predict.py, ~6 GB)
python -m pytest tests -q                  # 33 tests, no network or API key needed

python run_agent.py queue                  # flagged cases waiting for a decision
python run_agent.py investigate --next 1 --rules   # rule-based report, no API key needed

export ANTHROPIC_API_KEY=...               # then the actual agent:
python run_agent.py investigate --next 1
python run_agent.py decide --txn <id> --label true_fraud --action block_card --analyst <name>
python run_agent.py export-labels          # analyst-confirmed labels -> results/confirmed_labels.csv
```

---

## 2. How a case flows
```
fraudTest.csv ──► model (modelling/, unchanged) ──► score ≥ 0.184 ──► case record ──► agent ──► report ──► analyst decision
                                                     (build_cases.py)   reasons +      tools     (audit log)   (audit log →
                                                                        top features                            confirmed labels)
```

1. **`build_cases.py`** scores the test file exactly like `modelling/predict.py` (same code, saved model and
   threshold: 2,105 of 555,719 transactions flagged). For each flagged transaction it writes a case record with
   the score, the human-readable reasons and the 8 features that pushed the score up most. Per-feature
   contributions come from LightGBM's exact TreeSHAP (`pred_contrib`); the build checks that they add up to the
   model's score for every case.
2. **The agent** (`agent.py`) gets the case record and investigates with three tools, then returns a report
   in a fixed JSON format (section 4).
3. **The analyst** reads the report and records the decision. The agent never decides.

**The agent never sees the future or the labels.** Every tool is limited to transactions up to the case's
review time (the flag time, or `--review-delay-hours` later), and to the case's own card and merchant. The true
labels are written to a separate file (`data/labels.parquet`) that only `evaluate.py` reads, the same way the
stream keeps `labels.jsonl` apart.

---

## 3. The 8 reasons
Each reason owns a group of model features. A reason is reported for a transaction only if (a) its features
pushed the score up (their contributions sum to more than 0) and (b) its rule is true of the transaction. Reasons
are listed strongest first, and each one states the fact with the transaction's own numbers.

| Reason | Rule (reported only if true) | Example text |
|---|---|---|
| Card behaved suspiciously just before this | large night purchases in the last 72 h, a purchase ≥ 3x its category average, ≥ 60% of the last 5 at night, or last-24 h spending ≥ 3x usual | "the card averaged $484 per purchase over the previous 24 hours, 7.7x its usual $63; made 4 large night-time purchase(s)…" |
| Amount is high for this kind of purchase | amount ≥ 2.5x its category's average, or ≥ 3 std above similar cardholders | "$780.52 is 9.7x the average online miscellaneous purchase ($81)" |
| Amount is far above this card's normal spending | ≥ 3x the card's average, or above its largest earlier purchase | "about 13x this card's average purchase ($60)" |
| Made late at night | 22:00–03:59 | "made at 22:00-22:59, a night hour; 26% of this card's earlier purchases were at night" |
| Unusually many transactions in a short time | ≥ 3x its typical daily count today, ≥ 3 in 6 h, or ≥ 2 in 1 h | "2 other transactions on this card in the previous hour" |
| Unfamiliar kind of purchase for this card | first purchase in this category, or < 3% of its purchases | "only 16 of the card's 2060 earlier purchases (0.8%) were online grocery; first visit to this merchant" |
| Category often targeted by fraud | category fraud rate ≥ 2x overall (training period) | "online shopping purchases are 1.8% fraud in the training data, 3.0x the overall rate" |
| New card or little history | < 10 earlier transactions or < 30 days old | "first transaction ever seen on this card" |

On the test set, the most common strongest reason is "card behaved suspiciously just before this" (1,537 of
2,105 cases), then "amount is high for this kind of purchase" (509). That matches the modelling README: v2's
gain came from seeing bursts. The rules alone are also the "simple heuristic rules" planned for the Week 10
milestone (`Reason.applies`).

---

## 4. The report
Claude's final answer is constrained to this schema (structured outputs, `REPORT_SCHEMA` in `agent.py`):

| Field | Content |
|---|---|
| `summary` | one-line headline for the queue |
| `why_flagged` | plain-language explanation with numbers |
| `investigation_findings` | what the tools showed, one item each |
| `related_transaction_ids` | other transactions that look like the same fraud |
| `burst_detected` | true / false |
| `recommended_action` | `block_card`, `contact_customer` or `dismiss` |
| `confidence` | `low`, `medium`, `high` |
| `reasoning`, `counter_evidence` | why this action; what argues against it |

The saved report adds the case details, the reasons, every tool call, and (in Claude mode) the model, effort,
turns, token usage, an estimated cost and the request ids.

**Two modes.** `claude` (default) is the agent: Claude (`claude-opus-5-5`, effort `medium`) runs a tool-use loop
of at most 8 model calls. `rules` (`--rules`) decides from the calibration table (section 5) and reports the
same tool outputs. It is the baseline for evaluation and a fallback when there is no API key, so the dashboard
always gets a report.

**Claude settings.** Requests use prompt caching, and the server-side refusal fallback
(`fallbacks: "default"`): if the model declines a request, the API retries it on a fallback model instead of
failing the case. A refusal that still happens, a truncated answer or an invalid report fail that case with a
clear error, and the queue moves on.

---

## 5. Evaluation
```bash
python evaluate.py --mode rules --all                       # every flagged case, seconds, free
python evaluate.py --mode rules --split test --all          # the held-out half only
python evaluate.py --mode claude --split dev --n 56         # Claude: 28 false alarms + 28 fraud from the dev half
python evaluate.py --rescore results/eval_claude.jsonl      # re-score a saved run, no API calls
```
**Scoring.** A recommendation is *best*, *acceptable* ("contact customer") or *harmful* (missed fraud, or a
wrongly blocked customer), judged two ways:
- **transaction level:** was this transaction fraud?
- **card level:** was the card compromised at review time (a confirmed fraud on it in the 72 hours up to and
  including the case)? The actions act on the card, and 14 of the 56 false alarms are legitimate purchases in the
  middle of a real fraud burst, where blocking the card is right.

**Dev / test split.** The flagged cases are split in half by a hash of the transaction id (`--split`). Prompt and
rule changes are judged on dev; test is only used to report. 97% of flagged cases are fraud, so Claude samples
are balanced (half false alarms): a random sample would hardly contain any.

### What decides the right action: calibration
How often flagged cases turned out to be fraud, on the **validation period** (not test), by score band and by
whether the card had other flagged transactions in the previous 72 hours (`build_cases.py` writes this to
`context.json`, and every case carries its own row):

| Score band | Other flags on card | Cases | Fraud | Card compromised |
|---|---|---|---|---|
| 0.18-0.50 | no | 17 | 53% | 53% |
| 0.18-0.50 | yes | 7 | 71% | 86% |
| 0.50-0.90 | no | 20 | 70% | 70% |
| 0.50-0.90 | yes | 13 | 85% | 92% |
| 0.90-0.99 | no | 20 | 80% | 80% |
| 0.90-0.99 | yes | 30 | 100% | 100% |
| 0.99-1.00 | no | 80 | 100% | 100% |
| 0.99-1.00 | yes | 942 | 99.9% | 100% |

Even the weakest group is fraud about half the time, so **dismissing is almost never the right bet**. The useful
decision is between blocking (card very likely compromised) and contacting the customer (genuinely uncertain).
The rule-based mode now follows this table: block when similar cases had a compromised card at least 90% of the
time, dismiss at most 20%, otherwise contact the customer. Claude gets the same row with each case and is told
to start from it.

### Results on the held-out test half (1,060 cases, 28 false alarms)
| Card level | missed fraud | clean cards wrongly blocked (of 21) | compromised cards: contact instead of block | no harmful outcome |
|---|---|---|---|---|
| Rules v1 (score + flags) | 0 | 8 | 30 | 99.2% |
| **Rules v2 (calibrated)** | **0** | **5** | 44 | **99.5%** |

At transaction level v2 wrongly blocks 10 of 28 false alarms, against 13 for v1. **The gain is modest and a
trade-off:** 3 fewer legitimate customers blocked, for 14 more compromised cards that get a call first instead of
an immediate block. With 21 clean cards, the difference is suggestive, not conclusive.

### Claude agent
**First prompt** (`--n 20`, seed 0: the same 10 fraud + 10 false-alarm cases for both modes; effort `medium`;
20 reports, 0 errors, 78 s, $1.11, about $0.056 per case):

| | best | acceptable | missed fraud | wrongly blocked | no harmful outcome |
|---|---|---|---|---|---|
| Rules v1, card level | 10 | 7 | 1 | 2 | 85% |
| Claude, card level | 10 | 6 | 1 | 3 | 80% |

It was no better than the rules. Its reports showed why: it had only a flat "98% of flags are fraud", and read a
large earlier purchase the model had scored low as the start of a burst. **Its explanations and investigation
write-ups were good**: it found the rest of a burst on its own, listed the related transactions, and stated the
evidence on both sides.

**Second prompt** (current): adds the calibration row for each case, decides at card level (block / contact when
uncertain / dismiss only when similar cases were rarely fraud), and tells the agent that an earlier purchase the
model scored low is weak evidence of a burst. **Not measured yet:** the evaluation run stopped when the Anthropic
account ran out of credit. To measure it:
```bash
python evaluate.py --mode claude --split dev --n 56 --tag _v2_dev                     # ~$3
python evaluate.py --mode claude --split test --all --tag _v2_test \
       --exclude results/eval_claude.jsonl                                            # ~$60 for all ~1,050
```
Compare with `results/eval_rules_v2_dev.jsonl`. On the 56 dev cases, rules v2 wrongly blocks 2 of 21 clean cards and
misses no fraud (v1: 3 and 0), so that is the bar for Claude.

---

## 6. Files
```
agent/
├── settings.py      paths, Claude model/effort, turn limit, review delay
├── build_cases.py   SCRIPT  score transactions, write the case queue and stores to data/
├── explain.py       per-feature contributions to the model score (LightGBM TreeSHAP)
├── reasons.py       the 8 human-readable reasons
├── calibration.py   how often similar flagged cases were fraud on the validation period
├── store.py         data access: queue, card/merchant history (time-limited), SQLite audit log
├── tools.py         the agent's tools, bound to one case
├── agent.py         Investigator: Claude tool loop + rule-based mode, report schema
├── run_agent.py     SCRIPT  queue / investigate / show / decide / export-labels
├── evaluate.py      SCRIPT  recommendations vs true labels
├── tests/           pytest, no network (a scripted fake Claude client)
├── data/            (created) queue.jsonl, transactions.parquet, cards.parquet, context.json, labels.parquet, audit.db
└── results/         (created) reports.jsonl, eval_*.jsonl, confirmed_labels.csv
```

---

## 7. Moving it into the AWS pipeline (Team A)
- **One call per flagged transaction.** `Investigator(store).investigate(case)` takes a case record (the
  format in `data/queue.jsonl`) and returns the report. A Lambda triggered by a flagged transaction would build
  the case record (score, reasons and top features, which `build_cases.make_cases` computes from the feature
  row), call `investigate`, and write the report where the dashboard reads it.
- **One class to replace.** Only `store.py` reads or writes data. A DynamoDB/S3 version needs the same methods:
  `card_transactions`, `card_profile`, `merchant_activity`, `case`, `queue`, `save_report`, `latest_report`,
  `record_decision`, `card_decisions`, `confirmed_labels`.
- **Stream format.** The stream (`stream_generator.py`) renames fields and leaves out `gender`, `job`, `city`,
  `zip` and `unix_time`, which the model needs (`modelling/data.py`). Scoring stream records needs those fields
  added back, or an adapter that maps the stream schema to the model's.
- **Credentials.** Claude mode reads `ANTHROPIC_API_KEY`. On Amazon Bedrock the same loop works with the
  Bedrock client, without the server-side fallback, which Bedrock doesn't offer.

## 8. Limitations
- The data is simulated. Merchant locations are random points near home, so the tools don't report distances.
  The stream's demo attack relies mainly on a far-away location, which the model ignores.
- Only the test file and the validation period are scored; older training-file transactions show
  `fraud_score: null` in the tools.
- The calibration rests on 1,129 flagged validation cases, only 24 of them below 0.5: the low-score rates are
  rough.
- Cost and speed (measured on 20 cases at effort `medium`): about $0.056 and 15-20 s per case. Each report
  records its own estimate (`est_cost_usd`), and `evaluate.py` prints the total.
