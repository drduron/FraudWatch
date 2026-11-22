"""The analyst agent: investigates one flagged transaction and writes a case report.

    report = Investigator(store).investigate(case)                 # Claude, with tools
    report = Investigator(store, mode='rules').investigate(case)   # no LLM: fixed rules over the same tools

What it does (status update, "Planned Addition: An AI Agent for Analysts"):
  - explains why the transaction was flagged, in plain language, from the model's reasons (reasons.py);
  - investigates with tools (tools.py): the card's recent transactions (is this a burst?), its long-term profile,
    earlier analyst decisions on the card, and activity at the merchant;
  - recommends block_card / contact_customer / dismiss with its reasoning and confidence. It only recommends:
    the analyst records the decision (store.record_decision), which becomes a confirmed label for retraining;
  - every report is saved to the audit log by the caller (run_agent.py), with the tool calls it made.

Claude runs a plain Messages API tool-use loop, so this function can run anywhere the team hosts it (e.g. a
Lambda triggered per flagged transaction). The final answer is constrained to REPORT_SCHEMA (structured outputs).
"""
import json
import time

import settings
from tools import TOOL_DEFINITIONS, CaseTools, ToolError

# USD per million tokens for claude-opus-5-5 (input, output, cache read, cache write): only for the cost estimate
PRICES = {'input': 4.00, 'output': 20.00, 'cache_read': 0.20, 'cache_write': 5.00}
FALLBACK_BETA = 'server-side-fallback-2026-07-01'
# rule-based mode: block when similar validation cases had a compromised card at least this often, dismiss at most
RULE_BLOCK_AT, RULE_DISMISS_AT = 0.9, 0.2

SYSTEM_PROMPT = """\
You are the case-review assistant in FraudWatch, a credit-card fraud detection system. A machine-learning model \
scores every transaction; this one scored above the alert threshold. You prepare the case for a human fraud \
analyst, who makes the final decision.

For each case:
1. Explain in plain language why it was flagged, with concrete numbers (amounts, multiples of the card's usual \
spend, times, counts). The case lists the reasons that drove the model's score; check them against what the tools \
show and do not state facts you have not seen.
2. Investigate with the tools: the card's transactions in the hours and days before, its long-term profile and \
any earlier analyst decisions, and, when useful, activity at the merchant.
3. Recommend one action for the card and explain why. The action is about the card: if the card is being used \
fraudulently, block it even when this particular purchase looks ordinary.
   - block_card: the card is very likely compromised now, e.g. similar cases were fraud about 90% of the time or \
more, or the card shows a burst of flagged transactions.
   - contact_customer: genuinely uncertain, e.g. similar cases were fraud only about half to three quarters of \
the time and the tools add no strong evidence either way. A call costs little; blocking a legitimate customer's \
card, or missing fraud, costs much more.
   - dismiss: only when similar cases were rarely fraud and the card's history positively shows this purchase is \
normal for the cardholder. With this model that is rare.
   Give your confidence, and say what evidence cuts against your recommendation.

What is known about this data and model:
- Each case says how often similar flagged cases turned out to be fraud on validation data: same score band, \
with or without other flagged transactions on the card in the previous 72 hours. That is your starting point. \
Move away from it only for concrete evidence from the tools, and say so. Rates based on few cases are less \
certain.
- Fraud on a compromised card comes in a burst: several transactions within hours to about 3 days, often at \
night, at amounts far above the card's or the category's usual level, often in online shopping, online \
miscellaneous or in-store grocery. Once a burst starts, the card's later transactions are usually fraud too, \
including ordinary-looking ones.
- The model's score already takes the card's recent activity into account. An earlier purchase the model scored \
low is weak evidence of a burst on its own, even if it was large or at night: cardholders make occasional large \
or late purchases. A burst means several flagged, or clearly abnormal, transactions.
- The transactions are simulated. Merchant locations are random points near the cardholder's home, so distance \
is not evidence.

Write for a busy analyst: short, specific sentences. Refer to cards only by their last four digits.\
"""

REPORT_SCHEMA = {
    'type': 'object',
    'properties': {
        'summary': {'type': 'string', 'description': 'One-line headline for the review queue.'},
        'why_flagged': {'type': 'string', 'description': 'Plain-language explanation of the flag, with numbers.'},
        'investigation_findings': {'type': 'array', 'items': {'type': 'string'},
                                   'description': 'What the tools showed, one finding per item.'},
        'related_transaction_ids': {'type': 'array', 'items': {'type': 'string'},
                                    'description': 'Other transactions that look like part of the same fraud.'},
        'burst_detected': {'type': 'boolean'},
        'recommended_action': {'type': 'string', 'enum': settings.ACTIONS},
        'confidence': {'type': 'string', 'enum': ['low', 'medium', 'high']},
        'reasoning': {'type': 'string', 'description': 'Why this action.'},
        'counter_evidence': {'type': 'string', 'description': 'Evidence against the recommendation, or "none".'},
    },
    'required': ['summary', 'why_flagged', 'investigation_findings', 'related_transaction_ids', 'burst_detected',
                 'recommended_action', 'confidence', 'reasoning', 'counter_evidence'],
    'additionalProperties': False,
}


class AgentError(Exception):
    """The agent could not produce a report for this case (refusal, truncated output, invalid report, turn limit)."""


class AgentSetupError(Exception):
    """The Claude client cannot make requests at all (e.g. no credentials): stop, don't try the next case."""


def validate_report(report):
    """Check a report against REPORT_SCHEMA (structured outputs should guarantee this; checked anyway)."""
    props = REPORT_SCHEMA['properties']
    missing = [k for k in REPORT_SCHEMA['required'] if k not in report]
    extra = [k for k in report if k not in props]
    if missing or extra:
        raise AgentError(f'report fields: missing {missing}, unexpected {extra}')
    for key, spec in props.items():
        value = report[key]
        expected = {'string': str, 'boolean': bool, 'array': list}[spec['type']]
        if not isinstance(value, expected) or ('enum' in spec and value not in spec['enum']):
            raise AgentError(f'report field {key!r} has an invalid value: {value!r}')
        if spec['type'] == 'array' and not all(isinstance(v, str) for v in value):
            raise AgentError(f'report field {key!r} must be a list of strings')
    return report


def case_message(case, review_time):
    """The opening message: the model's output for the flagged transaction."""
    packet = {
        'transaction_id': case['trans_num'], 'card': case['card'], 'time': case['time'],
        'merchant': case['merchant'], 'category': case['category_label'], 'amount_usd': case['amount'],
        'fraud_score': case['fraud_probability'], 'alert_threshold': case['threshold'],
        'other_flagged_transactions_on_card_previous_72h': case.get('other_flags_72h'),
        'similar_cases_on_validation': case.get('calibration') or 'no similar validation cases',
        'reasons_driving_the_score': [{'reason': r['title'], 'detail': r['text'], 'weight_log_odds': r['weight']}
                                      for r in case['reasons']],
        'top_model_features': case['top_features'],
        'review_time': review_time,
    }
    return ('A transaction was flagged. Here is the model output (reason weights are contributions to the '
            'log-odds of fraud; larger means it pushed the score up more):\n\n'
            + json.dumps(packet, indent=1)
            + '\n\nInvestigate it with the tools, then write the case report.')


class Investigator:
    def __init__(self, store, mode='claude', client=None, model=settings.MODEL, effort=settings.EFFORT,
                 max_turns=settings.MAX_TURNS, review_delay_hours=settings.REVIEW_DELAY_HOURS):
        if mode not in ('claude', 'rules'):
            raise ValueError("mode must be 'claude' or 'rules'")
        self.store, self.mode, self.model, self.effort = store, mode, model, effort
        self.max_turns, self.review_delay_hours = max_turns, review_delay_hours
        self.client = client
        if mode == 'claude' and client is None:
            import anthropic
            self.client = anthropic.Anthropic()

    def investigate(self, case):
        t0 = time.time()
        tools = CaseTools(self.store, case, self.review_delay_hours)
        if self.mode == 'claude':
            body, meta = self._run_claude(case, tools)
        else:
            body, meta = rules_report(case, tools), {'model': None}
        validate_report(body)
        return {
            'trans_num': case['trans_num'], 'cc_num': case['cc_num'], 'card': case['card'],
            'time': case['time'], 'merchant': case['merchant'], 'category': case['category_label'],
            'amount': case['amount'], 'fraud_probability': case['fraud_probability'],
            'threshold': case['threshold'], 'reasons': case['reasons'],
            **body,
            'mode': self.mode, 'created_at': time.strftime('%Y-%m-%dT%H:%M:%S'),
            'elapsed_s': round(time.time() - t0, 2), 'tool_calls': tools.calls, **meta,
        }

    # ---- Claude tool-use loop -------------------------------------------------------------------------------
    def _run_claude(self, case, tools):
        review_time = time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(tools.until_ts))
        messages = [{'role': 'user', 'content': case_message(case, review_time)}]
        usage = {'input_tokens': 0, 'output_tokens': 0, 'cache_read_input_tokens': 0, 'cache_creation_input_tokens': 0}
        served_by, request_ids = set(), []
        for turn in range(1, self.max_turns + 1):
            last_turn = turn == self.max_turns
            params = dict(
                model=self.model, max_tokens=settings.MAX_TOKENS, system=SYSTEM_PROMPT, tools=TOOL_DEFINITIONS,
                messages=messages, cache_control={'type': 'ephemeral'},
                output_config={'effort': self.effort, 'format': {'type': 'json_schema', 'schema': REPORT_SCHEMA}},
                betas=[FALLBACK_BETA], fallbacks='default',
            )
            if last_turn:
                params['tool_choice'] = {'type': 'none'}
            try:
                response = self.client.beta.messages.create(**params)
            except TypeError as e:     # raised by the SDK before sending, e.g. when no credentials are found
                raise AgentSetupError(str(e)) from e
            request_ids.append(getattr(response, '_request_id', None))
            served_by.add(response.model)
            for key in usage:
                usage[key] += getattr(response.usage, key, 0) or 0

            if response.stop_reason == 'refusal':
                details = getattr(response, 'stop_details', None)
                raise AgentError(f'the model declined this case (category: {getattr(details, "category", None)})')
            if response.stop_reason == 'max_tokens':
                raise AgentError('the report was cut off at max_tokens')
            messages.append({'role': 'assistant', 'content': response.content})
            if response.stop_reason == 'pause_turn':
                continue
            if response.stop_reason == 'tool_use':
                results = []
                for block in response.content:
                    if block.type != 'tool_use':
                        continue
                    try:
                        results.append({'type': 'tool_result', 'tool_use_id': block.id,
                                        'content': tools.run(block.name, block.input)})
                    except (ToolError, TypeError) as e:
                        results.append({'type': 'tool_result', 'tool_use_id': block.id,
                                        'content': f'Error: {e}', 'is_error': True})
                if turn + 1 == self.max_turns:
                    results.append({'type': 'text', 'text': 'Tool budget reached: write the case report now.'})
                messages.append({'role': 'user', 'content': results})
                continue

            text = ''.join(b.text for b in response.content if b.type == 'text')
            try:
                report = json.loads(text)
            except json.JSONDecodeError as e:
                raise AgentError(f'the final answer is not valid JSON ({e})') from e
            meta = {'model': ', '.join(sorted(served_by)), 'effort': self.effort, 'turns': turn,
                    'usage': usage, 'est_cost_usd': estimate_cost(usage), 'request_ids': request_ids}
            return report, meta
        raise AgentError(f'no report after {self.max_turns} turns')


def estimate_cost(usage):
    return round((usage['input_tokens'] * PRICES['input'] + usage['output_tokens'] * PRICES['output']
                  + usage['cache_read_input_tokens'] * PRICES['cache_read']
                  + usage['cache_creation_input_tokens'] * PRICES['cache_write']) / 1e6, 4)


# ---- rule-based mode (no LLM) ---------------------------------------------------------------------------------
def rules_report(case, tools):
    """A fixed-rule report from the same evidence: the baseline for evaluate.py, and a fallback when no API key
    is available (the dashboard still gets an explanation and a recommendation)."""
    recent = json.loads(tools.run('get_card_transactions', {'hours_before': 72}))
    profile = json.loads(tools.run('get_card_profile', {}))
    merchant = json.loads(tools.run('get_merchant_activity', {'hours_before': 24}))
    score, others = case['fraud_probability'], recent['flagged_other_than_case']
    related = [t['id'] for t in recent['transactions'] if t['flagged'] and not t['is_case_transaction']]

    cell = case.get('calibration')
    if cell is None:   # no similar validation cases: fall back to the score
        action, confidence = ('block_card', 'medium') if score >= 0.9 else ('contact_customer', 'low')
        why = f'no similar validation cases; score {score:.2f}'
    else:
        rate = cell['card_compromised_rate']
        similar = (f"{cell['cases']} similar validation case(s) (score {cell['score_band']}, "
                   f"{'with' if cell['other_flags_on_card_72h'] else 'without'} other flags on the card): "
                   f"card compromised in {rate:.0%}")
        if rate >= RULE_BLOCK_AT:
            action, confidence = 'block_card', 'high' if rate >= 0.98 else 'medium'
        elif rate <= RULE_DISMISS_AT:
            action, confidence = 'dismiss', 'medium'
        else:
            action, confidence = 'contact_customer', 'medium'
        why = similar

    texts = [r['text'] for r in case['reasons']] or ['no single reason stands out; many small factors added up']
    texts[0] = texts[0][0].upper() + texts[0][1:]
    findings = [f"{recent['transactions_in_window']} transaction(s) on the card in the previous 72 hours, "
                f"{recent['flagged_in_window']} flagged (including this one)"]
    if profile.get('transactions_seen'):
        findings.append(f"card seen for {profile['days_of_history']} days ({profile['transactions_seen']} "
                        f"transactions); average purchase ${profile['amount_mean']:,.2f}, largest "
                        f"${profile['amount_max']:,.2f}")
    findings.append(f"{merchant['flagged']} flagged transaction(s) on {merchant['flagged_cards']} card(s) at "
                    f"{merchant['merchant']} in the previous 24 hours")
    if profile.get('earlier_analyst_decisions'):
        findings.append(f"{len(profile['earlier_analyst_decisions'])} earlier analyst decision(s) on this card")
    return {
        'summary': f"{case['card']}: ${case['amount']:,.2f} {case['category_label']} at {case['merchant']} - "
                   f"score {score:.2f} -> {action.replace('_', ' ')}",
        'why_flagged': f"Score {score:.2f} (threshold {case['threshold']:.2f}). " + '; '.join(texts) + '.',
        'investigation_findings': findings,
        'related_transaction_ids': related,
        'burst_detected': others >= 1,
        'recommended_action': action,
        'confidence': confidence,
        'reasoning': f'Rule-based: {why}.',
        'counter_evidence': 'not assessed (rule-based mode)',
    }
