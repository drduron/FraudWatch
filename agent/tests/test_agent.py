"""The Claude loop against a scripted fake client: no network, no API key."""
import json
from types import SimpleNamespace as NS

import pytest

from agent import REPORT_SCHEMA, AgentError, AgentSetupError, Investigator, validate_report

GOOD_REPORT = {
    'summary': 'Card ...1111 in an active fraud burst', 'why_flagged': '$700 online purchase at night.',
    'investigation_findings': ['2 other flagged transactions in the last 2 hours'],
    'related_transaction_ids': ['a_burst1', 'a_burst2'], 'burst_detected': True,
    'recommended_action': 'block_card', 'confidence': 'high', 'reasoning': 'Burst of flagged purchases.',
    'counter_evidence': 'none',
}


def usage(i=100, o=50):
    return NS(input_tokens=i, output_tokens=o, cache_read_input_tokens=10, cache_creation_input_tokens=0)


def tool_call(name, inp, id_='tu_1'):
    return NS(type='tool_use', id=id_, name=name, input=inp)


def text(t):
    return NS(type='text', text=t)


class FakeClient:
    """Returns scripted responses and records every request."""
    def __init__(self, responses):
        self.responses, self.requests = list(responses), []
        self.beta = NS(messages=NS(create=self.create))

    def create(self, **params):
        self.requests.append({**params, 'messages': list(params['messages'])})
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        stop, content = item
        return NS(stop_reason=stop, content=content, usage=usage(), model='claude-opus-5-5', stop_details=None)


def test_tool_loop_produces_validated_report(store):
    client = FakeClient([
        ('tool_use', [text('Checking the card.'), tool_call('get_card_transactions', {'hours_before': 24}, 'tu_1'),
                      tool_call('get_card_profile', {}, 'tu_2')]),
        ('end_turn', [text(json.dumps(GOOD_REPORT))]),
    ])
    report = Investigator(store, client=client).investigate(store.case('a_case'))
    assert report['recommended_action'] == 'block_card' and report['mode'] == 'claude'
    assert [c['tool'] for c in report['tool_calls']] == ['get_card_transactions', 'get_card_profile']
    assert report['turns'] == 2 and report['usage']['input_tokens'] == 200 and report['est_cost_usd'] > 0
    # request shape: structured output, fallbacks, caching; both tool results returned in ONE user message
    first, second = client.requests
    assert first['output_config']['format']['schema'] == REPORT_SCHEMA
    assert first['fallbacks'] == 'default' and first['cache_control'] == {'type': 'ephemeral'}
    assert 'tool_choice' not in first
    results = second['messages'][-1]['content']
    assert [r['tool_use_id'] for r in results] == ['tu_1', 'tu_2']
    assert 'a_burst1' in results[0]['content'] and not results[0].get('is_error')
    assert 'a_later' not in results[0]['content']                 # never sees the future


def test_bad_tool_input_is_returned_as_error_not_raised(store):
    client = FakeClient([
        ('tool_use', [tool_call('get_card_transactions', {'hours_before': 99999})]),
        ('end_turn', [text(json.dumps(GOOD_REPORT))]),
    ])
    Investigator(store, client=client).investigate(store.case('a_case'))
    result = client.requests[1]['messages'][-1]['content'][0]
    assert result['is_error'] is True and 'hours_before' in result['content']


def test_last_turn_disables_tools(store):
    client = FakeClient([('tool_use', [tool_call('get_card_profile', {})]),
                         ('end_turn', [text(json.dumps(GOOD_REPORT))])])
    Investigator(store, client=client, max_turns=2).investigate(store.case('a_case'))
    assert client.requests[1]['tool_choice'] == {'type': 'none'}
    assert client.requests[1]['messages'][-1]['content'][-1]['type'] == 'text'


def test_turn_limit_raises(store):
    client = FakeClient([('tool_use', [tool_call('get_card_profile', {})])] * 2)
    with pytest.raises(AgentError, match='no report'):
        Investigator(store, client=client, max_turns=2).investigate(store.case('a_case'))


@pytest.mark.parametrize('response,match', [
    (('refusal', []), 'declined'),
    (('max_tokens', [text('{"summary": "cut')]), 'max_tokens'),
    (('end_turn', [text('not json')]), 'not valid JSON'),
    (('end_turn', [text(json.dumps({**GOOD_REPORT, 'recommended_action': 'freeze'}))]), 'invalid value'),
])
def test_failures_raise_agent_error(store, response, match):
    with pytest.raises(AgentError, match=match):
        Investigator(store, client=FakeClient([response])).investigate(store.case('a_case'))


def test_missing_credentials_is_a_setup_error(store):
    client = FakeClient([TypeError('Could not resolve authentication method')])
    with pytest.raises(AgentSetupError):
        Investigator(store, client=client).investigate(store.case('a_case'))


def test_rules_mode_follows_calibration(store):
    rules = Investigator(store, mode='rules')
    burst = rules.investigate(store.case('a_case'))                 # similar cases: card compromised 100%
    assert burst['recommended_action'] == 'block_card' and burst['burst_detected']
    assert set(burst['related_transaction_ids']) == {'a_burst1', 'a_burst2'}
    isolated = rules.investigate(store.case('b_case'))              # similar cases: 53% -> uncertain
    assert isolated['recommended_action'] == 'contact_customer' and not isolated['burst_detected']
    rare = dict(store.case('b_case'), calibration={**store.case('b_case')['calibration'], 'card_compromised_rate': 0.1})
    assert rules.investigate(rare)['recommended_action'] == 'dismiss'
    unknown = dict(store.case('b_case'), calibration=None)            # no similar validation cases: score decides
    assert rules.investigate(unknown)['recommended_action'] == 'contact_customer'


def test_case_message_carries_calibration(store):
    from agent import case_message
    msg = case_message(store.case('b_case'), '2020-09-13 13:26:40')
    assert '"card_compromised_rate": 0.53' in msg and '"other_flagged_transactions_on_card_previous_72h": 0' in msg


def test_validate_report_rejects_extra_and_missing_fields():
    validate_report(dict(GOOD_REPORT))
    with pytest.raises(AgentError):
        validate_report({**GOOD_REPORT, 'extra': 1})
    with pytest.raises(AgentError):
        validate_report({k: v for k, v in GOOD_REPORT.items() if k != 'reasoning'})
