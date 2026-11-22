import json

import pytest

from conftest import CARD_A, HOUR, T0
from tools import CaseTools, ToolError


def test_card_history_never_goes_past_the_review_time(store):
    rows = store.card_transactions(CARD_A, T0 - 72 * HOUR, T0)
    assert 'a_case' in set(rows.trans_num)
    assert 'a_later' not in set(rows.trans_num)
    assert rows.ts.max() <= T0


def test_tools_respect_review_time_and_delay(store):
    case = store.case('a_case')
    now = json.loads(CaseTools(store, case).run('get_card_transactions', {'hours_before': 23}))
    assert [t['id'] for t in now['transactions']] == ['a_case', 'a_burst2', 'a_burst1']   # newest first
    assert now['flagged_other_than_case'] == 2
    later = json.loads(CaseTools(store, case, review_delay_hours=6).run('get_card_transactions', {'hours_before': 24}))
    assert 'a_later' in [t['id'] for t in later['transactions']]


def test_unscored_history_is_reported_as_null(store):
    out = json.loads(CaseTools(store, store.case('a_case')).run('get_card_transactions', {'hours_before': 720}))
    hist = [t for t in out['transactions'] if t['id'].startswith('a_hist')]
    assert hist and all(t['fraud_score'] is None and t['flagged'] is None for t in hist)


def test_tool_outputs_contain_no_labels_or_full_card_numbers(store):
    tools = CaseTools(store, store.case('b_case'))
    outputs = [tools.run('get_card_transactions', {'hours_before': 720}), tools.run('get_card_profile', {}),
               tools.run('get_merchant_activity', {'hours_before': 720})]
    for out in outputs:
        assert 'is_fraud' not in out and 'label' not in out
        assert str(CARD_A) not in out and '4222222222222222' not in out


def test_merchant_activity_counts_flagged_cards(store):
    out = json.loads(CaseTools(store, store.case('b_case')).run('get_merchant_activity', {'hours_before': 24}))
    assert out['flagged'] == 4 and out['flagged_cards'] == 2      # a_burst1, a_burst2, a_case, b_case
    assert {t['card'] for t in out['flagged_transactions']} == {'...1111', '...2222'}


def test_card_profile(store):
    p = json.loads(CaseTools(store, store.case('b_case')).run('get_card_profile', {}))
    assert p['transactions_seen'] == 11 and p['home'] == 'Austin, TX' and p['cardholder_age'] == 30
    assert p['earlier_analyst_decisions'] == []


@pytest.mark.parametrize('name,inp', [('get_card_transactions', {'hours_before': 0}),
                                      ('get_card_transactions', {'hours_before': 10_000}),
                                      ('get_card_transactions', {'hours_before': '24'}),
                                      ('no_such_tool', {})])
def test_bad_tool_calls_raise_tool_error(store, name, inp):
    with pytest.raises(ToolError):
        CaseTools(store, store.case('a_case')).run(name, inp)


def test_decisions_audit_log_and_labels(store):
    assert len(store.queue()) == 5
    store.save_report({'trans_num': 'a_case', 'cc_num': CARD_A, 'created_at': 'x', 'mode': 'rules',
                       'recommended_action': 'block_card'})
    store.record_decision('a_case', 'true_fraud', 'block_card', 'tester')
    store.record_decision('b_case', 'true_fraud', 'contact_customer', 'tester')
    store.record_decision('b_case', 'false_alarm', 'dismiss', 'tester', notes='customer confirmed')   # correction
    assert {c['trans_num'] for c in store.queue()} == {'a_burst1', 'a_burst2', 'a_later'}
    labels = store.confirmed_labels().set_index('trans_num').is_fraud.to_dict()
    assert labels == {'a_case': 1, 'b_case': 0}                    # latest decision wins
    # earlier decisions on the card show up for later cases only
    assert len(store.card_decisions(CARD_A, store.case('a_later')['ts'])) == 1
    assert store.card_decisions(CARD_A, store.case('a_case')['ts']) == []
    with pytest.raises(ValueError):
        store.record_decision('a_case', 'maybe', 'dismiss', 'tester')
