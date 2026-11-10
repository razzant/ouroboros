import copy

import pytest


from ouroboros.schedule_continuation import admission_basis
from ouroboros.owner_continue import work_order_text
from supervisor.followup_policy import current_hold
from tests.test_g1_followup_policy import world as world  # shared isolated scheduler fixture


def receipt():
    return {'schedule_admission': {'schedule_id': 's1', 'token': 'token', 'task': {
        'id': 'root', 'text': 'Check the garden', 'context': 'weekly report',
        'metadata': {'schedule_occurrence': {'schedule_id': 's1', 'token': 'token'},
                     'schedule_basis': {'author': 'task:parent', 'origin': {'task_id': 'parent'}}}}}}


def test_frozen_schedule_is_named_separately_from_owner_words():
    basis = admission_basis(receipt(), 'root')
    text = work_order_text('root', 'interrupted_retry_lost', {'schedule': basis})
    assert 'schedule provenance, not owner words' in text
    assert 'Check the garden' in text and 'task:parent' in text
    assert "Owner's original instruction" not in text


def test_foreign_occurrence_is_not_continue_authority():
    row = receipt()
    row['schedule_admission']['task']['metadata']['schedule_occurrence']['token'] = 'foreign'
    assert admission_basis(row, 'root') is None


def test_old_restart_hold_projection_is_idempotent_and_keeps_owed_claim():
    row = {'followup_hold': {'hold_id': 'old', 'controls': {'restart': 'r1'}},
           'occurrence': {'token': 'owed', 'phase': 'claimed'}}
    before = copy.deepcopy(row)
    assert current_hold(row) is None
    assert current_hold(row) is None
    assert row == before
    row['followup_hold']['controls']['stop:parent'] = 'owner-stop'
    assert current_hold(row)['controls'] == {'stop:parent': 'owner-stop'}
    assert current_hold(row)['hold_id'] != 'old'


def test_consumed_history_does_not_offer_a_hold(tmp_path):
    row = {'trigger': {'type': 'once'}, 'completed_at': '2026-10-01',
           'followup_hold': {'controls': {'restart': 'old'}}}
    from supervisor.followup_policy import observed_store
    before = copy.deepcopy(row)
    view = observed_store(tmp_path, {'tasks': [row]})['tasks'][0]
    assert not view.get('followup_hold')
    assert row == before


def test_owner_sources_accept_schedule_and_keep_it_on_successor_hops(tmp_path):
    from ouroboros.owner_continue import owner_sources
    row = receipt()
    row['owner_mailbox'] = {'read_complete': True, 'rows': []}
    sources = owner_sources(tmp_path, row, 'root')
    assert not sources['gaps']
    assert sources['schedule']['occurrence_token'] == 'token'
    successor = {'metadata': {'continuation': {'owner_sources': sources}},
                 'owner_mailbox': {'read_complete': True, 'rows': []}}
    again = owner_sources(tmp_path, successor, 'successor')
    assert not again['gaps']
    assert again['schedule'] == sources['schedule']


def test_schedule_root_and_its_successor_admit_standard_continue(tmp_path, monkeypatch):
    from tests.test_owner_continue import _interrupted, _install_queue
    from ouroboros.task_results import write_task_result
    from supervisor.continuation_admission import admit_continuation
    _queue, _state, workers = _install_queue(tmp_path, monkeypatch)
    scheduled = receipt()['schedule_admission']
    scheduled['task']['id'] = 'pred-1'
    _interrupted(tmp_path, reason_code='interrupted_retry_lost', schedule_admission=scheduled,
                 origin_message_text='', origin_message_ref={})
    first = admit_continuation('pred-1', action_nonce='schedule-continue-1')
    assert first['ok'], first
    task = workers.PENDING.pop()
    assert 'schedule provenance, not owner words' in task['text']
    assert 'Check the garden' in task['text']
    write_task_result(tmp_path, task['id'], 'running', metadata=task['metadata'])
    write_task_result(tmp_path, task['id'], 'failed', reason_code='interrupted_retry_lost', result='interrupted again')
    second = admit_continuation(task['id'], action_nonce='schedule-continue-2')
    assert second['ok'], second
    successor = workers.PENDING[-1]
    assert successor['metadata']['continuation']['owner_sources']['schedule']['occurrence_token'] == 'token'


def test_legacy_receipt_keeps_its_recorded_task_author_and_origin():
    row = receipt()
    meta = row['schedule_admission']['task']['metadata']
    del meta['schedule_basis']
    meta['objective_author'] = {'kind': 'task', 'task_id': 'author'}
    meta['followup_origin'] = {'task_id': 'author', 'root_task_id': 'original-root'}
    basis = admission_basis(row, 'root')
    assert basis['author'] == meta['objective_author']
    assert basis['origin'] == meta['followup_origin']


def test_restart_during_preparation_waits_one_pass_without_losing_claim(world, monkeypatch):
    from tests.test_g1_followup_policy import register, row
    from supervisor import queue, schedule_occurrence
    from supervisor.followup_policy import record_restart
    _, root, pending = world
    registered = register(world)
    original = schedule_occurrence.prepare
    def crossed(claimed):
        prepared = original(claimed)
        record_restart(root, new=True)
        return prepared
    monkeypatch.setattr(schedule_occurrence, 'prepare', crossed)
    queue.check_scheduled_tasks()
    before = row(root, registered['id'])
    assert not pending and before['followup_wait'] == 'owner_restart_in_progress'
    assert not before.get('followup_hold')
    occurrence = before['occurrence'].copy()
    monkeypatch.setattr(schedule_occurrence, 'prepare', original)
    queue.check_scheduled_tasks()
    assert len(pending) == 1 and pending[0]['id'] == occurrence['task_id']
    after = row(root, registered['id'])
    assert after['occurrence']['token'] == occurrence['token']
    assert not after.get('followup_hold') and not after.get('followup_wait')


def test_restart_audits_an_episode_and_new_schedules_have_no_obsolete_seen_marker(world, monkeypatch):
    from tests.test_g1_followup_policy import register
    from supervisor import queue_schedules as schedules
    from supervisor.followup_policy import record_restart

    _, root, _ = world
    audited = []
    audit = schedules._audit_schedule_mutation
    monkeypatch.setattr(schedules, '_audit_schedule_mutation',
                        lambda **kw: audited.append(kw) or audit(**kw))
    record_restart(root, new=True)
    assert [(row['action'], row['result']) for row in audited] == [
        ('record_restart', 'intended'), ('record_restart', 'recorded')]
    assert 'followup_restart_seen' not in register(world)


def test_disabled_settled_restart_hold_is_retired_once_through_audited_scheduler(world, monkeypatch):
    from tests.test_g1_followup_policy import register, row
    from supervisor import queue, queue_schedules as store
    from supervisor.followup_policy import scan_policy, observed_store
    from ouroboros.task_results import write_task_result
    _, root, pending = world
    registered = register(world)
    data = store.load_schedule_store(root)
    record = data['tasks'][0]
    from ouroboros.utils import utc_now_iso
    record.update(enabled=False, completed_at=utc_now_iso(), last_task_id='settled',
                  followup_hold={'hold_id': 'old', 'reason': 'origin_stopped', 'controls': {'restart': 'r1'}})
    write_task_result(root, 'settled', 'completed', result='Already done')
    store._write_scheduled_tasks(data, root)
    before = copy.deepcopy(record)
    assert not observed_store(root, data)['tasks'][0].get('followup_hold')
    assert row(root, registered['id']) == before  # GET remains read-only.
    calls = []
    audit = store._audit_schedule_mutation
    def observed_audit(**kwargs):
        calls.append(kwargs)
        return audit(**kwargs)
    monkeypatch.setattr(store, '_audit_schedule_mutation', observed_audit)
    queue.check_scheduled_tasks()
    after = row(root, registered['id'])
    assert not after.get('followup_hold') and after['enabled'] is False
    assert after['completed_at'] == before['completed_at'] and after['last_task_id'] == 'settled'
    assert not pending
    assert [(c['phase'], c['result']) for c in calls] == [('intent', 'intended'), ('outcome', 'eligibility_restored')]
    assert calls[0]['before']['followup_hold'] == before['followup_hold']
    assert registered['id'] not in scan_policy(root, store.load_schedule_store(root))
    queue.check_scheduled_tasks()
    assert len(calls) == 2


def test_retirement_audit_failure_keeps_restart_and_owed_occurrence(world, monkeypatch):
    from tests.test_g1_followup_policy import register, row
    from supervisor import queue, queue_schedules as store
    _, root, pending = world
    registered = register(world)
    data = store.load_schedule_store(root)
    record = data['tasks'][0]
    record.update(enabled=False, occurrence={'token': 'owed', 'phase': 'claimed', 'task_id': 'owed-root'},
                  followup_hold={'hold_id': 'old', 'reason': 'origin_stopped', 'controls': {'restart': 'r1'}})
    store._write_scheduled_tasks(data, root)
    before = copy.deepcopy(record)
    monkeypatch.setattr(store, '_audit_schedule_mutation', lambda **_: False)
    queue.check_scheduled_tasks()
    assert row(root, registered['id']) == before and not pending


def test_mixed_hold_reclassification_invalidates_old_restore_identity(world):
    from tests.test_g1_followup_policy import register, row, stop, restore
    from supervisor import queue, queue_schedules as store
    _, root, _ = world
    registered = register(world)
    stop(root)
    queue.check_scheduled_tasks()
    data = store.load_schedule_store(root)
    record = data['tasks'][0]
    stop_hold = copy.deepcopy(record['followup_hold'])
    record['followup_hold']['controls']['restart'] = 'old-restart'
    record['followup_hold']['hold_id'] = 'combined-old-id'
    store._write_scheduled_tasks(data, root)
    outcome = restore(root, registered['id'], 'combined-old-id')
    assert outcome['status'] == 'stale_hold'
    assert row(root, registered['id'])['followup_hold'] == stop_hold


# --- the schedule's text rides every Continue to the successor and its helpers (#2764) -----------------------------

CONDITION = "Do not publish anything; keep the summary in the project notes."


def _scheduled_occurrence(*, authors=True):
    """The frozen task an accepted occurrence runs, as ``queue_schedules._task_from_schedule`` builds it: the
    objective and a context-only condition, each with the supplier the Schedules API door recorded (or none)."""
    supplied = {"objective_author": {"kind": "schedule_api", "at": "2026-11-09T09:00:00+00:00"},
                "context_author": {"kind": "schedule_api", "at": "2026-11-09T09:05:00+00:00"}} if authors else {}
    return {"schedule_id": "digest", "token": "token", "task": {
        "id": "pred-1", "text": "Summarize the tickets.", "context": CONDITION,
        "task_contract": {"objective": "Summarize the tickets.", "context": CONDITION},
        "metadata": {"schedule_id": "digest", "schedule_occurrence": {"schedule_id": "digest", "token": "token"},
                     **supplied}}}


def _successor_reads(task, root):
    """The successor as its run starts: its corpus, what it hands a helper, a session and a reviewer, its cites."""
    from ouroboros.loop_messages import _initialize_owner_directives
    from ouroboros.owner_relay import task_words_text
    from ouroboros.owner_words import (SCHEDULE_FIELD, cited_owner_basis, governing_words_for_schedule,
                                       owner_words_contain, owner_words_text)
    from ouroboros.subagent_work_order import _owner_words_section
    from ouroboros.tools.tool_context import ToolContext

    ctx = ToolContext(repo_dir=root, drive_root=root, task_id=task["id"], task_metadata=task["metadata"],
                      task_contract=task.get("task_contract") or {})
    _initialize_owner_directives(ctx, [{"role": "user", "content": task["text"]}])
    handed = governing_words_for_schedule(ctx)
    child = {"id": "child", "root_task_id": task["id"], "metadata": {"delegation_role": "subagent", **handed}}
    renders = {"helper": task_words_text(None, child), "session": _owner_words_section({**child, **handed}),
               "reviewer": owner_words_text(ctx, audience="reviewer")}
    return ctx, handed.get(SCHEDULE_FIELD) or [], renders, cited_owner_basis, owner_words_contain


def _carried(task):
    return [(row["field"], row["content"], row.get("supplier", ""), row.get("supplied_at", ""))
            for row in task["metadata"]["owner_corpus"] if row["source"] == "schedule_text"]


@pytest.mark.parametrize("authors", [True, False], ids=["schedules_api", "legacy_unknown"])
def test_a_scheduled_run_s_continue_carries_its_schedule_text_through_every_successor(tmp_path, monkeypatch, authors):
    """One Continue and a repeated one: each successor's corpus holds the occurrence's accepted objective and its
    context-only condition under the supplier recorded for them (an unknown one stays unknown), each successor hands
    them to its helpers, sessions and reviewers, and none of them ever verifies as owner words, while my human's own
    mail to the work stays theirs and citable."""
    from tests.test_owner_continue import _install_queue, _interrupted, _owner_mail
    from ouroboros.owner_words import OWNER_SOURCES
    from ouroboros.task_results import write_task_result
    from supervisor.continuation_admission import admit_continuation

    _queue, _state, workers = _install_queue(tmp_path, monkeypatch)
    _interrupted(tmp_path, reason_code="interrupted_retry_lost", schedule_admission=_scheduled_occurrence(authors=authors),
                 origin_message_text="", origin_message_ref={})
    _owner_mail(tmp_path)  # my human's words to the work: "Also add the charts", "Use last week's numbers"
    expected = [("objective", "Summarize the tickets.", "schedule_api" if authors else "",
                 "2026-11-09T09:00:00+00:00" if authors else ""),
                ("context", CONDITION, "schedule_api" if authors else "", "2026-11-09T09:05:00+00:00" if authors else "")]
    predecessor = "pred-1"
    for hop in (1, 2):
        admitted = admit_continuation(predecessor, action_nonce=f"schedule-text-{hop}")
        assert admitted["ok"], admitted
        successor = workers.PENDING.pop()
        assert _carried(successor) == expected, f"hop {hop}: the accepted text, its supplier and date, unchanged"
        ctx, handed, renders, cited_owner_basis, owner_words_contain = _successor_reads(successor, tmp_path)
        owners = [row["content"] for row in ctx._owner_directives if row["source"] in OWNER_SOURCES]
        assert "Use last week's numbers" in owners, "my human's words stay theirs"
        assert [item["content"] for item in handed] == ["Summarize the tickets.", CONDITION]
        for audience, text in renders.items():
            assert CONDITION in text and "not my human's words" in text, (hop, audience)
            assert ("supplied by the Schedules API" if authors else "a supplier the host did not record") in text
        basis, refusal = cited_owner_basis(ctx, "Do not publish anything")
        assert basis is None and refusal and not owner_words_contain(ctx, CONDITION), "never cited as owner words"
        assert cited_owner_basis(ctx, "Use last week's numbers")[0]["source"] == "owner_mailbox"
        write_task_result(tmp_path, successor["id"], "running", metadata=successor["metadata"])
        write_task_result(tmp_path, successor["id"], "failed", reason_code="interrupted_retry_lost", result="again")
        predecessor = successor["id"]
