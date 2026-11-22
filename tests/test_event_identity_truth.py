"""Producer-known lineage and terminal observation times survive durable events."""
import json
from types import SimpleNamespace

import pytest

from ouroboros.task_results import write_task_result
from tests.test_task_done_durable_vs_transport import _mk_ctx



def _rows(root, kind):
    path = root / 'logs/events.jsonl'
    return [row for line in (path.read_text().splitlines() if path.exists() else [])
            if (row := json.loads(line)).get('type') == kind]


class _StopAfterReceived(Exception):
    pass


def _start(root, monkeypatch, task, *, stored=True):
    """``_prepare_task_context`` up to its ``task_received`` row (the dispatch mirror after it stops the start); the
    start record's stand-in notes the rows already written when it runs and answers ``stored``."""
    from ouroboros import agent as module

    monkeypatch.setattr(module, 'attach_task_contract', lambda task: task)
    monkeypatch.setattr(module, 'apply_consciousness_authority', lambda task: task)
    monkeypatch.setattr(module, 'resolve_dispatch_axes', lambda task: {})
    monkeypatch.setattr(module, '_record_executor_resolution', lambda *args: None)
    monkeypatch.setattr(module, 'emit_dispatch_resolution', lambda *args: (_ for _ in ()).throw(_StopAfterReceived()))
    seen = []

    def persist(_task):
        seen.append(_rows(root, 'task_received'))
        return stored

    host = SimpleNamespace(env=SimpleNamespace(drive_root=root, drive_path=lambda name: root / name),
                           _event_queue=None, _persist_running_record=persist)
    with pytest.raises(_StopAfterReceived):
        module.OuroborosAgent._prepare_task_context(host, task, authority_refusal={})
    return seen, _rows(root, 'task_received')


@pytest.mark.parametrize('lineage', [{}, {'root_task_id': 'root', 'parent_task_id': 'parent'}])
def test_task_received_carries_only_supplied_top_level_identity(tmp_path, monkeypatch, lineage):
    from ouroboros import agent as module

    monkeypatch.setattr(module, 'sanitize_task_for_event', lambda task, logs: task)
    _, (row,) = _start(tmp_path, monkeypatch, {'id': 'child', **lineage})
    assert row['task_id'] == row['task']['id'] == 'child'
    assert {key: row[key] for key in lineage} == lineage
    if not lineage:
        assert 'root_task_id' not in row and 'parent_task_id' not in row


def test_the_event_counts_the_routing_lists_only_after_the_start_record_holds_them(tmp_path, monkeypatch):
    """DEVELOPMENT 03 «Source-complete decision pipeline»: the start record stores the task, its routing lists whole,
    before ``task_received`` goes out, and only a stored record lets the event keep their lengths; when that write
    fails the event is their one stored copy and keeps them whole."""
    manifest = {'projects': [{'project_id': 'p'}], 'omissions': {'projects': 0}}
    for stored in (True, False):
        seen, (row,) = _start(tmp_path / str(stored), monkeypatch,
                              {'id': 'turn', 'metadata': {'main_routing_manifest': manifest}}, stored=stored)
        assert seen == [[]], "the start record is written before the event"
        assert row['task']['metadata']['main_routing_manifest'] == (
            {'projects_len': 1, 'omissions': {'projects': 0}} if stored else manifest)
    from ouroboros import agent as module  # the real start record answers whether it stored the task

    host = SimpleNamespace(env=SimpleNamespace(drive_root=tmp_path))
    assert module.OuroborosAgent._persist_running_record(host, {'id': 'stored'}) is True
    monkeypatch.setattr(module, 'write_task_result', lambda *args, **kwargs: (_ for _ in ()).throw(OSError('full')))
    assert module.OuroborosAgent._persist_running_record(host, {'id': 'refused'}) is False


@pytest.mark.parametrize(("lineage", "fault"), [
    pytest.param({}, False, id='lineage0-False'),
    pytest.param({'root_task_id': 'root', 'parent_task_id': 'parent'}, True, id='lineage1-True'),
])
def test_both_terminal_writers_stamp_append_time_without_clamping(tmp_path, monkeypatch, fault, lineage):
    from ouroboros import headless
    from supervisor import events, events_task_done as module

    ctx = _mk_ctx(tmp_path, [], [])
    task = {'id': 'child', 'type': 'task', 'chat_id': 0, **lineage}
    ctx.RUNNING['child'] = {'task': task}
    write_task_result(tmp_path, 'child', 'completed', result='saved', artifact_status='ready', **lineage)
    monkeypatch.setattr(headless, 'terminal_task_files_ready', lambda *args: True)
    monkeypatch.setattr(events, '_finish_task_done_dispatch', lambda *args, **kwargs: None)
    clock = {'now': '2040-01-01T00:00:00+00:00'}
    monkeypatch.setattr(module, 'utc_now_iso', lambda: clock['now'])

    def before_append(*args):
        clock['now'] = '1999-01-01T00:00:00+00:00'  # a real clock can step backward
        return {}

    if fault:
        monkeypatch.setattr(events, '_authoritative_terminal_cost', before_append)
        module._resolve_lifecycle_fault({'task_id': 'child'}, ctx, 'running')
    else:
        monkeypatch.setattr(events, '_task_done_review_projection', before_append)
        module._handle_task_done({'task_id': 'child', 'ts': '2040-01-01T00:00:00+00:00',
                                  '_files_prepared_attempt': 1}, ctx)
    row, = _rows(tmp_path, 'task_done')
    assert row['ts'] == '2040-01-01T00:00:00+00:00'
    assert row['logged_at'] == '1999-01-01T00:00:00+00:00'
    assert {key: row[key] for key in lineage} == lineage
    if not lineage:
        assert 'root_task_id' not in row and 'parent_task_id' not in row


def test_task_message_is_addressed_before_durable_append(tmp_path, monkeypatch):
    from supervisor import log_addressing, telemetry_events

    monkeypatch.setattr(log_addressing, 'resolve_project_chat', lambda *args: None)
    durable, live = [], []
    ctx = SimpleNamespace(DRIVE_ROOT=tmp_path,
        RUNNING={'child': {'task': {'root_task_id': 'root', 'parent_task_id': 'parent', 'chat_id': 0}}},
        append_jsonl=lambda path, row: durable.append(dict(row)),
        bridge=SimpleNamespace(push_log=lambda row: live.append(dict(row))))
    telemetry_events._handle_task_message_injected({'task_id': 'child', 'source_task_id': 'sender'}, ctx)
    assert durable == live and len(durable) == 1
    assert durable[0]['root_task_id'] == 'root' and durable[0]['parent_task_id'] == 'parent'
    assert durable[0]['chat_id'] == 0


@pytest.mark.parametrize('known', [False, True])
def test_supervision_retains_lineage_and_failed_append_result(tmp_path, monkeypatch, known):
    from ouroboros import delegate_supervision as module

    captured = []
    monkeypatch.setattr(module.custody, 'custody_root', lambda ctx: tmp_path)
    monkeypatch.setattr(module.custody, 'emit', lambda root, kind, row: captured.append(row) or False)
    metadata = {'root_task_id': 'root', 'parent_task_id': 'parent'} if known else {}
    assert module._emit(SimpleNamespace(task_id='child', task_metadata=metadata),
                        'delegate_supervision_wake_pending', {'run_id': 'run'}) is False
    assert captured == [{'task_id': 'child', **metadata, 'run_id': 'run'}]


@pytest.mark.parametrize('request_identity', [True, False])
def test_capability_delta_has_existing_task_root_wave_slot_and_run(tmp_path, monkeypatch, request_identity):
    from ouroboros import delegate_custody
    from ouroboros.review_execution import AgentSessionReviewExecutor, ReviewAssignment
    from ouroboros.review_records import ReviewRequest, ReviewSlot
    from ouroboros.usage_accounting import UsageScope, usage_scope

    rows = []
    monkeypatch.setattr(delegate_custody, 'emit', lambda root, kind, row: rows.append((kind, row)))
    request = ReviewRequest(surface='plan_review', goal='goal',
                            task_id='child' if request_identity else '',
                            resolved_wave_id='wave' if request_identity else '')
    executor = AgentSessionReviewExecutor(ReviewAssignment(request, ReviewSlot(slot_id='seat', model='model'),
                                                          custody_root=tmp_path))
    executor._run_id = 'run'
    with usage_scope(UsageScope(task_id='child', root_task_id='root', parent_task_id='parent', review_wave_id='wave')):
        executor._emit_capability_delta([{'reason': 'unknown'}], 'extraction')
    kind, row = rows[0]
    assert kind == 'review_slot_capability_delta'
    assert {key: row[key] for key in ('task_id', 'root_task_id', 'parent_task_id', 'review_wave_id', 'slot_id', 'run_id')} == {
        'task_id': 'child', 'root_task_id': 'root', 'parent_task_id': 'parent', 'review_wave_id': 'wave',
        'slot_id': 'seat', 'run_id': 'run'}
