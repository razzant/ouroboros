"""New-ID Continue carries an attributed view, never predecessor gate authority."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.test_review_decision_notes import first_plan, notes_for, LONG, NOTE, harness as _harness
from tests.test_review_view_integration import _apply, _capture, staged_body as _staged_body

harness, staged_body = _harness, _staged_body
OWNER = 'Keep the original protocol exactly; do not substitute a different format.'


def successor(ctx):
    from supervisor.continuation_admission import _successor_task
    from ouroboros.owner_continue import successor_id
    from ouroboros.task_results import load_task_result, write_task_result
    from ouroboros.agent_startup_checks import validate_task_authority_sources
    from ouroboros.contracts.task_contract import build_task_contract

    write_task_result(ctx.drive_root, ctx.task_id, 'failed', result='Interrupted on context size.',
        execution_status='infra_failed', reason_code='llm_api_error', last_llm_error_kind='context_overflow')
    row = load_task_result(ctx.drive_root, ctx.task_id, strict=True)
    binding = {'successor_task_id': successor_id(ctx.task_id, 'continue-view-test'), 'chat_id': 7,
               'workspace_root': str(ctx.repo_dir), 'workspace_mode': 'external'}
    task = _successor_task(None, ctx.task_id, row, binding, {'cause': 'context_overflow'},
                          {'original': {'content': OWNER}, 'later': []}, {})
    env = SimpleNamespace(drive_root=ctx.drive_root, repo_dir=ctx.repo_dir)
    assert validate_task_authority_sources(env, task) == {}
    task['task_contract'] = build_task_contract(task)
    write_task_result(ctx.drive_root, task['id'], 'scheduled', task_contract=task['task_contract'])
    return env, task


@pytest.mark.parametrize('child_drive', [False, True])
def test_successor_startup_short_view_exact_read_and_new_gate_unchanged(harness, child_drive):
    from ouroboros.context import _task_authority_projection
    from ouroboros.review_history_view import capture_review_history_messages, SELECTED_VIEW_FIELD
    from ouroboros.task_results import load_task_result, load_plan_review_state, task_result_path
    from ouroboros.tools.core_file_tools import _read_file
    from ouroboros.tools.registry import ToolContext

    ctx, sub = first_plan(harness)
    _, receipt, _ = _apply(ctx, _capture(ctx), note=NOTE, review_notes=notes_for(ctx))
    assert receipt['status'] == 'applied'
    env, task = successor(ctx)
    canonical_task = deepcopy(task)
    before = {tid: task_result_path(ctx.drive_root, tid).read_bytes() for tid in (ctx.task_id, task['id'])}
    old_gate = deepcopy(load_plan_review_state(ctx.drive_root, ctx.task_id))
    new_gate = deepcopy(load_plan_review_state(ctx.drive_root, task['id']))
    runtime = _task_authority_projection(env, task)
    prefix, tail = capture_review_history_messages(runtime, task_id=task['id'], drive_root=ctx.drive_root)
    messages = [{'role': 'system', 'content': json.dumps(prefix)}, {'role': 'user', 'content': task['text']}, *tail]
    assert LONG not in str(messages) and NOTE in str(messages)
    assert 'The tested route already covers the requirement.' in str(messages)
    historical = prefix['predecessor_authority']['historical_review_context']
    assert historical['task_id'] == ctx.task_id and historical['authored_account']['authorship'] == 'predecessor_actor'
    assert 'not this task' in historical['rule'] and historical['source_reads']
    assert not historical['source_gaps']
    assert 'plan_review_authority' not in prefix
    assert SELECTED_VIEW_FIELD not in load_task_result(ctx.drive_root, task['id'])
    assert load_plan_review_state(ctx.drive_root, task['id']) == new_gate
    assert load_plan_review_state(ctx.drive_root, ctx.task_id) == old_gate
    assert len(sub.calls) == 1
    assert task == canonical_task and OWNER in messages[1]['content']
    assert all(task_result_path(ctx.drive_root, tid).read_bytes() == raw for tid, raw in before.items())
    # A task-local relative reader would resolve to the successor's empty store.
    # Use the actual advertised address from the historical model view instead.
    drive = ctx.drive_root / 'different-child-drive' if child_drive else ctx.drive_root
    drive.mkdir(exist_ok=True)
    reader_ctx = ToolContext(repo_dir=ctx.repo_dir, drive_root=drive, task_id=task['id'],
                             task_contract=task['task_contract'], budget_drive_root=ctx.drive_root)
    seen = 0
    for source in historical['source_reads']:
        args = source['read']['arguments']
        assert args['root'] == 'artifact_store' and Path(args['path']).is_absolute()
        raw = Path(args['path']).read_text(encoding='utf-8')
        if LONG in raw:
            text = _read_file(reader_ctx, **{**args, 'start_char': raw.index(LONG), 'max_chars': 400})
            assert LONG[:200] in text, text
            seen += 1
    assert seen, 'No original decision source was actually read from the new ID'


def test_continue_lost_selected_source_discloses_gap_without_copying_authority(harness):
    from ouroboros.artifacts import task_artifact_dir_path
    from ouroboros.context import _task_authority_projection
    from ouroboros.task_results import load_plan_review_state, load_task_result
    from ouroboros.review_history_view import SELECTED_VIEW_FIELD
    ctx, _ = first_plan(harness)
    _, receipt, _ = _apply(ctx, _capture(ctx), review_notes=notes_for(ctx))
    env, task = successor(ctx)
    ref = receipt['selected_review_history_view']['source_ref']
    (task_artifact_dir_path(ctx.drive_root, ctx.task_id, create=False) / ref['path']).unlink()
    gate = deepcopy(load_plan_review_state(ctx.drive_root, task['id']))
    runtime = _task_authority_projection(env, task)
    assert runtime['predecessor_authority']['historical_review_context']['status'] == 'source_unavailable'
    assert load_plan_review_state(ctx.drive_root, task['id']) == gate
    assert SELECTED_VIEW_FIELD not in load_task_result(ctx.drive_root, task['id'])
    assert OWNER in task['text']


def test_two_successive_continues_keep_first_account_without_reauthoring(harness):
    from ouroboros.context import _task_authority_projection
    from ouroboros.review_history_view import capture_review_history_messages, SELECTED_VIEW_FIELD
    from ouroboros.task_results import load_task_result, load_plan_review_state
    from ouroboros.tools.core_file_tools import _read_file
    from ouroboros.tools.registry import ToolContext
    ctx, _ = first_plan(harness)
    _apply(ctx, _capture(ctx), note=NOTE, review_notes=notes_for(ctx))
    _, second = successor(ctx)
    second_ctx = ToolContext(repo_dir=ctx.repo_dir, drive_root=ctx.drive_root, task_id=second['id'],
                             task_contract=second['task_contract'])
    assert SELECTED_VIEW_FIELD not in load_task_result(ctx.drive_root, second['id'])
    env, third = successor(second_ctx)
    new_gate = deepcopy(load_plan_review_state(ctx.drive_root, third['id']))
    runtime = _task_authority_projection(env, third)
    prefix, tail = capture_review_history_messages(runtime, task_id=third['id'], drive_root=ctx.drive_root)
    assert NOTE in str([prefix, tail]) and LONG not in str([prefix, tail])
    assert 'The tested route already covers the requirement.' in str([prefix, tail])
    assert 'plan_review_authority' not in prefix
    assert SELECTED_VIEW_FIELD not in load_task_result(ctx.drive_root, third['id'])
    assert load_plan_review_state(ctx.drive_root, third['id']) == new_gate
    old = prefix['predecessor_authority']['previous_review_context']['historical_review_context']
    assert old['task_id'] == ctx.task_id
    reader_ctx = ToolContext(repo_dir=ctx.repo_dir, drive_root=ctx.drive_root, task_id=third['id'],
                             task_contract=third['task_contract'])
    read_count = 0
    for source in old['source_reads']:
        args = source['read']['arguments']
        raw = Path(args['path']).read_text(encoding='utf-8')
        if LONG in raw:
            result = _read_file(reader_ctx, **{**args, 'start_char': raw.index(LONG), 'max_chars': 400})
            assert LONG[:200] in result, result
            read_count += 1
    assert read_count


def test_continue_recovers_commit_account_without_inheriting_commit_gate(staged_body, tmp_path, monkeypatch):
    from ouroboros import capability_evidence, reviewer_window, review_substrate, review_ledger
    from ouroboros.context import _task_authority_projection
    from ouroboros.tools.registry import ToolContext
    from ouroboros.task_results import load_task_result
    from tests.test_review_cold_history import _run, _dispatch
    monkeypatch.setattr(capability_evidence, 'probe', lambda *a, **k: None)
    monkeypatch.setattr(reviewer_window, 'reviewer_context_window', lambda *a, **k: 1_000_000)
    monkeypatch.setattr(review_substrate, 'run_review_request', _dispatch([], 'Exact critic account.'))
    ctx = ToolContext(repo_dir=Path(staged_body['repo']), drive_root=tmp_path / 'data', task_id='prior-commit')
    rid = _run(ctx, 'commit', '')
    review_ledger.note_author_decision(ctx.drive_root, rid, {'disposition': 'rejected', 'rationale': LONG})
    canonical = deepcopy(review_ledger.load_record(ctx.drive_root, rid))
    _apply(ctx, _capture(ctx), note=NOTE, review_notes=notes_for(ctx))
    env, task = successor(ctx)
    runtime = _task_authority_projection(env, task)
    assert LONG not in str(runtime) and NOTE in str(runtime)
    history = runtime['predecessor_authority']['historical_review_context']
    assert [v['family'] for v in history['contexts']] == ['commit']
    assert 'commit_review_authority' not in runtime
    assert review_ledger.load_record(ctx.drive_root, rid) == canonical
    assert 'selected_review_history_view' not in load_task_result(ctx.drive_root, task['id'])
