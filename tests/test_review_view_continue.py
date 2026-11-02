"""New-ID Continue carries the predecessor mind's notes as a foldable row, never gate authority.

The successor's frozen prefix holds the compact brief and the exact reader of the
predecessor's record; the predecessor's surviving authored notes (its selected
review-history view) follow the assignment as ONE host prose row the mind can fold;
no ancestor chain is walked, and no prior finish/verdict/selection is installed on
the successor.
"""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from ouroboros.context_budget import HOST_CONTEXT_KIND_KEY
from tests.test_review_decision_notes import first_plan, notes_for, LONG, NOTE, harness as _harness
from tests.test_review_view_integration import _apply, _capture, staged_body as _staged_body
from tests._store_shared import result_signature

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


def notes_row(task, drive_root):
    from ouroboros.main_context_authority import PREDECESSOR_NOTES_NAME, predecessor_notes_message

    note = predecessor_notes_message(task, drive_root=drive_root)
    assert note is not None and note['role'] == 'user' and note[HOST_CONTEXT_KIND_KEY] == PREDECESSOR_NOTES_NAME
    assert set(note) == {'role', 'content', HOST_CONTEXT_KIND_KEY}, 'a plain host prose row, nothing typed beyond its kind'
    return note


@pytest.mark.parametrize('child_drive', [False, True])
def test_successor_startup_brief_notes_row_exact_read_and_new_gate_unchanged(harness, child_drive):
    from ouroboros.artifacts import read_actor_source_bytes
    from ouroboros.context import _task_authority_projection
    from ouroboros.context_compaction import context_units, unit_kind
    from ouroboros.main_context_authority import WORKING_NOTES_FIELD
    from ouroboros.review_history_view import capture_review_history_messages, SELECTED_VIEW_FIELD
    from ouroboros.task_results import load_task_result, load_plan_review_state
    from ouroboros.tools.control import _get_task_result
    from ouroboros.tools.registry import ToolContext

    ctx = first_plan(harness)
    sub = harness.install({})
    _, receipt, _ = _apply(ctx, _capture(ctx), note=NOTE, review_notes=notes_for(ctx))
    assert receipt['status'] == 'applied'
    env, task = successor(ctx)
    canonical_task = deepcopy(task)
    before = {tid: result_signature(ctx.drive_root, tid) for tid in (ctx.task_id, task['id'])}
    old_gate = deepcopy(load_plan_review_state(ctx.drive_root, ctx.task_id))
    new_gate = deepcopy(load_plan_review_state(ctx.drive_root, task['id']))
    runtime = _task_authority_projection(env, task)
    prefix, tail = capture_review_history_messages(runtime, task_id=task['id'], drive_root=ctx.drive_root)
    note = notes_row(task, ctx.drive_root)
    messages = [{'role': 'system', 'content': json.dumps(prefix)}, {'role': 'user', 'content': task['text']}, note, *tail]
    # The frozen prefix: the brief with its exact reader; no history walk, no inherited plan authority.
    brief = prefix['predecessor_authority']
    assert brief['source']['arguments'] == {'task_id': ctx.task_id, 'include_authority': True}
    assert brief['status'] == 'failed' and 'task_contract' not in brief
    assert brief['omitted_fields']['task_contract'] > 0
    assert 'historical_review_context' not in json.dumps(prefix) and 'plan_review_authority' not in prefix
    assert NOTE not in json.dumps(prefix) and LONG not in str(messages)
    assert tail == [], 'a fresh successor has no review rows of its own'
    # The notes row: the predecessor's notes as its last fold named them (here with its selected
    # review account), verbatim, right after the assignment, naming their exact retained source.
    assert NOTE in note['content'] and note['content'].count('[Actor-authored working view') == 1
    assert f"get_task_result(task_id={ctx.task_id!r}, include_authority=True)" in note['content']
    working = load_task_result(ctx.drive_root, ctx.task_id)[WORKING_NOTES_FIELD]
    assert working['path'] in note['content'] and working['sha256'] in note['content']
    ref = receipt['selected_review_history_view']['source_ref']
    # Foldable by the existing unit rules: one dialogue unit of kind user; the assignment is never one.
    units = context_units(messages, scope='dialogue')
    assert [unit.start for unit in units] == [2] and unit_kind(messages, units[0]) == 'user'
    assert context_units(messages, scope='tool') == ()
    assert SELECTED_VIEW_FIELD not in load_task_result(ctx.drive_root, task['id'])
    assert load_plan_review_state(ctx.drive_root, task['id']) == new_gate
    assert load_plan_review_state(ctx.drive_root, ctx.task_id) == old_gate
    assert sub.calls == []
    assert task == canonical_task and OWNER in messages[1]['content']
    assert all(result_signature(ctx.drive_root, tid) == raw for tid, raw in before.items())
    # The exact sources are one reader away from the new ID, from a different drive too.
    drive = ctx.drive_root / 'different-child-drive' if child_drive else ctx.drive_root
    drive.mkdir(exist_ok=True)
    reader_ctx = ToolContext(repo_dir=ctx.repo_dir, drive_root=drive, task_id=task['id'],
                             task_contract=task['task_contract'], budget_drive_root=ctx.drive_root)
    payload = json.loads(_get_task_result(reader_ctx, ctx.task_id, include_authority=True))
    assert payload['authority'][SELECTED_VIEW_FIELD] == receipt['selected_review_history_view']
    assert payload['authority']['task_contract'], 'the omitted contract stays readable at its exact source'
    record = json.loads(read_actor_source_bytes(ctx.drive_root, ctx.task_id, ref))
    assert NOTE in json.dumps(record) and record['covered']
    assert any(LONG in read_actor_source_bytes(ctx.drive_root, ctx.task_id, covered['source_ref']).decode('utf-8')
               for covered in record['covered']), 'the original decision source stays readable by address'
    assert Path(ctx.drive_root).is_dir()


@pytest.mark.parametrize('lost', ['working_notes', 'legacy_selection'])
def test_continue_lost_notes_source_is_a_typed_gap_row_without_copying_authority(harness, lost):
    from ouroboros.artifacts import task_artifact_dir_path
    from ouroboros.context import _task_authority_projection
    from ouroboros.main_context_authority import WORKING_NOTES_FIELD
    from ouroboros.task_results import load_plan_review_state, load_task_result, write_task_result
    from ouroboros.review_history_view import SELECTED_VIEW_FIELD
    ctx = first_plan(harness)
    _, receipt, _ = _apply(ctx, _capture(ctx), review_notes=notes_for(ctx))
    env, task = successor(ctx)
    ref = load_task_result(ctx.drive_root, ctx.task_id)[WORKING_NOTES_FIELD]  # the source the row reads
    if lost == 'legacy_selection':  # a record written before working notes: the row reads the selection
        write_task_result(ctx.drive_root, ctx.task_id, 'running', strict_existing_dict=True,
                          _field_projector=lambda existing, _new: {'status': existing.get('status'),
                                                                   WORKING_NOTES_FIELD: None})
        ref = receipt['selected_review_history_view']['source_ref']
    (task_artifact_dir_path(ctx.drive_root, ctx.task_id, create=False) / ref['path']).unlink()
    gate = deepcopy(load_plan_review_state(ctx.drive_root, task['id']))
    runtime = _task_authority_projection(env, task)
    assert 'historical_review_context' not in json.dumps(runtime)
    note = notes_row(task, ctx.drive_root)
    assert 'could not be read now' in note['content'] and NOTE not in note['content']
    assert f"get_task_result(task_id={ctx.task_id!r}, include_authority=True)" in note['content']
    assert load_plan_review_state(ctx.drive_root, task['id']) == gate
    assert SELECTED_VIEW_FIELD not in load_task_result(ctx.drive_root, task['id'])
    assert OWNER in task['text']


def test_a_second_continue_reads_only_its_own_predecessor(harness):
    """No chain: the third task's prefix names the second; the first task's account is one
    reader away through the second's own record, never re-read into the third's prefix."""
    from ouroboros.context import _task_authority_projection
    from ouroboros.main_context_authority import predecessor_notes_message
    from ouroboros.review_history_view import capture_review_history_messages, SELECTED_VIEW_FIELD
    from ouroboros.task_results import load_task_result, load_plan_review_state
    from ouroboros.tools.control import _get_task_result
    from ouroboros.tools.registry import ToolContext
    ctx = first_plan(harness)
    _apply(ctx, _capture(ctx), note=NOTE, review_notes=notes_for(ctx))
    _, second = successor(ctx)
    second_ctx = ToolContext(repo_dir=ctx.repo_dir, drive_root=ctx.drive_root, task_id=second['id'],
                             task_contract=second['task_contract'])
    assert SELECTED_VIEW_FIELD not in load_task_result(ctx.drive_root, second['id'])
    env, third = successor(second_ctx)
    new_gate = deepcopy(load_plan_review_state(ctx.drive_root, third['id']))
    runtime = _task_authority_projection(env, third)
    prefix, tail = capture_review_history_messages(runtime, task_id=third['id'], drive_root=ctx.drive_root)
    shown = json.dumps([prefix, tail])
    assert NOTE not in shown and LONG not in shown and 'previous_review_context' not in shown
    assert 'historical_review_context' not in shown and 'plan_review_authority' not in prefix
    brief = prefix['predecessor_authority']
    assert brief['source']['arguments']['task_id'] == second['id'] and brief['previous_task_id'] == ctx.task_id
    assert predecessor_notes_message(third, drive_root=ctx.drive_root) is None, 'the second kept no notes of its own'
    assert SELECTED_VIEW_FIELD not in load_task_result(ctx.drive_root, third['id'])
    assert load_plan_review_state(ctx.drive_root, third['id']) == new_gate
    # The second's complete record names the first; its notes are the first's own reader away.
    third_ctx = ToolContext(repo_dir=ctx.repo_dir, drive_root=ctx.drive_root, task_id=third['id'],
                            task_contract=third['task_contract'])
    payload = json.loads(_get_task_result(third_ctx, second['id'], include_authority=True))
    assert payload['authority']['task_contract']['predecessor_authority']['source']['task_id'] == ctx.task_id
    assert NOTE in predecessor_notes_message(second, drive_root=ctx.drive_root)['content']


def test_continue_carries_commit_review_notes_without_inheriting_commit_gate(staged_body, tmp_path, monkeypatch):
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
    assert LONG not in str(runtime) and NOTE not in str(runtime), 'the notes are not in the frozen prefix'
    note = notes_row(task, ctx.drive_root)
    assert NOTE in note['content'] and LONG not in note['content']
    assert 'commit_review_authority' not in runtime
    assert review_ledger.load_record(ctx.drive_root, rid) == canonical
    assert 'selected_review_history_view' not in load_task_result(ctx.drive_root, task['id'])
