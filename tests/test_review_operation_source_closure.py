"""An old host acceptance record stays readable by its exact digest, for its own task only.

The retired task acceptance stored its host panel records in the task's artifact store;
nothing writes a new one. ``get_task_result(review_source_sha256=...)`` reads one back
whole, pinned to the physical author, refusing children, Presence and corrupted bytes.
"""

import hashlib
import json

from ouroboros import artifacts
from ouroboros.headless import prepare_task_drive
from ouroboros.task_results import write_task_result
from ouroboros.tools.registry import ToolContext, ToolRegistry


def read_late_source(registry, selector):
    metadata = registry.execute_result(selector['tool'], selector['arguments'])
    assert metadata.status == 'ok', metadata.text
    envelope = json.loads(metadata.text)
    assert 'authority' not in envelope
    view = envelope['review_source']
    result = registry.execute_result(selector['tool'], {**selector['arguments'],
        'source_start_char': 0, 'source_end_char': view['complete_chars']})
    assert result.status == 'ok', result.text
    body = json.loads(result.text)['review_source']
    assert hashlib.sha256(body['text'].encode()).hexdigest() == view['complete_sha256']
    return body['text']


def _stored_host_acceptance_record(root, task_id):
    """A host task-acceptance panel record in the shape the retired machine stored it."""
    run = {"authority": "host_root", "panel_id": "p1", "aggregate_signal": "FAIL",
           "actors": [{"slot_id": "s1", "operation_id": "op1", "status": "ok",
                       "raw_text": "FAIL: the tests were never run", "semantic_verdict": "FAIL",
                       "operation_state": "settled"}],
           "request": {"surface": "task_acceptance", "retry_key": "rk-9", "subject": "answer A", "task_id": task_id}}
    return artifacts.store_actor_source_bytes(root, task_id, category='context_checkpoints', source_id='acceptance',
                                              data=json.dumps(run).encode(), extension='json')


def test_exact_review_reader_keeps_physical_identity_and_restrictions(tmp_path):
    canonical, repo = tmp_path / 'canonical', tmp_path / 'repo'
    repo.mkdir()
    ref = _stored_host_acceptance_record(canonical, 'author')
    write_task_result(canonical, 'author', 'failed', retry_handoff={'state': 'spawned', 'child_task_id': 'replacement'})
    write_task_result(canonical, 'replacement', 'completed', result='different task')
    receiver = prepare_task_drive(canonical, 'next', 'empty')
    write_task_result(canonical, 'next', 'running', root_task_id='next', task_attempt=1)
    tools = ToolRegistry(repo_dir=repo, drive_root=receiver)
    ctx = ToolContext(repo_dir=repo, drive_root=receiver, task_id='next', budget_drive_root=str(canonical),
                      task_metadata={'budget_drive_root': str(canonical)})
    tools.set_context(ctx)
    selector = {'tool': 'get_task_result', 'arguments': {'task_id': 'author', 'review_source_sha256': ref['sha256']}}
    assert 'answer A' in read_late_source(tools, selector)
    args = selector['arguments']
    invalid_range = tools.execute_result('get_task_result', {**args, 'source_start_char': -1, 'source_end_char': 1})
    assert invalid_range.status == 'error' and invalid_range.code == 'TOOL_ARG_ERROR'
    for digest in ('../escape', '0' * 64):
        result = tools.execute_result('get_task_result', {**args, 'review_source_sha256': digest})
        assert json.loads(result.text)['review_source']['status'] == 'unavailable'
    wrong_owner = tools.execute_result('get_task_result', {**args, 'task_id': 'replacement'})
    assert json.loads(wrong_owner.text)['review_source']['status'] == 'unavailable'
    for actor in ({'delegation_role': 'subagent'}, {'presence': {'binding_id': 'room'}}):
        ctx.task_metadata = {'budget_drive_root': str(canonical), **actor}
        refused = tools.execute_result('get_task_result', args)
        assert refused.status != 'ok' and ref['sha256'] not in refused.text
    ctx.task_metadata = {'budget_drive_root': str(canonical)}
    path = artifacts.task_artifact_dir_path(canonical, 'author') / ref['path']
    path.write_bytes(path.read_bytes().replace(b'answer A', b'answer B'))
    corrupt = tools.execute_result('get_task_result', args)
    assert json.loads(corrupt.text)['review_source']['reason'] == 'source_identity_mismatch'
