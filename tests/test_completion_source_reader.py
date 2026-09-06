"""Native readers can retrieve canonical completion evidence across task drives."""
import hashlib
import json
import shutil

import pytest

from ouroboros.artifacts import task_artifact_dir_path
from ouroboros.contracts.task_constraint import TaskConstraint
from ouroboros.task_finalization import build_completion_observations
from ouroboros.task_results import write_task_result
from ouroboros.tools.registry import ToolContext, ToolRegistry
from tests.test_completion_observations import _trace


def source_reader(tmp_path, monkeypatch, mode):
    canonical, execution, repo = [tmp_path / name for name in ('canonical', 'execution', 'repo')]
    for path in (canonical, execution, repo):
        path.mkdir()
    monkeypatch.setattr('ouroboros.skill_readiness.acceptance_skill_lifecycle', lambda *_a, **_k: [])
    trace = _trace()
    trace['tool_calls'][1]['result'] = 'Полный исходный ответ 🙂'
    source = build_completion_observations(execution,
        {'id': 'source-task', 'budget_drive_root': str(canonical)}, trace)
    row = {'completion_observations': source}
    write_task_result(canonical, 'source-task', 'completed', **row)
    actor_root = execution if mode.startswith('split') else canonical
    ctx = ToolContext(repo_dir=repo, drive_root=actor_root,
                      task_id='source-task' if mode.endswith('same') else 'later-task')
    ctx.task_metadata = {'budget_drive_root': str(canonical)}
    if mode.endswith('readonly'):
        ctx.task_constraint = TaskConstraint(mode='local_readonly_subagent')
    registry = ToolRegistry(repo_dir=repo, drive_root=actor_root)
    registry.set_context(ctx)
    ref = source['source_ref']
    path = task_artifact_dir_path(canonical, 'source-task') / ref['path']
    return registry, ref, path, canonical, execution, row


@pytest.mark.parametrize('mode', ['canonical_same', 'canonical_later', 'split_same', 'split_later', 'split_readonly'])
def test_native_selector_retains_source_task_and_canonical_drive(tmp_path, monkeypatch, mode):
    reg, ref, path, _canonical, execution, _row = source_reader(tmp_path, monkeypatch, mode)
    selector = ref['reader']
    assert selector == {'tool': 'get_task_result',
                        'arguments': {'task_id': 'source-task', 'include_completion_source': True}}
    raw = path.read_bytes()
    text = raw.decode('utf-8')
    assert len(raw) > len(text)  # offsets count characters, not UTF-8 bytes
    metadata = json.loads(reg.execute(selector['tool'], selector['arguments']))['completion_source']
    assert metadata['reason'] == 'source_range_required' and 'text' not in metadata
    assert metadata['complete_chars'] == len(text)
    assert metadata['complete_sha256'] == ref['sha256'] == hashlib.sha256(raw).hexdigest()
    parts = []
    for start, end in [(0, len(text) // 2), (len(text) // 2, len(text))]:
        result = json.loads(reg.execute(selector['tool'], {
            **selector['arguments'], 'source_start_char': start, 'source_end_char': end,
        }))['completion_source']
        assert result['text_sha256'] == hashlib.sha256(result['text'].encode()).hexdigest()
        assert result['complete_sha256'] == metadata['complete_sha256']
        parts.append(result['text'])
    assert ''.join(parts) == text
    assert len(json.loads(''.join(parts))['delivery_results']) == 41
    if mode.startswith('split'):
        shutil.rmtree(execution)
        result = json.loads(reg.execute('get_task_result', {
            **selector['arguments'], 'source_start_char': 0, 'source_end_char': len(text),
        }))['completion_source']
        assert result['text'] == text
    assert path.read_bytes() == raw


@pytest.mark.parametrize('start,end', [(None, 1), (0, None), (True, 2), (0, False), (-1, 2), (2, 2), (0, 1000000)])
def test_completion_reader_preserves_the_existing_strict_range_contract(tmp_path, monkeypatch, start, end):
    reg, _ref, _path, _canonical, _execution, _row = source_reader(tmp_path, monkeypatch, 'split_readonly')
    result = json.loads(reg.execute('get_task_result', {
        'task_id': 'source-task', 'include_completion_source': True,
        'source_start_char': start, 'source_end_char': end,
    }))['completion_source']
    assert result['reason'] == 'source_range_invalid' and 'text' not in result


@pytest.mark.parametrize('fault', ['digest', 'bytes', 'missing', 'traversal'])
def test_completion_reader_does_not_claim_a_different_or_missing_source(tmp_path, monkeypatch, fault):
    reg, ref, path, canonical, _execution, row = source_reader(tmp_path, monkeypatch, 'split_readonly')
    if fault == 'digest':
        ref['sha256'] = '0' * 64
    elif fault == 'bytes':
        ref['size'] += 1
    elif fault == 'traversal':
        ref['path'] = '../outside.json'
    else:
        path.unlink()
    write_task_result(canonical, 'source-task', 'completed', **row)
    result = json.loads(reg.execute('get_task_result', {
        'task_id': 'source-task', 'include_completion_source': True,
        'source_start_char': 0, 'source_end_char': 10,
    }))['completion_source']
    assert result['status'] == 'unavailable' and 'text' not in result
    assert result['reason'] in {'source_unavailable', 'source_identity_mismatch', 'source_ref_invalid'}


def test_completion_source_follows_the_effective_retry_identity(tmp_path, monkeypatch):
    reg, ref, path, canonical, _execution, _row = source_reader(tmp_path, monkeypatch, 'split_readonly')
    write_task_result(canonical, 'original-task', 'interrupted', superseded_by='source-task')
    text = path.read_text(encoding='utf-8')
    result = json.loads(reg.execute('get_task_result', {
        'task_id': 'original-task', 'include_completion_source': True,
        'source_start_char': 0, 'source_end_char': len(text),
    }))['completion_source']
    assert result['complete_sha256'] == ref['sha256']
    assert result['text'] == text
