"""Access choices reach the first physical run without changing task authority."""
import json
from types import SimpleNamespace

import pytest

from ouroboros import subagent_runtime as runtime
from ouroboros.delegate_shared import delegate_payload
from ouroboros.tools import delegate
from tests.test_delegated_full_access import full_run as full_run
from tests._delegated_transport_shared import _owned_gateway_uses_each_test_transport  # noqa: F401


def _settings(access=None):
    row = {'subagent_id': 'coder', 'recommended_use': 'Implement assigned work.',
           'route': {'kind': 'agent_session', 'target_id': 'some-route=selected-model', 'credential_profile_id': ''}, 'effort': 'max'}
    if access is not None:
        row['access'] = access
    return {'OUROBOROS_SUBAGENTS': {'enabled': True, 'items': [row]}}


@pytest.mark.parametrize('lower,expected', [(None, 'full'), ('workspace_write', 'workspace_write'), ('readonly', 'readonly')])
def test_direct_default_and_lowering_reach_actual_post(full_run, monkeypatch, lower, expected):
    ctx, target, facts = full_run
    monkeypatch.setattr(delegate, 'prepare_delegate_start_actor', runtime.prepare_delegate_start_actor)
    monkeypatch.setenv('OUROBOROS_SUBAGENTS', json.dumps(_settings()['OUROBOROS_SUBAGENTS']))
    options = {'subagent_id': 'coder', **({'access': lower} if lower else {})}
    result = delegate_payload(runtime.exact_start(ctx, 'Inspect or implement the assigned change.', options))
    assert result['status'] == 'started', result
    request, _ = facts['requests'][0]
    assert request['access'] == expected
    assert request['mode'] == ('ask' if expected == 'readonly' else 'agent')
    assert bool(facts['trust_posts']) is (expected == 'full')
    assert ctx.task_constraint.surface == 'self_worktree'
    if expected == 'readonly':
        assert request['scope']['root'] == target and 'execution' not in request
        assert not result.get('snapshot_id')


@pytest.mark.parametrize('lower,expected', [(None, 'full'), ('inherit', 'full'), ('workspace_write', 'workspace_write'), ('readonly', 'readonly')])
def test_schedule_snapshot_reaches_actor_first_post(full_run, monkeypatch, lower, expected):
    from ouroboros.tools import control
    from ouroboros.tools.registry import ToolContext
    from ouroboros.contracts.task_constraint import normalize_task_constraint
    from ouroboros.subagent_bootstrap import bootstrap_before_context
    from supervisor.task_dispatch import build_scheduled_task_payload
    from supervisor.events_subagent_admission import _resolve_subagent_constraint

    ctx, target, facts = full_run
    settings = _settings()
    monkeypatch.setattr(delegate, 'prepare_delegate_start_actor', runtime.prepare_delegate_start_actor)
    monkeypatch.setattr(control, 'load_settings', lambda: settings)
    monkeypatch.setenv('OUROBOROS_ALLOW_MUTATIVE_SUBAGENTS', 'true')
    parent = ToolContext(repo_dir=ctx.repo_dir, drive_root=ctx.drive_root, task_id='parent')
    parent.pending_events = []
    result = control._schedule_task(
        parent, subagent_id='coder', objective='Inspect the source and make the assigned edit.',
        expected_output='Verified result.', write_surface='external_workspace', write_root=target,
        **({'access': lower} if lower else {}))
    events = [e for e in parent.pending_events if e.get('type') == 'schedule_subagent']
    assert len(events) == 1, result
    event = events[0]
    constraint, workspace, mode, refusal = _resolve_subagent_constraint(
        SimpleNamespace(REPO_DIR=ctx.repo_dir, DRIVE_ROOT=ctx.drive_root),
        tid=event['task_id'], requested_constraint=event['task_constraint'],
        workspace_root=event.get('workspace_root', ''), workspace_mode=event.get('workspace_mode', ''),
        base_sha=event.get('base_sha', ''), parent_task_id='parent')
    assert not refusal, refusal
    task = build_scheduled_task_payload({**event, 'tid': event['task_id'],
                                        'desc': event['objective'], 'text': event['objective'],
                                        'task_constraint': constraint, 'workspace_root': workspace,
                                        'workspace_mode': mode,
                                        'parent_id': 'parent'})
    assert task['configured_subagent']['access'] == expected
    assert task['task_constraint']['surface'] == 'external_workspace'
    ctx.task_id = task['id']
    ctx.task_metadata = task['metadata']
    ctx.task_contract = task['task_contract']
    ctx.task_constraint = normalize_task_constraint(task['task_constraint'])
    ctx.workspace_mode = task['workspace_mode']
    ctx.workspace_root = task['workspace_root']
    settings['OUROBOROS_SUBAGENTS']['items'][0]['access'] = 'workspace_write'
    answer = bootstrap_before_context(ctx, task, SimpleNamespace(executor='harness', blocked=False))
    assert answer, getattr(ctx, '_configured_startup_refusal', None)
    assert json.loads(answer)['status'] == 'configured_session_started', answer
    assert len(facts['requests']) == 1
    assert facts['requests'][0][0]['access'] == expected


def test_lowering_and_legacy_snapshots_never_widen():
    from ouroboros.subagents import delegated_run_shape

    lower, _ = runtime.select_subagent_snapshot(_settings('workspace_write'), subagent_id='coder')
    assert runtime.validate_subagent_snapshot(lower, access='workspace_write')['access'] == 'workspace_write'
    assert runtime.validate_subagent_snapshot(lower, access='inherit') == lower
    readonly = runtime.validate_subagent_snapshot(lower, access='readonly')
    assert runtime.validate_subagent_snapshot(readonly, access='inherit') == readonly
    assert runtime.validate_subagent_snapshot(readonly, access='workspace_write')['access'] == 'readonly'
    assert delegated_run_shape(False, 'full').access == 'readonly'
    legacy = {key: value for key, value in lower.items() if key != 'access'}
    assert runtime.validate_subagent_snapshot(legacy).get('access', 'workspace_write') == 'workspace_write'
    assert runtime.validate_subagent_snapshot(legacy, access='inherit') == legacy
    with pytest.raises(runtime.SubagentSelectionError, match='subagent_access_invalid'):
        runtime.validate_subagent_snapshot(lower, access='full')


@pytest.mark.parametrize('configured', [False, True])
def test_retry_cannot_silently_drop_a_new_access_choice(full_run, monkeypatch, configured):
    ctx, _, facts = full_run
    if configured:
        ctx._configured_actor_bootstrap = {'selected_subagent_id': 'coder'}
    out = delegate_payload(runtime.delegate_start_entry(ctx, 'Recover.', retry_of='old', access='readonly'))
    assert out['reason'] == 'retry_selector_conflict'
    assert not facts['requests'] and not facts['trust_posts']


@pytest.mark.parametrize('geometry', [
    {'directory_strategy': 'copy', 'scope_paths': ['.']},
    {'root': 'skill_payload', 'bucket': 'external', 'skill_name': 'alpha'},
])
def test_lowered_readonly_refuses_write_geometry_before_post(full_run, monkeypatch, geometry):
    ctx, _, facts = full_run
    monkeypatch.setattr(delegate, 'prepare_delegate_start_actor', runtime.prepare_delegate_start_actor)
    snapshot, _ = runtime.select_subagent_snapshot(_settings(), subagent_id='coder', access='readonly')
    result = delegate_payload(runtime.exact_start(ctx, 'Read only.', {'snapshot': snapshot, **geometry}))
    assert result['reason'] in {'directory_execution_unavailable', 'payload_delegation_forbidden'}
    assert result['definitely_unrun'] is True
    assert not facts['requests'] and not facts['trust_posts']


@pytest.mark.parametrize('route', ['codex', 'claude', 'cursor', 'agy'])
def test_full_route_health_requires_advertised_full(route):
    from ouroboros.subagent_route_health import route_health
    from ouroboros.subagents import delegated_run_shape

    gateway = SimpleNamespace(engine_version='3.12.1', quota_snapshots=lambda: [],
        agent_capabilities=lambda: {'harnesses': [{'id': route, 'accessProfilesSupported': ['external_sandbox_full']}]})
    assert route_health(gateway, route, delegated_run_shape(True, 'full'))[0] == 'access_profile_unsupported:full'
    assert route_health(gateway, route, delegated_run_shape(True, 'workspace_write')) == ('', '')


def test_converting_a_session_reviewer_to_api_drops_only_its_session_access():
    """A review pool seat IS its catalog row: the wait card's ``reviewer:<id>``
    changes that row's route (a session row becomes an API row without a session
    access profile) and leaves the mark and every other field as saved."""
    from ouroboros.model_slots import apply_model_role_override

    actor = {**_settings('full')['OUROBOROS_SUBAGENTS']['items'][0], 'review_eligible': True}
    original = {'OUROBOROS_SUBAGENTS': json.dumps({'enabled': True, 'items': [actor]})}
    saved = apply_model_role_override(original, role='reviewer:coder', model='openai::review',
                                      credential_profile_id='', use_local=False)
    rows = json.loads(saved['OUROBOROS_SUBAGENTS'])['items']
    assert len(rows) == 1 and 'OUROBOROS_REVIEWER_SLOTS' not in saved
    assert rows[0]['route'] == {'kind': 'api_model', 'target_id': 'openai::review'}
    assert 'access' not in rows[0]
    assert {key: rows[0][key] for key in ('subagent_id', 'recommended_use', 'effort', 'review_eligible')} == {
        'subagent_id': 'coder', 'recommended_use': actor['recommended_use'], 'effort': 'max', 'review_eligible': True}
