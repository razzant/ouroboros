"""Owner-selected native access preserves the ordinary delegated snapshot contract."""
import json
from pathlib import Path

import httpx
import pytest

from ouroboros import delegate_custody as custody, subagents
from ouroboros.delegate_registration_policy import persistent_registration, record_persistent
from ouroboros.tools import delegate
from tests.test_owner_settings_write_seam import _settings_app, isolated_settings as isolated_settings
from tests._delegated_transport_shared import (
    _delegating_ctx, _gateway,
    _owned_gateway_uses_each_test_transport,  # noqa: F401
)


@pytest.mark.parametrize('selected,expected', [(None, 'workspace_write'), ('workspace_write', 'workspace_write'), ('full', 'full')])
def test_selected_access_applies_only_to_mutating_assignments(selected, expected):
    kwargs = {} if selected is None else {'access': selected}
    assert subagents.delegated_run_shape(True, **kwargs) == subagents.DelegatedRunShape(expected, 'agent', 'live', True)
    assert subagents.delegated_run_shape(False, **kwargs) == subagents.DelegatedRunShape('readonly', 'ask')


def test_invalid_access_is_not_silently_widened():
    with pytest.raises(ValueError, match='workspace_write or full'):
        subagents.delegated_run_shape(True, 'inherit_native')
    assert subagents.delegated_run_shape(False, 'full').access == 'readonly'


@pytest.fixture
def full_run(tmp_path, monkeypatch):
    from ouroboros import claudexor_daemon

    monkeypatch.setenv('OUROBOROS_SUBAGENT_HARNESS', 'some-route=selected-model:max')
    monkeypatch.setenv('OUROBOROS_DATA_DIR', str(tmp_path / 'data'))
    monkeypatch.setenv('OUROBOROS_SUBAGENT_WORKTREE_ROOT', str(tmp_path / 'snapshots'))
    ctx = _delegating_ctx(tmp_path, acting=True)
    target = str(Path(ctx.workspace_root).resolve())
    facts = {'requests': [], 'trust_posts': [], 'allow': False, 'recorded': False,
             'lost_start': False, 'fail_trust': False, 'selected_access': 'full'}
    state = {'repoRoot': target, 'path': '/fixture/trust/project.yaml',
             'allowFullAccess': False, 'accessDefault': 'workspace_write', 'testCommandGrantCount': 0}

    import tests._delegated_transport_shared as shared
    snapshot = shared._transport_snapshot
    monkeypatch.setattr(shared, '_transport_snapshot', lambda route: {
        **snapshot(route), 'access': facts['selected_access'],
    })

    def handler(request):
        assert request.headers['x-claudexor-protocol-major'] == '3'
        path = request.url.path
        if path == '/v2/handshake':
            assert json.loads(request.content) == {'protocolMajor': 3, 'client': 'ouroboros'}
            return httpx.Response(200, json={'compatible': True, 'protocolMajor': 3,
                                          'engine': {'version': '3.10.4', 'sha': '6b14d041'}})
        if path == '/v2/agent-capabilities':
            return httpx.Response(200, json={'harnesses': [{'id': 'some-route', 'enabled': True,
                                          'accessProfilesSupported': ['readonly', 'workspace_write', 'full']}]})
        if path == '/v2/quota':
            return httpx.Response(200, json={'snapshots': [], 'absences': []})
        if path == '/v2/projects':
            return httpx.Response(200, json={'projects': [{'id': 'stable-project', 'root': target}]})
        if path == '/v2/trust':
            if facts['fail_trust']:
                raise httpx.ReadTimeout('trust observation failed')
            if request.method == 'POST':
                body = json.loads(request.content)
                assert body == {'repoRoot': target, 'allowFullAccess': True}
                facts['trust_posts'].append(body)
                facts.update(allow=True, recorded=True)
                return httpx.Response(200, json={**state, 'allowFullAccess': True})
            if request.url.params.get('repoRoot'):
                assert request.url.params['repoRoot'] == target
                rows = [{**state, 'allowFullAccess': facts['allow']}]
            else:
                rows = [{**state, 'allowFullAccess': facts['allow']}] if facts['recorded'] else []
            return httpx.Response(200, json={'entries': rows})
        if path == '/v2/runs':
            body = json.loads(request.content)
            facts['requests'].append((body, request.headers['idempotency-key']))
            if facts['lost_start']:
                raise httpx.ReadTimeout('start response lost')
            return httpx.Response(200, json={'runId': 'full-run', 'runDir': str(tmp_path / 'run')})
        raise AssertionError((request.method, path))

    def connect():
        gateway = _gateway(handler)
        gateway.handshake()
        return gateway

    monkeypatch.setattr(claudexor_daemon, 'ensure_owned_gateway', connect)
    return ctx, target, facts


def test_full_start_http_contract_and_real_snapshot_capture(full_run):
    from ouroboros.tools.subagent_integration import _integrate_delegated_patch

    ctx, target, facts = full_run
    result = json.loads(delegate._delegate_start(ctx, 'Implement the fixture.'))
    assert result['status'] == 'started', result
    request, key = facts['requests'][0]
    assert request['access'] == 'full' and request['mode'] == 'agent'
    assert request['scope'] == {'kind': 'project', 'root': target}
    assert request['execution'] == {'isolation': 'live', 'delegated': True,
                                    'workspaceRoot': result['execution_root']}
    assert request['harnesses'] == ['some-route'] and request['authPreference'] == 'subscription'
    assert request['model'] == 'selected-model' and request['effort'] == 'max'
    assert 'No filesystem sandbox is requested' in request['instructions']
    assert 'ACCESS: you may edit inside this root with full native process access requested' in request['instructions']
    assert 'effective access is established by the run receipt' in request['instructions']
    assert 'private snapshot is not an OS sandbox' in request['instructions']
    assert request['instructions'].count('this line governs.') == 1
    assert 'OS-enforced boundary' not in result['note']
    assert facts['trust_posts'] == [{'repoRoot': target, 'allowFullAccess': True}]
    snapshot = Path(result['execution_root'])
    assert snapshot != Path(target) and (snapshot / 'README.md').read_text() == 'seed\n'
    (snapshot / 'native-result.py').write_text('result = 42\n')
    assert not (Path(target) / 'native-result.py').exists()
    row = custody.replay(custody.custody_root(ctx))['full-run']
    assert row.access == 'full' and row.project_persistent and row.snapshot_id == key
    row.settled = True
    custody._CUSTODY['full-run'] = row
    capture = delegate._capture_terminal_patch(ctx, row)
    assert capture['status'] == 'ready_with_changes'
    assert capture['authority_target_root'] == target
    assert not (Path(target) / 'native-result.py').exists()
    outcome = _integrate_delegated_patch(ctx, 'full-run', 'apply', 'Fixture verified.')
    assert '✅ Integrated' in outcome, outcome
    assert (Path(target) / 'native-result.py').read_text() == 'result = 42\n'


@pytest.mark.parametrize('recorded_access,current_access', [('full', 'workspace_write'), ('workspace_write', 'full')])
def test_retry_replays_original_access_and_snapshot(full_run, monkeypatch, recorded_access, current_access):
    ctx, target, facts = full_run
    facts['selected_access'] = recorded_access
    facts['lost_start'] = True
    initial = json.loads(delegate._delegate_start(ctx, 'Same complete work order.'))
    invocation = initial['pending_invocation_id']
    first_request = facts['requests'][0]
    grants_before = len(facts['trust_posts'])
    facts['selected_access'] = current_access
    facts['lost_start'] = False
    result = json.loads(delegate._delegate_start(ctx, 'Same complete work order.', retry_of=invocation))
    assert result['status'] == 'started' and result['idempotent_recovery'], result
    assert facts['requests'] == [first_request, first_request]
    assert len(facts['trust_posts']) == grants_before
    assert result['access'] == recorded_access
    row = custody.replay(custody.custody_root(ctx))['full-run']
    assert row.access == recorded_access and row.project_persistent
    assert row.target_root == target and row.snapshot_id == invocation


@pytest.mark.parametrize('failure', ['revoked', 'transport'])
def test_trust_refusal_does_not_start_run_or_leave_pending_snapshot(full_run, failure):
    from ouroboros.subagent_worktrees import find_execution_snapshot

    ctx, target, facts = full_run
    facts['recorded'] = failure == 'revoked'
    facts['fail_trust'] = failure == 'transport'
    result = json.loads(delegate._delegate_start(ctx, 'A new assignment.'))
    assert result['status'] == 'refused' and result['definitely_unrun'], result
    assert not facts['requests'] and not facts['trust_posts']
    assert not custody.pending_invocations(custody.custody_root(ctx))
    rows = list(custody._iter_rows(custody.event_log_path(custody.custody_root(ctx))))
    failures = [row for row in rows if row.get('type') == custody.START_FAILED]
    assert len(failures) == 1 and failures[0]['definite']
    assert find_execution_snapshot(failures[0]['invocation_id']) is None


def test_full_mutation_still_requires_active_matching_workspace(tmp_path, monkeypatch):
    ctx = _delegating_ctx(tmp_path, acting=True)
    shape = subagents.delegated_run_shape(True, 'full')
    ctx.workspace_mode = ''
    record, refusal = delegate._mutation_authority(ctx, shape)
    assert not record and 'workspace_not_active' in refusal


def test_full_retry_without_snapshot_binding_is_refused(tmp_path):
    ctx = _delegating_ctx(tmp_path, acting=True)
    drive = custody.custody_root(ctx)
    custody.record_start_requested(drive, run_id='', task_id=ctx.task_id, idempotency_key='original',
        invocation_id='original', request={'prompt': 'work', 'access': 'full', 'mode': 'agent',
        'scope': {'kind': 'project', 'root': str(ctx.workspace_root)},
        'execution': {'isolation': 'live', 'delegated': True}, 'primaryHarness': 'some-route'},
        project_id='stable', project_owned=False, route='some-route')
    binding, refusal = delegate._resolve_retry_invocation(ctx, drive, 'original', 'work')
    assert binding is None and 'retry_binding_absent' in refusal


def test_full_registration_and_access_evidence_keep_their_existing_owners():
    assert persistent_registration('/stable/project', 'full')
    assert record_persistent({'request': {'access': 'full', 'execution': {'workspaceRoot': '/snapshot'}}})
    shape = subagents.DelegatedRunShape('full', 'agent', 'live', True)
    detail = {'lastSeq': 5, 'summary': {'state': 'succeeded', 'effectiveAccess': 'full'}}
    assert delegate._widened_access(detail, shape.access) == ''
    assert delegate._terminal_payload('run', detail, shape)['access_evidence'] == {
        'requested': 'full', 'effective': 'full', 'verified': True, 'state': 'succeeded'}


def test_session_access_is_snapshotted_and_changes_existing_fingerprint():
    from ouroboros.subagent_runtime import select_subagent_snapshot, validate_subagent_snapshot, model_visible_subagent_catalog

    config = {'enabled': True, 'items': [{
        'subagent_id': 'native-coder', 'recommended_use': 'Native coding',
        'route': {'kind': 'agent_session', 'target_id': 'some-route=selected-model'},
        'effort': 'max',
    }]}
    settings = {'OUROBOROS_SUBAGENTS': config}
    before, _ = select_subagent_snapshot(settings, subagent_id='native-coder')
    assert before.get('access', 'workspace_write') == 'workspace_write'
    config['items'][0]['access'] = 'full'
    selected, _ = select_subagent_snapshot(settings, subagent_id='native-coder')
    assert selected['access'] == 'full'
    assert selected['config_fingerprint'] != before['config_fingerprint']
    config['items'][0]['access'] = 'workspace_write'
    assert validate_subagent_snapshot(selected)['access'] == 'full'
    catalog = model_visible_subagent_catalog(settings)
    assert catalog['rows'][0]['mutating_access'] == 'workspace_write'
    assert validate_subagent_snapshot(before).get('access', 'workspace_write') == 'workspace_write'


def test_saved_full_access_reaches_dispatch_preflight_without_mutable_settings(full_run, monkeypatch):
    from ouroboros import subagent_runtime

    ctx, target, facts = full_run
    observed = []
    monkeypatch.setattr(subagents, 'route_health', lambda gateway, route, shape, **kw: observed.append(shape) or ('', ''))
    snapshot = {'schema': 1, 'selected_subagent_id': 'coder', 'config_fingerprint': 'snapshot',
                'route': {'kind': 'agent_session', 'target_id': 'some-route=selected-model'},
                'effort': 'max', 'access': 'full'}
    result = subagent_runtime.resolve_configured_actor_dispatch({
        'configured_subagent': snapshot,
        'task_constraint': {'mode': 'acting_subagent', 'surface': 'self_worktree', 'write_root': target},
        'parent_cognitive_route': {'model': 'fixture-provider/model', 'effort': 'max'},
    }, task_type='task')
    assert result.executor == 'harness'
    assert observed == [subagents.DelegatedRunShape('full', 'agent', 'live', True)]


def test_api_snapshot_cannot_claim_full_session_access():
    from ouroboros.subagent_runtime import validate_subagent_snapshot, SubagentSelectionError

    snapshot = {'schema': 1, 'selected_subagent_id': 'api', 'config_fingerprint': 'snapshot',
                'route': {'kind': 'api_model', 'target_id': 'provider/model'}, 'access': 'full'}
    with pytest.raises(SubagentSelectionError, match='subagent_snapshot_invalid'):
        validate_subagent_snapshot(snapshot)


def test_owner_http_save_projects_full_choice_into_task_start_snapshot(monkeypatch, isolated_settings):
    import os
    from starlette.testclient import TestClient
    from ouroboros import server_runtime
    from ouroboros.configured_subagents import SUBAGENTS_SETTING
    from ouroboros.subagent_runtime import apply_task_start_settings, select_subagent_snapshot

    monkeypatch.setattr(os, 'environ', dict(os.environ))
    monkeypatch.delenv(SUBAGENTS_SETTING, raising=False)
    monkeypatch.setattr(server_runtime, 'apply_runtime_provider_defaults', lambda settings: (settings, False, []))
    app = _settings_app(monkeypatch, isolated_settings)
    config = {'enabled': True, 'items': [{
        'subagent_id': 'phone-coder', 'route': {'kind': 'agent_session', 'target_id': 'some-route=model'},
        'recommended_use': 'Implementation on this installation.', 'access': 'full',
    }]}
    with TestClient(app) as client:
        response = client.post('/api/settings', json={SUBAGENTS_SETTING: config})
        assert response.status_code == 200, response.text
        assert json.loads(json.loads(isolated_settings.read_text())[SUBAGENTS_SETTING])['items'][0]['access'] == 'full'
        start = apply_task_start_settings()
        selected, _ = select_subagent_snapshot(start.settings, subagent_id='phone-coder')
        assert selected['access'] == 'full'
        assert json.loads(start.environ[SUBAGENTS_SETTING])['items'][0]['access'] == 'full'
        config['items'][0]['access'] = 'workspace_write'
        response = client.post('/api/settings', json={SUBAGENTS_SETTING: config})
        assert response.status_code == 200
        assert selected['access'] == 'full'
        assert json.loads(start.environ[SUBAGENTS_SETTING])['items'][0]['access'] == 'full'
