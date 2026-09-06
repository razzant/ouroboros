"""Selected process configuration keeps values, source secrecy and actor authority."""
from __future__ import annotations

import gzip
import json
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from ouroboros import mcp_client
from ouroboros.contracts.task_constraint import TaskConstraint
from ouroboros.tools import services
from ouroboros.tools.registry import ToolContext, ToolRegistry
from ouroboros.tools.tool_result import ToolResult
from ouroboros.utils import sanitize_tool_args_for_log
from ouroboros.workspace_executor import resolve_process_env

pytestmark = pytest.mark.serial


@pytest.fixture
def process_context(tmp_path, monkeypatch):
    system, workspace, data = (tmp_path / name for name in ('system', 'workspace', 'data'))
    for path in (system, workspace, data):
        path.mkdir()
    monkeypatch.setenv('OUROBOROS_RUNTIME_MODE', 'advanced')
    monkeypatch.setattr('ouroboros.safety.check_safety', lambda *args, **kwargs: (True, ''))
    context = ToolContext(repo_dir=system, drive_root=data, workspace_root=workspace,
                          workspace_mode='external', task_id='process-env')
    registry = ToolRegistry(system, data)
    registry.set_context(context)
    yield registry, context, workspace, data
    services.kill_all_services(data)
    mcp_client.reset_manager_for_tests()


def test_environment_resolution_retains_settings_classification():
    env, secrets = resolve_process_env(
        {'PORT': '8080', 'DEBUG': '1', 'MODE': 'literal'},
        {'PASSWORD': 'CUSTOM_KEY', 'MODE': 'OPENAI_BASE_URL'},
        settings={'CUSTOM_KEY': '!', 'OPENAI_BASE_URL': 'info'},
    )
    assert env == {'PORT': '8080', 'DEBUG': '1', 'PASSWORD': '!', 'MODE': 'info'}
    assert secrets == ('!',)


def test_mcp_literal_environment_and_unknown_extras_preserve_working_server():
    raw = {'id': 'local', 'enabled': True, 'transport': 'stdio', 'command': 'unused',
           'env': {'PORT': '8080', 'DEBUG': '1'}, 'future_option': {'retained': True}}
    before = json.loads(json.dumps(raw))
    manager = mcp_client.MCPManager()

    async def listing(cfg, timeout):
        assert cfg.env == raw['env']
        return [{'name': 'probe', 'description': 'port=8080 attempt=1 /v1', 'input_schema': {}}]

    async def call(cfg, name, args, timeout):
        return ToolResult(status='ok', code='OK', text='port=8080 attempt=1 127.0.0.1 /v1')

    manager._async_list_tools, manager._async_call_tool = listing, call
    manager.reconfigure({'MCP_ENABLED': True, 'MCP_SERVERS': [raw]})
    tested = manager.test_server(raw)
    assert tested['ok'] and 'future_option' in tested['configuration_warnings'][0]
    assert manager.refresh_server('local')['ok']
    assert 'port=8080 attempt=1 /v1' in manager.list_tools_for_registry()[0]['description']
    assert 'port=8080 attempt=1 127.0.0.1 /v1' in manager.call_tool('mcp_local__probe', {})
    status = manager.status_payload()['servers'][0]
    assert not status['last_error'] and status['configuration_warnings']
    assert raw == before


def test_settings_wire_metadata_is_not_an_unknown_config_field():
    from ouroboros.gateway.settings import _mask_mcp_servers_payload

    raw = {'id': 'local', 'transport': 'stdio', 'command': 'unused', 'future_option': True}
    wire = _mask_mcp_servers_payload([raw])[0]
    assert wire['auth_configured'] is False
    cfg = mcp_client.normalize_server_config(wire)
    assert cfg is not None
    assert cfg.configuration_warnings == ('Fields retained but not applied: future_option',)


@pytest.mark.parametrize('secret', ['!', 'synthetic-quote"\\tail\nvalue'])
def test_mcp_reference_secrets_are_masked_without_a_length_floor(secret):
    raw = {'id': 'local', 'transport': 'stdio', 'command': 'unused',
           'env': {'PORT': '8080'}, 'env_from_settings': {'PASSWORD': 'CUSTOM_KEY'}}
    cfg = mcp_client.normalize_server_config(raw, settings={'CUSTOM_KEY': secret})
    assert cfg is not None and cfg.env['PASSWORD'] == secret
    diagnostic = 'port=8080 secret=' + secret + ' encoded=' + json.dumps(secret)[1:-1]
    masked = mcp_client._redact_error_text(diagnostic, cfg)
    assert secret not in masked and json.dumps(secret)[1:-1] not in masked
    assert 'port=8080' in masked and '***' in masked


def test_tool_argument_log_redacts_nested_keys_before_projection(tmp_path):
    from ouroboros.loop_tool_execution import _execute_single_tool

    secret = 'synthetic-password-without-provider-prefix'
    args = {'env': {'DB_PASSWORD': secret, 'PORT': '8080', 'DEBUG': '1'},
            'env_from_settings': {'SESSION': 'CUSTOM_KEY'}, 'items': [{'DB_PASSWORD': secret}]}
    sanitized = sanitize_tool_args_for_log('start_service', args)
    assert secret not in json.dumps(sanitized)
    assert sanitized['env']['PORT'] == '8080' and sanitized['env']['DEBUG'] == '1'
    assert sanitized['env_from_settings'] == {'SESSION': 'CUSTOM_KEY'}
    assert args['env']['DB_PASSWORD'] == secret
    received = []

    def execute(name, values):
        received.append(values)
        return ToolResult(status='ok', code='OK', text='fixture result')

    tools = SimpleNamespace(CODE_TOOLS={'start_service'}, _ctx=SimpleNamespace(task_metadata={}),
                            execute_result=execute)
    logs = tmp_path / 'logs'
    logs.mkdir()
    _execute_single_tool(tools, {'id': 'env-log', 'function': {'name': 'start_service',
                         'arguments': json.dumps(args)}}, logs, task_id='env-log')
    assert received == [args]
    assert secret not in (logs / 'tools.jsonl').read_text()


@pytest.mark.parametrize('actor', ['acting_subagent', 'local_readonly_subagent', 'presence'])
def test_restricted_service_refs_fail_before_settings_lookup(process_context, monkeypatch, actor):
    registry, ctx, workspace, _data = process_context
    if actor == 'presence':
        from ouroboros.presence_authority import build_presence_capability_ceiling, presence_ceiling_payload
        from ouroboros.presence_capabilities import PresenceResourceTarget, PresenceToolTarget
        from tests.test_presence_authority import _resolution

        ceiling = build_presence_capability_ceiling(
            skill_name='fixture', skill_content_hash='c' * 64, state_fingerprint='d' * 64,
            resolution=_resolution(PresenceToolTarget('builtin', 'start_service'),
                                   PresenceResourceTarget('active_workspace', ('service',), '.')),
        )
        ctx.task_contract = {'capability_ceiling': presence_ceiling_payload(ceiling)}
    else:
        ctx.task_constraint = TaskConstraint(mode=actor, surface='external_workspace', write_root=str(workspace))
    monkeypatch.setattr(services, 'load_settings', lambda: pytest.fail('restricted actor reached Settings'))
    monkeypatch.setattr('ouroboros.process_custody.spawn_supervised', lambda *a, **k: pytest.fail('must not spawn'))
    result = registry.execute_result('start_service', {'cmd': [sys.executable], 'cwd': str(workspace),
                                    'env_from_settings': {'TOKEN': 'CUSTOM_KEY'}})
    assert result.status == 'blocked' and result.code == 'ACCESS_BLOCKED'
    expected = 'LOCAL_READONLY_SUBAGENT_BLOCKED' if actor == 'local_readonly_subagent' else 'SERVICE_ENV_REFERENCE_BLOCKED'
    assert expected in result.text


def test_acting_service_literal_environment_still_runs(process_context, monkeypatch):
    registry, ctx, workspace, _data = process_context
    ctx.task_constraint = TaskConstraint(mode='acting_subagent', surface='external_workspace', write_root=str(workspace))
    monkeypatch.setattr(services, 'load_settings', lambda: pytest.fail('literal env must not read Settings'))
    script = workspace / 'service.py'
    script.write_text("import os,time; print('port='+os.environ['PORT']+' debug='+os.environ['DEBUG'],flush=True); time.sleep(30)")
    command = [sys.executable, str(script)]
    result = registry.execute_result('start_service', {'cmd': command, 'cwd': str(workspace),
        'env': {'PORT': '8080', 'DEBUG': '1'}, 'readiness': {'log_contains': 'port=8080 debug=1', 'timeout_sec': 3}})
    assert result.status == 'ok' and json.loads(result.text)['ready'], result.text
    assert 'port=8080 debug=1' in registry.execute('service_logs', {})


def test_service_replacement_keeps_uncertain_record_and_returns_start_error(process_context, monkeypatch):
    from ouroboros import workspace_executor

    registry, ctx, workspace, _data = process_context
    ctx.executor_ref = {'type': 'local', 'workspace_host_path': str(workspace), 'workspace_backend_path': '/workspace'}
    key = workspace_executor.service_key(ctx, 'service')
    record = object()
    monkeypatch.setattr(workspace_executor, '_SERVICES', {key: record})
    monkeypatch.setattr(workspace_executor, '_service_state', lambda item: 'unknown')
    monkeypatch.setattr(workspace_executor, '_stop_service_record', lambda *a, **k: {'stop_failed': True, 'stop_error': 'unknown'})
    monkeypatch.setattr('ouroboros.process_custody.spawn_supervised', lambda *a, **k: pytest.fail('must not spawn a duplicate'))
    try:
        result = registry.execute('start_service', {'cmd': [sys.executable], 'cwd': str(workspace)})
        assert 'SERVICE_START_ERROR' in result and 'not confirmed' in result
        assert workspace_executor._SERVICES[key] is record
    finally:
        workspace_executor._SERVICES.pop(key, None)


@pytest.mark.parametrize('cleanup', ['stop', 'task', 'global', 'global_nowait'])
def test_local_executor_finalizes_secret_log_before_forgetting_record(process_context, monkeypatch, cleanup):
    registry, ctx, workspace, data = process_context
    ctx.executor_ref = {'type': 'local', 'workspace_host_path': str(workspace), 'workspace_backend_path': '/workspace'}
    secret = 'synthetic-finalization-canary'
    monkeypatch.setattr(services, 'load_settings', lambda: {'CUSTOM_KEY': secret})
    command = [sys.executable, '-c', "import os,time; print(os.environ['TOKEN'],flush=True); print('port='+os.environ['PORT'],flush=True); time.sleep(30)"]
    started = registry.execute_result('start_service', {'cmd': command, 'cwd': str(workspace),
        'env': {'PORT': '8080'}, 'env_from_settings': {'TOKEN': 'CUSTOM_KEY'},
        'readiness': {'log_contains': secret, 'timeout_sec': 3}})
    assert started.status == 'ok' and json.loads(started.text)['ready']
    assert secret not in registry.execute('service_logs', {})
    durable = data / 'state' / 'workspace_executor_processes'
    assert all(secret not in path.read_text() for path in durable.glob('*.json'))
    if cleanup == 'stop':
        stopped = [json.loads(registry.execute('stop_service', {}))]
    elif cleanup == 'task':
        stopped = services.stop_task_services(ctx)
    else:
        stopped = services.kill_all_services(data, wait=cleanup != 'global_nowait')
    payload = next(item for item in stopped if item.get('name') == 'service')
    if payload.get('stop_failed'):
        # wait=False dispatch can return before SIGKILL is reaped. It must
        # retain env/custody rather than inventing completed log delivery.
        from ouroboros import workspace_executor

        assert cleanup == 'global_nowait'
        record = next(record for record in workspace_executor._services_snapshot() if record.name == 'service')
        assert record.secret_values == (secret,) and record.durable_record_path.exists()
        record.local_proc.wait(timeout=5)
        payload = next(item for item in services.kill_all_services(data) if item.get('name') == 'service')
    assert payload['state'] == 'stopped' and not payload.get('stop_failed')
    final = payload['log_finalization']
    assert final['deleted_live_log'] and not final['errors']
    text = gzip.decompress(Path(final['full_log_ref']['path']).read_bytes()).decode()
    assert secret not in text and 'port=8080' in text and '***' in text
    assert not list(durable.glob('*.json'))
    assert services.prune_service_logs(data, retention_days=0)['archived_files'] == 0


@pytest.mark.parametrize('actor', ['acting_subagent', 'local_readonly_subagent'])
def test_children_keep_the_existing_owner_configured_mcp_identity(process_context, actor):
    pytest.importorskip('mcp')
    registry, ctx, workspace, _data = process_context
    witness = workspace / 'mcp-witness.jsonl'
    script = workspace / 'mcp.py'
    script.write_text('''import json,os,sys
for line in sys.stdin:
    request=json.loads(line)
    if 'id' not in request: continue
    method=request['method']
    with open(sys.argv[1],'a') as f: f.write(json.dumps({'method':method,'token':os.environ['TOKEN']})+'\\n')
    if method=='initialize': result={'protocolVersion':request['params']['protocolVersion'],'capabilities':{'tools':{}},'serverInfo':{'name':'fixture','version':'1'}}
    elif method=='tools/list': result={'tools':[{'name':'probe','description':'Configured probe','inputSchema':{'type':'object'}}]}
    else: result={'content':[{'type':'text','text':'configured='+os.environ['TOKEN']}],'isError':False}
    print(json.dumps({'jsonrpc':'2.0','id':request['id'],'result':result}),flush=True)
''', encoding='utf-8')
    secret = 'synthetic-owner-mcp-identity'
    manager = mcp_client.get_manager()
    manager.reconfigure({'MCP_ENABLED': True, 'MCP_TOOL_TIMEOUT_SEC': 5, 'CUSTOM_KEY': secret,
        'MCP_SERVERS': [{'id': 'configured', 'enabled': True, 'transport': 'stdio', 'command': sys.executable,
            'args': [str(script), str(witness)], 'cwd': str(workspace), 'env_from_settings': {'TOKEN': 'CUSTOM_KEY'}}]})
    assert manager.refresh_server('configured')['ok']
    tool = 'mcp_configured__probe'
    ctx.task_constraint = TaskConstraint(mode=actor, surface='external_workspace', write_root=str(workspace),
                                         external_tool_grants=[tool])
    result = registry.execute_result(tool, {})
    assert result.status == 'ok' and 'configured=***' in result.text and secret not in result.text
    rows = [json.loads(line) for line in witness.read_text().splitlines()]
    assert any(row['method'] == 'tools/call' for row in rows)
    assert all(row['token'] == secret for row in rows)
    if actor == 'acting_subagent':
        ctx.task_constraint = replace(ctx.task_constraint, external_tool_grants=())
        refused = registry.execute_result(tool, {})
        assert refused.status == 'blocked' and refused.code == 'ACCESS_BLOCKED'
        assert len(witness.read_text().splitlines()) == len(rows)


@pytest.mark.parametrize('local_executor', [False, True])
def test_replacing_exited_service_finalizes_each_environment_revision(process_context, monkeypatch, local_executor):
    from ouroboros import workspace_executor

    registry, ctx, workspace, _data = process_context
    if local_executor:
        ctx.executor_ref = {'type': 'local', 'workspace_host_path': str(workspace), 'workspace_backend_path': '/workspace'}
    settings = {'CUSTOM_KEY': 'synthetic-first-environment'}
    monkeypatch.setattr(services, 'load_settings', lambda: dict(settings))
    finalized = []
    finalize = services._finalize_service_log_for_drive

    def observe(*args, **kwargs):
        result = finalize(*args, **kwargs)
        finalized.append(result)
        return result

    monkeypatch.setattr(services, '_finalize_service_log_for_drive', observe)
    script = workspace / 'one_shot.py'
    script.write_text("import os; print(os.environ['TOKEN'], flush=True)")
    for secret in ('synthetic-first-environment', 'synthetic-second-environment'):
        settings['CUSTOM_KEY'] = secret
        result = registry.execute_result('start_service', {'cmd': [sys.executable, str(script)],
            'cwd': str(workspace), 'env_from_settings': {'TOKEN': 'CUSTOM_KEY'}})
        assert result.status == 'ok', result.text
        records = workspace_executor._SERVICES if local_executor else services._SERVICES
        record = records[workspace_executor.service_key(ctx, 'service')]
        proc = record.local_proc if local_executor else record.proc
        assert proc.wait(timeout=5) == 0
    registry.execute('stop_service', {})
    assert len(finalized) == 2
    for result in finalized:
        assert result['deleted_live_log'] and not result['errors']
        text = gzip.decompress(Path(result['full_log_ref']['path']).read_bytes()).decode()
        assert 'synthetic-first-environment' not in text and 'synthetic-second-environment' not in text
        assert '***' in text
