#!/usr/bin/env python3
"""Credential-free real-engine continuation witness through production custody.

Always creates its own temporary HOME/data and checksum-verifies the checkout's
runtime pin. Only fake-hang/fake-implement run, on disposable git fixtures. This
proves host admission, settlement, continuation and retirement, not a vendor
session or a harness child's containment. No login or operator daemon is used.
The fake owner results are fixture inputs; all delegated custody is production.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import tempfile
import time
import uuid

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.claudexor_platform_smoke import (  # noqa: E402
    SmokeFailure, isolated_fixture_root, poll_to_terminal, seed_fixture_repo,
)


def require(condition, code, **facts):
    if not condition:
        raise SmokeFailure(code, code.replace('_', ' '), **facts)


@contextmanager
def fixture_catalog():
    """The production catalog excludes fakes; project their real manifests here.

    This is the only substituted transport method. Addressed fake discovery
    avoids probing any vendor credentials. Continuation keys are fixture intent,
    proven by actual admission below, not a claim of capability discovery.
    No run, artifact, cancellation, settlement or retirement response is faked.
    """
    from ouroboros.gateways.claudexor import ClaudexorGateway

    original = ClaudexorGateway.agent_capabilities

    def read(gateway, *, timeout_sec=None):
        body = gateway._request('GET', '/v2/harnesses?all=true&harness=fake-hang&harness=fake-implement',
                                timeout_sec=timeout_sec)
        rows = []
        for row in body.get('harnesses', []):
            require(row.get('id') in {'fake-hang', 'fake-implement'}, 'unexpected_catalog_route')
            manifest = row.get('manifest') or {}
            require(manifest.get('kind') == 'fake', 'non_fake_manifest')
            rows.append({'id': row['id'], 'enabled': True, 'status': row.get('status'),
                         'accessProfilesSupported': manifest.get('access_profiles_supported', [])})
        return {'harnesses': rows, 'runControlKeys': ['continueFrom', 'continueCarrier']}

    ClaudexorGateway.agent_capabilities = read
    try:
        yield
    finally:
        ClaudexorGateway.agent_capabilities = original


def isolate():
    """Bind all writable roots before importing any runtime configuration."""
    root = isolated_fixture_root()
    boundary = runpy.run_path(str(REPO_ROOT / 'ouroboros/test_environment.py'))['isolated_environment']
    env = boundary(root, REPO_ROOT)
    for key in list(env):
        if key.startswith(('CLAUDEXOR_', 'NPM_', 'npm_')):
            env.pop(key)
    env['OUROBOROS_TEST_TEMP_ROOT'] = str(root)
    os.environ.clear()
    os.environ.update(env)
    tempfile.tempdir = env['TMPDIR']
    print(f'LIFECYCLE_ROOT {root}', flush=True)
    return root


def owner_result(drive, task_id, **fields):
    """A synthetic owner's durable result, read by the real retirement sweep."""
    from ouroboros.utils import write_text_atomic

    path = drive / 'task_results' / f'{task_id}.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    write_text_atomic(path, json.dumps({'_schema_version': 1, 'task_id': task_id, **fields}))


def owner_context(drive, task_id):
    from ouroboros.contracts.task_constraint import TaskConstraint
    from ouroboros.tools.registry import ToolContext

    context = ToolContext(repo_dir=seed_fixture_repo(), drive_root=drive, task_id=task_id,
                          task_constraint=TaskConstraint(mode='local_readonly_subagent'))
    context.task_metadata = {'root_task_id': task_id}
    owner_result(drive, task_id, status='running')
    return context


def start(ctx, harness, seconds, predecessor=''):
    from ouroboros import delegate_custody as custody
    from ouroboros.subagent_runtime import exact_start

    require(harness in {'fake-hang', 'fake-implement'}, 'fixture_route_required')
    snapshot = {'schema': 1, 'selected_subagent_id': harness, 'config_fingerprint': 'fixture-config',
                'route': {'kind': 'agent_session', 'target_id': harness}, 'access': 'readonly'}
    spec = {'snapshot': snapshot, 'max_seconds': seconds}
    if predecessor:
        spec.update(continue_from=predecessor, continue_carrier='packet')
    result = json.loads(exact_start(ctx, 'Read the fixture and report what is present.', spec).text)
    require(result.get('status') == 'started', 'production_start_refused', result=result)
    row = custody.replay(ctx.drive_root).get(result['run_id'])
    require(row is not None and row.access == 'readonly', 'durable_start_missing', result=result)
    print(json.dumps({'step': 'started', 'run_id': row.run_id, 'harness': harness,
                      'continuation_of': row.continuation_of}), flush=True)
    return row


def wait_running(gateway, row, seconds):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        detail = gateway.get_run(row.run_id)
        state = (detail.get('summary') or {}).get('state')
        if state == 'running':
            return
        require(state not in {'succeeded', 'failed', 'cancelled', 'interrupted'},
                'hang_ended_before_interrupt', detail=detail)
        time.sleep(0.2)
    raise SmokeFailure('hang_never_running', 'The fake run never entered running.')


def settle(ctx, gateway, row, seconds, expected):
    from ouroboros import delegate_custody as custody
    from ouroboros.tools.delegate import _delegate_wait

    detail = poll_to_terminal(gateway, row.run_id, seconds)
    state = (detail.get('summary') or {}).get('state')
    require(state == expected, 'unexpected_terminal', state=state, expected=expected)
    answer = json.loads(_delegate_wait(ctx, row.run_id, wait_sec=1, gateway=gateway))
    require((answer.get('settlement') or {}).get('settled'), 'production_settlement_failed', answer=answer)
    recorded = custody.replay(ctx.drive_root)[row.run_id]
    require(recorded.settled and recorded.ledger_recorded, 'durable_settlement_missing')
    require(gateway.find_project_id(str(ctx.repo_dir)) == row.project_id,
            'project_retired_at_settlement', run_id=row.run_id)
    print(json.dumps({'step': 'settled_project_kept', 'run_id': row.run_id, 'state': state}), flush=True)


def capacity(manager):
    from ouroboros.claudexor_runtime import get_runtime_manager

    runtime = get_runtime_manager()
    pin = runtime.pin
    plain = runtime.resolve_command()
    probe, _ = runtime._probe_payload(plain, expected_node_version=pin.node_version)
    status = manager.status_dict()
    memory = status.get('memory')
    if 'launch' in probe:
        expected = probe['launch']['nodeArgs']
        require(memory is not None and memory.get('nodeHeapArgs') == expected,
                'engine_heap_arguments_not_applied', memory=memory, expected=expected)
        require(memory.get('heapLimitBytes') is not None and memory.get('atAdmission') is not None,
                'engine_memory_missing', memory=memory)
    else:
        require(memory is None, 'legacy_memory_not_unknown', memory=memory)
        require(manager._proc.args == plain, 'legacy_spawn_changed', command=manager._proc.args)
    return {'engine_version': pin.version, 'probe_launch': probe.get('launch'), 'memory': memory}


def lifecycle(seconds):
    from ouroboros import delegate_custody as custody
    from ouroboros.claudexor_daemon import ensure_owned_gateway, get_owned_daemon
    from ouroboros.claudexor_runtime import managed_runtime_root
    from ouroboros.config import DATA_DIR
    from ouroboros.gateways.claudexor import ClaudexorUnavailable
    from ouroboros.platform_layer import request_process_tree_kill
    from ouroboros.tools.delegate import _delegate_cancel

    # The repository fetcher and the production installer verify the same pin.
    subprocess.run([sys.executable, str(REPO_ROOT / 'scripts/fetch_claudexor_runtime.py'),
                    '--output-dir', str(managed_runtime_root() / 'cache')], check=True)
    manager = get_owned_daemon()
    children, gateway = [], None
    try:
        gateway = ensure_owned_gateway()
        require(manager._proc is not None, 'owned_child_not_started')
        children.append(manager._proc)
        facts = capacity(manager)
        first = owner_context(DATA_DIR, 'lifecycle-cancel-owner')
        a = start(first, 'fake-hang', seconds)
        require(a.project_owned and not a.project_persistent, 'fixture_project_not_owned')
        wait_running(gateway, a, seconds)
        cancel = json.loads(_delegate_cancel(first, a.run_id, reason='fixture interrupt A').text)
        require(cancel.get('status') in {'requested', 'confirmed'}, 'cancel_refused', result=cancel)
        settle(first, gateway, a, seconds, 'cancelled')
        a2 = start(first, 'fake-implement', seconds, a.run_id)
        settle(first, gateway, a2, seconds, 'succeeded')

        second = owner_context(DATA_DIR, 'lifecycle-crash-owner')
        b = start(second, 'fake-hang', seconds)
        wait_running(gateway, b, seconds)
        gateway.close()
        gateway = None
        victim = manager._proc
        require(victim in children and victim.poll() is None, 'kill_target_not_owned')
        killed = request_process_tree_kill(victim)
        require(killed.get('requested'), 'owned_kill_failed', receipt=killed)
        victim.wait(timeout=20)
        gateway = ensure_owned_gateway()
        require(manager._proc is not None and manager._proc.pid != victim.pid, 'host_did_not_respawn')
        children.append(manager._proc)
        settle(second, gateway, b, seconds, 'interrupted')
        b2 = start(second, 'fake-implement', seconds, b.run_id)
        settle(second, gateway, b2, seconds, 'succeeded')

        # Deferred registrations must be swept even when no orphan run remains.
        owner_result(DATA_DIR, first.task_id, status='completed')
        owner_result(DATA_DIR, second.task_id, status='failed', reason_code='worker_crash_signal')
        custody.reconcile_orphaned_runs(DATA_DIR, set())
        require(not gateway.find_project_id(str(first.repo_dir)), 'finished_owner_project_kept')
        require(gateway.find_project_id(str(second.repo_dir)) == b.project_id,
                'technical_continue_offer_lost')
        request = dict(custody.invocation_record(DATA_DIR, a2.invocation_id)['request'])
        request.update(continueFrom=a2.run_id, continueCarrier='packet')
        try:
            gateway.start_run(request, idempotency_key=str(uuid.uuid4()))
        except ClaudexorUnavailable as exc:
            # 3.22.0 resolves project scope before predecessor history; either
            # addressed 404 proves the post-retirement continuation refused.
            require(exc.status_code == 404 and exc.code in {
                'project_not_registered', 'continuation_predecessor_unknown'},
                    'unexpected_retirement_refusal', refusal_code=exc.code, status=exc.status_code)
            facts['retired_continuation_refusal'] = exc.code
        else:
            raise SmokeFailure('retired_predecessor_admitted', 'An archived predecessor was continued.')
        facts.update(cancel_run=a.run_id, continued_cancel=a2.run_id, killed_pid=victim.pid,
                     crash_run=b.run_id, continued_crash=b2.run_id, owned_pids=[p.pid for p in children],
                     technical_root_project_kept=True)
        return facts
    finally:
        if gateway is not None:
            gateway.close()
        # Capture even a child whose startup raised before it was returned.
        if manager._proc is not None and manager._proc not in children:
            children.append(manager._proc)
        try:
            manager.stop()
        finally:
            for child in children:
                if child.poll() is None:
                    request_process_tree_kill(child)
                child.wait(timeout=20)
            print(json.dumps({'step': 'owned_children_stopped',
                              'pids': [p.pid for p in children]}), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--max-seconds', type=int, default=300)
    args = parser.parse_args(argv)
    require(args.max_seconds > 0, 'positive_deadline_required')
    isolate()
    try:
        with fixture_catalog():
            facts = lifecycle(args.max_seconds)
    except SmokeFailure as exc:
        print(json.dumps({'verdict': 'FAILED', 'reason': exc.code, 'facts': exc.facts}), flush=True)
        return 1
    print(json.dumps({'verdict': 'PASSED', **facts}), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
