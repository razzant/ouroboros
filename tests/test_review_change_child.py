"""Ordinary authors ask independent reviewers; public dispatch owns the boundary."""
import dataclasses
import json
from pathlib import Path

import pytest

from ouroboros.contracts.task_constraint import TaskConstraint
from ouroboros.review_ledger import load_record
from ouroboros.review_state import load_state
from ouroboros.tools.registry import ToolContext, ToolRegistry
from tests._review_session_route_shared import (  # noqa: F401
    FakeGateway, _owned_gateway_uses_each_test_transport, _terminal_detail, fake_route as _fake_route,
)
from tests import _contributor_packet_shared as shared
from tests.review_pool_rosters import pool_roster, pool_seat, set_review_pool
from tests.test_review_change_end_to_end import _foreign_project

fake_route = _fake_route  # the shared fixture, requested by name below


def _registry(ctx):
    registry = ToolRegistry(repo_dir=ctx.repo_dir, drive_root=ctx.drive_root)
    registry.set_context(ctx)
    return registry


def _call(ctx, **args):
    text = _registry(ctx).execute('review_change', {'subject': 'index', 'goal': 'Bump', **args})
    assert '{' in text, text
    return json.loads(text[text.index('{'):])


def _starts():
    return [body for gateway in FakeGateway.instances for body in gateway.start_requests]


@pytest.fixture
def tree(tmp_path, monkeypatch, fake_route):
    parent, project = _foreign_project(tmp_path, monkeypatch)
    parent.task_id = 'tree-root'
    parent.task_metadata = {'root_task_id': parent.task_id}
    set_review_pool(monkeypatch, pool_roster(pool_seat(
        'independent', 'fake-review=fake-small', kind='agent_session', access='readonly')))
    FakeGateway.detail = _terminal_detail(shared.ANSWERS['t2'], conformance='passed')
    start = FakeGateway.start_run
    monkeypatch.setattr(FakeGateway, 'reviewed', [], raising=False)

    def unique_start(self, request, **kwargs):
        root = Path(request['scope']['root'])
        try:
            tree, diff = shared.git(root, 'write-tree'), shared.git(root, 'diff', '--cached')
        except AssertionError:  # a result subject's root is the folder holding its text, not a checkout
            tree = diff = None
        FakeGateway.reviewed.append({'root': str(root), 'tree': tree, 'diff': diff})
        result = start(self, request, **kwargs)
        return {**result, 'runId': f'run-{len(_starts())}'}

    monkeypatch.setattr(FakeGateway, 'start_run', unique_start)
    monkeypatch.setenv('OUROBOROS_REVIEW_MAX_CYCLES', '1')
    monkeypatch.setenv('OUROBOROS_RUNTIME_MODE', 'pro')
    (project / 'app.py').write_text('VALUE = 2\n')
    shared.git(project, 'add', 'app.py')
    child = ToolContext(repo_dir=parent.repo_dir, system_repo_dir=parent.system_repo_dir,
                        drive_root=tmp_path / 'child-drive', task_id='working-child',
                        workspace_root=project, workspace_mode='external',
                        task_metadata={'budget_drive_root': str(parent.drive_root),
                                       'root_task_id': parent.task_id, 'parent_task_id': parent.task_id,
                                       'delegation_role': 'subagent'},
                        task_constraint=TaskConstraint(mode='acting_subagent', surface='external_workspace',
                                                       write_root=str(project), allow_review=False))
    return parent, child, project


@pytest.mark.parametrize('mode', ['pro', 'cyber_pro'])
@pytest.mark.parametrize('first_author', ['parent', 'child'])
def test_public_split_drive_shares_paid_ceiling(tree, monkeypatch, first_author, mode):
    parent, child, project = tree
    # Cyber already had this capability before ordinary-child admission. This
    # reaches actual public dispatch on both sides of the pre-existing split.
    monkeypatch.setenv('OUROBOROS_RUNTIME_MODE', mode)
    first, second = (parent, child) if first_author == 'parent' else (child, parent)
    paid = _call(first)
    assert paid['state'] == 'settled' and len(_starts()) == 1, paid
    assert paid['subject']['root'] == str(project)
    refused = _call(second, goal='A different round')
    assert refused['dispatch_refusal']['kind'] == 'review_cycles_exhausted', refused
    assert len(_starts()) == 1
    assert len([row for row in load_state(parent.drive_root).attempts if row.paid]) == 1


def test_an_acting_child_can_ask_for_a_review_of_its_own_result(tree, monkeypatch):
    """No result review is forced on a child, but its schema offers one: the same pool, its own record."""
    parent, child, _project = tree
    # A result seat answers the one result question; under a JSON schema a clean answer is no findings.
    monkeypatch.setattr(FakeGateway, 'detail', _terminal_detail('{"findings": []}', conformance='passed'))
    result = _call(child, subject='result', answer='The child bumped VALUE to 2.')
    assert result['aggregate'] == 'PASS' and result['subject']['kind'] == 'result', result
    (start,) = _starts()
    assert start['outputSchema']['required'] == ['findings'], 'a result seat is asked for findings, not change/coupling'
    assert load_record(parent.drive_root, result['record_id'])['task_id'] == child.task_id


@pytest.mark.parametrize('subject', ['index', 'worktree', 'base..head'])
def test_ordinary_child_can_request_independent_review(tree, subject):
    parent, child, project = tree
    schema = _registry(child).get_schema_by_name('review_change')
    assert schema is not None
    props = schema['function']['parameters']['properties']
    assert props['subject']['enum'] == ['index', 'worktree', 'base..head', 'result']
    assert 'answer' in props and 'system' not in props['subject']['enum']
    assert props['surface']['enum'] == ['change', 'preflight']
    args = {'subject': subject}
    if subject == 'base..head':
        args.update(base=shared.git(project, 'rev-parse', 'HEAD'),
                    head=shared.git(project, 'commit-tree', shared.git(project, 'write-tree'),
                                    '-p', 'HEAD', '-m', 'synthetic review range'))
    result = _call(child, **args)
    assert FakeGateway.reviewed[0]['tree'] == result['subject']['tree_sha']
    assert 'VALUE = 2' in FakeGateway.reviewed[0]['diff']
    assert result['aggregate'] == 'PASS' and result['subject']['root'] == str(project), result
    assert len(_starts()) == 1 and _starts()[0]['mode'] == 'ask'
    assert _starts()[0]['access'] == 'readonly'
    assert load_record(parent.drive_root, result['record_id'])['task_id'] == child.task_id


@pytest.mark.parametrize('target', ['foreign', 'system', 'symlink', 'enclosing'])
def test_name_grant_alone_cannot_widen_the_review_target(tree, monkeypatch, target):
    parent, child, project = tree
    (parent.repo_dir / 'README.md').write_text('Foreign target change\n')
    shared.git(parent.repo_dir, 'add', 'README.md')
    # Qualification of the target boundary independently of the missing name:
    # only expose the proposed name, leaving real Registry/handler admission.
    from ouroboros import tool_capabilities
    from ouroboros.tools import review_change
    monkeypatch.setattr(tool_capabilities, 'ACTING_SUBAGENT_TOOL_NAMES',
                        tool_capabilities.ACTING_SUBAGENT_TOOL_NAMES | {'review_change'})
    deep_calls = []
    monkeypatch.setattr(review_change, 'run_system_review', lambda *_a, **_k: deep_calls.append(True) or {'message': ''})
    args = {'workspace_root': str(parent.repo_dir)}
    if target == 'system':
        args = {'subject': 'system', 'surface': 'system'}
    elif target == 'symlink':
        alias = project / 'foreign-alias'
        alias.symlink_to(parent.repo_dir, target_is_directory=True)
        args = {'workspace_root': str(alias)}
    elif target == 'enclosing':
        folder = project / 'assigned-subfolder'
        folder.mkdir()
        child.workspace_root = folder
        child.task_constraint = dataclasses.replace(child.task_constraint, write_root=str(folder))
        args = {}
    answer = _registry(child).execute('review_change', {'subject': 'index', **args})
    assert 'assigned' in answer.lower(), answer
    assert not _starts() and not deep_calls


def _own_copy(parent, child, project, tmp_path):
    from ouroboros.subagent_worktrees import provision_worktree
    from ouroboros.workspace_copies import admitted_copy_metadata
    handle = provision_worktree(repo_dir=project, task_id=child.task_id,
                                parent_task_id=parent.task_id, data_dir=parent.drive_root,
                                worktree_root=tmp_path / 'copies',
                                source_is_system_repo=project == parent.repo_dir)
    copy = Path(handle.path)
    child.workspace_root, child.workspace_mode = copy, 'self_worktree'
    child.task_constraint = dataclasses.replace(child.task_constraint, surface='self_worktree', write_root=str(copy))
    child.task_metadata['workspace_copy'] = admitted_copy_metadata(str(copy), parent.drive_root)
    (copy / 'app.py').write_text('VALUE = 3  # child-only change\n')
    shared.git(copy, 'add', 'app.py')
    return copy


def test_public_copy_shares_its_proven_source_pool(tree, tmp_path, monkeypatch):
    parent, child, project = tree
    copy = _own_copy(parent, child, project, tmp_path)
    monkeypatch.setenv('OUROBOROS_RUNTIME_MODE', 'cyber_pro')
    # Same drive to isolate repository identity from the split-drive defect.
    child.drive_root = parent.drive_root
    assert _call(child)['subject']['root'] == str(copy)
    refused = _call(parent)
    assert refused.get('dispatch_refusal', {}).get('kind') == 'review_cycles_exhausted', refused
    assert len(_starts()) == 1


@pytest.mark.parametrize('surface', ['change', 'preflight'])
@pytest.mark.parametrize('legacy', [False, True, 'lost'])
def test_child_cold_rejoin_survives_drive_removal_at_cap(tree, monkeypatch, surface, legacy):
    import shutil
    import threading
    from ouroboros import review_custody, review_pause
    from ouroboros.observability import read_call_payload
    from ouroboros.review_ledger import read_source, record_sources_resolvable
    from ouroboros.review_state import save_state, AdvisoryReviewState

    parent, child, project = tree
    entered, release, settled = threading.Event(), threading.Event(), threading.Event()
    get_run, settle = FakeGateway.get_run, review_custody._settle_review_attempt
    timeout = review_custody._late_or_timeout_actor

    def delayed(self, *args, **kwargs):
        entered.set()
        assert release.wait(15)
        return get_run(self, *args, **kwargs)

    def completed(*args, **kwargs):
        settle(*args, **kwargs)
        settled.set()

    def admitted_timeout(*args, **kwargs):
        assert entered.wait(15)
        return timeout(*args, **kwargs)

    monkeypatch.setattr(FakeGateway, 'get_run', delayed)
    monkeypatch.setattr(review_custody, '_settle_review_attempt', completed)
    monkeypatch.setattr(review_custody, '_late_or_timeout_actor', admitted_timeout)
    monkeypatch.setattr(review_custody, '_logical_timeout', lambda *_: .1)
    ask = {'surface': surface, 'reviewers': ['independent']}
    try:
        original = _call(child, **ask)
        assert original['state'] == 'pending' and len(_starts()) == 1, original
        [attempt] = load_state(parent.drive_root).attempts
        operation = attempt.triad_raw_results[0]['operation_id']
        assert attempt.triad_raw_results[0]['pending_invocation_id']
        before = read_call_payload(parent.drive_root, task_id=child.task_id, call_id=operation + '_prompt')
        assert before[0] and before[1]
        checkout = Path(original['subject']['retained_checkout'])
        assert checkout.is_relative_to(parent.drive_root) and checkout.is_dir()
        release.set()
        assert settled.wait(15)
        review_custody._ACTIVE.clear()
        review_custody._NO_RESEND.clear()
        if legacy is True:
            # A pre-update child drive had the sole attempt row. Model that
            # persistence geometry using an actual paid public operation.
            old = load_state(parent.drive_root)
            old.attempts[0].review_budget_repo_key = ''
            save_state(child.drive_root, old)
            save_state(parent.drive_root, AdvisoryReviewState())
        else:
            shutil.rmtree(child.drive_root, ignore_errors=True)
        if legacy == 'lost':
            from ouroboros.observability import call_manifest_path
            call_manifest_path(parent.drive_root, child.task_id, operation + '_prompt').unlink()
        set_review_pool(monkeypatch, pool_roster(pool_seat(
            'replacement', 'fake-review=different', kind='agent_session', access='full')))
        monkeypatch.setattr('ouroboros.claudexor_daemon.ensure_owned_gateway',
                            lambda: pytest.fail('a cold collection bought a new session'))
        restarted = dataclasses.replace(child)
        result = _call(restarted, **ask)
        if legacy == 'lost':
            # Without its assignment proof the seat's outcome is unknown and it is never re-sent
            # (custody_lost is terminal): the wave settles with no verdict instead of staying
            # pending forever, and nothing new is bought.
            assert (result['state'], result['aggregate']) == ('settled', 'QUORUM_FAILED'), result
            assert result['record_id'] == original['record_id'] and len(_starts()) == 1
            again = _call(dataclasses.replace(child), **ask)  # the same round rejoins at $0
            assert (again['state'], again['aggregate'], again['record_id']) == (
                'settled', 'QUORUM_FAILED', original['record_id']), again
            [paid] = load_state(parent.drive_root).attempts
            assert paid.paid and paid.triad_raw_results[0]['operation_id'] == operation
            assert len(_starts()) == 1
            return
        assert (result['state'], result['aggregate'], result['record_id']) == (
            'settled', 'PASS', original['record_id']), result
        assert len(_starts()) == 1
        [paid] = load_state(parent.drive_root).attempts
        assert paid.paid and not paid.late_result_pending
        assert paid.triad_raw_results[0]['operation_id'] == operation
        assert not load_state(child.drive_root).attempts  # moved, never mirrored
        assert read_call_payload(parent.drive_root, task_id=child.task_id, call_id=operation + '_prompt') == before
        record = load_record(parent.drive_root, original['record_id'])
        assert record_sources_resolvable(parent.drive_root, record)
        retained = [ref for row in record['rows'] for ref in row['source_refs'] if ref['status'] == 'retained']
        assert 'response' in {ref['role'] for ref in retained}
        prompts = [ref for ref in record['rows'][0]['source_refs'] if ref['role'] == 'observability_prompt']
        assert prompts and prompts[0]['ref']['call_id'] == operation + '_prompt', record['rows'][0]
        assert all(read_source(parent.drive_root, child.task_id, ref) for ref in retained)
        assert not checkout.exists()
        # The addressed-wait finalizer has to find the same attempt after the
        # author drive is gone; this read-only check emits no message.
        left = {'surface': 'change', 'drive_root': str(child.drive_root),
                'canonical_root': str(parent.drive_root), 'task_id': child.task_id,
                'retry_key': paid.review_retry_key, 'record_id': original['record_id']}
        assert review_pause._left_attempt(left).review_record_id == original['record_id']
        assert review_pause._left_roster_settled(left)
        refused = _call(restarted, **{**ask, 'reviewers': ['replacement']},
                        review_rebuttal='A new round still costs a cycle')
        assert (refused.get('dispatch_refusal') or {}).get('kind') == 'review_cycles_exhausted', refused
        assert len(_starts()) == 1
    finally:
        release.set()
        if entered.is_set():
            assert settled.wait(15)


def test_copy_sources_and_paid_owner_survive_execution_gc(tree, tmp_path):
    import shutil
    from ouroboros.observability import read_call_payload
    from ouroboros.review_ledger import read_source
    from ouroboros.subagent_worktrees import remove_worktree
    parent, child, project = tree
    copy = _own_copy(parent, child, project, tmp_path)
    result = _call(child)
    [attempt] = load_state(parent.drive_root).attempts
    operation = attempt.triad_raw_results[0]['operation_id']
    prompt = read_call_payload(parent.drive_root, task_id=child.task_id, call_id=operation + '_prompt')
    response = read_call_payload(parent.drive_root, task_id=child.task_id, call_id=operation + '_response')
    assert result['subject']['root'] == attempt.repo_key == str(copy)
    assert attempt.review_budget_repo_key == str(project)
    assert remove_worktree(task_id=child.task_id, worktree_root=tmp_path / 'copies', data_dir=parent.drive_root)
    shutil.rmtree(child.drive_root, ignore_errors=True)
    assert not copy.exists() and not child.drive_root.exists()
    record = load_record(parent.drive_root, result['record_id'])
    assert record['subject']['root'] == str(copy)  # no rewriting source identity after GC
    retained = [r for row in record['rows'] for r in row['source_refs'] if r['status'] == 'retained']
    assert retained and all(read_source(parent.drive_root, child.task_id, r) for r in retained)
    assert prompt[0] and response[0]
    assert read_call_payload(parent.drive_root, task_id=child.task_id, call_id=operation + '_prompt') == prompt
    assert read_call_payload(parent.drive_root, task_id=child.task_id, call_id=operation + '_response') == response
    refused = _call(parent)
    assert (refused.get('dispatch_refusal') or {}).get('kind') == 'review_cycles_exhausted', refused
    assert len(_starts()) == 1


def test_cycle_pools_keep_repo_tool_and_root_partitions(tree, tmp_path):
    import shutil
    from types import SimpleNamespace
    from ouroboros.tools.commit_gate import count_paid_review_cycles
    parent, child, project = tree
    assert _call(child)['aggregate'] == 'PASS'
    gate = SimpleNamespace(drive_root=parent.drive_root, repo_dir=project,
                           _current_review_tool_name='commit_reviewed')
    assert count_paid_review_cycles(gate, root_task_id=parent.task_id) == 0
    # Identical Git history and remote without host copy provenance is NOT a
    # shared paid owner. A different task tree is independently counted too.
    unrelated = tmp_path / 'unregistered-repo'
    shutil.copytree(project, unrelated)
    other = dataclasses.replace(parent, workspace_root=unrelated)
    assert _call(other)['aggregate'] == 'PASS'
    next_tree = dataclasses.replace(parent, task_id='different-root',
                                    task_metadata={'root_task_id': 'different-root'})
    assert _call(next_tree, goal='New task tree')['aggregate'] == 'PASS'
    assert len(_starts()) == 3
    refused = _call(parent, goal='Original tree is still capped')
    assert (refused.get('dispatch_refusal') or {}).get('kind') == 'review_cycles_exhausted', refused
    assert len(_starts()) == 3


def test_exact_sequential_cap_and_free_replay(tree, monkeypatch):
    parent, child, _ = tree
    monkeypatch.setenv('OUROBOROS_REVIEW_MAX_CYCLES', '3')
    results = [_call(ctx, goal=f'Round {i}') for i, ctx in enumerate((child, parent, child))]
    assert all(result['aggregate'] == 'PASS' for result in results) and len(_starts()) == 3
    assert _call(child, goal='Round 2')['reused'] is True
    refused = _call(parent, goal='Round 3', review_rebuttal='New evidence still pays')
    assert (refused.get('dispatch_refusal') or {}).get('kind') == 'review_cycles_exhausted', refused
    assert len(_starts()) == len([a for a in load_state(parent.drive_root).attempts if a.paid]) == 3


@pytest.mark.parametrize('constraint', ['readonly', 'invalid', 'disabled'])
def test_restricted_children_never_dispatch(tree, constraint):
    _, child, _ = tree
    if constraint == 'readonly':
        child.task_constraint = TaskConstraint(mode='local_readonly_subagent')
    elif constraint == 'invalid':
        child.task_constraint = TaskConstraint(mode='acting_subagent')
    else:
        child.task_contract = {'disabled_tools': ['review_change']}
    reg = _registry(child)
    assert reg.get_schema_by_name('review_change') is None
    assert 'BLOCKED' in reg.execute('review_change', {'subject': 'index'})
    assert not _starts()


def test_review_grants_no_commit_control_or_self_approval(tree):
    from ouroboros.tools.tool_discovery import bind_resident_schemas

    _, child, _ = tree
    reg = _registry(child)
    # #1860: the child carries the schema selector, which loads only schemas it may already call.
    assert reg.get_schema_by_name('enable_tools') is not None
    resident = []
    bind_resident_schemas(reg, resident)  # as the child's loop binds it
    for tool in ('commit_reviewed', 'vcs_commit_reviewed', 'request_restart',
                 'promote_to_stable', 'skill_review'):
        assert reg.get_schema_by_name(tool) is None, tool
        assert 'BLOCKED' in reg.execute(tool, {}), tool
        selected = reg.execute_result('enable_tools', {'tools': tool})  # hidden by policy, or an unloadable alias
        assert selected.status in ('blocked', 'error') and '✅' not in selected.text, (tool, selected.text)
        assert reg.get_schema_by_name(tool) is None and 'BLOCKED' in reg.execute(tool, {}), tool
    assert not resident
    allowed = reg.schemas()[0]['function']['name']  # and a schema the role may call does load
    assert 'Call schemas loaded' in reg.execute('enable_tools', {'tools': allowed})
    assert [schema['function']['name'] for schema in resident] == [allowed]
    assert not _starts()


def test_cyber_keeps_foreign_and_system_review_authority(tree, monkeypatch):
    parent, child, _ = tree
    from ouroboros.tools import review_change
    monkeypatch.setenv('OUROBOROS_RUNTIME_MODE', 'cyber_pro')
    (parent.repo_dir / 'README.md').write_text('A change in the serving body\n')
    shared.git(parent.repo_dir, 'add', 'README.md')
    reg = _registry(child)
    assert 'system' in reg.get_schema_by_name('review_change')['function']['parameters']['properties']['subject']['enum']
    result = _call(child, workspace_root=str(parent.repo_dir))
    assert result['subject']['root'] == str(parent.repo_dir) and len(_starts()) == 1
    calls = []
    monkeypatch.setattr(review_change, 'run_system_review',
                        lambda *_a, **_k: calls.append(True) or {'message': '', 'report': 'whole-system'})
    result = _call(child, subject='system', surface='system')
    assert result['report'] == 'whole-system' and calls == [True]


def test_interrupted_legacy_move_is_idempotent_on_public_retry(tree):
    from ouroboros.review_state import save_state
    parent, child, _ = tree
    first = _call(child)
    # A crash after canonical save, before clearing the old row. The resumed
    # call must retain canonical settlement and remove only the legacy copy.
    save_state(child.drive_root, load_state(parent.drive_root))
    again = _call(dataclasses.replace(child))
    assert again['record_id'] == first['record_id'] and again['reused']
    assert len(_starts()) == len(load_state(parent.drive_root).attempts) == 1
    assert not load_state(child.drive_root).attempts


@pytest.mark.parametrize('fence', ['pause', 'stop'])
def test_owner_fence_still_precedes_child_review(tree, fence):
    from ouroboros.owner_pause import install_fence
    from ouroboros.task_results import write_task_result
    parent, child, _ = tree
    write_task_result(parent.drive_root, parent.task_id, 'running', result='working')
    if fence == 'pause':
        install_fence(parent.drive_root, parent.task_id, request_id='test-pause')
    else:
        from ouroboros.cancel_intents import request_cancel
        request_cancel(parent.drive_root, child.task_id, reason='owner stop', source='test')
    from ouroboros.model_wait import task_model_wait_scope
    with task_model_wait_scope(task={'id': child.task_id, 'metadata': child.task_metadata, '_attempt': 1},
                               drive_root=child.drive_root, event_queue=None, worker_slot_held=False):
        if fence == 'stop':
            from ouroboros.llm_attempt import PhysicalDispatchInterrupted
            with pytest.raises(PhysicalDispatchInterrupted, match='cancelled'):
                _registry(child).execute('review_change', {'subject': 'index'})
        else:
            answer = _registry(child).execute('review_change', {'subject': 'index'})
            assert 'BLOCKED' in answer or 'NOT_STARTED' in answer, answer
    assert not _starts() and not load_state(parent.drive_root).attempts


def test_shared_money_fence_still_precedes_child_review(tree, monkeypatch):
    from ouroboros import usage_accounting as ua
    parent, child, _ = tree
    # The existing dollar wave gate prices API seats. Subscription sessions have
    # their own provider admission, so exercise the API consumer here.
    set_review_pool(monkeypatch, pool_roster(pool_seat('independent', 'test/reviewer', delivery='native')))
    monkeypatch.setattr(ua, '_reservation_cost', lambda *_a, **_kw: 1.)
    monkeypatch.setattr('ouroboros.review_substrate.run_review_request',
                        lambda *_a, **_kw: pytest.fail('exhausted money admitted a reviewer'))
    scope = ua.UsageScope(drive_root=str(parent.drive_root), task_id=parent.task_id,
                          root_task_id=parent.task_id, root_limit_usd=1., global_limit_usd=10.)
    with ua.usage_scope(scope):
        ua.execute_physical_attempt(ua.AttemptRequest(model='fake', provider='test', reservation_usd=1.),
                                    lambda: {}, extractor=lambda _: ({'prompt_tokens': 1}, 1., True))
    with ua.usage_scope(dataclasses.replace(scope, task_id=child.task_id, parent_task_id=parent.task_id)):
        result = _call(child)
    assert result['aggregate'] == 'NOT_DISPATCHED', result
    assert (result.get('dispatch_refusal') or {}).get('kind') == 'review_wave_budget_insufficient', result
    assert not _starts() and not load_state(parent.drive_root).attempts


def test_ordinary_body_copy_uses_its_tree_and_serving_governance(tree, tmp_path):
    parent, child, _ = tree
    copy = _own_copy(parent, child, parent.repo_dir, tmp_path)
    before = shared.git(copy, 'rev-parse', 'HEAD')
    result = _call(child)
    assert result['aggregate'] == 'PASS' and result['checklist']['layer'] == 'body', result
    assert result['subject']['root'] == str(copy)
    assert result['subject']['governance_root'] == str(parent.repo_dir)
    assert FakeGateway.reviewed[0]['tree'] == result['subject']['tree_sha']
    assert 'child-only change' in FakeGateway.reviewed[0]['diff']
    assert shared.git(copy, 'rev-parse', 'HEAD') == before


def test_paid_count_does_not_reread_copy_registry_for_unrelated_history(tree, monkeypatch):
    from ouroboros import subagent_worktrees
    from ouroboros.review_state import AdvisoryReviewState, save_state
    from ouroboros.review_state_records import CommitAttemptRecord
    from ouroboros.tools.commit_gate import count_paid_review_cycles

    parent, child, project = tree
    rows = [CommitAttemptRecord(
        ts='2026-10-17T00:00:00+00:00', commit_message='retained legacy wave',
        task_id=f'old-{i}', root_task_id=parent.task_id if i < 3 else 'another-root',
        repo_key=str(project), tool_name='review_change', paid=True, attempt=i + 1,
        status='reviewed', phase='review',
    ) for i in range(23)]
    save_state(parent.drive_root, AdvisoryReviewState(attempts=rows))
    calls, reader = [], subagent_worktrees.list_worktrees

    def observed(*args, **kwargs):
        calls.append(True)
        return reader(*args, **kwargs)

    monkeypatch.setattr(subagent_worktrees, 'list_worktrees', observed)
    child.repo_dir = project
    child._current_review_tool_name = 'review_change'
    assert count_paid_review_cycles(child, root_task_id=parent.task_id) == 3
    assert len(calls) == 1  # one owner resolution, not one full registry read per historical row


def test_the_acting_work_order_offers_the_review_its_tools_admit():
    """An obedient child reads its frozen work order, not the registry: the order must
    not forbid the optional ask, and only the acting branch names it."""
    from ouroboros.tool_capabilities import ACTING_SUBAGENT_TOOL_NAMES
    from supervisor.events_subagent_admission import _compose_subagent_text

    acting = _compose_subagent_text('obj', role='builder', expected_output='out', constraints='', context='',
                                    task_constraint={'mode': 'acting_subagent', 'surface': 'self_worktree',
                                                     'write_root': '/assigned'})
    readonly = _compose_subagent_text('obj', role='researcher', expected_output='out', constraints='', context='')
    assert 'review_change' in ACTING_SUBAGENT_TOOL_NAMES
    assert 'Do NOT commit, run review gates / runtime / skills lifecycle' in acting
    assert 'run review / ' not in acting
    assert ('optionally ask independent reviewers about work in your write root with review_change; '
            'it never approves, commits or integrates that work.') in acting
    assert 'review_change' not in readonly


def test_child_terminal_evidence_reads_its_canonical_change_attempts(tree, monkeypatch):
    """The terminal collector shows the child's own review_change wave, held canonically,
    beside its drive's other review rows; the parent's wave stays the parent's, and a
    stale legacy duplicate yields to the canonical row without anything being moved."""
    import copy
    from types import SimpleNamespace

    from ouroboros.agent_task_pipeline import _collect_review_evidence_bundle
    from ouroboros.review_state import update_state
    from ouroboros.review_state_records import CommitAttemptRecord

    parent, child, project = tree
    monkeypatch.setenv('OUROBOROS_REVIEW_MAX_CYCLES', '2')
    assert _call(child)['aggregate'] == _call(parent, goal='Parent round')['aggregate'] == 'PASS'
    [own] = [row for row in load_state(parent.drive_root).attempts if row.task_id == child.task_id]
    local = CommitAttemptRecord(ts='2026-10-17T00:00:00+00:00', commit_message='child-local gate', task_id=child.task_id,
                                repo_key=str(project), tool_name='commit_reviewed', status='blocked',
                                block_reason='child-local gate row')
    Path(child.drive_root).mkdir(parents=True, exist_ok=True)
    update_state(child.drive_root, lambda state: state.attempts.append(local))
    env = SimpleNamespace(drive_root=child.drive_root, repo_dir=str(project))
    task = {'id': child.task_id, 'budget_drive_root': str(parent.drive_root)}

    def shown():
        rows = _collect_review_evidence_bundle(env, child, task, Path(child.drive_root), [])['recent_attempts']
        return ([(row['triad_raw_results'][0]['operation_id'], row['status'])
                 for row in rows if row['tool_name'] == 'review_change'],
                [row['block_reason'] for row in rows if row['tool_name'] == 'commit_reviewed'])

    expected = ([(own.triad_raw_results[0]['operation_id'], own.status)], ['child-local gate row'])
    assert shown() == expected  # its own wave, never the parent's; the drive's row kept
    stale = copy.deepcopy(own)
    stale.status, stale.late_result_pending = 'reviewing', True
    update_state(child.drive_root, lambda state: state.attempts.append(stale))
    assert shown() == expected  # the canonical row wins its legacy duplicate
    assert len(load_state(child.drive_root).attempts) == 2  # a read moves nothing
    (Path(parent.drive_root) / 'state' / 'advisory_review.json').write_text('{unreadable')
    unread = _collect_review_evidence_bundle(env, child, task, Path(child.drive_root), [])
    assert unread['canonical_attempts_unreadable'] and unread['has_evidence'], unread  # not "none"
    assert shown()[1] == expected[1]


def test_cyber_child_review_status_lists_its_canonical_change_attempts(tree, tmp_path, monkeypatch):
    """A split-drive Cyber child's review_status (its full catalog) lists its own
    review_change wave, held canonically, beside its drive's other rows. The repo, tool,
    task and attempt filters and the physical reviewed root decide as before, the
    canonical settlement wins a legacy duplicate, a read moves nothing, and an
    unreadable canonical state is named, never shown as no attempts."""
    import copy as deep

    from ouroboros.review_state import update_state
    from ouroboros.review_state_records import CommitAttemptRecord

    parent, child, project = tree
    monkeypatch.setenv('OUROBOROS_RUNTIME_MODE', 'cyber_pro')
    monkeypatch.setenv('OUROBOROS_REVIEW_MAX_CYCLES', '2')
    copy = _own_copy(parent, child, project, tmp_path)
    assert _call(child)['aggregate'] == _call(parent, goal='Parent round')['aggregate'] == 'PASS'
    [own] = [row for row in load_state(parent.drive_root).attempts if row.task_id == child.task_id]
    assert (own.repo_key, own.review_budget_repo_key) == (str(copy), str(project))
    local = CommitAttemptRecord(ts='2026-10-17T00:00:00+00:00', commit_message='child-local gate',
                                task_id=child.task_id, repo_key=str(copy), tool_name='commit_reviewed',
                                status='blocked', attempt=1, block_reason='child-local gate row')
    update_state(child.drive_root, lambda state: state.attempts.append(local))

    def status(**args):
        text = _registry(child).execute('review_status', args)
        return json.loads(text[text.index('{'):])

    def rows(**args):
        return sorted((row['tool_name'], row['status']) for row in status(**args)['attempts'])

    mine = [('commit_reviewed', 'blocked'), ('review_change', 'reviewed')]
    assert rows(repo_key=str(copy)) == mine
    assert rows(repo_key=str(copy), tool_name='review_change', attempt=own.attempt) == mine[1:]
    assert rows(repo_key=str(copy), tool_name='commit_reviewed') == mine[:1]
    selected = status(repo_key=str(copy), tool_name='review_change')['selected_commit_attempt']
    assert (selected['task_id'], selected['status']) == (child.task_id, 'reviewed')
    # The parent's wave stays the parent's; the copy's paid pool is not its reviewed root.
    assert rows(repo_key=str(copy), task_id=parent.task_id) == rows(repo_key=str(project)) == []
    stale = deep.deepcopy(own)
    stale.status, stale.late_result_pending = 'reviewing', True
    update_state(child.drive_root, lambda state: state.attempts.append(stale))
    canonical = load_state(parent.drive_root).attempts
    assert rows(repo_key=str(copy)) == mine  # the canonical settlement wins its legacy duplicate
    assert len(load_state(child.drive_root).attempts) == 2 and load_state(parent.drive_root).attempts == canonical
    (Path(parent.drive_root) / 'state' / 'advisory_review.json').write_text('{unreadable')
    unread = status(repo_key=str(copy))
    assert unread['canonical_attempts_unreadable'] and 'unreadable' in unread['status_summary'], unread
    assert ('commit_reviewed', 'blocked') in rows(repo_key=str(copy))  # this drive's rows still shown
