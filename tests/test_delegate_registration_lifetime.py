"""Engine registration lifetime follows the host's continuation authority."""
from types import SimpleNamespace

import pytest

from ouroboros import delegate_collection, delegate_custody as custody
from ouroboros.delegate_continuation import bind_continuation
from ouroboros.owner_continue import binding_sha
from tests._store_shared import seed_result, unreadable_result


class Gateway:
    def __init__(self):
        self.removed = []

    def handshake(self):
        return {}

    def remove_project(self, project_id):
        self.removed.append(project_id)

    def get_run(self, run_id):  # a still-owed receipt is re-read by the ordinary collector
        return {"summary": {"state": "cancelled", "spendUsd": 0, "spendEstimated": False}}

    def close(self):
        pass


def result(root, tid, **fields):
    path = root / 'task_results' / f'{tid}.json'
    path.parent.mkdir(exist_ok=True)
    seed_result(root, tid, {'_schema_version': 1, 'task_id': tid,
                               'status': 'completed', **fields})


def seed(root, *, task_id='owner', root_task_id='', source='', project='project', run_id='run', invocation_id=''):
    row = custody.RunCustody(run_id=run_id, task_id=task_id, root_task_id=root_task_id, invocation_id=invocation_id,
                            route_id='route', model='model', project_id=project,
                            project_owned=True, source=source, ledger_root=str(root))
    assert custody.record_started(root, row, shape={'access': 'readonly', 'mode': 'ask'})
    return row


def settle(root, gateway, row):
    assert custody.settle_run(root, gateway, row, {'summary': {
        'state': 'cancelled', 'spendUsd': 0, 'spendEstimated': False}})['settled']


def sweep(root, gateway, live):
    custody.reconcile_orphaned_runs(root, live, gateway_factory=lambda: gateway)


@pytest.mark.parametrize('terminal', [
    {'status': 'completed'},
    {'status': 'cancelled', 'cancel_origin': {'source': 'owner_stop'}},
    {'status': 'failed', 'reason_code': 'budget_exhausted'},
])
def test_live_owner_can_continue_after_settlement_then_normal_finish_retires_once(tmp_path, terminal):
    gateway = Gateway()
    row = seed(tmp_path)
    result(tmp_path, 'owner', status='running')
    settle(tmp_path, gateway, row)
    assert gateway.removed == []
    sweep(tmp_path, gateway, {'owner'})
    assert gateway.removed == []
    facts, refusal, _ = bind_continuation(SimpleNamespace(task_id='owner'), tmp_path, 'run',
        actor={}, route=SimpleNamespace(route_id='route'),
        authority=SimpleNamespace(access='readonly'), target_root='')
    assert not refusal and facts['task_line'] == 'own'
    result(tmp_path, 'owner', **terminal)
    sweep(tmp_path, gateway, set())
    sweep(tmp_path, gateway, set())
    assert gateway.removed == ['project']


def test_retry_successor_keeps_predecessor_registration(tmp_path):
    gateway = Gateway()
    row = seed(tmp_path)
    result(tmp_path, 'owner', superseded_by='retry', retry_task_id='retry')
    result(tmp_path, 'retry', status='running', root_task_id='owner',
           supersedes_task_id='owner', original_task_id='owner', timeout_retry_from='owner')
    settle(tmp_path, gateway, row)
    sweep(tmp_path, gateway, {'retry'})
    assert gateway.removed == []
    result(tmp_path, 'retry', root_task_id='owner', supersedes_task_id='owner',
           original_task_id='owner', timeout_retry_from='owner')
    sweep(tmp_path, gateway, set())
    assert gateway.removed == ['project']


def test_timeout_retried_owner_releases_once_its_retry_finishes(tmp_path):
    # The production shape of a timeout retry (task_reaper): the predecessor row is
    # rewritten `interrupted` naming its retry, and only the retry settles later.
    gateway = Gateway()
    row = seed(tmp_path)
    result(tmp_path, 'owner', status='interrupted', reason_code='timeout_retry',
           superseded_by='retry', retry_task_id='retry')
    result(tmp_path, 'retry', status='running', supersedes_task_id='owner',
           original_task_id='owner', timeout_retry_from='owner')
    settle(tmp_path, gateway, row)
    sweep(tmp_path, gateway, {'retry'})
    assert gateway.removed == []
    result(tmp_path, 'retry', supersedes_task_id='owner', original_task_id='owner',
           timeout_retry_from='owner')
    sweep(tmp_path, gateway, set())
    assert gateway.removed == ['project']


@pytest.mark.parametrize('leaf, kept', [
    ({'status': 'failed', 'reason_code': 'worker_crash_signal'}, True),
    ({'status': 'completed', 'reason_code': 'round_limit'}, True),
    ({'status': 'completed'}, False),
])
def test_timeout_retried_owner_follows_the_card_offer_of_its_broken_retry(tmp_path, leaf, kept):
    # Before startup recovery heals the raw row, the card already offers Continue
    # from the retry leaf's technical end; the sweep must read the same offer.
    gateway = Gateway()
    row = seed(tmp_path)
    result(tmp_path, 'owner', status='interrupted', reason_code='timeout_retry',
           superseded_by='retry', retry_task_id='retry')
    result(tmp_path, 'retry', root_task_id='owner', supersedes_task_id='owner',
           original_task_id='owner', timeout_retry_from='owner', **leaf)
    settle(tmp_path, gateway, row)
    sweep(tmp_path, gateway, set())
    assert gateway.removed == ([] if kept else ['project'])


def test_root_offer_and_recorded_continue_chain_keep_child_run(tmp_path):
    gateway = Gateway()
    row = seed(tmp_path, task_id='child', root_task_id='root')
    result(tmp_path, 'child', parent_task_id='root', root_task_id='root')
    result(tmp_path, 'root', status='failed', reason_code='worker_crash_signal')
    settle(tmp_path, gateway, row)
    sweep(tmp_path, gateway, set())
    assert gateway.removed == []
    binding = {'predecessor_task_id': 'root', 'successor_task_id': 'next'}
    result(tmp_path, 'root', status='failed', reason_code='worker_crash_signal', continued_by={
        'successor_task_id': 'next', 'state': 'admitted', 'binding': binding,
        'binding_sha256': binding_sha(binding)})
    result(tmp_path, 'next', status='running')
    sweep(tmp_path, gateway, {'next'})
    assert gateway.removed == []
    result(tmp_path, 'next')
    sweep(tmp_path, gateway, set())
    assert gateway.removed == ['project']


@pytest.mark.parametrize('broken', [
    {'status': 'failed', 'reason_code': 'worker_crash_signal'},
    {'status': 'completed', 'reason_code': 'round_limit'},
    {'status': 'cancelled', 'cancel_origin': {'source': 'server_shutdown', 'reason': 'server_shutdown'}},
])
def test_technically_broken_continue_successor_keeps_its_offer(tmp_path, broken):
    # The owner pressed Continue, and the successor itself broke before it
    # continued the delegated run: its own card offers Continue again, so the
    # run's project must survive until that offer closes.
    gateway = Gateway()
    row = seed(tmp_path, task_id='child', root_task_id='root')
    result(tmp_path, 'child', parent_task_id='root', root_task_id='root')
    binding = {'predecessor_task_id': 'root', 'successor_task_id': 'next'}
    result(tmp_path, 'root', status='failed', reason_code='worker_crash_signal', continued_by={
        'successor_task_id': 'next', 'state': 'admitted', 'binding': binding,
        'binding_sha256': binding_sha(binding)})
    result(tmp_path, 'next', root_task_id='next', **broken)
    settle(tmp_path, gateway, row)
    sweep(tmp_path, gateway, set())
    assert gateway.removed == []
    result(tmp_path, 'next', root_task_id='next')  # an ordinary finish closes the offer
    sweep(tmp_path, gateway, set())
    assert gateway.removed == ['project']


@pytest.mark.parametrize('authority', ['missing', 'unreadable', 'unknown_live', 'reserved'])
def test_unknown_authority_or_reserved_owner_keeps_registration(tmp_path, monkeypatch, authority):
    gateway = Gateway()
    row = seed(tmp_path)
    if authority != 'missing':
        result(tmp_path, 'owner')
    if authority == 'unreadable':
        unreadable_result(monkeypatch, 'owner')
        # Its bucket is unreadable too: the receipt stays owed in custody, never a stand-in row.
        assert not custody.settle_run(tmp_path, gateway, row, {'summary': {
            'state': 'cancelled', 'spendUsd': 0, 'spendEstimated': False}})['settled']
    else:
        settle(tmp_path, gateway, row)
    custody.reconcile_orphaned_runs(tmp_path, None if authority == 'unknown_live' else set(),
        gateway_factory=lambda: gateway,
        recoverable_handles={(row.task_id, row.run_id, row.invocation_id)} if authority == 'reserved' else set())
    assert gateway.removed == []
    assert custody.replay(tmp_path)['run'].project_owned


@pytest.mark.parametrize('handle', [('owner', 'run-a', 'inv-a'), ('owner', 'run-a', '')])
def test_reserved_handle_keeps_only_its_registration_until_released(tmp_path, handle):
    gateway = Gateway()
    first = seed(tmp_path, run_id='run-a', invocation_id='inv-a', project='project-a')
    second = seed(tmp_path, run_id='run-b', invocation_id='inv-b', project='project-b')
    result(tmp_path, 'owner')
    for row in (first, second):
        settle(tmp_path, gateway, row)
    custody.reconcile_orphaned_runs(tmp_path, set(), gateway_factory=lambda: gateway,
                                   recoverable_handles={handle})
    assert gateway.removed == ['project-b']
    assert custody.replay(tmp_path)['run-a'].project_owned
    sweep(tmp_path, gateway, set())
    assert gateway.removed == ['project-b', 'project-a']


@pytest.mark.parametrize('handle', [('other', 'run', 'inv'), ('owner', 'run', 'other')])
def test_foreign_or_mismatched_invocation_reservation_does_not_keep_registration(tmp_path, handle):
    gateway = Gateway()
    row = seed(tmp_path, invocation_id='inv')
    result(tmp_path, 'owner')
    settle(tmp_path, gateway, row)
    custody.reconcile_orphaned_runs(tmp_path, set(), gateway_factory=lambda: gateway,
                                   recoverable_handles={handle})
    assert gateway.removed == ['project']


def test_review_settlement_keeps_immediate_retirement(tmp_path):
    gateway = Gateway()
    row = seed(tmp_path, source='review_substrate')
    settle(tmp_path, gateway, row)
    assert gateway.removed == ['project']


def _counting_full_reads(monkeypatch):
    """The project's sharers are ONE indexed store read per retire-eligible project and tick;
    the chain is never folded on the sweep."""
    from ouroboros import delegate_custody_current as current

    calls = []
    original = current.runs

    def counted(*args, **kwargs):
        if len(args) > 1 and args[1] == "project_id=?":
            calls.append(1)
        return original(*args, **kwargs)
    monkeypatch.setattr(current, 'runs', counted)
    monkeypatch.setattr(custody, '_iter_rows', lambda *a, **kw: pytest.fail('chain fold on the sweep'))
    return calls


def test_continuable_owned_project_never_pays_the_full_chain_read(tmp_path, monkeypatch):
    gateway = Gateway()
    row = seed(tmp_path)
    result(tmp_path, 'owner', status='failed', reason_code='worker_crash_signal')
    settle(tmp_path, gateway, row)
    # Terminal collection belongs to the body after readiness, not to the orphan sweep.
    delegate_collection.collect_terminal(tmp_path, 'run', gateway=gateway)
    calls = _counting_full_reads(monkeypatch)
    sweep(tmp_path, gateway, set())
    sweep(tmp_path, gateway, set())
    assert gateway.removed == [] and calls == []
    assert custody.replay(tmp_path)['run'].output_source, 'the body retained the terminal output'
    result(tmp_path, 'owner')  # Closing the Continue offer makes retirement eligible.
    sweep(tmp_path, gateway, set())
    assert gateway.removed == ['project'] and calls


def test_shared_project_kept_by_a_settled_sharer_reads_the_full_chain_once(tmp_path, monkeypatch):
    # The production sweep reads the held rows; a settled sharer that does not
    # own the registration is reached by one indexed read of the project's runs.
    from ouroboros import delegate_custody_current as current

    gateway = Gateway()
    owner = seed(tmp_path, task_id='a')
    sharer = custody.RunCustody(run_id='run-b', task_id='b', route_id='route', model='model',
                                project_id='project', project_owned=False, ledger_root=str(tmp_path))
    assert custody.record_started(tmp_path, sharer, shape={'access': 'readonly', 'mode': 'ask'})
    result(tmp_path, 'a')
    result(tmp_path, 'b', status='failed', reason_code='worker_crash_signal')
    settle(tmp_path, gateway, owner)
    settle(tmp_path, gateway, sharer)
    # Settlement retains an observation; the body (not the orphan sweep) finishes collection,
    # so the unowned sharer really is outside the held projection whose lookup this measures.
    for run_id in (owner.run_id, sharer.run_id):
        delegate_collection.collect_terminal(tmp_path, run_id, gateway=gateway)
    sweep(tmp_path, gateway, set())
    assert set(current.runs(tmp_path, 'state=?', ('held',))) == {owner.run_id}
    calls = _counting_full_reads(monkeypatch)
    sweep(tmp_path, gateway, set())
    first = len(calls)
    sweep(tmp_path, gateway, set())
    assert gateway.removed == []
    assert first == 1 and len(calls) == 2  # one indexed read per tick while the sharer keeps the project
    result(tmp_path, 'b')  # an ordinary finish closes the sharer's offer
    sweep(tmp_path, gateway, set())
    assert gateway.removed == ['project']
