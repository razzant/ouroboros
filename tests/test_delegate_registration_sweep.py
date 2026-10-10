"""Registration-sweep lifecycle: sharer-aware deferral and discharge.

Split from test_delegated_run_isolation.py (module line cap)."""
from __future__ import annotations

import itertools
import json

from ouroboros import delegate_custody as custody



def _finished_owner(root, task_id):
    path = root / 'task_results' / f'{task_id}.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({'_schema_version': 1, 'task_id': task_id,
                               'status': 'completed'}), encoding='utf-8')

def test_last_shared_project_sibling_retires_once_in_every_settlement_order(tmp_path):
    class _Gateway:
        def __init__(self):
            self.removals = []

        def remove_project(self, project_id):
            self.removals.append(project_id)

    run_ids = ("run-a", "run-b", "run-c")
    for case, order in enumerate(itertools.permutations(run_ids)):
        root = tmp_path / str(case)
        gateway = _Gateway()
        custody._CUSTODY.clear()
        for index, run_id in enumerate(run_ids):
            custody.record_started(root, custody.RunCustody(
                run_id=run_id,
                task_id=f"task-{run_id}",
                project_id="shared-project",
                project_owned=index == 0,
                ledger_root=str(root),
            ))
            custody.emit(root, custody.LEDGER_RECORDED, {"run_id": run_id})
            _finished_owner(root, f"task-{run_id}")

        for run_id in order:
            row = custody.replay(root)[run_id]
            custody.settle_run(root, gateway, row, {"summary": {"state": "succeeded"}})

        assert gateway.removals == [], order
        custody.retire_settled_registrations(root, gateway, live_task_ids=set())
        assert gateway.removals == ["shared-project"], order
        replayed = custody.replay(root)
        assert all(not row.project_owned for row in replayed.values()), order
        retired = [
            row for row in custody._iter_rows(custody.event_log_path(root))
            if row.get("type") == custody.PROJECT_RETIRED
        ]
        assert len(retired) == 1, order
    custody._CUSTODY.clear()

def test_registration_sweep_defers_behind_a_live_unowned_sharer(tmp_path):
    """Sharers are ALL runs in a project, owned or not: only the creator
    carries the registration, but the daemon refuses removal while any
    sibling lives - attempting anyway spammed PROJECT_RETIRE_FAILED on
    every sweep tick for the sibling's whole lifetime."""
    dc = custody

    class _Gateway:
        def __init__(self):
            self.removals = []

        def handshake(self, **_kw):
            return {}

        def remove_project(self, pid):
            self.removals.append(pid)

        def close(self):
            pass

    gateway = _Gateway()
    dc.record_started(tmp_path, dc.RunCustody(
        run_id="run-a", task_id="t-a", route_id="r", model="m",
        project_id="prj-shared", project_owned=True, ledger_root=str(tmp_path)))
    dc.record_started(tmp_path, dc.RunCustody(
        run_id="run-b", task_id="t-b", route_id="r", model="m",
        project_id="prj-shared", project_owned=False, ledger_root=str(tmp_path)))
    dc._CUSTODY.clear()
    dc.emit(tmp_path, dc.SETTLED, {"run_id": "run-a", "task_id": "t-a", "route": "r"})

    # Owner settled, unowned sibling still live: the sweep must not attempt.
    dc._CUSTODY.clear()
    dc.retire_settled_registrations(tmp_path, gateway, live_task_ids=set())
    assert gateway.removals == [], "a live unowned sharer defers the attempt"

    # Sibling and both owners settle: the next sweep discharges the registration.
    _finished_owner(tmp_path, "t-a")
    _finished_owner(tmp_path, "t-b")
    dc.emit(tmp_path, dc.SETTLED, {"run_id": "run-b", "task_id": "t-b", "route": "r"})
    dc._CUSTODY.clear()
    dc.retire_settled_registrations(tmp_path, gateway, live_task_ids=set())
    assert gateway.removals == ["prj-shared"]


# -- I9: the undeletable engine project stops being retried forever ----------
#
# Ouroboros creates sticky plan-review THREADS scoped to a project root and the
# gateway has no thread delete; the engine then refuses DELETE with the typed
# `{code: "project_has_threads", status: 409}` forever. The duty is discharged
# once under the engine's own code, never recorded as a deletion, and only when
# no unsettled run of the project exists (a replayed PROJECT_RETIRED clears
# `project_owned` for EVERY sibling of the project).


def _rows(root):
    path = root / "logs" / "events.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def _of(root, kind):
    return [row for row in _rows(root) if row.get("type") == kind]


class _RefusingGateway:
    """DELETE answers with a typed refusal; ``on_remove`` runs before it (race hook)."""

    def __init__(self, exc=None, on_remove=None):
        self.exc, self.on_remove, self.removals = exc, on_remove, []

    def handshake(self, **_kw):
        return {}

    def remove_project(self, project_id):
        self.removals.append(project_id)
        if self.on_remove is not None:
            self.on_remove()
        if self.exc is not None:
            raise self.exc

    def close(self):
        pass


def _has_threads():
    from ouroboros.gateways.claudexor import ClaudexorUnavailable

    return ClaudexorUnavailable("project_has_threads", "project prj-kept has 2 threads",
                                status_code=409)


def _start(root, run_id, *, owned, project_id="prj-kept", persistent=False):
    _finished_owner(root, f"t-{run_id}")
    custody.record_started(root, custody.RunCustody(
        run_id=run_id, task_id=f"t-{run_id}", route_id="r", model="m",
        project_id=project_id, project_owned=owned, project_persistent=persistent,
        ledger_root=str(root)))


def _settle(root, run_id):
    custody.emit(root, custody.SETTLED, {"run_id": run_id, "task_id": f"t-{run_id}", "route": "r"})


def test_exact_project_has_threads_with_every_run_settled_discharges_exactly_once(tmp_path):
    gateway = _RefusingGateway(_has_threads())
    _start(tmp_path, "run-a", owned=True)
    _settle(tmp_path, "run-a")
    custody._CUSTODY.clear()

    custody.retire_settled_registrations(tmp_path, gateway, live_task_ids=set())

    assert gateway.removals == ["prj-kept"]
    assert _of(tmp_path, custody.PROJECT_RETIRE_FAILED) == [], "a permanent refusal is not a failure"
    retired = _of(tmp_path, custody.PROJECT_RETIRED)
    assert len(retired) == 1
    row = retired[0]
    assert row["project_kept"] is True, "the engine keeps the project; the row must not claim deletion"
    assert row["reason"] == "project_has_threads"
    assert row["code"] == "project_has_threads" and row["status"] == 409
    assert row["run_id"] == "run-a" and row["project_id"] == "prj-kept"
    custody._CUSTODY.clear()
    assert custody.owned_project_registrations(tmp_path) == [], "the duty is discharged durably"

    # The next idle sweep has no registration to retire: it never touches the gateway.
    def _no_gateway():
        raise AssertionError("an idle sweep must not spawn or wait for the daemon")

    assert custody.reconcile_orphaned_runs(tmp_path, set(), gateway_factory=_no_gateway) == []
    # A second explicit sweep appends nothing and asks the daemon nothing.
    custody._CUSTODY.clear()
    before = len(_rows(tmp_path))
    custody.retire_settled_registrations(tmp_path, gateway, live_task_ids=set())
    assert gateway.removals == ["prj-kept"] and len(_rows(tmp_path)) == before


def test_a_run_started_during_the_refused_delete_blocks_the_discharge(tmp_path):
    """Replay order: STARTED A, STARTED B (open, same project), 409 on A => NO
    discharge — a PROJECT_RETIRED replayed after B's STARTED would strip B's
    ownership. B lands while the DELETE is in flight (STARTED appends take no
    lock), so the precondition must be re-read under the retirement lock."""
    _start(tmp_path, "run-a", owned=True)
    _settle(tmp_path, "run-a")
    custody._CUSTODY.clear()
    gateway = _RefusingGateway(
        _has_threads(), on_remove=lambda: _start(tmp_path, "run-b", owned=True))

    custody.retire_project(tmp_path, gateway, custody.replay(tmp_path)["run-a"], live_task_ids=set())

    assert gateway.removals == ["prj-kept"]
    assert _of(tmp_path, custody.PROJECT_RETIRED) == [], "an unsettled sibling forbids the discharge"
    failed = _of(tmp_path, custody.PROJECT_RETIRE_FAILED)
    assert len(failed) == 1
    assert failed[0]["code"] == "project_has_threads" and failed[0]["status"] == 409
    custody._CUSTODY.clear()
    replayed = custody.replay(tmp_path)
    assert replayed["run-b"].project_owned is True
    assert replayed["run-a"].project_owned is True, "still retryable"
    # B settles: the next sweep retries the same duty and discharges it then.
    _settle(tmp_path, "run-b")
    custody._CUSTODY.clear()
    gateway.on_remove = None
    custody.retire_settled_registrations(tmp_path, gateway, live_task_ids=set())
    retired = _of(tmp_path, custody.PROJECT_RETIRED)
    assert len(retired) == 1 and retired[0]["project_kept"] is True
    custody._CUSTODY.clear()
    assert custody.owned_project_registrations(tmp_path) == []


def test_any_other_refusal_stays_a_retryable_typed_failure(tmp_path):
    from ouroboros.gateways.claudexor import ClaudexorUnavailable

    refusals = [
        ClaudexorUnavailable("project_busy", "project has live runs", status_code=409),
        ClaudexorUnavailable("daemon_unreachable", "socket died", status_code=0),
        ClaudexorUnavailable("malformed_response", "non-JSON body", status_code=0),
        ClaudexorUnavailable("http_500", "project_has_threads", status_code=500),  # prose, not code
        RuntimeError("project_has_threads"),  # not even a typed refusal
    ]
    _start(tmp_path, "run-a", owned=True)
    _settle(tmp_path, "run-a")
    gateway = _RefusingGateway()
    for index, exc in enumerate(refusals, start=1):
        gateway.exc = exc
        custody._CUSTODY.clear()
        custody.retire_settled_registrations(tmp_path, gateway, live_task_ids=set())
        assert len(gateway.removals) == index, "every failure is retried"
        assert _of(tmp_path, custody.PROJECT_RETIRED) == [], f"no discharge for {exc!r}"
        failed = _of(tmp_path, custody.PROJECT_RETIRE_FAILED)
        assert len(failed) == index
        row = failed[-1]
        assert row["code"] == str(getattr(exc, "code", "") or "")
        assert row["status"] == int(getattr(exc, "status_code", 0) or 0)
        assert str(exc) in row["reason"], "the daemon's text still rides the row"
    custody._CUSTODY.clear()
    assert [c.run_id for c in custody.owned_project_registrations(tmp_path)] == ["run-a"]

    # The daemon accepts at last: an ordinary retirement, not a kept project.
    gateway.exc = None
    custody.retire_settled_registrations(tmp_path, gateway, live_task_ids=set())
    retired = _of(tmp_path, custody.PROJECT_RETIRED)
    assert len(retired) == 1 and "project_kept" not in retired[0]
    custody._CUSTODY.clear()
    assert custody.owned_project_registrations(tmp_path) == []


def test_persistent_sharer_branch_unchanged_and_a_new_registration_is_owned_again(tmp_path):
    # #362: a persistent sharer keeps the project without ever asking the daemon.
    gateway = _RefusingGateway(_has_threads())
    _start(tmp_path, "run-a", owned=True)
    _start(tmp_path, "run-p", owned=False, persistent=True)
    _settle(tmp_path, "run-a")
    _settle(tmp_path, "run-p")
    custody._CUSTODY.clear()
    custody.retire_settled_registrations(tmp_path, gateway, live_task_ids=set())
    assert gateway.removals == []
    retired = _of(tmp_path, custody.PROJECT_RETIRED)
    assert len(retired) == 1
    assert set(retired[0]) == {"ts", "type", "run_id", "task_id", "project_id", "project_kept"}
    assert retired[0]["project_kept"] is True

    # A discharged project, then a NEW registration of the same root (same id):
    # replay is sequential, so STARTED after RETIRED is owned again.
    root = tmp_path / "second"
    _start(root, "run-a", owned=True)
    _settle(root, "run-a")
    custody._CUSTODY.clear()
    custody.retire_settled_registrations(root, gateway, live_task_ids=set())
    assert gateway.removals == ["prj-kept"]
    assert [row["run_id"] for row in _of(root, custody.PROJECT_RETIRED)] == ["run-a"]
    _start(root, "run-c", owned=True)
    custody._CUSTODY.clear()
    replayed = custody.replay(root)
    assert replayed["run-a"].project_owned is False
    assert replayed["run-c"].project_owned is True
    assert [c.run_id for c in custody.owned_project_registrations(root)] == ["run-c"]
    # Its own duty is its own: discharged once more, under the engine's code.
    _settle(root, "run-c")
    custody._CUSTODY.clear()
    custody.retire_settled_registrations(root, gateway, live_task_ids=set())
    assert gateway.removals == ["prj-kept", "prj-kept"]
    assert [row["run_id"] for row in _of(root, custody.PROJECT_RETIRED)] == ["run-a", "run-c"]
    custody._CUSTODY.clear()
    assert custody.owned_project_registrations(root) == []
