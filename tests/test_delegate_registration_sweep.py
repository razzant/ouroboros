"""Registration-sweep lifecycle: sharer-aware deferral and discharge.

Split from test_delegated_run_isolation.py (module line cap)."""
from __future__ import annotations

import itertools

from ouroboros import delegate_custody as custody


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

        for run_id in order:
            row = custody.replay(root)[run_id]
            custody.settle_run(root, gateway, row, {"summary": {"state": "succeeded"}})

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
    dc.retire_settled_registrations(tmp_path, gateway)
    assert gateway.removals == [], "a live unowned sharer defers the attempt"

    # Sibling settles: the very next sweep discharges the registration.
    dc.emit(tmp_path, dc.SETTLED, {"run_id": "run-b", "task_id": "t-b", "route": "r"})
    dc._CUSTODY.clear()
    dc.retire_settled_registrations(tmp_path, gateway)
    assert gateway.removals == ["prj-shared"]
