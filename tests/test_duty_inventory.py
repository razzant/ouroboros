"""The reconciled duty inventory: one declaration the supervisor tick starts from.

``server_maintenance.HOST_DUTIES`` names every recurring pass the tick starts off-loop (its journal
name, single-flight latch and one-pass runner). The cost contract's rest law
(tests/test_cost_contract.py B7) reads the same table, so a pass that exists without a row is a
pass nobody measures: that is how 22.10's re-offered captures stayed invisible to the instrument.
Both directions are held here: a tick with every due condition true starts exactly the declared
names (a pass started without a row fails, a row that names a pass the tick never starts fails),
and the terminal-duty switch runs exactly the rows that declare it as their runner.
"""

from __future__ import annotations

import ast
import inspect
import threading
from typing import List

from ouroboros import server_maintenance as sm


def _started_by_one_tick(monkeypatch) -> List[str]:
    """Drive one maintenance tick with every due condition true; the names it starts, in order."""
    from ouroboros import maintenance
    from supervisor import continuation_admission, queue

    names: List[str] = []

    def start(latch, name, target, args):
        names.append(name)
        latch.release()  # the tick took it; the pass it would start is what releases it
        return True

    monkeypatch.setattr(sm, "_start_maintenance_thread", start)
    monkeypatch.setattr(sm, "_tick_maintenance_child", lambda: names.append("maintenance-child"))
    monkeypatch.setattr(sm, "_LAST_CANCEL_INTENT_SWEEP", [0.0])
    monkeypatch.setattr(sm, "_LAST_COLD_CUSTODY_SWEEP", [0.0])
    monkeypatch.setattr(queue, "RUNNING", {"cur-run": {"task": {"id": "cur-run"}}})  # an actor: cold custody is due
    monkeypatch.setattr(queue, "BUDGET_ROOT_FENCES", {"cur-paused": {}})  # a fence: owner-pause settlement is due
    monkeypatch.setattr(continuation_admission, "held_continuation_predecessors", lambda pending: ["cur-pred"])
    # A child of this test's own: owed by construction (a process's first tick), never the process
    # singleton an earlier test's server shutdown stopped (``server._emergency_process_cleanup``).
    monkeypatch.setattr(maintenance, "child", maintenance.MaintenanceChild())
    sm._periodic_supervisor_maintenance([0.0], [0.0])
    return names


def _undeclared_or_unstarted(started: List[str], declared: List[str]) -> set:
    return set(started) ^ set(declared)


def test_one_tick_starts_every_declared_duty_and_nothing_else(monkeypatch):
    started = _started_by_one_tick(monkeypatch)
    assert len(started) == len(set(started)), f"a duty started twice in one tick: {started}"
    assert not _undeclared_or_unstarted(started, [duty.name for duty in sm.HOST_DUTIES]), (started, sm.HOST_DUTIES)


def test_the_inventory_check_fails_in_both_directions(monkeypatch):
    """A pass the tick starts without a row, and a row that names a pass the tick never starts."""
    started = _started_by_one_tick(monkeypatch)
    declared = [duty.name for duty in sm.HOST_DUTIES]
    assert not _undeclared_or_unstarted(started, declared)
    assert _undeclared_or_unstarted(started, declared[:-1]) == {declared[-1]}  # started, undeclared
    assert _undeclared_or_unstarted(started, [*declared, "evidence-recheck"]) == {"evidence-recheck"}  # declared, never started


def test_the_terminal_duty_switch_runs_exactly_the_rows_that_name_it(monkeypatch):
    """``_run_terminal_duty`` runs a duty by name: the names it accepts are the rows whose runner it is;
    a name outside the table is a failed step (journaled, latch released), never a pass."""
    tree = ast.parse(inspect.getsource(sm._run_terminal_duty))
    accepted = {node.comparators[0].value for node in ast.walk(tree)
                if isinstance(node, ast.Compare) and isinstance(node.left, ast.Name) and node.left.id == "step"
                and isinstance(node.comparators[0], ast.Constant)}
    declared = {duty.name for duty in sm.HOST_DUTIES if duty.run is sm._run_terminal_duty}
    assert accepted == declared, (accepted ^ declared)
    monkeypatch.setattr(sm, "_STEP_FAILURES", {})
    latch = threading.Lock()
    latch.acquire()
    sm._run_terminal_duty(None, "evidence-recheck", latch, None)
    assert sm._STEP_FAILURES == {"evidence-recheck": 1} and not latch.locked()


def test_every_row_names_a_latch_and_a_runner():
    seen = set()
    for duty in sm.HOST_DUTIES:
        assert duty.name and duty.name not in seen
        seen.add(duty.name)
        assert duty.latch is None or isinstance(duty.latch, type(threading.Lock())), duty
        assert callable(duty.run), duty
    assert {duty.name for duty in sm.TWENTY_SECOND_DUTIES} < seen
