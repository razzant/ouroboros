"""Policy vocabulary lives in work_stop; point refusals survive queue publication."""
import ast
import copy
from pathlib import Path

import pytest

from ouroboros.queue_observation import load_queue_document
from ouroboros.work_stop import STOP_REASONS, continuation_refusal
from supervisor import events_budget as eb
from tests._budget_pause_exact_helpers import _install_queue


def test_release_consumers_do_not_branch_on_hold_names():
    root = Path(__file__).resolve().parents[1]
    names = set(STOP_REASONS) - {"owner", "budget", "authority", "sleep"}
    symbols = {key for key, value in vars(eb).items() if key.startswith("HOLD_") and value in names}
    symbols.add("HOLD_SAVED_SLEEP_RECOVERY")
    # Transport/source mechanisms (sleep, exact owner custody) are not release
    # names. UI and mind use the same table too.
    offenders = []
    for directory in ("supervisor", "ouroboros"):
        for path in (root / directory).rglob("*.py"):
            if path.name in {"work_stop.py", "work_stop_migration.py"}:
                continue
            source = path.read_text(encoding="utf-8")
            assert "SELECTABLE_HOLD_REASONS" not in source and "OWNER_ROOT_RESUME_HOLDS" not in source, path
            tree = ast.parse(source)
            for function in ast.walk(tree):
                if not isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                for node in ast.walk(function):
                    if not isinstance(node, (ast.Compare, ast.IfExp)):
                        continue
                    condition = node.test if isinstance(node, ast.IfExp) else node
                    if any(isinstance(item, ast.Constant) and isinstance(item.value, str) and item.value in names
                           or isinstance(item, ast.Name) and item.id in symbols for item in ast.walk(condition)):
                        offenders.append(f"{path.relative_to(root)}:{node.lineno}")
    assert not offenders, offenders


def test_refusal_and_stop_keep_distinct_snapshot_custody(tmp_path, monkeypatch):
    q, _state, workers = _install_queue(tmp_path, monkeypatch)
    task = {"id": "root", "_budget_pause": {"reason": "sleep", "checkpoint": {"pause_id": "p"}}}
    eb.hold_budget_row(task, reason=eb.HOLD_PANIC)
    before = copy.deepcopy(eb.budget_hold_fact(task))
    eb.hold_budget_row(task, reason=eb.HOLD_RECORD_UNREADABLE_AT_REVOKE,
                       extra={"pause_id": "p", "grant_id": "g"})
    workers.PENDING.append(task)
    assert q.persist_queue_snapshot(reason="point_refusal_contract")
    [row] = [item["task"] for item in load_queue_document(tmp_path)["pending"]]
    assert eb.budget_hold_fact(row) == before
    assert continuation_refusal(row)["pause_id"] == "p" and continuation_refusal(row)["grant_id"] == "g"
    assert row["_budget_pause"]["checkpoint"]["pause_id"] == "p"


def test_tick_redecides_spent_hold_without_reissuing_resume(tmp_path, monkeypatch):
    from supervisor.budget_resume import revalidate_consumed_resume_holds
    from ouroboros.task_results import write_task_result
    from ouroboros.cancel_intents import has_active_intent

    q, _state, workers = _install_queue(tmp_path, monkeypatch)
    for tid, basis in (("continuing", True), ("duplicate", False)):
        write_task_result(tmp_path, tid, "running", task_attempt=1)
        task = {"id": tid, "_attempt": 2}
        if basis:
            task["_working_recovery"] = {"source_task_id": tid, "from_attempt": 1, "cause": "update_aborted"}
        eb.hold_budget_row(task, reason=eb.HOLD_GRANT_CONSUMED_STALE_ROW, extra={"pause_id": "p", "grant_id": "g"})
        workers.PENDING.append(task)
    revalidate_consumed_resume_holds()
    assert not continuation_refusal(workers.PENDING[0])
    assert continuation_refusal(workers.PENDING[1])
    assert not has_active_intent(tmp_path, "continuing") and has_active_intent(tmp_path, "duplicate")
    assert all("_budget_pause_resume" not in task for task in workers.PENDING)
    assert not continuation_refusal(load_queue_document(tmp_path)["pending"][0]["task"])


@pytest.mark.parametrize("populated", [False, True])
def test_tick_without_spent_holds_does_no_store_work(tmp_path, monkeypatch, populated):
    from ouroboros import budget_pause, delegate_recovery, store
    from supervisor.budget_resume import revalidate_consumed_resume_holds

    q, _state, workers = _install_queue(tmp_path, monkeypatch)
    if populated:
        workers.PENDING.extend([
            {"id": "ordinary", "_attempt": 2},
            {"id": "other-hold", eb.BUDGET_HOLD_KEY: {"reason": eb.HOLD_OWNER_RESTART}},
            {"id": "exact-pause", "_budget_pause": {"exact_continuation": True},
             eb.BUDGET_HOLD_KEY: {"reason": eb.HOLD_STALE_GRANT_SUPERSEDED}},
        ])
    before = copy.deepcopy(workers.PENDING)

    def unexpected(*_args, **_kwargs):
        pytest.fail("a tick without spent holds must not read/write durable state")

    monkeypatch.setattr(store, "open_store", unexpected)
    monkeypatch.setattr(budget_pause, "budget_pause_row", unexpected)
    monkeypatch.setattr(delegate_recovery, "_read", unexpected)
    monkeypatch.setattr(q, "persist_queue_snapshot", unexpected)
    with q._queue_lock:
        revalidate_consumed_resume_holds()
        revalidate_consumed_resume_holds()
    assert workers.PENDING == before


def test_the_assignment_tick_entry_re_decides_spent_holds(tmp_path, monkeypatch):
    """``revalidate_project_holds`` (the assignment pass's hold revalidation) runs the re-decision first,
    so a stranded continuation returns on the next tick even when no Project hold is pending."""
    from ouroboros.task_results import write_task_result
    from supervisor.task_admission import revalidate_project_holds

    q, _state, workers = _install_queue(tmp_path, monkeypatch)
    write_task_result(tmp_path, "continuing", "running", task_attempt=1)
    task = {"id": "continuing", "_attempt": 2,
            "_working_recovery": {"source_task_id": "continuing", "from_attempt": 1, "cause": "update_aborted"}}
    eb.hold_budget_row(task, reason=eb.HOLD_GRANT_CONSUMED_STALE_ROW, extra={"pause_id": "p", "grant_id": "g"})
    workers.PENDING.append(task)
    assert not any(row.get("_project_admission_restore_hold") for row in workers.PENDING)
    with q._queue_lock:
        revalidate_project_holds()
    assert not continuation_refusal(workers.PENDING[0])
