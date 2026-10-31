"""Authority failure through the real completion transports.

Adapted from the retained independent Opus consumer probe (15 Oct 2026).

The final contribution exercises published Audit04 together with the current
Telegram producer. Real authority output, not a synthetic payload: run_llm_loop -> _handle_authority_unavailable -> derive_loop_outcome ->
terminal_projection._project_row -> Telegram _check_tasks_notify ON/OFF.
The pause case checks that a saved technical pause never becomes a settled terminal row.
"""
import asyncio
import copy
from types import SimpleNamespace

import pytest

from ouroboros import outcomes, project_dialogue as dialogue, terminal_projection
from ouroboros import usage_accounting as ua
from tests.test_billing_authority_unavailable import BINDING, _loop_driver, _run, _task_scope
from tests.test_telegram_notifications import _api, _load, _Rec, _write_rows


def _throw(*_a, **_k):
    raise OSError("store refuses writes")


def _produce(case, root, monkeypatch):
    from ouroboros import budget_pause, queue_observation
    from ouroboros.task_results import write_task_result

    if case == "refused":
        loop_mod, registry, _ = _loop_driver(root, monkeypatch, "orphan")
        with ua.usage_scope(_task_scope(root, "orphan", "gone-root")):
            return "orphan", _run(loop_mod, registry, root, "orphan")
    if case == "unreadable_no_owner":
        write_task_result(root, "solo", "running", billing_group={"billing_group_id": "solo"})
        loop_mod, registry, _ = _loop_driver(root, monkeypatch, "solo", exact_owner=False)
        with ua.usage_scope(_task_scope(root, "solo", "solo")):
            return "solo", _run(loop_mod, registry, root, "solo")
    if case == "unreadable_store_broken":
        write_task_result(root, "brk", "running", billing_group={"billing_group_id": "brk"})
        monkeypatch.setattr(budget_pause, "_HOLD_POLL_SEC", 0.01)
        monkeypatch.setattr(budget_pause, "set_budget_pause", _throw)
        loop_mod, registry, _ = _loop_driver(root, monkeypatch, "brk")
        with ua.usage_scope(_task_scope(root, "brk", "brk")):
            return "brk", _run(loop_mod, registry, root, "brk")
    if case == "fence_store_broken":
        write_task_result(root, "fence", "running", billing_group={**BINDING, "billing_group_id": "fence"})
        (root / "state").mkdir(exist_ok=True)
        (root / "state" / "queue_dispatch.json").write_text("{}", encoding="utf-8")
        monkeypatch.setattr(queue_observation, "dispatch_probe", lambda *_a, **_k: None)
        monkeypatch.setattr(budget_pause, "_HOLD_POLL_SEC", 0.01)
        monkeypatch.setattr(budget_pause, "set_budget_pause", _throw)
        loop_mod, registry, _ = _loop_driver(root, monkeypatch, "fence")
        with ua.usage_scope(_task_scope(root, "fence", "fence")):
            return "fence", _run(loop_mod, registry, root, "fence")
    raise AssertionError(case)


EXPECTED_CODE = {
    "refused": "billing_authority_refused",
    "unreadable_no_owner": "billing_authority_unreadable",
    "unreadable_store_broken": "billing_authority_unreadable",
    "fence_store_broken": "control_authority_unreadable",
}


@pytest.mark.parametrize("case", list(EXPECTED_CODE))
def test_actual_authority_fallback_reaches_phone_as_failure_not_money_or_review(case, tmp_path, monkeypatch):
    tid, (text, usage, trace) = _produce(case, tmp_path, monkeypatch)
    code = EXPECTED_CODE[case]
    assert (usage["execution_status"], usage["reason_code"]) == ("infra_failed", code), usage.get("reason_code")
    assert "not a budget limit" in text

    outcome = outcomes.derive_loop_outcome(text, usage, trace)
    assert outcome["outcome_axes"]["execution"]["status"] == "infra_failed"
    # The durable row status the pipeline writes for an infra_failed execution.
    record = {**outcome, "task_id": tid, "status": "failed", "chat_id": 1}
    before = copy.deepcopy(record)
    summary = terminal_projection._project_row(tid, record, {}, {"chat_id": 1})
    assert record == before
    print("\nCASE", case, "PHASE", summary["outcome_phase"], "NOTICE", summary["completion_notice"],
          "\nREASON_DETAIL", summary.get("reason_detail"))
    assert summary["outcome_phase"] == "error"
    notice = summary["completion_notice"]
    assert notice["neutral"] is False
    assert dialogue.TASK_CAUSE_PHRASES[code] in notice["detail"]  # human phrase, not the raw code
    assert "budget exhaustion was not established" in notice["detail"]
    assert "ran out of budget" not in notice["detail"].lower()

    nt = _load()
    api, data = _api(tmp_path / "tg")
    monkeypatch.setattr(nt, "TelegramClient", _Rec)
    _write_rows(data, [summary])
    _Rec.sent = []
    off = {"notified_task_ids": []}
    asyncio.run(nt._check_tasks_notify(api, {"TELEGRAM_NOTIFY_TASKS": "off"}, 42, off, "en"))
    assert not _Rec.sent and off["notified_task_ids"] == [tid]  # OFF: suppressed and consumed
    asyncio.run(nt._check_tasks_notify(api, {"TELEGRAM_NOTIFY_TASKS": "on"}, 42, off, "en"))
    assert not _Rec.sent  # re-enabling does not replay the OFF observation
    asyncio.run(nt._check_tasks_notify(api, {"TELEGRAM_NOTIFY_TASKS": "on"}, 42, {"notified_task_ids": []}, "en"))
    assert len(_Rec.sent) == 1
    message = _Rec.sent[0][1]
    print("PHONE", repr(message))
    assert message.startswith("❌") and "failed" in message
    assert "Review and verification details are in the app" not in message  # not the neutral template
    assert "budget exhaustion was not established" in message
    assert "Budget:" not in message and "ran out of budget" not in message.lower()


def test_saved_technical_pause_is_not_a_settled_terminal_row(tmp_path, monkeypatch):
    from ouroboros import budget_pause
    from ouroboros.task_results import load_task_result, write_task_result
    from supervisor.events import _handle_budget_pause
    from tests._budget_pause_exact_helpers import _install_queue, _supervisor_ctx

    queue, state, workers = _install_queue(tmp_path, monkeypatch)
    monkeypatch.setattr(state, "budget_remaining", lambda *_a, **_k: 5.0)
    monkeypatch.setattr(budget_pause, "_HOLD_POLL_SEC", 0.01)
    write_task_result(tmp_path, "tech", "running", billing_group={"billing_group_id": "tech"})
    loop_mod, registry, _ = _loop_driver(tmp_path, monkeypatch, "tech")
    with ua.usage_scope(_task_scope(tmp_path, "tech", "tech")), pytest.raises(budget_pause.BudgetPauseRequested) as paused:
        _run(loop_mod, registry, tmp_path, "tech")
    pause = paused.value.pause
    budget_pause.end_dispatch_fence("tech")
    task = {"id": "tech", "type": "task", "chat_id": 1, "root_task_id": "tech", "_attempt": 1}
    workers.RUNNING["tech"] = {"task": task, "worker_id": 0, "attempt": 1}
    workers.WORKERS[0] = SimpleNamespace(busy_task_id="tech")
    _handle_budget_pause({**budget_pause.pause_event(task, pause), "worker_id": 0},
                         _supervisor_ctx(tmp_path, workers, queue, [], []))
    row = load_task_result(tmp_path, "tech")
    print("\nPAUSED_ROW status", row.get("status"), "reason_code", row.get("reason_code"))
    assert row["reason_code"] == "authority_paused"
    assert not terminal_projection._settled(row)  # no task_summary is ever owed for it
    assert dialogue.outcome_phase(row, {}) == "working"  # the notifier skips working rows
