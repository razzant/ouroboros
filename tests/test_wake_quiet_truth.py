"""Quiet describes the whole wake's outward words, including mid-turn owner sends."""
import json
from types import SimpleNamespace

import pytest

from ouroboros.task_finalization import quiet_finish
from tests._wake_summary_shared import seed_wake


@pytest.mark.parametrize("silent,sent,notice,quiet", [(True, False, "", True),
    (True, True, "", False), (True, False, "A check failed.", False), (False, False, "", False)])
def test_quiet_finish_and_projected_row_use_all_delivery_facts(tmp_path, silent, sent, notice, quiet):
    from ouroboros.terminal_projection import _project_row

    row = seed_wake(tmp_path, wake_silent=silent, owner_message_sent=sent, terminal_host_notice=notice)
    assert quiet_finish(row) is quiet
    projected = _project_row(row["task_id"], row, {}, {"chat_id": 1})
    assert projected["quiet_effective"] is quiet
    assert (projected["text"] == "Done · no message") is quiet


@pytest.mark.parametrize("sent", [True, False])
def test_terminal_settlement_skips_main_mirror_only_for_a_quiet_wake(tmp_path, monkeypatch, sent):
    from ouroboros import headless, project_dialogue, terminal_projection as projection

    row = seed_wake(tmp_path, owner_message_sent=sent, canonical_terminal_projection=None)
    mirrors = []
    monkeypatch.setattr(headless, "terminal_task_files_ready", lambda *a: True)
    monkeypatch.setattr(project_dialogue, "project_completion_delivery_outcome",
        lambda *a, **kw: mirrors.append(1) or (project_dialogue.MAIN_MIRROR_INELIGIBLE, False))
    assert projection.settle_terminal_projection(tmp_path, row["task_id"]) == projection.SETTLEMENT_SETTLED
    assert mirrors == ([1] if sent else [])


@pytest.mark.parametrize("origin", ["Daily check", ""])
@pytest.mark.parametrize("fault", [False, True])
@pytest.mark.parametrize("sent", [False, True])
@pytest.mark.parametrize("summary", ["Проверил состояние системы.", ""])
def test_both_live_terminal_writers_keep_wake_words_and_delivery_truth(tmp_path, monkeypatch, fault, sent, summary, origin):
    from ouroboros import headless
    from supervisor import events, events_task_done

    # A schedule-born run carries its schedule's name; any other run's frame gains no origin key at all.
    fields = {**({"wake_summary": summary} if summary else {}), **({"schedule_name": origin} if origin else {})}
    stored = seed_wake(tmp_path, owner_message_sent=sent, **fields)
    ctx = SimpleNamespace(DRIVE_ROOT=tmp_path, RUNNING={})
    captured = []
    monkeypatch.setattr(headless, "terminal_task_files_ready", lambda *a: True)
    monkeypatch.setattr(events, "_finish_task_done_dispatch", lambda *a, **kw: captured.append(kw["task_done_event"]))
    monkeypatch.setattr(events, "_authoritative_terminal_cost", lambda *a: {})
    event = {"task_id": stored["task_id"], "status": "completed"}
    if fault:
        events_task_done._resolve_lifecycle_fault(event, ctx, "running")
    else:
        events_task_done._handle_task_done(event, ctx)
    assert len(captured) == 1
    assert captured[0]["quiet_effective"] is (not sent)
    assert captured[0].get("wake_summary", "") == summary
    assert captured[0].get("schedule_name", "absent") == (origin or "absent")
    durable = [json.loads(line) for line in (tmp_path / "logs/events.jsonl").read_text(encoding="utf-8").splitlines()]
    done = next(row for row in durable if row.get("type") == "task_done")
    assert done["quiet_effective"] is (not sent)
    assert done.get("wake_summary", "") == summary and done.get("schedule_name", "absent") == (origin or "absent")


@pytest.mark.parametrize("destination", ["current", "main"])
@pytest.mark.parametrize("accepted", [True, False])
def test_owner_send_flag_records_only_accepted_deliveries(tmp_path, monkeypatch, destination, accepted):
    from ouroboros.tools import control_runtime
    from ouroboros import presence_owner_contact

    ctx = SimpleNamespace(current_chat_id=1, task_id="wake", task_metadata={}, drive_logs=lambda: tmp_path)
    monkeypatch.setattr(control_runtime, "_main_notice_refusal", lambda *a: "")
    monkeypatch.setattr(presence_owner_contact, "contact_origin", lambda *a: {"task_id": "wake"})

    def deliver(*a, **kw):
        if not accepted:
            raise OSError("delivery unavailable")
        return "live"

    ctx.event_queue = SimpleNamespace(put_nowait=deliver)
    ctx.pending_events = SimpleNamespace(append=deliver)
    monkeypatch.setattr(presence_owner_contact, "send_main_message", deliver)
    if accepted or destination == "main":
        control_runtime._send_user_message(ctx, "An update.", destination=destination)
    else:
        with pytest.raises(OSError):
            control_runtime._send_user_message(ctx, "An update.", destination=destination)
    assert bool(getattr(ctx, "_owner_message_sent", False)) is accepted
