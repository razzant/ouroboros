"""Regression tests for the provider-death owner-notification single-shot gate.

Slime-saga TASK 3 settled a disputed claim by test: the old ``and task`` gate in
``_finish_task_done_dispatch`` claimed reaper-only delivery still notifies, but
the reaper LOOP pops RUNNING BEFORE ``reap_timed_out_task`` emits its task_done
(that function's docstring: "The loop already popped RUNNING/cleared
busy_task_id"), so a reaper-delivered provider-death terminal arrived with
``task={}`` and the notification was silently swallowed — the external
reviewer's claim was CORRECT. The fix keys single-shot on the process-local
``_PROVIDER_DEATH_NOTIFIED`` registry: a duplicate ``already_done`` terminal
stays silent, a reaper-delivered terminal fires.
"""

from __future__ import annotations

import types

import pytest

from supervisor import events as events_mod
from supervisor import events_task_done as task_done_mod


@pytest.fixture()
def sent_and_ctx(tmp_path, monkeypatch):
    monkeypatch.setattr(task_done_mod, "_PROVIDER_DEATH_NOTIFIED", set())
    sent: list[tuple[int, str]] = []

    def make_ctx(running):
        return types.SimpleNamespace(
            DRIVE_ROOT=tmp_path, RUNNING=running, PENDING=[], WORKERS={},
            send_with_budget=lambda cid, text, **_k: sent.append((cid, str(text))),
            append_jsonl=lambda *_a, **_k: None,
            persist_queue_snapshot=lambda **_k: True,
            bridge=types.SimpleNamespace(push_log=lambda _e: None),
        )

    return sent, make_ctx


def _provider_death_event(task_id: str) -> dict:
    return {
        "type": "task_done", "task_id": task_id, "chat_id": 7,
        "status": "failed", "reason_code": "provider_unavailable",
    }


def _outage_lines(sent):
    return [text for _cid, text in sent if "provider outage" in text]


def test_duplicate_already_done_after_normal_delivery_notifies_exactly_once(
    sent_and_ctx,
):
    """Path (a): the worker delivered its own task_done (RUNNING row present,
    notification fires and the dispatch releases the row), died, and the crash
    detector emitted a second already_done terminal whose dispatch sees
    task={} — the owner is notified exactly once."""
    sent, make_ctx = sent_and_ctx
    root_task = {"id": "rootA", "chat_id": 7}
    running = {"rootA": {"task": root_task, "worker_id": 0}}

    events_mod._finish_task_done_dispatch(
        {}, make_ctx(running),
        task_id="rootA", worker_id=0, task=root_task, final_task_result={},
        task_done_event=_provider_death_event("rootA"),
    )
    assert len(_outage_lines(sent)) == 1
    assert "rootA" not in running

    # The duplicate already_done terminal: RUNNING no longer holds the row.
    events_mod._finish_task_done_dispatch(
        {}, make_ctx({}),
        task_id="rootA", worker_id=0, task={}, final_task_result={},
        task_done_event=_provider_death_event("rootA"),
    )
    assert len(_outage_lines(sent)) == 1, "duplicate terminal must stay silent"


def test_reaper_delivered_terminal_with_popped_running_row_still_notifies(
    sent_and_ctx,
):
    """Path (b), the proven bug: the reaper loop pops RUNNING before the reap
    job's task_done dispatches, so the FIRST and only delivery arrives with
    task={} — the old `and task` gate swallowed the notification entirely.
    Also pins the wording: the resume endpoint only serves budget-paused
    PENDING tasks, so the message promises re-run and never resume."""
    sent, make_ctx = sent_and_ctx

    events_mod._finish_task_done_dispatch(
        {}, make_ctx({}),
        task_id="rootB", worker_id=0, task={}, final_task_result={},
        task_done_event=_provider_death_event("rootB"),
    )

    lines = _outage_lines(sent)
    assert lines, "reaper-delivered provider-death terminal must notify the owner"
    assert "NOT completed" in lines[0]
    assert "re-run" in lines[0]
    assert "resume" not in lines[0]


def test_raising_send_keeps_cleanup_and_allows_a_later_retry(sent_and_ctx):
    """A raising ``send_with_budget`` must not abort the task-done bookkeeping,
    and the id must NOT enter the single-shot registry (a later dispatch may
    retry the notification); the success path stays single-shot."""
    sent, make_ctx = sent_and_ctx
    root_task = {"id": "rootF", "chat_id": 7}
    running = {"rootF": {"task": root_task, "worker_id": 0}}
    ctx = make_ctx(running)

    def _boom(_cid, _text, **_k):
        raise RuntimeError("chat transport down")

    ctx.send_with_budget = _boom
    events_mod._finish_task_done_dispatch(
        {}, ctx,
        task_id="rootF", worker_id=0, task=root_task, final_task_result={},
        task_done_event=_provider_death_event("rootF"),
    )
    assert "rootF" not in running, "cleanup must run despite the failed send"
    assert "rootF" not in task_done_mod._PROVIDER_DEATH_NOTIFIED

    # A later dispatch (e.g. the duplicate already_done terminal) retries and
    # registers the id only now, on the successful send.
    events_mod._finish_task_done_dispatch(
        {}, make_ctx({}),
        task_id="rootF", worker_id=0, task={}, final_task_result={},
        task_done_event=_provider_death_event("rootF"),
    )
    assert len(_outage_lines(sent)) == 1
    assert "rootF" in task_done_mod._PROVIDER_DEATH_NOTIFIED


def test_reaper_delivered_child_terminal_stamps_parent_activity(sent_and_ctx):
    """The parent activity stamp must land even when ``task`` is {} (the
    reaper-delivered popped-RUNNING shape): ``parent_task_id`` falls back to
    the durable ``final_task_result``, same as the notification gate."""
    _sent, make_ctx = sent_and_ctx
    parent_meta = {"task": {"id": "rootG"}, "worker_id": 0}
    running = {"rootG": parent_meta}

    events_mod._finish_task_done_dispatch(
        {}, make_ctx(running),
        task_id="kidG", worker_id=1, task={},
        final_task_result={"parent_task_id": "rootG", "delegation_role": "subagent"},
        task_done_event=_provider_death_event("kidG"),
    )

    assert "last_progress_at" in parent_meta, (
        "a reaper-delivered child terminal must count as the parent's progress"
    )


def test_ephemeral_decision_turn_gets_no_duplicate_outage_ping(sent_and_ctx):
    """An ephemeral direct-chat decision turn already shows its failure inline;
    the provider-outage owner ping must stay silent and leave the registry
    untouched."""
    sent, make_ctx = sent_and_ctx
    root_task = {"id": "rootH", "chat_id": 7}

    events_mod._finish_task_done_dispatch(
        {"_ephemeral": True}, make_ctx({"rootH": {"task": root_task, "worker_id": 0}}),
        task_id="rootH", worker_id=0, task=root_task, final_task_result={},
        task_done_event=_provider_death_event("rootH"),
    )

    assert not _outage_lines(sent)
    assert "rootH" not in task_done_mod._PROVIDER_DEATH_NOTIFIED


def test_subagent_provider_death_never_pings_the_owner(sent_and_ctx):
    """A child's provider death keeps the ordinary subagent toast only — the
    parent absorbs child failures; the registry gate must not change that."""
    sent, make_ctx = sent_and_ctx
    child_task = {
        "id": "kidE", "chat_id": 7, "parent_task_id": "rootE",
        "root_task_id": "rootE", "delegation_role": "subagent",
    }

    events_mod._finish_task_done_dispatch(
        {"status": "failed"}, make_ctx({"kidE": {"task": child_task, "worker_id": 1}}),
        task_id="kidE", worker_id=1, task=child_task, final_task_result={},
        task_done_event=_provider_death_event("kidE"),
    )

    assert not _outage_lines(sent)
    assert [text for _cid, text in sent if "Subagent kidE failed" in text]
