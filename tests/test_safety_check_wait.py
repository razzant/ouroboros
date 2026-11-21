"""A tool call waiting on its Safety check is one host wait episode (Track09 #2863, owner A, 9 Nov 2026).

``safety_records.safety_check_wait`` writes ``entered`` before the first attempt and ``ended`` however the
check ends; the readers fold it like every host wait (``checkpoint_waits``). Presentation only: the check's
answer, failure and cancellation pass through untouched.
"""

from __future__ import annotations

import pathlib
from types import SimpleNamespace

import pytest

from ouroboros import safety, safety_records
from ouroboros.checkpoint_waits import fold_progress, wait_episode


@pytest.fixture
def emitted(monkeypatch):
    rows = []
    import ouroboros.loop_messages as loop_messages

    monkeypatch.setattr(loop_messages, "_emit_checkpoint_event",
                        lambda queue, task_id, logs, data: rows.append({"task_id": task_id, "logs": logs, **data}))
    return rows


def ctx(tmp_path, **extra):
    return SimpleNamespace(event_queue=object(), task_id="t1", drive_root=tmp_path, current_chat_id=7, **extra)


def call(context, client_result, monkeypatch):
    import ouroboros.llm_observability as observability

    def chat_observed(_client, **_kw):
        if isinstance(client_result, BaseException):
            raise client_result
        return client_result

    monkeypatch.setattr(observability, "chat_observed", chat_observed)
    return safety._safety_model_call(
        client=object(), ctx=context, tool_name="run_command", light_model="m", use_local=False,
        call_type="safety_check", user_prompt="u", on_usage=lambda _usage: None, model_role="test",
        system_prompt="s", latch=False)


def test_an_answered_check_is_one_episode_entered_then_ended(tmp_path, emitted, monkeypatch):
    assert call(ctx(tmp_path), ({"content": "{}"}, {}), monkeypatch) == ({"content": "{}"}, {})
    assert [row["phase"] for row in emitted] == ["entered", "ended"]
    entered, ended = emitted
    assert entered["episode_id"] == ended["episode_id"] and entered["checkpoint_kind"] == "safety_check_wait"
    assert (entered["text"], ended["text"]) == ("Checking safety", "Safety check ended")
    assert entered["tool"] == "run_command" and entered["chat_id"] == 7 and entered["task_id"] == "t1"
    assert ended["logs"] == pathlib.Path(tmp_path) / "logs" and ended["elapsed_sec"] >= entered["elapsed_sec"] >= 0
    assert wait_episode(ended) == ("safety_check_wait", ended["episode_id"], True, ended["elapsed_sec"])


@pytest.mark.parametrize("failure", [RuntimeError("provider down"), KeyboardInterrupt()])
def test_a_failed_or_cancelled_check_still_ends_its_episode_and_keeps_its_exception(tmp_path, emitted, monkeypatch, failure):
    with pytest.raises(type(failure)):
        call(ctx(tmp_path), failure, monkeypatch)
    assert [row["phase"] for row in emitted] == ["entered", "ended"]
    assert emitted[0]["episode_id"] == emitted[1]["episode_id"]


def test_a_latched_check_makes_no_attempt_and_no_episode(tmp_path, emitted, monkeypatch):
    monkeypatch.setattr(safety, "_SAFETY_STORM_UNTIL", 10 ** 12)
    with pytest.raises(safety._SafetyRateLimited):
        safety._safety_model_call(client=object(), ctx=ctx(tmp_path), tool_name="x", light_model="m", use_local=False,
                                  call_type="c", user_prompt="u", on_usage=lambda _u: None, model_role="test")
    assert emitted == []


def test_concurrent_checks_keep_their_own_identities_and_fold_to_one_line(tmp_path, emitted):
    context = ctx(tmp_path)
    with safety_records.safety_check_wait(context, "a"):
        with safety_records.safety_check_wait(context, "b"):
            pass
    first, second = emitted[0]["episode_id"], emitted[1]["episode_id"]
    assert first != second
    assert [(row["episode_id"], row["phase"]) for row in emitted] == [
        (first, "entered"), (second, "entered"), (second, "ended"), (first, "ended")]
    assert [entry["text"] for entry in fold_progress(emitted[:3])] == ["Checking safety · Safety check ended · 0.0 s"]


def test_no_task_no_episode_and_a_failing_writer_never_fails_the_check(tmp_path, emitted, monkeypatch):
    for context in (None, SimpleNamespace(event_queue=None, task_id="t"), SimpleNamespace(event_queue=object(), task_id="")):
        with safety_records.safety_check_wait(context, "x"):
            pass
    assert emitted == []
    import ouroboros.loop_messages as loop_messages

    def broken(*_a, **_k):
        raise OSError("disk full")

    monkeypatch.setattr(loop_messages, "_emit_checkpoint_event", broken)
    with safety_records.safety_check_wait(ctx(tmp_path), "x"):
        reached = True
    assert reached
