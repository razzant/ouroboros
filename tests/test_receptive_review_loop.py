"""#1741 receptive review through the REAL Main loop: ``run_llm_loop`` with the real tool
registry, round-top mailbox drain, owner-wait park (``direct_owner_wait``) and continuation
serializer, over the real commit_reviewed/review_change handlers, stage cycle, custody drain
and frozen checkout of ``test_receptive_commit_review`` (its scripted two-seat reviewer is the
only reviewer stand-in). Main is scripted at ``call_llm_with_retry``; every step asserts what
Main actually received, so an unscripted call (an empty polling round) fails the test.
Negative controls switch one seam off and show the old behavior it replaces.
"""
from __future__ import annotations

import copy
import json
import queue
import threading
import time
from types import SimpleNamespace

import pytest

from ouroboros import loop, review_pause
from ouroboros.owner_mailbox import write_owner_message
from ouroboros.owner_wait import direct_owner_wait
from ouroboros.task_results import load_task_result
from ouroboros.tools.registry import ToolRegistry
from tests.test_receptive_commit_review import (  # noqa: F401 -- ``author`` is the shared fixture
    TASK, _commit_count, _git, _notices, _parent, author, until,
)

pytestmark = pytest.mark.serial
COMMIT = {"commit_message": "Change value", "paths": ["change.py"], "skip_advisory_review": True}
REVIEW = {"root": "system_repo", "subject": "index", "goal": "Raise the value.", "scope": "change.py only"}


def call(name, arguments, identifier):
    return {"content": "", "tool_calls": [{"id": identifier, "type": "function",
                                           "function": {"name": name, "arguments": json.dumps(arguments)}}]}


def prose(text):
    return {"content": text}


def _tool_text(messages):
    return str([m for m in messages if m.get("role") == "tool"][-1]["content"])


def _json_tail(text):
    return json.loads(text if text.startswith("{") else text[text.index("\n{") + 1:])


def _reference(messages):
    return _json_tail(_tool_text(messages))["review_reference"]


@pytest.fixture
def main_loop(author, monkeypatch):  # noqa: F811 -- imported pytest fixture
    """The real loop over the receptive author's context, parked by the real direct owner wait."""
    import time as real_time

    a = author
    registry = ToolRegistry(repo_dir=a.repo, drive_root=a.data)
    registry._ctx = a.ctx
    a.ctx.owner_wait_callback = direct_owner_wait
    a.ctx.current_chat_id, a.ctx.is_direct_chat = 7, False
    monkeypatch.setenv("OUROBOROS_TASK_REVIEW_MODE", "off")
    monkeypatch.setenv("MCP_ENABLED", "false")
    monkeypatch.setattr(loop, "_maybe_inject_finalization_nudges", lambda *_a: False)
    # The park's own mailbox poll, only faster (the same code path).
    fast = SimpleNamespace(**{name: getattr(real_time, name) for name in dir(real_time) if not name.startswith("_")})
    fast.sleep = lambda seconds: real_time.sleep(min(seconds, 0.05))
    monkeypatch.setattr("ouroboros.owner_wait.time", fast)
    state = SimpleNamespace(a=a, registry=registry, inputs=[], progress=[], parks=[])
    park = a.ctx.owner_wait_callback

    def recording_park(ctx, checkpoint):
        state.parks.append(dict(checkpoint))
        return park(ctx, checkpoint)
    a.ctx.owner_wait_callback = recording_park

    def run(steps):
        def main(_llm, messages, *_a, **kwargs):
            state.inputs.append(copy.deepcopy(messages))
            index = len(state.inputs) - 1
            assert index < len(steps), f"an unscripted Main call: {str(messages[-1])[:400]}"
            response = steps[index](messages)
            observer = kwargs.get("model_context_observer")
            if callable(observer):
                observer(messages)
            return response, 0.0
        monkeypatch.setattr(loop, "call_llm_with_retry", main)
        with _parent(a) as owner:
            a.ctx.model_wait_context, owner.tool_context = owner, a.ctx
            return loop.run_llm_loop(
                messages=[{"role": "system", "content": "Commit the reviewed change."},
                          {"role": "user", "content": "Commit change.py once its review is in."}],
                tools=registry, llm=SimpleNamespace(default_model=lambda: "fixture/main"),
                drive_logs=a.data / "logs", emit_progress=lambda text, **_k: state.progress.append(text),
                incoming_messages=queue.Queue(), task_id=TASK, drive_root=a.data, event_queue=queue.Queue())
    state.run = run
    return state


def _mail_when_both_seats_started(a, text="How is the commit going?"):
    def owner():
        assert a.reviewer.sent["m1"].wait(60) and a.reviewer.preparing_m2.wait(60)
        write_owner_message(a.data, text, TASK)
    threading.Thread(target=owner, daemon=True).start()


def _release_later(a, delay=0.4):
    threading.Timer(delay, a.reviewer.release).start()


@pytest.mark.parametrize("mode,enforcement", [("pro", "advisory"), ("pro", "blocking"),
                                              ("cyber_pro", "advisory"), ("cyber_pro", "blocking")])
def test_commit_mail_yield_parks_without_polling_then_collects_and_continues(main_loop, monkeypatch, mode, enforcement):
    from ouroboros import config

    config.reset_runtime_mode_baseline_for_tests()
    config.initialize_runtime_mode_baseline(mode)
    monkeypatch.setenv("OUROBOROS_REVIEW_ENFORCEMENT", enforcement)
    f, a = main_loop, main_loop.a
    gate_blocks = mode == "pro" and enforcement == "blocking"

    def started(messages):
        _mail_when_both_seats_started(a)
        return call("commit_reviewed", COMMIT, "commit-1")

    def yielded(messages):
        text = _tool_text(messages)
        if gate_blocks:
            assert "REVIEW_PENDING" in text and "An addressed message ended your wait" in text, text
        else:
            assert text.startswith("Review outcome returned before commit: reviewers are still running."), text
            assert "An addressed message reached you" in text
        assert "How is the commit going?" in str(messages[-3:]), "the addressed message reached Main"
        assert _commit_count(a) == 1 and not a.reviewer.gates["m1"].is_set()
        _release_later(a)
        return prose("The commit review is still running; I will report once it settles.")

    def settled(messages):
        assert "[Review settled]" in str(messages[-2:]), "Main runs again only for the settlement notice"
        assert sorted(a.reviewer.rounds) == ["m1", "m2"]
        return call("commit_reviewed", COMMIT, "commit-2")

    def collected(messages):
        text = _tool_text(messages)
        assert text.startswith("Review outcome returned before commit."), text
        return call("commit_reviewed", {**COMMIT, "review_reference": _reference(messages), "author_disposition": {
            "disposition": "accepted", "rationale": "Read the settled PASS; committing exactly that candidate."}},
            "commit-3")

    def done(messages):
        assert _commit_count(a) == 2, _tool_text(messages)
        return prose("Committed after the review settled.")

    steps = [started, yielded, settled] + ([done] if gate_blocks else [collected, done])
    answer, _usage, _trace = f.run(steps)
    assert answer == "Committed after the review settled."
    assert len(f.inputs) == len(steps), "no empty author round while the wave ran"
    assert [p["review_binding"].split(":")[0] for p in f.parks] == ["commit-review"], "one zero-call park"
    assert sorted(a.reviewer.rounds) == ["m1", "m2"], "one paid wave, collected at $0"
    assert len(_notices(a)) == 1 and len(a.pushes) == 1


def test_review_change_mail_yield_parks_and_the_same_call_collects(main_loop, monkeypatch):
    from tests.review_pool_rosters import pool_roster, pool_seat, set_review_pool

    set_review_pool(monkeypatch, pool_roster(pool_seat("first", "test/m1", delivery="native"),
                                             pool_seat("second", "test/m2", delivery="native")))
    f, a = main_loop, main_loop.a
    _git(a.repo, "add", "change.py")

    def started(messages):
        _mail_when_both_seats_started(a, "Is the review done?")
        return call("review_change", REVIEW, "review-1")

    def yielded(messages):
        assert _tool_text(messages).startswith("An addressed message reached you while the reviewers ran")
        assert "Is the review done?" in str(messages[-3:])
        _release_later(a)
        return prose("Still under review; I will answer when it settles.")

    def settled(messages):
        notice = str(messages[-2:])
        assert "[Review settled]" in notice and "paid change review" in notice
        return call("review_change", REVIEW, "review-2")

    def collected(messages):
        payload = _json_tail(_tool_text(messages))
        assert payload["state"] == "settled" and payload["aggregate"] == "PASS", payload
        return prose("The review passed.")

    answer, _usage, _trace = f.run([started, yielded, settled, collected])
    assert answer == "The review passed." and len(f.inputs) == 4
    assert sorted(a.reviewer.rounds) == ["m1", "m2"] and len(_notices(a, "change")) == 1


def test_silence_keeps_one_wait_and_main_runs_only_for_the_outcome(main_loop):
    f, a = main_loop, main_loop.a

    def started(messages):
        _release_later(a, 0.6)
        return call("commit_reviewed", COMMIT, "commit-1")

    def outcome(messages):
        text = _tool_text(messages)
        assert text.startswith("Review outcome returned before commit.") and "still running" not in text, text
        return prose("The review passed; I will decide on the commit next turn.")

    answer, _usage, _trace = f.run([started, outcome])
    assert answer.startswith("The review passed") and len(f.inputs) == 2 and not f.parks
    assert not _notices(a) and a.ctx._left_review_wait is None


def test_an_explicit_author_change_reviews_and_commits_b_while_a_still_runs(main_loop):
    """The author changes course after leaving A: another intent B gets its own panel at once
    (never A's verdict), the author commits B explicitly, and A keeps its own custody."""
    f, a = main_loop, main_loop.a

    def started(messages):
        _mail_when_both_seats_started(a)
        return call("commit_reviewed", COMMIT, "commit-a")

    def change_course(messages):
        assert _tool_text(messages).startswith("Review outcome returned before commit: reviewers are still running.")
        return call("commit_reviewed", {**COMMIT, "commit_message": "Other intent"}, "commit-b")

    def b_outcome(messages):
        text = _tool_text(messages)
        assert text.startswith("Review outcome returned before commit."), text
        reference = _reference(messages)
        return call("commit_reviewed", {**COMMIT, "commit_message": "Other intent", "review_reference": reference,
                                        "author_disposition": {"disposition": "accepted", "rationale": "B passed."}},
                    "commit-b-continue")

    def done(messages):
        return prose("Committed B; A is still under review.")

    answer, _usage, _trace = f.run([started, change_course, b_outcome, done])
    assert answer.startswith("Committed B") and len(f.inputs) == 4 and not f.parks
    assert _git(a.repo, "log", "-1", "--format=%s") == "Other intent"
    assert not a.reviewer.gates["m1"].is_set() and a.reviewer.rounds.count("m1") == 2, "A still runs; B paid its own"
    a.reviewer.release()
    assert until(lambda: _notices(a)), "A's settlement is still announced"
    assert _commit_count(a) == 2, "A's late settlement commits nothing"


@pytest.mark.parametrize("control", ["owner_stop", "deadline"])
def test_stop_and_deadline_end_the_review_park_and_commit_nothing(main_loop, control):
    from datetime import datetime, timedelta, timezone

    from ouroboros.owner_mailbox import KIND_FINALIZE_NOW

    f, a = main_loop, main_loop.a

    def started(messages):
        _mail_when_both_seats_started(a)
        return call("commit_reviewed", COMMIT, "commit-1")

    def yielded(messages):
        if control == "owner_stop":
            def stop():
                from ouroboros.cancel_intents import STOP_POLICY_FINALIZE, request_cancel
                from supervisor.owner_stop import owner_stop_control_id

                intent = request_cancel(a.data, TASK, requested_stop_policy=STOP_POLICY_FINALIZE)
                write_owner_message(a.data, "owner_requested_finalization", TASK,
                                    msg_id=owner_stop_control_id(intent), kind=KIND_FINALIZE_NOW)
            threading.Timer(0.4, stop).start()
        else:
            a.ctx.task_metadata = {"deadline_at": (datetime.now(timezone.utc) + timedelta(seconds=1.5)).isoformat()}
        return prose("Waiting for the commit review.")

    def final(messages):
        return prose("Stopped before the review settled; nothing was committed.")

    started_at = time.monotonic()
    steps = [started, yielded] + ([final] if control == "owner_stop" else [])
    answer, usage, _trace = f.run(steps)
    if control == "owner_stop":  # the graceful Stop's one tool-less final answer
        assert answer == "Stopped before the review settled; nothing was committed."
    else:  # the park ends AT the deadline: no time is left for a call, so the typed fallback answers
        assert answer.startswith("⚠️ Task reached its deadline"), answer
    assert time.monotonic() - started_at < 30 and len(f.parks) == 1 and len(f.inputs) == len(steps)
    assert usage["reason_code"] == ("owner_requested_finalization" if control == "owner_stop" else "deadline_local")
    assert _commit_count(a) == 1 and not a.reviewer.gates["m1"].is_set(), "the paid wave stays in custody"


def test_a_settlement_whose_notice_cannot_be_written_still_ends_the_park_once(main_loop, monkeypatch):
    """The park also ends at the wave's settlement itself (``_sleep_wake``): a notice the
    mailbox refused is said inline in the author's next round, once, never an endless park."""
    from ouroboros import owner_mailbox

    f, a = main_loop, main_loop.a
    real_write = owner_mailbox.write_task_message
    monkeypatch.setattr(owner_mailbox, "write_task_message", lambda root, text, task_id, **kw: (
        False if str(kw.get("msg_id") or "").startswith("commit-review-settled:") else real_write(root, text, task_id, **kw)))

    def started(messages):
        _mail_when_both_seats_started(a)
        return call("commit_reviewed", COMMIT, "commit-1")

    def yielded(messages):
        _release_later(a)
        return prose("Waiting for the commit review.")

    def settled(messages):
        assert "[Review settled]" in str(messages[-2:]), "the unwritten notice is said inline"
        return prose("Noted; I will collect it next.")

    answer, _usage, _trace = f.run([started, yielded, settled])
    assert answer == "Noted; I will collect it next." and len(f.inputs) == 3 and len(f.parks) == 1
    assert not _notices(a) and a.ctx._left_review_wait is None


def test_a_child_message_is_an_addressed_event_too(main_loop):
    """A descendant task's message ends the author's wait exactly as the owner's words do."""
    from ouroboros.owner_mailbox import write_task_message

    f, a = main_loop, main_loop.a

    def started(messages):
        def child():
            assert a.reviewer.sent["m1"].wait(60) and a.reviewer.preparing_m2.wait(60)
            write_task_message(a.data, "Child result ready.", TASK, source_task_id="child-7",
                               provenance="descendant_task")
        threading.Thread(target=child, daemon=True).start()
        return call("commit_reviewed", COMMIT, "commit-1")

    def yielded(messages):
        assert _tool_text(messages).startswith("Review outcome returned before commit: reviewers are still running.")
        assert "Child result ready." in str(messages[-3:])
        _release_later(a)
        return prose("Read the child's result; the commit review is still running.")

    def settled(messages):
        assert "[Review settled]" in str(messages[-2:])
        return prose("Both are in.")

    answer, _usage, _trace = f.run([started, yielded, settled])
    assert answer == "Both are in." and len(f.inputs) == 3 and len(f.parks) == 1


def test_an_explicit_finish_ends_the_task_without_the_left_review(main_loop):
    """An explicit completion is the author's choice over the wait: no park, no Git, and the paid
    wave keeps its own custody (its notice still follows)."""
    f, a = main_loop, main_loop.a

    def started(messages):
        _mail_when_both_seats_started(a)
        return call("commit_reviewed", COMMIT, "commit-1")

    def finish(messages):
        assert _tool_text(messages).startswith("Review outcome returned before commit: reviewers are still running.")
        return call("finish_task", {"action": "finish", "answer": "Finished; the commit review continues separately."},
                    "finish-1")

    answer, _usage, _trace = f.run([started, finish])
    assert answer == "Finished; the commit review continues separately." and len(f.inputs) == 2 and not f.parks
    assert not a.reviewer.gates["m1"].is_set() and _commit_count(a) == 1
    a.reviewer.release()
    assert until(lambda: _notices(a))


def test_a_new_turn_waits_on_no_left_review(main_loop):
    f, a = main_loop, main_loop.a
    a.ctx._left_review_wait = {"surface": "commit", "binding": "commit-review:previous-turn", "owner":
                               review_pause.PROCESS_TOKEN, "done": False, "msg_id": "commit-review-settled:previous"}
    answer, _usage, _trace = f.run([lambda messages: prose("A new turn's answer.")])
    assert answer == "A new turn's answer." and not f.parks and a.ctx._left_review_wait is None


# --- Negative controls: each switches one seam off and shows the old behavior. ---

def test_negative_a_deaf_wait_holds_the_owner_message_until_every_reviewer_answers(main_loop, monkeypatch):
    monkeypatch.setattr(review_pause, "receptive_binding", lambda ctx, **_kw: None)  # the pre-#1741 author
    f, a = main_loop, main_loop.a

    def started(messages):
        def owner():
            assert a.reviewer.sent["m1"].wait(60)
            write_owner_message(a.data, "How is the commit going?", TASK)
            time.sleep(0.6)
            assert len(f.inputs) == 1, "the deaf wait never let Main answer"
            a.reviewer.release()
        threading.Thread(target=owner, daemon=True).start()
        return call("commit_reviewed", COMMIT, "commit-1")

    def late(messages):
        assert _tool_text(messages).startswith("Review outcome returned before commit."), "only after settlement"
        assert "How is the commit going?" in str(messages[-3:])
        assert "review_checkouts" not in a.roots["m1"], "the deaf wave read the live index"
        return prose("Answering the owner only now.")

    answer, _usage, _trace = f.run([started, late])
    assert answer == "Answering the owner only now." and not f.parks and not _notices(a)


def test_negative_without_the_park_prose_ends_the_task_while_the_review_runs(main_loop, monkeypatch):
    monkeypatch.setattr(loop, "park_for_left_review", lambda *_a: False)  # no carried wait intention
    f, a = main_loop, main_loop.a

    def started(messages):
        _mail_when_both_seats_started(a)
        return call("commit_reviewed", COMMIT, "commit-1")

    answer, _usage, _trace = f.run([started, lambda messages: prose("Still reviewing.")])
    assert answer == "Still reviewing.", "the interim prose became the final answer"
    assert not a.reviewer.gates["m1"].is_set() and not f.parks
    a.reviewer.release()
    assert until(lambda: _notices(a)), "the settlement still arrives, to a turn that already ended"


# --- Cold restart: a saved left-review park holds no live custody in the new process. ---

def _cold_registry(tmp_path, monkeypatch, left):
    from ouroboros import context, task_pacing
    from ouroboros.model_wait import TaskModelWait
    from ouroboros.owner_wait import checkpoint_owner_wait, set_owner_wait
    from ouroboros.task_results import write_task_result
    from tests.test_context_fit_v664 import _plan

    registry = ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path)
    ctx = registry._ctx
    ctx.task_id, ctx.task_attempt = "t-wait", 1
    ctx.active_model, ctx.active_effort = "same-model", "high"
    ctx.active_use_local, ctx.active_context_mode = False, "max"
    ctx._cost_ceiling = task_pacing.CostCeiling(state="disabled")
    ctx._left_review_wait = left
    messages = _plan().messages_for("max") + [
        {"role": "assistant", "tool_calls": [{"id": "commit", "type": "function",
          "function": {"name": "commit_reviewed", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "commit", "content": "Review outcome returned before commit: reviewers are still running."},
        {"role": "assistant", "content": "Waiting for the commit review."},
    ]
    write_task_result(tmp_path, ctx.task_id, "running")
    wait = checkpoint_owner_wait(ctx, messages, {"tool_calls": []}, {"cost": 1.0, "execution_id": "exec"}, 0, [], set(),
                                 review_binding=left["binding"])
    set_owner_wait(tmp_path, ctx.task_id, {**wait, "state": "waiting"})
    ctx._left_review_wait = None  # the new process starts without it; only the saved state carries it
    ctx.owner_wait_resume = {**wait, "restart_transaction_id": "observed-restart"}
    ctx.event_queue = queue.Queue()
    ctx.model_wait_context = TaskModelWait(task={"id": ctx.task_id}, drive_root=tmp_path,
                                           event_queue=ctx.event_queue, worker_slot_held=False)
    ctx.model_wait_context.tool_context = ctx
    ctx.owner_wait_callback = direct_owner_wait
    ctx.context_fit_plan = _plan(window=500_000, known=True)
    monkeypatch.setattr(context, "_context_fit_route", lambda task, **_: (
        {"model": task["model"], "provider": "openai", "use_local": False},
        SimpleNamespace(status="confirmed", stale=False, window_tokens=500_000, route_fp="route-a"),
    ))
    monkeypatch.setenv("OUROBOROS_CONTEXT_MODE", "max")
    monkeypatch.setenv("OUROBOROS_TASK_REVIEW_MODE", "off")
    return registry


@pytest.mark.parametrize("wake,written", [(True, True), (True, False), (False, True)])
def test_a_cold_restored_left_review_wakes_with_its_custody_fact(tmp_path, monkeypatch, wake, written):
    """The saved park's review lived in the ended process, so its notice can never come: the
    restore says so in the mailbox, which ends the restored wait at once. Negative control
    (``wake=False``): the same restore without it stays parked until unrelated owner mail."""
    import time as real_time

    from tests.test_loop_transport_wait import _loop_kwargs

    fast = SimpleNamespace(**{name: getattr(real_time, name) for name in dir(real_time) if not name.startswith("_")})
    fast.sleep = lambda seconds: real_time.sleep(min(seconds, 0.05))
    monkeypatch.setattr("ouroboros.owner_wait.time", fast)
    left = {"surface": "commit", "binding": "commit-review:left", "retry_key": "left", "attempt": 3, "record_id": "",
            "task_id": "t-wait", "reissue": "the same unchanged commit_reviewed call", "owner": "ended-process",
            "done": False, "unknown_seats": 0, "msg_id": "commit-review-settled:left"}
    registry = _cold_registry(tmp_path, monkeypatch, left)
    if not wake:
        monkeypatch.setattr(review_pause, "wake_restored_left_review", lambda ctx: False)
    if not written:
        monkeypatch.setattr("ouroboros.owner_mailbox.write_task_message", lambda *_a, **_kw: False)
    seen = []

    def dispatch(call_, disposition, **_kwargs):
        seen.append(copy.deepcopy(call_.messages))
        return {"role": "assistant", "content": "The left review ended with the restart; I will re-issue it.",
                "tool_calls": []}, 0.0
    monkeypatch.setattr(loop, "_dispatch_round_model", dispatch)
    outcome = {}
    kwargs = {**_loop_kwargs(tmp_path, registry, [], llm=SimpleNamespace(default_model=lambda: "same-model")),
              "emit_progress": lambda *_a, **_k: None}  # the checkpoint ran on Main; a cold resume follows it (route_choice)
    worker = threading.Thread(target=lambda: outcome.update(result=loop.run_llm_loop(**kwargs)))
    worker.start()
    worker.join(60 if wake else 2)  # the positive ends by itself: 60 s is only its hang bound
    stranded = worker.is_alive()
    if stranded:  # release the negative control, including a regressed positive, before asserting
        assert not seen, "the cold park cannot buy an empty model round"
        write_owner_message(tmp_path, "Any news?", "t-wait")
        worker.join(60)
    assert stranded is (not wake), "a failed notice write must not strand a restored review wait"
    assert not worker.is_alive() and outcome["result"][0].startswith("The left review ended")
    custody = "[Review custody] The commit review you left"
    if wake:
        assert len(seen) == 1 and custody in str(seen[0][-2:]), "one informed round, never a park on nothing"
    else:  # only unrelated mail woke it; the in-loop fallback then still says what the restore did not
        assert len(seen) == 2 and "Any news?" in str(seen[0][-2:]) and custody in str(seen[1][-2:])
    row = load_task_result(tmp_path, "t-wait")["owner_wait"]
    assert row["state"] == "resumed" and registry._ctx._left_review_wait is None
