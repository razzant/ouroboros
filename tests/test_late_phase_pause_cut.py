"""The owner's Pause ends an answered root's post-answer upkeep at once; nothing is saved.

Real consumers end to end: the owner's Pause ingress, the post-task stage
coordinator and its fallback-draft stage adapter, the shared Light transport (a
knowledge read, then its answer — one Light operation of two physical sends), the
usage ledger's physical-send gate, the activity census, held Continues and the
maintenance pass that ends a Pause an earlier version saved. The retired scratchpad
pass (#918) used to be that first paid stage; the fallback stage's draft selection is
now stood in for by that same two-send operation (``_fallback_stand_in``), and the
seeded scratchpad proves no stage touches it. Only the model transport is a
deterministic fake, and it still goes through ``execute_physical_attempt`` — the real
reservation and owner-fence gate. A Pause an earlier version saved keeps the stage
names that version had.
"""

from __future__ import annotations

import json
import pathlib
import threading
from types import SimpleNamespace

import pytest

from tests import _dialogue as dialogue_test
from tests.test_consolidator_context_fit import fit  # noqa: F401 — offline Light route/window facts

ROOT = "late-root"
ANSWER = "Already delivered answer"
DRAFT = "FALLBACK-DRAFT-STAND-IN"


def _scratchpad(root: pathlib.Path, count: int = 4, tag: str = "block") -> None:
    """Working memory past the retired compression threshold: no late stage may touch it (#918)."""
    from ouroboros.memory import Memory

    memory = Memory(root)
    for index in range(count):
        memory.append_scratchpad_block(f"{tag}-{index}-" + (chr(97 + index) * 8_000), source=f"source-{index}")


class Light:
    """The Light route's transport, behind the real model-wait and ledger seams."""

    def __init__(self, monkeypatch):
        from ouroboros.model_wait import model_waitable

        self.sends: list = []
        self.hooks: dict = {}
        self.cost = 0.01
        light = self

        @model_waitable
        def chat(self, messages, model="openai/gpt-4.1-nano", model_role="light", **_kwargs):
            from ouroboros.usage_accounting import AttemptRequest, current_usage_scope, execute_physical_attempt

            prompt = str(messages[0]["content"])
            kind = "draft" if DRAFT in prompt else "other"
            answered = messages[-1].get("role") == "tool"  # the read came back: this send answers
            scope = current_usage_scope()

            def send():
                light.sends.append({"kind": kind, "prompt": prompt, "scope": scope, "answer": answered})
                pending = light.hooks.get(kind) or []
                hook = pending.pop(0) if pending else None  # one per send of this kind, in order
                if hook is not None:
                    hook()
                usage = {"prompt_tokens": 1, "completion_tokens": 1}
                if kind == "draft" and not answered:
                    return {"content": "", "tool_calls": [{"id": "list", "type": "function", "function": {
                        "name": "knowledge_list", "arguments": "{}"}}]}, usage
                if kind == "draft":
                    return {"content": json.dumps({"text": f"draft-{len(light.sends)}", "quotes": []})}, usage
                return {"content": f"{kind}-{len(light.sends)}"}, usage

            return execute_physical_attempt(
                AttemptRequest(model=model or "openai/gpt-4.1-nano", provider="openai", reservation_usd=0.01),
                send, extractor=lambda response: (response[1], light.cost, True))

        monkeypatch.setattr("ouroboros.llm.LLMClient.chat", chat)

    def kinds(self) -> list:
        return [row["kind"] for row in self.sends]


def _fallback_stand_in(monkeypatch) -> None:
    """The late phase's first paid stage: the real ``_run_memory_fallback_draft`` adapter around one
    Light operation of two physical sends through the shared transport, in place of the fallback's
    draft selection (which needs an activated chronicle with a unit to draft)."""
    from ouroboros import consolidator, memory_fallback
    from ouroboros.tools.registry import ToolContext

    def run_fallback_draft(env, task, llm, _drive_logs, _trace):
        ctx = ToolContext(repo_dir=env.repo_dir, drive_root=env.drive_root, task_id=str(task.get("id") or ""))
        _text, usage = consolidator._call_consolidation_llm(
            llm, f"{DRAFT}: draft one page from the closed records.", memory_fallback.LABEL,
            knowledge=consolidator.KnowledgeReadContext(ctx))
        errors = list(usage.get("_consolidation_errors") or [])
        return memory_fallback.FallbackRun("failed" if errors else "nothing", usage=usage, errors=errors)

    monkeypatch.setattr(memory_fallback, "run_fallback_draft", run_fallback_draft)


@pytest.fixture
def late(tmp_path, monkeypatch, fit):  # noqa: F811
    from ouroboros import agent_task_pipeline as pipeline
    from ouroboros.task_results import write_task_result
    from tests._budget_pause_exact_helpers import _install_queue

    q, _state, workers = _install_queue(tmp_path, monkeypatch)
    monkeypatch.delenv("OUROBOROS_IN_WORKER", raising=False)
    monkeypatch.setattr(workers, "REPO_DIR", tmp_path, raising=False)
    (tmp_path / "memory").mkdir(parents=True, exist_ok=True)
    (tmp_path / "memory" / "identity.md").write_text("IDENTITY\n", encoding="utf-8")
    _scratchpad(tmp_path)
    task = {"id": ROOT, "root_task_id": ROOT, "type": "task", "chat_id": 1, "text": "Original requested work",
            "budget_drive_root": str(tmp_path)}
    write_task_result(tmp_path, ROOT, "completed", result=ANSWER, chat_id=1, text=task["text"],
                      root_phase_checkpoint={"post_task_synthesis": "pending_once"})
    env = SimpleNamespace(drive_root=tmp_path, repo_dir=tmp_path, drive_path=lambda rel: tmp_path / rel)
    light = Light(monkeypatch)
    _fallback_stand_in(monkeypatch)
    stages: list = []
    # Promotion is a real stage; its chooser is off for this root and only records that it ran.
    monkeypatch.setattr("ouroboros.post_task_evolution.maybe_promote", lambda *a, **k: stages.append("promotion"))
    return SimpleNamespace(root=tmp_path, q=q, workers=workers, task=task, env=env, light=light, stages=stages,
                           pipeline=pipeline)


def _spawned(call):
    """Run ``call``; return its result and the detached threads it started (the late phase)."""
    before = set(threading.enumerate())
    value = call()
    return value, [thread for thread in threading.enumerate() if thread not in before]


def _join(threads):
    # A liveness bound, not a speed claim: a finished late phase returns at once, and on a loaded
    # machine the mocked phase has taken up to ~15 s.
    for thread in threads:
        thread.join(60)
        assert not thread.is_alive()


def _start(f, *, blocking=False, cap=3.0, **kwargs):
    from ouroboros.usage_accounting import UsageScope, usage_scope

    scope = UsageScope(drive_root=f.root, task_id=ROOT, root_task_id=ROOT, category="task", source="agent.task",
                       root_limit_usd=cap, root_limit_source="task_admission")
    with usage_scope(scope):
        return _spawned(lambda: f.pipeline._run_post_task_processing_async(
            f.env, f.task, {"rounds": 1}, {"tool_calls": []}, f.root / "logs", blocking=blocking, **kwargs))[1]


def _resume(f):
    """The owner's Resume through the queue seam, joined with anything it starts."""
    answer, threads = _spawned(lambda: f.q.resume_budget_paused_task(ROOT))
    _join(threads)
    return answer


def _row(f):
    from ouroboros.task_results import load_task_result

    return load_task_result(f.root, ROOT, strict=True)


def _phase(f) -> str:
    return str((_row(f).get("root_phase_checkpoint") or {}).get("post_task_synthesis") or "")


def _scratch(f) -> list:
    from ouroboros.memory import Memory

    return Memory(f.root).load_scratchpad_blocks()


def _jsonl(path: pathlib.Path) -> list:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _facts_rows(f) -> list:
    return [row for row in dialogue_test.log_rows(f.root / "logs" / "chat.jsonl") if row.get("summary_kind") == "host_task_facts"]


def _events(f, kind: str) -> list:
    return [row for row in _jsonl(f.root / "logs" / "events.jsonl")
            if row.get("type") == kind and row.get("task_id") == ROOT]


def _census(f) -> dict:
    from ouroboros.gateway.state import _chat_activities_snapshot_safe

    rows = _chat_activities_snapshot_safe(f.root, {}, direct_turns=[])
    return {row["activity_id"]: row["phase"] for row in rows}


def _during_read(f, act, cap=3.0, **kwargs):
    """Run ``act`` while the draft operation's first send (its read) is on the wire."""
    entered, release, answer = threading.Event(), threading.Event(), {}

    def hook():
        entered.set()
        assert release.wait(10)

    f.light.hooks["draft"] = [hook]
    threads = _start(f, cap=cap, **kwargs)
    try:
        assert entered.wait(10)
        answer.update(act() or {})
    finally:
        release.set()
        _join(threads)
    return answer


def _pause_during_read(f, request_id="late-pause", cap=3.0, **kwargs):
    """The owner's Pause lands while the draft operation's first send (its read) is on the wire."""
    from supervisor.owner_pause_control import request_owner_pause

    return _during_read(f, lambda: request_owner_pause(ROOT, request_id=request_id), cap=cap, **kwargs)


def _stop(f) -> None:
    from ouroboros.cancel_intents import request_cancel

    request_cancel(f.root, ROOT, reason="owner stop", source="http", requested_by="owner",
                   requested_stop_policy="immediate", allow_settled_target=True)


def _maintenance(f) -> int:
    """The steady-state post-task recovery duty, as the maintenance loop runs it."""
    return f.pipeline.recover_pending_root_post_task_synthesis(
        f.root, f.root, startup=False, owner_liveness=f.q.task_settlement_liveness,
        claim_guard=f.q.task_settlement_interlock)


def _saved_by_an_earlier_version(f, *, stage="scratchpad_consolidation",
                                 completed=("memory_fallback_draft",), remaining=None,
                                 retired=(), stopped=False, payload=None):
    """What an earlier version left when the owner's Pause reached the upkeep: the open
    ``paused`` phase with its record and actor-source payload, the closed fence and its latch."""
    from ouroboros.artifacts import store_actor_source_bytes
    from ouroboros.owner_pause import install_fence, set_fence_state
    from ouroboros.task_results import write_task_result
    from supervisor.events_budget import _set_root_budget_pause_locked

    remaining = list(remaining if remaining is not None else [stage, "promotion"])
    body = {"schema": "ouroboros.late_phase_pause.v1", "task_id": ROOT, "stage": stage,
            "completed_stages": list(completed), "remaining_stages": remaining, "marks": [], "drafts": {},
            "task": f.task, "usage": {"rounds": 1}, "usage_snapshot": {"rounds": 1}, "trace": {"tool_calls": []},
            "review_evidence": {}, "sealed_final": None, "drive_logs": str(f.root / "logs"),
            "reflection_callback": None, **(payload or {})}
    ref = store_actor_source_bytes(f.root, ROOT, category="context_checkpoints", source_id="late-phase-pause-v1",
                                   extension="json", data=json.dumps(body).encode("utf-8"))
    record = {"pause_id": "earlier-pause", "paused_at": "2026-10-13T21:11:55Z", "stage": stage,
              "completed_stages": list(completed), "remaining_stages": remaining, "payload_ref": ref,
              "grant": {"grant_id": "unconsumed-grant", "fence_id": "", "granted_at": "2026-10-13T21:12:00Z"},
              **({"retired_stages": list(retired)} if retired else {}),
              **({"stopped_at": "2026-10-13T21:13:00Z"} if stopped else {})}
    write_task_result(f.root, ROOT, "completed", root_phase_checkpoint={
        "phase": "task_acceptance", "status": "not_required", "pass_index": 0,
        "post_task_synthesis": "paused", "post_task_pause": record})
    fence, _created = install_fence(f.root, ROOT, request_id="earlier-pause")
    set_fence_state(f.root, ROOT, fence_id=fence["fence_id"], state="paused", expected_state="requested")
    with f.q._queue_lock:
        _set_root_budget_pause_locked(ROOT, {"fence_id": fence["fence_id"], "cause": "owner_pause",
                                             "paused_at": fence["requested_at"]})
    return fence


def test_pause_during_the_upkeep_ends_it_at_once_with_the_skip_disclosed_and_the_root_settles(late):
    """No RUNNING row, as after a direct turn: the send on the wire finishes, the next is
    refused and no later stage runs; the answer stays Done, nothing is saved, and the Pause,
    with nothing left to hold, is released at once."""
    f = late
    from ouroboros.owner_pause import read_fence

    before_blocks = _scratch(f)
    pause = _pause_during_read(f)
    assert pause["ok"] and pause["root_task_id"] == ROOT, pause
    assert pause["state"] == "requested"  # accepted while the upkeep was still running
    # The read on the wire finished; the answering send never got a physical send, nor did promotion.
    assert f.light.kinds() == ["draft"] and f.stages == []
    row = _row(f)
    assert row["status"] == "completed" and row["result"] == ANSWER
    checkpoint = row["root_phase_checkpoint"]
    assert checkpoint["post_task_synthesis"] == "degraded"
    assert checkpoint["post_task_stop_reason"] == "owner_pause:skipped=memory_fallback_draft,promotion"
    assert "post_task_pause" not in checkpoint  # nothing saved to resume
    assert [event["post_task_stop_reason"] for event in _events(f, "task_cost_finalized")] == [
        checkpoint["post_task_stop_reason"]]
    assert _scratch(f) == before_blocks and len(_facts_rows(f)) == 1
    assert read_fence(f.root, ROOT)["state"] == "released" and ROOT not in f.q.BUDGET_ROOT_FENCES
    assert ROOT not in _census(f)
    # Nothing is left to resume: Resume finds no Pause and buys nothing.
    assert not _resume(f).get("ok") and f.light.kinds() == ["draft"] and _phase(f) == "degraded"


def test_pooled_pause_ends_the_upkeep_and_task_done_releases_the_pause(late, monkeypatch):
    """The pooled shape: the worker still holds RUNNING while its blocking upkeep runs. The
    Pause ends the upkeep at once; once task_done releases the slot nothing is left to pause."""
    f = late
    from ouroboros.model_wait import task_model_wait_scope
    from ouroboros.owner_pause import read_fence
    from supervisor.owner_pause_control import refresh_owner_pause_tree, request_owner_pause
    from supervisor.queue_transitions import clear_budget_root_fence_for_settled_tree

    monkeypatch.setattr("ouroboros.agent_task_pipeline.in_worker_process", lambda: True)
    f.workers.RUNNING[ROOT] = {"task": f.task, "attempt": 1, "worker_id": 0}
    answers = {}

    def hook():
        answers.update(request_owner_pause(ROOT, request_id="pooled-late-pause"))
        assert _census(f)[ROOT] == "budget_pausing"

    f.light.hooks["draft"] = [hook]
    with task_model_wait_scope(task=f.task, drive_root=f.root, event_queue=None, worker_slot_held=True):
        _start(f, blocking=True)
    assert answers["ok"], answers
    assert f.light.kinds() == ["draft"] and f.stages == [] and _phase(f) == "degraded"
    assert _row(f)["root_phase_checkpoint"]["post_task_stop_reason"] == (
        "owner_pause:skipped=memory_fallback_draft,promotion")
    # Ended, but the worker still holds its slot until task_done: the tree is still pausing.
    assert _census(f)[ROOT] == "budget_pausing" and read_fence(f.root, ROOT)["state"] == "requested"
    # The task_done handler's order: release the slot and the latch, settle the tree.
    f.workers.RUNNING.clear()
    assert clear_budget_root_fence_for_settled_tree({"id": ROOT, "root_task_id": ROOT})
    assert refresh_owner_pause_tree(ROOT) == "released" and ROOT not in _census(f)
    assert _row(f)["result"] == ANSWER and len(_facts_rows(f)) == 1


def test_a_stop_still_ends_the_upkeep_at_once(late):
    """Stop alone skips the rest of the upkeep as before, and saves nothing either."""
    f = late

    _during_read(f, lambda: _stop(f))
    stopped = _row(f)["root_phase_checkpoint"]
    assert stopped["post_task_synthesis"] == "degraded"
    assert stopped["post_task_stop_reason"] == "cancelled:skipped=promotion"
    assert f.light.kinds() == ["draft"] and f.stages == [] and "post_task_pause" not in stopped


@pytest.mark.parametrize("order,expected", [
    ("both_before_the_next_send", "cancelled:skipped=promotion"),
    ("stop_after_the_pause_cut", "owner_stopped:skipped=memory_fallback_draft,promotion"),
])
def test_a_stop_beside_the_pause_is_what_the_disclosure_names(late, monkeypatch, order, expected):
    """Stop outranks Pause: landing before the next send the Stop is what the send gate
    refuses with; landing after the Pause already cut the stage, the coordinator names it."""
    f = late
    from ouroboros import post_task_synthesis
    from supervisor.owner_pause_control import request_owner_pause

    if order == "stop_after_the_pause_cut":
        named = post_task_synthesis.post_task_interruption

        def then_stop(control):
            reason = named(control)
            assert reason == "owner_pause"
            _stop(f)
            return reason

        monkeypatch.setattr(post_task_synthesis, "post_task_interruption", then_stop)

    def pause_and_stop():
        assert request_owner_pause(ROOT, request_id="pause-and-stop")["ok"]
        if order == "both_before_the_next_send":
            _stop(f)

    _during_read(f, pause_and_stop)
    checkpoint = _row(f)["root_phase_checkpoint"]
    assert checkpoint["post_task_synthesis"] == "degraded" and checkpoint["post_task_stop_reason"] == expected
    assert f.light.kinds() == ["draft"] and f.stages == [] and "post_task_pause" not in checkpoint


def test_fully_finished_root_reports_pause_inapplicable_and_buys_nothing(late):
    """Positive no-extra-work control: a genuinely finished root is not paused."""
    f = late
    from ouroboros.owner_pause import read_fence
    from ouroboros.task_results import write_task_result
    from supervisor.owner_pause_control import request_owner_pause

    write_task_result(f.root, ROOT, "completed", root_phase_checkpoint={"post_task_synthesis": "completed"})
    for registered in (False, True):
        if registered:
            f.workers.RUNNING[ROOT] = {"task": f.task, "attempt": 1, "worker_id": 0}
        answer = request_owner_pause(ROOT, request_id=f"finished-{registered}")
        assert not answer["ok"] and answer["error"] == ("task_terminal" if registered else "task_not_live")
    assert read_fence(f.root, ROOT) == {} and f.light.sends == []
    f.workers.RUNNING.clear()


def _continue_eligible(f):
    """This answered root ended on a technical rail, so the owner may also Continue it."""
    from ouroboros.task_results import write_task_result

    write_task_result(f.root, ROOT, "completed", reason_code="round_limit", origin_message_text="Original requested work",
                      origin_message_ref={"chat_id": 1, "client_message_id": "m-1"},
                      deadline_at="2099-01-01T00:00:00+00:00", billing_group={
                          "billing_group_id": ROOT, "billing_group_limit_usd": 20.0,
                          "billing_group_limit_source": "initial_task_admission",
                          "billing_group_limit_revision": "admission-1"})


def test_running_upkeep_holds_a_continue_until_it_ends(late):
    """Positive control: the upkeep is predecessor custody while it runs; a Continue admitted
    then is held, and the existing release frees that same successor once the upkeep ended."""
    f = late
    from supervisor.continuation_admission import admit_continuation, release_settled_continuations
    from supervisor.events_budget import HOLD_CONTINUATION_WRITER, budget_hold_fact

    _continue_eligible(f)
    observed = {}

    def admit():
        ack = admit_continuation(ROOT, action_nonce="late-continue-0001")
        observed["released"] = release_settled_continuations(ROOT)
        return ack

    ack = _during_read(f, admit)
    assert ack["ok"] and ack["held"], ack
    assert {"kind": "late_phase_running", "task_id": ROOT} in ack["blockers"]
    assert observed == {"released": []}, "a running upkeep keeps the successor held"
    held = next(task for task in f.workers.PENDING if task["id"] == ack["successor_task_id"])
    assert _phase(f) == "completed"
    assert budget_hold_fact(held) is None or budget_hold_fact(held)["reason"] != HOLD_CONTINUATION_WRITER
    assert release_settled_continuations(ROOT) == []  # released once, by the upkeep's own end


def test_a_continue_is_never_held_by_upkeep_the_pause_ended(late):
    f = late
    from supervisor.continuation_admission import admit_continuation, conflicting_writers

    _continue_eligible(f)
    assert _pause_during_read(f)["ok"] and _phase(f) == "degraded"
    assert not [b for b in conflicting_writers(f.q, ROOT, drive_root=f.root) if str(b.get("kind")).startswith("late_phase")]
    ack = admit_continuation(ROOT, action_nonce="late-continue-0002")
    assert ack["ok"] and not ack.get("held"), ack


@pytest.mark.parametrize("retired,stopped,expected", [
    ((), False, "owner_pause:skipped=scratchpad_consolidation,promotion"),
    (("reflection",), False, "owner_pause:skipped=scratchpad_consolidation,promotion;stage_retired:skipped=reflection"),
    ((), True, "owner_stopped:skipped=scratchpad_consolidation,promotion"),
], ids=["paused", "retired_stage", "stop_landed"])
def test_the_maintenance_pass_ends_an_earlier_saved_pause_once_and_releases_its_latch(late, retired, stopped, expected):
    """A Pause an earlier version saved runs nothing and holds no writer; boot leaves it, and
    the maintenance pass ends it exactly once with the skip a Pause discloses today, releasing
    the latch and fence. The answer, its status and the saved payload stay."""
    f = late
    from ouroboros.owner_pause import read_fence
    from ouroboros.post_task_checkpoint import late_phase_state

    _saved_by_an_earlier_version(f, retired=retired, stopped=stopped)
    saved = dict(_row(f)["root_phase_checkpoint"]["post_task_pause"])
    assert late_phase_state(f.root, ROOT) == ""  # never a Continue blocker: it writes nothing
    # Startup runs nothing and leaves it for the maintenance pass.
    assert f.pipeline.recover_pending_root_post_task_synthesis(f.root, f.root) == 0
    assert _phase(f) == "paused" and read_fence(f.root, ROOT)["state"] == "paused"
    assert _maintenance(f) == 1
    row = _row(f)
    checkpoint = row["root_phase_checkpoint"]
    assert checkpoint["post_task_synthesis"] == "degraded" and checkpoint["post_task_stop_reason"] == expected
    assert checkpoint["post_task_pause"] == saved  # history is kept, never rewritten
    assert row["status"] == "completed" and row["result"] == ANSWER
    assert read_fence(f.root, ROOT)["state"] == "released" and ROOT not in f.q.BUDGET_ROOT_FENCES
    assert ROOT not in _census(f)
    assert [event["post_task_stop_reason"] for event in _events(f, "saved_late_phase_ended")] == [expected]
    # Exactly once: a later pass finds nothing owed; nothing was bought either time.
    assert _maintenance(f) == 0 and len(_events(f, "saved_late_phase_ended")) == 1
    assert f.light.sends == [] and f.stages == [] and len(_facts_rows(f)) == 0


def test_the_conversion_skips_a_phase_under_a_live_owner_and_a_closed_generation(late):
    """Guard and positive: a live owner or a stopped maintenance generation converts nothing;
    the same record is ended once that owner is gone."""
    f = late

    _saved_by_an_earlier_version(f)
    f.workers.RUNNING[ROOT] = {"task": f.task, "attempt": 1, "worker_id": 0}
    assert _maintenance(f) == 0 and _phase(f) == "paused"
    f.workers.RUNNING.clear()
    stopped = f.pipeline.recover_pending_root_post_task_synthesis(
        f.root, f.root, startup=False, owner_liveness=f.q.task_settlement_liveness,
        claim_guard=f.q.task_settlement_interlock, stop_requested=lambda: True)
    assert stopped == 0 and _phase(f) == "paused"
    assert _maintenance(f) == 1 and _phase(f) == "degraded"
