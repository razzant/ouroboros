"""Owner D10: Pause of an answered root's late phase and its soft same-task Resume.

Real consumers end to end: the owner's Pause ingress, the post-task stage
coordinator, scratchpad consolidation (a knowledge read, then its answer — one
Light operation of two physical sends), the usage ledger's physical-send gate, the
saved actor-source pause, startup recovery, the Resume grant/executor, the activity
census, Stop custody and held Continues. Only the model transport is a
deterministic fake, and it still goes through ``execute_physical_attempt`` — the
real reservation and owner-fence gate. (The retired dialogue writer was the first
paid stage these tests paused; the scratchpad stage now is.)
"""

from __future__ import annotations

import json
import pathlib
import threading
from types import SimpleNamespace

import pytest

from tests.test_consolidator_context_fit import fit  # noqa: F401 — offline Light route/window facts

ROOT = "late-root"
ANSWER = "Already delivered answer"


def _scratchpad(root: pathlib.Path, count: int = 4, tag: str = "block") -> None:
    """Enough scratchpad working memory that the post-task scratchpad stage runs."""
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
            kind = "scratchpad" if "scratchpad working memory" in prompt else "other"
            answered = messages[-1].get("role") == "tool"  # the read came back: this send answers
            scope = current_usage_scope()

            def send():
                light.sends.append({"kind": kind, "prompt": prompt, "scope": scope, "answer": answered})
                pending = light.hooks.get(kind) or []
                hook = pending.pop(0) if pending else None  # one per send of this kind, in order
                if hook is not None:
                    hook()
                usage = {"prompt_tokens": 1, "completion_tokens": 1}
                if kind == "scratchpad" and not answered:
                    return {"content": "", "tool_calls": [{"id": "list", "type": "function", "function": {
                        "name": "knowledge_list", "arguments": "{}"}}]}, usage
                if kind == "scratchpad":
                    return {"content": json.dumps({"knowledge_entries": [],
                                                   "compressed_block": f"compressed-{len(light.sends)}"})}, usage
                return {"content": f"{kind}-{len(light.sends)}"}, usage

            return execute_physical_attempt(
                AttemptRequest(model=model or "openai/gpt-4.1-nano", provider="openai", reservation_usd=0.01),
                send, extractor=lambda response: (response[1], light.cost, True))

        monkeypatch.setattr("ouroboros.llm.LLMClient.chat", chat)

    def kinds(self) -> list:
        return [row["kind"] for row in self.sends]


@pytest.fixture
def late(tmp_path, monkeypatch, fit):  # noqa: F811
    from ouroboros import agent_task_pipeline as pipeline
    from ouroboros.task_results import write_task_result
    from tests._budget_pause_exact_helpers import _install_queue

    q, _state, workers = _install_queue(tmp_path, monkeypatch)
    monkeypatch.delenv("OUROBOROS_IN_WORKER", raising=False)
    monkeypatch.setattr(workers, "REPO_DIR", tmp_path, raising=False)
    (tmp_path / "memory").mkdir(parents=True, exist_ok=True)
    (tmp_path / "memory" / "identity.md").write_text("IDENTITY-BEFORE-PAUSE\n", encoding="utf-8")
    _scratchpad(tmp_path)
    task = {"id": ROOT, "root_task_id": ROOT, "type": "task", "chat_id": 1, "text": "Original requested work",
            "budget_drive_root": str(tmp_path)}
    write_task_result(tmp_path, ROOT, "completed", result=ANSWER, chat_id=1, text=task["text"],
                      root_phase_checkpoint={"post_task_synthesis": "pending_once"})
    env = SimpleNamespace(drive_root=tmp_path, repo_dir=tmp_path, drive_path=lambda rel: tmp_path / rel)
    light = Light(monkeypatch)
    stages: list = []
    # Reflection/promotion are real stages; this root has nothing to reflect on.
    monkeypatch.setattr(pipeline, "_run_reflection", lambda *a, **k: stages.append("reflection"))
    return SimpleNamespace(root=tmp_path, q=q, workers=workers, task=task, env=env, light=light, stages=stages,
                           pipeline=pipeline)


def _spawned(call):
    """Run ``call``; return its result and the detached threads it started (the late phase)."""
    before = set(threading.enumerate())
    value = call()
    return value, [thread for thread in threading.enumerate() if thread not in before]


def _join(threads):
    for thread in threads:
        thread.join(10)
        assert not thread.is_alive()


def _start(f, *, blocking=False, cap=3.0):
    from ouroboros.usage_accounting import UsageScope, usage_scope

    scope = UsageScope(drive_root=f.root, task_id=ROOT, root_task_id=ROOT, category="task", source="agent.task",
                       root_limit_usd=cap, root_limit_source="task_admission")
    with usage_scope(scope):
        return _spawned(lambda: f.pipeline._run_post_task_processing_async(
            f.env, f.task, {"rounds": 1}, {"tool_calls": []}, {}, f.root / "logs", blocking=blocking))[1]


def _resume(f):
    """The owner's Resume through the queue seam, joined with the late phase it starts."""
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


def _facts_rows(f) -> list:
    rows = [json.loads(line) for line in (f.root / "logs" / "chat.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    return [row for row in rows if row.get("summary_kind") == "host_task_facts"]


def _census(f) -> dict:
    from ouroboros.gateway.state import _chat_activities_snapshot_safe

    rows = _chat_activities_snapshot_safe(f.root, {}, direct_turns=[])
    return {row["activity_id"]: row["phase"] for row in rows}


def _pause_during_read(f, request_id="late-pause", cap=3.0):
    """The owner's Pause lands while the scratchpad operation's first send (its read) is on the wire."""
    from supervisor.owner_pause_control import request_owner_pause

    entered, release, answer = threading.Event(), threading.Event(), {}

    def hook():
        entered.set()
        assert release.wait(10)

    f.light.hooks["scratchpad"] = [hook]
    threads = _start(f, cap=cap)
    assert entered.wait(10)
    answer.update(request_owner_pause(ROOT, request_id=request_id))
    release.set()
    _join(threads)
    assert _phase(f) == "paused"
    return answer


def test_pause_after_delivery_saves_the_remainder_and_soft_resume_reassesses_current_inputs(late):
    """D10 core: Pause after the answer, without a RUNNING row; the send on the wire
    finishes and the next is refused, inputs change while paused, Restart keeps the pause,
    and one Resume re-thinks the stopped stage on the SAME task under its original money
    scope against the CURRENT inputs."""
    f = late
    from ouroboros.artifacts import read_actor_source_bytes
    from ouroboros.owner_pause import read_fence
    from ouroboros.post_task_checkpoint import late_phase_pause_record
    from ouroboros.task_results import list_task_results

    before_blocks = _scratch(f)
    pause = _pause_during_read(f)
    assert pause["ok"] and pause["root_task_id"] == ROOT, pause
    # The read on the wire finished; the answering send never got a physical send.
    assert f.light.kinds() == ["scratchpad"]
    row = _row(f)
    assert row["status"] == "completed" and row["result"] == ANSWER
    assert _phase(f) == "paused"
    record = late_phase_pause_record(row)
    assert record["stage"] == "scratchpad_consolidation"
    assert record["remaining_stages"][0] == "scratchpad_consolidation" and "promotion" in record["remaining_stages"]
    assert "chat_consolidation" not in record["remaining_stages"]
    payload = json.loads(read_actor_source_bytes(f.root, ROOT, record["payload_ref"]))
    assert payload["drafts"] == {}  # no stage keeps a draft any more
    assert payload["money_scope"]["root_limit_usd"] == 3.0
    assert "calls" not in payload  # no request replay cache
    assert read_fence(f.root, ROOT)["state"] == "paused"
    assert _census(f)[ROOT] == "budget_paused"
    assert len(_facts_rows(f)) == 1 and f.stages == [] and _scratch(f) == before_blocks

    # Inputs change while paused: a new identity.
    (f.root / "memory" / "identity.md").write_text("IDENTITY-AFTER-PAUSE\n", encoding="utf-8")

    # Restart: startup recovery keeps the pause, starts nothing, and re-arms the latch a
    # stale snapshot would have dropped.
    f.q.BUDGET_ROOT_FENCES.clear()
    assert f.pipeline.recover_pending_root_post_task_synthesis(f.root, f.root) == 0
    assert _phase(f) == "paused" and f.light.kinds() == ["scratchpad"]
    assert f.q.BUDGET_ROOT_FENCES[ROOT]["cause"] == "owner_pause"
    assert _census(f)[ROOT] == "budget_paused"

    tasks_before = len(list_task_results(f.root))
    resumed = _resume(f)
    assert resumed["ok"] and resumed["late_phase"] == "resumed", resumed
    assert _phase(f) == "completed"
    kinds = f.light.kinds()
    # The stopped operation is re-thought whole (read, then answer) against the CURRENT identity.
    assert kinds == ["scratchpad"] * 3, kinds
    assert [send["answer"] for send in f.light.sends] == [False, False, True]
    assert "IDENTITY-BEFORE-PAUSE" in f.light.sends[0]["prompt"]
    assert "IDENTITY-AFTER-PAUSE" in f.light.sends[1]["prompt"] and "block-0-" in f.light.sends[1]["prompt"]
    # The same task, attempt identity and original money scope; no new task, no second answer.
    for send in f.light.sends:
        assert send["scope"].task_id == ROOT and send["scope"].root_task_id == ROOT
        assert send["scope"].root_limit_usd == 3.0 and send["scope"].root_limit_source == "task_admission"
    assert len(list_task_results(f.root)) == tasks_before
    assert _row(f)["result"] == ANSWER
    assert len(_facts_rows(f)) == 1 and f.stages == ["reflection"]
    assert _scratch(f)[0]["content"] == "compressed-3" and len(_scratch(f)) == 3
    assert read_fence(f.root, ROOT)["state"] == "released" and ROOT not in f.q.BUDGET_ROOT_FENCES
    assert ROOT not in _census(f)

    # A duplicate Resume cannot spend again.
    again = _resume(f)
    assert not again.get("ok") and f.light.kinds() == kinds


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


def test_pause_with_running_row_parks_late_phase_and_task_done_settles_it(late, monkeypatch):
    """The pooled shape: the worker still holds RUNNING while its blocking late phase runs."""
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

    f.light.hooks["scratchpad"] = [hook]
    with task_model_wait_scope(task=f.task, drive_root=f.root, event_queue=None, worker_slot_held=True):
        _start(f, blocking=True)
    assert answers["ok"], answers
    assert f.light.kinds() == ["scratchpad"] and _phase(f) == "paused"
    # Saved, but the worker still holds its slot until task_done: the tree is still pausing.
    assert _census(f)[ROOT] == "budget_pausing" and read_fence(f.root, ROOT)["state"] == "requested"
    # The task_done handler's order: release the slot, keep the owner latch, settle the tree.
    f.workers.RUNNING.clear()
    assert not clear_budget_root_fence_for_settled_tree({"id": ROOT, "root_task_id": ROOT})
    assert refresh_owner_pause_tree(ROOT) == "paused" and _census(f)[ROOT] == "budget_paused"
    assert _row(f)["result"] == ANSWER and len(_facts_rows(f)) == 1


def test_stop_cancels_the_saved_remainder_keeps_the_answer_and_releases_the_pause(late):
    f = late
    from ouroboros.cancel_intents import request_cancel
    from ouroboros.owner_pause import read_fence
    from ouroboros.task_results import load_task_result
    from supervisor.queue_transitions import task_has_live_ownership
    from supervisor.task_lifecycle import cancel_task_custody

    assert _pause_during_read(f)["ok"]
    assert task_has_live_ownership(ROOT)  # the saved remainder is addressable custody
    request_cancel(f.root, ROOT, reason="owner stop", source="http", requested_by="owner",
                   requested_stop_policy="immediate", allow_settled_target=True)
    assert cancel_task_custody(ROOT) == "cancelled"
    row = load_task_result(f.root, ROOT, strict=True)
    checkpoint = row["root_phase_checkpoint"]
    assert row["status"] == "completed" and row["result"] == ANSWER
    assert checkpoint["post_task_synthesis"] == "degraded"
    assert checkpoint["post_task_stop_reason"].startswith("owner_stopped:skipped=scratchpad_consolidation")
    assert read_fence(f.root, ROOT)["state"] == "released" and ROOT not in f.q.BUDGET_ROOT_FENCES
    assert not task_has_live_ownership(ROOT)
    resumed = _resume(f)
    assert not resumed.get("ok") and f.light.kinds() == ["scratchpad"]


@pytest.mark.parametrize("bound", ["deadline", "global_budget", "root_cap", "panic"])
def test_hard_bound_resume_refuses_without_grant_or_send_and_keeps_the_remainder(late, monkeypatch, bound):
    """A refused Resume mints/consumes nothing and keeps the saved remainder; once the
    bound is lifted through its existing owner the same Resume is admitted and finishes once."""
    f = late
    from ouroboros.post_task_checkpoint import late_phase_pause_record
    from ouroboros.task_results import write_task_result
    from supervisor import state

    if bound == "root_cap":
        f.light.cost = 0.05  # the paid read overruns this root's own cap
    assert _pause_during_read(f, cap=0.015 if bound == "root_cap" else 3.0)["ok"]
    before = late_phase_pause_record(_row(f))
    expected = {"deadline": "deadline_passed", "global_budget": "budget_still_exhausted",
                "root_cap": "root_hard_cap_exhausted", "panic": "restart_no_resume"}[bound]
    if bound == "deadline":
        write_task_result(f.root, ROOT, "completed", task_contract={"deadline_at": "2020-01-01T00:00:00Z"})
    elif bound == "global_budget":
        monkeypatch.setattr(state, "TOTAL_BUDGET_LIMIT", 0.005)
    elif bound == "panic":
        (f.root / "state").mkdir(exist_ok=True)
        (f.root / "state" / "panic_stop.flag").write_text("panic", encoding="utf-8")
    refused = _resume(f)
    assert not refused["ok"] and refused["error"] == expected, refused
    assert _phase(f) == "paused" and late_phase_pause_record(_row(f)) == before  # no grant, nothing consumed
    assert f.light.kinds() == ["scratchpad"] and f.q.BUDGET_ROOT_FENCES[ROOT]["cause"] == "owner_pause"
    if bound == "root_cap":
        return  # raising an original root cap is the existing owner amendment, not a Resume form
    if bound == "deadline":
        write_task_result(f.root, ROOT, "completed", task_contract={"deadline_at": "2999-01-01T00:00:00Z"})
    elif bound == "global_budget":
        monkeypatch.setattr(state, "TOTAL_BUDGET_LIMIT", 10.0)
    else:
        (f.root / "state" / "panic_stop.flag").unlink()
    resumed = _resume(f)
    assert resumed["ok"], resumed
    assert _phase(f) == "completed"
    assert f.light.kinds() == ["scratchpad"] * 3 and _row(f)["result"] == ANSWER


def _continue_eligible(f):
    """This answered root ended on a technical rail, so the owner may also Continue it."""
    from ouroboros.task_results import write_task_result

    write_task_result(f.root, ROOT, "completed", reason_code="round_limit", origin_message_text="Original requested work",
                      origin_message_ref={"chat_id": 1, "client_message_id": "m-1"},
                      deadline_at="2099-01-01T00:00:00+00:00", billing_group={
                          "billing_group_id": ROOT, "billing_group_limit_usd": 20.0,
                          "billing_group_limit_source": "initial_task_admission",
                          "billing_group_limit_revision": "admission-1"})


def test_continue_stays_held_behind_a_paused_or_running_late_phase_and_never_overlaps_its_resume(late):
    """The late phase is predecessor custody: a Continue admitted while it is paused is held,
    stays held while the resumed remainder runs, and is released only once it ended."""
    f = late
    from supervisor.continuation_admission import admit_continuation, release_settled_continuations
    from supervisor.events_budget import HOLD_CONTINUATION_WRITER, budget_hold_fact

    _continue_eligible(f)
    assert _pause_during_read(f)["ok"]
    ack = admit_continuation(ROOT, action_nonce="late-continue-0001")
    assert ack["ok"] and ack["held"], ack
    assert {"kind": "late_phase_paused", "task_id": ROOT} in ack["blockers"]
    held = next(task for task in f.workers.PENDING if task["id"] == ack["successor_task_id"])
    assert budget_hold_fact(held)["reason"] == HOLD_CONTINUATION_WRITER
    observed = {}
    # While the resumed remainder runs, the held successor stays held: no second writer.
    f.light.hooks["scratchpad"] = [lambda: observed.update(released=release_settled_continuations(ROOT))]
    resumed = _resume(f)
    assert resumed["ok"], resumed
    assert observed == {"released": []} and _phase(f) == "completed"
    # The remainder ended: the existing release path frees the SAME successor once.
    assert budget_hold_fact(held) is None
    assert release_settled_continuations(ROOT) == []


def test_a_live_continue_successor_refuses_late_resume_until_it_ended(late):
    """Guard and positive: a successor of this answered root that may already write blocks
    the late Resume (no overlapping writers); once it is gone the same Resume is admitted."""
    f = late
    from ouroboros.post_task_checkpoint import late_phase_pause_record

    assert _pause_during_read(f)["ok"]
    before = late_phase_pause_record(_row(f))
    successor = {"id": "successor-1", "root_task_id": "successor-1",
                 "metadata": {"continuation": {"predecessor_task_id": ROOT}}}
    f.workers.RUNNING["successor-1"] = {"task": successor, "attempt": 1, "worker_id": 0}
    refused = _resume(f)
    assert not refused["ok"] and refused["error"] == "owner_pause_effects_unsettled", refused
    assert {"kind": "continuation_successor", "task_id": "successor-1"} in refused["blockers"]
    assert late_phase_pause_record(_row(f)) == before and f.light.kinds() == ["scratchpad"]
    f.workers.RUNNING.clear()
    resumed = _resume(f)
    assert resumed["ok"], resumed
    assert _phase(f) == "completed" and f.light.kinds() == ["scratchpad"] * 3


def test_completed_effects_inside_a_paused_stage_do_not_repeat_and_the_rest_is_rethought(late, monkeypatch):
    """Promotion: the free backlog append happened before the Pause and is never appended
    again; the paid chooser it was inside is re-thought on Resume against the current backlog."""
    f = late
    from ouroboros import post_task_evolution
    from ouroboros.improvement_backlog import _count_of, load_backlog_items

    from ouroboros.memory import Memory

    Memory(f.root).mutate_scratchpad_blocks(lambda _blocks: [])  # nothing to consolidate: promotion is the stage
    entry = {"reflection": "lesson", "memory_actions": [],
             "backlog_candidates": [{"summary": "Make late work resumable", "category": "process"}]}
    monkeypatch.setattr(f.pipeline, "_run_reflection", lambda *a, **k: (f.stages.append("reflection"), entry)[1])
    chooser = []

    def maybe_promote(env, task, reflection_entry, llm_client):
        chooser.append([item["summary"] for item in load_backlog_items(env.drive_root)])
        for _ in range(2):  # a two-call chooser
            llm_client.chat(messages=[{"role": "user", "content": "choose a promotion"}], model_role="light")

    monkeypatch.setattr(post_task_evolution, "maybe_promote", maybe_promote)
    from supervisor.owner_pause_control import request_owner_pause

    f.light.hooks["other"] = [lambda: request_owner_pause(ROOT, request_id="promotion-pause")]
    _join(_start(f))
    assert _phase(f) == "paused" and f.light.kinds() == ["other"]
    items = load_backlog_items(f.root)
    assert len(items) == 1 and _count_of(items[0]) == 1
    assert _resume(f)["ok"]
    assert _phase(f) == "completed" and f.light.kinds() == ["other", "other", "other"]
    items = load_backlog_items(f.root)
    assert len(items) == 1 and _count_of(items[0]) == 1  # the applied append was not repeated
    assert f.stages == ["reflection"] and len(chooser) == 2 and chooser[1] == ["Make late work resumable"]


def test_restart_between_grant_and_start_revokes_the_unused_grant_and_resume_spends_once(late, monkeypatch):
    """The process dies after the Resume grant is recorded but before the executor consumed
    it: startup revokes that unused grant and runs nothing; a later Resume runs once."""
    f = late
    from ouroboros import agent_task_pipeline
    from ouroboros.post_task_checkpoint import late_phase_pause_record

    assert _pause_during_read(f)["ok"]
    real = agent_task_pipeline.recover_pending_root_post_task_synthesis

    def dies(*args, resume_task_id="", **kwargs):
        if resume_task_id:
            raise SystemExit("process died before the executor consumed the grant")
        return real(*args, **kwargs)

    monkeypatch.setattr(agent_task_pipeline, "recover_pending_root_post_task_synthesis", dies)
    with pytest.raises(SystemExit):
        f.q.resume_budget_paused_task(ROOT)
    grant = late_phase_pause_record(_row(f))["grant"]
    assert grant["grant_id"] and not grant.get("consumed_at") and not grant.get("revoked_at")
    monkeypatch.setattr(agent_task_pipeline, "recover_pending_root_post_task_synthesis", real)
    assert real(f.root, f.root) == 0  # Restart: no automatic work
    revoked = late_phase_pause_record(_row(f))["grant"]
    assert revoked["grant_id"] == grant["grant_id"] and revoked["revoke_reason"] == "restart_no_resume"
    assert _phase(f) == "paused" and f.light.kinds() == ["scratchpad"]
    assert _resume(f)["ok"] and _phase(f) == "completed"
    assert f.light.kinds() == ["scratchpad"] * 3
