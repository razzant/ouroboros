"""The owner-source acknowledgement tells the truth about what it could read.

The 2026-09-19 night (task 549a5405): a round-top re-capture of the owner-source
observation hit an inspect ``TimeoutError``, stored ``{}`` over the good
observation Main had been shown, and the host called Main's VALID control
"stale" — a lost answer. These tests pin the cure from both directions: an
unknown queue state is disclosed, never read as a refusal; only known facts are
compared and the queue's compare-and-seal keeps the fail-closed authority; every
refusal carries its typed cause in a durable worker-side row; and a never-shown
selector is rendered once the delivery control is armed.
"""
from __future__ import annotations

import json
import queue
import threading
from types import SimpleNamespace

import pytest

from ouroboros.delivery_seal import _begin_task_acceptance_fence, _end_task_acceptance_fence
from ouroboros.loop_delivery import _replace_delivery_candidate, apply_delivery_subject_decision
from ouroboros.loop_messages import (
    _record_owner_directive,
    acceptance_observation_prompt,
    acknowledge_acceptance_observation,
    capture_acceptance_observation,
    owner_source_sha256,
    prepare_acceptance_observation,
)
from ouroboros.owner_mailbox import write_owner_message, write_task_message
from tests.test_delivery_control_lineage import _start_control_episode
from tests.test_delivery_forced_finalization import _forced_test_context

INSPECT_FAILURE = TimeoutError("supervisor did not acknowledge acceptance fence")


@pytest.fixture
def case(tmp_path):
    """A root whose one retained answer was observed from the current owner source."""
    agent = SimpleNamespace(_owner_message_generation=0, _accepting_owner_messages=True)
    tool_ctx = SimpleNamespace(
        task_id="root", task_attempt=1, drive_root=tmp_path, budget_drive_root=tmp_path,
        task_metadata={"root_task_id": "root"}, task_contract={"expected_output": "Complete report"},
        _owner_directives=[], _loop_mailbox_seen_ids=set(), owner_message_admission_agent=agent,
        owner_message_admission_lock=threading.RLock(),
    )
    _record_owner_directive(tool_ctx, source="initial_user", content="Prepare the full report.", msg_id="first")
    tools = SimpleNamespace(_ctx=tool_ctx)
    trace = {"tool_calls": [], "reasoning_notes": []}
    ctx = SimpleNamespace(
        tools=tools, task_id="root", root_task_id="root", drive_root=tmp_path, status_drive_root=tmp_path,
        drive_logs=tmp_path / "logs", messages=[], incoming_messages=queue.Queue(),
        accumulated_usage={}, event_queue=None, round_idx=2, max_rounds=20, task_type="task",
    )
    capture_acceptance_observation(tool_ctx, trace, ctx.incoming_messages)
    candidate = _replace_delivery_candidate(tools, ctx, trace, "The complete report.", control="candidate")
    return tool_ctx, tools, ctx, trace, candidate


def _select(tools, ctx, trace, **request):
    from ouroboros.loop_delivery import completion_observation, consume_completion_request
    from ouroboros.tools.control_runtime import stage_completion_request

    tools._ctx._completion_observation = completion_observation(tools._ctx, trace)
    staged = stage_completion_request(tools._ctx, {"action": "finish", **request})
    assert json.loads(staged)["status"] == "completion_requested"
    return consume_completion_request(tools, ctx, trace)


def _ack_rows(root):
    path = root / "logs" / "events.jsonl"
    if not path.exists():
        return []
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [row for row in rows if row.get("type") == "acceptance_source_ack"]


def _queue_fence(tool_ctx, trace, *, generation=0):
    """A queue-owned fence whose inspect can be switched to failing."""
    state = {"owner_message_generation": generation, "fail": False}

    def inspect(**_kwargs):
        if state["fail"]:
            raise INSPECT_FAILURE
        return {"ok": True, "status": "active", "owner_message_generation": state["owner_message_generation"]}

    tool_ctx._task_acceptance_fence_token = "queue-fence"
    tool_ctx._task_acceptance_fence_generation = generation
    tool_ctx.inspect_acceptance_fence = inspect
    tool_ctx._execution_trace = trace
    return state


# T1 -----------------------------------------------------------------------------
def test_unknown_queue_inspection_does_not_refuse_a_valid_source_ack(case):
    tool_ctx, _tools, ctx, trace, _candidate = case
    state = _queue_fence(tool_ctx, trace)
    tool_ctx._current_llm_call_meta = {"round": 7}
    observed = capture_acceptance_observation(tool_ctx, trace, ctx.incoming_messages)
    assert observed["queue_generation"] == 0
    state["fail"] = True

    ack = acknowledge_acceptance_observation(tool_ctx, observed["owner_source_sha256"])

    assert ack and ack.ok and ack.cause == ""
    assert ack.unknown == "queue_state_unknown"
    assert tool_ctx._acceptance_ack_source_sha256 == observed["owner_source_sha256"]
    # The fence generation advances only from a KNOWN value: it keeps begin's value, never None.
    assert tool_ctx._task_acceptance_fence_generation == 0
    assert trace["review_decision"]["admission_inspection"] == {
        "status": "unknown", "reason": "queue_inspection_failed", "error_type": "TimeoutError",
    }
    row = _ack_rows(tool_ctx.drive_root)[-1]
    assert row["task_id"] == "root" and row["round"] == 7 and row["ok"] is True and row["cause"] == ""
    assert row["unknown"] == "queue_state_unknown" and row["error_type"] == "TimeoutError"
    assert row["observed"]["queue_generation"] == 0 and row["current"]["queue_generation"] is None
    assert row["observed"]["owner_source_sha256"] == observed["owner_source_sha256"]
    assert row["pending_kinds"] == []


# T2 -----------------------------------------------------------------------------
def test_failed_recapture_keeps_the_last_good_observation(case):
    """Scenario S2 of the night: a good observation was captured and shown to Main;
    the round-top re-capture fails on inspect; Main names the sha it was shown; the
    supervisor may even be healthy again at ack time."""
    tool_ctx, _tools, ctx, trace, _candidate = case
    state = _queue_fence(tool_ctx, trace)
    shown = capture_acceptance_observation(tool_ctx, trace, ctx.incoming_messages)
    shown_note = acceptance_observation_prompt(tool_ctx, shown)
    assert shown["owner_source_sha256"] in shown_note

    state["fail"] = True
    again = capture_acceptance_observation(tool_ctx, trace, ctx.incoming_messages)

    assert again["owner_source_sha256"] == shown["owner_source_sha256"], "a failed re-capture clobbered the good observation"
    assert again["queue_generation"] is None and again["fence_token"] == "queue-fence"
    assert again["admission_inspection"]["error_type"] == "TimeoutError"
    note = acceptance_observation_prompt(tool_ctx, again)
    assert "admission_inspection" not in note and "tool_count" not in note, "the unknown mark is never rendered"
    assert shown["owner_source_sha256"] in note

    state["fail"] = False  # the supervisor is healthy again
    ack = acknowledge_acceptance_observation(tool_ctx, shown["owner_source_sha256"])
    assert ack and ack.unknown == "queue_state_unknown"
    assert tool_ctx._task_acceptance_fence_generation == 0
    row = _ack_rows(tool_ctx.drive_root)[-1]
    assert row["ok"] is True and row["error_type"] == "TimeoutError"
    assert row["observed"]["queue_generation"] is None and row["current"]["queue_generation"] == 0


def test_a_pending_owner_signal_still_yields_an_empty_observation(case):
    """The other direction of the capture change: unread owner input is still {}."""
    tool_ctx, _tools, ctx, trace, _candidate = case
    assert write_owner_message(tool_ctx.drive_root, "Owner input.", "root", msg_id="unread")
    assert capture_acceptance_observation(tool_ctx, trace, ctx.incoming_messages) == {}


# T3 -----------------------------------------------------------------------------
def test_unknown_at_capture_and_ack_still_fails_closed_at_the_end_seal(case):
    tool_ctx, _tools, ctx, trace, _candidate = case
    queue_side = {"owner_message_generation": 0}
    tool_ctx.begin_acceptance_fence = lambda **_kw: {"token": "final", "owner_message_generation": 0}
    calls = []

    def end(**kwargs):
        calls.append(dict(kwargs))
        expected = kwargs.get("expected_generation")
        if expected is not None and int(expected) != queue_side["owner_message_generation"]:
            return {"ok": True, "status": "released", "generation_mismatch": True}
        return {"ok": True, "status": "sealed"}  # a blind seal when no generation is named

    tool_ctx.end_acceptance_fence = end
    tool_ctx.inspect_acceptance_fence = lambda **_kw: (_ for _ in ()).throw(INSPECT_FAILURE)
    tool_ctx._execution_trace = trace
    opened, token = _begin_task_acceptance_fence(tool_ctx, "root")
    assert opened and token == "final"
    assert tool_ctx._task_acceptance_fence_generation == 0

    observed = capture_acceptance_observation(tool_ctx, trace, ctx.incoming_messages)
    assert observed["queue_generation"] is None
    ack = acknowledge_acceptance_observation(tool_ctx, observed["owner_source_sha256"])
    assert ack and ack.unknown == "queue_state_unknown"
    assert tool_ctx._task_acceptance_fence_generation == 0, "an unknown read must not erase the known generation"

    queue_side["owner_message_generation"] = 1  # a real owner message admitted by the supervisor
    assert _end_task_acceptance_fence(tool_ctx, outcome="terminal")
    assert calls[-1]["outcome"] == "terminal" and calls[-1]["expected_generation"] == 0
    assert tool_ctx._task_acceptance_fence_generation_mismatch is True
    assert tool_ctx._task_acceptance_sealed_fence_token is None


# T4 -----------------------------------------------------------------------------
def test_real_unread_owner_text_is_refused_with_its_typed_cause(case):
    tool_ctx, tools, ctx, trace, candidate = case
    observed = capture_acceptance_observation(tool_ctx, trace, ctx.incoming_messages)
    assert write_owner_message(tool_ctx.drive_root, "Actually add X.", "root", msg_id="owner-1")

    ack = acknowledge_acceptance_observation(tool_ctx, observed["owner_source_sha256"])
    assert not ack and ack.cause == "owner_input_unread" and ack.unknown == ""
    assert ack.facts["pending_kinds"] == ["owner_text"]
    row = _ack_rows(tool_ctx.drive_root)[-1]
    assert row["ok"] is False and row["cause"] == "owner_input_unread" and row["pending_kinds"] == ["owner_text"]

    ok, error = apply_delivery_subject_decision(tools, ctx, trace, {"owner_source_sha256": observed["owner_source_sha256"]})
    assert not ok and error.startswith("owner_input_unread") and "owner_text" in error
    assert "stale" not in error and "consume the current message" not in error

    tool_ctx._delivery_control_required = True
    candidate.finalization_control = "awaiting_control"
    retained = candidate.full_text
    assert not _select(tools, ctx, trace, answer_sha256=candidate.content_sha256,
                       acceptance_subject={"owner_source_sha256": observed["owner_source_sha256"]})
    refusal = trace["completion_refusals"][-1]["reason"]
    assert "owner_input_unread" in refusal and "owner_text" in refusal
    assert "source_not_observed" not in refusal, "an overtaken valid selection is not Main's error"
    assert tools._ctx._delivery_candidate.full_text == retained
    assert not getattr(tools._ctx, "_completion_selected", None)


def test_a_stale_sha_after_a_rendered_selector_is_still_main_error(tmp_path):
    """The model-caused direction: the selector WAS shown and Main named another sha."""
    loop_mod, registry, ctx, trace, candidate = _start_control_episode(tmp_path)
    loop_mod._arm_delivery_control(registry, ctx, trace)
    assert "[ACCEPTANCE_SUBJECT_OBSERVATION]" in str(ctx.messages)
    assert not _select(registry, ctx, trace, answer_sha256=candidate.content_sha256,
                       acceptance_subject={"owner_source_sha256": "0" * 64})
    assert "source_not_observed" in trace["completion_refusals"][-1]["reason"]
    assert registry._ctx._delivery_candidate.full_text == candidate.full_text
    rows = _ack_rows(tmp_path)
    assert rows[-1]["cause"] == "source_not_observed" and rows[-1]["ok"] is False


# T6 -----------------------------------------------------------------------------
def test_an_envelope_error_is_not_masked_by_the_owner_generation_notice(tmp_path):
    loop_mod, registry, ctx, trace, candidate = _start_control_episode(tmp_path)
    loop_mod._arm_delivery_control(registry, ctx, trace)
    _record_owner_directive(registry._ctx, source="direct_incoming", content="Also cover Q3.", msg_id="followup")
    assert loop_mod._owner_generation_changed(registry._ctx)
    result = registry.execute_result("finish_task", {
        "action": "finish", "answer_sha256": candidate.content_sha256,
        "acceptance_subject": {"owner_source_sha256": owner_source_sha256(registry._ctx)},
        "effective_criteria": "misplaced at the top level",
    })
    assert result.status == "error" and "effective_criteria" in result.text
    assert "owner input has not been acknowledged" not in result.text
    assert not getattr(registry._ctx, "_completion_request", None)


# T7 -----------------------------------------------------------------------------
@pytest.mark.parametrize("shape", ["subagent", "review_off"])
def test_a_never_shown_selector_is_rendered_once_the_control_is_armed(tmp_path, monkeypatch, shape):
    """Finding A: a subagent and a root under review mode ``off`` are both armed with
    the control after a principal letter changes the owner source. The observation
    row must be rendered so a valid control can name it; no degraded_preserve."""
    loop_mod, registry, ctx, trace = _forced_test_context(tmp_path)
    tool_ctx = registry._ctx
    if shape == "subagent":
        tool_ctx.task_metadata.update({"root_task_id": "root0", "delegation_role": "subagent", "parent_task_id": "root0"})
        tool_ctx.parent_task_id, tool_ctx.root_task_id, tool_ctx.delegation_role = "root0", "root0", "subagent"
    else:
        monkeypatch.setenv("OUROBOROS_TASK_REVIEW_MODE", "off")
    _record_owner_directive(tool_ctx, source="initial_user", content="Do the child work.", msg_id="first")
    ctx.messages[:] = [{"role": "user", "content": "Do the child work."}]
    seen = set()
    assert write_task_message(tmp_path, "Wrap up now.", "parent1", source_task_id="root0",
                              provenance="ancestor_task", msg_id="parent-1")
    candidate = loop_mod._replace_delivery_candidate(registry, ctx, trace, "Child answer.", control="candidate")
    loop_mod._arm_delivery_control(registry, ctx, trace)  # the letter is unread: no selector row yet
    assert "[ACCEPTANCE_SUBJECT_OBSERVATION]" not in str(ctx.messages)
    before = owner_source_sha256(tool_ctx)
    loop_mod._drain_incoming_messages(ctx.messages, queue.Queue(), tmp_path, "parent1", None, seen, owner_ctx=tool_ctx)
    assert owner_source_sha256(tool_ctx) != before, "the principal letter entered the owner corpus"

    prepare_acceptance_observation(tool_ctx, trace, queue.Queue(), ctx.messages)

    rows = [row for row in ctx.messages if row.get("acceptance_observation")]
    assert len(rows) == 1 and tool_ctx._acceptance_observation["owner_source_sha256"] in rows[0]["content"]
    assert _select(registry, ctx, trace, answer_sha256=candidate.content_sha256,
                   acceptance_subject={"owner_source_sha256": tool_ctx._acceptance_observation["owner_source_sha256"]})
    assert registry._ctx._delivery_candidate.full_text == "Child answer."
    assert not registry._ctx._delivery_candidate.degraded


def test_an_unarmed_ineligible_turn_still_gets_no_selector(tmp_path, monkeypatch):
    """The gate's working direction: without an armed control the internal selector
    protocol stays out of an ordinary conversation (here a root under review mode ``off``)."""
    loop_mod, registry, ctx, trace = _forced_test_context(tmp_path)
    monkeypatch.setenv("OUROBOROS_TASK_REVIEW_MODE", "off")
    _record_owner_directive(registry._ctx, source="initial_user", content="Chat.", msg_id="first")
    loop_mod._replace_delivery_candidate(registry, ctx, trace, "An answer.", control="candidate")
    assert not getattr(registry._ctx, "_delivery_control_required", False)
    prepare_acceptance_observation(registry._ctx, trace, queue.Queue(), ctx.messages)
    assert not any(row.get("acceptance_observation") for row in ctx.messages)
