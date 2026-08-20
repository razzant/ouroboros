"""Phase B of the delegation-substrate sprint (owner decisions 2A/3A/5A/7A).

B1: the dispatch-time executor note (v7 home `agent_dispatch`, re-exported by
`agent`) conditionally supersedes native-self-execution framing —
and ONLY on the final post-preflight harness dispatch. B2: the frozen acting
preamble states the write-root boundary without the "do it yourself" imperative.
B3: a really-INJECTED finalization nudge is durably stamped by the worker, and a
COMPLETED harness child with zero delegated runs carries the typed
`nanny_finalized_after_nudge_without_delegation` substrate disclosure. B4:
`capability_delta` gains typed `reduction_reasons`/`substrate_disclosures` lists
and `reason` is DERIVED from them at every writer (byte-identical to the old
hand-splice). B5: `schedule_subagent` descriptions steer objective = outcome,
context = the delegated run's work order, model_lane = leave auto.
"""

from types import SimpleNamespace

import pytest

from ouroboros import subagents
from ouroboros.subagents import (
    CapabilityDelta,
    DelegationRoute,
    SubagentDispatch,
    SubagentExecutorResolution,
    SubagentLaneResolution,
    derive_capability_reason,
    envelope_from_task,
    resolve_subagent_executor,
    _disclose_native_only_substrate,
)


# ---------------------------------------------------------------- B1: dispatch note


def _resolution(executor="harness", reason="harness_ready", route="claude"):
    return SubagentExecutorResolution(
        requested="auto", executor=executor,
        route=DelegationRoute(route_id=route) if executor == "harness" else None,
        reason=reason,
    )


def test_dispatch_note_override_rides_only_the_harness_branch():
    from ouroboros.agent_dispatch import dispatch_executor_note

    note = dispatch_executor_note(_resolution())
    # The conditional supersede paragraph (owner decision 2A, verbatim wording).
    assert "superseded by this dispatch" in note
    assert "WORK ORDER" in note
    assert "not a script for you to execute natively" in note

    # A native child (the ordinary case) gets no note at all.
    assert dispatch_executor_note(_resolution("native", "requested_native")) == ""

    # A preflight-DEMOTED child runs native: it gets the metered-unavailable
    # marker, never the delegation override (it has no delegated run to route to).
    demoted = dispatch_executor_note(_resolution("native", "delegate_tools_invisible"))
    assert "METERED API tokens" in demoted
    assert "superseded by this dispatch" not in demoted
    assert "WORK ORDER" not in demoted

    # Blocked pins announce through the typed terminal, not this note.
    assert dispatch_executor_note(_resolution("blocked", "delegate_tools_invisible")) == ""


def test_agent_reexports_the_moved_note_pair():
    # F7: the pair lives in the v7 dispatch-seam leaf; the byte-pinned transport
    # suite imports both from ouroboros.agent, so the re-export must be the
    # SAME objects under the same names.
    import ouroboros.agent as agent
    import ouroboros.agent_dispatch as notes

    assert agent.dispatch_executor_note is notes.dispatch_executor_note
    assert agent.executor_blocked_outcome is notes.executor_blocked_outcome


# ---------------------------------------------------------------- B2: acting preamble


def test_acting_preamble_states_the_boundary_without_the_native_imperative():
    from supervisor.events import _compose_subagent_text

    text = _compose_subagent_text(
        "obj", role="builder", expected_output="out", constraints="", context="ctx",
        task_constraint={"mode": "acting_subagent", "surface": "self_worktree",
                         "write_root": "/tmp/wt"},
    )
    assert "[WRITE SURFACE]" in text
    assert "All changes land inside the write root only." in text
    # The retired imperative framed the child as the native executor of the work.
    assert "Make all changes" not in text
    # The rest of the frozen block is intact.
    assert "Do NOT commit" in text
    assert "workspace.patch" in text


# ---------------------------------------------------------- B4: derived reason (byte-equality)


def _harness_ready(monkeypatch):
    route = DelegationRoute(route_id="codex")
    monkeypatch.setattr(
        subagents, "dispatch_executor_resolution",
        lambda task: resolve_subagent_executor("auto", route=route),
    )


def test_writer1_resolution_reason_is_derived_and_byte_identical(tmp_path, monkeypatch):
    # Two dispatch axes at once (the incident shape): a lane slot reduction plus
    # an exhausted-window executor fallback. The string must be byte-identical
    # to the historical "; ".join AND derived from the typed list.
    monkeypatch.setenv("OUROBOROS_MODEL", "provider::main")
    monkeypatch.setenv("OUROBOROS_MODEL_HEAVY", "")
    route = DelegationRoute(route_id="codex")
    monkeypatch.setattr(
        subagents, "dispatch_executor_resolution",
        lambda task: resolve_subagent_executor(
            "auto", route=route, reset_at="2026-09-01T00:00:00Z"),
    )
    dispatch = subagents.resolve_subagent_dispatch(
        {"id": "c1", "type": "task", "requested_model_lane": "heavy"},
        task_type="task",
    )
    delta = dispatch.delta.as_dict()
    assert delta["reduction_reasons"] == [
        "lane_slot_unavailable=heavy", "subscription_window_exhausted"]
    assert delta["substrate_disclosures"] == []
    assert delta["reason"] == "lane_slot_unavailable=heavy; subscription_window_exhausted"
    assert delta["reason"] == derive_capability_reason(delta["reduction_reasons"])


def test_writer2_preflight_native_fallback_derives_the_reason(tmp_path, monkeypatch):
    from ouroboros.agent import preflight_delegate_visibility, resolve_dispatch_axes

    monkeypatch.setenv("OUROBOROS_MODEL", "provider::main")
    monkeypatch.setenv("OUROBOROS_MODEL_HEAVY", "provider::strong")
    monkeypatch.setenv("OUROBOROS_MODEL_LIGHT", "provider::cheap")
    _harness_ready(monkeypatch)
    task = {"id": "c-fb", "type": "task", "delegation_role": "subagent",
            "requested_model_lane": "auto", "parent_model_lane": "heavy",
            "requested_executor": "auto"}
    dispatch = resolve_dispatch_axes(task)
    tools = SimpleNamespace(available_tools=lambda: ["read_file"])
    amended, changed = preflight_delegate_visibility(tools, task, dispatch)
    assert changed is True
    delta = amended.delta.as_dict()
    assert delta["reduction_reasons"][-1] == "delegate_tools_invisible"
    assert delta["reason"] == derive_capability_reason(delta["reduction_reasons"])
    assert delta["substrate_disclosures"] == []
    # Byte-identical to the historical construction on this record too.
    assert delta["reason"] == "; ".join(delta["reduction_reasons"])


def test_writer3_append_reason_derives_on_the_unverified_path(monkeypatch):
    from ouroboros.agent import preflight_delegate_visibility

    lane = SubagentLaneResolution(
        requested_lane="auto", effective_lane="main", model="m", resolved_from="main")
    dispatch = SubagentDispatch(
        lane=lane, effort="low", executor="harness", route="claude",
        profile="local_readonly_subagent",
        delta=CapabilityDelta(requested_executor="auto", effective_executor="harness"),
        executor_resolution=_resolution(),
    )
    task = {"id": "c1", "delegation_role": "subagent", "requested_executor": "auto",
            "effective_executor": "harness", "executor_route": "claude"}

    def _boom():
        raise RuntimeError("registry exploded")

    amended, changed = preflight_delegate_visibility(
        SimpleNamespace(available_tools=_boom), task, dispatch)
    assert changed is True
    delta = amended.delta.as_dict()
    assert delta["reduction_reasons"] == ["delegate_visibility_unverified"]
    assert delta["reason"] == "delegate_visibility_unverified"
    assert delta["reason"] == derive_capability_reason(delta["reduction_reasons"])


def test_writer4_substrate_amendment_derives_and_stays_byte_identical():
    # NEW-format dict (typed lists present): the fact lands in the substrate
    # list, reason re-derives, and the string equals the historical concat.
    fresh = CapabilityDelta(
        requested_lane="auto", resolved_lane="heavy", effective_lane="main",
        reduction_reasons=("lane_slot_unavailable=heavy",),
        reason="lane_slot_unavailable=heavy", reduced=True,
    ).as_dict()
    amended = _disclose_native_only_substrate(fresh)
    assert amended["substrate_disclosures"] == ["delegated_substrate_unused"]
    assert amended["reduction_reasons"] == ["lane_slot_unavailable=heavy"]
    assert amended["reason"] == "lane_slot_unavailable=heavy; delegated_substrate_unused"
    assert amended["reason"] == derive_capability_reason(
        amended["reduction_reasons"], amended["substrate_disclosures"])
    # Idempotent, exactly as the old substring check was.
    assert _disclose_native_only_substrate(amended)["reason"] == amended["reason"]
    # The dispatch-time author's dict stays untouched (amend-a-copy contract).
    assert fresh["substrate_disclosures"] == []

    # LEGACY durable dict (string only, no lists): the historical concatenated
    # shape is preserved byte-for-byte.
    legacy = _disclose_native_only_substrate(
        {"reason": "lane_slot_unavailable=heavy", "reduced": True})
    assert legacy["reason"] == "lane_slot_unavailable=heavy; delegated_substrate_unused"
    assert legacy["substrate_disclosures"] == ["delegated_substrate_unused"]

    # An empty-reason amendment (the un-reduced harness dispatch) is the bare fact.
    empty = _disclose_native_only_substrate(CapabilityDelta().as_dict())
    assert empty["reason"] == "delegated_substrate_unused"
    assert empty["reduced"] is True


def test_substrate_facts_render_as_separate_disclosure_entries():
    from ouroboros.subagents import capability_delta_disclosures

    delta = _disclose_native_only_substrate(CapabilityDelta(
        requested_lane="auto", resolved_lane="heavy", effective_lane="main",
        reduction_reasons=("lane_slot_unavailable=heavy",),
        reason="lane_slot_unavailable=heavy", reduced=True,
    ).as_dict(), nudge_ignored=True)
    parts = capability_delta_disclosures(delta)
    # Slot fact and substrate facts are SEPARATE entries — never one fused phrase.
    assert "model_lane auto(inherited heavy)->main" in parts
    assert "delegated_substrate_unused" in parts
    assert "nanny_finalized_after_nudge_without_delegation" in parts
    assert not any("main" in p and "substrate" in p for p in parts)


def test_prompt_block_directs_harness_children_through_the_delegated_run():
    from ouroboros.agent import capability_delta_prompt_block

    def _block(effective_executor):
        delta = CapabilityDelta(
            requested_lane="heavy", resolved_lane="heavy", effective_lane="main",
            requested_executor="auto", effective_executor=effective_executor,
            reduction_reasons=("lane_slot_unavailable=heavy",),
            reason="lane_slot_unavailable=heavy", reduced=True,
        )
        return capability_delta_prompt_block(
            SimpleNamespace(delta=delta, executor_resolution=None))

    harness = _block("harness")
    assert "model_lane heavy->main" in harness
    assert "(lane_slot_unavailable=heavy)" in harness  # typed axes, not the raw string
    assert "routed through your delegated run" in harness
    native = _block("native")
    assert "Do the work anyway, but say" in native
    assert "delegated run" not in native


# ---------------------------------------------------------- B3: nudge stamp end-to-end


def _drive(tmp_path):
    (tmp_path / "logs").mkdir(parents=True, exist_ok=True)
    return tmp_path


def _stamp_ctx(drive):
    return SimpleNamespace(task_metadata={"budget_drive_root": str(drive)})


def _harness_task(drive, **extra):
    return {
        "id": "child-1", "delegation_role": "subagent",
        "requested_executor": "auto", "effective_executor": "harness",
        "executor_route": "claude", "budget_drive_root": str(drive),
        "capability_delta": {"reduced": False, "reason": "",
                             "reduction_reasons": [], "substrate_disclosures": []},
        **extra,
    }


def test_loop_injection_writes_the_durable_stamp(tmp_path):
    import json

    from ouroboros import delegate_custody as custody
    from ouroboros.delegate_evidence import NANNY_NUDGE_STAMP
    from ouroboros.loop import _maybe_inject_finalization_nudges

    drive = _drive(tmp_path)
    ctx = SimpleNamespace(
        _nanny_route_dispatched=True, _nanny_finalization_injected=False,
        task_metadata={"budget_drive_root": str(drive)},
    )
    tools = SimpleNamespace(_ctx=ctx, available_tools=lambda: ["delegate_start"])
    msgs: list = []
    assert _maybe_inject_finalization_nudges(
        tools, drive, "child-1",
        {"reasoning_notes": [], "tool_calls": []}, "done", msgs, lambda *_: None,
    ) is True
    rows = [json.loads(line) for line in
            (drive / "logs" / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    stamps = [r for r in rows if r.get("type") == NANNY_NUDGE_STAMP]
    assert stamps and stamps[0]["task_id"] == "child-1"
    assert stamps[0]["nanny_code"] == "NANNY_DID_NOT_DELEGATE"
    assert custody.task_execution_evidence(drive, "child-1")["nanny_nudge_recorded"] is True


def test_suppressed_nudge_writes_no_stamp(tmp_path):
    # Verbs policy-hidden: the ctx flag is still latched, the message is empty,
    # and NO stamp lands — the flag can never impersonate a fired nudge.
    from ouroboros import delegate_custody as custody
    from ouroboros.loop import _maybe_inject_finalization_nudges

    drive = _drive(tmp_path)
    ctx = SimpleNamespace(
        _nanny_route_dispatched=True, _nanny_finalization_injected=False,
        task_metadata={"budget_drive_root": str(drive)},
    )
    tools = SimpleNamespace(_ctx=ctx, available_tools=lambda: ["read_file"])
    assert _maybe_inject_finalization_nudges(
        tools, drive, "child-1",
        {"reasoning_notes": [], "tool_calls": []}, "done", [], lambda *_: None,
    ) is False
    assert ctx._nanny_finalization_injected is True  # the flag alone is not the signal
    assert custody.task_execution_evidence(drive, "child-1")["nanny_nudge_recorded"] is False


def test_nudge_fact_rides_only_completed_native_only_envelopes(tmp_path):
    from ouroboros.delegate_evidence import record_nanny_nudge_stamp

    drive = _drive(tmp_path)
    record_nanny_nudge_stamp(_stamp_ctx(drive), "child-1", "NANNY_DID_NOT_DELEGATE")

    # Nudge fired + zero starts + completed -> the typed fact rides the envelope.
    envelope = envelope_from_task(_harness_task(drive), status="completed")
    delta = envelope["capability_delta"]
    assert "nanny_finalized_after_nudge_without_delegation" in delta["substrate_disclosures"]
    assert "delegated_substrate_unused" in delta["substrate_disclosures"]
    assert delta["reason"] == (
        "delegated_substrate_unused; nanny_finalized_after_nudge_without_delegation")
    assert envelope["execution_evidence"]["nanny_nudge_recorded"] is True

    # Cancelled: the substrate amendment still discloses, the FINALIZED
    # accusation does not (completed-only, B3).
    cancelled = envelope_from_task(_harness_task(drive), status="cancelled")
    cdelta = cancelled["capability_delta"]
    assert "delegated_substrate_unused" in cdelta["substrate_disclosures"]
    assert "nanny_finalized_after_nudge_without_delegation" not in cdelta["substrate_disclosures"]


def test_a_refused_attempt_after_the_nudge_is_never_an_accusation(tmp_path):
    """Final-review finding (codex proton0): a nanny that OBEYED the nudge —
    called delegate_start and was refused typed before any run existed — must
    not be disclosed as having finalized "without delegation". Both durable
    attempt shapes count: the pre-mint typed blocker row and a request row
    whose POST then died."""
    from ouroboros import delegate_custody as custody
    from ouroboros.delegate_evidence import record_nanny_nudge_stamp, record_start_blocked

    # Shape 1: typed route_health blocker before any invocation was minted.
    drive = _drive(tmp_path / "blocked")
    record_nanny_nudge_stamp(_stamp_ctx(drive), "child-1", "NANNY_DID_NOT_DELEGATE")
    record_start_blocked(_stamp_ctx(drive), "child-1", "route_status_unavailable")
    evidence = custody.task_execution_evidence(drive, "child-1")
    assert evidence["delegate_start_attempted"] is True
    assert evidence["delegated_runs_started"] == 0
    envelope = envelope_from_task(_harness_task(drive), status="completed")
    delta = envelope["capability_delta"]
    assert "delegated_substrate_unused" in delta["substrate_disclosures"]
    assert "nanny_finalized_after_nudge_without_delegation" not in delta["substrate_disclosures"]

    # Shape 2: the durable request row landed, the POST failed, nothing started.
    drive2 = _drive(tmp_path / "requested")
    record_nanny_nudge_stamp(_stamp_ctx(drive2), "child-1", "NANNY_DID_NOT_DELEGATE")
    assert custody.emit(drive2, custody.START_REQUESTED, {
        "task_id": "child-1", "invocation_id": "inv-1", "route": "claude",
        "request": {"prompt": "x"}})
    envelope2 = envelope_from_task(_harness_task(drive2), status="completed")
    assert ("nanny_finalized_after_nudge_without_delegation"
            not in envelope2["capability_delta"]["substrate_disclosures"])

    # A truly idle nanny (stamp, zero attempts) is still disclosed.
    idle = _drive(tmp_path / "idle")
    record_nanny_nudge_stamp(_stamp_ctx(idle), "child-1", "NANNY_DID_NOT_DELEGATE")
    envelope3 = envelope_from_task(_harness_task(idle), status="completed")
    assert ("nanny_finalized_after_nudge_without_delegation"
            in envelope3["capability_delta"]["substrate_disclosures"])


def test_nudge_fact_absent_when_delegation_happened_or_nudge_never_fired(tmp_path):
    from ouroboros import delegate_custody as custody
    from ouroboros.delegate_evidence import record_nanny_nudge_stamp

    # Delegation happened (a run started): not native_only, no amendment at all.
    drive = _drive(tmp_path / "ran")
    record_nanny_nudge_stamp(_stamp_ctx(drive), "child-1", "NANNY_DELEGATED_RUN_PENDING")
    assert custody.emit(drive, custody.STARTED, {
        "run_id": "run-1", "task_id": "child-1", "route": "claude", "max_seconds": 300})
    envelope = envelope_from_task(_harness_task(drive), status="completed")
    assert envelope["actual_substrate"] == "harness_attempted"
    assert envelope["capability_delta"]["reduced"] is False

    # Nudge never fired (no stamp): native_only still discloses the unused
    # substrate, without the nudge-ignored accusation.
    quiet = _drive(tmp_path / "quiet")
    envelope = envelope_from_task(_harness_task(quiet), status="completed")
    delta = envelope["capability_delta"]
    assert delta["substrate_disclosures"] == ["delegated_substrate_unused"]
    assert delta["reason"] == "delegated_substrate_unused"


# ---------------------------------------------------------------- B5: schema descriptions


def test_schedule_subagent_descriptions_carry_the_delegation_guidance():
    from ouroboros.tools.control import schedule_subagent_properties

    props = schedule_subagent_properties()
    objective = props["objective"]["description"]
    assert "OUTCOME" in objective and "script" in objective
    context = props["context"]["description"]
    assert "not instructions" in context and "WORK ORDER" in context
    lane = props["model_lane"]["description"]
    assert "Leave auto" in lane and "OVERRIDES dispatch policy" in lane
    # Description-only change: the schema SHAPE is untouched.
    assert props["model_lane"]["enum"] == ["auto", "main", "heavy", "light"]
    assert props["model_lane"]["default"] == "auto"
    assert all("type" in spec for spec in props.values())


# ------------------------------------------------- as_dict carries the additive fields


def test_capability_delta_as_dict_carries_the_typed_lists_additively():
    delta = CapabilityDelta().as_dict()
    assert delta["reduction_reasons"] == []
    assert delta["substrate_disclosures"] == []
    # The pre-split keys all survive unchanged (additive-only, decision 7A).
    for key in ("requested_lane", "resolved_lane", "effective_lane", "lane_provenance",
                "derived_effort", "effective_effort", "requested_executor",
                "effective_executor", "reason", "reduced", "legacy_note"):
        assert key in delta


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-x", "-q"]))
