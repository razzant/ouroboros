"""Model answer identity and host disclosures through the real terminal consumers."""

import asyncio
from collections import deque
import hashlib
import json
import queue
from types import SimpleNamespace

import pytest

from ouroboros import agent_task_pipeline as pipeline, loop
from ouroboros.task_results import load_task_result, write_task_result
from tests._delivery_candidate_shared import write_child, write_confirmed_disposition_fixture
from tests.test_delivery_forced_finalization import _bind_host_pass, _forced_test_context
from tests.test_ui_smoke_playwright import direct_server_with_data as _direct_server_with_data

direct_server_with_data = _direct_server_with_data


ANSWER = "Exact model answer: λ\n\nThe useful result."
NOTICE = "Plan review remained open.\n\n⚠️ Deferred child result: child1."
# The owner's incident decision, fed to the real producer
# (ouroboros/owner_hurry.py plan_review_disclosure) rather than to a copied string.
ADVISORY_PLAN_DECISION = {
    "required": True, "status": "open", "outcome": "DEGRADED",
    "enforcement": "advisory", "allow": True,
}


@pytest.mark.parametrize("change", ["generation", "superseded", "replaced_panel"])
def test_equal_evidence_fields_cannot_revive_changed_owner_authority(tmp_path, monkeypatch, change):
    _loop, tools, ctx, trace = _forced_test_context(tmp_path)
    old = loop._replace_delivery_candidate(tools, ctx, trace, ANSWER, control="candidate")
    prior = _bind_host_pass(loop, tools, trace, old)
    before = old.evidence_revision, old.evidence_fingerprint
    if change == "generation":
        tools._ctx._task_acceptance_owner_generation = 1
        tools._ctx.owner_message_admission_agent = SimpleNamespace(_owner_message_generation=2)
    elif change == "superseded":
        # The release notification follows real ingress; it is no longer a
        # declaration that the task's meaning changed by itself.
        tools._ctx._owner_directives = [{"content": "Use the newly supplied source."}]
        loop._supersede_task_acceptance_for_owner_followup(tools._ctx, trace)
    else:
        prior["superseded_by_revision"] = True
        trace["review_runs"].append({**prior, "superseded_by_revision": False,
                                   "panel_id": "new-panel", "binding_hash": "new-binding", "aggregate_signal": "FAIL"})
        trace["review_decision"].update(panel_id="new-panel", binding_hash="new-binding")
    assert loop._current_delivery_candidate(ctx, trace) is None
    assert (old.evidence_revision, old.evidence_fingerprint) == before
    monkeypatch.setattr(loop, "call_llm_with_retry", lambda *_a, **_kw: (None, 0.0))
    text, usage, _trace = loop._forced_final_answer(
        ctx, prompt="finalize", fallback_text=ANSWER, reason_code="round_limit",
    )
    current = tools._ctx._delivery_candidate
    assert text == ANSWER and current is not old
    assert current.acceptance_binding["authoritative"] is False
    assert current.acceptance_binding["stale_evidence"] is True
    assert prior["superseded_by_revision"] is True
    assert "STALE-EVIDENCE NOTICE" in usage["terminal_host_notice"]


@pytest.mark.parametrize("answer", [ANSWER, " \n"])
def test_normal_finalization_without_a_candidate_keeps_raw_answer(tmp_path, monkeypatch, answer):
    _loop, tools, ctx, trace = _forced_test_context(tmp_path)
    assert loop._live_delivery_candidate(ctx) is None
    monkeypatch.setenv("OUROBOROS_TASK_REVIEW_MODE", "off")
    monkeypatch.setattr(loop, "_maybe_inject_finalization_nudges", lambda *_a, **_kw: False)
    monkeypatch.setattr(loop, "_force_plan_disclosure", lambda *_a, **_kw: NOTICE)
    text, usage, _trace = loop._no_tool_final_answer(answer, ctx, trace, tools, queue.Queue(), set(), lambda _t: None)
    assert text == answer and usage["terminal_host_notice"] == NOTICE
    assert NOTICE not in text


def test_simultaneous_plan_and_deferred_limitations_stay_one_notice_beside_a_clean_answer(
    tmp_path, monkeypatch,
):
    """Both incident facts at once on the NORMAL rail (ouroboros/loop_delivery.py).

    The answer keeps its own bytes, the two host sentences compose ONE notice in
    producer order, and the six typed fields every benchmark adapter republishes
    (devtools/benchmarks/common/result_index.py) keep today's values. The single
    ``degraded_reason`` slot holds the CHILD fact while the plan fact lives only
    in the notice text; a commit that also carries the plan fact in typed state
    must do so ADDITIVELY and leave this slot alone.

    The disclosure is injected through ``_force_plan_disclosure`` exactly as the
    neighbouring tests do, so what is pinned here is the COMPOSITION and the
    typed outcome, never the predicate that decides whether a review is open.
    """
    from ouroboros.outcomes import derive_loop_outcome
    from ouroboros.owner_hurry import plan_review_disclosure

    write_child(tmp_path)
    write_confirmed_disposition_fixture(
        tmp_path, disposition="deferred", rationale="defer until the next run",
    )
    plan_suffix = plan_review_disclosure(ADVISORY_PLAN_DECISION)
    assert plan_suffix.strip(), "the advisory branch still produces a disclosure"

    loop_mod, registry, ctx, trace = _forced_test_context(tmp_path)
    monkeypatch.setattr(loop_mod, "_compute_subagent_handoff", lambda *_a, **_k: None)
    monkeypatch.setattr(loop_mod, "_maybe_inject_finalization_nudges", lambda *_a, **_k: False)
    monkeypatch.setattr(loop_mod, "_force_plan_disclosure", lambda *_a, **_k: plan_suffix)
    monkeypatch.setattr(loop_mod, "_run_task_acceptance_review_once", lambda **_kw: False)

    result = loop_mod._no_tool_final_answer(
        ANSWER, ctx, trace, registry, queue.Queue(), set(), lambda _text: None,
    )
    assert result is not None
    text, usage, returned_trace = result

    # The answer is the model's alone: benchmark scorers read these bytes.
    assert text == ANSWER
    notice = usage["terminal_host_notice"]
    assert notice not in text
    assert "Plan review" not in text and "DEFERRED CHILD RESULTS" not in text

    # ONE notice: producer order (plan first), one blank-line join, prefix intact.
    plan_part, separator, orphan_part = notice.partition("\n\n")
    assert plan_part == plan_suffix.strip() and separator == "\n\n"
    assert orphan_part.startswith("⚠️ DEFERRED CHILD RESULTS: child1")
    assert notice.startswith("⚠️") and notice.count("DEFERRED CHILD RESULTS") == 1

    # One degraded slot: the child fact wins while both facts are true.
    candidate = registry._ctx._delivery_candidate
    assert candidate.degraded is True
    assert candidate.degraded_reason == "host_child_status_suffix"
    assert candidate.model_text == ANSWER and candidate.full_text == ANSWER

    outcome = derive_loop_outcome(text, usage, returned_trace)
    assert outcome["degraded"] is True
    assert outcome["degraded_reason"] == "host_child_status_suffix"
    assert outcome["reason_code"] == "child_results_deferred"
    execution = outcome["outcome_axes"]["execution"]
    assert execution["status"] == "degraded"
    assert execution["reason_code"] == "child_results_deferred"
    assert execution["failure"]["deferred_count"] == 1
    assert outcome["outcome_axes"]["objective"]["status"] == "best_effort"


@pytest.mark.parametrize("verdict", ["FAIL", "PASS"])
def test_host_notice_does_not_replace_or_supersede_an_unchanged_answer(tmp_path, monkeypatch, verdict):
    import ouroboros.review_substrate as rs
    from ouroboros.contracts.task_contract import build_task_contract

    _loop, tools, ctx, trace = _forced_test_context(tmp_path)
    tools._ctx.task_contract = build_task_contract({"id": "parent1", "expected_output": "A report"})
    write_task_result(tmp_path, "parent1", "running", task_contract=tools._ctx.task_contract)
    monkeypatch.setenv("OUROBOROS_TASK_REVIEW_MODE", "auto")
    monkeypatch.setattr(rs, "triad_delivery_slots", lambda **_kw: [object()])
    monkeypatch.setattr(loop, "_maybe_inject_finalization_nudges", lambda *_a, **_kw: False)
    monkeypatch.setattr(loop, "_force_plan_disclosure", lambda *_a, **_kw: NOTICE)
    subjects = []

    def review(request, **_kwargs):
        subjects.append(request.subject)
        return rs.ReviewRunResult(
            request={"surface": "task_acceptance", "policy": {"min_successful_slots": 1}},
            actors=[{"slot_id": "s0", "signal": verdict, "parsed": {
                "verdict": verdict, "outcome_tier": "solved" if verdict == "PASS" else "best_effort",
                "criteria_used": [{"criterion": "report", "status": "supported", "evidence_refs": ["artifact:1"]}],
            }}], parsed_findings=[], aggregate_signal=verdict,
        )

    monkeypatch.setattr(rs, "run_review_request", review)

    def finalize():
        result = loop._no_tool_final_answer(ANSWER, ctx, trace, tools, queue.Queue(), set(), lambda _t: None)
        if result is not None:
            assert result[0] == ANSWER
            assert result[1]["terminal_host_notice"] == NOTICE
        return result

    first = finalize()
    assert (first is None) is (verdict == "FAIL")  # A real FAIL still requests its ordinary improvement.
    candidate = tools._ctx._delivery_candidate
    assert finalize() is not None  # An unchanged answer reuses the verdict; its notice cannot demand a rewrite.
    binding = dict(candidate.acceptance_binding)
    assert finalize() is not None
    assert subjects == [ANSWER]
    assert tools._ctx._delivery_candidate is candidate
    assert candidate.content_sha256 == hashlib.sha256(ANSWER.encode()).hexdigest()
    assert candidate.revision == 1 and candidate.acceptance_binding == binding
    assert not trace["review_runs"][0].get("superseded_by_revision")
    assert trace["acceptance_decision"]["status"] == ("accepted" if verdict == "PASS" else "finalized_unaccepted")

    # #533: transport/finalization warning must not erase the bound assessment.
    from ouroboros.outcomes import derive_loop_outcome, normalize_outcome_axes, public_task_result
    from ouroboros.task_results import load_task_result

    trace["delivery_candidate"].update(degraded=True, degraded_reason="advisory_plan_review_open")
    usage = {"terminal_host_notice": NOTICE}
    axes = derive_loop_outcome(ANSWER, usage, trace)["outcome_axes"]
    expected = "pass" if verdict == "PASS" else "fail"
    assert axes["objective"]["status"] == expected
    assert axes["objective"]["source"] == "task_acceptance_review"
    assert axes["execution"]["status"] == "degraded"
    env = SimpleNamespace(drive_root=tmp_path, repo_dir=tmp_path)
    task = {"id": "parent1", "type": "task", "chat_id": 1, "text": "Produce a report",
            "task_contract": tools._ctx.task_contract, "_skip_post_task_synthesis": True}
    pipeline._store_task_result(env, task, ANSWER, usage, trace)
    stored = load_task_result(tmp_path, "parent1")
    assert stored["outcome_axes"]["objective"]["status"] == expected
    assert normalize_outcome_axes(public_task_result(stored))["objective"]["source"] == "task_acceptance_review"
    pending = []
    pipeline.emit_task_results(env, None, None, pending, task, ANSWER, usage, trace,
                              start_time=0.0, drive_logs=tmp_path / "logs")
    terminal = next(row for row in pending if row["type"] == "task_done")
    assert terminal["outcome_axes"]["objective"]["status"] == expected
    assert load_task_result(tmp_path, "parent1")["outcome_axes"]["execution"]["status"] == "degraded"

    # Real input changes still supersede the binding even when answer bytes match.
    tools._ctx._owner_directives = [{"text": "Use the newly supplied source."}]
    changed = loop._replace_delivery_candidate(tools, ctx, trace, ANSWER, control="candidate")
    assert changed is not candidate and changed.revision > candidate.revision
    assert changed.acceptance_binding["authoritative"] is False
    assert trace["review_runs"][0]["superseded_by_revision"] is True


def _emit_terminal(tmp_path, monkeypatch, *, direct=False, project=False, child=False, notice=NOTICE, answer=ANSWER):
    from ouroboros.task_finalization import set_terminal_host_notice

    monkeypatch.setattr(pipeline, "_run_post_task_processing_async", lambda *_a, **_kw: None)
    task = {"id": "notice-root", "type": "task", "chat_id": 1, "text": "Produce the report."}
    if child:
        task.update(id="child1", parent_task_id="parent1", root_task_id="parent1", delegation_role="subagent")
    if project:
        from ouroboros.projects_registry import bind_task_to_project, create_project

        row = create_project(tmp_path, "notice-project", name="Research")
        bind_task_to_project(tmp_path, task["id"], row["id"], row["chat_id"], origin={"absent": "system"})
        task.update(project_id=row["id"], chat_id=row["chat_id"])
    if direct:
        task.update(_is_direct_chat=True)
    usage = {"terminal_origin": "model_final"}
    set_terminal_host_notice(usage, notice)
    pending = []
    pipeline.emit_task_results(
        SimpleNamespace(drive_root=tmp_path, repo_dir=tmp_path), None, None,
        pending, task, answer, usage, {"tool_calls": [], "reasoning_notes": []},
        start_time=0.0, drive_logs=tmp_path / "logs",
    )
    return task, next(row for row in pending if row["type"] == "send_message")


@pytest.mark.parametrize("answer", [ANSWER, ""], ids=["answer", "no_answer"])
@pytest.mark.parametrize("notice", [NOTICE, NOTICE + "\n" + "retained host evidence " * 1500 + "\nEND NOTICE"],
                         ids=["short_notice", "long_notice"])
def test_parent_readers_receive_full_notice_and_budget_the_complete_body(tmp_path, monkeypatch, answer, notice):
    from ouroboros.task_finalization import provider_terminal_body
    from ouroboros.task_status import format_subagent_absorption_message
    from ouroboros.tools.control_task_results import _get_task_result, _wait_for_task
    from tests.test_child_result_disposition import _parent_ctx

    task, _event = _emit_terminal(tmp_path, monkeypatch, child=True, answer=answer, notice=notice)
    stored = load_task_result(tmp_path, task["id"])
    assert stored["result"] == answer and stored["terminal_host_notice"] == notice
    model_hash = hashlib.sha256(stored["result"].encode()).hexdigest()
    parent = _parent_ctx(tmp_path)
    from tests.wait_handoff_support import full_wait_payload
    observed = full_wait_payload(parent, _wait_for_task(parent, task["id"], timeout_sec=0))
    assert observed["tasks"][task["id"]]["result"] == answer
    assert observed["tasks"][task["id"]]["terminal_host_notice"] == notice
    for output in (_get_task_result(parent, task["id"]),):
        if answer:
            assert f"[BEGIN_SUBTASK_OUTPUT]\n{answer}\n[END_SUBTASK_OUTPUT]" in output
        else:
            assert stored["status"] == "failed" and "No details available." in output
        assert output.endswith("[Host status]\n" + notice)
        assert output.count(notice) == 1

    body = provider_terminal_body(answer, notice)
    full = format_subagent_absorption_message([stored], parent_task_id="parent1", budget_chars=len(body))
    assert body in full and "FULL RESULT OMITTED" not in full
    omitted = format_subagent_absorption_message([stored], parent_task_id="parent1", budget_chars=len(body) - 1)
    assert f"{len(body)} chars" in omitted and 'get_task_result("child1")' in omitted
    assert notice not in omitted and (not answer or answer not in omitted)
    combined = format_subagent_absorption_message(
        [stored, {**stored, "task_id": "child2"}], parent_task_id="parent1", budget_chars=2 * len(body) - 1,
    )
    assert combined.count(body) == 1 and 'get_task_result("child2")' in combined
    assert hashlib.sha256(load_task_result(tmp_path, task["id"])["result"].encode()).hexdigest() == model_hash


def test_changed_notice_reopens_parent_disposition_and_automatic_handoff(tmp_path, monkeypatch):
    from ouroboros.task_finalization import set_terminal_host_notice, terminal_result_fields
    from ouroboros.task_status import load_effective_task_result
    from ouroboros.tools.join_ledger import _child_result_sha256, _current_child_result_disposition
    from ouroboros.tools.task_tree import _tree_note
    from tests.test_child_result_disposition import _parent_ctx, _payload

    task, _event = _emit_terminal(tmp_path, monkeypatch, child=True)
    ctx = _parent_ctx(tmp_path)
    tools = SimpleNamespace(_ctx=ctx)
    first = load_effective_task_result(tmp_path, task["id"])
    old_hash = _child_result_sha256(first)
    model_hash = hashlib.sha256(first["result"].encode()).hexdigest()
    assert NOTICE in loop._compute_subagent_handoff(tools, tmp_path, "parent1", "")
    payload = _payload(task["id"], "integrated", old_hash)
    assert _tree_note(ctx, "decision", "absorbed answer and host notice", payload=payload).startswith("OK:")
    assert _current_child_result_disposition(load_effective_task_result(tmp_path, task["id"])) == "integrated"
    assert loop._compute_subagent_handoff(tools, tmp_path, "parent1", "") == ""

    notice = "The child result was preserved before newer evidence arrived."
    usage = {}
    set_terminal_host_notice(usage, notice)
    write_task_result(tmp_path, task["id"], first["status"], **terminal_result_fields(usage))
    changed = load_effective_task_result(tmp_path, task["id"])
    assert changed["result"] == first["result"] == ANSWER
    assert hashlib.sha256(changed["result"].encode()).hexdigest() == model_hash
    assert _child_result_sha256(changed) != old_hash
    assert _current_child_result_disposition(changed) == ""
    assert "CHILD_RESULT_STALE" in _tree_note(ctx, "decision", "old consumption", payload=payload)
    handoff = loop._compute_subagent_handoff(tools, tmp_path, "parent1", "")
    assert ANSWER + "\n\n[Host status]\n" + notice in handoff
    assert _child_result_sha256(changed) in handoff
    assert loop._compute_subagent_handoff(tools, tmp_path, "parent1", "") == ""


@pytest.mark.parametrize("mode", ["all_terminal", "any_terminal"])
@pytest.mark.parametrize("answer", [ANSWER, ANSWER + "\n" + "model detail " * 1500, ""],
                         ids=["short_answer", "long_answer", "no_answer"])
@pytest.mark.parametrize("notice", [NOTICE, NOTICE + "\n" + "retained host evidence " * 1500 + "\nEND NOTICE"],
                         ids=["short_notice", "long_notice"])
def test_batch_wait_delivers_notice_before_current_hash_disposition(tmp_path, monkeypatch, mode, answer, notice):
    """The real batch reader must deliver the warning before disposition hides handoff."""
    from ouroboros.task_status import load_effective_task_result
    from ouroboros.tools.join_ledger import _child_result_sha256, _current_child_result_disposition
    from ouroboros.tools.registry import ToolRegistry
    from tests.test_child_result_disposition import _payload

    task, _event = _emit_terminal(tmp_path, monkeypatch, child=True, answer=answer, notice=notice)
    stored = load_task_result(tmp_path, task["id"])
    tools = ToolRegistry(tmp_path / "repo", tmp_path / "parent-execution")
    tools._ctx.task_id = "parent1"
    tools._ctx.task_metadata = {"budget_drive_root": str(tmp_path), "root_task_id": "parent1"}
    args = {"task_ids": [task["id"]], "timeout_sec": 0, "mode": mode}
    result = tools.execute_result("wait_tasks", args)
    assert result.status == "ok"
    from tests.wait_handoff_support import full_wait_payload
    batch = full_wait_payload(tools._ctx, result.text)
    assert batch["all_terminal"] is True
    shown = batch["tasks"][task["id"]]
    assert shown["result"] == answer
    assert shown["terminal_host_notice"] == notice
    assert notice not in shown["result"]
    assert shown["child_result_sha256"] == _child_result_sha256(load_effective_task_result(tmp_path, task["id"]))
    assert "get_task_result" in batch["tasks_note"]
    assert not {"trace_refs", "loop_outcome", "verification_ledger"} & shown.keys()

    disposition = tools.execute("tree_note", {
        "kind": "decision", "text": "Absorbed the answer and its separately authored host limitation.",
        "payload": _payload(task["id"], "integrated", shown["child_result_sha256"]),
    })
    assert disposition.startswith("OK:")
    assert _current_child_result_disposition(load_effective_task_result(tmp_path, task["id"])) == "integrated"
    assert loop._compute_subagent_handoff(tools, tmp_path, "parent1", "") == ""
    assert full_wait_payload(tools._ctx, tools.execute("wait_tasks", args))["tasks"][task["id"]] == shown
    assert tools.execute("get_task_result", {"task_id": task["id"]}).endswith("[Host status]\n" + notice)
    assert load_task_result(tmp_path, task["id"]) == stored


def test_batch_wait_without_notice_keeps_the_original_projection(tmp_path, monkeypatch):
    from ouroboros.outcomes import normalize_outcome_axes
    from ouroboros.task_status import load_effective_task_result
    from ouroboros.tools.control_task_results import _wait_for_tasks
    from ouroboros.tools.join_ledger import _child_result_sha256
    from tests.test_child_result_disposition import _parent_ctx

    task, _event = _emit_terminal(tmp_path, monkeypatch, child=True, notice="")
    stored = load_task_result(tmp_path, task["id"])
    current = load_effective_task_result(tmp_path, task["id"])
    assert "terminal_host_notice" not in current
    batch = json.loads(_wait_for_tasks(_parent_ctx(tmp_path), [task["id"]], timeout_sec=0))
    assert batch["tasks"][task["id"]] == {
        "task_id": task["id"], "status": current["status"],
        "accounted_upper_bound_usd": current["accounted_upper_bound_usd"],
        "cost_final": current.get("cost_final"),
        "child_result_sha256": _child_result_sha256(current),
        "outcome_axes": normalize_outcome_axes(current),
        "result": ANSWER, "trace_summary": current.get("trace_summary"),
        "execution_observation": current["execution_observation"],
        "result_chars": len(ANSWER), "trace_summary_chars": len(current.get("trace_summary") or ""),
        "result_source": {"tool": "get_task_result", "arguments": {"task_id": task["id"], "include_authority": True}},
        "verification_summary": current["verification_ledger"].get("summary") or {},
    }
    assert load_task_result(tmp_path, task["id"]) == stored


def test_child_notice_hash_extension_preserves_legacy_hash_and_telemetry_exclusions():
    from ouroboros.tools.join_ledger import _child_result_sha256

    legacy = {"status": "completed", "result": "legacy answer", "trace_summary": "trace",
              "artifact_status": "ready", "artifacts": []}
    assert _child_result_sha256(legacy) == "cc3314bd27a9639006ccfafe9500bf25cfe5c3c6746b47c4d40736768b8b5985"
    current = {**legacy, "terminal_host_notice": NOTICE}
    assert _child_result_sha256(current) != _child_result_sha256(legacy)
    assert _child_result_sha256({**current, "terminal_host_notice": "changed"}) != _child_result_sha256(current)
    telemetry = {"cost_usd": 9, "accounted_upper_bound_usd": 10, "updated_at": "later", "ts": "later",
                 "parent_decision": "integrated", "queue_reconciliation_warning": "diagnostic",
                 "terminal_provider_notice": "older metadata remains outside this hash",
                 "terminal_origin": "host_salvage"}
    for row in (legacy, current):
        assert _child_result_sha256({**row, **telemetry}) == _child_result_sha256(row)

    # An EMPTY key is not an absent key. ``set_terminal_host_notice``
    # (ouroboros/task_finalization.py) POPS the key when the composed
    # notice is blank, which is the only reason clean children keep the legacy
    # hash asserted above. A writer that ALWAYS set the key would silently
    # re-hash every clean child result and invalidate every parent's
    # recorded disposition.
    from ouroboros.task_finalization import set_terminal_host_notice

    assert _child_result_sha256({**legacy, "terminal_host_notice": ""}) != _child_result_sha256(legacy)
    blank = {"terminal_host_notice": "stale"}
    set_terminal_host_notice(blank)
    assert "terminal_host_notice" not in blank

    # With an open delegated-custody audit the key is inserted from
    # terminal_host_notice_text even when nothing was stored, so "drop the
    # stored notice" is NOT by itself hash-preserving (join_ledger.py).
    custody = {**legacy, "delegate_terminal_reconciliation": {
        "audit_status": "ok", "open_run_ids": ["run-open"], "terminal_runs": [],
        "pending_invocation_ids": [], "undisposed_patch_run_ids": []}}
    assert _child_result_sha256(custody) != _child_result_sha256(legacy)
    assert _child_result_sha256({**custody, "terminal_host_notice": NOTICE}) != _child_result_sha256(custody)


@pytest.mark.parametrize("direct", [False, True])
@pytest.mark.parametrize("project", [False, True])
def test_a_host_notice_never_becomes_a_second_chat_row(tmp_path, monkeypatch, direct, project):
    """One voice: the disclosure stays a typed field OF THE RESULT (replay reads it
    from the stored row) and the chat carries the model's answer alone, live and on
    history replay; the outbox owes exactly that one row."""
    from ouroboros.gateway.history import make_chat_history_endpoint
    from ouroboros.utils import append_jsonl
    from supervisor import events_chat_delivery as delivery, message_bus
    from supervisor.terminal_delivery import build_completed_result_event, pending_deliveries

    task, event = _emit_terminal(tmp_path, monkeypatch, direct=direct, project=project)
    assert event["text"] == event["log_text"] == ANSWER
    assert "terminal_host_notice" not in event
    stored = load_task_result(tmp_path, task["id"])
    assert stored["result"] == ANSWER and stored["terminal_host_notice"] == NOTICE
    replay = build_completed_result_event(tmp_path, task, task["id"], stored)
    assert replay["text"] == ANSWER and "terminal_host_notice" not in replay
    assert replay["delivery_id"] == event["delivery_id"]
    assert "terminal_host_notice" not in pending_deliveries(tmp_path)[0]

    bridge = message_bus.LocalChatBridge({})
    frames = []
    bridge._broadcast_fn = frames.append
    monkeypatch.setattr(message_bus, "DATA_DIR", tmp_path)
    monkeypatch.setattr(message_bus, "get_bridge", lambda: bridge)
    monkeypatch.setattr(message_bus, "load_state", lambda: {"owner_id": 7})
    monkeypatch.setattr(message_bus, "_advance_project_visible_revision", lambda _chat: None)
    monkeypatch.setattr(message_bus, "publish_event", lambda *_a, **_kw: None)
    monkeypatch.setattr(delivery, "_DELIVERED_MESSAGE_IDS", deque(maxlen=256))
    ctx = SimpleNamespace(DRIVE_ROOT=tmp_path, RUNNING={}, append_jsonl=append_jsonl,
                          send_with_budget=message_bus.send_with_budget)
    delivery._handle_send_message(event, ctx)
    delivery._handle_send_message(event, ctx)
    chats = [row for row in frames if row.get("type") == "chat"]
    assert [(row["role"], row["content"]) for row in chats] == [("assistant", ANSWER)]
    assert all(row["chat_id"] == task["chat_id"] for row in chats)
    response = asyncio.run(make_chat_history_endpoint(tmp_path)(SimpleNamespace(query_params={"chat_id": str(task["chat_id"])})))
    messages = json.loads(response.body)["messages"]
    assert [(row["role"], row["text"]) for row in messages] == [("assistant", ANSWER)]
    assert pending_deliveries(tmp_path) == []


def test_the_notice_field_survives_every_machine_reader_without_a_chat_row(tmp_path, monkeypatch, capsys):
    """The do-not-break proof in one test: the stored field keeps its bytes and its
    place in the child-result hash, every machine reader (public result, parent
    reader, synthesis, CLI stderr) still receives it, and the chat carries ONE row."""
    from ouroboros import cli
    from ouroboros.gateway.tasks import api_task_get
    from ouroboros.outcomes import public_task_result
    from ouroboros.task_finalization import build_sealed_final_package, sealed_final_prompt_section
    from ouroboros.tools.control_task_results import _get_task_result
    from ouroboros.tools.join_ledger import _child_result_sha256
    from ouroboros.utils import append_jsonl
    from supervisor import events_chat_delivery as delivery, message_bus
    from tests.test_child_result_disposition import _parent_ctx

    task, event = _emit_terminal(tmp_path, monkeypatch, child=True)
    stored = load_task_result(tmp_path, task["id"])
    assert stored["terminal_host_notice"] == NOTICE and stored["result"] == ANSWER
    without = {key: value for key, value in stored.items() if key != "terminal_host_notice"}
    assert _child_result_sha256(stored) != _child_result_sha256(without)  # the field is hashed
    assert public_task_result(stored)["terminal_host_notice"] == NOTICE
    assert _get_task_result(_parent_ctx(tmp_path), task["id"]).endswith("[Host status]\n" + NOTICE)
    assert NOTICE in sealed_final_prompt_section(build_sealed_final_package(stored, ANSWER))
    response = asyncio.run(api_task_get(SimpleNamespace(
        path_params={"task_id": task["id"]}, app=SimpleNamespace(state=SimpleNamespace(drive_root=tmp_path)),
    )))
    cli_row = {**json.loads(response.body), "cost_final": True, "cost_with_children_partial": False}
    monkeypatch.setattr(cli, "_client", lambda *_a, **_kw: SimpleNamespace(
        request=lambda *_a, **_kw: {"task_id": task["id"]}))
    monkeypatch.setattr(cli, "_wait_task", lambda *_a, **_kw: cli_row)
    assert cli.main(["run", "--no-stream", "Produce the report."]) == 0
    assert capsys.readouterr().err == "[Host status]\n" + NOTICE + "\n"

    bridge = message_bus.LocalChatBridge({})
    frames = []
    bridge._broadcast_fn = frames.append
    monkeypatch.setattr(message_bus, "DATA_DIR", tmp_path)
    monkeypatch.setattr(message_bus, "get_bridge", lambda: bridge)
    monkeypatch.setattr(message_bus, "load_state", lambda: {"owner_id": 7})
    monkeypatch.setattr(message_bus, "_advance_project_visible_revision", lambda _chat: None)
    monkeypatch.setattr(message_bus, "publish_event", lambda *_a, **_kw: None)
    monkeypatch.setattr(delivery, "_DELIVERED_MESSAGE_IDS", deque(maxlen=256))
    delivery._handle_send_message(event, SimpleNamespace(
        DRIVE_ROOT=tmp_path, RUNNING={}, append_jsonl=append_jsonl, send_with_budget=message_bus.send_with_budget))
    chats = [row for row in frames if row.get("type") == "chat"]
    assert [(row["role"], row["content"]) for row in chats] == [("assistant", ANSWER)]
    rows = [json.loads(line) for line in (tmp_path / "logs" / "chat.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [row["text"] for row in rows if not row.get("type")] == [ANSWER]


def test_the_answer_delivery_id_ignores_every_host_disclosure(tmp_path):
    """The outbox identity digests the CORE answer only
    (supervisor/terminal_delivery.py build_completed_result_event). A host
    disclosure that appears, changes or disappears must never re-mint it, or a
    replay delivers the same answer a second time."""
    from supervisor.terminal_delivery import build_completed_result_event, delivery_id_for

    task = {"id": "id-root", "chat_id": 1}
    expected = delivery_id_for("id-root", ANSWER)
    for extra in (
        {},
        {"terminal_host_notice": NOTICE},
        {"terminal_host_notice": NOTICE + "\n\nand more"},
        {"terminal_host_notice": NOTICE, "terminal_origin": "model_final"},
    ):
        event = build_completed_result_event(
            tmp_path, task, "id-root", {"result": ANSWER, **extra},
        )
        assert event is not None and event["delivery_id"] == expected, extra
        assert event["text"] == ANSWER, extra


def _open_delegated_custody(tmp_path, task_id):
    """Record one still-open delegated run through the real custody rail."""
    from ouroboros import delegate_custody as custody, delegate_terminal

    write_task_result(tmp_path, task_id, "completed", result=ANSWER,
                      delegated_runs_unreconciled=["stale-run"])
    assert custody.emit(tmp_path, custody.STARTED, {
        "run_id": "run-open", "task_id": task_id, "route": "fixture",
        "model": "fixture-model", "profile_id": "fixture-profile",
        "selected_subagent_id": "fixture-actor", "snapshot_id": "snapshot-one", "shape": {},
    })
    assert delegate_terminal.refresh_terminal_reconciliation(tmp_path, task_id)


@pytest.mark.parametrize("base", [NOTICE, ""], ids=["with_host_notice", "custody_only"])
def test_open_custody_is_its_own_card_row_live_and_on_history_replay(tmp_path, monkeypatch, base):
    """Issue #1006: open delegated execution is a typed row OF the task card.

    It rides its own field on the send event, is owed before the answer is
    sent, never carries the answer's phase, and replays under the same
    placement identity. Without a base notice it is the ONLY extra row.
    """
    from ouroboros.gateway.history import make_chat_history_endpoint
    from ouroboros.utils import append_jsonl
    from supervisor import events_chat_delivery as delivery, message_bus
    from supervisor.terminal_delivery import build_completed_result_event, pending_deliveries

    _open_delegated_custody(tmp_path, "notice-root")
    task, event = _emit_terminal(tmp_path, monkeypatch, notice=base)
    custody = event["terminal_custody_notice"]
    row_id = event["delivery_id"] + ":custody_notice"
    assert event["text"] == ANSWER and "Open delegated execution: run-open." in custody
    assert custody not in event["text"] and "terminal_host_notice" not in event
    assert event["progress_meta"]["task_phase"] == "finalizing"
    stored = load_task_result(tmp_path, task["id"])
    replay = build_completed_result_event(tmp_path, task, task["id"], stored)
    assert replay["text"] == ANSWER and replay["terminal_custody_notice"] == custody
    assert replay["delivery_id"] == event["delivery_id"]

    bridge = message_bus.LocalChatBridge({})
    frames = []
    bridge._broadcast_fn = frames.append
    monkeypatch.setattr(message_bus, "DATA_DIR", tmp_path)
    monkeypatch.setattr(message_bus, "get_bridge", lambda: bridge)
    monkeypatch.setattr(message_bus, "load_state", lambda: {"owner_id": 7})
    monkeypatch.setattr(message_bus, "_advance_project_visible_revision", lambda _chat: None)
    monkeypatch.setattr(message_bus, "publish_event", lambda *_a, **_kw: None)
    monkeypatch.setattr(delivery, "_DELIVERED_MESSAGE_IDS", deque(maxlen=256))
    owed_while_sending = []

    def send(chat_id, text, **kwargs):
        owed_while_sending.append({row["delivery_id"] for row in pending_deliveries(tmp_path)})
        return message_bus.send_with_budget(chat_id, text, **kwargs)

    ctx = SimpleNamespace(DRIVE_ROOT=tmp_path, RUNNING={}, append_jsonl=append_jsonl,
                          send_with_budget=send)
    delivery._handle_send_message(event, ctx)
    delivery._handle_send_message(event, ctx)
    assert row_id in owed_while_sending[0], "the custody row is owed before the answer is sent"
    # With or without a stored host notice the custody row is the ONLY extra row.
    expected = [("assistant", ANSWER), ("system", custody)]
    chats = [row for row in frames if row.get("type") == "chat"]
    assert [(row["role"], row["content"]) for row in chats] == expected
    assert all(row["chat_id"] == task["chat_id"] for row in chats)
    assert chats[-1]["system_type"] == "custody_notice"
    assert chats[-1]["card_row"] == "timeline" and chats[-1]["card_row_id"] == row_id
    assert not {"task_phase", "task_terminal_status"} & chats[-1].keys()
    response = asyncio.run(make_chat_history_endpoint(tmp_path)(
        SimpleNamespace(query_params={"chat_id": str(task["chat_id"])})))
    messages = json.loads(response.body)["messages"]
    assert [(row["role"], row["text"]) for row in messages] == expected
    assert messages[-1]["system_type"] == "custody_notice"
    assert messages[-1]["card_row"] == "timeline" and messages[-1]["card_row_id"] == row_id
    assert not messages[-1].get("task_terminal_status")
    assert pending_deliveries(tmp_path) == []


@pytest.mark.parametrize("outcome", ["message", "deferred", "silent", "tool_delivered"])
def test_presence_preserves_authored_speech_and_keeps_host_notice_in_task(tmp_path, monkeypatch, outcome):
    from ouroboros.presence_runner import PresenceTurnGate, run_presence_turn
    from tests.test_presence_runner import _admission, _event

    class Agent:
        def handle_task(self, task):
            task["_skip_post_task_synthesis"] = True
            ctx = SimpleNamespace(_presence_completion={"outcome": outcome, "message": ANSWER},
                                  _presence_completion_accepted=True,
                                  _swarm_handoff_attempt={"status": "scheduled", "task_id": "next-task"})
            pending = []
            pipeline.emit_task_results(
                SimpleNamespace(drive_root=tmp_path, repo_dir=tmp_path), None, None,
                pending, task, ANSWER, {"terminal_origin": "model_final", "terminal_host_notice": NOTICE},
                {"tool_calls": [], "reasoning_notes": []}, start_time=0.0, drive_logs=tmp_path / "logs", ctx=ctx,
            )
            return pending

    args = dict(admission=_admission(), event=_event(), repo_dir=tmp_path, drive_root=tmp_path,
                agent_factory=lambda **_kw: Agent(), gate=PresenceTurnGate(2))
    first = run_presence_turn(**args)
    assert run_presence_turn(**args) == first
    assert first.outcome == outcome
    assert load_task_result(tmp_path, first.task_id)["result"] == ANSWER
    assert first.text == (ANSWER if outcome in {"message", "deferred"} else "")
    assert load_task_result(tmp_path, first.task_id)["terminal_host_notice"] == NOTICE


def test_a_failed_answer_send_stays_owed_and_owes_no_second_row(tmp_path, monkeypatch):
    """The retry coverage the split used to carry: a failed FIRST send leaves exactly
    one owed row (the answer, no role), the retry delivers it once, and no
    notice row is ever owed or sent."""
    from ouroboros.utils import append_jsonl
    from supervisor import events_chat_delivery as delivery
    from supervisor.terminal_delivery import pending_deliveries

    _task, event = _emit_terminal(tmp_path, monkeypatch)
    sent = []
    attempts = []

    def fail_first(_chat, text, **kwargs):
        attempts.append(text)
        if len(attempts) == 1:
            raise OSError("transport failed")
        sent.append(text)

    monkeypatch.setattr(delivery, "_DELIVERED_MESSAGE_IDS", deque(maxlen=256))
    ctx = SimpleNamespace(DRIVE_ROOT=tmp_path, RUNNING={}, append_jsonl=append_jsonl, send_with_budget=fail_first)
    delivery._handle_send_message(event, ctx)
    assert sent == [] and attempts == [ANSWER]
    [owed] = pending_deliveries(tmp_path)
    assert owed["text"] == ANSWER and not owed.get("role")
    assert "terminal_host_notice" not in owed
    delivery._handle_send_message(event, ctx)
    assert sent == [ANSWER] and attempts == [ANSWER, ANSWER]
    assert pending_deliveries(tmp_path) == []


@pytest.mark.parametrize("jsonl", [False, True])
def test_cli_no_stream_keeps_notices_without_changing_result_bytes(tmp_path, monkeypatch, capsys, jsonl):
    from ouroboros import cli
    from ouroboros.gateway.tasks import api_task_get

    task, _event = _emit_terminal(tmp_path, monkeypatch)
    response = asyncio.run(api_task_get(SimpleNamespace(
        path_params={"task_id": task["id"]}, app=SimpleNamespace(state=SimpleNamespace(drive_root=tmp_path)),
    )))
    assert response.status_code == 200
    stored = {**json.loads(response.body), "cost_final": True, "cost_with_children_partial": False}
    monkeypatch.setattr(cli, "_client", lambda *_a, **_kw: SimpleNamespace(
        request=lambda *_a, **_kw: {"task_id": task["id"]},
    ))
    monkeypatch.setattr(cli, "_wait_task", lambda *_a, **_kw: stored)
    assert cli.main(["run", "--no-stream", *(["--jsonl"] if jsonl else []), "Produce the report."]) == 0
    captured = capsys.readouterr()
    if jsonl:
        final = json.loads(captured.out.splitlines()[-1])["result"]
        assert final["result"] == ANSWER and final["terminal_host_notice"] == NOTICE
        assert NOTICE not in captured.err
    else:
        assert captured.out == ANSWER + "\n"
        assert captured.err == "[Host status]\n" + NOTICE + "\n"


def test_synthesis_keeps_notice_authorship_separate():
    from ouroboros.task_finalization import build_sealed_final_package, sealed_final_prompt_section

    sealed = build_sealed_final_package({"terminal_host_notice": NOTICE}, ANSWER)
    assert sealed["final_result_text"] == ANSWER
    assert sealed["terminal_host_notice"] == NOTICE
    assert "Host-authored terminal notice (separate from the model answer):\n" + NOTICE in sealed_final_prompt_section(sealed)


@pytest.mark.ui_browser
def test_browser_renders_the_model_answer_without_a_host_bubble(direct_server_with_data, monkeypatch):
    """Real terminal producer, delivery writer, HTTP history and rendered SPA: the
    answer bubble alone, live and after a reload; no System bubble carries the notice."""
    from playwright.sync_api import sync_playwright
    from ouroboros.utils import append_jsonl
    from supervisor import events_chat_delivery as delivery, message_bus

    data = direct_server_with_data["data_dir"]
    _task, event = _emit_terminal(data, monkeypatch)
    bridge = message_bus.LocalChatBridge({})
    monkeypatch.setattr(message_bus, "DATA_DIR", data)
    monkeypatch.setattr(message_bus, "get_bridge", lambda: bridge)
    monkeypatch.setattr(message_bus, "load_state", lambda: {"owner_id": 7})
    monkeypatch.setattr(message_bus, "publish_event", lambda *_a, **_kw: None)
    monkeypatch.setattr(delivery, "_DELIVERED_MESSAGE_IDS", deque(maxlen=256))
    delivery._handle_send_message(event, SimpleNamespace(
        DRIVE_ROOT=data, RUNNING={}, append_jsonl=append_jsonl, send_with_budget=message_bus.send_with_budget,
    ))
    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        try:
            page = browser.new_page(viewport={"width": 1280, "height": 900})
            page.goto(direct_server_with_data["url"], wait_until="domcontentloaded")
            answer = page.locator(".chat-bubble.assistant").filter(has_text="Exact model answer")
            answer.wait_for(state="visible", timeout=15000)
            assert "Plan review remained open" not in answer.inner_text()
            assert page.locator(".chat-bubble.system").count() == 0
            page.screenshot(path=str(data.parent / "terminal-host-notice.png"), full_page=True)
            page.reload(wait_until="domcontentloaded")
            answer.wait_for(state="visible", timeout=15000)
            assert answer.count() == 1 and page.locator(".chat-bubble.system").count() == 0
        finally:
            browser.close()


def test_a_custody_split_never_mints_a_task_independent_row_id(tmp_path, monkeypatch):
    """An answer that reaches the delivery seam without its owed id still yields a
    custody row keyed by the task's canonical identity; with no task at all the
    custody fact is still its own TYPED row, only without a delivery id (a bare
    ``:custody_notice`` id would be suppressed by the delivered registry for every
    later task), and no host-notice field rides either row."""
    from types import SimpleNamespace

    from supervisor import events_chat_delivery as ecd
    from supervisor.terminal_delivery import delivery_id_for

    real = ecd._handle_send_message
    sent = []
    monkeypatch.setattr(ecd, "_handle_send_message", lambda evt, ctx: sent.append(dict(evt)))
    monkeypatch.setattr("supervisor.terminal_delivery.register_pending_delivery", lambda root, row: True)
    ctx = SimpleNamespace(DRIVE_ROOT=tmp_path)
    custody = "Open delegated execution: run-x."
    real({"type": "send_message", "chat_id": 1, "task_id": "t1", "text": "the answer",
          "terminal_custody_notice": custody}, ctx)
    answer, row = sent
    assert row["delivery_id"] == delivery_id_for("t1", "the answer") + ":custody_notice"
    assert row["progress_meta"]["card_row_id"] == row["delivery_id"]
    assert row["system_type"] == "custody_notice" and row["text"] == custody
    assert "terminal_custody_notice" not in answer and "delivery_id" not in answer
    sent.clear()
    real({"type": "send_message", "chat_id": 1, "text": "the answer",
          "terminal_host_notice": "Budget stop retained.", "terminal_custody_notice": custody}, ctx)
    answer, custody_row = sent
    assert answer["text"] == "the answer" and "terminal_custody_notice" not in answer
    assert custody_row["system_type"] == "custody_notice" and custody_row["text"] == custody
    assert custody_row["role"] == "system" and "delivery_id" not in custody_row
    assert "terminal_host_notice" not in answer and "terminal_host_notice" not in custody_row
