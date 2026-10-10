"""Ordinary exposed-source recovery and separately disclosed late unconfirmed-exposure rescue."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import json

import pytest

from ouroboros import context_compaction as cc
from ouroboros.context_budget import HOST_CONTEXT_KIND_KEY
from ouroboros.artifacts import read_actor_source_bytes
from ouroboros.context_source_view import _EMERGENCY_LABELS, emergency_address_view
from ouroboros.loop_messages import CONTEXT_FACTS_NAME
from ouroboros.peer_roster import ROSTER_SNAPSHOT_NAME, ROSTER_UPDATE_NAME
from ouroboros.tool_result_record import TOOL_RESULT_RECORD_KEY, make_tool_result_record
from tests.test_context_reclaim_materializer import _request, _SPEC
from tests.test_main_authored_context import main_loop as _main_loop

main_loop = _main_loop

BODY = "Exact producer bytes: Ж, e\u0301, emoji 🧠.\r\n" * 1000 + "CONSEQUENTIAL-TAIL\r\n"
OWNER = "Keep the original contract until the replacement is verified."


def _tool(call_id="reused", *, invocation=None, facts=None, text=BODY):
    result = {"role": "tool", "tool_call_id": call_id, "content": text}
    if invocation is not None:
        result[TOOL_RESULT_RECORD_KEY] = make_tool_result_record(
            {"invocation_id": invocation, "tool_call_id": call_id, "execution_id": "exec",
             "round_id": "round", "task_attempt": 1,
             "trace_ref": {"manifest_ref": {"path": f"trace/{invocation}.json", "sha256": "a" * 64}}},
            text, facts=facts, source_ref={"path": f"source/{invocation}.txt"})
    return [{"role": "assistant", "content": "", "tool_calls": [{
        "id": call_id, "type": "function", "function": {"name": "run_command", "arguments": '{"cmd":["fixture"]}'}}]},
        result]


def _prefix():
    return [{"role": "system", "content": "Required core."}, {"role": "user", "content": "Inspect the result."}]


def _observation(messages):
    return {"messages": deepcopy(messages), "exposed_units": [
        {"unit_id": u.unit_id, "raw_sha256": u.raw_sha256} for u in cc.context_units(messages, scope="dialogue")]}


def _records(messages):
    return [m for m in messages if cc._capsule_metadata(m)[1]]


def _visible(message):
    return message["content"][0]["text"]


def _checkpoint(tmp_path, receipt, task="source-view"):
    return json.loads(read_actor_source_bytes(tmp_path, task, receipt.checkpoint_ref))


def _assert_exact_restore(tmp_path, original, request, receipt):
    checkpoint = _checkpoint(tmp_path, receipt)
    assert checkpoint["messages"] == original
    assert cc.context_reclaim_transcript_sha256(checkpoint["messages"]) == request.transcript_sha256
    units = {u.unit_id: u for u in cc.context_units(original, scope="dialogue")}
    members = [ref for ref in receipt.source_refs if "unit_id" in ref]
    restored, _ = cc._restored_source_views(members, drive_root=tmp_path, task_id="source-view", request=request)
    assert len(restored) == len(members) > 0
    for ref, row in zip(members, restored):
        unit = units[ref["unit_id"]]
        assert ref["raw_sha256"] == unit.raw_sha256
        source = json.loads(_visible(row).split("\n", 2)[2])
        assert source == original[unit.start:unit.end + 1]
        assert row["role"] == "user" and not row.get("tool_calls")  # source, never live protocol replay


def test_only_producer_labelled_replaced_host_snapshots_become_addresses(tmp_path, monkeypatch):
    monkeypatch.setattr(cc, "_call_summarizer", lambda *_a, **_kw: pytest.fail("No helper for an address view"))
    old_facts = {"role": "user", HOST_CONTEXT_KIND_KEY: CONTEXT_FACTS_NAME, "content": "Past exact facts. " * 1600}
    old_roster = {"role": "user", HOST_CONTEXT_KIND_KEY: ROSTER_SNAPSHOT_NAME, "content": "Past full roster. " * 1600}
    old_delta = {"role": "user", HOST_CONTEXT_KIND_KEY: ROSTER_UPDATE_NAME, "content": "Past delta. " * 1600}
    warning = {"role": "user", "content": "A warning that remains operative. " * 600}
    latest_facts = {"role": "user", HOST_CONTEXT_KIND_KEY: CONTEXT_FACTS_NAME, "content": "Current full facts"}
    latest_roster = {"role": "user", HOST_CONTEXT_KIND_KEY: ROSTER_SNAPSHOT_NAME, "content": "Current full roster"}
    latest_delta = {"role": "user", HOST_CONTEXT_KIND_KEY: ROSTER_UPDATE_NAME, "content": "Current later change. " * 1000}
    messages = [*_prefix(), old_facts, warning, old_roster, old_delta,
                {"role": "user", "content": OWNER}, latest_facts, latest_roster, latest_delta]
    before = deepcopy(messages)
    request = _request(messages, 1)
    rebuilt, receipt = emergency_address_view(messages, request, rung="host_copies", protected_texts=(OWNER,),
                                             drive_root=tmp_path, task_id="source-view")
    assert receipt.status == "applied" and receipt.reclaimed_tokens > 0
    assert messages == before
    assert all(row not in rebuilt for row in (old_facts, old_roster, old_delta))
    assert all(row in rebuilt for row in (warning, latest_facts, latest_roster, latest_delta, messages[6]))
    assert rebuilt[:2] == messages[:2]
    assert all(_EMERGENCY_LABELS["host_copies"] in _visible(row) for row in _records(rebuilt))
    assert all("nothing summarized" in _visible(row) for row in _records(rebuilt))
    assert len(_records(rebuilt)) == 1 and receipt.selected_unit_ids == ()
    assert _checkpoint(tmp_path, receipt)["messages"] == messages
    assert '"units"' not in _visible(_records(rebuilt)[0])  # host indexes are not selectable body units
    assert emergency_address_view(messages, replace(request, transcript_sha256="0" * 64), rung="host_copies",
                                  drive_root=tmp_path, task_id="source-view")[1].status == "binding_mismatch"


def test_latest_delta_and_unlabelled_host_prose_do_not_replace_a_full_snapshot(tmp_path):
    messages = [*_prefix(),
        {"role": "user", HOST_CONTEXT_KIND_KEY: ROSTER_SNAPSHOT_NAME, "content": "Full roster. " * 1600},
        {"role": "user", HOST_CONTEXT_KIND_KEY: ROSTER_UPDATE_NAME, "content": "Later delta. " * 1600},
        {"role": "user", HOST_CONTEXT_KIND_KEY: CONTEXT_FACTS_NAME, "content": "Only facts row. " * 1600},
        {"role": "user", "content": "[CONTEXT_FACTS] unlabelled legacy text " * 1600},
        {"role": "user", "content": "[INDEPENDENT_ROOTS] unlabelled legacy roster " * 1600},
        {"role": "user", "content": "[Context view receipt] still relevant warning " * 1600}]
    rebuilt, receipt = emergency_address_view(messages, _request(messages, 1), rung="host_copies",
                                             drive_root=tmp_path, task_id="source-view")
    assert receipt.status == "no_eligible" and receipt.checkpoint_ref is None
    assert rebuilt == messages


def test_reused_call_id_and_identical_text_keep_both_visible_outcomes(tmp_path):
    failed = _tool(invocation="failed-attempt", facts={"status": "error", "code": "EXIT_NONZERO", "is_error": True, "exit_code": 17})
    succeeded = _tool(invocation="successful-attempt", facts={"status": "ok", "code": "OK", "is_error": False, "exit_code": 0})
    assert failed[-1]["content"] == succeeded[-1]["content"] and failed[-1]["tool_call_id"] == succeeded[-1]["tool_call_id"]
    messages = [*_prefix(), *failed, {"role": "user", "content": "Separate recorded attempts."}, *succeeded]
    request = _request(messages, 100_000)
    rebuilt, receipt = emergency_address_view(messages, request, rung="bodies",
        observation=_observation(messages),
        trace_refs_by_tool_call_id={"reused": {"path": "WRONG-global-most-recent.json"}},
        drive_root=tmp_path, task_id="source-view")
    assert receipt.status == "applied"
    records = _records(rebuilt)
    assert len(records) == 2 and not any(m.get("role") == "tool" for m in rebuilt)
    for row, invocation, exit_code, is_error in zip(records, ("failed-attempt", "successful-attempt"), (17, 0), (True, False)):
        text = _visible(row)
        assert f'"exit_code":{exit_code}' in text and f'"is_error":{json.dumps(is_error)}' in text
        assert f"invocation {invocation}" in text and f"trace/{invocation}.json" in text and f"source/{invocation}.txt" in text
        assert "outcome binding unknown" not in text and "CONSEQUENTIAL-TAIL" not in text
        assert "WRONG-global-most-recent" not in json.dumps(row)
    _assert_exact_restore(tmp_path, messages, request, receipt)


@pytest.mark.parametrize("change,reason", [("legacy", "record_missing"), ("content", "content_mismatch"),
                                          ("invocation", "invocation_unrecorded"), ("call", "tool_result_mismatch")])
def test_unbound_outcome_does_not_borrow_a_global_call_id_trace(tmp_path, change, reason):
    pair = _tool(invocation="supposed-success", facts={"status": "ok", "is_error": False, "exit_code": 0})
    row = pair[-1]
    if change == "legacy":
        row.pop(TOOL_RESULT_RECORD_KEY)
    elif change == "content":
        row["content"] += "Changed after the producer record."
    elif change == "invocation":
        row[TOOL_RESULT_RECORD_KEY]["invocation"].pop("invocation_id")
    else:
        row[TOOL_RESULT_RECORD_KEY]["invocation"]["tool_call_id"] = "another-call"
    messages = [*_prefix(), *pair]
    request = _request(messages, 1)
    rebuilt, receipt = emergency_address_view(messages, request, rung="unseen_bodies",
        trace_refs_by_tool_call_id={"reused": {"path": "unbound-success-trace.json"}},
        drive_root=tmp_path, task_id="source-view")
    assert receipt.status == "applied"
    text = _visible(_records(rebuilt)[0])
    assert f"outcome binding unknown ({reason})" in text and '"status":"unknown"' in text
    assert '"exit_code":null' in text and '"is_error":null' in text
    assert "supposed-success" not in text and "unbound-success-trace" not in json.dumps(_records(rebuilt))
    _assert_exact_restore(tmp_path, messages, request, receipt)


def test_dialogue_question_quiz_warning_and_opaque_protocol_survive_body_relocation(tmp_path):
    from ouroboros.owner_quiz import quiz_answer_frame

    question = {"role": "assistant", "content": "Which behavior: A keeps the old API; B removes it?"}
    answer = {"role": "user", "content": "A"}
    quiz = {"role": "user", "content": quiz_answer_frame({"quiz_id": "bound-quiz", "asked_at": "2026-01-01",
        "answered_at": "2026-01-02", "question": "Ship now or preserve the old route?",
        "options": ["A: preserve", "B: ship"]}, 0, "Keep the original contract.")}
    warning = {"role": "user", "content": "The previous unknown mutation remains unresolved. " * 500}
    opaque = {"role": "assistant", "content": [{"type": "thinking", "thinking": "Native opaque unit", "signature": "signed"}]}
    active_call = _tool("still-running")[:1]
    fixed = [question, answer, quiz, warning, {"role": "user", "content": OWNER}, opaque]
    messages = [*_prefix(), *fixed, *_tool(), *active_call]
    request = _request(messages, 1)
    rebuilt, receipt = emergency_address_view(messages, request, rung="unseen_bodies", protected_texts=(OWNER, quiz["content"]),
                                             drive_root=tmp_path, task_id="source-view")
    assert receipt.status == "applied"
    assert all(row in rebuilt for row in fixed) and rebuilt[-1] == active_call[0]
    assert question in rebuilt and answer in rebuilt  # plain A remains with the actual question
    _assert_exact_restore(tmp_path, messages, request, receipt)


@pytest.mark.parametrize("rung", ["unseen_bodies", "host_copies"])
def test_short_unit_stays_raw_while_an_independent_large_unit_shrinks(tmp_path, rung):
    divider = {"role": "user", "content": "Keep these units separate."}
    if rung == "unseen_bodies":
        small = _tool("small", text="ok")
        large = _tool("large")
        tail = []
    else:
        small = [{"role": "user", HOST_CONTEXT_KIND_KEY: CONTEXT_FACTS_NAME, "content": "small"}]
        large = [{"role": "user", HOST_CONTEXT_KIND_KEY: ROSTER_SNAPSHOT_NAME, "content": "Large old roster. " * 2000}]
        tail = [{"role": "user", HOST_CONTEXT_KIND_KEY: CONTEXT_FACTS_NAME, "content": "current facts"},
                {"role": "user", HOST_CONTEXT_KIND_KEY: ROSTER_SNAPSHOT_NAME, "content": "current roster"}]
    messages = [*_prefix(), *small, divider, *large, *tail]
    rebuilt, receipt = emergency_address_view(messages, _request(messages, 1), rung=rung,
                                             drive_root=tmp_path, task_id="source-view")
    assert receipt.status == "applied" and receipt.reclaimed_tokens > 0
    assert all(row not in rebuilt for row in large)
    assert all((row in rebuilt) == (rung == "unseen_bodies") for row in small)
    assert all(row in rebuilt for row in tail)


def test_address_recovery_does_not_relax_authored_or_automatic_helper_exposure(tmp_path, monkeypatch):
    monkeypatch.setattr(cc, "_summarizer_spec", lambda: dict(_SPEC))
    monkeypatch.setattr(cc, "_call_summarizer", lambda *_a, **_kw: pytest.fail("Unseen bodies do not earn a helper summary"))
    messages = [*_prefix(), *_tool()]
    request = _request(messages, 1)
    unchanged, ordinary = emergency_address_view(messages, request, rung="bodies", drive_root=tmp_path, task_id="source-view")
    assert unchanged == messages and ordinary.status == "no_eligible"
    relocated, receipt = emergency_address_view(messages, request, rung="unseen_bodies", drive_root=tmp_path, task_id="source-view")
    assert receipt.status == "applied" and relocated != messages
    # A successful source relocation is not evidence that the actor understood it.
    authored = replace(request, working_note="", expected_view_revision=request.transcript_sha256, keep_unit_ids=())
    unchanged, receipt, usage = cc.compact_tool_history_llm(messages, request=authored, observed_messages=messages,
        exposed_units=[], tool_schemas=[], fit_candidate=lambda *_: {"accepted": True},
        drive_root=tmp_path, task_id="source-view")
    assert unchanged == messages and receipt.status == "no_op" and usage is None
    unchanged, receipt, _ = cc.compact_tool_history_llm(messages, request=request, exposed_units=[],
        automatic_deficit_tokens=0, provider_refused=True, drive_root=tmp_path, task_id="source-view")
    assert unchanged == messages and receipt.status == "no_eligible"


def test_explicit_count_only_helper_still_reads_complete_sources(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(cc, "_summarizer_spec", lambda: dict(_SPEC))

    def summarize(parts, **_kwargs):
        calls.extend(part.text for part in parts)
        return {part.source_id: "The helper retained the trailing conclusion." for part in parts}

    monkeypatch.setattr(cc, "_call_summarizer", summarize)
    messages = [*_prefix(), *_tool()]
    rebuilt, receipt, _ = cc.compact_tool_history_llm(messages, keep_recent=0, drive_root=tmp_path, task_id="source-view")
    assert calls and "CONSEQUENTIAL-TAIL" in "".join(calls)
    assert receipt.status == "applied" and rebuilt != messages
    assert cc._capsule_metadata(_records(rebuilt)[0])[1]["retention"] == "summarized"
    assert _checkpoint(tmp_path, receipt)["messages"] == messages


def test_first_main_refusal_recovers_unobserved_body_on_same_model_and_effort(main_loop, monkeypatch):
    import httpx
    from ouroboros import usage_accounting as ua
    from ouroboros.llm_attempt import _attempt_request, _physical_candidate
    from ouroboros import loop_model_call

    f = main_loop
    f.messages.extend([*_tool(), {"role": "user", "content": OWNER}])
    before_observation = []
    rungs = []
    apply_rung = loop_model_call._apply_overflow_rung

    def apply(ctx, rung, fit):
        rungs.append(rung)
        return apply_rung(ctx, rung, fit)

    monkeypatch.setattr(loop_model_call, "_apply_overflow_rung", apply)

    def refuse(kwargs):
        if any(_EMERGENCY_LABELS["unseen_bodies"] in str(row.get("content")) for row in kwargs["messages"]):
            return {"content": "Recovered from the exact address."}
        before_observation.append(getattr(f.ctx, "_last_context_observation", None))
        candidate = _physical_candidate({key: kwargs[key] for key in ("messages", "tools", "model")})
        request = _attempt_request({"provider": "openai", "usage_model": kwargs["model"]}, candidate)
        ua.adopt_physical_attempt_capture(ua.PhysicalAttemptCapture(
            "first-refused", kwargs["model"], "openai", "unresolved", "canonical_json_v1",
            physical_context=request.physical_context, candidate_raw_sha256=request.candidate_raw_sha256,
            candidate_raw_size_bytes=request.candidate_raw_size_bytes, candidate_context_sha256=request.candidate_context_sha256,
            candidate_context_size_bytes=request.candidate_context_size_bytes))
        raise httpx.HTTPStatusError("context_length_exceeded", request=httpx.Request("POST", "https://fixture.invalid"),
            response=httpx.Response(400, json={"error": {"code": "context_length_exceeded", "message": "Input exceeds context window"}}))

    monkeypatch.setattr("ouroboros.loop_llm_call._sleep_within_deadline", lambda *_a, **_kw: True)
    monkeypatch.setattr(cc, "_summarizer_spec", lambda: dict(_SPEC))
    answer, _, _ = f.run([refuse] * 8)
    assert answer == "Recovered from the exact address." and len(f.inputs) >= 2
    assert rungs == ["host_copies", "bodies", "helper", "memory", "low", "unseen_bodies"]
    assert before_observation and all(observation is None for observation in before_observation)
    assert len({entry["model"] for entry in f.inputs}) == 1
    assert len({entry["reasoning_effort"] for entry in f.inputs}) == 1
    first_facts = [row for row in f.inputs[0]["messages"] if row.get(HOST_CONTEXT_KIND_KEY) == CONTEXT_FACTS_NAME]
    retry_facts = [row for row in f.inputs[-1]["messages"] if row.get(HOST_CONTEXT_KIND_KEY) == CONTEXT_FACTS_NAME]
    assert first_facts and retry_facts
    assert first_facts[-1] in retry_facts  # already sent facts remain exact history
    assert retry_facts[-1] != first_facts[-1]  # the rebuilt send gets a fresh measured line
    assert any(row.get("tool_call_id") == "reused" for row in f.inputs[0]["messages"])
    assert not any(row.get("tool_call_id") == "reused" for row in f.inputs[-1]["messages"])
    records = _records(f.ctx.messages)
    record = next(row for row in records if _EMERGENCY_LABELS["unseen_bodies"] in _visible(row))
    assert "exposure is UNCONFIRMED" in _visible(record)
    checkpoint_ref = cc._capsule_metadata(record)[1]["checkpoint_ref"]
    checkpoint = json.loads(read_actor_source_bytes(f.ctx.drive_root, "authored-main", checkpoint_ref))
    assert checkpoint["messages"] == f.inputs[-2]["messages"]
    assert next(row["content"] for row in checkpoint["messages"] if row.get("tool_call_id") == "reused") == BODY
    assert {"role": "user", "content": OWNER} in f.ctx.messages


def test_short_obsolete_host_rows_and_review_indexes_share_one_exact_source(tmp_path):
    from ouroboros.review_history_view import _index_message

    obsolete = [{"role": "user", HOST_CONTEXT_KIND_KEY: CONTEXT_FACTS_NAME,
                 "content": f"round {n}: input 90000, reply 16000, window 128000, quota unknown"} for n in range(199)]
    old_index = _index_message({"closed": "Prior exact review state. " * 60}, "plan")
    latest_index = _index_message({"open": "Current obligation remains."}, "plan")
    foreign = _index_message({"open": "Another repository's current obligation."}, "commit", "/other")
    malformed = deepcopy(latest_index)
    malformed["content"] += "an unbound alteration"
    opaque = {"role": "user", HOST_CONTEXT_KIND_KEY: CONTEXT_FACTS_NAME,
              "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64,eA=="}}]}
    messages = [*_prefix(), old_index]
    for n, row in enumerate(obsolete):
        messages.extend([row, {"role": "assistant", "content": f"Retained narration {n}."}])
    latest = {"role": "user", HOST_CONTEXT_KIND_KEY: CONTEXT_FACTS_NAME, "content": "Current complete facts."}
    messages.extend([latest, latest_index, foreign, malformed, opaque, *_tool(invocation="still-raw")])
    rebuilt, receipt = emergency_address_view(messages, _request(messages, 1), rung="host_copies",
                                              drive_root=tmp_path, task_id="source-view")
    assert receipt.status == "applied" and receipt.reclaimed_tokens > 0
    assert len(_records(rebuilt)) == 1 and "200 obsolete host rows" in _visible(_records(rebuilt)[0])
    assert all(row not in rebuilt for row in [old_index, *obsolete])
    assert all(row in rebuilt for row in [latest, latest_index, foreign, malformed, opaque, *messages[-2:]])
    assert len([row for row in rebuilt if row.get("role") == "assistant"]) == 200
    assert _checkpoint(tmp_path, receipt)["messages"] == messages and receipt.selected_unit_ids == ()


def test_exposure_matches_shifted_cache_wrapped_occurrence_but_not_a_new_equal_body(tmp_path):
    old_facts = {"role": "user", HOST_CONTEXT_KIND_KEY: CONTEXT_FACTS_NAME, "content": "Old facts. " * 600}
    old = _tool(invocation="old-occurrence", facts={"status": "ok", "exit_code": 0})
    observed = [*_prefix(), old_facts, *old]
    observation = _observation(observed)
    latest = {"role": "user", HOST_CONTEXT_KIND_KEY: CONTEXT_FACTS_NAME, "content": "Current facts."}
    fresh = _tool(invocation="new-occurrence", facts={"status": "ok", "exit_code": 0})
    current = [*observed, latest, *fresh]
    compacted, host_receipt = emergency_address_view(current, _request(current, 1), rung="host_copies",
                                                    drive_root=tmp_path, task_id="source-view")
    assert host_receipt.status == "applied"
    # A new retained row shifts the old tool unit; its original positional ID
    # cannot match. Its result also gains the host's cache presentation wrapper.
    compacted.insert(2, {"role": "user", "content": "A new retained notice before the old call."})
    result = next(row for row in compacted if row.get(TOOL_RESULT_RECORD_KEY, {}).get("invocation", {}).get("invocation_id") == "old-occurrence")
    result["content"] = [{"type": "text", "text": result["content"], "cache_control": {"type": "ephemeral"}}]
    before = deepcopy(compacted)
    rebuilt, receipt = emergency_address_view(compacted, _request(compacted, 1), rung="bodies",
        observation=observation, drive_root=tmp_path, task_id="source-view")
    assert receipt.status == "applied" and len(receipt.selected_unit_ids) == 1
    assert compacted == before
    remaining = [row for row in rebuilt if row.get("role") == "tool"]
    assert len(remaining) == 1 and remaining[0][TOOL_RESULT_RECORD_KEY]["invocation"]["invocation_id"] == "new-occurrence"
    text = "\n".join(_visible(row) for row in _records(rebuilt))
    assert "invocation old-occurrence" in text and '"exit_code":0' in text
    assert "exposure proves no understanding" in text
    _assert_exact_restore(tmp_path, compacted, _request(compacted, 1), receipt)


def test_exposure_multiplicity_and_missing_invocation_do_not_credit_new_calls(tmp_path):
    known = _tool(invocation="one-recorded-occurrence", facts={"status": "ok"})
    observed = [*_prefix(), *known]
    current = [*deepcopy(observed), {"role": "user", "content": "another occurrence"}, *deepcopy(known)]
    rebuilt, receipt = emergency_address_view(current, _request(current, 100_000), rung="bodies",
        observation=_observation(observed), drive_root=tmp_path, task_id="source-view")
    assert receipt.status == "applied" and len(receipt.selected_unit_ids) == 1
    assert len([row for row in rebuilt if row.get("role") == "tool"]) == 1
    unbound = [*_prefix(), *_tool()]
    unchanged, ordinary = emergency_address_view(unbound, _request(unbound, 1), rung="bodies",
        observation=_observation(unbound), drive_root=tmp_path, task_id="source-view")
    assert unchanged == unbound and ordinary.status == "no_eligible"
    rescued, late = emergency_address_view(unbound, _request(unbound, 1), rung="unseen_bodies",
        observation=_observation(unbound), drive_root=tmp_path, task_id="source-view")
    assert late.status == "applied" and "exposure is UNCONFIRMED" in _visible(_records(rescued)[0])


def test_body_prefix_stops_at_final_measured_goal_and_checkpoints_once(tmp_path, monkeypatch):
    messages = _prefix()
    for n in range(3):
        messages += _tool(f"call-{n}", invocation=f"occurrence-{n}", text="x" * 50_000, facts={"status": "ok"})
    writes = []
    persist = cc._persist_reclaim_checkpoint

    def save(*args, **kwargs):
        writes.append(True)
        return persist(*args, **kwargs)

    def final_measure(rows):
        # Final request work can add more than the raw capsule. The selector must
        # keep going when an intermediate materializer alone would meet the goal.
        actual = deepcopy(rows)
        if _records(actual):
            actual.append({"role": "user", HOST_CONTEXT_KIND_KEY: CONTEXT_FACTS_NAME, "content": "f" * 30_000})
        return cc._context_tokens_for_messages(actual, 1.0)

    monkeypatch.setattr(cc, "_persist_reclaim_checkpoint", save)
    request = _request(messages, 6000)
    rebuilt, receipt = emergency_address_view(messages, request, rung="bodies", observation=_observation(messages),
        measure_candidate=final_measure, drive_root=tmp_path, task_id="source-view")
    assert receipt.status == "applied" and receipt.goal_reached and len(receipt.selected_unit_ids) == 2
    assert [row["tool_call_id"] for row in rebuilt if row.get("role") == "tool"] == ["call-2"]
    assert receipt.reclaimed_tokens == final_measure(messages) - final_measure(rebuilt)
    assert len(writes) == 1
    _assert_exact_restore(tmp_path, messages, request, receipt)


def test_failed_checkpoint_does_not_publish_any_emergency_view(tmp_path, monkeypatch):
    messages = [*_prefix(), *_tool(invocation="known", facts={"status": "ok"})]
    original = deepcopy(messages)
    monkeypatch.setattr(cc, "_persist_reclaim_checkpoint", lambda *_args, **_kwargs: None)
    rebuilt, receipt = emergency_address_view(messages, _request(messages, 1), rung="bodies",
        observation=_observation(messages), drive_root=tmp_path, task_id="source-view")
    assert rebuilt == messages == original and receipt.status == "checkpoint_failed"


def test_main_emergency_measurement_uses_sealed_final_facts_without_mutating_live_usage(tmp_path, monkeypatch):
    from ouroboros import loop
    from ouroboros.context_fit import seal_task_transcript
    from ouroboros.loop_messages import append_context_facts
    from ouroboros.loop_round_limits import _run_emergency_address_pass
    from tests.test_loop_compaction import _ctx, _fit

    ctx = _ctx(tmp_path)
    ctx.messages += _tool(invocation="previously-seen", facts={"status": "ok"})
    ctx.tools._ctx._last_context_observation = _observation(ctx.messages)
    ctx.accumulated_usage["retained_fact"] = "unchanged"
    measured = []

    def measure(trial, **_kwargs):
        from ouroboros.loop_model_call import _remember_main_fit
        disposition = _fit(estimated_input=cc._context_tokens_for_messages(trial.messages, 1.0), goal=1)
        _remember_main_fit(trial, disposition)
        measured.append((trial is ctx, deepcopy(trial.messages)))
        return disposition

    monkeypatch.setattr(loop, "_measure_round_main_fit", measure)
    monkeypatch.setattr(loop, "_wrapup_global_remaining", lambda: None)
    append_context_facts(ctx)
    before = cc._context_tokens_for_messages(ctx.messages, 1.0)
    usage_before = deepcopy(ctx.accumulated_usage)
    measured.clear()
    receipt = _run_emergency_address_pass(ctx, _fit(goal=1), rung="bodies")
    assert receipt.status == "applied" and ctx.accumulated_usage == usage_before
    assert measured and not any(is_live for is_live, _rows in measured)
    # What dispatch actually prepares includes the same new host facts and seal.
    seal_task_transcript(ctx.messages)
    append_context_facts(ctx)
    after = cc._context_tokens_for_messages(ctx.messages, 1.0)
    assert receipt.reclaimed_tokens == before - after
    assert any(row.get(HOST_CONTEXT_KIND_KEY) == CONTEXT_FACTS_NAME for row in measured[-1][1])
