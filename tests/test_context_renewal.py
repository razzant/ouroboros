"""The refusal path's floor: the same task continues in a fresh view (owner Q1=A, 17.10.2026)."""

from __future__ import annotations

import json
import pathlib
from dataclasses import replace
from types import SimpleNamespace

import pytest

from ouroboros import context_compaction as cc
from ouroboros.artifacts import read_actor_source_bytes
from ouroboros.context_budget import HOST_CONTEXT_KIND_KEY, OWNER_WORDS_KIND, bound_host_row
from ouroboros.context_renewal import RENEWAL_RECORD_NAME, renew_after_refusal, renewal_seed
from ouroboros.loop_messages import CONTEXT_FACTS_NAME
from ouroboros.main_context_authority import PREDECESSOR_NOTES_NAME
from tests._noted_transcript_fixture import DEEP_CALL, FEEDBACK, LEGACY_FACTS, NOTE, OWNER_LATER, _noted
from tests.test_context_attention_mind import _owner, _tool, _transcript
from tests.test_context_reclaim_materializer import _request


def test_the_seed_keeps_people_notes_and_recent_reasoning_and_addresses_the_rest(tmp_path):
    with pytest.raises(RecursionError):  # the fixture witnesses the decoder's depth refusal, not a truncation error
        json.loads(DEEP_CALL[0]["tool_calls"][0]["function"]["arguments"])
    messages = _noted(tmp_path)
    seed, left = renewal_seed(messages)
    texts = [json.dumps(row.get("content"), ensure_ascii=False) for row in seed]
    assert seed[0] == messages[0] and seed[1] == messages[1]  # the system view and the assignment
    assert _owner(OWNER_LATER) in seed and any(NOTE in text for text in texts)  # people's words and my note
    assert any(row.get(HOST_CONTEXT_KIND_KEY) == PREDECESSOR_NOTES_NAME for row in seed)
    assert {"role": "assistant", "content": "My reasoning since the note."} in seed
    assert {"role": "user", "content": "A plain notice delivered before typing."} in seed
    assert FEEDBACK in seed and LEGACY_FACTS in seed  # delivered messages stay; an unbound row stays whole
    assert {"role": "assistant", "content": "Checking the flag parser next."} in seed  # my words beside a call
    assert not any(row.get("role") == "tool" or row.get("tool_calls") for row in seed)  # bodies by address
    facts = [row for row in seed if row.get(HOST_CONTEXT_KIND_KEY) == CONTEXT_FACTS_NAME]
    assert facts == [LEGACY_FACTS]  # bound host state is sent afresh next round
    assert left["tool_rows"] >= 4 and left["host_rows"] >= 1


def test_renewal_publishes_the_seed_with_an_exact_source_and_continues_the_same_task(tmp_path):
    messages = _noted(tmp_path)
    original = json.loads(json.dumps(messages))
    events = []
    tools = SimpleNamespace(_ctx=SimpleNamespace(messages=messages))
    # The fallback walked after the passes failed for its own reason (a dead id, an exhausted pool with
    # waits disallowed): the size refusal this round earned still renews the view (review r1, 17.10).
    limit_ctx = SimpleNamespace(messages=messages, accumulated_usage={
        "_last_llm_error_kind": "bad_request", "resource_refusal": {"reason": "quota"},
        "_context_refusal_round": 12}, drive_root=tmp_path, drive_logs=tmp_path / "logs", task_id="renewal",
        round_idx=12, event_queue=None, tool_schemas=[])
    import ouroboros.loop as loop
    loop_emit = loop._emit_checkpoint_event
    loop._emit_checkpoint_event = lambda *_a, **_kw: events.append(_a[-1])
    try:
        assert renew_after_refusal(limit_ctx, tools) is True
    finally:
        loop._emit_checkpoint_event = loop_emit
    record = messages[-1]
    assert record[HOST_CONTEXT_KIND_KEY] == RENEWAL_RECORD_NAME and "nothing summarized" in record["content"]
    # Review r1 (17.10): every call since my newest note is indexed with its facts, shown or never shown.
    index = record["content"].split("Tool units since my newest note")[1]
    assert "three.py" in index and "cli.py" in index and "never shown to me" in index
    assert "one.py" not in index  # covered by the note, not since it
    assert tools._ctx.messages is messages and len(messages) < len(original)  # published in place, same task
    source = json.loads(record["content"].rsplit("rows: ", 1)[1].split("\n", 1)[0].rstrip("."))
    assert json.loads(read_actor_source_bytes(pathlib.Path(tmp_path), "renewal", source)) == original
    assert events and events[0]["checkpoint_kind"] == "context_view_renewed" and events[0]["owner_visible"]
    assert "context-size recovery" in messages[-1]["content"]
    assert "provider refused" not in messages[-1]["content"]


def test_the_passes_address_records_since_the_newest_note_stay_in_the_seed(tmp_path):
    """Review r1 (17.10): a refusal pass's records of what left the view since my newest note stay;
    an older address record (before the note) leaves with the rest."""
    from ouroboros.context_budget import ContextReclaimRequest
    from ouroboros.context_source_view import emergency_address_view

    messages = _noted(tmp_path)
    request = ContextReclaimRequest(route_fp="r", round_id="r:1", transcript_sha256=cc.context_reclaim_transcript_sha256(
        messages), measurement_basis="cold_estimate", measurement_density=1.0, reclaim_goal_tokens=2**31 - 1)
    addressed, receipt = emergency_address_view(messages, request, rung="unseen_bodies", observation={},
                                                drive_root=tmp_path, task_id="renewal")
    assert receipt.status == "applied"
    newest = max(i for i, row in enumerate(addressed) if cc._capsule_metadata(row)[0]
                 and (cc._capsule_metadata(row)[1] or {}).get("authorship") == "actor")
    records = [row for i, row in enumerate(addressed) if i > newest and cc._capsule_metadata(row)[0]]
    assert records  # the pass replaced units after the note by address records
    seed, _left = renewal_seed(addressed)
    assert all(record in seed for record in records)


def test_renewal_is_not_attempted_when_it_cannot_help_or_the_failure_is_not_size(tmp_path):
    def ctx(messages, kind="context_overflow", refused_round=3):
        return SimpleNamespace(messages=messages, accumulated_usage={
            "_last_llm_error_kind": kind, "_context_refusal_round": refused_round}, drive_root=tmp_path,
            drive_logs=tmp_path / "logs", task_id="renewal", round_idx=3, event_queue=None, tool_schemas=[])

    tools = SimpleNamespace(_ctx=SimpleNamespace())
    # No size refusal in this round (an earlier round's stamp, or none): another failure renews nothing.
    assert renew_after_refusal(ctx(_noted(tmp_path), kind="provider_unavailable", refused_round=2), tools) is False
    assert renew_after_refusal(ctx(_noted(tmp_path), refused_round=None), tools) is False
    only_host = [{"role": "system", "content": "S"}, {"role": "user", "content": "go"},
                 bound_host_row(CONTEXT_FACTS_NAME, "[CONTEXT_FACTS] round 1"),
                 {"role": "user", HOST_CONTEXT_KIND_KEY: OWNER_WORDS_KIND, "content": "A"}]
    before = list(only_host)
    assert renew_after_refusal(ctx(only_host), tools) is False and only_host == before  # nothing substantive leaves


def test_a_source_the_mind_restored_stays_whole_across_renewal(tmp_path):
    """Chosen sources stay, before or after a note; unasked work can still leave."""
    messages = _transcript()
    request = replace(_request(messages, 0), working_note=NOTE,
                      expected_view_revision=cc.context_reclaim_transcript_sha256(messages), keep_unit_ids=())
    folded, receipt = cc.compact_tool_history_llm(messages, request=request, observed_messages=messages,
        tool_schemas=[], drive_root=tmp_path, task_id="renewal", fit_candidate=lambda _m, _t: {"accepted": True})
    ref = next(dict(item) for item in receipt.source_refs if "unit_id" in item)
    restored, _refs = cc._restored_source_views(
        [ref], drive_root=tmp_path, task_id="renewal", request=replace(request, route_fp="actor"))
    note_at = max(i for i, row in enumerate(folded) if cc._capsule_metadata(row)[0]
                  and (cc._capsule_metadata(row)[1] or {}).get("authorship") == "actor")
    for at in (note_at, len(folded)):
        rows = [*folded[:at], *restored, *folded[at:], _owner(OWNER_LATER)]
        for extra in ([], _tool("unasked", "Disposable body. " * 1000)):
            seed, left = renewal_seed([*rows, *extra])
            assert left["restored_sources"] == 0 and restored[0] in seed and _owner(OWNER_LATER) in seed
            assert any(NOTE in json.dumps(row.get("content"), ensure_ascii=False) for row in seed)
            assert (left["tool_rows"] > 0) is bool(extra)
            assert not any(row.get("tool_call_id") == "unasked" for row in seed)
            from ouroboros.context_renewal import restored_sources
            assert restored_sources(seed) == [{key: ref[key] for key in ("checkpoint_ref", "unit_id", "raw_sha256")}]
            from ouroboros.context_renewal import physical_window_cut
            from ouroboros.context_fit import estimate_context_prompt_tokens
            expanded = [*rows[:2], {"role":"assistant", "content":"old whole reply "*10000}, *rows[2:]]
            cut, record = physical_window_cut(expanded, drive_root=tmp_path, task_id="renewal",
                measure=estimate_context_prompt_tokens, usable=estimate_context_prompt_tokens(rows)+1500)
            assert record and restored[0] in cut



def test_a_second_renewal_keeps_the_first_renewals_index_until_a_newer_note(tmp_path):
    """Review r2 (18.10): a second renewal without a newer note dropped the first record, and with it the
    facts of the calls it had indexed; they stay until the mind writes a newer note."""
    messages = _noted(tmp_path)
    first = bound_host_row(RENEWAL_RECORD_NAME, "[Host record: working view renewed] ... three.py ...")
    seed, _left = renewal_seed([*messages, first, *_tool("five", "more " * 500)])
    assert first in seed


@pytest.mark.parametrize(("provider", "effort"), [("openai", "high"), ("openai", "none"), ("openrouter", "high")])
def test_renewal_and_presend_measure_the_same_provider_effort_shape(tmp_path, monkeypatch, provider, effort):
    from ouroboros import capability_evidence as ce, context_fit as cf, loop
    from ouroboros.context_renewal import renewal_view
    from ouroboros.context_window import WindowLimit, WindowRecord
    from ouroboros.loop_model_call import _measure_round_main_fit
    from ouroboros.loop_messages import _context_components

    monkeypatch.setattr(ce, "canonical_evidence_root", lambda: tmp_path)
    monkeypatch.setattr(cf, "_route_calibration_ratio", lambda *_a, **_kw: 1.0)
    schemas = [{"type": "function", "function": {"name": f"tool_{i}", "description": "Read a fixture path.",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}}
        for i in range(60)]
    core = cf.ContextCore("Core", "BIBLE", "", "", "Identity", "Facts", '"Assignment"', False)
    plan = cf.build_context_fit_plan(None, core, {"id": "renewal"}, preferred_mode="max",
        tool_schemas=schemas, reasoning_effort=effort, route_resolver=lambda *_a, **_kw: (
            {"provider": provider, "model": provider + "::fixture"},
            SimpleNamespace(route_fp="fixture-route", status="confirmed", stale=False, window_tokens=1_000_000)))
    messages = [*plan.messages_for("max"), {"role": "assistant", "content": "Earlier progress. " * 4000},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "read1", "type": "function",
            "function": {"name": "tool_1", "arguments": '{"path":"fixture.txt"}'}}]},
        {"role": "tool", "tool_call_id": "read1", "content": "unasked fixture result " * 6000},
        {"role": "user", HOST_CONTEXT_KIND_KEY: OWNER_WORDS_KIND, "content": "Keep going with the current plan."}]
    original = json.loads(json.dumps(messages))
    canonical = lambda rows: cf.estimate_context_prompt_tokens(rows, schemas)
    seed, record = renewal_view(messages, observation=None, drive_root=tmp_path, task_id="renewal", measure=canonical)
    assert record and len(seed) < len(messages)
    # A fixed input window holds the canonical seed, but not OpenAI reasoning's longer custom catalog.
    usable = canonical(seed) + 100
    input_limit = WindowLimit(usable, "input", "metadata")
    plan = replace(plan, window_tokens=usable, window_record=WindowRecord(
        plan.route_fp, input=input_limit, declared=(input_limit,)))
    tools = SimpleNamespace(_ctx=SimpleNamespace(messages=messages, context_fit_plan=plan))
    ctx = SimpleNamespace(messages=messages, accumulated_usage={}, drive_root=tmp_path,
        drive_logs=tmp_path / "logs", task_id="renewal", round_idx=12, event_queue=None, tool_schemas=schemas,
        context_fit_plan=plan, active_model=plan.model, active_context_mode="max", active_effort=effort, tools=tools)
    before = _measure_round_main_fit(ctx, automatic_pass_used=True).measurement
    assert before.capacity_deficit_tokens > 0
    ctx.accumulated_usage["_context_refusal_round"] = ctx.round_idx
    events = []
    monkeypatch.setattr(loop, "_emit_checkpoint_event", lambda *_a, **_kw: events.append(_a[-1]))
    assert renew_after_refusal(ctx, tools)
    after = _measure_round_main_fit(ctx, automatic_pass_used=True).measurement
    cut = any(row.get(HOST_CONTEXT_KIND_KEY) == "context_physical_cut" for row in messages)
    schema_fact = dict(_context_components(ctx, measured=True))["tool schemas"]
    schema_shape = cf.tool_schema_tokens(schemas, provider=provider, reasoning_effort=effort)
    print("RENEWAL_PRESEND_SHAPE", {"provider": provider, "effort": effort,
        "canonical_seed": canonical(seed), "usable": usable, "actual_after": after.estimated_input_tokens,
        "deficit_after": after.capacity_deficit_tokens, "cut": cut,
        "schema_fact": schema_fact, "schema_shape": schema_shape})
    assert after.capacity_total_tokens == before.capacity_total_tokens
    assert after.response_reserve_tokens == before.response_reserve_tokens
    assert after.measurement_density == before.measurement_density == 1.0
    assert after.capacity_deficit_tokens == 0
    assert schema_fact == schema_shape  # the diagnostic names only the catalog segment, in the same shape
    assert cut is (provider == "openai" and effort == "high")
    assert [r["content"] for r in messages[:2]] == [r["content"] for r in original[:2]]
    assert original[-1]["content"] in [r.get("content") for r in messages]
    if not cut:
        assert [(r["role"], r.get("content")) for r in messages] == [
            (r["role"], r.get("content")) for r in seed]  # publication also seals the last row
    assert json.loads(read_actor_source_bytes(tmp_path, "renewal", events[0]["source"])) == original
