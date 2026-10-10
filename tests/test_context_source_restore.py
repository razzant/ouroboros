"""Restore-only requests keep the current view and append exact labelled sources."""

from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from ouroboros import context_compaction as cc, loop_round_limits as limits
from ouroboros.artifacts import read_actor_source_bytes
from ouroboros.context_budget import HOST_CONTEXT_KIND_KEY
from ouroboros.tools.compact_context import _compact_context, record_context_view
from tests.test_actor_context_view import _apply, _unit
from tests.test_context_view_tool import _context, _source
from tests.test_main_authored_context import call, main_loop  # noqa: F401


@pytest.fixture
def saved_source(tmp_path):
    source = _source()
    folded, receipt, usage = _apply(source, deepcopy(source), tmp_path)
    assert receipt.status == "applied" and usage is None
    ref = next(ref for ref in receipt.source_refs if "unit_id" in ref)
    raw = read_actor_source_bytes(tmp_path, "actor-view", ref["checkpoint_ref"])
    assert json.loads(raw)["messages"] == source
    return folded, ref, raw


def _run(tmp_path, monkeypatch, ctx, messages, schemas, fit=None):
    frame = _context(tmp_path, ctx, schemas, monkeypatch,
                     fit or (lambda *_: {"accepted": True}))
    frame.task_id = "actor-view"
    return limits._run_round_compaction(messages, frame)


def _sources(messages):
    return [row for row in messages
            if (cc._capsule_metadata(row)[1] or {}).get("retention") == "source_view"]


@pytest.mark.parametrize("blank_note", [False, True])
def test_restore_only_busy_view_appends_without_helper_note_or_prefix_rewrite(
        saved_source, tmp_path, monkeypatch, blank_note):
    folded, ref, raw = saved_source
    ctx = SimpleNamespace(active_context_mode="max", model_turn_state=object())
    native = ctx.model_turn_state
    schemas = [{"type": "function", "function": {"name": "read_file"}}]
    messages = deepcopy(folded)
    for n in range(9):
        messages.extend(_unit(f"busy-{n}", f"Current exact source {n}. " * 600))
    record_context_view(ctx, messages, schemas)
    observed = json.loads(_compact_context(ctx, inspect=True))
    messages.extend([{"role": "user", "content": "A newer exact owner correction."},
                     {"role": "assistant", HOST_CONTEXT_KIND_KEY: "owner_dialogue",
                      "content": "I told the owner what remains uncertain."}])
    record_context_view(ctx, messages, schemas)
    args = {"restore_unit_refs": [ref], "expected_view_revision": observed["view_revision"]}
    if blank_note:
        args["working_note"] = ""
    response = _compact_context(ctx, **args)
    assert "requested" in response and isinstance(ctx._pending_compaction, dict)
    messages.extend([call("compact_context", args, "restore"),
                     {"role": "tool", "tool_call_id": "restore", "content": response}])
    before = deepcopy(messages)
    for target, names in ((cc, ("_call_summarizer", "_summarizer_spec", "_persist_reclaim_checkpoint")),
                          (limits, ("invalidate_task_cache_splits", "prune_reclaim_trace_refs", "sanction_rewrite"))):
        for name in names:
            monkeypatch.setattr(target, name, lambda *a, **k: pytest.fail("restore must only append sources"))
    fits = []

    def fit(rows, tools):
        fits.append(deepcopy(rows))
        assert tools == schemas
        return {"accepted": True}

    after, usage = _run(tmp_path, monkeypatch, ctx, messages, schemas, fit)
    receipt = ctx._context_view_receipt
    assert usage is None and receipt["status"] == "applied"
    assert after[:len(before)] == before and messages == before
    assert ctx.model_turn_state is native
    assert receipt["selection_fingerprint"] == "" and receipt["checkpoint_ref"] is None
    assert receipt["restored_unit_refs"] == (ref,)
    [restored] = _sources(after)
    assert after[len(before)] == restored
    assert after[len(before) + 1]["content"].startswith("[Context view receipt]")
    original = next(u for u in cc.context_units(json.loads(raw)["messages"], scope="dialogue")
                    if u.unit_id == ref["unit_id"])
    assert restored["content"][0]["text"].split("\n", 2)[2] == original.source_text
    assert restored["role"] == "user" and not restored.get("tool_calls")
    assert len(fits) == 2 and fits[-1] == after[:len(before) + 2]
    assert [m for m in after if (cc._capsule_metadata(m)[1] or {}).get("authorship") == "actor"] == folded[2:]
    assert read_actor_source_bytes(tmp_path, "actor-view", ref["checkpoint_ref"]) == raw
    # Re-reading a source already resident is the existing no-op, not a duplicate.
    record_context_view(ctx, after, schemas)
    _compact_context(ctx, restore_unit_refs=[ref])
    again, _ = _run(tmp_path, monkeypatch, ctx, after, schemas)
    assert again is after and ctx._context_view_receipt["status"] == "no_op"


@pytest.mark.parametrize("failure", ["source", "final_fit"])
def test_restore_only_keeps_current_on_source_or_whole_candidate_refusal(
        saved_source, tmp_path, monkeypatch, failure):
    folded, ref, _ = saved_source
    ctx = SimpleNamespace(active_context_mode="max")
    messages = deepcopy(folded)
    ref = deepcopy(ref)
    if failure == "source":
        ref["raw_sha256"] = "f" * 64
    record_context_view(ctx, messages, [])
    _compact_context(ctx, restore_unit_refs=[ref])
    fits = []

    def fit(rows, _tools):
        fits.append(deepcopy(rows))
        return {"accepted": failure == "source" or not str(rows[-1].get("content")).startswith("[Context view receipt]")}

    after, usage = _run(tmp_path, monkeypatch, ctx, messages, [], fit)
    assert after[:len(messages)] == messages and usage is None and not _sources(after)
    if failure == "final_fit":
        assert after is messages and len(fits) == 2 and _sources(fits[0]) and _sources(fits[-1])
    assert ctx._context_view_receipt["status"] == ("source_unavailable" if failure == "source" else "fit_rejected")


@pytest.mark.parametrize("extra", [{"keep_last_n": 6}, {"keep_unit_ids": []},
    {"schema_names": []}, {"review_notes": [{}]}, {"review_transfers": [{}]}])
def test_bare_restore_rejects_ambiguous_selectors_without_queuing(saved_source, extra):
    folded, ref, _ = saved_source
    ctx = SimpleNamespace(active_context_mode="max", _active_builtin_tool_result=None)
    record_context_view(ctx, folded, [])
    response = _compact_context(ctx, restore_unit_refs=[ref], **extra)
    assert "working_note" in response
    assert ctx._active_builtin_tool_result.code == "TOOL_ARG_ERROR"
    assert ctx._active_builtin_tool_result.status == "error"
    assert getattr(ctx, "_pending_compaction", None) is None


def test_count_only_still_uses_helper_for_older_completed_units(tmp_path, monkeypatch):
    from tests.test_context_reclaim_materializer import _SPEC

    messages = _source()[:2]
    for n in range(9):
        messages.extend(_unit(f"old-{n}", f"Exact old source {n}. " * 600))
    ctx, calls = SimpleNamespace(active_context_mode="max"), []
    record_context_view(ctx, messages, [])
    ctx._last_context_observation["exposed_units"] = [
        {"unit_id": u.unit_id, "raw_sha256": u.raw_sha256} for u in cc.context_units(messages)]
    monkeypatch.setattr(cc, "_summarizer_spec", lambda: dict(_SPEC))

    def helper(parts, **kwargs):
        calls.extend(parts)
        return {part.source_id: "Helper fixture summary." for part in parts}

    monkeypatch.setattr(cc, "_call_summarizer", helper)
    assert "scheduled" in _compact_context(ctx, keep_last_n=6)
    assert ctx._pending_compaction == 6
    after, _ = _run(tmp_path, monkeypatch, ctx, messages, [])
    assert calls and "Helper fixture summary." in str(after)
    assert [m for m in after if m.get("role") == "tool"] == [m for m in messages[-12:] if m.get("role") == "tool"]
    assert not _sources(after)


@pytest.mark.parametrize("blank_note", [False, True])
def test_main_next_request_gets_exact_source_without_a_new_account(main_loop, monkeypatch, blank_note):  # noqa: F811
    f, prior, invalidations = main_loop, {}, []
    monkeypatch.setattr(limits, "invalidate_task_cache_splits", lambda task: invalidations.append(task))

    def restore(kwargs):
        prior["receipt"] = deepcopy(f.ctx._context_view_receipt)
        prior["prefix"] = deepcopy(kwargs["messages"])
        checkpoint = json.loads(read_actor_source_bytes(f.ctx.drive_root, "authored-main", prior["receipt"]["checkpoint_ref"]))
        unit = next(u for u in cc.context_units(checkpoint["messages"], scope="dialogue")
                    if any(row.get("tool_call_id") == "read" for row in checkpoint["messages"][u.start:u.end + 1]))
        prior["source_text"] = unit.source_text
        ref = next(ref for ref in prior["receipt"]["source_refs"] if ref.get("unit_id") == unit.unit_id)
        return call("compact_context", {"restore_unit_refs": [ref],
                    **({"working_note": ""} if blank_note else {})}, "restore")

    answer, _, _ = f.run([call("read_file", {"path": "evidence.txt"}, "read"),
        call("compact_context", {"working_note": "I retained the conclusion.", "keep_unit_ids": []}, "fold"),
        restore, {"content": "done"}])
    assert answer == "done" and prior["receipt"]["status"] == "applied"
    after = f.inputs[-1]["messages"]
    assert after[:len(prior["prefix"])] == prior["prefix"]
    assert f.inputs[-1]["tools"] == f.inputs[-2]["tools"]
    assert invalidations == ["authored-main"]  # the deliberate earlier fold only
    assert f.ctx._context_view_receipt["status"] == "applied"
    assert not f.ctx._context_view_receipt["selection_fingerprint"]
    [restored] = _sources(after)
    source_text = restored["content"][0]["text"].split("\n", 2)[2]
    assert source_text == prior["source_text"] and restored["role"] == "user"
    assert f.source in next(row["content"] for row in json.loads(source_text) if row.get("tool_call_id") == "read")
    assert not any(m.get("tool_call_id") == "read" for m in after)
    assert any(m.get("tool_call_id") == "restore" for m in after)
    assert len([m for m in after if (cc._capsule_metadata(m)[1] or {}).get("authorship") == "actor"]) == 1
    # The actual next Main input stays an extension after each supported OpenAI
    # family serializer too. These builders never contact a provider.
    from ouroboros.llm import LLMClient
    from tests.test_host_context_wire import _wire
    from tests.test_openai_system_prefix_split import _build, _target

    monkeypatch.setattr(LLMClient, "_SUPPORTED_PARAMS_FETCHED", True, raising=False)
    monkeypatch.setattr("ouroboros.pricing._fetch_live_rows", lambda *_a, **_kw: {})
    client, schemas = LLMClient(api_key="unused"), f.inputs[-1]["tools"]
    for model, env in (("openai::gpt-6-sol", {"OPENAI_API_KEY": "unused"}),
                       ("openai/gpt-6-sol", {"OPENROUTER_API_KEY": "unused"})):
        first = _build(client, _target(monkeypatch, client, model, env), prior["prefix"], schemas)
        last = _build(client, _target(monkeypatch, client, model, env), after, schemas)
        assert last["messages"][:len(first["messages"])] == first["messages"]
        assert last["tools"] == first["tools"]
    native = {"type": "codex.turn.v1", "opaque": "caller-owned continuation"}
    first = _wire(monkeypatch, prior["prefix"], schemas, native=native)
    last = _wire(monkeypatch, after, schemas, native=native)
    assert last["messages"][:len(first["messages"])] == first["messages"]
    assert last["tools"] == first["tools"]
    assert last["nativeContinuation"] == first["nativeContinuation"] == native
