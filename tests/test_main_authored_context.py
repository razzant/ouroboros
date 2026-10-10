"""Authored views use the real Main loop, tool boundary and source materializer."""

from copy import deepcopy
from dataclasses import replace
import json
import queue
from types import SimpleNamespace

import pytest

from ouroboros import context_compaction, loop, usage_accounting as ua
from ouroboros.artifacts import read_actor_source_bytes
from ouroboros.tools.registry import ToolRegistry
from tests.test_context_fit_integration import _plan


def call(name, arguments, identifier):
    return {"role": "assistant", "content": "", "tool_calls": [{"id": identifier, "type": "function",
        "function": {"name": name, "arguments": json.dumps(arguments)}}]}


@pytest.fixture
def main_loop(tmp_path, monkeypatch):
    monkeypatch.setenv("OUROBOROS_CONTEXT_MODE", "max")
    monkeypatch.setenv("OUROBOROS_TASK_REVIEW_MODE", "off")
    monkeypatch.setenv("OUROBOROS_SAFETY_MODE", "off")
    monkeypatch.setenv("OUROBOROS_MAX_ROUNDS", "12")
    monkeypatch.setenv("MCP_ENABLED", "false")
    monkeypatch.setattr(loop, "_maybe_inject_finalization_nudges", lambda *_args: False)
    monkeypatch.setattr(context_compaction, "_call_summarizer",
                        lambda *a, **kw: pytest.fail("Authored views must not call a helper model"))
    registry = ToolRegistry(repo_dir=tmp_path / "repo", drive_root=tmp_path / "data")
    registry._ctx.repo_dir.mkdir()
    registry._ctx.context_fit_plan = _plan(preferred="max", window=900_000)
    source = "Exact original evidence with its trailing conclusion.\n" * 160
    (registry._ctx.repo_dir / "evidence.txt").write_text(source, encoding="utf-8")
    f = SimpleNamespace(registry=registry, ctx=registry._ctx, source=source,
                        inputs=[], progress=[], incoming=queue.Queue(), events=queue.Queue())
    f.messages = f.ctx.context_fit_plan.messages_for("max")

    def run(responses):
        replies = iter(responses)

        class Provider:
            def default_model(self):
                return f.ctx.context_fit_plan.model

            def chat(self, **kwargs):
                from ouroboros.llm_attempt import _physical_candidate, _attempt_request
                from ouroboros.model_send_seal import persist_physical_candidate

                ua.adopt_physical_attempt_capture(None)
                f.inputs.append(deepcopy(kwargs))
                reply = next(replies)
                if isinstance(reply, Exception):
                    raise reply
                message = reply(kwargs) if callable(reply) else deepcopy(reply)
                candidate = _physical_candidate({key: kwargs[key] for key in ("messages", "tools", "model")})
                request = _attempt_request({"provider": "openai", "usage_model": kwargs["model"]}, candidate)
                attempt_id = f"authored-send-{len(f.inputs)}"
                persisted = persist_physical_candidate(f.ctx.drive_root, task_id="authored-main",
                    attempt_id=attempt_id, candidate=candidate, candidate_facts={})
                ua.adopt_physical_attempt_capture(ua.PhysicalAttemptCapture(
                    attempt_id, kwargs["model"], "openai", "settled", "canonical_json_v1",
                    candidate_manifest_ref=persisted["manifest_ref"], physical_context=request.physical_context,
                    candidate_raw_sha256=request.candidate_raw_sha256, candidate_raw_size_bytes=request.candidate_raw_size_bytes,
                    candidate_context_sha256=request.candidate_context_sha256,
                    candidate_context_size_bytes=request.candidate_context_size_bytes))
                return message, {"prompt_tokens": 100, "completion_tokens": 10,
                                 "cost": 0.0, "provider": "openai"}

        return loop.run_llm_loop(
            messages=f.messages, tools=f.registry, llm=Provider(),
            drive_logs=f.ctx.drive_root / "logs", drive_root=f.ctx.drive_root,
            emit_progress=lambda text, **kw: f.progress.append(text), incoming_messages=f.incoming,
            task_id="authored-main", event_queue=f.events)

    f.run = run
    return f


def test_main_records_and_applies_authored_view_without_manual_observation(main_loop):
    f = main_loop
    answer, _, _ = f.run([
        call("read_file", {"path": "evidence.txt"}, "read"),
        call("compact_context", {"working_note": "I retained the trailing conclusion.", "keep_unit_ids": []}, "compact"),
        {"content": "done"},
    ])
    assert answer == "done"
    receipt = f.ctx._context_view_receipt
    assert receipt["status"] == "applied" and receipt["reclaimed_tokens"] > 0
    checkpoint = json.loads(read_actor_source_bytes(f.ctx.drive_root, "authored-main", receipt["checkpoint_ref"]))
    before, after = f.inputs[1]["messages"], f.inputs[2]["messages"]
    assert checkpoint["messages"] == before
    assert f.source.splitlines()[-1] in str(checkpoint["messages"])
    assert before[:2] == after[:2]
    assert not any(m.get("tool_call_id") == "read" for m in after)
    assert any(m.get("tool_call_id") == "compact" for m in after)
    assert "I retained the trailing conclusion." in str(after)
    assert "[Context view receipt]" in str(after)


def test_explicit_recent_count_compacts_older_units_with_authored_note(main_loop):
    f = main_loop
    f.run([
        *(call("read_file", {"path": "evidence.txt"}, f"read-{i}") for i in range(4)),
        call("compact_context", {"working_note": "All four reads agree.", "keep_last_n": 2}, "compact"),
        {"content": "done"},
    ])
    after = f.inputs[-1]["messages"]
    assert f.ctx._context_view_receipt["status"] == "applied"
    assert [m["tool_call_id"] for m in after if m.get("role") == "tool"] == ["read-2", "read-3", "compact"]
    checkpoint = json.loads(read_actor_source_bytes(f.ctx.drive_root, "authored-main",
                                                  f.ctx._context_view_receipt["checkpoint_ref"]))
    assert [m["tool_call_id"] for m in checkpoint["messages"] if m.get("role") == "tool"] == [f"read-{i}" for i in range(4)]


def test_inspected_ids_override_count_and_new_owner_tail_survives(main_loop):
    f = main_loop
    pinned = {}

    def choose(kwargs):
        inspected = next(m for m in kwargs["messages"] if m.get("tool_call_id") == "inspect")
        pinned.update(json.loads(inspected["content"]))
        f.incoming.put({"text": "Keep the new exact owner correction.", "msg_id": "correction"})
        return call("compact_context", {"working_note": "Retained finding.", "keep_unit_ids": [],
                    "keep_last_n": 20, "expected_view_revision": pinned["view_revision"]}, "compact")

    f.run([call("read_file", {"path": "evidence.txt"}, "read"),
           call("compact_context", {"inspect": True}, "inspect"), choose, {"content": "done"}])
    receipt, after = f.ctx._context_view_receipt, f.inputs[-1]["messages"]
    assert receipt["status"] == "applied" and receipt["observed_view_revision"] == pinned["view_revision"]
    assert not any(m.get("tool_call_id") == "read" for m in after)
    assert all(any(m.get("tool_call_id") == name for m in after) for name in ("inspect", "compact"))
    assert any(m.get("role") == "user" and "Keep the new exact owner correction." in str(m.get("content")) for m in after)


def test_note_only_keeps_raw_and_repeated_note_is_cache_noop(main_loop, monkeypatch):
    from ouroboros import loop_round_limits

    f, invalidations, receipts = main_loop, [], []
    monkeypatch.setattr(loop_round_limits, "invalidate_task_cache_splits", lambda task: invalidations.append(task))
    note = {"working_note": "Retain these raw sources and this conclusion."}

    def repeat(kwargs):
        receipts.append(deepcopy(f.ctx._context_view_receipt))
        return call("compact_context", note, "again")

    f.run([call("read_file", {"path": "evidence.txt"}, "read"),
           call("compact_context", note, "first"), repeat, {"content": "done"}])
    assert receipts[0]["status"] == "applied" and f.ctx._context_view_receipt["status"] == "no_op"
    assert invalidations == ["authored-main"]
    assert f.ctx._context_view_receipt["checkpoint_ref"] is None
    assert any(m.get("tool_call_id") == "read" for m in f.inputs[-1]["messages"])
    assert f.inputs[-1]["tools"] == f.inputs[0]["tools"]


def test_projected_image_stays_in_canonical_authored_checkpoint(main_loop, monkeypatch):
    from ouroboros import vision_routing

    f = main_loop
    image = {"type": "image_url", "image_url": {"url": "data:image/png;base64,eA=="}}
    f.messages.append({"role": "user", "content": [deepcopy(image)]})
    monkeypatch.setattr(vision_routing, "get_image_input_mode", lambda: "off")
    f.run([call("read_file", {"path": "evidence.txt"}, "read"),
           call("compact_context", {"working_note": "Read the evidence.", "keep_unit_ids": []}, "compact"),
           {"content": "done"}])
    receipt = f.ctx._context_view_receipt
    assert receipt["status"] == "applied"
    checkpoint = json.loads(read_actor_source_bytes(f.ctx.drive_root, "authored-main", receipt["checkpoint_ref"]))
    assert {"role": "user", "content": [image]} in checkpoint["messages"]
    assert {"role": "user", "content": [image]} in f.ctx.messages
    assert "image omitted" in str(f.inputs[-1]["messages"])
    assert "image_url" not in str(f.inputs[-1]["messages"])


@pytest.mark.parametrize("first", ["empty", "malformed", "rejected"])
def test_failed_or_empty_response_cannot_supply_a_false_authored_receipt(main_loop, monkeypatch, first):
    from ouroboros import loop_llm_call
    from ouroboros.tools import compact_context

    f, observed, prior = main_loop, [], {}
    record = compact_context.record_context_view

    def observe(ctx, messages, schemas, **kwargs):
        observed.append(deepcopy(messages))
        record(ctx, messages, schemas, **kwargs)

    monkeypatch.setattr(compact_context, "record_context_view", observe)
    monkeypatch.setattr(loop_llm_call, "_sleep_within_deadline", lambda *a, **kw: True)
    bad = {"content": ""} if first == "empty" else call("compact_context", {
        "working_note": "bad", "expected_view_revision": "unrelated"}, "bad")
    if first == "malformed":
        bad["tool_calls"][0]["function"]["arguments"] = "{"

    def resume(kwargs):
        prior["receipt"] = getattr(f.ctx, "_context_view_receipt", None)
        prior["pending"] = getattr(f.ctx, "_pending_compaction", None)
        if first != "empty":
            prior["result"] = next((m["content"] for m in kwargs["messages"] if m.get("tool_call_id") == "bad"), None)
        return call("compact_context", {"working_note": "Verified source retained.", "keep_unit_ids": []}, "good")

    f.run([call("read_file", {"path": "evidence.txt"}, "read"), bad, resume, {"content": "done"}])
    assert len(observed) == (3 if first == "empty" else 4)
    assert f.ctx._context_view_receipt["status"] == "applied"
    assert prior["receipt"] is None and prior["pending"] is None
    if first != "empty":
        expected = "mismatch" if first == "rejected" else "TOOL_ARG_ERROR"
        assert expected in str(prior["result"])


def test_real_source_drift_is_a_visible_materializer_refusal(main_loop):
    f = main_loop

    def stale(kwargs):
        seen = json.loads(next(m["content"] for m in kwargs["messages"] if m.get("tool_call_id") == "inspect"))
        next(m for m in f.ctx.messages if m.get("tool_call_id") == "read")["content"] = "Changed source"
        return call("compact_context", {"expected_view_revision": seen["view_revision"],
                    "working_note": "Old conclusion", "keep_unit_ids": []}, "stale")

    f.run([call("read_file", {"path": "evidence.txt"}, "read"),
           call("compact_context", {"inspect": True}, "inspect"), stale, {"content": "done"}])
    assert f.ctx._context_view_receipt["status"] == "binding_mismatch"
    assert f.ctx._context_view_receipt["checkpoint_ref"] is None
    after = f.inputs[-1]["messages"]
    assert any(m.get("tool_call_id") == "read" and m["content"] == "Changed source" for m in after)
    assert "binding_mismatch" in str(after) and "[Context view receipt]" in str(after)


def test_main_fit_reports_known_pressure_without_rejecting_useful_shrink(tmp_path, monkeypatch):
    from ouroboros import capability_evidence
    from ouroboros.context_budget import ContextReclaimRequest
    from ouroboros.loop_model_call import _measure_main_context_view

    monkeypatch.setattr(capability_evidence, "resolve_main_token_density", lambda *a, **kw: (1.0, "cold_estimate"))
    plan = _plan(preferred="max", window=100)
    observed = [*plan.messages_for("max"), {"role": "user", "content": "Full owner requirement. " * 200},
                call("read_file", {}, "read"), {"role": "tool", "tool_call_id": "read", "content": "raw " * 2000}]
    request = ContextReclaimRequest("route", "round", context_compaction.context_reclaim_transcript_sha256(observed),
        "cold_estimate", 1.0, 0, working_note="I understood the source.",
        expected_view_revision=context_compaction.context_reclaim_transcript_sha256(observed), keep_unit_ids=())
    fit = lambda messages, tools: _measure_main_context_view(plan, messages, tools, "max", "high", "1")
    owner_words = ("Full owner requirement. " * 200,)  # what the loop passes from the typed owner corpus
    candidate, receipt, usage = context_compaction.compact_tool_history_llm(
        observed, request=request, observed_messages=observed, tool_schemas=[], fit_candidate=fit,
        drive_root=tmp_path, task_id="large-main", protected_texts=owner_words)
    assert receipt.status == "applied" and receipt.reclaimed_tokens > 0 and usage is None
    assert receipt.fit["accepted"] and receipt.fit["predicted_capacity_miss"]
    assert receipt.fit["strict_bound_proven"] is False
    assert candidate[:3] == observed[:3]  # the owner's words stay whole under keep_unit_ids=()
    untyped, untyped_receipt, _ = context_compaction.compact_tool_history_llm(
        observed, request=request, observed_messages=observed, tool_schemas=[], fit_candidate=fit,
        drive_root=tmp_path, task_id="large-main")
    assert untyped_receipt.status == "applied" and untyped[:2] == observed[:2]
    assert untyped[2] != observed[2]  # an untyped host prose row is an addressable dialogue unit (3A)
    unknown = _measure_main_context_view(replace(plan, status="unknown"), candidate, [], "max", "high", "1")
    assert unknown["accepted"] and unknown["capacity_total_tokens"] is None
    unbound = _measure_main_context_view(None, candidate, [], "max", "high", "1")
    assert unbound["accepted"] and unbound["capacity_total_tokens"] is None
    assert unbound["estimated_input_tokens"] > 0 and unbound["strict_bound_proven"] is False


def test_standalone_main_without_route_plan_can_author_a_view(main_loop):
    f = main_loop
    f.ctx.context_fit_plan = None
    f.ctx.task_model_override = "openai/test-model"
    f.run([call("read_file", {"path": "evidence.txt"}, "read"),
           call("compact_context", {"working_note": "The source is retained.", "keep_unit_ids": []}, "compact"),
           {"content": "done"}])
    assert f.ctx._context_view_receipt["status"] == "applied"
    assert f.ctx._context_view_receipt["fit"]["capacity_total_tokens"] is None


@pytest.mark.parametrize("mode", ["nano", "low", "max"])
def test_main_schema_only_view_uses_the_active_envelope(main_loop, monkeypatch, mode):
    f = main_loop
    monkeypatch.setenv("OUROBOROS_CONTEXT_MODE", mode)
    f.ctx.context_fit_plan = replace(f.ctx.context_fit_plan, preferred_mode=mode, initial_mode=mode,
        nano_projection=replace(f.ctx.context_fit_plan.max_projection, mode="nano"))
    f.messages = f.ctx.context_fit_plan.messages_for(mode)
    f.run([call("compact_context", {"working_note": "", "schema_names": ["read_file"]}, "schemas"),
           {"content": "done"}])
    before = {s["function"]["name"] for s in f.inputs[0]["tools"]}
    after = {s["function"]["name"] for s in f.inputs[1]["tools"]}
    if mode == "nano":
        assert "read_file" not in before and "read_file" in after
        assert f.ctx._context_view_receipt["status"] == "applied"
    else:
        assert before == after and f.ctx._context_view_receipt["status"] == "no_op"
    assert f.ctx._context_view_receipt["checkpoint_ref"] is None


def test_successful_fallback_owns_the_next_authored_view(main_loop, monkeypatch):
    import httpx
    from ouroboros import context

    f = main_loop
    monkeypatch.setenv("OUROBOROS_MODEL_FALLBACKS", "openai/alternate")
    monkeypatch.setattr(context, "_context_fit_route", lambda task, **kw: (
        {"model": task["model"], "provider": "openai"},
        SimpleNamespace(route_fp="fallback-route", status="confirmed", stale=False, window_tokens=900_000)))
    refused = httpx.HTTPStatusError("fixture refusal", request=httpx.Request("POST", "https://fixture.invalid"),
                                    response=httpx.Response(400, json={"error": {"message": "fixture refusal"}}))
    f.run([call("read_file", {"path": "evidence.txt"}, "read"), refused,
           call("compact_context", {"working_note": "The fallback retained the source.", "keep_unit_ids": []}, "compact"),
           {"content": "done"}])
    assert [row["model"] for row in f.inputs] == ["openai/test-model", "openai/test-model", "openai/alternate", "openai/alternate"]
    assert f.ctx._context_view_receipt["status"] == "applied"
    assert f.ctx._context_view_receipt["fit"]["route_fp"] == "fallback-route"
    assert "The fallback retained the source." in str(f.inputs[-1]["messages"])
    checkpoint = json.loads(read_actor_source_bytes(f.ctx.drive_root, "authored-main", f.ctx._context_view_receipt["checkpoint_ref"]))
    assert checkpoint["messages"] == f.inputs[2]["messages"]
