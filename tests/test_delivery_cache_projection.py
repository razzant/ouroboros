"""Real image attachment and outcome facts across cache seals and wire builders."""
from copy import deepcopy
from dataclasses import replace
import json
from types import SimpleNamespace

import pytest
from PIL import Image

from ouroboros.artifacts import read_actor_source_bytes
from ouroboros.context_budget import MAX_LIVE_IMAGE_BLOCKS
from ouroboros.context_fit import seal_task_transcript
from ouroboros.loop_tool_execution import process_tool_results
from ouroboros.tool_result_record import (
    TOOL_RESULT_RECORD_KEY, make_tool_result_record, read_tool_result_record,
)
from ouroboros.tools.registry import ToolContext
from ouroboros.tools.tool_result import ToolResult
from ouroboros.tools.vision import attach_local_image_to_context
from tests.test_context_reclaim_materializer import _request
from tests.test_main_authored_context import call, main_loop as _main_loop

main_loop = _main_loop


def _images(value):
    if isinstance(value, list):
        return [image for item in value for image in _images(item)]
    if not isinstance(value, dict):
        return []
    if value.get("type") == "image_url":
        return [value["image_url"]["url"].split(",", 1)[-1]]
    if value.get("type") == "image":
        return [value["source"]["data"]]
    return [image for item in value.values() for image in _images(item)]


def _wires(messages, monkeypatch):
    from ouroboros import config
    from ouroboros.llm import LLMClient
    from ouroboros.llm_claudexor import _request, ModelTurnState
    monkeypatch.setattr("ouroboros.llm_claudexor.owned_engine_version", lambda: config.CLAUDEXOR_MODEL_TURN_STATE_MIN_VERSION)
    client = LLMClient.__new__(LLMClient)
    out = {"codex": _request({"provider": "claudexor", "source": "codex", "resolved_model": "fixture-model"},
        deepcopy(messages), [], {"prospective": True, "_no_account_preference": True,
                                 "model_account_override": "", "model_turn_state": ModelTurnState()})}
    for provider in ("anthropic", "openrouter"):
        target = {"provider": provider, "resolved_model": "anthropic/claude-sonnet-4.5" if provider == "openrouter" else "claude-sonnet-4-5",
                  "supports_openrouter_extensions": provider == "openrouter"}
        out[provider] = client._build_remote_candidate(target, deepcopy(messages), "high", 65536,
                                                      "auto", None, [], skip_capability_fetch=True)
    return out


@pytest.mark.parametrize("batch_size", [1, 3, 7])
def test_real_png_auto_batches_share_eviction_projection_with_fit_and_all_wire_builders(tmp_path, monkeypatch, batch_size):
    repo = tmp_path / "repo"
    repo.mkdir()
    ctx = ToolContext(repo_dir=repo, drive_root=tmp_path / "data", task_id="image-projection")
    messages = [{"role": "system", "content": "Required core"}, {"role": "user", "content": "Inspect pixels"}]
    ctx.messages = messages
    paths = []
    for n in range(MAX_LIVE_IMAGE_BLOCKS + batch_size):
        path = repo / f"pixel-{n}.png"
        Image.new("RGB", (2, 2), (n, n + 1, n + 2)).save(path, format="PNG")
        paths.append(path)
    originals = [path.read_bytes() for path in paths]
    for path in paths[:MAX_LIVE_IMAGE_BLOCKS]:
        assert attach_local_image_to_context(ctx, str(path))[0]
    initial = deepcopy(messages)
    rows = [{"fn_name": "ext_fixture_screenshot", "tool_call_id": f"image-{n}",
             "result": json.dumps({"auto_attach_image": str(path)}), "is_error": False,
             "tool_args": {}, "args_for_log": {}, "result_meta": {}}
            for n, path in enumerate(paths[MAX_LIVE_IMAGE_BLOCKS:])]
    messages.append({"role": "assistant", "content": "", "tool_calls": [
        {"id": row["tool_call_id"], "type": "function", "function": {"name": row["fn_name"], "arguments": "{}"}}
        for row in rows]})
    before, measured = deepcopy(messages), []
    def fit(candidate, _schemas):
        assert messages == before, "Measurement mutated the live history before adoption"
        measured.append(deepcopy(candidate))
        return {"estimated_input_tokens": 1000, "capacity_total_tokens": 100000, "response_reserve_tokens": 1000}

    trace = {"tool_calls": []}
    process_tool_results(rows, messages, trace, lambda *_a, **_kw: None,
                         SimpleNamespace(_ctx=ctx), fit_candidate=fit, tool_schemas=[])
    assert measured and all(len(_images(m)) == MAX_LIVE_IMAGE_BLOCKS for m in measured)
    assert _images(messages) == _images(measured[-1])
    for wire in _wires(messages, monkeypatch).values():
        assert _images(wire["messages"]) == _images(messages)
        assert "_host_context_kind" not in json.dumps(wire)
    assert [m["role"] for m in messages[len(before):len(before) + len(rows)]] == ["tool"] * len(rows)
    assert all(row["image_attachment"] == {"status": "attached"} for row in trace["tool_calls"])
    assert [path.read_bytes() for path in paths] == originals
    # Positive direct path uses the same existing cap and image implementation.
    ctx.messages = deepcopy(initial)
    for path in paths[MAX_LIVE_IMAGE_BLOCKS:]:
        assert attach_local_image_to_context(ctx, str(path))[0]
    assert _images(ctx.messages) == _images(messages)


def test_typed_error_and_empty_result_remain_known_through_real_main_seals_and_address_recovery(main_loop):
    from ouroboros import context_compaction as cc
    from ouroboros.context_source_view import emergency_address_view
    f, produced = main_loop, []
    def command(_ctx, _resolved_binding=None, **_kw):
        i = len(produced)
        body = "" if i == 2 else f"Measured process row {i}.\n" * 600
        result = ToolResult(status="error" if i in (1, 2) else "ok",
                            code="SHELL_EXIT_ERROR" if i in (1, 2) else "OK", text=body,
                            meta={"exit_code": 17 if i in (1, 2) else 0})
        produced.append(result)
        return result
    f.registry.override_handler("run_command", command)
    f.run([*(call("run_command", {"cmd": ["pwd"]}, f"process-{n}") for n in range(9)), {"content": "done"}])
    assert len(produced) == 9
    sealed_seen = False
    for entry in f.inputs:
        for row in entry["messages"]:
            if row.get("tool_call_id") not in {"process-1", "process-2"}:
                continue
            fact = read_tool_result_record(row)
            assert fact["state"] == "recorded"
            assert fact["facts"] == {"status": "error", "code": "SHELL_EXIT_ERROR", "is_error": True, "exit_code": 17}
            if row["tool_call_id"] == "process-2":
                assert row["content"] == ""
            else:
                sealed_seen |= isinstance(row["content"], list)
    assert sealed_seen
    messages = f.inputs[7]["messages"]
    # Reclaim through the failed command, rather than stopping after the first
    # successful unit. Exposure comes from the real usable Main observation.
    goal = cc.context_units(messages)[0].context_size_tokens
    request = replace(_request(messages, 1), reclaim_goal_tokens=goal)
    rebuilt, receipt = emergency_address_view(messages, request, rung="bodies",
        observation=f.ctx._last_context_observation, drive_root=f.ctx.drive_root, task_id=f.ctx.task_id)
    assert receipt.status == "applied"
    visible = "\n".join(row["content"][0]["text"] for row in rebuilt if cc._capsule_metadata(row)[1])
    assert "SHELL_EXIT_ERROR" in visible and '"exit_code":17' in visible and '"is_error":true' in visible
    assert "content_mismatch" not in visible
    checkpoint = json.loads(read_actor_source_bytes(f.ctx.drive_root, f.ctx.task_id, receipt.checkpoint_ref))
    assert checkpoint["messages"] == messages
    # Authored relocation carries the exact known producer source too; a cache
    # marker cannot turn the bound invocation into an unknown legacy source.
    request = _request(messages, 1)
    request = replace(request, working_note="The failed commands returned exit 17.", keep_unit_ids=(),
                      expected_view_revision=request.transcript_sha256)
    authored, author_receipt, _ = cc.compact_tool_history_llm(messages, request=request,
        observed_messages=messages, fit_candidate=lambda *_: {"accepted": True},
        drive_root=f.ctx.drive_root, task_id=f.ctx.task_id)
    assert author_receipt.status == "applied"
    error = next(row for row in messages if row.get("tool_call_id") == "process-1")
    record = read_tool_result_record(error)
    refs = [ref for row in authored if (meta := cc._capsule_metadata(row)[1]) for ref in meta["source_refs"]]
    assert record["trace_ref"] in refs
    source = json.loads(read_actor_source_bytes(f.ctx.drive_root, f.ctx.task_id, author_receipt.checkpoint_ref))
    assert read_tool_result_record(next(row for row in source["messages"] if row.get("tool_call_id") == "process-1"))["facts"]["exit_code"] == 17


@pytest.mark.parametrize("body", ["changed bytes", [{"type": "image_url", "image_url": {"url": "opaque"}}],
                                 [{"type": "text", "text": "exact", "citations": ["meaning"]}]])
def test_cache_sealing_never_makes_a_changed_or_opaque_result_known(body):
    row = {"role": "tool", "tool_call_id": "same", "content": body,
           TOOL_RESULT_RECORD_KEY: make_tool_result_record({"tool_call_id": "same", "invocation_id": "host"},
               "exact", facts={"status": "ok", "is_error": False, "exit_code": 0})}
    messages = [{"role": "user", "content": "task"}, row]
    before = deepcopy(body)
    seal_task_transcript(messages, keep_active=0, min_prefix_tokens=0)
    assert read_tool_result_record(row)["state"] == "unknown"
    if isinstance(body, list):
        assert row["content"] == before


def test_cache_text_wrapper_preserves_text_boundaries_and_exact_empty_output():
    for value in ("exact", ""):
        row = {"role": "tool", "tool_call_id": "same", "content": [
            {"type": "text", "text": value[:2]},
            {"type": "text", "text": value[2:], "cache_control": {"type": "ephemeral"}}],
            TOOL_RESULT_RECORD_KEY: make_tool_result_record({"tool_call_id": "same", "invocation_id": "host"}, value,
                                                           facts={"status": "error", "is_error": True, "exit_code": 17})}
        assert read_tool_result_record(row)["state"] == "recorded"
        seal_task_transcript([{"role": "user", "content": "task"}, row], keep_active=0, min_prefix_tokens=0)
        assert read_tool_result_record(row)["state"] == "recorded"
        if not value:
            assert row["content"] == ""
