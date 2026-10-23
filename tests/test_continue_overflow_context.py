"""A real Continue brief survives Main's provider-refusal address recovery.

Admission, startup, first-request construction, file delivery and the recovery
loop are real. Only the model transport is a fixture; no provider is called.
"""
from __future__ import annotations

from copy import deepcopy
import json
import queue
from types import SimpleNamespace

import httpx
import pytest

from tests._budget_pause_exact_helpers import _install_queue
from tests.test_continue_successor_context import FIELD, _mutation_evidence
from tests.test_doc_context import _make_env_and_memory
from tests.test_main_authored_context import call
from tests.test_owner_continue import NONCE, _interrupted


def _run_refusal(registry, messages, task, monkeypatch):
    from ouroboros import loop, usage_accounting as ua
    from ouroboros.llm_attempt import _attempt_request, _physical_candidate
    from ouroboros.model_send_seal import persist_physical_candidate

    inputs = []
    monkeypatch.setattr(loop, "_maybe_inject_finalization_nudges", lambda *_a: False)
    monkeypatch.setattr("ouroboros.loop_llm_call._sleep_within_deadline", lambda *_a, **_kw: True)

    class Provider:
        def default_model(self):
            return registry._ctx.context_fit_plan.model

        def chat(self, **kwargs):
            ua.adopt_physical_attempt_capture(None)
            inputs.append(deepcopy(kwargs))
            assert len(inputs) <= 3, "recovery must not replay indefinitely"
            candidate = _physical_candidate({key: kwargs[key] for key in ("messages", "tools", "model")})
            request = _attempt_request({"provider": "openai", "usage_model": kwargs["model"]}, candidate)
            attempt_id = f"continue-send-{len(inputs)}"
            persisted = persist_physical_candidate(registry._ctx.drive_root, task_id=task["id"],
                attempt_id=attempt_id, candidate=candidate, candidate_facts={})
            ua.adopt_physical_attempt_capture(ua.PhysicalAttemptCapture(
                attempt_id, kwargs["model"], "openai", "settled", "canonical_json_v1",
                candidate_manifest_ref=persisted["manifest_ref"], physical_context=request.physical_context,
                candidate_raw_sha256=request.candidate_raw_sha256,
                candidate_raw_size_bytes=request.candidate_raw_size_bytes,
                candidate_context_sha256=request.candidate_context_sha256,
                candidate_context_size_bytes=request.candidate_context_size_bytes))
            if len(inputs) == 2:
                raise httpx.HTTPStatusError("context_length_exceeded",
                    request=httpx.Request("POST", "https://fixture.invalid"),
                    response=httpx.Response(400, json={"error": {"code": "context_length_exceeded",
                                                               "message": "Input exceeds context window"}}))
            response = (call("read_file", {"path": "evidence.txt", "max_lines": 2000}, "read-evidence")
                        if len(inputs) == 1 else {"content": "Continue recovered."})
            return response, {"prompt_tokens": 100, "completion_tokens": 10, "cost": 0.0, "provider": "openai"}

    answer, _, _ = loop.run_llm_loop(
        messages=messages, tools=registry, llm=Provider(),
        drive_logs=registry._ctx.drive_root / "logs", drive_root=registry._ctx.drive_root,
        emit_progress=lambda *_a, **_kw: None, incoming_messages=queue.Queue(),
        task_id=task["id"], event_queue=queue.Queue())
    assert answer == "Continue recovered." and len(inputs) == 3
    return inputs


@pytest.mark.parametrize("flag", [True, False, None])
@pytest.mark.parametrize("mode", ["max", "low"])
def test_continue_brief_survives_overflow_relocation(tmp_path, monkeypatch, flag, mode):
    from ouroboros.agent_startup_checks import validate_task_authority_sources
    from ouroboros.artifacts import read_actor_source_bytes
    from ouroboros.context import build_llm_messages
    from ouroboros.context_compaction import _capsule_metadata
    from ouroboros.main_context_authority import project_main_task_authority
    from ouroboros.task_results import load_task_result
    from ouroboros.tools.registry import ToolRegistry
    from supervisor.continuation_admission import admit_continuation

    env, memory = _make_env_and_memory(tmp_path)
    _q, _state, workers = _install_queue(env.drive_root, monkeypatch)
    for key, value in {"OUROBOROS_CONTEXT_MODE": mode, "OUROBOROS_TASK_REVIEW_MODE": "off",
                       "OUROBOROS_SAFETY_MODE": "off", "OUROBOROS_MAX_ROUNDS": "12",
                       "MCP_ENABLED": "false"}.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr("ouroboros.context.resolve_context_fit_route", lambda *_a, **_kw: (
        {"model": "openai/test-model", "provider": "openai"},
        SimpleNamespace(route_fp="fixture", status="asserted", stale=False, window_tokens=1_000_000)))
    workspace = tmp_path / "working-copy"
    workspace.mkdir()
    _interrupted(env.drive_root, workspace_root=str(workspace), workspace_mode="external",
        task_contract={"context": "PREDECESSOR-CONTRACT-SENTINEL", **({FIELD: flag} if flag is not None else {})},
        mutation_evidence=_mutation_evidence())
    ack = admit_continuation("pred-1", action_nonce=NONCE)
    assert ack["ok"] and not ack["held"], ack
    task = deepcopy(next(row for row in workers.PENDING if row["id"] == ack["successor_task_id"]))
    before = load_task_result(env.drive_root, "pred-1")
    assert validate_task_authority_sources(env, task) == {}
    registry = ToolRegistry(repo_dir=env.repo_dir, drive_root=env.drive_root)
    registry._ctx.task_metadata = task["metadata"]
    messages, _cap = build_llm_messages(env=env, memory=memory, task=task, ctx=registry._ctx)
    (env.repo_dir / "evidence.txt").write_text("Exact source with meaningful trailing evidence.\n" * 1000)

    inputs = _run_refusal(registry, messages, task, monkeypatch)

    # The successor inherits the predecessor's brief and its reader, never the contract body (THESIS I-7, C5).
    brief = project_main_task_authority(task)["predecessor_authority"]
    assert brief["source"]["arguments"] == {"task_id": "pred-1", "include_authority": True}
    assert brief["omitted_fields"]["task_contract"] > 0
    assert before["task_contract"]["context"] == "PREDECESSOR-CONTRACT-SENTINEL"
    for index, request in enumerate(inputs):
        system = json.dumps([m for m in request["messages"] if m["role"] == "system"])
        assert "PREDECESSOR-CONTRACT-SENTINEL" not in system and "predecessor_authority" in system
        # Low projects books as navigation, and a confirmed Max overflow retries in task-local
        # Low; the stored inclusion choice survives that rendering policy. User content may be blocks.
        assert sum(block.get("text", "").startswith("## DEVELOPMENT.md\n")
                   for m in request["messages"] if m["role"] == "system"
                   for block in m["content"]) == int(flag is True and mode == "max" and index < 2)
        if mode == "low" or index == 2:
            assert "## DEVELOPMENT.md (navigation map)" in system
        users = json.dumps([m for m in request["messages"] if m["role"] == "user"], ensure_ascii=False)
        assert json.dumps(task["text"], ensure_ascii=False)[1:-1] in users
    assert registry._ctx.context_fit_plan.core.docs_need_development is bool(flag)
    assert task["task_contract"][FIELD] is bool(flag)
    assert "owner pressed" not in task["text"].lower() and "already in your context" in task["text"]
    assert len({request["model"] for request in inputs}) == 1
    assert inputs[1]["reasoning_effort"] == inputs[2]["reasoning_effort"]
    refused_read = next(m for m in inputs[1]["messages"] if m.get("tool_call_id") == "read-evidence")
    # Books (Max) or retained host sections (Low) pay before the just-chosen read.
    assert refused_read in inputs[2]["messages"]
    assert not [m for m in registry._ctx.messages if _capsule_metadata(m)[1]]
    if mode == "low":
        receipts = registry._ctx.context_fit_plan.low_projection.memory_facts["host_sections_addressed"]
        assert receipts
        refused_system = "\n\n".join(block["text"] for m in inputs[1]["messages"]
                                     if m["role"] == "system" for block in m["content"])
        for receipt in receipts:
            whole = read_actor_source_bytes(env.drive_root, task["id"], receipt["source_ref"]).decode()
            assert whole in refused_system
    assert load_task_result(env.drive_root, "pred-1") == before, "recovery never rewrites predecessor authority"
