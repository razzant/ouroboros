"""Offline consumers of a declared-input child: projection, sends, and reclaim.

Transport is replaced by a local callable. These checks establish host input
composition and retained sources, not vendor context or model independence.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from tests._usage_store_testing import ledger_rows


@pytest.fixture
def declared_plan(tmp_path, monkeypatch):
    from ouroboros import context, context_fit
    from ouroboros.memory import Memory

    repo, drive = tmp_path / "repo", tmp_path / "data"
    sources = {
        "prompts/SYSTEM.md": "GOVERNANCE_SYSTEM",
        "BIBLE.md": "GOVERNANCE_CONSTITUTION",
        "docs/ARCHITECTURE.md": "# Architecture\n\nGOVERNANCE_ARCHITECTURE\n",
        "docs/DEVELOPMENT.md": "# Development\n\nGOVERNANCE_DEVELOPMENT\n",
    }
    for relative, text in sources.items():
        target = repo / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
    memory = Memory(drive_root=drive, repo_dir=repo)
    memory.ensure_files()
    (drive / "memory/identity.md").write_text("AUTOMATIC_PREVIOUS_IDENTITY", encoding="utf-8")
    (drive / "memory/dialogue_summary.md").write_text("AUTOMATIC_PREVIOUS_CONCLUSION", encoding="utf-8")
    (drive / "memory/scratchpad.md").write_text("AUTOMATIC_PARENT_WORKPAD", encoding="utf-8")
    task = {
        "id": "declared-child", "type": "task", "delegation_role": "subagent",
        "text": "DECLARED_QUESTION and DECLARED_EVIDENCE",
        "context": "AUTHOR_DECLARED_COMMON_FACT",
        "task_contract": {"input_sources": "declared", "constraints": "AUTHORITY_CONSTRAINT"},
        "configured_subagent": {"route": {"kind": "api_model", "target_id": "fixture"}},
    }
    env = SimpleNamespace(repo_dir=repo, drive_root=drive, repo_path=lambda p: repo / p,
                          drive_path=lambda p: drive / p)

    def route(task, **_kwargs):
        return ({"model": task.get("model") or "fixture", "provider": "fixture"},
                SimpleNamespace(route_fp="fixture", status="unprobeable", stale=False, window_tokens=0))

    monkeypatch.setattr(context, "_context_fit_route", route)
    monkeypatch.setattr(context_fit, "_route_calibration_ratio", lambda *_a, **_kw: 1.0)
    plan = context.build_context_fit_plan(
        env, memory, task, lambda: "AUTOMATIC_PREVIOUS_REVIEW", preferred_mode="max",
    )
    return plan, task, env


def _assert_declared(messages):
    text = json.dumps(messages, ensure_ascii=False)
    assert "DECLARED_QUESTION" in text and "DECLARED_EVIDENCE" in text
    assert "GOVERNANCE_CONSTITUTION" in text
    assert "Input source selection" in text
    assert "declared" in text
    for forbidden in ("AUTOMATIC_PREVIOUS_IDENTITY", "AUTOMATIC_PREVIOUS_CONCLUSION",
                      "AUTOMATIC_PARENT_WORKPAD", "AUTOMATIC_PREVIOUS_REVIEW"):
        assert forbidden not in text


@pytest.mark.parametrize("mode", ["max", "low", "nano"])
def test_selected_input_reaches_every_captured_projection(declared_plan, mode):
    plan, _task, _env = declared_plan
    _assert_declared(plan.messages_for(mode))


def _tool_round(text="TOOL_RETURNED_EVIDENCE"):
    return [
        {"role": "assistant", "content": "Inspect declared evidence", "tool_calls": [
            {"id": "read-1", "type": "function", "function": {
                "name": "read_file", "arguments": '{"path":"evidence.txt"}',
            }},
        ]},
        {"role": "tool", "tool_call_id": "read-1", "content": text},
    ]


def test_route_rebind_uses_captured_selection_and_retains_later_inputs(declared_plan):
    from ouroboros.loop_model_call import _rebind_context_fit_plan

    plan, task, env = declared_plan
    messages = plan.messages_for("max") + _tool_round()
    # A later automatic-memory edit cannot enter an already captured projection.
    (env.drive_root / "memory/dialogue_summary.md").write_text("AUTOMATIC_PREVIOUS_CONCLUSION CHANGED")
    tools = SimpleNamespace(_ctx=SimpleNamespace(task_id=task["id"], task_metadata=task))
    rebound, mode = _rebind_context_fit_plan(
        plan, tools, messages, model="fallback-fixture", use_local=False,
        preferred_mode="low", tool_schemas=[],
    )
    assert mode == "low"
    assert rebound.core_sha256 == plan.core_sha256
    assert messages[2:] == _tool_round()
    _assert_declared(messages)
    # Collaboration is ordinary additive input after the actor retains a position.
    messages.append({"role": "assistant", "content": "RETAINED_FIRST_POSITION"})
    messages.append({"role": "user", "content": "LATER_COLLABORATIVE_ORIGINAL"})
    restored = rebound.reproject_transcript(messages, "max")
    assert restored[2:] == messages[2:]
    _assert_declared(restored)


def test_authored_compaction_retains_selected_core_and_exact_tool_source(declared_plan):
    from ouroboros import context_compaction as cc
    from ouroboros.artifacts import read_actor_source_bytes
    from ouroboros.context_budget import ContextReclaimRequest

    plan, task, env = declared_plan
    messages = plan.messages_for("max") + _tool_round("TOOL_RETURNED_EVIDENCE " * 200)
    revision = cc.context_reclaim_transcript_sha256(messages)
    request = ContextReclaimRequest(
        route_fp=plan.route_fp, round_id="round-1", transcript_sha256=revision,
        measurement_basis="cold_estimate", measurement_density=1.0,
        reclaim_goal_tokens=1, working_note="Actor's retained evidence and unresolved question.",
        expected_view_revision=revision, keep_unit_ids=(),
    )
    rebuilt, receipt, usage = cc.compact_tool_history_llm(
        messages, request=request, observed_messages=copy.deepcopy(messages),
        observed_tool_schemas=[], tool_schemas=[],
        fit_candidate=lambda *_: {"accepted": True},
        drive_root=env.drive_root, task_id=task["id"],
    )
    assert receipt.status == "applied" and usage is None
    assert rebuilt[:2] == messages[:2]
    _assert_declared(rebuilt)
    retained = json.loads(read_actor_source_bytes(env.drive_root, task["id"], receipt.checkpoint_ref))
    assert retained["messages"] == messages
    assert "TOOL_RETURNED_EVIDENCE" in json.dumps(retained)
    assert "Actor's retained evidence" in json.dumps(rebuilt)


def test_original_actor_mailbox_exchange_after_retained_first_position(declared_plan):
    import queue
    from ouroboros.loop_round_limits import _drain_incoming_messages
    from ouroboros.owner_mailbox import write_task_message, PROVENANCE_PEER_TASK

    plan, task, env = declared_plan
    first = "FIRST_POSITION: specimen A. Inputs: DECLARED_EVIDENCE; no tool reads."
    assert write_task_message(env.drive_root, first, "parent", source_task_id=task["id"],
                              provenance=PROVENANCE_PEER_TASK, relation="parent", msg_id="first")
    retained = (env.drive_root / "memory/owner_mailbox/parent.jsonl").read_text()
    assert json.loads(retained)["text"] == first
    assert write_task_message(env.drive_root, "LATER_ORIGINAL_POSITION: specimen B.", task["id"],
                              source_task_id="sibling", provenance=PROVENANCE_PEER_TASK,
                              relation="sibling", msg_id="exchange")
    messages = plan.messages_for("max")
    ctx = SimpleNamespace(task_contract=task["task_contract"], task_metadata=task)
    _drain_incoming_messages(messages, queue.Queue(), env.drive_root, task["id"],
                             None, set(), owner_ctx=ctx)
    _assert_declared(messages)
    assert "LATER_ORIGINAL_POSITION" in json.dumps(messages)
    assert "Message from peer task sibling (sibling)" in json.dumps(messages)


def test_physical_retry_seals_each_actual_selected_send(declared_plan, monkeypatch):
    from ouroboros import model_send_seal, usage_accounting as ua
    from ouroboros.llm import LLMClient

    plan, task, env = declared_plan
    monkeypatch.setenv("OUROBOROS_DATA_DIR", str(env.drive_root))
    monkeypatch.setenv("OUROBOROS_SETTINGS_PATH", str(env.drive_root / "settings.json"))
    monkeypatch.setenv("TOTAL_BUDGET", "100")
    monkeypatch.setattr(ua, "estimate_cost_optional", lambda *_a, **_kw: 0.01)
    sends = []
    client = LLMClient(api_key="unused-offline")

    class Rejected(RuntimeError):
        status_code = 400
        body = {"error": {"code": "unsupported_parameter", "type": "invalid_request_error"}}

    class Response:
        def model_dump(self):
            return {"choices": [{"message": {"role": "assistant", "content": "FIRST_POSITION"}}],
                    "usage": {"prompt_tokens": 50, "completion_tokens": 2, "cost": 0.0}}

    def create(**candidate):
        sends.append(copy.deepcopy(candidate))
        if len(sends) == 1:
            raise Rejected("temperature unsupported")
        return Response()

    def retry_without_temperature(candidate, _model, _exc):
        updated = copy.deepcopy(candidate)
        updated.pop("temperature", None)
        return updated

    monkeypatch.setattr(client, "_retry_without_optional_sampling", retry_without_temperature)
    monkeypatch.setattr(client, "_openrouter_signature_retry_kwargs", lambda *_: None)
    payload = {"model": "fixture", "messages": plan.messages_for("max") + _tool_round(),
               "max_tokens": 32, "temperature": 0.7}
    scope = ua.UsageScope(drive_root=env.drive_root, task_id=task["id"], root_task_id=task["id"],
                          category="task", source="test.first_input")
    with ua.usage_scope(scope):
        client._create_chat_completion_with_retries(
            create, payload, {"provider": "openai", "usage_model": "openai/fixture", "resolved_model": "fixture"},
        )
    assert len(sends) == 2
    for sent in sends:
        _assert_declared(sent["messages"])
        assert "TOOL_RETURNED_EVIDENCE" in json.dumps(sent)
    finals = {}
    for row in ledger_rows(env.drive_root):
        finals[row["attempt_id"]] = row
    assert len(finals) == 2
    for row in finals.values():
        manifest = json.loads(Path(row["candidate_manifest_ref"]["path"]).read_text())
        seal = manifest["model_send_seal"]
        assert seal["pre_redaction_sha256"] == row["candidate_raw_sha256"]
        assert seal["attempt_id"] == row["attempt_id"]
        source, intact = model_send_seal._read_blob_bytes(manifest)
        assert intact
        _assert_declared(json.loads(source)["messages"])
        assert "provider_side_transform" in {item["class"] for item in seal["exclusions"]}
