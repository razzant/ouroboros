"""Host-wire qualification through the real native loop, SDK and local HTTP.

Scripted output measures host composition/continuity, not cognition or vendor
blindness. Every original HTTP body is retained beside its host-seal join.
"""
from __future__ import annotations

import hashlib
import io
import json
import queue
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.system_e2e.harness import ScriptedStubModel, MOCK_SLUG, keyless_settings
from tests._usage_store_testing import ledger_rows

FIRST = ("FIRST_POSITION: specimen A supports explanation alpha.\n\n"
         "Sources: DECLARED_QUESTION, DECLARED_FACTS, TOOL_DECLARED_OBSERVATION; "
         "governance and current authority retained. Own process observation preserved.\n"
         "Uncertainty: one observation; test the rival explanation. Конец позиции.")
PEER = "LATER_PEER_ORIGINAL: specimen B supports beta; objection: alpha needs replication."
REPLY = "ORIGINAL_CHILD_REPLY: retain alpha provisionally; accept the replication objection."
FORBIDDEN = (
    "INHERITED_BIOGRAPHY", "INHERITED_WORLD", "INHERITED_DIALOGUE", "INHERITED_REGISTRY",
    "INHERITED_GLOBAL_KNOWLEDGE", "INHERITED_PROJECT_KNOWLEDGE", "INHERITED_WORKPAD",
    "INHERITED_PARENT_CONTEXT", "INHERITED_PARENT_NOTES", "INHERITED_PARENT_REVIEW",
    "INHERITED_PREDECESSOR_BODY", "INHERITED_ATTACHMENT", "INHERITED_REVIEW",
    "OTHER_TASK_PROCESS",
)


def _write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _mail(drive, task_id):
    path = drive / "memory/owner_mailbox" / f"{task_id}.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()] if path.exists() else []


class WireModel(ScriptedStubModel):
    """Reuse the keyless fixture, adding raw-receive evidence and one HTTP 400."""

    def __init__(self, evidence, *, retry):
        super().__init__(model_ids=["mock-model", "mock-model-rebound"], final_answer="Cooperation complete.")
        self.evidence = evidence
        self.received = []
        self.first_retained_at = None
        self.errors = []
        self.before_answer = lambda _body: None
        self.boundary = lambda: False
        base = self._server.RequestHandlerClass
        outer = self

        class CapturingHandler(base):
            def do_POST(self):
                raw = self.rfile.read(int(self.headers.get("Content-Length") or 0))
                ordinal = len(outer.received) + 1
                path = evidence / f"request-{ordinal:02}.body"
                path.write_bytes(raw)
                retained = outer.boundary()
                outer.received.append({"path": str(path), "sha256": hashlib.sha256(raw).hexdigest(),
                                       "first_position_retained": retained})
                if retained and outer.first_retained_at is None:
                    outer.first_retained_at = ordinal
                if retry and ordinal == 1:
                    assert "stream" in json.loads(raw)
                    return self._send({"error": {"message": "Unsupported parameter: stream",
                        "type": "invalid_request_error", "code": "unsupported_parameter"}}, status=400)
                original = self.rfile
                self.rfile = io.BytesIO(raw)
                try:
                    super().do_POST()
                finally:
                    self.rfile = original

        self._server.RequestHandlerClass = CapturingHandler

    def _next_step(self, body):
        try:
            self.before_answer(body)
        except AssertionError as exc:
            self.errors.append(str(exc))
            return None
        return super()._next_step(body)


def _episode(tmp_path, monkeypatch, model, selection):
    from ouroboros import config, context, context_fit, context_health, usage_accounting as ua
    from ouroboros.contracts.task_contract import build_task_contract
    from ouroboros.memory import Memory
    from ouroboros.tools.registry import ToolRegistry
    from supervisor.task_dispatch import build_scheduled_task_payload

    repo, drive, workspace = tmp_path / "repo", tmp_path / "data", tmp_path / "workspace"
    cfg = keyless_settings(model, OUROBOROS_CONTEXT_MODE="max", OUROBOROS_REASONING_SUMMARY="off",
        OUROBOROS_MODEL_LIGHT="openai-compatible::mock-model-rebound",
        OUROBOROS_SUBAGENTS={"enabled": True, "items": [{"subagent_id": "scout",
            "recommended_use": "Inspect declared observations", "effort": "medium",
            "route": {"kind": "api_model", "target_id": MOCK_SLUG}}]})
    cfg["OUROBOROS_SUBAGENTS"] = json.dumps(cfg["OUROBOROS_SUBAGENTS"])
    for key, value in cfg.items():
        if isinstance(value, (str, int, float)):
            monkeypatch.setenv(key, str(value))
    _write(drive / "settings.json", json.dumps(cfg))
    monkeypatch.setattr(config, "SETTINGS_PATH", drive / "settings.json")
    monkeypatch.setattr(config, "DATA_DIR", drive)
    monkeypatch.setenv("OUROBOROS_SETTINGS_PATH", str(drive / "settings.json"))
    monkeypatch.setenv("OUROBOROS_DATA_DIR", str(drive))
    monkeypatch.setenv("OUROBOROS_REPO_DIR", str(repo))
    monkeypatch.setattr(ua, "estimate_cost_optional", lambda *_a, **_kw: 0.0)
    monkeypatch.setattr(context_health, "_stray_server_note", lambda *_: "")
    # Capacity is synthetic, not fetched from a vendor; projections stay real.
    monkeypatch.setattr(context, "_context_fit_route", lambda task, **_kw: (
        {"model": task.get("model") or MOCK_SLUG, "provider": "openai-compatible"},
        SimpleNamespace(route_fp="wire-" + str(task.get("model") or MOCK_SLUG),
                        status="confirmed", stale=False, window_tokens=2_000_000, source="fixture")))
    monkeypatch.setattr(context_fit, "_route_calibration_ratio", lambda *_a, **_kw: 1.0)
    for name, text in {"prompts/SYSTEM.md": "GOVERNANCE_SYSTEM",
                       "BIBLE.md": "GOVERNANCE_CONSTITUTION",
                       "docs/ARCHITECTURE.md": "# Architecture\n\nGOVERNANCE_ARCHITECTURE\n",
                       "docs/DEVELOPMENT.md": "# Development\n\nGOVERNANCE_DEVELOPMENT\n",
                       "evidence.txt": "TOOL_DECLARED_OBSERVATION\n" * 300}.items():
        _write((workspace if name == "evidence.txt" else repo) / name, text)
    # Seed WORLD before ensure_files so no operator environment profiler runs.
    _write(drive / "memory/WORLD.md", "INHERITED_WORLD")
    memory = Memory(drive_root=drive, repo_dir=repo)
    memory.ensure_files()
    for name, text in {"identity.md": "INHERITED_BIOGRAPHY", "dialogue_summary.md": "INHERITED_DIALOGUE",
                       "registry.md": "### INHERITED_REGISTRY\n- **path:** synthetic-source.md", "scratchpad.md": "INHERITED_WORKPAD",
                       "knowledge/overview.md": "INHERITED_GLOBAL_KNOWLEDGE",
                       "knowledge/index-full.md": "INHERITED_GLOBAL_KNOWLEDGE"}.items():
        _write(drive / "memory" / name, text)
    parent = ToolRegistry(repo_dir=repo, drive_root=drive)
    parent._ctx.task_id = "parent"
    parent._ctx.workspace_root = workspace
    parent._ctx.workspace_mode = "external"
    parent._ctx.pending_events = []
    parent._ctx.task_contract = build_task_contract({"task_contract": {
        "context": "INHERITED_PARENT_CONTEXT", "notes": "INHERITED_PARENT_NOTES",
        "review_notes": "INHERITED_PARENT_REVIEW", "constraints": "AUTHORITY_CONSTRAINT",
        "predecessor_authority": {"task_id": "previous", "result": "INHERITED_PREDECESSOR_BODY",
                                  "authority_sha256": "a" * 64},
    }})
    from ouroboros.artifacts import stage_task_attachments
    _write(workspace / "prior-note.txt", "READ_PREVIOUS_CASE_CONCLUSION")
    parent._ctx.task_contract["attachment_manifest"] = stage_task_attachments(
        drive, "parent", [{"path": str(workspace / "prior-note.txt"), "label": "INHERITED_ATTACHMENT"}])
    result = parent.execute_result("schedule_subagent", {"subagent_id": "scout",
        "objective": "DECLARED_QUESTION", "context": "DECLARED_FACTS", "constraints": "AUTHORITY_CONSTRAINT",
        "expected_output": "Retain complete first position then cooperate through ordinary mail.",
        "input_sources": selection})
    assert result.status == "ok", result.text
    event = parent._ctx.pending_events[-1]
    task = build_scheduled_task_payload({**event, "tid": event["task_id"], "parent_id": "parent",
                                       "task_context": event["context"], "desc": event["objective"]})
    task.update(model=MOCK_SLUG, budget_drive_root=str(drive), project_id="case",
                workspace_root=str(workspace), workspace_mode="external")
    _write(drive / "projects/case/knowledge/index-full.md", "INHERITED_PROJECT_KNOWLEDGE")
    _write(drive / "projects/case/workpad.md", "INHERITED_WORKPAD")
    from ouroboros.task_results import write_task_result
    for task_id, data in {"parent": {}, "sibling": {"parent_task_id": "parent"},
                          task["id"]: {"parent_task_id": "parent", "child_drive_root": str(drive)}}.items():
        write_task_result(drive, task_id, "running", root_task_id="parent", **data)
    _write(drive / "logs/progress.jsonl", "\n".join(json.dumps({"task_id": tid, "text": text}) for tid, text in
        [(task["id"], "OWN_PROCESS_OBSERVATION"), ("different", "OTHER_TASK_PROCESS")]) + "\n")
    env = SimpleNamespace(repo_dir=repo, drive_root=drive, repo_path=lambda p: repo / p,
                          drive_path=lambda p: drive / p)
    registry = ToolRegistry(repo_dir=repo, drive_root=drive)
    registry._ctx.task_id = task["id"]
    registry._ctx.task_constraint = task["task_constraint"]
    registry._ctx.workspace_root = workspace
    registry._ctx.workspace_mode = "external"
    registry._ctx.task_contract = task["task_contract"]
    registry._ctx.task_metadata = task
    registry._ctx.budget_drive_root = drive
    registry._ctx.task_model_override = MOCK_SLUG
    registry._ctx.context_fit_plan = context.build_context_fit_plan(
        env, memory, task, lambda: "INHERITED_REVIEW", ctx=registry._ctx, preferred_mode="max")
    peer = ToolRegistry(repo_dir=repo, drive_root=drive)
    peer._ctx.task_id = "sibling"
    peer._ctx.task_metadata = {"parent_task_id": "parent", "root_task_id": "parent"}
    return SimpleNamespace(task=task, tools=registry, peer=peer, drive=drive, env=env, memory=memory)


def _run(episode):
    from ouroboros import loop, usage_accounting as ua
    from ouroboros.llm import LLMClient

    scope = ua.UsageScope(drive_root=episode.drive, task_id=episode.task["id"], root_task_id="parent",
                          category="task", source="test.first_input_wire")
    with ua.usage_scope(scope):
        return loop.run_llm_loop(messages=episode.tools._ctx.context_fit_plan.messages_for("max"),
            tools=episode.tools, llm=LLMClient(api_key="unused-keyless"),
            drive_logs=episode.drive / "logs", drive_root=episode.drive,
            task_id=episode.task["id"], task_type="task", incoming_messages=queue.Queue(),
            emit_progress=lambda *_a, **_kw: None)


def _seals(episode, model):
    from ouroboros import model_send_seal
    from ouroboros.request_wire_contract import physical_candidate_bytes

    rows = {}
    for row in ledger_rows(episode.drive):
        if row.get("candidate_manifest_ref"):
            rows[row["attempt_id"]] = row
    assert len(rows) == len(model.received)
    remaining = list(rows.values())
    joined = []
    for received in model.received:
        body = json.loads(Path(received["path"]).read_bytes())
        matches = []
        for row in remaining:
            manifest = json.loads(Path(row["candidate_manifest_ref"]["path"]).read_text(encoding="utf-8"))
            blob, intact = model_send_seal._read_blob_bytes(manifest)
            assert intact
            candidate = json.loads(blob)
            # The SDK flattens extra_body; all message/tool bytes must still match.
            wire_candidate = {**candidate, **candidate.get("extra_body", {})}
            wire_candidate.pop("extra_body", None)
            if wire_candidate == body:
                matches.append((row, manifest, candidate))
        assert len(matches) == 1, (received, [row["attempt_id"] for row in remaining])
        row, manifest, candidate = matches[0]
        remaining.remove(row)
        seal = manifest["model_send_seal"]
        assert seal["attempt_id"] == row["attempt_id"]
        assert seal["pre_redaction_sha256"] == row["candidate_raw_sha256"]
        assert hashlib.sha256(physical_candidate_bytes(candidate)).hexdigest() == seal["pre_redaction_sha256"]
        assert {x["class"] for x in seal["exclusions"]} == {"transport_envelope", "provider_side_transform"}
        text = json.dumps(body, ensure_ascii=False)
        inherited = [marker for marker in FORBIDDEN if marker in text]
        joined.append({**received, "observed_inherited_markers": inherited,
                       "automatic_source_exclusion": "RED" if inherited else "GREEN",
                       "wire_object_comparison": "exact after SDK extra_body flattening",
                       "attempt_id": row["attempt_id"], "seal": seal,
                       "candidate_manifest_ref": row["candidate_manifest_ref"]})
    assert not remaining
    return joined


@pytest.mark.serial
@pytest.mark.parametrize("selection", ["shared", "declared"])
def test_native_loop_retains_first_position_then_cooperates_over_http(tmp_path, monkeypatch, selection):
    evidence = tmp_path / "wire-evidence"
    evidence.mkdir()
    with WireModel(evidence, retry=True) as model:
        ep = _episode(tmp_path, monkeypatch, model, selection)
        model.boundary = lambda: any(row["text"] == FIRST for row in _mail(ep.drive, "parent"))
        model.script = [
            {"tool": "read_file", "arguments": {"path": "evidence.txt", "root": "active_workspace"}},
            {"tool": "switch_model", "arguments": {"model": "openai-compatible::mock-model-rebound"}},
            {"tool": "read_file", "arguments": {"path": "evidence.txt", "root": "active_workspace"}},
            {"tool": "forward_to_worker", "arguments": {"task_id": "parent", "message": FIRST}},
            {"tool": "await_messages", "arguments": {"timeout_sec": 1}},
            {"tool": "forward_to_worker", "arguments": {"task_id": "sibling", "message": REPLY}},
        ]
        def cooperate(_body):
            if model._script_index == 4:
                assert model.boundary(), "peer evidence cannot arrive before the complete authored first position"
                sent = ep.peer.execute_result("forward_to_worker", {"task_id": ep.task["id"], "message": PEER})
                assert sent.status == "ok", sent.text
        model.before_answer = cooperate
        final, usage, trace = _run(ep)
        assert not model.errors, model.errors
        assert final == "Cooperation complete.", (final, usage, trace)
        assert model.script_consumed()
        assert all(not row["is_error"] for row in trace["tool_calls"]), trace["tool_calls"]
        assert len(model.received) == 8
        assert model.first_retained_at == 6
        bodies = [json.loads(Path(row["path"]).read_bytes()) for row in model.received]
        assert bodies[0]["stream"] is True and "stream" not in bodies[1]
        assert [b["model"] for b in bodies[:4]] == ["mock-model"] * 3 + ["mock-model-rebound"]
        for n, body in enumerate(bodies, 1):
            text = json.dumps(body, ensure_ascii=False)
            for marker in ("GOVERNANCE_SYSTEM", "GOVERNANCE_CONSTITUTION", "DECLARED_QUESTION",
                           "DECLARED_FACTS", "AUTHORITY_CONSTRAINT", "OWN_PROCESS_OBSERVATION"):
                assert marker in text, (n, marker)
            if selection == "declared":
                assert all(marker not in text for marker in FORBIDDEN), n
            else:
                # The SAME exclusion predicate is RED for the ordinary baseline.
                with pytest.raises(AssertionError):
                    assert all(marker not in text for marker in FORBIDDEN)
                # A helper's view keeps knowledge out of the request: it reads it on demand.
                knowledge = ("INHERITED_GLOBAL_KNOWLEDGE", "INHERITED_PROJECT_KNOWLEDGE", "INHERITED_WORKPAD")
                assert all(marker in text for marker in FORBIDDEN if marker not in ("OTHER_TASK_PROCESS", *knowledge)), n
                assert not any(marker in text for marker in knowledge), n
            assert (PEER in text) == (n >= 7), n
            if n >= 3:
                assert "TOOL_DECLARED_OBSERVATION" in text
        first = [row for row in _mail(ep.drive, "parent") if row["text"] == FIRST]
        assert len(first) == 1 and first[0]["source_task_id"] == ep.task["id"]
        assert first[0]["provenance"] == "peer_task" and first[0]["relation"] == "parent"
        reply = [row for row in _mail(ep.drive, "sibling") if row["text"] == REPLY]
        assert len(reply) == 1 and reply[0]["source_task_id"] == ep.task["id"]
        assert "Message from peer task sibling (sibling)" in json.dumps(bodies[6])
        assert "owner_mailbox_pending" in json.dumps(bodies[6])
        joined = _seals(ep, model)
        report = {"qualification": "host-wire only; scripted provider, not cognitive/vendor blindness",
                  "selection": selection, "first_position": first[0], "first_seen_retained_request": 6,
                  "requests": joined, "loop_trace": trace,
                  "limits": ["Scripted API provider; no vendor or cognitive blindness claim",
                             "Raw HTTP body hash and canonical host-seal hash are distinct byte domains",
                             "Automatic fallback and overflow/compaction not exercised on this wire",
                             "compact_context is refused by child tool profiles; authored compaction has separate focused coverage"]}
        _write(evidence / "qualification.json", json.dumps(report, ensure_ascii=False, indent=2))
        print(f"HOST_WIRE_EVIDENCE {evidence / 'qualification.json'}")


@pytest.mark.serial
@pytest.mark.parametrize("channel", ["tool", "mailbox"])
def test_pre_position_reads_and_mail_are_visible_broader_inputs(tmp_path, monkeypatch, channel):
    """Selection omits automatic biography; ordinary tools/mail can add sources."""
    evidence = tmp_path / "wire-evidence"
    evidence.mkdir()
    marker = "READ_PREVIOUS_CASE_CONCLUSION" if channel == "tool" else "EARLY_PEER_PREVIOUS_CONCLUSION"
    position = FIRST + "\nBroader source account: " + marker + " via " + channel + "."
    with WireModel(evidence, retry=False) as model:
        ep = _episode(tmp_path, monkeypatch, model, "declared")
        model.boundary = lambda: any(row["text"] == position for row in _mail(ep.drive, "parent"))
        if channel == "mailbox":
            sent = ep.peer.execute_result("forward_to_worker", {"task_id": ep.task["id"], "message": marker})
            assert sent.status == "ok", sent.text
        model.script = [
            {"tool": "read_file", "arguments": {"path": "prior-note.txt" if channel == "tool" else "evidence.txt", "root": "active_workspace"}},
            {"tool": "forward_to_worker", "arguments": {"task_id": "parent", "message": position}},
        ]
        final, _usage, trace = _run(ep)
        assert final == "Cooperation complete." and model.script_consumed()
        assert all(not row["is_error"] for row in trace["tool_calls"])
        assert len(model.received) == 3 and model.first_retained_at == 3
        bodies = [json.loads(Path(row["path"]).read_bytes()) for row in model.received]
        for n, body in enumerate(bodies, 1):
            text = json.dumps(body, ensure_ascii=False)
            assert "GOVERNANCE_SYSTEM" in text and "DECLARED_FACTS" in text
            assert "INHERITED_BIOGRAPHY" not in text
            assert (marker in text) == (channel == "mailbox" or n >= 2)
        if channel == "mailbox":
            assert "Message from peer task sibling (sibling)" in json.dumps(bodies[0])
        first = next(row for row in _mail(ep.drive, "parent") if row["text"] == position)
        assert first["source_task_id"] == ep.task["id"] and first["provenance"] == "peer_task"
        report = {"qualification": "host-wire only", "selection": "declared",
                  "broader_input_channel": channel, "additional_source": marker,
                  "first_position": first, "first_seen_retained_request": 3,
                  "requests": _seals(ep, model)}
        _write(evidence / "qualification.json", json.dumps(report, ensure_ascii=False, indent=2))
        print(f"HOST_WIRE_EVIDENCE {evidence / 'qualification.json'}")
