from __future__ import annotations

import asyncio
import importlib.util
import json
import types


def _load_script_module(repo_root):
    path = repo_root / "scripts" / "run_plan_review.py"
    spec = importlib.util.spec_from_file_location("run_plan_review_script", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_run_plan_review_script_runs_the_engine_on_the_new_envelope(monkeypatch, tmp_path):
    """The operator script drives the SAME engine plan_task uses — goal + plan prose +
    spec JSON + evidence locators — in an isolated drive root, and prints the recorded
    wave (slots, validated findings, host aggregate) plus the coordinated output."""
    import pathlib

    from ouroboros.review_substrate import ReviewSlot
    from ouroboros.tools import plan_review, plan_review_runtime

    repo = pathlib.Path(__file__).resolve().parents[1]
    script = _load_script_module(repo)
    captured = {}

    async def fake_run_slots(ctx, slots, *, system_prompt, user_content, session_task="",
                             session_root="", output_contract="", slot_messages=None,
                             slot_session_tasks=None, request_policy=None,
                             session_threads=None, retry_key="", reconcile_only=False,
                             reconciliation_identity=None, release_at_dispatch=False):
        captured["task_id"] = ctx.task_id
        captured["release_at_dispatch"] = release_at_dispatch
        captured["slots"] = [s.slot_id for s in slots]
        captured["system_prompt"] = system_prompt
        captured["user_content"] = user_content
        captured["retry_key"] = retry_key
        captured["reconcile_only"] = reconcile_only
        captured["reconciliation_identity"] = reconciliation_identity
        return [{
            "slot_id": s.slot_id, "model": s.model, "request_model": s.model, "route": "api_chat",
            "host_file_read_attestation": "host_assembled_packet",
            "text": "[]\nNO_FINDINGS", "error": None, "prompt_ref": {}, "response_ref": {},
            "tokens_in": 1, "tokens_out": 1, "cost": 0.0,
        } for s in slots]

    slots = [ReviewSlot(slot_id="slot_1", model="fake/reviewer", effort="high")]
    monkeypatch.setattr(plan_review, "_plan_review_slots", lambda: slots)
    monkeypatch.setattr(plan_review_runtime, "plan_review_slots", lambda: slots)
    monkeypatch.setattr(plan_review, "_run_plan_review_slots", fake_run_slots)
    monkeypatch.setattr(plan_review, "get_review_enforcement", lambda: "advisory")

    workspace = tmp_path / "ws"
    workspace.mkdir()
    plan_path = tmp_path / "plan.md"
    plan_path.write_text("# Plan\n\nImplement the accepted phase.\n", encoding="utf-8")
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps({
        "in_scope": ["the accepted phase"], "acceptance_claims": ["tests green"],
        "invariants": ["no new settings"],
        # An operator envelope carries the same required form as the agent's (owner 9=A).
        "affected_paths": [],
    }), encoding="utf-8")
    evidence_file = workspace / "notes.txt"
    evidence_file.write_text("inspect the existing SSOT", encoding="utf-8")
    args = types.SimpleNamespace(
        plan=str(plan_path),
        goal="Test plan-review script",
        spec_json=str(spec_path),
        evidence=[str(evidence_file), "https://example.com/spec"],
        subject_root=str(workspace),
        drive_root=str(tmp_path / "drive"),
        output="",
    )

    output = asyncio.run(script._run(args))

    assert "RESOLVED PLAN REVIEW CONFIG" in output
    assert "PLAN REVIEW WAVE" in output and "PLAN REVIEW COORDINATED OUTPUT" in output
    assert captured["task_id"] == "plan-review-cli"
    assert captured["slots"] == ["slot_1"]
    assert captured["retry_key"].startswith("plan_review:")
    assert captured["reconcile_only"] is False
    assert captured["reconciliation_identity"]["epoch"] == captured["retry_key"]
    assert "Implement the accepted phase." in captured["user_content"]
    assert "tests green" in captured["user_content"]
    assert "inspect the existing SSOT" in captured["user_content"]
    assert "| https://example.com/spec | url_not_fetched |" in captured["user_content"]
    assert "before the work starts" in captured["system_prompt"]
    assert '"aggregate": "GREEN"' in output
    assert (tmp_path / "drive" / "task_results" / "plan-review-cli.json").exists()


def test_run_plan_review_script_has_no_personal_key_fallback():
    import pathlib
    repo = pathlib.Path(__file__).resolve().parents[1]
    text = (repo / "scripts" / "run_plan_review.py").read_text(encoding="utf-8")
    assert "file1.txt" not in text


def test_run_plan_review_script_has_only_the_new_envelope_flags():
    import pathlib
    repo = pathlib.Path(__file__).resolve().parents[1]
    text = (repo / "scripts" / "run_plan_review.py").read_text(encoding="utf-8")
    for flag in ("--goal", "--plan", "--spec-json", "--evidence"):
        assert flag in text
    for retired in ("--context-level", "--plan-class", "--files-to-touch", "--include-tests", "--scout-handoff"):
        assert retired not in text
