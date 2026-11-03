"""The operator script drives the SAME planning organ ``plan_task`` uses — one revision
(goal, plan prose, done_when, affected paths) and one ask to the configured council — in an
isolated drive root; ``--dry-run`` assembles the deliveries and dispatches nothing."""
from __future__ import annotations

import importlib.util
import json
import types

from tests._store_shared import result_exists


def _load_script_module(repo_root):
    path = repo_root / "scripts" / "run_plan_review.py"
    spec = importlib.util.spec_from_file_location("run_plan_review_script", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _slots():
    from ouroboros.review_execution import ReviewRouteKind
    from ouroboros.review_substrate import ReviewSlot
    from ouroboros.tools.plan_review_runtime import PLAN_REVIEW_MAX_TOKENS

    return [ReviewSlot(slot_id="slot_1", model="fake/reviewer", effort="high", role_hint="plan reviewer",
                       route=ReviewRouteKind.API_CHAT, max_tokens=PLAN_REVIEW_MAX_TOKENS, default_temperature=0.2,
                       native_retrieval_override=False)]


def _args(tmp_path, workspace, plan_path, done_when_path, evidence_file, **over):
    base = dict(plan=str(plan_path), goal="Test the planning organ script", done_when_json=str(done_when_path),
                affected_path=[], attach=[str(evidence_file), "https://example.com/spec"], ask=[], message="First look.",
                subject_root=str(workspace), drive_root=str(tmp_path / "drive"), task_id="", dry_run=False, output="")
    base.update(over)
    return types.SimpleNamespace(**base)


def _fixture(tmp_path, monkeypatch):
    import pathlib

    from ouroboros.tools import plan_ask

    repo = pathlib.Path(__file__).resolve().parents[1]
    script = _load_script_module(repo)
    monkeypatch.setattr(plan_ask, "plan_review_slots", lambda default_effort="": _slots())
    monkeypatch.setattr("ouroboros.tools.plan_dialogue.session_input_limits", lambda _slots: {})
    monkeypatch.setenv("OUROBOROS_REVIEW_ENFORCEMENT", "advisory")
    workspace = tmp_path / "ws"
    workspace.mkdir()
    plan_path = tmp_path / "plan.md"
    plan_path.write_text("# Plan\n\nImplement the accepted phase.\n", encoding="utf-8")
    done_when_path = tmp_path / "done_when.json"
    done_when_path.write_text(json.dumps([{"id": "c1", "claim": "tests green", "support": "run the suite"}]), encoding="utf-8")
    evidence_file = workspace / "notes.txt"
    evidence_file.write_text("inspect the existing SSOT", encoding="utf-8")
    return script, workspace, plan_path, done_when_path, evidence_file


def test_run_plan_review_script_writes_a_revision_and_asks_the_council(monkeypatch, tmp_path):
    script, workspace, plan_path, done_when_path, evidence_file = _fixture(tmp_path, monkeypatch)
    captured = {}

    async def fake_run_slots(ctx, slots, *, system_prompt, user_content, session_task="", session_root="", output_contract="",
                             slot_messages=None, slot_session_tasks=None, request_policy=None, session_threads=None,
                             retry_key="", reconciliation_identity=None, release_at_dispatch=False):
        captured.update(task_id=ctx.task_id, slots=[s.slot_id for s in slots], system_prompt=system_prompt,
                        user_content=user_content, retry_key=retry_key,
                        reconciliation_identity=reconciliation_identity, release_at_dispatch=release_at_dispatch)
        return [{"slot_id": s.slot_id, "model": s.model, "request_model": s.model, "route": "api_chat",
                 "host_file_read_attestation": "host_assembled_packet",
                 "text": json.dumps({"prose": "Sound.", "objections": [], "withdrawn": []}), "error": None,
                 "prompt_ref": {}, "response_ref": {}, "tokens_in": 1, "tokens_out": 1, "cost": 0.0,
                 "operation_id": "op-1", "operation_state": "settled"} for s in slots]

    monkeypatch.setattr("ouroboros.tools.plan_ask.run_plan_review_slots", fake_run_slots)

    output = script._run(_args(tmp_path, workspace, plan_path, done_when_path, evidence_file))

    assert "RESOLVED PLAN CONFIG" in output and "PLAN RECORD" in output and "PLAN TASK OUTPUT" in output
    assert captured["task_id"] == "plan-review-cli" and captured["slots"] == ["slot_1"]
    assert captured["retry_key"] == "plan_ask:plan-review-cli:a1"  # an ask is a send: the runtime has no $0 collection
    assert captured["release_at_dispatch"] is True
    assert captured["reconciliation_identity"]["plan_ask_id"] == "a1"
    assert "Implement the accepted phase." in captured["user_content"] and "tests green" in captured["user_content"]
    assert "inspect the existing SSOT" in captured["user_content"]
    assert "| https://example.com/spec | url_not_fetched |" in captured["user_content"]
    assert "weighs an INTENTION" in captured["system_prompt"]
    assert '"status": "answered"' in output and "Revision 1 recorded" in output
    assert result_exists(tmp_path / "drive", "plan-review-cli")


def test_run_plan_review_script_dry_run_assembles_and_dispatches_nothing(monkeypatch, tmp_path):
    import pytest

    script, workspace, plan_path, done_when_path, evidence_file = _fixture(tmp_path, monkeypatch)
    monkeypatch.setattr("ouroboros.tools.plan_ask.run_plan_review_slots",
                        lambda *_a, **_k: pytest.fail("a dry run dispatches nobody"))

    output = script._run(_args(tmp_path, workspace, plan_path, done_when_path, evidence_file, dry_run=True))

    assert "PLAN ASK DRY RUN" in output and '"seat_id": "slot_1"' in output
    assert "Implement the accepted phase." in output and "tests green" in output
    assert not result_exists(tmp_path / "drive", "plan-review-cli")


def test_run_plan_review_script_has_no_personal_key_fallback():
    import pathlib

    repo = pathlib.Path(__file__).resolve().parents[1]
    text = (repo / "scripts" / "run_plan_review.py").read_text(encoding="utf-8")
    assert "file1.txt" not in text


def test_run_plan_review_script_has_only_the_new_form_flags(monkeypatch, capsys):
    import pathlib
    import sys

    import pytest

    repo = pathlib.Path(__file__).resolve().parents[1]
    script = _load_script_module(repo)
    received = []
    monkeypatch.setattr(script, "_load_settings_into_env", lambda: None)
    monkeypatch.setattr(script, "_run", lambda args: received.append(vars(args)) or "parsed")
    monkeypatch.setattr(sys, "argv", [
        "run_plan_review.py", "--goal", "goal", "--plan", "plan.md",
        "--done-when-json", "outcomes.json", "--affected-path", "a.py", "--affected-path", "b.py",
        "--attach", "notes.txt", "--attach", "task:evidence", "--ask", "council",
        "--message", "review this", "--dry-run",
    ])

    assert script.main() == 0
    assert capsys.readouterr().out == "parsed\n"
    [args] = received
    assert {key: args[key] for key in (
        "goal", "plan", "done_when_json", "affected_path", "attach", "ask", "message", "dry_run",
    )} == {
        "goal": "goal", "plan": "plan.md", "done_when_json": "outcomes.json",
        "affected_path": ["a.py", "b.py"], "attach": ["notes.txt", "task:evidence"],
        "ask": ["council"], "message": "review this", "dry_run": True,
    }

    for retired in ("--spec-json", "--evidence", "--context-level", "--plan-class", "--files-to-touch", "--include-tests", "--scout-handoff"):
        monkeypatch.setattr(sys, "argv", ["run_plan_review.py", "--plan", "plan.md", retired, "fixture"])
        with pytest.raises(SystemExit) as exc:
            script.main()
        assert exc.value.code == 2
        assert f"unrecognized arguments: {retired}" in capsys.readouterr().err
    assert len(received) == 1, "invalid CLI arguments must not dispatch a review"
