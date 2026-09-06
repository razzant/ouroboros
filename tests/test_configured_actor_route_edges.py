"""Small route/restart edge band for configured recursive actors."""

from __future__ import annotations

import json


def _settings(*rows):
    return {
        "OUROBOROS_SUBAGENTS": json.dumps({"enabled": True, "items": list(rows)}),
    }


def _api_row(row_id="api-builder", model="openai/gpt-5.6-sol", effort="high"):
    return {
        "subagent_id": row_id,
        "name": "API builder",
        "recommended_use": "Exact recursive API actor.",
        "route": {"kind": "api_model", "target_id": model},
        "effort": effort,
    }


def test_planned_restart_kill_keeps_selected_child_even_if_parent_is_interrupted(
    monkeypatch, tmp_path,
):
    from supervisor import queue as task_queue
    from supervisor import workers

    repo = tmp_path / "repo"
    repo.mkdir()
    workers.init(repo, tmp_path, 2)
    workers.WORKERS.clear()
    workers.PENDING.clear()
    workers.RUNNING.clear()
    workers.RUNNING.update({
        "parent": {"task": {"id": "parent", "root_task_id": "parent"}, "attempt": 1},
        "child": {"task": {
            "id": "child", "parent_task_id": "parent", "root_task_id": "parent",
        }, "attempt": 1},
    })
    monkeypatch.setattr(workers, "_write_failure_result", lambda *_a, **_k: "cancelled")
    monkeypatch.setattr(workers, "_emit_task_done_terminal", lambda *_a, **_k: True)
    monkeypatch.setattr(task_queue, "persist_queue_snapshot", lambda *a, **k: True)
    workers.kill_workers(
        terminal_status="cancelled", preserve_pending=True,
        preserve_running_task_ids={"child"},
    )
    assert [task["id"] for task in workers.PENDING] == ["child"]
    assert workers.PENDING[0]["_attempt"] == 2
    workers.PENDING.clear()
    workers.RUNNING.clear()


def test_api_row_is_refused_by_root_direct_exact_start_before_daemon(monkeypatch, tmp_path):
    import ouroboros.claudexor_daemon as daemon
    import ouroboros.tools.delegate as delegate
    from ouroboros.tools.registry import ToolContext

    settings = _settings(_api_row())
    monkeypatch.setattr("ouroboros.config.load_settings", lambda: settings)
    monkeypatch.setattr(daemon, "ensure_owned_gateway", lambda: (_ for _ in ()).throw(
        AssertionError("API rows never POST to Claudexor")))
    ctx = ToolContext(repo_dir=tmp_path, drive_root=tmp_path)
    ctx.task_id = "root1"
    schema = next(e.schema for e in delegate.get_tools() if e.name == "delegate_start")["parameters"]
    assert schema["required"] == ["prompt"] and not ({"anyOf", "oneOf", "allOf"} & schema.keys())
    missing = json.loads(delegate.exact_start(ctx, "bounded leaf"))
    assert missing["reason"] == "subagent_selection_required"
    out = json.loads(delegate.exact_start(
        ctx, "bounded leaf", {"subagent_id": "api-builder"},
    ))
    assert out["reason"] == "api_actor_requires_schedule_subagent"
    retry = json.loads(delegate.exact_start(
        ctx, "bounded leaf", {"subagent_id": "api-builder", "retry_of": "inv-old"},
    ))
    assert retry["reason"] == "retry_selector_conflict"


def test_heavy_is_absent_from_active_runtime_and_vision_consumers(monkeypatch):
    from ouroboros.llm import LLMClient
    from ouroboros.tools.vision import _vision_capable_slot_candidates

    monkeypatch.setenv("OUROBOROS_MODEL", "openai/main")
    monkeypatch.setenv("OUROBOROS_MODEL_LIGHT", "openai/light")
    monkeypatch.setenv("OUROBOROS_MODEL_HEAVY", "openai/legacy-heavy")
    assert "openai/legacy-heavy" not in LLMClient().available_models()
    assert "openai/legacy-heavy" not in _vision_capable_slot_candidates(LLMClient())
