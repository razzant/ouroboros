"""Post-task reflection and backlog promotion in ``ouroboros.agent_task_pipeline``.

Split out of ``tests/test_agent_task_pipeline.py`` when that module was divided
by theme; every moved block is verbatim. Covers `_run_reflection` entry
generation, `_update_improvement_backlog`, and the project-scoped channel
split: project memory stays project-local while backlog promotion goes to the
global drive through `_run_global_backlog_promotion_only`.
"""

import json
from types import SimpleNamespace

import ouroboros.agent_task_pipeline as pipeline


def test_project_scoped_post_task_processing_feeds_global_backlog_but_project_memory(tmp_path, monkeypatch):
    import ouroboros.post_task_evolution as post_task_evolution

    calls = []
    reflection = {"backlog_candidates": [{"summary": "tool friction"}], "memory_actions": [{"kind": "note"}]}
    monkeypatch.setattr(pipeline, "_run_task_summary", lambda *args, **kwargs: calls.append(("summary",)))
    monkeypatch.setattr(pipeline, "_run_reflection", lambda *args, **kwargs: reflection)
    monkeypatch.setattr(pipeline, "_update_improvement_backlog", lambda _env, entry: calls.append(("backlog", entry)) or 1)
    monkeypatch.setattr(
        pipeline,
        "_apply_reflection_memory_actions",
        lambda _env, entry, project_id="": calls.append(("memory", project_id, entry)) or 1,
    )
    monkeypatch.setattr(post_task_evolution, "maybe_promote", lambda _env, task, entry, _llm: calls.append(("promote", task.get("project_id"), entry)))
    env = SimpleNamespace(repo_dir=tmp_path, drive_root=tmp_path, drive_path=lambda rel: tmp_path / rel)

    pipeline._run_post_task_processing_async(
        env,
        {"id": "task-1", "type": "task", "project_id": "proj-1", "text": "fix workspace"},
        {"rounds": 3, "cost": 0.1},
        {"tool_calls": [], "reasoning_notes": []},
        {},
        tmp_path / "logs",
        blocking=True,
    )

    assert ("backlog", reflection) in calls
    assert ("memory", "proj-1", reflection) in calls
    assert ("promote", "proj-1", reflection) in calls


def test_project_global_promotion_uses_real_maybe_promote_without_project_scope(tmp_path, monkeypatch):
    import ouroboros.post_task_evolution as post_task_evolution

    monkeypatch.setattr("ouroboros.config.get_post_task_evolution_enabled", lambda: True)
    monkeypatch.setattr("ouroboros.config.get_runtime_mode", lambda: "pro")
    monkeypatch.setattr("ouroboros.config.get_post_task_evolution_cadence", lambda: "every_n:1")
    monkeypatch.setattr(
        post_task_evolution,
        "_decide_promotion",
        lambda *_args, **_kwargs: {
            "promote": True,
            "objective": "Improve Ouroboros workspace tool feedback",
            "requires_plan_review": True,
            "backlog_id": "",
        },
    )
    env = SimpleNamespace(drive_root=tmp_path, drive_path=lambda rel: tmp_path / rel)
    reflection = {
        "reflection": "Project-specific detail should not be forwarded.",
        "memory_actions": [{"kind": "note"}],
        "backlog_candidates": [{"summary": "Improve Ouroboros workspace tool feedback"}],
    }

    pipeline._run_global_backlog_promotion_only(
        env,
        {
            "id": "task-project",
            "project_id": "proj-1",
            "workspace_root": "/tmp/project",
            "workspace_mode": "external",
            "metadata": {"workspace_preflight": {"git": {"head": "abc"}}},
        },
        reflection,
        object(),
    )

    req = json.loads((tmp_path / "state" / "post_task_evolution_request.json").read_text(encoding="utf-8"))
    assert req["objective"] == "Improve Ouroboros workspace tool feedback"
    backlog = (tmp_path / "memory" / "knowledge" / "improvement-backlog.md").read_text(encoding="utf-8")
    assert "Project-specific detail" not in backlog


def test_update_improvement_backlog_appends_candidates(tmp_path):
    env = SimpleNamespace(drive_root=tmp_path)

    added = pipeline._update_improvement_backlog(
        env,
        {
            "backlog_candidates": [{
                "summary": "Reduce recurring task friction around REVIEW_BLOCKED",
                "category": "process",
                "source": "execution_reflection",
                "task_id": "task-backlog",
                "evidence": "REVIEW_BLOCKED",
                "context": "The task retried blocked review loops without narrowing scope.",
                "proposed_next_step": "Run plan_task before touching review prompts again.",
            }],
        },
    )

    assert added == 1
    backlog_path = tmp_path / "memory" / "knowledge" / "improvement-backlog.md"
    assert backlog_path.exists()
    text = backlog_path.read_text(encoding="utf-8")
    assert "Reduce recurring task friction around REVIEW_BLOCKED" in text


def test_run_reflection_returns_entry_when_generated(tmp_path):
    captured = {}

    class FakeLlm:
        def chat(self, *, messages, model, reasoning_effort, max_tokens):
            captured["prompt"] = messages[0]["content"]
            return {
                "content": (
                    "Reflection text.\n"
                    "BACKLOG_CANDIDATES_JSON: "
                    "[{\"summary\":\"Reduce recurring task friction around REVIEW_BLOCKED\","
                    "\"category\":\"process\","
                    "\"source\":\"execution_reflection\","
                    "\"evidence\":\"REVIEW_BLOCKED\"}]"
                )
            }, {"cost": 0}

    env = SimpleNamespace(drive_root=tmp_path)
    (tmp_path / "logs").mkdir(parents=True)

    entry = pipeline._run_reflection(
        env,
        FakeLlm(),
        {"id": "task-reflect", "type": "task", "text": "Fix it"},
        {"rounds": 2, "cost": 0.01},
        {"tool_calls": [{"tool": "commit_reviewed", "is_error": False, "result": "⚠️ REVIEW_BLOCKED"}]},
        {"recent_attempts": [], "open_obligations": [{"item": "tests_affected", "reason": "Fix the failing test before commit"}]},
    )

    assert entry is not None
    assert entry["task_id"] == "task-reflect"
    assert entry["reflection"] == "Reflection text."
    assert len(entry["backlog_candidates"]) == 1
    assert entry["backlog_candidates"][0]["summary"] == "Reduce recurring task friction around REVIEW_BLOCKED"
