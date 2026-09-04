"""Post-task cognition yields to the task's wall deadline.

CyberGym r8 (2026-09-04): tasks whose agent loop finished inside the 2 h
deadline spent 5-16 min in consolidation/summary/reflection under a 64-lane
load, crossed ``deadline_at``, and were killed by the reaper mid-reflection —
completed results were lost (lifecycle fault) or born cost-non-final (orphaned
post_task_synthesis ledger row), and the launcher's cancellation custody timed
out on solved tasks.
"""

from __future__ import annotations

import datetime as _dt
import json
from types import SimpleNamespace

import ouroboros.agent_task_pipeline as pipeline
from ouroboros.agent_task_pipeline import post_task_cognition_deadline_skip


def _iso(seconds_from_now: float) -> str:
    return (_dt.datetime.now(_dt.timezone.utc) + _dt.timedelta(seconds=seconds_from_now)).isoformat()


def test_no_deadline_never_skips(monkeypatch):
    monkeypatch.delenv("OUROBOROS_POST_TASK_COGNITION_MIN_REMAINING_SEC", raising=False)
    assert post_task_cognition_deadline_skip({"id": "t"}) is None
    assert post_task_cognition_deadline_skip({"id": "t", "deadline_at": "garbage"}) is None


def test_far_deadline_runs_and_near_deadline_skips(monkeypatch):
    monkeypatch.delenv("OUROBOROS_POST_TASK_COGNITION_MIN_REMAINING_SEC", raising=False)
    assert post_task_cognition_deadline_skip({"id": "t", "deadline_at": _iso(3600)}) is None
    skip = post_task_cognition_deadline_skip({"id": "t", "deadline_at": _iso(600)})
    assert skip is not None
    assert skip["reason"] == "deadline_near"
    assert skip["min_remaining_sec"] == 900.0
    assert 0 <= skip["remaining_sec"] <= 600
    # Already past the deadline: remaining is floored at 0 and the skip holds.
    assert post_task_cognition_deadline_skip({"id": "t", "deadline_at": _iso(-30)}) is not None


def test_deadline_in_task_metadata_is_honored_and_env_tunes_floor(monkeypatch):
    task = {"id": "t", "metadata": {"deadline_at": _iso(1200)}}
    monkeypatch.setenv("OUROBOROS_POST_TASK_COGNITION_MIN_REMAINING_SEC", "1800")
    assert post_task_cognition_deadline_skip(task) is not None
    monkeypatch.setenv("OUROBOROS_POST_TASK_COGNITION_MIN_REMAINING_SEC", "600")
    assert post_task_cognition_deadline_skip(task) is None
    monkeypatch.setenv("OUROBOROS_POST_TASK_COGNITION_MIN_REMAINING_SEC", "0")
    assert post_task_cognition_deadline_skip({"id": "t", "deadline_at": _iso(-30)}) is None


def test_runner_skips_every_paid_step_and_records_the_reason(tmp_path, monkeypatch):
    import ouroboros.llm as llm_module
    import ouroboros.memory as memory_module

    monkeypatch.delenv("OUROBOROS_POST_TASK_COGNITION_MIN_REMAINING_SEC", raising=False)
    calls: list[str] = []
    checkpoints: list[str] = []
    monkeypatch.setattr(pipeline, "_is_root_post_task", lambda task: True)
    monkeypatch.setattr(pipeline, "_root_post_task_already_completed", lambda env, task: False)
    monkeypatch.setattr(
        pipeline, "_set_root_post_task_checkpoint",
        lambda env, task, status, **kw: checkpoints.append(status),
    )
    monkeypatch.setattr(llm_module, "LLMClient", lambda: calls.append("llm_client"))
    monkeypatch.setattr(memory_module, "Memory", lambda **kwargs: calls.append("memory"))
    for name in ("_run_chat_consolidation", "_run_scratchpad_consolidation", "_run_task_summary"):
        monkeypatch.setattr(pipeline, name, lambda *a, _n=name, **k: calls.append(_n))
    monkeypatch.setattr(pipeline, "_run_reflection", lambda *a, **k: calls.append("reflection") or {})

    env = SimpleNamespace(drive_root=tmp_path, repo_dir=tmp_path, drive_path=lambda rel: tmp_path / rel)
    logs = tmp_path / "logs"
    logs.mkdir()
    task = {"id": "cybergym-x", "workspace_root": str(tmp_path), "deadline_at": _iso(120)}

    result = pipeline._run_post_task_processing_async(
        env, task, {"rounds": 3, "cost": 1.0}, {"tool_calls": []}, {}, logs, blocking=True,
    )

    assert result is None
    assert calls == []  # no model, no memory, no paid step
    assert checkpoints == ["degraded"]  # terminal at once: the frame can be born final
    rows = [json.loads(line) for line in (logs / "events.jsonl").read_text().splitlines()]
    assert [row["type"] for row in rows] == ["post_task_cognition_skipped"]
    assert rows[0]["task_id"] == "cybergym-x" and rows[0]["reason"] == "deadline_near"


def test_runner_still_runs_when_deadline_is_far(tmp_path, monkeypatch):
    import ouroboros.llm as llm_module
    import ouroboros.memory as memory_module

    monkeypatch.delenv("OUROBOROS_POST_TASK_COGNITION_MIN_REMAINING_SEC", raising=False)
    calls: list[str] = []
    monkeypatch.setattr(pipeline, "_is_root_post_task", lambda task: False)
    monkeypatch.setattr(llm_module, "LLMClient", lambda: object())
    monkeypatch.setattr(memory_module, "Memory", lambda **kwargs: object())
    for name in ("_run_chat_consolidation", "_run_scratchpad_consolidation", "_run_task_summary"):
        monkeypatch.setattr(pipeline, name, lambda *a, _n=name, **k: calls.append(_n))
    monkeypatch.setattr(pipeline, "_run_reflection", lambda *a, **k: calls.append("reflection") or {"ok": 1})
    monkeypatch.setattr(pipeline, "_update_improvement_backlog", lambda *a, **k: None)
    monkeypatch.setattr(pipeline, "_apply_reflection_memory_actions", lambda *a, **k: None)

    env = SimpleNamespace(drive_root=tmp_path, repo_dir=tmp_path, drive_path=lambda rel: tmp_path / rel)
    task = {"id": "cybergym-y", "workspace_root": str(tmp_path), "deadline_at": _iso(5400)}

    result = pipeline._run_post_task_processing_async(
        env, task, {"rounds": 3, "cost": 1.0}, {"tool_calls": []}, {}, tmp_path / "logs", blocking=True,
    )

    assert result == {"ok": 1}
    assert calls == ["_run_chat_consolidation", "_run_scratchpad_consolidation", "_run_task_summary", "reflection"]
