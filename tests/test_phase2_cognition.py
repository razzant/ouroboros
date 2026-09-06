from __future__ import annotations

import json
import pathlib
from types import SimpleNamespace


def _chat_rows(root):
    path = root / "logs" / "chat.jsonl"
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_split_project_root_summary_lands_canonically_once_before_child_gc(tmp_path):
    from ouroboros import agent_task_pipeline as pipeline

    canonical = tmp_path / "canonical"
    child = tmp_path / "child"
    (canonical / "logs").mkdir(parents=True)
    (child / "logs").mkdir(parents=True)
    env = SimpleNamespace(repo_dir=tmp_path / "repo", drive_root=child)
    task = {
        "id": "project-root",
        "root_task_id": "project-root",
        "project_id": "launch",
        "chat_id": 41,
        "type": "task",
        "text": "Ship the release",
        "budget_drive_root": str(canonical),
    }

    pipeline._run_task_summary(
        env, object(), task, {"rounds": 1, "cost": 0},
        {"tool_calls": []}, child / "logs",
    )

    rows = [row for row in _chat_rows(canonical) if row.get("type") == "task_summary"]
    assert len(rows) == 1
    assert rows[0]["task_id"] == "project-root"
    assert rows[0]["project_id"] == "launch"
    assert rows[0]["result_ref"] == {
        "kind": "task_result", "task_id": "project-root", "reader": "get_task_result",
    }

    # The execution drive is disposable; canonical biography is not.
    import shutil

    shutil.rmtree(child)
    assert len([row for row in _chat_rows(canonical) if row.get("task_id") == "project-root"]) == 1


def test_root_checkpoint_prevents_a_second_paid_authored_summary(tmp_path, monkeypatch):
    from ouroboros import agent_task_pipeline as pipeline
    from ouroboros.task_results import STATUS_COMPLETED, write_task_result

    class Llm:
        def __init__(self):
            self.calls = 0

        def chat(self, **_kwargs):
            self.calls += 1
            return {"content": "Authored once"}, {"cost": 0}

    import ouroboros.llm as llm_mod
    import ouroboros.memory as memory_mod
    import ouroboros.post_task_evolution as evolution

    llm = Llm()
    monkeypatch.setattr(llm_mod, "LLMClient", lambda: llm)
    monkeypatch.setattr(memory_mod, "Memory", lambda **_kwargs: object())
    monkeypatch.setattr(pipeline, "_run_chat_consolidation", lambda *_a, **_k: None)
    monkeypatch.setattr(pipeline, "_run_scratchpad_consolidation", lambda *_a, **_k: None)
    monkeypatch.setattr(pipeline, "_run_reflection", lambda *_a, **_k: None)
    monkeypatch.setattr(pipeline, "_update_improvement_backlog", lambda *_a, **_k: 0)
    monkeypatch.setattr(pipeline, "_apply_reflection_memory_actions", lambda *_a, **_k: 0)
    monkeypatch.setattr(evolution, "maybe_promote", lambda *_a, **_k: None)
    env = SimpleNamespace(
        repo_dir=tmp_path / "repo", drive_root=tmp_path,
        drive_path=lambda rel: tmp_path / rel,
    )
    task = {"id": "root-llm", "root_task_id": "root-llm", "chat_id": 1,
            "type": "task", "text": "Nontrivial task"}
    write_task_result(
        tmp_path, "root-llm", STATUS_COMPLETED,
        root_task_id="root-llm",
        root_phase_checkpoint={"post_task_synthesis": "pending_once"},
    )
    args = (
        env, task, {"rounds": 2, "cost": 0},
        {"tool_calls": [{"tool": "read_file"}]}, {}, tmp_path / "logs",
    )
    pipeline._run_post_task_processing_async(*args, blocking=True)
    # Raw dialogue may rotate/compact after the checkpoint settles. Its absence
    # must not authorize another paid synthesis.
    archive = tmp_path / "archive"
    archive.mkdir(exist_ok=True)
    (tmp_path / "logs" / "chat.jsonl").replace(archive / "chat_rotated.jsonl")
    pipeline._run_post_task_processing_async(*args, blocking=True)

    assert llm.calls == 1
    assert not (tmp_path / "logs" / "chat.jsonl").exists()
    assert "root-llm" in (archive / "chat_rotated.jsonl").read_text(encoding="utf-8")


def test_authored_summary_hot_path_never_scans_rotated_biography(tmp_path, monkeypatch):
    from ouroboros import agent_task_pipeline as pipeline
    import ouroboros.project_dialogue as dialogue

    archive = tmp_path / "archive"
    archive.mkdir()
    for index in range(206):
        (archive / f"chat_{index:04d}.jsonl").write_text(
            json.dumps({"type": "task_summary", "task_id": f"old-{index}"}) + "\n",
            encoding="utf-8",
        )
    scans = 0

    def forbidden_scan(*_args, **_kwargs):
        nonlocal scans
        scans += 1
        return iter(())

    monkeypatch.setattr(dialogue, "iter_jsonl_objects", forbidden_scan)

    class Llm:
        calls = 0

        def chat(self, **_kwargs):
            self.calls += 1
            return {"content": "One paid narrative"}, {"cost": 0}

    llm = Llm()
    pipeline._run_task_summary(
        SimpleNamespace(drive_root=tmp_path), llm,
        {"id": "new-root", "root_task_id": "new-root", "type": "task", "text": "work"},
        {"rounds": 2, "cost": 0}, {"tool_calls": [{"tool": "read_file"}]},
        tmp_path / "logs",
    )

    assert llm.calls == 1
    assert scans == 0


def test_missing_role_terminal_root_is_never_labeled_child(tmp_path):
    from ouroboros.project_dialogue import append_terminal_task_projection
    from ouroboros.task_results import STATUS_FAILED, write_task_result

    result = write_task_result(
        tmp_path, "roleless-root", STATUS_FAILED,
        root_task_id="roleless-root", result="failed",
    )
    assert append_terminal_task_projection(
        tmp_path, "roleless-root", {}, result,
        {"status": "failed", "chat_id": 1},
    )
    row = next(row for row in _chat_rows(tmp_path) if row.get("task_id") == "roleless-root")
    assert row["summary_kind"] == "terminal_root_projection"
    assert row["role"] == "root"
    assert "role=root" in row["text"]


def test_terminal_child_projection_is_idempotent_and_honest_for_all_outcomes(tmp_path):
    from ouroboros.project_dialogue import OUTCOME_PHASE_HEADLINE, append_terminal_task_projection

    cases = [
        ("ok", "completed", {"execution": {"status": "ok"}}, "Done"),
        ("failed", "failed", {"execution": {"status": "failed"}}, "Failed"),
        ("cancelled", "cancelled", {"execution": {"status": "ok"}}, "Cancelled"),
        ("degraded", "completed", {"execution": {"status": "best_effort"}}, "Done with warnings"),
    ]
    for suffix, status, axes, label in cases:
        task_id = f"child-{suffix}"
        task = {
            "id": task_id,
            "parent_task_id": "project-root",
            "root_task_id": "project-root",
            "project_id": "launch",
            "chat_id": 41,
            "delegation_role": "subagent",
            "role": "reviewer",
        }
        result = {
            **task, "task_id": task_id, "status": status,
            "outcome_axes": axes, "reason_code": f"reason-{suffix}",
            "result": f"result-{suffix}",
        }
        done = {"chat_id": 41, "status": status, "outcome_axes": axes}
        assert append_terminal_task_projection(tmp_path, task_id, task, result, done)
        assert not append_terminal_task_projection(tmp_path, task_id, task, result, done)

        row = next(row for row in _chat_rows(tmp_path) if row.get("task_id") == task_id)
        assert row["parent_task_id"] == "project-root"
        assert row["root_task_id"] == "project-root"
        assert row["project_id"] == "launch"
        assert row["role"] == "reviewer"
        assert row["status"] == status
        assert row["outcome"] == label
        assert OUTCOME_PHASE_HEADLINE[row["outcome_phase"]] == label
        assert row["reason_code"] == f"reason-{suffix}"
        assert row["result_ref"] == {
            "kind": "task_result", "task_id": task_id, "reader": "get_task_result",
        }
        assert f'get_task_result(task_id="{task_id}")' in row["text"]


def test_terminal_projection_dedup_does_not_lose_concurrent_chat_append(tmp_path):
    from concurrent.futures import ThreadPoolExecutor

    from ouroboros.project_dialogue import append_terminal_task_projection
    from ouroboros.task_results import STATUS_COMPLETED, write_task_result
    from ouroboros.utils import append_jsonl

    result = write_task_result(
        tmp_path, "concurrent-child", STATUS_COMPLETED,
        parent_task_id="root", root_task_id="root", delegation_role="subagent",
        result="done", outcome_axes={"execution": {"status": "ok"}},
    )
    task = {
        "id": "concurrent-child", "parent_task_id": "root", "root_task_id": "root",
        "delegation_role": "subagent", "chat_id": 1,
    }

    def project(_index):
        return append_terminal_task_projection(
            tmp_path, "concurrent-child", task, result,
            {"chat_id": 1, "status": "completed"},
        )

    def noise(index):
        return append_jsonl(
            tmp_path / "logs" / "chat.jsonl",
            {"type": "concurrent_noise", "index": index},
        )

    with ThreadPoolExecutor(max_workers=12) as pool:
        projection_results = list(pool.map(project, range(12)))
        noise_results = list(pool.map(noise, range(24)))

    rows = _chat_rows(tmp_path)
    assert projection_results.count(True) == 1
    assert all(noise_results)
    assert len([row for row in rows if row.get("summary_id") == "task-terminal:concurrent-child"]) == 1
    assert {row["index"] for row in rows if row.get("type") == "concurrent_noise"} == set(range(24))


def test_terminal_root_fallback_covers_cancel_without_preempting_open_synthesis(tmp_path):
    from types import SimpleNamespace

    from ouroboros.post_task_checkpoint import set_root_post_task_checkpoint
    from ouroboros.project_dialogue import append_terminal_task_projection
    from ouroboros.task_results import (
        STATUS_CANCELLED,
        STATUS_COMPLETED,
        load_task_result,
        write_task_result,
    )

    cancelled = write_task_result(
        tmp_path, "cancelled-root", STATUS_CANCELLED,
        root_task_id="cancelled-root", project_id="launch", result="Stopped by owner",
    )
    assert append_terminal_task_projection(
        tmp_path, "cancelled-root", {}, cancelled,
        {"chat_id": 41, "status": "cancelled"},
    )
    row = next(row for row in _chat_rows(tmp_path) if row.get("task_id") == "cancelled-root")
    assert row["summary_kind"] == "terminal_root_projection"
    assert row["outcome"] == "Cancelled"

    pending = write_task_result(
        tmp_path, "normal-root", STATUS_COMPLETED,
        root_task_id="normal-root",
        root_phase_checkpoint={"post_task_synthesis": "running"},
    )
    assert not append_terminal_task_projection(
        tmp_path, "normal-root", {}, pending,
        {"chat_id": 1, "status": "completed"},
    )
    assert not any(row.get("task_id") == "normal-root" for row in _chat_rows(tmp_path))
    assert load_task_result(tmp_path, "normal-root")["canonical_terminal_projection_ready"]["summary_id"] == "task-terminal:normal-root"
    set_root_post_task_checkpoint(
        SimpleNamespace(drive_root=tmp_path),
        {"id": "normal-root", "root_task_id": "normal-root", "chat_id": 1},
        "completed",
    )
    normal = next(row for row in _chat_rows(tmp_path) if row.get("task_id") == "normal-root")
    assert normal["summary_kind"] == "terminal_root_projection"
    assert normal["outcome"] == "Done"
    assert normal["outcome_final"] is True


def test_running_async_root_truth_survives_restart_degradation_once(tmp_path):
    from ouroboros.agent_task_pipeline import recover_pending_root_post_task_synthesis
    from ouroboros.project_dialogue import (
        append_terminal_task_projection,
        completion_status_label,
    )
    from ouroboros.task_results import STATUS_COMPLETED, load_task_result, write_task_result

    running = write_task_result(
        tmp_path, "restart-root", STATUS_COMPLETED,
        root_task_id="restart-root", result="Owner already received the answer",
        root_phase_checkpoint={"post_task_synthesis": "running"},
    )
    assert not append_terminal_task_projection(
        tmp_path, "restart-root", {}, running,
        {"chat_id": 1, "status": "completed"},
    )
    assert not any(row.get("task_id") == "restart-root" for row in _chat_rows(tmp_path))
    assert recover_pending_root_post_task_synthesis(tmp_path, repo_dir=tmp_path / "repo") == 1

    stored = load_task_result(tmp_path, "restart-root")
    assert stored["root_phase_checkpoint"]["post_task_synthesis"] == "degraded"
    rows = [row for row in _chat_rows(tmp_path) if row.get("task_id") == "restart-root"]
    assert len(rows) == 1
    assert rows[0]["outcome"] == "Done"
    assert rows[0]["outcome"] == completion_status_label(stored, {})
    assert rows[0]["outcome_final"] is True
    assert recover_pending_root_post_task_synthesis(tmp_path, repo_dir=tmp_path / "repo") == 0
    assert len([row for row in _chat_rows(tmp_path) if row.get("task_id") == "restart-root"]) == 1


def test_authored_narrative_never_suppresses_final_artifact_failure_truth(tmp_path):
    from ouroboros import agent_task_pipeline as pipeline
    from ouroboros.project_dialogue import append_terminal_task_projection
    from ouroboros.task_results import STATUS_COMPLETED, write_task_result

    initial = write_task_result(
        tmp_path, "artifact-root", STATUS_COMPLETED,
        root_task_id="artifact-root", result="Built output",
        outcome_axes={"execution": {"status": "ok"}, "artifacts": {"status": "ready"}},
    )
    pipeline._run_task_summary(
        SimpleNamespace(drive_root=tmp_path), object(),
        {"id": "artifact-root", "root_task_id": "artifact-root", "text": "build",
         "type": "task", "chat_id": 1},
        {"rounds": 1, "cost": 0, "outcome_axes": initial["outcome_axes"]},
        {"tool_calls": []}, tmp_path / "logs",
    )
    final = write_task_result(
        tmp_path, "artifact-root", STATUS_COMPLETED,
        artifact_status="failed", artifact_error="manifest copy failed",
        outcome_axes={"execution": {"status": "ok"}, "artifacts": {"status": "failed"}},
    )
    assert append_terminal_task_projection(
        tmp_path, "artifact-root", {}, final,
        {"chat_id": 1, "status": "completed", "artifact_status": "failed",
         "outcome_axes": initial["outcome_axes"]},
    )

    rows = [row for row in _chat_rows(tmp_path) if row.get("task_id") == "artifact-root"]
    assert [row["summary_kind"] for row in rows] == [
        "authored_root_summary", "terminal_root_projection",
    ]
    assert rows[-1]["outcome"] == "Failed"
    assert rows[-1]["outcome_axes"]["artifacts"]["status"] == "failed"
    assert rows[-1]["outcome_final"] is True


def test_split_authored_narrative_keeps_only_canonical_result_ref_after_child_gc(tmp_path):
    import shutil

    from ouroboros import agent_task_pipeline as pipeline
    from ouroboros.headless import copy_child_task_result
    from ouroboros.task_results import STATUS_COMPLETED, load_task_result, write_task_result

    canonical = tmp_path / "canonical"
    child = tmp_path / "child"
    dangling = child / "task_results" / "artifacts" / "split-root" / "bundle.txt"
    dangling.parent.mkdir(parents=True)
    dangling.write_text("artifact", encoding="utf-8")
    stored = write_task_result(
        child, "split-root", STATUS_COMPLETED,
        root_task_id="split-root", project_id="launch", result="done",
        artifacts=[{"path": str(dangling), "name": "bundle.txt"}],
        artifact_bundle={"status": "ready", "artifacts": [{"path": str(dangling)}]},
    )
    task = {
        "id": "split-root", "root_task_id": "split-root", "project_id": "launch",
        "chat_id": 41, "type": "task", "text": "build",
        "budget_drive_root": str(canonical),
        "drive_root": str(child),
    }
    pipeline._run_task_summary(
        SimpleNamespace(drive_root=child), object(), task,
        {"rounds": 1, "cost": 0}, {"tool_calls": []}, child / "logs",
    )
    copied = copy_child_task_result(canonical, task)
    assert copied and load_task_result(canonical, "split-root")
    shutil.rmtree(child)

    row = next(row for row in _chat_rows(canonical) if row.get("task_id") == "split-root")
    assert row["result_ref"] == {
        "kind": "task_result", "task_id": "split-root", "reader": "get_task_result",
    }
    assert "artifact_bundle" not in row
    assert str(dangling) not in json.dumps(row, ensure_ascii=False)
    assert stored["artifact_bundle"]["artifacts"][0]["path"] == str(dangling)
    canonical_result = load_task_result(canonical, "split-root")
    assert canonical_result["artifacts"][0]["path"] != str(dangling)
    assert pathlib.Path(canonical_result["artifacts"][0]["path"]).is_file()


def test_duplicate_task_done_after_child_copyback_appends_one_canonical_projection(tmp_path):
    from ouroboros.task_results import STATUS_COMPLETED, load_task_result, write_task_result
    from supervisor import events

    child_drive = tmp_path / "state" / "headless_tasks" / "child-copy" / "data"
    write_task_result(
        child_drive, "child-copy", STATUS_COMPLETED,
        result="Integrated the reviewed change",
        parent_task_id="parent-root", root_task_id="parent-root",
        project_id="launch", delegation_role="subagent", role="implementer",
        outcome_axes={"execution": {"status": "ok"}},
    )
    task = {
        "id": "child-copy", "drive_root": str(child_drive), "chat_id": 41,
        "parent_task_id": "parent-root", "root_task_id": "parent-root",
        "project_id": "launch", "delegation_role": "subagent", "role": "implementer",
        "task_constraint": {"mode": "local_readonly_subagent"},
    }
    worker = SimpleNamespace(busy_task_id="child-copy", reaping=False)
    ctx = SimpleNamespace(
        DRIVE_ROOT=tmp_path,
        RUNNING={"child-copy": {"task": task}}, WORKERS={7: worker},
        bridge=SimpleNamespace(push_log=lambda _row: None),
        send_with_budget=lambda *_args, **_kwargs: None,
        persist_queue_snapshot=lambda **_kwargs: None,
    )
    event = {"task_id": "child-copy", "worker_id": 7, "task_type": "task",
             "chat_id": 41, "status": "completed"}

    events._handle_task_done(event, ctx)
    events._handle_task_done(event, ctx)

    rows = [row for row in _chat_rows(tmp_path) if row.get("task_id") == "child-copy"]
    assert len(rows) == 1
    assert rows[0]["result_ref"]["reader"] == "get_task_result"
    assert load_task_result(tmp_path, "child-copy")["canonical_terminal_projection"]["summary_id"] == "task-terminal:child-copy"
    assert not child_drive.joinpath("task_results", "child-copy.json").samefile(
        tmp_path / "task_results" / "child-copy.json"
    )


def test_child_projection_enters_main_cognition_and_project_lineage_not_main_ui(tmp_path):
    from ouroboros.context import build_recent_sections
    from ouroboros.gateway.history import make_chat_history_endpoint
    from ouroboros.memory import Memory
    from ouroboros.project_dialogue import append_terminal_task_projection
    from ouroboros.projects_registry import create_project

    project = create_project(tmp_path, "launch", name="Launch")
    project_chat = int(project["chat_id"])
    child = {
        "id": "child-review", "parent_task_id": "root", "root_task_id": "root",
        "project_id": "launch", "chat_id": project_chat,
        "delegation_role": "subagent", "role": "reviewer",
    }
    result = {**child, "task_id": "child-review", "status": "completed",
              "outcome_axes": {"execution": {"status": "ok"}}, "result": "Reviewed exact SHA"}
    assert append_terminal_task_projection(
        tmp_path, "child-review", child, result,
        {"chat_id": project_chat, "status": "completed"},
    )

    main_context = "\n\n".join(build_recent_sections(Memory(tmp_path), env=None))
    project_context = "\n\n".join(
        build_recent_sections(Memory(tmp_path), env=None, thread_chat_id=project_chat)
    )
    assert "Reviewed exact SHA" in main_context
    assert "parent=root" in main_context
    assert "Reviewed exact SHA" in project_context
    assert "parent=root" in project_context

    import asyncio

    endpoint = make_chat_history_endpoint(tmp_path)
    main_rows = json.loads(asyncio.run(endpoint(SimpleNamespace(
        query_params={"chat_id": "1"},
    ))).body)["messages"]
    project_rows = json.loads(asyncio.run(endpoint(SimpleNamespace(
        query_params={"chat_id": str(project_chat)},
    ))).body)["messages"]
    assert not any(row.get("task_id") == "child-review" for row in main_rows)
    assert any(row.get("task_id") == "child-review" for row in project_rows)

    unscoped = {
        "id": "child-main", "parent_task_id": "main-root", "root_task_id": "main-root",
        "chat_id": 1, "delegation_role": "subagent", "role": "researcher",
    }
    assert append_terminal_task_projection(
        tmp_path, "child-main", unscoped,
        {**unscoped, "task_id": "child-main", "status": "completed",
         "result": "Unscoped child truth", "outcome_axes": {"execution": {"status": "ok"}}},
        {"chat_id": 1, "status": "completed"},
    )
    main_context = "\n\n".join(build_recent_sections(Memory(tmp_path), env=None))
    assert "Unscoped child truth" in main_context
    main_rows = json.loads(asyncio.run(endpoint(SimpleNamespace(
        query_params={"chat_id": "1"},
    ))).body)["messages"]
    assert not any(row.get("task_id") == "child-main" for row in main_rows)


def test_project_build_reads_canonical_scratchpad_and_mutates_only_project_workpad(
    tmp_path, monkeypatch,
):
    import ouroboros.config as config

    from ouroboros.context import build_llm_messages
    from ouroboros.memory import Memory
    from ouroboros.tools.project_journal import _workpad_write

    canonical = tmp_path / "canonical"
    child = tmp_path / "child"
    repo = tmp_path / "repo"
    for path in (repo / "prompts", repo / "docs", canonical / "memory", canonical / "logs",
                 canonical / "state", child / "memory", child / "logs", child / "state"):
        path.mkdir(parents=True, exist_ok=True)
    (repo / "prompts" / "SYSTEM.md").write_text("System", encoding="utf-8")
    (repo / "BIBLE.md").write_text("Bible", encoding="utf-8")
    (repo / "docs" / "ARCHITECTURE.md").write_text("Architecture", encoding="utf-8")
    (repo / "docs" / "DEVELOPMENT.md").write_text("Development", encoding="utf-8")
    (repo / "docs" / "CHECKLISTS.md").write_text("Checklists", encoding="utf-8")
    (repo / "README.md").write_text("Readme", encoding="utf-8")
    (repo / "VERSION").write_text("1.0.0", encoding="utf-8")
    (repo / "pyproject.toml").write_text('version = "1.0.0"', encoding="utf-8")
    (canonical / "state" / "state.json").write_text('{"spent_usd": 0}', encoding="utf-8")
    (canonical / "memory" / "scratchpad.md").write_text(
        "CANONICAL BIOGRAPHY NOTE", encoding="utf-8",
    )
    monkeypatch.setenv("OUROBOROS_CONTEXT_MODE", "low")
    monkeypatch.setattr(config, "DATA_DIR", child)

    class Env:
        drive_root = child
        repo_dir = repo
        budget_drive_root = canonical

        def drive_path(self, rel):
            return child / rel

        def repo_path(self, rel):
            return repo / rel

    messages, _ = build_llm_messages(
        Env(), Memory(child, repo_dir=repo),
        {"id": "project-root", "text": "continue", "project_id": "launch",
         "budget_drive_root": str(canonical)},
    )
    assert "CANONICAL BIOGRAPHY NOTE" in json.dumps(messages, ensure_ascii=False)

    ctx = SimpleNamespace(project_id="launch", drive_root=child)
    assert _workpad_write(ctx, "LOCAL PROJECT WORK", "launch").startswith("OK:")
    assert (canonical / "memory" / "scratchpad.md").read_text(encoding="utf-8") == "CANONICAL BIOGRAPHY NOTE"
    from ouroboros.project_facts import project_workpad_path

    assert project_workpad_path("launch").read_text(encoding="utf-8") == "LOCAL PROJECT WORK"
