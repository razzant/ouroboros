from __future__ import annotations

from tests import _dialogue as dialogue_test

from types import SimpleNamespace


def _chat_rows(root):
    path = root / "logs" / "chat.jsonl"
    return dialogue_test.log_rows(path)


def test_root_checkpoint_prevents_a_second_facts_row_and_buys_no_summary(tmp_path, monkeypatch):
    from ouroboros import agent_task_pipeline as pipeline
    from ouroboros.task_results import STATUS_COMPLETED, write_task_result

    class Llm:
        def __init__(self):
            self.calls = 0

        def chat(self, **_kwargs):
            self.calls += 1
            return {"content": "No paid narrative exists"}, {"cost": 0}

    import ouroboros.llm as llm_mod
    import ouroboros.memory as memory_mod
    import ouroboros.post_task_evolution as evolution

    llm = Llm()
    monkeypatch.setattr(llm_mod, "LLMClient", lambda: llm)
    monkeypatch.setattr(memory_mod, "Memory", lambda **_kwargs: object())
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
        {"tool_calls": [{"tool": "read_file"}]}, tmp_path / "logs",
    )
    pipeline._run_post_task_processing_async(*args, blocking=True)
    # The settled checkpoint, not the dialogue, decides: a second pass buys no
    # synthesis and writes no second facts row.
    pipeline._run_post_task_processing_async(*args, blocking=True)

    assert llm.calls == 0
    kinds = [row.get("summary_kind") for row in _chat_rows(tmp_path) if row.get("task_id") == "root-llm"]
    assert kinds.count("host_task_facts") == 1 and "authored_root_summary" not in kinds


def test_terminal_projection_dedup_does_not_lose_concurrent_chat_append(tmp_path):
    from concurrent.futures import ThreadPoolExecutor

    from ouroboros import dialogue_store
    from ouroboros.project_dialogue import append_terminal_task_projection
    from ouroboros.task_results import STATUS_COMPLETED, write_task_result

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
        return dialogue_store.append(tmp_path, "chat", {"type": "concurrent_noise", "index": index})

    with ThreadPoolExecutor(max_workers=12) as pool:
        projection_results = list(pool.map(project, range(12)))
        noise_results = list(pool.map(noise, range(24)))

    rows = _chat_rows(tmp_path)
    assert projection_results.count(True) == 1
    assert all(noise_results)
    assert len([row for row in rows if row.get("summary_id") == "task-terminal:concurrent-child"]) == 1
    assert {row["index"] for row in rows if row.get("type") == "concurrent_noise"} == set(range(24))
