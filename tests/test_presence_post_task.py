"""The post-task stage census of a Presence turn and of an ordinary task.

No stage writes a lesson, the Pattern Register or the backlog. A Presence turn
keeps its own memory upkeep and never reaches evolution; an ordinary task also runs
the owner-enabled promotion stage and the global callback a split Project root hands
it.
"""

from types import SimpleNamespace


def _census(monkeypatch):
    import ouroboros.agent_task_pipeline as pipeline
    import ouroboros.llm as llm_module
    import ouroboros.memory as memory_module
    import ouroboros.post_task_evolution as evolution

    calls = []
    monkeypatch.setattr(pipeline, "_is_root_post_task", lambda task: False)
    monkeypatch.setattr(llm_module, "LLMClient", lambda: object())
    monkeypatch.setattr(memory_module, "Memory", lambda **kwargs: object())
    for name, label in (("_record_task_facts", "facts"), ("_run_memory_fallback_draft", "memory_fallback_draft")):
        monkeypatch.setattr(pipeline, name, lambda *args, _label=label, **kwargs: calls.append(_label))
    monkeypatch.setattr(evolution, "maybe_promote", lambda *args, **kwargs: calls.append("maybe_promote"))
    return pipeline, calls


def _env(tmp_path):
    return SimpleNamespace(drive_root=tmp_path, repo_dir=tmp_path, drive_path=lambda relative: tmp_path / relative)


def test_presence_keeps_own_memory_but_skips_evolution_effects(tmp_path, monkeypatch):
    pipeline, calls = _census(monkeypatch)
    task = {
        "id": "presence-1",
        "type": "presence",
        "_presence_turn": True,
        "metadata": {"presence": {"binding_id": "b" * 32}},
    }

    result = pipeline._run_post_task_processing_async(
        _env(tmp_path),
        task,
        {"rounds": 2, "cost": 0.0},
        {"tool_calls": []},
        tmp_path / "logs",
        blocking=True,
        on_reflection=lambda *args: calls.append("on_reflection"),
    )

    assert result is None
    assert calls == [
        "facts",  # free, before every paid stage
        "memory_fallback_draft",
    ]
    assert not (tmp_path / "memory").exists()


def test_ordinary_task_retains_global_post_task_effects(tmp_path, monkeypatch):
    pipeline, calls = _census(monkeypatch)

    pipeline._run_post_task_processing_async(
        _env(tmp_path),
        {"id": "ordinary-1", "type": "task"},
        {"rounds": 2, "cost": 0.0},
        {"tool_calls": []},
        tmp_path / "logs",
        blocking=True,
        on_reflection=lambda *args: calls.append("on_reflection"),
    )

    assert calls == ["facts", "memory_fallback_draft", "maybe_promote", "on_reflection"]
    assert not (tmp_path / "memory").exists()
