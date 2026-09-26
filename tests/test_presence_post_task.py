from types import SimpleNamespace


def test_presence_keeps_own_memory_but_skips_evolution_effects(tmp_path, monkeypatch):
    import ouroboros.agent_task_pipeline as pipeline
    import ouroboros.llm as llm_module
    import ouroboros.memory as memory_module
    import ouroboros.post_task_evolution as evolution

    calls = []
    monkeypatch.setattr(pipeline, "_is_root_post_task", lambda task: False)
    monkeypatch.setattr(llm_module, "LLMClient", lambda: object())
    monkeypatch.setattr(memory_module, "Memory", lambda **kwargs: object())
    monkeypatch.setattr(
        pipeline,
        "_run_chat_consolidation",
        lambda *args, **kwargs: calls.append("chat_consolidation"),
    )
    monkeypatch.setattr(
        pipeline,
        "_run_scratchpad_consolidation",
        lambda *args, **kwargs: calls.append("scratchpad_consolidation"),
    )
    monkeypatch.setattr(
        pipeline,
        "_record_task_facts",
        lambda *args, **kwargs: calls.append("facts"),
    )
    monkeypatch.setattr(
        pipeline,
        "_run_reflection",
        lambda *args, **kwargs: calls.append("reflection") or {"reflection": "ok"},
    )
    monkeypatch.setattr(
        pipeline,
        "_update_improvement_backlog",
        lambda *args, **kwargs: calls.append("backlog"),
    )
    monkeypatch.setattr(
        pipeline,
        "_apply_reflection_memory_actions",
        lambda *args, **kwargs: calls.append("memory_actions"),
    )
    monkeypatch.setattr(
        evolution,
        "maybe_promote",
        lambda *args, **kwargs: calls.append("maybe_promote"),
    )
    env = SimpleNamespace(
        drive_root=tmp_path,
        repo_dir=tmp_path,
        drive_path=lambda relative: tmp_path / relative,
    )
    task = {
        "id": "presence-1",
        "type": "presence",
        "_presence_turn": True,
        "metadata": {"presence": {"binding_id": "b" * 32}},
    }

    result = pipeline._run_post_task_processing_async(
        env,
        task,
        {"rounds": 2, "cost": 0.0},
        {"tool_calls": []},
        {},
        tmp_path / "logs",
        blocking=True,
        on_reflection=lambda *args: calls.append("on_reflection"),
    )

    assert result == {"reflection": "ok"}
    assert calls == [
        "facts",  # free, before every paid stage
        "chat_consolidation",
        "scratchpad_consolidation",
        "reflection",
        "memory_actions",
    ]


def test_ordinary_task_retains_global_post_task_effects(tmp_path, monkeypatch):
    import ouroboros.agent_task_pipeline as pipeline
    import ouroboros.llm as llm_module
    import ouroboros.memory as memory_module
    import ouroboros.post_task_evolution as evolution

    calls = []
    monkeypatch.setattr(pipeline, "_is_root_post_task", lambda task: False)
    monkeypatch.setattr(llm_module, "LLMClient", lambda: object())
    monkeypatch.setattr(memory_module, "Memory", lambda **kwargs: object())
    monkeypatch.setattr(pipeline, "_run_chat_consolidation", lambda *args, **kwargs: None)
    monkeypatch.setattr(pipeline, "_run_scratchpad_consolidation", lambda *args, **kwargs: None)
    monkeypatch.setattr(pipeline, "_record_task_facts", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        pipeline,
        "_run_reflection",
        lambda *args, **kwargs: {"reflection": "ok"},
    )
    monkeypatch.setattr(
        pipeline,
        "_update_improvement_backlog",
        lambda *args, **kwargs: calls.append("backlog"),
    )
    monkeypatch.setattr(
        pipeline,
        "_apply_reflection_memory_actions",
        lambda *args, **kwargs: calls.append("memory_actions"),
    )
    monkeypatch.setattr(
        evolution,
        "maybe_promote",
        lambda *args, **kwargs: calls.append("maybe_promote"),
    )
    env = SimpleNamespace(
        drive_root=tmp_path,
        repo_dir=tmp_path,
        drive_path=lambda relative: tmp_path / relative,
    )

    pipeline._run_post_task_processing_async(
        env,
        {"id": "ordinary-1", "type": "task"},
        {"rounds": 2, "cost": 0.0},
        {"tool_calls": []},
        {},
        tmp_path / "logs",
        blocking=True,
        on_reflection=lambda *args: calls.append("on_reflection"),
    )

    assert calls == ["memory_actions", "backlog", "maybe_promote", "on_reflection"]


def test_presence_post_task_applies_own_experience_with_background_off(tmp_path, monkeypatch):
    import ouroboros.agent_task_pipeline as pipeline
    import ouroboros.llm as llm_module
    from ouroboros.knowledge import read_knowledge_note, resolve_knowledge_address

    monkeypatch.setattr(llm_module, "LLMClient", lambda: object())
    for name in ("_run_chat_consolidation", "_run_scratchpad_consolidation", "_record_task_facts"):
        monkeypatch.setattr(pipeline, name, lambda *a, **k: None)
    entry = {"reflection": "A useful shared moment.", "memory_actions": [{
        "type": "knowledge_write", "topic": "shared experience", "scope": "global",
        "canonical_root": str(tmp_path), "content": "We enjoyed exploring an idea together.",
        "task_id": "presence-memory",
    }]}
    monkeypatch.setattr(pipeline, "_run_reflection", lambda *a, **k: entry)
    monkeypatch.setattr(pipeline, "_update_improvement_backlog", lambda *a, **k: (_ for _ in ()).throw(AssertionError("Presence cannot promote evolution")))
    env = SimpleNamespace(drive_root=tmp_path, repo_dir=tmp_path, drive_path=lambda rel: tmp_path / rel)
    task = {"id": "presence-memory", "type": "presence", "_presence_turn": True,
            "_ephemeral_turn": True, "_is_direct_chat": True,
            "metadata": {"presence": {"binding_id": "b" * 32}}, "bg_consciousness_enabled": False}
    result = pipeline._run_post_task_processing_async(env, task, {"rounds": 2, "cost": 0.0},
                                                       {"tool_calls": []}, {}, tmp_path / "logs", blocking=True)
    assert result == entry
    note = read_knowledge_note(resolve_knowledge_address(tmp_path, "shared experience", "global"))
    assert "enjoyed exploring" in note.text
