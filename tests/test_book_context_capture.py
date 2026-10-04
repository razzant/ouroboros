"""Book projections share captured sources without narrowing Main Max."""

import json
from types import SimpleNamespace

from ouroboros import context
from ouroboros.context_fit import build_context_fit_plan
from ouroboros.memory import Memory


def _capture(tmp_path, task=None):
    repo, drive = tmp_path / "repo", tmp_path / "data"
    files = {
        "prompts/SYSTEM.md": "I am Ouroboros.", "BIBLE.md": "The complete constitution.",
        "docs/ARCHITECTURE.md": "# Body\n\nThe whole body map.\n\n## Chapters\n\n- [Runtime](architecture/runtime.md)\n",
        "docs/architecture/runtime.md": "# Runtime\n\nHow I keep work alive.\n\n## Execution\n\nFull mechanism and WHY.\n",
        "docs/DEVELOPMENT.md": "# Engineering\n\nHow I change.\n\n## Chapters\n\n- [Change](development/change.md)\n",
        "docs/development/change.md": "# Change\n\nHow changes remain coherent.\n\n## Practice\n\nFull engineering discipline.\n",
    }
    for path, text in files.items():
        target = repo / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text)
    memory = Memory(drive_root=drive, repo_dir=repo)
    memory.ensure_files()
    (drive / "memory" / "identity.md").write_text("My complete identity.")
    (drive / "memory" / "dialogue_summary.md").write_text("My whole earlier biography.")
    (drive / "memory" / "scratchpad.md").write_text("Parent's unselected current working note.")
    (drive / "logs").mkdir(exist_ok=True)
    (drive / "logs" / "chat.jsonl").write_text(json.dumps({"direction": "in", "chat_id": 1,
                                                          "text": "Unselected raw parent conversation."}) + "\n")
    env = SimpleNamespace(repo_dir=repo, drive_root=drive, repo_path=lambda p: repo / p,
                          drive_path=lambda p: drive / p)
    task = {"id": "book-test", "type": "task", "text": "Read my body", **(task or {})}
    return env, context._capture_context_core(env, memory, task, None, None), task


def _plan(env, core, task, mode):
    def route(*a, **kw):
        return ({"model": "fixture", "provider": "fixture"},
                SimpleNamespace(route_fp="fixture", status="unprobeable", stale=False, window_tokens=0))
    return build_context_fit_plan(env, core, task, preferred_mode=mode, route_resolver=route)


def _text(plan, mode):
    return "\n".join(p["text"] for p in plan.projection(mode).system_message()["content"])


def test_main_max_and_nano_use_the_same_captured_book_bytes(tmp_path):
    env, core, task = _capture(tmp_path)
    plan = _plan(env, core, task, "nano")
    assert plan.initial_mode == "nano"
    full, nano = _text(plan, "max"), _text(plan, "nano")
    for mode in ("max", "low", "nano"):
        text = _text(plan, mode)
        assert "My complete identity." in text
        assert "My whole earlier biography." in text
        assert "The complete constitution." in text
        assert text.count("How I keep work alive.") == 1
    assert "Full mechanism and WHY." in full and "Full engineering discipline." in full
    assert "Parent's unselected current working note." in full
    assert "Unselected raw parent conversation." in full
    assert "Full mechanism and WHY." not in nano and "Full engineering discipline." not in nano
    assert "docs/architecture/runtime.md" in nano and "docs/development/change.md" in nano
    assert str(env.repo_dir) not in plan.projection("max").system_message()["content"][0]["text"]
    # Reading a newer file cannot mutate an already captured source or cached prefix.
    before = json.dumps(plan.messages_for("max"), sort_keys=True)
    (env.repo_dir / "docs/architecture/runtime.md").write_text("changed source")
    assert json.dumps(plan.messages_for("max"), sort_keys=True) == before
    assert _text(plan, "nano") == nano


def test_child_keeps_biography_and_book_orientation_in_max(tmp_path):
    env, core, task = _capture(tmp_path, {"delegation_role": "subagent"})
    text = _text(_plan(env, core, task, "max"), "max")
    assert "My whole earlier biography." in text and "My complete identity." in text
    assert "How I keep work alive." in text and "How changes remain coherent." in text
    assert "Full mechanism and WHY." not in text
    assert "Parent's unselected current working note." not in text
    assert "Unselected raw parent conversation." not in text
    # The child role line is carried verbatim: it no longer names tools or claims the
    # parent's selected discussion is attached (that claim was false: nothing attached it).
    assert "Work from this assignment first" in text and "name what you read in your report" in text
    assert "The shared biography is loaded above" not in text
    assert "parent's selected discussion" not in text


def test_missing_chapter_cannot_masquerade_as_a_complete_book(tmp_path):
    env, _core, task = _capture(tmp_path)
    (env.repo_dir / "docs/architecture/runtime.md").unlink()
    memory = Memory(drive_root=env.drive_root, repo_dir=env.repo_dir)
    core = context._capture_context_core(env, memory, task, None, None)
    plan = _plan(env, core, task, "max")
    for mode in ("max", "low", "nano"):
        assert "Reference book source unavailable" in _text(plan, mode)
    assert not any(book.book_id == "architecture" for book in core.reference_books)


def test_large_nano_task_retains_exact_owner_source_without_changing_max(tmp_path):
    from ouroboros.artifacts import read_actor_source_bytes

    task_text = "The exact whole task with a late correction. " * 10000 + "FINAL OWNER CRITERION"
    env, core, task = _capture(tmp_path, {"text": task_text})
    plan = _plan(env, core, task, "nano")
    nano = plan.messages_for("nano")[-1]["content"]
    assert nano.startswith("[Exact task input source]")
    ref = json.loads(nano[nano.index("{"):])
    original = json.loads(read_actor_source_bytes(env.drive_root, task["id"], ref))
    assert original == task_text
    assert plan.messages_for("max")[-1]["content"] == task_text
    assert "FINAL OWNER CRITERION" in json.loads(core.user_content_json)
    assert plan.nano_projection.estimated_tokens < 81920


def test_authored_common_understanding_is_resident_with_every_note_summary(tmp_path):
    """The authored summary is the resident face of a note, common orientation or not."""
    from ouroboros.knowledge import resolve_knowledge_address, write_knowledge_note

    env, _core, task = _capture(tmp_path)
    write_knowledge_note(resolve_knowledge_address(env.drive_root, "overview"),
                        "Our shared understanding includes context-sensitive preferences.")
    write_knowledge_note(resolve_knowledge_address(env.drive_root, "people/alex"),
                        "---\ntype: relationship\ntitle: Alex\nsummary: Details belong to their source.\n---\nDetailed source.")
    sections = "\n".join(context.build_knowledge_sections(env))
    assert "context-sensitive preferences" in sections
    assert "people/alex" in sections and "Alex" in sections
    assert "Details belong to their source" in sections
    assert "Detailed source." not in sections  # the body still belongs to its own note
    assert "scope='global'" in sections
