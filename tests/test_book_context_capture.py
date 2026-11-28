"""The resident Body Map shares captured sources in every mode; no mode preloads a chapter body."""

import json
from types import SimpleNamespace

import pytest

from ouroboros import context
from ouroboros.context_fit import MinimumViewUnfit, build_context_fit_plan
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


def _plan(env, core, task, mode, window=0):
    def route(*a, **kw):
        return ({"model": "fixture", "provider": "fixture"}, SimpleNamespace(
            route_fp="fixture", status="asserted" if window else "unprobeable", stale=False, window_tokens=window))
    return build_context_fit_plan(env, core, task, preferred_mode=mode, route_resolver=route)


def _text(plan, mode):
    return "\n".join(p["text"] for p in plan.projection(mode).system_message()["content"])


def test_main_max_and_nano_use_the_same_captured_body_map_bytes(tmp_path):
    env, core, task = _capture(tmp_path)
    plan = _plan(env, core, task, "nano")
    assert plan.initial_mode == "nano"
    full, nano = _text(plan, "max"), _text(plan, "nano")
    for mode in ("max", "low", "nano"):
        text = _text(plan, mode)
        assert "My complete identity." in text
        assert "My whole earlier biography." in text
        assert "The complete constitution." in text
        # One Map, the same captured bytes in every mode: each chapter's path and opening, never a chapter body.
        assert text.count(core.body_map) == 1 and text.count("How I keep work alive.") == 1
        assert "docs/architecture/runtime.md" in text and "docs/development/change.md" in text
        assert "Full mechanism and WHY." not in text and "Full engineering discipline." not in text
    assert "Parent's unselected current working note." in full
    assert "Unselected raw parent conversation." in full
    assert str(env.repo_dir) not in plan.projection("max").system_message()["content"][0]["text"]
    # Reading a newer file cannot mutate an already captured source or cached prefix.
    before = json.dumps(plan.messages_for("max"), sort_keys=True)
    (env.repo_dir / "docs/architecture/runtime.md").write_text("changed source")
    assert json.dumps(plan.messages_for("max"), sort_keys=True) == before
    assert _text(plan, "nano") == nano


def test_child_keeps_biography_and_the_body_map_in_max(tmp_path):
    env, core, task = _capture(tmp_path, {"delegation_role": "subagent"})
    text = _text(_plan(env, core, task, "max"), "max")
    assert "My whole earlier biography." in text and "My complete identity." in text
    # The same resident Body Map as Main: chapter openings and both books' chapter addresses, no chapter body.
    assert core.body_map in text and "How I keep work alive." in text
    assert "- Change: `docs/development/change.md`" in text
    assert "Full mechanism and WHY." not in text and "Full engineering discipline." not in text
    assert "Parent's unselected current working note." not in text
    assert "Unselected raw parent conversation." not in text
    # The child role line is carried verbatim: it no longer names tools or claims the
    # parent's selected discussion is attached (that claim was false: nothing attached it).
    assert "Read memory or sources whenever they could change what you do" in text and "name what you read in your answer" in text
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


def test_a_later_capture_reuses_book_parses_and_restored_mtime_edit_is_fresh(tmp_path, monkeypatch):
    import os
    from ouroboros import reference_books as books

    env, _core, task = _capture(tmp_path)
    books._book_source.cache_clear()
    memory = Memory(drive_root=env.drive_root, repo_dir=env.repo_dir)
    context._capture_context_core(env, memory, task, None, None)  # an earlier capture in this process parsed them
    cold_parse = books.parse_markdown_source
    parses = []
    monkeypatch.setattr(books, "parse_markdown_source", lambda *a: (parses.append(a[1]), cold_parse(*a))[1])
    first = context._capture_context_core(env, memory, task, None, None)
    assert parses == []
    source = env.repo_path("docs/architecture/runtime.md")
    stamp = source.stat()
    # Same size, restored mtime: an edit of the chapter's opening (which the Map carries) and of its body (read on demand).
    source.write_bytes(source.read_bytes().replace(b"Full mechanism", b"Next mechanism").replace(b"keep work", b"hold work"))
    os.utime(source, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
    after = context._capture_context_core(env, memory, task, None, None)
    assert parses == ["docs/architecture/runtime.md"]
    assert after.body_map != first.body_map and "How I hold work alive." in after.body_map
    assert "Next mechanism" not in after.body_map  # the body stays behind the Map's address
    chapter = next(book for book in after.reference_books if book.book_id == "architecture").chapters[0]
    assert "Next mechanism" in chapter.text
    books._book_source.cache_clear()
    fresh = context._capture_context_core(env, memory, task, None, None)
    assert (fresh.body_map, fresh.development_md, fresh.memory_view_json) == (
        after.body_map, after.development_md, after.memory_view_json)
    source.unlink()
    assert context._capture_context_core(env, memory, task, None, None).reference_book_errors


def test_large_nano_task_retains_exact_owner_source_without_changing_max(tmp_path):
    from ouroboros.artifacts import read_actor_source_bytes

    task_text = "The exact whole task with a late correction. " * 10000 + "FINAL OWNER CRITERION"
    env, core, task = _capture(tmp_path, {"text": task_text})
    # An unknown window does not hide the owner's words to chase the Nano target (#1527, owner A).
    unknown = _plan(env, core, task, "nano")
    assert unknown.messages_for("nano")[-1]["content"] == task_text
    assert unknown.nano_projection.user_content_json is None
    # A known window that cannot hold them: the exact source replaces them, and the pointer request fits.
    window = 81920
    plan = _plan(env, core, task, "nano", window)
    nano = plan.messages_for("nano")[-1]["content"]
    assert nano.startswith("[Exact task input source]")
    ref = json.loads(nano[nano.index("{"):])
    original = json.loads(read_actor_source_bytes(env.drive_root, task["id"], ref))
    assert original == task_text
    assert plan.messages_for("max")[-1]["content"] == task_text
    assert "FINAL OWNER CRITERION" in json.loads(core.user_content_json)
    assert plan.nano_projection.estimated_tokens < window
    # Only a window that cannot hold even the pointer request names the incompatibility.
    with pytest.raises(MinimumViewUnfit, match="cannot fit"):
        _plan(env, core, task, "nano", 10_000)


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


@pytest.mark.parametrize("ceiling", ["no-ceiling", "system-docs", "no-reader", "workspace-reader"])
def test_a_presence_ceiling_that_grants_file_reading_follows_the_maps_addresses_and_nothing_more(tmp_path, ceiling):
    """The Map names its reader only for an actor that can follow it, and every address it names is followable by
    one that carries it and was granted file reading (owner's decision, 27 Nov 2026): a Presence turn whose positive
    ceiling grants ``read_file``, whatever its resource grants, reads the shipped manuals the Map names (both
    books' entrypoints and chapters, the Reference Atlas's documents) through the real capture and registry, and
    still reads nothing else its grants do not admit. A ceiling with no ``read_file`` reads none of them and gets
    ``BODY_MAP_NO_READER`` in the whole Map and in its parts form."""
    from ouroboros import reference_books
    from ouroboros.tools.registry import ToolContext, ToolRegistry
    from ouroboros.tools.registry_core import PRESENCE_RESOURCE_BLOCKED
    from tests import test_presence_authority as pa

    targets = {"no-ceiling": None, "no-reader": (),
               "workspace-reader": (pa.PresenceToolTarget("builtin", "read_file"),
                                    pa.PresenceResourceTarget("active_workspace", ("read",), "shared")),
               "system-docs": (pa.PresenceToolTarget("builtin", "read_file"),
                               pa.PresenceResourceTarget("system_repo", ("read",), "docs"))}[ceiling]
    contract = {} if targets is None else {"capability_ceiling": pa.presence_ceiling_payload(
        pa.build_presence_capability_ceiling(skill_name="moderator", skill_content_hash="c" * 64,
                                             state_fingerprint="d" * 64, resolution=pa._resolution(*targets)))}
    env, _plain, task = _capture(tmp_path, {"task_contract": contract} if contract else None)
    entry = env.repo_dir / "docs" / "ARCHITECTURE.md"
    entry.write_text(entry.read_text(encoding="utf-8") + "\n## Reference Atlas\n\n- [DESIGN.md](DESIGN.md) — how the "
                     "screens look.\n- `README.md` and `CONTRIBUTING.md` — the first look.\n", encoding="utf-8")
    for path in ("docs/DESIGN.md", "README.md", "docs/unlisted.md", "ouroboros/loop.py"):
        (env.repo_dir / path).parent.mkdir(parents=True, exist_ok=True)
        (env.repo_dir / path).write_text("text\n", encoding="utf-8")  # CONTRIBUTING.md is named and absent: no address
    ctx = ToolContext(repo_dir=env.repo_dir, drive_root=env.drive_root, task_contract=contract)
    registry = ToolRegistry(repo_dir=env.repo_dir, drive_root=env.drive_root)
    registry.set_context(ctx)
    core = context._capture_context_core(env, Memory(drive_root=env.drive_root, repo_dir=env.repo_dir), task, None, ctx,
                                         registry)
    readable = ceiling != "no-reader"
    promise = 'read_file(root="system_repo"'
    for form in (core.body_map, core.body_map_parts):
        assert (promise in form) is readable and (reference_books.BODY_MAP_NO_READER in form) is not readable, form
    assert (promise in _text(_plan(env, core, task, "max"), "max")) is readable

    def read(path):
        return registry.execute_result("read_file", {"root": "system_repo", "path": path})

    manuals = ("docs/ARCHITECTURE.md", "docs/architecture/runtime.md", "docs/DEVELOPMENT.md",
               "docs/development/change.md", "docs/DESIGN.md", "README.md")
    assert sorted(reference_books.body_map_addresses(env.repo_dir)) == sorted(manuals)
    for path in manuals:
        assert (read(path).status == "ok") is readable, (path, read(path).text[:200])
    # Nothing the Map does not name opens by this: only a grant that already admitted it (or no ceiling) reads it.
    for path, admitted in (("docs/unlisted.md", ("no-ceiling", "system-docs")), ("ouroboros/loop.py", ("no-ceiling",)),
                           ("BIBLE.md", ("no-ceiling",))):
        got = read(path)
        assert (got.status == "ok") is (ceiling in admitted), (path, got.text[:200])
        if ceiling in ("system-docs", "workspace-reader") and ceiling not in admitted:
            assert PRESENCE_RESOURCE_BLOCKED in got.text, (path, got.text[:200])
