"""The same Markdown source/address and revision survive every KB operation."""

from __future__ import annotations

import hashlib
import json
import threading
from contextlib import contextmanager

import pytest

from ouroboros import knowledge as store
from ouroboros import knowledge_versions, text_change
from ouroboros.tools import knowledge as tools
from ouroboros.tools.registry import ToolContext, ToolRegistry
from tests._knowledge_history_shared import applied_blocks, published, replaced


def address(tmp_path, topic="people/Антон", scope="global"):
    return store.resolve_knowledge_address(tmp_path, topic, scope)


def history(addr):
    return [json.loads(line) for line in (addr.shelf.parent / "knowledge_history.jsonl").read_text(encoding="utf-8").splitlines()]


def offsets(addr):
    """Each history row's byte offset in its shelf's journal, in order."""
    raw = (addr.shelf.parent / "knowledge_history.jsonl").read_bytes()
    return [0] + [index + 1 for index, byte in enumerate(raw[:-1]) if byte == 0x0A]


def test_plain_new_note_gets_minimal_format_without_invented_summary(tmp_path):
    target = address(tmp_path)
    result = store.write_knowledge_note(target, "# Антон\n\nA personal observation, not a permanent rule.")
    assert result.ok
    assert result.current.metadata == {"type": "note"}
    assert result.current.summary == ""
    assert "personal observation" not in (target.shelf / store.INDEX_FILE).read_text(encoding="utf-8")
    assert result.current.text.endswith("not a permanent rule.")
    assert result.current.revision == hashlib.sha256(target.path.read_bytes()).hexdigest()


def test_authored_multiline_summary_and_unknown_metadata_survive_revision(tmp_path):
    target = address(tmp_path)
    source = """---
type: evolving-understanding
title: Anton
summary: |
  Brevity helps when he is hurried.
  Detailed reasoning is welcome when he asks for it.
custom:
  evidence: [episode-1, episode-2]
---

# Anton

[Recent work](../work/research.md)
An interpretation grounded in two episodes.
"""
    first = store.write_knowledge_note(target, source)
    assert first.ok and first.current.text == source
    index = (target.shelf / store.INDEX_FILE).read_text(encoding="utf-8")
    assert "Brevity helps" in index and "Detailed reasoning" in index
    second = store.write_knowledge_note(target, "---\nsummary: A revised interpretation.\n---\nNew episode.",
                                       expected_revision=first.current.revision)
    assert second.ok
    assert second.current.metadata["type"] == "evolving-understanding"
    assert second.current.metadata["custom"] == {"evidence": ["episode-1", "episode-2"]}
    assert second.current.metadata["title"] == "Anton"
    assert second.current.summary == "A revised interpretation."
    assert replaced(tmp_path, history(target)[-1]) == source
    assert published(tmp_path, history(target)[-1]) == second.current.text


def test_legacy_read_append_and_explicit_overwrite_do_not_force_migration(tmp_path):
    target = address(tmp_path, "old")
    target.shelf.mkdir(parents=True)
    target.path.write_bytes(b"# Legacy\r\nA plain note.")
    before = {p.relative_to(tmp_path) for p in tmp_path.rglob("*")}
    read = store.read_knowledge_note(target)
    assert read.raw == b"# Legacy\r\nA plain note."
    assert read.metadata == {} and read.summary == ""
    store.inventory_knowledge(target)
    assert before == {p.relative_to(tmp_path) for p in tmp_path.rglob("*")}
    appended = store.write_knowledge_note(target, "New evidence.", "append")
    assert appended.ok and appended.current.raw == read.raw + b"\nNew evidence."
    changed = store.write_knowledge_note(target, "# Legacy\nReconsidered.", expected_revision=appended.current.revision)
    assert changed.ok and changed.current.text == "# Legacy\nReconsidered."


def test_stale_overwrite_preserves_newer_note_index_and_history(tmp_path):
    target = address(tmp_path)
    first = store.write_knowledge_note(target, "First understanding.").current
    second = store.write_knowledge_note(target, "New evidence.", expected_revision=first.revision).current
    before = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    stale = store.write_knowledge_note(target, "Outdated rewrite.", expected_revision=first.revision)
    assert not stale.ok and stale.reason == "revision_conflict"
    assert stale.current.raw == second.raw
    assert before == {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    no_revision = store.write_knowledge_note(target, "Unchecked replacement.")
    assert not no_revision.ok and no_revision.reason == "revision_required"
    assert no_revision.current.revision == second.revision


def test_two_readers_one_revision_cannot_both_overwrite(tmp_path):
    target = address(tmp_path)
    original = store.write_knowledge_note(target, "Original.").current
    start = threading.Barrier(3)
    results = []

    def write(content):
        start.wait()
        results.append(store.write_knowledge_note(target, content, expected_revision=original.revision))

    workers = [threading.Thread(target=write, args=(text,)) for text in ("Episode A", "Episode B")]
    for worker in workers:
        worker.start()
    start.wait()
    for worker in workers:
        worker.join(3)
    assert all(not worker.is_alive() for worker in workers)
    assert sorted(result.ok for result in results) == [False, True]
    winner = next(result.current for result in results if result.ok)
    assert store.read_knowledge_note(target).revision == winner.revision
    assert published(tmp_path, history(target)[-1]) == winner.text


def test_noop_does_not_publish_history_or_rewrite_index(tmp_path):
    target = address(tmp_path)
    first = store.write_knowledge_note(target, "---\ntype: note\n---\nSame.").current
    before = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    result = store.write_knowledge_note(target, first.text, expected_revision=first.revision)
    assert result.ok and result.reason == "unchanged"
    assert before == {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}


def test_a_body_only_edit_leaves_the_index_file_untouched_and_a_summary_change_rewrites_it(tmp_path):
    target = address(tmp_path, "people/rowan")
    first = store.write_knowledge_note(target, "---\ntype: person\nsummary: Rowan runs it.\n---\n# Rowan\n\nOld.\n").current
    index = target.shelf / store.INDEX_FILE
    before, stat = index.read_bytes(), index.stat()
    body = store.write_knowledge_note(target, "New.", "edit", first.revision, old_str="Old.")
    assert body.ok and body.reason == "saved" and index.read_bytes() == before
    assert (index.stat().st_ino, index.stat().st_mtime_ns) == (stat.st_ino, stat.st_mtime_ns)  # not replaced
    summary = store.write_knowledge_note(target, "", "edit", body.current.revision, summary="Rowan hands it over.")
    assert summary.ok and index.read_bytes() != before and b"Rowan hands it over." in index.read_bytes()


def test_exact_edit_preserves_other_knowledge_and_reports_its_delta(tmp_path):
    ctx = ToolContext(repo_dir=tmp_path, drive_root=tmp_path, task_id="editor")
    target = address(tmp_path, "people/rowan")
    original = store.write_knowledge_note(target,
        "---\ntype: person\nsummary: Rowan maintains the service.\ncustom: [kept]\n---\n\n"
        "# Rowan\n\n- Prefers async updates.\n- Old staging host.\n"
        "## Open question\n- Ownership of validation unknown.\n").current
    reply = tools._knowledge_write(ctx, "people/rowan", "- New staging host.", mode="edit",
                                   old_str="- Old staging host.", scope="global",
                                   expected_revision=original.revision)
    assert reply.startswith("✅")
    updated = store.read_knowledge_note(target)
    assert updated.text == original.text.replace("- Old staging host.", "- New staging host.")
    assert updated.metadata == original.metadata
    assert "# Rowan" in updated.text and "## Open question" in updated.text
    change = history(target)[-1]
    assert change["mode"] == "edit" and replaced(tmp_path, change) == original.text
    assert published(tmp_path, change) == updated.text
    assert change["delta"] == {"old_chars": len(original.text), "new_chars": len(updated.text),
                                "old_body_chars": store.body_chars(original), "new_body_chars": store.body_chars(updated),
                                "change_chars": len(updated.text) - len(original.text), "removed_headings": [],
                                "added_headings": [], "removed_chars": 3, "added_chars": 3,  # "Old" → "New"
                                "removed_sections": [],
                                "body_changed": True, "summary_changed": False}
    assert json.loads(reply.split("\n", 1)[1])["knowledge_delta"] == change["delta"]


def test_edit_requires_one_exact_body_anchor_and_current_revision(tmp_path):
    target = address(tmp_path, "body")
    original = store.write_knowledge_note(target, "# Heading\n\nRepeated. Repeated.\n").current
    before = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    for old_str, revision in (("Repeated.", original.revision), ("Absent.", original.revision),
                              ("# Heading", None), ("Repeated.", "stale")):
        result = store.write_knowledge_note(target, "changed", mode="edit", old_str=old_str,
                                            expected_revision=revision)
        assert not result.ok
    assert before == {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    assert store.write_knowledge_note(target, "changed", mode="edit", old_str="Repeated.",
                                      expected_revision=None).reason == "revision_required"
    assert store.write_knowledge_note(address(tmp_path, "missing"), "new", mode="edit",
                                      old_str="old", expected_revision="").reason == "edit_source_missing"
    overlapping = store.write_knowledge_note(address(tmp_path, "overlap"), "# Body\n\naaa").current
    ambiguous = store.write_knowledge_note(overlapping.address, "replacement", mode="edit",
                                           old_str="aa", expected_revision=overlapping.revision)
    assert not ambiguous.ok and "exactly once" in ambiguous.reason
    assert overlapping.address.path.read_bytes() == overlapping.raw


def test_edit_cannot_turn_a_plain_note_into_frontmatter(tmp_path):
    target = address(tmp_path, "plain")
    target.path.parent.mkdir(parents=True)
    target.path.write_bytes(b"# Body\n\nKnown fact.\n")  # legacy plain source, not an auto-formatted new note
    original = store.read_knowledge_note(target)
    # A plain note's entire text is body; editing its opening can otherwise
    # silently acquire metadata or become a malformed frontmatter preamble.
    for opening in ("---\ntype: person\n---\n", "---\ncustom: [bad\n---\n"):
        result = store.write_knowledge_note(target, opening, mode="edit", old_str="# Body\n",
                                            expected_revision=original.revision)
        assert not result.ok and result.reason.startswith("invalid_note")
        assert target.path.read_bytes() == original.raw
    assert not (target.shelf.parent / "knowledge_history.jsonl").exists()


def test_edit_accepts_recursive_yaml_alias_without_comparing_metadata_graphs(tmp_path):
    target = address(tmp_path, "recursive")
    target.path.parent.mkdir(parents=True)
    target.path.write_bytes(b"---\ntype: note\ncustom: &loop [*loop]\n---\nOld.\n")
    original = store.read_knowledge_note(target)
    assert original.source is not None and not original.parse_error
    result = store.write_knowledge_note(target, "New.", mode="edit", old_str="Old.",
                                        expected_revision=original.revision)
    assert result.ok and result.current.raw == original.raw.replace(b"Old.", b"New.")
    assert replaced(tmp_path, history(target)[-1]) == original.text


def test_summary_revision_still_refuses_a_rerender_that_changes_a_recursive_field(tmp_path, monkeypatch):
    target = address(tmp_path, "recursive")
    target.path.parent.mkdir(parents=True)
    target.path.write_bytes(b"---\ntype: note\ncustom: &loop [*loop]\n---\nOld.\n")
    original = store.read_knowledge_note(target)
    write_content = store._write_content
    # A re-render that loses the alias is a changed field, not a revised summary.
    monkeypatch.setattr(store, "_write_content", lambda *args: write_content(*args).replace(b"- *id001", b"- []"))
    result = store.write_knowledge_note(target, "New.", mode="edit", old_str="Old.", summary="New view.",
                                        expected_revision=original.revision)
    assert result.reason == "invalid_note: edit cannot change frontmatter"
    assert target.path.read_bytes() == original.raw
    assert not (target.shelf.parent / "knowledge_history.jsonl").exists()


def test_large_removed_heading_delta_keeps_tool_receipt_and_full_history(tmp_path):
    ctx = ToolContext(repo_dir=tmp_path, drive_root=tmp_path, task_id="editor")
    target = address(tmp_path, "large-headings")
    body = "".join(f"# Раздел {i:03d}\n" for i in range(240))
    created = json.loads(tools._knowledge_write(ctx, "large-headings", body, mode="overwrite", scope="global").split("\n", 1)[1])
    assert created["knowledge_delta"]["added_headings_count"] == 240 and created["knowledge_delta"]["added_headings_omitted"]
    assert "removed_headings_omitted" not in created["knowledge_delta"]  # an empty list has nothing to give way
    assert len(history(target)[-1]["delta"]["added_headings"]) == 240
    original = store.read_knowledge_note(target)
    reply = tools._knowledge_write(ctx, "large-headings", "# Short\n", mode="edit",
                                   old_str=body, scope="global", expected_revision=original.revision)
    assert reply.startswith("✅")
    result = store.read_knowledge_note(target)
    assert result.text.endswith("# Short\n")
    assert len(history(target)[-1]["delta"]["removed_headings"]) == 240
    reported = json.loads(reply.split("\n", 1)[1])["knowledge_delta"]
    assert reported["removed_headings_count"] == 240
    assert reported["removed_headings_omitted"] is True
    assert reported["removed_sections_count"] == 240 and reported["removed_sections_omitted"] is True
    assert reported["added_headings"] == ["# Short"]  # the largest lists give way first; a short one stays whole
    assert reported["old_chars"] == len(original.text)


HISTORY_READ = {"tool": "knowledge_read", "view": "history"}  # of the note knowledge_source names


def _long_topic(length):
    """``length`` Cyrillic letters in path components of at most 100 (a file name holds 255 bytes, 2 a letter)."""
    letters = "".join("абвгдежзийклмнопрстуфхцчшщэюя"[index % 29] for index in range(length))
    return "/".join(letters[start:start + 100] for start in range(0, length, 100))


def _landed(result, target):
    """A landed write's reply: ✅, its receipt (the JSON line) names the note's new revision, and the host meta is that
    receipt byte for byte or, when even the bounded receipt is over the meta limit, only the write's reason, state
    and revision and the receipt's size and sha256. Returns the receipt and whether the host meta was reduced."""
    assert result.text.startswith("✅"), result.text[:300]
    line = result.text.split("\n", 2)[1]
    receipt, revision = json.loads(line), store.read_knowledge_note(target).revision
    assert receipt["knowledge_source"]["revision"] == revision, line[:300]
    reduced = {"knowledge_write_reason": "saved", "knowledge_state": "active", "knowledge_revision": revision,
               "knowledge_receipt": {"omitted": True, "chars": len(line),
                                     "sha256": hashlib.sha256(line.encode("utf-8")).hexdigest()}}
    published = json.dumps(dict(result.meta), ensure_ascii=False, sort_keys=True)
    assert published == line or dict(result.meta) == reduced, published[:600]
    return receipt, published != line


def test_an_omitted_delta_list_names_the_history_read_that_shows_it(tmp_path):
    """A receipt that cannot hold a delta list keeps its count and sha256, and one pointer for every omitted list,
    omitted_read, names the read that shows them: view='history' of the note knowledge_source names, its topic and
    scope not repeated. Folding a 40-section Cyrillic note: the receipt's JSON escapes each Cyrillic character to 6
    bytes, so the removed sections give way; view='history' lists them; the host meta is the receipt, byte for byte.
    Each call in a receipt repeats its topic, so swapping 300 headings for 300 others under a 130-character Cyrillic
    topic after a 16,000-character reason, or a 205-character one with none, landed and answered TOOL_ERROR."""
    registry = ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path)
    registry.set_context(ToolContext(repo_dir=tmp_path, drive_root=tmp_path, task_id="fold"))
    sections = "".join(f"## Раздел {index:02d}\n\n{'Первая строка раздела про реку и старый мост. ' * 3}\n\n"
                       for index in range(40))

    def write(topic, **arguments):
        return registry.execute_result("knowledge_write", {"topic": topic, "scope": "global", **arguments})

    target = address(tmp_path, "folds/река")
    _landed(write("folds/река", mode="overwrite", content="# Река\n\n" + sections), target)
    receipt, reduced = _landed(write("folds/река", mode="overwrite", content="# Река\n\nСвёрнуто: сорок разделов в одном.\n",
                                     expected_revision=store.read_knowledge_note(target).revision), target)
    delta = receipt["knowledge_delta"]
    assert not reduced and delta["removed_sections"] == [] and delta["removed_sections_count"] == 40, delta
    assert delta["removed_sections_omitted"] and delta["omitted_read"] == HISTORY_READ, delta
    assert [key for key in delta if key.endswith("_read")] == ["omitted_read"], delta
    source = receipt["knowledge_source"]["read"]
    listing = registry.execute(source["tool"], {**source["arguments"], "view": "history"})
    folded = next(line for line in listing.splitlines() if line.startswith("- "))  # newest first: the fold's row
    assert all(f"## Раздел {index:02d} (" in folded for index in range(40)), folded
    for length, reason in ((130, {"reason": "причина " * 2000}), (205, {})):
        topic = _long_topic(length)
        target = address(tmp_path, topic)
        _landed(write(topic, mode="overwrite", content="".join(f"## Старый {index:03d}\n" for index in range(300)), **reason),
                target)
        receipt, _reduced = _landed(write(topic, mode="overwrite",
                                          content="".join(f"## Новый {index:03d}\n" for index in range(300)),
                                          expected_revision=store.read_knowledge_note(target).revision), target)
        delta = receipt["knowledge_delta"]
        assert all(delta[f"{field}_count"] == 300 and delta[f"{field}_omitted"]
                   for field in ("removed_headings", "added_headings", "removed_sections")), delta
        assert delta["omitted_read"] == HISTORY_READ and receipt["knowledge_revert"]["arguments"]["topic"] == topic
        assert (receipt["knowledge_previous"]["reason"]["omitted"] if reason
                else receipt["knowledge_previous"]["reason"] == "unknown"), receipt["knowledge_previous"]


@pytest.mark.parametrize("length", [200, 320])
def test_a_fold_and_a_bare_append_under_a_long_cyrillic_topic_answer_with_their_receipt(tmp_path, length):
    """A fold of 60 sections and a bare append of 60 headings under a long Cyrillic topic: each lands and answers ✅
    with its omission markers and its new revision. At 320 characters nothing left to give fits the receipt in the
    meta limit, so the host meta keeps the write's facts and the reply text carries the receipt whole."""
    registry = ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path)
    registry.set_context(ToolContext(repo_dir=tmp_path, drive_root=tmp_path, task_id="long"))
    topic = _long_topic(length)
    target = address(tmp_path, topic)

    def write(**arguments):
        return registry.execute_result("knowledge_write", {"topic": topic, "scope": "global", **arguments})

    sections = "".join(f"## Раздел {index:02d}\n\n{'Первая строка раздела про реку. ' * 3}\n\n" for index in range(60))
    _landed(write(mode="overwrite", content="# Река\n\n" + sections), target)
    folded, fold_reduced = _landed(write(mode="overwrite", content="# Река\n\nСвёрнуто.\n",
                                         expected_revision=store.read_knowledge_note(target).revision), target)
    delta = folded["knowledge_delta"]
    assert delta["removed_sections_count"] == 60 and delta["removed_sections_omitted"], delta
    assert delta["omitted_read"] == HISTORY_READ, delta
    entries = "".join(f"## Запись {index:02d}\n\nТекст.\n\n" for index in range(60))
    appended = write(mode="append", content="\n" + entries)
    receipt, append_reduced = _landed(appended, target)
    assert receipt["knowledge_outline"]["omitted"] and receipt["knowledge_delta"]["added_headings_omitted"], receipt
    outline = knowledge_versions.render_outline(store.read_knowledge_note(target))
    assert appended.text.split("\n\n", 1)[1] == outline  # the bare appender's outline follows the JSON line, whole
    assert receipt["knowledge_revert"]["arguments"]["expected_revision"] == store.read_knowledge_note(target).revision
    if length == 320:
        assert fold_reduced and append_reduced


def test_edit_preserves_crlf_and_multibyte_surroundings(tmp_path):
    target = address(tmp_path, "legacy")
    target.path.parent.mkdir(parents=True)
    target.path.write_bytes("# Река\r\nСтарый берег.\r\nДо встречи.\r\n".encode("utf-8"))
    original = store.read_knowledge_note(target)
    changed = store.write_knowledge_note(target, "Новый берег.", mode="edit", old_str="Старый берег.",
                                         expected_revision=original.revision)
    assert changed.ok
    assert changed.current.raw == original.raw.replace("Старый берег.".encode(), "Новый берег.".encode())
    assert changed.current.metadata == original.metadata == {}


def test_edit_delta_counts_real_markdown_headings_including_duplicates(tmp_path):
    target = address(tmp_path, "headings")
    original = store.write_knowledge_note(target,
        "# Repeat\n\nOnly the first section.\n\n# Repeat\n\nOther section.\n\n"
        "Setext\n------\n\n```\n# Not a heading\n```\n").current
    first = store.write_knowledge_note(target, "", mode="edit",
                                       old_str="# Repeat\n\nOnly the first section.\n\n",
                                       expected_revision=original.revision)
    assert first.ok and first.delta["removed_headings"] == ["# Repeat"] and first.delta["added_headings"] == []
    second = store.write_knowledge_note(target, "", mode="edit", old_str="Setext\n------\n\n",
                                        expected_revision=first.current.revision)
    assert second.ok and second.delta["removed_headings"] == ["## Setext"]
    fenced = store.write_knowledge_note(target, "ordinary text", mode="edit", old_str="# Not a heading",
                                        expected_revision=second.current.revision)
    assert fenced.ok and fenced.delta["removed_headings"] == [] and fenced.delta["added_headings"] == []
    noop = store.write_knowledge_note(target, "ordinary text", mode="edit", old_str="ordinary text",
                                      expected_revision=fenced.current.revision)
    assert noop.ok and noop.reason == "unchanged" and noop.delta["change_chars"] == 0
    assert noop.delta["added_headings"] == []
    again = store.write_knowledge_note(target, "# Repeat\n\nBack.\n\n## Новый\n", mode="append")
    assert again.delta["added_headings"] == ["# Repeat", "## Новый"] and again.delta["removed_headings"] == []
    assert history(target)[-1]["delta"]["added_headings"] == ["# Repeat", "## Новый"]  # recorded at write time


def test_edit_is_an_opt_in_tool_mode_the_backlog_shares(tmp_path):
    ctx = ToolContext(repo_dir=tmp_path, drive_root=tmp_path)
    schema = next(entry.schema for entry in tools.get_tools() if entry.name == "knowledge_write")
    assert "edit" in schema["parameters"]["properties"]["mode"]["enum"]
    assert "old_str" in schema["parameters"]["properties"]
    backlog = address(tmp_path, "improvement-backlog")
    assert "edit_source_missing" in tools._knowledge_write(
        ctx, "improvement-backlog", "new", mode="edit", old_str="old")
    assert not backlog.path.exists()
    # Once written, the backlog takes the same revision-checked anchored edit as any note.
    assert tools._knowledge_write(ctx, "improvement-backlog", "- old item\n", mode="overwrite").startswith("✅")
    assert tools._knowledge_write(ctx, "improvement-backlog", "- new item", mode="edit", old_str="- old item",
                                  expected_revision=store.read_knowledge_note(backlog).revision).startswith("✅")
    assert store.read_knowledge_note(backlog).text.endswith("- new item\n")
    assert "old_str is used only" in tools._knowledge_write(
        ctx, "somewhere", "new", mode="append", old_str="old")
    assert "non-empty old_str" in tools._knowledge_write(ctx, "somewhere", "new", mode="edit", old_str="")
    with pytest.raises(ValueError, match="old_str is used only"):
        store.write_knowledge_note(address(tmp_path, "somewhere"), "new", old_str="old")


def test_nested_inventory_and_links_keep_addresses_not_display_identity(tmp_path):
    for topic in ("people/a", "people/b", "work/research"):
        assert store.write_knowledge_note(address(tmp_path, topic), f"---\ntype: note\ntitle: Same name\n---\n{topic}").ok
    first = store.read_knowledge_note(address(tmp_path, "people/a"))
    store.write_knowledge_note(address(tmp_path, "people/b", "project:p"), "Another person with the same name.")
    changed = store.write_knowledge_note(first.address, "[Known](b.md)\n[Unknown](../not-written.md)\n[Project](../../../projects/p/knowledge/people/b.md)", expected_revision=first.revision)
    rows = store.inventory_knowledge(first.address)
    assert {row["topic"] for row in rows} == {"people/a", "people/b", "work/research"}
    links = store.knowledge_links(changed.current)
    assert links[0]["address"]["topic"] == "people/b" and links[0]["status"] == "present"
    assert links[1]["status"] == "unwritten"
    assert links[2]["path"] == str(tmp_path / "projects" / "p" / "knowledge" / "people" / "b.md")
    assert links[2]["address"]["scope"] == "project:p"
    assert links[2]["status"] == "present"


def test_project_and_global_scope_refs_resolve_from_forked_context(tmp_path, monkeypatch):
    monkeypatch.setattr("ouroboros.config.DATA_DIR", tmp_path)
    ctx = ToolContext(repo_dir=tmp_path, drive_root=tmp_path / "child", project_id="p")
    ctx.budget_drive_root = str(tmp_path)
    tools._knowledge_write(ctx, "same", "Project fact.", mode="overwrite")
    tools._knowledge_write(ctx, "same", "Global understanding.", mode="overwrite", scope="global")
    local = store.read_knowledge_note(address(tmp_path, "same", "project:p"))
    shared = store.read_knowledge_note(address(tmp_path, "same", "global"))
    assert local.raw != shared.raw
    assert not (ctx.drive_root / "memory" / "knowledge").exists()
    for note in (local, shared):
        ref = note.source_ref()
        rendered = tools._knowledge_read(ctx, **ref["read"]["arguments"])
        assert note.revision in rendered and note.text in rendered
        assert ref["canonical_root"] == str(tmp_path)
    assert "Project fact." not in tools._knowledge_read(ctx, "same", scope="global")


def test_model_visible_read_body_and_revision_have_exact_metadata(tmp_path):
    ctx = ToolContext(repo_dir=tmp_path, drive_root=tmp_path)
    ctx._active_builtin_tool_result = None
    note = store.write_knowledge_note(address(tmp_path, "facts"), "# Facts\nExact body.").current
    rendered = tools._knowledge_read(ctx, "facts")
    result = ctx._active_builtin_tool_result
    assert result.meta["knowledge_source"]["revision"] == note.revision
    assert note.revision in rendered
    start, length = result.meta["knowledge_body_start"], result.meta["knowledge_body_chars"]
    assert rendered[start:start + length] == note.text
    assert rendered.endswith(note.text)


def test_malformed_legacy_yaml_is_readable_and_does_not_break_inventory(tmp_path):
    bad = address(tmp_path, "broken")
    bad.shelf.mkdir(parents=True)
    raw = b"---\ncustom: [unfinished\n---\n# Still readable\nOriginal evidence.\n"
    bad.path.write_bytes(raw)
    read = store.read_knowledge_note(bad)
    assert read.raw == raw and read.parse_error
    good = store.write_knowledge_note(address(tmp_path, "good"), "Useful note.")
    assert good.ok
    rows = store.inventory_knowledge(bad)
    assert {row["topic"] for row in rows} == {"broken", "good"}
    assert "source metadata unavailable" in (bad.shelf / store.INDEX_FILE).read_text(encoding="utf-8")
    ctx = ToolContext(repo_dir=tmp_path, drive_root=tmp_path)
    assert "Original evidence." in tools._knowledge_read(ctx, "broken")


def test_history_failure_leaves_previous_source_and_index_intact(tmp_path, monkeypatch):
    target = address(tmp_path)
    original = store.write_knowledge_note(target, "Known state.").current
    index = (target.shelf / store.INDEX_FILE).read_bytes()
    monkeypatch.setattr(store, "append_jsonl", lambda *a, **k: False)
    result = store.write_knowledge_note(target, "Not safely recorded.", expected_revision=original.revision)
    assert not result.ok and result.reason == "history_unavailable"
    assert target.path.read_bytes() == original.raw
    assert (target.shelf / store.INDEX_FILE).read_bytes() == index


def test_index_publication_failure_reports_actual_new_source_and_retains_capture(tmp_path, monkeypatch):
    target = address(tmp_path)
    original = store.write_knowledge_note(target, "Known state.").current

    def fail_index(_address):
        raise OSError("index cannot publish")

    monkeypatch.setattr(store, "rebuild_knowledge_index", fail_index)
    result = store.write_knowledge_note(target, "New state.", expected_revision=original.revision)
    assert not result.ok and result.reason == "publication_incomplete"
    assert result.current.raw == target.path.read_bytes()
    assert "New state." in result.current.text
    assert replaced(tmp_path, history(target)[-1]) == original.text
    assert published(tmp_path, history(target)[-1]) == result.current.text
    assert result.delta == history(target)[-1]["delta"]
    # The source landed, so that row holds a real version: the next write takes it as its base, byte for byte.
    monkeypatch.undo()
    landed = offsets(target)[-1]
    later = store.write_knowledge_note(target, "Later state.", expected_revision=result.current.revision)
    assert later.ok and "old_content" not in history(target)[-1] and history(target)[-1]["base"]["offset"] == landed
    assert replaced(tmp_path, history(target)[-1]) == result.current.text
    # A write whose source never landed holds a version nobody published: it is nobody's base.
    real_write = store.write_bytes_atomic

    def fail_source(path, data):
        if path == target.path:
            raise OSError("disk")
        return real_write(path, data)

    monkeypatch.setattr(store, "write_bytes_atomic", fail_source)
    lost = store.write_knowledge_note(target, "Never landed.", expected_revision=later.current.revision)
    assert not lost.ok and target.path.read_bytes() == later.current.raw
    monkeypatch.undo()
    store.write_knowledge_note(target, "After the failure.", expected_revision=later.current.revision)
    assert history(target)[-1]["base"]["offset"] == offsets(target)[-3]  # later's row, not the capture after it
    assert replaced(tmp_path, history(target)[-1]) == later.current.text


def test_overwrite_malformed_legacy_source_discloses_unknown_heading_delta(tmp_path):
    target = address(tmp_path, "malformed")
    target.path.parent.mkdir(parents=True)
    target.path.write_bytes(b"---\ncustom: [unfinished\n---\n# Old heading\n")
    old = store.read_knowledge_note(target)
    assert old.parse_error and old.source is None
    changed = store.write_knowledge_note(target, "---\ntype: note\n---\n# New heading\n",
                                         expected_revision=old.revision)
    assert changed.ok and changed.delta["removed_headings"] is None and changed.delta["added_headings"] is None
    assert history(target)[-1]["delta"]["removed_headings"] is None


def test_append_to_malformed_legacy_source_keeps_history_without_heading_claim(tmp_path):
    target = address(tmp_path, "malformed-append")
    target.path.parent.mkdir(parents=True)
    target.path.write_bytes(b"---\ncustom: [unfinished\n---\n# Old heading\n")
    original = store.read_knowledge_note(target)
    assert original.parse_error and original.source is None
    appended = store.write_knowledge_note(target, "New evidence.\n", mode="append")
    assert appended.ok and appended.current.raw.endswith(b"# Old heading\nNew evidence.\n")
    assert appended.delta["removed_headings"] is None
    record = history(target)[-1]
    assert replaced(tmp_path, record) == original.text
    assert published(tmp_path, record) == appended.current.text
    assert record["delta"] == appended.delta


@pytest.mark.parametrize("value", ["null", "12", "''"])
def test_new_formal_note_requires_string_type_without_a_type_catalog(tmp_path, value):
    target = address(tmp_path)
    result = store.write_knowledge_note(target, f"---\ntype: {value}\n---\nBody.")
    assert not result.ok and result.reason.startswith("invalid_note")
    assert not target.path.exists()


@pytest.mark.parametrize("topic", ["../escape", "index-full", "a\\b"])
def test_invalid_address_is_not_normalized_into_another_source(tmp_path, topic):
    with pytest.raises(ValueError):
        address(tmp_path, topic)


def test_missing_read_and_list_create_no_state(tmp_path):
    ctx = ToolContext(repo_dir=tmp_path, drive_root=tmp_path / "not-created")
    assert "not found" in tools._knowledge_read(ctx, "missing")
    assert "empty" in tools._knowledge_list(ctx)
    assert not ctx.drive_root.exists()


def test_legacy_index_context_survives_until_authored_overview_without_recursive_growth(tmp_path):
    target = address(tmp_path, "details")
    target.shelf.mkdir(parents=True)
    original = "# Knowledge Base Index\n\n- **old**: A useful older understanding.\n"
    index = target.shelf / store.INDEX_FILE
    index.write_bytes(original.encode("utf-8"))
    assert store.write_knowledge_note(target, "New detailed evidence.").ok
    for topic in ("second", "third"):
        assert store.write_knowledge_note(address(tmp_path, topic), "A detail.").ok
        text = index.read_text(encoding="utf-8")
        assert text.count(original) == 1
        assert text.count("## Earlier generated context") == 1
        assert "historical context, not current authored summaries" in text
    assert sum(row.get("type") == "knowledge_index_source" for row in history(target)) == 1
    assert store.write_knowledge_note(address(tmp_path, "overview"),
                                      "# What I currently understand\n\nA reconsidered overview with [details](details.md).").ok
    assert "Earlier generated context" not in index.read_text(encoding="utf-8")
    assert any(row.get("old_content") == original for row in history(target))


@pytest.mark.parametrize("scope", ["global", "project:p"])
def test_history_stores_each_version_once_as_a_change_against_the_row_that_holds_it(tmp_path, scope):
    target = address(tmp_path, "work/report", scope)
    first = store.write_knowledge_note(target, "# Report\n\nFirst understanding.")
    second = store.write_knowledge_note(target, "# Report\n\nSecond understanding.",
                                        expected_revision=first.current.revision)
    rows = history(target)
    # The replaced version is already the first row's text: the second row names that row and carries the change.
    assert {"old_content", "new_content", "old_content_ref"}.isdisjoint(rows[-1])
    assert rows[-1]["base"] == {"offset": offsets(target)[-2], "sha256": first.current.revision}
    assert rows[-1]["change"] == [list(item) for item in text_change.exact_change(first.current.text, second.current.text)]
    assert replaced(tmp_path, rows[-1]) == first.current.text and published(tmp_path, rows[-1]) == second.current.text
    assert hashlib.sha256(replaced(tmp_path, rows[-1]).encode()).hexdigest() == rows[-1]["old_sha256"]
    # A newest row that holds no text for those bytes (a digest-era row) is no base: the older row that reads back is.
    journal = target.shelf.parent / "knowledge_history.jsonl"
    digest = {key: value for key, value in rows[-1].items() if key not in ("base", "change", "chain_bytes")}
    for row in (digest, {**rows[-1], "new_content": ""}):  # as the digest compaction spelled it; and on a change row
        with journal.open("ab") as stream:  # appended byte-exactly, as the writer appends: every offset holds
            stream.write((json.dumps({**row, "new_content": "", "content_digested": True}, ensure_ascii=False) + "\n")
                         .encode("utf-8"))
    again = store.write_knowledge_note(target, "# Report\n\nSecond, again.", expected_revision=second.current.revision)
    assert again.ok and history(target)[-1]["base"]["offset"] == offsets(target)[-4]
    assert replaced(tmp_path, history(target)[-1]) == second.current.text
    # A version no row recorded (changed outside the writer) stays inline, as does an archive's.
    target.path.write_bytes(b"# Report\n\nEdited by hand.")
    third = store.write_knowledge_note(target, "# Report\n\nThird.", expected_revision=store.read_knowledge_note(target).revision)
    assert third.ok and history(target)[-1]["old_content"] == "# Report\n\nEdited by hand."
    assert "base" not in history(target)[-1] and history(target)[-1]["new_content"] == third.current.text
    archived = store.write_knowledge_note(target, "", "archive", third.current.revision, reason="superseded")
    assert archived.ok and history(target)[-1]["old_content"] == third.current.text
    assert {"old_content_ref", "base", "change"}.isdisjoint(history(target)[-1])


def test_history_finds_the_base_in_a_tail_that_starts_mid_journal_and_keeps_both_texts_past_it(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "_JOURNAL_TAIL_BYTES", 8192)
    target = address(tmp_path, "work/report")
    for index in range(12):  # other notes push the journal past the tail the writer reads
        store.write_knowledge_note(address(tmp_path, f"filler/{index}"), "x" * 1500)
    first = store.write_knowledge_note(target, "# Report\n\nFirst.")
    assert (target.shelf.parent / "knowledge_history.jsonl").stat().st_size > 8192
    second = store.write_knowledge_note(target, "# Report\n\nSecond.", expected_revision=first.current.revision)
    row = history(target)[-1]
    assert row["base"]["offset"] == offsets(target)[-2] > 0 and replaced(tmp_path, row) == first.current.text
    for index in range(12):  # the base falls out of the tail: the next row keeps both texts whole
        store.write_knowledge_note(address(tmp_path, f"later/{index}"), "y" * 1500)
    store.write_knowledge_note(target, "# Report\n\nThird.", expected_revision=second.current.revision)
    assert history(target)[-1]["old_content"] == second.current.text and "base" not in history(target)[-1]


def _files(root):
    return {path: path.read_bytes() for path in root.rglob("*") if path.is_file() and not path.name.endswith(".lock")}


def _meta(reply):
    assert reply.startswith("✅"), reply
    return json.loads(reply.split("\n", 1)[1])


def _item(old_str=None, new_str=""):
    """An item as a strict provider sends it: both fields present."""
    return {"old_str": old_str, "new_str": new_str}


def test_several_exact_items_land_in_one_revision_or_none_does(tmp_path):
    ctx = ToolContext(repo_dir=tmp_path, drive_root=tmp_path, task_id="editor")
    target = address(tmp_path, "people/rowan")
    original = store.write_knowledge_note(target, "# Rowan\n\n- Host: old-a.\n" + "Filler line.\n" * 40
                                          + "- Port: 8080.\n").current
    rows, before = len(history(target)), _files(tmp_path)
    refused = tools._knowledge_write(ctx, "people/rowan", mode="edit", scope="global", expected_revision=original.revision,
                                     edits=[_item("old-a", new_str="new-a"), _item("Absent.", new_str="x")])
    assert "edit_refused: item 2 (old_str): not found" in refused and "Nothing was written." in refused
    assert original.text not in refused and "Outline of" not in refused  # the note is not resent for a quoted item
    assert _files(tmp_path) == before
    _meta(tools._knowledge_write(ctx, "people/rowan", mode="edit", scope="global", expected_revision=original.revision,
                                 edits=[_item("old-a", new_str="new-a"), _item("8080", new_str="9090")]))
    updated = store.read_knowledge_note(target)
    assert updated.text == original.text.replace("old-a", "new-a").replace("8080", "9090")
    assert len(history(target)) == rows + 1 and published(tmp_path, history(target)[-1]) == updated.text
    assert history(target)[-1]["mode"] == "edit" and replaced(tmp_path, history(target)[-1]) == original.text


def test_items_are_found_in_the_original_and_may_touch_but_never_overlap(tmp_path):
    target = address(tmp_path, "swap")
    original = store.write_knowledge_note(target, "# Swap\n\nalpha beta gamma\n").current
    swapped = store.write_knowledge_note(target, "", "edit", original.revision,
                                         edits=[_item("alpha", new_str="beta"), _item("beta", new_str="alpha")])
    assert swapped.ok and swapped.current.text.endswith("beta alpha gamma\n")  # no item sees another's result
    touching = store.write_knowledge_note(target, "", "edit", swapped.current.revision,
                                          edits=[_item("beta ", new_str="B "), _item("alpha", new_str="A")])
    assert touching.ok and touching.current.text.endswith("B A gamma\n")
    overlapping = store.write_knowledge_note(target, "", "edit", touching.current.revision,
                                             edits=[_item("B A", new_str="x"), _item("A gamma", new_str="y")])
    assert overlapping.reason == "edit_refused: item 2: overlaps item 1"
    assert target.path.read_bytes() == touching.current.raw


def test_a_refused_item_says_whether_its_text_is_missing_or_where_it_repeats(tmp_path):
    target = address(tmp_path, "lines")
    original = store.write_knowledge_note(target, "---\ntype: note\n---\n# Lines\n\nrepeat\nonce\nrepeat\n"
                                          + "x\n" * 10).current
    assert store.write_knowledge_note(target, "", "edit", original.revision, edits=[
        _item("once", new_str="1"), _item("repeat", new_str="2")]).reason == (
        "edit_refused: item 2 (old_str): found 2 times at lines 6, 8")
    assert store.write_knowledge_note(target, "", "edit", original.revision, edits=[
        _item("x\n", new_str="")]).reason == "edit_refused: item 1 (old_str): found 10 times at lines 9, 10, 11, 12, 13, 14, 15, 16"
    assert store.write_knowledge_note(target, "", "edit", original.revision, edits=[
        _item("missing", new_str="")]).reason == "edit_refused: item 1 (old_str): not found"
    assert store.write_knowledge_note(target, "", "edit", original.revision, edits=[_item("once", new_str="1")]).ok


def test_an_item_whose_text_is_only_in_the_frontmatter_says_where_it_is(tmp_path):
    """Items search the body. Text the mind saw in the note's summary (in the index or a read) is not there: the
    refusal says it occurs in the frontmatter and how a summary is revised, also when YAML wrapped the summary
    across lines. Text that is nowhere in the note is still plain "not found"."""
    registry = ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path)
    registry.set_context(ToolContext(repo_dir=tmp_path, drive_root=tmp_path, task_id="items"))
    summary = ("Нина — коллега из ML daily; ведёт ревью моделей ранжирования и спрашивает про метрики каждую среду, "
               "до планёрки.")

    def write(**arguments):
        return registry.execute("knowledge_write", {"topic": "people/nina", "scope": "global", **arguments})

    _meta(write(mode="overwrite", content="# Нина\n\nРаботает над ранжированием.\n", summary=summary))
    note = store.read_knowledge_note(address(tmp_path, "people/nina"))
    assert note.summary == summary and summary not in note.text  # the writer's YAML wraps a long summary
    in_front = "not found in the body; it occurs in the frontmatter (a summary is revised with summary=...)"
    before = _files(tmp_path)
    for old_str, said in (("коллега из ML daily", in_front), (summary, in_front), ("type: note", in_front),
                          ("Отсутствует.", "not found"), ("Нина — коллега из ML weekly", "not found")):
        reply = write(mode="edit", expected_revision=note.revision, edits=[_item("ранжированием", new_str="поиском"),
                                                                           _item(old_str, new_str="x")])
        assert reply == (f"⚠️ TOOL_ERROR: Knowledge write not completed: edit_refused: item 2 (old_str): {said}. "
                         "Nothing was written."), (old_str, reply)
    assert _files(tmp_path) == before
    _meta(write(mode="edit", expected_revision=note.revision, edits=[_item("ранжированием", new_str="поиском")]))
    _meta(write(mode="edit", expected_revision=store.read_knowledge_note(note.address).revision,
                summary=summary.replace("ML daily", "ML weekly")))
    assert store.read_knowledge_note(note.address).summary.startswith("Нина — коллега из ML weekly;")


def test_items_keep_crlf_multibyte_and_a_missing_final_newline_byte_exact(tmp_path):
    target = address(tmp_path, "legacy")
    target.path.parent.mkdir(parents=True)
    raw = "# Река\r\nСтарый берег 🌊.\r\nДо встречи.\r\n## Хвост\r\nбез перевода строки".encode("utf-8")
    target.path.write_bytes(raw)
    original = store.read_knowledge_note(target)
    changed = store.write_knowledge_note(target, "", "edit", original.revision,
                                         edits=[_item("Старый", new_str="Новый"), _item("строки", new_str="строк")])
    assert changed.ok and changed.current.raw == raw.replace("Старый".encode(), "Новый".encode()).replace(
        "строки".encode(), "строк".encode())


def test_item_shapes_and_spellings_that_cannot_mean_one_thing_are_argument_errors(tmp_path):
    """An item is {old_str, new_str}, both text: a section selector is no item field (it was dropped)."""
    ctx = ToolContext(repo_dir=tmp_path, drive_root=tmp_path, task_id="editor")
    original = store.write_knowledge_note(address(tmp_path, "shape"), "# Shape\n\nText.\n").current
    for kwargs, message in (
            ({"edits": [{"old_str": None, "section": "# Shape", "new_str": "x"}]}, "old_str and new_str only"),
            ({"edits": [{"old_str": "Text.", "section": None, "new_str": "x"}]}, "old_str and new_str only"),
            ({"edits": [_item(new_str="x")]}, "item 1: old_str must be non-empty exact body text"),
            ({"edits": [_item("", new_str="x")]}, "item 1: old_str must be non-empty exact body text"),
            ({"edits": [{"old_str": "Text."}]}, "item 1: new_str must be text"),
            ({"edits": [_item("Text.", new_str=None)]}, "item 1: new_str must be text"),
            ({"edits": []}, "mode=edit requires edits, or a non-empty old_str, or a summary (edits=[] ignored"),
            ({"edits": [_item("Text.", new_str="x")], "old_str": "Text."}, "edits carries its own replacements"),
            ({"edits": [_item("Text.", new_str="x")], "content": "x"}, "edits carries its own replacements")):
        reply = tools._knowledge_write(ctx, "shape", mode="edit", scope="global", expected_revision=original.revision, **kwargs)
        assert reply.startswith("⚠️ TOOL_ARG_ERROR") and message in reply, (kwargs, reply)
    assert "edits is used only with mode=edit" in tools._knowledge_write(
        ctx, "shape", "x", mode="overwrite", scope="global", expected_revision=original.revision, edits=[_item("x")])
    assert store.read_knowledge_note(original.address).raw == original.raw
    schema = next(entry for entry in tools.get_tools() if entry.name == "knowledge_write").schema
    assert schema["parameters"]["properties"]["edits"]["items"] == {
        "type": "object", "additionalProperties": False, "required": ["old_str", "new_str"],
        "properties": {"old_str": {"type": "string"}, "new_str": {"type": "string"}}}
    assert "section" not in json.dumps(schema["parameters"]["properties"]["edits"])
    _meta(tools._knowledge_write(ctx, "shape", mode="edit", scope="global", expected_revision=original.revision,
                                 edits=[_item("Text.", new_str="Prose.")]))
    assert store.read_knowledge_note(original.address).text.endswith("# Shape\n\nProse.\n")


def test_empty_optional_write_values_take_the_omitted_path_and_say_so(tmp_path):
    """A model may fill every optional field with an empty value: version='' outside revert, edits=[] and
    old_str='' outside edit ask for nothing, so the write goes ahead and its receipt names what was ignored;
    a value that asks for something (a version, an item, a text to replace) still refuses, typed."""
    ctx = ToolContext(repo_dir=tmp_path, drive_root=tmp_path, task_id="filler")
    registry = ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path)
    registry.set_context(ctx)
    note = store.write_knowledge_note(address(tmp_path, "filled"), "# Filled\n\nOne.\n").current
    empty = {"version": "", "edits": [], "old_str": "", "summary": "", "reason": ""}
    ignored = ["version='' ignored: only mode=revert takes a version",
               "edits=[] ignored: an empty list asks for no replacement",
               "old_str='' ignored: only mode=edit replaces text"]

    def write(**arguments):
        return registry.execute("knowledge_write", {"topic": "filled", "scope": "global", **arguments})

    overwrite = _meta(write(mode="overwrite", content="# Filled\n\nTwo.\n", expected_revision=note.revision, **empty))
    assert overwrite["ignored_arguments"] == ignored
    assert store.read_knowledge_note(note.address).text.endswith("# Filled\n\nTwo.\n")
    assert _meta(write(mode="append", content="Three.\n", **empty))["ignored_arguments"] == ignored
    current = store.read_knowledge_note(note.address)
    edited = _meta(write(mode="edit", old_str="Two.", content="2.", expected_revision=current.revision,
                         version="", edits=[]))
    assert edited["ignored_arguments"] == ignored[:2] and store.read_knowledge_note(note.address).text.endswith("2.\nThree.\n")
    current = store.read_knowledge_note(note.address)
    assert "ignored_arguments" not in _meta(write(mode="append", content="Four.\n"))
    current, before = store.read_knowledge_note(note.address), _files(tmp_path)
    for arguments, refusal in (
            ({"mode": "overwrite", "content": "x", "version": "abc"}, "version is used only with mode=revert"),
            ({"mode": "append", "content": "x", "edits": [_item("Four.", "4.")]}, "edits is used only with mode=edit"),
            ({"mode": "append", "content": "x", "old_str": "Four."}, "old_str is used only with mode=edit"),
            ({"mode": "revert", "version": "", "expected_revision": current.revision}, "mode=revert requires version"),
            ({"mode": "edit", "edits": [], "expected_revision": current.revision},
             "mode=edit requires edits, or a non-empty old_str, or a summary (edits=[] ignored")):
        reply = write(**arguments)
        assert reply.startswith("⚠️ TOOL_ARG_ERROR") and refusal in reply, (arguments, reply)
    assert _files(tmp_path) == before


def _model_call(registry, arguments: str):
    """One tool call as a model sends it: JSON arguments through the loop's own dispatch."""
    from ouroboros.loop_tool_execution import _execute_single_tool

    call = {"id": "mode-call", "function": {"name": "knowledge_write", "arguments": arguments}}
    return _execute_single_tool(registry, call, registry._ctx.drive_root.parent / "logs", registry._ctx.task_id)


def test_a_write_without_an_exact_mode_writes_nothing(tmp_path):
    """There is no default operation. An omitted mode once meant overwrite, so a call meant to add a section replaced
    the whole note (the 13 Nov Kimi turn: content, expected_revision and topic, then version alone, five times).
    Omission, null, blank text and any value that is not exactly an operation are refused typed before anything is
    read or written: note, history and index keep every byte; the operation, once named, does what it says."""
    drive = tmp_path / "drive"
    registry = ToolRegistry(repo_dir=tmp_path, drive_root=drive)
    registry.set_context(ToolContext(repo_dir=tmp_path, drive_root=drive, task_id="modeless"))
    target = address(drive, "people/polina")
    first = store.write_knowledge_note(target, "# Полина\n\n## Работа\n\nВыгрузки и датасеты.\n").current
    held = store.write_knowledge_note(target, first.text + "\n## Встречи\n\nДейли по средам.\n",
                                      expected_revision=first.revision).current
    news = "\n## Новость\n\nНовый запрос к выгрузкам.\n"
    shapes = {"omitted": {}, "null": {"mode": None}, "empty": {"mode": ""}, "space": {"mode": " "},
              "long blank": {"mode": " " * 9000}, "capitalized": {"mode": "Append"}, "padded": {"mode": " append"},
              "trailing": {"mode": "append "}, "unknown": {"mode": "replace"}, "number": {"mode": 1},
              "list": {"mode": ["append"]}, "object": {"mode": {"op": "append"}}, "boolean": {"mode": True}}
    received = {"omitted": "mode is missing or null", "null": "mode is missing or null",
                "empty": "mode='' is not an operation", "space": "mode=' ' is not an operation",
                "long blank": "mode=<9000 whitespace characters> is not an operation",
                "capitalized": "mode='Append' is not an operation", "padded": "mode=' append' is not an operation",
                "trailing": "mode='append ' is not an operation", "unknown": "mode='replace' is not an operation",
                "number": "mode=1 is not text", "list": "mode=['append'] is not text",
                "object": "mode={'op': 'append'} is not text", "boolean": "mode=True is not text"}
    before = _files(drive / "memory")
    for name, mode in shapes.items():
        for call in ({"topic": "people/polina", "scope": "global", "content": news, "expected_revision": held.revision},
                     {"topic": "people/polina", "scope": "global", "version": first.revision}):
            executed = _model_call(registry, json.dumps({**call, **mode}))
            result, reply = executed["tool_result"], executed["result"]
            assert executed["is_error"] and (result.status, result.code) == ("error", "TOOL_ARG_ERROR"), (name, reply)
            assert result.meta.get("operation_outcome") == "completed_no_effect", (name, result.meta)
            assert reply.startswith(f"⚠️ TOOL_ARG_ERROR: {received[name]}. knowledge_write has no default operation: "
                                    "pass mode as exactly one of overwrite"), (name, reply)
            assert reply.endswith(" Nothing was written."), (name, reply)
    assert ("revert (puts back the earlier version named by version)" in reply
            and "expected_revision is the current revision a replacing write checks" in reply), reply
    assert _files(drive / "memory") == before  # the note, its history journal and the index, byte for byte
    # Named, the operation is the one asked for: the same call appends the section and loses nothing.
    appended = _model_call(registry, json.dumps({"topic": "people/polina", "scope": "global", "mode": "append",
                                                 "content": news, "expected_revision": held.revision}))
    assert not appended["is_error"], appended["result"]
    assert target.path.read_bytes() == held.raw + news.encode("utf-8")
    assert [row["mode"] for row in history(target)] == ["overwrite", "overwrite", "append"]


def test_every_named_mode_keeps_its_operation_and_history_through_the_model_call_path(tmp_path):
    """Each documented operation, named explicitly, still does what it did: create and replace with overwrite, append,
    edit, revert to an exact earlier version, archive and restore, each with its own history row."""
    drive = tmp_path / "drive"
    registry = ToolRegistry(repo_dir=tmp_path, drive_root=drive)
    registry.set_context(ToolContext(repo_dir=tmp_path, drive_root=drive, task_id="named"))
    target = address(drive, "work/modes")

    def write(**arguments):
        executed = _model_call(registry, json.dumps({"topic": "work/modes", "scope": "global", **arguments}))
        assert not executed["is_error"], (arguments, executed["result"])
        return store.read_knowledge_note(target)

    created = write(mode="overwrite", content="# Modes\n\nOne.\n")
    appended = write(mode="append", content="Two.\n")
    assert appended.raw == created.raw + b"Two.\n"
    edited = write(mode="edit", old_str="One.", content="First.", expected_revision=appended.revision)
    assert edited.text.endswith("# Modes\n\nFirst.\nTwo.\n")
    replaced = write(mode="overwrite", content="# Modes\n\nReplaced on purpose.\n", expected_revision=edited.revision)
    assert replaced.text.endswith("# Modes\n\nReplaced on purpose.\n") and "Two." not in replaced.text
    reverted = write(mode="revert", version=edited.revision, expected_revision=replaced.revision)
    assert reverted.raw == edited.raw and reverted.revision == edited.revision
    archived = write(mode="archive", reason="kept in the overview", expected_revision=reverted.revision)
    assert archived.state == "archived" and archived.text.endswith("# Modes\n\nFirst.\nTwo.\n")
    restored = write(mode="restore", expected_revision=archived.revision)
    assert restored.state == "active" and restored.text.endswith("# Modes\n\nFirst.\nTwo.\n")
    assert [row["mode"] for row in history(target)] == [
        "overwrite", "append", "edit", "overwrite", "revert", "archive", "restore"]
    fresh = address(drive, "work/appended")  # append creates a missing note, as overwrite does
    assert not _model_call(registry, json.dumps({"topic": "work/appended", "scope": "global", "mode": "append",
                                                 "content": "Started by an append.\n"}))["is_error"]
    assert store.read_knowledge_note(fresh).text.endswith("Started by an append.\n")


def test_a_null_reason_takes_the_omitted_path_and_archive_still_requires_one(tmp_path):
    """reason is optional text outside archive, so a null one means what omission means: the write goes ahead,
    records no reason and says so. Archive still requires a non-empty reason and refuses, writing nothing."""
    registry = ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path)
    registry.set_context(ToolContext(repo_dir=tmp_path, drive_root=tmp_path, task_id="null"))
    target = address(tmp_path, "null-reason")
    first = store.write_knowledge_note(target, "# Null\n\nOne.\n").current
    said, rows = ["reason=None ignored: no reason is recorded"], len(history(target))

    def write(**arguments):
        revision = store.read_knowledge_note(target).revision
        return registry.execute("knowledge_write", {"topic": "null-reason", "scope": "global", "reason": None,
                                                    "expected_revision": revision, **arguments})

    for arguments in ({"mode": "edit", "old_str": "One.", "content": "Two."}, {"mode": "append", "content": "Three.\n"},
                      {"mode": "overwrite", "content": "# Null\n\nFour.\n"}, {"mode": "revert", "version": first.revision}):
        assert _meta(write(**arguments))["ignored_arguments"] == said, arguments
    assert target.path.read_bytes() == first.raw and len(history(target)) == rows + 4
    assert not any("reason" in row for row in history(target)[rows:])
    before = _files(tmp_path)
    for reason in (None, "", "  "):
        reply = write(mode="archive", reason=reason)
        assert reply.startswith("⚠️ TOOL_ARG_ERROR: reason is required for archive"), (reason, reply)
    assert _files(tmp_path) == before
    assert "ignored_arguments" not in _meta(write(mode="archive", reason="kept elsewhere"))
    assert _meta(write(mode="restore"))["ignored_arguments"] == said
    assert store.read_knowledge_note(target).state == "active"


def test_a_blank_version_outside_revert_takes_the_omitted_path_and_a_revert_still_needs_one(tmp_path):
    """A model fills version with whitespace as well as with nothing (the experiment saw ' ' 18 times): a blank
    value names no revision, so outside revert it is ignored and said, a long one by its length (9,000 spaces
    echoed whole overflowed the receipt of the write that had landed); inside revert 'requires version' stays,
    and a non-blank version outside revert still refuses, typed."""
    registry = ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path)
    registry.set_context(ToolContext(repo_dir=tmp_path, drive_root=tmp_path, task_id="blank"))
    target = address(tmp_path, "blank")
    store.write_knowledge_note(target, "# Blank\n\nOne.\n")

    def write(**arguments):
        return registry.execute("knowledge_write", {"topic": "blank", "scope": "global", **arguments})

    for blank, echo, old, new in ((" ", "' '", "One.", "Two."), ("\t\n ", r"'\t\n '", "Two.", "Three."),
                                  (" " * 9000, "<9000 whitespace characters>", "Three.", "Third.")):
        revision = store.read_knowledge_note(target).revision
        saved = _meta(write(mode="edit", old_str=old, content=new, expected_revision=revision, version=blank))
        assert saved["ignored_arguments"] == [f"version={echo} ignored: only mode=revert takes a version"], saved
    current = store.read_knowledge_note(target)
    assert current.text.endswith("# Blank\n\nThird.\n")
    appended = _meta(write(mode="append", content="Four.\n", version=" "))
    assert appended["ignored_arguments"] == ["version=' ' ignored: only mode=revert takes a version"]
    current, before = store.read_knowledge_note(target), _files(tmp_path)
    for arguments, refusal in (
            ({"mode": "revert", "version": " ", "expected_revision": current.revision}, "mode=revert requires version"),
            ({"mode": "revert", "version": "\n", "expected_revision": current.revision}, "mode=revert requires version"),
            ({"mode": "edit", "old_str": "Four.", "content": "4.", "version": " abc ",
              "expected_revision": current.revision}, "version is used only with mode=revert")):
        reply = write(**arguments)
        assert reply.startswith("⚠️ TOOL_ARG_ERROR") and refusal in reply, (arguments, reply)
    assert _files(tmp_path) == before


def test_a_version_equal_to_the_calls_expected_revision_asks_for_nothing_outside_revert(tmp_path):
    """A model passed version equal to its expected_revision on mode=edit: that names the revision the call already
    holds, so outside revert it takes the omitted path, said; any other version there still refuses, typed, and
    inside revert the version stays the revision to put back."""
    registry = ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path)
    registry.set_context(ToolContext(repo_dir=tmp_path, drive_root=tmp_path, task_id="same"))
    target = address(tmp_path, "same")
    first = store.write_knowledge_note(target, "# Same\n\nOne.\n").current

    def write(**arguments):
        return registry.execute("knowledge_write", {"topic": "same", "scope": "global", **arguments})

    def ignored(revision):
        return [f"version={revision!r} ignored: the same as expected_revision; only mode=revert takes a version"]

    edited = _meta(write(mode="edit", old_str="One.", content="Two.", expected_revision=first.revision,
                         version=first.revision))
    assert edited["ignored_arguments"] == ignored(first.revision)
    second = store.read_knowledge_note(target)
    assert second.text.endswith("# Same\n\nTwo.\n")
    overwritten = _meta(write(mode="overwrite", content="# Same\n\nThree.\n", expected_revision=second.revision,
                              version=second.revision))
    assert overwritten["ignored_arguments"] == ignored(second.revision)
    third, before = store.read_knowledge_note(target), _files(tmp_path)
    for arguments in ({"mode": "edit", "old_str": "Three.", "content": "Four.", "expected_revision": third.revision,
                       "version": first.revision},  # an earlier revision: a revert, not this edit
                      {"mode": "append", "content": "Four.\n", "version": third.revision}):  # no expected_revision
        reply = write(**arguments)
        assert reply.startswith("⚠️ TOOL_ARG_ERROR") and "version is used only with mode=revert" in reply, reply
    assert _files(tmp_path) == before
    same = write(mode="revert", version=third.revision, expected_revision=third.revision)
    assert same.startswith("✅ Knowledge 'same' unchanged (revert)") and "ignored_arguments" not in same, same
    _meta(write(mode="revert", version=first.revision, expected_revision=third.revision))
    assert target.path.read_bytes() == first.raw


def test_the_one_pair_edit_takes_its_replacement_as_content_or_new_str(tmp_path):
    """Other editors spell the replacement new_str: with mode=edit and old_str it is exactly content; an empty one
    beside a non-empty one is filler, ignored and said, either way round; both empty delete; both non-empty and
    different is an argument error; elsewhere an empty new_str is ignored and a non-empty one names content."""
    ctx = ToolContext(repo_dir=tmp_path, drive_root=tmp_path, task_id="pair")
    registry = ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path)
    registry.set_context(ctx)
    target = address(tmp_path, "pair")
    store.write_knowledge_note(target, "# Pair\n\nOld one.\nOld two.\nOld three.\n")

    def write(**arguments):
        revision = store.read_knowledge_note(target).revision
        return registry.execute("knowledge_write", {"topic": "pair", "scope": "global", "mode": "edit",
                                                    "expected_revision": revision, **arguments})

    assert "ignored_arguments" not in _meta(write(old_str="Old one.", new_str="New one."))
    _meta(write(old_str="Old two.", content="New two."))
    _meta(write(old_str="New two.", content="Newer two.", new_str="Newer two."))  # the same text twice is one
    filled = _meta(write(old_str="Old three.", content="", new_str="New three."))
    assert filled["ignored_arguments"] == ["content='' ignored: new_str gives the replacement"]
    mirror = _meta(write(old_str="New three.", content="Newest three.", new_str=""))  # the mirror spelling
    assert mirror["ignored_arguments"] == ["new_str='' ignored: content gives the replacement"]
    _meta(write(old_str="New one.\n", new_str=""))  # an empty new_str deletes, as empty content does
    assert store.read_knowledge_note(target).text.endswith("# Pair\n\nNewer two.\nNewest three.\n")
    both = _meta(write(old_str="Newer two.\n", content="", new_str=""))  # both empty: a deletion, nothing ignored
    assert "ignored_arguments" not in both and store.read_knowledge_note(target).text.endswith("# Pair\n\nNewest three.\n")
    before = _files(tmp_path)
    items = "each edits item carries its own new_str; a top-level new_str belongs to the one-pair spelling with old_str"
    for arguments, message in (
            ({"old_str": "Newest three.", "content": "A", "new_str": "B"}, "new_str and content both give"),
            ({"summary": "The pair.", "new_str": "B"}, "same as content; pass old_str with it"),
            ({"edits": [_item("Newest three.", new_str="A")], "new_str": "B"}, items),  # not "pass old_str with it"
            ({"edits": [_item("Newest three.", new_str="A")], "old_str": "Newest three.", "new_str": "B"}, items),
            ({"mode": "append", "content": "x", "new_str": "y"}, "mode=append takes its text as content"),
            ({"mode": "overwrite", "new_str": "# Pair\n"}, "mode=overwrite takes its text as content"),
            ({"mode": "rewrite", "new_str": "B"}, "mode='rewrite' is not an operation")):  # read before new_str
        reply = write(**arguments)
        assert reply.startswith("⚠️ TOOL_ARG_ERROR") and message in reply, (arguments, reply)
    assert _files(tmp_path) == before
    appended = _meta(registry.execute("knowledge_write", {"topic": "pair", "scope": "global", "mode": "append",
                                                          "content": "Tail.\n", "new_str": ""}))
    assert appended["ignored_arguments"] == ["new_str='' ignored: only an edit with old_str takes a replacement"]
    schema = next(entry for entry in tools.get_tools() if entry.name == "knowledge_write").schema
    assert schema["parameters"]["properties"]["new_str"]["type"] == "string"
    assert "one old_str by content (or new_str)" in schema["description"]


def test_revert_publishes_the_exact_bytes_of_an_earlier_version(tmp_path):
    ctx = ToolContext(repo_dir=tmp_path, drive_root=tmp_path, task_id="editor")
    target = address(tmp_path, "shore")
    target.path.parent.mkdir(parents=True)
    first_raw = "# Река\r\nСтарый берег.\r\n".encode("utf-8")
    target.path.write_bytes(first_raw)  # written outside the writer: only the next row's old_content holds it
    first = store.read_knowledge_note(target)
    second = store.write_knowledge_note(target, "# Река\r\nНовый берег.\r\n", expected_revision=first.revision).current
    assert "revision_required" in tools._knowledge_write(ctx, "shore", mode="revert", scope="global", version=first.revision)
    assert target.path.read_bytes() == second.raw
    _meta(tools._knowledge_write(ctx, "shore", mode="revert", scope="global", version=first.revision,
                                 expected_revision=second.revision, reason="the old shore was right"))
    restored = store.read_knowledge_note(target)
    assert restored.raw == first_raw and restored.revision == first.revision
    row = history(target)[-1]
    assert (row["mode"], row["reverted_from"], row["reason"]) == ("revert", first.revision, "the old shore was right")
    assert "old shore" not in restored.text and published(tmp_path, row) == first_raw.decode("utf-8")
    same = tools._knowledge_write(ctx, "shore", mode="revert", scope="global", version=first.revision,
                                  expected_revision=first.revision)
    assert same.startswith("✅ Knowledge 'shore' unchanged (revert)")
    for extra in ({"content": "x"}, {"summary": "s"}, {"old_str": "x"}, {"edits": [_item("x")]}):
        assert tools._knowledge_write(ctx, "shore", mode="revert", scope="global", version=second.revision,
                                      expected_revision=first.revision, **extra).startswith("⚠️ TOOL_ARG_ERROR")
    assert "mode=revert requires version" in tools._knowledge_write(ctx, "shore", mode="revert", scope="global",
                                                                     expected_revision=first.revision)


def test_a_version_without_verified_text_is_a_typed_refusal(tmp_path):
    target = address(tmp_path, "typed")
    note = store.write_knowledge_note(target, "# Typed\n\nNow.\n").current
    journal = target.shelf.parent / "knowledge_history.jsonl"
    claimed, digested = hashlib.sha256(b"never stored").hexdigest(), hashlib.sha256(b"digest era").hexdigest()
    base = {"topic": "typed", "address": {"scope": "global"}, "publication": "source_capture"}
    with journal.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({**base, "new_sha256": claimed, "new_content": "other text"}) + "\n")
        stream.write(json.dumps({**base, "new_sha256": digested, "new_content": "", "content_digested": True}) + "\n")
    before, ctx = _files(tmp_path), ToolContext(repo_dir=tmp_path, drive_root=tmp_path, task_id="typed")
    for version, why in (("0" * 64, "no version " + "0" * 64), (claimed, "the stored text does not hash to the version"),
                         (digested, "holds no text (digest-era)"), ("not-a-revision", "64 lowercase hex")):
        result = store.write_knowledge_note(target, "", "revert", note.revision, version=version)
        assert not result.ok and result.reason.startswith("version_unavailable: ") and why in result.reason, result.reason
        # The caller holds the current revision (it passed the check): the refusal says why, without the note.
        reply = tools._knowledge_write(ctx, "typed", mode="revert", scope="global", version=version,
                                       expected_revision=note.revision)
        assert reply == f"⚠️ TOOL_ERROR: Knowledge write not completed: {result.reason}. Nothing was written.", reply
    assert _files(tmp_path) == before


def test_a_reason_rides_on_the_history_row_and_never_into_the_note(tmp_path):
    target = address(tmp_path, "reasons")
    original = store.write_knowledge_note(target, "# Reasons\n\nOld claim.\n").current
    edited = store.write_knowledge_note(target, "New claim.", "edit", original.revision, old_str="Old claim.",
                                        reason="the measurement changed")
    assert edited.ok and history(target)[-1]["reason"] == "the measurement changed"
    assert "measurement" not in edited.current.text
    plain = store.write_knowledge_note(target, "Newer claim.", "edit", edited.current.revision, old_str="New claim.")
    assert plain.ok and "reason" not in history(target)[-1]
    with pytest.raises(ValueError, match="reason is required for archive"):
        store.write_knowledge_note(target, "", "archive", plain.current.revision)
    assert store.write_knowledge_note(target, "", "archive", plain.current.revision, reason="kept elsewhere").ok
    assert history(target)[-1]["reason"] == "kept elsewhere"


def test_the_receipt_names_the_replaced_version_counts_both_ways_and_its_calls_work(tmp_path):
    ctx = ToolContext(repo_dir=tmp_path, drive_root=tmp_path, task_id="editor")
    registry = ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path)
    registry.set_context(ctx)
    original = store.write_knowledge_note(address(tmp_path, "host"), "# Host\n\nThe host is old.\n\n## Gone\n\nObsolete.\n",
                                          writer="turn").current
    meta = _meta(tools._knowledge_write(ctx, "host", mode="edit", scope="global", expected_revision=original.revision,
                                        reason="the host moved", edits=[_item("old", new_str="new")]))
    delta = meta["knowledge_delta"]
    assert (delta["removed_chars"], delta["added_chars"], delta["change_chars"]) == (3, 3, 0)  # net zero, both counted
    previous = meta["knowledge_previous"]
    assert previous["revision"] == original.revision and previous["mode"] == "overwrite"
    assert previous["focus"] == "unknown" and previous["reason"] == "unknown" and previous["ts"] != "unknown"
    assert "The host is" not in json.dumps(meta, ensure_ascii=False)  # nothing the call quoted is echoed
    assert "knowledge_outline" not in meta  # the write named its revision: it saw the earlier text
    second = store.read_knowledge_note(original.address)
    gone = _meta(tools._knowledge_write(ctx, "host", mode="edit", scope="global", expected_revision=second.revision,
                                        edits=[_item("## Gone\n\nObsolete.\n")]))
    assert gone["knowledge_delta"]["removed_sections"] == [{"heading": "## Gone", "chars": len("## Gone\n\nObsolete.\n"),
                                                            "first_line": "Obsolete."}]
    assert gone["knowledge_previous"]["reason"] == "the host moved"
    assert gone["knowledge_previous"]["focus"]["task_id"] == "editor"
    read_back = registry.execute(gone["knowledge_read_previous"]["tool"], gone["knowledge_read_previous"]["arguments"])
    assert read_back.endswith(second.text) and "Earlier version, not the current note" in read_back
    _meta(registry.execute(gone["knowledge_revert"]["tool"], gone["knowledge_revert"]["arguments"]))
    assert store.read_knowledge_note(original.address).raw == second.raw


def test_a_stale_revision_its_history_holds_gets_the_changes_instead_of_the_whole_note(tmp_path):
    ctx = ToolContext(repo_dir=tmp_path, drive_root=tmp_path, task_id="mine")
    target = address(tmp_path, "plan")
    held = store.write_knowledge_note(target, "# Plan\n\nStep one.\n\n## Later\n\nDetail.\n", writer="turn").current
    current = store.write_knowledge_note(target, "Step one, revised.", "edit", held.revision, old_str="Step one.",
                                         reason="another focus learned more", focus={"role": "presence", "task_id": "other"}).current
    stale = tools._knowledge_write(ctx, "plan", "# Plan\n\nMine.\n", mode="overwrite", scope="global",
                                   expected_revision=held.revision)
    assert "revision_conflict" in stale and f"expected_revision={current.revision!r}" in stale
    assert "presence task_id=other · edit · reason: another focus learned more" in stale
    assert "- Step one.\n+ Step one, revised.\n" in stale and "under # Plan" in stale and "Outline of 'plan'" in stale
    assert applied_blocks(held.text, stale) == current.text  # your version plus the blocks is the current note
    assert "The complete current text is knowledge_read(topic='plan', scope='global')." in stale
    assert current.text not in stale and store.read_knowledge_note(target).raw == current.raw
    unknown = tools._knowledge_write(ctx, "plan", "Mine.", mode="overwrite", scope="global", expected_revision="0" * 64)
    assert "revision_conflict" in unknown and unknown.endswith(current.text)  # a revision the tail lacks: as before
    required = tools._knowledge_write(ctx, "plan", "Mine.", mode="overwrite", scope="global")
    assert "revision_required" in required and required.endswith(current.text)


def test_a_stale_reply_places_every_change_in_whole_lines(tmp_path):
    """A change inside a line is shown as its whole line on both sides, so it can be placed: a fragment that
    repeats in its line (``3`` → ``5``) and a pure insertion inside a line. Line by line, the held version plus
    the blocks is the current note."""
    ctx = ToolContext(repo_dir=tmp_path, drive_root=tmp_path, task_id="mine")
    target = address(tmp_path, "limits")
    held = store.write_knowledge_note(target, "# Limits\n\nRetries: 3; timeout: 3; workers: 3.\n\n## People\n\n"
                                      "Alice met Bob, and Alice called Carol.\n\nKept.\n").current
    current = store.write_knowledge_note(target, held.text.replace("timeout: 3", "timeout: 5").replace(
        "and Alice called", "and Alice never called"), expected_revision=held.revision).current
    stale = tools._knowledge_write(ctx, "limits", "# Limits\n\nMine.\n", mode="overwrite", scope="global",
                                   expected_revision=held.revision)
    assert "revision_conflict" in stale and "2 block(s)" in stale and current.text not in stale
    assert ("@ yours: 1 line(s) from line 7 → current: 1 line(s) from line 7, under # Limits: 1 chars removed, 1 added\n"
            "- Retries: 3; timeout: 3; workers: 3.\n+ Retries: 3; timeout: 5; workers: 3.\n") in stale
    assert ("under ## People: 0 chars removed, 6 added\n- Alice met Bob, and Alice called Carol.\n"
            "+ Alice met Bob, and Alice never called Carol.\n") in stale
    assert "\n- 3\n" not in stale and "\n+ never \n" not in stale  # no fragment without its line
    assert applied_blocks(held.text, stale) == current.text


def test_a_stale_revision_gets_every_change_since_or_the_whole_note_never_a_clipped_diff(tmp_path, monkeypatch):
    ctx = ToolContext(repo_dir=tmp_path, drive_root=tmp_path, task_id="mine")
    target = address(tmp_path, "plan")
    held = store.write_knowledge_note(target, "# Plan\n\n" + "".join(
        f"## Part {index}\n\nold words {index}.\n\n" for index in range(6))).current
    current = store.write_knowledge_note(target, held.text.replace("old words", "new words"),
                                         expected_revision=held.revision).current

    def stale():
        return tools._knowledge_write(ctx, "plan", "# Plan\n\nMine.\n", mode="overwrite", scope="global",
                                      expected_revision=held.revision)

    whole = tools._knowledge_write(ctx, "plan", "Mine.", mode="overwrite", scope="global",
                                   expected_revision="0" * 64)  # today's reply
    complete = stale()
    assert "6 block(s)" in complete and complete.count("\n- old words ") == 6 and complete.count("\n+ new words ") == 6
    assert applied_blocks(held.text, complete) == current.text
    assert current.text not in complete and whole.endswith(current.text) and "What changed" not in whole
    monkeypatch.setattr(tools, "tool_result_limit", lambda name: len(complete))
    assert stale() == complete  # exactly at the bound: every change, whole
    monkeypatch.setattr(tools, "tool_result_limit", lambda name: len(complete) - 1)
    assert stale() == whole  # one character over: the complete current source, exactly as today
    assert store.read_knowledge_note(target).raw == current.raw


def test_a_stale_revision_beyond_the_tail_gets_the_whole_note_as_before(tmp_path, monkeypatch):
    ctx = ToolContext(repo_dir=tmp_path, drive_root=tmp_path, task_id="mine")
    target = address(tmp_path, "far")
    held = store.write_knowledge_note(target, "# Far\n\nOne.\n").current
    current = store.write_knowledge_note(target, "Two.", "edit", held.revision, old_str="One.").current
    for index in range(4):
        store.write_knowledge_note(address(tmp_path, f"filler/{index}"), "z" * 3000)
    monkeypatch.setattr(store, "_JOURNAL_TAIL_BYTES", 4096)
    stale = tools._knowledge_write(ctx, "far", "Mine.", mode="overwrite", scope="global", expected_revision=held.revision)
    assert "revision_conflict" in stale and stale.endswith(current.text) and "What changed" not in stale
    lead = next(entry for entry in tools.get_tools() if entry.name == "knowledge_write").schema["description"]
    assert "in whole lines when the note's recent history holds it" in lead  # the writer's tail, not all history


def test_the_writer_reads_its_row_address_and_appends_under_the_shelf_lock(tmp_path, monkeypatch):
    target = address(tmp_path, "locked")
    first = store.write_knowledge_note(target, "# Locked\n\nFirst.\n").current
    held, seen = [], []
    real_lock, real_tail, real_append = store.knowledge_write_lock, store._tail_read, store.append_jsonl

    @contextmanager
    def lock(shelf):
        with real_lock(shelf):
            held.append(shelf)
            try:
                yield
            finally:
                held.pop()

    monkeypatch.setattr(store, "knowledge_write_lock", lock)
    monkeypatch.setattr(store, "_tail_read", lambda *args: seen.append(("row address", list(held))) or real_tail(*args))
    monkeypatch.setattr(store, "append_jsonl", lambda *args, **kwargs: seen.append(("append", list(held)))
                        or real_append(*args, **kwargs))
    assert store.write_knowledge_note(target, "# Locked\n\nSecond.\n", expected_revision=first.revision).ok
    assert seen == [("row address", [target.shelf]), ("append", [target.shelf])]
    assert replaced(tmp_path, history(target)[-1]) == first.text  # the base read under the lock names the row
