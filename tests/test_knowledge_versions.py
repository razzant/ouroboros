"""One note's history read back: its outline, its versions newest first, one version exactly."""
from __future__ import annotations

import hashlib
import json
import re

from ouroboros import knowledge as store
from ouroboros import knowledge_versions as versions
from ouroboros import markdown_source
from ouroboros import text_change
from ouroboros.tools import knowledge as tools
from ouroboros.tools.registry import ToolContext, ToolRegistry
from tests._knowledge_history_shared import applied_blocks

TOPIC = "notes/река"
RIVER = ("---\ntype: note\nsummary: When the river matters.\n---\n# Река\n\nБерег 🌊.\n\n## Мост\n\nСтарый мост.\n\n"
         "### Опоры\n\nДве.\n\n## Конец\n\nВсё.\n")


def address(tmp_path, topic=TOPIC):
    return store.resolve_knowledge_address(tmp_path, topic, "global")


def context(tmp_path):
    return ToolContext(repo_dir=tmp_path, drive_root=tmp_path, task_id="reader")


def body(reply):
    return reply.split("\n\n", 1)[1]


def test_outline_ranges_read_back_exactly_their_sections(tmp_path):
    ctx = context(tmp_path)
    note = store.write_knowledge_note(address(tmp_path), RIVER).current
    outline = tools._knowledge_read(ctx, TOPIC, view="outline")
    for fact in ("Summary: When the river matters.", f"Revision: {note.revision}", "State: active",
                 f"Total: {len(note.text)} characters", "a parent's range contains its children's"):
        assert fact in outline
    ranges, bytes_differ = {}, False
    for heading in note.source.headings:  # the parser's sections, whose spans are UTF-8 byte offsets
        title = "#" * heading.level + " " + heading.title
        line, start, end, size = map(int, re.search(
            rf"^{re.escape(title)} — line (\d+), chars (\d+)\.\.(\d+) \((\d+) chars\)$", outline, re.M).groups())
        assert note.text.splitlines()[line - 1] == title and size == end - start
        read = tools._knowledge_read(ctx, TOPIC, start_char=start, end_char=end)
        # The printed range reads back exactly that section, byte for byte, in start_char/end_char coordinates.
        assert body(read).encode("utf-8") == note.raw[heading.section.start_byte:heading.section.end_byte]
        bytes_differ = bytes_differ or (start, end) != (heading.section.start_byte, heading.section.end_byte)
        ranges[title] = (start, end)
    assert bytes_differ  # a Cyrillic note, frontmatter included: byte offsets would name other text
    assert note.text[slice(*ranges["## Конец"])] == "## Конец\n\nВсё.\n"
    assert ranges["## Мост"][0] < ranges["### Опоры"][0] < ranges["### Опоры"][1] == ranges["## Мост"][1]
    assert tools._knowledge_read(ctx, TOPIC).endswith(note.text)  # the default read is the note, unchanged


def test_the_outline_shows_each_heading_line_as_written(tmp_path):
    """A heading is printed as its own line reads in the note (spacing, closing hashes, indentation, no CR), so the
    outline's text can be quoted back as an exact old_str."""
    ctx = context(tmp_path)
    target = address(tmp_path, "setup")
    target.path.parent.mkdir(parents=True, exist_ok=True)
    target.path.write_bytes(b"# Guide\r\n\r\n##  Setup ##\r\n\r\nSteps.\r\n\r\n   ### Indented\r\n\r\nMore.\r\n")
    note = store.read_knowledge_note(target)
    outline = tools._knowledge_read(ctx, "setup", view="outline")
    for written in ("# Guide", "##  Setup ##", "   ### Indented"):
        found = re.search(rf"^{re.escape(written)} — line (\d+), chars ", outline, re.M)
        assert found and note.text.splitlines()[int(found[1]) - 1] == written, (written, outline)
    assert "\r" not in outline and not re.search(r"^## Setup —", outline, re.M)
    edited = tools._knowledge_write(ctx, "setup", mode="edit", old_str="##  Setup ##", content="## Setup",
                                    expected_revision=note.revision)
    assert edited.startswith("✅"), edited


def test_history_lists_versions_newest_first_marks_the_current_one_and_names_gaps(tmp_path):
    ctx = context(tmp_path)
    target = address(tmp_path)
    first = store.write_knowledge_note(target, "# Река\n\nOne.\n").current
    journal = target.shelf.parent / "knowledge_history.jsonl"
    torn_at = journal.stat().st_size
    with journal.open("a", encoding="utf-8") as stream:
        stream.write('{"topic": "notes/река", "torn\n')
    second = store.write_knowledge_note(target, "Two.", "edit", first.revision, old_str="One.", reason="corrected",
                                        focus={"role": "main", "task_id": "t2"}).current
    lost = hashlib.sha256(b"never landed").hexdigest()
    with journal.open("a", encoding="utf-8") as stream:  # a capture whose source never landed
        stream.write(json.dumps({"topic": TOPIC, "address": {"scope": "global"}, "publication": "source_capture",
                                 "mode": "overwrite", "old_sha256": second.revision, "new_sha256": lost,
                                 "new_content": "lost"}, ensure_ascii=False) + "\n")
    third = store.write_knowledge_note(target, "Three.", "edit", second.revision, old_str="Two.").current
    listing = tools._knowledge_read(ctx, TOPIC, view="history")
    rows = [line for line in body(listing).splitlines() if line.startswith("- ")]
    assert [row.split(" · ")[0][2:] for row in rows] == [third.revision, lost, second.revision, first.revision]
    assert rows[0].endswith("· current") and "not the base of the next write" in rows[1]
    assert "reason: corrected" in rows[2] and "focus main task_id=t2" in rows[2] and "3 chars removed, 3 added" in rows[2]
    assert "reason: not recorded" in rows[3] and "focus unknown" in rows[3]
    assert f"1 journal line(s) naming this note are not JSON and were skipped (the first at byte offset {torn_at})" in listing
    assert f"expected_revision={third.revision!r}" in listing
    page = tools._knowledge_read(ctx, TOPIC, view="history", start_char=10, end_char=60)
    assert body(page) == body(listing)[10:60] and '"complete_chars": ' in page


def test_history_prints_the_headings_each_row_recorded_and_parses_no_earlier_version(tmp_path, monkeypatch):
    target = address(tmp_path)
    first = store.write_knowledge_note(target, "# Река\n\nОдин.\n").current
    second = store.write_knowledge_note(target, "# Река\n\nОдин.\n\n## Мост\n\nНовый.\n", expected_revision=first.revision)
    assert second.delta["added_headings"] == ["## Мост"] and second.delta["removed_headings"] == []
    third = store.write_knowledge_note(target, "Два.", "edit", second.current.revision, old_str="Один.")
    assert third.delta["added_headings"] == []

    def parsed_again(*_args, **_kwargs):
        raise AssertionError("the listing parsed a version again instead of printing its row")

    monkeypatch.setattr(store, "parse_markdown_source", parsed_again)
    monkeypatch.setattr(markdown_source, "parse_markdown_source", parsed_again)
    rows = [line for line in versions.render_history(third.current).splitlines() if line.startswith("- ")]
    assert "headings added" not in rows[0] and "headings added: ## Мост" in rows[1]
    assert "headings added: # Река" in rows[2]  # the creating row recorded every heading it brought


def test_a_digest_era_row_says_its_text_is_not_stored(tmp_path):
    target = address(tmp_path)
    note = store.write_knowledge_note(target, "# Река\n").current
    journal = target.shelf.parent / "knowledge_history.jsonl"
    old = hashlib.sha256(b"digested").hexdigest()
    journal.write_text(json.dumps({"topic": TOPIC, "mode": "edit", "new_sha256": old, "new_content": "",
                                   "content_digested": True}) + "\n" + journal.read_text(encoding="utf-8"), encoding="utf-8")
    listing = versions.render_history(note)
    assert re.search(rf"^- {old} · .*text not stored", listing, re.M)


def test_a_version_reads_back_exactly_in_ranges_and_as_its_outline(tmp_path):
    ctx = context(tmp_path)
    target = address(tmp_path)
    first = store.write_knowledge_note(target, RIVER).current
    second = store.write_knowledge_note(target, "", "edit", first.revision,
                                        edits=[{"old_str": "## Мост\n\nСтарый мост.\n\n### Опоры\n\nДве.\n\n", "new_str": ""}]).current
    old = tools._knowledge_read(ctx, TOPIC, version=first.revision)
    assert old.endswith(first.text) and "Earlier version, not the current note" in old
    assert f"The current revision is {second.revision}" in old and "focus unknown (overwrite; reason: not recorded)" in old
    assert tools._knowledge_read(ctx, TOPIC, version=first.revision, start_char=3, end_char=40).endswith(first.text[3:40])
    outline = tools._knowledge_read(ctx, TOPIC, version=first.revision, view="outline")
    assert "an earlier version, not the current note" in outline and "### Опоры" in outline
    assert f"Revision: {first.revision}" in outline
    assert "### Опоры" not in tools._knowledge_read(ctx, TOPIC, view="outline")
    assert tools._knowledge_read(ctx, TOPIC, version=second.revision).endswith(second.text)  # the current one too
    missing = tools._knowledge_read(ctx, TOPIC, version="0" * 64)
    assert missing.startswith("⚠️ TOOL_ERROR: version_unavailable: no version") and second.text not in missing
    assert "view=history lists them all" in tools._knowledge_read(ctx, TOPIC, view="history", version=first.revision)
    assert "view='diff'" in tools._knowledge_read(ctx, TOPIC, view="diff")


def test_an_empty_or_null_view_or_version_reads_the_current_note_and_says_so(tmp_path):
    """A value that means what omission means takes the omitted path, said in the header (tools/arg_feedback);
    a version that names something malformed still refuses, typed."""
    ctx = context(tmp_path)
    registry = ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path)
    registry.set_context(ctx)
    note = store.write_knowledge_note(address(tmp_path), RIVER).current
    plain = registry.execute("knowledge_read", {"topic": TOPIC, "scope": "global"})
    assert plain.endswith(note.text) and "[Argument note]" not in plain
    for arguments, said in (({"version": ""}, ["version='' ignored: no revision named; this is the current note"]),
                            ({"view": ""}, ["view='' ignored: the default view, the note itself"]),
                            ({"view": None}, ["view=None ignored: the default view, the note itself"]),
                            ({"view": None, "version": ""}, ["view=None ignored", "version='' ignored"])):
        read = registry.execute("knowledge_read", {"topic": TOPIC, "scope": "global", **arguments})
        assert read.endswith(note.text) and read.split("\n", 1)[0] == plain.split("\n", 1)[0], (arguments, read[:300])
        assert all(f"[Argument note] {item}" in read for item in said), (arguments, read[:600])
    history = registry.execute("knowledge_read", {"topic": TOPIC, "scope": "global", "view": "history", "version": ""})
    assert history.startswith("[Knowledge history]") and "[Argument note] version='' ignored" in history
    malformed = registry.execute("knowledge_read", {"topic": TOPIC, "scope": "global", "version": "abc"})
    assert malformed.startswith("⚠️ TOOL_ERROR: version_unavailable: a version is a revision") and note.text not in malformed
    assert "view='diff'" in registry.execute("knowledge_read", {"topic": TOPIC, "scope": "global", "view": "diff"})


def test_an_emptied_plain_note_keeps_its_revision_and_its_calls_work(tmp_path):
    """Emptying a note without frontmatter publishes the version sha256(b""): its row records it, the next
    receipt's read and revert calls reach it, and the listing shows it stored. A note's absence is no version."""
    registry = ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path)
    registry.set_context(context(tmp_path))
    target, empty = address(tmp_path, "plain"), hashlib.sha256(b"").hexdigest()
    target.path.parent.mkdir(parents=True, exist_ok=True)
    target.path.write_bytes(b"# Plain\n\nNo frontmatter.\n")
    emptied = store.write_knowledge_note(target, "", expected_revision=store.read_knowledge_note(target).revision)
    assert emptied.ok and target.path.read_bytes() == b"" and emptied.current.revision == empty
    journal = target.shelf.parent / "knowledge_history.jsonl"
    assert json.loads(journal.read_text(encoding="utf-8").splitlines()[-1])["new_sha256"] == empty
    receipt = registry.execute("knowledge_write", {"topic": "plain", "scope": "global", "content": "# Again\n",
                                                   "expected_revision": empty})
    assert receipt.startswith("✅"), receipt
    meta = json.loads(receipt.split("\n", 1)[1])
    assert meta["knowledge_previous"]["revision"] == empty
    assert json.loads(journal.read_text(encoding="utf-8").splitlines()[-1])["old_sha256"] == empty
    stale = registry.execute("knowledge_write", {"topic": "plain", "scope": "global", "content": "Mine.\n",
                                                 "expected_revision": empty})  # the empty version, held
    assert "revision_conflict" in stale and "1 block(s)" in stale and "\n+ # Again\n" in stale, stale
    read = registry.execute(meta["knowledge_read_previous"]["tool"], meta["knowledge_read_previous"]["arguments"])
    assert "Earlier version, not the current note" in read and '"complete_chars": 0' in read, read
    reverted = registry.execute(meta["knowledge_revert"]["tool"], meta["knowledge_revert"]["arguments"])
    assert reverted.startswith("✅") and target.path.read_bytes() == b"", reverted
    listing = versions.render_history(store.read_knowledge_note(target))
    assert f"- {empty} · " in listing and "(no revision recorded)" not in listing and "text not stored" not in listing
    created = store.write_knowledge_note(address(tmp_path, "created"), "# Created\n").current
    assert json.loads(journal.read_text(encoding="utf-8").splitlines()[-1])["old_sha256"] == ""  # absent: no version
    refused = store.write_knowledge_note(created.address, "", "revert", created.revision, version=empty)
    assert not refused.ok and refused.reason.startswith(f"version_unavailable: no version {empty}"), refused.reason


def test_a_version_lookup_that_misses_names_the_torn_lines_of_this_note(tmp_path):
    """A torn journal line naming the note may have lost the very version asked for: a miss counts every such
    line, not only those holding the version, and says so; a clean miss names no gap."""
    ctx = context(tmp_path)
    target = address(tmp_path)
    store.write_knowledge_note(target, "# Река\n\nOne.\n")
    missing = "0" * 64
    clean = tools._knowledge_read(ctx, TOPIC, version=missing)
    assert clean == f"⚠️ TOOL_ERROR: version_unavailable: no version {missing} of this note in its history"
    journal = target.shelf.parent / "knowledge_history.jsonl"
    offset = journal.stat().st_size
    with journal.open("a", encoding="utf-8") as stream:  # torn before its revision fields
        stream.write(json.dumps({"topic": TOPIC, "mode": "edit"}, ensure_ascii=False)[:-1] + ', "new_sh\n')
    torn = tools._knowledge_read(ctx, TOPIC, version=missing)
    assert torn == (f"⚠️ TOOL_ERROR: version_unavailable: no version {missing} of this note in its history (1 journal "
                    f"line(s) naming this note are not JSON and were skipped (the first at byte offset {offset}))")


def test_the_backward_scan_finds_every_row_across_block_edges(tmp_path, monkeypatch):
    monkeypatch.setattr(versions, "_BLOCK", 97)  # shorter than every row: lines span several reads
    target = address(tmp_path)
    note = store.write_knowledge_note(target, "# Река\n\nv0\n").current
    revisions = [note.revision]
    for index in range(1, 6):
        store.write_knowledge_note(address(tmp_path, f"other/{index}"), "x" * (40 * index) + " notes/река")
        note = store.write_knowledge_note(target, f"v{index}", "edit", note.revision, old_str=f"v{index - 1}").current
        revisions.append(note.revision)
    listed = [line.split(" · ")[0][2:] for line in versions.render_history(note).splitlines() if line.startswith("- ")]
    assert listed == list(reversed(revisions))
    for revision in revisions:
        text, _row = versions.find_version(target, revision)
        assert hashlib.sha256(text.encode("utf-8")).hexdigest() == revision


def test_the_changes_since_a_held_version_show_every_block_whole(tmp_path):
    target = address(tmp_path)
    held = store.write_knowledge_note(target, "# Plan\n\n" + "".join(
        f"## Part {index}\n\n{'old words ' * 150}\n\n" for index in range(6)) + "## Small\n\ntiny\n").current
    current = store.write_knowledge_note(target, held.text.replace("old words", "new words").replace("tiny", "small"),
                                         expected_revision=held.revision).current
    facts = versions.held_version(target.shelf.parent / "knowledge_history.jsonl", target, held.revision, 1 << 23)
    rendered = versions.render_changes(current, facts)
    change = text_change.exact_change(held.text, current.text)
    assert len(change) == 7 and "7 block(s)" in rendered and len(rendered) > 7500  # no bound shapes it
    for old_line, new_line in (("old words " * 150, "new words " * 150), ("tiny", "small")):  # whole lines, uncut
        assert f"\n- {old_line}\n+ {new_line}\n" in rendered
    assert rendered.count("\n@ yours: 1 line(s) from line ") == 7 and "not shown" not in rendered
    assert applied_blocks(held.text, rendered) == current.text
    assert rendered.endswith("The complete current text is knowledge_read(topic='notes/река', scope='global').\n")


def test_removed_sections_name_heading_size_and_first_line_cut_to_120(tmp_path):
    target = address(tmp_path)
    long_line = "Первая строка под заголовком. " * 6
    original = store.write_knowledge_note(target, f"# Keep\n\nx\n\n## Gone\n\n\n{long_line}\nsecond\n").current
    changed = store.write_knowledge_note(target, "# Keep\n\nx\n", expected_revision=original.revision)
    assert changed.delta["removed_sections"] == [{"heading": "## Gone", "first_line": long_line.strip()[:120],
                                                  "chars": len(f"## Gone\n\n\n{long_line}\nsecond\n")}]
    assert changed.delta["removed_headings"] == ["## Gone"]
