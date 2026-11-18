"""One note's history read back: its outline, its versions newest first, one version exactly."""
from __future__ import annotations

import hashlib
import json
import re

import pytest

from ouroboros import knowledge as store
from ouroboros import knowledge_versions as versions
from ouroboros import markdown_source
from ouroboros import text_change
from ouroboros.tools import knowledge as tools
from ouroboros.tools.registry import ToolContext, ToolRegistry
from tests._knowledge_history_shared import applied_blocks, changed_rows, printed_call

TOPIC = "notes/река"
RIVER = ("---\ntype: note\nsummary: When the river matters.\n---\n# Река\n\nБерег 🌊.\n\n## Мост\n\nСтарый мост.\n\n"
         "### Опоры\n\nДве.\n\n## Конец\n\nВсё.\n")


def address(tmp_path, topic=TOPIC):
    return store.resolve_knowledge_address(tmp_path, topic, "global")


def context(tmp_path):
    return ToolContext(repo_dir=tmp_path, drive_root=tmp_path, task_id="reader")


def body(reply):
    return reply.split("\n\n", 1)[1]


@pytest.mark.parametrize("scope", ["global", "project:river"])
def test_exact_version_survives_rename_and_the_unversioned_address_reports_the_missing_note(tmp_path, scope):
    target = store.resolve_knowledge_address(tmp_path, TOPIC, scope)
    first = store.write_knowledge_note(target, RIVER.replace("\n", "\r\n")).current
    second = store.write_knowledge_note(target, RIVER.replace("## Мост", "## Пристань"), expected_revision=first.revision).current
    ctx = context(tmp_path)
    registry = ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path)
    registry.set_context(ctx)
    args = {"topic": target.topic, "scope": scope}
    assert body(registry.execute("knowledge_read", args)) == second.text
    start, end = second.text.index("## Пристань"), second.text.index("## Конец")
    bounds = {"start_char": start, "end_char": end}
    pin = {**args, "version": second.revision}
    note = registry.execute_result("knowledge_read", pin)
    page = registry.execute_result("knowledge_read", {**pin, **bounds})
    outline = registry.execute_result("knowledge_read", {**pin, "view": "outline"})
    assert body(note.text) == second.text and body(page.text) == second.text[start:end]
    assert "an earlier version" not in body(outline.text) and "## Пристань" in body(outline.text)
    assert f"default view and version={second.revision}." in body(outline.text)
    readers = [note.meta["knowledge_source"]["read"], page.meta["knowledge_source"]["read"],
               outline.meta["knowledge_view"]["read"]]
    assert all(reader == {"tool": "knowledge_read", "arguments": pin} for reader in readers)
    target.path.rename(target.path.with_name("new-name.md"))

    exact = registry.execute_result("knowledge_read", {**args, "version": first.revision})
    assert exact.status == "ok" and "The live note is absent at this address." in exact.text
    assert exact.text[exact.meta["knowledge_body_start"]:].encode("utf-8") == first.raw
    assert exact.meta["knowledge_source"]["read"]["arguments"] == {**args, "version": first.revision}
    historical_outline = body(registry.execute("knowledge_read", {**args, "version": first.revision, "view": "outline"}))
    assert "an earlier version, not the current note (live note absent at this address)" in historical_outline
    assert "## Мост" in historical_outline and "## Пристань" not in historical_outline
    assert f"Revision: {first.revision}" in historical_outline
    missing = registry.execute_result("knowledge_read", args)
    assert missing.meta["knowledge_missing"] is True and "Topic" in missing.text and "not found" in missing.text
    unknown = registry.execute_result("knowledge_read", {**args, "version": "0" * 64})
    assert unknown.status == "error" and "version_unavailable" in unknown.text

    replacement = store.write_knowledge_note(target, "# Replacement\n\nCurrent contents.\n").current
    current = registry.execute_result("knowledge_read", args)
    assert body(current.text) == replacement.text and "version" not in current.meta["knowledge_source"]["read"]["arguments"]
    assert "# Replacement" in registry.execute("knowledge_read", {**args, "view": "outline"})
    for reader, options in zip(readers, ({}, bounds, bounds)):
        recovered = registry.execute_result(reader["tool"], {**reader["arguments"], **options})
        assert recovered.status == "ok" and f"The current revision is {replacement.revision}." in recovered.text
        assert body(recovered.text) == (second.text[start:end] if options else second.text)


@pytest.mark.parametrize("occupant", ["directory", "non-utf8"])
def test_a_pinned_read_recovers_when_the_live_path_is_unreadable(tmp_path, occupant):
    registry = ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path)
    registry.set_context(context(tmp_path))
    target = address(tmp_path)
    saved = store.write_knowledge_note(target, RIVER.replace("\n", "\r\n")).current
    target.path.rename(target.path.with_name("moved.md"))
    args = {"topic": target.topic, "scope": target.scope}
    earlier = {**args, "version": saved.revision}
    missing = registry.execute_result("knowledge_read", args)
    assert missing.meta["knowledge_missing"] is True
    assert "The live note is absent at this address." in registry.execute("knowledge_read", earlier)
    if occupant == "directory":
        target.path.mkdir()
        expected_code = "TOOL_REPORTED_FAILURE"
    else:
        target.path.write_bytes(RIVER.encode("utf-16"))
        expected_code = "TOOL_ARG_ERROR"
    ordinary = registry.execute_result("knowledge_read", args)
    assert ordinary.status == "error" and ordinary.code == expected_code
    assert not ordinary.meta.get("knowledge_missing") and "The live note is" not in ordinary.text
    exact = registry.execute_result("knowledge_read", earlier)
    assert exact.status == "ok" and body(exact.text).encode() == saved.raw
    assert "The live note is unreadable at this address." in exact.text
    outline = registry.execute_result("knowledge_read", {**earlier, "view": "outline"})
    assert "live note unreadable at this address" in body(outline.text)
    assert body(registry.execute("knowledge_read", outline.meta["knowledge_view"]["read"]["arguments"])) == saved.text


@pytest.mark.parametrize("legacy", [False, True], ids=["on-disk-spelling", "legacy-case-alias"])
def test_case_aliases_use_the_journal_address_for_pins_history_and_revert(tmp_path, monkeypatch, legacy):
    registry = ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path)
    registry.set_context(context(tmp_path))
    target, alias = address(tmp_path, "Field/CaseNote"), address(tmp_path, "field/casenote")
    if legacy:
        target.path.parent.mkdir(parents=True)
        target.path.write_bytes(b"Handwritten seed.\n")
        if not alias.path.exists():
            pytest.skip("legacy alias writes require a case-insensitive volume")
        seed = store.read_knowledge_note(target)
        with monkeypatch.context() as patch:
            # Older writers journaled the requested address even when the shelf spelled it differently.
            patch.setattr(store, "shelf_spelling", lambda requested: requested)
            first = store.write_knowledge_note(alias, RIVER, expected_revision=seed.revision).current
            second = store.write_knowledge_note(alias, RIVER.replace("Мост", "Пристань"), expected_revision=first.revision).current
    else:
        first = store.write_knowledge_note(target, RIVER).current
        second = store.write_knowledge_note(target, RIVER.replace("Мост", "Пристань"), expected_revision=first.revision).current

    args = {"topic": alias.topic, "scope": alias.scope}
    if not alias.path.exists():
        separate = store.write_knowledge_note(alias, "Independent note.\n").current
        listing = registry.execute("knowledge_read", {**args, "view": "history"})
        assert "1 recorded version(s)" in listing and separate.revision in listing and first.revision not in listing
        assert "version_unavailable" in registry.execute("knowledge_read", {**args, "version": first.revision})
        return

    history = registry.execute_result("knowledge_read", {**args, "view": "history"})
    listing = body(history.text)
    assert "2 recorded version(s)" in listing and all(note.revision in listing for note in (first, second))
    cursor = history.meta["knowledge_view"]
    page = registry.execute("knowledge_read", {**args, "topic": cursor["topic"], "view": "history",
                                              "fingerprint": cursor["fingerprint"], "start_char": 40})
    assert body(page) == listing[40:]
    journal_topic = alias.topic if legacy else target.topic
    assert body(registry.execute("knowledge_read", {**args, "topic": journal_topic, "version": first.revision})) == first.text
    readers = []
    for note in (first, second):
        pin = {**args, "version": note.revision}
        exact = registry.execute_result("knowledge_read", pin)
        page = registry.execute_result("knowledge_read", {**pin, "start_char": 1, "end_char": 10})
        outline = registry.execute_result("knowledge_read", {**pin, "view": "outline"})
        assert body(exact.text) == note.text and body(page.text) == note.text[1:10]
        assert f"version={note.revision}" in body(outline.text)
        expected_reader = {"tool": "knowledge_read", "arguments": {**pin, "topic": journal_topic}}
        assert exact.meta["knowledge_source"]["read"] == page.meta["knowledge_source"]["read"] == expected_reader
        assert outline.meta["knowledge_view"]["read"] == expected_reader
        readers.append((expected_reader, note.text))

    target.path.write_bytes(b"\xff")
    unreadable = registry.execute("knowledge_read", {**args, "version": first.revision})
    assert "The live note is unreadable at this address." in unreadable and body(unreadable) == first.text
    target.path.write_bytes(second.raw)
    current = second
    if legacy:
        current = store.write_knowledge_note(alias, "Newly journaled under the on-disk spelling.\n", mode="append").current
        listing = registry.execute("knowledge_read", {**args, "view": "history"})
        assert "3 recorded version(s)" in listing
        assert listing.index("current") > listing.index(current.revision)
        assert all(note.revision in listing for note in (first, second))
    reverted = registry.execute_result("knowledge_write", {**args, "mode": "revert", "version": first.revision,
                                                           "expected_revision": current.revision})
    assert reverted.status == "ok" and target.path.read_bytes() == first.raw

    target.path.rename(target.path.with_name("Moved.md"))
    for reader, expected in readers:
        assert body(registry.execute(reader["tool"], reader["arguments"])) == expected


def test_a_malformed_outline_names_the_pin_that_reads_its_complete_source(tmp_path):
    registry = ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path)
    registry.set_context(context(tmp_path))
    target = address(tmp_path)
    target.path.parent.mkdir(parents=True)
    target.path.write_bytes(b"---\nsummary: [\n---\n## Earlier body\n")
    malformed = store.read_knowledge_note(target)
    assert malformed.parse_error
    revised = store.write_knowledge_note(target, RIVER, expected_revision=malformed.revision).current
    args = {"topic": target.topic, "scope": target.scope, "view": "outline"}
    assert "Headings: unknown" not in registry.execute("knowledge_read", args)
    assert f"version={revised.revision}" not in registry.execute("knowledge_read", args)

    old = registry.execute_result("knowledge_read", {**args, "version": malformed.revision})
    assert f"the complete note is the default read with version={malformed.revision}" in body(old.text)
    reader = old.meta["knowledge_view"]["read"]
    assert body(registry.execute(reader["tool"], reader["arguments"])) == malformed.text


def test_knowledge_changes_skip_unchanged_journals_and_read_only_the_appended_window(tmp_path, monkeypatch):
    target = address(tmp_path)
    first = store.write_knowledge_note(target, "# Earlier\n" + "old body\n" * 2000).current
    journal = versions.journal_path(target)
    gaps = set()
    events, baseline = versions.knowledge_changes(tmp_path, None, gaps)
    assert events == [] and not gaps and baseline["global"]["offset"] == journal.stat().st_size
    original_open, original_parse, opened, parsed = type(journal).open, versions._parse, [], []

    def opening(path, *args, **kwargs):
        if path == journal:
            opened.append(path)
        return original_open(path, *args, **kwargs)

    def parsing(line):
        parsed.append(line)
        return original_parse(line)

    with monkeypatch.context() as spy:
        spy.setattr(type(journal), "open", opening)
        assert versions.knowledge_changes(tmp_path, baseline, gaps) == ([], baseline)
        assert opened == []
    # Appended later with an older stamp: publication position, never timestamp, orders this window.
    later = {"topic": TOPIC, "publication": "source_capture", "mode": "edit", "ts": "2000-01-01T00:00:00Z",
             "address": {"scope": "global"}, "focus": {"role": "main", "room_id": "fixture-room"},
             "new_sha256": first.revision, "new_content": first.text}
    with journal.open("ab") as stream:
        stream.write((json.dumps(later) + "\n").encode("utf-8"))
    with monkeypatch.context() as spy:
        spy.setattr(type(journal), "open", opening)
        spy.setattr(versions, "_parse", parsing)
        changes, accepted = versions.knowledge_changes(tmp_path, baseline, gaps)
    assert opened == [journal] and len(parsed) == len(changes) == 1 and not gaps
    assert "1 write; by main room_id=fixture-room; first 2000-01-01T00:00:00Z; last 2000-01-01T00:00:00Z" in changes[0][2]
    assert f"version={first.revision!r}" in changes[0][2]
    assert accepted["global"]["offset"] == journal.stat().st_size
    assert versions.knowledge_changes(tmp_path, accepted, gaps) == ([], accepted)


def test_knowledge_changes_group_pages_in_first_write_order_and_keep_all_versions(tmp_path, monkeypatch):
    target, other = address(tmp_path, "zeta"), address(tmp_path, "alpha")
    first = store.write_knowledge_note(target, "Before this window.").current
    gaps = set()
    _, baseline = versions.knowledge_changes(tmp_path, None, gaps)
    writers = ({"role": "root", "task_id": "fixture-z-writer"},
               {"role": "presence", "room_id": "fixture-a-room"})
    stamps = [f"2001-01-01T00:{index:02d}:00Z" for index in range(13)]
    stamps[-1] = "1999-01-01T00:00:00Z"  # append order, not timestamp order, chooses the latest version
    notes, revision = [], first.revision
    for index, stamp in enumerate(stamps):
        monkeypatch.setattr(store, "utc_now_iso", lambda: stamp)
        written = store.write_knowledge_note(target, f"Write {index}.\n", expected_revision=revision,
                                            focus=writers[index % 2])
        assert written.ok, written.reason
        notes.append(written.current)
        revision = written.current.revision
        if index == 0:
            second_page = store.write_knowledge_note(other, "Another page.\n", focus=writers[1]).current
    own = store.write_knowledge_note(target, "The wake's own later write.\n", expected_revision=revision,
                                    focus={"role": "consciousness", "task_id": "fixture-own-wake"}).current
    before = {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    changes, accepted = versions.knowledge_changes(tmp_path, baseline, gaps)
    assert len(changes) == 2 and not gaps
    assert changes == [
        ("knowledge_change", None,
         f"- knowledge global/zeta: 13 writes; by root task_id=fixture-z-writer, presence room_id=fixture-a-room; "
         f"first {stamps[0]}; last {stamps[-1]}; "
         f"knowledge_read(topic='zeta', scope='global', version={revision!r})"),
        ("knowledge_change", None,
         f"- knowledge global/alpha: 1 write; by presence room_id=fixture-a-room; "
         f"first {stamps[0]}; last {stamps[0]}; "
         f"knowledge_read(topic='alpha', scope='global', version={second_page.revision!r})"),
    ]
    assert {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()} == before
    ctx = context(tmp_path)
    for change, note in zip(changes, (notes[-1], second_page)):
        tool, arguments = printed_call(change[2].split("; ")[-1])
        assert tool == "knowledge_read" and tools._knowledge_read(ctx, **arguments).endswith(note.text)
    for note in (first, *notes, own):
        assert tools._knowledge_read(ctx, "zeta", version=note.revision).endswith(note.text)
    assert versions.knowledge_changes(tmp_path, baseline, gaps)[0] == changes  # unaccepted: replay the same window
    assert versions.knowledge_changes(tmp_path, accepted, gaps) == ([], accepted)
    assert accepted["global"]["offset"] == versions.journal_path(target).stat().st_size


@pytest.mark.parametrize("fault", ["ending_changed", "unfinished"])
def test_knowledge_journal_gaps_keep_the_accepted_position_and_intact_journals_continue(tmp_path, fault):
    target = address(tmp_path)
    first = store.write_knowledge_note(target, "# First\n").current
    journal = versions.journal_path(target)
    gaps = set()
    _, baseline = versions.knowledge_changes(tmp_path, None, gaps)
    second = store.write_knowledge_note(target, "# Second\n", expected_revision=first.revision).current
    good = journal.read_bytes()
    if fault == "ending_changed":
        end = baseline["global"]["offset"]
        journal.write_bytes(good[:end - 2] + b" \n" + good[end:])
    else:
        journal.write_bytes(good[:-1])
    changes, retained = versions.knowledge_changes(tmp_path, baseline, gaps)
    assert changes == [] and retained == baseline
    assert any("knowledge changes unreadable (global)" in gap for gap in gaps)
    journal.write_bytes(good)
    gaps.clear()
    changes, accepted = versions.knowledge_changes(tmp_path, retained, gaps)
    assert len(changes) == 1 and f"version={second.revision!r}" in changes[0][2] and not gaps
    assert accepted["global"]["offset"] == len(good)


def test_outline_ranges_read_back_exactly_their_sections(tmp_path):
    ctx = context(tmp_path)
    note = store.write_knowledge_note(address(tmp_path), RIVER).current
    outline = tools._knowledge_read(ctx, TOPIC, view="outline")
    for fact in ("Summary: When the river matters.", f"Revision: {note.revision}", "State: active",
                 f"Total: {len(note.text)} characters", "a parent's range contains its children's",
                 # A range passed with view='outline' pages the outline's own text; the printed ranges are the note's.
                 "size). The ranges are of the note itself, read with the default view. A section runs"):
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


def test_a_setext_heading_that_opens_the_body_has_no_section_of_its_own_in_the_outline(tmp_path):
    """The parser's implicit preamble section begins at a setext heading that opens the body (a plain note, or one
    whose frontmatter it directly follows) and stops at the first ATX heading, leaving out the child: that is not
    the heading's own section. An ATX heading there has its whole section, child included."""
    ctx = context(tmp_path)
    child = "Body one.\n\n## Sub\n\nChild.\n"
    for topic, raw in (("plain", "Title\n=====\n\n" + child), ("fronted", "---\ntype: note\n---\nTitle\n=====\n\n" + child),
                       ("atx", "# Title\n\n" + child)):
        target = address(tmp_path, topic)
        target.path.parent.mkdir(parents=True, exist_ok=True)
        target.path.write_bytes(raw.encode("utf-8"))
        outline = tools._knowledge_read(ctx, topic, view="outline")
        found = re.search(r"^#? ?Title — line \d+, chars (\d+)\.\.(\d+) \((\d+) chars(.*)\)$", outline, re.M)
        assert found, outline
        section = raw[int(found[1]):int(found[2])]
        if topic == "atx":
            assert section == "# Title\n\n" + child and found[4] == ""
        else:
            assert section == "Title\n=====\n" and found[4] == (
                "; no section of its own (a level-1 setext heading): the range is its heading"), (topic, outline)


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
    assert listing.endswith(f"expected_revision={third.revision!r}). Each line is the note after its write; its undo "
                            "puts back the version that write replaced, which is not always the line below.\n"), listing
    # Each line names the version its write actually replaced: the capture that never landed replaced `second` too.
    assert [re.search(r" · (replaced [0-9a-f]{64}|created it: no earlier version to put back) · ", row)[1]
            for row in rows] == [f"replaced {second.revision}", f"replaced {second.revision}",
                                 f"replaced {first.revision}", "created it: no earlier version to put back"]
    fingerprint = re.search(r'"fingerprint": "([0-9a-f]{16})"', listing)[1]
    page = tools._knowledge_read(ctx, TOPIC, view="history", start_char=10, end_char=60, fingerprint=fingerprint)
    assert body(page) == body(listing)[10:60] and '"complete_chars": ' in page
    # Undoing the write that made `second` puts back the version its own line names.
    undo = tools._knowledge_write(ctx, TOPIC, mode="revert", version=re.search(r"replaced ([0-9a-f]{64})", rows[2])[1],
                                  expected_revision=third.revision, scope="global")
    assert undo.startswith("✅") and store.read_knowledge_note(target).text == first.text, undo


def test_the_listing_names_a_version_found_only_on_a_rows_old_side_as_that_writes_undo(tmp_path):
    """Undo is the version a write replaced, not the line below: after an outside edit the next write's row holds the
    outside text only on its old side, and the listing names exactly that version, which reads back and reverts."""
    ctx, target = context(tmp_path), address(tmp_path)
    first = store.write_knowledge_note(target, "# Река\n\nOne.\n").current
    target.path.write_bytes(first.raw.replace(b"One.", b"Outside."))  # an edit outside the writer: no row
    outside = store.read_knowledge_note(target)
    second = store.write_knowledge_note(target, "# Река\n\nTwo.\n", expected_revision=outside.revision).current
    rows = [line for line in versions.render_history(second).splitlines() if line.startswith("- ")]
    assert rows[0].startswith(f"- {second.revision} · ") and f" · replaced {outside.revision} · " in rows[0], rows
    assert rows[1].startswith(f"- {first.revision} · ") and "not the base of the next write" in rows[1], rows
    assert outside.revision not in rows[1]  # no line wrote the outside text: only the next row's old side holds it
    assert versions.find_version(target, outside.revision) == (outside.text, {})
    undo = tools._knowledge_write(ctx, TOPIC, mode="revert", version=outside.revision,
                                  expected_revision=second.revision, scope="global")
    assert undo.startswith("✅") and target.path.read_bytes() == outside.raw, undo


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
    journal.write_text(json.dumps({"topic": TOPIC, "mode": "edit", "new_sha256": old, "new_content": "",  # as the writer
                                   "content_digested": True}, ensure_ascii=False)  # and the digest compaction spell it
                       + "\n" + journal.read_text(encoding="utf-8"), encoding="utf-8")
    listing = versions.render_history(note)
    assert re.search(rf"^- {old} · .*text not stored", listing, re.M)


def test_a_version_reads_back_exactly_in_ranges_and_as_its_outline(tmp_path):
    ctx = context(tmp_path)
    target = address(tmp_path)
    first = store.write_knowledge_note(target, RIVER + "[Old gap](old-gap.md)\n").current
    second = store.write_knowledge_note(target, "", "edit", first.revision,
                                        edits=[{"old_str": "## Мост\n\nСтарый мост.\n\n### Опоры\n\nДве.\n\n", "new_str": ""},
                                               {"old_str": "[Old gap](old-gap.md)", "new_str": "[New gap](new-gap.md)"}]).current
    store.write_knowledge_note(address(tmp_path, "later-source"), "[Now](notes/река.md)\n")
    old = tools._knowledge_read(ctx, TOPIC, version=first.revision)
    assert '[Knowledge links] {"unwritten": [{"scope": "global", "topic": "notes/old-gap"}]}' in old
    assert '[Knowledge backlinks (readable shelves now)] [{"scope": "global", "topic": "later-source", "state": "active"}]' in old
    assert '"topic": "notes/new-gap"' in tools._knowledge_read(ctx, TOPIC)
    assert old.endswith(first.text) and "Earlier version, not the current note" in old
    assert f"The current revision is {second.revision}" in old and "focus unknown (overwrite; reason: not recorded)" in old
    assert tools._knowledge_read(ctx, TOPIC, version=first.revision, start_char=3, end_char=40).endswith(first.text[3:40])
    outline = tools._knowledge_read(ctx, TOPIC, version=first.revision, view="outline")
    assert "an earlier version, not the current note" in outline and "### Опоры" in outline
    assert f"Revision: {first.revision}" in outline
    # Its ranges are of that version, not of the current note: the outline names the version to read them with.
    assert f"The ranges are of this version itself, read with the default view and version={first.revision}." in outline
    start, end = map(int, re.search(r"^### Опоры — line \d+, chars (\d+)\.\.(\d+) ", outline, re.M).groups())
    assert tools._knowledge_read(ctx, TOPIC, version=first.revision, start_char=start, end_char=end).endswith(
        "### Опоры\n\nДве.\n\n") and first.text[start:end] == "### Опоры\n\nДве.\n\n"
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
                            ({"version": " "}, ["version=' ' ignored: no revision named; this is the current note"]),
                            ({"version": "\t\n"}, ["version='\\t\\n' ignored: no revision named"]),
                            ({"view": ""}, ["view='' ignored: the default view, the note itself"]),
                            ({"view": None}, ["view=None ignored: the default view, the note itself"]),
                            ({"view": None, "version": ""}, ["view=None ignored", "version='' ignored"])):
        read = registry.execute("knowledge_read", {"topic": TOPIC, "scope": "global", **arguments})
        assert read.endswith(note.text) and read.split("\n", 1)[0] == plain.split("\n", 1)[0], (arguments, read[:300])
        assert all(f"[Argument note] {item}" in read for item in said), (arguments, read[:600])
    history = registry.execute("knowledge_read", {"topic": TOPIC, "scope": "global", "view": "history", "version": ""})
    assert history.startswith("[Knowledge history]") and "[Argument note] version='' ignored" in history
    history = registry.execute("knowledge_read", {"topic": TOPIC, "scope": "global", "view": "history", "version": " "})
    assert history.startswith("[Knowledge history]") and "[Argument note] version=' ' ignored" in history
    for malformed in ("abc", " abc "):  # a value that names something still refuses, typed
        read = registry.execute("knowledge_read", {"topic": TOPIC, "scope": "global", "version": malformed})
        assert read.startswith("⚠️ TOOL_ERROR: version_unavailable: a version is a revision") and note.text not in read
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
    receipt = registry.execute("knowledge_write", {"topic": "plain", "scope": "global", "mode": "overwrite",
                                                   "content": "# Again\n", "expected_revision": empty})
    assert receipt.startswith("✅"), receipt
    meta = json.loads(receipt.split("\n", 2)[1])
    assert meta["knowledge_previous"]["revision"] == empty
    assert json.loads(journal.read_text(encoding="utf-8").splitlines()[-1])["old_sha256"] == empty
    stale = registry.execute("knowledge_write", {"topic": "plain", "scope": "global", "mode": "overwrite",
                                                 "content": "Mine.\n", "expected_revision": empty})  # the empty version, held
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
    error = f"⚠️ TOOL_ERROR: version_unavailable: no version {missing} of this note in its history"
    situation = "This situation's names: room:1 (Main)"
    assert clean.splitlines() == [error, situation]
    journal = target.shelf.parent / "knowledge_history.jsonl"
    offset = journal.stat().st_size
    with journal.open("a", encoding="utf-8") as stream:  # torn before its revision fields
        stream.write(json.dumps({"topic": TOPIC, "mode": "edit"}, ensure_ascii=False)[:-1] + ', "new_sh\n')
    torn = tools._knowledge_read(ctx, TOPIC, version=missing)
    assert torn.splitlines() == [
        f"{error} (1 journal line(s) naming this note are not JSON and were skipped (the first at byte offset {offset}))",
        situation,
    ]


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


def test_a_topic_that_is_not_ascii_is_found_by_one_pass_over_the_journal(tmp_path, monkeypatch):
    """Rows are found by the topic as the one writer spells it (JSON with ensure_ascii=False), each search starting
    below the line last found: a non-ASCII topic costs about one pass over the journal, however many rows it has.
    A spelling no row carries would rescan the whole block for every row found."""
    target = address(tmp_path)
    note = store.write_knowledge_note(target, "# Река\n\nv0\n").current
    for index in range(1, 30):
        store.write_knowledge_note(address(tmp_path, f"notes/other-{index}"), f"# Other {index}\n")
        note = store.write_knowledge_note(target, f"v{index}", "edit", note.revision, old_str=f"v{index - 1}").current
    journal = target.shelf.parent / "knowledge_history.jsonl"
    looked_at, real = [0], versions._matching_lines

    class Counted(bytes):  # the bytes each backward search covers, from its end down to its hit (or its start)
        def rfind(self, sub, start=0, end=None):
            end = len(self) if end is None else end
            hit = bytes.rfind(self, sub, start, end)
            looked_at[0] += end - (start if hit < 0 else hit)
            return hit

    monkeypatch.setattr(versions, "_matching_lines", lambda region, base, *needles: real(Counted(region), base, *needles))
    rows = list(versions.NoteHistory(target))
    assert len(rows) == 30 and rows[0][1]["new_sha256"] == note.revision and all(row["topic"] == TOPIC for _, row in rows)
    assert looked_at[0] <= 2 * journal.stat().st_size, (looked_at[0], journal.stat().st_size)
    looked_at[0] = 0  # the stale-revision lookup the writer makes under the shelf lock reads the tail the same way
    assert len(list(versions.tail_rows(journal, target, 1 << 23))) == 30
    assert looked_at[0] <= 2 * journal.stat().st_size, (looked_at[0], journal.stat().st_size)


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


def test_stale_write_chooses_diff_or_source_before_attaching_the_names_line(tmp_path, monkeypatch):
    from ouroboros.artifacts import read_actor_source_bytes
    from ouroboros.tool_result_delivery import project_tool_result_batch

    monkeypatch.setattr(store, "utc_now_iso", lambda: "2026-11-18T12:00:00+00:00")
    registry = ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path)
    root = context(tmp_path)
    child = ToolContext(repo_dir=tmp_path, drive_root=tmp_path, task_id="child",
                        task_metadata={"delegation_role": "subagent"})
    names_line = "This situation's names: room:1 (Main)\n"

    def conflict(topic, reason):
        target = address(tmp_path, topic)
        old = "# Plan\n\nBefore.\n\n" + "Steady words.\n" * 1600
        held = store.write_knowledge_note(target, old).current
        current = store.write_knowledge_note(target, held.text.replace("Before.", "After."),
                                             expected_revision=held.revision, reason=reason).current
        args = {"topic": topic, "scope": "global", "mode": "overwrite", "content": "Stale edit.",
                "expected_revision": held.revision}
        registry.set_context(child)
        control = registry.execute_result("knowledge_write", args)
        registry.set_context(root)
        reply = registry.execute_result("knowledge_write", args)
        assert control.status == reply.status == "error" and target.path.read_bytes() == current.raw
        assert reply.text.count(names_line) == 1 and reply.text.replace(names_line, "", 1) == control.text
        return control, reply, held, current

    limit = tools.tool_result_limit("knowledge_write")
    probe, _, _, _ = conflict("probe", "r")
    padding = limit - len(probe.text)
    assert probe.meta["knowledge_changes_since"] and padding > 0
    fit, diff, held, current = conflict("equal", "r" * (padding + 1))
    assert len(fit.text) == limit and len(diff.text) > limit
    assert diff.meta["knowledge_changes_since"] == held.revision
    assert applied_blocks(held.text, diff.text) == current.text
    _, source, _, large_current = conflict("large", "r" * (padding + 2))
    assert "knowledge_changes_since" not in source.meta
    assert source.text[source.meta["knowledge_body_start"]:] == large_current.text

    for position, result in enumerate((diff, source)):
        row = {"tool_call_id": f"stale-{position}", "fn_name": "knowledge_write", "result": result.text,
               "result_meta": result.meta, "is_error": True}
        projected, receipt = project_tool_result_batch([row], [], [], drive_root=tmp_path, task_id=root.task_id,
                                                       fit_candidate=lambda messages, _: {"accepted": len(messages[-1]["content"]) <= 2500})
        assert receipt["status"] == "projected" and projected[0]["result_partial"]
        assert read_actor_source_bytes(tmp_path, root.task_id, projected[0]["result_source_ref"]) == result.text.encode()


@pytest.mark.parametrize("eol", ["\n", "\r\n"])
def test_a_heading_whose_title_spans_lines_is_named_by_its_first_line_in_block_headers_and_history_rows(tmp_path, eol):
    """A paragraph followed by --- is a level-2 heading whose title spans lines. A stale reply's block header names
    it by its first line as the note writes it (the outline's form), so each header stays one line and the reply
    still applies; a history row names it by its first line too."""
    setext = f"The review runs weekly{eol}and the owner reads it first.{eol}---{eol}"
    target = address(tmp_path, "notes/weekly")
    held = store.write_knowledge_note(target, f"# Notes{eol}{eol}Intro.{eol}{eol}{setext}{eol}Timeout: 3.{eol}").current
    current = store.write_knowledge_note(target, held.text.replace("Timeout: 3.", "Timeout: 5."),
                                         expected_revision=held.revision).current
    stale = tools._knowledge_write(context(tmp_path), "notes/weekly", "Mine.", mode="overwrite", scope="global",
                                   expected_revision=held.revision)
    assert "revision_conflict" in stale and "1 block(s)" in stale, stale
    line = held.text[:held.text.index("Timeout: 3.")].count("\n") + 1  # after the frontmatter the writer adds
    assert (f"\n@ yours: 1 line(s) from line {line} → current: 1 line(s) from line {line}, under The review runs "
            "weekly: 1 chars removed, 1 added\n") in stale
    assert all(line.endswith(" added") for line in stale.split("\n") if line.startswith("@ yours: ")), stale
    assert applied_blocks(held.text, stale) == current.text
    gone = store.write_knowledge_note(target, current.text.replace(setext + eol, ""), expected_revision=current.revision)
    listing = versions.render_history(gone.current)
    assert all(line.startswith(("History of ", "- ", "Read one: ")) for line in listing.splitlines()), listing
    assert f"sections removed: ## The review runs weekly ({len(setext)} chars) · current\n" in listing, listing
    assert "headings added: # Notes; ## The review runs weekly\n" in listing, listing


def test_removed_sections_name_heading_size_and_first_line_cut_to_120(tmp_path):
    target = address(tmp_path)
    long_line = "Первая строка под заголовком. " * 6
    original = store.write_knowledge_note(target, f"# Keep\n\nx\n\n## Gone\n\n\n{long_line}\nsecond\n").current
    changed = store.write_knowledge_note(target, "# Keep\n\nx\n", expected_revision=original.revision)
    assert changed.delta["removed_sections"] == [{"heading": "## Gone", "first_line": long_line.strip()[:120],
                                                  "chars": len(f"## Gone\n\n\n{long_line}\nsecond\n")}]
    assert changed.delta["removed_headings"] == ["## Gone"]


def test_versions_stored_as_changes_read_back_exactly_and_list_as_stored(tmp_path):
    """Rows that carry a change against their base read back through the version lookup, the version read and the
    stale-revision lookup, exactly; the listing counts such a row's text as stored without rebuilding any version."""
    ctx, target = context(tmp_path), address(tmp_path)
    bridge = RIVER.replace("Старый мост.", "Новый мост.")
    texts = [RIVER, bridge, bridge.replace("Две.", "Три.")]
    journal = target.shelf.parent / "knowledge_history.jsonl"
    changed_rows(journal, texts, TOPIC)
    target.path.parent.mkdir(parents=True, exist_ok=True)
    target.path.write_bytes(texts[-1].encode("utf-8"))
    stored = [json.loads(line) for line in journal.read_text(encoding="utf-8").splitlines()]
    assert all("change" in row and "new_content" not in row for row in stored[1:])
    revisions = [hashlib.sha256(text.encode("utf-8")).hexdigest() for text in texts]
    for text, revision in zip(texts, revisions):
        found, row = versions.find_version(target, revision)
        assert found == text and row["new_sha256"] == revision
    assert tools._knowledge_read(ctx, TOPIC, version=revisions[1]).endswith(texts[1])
    held = versions.held_version(journal, target, revisions[1], 1 << 23)
    assert held["text"] == texts[1] and [item["revision"] for item in held["since"]] == [revisions[2]]
    listing = versions.render_history(store.read_knowledge_note(target))
    assert "text not stored" not in listing and [line.split(" · ")[0][2:] for line in listing.splitlines()
                                                 if line.startswith("- ")] == revisions[::-1]


def test_writes_reads_and_a_revert_round_trip_over_rows_that_store_changes(tmp_path):
    """knowledge_write → knowledge_read(version=…) → mode='revert' when every row after the first stores a change:
    each earlier version reads back exactly, the revert publishes the first version's bytes, and every version,
    the ones reverted over included, still reads."""
    registry = ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path)
    registry.set_context(context(tmp_path))
    target, sha = address(tmp_path), (lambda text: hashlib.sha256(text.encode("utf-8")).hexdigest())
    note = store.write_knowledge_note(target, RIVER).current
    texts = [note.text]
    for old, new in (("Старый мост.", "Новый мост."), ("Две.", "Три."), ("Всё.", "Всё, до встречи.")):
        reply = registry.execute("knowledge_write", {"topic": TOPIC, "scope": "global", "mode": "edit", "old_str": old,
                                                     "content": new, "expected_revision": note.revision})
        assert reply.startswith("✅"), reply
        note = store.read_knowledge_note(target)
        texts.append(note.text)
    journal = target.shelf.parent / "knowledge_history.jsonl"
    rows = [json.loads(line) for line in journal.read_text(encoding="utf-8").splitlines()]
    assert all("change" in row and not {"new_content", "old_content"} & set(row) for row in rows[1:])
    for text in texts[:-1]:
        read = registry.execute("knowledge_read", {"topic": TOPIC, "scope": "global", "version": sha(text)})
        assert read.endswith(text) and "Earlier version, not the current note" in read, read[:400]
    reverted = registry.execute("knowledge_write", {"topic": TOPIC, "scope": "global", "mode": "revert",
                                                    "version": sha(texts[0]), "expected_revision": note.revision,
                                                    "reason": "back to the start"})
    assert reverted.startswith("✅") and target.path.read_bytes() == texts[0].encode("utf-8"), reverted
    row = json.loads(journal.read_text(encoding="utf-8").splitlines()[-1])
    assert (row["reverted_from"], row["base"]["sha256"], row["new_sha256"]) == (sha(texts[0]), sha(texts[-1]), sha(texts[0]))
    for text in texts:
        assert versions.find_version(target, sha(text))[0] == text
    assert registry.execute("knowledge_read", {"topic": TOPIC, "scope": "global", "version": sha(texts[2])}).endswith(texts[2])


def test_a_row_marked_digested_lists_its_text_as_not_stored_whatever_else_it_carries(tmp_path):
    """The listing and the reader agree on a digest-era mark: a row marked ``content_digested`` holds no text, even
    beside a change and a base, so it lists as "text not stored" and its version is a typed miss; a change row with a
    base lists as stored."""
    target, sha = address(tmp_path), (lambda text: hashlib.sha256(text.encode("utf-8")).hexdigest())
    journal = target.shelf.parent / "knowledge_history.jsonl"
    texts = [RIVER, RIVER.replace("Старый мост.", "Новый мост.")]
    changed_rows(journal, texts, TOPIC)
    change_row = json.loads(journal.read_bytes().splitlines()[-1])
    assert "change" in change_row and "new_content" not in change_row
    digested = {**change_row, "new_sha256": sha("a digested version"), "content_digested": True}
    with journal.open("ab") as stream:
        stream.write((json.dumps(digested, ensure_ascii=False) + "\n").encode("utf-8"))
    target.path.parent.mkdir(parents=True, exist_ok=True)
    target.path.write_bytes(texts[-1].encode("utf-8"))
    rows = [line for line in versions.render_history(store.read_knowledge_note(target)).splitlines() if line.startswith("- ")]
    assert rows[0].startswith(f"- {digested['new_sha256']} · ") and rows[0].endswith("text not stored"), rows[0]
    assert rows[1].startswith(f"- {sha(texts[1])} · ") and "text not stored" not in rows[1], rows[1]
    with pytest.raises(versions.VersionUnavailable, match=r"holds no text \(digest-era\)"):
        versions.find_version(target, digested["new_sha256"])


def test_a_version_whose_forward_path_is_damaged_reads_through_the_live_note(tmp_path):
    """The version lookup hands the reader the note's current text: with B's change row damaged, B comes back from the
    live note through the change of the row after it; with the note gone it is a typed miss that names the line."""
    target = address(tmp_path)
    a = store.write_knowledge_note(target, "# Река\n\nOne.\n").current
    b = store.write_knowledge_note(target, "# Река\n\nOne, two.\n", expected_revision=a.revision).current
    store.write_knowledge_note(target, "# Река\n\nOne, two, three.\n", expected_revision=b.revision)
    journal = target.shelf.parent / "knowledge_history.jsonl"
    raw = journal.read_bytes()
    b_at = raw.index(b"\n") + 1
    end = raw.index(b"\n", b_at)
    rows = [json.loads(line) for line in raw.splitlines()]
    assert len(rows) == 3 and all("change" in row and "new_content" not in row for row in rows[1:])
    journal.write_bytes(raw[:b_at] + b"#" * (end - b_at) + raw[end:])
    assert versions.find_version(target, b.revision) == (b.text, {})  # from the next row's old side, rebuilt
    target.path.unlink()
    with pytest.raises(versions.VersionUnavailable, match=f"the line at byte offset {b_at} is not a JSON row"):
        versions.find_version(target, b.revision)


def test_a_history_continuation_is_bound_to_the_list_its_first_page_read(tmp_path):
    """K6: a later page names its list's fingerprint; after a write and a revert back to the same revision the list
    differs although the note's revision does not, and the old continuation is refused, never spliced."""
    ctx, target = context(tmp_path), address(tmp_path)
    a = store.write_knowledge_note(target, "# Река\n\nOne.\n").current
    b = store.write_knowledge_note(target, "# Река\n\nTwo.\n", expected_revision=a.revision).current
    first = tools._knowledge_read(ctx, TOPIC, view="history", end_char=120)
    fingerprint = re.search(r'"fingerprint": "([0-9a-f]{16})"', first)[1]
    same = tools._knowledge_read(ctx, TOPIC, view="history", start_char=120, fingerprint=fingerprint)
    whole = tools._knowledge_read(ctx, TOPIC, view="history")
    assert body(first) + body(same) == body(whole)  # an unchanged list continues exactly
    assert "position_mismatch" in tools._knowledge_read(ctx, TOPIC, view="history", start_char=120)  # no list named
    c = store.write_knowledge_note(target, "# Река\n\nThree.\n", expected_revision=b.revision).current
    assert store.write_knowledge_note(target, "", "revert", c.revision, version=b.revision).current.revision == b.revision
    moved = tools._knowledge_read(ctx, TOPIC, view="history", start_char=120, fingerprint=fingerprint)
    assert moved.startswith("⚠️ TOOL_ERROR: position_mismatch") and "Two." not in moved, moved
    assert "fingerprint continues a list of versions" in tools._knowledge_read(ctx, TOPIC, fingerprint=fingerprint)


def test_each_history_row_prints_the_call_that_undoes_its_own_write(tmp_path):
    """K8: every row prints the revert call to the version ITS write replaced (that old_sha256, the current revision,
    the note's address), never the line below's; running the printed call of a write over an outside edit puts that
    outside text back exactly; a creating row says there is no earlier version; a row whose replaced version is the
    current text says there is nothing to put back."""
    registry = ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path)
    registry.set_context(context(tmp_path))
    target = address(tmp_path)
    first = store.write_knowledge_note(target, "# Река\n\nOne.\n").current
    target.path.write_bytes(first.raw.replace(b"One.", b"Outside."))  # no row writes this version
    outside = store.read_knowledge_note(target)
    second = store.write_knowledge_note(target, "# Река\n\nTwo.\n", expected_revision=outside.revision).current
    third = store.write_knowledge_note(target, "# Река\n\nThree.\n", expected_revision=second.revision).current
    rows = [line for line in versions.render_history(third).splitlines() if line.startswith("- ")]
    assert [row.split(" · ")[0][2:] for row in rows] == [third.revision, second.revision, first.revision]
    calls = [re.search(r" · undo: (knowledge_write\(.*?\)) · ", row) for row in rows]
    where = {"topic": TOPIC, "scope": "global", "mode": "revert", "expected_revision": third.revision}
    assert printed_call(calls[0][1]) == ("knowledge_write", {**where, "version": second.revision}), rows[0]
    assert printed_call(calls[1][1]) == ("knowledge_write", {**where, "version": outside.revision}), rows[1]
    assert first.revision not in rows[1] and calls[2] is None  # never the adjacent row; a creation has no undo
    assert " · created it: no earlier version to put back · " in rows[2] and "undo" not in rows[2], rows[2]
    tool, arguments = printed_call(calls[1][1])
    undone = registry.execute(tool, arguments)
    assert undone.startswith("✅") and target.path.read_bytes() == outside.raw, undone
    after = [line for line in versions.render_history(store.read_knowledge_note(target)).splitlines()
             if line.startswith("- ")]
    assert f"version={third.revision!r}, expected_revision={outside.revision!r})" in after[0], after[0]
    assert " · undo: nothing to put back, that version is the current text · " in after[2], after[2]


def test_a_bad_position_or_fingerprint_is_an_argument_error_before_the_list_is_compared(tmp_path):
    """C4: a start_char or end_char that is negative or not an integer, and a fingerprint that is not text, are argument
    errors naming the argument (TOOL_ARG_ERROR), before the list's fingerprint is compared: never position_mismatch. A
    valid position with another list's fingerprint is still position_mismatch, and the right one still continues."""
    ctx, target = context(tmp_path), address(tmp_path)
    a = store.write_knowledge_note(target, "# Река\n\nOne.\n").current
    store.write_knowledge_note(target, "# Река\n\nTwo.\n", expected_revision=a.revision)
    first = tools._knowledge_read(ctx, TOPIC, view="history", end_char=120)
    fingerprint = re.search(r'"fingerprint": "([0-9a-f]{16})"', first)[1]
    for arguments, named in (({"start_char": -1}, "start_char=-1 is not a position"),
                             ({"start_char": "120"}, "start_char='120' is not a position"),
                             ({"start_char": 1.5}, "start_char=1.5 is not a position"),
                             ({"start_char": 10, "end_char": -5}, "end_char=-5 is not a position"),
                             ({"start_char": 120, "fingerprint": 123}, "fingerprint is the text")):
        reply = tools._knowledge_read(ctx, TOPIC, view="history", **arguments)
        assert reply.startswith("⚠️ TOOL_ARG_ERROR: ") and named in reply and "position_mismatch" not in reply, (
            arguments, reply)
    assert "position_mismatch" in tools._knowledge_read(ctx, TOPIC, view="history", start_char=120, fingerprint="0" * 16)
    later = tools._knowledge_read(ctx, TOPIC, view="history", start_char=120, fingerprint=fingerprint)
    assert later.startswith("[Knowledge history]") and body(first) + body(later) == body(
        tools._knowledge_read(ctx, TOPIC, view="history")), later


def test_a_first_page_with_another_lists_fingerprint_is_told_to_read_without_one(tmp_path):
    """R5: a first page that carries another list's fingerprint is refused (position_mismatch) and told to read
    without the fingerprint, not to start again at the start it already had; a later page is still told to read again
    from start_char=0."""
    ctx, target = context(tmp_path), address(tmp_path)
    a = store.write_knowledge_note(target, "# Река\n\nOne.\n").current
    store.write_knowledge_note(target, "# Река\n\nTwo.\n", expected_revision=a.revision)
    first = tools._knowledge_read(ctx, TOPIC, view="history", fingerprint="0" * 16)
    assert first.startswith("⚠️ TOOL_ERROR: position_mismatch: the fingerprint names another list of versions") and \
        first.splitlines()[0].endswith("a first page takes none, so read it without the fingerprint") and "start_char=0" not in first, first
    later = tools._knowledge_read(ctx, TOPIC, view="history", start_char=40, fingerprint="0" * 16)
    assert later.splitlines()[0].endswith("read it again from start_char=0"), later
    assert tools._knowledge_read(ctx, TOPIC, view="history").startswith("[Knowledge history]")


def test_an_archive_or_restore_row_offers_the_opposite_lifecycle_call(tmp_path):
    """R7: the writer refuses a revert across archive states, so a note's archive or restore row prints the opposite
    lifecycle call as its undo (restore; archive with the reason it requires), never a revert; content rows keep their
    revert calls. The archive row's printed restore, run, brings the note back active with its bytes; the restore row
    then offers the archive, and the archive row, whose replaced version is current again, has nothing to put back."""
    registry = ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path)
    registry.set_context(context(tmp_path))
    target = address(tmp_path)
    first = store.write_knowledge_note(target, "# Река\n\nOne.\n").current
    second = store.write_knowledge_note(target, "# Река\n\nTwo.\n", expected_revision=first.revision).current
    archived = store.write_knowledge_note(target, "", "archive", second.revision, reason="done with it").current
    assert archived.state == "archived"
    rows = [line for line in versions.render_history(archived).splitlines() if line.startswith("- ")]
    where = f"topic={TOPIC!r}, scope='global'"
    assert f" · undo: knowledge_write({where}, mode='restore', expected_revision={archived.revision!r}) · " in rows[0], rows[0]
    assert "mode='revert'" not in rows[0] and f"mode='revert', version={first.revision!r}" in rows[1], rows
    listing = versions.render_history(archived)
    assert "an archive or restore line's undo is the opposite lifecycle mode, which changes the archive field only." in listing
    tool, arguments = printed_call(re.search(r" · undo: (knowledge_write\(.*?\)) · ", rows[0])[1])
    restored = registry.execute(tool, arguments)
    current = store.read_knowledge_note(target)
    assert restored.startswith("✅") and current.state == "active" and current.raw == second.raw, restored
    rows = [line for line in versions.render_history(current).splitlines() if line.startswith("- ")]
    assert (f" · undo: knowledge_write({where}, mode='archive', reason=<why>, expected_revision={current.revision!r}) · "
            in rows[0]), rows[0]
    assert " · undo: nothing to put back, that version is the current text · " in rows[1], rows[1]
    plain = versions.render_history(store.write_knowledge_note(address(tmp_path, "plain"), "# Plain\n").current)
    assert "lifecycle" not in plain  # a history without archive or restore rows says nothing of them
