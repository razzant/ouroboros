"""The plain memory texts' editor (memory_texts), through identity's writer: edit, revert, unchanged, stale revisions,
empty results, a missing file and knowledge_write's argument rules."""
from __future__ import annotations

import hashlib
import json
import re

import pytest

from ouroboros.memory import Memory
from ouroboros.tools.control_runtime import _update_identity
from ouroboros.tools.registry import ToolContext, ToolRegistry

IDENTITY = ("# Who I am\n\nOne continuous mind across my rooms.\n\n## Commitments\n\nKeep what I learn.\n"
            "Read before I revise.\n\n## Doubts\n\nWhether a page can hold a self.\n")


def _ctx(root, task="focus-1"):
    return ToolContext(repo_dir=root / "repo", drive_root=root, task_id=task)


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _rows(root):
    path = root / "memory" / "identity_journal.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()] if path.exists() else []


@pytest.fixture
def identity(tmp_path):
    path = tmp_path / "memory" / "identity.md"
    path.parent.mkdir(parents=True)
    path.write_bytes(IDENTITY.encode("utf-8"))
    return path


def test_an_edit_changes_only_its_items_and_the_receipt_says_what_left(tmp_path, identity):
    ctx = _ctx(tmp_path)
    first = _update_identity(ctx, IDENTITY.replace("Keep", "Keep, and share,"), _sha(identity.read_bytes()),
                             reason="sharing matters")
    assert first.startswith("OK: identity updated (overwrite"), first
    current = _sha(identity.read_bytes())
    reply = _update_identity(ctx, mode="edit", expected_revision=current, reason="doubt resolved", edits=[
        {"old_str": "## Doubts\n\nWhether a page can hold a self.\n", "new_str": ""},
        {"old_str": "Read before I revise.", "new_str": "Read before I revise; say what left."}])
    assert reply.startswith("OK: identity updated (edit"), reply
    text = identity.read_text(encoding="utf-8")
    assert text == IDENTITY.replace("Keep", "Keep, and share,").replace(
        "## Doubts\n\nWhether a page can hold a self.\n", "").replace("Read before I revise.",
                                                                     "Read before I revise; say what left.")
    assert "doubt resolved" not in text  # the reason lives in history, never in the text
    assert f"Replaced {current}: written " in reply and "by focus root task_id=focus-1 room_id=1 (overwrite; reason: " \
        "sharing matters)" in reply, reply
    assert "sections removed: ## Doubts" in reply and re.search(r"\d+ chars removed, \d+ added", reply), reply
    assert f"memory_read(source='identity', version='{current}')" in reply
    new = _sha(identity.read_bytes())
    assert f"update_identity(mode='revert', version='{current}', expected_revision='{new}')" in reply
    row = _rows(tmp_path)[-1]
    assert (row["mode"], row["reason"], row["old_sha256"], row["new_sha256"]) == ("edit", "doubt resolved", current, new)
    assert row["old_content"] and row["new_content"] == text and row["delta"]["removed_sections"][0]["heading"] == "## Doubts"


def test_one_pair_spelling_and_whitespace_anchor_are_the_note_editor_s(tmp_path, identity):
    ctx = _ctx(tmp_path)
    revision = _sha(identity.read_bytes())
    via_new_str = _update_identity(ctx, mode="edit", old_str="One continuous", new_str="One unbroken",
                                   expected_revision=revision)
    assert via_new_str.startswith("OK:"), via_new_str
    revision = _sha(identity.read_bytes())
    gap = _update_identity(ctx, "\n\n\n", mode="edit", old_str="\n\n## Doubts", expected_revision=revision)
    assert gap.startswith("OK:") and "rooms.\n\n## Commitments" in identity.read_text(encoding="utf-8"), gap
    assert "\n\n\n\nWhether" in identity.read_text(encoding="utf-8")


@pytest.mark.parametrize(("items", "said"), [
    ([{"old_str": "nowhere", "new_str": "x"}], "item 1 (old_str): not found"),
    ([{"old_str": "Keep what I learn.", "new_str": "a"}, {"old_str": "\n\n", "new_str": "b"}],
     "item 2 (old_str): found 5 times at lines 1, 3, 5, 8, 10"),
    ([{"old_str": "Keep what", "new_str": "a"}, {"old_str": "what I learn", "new_str": "b"}], "item 2: overlaps item 1"),
])
def test_a_refused_item_names_itself_and_writes_nothing(tmp_path, identity, items, said):
    before = identity.read_bytes()
    reply = _update_identity(_ctx(tmp_path), mode="edit", edits=items, expected_revision=_sha(before))
    assert reply.startswith("⚠️ TOOL_ERROR: edit_refused: ") and said in reply, reply
    assert identity.read_bytes() == before and _rows(tmp_path) == []


def test_revert_puts_back_an_earlier_version_byte_for_byte(tmp_path, identity):
    ctx = _ctx(tmp_path)
    original = identity.read_bytes()
    first = _sha(original)
    _update_identity(ctx, "A shorter self.\r\n", first)  # no fifty-character floor, no shrink notice
    assert identity.read_bytes() == b"A shorter self.\r\n"
    reverted = _update_identity(ctx, mode="revert", version=first, expected_revision=_sha(identity.read_bytes()),
                                reason="that rewrite lost too much")
    assert reverted.startswith("OK: identity updated (revert") and f"Put back version {first}." in reverted, reverted
    assert identity.read_bytes() == original and f"revision={first}." in reverted
    row = _rows(tmp_path)[-1]
    assert (row["reverted_from"], row["new_sha256"], row["mode"]) == (first, first, "revert")
    missing = _update_identity(ctx, mode="revert", version="0" * 64, expected_revision=first)
    assert missing.startswith("⚠️ TOOL_ERROR: version_unavailable: no version " + "0" * 64 + " of this identity")
    assert identity.read_bytes() == original


def test_an_identical_result_is_unchanged_and_writes_no_row(tmp_path, identity):
    ctx = _ctx(tmp_path)
    revision = _sha(identity.read_bytes())
    same = _update_identity(ctx, IDENTITY, revision)
    assert same.startswith("OK: identity unchanged (overwrite)") and _rows(tmp_path) == [], same
    noop = _update_identity(ctx, mode="edit", old_str="Keep", content="Keep", expected_revision=revision)
    assert noop.startswith("OK: identity unchanged (edit)") and _rows(tmp_path) == []
    current = _update_identity(ctx, mode="revert", version=revision, expected_revision=revision)
    assert current.startswith("OK: identity unchanged (revert)") and _rows(tmp_path) == [], current


def test_an_empty_result_is_an_argument_error_in_every_mode(tmp_path, identity):
    ctx = _ctx(tmp_path)
    revision = _sha(identity.read_bytes())
    for reply in (_update_identity(ctx, "  \n", revision),
                  _update_identity(ctx, mode="edit", old_str=IDENTITY, content=" ", expected_revision=revision)):
        assert reply.startswith("⚠️ TOOL_ARG_ERROR: ") and "never empty" in reply, reply
    assert identity.read_bytes() == IDENTITY.encode("utf-8") and _rows(tmp_path) == []
    identity.write_bytes(b"")  # an empty version on file: putting it back is refused too
    empty = _sha(b"")
    _update_identity(ctx, "Back again.", empty)
    reverted = _update_identity(ctx, mode="revert", version=empty, expected_revision=_sha(identity.read_bytes()))
    assert reverted.startswith("⚠️ TOOL_ARG_ERROR: the revert leaves the identity empty"), reverted
    assert identity.read_bytes() == b"Back again."


def test_edit_or_revert_of_a_missing_identity_is_refused_and_overwrite_creates_it(tmp_path):
    ctx = _ctx(tmp_path)
    for mode, extra in (("edit", {"old_str": "a", "content": "b"}), ("revert", {"version": "1" * 64})):
        reply = _update_identity(ctx, mode=mode, expected_revision="", **extra)
        assert reply.startswith(f"⚠️ TOOL_ERROR: {mode}_source_missing"), reply
    created = _update_identity(ctx, "I begin.")
    assert created.startswith("OK: identity created (overwrite") and Memory(tmp_path).identity_path().read_bytes() == b"I begin."
    assert _rows(tmp_path)[-1]["old_sha256"] == ""


def test_a_stale_revision_gets_every_change_since_the_version_it_holds(tmp_path, identity):
    _update_identity(_ctx(tmp_path), IDENTITY + "\nHeld.\n", _sha(identity.read_bytes()))
    held = _sha(identity.read_bytes())  # a version a row wrote: the journal's tail holds it
    _update_identity(_ctx(tmp_path, "other"), identity.read_text(encoding="utf-8").replace(
        "Keep what I learn.", "Keep what I learn, dated."), held)
    stale = _update_identity(_ctx(tmp_path), "Mine.", held)
    assert "revision_conflict" in stale and "1 block(s)" in stale and "focus root task_id=other" in stale, stale
    assert "\n- Keep what I learn.\n+ Keep what I learn, dated.\n" in stale
    assert "memory_read(source='identity', version=" in stale and identity.read_text(encoding="utf-8").count("dated") == 1


def test_a_stale_revision_no_row_holds_gets_the_whole_text_or_its_read(tmp_path, identity, monkeypatch):
    identity.write_bytes(b"Written outside the writer.\n")
    stale = _update_identity(_ctx(tmp_path), "Mine.", "0" * 64)
    assert "revision_conflict" in stale and stale.split("\n\n", 2)[2].startswith("Written outside the writer.\n"), stale
    monkeypatch.setattr("ouroboros.tool_capabilities.tool_result_limit", lambda _name: 200)
    clipped = _update_identity(_ctx(tmp_path), "Mine.", "0" * 64)
    current = _sha(identity.read_bytes())
    assert "does not fit this result" in clipped and f"memory_read(source='identity', version='{current}')" in clipped
    assert "Written outside" not in clipped  # never a cut text


@pytest.mark.parametrize(("arguments", "said"), [
    ({"content": "x", "old_str": "Keep"}, "old_str is used only with mode=edit"),
    ({"content": "x", "edits": [{"old_str": "a", "new_str": "b"}]}, "edits is used only with mode=edit"),
    ({"mode": "edit", "edits": []}, "mode=edit requires edits, or a non-empty old_str ("),
    ({"mode": "revert"}, "mode=revert requires version"),
    ({}, "mode=overwrite requires content"),
    ({"mode": "append", "content": "x"}, "mode='append' is not an operation. Pass mode as exactly one of overwrite"),
    ({"content": ["x"]}, "content must be Markdown text, not list"),
    ({"content": "x", "version": "abc"}, "version is used only with mode=revert; memory_read(source='identity'"),
])
def test_the_argument_rules_are_knowledge_writes(tmp_path, identity, arguments, said):
    reply = _update_identity(_ctx(tmp_path), expected_revision=_sha(identity.read_bytes()), **arguments)
    assert reply.startswith("⚠️ TOOL_ARG_ERROR: ") and said in reply, reply
    assert identity.read_bytes() == IDENTITY.encode("utf-8")


def test_an_empty_expected_revision_is_create_only(tmp_path, identity):
    refused = _update_identity(_ctx(tmp_path), "Replaced blindly.", "")
    assert "revision_conflict" in refused and identity.read_bytes() == IDENTITY.encode("utf-8")
    assert "revision_required" in _update_identity(_ctx(tmp_path), "Replaced blindly.")


def test_the_registry_admits_an_edit_without_content(tmp_path, identity):
    registry = ToolRegistry(repo_dir=tmp_path / "repo", drive_root=tmp_path)
    registry.set_context(_ctx(tmp_path))
    reply = registry.execute("update_identity", {"mode": "edit", "expected_revision": _sha(identity.read_bytes()),
                                                 "edits": [{"old_str": "Doubts", "new_str": "Questions"}]})
    assert reply.startswith("OK: identity updated (edit"), reply
    assert "## Questions" in identity.read_text(encoding="utf-8")


# --- memory_read(source=...): the version door ------------------------------------------------------------------------

def _read(ctx, **arguments):
    from ouroboros.tools.chronicle import _memory_read

    return _memory_read(ctx, **arguments)


def test_the_door_lists_versions_newest_first_and_reads_one_without_the_chronicle(tmp_path, identity, monkeypatch):
    import ouroboros.tools.chronicle as chronicle

    monkeypatch.setattr(chronicle, "_activated", lambda root: (_ for _ in ()).throw(AssertionError("chronicle woke")))
    ctx = _ctx(tmp_path)
    outside = _sha(identity.read_bytes())  # written before history: no row wrote it
    current = _read(ctx, source="identity", version=outside)
    assert current.startswith(f"[identity version {outside}: the current text; no history row wrote it") and \
        current.endswith(IDENTITY), current
    _update_identity(ctx, IDENTITY + "\nOne more line.\n", outside, reason="growing")
    second = _sha(identity.read_bytes())
    listing = _read(ctx, source="identity")
    rows = [line for line in listing.splitlines() if line.startswith("- ")]
    assert len(rows) == 1 and rows[0].startswith(f"- {second} · ") and f" · replaced {outside} · " in rows[0], listing
    assert "reason: growing" in rows[0] and "· current" in rows[0]
    assert "Put one back: update_identity(mode='revert', version=<revision>, expected_revision=" in listing
    earlier = _read(ctx, source="identity", version=outside)  # now only the row's old side holds it
    assert earlier.endswith(IDENTITY) and "an earlier version, not the current text" in earlier, earlier[:300]


def test_the_doors_list_pages_by_its_fingerprint_and_refuses_a_changed_list(tmp_path, identity, monkeypatch):
    monkeypatch.setattr("ouroboros.tool_capabilities.tool_result_limit", lambda _name: 700)
    ctx = _ctx(tmp_path)
    for number in range(3):
        _update_identity(ctx, IDENTITY + f"\nVersion {number}.\n", _sha(identity.read_bytes()))
    first = _read(ctx, source="identity")
    fingerprint = re.search(r"fingerprint='([0-9a-f]{16})'", first)[1]
    after = int(re.search(r"next_start=(\d+)", first)[1])
    assert f"memory_read(source='identity', start={after}, fingerprint='{fingerprint}')" in first
    assert "position_mismatch" in _read(ctx, source="identity", start=after)  # a continuation names its list
    assert _read(ctx, source="identity", start=after, fingerprint=fingerprint).startswith("[identity: versions newest")
    _update_identity(ctx, IDENTITY + "\nLater.\n", _sha(identity.read_bytes()))
    assert "position_mismatch" in _read(ctx, source="identity", start=after, fingerprint=fingerprint)


@pytest.mark.parametrize(("arguments", "said"), [
    ({"source": "identity", "node_id": "rec-1"}, "drop node_id"),
    ({"source": "identity", "room_id": "1"}, "drop room_id"),
    ({"source": "identity", "revision": "r"}, "drop revision"),
    ({"version": "a" * 64}, "name its source"),
    ({"source": "identity", "version": "a" * 64, "fingerprint": "f"}, "fingerprint continues the list"),
    ({"source": "scratchpad"}, "a memory text is identity, world, workpad"),
    ({"source": "workpad"}, "this task has none"),
])
def test_the_door_refuses_conflicting_selectors(tmp_path, identity, arguments, said):
    reply = _read(_ctx(tmp_path), **arguments)
    assert reply.startswith("⚠️ TOOL_ARG_ERROR") and said in reply, reply


def test_a_workpad_reads_through_its_project_scope_and_its_versions_outlive_the_page(tmp_path, monkeypatch):
    import types

    from ouroboros.tools.project_journal import _workpad_write

    monkeypatch.setattr("ouroboros.config.DATA_DIR", tmp_path / "data")
    own = types.SimpleNamespace(project_id="atlas", task_id="t-1", drive_root=tmp_path, task_metadata={})
    _workpad_write(own, "# Plan\n\n- one\n")
    page = tmp_path / "data" / "projects" / "atlas" / "workpad.md"
    first = _sha(page.read_bytes())
    _workpad_write(own, mode="edit", old_str="- one", content="- one, done", expected_revision=first)
    listing = _read(own, source="workpad")
    assert "History of workpad:atlas (workpad)" in listing and f"replaced {first}" in listing, listing
    assert "workpad_write(project_id='atlas', mode='revert', version=<revision>" in listing
    child = types.SimpleNamespace(project_id="other", task_id="kid", drive_root=tmp_path,
                                  task_metadata={"delegation_role": "subagent"})
    from ouroboros.tools.tool_result import _TOOL_RESULT_ATTR

    setattr(child, _TOOL_RESULT_ATTR, None)
    withheld = _read(child, source="workpad:atlas")
    assert withheld.startswith("⚠️ TOOL_FORBIDDEN") and "only its current project" in withheld, withheld
    assert getattr(child, _TOOL_RESULT_ATTR).code == "ACCESS_BLOCKED"  # the resolver's own code, not its text
    malformed = _read(child, source="workpad:a/b")
    assert malformed.startswith("⚠️ TOOL_ARG_ERROR") and getattr(child, _TOOL_RESULT_ATTR).code == "TOOL_ARG_ERROR"
    root = types.SimpleNamespace(project_id="", task_id="main", drive_root=tmp_path, task_metadata={"root_task_id": "main"})
    page.unlink()  # a historical version reads even when the live page is gone
    gone = _read(root, source="workpad:atlas", version=first)
    assert gone.endswith("# Plan\n\n- one\n") and "the text does not exist now" in gone, gone


def test_a_blank_source_or_version_takes_the_chronicle_path_with_a_note(tmp_path, identity, monkeypatch):
    """K5: a blank ``source`` (or a blank ``version`` without one) names nothing: the chronicle read runs as on the base,
    the answer naming the ignored argument; the door stays shut."""
    import ouroboros.memory_texts as texts

    monkeypatch.setattr(texts, "read_versions", lambda *a, **k: pytest.fail("the door opened for a blank source"))
    ctx = _ctx(tmp_path)
    plain = _read(ctx, owner_words=True)  # a chronicle-path read that needs no activated chronicle
    for arguments, said in (({"source": ""}, "source='' ignored"), ({"source": "  "}, "source='  ' ignored"),
                            ({"version": ""}, "version='' ignored"), ({"source": "", "version": " "}, "version=' ' ignored")):
        reply = _read(ctx, owner_words=True, **arguments)
        assert reply.startswith(plain) and said in reply[len(plain):], (arguments, reply)


def test_a_real_selector_beside_source_is_refused_and_a_default_is_not_a_selector(tmp_path, identity):
    ctx = _ctx(tmp_path)
    for arguments in ({"room_id": 0}, {"room_id": "0"}, {"after_seq": 3}, {"rows": True}, {"story": True}):
        reply = _read(ctx, source="identity", **arguments)
        assert reply.startswith("⚠️ TOOL_ARG_ERROR") and f"drop {next(iter(arguments))}" in reply, (arguments, reply)
    for arguments in ({"room_id": None}, {"room_id": ""}, {"after_seq": 0}, {"rows": False}, {"story": False},
                      {"owner_words": False}, {"node_id": ""}, {"revision": ""}):
        assert _read(ctx, source="identity", **arguments).startswith("[identity: versions newest first"), arguments


def test_a_stale_revision_held_only_on_a_rows_old_side_gets_the_changes_since(tmp_path, identity):
    """K7: the identity was written before history; its first row keeps it on its old side. A stale writer holding that
    original gets every change since in whole lines (verified), the replacing write among them, and is told no row
    wrote the version it holds; a version no row holds still gets the whole text (the test above)."""
    original = _sha(identity.read_bytes())
    _update_identity(_ctx(tmp_path, "other"), IDENTITY.replace("Keep what I learn.", "Keep what I learn, dated."),
                     original)
    stale = _update_identity(_ctx(tmp_path), "Mine.", original)
    assert "revision_conflict" in stale and "1 block(s)" in stale, stale
    assert "earlier version of this identity; no history row wrote it (an outside edit" in stale
    assert "Written since that version, oldest first:\n- " in stale and "focus root task_id=other" in stale
    assert "\n- Keep what I learn.\n+ Keep what I learn, dated.\n" in stale and "Mine." not in identity.read_text("utf-8")


def test_each_listed_write_prints_its_own_undo_with_the_project_and_running_it_restores_that_version(
        tmp_path, monkeypatch):
    """K8 through the door: each row of a workpad's list prints the workpad_write call (project_id included) that puts
    back the version ITS write replaced over the current revision, never the line below's; the printed call of a write
    over an outside edit restores the outside bytes exactly; the creating row has no earlier version; while the page
    does not exist no revert is offered (a revert needs the page)."""
    import types

    from ouroboros.tools.project_journal import _workpad_write
    from tests._knowledge_history_shared import printed_call

    monkeypatch.setattr("ouroboros.config.DATA_DIR", tmp_path / "data")
    own = types.SimpleNamespace(project_id="atlas", task_id="t-1", drive_root=tmp_path, task_metadata={})
    _workpad_write(own, "# Plan\n\n- one\n")
    page = tmp_path / "data" / "projects" / "atlas" / "workpad.md"
    first = _sha(page.read_bytes())
    page.write_bytes(b"# Plan\r\n\r\n- outside\r\n")  # no row writes this version
    outside = _sha(page.read_bytes())
    _workpad_write(own, "# Plan\n\n- two\n", expected_revision=outside)
    current = _sha(page.read_bytes())
    rows = [line for line in _read(own, source="workpad").splitlines() if line.startswith("- ")]
    calls = [re.search(r" · undo: (workpad_write\(.*?\)) · ", row) for row in rows]
    assert printed_call(calls[0][1]) == ("workpad_write", {"project_id": "atlas", "mode": "revert", "version": outside,
                                                            "expected_revision": current}), rows[0]
    assert first not in rows[0] and calls[1] is None and " · created it: no earlier version to put back · " in rows[1]
    tool, arguments = printed_call(calls[0][1])
    undone = _workpad_write(own, **arguments)
    assert undone.startswith("OK: workpad updated (revert") and page.read_bytes() == b"# Plan\r\n\r\n- outside\r\n", undone
    page.unlink()
    gone = _read(own, source="workpad")
    assert "mode='revert'" not in gone and " · undo: the workpad does not exist now, so read that version" in gone, gone
    assert "Put one back: the workpad does not exist now, so mode=revert cannot run" in gone


def test_a_legacy_row_lists_the_sizes_it_recorded_and_a_new_row_its_delta(tmp_path, identity):
    """K9: an identity row written before rows carried a delta (the whole-file form ``publish_memory_file`` still
    writes, the identity writer's before this change) lists the sizes it recorded, old_len → new_len, and no removed or
    added counts nobody recorded; a row this writer wrote lists its delta, not its lengths a second time."""
    from ouroboros.knowledge import focus_signature
    from ouroboros.memory_publication import publish_memory_file

    ctx = _ctx(tmp_path)
    legacy = publish_memory_file(identity, _sha(identity.read_bytes()), lambda old: old + "A legacy line.\n",
                                 journal=identity.parent / "identity_journal.jsonl", author=focus_signature(ctx),
                                 source_type="task")
    assert legacy.ok and "delta" not in _rows(tmp_path)[-1] and _rows(tmp_path)[-1]["new_len"] == len(IDENTITY) + 15
    _update_identity(ctx, mode="edit", old_str="A legacy line.", content="A newer line.", expected_revision=legacy.revision)
    rows = [line for line in _read(ctx, source="identity").splitlines() if line.startswith("- ")]
    assert f" · {len(IDENTITY)} → {len(IDENTITY) + 15} chars" in rows[1], rows[1]
    assert "removed" not in rows[1] and "added" not in rows[1], rows[1]  # nothing invented beside what it recorded
    assert re.search(r" · \d+ chars removed, \d+ added · ", rows[0]) and "→" not in rows[0], rows[0]


def _model_call(registry, tool, arguments):
    """One tool call as a model sends it: JSON arguments through the loop's own dispatch."""
    from ouroboros.loop_tool_execution import _execute_single_tool

    call = {"id": "mode-call", "function": {"name": tool, "arguments": json.dumps(arguments)}}
    return _execute_single_tool(registry, call, registry._ctx.drive_root / "logs", registry._ctx.task_id)


def test_a_null_blank_or_unknown_mode_is_refused_and_only_an_omitted_one_overwrites(tmp_path, monkeypatch):
    """K13: a filled-in null or blank mode may have meant an edit, so it is refused, never taken as overwrite; the
    refusal names mode as received and its three values, and nothing is read or written. An OMITTED mode keeps the
    writers' default, overwrite, as their older callers send content alone. Through the loop's dispatch, all three."""
    from ouroboros import world_profiler as wprof

    monkeypatch.setattr("ouroboros.config.DATA_DIR", tmp_path / "data")
    registry = ToolRegistry(repo_dir=tmp_path / "repo", drive_root=tmp_path)
    registry.set_context(ToolContext(repo_dir=tmp_path / "repo", drive_root=tmp_path, task_id="modes", project_id="atlas"))
    block = f"{wprof.MEASURED_START}\n## System (measured)\n{wprof.MEASURED_END}"
    files = {"update_identity": (tmp_path / "memory" / "identity.md", IDENTITY),
             "update_world": (tmp_path / "memory" / "WORLD.md", f"{wprof.TITLE}\n\n{block}\n\n- a note\n"),
             "workpad_write": (tmp_path / "data" / "projects" / "atlas" / "workpad.md", "# Plan\n\n- one\n")}
    for path, text in files.values():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(text.encode("utf-8"))
    received = {"null": (None, "mode is null"), "empty": ("", "mode='' is not an operation"),
                "space": (" ", "mode=' ' is not an operation"), "capitalized": ("Edit", "mode='Edit' is not an operation"),
                "unknown": ("append", "mode='append' is not an operation"), "number": (1, "mode=1 is not text")}
    journals = [path.with_name(name) for (path, _text), name in zip(files.values(), (
        "identity_journal.jsonl", "world_journal.jsonl", "workpad_history.jsonl"))]
    before = {path: path.read_bytes() for path, _text in files.values()}
    for tool, (path, text) in files.items():
        call = {"content": text.replace("one", "two").replace("a note", "a newer note").replace("Keep", "Hold"),
                "expected_revision": _sha(path.read_bytes())}
        for name, (mode, said) in received.items():
            executed = _model_call(registry, tool, {**call, "mode": mode})
            result, reply = executed["tool_result"], executed["result"]
            assert (result.status, result.code) == ("error", "TOOL_ARG_ERROR"), (tool, name, reply)
            assert result.meta.get("operation_outcome") == "completed_no_effect", (tool, name, result.meta)
            assert reply.startswith(f"⚠️ TOOL_ARG_ERROR: {said}. Pass mode as exactly one of overwrite (content "
                                    "replaces the whole ") and "edit (old_str with content, or edits," in reply, reply
            assert "or revert (puts back the earlier version named by version). Nothing was written." in reply, reply
    assert {path: path.read_bytes() for path, _text in files.values()} == before
    assert not any(journal.exists() for journal in journals)  # nothing was written, not even a row
    nouns = {"update_identity": "identity", "update_world": "world map", "workpad_write": "workpad"}
    for tool, (path, text) in files.items():  # omitted: today's default, the whole text replaced
        new = text.replace("one", "two").replace("a note", "a newer note").replace("Keep", "Hold")
        executed = _model_call(registry, tool, {"content": new, "expected_revision": _sha(path.read_bytes())})
        assert executed["result"].startswith(f"OK: {nouns[tool]} updated (overwrite"), executed["result"]
        assert path.read_bytes() == new.encode("utf-8"), tool
    assert all(journal.exists() for journal in journals)


def test_an_empty_expected_revision_on_an_existing_page_says_it_creates_only(tmp_path, monkeypatch):
    """K13: expected_revision="" is create-only. On an existing page the conflict says so (nothing came since the
    caller read; it read nothing) and the page stays; a stale non-empty revision keeps its own wording."""
    import types

    from ouroboros.tools.project_journal import _workpad_write

    monkeypatch.setattr("ouroboros.config.DATA_DIR", tmp_path / "data")
    own = types.SimpleNamespace(project_id="atlas", task_id="t-1", drive_root=tmp_path, task_metadata={})
    _workpad_write(own, "# Plan\n")
    page = tmp_path / "data" / "projects" / "atlas" / "workpad.md"
    empty = _workpad_write(own, "# Mine\n", expected_revision="")
    assert "revision_conflict" in empty and "An empty expected_revision creates a missing text only; this one exists: " \
        f"pass expected_revision='{_sha(page.read_bytes())}'." in empty and page.read_bytes() == b"# Plan\n", empty
    assert "Reconcile your change" not in empty
    stale = _workpad_write(own, "# Mine\n", expected_revision="0" * 64)
    assert "Reconcile your change with what came since" in stale and "creates a missing text only" not in stale
