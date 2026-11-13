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
    ({"mode": "append", "content": "x"}, "mode must be overwrite, edit, revert"),
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
