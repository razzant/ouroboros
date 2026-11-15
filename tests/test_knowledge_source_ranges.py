"""Partial note reads retain one complete source identity and honest coverage."""

import hashlib
import json

from ouroboros.tools.knowledge import _knowledge_read
from ouroboros.tools.registry import ToolContext
from ouroboros.tools.tool_result import _install_tool_result_sidecar, _published_tool_result, _restore_tool_result_sidecar


def test_partial_reads_reconstruct_exact_note_with_one_revision(tmp_path):
    text = "---\ntype: note\nsummary: Authored understanding.\n---\n\n" + "αβγ\r\n" * 40
    path = tmp_path / "memory/knowledge/person.md"
    path.parent.mkdir(parents=True)
    path.write_bytes(text.encode())
    ctx = ToolContext(repo_dir=tmp_path, drive_root=tmp_path)
    parts = []
    for start, end in [(0, 80), (80, len(text))]:
        sentinel = object()
        token = _install_tool_result_sidecar(ctx, sentinel)
        try:
            result = _knowledge_read(ctx, "person", start_char=start, end_char=end)
            meta = _published_tool_result(ctx, sentinel).meta
        finally:
            _restore_tool_result_sidecar(token)
        assert not meta["knowledge_source_complete"]
        source = meta["knowledge_source"]
        assert source["revision"] == hashlib.sha256(text.encode()).hexdigest()
        assert source["complete_chars"] == len(text)
        assert (source["start_char"], source["end_char"]) == (start, end)
        body = result[meta["knowledge_body_start"]:]
        assert body == text[start:end]
        parts.append(body)
        assert json.loads(result.splitlines()[0].removeprefix("[Knowledge source] ")) == source
    assert "".join(parts) == text


def test_invalid_range_cannot_claim_full_source(tmp_path):
    path = tmp_path / "memory/knowledge/note.md"
    path.parent.mkdir(parents=True)
    path.write_text("Source")
    ctx = ToolContext(repo_dir=tmp_path, drive_root=tmp_path)
    refused = _knowledge_read(ctx, "note", start_char=10, end_char=20)
    assert "range" in refused and "complete_chars=6" in refused and "start_char=10" in refused


def _read(ctx, **args):
    sentinel = object()
    token = _install_tool_result_sidecar(ctx, sentinel)
    try:
        text = _knowledge_read(ctx, "note", **args)
        return text, _published_tool_result(ctx, sentinel)
    finally:
        _restore_tool_result_sidecar(token)


def test_a_bound_that_asks_for_nothing_reads_instead_of_refusing(tmp_path):
    """Models fill both optional bounds: an end past the note, 0..0, or one bound alone
    each used to cost a refused round (and a learned two-call size probe). The range
    RETURNED is always the range delivered, so coverage arithmetic stays exact."""
    path = tmp_path / "memory/knowledge/note.md"
    path.parent.mkdir(parents=True)
    path.write_text("Source text", encoding="utf-8")
    ctx = ToolContext(repo_dir=tmp_path, drive_root=tmp_path)
    for args, expected, note in [
        ({"start_char": 0, "end_char": 20000}, (0, 11), "end_char=20000 ignored: the note ends at 11"),
        ({"start_char": 0, "end_char": 0}, (0, 11), "end_char=0 ignored: a 0..0 range selects nothing"),
        ({"start_char": 7}, (7, 11), ""),
        ({"end_char": 6}, (0, 6), ""),
    ]:
        text, result = _read(ctx, **args)
        source = result.meta["knowledge_source"]
        assert result.status == "ok" and (source["start_char"], source["end_char"]) == expected
        assert text[result.meta["knowledge_body_start"]:] == "Source text"[expected[0]:expected[1]]
        assert result.meta["knowledge_body_chars"] == expected[1] - expected[0]
        assert result.meta["knowledge_source_complete"] == (expected == (0, 11))
        assert json.loads(text.splitlines()[0].removeprefix("[Knowledge source] ")) == source
        assert (f"[Range note] {note}" in text) if note else ("[Range note]" not in text)
    for args in ({"start_char": -1, "end_char": 4}, {"start_char": 5, "end_char": 2}, {"start_char": "0", "end_char": 4}):
        text, result = _read(ctx, **args)
        assert result.status == "error" and result.code == "TOOL_ARG_ERROR" and "complete_chars=11" in text


def test_blank_revision_creates_only_a_missing_note(tmp_path):
    """The words and the code agree: omit expected_revision for a bare append; an empty one only creates."""
    from ouroboros.tools.knowledge import _knowledge_write, get_tools

    schema = next(entry for entry in get_tools() if entry.name == "knowledge_write").schema
    words = schema["parameters"]["properties"]["expected_revision"]["description"]
    assert "Omit it for a bare append or to create a note; pass empty only to create a missing note" in words
    assert "creating a note and a bare append need none" in schema["description"]
    ctx = ToolContext(repo_dir=tmp_path, drive_root=tmp_path)
    assert "saved" in _knowledge_write(ctx, "new", "First body", mode="overwrite", expected_revision="")
    path = tmp_path / "memory/knowledge/new.md"
    original = path.read_bytes()
    for mode in ("overwrite", "append"):
        assert "revision_conflict" in _knowledge_write(ctx, "new", "Unsafe change", mode=mode, expected_revision="")
        assert path.read_bytes() == original
    assert "revision_required" in _knowledge_write(ctx, "new", "Unsafe change", mode="overwrite")
    revision = hashlib.sha256(original).hexdigest()
    assert "saved" in _knowledge_write(ctx, "new", "Fresh body", mode="overwrite", expected_revision=revision)
    assert "Fresh body" in path.read_text()
    assert "revision_conflict" in _knowledge_write(ctx, "new", "Stale body", mode="overwrite", expected_revision=revision)
    fresh = path.read_bytes()
    assert "revision_conflict" in _knowledge_write(ctx, "new", "A bare append.", mode="append", expected_revision="")
    assert path.read_bytes() == fresh  # an empty expectation on an existing note writes nothing
    assert "saved" in _knowledge_write(ctx, "new", "A bare append.", mode="append")  # omitted: a bare append
    assert path.read_bytes() == fresh + b"\nA bare append."


def test_boolean_range_bounds_are_not_integer_offsets(tmp_path):
    path = tmp_path / "memory/knowledge/note.md"
    path.parent.mkdir(parents=True)
    path.write_text("Source")
    ctx = ToolContext(repo_dir=tmp_path, drive_root=tmp_path)
    for args in ({"start_char": False}, {"end_char": True}):
        text, result = _read(ctx, **args)
        assert result.code == "TOOL_ARG_ERROR" and "complete_chars=6" in text
