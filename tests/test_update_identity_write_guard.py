"""Tests for the post-write verify guard in `_update_identity`.

Background (ibl-local-f3628ac3c9a0): twice in 2 days the bytes actually written
to identity.md diverged from the input past ~byte 4648 (single continuous
corruption run, not truncation) and the tool reported OK. The handler code
itself looked clean, so the corruption is upstream (arg transport). These tests
verify the guard: re-read + sha256 compare after write_text, and a loud
non-OK return with `verify_failed: true` in the journal when the bytes do
not round-trip.
"""

from __future__ import annotations

import json
import pathlib
import unittest
from hashlib import sha256
from unittest import mock


from ouroboros.tools.control import _update_identity
from ouroboros.tools.registry import ToolContext


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _make_ctx(tmp_path: pathlib.Path, *, project_id: str = "") -> ToolContext:
    drive_root = tmp_path / "drive"
    drive_root.mkdir(parents=True, exist_ok=True)
    return ToolContext(
        repo_dir=tmp_path,
        drive_root=drive_root,
        task_id="guard-test",
        project_id=project_id,
    )


def _seed_identity(tmp_path: pathlib.Path, content: str) -> None:
    mem_dir = tmp_path / "drive" / "memory"
    mem_dir.mkdir(parents=True, exist_ok=True)
    (mem_dir / "identity.md").write_text(content, encoding="utf-8")


def _read_journal(tmp_path: pathlib.Path) -> list[dict]:
    journal = tmp_path / "drive" / "memory" / "identity_journal.jsonl"
    if not journal.exists():
        return []
    return [
        json.loads(line)
        for line in journal.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _corrupt_first_identity_write(
    tmp_path: pathlib.Path, *, replace: str | None = None, append: str = ""
) -> unittest.mock._patch:
    """Patch pathlib.Path.write_text so the FIRST call whose path ends with
    'identity.md' writes corrupted bytes; subsequent writes (e.g. the
    best-effort restore) fall through to the real implementation. The
    corruption is applied at the path resolved against tmp_path so the
    patched test drive is targeted, not the live system drive.
    """
    real_write_text = pathlib.Path.write_text
    state = {"used": False}

    def patched(self, data, *args, **kwargs):
        path_str = str(self)
        ends_with_id = path_str.endswith("identity.md")
        is_test_drive = str(tmp_path) in path_str
        if ends_with_id and is_test_drive and not state["used"]:
            state["used"] = True
            if replace is not None:
                return real_write_text(self, replace, *args, **kwargs)
            if append:
                return real_write_text(self, data + append, *args, **kwargs)
            return real_write_text(self, "X" * len(data), *args, **kwargs)
        return real_write_text(self, data, *args, **kwargs)

    return mock.patch.object(pathlib.Path, "write_text", patched)


# ---------------------------------------------------------------------------
# (a) normal update — guard does not change existing happy path
# ---------------------------------------------------------------------------


def test_normal_update_writes_and_returns_ok(tmp_path):
    _seed_identity(tmp_path, "old identity content " * 50)
    ctx = _make_ctx(tmp_path)
    new_content = "completely new identity " * 50

    result = _update_identity(ctx, new_content)

    assert result.startswith("OK: identity updated"), result
    assert "IDENTITY_WRITE_CORRUPTED" not in result
    assert (tmp_path / "drive" / "memory" / "identity.md").read_text(
        encoding="utf-8"
    ) == new_content

    entries = _read_journal(tmp_path)
    assert len(entries) == 1, entries
    expected_sha = sha256(new_content.encode("utf-8")).hexdigest()
    assert entries[0]["new_sha256"] == expected_sha
    assert entries[0].get("verify_failed") is not True
    assert "expected_sha256" not in entries[0]
    assert "written_sha256" not in entries[0]


# ---------------------------------------------------------------------------
# (b) corrupted write — guard catches, returns loud warning, restores file
# ---------------------------------------------------------------------------


def test_corrupted_write_returns_warning_and_restores(tmp_path):
    old_content = "old identity content " * 50
    _seed_identity(tmp_path, old_content)
    ctx = _make_ctx(tmp_path)
    new_content = "completely new identity " * 100

    with _corrupt_first_identity_write(tmp_path):
        result = _update_identity(ctx, new_content)

    # Loud non-OK return, no OK line at all.
    assert "IDENTITY_WRITE_CORRUPTED" in result, result
    assert "OK: identity updated" not in result
    # identity.md was restored to the prior content.
    assert (tmp_path / "drive" / "memory" / "identity.md").read_text(
        encoding="utf-8"
    ) == old_content
    # Journal entry carries verify_failed + both shas.
    entries = _read_journal(tmp_path)
    assert len(entries) == 1, entries
    entry = entries[0]
    assert entry.get("verify_failed") is True
    assert "expected_sha256" in entry and "written_sha256" in entry
    assert (
        entry["expected_sha256"]
        == sha256(new_content.encode("utf-8")).hexdigest()
    )
    assert entry["written_sha256"] != entry["expected_sha256"]
    # Full content stored (no [:N] preview slices) — identity_journal is
    # the only place the agent's intended new identity survives when the
    # file write was corrupt. Bare preview slices would silently lose data
    # on the failure path (BIBLE P1 cognitive-artifact integrity).
    assert entry["expected_content"] == new_content
    assert "corrupt_written" in entry
    # Byte-length fields are present for forensic comparison.
    assert entry["content_byte_length"] == len(new_content.encode("utf-8"))
    # Failure entry must NOT look like a clean success entry.
    assert "new_sha256" not in entry
    assert "new_content" not in entry
    assert "expected_preview" not in entry
    assert "corrupt_written_preview" not in entry
    # The two shas the task requires, plus first-divergence offset.
    assert "first_divergence_offset" in entry


# ---------------------------------------------------------------------------
# (c) first-divergence offset reported correctly
# ---------------------------------------------------------------------------


def test_first_divergence_offset_reported_correctly(tmp_path):
    _seed_identity(tmp_path, "old " * 100)
    ctx = _make_ctx(tmp_path)
    # New content whose bytes match for the first 5 chars, then diverge.
    new_content = "abcde" + ("Z" * 200) + "tail"

    with _corrupt_first_identity_write(
        tmp_path, replace=("abcde" + ("X" * 200) + "tail")
    ):
        result = _update_identity(ctx, new_content)

    assert "IDENTITY_WRITE_CORRUPTED" in result
    assert "first divergence at byte 5" in result, result
    entries = _read_journal(tmp_path)
    assert entries[0]["first_divergence_offset"] == 5


def test_first_divergence_offset_at_zero_when_first_byte_differs(tmp_path):
    _seed_identity(tmp_path, "old " * 100)
    ctx = _make_ctx(tmp_path)
    new_content = "abc" + ("Q" * 100) + "tail"

    with _corrupt_first_identity_write(
        tmp_path, replace=("XYZ" + ("Q" * 100) + "tail")
    ):
        result = _update_identity(ctx, new_content)

    assert "first divergence at byte 0" in result, result
    entries = _read_journal(tmp_path)
    assert entries[0]["first_divergence_offset"] == 0


def test_first_divergence_offset_uses_byte_offset_for_unicode(tmp_path):
    """The IBL describes 'past ~byte 4648' — a byte offset. For non-ASCII
    content the char index diverges from the byte offset, so the guard
    MUST report the byte offset, not the char index.

    Setup: 'abc' + 'ё' (U+0451 = 0xD1 0x91) + 'tail' + ' ' * N (≥ 50 chars).
    The corruption rewrites byte 4 (0x91) to 0x84, turning ё into ф
    (U+0444 = 0xD1 0x84) — both are valid 2-byte UTF-8 sequences and the
    byte length is preserved.

    Expected: first divergent byte is 4 (the continuation byte of ё vs
    the continuation byte of ф). Char index would be 3 — byte and char
    indices diverge for multi-byte characters.
    """
    _seed_identity(tmp_path, "old " * 100)
    ctx = _make_ctx(tmp_path)
    # 30 'a's + 'abcёtail' (9 bytes, 8 chars) + 30 'b's.
    # ё is at char 33 and at byte 33; its continuation byte is at byte 34.
    # Length after strip(): 30+8+30 = 68, well over the 50-char floor.
    new_content = ("a" * 30) + "abcёtail" + ("b" * 30)
    content_bytes = new_content.encode("utf-8")
    assert len(new_content) == 68
    # ё is at bytes 33-34 in the encoded form.
    assert content_bytes[33:35] == b"\xd1\x91", content_bytes

    real_write_text = pathlib.Path.write_text
    state = {"used": False}

    def patch_byte_34_to_84(self, data, *args, **kwargs):
        path_str = str(self)
        if (
            path_str.endswith("identity.md")
            and str(tmp_path) in path_str
            and not state["used"]
        ):
            state["used"] = True
            # Mutate byte 34 (the continuation byte of ё) from 0x91 to
            # 0x84 — turns ё into ф. Valid 2-byte UTF-8, same byte
            # length, so the round-trip re-read produces the same byte
            # count and the divergence is observable.
            data_bytes = data.encode("utf-8")
            mutated = data_bytes[:34] + b"\x84" + data_bytes[35:]
            return real_write_text(self, mutated.decode("utf-8"), *args, **kwargs)
        return real_write_text(self, data, *args, **kwargs)

    with mock.patch.object(pathlib.Path, "write_text", patch_byte_34_to_84):
        result = _update_identity(ctx, new_content)

    assert "IDENTITY_WRITE_CORRUPTED" in result
    # Byte 34 is the first divergent byte. Char index 33 ('ё') would
    # be wrong — char comparison is not the same as byte comparison
    # for non-ASCII content.
    assert "first divergence at byte 34" in result, result
    entries = _read_journal(tmp_path)
    assert entries[0]["first_divergence_offset"] == 34
    # 30 + 3 + 2 + 4 + 30 = 69 bytes
    assert entries[0]["content_byte_length"] == 69

