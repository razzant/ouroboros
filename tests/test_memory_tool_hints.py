"""Tests for the memory-artifact tool-error hints (2026-05-01).

Documented friction: every fresh task, the agent runs `repo_read("identity.md")`
or `repo_read("scratchpad.md")`, gets ENOENT, then tries `data_read` with the
full drive_root-prefixed path, gets path-doubled ENOENT, and burns several
rounds before finding the right call. These hints address the friction
without changing the constitutional core.

Two layers:
  1. ``repo_read("identity.md")`` returns a friendly NOT_FOUND that names
     the actual location AND tells the agent the content is already in
     the system prompt.
  2. ``data_read(".tmp-data-foo/data/memory/identity.md")`` strips the
     duplicate drive_root prefix so the call resolves correctly.
"""

from __future__ import annotations

import pathlib
from dataclasses import dataclass

import pytest


@dataclass
class _FakeCtx:
    """Minimal ToolContext stand-in for direct testing of _repo_read /
    _data_read without the full agent stack."""
    repo_dir: pathlib.Path
    drive_root: pathlib.Path

    def repo_path(self, rel: str) -> pathlib.Path:
        return self.repo_dir / rel

    def drive_path(self, rel: str) -> pathlib.Path:
        return self.drive_root / rel


# ---------------------------------------------------------------------------
# repo_read returns a friendly hint for memory artifacts at the wrong location
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", [
    "identity.md",
    "scratchpad.md",
    "dialogue_summary.md",
    "registry.md",
    "deep_review.md",
])
def test_repo_read_memory_artifact_returns_hint(tmp_path, name):
    from ouroboros.tools.core_file_tools import _repo_read

    ctx = _FakeCtx(
        repo_dir=tmp_path / "repo",
        drive_root=tmp_path / "data",
    )
    (tmp_path / "repo").mkdir(parents=True, exist_ok=True)
    # File is NOT at repo root (correct — it lives elsewhere)
    result = _repo_read(ctx, name)
    assert "NOT_FOUND" in result
    assert "data_root/memory/" in result
    assert name in result
    assert "raw memory state" in result.lower()
    # Suggested replacement call
    assert f"read_file(root='runtime_data', path='memory/{name}')" in result


def test_repo_read_normal_file_unchanged(tmp_path):
    """Non-memory files at repo root behave normally."""
    from ouroboros.tools.core_file_tools import _repo_read

    ctx = _FakeCtx(
        repo_dir=tmp_path / "repo",
        drive_root=tmp_path / "data",
    )
    (tmp_path / "repo").mkdir(parents=True, exist_ok=True)
    (tmp_path / "repo" / "BIBLE.md").write_text("constitutional", encoding="utf-8")
    result = _repo_read(ctx, "BIBLE.md")
    assert "BIBLE.md" in result
    assert "constitutional" in result
    assert "NOT_FOUND" not in result


@pytest.mark.parametrize("name", [
    "identity.md",
    "scratchpad.md",
    "registry.md",
])
def test_repo_read_real_memory_named_file_at_repo_root_wins(tmp_path, name):
    """The friendly hint is only for missing files, not real repo files."""
    from ouroboros.tools.core_file_tools import _repo_read

    ctx = _FakeCtx(
        repo_dir=tmp_path / "repo",
        drive_root=tmp_path / "data",
    )
    (tmp_path / "repo").mkdir(parents=True, exist_ok=True)
    (tmp_path / "repo" / name).write_text("repo-local content", encoding="utf-8")

    result = _repo_read(ctx, name)
    assert "repo-local content" in result
    assert "NOT_FOUND" not in result
    assert "read_file(root='runtime_data'" not in result


@pytest.mark.parametrize("name", [
    "deep_review.md",
    "WORLD.md",
])
def test_repo_read_memory_hint_does_not_claim_artifact_is_loaded(tmp_path, name):
    """Hints for partial/not-always-loaded artifacts must not discourage raw reads."""
    from ouroboros.tools.core_file_tools import _repo_read

    ctx = _FakeCtx(
        repo_dir=tmp_path / "repo",
        drive_root=tmp_path / "data",
    )
    (tmp_path / "repo").mkdir(parents=True, exist_ok=True)

    result = _repo_read(ctx, name)
    assert "NOT_FOUND" in result
    assert "already injected" not in result.lower()
    assert "don't need to read" not in result.lower()
    assert f"read_file(root='runtime_data', path='memory/{name}')" in result


def test_repo_read_subdirectory_path_unchanged(tmp_path):
    """Memory artifact requested with a directory prefix is treated as a
    regular request — only bare-name lookups at repo root trigger the hint."""
    from ouroboros.tools.core_file_tools import _repo_read

    ctx = _FakeCtx(
        repo_dir=tmp_path / "repo",
        drive_root=tmp_path / "data",
    )
    (tmp_path / "repo" / "memory").mkdir(parents=True, exist_ok=True)
    (tmp_path / "repo" / "memory" / "identity.md").write_text(
        "should be readable", encoding="utf-8",
    )
    result = _repo_read(ctx, "memory/identity.md")
    assert "should be readable" in result
    assert "NOT_FOUND" not in result


# ---------------------------------------------------------------------------
# data_read strips duplicate drive_root prefix
# ---------------------------------------------------------------------------

def test_data_read_strips_tmp_data_prefix(tmp_path):
    """When the agent (or operator) passes the full ``.tmp-data-XXX/data/
    memory/identity.md`` path, drive_path() would double it. Strip the
    prefix and resolve correctly."""
    from ouroboros.tools.core_file_tools import _data_read

    drive = tmp_path / ".tmp-data-test" / "data"
    drive.mkdir(parents=True, exist_ok=True)
    (drive / "memory").mkdir(exist_ok=True)
    target = drive / "memory" / "identity.md"
    target.write_text("I am Ouroboros", encoding="utf-8")

    ctx = _FakeCtx(
        repo_dir=tmp_path / "repo",
        drive_root=drive,
    )
    # Agent passes the path with the duplicate .tmp-data prefix
    result = _data_read(ctx, ".tmp-data-test/data/memory/identity.md")
    assert "I am Ouroboros" in result


def test_data_read_strips_absolute_drive_prefix(tmp_path):
    """Same idea but the agent passes the full absolute path."""
    from ouroboros.tools.core_file_tools import _data_read

    drive = tmp_path / "data"
    drive.mkdir(parents=True, exist_ok=True)
    (drive / "memory").mkdir(exist_ok=True)
    (drive / "memory" / "scratchpad.md").write_text("scratch content", encoding="utf-8")

    ctx = _FakeCtx(
        repo_dir=tmp_path / "repo",
        drive_root=drive,
    )
    # Absolute path
    abs_path = str(drive / "memory" / "scratchpad.md")
    result = _data_read(ctx, abs_path)
    assert "scratch content" in result


def test_data_read_normal_relative_path_unchanged(tmp_path):
    """The well-formed call ``data_read('memory/identity.md')`` must still work."""
    from ouroboros.tools.core_file_tools import _data_read

    drive = tmp_path / "data"
    drive.mkdir(parents=True, exist_ok=True)
    (drive / "memory").mkdir(exist_ok=True)
    (drive / "memory" / "identity.md").write_text("identity", encoding="utf-8")

    ctx = _FakeCtx(
        repo_dir=tmp_path / "repo",
        drive_root=drive,
    )
    result = _data_read(ctx, "memory/identity.md")
    assert "identity" in result


def test_data_read_missing_memory_path_returns_sentinel(tmp_path):
    """Missing memory paths get the cold-start sentinel, not raw ENOENT."""
    from ouroboros.tools.core_file_tools import _data_read

    drive = tmp_path / "data"
    drive.mkdir(parents=True, exist_ok=True)

    ctx = _FakeCtx(
        repo_dir=tmp_path / "repo",
        drive_root=drive,
    )

    result = _data_read(ctx, "memory/knowledge/patterns.md")
    assert "DATA_NOT_YET_CREATED" in result
    assert "lazily on first write" in result


def test_data_read_missing_non_memory_path_uses_narrower_sentinel(tmp_path):
    """Non-memory paths should not overclaim lazy creation."""
    from ouroboros.tools.core_file_tools import _data_read

    drive = tmp_path / "data"
    drive.mkdir(parents=True, exist_ok=True)

    ctx = _FakeCtx(
        repo_dir=tmp_path / "repo",
        drive_root=drive,
    )

    result = _data_read(ctx, "logs/missing.jsonl")
    assert "DATA_NOT_YET_CREATED" in result
    assert "lazily on first write" not in result
    assert "not guaranteed" in result


# ---------------------------------------------------------------------------
# Section header annotation
# ---------------------------------------------------------------------------

def test_memory_section_headers_indicate_already_loaded():
    """Inspect the source: build_memory_sections must annotate identity
    and scratchpad so the agent doesn't re-read them."""
    import inspect
    import ouroboros.context as ctx_mod
    src = inspect.getsource(ctx_mod.build_memory_sections)
    assert "memory/scratchpad.md" in src
    assert "memory/identity.md" in src
    assert "already loaded" in src
    assert "do not re-read" in src
