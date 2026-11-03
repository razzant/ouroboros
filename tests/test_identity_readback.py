"""Forked identity consumers follow the reader emitted by the real registry."""
from __future__ import annotations

import ast
import hashlib
import re

import pytest

from ouroboros.tools.core_file_tools import delivered_source_prefix
from ouroboros.tools.registry import ToolContext, ToolRegistry


@pytest.fixture
def identity(tmp_path, monkeypatch):
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "cyber_pro")
    root, fork, repo = (tmp_path / name for name in ("data", "fork", "repo"))
    for base in (root, fork, repo):
        (base / "memory").mkdir(parents=True)
    monkeypatch.setattr("ouroboros.config.DATA_DIR", root)
    canonical = root / "memory/identity.md"
    local = fork / "memory/identity.md"
    canonical.write_bytes(b"Canonical identity with independent current commitments.\r\n" * 3)
    local.write_bytes(b"Execution-drive identity from an earlier snapshot.\r\n" * 3)

    def registry():
        ctx = ToolContext(repo_dir=repo, drive_root=fork, task_id="identity-reader",
                          budget_drive_root=str(root), task_metadata={"budget_drive_root": str(root)})
        reg = ToolRegistry(repo_dir=repo, drive_root=fork)
        reg.set_context(ctx)
        return ctx, reg

    return canonical, local, registry


def _recipe(text):
    line = next((line for line in text.splitlines() if line.startswith("read_file(")), "")
    assert line, text
    expr = ast.parse(line.split(" (reader address,", 1)[0], mode="eval").body
    return {kw.arg: ast.literal_eval(kw.value) for kw in expr.keywords}


def _revision(text):
    return re.search(r"(?:^|\s)revision=([0-9a-f]{64})", text).group(1)


def test_fork_reader_recipe_write_readback_second_write(identity):
    canonical, local, registry = identity
    original = local.read_bytes()
    ctx, reg = registry()
    first = reg.execute("read_file", {"root": "runtime_data", "path": "memory/identity.md"})
    assert "identity_source=execution_drive" in first, first
    assert _revision(first) == hashlib.sha256(original).hexdigest()
    current = reg.execute("read_file", _recipe(first))
    assert "identity_source=canonical" in current, current
    assert delivered_source_prefix(ctx.last_read_view, current, len(current)) == canonical.read_text()
    updated = "I evolve the commitments I just read, keeping a continuous identity. " * 3
    saved = reg.execute("update_identity", {"content": updated, "expected_revision": _revision(current)})
    assert saved.startswith("OK:"), saved
    # A restored actor has no previous read cache; only the returned recipe is needed.
    restored, reg = registry()
    readback = reg.execute("read_file", _recipe(saved))
    assert updated == delivered_source_prefix(restored.last_read_view, readback, len(readback))
    second = reg.execute("update_identity", {"content": updated + "Next commitment.",
                                            "expected_revision": _revision(readback)})
    assert second.startswith("OK:"), second
    assert canonical.read_text() == updated + "Next commitment."
    assert local.read_bytes() == original


def test_conflict_emits_reader_without_overwriting_concurrent_identity(identity):
    canonical, local, registry = identity
    _, reg = registry()
    first = reg.execute("read_file", {"root": "runtime_data", "path": str(canonical)})
    concurrent = "Another focus has revised this shared identity with a new commitment. " * 3
    canonical.write_text(concurrent)
    refused = reg.execute("update_identity", {"content": "My proposed identity. " * 5,
                                              "expected_revision": _revision(first)})
    assert "revision_conflict" in refused and canonical.read_text() == concurrent
    fresh = reg.execute("read_file", _recipe(refused))
    assert concurrent in fresh and _revision(fresh) == hashlib.sha256(canonical.read_bytes()).hexdigest()
    assert "Execution-drive" in local.read_text()


def test_empty_missing_denied_and_unreadable_identity_keep_their_meaning(identity, monkeypatch):
    canonical, _, registry = identity
    ctx, reg = registry()
    canonical.write_bytes(b"")
    empty = reg.execute("read_file", {"root": "runtime_data", "path": str(canonical)})
    assert _revision(empty) == hashlib.sha256(b"").hexdigest()
    assert reg.execute("update_identity", {"content": "An initialized identity. " * 4,
                                           "expected_revision": _revision(empty)}).startswith("OK:")
    canonical.unlink()
    missing = reg.execute("read_file", {"root": "runtime_data", "path": str(canonical)})
    assert "DATA_NOT_YET_CREATED" in missing and "revision=" not in missing and ctx.last_read_view is None
    canonical.write_text("Private identity. " * 5)
    ctx.task_contract = {"resource_policy": {"protected_artifacts": [
        {"id": "identity", "paths": [str(canonical)], "allow": ["execute"]}]}}
    denied = reg.execute("read_file", {"root": "runtime_data", "path": str(canonical)})
    assert "RESOURCE_POLICY_BLOCKED" in denied and "Private identity" not in denied
    ctx.task_contract = {}
    from ouroboros import store
    original_read = store.read_file_bytes

    def read(path):
        if path == canonical:
            raise PermissionError("fixture unreadable identity")
        return original_read(path)

    monkeypatch.setattr(store, "read_file_bytes", read)
    unreadable = reg.execute_result("read_file", {"root": "runtime_data", "path": str(canonical)})
    assert unreadable.status == "error" and "fixture unreadable identity" in unreadable.text
    assert "revision=" not in unreadable.text and ctx.last_read_view is None


def test_missing_fork_file_does_not_imply_missing_shared_identity(identity):
    canonical, local, registry = identity
    local.unlink()
    ctx, reg = registry()
    missing = reg.execute("read_file", {"root": "runtime_data", "path": "memory/identity.md"})
    assert "DATA_NOT_YET_CREATED" in missing and "revision=" not in missing
    assert "identity_source=execution_drive" in missing and ctx.last_read_view is None
    current = reg.execute("read_file", _recipe(missing))
    assert delivered_source_prefix(ctx.last_read_view, current, len(current)) == canonical.read_text()


def test_emitted_identity_reader_grants_no_permission(identity):
    canonical, _, registry = identity
    ctx, reg = registry()
    local = reg.execute("read_file", {"root": "runtime_data", "path": "memory/identity.md"})
    reader = _recipe(local)
    ctx.task_contract = {"resource_policy": {"protected_artifacts": [
        {"id": "identity", "paths": [str(canonical)], "allow": ["execute"]}]}}
    denied = reg.execute("read_file", reader)
    assert "RESOURCE_POLICY_BLOCKED" in denied and "Canonical identity" not in denied
    ctx.task_contract = {"disabled_tools": ["read_file"]}
    assert "RESOURCE_CONSTRAINT_BLOCKED" in reg.execute("read_file", reader)


def test_initial_identity_section_addresses_the_bytes_it_loaded(identity):
    from ouroboros.context import build_memory_sections
    from ouroboros.memory import Memory

    canonical, local, registry = identity
    ctx, reg = registry()
    section = build_memory_sections(Memory(drive_root=canonical.parent.parent), partition="stable", ctx=ctx)[0]
    assert str(canonical) in section and "Canonical identity" in section
    recipe = _recipe(section.replace("Source reader: ", ""))
    current = reg.execute("read_file", recipe)
    assert "identity_source=canonical" in current and "Execution-drive identity" not in current
    assert str(local) not in section
