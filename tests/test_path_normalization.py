"""T2 (v6.35.0): workspace-relative path normalization for structural/read tools.

An agent working in an external workspace naturally writes absolute paths
(``/app/foo``) or a redundant root-basename prefix (``app/foo``). Before this
change ``search_code``/``list_files``/``query_code`` rejected those
(``SEARCH_ERROR: path not found: active_workspace:app`` / ``path escapes root``).
``normalize_root_relative`` maps such paths to the root-relative form WITHOUT
widening access: genuine escapes (outside-root, ``..``-traversal, symlink) are
still rejected by the unchanged ``safe_relpath`` + ``relative_to`` confinement.
"""

from __future__ import annotations

from ouroboros.tool_access import normalize_root_relative
from ouroboros.tools.registry import ToolContext, ToolRegistry


def test_normalize_root_relative_table(tmp_path):
    root = tmp_path / "app"
    (root / "foo").mkdir(parents=True)
    (root / "foo" / "bar.txt").write_text("x", encoding="utf-8")
    base = root.name  # "app"

    assert normalize_root_relative(root, str(root / "foo")) == "foo"  # abs inside -> rel
    assert normalize_root_relative(root, str(root)) == "."           # abs == root -> .
    assert normalize_root_relative(root, f"{base}/foo") == "foo"       # redundant prefix -> stripped
    assert normalize_root_relative(root, "foo/bar.txt") == "foo/bar.txt"  # relative unchanged
    assert normalize_root_relative(root, "/etc/passwd") == "/etc/passwd"  # abs outside -> unchanged
    assert normalize_root_relative(root, "foo") == "foo"             # plain relative unchanged
    assert normalize_root_relative(root, ".") == "."
    # redundant prefix for a NEW (not-yet-existing) target still normalizes, so
    # write_file/edit_text create-paths work, since there's no real <base>/ subdir.
    assert normalize_root_relative(root, f"{base}/new.py") == "new.py"

    # A REAL same-named subdir makes '<base>/x' genuinely ambiguous -> kept.
    (root / base).mkdir()
    assert normalize_root_relative(root, f"{base}/inner.py") == f"{base}/inner.py"


def _ext_registry(tmp_path):
    system = tmp_path / "system"
    workspace = tmp_path / "workspace"
    data = tmp_path / "data"
    for p in (system, workspace, data):
        p.mkdir()
    (data / "settings.json").write_text("{}", encoding="utf-8")
    (workspace / "pkg").mkdir()
    (workspace / "pkg" / "mod.py").write_text("def hello_world():\n    return 1\n", encoding="utf-8")
    reg = ToolRegistry(repo_dir=system, drive_root=data)
    reg.set_context(ToolContext(repo_dir=system, drive_root=data, workspace_root=workspace, workspace_mode="external"))
    return reg, workspace


def test_search_code_accepts_absolute_and_redundant_paths(tmp_path):
    reg, ws = _ext_registry(tmp_path)
    base = ws.name
    # absolute within root
    out_abs = reg.execute("search_code", {"query": "hello_world", "path": str(ws / "pkg"), "root": "active_workspace"})
    assert "SEARCH_ERROR" not in out_abs and "hello_world" in out_abs
    # redundant root-basename prefix
    out_red = reg.execute("search_code", {"query": "hello_world", "path": f"{base}/pkg", "root": "active_workspace"})
    assert "SEARCH_ERROR" not in out_red and "hello_world" in out_red
    # outside-root absolute path is still rejected (no leak)
    out_bad = reg.execute("search_code", {"query": "root", "path": "/etc", "root": "active_workspace"})
    assert "SEARCH_ERROR" in out_bad


def test_list_files_accepts_absolute_dir(tmp_path):
    reg, ws = _ext_registry(tmp_path)
    out = reg.execute("list_files", {"path": str(ws / "pkg"), "root": "active_workspace"})
    assert "Directory not found" not in out and "Not a directory" not in out
    assert "mod.py" in out


def test_read_file_accepts_absolute_and_redundant(tmp_path):
    reg, ws = _ext_registry(tmp_path)
    base = ws.name
    out_abs = reg.execute("read_file", {"path": str(ws / "pkg" / "mod.py"), "root": "active_workspace"})
    assert "hello_world" in out_abs
    out_red = reg.execute("read_file", {"path": f"{base}/pkg/mod.py", "root": "active_workspace"})
    assert "hello_world" in out_red


def test_resolve_resource_path_normalizes_like_repo_path(tmp_path):
    """v6.35.0 security: resolve_resource_path (used by the protected-artifact
    guards) must normalize absolute/redundant-root-prefix paths the SAME way the
    write path (repo_path) does, so a guard resolving through it sees the same
    target the handler touches — no desync bypass for a protected_artifacts path."""
    from ouroboros.tool_access import resolve_resource_path

    base = tmp_path / "repo"
    base.mkdir()
    ctx = ToolContext(repo_dir=base, drive_root=tmp_path / "d")
    (tmp_path / "d").mkdir()
    # redundant root-basename prefix -> real target (matches repo_path resolution)
    assert resolve_resource_path(ctx, root="active_workspace", path="repo/secret.bin") == (base / "secret.bin").resolve()
    # absolute-inside-root -> real target
    assert resolve_resource_path(ctx, root="active_workspace", path=str(base / "secret.bin")) == (base / "secret.bin").resolve()
    # plain relative unchanged
    assert resolve_resource_path(ctx, root="active_workspace", path="sub/x.txt") == (base / "sub" / "x.txt").resolve()


def test_resolve_resource_path_keeps_non_repo_roots_raw(tmp_path):
    """v6.35.0 security: resolve_resource_path must NOT strip a redundant
    drive-basename prefix for NON-repo roots (runtime_data, ...). Their operations
    (_data_read via _normalize_data_read_path) resolve the RAW path, so normalizing
    only the guard side here would desync the guard from the operation."""
    from ouroboros.tool_access import resolve_resource_path

    data = tmp_path / "data"
    system = tmp_path / "system"
    for p in (data, system):
        p.mkdir()
    ctx = ToolContext(repo_dir=system, drive_root=data)
    # runtime_data root resolves to drive_root; a redundant 'data/' is kept RAW.
    assert resolve_resource_path(ctx, root="runtime_data", path="data/x") == (data / "data" / "x").resolve()
    # repo roots still normalize (consistent with the dispatch boundary).
    assert resolve_resource_path(ctx, root="system_repo", path=f"{system.name}/y") == (system / "y").resolve()


def test_search_code_runtime_data_redundant_prefix_cannot_reach_project_store(tmp_path):
    """v6.35.0 security: a redundant drive-basename prefix must NOT let a
    runtime_data search slip the project-store guard (which matches the RAW path)
    and then search the normalized 'projects/...' store — the desync class codex
    flagged. Normalization is restricted to the repo roots."""
    data = tmp_path / "data"
    system = tmp_path / "system"
    for p in (data, system):
        p.mkdir()
    (data / "settings.json").write_text("{}", encoding="utf-8")
    reg = ToolRegistry(repo_dir=system, drive_root=data)
    # runtime_data `search` is a top-level managed/operator capability.
    ctx = ToolContext(repo_dir=system, drive_root=data)
    ctx.is_direct_chat = True
    reg.set_context(ctx)

    # Positive control: runtime_data search works for an ordinary subdir.
    logs = data / "logs"
    logs.mkdir()
    (logs / "notes.txt").write_text("RUNTIME_MARK_OK\n", encoding="utf-8")
    ok = reg.execute("search_code", {"query": "RUNTIME_MARK_OK", "path": "logs", "root": "runtime_data"})
    assert "RUNTIME_MARK_OK" in ok

    # The protected per-project store.
    store = data / "projects" / "proj1"
    store.mkdir(parents=True)
    (store / "secret.txt").write_text("PROJSTORE_SECRET_ZZZ\n", encoding="utf-8")

    # Direct store path is blocked by the project-store guard.
    direct = reg.execute("search_code", {"query": "PROJSTORE_SECRET_ZZZ", "path": "projects/proj1", "root": "runtime_data"})
    assert "PROJSTORE_SECRET_ZZZ" not in direct
    # Redundant drive-basename prefix must NOT slip the guard and reach the store.
    bypass = reg.execute("search_code", {"query": "PROJSTORE_SECRET_ZZZ", "path": "data/projects/proj1", "root": "runtime_data"})
    assert "PROJSTORE_SECRET_ZZZ" not in bypass


def test_write_file_normalizes_redundant_prefix_for_new_file(tmp_path):
    reg, ws = _ext_registry(tmp_path)
    base = ws.name
    out = reg.execute("write_file", {"root": "active_workspace", "path": f"{base}/created.txt", "content": "hi\n"})
    assert "ERROR" not in out
    # The redundant root-basename prefix is stripped: file lands at the root,
    # NOT double-nested under <root>/<base>/.
    assert (ws / "created.txt").is_file()
    assert not (ws / base / "created.txt").exists()


def test_query_code_path_does_not_escape_on_redundant_prefix(tmp_path):
    reg, ws = _ext_registry(tmp_path)
    base = ws.name
    out = reg.execute("query_code", {"op": "digest", "path": f"{base}/pkg/mod.py", "root": "active_workspace"})
    # The redundant prefix must no longer produce a path-escape / arg error.
    assert "path escapes root" not in out and "TOOL_ARG_ERROR" not in out
