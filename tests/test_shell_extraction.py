"""Structural contracts for the semantic-no-op shell tool extraction."""

from __future__ import annotations

import ast
import hashlib
import json
import pathlib

from ouroboros.tool_module_inventory import (
    build_frozen_tool_manifest,
    discover_tool_module_inventory,
    load_frozen_tool_modules,
)
from ouroboros.tools import (
    shell,
    shell_effects,
    shell_outputs,
    shell_process,
)


REPO = pathlib.Path(__file__).parents[1]
TOOLS = REPO / "ouroboros" / "tools"

_LEAVES = (shell_process, shell_outputs, shell_effects)

_MOVED_OWNERS = {
    "_RUN_SHELL_DEFAULT_TIMEOUT_SEC": shell_process,
    "_active_subprocesses": shell_process,
    "_describe_returncode": shell_process,
    "_executor_can_run_cwd": shell_process,
    "_format_process_output": shell_process,
    "_kill_process_group": shell_process,
    "_resolve_effective_timeout": shell_process,
    "_shell_env_for_cwd": shell_process,
    "_subprocess_lock": shell_process,
    "_tracked_subprocess_run": shell_process,
    "kill_all_tracked_subprocesses": shell_process,
    "_EMBEDDED_OUTPUT_PATH_RE": shell_outputs,
    "_OUTPUT_CALL_PATH_RE": shell_outputs,
    "_OUTPUT_DIR_MAX_BYTES": shell_outputs,
    "_OUTPUT_DIR_MAX_FILES": shell_outputs,
    "_OUTPUT_REDIRECT_PATH_RE": shell_outputs,
    "_OUTPUT_STAT_SLACK_SEC": shell_outputs,
    "_SENSITIVE_OUTPUT_COMPONENT_NAMES": shell_outputs,
    "_SENSITIVE_OUTPUT_MARKERS": shell_outputs,
    "_SENSITIVE_OUTPUT_NAMES": shell_outputs,
    "_SENSITIVE_OUTPUT_SUFFIXES": shell_outputs,
    "_UNDECLARED_OUTPUTS_MARKER": shell_outputs,
    "_USER_FILE_OPEN_WRITE_CALL_RE": shell_outputs,
    "_USER_FILE_REDIRECT_RE": shell_outputs,
    "_USER_FILE_WRITE_CALL_RE": shell_outputs,
    "_allowed_output_roots": shell_outputs,
    "_bounded_directory_fingerprint": shell_outputs,
    "_changed_path_covers": shell_outputs,
    "_directory_fingerprint_from_entries": shell_outputs,
    "_fingerprint_output": shell_outputs,
    "_mentioned_user_file_outputs_without_declaration": shell_outputs,
    "_protected_output_source_reason": shell_outputs,
    "_register_process_outputs": shell_outputs,
    "_resolve_declared_output": shell_outputs,
    "_scan_directory_output_members": shell_outputs,
    "_sensitive_output_component_reason": shell_outputs,
    "_snapshot_declared_outputs": shell_outputs,
    "_get_changed_files": shell_effects,
    "_get_diff_stat": shell_effects,
    "_protected_runtime_dirty_paths": shell_effects,
    "_record_scratch_fingerprints": shell_effects,
    "_resolve_git_root": shell_effects,
    "_resolve_scratch_abs": shell_effects,
    "_restore_protected_runtime_paths": shell_effects,
    "_scratch_safety_reason": shell_effects,
    "_shallow_listing": shell_effects,
    "_status_snapshot": shell_effects,
    "_tree_fingerprint": shell_effects,
    "_user_files_run_had_effect": shell_effects,
}


def test_shell_leaves_are_non_catalog_owners_without_shell_backedges(tmp_path):
    for module in _LEAVES:
        source_path = pathlib.Path(module.__file__)
        tree = ast.parse(source_path.read_text(encoding="utf-8"))
        assert not any(
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == "get_tools"
            for node in tree.body
        )
        assert not any(
            isinstance(node, ast.ImportFrom)
            and node.module == "ouroboros.tools.shell"
            for node in ast.walk(tree)
        )
        assert not any(
            isinstance(node, ast.Import)
            and any(alias.name == "ouroboros.tools.shell" for alias in node.names)
            for node in ast.walk(tree)
        )

    source_inventory = discover_tool_module_inventory(TOOLS)
    assert "shell" in source_inventory.tool_modules
    for module in _LEAVES:
        assert module.__name__.rsplit(".", 1)[-1] not in source_inventory.tool_modules
    manifest = tmp_path / "_frozen_tool_modules.v1.json"
    build_frozen_tool_manifest(TOOLS, manifest)
    assert load_frozen_tool_modules(manifest) == source_inventory.tool_modules


def test_shell_catalog_schema_bytes_and_handler_owners_are_stable():
    entries = shell.get_tools()
    assert tuple(entry.name for entry in entries) == ("run_command", "run_script")
    schema_bytes = json.dumps(
        [entry.schema for entry in entries],
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode()
    assert hashlib.sha256(schema_bytes).hexdigest() == (
        "1e012faf410bf57c91a896d227aa4175cd08d38794c4b8f0404e390e79b1730a"
    )
    assert {
        entry.name: (entry.handler.__module__, entry.handler.__name__)
        for entry in entries
    } == {
        "run_command": ("ouroboros.tools.shell", "_run_shell"),
        "run_script": ("ouroboros.tools.shell", "_run_script"),
    }


def test_shell_facade_reexports_every_moved_identity():
    """``tools/shell.py`` keeps the exact objects, so existing importers — the
    supervisor, server panic paths, skill exec, verify, media and vision — see no
    identity change."""
    for name, owner in _MOVED_OWNERS.items():
        assert hasattr(shell, name), name
        assert getattr(shell, name) is getattr(owner, name), name
    owned = {name for module in _LEAVES for name in vars(module)}
    assert set(_MOVED_OWNERS) <= owned


def test_shell_extraction_size_bounds_have_meaningful_headroom():
    counts = {
        module.__name__: len(
            pathlib.Path(module.__file__).read_text(encoding="utf-8").splitlines()
        )
        for module in (shell, *_LEAVES)
    }
    assert counts["ouroboros.tools.shell"] <= 800
    assert all(count <= 1000 for count in counts.values())
    assert 400 <= counts["ouroboros.tools.shell_outputs"] <= 1000
