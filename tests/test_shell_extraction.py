"""Structural contracts for the semantic-no-op shell tool extraction.

Carried from the v7 reference (ouroboros_v7_wip @ 9f691656) with the following
identity continuations to THIS tree's bytes:

1. Ten output-audit owners the reference placed in ``shell_outputs`` were
   relocated by upstream itself into ``ouroboros/tools/shell_audit.py`` (a
   post-cutoff upstream extraction); the ownership map below names the
   upstream owner for those rows and the facade identity clause still holds
   for every one of them.
2. The frozen-tool-inventory clauses are dropped: ``ouroboros.tool_module_
   inventory`` is a D04-family v7 leaf absent from this tree; the
   non-catalog-owner and no-backedge clauses keep the structural half of that
   contract. The inventory clause returns with its leaf.
"""

from __future__ import annotations

import ast
import hashlib
import json
import pathlib

from ouroboros.tools import (
    shell,
    shell_audit,
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
    "_OUTPUT_DIR_MAX_BYTES": shell_outputs,
    "_OUTPUT_DIR_MAX_FILES": shell_outputs,
    "_SENSITIVE_OUTPUT_COMPONENT_NAMES": shell_outputs,
    "_SENSITIVE_OUTPUT_MARKERS": shell_outputs,
    "_SENSITIVE_OUTPUT_NAMES": shell_outputs,
    "_SENSITIVE_OUTPUT_SUFFIXES": shell_outputs,
    "_bounded_directory_fingerprint": shell_outputs,
    "_changed_path_covers": shell_outputs,
    "_directory_fingerprint_from_entries": shell_outputs,
    "_fingerprint_output": shell_outputs,
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
    # Owners the reference assigned to shell_outputs that upstream itself
    # extracted into tools/shell_audit.py after the reference cutoff. The
    # facade identity contract below covers them all the same.
    "_EMBEDDED_OUTPUT_PATH_RE": shell_audit,
    "_OUTPUT_CALL_PATH_RE": shell_audit,
    "_OUTPUT_REDIRECT_PATH_RE": shell_audit,
    "_OUTPUT_STAT_SLACK_SEC": shell_audit,
    "_UNDECLARED_OUTPUTS_MARKER": shell_audit,
    "_USER_FILE_OPEN_WRITE_CALL_RE": shell_audit,
    "_USER_FILE_REDIRECT_RE": shell_audit,
    "_USER_FILE_WRITE_CALL_RE": shell_audit,
    "_allowed_output_roots": shell_audit,
    "_mentioned_user_file_outputs_without_declaration": shell_audit,
}


def test_shell_leaves_are_non_catalog_owners_without_shell_backedges():
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


def test_shell_catalog_schema_bytes_and_handler_owners_are_stable():
    entries = shell.get_tools()
    assert tuple(entry.name for entry in entries) == ("run_command", "run_script")
    schema_bytes = json.dumps(
        [entry.schema for entry in entries],
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode()
    # Re-pinned for upstream ``d0caa69b`` ("tool refusals: name the unsupported
    # argument and own the timeout alias once"), which deleted the duplicate
    # ``timeout`` alias property from BOTH schemas because the alias now lives
    # once in ``tool_resolution._TOOL_ARG_ALIASES["*"]``. That deletion is the
    # only difference from the previous digest
    # ``1e012faf410bf57c91a896d227aa4175cd08d38794c4b8f0404e390e79b1730a``:
    # re-inserting the two removed properties into these very schema objects
    # reproduces it byte for byte.
    assert hashlib.sha256(schema_bytes).hexdigest() == (
        "0c0215cafdb54ee2231e43f9edafbe8aa21d4a5aa3841eba4ab45b5ab5e3eb42"
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
    owned |= set(vars(shell_audit))
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
