"""Structural contracts for the semantic-no-op shell tool extraction.

The output-audit helpers live in ``ouroboros.tools.shell_audit``. The owner
map names that module alongside the process, output and effects leaves;
the facade identity clause covers every moved helper. The leaves remain
non-catalog owners without backedges into the facade.
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
    "_directory_fingerprint": shell_outputs,
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
    # run_script accepts any installed file interpreter; its temporary file
    # lives in an ignored workspace directory or the existing task drive.
    # Workflow scope: explicit saved-setting references and lazy-output guidance.
    # run_command states its real contract: only a bare builtin as cmd[0] is
    # refused, a background child stalls the call and is never tracked after it
    # (tests/test_run_command_schema_truth.py). Rolled once more: the sentence no
    # longer says the child is never STOPPED, which is false where the timeout kill
    # still reaches the exited shell's process group (Linux). Rolled again for
    # source-addressed delivery (owner Q4): both tools gain the additive
    # `view_head_chars`/`view_tail_chars` first-view request (the complete output
    # stays an exact readable source either way); the handlers' **kwargs bind them.
    assert hashlib.sha256(schema_bytes).hexdigest() == (
        "47e5051938e513c9798291e8d24224d0a9f8825cd9554bbc7ed16721772bc72f"
    )
    original = json.loads(schema_bytes)
    for schema in original:
        schema["parameters"]["properties"].pop("env_from_settings")
        schema["parameters"]["properties"]["outputs"]["description"] = (
            "Generated file paths to copy/register into the task artifact store after success."
        )
    original[1]["description"] = (
        "Run a short task-scoped temporary script with a declared interpreter. "
        "Use for multi-line diagnostics or harness helpers; generated script files live under the task drive. "
        "The underlying command result echoes the resolved cwd."
    )
    assert hashlib.sha256(json.dumps(original, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":")).encode()).hexdigest() == (
        "13ade213656361e8faae67439a364d8629d076fe3f27e72e4dfb03f48094889b"
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
    assert counts["ouroboros.tools.shell_outputs"] <= 1000
