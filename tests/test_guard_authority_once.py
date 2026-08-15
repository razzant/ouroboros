# tests/test_guard_authority_once.py — every guard authority runs ONCE per call.
#
# §3.3: anti-duplication is BEHAVIORAL, not naming. The postmortem shape was one
# policy enforced per channel — a second copy of a guard for a second placement,
# breached through whichever channel nobody remembered to update. A duplicate
# named `ssh_guard` cannot evade this test, because it counts INVOCATIONS of the
# authorities themselves through the golden-trace instrumentation.
#
# `workspace_mode_block_reason` is deliberately excluded: in this codebase that
# symbol is `ToolContext.is_workspace_mode()`'s internal metadata validation, so
# it fires once per is_workspace_mode() call all over the pipeline. It is a
# technical predicate, not a guard authority with a decision of its own — and its
# multiplicity is exactly what the golden traces already pin.
from collections import Counter

import pytest

from tests.golden_traces import capture, scenarios

# The pipeline's guard AUTHORITIES: each owns one policy decision, so each may
# be consulted at most once per dispatch.
_AUTHORITIES = frozenset({
    "_ephemeral_block",
    "_disabled_tools",
    "_builtin_tool_availability",
    "_subagent_and_update_gate",
    "_heal_mode_block",
    "_resolve_python_predispatch",
    "_run_shell_safety_check",
    "check_safety",
    "light_cognitive_or_root_redirect",
    "protected_paths_in",
    "_normalize_dispatch_path_args",
    "_snapshot_owner_files",
    "_light_repo_snapshot",
    "_git_ref_snapshot",
    "_invoke_builtin_handler",
    "_run_shell_post_checks",
    "_compose_execute_result",
})

# Authorities that MUST run for a normal (non-early-return) process dispatch.
_REQUIRED_FOR_PROCESS_CALL = (
    "_normalize_dispatch_path_args",
    "_ephemeral_block",
    "_disabled_tools",
    "_builtin_tool_availability",
    "_subagent_and_update_gate",
    "_resolve_python_predispatch",
    "_run_shell_safety_check",
    "check_safety",
    "_invoke_builtin_handler",
    "_compose_execute_result",
)


def _authority_counts(registry, tool, args):
    recorder = capture.TraceRecorder(capture.Normalizer({}))
    with capture.capture_execute_trace(recorder):
        result = registry.execute(tool, dict(args))
    counts = Counter(e["fn"] for e in recorder.events if e["fn"] in _AUTHORITIES)
    return counts, result


@pytest.mark.serial
def test_each_guard_authority_is_consulted_at_most_once(tmp_path):
    registry, _roots = scenarios._normal(tmp_path)
    counts, result = _authority_counts(registry, "run_command", {"cmd": ["echo", "once"]})
    assert "once" in result
    repeated = {fn: n for fn, n in counts.items() if n > 1}
    assert not repeated, repeated
    for fn in _REQUIRED_FOR_PROCESS_CALL:
        assert counts[fn] == 1, f"{fn} ran {counts[fn]} times"


@pytest.mark.serial
def test_workspace_placement_does_not_duplicate_any_authority(tmp_path):
    """An external workspace routes more work through the placement seams; no
    authority may be consulted twice because of it."""
    registry, _roots = scenarios._workspace(tmp_path)
    counts, _result = _authority_counts(registry, "run_command", {"cmd": ["echo", "ws"]})
    repeated = {fn: n for fn, n in counts.items() if n > 1}
    assert not repeated, repeated


@pytest.mark.parametrize(
    "tool,args",
    [
        ("read_file", {"path": "hello.txt", "root": "active_workspace"}),
        ("list_files", {"path": ".", "root": "active_workspace"}),
        ("write_file", {"path": "new.txt", "content": "x", "root": "task_drive"}),
        ("edit_text", {"path": "hello.txt", "old": "hello", "new": "hi", "root": "active_workspace"}),
    ],
)
def test_file_tools_consult_each_authority_at_most_once(tmp_path, tool, args):
    registry, _roots = scenarios._normal(tmp_path)
    counts, _result = _authority_counts(registry, tool, args)
    repeated = {fn: n for fn, n in counts.items() if n > 1}
    assert not repeated, repeated


def test_an_early_return_consults_no_later_authority(tmp_path):
    """Error precedence: a tool withheld by the contract never reaches the
    path/cwd/interpreter guards at all."""
    registry, _roots = scenarios._normal(tmp_path)
    registry._ctx.task_metadata["task_contract"] = {"disabled_tools": ["run_command"]}
    counts, result = _authority_counts(registry, "run_command", {"cmd": ["echo", "x"]})
    assert result.startswith("⚠️ RESOURCE_CONSTRAINT_BLOCKED")
    assert counts["_disabled_tools"] == 1
    for later in ("_resolve_python_predispatch", "_run_shell_safety_check", "check_safety", "_invoke_builtin_handler"):
        assert counts[later] == 0


@pytest.mark.serial
def test_every_recorded_authority_name_is_classified(tmp_path):
    """A newly instrumented guard must be classified as an authority or
    explicitly excluded — otherwise this suite would silently stop covering it."""
    registry, _roots = scenarios._normal(tmp_path)
    recorder = capture.TraceRecorder(capture.Normalizer({}))
    with capture.capture_execute_trace(recorder):
        registry.execute("run_command", {"cmd": ["echo", "classified"]})
    seen = {e["fn"] for e in recorder.events}
    known = _AUTHORITIES | {
        # technical predicates, multiplicity pinned by the golden traces instead
        "workspace_mode_block_reason",
        "_resource_allowed",
        "_worktree_status_snapshot",
    }
    assert seen <= known, sorted(seen - known)
