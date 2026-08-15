# tests/test_operation_cwd_resolution.py — one cwd resolution per operation (RWS v2 §3.1, D4).
#
# The dispatcher, the shell guards, the python resolver and the handler each used
# to call the cwd resolver independently: up to six resolutions of the same request
# per tool call, each free to disagree. This pins the replacement — and it pins the
# PROPERTY, not the machinery.
#
# The machinery changed once already. This branch first held it with an
# operation-scoped memo (`ShellCwdResolutions`): resolve eagerly at the dispatch
# boundary, replay for every later consumer, tear the memo down after. Upstream
# then arrived at the same invariant from the other side, with a SEALED VALUE
# instead of a cache — `ResolvedResourceBinding`, built once in `_dispatch` and
# threaded to each consumer as an argument. The binding is the better answer: a memo
# makes re-resolution CHEAP, a value makes it IMPOSSIBLE, and "impossible" is what
# an invariant wants. So the memo is gone and its lifecycle tests with it.
#
# What must not change is the count. `_select_process_target` is the resolver body
# every Home path bottoms out in — `resolve_shell_cwd` and
# `build_resolved_resource_binding` are both front doors onto it — so counting its
# invocations across a whole dispatch is the same measurement the memo tests made,
# taken at the one site that is still there.
import pathlib
import types

import pytest

from ouroboros import tool_access
from ouroboros.tool_access import build_resolved_resource_binding, resolve_shell_cwd
from tests.golden_traces import scenarios


@pytest.fixture
def resolver_counter(monkeypatch):
    """Count invocations of the resolver BODY, whichever front door reached it."""
    calls: list[tuple[str, str]] = []
    original = tool_access._select_process_target

    def counted(ctx, cwd, operation, **kwargs):
        calls.append((str(cwd or ""), str(operation)))
        return original(ctx, cwd, operation, **kwargs)

    monkeypatch.setattr(tool_access, "_select_process_target", counted)
    return calls


def _ctx(base: pathlib.Path):
    repo = base / "repo"
    (repo / "sub").mkdir(parents=True)
    drive = base / "data"
    (drive / "task_drives" / "t1").mkdir(parents=True)
    return types.SimpleNamespace(
        repo_dir=repo,
        drive_root=drive,
        task_id="t1",
        workspace_root=None,
        workspace_mode="",
        task_metadata={},
        is_workspace_mode=lambda: False,
    )


# ── the binding is the resolution ───────────────────────────────────────────
def test_the_binding_carries_the_resolution_rather_than_repeating_it(tmp_path, resolver_counter):
    """Building the binding resolves once; reading it back resolves nothing.

    This is the property the memo used to buy with a cache. A consumer handed the
    binding cannot re-resolve even by accident, because there is no resolver call
    left to make — the answer is a field.
    """
    ctx = _ctx(tmp_path)
    binding = build_resolved_resource_binding(ctx, operation="shell", process_cwd="sub")
    assert resolver_counter == [("sub", "shell")]
    assert binding.target_path == (tmp_path / "repo" / "sub")
    assert binding.target_path == (tmp_path / "repo" / "sub")
    assert resolver_counter == [("sub", "shell")]


def test_distinct_operations_each_resolve_once(tmp_path, resolver_counter):
    ctx = _ctx(tmp_path)
    build_resolved_resource_binding(ctx, operation="shell", process_cwd="sub")
    build_resolved_resource_binding(ctx, operation="service", process_cwd="sub")
    assert resolver_counter == [("sub", "shell"), ("sub", "service")]


def test_a_refusal_is_a_refusal_at_the_one_site(tmp_path, resolver_counter):
    """A bad cwd fails where it is resolved, once — not once per consumer."""
    ctx = _ctx(tmp_path)
    with pytest.raises(Exception):
        build_resolved_resource_binding(ctx, operation="shell", process_cwd="../escape")
    assert len(resolver_counter) == 1


def test_outside_a_dispatch_the_resolver_runs_every_time(tmp_path, resolver_counter):
    """Admission, CLI and tests resolve directly — unmemoized, unchanged."""
    ctx = _ctx(tmp_path)
    resolve_shell_cwd(ctx, "")
    resolve_shell_cwd(ctx, "")
    assert resolver_counter == [("", "shell"), ("", "shell")]


# ── the dispatch invariant ──────────────────────────────────────────────────
@pytest.mark.serial
def test_execute_resolves_the_process_cwd_exactly_once(tmp_path, resolver_counter):
    """A whole run_command dispatch — dispatcher arms, shell guards, python
    resolver, argument projection, handler and post-checks — resolves its cwd ONE
    time."""
    registry, _roots = scenarios._normal(tmp_path)
    out = registry.execute("run_command", {"cmd": ["echo", "once"]})
    assert "once" in out
    assert resolver_counter == [("", "shell")], resolver_counter


@pytest.mark.serial
def test_execute_resolves_a_workspace_process_cwd_exactly_once(tmp_path, resolver_counter):
    registry, _roots = scenarios._workspace(tmp_path)
    registry.execute("run_command", {"cmd": ["echo", "ws"]})
    assert resolver_counter == [("", "shell")], resolver_counter


def test_execute_of_a_cwd_less_tool_resolves_no_process_cwd(tmp_path, resolver_counter):
    """read_file has no process cwd: resolving one would fabricate a fact — and a
    possible refusal — the operation does not have."""
    registry, _roots = scenarios._normal(tmp_path)
    registry.execute("read_file", {"path": "hello.txt", "root": "active_workspace"})
    assert resolver_counter == []
