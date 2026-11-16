"""Skill-family Node runtime policy (bundled-first + health rollback) tests.

Covers the ONE owner of the precedence (platform_layer.select_skill_node_runtime
and the emergency PATH-prepend derived from it), the skill_exec/skill_preflight
routing through it, the workspace_preflight node exec probe (broken statuses,
npm-family interpreter_broken, summary/render, bounded-path probe timeout), and
the world_profiler honest node markers. All probes are mocked (no real node is
executed); the platform memo is bypassed naturally because every fake runtime
lives at a unique tmp_path.
"""
from __future__ import annotations

import os
import pathlib

import pytest

from ouroboros import node_runtime
from ouroboros import platform_layer
from ouroboros import workspace_preflight as wp
from ouroboros import world_profiler as wprof
from ouroboros.platform_layer import NodeRuntimeHealth



@pytest.fixture(autouse=True)
def _no_bundle_bases_of_this_host(monkeypatch):
    """A venv made from the installed app's Python finds that app's Node among the bundle bases: tests name theirs."""
    monkeypatch.setattr(platform_layer, "bundled_resource_bases", lambda: [])


def _unpinned(monkeypatch):
    """A host the pin does not govern (no pin, or no archive for it): skills keep a shipped copy or PATH."""
    monkeypatch.setattr(node_runtime, "pinned_node_wait", lambda: (False, ""))


@pytest.fixture(autouse=True)
def _isolated_node_health_memo():
    """T18: the probe memo is a module-level registry — reset it around every
    test so no verdict leaks between tests on the same xdist worker."""
    saved = dict(node_runtime._NODE_HEALTH_MEMO)
    node_runtime._NODE_HEALTH_MEMO.clear()
    try:
        yield
    finally:
        node_runtime._NODE_HEALTH_MEMO.clear()
        node_runtime._NODE_HEALTH_MEMO.update(saved)

def _make_exec(dir_path: pathlib.Path, name: str) -> pathlib.Path:
    """Create a fake executable findable by shutil.which on POSIX and Windows."""
    dir_path.mkdir(parents=True, exist_ok=True)
    for candidate in (name, f"{name}.exe"):
        target = dir_path / candidate
        target.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        target.chmod(0o755)
    return dir_path / (f"{name}.exe" if os.name == "nt" else name)


def _patch_probe(monkeypatch, outcomes_by_prefix):
    """Route _probe_node_version_outcome by path prefix: {prefix: (version, reason)}."""

    def fake_probe(path, timeout_sec=10):
        for prefix, outcome in outcomes_by_prefix.items():
            if str(path).startswith(str(prefix)):
                return outcome
        raise AssertionError(f"unexpected probe of {path!r}")

    monkeypatch.setattr(node_runtime, "_probe_node_version_outcome", fake_probe)


def _isolate_path(monkeypatch, bin_dir: pathlib.Path) -> None:
    monkeypatch.setattr(node_runtime._platform, "bootstrap_process_path", lambda: [])
    monkeypatch.setenv("PATH", str(bin_dir))


# ---------------------------------------------------------------------------
# select_skill_node_runtime — the single owner of the precedence order
# ---------------------------------------------------------------------------


def test_select_bundled_healthy_wins_even_over_healthy_path_node(tmp_path, monkeypatch):
    bundle_bin = tmp_path / "bundle" / "bin"
    bundled = _make_exec(bundle_bin, "node")
    path_bin = tmp_path / "hostbin"
    _make_exec(path_bin, "node")
    _isolate_path(monkeypatch, path_bin)
    monkeypatch.setattr(node_runtime._platform, "resolve_bundled_node", lambda: str(bundled))
    _patch_probe(monkeypatch, {bundle_bin: ("24.16.0", ""), path_bin: ("25.8.1", "")})

    assert node_runtime.select_skill_node_runtime() == (str(bundled), "bundled")


def test_select_rolls_back_to_healthy_path_node_when_bundled_broken(tmp_path, monkeypatch):
    _unpinned(monkeypatch)
    bundle_bin = tmp_path / "bundle" / "bin"
    bundled = _make_exec(bundle_bin, "node")
    path_bin = tmp_path / "hostbin"
    path_node = _make_exec(path_bin, "node")
    _isolate_path(monkeypatch, path_bin)
    monkeypatch.setattr(node_runtime._platform, "resolve_bundled_node", lambda: str(bundled))
    _patch_probe(monkeypatch, {bundle_bin: ("", "signal:SIGKILL"), path_bin: ("25.8.1", "")})

    selected, provenance = node_runtime.select_skill_node_runtime()
    assert provenance == "path"
    assert pathlib.Path(selected) == path_node


def test_select_uses_path_node_when_no_bundle(tmp_path, monkeypatch):
    _unpinned(monkeypatch)
    path_bin = tmp_path / "hostbin"
    path_node = _make_exec(path_bin, "node")
    _isolate_path(monkeypatch, path_bin)
    monkeypatch.setattr(node_runtime._platform, "resolve_bundled_node", lambda: None)
    _patch_probe(monkeypatch, {path_bin: ("25.8.1", "")})

    selected, provenance = node_runtime.select_skill_node_runtime()
    assert provenance == "path"
    assert pathlib.Path(selected) == path_node


def test_select_reports_both_probe_verdicts_when_nothing_usable(tmp_path, monkeypatch):
    _unpinned(monkeypatch)
    bundle_bin = tmp_path / "bundle" / "bin"
    bundled = _make_exec(bundle_bin, "node")
    path_bin = tmp_path / "hostbin"
    _make_exec(path_bin, "node")
    _isolate_path(monkeypatch, path_bin)
    monkeypatch.setattr(node_runtime._platform, "resolve_bundled_node", lambda: str(bundled))
    _patch_probe(monkeypatch, {bundle_bin: ("", "exit:1"), path_bin: ("", "signal:SIGKILL")})

    selected, reason = node_runtime.select_skill_node_runtime()
    assert selected == ""
    assert "bundled:broken:exit:1" in reason
    assert "path:broken:signal:SIGKILL" in reason


def test_select_reports_absence_when_neither_exists(tmp_path, monkeypatch):
    _unpinned(monkeypatch)
    empty_bin = tmp_path / "empty"
    empty_bin.mkdir()
    _isolate_path(monkeypatch, empty_bin)
    monkeypatch.setattr(node_runtime._platform, "resolve_bundled_node", lambda: None)
    _patch_probe(monkeypatch, {})  # any probe would raise

    selected, reason = node_runtime.select_skill_node_runtime()
    assert selected == ""
    assert "bundled:absent" in reason
    assert "path:missing:not_on_path" in reason


# ---------------------------------------------------------------------------
# skill_node_emergency_path_dir — the pin's Node first, else an emergency-only PATH prepend
# ---------------------------------------------------------------------------


def test_emergency_dir_returned_only_when_path_node_unusable(tmp_path, monkeypatch):
    bundle_bin = tmp_path / "bundle" / "bin"
    bundled = _make_exec(bundle_bin, "node")
    path_bin = tmp_path / "hostbin"
    _make_exec(path_bin, "node")
    _isolate_path(monkeypatch, path_bin)
    monkeypatch.setattr(node_runtime._platform, "resolve_bundled_node", lambda: str(bundled))
    _patch_probe(monkeypatch, {bundle_bin: ("24.16.0", ""), path_bin: ("", "signal:SIGKILL")})

    assert node_runtime.skill_node_emergency_path_dir() == str(bundled.parent)


@pytest.mark.parametrize("pinned", [False, True])
def test_emergency_dir_on_healthy_path_node_only_under_the_pin(tmp_path, monkeypatch, pinned):
    bundle_bin = tmp_path / "bundle" / "bin"
    bundled = _make_exec(bundle_bin, "node")
    path_bin = tmp_path / "hostbin"
    _make_exec(path_bin, "node")
    _isolate_path(monkeypatch, path_bin)
    monkeypatch.setattr(node_runtime._platform, "resolve_bundled_node", lambda: str(bundled))
    monkeypatch.setattr(node_runtime, "pinned_node_wait", lambda: (True, str(bundled)) if pinned else (False, ""))
    _patch_probe(monkeypatch, {bundle_bin: ("24.22.3", ""), path_bin: ("25.8.1", "")})

    # Bundled wins the direct selection. Without a pin there is NO emergency: the child env stays byte-identical on a
    # healthy PATH. Under the pin (R7, Opus: an `npm` companion's shebang found PATH node) the reviewed Node comes first.
    assert node_runtime.skill_node_emergency_path_dir() == (str(bundled.parent) if pinned else "")


def test_emergency_dir_empty_without_bundle_and_probes_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr(node_runtime._platform, "resolve_bundled_node", lambda: None)
    _patch_probe(monkeypatch, {})  # any probe would raise

    assert node_runtime.skill_node_emergency_path_dir() == ""


# ---------------------------------------------------------------------------
# skill_exec / skill_preflight route node through the policy helper
# ---------------------------------------------------------------------------


def test_skill_exec_resolver_routes_node_through_policy(monkeypatch):
    from ouroboros.tools import skill_exec as skill_exec_mod

    monkeypatch.setattr(
        platform_layer,
        "select_skill_node_runtime",
        lambda timeout_sec=10: ("/x/bundle/bin/node", "bundled"),
    )
    assert skill_exec_mod._resolve_runtime_binary("node") == ("/x/bundle/bin/node", "")

    monkeypatch.setattr(
        platform_layer,
        "select_skill_node_runtime",
        lambda timeout_sec=10: ("", "bundled:broken:signal:SIGKILL; path:missing:not_on_path"),
    )
    resolved, reason = skill_exec_mod._resolve_runtime_binary("node")
    assert resolved is None
    assert "bundled:broken:signal:SIGKILL" in reason


def test_skill_preflight_resolver_routes_node_through_policy(monkeypatch):
    from ouroboros.tools import skill_preflight as sp

    monkeypatch.setattr(
        platform_layer,
        "select_skill_node_runtime",
        lambda timeout_sec=10: ("/x/host/node", "path"),
    )
    assert sp._resolve_runtime("node") == ("/x/host/node", "")

    monkeypatch.setattr(
        platform_layer,
        "select_skill_node_runtime",
        lambda timeout_sec=10: ("", "bundled:absent; path:broken:signal:SIGKILL"),
    )
    resolved, reason = sp._resolve_runtime("node")
    assert resolved is None
    assert "path:broken:signal:SIGKILL" in reason


# ---------------------------------------------------------------------------
# workspace_preflight — node exec probe, family downgrade, summary/render
# ---------------------------------------------------------------------------


def _broken_node_health(path, timeout_sec=10):
    return NodeRuntimeHealth(status="broken", reason="signal:SIGKILL", path=str(path))


def test_preflight_marks_broken_node_and_family_without_family_probes(tmp_path, monkeypatch):
    host_bin = tmp_path / "hostbin"
    node_path = _make_exec(host_bin, "node")
    _make_exec(host_bin, "npm")
    _make_exec(host_bin, "npx")
    monkeypatch.setenv("PATH", str(host_bin))

    bundled = str(tmp_path / "bundle" / "bin" / "node")
    probed = []

    def fake_health(path, timeout_sec=10):
        probed.append(str(path))
        if str(path) == bundled:
            return NodeRuntimeHealth(status="healthy", version="24.16.0", path=str(path))
        return _broken_node_health(path, timeout_sec)

    monkeypatch.setattr(wp, "node_runtime_health", fake_health)
    monkeypatch.setattr(wp, "resolve_bundled_node", lambda: bundled)

    tools = wp._probe_tools(["node", "npm", "npx", "pnpm", "yarn", "git"])

    node = tools["node"]
    assert node["available"] is False
    assert node["status"] == "found_but_broken"
    assert node["probe_reason"] == "signal:SIGKILL"
    assert node["bundled_fallback"] == bundled
    assert pathlib.Path(node["path"]) == node_path
    for name in ("npm", "npx"):
        assert tools[name]["available"] is False
        assert tools[name]["status"] == "interpreter_broken"
        assert tools[name]["reason"] == "interpreter_broken"
    # pnpm/yarn are simply missing (which() found nothing) — no interpreter marker.
    assert tools["pnpm"] == {"available": False, "path": ""}
    assert tools["yarn"] == {"available": False, "path": ""}
    # ONLY node (and the bundled fallback) is exec-probed — never the family.
    assert probed
    assert all(pathlib.Path(p).name.split(".")[0] == "node" for p in probed)


def test_preflight_summary_and_render_broken_tools_line(tmp_path, monkeypatch):
    host_bin = tmp_path / "hostbin"
    _make_exec(host_bin, "node")
    _make_exec(host_bin, "npm")
    monkeypatch.setenv("PATH", str(host_bin))
    bundled = str(tmp_path / "bundle" / "bin" / "node")
    monkeypatch.setattr(
        wp,
        "node_runtime_health",
        lambda path, timeout_sec=10: (
            NodeRuntimeHealth(status="healthy", version="24.16.0", path=str(path))
            if str(path) == bundled
            else _broken_node_health(path)
        ),
    )
    monkeypatch.setattr(wp, "resolve_bundled_node", lambda: bundled)

    tools = wp._probe_tools(["node", "npm", "git"])
    summary = wp.summarize_workspace_preflight({"workspace_root": str(tmp_path), "tools": tools})

    assert "node" not in summary["tools"]["available"]
    assert "node" not in summary["tools"]["missing"]
    assert summary["tools"]["broken"] == [
        {"tool": "node", "reason": "signal:SIGKILL; bundled fallback available"},
        {"tool": "npm", "reason": "interpreter_broken"},
    ]

    rendered = wp.render_workspace_preflight_summary(summary)
    assert (
        "- broken_tools: node (signal:SIGKILL; bundled fallback available), "
        "npm (interpreter_broken)" in rendered
    )


def test_preflight_healthy_node_keeps_summary_and_render_shape(tmp_path, monkeypatch):
    host_bin = tmp_path / "hostbin"
    _make_exec(host_bin, "node")
    monkeypatch.setenv("PATH", str(host_bin))
    monkeypatch.setattr(
        wp,
        "node_runtime_health",
        lambda path, timeout_sec=10: NodeRuntimeHealth(
            status="healthy", version="25.8.1", path=str(path)
        ),
    )
    monkeypatch.setattr(
        wp, "resolve_bundled_node",
        lambda: (_ for _ in ()).throw(AssertionError("bundled must not be consulted on a healthy PATH")),
    )

    tools = wp._probe_tools(["node", "git"])
    assert tools["node"]["available"] is True
    assert tools["node"]["status"] == "available"
    assert tools["node"]["version"] == "25.8.1"

    summary = wp.summarize_workspace_preflight({"workspace_root": str(tmp_path), "tools": tools})
    assert "broken" not in summary["tools"]
    rendered = wp.render_workspace_preflight_summary(summary)
    assert "broken_tools" not in rendered
    assert "node" in summary["tools"]["available"]


def test_preflight_node_probe_uses_short_timeout_for_bounded_admission(tmp_path, monkeypatch):
    host_bin = tmp_path / "hostbin"
    _make_exec(host_bin, "node")
    monkeypatch.setenv("PATH", str(host_bin))
    seen = {}

    def fake_health(path, timeout_sec=10):
        seen["timeout_sec"] = timeout_sec
        return NodeRuntimeHealth(status="healthy", version="25.8.1", path=str(path))

    monkeypatch.setattr(wp, "node_runtime_health", fake_health)

    wp._probe_tools(["node"])
    assert seen["timeout_sec"] == wp._NODE_PROBE_TIMEOUT_SEC == 3.0


def test_preflight_missing_node_stays_missing(tmp_path, monkeypatch):
    empty_bin = tmp_path / "empty"
    empty_bin.mkdir()
    monkeypatch.setenv("PATH", str(empty_bin))
    monkeypatch.setattr(wp, "resolve_bundled_node", lambda: None)
    monkeypatch.setattr(
        wp, "node_runtime_health",
        lambda path, timeout_sec=10: (_ for _ in ()).throw(AssertionError("no probe for a missing node")),
    )

    tools = wp._probe_tools(["node", "npm"])
    assert tools["node"] == {"available": False, "path": "", "status": "missing"}
    summary = wp.summarize_workspace_preflight({"workspace_root": str(tmp_path), "tools": tools})
    assert "node" in summary["tools"]["missing"]
    assert "broken" not in summary["tools"]


# ---------------------------------------------------------------------------
# world_profiler — honest node markers
# ---------------------------------------------------------------------------


def _profile(tmp_path, monkeypatch, *, path_bin, health_fn, bundled) -> str:
    monkeypatch.setenv("PATH", str(path_bin))
    monkeypatch.setattr(wprof, "bootstrap_process_path", lambda: [])
    monkeypatch.setattr(wprof, "get_system_memory", lambda: "1.0 GB")
    monkeypatch.setattr(wprof, "get_cpu_info", lambda: "test-cpu")
    monkeypatch.setattr(wprof, "node_runtime_health", health_fn)
    monkeypatch.setattr(wprof, "resolve_bundled_node", lambda: bundled)
    out = tmp_path / "WORLD.md"
    wprof.generate_world_profile(str(out))
    return out.read_text(encoding="utf-8")


def test_world_profile_marks_broken_node_with_bundled_fallback(tmp_path, monkeypatch):
    host_bin = tmp_path / "hostbin"
    _make_exec(host_bin, "node")
    bundled = str(tmp_path / "bundle" / "bin" / "node")

    def fake_health(path, timeout_sec=10):
        if str(path) == bundled:
            return NodeRuntimeHealth(status="healthy", version="24.16.0", path=str(path))
        return _broken_node_health(path)

    content = _profile(tmp_path, monkeypatch, path_bin=host_bin, health_fn=fake_health, bundled=bundled)
    assert "node (broken: signal:SIGKILL; bundled fallback available)" in content


def test_world_profile_marks_broken_node_without_bundle(tmp_path, monkeypatch):
    host_bin = tmp_path / "hostbin"
    _make_exec(host_bin, "node")
    content = _profile(
        tmp_path, monkeypatch, path_bin=host_bin,
        health_fn=lambda path, timeout_sec=10: _broken_node_health(path), bundled=None,
    )
    assert "node (broken: signal:SIGKILL)" in content


def test_world_profile_lists_bundled_fallback_when_node_missing(tmp_path, monkeypatch):
    empty_bin = tmp_path / "empty"
    empty_bin.mkdir()
    bundled = str(tmp_path / "bundle" / "bin" / "node")
    content = _profile(
        tmp_path, monkeypatch, path_bin=empty_bin,
        health_fn=lambda path, timeout_sec=10: NodeRuntimeHealth(
            status="healthy", version="24.16.0", path=str(path)
        ),
        bundled=bundled,
    )
    assert "node (bundled fallback available)" in content


def test_world_profile_healthy_node_listed_plainly(tmp_path, monkeypatch):
    host_bin = tmp_path / "hostbin"
    _make_exec(host_bin, "node")

    def fail_bundled():
        raise AssertionError("bundled must not be consulted for a healthy PATH node")

    monkeypatch.setenv("PATH", str(host_bin))
    monkeypatch.setattr(wprof, "bootstrap_process_path", lambda: [])
    monkeypatch.setattr(wprof, "get_system_memory", lambda: "1.0 GB")
    monkeypatch.setattr(wprof, "get_cpu_info", lambda: "test-cpu")
    monkeypatch.setattr(
        wprof, "node_runtime_health",
        lambda path, timeout_sec=10: NodeRuntimeHealth(status="healthy", version="25.8.1", path=str(path)),
    )
    monkeypatch.setattr(wprof, "resolve_bundled_node", fail_bundled)
    out = tmp_path / "WORLD.md"
    wprof.generate_world_profile(str(out))
    content = out.read_text(encoding="utf-8")
    assert "node" in content
    assert "node (broken" not in content
    assert "bundled fallback" not in content


# ---------------------------------------------------------------------------
# node_runtime_health memo mechanics (adversarial findings C-2/C-4)
# ---------------------------------------------------------------------------


def _real_health(monkeypatch, outcomes):
    """Feed a queue of (version, reason) outcomes to the REAL memoized helper."""
    calls = []

    def fake_probe(path, timeout_sec=10):
        calls.append((str(path), timeout_sec))
        return outcomes.pop(0)

    monkeypatch.setattr(node_runtime, "_probe_node_version_outcome", fake_probe)
    return calls


def test_memo_timeout_is_budget_scoped(tmp_path, monkeypatch):
    """C-2 + T15: a short-budget timeout must not poison longer-budget
    consumers (a bigger budget re-probes), while a same-or-smaller budget
    reuses the cached verdict instead of stalling for the full timeout on
    every skill-summary call."""
    node = _make_exec(tmp_path / "bin", "node")
    calls = _real_health(monkeypatch, [("", "timeout"), ("25.8.1", "")])
    first = node_runtime.node_runtime_health(str(node), timeout_sec=3)
    assert first.status == "broken" and first.reason == "timeout"
    assert first.probed_timeout == 3
    repeat = node_runtime.node_runtime_health(str(node), timeout_sec=3)
    assert repeat is first
    assert len(calls) == 1  # same budget: served from the memo, no re-stall
    second = node_runtime.node_runtime_health(str(node), timeout_sec=10)
    assert second.healthy and second.version == "25.8.1"
    assert len(calls) == 2  # larger budget re-probed the timeout verdict


def test_memo_broken_verdict_is_memoized(tmp_path, monkeypatch):
    node = _make_exec(tmp_path / "bin", "node")
    calls = _real_health(monkeypatch, [("", "signal:SIGKILL")])
    first = node_runtime.node_runtime_health(str(node))
    second = node_runtime.node_runtime_health(str(node))
    assert first.reason == second.reason == "signal:SIGKILL"
    assert len(calls) == 1  # served from the memo


def test_memo_invalidated_by_byte_change(tmp_path, monkeypatch):
    node = _make_exec(tmp_path / "bin", "node")
    calls = _real_health(monkeypatch, [("", "signal:SIGKILL"), ("24.16.0", "")])
    assert node_runtime.node_runtime_health(str(node)).status == "broken"
    node.write_text("#!/bin/sh\necho v24.16.0\n", encoding="utf-8")
    os.utime(node, ns=(1, 1))  # force a distinct mtime_ns even on coarse clocks
    assert node_runtime.node_runtime_health(str(node)).healthy
    assert len(calls) == 2


def test_memo_missing_is_never_memoized(tmp_path, monkeypatch):
    """D9: a runtime installed mid-session (brew reinstall) is noticed live."""
    node = tmp_path / "bin" / "node"
    calls = _real_health(monkeypatch, [("24.16.0", "")])
    assert node_runtime.node_runtime_health(str(node)).status == "missing"
    _make_exec(tmp_path / "bin", "node")
    assert node_runtime.node_runtime_health(str(node)).healthy
    assert len(calls) == 1  # only the post-install probe ran


def test_probe_empty_version_output_gets_a_named_reason(tmp_path, monkeypatch):
    node = _make_exec(tmp_path / "bin", "node")
    calls = _real_health(monkeypatch, [("", "empty_version_output")])
    health = node_runtime.node_runtime_health(str(node))
    assert health.status == "broken" and health.reason == "empty_version_output"
    assert calls  # probed, not short-circuited


# ---------------------------------------------------------------------------
# missing-node downgrades (adversarial finding C-5)
# ---------------------------------------------------------------------------


def test_preflight_missing_node_downgrades_found_npm_family(tmp_path, monkeypatch):
    """npm/npx start through `#!/usr/bin/env node`: with node ABSENT they are
    as dead as with node broken, however present their own files are."""
    npm_bin = tmp_path / "npmbin"
    _make_exec(npm_bin, "npm")
    monkeypatch.setenv("PATH", str(npm_bin))
    bundled = str(tmp_path / "bundle" / "bin" / "node")
    monkeypatch.setattr(wp, "resolve_bundled_node", lambda: bundled)
    monkeypatch.setattr(
        wp, "node_runtime_health",
        lambda path, timeout_sec=10: NodeRuntimeHealth(
            status="healthy", version="24.16.0", path=str(path)
        ),
    )

    tools = wp._probe_tools(["node", "npm"])
    assert tools["node"]["status"] == "missing"
    assert tools["node"]["bundled_fallback"] == bundled
    assert tools["npm"]["available"] is False
    assert tools["npm"]["status"] == "interpreter_broken"


def test_world_profile_gates_npm_on_a_usable_node(tmp_path, monkeypatch):
    host_bin = tmp_path / "hostbin"
    _make_exec(host_bin, "node")
    _make_exec(host_bin, "npm")
    content = _profile(
        tmp_path, monkeypatch, path_bin=host_bin,
        health_fn=lambda path, timeout_sec=10: _broken_node_health(path), bundled=None,
    )
    assert "npm (needs a working node)" in content
    assert "node (broken: signal:SIGKILL)" in content


def test_node_runtime_imports_clean_in_both_orders():
    """T1 pin (grok-C CRITICAL class): a fresh interpreter must be able to
    import ouroboros.node_runtime BEFORE ouroboros.platform_layer and vice
    versa — the platform_layer re-export is lazy (PEP 562), so no order can
    hit a partially-initialized module."""
    import subprocess
    import sys

    for first, second in (
        ("ouroboros.node_runtime", "ouroboros.platform_layer"),
        ("ouroboros.platform_layer", "ouroboros.node_runtime"),
    ):
        code = (
            f"import {first}; import {second}; "
            "from ouroboros.platform_layer import node_runtime_health, select_skill_node_runtime; "
            "print('ok')"
        )
        proc = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, timeout=60
        )
        assert proc.returncode == 0 and proc.stdout.strip() == "ok", (
            first, proc.stderr[-500:],
        )


def test_companion_manifest_path_merges_case_aware(monkeypatch):
    """D2-8 pin: a companion manifest that declares "Path" must REPLACE the
    allowlisted "PATH" through the shared case-aware merge on Windows — never
    produce duplicate case-variant keys in the descriptor env."""
    from ouroboros import workspace_executor as wx

    monkeypatch.setattr(wx, "IS_WINDOWS", True)
    base = {"PATH": "C:/allowlisted", "HOME": "/h"}
    merged = wx.overlay_env(base, {"Path": "C:/manifest"})
    assert merged["Path"] == "C:/manifest"
    assert "PATH" not in merged
    assert merged["HOME"] == "/h"


def test_skill_path_rollback_rejects_relative_which(monkeypatch):
    """F-2 pin: the skill-family PATH rollback never trusts a RELATIVE which
    result — health probed against the server cwd proves nothing about the
    skill cwd, so the verdict is missing (same contract as the generic T10)."""
    monkeypatch.setattr(node_runtime.shutil, "which", lambda tok: "bin/node")
    monkeypatch.setattr(node_runtime._platform, "bootstrap_process_path", lambda: [])
    health = node_runtime._path_node_runtime_health()
    assert health.status == "missing"
    assert health.reason == "relative_path_entry_unprovable"


# ---------------------------------------------------------------------------
# The reviewed pin's Node for host consumers and Playwright's driver
# ---------------------------------------------------------------------------


PLANNED = "/data/state/cx/node/24.22.3-darwin-arm64.g-ff6e17d4874b-00000000/node-standalone/bin/node"


def _pinned_manager(monkeypatch, *, pinned="/managed/node-24.22.3", version="24.22.3", pin_error="", planned=PLANNED):
    from types import SimpleNamespace

    from ouroboros import claudexor_runtime

    calls = []

    def reviewed_node(allow_install=False, prepared=False):
        calls.append("prepared" if prepared else allow_install)
        if pin_error:
            raise claudexor_runtime.ClaudexorRuntimeError("runtime_pin_invalid", pin_error)
        if not pinned:
            missing = claudexor_runtime.ClaudexorRuntimeError("runtime_node_unavailable", "Node 24.22.3 is not prepared yet")
            missing.planned_node = planned
            raise missing
        return pinned

    manager = SimpleNamespace(pin=SimpleNamespace(node_version=version) if version else None,
                              pin_error=pin_error, reviewed_node=reviewed_node, planned_node=lambda _pin: planned)
    monkeypatch.setattr(claudexor_runtime, "get_runtime_manager", lambda: manager)
    return calls


@pytest.mark.parametrize("bundled_version,pinned,expected", [
    ("24.22.3", "/managed/node", "bundle"),       # a current bundle is used as before
    ("24.16.0", "/managed/node", "/managed/node"),  # an older kept bundle yields the pinned copy
    ("24.16.0", "", None),                         # ... or nothing: never the older shipped Node
    ("", "/managed/node", "/managed/node"),        # a broken first copy does not hide a prepared pin
    ("", "", "bundle"),                            # ... and stays visible to its callers when nothing qualifies
])
def test_resolve_bundled_node_never_returns_a_shipped_node_older_than_the_pin(
        tmp_path, monkeypatch, bundled_version, pinned, expected):
    bundled = _make_exec(tmp_path / "bundle" / "bin", "node")
    managed = _make_exec(tmp_path / "managed" / "bin", "node")
    monkeypatch.setattr(platform_layer, "_resolve_bundled_payload", lambda candidates: str(bundled))
    probed = []

    def probe(path, timeout_sec=10):
        probed.append(str(path))
        return ("24.22.3", "") if str(path) == str(managed) else (
            bundled_version, "" if bundled_version else "signal:SIGKILL")

    monkeypatch.setattr(node_runtime, "_probe_node_version_outcome", probe)
    calls = _pinned_manager(monkeypatch, pinned=str(managed) if pinned else "")
    answer = {"bundle": str(bundled), "/managed/node": str(managed), None: None}[expected]
    assert [platform_layer.resolve_bundled_node() for _ in range(3)] == [answer] * 3
    consulted = bundled_version != "24.22.3"
    assert calls == (["prepared"] * 3 if consulted else [])  # looked up by its record, never the install
    # R4 (Opus, Fable): every skill and node-family command of such an installation re-ran ``node --version``. Each
    # file runs once through the health memo; an unprepared pin runs nothing but the shipped copy's own probe.
    assert probed == [str(bundled)] + ([str(managed)] if consulted and pinned else [])


def test_a_managed_copy_that_does_not_run_as_the_pin_costs_the_probed_lookup_of_the_other_generations(
        tmp_path, monkeypatch):
    bundled = _make_exec(tmp_path / "bundle" / "bin", "node")
    damaged = _make_exec(tmp_path / "managed" / "bin", "node")
    monkeypatch.setattr(platform_layer, "_resolve_bundled_payload", lambda candidates: str(bundled))
    _patch_probe(monkeypatch, {bundled: ("24.16.0", ""), damaged: ("", "signal:SIGKILL")})
    calls = _pinned_manager(monkeypatch, pinned=str(damaged))
    assert platform_layer.resolve_bundled_node() == str(damaged)  # the stub's probed lookup answers for the others
    assert calls == ["prepared", False]  # never the install (real generations: test_claudexor_node_generations)


@pytest.mark.parametrize("first", ["absent", "older"])
def test_a_prepared_pin_or_an_exact_second_bundle_wins_over_path_whatever_the_first_bundle(
        tmp_path, monkeypatch, first):
    """Astra/Fable R5: with no first bundle the lookup returned None before asking for the prepared pin, and a first
    older bundle hid an exact one at a later base; skills, ``node --check`` and the web lane then ran an older PATH
    Node. Now the pin's Node is found wherever it is: prepared generation, else an exact copy at any base."""
    from ouroboros import preflight_node

    first_base, second_base = tmp_path / "app", tmp_path / "checkout"
    if first == "older":  # each copy where this OS's bundle keeps it (`node-standalone/node.exe` on Windows)
        _make_exec(platform_layer.embedded_node_candidates(first_base)[0].parent, "node")
    exact = _make_exec(platform_layer.embedded_node_candidates(second_base)[0].parent, "node")
    path_bin = tmp_path / "hostbin"
    _make_exec(path_bin, "node")
    _isolate_path(monkeypatch, path_bin)
    monkeypatch.setattr(platform_layer, "bundled_resource_bases", lambda: [first_base, second_base])
    _patch_probe(monkeypatch, {first_base: ("24.16.0", ""), second_base: ("24.22.3", ""), path_bin: ("22.11.0", "")})
    calls = _pinned_manager(monkeypatch, pinned="")
    for _ in range(3):  # one probe per file per process
        assert platform_layer.resolve_bundled_node() == str(exact)
        assert node_runtime.select_skill_node_runtime() == (str(exact), "bundled")
        assert preflight_node.resolve_node() == str(exact)
    assert all(call == "prepared" for call in calls)  # a lookup, never an install
    prepared = _make_exec(tmp_path / "managed" / "bin", "node")
    _patch_probe(monkeypatch, {first_base: ("24.16.0", ""), second_base: ("24.22.3", ""), prepared: ("24.22.3", "")})
    _pinned_manager(monkeypatch, pinned=str(prepared))
    # The first shipped copy serves as found when it runs as the pin; otherwise the prepared pin comes before a later
    # base's exact copy.
    expected = str(exact) if first == "absent" else str(prepared)
    assert platform_layer.resolve_bundled_node() == expected and preflight_node.resolve_node() == expected


def test_an_unreadable_pin_qualifies_no_shipped_node_but_no_pin_at_all_keeps_it(tmp_path, monkeypatch, caplog):
    bundled = _make_exec(tmp_path / "bundle" / "bin", "node")
    _patch_probe(monkeypatch, {bundled: ("24.16.0", "")})
    _pinned_manager(monkeypatch, version="", pin_error="runtime_pin_invalid: pin root must contain exactly ...")
    assert node_runtime.reviewed_host_node(str(bundled)) is None
    assert "reviewed Node pin is unreadable" in caplog.text
    _pinned_manager(monkeypatch, version="")  # ``release: null``: nothing reviewed to compare against
    assert node_runtime.reviewed_host_node(str(bundled)) == str(bundled)
    assert node_runtime.reviewed_host_node(None) is None  # source checkouts keep their PATH Node


def test_a_real_corrupt_pin_reaches_every_new_selection_path(tmp_path, monkeypatch):
    from ouroboros import claudexor_runtime

    bad = tmp_path / "pin.json"
    bad.write_text("{not json", encoding="utf-8")
    real_load = claudexor_runtime.load_runtime_pin
    monkeypatch.setattr(claudexor_runtime, "load_runtime_pin", lambda path=None, **kw: real_load(bad, **kw))
    manager = claudexor_runtime.ClaudexorRuntimeManager()
    assert manager.pin is None and manager.pin_error.startswith("runtime_pin_invalid")
    monkeypatch.setattr(claudexor_runtime, "get_runtime_manager", lambda: manager)
    bundled = _make_exec(tmp_path / "bundle" / "bin", "node")
    _patch_probe(monkeypatch, {bundled: ("24.16.0", "")})
    assert node_runtime.reviewed_host_node(str(bundled)) is None
    monkeypatch.delenv("PLAYWRIGHT_NODEJS_PATH", raising=False)
    with pytest.raises(claudexor_runtime.ClaudexorRuntimeError, match="runtime_pin_invalid"):
        node_runtime.select_playwright_node({}, allow_install=True)  # the browser tool refuses, names the pin
    child = {"PATH": "/usr/bin"}
    reason = node_runtime.forward_playwright_node(child)
    assert child == {"PATH": "/usr/bin"} and "runtime_pin_invalid" in reason  # no shipped Node, honest reason
    child = {"PLAYWRIGHT_NODEJS_PATH": "/operator/node"}
    assert node_runtime.forward_playwright_node(child) == ""  # an explicit choice keeps its authority
    assert child == {"PLAYWRIGHT_NODEJS_PATH": "/operator/node"}


@pytest.mark.parametrize("shipped", ["older", "none"])
def test_the_web_lane_may_name_a_path_node_while_skills_wait_for_the_pin(tmp_path, monkeypatch, shipped):
    """Owner 14 Nov 2026 (quiz 75067ff4): Ouroboros's functions require the new Node, ordinary JS tests may use the
    developer's own. Unprepared pin, an older shipped copy or none, a healthy PATH Node: the web lane runs PATH (and
    names it), a skill waits and its reason names the pin, the path its preparation publishes and the refused copy."""
    from ouroboros import preflight_node

    bundled = _make_exec(tmp_path / "bundle" / "bin", "node")
    path_bin = tmp_path / "hostbin"
    path_node = _make_exec(path_bin, "node")
    _isolate_path(monkeypatch, path_bin)
    monkeypatch.setattr(platform_layer, "_resolve_bundled_payload",
                        lambda candidates: str(bundled) if shipped == "older" else None)
    _patch_probe(monkeypatch, {bundled: ("24.16.0", ""), path_bin: ("25.8.1", "")})
    _pinned_manager(monkeypatch, pinned="")
    # PATH's Node as `which` spells it (`node.EXE` on Windows, from PATHEXT): the same file, compared as the OS does.
    assert os.path.normcase(preflight_node.resolve_node()) == os.path.normcase(str(path_node))
    selected, reason = node_runtime.select_skill_node_runtime()
    assert selected == "" and "pinned:24.22.3:not_prepared" in reason and PLANNED in reason
    assert ("refused shipped " + str(bundled) + " v24.16.0" in reason) == (shipped == "older")
    assert "path:" not in reason  # PATH is never consulted for a skill under the pin
    skill = {"runtime": "node"}
    with pytest.raises(node_runtime.NodeRuntimeUnavailable, match="pinned:24.22.3"):
        node_runtime.skill_node_argv(skill, "node", ["node", "main.js"])  # never a bare `node` from PATH
    assert node_runtime.skill_node_argv({"env": {"PATH": "/mine"}}, "node", ["node", "x"]) == ["node", "x"]
    _pinned_manager(monkeypatch, pinned="", planned="")  # a host the pin has no archive for keeps its PATH Node
    selected, provenance = node_runtime.select_skill_node_runtime()
    assert (os.path.normcase(selected), provenance) == (os.path.normcase(str(path_node)), "path")


def test_playwright_driver_keeps_an_explicit_node_else_selects_the_pinned_one(monkeypatch):
    from ouroboros import claudexor_runtime

    calls = _pinned_manager(monkeypatch)
    env = {"PLAYWRIGHT_NODEJS_PATH": "/operator/node"}
    assert node_runtime.select_playwright_node(env, allow_install=True) == "/operator/node" and calls == []
    env = {}
    assert node_runtime.select_playwright_node(env, allow_install=True) == "/managed/node-24.22.3"
    assert env == {"PLAYWRIGHT_NODEJS_PATH": "/managed/node-24.22.3",
                   "OUROBOROS_PLAYWRIGHT_NODE_SELECTED": "/managed/node-24.22.3"} and calls == [True]
    _pinned_manager(monkeypatch, pinned="")
    env = {}
    with pytest.raises(claudexor_runtime.ClaudexorRuntimeError):
        node_runtime.select_playwright_node(env)  # no silent fallback to the SDK's own Node
    assert env == {}
    _pinned_manager(monkeypatch, pin_error="unreadable")
    with pytest.raises(claudexor_runtime.ClaudexorRuntimeError):
        node_runtime.select_playwright_node(env)  # an unreadable pin qualifies no Node, the SDK's included
    assert env == {}


@pytest.mark.parametrize("pin", ["no-artifact", "no-pin"])
def test_a_host_without_a_pinned_node_archive_keeps_the_sdk_node_and_says_so(monkeypatch, caplog, pin):
    """Opus R2: Windows on Arm and 32-bit Windows have no pinned Node archive (``node_distribution_platform`` is
    ``""``) yet the pinned Chromium and Playwright wheels; refusing both engines there narrowed a supported platform.
    Like children, the browser keeps the SDK's own Node there and the log says it is not the reviewed pin."""
    calls = _pinned_manager(monkeypatch, pinned="", planned="", version="24.22.3" if pin == "no-artifact" else "")
    env = {"OUROBOROS_PLAYWRIGHT_NODE_SELECTED": "/stale", "PLAYWRIGHT_NODEJS_PATH": "/stale"}
    with caplog.at_level("WARNING", logger="ouroboros.node_runtime"):
        assert node_runtime.select_playwright_node(env, allow_install=True) == ""
    assert env == {} and calls == [True] and "no reviewed Node for this host" in caplog.text


def test_children_get_the_driver_node_before_any_browser_call(tmp_path, monkeypatch):
    from ouroboros.tools import skill_exec

    monkeypatch.delenv("PLAYWRIGHT_NODEJS_PATH", raising=False)
    pinned = _make_exec(tmp_path / "managed" / "bin", "node")
    calls = _pinned_manager(monkeypatch, pinned=str(pinned))
    env = skill_exec._scrub_env([], tmp_path / "state", "fixture-skill")
    assert env["PLAYWRIGHT_NODEJS_PATH"] == str(pinned) and calls == ["prepared"]  # no task context: lookup only
    child = {}
    node_runtime.forward_playwright_node(child)
    assert child["PLAYWRIGHT_NODEJS_PATH"] == str(pinned) and calls == ["prepared"] * 2  # the owner decides each time
    from ouroboros.workspace_executor import service_env, _executor_service_env

    for baseline in (service_env, _executor_service_env):
        assert baseline()["PLAYWRIGHT_NODEJS_PATH"] == str(pinned)
    assert calls == ["prepared"] * 4
    assert "PLAYWRIGHT_NODEJS_PATH" not in os.environ  # this process is not changed
    child = {"PLAYWRIGHT_NODEJS_PATH": "/skill/own"}
    node_runtime.forward_playwright_node(child)
    assert child == {"PLAYWRIGHT_NODEJS_PATH": "/skill/own"}
    monkeypatch.setenv("PLAYWRIGHT_NODEJS_PATH", "/operator/node")
    child = {}
    node_runtime.forward_playwright_node(child)
    assert child == {"PLAYWRIGHT_NODEJS_PATH": "/operator/node"}


@pytest.mark.parametrize("operator", ["", "/operator/node"])
def test_a_body_candidate_gets_the_serving_pinned_node_beside_an_operators_literal_driver(tmp_path, monkeypatch, operator):
    """Plan rev5: the candidate's isolated environment carries the serving installation's prepared pinned Node, found
    by record in its data root, as its own fact for the lookups inside (a nested run_tests' Node lane), never the
    serving store itself. Plan rev12: an operator's literal driver stays the driver's and is never marked automatic,
    even when it equals the carried executable."""
    from types import SimpleNamespace

    from ouroboros import body_candidate, claudexor_runtime, config
    from tests.body_candidate_support import isolate, make_ctx, make_serving

    serving, data = make_serving(tmp_path), isolate(monkeypatch, tmp_path)
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "pro")
    node, roots = "/serving/state/cx/node/24.22.3/bin/node", []
    monkeypatch.setattr(claudexor_runtime, "get_runtime_manager", lambda: SimpleNamespace(
        reviewed_node=lambda allow_install=False, prepared=False: node,
        prepared_node=lambda root: roots.append(root) or node))
    for name in ("PLAYWRIGHT_NODEJS_PATH", "OUROBOROS_PLAYWRIGHT_NODE_SELECTED", "OUROBOROS_PINNED_NODE_CARRIED"):
        monkeypatch.delenv(name, raising=False)
    if operator:
        monkeypatch.setenv("PLAYWRIGHT_NODEJS_PATH", operator)
    ctx = make_ctx(serving, data, "pinned-node")
    body_candidate.prepare(ctx)
    env = body_candidate.process_environment(ctx, ctx.repo_dir)
    assert env["OUROBOROS_DATA_DIR"] != str(config.DATA_DIR)  # still isolated: one executable path crossed
    driver = (env["PLAYWRIGHT_NODEJS_PATH"], env.get("OUROBOROS_PLAYWRIGHT_NODE_SELECTED", ""))
    assert driver == ((operator, "") if operator else (node, node)) and env["OUROBOROS_PINNED_NODE_CARRIED"] == node
    assert roots == [str(config.DATA_DIR)]  # looked up in the serving data root, by record


@pytest.mark.parametrize("explicit", ["/operator/node", "/operator/not-installed-mynode/node",
                                      "/data/state/cx/node/not-installed-24.22.3/node"])
def test_any_explicit_driver_node_is_kept_verbatim_and_nothing_is_prepared_for_it(monkeypatch, explicit):
    calls = _pinned_manager(monkeypatch)
    env = {"PLAYWRIGHT_NODEJS_PATH": explicit}
    assert node_runtime.select_playwright_node(env, allow_install=True) == explicit
    child = {"PLAYWRIGHT_NODEJS_PATH": explicit}
    assert node_runtime.forward_playwright_node(child) == ""
    monkeypatch.setenv("PLAYWRIGHT_NODEJS_PATH", explicit)
    inherited = {}
    node_runtime.forward_playwright_node(inherited)
    assert child == inherited == {"PLAYWRIGHT_NODEJS_PATH": explicit} and calls == []
    from ouroboros.workspace_executor import service_env, _executor_service_env

    for baseline in (service_env, _executor_service_env):
        assert baseline()["PLAYWRIGHT_NODEJS_PATH"] == explicit
    assert calls == []


def test_the_boot_step_selects_locally_and_prepares_off_the_serial_path(tmp_path, monkeypatch):
    """R2 (Opus, Fable): the step downloaded Node inline before skills/extensions/supervisor (12-17 s, up to the 300 s
    connect timeout offline). Now it only selects, on disk; the preparation is a daemon thread's, none under
    OUROBOROS_PYTEST_ACTIVE=1 (Fable: six subprocess harnesses downloaded 188 MB per boot)."""
    import inspect
    import threading

    import server
    from ouroboros import server_boot
    from ouroboros.server_liveness import BootDocument, run_boot_steps

    monkeypatch.delenv("PLAYWRIGHT_NODEJS_PATH", raising=False)
    monkeypatch.delenv("OUROBOROS_PLAYWRIGHT_NODE_SELECTED", raising=False)
    assert os.environ["OUROBOROS_PYTEST_ACTIVE"] == "1"
    monkeypatch.setattr("ouroboros.claudexor_node_seed.bundled_node", lambda version: "")  # no exact bundle here
    calls = _pinned_manager(monkeypatch)
    assert server_boot.prepare_pinned_node(tmp_path) is None  # prepared: selected, nothing to prepare
    assert os.environ["PLAYWRIGHT_NODEJS_PATH"] == os.environ["OUROBOROS_PLAYWRIGHT_NODE_SELECTED"] == \
        "/managed/node-24.22.3" and calls == ["prepared"]
    calls = _pinned_manager(monkeypatch, pinned="")
    finished = []
    monkeypatch.setattr(server_boot, "finish_pinned_node", lambda root: finished.append(root))
    document = BootDocument()
    run_boot_steps([("pinned_node", lambda: server_boot.prepare_pinned_node(tmp_path)),
                    ("skills", lambda: calls.append("next"))], stop=threading.Event(), document=document)
    assert calls == ["prepared", "next"] and finished == []  # under pytest: a lookup only, no preparation thread
    assert document._steps["pinned_node"]["state"] == "done" and os.environ["PLAYWRIGHT_NODEJS_PATH"] == PLANNED
    blocked = threading.Event()  # a download that has not finished, or a blackholed nodejs.org
    monkeypatch.setattr(server_boot, "finish_pinned_node", lambda root: blocked.wait(10) and finished.append(root))
    with monkeypatch.context() as outside_tests:
        outside_tests.setenv("OUROBOROS_PYTEST_ACTIVE", "0")  # only for this call; the recorder does no I/O
        thread = server_boot.prepare_pinned_node(tmp_path)
    assert thread.is_alive() and finished == []  # the step returned; skills, extensions and the supervisor go on
    blocked.set()
    thread.join(5)
    assert thread.daemon and thread.name == "pinned-node" and finished == [tmp_path]
    # R4 (Fable): an operator's explicit driver made the step prepare nothing, so skills' node, ``node --check`` and
    # the gate's Node lane stayed without the pin's Node. The driver's value is the operator's; the preparation is not.
    monkeypatch.setenv("PLAYWRIGHT_NODEJS_PATH", "/operator/node")
    monkeypatch.delenv("OUROBOROS_PLAYWRIGHT_NODE_SELECTED", raising=False)
    with monkeypatch.context() as outside_tests:
        outside_tests.setenv("OUROBOROS_PYTEST_ACTIVE", "0")
        thread = server_boot.prepare_pinned_node(tmp_path / "explicit")
    thread.join(5)
    assert finished == [tmp_path, tmp_path / "explicit"] and calls[-1] == "prepared"
    assert os.environ["PLAYWRIGHT_NODEJS_PATH"] == "/operator/node"
    assert "OUROBOROS_PLAYWRIGHT_NODE_SELECTED" not in os.environ  # never rewritten, never marked automatic
    calls = _pinned_manager(monkeypatch)
    with monkeypatch.context() as outside_tests:
        outside_tests.setenv("OUROBOROS_PYTEST_ACTIVE", "0")
        assert server_boot.prepare_pinned_node(tmp_path / "explicit") is None  # prepared: nothing to do
    assert calls == ["prepared"] and os.environ["PLAYWRIGHT_NODEJS_PATH"] == "/operator/node"
    for unpreparable in ({"pinned": "", "planned": ""}, {"pin_error": "unreadable"}):  # no archive for this host; no pin
        _pinned_manager(monkeypatch, **unpreparable)
        with monkeypatch.context() as outside_tests:
            outside_tests.setenv("OUROBOROS_PYTEST_ACTIVE", "0")
            assert server_boot.prepare_pinned_node(tmp_path / "explicit") is None
    lifespan = inspect.getsource(server.lifespan)
    assert (lifespan.index('("usage_store", _usage_store)') < lifespan.index('("pinned_node", lambda: live and '
            'server_boot.prepare_pinned_node(lifespan_drive_root))') < lifespan.index('("skills", '))


def test_the_boot_preparation_publishes_then_names_the_node_and_asks_enabled_extensions_not_live_again(
        tmp_path, monkeypatch):
    """Failed, or still loading when this reads (a real server.py showed one recording its failure after the read and
    staying failed): every enabled extension not live is asked once; a live or disabled one never is."""
    from types import SimpleNamespace

    from ouroboros import extension_reconcile_queue, server_boot, skill_loader
    from ouroboros.extension_registry_state import _extensions, _lock

    monkeypatch.setenv("PLAYWRIGHT_NODEJS_PATH", PLANNED)
    monkeypatch.setenv("OUROBOROS_PLAYWRIGHT_NODE_SELECTED", PLANNED)
    calls = _pinned_manager(monkeypatch)
    extension = SimpleNamespace(is_extension=lambda: True)
    monkeypatch.setattr(skill_loader, "discover_skills", lambda root, repo_path=None: [
        SimpleNamespace(name=name, enabled=enabled, manifest=manifest) for name, enabled, manifest in (
            ("failed-or-loading", True, extension), ("live", True, extension), ("owner-off", False, extension),
            ("instruction-skill", True, SimpleNamespace(is_extension=lambda: False)))])
    with _lock:
        saved = dict(_extensions)
        _extensions["live"] = object()
    try:
        assert server_boot.finish_pinned_node(tmp_path) == "/managed/node-24.22.3"
    finally:
        with _lock:
            _extensions.clear()
            _extensions.update(saved)
    assert calls == [True, False] and os.environ["PLAYWRIGHT_NODEJS_PATH"] == "/managed/node-24.22.3"
    asked = extension_reconcile_queue.list_extension_reconcile_requests(tmp_path)
    assert [(row["skill"], row["source"]) for row in asked] == [("failed-or-loading", "server")]
    calls = _pinned_manager(monkeypatch, pinned="")
    assert server_boot.finish_pinned_node(tmp_path / "offline") == ""  # logged; the next call or boot tries again
    assert not extension_reconcile_queue.list_extension_reconcile_requests(tmp_path / "offline")


def test_an_unprepared_node_reaches_children_as_its_planned_destination_and_this_process_is_untouched(
        tmp_path, monkeypatch):
    from types import SimpleNamespace

    from ouroboros import extension_companion
    from ouroboros.tools import skill_exec
    from ouroboros.tools.shell_process import _shell_env_for_cwd

    monkeypatch.delenv("PLAYWRIGHT_NODEJS_PATH", raising=False)
    calls = _pinned_manager(monkeypatch, pinned="")
    skill = skill_exec._scrub_env([], tmp_path / "state", "fixture-skill")
    shell = _shell_env_for_cwd(SimpleNamespace(repo_dir=str(tmp_path), task_metadata={}), tmp_path)
    companion = extension_companion._companion_base_env()
    from ouroboros.workspace_executor import service_env

    assert service_env()["PLAYWRIGHT_NODEJS_PATH"] == PLANNED
    assert skill["PLAYWRIGHT_NODEJS_PATH"] == shell["PLAYWRIGHT_NODEJS_PATH"] == PLANNED
    assert companion["PLAYWRIGHT_NODEJS_PATH"] == PLANNED and set(calls) == {"prepared"}  # lookups only, no install
    assert "PLAYWRIGHT_NODEJS_PATH" not in os.environ  # never written here, so never mistaken for an override
    child = {"PATH": "/usr/bin"}
    reason = node_runtime.forward_playwright_node(child)
    assert child == {"PATH": "/usr/bin", "PLAYWRIGHT_NODEJS_PATH": PLANNED,
                     "OUROBOROS_PLAYWRIGHT_NODE_SELECTED": PLANNED}  # generic PATH untouched
    assert PLANNED in reason and "PLAYWRIGHT_NODEJS_PATH" in reason
    calls = _pinned_manager(monkeypatch)  # prepared meanwhile: the next child reads the current fact
    child = {}
    node_runtime.forward_playwright_node(child)
    assert child == {"PLAYWRIGHT_NODEJS_PATH": "/managed/node-24.22.3",
                     "OUROBOROS_PLAYWRIGHT_NODE_SELECTED": "/managed/node-24.22.3"}


def test_a_stdio_mcp_server_gets_the_driver_node_unless_its_config_chose_one(monkeypatch):
    from ouroboros import mcp_client

    _pinned_manager(monkeypatch)
    captured = {}
    monkeypatch.setattr(mcp_client, "_stdio_with_diagnostics", lambda params, cfg, launch: captured.update(
        env=dict(params.env or {})))
    monkeypatch.setattr("ouroboros.execution_domain.attached_launch", lambda *a, **k: None)
    for env, expected in (({}, "/managed/node-24.22.3"), ({"PLAYWRIGHT_NODEJS_PATH": "/cfg/node"}, "/cfg/node")):
        cfg = mcp_client.MCPServerConfig(id="fixture", name="fixture", enabled=True, transport="stdio", url="",
                                         command="python3", args=[], auth_header="", auth_token="", allowed_tools=[],
                                         env=env)
        mcp_client._transport_factory(cfg)
        assert captured["env"]["PLAYWRIGHT_NODEJS_PATH"] == expected


def test_the_real_sdk_starts_its_driver_with_the_selected_node(monkeypatch):
    driver = pytest.importorskip("playwright._impl._driver")
    monkeypatch.setenv("PLAYWRIGHT_NODEJS_PATH", "/managed/node-24.22.3")
    executable, cli = driver.compute_driver_executable()
    assert executable == "/managed/node-24.22.3" and pathlib.Path(cli).name == "cli.js"


def test_this_process_driver_selection_is_never_taken_for_an_explicit_one(monkeypatch):
    """The SDK reads only this process's environment, so the browser tool writes its selection there. A read-only call
    that found the app's exact bundled copy neither freezes it for later calls nor hands it to children as a choice."""
    from types import SimpleNamespace

    from ouroboros import claudexor_runtime
    from ouroboros.workspace_executor import service_env

    bundle, generation = "/Applications/Ouroboros.app/Contents/Resources/node/bin/node", "/data/state/cx/node/g0/bin/node"
    published, calls = [], []

    def reviewed_node(allow_install=False, prepared=False):
        calls.append("prepared" if prepared else allow_install)
        if allow_install:
            published.append(generation)
        if published:
            return generation
        if not prepared:
            return bundle  # the read-only lookup also accepts the exact bundled copy
        missing = claudexor_runtime.ClaudexorRuntimeError("runtime_node_unavailable", "Node 24.22.3 is not prepared yet")
        missing.planned_node = PLANNED
        raise missing

    manager = SimpleNamespace(pin=SimpleNamespace(node_version="24.22.3"), pin_error="", reviewed_node=reviewed_node)
    monkeypatch.setattr(claudexor_runtime, "get_runtime_manager", lambda: manager)
    # Register both keys for undo even when absent: selection writes os.environ directly.
    monkeypatch.setenv("PLAYWRIGHT_NODEJS_PATH", "")
    monkeypatch.setenv("OUROBOROS_PLAYWRIGHT_NODE_SELECTED", "")
    assert node_runtime.select_playwright_node() == bundle == os.environ["PLAYWRIGHT_NODEJS_PATH"]
    child = dict(os.environ)  # copied from this process, so it carries the selection too
    node_runtime.forward_playwright_node(child)
    assert child["PLAYWRIGHT_NODEJS_PATH"] == service_env()["PLAYWRIGHT_NODEJS_PATH"] == PLANNED  # never the bundle
    assert node_runtime.select_playwright_node(allow_install=True) == generation == os.environ["PLAYWRIGHT_NODEJS_PATH"]
    child = dict(os.environ)
    node_runtime.forward_playwright_node(child)
    assert child["PLAYWRIGHT_NODEJS_PATH"] == generation and calls == [False, "prepared", "prepared", True, "prepared"]
    monkeypatch.setenv("PLAYWRIGHT_NODEJS_PATH", "/operator/node")  # the operator's own choice still wins everywhere
    child = {}
    node_runtime.forward_playwright_node(child)
    assert node_runtime.select_playwright_node(allow_install=True) == child["PLAYWRIGHT_NODEJS_PATH"] == "/operator/node"
    assert len(calls) == 5


def test_a_host_extension_child_gets_the_driver_node_beside_a_work_domain_and_a_domain_skill_does_not(
        tmp_path, monkeypatch):
    """Under an execution domain a script skill runs in the work domain (never a host path), while a trusted
    extension's out-of-process child runs on this host, as its companions do, and needs the pinned driver Node."""
    from ouroboros import extension_process_runner
    from ouroboros.tools import skill_exec

    monkeypatch.delenv("PLAYWRIGHT_NODEJS_PATH", raising=False)
    _pinned_manager(monkeypatch)
    monkeypatch.setattr(skill_exec, "current_execution_domain", lambda: object())  # a managed work domain
    assert "PLAYWRIGHT_NODEJS_PATH" not in skill_exec._scrub_env([], tmp_path / "state", "fixture-skill")
    child = extension_process_runner._child_env(drive_root=tmp_path / "drive", repo_dir=tmp_path, skill_name="ext",
                                                skill_dir=tmp_path / "ext", env_allowlist=[], granted_keys=[],
                                                host_access=False)
    assert child["PLAYWRIGHT_NODEJS_PATH"] == "/managed/node-24.22.3"


_SUCCESSOR = """
import json, sys
from types import SimpleNamespace
sys.path.insert(0, sys.argv[1])
from ouroboros import claudexor_runtime, node_runtime
manager = SimpleNamespace(pin=SimpleNamespace(node_version="24.24.0"), pin_error="",
                          reviewed_node=lambda allow_install=False, prepared=False: "/data/state/cx/node/g-new/bin/node")
claudexor_runtime.get_runtime_manager = lambda: manager
import os
child = dict(os.environ)
node_runtime.forward_playwright_node(child)
print(json.dumps({"browser": node_runtime.select_playwright_node(allow_install=True),
                  "child": child.get("PLAYWRIGHT_NODEJS_PATH")}))
"""


@pytest.mark.parametrize("state", ["selected", "planned", "operator", "operator-beside-carried"])
def test_a_reexec_successor_selects_its_own_pin_and_keeps_an_operator_choice_byte_exact(
        tmp_path, monkeypatch, state):
    """A browser call or the boot step writes the driver Node into this process's environment, and a direct restart,
    update or rollback re-execs with that environment. The successor (here: a new pin) must select anew for its
    browser and children, while an operator's own value stays exactly what the operator set.

    R4 (Opus): the automatic value itself crossed the exec. A body from before the marker (a rollback) took it for
    the operator's choice and started its driver from it, a planned path nothing may ever have prepared. Plan rev12:
    neither an automatic value nor its marker crosses; a boundary's carried executable does, as its own fact."""
    import json
    import logging
    import subprocess
    import sys

    from ouroboros import server_control

    old_generation, chosen = "/data/state/cx/node/g-old/bin/node", "/operator/my node/bin/node"
    operator = state.startswith("operator")
    _pinned_manager(monkeypatch, pinned="" if state == "planned" else old_generation)
    monkeypatch.setattr("ouroboros.claudexor_node_seed.bundled_node", lambda version: "")  # no exact bundle here
    monkeypatch.setenv("PLAYWRIGHT_NODEJS_PATH", chosen if operator else "")
    monkeypatch.setenv("OUROBOROS_PLAYWRIGHT_NODE_SELECTED", "")
    monkeypatch.setenv("OUROBOROS_PINNED_NODE_CARRIED", old_generation if state == "operator-beside-carried" else "")
    if state == "planned":
        assert node_runtime.select_local_playwright_node() and os.environ["PLAYWRIGHT_NODEJS_PATH"] == PLANNED
    else:
        assert node_runtime.select_playwright_node(allow_install=True) == (chosen if operator else old_generation)
    handed = {}
    monkeypatch.setattr(server_control, "IS_WINDOWS", False)
    monkeypatch.setattr(server_control.os, "execvpe", lambda path, argv, env: handed.update(env))
    server_control.restart_current_process("127.0.0.1", 8765, repo_dir=tmp_path, log=logging.getLogger(__name__))
    # What a body without the marker logic reads: the operator's value, or nothing (never a path selected for it).
    assert handed.get("PLAYWRIGHT_NODEJS_PATH") == (chosen if operator else None)
    assert not handed.get("OUROBOROS_PLAYWRIGHT_NODE_SELECTED")  # the successor selects anew
    assert (handed.get("OUROBOROS_PINNED_NODE_CARRIED") or None) == (
        old_generation if state == "operator-beside-carried" else None)
    assert os.environ["PLAYWRIGHT_NODEJS_PATH"] == (
        chosen if operator else PLANNED if state == "planned" else old_generation)  # this process keeps its own
    run = subprocess.run([sys.executable, "-c", _SUCCESSOR, str(pathlib.Path(node_runtime.__file__).parents[1])],
                         env=handed, capture_output=True, text=True, timeout=120)
    assert run.returncode == 0, run.stderr
    expected = chosen if operator else "/data/state/cx/node/g-new/bin/node"
    assert json.loads(run.stdout.strip().splitlines()[-1]) == {"browser": expected, "child": expected}


@pytest.mark.parametrize("builder", ["companion", "extension"])
def test_forwarded_selection_is_provenance_in_a_built_child_environment_and_refreshes_there(tmp_path, monkeypatch, builder):
    """At exec, an automatic driver path is provenance, not a new operator choice: the environment each builder hands
    a child carries the marker beside the value, so the child's own selection (its browser, its children) reads the
    pin current there. The process boundary itself is witnessed once, by the re-exec successor above."""
    from ouroboros import extension_companion, extension_process_runner

    monkeypatch.delenv("PLAYWRIGHT_NODEJS_PATH", raising=False)
    monkeypatch.delenv("OUROBOROS_PLAYWRIGHT_NODE_SELECTED", raising=False)
    old, new = "/data/state/cx/node/g-old/bin/node", "/data/state/cx/node/g-new/bin/node"
    _pinned_manager(monkeypatch, pinned=old)
    env = (extension_companion._companion_base_env() if builder == "companion" else
           extension_process_runner._child_env(drive_root=tmp_path / "drive", repo_dir=tmp_path,
                skill_name="fixture", skill_dir=tmp_path, env_allowlist=[], granted_keys=[]))
    assert env["PLAYWRIGHT_NODEJS_PATH"] == env["OUROBOROS_PLAYWRIGHT_NODE_SELECTED"] == old
    _pinned_manager(monkeypatch, pinned=new)  # the child process: its own pin
    for name in ("PLAYWRIGHT_NODEJS_PATH", "OUROBOROS_PLAYWRIGHT_NODE_SELECTED"):
        monkeypatch.setenv(name, env[name])  # what exec gives it
    grandchild = dict(env)
    node_runtime.forward_playwright_node(grandchild)
    assert node_runtime.select_playwright_node(allow_install=True) == grandchild["PLAYWRIGHT_NODEJS_PATH"] == new


def test_a_changing_selection_is_assigned_in_place_and_only_a_variable_without_a_value_is_removed(monkeypatch):
    """R4 (Opus): every write deleted both variables and added them again, also in ``os.environ``, which the boot's
    preparation thread and a worker's browser call write while other threads copy it to spawn children: a copy that
    listed the keys before the delete raised KeyError, and a Playwright start inside the gap ran the SDK's own Node."""
    removed = []

    class Watched(dict):
        def pop(self, key, *default):
            removed.append(key)
            return super().pop(key, *default)

        def __delitem__(self, key):
            removed.append(key)
            super().__delitem__(key)

    old, new = "/generation/old", "/generation/new"
    env = Watched({"PATH": "/usr/bin", "PLAYWRIGHT_NODEJS_PATH": old, "OUROBOROS_PLAYWRIGHT_NODE_SELECTED": old})
    _pinned_manager(monkeypatch, pinned=new)
    monkeypatch.delenv("PLAYWRIGHT_NODEJS_PATH", raising=False)
    assert node_runtime.forward_playwright_node(env) == "" and node_runtime.select_playwright_node(env) == new
    assert env == {"PATH": "/usr/bin", "PLAYWRIGHT_NODEJS_PATH": new, "OUROBOROS_PLAYWRIGHT_NODE_SELECTED": new}
    assert removed == []
    unset, real_unset = [], os.unsetenv  # the same through the real os.environ, which a spawning thread copies
    monkeypatch.setattr(os, "unsetenv", lambda key: unset.append(os.fsdecode(key)) or real_unset(key))
    monkeypatch.setenv("PLAYWRIGHT_NODEJS_PATH", old)
    monkeypatch.setenv("OUROBOROS_PLAYWRIGHT_NODE_SELECTED", old)
    assert node_runtime.select_playwright_node() == new and node_runtime.forward_playwright_node(os.environ) == ""
    assert os.environ["PLAYWRIGHT_NODEJS_PATH"] == os.environ["OUROBOROS_PLAYWRIGHT_NODE_SELECTED"] == new
    assert unset == []
    node_runtime.keep_explicit_driver_node(env, {"PLAYWRIGHT_NODEJS_PATH": new})  # a manifest's own, equal to ours
    assert env == {"PATH": "/usr/bin", "PLAYWRIGHT_NODEJS_PATH": new}
    assert removed == ["OUROBOROS_PLAYWRIGHT_NODE_SELECTED"]  # the one variable that has no value now
    monkeypatch.setattr(platform_layer, "IS_WINDOWS", True)
    env = Watched({"Playwright_Nodejs_Path": old, "ouroboros_playwright_node_selected": old, "PLAYWRIGHT_NODEJS_PATH": old})
    removed.clear()
    assert node_runtime.select_playwright_node(env) == new
    assert env == {"PLAYWRIGHT_NODEJS_PATH": new, "OUROBOROS_PLAYWRIGHT_NODE_SELECTED": new}
    assert removed == ["Playwright_Nodejs_Path", "ouroboros_playwright_node_selected"]  # true case duplicates only


def test_an_unprepared_node_is_warned_about_once_per_reason_then_at_debug(monkeypatch, caplog):
    """R4 (Opus, Fable): while the pin's Node is unprepared every child environment built (each shell command, skill
    run, service, MCP stdio spawn) logged the same WARNING."""
    monkeypatch.setattr(node_runtime, "_FORWARD_REASONS_LOGGED", set())
    for name in ("PLAYWRIGHT_NODEJS_PATH", "OUROBOROS_PLAYWRIGHT_NODE_SELECTED"):
        monkeypatch.delenv(name, raising=False)
    _pinned_manager(monkeypatch, pinned="")
    with caplog.at_level("DEBUG", logger="ouroboros.node_runtime"):
        reasons = [node_runtime.forward_playwright_node({}) for _ in range(3)]
        _pinned_manager(monkeypatch, pinned="", planned="/data/state/cx/node/another/node")  # another fact: said again
        reasons.append(node_runtime.forward_playwright_node({}))
    assert reasons[0] == reasons[1] == reasons[2] != reasons[3] and PLANNED in reasons[0]  # callers still get it
    assert [record.levelname for record in caplog.records if "no reviewed Node" in record.getMessage()] == [
        "WARNING", "DEBUG", "DEBUG", "WARNING"]


@pytest.mark.parametrize("windows", [False, True], ids=["posix", "windows-semantics"])
def test_driver_selection_honors_platform_env_case_and_refreshes_reused_env(monkeypatch, windows):
    monkeypatch.setattr(platform_layer, "IS_WINDOWS", windows)
    monkeypatch.delenv("PLAYWRIGHT_NODEJS_PATH", raising=False)
    env = {"Playwright_Nodejs_Path": "/operator/Node With Spaces"}
    calls = _pinned_manager(monkeypatch, pinned="/generation/old")
    node_runtime.forward_playwright_node(env)
    assert env["PLAYWRIGHT_NODEJS_PATH"] == ("/operator/Node With Spaces" if windows else "/generation/old")
    assert bool(calls) is not windows
    if windows:
        assert "Playwright_Nodejs_Path" not in env
    else:
        assert env["Playwright_Nodejs_Path"] == "/operator/Node With Spaces"
    _pinned_manager(monkeypatch, pinned="/generation/new")
    node_runtime.forward_playwright_node(env)
    assert env["PLAYWRIGHT_NODEJS_PATH"] == ("/operator/Node With Spaces" if windows else "/generation/new")


def test_companion_popen_overlay_is_case_aware_on_windows(tmp_path, monkeypatch):
    """Exercise the final spawn env, with simulated Windows semantics, not native CreateProcess."""
    from ouroboros import extension_companion as companion, workspace_executor

    monkeypatch.setattr(platform_layer, "IS_WINDOWS", True)
    monkeypatch.setattr(workspace_executor, "IS_WINDOWS", True)
    monkeypatch.setattr(companion, "is_server_process", lambda: True)
    monkeypatch.setenv("PATH", "C:/host")
    monkeypatch.setenv("PLAYWRIGHT_NODEJS_PATH", "C:/host/node")
    seen = {}

    def capture(*args, **kwargs):
        seen.update(kwargs["env"])
        raise OSError("witness stops before native spawn")

    monkeypatch.setattr(companion.subprocess, "Popen", capture)
    descriptor = companion.CompanionDescriptor("fixture", "driver", ["fixture"], tmp_path,
                    {"Path": "C:/manifest", "Playwright_Nodejs_Path": "C:/operator/my node"})
    with pytest.raises(OSError, match="witness stops"):
        companion.CompanionSupervisor(tmp_path).start(descriptor)
    assert seen["Path"] == "C:/manifest" and "PATH" not in seen
    assert seen["PLAYWRIGHT_NODEJS_PATH"] == "C:/operator/my node"
    assert "Playwright_Nodejs_Path" not in seen
    assert node_runtime.SELECTED_DRIVER_NODE_ENV not in seen


@pytest.mark.parametrize("authoritative", [True, False])
@pytest.mark.parametrize("inherited", ["automatic", "operator"])
def test_a_benchmark_server_never_reads_an_automatic_driver_as_an_operator_choice(
        tmp_path, monkeypatch, inherited, authoritative):
    """Fable R5: the settings-authoritative benchmark env stripped the whole OUROBOROS_ namespace, the automatic
    selection's marker included, and kept PLAYWRIGHT_NODEJS_PATH, so its isolated server took the parent's automatic
    (possibly only planned) path for an operator's literal and never selected anew. Carbon R7: the default mode (CL-Bench
    host runs, e2e_live, keyless servers) handed the automatic pair on, which a clone from before the marker reads as
    the operator's; now the shared ``without_selected_driver_node`` drops it in both modes. A literal stays byte-exact."""
    import json

    from tests.test_cybergym_server import _host, _RootlessIsolatedServer, _seed_repo

    seed, _commit = _seed_repo(tmp_path)
    settings = tmp_path / "settings.json"
    settings.write_text(json.dumps({"OUROBOROS_RUNTIME_MODE": "pro"}), encoding="utf-8")
    value = "/operator/my node/bin/node" if inherited == "operator" else PLANNED
    monkeypatch.setenv("PLAYWRIGHT_NODEJS_PATH", value)
    monkeypatch.setenv("OUROBOROS_PLAYWRIGHT_NODE_SELECTED", PLANNED)  # this parent's own automatic selection
    monkeypatch.setenv("OUROBOROS_PINNED_NODE_CARRIED", PLANNED)
    env = _RootlessIsolatedServer(seed, tmp_path / "data", settings, docker_host=_host(), provider_key="k",
                                  settings_authoritative_env=authoritative)._env()
    assert env.get("PLAYWRIGHT_NODEJS_PATH") == (value if inherited == "operator" else None)
    if authoritative:  # the namespace goes whole
        assert "OUROBOROS_PLAYWRIGHT_NODE_SELECTED" not in env and "OUROBOROS_PINNED_NODE_CARRIED" not in env
    else:  # the carried executable stays for the server's lookups; a marker survives only beside a literal it is not
        assert env.get("OUROBOROS_PINNED_NODE_CARRIED") == PLANNED
        assert env.get("OUROBOROS_PLAYWRIGHT_NODE_SELECTED") == (PLANNED if inherited == "operator" else None)
        assert node_runtime._explicit_driver_node(env) == (value if inherited == "operator" else "")


def test_an_attach_login_waiting_for_the_pinned_node_asks_for_it_once_and_answers_retry(monkeypatch):
    """R5 (Opus): ``retry_setup_login`` kept answering 503 while nothing prepared the pin's Node after a failed boot.
    The login that needs it now asks once (single-flight, no timer); the retry finds the prepared Node."""
    from types import SimpleNamespace

    from ouroboros import claudexor_daemon, claudexor_runtime, server_boot
    from ouroboros.gateways.claudexor import ClaudexorUnavailable

    def unprepared(**_kwargs):
        raise claudexor_runtime.ClaudexorRuntimeError("runtime_node_unavailable", "Node 24.22.3 is not prepared yet")

    monkeypatch.setattr(claudexor_runtime, "get_runtime_manager",
                        lambda: SimpleNamespace(resolve_serving_role_command=unprepared))
    asked = []
    monkeypatch.setattr(server_boot, "request_pinned_node", lambda root=None: asked.append(root))
    with pytest.raises(ClaudexorUnavailable) as refused:
        claudexor_daemon.resolve_attach_login_argv({"version": "6.1.1", "sha": "x", "entry": "/e"})
    assert refused.value.status_code == 503 and refused.value.required_actions == ("retry_setup_login",)
    assert asked == [None]


# ---------------------------------------------------------------------------
# An npm child of the skill family runs on the reviewed Node (``reviewed_npm_command``)
# ---------------------------------------------------------------------------


def _npm_cli(root: pathlib.Path) -> pathlib.Path:
    cli = root / "node_modules" / "npm" / "bin" / "npm-cli.js"
    cli.parent.mkdir(parents=True)
    cli.write_text("// npm\n", encoding="utf-8")
    return cli


def _governs(monkeypatch, selected: str, *, windows: bool = False):
    monkeypatch.setattr(node_runtime, "pinned_node_wait", lambda: (True, "/planned/node"))
    verdict = (selected, "bundled") if selected else ("", "pinned:24.22.3:not_prepared")
    monkeypatch.setattr(node_runtime, "select_skill_node_runtime", lambda timeout_sec=10: verdict)
    monkeypatch.setattr(node_runtime._platform, "IS_WINDOWS", windows)


@pytest.mark.parametrize("windows", [False, True])
def test_an_npm_launcher_runs_its_cli_on_the_reviewed_node_not_the_node_it_would_choose(tmp_path, monkeypatch, windows):
    """The official Windows ``npm.cmd`` (node-v24.22.3-win-x64.zip, sha256 21b46c69...) sets
    ``NODE_EXE=%~dp0\\node.exe`` and falls back to PATH's ``node`` only when that file is absent: an older Node install
    on PATH kept running npm although the child PATH started with the pin's Node. The launcher's own CLI now runs on the
    reviewed Node; a POSIX install links ``npm`` to the same ``npm-cli.js``."""
    pinned = _make_exec(tmp_path / "pinned" / "bin", "node")
    _governs(monkeypatch, str(pinned), windows=windows)
    host = tmp_path / "Program Files" / "nodejs"
    cli = _npm_cli(host)
    _make_exec(host, "node.exe")  # the older Node that npm.cmd would run
    launcher = _make_exec(host, "npm.cmd")
    assert node_runtime.reviewed_npm_command(str(launcher)) == [str(pinned), str(cli)]
    linked = tmp_path / "usr" / "local" / "bin" / "npm"
    linked.parent.mkdir(parents=True)
    linked.symlink_to(cli)
    assert node_runtime.reviewed_npm_command(str(linked)) == [str(pinned), str(cli.resolve())]


def _literal_exec(path: pathlib.Path) -> pathlib.Path:
    """A fake executable under exactly this name on any OS: a layout fixture whose names are the layout's own."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    path.chmod(0o755)
    return path


def test_the_pins_own_npm_beside_its_node_wins_over_the_launchers(tmp_path, monkeypatch):
    """A complete generation (``_NODE_META_SCHEMA_VERSION`` 2 records ``archive_npm_cli``) holds npm beside its Node, in
    the layout ``ClaudexorRuntimeManager._managed_npm_cli`` reads: that npm runs, whatever a host launcher is. The
    POSIX layout's Node is literally ``bin/node`` and the Windows one's ``node.exe``, on whichever OS the test runs."""
    posix_pinned = _literal_exec(tmp_path / "gen" / "node-standalone" / "bin" / "node")
    posix_cli = _npm_cli(tmp_path / "gen" / "node-standalone" / "lib")
    host = tmp_path / "host"
    _npm_cli(host)
    launcher = _make_exec(host, "npm.cmd")
    _governs(monkeypatch, str(posix_pinned))
    assert node_runtime.reviewed_npm_command(str(launcher)) == [str(posix_pinned), str(posix_cli)]
    windows_pinned = _literal_exec(tmp_path / "wgen" / "node-standalone" / "node.exe")
    windows_cli = _npm_cli(tmp_path / "wgen" / "node-standalone")
    _governs(monkeypatch, str(windows_pinned), windows=True)
    assert node_runtime.reviewed_npm_command("npm") == [str(windows_pinned), str(windows_cli)]  # no launcher needed


@pytest.mark.parametrize("complete", [False, True], ids=["node-only", "full"])
def test_a_locally_installed_npm_runs_its_own_cli_on_the_reviewed_node(tmp_path, monkeypatch, complete):
    """An install spec may install an npm version (``npm@11.19.1``) and check it with a bare ``npm --version``:
    ``augment_env_for_skill_deps`` puts ``node/node_modules/.bin`` first, so ``npm`` is the launcher npm generated
    there (a cmd-shim ``npm.cmd`` on Windows, a link on POSIX) whose CLI is ``../npm/bin/npm-cli.js``. A ready
    Node-only pin refused it on Windows before any spawn. It runs on the reviewed Node, before the pin's own npm too:
    the pin chooses the Node, not the npm version a skill installed."""
    pinned = _literal_exec(tmp_path / "gen" / "node-standalone" / "bin" / "node")
    if complete:
        _npm_cli(tmp_path / "gen" / "node-standalone" / "lib")
    local = tmp_path / ".ouroboros_env" / "node"
    cli = _npm_cli(local)
    bins = local / "node_modules" / ".bin"
    shim = _literal_exec(bins / "npm.cmd")  # cmd-shim runs %dp0%\..\npm\bin\npm-cli.js; no node.exe beside it
    for windows in (False, True):
        _governs(monkeypatch, str(pinned), windows=windows)
        assert node_runtime.reviewed_npm_command(str(shim)) == [str(pinned), str(cli)]
    if os.name != "nt":  # npm links the POSIX bin relatively
        (bins / "npm").symlink_to(pathlib.Path("..", "npm", "bin", "npm-cli.js"))
        _governs(monkeypatch, str(pinned))
        assert node_runtime.reviewed_npm_command(str(bins / "npm")) == [str(pinned), str(cli.resolve())]


def test_a_launcher_with_no_npm_cli_is_refused_on_any_os(tmp_path, monkeypatch):
    """A shim whose CLI is not found (Volta, Scoop, asdf, a script naming an absolute older Node) may choose its own
    Node, on POSIX as on Windows: it is refused, never run as if it were the reviewed Node, as a refusal preparing the
    pin's Node again cannot change. Only npm's own layouts are read: a ``.bin`` outside ``node_modules`` is not one."""
    pinned = _make_exec(tmp_path / "pinned" / "bin", "node")
    shim = _make_exec(tmp_path / "shims", "npm")
    _npm_cli(tmp_path / "tool")
    stray = _literal_exec(tmp_path / "tool" / ".bin" / "npm.cmd")
    for windows in (False, True):
        _governs(monkeypatch, str(pinned), windows=windows)
        for launcher in (str(shim), "npm", str(stray)):  # "npm": nothing resolved, never a relative CLI guess
            with pytest.raises(node_runtime.NpmLauncherRefused, match="would choose its own Node"):
                node_runtime.reviewed_npm_command(launcher)


def test_npm_keeps_its_launcher_without_a_pin_and_waits_for_an_unprepared_one(tmp_path, monkeypatch):
    host = tmp_path / "host"
    _npm_cli(host)
    launcher = _make_exec(host, "npm.cmd")
    monkeypatch.setattr(node_runtime, "pinned_node_wait", lambda: (False, ""))
    monkeypatch.setattr(node_runtime, "select_skill_node_runtime",
                        lambda timeout_sec=10: (_ for _ in ()).throw(AssertionError("no selection without a pin")))
    assert node_runtime.reviewed_npm_command(str(launcher)) == [str(launcher)]
    _governs(monkeypatch, "")
    with pytest.raises(node_runtime.NodeRuntimeUnavailable, match="not_prepared") as waiting:
        node_runtime.reviewed_npm_command(str(launcher))
    assert not isinstance(waiting.value, node_runtime.NpmLauncherRefused)


def test_a_skill_npm_child_is_rewritten_only_under_the_pin_and_never_past_a_manifest_path(tmp_path, monkeypatch):
    pinned = _make_exec(tmp_path / "pinned" / "bin", "node")
    host = tmp_path / "host"
    cli = _npm_cli(host)
    launcher = _make_exec(host, "npm.cmd")
    monkeypatch.setattr(node_runtime._platform, "bootstrap_process_path", lambda: [])
    monkeypatch.setattr(node_runtime.shutil, "which", lambda name, *a, **k: str(launcher) if name == "npm" else None)
    _governs(monkeypatch, str(pinned), windows=True)
    argv = ["npm", "run", "start"]
    assert node_runtime.skill_node_argv({}, "npm", argv) == [str(pinned), str(cli), "run", "start"]
    assert node_runtime.skill_node_argv({"env": {"Path": "C:\\tools"}}, "npm", argv) == argv  # the author's runtime
    monkeypatch.setattr(node_runtime, "pinned_node_wait", lambda: (False, ""))
    assert node_runtime.skill_node_argv({}, "npm", argv) == argv
