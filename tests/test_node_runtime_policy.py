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
    path_bin = tmp_path / "hostbin"
    path_node = _make_exec(path_bin, "node")
    _isolate_path(monkeypatch, path_bin)
    monkeypatch.setattr(node_runtime._platform, "resolve_bundled_node", lambda: None)
    _patch_probe(monkeypatch, {path_bin: ("25.8.1", "")})

    selected, provenance = node_runtime.select_skill_node_runtime()
    assert provenance == "path"
    assert pathlib.Path(selected) == path_node


def test_select_reports_both_probe_verdicts_when_nothing_usable(tmp_path, monkeypatch):
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
# skill_node_emergency_path_dir — emergency-only PATH prepend
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


def test_emergency_dir_empty_on_healthy_path_node(tmp_path, monkeypatch):
    bundle_bin = tmp_path / "bundle" / "bin"
    bundled = _make_exec(bundle_bin, "node")
    path_bin = tmp_path / "hostbin"
    _make_exec(path_bin, "node")
    _isolate_path(monkeypatch, path_bin)
    monkeypatch.setattr(node_runtime._platform, "resolve_bundled_node", lambda: str(bundled))
    _patch_probe(monkeypatch, {bundle_bin: ("24.16.0", ""), path_bin: ("25.8.1", "")})

    # Bundled still wins the direct selection, but there is NO emergency: the
    # child env stays byte-identical on a healthy PATH.
    assert node_runtime.skill_node_emergency_path_dir() == ""


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
