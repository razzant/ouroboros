"""Light-mode skill-payload authoring through root=skill_payload plus bucket/skill_name.

Split verbatim out of ``tests/test_runtime_mode_core.py`` by theme. This module owns
the buckets the short form may target, the native bucket refused at the gate, the
specific errors partial arguments must surface instead of a generic light block, the
stray bucket an external workspace ignores, and the control-plane sidecar that stays
blocked either way.
"""

from __future__ import annotations


import pytest

from ouroboros.tools.registry import ToolRegistry

from tests._runtime_mode_core_shared import _git_repo, _make_skill_payload, _registry


# ===========================================================================
# Part: light-mode bucket+skill_name short-form authoring (v5.16.0-rc.1)
# ===========================================================================
#
# Under runtime_mode=light, skill-payload edits use Tool API v2
# root=skill_payload plus bucket/skill_name. Legacy private aliases still
# route through the same policy for compatibility, but are not public schemas.


@pytest.mark.parametrize("bucket", ["external", "clawhub", "ouroboroshub"])
def test_light_write_file_with_skill_payload_root_allowed(bucket, tmp_path, monkeypatch):
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "light")
    _make_skill_payload(tmp_path, bucket, "alpha")
    reg = _registry(tmp_path)
    result = reg.execute(
        "write_file",
        {
            "root": "skill_payload",
            "path": "new.py",
            "content": "VALUE = 1\n",
            "bucket": bucket,
            "skill_name": "alpha",
        },
    )
    assert "LIGHT_MODE_BLOCKED" not in result, result[:200]
    assert (tmp_path / "skills" / bucket / "alpha" / "new.py").is_file()


@pytest.mark.parametrize("bucket", ["external", "clawhub", "ouroboroshub"])
def test_light_str_replace_editor_with_bucket_skill_name_allowed(bucket, tmp_path, monkeypatch):
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "light")
    _make_skill_payload(tmp_path, bucket, "beta")
    reg = _registry(tmp_path)
    result = reg.execute(
        "edit_text",
        {
            "root": "skill_payload",
            "path": "plugin.py",
            "old_str": "pass",
            "new_str": "return None",
            "bucket": bucket,
            "skill_name": "beta",
        },
    )
    assert "LIGHT_MODE_BLOCKED" not in result, result[:200]
    assert "Replaced" in result


def test_light_data_write_with_bucket_skill_name_resolves_under_payload(tmp_path, monkeypatch):
    """write_file with root=skill_payload resolves the short path under
    data/skills/<bucket>/<skill>/ so a file lands inside the payload."""
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "light")
    _make_skill_payload(tmp_path, "external", "gamma")
    reg = _registry(tmp_path)
    result = reg.execute(
        "write_file",
        {
            "root": "skill_payload",
            "path": "lib/utils.py",
            "content": "def hi(): return 'ok'\n",
            "bucket": "external",
            "skill_name": "gamma",
        },
    )
    assert "DATA_WRITE_ERROR" not in result, result[:200]
    assert "DATA_WRITE_BLOCKED" not in result, result[:200]
    landed = tmp_path / "skills" / "external" / "gamma" / "lib" / "utils.py"
    assert landed.is_file(), f"expected file at {landed}; got result={result[:200]}"


def test_light_bucket_native_rejected_at_gate(tmp_path, monkeypatch):
    """bucket=native MUST not be honoured — launcher seed update lane stays
    authoritative. With the post-triad partial-args check in place, the gate
    surfaces the specific SKILL_PAYLOAD_ARG_ERROR (which lists `native excluded`)
    BEFORE the generic LIGHT_MODE_BLOCKED would fire — giving the agent a
    clearer signal."""
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "light")
    reg = _registry(tmp_path)
    result = reg.execute(
        "write_file",
        {
            "root": "skill_payload",
            "path": "plugin.py",
            "content": "x",
            "bucket": "native",
            "skill_name": "anything",
        },
    )
    assert "SKILL_PAYLOAD_ARG_ERROR" in result, result[:200]
    assert "read/review only" in result
    assert "root=system_repo" in result


@pytest.mark.parametrize("tool_name,base_args", [
    ("write_file", {"path": "plugin.py", "content": "x"}),
    ("edit_text", {"path": "plugin.py", "old_str": "a", "new_str": "b"}),
    ("write_file", {"root": "skill_payload", "path": "plugin.py", "content": "x"}),
])
@pytest.mark.parametrize("partial", [
    {"bucket": "external"},
    {"skill_name": "alpha"},
    {"bucket": "native", "skill_name": "alpha"},
    {"bucket": "external", "skill_name": "...."},  # sanitizes to empty
])
def test_light_partial_args_surface_specific_error_not_generic_light_block(
    tool_name, base_args, partial, tmp_path, monkeypatch
):
    """Partial / invalid bucket+skill_name must yield a SPECIFIC actionable
    error before the generic LIGHT_MODE_BLOCKED. Triad reviewer round 1
    flagged the older test as codifying a weaker contract — this test pins
    the documented behaviour: ⚠️ SKILL_PAYLOAD_ARG_ERROR surfaces uniformly
    across all three payload-mutating tools, regardless of which partial
    shape the caller used."""
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "light")
    reg = _registry(tmp_path)
    args = {**base_args, **partial}
    result = reg.execute(tool_name, args)
    assert "SKILL_PAYLOAD_ARG_ERROR" in result, (
        f"expected specific partial-args error for {tool_name} {partial!r}; "
        f"got: {result[:300]}"
    )
    assert any(
        hint in result
        for hint in (
            "bucket and skill_name must be supplied together",
            "requires a non-empty skill_name",
            "requires bucket/location",
            "read/review only",
        )
    ), result[:300]


def test_b2_external_workspace_stray_bucket_is_ignored_not_blocked(tmp_path, monkeypatch):
    """B2 (v6.33.0) footgun: in an external WORKSPACE edit, a reflexive
    bucket="external" (a real skill-bucket name) on a normal active_workspace
    edit must NOT hard-block with SKILL_PAYLOAD_ARG_ERROR — the stray
    bucket/skill_name are dropped and the workspace edit proceeds. An explicit
    root=skill_payload edit still surfaces the specific error."""
    import ouroboros.safety as safety_mod
    from ouroboros.tools.registry import ToolContext

    system_repo = _git_repo(tmp_path)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    data = tmp_path / "drive"
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "pro")
    monkeypatch.setattr(safety_mod, "check_safety", lambda *a, **k: (True, ""))
    reg = ToolRegistry(repo_dir=system_repo, drive_root=data)
    reg.set_context(ToolContext(
        repo_dir=system_repo,
        drive_root=data,
        workspace_root=workspace,
        workspace_mode="external",
    ))

    # Footgun: stray bucket on a normal workspace edit -> ignored, edit lands.
    result = reg.execute(
        "write_file",
        {"root": "active_workspace", "path": "module.py", "content": "x = 1\n", "bucket": "external"},
    )
    assert "SKILL_PAYLOAD_ARG_ERROR" not in result, result[:300]
    assert (workspace / "module.py").read_text(encoding="utf-8") == "x = 1\n"

    # Explicit skill-payload intent still surfaces the specific error.
    result2 = reg.execute(
        "write_file",
        {"root": "skill_payload", "path": "plugin.py", "content": "x", "bucket": "external"},
    )
    assert "SKILL_PAYLOAD_ARG_ERROR" in result2, result2[:300]


def test_light_control_plane_sidecar_still_blocked_with_bucket_skill_name(tmp_path, monkeypatch):
    """Even with a valid bucket+skill_name pair, the gate refuses control-plane
    sidecars (allow_control_plane=False is preserved). Same protection as repair
    mode — sidecar paths cannot be rewritten via generic tools."""
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "light")
    _make_skill_payload(tmp_path, "ouroboroshub", "delta")
    reg = _registry(tmp_path)
    result = reg.execute(
        "edit_text",
        {
            "path": ".ouroboroshub.json",
            "old_str": "x",
            "new_str": "y",
            "bucket": "ouroboroshub",
            "skill_name": "delta",
        },
    )
    assert "LIGHT_MODE_BLOCKED" in result, result[:200]


def test_light_mode_blocked_message_lists_three_paths(tmp_path, monkeypatch):
    """LIGHT_MODE_BLOCKED message documents all three valid escape hatches so
    agents do not silently fall back to less-idiomatic tools."""
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "light")
    reg = _registry(tmp_path)
    result = reg.execute("write_file", {"path": "README.md", "content": "x"})
    assert "LIGHT_MODE_BLOCKED" in result, result[:200]
    assert "skill_repair" in result
    assert "data/skills/<bucket>" in result
    assert "bucket and skill_name" in result
