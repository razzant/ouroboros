"""The Home/execd capability contract (RWS v2 §3.1 admission, §9 exhaustiveness).

Three things are pinned here, and each one is a way the contract could rot
silently:

* **Exhaustiveness** — every visible workspace tool has an execution-affinity
  declaration and a native operation (or an explicit Home-only classification).
  A new tool added to the workspace surface with neither must fail the build, not
  become an unclassified call whose placement nobody decided.
* **Determinism** — the manifest is a property of the BUILD. If a task's
  visibility filter or a drive root could change the digest, two tasks on one
  server would disagree about which targets are admissible.
* **Drift detection** — a target whose schemas differ from Home's is refused at
  admission rather than trusted, and the refusal names WHICH half differs.
"""

from __future__ import annotations

import hashlib
import pathlib

import pytest

from ouroboros.tool_capabilities import (
    HOME_ONLY_TOOL_NAMES,
    WORKSPACE_TOOL_EXECUTION_AFFINITY,
    assert_workspace_capability_compatible,
    build_workspace_capability_manifest,
)
from ouroboros.tools.registry import ToolRegistry
from ouroboros.workspace_native_contract import (
    MANDATORY_REMOTE_NATIVE_OPERATIONS,
    REMOTE_NATIVE_KERNEL_MODULES,
)

REPO = pathlib.Path(__file__).resolve().parents[1]


def _registry(tmp_path: pathlib.Path) -> ToolRegistry:
    return ToolRegistry(repo_dir=REPO, drive_root=tmp_path)


def _manifest(tmp_path: pathlib.Path) -> dict:
    return _registry(tmp_path).workspace_capability_manifest(repo_root=REPO)


# ── exhaustiveness ───────────────────────────────────────────────────────────


def test_a_tool_is_either_placed_or_home_only_never_both():
    """§9: the two placement tables must PARTITION, not overlap.

    This used to compare `WORKSPACE_TOOL_EXECUTION_AFFINITY` against
    `registry._WORKSPACE_ALLOWED_TOOLS`, but upstream retired that allowlist — a
    workspace task now sees the full registry, narrowed by the ordinary profile and
    contract gates rather than by a hand-kept visibility list. The sibling test
    below already said what that comparison was worth ("proves they agree with each
    other and nothing more"), so re-pointing it at the retired constant's successor
    would rebuild a tautology.

    What still needs pinning is that the two tables are DISJOINT. They answer the
    same question — where does this tool execute — and a name in both would let the
    manifest and the Home-only fence disagree about one tool, which is precisely
    the ambiguity the exhaustiveness gate exists to make impossible.
    """

    both = sorted(set(WORKSPACE_TOOL_EXECUTION_AFFINITY) & set(HOME_ONLY_TOOL_NAMES))
    assert not both, (
        "these tools claim a workspace affinity AND Home-only placement; one tool "
        f"cannot execute in two places: {both}"
    )


def test_every_registered_builtin_has_an_explicit_placement(tmp_path):
    """The exhaustiveness claim, checked against the tools that EXIST.

    The assertion above compares two hand-maintained constants, so it proves they
    agree with each other and nothing more: a genuinely new built-in in NEITHER
    list passed every gate and was classified Home-only by silence. Verified on a
    copy of the repo before this landed. Now the registry is the authority — a new
    built-in must be given a workspace affinity or declared in
    `HOME_ONLY_TOOL_NAMES`, and doing neither fails here with its name.
    """

    registered = set(_registry(tmp_path)._entries)
    declared = set(WORKSPACE_TOOL_EXECUTION_AFFINITY) | set(HOME_ONLY_TOOL_NAMES)
    undeclared = sorted(registered - declared)
    assert not undeclared, (
        "these built-in tools have no placement declaration — add a workspace "
        "affinity to WORKSPACE_TOOL_EXECUTION_AFFINITY or name them in "
        f"HOME_ONLY_TOOL_NAMES: {undeclared}"
    )
    stale = sorted(declared - registered)
    assert not stale, (
        f"these names are declared but no longer registered: {stale}"
    )


def test_the_two_placement_sets_do_not_overlap():
    """A tool cannot be both a workspace surface and declared Home-only."""

    assert not set(WORKSPACE_TOOL_EXECUTION_AFFINITY) & set(HOME_ONLY_TOOL_NAMES)


def test_every_native_operation_backs_a_target_native_affinity():
    """The native allowlist and the affinity table describe the same surface.

    Every operation execd will accept exists because some classified tool needs
    the target to answer for it; a native operation with no such tool would be a
    remote capability Home can never reach, i.e. dead attack surface.
    """

    target_native = {
        name
        for name, affinity in WORKSPACE_TOOL_EXECUTION_AFFINITY.items()
        if affinity in {"root", "cwd", "workspace", "service"}
    }
    # The operations that exist for a classified tool, by name.
    tool_backed = MANDATORY_REMOTE_NATIVE_OPERATIONS & target_native
    # The rest are the target's own internal halves of hybrid/Home tools plus the
    # policy-carrying operations; each must be named, never implicit.
    unmatched = sorted(MANDATORY_REMOTE_NATIVE_OPERATIONS - tool_backed)
    assert unmatched == [
        "classify_ambiguous_workspace_path",
        "execute_reviewed_payload",
        "extract_video_frames",
        "guarded_patch_apply",
        "snapshot_manifest_and_blob_export",
        "verify_remote_check",
    ], unmatched
    assert target_native - MANDATORY_REMOTE_NATIVE_OPERATIONS == set()


# ── determinism ──────────────────────────────────────────────────────────────


def test_the_manifest_is_a_property_of_the_build_not_of_a_task(tmp_path):
    first = _manifest(tmp_path / "drive-a")
    second = _manifest(tmp_path / "drive-b")
    assert first == second
    assert first["manifest_sha256"] == second["manifest_sha256"]
    assert len(first["public_tools"]) == len(WORKSPACE_TOOL_EXECUTION_AFFINITY)
    assert {row["name"] for row in first["native_operations"]} == set(
        MANDATORY_REMOTE_NATIVE_OPERATIONS
    )


def test_the_published_kernel_is_what_actually_travels(tmp_path):
    """`native_kernel_modules` is the declared bundle, not the closure's roots.

    The import closure SEEDS the Home-side transport in the reverse direction
    (`remote_ssh`, `remote_ssh_config`) so that it too is checked for Home policy
    imports. Those never travel in the bundle, so publishing the closure roots as
    the kernel would describe a bundle that does not exist.
    """

    manifest = _manifest(tmp_path)
    assert manifest["native_kernel_modules"] == sorted(REMOTE_NATIVE_KERNEL_MODULES)
    assert "ouroboros.remote_ssh" not in manifest["native_kernel_modules"]


def test_the_wire_projection_carries_the_hash_and_bare_operation_names(tmp_path):
    from ouroboros.remote_worker_proxy import capability_projection

    manifest = _manifest(tmp_path)
    projection = capability_projection(manifest)
    assert projection["manifest_sha256"] == manifest["manifest_sha256"]
    assert projection["native_operations"] == sorted(MANDATORY_REMOTE_NATIVE_OPERATIONS)
    # The public schemas stay HOME-side: nothing about tool arguments is uploaded.
    assert "public_tools" not in projection


# ── drift detection ──────────────────────────────────────────────────────────


def test_a_matching_manifest_is_compatible_with_itself(tmp_path):
    manifest = _manifest(tmp_path)
    assert_workspace_capability_compatible(manifest, manifest)


def test_public_schema_drift_is_refused_and_named(tmp_path):
    home = _manifest(tmp_path)
    backend = _manifest(tmp_path)
    backend["public_tools"][0]["function"]["description"] = "drifted"
    # Re-seal so the refusal is about the SCHEMAS, not about a broken hash.
    from ouroboros.tool_capabilities import _canonical_json_bytes

    backend["public_schema_sha256"] = hashlib.sha256(
        _canonical_json_bytes(backend["public_tools"])
    ).hexdigest()
    resealed = {k: v for k, v in backend.items() if k != "manifest_sha256"}
    backend["manifest_sha256"] = hashlib.sha256(
        _canonical_json_bytes(resealed)
    ).hexdigest()
    with pytest.raises(ValueError, match="schema digest"):
        assert_workspace_capability_compatible(home, backend)


def test_a_tampered_hash_is_refused_before_any_comparison(tmp_path):
    home = _manifest(tmp_path)
    backend = _manifest(tmp_path)
    backend["public_tools"][0]["function"]["description"] = "drifted"
    with pytest.raises(ValueError, match="hash is invalid"):
        assert_workspace_capability_compatible(home, backend)


def test_a_missing_public_schema_is_refused():
    with pytest.raises(ValueError, match="missing public schemas"):
        build_workspace_capability_manifest([], repo_root=REPO)


def test_a_forbidden_native_operation_module_is_refused(tmp_path):
    registry = _registry(tmp_path)
    from ouroboros.tool_capabilities import WORKSPACE_TOOL_EXECUTION_AFFINITY as affinity

    schemas = [
        registry._schema_for_entry(registry._entries[name])
        for name in sorted(affinity)
    ]
    with pytest.raises(ValueError):
        build_workspace_capability_manifest(
            schemas,
            repo_root=REPO,
            operation_modules={
                name: ("ouroboros.config" if name == "read_file" else "ouroboros.workspace_native")
                for name in MANDATORY_REMOTE_NATIVE_OPERATIONS
            },
        )


def test_the_manifest_hash_is_the_same_under_every_tool_profile(tmp_path):
    """"Unfiltered" has to mean unfiltered, or the hash is not an identity.

    The manifest used to be built through `_schema_for_entry`, which is the PER-TASK
    projection: it narrows `root` enums and drops fields for a local-readonly or an
    acting subagent, reading the live `ToolContext`. So the docstring promised a
    property of the BUILD while the code delivered a property of whichever context
    happened to be current — and `manifest_sha256` is a Home↔execd compatibility
    identity that admission compares, so the same build could admit a target for one
    task and refuse it for another with nothing about the target having changed.

    Driven through the profile PREDICATES the registry actually consults, so the
    schemas really do take the narrowing path in the filtered runs.
    """
    from ouroboros.tool_capabilities import (
        ACTING_SUBAGENT_MODE,
        LOCAL_READONLY_SUBAGENT_MODE,
    )

    baseline = _registry(tmp_path / "plain")
    reference = baseline.workspace_capability_manifest(repo_root=REPO)

    profiles = {
        "local_readonly": {"mode": LOCAL_READONLY_SUBAGENT_MODE},
        "acting": {"mode": ACTING_SUBAGENT_MODE, "surface": "active_workspace"},
    }
    for label, constraint in profiles.items():
        registry = _registry(tmp_path / label)
        registry._ctx.task_constraint = constraint
        # The profile really is in force: the live schema surface IS narrowed.
        narrowed = registry._schema_for_entry(registry._entries["read_file"])
        unfiltered = {
            "type": "function",
            "function": registry._entries["read_file"].schema,
        }
        assert narrowed != unfiltered, (
            f"{label} did not narrow anything, so this test would pass vacuously"
        )
        under_profile = registry.workspace_capability_manifest(repo_root=REPO)
        assert under_profile["public_schema_sha256"] == reference["public_schema_sha256"], label
        assert under_profile["manifest_sha256"] == reference["manifest_sha256"], label
        assert under_profile["public_tools"] == reference["public_tools"], label

    # And building it never mutates the registry's own SSOT schemas.
    assert baseline.workspace_capability_manifest(repo_root=REPO) == reference
