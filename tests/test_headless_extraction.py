"""Structural contracts for the semantic-no-op headless extraction."""

from __future__ import annotations

import ast
import pathlib

from ouroboros import headless, headless_status, workspace_file_outputs, workspace_patch_capture


REPO = pathlib.Path(__file__).parents[1]

_LEAVES = (headless_status, workspace_patch_capture, workspace_file_outputs)

_MOVED_OWNERS = {
    "_directory_direct_artifacts": workspace_file_outputs,
    "ARTIFACT_STATUS_FAILED": headless_status,
    "ARTIFACT_STATUS_FINALIZING": headless_status,
    "ARTIFACT_STATUS_MISSING": headless_status,
    "ARTIFACT_STATUS_PENDING": headless_status,
    "ARTIFACT_STATUS_READY": headless_status,
    "ARTIFACT_STATUS_READY_NO_CHANGES": headless_status,
    "ARTIFACT_STATUS_READY_WITH_CHANGES": headless_status,
    "ARTIFACT_TERMINAL_STATUSES": headless_status,
    "_ARTIFACT_LIFECYCLE_FIELDS": headless_status,
    "_FINAL_STATUSES": headless_status,
    "_LOCAL_READONLY_SUBAGENT_MODE": headless_status,
    "SCRATCH_MANIFEST_NAME": workspace_patch_capture,
    "_GIT_UNBORN_HEAD": workspace_patch_capture,
    "_acting_constraint_from_task": workspace_patch_capture,
    "_append_git_output": workspace_patch_capture,
    "_empty_patch_manifest": workspace_patch_capture,
    "_git_bytes": workspace_patch_capture,
    "_git_empty_tree_oid": workspace_patch_capture,
    "_git_path_list": workspace_patch_capture,
    "_git_stdout": workspace_patch_capture,
    "_head_reflog_exists": workspace_patch_capture,
    "_looks_like_git_oid": workspace_patch_capture,
    "_preflight_head_from_task": workspace_patch_capture,
    "_preflight_head_present": workspace_patch_capture,
    "_untracked_blob_exclude_reason": workspace_patch_capture,
    "_workspace_patch_base": workspace_patch_capture,
    "_write_patch_separator": workspace_patch_capture,
    "build_workspace_patch": workspace_patch_capture,
    "untracked_capture_veto_reason": workspace_patch_capture,
    "write_workspace_patch_artifacts": workspace_patch_capture,
}


def test_headless_leaves_are_non_catalog_owners_without_headless_backedges():
    for module in (headless, *_LEAVES):
        source_path = pathlib.Path(module.__file__)
        tree = ast.parse(source_path.read_text(encoding="utf-8"))
        assert not any(
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == "get_tools"
            for node in tree.body
        )
    for module in _LEAVES:
        tree = ast.parse(pathlib.Path(module.__file__).read_text(encoding="utf-8"))
        assert not any(
            isinstance(node, ast.ImportFrom) and node.module == "ouroboros.headless"
            for node in ast.walk(tree)
        )
        assert not any(
            isinstance(node, ast.Import)
            and any(alias.name == "ouroboros.headless" for alias in node.names)
            for node in ast.walk(tree)
        )

    # None of these modules defines get_tools, so they remain non-catalog owners.


def test_headless_public_export_list_preserves_extraction_and_terminal_file_helpers():
    """Terminal file preparation/readiness extend the preserved extraction ABI."""
    assert headless.__all__ == [
        "ARTIFACT_STATUS_FAILED",
        "ARTIFACT_STATUS_FINALIZING",
        "ARTIFACT_STATUS_PENDING",
        "ARTIFACT_STATUS_READY",
        "build_memory_export",
        "build_workspace_patch",
        "copy_child_task_result",
        "prepare_terminal_task_files",
        "terminal_task_files_ready",
        "finalize_task_artifacts",
        "task_is_readonly_subagent",
        "task_requires_workspace_capture",
        "prepare_task_drive",
        "drive_candidates",
        "prune_headless_task_drives",
        "prune_task_drives",
        "task_artifacts_dir",
        "task_state_dir",
        "write_workspace_patch_artifacts",
        "write_workspace_preflight_artifact",
    ]
    for name in headless.__all__:
        assert hasattr(headless, name), name


def test_headless_facade_reexports_every_moved_identity():
    """``headless`` keeps the exact objects, so the supervisor, the gateway,
    outcomes, task_status, artifacts and the delegation owners see no identity
    change."""
    for name, owner in _MOVED_OWNERS.items():
        assert hasattr(headless, name), name
        assert getattr(headless, name) is getattr(owner, name), name
    owned = {name for module in _LEAVES for name in vars(module)}
    assert set(_MOVED_OWNERS) <= owned


def test_headless_extraction_size_bounds_have_meaningful_headroom():
    from ouroboros.size_ratchet_manifest import BAND_PATHS

    counts = {
        module.__name__: len(
            pathlib.Path(module.__file__).read_text(encoding="utf-8").splitlines()
        )
        for module in (headless, *_LEAVES)
    }
    assert counts["ouroboros.headless"] <= 1000 or (
        counts["ouroboros.headless"] <= 1500 and BAND_PATHS.get("ouroboros/headless.py", "").strip())
    assert all(counts[module.__name__] <= 1000 for module in _LEAVES)
    assert counts["ouroboros.workspace_patch_capture"] <= 1000


def test_terminal_file_helper_preserves_legacy_ready_without_finalized_timestamp(tmp_path, monkeypatch):
    from ouroboros.task_results import load_task_result, write_task_result

    task = {"id": "legacy", "workspace_root": str(tmp_path / "workspace")}
    stored = write_task_result(tmp_path, "legacy", "completed", result="approved result", artifact_status="ready")
    def no_capture(*args):
        raise AssertionError("legacy terminal artifacts must not be recaptured")
    monkeypatch.setattr(headless, "finalize_task_artifacts", no_capture)
    prepared = headless.prepare_terminal_task_files(tmp_path, task)
    assert prepared["error"] == "" and prepared["terminal_source_present"] is True
    assert prepared["result"] == stored == load_task_result(tmp_path, "legacy")
    assert "artifact_finalized_at" not in prepared["result"]
    assert headless.terminal_task_files_ready(tmp_path, task, prepared["result"])


def test_current_metadata_race_reuses_unchanged_manifest_publication(tmp_path, monkeypatch):
    import gzip
    import json
    from collections import Counter
    from ouroboros import observability
    from ouroboros.task_results import write_task_result
    from tests.test_copyback_terminal_files import _child, _call, _review, _store

    parent, child, task = _child(tmp_path)
    ref = _call(child)["manifest_ref"]
    payload = json.loads(pathlib.Path(ref["path"]).read_text(encoding="utf-8"))["full_payload_ref"]
    source_blob = pathlib.Path(payload["path"]).resolve()
    target_blob = (parent / "observability" / "blobs" / source_blob.name).resolve()
    review = _review(ref)
    _store(child, task["id"], ref, review_projection=review)
    write_task_result(parent, task["id"], "completed", review_projection=review)
    reads, writes = Counter(), []
    original_open = gzip.open
    original_write = observability.write_call_manifest
    original_rewrite = observability._rewrite_child_ref_tree
    changed = []

    def counted_open(path, mode="rb", *args, **kwargs):
        if "r" in mode:
            reads[str(pathlib.Path(path).resolve())] += 1
        return original_open(path, mode, *args, **kwargs)

    def counted_write(*args, **kwargs):
        writes.append(kwargs["call_id"])
        return original_write(*args, **kwargs)

    def concurrent_metadata(value, *args, **kwargs):
        if isinstance(value, dict) and "panels" in value and not changed:
            changed.append(True)
            assert not (parent / "task_results" / (task["id"] + ".json.lock")).exists()
            latest = _review(ref, headline="CURRENT metadata")
            write_task_result(parent, task["id"], "completed", accounted_upper_bound_usd=42,
                              _field_projector=lambda live, fields: {**fields, "review_projection": latest})
        return original_rewrite(value, *args, **kwargs)

    monkeypatch.setattr(gzip, "open", counted_open)
    monkeypatch.setattr(observability, "write_call_manifest", counted_write)
    monkeypatch.setattr(observability, "_rewrite_child_ref_tree", concurrent_metadata)
    copied = headless.copy_child_task_result(parent, task)
    assert not changed and not reads and not writes, "adoption does not replay history"
    copied = headless.retry_child_task_refs(parent, child, task["id"])
    assert copied["accounted_upper_bound_usd"] == 42
    assert copied["review_projection"]["panels"][0]["headline"] == "CURRENT metadata"
    assert writes == ["call"]
    # A changed CURRENT basis is revalidated without retaining decoded payloads.
    # Source and destination verification still run; identical manifest bytes do not republish.
    assert {path for path in reads if path.endswith(".json.gz")} == {str(source_blob), str(target_blob)}


def test_fork_memory_copy_preserves_identity_and_gives_a_project_no_global_knowledge(tmp_path):
    files = ["identity.md", "WORLD.md", "registry.md", "knowledge/patterns.md", "knowledge/topic.md"]
    for name in files:
        path = tmp_path / "memory" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("original " + name, encoding="utf-8")
    project = headless.prepare_task_drive(tmp_path, "project", "forked", project_id="room")
    plain = headless.prepare_task_drive(tmp_path, "plain", "forked")
    for name in files:
        assert (plain / "memory" / name).read_text(encoding="utf-8") == "original " + name
        if not name.startswith("knowledge/"):
            assert (project / "memory" / name).read_text(encoding="utf-8") == "original " + name
    # A project fork inherits no global knowledge note, the former register included.
    assert not (project / "memory/knowledge/topic.md").exists()
    assert not (project / "memory/knowledge/patterns.md").exists()
