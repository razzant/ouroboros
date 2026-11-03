"""Regression tests for the delegated authority/execution-root seam."""

from __future__ import annotations

import os
from types import SimpleNamespace

from ouroboros.delegate_start_instructions import (
    apply_execution_binding, directory_copy_binding_instruction,
    execution_binding_instruction,
)
from ouroboros.tools.delegate_integration import _capture_block


def test_execution_binding_makes_private_root_the_only_work_product_target():
    from ouroboros.delegate_start_instructions import scratch_fact

    text = execution_binding_instruction("/tmp/private-snapshot", "/tmp/authority")
    assert "the private execution root for this run is /tmp/private-snapshot:" in text
    assert "goes only under it" in text and "every shell" not in text  # work product, not every process write
    assert "/tmp/authority" in text
    assert "read-only identity/reference" in text
    assert "sha256=" in text
    assert "typed execution-root mismatch" in text
    # #2070: disposable process state has a lawful place, the one safe_test itself defaults to,
    # named without the body's own launcher: a foreign project has no safe_test.
    assert "outside every Git working tree" in text and scratch_fact() in text and "safe_test" not in text
    evidence = os.path.join("/tmp/private-snapshot", ".review-evidence")
    assert f"evidence worth returning under {evidence}: the host retains new files there" in text
    assert "never applies them; files the project already tracks there stay ordinary work product" in text


def test_runtime_child_environment_drops_launcher_authority(monkeypatch):
    from ouroboros.settings_integrity import runtime_environ

    monkeypatch.setenv("OUROBOROS_MANAGED_BY_LAUNCHER", "1")
    monkeypatch.setenv("OUROBOROS_MANAGED_REPO_DIR", "/private/launcher-repo")
    assert "OUROBOROS_MANAGED_BY_LAUNCHER" not in runtime_environ()
    assert "OUROBOROS_MANAGED_REPO_DIR" not in runtime_environ()


def test_execution_binding_appends_without_rewriting_canonical_work_order():
    instructions = (
        "HOST TASK CONTRACT AUTHORITY (normalized JSON; predecessor is a brief):\n"
        '{"workspace_root":"/authority","task_constraint":{"write_root":"/authority"}}'
    )
    bound = apply_execution_binding(instructions, "/private/snapshot", "/authority")
    assert bound.startswith(instructions)
    assert '"workspace_root":"/authority"' in bound
    assert '"write_root":"/authority"' in bound
    assert "the private execution root for this run is /private/snapshot:" in bound


def test_directory_copy_binding_does_not_invent_source_as_execution_root():
    text = directory_copy_binding_instruction("/authority")
    assert "/authority" in text
    assert "engine will create a private execution copy" in text
    assert "Do not write to that authority folder directly" in text


def test_capture_block_names_the_saved_result_and_git_failures_in_git_s_words(tmp_path):
    """A capture block reads the saved result: a ready one names its result commit and says the
    checkout goes once it is saved; a failed one carries the failure itself (a typed note first,
    else git's own stderr), never boilerplate. Target drift is no longer part of a block."""
    entry = SimpleNamespace(baseline_sha="base-sha", execution_root="/private/snapshot",
                            target_root="/authority/tree", authority_source="external_workspace_root",
                            patch_disposed="")
    ready = _capture_block(entry, tmp_path, {"status": "ready_with_changes", "sha256": "s", "diffstat": "1 file",
                                             "result_commit": "c" * 40, "current_head": "base-sha"})
    assert ready["result_commit"] == "c" * 40 and "head_moved" not in ready
    assert "NOT APPLIED" in ready["note"] and "checkout itself is released" in ready["note"]
    failed = _capture_block(entry, tmp_path, {"status": "failed", "errors": [
        {"type": "result_commit_failed", "message": "git diff-tree exited 128", "stderr": "fatal: bad object"}]})
    assert failed["note"] == "capture failed: fatal: bad object" and "result_commit" not in failed
    typed = _capture_block(entry, tmp_path, {"status": "failed", "note": "unreviewable_metadata_change",
                                             "errors": [{"stderr": "fatal: bad object"}]})
    assert typed["note"] == "unreviewable_metadata_change"
    for block in (ready, failed, typed):
        assert not {"target_drift_checked", "target_mutated_during_run", "target_drift_unknown"} & set(block)
    moved = _capture_block(entry, tmp_path, {"status": "ready_with_changes", "current_head": "other-sha"})
    assert moved["head_moved"]["current"] == "other-sha"  # a moved HEAD is still disclosed
