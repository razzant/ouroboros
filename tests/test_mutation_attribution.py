from __future__ import annotations

import pathlib
import subprocess

import pytest


def _git(root: pathlib.Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _repo(tmp_path: pathlib.Path) -> pathlib.Path:
    root = tmp_path / "repo"
    root.mkdir(parents=True)
    _git(root, "init", "-q")
    _git(root, "config", "user.email", "tests@example.invalid")
    _git(root, "config", "user.name", "Tests")
    (root / "clean.txt").write_text("base\n", encoding="utf-8")
    (root / "dirty.txt").write_text("base\n", encoding="utf-8")
    _git(root, "add", ".")
    _git(root, "commit", "-qm", "base")
    return root


def test_git_baseline_attributes_only_paths_clean_at_capture(tmp_path):
    from ouroboros.mutation_attribution import (
        attributed_git_candidates,
        capture_mutation_baseline,
    )
    from ouroboros.task_results import STATUS_RUNNING, write_task_result

    root = _repo(tmp_path)
    data = tmp_path / "data"
    write_task_result(data, "task-1", STATUS_RUNNING)
    (root / "dirty.txt").write_text("owner change\n", encoding="utf-8")

    evidence = capture_mutation_baseline(
        data,
        "task-1",
        [{"surface_type": "system_repo", "host_root": str(root)}],
    )
    assert len(evidence["baseline"]["baseline_hash"]) == 64
    assert evidence["effect_state"] == "observed_window"

    (root / "clean.txt").write_text("task change\n", encoding="utf-8")
    (root / "dirty.txt").write_text("owner + task change\n", encoding="utf-8")
    (root / "new.txt").write_text("new\n", encoding="utf-8")

    candidates = attributed_git_candidates(data, "task-1", root)
    assert candidates["candidates"] == ["clean.txt", "new.txt"]
    assert candidates["excluded_preexisting_dirty"] == ["dirty.txt"]
    assert "preexisting_dirty_changed" in candidates["blockers"]


def test_git_candidate_fails_closed_when_head_changes(tmp_path):
    from ouroboros.mutation_attribution import (
        attributed_git_candidates,
        capture_mutation_baseline,
    )
    from ouroboros.task_results import STATUS_RUNNING, write_task_result

    root = _repo(tmp_path)
    data = tmp_path / "data"
    write_task_result(data, "task-2", STATUS_RUNNING)
    capture_mutation_baseline(
        data,
        "task-2",
        [{"surface_type": "system_repo", "host_root": str(root)}],
    )
    (root / "clean.txt").write_text("other commit\n", encoding="utf-8")
    _git(root, "add", "clean.txt")
    _git(root, "commit", "-qm", "foreign")

    candidates = attributed_git_candidates(data, "task-2", root)
    assert candidates["candidates"] == []
    assert "baseline_stale" in candidates["blockers"]


def test_user_files_baseline_fingerprints_only_declared_targets(tmp_path):
    from ouroboros.mutation_attribution import capture_mutation_baseline
    from ouroboros.task_results import STATUS_RUNNING, write_task_result

    data = tmp_path / "data"
    home = tmp_path / "home"
    home.mkdir()
    (home / "declared.txt").write_text("known\n", encoding="utf-8")
    (home / "secret.txt").write_text("must not be scanned\n", encoding="utf-8")
    write_task_result(data, "task-3", STATUS_RUNNING)

    evidence = capture_mutation_baseline(
        data,
        "task-3",
        [{
            "surface_type": "user_files",
            "host_root": str(home),
            "known_paths": ["declared.txt", "future.txt"],
        }],
    )
    surface = evidence["baseline"]["surfaces"][0]
    assert set(surface["known_path_fingerprints"]) == {"declared.txt", "future.txt"}
    assert "secret.txt" not in repr(surface)


def test_effect_flags_block_clean_attribution(tmp_path):
    from ouroboros.mutation_attribution import (
        attributed_git_candidates,
        capture_mutation_baseline,
    )
    from ouroboros.task_results import (
        STATUS_RUNNING,
        load_task_result,
        write_task_result,
    )

    root = _repo(tmp_path)
    data = tmp_path / "data"
    write_task_result(data, "task-4", STATUS_RUNNING)
    capture_mutation_baseline(
        data,
        "task-4",
        [{"surface_type": "system_repo", "host_root": str(root)}],
    )
    (root / "clean.txt").write_text("late\n", encoding="utf-8")
    # A host that observed an anomaly appends a flag to the durable evidence;
    # the flag rides into blockers without vetoing anything downstream.
    evidence = dict((load_task_result(data, "task-4") or {})["mutation_evidence"])
    evidence["flags"] = [{"flag": "external_writer_observed"}]
    write_task_result(data, "task-4", STATUS_RUNNING, mutation_evidence=evidence)

    candidates = attributed_git_candidates(data, "task-4", root)
    assert candidates["candidates"] == ["clean.txt"]
    assert "external_writer_observed" in candidates["blockers"]


def test_late_surface_extension_never_rebases_an_active_root(tmp_path):
    from ouroboros.mutation_attribution import (
        attributed_git_candidates,
        capture_mutation_baseline,
    )
    from ouroboros.task_results import STATUS_RUNNING, write_task_result

    first = _repo(tmp_path / "first")
    second = _repo(tmp_path / "second")
    data = tmp_path / "data"
    write_task_result(data, "task-late", STATUS_RUNNING)
    initial = capture_mutation_baseline(
        data,
        "task-late",
        [{"surface_type": "system_repo", "host_root": str(first)}],
    )
    (first / "clean.txt").write_text("first writer\n", encoding="utf-8")

    extended = capture_mutation_baseline(
        data,
        "task-late",
        [{"surface_type": "external_workspace", "host_root": str(second)}],
    )
    assert len(extended["baseline"]["surfaces"]) == 2
    assert extended["baseline"]["baseline_hash"] != initial["baseline"]["baseline_hash"]
    assert attributed_git_candidates(data, "task-late", first)["candidates"] == ["clean.txt"]

    unchanged = capture_mutation_baseline(
        data,
        "task-late",
        [{"surface_type": "system_repo", "host_root": str(first)}],
    )
    assert unchanged["baseline"]["baseline_hash"] == extended["baseline"]["baseline_hash"]


def test_reentrant_user_surface_baselines_each_new_exact_target(tmp_path):
    from ouroboros.mutation_attribution import capture_mutation_baseline
    from ouroboros.task_results import STATUS_RUNNING, write_task_result

    data = tmp_path / "data"
    user_root = tmp_path / "user"
    user_root.mkdir()
    first = user_root / "first.txt"
    second = user_root / "second.txt"
    first.write_text("owner-first\n", encoding="utf-8")
    second.write_text("owner-second\n", encoding="utf-8")
    write_task_result(data, "task-targets", STATUS_RUNNING)

    initial = capture_mutation_baseline(
        data,
        "task-targets",
        [{
            "surface_type": "user_files",
            "host_root": str(user_root),
            "known_paths": [str(first)],
        }],
    )
    first.write_text("task-first\n", encoding="utf-8")
    extended = capture_mutation_baseline(
        data,
        "task-targets",
        [{
            "surface_type": "user_files",
            "host_root": str(user_root),
            "known_paths": [str(second)],
        }],
    )

    before = initial["baseline"]["surfaces"][0]
    after = extended["baseline"]["surfaces"][0]
    assert after["known_paths"] == ["first.txt", "second.txt"]
    assert after["known_path_fingerprints"]["first.txt"] == before[
        "known_path_fingerprints"
    ]["first.txt"]
    assert after["known_path_fingerprints"]["second.txt"]["sha256"] != after[
        "known_path_fingerprints"
    ]["first.txt"]["sha256"]


def test_missing_baseline_never_returns_candidates(tmp_path):
    from ouroboros.mutation_attribution import attributed_git_candidates

    root = _repo(tmp_path)
    (root / "clean.txt").write_text("changed\n", encoding="utf-8")
    result = attributed_git_candidates(tmp_path / "data", "missing", root)
    assert result["candidates"] == []
    assert result["blockers"] == ["baseline_missing"]


def test_attribution_task_id_prefers_lineage_order_with_baseline(tmp_path):
    from ouroboros.mutation_attribution import (
        attribution_task_id,
        capture_mutation_baseline,
    )
    from ouroboros.task_results import STATUS_RUNNING, write_task_result

    root = _repo(tmp_path)
    data = tmp_path / "data"
    write_task_result(data, "root-task", STATUS_RUNNING)
    write_task_result(data, "child-task", STATUS_RUNNING)
    capture_mutation_baseline(
        data,
        "root-task",
        [{"surface_type": "system_repo", "host_root": str(root)}],
    )

    # Lineage order (root first) wins directly.
    assert attribution_task_id(data, ("root-task", "child-task")) == "root-task"
    # A baseline-less candidate never claims attribution; lookup falls through.
    assert attribution_task_id(data, ("child-task", "root-task")) == "root-task"
    # Blank/None candidates are skipped, not treated as matches.
    assert attribution_task_id(data, ("", None, "root-task")) == "root-task"


def test_attribution_task_id_empty_without_any_baseline(tmp_path):
    from ouroboros.mutation_attribution import attribution_task_id
    from ouroboros.task_results import STATUS_RUNNING, write_task_result

    data = tmp_path / "data"
    write_task_result(data, "plain-task", STATUS_RUNNING)

    assert attribution_task_id(data, ("plain-task", "missing-task")) == ""
    assert attribution_task_id(data, ()) == ""


def test_explicit_staging_paths_must_be_candidate_subset(tmp_path):
    from ouroboros.mutation_attribution import (
        capture_mutation_baseline,
        resolve_attributed_git_paths,
    )
    from ouroboros.task_results import STATUS_RUNNING, write_task_result

    root = _repo(tmp_path)
    data = tmp_path / "data"
    write_task_result(data, "task-5", STATUS_RUNNING)
    capture_mutation_baseline(
        data,
        "task-5",
        [{"surface_type": "system_repo", "host_root": str(root)}],
    )
    (root / "clean.txt").write_text("changed\n", encoding="utf-8")

    selected, _evidence, error = resolve_attributed_git_paths(
        data, "task-5", root, ["clean.txt"]
    )
    assert error == ""
    assert selected == ["clean.txt"]

    selected, _evidence, error = resolve_attributed_git_paths(
        data, "task-5", root, ["dirty.txt"]
    )
    assert selected == []
    assert "subset" in error


def test_empty_explicit_or_computed_candidate_never_means_add_all(tmp_path):
    from ouroboros.mutation_attribution import (
        capture_mutation_baseline,
        resolve_attributed_git_paths,
    )
    from ouroboros.task_results import STATUS_RUNNING, write_task_result

    root = _repo(tmp_path)
    data = tmp_path / "data"
    write_task_result(data, "task-6", STATUS_RUNNING)
    capture_mutation_baseline(
        data,
        "task-6",
        [{"surface_type": "system_repo", "host_root": str(root)}],
    )
    for explicit in (None, []):
        selected, _evidence, error = resolve_attributed_git_paths(
            data, "task-6", root, explicit
        )
        assert selected == []
        assert "GIT_NO_ATTRIBUTED_CHANGES" in error


def test_terminal_candidate_snapshot_reuses_baseline_evidence(tmp_path):
    from ouroboros.mutation_attribution import (
        capture_mutation_baseline,
        record_terminal_mutation_candidates,
    )
    from ouroboros.task_results import STATUS_COMPLETED, STATUS_RUNNING, write_task_result

    root = _repo(tmp_path)
    data = tmp_path / "data"
    write_task_result(data, "task-terminal", STATUS_RUNNING)
    capture_mutation_baseline(
        data,
        "task-terminal",
        [{"surface_type": "system_repo", "host_root": str(root)}],
    )
    (root / "clean.txt").write_text("terminal change\n", encoding="utf-8")
    write_task_result(data, "task-terminal", STATUS_COMPLETED)

    evidence = record_terminal_mutation_candidates(data, "task-terminal")

    assert evidence["effect_state"] == "quiescent"
    snapshot = evidence["terminal_candidate_snapshot"]
    assert snapshot["baseline_hash"] == evidence["baseline"]["baseline_hash"]
    assert snapshot["surfaces"][0]["candidates"] == ["clean.txt"]
    assert snapshot["surfaces"][0]["blockers"] == []


def test_terminal_candidate_snapshot_keeps_committed_task_delta(tmp_path):
    from ouroboros.mutation_attribution import (
        capture_mutation_baseline,
        record_terminal_mutation_candidates,
    )
    from ouroboros.task_results import STATUS_COMPLETED, STATUS_RUNNING, write_task_result

    root = _repo(tmp_path)
    data = tmp_path / "data"
    write_task_result(data, "task-committed", STATUS_RUNNING)
    capture_mutation_baseline(
        data,
        "task-committed",
        [{"surface_type": "system_repo", "host_root": str(root)}],
    )
    (root / "clean.txt").write_text("committed task change\n", encoding="utf-8")
    subprocess.run(["git", "add", "clean.txt"], cwd=root, check=True)
    subprocess.run(
        ["git", "-c", "user.name=Test", "-c", "user.email=test@example.com", "commit", "-m", "task"],
        cwd=root,
        check=True,
        capture_output=True,
    )
    write_task_result(data, "task-committed", STATUS_COMPLETED)

    evidence = record_terminal_mutation_candidates(data, "task-committed")

    surface = evidence["terminal_candidate_snapshot"]["surfaces"][0]
    assert surface["candidates"] == ["clean.txt"]
    assert surface["blockers"] == []
    assert surface["head_advanced"] is True


def test_quiescent_evidence_without_terminal_snapshot_fails_closed(tmp_path):
    from ouroboros.mutation_attribution import (
        capture_mutation_baseline,
        mutation_evidence_projection,
    )
    from ouroboros.task_results import STATUS_RUNNING, write_task_result

    root = _repo(tmp_path)
    data = tmp_path / "data"
    write_task_result(data, "task-missing-terminal", STATUS_RUNNING)
    evidence = capture_mutation_baseline(
        data,
        "task-missing-terminal",
        [{"surface_type": "system_repo", "host_root": str(root)}],
    )

    # A quiescent claim is only credible together with the terminal snapshot
    # record_terminal_mutation_candidates publishes atomically.
    projection = mutation_evidence_projection({**evidence, "effect_state": "quiescent"})

    assert projection["clean_eligible"] is False
    assert "terminal_snapshot_missing" in projection["blockers"]


def test_mutation_blockers_never_veto_outcome_but_ride_the_evidence():
    from ouroboros.outcomes import derive_loop_outcome

    projection = {
        "version": 1,
        "present": True,
        "effect_state": "observed_window",
        "blockers": ["preexisting_dirty_changed"],
        "clean_eligible": False,
    }
    trace = {
        "tool_calls": [],
        "review_runs": [{
            "aggregate_signal": "PASS",
            "actors": [{
                "status": "ok",
                "parsed": {"verdict": "PASS", "outcome_tier": "solved"},
            }],
        }],
        "mutation_attribution": projection,
    }

    outcome = derive_loop_outcome("done", {}, trace)
    axes = outcome["outcome_axes"]
    # Evidence never vetoes: execution and objective stay whatever the run and
    # the reviewing panels earned...
    assert axes["review"]["status"] == "pass"
    assert axes["execution"]["status"] == "ok"
    assert axes["execution"]["reason_code"] == "final_message"
    assert axes["objective"]["status"] == "pass"
    assert outcome["failure"] is None
    # ...while the projection (blockers included) is attached for the panels.
    assert axes["execution"]["mutation_attribution"] == projection
    assert axes["execution"]["mutation_attribution"]["blockers"] == [
        "preexisting_dirty_changed"
    ]


def test_worker_outcome_loads_mutation_evidence_from_canonical_budget_root(tmp_path):
    from types import SimpleNamespace

    from ouroboros import agent_task_pipeline
    from ouroboros.task_results import STATUS_RUNNING, write_task_result

    canonical = tmp_path / "canonical"
    child_drive = tmp_path / "child"
    write_task_result(
        canonical,
        "root-task",
        STATUS_RUNNING,
        mutation_evidence={
            "baseline": {
                "baseline_hash": "baseline-hash",
                "surfaces": [{
                    "surface_type": "external_workspace",
                    "known_paths": [],
                }],
            },
            "effect_state": "observed_window",
            "flags": [],
        },
    )
    trace = {
        "tool_calls": [],
        "review_runs": [{
            "aggregate_signal": "PASS",
            "actors": [{
                "status": "ok",
                "parsed": {"verdict": "PASS", "outcome_tier": "solved"},
            }],
        }],
    }
    env = SimpleNamespace(
        drive_root=child_drive,
        budget_drive_root=canonical,
    )

    outcome = agent_task_pipeline._derive_host_bound_loop_outcome(
        env,
        {
            "id": "child-task",
            "root_task_id": "root-task",
            "budget_drive_root": str(canonical),
        },
        "done",
        {},
        trace,
    )

    projection = trace["mutation_attribution"]
    assert projection["present"] is True
    assert projection["effect_state"] == "observed_window"
    assert outcome["outcome_axes"]["execution"]["status"] == "ok"
    assert outcome["outcome_axes"]["execution"]["mutation_attribution"] == projection


def test_root_outcome_derivation_records_terminal_snapshot(tmp_path):
    from types import SimpleNamespace

    from ouroboros import agent_task_pipeline
    from ouroboros.mutation_attribution import capture_mutation_baseline
    from ouroboros.task_results import (
        STATUS_RUNNING,
        load_task_result,
        write_task_result,
    )

    root = _repo(tmp_path)
    data = tmp_path / "data"
    write_task_result(data, "root-task", STATUS_RUNNING)
    capture_mutation_baseline(
        data,
        "root-task",
        [{"surface_type": "system_repo", "host_root": str(root)}],
    )
    (root / "clean.txt").write_text("root work\n", encoding="utf-8")
    env = SimpleNamespace(drive_root=data, budget_drive_root=data)
    trace = {"tool_calls": []}

    outcome = agent_task_pipeline._derive_host_bound_loop_outcome(
        env,
        {"id": "root-task", "root_task_id": "root-task"},
        "done",
        {},
        trace,
    )

    stored = (load_task_result(data, "root-task") or {})["mutation_evidence"]
    assert stored["effect_state"] == "quiescent"
    assert stored["terminal_candidate_snapshot"]["surfaces"][0]["candidates"] == [
        "clean.txt"
    ]
    projection = trace["mutation_attribution"]
    assert projection["effect_state"] == "quiescent"
    assert projection["terminal_snapshot_present"] is True
    assert projection["clean_eligible"] is True
    assert outcome["outcome_axes"]["execution"]["status"] == "ok"
    assert outcome["outcome_axes"]["execution"]["mutation_attribution"] == projection


def test_clean_observed_window_does_not_degrade_outcome(tmp_path):
    from ouroboros.mutation_attribution import (
        capture_mutation_baseline,
        load_mutation_evidence_projection,
    )
    from ouroboros.outcomes import derive_loop_outcome
    from ouroboros.task_results import STATUS_RUNNING, write_task_result

    root = _repo(tmp_path)
    data = tmp_path / "data"
    write_task_result(data, "task-clean", STATUS_RUNNING)
    capture_mutation_baseline(
        data,
        "task-clean",
        [{"surface_type": "system_repo", "host_root": str(root)}],
    )
    projection = load_mutation_evidence_projection(data, "task-clean")
    assert projection["effect_state"] == "observed_window"
    assert projection["clean_eligible"] is True

    outcome = derive_loop_outcome(
        "done",
        {},
        {"tool_calls": [], "mutation_attribution": projection},
    )
    assert outcome["outcome_axes"]["execution"]["status"] == "ok"
    assert outcome["outcome_axes"]["execution"]["mutation_attribution"] == projection


def test_unchanged_persisting_owner_dirt_never_blocks_staging(tmp_path):
    """Owner WIP that merely persists (same content) must not wedge commits."""
    from ouroboros.mutation_attribution import (
        attributed_git_candidates,
        capture_mutation_baseline,
    )
    from ouroboros.task_results import STATUS_RUNNING, write_task_result

    root = _repo(tmp_path)
    data = tmp_path / "data"
    write_task_result(data, "task-persist", STATUS_RUNNING)
    (root / "owner_wip.txt").write_text("owner WIP\n", encoding="utf-8")
    capture_mutation_baseline(
        data,
        "task-persist",
        [{"surface_type": "system_repo", "host_root": str(root)}],
    )
    (root / "clean.txt").write_text("task change\n", encoding="utf-8")

    candidates = attributed_git_candidates(data, "task-persist", root)
    assert candidates["candidates"] == ["clean.txt"]
    # Still-dirty owner file is excluded evidence, but NOT a blocker while its
    # content is byte-identical to the baseline fingerprint.
    assert candidates["excluded_preexisting_dirty"] == ["owner_wip.txt"]
    assert candidates["blockers"] == []


def test_attributed_staging_survives_own_commit_via_epoch_advance(tmp_path):
    """A task's own commit advances the baseline epoch instead of staling it."""
    from ouroboros.mutation_attribution import (
        advance_mutation_baseline,
        attributed_git_candidates,
        capture_mutation_baseline,
    )
    from ouroboros.task_results import STATUS_RUNNING, write_task_result

    root = _repo(tmp_path)
    data = tmp_path / "data"
    write_task_result(data, "task-seq", STATUS_RUNNING)
    (root / "owner_wip.txt").write_text("owner WIP\n", encoding="utf-8")
    capture_mutation_baseline(
        data,
        "task-seq",
        [{"surface_type": "system_repo", "host_root": str(root)}],
    )

    # First attributed commit by the task itself.
    (root / "first.txt").write_text("first change\n", encoding="utf-8")
    first = attributed_git_candidates(data, "task-seq", root)
    assert first["candidates"] == ["first.txt"] and first["blockers"] == []
    _git(root, "add", "first.txt")
    _git(root, "commit", "-qm", "task commit 1")
    advanced = advance_mutation_baseline(data, "task-seq", root)
    epochs = advanced["baseline"]["epochs"]
    assert epochs and epochs[-1]["reason"] == "attributed_commit"

    # Second commit window opens cleanly from the new HEAD.
    (root / "second.txt").write_text("second change\n", encoding="utf-8")
    second = attributed_git_candidates(data, "task-seq", root)
    assert second["candidates"] == ["second.txt"]
    assert second["blockers"] == []
    assert second["excluded_preexisting_dirty"] == ["owner_wip.txt"]

    # An UNEXPLAINED head move (no advance) still fails closed.
    (root / "clean.txt").write_text("foreign commit\n", encoding="utf-8")
    _git(root, "add", "clean.txt")
    _git(root, "commit", "-qm", "foreign")
    stale = attributed_git_candidates(data, "task-seq", root)
    assert "baseline_stale" in stale["blockers"]


def test_oversized_dirty_artifact_gets_size_only_fingerprint(tmp_path, monkeypatch):
    from ouroboros import mutation_attribution as ma

    monkeypatch.setattr(ma, "_FINGERPRINT_MAX_BYTES", 8)
    big = tmp_path / "big.bin"
    big.write_bytes(b"0123456789abcdef")
    fingerprint = ma._path_fingerprint(big)
    assert fingerprint == {
        "kind": "file",
        "size": 16,
        "sha256_skipped": "over_size_cap",
    }


def test_epoch_advance_keeps_own_uncommitted_leftovers_attributable(tmp_path):
    """Partial staging: the task's own leftover file survives the epoch advance."""
    from ouroboros.mutation_attribution import (
        advance_mutation_baseline,
        attributed_git_candidates,
        capture_mutation_baseline,
    )
    from ouroboros.task_results import STATUS_RUNNING, write_task_result

    root = _repo(tmp_path)
    data = tmp_path / "data"
    write_task_result(data, "task-partial", STATUS_RUNNING)
    capture_mutation_baseline(
        data,
        "task-partial",
        [{"surface_type": "system_repo", "host_root": str(root)}],
    )
    (root / "committed.txt").write_text("goes into commit 1\n", encoding="utf-8")
    (root / "leftover.txt").write_text("waits for commit 2\n", encoding="utf-8")
    _git(root, "add", "committed.txt")
    _git(root, "commit", "-qm", "task commit 1 (partial)")
    advance_mutation_baseline(data, "task-partial", root)

    second = attributed_git_candidates(data, "task-partial", root)
    assert second["candidates"] == ["leftover.txt"]
    assert second["blockers"] == []
    assert second["excluded_preexisting_dirty"] == []


def test_harmless_foreign_fast_forward_reanchors_instead_of_wedging(tmp_path):
    """A foreign commit touching none of the dirty paths must not wedge staging."""
    from ouroboros.mutation_attribution import (
        capture_mutation_baseline,
        resolve_attributed_git_paths,
    )
    from ouroboros.task_results import STATUS_RUNNING, load_task_result, write_task_result

    root = _repo(tmp_path)
    data = tmp_path / "data"
    write_task_result(data, "task-ff", STATUS_RUNNING)
    capture_mutation_baseline(
        data,
        "task-ff",
        [{"surface_type": "system_repo", "host_root": str(root)}],
    )
    (root / "mine.txt").write_text("task change\n", encoding="utf-8")
    # Foreign fast-forward that does not touch any currently dirty path.
    (root / "unrelated.txt").write_text("someone else\n", encoding="utf-8")
    _git(root, "add", "unrelated.txt")
    _git(root, "commit", "-qm", "foreign unrelated")

    selected, evidence, error = resolve_attributed_git_paths(data, "task-ff", root, None)
    assert error == ""
    assert selected == ["mine.txt"]
    epochs = (
        load_task_result(data, "task-ff")["mutation_evidence"]["baseline"]["epochs"]
    )
    assert epochs[-1]["reason"] == "foreign_ff_reanchor"


def test_conflicting_foreign_commit_stays_fail_closed(tmp_path):
    """A foreign commit touching a dirty path keeps the stale blocker."""
    from ouroboros.mutation_attribution import (
        capture_mutation_baseline,
        resolve_attributed_git_paths,
    )
    from ouroboros.task_results import STATUS_RUNNING, write_task_result

    root = _repo(tmp_path)
    data = tmp_path / "data"
    write_task_result(data, "task-conflict", STATUS_RUNNING)
    capture_mutation_baseline(
        data,
        "task-conflict",
        [{"surface_type": "system_repo", "host_root": str(root)}],
    )
    (root / "mine.txt").write_text("task change\n", encoding="utf-8")
    # Foreign commit REWRITES the very path the task is editing.
    _git(root, "-c", "user.name=Other", "-c", "user.email=other@example.invalid",
         "commit", "-qam", "foreign touches clean.txt", "--allow-empty")
    (root / "clean.txt").write_text("foreign committed content\n", encoding="utf-8")
    _git(root, "add", "clean.txt")
    _git(root, "commit", "-qm", "foreign conflicting")
    (root / "clean.txt").write_text("task also edits it now\n", encoding="utf-8")

    selected, evidence, error = resolve_attributed_git_paths(data, "task-conflict", root, None)
    assert "GIT_ATTRIBUTION_BLOCKED" in error
    assert "baseline_stale" in (evidence.get("blockers") or [])


@pytest.mark.parametrize("surface_type", ["system_repo", "external_workspace"])
def test_trashed_worktree_overflow_is_a_typed_blocker(tmp_path, monkeypatch, surface_type):
    """An admitted folder's work is its delivery, never a body candidate's uncommitted remainder, past the dirty cap
    too: only the body's own overflow leaves an unknown remainder for the card."""
    from ouroboros import mutation_attribution as ma
    from ouroboros.task_results import STATUS_RUNNING, write_task_result

    monkeypatch.setattr(ma, "_BASELINE_DIRTY_PATHS_MAX", 2)
    root = _repo(tmp_path)
    data = tmp_path / "data"
    write_task_result(data, "task-trash", STATUS_RUNNING)
    for idx in range(4):
        (root / f"junk_{idx}.tmp").write_text("x\n", encoding="utf-8")
    ma.capture_mutation_baseline(
        data,
        "task-trash",
        [{"surface_type": surface_type, "host_root": str(root)}],
    )
    candidates = ma.attributed_git_candidates(data, "task-trash", root)
    assert candidates["candidates"] == []
    assert candidates["blockers"] == ["baseline_dirty_overflow"]
    evidence = ma.record_terminal_mutation_candidates(data, "task-trash")
    [row] = evidence["terminal_candidate_snapshot"]["surfaces"]
    assert row["blockers"] == ["baseline_dirty_overflow"]
    remainder = ma.mutation_evidence_projection(evidence).get("uncommitted_remainder")
    if surface_type == "system_repo":
        assert row["uncommitted_state"] == "unknown" and remainder["state"] == "unknown"
    else:
        assert "uncommitted_paths" not in row and "uncommitted_state" not in row and remainder is None


def _unborn(path: pathlib.Path) -> pathlib.Path:
    path.mkdir(parents=True)
    _git(path, "init", "-q")
    _git(path, "config", "user.email", "tests@example.invalid")
    _git(path, "config", "user.name", "Tests")
    return path


def test_an_unborn_admitted_folder_is_baselined_against_the_empty_tree_beside_the_system_repo(tmp_path):
    """Astra, 8 Nov 2026: a freshly initialized folder has no HEAD yet. Its baseline is the empty tree on a branch
    ``workspace_preflight.git_head_basis`` proves unborn, so a process's first file is this task's work before any
    commit, the owner's earlier untracked file stays foreign, and the system repo's baseline is kept beside it. The
    task's own first (root) commit stays its attributed path; a first commit no epoch records is a moved base."""
    from ouroboros.mutation_attribution import (
        advance_mutation_baseline, attributed_commit_paths, attributed_git_candidates, capture_root_baseline,
        record_terminal_mutation_candidates,
    )
    from ouroboros.task_results import STATUS_RUNNING, write_task_result

    system, data = _repo(tmp_path / "system"), tmp_path / "data"
    empty = subprocess.run(["git", "hash-object", "-t", "tree", "--stdin"], cwd=system, input="",
                           capture_output=True, text=True, check=True).stdout.strip()
    for task_id in ("own", "shell"):
        folder = _unborn(tmp_path / task_id)
        (folder / "owner.txt").write_text("the owner's file before the task\n", encoding="utf-8")
        write_task_result(data, task_id, STATUS_RUNNING)
        evidence = capture_root_baseline(data, {"id": task_id}, system_repo=system, workspace=folder, conversation=False)
        rows = {row["surface_type"]: row for row in evidence["baseline"]["surfaces"]}
        assert sorted(rows) == ["external_workspace", "system_repo"], "the system repo's baseline is never lost with it"
        git = rows["external_workspace"]["git"]
        assert (git["base_commit"], git["base_tree"], git["head_state"], git["dirty_paths"]) == (
            "", empty, "unborn", ["owner.txt"])
        assert "head_state" not in rows["system_repo"]["git"], "a born repository's row keeps its shape"
        (folder / "first.txt").write_text("made by a process\n", encoding="utf-8")
        found = attributed_git_candidates(data, task_id, folder)
        assert (found["candidates"], found["excluded_preexisting_dirty"], found["blockers"], found["base_tree"]) == (
            ["first.txt"], ["owner.txt"], [], empty)
        _git(folder, "add", "first.txt")
        _git(folder, "commit", "-qm", "first")
        if task_id == "own":  # the task's own reviewed commit opens the next epoch from it
            advance_mutation_baseline(data, task_id, folder, commit=_git(folder, "rev-parse", "HEAD"))
            assert attributed_commit_paths(data, task_id, folder) == ["first.txt"], "a root commit lists its paths"
            assert attributed_git_candidates(data, task_id, folder)["blockers"] == []
            continue
        assert attributed_commit_paths(data, task_id, folder) == [], "a commit no epoch records is no authorship"
        assert "baseline_stale" in attributed_git_candidates(data, task_id, folder)["blockers"]
        terminal = record_terminal_mutation_candidates(data, task_id)["terminal_candidate_snapshot"]["surfaces"]
        [row] = [row for row in terminal if row["surface_type"] == "external_workspace"]
        assert (row["candidates"], row["blockers"], row["head_advanced"]) == (["first.txt"], [], True), \
            "the empty tree..HEAD interval is read like any base..HEAD"


@pytest.mark.parametrize("broken", ["folder", "system"])
def test_a_surface_whose_head_cannot_be_proven_is_left_out_and_the_other_kept(tmp_path, broken):
    """A HEAD that is neither a commit nor a positively unborn branch (a deleted branch with history) is an
    unknown basis: that surface gets no baseline (a named gap), never a guessed empty tree, and the root's other
    admitted surface is still recorded."""
    from ouroboros.mutation_attribution import attributed_git_candidates, capture_root_baseline
    from ouroboros.task_results import STATUS_RUNNING, write_task_result

    system, data = _repo(tmp_path / "system"), tmp_path / "data"
    folder = _repo(tmp_path / "folder")
    _git(folder if broken == "folder" else system, "symbolic-ref", "HEAD", "refs/heads/gone")
    write_task_result(data, "task-1", STATUS_RUNNING)
    evidence = capture_root_baseline(data, {"id": "task-1"}, system_repo=system, workspace=folder, conversation=False)
    assert [row["surface_type"] for row in evidence["baseline"]["surfaces"]] == (
        ["system_repo"] if broken == "folder" else ["external_workspace"])
    missing = attributed_git_candidates(data, "task-1", folder if broken == "folder" else system)
    assert missing["blockers"] == ["baseline_surface_missing"]


def test_a_roots_admitted_git_folder_is_baselined_at_its_start_for_queued_and_conversation_roots(tmp_path):
    """Owner, 8 Nov 2026: queue membership is not what makes production attributable. Any root admitted to a Git
    folder of its own gets that folder's baseline at its start; a queued root also keeps its system repo. A resumed
    task keeps the surfaces it started with, and a folder inside another tree is no surface of its own."""
    from ouroboros.mutation_attribution import (
        attributed_git_candidates, attribution_task_id, capture_mutation_baseline, capture_root_baseline,
    )
    from ouroboros.task_results import STATUS_RUNNING, write_task_result

    system, folder = _repo(tmp_path / "system"), _repo(tmp_path / "folder")
    data = tmp_path / "data"
    for task_id in ("queued", "chat", "resumed", "nested"):
        write_task_result(data, task_id, STATUS_RUNNING)

    def surfaces(evidence):
        return sorted(row["surface_type"] for row in (evidence or {}).get("baseline", {}).get("surfaces", []))

    queued = capture_root_baseline(data, {"id": "queued"}, system_repo=system, workspace=folder, conversation=False)
    assert surfaces(queued) == ["external_workspace", "system_repo"]
    chat = capture_root_baseline(data, {"id": "chat"}, system_repo=system, workspace=folder, conversation=True)
    assert surfaces(chat) == ["external_workspace"], "a conversation root never takes the body's staging baseline"
    assert capture_root_baseline(data, {"id": "x"}, system_repo=system, workspace=system, conversation=True) is None

    (folder / "report.md").write_text("made\n", encoding="utf-8")
    assert attributed_git_candidates(data, "chat", folder)["candidates"] == ["report.md"]
    # The commit gate's staging contract is unchanged where the baseline names another folder (legacy staging).
    assert attribution_task_id(data, ("chat",), repo_root=system) == ""
    assert attribution_task_id(data, ("chat",), repo_root=folder) == attribution_task_id(data, ("chat",)) == "chat"
    assert attribution_task_id(data, ("queued",), repo_root=system) == "queued"

    capture_mutation_baseline(data, "resumed", [{"surface_type": "system_repo", "host_root": str(system)}])
    resumed = capture_root_baseline(data, {"id": "resumed"}, system_repo=system, workspace=folder, conversation=False)
    assert surfaces(resumed) == ["system_repo"], "no folder is baselined after its work may have been done"
    (folder / "inner").mkdir()
    nested = capture_root_baseline(data, {"id": "nested"}, system_repo=system, workspace=folder / "inner",
                                   conversation=True)
    assert nested is None, "a subfolder's status paths are its tree's, never its own"


def test_the_tasks_own_reviewed_commits_stay_its_attributed_paths_but_a_pull_never_does(tmp_path):
    from ouroboros.mutation_attribution import (
        advance_mutation_baseline, attributed_commit_paths, capture_mutation_baseline,
    )
    from ouroboros.task_results import STATUS_RUNNING, write_task_result

    root = _repo(tmp_path)
    data = tmp_path / "data"
    write_task_result(data, "task-1", STATUS_RUNNING)
    capture_mutation_baseline(data, "task-1", [{"surface_type": "external_workspace", "host_root": str(root)}])
    (root / "clean.txt").write_text("the task's change\n", encoding="utf-8")
    _git(root, "commit", "-qam", "own reviewed commit")
    advance_mutation_baseline(data, "task-1", root, commit=_git(root, "rev-parse", "HEAD"))
    (root / "foreign.txt").write_text("someone else\n", encoding="utf-8")
    _git(root, "add", "foreign.txt")
    _git(root, "commit", "-qm", "a pulled commit")
    assert attributed_commit_paths(data, "task-1", root) == ["clean.txt"]


def test_git_paths_are_utf8_whatever_the_process_default_encoding(tmp_path, monkeypatch):
    """Windows CI, 10 Nov 2026: ``text=True`` without an encoding decoded Git's ``-z`` path lists with the locale's
    code page (cp1252 there). A Cyrillic name broke the status walk and every folder attribution built on it, and the
    project import's listing lost its credential check with it. Git writes those paths as UTF-8 bytes on every OS;
    the emulated cp1252 default below stands for Windows on any test machine."""
    from ouroboros import mutation_attribution
    from ouroboros.project_sources import _unstage_sensitive_paths

    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "advanced")
    root = _repo(tmp_path)
    (root / "файл.txt").write_bytes(b"base\n")
    _git(root, "add", "файл.txt")
    _git(root, "commit", "-qm", "a Cyrillic name")
    (root / "файл.txt").write_bytes(b"changed\n")
    (root / "я.txt").write_bytes(b"new\n")
    project = tmp_path / "project"
    project.mkdir()
    _git(project, "init", "-q")
    (project / "я.txt").write_bytes(b"kept\n")
    (project / ".env").write_bytes(b"TOKEN=1\n")
    _git(project, "add", "-A")
    real_run = subprocess.run

    def windows_default(cmd, *args, **kwargs):
        if (kwargs.get("text") or kwargs.get("universal_newlines")) and kwargs.get("encoding") is None:
            kwargs["encoding"] = "cp1252"
        return real_run(cmd, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(subprocess, "run", windows_default)
        assert mutation_attribution._git_status_paths(root) == ["файл.txt", "я.txt"]
        assert _unstage_sensitive_paths(project) == [".env"]
    staged = subprocess.run(["git", "diff", "--cached", "--name-only", "-z"], cwd=project, check=True,
                            capture_output=True).stdout
    assert staged == "я.txt".encode("utf-8") + b"\0", "the credential is unstaged and the other name kept"
