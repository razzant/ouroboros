from __future__ import annotations

import pathlib
import subprocess


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


def test_acceptance_gets_bounded_host_mutation_evidence(tmp_path):
    from ouroboros.mutation_attribution import capture_mutation_baseline
    from ouroboros.review_evidence import build_task_acceptance_evidence
    from ouroboros.task_results import (
        STATUS_RUNNING,
        load_task_result,
        write_task_result,
    )
    from ouroboros.tools.registry import ToolContext

    root = _repo(tmp_path)
    data = tmp_path / "data"
    write_task_result(data, "task-evidence", STATUS_RUNNING)
    capture_mutation_baseline(
        data,
        "task-evidence",
        [{"surface_type": "system_repo", "host_root": str(root)}],
    )
    evidence = dict((load_task_result(data, "task-evidence") or {})["mutation_evidence"])
    evidence["flags"] = [{
        "flag": "opaque_unregistered_effect",
        "detail": str(root / "secret-name.txt"),
    }]
    write_task_result(data, "task-evidence", STATUS_RUNNING, mutation_evidence=evidence)
    ctx = ToolContext(repo_dir=root, system_repo_dir=root, drive_root=data, task_id="task-evidence")

    packet = build_task_acceptance_evidence(
        ctx,
        drive_root=data,
        task_id="task-evidence",
    )
    projection = packet["mutation_attribution"]
    assert packet["__provenance__"]["mutation_attribution"] == "host_attested"
    assert projection["clean_eligible"] is False
    assert projection["blockers"] == ["opaque_unregistered_effect"]
    assert str(root) not in repr(projection)
    assert "secret-name.txt" not in repr(projection)


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


def test_trashed_worktree_overflow_is_a_typed_blocker(tmp_path, monkeypatch):
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
        [{"surface_type": "system_repo", "host_root": str(root)}],
    )
    candidates = ma.attributed_git_candidates(data, "task-trash", root)
    assert candidates["candidates"] == []
    assert candidates["blockers"] == ["baseline_dirty_overflow"]


def test_acceptance_evidence_aggregates_capability_deltas(tmp_path):
    """W3 adjacent (f): the finalizer's evidence packet carries ONE typed
    host-attested aggregate of capability reductions — the task's own dispatch
    delta plus every direct child that ran below what was asked — using the
    SAME disclosable predicate the absorption surfaces use. Nothing reduced =
    no section (noise-free)."""
    from ouroboros.review_evidence import build_task_acceptance_evidence
    from ouroboros.task_results import STATUS_COMPLETED, STATUS_RUNNING, write_task_result
    from ouroboros.tools.registry import ToolContext

    root = _repo(tmp_path)
    data = tmp_path / "data"
    write_task_result(
        data, "task-parent", STATUS_RUNNING,
        capability_delta={"reduced": True, "reason": "subscription_window_exhausted"},
    )
    write_task_result(
        data, "task-kid-a", STATUS_COMPLETED,
        parent_task_id="task-parent", root_task_id="task-parent",
        delegation_role="subagent",
        capability_delta={"reduced": True, "reason": "lane_ran_on_main"},
    )
    # A child whose delta took nothing away is noise and must not appear.
    write_task_result(
        data, "task-kid-b", STATUS_COMPLETED,
        parent_task_id="task-parent", root_task_id="task-parent",
        delegation_role="subagent",
        capability_delta={"reduced": False},
    )
    ctx = ToolContext(repo_dir=root, system_repo_dir=root, drive_root=data, task_id="task-parent")

    packet = build_task_acceptance_evidence(ctx, drive_root=data, task_id="task-parent")
    section = packet["capability_deltas"]
    assert packet["__provenance__"]["capability_deltas"] == "host_attested"
    assert section["own"]["reason"] == "subscription_window_exhausted"
    assert section["children_reduced_count"] == 1
    assert section["children"][0]["task_id"] == "task-kid-a"
    assert section["children"][0]["capability_delta"]["reason"] == "lane_ran_on_main"

    # Nothing reduced anywhere -> the section is absent, not an empty stub.
    write_task_result(data, "task-clean", STATUS_RUNNING)
    ctx_clean = ToolContext(repo_dir=root, system_repo_dir=root, drive_root=data, task_id="task-clean")
    clean_packet = build_task_acceptance_evidence(ctx_clean, drive_root=data, task_id="task-clean")
    assert "capability_deltas" not in clean_packet


def test_acceptance_packet_reads_mutation_evidence_from_the_canonical_root(tmp_path):
    """AP4(1): the writer and the outcome consumer resolve `budget_drive_root`
    first. Reading the EXECUTION drive made the whole section vanish from the
    packet on a split-root install, with no absence marker to notice."""
    from ouroboros.mutation_attribution import (
        capture_mutation_baseline,
        load_mutation_evidence_projection,
    )
    from ouroboros.review_evidence import build_task_acceptance_evidence
    from ouroboros.task_results import STATUS_RUNNING, write_task_result
    from ouroboros.tools.registry import ToolContext

    root = _repo(tmp_path)
    canonical = tmp_path / "canonical"
    local = tmp_path / "local"
    local.mkdir(parents=True)
    write_task_result(canonical, "task-split", STATUS_RUNNING)
    capture_mutation_baseline(
        canonical, "task-split",
        [{"surface_type": "system_repo", "host_root": str(root)}],
    )
    assert load_mutation_evidence_projection(canonical, "task-split")
    assert not load_mutation_evidence_projection(local, "task-split")

    ctx = ToolContext(
        repo_dir=root, system_repo_dir=root, drive_root=local,
        budget_drive_root=canonical, task_id="task-split",
    )
    packet = build_task_acceptance_evidence(ctx, drive_root=local, task_id="task-split")
    assert packet["mutation_attribution"]["present"] is True
    assert packet["__provenance__"]["mutation_attribution"] == "host_attested"
