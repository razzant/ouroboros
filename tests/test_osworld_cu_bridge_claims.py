"""Claim custody: who owns a task, and what may release the claim.

Split verbatim out of ``tests/test_osworld_cu_bridge.py`` by theme. This module owns the
per-task claim lock, the fail-closed scored marker that survives a dying lane, the
confinement of the claim directory, and the rule that two overlapping attempts never
share one canonical record.

These exercise the pure helpers only — no OSWorld VM, no Ouroboros server.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import pytest

from devtools.benchmarks.osworld import run_cu_bridge_agent as rcb

from tests._osworld_cu_bridge_shared import (
    _attempt_dirs,
    _cu_bridge_argv,
    _cu_bridge_stubs,
)


def test_task_claim_serializes_lanes_and_first_scored_attempt_wins(tmp_path):
    from devtools.benchmarks.osworld.run_step_agent import (
        acquire_task_claim,
        claim_stale_sec,
        release_task_claim,
        task_claim_key,
    )

    claims = tmp_path / "claims"
    key = task_claim_key("multi_apps", "48d05431-6cd5-4e76")
    stale = claim_stale_sec(3600, 900, 900)
    # stale_sec must exceed every wall-clock rail the holder can still be inside, and the
    # holder gets TWO startup windows (constructor, then reset-to-screenshot) — a one-window
    # bound expires while a lane is still legitimately working and two lanes take one task.
    # The unbounded env.evaluate() that follows is covered by the margin, not the formula.
    assert stale == 3600 + 2 * 900 + 900
    assert claim_stale_sec(3600, 900, -5) == 3600 + 2 * 900  # negative margin never shortens

    lane_a, reason_a = acquire_task_claim(claims, key, stale_sec=stale, repo_dir=tmp_path / "repo")
    assert lane_a is not None and reason_a == "claimed"
    # A second lane must NOT get the same task while the first is working.
    lane_b, reason_b = acquire_task_claim(claims, key, stale_sec=stale, repo_dir=tmp_path / "repo")
    assert lane_b is None and reason_b == "in_flight"

    # Unscored attempt -> the task stays claimable, so a retry lane may take it.
    release_task_claim(claims, key, lane_a, scored=False, repo_dir=tmp_path / "repo")
    lane_c, reason_c = acquire_task_claim(claims, key, stale_sec=stale, repo_dir=tmp_path / "repo")
    assert lane_c is not None and reason_c == "claimed"

    # Scored attempt -> permanent marker; later lanes step aside regardless of value.
    release_task_claim(claims, key, lane_c, scored=True, repo_dir=tmp_path / "repo", payload={"reward": 0.0})
    lane_d, reason_d = acquire_task_claim(claims, key, stale_sec=stale, repo_dir=tmp_path / "repo")
    assert lane_d is None and reason_d == "already_scored"
    assert (claims / f"{key}.scored").is_file()
    assert not (claims / f"{key}.lock").exists()

def test_task_claim_key_is_filesystem_safe():
    from devtools.benchmarks.osworld.run_step_agent import task_claim_key

    key = task_claim_key("multi/apps", "a b/c..json")
    assert "/" not in key and " " not in key and key.count("__") >= 1

def test_amend_task_manifest_merges_without_mutating_the_base():
    from devtools.benchmarks.osworld.run_step_agent import amend_task_manifest

    base = {"schema": "x", "output_paths": {"a": "1"}, "extra": {"allow_dirty_seed": False}}
    merged = amend_task_manifest(base, output_paths={"b": "2"}, extra={"reward": 1.0})
    assert merged["output_paths"] == {"a": "1", "b": "2"}
    assert merged["extra"] == {"allow_dirty_seed": False, "reward": 1.0}
    assert base["output_paths"] == {"a": "1"} and base["extra"] == {"allow_dirty_seed": False}

def test_cu_bridge_gates_provenance_before_the_vm_and_records_the_escape():
    """The clean-seed gate must run BEFORE paid work, not at outcome time."""
    src = (Path(__file__).resolve().parent.parent
           / "devtools" / "benchmarks" / "osworld" / "run_cu_bridge_agent.py").read_text(encoding="utf-8")
    gate = src.index("require_clean=not args.allow_dirty_seed")
    assert gate < src.index("from desktop_env.desktop_env import DesktopEnv")
    assert gate < src.index("enabled = _enable_skill(")
    assert '"allow_dirty_seed": bool(args.allow_dirty_seed)' in src
    # The per-outcome manifest amends the single early one instead of rebuilding it.
    assert "amend_task_manifest(" in src

def test_cu_bridge_claim_is_acquired_inside_the_try_that_releases_it():
    """The claim lock must not outlive a failure between claim and VM boot: an unimportable
    `desktop_env` used to leave the lock on disk with no `.scored` marker, so the task was
    neither scored nor claimable for the whole staleness window — the opposite of the
    mechanism's own 'an unscored attempt stays claimable' contract."""
    src = (Path(__file__).resolve().parent.parent
           / "devtools" / "benchmarks" / "osworld" / "run_cu_bridge_agent.py").read_text(encoding="utf-8")
    body = src[src.index("claim_fd: int | None = None"):]
    assert body.index("\n    try:") < body.index("acquire_task_claim(")
    assert body.index("acquire_task_claim(") < body.index("from desktop_env.desktop_env import DesktopEnv")
    assert body.index("from desktop_env.desktop_env import DesktopEnv") < body.index("release_task_claim(")
    # A lane that never took the lock must not delete the holder's lockfile.
    assert "if claims_dir is not None and claim_fd is not None:" in src
    # The runtime attestation admits the run before the claim and before the first paid POST
    # of the RUN FLOW. Anchored on `body` (the flow, from the claim declaration on), not the
    # whole file: module-level helpers defined above the flow (`_gate_round`) legitimately
    # contain the same POST literal but are only ever CALLED from inside the flow.
    assert src.index("runtime_attestation(args.ouroboros_url, repo_dir)") < src.index("acquire_task_claim(\n")
    first_paid_post_in_flow = src.index("claim_fd: int | None = None") + body.index('"POST", "/api/tasks"')
    assert src.index("runtime_attestation(args.ouroboros_url, repo_dir)") < first_paid_post_in_flow

def test_scored_claim_is_fail_closed_and_is_never_released_without_a_durable_marker(
        tmp_path, monkeypatch):
    """The `.scored` marker is the AUTHORITY behind "first scored attempt wins", not an
    optimisation. It used to be written inside a bare `except: pass` and the lock released
    anyway, so one disk error handed an already-scored task back to the next lane."""
    import ouroboros.utils as ouroboros_utils
    from devtools.benchmarks.osworld.run_step_agent import (
        ClaimMarkerNotDurable,
        acquire_task_claim,
        release_task_claim,
        task_claim_key,
    )

    claims = tmp_path / "claims"
    key = task_claim_key("os", "abc")
    lock_fd, reason = acquire_task_claim(claims, key, stale_sec=3600, repo_dir=tmp_path / "repo")
    assert lock_fd is not None and reason == "claimed"

    def _enospc(*_a, **_k):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(ouroboros_utils, "atomic_write_json", _enospc)
    with pytest.raises(ClaimMarkerNotDurable) as refused:
        release_task_claim(claims, key, lock_fd, scored=True, repo_dir=tmp_path / "repo", payload={"reward": 1.0})
    # Neither marker could be written, so NOTHING on disk records the score: that is the one
    # case with no honest protection left, and the refusal says so (`unconfirmed_marker is
    # None`) instead of inventing a third layer of best-effort.
    assert refused.value.unconfirmed_marker is None
    assert "claim directory is unusable" in str(refused.value)
    # Surfaced, not swallowed — AND the lock is still held, so no other attempt may take a task
    # that already has an official score while this process is alive.
    assert (claims / f"{key}.lock").exists()
    assert not (claims / f"{key}.scored").exists()
    assert not (claims / f"{key}.scored_unconfirmed").exists()
    other_fd, other_reason = acquire_task_claim(claims, key, stale_sec=3600, repo_dir=tmp_path / "repo")
    assert other_fd is None and other_reason == "in_flight"

    # With a working disk the same call marks and releases.
    monkeypatch.undo()
    release_task_claim(claims, key, lock_fd, scored=True, repo_dir=tmp_path / "repo", payload={"reward": 1.0})
    assert (claims / f"{key}.scored").is_file()
    assert not (claims / f"{key}.lock").exists()

def test_a_lane_that_dies_between_scoring_and_its_finally_keeps_the_task_scored(tmp_path):
    """Crash boundary. The marker used to be written in `finally`, AFTER `env.evaluate()` and
    the result projection, so a process death in between left no marker at all and another
    lane reran a task that already had an official score."""
    from devtools.benchmarks.osworld.run_step_agent import (
        acquire_task_claim,
        mark_task_scored,
        task_claim_key,
    )

    claims = tmp_path / "claims"
    key = task_claim_key("os", "abc")
    lock_fd, _ = acquire_task_claim(claims, key, stale_sec=3600, repo_dir=tmp_path / "repo")
    assert lock_fd is not None
    # The transition the runner performs immediately after env.evaluate()...
    mark_task_scored(claims, key, repo_dir=tmp_path / "repo", payload={"reward": 0.0})
    # ...and then the process dies: no release, no `finally`, the lock file is orphaned and
    # will look stale to the next lane. The marker still decides.
    later_fd, later_reason = acquire_task_claim(claims, key, stale_sec=0.0, repo_dir=tmp_path / "repo")
    assert later_fd is None and later_reason == "already_scored"
    # The FIRST scored attempt owns the marker; a later call never overwrites its payload.
    marker = json.loads((claims / f"{key}.scored").read_text(encoding="utf-8"))
    mark_task_scored(claims, key, repo_dir=tmp_path / "repo", payload={"reward": 1.0})
    assert json.loads((claims / f"{key}.scored").read_text(encoding="utf-8")) == marker

def test_a_scored_but_unmarked_task_stays_refused_after_its_lock_goes_stale(tmp_path, monkeypatch):
    """A protection with an expiry date fails open. `stale_sec` reclaims a crashed holder's lock
    BY DESIGN, so retaining that lock for a scored-but-unmarked task only delayed the rerun: once
    the bound elapsed, another attempt claimed a task that already had an official score. The
    durable `.scored_unconfirmed` marker refuses it regardless of staleness."""
    import ouroboros.utils as ouroboros_utils
    from devtools.benchmarks.osworld.run_step_agent import (
        ClaimMarkerNotDurable,
        acquire_task_claim,
        claim_stale_sec,
        mark_task_scored,
        scored_claim_state,
        task_claim_key,
    )

    claims = tmp_path / "claims"
    key = task_claim_key("os", "abc")
    stale = claim_stale_sec(3600, 900, 900)
    lock_fd, _ = acquire_task_claim(claims, key, stale_sec=stale, repo_dir=tmp_path / "repo")
    assert lock_fd is not None

    real_write = ouroboros_utils.atomic_write_json

    def _fail_only_the_canonical_marker(path, payload, **kwargs):
        if str(path).endswith(".scored"):
            raise OSError(28, "No space left on device")
        return real_write(path, payload, **kwargs)

    monkeypatch.setattr(ouroboros_utils, "atomic_write_json", _fail_only_the_canonical_marker)
    with pytest.raises(ClaimMarkerNotDurable) as refused:
        mark_task_scored(claims, key, repo_dir=tmp_path / "repo", payload={"reward": 0.5})
    assert refused.value.unconfirmed_marker == claims / f"{key}.scored_unconfirmed"
    monkeypatch.undo()

    # Age the lock well past the staleness bound: `acquire_exclusive_file_lock` reclaims a lock
    # whose mtime is older than `stale_sec`, which is exactly the "nobody waited long enough"
    # case the lock-only protection lost.
    lock_path = claims / f"{key}.lock"
    ancient = time.time() - (stale + 60)
    os.utime(lock_path, (ancient, ancient))
    contender_fd, contender_reason = acquire_task_claim(claims, key, stale_sec=stale, repo_dir=tmp_path / "repo")
    assert contender_fd is None and contender_reason == "scored_unconfirmed"

    # ...and it is not the lock doing the work: delete it entirely and the task is STILL refused.
    # The holder's descriptor is closed FIRST because the state being modelled is a dead holder,
    # whose descriptors the OS closed for it. It also has to be: Windows refuses to delete a file
    # while any handle to it is open (POSIX allows it), so keeping ours open fails the deletion
    # instead of testing the refusal. Same close-then-unlink order `release_exclusive_file_lock`
    # already uses.
    os.close(lock_fd)
    lock_path.unlink()
    assert scored_claim_state(claims, key) == "scored_unconfirmed"
    assert acquire_task_claim(claims, key, stale_sec=0.0, repo_dir=tmp_path / "repo") == (None, "scored_unconfirmed")
    # The reason is its own, so an operator sees a state that needs attention rather than a
    # task that silently became claimable.
    assert contender_reason not in ("in_flight", "already_scored", "claimed")

def test_the_unconfirmed_marker_does_not_disturb_the_healthy_scored_path(tmp_path):
    """The new state must refuse ONLY when it exists: a clean claim dir stays claimable even
    with a stale lock, and a properly marked task still reports `already_scored`."""
    from devtools.benchmarks.osworld.run_step_agent import (
        acquire_task_claim,
        mark_task_scored,
        release_task_claim,
        scored_claim_state,
        task_already_scored,
        task_claim_key,
    )

    claims = tmp_path / "claims"
    key = task_claim_key("os", "healthy")
    assert scored_claim_state(claims, key) == "" and task_already_scored(claims, key) is False

    lock_fd, reason = acquire_task_claim(claims, key, stale_sec=3600, repo_dir=tmp_path / "repo")
    assert lock_fd is not None and reason == "claimed"
    release_task_claim(claims, key, lock_fd, scored=True, repo_dir=tmp_path / "repo", payload={"reward": 1.0})
    assert (claims / f"{key}.scored").is_file()
    assert not (claims / f"{key}.scored_unconfirmed").exists()   # no fallback was needed
    assert scored_claim_state(claims, key) == "already_scored"
    assert acquire_task_claim(claims, key, stale_sec=3600, repo_dir=tmp_path / "repo") == (None, "already_scored")

    # A DIFFERENT task in the same claim dir is unaffected — the refusal is per-task state, not
    # a blanket on the directory — and a stale lock on it is still reclaimable as designed.
    other = task_claim_key("os", "other")
    other_fd, other_reason = acquire_task_claim(claims, other, stale_sec=3600, repo_dir=tmp_path / "repo")
    assert other_fd is not None and other_reason == "claimed"
    # The holder this reclaim is aimed at CRASHED: its lock file outlives it but its descriptors
    # do not, so ours is closed to model that. It also has to be: the reclaim unlinks the stale
    # lock, Windows refuses to unlink a file with an open handle, and that failure is swallowed
    # inside `acquire_exclusive_file_lock` — the reclaim would silently time out into `in_flight`
    # rather than raise, which is a stale lock that can never be reclaimed on that platform.
    os.close(other_fd)
    second_fd, second_reason = acquire_task_claim(claims, other, stale_sec=0.0, repo_dir=tmp_path / "repo")
    assert second_fd is not None and second_reason == "claimed"   # stale lock reclaimed
    os.close(second_fd)
    mark_task_scored(claims, other, repo_dir=tmp_path / "repo", payload={"reward": 0.0})
    assert scored_claim_state(claims, other) == "already_scored"

def test_cu_bridge_refuses_loudly_when_no_scored_state_can_be_recorded_at_all(
        tmp_path, monkeypatch, capsys):
    """The disk is genuinely gone: neither marker persists, so nothing on disk remembers the
    score and the retained lock WILL expire. There is no protection left to promise, so the
    honest outcome is a loud, distinctly-typed refusal — not a third layer of best-effort."""
    import ouroboros.utils as ouroboros_utils

    claims = tmp_path / "claims"
    rcb, _env = _cu_bridge_stubs(monkeypatch, tmp_path, reward=1.0)
    argv, results = _cu_bridge_argv(tmp_path, claims)
    monkeypatch.setattr(sys, "argv", argv)

    real_write = ouroboros_utils.atomic_write_json

    def _fail_every_claim_marker(path, payload, **kwargs):
        if ".scored" in str(path):                  # canonical AND fallback
            raise OSError(28, "No space left on device")
        return real_write(path, payload, **kwargs)

    monkeypatch.setattr(ouroboros_utils, "atomic_write_json", _fail_every_claim_marker)

    assert rcb.main() == 3                          # distinct from the ordinary failure (1/2)
    err = capsys.readouterr().err
    assert "FATAL: the claim directory is unusable" in err
    assert "do not run further tasks" in err
    extra = json.loads((results / "chrome" / "abc" / "task_run_manifest.json")
                       .read_text(encoding="utf-8"))["extra"]
    assert extra["outcome"] == "claim_state_unrecoverable"
    assert extra["exit_code"] == 3                  # == the process status
    assert extra["refusal"] == {"stage": "scored_claim_marker",
                                "reason": "claim_state_unrecoverable", "exit_code": 3}
    assert extra["claim_state_unrecoverable"] is True
    outcome = json.loads((results / "chrome" / "abc" / "task_outcome.json").read_text(encoding="utf-8"))
    assert outcome["reward"] == 1.0                 # the official score is still reported
    key = "chrome__abc"
    assert not (claims / f"{key}.scored").exists()
    assert not (claims / f"{key}.scored_unconfirmed").exists()

def test_an_interrupt_between_the_score_and_its_marker_does_not_release_the_claim(
        tmp_path, monkeypatch, capsys):
    """`KeyboardInterrupt` and `SystemExit` derive from BaseException, not Exception — the same
    trap that made a refusal handler inert in phase P1. A Ctrl-C inside `mark_task_scored` used
    to unwind straight through the `finally`, which releases the claim with `scored=False`.

    THE PART THAT ACTUALLY MATTERS IS SURVIVING THE LOCK. Retaining the `.lock` was the whole
    protection this arm used to offer, and that lock is EXPIRABLE by design: after `stale_sec`,
    `acquire_task_claim` reclaims it and reruns a task whose official score was already durably
    recorded — a genuine double count. So the refusal is asserted with the lock AGED AWAY, which
    is the only way to tell a durable protection from a countdown."""
    from devtools.benchmarks.osworld import run_step_agent
    from devtools.benchmarks.osworld.run_step_agent import (
        acquire_task_claim,
        scored_claim_state,
        task_claim_key,
    )

    claims = tmp_path / "claims"
    repo_dir = tmp_path / "repo"
    rcb, env = _cu_bridge_stubs(monkeypatch, tmp_path, reward=1.0)
    argv, _results = _cu_bridge_argv(tmp_path, claims)
    monkeypatch.setattr(sys, "argv", argv)

    def _interrupt(*_a, **_k):
        raise KeyboardInterrupt

    monkeypatch.setattr(run_step_agent, "mark_task_scored", _interrupt)

    # The retained lock is deleted below, and the lane that took it is a process on its way out
    # — the OS closes its descriptors. Recording the descriptor lets the test close it and model
    # that; on Windows it is mandatory, since a file with an open handle cannot be deleted.
    lane_fds: list[int] = []
    real_acquire = run_step_agent.acquire_task_claim

    def _recording_acquire(*a, **k):
        fd, reason = real_acquire(*a, **k)
        if fd is not None:
            lane_fds.append(fd)
        return fd, reason

    monkeypatch.setattr(run_step_agent, "acquire_task_claim", _recording_acquire)

    # The operator's interrupt still stops the run...
    with pytest.raises(KeyboardInterrupt):
        rcb.main()
    key = task_claim_key("chrome", "abc")
    # ...and the claim was NOT handed to another attempt on the way out.
    assert (claims / f"{key}.lock").exists()
    contender_fd, contender_reason = acquire_task_claim(claims, key, stale_sec=3600,
                                                        repo_dir=repo_dir)
    assert contender_fd is None and contender_reason == "scored_unconfirmed"
    assert "RETAINING the claim" in capsys.readouterr().err
    assert env.closed is True                     # the VM is still torn down on the way out

    # THE REGRESSION: the scored-but-unmarked state is on disk, and it carries the score.
    unconfirmed = json.loads((claims / f"{key}.scored_unconfirmed").read_text(encoding="utf-8"))
    assert unconfirmed["reason"] == "interrupted_before_scored_marker:KeyboardInterrupt"
    assert unconfirmed["reward"] == 1.0
    # A zero staleness bound makes the lock immediately reclaimable, and deleting it removes
    # even that. The task must STILL be refused, because the refusal never came from the lock.
    assert acquire_task_claim(claims, key, stale_sec=0.0, repo_dir=repo_dir) == (
        None, "scored_unconfirmed")
    for fd in lane_fds:
        os.close(fd)
    (claims / f"{key}.lock").unlink()
    assert scored_claim_state(claims, key) == "scored_unconfirmed"
    assert acquire_task_claim(claims, key, stale_sec=0.0, repo_dir=repo_dir) == (
        None, "scored_unconfirmed")

def test_claim_dir_is_confined_to_outside_repo_and_live_data(tmp_path, monkeypatch):
    """The claim dir is operator-supplied and the helpers CREATE it and write `.lock`,
    `.scored` and `.scored_unconfirmed` into it, so a mistaken path mutates the repository or
    the owner's live runtime data. Same boundary every other benchmark output root uses."""
    from devtools.benchmarks.osworld.run_step_agent import (
        ClaimDirNotConfined,
        acquire_task_claim,
        confined_claims_dir,
        mark_task_scored,
        task_claim_key,
    )

    repo_root = Path(__file__).resolve().parent.parent
    live_data = tmp_path / "live-data"
    live_data.mkdir()
    monkeypatch.setenv("OUROBOROS_DATA_DIR", str(live_data))
    key = task_claim_key("os", "abc")

    for bad in (repo_root / "devtools" / "claims-inside-repo",
                repo_root / ".claims",
                live_data / "state" / "claims",
                live_data):
        with pytest.raises(ClaimDirNotConfined):
            confined_claims_dir(bad, repo_dir=tmp_path / "repo")
        # ...and the refusal is enforced by the helpers that would create it, not only by the
        # CLI, so no caller can reach the filesystem around it.
        with pytest.raises(ClaimDirNotConfined):
            acquire_task_claim(bad, key, stale_sec=3600, repo_dir=tmp_path / "repo")
        with pytest.raises(ClaimDirNotConfined):
            mark_task_scored(bad, key, repo_dir=tmp_path / "repo", payload={"reward": 1.0})
        if bad != live_data:
            assert not Path(bad).exists()                # nothing was created
    assert not any(live_data.iterdir())                  # ...and nothing written into it
    # A confined dir still works exactly as before.
    good = confined_claims_dir(tmp_path / "claims", repo_dir=tmp_path / "repo")
    lock_fd, reason = acquire_task_claim(good, key, stale_sec=3600, repo_dir=tmp_path / "repo")
    assert lock_fd is not None and reason == "claimed"

def test_claim_dir_is_confined_against_the_execution_checkout_not_only_the_launcher(tmp_path):
    """INVARIANT B on the claim dir: the authority is the checkout being EXECUTED.

    `confined_claims_dir` derived its authority from this module's own location
    (`repo_root_from_devtools()`), so `--repo-dir /other/bench-clone --claim-dir
    /other/bench-clone/.claims` was waved through and the helpers wrote `.lock` and `.scored`
    state straight into the execution checkout — the very tree whose cleanliness the seed gate
    is about to attest, and which those files then dirty.

    The clone here is a SECOND checkout under tmp_path, never the ambient one, so the verdict
    is a property of the argument rather than of where the test happens to run.
    """
    from devtools.benchmarks.osworld.run_step_agent import (
        ClaimDirNotConfined,
        acquire_task_claim,
        confined_claims_dir,
        mark_task_scored,
        task_claim_key,
    )

    alt_clone = tmp_path / "other-bench-clone"
    (alt_clone / "devtools" / "benchmarks").mkdir(parents=True)
    unrelated = tmp_path / "unrelated-checkout"
    unrelated.mkdir()
    key = task_claim_key("os", "abc")

    for bad in (alt_clone / ".claims", alt_clone / "bench_runs" / "claims", alt_clone):
        with pytest.raises(ClaimDirNotConfined):
            confined_claims_dir(bad, repo_dir=alt_clone)
        # ...and by the helpers that would CREATE it, not only by the resolver, so no caller
        # can reach the filesystem around the boundary.
        with pytest.raises(ClaimDirNotConfined):
            acquire_task_claim(bad, key, stale_sec=3600, repo_dir=alt_clone)
        with pytest.raises(ClaimDirNotConfined):
            mark_task_scored(bad, key, repo_dir=alt_clone, payload={"reward": 1.0})
    assert not (alt_clone / ".claims").exists() and not (alt_clone / "bench_runs").exists()

    # THE SAME PATH is fine when a DIFFERENT checkout is the one executing: the answer depends
    # on the active checkout, which is exactly what a statically derived root cannot express.
    assert confined_claims_dir(alt_clone / ".claims", repo_dir=unrelated) == \
        (alt_clone / ".claims").resolve()
    # The launcher's own checkout stays an authority too — both are checked, not either/or.
    ambient = Path(__file__).resolve().parent.parent
    with pytest.raises(ClaimDirNotConfined):
        confined_claims_dir(ambient / "devtools" / ".claims", repo_dir=alt_clone)

def test_cu_bridge_refuses_a_claim_dir_inside_the_checkout_it_was_handed(tmp_path, monkeypatch):
    """The same defect end to end: `--claim-dir` inside `--repo-dir`. Nothing is created, and
    the refusal is pure argument validation, so it precedes admission (invariant A)."""
    _rcb, _env = _cu_bridge_stubs(monkeypatch, tmp_path)
    execution_checkout = tmp_path / "repo"           # this is what `--repo-dir` points at
    claims = execution_checkout / ".claims"
    argv, results = _cu_bridge_argv(tmp_path, claims)
    monkeypatch.setattr(sys, "argv", argv)

    with pytest.raises(SystemExit) as refused:
        rcb.main()
    assert "refusing --claim-dir" in str(refused.value)
    assert not claims.exists()
    assert not results.exists()                      # not even an admission record

def test_cu_bridge_refuses_an_unconfined_claim_dir_before_anything_is_created(
        tmp_path, monkeypatch):
    """CLI-level refusal, as pure argument validation before admission: nothing on disk."""
    claims = Path(__file__).resolve().parent.parent / "devtools" / "claims-must-not-appear"
    _rcb, _env = _cu_bridge_stubs(monkeypatch, tmp_path)
    argv, results = _cu_bridge_argv(tmp_path, claims)
    monkeypatch.setattr(sys, "argv", argv)

    with pytest.raises(SystemExit) as refused:
        rcb.main()
    assert "refusing --claim-dir" in str(refused.value)
    assert not claims.exists()
    assert not results.exists()                          # not even an admission record

def test_cu_bridge_marks_the_score_before_it_projects_the_result_anywhere():
    """Ordering is the whole mechanism: mark, THEN publish. Reversed, a crash in between
    leaves a published score with no marker — the one ordering that makes a lane rerun it."""
    src = (Path(__file__).resolve().parent.parent
           / "devtools" / "benchmarks" / "osworld" / "run_cu_bridge_agent.py").read_text(encoding="utf-8")
    evaluate = src.index("reward = float(env.evaluate())")
    mark = src.index("mark_task_scored(claims_dir, claim_key,")
    result_txt = src.index('(run_dir / "result.txt").write_text')
    projection = src.index('_write_outcome(reward, "completed"')
    assert evaluate < mark < result_txt < projection
    # ...and the release only ever claims `scored` for a marker that was CONFIRMED durable.
    assert "scored=claim_scored" in src
    assert "claim_scored = True" in src

def test_two_overlapping_attempts_never_share_one_canonical_record(tmp_path, monkeypatch, capsys):
    """The claim is only half the protection if both attempts still write the same files.

    `run_dir` is keyed by the TASK, so two lanes running the same task shared
    `run_dir/task_run_manifest.json`: both wrote their admission record there before either had
    claimed anything, and the loser then finalized `skipped_in_flight` into the file while the
    holder was still running — defeating both the claim's ownership contract and the
    append-only evidence contract. Each attempt now records into `attempts/<id>/`, and only the
    claim holder writes the canonical artefacts.
    """
    from devtools.benchmarks.osworld.run_step_agent import (
        acquire_task_claim,
        release_task_claim,
        task_claim_key,
    )

    claims = tmp_path / "claims"
    repo_dir = tmp_path / "repo"
    rcb, _env = _cu_bridge_stubs(monkeypatch, tmp_path, reward=1.0)
    argv, results = _cu_bridge_argv(tmp_path, claims)
    monkeypatch.setattr(sys, "argv", argv)
    run_dir = results / "chrome" / "abc"
    key = task_claim_key("chrome", "abc")

    # LANE A holds the task, exactly as a concurrent runner would.
    holder_fd, holder_reason = acquire_task_claim(claims, key, stale_sec=3600, repo_dir=repo_dir)
    assert holder_fd is not None and holder_reason == "claimed"

    # LANE B runs the same task and steps aside.
    assert rcb.main() == 4
    assert json.loads(capsys.readouterr().out.splitlines()[-1])["claim"] == "in_flight"
    bystander = _attempt_dirs(run_dir)
    assert len(bystander) == 1
    assert json.loads((bystander[0] / "task_run_manifest.json").read_text(
        encoding="utf-8"))["extra"]["outcome"] == "skipped_in_flight"
    # NOTHING canonical was written: not the manifest the holder will write, not the task copy,
    # not an outcome. The holder's directory is untouched by a lane that never owned it.
    assert not (run_dir / "task_run_manifest.json").exists()
    assert not (run_dir / "task.json").exists()
    assert not (run_dir / "task_outcome.json").exists()

    # LANE A crashes without scoring, so the task is claimable again (an UNSCORED attempt never
    # blocks a retry), and the next attempt wins it for real.
    release_task_claim(claims, key, holder_fd, scored=False, repo_dir=repo_dir)
    assert rcb.main() == 0

    attempts = _attempt_dirs(run_dir)
    assert len(attempts) == 2 and attempts[0] == bystander[0]     # append-only: not overwritten
    winner = json.loads((attempts[1] / "task_run_manifest.json").read_text(encoding="utf-8"))
    assert winner["extra"]["outcome"] == "completed" and winner["extra"]["claim_owner"] is True
    # The loser's terminal outcome is still its own, in its own file.
    assert json.loads((attempts[0] / "task_run_manifest.json").read_text(
        encoding="utf-8"))["extra"]["outcome"] == "skipped_in_flight"
    # ...and the canonical record belongs to the holder alone.
    canonical = json.loads((run_dir / "task_run_manifest.json").read_text(encoding="utf-8"))
    assert canonical["extra"]["outcome"] == "completed"
    assert (run_dir / "task.json").is_file() and (run_dir / "result.txt").is_file()
    assert json.loads((run_dir / "task_outcome.json").read_text(
        encoding="utf-8"))["claim_owner"] is True

def test_cu_bridge_retains_the_lock_when_the_scored_marker_will_not_persist(tmp_path, monkeypatch):
    """INTEGRATED regression for the real try/except/finally path.

    The helper-level test cannot see this: inside `_run_cu_bridge`, a `ClaimMarkerNotDurable`
    raised after `env.evaluate()` was swallowed by the broad `except Exception`, which left
    `claim_scored` False, so the `finally` released the lock and the ALREADY-EVALUATED task
    became immediately claimable again — precisely the corruption the fail-closed marker
    exists to prevent.
    """
    import ouroboros.utils as ouroboros_utils

    from devtools.benchmarks.osworld.run_step_agent import acquire_task_claim, task_claim_key

    claims = tmp_path / "claims"
    rcb, env = _cu_bridge_stubs(monkeypatch, tmp_path, reward=1.0)
    argv, results = _cu_bridge_argv(tmp_path, claims)
    monkeypatch.setattr(sys, "argv", argv)

    real_write = ouroboros_utils.atomic_write_json

    def _fail_only_the_marker(path, payload, **kwargs):
        if str(path).endswith(".scored"):
            raise OSError(28, "No space left on device")
        return real_write(path, payload, **kwargs)

    monkeypatch.setattr(ouroboros_utils, "atomic_write_json", _fail_only_the_marker)

    assert rcb.main() == 2
    key = task_claim_key("chrome", "abc")
    # THE ASSERTION: the scored-but-unmarked state is recorded DURABLY, so the refusal does not
    # depend on the lock — which `stale_sec` reclaims by design. The lock is retained too, but
    # only as interim cover.
    assert (claims / f"{key}.lock").exists()
    assert not (claims / f"{key}.scored").exists()
    unconfirmed = json.loads((claims / f"{key}.scored_unconfirmed").read_text(encoding="utf-8"))
    assert unconfirmed["reason"] == "scored_marker_write_failed"
    assert unconfirmed["reward"] == 1.0                      # the score is not lost
    contender_fd, contender_reason = acquire_task_claim(claims, key, stale_sec=3600, repo_dir=tmp_path / "repo")
    assert contender_fd is None and contender_reason == "scored_unconfirmed"
    # The official score is not thrown away, and the bookkeeping failure is disclosed.
    outcome = json.loads((results / "chrome" / "abc" / "task_outcome.json").read_text(encoding="utf-8"))
    assert outcome["reward"] == 1.0
    assert outcome["reason_code"] == "claim_marker_not_durable"
    assert outcome["claim_lock_retained"] is True
    extra = json.loads((results / "chrome" / "abc" / "task_run_manifest.json")
                       .read_text(encoding="utf-8"))["extra"]
    assert extra["outcome"] == "scored_claim_marker_failed" and extra["exit_code"] == 2
    assert extra["claim_unconfirmed_marker"].endswith(".scored_unconfirmed")
    assert env.closed is True                       # the VM is still torn down

def test_cu_bridge_releases_the_lock_and_keeps_the_marker_on_a_healthy_scored_run(
        tmp_path, monkeypatch):
    """The same integrated path when the marker DOES persist: marker kept, lock released."""
    from devtools.benchmarks.osworld.run_step_agent import acquire_task_claim, task_claim_key

    claims = tmp_path / "claims"
    rcb, _env = _cu_bridge_stubs(monkeypatch, tmp_path, reward=0.0)
    argv, results = _cu_bridge_argv(tmp_path, claims)
    monkeypatch.setattr(sys, "argv", argv)

    assert rcb.main() == 0
    key = task_claim_key("chrome", "abc")
    assert (claims / f"{key}.scored").is_file()
    assert not (claims / f"{key}.lock").exists()
    # ...and a later lane steps aside on the marker, not on the lock.
    later_fd, later_reason = acquire_task_claim(claims, key, stale_sec=3600, repo_dir=tmp_path / "repo")
    assert later_fd is None and later_reason == "already_scored"
    assert json.loads((results / "chrome" / "abc" / "result.txt").read_text(encoding="utf-8") or 0) == 0.0

def test_claim_rechecks_the_marker_after_winning_the_lock(tmp_path, monkeypatch):
    """TOCTOU: the marker was read only BEFORE waiting for the lock and never again.

    Two lanes both see no marker; the first wins the lock, scores, marks and releases; the
    second then acquires the lock with the marker already on disk and used to be told
    `claimed` — rerunning a task that already has an official score.
    """
    import ouroboros.platform_layer as platform_layer

    from devtools.benchmarks.osworld.run_step_agent import (
        acquire_task_claim,
        mark_task_scored,
        task_claim_key,
    )

    claims = tmp_path / "claims"
    key = task_claim_key("os", "abc")
    real_acquire = platform_layer.acquire_exclusive_file_lock

    def _score_while_the_contender_waits(lock_path, **kwargs):
        fd = real_acquire(lock_path, **kwargs)
        # The previous holder finished, marked and released WHILE we were blocking here.
        mark_task_scored(claims, key, repo_dir=tmp_path / "repo", payload={"reward": 1.0})
        return fd

    monkeypatch.setattr(platform_layer, "acquire_exclusive_file_lock",
                        _score_while_the_contender_waits)
    lock_fd, reason = acquire_task_claim(claims, key, stale_sec=3600, repo_dir=tmp_path / "repo")
    assert lock_fd is None and reason == "already_scored"
    # ...and the lock we took in order to look is given back, not parked for a whole window.
    assert not (claims / f"{key}.lock").exists()
    monkeypatch.undo()
    assert acquire_task_claim(claims, key, stale_sec=3600, repo_dir=tmp_path / "repo") == (None, "already_scored")
