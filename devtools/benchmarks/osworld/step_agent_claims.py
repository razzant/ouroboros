"""Cross-lane task claims and the scored-claim ledger for OSWorld runs.

Verbatim extraction from ``run_step_agent.py`` (v7 stream W): claim-directory
confinement, the claim key, staleness, acquisition and release, plus the
durable unconfirmed/scored markers that make overlapping lanes, resumes and
retry passes safe over one shared results tree (METHODOLOGY §7.9).
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

from devtools.benchmarks.common.run_roots import assert_outside_repo, repo_root_from_devtools
from devtools.benchmarks.osworld.step_agent_common import _safe_slug

class ClaimDirNotConfined(ValueError):
    """The claim directory would put lock/marker files inside repo/ or the live data root."""


def confined_claims_dir(claims_dir: Path, *, repo_dir: Path) -> Path:
    """Resolve a claim directory, REFUSING one inside ANY checkout this run touches.

    The claim dir is operator-supplied (`--claim-dir`) and the helpers below create it and
    write `.lock`, `.scored` and `.scored_unconfirmed` into it, so an unchecked path mutates a
    repository or the live runtime data. Routed through the SAME boundary every benchmark
    output root uses (`assert_outside_repo`, which also covers `live_data_roots()`), in its
    PURE form so the refusal happens before anything is created.

    ``repo_dir`` is REQUIRED and is the checkout actually being executed (`--repo-dir`, the one
    the run manifest attests). Deriving the authority from this module's own location instead —
    which is all this helper used to do — confined the claim dir against the LAUNCHER's
    checkout, so `--repo-dir /other/bench-clone --claim-dir /other/bench-clone/.claims` wrote
    lock and marker state straight into the execution checkout: the very tree whose cleanliness
    the seed gate is about to attest. Both roots are checked, active checkout first; the static
    one is belt-and-braces, never the sole authority.
    """
    resolved = Path(claims_dir)
    for authority in (Path(repo_dir).expanduser(), repo_root_from_devtools()):
        try:
            resolved = assert_outside_repo(resolved, authority)
        except ValueError as exc:
            raise ClaimDirNotConfined(f"--claim-dir is not confined: {exc}") from exc
    return resolved


def task_claim_key(domain: str, example_id: str) -> str:
    """Filesystem-safe claim identity for one OSWorld task."""
    return f"{_safe_slug(str(domain))}__{_safe_slug(str(example_id))}"


def claim_stale_sec(task_timeout_sec: float, startup_timeout_sec: float, margin_sec: float) -> float:
    """Lock staleness bound: longer than every rail the legitimate holder can be inside.

    A holder spends TWO startup windows, not one: ``construct_desktop_env`` gets its
    own ``startup_timeout`` deadline and the reset-to-usable-screenshot loop then gets
    a fresh one (sharing a single window would let a slow boot eat the reset budget).
    Adding the task timeout, the bound is ``task_timeout + 2 * startup_timeout +
    margin``. A one-window bound could expire while a lane was still legitimately
    working, which is exactly how two lanes end up on one task.

    ``env.evaluate()`` runs after all of those and is UNBOUNDED — upstream getters may
    fetch over the network — so no formula can cover it. That residual is what
    ``margin`` is for: raise ``--claim-margin-sec`` for domains with slow evaluators
    instead of widening the formula with a term nothing enforces.
    """
    return (float(task_timeout_sec) + 2.0 * float(startup_timeout_sec)
            + max(0.0, float(margin_sec)))


def acquire_task_claim(claims_dir: Path, claim_key: str, *, stale_sec: float,
                       repo_dir: Path, metadata: str = "") -> tuple[int | None, str]:
    """Claim one task for this lane. Returns ``(lock_fd, reason)``.

    ``lock_fd is None`` means DO NOT run this task; ``reason`` is one of
    ``already_scored`` (another attempt produced an official score — the "first scored attempt
    wins" rule, enforced rather than merely documented), ``scored_unconfirmed`` (a score exists
    but its canonical marker could not be persisted; see ``mark_task_scored``) or ``in_flight``
    (another attempt holds the lock). Reuses the portable O_EXCL lockfile from
    ``ouroboros.platform_layer``; no daemon, no registry, no lease.

    The scored STATE is read from markers, never from the lock, so a refusal on it is
    STALENESS-INDEPENDENT: the lock is deliberately expirable (`stale_sec` reclaims a crashed
    holder's task) and a protection built on it would fail open the moment somebody waited long
    enough. `scored_unconfirmed` therefore refuses forever, until an operator clears it.

    The state is checked TWICE, and the second check is the load-bearing one. Checking it only
    BEFORE waiting for the lock is a live TOCTOU hole: two attempts both see no marker, the
    first wins the lock, scores, marks and releases, and the second then acquires the lock with
    the marker already on disk and would still be told ``claimed`` — rerunning a task that
    already has an official score, which is the exact corruption the rule forbids. So the state
    is re-read once the lock is HELD (nobody can be mid-transition then) and the lock we just
    took is released again if the answer changed.
    """
    from ouroboros.platform_layer import acquire_exclusive_file_lock, release_exclusive_file_lock

    claims_dir = confined_claims_dir(claims_dir, repo_dir=repo_dir)
    claims_dir.mkdir(parents=True, exist_ok=True)
    state = scored_claim_state(claims_dir, claim_key)
    if state:
        return None, state
    lock_path = claims_dir / f"{claim_key}.lock"
    fd = acquire_exclusive_file_lock(
        lock_path, timeout_sec=1.0, stale_sec=stale_sec,
        metadata=metadata or f"pid={os.getpid()} ts={time.time()}\n",
    )
    if fd is None:
        return None, "in_flight"
    state = scored_claim_state(claims_dir, claim_key)
    if state:
        # Scored by the previous holder while we were blocking on the lock. Give back the lock
        # we just took — keeping it would park a task nobody may run for the whole staleness
        # window — and step aside.
        release_exclusive_file_lock(lock_path, fd)
        return None, state
    return fd, "claimed"


UNCONFIRMED_SCORE_SUFFIX = ".scored_unconfirmed"


class ClaimMarkerNotDurable(RuntimeError):
    """The permanent ``<key>.scored`` marker could not be persisted.

    "First SCORED attempt wins" (owner Q14=A) is an AUTHORITY fixed before any numbers were
    read, not an optimisation: with no marker another attempt reruns a task that already has an
    official score, and the pre-registered dedup rule is violated in the direction that
    CORRUPTS results. So a marker-persistence failure is raised rather than swallowed.

    ``unconfirmed_marker`` is the durable record of the "scored but unmarked" state — the
    ``<key>.scored_unconfirmed`` path — or ``None`` when even THAT could not be written. The
    distinction is the whole recovery story: with the marker, the refusal is permanent and
    visible; without it, nothing on disk remembers that a score exists, so retaining the
    in-flight lock is all that is left and that lock EXPIRES. The caller must then refuse
    loudly rather than pretend the task is protected.
    """

    def __init__(self, message: str, *, unconfirmed_marker: Path | None = None) -> None:
        super().__init__(message)
        self.unconfirmed_marker = unconfirmed_marker


def record_unconfirmed_score(claims_dir: Path, claim_key: str, *, repo_dir: Path, reason: str,
                             payload: dict[str, Any] | None = None) -> Path | None:
    """Durably record "this task HAS an official score that no canonical marker names".

    Returns the marker path, or ``None`` when even THIS could not be written. Never raises: it
    is called on paths whose job is to decide WHICH refusal to make, including one that is
    already unwinding a ``KeyboardInterrupt``, and a second failure there must not replace the
    operator's interrupt with a disk error.

    ``<key>.scored_unconfirmed`` is the only STALENESS-INDEPENDENT protection available once the
    canonical marker is missing: `stale_sec` reclaims the in-flight lock BY DESIGN, so a
    lock-only protection fails open the moment somebody waits long enough. This marker never
    expires, so ``scored_claim_state`` refuses the task until an operator clears it.

    Idempotent in the direction that matters: an existing canonical ``.scored`` marker means the
    score IS properly recorded, so it is returned untouched and no unconfirmed state is created.
    """
    from ouroboros.utils import atomic_write_json

    try:
        claims_dir = confined_claims_dir(claims_dir, repo_dir=repo_dir)
        marker = claims_dir / f"{claim_key}.scored"
        if marker.is_file():
            return marker
        unconfirmed = claims_dir / f"{claim_key}{UNCONFIRMED_SCORE_SUFFIX}"
        claims_dir.mkdir(parents=True, exist_ok=True)
        atomic_write_json(
            unconfirmed,
            {"claim_key": claim_key, "ts_unix": time.time(), "reason": reason,
             "canonical_marker": str(marker), **(payload or {})},
            trailing_newline=True,
            fsync=True,
        )
        return unconfirmed if unconfirmed.is_file() else None
    except BaseException:  # noqa: BLE001 - the caller ESCALATES a None, never a new exception
        return None


def mark_task_scored(claims_dir: Path, claim_key: str, *, repo_dir: Path,
                     payload: dict[str, Any] | None = None) -> Path:
    """Fail-CLOSED durable claim transition: this task HAS an official score.

    Called immediately after ``env.evaluate()`` and BEFORE the score is projected into any
    result artefact, which is what makes the rule survive a crash. The only orderings a
    process death can then produce are "marker, no result" (a later lane steps aside; the
    denominator shows the missing row) and "no marker, no result" (a later lane legitimately
    retries). "Result without marker" — the one ordering that makes a lane rerun an
    already-scored task — is unreachable.

    Written with ``fsync=True``: "durable" has to mean survived-the-power-cut, not
    reached-the-page-cache. Idempotent — an existing marker IS the first scored attempt and
    is never overwritten.

    If the canonical marker cannot be written, the "scored but unmarked" state is recorded
    DURABLY at ``<key>.scored_unconfirmed`` instead, and the refusal carries that path. Leaving
    only the in-flight lock behind was a protection with an expiry date: `stale_sec` makes that
    lock reclaimable BY DESIGN, so once enough time passed another attempt claimed a task that
    already had an official score — the same corruption, merely delayed. The marker never
    expires, so the refusal is permanent and an operator can see it.
    """
    from ouroboros.utils import atomic_write_json

    claims_dir = confined_claims_dir(claims_dir, repo_dir=repo_dir)
    marker = claims_dir / f"{claim_key}.scored"
    unconfirmed = claims_dir / f"{claim_key}{UNCONFIRMED_SCORE_SUFFIX}"
    try:
        claims_dir.mkdir(parents=True, exist_ok=True)
        if not marker.exists():
            atomic_write_json(
                marker,
                {"claim_key": claim_key, "ts_unix": time.time(), **(payload or {})},
                trailing_newline=True,
                fsync=True,
            )
        if not marker.is_file():
            raise OSError(f"scored marker is absent after a successful write: {marker}")
    except Exception as exc:  # noqa: BLE001 - re-raised as the typed fail-closed refusal
        # SECOND and LAST attempt, at a different path. Not a further layer of best-effort: it
        # decides WHICH refusal the caller must make. If it succeeds the task is permanently
        # refused by `scored_claim_state`; if it fails, nothing on disk remembers the score and
        # the caller has to say so loudly instead of promising a protection that expires.
        recorded = record_unconfirmed_score(
            claims_dir, claim_key, repo_dir=repo_dir, reason="scored_marker_write_failed",
            payload={"error": f"{type(exc).__name__}: {exc}", **(payload or {})},
        )
        if recorded is not None:
            raise ClaimMarkerNotDurable(
                f"could not persist the scored-claim marker {marker}: {type(exc).__name__}: "
                f"{exc}; recorded the scored-but-unmarked state at {recorded} instead, which "
                "refuses this task permanently (staleness cannot reclaim it)",
                unconfirmed_marker=recorded,
            ) from exc
        raise ClaimMarkerNotDurable(
            f"could not persist the scored-claim marker {marker} NOR the fallback "
            f"{unconfirmed}: {type(exc).__name__}: {exc}; the claim directory is unusable, so "
            "NOTHING on disk records that this task has an official score and the in-flight "
            "lock will expire — refuse loudly and do not continue against this claim dir",
            unconfirmed_marker=None,
        ) from exc
    return marker


def scored_claim_state(claims_dir: Path | None, claim_key: str) -> str:
    """READ-ONLY ownership question. ``""``, ``"already_scored"`` or ``"scored_unconfirmed"``.

    Deliberately pure (``exists()`` only — no mkdir, no lock, no write) so a launcher can ask it
    BEFORE admission and step aside leaving ZERO footprint. "First SCORED attempt wins" means a
    later attempt must not even write its own admission record into the winner's per-task run
    directory: the manifest path is shared between attempts, so a footprint there is a clobber.

    Neither answer involves the lock, so neither expires. ``scored_unconfirmed`` means a score
    exists but its canonical marker could not be persisted (``mark_task_scored``); it needs an
    operator, and until then the task stays refused rather than silently becoming claimable.
    """
    if claims_dir is None:
        return ""
    claims_dir = Path(claims_dir)
    if (claims_dir / f"{claim_key}.scored").exists():
        return "already_scored"
    if (claims_dir / f"{claim_key}{UNCONFIRMED_SCORE_SUFFIX}").exists():
        return "scored_unconfirmed"
    return ""


def task_already_scored(claims_dir: Path | None, claim_key: str) -> bool:
    """True when this task must not be run again — either scored state counts."""
    return bool(scored_claim_state(claims_dir, claim_key))


def release_task_claim(claims_dir: Path, claim_key: str, lock_fd: int | None, *,
                       scored: bool, repo_dir: Path,
                       payload: dict[str, Any] | None = None) -> None:
    """Release the in-flight lock; a SCORED attempt keeps its permanent marker.

    Only a scored attempt owns ``<key>.scored``. An unscored attempt (adapter error,
    preflight block, crashed lane) deliberately leaves the task claimable again so a later
    lane may retry it — "first SCORED attempt wins", not "first attempt wins".

    A scored claim is released ONLY once its marker is confirmed on disk. The marker is the
    entire mechanism that stops a rerun, so releasing the lock without it hands the task
    straight back to the next attempt; ``mark_task_scored`` raises ``ClaimMarkerNotDurable``
    instead (the write used to be wrapped in a bare ``except: pass``) and the release below
    never runs.
    """
    from ouroboros.platform_layer import release_exclusive_file_lock

    claims_dir = Path(claims_dir)
    if scored:
        mark_task_scored(claims_dir, claim_key, repo_dir=repo_dir, payload=payload)
    release_exclusive_file_lock(claims_dir / f"{claim_key}.lock", lock_fd)
