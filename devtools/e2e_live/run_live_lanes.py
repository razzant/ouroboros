#!/usr/bin/env python3
"""Live E2E stand runner: K isolated real servers, staggered, one scenario attempt each.

    python -m devtools.e2e_live.run_live_lanes --stub --lanes 2                     # $0 rehearsal
    OUROBOROS_E2E_LIVE_OPENROUTER_KEY=... python -m devtools.e2e_live.run_live_lanes \
        --lanes 4 --scenarios SM1,SW1,SK1 --attempts 3 --pass-of 2 --total-budget 100  # paid

Order is load-bearing (the benchmark family's launcher gate, ``launcher_audit``): argument-shaped work, then
``admit_benchmark_run`` over the SOURCE checkout (persisted BEFORE anything can fail; its cleanliness is
disclosed, not enforced — see ``materialize_seed``), then — inside ``finalize_run_manifest`` — the key by NAME
from the environment (never a pool file, never printed), the credit preflight ``min(key limit remaining,
account credits)``, the SEED as a clean DETACHED clone of ``--seed`` (a commit or ref of the source; never the
operator's live worktree, so concurrent edits cannot dirty it), the effective settings written from the TREE'S
DEFAULTS (D-09) with the budget knobs as settings keys (never env), and the lane pool. ``--total-budget`` is a
RUN-WIDE cap kept by ``RunBudget`` (reservation ``--per-task-usd x (root tasks + 1 with --self-mod)``,
wait-then-refuse admission PER attempt, FIFO in ``dispatch_order`` — round-robin by attempt index — and each
lane's TOTAL_BUDGET = its own reservation); ``budget_preflight`` prints the reservation table and refuses,
before any spend, a run whose attempts can never all be admitted. The run-root template is redacted (the key
value lives only in each lane's 0600 settings file and is disclosed by fingerprint). The manifest names the
model from the APPLIED settings file, not argv. Every lane leaves ``lanes/<id>_a<n>/result.json`` (checks,
digests, grants by fingerprint, settings sha256, seed describe, the lane's spend, a typed refusal on infra
failure) plus screenshots when a browser client exists; a watcher prints lane states, the running spend
against the cap, free disk on ``/`` and ``/mnt/data`` and the key headroom from an informational, bounded,
backing-off probe.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from typing import Callable

if __package__ in {None, ""}:
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from devtools.benchmarks.common.manifests import (
    admit_benchmark_run,
    finalize_run_manifest,
    model_slot_snapshot,
    openrouter_account_credits,
    openrouter_key_remaining,
    provider_credential_disclosure,
    repo_provenance,
)
from devtools.benchmarks.common.result_index import append_result_index, runtime_terminal_disclosure, task_result_row
from devtools.benchmarks.common.run_roots import assert_outside_repo, repo_root_from_devtools, run_root
from devtools.benchmarks.common.secrets import credential_fingerprint, isolated_credential_grants
from devtools.benchmarks.common.server_runner import (
    IsolatedServer,
    _api,
    _settings_json_bytes,
    absorbed_cycles_done,
    build_isolated_settings,
    seed_owner_state,
)
from devtools.e2e_live import stub_lane
from devtools.e2e_live.scenarios import SCENARIOS, LaneContext, diff_sha256, head_sha, now_iso
from devtools.e2e_live.ui_probe import resolve_ui_client
from ouroboros.provider_models import ALL_PROVIDER_CREDENTIAL_KEYS, declared_model_settings

MAX_LANES = 6
STAGGER_BOUNDS = (2.0, 3.0)
# The commit gate's hermetic pytest pass runs INSIDE each lane server and resolves ``-n auto`` to the host's
# CPU count (128 here) with no ceiling: the paid run of 2026-09-04 started >= 104 xdist workers per self-mod
# lane, three lanes at once, on a shared host. The runtime's own lever (``preflight_runner._PREFLIGHT_WORKERS_ENV``,
# floor 2 so PREFLIGHT_PARALLELISM_LOST can never trip) is set in this process and forwarded by ``IsolatedServer``
# in settings-authoritative mode (its keep-list); the budget is split evenly across the concurrent lanes so the
# whole stand stays within the shared-host rule of at most 16 pytest workers.
PREFLIGHT_WORKERS_ENV = "OUROBOROS_PREFLIGHT_TEST_WORKERS"
PREFLIGHT_WORKER_BUDGET = 16
PREFLIGHT_WORKERS_FLOOR = 2
TMPDIR_MAX_CHARS = 70          # AF_UNIX 108-byte cap on the workers' Manager socket path
DEFAULT_KEY_ENV = "OUROBOROS_E2E_LIVE_OPENROUTER_KEY"
DISK_ALERT_GIB = {"/": 40.0, "/mnt/data": 60.0}
PROBE_TIMEOUT_SEC = 8.0        # the key probe's HTTP bound: shorter than a watcher tick
WATCH_INTERVAL_MIN_SEC = 5.0   # a watcher tick below this is a hot loop, not monitoring
PROBE_MIN_INTERVAL_SEC = 60.0  # two provider requests per probe: never more often than this
PROBE_BACKOFF_MAX_SEC = 900.0  # consecutive probe failures double the wait up to here
SEED_POLICY = "detached_clone_of_ref"
PROCFS_AVAILABLE = os.path.isdir("/proc")   # the orphan scan reads /proc environ: Linux only
# A lane's TOTAL_BUDGET must stay POSITIVE: the runtime reads a non-positive value as "no finite
# global budget" (``settings_setup_contract.resolve_total_budget_usd``), the opposite of a cap.
LANE_BUDGET_FLOOR_USD = 0.01
RESERVATION_RULE = (f"max({LANE_BUDGET_FLOOR_USD:g}, per_task_usd x (root_tasks + 1 if --self-mod and the scenario absorbs else root_tasks)) "
                    "— the runtime fences each root task tree at OUROBOROS_PER_TASK_COST_USD; --self-mod adds one root for the post-task "
                    "cycle of a lane that promotes (SM1; SW1/SK1 pin it off). The lane's TOTAL_BUDGET is that reservation — the true fence")


def _log(msg: str) -> None:
    print(f"[e2e_live {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lanes", type=int, default=4, help=f"concurrent isolated servers (1..{MAX_LANES})")
    ap.add_argument("--stagger", type=float, default=2.5, help="seconds between lane starts, clamped to [2, 3]")
    ap.add_argument("--scenarios", default="SM1,SW1,SK1")
    ap.add_argument("--attempts", type=int, default=1, help="attempts per scenario (every one runs and is recorded)")
    ap.add_argument("--pass-of", type=int, default=1, help="passes needed for a scenario verdict")
    ap.add_argument("--total-budget", type=float, default=100.0,
                    help="RUN-WIDE USD cap shared by every lane (each lane's TOTAL_BUDGET is its own reservation)")
    ap.add_argument("--per-task-usd", type=float, default=8.0,
                    help="OUROBOROS_PER_TASK_COST_USD in the lane settings; an attempt reserves this x its root "
                         "tasks (+1 with --self-mod: the evolution cycle is a root under the same lane fence)")
    ap.add_argument("--task-timeout", type=int, default=1500)
    ap.add_argument("--ready-timeout", type=int, default=300)
    ap.add_argument("--profile", choices=("full", "wiring"), default="full",
                    help="full = the scenario's own enforcement; wiring = advisory review, the cheap smoke")
    ap.add_argument("--self-mod", action="store_true",
                    help="post-task evolution with a real re-exec restart (D-11); a CONFIRMED absorb is then a required "
                         "check of every lane whose scenario expects one (SM1: it lands the commit to absorb)")
    ap.add_argument("--model", default="", help="pin OUROBOROS_MODEL (paid runs only)")
    ap.add_argument("--key-env", default=DEFAULT_KEY_ENV, help="NAME of the env var carrying the OpenRouter key")
    ap.add_argument("--min-credit-usd", type=float, default=None, help="refuse below this headroom (default: --total-budget)")
    ap.add_argument("--out", default="", help="run root (default: bench_runs/e2e_live/<run id>; never repo/ or live data/)")
    ap.add_argument("--seed", default="HEAD",
                    help="commit or ref of --source-repo materialized as a clean DETACHED clone under the run root "
                         "(never a live worktree)")
    ap.add_argument("--source-repo", default="", help="checkout the seed ref is resolved in (default: this tree)")
    ap.add_argument("--stub", action="store_true", help="loopback stub model, no key, no money")
    ap.add_argument("--watch-interval", type=float, default=30.0)
    ap.add_argument("--prune-clones", action="store_true", help="delete lane clones after each lane (results stay)")
    args = ap.parse_args(argv)
    if not 1 <= args.lanes <= MAX_LANES:
        ap.error(f"--lanes must be within 1..{MAX_LANES} (R21), got {args.lanes}")
    args.stagger = min(max(float(args.stagger), STAGGER_BOUNDS[0]), STAGGER_BOUNDS[1])
    args.scenario_ids = [s.strip() for s in str(args.scenarios).split(",") if s.strip()]
    unknown = [s for s in args.scenario_ids if s not in SCENARIOS]
    if unknown or not args.scenario_ids:
        ap.error(f"unknown scenarios {unknown}; known: {sorted(SCENARIOS)}")
    if args.attempts < 1 or not 1 <= args.pass_of <= args.attempts:
        ap.error("--attempts must be >= 1 and 1 <= --pass-of <= --attempts")
    if args.stub and args.model:
        ap.error("--model cannot be combined with --stub (the stub IS the model)")
    args.min_credit_usd = float(args.total_budget if args.min_credit_usd is None else args.min_credit_usd)
    args.preflight_test_workers = max(PREFLIGHT_WORKERS_FLOOR, PREFLIGHT_WORKER_BUDGET // args.lanes)
    for name, value in (("--total-budget", args.total_budget), ("--per-task-usd", args.per_task_usd),
                        ("--min-credit-usd", args.min_credit_usd), ("--watch-interval", args.watch_interval)):
        if not math.isfinite(float(value)) or float(value) <= 0:
            ap.error(f"{name} must be a finite positive number, got {value!r} "
                     "(a non-positive TOTAL_BUDGET means NO cap; a non-positive tick is a hot loop)")
    if args.watch_interval < WATCH_INTERVAL_MIN_SEC:
        ap.error(f"--watch-interval must be >= {WATCH_INTERVAL_MIN_SEC:g}s, got {args.watch_interval!r}")
    if not str(args.seed).strip():
        ap.error("--seed must name a commit or ref")
    tmpdir = tempfile.gettempdir()
    if len(tmpdir) > TMPDIR_MAX_CHARS:
        ap.error(f"TMPDIR {tmpdir!r} is longer than {TMPDIR_MAX_CHARS} chars: the workers' AF_UNIX "
                 "socket path would overflow (export a short TMPDIR, e.g. /tmp/claude-1006/x)")
    return args


# --------------------------------------------------------------------------- #
# Effective settings: the tree's defaults + the run's explicit knobs, written ONCE
# --------------------------------------------------------------------------- #

def effective_settings(args: argparse.Namespace, key: str) -> dict:
    """The run template (D-09: the defaults of the tree under test, never the owner's live
    settings). Every model slot is written explicitly so the manifest can name it from the FILE;
    in stub mode the slots name the loopback stub, which IS the model of that run. The
    template's TOTAL_BUDGET is the RUN cap; each lane rewrites it with its own ceiling
    (``RunBudget.ceiling``: its reservation) before its server starts."""
    slots = stub_lane.STUB_MODEL_SLOTS if args.stub else declared_model_settings({})
    overrides = {
        **slots,
        "OUROBOROS_RUNTIME_MODE": "advanced",
        "OUROBOROS_MAX_WORKERS": 4,
        "TOTAL_BUDGET": float(args.total_budget),
        "OUROBOROS_PER_TASK_COST_USD": float(args.per_task_usd),
        "OUROBOROS_POST_TASK_EVOLUTION": "true" if args.self_mod else "false",
        **({"OUROBOROS_POST_TASK_EVOLUTION_CADENCE": "every_n:1"} if args.self_mod else {}),
    }
    if args.model:
        overrides["OUROBOROS_MODEL"] = str(args.model)
    if key:
        overrides["OPENROUTER_API_KEY"] = key
    return build_isolated_settings({}, **overrides)


def redacted_template(cfg: dict) -> dict:
    """The run-root copy of the template WITHOUT credential values: the key reaches disk only
    inside each lane's 0600 settings file, never in a run-level artifact."""
    return {k: v for k, v in cfg.items() if k not in ALL_PROVIDER_CREDENTIAL_KEYS}


def template_credentials(cfg: dict) -> dict:
    return {k: v for k, v in cfg.items() if k in ALL_PROVIDER_CREDENTIAL_KEYS and str(v or "").strip()}


def write_settings(path: pathlib.Path, cfg: dict) -> str:
    """0600-before-content write; returns the sha256 of the exact bytes on disk."""
    raw = _settings_json_bytes(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        if hasattr(os, "fchmod"):
            os.fchmod(fd, 0o600)   # O_CREAT's mode applies on creation only
        os.write(fd, raw)
    finally:
        os.close(fd)
    if not hasattr(os, "fchmod"):   # Windows: no fchmod; chmod after the write is the best it has
        os.chmod(path, 0o600)
    return hashlib.sha256(raw).hexdigest()


def config_sha256(cfg: dict) -> str:
    """Digest of the settings WITHOUT secret values: comparable across runs on different keys."""
    scrubbed = {k: (credential_fingerprint(v) if k in ALL_PROVIDER_CREDENTIAL_KEYS else v)
                for k, v in sorted(cfg.items())}
    return hashlib.sha256(json.dumps(scrubbed, sort_keys=True).encode("utf-8")).hexdigest()


def credit_preflight(key: str, *, timeout: float = 10.0) -> dict:
    """``min`` over both planes; None everywhere = uncapped. Numbers only, never the key."""
    limit = openrouter_key_remaining(key, timeout=int(timeout))
    credits = openrouter_account_credits(key, timeout=int(timeout))
    bounds = [v for v in (limit, credits) if v is not None]
    return {"key_limit_remaining_usd": limit, "account_credits_usd": credits,
            "remaining_usd": min(bounds) if bounds else None}


# --------------------------------------------------------------------------- #
# The run-wide money ledger
# --------------------------------------------------------------------------- #

def lane_spend(data_root: pathlib.Path) -> tuple[float, int]:
    """``(USD summed over the lane's durable llm_usage rows, rows whose cost is unknown)``. The system-E2E
    harness oracle is the reader (the same ``logs/events.jsonl`` the runtime writes); the server-level file
    carries every row of the lane — task loops, review organs, safety — verified on the first paid run
    against the ``task_results`` accounting. A row without a numeric cost is counted, never priced."""
    from tests.system_e2e import harness  # durable readers (runtime-only import, outside the audit walk)

    spent, unknown = 0.0, 0
    for row in harness.ArtifactOracle(data_root).events("llm_usage"):
        cost = row.get("cost")
        if isinstance(cost, (int, float)) and not isinstance(cost, bool):
            spent += float(cost)
        else:
            unknown += 1
    return spent, unknown


class RunBudget:
    """The RUN-WIDE ledger behind ``--total-budget`` (the first paid run gave every lane the whole cap).

    Reservation rule, per attempt: ``per_task_usd x (root_tasks + int(self_mod and absorbs))`` — the runtime fences
    each ROOT task tree at ``OUROBOROS_PER_TASK_COST_USD`` and children spend under their root's ceiling, so SW1 (one
    root, two scouts) reserves for one root, SK1 (author + dispatch) for two, and ``--self-mod`` adds one root for
    the post-task cycle of a lane that promotes (SM1 only), all under the lane fence. Admission asks, PER attempt:
    ``spent + reservation > cap`` — it can NEVER fit, refused and recorded ``not_run`` (a later, smaller attempt is
    asked on its own; nothing halts the run); otherwise it reserves only when no earlier-dispatched attempt is
    still asking (FIFO by ``dispatch_index``, see ``admit``) and ``spent + reserved(in flight) + reservation <=
    cap``, else it waits on the ledger and re-asks after every settle (the first paid run wrote SW1/SK1 off at
    t=+21 min while the blocker was two SM1 reservations in flight, not the cap). ``spent`` is re-read from the
    lanes' durable usage on every question (it only grows, so a waiter can end refused — its ``waited_sec`` is
    recorded). Each lane's TOTAL_BUDGET is its OWN reservation (``ceiling``): the ceilings in flight are disjoint
    and their sum plus the settled spend never exceeds the cap. No ``halted`` flag: ``refusals`` lists every
    refused attempt's facts, ``first_refused`` the first.
    """

    def __init__(self, cap_usd: float, per_task_usd: float, reader: Callable[[pathlib.Path], tuple[float, int]] | None = None,
                 *, self_mod: bool = False) -> None:
        self.cap, self.per_task, self.self_mod = float(cap_usd), float(per_task_usd), bool(self_mod)
        self._read = reader or lane_spend
        self._lock = threading.Condition()                         # reserve/refuse/settle wake every waiting admission
        self._pending: dict[int, str] = {}                         # dispatch index -> attempt name, while asking
        self._live: dict[tuple, tuple[pathlib.Path, float]] = {}   # job -> (data root, reservation)
        self._final: dict[tuple, tuple[float, int]] = {}           # job -> (spent, unknown rows)
        self.refusals: list[dict] = []
        self.not_run: list[str] = []

    def reservation(self, root_tasks: int, absorbs: bool = False) -> float:
        """The ONE effective ceiling of an attempt (admission, the lane's TOTAL_BUDGET and the reports carry this
        same number): ``per_task_usd x (root_tasks + int(self_mod and absorbs))``, floored, never rounded up."""
        return max(LANE_BUDGET_FLOOR_USD, self.per_task * (max(1, int(root_tasks or 1)) + int(self.self_mod and absorbs)))

    def _spent_locked(self) -> tuple[float, int]:
        rows = list(self._final.values()) + [self._read(root) for root, _reserved in self._live.values()]
        return sum((usd for usd, _n in rows), 0.0), sum(n for _usd, n in rows)

    def _reserved_locked(self) -> float:
        return sum(reserved for _root, reserved in self._live.values())

    def admit(self, job: tuple, root_tasks: int, data_root: pathlib.Path, *, dispatch_index: int,
              on_wait: Callable[[str], None] | None = None, absorbs: bool = False) -> tuple[bool, dict]:
        """FIFO by ``dispatch_index`` (``dispatch_order``'s position): an attempt reserves only when no earlier-
        dispatched attempt is still asking and the cap has room beside the reservations in flight; one that can
        never fit is refused at once and leaves the line. Without the line the freed lane's NEXT job took the lock
        before the woken waiter every time (300/300: settle in ``run_attempt``'s finally, return, the executor's
        next job, ``admit`` on the same thread), so a round overflowing the cap let later attempts leapfrog the
        waiter. No deadlock: a waiter waits only while something is in flight or an earlier attempt is asking; the
        earliest never waits on the line and, with nothing in flight, either fits or is refused; reserve, refusal
        and settle wake every waiter, the line predicate re-parks the later ones. Cost: a large head can idle lanes
        a smaller attempt would use. ``on_wait`` runs UNDER the budget lock (it must not touch the budget), told
        once, with the reason the wait begins with; ``facts["waited_sec"]`` is how long."""
        need, name, waited_from = self.reservation(root_tasks, absorbs), f"{job[0]}_a{job[1]}", None
        with self._lock:
            self._pending[dispatch_index] = name
            try:
                while True:
                    spent, unknown = self._spent_locked()
                    reserved = self._reserved_locked()
                    waited = round(time.monotonic() - waited_from, 3) if waited_from is not None else 0.0
                    facts = {"cap_usd": self.cap, "spent_usd": round(spent, 4), "reserved_usd": round(reserved, 4),
                             "reservation_usd": need, "unknown_cost_rows": unknown, "waited_sec": waited}
                    head = min(self._pending)                      # the earliest attempt still asking
                    if spent + need > self.cap:                    # can never fit: refused, this attempt only
                        self.refusals.append({"attempt": name, "reason": "budget_cap", "at": now_iso(), **facts})
                        self.not_run.append(name)
                        return False, facts
                    if head < dispatch_index or (reserved > 0 and spent + reserved + need > self.cap):
                        if waited_from is None:
                            waited_from = time.monotonic()
                            if on_wait is not None:
                                on_wait((f"waiting — behind {self._pending[head]} in dispatch order" if head < dispatch_index
                                         else f"waiting — in flight reserved ${reserved:.2f}")
                                        + f", needs ${need:.2f}, spent ${spent:.2f}, cap ${self.cap:.2f}")
                        self._lock.wait()
                        continue
                    self._live[job] = (pathlib.Path(data_root), need)
                    return True, facts
            finally:                                               # out of the line, even on a reader error: the next may reserve
                del self._pending[dispatch_index]
                self._lock.notify_all()

    def ceiling(self, job: tuple) -> float:
        """The lane's TOTAL_BUDGET: its own reservation — immutable, disjoint from every other
        lane's, never non-positive (the runtime reads a non-positive budget as NO cap)."""
        with self._lock:
            entry = self._live.get(job)
            return entry[1] if entry else LANE_BUDGET_FLOOR_USD   # exactly what admission reserved

    def settle(self, job: tuple) -> None:
        with self._lock:
            entry = self._live.pop(job, None)
            if entry is not None:
                self._final[job] = self._read(entry[0])
            self._lock.notify_all()                                # every waiter re-asks against the new ledger

    def snapshot(self) -> dict:
        with self._lock:
            spent, unknown = self._spent_locked()
            return {"cap_usd": self.cap, "per_task_usd": self.per_task, "reservation_rule": RESERVATION_RULE,
                    "spent_usd": round(spent, 4), "reserved_usd": round(self._reserved_locked(), 4),
                    "unknown_cost_rows": unknown, "lanes_in_flight": len(self._live), "lanes_settled": len(self._final),
                    "refusals": [dict(r) for r in self.refusals],
                    "first_refused": self.refusals[0]["attempt"] if self.refusals else None,
                    "attempts_not_run": list(self.not_run)}


def budget_preflight(budget: RunBudget, scenario_ids: list[str], attempts: int, lanes: int) -> dict:
    """The reservation arithmetic BEFORE any lane spends, in ``credit_preflight``'s typed shape: a row per
    scenario, the worst case ``sum(reservation x attempts)`` against the cap, the per-ROUND worst case (the
    ``lanes`` largest reservations, ONE attempt per scenario — above the cap the round's last lane WAITS on
    a settle, no refusal; with ``lanes`` above the scenario count it understates what admission can put in
    flight) and ``unreachable``: a reservation above the cap, or equal to it with attempts >= 2 (the second
    can never be admitted after any spend). No override flag: the operator changes the flags."""
    rows = []
    for sid in scenario_ids:
        need = budget.reservation(SCENARIOS[sid].root_tasks, SCENARIOS[sid].expects_absorb)
        rows.append({"scenario": sid, "root_tasks": SCENARIOS[sid].root_tasks, "reservation_usd": need,
                     "attempts": int(attempts), "worst_case_usd": round(need * attempts, 4),
                     "unreachable": need > budget.cap or (attempts >= 2 and need >= budget.cap)})
    return {"cap_usd": budget.cap, "per_task_usd": budget.per_task, "self_mod": budget.self_mod,
            "reservation_rule": RESERVATION_RULE, "scenarios": rows,
            "worst_case_usd": round(sum(r["worst_case_usd"] for r in rows), 4), "lanes": int(lanes),
            "round_worst_case_usd": round(sum(sorted((r["reservation_usd"] for r in rows), reverse=True)[:int(lanes)]), 4),
            "unreachable": [r["scenario"] for r in rows if r["unreachable"]]}


def dispatch_order(budget: RunBudget, requested: list[tuple[str, int]]) -> list[tuple[str, int]]:
    """Round-robin by attempt (a1 of every scenario, then a2, ...), largest reservation first within a round
    (stable among equals); admission keeps this order (``RunBudget.admit``). The verdict is pass-of PER
    scenario, so the MINIMUM admitted attempts per scenario is what the order protects, not the sum."""
    return sorted(requested, key=lambda job: (job[1], -budget.reservation(SCENARIOS[job[0]].root_tasks, SCENARIOS[job[0]].expects_absorb)))


# --------------------------------------------------------------------------- #
# The watcher's key probe: bounded, informational, never on the tick's path
# --------------------------------------------------------------------------- #

class KeyProbe:
    """Key headroom on its own thread with a bounded HTTP timeout. The first paid run's watcher probed inline
    and reported ``RemoteDisconnected``/``TimeoutError`` twice while every lane was healthy: a failed or slow
    probe is INFORMATIONAL (the lanes' spend is the ledger's business), never an ALERT and never a delay of
    the tick. ALERT only on a GOOD reading under the floor. Bounded cadence: never more often than
    PROBE_MIN_INTERVAL_SEC (two provider requests per probe), and consecutive failures back off exponentially."""

    def __init__(self, probe: Callable[[], float | None], *, floor: float, interval: float,
                 stop: threading.Event) -> None:
        self._probe, self.floor, self._stop = probe, float(floor), stop
        self.interval = max(float(interval), PROBE_MIN_INTERVAL_SEC)
        self._lock = threading.Lock()
        self.remaining: float | None = None
        self.read_at = 0.0
        self.error = ""
        self.failures = 0

    def seed(self, remaining: float | None) -> "KeyProbe":
        with self._lock:
            self.remaining, self.read_at = remaining, time.time()
        return self

    def start(self) -> "KeyProbe":
        threading.Thread(target=self._loop, daemon=True).start()
        return self

    def next_wait(self) -> float:
        with self._lock:
            failures = self.failures
        return min(self.interval * (2 ** failures), PROBE_BACKOFF_MAX_SEC)

    def _loop(self) -> None:
        while not self._stop.wait(self.next_wait()):
            self.poll_once()

    def poll_once(self) -> None:
        try:
            value = self._probe()
        except Exception as exc:  # noqa: BLE001 - a failed probe is a reported fact, never fatal
            with self._lock:
                self.error, self.failures = f"{type(exc).__name__}: {exc}"[:120], self.failures + 1
            return
        with self._lock:
            self.remaining, self.read_at, self.error, self.failures = value, time.time(), "", 0

    def fragment(self) -> str:
        with self._lock:
            remaining, read_at, error = self.remaining, self.read_at, self.error
        if not read_at:
            return f"key probe failed: {error} (informational)" if error else "key probe pending"
        text = f"key remaining ${remaining:.2f}" if remaining is not None else "key uncapped"
        if remaining is not None and remaining < self.floor:
            text += " ALERT"
        if error:
            text += f" ({time.time() - read_at:.0f}s old; last probe failed: {error}; informational)"
        return text


# --------------------------------------------------------------------------- #
# Seed: a clean detached clone of the requested commit, never a live worktree
# --------------------------------------------------------------------------- #

class SeedMaterializeRefused(RuntimeError):
    def __init__(self, reason: str, message: str) -> None:
        super().__init__(message)
        self.reason = reason


def materialize_seed(source: pathlib.Path, ref: str, seed: pathlib.Path) -> dict:
    """A clean DETACHED clone of ``ref`` (resolved in ``source``) at ``seed``: the tree every lane clones.
    Never the operator's checkout itself — the first paid run seeded from a live worktree edited concurrently
    and SK1_a3 recorded ``seed_clean=false`` — so the source may be dirty or move under the run without
    touching what is under test. Post-admission by design (a clone is world-shaped work); the manifest's
    ``source`` block discloses the source's own state, ``manifest["seed"]`` records what actually ran."""
    resolved = subprocess.run(["git", "rev-parse", "--verify", "--quiet", f"{ref}^{{commit}}"],
                              cwd=str(source), check=False, capture_output=True, text=True)
    sha = (resolved.stdout or "").strip()
    if resolved.returncode != 0 or not sha:
        raise SeedMaterializeRefused("ref_unresolved", f"--seed {ref!r} does not name a commit in {source}")
    if seed.exists():
        raise SeedMaterializeRefused("seed_dir_exists", f"seed directory already exists: {seed}")
    try:
        subprocess.run(["git", "clone", "--no-hardlinks", "--no-checkout", "-q", str(source), str(seed)],
                       check=True, capture_output=True)
        subprocess.run(["git", "checkout", "-q", "--detach", sha], cwd=str(seed), check=True, capture_output=True)
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or b"").decode("utf-8", errors="replace").strip()[:300]
        raise SeedMaterializeRefused("clone_failed", f"could not materialize {sha[:12]} from {source}: {detail}") from exc
    subprocess.run(["git", "remote", "remove", "origin"], cwd=str(seed), check=False, capture_output=True)
    return {"policy": SEED_POLICY, "requested_ref": ref, "resolved_sha": sha,
            "seed_dir": str(seed), "source_repo": str(source)}


def seed_is_clean(provenance: dict, resolved_sha: str) -> bool:
    return (bool(provenance.get("git_available")) and bool(provenance.get("status_available"))
            and not provenance.get("dirty") and str(provenance.get("head") or "") == resolved_sha)


# --------------------------------------------------------------------------- #
# Self-modification: the absorb/re-exec must be CONFIRMED, not assumed
# --------------------------------------------------------------------------- #

def self_mod_snapshot(server: IsolatedServer, clone: pathlib.Path, data_root: pathlib.Path) -> dict:
    """Taken BEFORE the task: the clone HEAD, the served sha/uptime and the absorbed-cycle
    counter a confirmed absorb must move away from."""
    try:
        st = _api(server.base_url, "GET", "/api/state", timeout=10)
    except Exception:  # noqa: BLE001 - a missing snapshot is a recorded gap the confirmation fails on
        st = {}
    return {"head": head_sha(clone), "sha": str(st.get("sha") or ""), "uptime": int(st.get("uptime") or 0),
            "cycles": absorbed_cycles_done(data_root), "at": time.time(), "state_read": bool(st)}


def _newest_transaction(data_root: pathlib.Path) -> dict:
    path = pathlib.Path(data_root) / "state" / "evolution_campaign.json"
    try:
        campaign = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    history = campaign.get("transaction_history") if isinstance(campaign, dict) else None
    rows = [tx for tx in (history or []) if isinstance(tx, dict)]
    return rows[-1] if rows else {}


def confirm_absorb(server: IsolatedServer, clone: pathlib.Path, data_root: pathlib.Path, pre: dict, *,
                   timeout: float, ready_timeout: float) -> dict:
    """POSITIVE evidence of an absorbed post-task evolution, or a typed non-confirmation.
    ``wait_for_absorb`` answers ``absorbed=False`` typed (``absorb_idle_reason`` or ``timeout``), and a runner that
    only checks liveness afterwards lets ``--self-mod`` PASS with no restart at all. Confirmed means
    ALL of: the campaign's absorbed-cycle counter advanced past the pre-task snapshot, the served sha
    moved off the snapshot (``wait_for_absorb``'s own condition), the server re-exec'd (``/api/state``
    uptime reset below the time elapsed since the snapshot) and answers ready again. The clone HEAD
    after the cycle and the newest campaign transaction (commit sha, restart_verified, verified_by)
    are the recorded diagnostic trail; ``serving_head`` says whether the served sha is that HEAD."""
    wait = server.wait_for_absorb(pre["sha"], pre["cycles"], timeout=timeout)
    healthy = server.wait_for_health(timeout=ready_timeout)
    try:
        st = _api(server.base_url, "GET", "/api/state", timeout=10)
    except Exception:  # noqa: BLE001 - an unreadable post-state is a failed confirmation, recorded typed
        st = {}
    elapsed = time.time() - float(pre.get("at") or time.time())
    uptime = int(st.get("uptime") or 0) if st else None
    restarted = uptime is not None and uptime < int(pre.get("uptime") or 0) + elapsed - 2
    cycles = absorbed_cycles_done(data_root)
    head_after = head_sha(clone)
    served = str(st.get("sha") or "")
    tx = _newest_transaction(data_root)
    confirmed = bool(pre.get("state_read")) and bool(wait.get("absorbed")) and cycles > int(pre["cycles"]) \
        and restarted and healthy
    if confirmed:
        reason = "absorbed"
    elif not pre.get("state_read"):
        reason = "pre_snapshot_unavailable"
    elif not wait.get("absorbed") or cycles <= int(pre["cycles"]):
        reason = str(wait.get("reason") or "cycle_not_absorbed")
    elif not healthy:
        reason = "unhealthy"
    else:
        reason = "not_restarted"
    return {"confirmed": confirmed, "reason": reason, "pre": dict(pre),
            "post": {"head": head_after, "sha": served, "uptime": uptime, "cycles": cycles,
                     "elapsed_sec": round(elapsed, 1)},
            "head_moved": bool(head_after) and head_after != str(pre.get("head") or ""),
            "serving_head": bool(served) and head_after.startswith(served),
            "wait": wait, "healthy": healthy, "restarted": restarted,
            "transaction": {"commit_sha": str(tx.get("commit_sha") or ""), "cycle_outcome": str(tx.get("cycle_outcome") or ""),
                            "restart_verified": bool(tx.get("restart_verified")), "verified_by": str(tx.get("verified_by") or "")}}


# --------------------------------------------------------------------------- #
# Lane pool
# --------------------------------------------------------------------------- #

class Stagger:
    def __init__(self, seconds: float) -> None:
        self.seconds, self._last, self._lock = float(seconds), 0.0, threading.Lock()

    def wait_turn(self) -> None:
        with self._lock:
            time.sleep(max(0.0, self._last + self.seconds - time.monotonic()))
            self._last = time.monotonic()


def clone_seed(seed: pathlib.Path, clone: pathlib.Path) -> None:
    subprocess.run(["git", "clone", "--no-hardlinks", "-q", str(seed), str(clone)], check=True, capture_output=True)
    subprocess.run(["git", "checkout", "-q", "-B", "ouroboros"], cwd=str(clone), check=True, capture_output=True)
    subprocess.run(["git", "remote", "remove", "origin"], cwd=str(clone), check=False, capture_output=True)
    subprocess.run(["git", "config", "user.name", "Ouroboros E2E stand"], cwd=str(clone), check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "e2e-live@ouroboros.invalid"], cwd=str(clone), check=True, capture_output=True)


def _lane_row(job: tuple[str, int], args: argparse.Namespace) -> dict:
    sid, attempt = job
    return {"schema": "ouroboros.e2e_live.lane_result.v1", "scenario": sid, "attempt": attempt,
            "title": SCENARIOS[sid].title, "status": "infra_error", "stub": bool(args.stub), "profile": args.profile,
            "self_mod": bool(args.self_mod), "preflight_test_workers": int(args.preflight_test_workers),
            "started_at": now_iso(), "checks": {}, "facts": {}, "error": "",
            "screenshots": [], "ui": {"available": False, "reason": ""}, "budget": {},
            "self_mod_absorb": {"expected": bool(args.self_mod) and SCENARIOS[sid].expects_absorb}}


def _proc_cmdline(pid: int) -> str:
    try:
        raw = pathlib.Path(f"/proc/{pid}/cmdline").read_bytes()
    except OSError:
        return ""
    return raw.replace(b"\0", b" ").decode("utf-8", "replace").strip()[:120]


def _apply_orphan_scan(row: dict, survivors: list | None) -> None:
    """The last check of a lane: a process still carrying the lane's data root after stop fails a passing
    lane, and the index never carries a failed row with an empty reason. ``survivors`` are the pids the
    scan still found after the stop wait (empty = clean); ``None`` = no procfs: a typed fact, not a passed
    check. Survivors are NAMED in ``row["orphans"]`` (pid + cmdline head, first 20): the rc.14 paid run
    recorded a bare ``no_orphans_after_stop=false`` with no way to tell which processes outlived the stop."""
    if survivors is None:
        row["no_orphans_after_stop"] = None
        row["orphan_scan"] = "unavailable:no_procfs"
        return
    row["no_orphans_after_stop"] = gone = not survivors
    if not gone:
        row["orphans"] = [{"pid": int(pid), "cmdline": _proc_cmdline(int(pid))} for pid in survivors[:20]]
        if len(survivors) > 20:
            row["orphans_omitted"] = len(survivors) - 20
        if row["status"] == "pass":
            row["status"], row["reason_code"] = "fail", "checks_failed"
    row["checks"]["no_orphans_after_stop"] = gone


def _record_row(out: pathlib.Path, lane: pathlib.Path, row: dict) -> None:
    lane.mkdir(parents=True, exist_ok=True)
    (lane / "result.json").write_text(json.dumps(row, indent=2, ensure_ascii=False), encoding="utf-8")
    append_result_index(out, task_result_row(
        benchmark="e2e_live", instance_id=f"{row['scenario']}_a{row['attempt']}", status=row["status"],
        reason_code=str(row.get("reason_code") or ""), runtime_result=row.get("runtime_outcome"), error=row["error"],
        details={"checks": row["checks"], "duration_sec": row.get("duration_sec"), "budget": row.get("budget"),
                 "refusal": row.get("refusal")}))


def run_attempt(job: tuple[str, int], args: argparse.Namespace, out: pathlib.Path, template: dict,
                stagger: Stagger, states: dict, seed: pathlib.Path, budget: RunBudget, *,
                dispatch_index: int, key: str = "", seed_sha: str = "") -> dict:
    """Budget admission around one lane: reserve (waiting out an earlier attempt still asking, or the
    reservations in flight when the attempt fits the cap but not them yet), run, settle. A refused
    attempt is a recorded ``not_run`` row — never a silent gap in the index."""
    sid, attempt = job
    lane = out / "lanes" / f"{sid}_a{attempt}"

    def waiting(msg: str) -> None:
        _log(f"{sid}_a{attempt}: {msg}")
        states[job] = ("waiting (budget)", time.time())

    admitted, facts = budget.admit(job, SCENARIOS[sid].root_tasks, lane / "data", dispatch_index=dispatch_index,
                                   on_wait=waiting, absorbs=SCENARIOS[sid].expects_absorb)
    if not admitted:
        row = {**_lane_row(job, args), "status": "not_run", "reason_code": "budget_cap", "budget": facts,
               "refusal": {"type": "RunBudgetCap", "code": "budget_cap", "message": "run-wide budget cap reached"},
               "ended_at": now_iso(), "duration_sec": 0.0}
        _log(f"{sid}_a{attempt}: not run — run budget cap: spent ${facts['spent_usd']:.2f} + reserved "
             f"${facts['reserved_usd']:.2f} + needed ${facts['reservation_usd']:.2f} > cap ${facts['cap_usd']:.2f}"
             + (f" (after waiting {facts['waited_sec']:.0f}s)" if facts["waited_sec"] else ""))
        _record_row(out, lane, row)
        states[job] = ("not_run (budget cap)", time.time())
        return row
    try:
        return run_lane(job, args, out, template, stagger, states, seed, budget, key=key, seed_sha=seed_sha)
    finally:
        budget.settle(job)


def run_lane(job: tuple[str, int], args: argparse.Namespace, out: pathlib.Path, template: dict,
             stagger: Stagger, states: dict, seed: pathlib.Path, budget: RunBudget, *,
             key: str = "", seed_sha: str = "") -> dict:
    sid, attempt = job
    scenario = SCENARIOS[sid]
    lane = out / "lanes" / f"{sid}_a{attempt}"
    clone, data_root, shots = lane / "clone", lane / "data", lane / "shots"
    settings_path = data_root / "settings.json"
    started = time.time()
    row = _lane_row(job, args)

    def log(msg: str) -> None:
        _log(f"{sid}_a{attempt}: {msg}")
        states[job] = (msg[:60], time.time())

    server = stub = ctx = None
    # The absorb wait and check follow ``Scenario.expects_absorb`` (rc.15 SK1_a1: a lane that commits nothing
    # waited --task-timeout for a promotion that could not happen, then failed the check by construction).
    absorb, absorbing = None, row["self_mod_absorb"]["expected"]
    from tests.system_e2e import harness  # durable readers + /proc oracles (runtime-only import)
    try:
        log("cloning seed")
        lane.mkdir(parents=True, exist_ok=True)
        clone_seed(seed, clone)
        # The clone IS the admitted seed: the exact sha the manifest names, clean at start.
        pre_head = head_sha(clone)
        row["checks"]["clone_at_seed_sha"] = bool(seed_sha) and pre_head == seed_sha
        row["checks"]["clone_clean_at_start"] = subprocess.run(
            ["git", "status", "--porcelain"], cwd=str(clone), check=False, capture_output=True, text=True).stdout == ""
        cfg = dict(template)
        if key:  # the key reaches disk ONLY here, in this lane's 0600 file
            cfg["OPENROUTER_API_KEY"] = key
        child_model = str(cfg.get("OUROBOROS_MODEL") or "")
        if args.stub:
            stub = stub_lane.routed_stub_model(scenario.stub_script(clone)).__enter__()
            child_model = stub_lane.STUB_CHILD_SLUG
            cfg = stub_lane.stub_settings(stub, template)
        cfg.update(scenario.overrides(child_model))
        if args.profile == "wiring":
            cfg["OUROBOROS_REVIEW_ENFORCEMENT"] = "advisory"
        # The lane's ceiling is its own reservation: disjoint from the other lanes', never the whole cap.
        cfg["TOTAL_BUDGET"] = budget.ceiling(job)
        row["budget"] = {"reservation_usd": budget.reservation(scenario.root_tasks, scenario.expects_absorb),
                         "lane_total_budget_usd": cfg["TOTAL_BUDGET"], "per_task_usd": float(args.per_task_usd)}
        sha = write_settings(settings_path, cfg)
        # Owner id only, never a campaign: the task's post-task promotion enables the one-shot one (rc.15 run2 pin).
        seed_owner_state(data_root, evolution_enabled=False)
        from supervisor import state as sstate
        (data_root / sstate.ISOLATED_BENCHMARK_SENTINEL).write_text("isolated e2e_live data root\n", encoding="utf-8")
        oracle = harness.ArtifactOracle(data_root)

        def start_server(expected_sha: str) -> IsolatedServer:
            srv = IsolatedServer(clone, data_root, settings_path, settings_authoritative_env=True,
                                 expected_settings_sha256=expected_sha)
            stagger.wait_turn()
            log(f"starting server on {srv.base_url}")
            srv.start(ready_timeout=args.ready_timeout)
            return srv

        def wait_absorb() -> dict:
            log("waiting for the evolve/absorb re-exec")
            result = confirm_absorb(server, clone, data_root, pre_mod, timeout=args.task_timeout,
                                    ready_timeout=args.ready_timeout)
            log(f"absorb: {result['reason']}")
            return result

        def restart() -> IsolatedServer:
            nonlocal server, absorb
            if absorbing:
                absorb = wait_absorb()
                if not absorb["healthy"]:
                    raise RuntimeError("server unhealthy after the self-mod restart")
                return server
            log("restarting server on the committed tree")
            server.stop()
            server = start_server(hashlib.sha256(settings_path.read_bytes()).hexdigest())
            return server

        server = start_server(sha)
        row["attestation"] = dict(server.attestation)
        if scenario.needs_ui:
            # An AVAILABILITY probe only (the browser launches), closed at once: the scenario's client is opened
            # by ``LaneContext.ui`` at use time, after the task.
            probe, reason = resolve_ui_client(server.base_url)
            if probe is not None:
                probe.close()
            row["ui"] = {"available": probe is not None, "reason": reason}
        # The absorb snapshot is taken BEFORE the task (at restart time it would already see the task's own
        # commit and the evolve cycle it triggered).
        pre_mod = self_mod_snapshot(server, clone, data_root) if absorbing else {}
        ctx = LaneContext(server=server, clone=clone, data_root=data_root, oracle=oracle, harness=harness,
                          ui_resolver=resolve_ui_client if scenario.needs_ui else None,
                          ui_reason=row["ui"]["reason"], shots=shots, log=log,
                          task_timeout=args.task_timeout, restart=restart)
        log("running scenario")
        scenario.acceptance(ctx)
        server = ctx.server
        row["checks"].update(ctx.checks)
        row.update({"facts": ctx.facts, "screenshots": ctx.screenshots})
        if absorbing:
            if absorb is None:  # an absorbing scenario that never restarts still owes the post-task absorb
                absorb = wait_absorb()
            row["self_mod_absorb"].update(absorb)
            row["checks"]["self_mod_absorb_confirmed"] = bool(absorb["confirmed"])
            row["facts"]["self_mod_absorb_reason"] = absorb["reason"]
        seed_desc = repo_provenance(seed)
        row["checks"]["seed_clean"] = not seed_desc.get("dirty") and bool(seed_desc.get("status_available"))
        post_head = head_sha(clone)
        applied = json.loads(settings_path.read_text(encoding="utf-8"))
        row["digests"] = {"settings_sha256": hashlib.sha256(settings_path.read_bytes()).hexdigest(),
                          "settings_config_sha256": config_sha256(applied),
                          "seed_head": seed_desc.get("head", ""), "seed_describe": seed_desc.get("describe", ""),
                          "pre_head": pre_head, "post_head": post_head,
                          "diff_sha256": diff_sha256(clone, pre_head, post_head)}
        row["model_slots"] = model_slot_snapshot(settings_path, env_overrides=False)
        row["grants"] = isolated_credential_grants(applied)
        row["runtime_outcome"] = runtime_terminal_disclosure(ctx.facts.pop("runtime_result", None))
        if args.stub:
            row["facts"]["stub_unconsumed"] = stub.consumed()
            kinds = stub.kinds()  # the review-organ branches the stub actually served
            row["facts"]["stub_call_kinds"] = {kind: kinds.count(kind) for kind in sorted(set(kinds))}
        row["status"] = "pass" if all(row["checks"].values()) else "fail"
        row["reason_code"] = "" if row["status"] == "pass" else "checks_failed"
    except Exception as exc:  # noqa: BLE001 - an infra failure is a recorded row, never a lost lane
        # A typed refusal, not only a flattened string: the exception type, the code the raiser
        # carries (``reason`` on the stand's own refusals) and a bounded message.
        code = str(getattr(exc, "reason", "") or type(exc).__name__)
        row["refusal"] = {"type": type(exc).__name__, "code": code, "message": str(exc)[:2000]}
        row["reason_code"] = f"infra_error:{code}"
        row["error"] = f"{type(exc).__name__}: {exc}"[:2000]
        log(f"infra error: {row['error'][:200]}")
    finally:
        if ctx is not None:
            ctx.close_ui()
            if scenario.needs_ui:
                row["ui"] = {"available": not ctx.ui_reason, "reason": ctx.ui_reason}
        if server is not None:
            server.stop()
        if stub is not None:
            stub.__exit__(None, None, None)
        survivors = None
        if PROCFS_AVAILABLE:
            gone = harness.wait_until(lambda: not harness.pids_with_env_value(str(data_root)), 30)
            survivors = [] if gone else harness.pids_with_env_value(str(data_root))
        _apply_orphan_scan(row, survivors)
        row["budget"]["spent_usd"], row["budget"]["unknown_cost_rows"] = lane_spend(data_root)
        row["ended_at"], row["duration_sec"] = now_iso(), round(time.time() - started, 1)
        _record_row(out, lane, row)
        if args.prune_clones:
            shutil.rmtree(clone, ignore_errors=True)
        states[job] = (f"{row['status']} ({row.get('duration_sec')}s)", time.time())
    return row


def watcher(stop: threading.Event, states: dict, interval: float, budget: RunBudget,
            probe: KeyProbe | None) -> None:
    started = time.time()
    while not stop.wait(interval):
        lanes = " ".join(f"{sid}_a{n}={txt}" for (sid, n), (txt, _) in sorted(states.items()))
        disks = []
        for mount, alert in DISK_ALERT_GIB.items():
            if not pathlib.Path(mount).exists():
                continue
            free = shutil.disk_usage(mount).free / 2**30
            disks.append(f"{mount}={free:.0f}G" + (" ALERT" if free < alert else ""))
        snap = budget.snapshot()
        spend = (f"spent ${snap['spent_usd']:.2f}/${snap['cap_usd']:.2f} reserved ${snap['reserved_usd']:.2f}"
                 + (f" unknown-cost rows {snap['unknown_cost_rows']}" if snap["unknown_cost_rows"] else "")
                 + (f" not_run {len(snap['attempts_not_run'])}" if snap["attempts_not_run"] else ""))
        line = f"t=+{time.time() - started:.0f}s lanes: {lanes or '-'} | {spend} | free {' '.join(disks)}"
        if probe is not None:
            line += " | " + probe.fragment()
        _log("[watch] " + line)


# --------------------------------------------------------------------------- #
# main: admission is the outer boundary; everything else inside the finalization seam
# --------------------------------------------------------------------------- #

def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    source = pathlib.Path(args.source_repo).expanduser().resolve(strict=False) if args.source_repo \
        else repo_root_from_devtools()
    out = pathlib.Path(args.out).expanduser() if args.out else run_root("e2e_live")
    out = assert_outside_repo(out, source)
    seed = out / "seed"
    budget = RunBudget(args.total_budget, args.per_task_usd, self_mod=args.self_mod)
    requested = [(sid, n) for sid in args.scenario_ids for n in range(1, args.attempts + 1)]
    jobs = dispatch_order(budget, requested)   # the pool's order AND the ledger's: admission is FIFO by this index
    manifest_path = out / "run_manifest.json"
    # The SOURCE is attested for provenance only (require_clean=False discloses its state): the tree under test
    # is the detached seed materialized from a COMMITTED sha inside the seam (its own gate; manifest["seed"]).
    manifest = admit_benchmark_run(
        manifest_path, benchmark="e2e_live", run_root=out, repo_dir=source,
        requested_task_ids=[f"{sid}_a{n}" for sid, n in requested], require_clean=False,
        settings_authoritative_env=True, isolated_data_root=str(out / "lanes"),
        output_paths={"lanes": str(out / "lanes"), "seed": str(seed), "result_index": str(out / "result_index.jsonl"),
                      "effective_settings": str(out / "effective_settings.json")},
        extra={"outcome": "started", "lanes": args.lanes, "stagger_sec": args.stagger, "profile": args.profile,
               "self_mod": bool(args.self_mod), "stub": bool(args.stub), "scenarios": args.scenario_ids,
               "attempts": args.attempts, "pass_of": args.pass_of, "total_budget_usd": args.total_budget,
               "per_task_usd": args.per_task_usd, "key_env": args.key_env if not args.stub else "",
               "seed_ref": str(args.seed), "seed_policy": SEED_POLICY, "source_repo": str(source),
               "preflight_test_workers": args.preflight_test_workers},
    )
    _log(f"run root: {out}")
    with finalize_run_manifest(manifest_path, manifest) as final:
        preflight = budget_preflight(budget, args.scenario_ids, args.attempts, args.lanes)
        final["budget_preflight"] = preflight
        for r in preflight["scenarios"]:
            _log(f"reservation {r['scenario']}: ${r['reservation_usd']:.2f} x {r['attempts']} ({r['root_tasks']} root"
                 f"{' + evolution' if preflight['self_mod'] else ''})" + (" UNREACHABLE" if r["unreachable"] else ""))
        _log(f"worst case sum of reservations ${preflight['worst_case_usd']:.2f} vs cap ${budget.cap:.2f}; per round ({args.lanes} "
             f"lanes) ${preflight['round_worst_case_usd']:.2f}; dispatch order: {', '.join(f'{sid}_a{n}' for sid, n in jobs)}")
        if preflight["unreachable"]:
            final.update({"outcome": "refused", "exit_code": 3,
                          "refusal": {"stage": "budget_preflight", "reason": "reservation_unreachable", **preflight}})
            _log(f"refused: {preflight['unreachable']} can never run {args.attempts}x under cap ${budget.cap:.2f}; change the flags")
            return 3
        key = ""
        headroom: dict = {}
        if not args.stub:
            key = str(os.environ.get(args.key_env) or "").strip()
            if not key:
                final.update({"outcome": "refused", "exit_code": 3,
                              "refusal": {"stage": "credential", "reason": "key_env_absent", "env": args.key_env}})
                _log(f"refused: ${args.key_env} is empty (export the key under that NAME; the pool file is never read)")
                return 3
            final["credential_fingerprint"] = credential_fingerprint(key)
            try:
                headroom = credit_preflight(key)
            except Exception as exc:  # noqa: BLE001 - an unusable key is a typed refusal
                final.update({"outcome": "refused", "exit_code": 3,
                              "refusal": {"stage": "credit_preflight", "reason": "key_unusable",
                                          "error": f"{type(exc).__name__}: {exc}"[:300]}})
                return 3
            final["credit_preflight"] = {**headroom, "floor_usd": args.min_credit_usd}
            _log(f"key headroom: limit_remaining={headroom['key_limit_remaining_usd']} "
                 f"account_credits={headroom['account_credits_usd']} -> min={headroom['remaining_usd']}")
            if headroom["remaining_usd"] is not None and headroom["remaining_usd"] < args.min_credit_usd:
                final.update({"outcome": "refused", "exit_code": 3,
                              "refusal": {"stage": "credit_preflight", "reason": "insufficient_remaining", **headroom,
                                          "floor_usd": args.min_credit_usd}})
                return 3
        try:
            manifest["seed"] = materialize_seed(source, str(args.seed), seed)
        except SeedMaterializeRefused as exc:
            final.update({"outcome": "refused", "exit_code": 3,
                          "refusal": {"stage": "seed_materialize", "reason": exc.reason, "error": str(exc)[:300]}})
            _log(f"refused: {exc}")
            return 3
        provenance = repo_provenance(seed)
        manifest["seed"]["provenance"] = provenance
        manifest["seed"]["clean"] = seed_is_clean(provenance, manifest["seed"]["resolved_sha"])
        if not manifest["seed"]["clean"]:
            final.update({"outcome": "refused", "exit_code": 3,
                          "refusal": {"stage": "seed_materialize", "reason": "seed_not_clean",
                                      "describe": provenance.get("describe", ""), "head": provenance.get("head", "")}})
            _log(f"refused: materialized seed is not clean: {provenance.get('describe')}")
            return 3
        final["seed_head"], final["seed_describe"] = provenance.get("head", ""), provenance.get("describe", "")
        _log(f"seed: {args.seed} -> {provenance.get('describe')} (detached clone at {seed})")
        if manifest.get("source", {}).get("dirty"):
            _log(f"WARNING: the source checkout {source} is dirty ({manifest['source'].get('status_entries')} "
                 "entries): the seed is the COMMITTED ref above; uncommitted edits are NOT under test")
        full = effective_settings(args, key)
        # The template the run keeps and hands to lanes is REDACTED (no credential value in any run-level artifact
        # or shared object); the key is injected into each lane's own 0600 settings file, disclosed by fingerprint.
        template = redacted_template(full)
        template_path = out / "effective_settings.json"
        write_settings(template_path, template)
        manifest["model_slots"] = model_slot_snapshot(template_path, env_overrides=False)
        manifest["provider_credentials"] = provider_credential_disclosure(
            template_path, runtime_credentials=template_credentials(full))
        final["effective_model"] = manifest["model_slots"].get("OUROBOROS_MODEL", "") if not args.stub else "loopback stub"
        final["settings_config_sha256"] = config_sha256(full)
        granted = set(template_credentials(full))
        missing = [k for k in manifest["provider_credentials"].get("planned_keys") or [] if k not in granted]
        if missing and not args.stub:
            _log(f"WARNING: declared slots without a credential: {missing}")
        seed_sha = str(manifest["seed"]["resolved_sha"])
        states: dict = {}
        stop = threading.Event()
        probe = None
        if key:
            probe = KeyProbe(lambda: credit_preflight(key, timeout=PROBE_TIMEOUT_SEC)["remaining_usd"],
                             floor=args.min_credit_usd, interval=args.watch_interval, stop=stop)
            probe.seed(headroom.get("remaining_usd")).start()
        threading.Thread(target=watcher, args=(stop, states, args.watch_interval, budget, probe), daemon=True).start()
        rows: list[dict] = []
        gate = Stagger(args.stagger)
        # Every lane server inherits THIS process's environment (``IsolatedServer._env`` copies it and keeps this
        # key through the authoritative sweep); set once, before the first lane starts, and unconditionally: an
        # ambient value from the operator shell must not decide the stand's load.
        os.environ[PREFLIGHT_WORKERS_ENV] = str(args.preflight_test_workers)
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=args.lanes) as pool:
                futures = [pool.submit(run_attempt, job, args, out, template, gate, states, seed, budget,
                                       dispatch_index=index, key=key, seed_sha=seed_sha) for index, job in enumerate(jobs)]
                rows = [f.result() for f in futures]
        finally:
            stop.set()
        verdicts = {}
        for sid in args.scenario_ids:
            passed = sum(1 for r in rows if r["scenario"] == sid and r["status"] == "pass")
            verdicts[sid] = {"attempts": args.attempts, "passed": passed,
                             "infra_errors": sum(1 for r in rows if r["scenario"] == sid and r["status"] == "infra_error"),
                             "not_run": sum(1 for r in rows if r["scenario"] == sid and r["status"] == "not_run"),
                             "verdict": "pass" if passed >= args.pass_of else "fail"}
        ok = all(v["verdict"] == "pass" for v in verdicts.values())
        if args.self_mod:
            # Run-level gate: EVERY absorbing lane (``Scenario.expects_absorb``) that ran must carry a CONFIRMED absorb.
            unconfirmed = sorted(f"{r['scenario']}_a{r['attempt']}" for r in rows
                                 if r["status"] != "not_run" and SCENARIOS[r["scenario"]].expects_absorb
                                 and not (r.get("self_mod_absorb") or {}).get("confirmed"))
            final["self_mod"] = {"lanes": sum(1 for r in rows if r["status"] != "not_run"),
                                 "absorb_expected": sum(1 for r in rows if r["status"] != "not_run"
                                                        and SCENARIOS[r["scenario"]].expects_absorb),
                                 "absorb_unconfirmed": unconfirmed}
            ok = ok and not unconfirmed
        for r in rows:
            failed = sorted(k for k, v in r["checks"].items() if not v)
            _log(f"{r['scenario']}_a{r['attempt']}: {r['status']} in {r.get('duration_sec')}s"
                 + (f" failed checks: {failed}" if failed else "") + (f" error: {r['error'][:160]}" if r["error"] else ""))
        budget_final = budget.snapshot()
        _log(f"verdicts: {json.dumps(verdicts)}")
        _log(f"budget: spent ${budget_final['spent_usd']:.2f} of cap ${budget_final['cap_usd']:.2f}"
             + (f"; {len(budget_final['refusals'])} attempt(s) refused at the cap, first {budget_final['first_refused']}"
                if budget_final["refusals"] else ""))
        final.update({"outcome": "completed" if ok else "failed", "exit_code": 0 if ok else 1,
                      "scenarios": verdicts, "lanes_run": sum(1 for r in rows if r["status"] != "not_run"),
                      "budget": budget_final})
        if budget_final["refusals"]:
            final["stop_reason"] = "budget_cap"
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
