#!/usr/bin/env python3
"""One small delegated task through Claudexor, from inside Ouroboros, on one platform.

WHAT THIS PROVES, AND WHAT IT DOES NOT. Stated first because a green check is read
by people who will not open this file (owner directive D20, and the question he
left open beside it as Q8):

  PROVES  — that on THIS operating system Ouroboros can discover the Claudexor
            daemon, negotiate the protocol, register a run root, start a MUTATING
            run pinned to the named harness, poll it to a terminal state, read its
            PRIMARY ARTIFACT to EOF, observe the child's edit on disk, and read
            back the applied-containment facts. For a managed runtime it also
            exercises the serving closure's identity-bound graceful stop, then
            ordinary CLI shutdown over an active fake run and legacy custody. That is
            the transport, lifecycle, platform paths and Ouroboros-to-Claudexor seam. The artifact read
            is a plain EOF read verified against the size the run itself reported
            — it is NOT the production custody acknowledgement, which never runs
            here.

  DOES NOT PROVE — that a SUBSCRIPTION harness works on this platform. No
            subscription can be authenticated in CI: the vendor logins are
            interactive and machine-bound, and putting a live OAuth token in a
            repository secret is not something this project does. The live lane
            therefore runs on explicit API keys: ``authPreference`` is ``api_key``
            where production sends ``subscription``. A subscription regression is
            caught by the operator's own local run before the PR, not here.

  ALSO DOES NOT PROVE (fixture lane only) — anything about what wraps a real child
            PROCESS. The ``fake-*`` harnesses are in-process generators; nothing is
            spawned. Only the live lane spawns. And the fixture request does NOT
            carry the ``execution.delegated`` marker: on a host with a kernel
            boundary the delegated mutating shape demands ``external_sandbox_full``,
            which no ``fake-*`` adapter maps, so the engine refuses the lane
            (verified against 3.3.7-rc.0). The delegated run shape itself is
            exercised only by the live lane.

WHY THE POOL IS PINNED. ``primaryHarness`` alone only prefers a lane WITHIN the
engine's pool, and the engine's AUTO-pool excludes ``fake-*`` routes by design
(``manifest.kind === "fake"`` is filtered before routing), so an unpinned fixture
request never starts the fake lane at all — it routes to whatever real harness the
doctor admits, or dies with ``no doctor-ok harness``. Both requests therefore pin
``harnesses: [<route>]`` next to ``primaryHarness``; an explicit pool also turns any
later ineligibility into a loud typed refusal instead of a silent substitution.

NO PLATFORM BRANCHES IN THE VERDICT. Whether a boundary exists is read out of what
the engine RECORDED for this attempt (``confinement_mechanism`` together with the
path it proved it denies), never from ``sys.platform`` — D21, and
``docs/DELEGATED_ADMISSION.md`` §8. Only macOS has a kernel mechanism and the owner
has ruled out kernel boundaries elsewhere, so off macOS a delegated run gets a scoped
HOME, records no mechanism, and completes normally with that absence disclosed. This
script therefore asserts NOTHING about containment: it reports whatever the attempt
recorded, so both outcomes are legible and neither is a refusal.

FAILS LOUDLY, NEVER SKIPS. Every refusal on the way — an absent seam, an engine
below the delegated-marker floor, an unreachable daemon, a run that never reaches a
terminal state, a child that produced no edit — exits non-zero with a named reason.
The live lane permits one second model attempt only when the first run succeeded,
its primary artifact was read to EOF, and the README stayed byte-identical; a second
no-edit result is still hard red. There is deliberately no "harness unavailable,
skipping" branch: the whole point of the gate is to notice when the delegated path
stops working on a platform.

Usage:
    python scripts/claudexor_platform_smoke.py --managed-runtime --lane fixture
    python scripts/claudexor_platform_smoke.py --managed-runtime --lane live --harness claude \
        --model claude-haiku-4-5 --effort low \
        --secret-name anthropic --secret-env ANTHROPIC_API_KEY
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Claudexor's own terminal vocabulary, mirrored from ouroboros/tools/delegate.py.
# Kept here rather than imported because that name is private to the tool module.
TERMINAL_STATES = frozenset({"succeeded", "failed", "cancelled", "interrupted"})
SUCCESS_STATE = "succeeded"
POLL_INTERVAL_SEC = 3.0

# The file `fake-implement` deterministically creates in the run cwd
# (Claudexor packages/harness-fake/src/index.ts). The fixture lane's on-disk proof.
FIXTURE_CHANGE_FILE = "FAKE_CHANGE.txt"

README_NAME = "README.md"
README_SEED = """# Delegated smoke fixture

This repository exists for one CI check and is deleted when it finishes.

## Status

pending
"""

# The owner's own test rule (plan §4.10, his answer 84): "put low effort and weak
# models for tests. Give some simple task like editing a readme."
LIVE_PROMPT = (
    "Use your filesystem tools now; do not merely explain or print a patch. Edit "
    + README_NAME + " in this repository: replace the single word `pending` "
    "under the `## Status` heading with the single word `done`. Change nothing else. "
    "Read the file back and verify that exact edit before answering. "
    "Do not create files, do not commit, do not run tests."
)
LIVE_EXPECT_TOKEN = "done"


class SmokeFailure(RuntimeError):
    """A named, machine-readable refusal. Every exit path that is not success."""

    def __init__(self, code: str, message: str, **facts: Any) -> None:
        super().__init__(message)
        self.code = str(code)
        self.facts: Dict[str, Any] = dict(facts)


# ---------------------------------------------------------------------------
# Path handling
# ---------------------------------------------------------------------------


def confined_child(root: pathlib.Path, *parts: str) -> pathlib.Path:
    """Join under ``root`` and confine by STRING SHAPE, not by membership.

    A membership test ("is this path somewhere under $HOME?") is the check that has
    already bitten this repository once: on Windows the runner's temp directory
    lives UNDER the user profile, so a `$HOME`-membership guard passes for paths a
    POSIX guard rejects. The shape rule below does not care where the root sits: a
    component may not be empty, may not contain a separator or a drive letter, and
    may not be `..`; the joined path is resolved and must still spell out as the
    resolved root followed by a separator.
    """
    for part in parts:
        if not part or part in (".", ".."):
            raise SmokeFailure("unsafe_path_component", f"refusing path component {part!r}")
        if "/" in part or "\\" in part or ":" in part:
            raise SmokeFailure("unsafe_path_component", f"refusing path component {part!r}")
    base = pathlib.Path(os.path.realpath(str(root)))
    candidate = pathlib.Path(os.path.realpath(str(base.joinpath(*parts))))
    prefix = str(base).rstrip(os.sep) + os.sep
    if not str(candidate).startswith(prefix):
        raise SmokeFailure(
            "path_escaped_root",
            f"{candidate} does not spell out under {base}",
        )
    return candidate


def _git(cwd: pathlib.Path, *args: str) -> str:
    proc = subprocess.run(
        ["git", *args], cwd=str(cwd), capture_output=True, text=True, check=False,
    )
    if proc.returncode != 0:
        raise SmokeFailure(
            "git_failed",
            f"git {' '.join(args)} failed in {cwd}: {proc.stderr.strip()[:400]}",
        )
    return proc.stdout


def seed_fixture_repo() -> pathlib.Path:
    """A throwaway git repo with one README. Claudexor requires a git root.

    Deliberately NOT the Ouroboros checkout: a delegated mutating child gets a real
    shell in the run root, and the run root of a CI smoke has no business being the
    tree the job was built from.
    """
    root = pathlib.Path(os.path.realpath(tempfile.mkdtemp(prefix="ouroboros-cxsmoke-")))
    confined_child(root, README_NAME).write_text(README_SEED, encoding="utf-8")
    _git(root, "init", "--quiet")
    # Pin identity locally: a runner with no global git identity would fail the commit.
    _git(root, "config", "user.email", "ci@ouroboros.invalid")
    _git(root, "config", "user.name", "Ouroboros CI")
    _git(root, "add", README_NAME)
    _git(root, "commit", "--quiet", "-m", "seed")
    return root


def isolated_fixture_root() -> pathlib.Path:
    """Create the managed smoke root where Claudexor's Unix socket can fit.

    macOS caps pathname-based AF_UNIX sockets at a little over 100 bytes.  The
    runner's default temporary directory can already consume most of that
    budget, and the managed daemon appends ``data/claudexor/daemon/claudexord.sock``.
    Keep only this disposable smoke root under the short POSIX temp alias; the
    product's configured data directory remains untouched.
    """
    options = {"prefix": "cx-"}
    if os.name != "nt" and pathlib.Path("/tmp").is_dir():
        options["dir"] = "/tmp"
    return pathlib.Path(tempfile.mkdtemp(**options)).resolve()


# ---------------------------------------------------------------------------
# The seam
# ---------------------------------------------------------------------------


def load_seam() -> Any:
    """Import the Ouroboros side of the seam, or refuse with a readable reason."""
    try:
        from ouroboros import subagents  # noqa: F401
        from ouroboros.config import (  # noqa: F401
            CLAUDEXOR_DELEGATED_MARKER_MIN_VERSION,
            CLAUDEXOR_MIN_VERSION,
        )
        from ouroboros.gateways import claudexor as cx  # noqa: F401
    except ImportError as exc:
        raise SmokeFailure(
            "seam_absent",
            "The Ouroboros side of the Claudexor seam is not importable "
            f"({type(exc).__name__}: {exc}). This job exercises "
            "ouroboros.gateways.claudexor + ouroboros.subagents.delegated_run_shape; "
            "a tree that predates the delegated-transport merge cannot run it.",
        ) from exc
    return sys.modules["ouroboros.gateways.claudexor"]


# ---------------------------------------------------------------------------
# The run
# ---------------------------------------------------------------------------


def build_request(lane: str, harness: str, model: str, effort: str,
                  root: pathlib.Path, prompt: str, max_seconds: int) -> Dict[str, Any]:
    """The production run SHAPE, from Ouroboros's own authority derivation.

    The stated divergences from what ``ouroboros/tools/delegate.py`` sends — each
    of them carried in the step summary rather than hidden here:

    - both lanes PIN the pool (``harnesses: [<route>]``): the module docstring's
      "WHY THE POOL IS PINNED" — ``primaryHarness`` alone cannot start a lane the
      auto-pool excludes, and every ``fake-*`` route is excluded by design;
    - the live lane sends ``authPreference: "api_key"`` where production sends
      ``subscription``;
    - the fixture lane drops the ``execution.delegated`` marker: the delegated
      mutating shape requires ``external_sandbox_full`` wherever the host has a
      kernel boundary, and no ``fake-*`` adapter maps that profile, so the engine
      refuses the pinned lane with a typed routing failure (verified against
      3.3.7-rc.0). The delegated shape is exercised by the live lane only.
    """
    from ouroboros.subagents import delegated_run_shape

    shape = delegated_run_shape(acting=True)
    request: Dict[str, Any] = {
        "prompt": prompt,
        "mode": shape.mode,
        "scope": {"kind": "project", "root": str(root)},
        "harnesses": [harness],
        "primaryHarness": harness,
        "access": shape.access,
        "execution": {
            "isolation": shape.isolation,
            "delegated": shape.delegated if lane == "live" else False,
        },
        "maxSeconds": int(max_seconds),
    }
    if lane == "live":
        # Production binds a fresh delegated mutating run to the caller-owned
        # execution tree. The smoke owns ``root`` too, so exercise that same
        # admission contract instead of relying on the project scope as a shim.
        request["execution"]["workspaceRoot"] = str(root)
        # Production sends "subscription". CI cannot authenticate one, so it asks
        # for the substrate it actually has. This is THE difference between what
        # this gate exercises and what production does.
        request["authPreference"] = "api_key"
        if model:
            request["model"] = model
        if effort:
            request["effort"] = effort
    # The fixture lane sends no authPreference on purpose: the fake harness
    # authenticates nothing, and naming a preference would fabricate a claim about
    # a credential route that was never taken.
    return request


def poll_to_terminal(gateway: Any, run_id: str, budget_sec: int) -> Dict[str, Any]:
    deadline = time.monotonic() + budget_sec
    last_state = ""
    transient_git_object_races = 0
    while True:
        try:
            detail = gateway.get_run(run_id)
        except Exception as exc:
            if (
                transient_git_object_races < 3
                and _is_transient_git_object_enoent(exc)
                and time.monotonic() < deadline
            ):
                transient_git_object_races += 1
                print(
                    f"[smoke] transient Git object race while polling {run_id}; "
                    f"retry {transient_git_object_races}/3",
                    flush=True,
                )
                time.sleep(POLL_INTERVAL_SEC)
                continue
            raise
        summary = detail.get("summary") if isinstance(detail.get("summary"), dict) else {}
        state = str(summary.get("state") or "")
        if state != last_state:
            print(f"[smoke] run {run_id} state={state or '(none)'}", flush=True)
            last_state = state
        if state in TERMINAL_STATES:
            return detail
        if time.monotonic() >= deadline:
            raise SmokeFailure(
                "run_never_terminal",
                f"run {run_id} was still {state or '(no state)'} after {budget_sec}s",
                run_id=run_id, state=state,
            )
        time.sleep(POLL_INTERVAL_SEC)


def _is_transient_git_object_enoent(exc: Exception) -> bool:
    """Recognize only Git's disappearing atomic-object scratch path."""
    detail = str(exc).replace("\\", "/").lower()
    code = str(getattr(exc, "code", "") or "").upper()
    return (
        (code == "ENOENT" or "enoent" in detail)
        and "/.git/objects/" in detail
        and "/tmp_obj_" in detail
    )


def containment_report(cx: Any, run_dir: str) -> Dict[str, Any]:
    """The applied-fact reader, verbatim from the seam. Never a platform test."""
    attempts = list(cx.attempt_containment(str(run_dir or "")))
    disclosed = sum(1 for a in attempts if a.home_isolated is not None)
    mechanisms = sorted({a.boundary_mechanism for a in attempts})
    boundary = mechanisms[0] if attempts and len(mechanisms) == 1 and mechanisms[0] else ""
    return {
        "attempts": len(attempts),
        "home_facts_disclosed": disclosed,
        "os_boundary": boundary,
        "boundary_verified": bool(boundary and disclosed == len(attempts) and attempts),
    }


# Mirrors ouroboros/tools/delegate.py: the engine's artifact route serves text
# through redactSecrets, so the served length may legally differ from the on-disk
# ``bytes`` — verification then falls back to the preview being a prefix, with a
# bounded slack at the preview boundary where the engine's redaction overlap differs.
_PREVIEW_PREFIX_SLACK = 2_048


def read_primary_artifact_to_eof(gateway: Any, run_id: str,
                                 detail: Dict[str, Any]) -> Dict[str, Any]:
    """Read the run's PRIMARY artifact to EOF and verify it against the run's claim.

    The run detail's ``primaryOutput.text`` is a bounded 256 KiB PREVIEW; the full
    file lives behind ``GET /v2/runs/:id/artifacts/<path>``. A result counts as
    obtained only after that body has been read whole (D7 for this gate) — so a
    terminal run with no primary artifact, an unreadable artifact, or a body that
    matches neither the reported size nor the reported preview is a NAMED refusal,
    never a green check standing on a claim nobody read.

    What "verify" is worth here, exactly: the engine publishes NO content hash for
    the primary output, so both checks compare the body against the run's own
    DESCRIPTION of it — an equal byte count, or the preview carried as a prefix.
    That catches a truncated, empty or wrong-length read, which is what this gate is
    for. It does not bind the bytes to a digest, so it cannot tell a correct body
    from a corrupted one of the same length. Stated rather than implied, because the
    strong reading is the tempting one.

    This is a plain EOF read: the production custody acknowledgement (the durable
    D7 row) is part of the delegation path this smoke deliberately does not drive.
    """
    primary = detail.get("primaryOutput")
    if not isinstance(primary, dict) or not str(primary.get("path") or ""):
        raise SmokeFailure(
            "primary_artifact_missing",
            f"the terminal run reported no primary artifact to read: {primary!r}",
            run_id=run_id,
        )
    path = str(primary.get("path") or "")
    try:
        raw = gateway.get_run_artifact(run_id, path)
    except Exception as exc:
        raise SmokeFailure(
            "primary_artifact_unread",
            f"the primary artifact {path!r} could not be read to EOF: "
            f"{getattr(exc, 'code', type(exc).__name__)}: {exc}",
            run_id=run_id, path=path,
        ) from exc
    reported = primary.get("bytes")
    preview = primary.get("text") if isinstance(primary.get("text"), str) else ""
    if isinstance(reported, int) and not isinstance(reported, bool) and len(raw) == reported:
        verified = "size"
    else:
        text = raw.decode("utf-8", errors="replace")
        prefix = preview[:max(0, len(preview) - _PREVIEW_PREFIX_SLACK)]
        if prefix and text.startswith(prefix) and len(text) >= len(preview):
            verified = "preview_prefix"
        else:
            raise SmokeFailure(
                "primary_artifact_unverified",
                f"the primary artifact {path!r} was fetched ({len(raw)} bytes) but "
                f"matches neither the reported size ({reported!r}) nor the reported "
                "preview; the engine publishes no content hash, so those two are the "
                "whole of the check, and a body that matches neither does not count "
                "as the result",
                run_id=run_id, path=path, read_bytes=len(raw), reported_bytes=reported,
            )
    return {
        "kind": str(primary.get("kind") or ""),
        "path": path,
        "reported_bytes": reported,
        "read_bytes": len(raw),
        "verified": verified,
    }


def mutation_evidence(lane: str, root: pathlib.Path) -> Tuple[bool, str]:
    """Did the child actually change the tree? On-disk proof, not the run's claim."""
    if lane == "fixture":
        marker = confined_child(root, FIXTURE_CHANGE_FILE)
        if marker.exists():
            return True, f"{FIXTURE_CHANGE_FILE} was created in the run root"
        return False, f"{FIXTURE_CHANGE_FILE} is absent — the delegated child wrote nothing"
    text = confined_child(root, README_NAME).read_text(encoding="utf-8", errors="replace")
    changed = text != README_SEED
    if not changed:
        return False, f"{README_NAME} is byte-identical to the seed — the child edited nothing"
    if LIVE_EXPECT_TOKEN not in text:
        return False, (
            f"{README_NAME} changed but does not contain {LIVE_EXPECT_TOKEN!r}; "
            "the edit is not the one that was asked for"
        )
    return True, f"{README_NAME} was edited and contains {LIVE_EXPECT_TOKEN!r}"


def should_retry_live_no_mutation(
    lane: str, attempt: int, mutated: bool, seed_unchanged: bool,
) -> bool:
    """Allow exactly one retry for a successful live run that made no edit."""
    return lane == "live" and attempt == 1 and not mutated and seed_unchanged


def live_seed_is_unchanged(root: pathlib.Path) -> bool:
    """Compare text, not host-specific newline bytes (CRLF vs LF)."""
    return (
        confined_child(root, README_NAME).read_text(
            encoding="utf-8", errors="replace",
        )
        == README_SEED
    )


def graceful_stop_managed_runtime(
    command: List[str], config_dir: pathlib.Path, version: str, build_sha: str
) -> Dict[str, Any]:
    """Exercise the serving closure's identity-bound replacement-stop contract."""
    if not command or not version or len(build_sha) != 40 or any(
        char not in "0123456789abcdef" for char in build_sha
    ):
        raise SmokeFailure(
            "graceful_stop_identity_missing",
            "the managed daemon handshake did not provide an exact serving version/build SHA",
        )
    from ouroboros.platform_layer import subprocess_hidden_kwargs

    env = dict(os.environ)
    env["CLAUDEXOR_CONFIG_DIR"] = str(config_dir)
    for crossing in ("CLAUDEXOR_DAEMON_SOCK", "CLAUDEXOR_CONTROL_PORT"):
        env.pop(crossing, None)
    try:
        completed = subprocess.run(
            [*command, "--stop", version, build_sha],
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=60,
            env=env,
            **subprocess_hidden_kwargs(),
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise SmokeFailure(
            "graceful_stop_failed",
            f"the serving closure's --stop command failed: {type(exc).__name__}: {exc}",
        ) from exc
    receipt = None
    for line in reversed((completed.stdout or "").splitlines()):
        try:
            candidate = json.loads(line)
        except ValueError:
            continue
        if isinstance(candidate, dict):
            receipt = candidate
            break
    if completed.returncode != 0 or not isinstance(receipt, dict) or receipt.get("stopped") is not True:
        detail = (completed.stderr or completed.stdout or "").strip()[-500:]
        raise SmokeFailure(
            "graceful_stop_failed",
            f"the serving closure did not confirm a graceful stop (exit {completed.returncode})"
            + (f": {detail}" if detail else ""),
        )
    return {
        "stopped": True,
        "already_stopped": receipt.get("alreadyStopped") is True,
        "outcome": str(receipt.get("outcome") or ""),
    }


def verify_operator_stop_fixture(root: pathlib.Path, facts: Dict[str, Any]) -> None:
    """Real operator CLI + writer lease, on the smoke's own fresh generation only."""
    from unittest.mock import patch
    from ouroboros import claudexor_daemon as daemon, claudexor_runtime, config, process_custody as custody
    from ouroboros.platform_layer import pid_is_alive
    from ouroboros.utils import atomic_write_json

    owner = daemon.get_owned_daemon()
    if owner._proc is not None:
        owner._proc.wait(timeout=5)  # Replacement ACK may precede the old Node's exit.
    try:
        gateway = daemon.ensure_owned_gateway()
    except BaseException:
        owner.stop()
        raise
    proc = owner._proc  # Capture the fixture we spawned; the stop actor has no Popen authority.
    run_id, stopped = "", False
    proof: Dict[str, Any] = {}
    facts["operator_stop"] = proof
    try:
        if proc is None or proc.poll() is not None:
            raise SmokeFailure("operator_fixture_not_owned", "the fixture must spawn its own daemon")
        proof["daemon_pid"] = proc.pid
        manager = claudexor_runtime.get_runtime_manager()
        cli, pin = manager.resolve_cli_command(require_npm=False), manager.pin
        body = gateway.handshake()
        if not cli or pin is None or gateway.engine_version != pin.version or handshake_engine_sha(body) != pin.build_sha:
            raise SmokeFailure("operator_fixture_identity", "the managed CLI/daemon pin is unavailable or mismatched")
        proof.update(cli_command=cli, engine_version=pin.version, engine_sha=pin.build_sha)
        home = daemon.owned_config_dir() / "daemon"
        candidates = [*home.glob("*.writer/owner.json"), *home.glob("*.writer/active.writer/owner.json")]
        leases = [(p, json.loads(p.read_text(encoding="utf-8"))) for p in candidates]
        leases = [(p, row) for p, row in leases if row.get("pid") == proc.pid]
        rows = custody._read_ledger(config.DATA_DIR)
        own = [r for r in rows if r.get("pid") == proc.pid and r.get("purpose") == daemon.CUSTODY_PURPOSE]
        if len(leases) != 1 or len(own) != 1:
            raise SmokeFailure("operator_fixture_custody", "expected one exact fixture lease and custody row")
        lease, lease_owner = leases[0]
        lease_owner.pop("identity", None)  # Absent is legacy; null would be malformed.
        atomic_write_json(lease, lease_owner)
        own[0]["fingerprint"]["start_time"] = ""
        own[0]["fingerprint"].pop("start_time_boot", None)
        custody._rewrite_ledger(config.DATA_DIR, rows)
        if custody._fingerprint_matches(own[0], require_measured=True):
            raise SmokeFailure("operator_fixture_not_legacy", "legacy row unexpectedly grants signal authority")
        proof.update(legacy_birth=True, lease_path=str(lease), lease_identity_absent=True)
        gateway.register_project(str(root))
        handle = gateway.start_run(
            build_request("fixture", "fake-hang", "", "", root, "Wait for operator stop", 300),
            idempotency_key=uuid.uuid4().hex,
        )
        run_id = str(handle.get("runId") or handle.get("jobId") or "")
        if not run_id:
            raise SmokeFailure("operator_fixture_run_missing", "fake-hang returned no run id")
        proof["run_id"] = run_id
        deadline = time.monotonic() + 30
        while True:
            summary = gateway.get_run(run_id, timeout_sec=2).get("summary") or {}
            run_dir = pathlib.Path(summary["runDir"]) if summary.get("runDir") else None
            entered = False
            for path in run_dir.glob("events.jsonl") if run_dir else []:
                # A live writer may not have finished its final line yet.
                raw = path.read_text(encoding="utf-8")
                for line in raw.splitlines(keepends=True):
                    if line.strip() and line.endswith("\n"):
                        row = json.loads(line)
                        event = row.get("payload") or {}
                        if row.get("type") == "harness.event" and event.get("harness_id") == "fake-hang" and event.get("type") == "thinking":
                            entered = True
                            proof["entered_event_seq"] = row.get("seq")
            if summary.get("state") == "running" and entered:
                proof.update(fake_hang_entered=True, run_dir=str(run_dir))
                break
            if time.monotonic() >= deadline or summary.get("state") in TERMINAL_STATES:
                raise SmokeFailure("operator_fixture_not_running", "fake-hang did not enter its active wait")
            time.sleep(.1)
        fresh = daemon.OwnedClaudexorDaemon()
        actual_run, observed = subprocess.run, []

        def record(argv: List[str], **kwargs: Any) -> Any:
            result = actual_run(argv, **kwargs)
            if list(argv) == [*cli, "daemon", "stop", "--json"]:
                receipt = json.loads(result.stdout) if result.returncode == 0 else None
                observed.append({"exit_code": result.returncode, "receipt": receipt})
                proof["cli_attempts"] = observed
            return result

        if fresh._proc is not None:
            raise SmokeFailure("operator_fixture_not_attached", "stop actor unexpectedly owns a Popen handle")
        with patch.object(daemon.subprocess, "run", side_effect=record), patch.object(
            custody, "stop_ledgered_processes", side_effect=AssertionError("forced fallback is not cooperative proof"),
        ):
            outcome = fresh.stop_outcome()
        if (outcome != "stopped" or len(observed) != 1 or observed[0]["exit_code"] != 0
                or not isinstance(observed[0]["receipt"], dict)
                or observed[0]["receipt"].get("ok") is not True
                or observed[0]["receipt"].get("stopped") is not True
                or observed[0]["receipt"].get("outcome") != "exited"):
            raise SmokeFailure("operator_stop_unconfirmed", "ordinary CLI did not prove cooperative shutdown")
        proof["daemon_exit_code"] = proc.wait(timeout=5)
        retained = [r for r in custody._read_ledger(config.DATA_DIR) if r.get("pid") == proc.pid]
        if (proof["daemon_exit_code"] != 0 or pid_is_alive(proc.pid) or lease.parent.exists()
                or len(retained) != 1 or retained[0]["fingerprint"].get("start_time") != ""):
            raise SmokeFailure("operator_stop_not_settled", "process/lease death or unchanged legacy custody not proven")
        stopped = True
        proof.update(stopped=True, pid_gone=True, lease_released=True, forced_fallback=False)
    finally:
        if run_id and not stopped:
            try:
                gateway.cancel_run(run_id, reason="operator-stop fixture teardown")
            except Exception:
                pass
        gateway.close()
        # The original fixture Popen remains cleanup authority even if CLI proof fails.
        owner.stop()
        if proc is not None and proc.poll() is None:
            raise SmokeFailure("operator_fixture_leaked", "fixture daemon did not exit during teardown")


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _limits_block(lane: str) -> List[str]:
    lines = [
        "### What this check does NOT cover",
        "",
        "- **Subscription harnesses are not covered.** A subscription cannot be "
        "authenticated in CI (interactive, machine-bound logins), so no green run here "
        "says anything about the subscription route that this integration exists for. "
        "That route is verified only by the operator's own local run.",
        "- The `containment` row is a REPORT, never an assertion. Only macOS has a "
        "kernel boundary mechanism; elsewhere a delegated run gets a scoped HOME, "
        "records no mechanism, and completes normally with the absence disclosed. An "
        "empty `os_boundary` off macOS is the expected shape, not a failure — and this "
        "check does not decide that by looking at which platform it is on.",
    ]
    lines.append(
        "- The request pins its route explicitly (`harnesses: [<route>]` + "
        "`primaryHarness`): the engine's auto-pool excludes `fake-*` routes by design "
        "and may rank a different real harness first, so an unpinned request would not "
        "prove anything about the lane this check names.",
    )
    if lane == "live":
        lines += [
            "- The request sent here differs from the production request in exactly one "
            "other field: `authPreference: \"api_key\"` where production sends "
            "`\"subscription\"`.",
            "- A live model gets **one bounded second attempt** only when its first run "
            "reached success, its primary artifact was read to EOF, and the README "
            "remained byte-identical to the seed. Both attempts use the same repo and "
            "request with fresh idempotency keys. A second no-edit result is still a "
            "hard failure; routing, auth, engine, process, artifact, and timeout failures "
            "are never retried.",
        ]
    else:
        lines += [
            "- **No model and no credentials were involved.** The `fake-*` harness is a "
            "deterministic offline fixture; this lane proves the seam's wiring, not that "
            "any harness runs.",
            "- **No child process was spawned**, so nothing this platform would have "
            "wrapped a real child in was exercised. Only the live lane spawns a real child.",
            "- **The `execution.delegated` marker was not sent.** A delegated mutating "
            "run requires `external_sandbox_full` wherever the host has a kernel "
            "boundary, and no `fake-*` adapter maps that profile — the engine would "
            "refuse the lane. The delegated run shape is exercised only by the live lane.",
            "- The primary artifact was read to EOF, but the production custody "
            "acknowledgement (the durable D7 row) is part of the delegation path this "
            "smoke does not drive.",
            "- Managed fixture shutdown also runs the real operator CLI over an active "
            "fake-hang and a legacy writer lease; that fake still spawns no harness process.",
        ]
    return lines


def emit_summary(lane: str, facts: Dict[str, Any], verdict: str, detail: str) -> None:
    plat = f"{sys.platform} / python {sys.version.split()[0]}"
    lines = [
        f"## Claudexor platform smoke — `{lane}` lane on `{plat}`",
        "",
        f"**{verdict}** — {detail}",
        "",
        "| fact | value |",
        "| --- | --- |",
    ]
    for key in sorted(facts):
        value = facts[key]
        if isinstance(value, (dict, list)):
            value = json.dumps(value, ensure_ascii=False)
        lines.append(f"| `{key}` | {value} |")
    lines += ["", *_limits_block(lane), ""]
    text = "\n".join(lines)
    print(text, flush=True)
    target = os.environ.get("GITHUB_STEP_SUMMARY", "")
    if target:
        try:
            with open(target, "a", encoding="utf-8") as fh:
                fh.write(text + "\n")
        except OSError as exc:  # never let reporting mask the verdict
            print(f"[smoke] could not write GITHUB_STEP_SUMMARY: {exc}", flush=True)


def handshake_engine_sha(body: Dict[str, Any]) -> str:
    """Return the serving build identity from the frozen protocol-v3 shape."""
    engine = body.get("engine")
    return str(engine.get("sha") or "") if isinstance(engine, dict) else ""


# ---------------------------------------------------------------------------


def run_smoke(args: argparse.Namespace) -> Dict[str, Any]:
    cx = load_seam()
    from ouroboros.config import (
        CLAUDEXOR_DELEGATED_MARKER_MIN_VERSION,
        CLAUDEXOR_MIN_VERSION,
    )

    facts: Dict[str, Any] = {
        "lane": args.lane,
        "harness": args.harness,
        "transport_floor": CLAUDEXOR_MIN_VERSION,
        "delegated_marker_floor": CLAUDEXOR_DELEGATED_MARKER_MIN_VERSION,
    }
    if args.managed_runtime and args.lane == "fixture":
        from ouroboros.config import DATA_DIR
        facts["fixture_data_root"] = str(DATA_DIR)

    owned_daemon = None
    if args.managed_runtime:
        from ouroboros.claudexor_daemon import ensure_owned_gateway, get_owned_daemon

        owned_daemon = get_owned_daemon()
        try:
            gateway = ensure_owned_gateway()
        except BaseException:
            if args.lane == "fixture" and owned_daemon._proc is not None:
                owned_daemon.stop()
            raise
        owned_status = owned_daemon.status_dict()
        facts["daemon"] = str(owned_status.get("config_dir") or "managed")
        facts["managed_runtime"] = owned_status.get("runtime") or {}
    else:
        endpoint = cx.discover_daemon()
        facts["daemon"] = f"{endpoint.host}:{endpoint.port}"
        gateway = cx.ClaudexorGateway(endpoint)
    root: Optional[pathlib.Path] = None
    registration = None

    def release_fixture_registration():
        nonlocal registration
        from ouroboros.gateways.claudexor_maintenance import bound_registration_supported

        receipt, registration = registration, None  # an unanswered release must not be repeated in finally
        if (receipt and receipt["created"] is True and receipt.get("registration_store_id")
                and receipt.get("registration_root") and bound_registration_supported(gateway)):
            gateway.bound_registration(receipt["id"], receipt["registration_store_id"], receipt["registration_root"],
                                       release=True)

    run_ids: List[str] = []
    facts["run_ids"] = run_ids
    facts["attempts"] = []
    gateway_closed = False
    try:
        body = gateway.handshake()  # enforces the TRANSPORT floor, raises if below
        engine = str(gateway.engine_version or "")
        engine_build_sha = handshake_engine_sha(body)
        facts["engine_version"] = engine or "(unreported)"
        facts["engine_build_sha"] = engine_build_sha or "(unreported)"
        facts["protocol_major"] = body.get("protocolMajor")

        # The MARKER floor. Below it the engine answers `execution.delegated` with a
        # 400 and no run exists, so refuse here with a named reason rather than spend
        # a dispatch on a certain failure (docs/DELEGATED_ADMISSION.md §6).
        if not cx.engine_at_least(engine, CLAUDEXOR_DELEGATED_MARKER_MIN_VERSION):
            raise SmokeFailure(
                "engine_below_delegated_marker_floor",
                f"Claudexor {engine or 'unknown'} is below the delegated-marker floor "
                f"{CLAUDEXOR_DELEGATED_MARKER_MIN_VERSION}; a mutating delegated run "
                "would be rejected with a 400 before it started. This is a hard red on "
                "purpose — the gate exists to notice exactly this.",
                engine_version=engine,
            )

        if args.secret_env:
            value = str(os.environ.get(args.secret_env) or "")
            if not value:
                raise SmokeFailure(
                    "secret_env_missing",
                    f"the requested secret environment variable {args.secret_env!r} is empty",
                )
            gateway.set_secret(args.secret_name, value)
            facts["managed_secret"] = args.secret_name

        root = seed_fixture_repo()
        facts["run_root"] = str(root)
        prompt = "Create the deterministic fixture change." if args.lane == "fixture" else LIVE_PROMPT
        request = build_request(
            args.lane, args.harness, args.model, args.effort, root, prompt, args.max_seconds,
        )
        facts["request_shape"] = {
            k: request[k]
            for k in ("mode", "access", "execution", "harnesses", "primaryHarness",
                      "authPreference")
            if k in request
        }

        registration = gateway.project_registration_receipt(str(root))
        facts["project_id"] = registration["id"]

        attempt = 1
        while True:
            attempt_facts: Dict[str, Any] = {"attempt": attempt}
            facts["attempts"].append(attempt_facts)
            handle = gateway.start_run(request, idempotency_key=uuid.uuid4().hex)
            run_id = str(handle.get("runId") or handle.get("jobId") or "")
            if not run_id:
                raise SmokeFailure(
                    "queued_without_run_id",
                    f"Claudexor returned a queued handle with no run id: {handle!r}",
                )
            run_ids.append(run_id)
            attempt_facts["run_id"] = run_id
            facts["run_id"] = run_id

            detail = poll_to_terminal(
                gateway, run_id, args.max_seconds + args.grace_seconds,
            )
            summary = detail.get("summary") if isinstance(detail.get("summary"), dict) else {}
            state = str(summary.get("state") or "")
            run_dir = str(summary.get("runDir") or "")
            attempt_facts["state"] = facts["state"] = state
            attempt_facts["run_dir"] = facts["run_dir"] = run_dir or "(none)"

            # Applied facts are reported per attempt. A host with no mechanism is
            # disclosed, never refused.
            containment = containment_report(cx, run_dir)
            attempt_facts["containment"] = facts["containment"] = containment

            if state != SUCCESS_STATE:
                failure = summary.get("failure")
                raise SmokeFailure(
                    "run_not_successful",
                    f"the run reached terminal state {state!r}: "
                    f"{json.dumps(failure, ensure_ascii=False)[:400]}",
                    **facts,
                )

            # D7 for this gate: retry eligibility is considered only after the result
            # has been read to EOF and tied to the run's own size/preview claim.
            artifact = read_primary_artifact_to_eof(gateway, run_id, detail)
            attempt_facts["primary_artifact"] = facts["primary_artifact"] = artifact

            mutated, why = mutation_evidence(args.lane, root)
            attempt_facts["mutation_evidence"] = facts["mutation_evidence"] = why
            attempt_facts["mutated"] = mutated
            # Use the same text semantics as mutation_evidence. On Windows,
            # write_text materializes CRLF bytes while read_text normalizes them;
            # a raw-byte comparison would incorrectly suppress the one retry.
            seed_unchanged = (
                args.lane == "live"
                and live_seed_is_unchanged(root)
            )
            if should_retry_live_no_mutation(
                args.lane, attempt, mutated, seed_unchanged,
            ):
                attempt_facts["outcome"] = "retryable_no_mutation"
                print(
                    "[smoke] successful live run made no edit; using the one "
                    "bounded model-output retry",
                    flush=True,
                )
                attempt += 1
                continue
            if not mutated:
                attempt_facts["outcome"] = "no_mutation"
                attempt_facts["answer_preview"] = str(
                    detail.get("primaryOutput", {}).get("text", "")
                    if isinstance(detail.get("primaryOutput"), dict)
                    else ""
                )[:1_000]
                raise SmokeFailure("no_mutation", why, **facts)
            attempt_facts["outcome"] = "mutated"
            break
        if owned_daemon is not None:
            release_fixture_registration()
            gateway.close()
            gateway_closed = True
            from ouroboros.claudexor_daemon import owned_config_dir
            from ouroboros.claudexor_runtime import get_runtime_manager

            facts["graceful_stop"] = graceful_stop_managed_runtime(
                get_runtime_manager().resolve_command(),
                owned_config_dir(),
                engine,
                engine_build_sha,
            )
            if args.lane == "fixture":
                verify_operator_stop_fixture(root, facts)
        return facts
    except SmokeFailure as exc:
        # Everything learned before the refusal belongs in the report: a summary that
        # names only the code makes the reader open the raw log to find out which lane,
        # which engine and which daemon it was talking to.
        exc.facts = {**facts, **exc.facts}
        raise
    except cx.ClaudexorUnavailable as exc:
        raise SmokeFailure(
            getattr(exc, "code", "claudexor_unavailable"), str(exc), **facts,
        ) from exc
    finally:
        if not gateway_closed:
            for started_run_id in run_ids:
                try:
                    gateway.cancel_run(started_run_id, reason="smoke teardown")
                except Exception:  # already terminal in the happy path
                    pass
        if registration:
            try:
                release_fixture_registration()
            except Exception as exc:
                print(f"[smoke] could not retire project {facts.get('project_id')}: {exc}", flush=True)
        if not gateway_closed:
            gateway.close()
        if owned_daemon is not None:
            owned_daemon.stop()
        if root is not None:
            shutil.rmtree(str(root), ignore_errors=True)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lane", choices=("fixture", "live"), required=True)
    parser.add_argument(
        "--managed-runtime", action="store_true",
        help="Install/probe/start Ouroboros's exact pinned runtime before the smoke.",
    )
    parser.add_argument(
        "--harness", default="",
        help="Claudexor route id. Defaults to fake-implement on the fixture lane.",
    )
    parser.add_argument("--model", default="")
    parser.add_argument("--effort", default="low")
    parser.add_argument("--secret-name", default="")
    parser.add_argument("--secret-env", default="")
    parser.add_argument("--max-seconds", type=int, default=600)
    parser.add_argument("--grace-seconds", type=int, default=120)
    args = parser.parse_args(argv)
    if not args.harness:
        args.harness = "fake-implement" if args.lane == "fixture" else "claude"
    if bool(args.secret_name) != bool(args.secret_env):
        parser.error("--secret-name and --secret-env must be provided together")
    if args.lane == "fixture":
        # A fixture lane that quietly accepted a real route would spend money under a
        # name that promises it does not.
        if not args.harness.startswith("fake-"):
            print(
                f"[smoke] the fixture lane refuses the non-fake route {args.harness!r}",
                file=sys.stderr,
            )
            return 2
        args.model, args.effort = "", ""
        if args.secret_env:
            parser.error("the fixture lane accepts no credentials")
        if args.managed_runtime:
            # The fixture edits its own legacy custody. It must never attach to
            # an installed daemon or discover the operator's native accounts.
            isolated = isolated_fixture_root()
            home = isolated / "home"
            home.mkdir()
            for key in list(os.environ):
                if key.endswith(("API_KEY", "TOKEN", "PASSWORD", "SECRET", "CREDENTIALS")) or (
                    key.startswith(("OUROBOROS_", "CLAUDEXOR_")) and key != "OUROBOROS_BUNDLE_DIR"
                ):
                    os.environ.pop(key, None)
            os.environ.update(
                HOME=str(home), USERPROFILE=str(home), XDG_CONFIG_HOME=str(home / ".config"),
                APPDATA=str(home / "AppData" / "Roaming"), LOCALAPPDATA=str(home / "AppData" / "Local"),
                OUROBOROS_APP_ROOT=str(isolated), OUROBOROS_DATA_DIR=str(isolated / "data"),
                OUROBOROS_SETTINGS_PATH=str(isolated / "data" / "settings.json"),
            )
            print(f"[smoke] isolated fixture root: {isolated}", flush=True)

    try:
        facts = run_smoke(args)
    except SmokeFailure as exc:
        facts = {**exc.facts, "refusal_code": exc.code}
        emit_summary(args.lane, facts, "FAILED", f"`{exc.code}` — {exc}")
        print(f"[smoke] FAILED {exc.code}: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # unexpected: still loud, still named
        emit_summary(args.lane, {"refusal_code": "unexpected_error"}, "FAILED",
                     f"`unexpected_error` — {type(exc).__name__}: {exc}")
        raise
    emit_summary(
        args.lane, facts, "PASSED",
        "the pinned run completed on this platform, its primary artifact was read "
        "to EOF, and its edit is on disk"
        + (", with the managed daemon stopped gracefully" if args.managed_runtime else ""),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
