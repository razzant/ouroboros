"""At-most-once official Cowork evaluation and the opt-in residual-state diagnostic.

Standard library only: the derived image copies this file and ``official_receipt.py`` next
to the entrypoint, and the host launcher and audit read the same records. Every record lives
in the task dump root, beside ``traj_log.json`` and outside the agent's ``workspace/``.

Official attempt. Before the upstream ``evaluate_from_log_file`` call the eval phase
publishes one exclusive claim per run/task with a random attempt ID. The exact log, config,
evaluator and entrypoint facts are bindings of that attempt, never a retry key. The terminal
receipt, published once under an attempt-specific name, keeps the returned value apart from
the ``eval_res.json`` file state. A reentry with the same bindings replays the recorded
verdict without calling the evaluator; an unfinished claim, changed bindings or an unclaimed
existing ``eval_res.json`` never lead to another official effect or an overwrite.

Residual-state diagnostic (``diagnostic_eval_on_truncation``). After a proven status-gate
decline of the same attempt for a concrete time/round/budget stop, a separate Python process
whose fd 1/2 already point to its own log rebuilds the pinned evaluator's checker command
from the saved task config and runs it once more on the state left after the agent stopped
and the official evaluator ran. It is capped, audit-only and never an exact-cutoff verdict:
it writes no ``eval_res.json``, status or log and prints nothing to the shared runner log.
"""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
import signal
import socket
import subprocess
import sys
import time
import traceback
import uuid
from typing import Any, Callable

try:
    from devtools.benchmarks.cowork_bench.official_receipt import (
        GATE_LOG_NAME,
        RESULT_NAME,
        _read_object,
        read_official_receipt,
    )
except ImportError:  # inside the image, beside the entrypoint
    from official_receipt import (  # type: ignore[no-redef]
        GATE_LOG_NAME,
        RESULT_NAME,
        _read_object,
        read_official_receipt,
    )

SCHEMA = "ouroboros.cowork.eval_attempt.v1"
FLAG = "diagnostic_eval_on_truncation"
# Injected by docker_limits.sh into the eval exec only after the daemon listed no container
# with the exact agent name; the eval phase removes it before the official evaluator runs.
AGENT_STATE_ENV = "COWORK_AGENT_CONTAINER_STATE"
CLAIM_NAME = "ouroboros_eval_claim.json"
RECEIPT_PREFIX = "ouroboros_eval_receipt-"
DIAGNOSTIC_CLAIM_NAME = "ouroboros_diagnostic_claim.json"
DIAGNOSTIC_RECEIPT_PREFIX = "ouroboros_diagnostic_receipt-"
DIAGNOSTIC_REPORT_PREFIX = "ouroboros_diagnostic_report-"
DIAGNOSTIC_LOG_PREFIX = "ouroboros_diagnostic-"
UPSTREAM_EVALUATOR = "utils/evaluation/evaluator.py"
# A ceiling, not a promised window: the pinned eval container lives `sleep 1800` in total.
DIAGNOSTIC_CAP_SEC = 900
TEARDOWN_GRACE_SEC = 5.0
# Concrete runtime/adapter stops only. Owner stops, unabsorbed children, finalization grace
# and infrastructure codes stay ineligible even though they are also disclosed truncations.
ELIGIBLE_REASON_CODES = frozenset({"deadline_local", "wall_clock_timeout", "round_limit", "budget_exhausted"})
SEMANTICS = "residual_state_after_agent_stop_not_exact_cutoff"


class NotRun(Exception):
    """A preparation refusal before any official effect; the attempt settles unevaluated."""


def file_facts(path: pathlib.Path) -> dict[str, Any]:
    """Exact bytes identity: ``absent``, ``unreadable`` or ``present`` with size and SHA-256."""
    facts: dict[str, Any] = {"path": os.path.abspath(path), "state": "absent", "bytes": None, "sha256": None}
    try:
        raw = pathlib.Path(path).read_bytes()
    except FileNotFoundError:
        return facts
    except OSError as exc:
        return {**facts, "state": "unreadable", "cause": f"os_error:{type(exc).__name__}"}
    return {**facts, "state": "present", "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def _encode(record: dict[str, Any]) -> bytes:
    return (json.dumps(record, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def _fsync_dir(directory: pathlib.Path) -> None:
    try:
        fd = os.open(directory, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def publish_exclusive(path: pathlib.Path, record: dict[str, Any]) -> bool:
    """Publish complete bytes once; ``False`` when the name already exists.

    A hard link from a synced private file makes creation exclusive and atomic. Filesystems
    without hard links fall back to exclusive creation, where a torn write reads as invalid
    and therefore still blocks another effect."""
    data = _encode(record)
    staged = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    fd = os.open(staged, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        os.write(fd, data)
        os.fsync(fd)
    finally:
        os.close(fd)
    try:
        os.link(staged, path)
    except FileExistsError:
        return False
    except OSError:
        try:
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        except FileExistsError:
            return False
        try:
            os.write(fd, data)
            os.fsync(fd)
        finally:
            os.close(fd)
    finally:
        staged.unlink(missing_ok=True)
    _fsync_dir(path.parent)
    return True


def _load(path: pathlib.Path) -> tuple[str, dict[str, Any] | None]:
    facts, value = _read_object(path)
    return facts["state"], value


def claim_protocol(run_root: pathlib.Path) -> str:
    """Only a readable pre-protocol manifest proves that an unclaimed file is legacy.

    The current launcher always writes FLAG, even when the diagnostic is off. A
    missing or unreadable manifest cannot license scoring an unclaimed old file.
    """
    state, manifest = _load(run_root / "run_manifest.json")
    harness = manifest.get("harness") if state == "parsed" else None
    config = harness.get("applied_config") if isinstance(harness, dict) else None
    if not isinstance(config, dict):
        return "unknown"
    return "current" if FLAG in config else "legacy"


def _valid_id(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 32 and all(c in "0123456789abcdef" for c in value)


def _read_claim(path: pathlib.Path, kind: str) -> tuple[str, dict[str, Any] | None]:
    """``absent``, ``invalid`` or ``valid`` plus the claim."""
    state, claim = _load(path)
    if state == "absent":
        return "absent", None
    if state != "parsed" or claim.get("kind") != kind or not _valid_id(claim.get("attempt_id")):
        return "invalid", None
    return "valid", claim


def read_receipt(task_root: pathlib.Path, attempt_id: str, *, kind: str = "official") -> dict[str, Any] | None:
    """The terminal receipt of exactly this attempt, never a name-alike from elsewhere."""
    if not _valid_id(attempt_id):
        return None
    prefix = RECEIPT_PREFIX if kind == "official" else DIAGNOSTIC_RECEIPT_PREFIX
    state, receipt = _load(task_root / f"{prefix}{attempt_id}.json")
    if state != "parsed" or receipt.get("kind") != kind or receipt.get("attempt_id") != attempt_id:
        return None
    if kind == "official" and not isinstance(receipt.get("official_run"), bool):
        return None
    return receipt


def task_dump_root(dump_dir: str, task_dir: str) -> pathlib.Path:
    """``TaskConfig``'s single-turn dump root under ``./dumps``, the launcher's ledger path."""
    parts = pathlib.PurePath(task_dir).parts
    return pathlib.Path(os.path.abspath(os.path.join("dumps", dump_dir, *parts[:-1], f"SingleUserTurn-{parts[-1]}")))


def official_bindings(task_root: pathlib.Path, task_dir: str, *, config_path: str, entry_path: str) -> dict[str, Any]:
    return {
        "task_dir": task_dir, "task_root": str(task_root),
        "log": file_facts(task_root / GATE_LOG_NAME), "config": file_facts(pathlib.Path(config_path)),
        "evaluator": file_facts(pathlib.Path(UPSTREAM_EVALUATOR)),
        "entry": file_facts(pathlib.Path(entry_path)), "helper": file_facts(pathlib.Path(__file__)),
    }


def _settle(task_root: pathlib.Path, claim: dict[str, Any], *, official_run: bool, cause: str = "",
            returned: Any = None, raised: str | None = None) -> dict[str, Any]:
    receipt = {"schema": SCHEMA, "kind": "official", "attempt_id": claim["attempt_id"],
               "bindings": claim["bindings"], "official_run": official_run, "cause": cause,
               "raised": raised, "returned": None, "returned_unserializable": None,
               "result_file": file_facts(task_root / RESULT_NAME), "completed_at": time.time()}
    try:
        _encode({"returned": returned})
        receipt["returned"] = returned
    except (TypeError, ValueError):
        receipt["returned_unserializable"] = type(returned).__name__
    try:
        publish_exclusive(task_root / f"{RECEIPT_PREFIX}{claim['attempt_id']}.json", receipt)
    except OSError as exc:
        # The official effect stands; only reentry loses its replay and stays refused.
        print(f"[ouroboros] official attempt receipt not persisted: {type(exc).__name__}", file=sys.stderr, flush=True)
    return receipt


def official_attempt(task_root: pathlib.Path, bindings: dict[str, Any], evidence: dict[str, Any],
                     prepare: Callable[[], Any], evaluate: Callable[[Any], Any]) -> tuple[dict[str, Any], bool]:
    """Return ``(record, ran_here)``; ``ran_here`` is true only when this call made the effect.

    ``prepare`` may only fail before the official effect: ``NotRun`` settles quietly, any
    other exception settles and propagates. An exception from ``evaluate`` settles as raised
    and propagates, exactly like the unwrapped call did."""
    claim_path = task_root / CLAIM_NAME
    state, claim = _read_claim(claim_path, "official")
    if state == "absent":
        prior = file_facts(task_root / RESULT_NAME)
        fresh = {"schema": SCHEMA, "kind": "official", "attempt_id": uuid.uuid4().hex,
                 "created_at": time.time(), "eval_host": socket.gethostname(), "pid": os.getpid(),
                 "bindings": bindings, "evidence": {**evidence, "prior_result": prior}}
        task_root.mkdir(parents=True, exist_ok=True)
        if publish_exclusive(claim_path, fresh):
            if prior["state"] != "absent":
                # Preserve legacy/unprovenanced bytes: no evaluator, no overwrite, no diagnostic.
                return _settle(task_root, fresh, official_run=False, cause="unclaimed_prior_result"), False
            return _run(task_root, fresh, prepare, evaluate), True
        state, claim = _read_claim(claim_path, "official")
    if state != "valid":
        return {"state": "refused", "cause": "claim_unreadable", "attempt_id": None}, False
    if claim.get("bindings") != bindings:
        return {"state": "refused", "cause": "binding_mismatch", "attempt_id": claim["attempt_id"]}, False
    receipt = read_receipt(task_root, claim["attempt_id"])
    if receipt is None:
        # In flight, killed or unpublished: unknown, and never a reason for another effect.
        return {"state": "refused", "cause": "unfinished_claim", "attempt_id": claim["attempt_id"]}, False
    return receipt, False


def _run(task_root: pathlib.Path, claim: dict[str, Any], prepare: Callable[[], Any],
         evaluate: Callable[[Any], Any]) -> dict[str, Any]:
    try:
        target = prepare()
    except NotRun as exc:
        return _settle(task_root, claim, official_run=False, cause=str(exc))
    except Exception as exc:
        _settle(task_root, claim, official_run=False, cause=f"preparation_error:{type(exc).__name__}")
        raise
    try:
        returned = evaluate(target)
    except BaseException as exc:
        _settle(task_root, claim, official_run=True, raised=type(exc).__name__)
        raise
    return _settle(task_root, claim, official_run=True, returned=returned)


def official_verdict(record: dict[str, Any]) -> dict[str, Any] | None:
    returned = record.get("returned")
    return returned if record.get("official_run") is True and isinstance(returned, dict) else None


def refusal_line(record: dict[str, Any]) -> str:
    """One runner-log line that matches neither ``Status:`` nor ``Pass:`` greps."""
    attempt = record.get("attempt_id") or "unknown"
    if record.get("state") == "refused":
        return f"[ouroboros] official evaluation not repeated: {record['cause']}; attempt {attempt}"
    if record.get("official_run") is False:
        return f"[ouroboros] official evaluator not run: {record.get('cause')}; attempt {attempt}"
    ended = record.get("raised") or record.get("returned_unserializable") or type(record.get("returned")).__name__
    return f"[ouroboros] official evaluation ended without a usable returned result ({ended}); attempt {attempt}"


# --------------------------------------------------------------------------- host readers


def attempt_facts(task_dump: pathlib.Path) -> dict[str, Any]:
    """Compact official-attempt facts for the ledger, without the returned payload text."""
    state, claim = _read_claim(task_dump / CLAIM_NAME, "official")
    if state != "valid":
        return {"state": state}
    facts = {"state": "claimed", "attempt_id": claim["attempt_id"],
             "agent_container_state": (claim.get("evidence") or {}).get("agent_container_state")}
    receipt = read_receipt(task_dump, claim["attempt_id"])
    if receipt is None:
        return facts
    result_file = receipt.get("result_file")
    result_file = result_file if isinstance(result_file, dict) else {}
    current, payload = _read_object(task_dump / RESULT_NAME)
    # The claim alone does not authenticate an old/foreign eval_res.json. A scored
    # verdict requires the same bytes that the completed effect observed AND the
    # returned value from that effect; exceptions cannot borrow a pre-existing pass.
    file_matches_returned = (receipt["official_run"] is True and not receipt.get("raised")
                             and isinstance(receipt.get("returned"), dict)
                             and result_file.get("state") == "present"
                             and current["state"] == "parsed"
                             and current["sha256"] == result_file.get("sha256")
                             and payload == receipt["returned"])
    return {**facts, "state": "terminal", "official_run": receipt["official_run"],
            "cause": receipt.get("cause") or "", "raised": receipt.get("raised"),
            "result_sha256": result_file.get("sha256"),
            "file_matches_returned": file_matches_returned}


def diagnostic_facts(task_dump: pathlib.Path) -> dict[str, Any]:
    state, claim = _read_claim(task_dump / DIAGNOSTIC_CLAIM_NAME, "diagnostic")
    if state != "valid":
        return {"state": state}
    receipt = read_receipt(task_dump, claim["attempt_id"], kind="diagnostic")
    if receipt is None:
        return {"state": "claimed", "attempt_id": claim["attempt_id"]}
    keys = ("outcome", "reason", "eligible", "reason_code", "elapsed_sec", "cap_sec", "custody",
            "checker_exit_code", "command_sha256", "log_path")
    return {"state": "terminal", "attempt_id": claim["attempt_id"], **{key: receipt.get(key) for key in keys}}


# --------------------------------------------------------------------------- diagnostic parent


def _group_alive(pgid: int) -> bool:
    """Whether any non-zombie process remains in the group; unknown counts as alive."""
    proc = pathlib.Path("/proc")
    if proc.is_dir():
        for entry in proc.iterdir():
            if not entry.name.isdigit():
                continue
            try:
                stat = (entry / "stat").read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            fields = stat[stat.rfind(")") + 2:].split()
            if len(fields) > 2 and fields[2] == str(pgid) and fields[0] not in {"Z", "X"}:
                return True
        return False
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    except OSError:
        return True
    return True


def _stop_group(proc: subprocess.Popen) -> bool:
    """End the owned group, including descendants of an exited leader; prove their death."""
    for sig in (None, signal.SIGTERM, signal.SIGKILL):
        if sig is not None:
            try:
                os.killpg(proc.pid, sig)
            except ProcessLookupError:
                pass
            except OSError:
                return False
        deadline = time.monotonic() + (TEARDOWN_GRACE_SEC if sig is not None else 0)
        while True:
            if proc.poll() is not None and not _group_alive(proc.pid):
                return True
            if time.monotonic() >= deadline:
                break
            time.sleep(0.05)
    return False


def run_residual_diagnostic(task_root: pathlib.Path, attempt_id: str, *,
                            cap_sec: float = DIAGNOSTIC_CAP_SEC) -> dict[str, Any] | None:
    """Never prints and never raises: the official lines and exit code are already final."""
    try:
        return _run_diagnostic(task_root, attempt_id, min(float(cap_sec), DIAGNOSTIC_CAP_SEC))
    except Exception:
        return None


def _run_diagnostic(task_root: pathlib.Path, attempt_id: str, cap_sec: float) -> dict[str, Any] | None:
    base = {"schema": SCHEMA, "kind": "diagnostic", "attempt_id": attempt_id, "semantics": SEMANTICS,
            "cap_sec": cap_sec, "official_verdict_unchanged": True}
    if not publish_exclusive(task_root / DIAGNOSTIC_CLAIM_NAME, {**base, "created_at": time.time()}):
        return None
    log_path = task_root / f"{DIAGNOSTIC_LOG_PREFIX}{attempt_id}.log"
    report_path = task_root / f"{DIAGNOSTIC_REPORT_PREFIX}{attempt_id}.json"
    receipt: dict[str, Any] = {**base, "log_path": str(log_path)}
    started = time.monotonic()
    try:
        with open(log_path, "xb") as log:
            # fd 1/2 are this log before the child interpreter imports or prints anything.
            proc = subprocess.Popen(
                [sys.executable, "-u", os.path.abspath(__file__), "diagnose", str(task_root), attempt_id],
                stdin=subprocess.DEVNULL, stdout=log, stderr=log, start_new_session=True,
            )
    except Exception as exc:
        receipt.update(outcome="diagnostic_error", reason=f"spawn_error:{type(exc).__name__}")
        publish_exclusive(task_root / f"{DIAGNOSTIC_RECEIPT_PREFIX}{attempt_id}.json", receipt)
        return receipt
    timed_out = False
    try:
        proc.wait(timeout=cap_sec)
    except subprocess.TimeoutExpired:
        timed_out = True
    dead = _stop_group(proc)
    _, report = _load(report_path)
    report = report or {}
    receipt.update({key: report.get(key) for key in (
        "eligible", "reason_code", "checker_exit_code", "command", "command_sha256")})
    receipt.update(elapsed_sec=round(time.monotonic() - started, 1), exit_code=proc.returncode,
                   custody="group_dead" if dead else "unconfirmed", report_stage=report.get("stage"))
    if not dead:
        receipt.update(outcome="unknown", reason="process_group_death_unconfirmed")
    elif timed_out:
        receipt.update(outcome="timeout", reason="diagnostic_cap_reached")
    elif report.get("stage") == "final" and report.get("outcome"):
        receipt.update(outcome=report["outcome"], reason=report.get("reason") or "")
    else:
        receipt.update(outcome="diagnostic_error", reason="no_final_report")
    publish_exclusive(task_root / f"{DIAGNOSTIC_RECEIPT_PREFIX}{attempt_id}.json", receipt)
    return receipt


# --------------------------------------------------------------------------- diagnostic child


class Unavailable(Exception):
    """Eligible by policy, but the linked inputs or command cannot be reconstructed."""


def _write_report(path: pathlib.Path, record: dict[str, Any]) -> None:
    staged = path.with_name(f".{path.name}.tmp")
    staged.write_bytes(_encode(record))
    os.replace(staged, path)


def eligibility(task_root: pathlib.Path, claim: dict[str, Any], receipt: dict[str, Any] | None) -> tuple[str, str]:
    """``("", reason_code)`` when eligible, else ``(ineligible_reason, reason_code)``."""
    if receipt is None or receipt.get("official_run") is not True:
        return "official_not_completed", ""
    log_facts, log = _read_object(task_root / GATE_LOG_NAME)
    if log_facts["state"] != "parsed" or log_facts["sha256"] != claim["bindings"]["log"]["sha256"]:
        return "log_changed_or_unreadable", ""
    official, _ = read_official_receipt(task_root)
    result_facts, result = _read_object(task_root / RESULT_NAME)
    gate = official.get("gate") or {}
    if (official["official_eval_status"] != "declined" or gate.get("log_sha256") != log_facts["sha256"]
            or result_facts["sha256"] != (receipt.get("result_file") or {}).get("sha256")
            or receipt.get("returned") != result):
        return "not_same_attempt_status_gate_decline", ""
    summary_facts, summary = _read_object(task_root / "ouroboros_summary.json")
    if summary_facts["state"] != "parsed":
        return "adapter_summary_unavailable", ""
    reason_code = summary.get("reason_code") if isinstance(summary.get("reason_code"), str) else ""
    if summary.get("task") != claim["bindings"]["task_dir"] or summary.get("bench_status") != log.get("status"):
        return "adapter_summary_not_linked", reason_code
    if summary.get("infra_failed") is not False:
        return "infrastructure_outcome", reason_code
    if summary.get("task_submission_started") is not True:
        return "no_task_submission", reason_code
    if reason_code not in ELIGIBLE_REASON_CODES:
        return "stop_reason_not_eligible", reason_code
    activity = summary.get("model_activity_observed")
    if activity is not True:
        return ("no_agent_activity" if activity is False else "agent_activity_unknown"), reason_code
    if (claim.get("evidence") or {}).get("agent_container_state") != "absent":
        return "agent_termination_unproven", reason_code
    return "", reason_code


def evaluation_command(task_root: pathlib.Path, task_dir: str) -> str:
    """The pinned upstream ``TaskEvaluator.evaluate_one`` command, from the saved task config.

    Mirrors ``0717376/cowork_bench@d943e75`` ``utils/evaluation/evaluator.py``: saved
    ``TaskConfig.from_dict``, ``Evaluation.build`` fallback when the command or groundtruth
    is None, the first two ``launch_time`` tokens, identical argument text."""
    from utils.data_structures.task_config import Evaluation, TaskConfig

    _, log = _read_object(task_root / GATE_LOG_NAME)
    saved = (log or {}).get("config")
    # ``TaskConfig.__post_init__`` deletes ``<task_root>/eval_res.json``; only a prefixed
    # single-turn root makes that path differ from the official result it must not touch.
    if not isinstance(saved, dict) or saved.get("single_turn_mode") is not True:
        raise Unavailable("saved_config_unsupported")
    task_config = TaskConfig.from_dict(saved)
    if task_config.task_dir != task_dir or os.path.abspath(task_config.log_file) != str(task_root / GATE_LOG_NAME):
        raise Unavailable("saved_config_not_linked")
    groundtruth_workspace = task_config.evaluation.groundtruth_workspace
    eval_command = task_config.evaluation.evaluation_command
    launch_time = " ".join((task_config.launch_time or "").split()[:2])
    if eval_command is None or groundtruth_workspace is None:
        fresh = Evaluation.build(task_config.task_dir, cn_mode=getattr(task_config, "cn_mode", False))
        eval_command = eval_command or fresh.evaluation_command
        groundtruth_workspace = groundtruth_workspace or fresh.groundtruth_workspace
    if eval_command is None:
        # Upstream turns a missing command into pass True; a diagnostic never does.
        raise Unavailable("no_eval_command")
    args = (f"--res_log_file {task_config.log_file} --agent_workspace {task_config.agent_workspace} "
            f"--groundtruth_workspace {groundtruth_workspace} --launch_time \"{launch_time}\"")
    return f"{eval_command} {args}"


def diagnose(task_root: pathlib.Path, attempt_id: str, report_path: pathlib.Path) -> dict[str, Any]:
    state, claim = _read_claim(task_root / CLAIM_NAME, "official")
    if state != "valid" or claim["attempt_id"] != attempt_id:
        return {"outcome": "unavailable", "reason": "official_claim_not_linked", "eligible": None}
    reason, reason_code = eligibility(task_root, claim, read_receipt(task_root, attempt_id))
    if reason:
        return {"outcome": "ineligible", "reason": reason, "reason_code": reason_code, "eligible": False}
    try:
        command = evaluation_command(task_root, claim["bindings"]["task_dir"])
    except Unavailable as exc:
        return {"outcome": "unavailable", "reason": str(exc), "reason_code": reason_code, "eligible": True}
    started = {"eligible": True, "reason_code": reason_code, "command": command,
               "command_sha256": hashlib.sha256(command.encode("utf-8")).hexdigest()}
    _write_report(report_path, {**started, "stage": "checker_started"})
    # Same shell, cwd and environment as upstream ``run_command``; output stays in this log.
    code = subprocess.run(command, shell=True, stdin=subprocess.DEVNULL).returncode  # noqa: S602
    return {**started, "outcome": "checks_passed" if code == 0 else "checks_failed", "checker_exit_code": code}


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 3 or args[0] != "diagnose":
        print("usage: eval_attempt.py diagnose TASK_ROOT ATTEMPT_ID", file=sys.stderr)
        return 2
    task_root, attempt_id = pathlib.Path(args[1]), args[2]
    # The entrypoint resolves upstream `utils` from its /workspace directory, the eval cwd.
    sys.path.insert(0, os.getcwd())
    report_path = task_root / f"{DIAGNOSTIC_REPORT_PREFIX}{attempt_id}.json"
    try:
        result = diagnose(task_root, attempt_id, report_path)
    except Exception as exc:  # noqa: BLE001 - the traceback belongs in this diagnostic log
        traceback.print_exc()
        result = {"outcome": "diagnostic_error", "reason": f"exception:{type(exc).__name__}"}
    _write_report(report_path, {**result, "stage": "final"})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
