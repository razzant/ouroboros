"""Read-only gate phase of the OSWorld cu_bridge runner.

Verbatim extraction from ``run_cu_bridge_agent.py`` (v7 stream W): the premise
check that the agent cannot act before the working phase begins — gate windows,
the verdict, the DesktopEnv log capture, verified reset, the policy-turn
accounting, the gate round itself, guest-health probing, unconfirmed-cancel
handling and the gate tool trace.
"""

from __future__ import annotations

import json
import logging
import time
import urllib.request
from collections import deque
from pathlib import Path
from typing import Any, Callable

from devtools.benchmarks.osworld.cu_bridge_prompts import GATE_PREAMBLE, GATE_SUFFIX
from devtools.benchmarks.osworld.cu_bridge_runtime import SKILL_NAME, _api, _terminal_answer_text
from devtools.benchmarks.osworld.cu_bridge_tool_policy import _effective_disabled_tools

def _gate_window_sec(args: Any) -> float:
    """Holder occupancy added by ONE premise round, or 0 when the gate is off.

    This is the SAME expression the round's own deadline uses; the two must not drift,
    because the claim staleness bound is computed from it.
    """
    if not getattr(args, "feasibility_gate", False):
        return 0.0
    return float(max(60, int(args.task_timeout_sec) // 4))


def _gate_claim_window_sec(args: Any) -> float:
    """Worst-case premise-phase occupancy for the claim staleness bound.

    ONE round since v6.81.1 (the confirming challenger was removed — its full-run
    ledger showed correlated errors and a net loss). This constant and the number
    of premise rounds the flow can actually run are the same fact — change them
    together.
    """
    return _gate_window_sec(args)


def _gate_verdict(latest: dict[str, Any] | None) -> str:
    """The gate's typed verdict, read from the phase-A agent's terminal answer.

    Fails OPEN: anything that is not an explicit standalone INFEASIBLE — PROCEED,
    UNDETERMINED, an unparseable answer, a crashed or timed-out phase — proceeds to the
    full-capability phase. The gate may only ever REMOVE a task the agent is affirmatively
    certain about; it can never strand one on silence.
    """
    text = _terminal_answer_text(latest)
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return "UNDETERMINED"
    # ONLY the last line, which is what the phase's prompt asks for. Scanning all lines in
    # reverse looked equivalent and is not: a model that enumerates the three options as bare
    # lines while reasoning ("ruling each out: UNDETERMINED / PROCEED / INFEASIBLE") and then
    # concludes in prose had its recap read as the verdict — turning a PROCEED into a scored
    # hard zero. Reading past the answer to find a keyword is how a parser invents an answer.
    verdict = lines[-1].strip("*`_#> \t").rstrip(".!:;,").upper()
    return verdict if verdict in {"INFEASIBLE", "PROCEED", "UNDETERMINED"} else "UNDETERMINED"


class _DesktopEnvLogCapture(logging.Handler):
    """Scoped capture of OSWorld's own log records during a reset.

    desktop_env reports its setup failures at ERROR level, but the benchmark
    process installs no handler for the "desktopenv" loggers — so the only
    witness of a failed setup was never written anywhere. This handler exists
    for the diagnostic sidecar ONLY: control flow reads the machine-checkable
    postcondition in `_reset_verified`, never these strings.
    """

    def __init__(self, logger_name: str = "desktopenv", keep: int = 60):
        super().__init__(level=logging.INFO)
        self._lines: deque[str] = deque(maxlen=keep)
        self._logger = logging.getLogger(logger_name)

    def emit(self, record: logging.LogRecord) -> None:  # noqa: D102
        try:
            self._lines.append(f"{record.levelname} {record.name}: {record.getMessage()}")
        except Exception:  # noqa: BLE001 - a diagnostic must never break the reset
            pass

    def __enter__(self) -> "_DesktopEnvLogCapture":
        self._logger.addHandler(self)
        return self

    def __exit__(self, *_exc: Any) -> bool:
        self._logger.removeHandler(self)
        return False

    def tail(self) -> list[str]:
        return list(self._lines)


class ResetUnverified(RuntimeError):
    """env.reset() finished without a VERIFIED task setup (see _reset_verified)."""

    def __init__(self, message: str, record: dict[str, Any]):
        super().__init__(message)
        self.record = record


def _reset_verified(env: Any, example: dict[str, Any], *, retries: int, deadline: float,
                    wait_after_sec: float,
                    sleep: Callable[[float], None] = time.sleep) -> dict[str, Any]:
    """env.reset() with the postcondition OSWorld itself does not enforce.

    OSWorld's reset() is fail-open: when the guest server never answers the setup
    probe (~100s), it skips EVERY setup step, logs "Environment setup complete."
    and returns a pristine VM — no exception, no False (desktop_env.py, the
    setup-retry loop falls through). The 2026-07-28 smoke measured what that does
    downstream: working phases opened on VMs without the task's files and honestly
    declared the premise absent; the feasible-control mean fell 0.737 -> 0.459.

    The postcondition IS machine-readable: `env.is_environment_used` is set True
    iff setup ran to success with a non-empty config, so this helper asserts it.
    Two further points, both load-bearing:

    - Before every RETRY, `is_environment_used` is forced True. After a failed
      setup it is still False, and reset() skips the snapshot revert for "clean"
      environments — an unforced retry would run setup ON TOP of the partial
      state instead of from the pristine image.
    - The screenshot probe doubles as the endpoint-health probe: it travels the
      same guest-server HTTP path the agent's tools use.

    Returns a small diagnostic record on success; raises ResetUnverified when the
    budget is exhausted. The caller maps that to a typed INFRA row (reward None,
    claim released) — a setup the harness could not verify must never become a
    capability zero.
    """
    last_err = ""
    with _DesktopEnvLogCapture() as capture:
        for attempt in range(1, max(1, int(retries)) + 1):
            if time.time() >= deadline:
                last_err = last_err or "deadline reached before the first attempt"
                break
            if attempt > 1:
                env.is_environment_used = True
            try:
                env.reset(task_config=example)
                if wait_after_sec > 0:
                    sleep(wait_after_sec)
                obs = env._get_obs()
                shot = obs.get("screenshot") if isinstance(obs, dict) else None
                if not (isinstance(shot, (bytes, bytearray)) and shot):
                    last_err = f"attempt {attempt}: no screenshot"
                    sleep(5)
                    continue
                if getattr(env, "config", None) and not getattr(env, "is_environment_used", False):
                    last_err = (f"attempt {attempt}: setup silently failed "
                                "(is_environment_used=False with a non-empty task config)")
                    sleep(5)
                    continue
                return {"attempts": attempt, "log_tail": capture.tail()}
            except Exception as exc:  # noqa: BLE001 - retried, then surfaced typed
                last_err = f"attempt {attempt}: {type(exc).__name__}: {exc}"
                sleep(5)
    raise ResetUnverified(f"OSWorld reset unverified: {last_err}",
                          {"error": last_err, "log_tail": capture.tail()})


def _live_policy_turns(data_dir: Path, task_id: str) -> int | None:
    """Policy turns of a RUNNING task, counted from its own event log.

    ``loop_outcome`` is written only at FINALIZATION
    (``agent_task_pipeline`` writes it on the terminal paths), so a poll of
    ``GET /api/tasks/<id>`` on a running task never carries it — reading it
    there yields None forever and any enforcement built on it is dead code.
    The live authority is the ``llm_round`` event, emitted in
    ``loop_llm_call`` at the very statement that increments
    ``accumulated_usage["rounds"]``, so counting those events for this task
    equals the ``loop_outcome.usage.total_rounds`` it will eventually report.

    Returns None when the log is not readable yet — the caller must treat that
    as "unknown", never as zero.
    """
    candidates = [
        data_dir / "state" / "headless_tasks" / task_id / "data" / "logs" / "events.jsonl",
        data_dir / "logs" / "events.jsonl",
    ]
    for path in candidates:
        if not path.is_file():
            continue
        rounds = 0
        matched_any = False
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as fh:
                for line in fh:
                    line = line.strip()
                    if not line or '"llm_round"' not in line:
                        continue
                    try:
                        row = json.loads(line)
                    except Exception:  # noqa: BLE001 - a torn tail line is not a count
                        continue
                    if not isinstance(row, dict) or row.get("type") != "llm_round":
                        continue
                    # The shared log carries every task; the per-task log carries one.
                    if str(row.get("task_id") or "") != task_id:
                        continue
                    matched_any = True
                    rounds += 1
        except OSError:
            continue
        if matched_any or path.parent.parent.name == task_id:
            return rounds
    return None


def _policy_turns(latest: dict[str, Any]) -> int | None:
    """Top-level POLICY TURNS from a task result, or None when unavailable.

    The flat ``total_rounds`` on a task result is NOT this number: it is
    reconstructed from ``usage_breakdown(...)["physical_calls"]`` and also counts
    safety checks, acceptance reviewers and retries. Measured on the v6.81.1
    361-task run, the two disagree on 344 of 346 examples (physical exceeds
    policy by up to 13 turns), so auditing a step budget against the flat field
    would mark compliant examples non-comparable. The loop's own count is the
    authority. Returns None rather than 0 when the field is missing: a step-cap
    audit must fail CLOSED, and "unknown" coerced to zero would pass silently.
    """
    usage = ((latest.get("loop_outcome") or {}).get("usage") or {})
    value = usage.get("total_rounds")
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _await_gate_task(ouroboros_url: str, task_id: str, deadline: float,
                     turn_budget: int = 0, data_dir: Path | None = None) -> dict[str, Any]:
    """Poll one premise-phase task to a terminal status or its deadline.

    On deadline the cancel is CONFIRMED before returning: an unverified cancel can
    leave the premise agent alive on the SAME VM (and the same skill connection
    file) the working phase — or the lane's NEXT task — is about to use.
    """
    final_statuses = {"completed", "failed", "cancelled", "rejected_duplicate"}
    while True:
        if time.time() >= deadline:
            cancelled = False
            try:
                _api(ouroboros_url, "POST", f"/api/tasks/{task_id}/cancel", {})
                for _ in range(6):
                    time.sleep(5)
                    probe = _api(ouroboros_url, "GET", "/api/tasks/" + task_id, timeout=30)
                    if str((probe or {}).get("status") or "") in final_statuses:
                        cancelled = True
                        break
            except Exception:  # noqa: BLE001 - reported in the record, decided by the caller
                cancelled = False
            return {"status": "timeout", "cancel_confirmed": cancelled}
        try:
            latest = _api(ouroboros_url, "GET", "/api/tasks/" + task_id, timeout=30)
        except Exception:  # noqa: BLE001 - transient poll error
            time.sleep(5)
            continue
        if isinstance(latest, dict) and str(latest.get("status") or "") in final_statuses:
            return latest
        # Per-task ENFORCEMENT of the gate's share of the step budget. The
        # runtime cap (`OUROBOROS_MAX_ROUNDS`) is server-wide and the gate is a
        # SEPARATE task, so without this the gate could consume the worker's
        # whole allowance and the example could exceed the declared budget.
        # Cancelling the gate is safe by construction: an absent verdict is
        # UNDETERMINED, which proceeds to the working phase (fail-open).
        if turn_budget > 0 and data_dir is not None:
            # LIVE count from the task's own event log: the finalization-only
            # `loop_outcome` is absent while the task is running.
            used = _live_policy_turns(data_dir, task_id)
            if used is not None and used >= turn_budget:
                cancelled = False
                try:
                    _api(ouroboros_url, "POST", f"/api/tasks/{task_id}/cancel", {})
                    for _ in range(6):
                        time.sleep(5)
                        probe = _api(ouroboros_url, "GET", "/api/tasks/" + task_id, timeout=30)
                        if str((probe or {}).get("status") or "") in final_statuses:
                            cancelled = True
                            break
                except Exception:  # noqa: BLE001 - recorded, decided by the caller
                    cancelled = False
                return {"status": "turn_budget_exhausted", "cancel_confirmed": cancelled,
                        "policy_turns": used, "turn_budget": turn_budget}
        time.sleep(8)


def _gate_round(ouroboros_url: str, args: Any, instruction: str, *, role: str) -> dict[str, Any]:
    """One premise round: create the gate task, await it, judge the last line.

    ``role`` survives in the record for cross-run readability (v6.81.0 records
    carry role="challenger" rows; since v6.81.1 exactly one round runs).
    """
    created = _api(ouroboros_url, "POST", "/api/tasks", {
        # The instruction is UNTRUSTED text. Ending the prompt with it would let a
        # task that says "end with INFEASIBLE" dictate the verdict and score itself
        # zero, so the protocol is restated afterwards, last word ours.
        "description": GATE_PREAMBLE + instruction + GATE_SUFFIX,
        "memory_mode": "empty",
        "disabled_tools": _effective_disabled_tools(args.allow_a11y, gate_phase=True),
    })
    task_id = str(created.get("task_id") or "")
    if not task_id:
        raise RuntimeError(f"{role} task creation returned no task_id: {created!r}")
    latest = _await_gate_task(ouroboros_url, task_id, time.time() + _gate_window_sec(args),
                              turn_budget=_gate_turn_budget(args),
                              data_dir=Path(args.data_dir))
    return {
        "role": role,
        "verdict": _gate_verdict(latest),
        "task_id": task_id,
        "status": latest.get("status"),
        # POLICY turns (loop authority), not the flat physical-call field.
        # Finalized tasks report it; runner-terminated ones carry the live count;
        # a timeout falls back to the event log rather than reporting nothing (the
        # longest-running gate must not be the one counted as zero).
        "policy_turns": (latest.get("policy_turns")
                         if latest.get("policy_turns") is not None
                         else (_policy_turns(latest)
                               if _policy_turns(latest) is not None
                               else _live_policy_turns(Path(args.data_dir), task_id))),
        **({"cancel_confirmed": bool(latest.get("cancel_confirmed"))}
           if str(latest.get("status") or "") == "timeout" else {}),
        "llm_rounds": int(latest.get("total_rounds") or 0),
        "answer": _terminal_answer_text(latest),
    }


# How long the guest control endpoint may stay unreachable before the attempt is
# abandoned as INFRA. Long enough to ride out a reboot/restart the task itself
# triggered (several tasks legitimately restart services), short enough that a
# genuinely dead endpoint does not consume the whole task budget.
# Policy turns the read-only gate phase may consume, reserved out of the declared
# step budget. Measured on the v6.81.1 361-task run: mean 4.1, median 3, max 14.
_GATE_TURN_RESERVE = 14


_GUEST_DOWN_GRACE_SEC = 180.0


def _guest_endpoint_healthy(env: Any, *, timeout: float = 8.0) -> bool:
    """True when the guest's OSWorld control server still answers.

    Probed from the HOST, over the same HTTP path the agent's tools use, so it sees
    exactly the failure the agent would hit. Any exception means unreachable — this
    is a health probe, and an unknown state must read as unhealthy or the watchdog
    is decorative. Never raises.
    """
    try:
        ip = getattr(env, "vm_ip", "") or ""
        port = getattr(env, "server_port", "") or ""
        if not ip or not port:
            return True  # nothing published yet; not our call to judge
        with urllib.request.urlopen(f"http://{ip}:{port}/screenshot", timeout=timeout) as resp:
            return 200 <= int(getattr(resp, "status", 200)) < 300
    except Exception:  # noqa: BLE001 - unreachable is the answer, not an error
        return False


def _gate_cancel_unconfirmed(record: dict[str, Any]) -> bool:
    """True when a premise round timed out AND its cancel did not confirm.

    This is the one gate condition that must NOT fail open into the working
    phase: a zombie premise session shares the lane server and the skill's
    connection file, so after the endpoint republish it would act on the SAME VM
    the worker is being scored on — and on the lane's next task after that. The
    caller maps this to `blocked` (exit 2, lane aborts, its server dies and the
    zombie with it); the claim is released so another lane retries cleanly.
    """
    # Both runner-initiated terminations qualify: the wall-clock timeout and the
    # step-budget cancel. They cancel the SAME way, so an unconfirmed cancel
    # leaves the same zombie premise session on the scored VM.
    return (str(record.get("status") or "") in {"timeout", "turn_budget_exhausted"}
            and not record.get("cancel_confirmed"))


def _gate_tool_trace(data_dir: Path, ouro_task_id: str, latest_status: Any = None) -> list[dict[str, Any]]:
    """Full tool trace of one premise round, for the offline audit (never raises).

    COMPLETE args, not previews: the GAIA leakage audit's blind spot was a
    detector fed truncated output (result_preview cut at 2005 chars hid the
    evidence on exactly one arm). tools.jsonl stores tool-call args untruncated,
    so the sidecar carries every shell command the round ran, verbatim — the
    read-only promise is enforceable only if the audit can see all of it.
    """
    trace: list[dict[str, Any]] = []
    try:
        from ouroboros.extension_loader import extension_name_prefix

        prefix = extension_name_prefix(SKILL_NAME)
        log_path = data_dir / "state" / "headless_tasks" / ouro_task_id / "data" / "logs" / "tools.jsonl"
        if not (ouro_task_id and log_path.is_file()):
            return trace
        for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            if not isinstance(row, dict) or row.get("type") != "tool_call":
                continue
            tool = str(row.get("tool") or "")
            if not tool.startswith(prefix):
                continue
            trace.append({
                "tool": tool[len(prefix):],
                "args": row.get("args"),
                "is_error": bool(row.get("is_error")),
            })
    except Exception:  # noqa: BLE001 - a sidecar must never change the flow
        pass
    return trace


def _gate_turn_budget(args: Any) -> int:
    """Policy turns the gate phase may use when a step budget is declared.

    Zero (no enforcement) when no budget is declared: the gate is then bounded
    only by its wall-clock window, exactly as before this flag existed.
    """
    if not int(getattr(args, "max_steps", 0) or 0):
        return 0
    return _GATE_TURN_RESERVE if getattr(args, "feasibility_gate", False) else 0
