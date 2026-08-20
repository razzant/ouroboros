"""Step/round budget accounting and refusal gates of the cu_bridge runner.

Verbatim extraction from ``run_cu_bridge_agent.py`` (v7 stream W): the declared
step budget and its audit, the worker round cap and its publication, the
task-scoped proxy configuration and its liveness check, the setup-effect
verification, the dataset-commit and uncapped-step-claim refusals, and the
disclosure counters.
"""

from __future__ import annotations

import contextlib
import json
import os
from pathlib import Path
from typing import Any

from devtools.benchmarks.osworld.cu_bridge_gate import _GATE_TURN_RESERVE, _policy_turns
from devtools.benchmarks.osworld.cu_bridge_runtime import SKILL_NAME
from devtools.benchmarks.osworld.cu_bridge_tool_policy import _GUI_ACTION_TOOLS

def _effective_max_rounds(settings_path: Path) -> dict[str, Any]:
    """Report the round budget the bench server actually honors, with provenance.

    The server applies settings.json over env at startup, so settings wins; this
    is best-effort disclosure, not enforcement (there is no per-task step cap)."""
    try:
        settings = json.loads(Path(settings_path).read_text(encoding="utf-8"))
        if isinstance(settings, dict) and settings.get("OUROBOROS_MAX_ROUNDS") is not None:
            return {"value": int(settings["OUROBOROS_MAX_ROUNDS"]), "source": "settings"}
    except Exception:
        pass
    env_val = os.environ.get("OUROBOROS_MAX_ROUNDS")
    if env_val:
        try:
            return {"value": int(env_val), "source": "env"}
        except ValueError:
            pass
    return {"value": 200, "source": "default"}


def _step_budget(args: Any, effective_rounds: dict[str, Any]) -> dict[str, Any]:
    """Typed step-budget provenance for the manifest (never raises).

    A leaderboard "step" is ONE TOP-LEVEL POLICY TURN: the official loop
    increments ``step_idx`` once per ``agent.predict()`` and executes every
    action that call emitted inside that one step
    (``lib_run_single.py`` on the graded pin), so a turn that emits four
    clicks is one step, not four. Our ``llm_rounds`` is therefore the
    step-equivalent — and the earlier "0.42 GUI actions per round" mapping
    compared a turn against an action and understated our budget by ~2.4x.

    The declared budget covers EVERY policy turn the example consumes: the
    read-only gate phase (a separate task, measured mean 4.1 / max 14 turns on
    the v6.81.1 run) plus the working phase plus one reserved tool-less
    terminal turn, so a forced finalization cannot become step N+1.
    """
    claimed = max(0, int(getattr(args, "max_steps", 0) or 0))
    gate_reserve = _GATE_TURN_RESERVE if getattr(args, "feasibility_gate", False) else 0
    terminal_reserve = 1
    worker_cap = claimed - gate_reserve - terminal_reserve if claimed else 0
    return {
        "step_semantics": "top_level_policy_turn",
        "step_definition_ref": "OSWorld lib_run_single.py: step_idx += 1 per agent.predict()",
        "max_steps_claimed": claimed or None,
        "enforced": bool(claimed),
        "gate_turn_reserve": gate_reserve,
        "terminal_turn_reserve": terminal_reserve,
        "action_capable_round_cap": worker_cap or None,
        "server_round_cap": effective_rounds,
    }


@contextlib.contextmanager
def _official_evaluate_cwd(osworld_root: Path):
    """Evaluate with the checkout root as CWD, exactly like the official runner.

    Evaluator fixtures are declared RELATIVE to the checkout
    (``{"type": "local_file", "path": "evaluation_examples/examples/.../x_gold.txt"}``)
    and ``get_local_file`` tests that string with a bare ``os.path.exists``, so the
    grader silently resolves it against the PROCESS CWD. The official harness runs
    from the checkout root and never notices; this bridge does not, and the getter
    then returns None — a task whose answer was byte-exact scores 0 with only a
    line in the lane log (measured: multi_apps/7f35355e produced the correct
    25.27 and still scored 0.0).

    Scoped to the evaluate call and restored on every path. It exists ONLY to
    resolve relative fixture paths: the env's cache root is passed absolute at
    construction, so nothing else is allowed to depend on this window.
    """
    previous = os.getcwd()
    try:
        os.chdir(str(osworld_root))
        yield
    finally:
        try:
            os.chdir(previous)
        except OSError:  # noqa: BLE001 - the original cwd vanished; nothing to restore to
            pass


def _worker_round_cap(budget: dict[str, Any], gate_turns: int | None) -> int | None:
    """Turns the WORKER may use, once the gate's actual consumption is known.

    The static reserve is worst-case: the gate is budgeted 14 turns but spent a
    mean of 4 on the v6.83.0 run, so a flat ``max_steps - 14 - 1`` threw away
    ~10 turns of every example and 13 of 56 opus failures died at 89-92 total
    turns inside a 100-turn budget. Returning the UNUSED reserve keeps the
    declared total intact (gate + worker + 1 terminal <= max_steps) while giving
    long-horizon tasks the turns they were always entitled to.

    None when no budget is declared (nothing to enforce).
    """
    claimed = int(budget.get("max_steps_claimed") or 0)
    if not claimed:
        return None
    # UNKNOWN is not zero: an unreadable gate count must keep the worst-case
    # reserve, otherwise a worker could take claimed-1 turns after an
    # unmeasured gate and blow the declared total.
    used = int(gate_turns) if gate_turns is not None else int(budget.get("gate_turn_reserve") or 0)
    return max(1, claimed - used - int(budget.get("terminal_turn_reserve") or 1))


def _publish_worker_round_cap(settings_path: Path, cap: int) -> dict[str, Any]:
    """Write the worker's round cap into the lane settings the server hot-reloads.

    ``Agent.handle_task`` re-applies settings from disk at the start of EVERY
    task, so writing this between the gate and the worker is what makes the cap
    per-phase without a per-task API. Adapter-only: no core contract changes.
    Never raises here; the CALLER aborts the attempt on failure, because a cap
    left over from an earlier task on this lane may be LARGER than this example
    allows — an unapplied write is an unknown budget, not a safe one.
    """
    record: dict[str, Any] = {"requested": int(cap), "applied": False}
    try:
        path = Path(settings_path)
        settings = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
        if not isinstance(settings, dict):
            record["error"] = "settings.json is not an object"
            return record
        record["previous"] = settings.get("OUROBOROS_MAX_ROUNDS")
        settings["OUROBOROS_MAX_ROUNDS"] = int(cap)
        # Unique temp (a fixed sibling collides between lanes) and the ORIGINAL
        # mode preserved: this file carries provider credentials and is 0600, but
        # a fresh write would take the process umask (0664 here).
        mode = path.stat().st_mode & 0o777 if path.is_file() else 0o600
        tmp = path.with_name(f"{path.name}.{os.getpid()}.part")
        tmp.write_text(json.dumps(settings, ensure_ascii=False, indent=2), encoding="utf-8")
        os.chmod(tmp, mode)
        tmp.replace(path)
        record["applied"] = True
    except Exception as exc:  # noqa: BLE001 - disclosure, never fatal
        record["error"] = f"{type(exc).__name__}: {exc}"
    return record


def _proxy_trace_shows_exhaustion(data_dir: Path, task_id: str) -> bool:
    """True if this task's tool trace carries a proxy-exhaustion signature (never raises).

    Scans the same tools.jsonl the counters read. A 407 TRAFFIC_EXHAUSTED (or a bare
    407) inside a proxy:true task means the residential upstream ran out mid-run;
    that is an infra fault to quarantine, not an agent failure to score.
    """
    # TASK-LOCAL ONLY. The lane-wide aggregate carries every earlier task on the
    # same server, so falling back to it quarantined later tasks for a neighbour's
    # outage (3 of them were wins in the previous run). No task id, no verdict.
    path = data_dir / "state" / "headless_tasks" / task_id / "data" / "logs" / "tools.jsonl"
    if not task_id or not path.is_file():
        return False
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                # The unambiguous upstream signature only. A bare "407" appears in
                # page content and ordinary prose; matching it read origin data as
                # proxy failure.
                if "TRAFFIC_EXHAUSTED" in line:
                    return True
    except OSError:
        return False
    return False


def _verify_setup_effect(env: Any, example: dict[str, Any]) -> dict[str, Any]:
    """Check that the task's setup COMMANDS actually succeeded (never raises).

    Upstream's ``SetupController._execute_setup`` treats any HTTP 200 from the guest
    as success and never inspects the command's exit status, so a setup step that
    fails inside the VM is logged as "Command executed successfully". Measured on
    chrome/3299584d: the task's ``apt install jq`` silently did nothing, the premise
    the gate had verified was gone by the time the worker ran, the agent honestly
    reported the task impossible and scored 0 — while doing nothing at all would
    have scored 1.

    We re-run each setup ``execute`` step's own command as a READ-ONLY presence
    probe where that is meaningful (a package/binary the step installs), and report
    what we found. Advisory: the caller records it in the manifest rather than
    failing the task, because a false alarm here would cost a scored task.
    """
    report: dict[str, Any] = {"checked": 0, "missing": []}
    try:
        for step in (example.get("config") or []):
            if not isinstance(step, dict) or step.get("type") != "execute":
                continue
            cmd = step.get("parameters", {}).get("command")
            parts = cmd if isinstance(cmd, list) else str(cmd or "").split()
            # Stop at the first shell separator: a string command like
            # `apt-get install -y jq && tar xf archive.tgz` otherwise probes `&&`,
            # the archive path and `rm` as if they were installed binaries.
            for sep in ("&&", "||", ";", "|"):
                if sep in parts:
                    parts = parts[:parts.index(sep)]
            if "install" not in parts:
                continue
            tail = [p for p in parts[parts.index("install") + 1:]
                    if not p.startswith("-") and "/" not in p and "." not in p][:4]
            for pkg in tail:
                report["checked"] += 1
                try:
                    out = env.controller.execute_python_command(
                        f"import shutil,sys; sys.stdout.write('1' if shutil.which({pkg!r}) else '0')"
                    )
                    if "1" not in str((out or {}).get("output", "")):
                        report["missing"].append(pkg)
                except Exception:  # noqa: BLE001 - probe only
                    pass
    except Exception as exc:  # noqa: BLE001 - never fail a task on diagnostics
        report["error"] = f"{type(exc).__name__}: {exc}"
    return report


def _task_scoped_proxy_config(config_path: str, state_dir: Path, tag: str) -> str:
    """Write a task-local proxy config whose username carries a sticky session id.

    The shared config is a single entry on the rotating gateway, so every lane of
    every concurrent campaign draws a fresh exit IP per request. That breaks any
    site that ties a session to an address (a search that re-challenges, a booking
    flow that loses its cart) and concentrates all our traffic on one account's
    reputation. DataImpulse binds a session with a ``;sessid.<value>`` suffix on
    the username, so one task keeps one exit for its whole trajectory while
    different tasks land on different exits.

    Written to a LANE-PRIVATE state directory, never under ``results/``: the file
    contains the account password and the results tree is what gets published.
    Returns the new path, or the original on any failure — a proxy we could not
    scope is still better than none, and this must never fail a task.
    """
    try:
        entries = json.loads(Path(config_path).read_text(encoding="utf-8"))
        if not isinstance(entries, list) or not entries:
            return config_path
        scoped = []
        for e in entries:
            e = dict(e)
            user = str(e.get("username") or "")
            if user and ";sessid." not in user:
                e["username"] = f"{user};sessid.{tag}"
            scoped.append(e)
        # NEVER under results/: that tree is the publication artefact and this file
        # carries the account password. Lane-private state dir only.
        state_dir.mkdir(parents=True, exist_ok=True)
        out = state_dir / f"proxy_{tag}.json"
        out.write_text(json.dumps(scoped, indent=2), encoding="utf-8")
        os.chmod(out, 0o600)
        return str(out)
    except Exception:  # noqa: BLE001 - fall back to the shared config
        return config_path


def _proxy_config_is_live(config_path: str, *, timeout: float = 20.0) -> bool:
    """Probe the FIRST proxy in the config with a real HTTPS CONNECT (never raises).

    Config-exists is not proxy-alive: an exhausted DataImpulse account keeps its
    file but answers 407 TRAFFIC_EXHAUSTED. A dead proxy scores proxy:true tasks
    worse than no proxy, so this gate decides whether to route through it at all.
    Fails CLOSED (returns False) on any error — better to run those tasks direct
    and quarantine them than to poison them through a dead upstream.
    """
    try:
        import json as _json
        import urllib.request
        entries = _json.loads(open(config_path, encoding="utf-8").read())
        if not isinstance(entries, list) or not entries:
            return False
        e = entries[0]
        user = str(e.get("username") or "")
        pwd = str(e.get("password") or "")
        host = str(e.get("host") or "")
        port = int(e.get("port") or 0)
        if not (host and port):
            return False
        auth = f"{user}:{pwd}@" if user else ""
        proxy_url = f"http://{auth}{host}:{port}"
        opener = urllib.request.build_opener(
            urllib.request.ProxyHandler({"http": proxy_url, "https": proxy_url})
        )
        with opener.open("https://api.ipify.org", timeout=timeout) as resp:
            body = resp.read(64).decode("ascii", "replace").strip()
        # A residential exit returns an IP; a dead account returns nothing usable.
        return bool(body) and body.count(".") == 3
    except Exception:  # noqa: BLE001 - any failure is a dead proxy for our purposes
        return False


def _refuse_wrong_dataset_commit(expected: str, checkout: dict[str, Any]) -> None:
    """Refuse a checkout that is not the one the campaign is graded against.

    The graded-spec pin decides BOTH the instruction handed to the agent and the
    evaluator that scores it, so it is a gate, not a manifest footnote. Empty
    ``expected`` keeps the old report-only behaviour for exploratory runs; a
    campaign passes it (``--expect-dataset-commit`` / ``OSWORLD_EXPECT_COMMIT``)
    and any drift then costs nothing because it stops before the VM boots.
    """
    want = str(expected or "").strip().lower()
    if not want:
        return
    got = str((checkout or {}).get("git_commit") or "").strip().lower()
    if not got:
        raise SystemExit(
            "--expect-dataset-commit was given but the OSWorld checkout has no readable git "
            f"identity ({checkout!r}); refusing rather than grading against an unknown spec"
        )
    if not (got.startswith(want) or want.startswith(got)):
        raise SystemExit(
            f"OSWorld checkout is {got[:12]} but this campaign is graded against {want[:12]}; "
            "point --osworld-root at the campaign checkout (a different checkout supplies "
            "different task instructions AND a different evaluator)"
        )


def _refuse_uncapped_step_claim(budget: dict[str, Any]) -> None:
    """Refuse a step claim the bench server would not actually honor.

    Enforcement lives in the RUNTIME cap (the loop refuses to open a round past
    ``OUROBOROS_MAX_ROUNDS``), so the runner's job is to prove that cap is at or
    below the declared budget BEFORE anything costs money. A post-hoc "most
    tasks finished early" argument cannot substitute: comparability is a
    per-task property.
    """
    if not budget.get("enforced"):
        return
    # The runner republishes the worker cap after the gate (see
    # `_publish_worker_round_cap`), so the base setting only has to be within the
    # declared total; the per-phase value is what the loop actually enforces.
    worker_cap = int(budget.get("max_steps_claimed") or 0) - int(budget.get("terminal_turn_reserve") or 1)
    if worker_cap < 1:
        raise SystemExit(
            f"--max-steps={budget.get('max_steps_claimed')} leaves no working turns after the "
            f"gate ({budget.get('gate_turn_reserve')}) and terminal ({budget.get('terminal_turn_reserve')}) "
            "reserves"
        )
    server = budget.get("server_round_cap") or {}
    server_value = int(server.get("value") or 0)
    if server_value > worker_cap:
        raise SystemExit(
            f"server round cap {server_value} (source: {server.get('source')}) exceeds the "
            f"{worker_cap} action-capable turns implied by --max-steps="
            f"{budget.get('max_steps_claimed')}; set OUROBOROS_MAX_ROUNDS={worker_cap} in the "
            "lane settings.json so the declared budget is the one the runtime enforces"
        )


def _audit_step_budget(budget: dict[str, Any], worker_turns: int | None,
                       gate_turns: int | None, *, gate_expected: bool = False) -> dict[str, Any]:
    """Post-run check that the example actually stayed inside the declared budget.

    Both inputs are POLICY turns from the loop's own accounting (see
    ``_policy_turns``), never the flat physical-call field — those disagree on
    almost every example and the flat one runs higher.

    An overrun here is a HARNESS FAULT, not a filtering criterion: enforcement
    is supposed to make it unreachable (the runtime cap bounds the worker, the
    runner cancels the gate at its reserve), so seeing one means the enforcement
    drifted. Excluding such an example from the scored denominator would quietly
    shrink the denominator the methodology fixes at the attempted-task count, so
    the audit reports ``budget_fault`` and the CAMPAIGN is what must be treated
    as non-comparable — a decision for the operator, not a silent per-row drop.
    Missing counts fail CLOSED (unknown is not compliance).
    """
    if not budget.get("enforced"):
        return {"audited": False, "reason": "no step budget declared"}
    claimed = int(budget.get("max_steps_claimed") or 0)
    if worker_turns is None or (gate_expected and gate_turns is None):
        missing = "worker" if worker_turns is None else "gate"
        return {"audited": True, "counts_available": False, "budget_fault": True,
                "reason": f"{missing} policy turn count unavailable",
                "max_steps_claimed": claimed}
    total = int(worker_turns) + int(gate_turns or 0)
    return {
        "audited": True,
        "counts_available": True,
        "turn_source": "loop_outcome.usage.total_rounds",
        "policy_turns_used": total,
        "worker_turns": int(worker_turns),
        "gate_turns": int(gate_turns or 0),
        "max_steps_claimed": claimed,
        "within_budget": total <= claimed,
        "budget_fault": total > claimed,
    }


def _collect_budget_counters(data_dir: Path, latest: dict[str, Any], ouro_task_id: str) -> dict[str, Any]:
    """Disclosure counters for leaderboard comparability (never raises).

    A leaderboard "step" is one model turn; our rounds are not step-equivalent,
    so we publish the raw counts: llm rounds (authoritative, from the task
    result) plus per-tool call counts parsed from the task's own tools.jsonl.
    """
    from ouroboros.extension_loader import extension_name_prefix

    # `llm_rounds` is the FLAT task-result field: physical model calls (safety
    # checks, acceptance reviewers and retries included), kept for continuity
    # with earlier runs. `policy_turns` is the loop's own turn count and is the
    # step-equivalent — the two disagree on nearly every example.
    counters: dict[str, Any] = {
        "llm_rounds": int(latest.get("total_rounds") or 0),
        "physical_model_calls": int(latest.get("total_rounds") or 0),
        "policy_turns": _policy_turns(latest),
    }
    prefix = extension_name_prefix(SKILL_NAME)
    child = latest.get("child_drive_root")
    log_path = (Path(child) / "logs" / "tools.jsonl") if child else (
        data_dir / "state" / "headless_tasks" / ouro_task_id / "data" / "logs" / "tools.jsonl"
    )
    fallback = data_dir / "logs" / "tools.jsonl"
    screenshots = gui = remote_exec = total = 0
    src = log_path if log_path.is_file() else (fallback if fallback.is_file() else None)
    if src is not None:
        for line in src.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            if not isinstance(row, dict) or row.get("type") != "tool_call":
                continue
            if src is fallback and str(row.get("task_id") or "") != ouro_task_id:
                continue
            tool = str(row.get("tool") or "")
            if not tool.startswith(prefix):
                continue
            short = tool[len(prefix):]
            total += 1
            if short == "screenshot":
                screenshots += 1
            elif short == "remote_exec":
                remote_exec += 1
            elif short in _GUI_ACTION_TOOLS:
                gui += 1
    counters.update({
        "screenshots": screenshots,
        "gui_action_calls": gui,
        "remote_exec_calls": remote_exec,
        "skill_tool_calls": total,
        "tools_log": str(src) if src is not None else "",
    })
    return counters
