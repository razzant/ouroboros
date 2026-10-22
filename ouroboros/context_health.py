"""Health-invariant probes and the log-analysis checks they read.

Extracted from ``context.py`` (v6.90.x submarine unwind) so the context builder
stays under the hard module gate.  This module is a LEAF: it reads runtime logs,
version carriers and custody ledgers and renders the "## Health Invariants"
section.  It must never import ``ouroboros.context`` — ``context`` re-exports
``safe_read`` / ``build_health_invariants`` so every historical import site
(``ouroboros.context.safe_read``, ``ouroboros.consciousness_wake``, the tests that
monkeypatch ``context._STRAY_PROBE_CACHE``) keeps working unchanged.
"""

from __future__ import annotations

import logging
import json
import pathlib
import re
from collections import Counter
from typing import Any, Dict, List

from ouroboros.context_budget import SCRATCHPAD_BLOAT_WARN_CHARS
from ouroboros.utils import iter_jsonl_objects, read_json_dict, read_text
from ouroboros.store import StoreError
from ouroboros.context_timing import timed_section

log = logging.getLogger(__name__)


def safe_read(path: pathlib.Path, fallback: str = "") -> str:
    try:
        exists = path.exists()
    except Exception:
        log.debug("safe_read: path.exists() raised for %s", path, exc_info=True)
        return fallback
    if not exists:
        return fallback
    try:
        return read_text(path)
    except Exception as exc:
        log.warning("safe_read: file %s exists but read failed (%s: %s); using fallback", path, type(exc).__name__, exc)
        return fallback


def _iter_recent_jsonl(path: pathlib.Path, max_bytes: int = 256_000):
    yield from iter_jsonl_objects(path, tail_bytes=max_bytes)


def _collect_log_analysis_checks(env: Any, checks: List[str]) -> None:
    import hashlib
    import time as _time

    try:
        msg_hash_to_tasks: Dict[str, set] = {}
        for log_path, type_field, type_value in (
            (env.drive_path("logs/events.jsonl"), "type", "owner_message_injected"),
            (env.drive_path("logs/supervisor.jsonl"), "event_type", "owner_message_injected"),
        ):
            for ev in _iter_recent_jsonl(log_path):
                if ev.get(type_field) != type_value:
                    continue
                text = ev.get("text", "")
                if not text and "event_repr" in ev:
                    event_repr = str(ev.get("event_repr", ""))
                    text = event_repr[:200] + f" [...{len(event_repr) - 200} chars omitted]" if len(event_repr) > 200 else event_repr
                if text:
                    task_ids = msg_hash_to_tasks.setdefault(hashlib.md5(text.encode()).hexdigest()[:12], set())
                    task_ids.add(ev.get("task_id") or "unknown")
        dupes = {h: tids for h, tids in msg_hash_to_tasks.items() if len(tids) > 1}
        if dupes:
            checks.append(f"CRITICAL: DUPLICATE PROCESSING — {len(dupes)} message(s) appeared in multiple tasks: {', '.join(str(sorted(tids)) for tids in dupes.values())}")
        else:
            checks.append("OK: no duplicate message processing detected")
    except Exception:
        pass

    try:  # who changed sent history under the running code; a provider's cache share is not a health verdict
        from ouroboros.transcript_prefix import history_changes_line

        line = history_changes_line(getattr(env, "budget_drive_root", None) or getattr(env, "drive_root", None))
        if line:
            checks.append(line)
    except Exception:
        log.debug("sent-history change fact unavailable", exc_info=True)

    try:
        events_path = env.drive_path("logs/events.jsonl")
        llm_error_models: Counter = Counter()
        local_overflow_models: Counter = Counter()
        remote_overflow_models: Counter = Counter()
        for ev in _iter_recent_jsonl(events_path):
            evt_type = str(ev.get("type") or "")
            model = str(ev.get("model") or "unknown")
            if evt_type in {"llm_api_error", "review_model_error", "provider_incomplete_response"}:
                llm_error_models[model] += 1
            elif evt_type == "local_context_overflow":
                local_overflow_models[model] += 1
            elif evt_type == "remote_context_overflow":
                remote_overflow_models[model] += 1
        if llm_error_models:
            top = ", ".join(f"{model} x{count}" for model, count in llm_error_models.most_common(3))
            checks.append(f"WARNING: PROVIDER/ROUTING ERRORS — {sum(llm_error_models.values())} recent failures ({top}). Reliability or failover may need attention.")
        else:
            checks.append("OK: no recent provider/routing errors")
        if local_overflow_models:
            top = ", ".join(f"{model} x{count}" for model, count in local_overflow_models.most_common(3))
            checks.append(f"WARNING: LOCAL CONTEXT OVERFLOW — {sum(local_overflow_models.values())} recent overflow event(s) ({top}). Local context may need more compaction or a larger window.")
        else:
            checks.append("OK: no recent local context overflows")
        if remote_overflow_models:
            top = ", ".join(f"{model} x{count}" for model, count in remote_overflow_models.most_common(3))
            checks.append(f"WARNING: REMOTE CONTEXT OVERFLOW — {sum(remote_overflow_models.values())} recent provider context-window rejection(s) ({top}). Switch to low context mode or reduce the prompt footprint before retrying the same request.")
    except Exception:
        pass

    try:
        rescue_dir = env.drive_path("archive/rescue")
        if rescue_dir.exists():
            recent = []
            now = _time.time()
            for entry in sorted(rescue_dir.iterdir(), reverse=True):
                if not entry.is_dir():
                    continue
                age_sec = now - entry.stat().st_mtime
                if age_sec < 7200:
                    file_count = sum(1 for item in entry.rglob("*") if item.is_file())
                    age_str = f"{int(age_sec // 60)}m ago" if age_sec < 3600 else f"{age_sec / 3600:.1f}h ago"
                    recent.append(f"{entry.name} ({age_str}, {file_count} files)")
                if len(recent) >= 3:
                    break
            if recent:
                checks.append(
                    f"WARNING: RESCUE SNAPSHOT AVAILABLE — {', '.join(recent)}. "
                    "Uncommitted changes were saved before last restart. "
                    "Use read_file(root='runtime_data', path='archive/rescue/<dirname>/rescue_meta.json') "
                    "and changes.diff to decide if recovery is needed."
                )
    except Exception:
        pass



# Live stray-server probe cache: pgrep + per-pid /proc reads are cheap but not
# free on every task turn; a 15-minute TTL keeps the invariant LIVE (the v6.70.0
# field incident was weeks-long, not minutes-long) without a per-turn scan.
_STRAY_PROBE_CACHE: Dict[str, Any] = {"ts": 0.0, "note": ""}
_STRAY_PROBE_TTL_SEC = 900


def _stray_server_note(env: Any) -> str:
    import time as _time

    now = _time.time()
    if now - float(_STRAY_PROBE_CACHE["ts"]) < _STRAY_PROBE_TTL_SEC:
        return str(_STRAY_PROBE_CACHE["note"])
    from ouroboros.agent_startup_checks import check_stray_server_processes

    result, issues = check_stray_server_processes(env)
    note = ""
    if issues and result.get("status") == "stray_processes":
        procs = result.get("processes") or []
        preview = ", ".join(str(p.get("pid")) for p in procs[:5])
        note = (
            f"WARNING: STRAY SERVER PROCESS(ES) — {len(procs)} ouroboros server process(es) "
            f"outside this install (pids: {preview}). Another install may be mutating shared state; "
            f"investigate before assuming exclusive custody."
        )
    _STRAY_PROBE_CACHE["ts"] = now
    _STRAY_PROBE_CACHE["note"] = note
    return note


def _plan_review_note(env: Any, task_id: str) -> str:
    """This task's plan asks still awaiting answers (a snapshot, not a live lease); a
    legacy v2 wave with pending custody keeps its own note."""
    if not task_id:
        return ""
    try:
        from ouroboros import plan_record
        from ouroboros.task_results import current_plan_review_wave, load_plan_review_state

        root = getattr(env, "drive_root", None) or env.drive_path("state").parent
        record = plan_record.load_plan_record(root, task_id)
        pending = plan_record.pending_asks(record)
        if pending:
            seats = [str(p.get("seat_id") or "?") for ask in pending for p in ask.get("participants") or [] if p.get("status") == "pending"]
            return (f"PLAN ASK OPEN: {', '.join(str(a.get('ask_id') or '?') for a in pending)} — "
                    f"{len(seats)} participant(s) have not answered yet ({', '.join(seats)}); silence holds nothing. "
                    "Answers arrive in the mailbox; plan_task() reads them.")
        if plan_record.engaged(record):
            return ""
        wave = current_plan_review_wave(load_plan_review_state(root, task_id)) or {}
        if wave.get("closed") or not wave.get("custody_pending"):
            return ""
        pending = sum(1 for actor in wave.get("actors") or [] if isinstance(actor, dict)
                      and actor.get("operation_state") in {"pending_dispatch", "in_flight"})
        return (
            f"PLAN REVIEW WAVE OPEN: {str(wave.get('request_fingerprint') or '?')[:8]}, "
            f"{pending} reviewer slot(s) recorded pending at {wave.get('reviewed_at') or '?'}. "
            "Work may still be running or awaiting collection."
        )
    except (OSError, ValueError, TimeoutError, StoreError):
        log.warning("Unable to read plan-review health for %s", task_id, exc_info=True)
        return ""


def _memory_health_lines(env: Any) -> List[str]:
    """Own-memory maintenance signals for the two authored files, identity and scratchpad.

    The old dialogue writer's state lines (unreadable cursor, open nominations, a failed
    run, a withheld era) left with that writer: its pending nominations are imported as
    marks the mind releases, and an unreadable cursor is the import's ``cursor_gap``.
    """
    import time as _time

    lines: List[str] = []
    try:
        identity_path = env.drive_path("memory/identity.md")
        if identity_path.exists():
            age_hours = (_time.time() - identity_path.stat().st_mtime) / 3600
            if age_hours > 8:
                lines.append(f"WARNING: STALE IDENTITY — identity.md last updated {age_hours:.0f}h ago")
            else:
                lines.append("OK: identity.md recent")
    except Exception:
        pass
    try:
        identity_content = read_text(env.drive_path("memory/identity.md"))
        if len(identity_content.strip()) < 200:
            lines.append(f"WARNING: THIN IDENTITY — identity.md is only {len(identity_content)} chars. Cognitive decay signal.")
    except Exception:
        pass

    try:
        sp_len = len(read_text(env.drive_path("memory/scratchpad.md")).strip())
        if sp_len < 50:
            lines.append("WARNING: EMPTY SCRATCHPAD — scratchpad is nearly empty. Memory loss signal.")
        elif sp_len > SCRATCHPAD_BLOAT_WARN_CHARS:
            lines.append(f"WARNING: BLOATED SCRATCHPAD — {sp_len} chars. Extract durable insights to knowledge base.")
        else:
            lines.append(f"OK: scratchpad size ({sp_len} chars)")
    except Exception:
        pass
    return lines


def _unread_output_line(custody_root: Any, run: Any, task_id: str) -> str:
    """The D7 NEVER-READ line for one settled, complete, unread staged output of the reader's own run.

    Another task's runs fold into ``_foreign_work_line``; an unattributed reader (empty ``task_id``)
    may be the owner and reads this line too. A retained (#1573) output names its exact canonical read
    address; a legacy row keeps its task-drive wording byte for byte.
    """
    from ouroboros.delegate_output import staged_output_address

    address = staged_output_address(custody_root, run) if run.output_source else {}
    how = f"root='{address['root']}' at that path" if address else "root='task_drive'"
    action = (
        f"Read it with read_file {how} until the artifact is "
        "covered end to end, or say plainly that the result was not "
        "collected."
    )
    if getattr(run, "terminal_source", None) and not run.unread_disclosed:
        return (f"INFO: DELEGATED RESULT AVAILABLE — run {run.run_id or '?'} has retained full output at "
                f"{address.get('path') or run.output_artifact}, not yet read by its owner. "
                "Body collection proves availability, not acceptance. " + action)
    return (f"WARNING: DELEGATED RESULT NEVER READ — run {run.run_id or '?'} settled "
            f"with its full output staged at {address.get('path') or run.output_artifact} and never read to "
            f"EOF (owner task {run.task_id or '?'}). {action}")


def _foreign_owner(run: Any, task_id: str) -> str:
    """The run's owner task when this reader is ANOTHER task, else "" (its own run or an unattributed reader)."""
    return str(run.task_id) if task_id and run.task_id and task_id != run.task_id else ""


def _foreign_work_line(root: Any, foreign: Dict[str, Dict[str, List[str]]]) -> str:
    """One line for all foreign obligations; the exact list uses the existing blob reader.

    No seen-state: identical lists reuse content-addressed source bytes. If retention fails,
    keep the list in the line rather than publishing an unreadable pointer.
    """
    from ouroboros.observability import write_blob

    try:
        ref = write_blob(root, foreign)
        source = f"query_data(op='blob', sha256='{ref['sha256']}')"
    except OSError:
        source = "retention unavailable; exact list: " + json.dumps(foreign, ensure_ascii=False, sort_keys=True)
    severity = "WARNING" if any(work["patches"] for work in foreign.values()) else "INFO"
    return (f"{severity}: OTHER TASKS' DELEGATED WORK — full owner/run list (unread results, undisposed patches, "
            f"'yours' = matching active target): {source}. Only the owner acknowledges results and decides patches "
            "while live; after it ends, a live top-level task may reject from any root or apply from the recorded "
            "target or a fresh payload binding via integrate_delegated_patch.")


# One line per own settled run, bounded like ``terminal_custody_notice``.
_OWN_RUN_OUTCOME_LINES = 10


def _own_run_outcome_lines(custody_root: Any, custody_state: Any, task_id: str,
                           attempt_key: str) -> List[str]:
    """The reader's OWN settled runs whose terminal payload its current transcript lacks.

    #1573: a run the sweep settled while its owner was parked (or that it never
    waited on) left no fact in the owner's context. From the ONE custody state the
    other delegated-run checks already read: own (never a foreign or review-owned
    run), settled terminal, recorded by a writer that also records the terminal
    acknowledgement (``engine_locator`` is not None — before it, absence proved
    nothing), not acknowledged in THIS attempt, and not already a D7 line. Facts and
    locators only, never output text or a command. A retained payload is called
    full only when its custody says ``output_complete``; an unverified engine
    preview keeps its address but is named incomplete. The line leaves the next
    fresh context build after the run's terminal wake is appended in this attempt.
    """
    if not task_id or not attempt_key or custody_state is None:
        return []
    from ouroboros.delegate_custody import TERMINAL_STATES, settled_output_unread
    from ouroboros.delegate_output import staged_output_address

    runs = sorted((run for run in custody_state.values()
                   if run.task_id == task_id and run.settled and not run.review_owned
                   and run.terminal_state in TERMINAL_STATES and run.engine_locator is not None
                   and run.terminal_ack_attempt != attempt_key and not settled_output_unread(run)),
                  key=lambda run: (run.settled_at, run.run_id), reverse=True)
    lines = []
    for run in runs[:_OWN_RUN_OUTCOME_LINES]:
        locator = run.engine_locator or {}
        primary = locator.get("primary_output") if isinstance(locator.get("primary_output"), dict) else {}
        engine = [f"run folder {locator['run_dir']}"] if locator.get("run_dir") else []
        if primary:
            engine.append("primary output " + " ".join(str(part) for part in (
                primary.get("kind"), primary.get("path"),
                f"({primary['bytes']} bytes)" if "bytes" in primary else None,
                "(truncated preview)" if primary.get("truncated") else None) if part))
        address = staged_output_address(custody_root, run)
        retained = ("retained full output" if run.output_complete else
                    "retained INCOMPLETE payload (unverified engine preview, never acknowledgeable as read)")
        lines.append(
            f"INFO: OWN DELEGATED RUN ENDED — run {run.run_id} {run.terminal_state}"
            + (f" ({run.terminal_reason})" if run.terminal_reason else "")
            + (f" at {run.settled_at}" if run.settled_at else "")
            + "; at context capture, no terminal-wake acknowledgement was recorded for this attempt"
            + ("; engine-owned, may be reclaimed: " + "; ".join(engine) if engine else "")
            + (f"; {retained}: read_file root='{address['root']}' path='{address['path']}'"
               if address else "")
            + ".")
    if len(runs) > _OWN_RUN_OUTCOME_LINES:
        lines.append(f"INFO: OWN DELEGATED RUNS ENDED — +{len(runs) - _OWN_RUN_OUTCOME_LINES} older "
                     "settled runs also lacked this attempt's acknowledgement at context capture.")
    return lines


@timed_section("health")
def build_health_invariants(env: Any, task_id: str = "", active_root: str = "",
                            attempt_key: str = "") -> str:
    """Render the health-invariant WARNING block for one reader's context.

    Called once when a task's context is built (a task-start snapshot, not
    refreshed per round).

    ``task_id`` names the READING task so delegated-run obligations follow
    ownership: they stay globally visible (owner doctrine D7 — a
    preserved-and-invisible result is how work rots on disk), the OWNER task
    receives each run's line with its call-shaped instruction, and another
    task's runs fold into ONE line per owner task with their run ids, the
    authority rule and a run's exact reader (owner decision Q3=A, 18.10.2026).
    Empty ``task_id`` (legacy callers) keeps every line call-shaped — an
    unattributed reader may be the owner.

    ``active_root`` names that reader's own active Git root, which this module
    cannot look up (it has no ``ctx``) and the caller already holds. One
    comparison through the apply gate's own predicate then names, inside the
    owner's line, the concrete call for another task's patches the reader's
    root already satisfies.

    ``attempt_key`` is the reader's actual task attempt, normalized exactly as the
    supervision acknowledgement records it; with ``task_id`` it names the reader's
    OWN settled runs whose terminal payload this attempt's transcript never took
    (``_own_run_outcome_lines``). Empty renders none of those lines.
    """
    checks: List[str] = []

    try:
        from ouroboros.claudexor_exit_facts import health_line
        engine_exit = health_line(env.drive_path("logs/supervisor.jsonl").parent.parent)
        if engine_exit:
            checks.append(engine_exit)
    except Exception:
        log.debug("Engine exit observation unavailable", exc_info=True)

    try:
        from ouroboros.tools.release_sync import (
            _normalize_pep440,
            _shields_escape,
            extract_architecture_header_version,
            extract_readme_badge_version,
            is_release_version,
        )
        ver_file = read_text(env.repo_path("VERSION")).strip()
        desync_parts = []
        pyproject_ver = next(
            (
                line.split("=", 1)[1].strip().strip('"').strip("'")
                for line in read_text(env.repo_path("pyproject.toml")).splitlines()
                if line.strip().startswith("version")
            ),
            "",
        )
        if is_release_version(ver_file) and pyproject_ver and _normalize_pep440(ver_file) != pyproject_ver:
            desync_parts.append(f"pyproject.toml={pyproject_ver}")
        try:
            web_package = read_text(env.repo_path("web/package.json"))
            web_match = re.search(r'"version"\s*:\s*"([^"]+)"', web_package)
            web_ver = str(web_match.group(1) or "").strip() if web_match else ""
            if is_release_version(ver_file) and web_ver and web_ver != ver_file:
                desync_parts.append(f"web/package.json={web_ver}")
        except Exception:
            pass
        try:
            readme = read_text(env.repo_path("README.md"))
            badge_ver = extract_readme_badge_version(readme)
            rm = None if badge_ver else re.search(r'\*\*Version:\*\*\s*([^\s]+)', readme)
            readme_ver = badge_ver or (str(rm.group(1) or "").strip() if rm else "")
            badge_token_ok = not (badge_ver and is_release_version(ver_file)) or f"version-{_shields_escape(ver_file)}-green" in readme
            if readme_ver and readme_ver != ver_file:
                desync_parts.append(f"README={readme_ver}")
            elif readme_ver and not badge_token_ok:
                desync_parts.append("README badge URL token")
        except Exception:
            pass
        try:
            arch = read_text(env.repo_path("docs/ARCHITECTURE.md"))
            arch_ver = extract_architecture_header_version(arch)
            if arch_ver and arch_ver != ver_file:
                desync_parts.append(f"ARCHITECTURE.md={arch_ver}")
        except Exception:
            pass
        if desync_parts:
            checks.append(f"CRITICAL: VERSION DESYNC — VERSION={ver_file}, {', '.join(desync_parts)}")
        elif ver_file:
            checks.append(f"OK: version sync ({ver_file})")
    except Exception:
        pass

    try:
        state_data = read_json_dict(env.drive_path("state/state.json")) or {}
        from ouroboros.usage_accounting import usage_breakdown

        accounted = float(usage_breakdown(env.drive_root).get("accounted_usd") or 0.0)
        if state_data.get("budget_drift_alert"):
            checks.append(f"WARNING: BUDGET DRIFT {state_data.get('budget_drift_pct', 0):.1f}% — tracked=${accounted:.2f} vs OpenRouter=${state_data.get('openrouter_total_usd', 0):.2f}")
        else:
            checks.append("OK: budget drift within tolerance")
    except Exception:
        checks.append("WARNING: COST ACCOUNTING UNAVAILABLE — budget drift check skipped")

    try:
        from supervisor.state import per_task_cost_summary
        costly = [t for t in per_task_cost_summary(5) if t["cost"] > 5.0]
        for t in costly:
            checks.append(f"WARNING: HIGH-COST TASK — task_id={t['task_id']} cost=${t['cost']:.2f} rounds={t['rounds']}")
        if not costly:
            checks.append("OK: no high-cost tasks (>$5)")
    except Exception:
        checks.append("WARNING: COST ACCOUNTING UNAVAILABLE — high-cost task check skipped")

    checks.extend(_memory_health_lines(env))

    # state/crash_report.json retired (CPL4-C9, owner 2A): its writer — the
    # crash-rollback path — no longer exists in this tree, so the reader and
    # its CRITICAL health line were dead surface. Stale files are inert.

    try:
        from ouroboros.extension_health import regressed_extensions

        drive_root = getattr(env, "drive_root", None) or env.drive_path("state").parent
        for rec in regressed_extensions(drive_root):
            good = rec.get("last_known_good") or {}
            observed = rec.get("last_observed") or {}
            checks.append(
                f"CRITICAL: EXTENSION REGRESSION — {rec.get('skill', '?')} was live at "
                f"{str(good.get('sha') or '?')[:12]} ({good.get('version') or '?'}), broken now at "
                f"{str(observed.get('sha') or '?')[:12]}: {str(observed.get('load_error') or '')[:200]}"
            )
    except Exception:
        pass

    plan_note = _plan_review_note(env, task_id)
    if plan_note:
        checks.append(plan_note)
    # The three delegated-run blocks below share ONE custody read of the rows they
    # can name (``health_runs``: own, held, maybe-unread), so a build decodes neither
    # every settled run nor each block's own copy (I18); their predicates and order
    # decide. A failed selection reads every row into the same state, because the
    # own-run lines have no read of their own; only if that fails too does each
    # remaining block replay for itself, under its own fail-soft arm.
    custody_root = None
    custody_state = None
    try:
        from ouroboros.delegate_custody import replay as replay_custody
        from ouroboros.delegate_custody_current import health_runs

        custody_root = getattr(env, "drive_root", None) or env.drive_path("state").parent
        try:
            custody_state = health_runs(custody_root, task_id)
        except Exception:
            log.debug("Selected custody read failed; reading every row", exc_info=True)
            custody_state = replay_custody(custody_root)
    except Exception:
        custody_state = None

    foreign: Dict[str, Dict[str, List[str]]] = {}  # exact source for one line covering all foreign owners

    def foreign_work(owner: str) -> Dict[str, List[str]]:
        return foreign.setdefault(owner, {"unread": [], "patches": [], "yours": []})

    try:
        from ouroboros.delegate_custody import settled_unread_outputs

        for run in settled_unread_outputs(custody_root, custody_state):
            # Owner doctrine D7, made load-bearing: a delegated result that was paid for
            # and never read to EOF is the launched-never-collected class. Not CRITICAL —
            # nothing is live and nothing is mutating — but it stays visible until the
            # read happens, which is the whole difference between a disclosure and a fact
            # someone acts on. It clears itself the moment the acknowledgement lands.
            owner = _foreign_owner(run, task_id)
            if owner:
                foreign_work(owner)["unread"].append(str(run.run_id or "?"))
            else:
                checks.append(_unread_output_line(custody_root, run, task_id))
    except Exception:
        pass

    try:
        checks.extend(_own_run_outcome_lines(custody_root, custody_state, task_id, attempt_key))
    except Exception:
        pass

    try:
        from ouroboros.delegate_custody import undisposed_patches
        from ouroboros.delegate_shared import orphan_apply_target_ok

        for run in undisposed_patches(custody_root, custody_state):
            # C1: a settled mutating run's work lives in its private snapshot (and its
            # captured patch, once one exists) until someone explicitly applies or
            # rejects it. The GC preserves the material, but preserved-and-invisible is
            # how an orphaned run's work sits on disk forever — so the pending
            # disposition stays a visible obligation until the PATCH_DISPOSED row
            # clears it. The wording follows the entry's REAL capture state (C1-R2):
            # a run reconciled over an absent daemon settled with NO capture (its
            # state was unknowable then), and claiming "captured" there would be a
            # receipt over work the child might still have been writing.
            if run.patch_captured:
                state_clause = (
                    f"settled with its changes captured from a private snapshot of "
                    f"{run.target_root or '?'}"
                )
                persist_clause = "The snapshot and patch persist until that disposition."
            else:
                state_clause = (
                    f"settled with its work preserved in its private snapshot of "
                    f"{run.target_root or '?'} (no patch captured yet — "
                    f"integrate_delegated_patch captures it at disposition and returns it for inspection "
                    f"before any apply)"
                )
                persist_clause = "The snapshot persists until that disposition."
            # Keep target-match facts in the retained list; the apply tool re-verifies every guard.
            owner = _foreign_owner(run, task_id)
            if owner:
                foreign_work(owner)["patches"].append(str(run.run_id or "?"))
                if active_root and orphan_apply_target_ok(run.target_root, active_root):
                    foreign_work(owner)["yours"].append(str(run.run_id))
                continue
            decide_clause = (
                f"decide with integrate_delegated_patch(run_id='{run.run_id}', "
                f"decision='apply'|'reject')."
            )
            checks.append(
                f"WARNING: DELEGATED PATCH AWAITS DISPOSITION — run {run.run_id or '?'} "
                f"(owner task {run.task_id or '?'}) {state_clause} and no apply/reject "
                f"recorded. Nothing reaches the shared tree by itself: {decide_clause} "
                f"{persist_clause}"
            )
    except Exception:
        pass
    if foreign:
        checks.append(_foreign_work_line(getattr(env, "budget_drive_root", None) or custody_root, foreign))

    try:
        from ouroboros.delegate_custody import open_containment_faults

        drive_root = getattr(env, "drive_root", None) or env.drive_path("state").parent
        for fault in open_containment_faults(drive_root):
            # An overpowered mutating run we asked to stop and could not verify stopped
            # is an incident, not a tool-result string. It stays CRITICAL until a
            # terminal receipt or a settlement clears it.
            checks.append(
                f"CRITICAL: DELEGATED RUN MAY STILL BE LIVE — run {fault.get('run_id') or '?'} "
                f"({fault.get('reason') or 'unverified'}), owner task "
                f"{fault.get('task_id') or '?'}, since {fault.get('ts') or '?'}"
            )
    except Exception:
        pass

    try:
        stray_note = _stray_server_note(env)
        if stray_note:
            checks.append(stray_note)
    except Exception:
        pass

    _collect_log_analysis_checks(env, checks)
    if not checks:
        return ""
    return "## Health Invariants\n\n" + "\n".join(f"- {check}" for check in checks)
