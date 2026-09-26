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
import pathlib
import re
from collections import Counter
from typing import Any, Dict, List, Optional

from ouroboros.context_budget import SCRATCHPAD_BLOAT_WARN_CHARS
from ouroboros.utils import iter_jsonl_objects, read_json_dict, read_text

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

    try:
        hit_rate = _compute_cache_hit_rate(env)
        if hit_rate is not None:
            if hit_rate < 0.30:
                checks.append(f"WARNING: LOW CACHE HIT RATE — {hit_rate:.0%} cached. Context structure may be degrading prompt caching efficiency.")
            elif hit_rate >= 0.50:
                checks.append(f"OK: cache hit rate ({hit_rate:.0%})")
            else:
                checks.append(f"INFO: cache hit rate moderate ({hit_rate:.0%})")
    except Exception:
        pass

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
    """Read this task's recorded wave; its timestamp is a snapshot, not a live lease."""
    if not task_id:
        return ""
    try:
        from ouroboros.task_results import current_plan_review_wave, load_plan_review_state

        root = getattr(env, "drive_root", None) or env.drive_path("state").parent
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
    except (OSError, ValueError, TimeoutError):
        log.warning("Unable to read plan-review health for %s", task_id, exc_info=True)
        return ""


def _memory_health_lines(env: Any) -> List[str]:
    """Own-memory maintenance signals: the two authored files and the dialogue pipeline.

    Health is where stale memory becomes visible, so a dialogue-consolidation run that
    failed, or a nomination batch that was accepted and then not published, is named
    here rather than left to a log nobody reads. Both lines are STATE read from
    ``memory/dialogue_meta.json``; neither carries a timestamp, because this block is
    rendered dynamically and the facts are latest-run facts, not events.
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

    from ouroboros.memory_nomination_receipts import DialogueMetaUnreadable, load_meta

    try:
        meta = load_meta(env.drive_path("memory/dialogue_meta.json"))
    except DialogueMetaUnreadable:
        # A broken existing meta file is not an empty nomination/cursor state.
        lines.append("WARNING: DIALOGUE META UNREADABLE — memory/dialogue_meta.json; "
                     "consolidation withheld to preserve existing bytes")
    else:
        pending = meta.get("pending_knowledge_nominations")
        if pending:
            # load_meta already validates the whole list; malformed state raises.
            sample = ", ".join(row["id"].split(":")[0][:12] + ":" +
                               ":".join(row["id"].split(":")[-2:])
                               for row in pending[:3])
            lines.append(
                f"WARNING: DIALOGUE KNOWLEDGE PUBLICATION OPEN — {len(pending)} source-addressed "
                f"nominations (first {min(3, len(pending))}: {sample}; omitted {max(0, len(pending)-3)}). "
                "Read memory/dialogue_meta.json and memory/knowledge_history.jsonl for full source. "
                "No automatic or tool-level discharge exists yet; later successes cannot retire older entries."
            )
        receipt = meta.get("last_unpublished_nominations")
        if isinstance(receipt, dict):
            failed = receipt.get("failed")
            if type(failed) is int and failed > 0:
                # This old batch receipt cannot be retired by a later new-source success.
                lines.append(
                    f"WARNING: LAST DIALOGUE KNOWLEDGE PUBLICATION INCOMPLETE — {failed} of "
                    f"{receipt.get('total')} nominations in a legacy consolidation batch remain unresolved "
                    f"(entry_id {receipt.get('entry_id')}); from the main chat, read_file(root='runtime_data', "
                    "path='memory/knowledge_history.jsonl') and publish what still holds"
                )
            elif failed is not None and failed != 0:
                lines.append("WARNING: DIALOGUE LEGACY NOMINATION RECEIPT INVALID — "
                             "memory/dialogue_meta.json; inspect the original receipt")
        error = meta.get("last_consolidation_error")
        if isinstance(error, dict):
            lines.append(
                f"WARNING: LAST DIALOGUE CONSOLIDATION FAILED — kind={error.get('kind') or 'unknown'} "
                f"at cursor {error.get('cursor_offset')}"
            )
        from ouroboros.consolidator import _era_retry_runs
        runs = _era_retry_runs(meta)
        for shown, (source_sha256, record) in enumerate(runs.items()):
            if shown == 3:
                lines.append(f"WARNING: DIALOGUE ERA COMPRESSION WITHHELD — {len(runs) - 3} more run(s) recorded in era_retry")
                break
            route = record.get("route")
            lines.append(
                f"WARNING: DIALOGUE ERA COMPRESSION WITHHELD — the era for source run "
                f"{source_sha256[:12]} on route "
                f"{route.get('model') if isinstance(route, dict) else route} was not shorter than its blocks; "
                "blocks retained, no paid repeat until that run or the route changes"
            )
    return lines


def build_health_invariants(env: Any, task_id: str = "", active_root: str = "") -> str:
    """Render the health-invariant WARNING block for one reader's context.

    ``task_id`` names the READING task so delegated-run obligations can shape
    their instruction clause by ownership: the obligations stay globally
    visible (owner doctrine — a preserved-and-invisible result is how work
    rots on disk), but only the OWNER task receives the call-shaped
    instruction. A non-owner told to call ``integrate_delegated_patch`` gets
    a structural ``run_not_owned`` refusal and an obligation it can never
    discharge. Empty ``task_id`` (legacy callers)
    keeps the call-shaped wording — an unattributed reader may be the owner.

    ``active_root`` names that reader's own active Git root, which this module
    cannot look up (it has no ``ctx``) and the caller already holds. One
    comparison through the apply gate's own predicate then names the concrete
    call for a FOREIGN obligation the reader's root already satisfies, instead
    of leaving sixteen identical abstract rows. Empty keeps the static wording
    byte-for-byte.
    """
    checks: List[str] = []

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
    # Both delegated-run obligations below read the SAME rotated custody chain,
    # so one traversal serves both instead of a full replay each (I18). A failed
    # read leaves the state None and each block replays for itself exactly as
    # before, under its own fail-soft arm.
    custody_root = None
    custody_state = None
    try:
        from ouroboros.delegate_custody import replay as replay_custody

        custody_root = getattr(env, "drive_root", None) or env.drive_path("state").parent
        custody_state = replay_custody(custody_root)
    except Exception:
        custody_state = None

    try:
        from ouroboros.delegate_custody import settled_unread_outputs

        for run in settled_unread_outputs(custody_root, custody_state):
            # Owner doctrine D7, made load-bearing: a delegated result that was paid for
            # and never read to EOF is the launched-never-collected class. Not CRITICAL —
            # nothing is live and nothing is mutating — but it stays visible until the
            # read happens, which is the whole difference between a disclosure and a fact
            # someone acts on. It clears itself the moment the acknowledgement lands.
            # The instruction clause is ownership-aware: the staged artifact lives
            # under the OWNER's task drive and the read acknowledgement only
            # credits the owner, so telling a foreign task to read it mints an
            # obligation it structurally cannot discharge.
            if task_id and run.task_id and task_id != run.task_id:
                action = (
                    f"Only its owner task {run.task_id} can read and acknowledge it; "
                    f"note the gap honestly instead of attempting the read here."
                )
            else:
                action = (
                    "Read it with read_file root='task_drive' until the artifact is "
                    "covered end to end, or say plainly that the result was not "
                    "collected."
                )
            checks.append(
                f"WARNING: DELEGATED RESULT NEVER READ — run {run.run_id or '?'} settled "
                f"with its full output staged at {run.output_artifact} and never read to "
                f"EOF (owner task {run.task_id or '?'}). {action}"
            )
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
                    f"integrate_delegated_patch captures it at disposition)"
                )
                persist_clause = "The snapshot persists until that disposition."
            # Ownership-aware instruction: a foreign reader is told WHO must act
            # instead of being handed a call that structurally refuses. The
            # wording is STATIC — stating the rule costs no per-orphan
            # task_result read, and the tool itself is the authority on whether
            # this particular caller may dispose this particular run.
            if task_id and run.task_id and task_id != run.task_id:
                decide_clause = (
                    f"Only its owner task {run.task_id} can decide it while that task "
                    f"is live; once that task is terminal, a live top-level task may dispose it: "
                    f"apply requires the caller's active Git root or fresh payload binding to equal "
                    f"the run's recorded target, while reject requires only terminality and reject "
                    f"may release it even from a different active root; the disposition row records "
                    f"who acted (a live-owner foreign call is still refused as run_not_owned)."
                )
                # One comparison over the root the caller already holds, through the
                # SAME predicate the apply gate uses: when this reader's own active
                # root already satisfies the recorded target, the abstract rule alone
                # left it guessing (sixteen such rows in one prompt). No per-orphan
                # task_result read is added, and the tool re-verifies every guard.
                if active_root and orphan_apply_target_ok(run.target_root, active_root):
                    decide_clause += (
                        f" Your own active root already satisfies that target, so once the owner "
                        f"is terminal the call is integrate_delegated_patch(run_id='{run.run_id}', "
                        f"decision='apply'|'reject')."
                    )
            else:
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

    try:
        # Probe body lives beside the other live-reused startup checks
        # (the check_stray_server_processes pattern); thresholds are the
        # justified constants in ouroboros/context_budget.py.
        from ouroboros.agent_startup_checks import hot_store_growth_notes

        checks.extend(hot_store_growth_notes(env))
    except Exception:
        pass

    _collect_log_analysis_checks(env, checks)
    if not checks:
        return ""
    return "## Health Invariants\n\n" + "\n".join(f"- {check}" for check in checks)


def _compute_cache_hit_rate(env: Any) -> Optional[float]:
    total_prompt = total_cached = reported = 0
    try:
        for ev in _iter_recent_jsonl(env.drive_path("logs/events.jsonl")):
            if ev.get("type") != "llm_round":
                continue
            usage = ev.get("usage", ev)
            pt = int(usage.get("prompt_tokens", 0))
            # An absent key and an explicit null both mean the round measured
            # nothing, so it joins NEITHER side of the ratio; only a number is
            # a report, and an explicit 0 is a real measured miss.
            cached = usage.get("cached_tokens")
            if pt > 0 and cached is not None:
                total_prompt += pt
                total_cached += int(cached or 0)
                reported += 1
    except Exception:
        return None
    # Nobody reporting a cache is not a cache that missed: without the key the
    # share is UNKNOWN, and the arithmetic zero below would render that absence
    # as an honest 0% and send the owner hunting a caching regression no round
    # ever measured. The window is the REPORTING rounds for the same reason: on
    # an install whose providers are mixed, charging a silent round's prompt to
    # the denominator understates the share the measuring rounds actually saw.
    if reported < 5 or total_prompt == 0:
        return None
    return total_cached / total_prompt
