"""Worker-boot checks for dirty repo, version sync, budget, and memory files."""

from __future__ import annotations

import logging
import os
import re
import subprocess
import time
from typing import Any, Dict, Tuple

from ouroboros.utils import atomic_write_json, utc_now_iso, read_text, append_jsonl, read_json_dict, update_json_locked

log = logging.getLogger(__name__)


def _is_release_tag(tag: str) -> bool:
    from ouroboros.tools.release_sync import normalize_release_tag

    return bool(normalize_release_tag(tag))


def check_uncommitted_changes(env: Any) -> Tuple[dict, int]:
    """Warn on dirty worker boot; rescue/reset is supervisor-owned, never worker-owned."""
    try:
        lock_path = env.repo_path(".git/index.lock")
        if lock_path.exists():
            try:
                # Age gate (mirrors supervisor.git_ops._stale_git_lock_paths):
                # an unconditional unlink could yank index.lock from under a
                # LIVE supervisor git operation during worker boot, corrupting
                # the index. Only locks orphaned by dead processes are stale.
                age_sec = time.time() - lock_path.stat().st_mtime
                if age_sec >= 15.0:
                    lock_path.unlink()
                    log.warning("Removed stale git index.lock (age %.0fs)", age_sec)
            except OSError:
                pass

        result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=str(env.repo_dir),
            capture_output=True, text=True, timeout=10, check=True
        )
        dirty_files = [l.strip() for l in result.stdout.strip().split('\n') if l.strip()]
        if dirty_files:
            log.warning(
                "Uncommitted changes detected on worker boot; diagnostic-only, "
                "rescue is owned by supervisor-side safe_restart(rescue_and_reset)"
            )
            return {
                "status": "warning",
                "files": dirty_files[:20],
                "auto_committed": False,
                "auto_rescue_skipped": "supervisor_side_rescue_owns_this",
            }, 1
        return {"status": "ok"}, 0
    except Exception as e:
        return {"status": "error", "error": str(e)}, 0


def check_version_sync(env: Any) -> Tuple[dict, int]:
    """Check VERSION file sync with git tags and pyproject.toml."""
    try:
        from ouroboros.tools.release_sync import (
            _normalize_pep440,
            _shields_escape,
            extract_architecture_header_version,
            extract_readme_badge_version,
            is_release_version,
        )
        version_file = read_text(env.repo_path("VERSION")).strip()
        issue_count = 0
        result_data: Dict[str, Any] = {"version_file": version_file}

        pyproject_path = env.repo_path("pyproject.toml")
        pyproject_content = read_text(pyproject_path)
        match = re.search(r'^version\s*=\s*["\']([^"\']+)["\']', pyproject_content, re.MULTILINE)
        if match:
            pyproject_version = match.group(1)
            result_data["pyproject_version"] = pyproject_version
            expected_pyproject = _normalize_pep440(version_file) if is_release_version(version_file) else version_file
            if expected_pyproject != pyproject_version:
                result_data["status"] = "warning"
                issue_count += 1

        try:
            readme_content = read_text(env.repo_path("README.md"))
            badge_version = extract_readme_badge_version(readme_content)
            readme_version = badge_version
            if not readme_version:
                readme_match = re.search(r'\*\*Version:\*\*\s*([^\s]+)', readme_content)
                readme_version = str(readme_match.group(1) or "").strip() if readme_match else ""
            if readme_version:
                result_data["readme_version"] = readme_version
                badge_token_ok = True
                if badge_version and is_release_version(version_file):
                    badge_token_ok = f"version-{_shields_escape(version_file)}-green" in readme_content
                result_data["readme_badge_url_valid"] = badge_token_ok
                if version_file != readme_version or not badge_token_ok:
                    result_data["status"] = "warning"
                    issue_count += 1
        except Exception:
            log.debug("Failed to check README.md version", exc_info=True)

        try:
            arch_content = read_text(env.repo_path("docs/ARCHITECTURE.md"))
            arch_version = extract_architecture_header_version(arch_content)
            if arch_version:
                result_data["architecture_version"] = arch_version
                if version_file != arch_version:
                    result_data["status"] = "warning"
                    issue_count += 1
        except Exception:
            log.debug("Failed to check ARCHITECTURE.md version", exc_info=True)

        result = subprocess.run(
            ["git", "describe", "--tags", "--abbrev=0"],
            cwd=str(env.repo_dir),
            capture_output=True, text=True, timeout=10
        )
        if result.returncode != 0:
            result_data["status"] = "warning"
            result_data["message"] = "no_tags"
            return result_data, issue_count
        else:
            latest_tag = result.stdout.strip().lstrip('v')
            result_data["latest_tag"] = latest_tag
            if _is_release_tag(latest_tag) and version_file != latest_tag:
                result_data["status"] = "warning"
                issue_count += 1
            elif not _is_release_tag(latest_tag):
                result_data["tag_sync"] = "ignored_non_release_tag"

        if issue_count == 0:
            result_data["status"] = "ok"

        return result_data, issue_count
    except Exception as e:
        return {"status": "error", "error": str(e)}, 0


def check_budget(env: Any) -> Tuple[dict, int]:
    """Check budget remaining with warning thresholds."""
    try:
        state_path = env.drive_path("state") / "state.json"
        state_data = read_json_dict(state_path)
        if state_data is None:
            return {
                "status": "error",
                "error": "state.json missing or invalid",
                "path": str(state_path),
            }, 1
        total_budget_str = os.environ.get("TOTAL_BUDGET", "")

        if not total_budget_str or float(total_budget_str) == 0:
            return {"status": "unconfigured"}, 0
        else:
            total_budget = float(total_budget_str)
            spent = float(state_data.get("spent_usd", 0))
            remaining = max(0, total_budget - spent)

            if remaining < 0.5:
                status = "emergency"
                issues = 1
            elif remaining < 2:
                status = "critical"
                issues = 1
            elif remaining < 5:
                status = "warning"
                issues = 0
            else:
                status = "ok"
                issues = 0

            return {
                "status": status,
                "remaining_usd": round(remaining, 2),
                "total_usd": total_budget,
                "spent_usd": round(spent, 2),
            }, issues
    except Exception as e:
        return {"status": "error", "error": str(e)}, 0


def check_review_continuations(env: Any) -> Tuple[dict, int]:
    try:
        from ouroboros.task_continuation import list_review_continuations
        from ouroboros.task_results import (
            STATUS_CANCELLED,
            STATUS_COMPLETED,
            STATUS_FAILED,
            STATUS_INTERRUPTED,
            STATUS_REJECTED_DUPLICATE,
            STATUS_REQUESTED,
            STATUS_RUNNING,
            STATUS_SCHEDULED,
            list_task_results,
        )

        continuations, corrupt = list_review_continuations(env.drive_root)
        task_rows = list_task_results(
            env.drive_root,
            statuses=[
                STATUS_REQUESTED,
                STATUS_SCHEDULED,
                STATUS_RUNNING,
                STATUS_INTERRUPTED,
                STATUS_COMPLETED,
                STATUS_FAILED,
                STATUS_CANCELLED,
                STATUS_REJECTED_DUPLICATE,
            ],
        )
        task_by_id = {
            str(item.get("task_id") or ""): item
            for item in task_rows
            if str(item.get("task_id") or "").strip()
        }

        rows = []
        interrupted = []
        for item in continuations:
            task_status = str((task_by_id.get(item.task_id) or {}).get("status") or "")
            row = {
                "task_id": item.task_id,
                "task_status": task_status or "missing",
                "source": item.source,
                "stage": item.stage,
                "repo_key": item.repo_key,
                "tool_name": item.tool_name,
                "attempt": item.attempt,
                "block_reason": item.block_reason,
                "obligation_ids": list(item.obligation_ids or []),
                "critical_findings": len(item.critical_findings or []),
                "advisory_findings": len(item.advisory_findings or []),
                "updated_ts": item.updated_ts,
            }
            rows.append(row)
            if task_status == STATUS_INTERRUPTED:
                interrupted.append(row)

        status = "ok"
        issues = 0
        if rows or corrupt:
            status = "warning"
        if rows:
            issues += 1
        if corrupt:
            status = "error"
            issues += 1

        return {
            "status": status,
            "open_review_continuations": rows[:20],
            "interrupted_tasks": interrupted[:20],
            "corrupt": corrupt[:20],
        }, issues
    except Exception as e:
        return {"status": "error", "error": str(e)}, 1


def check_extension_health(env: Any) -> Tuple[Dict[str, Any], int]:
    """Surface extensions that were live at a prior version but are broken now (P1/P3)."""
    try:
        import pathlib
        from ouroboros.extension_health import regressed_extensions

        drive_root = pathlib.Path(getattr(env, "drive_root", None) or env.drive_path("state").parent)
        regressed = regressed_extensions(drive_root)
    except Exception:
        return {"status": "skipped"}, 0
    if regressed:
        names = [str(r.get("skill") or "?") for r in regressed]
        log.warning("Extension regression(s) detected since last healthy version: %s", names)
        return {"status": "regressed", "skills": names}, 1
    return {"status": "ok"}, 0


def verify_system_state(env: Any, git_sha: str) -> None:
    """Bible Principle 1: verify system state on every startup."""
    checks: Dict[str, Any] = {}
    issues = 0
    drive_logs = env.drive_path("logs")

    checks["uncommitted_changes"], issue_count = check_uncommitted_changes(env)
    issues += issue_count

    checks["version_sync"], issue_count = check_version_sync(env)
    issues += issue_count

    checks["budget"], issue_count = check_budget(env)
    issues += issue_count

    memory_dir = env.drive_path("memory")
    identity_path = memory_dir / "identity.md"
    scratchpad_path = memory_dir / "scratchpad.md"
    world_path = memory_dir / "WORLD.md"

    identity_ok = identity_path.exists() and identity_path.stat().st_size > 0
    scratchpad_ok = scratchpad_path.exists()
    world_ok = world_path.exists()

    checks["identity"] = {"exists": identity_path.exists(), "non_empty": identity_ok}
    checks["scratchpad"] = {"exists": scratchpad_ok}
    checks["world_profile"] = {"exists": world_ok}

    if not identity_ok:
        issues += 1
        log.warning("identity.md missing or empty — continuity at risk (Bible P1)")
    if not scratchpad_ok:
        issues += 1
        log.warning("scratchpad.md missing — working memory not available (Bible P1)")
    if not world_ok:
        issues += 1
        log.warning("WORLD.md missing — environment profile not available")

    configured_model = os.environ.get("OUROBOROS_MODEL", "")
    checks["model"] = {"configured": configured_model or "(not set)"}
    if not configured_model:
        issues += 1

    checks["extension_health"], issue_count = check_extension_health(env)
    issues += issue_count

    # Reconcile stale hung reviewed attempts left by abrupt process death
    try:
        import pathlib
        from ouroboros.review_state import _utc_now, update_state
        drive_root = pathlib.Path(env.drive_root) if hasattr(env, "drive_root") else env.drive_path("").parent
        expired = update_state(
            drive_root,
            lambda st: st.expire_stale_attempts(now_ts=_utc_now()),
        )
        if expired:
            log.warning("Auto-expired %d stale reviewed attempt(s) on startup", len(expired))
    except Exception:
        log.debug("Failed to reconcile commit attempt state", exc_info=True)

    checks["review_continuations"], issue_count = check_review_continuations(env)
    issues += issue_count

    event = {
        "ts": utc_now_iso(),
        "type": "startup_verification",
        "checks": checks,
        "issues_count": issues,
        "git_sha": git_sha,
    }
    append_jsonl(drive_logs / "events.jsonl", event)

    if issues > 0:
        log.warning(f"Startup verification found {issues} issue(s): {checks}")


def inject_crash_report(env: Any) -> None:
    """If a crash report exists from a rollback, log it to events.

    The file is NOT deleted — it stays so that build_health_invariants()
    shows CRITICAL: RECENT CRASH ROLLBACK on every task until the issue
    is investigated and removed via run_command (LLM-first, P5).
    """
    try:
        crash_path = env.drive_path("state") / "crash_report.json"
        if not crash_path.exists():
            return
        crash_data = read_json_dict(crash_path)
        if not crash_data:
            append_jsonl(env.drive_path("logs") / "events.jsonl", {
                "ts": utc_now_iso(),
                "type": "crash_report_invalid",
                "path": str(crash_path),
            })
            log.warning("Crash report exists but is not valid JSON object: %s", crash_path)
            return
        append_jsonl(env.drive_path("logs") / "events.jsonl", {
            "ts": utc_now_iso(),
            "type": "crash_rollback_detected",
            "crash_data": crash_data,
        })
        log.warning("Crash rollback detected: %s", crash_data)
    except Exception:
        log.debug("Failed to process crash report", exc_info=True)


def _record_pending_owner_report(campaign: Dict[str, Any], tx: Dict[str, Any]) -> None:
    """Stage the WS-13.5 owner absorb/abandon report ON THE CAMPAIGN for the
    server to deliver.

    verify_restart runs in the WORKER process, where the message bus is not
    init()'d (init happens only in server.py), so send_with_budget cannot reach
    the owner from here. Instead we persist a ``pending_owner_report`` into the
    campaign; the supervisor (server process, bus live) drains and delivers it
    via enqueue_evolution_task_if_needed. Caller must persist the campaign.
    """
    outcome = str(tx.get("cycle_outcome") or "")
    if outcome not in ("absorbed", "abandoned"):
        return
    # tx-shaped so the server can hand it straight to notify_owner_cycle_outcome.
    campaign["pending_owner_report"] = {
        "cycle_outcome": outcome,
        "commit_sha": str(tx.get("commit_sha") or "").strip(),
        "abandoned_reason": str(tx.get("abandoned_reason") or ""),
    }


def _append_cycle_outcome_tag(env: Any, *, campaign: Any, transaction: Any, source: str, backlog_id: str) -> None:
    """Solve-capability ledger tag (Block 5C): the task-done checkpoint recorded
    waiting_for_restart; this writes the post-restart resolution. Never raises."""
    try:
        from ouroboros.evolution_checkpoints import append_cycle_outcome_checkpoint
        append_cycle_outcome_checkpoint(
            env.drive_root,
            campaign=campaign,
            transaction=transaction,
            source=source,
            backlog_id=backlog_id,
        )
    except Exception:
        log.debug("Failed to append %s cycle-outcome checkpoint", source, exc_info=True)


def verify_restart(env: Any, git_sha: str) -> None:
    """Best-effort restart verification."""
    def _append_unique_transaction(campaign: Dict[str, Any], tx: Dict[str, Any]) -> None:
        tx_history = list(campaign.get("transaction_history") or [])
        tx_id = str(tx.get("transaction_id") or "")
        if tx_id and any(
            isinstance(item, dict) and str(item.get("transaction_id") or "") == tx_id
            for item in tx_history
        ):
            campaign["transaction_history"] = tx_history[-50:]
            return
        tx_history.append(dict(tx))
        campaign["transaction_history"] = tx_history[-50:]

    def _close_post_task_backlog(campaign: Dict[str, Any]) -> None:
        backlog_id = str(campaign.get("post_task_backlog_id") or "").strip()
        if not backlog_id:
            return
        try:
            from ouroboros.improvement_backlog import close_backlog_items

            drive_root = getattr(env, "drive_root", None) or env.drive_path("memory").parent
            close_backlog_items(drive_root, ids=[backlog_id])
        except Exception:
            log.debug("Post-task backlog close-on-absorb failed", exc_info=True)
        campaign.pop("post_task_backlog_id", None)

    def _commit_reachable(commit_sha: str, observed_sha: str) -> bool:
        if commit_sha and observed_sha and commit_sha == observed_sha:
            return True
        try:
            repo_dir = getattr(env, "repo_dir", None) or env.repo_path(".").parent
            result = subprocess.run(
                ["git", "merge-base", "--is-ancestor", commit_sha, observed_sha or "HEAD"],
                cwd=repo_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=10,
            )
            return result.returncode == 0
        except Exception:
            return False

    def _reconcile_dangling_campaign_transaction(observed_sha: str) -> None:
        try:
            campaign_path = env.drive_path("state") / "evolution_campaign.json"
            # Reconcile AT MOST ONCE per server generation. A genuine restart
            # begins a new custody generation (NW-10 session id, which workers
            # inherit from the server); a routine worker RESPAWN keeps the same
            # generation. Without this gate, a respawn mid-cycle — after the
            # reviewed commit lands on HEAD but before the real restart — would
            # falsely mark the cycle absorbed/restart-verified (verified_by=
            # boot_reconciliation) while the evolution task is still running and
            # nothing was actually restarted. Honors the owner's reconcile_yes
            # choice: it still reconciles on the next genuine new-generation boot.
            # The gen value depends only on the custody session, not the campaign.
            try:
                from ouroboros.process_custody import current_custody_session_id
                gen = str(current_custody_session_id() or "")
            except Exception:
                gen = ""
            # The whole read-check-write runs under update_json_locked so the
            # gen-gate is ATOMIC: at boot ~10 workers whose os.rename of the pending
            # file lost all call this; the lock + in-lock re-read make exactly one
            # of them reconcile per generation (the rest see the claimed gen and
            # abort), instead of an unlocked stampede that double-increments
            # absorbed_cycles_done / lost-updates each other. (The os.rename WINNER
            # runs _mark_campaign_restart_verified, which stays unlocked; the narrow
            # winner-vs-loser ordering edge on an ancestor-not-HEAD commit is
            # idempotency-mitigated — see _mark_campaign_restart_verified.)
            event: Dict[str, Any] = {}  # captured in-lock for post-lock event logging

            outcome_snapshot: Dict[str, Any] = {}

            def _mutate(campaign: Dict[str, Any]):
                if not isinstance(campaign, dict):
                    return None
                if gen and str(campaign.get("last_boot_reconcile_gen") or "") == gen:
                    return None  # already reconciled this generation — abort, no write
                tx = campaign.get("active_transaction") if isinstance(campaign.get("active_transaction"), dict) else {}
                commit_sha = str(tx.get("commit_sha") or "").strip()
                # Capture before the absorbed branch pops it via _close_post_task_backlog.
                outcome_snapshot["backlog_id"] = str(campaign.get("post_task_backlog_id") or "")
                if not commit_sha or bool(tx.get("restart_verified")):
                    # Nothing to reconcile this generation — record it so a later
                    # respawn (same generation) does not re-enter and absorb a commit
                    # that lands after this point.
                    if gen and str(campaign.get("last_boot_reconcile_gen") or "") != gen:
                        campaign["last_boot_reconcile_gen"] = gen
                        campaign["updated_at"] = utc_now_iso()
                        return campaign
                    return None
                now = utc_now_iso()
                campaign["last_boot_reconcile_gen"] = gen
                tx["restart_verified_at"] = now
                tx["restart_observed_sha"] = observed_sha
                tx["updated_at"] = now
                if _commit_reachable(commit_sha, observed_sha):
                    tx["restart_required"] = False
                    tx["restart_verified"] = True
                    tx["verified_by"] = "boot_reconciliation"
                    tx["cycle_outcome"] = "absorbed"
                    if not tx.get("absorbed_counted"):
                        campaign["absorbed_cycles_done"] = int(campaign.get("absorbed_cycles_done") or 0) + 1
                        tx["absorbed_counted"] = True
                    _append_unique_transaction(campaign, tx)
                    campaign.pop("active_transaction", None)
                    _close_post_task_backlog(campaign)
                    from supervisor.evolution_lifecycle import _clear_objective_repeat_count
                    _clear_objective_repeat_count(campaign, tx)  # BUG3: absorb clears this fp
                    campaign["progress_notes"] = (
                        f"Restart reconciled for reviewed commit {commit_sha[:12]}; "
                        "self-evolution cycle absorbed at boot."
                    )
                    event_type, ok = "evolution_tx_reconciled", True
                else:
                    tx["restart_required"] = False
                    tx["restart_verified"] = False
                    tx["cycle_outcome"] = "abandoned"
                    tx["abandoned_at"] = now
                    tx["abandoned_reason"] = "commit_not_reachable_at_boot"
                    _append_unique_transaction(campaign, tx)
                    campaign.pop("active_transaction", None)
                    campaign.pop("post_task_backlog_id", None)
                    from supervisor.evolution_lifecycle import _bump_objective_repeat_count
                    _bump_objective_repeat_count(campaign, tx)  # BUG3: commit-but-never-absorbs counts
                    campaign["progress_notes"] = (
                        f"Restart reconciliation abandoned commit {commit_sha[:12]} "
                        f"because observed HEAD {observed_sha[:12]} does not contain it."
                    )
                    event_type, ok = "evolution_tx_abandoned", False
                # WS-13.5 (e5): stage the owner absorb/abandon report (server delivers).
                _record_pending_owner_report(campaign, tx)
                campaign["updated_at"] = now
                event.update({
                    "ts": now, "type": event_type, "ok": ok,
                    "commit_sha": commit_sha, "observed_sha": observed_sha,
                })
                outcome_snapshot.update({
                    "campaign": {"id": campaign.get("id"), "objective": campaign.get("objective")},
                    "transaction": dict(tx),
                })
                return campaign

            update_json_locked(campaign_path, _mutate)
            if event:
                append_jsonl(env.drive_path("logs") / "events.jsonl", event)
            if outcome_snapshot.get("transaction"):
                _append_cycle_outcome_tag(
                    env,
                    campaign=outcome_snapshot.get("campaign"),
                    transaction=outcome_snapshot.get("transaction"),
                    source="boot_reconcile",
                    backlog_id=str(outcome_snapshot.get("backlog_id") or ""),
                )
        except Exception:
            log.debug("Failed to reconcile dangling evolution transaction", exc_info=True)

    def _mark_campaign_restart_verified(expected_sha: str, observed_sha: str, ok: bool) -> bool:
        # Runs only on the single worker that won the os.rename of the pending claim,
        # so there is no marker-vs-marker race. It stays unlocked (its strict
        # expected==observed check and the active_transaction-popped guard make the
        # outcome consistent with a concurrent boot reconcile in the common case where
        # the reviewed commit IS HEAD). KNOWN LIMITATION: if the reviewed commit is an
        # ANCESTOR of HEAD but not HEAD itself, this marker blocks (mismatch) while a
        # boot reconciler would absorb (commit reachable) — a narrow, idempotency-
        # mitigated ordering edge. Fully resolving it needs both writers to share the
        # campaign lock; deferred as a focused follow-up (advisory, restart-critical).
        try:
            campaign_path = env.drive_path("state") / "evolution_campaign.json"
            campaign = read_json_dict(campaign_path) or {}
            if not isinstance(campaign, dict):
                return bool(ok)
            tx = campaign.get("active_transaction") if isinstance(campaign.get("active_transaction"), dict) else {}
            if not tx:
                return bool(ok)
            # Captured before the absorbed branch pops it via _close_post_task_backlog.
            backlog_id_before_close = str(campaign.get("post_task_backlog_id") or "")
            commit_sha = str(tx.get("commit_sha") or "").strip()
            if commit_sha and commit_sha != expected_sha:
                tx["restart_required"] = True
                tx["restart_verified"] = False
                tx["restart_verified_at"] = utc_now_iso()
                tx["restart_expected_sha"] = expected_sha
                tx["restart_observed_sha"] = observed_sha
                tx["restart_mismatch"] = {
                    "active_commit_sha": commit_sha,
                    "pending_expected_sha": expected_sha,
                    "observed_sha": observed_sha,
                }
                tx["updated_at"] = utc_now_iso()
                campaign["active_transaction"] = tx
                campaign["progress_notes"] = (
                    f"Restart verification claim mismatch: active transaction expects {commit_sha[:12]}, "
                    f"pending claim expected {expected_sha[:12]} and observed {observed_sha[:12]}. "
                    "Next campaign cycle is blocked."
                )
                campaign["updated_at"] = utc_now_iso()
                atomic_write_json(campaign_path, campaign, trailing_newline=True)
                return False
            tx["restart_required"] = bool(not ok)
            tx["restart_verified"] = bool(ok)
            tx["restart_verified_at"] = utc_now_iso()
            tx["restart_expected_sha"] = expected_sha
            tx["restart_observed_sha"] = observed_sha
            tx["updated_at"] = utc_now_iso()
            if ok and commit_sha:
                if not tx.get("absorbed_counted"):
                    campaign["absorbed_cycles_done"] = int(campaign.get("absorbed_cycles_done") or 0) + 1
                    tx["absorbed_counted"] = True
                # Set the outcome BEFORE appending: _append_unique_transaction stores
                # a COPY (dict(tx)), so the durable history entry only carries the
                # absorbed outcome if it is set at append time, not afterwards.
                tx["cycle_outcome"] = "absorbed"
                _append_unique_transaction(campaign, tx)
                campaign.pop("active_transaction", None)
                # Close-on-commit (Phase 2 C): only NOW — when the reviewed self-mod
                # commit is restart-verified and absorbed — mark the promoted backlog
                # item done. Doing this earlier (at commit_sha time) could close an
                # item whose commit later fails restart verification.
                _close_post_task_backlog(campaign)
                from supervisor.evolution_lifecycle import _clear_objective_repeat_count
                _clear_objective_repeat_count(campaign, tx)  # BUG3: absorb clears this fp
                campaign["progress_notes"] = (
                    f"Restart verified for reviewed commit {observed_sha[:12]}; "
                    "self-evolution cycle absorbed."
                )
            elif ok and not commit_sha:
                tx["restart_no_commit"] = True
                _append_unique_transaction(campaign, tx)
                campaign.pop("active_transaction", None)
                # This cycle absorbed no reviewed commit, so the promoted item was
                # NOT addressed: clear the stale link WITHOUT closing it, so a later
                # unrelated absorbed commit cannot close the wrong backlog item.
                campaign.pop("post_task_backlog_id", None)
                from supervisor.evolution_lifecycle import _bump_objective_repeat_count
                _bump_objective_repeat_count(campaign, tx)  # BUG3: verified-but-no-absorb counts
                campaign["progress_notes"] = (
                    f"Restart verified for {observed_sha[:12]}; no reviewed self-mod "
                    "commit was present, so no evolution cycle was absorbed."
                )
            else:
                campaign["active_transaction"] = tx
                campaign["progress_notes"] = (
                    f"Restart verification failed for expected {expected_sha[:12]} "
                    f"(observed {observed_sha[:12]}). Next campaign cycle is blocked."
                )
            # WS-13.5 (e5): the absorb transition for the NORMAL auto-restart flow
            # happens HERE (task-done only ever sees "waiting_for_restart"), so the
            # owner absorb-report must be staged here. Persisted in the SAME write
            # below; the server delivers it (the worker has no live message bus).
            if tx.get("cycle_outcome") == "absorbed":
                _record_pending_owner_report(campaign, tx)
            campaign["updated_at"] = utc_now_iso()
            atomic_write_json(campaign_path, campaign, trailing_newline=True)
            if tx.get("cycle_outcome") == "absorbed":
                _append_cycle_outcome_tag(
                    env,
                    campaign={"id": campaign.get("id"), "objective": campaign.get("objective")},
                    transaction=tx,
                    source="restart_verified",
                    backlog_id=backlog_id_before_close,
                )
            return bool(ok)
        except Exception:
            log.debug("Failed to update evolution campaign restart verification", exc_info=True)
            return bool(ok)

    try:
        pending_path = env.drive_path('state') / 'pending_restart_verify.json'
        claim_path = pending_path.with_name(f"pending_restart_verify.claimed.{os.getpid()}.json")
        try:
            os.rename(str(pending_path), str(claim_path))
        except (FileNotFoundError, Exception):
            _reconcile_dangling_campaign_transaction(git_sha)
            return
        try:
            claim_data = read_json_dict(claim_path)
            if claim_data is None:
                append_jsonl(env.drive_path('logs') / 'events.jsonl', {
                    'ts': utc_now_iso(), 'type': 'restart_verify',
                    'pid': os.getpid(), 'ok': False,
                    'error': 'pending_restart_verify_invalid',
                    'observed_sha': git_sha,
                })
                return
            expected_sha = str(claim_data.get("expected_sha", "")).strip()
            sha_ok = bool(expected_sha and expected_sha == git_sha)
            campaign_ok = _mark_campaign_restart_verified(expected_sha, git_sha, sha_ok)
            ok = bool(sha_ok and campaign_ok)
            append_jsonl(env.drive_path('logs') / 'events.jsonl', {
                'ts': utc_now_iso(), 'type': 'restart_verify',
                'pid': os.getpid(), 'ok': ok,
                'expected_sha': expected_sha, 'observed_sha': git_sha,
            })
        except Exception:
            log.debug("Failed to log restart verify event", exc_info=True)
            pass
        try:
            claim_path.unlink()
        except Exception:
            log.debug("Failed to delete restart verify claim file", exc_info=True)
            pass
    except Exception:
        log.debug("Restart verification failed", exc_info=True)
        pass
