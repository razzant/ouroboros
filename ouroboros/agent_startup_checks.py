"""Worker-boot checks for dirty repo, version sync, budget, and memory files."""

from __future__ import annotations

import copy
import json
import logging
import os
import pathlib
import re
import subprocess
from typing import Any, Dict, Tuple

from ouroboros.evolution_checkpoints import (
    append_cycle_outcome_tag,
    backfill_missing_cycle_outcomes,
)
from ouroboros.task_finalization import TERMINAL_ORIGIN_HOST_SALVAGE
from ouroboros.tool_capabilities import DEFAULT_TOOL_RESULT_LIMIT
from ouroboros.utils import (
    append_jsonl,
    atomic_write_json,
    read_json_dict,
    read_text,
    update_json_locked,
    utc_now_iso,
)
from ouroboros.config import runtime_setting
from ouroboros.store import StoreError, StoreMigrating

log = logging.getLogger(__name__)


def _candidate_restart_guidance(tx, commit_sha, *, held=True):
    prefix = "Reviewed candidate awaits deliberate adoption." if held else "Candidate availability is unconfirmed; inspect the retained transaction."
    return (f"{prefix} Nothing was adopted automatically. When adoption is authorized, "
            f"select the ended owner's candidate with prepare_self_change(resume={str(tx.get('task_id') or '')!r}), "
            f"then request_restart(adopt_commit={commit_sha!r}); an active owner resumes its own task.")


_TASK_RESULT_PROCESS_EVIDENCE_FIELDS = frozenset({
    # These are raw reasoning/transport records, not terminal task authority.
    # Exact immutable refs and compact terminal facts remain top-level and are
    # therefore inherited automatically; the transcripts themselves stay in
    # observability/checkpoint storage for explicit reads.
    "loop_outcome",
    "metadata",
    "llm_trace",
    "reasoning_notes",
    "tool_calls",
    "candidate_answers",
    "messages",
    "transcript",
    "model_transcript",
    "tool_transcript",
    "review_runs",
    "review_evidence",
    "review_projection",
    "trace_refs",
    "root_phase_checkpoint",
    "raw_request",
    "raw_response",
    "request_wire",
    "request_wire_history",
})

# Automatic continuation gets at most one ordinary tool-result-sized slice of
# unreviewed host salvage. The full canonical result remains available through
# the named get_task_result(include_authority=True) source below.
_AUTOMATIC_HOST_SALVAGE_RESULT_CHARS = DEFAULT_TOOL_RESULT_LIMIT


def _authority_verification_receipts(
    row: Dict[str, Any], drive_root: Any,
) -> list[Dict[str, Any]]:
    """Read the canonical receipt store, including the pre-copy-back child window."""

    if drive_root is None:
        return []
    task_id = str(row.get("task_id") or row.get("id") or "").strip()
    if not task_id:
        return []
    from ouroboros.outcomes import read_verification_receipts_from_roots
    from ouroboros.task_status import _child_drive_candidates

    roots = [*_child_drive_candidates(row), drive_root]
    return read_verification_receipts_from_roots(roots, task_id)


def task_result_authority_projection(
    row: Dict[str, Any], *, drive_root: Any = None,
) -> Dict[str, Any]:
    """Exact terminal authority, excluding raw model/tool/loop process evidence.

    Task-result writers are intentionally additive.  Copying every top-level
    terminal field except the explicit process-evidence carriers means a future
    artifact, custody, verification, or capability fact cannot silently vanish
    merely because this reader predates its field name.
    """

    authority = {
        key: copy.deepcopy(value)
        for key, value in row.items()
        if key not in _TASK_RESULT_PROCESS_EVIDENCE_FIELDS
    }
    from ouroboros.task_finalization import terminal_host_notice_text

    notice = terminal_host_notice_text(row)
    if notice:
        authority["terminal_host_notice"] = notice
    contract = row.get("task_contract")
    if isinstance(contract, dict):
        authority["task_contract"] = copy.deepcopy(contract)
    else:
        from ouroboros.contracts.task_contract import build_task_contract

        if contract not in (None, "") and "task_contract_malformed" not in authority:
            # The malformed original is still a terminal fact: it rides (the
            # envelope bounds it), never silently replaced by the rebuild -
            # and never clobbering a row's own field of this name.
            authority["task_contract_malformed"] = copy.deepcopy(contract)
        authority["task_contract"] = build_task_contract(row)
    receipts = _authority_verification_receipts(row, drive_root)
    if receipts:
        authority["verification_receipts"] = copy.deepcopy(receipts)
    if isinstance(authority.get("plan_review_state"), dict):
        from ouroboros.tools.plan_review_artifacts import projected_authority_state

        authority["plan_review_state"], gap = projected_authority_state(
            drive_root, str(row.get("task_id") or row.get("id") or ""), authority["plan_review_state"],
            promotion=row.get("child_ref_promotion"))
        # Beside the recorded state, never inside it: every projection keeps this small field whole.
        authority.update({"plan_review_source_gap": gap} if gap else {})
    from ouroboros.delegate_output import project_task_output_readers

    return project_task_output_readers(authority)


def valid_task_result_authority_source(source: Any, task_id: Any) -> bool:
    """Whether a named predecessor is the exact host-issued actor pointer."""
    if not isinstance(source, dict):
        return False
    selected_id = str(task_id or "").strip()
    arguments = source.get("arguments")
    return bool(
        selected_id
        and str(source.get("kind") or "") == "task_result"
        and str(source.get("task_id") or "") == selected_id
        and str(source.get("tool") or "") == "get_task_result"
        and isinstance(arguments, dict)
        and str(arguments.get("task_id") or "") == selected_id
        and arguments.get("include_authority") is True
        and arguments.get("authority_fields") is None
        and arguments.get("source_only") is not True
    )


def _automatic_predecessor_authority_projection(
    row: Dict[str, Any], source: Dict[str, Any], *, drive_root: Any,
) -> Dict[str, Any]:
    """Bounded continuation envelope of the predecessor's record, digest included.

    The shared envelope producer (contracts SSOT) bounds the two growth carriers
    structurally: the nested ``task_contract`` inherits its OPERATIVE core minus
    the nested predecessor, and any field whose body outgrows the tool-result
    budget - measured on its serialized form - rides as a typed preview beside
    the named pull source. The digest is observed at binding over the complete
    projection. The successor's start materializes only the compact brief of
    this envelope (``main_context_authority.project_helper_predecessor_authority``);
    the complete canonical authority stays in the task store, pullable whole
    via the named source.
    """

    from ouroboros.contracts.task_contract import bounded_continuation_envelope

    authority = task_result_authority_projection(row, drive_root=drive_root)
    if isinstance(authority.get("plan_review_state"), dict):
        from ouroboros.task_results import plan_review_authority_core

        authority["plan_review_state"] = plan_review_authority_core(authority["plan_review_state"], source_ref=source)
    contract = authority.get("task_contract") if isinstance(authority.get("task_contract"), dict) else {}
    nested = contract.get("predecessor_authority") if isinstance(contract.get("predecessor_authority"), dict) else {}
    nested_source = nested.get("source") if isinstance(nested.get("source"), dict) else {}
    salvage = str(row.get("terminal_origin") or "") == TERMINAL_ORIGIN_HOST_SALVAGE
    return bounded_continuation_envelope(
        authority,
        digest_semantics="observed_at_binding",
        source_ref=source,
        salvage=salvage,
        reserve_source=True,
        extra={
            "previous_task_id": str(
                nested.get("task_id") or nested.get("previous_task_id")
                or nested_source.get("task_id") or ""
            ),
        },
    )

def validate_task_authority_sources(env: Any, task: Dict[str, Any]) -> Dict[str, Any]:
    """Materialize named authority sources or return one typed startup refusal."""
    root = pathlib.Path(
        task.get("budget_drive_root")
        or getattr(env, "budget_drive_root", None)
        or env.drive_root
    )

    def _unavailable(source: Dict[str, Any], detail: str) -> Dict[str, Any]:
        return {
            "reason_code": "authority_source_unavailable",
            "human_label": str(source.get("human_label") or source.get("task_id") or "task authority"),
            "source": dict(source),
            "detail": detail,
            "recovery_choices": ["retry_after_source_recovery", "start_explicit_fresh_task"],
        }

    origin_ref = task.get("origin_message_ref")
    owner_source = {"kind": "owner_message", "human_label": "originating owner message", "ref": origin_ref}
    if origin_ref not in (None, {}) and not isinstance(origin_ref, dict):
        return _unavailable(owner_source, "named owner source reference is not an object")
    if isinstance(origin_ref, dict) and origin_ref:
        from ouroboros.project_dialogue import (
            _text_sha256,
            owner_message_ref_is_valid,
            resolve_owner_message_source,
        )

        if not owner_message_ref_is_valid(origin_ref):
            return _unavailable(owner_source, "named owner source reference has an invalid host identity shape")

        origin_text = task.get("origin_message_text")
        if isinstance(origin_text, str) and origin_text:
            if _text_sha256(origin_text) != str(origin_ref.get("text_sha256") or ""):
                return _unavailable(owner_source, "retained origin text does not match its ingress checksum")
        else:
            try:
                matching = resolve_owner_message_source(root, origin_ref)
            except StoreMigrating:  # unknown yet, never absent: the start waits for the upgrade, not a crash
                return _unavailable(owner_source, "the named owner message is in saved history that is still "
                                                  "being upgraded; retry when the upgrade finishes")
            except StoreError as exc:
                return _unavailable(owner_source, f"the dialogue store could not answer: {exc}")
            if matching is None:
                return _unavailable(
                    owner_source, "named canonical owner row is not readable and no retention-proof text is stored")
            task["origin_message_text"] = str(matching.get("text") or "")

    predecessor = task.get("predecessor_authority_source")
    if predecessor not in (None, {}) and not isinstance(predecessor, dict):
        return _unavailable(
            {"kind": "task_result", "human_label": "predecessor task authority", "raw": predecessor},
            "named predecessor authority source is not an object",
        )
    if isinstance(predecessor, dict) and predecessor:
        predecessor_id = str(predecessor.get("task_id") or "").strip()
        if not valid_task_result_authority_source(predecessor, predecessor_id):
            return _unavailable(predecessor, "named predecessor authority source shape is invalid")
        from ouroboros.task_status import load_effective_task_result

        predecessor_row = (
            load_effective_task_result(root, predecessor_id, materialize_artifacts=False)
            if predecessor_id else None
        )
        # A missing file reads back as ``{}``: an empty row is no predecessor either.
        if not isinstance(predecessor_row, dict) or not predecessor_row:
            return _unavailable(predecessor, "named predecessor task result is missing or unreadable")
        from ouroboros.contracts.task_contract import task_input_sources
        from ouroboros.main_context_authority import project_helper_predecessor_authority

        # The pull pointer is written LAST: a projected row field named
        # ``source`` must not clobber the authority route the pull flow and
        # identity checks depend on. Every consumer of a predecessor gets the
        # same compact brief; the predecessor mind's own notes follow the
        # assignment as a foldable row, and the pointer reads the whole record.
        task["predecessor_authority"] = project_helper_predecessor_authority({
            **_automatic_predecessor_authority_projection(
                predecessor_row, predecessor, drive_root=root,
            ),
            "source": dict(predecessor),
        }, declared=task_input_sources(task) == "declared")
    else:
        metadata = task.get("metadata") if isinstance(task.get("metadata"), dict) else {}
        previous = metadata.get("project_last_task_result")
        if isinstance(previous, dict) and previous and not previous.get("authority_source"):
            task.setdefault("authority_historical_gaps", []).append({
                "kind": "legacy_predecessor_without_source",
                "task_id": str(previous.get("task_id") or ""),
            })

    current_id = str(task.get("id") or "").strip()
    if current_id:
        try:
            from ouroboros import plan_record
            from ouroboros.task_results import load_plan_review_state
            from ouroboros.task_store import exists

            if exists(root, current_id):
                load_plan_review_state(root, current_id)
                plan_record.load_plan_record(root, current_id)
        except Exception as exc:
            return _unavailable(
                {"kind": "plan_review_state", "task_id": current_id,
                 "human_label": str(task.get("title") or task.get("objective") or current_id)},
                f"current plan/review authority is unreadable ({type(exc).__name__})",
            )
    return {}


def persist_early_origin_stub(
    drive_root: Any, task: Dict[str, Any], *, write_result: Any = None,
) -> None:
    """Merge-persist ingress authority before the convertible task card exists.

    Origin-less turns write nothing. A storage failure is loud but
    non-fatal: the owner's task outlives its start message, and the subsequent
    full RUNNING write will encounter the same storage fault. ``write_result``
    preserves the existing agent test seam while production uses the canonical
    task-result writer.
    """
    ref = task.get("origin_message_ref")
    if not (isinstance(ref, dict) and ref):
        return
    from ouroboros.task_results import STATUS_RUNNING, write_task_result

    writer = write_result or write_task_result

    for attempt in range(2):
        try:
            writer(
                drive_root, str(task.get("id") or ""), STATUS_RUNNING,
                chat_id=task.get("chat_id"), origin_message_ref=dict(ref),
                origin_message_text=task.get("origin_message_text"), result="Task is starting.",
            )
            return
        except Exception:
            if attempt:
                log.warning("Early origin stub persistence failed", exc_info=True)
    try:
        append_jsonl(pathlib.Path(drive_root) / "logs" / "events.jsonl", {
            "ts": utc_now_iso(), "type": "origin_stub_persist_failed",
            "task_id": str(task.get("id") or ""),
        })
    except Exception:
        log.debug("origin_stub_persist_failed event write failed", exc_info=True)


def _is_release_tag(tag: str) -> bool:
    from ouroboros.tools.release_sync import normalize_release_tag

    return bool(normalize_release_tag(tag))


def check_uncommitted_changes(env: Any) -> Tuple[dict, int]:
    """Warn on dirty worker boot; rescue/reset is supervisor-owned, never worker-owned.

    A read only: it takes no optional index lock and removes none, since a lock's
    age says nothing about whether its writer is still running.
    """
    from ouroboros.git_observation import host_git_env

    try:
        result = subprocess.run(
            ["git", "--no-optional-locks", "status", "--porcelain"],
            cwd=str(env.repo_dir),
            capture_output=True, text=True, timeout=10, check=True, env=host_git_env()
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
    """Check budget remaining with warning thresholds.

    Server-side only: a worker verifying itself at construction need not read
    the money store once per worker; the server's own check already covers the
    install, and every reservation re-checks the limits anyway.
    """
    from ouroboros.utils import in_worker_process

    if in_worker_process():
        return {"status": "skipped", "reason": "worker_process"}, 0
    try:
        accounting_root = pathlib.Path(
            getattr(env, "budget_drive_root", None) or env.drive_path("state").parent
        )
        state_path = accounting_root / "state" / "state.json"
        state_data = read_json_dict(state_path)
        if state_data is None:
            return {
                "status": "error",
                "error": "state.json missing or invalid",
                "path": str(state_path),
            }, 1
        from ouroboros.settings_setup_contract import resolve_total_budget_usd
        total_budget = resolve_total_budget_usd()

        if total_budget is None:
            return {"status": "unconfigured"}, 0
        else:
            from ouroboros.usage_accounting import usage_projection

            accounting = usage_projection(accounting_root, global_limit_usd=total_budget)
            spent = float(accounting.get("settled_usd") or 0.0)
            remaining = float(accounting.get("remaining_known_usd") or 0.0)
            integrity_degraded = bool(accounting.get("integrity_degraded"))

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
            if integrity_degraded:
                status = "integrity_degraded"
                issues = max(1, issues)

            return {
                "status": status,
                "remaining_usd": round(remaining, 2),
                "total_usd": total_budget,
                "spent_usd": round(spent, 2),
                "confirmed_usd": float(accounting.get("confirmed_usd") or 0.0),
                "reserved_usd": float(accounting.get("reserved_usd") or 0.0),
                "unresolved_upper_bound_usd": float(
                    accounting.get("unresolved_upper_bound_usd") or 0.0
                ),
                "unknown_unmetered": int(accounting.get("unknown_unmetered") or 0),
                "cost_final": bool(accounting.get("cost_final")),
                "integrity_degraded": integrity_degraded,
                **({"warning": "quarantined ledger tail; paid cost may be incomplete"}
                   if integrity_degraded else {}),
                "accounting_authority": "physical_attempt_ledger",
            }, issues
    except Exception as e:
        return {
            "status": "error",
            "error": f"physical-attempt accounting unavailable: {e}",
        }, 1


def check_review_continuations(env: Any) -> Tuple[dict, int]:
    try:
        from ouroboros.task_continuation import list_review_continuations
        from ouroboros.task_results import STATUS_INTERRUPTED, load_task_result

        continuations, corrupt = list_review_continuations(env.drive_root)

        def _status(task_id: str) -> str:
            # One row read per continuation (TZ-1 A): a startup check never walks the store.
            try:
                return str((load_task_result(env.drive_root, task_id) or {}).get("status") or "")
            except (OSError, ValueError, StoreError):
                return ""

        rows = []
        interrupted = []
        for item in continuations:
            task_status = _status(item.task_id)
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


# The stray scan reads command lines with one ``ps`` per batch of pids (BSD and procps both take a comma
# list for ``-p``); a technical bound on argv length, not a judgment about which process counts.
_PS_BATCH = 256


def _commands_of(pids: list[int]) -> dict[int, str]:
    """Each pid's command line, as ``process_command`` reads it: one ``ps`` per batch, and a pid the batch
    did not answer (it exited, or the batch failed) read on its own as before, so no answer differs."""
    import subprocess as _subprocess

    from ouroboros.platform_layer import process_command

    commands: dict[int, str] = {}
    for start in range(0, len(pids), _PS_BATCH):
        batch = set(pids[start:start + _PS_BATCH])
        try:
            out = _subprocess.run(["ps", "-ww", "-o", "pid=,command=", "-p", ",".join(map(str, sorted(batch)))],
                                  capture_output=True, text=True, timeout=5)
        except Exception:
            continue
        for line in (out.stdout or "").splitlines():
            head, _sep, command = line.strip().partition(" ")
            if head.isdigit() and int(head) in batch:
                commands[int(head)] = command.strip()
    return {pid: commands[pid] if pid in commands else process_command(pid) for pid in pids}


def check_stray_server_processes(env: Any) -> Tuple[Dict[str, Any], int]:
    """Report ouroboros-server processes that belong to NO current install (v6.70.0).

    Field incident: four foreign `ouroboros server` processes lived on the
    owner's machine for weeks without any invariant noticing. This check only
    REPORTS (never kills — a sibling install may be legitimate): a pid counts
    as stray when its command line looks like an ouroboros server and it is
    neither this process tree, the recorded server process, nor any pid ever
    recorded in the custody ledger (conservative under-reporting: a reused
    ledger pid masks a stray rather than false-flagging a legitimate one). Scans
    only THIS user's processes; non-POSIX platforms skip (pgrep-based).

    Each report carries a ``scope``: ``same_install`` when the command line names
    THIS install's ``<REPO_DIR>/server.py``, else ``foreign``. The launcher reaps
    proven same-install strays before every generation
    (``ouroboros.launcher_server_reaper``), so a same_install WARN that persists
    means one of: a direct (unmanaged) run of this checkout, a process spared for
    lack of readable env proof, or a pid that survived the kill passes."""
    import os as _os
    import pathlib as _pathlib
    import re as _re
    import subprocess as _subprocess

    try:
        from ouroboros.platform_layer import IS_WINDOWS

        if IS_WINDOWS:
            return {"status": "skipped", "reason": "non-posix"}, 0
        out = _subprocess.run(
            # -U: this user's processes only — on shared multi-user hosts other
            # accounts run their own legitimate ouroboros installs. -i: packaged
            # desktop installs run "EMBEDDED_PYTHON server.py" under ~/Ouroboros
            # (capital O) — the very sibling class this invariant was built for.
            ["pgrep", "-U", str(_os.getuid()), "-fi", "ouroboros"],
            capture_output=True, text=True, timeout=5,
        )
        known: set[int] = {_os.getpid(), _os.getppid()}
        drive_root = _pathlib.Path(getattr(env, "drive_root", None) or env.drive_path("state").parent)
        try:
            import json as _json

            record = _json.loads((drive_root / "state" / "server_process.json").read_text(encoding="utf-8"))
            for key in ("pid", "pgid"):
                if isinstance(record.get(key), int):
                    known.add(int(record[key]))
        except Exception:
            pass
        try:
            for line in (drive_root / "state" / "process_ledger.jsonl").read_text(encoding="utf-8").splitlines():
                try:
                    row = _json.loads(line)
                except Exception:
                    continue
                if isinstance(row.get("pid"), int):
                    known.add(int(row["pid"]))
        except Exception:
            pass
        server_shape = _re.compile(
            r"(ouroboros(\.cli)?\s+server|ouroboros/(repo/)?server\.py|-m\s+ouroboros\.cli\s+server)",
            _re.IGNORECASE,
        )
        # The reaper's own identity rule (exact argv token after a python
        # interpreter, literal + resolved spellings): the scope label must agree
        # with what the launcher sweep would actually treat as a server, or an
        # editor merely opening server.py would read as a same_install stray.
        try:
            from ouroboros.config import REPO_DIR as _repo_dir
            from ouroboros.launcher_server_reaper import (
                command_names_our_server as _names_our_server,
                install_server_path_forms as _server_path_forms,
            )

            same_install_paths = _server_path_forms(_repo_dir)
        except Exception:
            # The identity rule could not load: an unlabelled answer is honest,
            # a hard 'foreign' would deny a genuine same-install stray.
            same_install_paths = set()
            _names_our_server = None
        stray: list[dict[str, Any]] = []
        candidates: list[int] = []
        for line in (out.stdout or "").splitlines():
            try:
                pid = int(line.strip())
            except ValueError:
                continue
            if pid not in known:
                candidates.append(pid)
        commands = _commands_of(candidates)
        known_groups: set[int] | None = None
        for pid in candidates:
            command = commands[pid]
            if not server_shape.search(command or ""):
                continue
            # Same process group as a known pid => part of a known tree, the known groups read once.
            # (Per-pid defensiveness: most ledger pids are dead, and one
            # ProcessLookupError must not disable the exclusion entirely.)
            try:
                from ouroboros.platform_layer import process_group_id

                if known_groups is None:
                    groups = set()
                    for k in known:
                        if k > 0:
                            try:
                                groups.add(process_group_id(k))
                            except Exception:
                                continue
                    known_groups = groups
                if process_group_id(pid) in known_groups:
                    continue
            except Exception:
                pass
            if _names_our_server is None:
                scope = "unknown"
            else:
                scope = ("same_install"
                         if _names_our_server(command or "", same_install_paths)
                         else "foreign")
            stray.append({"pid": pid, "command": command[:160], "scope": scope})
        if stray:
            log.warning("Stray ouroboros server process(es) outside this install: %s", stray)
            return {"status": "stray_processes", "processes": stray}, 1
        return {"status": "ok"}, 0
    except Exception:
        return {"status": "skipped"}, 0


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
    # The map's measured block is the host's: refresh it in place at every start, keeping
    # the text the mind wrote around it (a missing file is generated).
    try:
        from ouroboros.world_profiler import refresh_world_profile

        world_refreshed = refresh_world_profile(
            str(world_path), str(getattr(env, "repo_dir", "") or ""), str(memory_dir.parent))
    except Exception:
        log.debug("WORLD.md measured block refresh failed", exc_info=True)
        world_refreshed = False
    world_ok = world_path.exists()

    checks["identity"] = {"exists": identity_path.exists(), "non_empty": identity_ok}
    checks["scratchpad"] = {"exists": scratchpad_ok}
    checks["world_profile"] = {"exists": world_ok, "measured_refreshed": world_refreshed}

    if not identity_ok:
        issues += 1
        log.warning("identity.md missing or empty — continuity at risk (Bible P1)")
    if not scratchpad_ok:
        issues += 1
        log.warning("scratchpad.md missing — working memory not available (Bible P1)")
    if not world_ok:
        issues += 1
        log.warning("WORLD.md missing — environment profile not available")

    configured_model = runtime_setting("OUROBOROS_MODEL", "")
    checks["model"] = {"configured": configured_model or "(not set)"}
    if not configured_model:
        issues += 1

    checks["extension_health"], issue_count = check_extension_health(env)
    issues += issue_count

    checks["stray_server_processes"], issue_count = check_stray_server_processes(env)
    issues += issue_count

    # The startup boundary proves that process-local reviewer threads from the
    # prior worker are gone. Delegated rows with a durable invocation token keep
    # their existing recovery path; TTL never authorizes a paid resend.
    try:
        review_reconciliation = _reconcile_review_attempts_on_startup(env)
        reconciled = review_reconciliation["reconciled"]
        expired = review_reconciliation["expired"]
        if reconciled:
            log.warning(
                "Reconciled custody for %d reviewed attempt(s) on startup",
                len(reconciled),
            )
        if expired:
            log.warning(
                "Auto-expired %d stale unpaid reviewed attempt(s) on startup",
                len(expired),
            )
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
    # No stdlib WARNING beside the durable ``startup_verification`` row: the row and the Logs
    # panel carry every check (#1184); a fact with a durable row gets no second line.


def _reconcile_review_attempts_on_startup(env: Any) -> Dict[str, Any]:
    """Reconcile only owners proven dead from an older server generation."""
    from ouroboros.review_owner_custody import (
        reconcile_review_custody_on_process_start,
    )

    drive_root = (
        pathlib.Path(env.drive_root)
        if hasattr(env, "drive_root")
        else env.drive_path("").parent
    )
    return reconcile_review_custody_on_process_start(drive_root)


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


def _native_restart_error(git_sha: str) -> str:
    # The outer launcher verified the installed native inputs before spawning
    # this core. Workers inherit that generation's fact, not a persisted PASS.
    # No APK/signature subprocess runs under the campaign's state lock.
    if not os.environ.get("OUROBOROS_EXTERNAL_HOST_UPDATE"):
        return ""
    try:
        native = json.loads(os.environ.get("OUROBOROS_EXTERNAL_HOST_RESULT", "{}"))
        if (native.get("status") == "verified" and native.get("source_commit") == git_sha
                and all(isinstance(native.get(key), str) and len(native[key]) == 64
                        for key in ("input_sha256", "apk_sha256", "signer_sha256"))):
            return ""
    except (ValueError, TypeError, AttributeError):
        pass
    return "native_update_failed_or_unverified"


def verify_restart(env: Any, git_sha: str) -> None:
    """Best-effort restart verification."""
    from supervisor import state as supervisor_state

    campaign_path = env.drive_path("state") / "evolution_campaign.json"
    supervisor_state.assert_test_data_path(campaign_path)
    # The campaign write and the ledger append are two writes: re-derive any
    # cycle-outcome row a crash between them lost, before anything reads them.
    # The scan-then-append runs under the SAME locks/state.lock both resolution
    # writers below hold across their campaign write AND their tag, so a second
    # booting worker sees either the unresolved campaign or the tagged ledger —
    # never the gap in between (an unlocked scan landing there is exactly the
    # duplicate-row writer S22 caught). An unavailable lock SKIPS the repair
    # (the next boot replays it) rather than backfilling unlocked; the boot
    # itself never fails on it.
    drive_root = campaign_path.parent.parent
    state_lock_path = env.drive_path("locks") / "state.lock"
    try:
        backfill_fd = supervisor_state.acquire_file_lock(state_lock_path)
        if backfill_fd is None:
            log.warning("Skipped cycle-outcome backfill: %s unavailable", state_lock_path)
        else:
            try:
                backfill_missing_cycle_outcomes(drive_root, read_json_dict(campaign_path) or {})
            finally:
                supervisor_state.release_file_lock(state_lock_path, backfill_fd)
    except Exception:
        log.debug("Failed to backfill cycle-outcome checkpoints at boot", exc_info=True)

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

    def _restart_authority_error(
        campaign: Dict[str, Any], tx: Dict[str, Any], *,
        claim: Any = None, require_claim: bool = False,
    ) -> str:
        """Validate v2 restart authority while preserving legacy v1 recovery."""
        try:
            schema_version = int(tx.get("schema_version") or 1)
        except (TypeError, ValueError):
            schema_version = 1
        strict = (
            schema_version >= 2
            or "commit_receipt" in tx
            or claim is not None
        )
        if not strict:
            return _native_restart_error(git_sha)
        if require_claim and not isinstance(claim, dict):
            return "restart_claim_missing" if claim is None else "restart_claim_invalid"
        expected = {
            "campaign_id": str(campaign.get("id") or ""),
            "transaction_id": str(tx.get("transaction_id") or ""),
            "task_id": str(tx.get("task_id") or ""),
            "commit_sha": str(tx.get("commit_sha") or ""),
        }
        if isinstance(claim, dict) and any(
            str(claim.get(key) or "") != value for key, value in expected.items()
        ):
            return "restart_claim_mismatch"
        from supervisor.evolution_lifecycle import evolution_commit_receipt_error

        return evolution_commit_receipt_error(tx, **expected) or _native_restart_error(git_sha)

    def _boot_reconcile_generation() -> str:
        from supervisor.evolution_lifecycle import current_evolution_boot_generation

        return current_evolution_boot_generation()

    from supervisor.evolution_lifecycle import adopt_evolution_commit_intent

    def _reconcile_dangling_campaign_transaction(observed_sha: str) -> None:
        try:
            snapshot = read_json_dict(campaign_path) or {}
            snapshot_tx = (
                snapshot.get("active_transaction")
                if isinstance(snapshot.get("active_transaction"), dict) else {}
            )
            snapshot_sha = str(snapshot_tx.get("commit_sha") or "").strip()
            snapshot_reachable = bool(
                snapshot_sha and _commit_reachable(snapshot_sha, observed_sha)
            )
            # #1539: a commit its task's body candidate still holds awaits adoption; read like
            # reachability, outside the lock (None: unreadable, which never proves it lost).
            snapshot_task, snapshot_held = str(snapshot_tx.get("task_id") or ""), None
            if snapshot_sha and not snapshot_reachable:
                from ouroboros.body_candidate import holds_commit

                snapshot_held = holds_commit(snapshot_task, snapshot_sha, drive_root)
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
            gen = _boot_reconcile_generation()
            # The whole read-check-write runs under update_json_locked so the
            # gen-gate is ATOMIC: at boot ~10 workers whose os.rename of the pending
            # file lost all call this; the lock + in-lock re-read make exactly one
            # of them reconcile per generation (the rest see the claimed gen and
            # abort), instead of an unlocked stampede that double-increments
            # absorbed_cycles_done / lost-updates each other. Invariant shared with
            # the os.rename WINNER's _mark_campaign_restart_verified: the campaign
            # resolution and its cycle_outcome ledger tag are ONE critical section
            # under locks/state.lock, and the repair-side reader (the boot backfill
            # at the top of verify_restart) takes the same lock — so no actor can
            # observe a resolved campaign whose tag is still pending. (The narrow
            # winner-vs-loser ordering edge on an ancestor-not-HEAD commit is
            # idempotency-mitigated — see _mark_campaign_restart_verified.)
            event: Dict[str, Any] = {}  # captured in-lock for post-lock event logging

            outcome_snapshot: Dict[str, Any] = {}
            state_path = env.drive_path("state") / "state.json"

            def _mutate(campaign: Dict[str, Any]):
                if not isinstance(campaign, dict):
                    return None
                from supervisor.state import control_in_copy

                stop_known, stopped = control_in_copy(state_path, "evolution_owner_stopped")
                if (
                    not stop_known or bool(stopped)  # an unknown owner stop is not "not stopped" (#1307)
                    or isinstance(campaign.get("stop_intent"), dict)  # a Stop recorded beside state (#1307)
                    or campaign.get("status") not in {"active", "paused"}
                ):
                    return None
                if gen and str(campaign.get("last_boot_reconcile_gen") or "") == gen:
                    return None  # already reconciled this generation — abort, no write
                tx = campaign.get("active_transaction") if isinstance(campaign.get("active_transaction"), dict) else {}
                commit_sha = str(tx.get("commit_sha") or "").strip()
                expected_sha, reachable = snapshot_sha, snapshot_reachable
                if not commit_sha and not bool(tx.get("restart_verified")):
                    # A crash between the reviewed commit and its SHA receipt: the
                    # pre-commit intent proves the commit sitting on HEAD is this
                    # transaction's, so finish the receipt in THIS write.
                    commit_sha = adopt_evolution_commit_intent(campaign, tx, observed_sha)
                    if commit_sha:
                        expected_sha, reachable = commit_sha, True
                    elif (recovered := adopt_evolution_commit_intent(campaign, tx)):
                        # ...or the task's body candidate holds it (#1539): its receipt and
                        # reviewed provenance are restored, but only the serving SHA proves a
                        # restart, so the unadopted commit stays open until adoption lands it.
                        tx["restart_guidance"] = _candidate_restart_guidance(tx, recovered)
                        if gen:
                            campaign["last_boot_reconcile_gen"] = gen
                        campaign["updated_at"] = utc_now_iso()
                        return campaign
                # Capture before the absorbed branch pops the campaign's link.
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
                if commit_sha != expected_sha:
                    if gen:
                        campaign["last_boot_reconcile_gen"] = gen
                        campaign["updated_at"] = utc_now_iso()
                        return campaign
                    return None
                now = utc_now_iso()
                authority_error = _restart_authority_error(campaign, tx)
                if authority_error:
                    campaign["last_boot_reconcile_gen"] = gen
                    tx["restart_required"] = True
                    tx["restart_verified"] = False
                    tx["restart_authority_error"] = authority_error
                    tx["restart_observed_sha"] = observed_sha
                    tx["updated_at"] = now
                    campaign["active_transaction"] = tx
                    campaign["progress_notes"] = (
                        "Restart reconciliation kept the evolution transaction open: "
                        f"exact commit authority failed ({authority_error})."
                    )
                    campaign["updated_at"] = now
                    event.update({
                        "ts": now,
                        "type": "evolution_tx_reconcile_blocked",
                        "ok": False,
                        "reason": authority_error,
                        "campaign_id": str(campaign.get("id") or ""),
                        "transaction_id": str(tx.get("transaction_id") or ""),
                        "task_id": str(tx.get("task_id") or ""),
                        "commit_sha": commit_sha,
                        "observed_sha": observed_sha,
                    })
                    return campaign
                held = snapshot_held if str(tx.get("task_id") or "") == snapshot_task else None
                if not reachable and held is not False:
                    # Before its restart-bound adoption the serving HEAD lacks the candidate's
                    # commit: keep the exact transaction, receipt and backlog link; adopt nothing.
                    tx.update({"restart_required": True, "restart_verified": False,
                               "restart_observed_sha": observed_sha, "updated_at": now,
                               "restart_guidance": _candidate_restart_guidance(tx, commit_sha, held=held)})
                    campaign.update({"last_boot_reconcile_gen": gen, "active_transaction": tx, "updated_at": now})
                    reason = "candidate_holds_commit" if held else "candidate_unreadable"
                    campaign["progress_notes"] = (
                        f"Restart reconciliation kept the evolution transaction open ({reason}): "
                        f"reviewed commit {commit_sha[:12]} awaits its restart-bound adoption.")
                    event.update({"ts": now, "type": "evolution_tx_awaiting_adoption", "ok": False,
                                  "reason": reason, "task_id": str(tx.get("task_id") or ""),
                                  "commit_sha": commit_sha, "observed_sha": observed_sha})
                    return campaign
                campaign["last_boot_reconcile_gen"] = gen
                tx["restart_verified_at"] = now
                tx["restart_observed_sha"] = observed_sha
                tx["updated_at"] = now
                if reachable:
                    tx["restart_required"] = False
                    tx["restart_verified"] = True
                    tx["verified_by"] = "boot_reconciliation"
                    tx["cycle_outcome"] = "absorbed"
                    if not tx.get("absorbed_counted"):
                        campaign["absorbed_cycles_done"] = int(campaign.get("absorbed_cycles_done") or 0) + 1
                        tx["absorbed_counted"] = True
                    _append_unique_transaction(campaign, tx)
                    campaign.pop("active_transaction", None)
                    # Absorption is evidence, not proof an authored intention is met: the
                    # backlog note stays its author's; the link lives on in the ledger row.
                    campaign.pop("post_task_backlog_id", None)
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

            lock_fd = supervisor_state.acquire_file_lock(state_lock_path)
            if lock_fd is None:
                return
            try:
                update_json_locked(campaign_path, _mutate)
                # The tag lands INSIDE the state lock, like the marker path's: a
                # ledger failure is swallowed by append_cycle_outcome_tag, so the
                # boot never breaks on it — only the lock hold grows by one append.
                if outcome_snapshot.get("transaction"):
                    append_cycle_outcome_tag(
                        drive_root,
                        campaign=outcome_snapshot.get("campaign"),
                        transaction=outcome_snapshot.get("transaction"),
                        source="boot_reconcile",
                        backlog_id=str(outcome_snapshot.get("backlog_id") or ""),
                    )
            finally:
                supervisor_state.release_file_lock(state_lock_path, lock_fd)
            if event:
                append_jsonl(env.drive_path("logs") / "events.jsonl", event)
        except Exception:
            log.debug("Failed to reconcile dangling evolution transaction", exc_info=True)

    mark_error: Dict[str, str] = {"reason": _native_restart_error(git_sha)}

    def _mark_campaign_restart_verified(
        expected_sha: str, observed_sha: str, ok: bool, claim: Any = None,
    ) -> bool:
        # Only the os.rename winner reaches this transition. It stamps the custody
        # generation before removing the claim; losers either see the claimed file or
        # the generation stamp, so markerless reconciliation cannot bypass the claim.
        lock_fd = None
        try:
            lock_fd = supervisor_state.acquire_file_lock(state_lock_path)
            if lock_fd is None:
                mark_error["reason"] = "state_lock_unavailable"
                return False
            campaign = read_json_dict(campaign_path) or {}
            if not isinstance(campaign, dict):
                return bool(ok)
            tx = campaign.get("active_transaction") if isinstance(campaign.get("active_transaction"), dict) else {}
            if not tx:
                if claim is not None:
                    mark_error["reason"] = "transaction_missing"
                    mark_error["durable"] = "1"
                    return False
                mark_error["durable"] = "1"
                return bool(ok and not mark_error.get("reason"))
            from supervisor.state import control_in_copy

            stop_known, stopped = control_in_copy(env.drive_path("state") / "state.json", "evolution_owner_stopped")
            if not stop_known or bool(stopped) or isinstance(campaign.get("stop_intent"), dict):
                mark_error["reason"] = "owner_stopped"
                mark_error["durable"] = "1"
                return False
            if campaign.get("status") not in {"active", "paused"}:
                mark_error["reason"] = "campaign_not_active"
                mark_error["durable"] = "1"
                return False
            prior_gen = str(campaign.get("last_boot_reconcile_gen") or "")
            gen = _boot_reconcile_generation()
            # Captured before the absorbed branch pops the campaign's link.
            backlog_id_before_close = str(campaign.get("post_task_backlog_id") or "")
            commit_sha = str(tx.get("commit_sha") or "").strip()
            authority_error = _restart_authority_error(
                campaign, tx, claim=claim, require_claim=True,
            )
            if authority_error:
                now = utc_now_iso()
                mark_error["reason"] = authority_error
                if gen:
                    campaign["last_boot_reconcile_gen"] = gen
                tx["restart_required"] = True
                tx["restart_verified"] = False
                tx["restart_verified_at"] = now
                tx["restart_expected_sha"] = expected_sha
                tx["restart_observed_sha"] = observed_sha
                tx["restart_authority_error"] = authority_error
                tx["updated_at"] = now
                campaign["active_transaction"] = tx
                campaign["progress_notes"] = (
                    "Restart verification kept the evolution transaction open: "
                    f"exact commit authority failed ({authority_error})."
                )
                campaign["updated_at"] = now
                atomic_write_json(campaign_path, campaign, trailing_newline=True)
                mark_error["durable"] = "1"
                return False
            if isinstance(claim, dict) and gen and prior_gen == gen:
                # A replacement worker inherits the server's custody generation.
                # Keep the marker pending until the server itself has restarted.
                mark_error["reason"] = "restart_generation_unchanged"
                return False
            if gen:
                campaign["last_boot_reconcile_gen"] = gen
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
                mark_error["durable"] = "1"
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
                # Absorption is evidence, not proof an authored intention is met: the
                # backlog note stays its author's; the link lives on in the ledger row.
                campaign.pop("post_task_backlog_id", None)
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
                # This cycle absorbed no reviewed commit: the link has no outcome to ride.
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
            mark_error["durable"] = "1"
            if tx.get("cycle_outcome") == "absorbed":
                append_cycle_outcome_tag(
                    drive_root,
                    campaign={"id": campaign.get("id"), "objective": campaign.get("objective")},
                    transaction=tx,
                    source="restart_verified",
                    backlog_id=backlog_id_before_close,
                )
            return bool(ok)
        except Exception:
            mark_error["reason"] = "restart_campaign_write_failed"
            log.debug("Failed to update evolution campaign restart verification", exc_info=True)
            return False
        finally:
            if lock_fd is not None:
                supervisor_state.release_file_lock(state_lock_path, lock_fd)

    try:
        pending_path = env.drive_path('state') / 'pending_restart_verify.json'
        claim_path = pending_path.with_name(f"pending_restart_verify.claimed.{os.getpid()}.json")
        try:
            os.rename(str(pending_path), str(claim_path))
        except (FileNotFoundError, Exception):
            claimed_paths = list(pending_path.parent.glob(
                f"{pending_path.stem}.claimed.*{pending_path.suffix}"
            ))
            if not claimed_paths:
                _reconcile_dangling_campaign_transaction(git_sha)
                return
            from ouroboros.platform_layer import pid_is_alive

            prefix = f"{pending_path.stem}.claimed."
            reclaimed = False
            for stale_path in claimed_paths:
                raw_pid = stale_path.name[len(prefix):-len(pending_path.suffix)]
                if not raw_pid.isdecimal() or pid_is_alive(int(raw_pid)):
                    return
                try:
                    os.rename(str(stale_path), str(claim_path))
                    reclaimed = True
                    break
                except Exception:
                    continue
            if not reclaimed:
                return
        try:
            claim_data = read_json_dict(claim_path)
            if claim_data is None:
                _mark_campaign_restart_verified("", git_sha, False, None)
                append_jsonl(env.drive_path('logs') / 'events.jsonl', {
                    'ts': utc_now_iso(), 'type': 'restart_verify',
                    'pid': os.getpid(), 'ok': False,
                    'error': 'pending_restart_verify_invalid',
                    'authority_error': mark_error.get("reason", ""),
                    'observed_sha': git_sha,
                })
            else:
                expected_sha = str(claim_data.get("expected_sha", "")).strip()
                sha_ok = bool(expected_sha and expected_sha == git_sha)
                evolution_claim = claim_data.get("evolution_claim")
                campaign_ok = _mark_campaign_restart_verified(
                    expected_sha, git_sha, sha_ok, evolution_claim,
                )
                ok = bool(sha_ok and campaign_ok)
                event = {
                    'ts': utc_now_iso(), 'type': 'restart_verify',
                    'pid': os.getpid(), 'ok': ok,
                    'expected_sha': expected_sha, 'observed_sha': git_sha,
                }
                if mark_error.get("reason"):
                    event["error"] = mark_error["reason"]
                append_jsonl(env.drive_path('logs') / 'events.jsonl', event)
        except Exception:
            log.debug("Failed to log restart verify event", exc_info=True)
            pass
        if mark_error.get("durable") == "1":
            try:
                claim_path.unlink()
            except Exception:
                log.debug("Failed to delete restart verify claim file", exc_info=True)
                pass
        else:
            try:
                if claim_path.is_file() and not pending_path.exists():
                    os.rename(str(claim_path), str(pending_path))
            except Exception:
                log.debug("Failed to restore unresolved restart verify claim", exc_info=True)
                pass
    except Exception:
        log.debug("Restart verification failed", exc_info=True)
        pass
