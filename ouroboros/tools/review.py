"""Multi-model review and unified pre-commit review gate."""

import os  # noqa: F401 -- historical import surface kept for monkeypatching tests
import json
import asyncio  # noqa: F401 -- historical import surface kept for monkeypatching tests
import logging
import pathlib
from typing import Any, List, Optional

from ouroboros.llm import LLMClient  # noqa: F401 -- historical import surface kept for monkeypatching tests
from ouroboros.utils import (
    run_cmd,
    append_jsonl,
    estimate_tokens,
    truncate_review_artifact,  # noqa: F401 -- historical import surface kept for monkeypatching tests
    utc_now_iso,
)
from ouroboros import config as _cfg
from ouroboros.review_substrate import SLOT_ID_PREFIX, slot_id_for_row  # noqa: F401 -- historical import surface kept for monkeypatching tests
from ouroboros.tools.registry import ToolEntry, ToolContext
from ouroboros.triad_review import (
    REVIEW_JSON_ARRAY_CONTRACT,
    extract_json_array,
    parse_model_review_results,
    review_query_error_payload as _review_query_error_payload,  # noqa: F401 -- historical import surface kept for monkeypatching tests
)

log = logging.getLogger(__name__)

from ouroboros.reviewer_window import reviewer_context_window, window_scaled_reserves
from ouroboros.tools.review_synthesis import quorum_input_token_limit as _quorum_input_token_limit
from ouroboros.tools.review_helpers import (
    REPO_ROOT as _REPO_ROOT,
    load_checklist_section as _load_checklist_section_precise,
    load_governance_doc,
    build_touched_file_pack,
    build_goal_section,
    build_scope_section,
    review_drive_root,  # noqa: F401 -- historical import surface kept for monkeypatching tests
    build_rebuttal_section,
    CRITICAL_FINDING_CALIBRATION,
    REPO_ANTI_PATTERN_LOCK_GUARD,
    REVIEW_PREAMBLE,
    build_self_verification_template,
    build_review_history_section as _build_review_history_section,
    calibrated_input_token_limit,
    emit_review_usage,  # noqa: F401 -- historical import surface kept for monkeypatching tests
    format_name_status_for_preflight,
    format_review_history_entry as _format_review_entry,
    REVIEW_PROMPT_TOKEN_BUDGET,
    single_line as _single_line,
)
from ouroboros.tools.review_multi_model import (  # noqa: F401 -- intentional public re-exports
    CONCURRENCY_LIMIT,
    DEFAULT_REVIEW_MODEL_TIMEOUT_SEC,
    MAX_MODELS,
    _CONSTITUTIONAL_PREAMBLE,
    _handle_multi_model_review,
    _multi_model_review_async,
    _parse_model_response,
    _query_model,
    _review_model_timeout_sec,
    _review_output_budget,
)


# Derived alias; ``review_helpers.REPO_ROOT`` remains the repo-root SSOT.
_CHECKLISTS_PATH = _REPO_ROOT / "docs" / "CHECKLISTS.md"


def get_tools():
    return [
        ToolEntry(
            name="task_acceptance_review",
            schema={
                "name": "task_acceptance_review",
                "description": (
                    "Record a task-result claim, checklist, evidence, and optional agent disposition. "
                    "For a root task in auto/required mode this is a cheap evidence call: the host runs "
                    "the only authoritative reviewer panel after the turn becomes structurally eligible. "
                    "Child-task and off-mode behavior is unchanged."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "claim": {"type": "string", "description": "Final claim or task result the agent intends to release."},
                        "goal": {"type": "string", "description": "Original task goal."},
                        "evidence": {"type": "object", "description": "Relevant tool trace, artifacts, tests, and observed facts."},
                        "checklist": {"type": "string", "default": "", "description": "Optional acceptance checklist."},
                        "agent_disposition": {
                            "type": "string",
                            "enum": ["accepted", "rejected", "partial", "deferred"],
                            "default": "",
                            "description": "Optional agent-authored stance on the acceptance review: accepted, rejected, partial, or deferred. Advisory only.",
                        },
                        "rationale": {
                            "type": "string",
                            "default": "",
                            "description": "Optional concise rationale for agent_disposition, especially when rejecting, partially accepting, or deferring reviewer feedback. If rationale is provided without a disposition, the stance defaults to partial.",
                        },
                        "obligation_dispositions": {
                            "type": "array",
                            "default": [],
                            "description": "Optional per-obligation dispositions when the host surfaced OPEN OBLIGATIONS (blocking review policy): one entry per obligation id with disposition addressed|rejected|deferred and a short reason.",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "id": {"type": "string"},
                                    "disposition": {"type": "string", "enum": ["addressed", "rejected", "deferred"]},
                                    "reason": {"type": "string"},
                                },
                                "required": ["id", "disposition"],
                            },
                        },
                    },
                    "required": ["claim", "goal"],
                },
            },
            handler=_handle_task_acceptance_review,
            timeout_sec=900,
        )
    ]


def _handle_task_acceptance_review(
    ctx: ToolContext,
    claim: str = "",
    goal: str = "",
    evidence: Optional[dict] = None,
    checklist: str = "",
    agent_disposition: str = "",
    rationale: str = "",
    obligation_dispositions: Optional[list] = None,
) -> str:
    from ouroboros.config import get_task_review_mode, resolve_effort
    from ouroboros.review_evidence import (
        build_task_acceptance_evidence,
        task_acceptance_evidence_revision,
    )
    from ouroboros.task_results import resolve_task_lineage

    # v6.51.0 idea-2: build the process-aware evidence packet (full contract +
    # first-class verification_summary + host-collected redacted repo_diff + leak-safe
    # artifacts + provenance tags). The agent-tool (auto) path has no host-owned turn
    # trace, so there is no tool_trajectory and include_recent_commit stays False (it
    # cannot prove a commit happened THIS turn). The agent's own evidence is preserved
    # under `agent_supplied` (its repo_diff demoted to agent_supplied_repo_diff) — never
    # promoted to host-fact status; repo_diff is ALWAYS the HOST-collected structural fact.
    legacy_aliases = []
    if str(agent_disposition or "").strip():
        legacy_aliases.append("agent_disposition")
    if obligation_dispositions:
        legacy_aliases.append("obligation_dispositions")
    if legacy_aliases:
        try:
            append_jsonl(ctx.drive_logs() / "events.jsonl", {
                "ts": utc_now_iso(),
                "type": "deprecated_task_acceptance_alias",
                "task_id": str(getattr(ctx, "task_id", "") or ""),
                "aliases": legacy_aliases,
                "removal": "next_major",
            })
            log.warning(
                "Deprecated task-acceptance aliases used: %s (removal: next major)",
                ", ".join(legacy_aliases),
            )
        except Exception:
            log.warning(
                "Failed to persist deprecated task-acceptance alias event for %s",
                legacy_aliases,
                exc_info=True,
            )

    agent_evidence = dict(evidence or {})
    # Bind the cheap evidence revision to the agent's actual acceptance claim,
    # goal, and checklist as well as its supporting references.  Otherwise two
    # materially different claims over the same evidence dict would share a
    # misleading revision even though the host panel must treat them separately.
    agent_evidence["acceptance_request"] = {
        "claim": str(claim or ""),
        "goal": str(goal or ""),
        "checklist": str(checklist or ""),
    }
    disposition = str(agent_disposition or "").strip().lower()
    if disposition not in {"accepted", "rejected", "partial", "deferred"}:
        disposition = ""
    agent_rationale = " ".join(str(rationale or "").split()).strip()
    # v6.54.4 obligations layer: normalized per-obligation dispositions ride the
    # same agent_decision envelope (the existing v6.54.0 mechanism, extended to
    # obligation granularity). The host loop applies them to the per-task
    # acceptance_obligations it collected under blocking enforcement.
    normalized_ob: list = []
    for entry in (obligation_dispositions or []):
        if not isinstance(entry, dict):
            continue
        oid = str(entry.get("id") or "").strip()
        odisp = str(entry.get("disposition") or "").strip().lower()
        if not oid or odisp not in {"addressed", "rejected", "deferred"}:
            continue
        normalized_ob.append({
            "id": oid[:40],
            "disposition": odisp,
            "reason": " ".join(str(entry.get("reason") or "").split())[:500],
        })
    agent_decision = {}
    if disposition or agent_rationale or normalized_ob:
        agent_decision = {
            "disposition": disposition or "partial",
            "rationale": agent_rationale[:1000],
            "source": "agent_task_acceptance_review_tool",
        }
        if normalized_ob:
            agent_decision["obligation_dispositions"] = normalized_ob
        agent_evidence["agent_decision"] = agent_decision

    evidence = build_task_acceptance_evidence(
        ctx,
        agent_evidence=agent_evidence,
        drive_root=pathlib.Path(ctx.drive_root) if getattr(ctx, "drive_root", None) else None,
        task_id=str(getattr(ctx, "task_id", "") or ""),
    )

    metadata = (
        getattr(ctx, "task_metadata", {})
        if isinstance(getattr(ctx, "task_metadata", {}), dict)
        else {}
    )
    lineage = resolve_task_lineage(
        getattr(ctx, "task_id", ""),
        metadata=metadata,
        root_task_id=getattr(ctx, "root_task_id", None),
        parent_task_id=getattr(ctx, "parent_task_id", None),
        delegation_role=getattr(ctx, "delegation_role", None),
        original_task_id=getattr(ctx, "original_task_id", None),
        timeout_retry_from=getattr(ctx, "timeout_retry_from", None),
    )
    task_id = str(lineage["task_id"])
    is_root_task = bool(lineage["is_root_task"])
    if get_task_review_mode() in {"auto", "required"} and is_root_task:
        evidence_revision = task_acceptance_evidence_revision(evidence)
        deferred = {
            "status": "deferred_to_host_acceptance",
            "authoritative": False,
            "evidence_revision": evidence_revision,
            "request": {
                "surface": "task_acceptance",
                "goal": str(goal or ""),
                "subject": str(claim or ""),
                "checklist": str(checklist or ""),
                "task_id": task_id,
            },
            # The host rebuilds host-attested evidence at the authoritative
            # fence, but it cannot reconstruct the agent's claims/references
            # from the capped tool trajectory.  Preserve the already redacted,
            # bounded agent-supplied section in this existing trace record so
            # the one host panel sees exactly what the cheap root call recorded.
            "evidence_refs": {
                "revision": evidence_revision,
                "sections": sorted(
                    str(key) for key in evidence if str(key) != "__provenance__"
                ),
                "canonical_payload": evidence.get("canonical_payload") or {},
                "aliases": evidence.get("aliases") or {},
                "provenance": evidence.get("__provenance__") or {},
            },
            "agent_supplied": evidence.get("agent_supplied") or {},
        }
        if agent_decision:
            deferred["agent_decision"] = agent_decision
        return json.dumps(deferred, ensure_ascii=False, indent=2, default=str)

    from ouroboros.review_substrate import (
        ReviewRequest,
        build_improvement_capsule,
        dissent_findings,
        reviewer_slots,
        run_review_request,
    )

    request = ReviewRequest(
        surface="task_acceptance",
        goal=goal,
        subject=claim,
        evidence=evidence,
        checklist=checklist,
        policy={
            "raw_output_must_be_preserved": True,
            # min_successful_slots is set below from adaptive_quorum(len(slots)) —
            # the SSOT — once the actual reviewer slot count is known.
            "fail_closed_on_errors": True,
            "classify_outcome_tier": True,
            "max_physical_attempts_per_actor": 2,
        },
        task_id=str(getattr(ctx, "task_id", "") or ""),
    )
    # Task acceptance stays on the API by owner decision (D15: harness slots
    # only for commit triad, scope, advisory). No route_env_key = api_chat pin.
    slots = reviewer_slots(effort=resolve_effort("review"), role_hint="task acceptance")
    request.policy["min_successful_slots"] = _cfg.adaptive_quorum(len(slots))
    result = run_review_request(request, slots=slots, drive_root=pathlib.Path(ctx.drive_root), usage_ctx=ctx)
    # Agent self-call (auto): lead with the compact improvement capsule (the
    # actionable feedback) and keep the full structured result available for the
    # agent that explicitly asked for detail.
    capsule = build_improvement_capsule(result)
    payload_dict = dict(result.__dict__)
    # Dissent is recorded on the agent-called path too, so the tool-result
    # capture lands acceptance_decision.dissent_noted on EVERY path.
    payload_dict["dissent_noted"] = bool(dissent_findings(result))
    if agent_decision:
        payload_dict["agent_decision"] = agent_decision
    payload = json.dumps(payload_dict, ensure_ascii=False, indent=2, default=str)
    return f"{capsule}\n\n<full_review>\n{payload}\n</full_review>" if capsule else payload


# Unified pre-commit review gate.

def _load_checklist_section() -> str:
    """Load Repo Commit Checklist, fail-closed if missing/malformed."""
    try:
        return _load_checklist_section_precise("Repo Commit Checklist")
    except (FileNotFoundError, ValueError):
        raise
    except Exception as e:
        raise FileNotFoundError(
            f"docs/CHECKLISTS.md not found or malformed: {e}"
        ) from e


# The triad prompt is assembled STABLE-FIRST for provider prompt caching:
# fixed instructions + checklist + governance docs form a byte-stable prefix
# reused across review rounds (marked with a cache breakpoint at dispatch),
# while goal/scope/files/diff/history are the per-commit dynamic tail.
_REVIEW_PROMPT_TEMPLATE_STABLE = """\
{preamble}

## Review instructions

Read the staged diff and the supplied post-change file context (both appear
AFTER the governance documents below). On very large changes, the fit note may
replace duplicated full-file snapshots with a path manifest; in that case the
complete added/deleted lines remain in the staged diff. Review every checklist
item, report every distinct current problem, and make every FAIL actionable
with file/symbol evidence and a concrete fix.

{critical_calibration}

{json_contract}

If an open obligation record below already names an `obligation_id` for this root cause,
reuse that exact `obligation_id`. Do NOT invent a new id when the same root cause persists.

## Anti pattern-lock guard

Run the shared semantic-breadth guard before returning:
{anti_pattern_lock_guard}

{checklist_section}

- Output ONLY a valid JSON array.  No markdown fences, no text outside the JSON.

## DEVELOPMENT.md

{dev_guide_text}

## ARCHITECTURE.md

{architecture_section}
"""

_REVIEW_PROMPT_TEMPLATE_DYNAMIC = """\
{goal_section}

{scope_section}

## Current touched files (full content)

{current_files_section}

## Staged diff

{diff_text}

## Changed files

{changed_files}

{rebuttal_section}{review_history_section}
"""


def _parse_review_json(raw: str) -> Optional[list]:
    """Best-effort extraction of a JSON array from model output."""
    return extract_json_array(raw, normalize=True)


def _git_show_staged(repo_dir, path: str) -> str:
    """Return staged index content via ``git show :PATH`` or ``""``."""
    import subprocess
    try:
        result = subprocess.run(
            ["git", "show", f":{path}"],
            cwd=str(repo_dir),
            capture_output=True,
            text=True,
            timeout=10,
        )
        return result.stdout if result.returncode == 0 else ""
    except Exception:
        return ""


def _preflight_check(commit_message: str, staged_files: str,
                     repo_dir) -> Optional[str]:
    """Fast deterministic review preflight for common incomplete staged diffs."""
    import re
    import string as _string

    # Accept either name-status lines ("A  path") or plain filenames.
    raw_lines = staged_files.strip().splitlines()
    file_status: list[tuple[str, str]] = []  # (status_char, filepath)
    for raw in raw_lines:
        raw = raw.strip()
        if not raw:
            continue
        # Name-status format: "X  path".
        if (len(raw) >= 4
                and raw[0] in _string.ascii_uppercase
                and raw[1:3] == "  "):
            status = raw[0].upper()
            path = raw[3:].strip()
            # Renames display as "R  old -> new".
            if " -> " in path:
                path = path.split(" -> ")[-1].strip()
            file_status.append((status, path))
        else:
            # Plain filenames are treated as modified.
            file_status.append(("M", raw))

    # active_staged excludes deletions for companion-file checks.
    staged_set = {path for _, path in file_status}
    active_staged = {path for status, path in file_status if status != "D"}
    # Added/Copied count as new modules; renames do not.
    new_files = {path for status, path in file_status if status in ("A", "C")}
    has_version_ref = bool(re.search(r'v?\d+\.\d+\.\d+', commit_message)) or "version" in commit_message.lower()
    version_staged = "VERSION" in active_staged

    missing = []

    # VERSION staged but README missing.
    if version_staged and "README.md" not in active_staged:
        missing.append("README.md (badge + changelog)")

    # Commit message references version but VERSION is not staged.
    if has_version_ref and not version_staged:
        if any(f.endswith(('.py', '.md')) and f != 'VERSION' for f in active_staged):
            missing.append("VERSION")

    if missing:
        return (
            f"⚠️ PREFLIGHT_BLOCKED: Staged diff is incomplete — fix before review.\n"
            f"  Missing from staged: {', '.join(missing)}\n"
            f"  Currently staged: {', '.join(sorted(staged_set)) or '(none)'}\n\n"
            "Stage all related files together. Use write_file for all files first,\n"
            "then commit_reviewed to stage and commit everything in one diff."
        )

    # Python logic touched without active tests staged.
    _LOGIC_DIRS = ("ouroboros/", "supervisor/")
    logic_changed = any(
        f.startswith(_LOGIC_DIRS) and f.endswith(".py")
        for f in staged_set  # all statuses including D
    )
    tests_staged = any(f.startswith("tests/") for f in active_staged)
    if logic_changed and not tests_staged:
        return (
            "⚠️ PREFLIGHT_BLOCKED: Python logic changed in ouroboros/ or supervisor/ "
            "but no tests/ files are staged.\n"
            "  Add or update tests to cover the changed behaviour, then re-stage.\n"
            "  If this is a docs/config-only change that triggered a false positive, "
            "check that no .py files from ouroboros/ or supervisor/ are in your staged set.\n"
            f"  Currently staged: {', '.join(sorted(staged_set)) or '(none)'}"
        )

    # New logic modules require active ARCHITECTURE.md update.
    new_logic_files = [
        f for f in new_files
        if f.startswith(_LOGIC_DIRS) and f.endswith(".py")
    ]
    if new_logic_files and "docs/ARCHITECTURE.md" not in active_staged:
        return (
            "⚠️ PREFLIGHT_BLOCKED: New files added in ouroboros/ or supervisor/ "
            "but docs/ARCHITECTURE.md is not staged.\n"
            "  New structural additions must be documented in ARCHITECTURE.md "
            "(Bible P6: authenticity / architectural mirror).\n"
            f"  New files: {new_logic_files[:5]}\n"
            f"  Currently staged: {', '.join(sorted(staged_set)) or '(none)'}"
        )

    # VERSION changes must keep staged version carriers synchronized.
    if version_staged:
        try:
            from ouroboros.tools.release_sync import (
                is_release_version,
                version_carrier_desyncs,
            )
            version_str = _git_show_staged(repo_dir, "VERSION").strip()
            if is_release_version(version_str):
                desync = version_carrier_desyncs(
                    version_str,
                    pyproject_text=_git_show_staged(repo_dir, "pyproject.toml"),
                    uv_lock_text=_git_show_staged(repo_dir, "uv.lock"),
                    web_package_text=_git_show_staged(repo_dir, "web/package.json"),
                    readme_text=_git_show_staged(repo_dir, "README.md"),
                    arch_text=_git_show_staged(repo_dir, "docs/ARCHITECTURE.md"),
                    api_types_text=_git_show_staged(repo_dir, "web/modules/api_types.js"),
                    download_readme_text=_git_show_staged(repo_dir, "README.md"),
                    site_install_text=_git_show_staged(repo_dir, "site/install/index.html"),
                    docs_install_text=_git_show_staged(repo_dir, "docs/install/index.html"),
                    detailed=True,
                )
                if desync:
                    return (
                        f"⚠️ PREFLIGHT_BLOCKED: VERSION file says {version_str} but "
                        "the following staged files have a different version value:\n"
                        + "".join(f"  - {d}\n" for d in desync)
                        + "Update all version references to match VERSION before committing.\n"
                        f"  Currently staged: {', '.join(sorted(staged_set)) or '(none)'}"
                    )
        except Exception:
            pass  # Non-fatal: LLM reviewers handle version sync

    # VERSION changes need a staged README changelog row, and the staged README
    # must respect P9 history limits.
    if version_staged:
        try:
            from ouroboros.tools.release_sync import is_release_version
            version_str = _git_show_staged(repo_dir, "VERSION").strip()
            if is_release_version(version_str):
                readme_text = _git_show_staged(repo_dir, "README.md")
                if readme_text and not re.search(r'\|\s*' + re.escape(version_str) + r'\s*\|', readme_text):
                    return (
                        f"⚠️ PREFLIGHT_BLOCKED: VERSION is {version_str} but README.md "
                        "changelog has no table row for this version.\n"
                        "  Add a changelog entry in the Version History table in README.md.\n"
                        f"  Currently staged: {', '.join(sorted(staged_set)) or '(none)'}"
                    )
        except Exception:
            pass  # Non-fatal
        try:
            readme_staged = _git_show_staged(repo_dir, "README.md")
            if readme_staged:
                from ouroboros.tools.release_sync import check_history_limit
                limit_warnings = check_history_limit(readme_staged)
                if limit_warnings:
                    return (
                        "⚠️ PREFLIGHT_BLOCKED: README.md Version History exceeds BIBLE.md P9 limits.\n"
                        + "".join(f"  - {w}\n" for w in limit_warnings)
                        + "  Trim the oldest entry in the over-limit category before committing.\n"
                        + "  Quick check: python -c \"from ouroboros.tools.release_sync import "
                        "check_history_limit; print(check_history_limit(open('README.md').read()))\"\n"
                        + f"  Currently staged: {', '.join(sorted(staged_set)) or '(none)'}"
                    )
        except Exception:
            pass  # Non-fatal: LLM reviewers handle P9 limits as advisory fallback

    # conftest.py must not contain collectable module-level tests.
    conftest_files = [f for f in active_staged if pathlib.Path(f).name == "conftest.py"]
    if conftest_files:
        import ast as _ast
        for cf in conftest_files:
            try:
                cf_text = _git_show_staged(repo_dir, cf)
                if not cf_text:
                    continue
                tree = _ast.parse(cf_text, filename=cf)
                # Nested helpers inside fixtures are not pytest-collected.
                test_fns = [
                    node.name for node in tree.body
                    if isinstance(node, (_ast.FunctionDef, _ast.AsyncFunctionDef))
                    and node.name.startswith("test_")
                ]
                if test_fns:
                    shown = test_fns[:5]
                    omission = f" (⚠️ showing first 5 of {len(test_fns)})" if len(test_fns) > 5 else ""
                    return (
                        f"⚠️ PREFLIGHT_BLOCKED: {cf} contains test functions: "
                        f"{shown}{omission}.\n"
                        "  conftest.py is for fixtures/hooks only. Move test_ functions "
                        "to a test_*.py file so pytest can discover them properly.\n"
                        f"  Currently staged: {', '.join(sorted(staged_set)) or '(none)'}"
                    )
            except Exception:
                pass  # Non-fatal: AST parse failure or git error, skip this file

    return None


def _review_entry(
    *,
    severity: str,
    item: str,
    reason: str,
    model: str = "",
    tag: str = "triad",
    verdict: str = "FAIL",
    obligation_id: str = "",
) -> dict:
    entry = {
        "severity": severity,
        "item": item,
        "reason": reason,
        "tag": tag,
        "verdict": verdict,
    }
    if model:
        entry["model"] = model
    if obligation_id:
        entry["obligation_id"] = obligation_id
    return entry


def _append_review_warning(ctx: ToolContext, text: Any) -> None:
    if isinstance(text, dict):
        ctx._review_advisory.append(text)
        return
    warning = _single_line(str(text))
    if warning:
        ctx._review_advisory.append(warning)


def _handle_review_block_or_warning(
    ctx: ToolContext,
    blocking_review: bool,
    blocked_msg: str,
    advisory_prefix: str,
) -> Optional[str]:
    """Either block immediately or downgrade to advisory warning."""
    if blocking_review:
        return blocked_msg
    _record_advisory_override(ctx, blocked_msg)
    _append_review_warning(ctx, advisory_prefix + blocked_msg)
    ctx._review_iteration_count = 0
    ctx._review_history = []
    return None


def _record_advisory_override(ctx: ToolContext, blocked_msg: str) -> None:
    """Durable trace of a blocking signal waved through by advisory enforcement.

    Constitutional requirement (BIBLE P3 "Owner-chosen enforcement, loud
    advisory"): every decision blocking enforcement would have stopped must
    leave a durable, owner-visible trace. Persisted to events.jsonl AND to a
    persistent counter file surfaced by the review_status tool.
    """
    reason = str(getattr(ctx, "_last_review_block_reason", "") or "unknown")
    try:
        append_jsonl(ctx.drive_logs() / "events.jsonl", {
            "ts": utc_now_iso(),
            "type": "review_advisory_override",
            "block_reason": reason,
            "message_head": str(blocked_msg or "")[:600],
            "task_id": str(getattr(ctx, "task_id", "") or ""),
        })
    except Exception:
        log.debug("Failed to emit review_advisory_override event", exc_info=True)
    try:
        from ouroboros.utils import update_json_locked

        path = ctx.drive_root / "state" / "advisory_overrides.json"

        def _bump(current: dict) -> dict:
            recent = list(current.get("recent") or [])
            recent.append({
                "ts": utc_now_iso(),
                "block_reason": reason,
                "message_head": str(blocked_msg or "")[:300],
            })
            return {
                "count": int(current.get("count") or 0) + 1,
                "recent": recent[-10:],
            }

        update_json_locked(path, _bump)
    except Exception:
        log.warning("Failed to persist advisory override visibility", exc_info=True)


def _collect_review_findings(ctx: ToolContext, model_results: list) -> tuple[list[str], list[str], list[str], list[dict]]:
    parsed = parse_model_review_results({"results": model_results})
    critical_fails: List[str] = []
    advisory_warns: List[str] = []
    structured_critical: List[dict] = []
    structured_advisory: List[dict] = []
    triad_raw_results = [record.to_dict() for record in parsed.actor_records]
    errored_models = [record.model_id for record in parsed.actor_records if record.status == "error"]

    for record in parsed.actor_records:
        if record.status == "error":
            advisory_warns.append(
                f"[{record.model_id}] Model unavailable this round (transport error). "
                "Full raw response preserved in triad_raw_results (status='error')."
            )
            structured_advisory.append(_review_entry(
                severity="advisory",
                item="review_model_unavailable",
                reason=(
                    f"Model unavailable this round (transport error): {record.model_id}. "
                    "Full raw response preserved in triad_raw_results actor record."
                ),
                model=record.model_id,
            ))
            try:
                append_jsonl(ctx.drive_logs() / "events.jsonl", {
                    "ts": utc_now_iso(),
                    "type": "review_model_error",
                    "model": record.model_id,
                    "error_note": "Full raw response preserved in triad_raw_results.",
                })
            except Exception:
                pass
            continue
        if record.status == "parse_failure":
            advisory_warns.append(
                f"[{record.model_id}] Could not parse structured review output (parse_failure). "
                "Full raw response preserved in triad_raw_results (status='parse_failure')."
            )
            structured_advisory.append(_review_entry(
                severity="advisory",
                item="review_model_parse_failure",
                reason=(
                    f"Could not parse structured review output from {record.model_id}. "
                    "Full raw response preserved in triad_raw_results actor record."
                ),
                model=record.model_id,
            ))
            continue
        for item in record.parsed_items:
            if str(item.get("verdict", "")).upper() != "FAIL":
                continue
            desc = f"[{record.model_id}] {item.get('item', '?')}: {item.get('reason', '')}"
            target = structured_critical if item.get("severity") == "critical" else structured_advisory
            target.append(_review_entry(
                severity="critical" if target is structured_critical else "advisory",
                item=str(item.get("item", "?")),
                reason=str(item.get("reason", "")),
                model=record.model_id,
                obligation_id=str(item.get("obligation_id", "") or ""),
            ))
            (critical_fails if target is structured_critical else advisory_warns).append(desc)

    ctx._last_review_critical_findings = structured_critical
    ctx._last_review_advisory_findings = structured_advisory
    ctx._last_triad_raw_results = triad_raw_results
    if parsed.degraded_reasons:
        if not hasattr(ctx, "_review_degraded_reasons"):
            ctx._review_degraded_reasons = []
        ctx._review_degraded_reasons.extend(parsed.degraded_reasons)
    return critical_fails, advisory_warns, errored_models, triad_raw_results


def _build_critical_block_message(
    ctx: ToolContext,
    commit_message: str,
    critical_fails: List[str],
    advisory_warns: List[str],
    errored_note: str,
) -> str:
    critical_entries = list(getattr(ctx, "_last_review_critical_findings", []) or critical_fails)
    advisory_entries = list(getattr(ctx, "_last_review_advisory_findings", []) or advisory_warns)
    ctx._review_history.append({
        "attempt": ctx._review_iteration_count,
        "commit_message": commit_message,  # full — no [:200] truncation
        "critical": critical_entries,
        "advisory": advisory_entries,
    })

    iteration_note = f" (attempt {ctx._review_iteration_count})"

    retry_coaching = build_self_verification_template(
        critical_entries,
        attempt_idx=ctx._review_iteration_count,
        tool_name="commit_reviewed",
        context_noun="diff",
    )

    return (
        f"⚠️ REVIEW_BLOCKED{iteration_note}: Critical issues found by reviewers.\n"
        "Commit has NOT been created. Fix the issues and try again. Use review_rebuttal\n"
        "ONLY if a finding is factually incorrect — not to argue against requested tests\n"
        "or artifacts. If the same finding repeats after a rebuttal, implement the fix\n"
        "instead of re-arguing.\n\n"
        + "Critical findings:\n"
        + "\n".join(f"  - {_format_review_entry(f, default_severity='critical')}" for f in critical_entries)
        + (
            "\n\nAdvisory warnings:\n"
            + "\n".join(f"  - {_format_review_entry(w)}" for w in advisory_entries)
            if advisory_entries else ""
        )
        + errored_note
        + retry_coaching
    )


def _build_preflight_staged(target_repo: str, fallback: str = "") -> str:
    """Convert git name-status to the compact preflight format."""
    try:
        name_status = run_cmd(
            ["git", "diff", "--cached", "--name-status"], cwd=target_repo
        )
        return format_name_status_for_preflight(name_status, fallback=fallback)
    except Exception:
        return fallback  # check 4 may not fire, but checks 1-3 still work


def _fit_triad_prompt(api_models: list, assemble, current_files_section: str,
                      diff_text: str, changed: str, target_repo) -> tuple:
    """The api pack's guaranteed-fit ladder (P3 one-pass): drop only evidence
    duplicated by the complete staged diff — full snapshots first, then unchanged
    diff context. Each api slot's limit uses its REAL window from Capability
    Evidence (a hardcoded 1M treated a 200K reviewer as 1M-capable and lost its
    whole review to a deterministic prompt-too-long 400), with sub-1M windows
    scaling their reserves so a small-window slot gets a fit-sized pack, not a
    zero limit; the shared prompt is sized to the review QUORUM — the same SSOT
    plan review uses — so one small slot degrades its OWN seat rather than
    blocking the gate for the whole panel. Session rows are not constrained by
    this pack at all (5.2/5.7): they retrieve with their own tools. Returns
    ``(prompt, stable_prefix_len, block_message_or_empty)``."""
    def _slot_input_limit(slot_model: str) -> int:
        window = reviewer_context_window(slot_model)
        output_reserve, tokenizer_margin = window_scaled_reserves(
            window,
            output_reserve=_review_output_budget(),
            tokenizer_margin=50_000,
        )
        return max(0, calibrated_input_token_limit(
            slot_model,
            context_window=window,
            output_reserve=output_reserve,
            tokenizer_margin=tokenizer_margin,
            budget_cap=REVIEW_PROMPT_TOKEN_BUDGET,
        ))

    input_limit = _quorum_input_token_limit(
        api_models, {m: _slot_input_limit(m) for m in api_models})
    prompt, stable_prefix_len = assemble(current_files_section, diff_text)
    if input_limit and estimate_tokens(prompt) > input_limit:
        touched_paths = [line.strip() for line in changed.splitlines() if line.strip()]
        fit_note = (
            "TRIAD FIT NOTE: Full post-change snapshots were omitted because they "
            "duplicate the complete staged diff and would exceed the strictest "
            "configured reviewer's input limit. Every touched path is listed below; "
            "all added/deleted lines remain in the staged diff.\n\n"
            + ("\n".join(f"- {path}" for path in touched_paths) or "(no paths reported)")
        )
        prompt, stable_prefix_len = assemble(fit_note, diff_text)
        if input_limit and estimate_tokens(prompt) > input_limit:
            from ouroboros.tools.review_binary_context import (
                StagedDiffUnavailable, capture_staged_diff)
            try:  # the SAME hardened capture as the primary diff, at zero context
                compact_diff = capture_staged_diff(target_repo, unified=0)
            except StagedDiffUnavailable:
                compact_diff = ""  # keep the hardened full diff; the gate below blocks if it still overflows
            if compact_diff.strip():
                prompt, stable_prefix_len = assemble(fit_note, compact_diff)
    prompt_tokens = estimate_tokens(prompt)
    if not input_limit or prompt_tokens > input_limit:
        return prompt, stable_prefix_len, (
            "⚠️ REVIEW_BLOCKED: The irreducible one-pass triad prompt does not "
            f"fit every configured reviewer ({prompt_tokens:,} estimated input "
            f"tokens; limit {input_limit:,}). Split or shrink the staged change; "
            "reviewer models and evidence authority were not degraded."
        )
    return prompt, stable_prefix_len, ""


def _triad_session_task(ctx: ToolContext, *, goal_section: str, scope_section: str,
                        checklist_section: str, rebuttal_section: str,
                        review_history_section: str, dev_guide_text: str,
                        architecture_text: str) -> str:
    """The commit-triad task in SESSION delivery (5.2/5.3): the SAME preamble,
    calibration, checklist and goal/scope/history the api pack carries — but no
    assembled evidence. The subject is a pointer (the session takes the staged
    diff itself) and the governance docs arrive as navigation maps (5.7)."""
    from ouroboros.context_layout import generate_doc_nav_map

    nav_maps = [
        generate_doc_nav_map(text, title=title, rel_path=rel)
        for title, rel, text in (
            ("DEVELOPMENT.md", "docs/DEVELOPMENT.md", dev_guide_text),
            ("ARCHITECTURE.md", "docs/ARCHITECTURE.md", architecture_text),
        )
        if str(text or "").strip()
    ]
    return "\n\n".join(part for part in [
        REVIEW_PREAMBLE,
        CRITICAL_FINDING_CALIBRATION,
        REPO_ANTI_PATTERN_LOCK_GUARD,
        checklist_section,
        goal_section,
        scope_section,
        rebuttal_section,
        review_history_section,
        "## Subject (session delivery)\n"
        "The review subject is the STAGED diff of the repository you are running "
        "in. Retrieve it yourself with whatever your read-only tools allow: if you "
        "can run commands, `git diff --cached` (and `git diff --cached --name-only` "
        "for the file list); if your read-only mode withholds command execution — it "
        "commonly does — read the touched files directly and compare them against "
        "`.git`. Read the touched files as needed either way.",
        "## Governance context (navigation maps)\n"
        "Read BIBLE.md in full from the repository root. The maps below index "
        "the other governance docs by line range; the paths are relative to the "
        "repository root — read the sections you need with your own tools.",
        *nav_maps,
    ] if str(part or "").strip())


def _capture_triad_staged_diff(
    ctx: ToolContext, target_repo, blocking_review: bool
) -> tuple[Optional[str], Optional[str]]:
    """Capture the triad's staged-diff evidence, or route a capture failure.

    Returns ``(diff_text, None)`` on success and ``(None, block_result)`` on
    failure — the fail-closed message in blocking mode, ``None`` (advisory skip)
    otherwise. The diff is the triad's primary change evidence, so it is taken
    byte-exact and hardened against operator diff config (the same
    ``capture_staged_diff`` the scope reviewer uses); a genuine failure fails
    closed rather than reviewing a placeholder that would yield authoritative
    findings about a diff nobody has.
    """
    from ouroboros.tools.review_binary_context import (
        StagedDiffUnavailable, capture_staged_diff)

    try:
        return capture_staged_diff(target_repo), None
    except StagedDiffUnavailable as exc:
        ctx._last_review_block_reason = "infra_failure"
        return None, _handle_review_block_or_warning(
            ctx, blocking_review,
            "⚠️ REVIEW_BLOCKED: Cannot capture the staged diff — commit cannot "
            f"proceed.\nError: {exc}\n"
            "Ensure git is available and the repository is in a valid state.",
            "Review enforcement=Advisory: staged diff capture failed; triad "
            "review skipped rather than run against a placeholder. ",
        )


def _run_unified_review(ctx: ToolContext, commit_message: str,
                        review_rebuttal: str = "",
                        repo_dir=None,
                        goal: str = "",
                        scope: str = "") -> Optional[str]:
    """Run triad pre-commit review; return a block message or ``None``."""
    target_repo = repo_dir or ctx.repo_dir
    ctx._review_iteration_count += 1
    ctx._last_review_block_reason = ""  # reset per attempt
    ctx._last_triad_models = []  # reset forensic field so stale values never persist on early exit
    ctx._last_review_critical_findings = []  # reset to avoid stale findings from previous attempts
    ctx._last_triad_raw_results = []  # reset per-model actor records
    ctx._review_degraded_reasons = []  # reset degraded participation markers
    review_enforcement = _cfg.get_review_enforcement()
    blocking_review = review_enforcement == "blocking"

    diff_text, capture_block = _capture_triad_staged_diff(ctx, target_repo, blocking_review)
    if diff_text is None:  # capture failed: block (blocking) or advisory-skip (None)
        return capture_block
    if not diff_text.strip():
        return None

    try:
        changed = run_cmd(["git", "diff", "--cached", "--name-only"], cwd=target_repo)
    except Exception:
        changed = ""

    preflight_staged = _build_preflight_staged(target_repo, fallback=changed)

    preflight_err = _preflight_check(commit_message, preflight_staged, target_repo)
    if preflight_err:
        ctx._last_review_block_reason = "preflight"
        result = _handle_review_block_or_warning(
            ctx, blocking_review, preflight_err,
            "Review enforcement=Advisory: preflight warning did not block commit. ",
        )
        if result is not None:
            return result

    rebuttal_section = build_rebuttal_section(review_rebuttal)

    try:
        checklist_section = _load_checklist_section()
    except (FileNotFoundError, ValueError) as e:
        log.error("Checklist loading failed (fail-closed): %s", e)
        ctx._last_review_block_reason = "infra_failure"
        blocked_msg = (
            "⚠️ REVIEW_BLOCKED: Cannot load review checklist — commit cannot proceed.\n"
            f"Error: {e}\n"
            "Ensure docs/CHECKLISTS.md exists and contains the expected section headers."
        )
        return _handle_review_block_or_warning(
            ctx, blocking_review, blocked_msg,
            "Review enforcement=Advisory: review checklist failed to load; commit proceeding anyway. ",
        )

    dev_guide_text = load_governance_doc(pathlib.Path(ctx.repo_dir), "docs/DEVELOPMENT.md", on_missing="explicit")
    architecture_text = load_governance_doc(pathlib.Path(ctx.repo_dir), "docs/ARCHITECTURE.md", on_missing="explicit")

    # Durable open obligations reduce review thrashing across restarts.
    _open_obs_for_review = []
    try:
        from ouroboros.review_state import load_state, make_repo_key
        _rs = load_state(pathlib.Path(ctx.drive_root))
        _repo_key = make_repo_key(pathlib.Path(ctx.repo_dir))
        _open_obs_for_review = _rs.get_open_obligations(repo_key=_repo_key)
    except Exception:
        pass  # Non-fatal: anti-thrashing hint is best-effort
    review_history_section = _build_review_history_section(
        ctx._review_history, open_obligations=_open_obs_for_review,
    )

    # Build touched-file pack for full current context.
    try:
        touched_paths = [f.strip() for f in changed.strip().splitlines() if f.strip()]
        from ouroboros.tools.registry import _authorized_managed_update_resolver

        current_files_section, _omitted = build_touched_file_pack(
            pathlib.Path(target_repo),
            touched_paths,
            represent_binary=_authorized_managed_update_resolver(ctx),
        )
        if _omitted:
            current_files_section += (
                f"\n\n⚠️ OMISSION NOTE: {len(_omitted)} file(s) omitted from direct context: "
                f"{', '.join(_omitted)}"
            )
        if not current_files_section.strip():
            current_files_section = "(no touched files could be read)"
    except Exception as e:
        log.warning("Failed to build touched file pack for triad review: %s", e)
        current_files_section = f"(touched file pack unavailable: {e})"

    # Per-row identity/delivery/strength from the ONE reviewer-slot SSOT (6.1):
    # structured rows when configured, the migrated comma-lists otherwise. A
    # malformed configuration is an infra failure, never a silent api spend.
    from ouroboros.review_execution import ReviewRouteKind
    from ouroboros.reviewer_slot_config import commit_triad_delivery

    try:
        row_plan = commit_triad_delivery()
    except ValueError as exc:
        ctx._last_review_block_reason = "infra_failure"
        return _handle_review_block_or_warning(
            ctx, blocking_review,
            f"⚠️ REVIEW_BLOCKED: invalid reviewer-slot configuration — {exc}",
            "Review enforcement=Advisory: invalid reviewer-slot configuration did not block commit. ",
        )
    models, row_routes = row_plan["models"], row_plan["routes"]
    ctx._last_triad_models = list(models)  # forensic: actual resolved model IDs
    api_models = [m for m, r in zip(models, row_routes) if r is ReviewRouteKind.API_CHAT]

    goal_section = build_goal_section(goal, scope, commit_message)
    scope_section = build_scope_section(scope)

    def _assemble_prompt(files_section: str, staged_diff: str) -> tuple:
        """Return (prompt, stable_prefix_len): the stable governance prefix is
        byte-identical across rounds and becomes the cache-marked block."""
        stable = _REVIEW_PROMPT_TEMPLATE_STABLE.format(
            preamble=REVIEW_PREAMBLE,
            critical_calibration=CRITICAL_FINDING_CALIBRATION,
            json_contract=REVIEW_JSON_ARRAY_CONTRACT,
            anti_pattern_lock_guard=REPO_ANTI_PATTERN_LOCK_GUARD,
            checklist_section=checklist_section,
            dev_guide_text=dev_guide_text or "(DEVELOPMENT.md not found)",
            architecture_section=architecture_text or "(ARCHITECTURE.md not found)",
        )
        dynamic = _REVIEW_PROMPT_TEMPLATE_DYNAMIC.format(
            goal_section=goal_section,
            scope_section=scope_section,
            current_files_section=files_section,
            rebuttal_section=rebuttal_section,
            review_history_section=review_history_section,
            diff_text=staged_diff,
            changed_files=changed,
        )
        return stable + "\n" + dynamic, len(stable) + 1

    # P3 stays one-pass. The api pack, its fit ladder and the fixed_overflow
    # gate exist ONLY for the api rows (5.2/5.7): a session row retrieves with
    # its own tools, so it neither constrains the fit limit nor is blocked by
    # it, and a panel with no api rows skips pack assembly entirely.
    prompt, stable_prefix_len = "", 0
    if api_models:
        prompt, stable_prefix_len, fit_error = _fit_triad_prompt(
            api_models, _assemble_prompt, current_files_section, diff_text,
            changed, target_repo,
        )
        if fit_error:
            ctx._last_review_block_reason = "fixed_overflow"
            return fit_error

    session_task = ""
    if len(api_models) < len(models):
        session_task = _triad_session_task(
            ctx,
            goal_section=goal_section,
            scope_section=scope_section,
            checklist_section=checklist_section,
            rebuttal_section=rebuttal_section,
            review_history_section=review_history_section,
            dev_guide_text=dev_guide_text,
            architecture_text=architecture_text,
        )

    try:
        result_json = _handle_multi_model_review(
            ctx,
            content="Review the staged diff and context provided in the instructions above.",
            prompt=prompt,
            models=models,
            stable_prefix_len=stable_prefix_len,
            routes=row_routes,
            session_task=session_task,
            session_root=str(target_repo),
            row_plan=row_plan,
        )
        result = json.loads(result_json)
    except Exception as e:
        log.error("Unified review infrastructure failure: %s", e)
        ctx._last_review_block_reason = "infra_failure"
        blocked_msg = (
            "⚠️ REVIEW_BLOCKED: Review infrastructure failed — commit cannot proceed "
            "without a successful review.\n"
            f"Error: {e}\n"
            "Check OPENROUTER_API_KEY, network connectivity, and retry."
        )
        return _handle_review_block_or_warning(
            ctx, blocking_review, blocked_msg,
            "Review enforcement=Advisory: review infrastructure failure did not block commit. ",
        )

    if "error" in result:
        log.error("Review returned error: %s", result["error"])
        ctx._last_review_block_reason = "infra_failure"
        blocked_msg = (
            "⚠️ REVIEW_BLOCKED: Review service returned an error — commit cannot proceed "
            "without a successful review.\n"
            f"Error: {result['error']}\n"
            "Check OPENROUTER_API_KEY, network connectivity, and retry."
        )
        return _handle_review_block_or_warning(
            ctx, blocking_review, blocked_msg,
            "Review enforcement=Advisory: review service error did not block commit. ",
        )

    model_results = result.get("results", [])
    if not model_results:
        ctx._last_review_block_reason = "infra_failure"
        blocked_msg = (
            "⚠️ REVIEW_BLOCKED: Review returned no results from any model — "
            "commit cannot proceed without a successful review."
        )
        return _handle_review_block_or_warning(
            ctx, blocking_review, blocked_msg,
            "Review enforcement=Advisory: review returned no model results; commit proceeding anyway. ",
        )

    critical_fails, advisory_warns, errored_models, _triad_raw = _collect_review_findings(ctx, model_results)
    models_total = len(model_results)

    # Quorum counts only parseable responded actors, not errors/parse failures.
    triad_raw = getattr(ctx, "_last_triad_raw_results", []) or []
    successful_reviewers = sum(1 for r in triad_raw if r.get("status") == "responded")
    # Non-successful actors are shown for transport/parse diagnostics.
    failed_actors = [
        r["model_id"] for r in triad_raw if r.get("status") != "responded"
    ]
    required_quorum = _cfg.adaptive_quorum(models_total)
    if successful_reviewers < required_quorum:
        ctx._last_review_block_reason = "review_quorum"
        unavailable_str = ", ".join(failed_actors) if failed_actors else ", ".join(errored_models)
        blocked_msg = (
            f"⚠️ REVIEW_BLOCKED: Only {successful_reviewers} of {models_total} review "
            f"models responded successfully (minimum {required_quorum} required). "
            f"Unavailable/failed: {unavailable_str}.\n"
            "Retry the commit — transient model failures usually resolve quickly."
        )
        return _handle_review_block_or_warning(
            ctx, blocking_review, blocked_msg,
            "Review enforcement=Advisory: review quorum failure did not block commit. ",
        )

    if models_total < 2:
        # A single configured reviewer is honored (owner's explicit setup), but
        # the lost cross-model diversity is recorded LOUDLY (Bible P3): the immune
        # gate ran with no second opinion. Record it on the DURABLE degraded-reasons
        # channel (persisted into the commit review record by git_ops) so it
        # survives in review history/status, not just a transient log line.
        ctx._single_reviewer_no_diversity = True
        if not hasattr(ctx, "_review_degraded_reasons"):
            ctx._review_degraded_reasons = []
        if "single_reviewer_no_diversity" not in ctx._review_degraded_reasons:
            ctx._review_degraded_reasons.append("single_reviewer_no_diversity")
        log.warning("Commit review ran with a single reviewer (single_reviewer_no_diversity).")

    errored_note = ""
    all_non_responded = failed_actors or errored_models
    if all_non_responded:
        errored_note = (
            f"\n\nNote: {len(all_non_responded)} of {models_total} review models "
            f"were unavailable or failed to parse ({', '.join(all_non_responded)}). "
            f"Target is {models_total} working reviewers."
        )

    if critical_fails:
        # All parse issues get a parse_failure block reason.
        all_parse = all("Could not parse" in f for f in critical_fails)
        ctx._last_review_block_reason = "parse_failure" if all_parse else "critical_findings"
        if blocking_review:
            return _build_critical_block_message(
                ctx, commit_message, critical_fails, advisory_warns, errored_note,
            )

        _record_advisory_override(ctx, "; ".join(critical_fails[:5]))
        _append_review_warning(
            ctx,
            "Review enforcement=Advisory: critical review findings did not block commit.",
        )
        for finding in getattr(ctx, "_last_review_critical_findings", []) or []:
            _append_review_warning(ctx, finding)
        for warning in getattr(ctx, "_last_review_advisory_findings", []) or []:
            _append_review_warning(ctx, warning)
        if errored_note:
            _append_review_warning(ctx, errored_note)

    if not critical_fails:
        # All clear: reset iteration state. With critical findings present
        # (advisory enforcement), the anti-thrashing history must SURVIVE so
        # repeat findings on the next attempt are still recognized as repeats.
        ctx._review_iteration_count = 0
        ctx._review_history = []

    if errored_note or advisory_warns or getattr(ctx, "_last_review_advisory_findings", None):
        ctx._review_advisory = list(getattr(ctx, "_last_review_advisory_findings", []) or [])
        if errored_note:
            ctx._review_advisory.append(errored_note.strip())
    return None
