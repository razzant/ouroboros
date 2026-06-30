"""Review collection and complexity metrics."""

from __future__ import annotations

import pathlib
from typing import Any, Dict, List, Tuple

from ouroboros.tools.review_helpers import (
    iter_repo_pack_entries,
)


_HEALTH_SKIP_DIR_PREFIXES = (
    ".git/",
    ".pytest_cache/",
    ".mypy_cache/",
    "node_modules/",
    ".venv/",
    "devtools/",
    "tests/",
)
TARGET_MODULE_LINES = 1000
MAX_MODULE_LINES = 1600
TARGET_FUNCTION_LINES = 150
# Advisory SDK orchestration stays single-flow; split tracked as tech debt.
MAX_FUNCTION_LINES = 300
# Deterministic anti-bloat brake (BIBLE P3 "codebase size" gate, P7 minimalism):
# tests/test_smoke.py::test_function_count_reasonable enforces this in CI and in
# the hermetic self-commit preflight. Owner decision 2026-06-10: first paydown
# in gate history (consolidation pass removed ~60 dead/duplicate/trivial-wrapper
# functions) plus headroom to 3500 so routine fixes stop churning this constant.
# v6.45.1: the 4-TZ mega-sprint plus managed #53 added reviewed GAIA/vision/
# benchmark/evolution helper surfaces; accepted with explicit release-review debt.
# The convention stands: growth must be acknowledged — bump deliberately with a
# one-line justification here. Value archaeology lives in git history of this line.
# v6.46.0: GAIA-forensic hardening added reviewed single-purpose helpers (fail_tasks,
# _deliverables_root, _resolve_or_provider, the generative context-window probe, the
# GAIA adapter knobs); bump with small headroom so the release's tests do not re-churn.
# v6.47.0: the verify-before-done flagship (verify_and_record + receipt-store/grounding/
# flag/nudge helpers in outcomes.py + loop.py), FR2 cooperative-subagent helpers, FR1
# skill-publish eligibility predicate, and the query_code/shell/control extracted helpers.
# v6.50.0: reconciliation-layer helpers (typed delegation constraints, schedule-time
# capability reconciliation, child-absorption outcome shelf) plus SWE-Pro adapter
# hardening tests. Small headroom; split/paydown remains tracked in DEVELOPMENT.md.
# scope-review false-1M fix: +4 plus headroom (3575 -> 3582) — reviewed
# single-purpose helpers in tools/scope_review.py: _is_designated_default_reviewer +
# its nested _normalized, _provider_error_is_oversize, and
# _scope_oversize_advisory_result (the last DEDUPES the two oversize→advisory
# branches, keeping run_scope_review under the per-function line gate).
# v6.50.3: +5 TB/swarm capability helpers — A1 verify._expected_matches (exact/
# structured match), A3 loop._contract_expected_output (no-op nudge gate), B1
# agent_task_pipeline._build_swarm_efficiency (delegation rollup), B2
# control._count_live_sibling_children (burst/absorb advisory), A2
# outcomes._is_ignored_readonly_block (SSOT predicate shared by the execution axis
# and the verification ledger). Cap 3585 = current count 3584 + 1 slot headroom
# (rebased on #57's scope-review fix, which already moved the cap to 3582).
# v6.51.0: +11 net = shell_parse.recover_stringified_argv + normalize_check_argv (verify
# argv/PATH SSOT, idea-1), outcomes.latest_unreconciled_failed_receipt + ...verification
# (red-finalize predicate, idea-3), and the review_evidence process-aware acceptance packet
# (build_task_acceptance_evidence + 6 bounded/redacted/leak-safe helpers, idea-2), minus
# verify._normalize_check (now an alias to the shell_parse SSOT). The 6 _accept_* helpers
# keep build_task_acceptance_evidence under the 150-line method gate. +1 (review round-1):
# the _accept_enforce_budget `_size()` closure (disclosed-truncation ladder, leak-safety fix).
# Cap 3597 = 3585 + 12, no extra headroom (acknowledged growth, per the gate's purpose).
# v6.52.0 +15 = 3612: media.py ocr_pdf/youtube_transcript + 4 helpers (P4b); llm._is_deferrable_image_user_turn
# (P4a ordering); verify._confine_artifact_path + _probe_artifact_lifecycle (C); artifacts.stage_task_attachments
# + _safe_attachment_name + context._build_attachment_image_blocks + gateway._render_attachment_lines
# + ws._chat_attachment_uploads (P1 attachment substrate + full desktop unify).
# v6.52.2 +8 = 3620: ephemeral-scratch + exit-masking integrity. shell._resolve_scratch_abs +
# shell._scratch_safety_reason + shell._record_scratch_fingerprints (Fix #1 scratch guard +
# every-exit-path fingerprint recording); artifacts.record_task_scratch +
# artifacts.read_task_scratch_fingerprints (fingerprinted scratch manifest); verify._check_has_exit_masking
# (Fix #2 sensor); outcomes.latest_unreconciled_masked_pass + outcomes.latest_unreconciled_masked_verification.
# evolution-stop authoritative fix: +2 functions -> new count 3622; cap set to 3624 (+2 headroom).
# evolution_lifecycle.complete_evolution_campaign (terminal owner-stop, distinct from the resumable pause) +
# post_task_evolution.drop_pending_request (clear a queued promotion at the owner-stop sites; the durable backstop
# is the evolution_owner_stopped flag read in apply_pending_request).
MAX_TOTAL_FUNCTIONS = 3624
GRANDFATHERED_OVERSIZED_FUNCTIONS = {
    ("agent_startup_checks.py", "verify_restart"),  # managed #53 boot diagnostic flow, 307 lines
    ("git.py", "_run_reviewed_stage_cycle"),  # reviewed-commit gate orchestration, 302 lines
    ("events.py", "_handle_schedule_task"),  # v6.50 admission reconciliation grew the existing scheduling choke point.
}
# Grandfathered modules are accepted debt until their surfaces stabilize/split.
GRANDFATHERED_OVERSIZED_MODULES = {
    "llm.py",
    "claude_advisory_review.py",
    "review_state.py",
    "server.py",
    "git.py",
    # Core extension loader (PluginAPI impl + registries + in/out-of-process load).
    # v6.15.0's OOP parity grew it from ~1573 to ~1777 lines, crossing the 1600
    # hard-fail for the first time. Splitting the registry-coupled PluginAPIImpl/loader
    # is a tracked follow-up (avoid cross-module private-registry access); accepted
    # debt until then.
    "extension_loader.py",
    # v6.20.0 acting (mutative) subagents added the acting authority/gating to the
    # tool dispatcher and the supervisor schedule handler. Both modules were ~1591
    # lines (just under the gate) and crossed 1600 with the new gating; reducing
    # these safety-critical dispatch/event modules by extraction is higher-risk and
    # is tracked as accepted debt to pay down after the feature stabilizes.
    "registry.py",
    "events.py",
    "control.py",
    "workers.py",
    # v6.33.0 reliability work crossed three core modules that were at/near the
    # ceiling. loop.py (was 1523) gained deadline-aware finalization + intrinsic
    # pacing; the helpers are tightly coupled to loop internals (_forced_final_answer,
    # _RoundLimitContext, _emit_checkpoint_event), so a sibling extraction would
    # introduce import cycles. shell.py (was 1600) and core.py (was 1599) gained the
    # brace-group sh -c hint, single-file search_code, and the re-read awareness
    # nudge. The function-size gate also forces helper extraction that GROWS the
    # module, so squeezing under 1600 fights itself. Splitting these hot tool/loop
    # modules cleanly is tracked debt for a follow-up release.
    "loop.py",
    "shell.py",
    "core.py",
}
# Bundle-only launcher is not part of the self-editable function budget.
FUNCTION_COUNT_EXCLUDED_FILES = {"launcher.py"}


def compute_complexity_metrics(sections: List[Tuple[str, str]]) -> Dict[str, Any]:
    """Compute codebase complexity metrics from collected sections."""
    file_sizes: List[Tuple[str, int]] = []
    function_lengths: List[Tuple[str, int, int]] = []
    for path, content in sections:
        lines = content.splitlines()
        file_sizes.append((path, len(lines)))
        if not path.endswith(".py") or pathlib.Path(path).name in FUNCTION_COUNT_EXCLUDED_FILES:
            continue
        starts = [
            idx for idx, line in enumerate(lines)
            if line.strip().startswith(("def ", "async def "))
        ]
        for pos, start in enumerate(starts):
            def_indent = len(lines[start]) - len(lines[start].lstrip())
            next_start = starts[pos + 1] if pos + 1 < len(starts) else len(lines)
            end = next_start
            for idx in range(start + 1, next_start):
                stripped = lines[idx].strip()
                if not stripped or stripped.startswith("#"):
                    continue
                if len(lines[idx]) - len(lines[idx].lstrip()) <= def_indent:
                    end = idx
                    break
            function_lengths.append((path, start, end - start))

    total_lines = sum(size for _path, size in file_sizes)
    func_lens = [length for _, _, length in function_lengths]
    py_files = [item for item in file_sizes if item[0].endswith(".py")]
    target_drift_modules = [(p, n) for p, n in py_files if n > TARGET_MODULE_LINES]
    hard_modules = [(p, n) for p, n in py_files if n > MAX_MODULE_LINES]

    return {
        "total_files": len(sections),
        "py_files": len(py_files),
        "total_lines": total_lines,
        "total_functions": len(function_lengths),
        "avg_function_length": round(sum(func_lens) / max(1, len(func_lens)), 1) if func_lens else 0,
        "max_function_length": max(func_lens) if func_lens else 0,
        "largest_files": sorted(file_sizes, key=lambda x: x[1], reverse=True)[:10],
        "longest_functions": sorted(function_lengths, key=lambda x: x[2], reverse=True)[:10],
        "target_drift_functions": [item for item in function_lengths if item[2] > TARGET_FUNCTION_LINES],
        "oversized_functions": [item for item in function_lengths if item[2] > MAX_FUNCTION_LINES],
        "target_drift_modules": target_drift_modules,
        "grandfathered_modules": [(p, n) for p, n in hard_modules if pathlib.Path(p).name in GRANDFATHERED_OVERSIZED_MODULES],
        "oversized_modules": [(p, n) for p, n in hard_modules if pathlib.Path(p).name not in GRANDFATHERED_OVERSIZED_MODULES],
    }

def collect_sections(
    repo_dir: pathlib.Path,
    drive_root: pathlib.Path,
) -> Tuple[List[Tuple[str, str]], Dict[str, Any]]:
    """Collect reviewable repo files for codebase-health metrics."""
    entries, omitted = iter_repo_pack_entries(
        repo_dir,
        skip_dir_prefixes=_HEALTH_SKIP_DIR_PREFIXES,
    )
    sections = [(f"repo/{rel}", content) for rel, content, _lang, _note in entries]
    total_chars = sum(len(content) for _path, content in sections)
    stats = {
        "files": len(sections),
        "chars": total_chars,
        "omitted": len(omitted),
    }
    return sections, stats
