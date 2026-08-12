"""Review collection and complexity metrics."""

from __future__ import annotations

import pathlib
from typing import Any, Dict, List, Tuple

from ouroboros.tools.review_helpers import (
    _VENDORED_NAMES,
    _VENDORED_SUFFIXES,
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
# v6.53.0 benchmark-generalization hardening adds small typed helpers for Observable Acceptance Claims,
# support_refs, GAIA profiles, media frame extraction, VLM timeout wrapping, and workspace inheritance.
# Cap intentionally moves with a small headroom rather than hiding growth elsewhere.
# v6.54.3 runtime-reliability: +~20 functions for the OUROBOROS_SAFETY_MODE owner-guard set
# (config ratchet/getters, owner endpoint, registry/browser detectors — mirrors the established
# context-mode/scope-floor pattern), transport-timeout SSOT getters, the read-vs-write
# runtime_data scan refinement, and slot-visibility helpers; cap 3636 -> 3690 with small headroom.
# v6.56.0 cost-axis/bench sprint: +~5 functions (task_pacing cost milestones +
# ceiling resolver, loop transport wrapper, media ffmpeg resolver chain,
# protected-artifact round-2 classifiers) — deliberate feature growth; 3690 -> 3699
# with the usual small headroom.
# v6.57.0 swarm/outcome-honesty + settings sprint (Phase 1): +~25 functions —
# find_child_tasks scope + compute_cost_with_children, the policy_denials bucket,
# verify refused_out_of_scope helpers, the EFFORT_SCALE SSOT (rank/clamp/step-down)
# + learned per-route effort-ceiling store (capability_evidence) + llm.py clamp/record,
# the subagent profile-summary + capability-mismatch message, and the Safety-mode
# UI wiring. Deliberate feature growth; 3699 -> 3740 with the usual small headroom.
# v6.58.0–v6.59.0 projects foundation + entry (Phases 2–3): +~20 functions —
# workspace_admission SSOT (validator/room-resolver/preflight-cap/workspace-block),
# coop_checkpoint pair, project_sources attach/clone, the projects update/delete/
# fs-dirs gateway handlers, and registry provenance/delete helpers; 3740 -> 3775.
# v6.62.0 outbound chat file delivery (ported from the 6.57.0–6.58.7 line): +5 functions
# — _detect_document_mime + _send_file (tools/core.py), send_document (message_bus.py),
# _handle_send_document (events.py), download_url_for_local_file (gateway/files.py) for the
# send_file capability + WKWebView-safe download. Measured merged count is 3770 — it fits
# within the existing 3775 cap (5 headroom), so no bump is needed.
# v6.63.0 unix_computer_use remote backends (OSWorld HTTP / SSH macOS, PR #64): +~29
# functions in skills/unix_computer_use/plugin.py (connection registry, remote
# screenshot/input/exec translation, fail-closed guards) — deliberate feature
# growth for the OSWorld cu_bridge runner; 3775 -> 3805 with the usual headroom.
# v6.64.0 clean rebuild: +195 measured functions over the 3801-function base,
# concentrated in the physical-attempt ledger (+44), root task-acceptance and
# lifecycle seams, Project routing/dialogue/tombstones, and ordinary-task context
# fit. A caller/AST audit removed redundant quorum/lock/startup/mailbox wrappers;
# no generic review/fanout/admission platform remains. The owner chose a stable
# ceiling of 5000 instead of continuing the historical current-count-plus-slack
# churn.
# The owner explicitly raised this structural ceiling for v6.64.0 so normal
# delivery work is not blocked by a moving "current count + epsilon" gate.
# This is still a coarse smoke alarm; per-module/function complexity checks and
# review remain the tools for preventing local bloat.
# 5000 -> 5100 at the v6.89.0 Claudexor-integration synthesis: eight branches
# each fit under 5000 alone and their union ships a whole delegation subsystem
# (nanny transport, custody, review lanes, owned daemon, slots UI) while also
# DELETING the claude_code edit path, the multi-slot fan-out and the enumerated
# write fence. A disclosed budget raise, not a silent one (P7: the next cycle
# owes consolidation before growth).
# 5100 → 6000 (2026-08-05, owner decision): the merge of the public v6.88.0
# line (MiniMax provider, crash-safe managed updates, two-pass gate,
# process_containment) into the Claudexor-integration line unions two
# independently-green trees, and the owner raised the ceiling with real
# headroom outright — a gate that fails on every routine union is churn, not
# protection ("подними сразу до 6000 чтобы не бесило"). The consolidation
# debt above still stands; the gate now guards against runaway growth only.
MAX_TOTAL_FUNCTIONS = 6000
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
    # v6.62.0 unix_computer_use remote backends (OSWorld HTTP / SSH macOS, PR #64)
    # grew skills/unix_computer_use/plugin.py past the 1600 gate: the remote
    # screenshot/input/exec translation + connection registry + fail-closed guards
    # live inline with the local backend. Extracting the remote translation layer
    # into a sibling payload module is a tracked follow-up (the skill loader would
    # need to import a second payload file); accepted debt until then. Keyed by
    # REPO-RELATIVE path (not the bare basename) so a future skill's plugin.py is
    # not silently exempted (SKILL.md convention is `entry: plugin.py`).
    "skills/unix_computer_use/plugin.py",
    # 2026-08-05 merge of the public v6.88.0 line: its crash-safe managed-update
    # hardening grew git_ops.py to 1597 (three lines under the gate), and the
    # Claudexor-integration line's owner-restart lever tipped the union past
    # 1600. Trimming a freshly-hardened update/rescue module at the merge seam
    # is higher-risk than the debt; splitting the managed-update transaction
    # helpers out of git_ops is the tracked follow-up (same class as
    # registry/events above).
    "git_ops.py",
    # 2026-08-08 perf/lifecycle sprint (P4): web/**/*.js joined the module-size
    # gate — the chat module had grown to 4067 lines with no deterministic brake
    # because the gate filtered on `.py` only. chat.js is the single existing
    # offender above the 1600 hard gate; splitting its history/scroll/attachment
    # concerns is the tracked follow-up. Keyed by REPO-RELATIVE path (not the
    # bare basename), following the skills/unix_computer_use/plugin.py precedent,
    # so a future chat.js anywhere else is not silently exempted.
    "web/modules/chat.js",
    # 2026-08-09 web-UI redesign rebase: config.py was sitting at EXACTLY 1600
    # upstream, so the Changes screen's one new owner setting
    # (OUROBOROS_TASK_DIFF_GIT_TIMEOUT_SEC — the clamped per-invocation `git`
    # timeout behind GET /api/tasks/{id}/diff) plus its clamped getter does not
    # fit: the module has no reclaimable line left. config.py is a settings
    # REGISTRY that grows with every owner-facing knob, and its accessors are
    # bound to SETTINGS_DEFAULTS and the private clamp helper, so relocating one
    # getter to its consumer would either duplicate the clamp or reach through a
    # private API. Splitting the registry from the accessors is the tracked
    # follow-up; same "at the ceiling, crossed with a feature" class as
    # loop/shell/core above. NOTE: agent_task_pipeline.py hit the same wall in
    # this rebase and was made to FIT (no debt) — only config.py had no room.
    "config.py",
    # 2026-08-10 project-threads integration review (I4): supervisor/queue.py was
    # sitting at EXACTLY 1600, and the fix is ONE behavioural line — the enqueue
    # SSOT must STRIP the writer-lane pin, because a task in PENDING holds no lane
    # and the in-process crash retry re-enqueues the very dict `assign_tasks`
    # stamped, so attempt 2 held a folder it does not write in while the next
    # candidate for the folder it DOES write in read that folder as free. The line
    # has to live there: `enqueue_task` is the one door every requeue path goes
    # through, and putting the strip in a single caller reopens the others. Its
    # explanation was moved into `project_lease`'s docstring and the comment cut to
    # two lines, which still leaves the module 4 over.
    #
    # Why an exemption and not a trim, stated honestly — "it has no reclaimable
    # line" was the earlier wording and it is only half true. There is no SMALL
    # one: every private helper in the module has a live caller (checked one by
    # one — `queue_has_task_type`, `_kept_service_pids`, `_schedule_running_or_
    # queued` and the rest are all reached), and the remaining comments are prose
    # about defects. But there IS a large separable block: the ~320-line scheduled
    # tasks half, `_scheduled_tasks_path` through `check_scheduled_tasks`, which is
    # precisely the follow-up named below. So this entry is a deliberate decision
    # NOT to move 320 lines out of the queue SSOT — the module that owns
    # `_queue_lock`, the PENDING/RUNNING refs, cancellation and timeout enforcement
    # — inside a fix round, where P7 forbids the refactor and this feature's own
    # two-stack synthesis is the standing evidence of what a careless module move
    # costs. The precedent it rides is exact rather than approximate: `config.py`
    # sat at EXACTLY 1600 and crossed on one owner setting plus its clamped getter;
    # `workers.py`/`events.py`/`control.py`/`registry.py` sat at ~1591 and crossed
    # on the acting-subagent gating. Each records a reason and a tracked follow-up,
    # and so does this. The counter-example in `config.py`'s entry
    # (`agent_task_pipeline.py` hit the same wall and was made to FIT, no debt) is
    # the test that was applied here first: fitting this module needs the split,
    # not a trim, so the debt is taken deliberately and named. Splitting the
    # evolution/scheduling half out of the queue is the tracked follow-up (same "at
    # the ceiling, crossed by a one-line safety fix" class as config.py above).
    # Keyed by REPO-RELATIVE path so no other queue.py is silently exempted.
    "supervisor/queue.py",
}
# Bundle-only launcher is not part of the self-editable function budget.
FUNCTION_COUNT_EXCLUDED_FILES = {"launcher.py"}


def is_gated_js_module(path: str) -> bool:
    """True if `path` is a web JS module subject to the module-size gate.

    Accepts an optional leading `repo/` (health sections are prefixed) and
    covers ALL of web/**/*.js — not only web/modules/ (app.js must not escape) —
    excluding `web/tests/` and vendored/minified payloads
    (`_VENDORED_SUFFIXES`/`_VENDORED_NAMES`; `iter_repo_pack_entries` already
    drops those from health sections — excluded here too so the smoke-gate walk
    in tests/test_smoke.py shares this exact definition)."""
    posix = pathlib.PurePosixPath(str(path).replace("\\", "/"))
    rel = posix.as_posix()
    if rel.startswith("repo/"):
        rel = rel[len("repo/"):]
    if not (rel.startswith("web/") and rel.endswith(".js")):
        return False
    if rel.startswith("web/tests/"):
        return False
    name = posix.name
    if name in _VENDORED_NAMES or any(name.endswith(s) for s in _VENDORED_SUFFIXES):
        return False
    return True


def module_is_grandfathered(path: str) -> bool:
    """True if `path` matches GRANDFATHERED_OVERSIZED_MODULES by bare basename OR
    by repo-relative path. Accepts an optional leading `repo/` (health sections
    are prefixed `repo/...`) so the SSOT set is matched identically by the smoke
    gate and the codebase_health metric — the two consumers must never diverge."""
    posix = pathlib.PurePosixPath(str(path).replace("\\", "/"))
    rel = posix.as_posix()
    if rel.startswith("repo/"):
        rel = rel[len("repo/"):]
    return posix.name in GRANDFATHERED_OVERSIZED_MODULES or rel in GRANDFATHERED_OVERSIZED_MODULES


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
    # JS size gate (perf/lifecycle sprint): web JS modules join the drift/hard
    # buckets on line count only — the function-length scan above stays
    # Python-only (no JS parser; disclosed in DEVELOPMENT.md Module Size).
    js_files = [item for item in file_sizes if is_gated_js_module(item[0])]
    gated_files = py_files + js_files
    target_drift_modules = [(p, n) for p, n in gated_files if n > TARGET_MODULE_LINES]
    hard_modules = [(p, n) for p, n in gated_files if n > MAX_MODULE_LINES]

    return {
        "total_files": len(sections),
        "py_files": len(py_files),
        "js_files": len(js_files),
        "total_lines": total_lines,
        "total_functions": len(function_lengths),
        "avg_function_length": round(sum(func_lens) / max(1, len(func_lens)), 1) if func_lens else 0,
        "max_function_length": max(func_lens) if func_lens else 0,
        "largest_files": sorted(file_sizes, key=lambda x: x[1], reverse=True)[:10],
        "longest_functions": sorted(function_lengths, key=lambda x: x[2], reverse=True)[:10],
        "target_drift_functions": [item for item in function_lengths if item[2] > TARGET_FUNCTION_LINES],
        "oversized_functions": [item for item in function_lengths if item[2] > MAX_FUNCTION_LINES],
        "target_drift_modules": target_drift_modules,
        "grandfathered_modules": [(p, n) for p, n in hard_modules if module_is_grandfathered(p)],
        "oversized_modules": [(p, n) for p, n in hard_modules if not module_is_grandfathered(p)],
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
