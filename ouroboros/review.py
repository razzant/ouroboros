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
    "tests/",
)
TARGET_MODULE_LINES = 1000
MAX_MODULE_LINES = 1600
TARGET_FUNCTION_LINES = 150
# Advisory SDK orchestration stays single-flow; split tracked as tech debt.
MAX_FUNCTION_LINES = 300
# Ceiling covers safety, review-state, tools/git, skills/extensions, gateway
# helpers, the packaged CLI bridge/installer, the v5.32 generated Atlas
# compiler/tests, v5.33 external-workspace CLI artifact/preflight contract,
# and the v6.1 subagent status SSOT/wait/browser-ingress hardening. Keep this
# tight and lower it again when the headless/subagent helpers settle.
# v6.4.0-rc.1 adds the send_video tool/bridge/event path; keep the cap tight
# while allowing that first-class transport surface.
# v6.5.0-rc.1 adds task-scoped artifact/user_files/root-resolution helpers
# for light-mode external deliverables; keep the headroom narrow.
# v6.6.0-rc.1 adds 4 helpers for effect-gated task-acceptance review and light-mode
# cognitive/root redirects: turn_has_reviewable_effects, _user_file_basenames,
# _extract_fenced_json, light_cognitive_or_root_redirect. Keep the headroom narrow.
# v6.7.0-rc.1 adds reliability helpers for the subagent/worker fixes: worker
# network policy (in_worker_process), monotonic status guard
# (_is_status_regression), terminal-event/cancel emitters (_emit_task_done_terminal,
# _emit_cancel_task_done, _drop_cancelled_pending), bundled-node resolution
# (embedded_node_candidates, resolve_bundled_node), marketplace opener
# (_build_opener), and the schedule pool-unavailable guard
# (_reject_schedule_pool_unavailable). v6.7.1 adds the out-of-process extension
# runner and proxy validation helpers. v6.9.0-rc.1 adds the first-class
# evolution campaign, schedule, and memory-provenance helpers. v6.10.0 adds
# adaptive LLM request normalization plus role-based remote/Colab bootstrap
# helpers; v6.11 adds narrow safety-critical headroom for hermetic preflight,
# live-mutation fuses, child advisory crash isolation, and evolution transaction
# evidence; v6.12 adds the QA-fix helpers (marketplace collision-rename, task-cost
# reconstruction, evolution-stop cancel, preflight process-tree reaping, worker
# log forwarding). Keep the headroom narrow and pay down after surfaces stabilize.
# GigaChat provider adds the native gigachat:: execution path helpers in llm.py
# (_get_gigachat_client, _gigachat_text, _gigachat_function_result,
# _gigachat_messages, _gigachat_functions, _gigachat_sanitize_schema,
# _chat_gigachat, _normalize_gigachat_response) plus the gateway catalog fetcher
# (_fetch_gigachat_model_catalog). Keep the headroom narrow.
MAX_TOTAL_FUNCTIONS = 2600
# Grandfathered modules are accepted debt until their surfaces stabilize/split.
GRANDFATHERED_OVERSIZED_MODULES = {
    "llm.py",
    "claude_advisory_review.py",
    "review_state.py",
    "server.py",
    "git.py",
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
