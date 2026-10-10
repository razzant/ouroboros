"""Codebase health tool — complexity metrics and self-assessment."""

from ouroboros.tools.tool_result import ToolResult, _publish_tool_result, completed_local_read

import logging
import pathlib

from ouroboros.tools.registry import ToolContext, ToolEntry

log = logging.getLogger(__name__)


@completed_local_read
def _codebase_health(ctx: ToolContext) -> str:
    """Compute and format codebase health report."""
    try:
        from ouroboros.review import collect_size_ratchet_inventory, compute_repo_complexity_metrics, size_headroom_lines

        repo_dir = pathlib.Path(ctx.repo_dir)
        inventory = collect_size_ratchet_inventory(repo_dir)
        metrics = compute_repo_complexity_metrics(repo_dir, inventory=inventory)
        stats = {
            "files": metrics["total_files"],
            "chars": metrics["total_bytes"],
        }

        # Format report
        lines = []
        lines.append("## Codebase Health Report\n")
        lines.append(f"**Analyzed:** {stats['files']} files, {stats['chars']:,} UTF-8 bytes")
        if stats.get("truncated"):
            lines.append(f"**Compacted files:** {stats['truncated']}")
        if stats.get("dropped"):
            dropped_paths = stats.get("dropped_paths") or []
            preview = ", ".join(dropped_paths[:5])
            lines.append(
                f"**Dropped files due review budget:** {stats['dropped']}"
                + (f" ({preview}{' ...' if len(dropped_paths) > 5 else ''})" if preview else "")
            )
        lines.append(
            f"**Files:** {metrics['total_files']} ({metrics['py_files']} Python, {metrics.get('js_files', 0)} gated JS)"
        )
        lines.append(f"**Total lines:** {metrics['total_lines']:,}")
        lines.append(f"**Functions:** {metrics['total_functions']}")
        lines.append(f"**Avg function length:** {metrics['avg_function_length']} lines")
        lines.append(f"**Max function length:** {metrics['max_function_length']} lines")
        lines.append("\n### Size Headroom (information; official CI enforces the limits)")
        lines.extend(size_headroom_lines(inventory))
        from ouroboros.reference_books import BOOK_GROWTH_RULE, book_balances, render_book_balance

        books = book_balances(repo_dir)
        if books:
            lines.append("\n### Reference books (information; official CI enforces)")
            lines.extend(f"  {render_book_balance(balance)}" for balance in books)
            lines.append(f"  {BOOK_GROWTH_RULE}")

        from ouroboros.review import (
            MAX_FUNCTION_LINES,
            MAX_MODULE_LINES,
            TARGET_FUNCTION_LINES,
            TARGET_MODULE_LINES,
        )

        # Largest files
        if metrics.get("largest_files"):
            lines.append("\n### Largest Files")
            for path, size in metrics["largest_files"][:10]:
                if size > MAX_MODULE_LINES:
                    marker = " 🚫 HARD LIMIT"
                elif size > TARGET_MODULE_LINES:
                    marker = " ⚠️ TARGET DRIFT"
                else:
                    marker = ""
                lines.append(f"  {path}: {size} lines{marker}")

        # Longest functions
        if metrics.get("longest_functions"):
            lines.append("\n### Longest Functions")
            for path, start, length in metrics["longest_functions"][:10]:
                if length > MAX_FUNCTION_LINES:
                    marker = " 🚫 HARD LIMIT"
                elif length > TARGET_FUNCTION_LINES:
                    marker = " ⚠️ TARGET DRIFT"
                else:
                    marker = ""
                lines.append(f"  {path}:{start}: {length} lines{marker}")

        # Warnings
        target_drift_funcs = metrics.get("target_drift_functions", [])
        target_drift_mods = metrics.get("target_drift_modules", [])
        grandfathered_funcs = metrics.get("grandfathered_functions", [])
        grandfathered_mods = metrics.get("grandfathered_modules", [])
        oversized_funcs = metrics.get("oversized_functions", [])
        oversized_mods = metrics.get("oversized_modules", [])

        if (
            oversized_funcs
            or oversized_mods
            or grandfathered_funcs
            or grandfathered_mods
            or target_drift_funcs
            or target_drift_mods
        ):
            lines.append("\n### Complexity Status (Principle 7: Minimalism)")
            if oversized_funcs:
                lines.append(f"  Hard-limit functions > {MAX_FUNCTION_LINES} lines: {len(oversized_funcs)}")
                for path, start, length in oversized_funcs:
                    lines.append(f"    - {path}:{start} ({length} lines)")
            elif target_drift_funcs:
                lines.append(f"  Target-drift functions > {TARGET_FUNCTION_LINES} lines: {len(target_drift_funcs)}")
            if grandfathered_funcs:
                lines.append(
                    f"  Grandfathered functions still above {MAX_FUNCTION_LINES} lines: {len(grandfathered_funcs)}"
                )
                for path, start, length in grandfathered_funcs:
                    lines.append(f"    - {path}:{start} ({length} lines)")
            if oversized_mods:
                lines.append(f"  Hard-limit modules > {MAX_MODULE_LINES} lines: {len(oversized_mods)}")
                for path, size in oversized_mods:
                    lines.append(f"    - {path} ({size} lines)")
            if grandfathered_mods:
                lines.append(f"  Grandfathered modules still above {MAX_MODULE_LINES} lines: {len(grandfathered_mods)}")
                for path, size in grandfathered_mods:
                    lines.append(f"    - {path} ({size} lines)")
            elif target_drift_mods:
                lines.append(f"  Target-drift modules > {TARGET_MODULE_LINES} lines: {len(target_drift_mods)}")
        else:
            lines.append(
                "\n✅ No hard P7 limit violations detected "
                f"(all functions <= {MAX_FUNCTION_LINES} lines, "
                f"all non-grandfathered modules <= {MAX_MODULE_LINES} lines)"
            )

        # Size-ratchet validator findings (manifest matches the tree + shrink-only
        # transition). The official repository CI `size_ratchet` lane is the
        # enforcing surface; this report and check_worktree_readiness only warn.
        try:
            from ouroboros.review import validate_size_ratchet

            ratchet_findings = validate_size_ratchet(repo_dir, inventory=inventory)
        except Exception as ratchet_exc:
            lines.append(f"\n### Size-Ratchet Findings\n  ⚠️ validator unavailable: {ratchet_exc}")
        else:
            if ratchet_findings:
                lines.append("\n### Size-Ratchet Findings (official CI will enforce)")
                for finding in ratchet_findings:
                    lines.append(f"  - {finding}")
            else:
                lines.append(
                    "\n✅ Size-ratchet manifest matches the live tree and is shrink-only "
                    "against the committed authority"
                )

        return "\n".join(lines)

    except Exception as e:
        log.warning("codebase_health failed: %s", e, exc_info=True)
        return _publish_tool_result(ctx, ToolResult(status="error", code="TOOL_ERROR", text=(f"⚠️ Failed to compute codebase health: {e}")))


def get_tools():
    return [
        ToolEntry(
            "codebase_health",
            {
                "name": "codebase_health",
                "description": "Get codebase complexity metrics: file sizes, longest functions, modules exceeding limits. Useful for self-assessment per Bible Principle 7 (Minimalism).",
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
            _codebase_health,
        ),
    ]
