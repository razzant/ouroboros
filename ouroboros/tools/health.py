"""Codebase health tool — complexity metrics and self-assessment."""

import logging
import os
import pathlib
from typing import Any, Dict

from ouroboros.tools.registry import ToolContext, ToolEntry

log = logging.getLogger(__name__)


def _codebase_health(ctx: ToolContext) -> str:
    """Compute and format codebase health report."""
    try:
        from ouroboros.review import collect_sections, compute_complexity_metrics

        repo_dir = pathlib.Path(ctx.repo_dir)
        drive_root = pathlib.Path(os.environ.get("DRIVE_ROOT", "/home/zera/ouroboros/drive"))

        sections, stats = collect_sections(repo_dir, drive_root)
        metrics = compute_complexity_metrics(sections)

        # Format report
        lines = []
        lines.append("## Codebase Health Report\n")
        lines.append(f"**Analyzed:** {stats['files']} files, {stats['chars']:,} chars")
        lines.append(f"**Files:** {metrics['total_files']} ({metrics['py_files']} Python)")
        lines.append(f"**Total lines:** {metrics['total_lines']:,}")
        lines.append(f"**Functions:** {metrics['total_functions']}")
        lines.append(f"**Avg function length:** {metrics['avg_function_length']} lines")
        lines.append(f"**Max function length:** {metrics['max_function_length']} lines")

        # Largest files
        if metrics.get("largest_files"):
            lines.append("\n### Largest Files")
            for path, size in metrics["largest_files"][:10]:
                marker = " ⚠️ OVERSIZED" if size > 1000 else ""
                lines.append(f"  {path}: {size} lines{marker}")

        # Longest functions
        if metrics.get("longest_functions"):
            lines.append("\n### Longest Functions")
            for path, start, length in metrics["longest_functions"][:10]:
                marker = " ⚠️ OVERSIZED" if length > 150 else ""
                lines.append(f"  {path}:{start}: {length} lines{marker}")

        # Warnings
        oversized_funcs = metrics.get("oversized_functions", [])
        oversized_mods = metrics.get("oversized_modules", [])

        if oversized_funcs or oversized_mods:
            lines.append("\n### ⚠️ Bible Violations (Principle 5: Minimalism)")
            if oversized_funcs:
                lines.append(f"  Functions > 150 lines: {len(oversized_funcs)}")
                for path, start, length in oversized_funcs:
                    lines.append(f"    - {path}:{start} ({length} lines)")
            if oversized_mods:
                lines.append(f"  Modules > 1000 lines: {len(oversized_mods)}")
                for path, size in oversized_mods:
                    lines.append(f"    - {path} ({size} lines)")
        else:
            lines.append("\n✅ No Bible violations detected (all functions < 150 lines, all modules < 1000 lines)")

        return "\n".join(lines)

    except Exception as e:
        log.warning("codebase_health failed: %s", e, exc_info=True)
        return f"⚠️ Failed to compute codebase health: {e}"


def get_tools():
    return [
        ToolEntry("codebase_health", {
            "name": "codebase_health",
            "description": "Get codebase complexity metrics: file sizes, longest functions, modules exceeding limits. Useful for self-assessment per Bible Principle 5 (Minimalism).",
            "parameters": {"type": "object", "properties": {}, "required": []},
        }, _codebase_health),
    ]
