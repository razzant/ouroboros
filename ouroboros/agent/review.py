"""
Ouroboros agent review context builder.

Collects code snapshot + complexity metrics for review tasks.
"""

from __future__ import annotations

from typing import Any, Dict


def build_review_context(repo_dir, drive_root) -> str:
    """Collect code snapshot + complexity metrics for review tasks."""
    try:
        from ouroboros.review import collect_sections, compute_complexity_metrics, format_metrics
        sections, stats = collect_sections(repo_dir, drive_root)
        metrics = compute_complexity_metrics(sections)

        parts = [
            "## Code Review Context\n",
            format_metrics(metrics),
            f"\nFiles: {stats['files']}, chars: {stats['chars']}\n",
            "\nUse repo_read to inspect specific files. "
            "Use run_shell for tests. Key files below:\n",
        ]

        total_chars = 0
        max_chars = 80_000
        files_added = 0
        for path, content in sections:
            if total_chars >= max_chars:
                parts.append(f"\n... ({len(sections) - files_added} more files, use repo_read)")
                break
            preview = content[:2000] if len(content) > 2000 else content
            file_block = f"\n### {path}\n```\n{preview}\n```\n"
            total_chars += len(file_block)
            parts.append(file_block)
            files_added += 1

        return "\n".join(parts)
    except Exception as e:
        return f"## Code Review Context\n\n(Failed to collect: {e})\nUse repo_read and repo_list to inspect code."
