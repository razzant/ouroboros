"""Generate evolution.json metrics from sampled git history."""

from __future__ import annotations

import hashlib
import json
import logging
import re
import subprocess
import time
from typing import Any

from ouroboros.config import resolve_repo_dir
from ouroboros.tools.tool_result import ToolResult, _publish_tool_result

log = logging.getLogger(__name__)

_VERSION_RE = re.compile(r"v(\d+\.\d+\.\d+)")

MAX_POINTS = 100

def _git(args: list[str], timeout: int = 15) -> str:
    """Run git command in repo dir, return stdout or empty string on error."""
    try:
        r = subprocess.run(
            ["git"] + args,
            cwd=str(resolve_repo_dir()),  # per call: the serving checkout, never a home default
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=timeout,
        )
        return r.stdout if r.returncode == 0 else ""
    except Exception as e:
        log.warning("git %s failed: %s", args[:2], e)
        return ""


def _count_py_lines(commit_hash: str) -> tuple[int, int]:
    """Return (total_py_lines, module_count) for a commit using git show."""
    tree = _git(["ls-tree", "-r", "--name-only", commit_hash])
    py_files = [f for f in tree.splitlines() if f.endswith(".py")]
    total_lines = 0
    for f in py_files:
        content = _git(["show", f"{commit_hash}:{f}"], timeout=10)
        total_lines += content.count("\n")
    return total_lines, len(py_files)


def _get_file_bytes(commit_hash: str, *candidate_paths: str) -> int:
    """Return byte size of first existing file path in the commit, or 0."""
    for path in candidate_paths:
        content = _git(["show", f"{commit_hash}:{path}"], timeout=10)
        if content:
            return len(content.encode("utf-8"))
    return 0


def _extract_version(msg: str) -> str | None:
    m = _VERSION_RE.search(msg)
    return m.group(1) if m else None


def _collect_data() -> list[dict[str, Any]]:
    """Walk git history, sample commits, extract metrics."""
    log.info("evolution_stats: reading git log...")
    log_out = _git(["log", "--pretty=format:%H|%aI|%s", "--no-merges"])
    all_commits = []
    for line in log_out.splitlines():
        parts = line.split("|", 2)
        if len(parts) == 3:
            all_commits.append({"hash": parts[0], "ts": parts[1], "msg": parts[2]})

    if not all_commits:
        log.warning("evolution_stats: no commits found")
        return []

    n = len(all_commits)
    log.info("evolution_stats: %d commits total", n)

    # Select version-tagged commits + evenly spaced sample, always include first/last
    version_idx = {i for i, c in enumerate(all_commits) if _extract_version(c["msg"])}
    must_include = version_idx | {0, n - 1}

    step = max(1, n // MAX_POINTS)
    spaced_idx = set(range(0, n, step))
    candidate = sorted(must_include | spaced_idx)

    # Cap at MAX_POINTS while keeping all version commits
    if len(candidate) > MAX_POINTS:
        non_version = [i for i in candidate if i not in must_include]
        extra_slots = MAX_POINTS - len(must_include)
        if extra_slots > 0 and non_version:
            step2 = max(1, len(non_version) // extra_slots)
            extra = non_version[::step2][:extra_slots]
        else:
            extra = []
        candidate = sorted(must_include | set(extra))

    # Process in chronological order (oldest → newest)
    selected = list(reversed(candidate))
    log.info("evolution_stats: processing %d sampled commits...", len(selected))
    t0 = time.time()

    points: list[dict[str, Any]] = []
    for pos, idx in enumerate(selected):
        c = all_commits[idx]
        h = c["hash"]
        py_lines, module_count = _count_py_lines(h)
        bible_bytes = _get_file_bytes(h, "BIBLE.md", "prompts/BIBLE.md")
        system_bytes = _get_file_bytes(h, "prompts/SYSTEM.md", "SYSTEM.md")
        points.append({
            "ts": c["ts"],
            "hash": h[:8],
            "msg": c["msg"][:80],
            "version": _extract_version(c["msg"]),
            "py_lines": py_lines,
            "module_count": module_count,
            "bible_bytes": bible_bytes,
            "system_bytes": system_bytes,
        })
        if (pos + 1) % 10 == 0:
            log.info(
                "evolution_stats: %d/%d done (%.1fs)",
                pos + 1, len(selected), time.time() - t0,
            )

    log.info("evolution_stats: collected %d points in %.1fs", len(points), time.time() - t0)
    return points


def generate_evolution_stats(ctx: Any = None) -> str:
    """Export the legacy commit-sample metrics as an immutable local task file."""
    from ouroboros.artifacts import store_task_artifact_bytes, task_artifact_dir_path

    if not getattr(ctx, "task_id", "") or not getattr(ctx, "drive_root", None):
        return _publish_tool_result(ctx, ToolResult(
            status="unavailable", code="LEGACY_UNAVAILABLE",
            text="EVOLUTION_EXPORT_UNAVAILABLE: a task artifact store is required."))
    points = _collect_data()
    if not points:
        return "No data collected (empty git history?)"

    data = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "total_commits_sampled": len(points),
        "max_py_lines": max((p["py_lines"] for p in points), default=0),
        "max_bible_bytes": max((p["bible_bytes"] for p in points), default=0),
        "max_system_bytes": max((p["system_bytes"] for p in points), default=0),
        "points": points,
    }

    content = (json.dumps(data, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    name = f"evolution-{hashlib.sha256(content).hexdigest()}.json"
    try:
        artifact = store_task_artifact_bytes(ctx.drive_root, ctx.task_id, name, content)
    except (OSError, ValueError) as exc:
        return _publish_tool_result(ctx, ToolResult(
            status="error", code="TOOL_ERROR", text=f"EVOLUTION_EXPORT_FAILED: {exc}"))
    last = points[-1]
    text = (
        f"evolution_stats: saved {len(points)} points locally | "
        f"span={points[0]['ts'][:10]}…{last['ts'][:10]} | "
        f"py_lines={last['py_lines']} bible={last['bible_bytes']}B system={last['system_bytes']}B"
    )
    return json.dumps({"summary": text, "artifact": artifact, "publication": "local_only",
                       "file_path": str(task_artifact_dir_path(ctx.drive_root, ctx.task_id) / name)}, ensure_ascii=False)


def get_tools():
    """Auto-discovery entry point for ToolRegistry."""
    from ouroboros.tools.registry import ToolEntry

    return [
        ToolEntry(
            "generate_evolution_stats",
            {
                "name": "generate_evolution_stats",
                "description": (
                    "Export Evolution Time-Lapse data from the serving repository's git history to a local task file. "
                    "Collects per-commit metrics across three axes: "
                    "Technical (Python lines of code), Philosophical (BIBLE.md size), "
                    "Self-Concept (SYSTEM.md size). "
                    "Returns an immutable artifact_store JSON reference for reading or delivery with send_file. "
                    "No network publication, Git commit or push. The Dashboard uses its own local tag metrics. "
                    "Sampling a large history can take time."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
            },
            lambda ctx, **_: generate_evolution_stats(ctx),
        )
    ]
