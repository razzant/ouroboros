"""Shared configuration objects and primitives for the OSWorld step loop.

Verbatim extraction from ``run_step_agent.py`` (v7 stream W). A leaf may never
import the launcher (cycle), so the typed run configuration and the two tiny
primitives the leaves share with it are owned here. ``run_step_agent.py``
re-exports every name, so its module surface and behaviour are unchanged.
"""

from __future__ import annotations

import json
import re
import urllib.request
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

@dataclass
class StepAgentConfig:
    ouroboros_bin: str
    ouroboros_url: str
    repo_dir: Path
    data_dir: Path
    settings_path: Path
    result_dir: Path
    task_id: str
    model: str
    timeout_sec: int
    max_obs_chars: int
    screenshot_check_only: bool
    disable_tools: str = "claude_code_edit"


@dataclass
class TaskRecordConfig:
    run_dir: Path
    result_root: Path
    repo_dir: Path
    settings_path: Path
    example_id: str
    domain: str
    reward: float | None
    steps: int
    status: str
    reason_code: str
    # The ADMITTED manifest (persisted by `admit_benchmark_run` before anything could
    # refuse). Required, and deliberately without a default: the records used to fall back
    # to REBUILDING it with `require_clean=False`, which wrote a manifest whose `seed_gate`
    # said the run was admissible on exactly the path where the gate had REFUSED it.
    base_manifest: dict[str, Any]
    error: str = ""
    extra: dict[str, Any] | None = None


@dataclass
class PreflightConfig:
    osworld_root: Path
    task_path: Path
    path_to_vm: str
    repo_dir: Path
    data_dir: Path
    settings_path: Path
    result_root: Path
    ouroboros_url: str
    model: str
    provider_name: str = "vmware"
    allow_scaffold_mismatch: bool = False


def _safe_slug(text: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "-", text).strip("-._")
    return cleaned[:80] or uuid.uuid4().hex[:8]


def _http_json(url: str, timeout: float = 5.0) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        raw = resp.read().decode("utf-8", errors="replace")
    return json.loads(raw) if raw.strip().startswith("{") else {"raw": raw}
