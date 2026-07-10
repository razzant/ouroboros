"""Inspect solver shim that invokes ``ouroboros run --result-json-out``.

This module is imported by inspect_evals when running GAIA. It is deliberately
small: official task construction/scoring stays in inspect_evals, while this
shim is only responsible for obtaining Ouroboros's structured final_answer.
"""

from __future__ import annotations

import json
import os
import pathlib
import re
import shutil
import subprocess
import sys
import time
from types import SimpleNamespace
from typing import Any

if str(pathlib.Path(__file__).resolve().parents[4]) not in sys.path:
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[4]))

from devtools.benchmarks.common.run_roots import ensure_outside_repo, run_root
from devtools.benchmarks.gaia.inspect_solver import GAIA_ANTI_LEAK_INSTRUCTION, GAIA_FORMAT_INSTRUCTION
from devtools.benchmarks.gaia.bwrap_isolate import wrap as _bwrap_wrap

_SHARED_FILE_RE = re.compile(r"(?P<path>/shared_files/\S+)")


try:
    from inspect_ai.solver import Generate, TaskState, solver
except Exception:  # pragma: no cover - inspect is an optional benchmark dependency
    Generate = Any  # type: ignore
    TaskState = Any  # type: ignore

    def solver(fn):  # type: ignore
        return fn


def _ensure_gaia_run_root(path: pathlib.Path, repo: pathlib.Path) -> pathlib.Path:
    """Validate the benchmark run root without treating its own env as live data."""
    saved = {key: os.environ.pop(key, None) for key in ("OUROBOROS_DATA_DIR", "OUROBOROS_SETTINGS_PATH")}
    try:
        return ensure_outside_repo(path, repo)
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def run_ouroboros(prompt: str, sample_id: str = "sample", attachments: list[pathlib.Path] | None = None) -> dict:
    repo = pathlib.Path(__file__).resolve().parents[4]
    root = pathlib.Path(os.environ.get("GAIA_OUROBOROS_RUN_ROOT") or run_root("gaia")).resolve(strict=False)
    root = _ensure_gaia_run_root(root, repo)
    sample_dir = root / "samples" / "".join(ch if ch.isalnum() or ch in "-_." else "-" for ch in sample_id)
    sample_dir.mkdir(parents=True, exist_ok=True)
    result_json = sample_dir / "result.json"
    cmd = [
        sys.executable,
        "-m",
        "ouroboros.cli",
        "--url",
        os.environ.get("GAIA_OUROBOROS_URL", "http://127.0.0.1:8765"),
        "run",
        "--start",
        "--memory-mode",
        "empty",
        "--quiet",
        "--disable-tools",
        os.environ.get("GAIA_DISABLE_TOOLS", "web_search,claude_code_edit"),
        "--result-json-out",
        str(result_json),
    ]
    # P6: give the agent a VISIBLE deadline STRICTLY tighter than the outer hard kill
    # (GAIA_SAMPLE_TIMEOUT_SEC). The gateway derives deadline_at = now + timeout_sec,
    # so the agent gets 50/25/10% milestones + a save-at-10% nudge and finalizes before
    # the SIGKILL backstop. Honest: the visible deadline == the real budget minus a
    # finalization reserve (10%, capped at 240s); the agent is never told a deadline it
    # is killed before reaching. GAIA itself imposes no per-task time limit (the timeout
    # is an operator budget), so disclosing it to the agent is methodology-sanctioned.
    sample_timeout = float(os.environ.get("GAIA_SAMPLE_TIMEOUT_SEC", "7200") or "7200")
    reserve = min(240.0, max(1.0, round(sample_timeout * 0.1)))
    visible_deadline = max(1.0, sample_timeout - reserve)
    cmd.extend(["--timeout", str(visible_deadline)])
    # v6.60.0: the FINAL ANSWER discipline is a typed CONTRACT field now, not a global
    # SYSTEM.md rule — declare the protocol so the runtime injects the marker
    # instruction/nudges for THIS task (the prompt keeps the GAIA-specific format text).
    cmd.extend(["--task-metadata-json", '{"answer_protocol": "final_answer_line"}'])
    for path in [str(path) for path in (attachments or [])]:
        cmd.extend(["--attach", path])
    # P3: official GAIA answer protocol — append the shared format instruction (SSOT in
    # inspect_solver/__init__.py) so the runtime's typed extractor captures a clean,
    # correctly-shaped deliverable (number / few words / no units) instead of prose.
    if "FINAL ANSWER:" not in prompt:
        prompt = prompt + GAIA_FORMAT_INSTRUCTION
    # Anti-lookup rule (SSOT, identical across harnesses; see METHODOLOGY.md).
    if GAIA_ANTI_LEAK_INSTRUCTION not in prompt:
        prompt = prompt + GAIA_ANTI_LEAK_INSTRUCTION
    cmd.append(prompt)
    # Filesystem isolation: mask the GAIA answer cache from the whole `ouroboros run`
    # subprocess (server + task share one mount namespace; the dedicated server binds
    # loopback INSIDE the namespace, so the CLI still reaches it). Symmetric with the
    # CLI harnesses. No-op if GAIA_BWRAP_ISOLATE=0. See bwrap_isolate.py.
    cmd = _bwrap_wrap(cmd)
    timeout_sec = sample_timeout  # outer hard-kill backstop (> the visible deadline)
    proc = None
    for attempt in range(5):
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_sec)
        except subprocess.TimeoutExpired as exc:
            # Crash isolation: one hung sample must NEVER propagate and abort the whole
            # eval. Return a terminal per-sample result so inspect scores it and moves on.
            return {
                "final_answer": "",
                "returncode": -1,
                "result_json": str(result_json),
                "stderr_tail": f"TIMEOUT after {timeout_sec:g}s: {str(exc)[:500]}",
            }
        except Exception as exc:  # noqa: BLE001 - any spawn/env/OS failure is isolated too
            # Same crash isolation for non-timeout failures (spawn error, bad env, OSError):
            # a single sample's failure must produce a terminal result, never abort the eval.
            return {
                "final_answer": "",
                "returncode": -1,
                "result_json": str(result_json),
                "stderr_tail": f"SUBPROCESS ERROR: {type(exc).__name__}: {str(exc)[:500]}",
            }
        if proc.returncode == 0 or "supervisor is still starting" not in str(proc.stderr):
            break
        time.sleep(min(2.0 * (attempt + 1), 10.0))
    assert proc is not None
    payload = {}
    if result_json.exists():
        try:
            payload = json.loads(result_json.read_text(encoding="utf-8"))
        except Exception:
            payload = {}
    answer = payload.get("final_answer") or payload.get("result") or ""
    return {
        "final_answer": str(answer or "").strip(),
        "returncode": proc.returncode,
        "result_json": str(result_json),
        "stderr_tail": proc.stderr[-4000:],
    }


def _state_prompt(state: Any) -> str:
    user_prompt = getattr(state, "user_prompt", None)
    if getattr(user_prompt, "text", None):
        return str(user_prompt.text)
    if getattr(state, "input_text", None):
        return str(state.input_text)
    if getattr(state, "input", None):
        return str(state.input)
    return ""


def _attachment_paths_from_state(
    state: Any,
    sample_dir: pathlib.Path | None = None,
    prompt: str = "",
) -> list[pathlib.Path]:
    """Collect the REAL host attachment paths from a GAIA TaskState.

    v6.53.0: resolve real host file paths, then copy them into a per-sample
    run-root attachment directory before passing them via ``--attach``. The core
    still owns final staging into ``artifact_store/attachments``; the adapter copy
    keeps the CLI input path safe and run-local instead of asking the agent to
    read a stale ``/shared_files`` path or broad host cache.

    ``sample_dir`` is the run-local copy target. ``prompt`` is used only to
    resolve legacy `/shared_files/...` references when Inspect TaskState does not
    expose real attachment paths."""
    raw_items: list[Any] = []
    # GAIA's TaskState.files maps a sandbox path -> a host path; depending on the
    # inspect version the real host file can be the dict VALUE or the KEY, so collect
    # BOTH (the existence check below filters non-files). This was the staging bug:
    # reading only .values() staged zero files on this inspect version.
    for attr in ("files", "attachments"):
        value = getattr(state, attr, None)
        if isinstance(value, dict):
            raw_items.extend(value.values())
            raw_items.extend(value.keys())
        elif isinstance(value, (list, tuple)):
            raw_items.extend(value)
    metadata = getattr(state, "metadata", {}) or {}
    if isinstance(metadata, dict):
        for key in ("files", "attachments"):
            value = metadata.get(key)
            if isinstance(value, dict):
                raw_items.extend(value.values())
                raw_items.extend(value.keys())
            elif isinstance(value, (list, tuple)):
                raw_items.extend(value)
    shared_root = pathlib.Path(os.environ.get("GAIA_SHARED_FILES_ROOT") or "").expanduser()
    if shared_root and str(shared_root) not in ("", "."):
        shared_root_resolved = shared_root.resolve(strict=False)
        for match in _SHARED_FILE_RE.finditer(str(prompt or "")):
            trimmed = match.group("path").rstrip(".,;:)\\]\"'")
            pure = pathlib.PurePosixPath(trimmed)
            parts = [part for part in pure.parts if part not in {"", "/"}]
            rel_parts = parts[1:] if parts and parts[0] == "shared_files" else parts
            if rel_parts:
                direct = shared_root_resolved.joinpath(*rel_parts).resolve(strict=False)
                try:
                    direct.relative_to(shared_root_resolved)
                except ValueError:
                    continue
                if direct.exists():
                    raw_items.append(direct)
                else:
                    rel = pathlib.PurePosixPath(*rel_parts).name
                    try:
                        found = next(shared_root.rglob(rel))
                    except StopIteration:
                        found = None
                    if found is not None:
                        raw_items.append(found)
    out: list[pathlib.Path] = []
    seen: set[str] = set()
    repo = pathlib.Path(__file__).resolve().parents[4].resolve(strict=False)
    live_data = repo.parent / "data"
    for item in raw_items:
        path = pathlib.Path(str(getattr(item, "path", item))).expanduser().resolve(strict=False)
        if not path.exists() or not path.is_file():
            continue
        key = str(path)
        if key in seen:
            continue
        try:
            path.relative_to(repo)
            continue
        except ValueError:
            pass
        try:
            path.relative_to(live_data)
            continue
        except ValueError:
            pass
        lower = path.name.lower()
        secret_dirs = {".ssh", ".aws", ".config", ".gnupg"}
        if any(part.lower() in secret_dirs for part in path.parts):
            continue
        if any(token in lower for token in ("key", "token", "credential", ".env", "settings", "id_rsa", "id_ed25519")):
            continue
        seen.add(key)
        out.append(path)
    if sample_dir is not None and out:
        safe_dir = pathlib.Path(sample_dir) / "attachments"
        safe_dir.mkdir(parents=True, exist_ok=True)
        copied: list[pathlib.Path] = []
        used_names: set[str] = set()
        for src in out:
            stem = src.stem
            suffix = src.suffix
            name = src.name
            counter = 2
            while name in used_names:
                name = f"{stem}_{counter}{suffix}"
                counter += 1
            used_names.add(name)
            dst = safe_dir / name
            try:
                shutil.copy2(src, dst)
                copied.append(dst.resolve(strict=False))
            except Exception:
                copied.append(src)
        return copied
    return out


def _rewrite_shared_file_prompt(prompt: str, attachments: list[pathlib.Path]) -> str:
    """Replace stale GAIA /shared_files paths with attachment-manifest guidance."""
    if not attachments or "/shared_files/" not in str(prompt or ""):
        return prompt
    names = {p.name for p in attachments}

    def _replace(match: re.Match[str]) -> str:
        raw = match.group("path")
        trimmed = raw.rstrip(".,;:)\\]\"'")
        suffix = raw[len(trimmed):]
        name = pathlib.PurePosixPath(trimmed).name
        if name in names:
            return f"the attached file {name} (see the [ATTACHMENTS] manifest){suffix}"
        return match.group("path")

    note = (
        "\n\n[ATTACHMENT NOTE]\n"
        "The original GAIA prompt may mention /shared_files paths. In this runtime, "
        "those files are attached explicitly and are available through the task's "
        "[ATTACHMENTS] manifest, not by reading /shared_files directly.\n"
    )
    return _SHARED_FILE_RE.sub(_replace, prompt) + note


@solver
def ouroboros_solver():
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        sample_id = str(getattr(state, "sample_id", "") or getattr(state, "id", "") or "sample")
        repo = pathlib.Path(__file__).resolve().parents[4]
        # Validate the benchmark run root (must be outside the repo / not live data).
        # The path itself is no longer needed here (attachments now resolve to real
        # host paths staged by the core, not copies under a per-sample dir).
        _ensure_gaia_run_root(
            pathlib.Path(os.environ.get("GAIA_OUROBOROS_RUN_ROOT") or run_root("gaia")).resolve(strict=False),
            repo,
        )
        prompt = _state_prompt(state)
        root = pathlib.Path(os.environ.get("GAIA_OUROBOROS_RUN_ROOT") or run_root("gaia")).resolve(strict=False)
        sample_dir = root / "samples" / "".join(ch if ch.isalnum() or ch in "-_." else "-" for ch in sample_id)
        attachments = _attachment_paths_from_state(state, sample_dir=sample_dir, prompt=prompt)
        prompt = _rewrite_shared_file_prompt(prompt, attachments)
        result = run_ouroboros(prompt, sample_id=sample_id, attachments=attachments)
        if not hasattr(state, "metadata") or getattr(state, "metadata") is None:
            state.metadata = {}
        state.metadata["ouroboros_result_json"] = result.get("result_json", "")
        if not hasattr(state, "output") or getattr(state, "output") is None:
            state.output = SimpleNamespace(completion="")
        state.output.completion = result["final_answer"]
        return state

    return solve
