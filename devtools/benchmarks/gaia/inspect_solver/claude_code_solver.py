"""Inspect solver that drives **Claude Code** (`claude -p`) as a GAIA harness.

Part of the multi-harness GAIA comparison rig. Like ``ouroboros_solver``, this is a
thin shim: official GAIA task construction/scoring stays in inspect_evals; this shim
only obtains a structured final answer from the external Claude Code CLI.

Auth: Claude Code headless needs credentials. We inject ANTHROPIC_API_KEY (resolved
from the environment, then ``data/settings.json``, then ``~/file1.txt``) into the
subprocess env so a non-interactive run does not hit "Not logged in". The key is
never logged.

Run it directly for a driver self-test:
    OUROBOROS skip -- python3.11 claude_code_solver.py "What is 7 times 8?"
"""

from __future__ import annotations

import json
import os
import pathlib
import re
import subprocess
import sys
from types import SimpleNamespace
from typing import Any

if str(pathlib.Path(__file__).resolve().parents[4]) not in sys.path:
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[4]))

# Reuse the proven, hardened staging + prompt extraction from the Ouroboros solver
# (it stages GAIA attachments and filters repo/data/secret paths — identical needs here).
from devtools.benchmarks.gaia.inspect_solver.ouroboros_solver import (  # noqa: E402
    _attachment_paths_from_state,
    _ensure_gaia_run_root,
    _state_prompt,
)
from devtools.benchmarks.common.run_roots import run_root  # noqa: E402

try:
    from inspect_ai.solver import Generate, TaskState, solver
except Exception:  # pragma: no cover - inspect is an optional benchmark dependency
    Generate = Any  # type: ignore
    TaskState = Any  # type: ignore

    def solver(fn):  # type: ignore
        return fn


_FINAL_RE = re.compile(r"FINAL ANSWER:\s*(.+?)\s*$", re.IGNORECASE | re.DOTALL)
# Fallback patterns for when the model omits the FINAL ANSWER marker and the Claude Code
# CLI wraps the bare answer in a meta sentence ("...the answer remains **Fred**", "...the
# answer was already computed: 17.056"). Without these the verbose line defeats the GAIA
# exact-match scorer even though the answer is correct.
_ANSWER_FALLBACK_RES = [
    re.compile(r"answer (?:was already (?:computed|determined)|is|was|remains)\s*[:\-]?\s*(.+)", re.IGNORECASE),
    re.compile(r"already (?:computed|determined)\s*[:\-]?\s*(.+)", re.IGNORECASE),
]


def _clean_answer(s: str) -> str:
    """First line; strip markdown bold/emphasis and surrounding/trailing punctuation."""
    s = (s or "").strip()
    if not s:
        return ""
    s = s.splitlines()[0].strip()
    s = s.replace("**", "").replace("*", "").strip()
    return s.strip().rstrip(".").strip()


def _resolve_anthropic_key() -> str:
    """Env first, then data/settings.json, then ~/file1.txt. Never logged."""
    key = os.environ.get("ANTHROPIC_API_KEY")
    if key:
        return key
    repo = pathlib.Path(__file__).resolve().parents[4]
    try:
        s = json.loads((repo.parent / "data" / "settings.json").read_text(encoding="utf-8"))
        for k in ("ANTHROPIC_API_KEY", "anthropic_api_key"):
            if s.get(k):
                return str(s[k])
    except Exception:
        pass
    try:
        for line in (pathlib.Path.home() / "file1.txt").read_text(encoding="utf-8").splitlines():
            if line.strip().lower().startswith("anthropic:"):
                return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return ""


def _extract_final_answer(text: str) -> str:
    """Pull the bare answer after the last 'FINAL ANSWER:' marker; else last non-empty line."""
    if not text:
        return ""
    matches = list(_FINAL_RE.finditer(text))
    if matches:
        return _clean_answer(matches[-1].group(1))
    for rx in _ANSWER_FALLBACK_RES:  # no marker -> recover a verbose-wrapped answer
        m = list(rx.finditer(text))
        if m:
            return _clean_answer(m[-1].group(1))
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return _clean_answer(lines[-1]) if lines else ""


def run_claude_code(
    prompt: str,
    sample_id: str = "sample",
    attachments: list[pathlib.Path] | None = None,
    workdir: pathlib.Path | None = None,
) -> dict:
    model = os.environ.get("GAIA_CLAUDE_MODEL", "claude-sonnet-4-5")
    max_turns = os.environ.get("GAIA_CLAUDE_MAX_TURNS", "40")
    effort = os.environ.get("GAIA_CLAUDE_EFFORT", "high")  # reasoning effort: low|medium|high|max
    allowed = os.environ.get("GAIA_CLAUDE_ALLOWED_TOOLS", "Bash Read WebSearch WebFetch Glob Grep")
    timeout_sec = float(os.environ.get("GAIA_SAMPLE_TIMEOUT_SEC", "3600") or "3600")

    work = pathlib.Path(workdir) if workdir else pathlib.Path.cwd()
    work.mkdir(parents=True, exist_ok=True)

    full_prompt = prompt
    if attachments:
        names = ", ".join(p.name for p in attachments)
        full_prompt += f"\n\nProvided file(s) are in your current working directory: {names}"
    if "FINAL ANSWER:" not in full_prompt:
        full_prompt += (
            "\n\nWork through the task, then end your response with a single line, "
            "exactly: FINAL ANSWER: <your answer>\nThe answer must be a number or as few "
            "words as possible, with no units unless asked."
        )

    cmd = [
        "claude", "-p", full_prompt,
        "--output-format", "json",
        "--model", model,
        "--effort", effort,
        "--allowedTools", allowed,
        "--max-turns", str(max_turns),
        "--dangerously-skip-permissions",
    ]
    env = dict(os.environ)
    key = _resolve_anthropic_key()
    if key:
        env["ANTHROPIC_API_KEY"] = key
    env.pop("ANTHROPIC_AUTH_TOKEN", None)  # avoid OAuth path stealing precedence

    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout_sec, cwd=str(work), env=env
        )
    except subprocess.TimeoutExpired as exc:  # crash isolation: one hang never aborts the eval
        return {"final_answer": "", "returncode": -1, "raw": "", "stderr_tail": f"TIMEOUT: {str(exc)[:300]}"}
    except Exception as exc:  # noqa: BLE001
        return {"final_answer": "", "returncode": -1, "raw": "", "stderr_tail": f"SPAWN ERROR: {type(exc).__name__}: {str(exc)[:300]}"}

    raw = proc.stdout or ""
    result_text = ""
    cost_usd = 0.0
    usage: dict = {}
    try:
        env_obj = json.loads(raw)
        result_text = str(env_obj.get("result", "") or "")
        cost_usd = float(env_obj.get("total_cost_usd") or env_obj.get("cost_usd") or 0.0)
        usage = env_obj.get("usage") or {}
        if env_obj.get("is_error"):
            return {"final_answer": "", "returncode": proc.returncode, "raw": raw[:2000],
                    "cost_usd": cost_usd, "usage": usage,
                    "stderr_tail": f"claude is_error: {result_text[:300]}"}
    except Exception:
        result_text = raw  # non-JSON fallback
    return {
        "final_answer": _extract_final_answer(result_text),
        "returncode": proc.returncode,
        "raw": result_text[:4000],
        "cost_usd": cost_usd,
        "usage": usage,
        "stderr_tail": (proc.stderr or "")[-2000:],
    }


@solver
def claude_code_solver():
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        sample_id = str(getattr(state, "sample_id", "") or getattr(state, "id", "") or "sample")
        repo = pathlib.Path(__file__).resolve().parents[4]
        root = _ensure_gaia_run_root(
            pathlib.Path(os.environ.get("GAIA_HARNESS_RUN_ROOT") or run_root("gaia_harness")).resolve(strict=False),
            repo,
        )
        safe = "".join(ch if ch.isalnum() or ch in "-_." else "-" for ch in sample_id)
        sample_dir = root / "samples" / safe
        workdir = sample_dir / "workdir"
        prompt = _state_prompt(state)
        attachments = _attachment_paths_from_state(state, sample_dir, prompt)
        for a in attachments:  # mirror into the agent cwd so Read/Bash can reach them
            try:
                (workdir).mkdir(parents=True, exist_ok=True)
                dest = workdir / a.name
                if a.resolve() != dest.resolve():
                    dest.write_bytes(a.read_bytes())
            except Exception:
                pass
        result = run_claude_code(prompt, sample_id=sample_id, attachments=attachments, workdir=workdir)
        if getattr(state, "metadata", None) is None:
            state.metadata = {}
        state.metadata["claude_code_raw"] = result.get("raw", "")
        state.metadata["claude_code_stderr"] = result.get("stderr_tail", "")
        state.metadata["claude_code_cost_usd"] = result.get("cost_usd", 0.0)
        state.metadata["claude_code_usage"] = result.get("usage", {})
        if getattr(state, "output", None) is None:
            state.output = SimpleNamespace(completion="")
        state.output.completion = result["final_answer"]
        return state

    return solve


if __name__ == "__main__":  # driver self-test (no inspect): python3.11 claude_code_solver.py "<q>"
    q = sys.argv[1] if len(sys.argv) > 1 else "What is 7 times 8?"
    print(json.dumps(run_claude_code(q, sample_id="selftest"), indent=2)[:1500])
