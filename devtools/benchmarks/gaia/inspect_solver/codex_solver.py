"""Inspect solver that drives **OpenAI Codex CLI** (`codex exec`) as a GAIA harness.

Part of the multi-harness GAIA comparison rig (sibling of ``claude_code_solver``).
Codex is OpenAI-lane (model-locked to OpenAI providers in practice); default model
is gpt-5.5. Auth via OPENAI_API_KEY injected from env / ``data/settings.json`` /
``~/file1.txt`` (never logged).

Driver self-test:  python3.11 codex_solver.py "What is 7 times 8?"
"""

from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys
import tempfile
from types import SimpleNamespace
from typing import Any

if str(pathlib.Path(__file__).resolve().parents[4]) not in sys.path:
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[4]))

from devtools.benchmarks.gaia.inspect_solver.claude_code_solver import _extract_final_answer  # noqa: E402
from devtools.benchmarks.gaia.inspect_solver.ouroboros_solver import (  # noqa: E402
    _attachment_paths_from_state,
    _ensure_gaia_run_root,
    _state_prompt,
)
from devtools.benchmarks.common.run_roots import run_root  # noqa: E402
from devtools.benchmarks.gaia.inspect_solver import GAIA_ANTI_LEAK_INSTRUCTION, GAIA_FORMAT_INSTRUCTION  # noqa: E402
from devtools.benchmarks.gaia.bwrap_isolate import wrap as _bwrap_wrap  # noqa: E402

try:
    from inspect_ai.solver import Generate, TaskState, solver
except Exception:  # pragma: no cover
    Generate = Any  # type: ignore
    TaskState = Any  # type: ignore

    def solver(fn):  # type: ignore
        return fn


def _resolve_openai_key() -> str:
    key = os.environ.get("OPENAI_API_KEY")
    if key:
        return key
    repo = pathlib.Path(__file__).resolve().parents[4]
    try:
        s = json.loads((repo.parent / "data" / "settings.json").read_text(encoding="utf-8"))
        for k in ("OPENAI_API_KEY", "openai_api_key"):
            if s.get(k):
                return str(s[k])
    except Exception:
        pass
    try:
        for line in (pathlib.Path.home() / "file1.txt").read_text(encoding="utf-8").splitlines():
            if line.strip().lower().startswith("openai:"):
                return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return ""


def run_codex(
    prompt: str,
    sample_id: str = "sample",
    attachments: list[pathlib.Path] | None = None,
    workdir: pathlib.Path | None = None,
    trace_path: pathlib.Path | None = None,
) -> dict:
    model = os.environ.get("GAIA_CODEX_MODEL", "gpt-5.5")
    # Reasoning effort: codex's own default is "xhigh" (from ~/.codex/config.toml). For an
    # apples-to-apples comparison with Ouroboros we pin "high" by default; `-c key=value`
    # overrides the config value. Set GAIA_CODEX_EFFORT=xhigh to reproduce the codex default.
    effort = os.environ.get("GAIA_CODEX_EFFORT", "high")  # low|medium|high|xhigh
    timeout_sec = float(os.environ.get("GAIA_SAMPLE_TIMEOUT_SEC", "3600") or "3600")
    work = pathlib.Path(workdir) if workdir else pathlib.Path(tempfile.mkdtemp())
    work.mkdir(parents=True, exist_ok=True)

    full_prompt = prompt
    if attachments:
        names = ", ".join(p.name for p in attachments)
        full_prompt += f"\n\nProvided file(s) are in your current working directory: {names}"
    if "FINAL ANSWER:" not in full_prompt:
        full_prompt += GAIA_FORMAT_INSTRUCTION
    # Anti-lookup rule (SSOT, identical across harnesses; see METHODOLOGY.md).
    if GAIA_ANTI_LEAK_INSTRUCTION not in full_prompt:
        full_prompt += GAIA_ANTI_LEAK_INSTRUCTION

    last_msg = work / ".codex_last_message.txt"
    # --json streams JSONL events (tool/web-search activity) to stdout: without it
    # `codex exec` is a black box and the leakage audit scores the row clean by
    # construction. The final answer still comes from `-o last_message.txt`.
    cmd = ["codex", "exec", full_prompt, "--json",
           "--skip-git-repo-check", "--dangerously-bypass-approvals-and-sandbox",
           "-C", str(work), "-o", str(last_msg)]
    # Transport/auth fix (codex-cli >= 0.142): the default `openai` provider uses a
    # WebSocket transport to /v1/responses that does NOT carry OPENAI_API_KEY for
    # API-key (non-ChatGPT-login) auth, so a service-account key 401s ("Missing
    # bearer"). Define a custom HTTP provider hitting the SAME OpenAI endpoint/model
    # over the `responses` HTTP wire API, which authenticates from OPENAI_API_KEY.
    # This is transport-only: identical endpoint (api.openai.com), model, and direct
    # OpenAI routing — it does not align/relocate the endpoint. Disable via
    # GAIA_CODEX_HTTP_PROVIDER=0 if a future codex build fixes WS API-key auth.
    if os.environ.get("GAIA_CODEX_HTTP_PROVIDER", "1") != "0":
        cmd[3:3] = [
            "-c", 'model_providers.openai_http.name="OpenAI HTTP"',
            "-c", 'model_providers.openai_http.base_url="https://api.openai.com/v1"',
            "-c", 'model_providers.openai_http.env_key="OPENAI_API_KEY"',
            "-c", 'model_providers.openai_http.wire_api="responses"',
            "-c", 'model_provider="openai_http"',
        ]
    if effort:
        cmd[2:2] = ["-c", f"model_reasoning_effort={effort}"]  # override config.toml default
    if model:
        cmd[2:2] = ["--model", model]  # insert before the prompt is fine; codex parses flags anywhere
    env = dict(os.environ)
    key = _resolve_openai_key()
    if key:
        env["OPENAI_API_KEY"] = key

    try:
        proc = subprocess.run(
            _bwrap_wrap(cmd), capture_output=True, text=True, timeout=timeout_sec,
            cwd=str(work), env=env, stdin=subprocess.DEVNULL,  # DEVNULL: codex exec else waits on stdin
        )
    except subprocess.TimeoutExpired as exc:
        return {"final_answer": "", "returncode": -1, "raw": "", "stderr_tail": f"TIMEOUT: {str(exc)[:300]}"}
    except Exception as exc:  # noqa: BLE001
        return {"final_answer": "", "returncode": -1, "raw": "", "stderr_tail": f"SPAWN ERROR: {type(exc).__name__}: {str(exc)[:300]}"}

    if trace_path is not None and proc.stdout:
        try:  # pure JSONL event dump for audit_leakage
            trace_path.parent.mkdir(parents=True, exist_ok=True)
            trace_path.write_text(proc.stdout, encoding="utf-8")
        except Exception:
            pass
    result_text = ""
    try:
        if last_msg.exists():
            result_text = last_msg.read_text(encoding="utf-8", errors="replace")
    except Exception:
        pass
    if not result_text:
        result_text = proc.stdout or ""
    return {
        "final_answer": _extract_final_answer(result_text),
        "returncode": proc.returncode,
        "raw": result_text[:4000],
        "effort": effort,
        "stderr_tail": (proc.stderr or "")[-2000:],
    }


@solver
def codex_solver():
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
        for a in attachments:
            try:
                workdir.mkdir(parents=True, exist_ok=True)
                dest = workdir / a.name
                _n = 1
                while dest.exists() and dest.resolve() != a.resolve():  # collision-safe: never clobber a sibling
                    dest = workdir / f"{a.stem}-{_n}{a.suffix}"
                    _n += 1
                if a.resolve() != dest.resolve():
                    dest.write_bytes(a.read_bytes())
            except Exception:
                pass
        result = run_codex(prompt, sample_id=sample_id, attachments=attachments, workdir=workdir,
                           trace_path=sample_dir / "codex_trace.jsonl")
        if getattr(state, "metadata", None) is None:
            state.metadata = {}
        state.metadata["codex_raw"] = result.get("raw", "")
        state.metadata["codex_stderr"] = result.get("stderr_tail", "")
        state.metadata["codex_effort"] = result.get("effort", "")
        if getattr(state, "output", None) is None:
            state.output = SimpleNamespace(completion="")
        state.output.completion = result["final_answer"]
        return state

    return solve


if __name__ == "__main__":
    q = sys.argv[1] if len(sys.argv) > 1 else "What is 7 times 8?"
    print(json.dumps(run_codex(q, sample_id="selftest"), indent=2)[:1500])
