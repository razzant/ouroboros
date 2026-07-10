"""Inspect solver that drives **Hermes Agent** (NousResearch, `hermes -z`) as a
GAIA harness.

Part of the multi-harness GAIA comparison rig (sibling of ``codex_solver`` /
``claude_code_solver``). Hermes is model-flexible: we pin it to the measured
solve model via ``-m`` + ``--provider`` (default OpenRouter gpt-5.5, matching the
Ouroboros/Codex gpt-5.5 lane). Reasoning effort is set in ``~/.hermes/config.yaml``
(``agent.reasoning_effort``, default we configure to ``high``); ``hermes -z`` has
no per-run effort flag. Web tools stay ENABLED for parity with the web-using
baselines (do NOT pass ``--toolsets none``); leakage is checked post-hoc by
``audit_leakage.py``.

``hermes -z PROMPT`` is Hermes's one-shot/headless mode: single prompt in, final
reply text out on stdout, nothing else — ideal for a solver.

Driver self-test:  python3 hermes_solver.py "What is 7 times 8?"
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


def _resolve_provider_key() -> tuple[str, str]:
    """Return (env_var_name, key) for the configured provider. Env first, then
    data/settings.json, then ~/file1.txt. Never logged."""
    provider = os.environ.get("GAIA_HERMES_PROVIDER", "openrouter").lower()
    env_var = "OPENAI_API_KEY" if provider == "openai" else "OPENROUTER_API_KEY"
    if os.environ.get(env_var):
        return env_var, os.environ[env_var]
    repo = pathlib.Path(__file__).resolve().parents[4]
    try:
        s = json.loads((repo.parent / "data" / "settings.json").read_text(encoding="utf-8"))
        if s.get(env_var):
            return env_var, str(s[env_var])
    except Exception:
        pass
    file1_key = "openai" if provider == "openai" else "openrouter"
    try:
        for line in (pathlib.Path.home() / "file1.txt").read_text(encoding="utf-8").splitlines():
            if line.strip().lower().startswith(f"{file1_key}:"):
                return env_var, line.split(":", 1)[1].strip()
    except Exception:
        pass
    return env_var, ""


def run_hermes(
    prompt: str,
    sample_id: str = "sample",
    attachments: list[pathlib.Path] | None = None,
    workdir: pathlib.Path | None = None,
    trace_path: pathlib.Path | None = None,
) -> dict:
    model = os.environ.get("GAIA_HERMES_MODEL", "openai/gpt-5.5")
    provider = os.environ.get("GAIA_HERMES_PROVIDER", "openrouter")
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

    # `hermes chat -q PROMPT --verbose`: one-shot programmatic mode that ALSO
    # emits the tool trace (web_search/browser calls) to stdout — required so the
    # leakage audit can see whether Hermes fetched benchmark answers. `-z` would be
    # cleaner (final line only) but blinds the audit (see METHODOLOGY §leakage).
    # `_extract_final_answer` still recovers "FINAL ANSWER:" from the noisy stream.
    cmd = ["hermes", "chat", "-q", full_prompt, "--verbose", "-m", model, "--provider", provider]
    env = dict(os.environ)
    env_var, key = _resolve_provider_key()
    if key:
        env[env_var] = key
    env.pop("PYTHONPATH", None)  # avoid shadowing hermes's own venv (installer warns about this)

    try:
        proc = subprocess.run(
            _bwrap_wrap(cmd), capture_output=True, text=True, timeout=timeout_sec,
            cwd=str(work), env=env, stdin=subprocess.DEVNULL,
        )
    except subprocess.TimeoutExpired as exc:
        return {"final_answer": "", "returncode": -1, "raw": "", "stderr_tail": f"TIMEOUT: {str(exc)[:300]}"}
    except Exception as exc:  # noqa: BLE001
        return {"final_answer": "", "returncode": -1, "raw": "", "stderr_tail": f"SPAWN ERROR: {type(exc).__name__}: {str(exc)[:300]}"}

    # verbose trace merges the tool activity (stdout) + hermes logs (stderr).
    result_text = proc.stdout or ""
    full_trace = result_text + "\n" + (proc.stderr or "")
    if trace_path is not None:
        try:
            trace_path.parent.mkdir(parents=True, exist_ok=True)
            trace_path.write_text(full_trace, encoding="utf-8")  # full, unclipped, for audit_leakage
        except Exception:
            pass
    return {
        "final_answer": _extract_final_answer(result_text),
        "returncode": proc.returncode,
        "raw": result_text[-4000:],  # tail: FINAL ANSWER lives at the end of the verbose stream
        "model": model,
        "provider": provider,
        "stderr_tail": (proc.stderr or "")[-2000:],
    }


@solver
def hermes_solver():
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
                while dest.exists() and dest.resolve() != a.resolve():  # collision-safe
                    dest = workdir / f"{a.stem}-{_n}{a.suffix}"
                    _n += 1
                if a.resolve() != dest.resolve():
                    dest.write_bytes(a.read_bytes())
            except Exception:
                pass
        result = run_hermes(prompt, sample_id=sample_id, attachments=attachments, workdir=workdir,
                            trace_path=sample_dir / "hermes_trace.txt")
        if getattr(state, "metadata", None) is None:
            state.metadata = {}
        state.metadata["hermes_raw"] = result.get("raw", "")
        state.metadata["hermes_stderr"] = result.get("stderr_tail", "")
        state.metadata["hermes_model"] = result.get("model", "")
        if getattr(state, "output", None) is None:
            state.output = SimpleNamespace(completion="")
        state.output.completion = result["final_answer"]
        return state

    return solve


if __name__ == "__main__":
    q = sys.argv[1] if len(sys.argv) > 1 else "What is 7 times 8?"
    print(json.dumps(run_hermes(q, sample_id="selftest"), indent=2)[:1500])
