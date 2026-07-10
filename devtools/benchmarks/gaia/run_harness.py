#!/usr/bin/env python3.11
"""Reproducible launcher for the multi-harness GAIA comparison.

Runs ONE external agent harness on the official ``inspect_evals/gaia`` task with the
official GAIA scorer, so every harness is measured on the identical task set + metric.

Harnesses (see ``inspect_solver/``):
  * ``claude_code`` - Anthropic Claude Code CLI (``claude -p``); model via --model (e.g. claude-sonnet-4-6)
  * ``codex``       - OpenAI Codex CLI (``codex exec``);       model via --model (e.g. gpt-5.5)
  * ``null``        - zero-capability integrity probe (MUST score ~0)

For the **Ouroboros** harness use ``run_gaia.py`` instead (it starts a dedicated server).
See ``README_harness_compare.md`` for setup (brew install codex, API keys, Docker, dataset)
and the model-lock / integrity caveats.

Example (all levels, parallel 3):
    python3.11 run_harness.py --harness codex --model gpt-5.5 \
        --subset 2023_all --limit 165 --max-samples 3 --out-dir <dir>

Detach a multi-hour run with ``daemonize.py``:
    python3.11 daemonize.py <out> -- python3.11 run_harness.py --harness codex ... --out-dir <out>
"""

from __future__ import annotations

import argparse
import os
import pathlib
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[2]

# harness -> (solver "file@fn" relative to inspect_solver/, env var carrying the model id)
SOLVERS = {
    "claude_code": ("claude_code_solver.py@claude_code_solver", "GAIA_CLAUDE_MODEL"),
    "codex": ("codex_solver.py@codex_solver", "GAIA_CODEX_MODEL"),
    "hermes": ("hermes_solver.py@hermes_solver", "GAIA_HERMES_MODEL"),
    "null": ("null_solver.py@null_solver", None),
}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--harness", required=True, choices=sorted(SOLVERS))
    p.add_argument("--model", default="", help="model id for the harness (claude/codex)")
    p.add_argument("--subset", default="2023_all", help="2023_all | 2023_level1 | 2023_level2 | 2023_level3")
    p.add_argument("--split", default="validation")
    p.add_argument("--limit", type=int, default=165)
    p.add_argument("--sample-id", default="", help="comma-separated sample ids to run (resume a subset of remaining tasks; overrides --limit)")
    p.add_argument("--max-samples", type=int, default=3, help="inspect parallel samples (Docker sandboxes)")
    p.add_argument("--max-sandboxes", type=int, default=3)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--sample-timeout-sec", type=int, default=1200)
    p.add_argument("--max-turns", type=int, default=40, help="Claude Code --max-turns")
    a = p.parse_args(argv)

    out = pathlib.Path(a.out_dir).resolve(strict=False)
    out.mkdir(parents=True, exist_ok=True)
    rel, model_env = SOLVERS[a.harness]
    fname, fn = rel.split("@")
    solver = f"{HERE / 'inspect_solver' / fname}@{fn}"

    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO)
    env["GAIA_HARNESS_RUN_ROOT"] = str(out / "run_root")
    env["GAIA_SAMPLE_TIMEOUT_SEC"] = str(a.sample_timeout_sec)
    env["GAIA_CLAUDE_MAX_TURNS"] = str(a.max_turns)
    if model_env and a.model:
        env[model_env] = a.model

    cmd = [
        sys.executable, "-m", "inspect_ai", "eval", "inspect_evals/gaia",
        "-T", f"subset={a.subset}", "-T", f"split={a.split}",
        "--solver", solver,
        "--max-samples", str(a.max_samples),
        "--max-sandboxes", str(a.max_sandboxes),
        "--log-format", "json", "--log-dir", str(out / "inspect_logs"),
    ]
    sample_ids = str(a.sample_id or "").strip()
    if sample_ids:  # resume a specific subset of remaining tasks (mirrors run_gaia)
        cmd += ["--sample-id", sample_ids]
    else:
        cmd += ["--limit", str(a.limit)]
    print(f"[run_harness] {a.harness} model={a.model or '(default)'} subset={a.subset} -> {out}")
    return subprocess.run(cmd, env=env).returncode


if __name__ == "__main__":
    raise SystemExit(main())
