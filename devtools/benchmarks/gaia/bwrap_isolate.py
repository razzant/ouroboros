"""Filesystem isolation for GAIA solver subprocesses (bubblewrap answer-cache mask).

GAIA validation answers ship in the host HuggingFace/inspect cache (metadata.parquet
carries the `Final answer` column; per-task .jsonld files sit beside it). Every solver
runs its agent CLI (`codex exec`, `claude -p`, `hermes chat`) or `ouroboros run` as a
HOST subprocess with shell access, so the agent can read that answer key straight off
disk — a filesystem sibling of the web-lookup cheat the leakage audit already covers
(observed 2026-07-04: codex ran `find … gaia_dataset … jq <sample>.jsonld`).

This wraps the solver command in `bwrap` with the answer-cache directories masked by
empty tmpfs mounts, while leaving the rest of the filesystem, the network (web search),
and the CLI's own config/binaries intact. The inspect SCORER runs in the main process
OUTSIDE this wrapper and reads the dataset normally, so scoring is unaffected.

Mask-only (not a whitelist): `--dev-bind / /` passes the real FS through, then a
`--tmpfs` overlay hides each answer-cache path. Robust (does not break the CLIs) and
minimal. Disable with GAIA_BWRAP_ISOLATE=0.
"""
from __future__ import annotations

import os
import pathlib
import shutil

# Directories on this host that contain (or cache) the GAIA answer key. Masked with
# empty tmpfs inside the sandbox. Extend if a new answer-bearing cache path appears.
ANSWER_CACHE_DIRS: tuple[str, ...] = (
    "~/.cache/inspect_evals",                                   # gaia_dataset/**/*.jsonld + metadata*.parquet
    "~/.cache/huggingface/datasets",                            # arrow-backed gaia splits
    "~/.cache/huggingface/hub/datasets--gaia-benchmark--GAIA",  # raw parquet blobs
)


def isolation_enabled() -> bool:
    return os.environ.get("GAIA_BWRAP_ISOLATE", "1") != "0"


def _mask_dirs() -> list[str]:
    out: list[str] = []
    for d in ANSWER_CACHE_DIRS:
        p = pathlib.Path(d).expanduser()
        if p.exists():
            out.append(str(p))
    return out


def bwrap_prefix() -> list[str]:
    """Return the ``bwrap … --`` argv prefix to prepend to a solver command, or [] when
    isolation is disabled. Fails LOUDLY (SystemExit) if isolation is requested but bwrap
    is unavailable — silently running unprotected would defeat the integrity control."""
    if not isolation_enabled():
        return []
    bwrap = shutil.which("bwrap")
    if not bwrap:
        raise SystemExit(
            "GAIA_BWRAP_ISOLATE=1 but `bwrap` is not on PATH. Install bubblewrap or set "
            "GAIA_BWRAP_ISOLATE=0 to run without filesystem answer-cache isolation "
            "(the run would then rely on the post-hoc leakage audit alone)."
        )
    prefix = [bwrap, "--dev-bind", "/", "/", "--proc", "/proc"]
    for d in _mask_dirs():
        prefix += ["--tmpfs", d]
    prefix.append("--")
    return prefix


def wrap(cmd: list[str]) -> list[str]:
    """Prepend the bwrap answer-cache mask to a solver command (no-op when disabled)."""
    return bwrap_prefix() + list(cmd)


if __name__ == "__main__":  # self-test: prove the answer cache is invisible inside the wrap
    import subprocess
    probe = "; ".join(
        f"echo {pathlib.Path(d).expanduser()}: $(ls {pathlib.Path(d).expanduser()} 2>/dev/null | wc -l)"
        for d in ANSWER_CACHE_DIRS
    )
    print("enabled:", isolation_enabled(), "| mask dirs:", _mask_dirs())
    subprocess.run(wrap(["bash", "-lc", probe]), check=False)
