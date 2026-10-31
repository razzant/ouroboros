"""The optional persistent runtime venv a launcher selects with ``--runtime-venv``.

A managed container's image interpreter starts the cold launcher; the body, its
dependency installs and every later source-launcher handoff run on a venv kept on a
persistent volume, so code and packages survive image recreation together. This
module only prepares and validates that venv; the launcher's existing bootstrap,
``install_deps`` and restart flow use it unchanged.

Preservation is the rule. A venv is created only at its final path, in an absent or
empty directory, after ``ouroboros-runtime-venv.json`` records the Python major/minor
and ``preparing``; it becomes ``ready`` only after the dependency install succeeded.
An interrupted preparation of the same Python retries additively. Nothing here runs
``--clear``, removes, renames or overwrites a directory: an unrelated non-empty
directory, or an established venv whose recorded Python, ``pyvenv.cfg`` or actual
interpreter disagree (an image that changed Python under it), is refused with the
exact recovery step. Package changes after an Update stay additive: Git rollback
restores code and its lock, not an exact package set.
"""

from __future__ import annotations

import json
import pathlib
import subprocess
import sys
from dataclasses import dataclass
from typing import Any, Callable

from ouroboros.platform_layer import IS_WINDOWS

MARKER_NAME = "ouroboros-runtime-venv.json"
SCHEMA_VERSION = 1
PREPARATION_DEPS_TIMEOUT_SEC = 1800.0  # a preparing venv installs the whole lock, not one Update's delta


class RuntimeVenvRefused(RuntimeError):
    """The requested runtime venv cannot be used without destroying or disguising something."""


@dataclass(frozen=True)
class RuntimeVenv:
    root: pathlib.Path
    python: str  # lexical: Python finds pyvenv.cfg beside the path it was started from
    bin_dir: pathlib.Path
    ready: bool


def _layout(root: pathlib.Path) -> tuple[pathlib.Path, pathlib.Path]:
    bin_dir = root / ("Scripts" if IS_WINDOWS else "bin")
    return bin_dir, bin_dir / ("python.exe" if IS_WINDOWS else "python")


def _major_minor() -> str:
    return f"{sys.version_info.major}.{sys.version_info.minor}"


def _pyvenv_version(root: pathlib.Path) -> str:
    try:
        lines = (root / "pyvenv.cfg").read_text(encoding="utf-8").splitlines()
    except OSError:
        return ""
    for line in lines:
        key, _, value = line.partition("=")
        if key.strip() in {"version", "version_info"}:
            return ".".join(value.strip().split(".")[:2])
    return ""


def _interpreter_version(python: pathlib.Path, run: Callable[..., Any]) -> str:
    try:
        probe = run([str(python), "-c", "import sys; print('%d.%d' % sys.version_info[:2], sys.prefix)"],
                    capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        return ""
    words = str(getattr(probe, "stdout", "") or "").split(None, 1)
    return words[0] if getattr(probe, "returncode", 1) == 0 and words else ""


def _write_marker(root: pathlib.Path, state: str) -> None:
    from ouroboros.utils import atomic_write_json, utc_now_iso

    try:
        atomic_write_json(root / MARKER_NAME, {"schema_version": SCHEMA_VERSION, "python": _major_minor(),
                                               "state": state, "updated_at": utc_now_iso()},
                          trailing_newline=True)
    except OSError as exc:
        raise _recovery(root, f"cannot record {state}: {exc}", modified=True) from exc


def _recovery(root: pathlib.Path, problem: str, *, modified: bool = False) -> RuntimeVenvRefused:
    effect = "Preparation may have created or changed files; they were retained" if modified else "It was left untouched"
    return RuntimeVenvRefused(
        f"Runtime venv {root} is not usable: {problem}. {effect}. To recover, stop "
        f"Ouroboros, move that directory aside yourself (or pass another absolute --runtime-venv), "
        f"then start again: a fresh venv is prepared and the current dependency lock installed into it. "
        f"Code and data are unaffected; packages installed only into the old venv are not carried over.")


def prepare_runtime_venv(path: "pathlib.Path | str", *, run: Callable[..., Any] = subprocess.run,
                         log: Any = None, base_python: str | None = None) -> RuntimeVenv:
    """Prepare a venv without overwrites; ``base_python`` must share the running launcher's ABI."""
    root = pathlib.Path(path)
    if not root.is_absolute():
        raise RuntimeVenvRefused(f"--runtime-venv must be an absolute path, got {path!r}")
    bin_dir, python = _layout(root)
    marker = root / MARKER_NAME
    current = _major_minor()
    if marker.is_file():
        try:
            recorded = json.loads(marker.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            raise _recovery(root, f"its {MARKER_NAME} is unreadable") from None
        if not isinstance(recorded, dict) or recorded.get("schema_version") != SCHEMA_VERSION:
            raise _recovery(root, f"its {MARKER_NAME} has an unknown schema")
        if recorded.get("python") != current:
            raise _recovery(root, f"it was created for Python {recorded.get('python')}, this image runs {current}")
        state = recorded.get("state")
    elif root.exists() and (not root.is_dir() or any(root.iterdir())):
        raise _recovery(root, "the path holds something Ouroboros did not create")
    else:
        try:
            root.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise _recovery(root, f"cannot create its directory: {exc}", modified=True) from exc
        _write_marker(root, "preparing")  # before any provisioning, so an interruption is recognizable
        state = "preparing"
    if state not in ("preparing", "ready"):
        raise _recovery(root, f"its recorded state {state!r} is unknown")
    if state == "preparing":
        # A retry must not overwrite evidence of a different ABI. The cfg is
        # written before Python's executable; either may be absent after interruption.
        cfg = root / "pyvenv.cfg"
        if cfg.exists() and _pyvenv_version(root) != current:
            raise _recovery(root, f"pyvenv.cfg reports Python {_pyvenv_version(root) or 'unavailable'}, expected {current}")
        if python.exists() and (not cfg.exists() or _interpreter_version(python, run) != current):
            raise _recovery(root, "its interpreter or pyvenv.cfg cannot prove the preparing Python version")
        if log is not None:
            log.info("Preparing runtime venv %s (Python %s)", root, current)
        # bin/python can exist before ensurepip completes. Re-run venv without
        # --clear: ensurepip repairs missing pip while installed packages stay.
        try:
            # An embedded shell's sys.executable may be the macOS app entry.
            created = run([base_python or sys.executable, "-m", "venv", str(root)],
                          capture_output=True, text=True, timeout=600)
        except (OSError, subprocess.SubprocessError) as exc:
            raise _recovery(root, f"preparation failed: {exc}", modified=True) from exc
        if getattr(created, "returncode", 1) != 0:
            output = str(getattr(created, "stderr", "") or getattr(created, "stdout", "") or "").strip()
            raise RuntimeVenvRefused(f"Creating runtime venv {root} failed: {output[-600:]}; "
                                     "the preparing venv is retried additively on the next start")
    for observed, label in ((_pyvenv_version(root), "pyvenv.cfg"), (_interpreter_version(python, run), "its interpreter")):
        if observed != current:
            raise _recovery(root, f"{label} reports Python {observed or 'unavailable'}, expected {current}",
                            modified=state == "preparing")
    return RuntimeVenv(root=root, python=str(python), bin_dir=bin_dir, ready=state == "ready")


def mark_ready(venv: RuntimeVenv) -> RuntimeVenv:
    """Record that the dependency lock landed; later starts skip preparation."""
    _write_marker(venv.root, "ready")
    return RuntimeVenv(root=venv.root, python=venv.python, bin_dir=venv.bin_dir, ready=True)
