"""Which Git Ouroboros's own processes run: the person's working Git, else the copy inside the app.

``select_git`` runs once at process entry (``launcher.check_git``, ``packaged_cli``), before anything runs Git, and
its answer is the gate: ``"missing"`` shows how to install Git. A Git counts only when a bounded ``git --version``
answers in the environment it would run with (a broken install or a dead shim is no Git). One search with one time
budget (``_SEARCH_SECONDS``) judges every candidate, Apple's ``xcode-select`` and the bundled copy; each program gets
at most ``_PROBE_SECONDS`` of it, so one that hangs leaves time for the next. A Git the budget left unjudged is
unverified, never refused: ``GitUnverified`` is raised and nothing is applied, as when the caller's runner is stopped
(the launcher's bootstrap runner raises ``InterruptedError`` on Quit or Panic). The choice is kept in
``OUROBOROS_GIT`` for app-owned calls using ``selected_git()`` directly (``launcher_bootstrap``,
``supervisor/git_ops``, ``update_source``, ``update_carriers``, ``store_switch``). Callers pass ``prepend_dirs`` for a
GUI/server process tree whose normal augmented PATH puts common tool folders ahead of inherited entries
(``launcher.check_git``), or ``extra_dirs`` for a terminal CLI whose inherited PATH stays first (``packaged_cli``).
The first working Git on that search path wins, and PATH and the person's Git settings stay in that healthy order, so
their system configuration, credential helpers, Git LFS, certificates and non-Git tools stay theirs. When another Git
stands before the chosen one (an unusable one, Apple's stub without a working developer Git, a relative entry a child
running in another folder would read anew), its absolute folder joins just before that entry, every folder before
keeping its place; when PATH holds no Git at all, its absolute folder is appended so non-Git tools already on PATH
keep their priority while every later bare ``git`` reaches it (including ``supervisor/git_ops_reset``'s bare ``git``
steps during regular Update, protected by ``_reach`` and ``keep_reached_git``). Every path kept is absolute: the file
that answered ``--version``. Only when none works is the bundled copy (``git-standalone/`` beside the other bundled
runtimes, staged by ``scripts/download_git_standalone.py``) put first, with the paths its build cannot find by itself;
the server, workers and the model's shell inherit it, the person's own terminal never sees it. Where a Git was chosen,
a common tool folder a process of that tree adds later (``bootstrap_process_path``) never goes ahead of the Git its
PATH reaches (``keep_reached_git``). On macOS ``/usr/bin/git`` is Apple's stub: when ``xcode-select -p`` names a
developer folder whose Git runs, that Git is chosen and PATH stays, the stub forwarding a bare ``git`` there (its
developer folder put first would also change ``cc`` and ``clang``); the stub is never executed to find out (without
the tools it opens Apple's install dialog). Every entry decides again from the facts; a copy of ours that a parent
process put on PATH is not taken for the person's. Stdlib only, Python 3.10: an older app's frozen interpreter imports
this file after a source handoff.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Callable, Iterable, MutableMapping, Optional

BUNDLED_GIT_DIR = "git-standalone"
APPLE_GIT_STUB = "/usr/bin/git"
SELECTED_GIT_ENV = "OUROBOROS_GIT"  # the chosen executable, absolute: app-owned Git calls run it by this path
_RELOCATION_VARIABLES = ("GIT_EXEC_PATH", "GIT_TEMPLATE_DIR")
_SEARCH_SECONDS = 20.0  # the whole search: the person's Gits, xcode-select and the bundled copy
_PROBE_SECONDS = 10.0  # one program's share of it


class GitUnverified(TimeoutError):
    """The search's time ran out before this Git (its path, ``str()``) could be judged: no choice, not "missing"."""


def bundled_git_root(bundle_dir, platform: str = sys.platform) -> Optional[Path]:
    """The staged copy under ``bundle_dir``, absolute, when its entry exists, else None."""
    if not bundle_dir:
        return None
    root = Path(os.path.abspath(Path(bundle_dir) / BUNDLED_GIT_DIR))
    return root if _bundled_entry(root, platform).is_file() else None


def _prepend(folder, env: MutableMapping[str, str]) -> str:
    return os.pathsep.join(part for part in (str(folder), env.get("PATH", "")) if part)


def bundled_environment(root: Path, env: MutableMapping[str, str], platform: str = sys.platform) -> dict:
    """The variables the copy at ``root`` needs on top of ``env``. MinGit finds its own tree from ``cmd\\git.exe``;
    the macOS and Linux build is compiled for ``/`` and is told its exec path and templates."""
    if platform == "win32":
        return {"PATH": _prepend(root / "cmd", env)}
    return {"PATH": _prepend(root / "bin", env), "GIT_EXEC_PATH": str(root / "libexec" / "git-core"),
            "GIT_TEMPLATE_DIR": str(root / "share" / "git-core" / "templates")}


def _inside(value: str, root: Optional[Path]) -> bool:
    if root is None or not value:
        return False
    try:
        resolved, own = Path(value).resolve(), root.resolve()
    except (OSError, RuntimeError, ValueError):  # RuntimeError: a symlink loop before Python 3.13
        return False
    return resolved == own or own in resolved.parents


def _bounded(run: Callable, argv: list, git: str, deadline: float, **kwargs):
    """``run(argv)`` within one program's share of the search: its result, or None when it failed or did not answer
    in that share (a timed-out child is killed and only that child is waited for). ``GitUnverified(git)`` when the
    search's own time cut it short or was gone; a stopped runner's ``InterruptedError`` passes through."""
    seconds = min(_PROBE_SECONDS, deadline - time.monotonic())
    if seconds <= 0:
        raise GitUnverified(git)
    try:
        return run(argv, timeout=seconds, **kwargs)
    except InterruptedError:
        raise
    except subprocess.TimeoutExpired:
        if seconds < _PROBE_SECONDS:
            raise GitUnverified(git) from None
    except (OSError, ValueError, subprocess.SubprocessError):
        pass
    return None


def _apple_developer_git(env: MutableMapping[str, str], run: Callable, deadline: float) -> Optional[str]:
    """The developer Git Apple's stub would forward to, from ``xcode-select -p`` (no dialog); never the stub."""
    answer = _bounded(run, ["/usr/bin/xcode-select", "-p"], APPLE_GIT_STUB, deadline, capture_output=True,
                      env=dict(env))
    folder = os.fsdecode(answer.stdout or b"").strip() if answer is not None and answer.returncode == 0 else ""
    git = os.path.join(folder, "usr", "bin", "git") if folder else ""
    return os.path.abspath(git) if git and os.path.isfile(git) else None


def _bundled_entry(root: Path, platform: str) -> Path:
    return root / "cmd" / "git.exe" if platform == "win32" else root / "bin" / "git"


def _git_in(folder: str, platform: str) -> Optional[str]:
    """The Git a bare ``git`` finds in this PATH entry, absolute (a relative entry is read from the current folder):
    on Windows ``git.exe`` there, never the current folder."""
    if not folder:
        return None
    if platform == "win32":
        candidate = os.path.join(folder, "git.exe")
        return os.path.abspath(candidate) if os.path.isfile(candidate) else None
    found = shutil.which("git", path=folder)
    return os.path.abspath(found) if found else None


def _answers(git: str, env: MutableMapping[str, str], run: Callable, deadline: float) -> bool:
    """A bounded ``git --version`` (``_bounded``) read as bytes from a file: no pipe a stray descendant can hold open,
    no decoding to fail, only the first bytes looked at."""
    hidden = {"creationflags": getattr(subprocess, "CREATE_NO_WINDOW", 0)} if sys.platform == "win32" else {}
    with tempfile.TemporaryFile() as out:
        done = _bounded(run, [git, "--version"], git, deadline, stdin=subprocess.DEVNULL, stdout=out,
                        stderr=subprocess.DEVNULL, env=dict(env), **hidden)
        out.seek(0)
        return done is not None and done.returncode == 0 and out.read(64).startswith(b"git version")


def _search(folders: Iterable[str], env: MutableMapping[str, str], *, platform: str, run: Callable, host: Callable,
            deadline: float) -> tuple[Optional[str], str]:
    """The first Git in ``folders``, folder by folder, that answers in the environment it would run with (its folder
    first on PATH, ``host`` applied: the Linux desktop's saved loader values), each distinct file judged once, and the
    real path of the file that led to it. For Apple's stub the answer is the developer Git, never the stub."""
    judged: dict = {}
    for folder in folders:
        if (candidate := _git_in(folder, platform)) is None:
            continue
        if (real := os.path.realpath(candidate)) not in judged:
            if platform == "darwin" and real == APPLE_GIT_STUB:
                candidate = _apple_developer_git(env, run, deadline)
            judged[real] = candidate if candidate and _answers(
                candidate, host({**env, "PATH": _prepend(os.path.dirname(candidate), env)}), run, deadline) else None
        if judged[real] is not None:
            return judged[real], real
    return None, ""


def _first_git(parts: list, platform: str) -> tuple[int, str]:
    """The index of the first entry of ``parts`` holding a Git and that Git's real path; ``(-1, "")`` when none does."""
    return next(((index, os.path.realpath(git)) for index, folder in enumerate(parts)
                 if (git := _git_in(folder, platform))), (-1, ""))


def _reach(found: str, via: str, path: str, platform: str) -> str:
    """``path`` on which a bare ``git`` reaches ``found``: as it is when its first Git is ``via`` (the file that led to
    ``found``; Apple's stub forwards to the developer Git it named) in an absolute folder; with ``found``'s folder just
    before that first Git's entry when another or relative Git stands earlier, keeping every folder before it in place;
    or appended when ``path`` holds no Git (preserving the priority of non-Git tools already on ``path``)."""
    parts = path.split(os.pathsep) if path else []
    index, real = _first_git(parts, platform)
    if index >= 0 and real == via and os.path.isabs(parts[index]):
        return path
    if index < 0:
        return os.pathsep.join([*parts, os.path.dirname(found)])
    return os.pathsep.join([*parts[:index], os.path.dirname(found), *parts[index:]])


def keep_reached_git(added: list, parts: list, env: MutableMapping[str, str] = os.environ,
                     platform: str = sys.platform) -> list:
    """PATH entries with the folders ``added`` (``platform_layer.bootstrap_process_path``: common tool folders a
    process lacks) ahead of ``parts``, as always where no Git was chosen. Where one was (``OUROBOROS_GIT``, inherited by
    the server, its workers and their shells), a folder holding a Git other than the one ``parts`` reaches first goes
    right after that Git's entry instead, its other programs with it, so adding folders never changes which Git a bare
    ``git`` runs. The files decide; nothing is run."""
    index, reached = _first_git(parts, platform) if env.get(SELECTED_GIT_ENV) else (-1, "")
    other = [folder for folder in added if index >= 0 and (git := _git_in(folder, platform))
             and os.path.realpath(git) != reached]
    return [*(folder for folder in added if folder not in other), *parts[:index + 1], *other, *parts[index + 1:]]


def selected_git(env: MutableMapping[str, str] = os.environ) -> str:
    """What app-owned Git calls run: the executable ``select_git`` chose, else a bare ``git``."""
    return env.get(SELECTED_GIT_ENV) or "git"


def _augment(path: str, prepend_dirs: Iterable[str], root: Optional[Path], platform: str) -> str:
    """``path`` with non-empty ``prepend_dirs`` (not already on ``path`` or inside ``root``) ahead of it."""
    if not prepend_dirs:
        return path
    existing = [part for part in path.split(os.pathsep) if part]
    seen = {part.lower() if platform == "win32" else part for part in existing}
    prefix: list[str] = []
    for folder in prepend_dirs:
        if not folder or _inside(folder, root):
            continue
        key = folder.lower() if platform == "win32" else folder
        if key not in seen:
            seen.add(key)
            prefix.append(folder)
    return os.pathsep.join([*prefix, *existing]) if prefix else path


def select_git(bundle_dir, env: MutableMapping[str, str] = os.environ, *, prepend_dirs: Iterable[str] = (),
               extra_dirs: Iterable[str] = (), platform: str = sys.platform, run: Callable = subprocess.run,
               host: Optional[Callable] = None) -> str:
    """Decide and apply in ``env``: ``"system"`` (the person's Git, the one a bare ``git`` now reaches),
    ``"bundled"`` (the app's copy first on PATH) or ``"missing"`` (neither works; the caller says how to install
    Git). The chosen executable is also kept, absolute, in ``OUROBOROS_GIT``. ``prepend_dirs`` are prepended ahead
    of inherited PATH before searching (for GUI/server trees whose bootstrap prepends common tool folders) and kept
    when a Git is selected; ``extra_dirs`` are searched only after inherited PATH (for terminal CLIs). ``host`` turns
    an environment into the one the chosen Git will actually run with (``desktop_media.host_environment`` on the Linux
    desktop). Only values a selection of this same copy added are ever removed. ``GitUnverified`` (the search's time
    ran out first) and ``run``'s ``InterruptedError`` (a stopped runner) leave ``env`` as it was."""
    root = bundled_git_root(bundle_dir, platform)
    wanted = dict(env)
    wanted.pop(SELECTED_GIT_ENV, None)
    if "PATH" in env:
        wanted["PATH"] = os.pathsep.join(part for part in env["PATH"].split(os.pathsep) if not _inside(part, root))
    for name in _RELOCATION_VARIABLES:
        if _inside(env.get(name, ""), root):
            del wanted[name]
    host, path, deadline = host or dict, wanted.get("PATH", ""), time.monotonic() + _SEARCH_SECONDS
    search_path = _augment(path, prepend_dirs, root, platform)
    probe_env = {**wanted, "PATH": search_path} if search_path else wanted
    found, via = _search([*search_path.split(os.pathsep), *extra_dirs], probe_env, platform=platform, run=run,
                         host=host, deadline=deadline)
    choice = "missing" if found is None else "system"
    if found is not None:
        wanted["PATH"] = _reach(found, via, search_path, platform)
    elif root is not None:
        entry = str(_bundled_entry(root, platform))
        bundled = {**wanted, **bundled_environment(root, probe_env, platform)}
        if _answers(entry, host(bundled), run, deadline):
            choice, found, wanted = "bundled", entry, bundled
    if found is not None:
        wanted[SELECTED_GIT_ENV] = found
    for name in [name for name in env if name not in wanted]:
        del env[name]
    for name, value in wanted.items():
        if env.get(name) != value:
            env[name] = value
    return choice
