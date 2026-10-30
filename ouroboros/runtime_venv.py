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

An Update's private environment for a bare interpreter is a venv of that interpreter with a pip of
its own (``create_with_pip``: ``ensurepip``, else the interpreter's own pip), and
``private_entry_interpreter`` states when the package-init hook runs an entry there.

The marker is also how a body tells it runs in such a venv (``recorded_state``): an
Update that replaces the macOS app prepares its packages in a clone beside this venv
and opens the new app with ``--runtime-venv`` at the same path
(``supervisor/update_native``). A venv of the installed app's Python has its clone
pointed at the new app's Python while they install and are checked; pointing
(``point_base``, ``restore_base``) rewrites only the given venv's ``bin/`` links and
``pyvenv.cfg``: the Update's private clone, never the venv a launcher runs. A venv of
another Python of the same version, which a launcher accepts as well, keeps that
Python (``own_python``).
"""

from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys
from dataclasses import dataclass
from typing import Any, Callable

from ouroboros.platform_layer import IS_WINDOWS

MARKER_NAME = "ouroboros-runtime-venv.json"
SCHEMA_VERSION = 1
PREPARATION_DEPS_TIMEOUT_SEC = 1800.0  # a preparing venv installs the whole lock, not one Update's delta
# A bare interpreter's private packages under ``<data>/state`` and the record naming the checkout they were made for
# (``update_prepare.private_environment_root``); the stdlib package-init hook restates both literals.
PRIVATE_NAME, PRIVATE_MARKER = "python-private", "ouroboros-private-environment.json"


def private_entry_interpreter(data_dir: "pathlib.Path | str", repo_dir: "pathlib.Path | str", python: str) -> str:
    """The private environment's interpreter when an entry of ``repo_dir`` started on ``python`` runs there (the
    package-init hook's rule, ``ouroboros/__init__.py``): that environment bound to this checkout, ``python`` the
    same real file it runs (on Windows, whose venv interpreter is a copied redirector, the Python its
    ``pyvenv.cfg`` names) and no venv's own. ``""`` otherwise."""
    root = os.path.join(data_dir, "state", PRIVATE_NAME)  # a read, spelled as the hook spells it (no writer here)
    try:
        with open(os.path.join(root, PRIVATE_MARKER), encoding="utf-8") as fh:
            checkout = json.load(fh)["repo_dir"]
    except (OSError, ValueError, KeyError, TypeError):
        return ""
    own = _layout(pathlib.Path(root))[1]
    runs = made_from(root) if IS_WINDOWS else str(own)
    if checkout != os.path.realpath(repo_dir) or os.path.normcase(os.path.realpath(runs)) != os.path.normcase(
            os.path.realpath(python)) or (pathlib.Path(python).parent.parent / "pyvenv.cfg").is_file():
        return ""
    return str(own)


def made_from(root: "pathlib.Path | str") -> str:
    """The Python the venv at ``root`` runs: its ``pyvenv.cfg``'s ``executable``, else ``home``'s interpreter."""
    try:
        with open(os.path.join(root, "pyvenv.cfg"), encoding="utf-8") as fh:
            config = {key.strip(): value.strip() for key, _, value in (line.partition("=") for line in fh)}
    except OSError:
        return ""
    name = "python.exe" if IS_WINDOWS else "python"
    return config.get("executable") or (os.path.join(config["home"], name) if config.get("home") else "")


class RuntimeVenvRefused(RuntimeError):
    """The requested runtime venv cannot be used without destroying or disguising something."""


def dependency_python(data_dir: "pathlib.Path | str", repo_dir: "pathlib.Path | str", python: str) -> str:
    """The interpreter whose packages a dependency install for entries of ``repo_dir`` on ``python`` may change.

    The private environment those entries run in; ``python`` itself where an Update never selected one (no
    ``python-private`` and no ``private_environment`` fact, which ``update_prepare.record_activation`` keeps
    once an Update adopted one), in a venv or for the app's interpreter. A selected private environment that
    is gone or does not serve these entries (its marker missing or unreadable, another checkout's or another
    interpreter's) refuses, and so does a gone one whose record exists but cannot be read (it may hold that
    fact): the interpreter's own packages are never the fallback for packages made private."""
    private = private_entry_interpreter(data_dir, repo_dir, python)
    root = os.path.join(data_dir, "state", PRIVATE_NAME)
    if private or (pathlib.Path(python).parent.parent / "pyvenv.cfg").is_file():
        return private or python
    unread = ""
    try:  # the retained record (update_prepare.ACTIVATION_REL), read as the hook reads: no import of its owner
        with open(os.path.join(data_dir, "state", "update_activation.json"), encoding="utf-8") as fh:
            selected = bool(json.load(fh).get("private_environment"))
    except FileNotFoundError:
        selected = False  # no record: no Update selected one
    except (OSError, ValueError, AttributeError) as exc:  # cut, garbled or refused: unknown, never "none selected"
        selected, unread = False, type(exc).__name__
    gone = not os.path.lexists(root)
    if gone and not (selected or unread):
        return python
    why = (f"does not serve {repo_dir} on {python} (its marker is missing, unreadable or names another checkout or "
           "interpreter)") if not gone else (
        "is missing and state/update_activation.json, which records whether an Update selected it, cannot be read "
        f"({unread})") if unread else "that an Update selected for this installation is missing"
    escape = "that record" if gone and unread else "private_environment from state/update_activation.json"
    raise RuntimeVenvRefused(
        f"the private Python environment {root} {why}; nothing is installed into that interpreter's own packages. "
        "Restart to update: an Update prepares a new one and keeps a broken one aside. To run on the interpreter's "
        f"own packages instead, remove {escape} and that directory")


@dataclass(frozen=True)
class RuntimeVenv:
    root: pathlib.Path
    python: str  # lexical: Python finds pyvenv.cfg beside the path it was started from
    bin_dir: pathlib.Path
    ready: bool


def _layout(root: pathlib.Path) -> tuple[pathlib.Path, pathlib.Path]:
    bin_dir = root / ("Scripts" if IS_WINDOWS else "bin")
    return bin_dir, bin_dir / ("python.exe" if IS_WINDOWS else "python")


def interpreter(root: "pathlib.Path | str") -> str:
    """The interpreter a launcher runs the body with from the venv at ``root`` (``RuntimeVenv.python``)."""
    return str(_layout(pathlib.Path(root))[1])


def point_base(root: "pathlib.Path | str", installed: str, staged: str) -> dict[str, str]:
    """Point the venv at ``root`` from one app's Python to the same files of another (a staged app).

    A venv runs the Python its ``bin/`` links and ``pyvenv.cfg``'s ``home`` name. CPython's
    ``venv`` links the name it was run by (``python3`` for an app's, the launcher's ``base_python``)
    to that Python's path and makes ``python``, ``python3`` and ``python3.X`` relative aliases of
    it; ``base_interpreter`` follows them. That Python must lie under ``installed``: each absolute
    ``bin/`` link and ``home``/``executable`` value under it then names the same file under
    ``staged`` (which must exist), and the aliases stay as they are. Returns each changed name with
    its exact previous text (``restore_base``). A venv of another Python (the marker records a
    version, not where that Python lives) is refused unchanged; a failure part-way writes back
    whatever had changed and raises.
    """
    root, bases = pathlib.Path(root), {installed.rstrip(os.sep), os.path.realpath(installed)}

    def moved(path: str) -> str:
        found = next((staged.rstrip(os.sep) + path[len(base):] for base in bases if path.startswith(base + os.sep)), "")
        return found if found and os.path.exists(found) else ""

    if not moved(base_interpreter(root)):
        raise RuntimeVenvRefused(f"the runtime venv {root} does not run the Python of {installed}, so it is not "
                                 "pointed at another app's")
    saved: dict[str, str] = {}
    folder = root / ("Scripts" if IS_WINDOWS else "bin")
    try:
        for link in sorted(folder.iterdir()):
            text = os.readlink(link) if link.is_symlink() else ""
            target = moved(text) if os.path.isabs(text) else ""
            if target:
                _replace_link(link, target)
                saved[f"{folder.name}/{link.name}"] = text
        config = root / "pyvenv.cfg"
        text = config.read_text(encoding="utf-8") if config.is_file() else ""
        lines = []
        for line in text.splitlines():
            key, sep, value = line.partition("=")
            target = moved(value.strip()) if sep and key.strip() in ("home", "executable") else ""
            lines.append(f"{key}{sep} {target}" if target else line)
        rewritten = "\n".join(lines) + ("\n" if text.endswith("\n") else "")
        if rewritten != text:
            _replace_text(config, rewritten)
            saved["pyvenv.cfg"] = text
    except BaseException:
        restore_base(root, saved)
        raise
    return saved


def base_interpreter(root: "pathlib.Path | str") -> str:
    """The Python the venv at ``root`` runs, as its links name it: ``bin/python`` followed through the venv's
    own relative aliases to the first absolute link. ``""`` for a copied interpreter, a loop, or a link that
    leaves the venv by a relative path."""
    root = pathlib.Path(root)
    path, seen = _layout(root)[1], set()
    while path.is_symlink() and path not in seen:
        seen.add(path)
        text = os.readlink(path)
        if os.path.isabs(text):
            return text
        path = pathlib.Path(os.path.normpath(path.parent / text))
        if root not in path.parents:
            return ""
    return ""


def own_python(root: "pathlib.Path | str", app: "pathlib.Path | str") -> str:
    """The real path of the Python the venv at ``root`` runs when that Python is not ``app``'s own, else ``""``.

    A launcher accepts a marked venv of any Python of its version (the marker records a version,
    not where that Python lives): one made on another Python is the owner's choice, which an
    Update of ``app`` keeps and never points at the new app's. ``""`` when the venv runs the
    app's Python, by its links or their real path (``point_base`` then decides), and when its
    links name no Python, a copied interpreter (``point_base`` refuses it).
    """
    python = base_interpreter(root)
    if not python:
        return ""
    real = os.path.realpath(python)
    bases = {str(app).rstrip(os.sep), os.path.realpath(app)}
    return "" if any(path.startswith(base + os.sep) for path in (python, real) for base in bases) else real


def restore_base(root: "pathlib.Path | str", saved: dict[str, str]) -> None:
    """Write back exactly what ``point_base`` changed (``OSError`` when it cannot)."""
    for name, text in sorted(saved.items()):
        if name == "pyvenv.cfg":
            _replace_text(pathlib.Path(root) / name, text)
        else:
            _replace_link(pathlib.Path(root) / name, text)


def system_site(root: "pathlib.Path | str", visible: bool) -> None:
    """Whether the venv at ``root`` sees its interpreter's own packages (``pyvenv.cfg``'s
    ``include-system-site-packages``); an Update's private environment hides them while pip installs."""
    config = pathlib.Path(root) / "pyvenv.cfg"
    lines = [line for line in config.read_text(encoding="utf-8").splitlines()
             if line.partition("=")[0].strip() != "include-system-site-packages"]
    _replace_text(config, "\n".join([*lines, f"include-system-site-packages = {str(visible).lower()}"]) + "\n")


# The runtime lock's pip (``requirements-runtime.lock``), fetched only for an interpreter that brings none.
PIP_WHEEL = {"url": "https://files.pythonhosted.org/packages/f3/6e/1736e5b4ae2b778ef2f81c47d797de9f891d4d8acb047a2"
                    "4ca37a60294dd/pip-26.2.1-py3-none-any.whl",
             "size_bytes": 1816632, "sha256": "71138adf1f4ca900cdb7d289c21b7494329f2332b6d85f0e1c42108c0384ed3e"}


def fetch_pip_wheel(directory: "pathlib.Path | str") -> pathlib.Path:
    """``PIP_WHEEL`` in ``directory``, bound to its reviewed size and digest (``verified_download``)."""
    from ouroboros.verified_download import fetch_exact_file, verify_exact_file

    label = "the runtime lock's pip wheel"
    verify = lambda path: verify_exact_file(  # noqa: E731
        path, size_bytes=PIP_WHEEL["size_bytes"], sha256=PIP_WHEEL["sha256"], code_prefix="pip_wheel", label=label)
    target = pathlib.Path(directory) / PIP_WHEEL["url"].rsplit("/")[-1]
    return fetch_exact_file(url=PIP_WHEEL["url"], destination=target, verify=verify, size_bytes=PIP_WHEEL["size_bytes"],
                            overflow_code="pip_wheel_size_mismatch", failure_code="pip_wheel_download_failed",
                            label=label, timeout_sec=120)


def _write_layout(root: pathlib.Path, python: str, version: str) -> None:
    """What ``venv --without-pip`` writes on POSIX, for an interpreter shipped without ``venv``: the ``pyvenv.cfg``
    CPython reads at startup, ``bin/python`` linked to ``python`` with its aliases, an empty ``site-packages``."""
    short = ".".join(version.split(".")[:2])
    (root / "lib" / f"python{short}" / "site-packages").mkdir(parents=True)
    (root / "bin").mkdir()
    (root / "bin" / "python").symlink_to(python)
    for alias in ("python3", f"python{short}"):
        (root / "bin" / alias).symlink_to("python")
    real = os.path.realpath(python)
    (root / "pyvenv.cfg").write_text(f"home = {os.path.dirname(real)}\ninclude-system-site-packages = false\n"
                                     f"version = {version}\nexecutable = {real}\n", encoding="utf-8")


def create_with_pip(root: "pathlib.Path | str", python: str, run: Callable[[list], dict],
                    fetch: Callable[[pathlib.Path], pathlib.Path] = fetch_pip_wheel) -> str:
    """A venv of exactly ``python`` at ``root`` with a pip of its own; returns how it got that pip.

    ``python -m venv`` brings pip through ``ensurepip`` (``"ensurepip"``). Where that fails (Debian and
    Ubuntu without ``python3-venv`` lack or disable ``ensurepip``; some builds ship no ``venv``), the venv is
    made without pip: by ``venv --without-pip`` or, on POSIX without ``venv``, as its files (``"layout+"``).
    Its pip then comes from the interpreter's own pip, visible inside for that one step
    (``"interpreter_pip"``), or, for an interpreter without pip, from ``PIP_WHEEL`` importing itself from
    the wheel as ``ensurepip`` runs its bundled one (``"pip_wheel"``; fetched inside ``root``, then removed).
    The venv must then run its own pip with the interpreter's packages hidden. ``run(cmd)`` answers
    ``{"returncode", "stdout", "stderr"}``. Nothing outside ``root`` is written; a failure removes ``root``
    and raises ``RuntimeVenvRefused`` with what each step said.
    """
    import shutil

    from ouroboros.verified_download import VerifiedDownloadError

    root = pathlib.Path(root)
    said = lambda done: str(done.get("stderr") or done.get("stdout") or "").strip()[-400:]  # noqa: E731
    made = run([python, "-m", "venv", str(root)])
    if made["returncode"] == 0:
        return "ensurepip"
    shutil.rmtree(root, ignore_errors=True)
    steps, how = [("venv", made)], ""

    def step(name: str, cmd: list) -> bool:
        steps.append((name, run(cmd)))
        return steps[-1][1]["returncode"] == 0

    try:
        if not step("venv --without-pip", [python, "-m", "venv", "--without-pip", str(root)]):
            shutil.rmtree(root, ignore_errors=True)
            version = [python, "-c", "import sys; print('%d.%d.%d' % sys.version_info[:3])"]
            if IS_WINDOWS or not step("its version", version):
                raise RuntimeVenvRefused("no venv")
            _write_layout(root, python, str(steps[-1][1]["stdout"]).strip())
            how = "layout+"
        system_site(root, True)
        if step("its own pip installing pip", [interpreter(root), "-m", "pip", "install", "-q", "--ignore-installed",
                                               "pip"]):
            how += "interpreter_pip"
        else:
            system_site(root, False)
            wheel = str(fetch(root / ".pip-wheel"))
            args = ["install", "-q", "--no-index", "--no-deps", wheel]
            code = ("import runpy, sys; sys.path = [%r] + sys.path; sys.argv[1:] = %r; runpy.run_module('pip', "
                    "run_name='__main__', alter_sys=True)" % (wheel, args))
            installed = step("the lock's pip wheel installing pip", [interpreter(root), "-c", code])
            shutil.rmtree(root / ".pip-wheel", ignore_errors=True)
            if not installed:
                raise RuntimeVenvRefused("no pip")
            how += "pip_wheel"
        system_site(root, False)
        if not step("the venv's own pip", [interpreter(root), "-m", "pip", "--version"]):
            raise RuntimeVenvRefused("no own pip")
        return how
    except (RuntimeVenvRefused, VerifiedDownloadError, OSError) as exc:
        shutil.rmtree(root, ignore_errors=True)
        failed = [] if isinstance(exc, RuntimeVenvRefused) else [(type(exc).__name__, {"returncode": "-",
                                                                                      "stderr": str(exc)})]
        raise RuntimeVenvRefused("a venv with a pip of its own could not be made: " + "; ".join(
            f"{name} rc={result['returncode']}: {said(result)}" for name, result in steps + failed)) from exc


def _replace_link(path: pathlib.Path, target: str) -> None:
    temporary = path.with_name(f".{path.name}.relink")
    os.symlink(target, temporary)
    os.replace(temporary, path)


def _replace_text(path: pathlib.Path, text: str) -> None:
    temporary = path.with_name(f".{path.name}.rewrite")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def recorded_state(root: "pathlib.Path | str") -> str:
    """``preparing`` or ``ready`` as this module recorded it for the venv at ``root``, else ``""``.

    Only a venv with such a record is one ``prepare_runtime_venv`` selects again at that path;
    any other directory there is refused, never adopted.
    """
    try:
        recorded = json.loads((pathlib.Path(root) / MARKER_NAME).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return ""
    if not isinstance(recorded, dict) or recorded.get("schema_version") != SCHEMA_VERSION:
        return ""
    state = recorded.get("state")
    return state if state in ("preparing", "ready") else ""


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
