"""The launch tuple and the work-root basis, settled before any path-freezing import (#1239).

APP_ROOT is the one parent a new configuration needs: repository, data, settings, PID and
port files and the three work roots default beneath it. Each explicit ``OUROBOROS_*`` value
still wins on its own and none is inferred from another path's parent, so an old split
(checkout, data and settings in three places) keeps working. Standard library only (plus the
stdlib leaf ``installation_stop``): entry points call it before an import freezes a path.

Two existing layouts keep what they had instead of being relocated in silence:

- A direct server given only APP_ROOT kept supervisor memory in ``~/Ouroboros/data`` while
  config read ``<APP_ROOT>/data/settings.json``. When that historical root shows a
  supervisor's traces, or cannot be examined, the start refuses and names both choices
  (``AmbiguousDataRoot``); otherwise the server uses ``<APP_ROOT>/data``.
- A work root without a saved or environment value follows its data root's basis, carried
  by the initialization witness (``state/state.initialized.json``, #1307): ``home`` keeps
  ``~/Ouroboros/<leaf>``, ``app_root`` uses ``<APP_ROOT>/<leaf>``. A witness without the
  field predates this rule and is ``home``; a new initialization records ``app_root``;
  Reset and legacy adoption carry the basis forward. An unreadable witness, or one lost
  beside an initialized state, is ``unknown``: while the two parents differ, a default work
  root refuses with both candidates (``WorkRootUnavailable``) rather than guess.

A server settles the basis as soon as it holds its data root (``pin_basis`` from
``server_entrypoint.admit_data_root``), before a boot step can write history or allocate a
folder, and exports it: workers, extension children and headless tasks inherit the
installation's basis even on another drive. Its state initialization pins it once more from
the witness it completed (``supervisor.state.init_state``), before the supervisor starts a
worker. Nothing here moves, copies or deletes a folder.
"""

from __future__ import annotations

import json
import os
import pathlib
import sys
from typing import Any, Callable, Dict, NamedTuple, Optional, Tuple

from ouroboros.installation_stop import confirm_absent  # stays there: that reader runs alone (-I -S)

Path = pathlib.Path

TUPLE_KEYS = ("OUROBOROS_APP_ROOT", "OUROBOROS_REPO_DIR", "OUROBOROS_DATA_DIR", "OUROBOROS_SETTINGS_PATH",
              "OUROBOROS_PID_FILE", "OUROBOROS_PORT_FILE")  # the launch tuple's overrides
WORK_ROOT_KEYS = ("OUROBOROS_SUBAGENT_WORKTREE_ROOT", "OUROBOROS_SUBAGENT_PROJECTS_ROOT",
                  "OUROBOROS_DELIVERABLES_ROOT")
BASIS_ENV = "OUROBOROS_WORK_ROOT_BASIS"
BASIS_KEY = "work_root_basis"  # the witness field
HOME_BASIS, APP_ROOT_BASIS, UNKNOWN_BASIS = "home", "app_root", "unknown"
BASES = (HOME_BASIS, APP_ROOT_BASIS, UNKNOWN_BASIS)

WITNESS_REL = Path("state") / "state.initialized.json"
STATE_COPIES = (Path("state") / "state.json", Path("state") / "state.last_good.json")
# Each is written only by a running supervisor or its tasks, never by bootstrap.
EVIDENCE_FILES = (
    Path("state") / "queue_snapshot.json",        # every supervisor start, after init
    Path("state") / "evolution_campaign.json",    # owner/agent evolution control
    Path("state") / "project_task_bindings.json",  # task -> project binding
)
EVIDENCE_NONEMPTY_FILES = (Path("logs") / "chat.jsonl",)
EVIDENCE_DIRS = (Path("task_results"),)
DIALOGUE_STORE_REL = Path("state") / "dialog.sqlite"  # dialogue_store.STORE_REL; the cutover creates it before init


class LaunchRoots(NamedTuple):
    app_root: Path
    repo_dir: Path
    data_dir: Path
    settings_path: Path
    pid_file: Path
    port_file: Path


def home_parent() -> Path:
    """The historical parent of every default: ``~/Ouroboros``."""
    return Path.home() / "Ouroboros"


def app_root(environ: Any = None) -> Path:
    env = os.environ if environ is None else environ
    return Path(env.get("OUROBOROS_APP_ROOT", home_parent()))


def data_dir(environ: Any = None) -> Path:
    env = os.environ if environ is None else environ
    return Path(env.get("OUROBOROS_DATA_DIR", app_root(env) / "data"))


def configured_roots(environ: Any = None) -> LaunchRoots:
    """The tuple ``config`` freezes and a launcher (desktop or packaged CLI) uses: each explicit
    value, else derived from APP_ROOT; the repository is the managed checkout there."""
    env = os.environ if environ is None else environ
    root, data = app_root(env), data_dir(env)
    return LaunchRoots(root, Path(env.get("OUROBOROS_REPO_DIR", root / "repo")), data,
                       Path(env.get("OUROBOROS_SETTINGS_PATH", data / "settings.json")),
                       Path(env.get("OUROBOROS_PID_FILE", root / "ouroboros.pid")),
                       Path(env.get("OUROBOROS_PORT_FILE", "").strip() or data / "state" / "server_port"))


def same_location(a: Any, b: Any) -> bool:
    """One place under two spellings: the same inode when both exist (links, case-insensitive
    volumes), else the same resolved spelling. Never decided by lowercasing a path."""
    try:
        return os.path.samefile(a, b)
    except OSError:
        pass
    if os.path.lexists(a) != os.path.lexists(b):
        return False
    return os.path.realpath(a) == os.path.realpath(b)


# --------------------------------------------------------------------------- the witness

def witness_path(drive_root: Any) -> Path:
    return Path(drive_root) / WITNESS_REL


def read_witness(drive_root: Any) -> Tuple[str, Dict[str, Any]]:
    """``("missing"|"ok"|"invalid"|"unreadable", witness)``; only proven absence is missing."""
    try:
        try:
            raw = witness_path(drive_root).read_bytes()
        except FileNotFoundError:
            confirm_absent(witness_path(drive_root))
            return "missing", {}
    except OSError:  # NotADirectoryError included: a file where ``state/`` belongs is not absence
        return "unreadable", {}
    try:
        data = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        return "invalid", {}
    if (not isinstance(data, dict) or data.get("phase") not in {"pending", "complete"}
            or not str(data.get("initialization_id") or "")):
        return "invalid", {}
    return "ok", data


def history_evidence(drive_root: Any, *, store_chat: Callable[[Path], bool],
                     store_errors: Tuple[type, ...] = ()) -> Tuple[str, str]:
    """``("none"|"present"|"unknown", where)`` over the witness owner's fixed, non-recursive
    vocabulary (``state_initialization.supervisor_evidence`` is this walk with the dialogue
    store's chat rows as ``store_chat``). Only proven absence is absence."""
    root = Path(drive_root)
    try:
        for rel in EVIDENCE_FILES:
            try:
                os.lstat(root / rel)
                return "present", str(rel)
            except FileNotFoundError:
                confirm_absent(root / rel)
        if store_chat(root):
            return "present", str(DIALOGUE_STORE_REL)
        # Before the first cutover an older body's nonempty live file is also
        # evidence. This is a stat only; initialization never imports history.
        for rel in EVIDENCE_NONEMPTY_FILES:
            try:
                if os.lstat(root / rel).st_size > 0:
                    return "present", str(rel)
            except FileNotFoundError:
                confirm_absent(root / rel)
        for rel in EVIDENCE_DIRS:
            try:
                with os.scandir(root / rel) as entries:
                    if next(entries, None) is not None:
                        return "present", str(rel)
            except FileNotFoundError:
                confirm_absent(root / rel)
    except OSError as exc:
        return "unknown", f"{type(exc).__name__} errno={exc.errno}"
    except store_errors as exc:
        return "unknown", str(exc)
    return "none", ""


class _StoreUnread(Exception):
    """A present dialogue store whose chat rows only its own reader can count."""


def _store_unread(root: Path) -> bool:
    if os.path.lexists(root / DIALOGUE_STORE_REL):
        raise _StoreUnread(f"{DIALOGUE_STORE_REL} is present and its rows were not read")
    return False


# --------------------------------------------------------------------------- the basis

def adopted_basis(state_identity: Any) -> str:
    """The basis a re-minted witness gets beside a readable state: a state without an
    initialization identity predates the witness and every APP_ROOT default (``home``); one
    with an identity lost its witness, so its basis cannot be known."""
    return UNKNOWN_BASIS if str(state_identity or "") else HOME_BASIS


def _state_identity(root: Path) -> Tuple[str, str]:
    """``("absent"|"read"|"unknown", identity-or-detail)`` of the copy the state owner adopts: the
    primary, else (missing, or not a nonempty object) the backup it recovers from."""
    invalid = ""
    for rel in STATE_COPIES:
        try:
            try:
                raw = (root / rel).read_bytes()
            except FileNotFoundError:
                confirm_absent(root / rel)
                continue
            data = json.loads(raw.decode("utf-8"))
        except OSError as exc:
            return "unknown", f"{rel}: {type(exc).__name__} errno={exc.errno}"
        except (UnicodeDecodeError, ValueError):
            data = None
        if not isinstance(data, dict) or not data:
            invalid = invalid or f"{rel} is not a JSON object"
            continue
        return "read", str(data.get("initialization_id") or "")
    return ("unknown", invalid) if invalid else ("absent", "")


def durable_basis(drive_root: Any, *,
                  evidence: Optional[Callable[[Any], Tuple[str, str]]] = None) -> Tuple[str, str]:
    """``(basis, why)`` from this data root's durable facts alone. ``evidence`` is the full
    vocabulary walk (the server passes ``supervisor_evidence``); without it a present dialogue
    store, whose rows only its reader can count, leaves the answer unknown."""
    root = Path(drive_root)
    status, witness = read_witness(root)
    if status == "ok":
        value = witness.get(BASIS_KEY, HOME_BASIS)
        return (value, "witness") if value in BASES else (UNKNOWN_BASIS, f"witness basis {value!r}")
    if status != "missing":
        return UNKNOWN_BASIS, f"{WITNESS_REL} is {status}"
    state, identity = _state_identity(root)
    if state == "unknown":
        return UNKNOWN_BASIS, identity
    if state == "read":
        basis = adopted_basis(identity)
        return basis, ("a state that predates the witness" if basis == HOME_BASIS
                       else f"initialization {identity} lost its witness")
    found, where = (evidence or (lambda r: history_evidence(r, store_chat=_store_unread,
                                                            store_errors=(_StoreUnread,))))(root)
    if found == "none":
        return APP_ROOT_BASIS, "a new data root"
    return UNKNOWN_BASIS, (f"supervisor history without state ({where})" if found == "present" else where)


def pin_basis(drive_root: Any, *, evidence: Optional[Callable[[Any], Tuple[str, str]]] = None) -> str:
    """Decide this held data root's basis once and export it to the process and its children.
    A server always re-derives it from durable facts: an inherited value never names its root."""
    basis, _why = durable_basis(drive_root, evidence=evidence)
    os.environ[BASIS_ENV] = basis
    return basis


def current_basis() -> Tuple[str, str]:
    """The launch's basis (pinned by the server, or inherited from it), else this process's
    data root's durable answer."""
    inherited = os.environ.get(BASIS_ENV, "")
    return (inherited, "launch") if inherited in BASES else durable_basis(data_dir())


class WorkRootUnavailable(ValueError):
    """A default work root cannot be chosen: its basis is unknown and the two candidates differ."""


def work_root_default(key: str, leaf: str) -> str:
    """The default of one work root (``key`` empty in settings and environment)."""
    basis, why = current_basis()
    historical = os.path.expanduser(os.path.join("~", "Ouroboros", leaf))
    configured = os.path.expanduser(str(app_root() / leaf))
    if basis == APP_ROOT_BASIS:
        return configured
    if basis == HOME_BASIS or same_location(app_root(), home_parent()):
        return historical
    raise WorkRootUnavailable(
        f"the default {leaf} folder is undecided: this data root's work-root basis is unknown "
        f"({why}) and the candidates differ: {historical} (earlier releases) and {configured} "
        f"(OUROBOROS_APP_ROOT). Set {key} to the folder to use; nothing was created or moved.")


# --------------------------------------------------------------------------- direct entry

class AmbiguousDataRoot(RuntimeError):
    """A direct server given only APP_ROOT whose historical data root may hold memory."""


def _historical_memory(root: Path) -> Tuple[str, str]:
    """Bounded, stat-only: the witness, a state copy, the fixed evidence, or any dialogue store.
    No SQLite open and no walk beyond these names."""
    try:
        for rel in (WITNESS_REL, *STATE_COPIES):
            try:
                os.lstat(root / rel)
                return "present", str(rel)
            except FileNotFoundError:
                confirm_absent(root / rel)
    except OSError as exc:
        return "unknown", f"{type(exc).__name__} errno={exc.errno}"
    return history_evidence(root, store_chat=lambda r: os.path.lexists(r / DIALOGUE_STORE_REL))


def direct_data_dir(environ: Any = None) -> Path:
    """A direct server's data root: explicit, else ``<APP_ROOT>/data`` unless that would leave
    historical memory unopened (``AmbiguousDataRoot`` names both exact choices)."""
    env = os.environ if environ is None else environ
    if "OUROBOROS_DATA_DIR" in env:
        return Path(env["OUROBOROS_DATA_DIR"])
    candidate, historical = app_root(env) / "data", home_parent() / "data"
    if not env.get("OUROBOROS_APP_ROOT") or same_location(candidate, historical):
        return candidate
    found, where = _historical_memory(historical)
    if found == "none":
        return candidate
    seen = f"holds a supervisor's history ({where})" if found == "present" else f"cannot be examined ({where})"
    keep = f"OUROBOROS_DATA_DIR={historical}"
    if "OUROBOROS_SETTINGS_PATH" not in env:
        keep += f" OUROBOROS_SETTINGS_PATH={candidate / 'settings.json'}"
    raise AmbiguousDataRoot(
        f"OUROBOROS_APP_ROOT is set without OUROBOROS_DATA_DIR, and {historical} {seen}: earlier "
        f"releases ran a direct server on it while reading settings from {candidate / 'settings.json'}. "
        f"Choose explicitly: keep that memory with {keep}, or start this APP_ROOT's own data with "
        f"OUROBOROS_DATA_DIR={candidate}. Nothing was opened or written.")


def direct_entry_environment(executing_repo: Any, environ: Any = None) -> Dict[str, str]:
    """What a direct server adds to its environment before its first project import: the
    executing checkout as the implicit repository, and a data root other than
    ``~/Ouroboros/data`` so config, the server and every child agree. Explicit values are
    never replaced; an ambiguous one raises ``AmbiguousDataRoot``."""
    env = os.environ if environ is None else environ
    added = {} if "OUROBOROS_REPO_DIR" in env else {"OUROBOROS_REPO_DIR": str(executing_repo)}
    data = direct_data_dir(env)
    if "OUROBOROS_DATA_DIR" not in env and not same_location(data, home_parent() / "data"):
        added["OUROBOROS_DATA_DIR"] = str(data)
    return added


def enter_direct(executing_repo: Any) -> Dict[str, str]:
    """``direct_entry_environment`` for an entry module, which applies it to ``os.environ``;
    an ambiguous start says why and exits 2."""
    try:
        return direct_entry_environment(executing_repo)
    except AmbiguousDataRoot as refusal:
        print(f"Ouroboros server not started: {refusal}", file=sys.stderr, flush=True)
        raise SystemExit(2) from None
