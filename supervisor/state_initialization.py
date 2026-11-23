"""Is this data root new? The one durable answer ``init_state`` consults (#1307).

Two missing ``state.json`` copies used to mean "fresh install", so a lost or
unreadable state minted a new owner slot, session and cleared Stop. First
initialization is now positive: ``state/state.initialized.json`` records ONE
initialization identity (``pending`` before the first state is written,
``complete`` after), written by supervisor boot (or an isolated benchmark's
explicit seed before boot) before admission. Owner Reset writes ``resetting``
on a confirmed request. Its departing process hands over only once nothing of the profile can still write
(its own stop, and the custody owners' records of delegated runs and processes), and the next exclusive
admission records ``pending`` after setting the old Project shelves aside, clearing the profile and
publishing the registry. Reset keeps what belongs to the installation (``_STATE_KEPT``) and never touches
Panic's records. The witness holds identity, phase and the work-root basis
(``launch_roots``: a first initialization records ``app_root``, Reset and
adoption carry theirs) — no permissions, no history.

With both copies absent the decision is:

- a ``complete`` witness: initialized before, state LOST -> refuse (unavailable);
- a ``pending`` witness: an interrupted initialization -> finish the SAME id,
  unless supervisor-run evidence appeared since (then refuse);
- ``resetting`` without the one-shot hand-off: hold the identity and basis;
  another explicit request is needed before cleanup;
- no witness: create only when a fixed, non-recursive set of evidence that a
  supervisor already ran here is positively absent (a legacy root whose state
  was lost is recovery, not first boot);
- an unreadable witness or evidence probe: refuse; unknown never mints identity.

Absence is proven, not inferred from an errno: Windows reports a file where
``state/`` belongs as ERROR_PATH_NOT_FOUND, the same ``FileNotFoundError`` as a
missing directory, so ``confirm_absent`` checks the entry and its ancestors.

Bootstrap scaffolding that may precede init — empty ``state``/``logs``/
``memory``/``task_results`` directories, ``settings.json`` from onboarding,
the usage ledger import, a benchmark sentinel — is deliberately not evidence.
A root whose every trace was deleted is indistinguishable from a new one: that
is a disclosed residual, not proof of historylessness.

What a created first state is born with is ``first_state`` (a re-mint after loss
and an owner Reset included): update notices a fresh install owes nothing, the
consciousness switch its setup left on, and the first meeting its setup's Start opens.
"""

from __future__ import annotations

import errno
import hashlib
import json
import logging
import os
import pathlib
import re
import stat
import uuid
from typing import Any, Callable, Dict, Tuple

# The witness reader and its fixed evidence vocabulary live in the stdlib leaf, so a direct entry
# can read them before its first path-freezing import (#1239); this module owns the writes.
from ouroboros.installation_stop import confirm_absent, panic_input
from ouroboros.launch_roots import (
    APP_ROOT_BASIS,
    BASIS_KEY,
    RESETTING,
    adopted_basis,
    durable_basis,
    history_evidence,
    read_witness,
    reset_receipt,
    witness_path,
)
from ouroboros.utils import assert_test_data_path, fsync_folder, utc_now_iso, write_bytes_atomic

log = logging.getLogger(__name__)
_requested_reset: str | None = None  # only the generation that accepted the owner click can hand it off
RESET_CONFIRMATION = "reset_all_data"  # ``ResetRequest.confirm``: what a POST names to request Reset
# Where one Reset keeps the previous profile's Project shelves, relative to the data root. The name
# is recorded in the witness before the first move: a repeated request must reuse it, never re-derive it.
SET_ASIDE_KEY = "projects_set_aside"
_SET_ASIDE = re.compile(r"projects/\.before-reset-\d{8}-[0-9a-f]{8}")
_NO_EXCLUSIVE_RENAME = {errno.ENOSYS, errno.ENOTSUP, errno.EOPNOTSUPP, errno.EINVAL}
# What Reset removes whole, in its order; ``state/`` is cleared except ``_STATE_KEPT``. One Reset's passes
# record each declared removal as pending before its effect, where ``state/`` clearing began and the Panic
# input its owner confirmed over; the receipt proves the absence that completes each pending record.
_REMOVED_FOLDERS = ("memory", "archive", "task_results", "uploads")
_REMOVED_FILE = "settings.json"  # the one file Reset removes whole
_REMOVED = ("logs", *_REMOVED_FOLDERS, _REMOVED_FILE)
_REMOVED_KEY, _CLEARING_KEY, _PANIC_KEY = "reset_removed", "reset_clearing_state", "panic_input"
# The irreversible boundary, saved before the first removal starts: a pass cut inside that step has removed
# something no completed step names, so what the passes removed is never the proof that nothing was.
_EFFECT_KEY = "reset_removing"
_REPLACED_KEY = "replaced"  # the witness the request replaced: what a Panic before any effect returns to
_PROOF_KEY = "reset_working_folders"  # the working-folder proof saved before the first effect
_CLEANUP_DONE_KEY = "reset_cleanup_done"  # same-operation cleanup receipt saved before registry publication
_NO_REASON = "Reset stopped before recording why."
# What clearing ``state/`` leaves beside the witness and the admission lock: Emergency Stop's records, then the
# installation's identity, its process custody and its runtimes. Each has its own writer and lifetime
# (PERSISTENCE), outlives a profile, and may belong to a process that is still running: the shared engine
# daemon and its other clients, this server and its launcher.
_STATE_KEPT = (
    "panic_stop.flag", "panic_sweep_pending.json",
    "install.json", "installation_scope.json",
    "process_ledger.jsonl", "owned_processes.json", "owned_processes.pending", "task_scopes",
    "workspace_executor_processes", "extension_companions.json", "server_process.json",
    "cx", "python-userbase", "body_adoption",
)
# The open journals the logs clear retains: Windows keeps these regular files held open by the
# desktop's output copier and the rotating handlers; the receipt names the ones that stayed.
_ACTIVE_LOGS = ("agent_stdout.log", "launcher.log", "server.log")


def _write(drive_root: Any, witness: Dict[str, Any]) -> None:
    path = witness_path(drive_root)
    write_bytes_atomic(path, (json.dumps({"schema_version": 1, **witness}, ensure_ascii=False, indent=2)
                              + "\n").encode("utf-8"), fsync=True)


def mark_pending(drive_root: Any, *, origin: str, work_root_basis: str = "") -> str:
    """Record a fresh identity before its first state writes."""
    initialization_id = uuid.uuid4().hex
    _write(drive_root, {"initialization_id": initialization_id, "phase": "pending",
                        "origin": str(origin), "created_at": utc_now_iso(),
                        **({BASIS_KEY: work_root_basis} if work_root_basis else {})})
    return initialization_id


def supervisor_evidence(drive_root: Any) -> Tuple[str, str]:
    """``("none"|"present"|"unknown", path)``: did a supervisor already run here? The fixed
    walk is ``launch_roots.history_evidence``; only this reader can count the store's chat rows."""
    from ouroboros import dialogue_store as ds

    return history_evidence(drive_root, store_chat=lambda root: ds.present(root) and ds.stream_end(root, "chat") > 0,
                            store_errors=(ds.StoreError,))


def _require_plain_directory(path: pathlib.Path) -> bool:
    try:
        info = path.lstat()
    except FileNotFoundError:
        confirm_absent(path)
        return False
    if (not stat.S_ISDIR(info.st_mode)
            or getattr(info, "st_file_attributes", 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT):
        raise NotADirectoryError(f"Reset requires an ordinary directory: {path}")
    return True


def _remove_for_reset(path: pathlib.Path, remove) -> bool:
    try:
        os.lstat(path)
    except FileNotFoundError:
        confirm_absent(path)
        return False
    remove(path)
    confirm_absent(path)
    return True


def _remove_tree_for_reset(path: pathlib.Path) -> None:
    """The shared permission-granting removal (``force_remove.force_rmtree``): read-only files this
    profile owns go too, never through a link. It is best effort by design — the caller's
    ``confirm_absent`` proves the actual result, so a swallowed failure still interrupts truthfully."""
    from ouroboros.force_remove import force_rmtree

    force_rmtree(path)


def _unlink_for_reset(path: pathlib.Path) -> None:
    """One unlink; a read-only regular file gets its write bit back once and is retried (the shared
    ``force_remove`` idea), never through a link, never a directory."""
    try:
        path.unlink()
    except PermissionError:
        info = path.lstat()
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
            raise
        os.chmod(path, stat.S_IMODE(info.st_mode) | stat.S_IWRITE | stat.S_IREAD)
        path.unlink()


def _clear_reset_directory(path: pathlib.Path, *, entries, keep=(), active_logs=(),
                           remove_if_empty=False) -> tuple[bool, list[str]]:
    """Consume the guarded entry list. A new, unexpected survivor interrupts the clear."""
    if not _require_plain_directory(path):
        return False, []
    for entry in entries:
        if entry.name in keep:
            continue
        try:
            mode = entry.lstat().st_mode
        except FileNotFoundError:
            confirm_absent(entry)
            continue
        try:
            if stat.S_ISDIR(mode):
                _remove_tree_for_reset(entry)
            else:
                _unlink_for_reset(entry)
        except FileNotFoundError:
            confirm_absent(entry)
        except PermissionError:
            # The desktop's output copier and both rotating log handlers keep
            # regular files open on Windows. None is a dialogue import source.
            if entry.name not in active_logs or not stat.S_ISREG(mode):
                raise
    survivors = list(path.iterdir())
    for entry in survivors:
        if entry.name not in keep and not (entry.name in active_logs and stat.S_ISREG(entry.lstat().st_mode)):
            raise OSError(f"Reset could not clear {entry}")
    if survivors:
        return False, sorted(entry.name for entry in survivors)
    if not remove_if_empty:
        return False, []
    return _remove_for_reset(path, pathlib.Path.rmdir), []


def confirmation_refusal(body: Any) -> str:
    """Why a POST does not request Reset, else "". Only a body that names the confirmation requests it: a client
    written for the earlier answer (erasure reported in the reply) sends none, and would read ``requested`` as a
    failure while the Reset went on."""
    if isinstance(body, dict) and body.get("confirm") == RESET_CONFIRMATION:
        return ""
    return "Reset was not requested: this page is older than Ouroboros. Reload it, then confirm Reset all data again."


def _panic_input_now(drive_root: pathlib.Path) -> dict:
    """Observe Panic's latch and records; Reset never acknowledges, moves or removes them. Returns the identity
    of its pending input (``{}`` without one). Processes Panic could not confirm stopped block every step, and so
    does an input that cannot be read."""
    from ouroboros.process_scope import panic_sweep_pending_path
    from ouroboros.server_control import panic_termination_requested

    if panic_termination_requested():
        raise OSError("Panic interrupted Reset")
    try:
        confirm_absent(panic_sweep_pending_path(drive_root))
        return panic_input(drive_root)
    except (OSError, UnicodeError) as exc:
        raise OSError("Reset cannot proceed until the outstanding Panic records are settled") from exc


def _require_no_new_panic(drive_root: pathlib.Path, reset: dict) -> None:
    """A Panic that arrives after the owner confirmed interrupts Reset at every later step, a mere boot included
    (``_withdrawn_by_panic`` then returns the profile when nothing was cleared yet). An input already there at that
    confirmation stays where it is: a profile held for Reset has no controls left to take it, so the next
    supervisor start acknowledges it in the new profile's controls, as it does after any Panic
    (``worker_chat_lane.auto_resume_after_restart``)."""
    current = _panic_input_now(drive_root)
    if current and current != reset.get(_PANIC_KEY):
        raise OSError("Panic interrupted Reset")


def _withdrawn_by_panic(drive_root: pathlib.Path, reset: dict) -> bool:
    """A Panic that arrived after the owner confirmed stops a Reset that has had no effect for good: the witness
    its request replaced returns, so this start is the ordinary one after Panic and acknowledges it in the
    profile's own controls. An absent or empty set-aside folder proves no shelf is still moved. A removal that
    may have started, an uncertain move, or no prior witness leaves the Reset interrupted."""
    replaced = reset.get(_REPLACED_KEY)
    if (not isinstance(replaced, dict) or not str(replaced.get("initialization_id") or "")
            or replaced.get("phase") not in ("pending", "complete")
            or reset.get(_EFFECT_KEY) is True or _removed_names(reset)
            or reset.get(_CLEARING_KEY) is True or _CLEANUP_DONE_KEY in reset):
        return False
    try:
        current = panic_input(drive_root)
    except (OSError, UnicodeError):
        return False  # unknown: nothing is withdrawn on a guess
    if not current or current == reset.get(_PANIC_KEY):
        return False
    if reset.get(SET_ASIDE_KEY):
        name = _set_aside_name(reset)
        try:
            if not name or (_require_plain_directory(drive_root / "projects") and
                            _holds_entries(drive_root.joinpath(*name.split("/")))):
                return False
        except OSError:
            return False  # a planned folder that cannot be checked is not proof of no effect
    _write(drive_root, replaced)
    fsync_folder(drive_root / "state")
    log.warning("Panic withdrew the requested Reset before anything was moved or removed")
    return True


def _removed_names(record: dict) -> list[str]:
    names = record.get(_REMOVED_KEY)
    return [name for name in _REMOVED if isinstance(names, list) and name in names]


def _proven_absent_now(drive_root: Any, name: str) -> bool:
    """Positively absent at this moment, or not: an unreadable entry is never a removal."""
    try:
        confirm_absent(pathlib.Path(drive_root) / name)
        return True
    except OSError:
        return False


class ResetProtectionUnavailable(RuntimeError):
    def __init__(self):
        super().__init__("Reset did not start: this server could not verify that the restart would keep its "
                         "network access protected. No data was changed.")


def request_runtime_reset(drive_root: pathlib.Path, *, launch_protection: Callable[[], str]) -> str:
    """Record the owner's confirmed request before the restart is asked for. Serialize with state initialization.
    A repeated request of an interrupted Reset keeps its identity, its folder of set-aside shelves, what its
    earlier passes removed and the witness the first request replaced, so the receipt covers the whole Reset.
    Returns the identity this durable write recorded for that Reset: what a receipt and a ``reset_status`` read
    name, never one read back later from this module's state."""
    global _requested_reset
    from supervisor.state import acquire_file_lock, release_file_lock

    assert_test_data_path(witness_path(drive_root))
    _require_plain_directory(drive_root / "state")
    lock = drive_root / "locks/state.lock"
    fd = acquire_file_lock(lock)
    if fd is None:
        raise OSError("State initialization is busy")
    try:
        from ouroboros.launch_roots import RESET_PROTECTION_KEY, RESET_PROTECTION_MODES

        status, previous = read_witness(drive_root)
        mode = launch_protection()
        old_mode = previous.get(RESET_PROTECTION_KEY) if status == RESETTING else None
        if mode not in RESET_PROTECTION_MODES or old_mode not in (None, mode):
            raise ResetProtectionUnavailable()
        basis, _ = durable_basis(drive_root, evidence=supervisor_evidence)
        repeated = previous if status == RESETTING else {}
        identity = repeated["initialization_id"] if repeated else uuid.uuid4().hex
        replaced = repeated.get(_REPLACED_KEY) if repeated else previous if status == "ok" else None
        carried = {SET_ASIDE_KEY: _set_aside_name(repeated), _REMOVED_KEY: _removed_names(repeated),
                   _EFFECT_KEY: repeated.get(_EFFECT_KEY) is True, _CLEARING_KEY: repeated.get(_CLEARING_KEY) is True,
                   _PANIC_KEY: _panic_input_now(drive_root),
                   _REPLACED_KEY: replaced if isinstance(replaced, dict) else None}
        proof = _working_folder_proof(repeated)
        record = {"initialization_id": identity, "phase": RESETTING, "origin": "owner_reset",
                  "created_at": utc_now_iso(), BASIS_KEY: basis, RESET_PROTECTION_KEY: mode,
                  **{key: value for key, value in carried.items() if value}}
        if proof is not None:  # a possibly empty list survives a repeated request
            record[_PROOF_KEY] = proof
        if _CLEANUP_DONE_KEY in repeated:  # malformed evidence must not turn into permission to clear again
            record[_CLEANUP_DONE_KEY] = repeated[_CLEANUP_DONE_KEY]
        _write(drive_root, record)
        fsync_folder(drive_root / "state")
        _requested_reset = identity
        return identity
    finally:
        release_file_lock(lock, fd)


def hand_off_runtime_reset(drive_root: pathlib.Path, blockers: list[str]) -> str | None:
    """At the existing shutdown census, pass the owner's request once or leave it interrupted. ``blockers`` are
    this generation's own stop facts; with none, the custody owners' durable records must also show that nothing
    of the profile can still write (``_profile_work_blockers``). An interrupted request clears nothing."""
    global _requested_reset
    if not _requested_reset:
        return None
    status, reset = read_witness(drive_root)
    if status != RESETTING or reset["phase"] != RESETTING or reset.get("initialization_id") != _requested_reset:
        return None
    error = ", ".join(blockers)
    if not error:
        try:
            _require_no_new_panic(drive_root, reset)
            error = ", ".join(_profile_work_blockers(drive_root, reset))
            if not error:
                _require_no_new_panic(drive_root, reset)
                reset["reset_handed_off"] = True
        except OSError as exc:
            error = str(exc)
    if error:
        reset["reset_error"] = error
    _write(drive_root, reset)
    fsync_folder(drive_root / "state")
    _requested_reset = None
    return "reset_interrupted" if error else "reset_requested"


def _profile_work_blockers(drive_root: pathlib.Path, reset: dict) -> list[str]:
    """What may still write this profile once this generation's own stop is confirmed, read from the custody
    owners' durable records and never from this process's census: a generation that started none of that work
    has an empty one. Unknown blocks. A boot held for this Reset has retried the previous generation's stops
    (``server_boot.hold_interrupted_reset``), so a repeated request can pass."""
    return [*_delegated_run_blockers(drive_root, cleared=reset.get(_CLEARING_KEY) is True),
            *_process_blockers(drive_root)]


def _delegated_run_blockers(drive_root: pathlib.Path, *, cleared: bool) -> list[str]:
    """Every open delegated run gets the owner's Stop over an engine that is only attached to, never started
    (``stop_tree_runs``), and none may stay open. A pending start is held work until its own custody settles it:
    the custody owner's per-engine read binds what that start's own engine and store accepted for it
    (``reconcile_orphaned_runs`` with reads admitted: nothing is sent, replayed or replaced), which makes it an
    open run to stop. A start that stays pending blocks, whatever engine answers: a miss, another store's answer
    or a review panel's own start proves no absence of work (``delegate_accepted``).

    ``cleared``: an earlier pass of this same Reset began clearing ``state/``, so the store inside it is no
    authority at all; what still decides is the retained chain (``_retained_chain_blockers``), and its
    unreadable state blocks as unknown, never as empty. "Never imported" is the generation record's own
    positive answer, never the store's absence: stamped and gone is lost custody, a torn record unknown."""
    from ouroboros import delegate_custody_reconcile as runs
    from ouroboros.startup_migrations import watermarks
    from ouroboros.store import generation_stamped, store_path

    try:
        try:
            store_path(drive_root).lstat()
        except FileNotFoundError:
            confirm_absent(store_path(drive_root))
            # The store's absence alone settles nothing: a root that never imported still left the
            # retained event chain, and only its owner's fold says whether runs are open.
            legacy = _retained_chain_blockers(drive_root)
            if legacy:
                return legacy
            if cleared:  # this Reset's own pass removed the store with the ``state/`` folder around it
                return []
            # "Never imported" must be the generation record's positive answer, not an unreadable one:
            # ``generation_stamped`` reads a torn record as unstamped, and a torn record is unknown work.
            try:
                watermarks(drive_root)
            except ValueError:
                return ["delegated_generation_unreadable"]
            # Never imported: this root recorded no delegated work. Stamped and gone: its custody is lost.
            return ["delegated_custody_lost"] if generation_stamped(drive_root) else []
        starts, held = runs.pending_invocations(drive_root), runs.open_runs(drive_root)
    except Exception as exc:
        # An unreadable store never reads as empty. Once this Reset's own pass began clearing ``state/``
        # the store is no authority at all; the retained chain then decides, and its own unreadable
        # state blocks as unknown rather than empty.
        legacy = _retained_chain_blockers(drive_root)
        if legacy:
            return legacy
        if cleared:
            return []
        log.warning("Reset could not read the delegated-run custody", exc_info=True)
        return [f"delegated_runs_unknown:{type(exc).__name__}"]
    try:
        if starts:
            runs.reconcile_orphaned_runs(drive_root, set(), gateway_factory=runs.attach_only_gateway,
                                         observe_pending=lambda: True)
            # The one start class that sweep leaves to its panel (#1006): a review start the engine may
            # have accepted, whose START_REQUESTED row outlived its worker. An owner Reset stops this
            # generation first, so no recovering seat ever rejoins it (rejoin_recorded_review is a live
            # seat's first step); unread, it would hold this Reset as delegated_starts_unknown forever.
            # The custody owner reads that recorded start itself — the same read-only, per-engine bind
            # the sweep's watched lane uses, never a send — so a positively bound run joins ``held`` for
            # the normal Stop below. A miss, an unreachable engine or a foreign store leaves the start
            # pending and this Reset interrupted: no answer about a review start reads as absence of work.
            runs.observe_review_pending_starts(drive_root)
            held = runs.open_runs(drive_root)
        for tree in sorted({runs.run_tree_id(run) for run in held}):
            runs.stop_tree_runs(drive_root, tree, gateway_factory=runs.attach_only_gateway)
        left, unsettled = runs.open_runs(drive_root), runs.pending_invocations(drive_root) if starts else []
    except Exception as exc:  # work a readable store records stays unknown, also on a repeat
        log.warning("Reset could not confirm that delegated runs stopped", exc_info=True)
        return [f"delegated_runs_unknown:{type(exc).__name__}"]
    return [*([f"delegated_runs_unconfirmed:{len(left)}"] if left else []),
            *([f"delegated_starts_unknown:{len(unsettled)}"] if unsettled else [])]


def _retained_chain_blockers(drive_root: pathlib.Path) -> list[str]:
    """Legacy delegated custody established through the retained event chain's own owner
    (``delegate_custody``: ``event_log_path`` / ``custody_chain_unreadable`` / ``rows_after``, folded by
    ``delegate_custody_current.fold``). A proven-absent chain is empty; an unreadable one blocks as
    unknown, never as empty; a readable one with an open run or a pending invocation blocks the Reset."""
    from ouroboros import delegate_custody as custody
    from ouroboros.delegate_custody_current import fold

    try:
        if custody.custody_chain_unreadable(drive_root):
            return ["delegated_chain_unreadable"]
        rows, _boundary = custody.rows_after(custody.event_log_path(drive_root), 0)
        folded = fold(rows)
    except Exception as exc:
        log.warning("Reset could not read the retained delegated-run chain", exc_info=True)
        return [f"delegated_chain_unknown:{type(exc).__name__}"]
    open_runs = [rid for rid, (facts, _hold) in folded["runs"].items() if not facts.get("settled")]
    pending = list(folded["open"])
    return [*([f"delegated_runs_unconfirmed:{len(open_runs)}"] if open_runs else []),
            *([f"delegated_starts_unknown:{len(pending)}"] if pending else [])]


def _process_blockers(drive_root: pathlib.Path) -> list[str]:
    """No task, session or companion process the installation's ledger and ownership set still name may be alive
    (``launch_custody.unsettled_selection``: a read, never a signal). Only an installation-retained daemon — a
    ``daemon`` row or a legacy row under the daemon's own retained purpose — that is uniquely verified, live and
    fingerprint-matching (``process_custody.spared_daemon_roots``), is not this profile's writer; every other
    row stays, companion rows included, so actual profile or unknown custody is held and never spared by its
    scope or purpose alone. No engine is provisioned to ask."""
    from ouroboros import launch_custody, process_custody as pc
    from ouroboros.claudexor_daemon import CUSTODY_PURPOSE

    try:
        custody = launch_custody.read_custody(drive_root, every_drive=True)
        if not custody.readable:
            return ["process_custody_unreadable"]
        spared = pc.spared_daemon_roots(drive_root)  # None: the daemon roots are unknown, nothing is spared
        # A row leaves the selection only through that verified identity — a ``daemon`` row or a legacy
        # row under the same retained purpose; a companion row stays held either way, and only a live
        # fingerprint-matching pid that ``spared_daemon_roots`` named spares, never the scope or purpose alone.
        work = [row for row in custody.rows
                if str(row.get("purpose") or "").startswith("companion:")
                or not (row.get("scope") == "daemon" or row.get("purpose") == CUSTODY_PURPOSE)
                or spared is None or int(row.get("pid") or 0) not in spared]
        if alive := launch_custody.unsettled_selection(drive_root, work, custody):
            return [f"processes_unconfirmed:{len(alive)}"]
        return ["process_custody_unknown"] if custody.blind else []
    except Exception as exc:
        log.warning("Reset could not confirm that the profile's processes stopped", exc_info=True)
        return [f"process_custody_unknown:{type(exc).__name__}"]


def _set_aside_name(record: dict) -> str | None:
    name = record.get(SET_ASIDE_KEY)
    return name if isinstance(name, str) and _SET_ASIDE.fullmatch(name) else None


def _working_folder_proof(record: dict) -> list[str] | None:
    """The saved working-folder proof (a possibly empty list) or ``None`` when none was saved."""
    proof = record.get(_PROOF_KEY)
    return [str(f) for f in proof if f] if isinstance(proof, list) else None


def _record_working_folder_proof(drive_root: Any, reset: dict, folders: list[str],
                                 configured: list[str]) -> None:
    """Save the working-folder proof in the witness before any effect: a continuation of the
    same Reset reads it instead of a registry or settings the first pass may remove."""
    if _PROOF_KEY in reset:
        return
    reset[_PROOF_KEY] = sorted(set(str(f) for f in (*folders, *configured) if f))
    _write(drive_root, reset)
    fsync_folder(drive_root / "state")


def _set_aside_facts(drive_root: Any, record: dict) -> dict:
    """The recorded folder, and where it is under this data root: a dot folder is found by its path."""
    name = _set_aside_name(record)
    return {SET_ASIDE_KEY: name,
            f"{SET_ASIDE_KEY}_path": str(pathlib.Path(drive_root).joinpath(*name.split("/"))) if name else None}


def _holds_entries(folder: pathlib.Path) -> bool:
    """Is this an ordinary directory with at least one entry? Anything else there, or one that cannot be read, raises."""
    return _require_plain_directory(folder) and any(folder.iterdir())


def _set_aside_holds_notes(drive_root: Any, record: dict) -> bool:
    """Positive evidence, now, that the recorded folder holds set-aside notes: its name is written before the first
    move and a move that fails is undone, so the name alone proves nothing."""
    name = _set_aside_name(record)
    try:
        return bool(name) and _holds_entries(pathlib.Path(drive_root).joinpath(*name.split("/")))
    except OSError:
        return False


def reset_status(drive_root: pathlib.Path) -> dict:
    status, record = read_witness(drive_root)
    if status in {"invalid", "unreadable"}:
        return {"status": "unknown"}
    if record.get("origin") != "owner_reset":
        return {"status": "none"}
    # The identity THIS Reset's durable write recorded, projected beside its status: a client that asked for that
    # Reset matches a later answer by it. Never minted here, and a profile with no such Reset names none above.
    reset_id = str(record["initialization_id"])
    if status == RESETTING:
        if (record["phase"] == RESETTING and record.get("reset_handed_off") is True
                or record["initialization_id"] == _requested_reset):
            return {"status": "requested", "reset_id": reset_id}
        # ``deleted``: the removals this Reset's passes recorded, proven absent now — never one that survived.
        # The folder: only while it holds notes.
        removed = [name for name in _removed_names(record) if _proven_absent_now(drive_root, name)]
        return {"status": "interrupted", "error": str(record.get("reset_error") or _NO_REASON),
                **({"deleted": removed} if removed else {}),
                **(_set_aside_facts(drive_root, record) if _set_aside_holds_notes(drive_root, record) else {}),
                "reset_id": reset_id}
    receipt = reset_receipt(record)
    if receipt:
        return {"status": "complete", "deleted": receipt["deleted"], "retained_logs": receipt["retained_logs"],
                **_set_aside_facts(drive_root, receipt), "reset_id": reset_id}
    # A receipt that IS recorded but does not validate — a wrong shape, or not this Reset's identity — is
    # never silently certified as complete without details: that witness is as unknown as an unreadable one.
    # It names no identity either: the one identity here is the one that broken receipt failed to match.
    if "reset_receipt" in record:
        return {"status": "unknown"}
    # An older Reset completed without the detailed receipt: it is done, not a new failure; its
    # details are unknown and stay out of the answer rather than being invented.
    return ({"status": "complete", "reset_id": reset_id} if record.get("phase") == "complete"
            else {"status": "unknown"})


def accept_runtime_reset(drive_root: pathlib.Path, *, boot_protection: Callable[[], str] | None = None) -> None:
    """Called only on first exclusive admission. Consume the hand-off before touching any profile data: without
    one, a boot clears nothing. An interruption keeps what this pass did in the witness for the explicit repeat."""
    status, reset = read_witness(drive_root)
    if status != RESETTING or _withdrawn_by_panic(drive_root, reset):
        return
    from ouroboros.launch_roots import RESET_PROTECTION_KEY, RESET_PROTECTION_MODES

    if reset["phase"] != RESETTING:
        return
    handoff = reset.get("reset_handed_off") is True
    expected = reset.get(RESET_PROTECTION_KEY)
    if handoff or expected is not None:
        if expected not in RESET_PROTECTION_MODES or boot_protection is None or boot_protection() != expected:
            raise OSError("The Reset was left intact: this start cannot verify its expected network protection.")
    if not handoff:
        return  # a protected read-hold is not another authorization to clean up
    reset.pop("reset_handed_off")
    reset.pop("reset_error", None)
    _write(drive_root, reset)
    fsync_folder(drive_root / "state")
    try:
        _require_no_new_panic(drive_root, reset)
        if _CLEANUP_DONE_KEY not in reset:
            plan = _reset_footprint(drive_root, reset)
            _require_working_folders_kept(drive_root, reset, plan)
            receipt = _clear_runtime_data(drive_root, reset, plan)
            # Commit the proved cleanup before publishing a successor. Re-entry may publish again, never clear again.
            reset[_CLEANUP_DONE_KEY] = {"initialization_id": reset["initialization_id"], **receipt}
            _write(drive_root, reset)
            fsync_folder(drive_root / "state")
        receipt = reset_receipt({**reset, "reset_receipt": reset[_CLEANUP_DONE_KEY]})
        if not receipt or reset.get(_CLEARING_KEY) is not True:
            raise OSError("The recorded cleanup could not be confirmed for this Reset")
        from ouroboros.project_registry_recovery import initialize_new_root

        _require_no_new_panic(drive_root, reset)
        initialize_new_root(drive_root, after_reset=True)
        _require_no_new_panic(drive_root, reset)
        excluded = (SET_ASIDE_KEY, _REMOVED_KEY, _EFFECT_KEY, _CLEARING_KEY, _PANIC_KEY,
                    _REPLACED_KEY, _PROOF_KEY, _CLEANUP_DONE_KEY, RESET_PROTECTION_KEY)
        done = {key: value for key, value in reset.items() if key not in excluded}
        _write(drive_root, {**done, "phase": "pending", "reset_receipt": receipt})
        fsync_folder(drive_root / "state")
    except Exception as exc:
        _write(drive_root, {**reset, "reset_error": str(exc) or type(exc).__name__})
        log.error("Runtime Reset interrupted: %s", exc)


def _clearable_entries(area: pathlib.Path, keep: tuple[str, ...], *, open_files: bool) -> list[pathlib.Path]:
    """The entries one ``_clear_reset_directory`` pass would remove now: every entry the clear's
    own rule does not keep. ``open_files`` is the logs' rule: a kept name still goes unless it is
    the regular open journal the clear retains; without it every kept name stays whatever it is.
    An area that is not a plain directory raises; a proven-absent one clears nothing."""
    entries: list[pathlib.Path] = []
    if not _require_plain_directory(area):
        return entries
    for entry in sorted(area.iterdir()):
        try:
            mode = entry.lstat().st_mode
        except FileNotFoundError:
            confirm_absent(entry)
            continue
        if entry.name in keep and (not open_files or stat.S_ISREG(mode)):
            continue
        entries.append(entry)
    return entries


def _reset_footprint(drive_root: pathlib.Path, reset: dict) -> dict:
    """What this Reset would actually do to the old profile, computed once, read-only, before any
    effect: the shelves it moves (source and destination), the folders and the file it removes
    whole, and the two directory clears — ``state`` keeping its witness, lock and the
    installation's own records, ``logs`` keeping the open journals — with the very keep sets the
    cleanup acts on. The cleanup below consumes this same plan, so what the guard checks is what
    happens: no second copy of any list decides."""
    from ouroboros.server_entrypoint import SERVER_LOCK_NAME

    root = pathlib.Path(drive_root)
    projects = root / "projects"
    shelves: list[pathlib.Path] = []
    try:
        # A plain, confined directory is validated before any traversal: a link or a file where
        # ``projects/`` belongs is an interruption, never walked.
        if _require_plain_directory(projects):
            shelves = sorted(entry for entry in projects.iterdir() if not entry.name.startswith("."))
    except FileNotFoundError:
        confirm_absent(projects)
    name = _set_aside_name(reset)
    if not name and shelves:
        day, identity = utc_now_iso()[:10].replace("-", ""), str(reset["initialization_id"])
        short = (identity[:8] if re.fullmatch(r"[0-9a-f]{8}", identity[:8])
                 else hashlib.sha256(identity.encode()).hexdigest()[:8])
        name = f"projects/.before-reset-{day}-{short}"
    container = root.joinpath(*name.split("/")) if name else None
    moves = [(entry, container / entry.name) for entry in shelves] if container else []
    logs_clear = {"path": root / "logs", "keep": _ACTIVE_LOGS}
    state_clear = {"path": root / "state",
                   "keep": (witness_path(drive_root).name, SERVER_LOCK_NAME, *_STATE_KEPT)}
    logs_clear["entries"] = _clearable_entries(logs_clear["path"], logs_clear["keep"], open_files=True)
    state_clear["entries"] = _clearable_entries(state_clear["path"], state_clear["keep"], open_files=False)
    targets = [("move", source) for source, _ in moves] + [("move", destination) for _, destination in moves]
    whole_targets = []
    for item in (*_REMOVED_FOLDERS, _REMOVED_FILE):
        path = root / item
        try:
            confirm_absent(path)  # proven absent: an earlier pass's removal is no target
        except OSError:
            targets.append(("remove", path))  # present or unprovable: a target for the guard
            whole_targets.append(item)
    targets.extend(("remove", entry) for entry in logs_clear["entries"])
    logs_clear["remove_if_empty"] = False
    try:
        confirm_absent(logs_clear["path"])
    except OSError:
        # An empty clear removes the logs folder itself; ``state`` never goes: its witness and
        # admission lock survive the clear, so the folder itself is not one of its targets.
        targets.append(("remove", logs_clear["path"]))
        logs_clear["remove_if_empty"] = True
    targets.extend(("remove", entry) for entry in state_clear["entries"])
    return {"container": name, "shelves": shelves, "moves": moves, "targets": targets,
            "removed_folders": [name for name in whole_targets if name != _REMOVED_FILE],
            "removed_file": _REMOVED_FILE if _REMOVED_FILE in whole_targets else None,
            "clear_logs": logs_clear, "clear_state": state_clear}


def _require_working_folders_kept(drive_root: pathlib.Path, reset: dict, plan: dict) -> None:
    """The pre-effect guard: every working folder this profile owns — the prior registry's and the
    configured work roots' — is still where it is when this Reset is done. A folder the plan would
    move or remove stops the Reset before any effect, and a prior registry or a settings document
    that cannot be proven stops it too: unknown never reads as empty. The comparison is the
    admission geometry (``workspace_admission.folders_overlap``): two paths hold one another in
    either direction, however the volume spells them. The proof is saved in the witness before
    any effect: a continuation of the same Reset reads it instead of a source the first pass
    may remove."""
    from ouroboros.project_registry_recovery import prior_working_folders
    from ouroboros.workspace_admission import folders_overlap

    if not plan["targets"]:
        return  # this Reset's remaining footprint is empty: no folder can be lost in it
    folders, unknown = prior_working_folders(drive_root)
    if unknown:
        raise OSError(f"Reset did not start: {unknown}")
    configured = _configured_work_roots(drive_root)
    _record_working_folder_proof(drive_root, reset, folders, configured)
    for folder in [*folders, *configured]:
        path = pathlib.Path(folder)
        for verb, target in plan["targets"]:
            if folders_overlap(path, target):
                raise OSError(f"Reset did not start: it would {verb} the working folder {folder} at {target}")


def _configured_work_roots(drive_root: pathlib.Path) -> list[str]:
    """The work roots the profile being reset was configured with, from the settings document that
    profile read — the module's own configured-root rule (``setup_left_consciousness_on``): the
    configured document when this root is the configured data root, where an explicit
    ``SETTINGS_PATH`` sits separately from the file Reset removes, else this root's own document.
    A document that cannot be read stops the Reset: unknown never reads as defaults. A key the
    document leaves unset falls to config's own rule only where that rule resolves this root — a
    root this process never configured keeps only what its own document names, and its defaults
    are not this process's to guess."""
    from ouroboros import config
    from ouroboros.launch_roots import WorkRootUnavailable

    document, configured = _profile_settings_document(drive_root)
    roots = []
    for key, getter in (("OUROBOROS_SUBAGENT_WORKTREE_ROOT", config.get_subagent_worktree_root),
                        ("OUROBOROS_SUBAGENT_PROJECTS_ROOT", config.get_subagent_projects_root),
                        ("OUROBOROS_DELIVERABLES_ROOT", config.get_deliverables_root)):
        value = str(document.get(key) or "").strip()
        if not value and configured:
            try:
                # config's rule for THIS root: the same document, the environment, the work-root default.
                value = str(getter() or "").strip()
            except WorkRootUnavailable as exc:
                # The default is undecided (basis ambiguous, two candidates differ): unknown,
                # not a positive absence.  The guard refuses rather than silently unprotected.
                raise OSError(f"Reset did not start: the work root {key} is undecided") from exc
        if value:
            roots.append(value)
    return roots


def _profile_settings_document(drive_root: pathlib.Path) -> tuple[dict, bool]:
    """The profile's settings document, and whether this root is the configured data root. An
    absent document is the positive empty one; one that exists but cannot be read is unknown,
    never a defaults guess (``read_settings_json_strict``). Split settings: when the configured
    document lives outside this data root, the root's own ``settings.json`` is a separate
    document Reset removes; its keys fill gaps the configured one leaves."""
    from ouroboros import config

    configured = drive_root.resolve() == pathlib.Path(config.DATA_DIR).resolve()
    document_path = pathlib.Path(config.SETTINGS_PATH if configured
                                 else pathlib.Path(drive_root) / "settings.json")
    document = _read_settings_or_empty(document_path)
    own_path = pathlib.Path(drive_root) / "settings.json"
    if configured and own_path != document_path:
        for key, value in _read_settings_or_empty(own_path).items():
            document.setdefault(key, value)  # the configured document wins on conflicts
    return document, configured


def _read_settings_or_empty(path: pathlib.Path) -> dict:
    """One settings document through the verified reader: absent is the positive empty one;
    an unreadable or undecodable document is unknown, never a defaults guess."""
    from ouroboros.settings_integrity import read_settings_json_strict

    try:
        document = read_settings_json_strict(path)
    except FileNotFoundError:
        confirm_absent(path)
        return {}
    except Exception as exc:
        raise OSError(f"the settings document {path} could not be read") from exc
    if document is None:
        return {}
    if not isinstance(document, dict):
        raise OSError(f"the settings document {path} is not a settings object")
    return document


def _rename_without_replacement(source: pathlib.Path, target: pathlib.Path) -> None:
    """One atomic rename to a name that must not exist. Where the volume lacks that primitive (NFS,
    FUSE, some mounts), a plain rename follows only a proof that the target is absent: this first
    exclusive admission is the root's one writer, and a plain rename alone would replace an empty directory."""
    from ouroboros.platform_file_rename import rename_exclusive

    try:
        rename_exclusive(source, target)
    except OSError as exc:
        if exc.errno not in _NO_EXCLUSIVE_RENAME:
            raise
        confirm_absent(target)
        os.rename(source, target)


def _set_aside_project_shelves(drive_root: pathlib.Path, reset: dict, plan: dict) -> str | None:
    """Move the previous profile's Project shelves into one dot folder of ``projects/`` — the shelves
    this Reset's own plan named, so what the guard checked is what moves.

    Project ids repeat for the same name or working folder, so a shelf left in place would hand
    its notes to the next profile's Project. Each entry moves by one rename that never replaces,
    and no new move starts once Panic is observed; nothing is copied, merged or deleted. A failure
    here puts back every shelf this pass moved, so an interrupted Reset leaves the still-running
    profile all its notes and names no folder. Only a pass cut between moves, or a move back that
    fails, leaves shelves in the folder: the witness keeps naming it and an explicit repeat fills it.
    Readers of ``projects/`` skip dot names, so the folder is no room; ``read_file``, ``list_files``
    and search refuse the whole subtree."""
    projects = drive_root / "projects"
    assert_test_data_path(projects)
    entries = list(plan["shelves"])
    name = plan["container"]
    if not entries:  # nothing to move: a repeat names the folder only while its earlier moves are there
        return name if name and _holds_entries(drive_root.joinpath(*name.split("/"))) else None
    if not name:  # the plan named no folder: nothing moves
        raise OSError("Reset's plan named no set-aside folder for the shelves it would move")
    if _set_aside_name(reset) != name:  # the name is written before the first move
        reset[SET_ASIDE_KEY] = name
        _write(drive_root, reset)
        fsync_folder(drive_root / "state")
    container = drive_root.joinpath(*name.split("/"))
    for entry in entries:  # an earlier cut pass may hold a shelf the old profile has written again since
        try:
            (container / entry.name).lstat()
        except FileNotFoundError:
            confirm_absent(container / entry.name)
        else:
            raise FileExistsError(f"Reset found both {entry} and {container / entry.name} and merged nothing: "
                                  "move or remove one of them, then confirm Reset again")
    moved: list[pathlib.Path] = []
    step = f"could not prepare {container}"
    try:
        _require_no_new_panic(drive_root, reset)  # read-only: a Panic that arrived since starts no new move
        container.mkdir(exist_ok=True)
        _require_plain_directory(container)
        for entry in entries:
            step = f"could not move {entry} to {container / entry.name}"
            _require_no_new_panic(drive_root, reset)  # read-only: no new move starts once Panic is observed
            _rename_without_replacement(entry, container / entry.name)
            moved.append(entry)
        step = f"could not save the moves into {container}"
        fsync_folder(container)
        fsync_folder(projects)
    except (OSError, RuntimeError, ValueError) as exc:
        raise OSError(_undo_set_aside(container, moved, reset, f"{step}: {exc}")) from exc
    return name


def _undo_set_aside(container: pathlib.Path, moved: list, reset: dict, cause: str) -> str:
    """Put this pass's shelves back, newest first, and say what the interrupted Reset left."""
    for entry in reversed(moved):
        try:
            _rename_without_replacement(container / entry.name, entry)
        except OSError as exc:
            return (f"Reset {cause}; then could not move {container / entry.name} back to {entry}: {exc}. "
                    f"Project notes moved so far stay in {container}")
    try:
        if os.path.lexists(container):
            container.rmdir()  # only an empty folder goes: an earlier pass's shelves keep it, and its name
        fsync_folder(container.parent)  # the shelves are back for good before the witness forgets the name
    except OSError:
        return f"Reset moved no more Project notes into {container}: {cause}"
    reset.pop(SET_ASIDE_KEY, None)  # the caller writes this record with the error
    return f"Reset moved no Project notes: {cause}"


def _clear_runtime_data(drive_root: pathlib.Path, reset: dict, plan: dict) -> dict[str, Any]:
    from ouroboros.store import close_connections_under

    state_dir = drive_root / "state"
    marker = witness_path(drive_root)
    assert_test_data_path(marker)
    _require_plain_directory(state_dir)
    _require_no_new_panic(drive_root, reset)  # before the first new mutation of this admission
    set_aside = _set_aside_project_shelves(drive_root, reset, plan)  # before the first deletion
    close_connections_under(state_dir)
    removed = _removed_names(reset)  # this Reset's earlier passes recorded these removals as pending

    def pending(name: str) -> None:
        """Record one declared top-level removal as pending, BEFORE the effect, while its presence is
        positively observed: a pass cut between the removal and any later receipt leaves the pending
        record, and the receipt's proven absence reconciles it. The witness is the only record."""
        if name in removed:
            return
        path = drive_root / name
        try:
            path.lstat()
        except FileNotFoundError:
            confirm_absent(path)  # no presence observed: this pass records no pending removal
            return
        removed.append(name)
        reset[_REMOVED_KEY] = removed
        _write(drive_root, reset)

    def step() -> None:
        """Before each removal: a Panic that arrived since starts no further one (Reset only reads Panic's
        records here), and the first removal's boundary is saved before it begins."""
        _require_no_new_panic(drive_root, reset)
        if reset.get(_EFFECT_KEY) is not True:
            reset[_EFFECT_KEY] = True
            _write(drive_root, reset)
            fsync_folder(state_dir)

    step()
    logs_clear = plan["clear_logs"]  # the open journals the plan keeps are the ones this clear retains
    if logs_clear["remove_if_empty"]:
        pending("logs")
    _require_no_new_panic(drive_root, reset)  # read-only: no new clear starts once Panic is observed
    _, retained_logs = _clear_reset_directory(  # gone now, or retained open logs: the receipt says which
        logs_clear["path"], entries=logs_clear["entries"], active_logs=logs_clear["keep"],
        remove_if_empty=logs_clear["remove_if_empty"])
    for name in plan["removed_folders"]:
        step()
        pending(name)
        _require_no_new_panic(drive_root, reset)  # read-only: no new removal starts once Panic is observed
        _remove_for_reset(drive_root / name, _remove_tree_for_reset)
    if plan["removed_file"] is not None:
        step()
        pending(plan["removed_file"])
        _require_no_new_panic(drive_root, reset)  # read-only: no new removal starts once Panic is observed
        _remove_for_reset(drive_root / plan["removed_file"], _unlink_for_reset)
    _forget_settings_projection(drive_root)
    step()
    if reset.get(_CLEARING_KEY) is not True:  # from here the stores inside ``state/`` are no proof of anything
        reset[_CLEARING_KEY] = True
        _write(drive_root, reset)
        _require_no_new_panic(drive_root, reset)  # read-only: no new clear starts once Panic is observed
    # No second admission inode, no erasure of an emergency stop record, no interval where the only durable
    # initialization identity disappears, and nothing of the installation or of a process that outlives the profile.
    state_clear = plan["clear_state"]  # the plan's keep set is the one this clear acts on
    _clear_reset_directory(state_clear["path"], entries=state_clear["entries"], keep=state_clear["keep"])
    for name in (*_REMOVED_FOLDERS, _REMOVED_FILE):
        confirm_absent(drive_root / name)
    status, saved = read_witness(drive_root)
    if status != RESETTING or saved["initialization_id"] != reset["initialization_id"]:
        raise OSError("Reset cleanup has no matching initialization witness")
    fsync_folder(drive_root)
    fsync_folder(state_dir)
    return {"deleted": [name for name in _REMOVED if name in removed and not (name == "logs" and retained_logs)],
            "retained_logs": retained_logs, SET_ASIDE_KEY: set_aside}


def _forget_settings_projection(drive_root: pathlib.Path) -> None:
    """The cleared settings document's mirror leaves this process's environment with it, as its launcher, exec or
    spawn handed it over with the projection's own record (``settings_integrity.drop_projection``: an operator's
    variables stay; a mirrored network password is not an independent launch input).
    Only when this process's configured document is the file Reset just removed from its own data root: another
    root's Reset, or a document kept elsewhere (``OUROBOROS_SETTINGS_PATH``), leaves this environment alone."""
    from ouroboros import config
    from ouroboros.launch_roots import same_location
    from ouroboros.settings_integrity import drop_projection

    document = drive_root / "settings.json"
    if (drive_root.resolve() == pathlib.Path(config.DATA_DIR).resolve()
            and same_location(config.SETTINGS_PATH, document) and not os.path.lexists(document)):
        drop_projection(os.environ)


def _decide(drive_root: Any) -> Dict[str, Any]:
    """``initialization_decision`` without its writes: a create on a root with no witness has no id yet."""
    status, witness = read_witness(drive_root)
    if status == RESETTING:
        return {"create": False, "reason": "reset_cleanup_unconfirmed",
                "detail": "Only an explicit Reset can finish interrupted cleanup"}
    if status in {"invalid", "unreadable"}:
        return {"create": False, "reason": "initialization_witness_unreadable", "detail": status}
    if status == "ok" and witness.get("phase") == "complete":
        return {"create": False, "reason": "initialized_state_lost",
                "detail": f"initialization {witness.get('initialization_id')} completed at "
                          f"{witness.get('completed_at') or '?'}"}
    explicit_reset = status == "ok" and witness.get("origin") == "owner_reset"
    if not explicit_reset:
        evidence, where = supervisor_evidence(drive_root)
        if evidence == "unknown":
            return {"create": False, "reason": "initialization_evidence_unknown", "detail": where}
        if evidence == "present":
            return {"create": False, "reason": "prior_history_without_state", "detail": where}
    if explicit_reset and not _readable_reset_registry(drive_root):
        return {"create": False, "reason": "reset_registry_unavailable",
                "detail": "Project initialization after Reset is still pending"}
    return {"create": True, **({"initialization_id": str(witness["initialization_id"])} if status == "ok" else {})}


def initialization_decision(drive_root: Any, *, origin: str = "first_boot") -> Dict[str, Any]:
    """Both state copies are positively absent: may a FIRST state be created?

    On yes, the witness is ``pending`` with the returned id before the caller
    writes the state; the caller completes it (``complete``) after."""
    decision = _decide(drive_root)
    if not decision["create"]:
        return decision
    status, witness = read_witness(drive_root)
    explicit_reset = status == "ok" and witness.get("origin") == "owner_reset"
    initialization_id = decision.get("initialization_id") or mark_pending(
        drive_root, origin=origin or "first_boot", work_root_basis=APP_ROOT_BASIS)  # a new root: its work roots follow APP_ROOT
    # A cut after recording the intent retries registry initialization for the same
    # identity. Reset publishes its own registry after setting the old shelves aside.
    if not explicit_reset and witness.get("origin") != "legacy_adopted":
        from ouroboros.project_registry_recovery import initialize_new_root
        initialize_new_root(drive_root)
    return {"create": True, "initialization_id": initialization_id}


def _readable_reset_registry(drive_root: Any) -> bool:
    from ouroboros.project_registry_recovery import read_bytes, read_primary, registry_path, validated_registry
    try:
        validated_registry(read_bytes(read_primary(registry_path(drive_root))[1]))
        return True
    except (OSError, ValueError):
        return False


def pending_reset_identity(drive_root: Any) -> "str | None":
    """The identity an owner Reset's ``pending`` witness names, else ``None``. Under it boot sets aside every state copy
    ``reset_sets_aside`` names (``state._set_aside_copies_older_than_a_reset``) and mints that identity."""
    status, witness = read_witness(drive_root)
    if status == "ok" and witness.get("phase") == "pending" and witness.get("origin") == "owner_reset":
        return str(witness.get("initialization_id") or "")
    return None


def reset_sets_aside(status: str, copy: Any, identity: str) -> bool:
    """Is a state copy read as ``(status, copy)`` older than the pending Reset of ``identity``? Every present copy but one
    of that identity; boot refuses an ``unreadable`` one instead of moving it."""
    return status != "missing" and not (status == "ok" and copy.get("initialization_id") == identity)


def first_state_pending(drive_root: Any) -> bool:
    """Would the next ``init_state`` CREATE a first state here? The same decision boot makes, read without writing
    anything (the setup bootstrap and completion ask it): both copies absent once a pending owner Reset has set aside
    the older ones, and the decision above. A completed witness whose copies are lost, a copy boot would adopt and a
    copy that cannot be read are not a first state: no."""
    from supervisor.state import read_state_copy

    root = pathlib.Path(drive_root)
    identity = pending_reset_identity(root)
    for name in ("state.json", "state.last_good.json"):
        status, copy, _detail = read_state_copy(root / "state" / name)
        if status != "missing" and (identity is None or status == "unreadable"
                                    or not reset_sets_aside(status, copy, identity)):
            return False
    return bool(_decide(root)["create"])


def first_state(drive_root: Any, initialization_id: str) -> Dict[str, Any]:
    """What a created first state of ``drive_root`` is born with, before ``init_state`` writes it.

    Runs under STATE_LOCK: nothing here calls ``update_state`` (the lock is not re-entrant) or
    takes a settings lock. A fresh install never ran an earlier release, so the notices
    ``upgrade_notices.BORN_SETTLED_NOTICE_KEYS`` names are born marked; that stamp is not a
    delivery (no chat receipt exists for it), and the notices judged by facts stay owed.
    Consciousness is born on when this root's setup left its switch on (``setup_left_consciousness_on``),
    and a state minted after a recorded setup owes the first meeting its Start opens (``supervisor.first_meeting``)."""
    from ouroboros.upgrade_notices import BORN_SETTLED_NOTICE_KEYS  # D08 -> D11 is allowed only lazily
    from supervisor.first_meeting import FIRST_MEETING_KEY, owed

    pressed_at = setup_pressed_start_at(drive_root)
    return {"initialization_id": initialization_id, **dict.fromkeys(BORN_SETTLED_NOTICE_KEYS, utc_now_iso()),
            **({"bg_consciousness_enabled": True} if setup_left_consciousness_on(drive_root) else {}),
            **({FIRST_MEETING_KEY: owed(pressed_at)} if pressed_at else {})}


def _setup_document(drive_root: Any) -> Dict[str, Any]:
    """The settings document of ``drive_root`` as setup left it: the configured document when that root is the
    configured data root, else ``<root>/settings.json``. Read lock-free and verified (the caller may hold STATE_LOCK and
    takes no settings lock); ``{}`` for none, and a document that cannot be read raises."""
    from ouroboros import config
    from ouroboros.settings_integrity import read_settings_json_verified

    root = pathlib.Path(drive_root)
    configured = root.resolve() == pathlib.Path(config.DATA_DIR).resolve()
    document = read_settings_json_verified(config.SETTINGS_PATH if configured else root / "settings.json")
    return document if isinstance(document, dict) else {}


def setup_left_consciousness_on(drive_root: Any) -> bool:
    """The setup choice the settings document of ``drive_root`` records (owner decision 2026-11-20); an unreadable
    document is no choice, and a pending installation Panic withholds it (Panic's own controls write keeps it off)."""
    from ouroboros.installation_stop import installation_panic
    from ouroboros.settings_setup_contract import ONBOARDING_CONSCIOUSNESS_KEY

    try:
        chosen = _setup_document(drive_root).get(ONBOARDING_CONSCIOUSNESS_KEY) == "on"
        return chosen and not installation_panic(pathlib.Path(drive_root))
    except Exception:
        log.warning("the setup's consciousness choice could not be read; the first state is born off", exc_info=True)
        return False


def setup_pressed_start_at(drive_root: Any) -> str:
    """When my human pressed Start at the end of setup, as the settings document of ``drive_root`` records it (the
    completion fact ``OUROBOROS_ONBOARDING_COMPLETED_AT``); "" for no recorded setup (Colab, headless, env-only) or a
    document that cannot be read."""
    from ouroboros.settings_setup_contract import ONBOARDING_COMPLETED_KEY

    try:
        return str(_setup_document(drive_root).get(ONBOARDING_COMPLETED_KEY) or "").strip()
    except Exception:
        log.warning("the setup's completion could not be read; the first state owes no first meeting", exc_info=True)
        return ""


def authority_reason(drive_root: Any, initialization_id: str) -> str:
    """Read-only admission proof: a durable completion of this exact identity."""
    status, witness = read_witness(drive_root)
    if status != "ok":
        return f"initialization_witness_{status}"
    if not initialization_id or witness["initialization_id"] != initialization_id:
        return "initialization_identity_mismatch"
    return "" if witness["phase"] == "complete" else "initialization_incomplete"


def prepare_adoption(drive_root: Any, state: Dict[str, Any]) -> str:
    """Bind readable legacy data or resume a pending identity before writing it.

    Caller holds STATE_LOCK. Readable legacy session/creation facts, rather than
    absence of files, are the adoption evidence. An existing witness never changes
    identity here; only the explicit Reset owner may replace it.
    """
    status, witness = read_witness(drive_root)
    identity = str(state.get("initialization_id") or "")
    source = pathlib.Path(drive_root) / "state" / "state.json"
    source_digest = hashlib.sha256(source.read_bytes()).hexdigest()
    if status == "ok":
        if (not identity and witness.get("phase") == "pending"
                and witness.get("origin") == "legacy_adopted"
                and witness.get("legacy_source_sha256") == source_digest):
            return str(witness["initialization_id"])
        if identity != witness["initialization_id"]:
            raise ValueError("initialization_identity_mismatch")
        return identity
    if status != "missing":
        raise ValueError(f"initialization_witness_{status}")
    if not identity and not (state.get("session_id") and state.get("created_at")):
        raise ValueError("legacy_initialization_evidence_missing")
    basis = adopted_basis(identity)  # predates the witness: home; lost its witness: unknown
    identity = identity or uuid.uuid4().hex
    _write(drive_root, {"initialization_id": identity, "phase": "pending",
                        "origin": "legacy_adopted", "created_at": utc_now_iso(),
                        "legacy_source_sha256": source_digest, BASIS_KEY: basis})
    return identity


def complete(drive_root: Any, initialization_id: str, *, adopted: bool) -> bool:
    """Complete only the exact pending identity after both state writes succeeded."""
    try:
        status, witness = read_witness(drive_root)
        if (status != "ok" or not initialization_id
                or witness.get("initialization_id") != initialization_id):
            return False
        if witness["phase"] != "complete":
            explicit_reset = witness.get("origin") == "owner_reset"
            if explicit_reset and not _readable_reset_registry(drive_root):
                return False
            if adopted and witness.get("origin") in {"first_boot", "isolated_benchmark"}:
                # Resume a pending initialization that already published its state,
                # including one begun by a version without a Project initializer.
                from ouroboros.project_registry_recovery import initialize_new_root
                initialize_new_root(drive_root)
            _write(drive_root, {**witness, "phase": "complete", "completed_at": utc_now_iso()})
        return not authority_reason(drive_root, initialization_id)
    except Exception:
        log.warning("state initialization witness could not be completed", exc_info=True)
        return False
