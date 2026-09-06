"""The owner-scoped settings WRITE seam — one lock, one commit boundary.

Every owner endpoint that persists SETTINGS goes through here: the generic
``POST /api/settings``, the FOUR single-decision owner endpoints (runtime mode,
auto-grant, context mode, safety mode), and the atomic
``POST /api/onboarding/complete``. Membership is defined by calling
``_owner_update_settings`` — directly with a transform, or through
``_owner_write_settings`` with a whole document — not by wearing the decorator:
the capability-evidence acknowledgement writes its own route-fingerprinted
ledger and never touches settings.json, so it is NOT one of these and holds no
settings lock. Four
invariants live in this module because each of them was previously
re-implemented (or silently skipped) per call site:

1. **The lock is a precondition, not a hint.** ``_acquire_settings_lock``
   returns ``None`` when it times out; writing anyway made "atomic" a claim the
   code did not keep. A failed acquisition now aborts BEFORE any precondition
   or write, with a typed ``SettingsLockUnavailable``.
2. **A precondition is proved under that lock**, against the state the write is
   about to overwrite — not against a read taken before a multi-second daemon
   call. A refusal aborts with ``SettingsPreconditionFailed`` and writes
   nothing.
3. **The commit boundary is visible.** Once ``write_text_atomic`` returns, the
   bytes ARE on disk; a failure in a LATER step (env projection, supervisor
   start, hot-reload side effects) must be reported as its own fact, never as
   "nothing was saved" (BIBLE P1). ``CommitBoundary`` carries that distinction
   from the writer to the response.
4. **``saved`` is a field on BOTH sides of that boundary.** Once a post-commit
   failure started answering ``saved=true``, an error envelope that merely omits
   the field became unreadable — the client cannot tell "nothing was written"
   from an old or truncated response. ``unsaved_error`` is the one pre-commit
   refusal shape, and it always carries ``saved=false``.
"""

from __future__ import annotations

import contextlib
import functools
import logging
import pathlib
import threading
from typing import Any, Callable, Dict, Optional, Sequence

from starlette.requests import Request
from starlette.responses import JSONResponse

from ouroboros.config import DATA_DIR
from ouroboros.config import SETTINGS_DEFAULTS as _SETTINGS_DEFAULTS
from ouroboros.context_mode_compat import normalize_context_mode_compat
from ouroboros.gateway._helpers import json_error, request_drive_root
from ouroboros.settings_integrity import SettingsIntegrityError, read_settings_json_verified
from ouroboros.utils import append_jsonl, utc_now_iso, write_text_atomic

log = logging.getLogger(__name__)

# The context mode and its one-window false provenance tombstone are authored together by
# the owner endpoint, never by a generic save (see prepare_settings_for_persist).
_CONTEXT_MODE_KEYS = ("OUROBOROS_CONTEXT_MODE", "OUROBOROS_CONTEXT_MODE_AUTO_LOW")


# In-PROCESS serialization for every read-merge-write on the settings document.
# The FILE lock inside ``_owner_update_settings`` serializes only the WRITES: a
# writer that reads the document, merges (possibly for a long time, off the
# event loop), then writes, would silently revert a single-decision endpoint
# that landed in between. Every event-loop writer used to inherit this
# serialization for free from the loop itself; a threaded writer does not.
# A read-fingerprint ``precondition`` (the onboarding transaction) refuses
# THIS writer's own stale merge, but cannot stop a lock-holding writer whose
# read predated this write from landing afterwards — so the onboarding
# transaction holds this lock TOO, from its write through its environment
# projection and hot-reload effects, symmetric with the generic save.
_settings_document_lock = threading.Lock()


class SettingsDocumentBusy(TimeoutError):
    """The in-process settings-document lock was not acquired within its bound.

    A writer wedged inside its hot-reload side effects would otherwise hold
    every later writer forever — the onboarding save among them ("Saving..."
    with nothing ever written). Typed so the endpoints answer it honestly.
    """


@contextlib.contextmanager
def settings_document_mutation():
    """Hold the in-process document lock across one read-merge-write (bounded)."""
    from ouroboros.config import get_settings_document_lock_timeout_sec

    timeout_sec = get_settings_document_lock_timeout_sec()
    if not _settings_document_lock.acquire(timeout=timeout_sec):
        raise SettingsDocumentBusy(
            f"another settings write is still in progress after {timeout_sec}s; "
            "nothing was written — try again"
        )
    try:
        yield
    finally:
        _settings_document_lock.release()


class SettingsPreconditionFailed(RuntimeError):
    """A locked-in precondition refused the write; nothing was persisted."""


class SettingsLockUnavailable(RuntimeError):
    """The settings lock could not be taken; nothing was checked or persisted."""


class CommitBoundary:
    """Where "saved" starts being true.

    ``committed`` flips the moment the settings bytes land on disk. Everything
    after that is a POST-commit step, named in ``stage`` so a failure can say
    WHICH step failed instead of implying the save itself did."""

    def __init__(self) -> None:
        self.committed = False
        self.stage = ""

    def commit(self) -> None:
        self.committed = True
        self.stage = ""

    def at(self, stage: str) -> None:
        self.stage = str(stage or "")


def unsaved_error(message: str, status: int = 400, **extra: Any) -> JSONResponse:
    """A settings write that failed BEFORE the commit. Nothing is on disk.

    The counterpart of ``post_commit_failure_response``, and it exists for the
    same reason: ``saved`` has to be a FIELD on both sides of the boundary. Once
    a post-commit failure started answering ``saved=true``, an envelope that
    merely omits ``saved`` became ambiguous — a client cannot tell "nothing was
    written" from an older or truncated response. Every pre-commit refusal on an
    owner settings-write surface goes through here."""
    return json_error(message, status, saved=False, **extra)


def post_commit_failure_response(exc: BaseException, boundary: CommitBoundary) -> JSONResponse:
    """The settings ARE saved and a later step failed. Say both, in that order."""
    stage = boundary.stage or "post-save"
    log.error("Settings saved, but the %s step failed afterwards", stage, exc_info=True)
    # Built directly rather than through ``json_error``, whose second parameter
    # IS named ``status`` — the envelope's own ``status`` field is a different
    # thing and must reach the body.
    return JSONResponse(
        {
            "error": (f"Settings were saved to disk, but the {stage} step failed "
                      f"afterwards: {type(exc).__name__}: {exc}"),
            "status": "saved_with_post_commit_error",
            "saved": True,
            "post_commit_failed": stage,
        },
        status_code=500,
    )


def owner_write_guard(endpoint: Callable) -> Callable:
    """Map a REFUSED owner settings write to an honest typed response.

    Without it a contended lock or a failed precondition leaves the endpoint
    raising into Starlette's 500 handler, which says nothing about whether the
    file changed. Both refusals persist nothing, so both are safe to retry.

    Belongs ONLY on an endpoint that actually calls ``_owner_update_settings``
    (directly, or through ``_owner_write_settings``).
    On one that does not, it translates exceptions that cannot be raised, and
    the decoration itself becomes the claim that the endpoint is lock-guarded —
    which the next reader (and the last reviewer) will believe."""

    @functools.wraps(endpoint)
    async def _guarded(request: Request) -> JSONResponse:
        try:
            return await endpoint(request)
        except SettingsLockUnavailable as exc:
            return unsaved_error(str(exc), 503, code="settings_locked")
        except SettingsDocumentBusy as exc:
            return unsaved_error(str(exc), 503, code="settings_busy")
        except SettingsPreconditionFailed as exc:
            return unsaved_error(str(exc), 409, code="settings_precondition_failed")

    return _guarded


def _owner_audit(request: Request, action: str, payload: Dict[str, Any]) -> None:
    try:
        drive_root = request_drive_root(request)
    except Exception:
        drive_root = pathlib.Path(DATA_DIR)
    try:
        client = getattr(request, "client", None)
        append_jsonl(
            drive_root / "logs" / "events.jsonl",
            {
                "ts": utc_now_iso(),
                "type": "owner_api_action",
                "action": str(action or ""),
                "client_host": str(getattr(client, "host", "") or ""),
                **{
                    key: value
                    for key, value in dict(payload or {}).items()
                    if "key" not in str(key).lower() and "secret" not in str(key).lower()
                },
            },
        )
    except Exception:
        log.debug("Failed to write owner API audit event", exc_info=True)


def settings_document_digest() -> str:
    """What the settings document looked like at a given instant.

    A digest of the raw BYTES, not of a parsed dict: it identifies the file a write is
    about to replace. Exactly two answers can ever COMPARE EQUAL — a digest, and the
    absent sentinel. An unreadable file is neither: it is returned as a value that never
    equals anything, itself included, because a stable ``unreadable:PermissionError``
    token on both sides would let a swap between two DIFFERENT unreadable files satisfy
    the check. That is fail-OPEN, and it is reachable — a reader silently falls back to
    defaults when it cannot read the file, while the atomic rename still lands because
    the parent directory is writable. So an unreadable settings file refuses the write."""
    from hashlib import sha256
    from uuid import uuid4

    from ouroboros.config import SETTINGS_PATH

    try:
        return sha256(SETTINGS_PATH.read_bytes()).hexdigest()
    except FileNotFoundError:
        return "absent"
    except OSError as exc:
        return f"unreadable:{type(exc).__name__}:{uuid4()}"


def _owner_read_settings_raw() -> Dict[str, Any]:
    """Read settings for owner endpoints without applying runtime-mode ratchets.

    "Raw" is about the RATCHETS, never about the migrations: the document is normalized
    through ``config.normalize_settings_raw`` BEFORE the defaults are merged, exactly as
    ``load_settings`` does. Skipping that step made every renamed slot answer its shipped
    default while the legacy key it should have been promoted from sat untouched in the
    same mapping — and because these endpoints write the mapping back, the defaults the
    merge invented were persisted as owner choices and the rename migration never fired
    again. The READ is the loader's too: one verified primitive, so a pinned benchmark
    snapshot that changed refuses this reader exactly as it refuses ``load_settings``
    instead of serving the unverified file as defaults-with-a-document. Only the
    ratchets and the one-window context-pair persistence are skipped here."""
    from ouroboros import config as _config

    merged = dict(_SETTINGS_DEFAULTS)
    try:
        raw = read_settings_json_verified(_config.SETTINGS_PATH)
        if isinstance(raw, dict):
            raw = normalize_context_mode_compat(
                raw, settings_path=_config.SETTINGS_PATH, warn_ambiguous=True,
            )
            merged.update(_config.normalize_settings_raw(raw))
    except SettingsIntegrityError:
        raise
    except Exception:
        log.debug("Failed to read raw owner settings; using defaults", exc_info=True)
    return merged


def _owner_write_settings(
    settings: Dict[str, Any],
    *,
    authored_keys: Sequence[str] = (),
    allow_context_lowering: bool = False,
    allow_safety_lowering: bool = False,
    precondition: Optional[Callable[[], str]] = None,
    boundary: Optional[CommitBoundary] = None,
) -> None:
    """Write owner-controlled settings without applying the runtime-mode ratchet.

    Skipping that ONE ratchet is the whole reason this writer exists; everything else comes from
    ``config.prepare_settings_for_persist``, the single point every persisting writer passes through.
    An endpoint that genuinely authors a disk-authored key (context mode, safety mode, the false
    compatibility tombstone) must name it in ``authored_keys`` — otherwise a POST about an unrelated key would
    author a mode decision out of the defaults merge that ``_owner_read_settings_raw`` performs.

    The settings lock is REQUIRED, not attempted: a timed-out acquisition raises
    ``SettingsLockUnavailable`` before the precondition runs and before anything is written, so a
    contended write can never be the one that skips the check it advertises.

    ``precondition`` (optional) is re-evaluated INSIDE that lock, immediately before the write, so
    an install-time transaction proves its eligibility against the state it is about to overwrite;
    a non-empty return value aborts with ``SettingsPreconditionFailed``. ``boundary`` (optional) is
    marked committed the moment the bytes land, so the caller can tell a failed save from a failed
    post-save step."""
    _owner_update_settings(
        lambda _current: settings,
        authored_keys=authored_keys,
        allow_context_lowering=allow_context_lowering,
        allow_safety_lowering=allow_safety_lowering,
        precondition=precondition,
        boundary=boundary,
    )


STALE_SETTINGS_READ_REFUSAL = (
    "The settings file changed while this change was being saved, so saving it would have "
    "overwritten that change; nothing was written. Try again."
)


def _owner_update_settings(
    transform: Callable[[Dict[str, Any]], Optional[Dict[str, Any]]],
    expected_digest: str = "",
    *,
    authored_keys: Sequence[str] = (),
    allow_context_lowering: bool = False,
    allow_safety_lowering: bool = False,
    precondition: Optional[Callable[[], str]] = None,
    boundary: Optional[CommitBoundary] = None,
) -> None:
    """Read, change and persist ONE settings document inside ONE settings lock.

    An owner endpoint changes a single decision inside a document it does not otherwise
    own, so it must read the whole document and write the whole document back. Doing that
    around the lock rather than inside it makes every such endpoint a last-writer-wins
    race: a concurrent owner change that lands between the read and the write is reverted
    key by key while this request answers "saved" (BIBLE P1). Here ``transform`` receives
    the settings as they are INSIDE the lock and returns the document to persist, or
    ``None`` to persist nothing — which is also how a no-change decision avoids rewriting
    the file at all.

    ``expected_digest`` closes the other half: an endpoint that took a DECISION from an
    earlier read (the previous mode, whether anything changed at all) passes the digest
    that read saw, and a mismatch refuses with ``SettingsPreconditionFailed`` before the
    transform runs. It is the same fingerprint precondition the onboarding transaction
    uses, and it deliberately over-refuses in two narrow cases — a write landing in the
    microseconds between digest and read, and a formatting-only rewrite of identical
    content — rather than risk under-refusing in any. Both cost one retry; the opposite
    error costs the owner a change they made.

    Everything else is the contract ``_owner_write_settings`` already advertised, now
    genuinely held: the lock is REQUIRED (a timed-out acquisition raises
    ``SettingsLockUnavailable`` before anything is read, checked or written), the
    persistence prologue (`config.prepare_settings_for_persist`, which proves the
    context/safety ratchets against the value ON DISK) runs while that lock is held rather
    than before it, and ``boundary`` flips the instant the bytes land."""
    from ouroboros import config as _config

    _config._guard_live_settings_write()
    _config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    fd = _config._acquire_settings_lock()
    if fd is None:
        raise SettingsLockUnavailable(
            f"Another process is holding the settings lock ({_config._settings_lock_path()}); "
            "nothing was saved. Retry in a moment."
        )
    try:
        if expected_digest and settings_document_digest() != expected_digest:
            raise SettingsPreconditionFailed(STALE_SETTINGS_READ_REFUSAL)
        if precondition is not None:
            refusal = str(precondition() or "")
            if refusal:
                raise SettingsPreconditionFailed(refusal)
        proposed = transform(_owner_read_settings_raw())
        if proposed is None:
            return
        to_write = _config.prepare_settings_for_persist(
            dict(proposed), authored_keys=authored_keys,
            allow_context_lowering=allow_context_lowering,
            allow_safety_lowering=allow_safety_lowering)
        write_text_atomic(_config.SETTINGS_PATH, _config.serialize_settings(to_write))
        if boundary is not None:
            boundary.commit()
    finally:
        _config._release_settings_lock(fd)


__all__ = [
    "CommitBoundary",
    "STALE_SETTINGS_READ_REFUSAL",
    "SettingsLockUnavailable",
    "SettingsPreconditionFailed",
    "settings_document_mutation",
    "_CONTEXT_MODE_KEYS",
    "_owner_audit",
    "_owner_read_settings_raw",
    "_owner_update_settings",
    "_owner_write_settings",
    "owner_write_guard",
    "post_commit_failure_response",
    "settings_document_digest",
    "unsaved_error",
]
