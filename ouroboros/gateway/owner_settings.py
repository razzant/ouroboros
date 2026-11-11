"""The owner-scoped settings WRITE seam — one lock, one commit boundary.

Every owner endpoint that persists SETTINGS goes through here: the generic
``POST /api/settings``, the single-decision owner endpoints (runtime mode,
auto-grant, context mode, the Safety model, levels and rules, desktop keep-running),
a model-wait card's saved choice, and the atomic ``POST /api/onboarding/complete``. Membership is defined by calling
``_owner_update_settings`` — directly with a transform, or through
``_owner_write_settings`` with a whole document — not by wearing the decorator:
the capability-evidence acknowledgement writes its own route-fingerprinted
ledger and never touches settings.json, so it is NOT one of these and holds no
settings lock. Incidental writers outside those endpoints (the launcher's confirmed
decisions, the agent's ``set_tool_timeout``) pass ``fields_on_saved_document``: their
named fields land and nothing else, so they author no runtime mode they did not name. Four
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
5. **The writer's receipt is the settings history.** ``_owner_update_settings``
   returns the before → after of every key its commit changed, taken under its
   own lock; the generic save, the Safety owner endpoints and a model-wait card's
   saved choice record exactly that (``record_settings_change``), so no endpoint
   keeps a key list of its own. The other writers keep no such record.
"""

from __future__ import annotations

import contextlib
import functools
import json
import logging
import pathlib
import re
import threading
from typing import Any, Callable, Dict, Optional, Sequence

from starlette.requests import Request
from starlette.responses import JSONResponse

from ouroboros.config import defaults_for_settings_document as _defaults_for_settings_document
from ouroboros.context_mode_compat import normalize_context_mode_compat
from ouroboros.control_actor import PAUSE_AUTHORS
from ouroboros.gateway._helpers import json_error, request_drive_root
from ouroboros.settings_integrity import (
    SettingsDocumentUnknown, SettingsIntegrityError, read_settings_json_strict, read_settings_json_verified,
)
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

    The bound refuses the waiting writer; it never interrupts the holder, which
    keeps the lock until its body ends.
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


class SettingsChangedElsewhere(SettingsPreconditionFailed):
    """A per-key condition no longer describes the stored value."""


class SettingsDocumentUnreadable(SettingsPreconditionFailed, SettingsDocumentUnknown):
    """The settings document exists but does not read as a JSON object (an open the OS refuses, broken
    JSON, another JSON type). Only an absent document is new: this one is never merged into or written
    over as if it were empty, and its bytes stay exactly as they are. The owner endpoints' form of the
    one strict read's ``SettingsDocumentUnknown`` (``settings_integrity.read_settings_json_strict``)."""


UNREADABLE_SETTINGS_REFUSAL = (
    "settings.json exists but could not be read as a settings document, so a write would replace what "
    "it holds; nothing was written. Repair or move the file, then try again."
)


class SettingsLockUnavailable(RuntimeError):
    """The settings lock could not be taken; nothing was checked or persisted."""


# The typed code of a write the deployment policy refused (``deployment_policy.DeploymentPolicyRefusal``).
MANAGED_BY_DEPLOYMENT = "managed_by_deployment"


class CommitBoundary:
    """Where "saved" starts being true.

    ``committed`` flips the moment the settings bytes land on disk. Everything
    after that is a POST-commit step, named in ``stage`` so a failure can say
    WHICH step failed instead of implying the save itself did."""

    def __init__(self) -> None:
        self.committed = False
        self.stage = ""
        self.receipt: dict = {"receipt": "unavailable"}

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
            **boundary.receipt,
        },
        status_code=500,
    )


def settings_write_refusal(exc: Exception) -> JSONResponse | None:
    """The same typed pre-commit refusals on generic and dedicated owner writers."""
    from ouroboros.deployment_policy import DeploymentPolicyRefusal

    for error, status, code in (
        (DeploymentPolicyRefusal, 403, MANAGED_BY_DEPLOYMENT),
        (SettingsLockUnavailable, 503, "settings_locked"),
        (SettingsDocumentBusy, 503, "settings_busy"),
        (SettingsDocumentUnreadable, 409, "settings_unreadable"),
        (SettingsChangedElsewhere, 409, "settings_changed_elsewhere"),
        (SettingsPreconditionFailed, 409, "settings_precondition_failed"),
    ):
        if isinstance(exc, error):
            return unsaved_error(str(exc), status, code=code)
    return None


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
        except Exception as exc:
            if refusal := settings_write_refusal(exc):
                return refusal
            raise

    return _guarded


def owner_audit(request: Request, action: str, payload: Dict[str, Any]) -> None:
    """Record an owner action, including OS controls outside settings.json."""
    try:
        drive_root = request_drive_root(request)
    except Exception:
        from ouroboros import config

        drive_root = pathlib.Path(config.DATA_DIR)
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


_owner_audit = owner_audit  # Existing settings writers retain their import surface.

# A settings write may declare where it came from, as a Pause press declares its author: a declaration
# grants nothing and identifies nobody (``client_host`` is never an actor). The app's own pages declare
# ``app``; a write without a valid declaration came from the ``api`` and its actor is ``not declared``.
SURFACE_HEADER, ACTOR_HEADER, TASK_HEADER = "X-Ouroboros-Surface", "X-Ouroboros-Actor", "X-Ouroboros-Task-Id"
APP_SURFACE = "app"
_SURFACE_RE = re.compile(r"[a-z][a-z0-9_-]{0,31}")
_ROW_PHRASES = {  # the two decisions the owner reads by name; any other key is said by its name
    "OUROBOROS_MODEL": "Main model: {old} → {new}.",
    "OUROBOROS_CONSCIOUSNESS_DAILY_USD": "Consciousness allowance: {old} → {new} USD per rolling 24 hours.",
}
# Who changed it, in the owner's words; the surface and the task stay in the event (DESIGN principle 7).
_ROW_ACTORS = {"ouroboros": "Changed by Ouroboros.", "owner": "Changed by you."}


def declared_settings_origin(request: Request) -> Dict[str, str]:
    """``{surface, actor[, actor_task_id]}`` as the request declared them; malformed parts are dropped."""
    from ouroboros.task_results import validate_task_id

    headers = getattr(request, "headers", None) or {}
    surface = str(headers.get(SURFACE_HEADER) or "").strip().lower()
    actor = str(headers.get(ACTOR_HEADER) or "").strip().lower()
    origin = {"surface": surface if _SURFACE_RE.fullmatch(surface) else "api",
              "actor": actor if actor in PAUSE_AUTHORS else "not declared"}
    try:
        origin["actor_task_id"] = validate_task_id(headers.get(TASK_HEADER))
    except ValueError:
        pass
    return origin


def _withheld(key: str, change: Dict[str, Any]) -> str:
    """Why a change keeps only its key name: ``secret`` (the contract's registry, an owner-defined
    secret) or ``structure`` (rows, maps, a roster: they can carry credentials); ``""`` for a scalar."""
    from ouroboros.config import SETTINGS_DEFAULTS
    from ouroboros.secret_masking import is_custom_secret_setting_key
    from ouroboros.settings_setup_contract import SECRET_SETTING_KEYS, secret_provider_setting_keys

    if (key in SECRET_SETTING_KEYS or key in secret_provider_setting_keys()
            or is_custom_secret_setting_key(key, known_setting_keys=SETTINGS_DEFAULTS)):
        return "secret"
    for value in change.values():
        if isinstance(value, str) and value.lstrip()[:1] in ("[", "{"):
            try:
                value = json.loads(value)
            except ValueError:
                continue
        if isinstance(value, (dict, list)):
            return "structure"
    return ""


def _shown(change: Dict[str, Any], side: str) -> str:
    if side not in change:
        return "not set"
    value = change[side]
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return (value or "empty") if isinstance(value, str) else json.dumps(value)


def settings_change_text(changes: Dict[str, Dict[str, Any]], origin: Dict[str, str]) -> str:
    """The one System row of a change the app did not declare: what changed and who changed it, in one
    line, so a withheld value or a text of several lines (the owner's rules) is said by its key alone. The
    event keeps the text, the surface, the declared actor and its task."""
    parts = [f"{key} changed." if "withheld" in change or any("\n" in str(value) for value in change.values())
             else _ROW_PHRASES.get(key, key + ": {old} → {new}.").format(old=_shown(change, "old"), new=_shown(change, "new"))
             for key, change in changes.items()]
    return " ".join(parts) + " " + _ROW_ACTORS.get(str(origin.get("actor") or ""), "Changed outside the app.")


def project_settings_changes(changes: dict, *, tokens: bool = False) -> dict:
    """The audit and HTTP share this whitelist; raw receipts can contain credentials."""
    projected = {}
    for key, change in changes.items():
        reason = _withheld(key, {side: change[side] for side in ("old", "new") if side in change})
        row = {"withheld": reason} if reason else {side: change[side] for side in ("old", "new") if side in change}
        if tokens and reason != "secret":
            row.update({field: change[field] for field in (("token",) if reason else ("token", "old_absent")) if field in change})
        projected[key] = row
    return projected


def settings_receipt(changes: dict | None) -> dict:
    """No receipt is different from a receipt of no change. Projection cannot unsave bytes."""
    try:
        if changes is not None:
            return {"changes": project_settings_changes(changes, tokens=True)}
    except Exception:
        log.warning("Settings saved; public receipt unavailable", exc_info=True)
    return {"receipt": "unavailable"}


def merged_changes(changes: dict | None, followup: dict | None) -> dict | None:
    """One receipt for a save and its own follow-up write: the save's old side, the follow-up's new side."""
    if changes is None or followup is None:
        return None
    return {**changes, **{key: {**change, **{side: changes[key][side] for side in ("old", "old_absent")
                                            if side in changes.get(key, ())}} for key, change in followup.items()}}


def record_settings_change(request: Request | None, changes: Optional[Dict[str, Dict[str, Any]]], *,
                           origin: Dict[str, str] | None = None) -> None:
    """Record one committed settings write: the writer's receipt, never a second reading of the file.

    One ``owner_api_action`` (``action="settings_changed"``) names every changed key: a scalar with its
    effective old/new, a secret or a structure as
    ``{"withheld": "secret"|"structure"}``, plus the declared origin. A write the app did not declare also
    puts one quiet System row (``settings_changed``) in Main; the app's own Save already says what it
    saved. Neither step can undo or fail the save."""
    if not changes:
        return
    try:
        origin = origin or declared_settings_origin(request)
        projected = project_settings_changes(changes)
        owner_audit(request, "settings_changed", {**origin, "changes": projected})
        if origin["surface"] == APP_SURFACE:
            return
        from supervisor import message_bus
        from supervisor.state import control_value, load_state

        known, owner_chat = control_value(load_state(), "owner_chat_id")
        if known and owner_chat:  # a display-only backup value never addresses Main
            message_bus.send_with_budget(int(owner_chat), settings_change_text(projected, origin),
                                         role="system", system_type="settings_changed")
    except Exception:
        log.warning("Settings saved; their change record or Main row failed", exc_info=True)


OWNER_SETTING_WRITERS = {
    "OUROBOROS_CONTEXT_MODE": "POST /api/owner/context-mode",
    "OUROBOROS_RUNTIME_MODE": "POST /api/owner/runtime-mode",
    "OUROBOROS_AUTO_GRANT_REVIEWED_SKILLS": "POST /api/owner/auto-grant",
    "OUROBOROS_SAFETY_MODE": "POST /api/owner/safety-mode",
    "OUROBOROS_CONSCIOUSNESS_SAFETY_MODE": "POST /api/owner/safety-mode",
    "OUROBOROS_MODEL_SAFETY": "POST /api/owner/safety-model",
    "OUROBOROS_SAFETY_OWNER_RULES": "POST /api/owner/safety-rules",
}


def setting_tokens(document: dict, keys: Sequence[str]) -> dict[str, str]:
    """Hash canonical stored values, including presence. Never hash a scalar secret."""
    from hashlib import sha256
    from ouroboros.config import normalize_settings_raw

    canonical = normalize_settings_raw(document)
    return {key: sha256(json.dumps([key in canonical, canonical.get(key)], sort_keys=True,
                                  ensure_ascii=False, separators=(",", ":")).encode("utf-8")).hexdigest()[:24]
            for key in keys if _withheld(key, {}) != "secret"}


def check_setting_tokens(document: dict, condition: Any) -> None:
    if condition is None:
        return
    if not isinstance(condition, dict) or any(not isinstance(k, str) or not isinstance(v, str) for k, v in condition.items()):
        raise ValueError("'if_unchanged' must map setting keys to tokens")
    actual = setting_tokens(document, tuple(condition))
    if set(condition) - set(actual):
        raise ValueError("Secret settings cannot have an 'if_unchanged' token")
    if any(actual[key] != token for key, token in condition.items()):
        raise SettingsChangedElsewhere("Changed on another device. Reload the current value before saving.")


def settings_read_view() -> tuple[dict, dict]:
    """GET values and tokens share one source read, including the loader's compatibility rules."""
    from ouroboros import config

    raw = read_settings_json_verified(config.SETTINGS_PATH, behind_writer=True)
    present = raw is not None or config.SETTINGS_PATH.exists()
    raw = normalize_context_mode_compat(raw, settings_path=config.SETTINGS_PATH, warn_ambiguous=True) if isinstance(raw, dict) else {}
    return config.settings_of_document(raw, present=present), config.normalize_settings_raw(raw)


def settings_in_place_of(document: dict) -> dict:
    """What the loader reads for ``document`` once the values this tree mirrored into the environment no longer
    stand in for a key it leaves unset (``skill_environ``); an operator's own variable still does."""
    from ouroboros import config
    from ouroboros.settings_integrity import settings_read_environ, skill_environ

    with settings_read_environ(skill_environ()):
        return config.settings_of_document(document)


def settings_roster_error(raw: Any, stored: dict, body: dict) -> str:
    """A switch-only edit is judged on the rows it switches: a stored twin pair never blocks another
    row's switch or the roster's own, while switching a twin keeps that twin. Other edits are judged whole."""
    from ouroboros.configured_subagents import (
        SUBAGENTS_SETTING, configured_subagents_dict, parse_configured_subagents, roster_save_error, subagent_handle,
    )

    try:
        before, after = (parse_configured_subagents(value) for value in (stored.get(SUBAGENTS_SETTING), raw))
    except (TypeError, ValueError):
        return roster_save_error(raw, stored, body)  # the authoritative parser and judge answer
    def seats(catalog, facts):
        return [({k: v for k, v in row.items() if k != "enabled"}, item.route.kind, subagent_handle(item, facts))
                for row, item in zip(configured_subagents_dict(catalog)["items"], catalog.items)]
    old, new = seats(before, stored), seats(after, {**stored, **body})
    engines = [seat[1:] for seat in new]
    switched = [seat[1:] for seat, a, b in zip(new, before.items, after.items) if a.enabled != b.enabled]
    if old != new or any(engines.count(engine) > 1 for engine in switched):
        return roster_save_error(raw, stored, body)
    return ""


def settings_unset_keys(body: dict) -> tuple[str, ...]:
    """Undo may restore omission only for ordinary, non-secret settings; never owner controls."""
    from ouroboros.config import SETTINGS_DEFAULTS, ENDPOINT_AUTHORED_SETTINGS, ENDPOINT_WRITTEN_SETTINGS

    keys = body.get("unset", [])
    forbidden = set(OWNER_SETTING_WRITERS) | ENDPOINT_AUTHORED_SETTINGS | ENDPOINT_WRITTEN_SETTINGS | set(_CONTEXT_MODE_KEYS)
    if (not isinstance(keys, list) or any(not isinstance(key, str) or key not in SETTINGS_DEFAULTS
            or key in forbidden or key in body or _withheld(key, {}) == "secret" for key in keys)):
        raise ValueError("'unset' must list ordinary non-secret settings, without also assigning them")
    return tuple(keys)


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


def _owner_read_settings_raw(document_out: Optional[Dict[str, Any]] = None, *, strict: bool = False,
                             behind_writer: bool = False) -> Dict[str, Any]:
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
    ratchets and the one-window context-pair persistence are skipped here. ``document_out``
    receives the normalized document alone: what the owner has authored, no defaults.

    ``strict`` (every write, and a form that judges a key against the document) tells an absent
    document, which is new and empty, from a present one that cannot be read as an object, which
    raises ``SettingsDocumentUnreadable`` instead of reading as defaults. It is the one strict read
    the skills' settings use (``read_settings_json_strict``). ``behind_writer`` (a reader outside
    the settings lock) reads a torn document once more behind that lock, as a skill's read does; a
    writer already holds the lock, which is not re-entrant, so it never asks for it."""
    from ouroboros import config as _config

    # An existing (even unreadable) document keeps the optional bounds it ran under.
    merged = _defaults_for_settings_document(_config.SETTINGS_PATH.exists())
    document: Dict[str, Any] = {}
    try:
        if strict:
            try:
                raw = read_settings_json_strict(_config.SETTINGS_PATH, behind_writer=behind_writer)
            except SettingsDocumentUnknown as exc:
                raise SettingsDocumentUnreadable(UNREADABLE_SETTINGS_REFUSAL) from exc
        else:
            raw = read_settings_json_verified(_config.SETTINGS_PATH)
        if isinstance(raw, dict):
            raw = normalize_context_mode_compat(
                raw, settings_path=_config.SETTINGS_PATH, warn_ambiguous=True,
            )
            document = _config.normalize_settings_raw(raw)
    except (SettingsIntegrityError, SettingsDocumentUnreadable):
        raise
    except Exception as exc:
        if strict:
            raise SettingsDocumentUnreadable(UNREADABLE_SETTINGS_REFUSAL) from exc
        log.debug("Failed to read raw owner settings; using defaults", exc_info=True)
    if document_out is not None:
        document_out.update(document)
    merged.update(_config.normalize_effort_document(document))
    return merged


def _settings_document_present() -> bool:
    from ouroboros import config as _config

    try:
        return _config.SETTINGS_PATH.exists()
    except OSError:
        return True  # a document that cannot even be looked at is never an absent one


def _owner_settings_document(*, strict: bool = False) -> Dict[str, Any]:
    """The normalized settings document alone (what the owner authored, no defaults); ``strict`` as
    in ``_owner_read_settings_raw``, read outside the settings lock (never by a writer holding it), so
    a document a writer is replacing is read once more behind that lock instead of called unreadable."""
    document: Dict[str, Any] = {}
    _owner_read_settings_raw(document_out=document, strict=strict, behind_writer=strict)
    return document


class _SavedDocumentFields:
    """``fields_on_saved_document``'s transform. The writer hands it the authored document of its own
    single read under the lock, so the optional check and the merge judge exactly what is replaced."""

    def __init__(self, changes: Dict[str, Any], check: Optional[Callable[[Dict[str, Any]], str]]) -> None:
        self.changes, self.check = changes, check

    def __call__(self, document: Dict[str, Any]) -> Dict[str, Any]:
        from ouroboros import config as _config
        from ouroboros.deployment_policy import DeploymentPolicyRefusal, choice_refusal
        from ouroboros.settings_scales import OPTIONAL_BOUND_LEGACY

        refusal = str(self.check(document) or "") if self.check is not None else ""
        if refusal:
            raise SettingsPreconditionFailed(refusal)
        if managed := next(filter(None, (choice_refusal(key, value) for key, value in self.changes.items())), ""):
            raise DeploymentPolicyRefusal(managed)  # each field is an explicit choice, even one the document holds
        if _settings_document_present():
            return {**document, **self.changes}
        effective = _config.load_settings_lock_held()
        return {**{key: effective[key] for key in OPTIONAL_BOUND_LEGACY}, **self.changes}


def fields_on_saved_document(changes: Dict[str, Any],
                             check: Optional[Callable[[Dict[str, Any]], str]] = None) -> _SavedDocumentFields:
    """The ``_owner_update_settings`` transform of an incidental writer: exactly ``changes`` land on
    the document as it is on disk inside that lock. Nothing the merged read shows is authored beside
    them (a default, an environment value, an unauthored proposal) and every saved value stays as the
    read translates it, a pending next-boot runtime mode included, so the ratchets meet only unchanged
    values. A document this creates carries the optional bounds as they resolve now, because a
    document's presence turns their absence into the legacy finite value
    (``settings_scales.defaults_for_settings_document``). ``check`` (optional) judges that same document
    first; a non-empty reason refuses with ``SettingsPreconditionFailed`` and writes nothing."""
    return _SavedDocumentFields(changes, check)


def _owner_write_settings(
    settings: Dict[str, Any],
    *,
    authored_keys: Sequence[str] = (),
    allow_context_lowering: bool = False,
    allow_safety_lowering: bool = False,
    precondition: Optional[Callable[[], str]] = None,
    boundary: Optional[CommitBoundary] = None,
    expected_digest: str = "",
    if_unchanged: Optional[dict] = None,
    unset: Sequence[str] = (),
) -> Optional[Dict[str, Dict[str, Any]]]:
    """Write owner-controlled settings without applying the runtime-mode ratchet; returns the receipt.

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
    return _owner_update_settings(
        lambda _current: settings,
        authored_keys=authored_keys,
        allow_context_lowering=allow_context_lowering,
        allow_safety_lowering=allow_safety_lowering,
        precondition=precondition,
        boundary=boundary,
        whole_document=True, expected_digest=expected_digest, if_unchanged=if_unchanged, unset=unset,
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
    whole_document: bool = False,
    if_unchanged: Optional[dict] = None,
    unset: Sequence[str] = (),
) -> Optional[Dict[str, Dict[str, Any]]]:
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
    than before it, and ``boundary`` flips the instant the bytes land. A model slot the document
    never authored and the transform left as read is not written (``unedited_unauthored_slots``),
    unless the caller names it in ``authored_keys`` (a wait card's explicit choice, default-equal
    included) or hands over a ``whole_document`` it authored (Settings save, onboarding).

    The read is strict: only an absent document is new, and a present one that does not read as
    an object refuses every writer with ``SettingsDocumentUnreadable``, its bytes untouched. A
    ``fields_on_saved_document`` transform receives that one read's authored document.

    Returns the commit's receipt (``_committed_changes``), taken under the same lock after the bytes
    land; empty when nothing was written or nothing changed."""
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
        if precondition is not None:
            refusal = str(precondition() or "")
            if refusal:
                raise SettingsPreconditionFailed(refusal)
        document: Dict[str, Any] = {}
        current = _owner_read_settings_raw(document_out=document, strict=True)
        if expected_digest and settings_document_digest() != expected_digest:
            raise SettingsPreconditionFailed(STALE_SETTINGS_READ_REFUSAL)
        check_setting_tokens(document, if_unchanged)
        shown = dict(current)
        effective_before = _config.settings_of_document(document, present=_settings_document_present())
        proposed = transform(dict(document) if isinstance(transform, _SavedDocumentFields) else current)
        if proposed is None:
            return {}
        to_write = dict(proposed)
        if not whole_document:
            from ouroboros.server_runtime import unedited_unauthored_slots

            for key in unedited_unauthored_slots(document, to_write, shown) - set(authored_keys):
                to_write.pop(key, None)
        to_write = _config.prepare_settings_for_persist(
            to_write, authored_keys=authored_keys,
            allow_context_lowering=allow_context_lowering,
            allow_safety_lowering=allow_safety_lowering)
        for key in unset:
            to_write.pop(key, None)
        write_text_atomic(_config.SETTINGS_PATH, _config.serialize_settings(to_write))
        if boundary is not None:
            boundary.commit()
        try:
            changes = _committed_changes(document, shown, to_write, effective_before)
            if boundary is not None:
                boundary.receipt = settings_receipt(changes)
            return changes
        except Exception:  # the bytes are saved; a receipt that cannot be taken never unsaves them
            log.warning("Settings saved; their change receipt could not be taken", exc_info=True)
            return None
    finally:
        _config._release_settings_lock(fd)


def _committed_changes(document: dict, shown: dict, written: dict, effective_before: dict | None = None) -> dict:
    """Each key one commit changed, with the token it installed, taken under the file lock.

    A key changed when its document entry did (judged over the same defaults and types, so writing a
    default out is no change) and the value in force moved. ``old`` is that value immediately before:
    the environment, proposals and provider spelling the owner was shown. ``new`` is the written value
    or, for a key the commit removed, what the loader reads in its place once the values this tree
    mirrored into the environment no longer stand in (``skill_environ``)."""
    from ouroboros import config
    from ouroboros.server_runtime import apply_runtime_provider_defaults
    from ouroboros.settings_scales import normalize_effort_document

    before = config.coerce_settings_raw(shown)
    after = config.coerce_settings_raw({**_defaults_for_settings_document(True), **normalize_effort_document(written)})
    in_force = config.coerce_settings_raw(apply_runtime_provider_defaults(dict(effective_before or shown))[0])
    removed = {key for key in document if key not in written}
    in_place = config.coerce_settings_raw(settings_in_place_of(written)) if removed else {}
    changes = {}
    for key in sorted(set(before) | set(after)):
        if before.get(key, _ABSENT) == after.get(key, _ABSENT) and key not in removed:
            continue
        sides = {"old": in_force.get(key, before.get(key, _ABSENT)),
                 "new": (in_place if key in removed else after).get(key, _ABSENT)}
        if sides["old"] != sides["new"]:
            changes[key] = {**{side: value for side, value in sides.items() if value is not _ABSENT},
                            **({"old_absent": True} if key not in document else {})}
    for key, token in setting_tokens(written, tuple(changes)).items():
        changes[key]["token"] = token
    return changes


_ABSENT = object()


__all__ = [
    "CommitBoundary",
    "MANAGED_BY_DEPLOYMENT",
    "STALE_SETTINGS_READ_REFUSAL",
    "SettingsDocumentUnreadable",
    "SettingsLockUnavailable",
    "SettingsPreconditionFailed", "SettingsChangedElsewhere",
    "OWNER_SETTING_WRITERS", "setting_tokens", "settings_receipt", "merged_changes", "settings_read_view", "settings_in_place_of",
    "project_settings_changes", "settings_unset_keys", "settings_roster_error", "settings_write_refusal",
    "UNREADABLE_SETTINGS_REFUSAL",
    "settings_document_mutation",
    "_CONTEXT_MODE_KEYS",
    "_owner_audit",
    "owner_audit",
    "_owner_read_settings_raw",
    "_owner_settings_document",
    "_owner_update_settings",
    "fields_on_saved_document",
    "_owner_write_settings",
    "owner_write_guard",
    "post_commit_failure_response",
    "record_settings_change",
    "settings_document_digest",
    "unsaved_error",
]
