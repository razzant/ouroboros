"""The interface language and its translation memory: the browser gateway.

One install-wide fact, ``OUROBOROS_UI_LANGUAGE`` (``ouroboros/ui_language.py``), and one
memory file per language (``ouroboros/i18n_memory.py``). The browser reads both here
and never receives a dictionary from the repository; Telegram and host-composed text
read the same memory through ``i18n_memory.tr``.

Routes:

* ``GET  /api/ui/i18n``             — the current language, its memory and status counts
  (side-effect free: a read never starts a model call).
* ``POST /api/ui/i18n/language``    — choose the language. Writes the settings key through
  the locked owner-settings writer, creates the memory header, broadcasts
  ``ui_language_changed`` and hands the tag to the generator hooks. A non-tag
  description ("Quenya", "invent a language") is the generator's job to normalize into a
  profile; until that hook answers, a tag is required.
* ``POST /api/ui/i18n/missing``     — the renderer's unknown strings, shape-filtered and
  bounded, queued for the generator.
* ``POST /api/ui/i18n/import``      — an enterprise/community memory file (schema 1).
* ``GET  /api/ui/i18n/export``      — the memory file as a download.
* ``POST /api/ui/i18n/regenerate``  — drop generated entries (owner and imported stay) and
  let the generator rebuild.

Frames: ``ui_language_changed`` (tag) and ``ui_i18n_updated`` (tag, revision). The posting
client paints from the POST body; every other client re-reads ``GET /api/ui/i18n`` on
the frame and on reconnect, because the settings save path broadcasts nothing.
"""

from __future__ import annotations

import logging
import os
import pathlib
from typing import Any, Callable, Dict, List, Optional, Tuple

from starlette.requests import Request
from starlette.responses import JSONResponse

from ouroboros import i18n_memory as memory
from ouroboros.gateway._helpers import json_error, request_drive_root, request_json_or
from ouroboros.gateway.owner_settings import (
    _owner_audit,
    _owner_update_settings,
    owner_write_guard,
    settings_document_mutation,
    unsaved_error,
)
from ouroboros.ui_language import LANGUAGE_NOT_CHOSEN, SETTING_KEY, is_english, normalize_language_tag

log = logging.getLogger(__name__)

MAX_MISSING_ITEMS = 200
_LANGUAGE_HOOKS: List[Callable[[str, pathlib.Path, str], None]] = []


def register_language_hook(fn: Callable[[str, pathlib.Path, str], None]) -> None:
    """``fn(event, drive_root, tag)`` for ``language_set`` | ``missing`` | ``regenerate`` |
    ``imported``; the generator registers itself here. Hooks never raise into a response."""
    if fn not in _LANGUAGE_HOOKS:
        _LANGUAGE_HOOKS.append(fn)


def _notify(event: str, drive_root: pathlib.Path, tag: str) -> None:
    for fn in list(_LANGUAGE_HOOKS):
        try:
            fn(event, drive_root, tag)
        except Exception:
            log.warning("ui language hook %s failed on %s", getattr(fn, "__name__", fn), event, exc_info=True)


def _broadcast(message: Dict[str, Any]) -> None:
    try:
        from ouroboros.gateway.ws import broadcast_ws_sync

        broadcast_ws_sync(message)
    except Exception:
        log.debug("ui language broadcast skipped", exc_info=True)


def _source_hashes() -> Optional[Dict[str, str]]:
    """Current English of the code/fmt keys, from the generator's catalog registry when
    it is loaded; ``None`` keeps the stale count honestly unknown."""
    try:
        from ouroboros import ui_translation  # noqa: WPS433 (optional generator)
    except Exception:
        return None
    collect = getattr(ui_translation, "catalog_source_hashes", None)
    if collect is None:
        return None
    try:
        return collect()
    except Exception:
        log.debug("catalog source hashes unavailable", exc_info=True)
        return None


def _generator_status() -> Dict[str, Any]:
    try:
        from ouroboros.ui_translation import generator_status

        return generator_status()
    except Exception:
        return {"state": "idle", "error": "", "updated_at": "", "applied": 0, "language": "", "in_flight": 0}


def memory_payload(drive_root: pathlib.Path, tag: str) -> Dict[str, Any]:
    """The GET body for ``tag``: entries only for a non-English language, counts always."""
    english = is_english(tag)
    doc: Optional[Dict[str, Any]] = None
    error = ""
    if not english:
        try:
            doc = memory.load_memory(drive_root, tag)
        except memory.MemoryFormatError as exc:
            error = str(exc)
    generator = _generator_status()
    pending = 0 if english else memory.pending_count(drive_root, tag)
    if not english and generator.get("language") == tag:
        pending += int(generator.get("in_flight") or 0)  # a batch out at the model is still pending work
    # The lexicon (up to 60k chars of an invented language's rules) is the generator's prompt
    # material, kept in the file and the export, never shipped to every client on every frame.
    profile = ({k: v for k, v in doc["profile"].items() if k != "lexicon"} | {"lexicon_chars": len(doc["profile"].get("lexicon") or "")}) if doc else None
    return {
        "language": tag,
        "chosen": tag != LANGUAGE_NOT_CHOSEN,
        "english": english,
        "revision": int(doc["revision"]) if doc else 0,
        "profile": profile,
        "plural_select": doc.get("plural_select") if doc else None,
        "plural_categories": doc.get("plural_categories") if doc else None,
        "entries": doc["entries"] if doc else {},
        "stats": memory.stats(doc, source_hashes=_source_hashes(), pending=pending),
        "updated_at": str(doc.get("updated_at") or "") if doc else "",
        "memory_error": error,
        "languages": memory.list_languages(drive_root),
        "generator": generator,
    }


async def api_ui_i18n_get(request: Request) -> JSONResponse:
    drive_root = request_drive_root(request)
    try:
        return JSONResponse(memory_payload(drive_root, memory.current_language()))
    except Exception as exc:  # a read must answer; the error is the payload
        log.warning("ui i18n read failed", exc_info=True)
        return JSONResponse({**memory_payload(drive_root, LANGUAGE_NOT_CHOSEN), "memory_error": str(exc)})


def _merge_header(doc: Dict[str, Any], profile: Dict[str, Any], plural_select: Any,
                  plural_categories: Any) -> Optional[Dict[str, Any]]:
    changed = False
    if profile:
        merged = {**doc.get("profile", {}), **{k: v for k, v in profile.items() if v is not None}}
        if merged != doc.get("profile"):
            doc["profile"] = merged
            changed = True
    if plural_select is not None and plural_select != doc.get("plural_select"):
        doc["plural_select"] = plural_select
        changed = True
    if plural_categories is not None and plural_categories != doc.get("plural_categories"):
        doc["plural_categories"] = plural_categories
        changed = True
    return doc if changed else None


def choose_language(drive_root: pathlib.Path, body: Any, *,
                    audit: Optional[Callable[[Dict[str, Any]], None]] = None) -> Tuple[int, Dict[str, Any]]:
    """The ONE writer of the install's interface language: ``(status, payload)``.

    ``body["language"]`` is a BCP-47 tag, or a name/description the generator resolves into
    a tag and a profile (``language_needs_model`` 400 without a credentialed light model,
    ``language_resolve_failed`` 502). The tag is written through the locked owner-settings
    writer, the memory header is created or merged, ``ui_language_changed`` is broadcast and
    the generator hooks fire. Shared by ``POST /api/ui/i18n/language`` (the browser) and the
    Host Service's ``/ui/language`` (a skill relaying the owner, e.g. Telegram's ``/language``).
    ``SettingsDocumentBusy`` propagates: each caller answers it in its own typed shape."""
    raw = body.get("language") if isinstance(body, dict) else None
    if not isinstance(raw, str):
        return 400, {"ok": False, "saved": False, "code": "language_not_a_tag",
                     "error": "language must be a string: a BCP-47 tag such as ru or pt-BR, or a language name"}
    tag = normalize_language_tag(raw)
    profile = dict(body.get("profile") or {}) if isinstance(body.get("profile"), dict) else {}
    if tag is None:
        # A name or a description ("Quenya", "invent a language"): the generator turns it into
        # a tag and a profile with one light call; without a model it says so, never guesses.
        from ouroboros.ui_translation import LanguageResolveError, resolve_language_request

        try:
            resolved = resolve_language_request(raw, drive_root=drive_root)
        except LanguageResolveError as exc:
            return exc.status, {"ok": False, "saved": False, "code": exc.code, "error": str(exc), "language": raw[:120]}
        tag = resolved["tag"]
        profile = {**{k: v for k, v in resolved.items() if k != "tag"}, **profile}
    if isinstance(body.get("label"), str) and body["label"].strip():
        profile["label"] = body["label"].strip()
    plural_select = body.get("plural_select") if isinstance(body.get("plural_select"), dict) else None
    plural_categories = body.get("plural_categories") if isinstance(body.get("plural_categories"), list) else None
    previous = memory.current_language()

    def _set(current: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if str(current.get(SETTING_KEY) or LANGUAGE_NOT_CHOSEN) == tag:
            return None
        current[SETTING_KEY] = tag
        return current

    with settings_document_mutation():
        # This endpoint IS the author of the key, so an explicit English ("en") persists
        # as chosen rather than reading back as the not-chosen default.
        _owner_update_settings(_set, authored_keys=(SETTING_KEY,))
        # Same-lock projection into this process's environment, like the other owner
        # decisions: readers in this process see the tag before the next settings load.
        os.environ[SETTING_KEY] = tag
    memory_error = ""
    if not is_english(tag):
        try:
            memory.update_memory(
                drive_root, tag,
                lambda doc: _merge_header(doc, profile, plural_select, plural_categories),
                create=True, profile=profile or None, plural_select=plural_select,
                plural_categories=plural_categories)
        except memory.MemoryFormatError as exc:
            memory_error = str(exc)
        except TimeoutError as exc:
            # The setting is written; the header waits for the next writer. Said, not hidden.
            memory_error = f"the translation memory was busy: {exc}"
    if audit is not None:
        audit({"ui_language": tag, "previous_ui_language": previous})
    _broadcast({"type": "ui_language_changed", "language": tag})
    _notify("language_set", drive_root, tag)
    payload = memory_payload(drive_root, tag)
    if memory_error and not payload.get("memory_error"):
        payload["memory_error"] = memory_error
    return 200, {"ok": True, **payload}


def _language_post_sync(request: Request, body: Any) -> JSONResponse:
    status, payload = choose_language(
        request_drive_root(request), body,
        audit=lambda facts: _owner_audit(request, "ui_language", facts))
    if status >= 400:
        extra = {k: v for k, v in payload.items() if k not in ("ok", "saved", "error")}
        return unsaved_error(str(payload.get("error") or "language refused"), status, **extra)
    return JSONResponse(payload)


@owner_write_guard
async def api_ui_i18n_language_post(request: Request) -> JSONResponse:
    """Choose the install's interface language (owner decision; locked settings write)."""
    body = await request_json_or(request, None)
    if not isinstance(body, dict):
        return json_error("request body must be a JSON object", 400)
    from ouroboros.gateway.settings import _run_settings_writer

    return await _run_settings_writer(_language_post_sync, request, body)


async def api_ui_i18n_missing_post(request: Request) -> JSONResponse:
    body = await request_json_or(request, None)
    if not isinstance(body, dict):
        return json_error("request body must be a JSON object", 400)
    drive_root = request_drive_root(request)
    current = memory.current_language()
    tag = normalize_language_tag(body.get("language"))
    if not tag or tag != current:
        return json_error("language is not the install's current language", 409,
                          code="language_mismatch", language=current)
    items = body.get("items")
    if not isinstance(items, list):
        return json_error("items must be a list", 400)
    if is_english(tag):
        return JSONResponse({"accepted": 0, "dropped": len(items), "pending": 0})
    cleaned = [item for item in items[:MAX_MISSING_ITEMS] if isinstance(item, dict)]
    result = memory.record_missing(drive_root, tag, cleaned)
    result["dropped"] += len(items) - len(cleaned)
    if result["accepted"]:
        _notify("missing", drive_root, tag)
    return JSONResponse(result)


async def api_ui_i18n_import_post(request: Request) -> JSONResponse:
    body = await request_json_or(request, None)
    if not isinstance(body, dict):
        return json_error("request body must be a translation memory JSON object", 400)
    tag = normalize_language_tag(body.get("language"))
    if not tag or is_english(tag):
        return json_error("import needs a non-English language tag in `language`", 400, code="memory_invalid")
    if "schema" in body and body.get("schema") != memory.SCHEMA:
        return json_error(f"unsupported translation memory schema {body.get('schema')!r}; this install reads schema {memory.SCHEMA}",
                          400, code="memory_invalid")
    drive_root = request_drive_root(request)
    counts: Dict[str, int] = {}

    def _import(doc: Dict[str, Any]) -> Dict[str, Any]:
        counts.update(memory.apply_import(doc, body))
        return doc

    try:
        doc = memory.update_memory(drive_root, tag, _import, create=True,
                                   profile=body.get("profile") if isinstance(body.get("profile"), dict) else None)
    except memory.MemoryFormatError as exc:
        return json_error(f"the file is not a valid translation memory: {exc}", 400, code="memory_invalid")
    except TimeoutError as exc:
        return json_error(str(exc), 503, code="memory_busy")
    if tag == memory.current_language():
        _broadcast({"type": "ui_i18n_updated", "language": tag, "revision": doc["revision"]})
    _notify("imported", drive_root, tag)
    return JSONResponse({"ok": True, "language": tag, "revision": doc["revision"], "result": counts,
                         "stats": memory.stats(doc, source_hashes=_source_hashes(),
                                               pending=memory.pending_count(drive_root, tag))})


async def api_ui_i18n_export_get(request: Request) -> JSONResponse:
    drive_root = request_drive_root(request)
    requested = request.query_params.get("language")
    tag = normalize_language_tag(requested) if requested is not None else memory.current_language()
    if not tag or is_english(tag):
        return json_error("no translation memory exists for English", 404, code="no_memory")
    try:
        doc = memory.load_memory(drive_root, tag)
    except memory.MemoryFormatError as exc:
        return json_error(f"the stored memory is malformed: {exc}", 409, code="memory_invalid")
    if doc is None:
        return json_error(f"no translation memory exists for {tag}", 404, code="no_memory", language=tag)
    return JSONResponse(doc, headers={"Content-Disposition": f'attachment; filename="ouroboros-i18n-{tag}.json"'})


async def api_ui_i18n_regenerate_post(request: Request) -> JSONResponse:
    body = await request_json_or(request, {})
    if not isinstance(body, dict):
        return json_error("request body must be a JSON object", 400)
    drive_root = request_drive_root(request)
    requested = body.get("language")
    tag = normalize_language_tag(requested) if requested else memory.current_language()
    if not tag or is_english(tag):
        return json_error("nothing to regenerate for English", 400, code="no_memory")
    removed = {"count": 0}

    def _reset(doc: Dict[str, Any]) -> Dict[str, Any]:
        removed["count"] = memory.regenerate_reset(doc)
        return doc

    try:
        doc = memory.update_memory(drive_root, tag, _reset, create=True)
    except memory.MemoryFormatError as exc:
        return json_error(f"the stored memory is malformed: {exc}", 409, code="memory_invalid")
    except TimeoutError as exc:
        return json_error(str(exc), 503, code="memory_busy")
    _broadcast({"type": "ui_i18n_updated", "language": tag, "revision": doc["revision"]})
    _notify("regenerate", drive_root, tag)
    return JSONResponse({"ok": True, "language": tag, "revision": doc["revision"], "removed": removed["count"],
                         "stats": memory.stats(doc, source_hashes=_source_hashes(),
                                               pending=memory.pending_count(drive_root, tag))})


__all__ = [
    "register_language_hook", "memory_payload", "choose_language",
    "api_ui_i18n_get", "api_ui_i18n_language_post", "api_ui_i18n_missing_post",
    "api_ui_i18n_import_post", "api_ui_i18n_export_get", "api_ui_i18n_regenerate_post",
]
