"""The owner's Remote switch: where Main and new agents run (``OUROBOROS_REMOTE_EXECUTION``).

One saved preference, ``{"on": bool, "engine": "<connection>"}``, stored as its canonical JSON text (empty: Off,
nothing remembered; ``remote_execution_text``) and written by a room's bar (``web/modules/remote_switch.js``) through the
ordinary settings save. It is a route key (``route_choice.ROUTE_SETTING_KEYS``): every reader takes the document as it
is now, so a press reaches running work at its next call and the helpers started after it, a parent's that began before
it included. Nothing here rewrites Main, a role, the Agents roster or a reviewer row: the preference is applied where a
route is resolved, and only there.

While it is on (``placement_engine``):

- Main, as the work that follows it runs it (``effective_main``): a subscription route of this computer's engine
  (``claudexor::<source>=<model>``) runs as the same source and model on that engine (``claudexor@<engine>::…``).
  ``route_choice.following_route`` and ``configured_references`` resolve through it, so a turn that follows Main (a room
  on Main, a role whose own slot is empty) moves at its next logical call through the existing handover and its route
  notice, and ``/api/state``'s ``main_route`` names where Main runs now; the call already in flight finishes where it was
  sent. An API key, the local model, a Main naming an engine of its own or a route this cannot split stay as configured
  (``main_scope`` names which).
- A fresh helper (``place_helper``, at ``schedule_subagent`` and ``delegate_start``): a row naming no engine (a session
  row, or an API row on a subscription route of this computer's engine) starts on that engine and keeps it: a retry
  replays its snapshot, a continuation follows its predecessor's recorded engine and request (``place_delegated``),
  and its stop and collection follow its recorded binding. Reviewers, the Safety supervisor and rows naming an engine keep theirs.

An account pin and an acknowledged context window are an engine's own (``model_slots.model_role_option`` with
``model``): a placed route runs under that engine's Auto account and the model's own window, and the saved pin stays
saved. A placed route is an ordinary choice naming an engine: switched off, removed or unreachable it is refused
(``connection_unavailable`` and the engine's own refusals), never moved to another engine or back here; the owner's
Fallback models apply as for any route, and session shapes a remote engine cannot run (``REMOTE_SESSION_LIMITS``) are
refused there, typed. A press moves Main-following work at its next call and places the helpers started after it;
helpers already running keep their engine, and nothing is stopped, disconnected or rewritten. No host is contacted here.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, Mapping, Optional, Tuple

log = logging.getLogger(__name__)

REMOTE_EXECUTION_KEY = "OUROBOROS_REMOTE_EXECUTION"
_FIELDS = frozenset({"on", "engine"})
# What a remote engine refuses of a new session (``remote_engines.RemoteEngineGateway``,
# ``subagent_work_order.in_place_scratch``), said wherever the switch places helpers.
REMOTE_SESSION_LIMITS = ("plan turns and other thread sessions", "workspace='in_place'",
                         "a folder that is not a Git project")


def normalize_remote_execution(raw: Any) -> Tuple[dict, str]:
    """``(value, error)`` of a saved switch: ``{"on", "engine"}`` (``{}``: Off, nothing remembered), or the first problem
    as one owner-facing sentence (nothing saved). Shape only: ``remote_execution_save_error`` judges the engine."""
    from ouroboros.remote_engines import _RESERVED_NAMES, CONNECTION_NAME

    key = REMOTE_EXECUTION_KEY
    if raw is None or raw == "":
        return {}, ""
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except ValueError:
            return {}, f'{key} must be a JSON object such as {{"on": true, "engine": "build-box"}}.'
    if not isinstance(raw, dict):
        return {}, f'{key} must be an object {{"on", "engine"}}.'
    unknown = sorted(str(name) for name in set(raw) - _FIELDS)
    if unknown:
        return {}, f"{key}: unknown field(s) {', '.join(unknown)}."
    on, engine = raw.get("on", False), raw.get("engine", "")
    if not isinstance(on, bool):
        return {}, f"{key}: on must be true or false."
    if not isinstance(engine, str) or (engine and (not CONNECTION_NAME.fullmatch(engine) or engine in _RESERVED_NAMES)):
        return {}, f"{key}: engine must be the name of a saved remote engine."
    if on and not engine:
        return {}, f"{key}: on needs the remote engine new work runs on."
    return ({"on": on, "engine": engine} if on or engine else {}), ""


def remote_execution_text(raw: Any) -> str:
    """The stored spelling, written by the settings save (``gateway/settings.py``, which refuses an invalid value by
    name): a valid value's canonical JSON, "" for Off with nothing remembered. Any other text is kept as it is, and
    every reader takes it as Off (``remote_execution``)."""
    value, error = normalize_remote_execution(raw)
    if error:
        return raw if isinstance(raw, str) else json.dumps(raw, ensure_ascii=False, sort_keys=True, default=str)
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) if value else ""


def remote_execution(settings: Optional[Mapping[str, Any]] = None) -> dict:
    """The saved switch of ``settings``, else of the document as it is now; an unreadable value reads as Off."""
    if settings is None:
        from ouroboros.settings_integrity import runtime_setting

        raw = runtime_setting(REMOTE_EXECUTION_KEY, None)
    else:
        raw = settings.get(REMOTE_EXECUTION_KEY)
    value, error = normalize_remote_execution(raw)
    if error:
        log.warning("Remote switch unreadable, read as Off: %s", error)
    return value


def placement_engine(settings: Optional[Mapping[str, Any]] = None) -> str:
    """The engine new eligible work is placed on now, "" while Off. A target since removed or switched off is still
    named: the work placed there is refused, never moved."""
    value = remote_execution(settings)
    return str(value.get("engine") or "") if value.get("on") else ""


def remote_settings_error(body: Dict[str, Any], stored: Mapping[str, Any]) -> Tuple[str, str]:
    """``(message, code)`` refusing a settings save's remote keys, ``("", "")`` when it may go: the engine rows' shape,
    the switch's shape (made canonical in ``body`` when valid) and, when this save changes the switch, that an On
    switch names a saved engine taking new work among the rows this save writes, else the saved ones (``stored``). A
    save that re-sends the saved switch unchanged is not judged again."""
    from ouroboros.remote_engines import ENGINE_CONNECTIONS_KEY, normalize_engine_connections

    if ENGINE_CONNECTIONS_KEY in body and (error := normalize_engine_connections(body[ENGINE_CONNECTIONS_KEY])[1]):
        return error, "engine_connections_invalid"
    if REMOTE_EXECUTION_KEY not in body:
        return "", ""
    if error := normalize_remote_execution(body[REMOTE_EXECUTION_KEY])[1]:
        return error, "remote_execution_invalid"
    body[REMOTE_EXECUTION_KEY] = text = remote_execution_text(body[REMOTE_EXECUTION_KEY])
    if text == remote_execution_text(stored.get(REMOTE_EXECUTION_KEY)):
        return "", ""
    rows = body[ENGINE_CONNECTIONS_KEY] if ENGINE_CONNECTIONS_KEY in body else stored.get(ENGINE_CONNECTIONS_KEY)
    error = remote_execution_save_error({REMOTE_EXECUTION_KEY: text, ENGINE_CONNECTIONS_KEY: rows})
    return (error, "remote_execution_invalid") if error else ("", "")


def remote_execution_save_error(settings: Mapping[str, Any]) -> str:
    """Why ``settings`` (the document a save would write) cannot keep its switch, "" when it can: On names a saved
    remote engine that takes new work. A later Remove or Turn off of that engine is not refused for it: the work the
    switch places there is then refused until the owner chooses again."""
    from ouroboros.remote_engines import ENGINE_CONNECTIONS_KEY, normalize_engine_connections

    value, error = normalize_remote_execution(settings.get(REMOTE_EXECUTION_KEY))
    if error or not value.get("on"):
        return error
    rows, rows_error = normalize_engine_connections(settings.get(ENGINE_CONNECTIONS_KEY))
    row = None if rows_error else next((item for item in rows if item.get("name") == value["engine"]), None)
    if row is None:
        return (f"No remote engine named {value['engine']!r} is saved in Settings › Accounts and keys › Remote "
                "engines, so Remote cannot be turned on for it; nothing was saved.")
    if row.get("enabled", True) is False:
        return (f"The remote engine {str(row.get('label') or '').strip() or row['name']} is switched off in Settings "
                "› Accounts and keys › Remote engines, so it takes no new work; turn it on there first. Nothing was "
                "saved.")
    return ""


# --- where Main runs ------------------------------------------------------------------------------------------------

def on_engine(model: str, engine: str) -> str:
    """``model`` on ``engine``: a subscription route of this computer's engine as the same source and model there; ""
    for any other route (an API key, the local lane, a route naming an engine, one the grammar cannot split)."""
    from ouroboros.provider_models import claudexor_route, parse_claudexor_model, parse_model_route

    route = parse_model_route(model)
    if not engine or route.kind != "claudexor" or not route.valid or route.connection:
        return ""
    try:
        source, native = parse_claudexor_model(model)
    except ValueError:
        return ""
    return claudexor_route(engine, source, native)


def effective_main(model: str, use_local: bool, *, settings: Optional[Mapping[str, Any]] = None) -> Tuple[str, bool]:
    """Main's configured ``(model, use_local)`` as the work that follows Main runs it now."""
    placed = "" if use_local else on_engine(str(model or "").strip(), placement_engine(settings))
    return (placed, False) if placed else (model, bool(use_local))


def main_scope(model: str, use_local: bool) -> str:
    """Whether the switch moves Main: ``follows`` (a subscription route of this computer's engine: it runs on the
    target while On), else why not: ``local`` (the local model), ``api`` (an API key), ``engine`` (Main names an engine
    of its own) or ``unreadable``."""
    from ouroboros.provider_models import parse_claudexor_model, parse_model_route

    text = str(model or "").strip()
    route = parse_model_route(text)
    if use_local or route.kind == "local":
        return "local"
    if route.kind != "claudexor":
        return "api"
    if route.valid and route.connection:
        return "engine"
    try:
        parse_claudexor_model(text)
    except ValueError:
        return "unreadable"
    return "follows" if route.valid else "unreadable"


def state_view() -> Dict[str, Any]:
    """``/api/state``'s ``remote_execution``, from the readers dispatch uses and no host: the saved preference (``on``,
    ``engine``: the remembered target, "" for none), the saved ``engines`` (``name``, ``label``, ``enabled``) and
    ``main``: Main as configured, where it runs now (``effective``, the state's ``main_route``) and ``scope``
    (``main_scope``)."""
    from ouroboros.model_slots import _main_model
    from ouroboros.remote_engines import engine_connection_rows
    from ouroboros.settings_integrity import runtime_setting

    value = remote_execution()
    engines = [{"name": str(row["name"]), "label": str(row.get("label") or "").strip() or str(row["name"]),
                "enabled": row.get("enabled", True) is not False} for row in engine_connection_rows()]
    model = _main_model()
    local = str(runtime_setting("USE_LOCAL_MAIN", "") or "").strip().lower() in ("true", "1")
    effective, effective_local = effective_main(model, local)
    return {"on": bool(value.get("on")), "engine": str(value.get("engine") or ""), "engines": engines,
            "main": {"model": model, "use_local": local, "scope": main_scope(model, local),
                     "effective": {"model": effective, "use_local": effective_local}}}


# --- where a new helper runs ----------------------------------------------------------------------------------------

def place_helper(snapshot: Any, *, engine: Optional[str] = None, account: str = "", by: str = "remote_switch") -> Any:
    """The actor snapshot a delegated start freezes: ``snapshot`` placed on ``engine`` (the switch's now when None;
    "" places nothing) when its row names no engine. A session row gets ``route.engine``; an API row on a subscription
    route of this computer's engine gets that route on the engine. The row's account pin is this computer's engine's
    own, so the placed route asks for ``account`` there (Auto, unless a continuation keeps its predecessor's request);
    ``placement`` records who placed it, the engine and the row's pin left behind. Any other snapshot (a row naming an
    engine, an API key, the local model) is returned as it is."""
    engine = placement_engine() if engine is None else engine
    route = snapshot.get("route") if isinstance(snapshot, Mapping) else None
    if not engine or not isinstance(route, Mapping) or route.get("engine"):
        return snapshot
    kind, target = str(route.get("kind") or ""), str(route.get("target_id") or "")
    if kind == "agent_session":
        placed = {**route, "engine": engine}
    elif kind == "api_model" and (moved := on_engine(target, engine)):
        placed = {**route, "target_id": moved}
    else:
        return snapshot
    pin = str(route.get("credential_profile_id") or "")
    if "credential_profile_id" in placed or account:
        placed["credential_profile_id"] = str(account or "")
    return {**snapshot, "route": placed,
            "placement": {"by": by, "engine": engine, **({"row_account": pin} if pin else {})}}


def placed_on(route: Any, settings: Optional[Mapping[str, Any]] = None) -> str:
    """The remote engine a fresh start of a saved row's ``route`` (a ``RouteSpec``) runs on by the switch under
    ``settings`` (else the document now), "" when it runs where the row says."""
    engine = placement_engine(settings)
    snapshot = {"route": {"kind": route.kind, "target_id": route.target_id, "engine": route.engine}}
    return engine if engine and "placement" in place_helper(snapshot, engine=engine) else ""


def catalog_facts(engine: str) -> Dict[str, Any]:
    """The Available-subagents fact beside rows the switch places (``runs_on``): where, under which account, and
    what that engine refuses of a session (``REMOTE_SESSION_LIMITS``), so such a start is chosen knowingly."""
    return {"remote_switch": {"engine": engine, "account": "that engine's own Auto account",
                              "refused_there": list(REMOTE_SESSION_LIMITS)}}


def place_delegated(ctx: Any, snapshot: Any, continue_from: Any = None) -> Any:
    """The snapshot a delegated start (``schedule_subagent``, ``delegate_start``) freezes. A fresh start is placed where
    the switch says now (``place_helper``). A continuation is never placed by the switch: it runs where its predecessor
    ran (``execution_domain.start_engine``), and an account is its engine's own. So a row whose engine (this computer's,
    or the one it names) is the predecessor's keeps its own route and pin as before, another row or pin on that engine
    included; any other row asks on the predecessor's engine (its declared workspace too) for what the predecessor
    asked (``RunCustody.profile_id``: Auto when the switch placed it), never for another engine's pin. An unread
    predecessor places nothing: its start refuses on that same read."""
    token = str(continue_from or "").strip()
    if not token:
        return place_helper(snapshot)
    route = snapshot.get("route") if isinstance(snapshot, Mapping) else None
    try:
        from ouroboros.delegate_custody import custody_root
        from ouroboros.delegate_custody_current import runs as current_runs
        from ouroboros.execution_domain import is_remote_ref, recorded_ref

        prior = current_runs(custody_root(ctx), "run_id=?", (token,)).get(token)
        ref = recorded_ref(prior) if prior is not None else {}
    except Exception:
        return snapshot
    engine = str(ref.get("connection") or "") if is_remote_ref(ref) else ""
    if prior is None or not isinstance(route, Mapping) or str(route.get("engine") or "") == engine:
        return snapshot
    placed = {key: value for key, value in route.items() if key not in ("engine", "workspace")}
    placed.update({"credential_profile_id": str(getattr(prior, "profile_id", "") or "")},
                  **({"engine": engine} if engine else {}),
                  **({"workspace": str(ref["workspace"])} if engine and ref.get("workspace") else {}))
    pin, row_engine = str(route.get("credential_profile_id") or ""), str(route.get("engine") or "")
    return {**snapshot, "route": placed, "placement": {"by": "predecessor", "engine": engine,
                                                       **({"row_account": pin} if pin else {}),
                                                       **({"row_engine": row_engine} if row_engine else {})}}


__all__ = [
    "REMOTE_EXECUTION_KEY", "REMOTE_SESSION_LIMITS", "catalog_facts", "effective_main", "main_scope",
    "normalize_remote_execution", "on_engine", "place_delegated", "place_helper", "placed_on", "placement_engine",
    "remote_execution",
    "remote_execution_save_error", "remote_execution_text", "remote_settings_error", "state_view",
]
