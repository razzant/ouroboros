"""The route of running work: live route settings and the discriminated route choice.

Two facts the loop lacked (R1 §6.0): whether a task's route follows its role or is the task's own
choice, and that a role's binding is a reference, not a value frozen when the turn opened. This
leaf answers both; it holds no model call and no checkpoint store of its own.

Live view. ``ROUTE_SETTING_KEYS`` (the role model slots, their local flags, accounts, windows,
processing preferences, the OpenRouter routing preset and every provider credential key) are read
from the settings document as it is NOW, never from the task's entry snapshot: a saved Main reaches
work that follows Main on its next call, and a cleared or disabled key stops every task's next
send. The read has ``skill_settings()`` semantics (the full effective loader over
``skill_environ()``: an operator's own variable still stands in for a key the document leaves
unset; this process tree's own mirrored values never do, so a key the owner cleared cannot come
back from a reused worker's environment), memoised behind the document's stat and the operator
variables. A document that exists but cannot be read makes credential keys unavailable; nothing
answers from stale env. A process with no saved document (an operator-run one) has nothing to
resurrect and keeps reading its environment as every other key. Every other setting keeps the
task's entry snapshot (context mode, effort, review settings, the subagent catalog, access).

Choice. ``following(role)``: the acting route is the role's current binding, re-read at every new
logical call; a changed binding moves the route through the existing switch handover (an acting
route that differs for another reason, a card's reprepare, is left alone); ``explicit(source)``: the task's own route
(a configured row, a Presence turn, a task argument, ``switch_model``, a wait-card choice, an
adopted fallback) stays until the mind returns to its primary. The choice rides the continuation
``route`` block; a checkpoint written without one is classified from what it did carry
(``classify_legacy``: a disclosed migration policy, never reconstructed intent).
"""

from __future__ import annotations

import copy
import json
import os
import threading
from typing import Any, Mapping, Optional, Tuple

from ouroboros import provider_models
from ouroboros.model_slots import (
    MODEL_ACCOUNTS_KEY, MODEL_CONTEXT_WINDOWS_KEY, MODEL_PROCESSING_PREFERENCES_KEY, MODEL_ROLE_SETTINGS,
    PROCESSING_PREFERENCE_KEY,
)
from ouroboros.settings_defaults import SETTINGS_DEFAULTS, settings_env_keys
from ouroboros.settings_integrity import PROJECTED_PROPOSALS_ENV, PROJECTED_SETTINGS_ENV, runtime_setting

#: Why an acting route is the task's own: the record names the chooser, never a model string.
ROUTE_SOURCES = ("task_argument", "configured_row", "switch_model", "wait_card", "fallback", "presence")
ROLE_LABELS = {"main": "Main", "light": "Light", "safety": "Supervisor", "vision": "Vision",
               "consciousness": "Consciousness", "deep_review": "Deep self-review", "websearch": "Web search"}

ROUTE_CREDENTIAL_SETTING_KEYS: frozenset = frozenset(provider_models.ROUTE_CREDENTIAL_SETTING_KEYS)  # the kind table's
ROUTE_SETTING_KEYS: frozenset = frozenset({
    *MODEL_ROLE_SETTINGS.values(), *(key for key in SETTINGS_DEFAULTS if key.startswith("USE_LOCAL_")),
    MODEL_ACCOUNTS_KEY, MODEL_CONTEXT_WINDOWS_KEY, MODEL_PROCESSING_PREFERENCES_KEY, PROCESSING_PREFERENCE_KEY,
    "OUROBOROS_OR_PROVIDER", *ROUTE_CREDENTIAL_SETTING_KEYS,
})
_EXPORTED_ROUTE_KEYS = frozenset(settings_env_keys()) & ROUTE_SETTING_KEYS
# The operator variables a read lets stand in, and the projection records that tell them from our echo.
_SIGNATURE_KEYS = (*sorted(ROUTE_SETTING_KEYS), PROJECTED_PROPOSALS_ENV, PROJECTED_SETTINGS_ENV)

_VIEW_LOCK = threading.Lock()
_VIEW: Optional[tuple] = None  # (document stamp, operator-variable signature, effective document, env-shaped projection)


# --- the live view ------------------------------------------------------------------------------------

def _document_stamp() -> tuple:
    from ouroboros import config

    path = config.SETTINGS_PATH
    try:
        found = path.stat()
    except OSError:
        return (str(path), None)
    return (str(path), found.st_dev, found.st_ino, found.st_mtime_ns, found.st_size)


def _read_source() -> Mapping:
    """The environment a read lets stand in: the task's own, else this process's."""
    from ouroboros.settings_integrity import _TASK_SETTINGS

    snapshot = _TASK_SETTINGS.get()
    return os.environ if snapshot is None else snapshot.environ


def _project(effective: dict) -> dict:
    """The env-shaped spelling of the route keys, as ``apply_settings_to_env`` writes them (a blank Fallbacks
    stays present, role options canonical, structured values JSON); a key never exported to an environment
    (the connection rows) is spelled the same way for ``runtime_setting`` readers."""
    from ouroboros import config

    projected: dict = {}
    config.apply_settings_to_env(effective, environ=projected)
    for key in ROUTE_SETTING_KEYS - _EXPORTED_ROUTE_KEYS:
        value = effective.get(key)
        if value not in (None, ""):
            projected[key] = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return {key: projected[key] for key in ROUTE_SETTING_KEYS if key in projected}


_ABSENT: Tuple[dict, dict] = ({}, {})  # no saved document: the route keys read as every other key


def route_settings_view() -> Optional[Tuple[dict, dict]]:
    """``(effective document, env-shaped projection)`` of the route keys as the document reads now.

    ``_ABSENT`` (nothing served) while no document is saved: an operator-run process (a bench harness, an
    env-configured container) has nothing to resurrect, and its environment stays the configuration the
    ordinary readers serve. ``None`` when a document exists but could not be read (the verified reader
    refused it, or it does not decode): the failure is never memoised, so the next read tries again. The
    read takes no lock and persists no compatibility migration (writers rename atomically); integrity-pinned
    isolated runs keep the verified reader the loader already uses."""
    global _VIEW
    from ouroboros import config
    from ouroboros.server_runtime import apply_runtime_provider_defaults
    from ouroboros.settings_integrity import read_settings_json_verified, skill_settings

    source = _read_source()
    stamp, signature = _document_stamp(), tuple(source.get(key) for key in _SIGNATURE_KEYS)
    if stamp[1] is None:
        return _ABSENT
    view = _VIEW
    if view is not None and view[0] == stamp and view[1] == signature:
        return view[2], view[3]
    with _VIEW_LOCK:
        view = _VIEW
        if view is not None and view[0] == stamp and view[1] == signature:
            return view[2], view[3]
        try:
            if not isinstance(read_settings_json_verified(config.SETTINGS_PATH), dict):
                return None
            effective, _changed, _keys = apply_runtime_provider_defaults(
                skill_settings(lambda: config.load_settings_lock_held(_settings_lock_held=False)))
        except Exception:
            return None
        _VIEW = (stamp, signature, effective, _project(effective))
        return _VIEW[2], _VIEW[3]


def live_setting(key: str, default: Any = None) -> Tuple[bool, Any]:
    """``(served, value)`` of a route key for ``runtime_setting``: the live env-shaped value, ``default`` when
    the document leaves it unset or an unreadable document makes a credential key unavailable. ``served`` is
    False without a saved document, and for a non-credential key of an unreadable one: the caller keeps its
    entry view."""
    view = route_settings_view()
    if view is _ABSENT:
        return False, None
    if view is None:
        return (True, default) if key in ROUTE_CREDENTIAL_SETTING_KEYS else (False, None)
    return True, view[1].get(key, default)


def live_environ_overlay(env: dict) -> None:
    """The exported route keys of a child environment (``runtime_environ``), live."""
    view = route_settings_view()
    if view is _ABSENT:
        return
    for key in _EXPORTED_ROUTE_KEYS:
        if view is None:
            if key in ROUTE_CREDENTIAL_SETTING_KEYS:
                env.pop(key, None)
        elif key in view[1]:
            env[key] = view[1][key]
        else:
            env.pop(key, None)


def live_settings_overlay(settings: dict) -> None:
    """The route keys of a settings mapping (``runtime_settings``), live; values are private copies."""
    view = route_settings_view()
    if view is _ABSENT:
        return
    for key in ROUTE_SETTING_KEYS:
        if view is None:
            if key in ROUTE_CREDENTIAL_SETTING_KEYS:
                settings.pop(key, None)
        elif key in view[0]:
            settings[key] = copy.deepcopy(view[0][key])
        else:
            settings.pop(key, None)


# --- the choice ----------------------------------------------------------------------------------------

def following(role: str) -> dict:
    return {"kind": "following", "role": str(role or "main")}


def explicit(source: str) -> dict:
    return {"kind": "explicit", "source": str(source or "task_argument")}


def follows(choice: Any) -> str:
    """The role a choice follows live; ``""`` for an explicit choice or none."""
    return str(choice.get("role") or "main") if isinstance(choice, dict) and choice.get("kind") == "following" else ""


def model_role_slot_override(task_metadata: Mapping) -> Optional[Tuple[str, bool]]:
    """The ``(model, use_local)`` pinned by a task's ``metadata.model_role``; ``None`` = Main.

    A consciousness wake-up carries ``model_role="consciousness"`` (owner decision
    В25=B): when the consciousness model slot is set, the turn runs on that model
    with that slot's local flag; an empty slot means Main — the same model, the
    same prefix, the same prompt cache. Roles without a single-model slot (Main
    itself, the fallback chain) and unknown roles resolve to Main here; their
    effort and account binding still follow the role (``_initial_effort_for``,
    ``task_model_binding``).
    """
    metadata = task_metadata if isinstance(task_metadata, Mapping) else {}
    role = str(metadata.get("model_role") or "").strip().lower()
    key = MODEL_ROLE_SETTINGS.get(role)
    if not key or role in ("main", "fallback"):
        return None
    model = str(runtime_setting(key, "") or "").strip()
    raw_local = str(runtime_setting(f"USE_LOCAL_{role.upper()}", "") or "").strip().lower()
    use_local = raw_local in ("true", "1")
    if not model:
        # An empty model slot means Main's model — but the role's own local flag, when the
        # owner set it and it differs from Main's, still decides the route (В25=B honors
        # every slot; with equal flags nothing changes and the prefix stays Main's).
        main_local = str(runtime_setting("USE_LOCAL_MAIN", "") or "").strip().lower() in ("true", "1")
        main_model = str(runtime_setting("OUROBOROS_MODEL", "") or "").strip()
        if not raw_local or use_local == main_local or not main_model:
            return None
        return main_model, use_local
    return model, use_local


def following_route(role: str, llm: Any = None) -> Tuple[str, bool]:
    """The binding a following route uses NOW: the role's own slot (an empty slot is Main with the slot's
    own local flag honoured), else Main and its local flag; ``llm`` is the loop's client when it has one."""
    slot = model_role_slot_override({"model_role": role})
    if slot is not None:
        return slot
    from ouroboros.model_slots import _main_model

    model = str(llm.default_model() if llm is not None else _main_model())
    return model, str(runtime_setting("USE_LOCAL_MAIN", "") or "").strip().lower() in ("true", "1")


def _metadata(ctx: Any) -> dict:
    metadata = getattr(ctx, "task_metadata", None)
    return metadata if isinstance(metadata, dict) else {}


def _cognition(metadata: Mapping) -> Tuple[dict, str]:
    """A child's frozen parent cognition and its configured row kind."""
    cognitive = metadata.get("parent_cognitive_route")
    actor = metadata.get("configured_subagent")
    kind = str(((actor.get("route") if isinstance(actor, dict) else None) or {}).get("kind") or "")
    return (cognitive if isinstance(cognitive, dict) else {}), kind


def opening_role(ctx: Any) -> str:
    """The role a turn opened on when it carried no model of its own: a nanny of a following parent follows
    that parent's role; a role-named task (a consciousness wake) its slot; everything else Main."""
    metadata = _metadata(ctx)
    cognitive, kind = _cognition(metadata)
    if kind == "agent_session" and cognitive.get("follows_role"):
        return str(cognitive["follows_role"])
    primary = getattr(ctx, "primary_route", None)
    role = str(metadata.get("model_role") or (primary.get("role") if isinstance(primary, dict) else "") or "")
    role = role.strip().lower()
    return role if role in MODEL_ROLE_SETTINGS and role not in ("main", "fallback") else "main"


def opening_choice(ctx: Any) -> dict:
    """How this turn chose its primary: the model the task carries (explicit), else its role (following)."""
    if str(getattr(ctx, "task_model_override", "") or "").strip():
        return explicit(str(getattr(ctx, "task_route_source", "") or "") or "task_argument")
    return following(opening_role(ctx))


def open_route(ctx: Any, llm: Any) -> Tuple[str, bool, dict]:
    """The route a turn opens on and its choice; a locality the task carries without a model still applies."""
    override = str(getattr(ctx, "task_model_override", "") or "").strip()
    local_override = getattr(ctx, "task_use_local_override", None)
    ctx._route_binding = None  # the binding a following route last resolved to (``round_route``)
    if override:
        return override, bool(local_override), opening_choice(ctx)
    role = opening_role(ctx)
    model, use_local = ctx._route_binding = following_route(role, llm)
    return model, bool(local_override) if local_override is not None else use_local, following(role)


def _dispatch_source(metadata: Mapping) -> str:
    cognitive, kind = _cognition(metadata)
    by_parent = {"delegator": "configured_row", "parent_fallback": "fallback", "parent_switch": "switch_model"}
    if kind == "agent_session" and cognitive.get("chosen_by") in by_parent:
        return by_parent[str(cognitive["chosen_by"])]
    return "configured_row" if kind == "api_model" else "task_argument"


def bind_task_route(ctx: Any, task_metadata: Mapping, *, presence: bool) -> None:
    """At dispatch, the model a task carries binds its route: a configured row, a Presence turn, a lane child,
    a nanny of a parent that chose its own route. A nanny of a following parent and a role-named task carry
    none and open on that role's current binding instead (``open_route``). Idempotent: an amended resolution
    rebinds."""
    ctx.task_model_override = ctx.task_use_local_override = ctx.task_route_source = None
    delegated = str(task_metadata.get("delegation_role") or "").lower() == "subagent"
    cognitive, kind = _cognition(task_metadata)
    if not (delegated or presence) or (delegated and kind == "agent_session" and cognitive.get("follows_role")):
        return
    model = str(task_metadata.get("model") or "").strip()
    if model:
        ctx.task_model_override = model
        ctx.task_route_source = "presence" if presence else _dispatch_source(task_metadata)
    if "use_local_model" in task_metadata:
        ctx.task_use_local_override = bool(task_metadata.get("use_local_model"))


def round_route(ctx: Any, llm: Any, acting: Tuple[str, bool]) -> Optional[Tuple[str, bool]]:
    """At a new logical call: the binding a following route moves to, or ``None``.

    A move is a changed binding (a Settings save), never an acting route that differs for another
    reason (a card's reprepare, the task's own locality); the first round after a restore compares to
    the acting route, which a following route ran on. A following primary is re-read too, so the primary
    stays a reference to its role (``switch_model(primary="return")`` returns to the current one) and the
    configured-route walk tries the current Main. A turn without a recorded choice keeps its route."""
    choice = getattr(ctx, "route_choice", None)
    if not isinstance(choice, dict):
        return None
    opening = opening_choice(ctx)
    primary = getattr(ctx, "primary_route", None)
    if follows(opening) and isinstance(primary, dict) and primary.get("model"):
        primary["model"], primary["use_local"] = following_route(follows(opening), llm)
    role = follows(choice)
    if not role:
        return None
    acting = (str(acting[0] or ""), bool(acting[1]))
    bound = tuple(getattr(ctx, "_route_binding", None) or acting)
    current = ctx._route_binding = following_route(role, llm)
    return None if current in (acting, bound) else current


def note_settings_change(ctx: Any, previous: str, current: str, role: str) -> None:
    """One route-facts notice to the mind: its following route moved with Settings."""
    label = ROLE_LABELS.get(role, role)
    ctx._route_facts_pending = (
        f"[ROUTE FACTS] {label} changed in Settings: {previous} → {current}. This task follows {label}, so this "
        f"call and the next ones run on {current}; the transcript carries over. A route you choose with "
        'switch_model stays yours until switch_model(primary="return").')


def after_fallback(ctx: Any, is_primary: bool) -> dict:
    """The choice after a configured route answered: the primary rejoins its own choice, a chain entry is explicit."""
    return opening_choice(ctx) if is_primary else explicit("fallback")


def restored_choice(ctx: Any, state: Mapping) -> dict:
    """The choice a continuation carries, or the legacy classification of one that carries none."""
    saved = (state.get("route") or {}).get("route_choice")
    if isinstance(saved, dict) and saved.get("kind") in ("following", "explicit"):
        return dict(saved)
    return classify_legacy(ctx, state)


def classify_legacy(ctx: Any, state: Mapping) -> dict:
    """A checkpoint without a choice (D5): explicit when any explicit carrier it holds says so — the task's
    own model, a wait-card override for the primary's role, a fallback fit role, a pending ``switch_model``
    override, or an acting binding other than the primary's; otherwise following its role. A checkpoint with
    no primary and no carrier follows Main, so the oldest resumes do not keep an old route forever."""
    route = state.get("route") if isinstance(state.get("route"), dict) else {}
    primary = route.get("primary_route") if isinstance(route.get("primary_route"), dict) else {}
    overrides = getattr(getattr(ctx, "model_wait_context", None), "overrides", None) or {}
    if str(getattr(ctx, "task_model_override", "") or "").strip():
        return opening_choice(ctx)
    if primary.get("role") and overrides.get(str(primary["role"])):
        return explicit("wait_card")
    if str(state.get("context_model_role") or "").startswith("fallback:"):
        return explicit("fallback")
    if route.get("active_model_override"):
        return explicit("switch_model")
    acting = (str(route.get("active_model") or ""), bool(route.get("active_use_local")))
    if primary.get("model") and acting != (str(primary["model"]), bool(primary.get("use_local"))):
        return explicit("switch_model")
    return following(opening_role(ctx))
