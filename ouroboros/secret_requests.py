"""Integration-secret requests: ``showSecretHitl`` asks, the owner's protected form saves.

One request is one owner question (``owner_quiz``) whose block carries typed, metadata-only
facts beside the ordinary lifecycle, so publication, waiting, history, recovery and the Main
mirror stay the existing question machinery:

    "secret_request": {"plugin_name", "fields": [{"id", "setting_key"}]},
    "secret_save"?:   {"request_id", "state": writing|saved, "started_at", "saved_at"?,
                       "unconfirmed_request_ids"?},
    "secret_outcome"?: {"event": secret_filled|secret_replaced|secret_cancelled, "pluginName",
                        "fields", "at", "settings_keys", "replaced_fields"?,
                        "consumers"?|"consumers_unknown"?, "late"?, "interrupted_save"?}

A value travels exactly once: browser password control -> ``POST /api/decisions/secret`` ->
the existing owner Settings writer (``settings.json``). It never enters a tool argument or
result, a quiz comment, a mailbox frame, a history row, an event, a log line, an error or a
response; the model learns the event, the plugin and the field ids.

Which Settings key a field writes is the skill's reviewed declaration, never the request's:
``SkillManifest.secret_fields`` maps public field ids to keys the manifest already lists in
``env_from_settings`` (``secret_fields: {botToken: TELEGRAM_BOT_TOKEN}``). Such a key may name
an ordinary setting, so the host admits only an owner-defined custom secret key. The ask
records the mapping it was given; a save writes only while the installed skill still maps the
same ids to the same keys, so an old request never gains a target from a changed skill.

The save writer grants, enables and reloads nothing and contacts no provider; enabled consumers may apply
new values themselves. The outcome names their access (``key_consumers``), or an unreadable consumer list
(``consumers_unknown``). The form
stays usable after the asking task ends, until it is saved or explicitly cancelled; a late
save is recorded on the card and in history, never as the owner's words or new work.

First winner: every save and cancel of a request runs under the one in-process settings
document lock, and a save claims the block (``secret_save.state == writing``) before its write,
so a closure from any other path is refused while a write may be in flight. Under that lock a
``writing`` claim is an attempt whose result is unknown and is never reported as saved: its
request, and every earlier unknown attempt the block lists in ``unconfirmed_request_ids``, is
refused, and only a NEW owner request (a deliberate save again) takes it over.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

from ouroboros.utils import utc_now_iso

MAX_SECRET_VALUE_CHARS = 4096
EVENT_FILLED = "secret_filled"
EVENT_REPLACED = "secret_replaced"
EVENT_CANCELLED = "secret_cancelled"
SAVED_EVENTS = (EVENT_FILLED, EVENT_REPLACED)
SAVE_WRITING = "writing"
SAVE_SAVED = "saved"
CANCEL_REASON = "The owner cancelled the secret request; this request saved nothing."
CANCEL_REASON_INTERRUPTED = ("The owner cancelled the secret request; an earlier save attempt was never "
                             "confirmed, so whether its value reached Settings is unknown.")
_CONTROL_RE = re.compile(r"[\x00-\x1f\x7f]")


class SecretRequestError(ValueError):
    """A typed refusal whose message names fields and keys, never a value."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def declared_secret_fields(manifest: Any) -> Dict[str, str]:
    """The manifest's ``secret_fields`` as ``{field_id: settings_key}`` (empty when absent).

    The parser already proved the shape and that each key is in ``env_from_settings``; this
    re-checks both (a hand-built manifest is not parsed) and adds the host's classification:
    every target must be an owner-defined custom Settings secret, so provider, network and
    core keys are refused even when listed.
    """
    from ouroboros.contracts.skill_manifest import SkillManifestError, _secret_fields

    raw = getattr(manifest, "secret_fields", None) or {}
    try:
        fields = _secret_fields(raw, list(getattr(manifest, "env_from_settings", None) or []))
    except SkillManifestError as exc:
        raise SecretRequestError("SECRET_FIELDS_INVALID", str(exc)) from None
    from ouroboros.config import SETTINGS_DEFAULTS
    from ouroboros.contracts.plugin_api import FORBIDDEN_SKILL_SETTINGS
    from ouroboros.secret_masking import MASKED_SECRET_SETTING_KEYS, is_custom_secret_setting_key

    for field_id, key in fields.items():
        if (key in FORBIDDEN_SKILL_SETTINGS or key in MASKED_SECRET_SETTING_KEYS
                or not is_custom_secret_setting_key(key, known_setting_keys=SETTINGS_DEFAULTS)):
            raise SecretRequestError("SECRET_FIELDS_INVALID", f"secret field {field_id} targets {key}, which is not "
                                     "a custom Settings secret; provider, network and core keys stay in Settings.")
    return fields


def _repo_path() -> Optional[str]:
    from ouroboros.config import get_skills_repo_path

    return get_skills_repo_path() or None


def resolve_secret_request(drive_root: Any, plugin_name: Any, fields: Any) -> Dict[str, Any]:
    """The request's typed metadata for ``plugin_name``'s declared ``fields``, or a refusal."""
    from ouroboros.contracts.skill_manifest import MAX_SECRET_FIELDS
    from ouroboros.skill_loader import find_skill

    name = str(plugin_name or "").strip()
    if not name:
        raise SecretRequestError("SECRET_PLUGIN_REQUIRED", "pluginName must name an installed skill.")
    skill = find_skill(drive_root, name, repo_path=_repo_path())
    if skill is None or getattr(skill, "identity_collision", False):
        raise SecretRequestError("SECRET_PLUGIN_UNKNOWN", f"no single installed skill is named {name!r}.")
    if skill.load_error:
        raise SecretRequestError("SECRET_PLUGIN_BROKEN", f"skill {skill.name} has a manifest error; fix it first.")
    declared = declared_secret_fields(skill.manifest)
    if not declared:
        raise SecretRequestError("SECRET_FIELDS_UNDECLARED", f"skill {skill.name} declares no secret_fields in its "
                                 "manifest, so the host knows no protected field to ask for.")
    if not isinstance(fields, list) or not fields or len(fields) > MAX_SECRET_FIELDS:
        raise SecretRequestError("SECRET_FIELDS_INVALID", f"fields must list 1 to {MAX_SECRET_FIELDS} declared field ids.")
    requested: List[str] = []
    for raw in fields:
        field_id = str(raw or "").strip()
        if field_id not in declared:
            raise SecretRequestError("SECRET_FIELD_UNDECLARED", f"{skill.name} declares the secret fields "
                                     f"{', '.join(sorted(declared))}; {field_id or '(empty)'} is not one of them.")
        if field_id in requested:
            raise SecretRequestError("SECRET_FIELDS_INVALID", f"field {field_id} is listed twice.")
        requested.append(field_id)
    return {"plugin_name": skill.name,
            "fields": [{"id": field_id, "setting_key": declared[field_id]} for field_id in requested]}


def binding_refusal(drive_root: Any, request: Dict[str, Any]) -> str:
    """Why the asked mapping no longer holds (``""`` while it does): the skill is gone or broken,
    or a field now names another key. A skill update that keeps the mapping keeps the request."""
    try:
        current = resolve_secret_request(drive_root, request.get("plugin_name"), field_ids(request))
    except SecretRequestError as exc:
        return exc.code.lower()
    except Exception as exc:  # an unreadable skill tree is not a current mapping
        return f"skill_unreadable:{type(exc).__name__}"
    return "" if current["fields"] == request.get("fields") else "mapping_changed"


def secret_request_of(block: Any) -> Optional[Dict[str, Any]]:
    request = block.get("secret_request") if isinstance(block, dict) else None
    return request if isinstance(request, dict) and request.get("plugin_name") else None


def field_ids(request: Dict[str, Any]) -> List[str]:
    return [str(row.get("id") or "") for row in request.get("fields") or []]


def settings_keys(request: Dict[str, Any]) -> List[str]:
    return [str(row.get("setting_key") or "") for row in request.get("fields") or []]


def configured_fields(document: Dict[str, Any], request: Dict[str, Any]) -> Dict[str, bool]:
    """``{field_id: True}`` where the field's key holds a non-empty value in the settings document.
    A boolean only: no value, prefix or length ever leaves here."""
    return {str(row["id"]): bool(str(document.get(str(row["setting_key"])) or "").strip())
            for row in request.get("fields") or []}


def key_consumers(drive_root: Any, keys: List[str]) -> List[Dict[str, Any]]:
    """Installed skills that read any of ``keys`` (their ``env_from_settings``), with what their
    CURRENT grant covers: read-only facts for the form and the outcome, never a grant. A skills
    directory that cannot be listed raises (the callers record the readers as unknown): an empty
    list always means that nothing readable reads the key."""
    from ouroboros.skill_loader import discover_skills, grant_status_for_skill

    wanted = [str(key).upper() for key in keys]
    rows: List[Dict[str, Any]] = []
    for skill in discover_skills(drive_root, repo_path=_repo_path(), strict=True):
        if getattr(skill, "identity_collision", False) or skill.load_error:
            continue
        declared = {str(key or "").strip().upper() for key in skill.manifest.env_from_settings or []}
        reads = [key for key in wanted if key in declared]
        if not reads:
            continue
        row: Dict[str, Any] = {"name": skill.name, "enabled": bool(skill.enabled), "reads": reads}
        try:
            status = grant_status_for_skill(drive_root, skill)
        except Exception:
            status = None
        if not isinstance(status, dict):
            row["grant_unknown"] = True
        elif status.get("unsupported_for_skill_type"):
            row["unsupported"] = True  # an instruction skill receives no Settings key at all
        else:
            row["granted"] = [key for key in reads if key in set(status.get("granted_keys") or [])]
        rows.append(row)
    return sorted(rows, key=lambda item: item["name"])


def public_event(block: Dict[str, Any]) -> Dict[str, Any]:
    """The outcome's public event: its name, ``pluginName`` and the field ids, no value."""
    outcome = block.get("secret_outcome") if isinstance(block.get("secret_outcome"), dict) else {}
    request = secret_request_of(block) or {}
    return {"event": str(outcome.get("event") or ""), "pluginName": str(request.get("plugin_name") or ""),
            "fields": field_ids(request)}


def question_text(request: Dict[str, Any]) -> str:
    """The card's host-authored question."""
    return f"Secret needed for {request['plugin_name']}: {', '.join(field_ids(request))}."


def transport_hint(request: Dict[str, Any]) -> str:
    """What a chat transport shows instead of an answerable card: where to enter it, no link."""
    return (f"{question_text(request)} Enter it in the protected form in the Ouroboros app on the web or "
            "desktop; do not send it in this chat.")


def validate_values(request: Dict[str, Any], values: Any) -> Dict[str, str]:
    """``{settings_key: value}`` for exactly the asked fields, or a refusal naming only fields.

    A value Settings would read back as a masked placeholder (and blank on load) is refused
    rather than reported saved; so is anything a password control cannot carry.
    """
    from ouroboros.secret_masking import looks_masked_settings_secret

    asked = {str(row["id"]): str(row["setting_key"]) for row in request.get("fields") or []}
    if not isinstance(values, dict) or set(values) != set(asked):
        raise SecretRequestError("secret_fields_mismatch", "values must carry exactly the requested fields: "
                                 f"{', '.join(asked)}.")
    changes: Dict[str, str] = {}
    for field_id, key in asked.items():
        value = values[field_id]
        if not isinstance(value, str) or not value.strip():
            raise SecretRequestError("secret_value_empty", f"{field_id} is empty.")
        if len(value) > MAX_SECRET_VALUE_CHARS:
            raise SecretRequestError("secret_value_too_long", f"{field_id} is longer than {MAX_SECRET_VALUE_CHARS} characters.")
        if _CONTROL_RE.search(value):
            raise SecretRequestError("secret_value_invalid", f"{field_id} contains a line break or control character.")
        if looks_masked_settings_secret(key, value):
            raise SecretRequestError("secret_value_placeholder", f"{field_id} has the shape of a masked placeholder, "
                                     "which Settings would not keep; enter the real value.")
        changes[key] = value
    return changes


def validate_observed(request: Dict[str, Any], observed: Any) -> Dict[str, bool]:
    """Which asked keys the form showed as already set, ``{field_id: bool}`` for every field.

    The save writes only while Settings still says exactly this, so a form that showed an empty
    key never silently replaces a value saved since, and Replace is always an explicit choice."""
    ids = field_ids(request)
    if (not isinstance(observed, dict) or set(observed) != set(ids)
            or not all(isinstance(value, bool) for value in observed.values())):
        raise SecretRequestError("observed_required", "observed must state, for every requested field, whether "
                                 "the form showed its key as set (read status first).")
    return {field_id: observed[field_id] for field_id in ids}


def _projection():
    from ouroboros.owner_quiz import _KEEP, _delivery_debt, _mutate_projection

    return _KEEP, _delivery_debt, _mutate_projection


def claim_save(drive_root: Any, task_id: str, quiz_id: str, request_id: str) -> Dict[str, Any]:
    """Claim the block for one save (caller holds the settings document lock).

    ``duplicate`` replays a recorded success without writing again. A ``writing`` claim is an
    attempt whose outcome is unknown (the lock proves nobody here still holds it): the SAME
    request is refused (``secret_save_unconfirmed``) rather than repeated blind over a write that
    may have landed; a NEW request is the owner's explicit save-again and takes it over, adding the
    taken-over id to this block's ``unconfirmed_request_ids``, so no earlier unknown attempt is ever
    accepted again either. A release before the write restores the previous claim, so a request
    refused before it wrote stays retryable. An unanswered question stays saveable after its task
    ended (``expired_terminal``)."""
    keep, _debt, mutate_projection = _projection()
    outcome: Dict[str, Any] = {}

    def mutate(quizzes: Dict[str, Dict[str, Any]]) -> Any:
        block = quizzes.get(str(quiz_id))
        if not block:
            outcome.update(error="secret_request_not_found")
            return keep
        if secret_request_of(block) is None:
            outcome.update(error="not_a_secret_request")
            return keep
        state = str(block.get("state") or "open")
        if (state == "answered" and (block.get("secret_outcome") or {}).get("event") in SAVED_EVENTS
                and block.get("request_id") == request_id):
            outcome.update(ok=True, duplicate=True, block=dict(block))
            return keep
        if state not in ("open", "expired_terminal"):
            outcome.update(error="secret_request_settled", block=dict(block))
            return keep
        claim = dict(block.get("secret_save") or {})
        outcome["previous_claim"] = claim
        unconfirmed = [str(rid) for rid in claim.get("unconfirmed_request_ids") or []]
        if claim.get("state") == SAVE_WRITING:
            unconfirmed.append(str(claim.get("request_id") or ""))
        if request_id in unconfirmed:
            outcome.update(error="secret_save_unconfirmed", block=dict(block))
            return keep
        block["secret_save"] = {"request_id": request_id, "state": SAVE_WRITING, "started_at": utc_now_iso(),
                                **({"unconfirmed_request_ids": unconfirmed} if unconfirmed else {})}
        outcome.update(ok=True, block=dict(block))
        return True

    mutate_projection(drive_root, task_id, mutate)
    return outcome


def release_save(drive_root: Any, task_id: str, quiz_id: str, request_id: str, previous: Dict[str, Any]) -> None:
    """Nothing was written by ``request_id``: restore what the block said before its claim."""
    keep, _debt, mutate_projection = _projection()

    def mutate(quizzes: Dict[str, Dict[str, Any]]) -> Any:
        block = quizzes.get(str(quiz_id))
        claim = (block or {}).get("secret_save") or {}
        if not block or claim.get("request_id") != request_id or claim.get("state") != SAVE_WRITING:
            return keep
        if previous:
            block["secret_save"] = dict(previous)
        else:
            block.pop("secret_save", None)
        return True

    mutate_projection(drive_root, task_id, mutate)


def finalize_save(drive_root: Any, task_id: str, quiz_id: str, request_id: str,
                  outcome_facts: Dict[str, Any]) -> Dict[str, Any]:
    """The value is in Settings: record the host's typed completion as the block's answer.

    Never a comment and never an option: ``secret_outcome`` is the answer, and its delivery debt
    is the one an ordinary answer owes. ``outcome_facts`` carries ``replaced_fields`` (keys that
    held a value just before this write) and ``consumers`` (what readers can read after it); facts
    without a ``consumers`` list record ``consumers_unknown``, never an empty list of readers."""
    keep, delivery_debt, mutate_projection = _projection()
    result: Dict[str, Any] = {}

    def mutate(quizzes: Dict[str, Dict[str, Any]]) -> Any:
        block = quizzes.get(str(quiz_id))
        claim = (block or {}).get("secret_save") or {}
        if (not block or claim.get("request_id") != request_id or claim.get("state") != SAVE_WRITING
                or block.get("state") not in ("open", "expired_terminal")):
            result.update(error="secret_claim_lost", block=dict(block or {}))
            return keep
        stamp = utc_now_iso()
        request = secret_request_of(block) or {}
        late = block.get("state") == "expired_terminal"
        replaced = [field for field in field_ids(request) if field in set(outcome_facts.get("replaced_fields") or [])]
        consumers = outcome_facts.get("consumers")
        block.update(state="answered", answered_at=stamp, request_id=request_id, answer_source="web",
                     delivery=delivery_debt(task_id, quiz_id, "answer"),
                     secret_save={**claim, "state": SAVE_SAVED, "saved_at": stamp},
                     secret_outcome={"event": EVENT_REPLACED if replaced else EVENT_FILLED,
                                     "pluginName": request.get("plugin_name", ""), "fields": field_ids(request),
                                     "at": stamp, "settings_keys": settings_keys(request),
                                     **({"replaced_fields": replaced} if replaced else {}),
                                     **({"consumers": list(consumers)} if isinstance(consumers, list)
                                        else {"consumers_unknown": True}),
                                     **({"late": True} if late else {})},
                     **({"answered_after_terminal": True} if late else {}))
        result.update(ok=True, block=dict(block))
        return True

    mutate_projection(drive_root, task_id, mutate)
    return result


def cancel_outcome(block: Dict[str, Any]) -> Dict[str, Any]:
    """The typed closure facts of an owner cancel, including an interrupted earlier save."""
    request = secret_request_of(block) or {}
    claim = block.get("secret_save") if isinstance(block.get("secret_save"), dict) else {}
    return {"event": EVENT_CANCELLED, "pluginName": request.get("plugin_name", ""), "fields": field_ids(request),
            "at": utc_now_iso(), "settings_keys": settings_keys(request),
            **({"interrupted_save": True} if claim.get("state") == SAVE_WRITING else {})}


def _event_line(block: Dict[str, Any]) -> str:
    event = public_event(block)
    return f"{event['event']}: " + json.dumps({"pluginName": event["pluginName"], "fields": event["fields"]},
                                             ensure_ascii=False)


def _consumer_line(outcome: Dict[str, Any]) -> str:
    """What each skill that reads the key can read now, from the facts recorded after the save."""
    consumers = outcome.get("consumers")
    if outcome.get("consumers_unknown") or not isinstance(consumers, list):
        return ("Which installed skills read these keys could not be read; check each skill's grant and status "
                "before relying on it.")
    if not consumers:
        return "No installed skill reads these keys."
    parts = []
    for row in consumers:
        state = "enabled" if row.get("enabled") else "disabled"
        if row.get("grant_unknown"):
            grant = "its grant could not be read"
        elif row.get("unsupported"):
            grant = "this skill type receives no Settings keys"
        else:
            missing = [key for key in row.get("reads") or [] if key not in set(row.get("granted") or [])]
            grant = f"needs the owner's grant for {', '.join(missing)} in Skills" if missing else "its grant covers it"
            if missing and row.get("enabled"):
                grant += "; after saving, this enabled skill cannot run until the keys are granted"
        parts.append(f"{row.get('name')} ({state}; {grant})")
    return "Skills that read them: " + "; ".join(parts) + "."


def answer_frame(block: Dict[str, Any]) -> str:
    """The model's ``secret_filled``/``secret_replaced`` message: names and facts, never a value."""
    outcome = block.get("secret_outcome") or {}
    request = secret_request_of(block) or {}
    keys = ", ".join(settings_keys(request))
    replaced = outcome.get("event") == EVENT_REPLACED
    lines = [
        f"[Owner secret request] quiz {block.get('quiz_id')} — asked {block.get('asked_at')}, "
        f"saved {block.get('answered_at')}.",
        _event_line(block),
        (f"The owner {'replaced the existing value' if replaced else 'entered the value'} in the protected form "
         f"and the host saved it to Settings ({keys}). The value is not in this conversation; do not ask for it here."),
        ("Saving is not a connection: nothing was granted, enabled, reloaded or contacted. "
         + _consumer_line(outcome)),
    ]
    if replaced:
        lines.append("Enabled skills that read a replaced key may use the new value on their next call; some keep "
                     "the old one until the owner disables and re-enables them, or reconnects them.")
    if block.get("answered_after_terminal") or block.get("delivery_after_terminal"):
        lines.append("The task that asked had already finished; this save resumed nothing.")
    return "\n".join(lines)


def cancelled_frame(block: Dict[str, Any]) -> str:
    """The model's distinct cancellation message: what is still missing, and what is unknown."""
    request = secret_request_of(block) or {}
    names, keys = ", ".join(field_ids(request)), ", ".join(settings_keys(request))
    if (block.get("secret_outcome") or {}).get("interrupted_save"):
        saved = (f"The owner cancelled {request.get('plugin_name')}: {names}. The cancel itself saved "
                 "nothing, but an earlier save attempt was interrupted before it was confirmed, so whether "
                 f"its value reached Settings ({keys}) is unknown; do not assume it is there or absent.")
    else:
        saved = (f"The owner cancelled; this request saved nothing for {request.get('plugin_name')}: {names} "
                 f"({keys} unchanged by it).")
    return "\n".join([
        f"[Owner secret request cancelled] quiz {block.get('quiz_id')} — closed {block.get('closed_at')}.",
        _event_line(block), saved,
        "[Host note] The task can tell the owner what is still missing to continue and ask again only "
        "if they want to.",
    ])


__all__ = [
    "CANCEL_REASON", "CANCEL_REASON_INTERRUPTED", "EVENT_CANCELLED", "EVENT_FILLED", "EVENT_REPLACED",
    "MAX_SECRET_VALUE_CHARS", "SAVED_EVENTS", "SecretRequestError", "answer_frame", "binding_refusal",
    "cancel_outcome", "cancelled_frame", "claim_save", "configured_fields", "declared_secret_fields",
    "field_ids", "finalize_save", "key_consumers", "public_event", "question_text", "release_save",
    "resolve_secret_request", "secret_request_of", "settings_keys", "transport_hint", "validate_observed",
    "validate_values",
]
