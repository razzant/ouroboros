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
(``consumers_unknown``). The form's separate "Save and allow" adds ``allow`` to one save: the request's own
skill, its fields, and the files and known installation the form showed (``skill_facts``); a save writes only
while both are still current. After the lock one lifecycle job claims it (``running``: a started step never runs
twice) and grants it once exactly those files are admitted (``files_accepted``, never runtime authority alone),
else keeps the owner's consent until their admission (``skill_grant_follow``); its result, like turning the
skill on, is a fact of its own on ``secret_outcome.allow``:

    "allow"?: {"skill", "fields", "keys", "content_hash", "installation", "enable", "at",
               "state": requested|running|granted|deferred|refused|unknown, "reason"?, "granted_keys"?,
               "enabled"?, "enable_reason"?, "started_at"?, "settled_at"?, "resolved_at"?}

No other skill is granted anything; the readers that can already read a key stay named. The form
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
import logging
import re
import threading
from typing import Any, Dict, List, Optional, Tuple

from ouroboros.utils import utc_now_iso

log = logging.getLogger(__name__)

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
ALLOW_REQUESTED, ALLOW_RUNNING, ALLOW_GRANTED, ALLOW_DEFERRED, ALLOW_REFUSED, ALLOW_UNKNOWN = (
    "requested", "running", "granted", "deferred", "refused", "unknown")
_ALLOW_BODY = frozenset({"content_hash", "installation", "fields", "enable"})
_ALLOW_RESULT = frozenset({"state", "reason", "granted_keys", "enabled", "enable_reason"})
_REFUSAL_REASONS = {"revision_changed": "skill_changed", "not_found": "skill_unavailable", "acceptance": "not_admitted",
                    "unsupported_type": "unsupported", "not_requested": "not_requested"}
ALLOW_ORPHANED = {"state": ALLOW_UNKNOWN, "reason": "grant_unconfirmed"}  # a started step no live process holds
ALLOW_NOT_STARTED = {"state": ALLOW_REFUSED, "reason": "not_started"}  # its grant step never began, nothing granted
_running: set = set()  # questions whose allow step runs (or is reserved to run) in this process
_running_lock = threading.Lock()


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


def validate_allow(request: Dict[str, Any], allow: Any) -> Optional[Dict[str, Any]]:
    """The save's explicit allow, or ``None`` without one (a plain Save or Replace, an older form or client).

    It covers exactly the request's own skill and every asked field, for the files and the installation the form
    showed (``content_hash``, ``installation``); ``enable`` says whether the pressed button named turning it on."""
    if allow is None:
        return None
    ids = field_ids(request)
    allow = allow if isinstance(allow, dict) else {}
    fields, digest, installation = allow.get("fields"), allow.get("content_hash"), allow.get("installation")
    if (set(allow) != _ALLOW_BODY or not isinstance(allow["enable"], bool)
            or not all(isinstance(value, str) and value.strip() and len(value) <= 128 for value in (digest, installation))
            or not isinstance(fields, list) or len(fields) != len(ids) or sorted(map(str, fields)) != sorted(ids)):
        raise SecretRequestError("allow_invalid", "allow must be {content_hash, installation, fields, enable}: the "
                                 f"files and installation the form showed, exactly the requested fields "
                                 f"({', '.join(ids)}) and whether it turns the skill on.")
    return {"content_hash": digest.strip(), "installation": installation.strip(), "fields": ids,
            "enable": allow["enable"]}


def _shown_files(drive_root: Any, plugin_name: Any) -> Any:
    from ouroboros.skill_loader import find_skill

    skill = find_skill(drive_root, plugin_name, repo_path=_repo_path())
    return None if skill is None or getattr(skill, "identity_collision", False) or skill.load_error else skill


def skill_facts(drive_root: Any, request: Dict[str, Any]) -> Dict[str, Any]:
    """What the form's allow needs about the request's own skill, never a value: its files (``content_hash``) and
    known installation (``installation``, ``""`` when none is known), whether these files are admitted
    (``skill_grant_follow.files_accepted``: a recorded acceptance or a passed review of exactly them, never runtime
    authority alone), whether its type receives Settings keys, which asked keys its grant covers or a consent for
    these files already waits for, and its saved switch (``skill_lifecycle_actions.owner_switch``)."""
    from ouroboros.skill_grant_follow import allow_installation, consent_of, files_accepted
    from ouroboros.skill_lifecycle_actions import owner_switch, update_pending
    from ouroboros.skill_loader import grant_status_for_skill

    skill = _shown_files(drive_root, request["plugin_name"])
    if skill is None:
        return {"name": str(request["plugin_name"]), "available": False}
    keys, installation, waiting = settings_keys(request), allow_installation(drive_root, skill), consent_of(drive_root, skill.name)
    granted = set(grant_status_for_skill(drive_root, skill).get("granted_keys") or [])
    waits = (not waiting.get("claimed_at") and waiting.get("digest") == skill.content_hash
             and waiting.get("installation") == installation)
    return {"name": skill.name, "available": True, "content_hash": skill.content_hash, "installation": installation,
            "supported": bool(skill.manifest.is_script() or skill.manifest.is_extension()),
            "admitted": files_accepted(skill.review, skill.content_hash) and not update_pending(drive_root, skill.name),
            "granted": [key for key in keys if key in granted],
            "waiting": [key for key in keys if waits and key in waiting["keys"]],
            "switch": owner_switch(drive_root, skill.name) or "unset"}


def allow_refusal(drive_root: Any, request: Dict[str, Any], allow: Optional[Dict[str, Any]]) -> str:
    """``""`` while the files and the installation the allow names are the skill's current ones; else why not."""
    if allow is None:
        return ""
    from ouroboros.skill_grant_follow import allow_installation

    try:
        skill = _shown_files(drive_root, request.get("plugin_name"))
        installation = allow_installation(drive_root, skill) if skill is not None else ""
    except Exception:  # an unreadable skill tree shows no current files
        return "skill_unreadable"
    if skill is None or skill.content_hash != allow["content_hash"]:
        return "skill_changed"
    return "" if installation and installation == allow["installation"] else "installation_changed"


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
    without a ``consumers`` list record ``consumers_unknown``, never an empty list of readers. An
    ``allow`` (``validate_allow``) is recorded ``requested``: its own step settles it after the lock."""
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
        consumers, allow = outcome_facts.get("consumers"), outcome_facts.get("allow")
        block.update(state="answered", answered_at=stamp, request_id=request_id, answer_source="web",
                     delivery=delivery_debt(task_id, quiz_id, "answer"),
                     secret_save={**claim, "state": SAVE_SAVED, "saved_at": stamp},
                     secret_outcome={"event": EVENT_REPLACED if replaced else EVENT_FILLED,
                                     "pluginName": request.get("plugin_name", ""), "fields": field_ids(request),
                                     "at": stamp, "settings_keys": settings_keys(request),
                                     **({"replaced_fields": replaced} if replaced else {}),
                                     **({"consumers": list(consumers)} if isinstance(consumers, list)
                                        else {"consumers_unknown": True}),
                                     **({"late": True} if late else {}),
                                     **({"allow": {"skill": request.get("plugin_name", ""), "fields": field_ids(request),
                                                   "keys": settings_keys(request),
                                                   "content_hash": allow["content_hash"],
                                                   "installation": allow["installation"], "enable": allow["enable"],
                                                   "state": ALLOW_REQUESTED, "at": stamp}} if allow else {})},
                     **({"answered_after_terminal": True} if late else {}))
        result.update(ok=True, block=dict(block))
        return True

    mutate_projection(drive_root, task_id, mutate)
    return result


def allow_result(result: Dict[str, Any], *, enable: bool, attempted: bool, switch: str,
                 enable_ready: bool = True) -> Dict[str, Any]:
    """A grant step's answer as the allow's separate facts: access (``state``, ``reason``, ``granted_keys``) and,
    only when the pressed button named turning on, the switch (``enabled``: true, false, or null when unknown;
    ``enable_reason`` whenever it is not on)."""
    if result.get("deferred"):
        facts: Dict[str, Any] = {"state": ALLOW_DEFERRED, "reason": "not_admitted"}
    elif result.get("ok"):
        facts = {"state": ALLOW_GRANTED, "granted_keys": [str(key) for key in result.get("granted_keys") or []]}
    else:
        code = str(result.get("code") or "")
        facts = {"state": ALLOW_REFUSED, "reason": result.get("reason") or _REFUSAL_REASONS.get(code, "refused")}
    if not enable:
        return facts
    if switch == "on":
        return {**facts, "enabled": True}
    if attempted and facts["state"] == ALLOW_GRANTED:
        on = result.get("enabled") if isinstance(result.get("enabled"), bool) else None
        code = str(result.get("setup_code") or "")
        return {**facts, "enabled": on, **({} if on else {
            "enable_reason": code or "error" if on is False else code if code == "switch_unreadable" else "unknown"})}
    if switch == "unknown":
        return {**facts, "enabled": None, "enable_reason": "switch_unreadable"}
    reason = ("owner_off" if switch == "off" else "deferred" if facts["state"] == ALLOW_DEFERRED
              else "not_allowed" if facts["state"] != ALLOW_GRANTED
              else "dependencies" if not enable_ready else "not_attempted")
    return {**facts, "enabled": False, "enable_reason": reason}


def send_card_state(block: Dict[str, Any], task_id: str) -> None:
    """Best-effort live update of the question's cards; the canonical block is the truth and history replays it."""
    try:
        from ouroboros.owner_quiz import question_delivery_projection
        from supervisor.message_bus import get_bridge

        get_bridge().send_quiz_state(
            str(block.get("quiz_id") or ""), task_id, str(block.get("state") or ""),
            secret_outcome=block.get("secret_outcome"), question_delivery=question_delivery_projection(block) or None,
            closed_reason=block.get("closed_reason"), closed_at=block.get("closed_at"),
            closed_actor=block.get("closed_actor"), closed_request_id=block.get("closed_request_id"),
            chat_id=int(block.get("publication_chat_id") or 0))
    except Exception:
        log.debug("secret request state broadcast failed for %s", block.get("quiz_id"), exc_info=True)


def _decision_block(drive_root: Any, decision: Dict[str, Any]) -> Tuple[str, str, Dict[str, Any]]:
    parts = str(decision.get("decision_id") or "").split(":", 2)
    if len(parts) != 3 or parts[0] != "quiz":
        return "", "", {}
    from ouroboros.owner_quiz import quiz_states

    return parts[1], parts[2], quiz_states(drive_root, parts[1]).get(parts[2]) or {}


def _allow_matches(allow: Dict[str, Any], request_id: Any, decision: Dict[str, Any], expect: Tuple[str, ...]) -> bool:
    return (allow.get("state") in expect and (not decision.get("request_id") or request_id == decision["request_id"])
            and all(allow.get(key) == value for key, value in (decision.get("binding") or {}).items()))


def _settled_allow(allow: Dict[str, Any], facts: Dict[str, Any], at: str, *, deferred: bool) -> Dict[str, Any]:
    kept = {key: value for key, value in allow.items() if key not in _ALLOW_RESULT}
    result = {key: value for key, value in facts.items() if key in _ALLOW_RESULT}
    if "granted_keys" in result:
        own = set(allow.get("keys") or [])
        result["granted_keys"] = [key for key in result["granted_keys"] if key in own]
    if not kept.get("enable"):
        result.pop("enabled", None)
        result.pop("enable_reason", None)
    elif "enabled" not in result and result.get("state") in (ALLOW_UNKNOWN, ALLOW_REFUSED):
        unknown = result["state"] == ALLOW_UNKNOWN
        result.update(enabled=None if unknown else False, enable_reason="unknown" if unknown else "not_allowed")
    return {**kept, **result, "settled_at": allow.get("settled_at") or at, **({"resolved_at": at} if deferred else {})}


def keep_allow_card(block: Dict[str, Any]) -> bool:
    """Only a live grant step or a wait for admission needs the full card."""
    allow = (block.get("secret_outcome") or {}).get("allow") or {}
    return block.get("state") == "answered" and allow.get("state") in (
        ALLOW_REQUESTED, ALLOW_RUNNING, ALLOW_DEFERRED)


def consent_result_pending(block: Any) -> bool:
    if not isinstance(block, dict) or block.get("state") != "answered":
        return False
    allow = (block.get("secret_outcome") or {}).get("allow") or {}
    return allow.get("state") in (ALLOW_RUNNING, ALLOW_DEFERRED, ALLOW_UNKNOWN)


def _consent_result_expected(facts: Dict[str, Any]) -> Tuple[str, ...]:
    # Positive evidence can resolve a previously unknown result; the reverse cannot happen.
    return ((ALLOW_RUNNING, ALLOW_DEFERRED) if facts.get("state") == ALLOW_UNKNOWN
            else (ALLOW_RUNNING, ALLOW_DEFERRED, ALLOW_UNKNOWN))


def project_consent_result(block: Dict[str, Any], task_id: str, settlements: Dict[str, Any]) -> Dict[str, Any]:
    """Read-only fallback when the known result could not yet reach its own card."""
    if not consent_result_pending(block):
        return block
    outcome = block["secret_outcome"]
    allow = outcome["allow"]
    rows = settlements.get(allow.get("skill"))
    if not isinstance(rows, dict):
        return block
    return _apply_consent_result(block, rows.get(f"quiz:{task_id}:{block.get('quiz_id')}"))


def _apply_consent_result(block: Dict[str, Any], entry: Any) -> Dict[str, Any]:
    if not isinstance(entry, dict) or not entry.get("request_id") or not entry.get("binding") or not entry.get("at"):
        return block
    outcome = block["secret_outcome"]
    allow = outcome["allow"]
    facts = entry.get("facts")
    if (not isinstance(facts, dict) or facts.get("state") not in (ALLOW_GRANTED, ALLOW_REFUSED, ALLOW_UNKNOWN)
            or not _allow_matches(allow, block.get("request_id"), entry, _consent_result_expected(facts))):
        return block
    readers = entry.get("consumers")
    view = {key: value for key, value in outcome.items() if key not in ("consumers", "consumers_unknown")}
    view.update({"consumers": readers} if isinstance(readers, list) else {"consumers_unknown": True})
    return {**block, "secret_outcome": {**view, "allow": _settled_allow(allow, facts, entry["at"], deferred=True)}}


_ALLOW_HISTORY = "secret_allow_result"
_SETTLED_ROWS = {"quiz_answer": ("answered", ("owner_quiz_answer",)), "quiz_closed": ("closed", ("owner_question_close", "model_question_close"))}


def _archived(row: Dict[str, Any]) -> Dict[str, Any]:
    """A question as its settled row recorded it, minus the delivery that row froze before its ACK: an answer keeps its row."""
    quiz = {key: value for key, value in row["quiz"].items() if key != "question_delivery"}
    return {**quiz, "delivery": {"answer_row": row} if row.get("type") == "quiz_answer" else {}}


def _answer_snapshot(root: Any, task_id: str, quiz_id: str, row: Any = None, *, chat_id: Any = -1) -> Dict[str, Any]:
    """An evicted secret request read back from its validated answer or closure row (``_archived``), for display."""
    from ouroboros.dialogue_store import rows_by_client_id

    candidates = [row] if row is not None else [item.row for kind in _SETTLED_ROWS for item in rows_by_client_id(
        root, f"{kind}:{task_id}:{quiz_id}", chat_id=chat_id)]
    for answer in candidates:
        kind, quiz = str(answer.get("type") or ""), answer.get("quiz") or {}
        state, sources = _SETTLED_ROWS.get(kind, ("", ()))
        if (state and answer.get("source") in sources and answer.get("direction") == "system"
                and answer.get("task_id") == task_id and answer.get("client_message_id") == f"{kind}:{task_id}:{quiz_id}"
                and quiz.get("quiz_id") == quiz_id and (chat_id == -1 or answer.get("chat_id") == chat_id)
                and quiz.get("publication_chat_id") == answer.get("chat_id")
                and quiz.get("state") == state and secret_request_of(quiz) is not None):
            return _archived(answer)
    return {}


def _history_allow(root: Any, task_id: str, block: Dict[str, Any]) -> Dict[str, Any]:
    from ouroboros.dialogue_store import rows_by_client_id

    if not consent_result_pending(block) or type(chat_id := block.get("publication_chat_id")) is not int:
        return block
    quiz_id = block["quiz_id"]
    for item in reversed(rows_by_client_id(root, f"{_ALLOW_HISTORY}:{task_id}:{quiz_id}", chat_id=chat_id)):
        row = item.row
        if (row.get("type") == _ALLOW_HISTORY and row.get("direction") == "system"
                and row.get("task_id") == task_id and row.get("quiz_id") == quiz_id):
            block = _apply_consent_result(block, row.get("result"))
    return block


def historical_secret_answer(root: Any, task_id: str, quiz_id: str) -> Dict[str, Any]:
    """Display only, with any result learned since. Never feed a history snapshot to a save, cancel or grant."""
    from ouroboros.skill_grant_follow import read_consent_outcomes

    block = _history_allow(root, task_id, _answer_snapshot(root, task_id, quiz_id))
    return project_consent_result(block, task_id, read_consent_outcomes(root)) if consent_result_pending(block) else block


def quiz_history_view(root: Any, quiz_source: Any) -> Tuple[Any, Any]:
    """``project(row, answer)``: the block a native card and Project pointer show, live, else its row's history without
    delivery (``_archived``); ``evidence(row)``: the frame its replayed row applies, delivery only as current."""
    from functools import cache

    from ouroboros.owner_quiz import SECRET_FIELDS, question_delivery_now
    from ouroboros.skill_grant_follow import read_consent_outcomes

    @cache
    def owed():
        return read_consent_outcomes(root)

    def live(row):
        return quiz_source(str(row.get("task_id") or ""))["quizzes"].get(str(row["quiz"].get("quiz_id") or ""))

    def project(row, answer):
        task_id, quiz = str(row.get("task_id") or ""), row["quiz"]
        current = live(row)
        if not quiz.get("secret_request"):  # an archived row tells its outcome, never today's delivery
            return current or (_archived(answer) if isinstance((answer or {}).get("quiz"), dict) else None)
        block = current or _answer_snapshot(root, task_id, str(quiz.get("quiz_id") or ""), answer, chat_id=row.get("chat_id", 1))
        block = _history_allow(root, task_id, block)
        return project_consent_result(block, task_id, owed()) if consent_result_pending(block) else block

    def evidence(row):
        late = row.get("type") == _ALLOW_HISTORY  # a result after the answer: the answered card as it is now
        block = (project(row, None if late else row) or {}) if late or row["quiz"].get("secret_request") else {}
        if late and (block.get("state") != "answered" or not block.get("secret_outcome")):
            return None
        quiz = {key: block[key] for key in ("quiz_id", "state", *SECRET_FIELDS) if key in block} if late else dict(row["quiz"])
        if block.get("secret_outcome"):
            quiz["secret_outcome"] = block["secret_outcome"]
        quiz = {key: value for key, value in quiz.items() if key not in ("delivery", "question_delivery")}
        return {**quiz, **question_delivery_now(live(row))}

    return project, evidence


def _record_late_allow(root: Any, task_id: str, block: Dict[str, Any], decision: Dict[str, Any],
                       facts: Dict[str, Any], at: str, consumers: Any) -> None:
    """A deferred result as one host row of the question's room, dated by itself (``at``): card evidence, never speech."""
    from ouroboros.contracts.chat_id_policy import HIDDEN_CHAT_ID, is_a2a_chat_id
    from ouroboros.dialogue_store import append

    block = _history_allow(root, task_id, block)
    if not consent_result_pending(block):
        return
    chat_id = block.get("publication_chat_id")
    if (type(chat_id) is not int or chat_id == HIDDEN_CHAT_ID or is_a2a_chat_id(chat_id)
            or block.get("publication_evidence") not in {"pre_exposure", "history"}):
        return
    result = {"request_id": decision["request_id"], "binding": decision["binding"], "at": at,
              "facts": {key: facts[key] for key in _ALLOW_RESULT if key in facts},
              **({"consumers": consumers} if isinstance(consumers, list) else {})}
    if _apply_consent_result(block, result) is not block:
        quiz_id = block["quiz_id"]
        append(root, "chat", {"type": _ALLOW_HISTORY, "direction": "system", "chat_id": chat_id,
                              "task_id": task_id, "quiz_id": quiz_id, "ts": at,
                              "quiz": {"quiz_id": quiz_id, "secret_request": dict(block["secret_request"])},
                              "client_message_id": f"{_ALLOW_HISTORY}:{task_id}:{quiz_id}", "result": result},
               idempotent=True)


def _change_allow(drive_root: Any, decision: Dict[str, Any], expect: Tuple[str, ...],
                  change: Any) -> Dict[str, Any]:
    """``change(allow)`` on the question's allow while it says one of ``expect`` (and names the same request);
    returns the changed block, else ``{}``."""
    task_id, quiz_id, _block = _decision_block(drive_root, decision)
    if not task_id:
        return {}
    keep, _debt, mutate_projection = _projection()
    out: Dict[str, Any] = {}

    def mutate(quizzes: Dict[str, Dict[str, Any]]) -> Any:
        block = quizzes.get(quiz_id) or {}
        outcome = block.get("secret_outcome") if isinstance(block.get("secret_outcome"), dict) else {}
        allow = outcome.get("allow") if isinstance(outcome.get("allow"), dict) else {}
        if not _allow_matches(allow, block.get("request_id"), decision, expect):
            return keep
        change(outcome, allow)
        out.update(block=dict(block))
        return True

    mutate_projection(drive_root, task_id, mutate)
    return out.get("block") or {}


def claim_allow(drive_root: Any, decision: Dict[str, Any]) -> bool:
    """Mark the question's allow started (``requested`` -> ``running``), once: the grant step runs only after it,
    inside the lifecycle lane, so a started step whose outcome a restart cut is never run again."""
    def start(outcome: Dict[str, Any], allow: Dict[str, Any]) -> None:
        outcome["allow"] = {**allow, "state": ALLOW_RUNNING, "started_at": utc_now_iso()}

    return bool(_change_allow(drive_root, decision, (ALLOW_REQUESTED,), start))


_UNSCANNED = object()  # no reader snapshot came with a result: its readers are read when it is recorded


def decision_consumers(drive_root: Any, decision: Dict[str, Any], memo: Optional[Dict[Any, Any]] = None, *, block: Any = None) -> Any:
    """Its asked keys' readers now (``None``: unknown), the snapshot kept with a settled result; ``memo``: one discovery."""
    if block is None:
        task_id, quiz_id, block = _decision_block(drive_root, decision)
        block = block or (_answer_snapshot(drive_root, task_id, quiz_id) if task_id else {})
    request = secret_request_of(block)
    if request is None:
        return None
    keys, memo = tuple(settings_keys(request)), {} if memo is None else memo
    if keys not in memo:
        try:
            memo[keys] = key_consumers(drive_root, list(keys))
        except Exception:
            memo[keys] = None
    return memo[keys]


def record_allow(drive_root: Any, decision: Dict[str, Any], facts: Dict[str, Any], *,
                 expect: Tuple[str, ...], at: str = "", consumers: Any = _UNSCANNED) -> Dict[str, Any]:
    """Record this decision's allow result with its readers (``consumers``, else read now) and send its cards."""
    task_id, quiz_id, current = _decision_block(drive_root, decision)
    if at and task_id and not current:
        current = _answer_snapshot(drive_root, task_id, quiz_id)
    if consumers is _UNSCANNED:
        consumers = decision_consumers(drive_root, decision, block=current)

    if at and current:
        _record_late_allow(drive_root, task_id, current, decision, facts, at, consumers)

    def settle(outcome: Dict[str, Any], allow: Dict[str, Any]) -> None:
        outcome["allow"] = _settled_allow(allow, facts, at or utc_now_iso(),
                                          deferred=bool(at) or allow.get("state") == ALLOW_DEFERRED)
        if isinstance(consumers, list):
            outcome["consumers"] = list(consumers)
            outcome.pop("consumers_unknown", None)
        else:
            outcome.pop("consumers", None)
            outcome["consumers_unknown"] = True

    block = _change_allow(drive_root, decision, expect, settle)
    if block:
        send_card_state(block, task_id)
    return block


def settle_decisions(drive_root: Any, decisions: List[Dict[str, Any]], facts: Dict[str, Any], *,
                     at: str, consumers: Any = _UNSCANNED) -> List[Dict[str, Any]]:
    """Return writes that failed; the receipt retains them. An already settled or absent card needs no write."""
    unrecorded: List[Dict[str, Any]] = []
    for decision in decisions:
        try:
            record_allow(drive_root, decision, facts, expect=_consent_result_expected(facts), at=at, consumers=consumers)
        except Exception:
            unrecorded.append(decision)
            log.warning("An allow outcome was not recorded on %s", decision.get("decision_id"), exc_info=True)
    return unrecorded


def reserve_allow(task_id: str, quiz_id: str) -> bool:
    """Hold the question's allow step for this process (the save reserves it before recording the allow)."""
    with _running_lock:
        if f"{task_id}:{quiz_id}" in _running:
            return False
        _running.add(f"{task_id}:{quiz_id}")
        return True


def release_allow(task_id: str, quiz_id: str) -> None:
    with _running_lock:
        _running.discard(f"{task_id}:{quiz_id}")


def run_allow(drive_root: Any, task_id: str, quiz_id: str, *, progress: Any = None,
              reserved: bool = False) -> Dict[str, Any]:
    """Settle the allow a recorded save asked for, after the settings lock: the save's own request (which
    ``reserved`` it) or the owner's repeat of it. One lifecycle job claims it and grants it, or keeps the consent for
    files not admitted yet (``skill_lifecycle_actions.grant_from_secret_form``). A step that started and that no
    process here holds any more is never run again: it is what its records say (``skill_grant_follow.orphan_facts``)."""
    from ouroboros.owner_quiz import quiz_states
    from ouroboros.skill_grant_follow import orphan_facts

    if not reserved and not reserve_allow(task_id, quiz_id):
        return quiz_states(drive_root, task_id).get(quiz_id) or {}
    try:
        block = quiz_states(drive_root, task_id).get(quiz_id) or {}
        allow = ((block.get("secret_outcome") or {}).get("allow") or {}) if block.get("state") == "answered" else {}
        decision = {"decision_id": f"quiz:{task_id}:{quiz_id}", "request_id": str(block.get("request_id") or "")}
        if allow.get("state") == ALLOW_RUNNING:
            return record_allow(drive_root, decision, orphan_facts(drive_root, decision, allow), expect=(ALLOW_RUNNING,)) or block
        if allow.get("state") != ALLOW_REQUESTED:
            return block
        try:
            from ouroboros.skill_lifecycle_actions import grant_from_secret_form

            grant_from_secret_form(drive_root, allow["skill"], content_hash=allow["content_hash"],
                                   installation=str(allow.get("installation") or ""), keys=list(allow["keys"]),
                                   enable=bool(allow["enable"]), decision=decision, progress=progress)
        except Exception as exc:  # no answer from the job: whatever it did stands, and it never runs after this
            log.warning("The allow of %s has no confirmed outcome: %s", quiz_id, type(exc).__name__)
            now = ((quiz_states(drive_root, task_id).get(quiz_id) or {}).get("secret_outcome") or {}).get("allow") or {}
            if now.get("state") in (ALLOW_REQUESTED, ALLOW_RUNNING):  # "not started" only where the lane proves it
                record_allow(drive_root, decision, orphan_facts(drive_root, decision, now, began=not getattr(
                    exc, "grant_step_not_started", False)), expect=(now["state"],))
        return quiz_states(drive_root, task_id).get(quiz_id) or block
    finally:
        release_allow(task_id, quiz_id)


def allow_pending(block: Dict[str, Any]) -> bool:
    """A recorded save whose allow has no result yet (not started, or started)."""
    outcome = block.get("secret_outcome") if isinstance(block.get("secret_outcome"), dict) else {}
    return block.get("state") == "answered" and (outcome.get("allow") or {}).get("state") in (ALLOW_REQUESTED,
                                                                                              ALLOW_RUNNING)


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
    """Reader snapshot recorded after the save (or its allow): each reader's own asked keys and those its grant let it
    read, names only; allow receipts use historical wording."""
    allow = outcome.get("allow") if isinstance(outcome.get("allow"), dict) else {}
    hist, waiting, consumers = bool(allow), (allow.get("skill") if allow.get("state") == ALLOW_DEFERRED else None), outcome.get("consumers")
    if outcome.get("consumers_unknown") or not isinstance(consumers, list):
        return (f"Which installed skills {'requested these keys at the time' if hist else 'request these keys'} "
                "could not be read; check each skill's grant and status before relying on it.")
    if not consumers:
        return f"No installed skill {'requested these keys at the time' if hist else 'requests these keys'}."
    parts, verb = [], "was" if hist else "is"
    for row in consumers:
        reads = [str(key) for key in row.get("reads") or []]
        state = ("was enabled" if row.get("enabled") else "was disabled") if hist else ("enabled" if row.get("enabled") else "disabled")
        if row.get("grant_unknown"):
            grant = "its grant could not be read"
        elif row.get("unsupported"):
            grant = "this skill type receives no Settings keys"
        else:  # grant facts only: whether the skill runs is its own readiness, which a saved card cannot keep current
            allowed = [key for key in reads if key in set(row.get("granted") or [])]
            missing = [key for key in reads if key not in allowed]
            grant = (f"{verb} allowed to read {', '.join(allowed)}" + (f", not {', '.join(missing)}" if missing else "")
                     if allowed else f"{verb} not allowed to read {', '.join(missing)}") + (
                "" if not missing else "; the owner's allow was recorded to wait for those files' admission"
                if row.get("name") == waiting else "; use Grant access in Skills")
        parts.append(f"{row.get('name')} ({state}; {'requested' if hist else 'requests'} {', '.join(reads)}; {grant})")
    return ("Skills that requested them at the time: " if hist else "Skills that request them: ") + "; ".join(parts) + "."


_ALLOW_REASON_WORDS = {
    "skill_changed": "its files changed after the form showed them", "skill_unavailable": "the skill could not be loaded",
    "installation_changed": "it was not the installation the form showed",
    "installation_ended": "the installation it was for ended", "not_admitted": "its files were not admitted yet",
    "unsupported": "this skill type receives no Settings keys", "not_requested": "its manifest no longer requested these keys",
    "consent_unrecorded": "the consent could not be recorded", "update_rolled_back": "the update was rolled back",
    "not_started": "the grant step did not start", "grant_unconfirmed": "the grant step gave no confirmed answer",
}
_ENABLE_REASON_WORDS = {
    "owner_off": "it was turned off, and that choice stands", "switch_unreadable": "its saved switch could not be read",
    "not_allowed": "access was not granted", "dependencies": "its dependencies were not ready",
    "acceptance": "its files were not admitted", "grants": "it still lacked other keys or permissions",
    "conflict": "another enabled skill used the same tool names", "load_error": "it could not start",
    "dependencies_failed": "its dependencies could not be installed", "managed_by_deployment": "the organization managed it",
}


def _allow_lines(outcome: Dict[str, Any]) -> List[str]:
    """The allow's own facts for the model: what held at the time (access, then the switch when named)."""
    allow = outcome.get("allow") if isinstance(outcome.get("allow"), dict) else None
    if not allow:
        return []
    skill, keys, state = allow.get("skill"), ", ".join(allow.get("keys") or []), allow.get("state")
    named = (f"With the same press the owner allowed {skill} to read {keys}"
             + (" and asked to turn it on" if allow.get("enable") else ""))
    if state == ALLOW_GRANTED:
        line = f"{named}: granted for the files the form showed{' once they were admitted' if allow.get('resolved_at') else ''}."
    elif state == ALLOW_DEFERRED:
        line = (f"{named}. Those files were not admitted yet, so nothing was granted at the time: consent was recorded "
                "to grant it once exactly those files passed a review or were explicitly accepted; other files, another "
                "installation or a reinstall would get nothing.")
    elif state == ALLOW_REFUSED:
        line = f"{named}, but nothing was granted: {_ALLOW_REASON_WORDS.get(str(allow.get('reason')), 'the host refused it')}."
    else:
        line = f"{named}; whether it was granted is unknown. Check the skill's status before relying on it."
    lines = [line]
    if allow.get("enable"):
        reason = str(allow.get("enable_reason") or "")
        lines.append(f"{skill} was on at the time." if allow.get("enabled") is True
                     else (f"{skill} was not turned on at the time; it was recorded to turn on with that grant "
                           "unless it was turned off first.") if reason == "deferred"
                     else (f"Whether {skill} was on at the time is unknown"
                           + (f": {_ENABLE_REASON_WORDS[reason]}." if reason in _ENABLE_REASON_WORDS else "."))
                     if allow.get("enabled") is None
                     else f"{skill} was not turned on: {_ENABLE_REASON_WORDS.get(reason, 'turning it on failed')}.")
    return [*lines, "No other skill was granted anything."]


def answer_frame(block: Dict[str, Any]) -> str:
    """The model's ``secret_filled``/``secret_replaced`` message: names and facts, never a value."""
    outcome = block.get("secret_outcome") or {}
    request = secret_request_of(block) or {}
    keys = ", ".join(settings_keys(request))
    replaced = outcome.get("event") == EVENT_REPLACED
    allow_lines = _allow_lines(outcome)
    lines = [
        f"[Owner secret request] quiz {block.get('quiz_id')} — asked {block.get('asked_at')}, "
        f"saved {block.get('answered_at')}.",
        _event_line(block),
        (f"The owner {'replaced the existing value' if replaced else 'entered the value'} in the protected form "
         f"and the host saved it to Settings ({keys}). The value is not in this conversation; do not ask for it here."),
        *allow_lines,
        (("" if allow_lines else "Saving is not a connection: nothing was granted, enabled, reloaded or contacted. ")
         + _consumer_line(outcome)),
    ]
    if replaced:
        lines.append("Enabled skills granted a replaced key may use the new value on their next call; some keep "
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
    "ALLOW_DEFERRED", "ALLOW_GRANTED", "ALLOW_NOT_STARTED", "ALLOW_ORPHANED", "ALLOW_REFUSED", "ALLOW_REQUESTED",
    "ALLOW_RUNNING", "ALLOW_UNKNOWN", "CANCEL_REASON", "CANCEL_REASON_INTERRUPTED", "EVENT_CANCELLED", "EVENT_FILLED",
    "EVENT_REPLACED", "MAX_SECRET_VALUE_CHARS", "SAVED_EVENTS", "SecretRequestError", "allow_pending",
    "allow_refusal", "allow_result", "answer_frame", "binding_refusal", "cancel_outcome",
    "cancelled_frame", "claim_allow", "claim_save", "configured_fields", "decision_consumers", "declared_secret_fields", "field_ids",
    "finalize_save", "key_consumers", "public_event", "question_text", "record_allow",
    "release_allow", "release_save", "reserve_allow", "resolve_secret_request", "run_allow", "secret_request_of",
    "send_card_state", "settings_keys", "settle_decisions", "skill_facts", "transport_hint", "validate_allow",
    "validate_observed", "validate_values",
]
