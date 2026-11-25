"""``POST /api/decisions/secret`` — the protected form of an integration-secret request.

The browser's ``saveSecretToken`` sends ``{request_id, decision_id, values, observed, allow?}`` here and
nowhere else; ``action: "status"`` reads what the form needs (whether each asked key is set, which
skills read it and what their grants cover, and the request's own skill: ``skill``), ``action: "cancel"``
closes the request. ``decision_id`` is the request's ordinary question identity ``quiz:{task_id}:{quiz_id}``.
A save with ``allow`` (the form's separate "Save and allow") also writes only while the skill's files and
installation are the ones the form showed (``secret_allow_stale``); its grant step runs after the lock and the reply waits
for it at most ``JOB_ANSWER_WAIT_SEC``; after that an early reply comes only once the job is bound (``allow_job``
names it; its save receipt was taken before that, and the question's state brings the result), and the asker
learns the save with the allow's result.

A value is held in this request's memory only: validated against the request's declared fields,
written by the owner Settings writer (``_owner_update_settings`` with ``fields_on_saved_document``)
under the settings document lock, and dropped. That lock's one strict read is both checked (the
skill's mapping, and each asked key still set or unset exactly as the form showed, ``observed``) and
merged, so a stale empty form never replaces a value saved since and Replace is explicit; a document
that exists but cannot be read refuses (``settings_unreadable``) with its bytes untouched.
Every response, refusal, log line, audit row, history row and mailbox frame carries field ids,
Settings key names and lifecycle facts only. ``saved`` is a field of every save and cancel reply and
of every refusal (a ``status`` read carries the lifecycle and form facts only): a refusal says
``saved: false``; a write that landed but could not be verified or recorded says ``saved: true`` with
``secret_save_unverified``/``secret_not_retained``/``secret_completion_unrecorded``; an attempt of
unknown outcome says ``saved: null`` (``secret_save_unconfirmed``) and, like every earlier unknown
attempt on the same request, is never repeated under its own id — the owner reads status and decides
to save again with a new one.

The form stays usable after the asking task ends. A living asker (reading, paused, held or queued)
receives the outcome through the ordinary question mailbox; with none, the history row is the
record and nothing is forwarded as an owner turn or started as new work. ``POST /api/decisions``
refuses an ordinary answer to such a question (``secret_request_form_only``) and routes its
closure here.
"""

from __future__ import annotations

import logging
import pathlib
from types import SimpleNamespace
from typing import Any, Dict, Tuple

from starlette.requests import Request
from starlette.responses import JSONResponse

from ouroboros import secret_requests as sr
from ouroboros.gateway._helpers import request_drive_root, request_json_or, run_sync_to_completion

log = logging.getLogger(__name__)

_REQUEST_ID_MAX = 128
_FIELDS = {"request_id", "decision_id", "action", "values", "observed", "allow"}
_NO_STORE = {"Cache-Control": "no-store"}
_KEY_STATE_CHANGED = "secret_key_state_changed"
_ALLOW_STALE = "secret_allow_stale"


def _refusal(status: int, code: str, message: str, decision_id: str, block: Any = None,
             *, saved: Any = False, root: Any = None) -> Tuple[int, Dict[str, Any]]:
    payload: Dict[str, Any] = {"ok": False, "saved": saved, "error": message, "reason_code": code,
                               "decision_id": decision_id}
    if isinstance(block, dict) and block:
        payload.update(_lifecycle(block, root))
    return status, payload


def _lifecycle(block: Dict[str, Any], root: Any = None) -> Dict[str, Any]:
    """The request's current facts, so a losing tab settles on the winner instead of retrying.
    With ``root``, a request still taking a save also carries the form's key and reader facts."""
    from ouroboros.owner_quiz import CLOSURE_FIELDS, question_delivery_projection

    view: Dict[str, Any] = {"state": str(block.get("state") or "")}
    outcome = block.get("secret_outcome") if isinstance(block.get("secret_outcome"), dict) else {}
    if outcome.get("event") in (*sr.SAVED_EVENTS, sr.EVENT_CANCELLED):
        event = sr.public_event(block)
        view[event["event"]] = {"pluginName": event["pluginName"], "fields": event["fields"]}
        view["secret_outcome"] = dict(outcome)
    claim = block.get("secret_save") if isinstance(block.get("secret_save"), dict) else {}
    if claim.get("state") == sr.SAVE_WRITING:
        view["save_unconfirmed"] = True  # an attempt nobody here holds: its outcome is unknown
    view.update({key: block[key] for key in (*CLOSURE_FIELDS, "answered_after_terminal") if key in block})
    if delivery := question_delivery_projection(block):
        view["question_delivery"] = delivery
    request = sr.secret_request_of(block)
    if root is not None and request is not None and view["state"] in ("open", "expired_terminal"):
        view.update(_form_facts(root, request))
    return view


def _form_facts(root: Any, request: Dict[str, Any]) -> Dict[str, Any]:
    """Whether each asked key is set now (a boolean, never a value) and who reads it."""
    from ouroboros.gateway.owner_settings import _owner_settings_document

    facts: Dict[str, Any] = {}
    try:  # strict: a present document that cannot be read is unknown, never "not set"
        configured = sr.configured_fields(_owner_settings_document(strict=True), request)
        facts["keys"] = [{"id": row["id"], "setting_key": row["setting_key"], "configured": configured[row["id"]]}
                         for row in request.get("fields") or []]
    except Exception as exc:
        log.warning("Secret request key state unreadable: %s", type(exc).__name__)
        facts["keys_unknown"] = True
    try:
        facts["consumers"] = sr.key_consumers(root, sr.settings_keys(request))
    except Exception as exc:
        log.warning("Secret request readers unreadable: %s", type(exc).__name__)
        facts["consumers_unknown"] = True
    try:  # the request's own skill, for the form's allow: unknown never offers it
        facts["skill"] = sr.skill_facts(root, request)
    except Exception as exc:
        log.warning("Secret request skill unreadable: %s", type(exc).__name__)
        facts["skill"] = {"name": str(request.get("plugin_name") or ""), "available": False}
    return facts


def _receipt(decision_id: str, block: Dict[str, Any], *, duplicate: bool, root: Any = None) -> Dict[str, Any]:
    """``saved`` is true for a recorded save, null when an interrupted earlier attempt's write is
    unknown (a cancel after it saved nothing itself, yet cannot say nothing is stored), else false."""
    outcome = block.get("secret_outcome") if isinstance(block.get("secret_outcome"), dict) else {}
    saved = True if outcome.get("event") in sr.SAVED_EVENTS else None if outcome.get("interrupted_save") else False
    return {"ok": True, "decision_id": decision_id, "duplicate": duplicate, "saved": saved,
            **_lifecycle(block, root)}


def _parse(body: Any) -> Tuple[str, str, str, str, str]:
    if not isinstance(body, dict):
        raise sr.SecretRequestError("request_invalid", "request body must be a JSON object.")
    if set(body) - _FIELDS:
        raise sr.SecretRequestError("unexpected_fields", "a secret decision accepts only "
                                    "{request_id, decision_id, action?, values?, observed?, allow?}.")
    action = body.get("action", "save")
    if action not in ("save", "cancel", "status"):
        raise sr.SecretRequestError("action_invalid", "action must be save, cancel or status.")
    request_id = body.get("request_id")
    if action != "status" and (not isinstance(request_id, str) or not request_id.strip()
                               or len(request_id) > _REQUEST_ID_MAX):
        raise sr.SecretRequestError("request_id_required", "request_id is required "
                                    "(a stable client-generated id, reused only for the same values).")
    decision_id = str(body.get("decision_id") or "").strip()
    parts = decision_id.split(":", 2)
    if len(parts) != 3 or parts[0] != "quiz" or not parts[1] or not parts[2]:
        raise sr.SecretRequestError("malformed_decision_id", "decision_id must be quiz:{task_id}:{quiz_id}.")
    from ouroboros.task_results import validate_task_id

    try:
        task_id = validate_task_id(parts[1])
    except ValueError:
        raise sr.SecretRequestError("malformed_decision_id", "decision_id names an invalid task id.") from None
    if action != "save" and ("values" in body or "observed" in body or "allow" in body):
        raise sr.SecretRequestError(f"{action}_payload_invalid", f"a {action} carries no values.")
    return str(request_id or "").strip(), decision_id, task_id, parts[2], action


def _send_state(block: Dict[str, Any], task_id: str) -> None:
    """Best-effort live update; the canonical block is the truth and history replays it."""
    sr.send_card_state(block, task_id)


def deliver_secret_outcome(drive_root: pathlib.Path, task_id: str, quiz_id: str, *,
                           recovery: bool = False) -> Dict[str, Any]:
    """Deliver a recorded save like an answer: history row, then the living asker's mailbox.

    A reading, paused, held or queued asker (or its admitted Continue) holds it until it reads;
    none means the history row is the record, so a finished task is not resumed and the outcome
    is never forwarded as an owner turn."""
    from ouroboros.gateway.task_decision import _late_answer_destination, _mail_holder, _record_quiz_answer_history
    from ouroboros.owner_quiz import (
        claim_quiz_delivery,
        question_delivery_settled,
        quiz_states,
        record_answer_delivery_after_terminal,
        record_quiz_delivery_failure,
    )
    from ouroboros.question_answers import _update_delivery, deliver_answer

    block = quiz_states(drive_root, task_id).get(quiz_id) or {}
    if block.get("state") != "answered" or question_delivery_settled(block) or sr.secret_request_of(block) is None:
        return block
    if sr.allow_pending(block):
        if not sr.reserve_allow(task_id, quiz_id):  # its own step holds the reservation and delivers once settled
            return block
        try:
            # Re-read under the reservation: a live step may have settled, advanced or removed the block meanwhile.
            block = quiz_states(drive_root, task_id).get(quiz_id) or {}
            if (block.get("state") != "answered" or question_delivery_settled(block)
                    or sr.secret_request_of(block) is None):
                return block
            if sr.allow_pending(block):
                from ouroboros.skill_grant_follow import orphan_facts

                allow = (block.get("secret_outcome") or {}).get("allow") or {}
                decision = {"decision_id": f"quiz:{task_id}:{quiz_id}", "request_id": str(block.get("request_id") or "")}
                # No step of this process holds it (a restart cut it): never run now; record what its records say.
                block = (sr.record_allow(drive_root, decision, orphan_facts(drive_root, decision, allow),
                                         expect=(str(allow.get("state") or ""),))
                         or quiz_states(drive_root, task_id).get(quiz_id) or {})
        finally:
            sr.release_allow(task_id, quiz_id)
        if (block.get("state") != "answered" or question_delivery_settled(block)
                or sr.secret_request_of(block) is None or sr.allow_pending(block)):
            return block
    claimed = claim_quiz_delivery(drive_root, task_id, quiz_id, force=not recovery)
    if not claimed:
        return block
    task, refusal = _mail_holder(task_id)
    if task is None and refusal != "task_not_reading" and not (claimed.get("delivery") or {}).get("read_at"):
        claimed = record_answer_delivery_after_terminal(drive_root, task_id, quiz_id) or claimed
    destination, reason = _late_answer_destination(drive_root, task_id, claimed)
    if destination is not None:
        try:
            _record_quiz_answer_history(drive_root, task_id, task, claimed, duplicate=False)
        except Exception as exc:
            log.warning("Secret request history row remains owed for %s: %s", quiz_id, type(exc).__name__)
            record_quiz_delivery_failure(drive_root, task_id, quiz_id, "quiz_history_write_failed")
            return quiz_states(drive_root, task_id).get(quiz_id) or claimed

    def history_only(root: Any, asker: str, qid: str, _block: Any, source: str = "web") -> Tuple[bool, str]:
        # No living holder: the written history row is the whole delivery; no fresh head starts.
        if not ((quiz_states(root, asker).get(qid) or {}).get("delivery") or {}).get("history_at"):
            return False, reason or "publication_address_unknown"
        _update_delivery(root, asker, qid, {"route": "history", "msg_id": "", "reason_code": ""})
        return True, ""

    _forwarded, why = deliver_answer(drive_root, task_id, quiz_id, claimed, _mail_holder, history_only)
    if reason or why:
        record_quiz_delivery_failure(drive_root, task_id, quiz_id, reason or why)
    return quiz_states(drive_root, task_id).get(quiz_id) or claimed


def _save_locked(drive_root: pathlib.Path, decision_id: str, task_id: str, quiz_id: str, request_id: str,
                 request: Dict[str, Any], changes: Dict[str, str], observed: Dict[str, bool],
                 allow: Any = None) -> Tuple[int, Dict[str, Any], Dict[str, Any]]:
    """Claim -> file-locked mapping, key-state and allow check, write -> read-back -> completion. The third
    value is what the recorded save still owes after the lock: ``{task_id, quiz_id, allow}`` or ``{}``."""
    from ouroboros.gateway.owner_settings import (
        CommitBoundary,
        SettingsDocumentUnreadable,
        SettingsLockUnavailable,
        SettingsPreconditionFailed,
        _owner_settings_document,
        _owner_update_settings,
        fields_on_saved_document,
    )

    claim = sr.claim_save(drive_root, task_id, quiz_id, request_id)
    if not claim.get("ok"):
        code = str(claim.get("error") or "secret_request_refused")
        messages = {
            "secret_request_not_found": "No such secret request.",
            "not_a_secret_request": "This question is not a secret request.",
            "secret_request_settled": "This secret request was already saved or cancelled; nothing was saved now.",
            "secret_save_unconfirmed": "This save attempt was never confirmed and its result is unknown, so it "
                                       "is not repeated. Check status; saving again starts a new request.",
        }
        return (*_refusal(404 if code == "secret_request_not_found" else 409, code,
                          messages.get(code, "The secret request was refused; nothing was saved."),
                          decision_id, claim.get("block"), root=drive_root,
                          saved=None if code == "secret_save_unconfirmed" else False), {})
    if claim.get("duplicate"):  # never written again; an allow its first request left unsettled continues once
        owed = {"task_id": task_id, "quiz_id": quiz_id, "allow": True} if sr.allow_pending(claim["block"]) else {}
        return 200, _receipt(decision_id, claim["block"], duplicate=True), owed
    previous = dict(claim.get("previous_claim") or {})
    boundary = CommitBoundary()

    def check(document: Dict[str, Any]) -> str:
        # On the write side of the file lock's wait, against the one read the values land on.
        stale = sr.binding_refusal(drive_root, request)
        if stale:
            return f"secret_request_stale:{stale}"
        if sr.configured_fields(document, request) != observed:
            return _KEY_STATE_CHANGED
        return _ALLOW_STALE if sr.allow_refusal(drive_root, request, allow) else ""

    try:
        _owner_update_settings(fields_on_saved_document(dict(changes), check), boundary=boundary)
    except Exception as exc:
        if boundary.committed:  # unreachable today: the write is the writer's last step
            log.warning("Secret saved; a later settings step failed: %s", type(exc).__name__)
        else:
            sr.release_save(drive_root, task_id, quiz_id, request_id, previous)
            if isinstance(exc, SettingsDocumentUnreadable):
                return (*_refusal(409, "settings_unreadable", "Settings exist but could not be read, so nothing was "
                                  "saved or overwritten. Repair the settings file, then check status.", decision_id,
                                  sr_block(drive_root, task_id, quiz_id), root=drive_root), {})
            if isinstance(exc, SettingsPreconditionFailed) and str(exc) == _KEY_STATE_CHANGED:
                block = sr_block(drive_root, task_id, quiz_id)
                return (*_refusal(409, _KEY_STATE_CHANGED, "A key this form asks for was set or cleared in Settings "
                                  "after the form read it; nothing was saved. Review the form and save again.",
                                  decision_id, block, root=drive_root), {})
            if isinstance(exc, SettingsPreconditionFailed) and str(exc) == _ALLOW_STALE:
                return (*_refusal(409, _ALLOW_STALE, "The skill's files or installation changed after the form read "
                                  "them; nothing was saved or allowed. Review the form and choose again.", decision_id,
                                  sr_block(drive_root, task_id, quiz_id), root=drive_root), {})
            if isinstance(exc, SettingsPreconditionFailed):
                return (*_refusal(409, "secret_request_stale", "The skill no longer maps these fields to the same "
                                  "Settings keys; nothing was saved. Ask again.", decision_id), {})
            log.warning("Secret save refused before commit: %s", type(exc).__name__)
            code = "settings_locked" if isinstance(exc, SettingsLockUnavailable) else "secret_save_failed"
            return (*_refusal(503, code, f"Settings could not be written ({type(exc).__name__}); nothing was "
                              "saved. Try again.", decision_id), {})
    if boundary.committed:
        # The credential changed even if later read-back or completion recording fails.
        from ouroboros.skill_mcp import lifecycle_settled

        lifecycle_settled(reason="secret_saved")
    try:
        document = _owner_settings_document(strict=True)
        retained = all(document.get(key) == value for key, value in changes.items())
    except Exception as exc:
        log.warning("Secret written; its read-back failed: %s", type(exc).__name__)
        retained = None
    if not retained:
        # The commit is known; what Settings now holds is not. The claim stays ``writing``
        # (unknown) and nothing claims a save the read-back did not confirm.
        code = "secret_save_unverified" if retained is None else "secret_not_retained"
        return (*_refusal(500, code, "Settings were written, but reading the value back "
                          + ("failed" if retained is None else "did not match what was entered")
                          + ", so it is not reported saved and the task was not told. Check status before "
                          "saving again.", decision_id, saved=True), {})
    facts: Dict[str, Any] = {"replaced_fields": [field for field, was_set in observed.items() if was_set],
                             **({"allow": allow} if allow else {})}
    # Discovery stays here, inside the lock: a custom key joins a reader's grant request only once
    # saved, so only a read after the write is true, and a cancel in a gap before the completion
    # would record this landed write as an interrupted one. Absent facts are recorded as unknown.
    try:
        facts["consumers"] = sr.key_consumers(drive_root, sr.settings_keys(request))
    except Exception as exc:
        log.warning("Secret saved; its readers could not be read: %s", type(exc).__name__)
    # Reserved before the completion: no owed-delivery pass reads the recorded allow as an orphan meanwhile.
    reserved = bool(allow) and sr.reserve_allow(task_id, quiz_id)
    try:
        final = sr.finalize_save(drive_root, task_id, quiz_id, request_id, facts)
    except Exception as exc:
        log.warning("Secret saved; its completion could not be recorded: %s", type(exc).__name__)
        final = {}
    if not final.get("ok"):
        if reserved:
            sr.release_allow(task_id, quiz_id)
        return (*_refusal(500, "secret_completion_unrecorded", "The secret was saved to Settings, but the request "
                          "could not be marked complete and the task was not told. Check status before trying again.",
                          decision_id, saved=True), {})
    return 200, _receipt(decision_id, final["block"], duplicate=False), {
        "task_id": task_id, "quiz_id": quiz_id, "allow": bool(allow), "reserved": reserved}


def sr_block(drive_root: pathlib.Path, task_id: str, quiz_id: str) -> Dict[str, Any]:
    from ouroboros.owner_quiz import quiz_states

    return quiz_states(drive_root, task_id).get(quiz_id) or {}


def _settle_finished_asker(drive_root: pathlib.Path, task_id: str) -> None:
    """An ``open`` block is a projection the task-done seam may not have reached yet (a crash or
    post-work window). Settle it from the live queue as an ordinary answer does, so the card says
    the task finished and a save on it is recorded as late; this never refuses the save."""
    from ouroboros.gateway.task_decision import _live_root_task, _send_quiz_state
    from ouroboros.owner_quiz import reconcile_terminal

    try:
        task, refusal = _live_root_task(task_id)
        if task is None and refusal not in ("task_not_reading", "not_a_root_task"):
            for qid in reconcile_terminal(drive_root, task_id):
                _send_quiz_state(qid, task_id, "expired_terminal")
    except Exception as exc:  # the block's own state still decides, as before this read
        log.warning("Secret request could not read its asking task: %s", type(exc).__name__)


def _request_block(drive_root: pathlib.Path, decision_id: str, task_id: str,
                   quiz_id: str) -> Tuple[Dict[str, Any], Any]:
    block = sr_block(drive_root, task_id, quiz_id)
    if not block:
        return block, _refusal(404, "secret_request_not_found", "No such secret request.", decision_id)
    if sr.secret_request_of(block) is None:
        return block, _refusal(409, "not_a_secret_request", "This question is not a secret request.", decision_id)
    return block, None


def _save_secret(drive_root: pathlib.Path, body: Any) -> Tuple[int, Dict[str, Any], Dict[str, Any]]:
    """The request under the settings lock, and what its recorded save owes afterwards (``_save_locked``)."""
    from ouroboros.gateway.owner_settings import SettingsDocumentBusy, settings_document_mutation

    try:
        request_id, decision_id, task_id, quiz_id, action = _parse(body)
    except sr.SecretRequestError as exc:
        return (*_refusal(400, exc.code, str(exc), ""), {})  # nothing the client sent is echoed before it parses
    if action == "cancel":
        return (*cancel_secret(drive_root, decision_id, task_id, quiz_id, request_id), {})
    block, refused = _request_block(drive_root, decision_id, task_id, quiz_id)
    if refused:
        if action != "status" or block:
            return (*refused, {})
        block = sr.historical_secret_answer(drive_root, task_id, quiz_id)
        if not block:
            return (*refused, {})
        return 200, {"ok": True, "decision_id": decision_id, "secret_request": dict(block["secret_request"]),
                     **_lifecycle(block)}, {}
    if action == "status":
        if sr.consent_result_pending(block):
            from ouroboros.skill_grant_follow import publish_consent_outcomes, read_consent_outcomes

            settlements = read_consent_outcomes(drive_root)
            projected = sr.project_consent_result(block, task_id, settlements)
            if projected is not block:
                publish_consent_outcomes(drive_root, block["secret_outcome"]["allow"]["skill"], decision_id=decision_id)
                current = sr_block(drive_root, task_id, quiz_id)
                block = sr.project_consent_result(current, task_id, settlements) if current else projected
        return 200, {"ok": True, "decision_id": decision_id, "secret_request": dict(block["secret_request"]),
                     **_lifecycle(block, drive_root)}, {}
    request = sr.secret_request_of(block)
    try:
        changes = sr.validate_values(request, body.get("values"))
        observed = sr.validate_observed(request, body.get("observed"))
        allow = sr.validate_allow(request, body.get("allow"))
    except sr.SecretRequestError as exc:
        return (*_refusal(400, exc.code, str(exc), decision_id), {})
    _settle_finished_asker(drive_root, task_id)
    try:
        with settings_document_mutation():
            return _save_locked(drive_root, decision_id, task_id, quiz_id, request_id, request, changes, observed,
                                allow)
    except SettingsDocumentBusy as exc:
        return (*_refusal(503, "settings_busy", str(exc), decision_id), {})
    finally:
        changes.clear()


def _finish_save(drive_root: pathlib.Path, owed: Dict[str, Any], payload: Dict[str, Any],
                 progress: Any = None) -> Dict[str, Any]:
    """After the lock: the allow's own step (``sr.run_allow``), then the outcome's delivery, which waits for
    it, so the asker learns the save and the allow together. Returns the reply."""
    task_id, quiz_id = owed["task_id"], owed["quiz_id"]
    block: Dict[str, Any] = {}
    if owed.get("allow"):
        try:
            block = sr.run_allow(drive_root, task_id, quiz_id, progress=progress, reserved=bool(owed.get("reserved")))
        except Exception as exc:  # its result stays unrecorded: the owed-delivery pass reads it as unknown
            log.warning("Secret request allow step failed for %s: %s", quiz_id, type(exc).__name__)
            sr.release_allow(task_id, quiz_id)
    if not sr.allow_pending(block or sr_block(drive_root, task_id, quiz_id)):  # another request's step delivers
        try:
            block = deliver_secret_outcome(drive_root, task_id, quiz_id)
        except Exception as exc:
            from ouroboros.owner_quiz import record_quiz_delivery_failure

            log.warning("Secret request delivery remains owed for %s: %s", quiz_id, type(exc).__name__)
            record_quiz_delivery_failure(drive_root, task_id, quiz_id, "quiz_delivery_failed")
    block = sr_block(drive_root, task_id, quiz_id) or block
    _send_state(block, task_id)
    return _receipt(str(payload.get("decision_id") or ""), block, duplicate=bool(payload.get("duplicate"))) if block \
        else payload


def save_secret(drive_root: pathlib.Path, body: Any) -> Tuple[int, Dict[str, Any]]:
    """The whole decision in the calling thread: the save, then its allow step and delivery."""
    status, payload, owed = _save_secret(drive_root, body)
    return status, (_finish_save(drive_root, owed, payload) if owed else payload)


def cancel_secret(drive_root: pathlib.Path, decision_id: str, task_id: str, quiz_id: str,
                  request_id: str) -> Tuple[int, Dict[str, Any]]:
    """Owner cancel: a closure with typed ``secret_cancelled`` facts; first winner against a save."""
    from ouroboros.gateway.owner_settings import SettingsDocumentBusy, settings_document_mutation
    from ouroboros.owner_quiz import record_closed

    block, refused = _request_block(drive_root, decision_id, task_id, quiz_id)
    if refused:
        return refused
    try:
        with settings_document_mutation():
            facts = sr.cancel_outcome(sr_block(drive_root, task_id, quiz_id) or block)
            reason = sr.CANCEL_REASON_INTERRUPTED if facts.get("interrupted_save") else sr.CANCEL_REASON
            outcome = record_closed(drive_root, task_id, quiz_id=quiz_id, reason=reason, actor="owner",
                                    request_id=request_id, secret_outcome=facts)
    except SettingsDocumentBusy as exc:
        return _refusal(503, "settings_busy", str(exc), decision_id)
    block = outcome.get("block") or {}
    if not outcome.get("ok"):
        code = str(outcome.get("error") or "secret_cancel_refused")
        return _refusal(404 if code == "quiz_not_found" else 409, code,
                        "This secret request was already saved or closed." if code != "quiz_not_found"
                        else "No such secret request.", decision_id, block)
    if not outcome.get("duplicate"):
        from supervisor.events_chat_delivery import _handle_send_quiz_state
        from supervisor.message_bus import get_bridge

        try:
            _handle_send_quiz_state({"task_id": task_id, "quiz_id": quiz_id, "owner_retry": True},
                                    SimpleNamespace(DRIVE_ROOT=drive_root, bridge=get_bridge()))
        except Exception as exc:
            log.warning("Secret request cancellation delivery remains owed for %s: %s", quiz_id, type(exc).__name__)
    return 200, _receipt(decision_id, sr_block(drive_root, task_id, quiz_id) or block,
                         duplicate=bool(outcome.get("duplicate")))


def is_secret_question(drive_root: pathlib.Path, task_id: str, quiz_id: str) -> bool:
    return sr.secret_request_of(sr_block(drive_root, task_id, quiz_id)) is not None


async def api_secret_decision(request: Request) -> JSONResponse:
    """POST /api/decisions/secret — ``saveSecretToken``, its cancel and its status read."""
    from ouroboros.gateway.owner_settings import owner_audit
    from ouroboros.skill_lifecycle_queue import answer_or_job

    body = await request_json_or(request, None)
    action = body.get("action", "save") if isinstance(body, dict) else "save"
    allowing = isinstance(body, dict) and "allow" in body
    drive_root = request_drive_root(request)
    if not allowing:
        status, payload = await run_sync_to_completion(save_secret, drive_root, body)
    else:
        # One background work: save, lifecycle job and delivery. An early reply names an already
        # bound job; its save receipt was captured before the bind and can predate a live result.
        saved: Dict[str, Any] = {}

        def work(handle: Any) -> Tuple[int, Dict[str, Any]]:
            try:
                status, receipt, owed = _save_secret(drive_root, body)
            finally:
                values = body.get("values") if isinstance(body, dict) else None
                if isinstance(values, dict):
                    values.clear()  # the values leave with the save, not with the grant
            saved.update(status=status, payload=receipt)
            return status, (_finish_save(drive_root, owed, receipt, handle) if owed else receipt)

        result = await answer_or_job(work, log_label="secret form allow", allow_pending=True)
        if isinstance(result, dict) and result.get("pending"):
            status, payload = saved["status"], {**saved["payload"], "allow_job": result["job_id"]}
        else:
            status, payload = result
    body = None  # the values leave this frame with the request; the audit names only host facts
    if action in ("save", "cancel"):
        event = payload.get("secret_cancelled" if action == "cancel" else "secret_filled") or payload.get(
            "secret_replaced") or {}
        owner_audit(request, "secret_request_cancel" if action == "cancel" else "secret_request_save", {
            "decision_id": str(payload.get("decision_id") or ""), "plugin": str(event.get("pluginName") or ""),
            "fields": list(event.get("fields") or []),
            "outcome": "ok" if payload.get("ok") else str(payload.get("reason_code") or "refused"),
            "saved": payload.get("saved"),
            **({"allow": str(((payload.get("secret_outcome") or {}).get("allow") or {}).get("state") or "not_recorded")}
               if allowing else {}),
        })
    return JSONResponse(payload, status_code=status, headers=_NO_STORE)


__all__ = ["api_secret_decision", "cancel_secret", "deliver_secret_outcome", "is_secret_question", "save_secret"]
