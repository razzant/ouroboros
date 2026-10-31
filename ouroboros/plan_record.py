"""The per-task plan record: the mind's revisions, its asks to the council and
each participant's own replies, with every projection the host derives from them.

One intention, one record, three kinds of fact the mind writes through ``plan_task``:

* a REVISION — goal, ``done_when`` (the acceptance claim grammar: id / claim /
  support), ``affected_paths`` and the plan prose; omitted fields carry over,
  identical content creates no revision; the body flag is computed at write;
* an ASK — the revision it carried (``None``: a reading of the request before any
  plan), the mind's message, the attachments, the participants it was dispatched to
  (each with the recorded fact whether it was an owner-marked reviewer OF THAT ASK)
  and the review-ledger record that retains the exact texts;
* a REPLY — one participant's parsed ``{prose, objections, withdrawn}``, or its
  typed absence (pending, failed, not dispatched, unparseable).

Everything else is derived here and never stored: the open objections (a fold over
EACH SEAT'S OWN replies to stored revisions — an objection stands until the same seat
lists it under ``withdrawn``; omission keeps it and flags ``not_adjudicated``; an
unparseable reply changes nothing; a reading's reply is advice and moves no objection),
who last saw which revision, the gate projection
(``gate_projection``), the claims view (``acceptance_view``) and the review
history the context views read (``record_history``). There is no cycle counter, no
quorum, no aggregate and no fingerprint routing for plans.

Durable form: the task result's ``plan_record`` key (schema 3). The exact text of
every revision is retained once in the task's artifact store (``retain_revision``);
the index keeps the prose inline while it fits and names the exact source otherwise.
Finished v1/v2 ``plan_review_state`` records are never rewritten:
``import_legacy_v2`` reads them (open blocking findings of an in-flight task become
open objections of their seats; a closed wave's or an Advisory author's selected
claims become ``done_when`` with legacy provenance).
"""

from __future__ import annotations

import copy
import difflib
import json
import re
from hashlib import sha256
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional

from ouroboros.store import StoreError
from ouroboros.task_result_schema import require_writable_task_result_schema, stamp_task_result_schema
from ouroboros.task_store import patch_document, read_document
from ouroboros.utils import utc_now_iso

PLAN_RECORD_KEY = "plan_record"
PLAN_RECORD_VERSION = 3
# The bounded hot index: past it, the oldest prose moves out to its exact source.
_PLAN_RECORD_MAX_BYTES = 1_000_000
_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}")
PARTICIPANT_STATUSES = frozenset({"pending", "answered", "unparseable", "failed", "not_dispatched"})
GATE_STATUSES = frozenset({"not_required", "held", "objections_open", "awaiting", "clear",
                           "rail_degraded", "unreadable"})
LEGACY_ASK_ID = "legacy"
_DONE_WHEN_KEYS = ("id", "claim", "support", "surface")


class PlanRecordInvalid(ValueError):
    """The durable plan record cannot be read as written."""


# ------------------------------------------------------------------ record shape

def empty_record() -> Dict[str, Any]:
    return {"schema_version": PLAN_RECORD_VERSION, "revisions": [], "asks": []}


def _as_text(value: Any) -> str:
    return str(value if value is not None else "").strip()


def normalize_done_when(raw: Any) -> List[Dict[str, str]]:
    """``done_when`` rows in the acceptance claim grammar (``normalize_acceptance_claims``
    mints ids, dedups them and drops empty claims); the stored row keeps id / claim /
    support (+ surface when given) and no priority."""
    from ouroboros.contracts.task_contract import normalize_acceptance_claims

    rows = []
    for row in normalize_acceptance_claims(raw if isinstance(raw, list) else []):
        kept = {key: str(row.get(key) or "") for key in _DONE_WHEN_KEYS if key != "surface"}
        if str(row.get("surface") or ""):
            kept["surface"] = str(row["surface"])
        rows.append(kept)
    return rows


def content_sha256(fields: Mapping[str, Any]) -> str:
    """Identity of a revision's content (provenance only, never routing)."""
    payload = {
        "goal": _as_text(fields.get("goal")), "plan": str(fields.get("plan") or ""),
        "done_when": [dict(row) for row in fields.get("done_when") or []],
        "affected_paths": [str(p) for p in fields.get("affected_paths") or []],
    }
    return sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def compose_revision(latest: Optional[Mapping[str, Any]], *, goal: Any = None, plan: Any = None,
                     done_when: Any = None, affected_paths: Any = None) -> Dict[str, Any]:
    """The next revision's content: the given fields over the latest revision's
    (omitted fields carry over). ``done_when`` is normalized; paths are stripped."""
    base = latest or {}
    fields = {
        "goal": _as_text(goal) if goal is not None else str(base.get("goal") or ""),
        "plan": str(plan) if plan is not None else str(base.get("plan") or ""),
        "done_when": normalize_done_when(done_when) if done_when is not None else [dict(r) for r in base.get("done_when") or []],
        "affected_paths": ([_as_text(p) for p in affected_paths if _as_text(p)] if affected_paths is not None
                           else [str(p) for p in base.get("affected_paths") or []]),
    }
    fields["content_sha256"] = content_sha256(fields)
    return fields


def latest_revision(record: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    revisions = record.get("revisions") or []
    return revisions[-1] if revisions else None


def revision_by_number(record: Mapping[str, Any], n: Any) -> Optional[Dict[str, Any]]:
    return next((r for r in record.get("revisions") or [] if r.get("n") == n), None)


READING_SUBJECT = "the reading of the request"


def ask_subject(revision: Any, *, plan: bool = False) -> str:
    """What an ask was about, as every reader names it: ``revision N`` (``plan revision N``),
    or the reading of the request an ask before any plan carries (``revision: None``)."""
    return READING_SUBJECT if revision is None else f"{'plan ' if plan else ''}revision {revision}"


def revision_delta(previous: Optional[Mapping[str, Any]], current: Mapping[str, Any]) -> Dict[str, Any]:
    """What changed between two revisions: field flags, ``done_when`` by id (added /
    removed / reworded with both texts) and a unified diff of the plan prose."""
    if previous is None:
        return {"first": True}
    before = {str(r.get("id") or ""): r for r in previous.get("done_when") or []}
    after = {str(r.get("id") or ""): r for r in current.get("done_when") or []}
    reworded = [{"id": key, "before": before[key], "after": after[key]}
                for key in after if key in before and after[key] != before[key]]
    prose_before, prose_after = str(previous.get("plan") or ""), str(current.get("plan") or "")
    diff = list(difflib.unified_diff(prose_before.splitlines(), prose_after.splitlines(),
                                     fromfile=f"revision {previous.get('n')}", tofile=f"revision {current.get('n')}",
                                     lineterm="", n=2)) if prose_before != prose_after else []
    return {
        "first": False, "from": previous.get("n"), "to": current.get("n"),
        "goal_changed": str(previous.get("goal") or "") != str(current.get("goal") or ""),
        "plan_changed": prose_before != prose_after,
        "done_when_added": [dict(after[k]) for k in after if k not in before],
        "done_when_removed": [dict(before[k]) for k in before if k not in after],
        "done_when_reworded": reworded,
        "affected_paths_added": [p for p in current.get("affected_paths") or [] if p not in (previous.get("affected_paths") or [])],
        "affected_paths_removed": [p for p in previous.get("affected_paths") or [] if p not in (current.get("affected_paths") or [])],
        "body_changed": bool(previous.get("body")) != bool(current.get("body")),
        "plan_diff": diff,
    }


def validate_record(value: Any) -> Dict[str, Any]:
    """The stored record, validated; absent → empty; malformed → ``PlanRecordInvalid``."""
    if value in (None, {}):
        return empty_record()
    if not isinstance(value, dict) or value.get("schema_version") != PLAN_RECORD_VERSION:
        raise PlanRecordInvalid("PLAN_RECORD_INVALID: unsupported or malformed schema")
    revisions, asks = value.get("revisions"), value.get("asks")
    if not isinstance(revisions, list) or not isinstance(asks, list) or not isinstance(value.get("attempts", []), list):
        raise PlanRecordInvalid("PLAN_RECORD_INVALID: revisions, asks and attempts must be lists")
    for index, revision in enumerate(revisions, start=1):
        if not isinstance(revision, dict) or revision.get("n") != index:
            raise PlanRecordInvalid("PLAN_RECORD_INVALID: revisions must be numbered 1..N in order")
        if not isinstance(revision.get("done_when"), list) or not isinstance(revision.get("affected_paths"), list):
            raise PlanRecordInvalid("PLAN_RECORD_INVALID: revision done_when and affected_paths must be lists")
    seen: set = set()
    for ask in asks:
        if not isinstance(ask, dict) or not _ID_RE.fullmatch(str(ask.get("ask_id") or "")) or ask["ask_id"] in seen:
            raise PlanRecordInvalid("PLAN_RECORD_INVALID: ask ids must be unique")
        seen.add(ask["ask_id"])
        # ``revision: None`` is a reading of the request before any plan; anything else must be
        # the number of a stored revision (a bool is not one, though ``True == 1``).
        if "revision" not in ask or (ask["revision"] is not None and (
                isinstance(ask["revision"], bool) or not isinstance(ask["revision"], int)
                or revision_by_number(value, ask["revision"]) is None)):
            raise PlanRecordInvalid("PLAN_RECORD_INVALID: an ask names a revision the record does not hold")
        if not isinstance(ask.get("participants"), list):
            raise PlanRecordInvalid("PLAN_RECORD_INVALID: ask participants must be a list")
        for participant in ask["participants"]:
            if not isinstance(participant, dict) or not str(participant.get("seat_id") or ""):
                raise PlanRecordInvalid("PLAN_RECORD_INVALID: a participant needs a seat id")
            if str(participant.get("status") or "") not in PARTICIPANT_STATUSES:
                raise PlanRecordInvalid("PLAN_RECORD_INVALID: unknown participant status")
    return {**empty_record(), **copy.deepcopy(value)}


# ------------------------------------------------------------------ reply parsing

_REPLY_KEYS = ("prose", "objections", "withdrawn")


def _reply_object(text: str) -> Optional[Dict[str, Any]]:
    """The reply object in a participant's text: scanning every ``{``, the first JSON object
    that carries at least one of ``prose`` / ``objections`` / ``withdrawn``. A quoted
    ``done_when`` row, a fenced example or any other object in a preamble is not a reply."""
    raw = str(text or "")
    decoder = json.JSONDecoder()
    start = raw.find("{")
    while start >= 0:
        try:
            value, _end = decoder.raw_decode(raw, start)
        except ValueError:
            value = None
        if isinstance(value, dict) and any(key in value for key in _REPLY_KEYS):
            return value
        start = raw.find("{", start + 1)
    return None


def _normalized_id(value: Any) -> str:
    """An LLM-written id made addressable: whitespace trimmed and collapsed to ``_`` so the
    same seat's later withdrawal of the same id matches; empty when nothing usable remains."""
    text = "_".join(_as_text(value).split())
    return text if _ID_RE.fullmatch(text) else ""


def parse_reply(text: str) -> tuple[Optional[Dict[str, Any]], str]:
    """A participant's reply → ``({prose, objections[{id,text}], withdrawn[{id,why}]}, "")``
    or ``(None, error)``. A missing list is an empty list (an omitted ``withdrawn`` keeps
    every objection). A malformed ITEM — not an object, no usable id, a body that is not
    text — is skipped and NAMED under ``ignored`` while the valid items act: one bad id never
    voids the withdrawal beside it. A list that is not a list, or prose that is not text, is
    still an error for the whole reply. Ids are normalized (``_normalized_id``)."""
    payload = _reply_object(text)
    if payload is None:
        return None, "no reply object found (no JSON object carrying prose, objections or withdrawn)"
    prose = payload.get("prose", "")
    if prose is not None and not isinstance(prose, str):
        return None, "prose must be text"
    reply: Dict[str, Any] = {"prose": str(prose or ""), "objections": [], "withdrawn": []}
    ignored: List[Dict[str, Any]] = []
    for key, text_key in (("objections", "text"), ("withdrawn", "why")):
        items = payload.get(key)
        if items is None:
            continue
        if not isinstance(items, list):
            return None, f"{key} must be an array"
        by_id: Dict[str, Dict[str, str]] = {}
        for index, item in enumerate(items):
            if not isinstance(item, dict):
                ignored.append({"key": key, "index": index, "id": "", "reason": "not an object"})
                continue
            item_id, body = _normalized_id(item.get("id")), item.get(text_key, "")
            if not item_id:
                ignored.append({"key": key, "index": index, "id": _as_text(item.get("id")), "reason": "no usable id"})
            elif body is not None and not isinstance(body, str):
                ignored.append({"key": key, "index": index, "id": item_id, "reason": f"{text_key} must be text"})
            else:
                by_id[item_id] = {"id": item_id, text_key: str(body or "")}
        reply[key] = list(by_id.values())
    if ignored:
        reply["ignored"] = ignored
    return reply, ""


# ------------------------------------------------------------------ folds

def _reply_rows(record: Mapping[str, Any], legacy: Optional[Mapping[str, Any]] = None) -> List[Dict[str, Any]]:
    """Every participant row that carries a reply to a stored revision, in ask order, legacy
    first. A reading of the request (an ask before any plan, ``revision: None``) is not among
    them, whenever it settles: its reply is advice that raises, restates and withdraws no
    standing objection, and its replies, history and panel rows keep showing it."""
    rows: List[Dict[str, Any]] = []
    for objection in (legacy or {}).get("open_objections") or []:
        rows.append({"ask_id": LEGACY_ASK_ID, "revision": 0, "seat_id": str(objection.get("seat_id") or ""),
                     "owner_marked": bool(objection.get("owner_marked", True)),
                     "reply": {"objections": [{"id": str(objection.get("id") or ""), "text": str(objection.get("text") or "")}],
                               "withdrawn": []}, "legacy": True})
    for ask in record.get("asks") or []:
        if ask.get("revision") is None:
            continue
        for participant in ask.get("participants") or []:
            if participant.get("status") == "answered" and isinstance(participant.get("reply"), dict):
                rows.append({"ask_id": ask.get("ask_id"), "revision": ask.get("revision"),
                             "seat_id": str(participant.get("seat_id") or ""),
                             "owner_marked": bool(participant.get("owner_marked")),
                             "reply": participant["reply"], "legacy": False})
    return rows


def fold_objections(record: Mapping[str, Any], legacy: Optional[Mapping[str, Any]] = None) -> List[Dict[str, Any]]:
    """Every objection ever raised, with its current standing: the fold over each seat's
    OWN replies. Raising or restating sets the text and the ask it answered in (and the
    owner-marked fact OF THAT ASK); only the same seat's ``withdrawn`` closes it; a reply
    that mentions neither leaves it open and flags ``not_adjudicated`` until the seat
    speaks to it again. A withdrawal of an id the seat never raised is recorded as
    ``unknown_withdrawal`` and changes nothing."""
    objections: Dict[tuple, Dict[str, Any]] = {}
    for row in _reply_rows(record, legacy):
        seat = row["seat_id"]
        reply = row["reply"]
        mentioned: set = set()
        for item in reply.get("objections") or []:
            key = (seat, str(item.get("id") or ""))
            mentioned.add(key)
            current = objections.get(key)
            if current is None:
                objections[key] = current = {"seat_id": seat, "id": key[1], "raised_in": row["ask_id"],
                                             "raised_revision": row["revision"], "legacy": row["legacy"]}
            current.update(text=str(item.get("text") or ""), ask_id=row["ask_id"], revision=row["revision"],
                           owner_marked=row["owner_marked"], open=True, not_adjudicated=False,
                           withdrawn_in="", why="")
        for item in reply.get("withdrawn") or []:
            key = (seat, str(item.get("id") or ""))
            mentioned.add(key)
            current = objections.get(key)
            if current is None or not current.get("open"):
                objections.setdefault(("unknown", f"{seat}:{key[1]}:{row['ask_id']}"), {
                    "seat_id": seat, "id": key[1], "unknown_withdrawal": True, "ask_id": row["ask_id"],
                    "open": False, "why": str(item.get("why") or "")})
                continue
            current.update(open=False, not_adjudicated=False, withdrawn_in=row["ask_id"], why=str(item.get("why") or ""))
        for key, current in objections.items():
            if key[0] == seat and current.get("open") and key not in mentioned and not current.get("unknown_withdrawal"):
                current["not_adjudicated"] = True
    out = []
    for key, current in objections.items():
        if current.get("unknown_withdrawal"):
            out.append(current)
            continue
        out.append({**current, "holds": bool(current.get("open") and current.get("owner_marked"))})
    return out


def open_objections(record: Mapping[str, Any], legacy: Optional[Mapping[str, Any]] = None) -> List[Dict[str, Any]]:
    return [o for o in fold_objections(record, legacy) if o.get("open")]


def participant_views(record: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Per seat: the revision it last saw (the newest ask that dispatched it), the ask
    it last answered, and its latest prose."""
    views: Dict[str, Dict[str, Any]] = {}
    for ask in record.get("asks") or []:
        for participant in ask.get("participants") or []:
            seat = str(participant.get("seat_id") or "")
            view = views.setdefault(seat, {"seat_id": seat, "model": str(participant.get("model") or ""),
                                           "owner_marked": bool(participant.get("owner_marked")),
                                           "last_seen_revision": None, "last_seen_ask": "", "last_answered_ask": "",
                                           "latest_prose": "", "status": ""})
            view["model"] = str(participant.get("model") or view["model"])
            view["status"] = str(participant.get("status") or "")
            if participant.get("dispatched"):
                view["last_seen_revision"], view["last_seen_ask"] = ask.get("revision"), str(ask.get("ask_id") or "")
            if participant.get("status") == "answered" and isinstance(participant.get("reply"), dict):
                view["last_answered_ask"] = str(ask.get("ask_id") or "")
                view["latest_prose"] = str(participant["reply"].get("prose") or "")
    return views


def pending_asks(record: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """Asks with a counted participant pending; advisers retain their own custody only."""
    return [ask for ask in record.get("asks") or []
            if any(p.get("owner_marked") and p.get("status") == "pending" for p in ask.get("participants") or [])]


def asks_dispatched(record: Mapping[str, Any]) -> int:
    """Asks that reached a participant counted by that ask."""
    return sum(1 for ask in record.get("asks") or []
               if any(p.get("dispatched") and p.get("owner_marked") for p in ask.get("participants") or []))


def engaged(record: Mapping[str, Any]) -> bool:
    return bool(record.get("revisions") or record.get("asks"))


def unroutable_asks(record: Mapping[str, Any]) -> int:
    """How many asks the mind made that POSITIVELY reached nobody: the recorded attempts (refused
    before any row was written) plus the recorded asks whose every participant carries a typed
    pre-dispatch refusal (``not_dispatched``). An ask whose dispatch outcome is unknown (an
    exception after the hand-off) is never counted here: missing state proves no "no send"."""
    refused = sum(1 for ask in record.get("asks") or []
                  if ask.get("participants") and all(p.get("status") == "not_dispatched" for p in ask["participants"]))
    return len(record.get("attempts") or []) + refused


def unknown_outcome_asks(record: Mapping[str, Any]) -> List[str]:
    """Asks with a seat whose outcome is unknown (pending and unresolved: custody lost, or the
    dispatch lost after the hand-off) — a new ask to them may pay twice."""
    return [str(a.get("ask_id") or "") for a in record.get("asks") or []
            if any(p.get("status") == "pending" and p.get("unresolved") for p in a.get("participants") or [])]


# ------------------------------------------------------------------ gate

def gate_projection(record: Mapping[str, Any], *, enforcement: str, cyber_pro: bool, hurry: bool,
                    hard_rail: str = "", legacy: Optional[Mapping[str, Any]] = None,
                    unreadable: bool = False) -> Dict[str, Any]:
    """The ONE finalization decision over the plan record (pure).

    ``held`` ⟺ Blocking, not Cyber Pro, no owner hurry, and some open objection was
    raised by an owner-marked reviewer of the ask it answered — for any task. Silence,
    pending asks, refusals and mind-added critics never hold. An unreadable record
    is engaged, never absent.
    """
    policy = "blocking" if str(enforcement or "").lower() == "blocking" else "advisory"
    legacy_engaged = bool((legacy or {}).get("engaged"))
    if not unreadable and not engaged(record) and not legacy_engaged:
        return {"required": False, "allow": True, "status": "not_required", "enforcement": policy,
                "open_objections": [], "holding": [], "pending_asks": [], "revision": 0}
    objections = open_objections(record, legacy)
    holding = [o for o in objections if o.get("holds")]
    pending = [str(a.get("ask_id") or "") for a in pending_asks(record)] + [str(x) for x in (legacy or {}).get("pending_asks") or []]
    dispatched = asks_dispatched(record) + int((legacy or {}).get("asks_dispatched") or 0)
    latest = latest_revision(record)
    decision: Dict[str, Any] = {
        "required": True, "enforcement": policy,
        "open_objections": [{k: o.get(k) for k in ("seat_id", "id", "text", "holds", "not_adjudicated", "ask_id",
                                                     "revision", "owner_marked", "legacy")} for o in objections],
        "holding": [f"{o['seat_id']}:{o['id']}" for o in holding],
        "pending_asks": pending, "asks_dispatched": dispatched, "asks_unroutable": unroutable_asks(record),
        "unknown_outcome": unknown_outcome_asks(record),
        "revision": int(latest.get("n") or 0) if latest else 0,
        "unreadable": bool(unreadable), "reason": str(hard_rail or ""),
        "source": "legacy_v2" if (legacy_engaged and not engaged(record)) else "plan_record",
    }
    if cyber_pro:
        decision["decision_authority"] = "cyber_pro"
    if hurry and policy == "blocking":
        decision["owner_hurry_local_advisory"] = True
        decision["configured_enforcement"] = "blocking"
    if hard_rail:
        status, allow = "rail_degraded", True
    elif cyber_pro:
        status, allow = ("objections_open" if objections else "awaiting" if pending else "clear"), True
    elif unreadable:
        status, allow = "unreadable", False  # a gate that cannot read its authority is engaged, not absent
    elif holding and policy == "blocking" and not hurry:
        status, allow = "held", False
    elif objections:
        status, allow = "objections_open", True
    elif pending:
        status, allow = "awaiting", True
    else:
        status, allow = "clear", True
    decision.update(status=status, allow=allow, review_only_awaited=status == "awaiting")
    return decision


# ------------------------------------------------------------------ claims view

def acceptance_view(record: Mapping[str, Any], ingress_claims: List[Dict[str, Any]], *,
                    legacy: Optional[Mapping[str, Any]] = None) -> tuple[List[Dict[str, str]], str, Dict[str, Any]]:
    """``(claims, source, exhibit)`` — the claims a task's result is held to (result
    review, plan facts): ingress claims win; otherwise the LATEST revision's
    ``done_when`` binds, open objections or not (legacy selected claims when the task
    only has a v1/v2 record). The exhibit carries the ACTUAL text
    of every revision's ``done_when``, which revisions the council saw, and the
    objections open now — and, when ingress wins, the adopted promises stay visible."""
    from ouroboros.contracts.task_contract import normalize_acceptance_claims

    revisions = list(record.get("revisions") or [])
    seen_by: Dict[int, List[str]] = {}
    for ask in record.get("asks") or []:
        if any(p.get("dispatched") for p in ask.get("participants") or []):
            seen_by.setdefault(int(ask.get("revision") or 0), []).append(str(ask.get("ask_id") or ""))
    history = [{
        "n": r.get("n"), "ts": r.get("ts"), "goal": r.get("goal"), "done_when": [dict(x) for x in r.get("done_when") or []],
        "affected_paths": list(r.get("affected_paths") or []), "body": bool(r.get("body")),
        "seen_by_asks": seen_by.get(int(r.get("n") or 0), []),
    } for r in revisions]
    objections = [{k: o.get(k) for k in ("seat_id", "id", "text", "holds", "not_adjudicated", "ask_id", "revision")}
                  for o in open_objections(record, legacy)]
    latest = latest_revision(record)
    exhibit: Dict[str, Any] = {"revisions": history, "open_objections": objections}
    if not revisions and legacy:
        exhibit["legacy"] = {k: legacy.get(k) for k in ("done_when_source", "fingerprint") if legacy.get(k)}
    if ingress_claims:
        exhibit["binding"] = "ingress_contract"
        if latest:
            exhibit["adopted_promises"] = [dict(x) for x in latest.get("done_when") or []]
        return ingress_claims, "ingress_contract", exhibit
    if latest is not None:
        exhibit["binding"] = f"latest_revision:{latest.get('n')}"
        return normalize_acceptance_claims(latest.get("done_when") or []), "plan_review", exhibit
    legacy_claims = normalize_acceptance_claims((legacy or {}).get("done_when") or [])
    if legacy_claims:
        exhibit["binding"] = str((legacy or {}).get("done_when_source") or "legacy")
        return legacy_claims, "plan_review", exhibit
    exhibit["binding"] = "none"
    return [], "", exhibit


# ------------------------------------------------------------------ legacy import

def import_legacy_v2(state: Any, *, author_plan: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """Read-only projection of a v1/v2 ``plan_review_state``: ``engaged`` (the task once
    entered the old gate), the OPEN blocking findings of the current open wave as
    objections of their seats (pool rows of that wave were owner-marked), and the
    claims the old organ had selected (a CLOSED current wave's spec claims, or an
    Advisory author ``finish`` selection) as ``done_when`` with legacy provenance, and
    an open wave whose reviewers had not all answered as one pending ask (``legacy wave
    <fingerprint>``: silence never holds, but the final answer discloses it).
    Malformed attempts and waves are history, never new objections or promises."""
    if not isinstance(state, dict):
        return {"engaged": False, "open_objections": [], "done_when": [], "done_when_source": "", "asks_dispatched": 0,
                "pending_asks": []}
    waves = [w for w in state.get("waves") or [] if isinstance(w, dict)]
    attempt = state.get("current_attempt") if isinstance(state.get("current_attempt"), dict) else {}
    legacy_v1 = state.get("legacy_v1_projection") if isinstance(state.get("legacy_v1_projection"), dict) else {}
    engaged_fact = bool(waves or attempt or legacy_v1.get("status") not in (None, "", "absent"))
    fingerprint = str(attempt.get("fingerprint") or "")
    matched = next((w for w in waves if str(w.get("request_fingerprint") or "") == fingerprint), None) if fingerprint else None
    # Claims come only from the wave the attempt still points at; objections and owed answers come
    # from the LATEST wave even when the author moved on (a changed spec at the cap, an author plan):
    # a reviewer's standing objection is withdrawn by its raiser, never by a revision (DESIGN §8).
    current = matched if matched is not None else (waves[-1] if waves else None)
    objections: List[Dict[str, Any]] = []
    done_when: List[Dict[str, Any]] = []
    source = ""
    pending: List[str] = []
    pending_wave: Dict[str, Any] = {}
    if isinstance(current, dict) and not current.get("closed"):
        actors = [a for a in current.get("actors") or [] if isinstance(a, dict)]
        owed = [a for a in actors if a.get("late_result_pending")
                or str(a.get("operation_state") or "") in ("pending_dispatch", "in_flight")]
        if current.get("custody_pending") or owed:
            wave_fp = str(current.get("request_fingerprint") or fingerprint)
            pending.append(f"legacy wave {wave_fp[:12] or '?'}")
            # The rows a reader may still collect, read-only, through the shared collector (03 D7):
            # paid before the upgrade, never re-sent, never written back into this state.
            pending_wave = {"fingerprint": wave_fp, "retry_key": str(current.get("retry_key") or ""), "actors": copy.deepcopy(owed)}
        for finding in current.get("findings") or []:
            if not isinstance(finding, dict) or str(finding.get("class") or "") != "blocking":
                continue
            seat = str(finding.get("slot") or "")
            fid = str(finding.get("finding_id") or finding.get("id") or "")
            if seat and fid:
                objections.append({"seat_id": seat, "id": fid, "owner_marked": True,
                                   "text": str(finding.get("summary") or ""), "fingerprint": str(current.get("request_fingerprint") or fingerprint)})
    author = (author_plan or {}).get("author_disposition") if isinstance(author_plan, Mapping) else None
    if isinstance(author, Mapping) and str(author.get("action") or "finish") == "finish" and str(author.get("enforcement") or "") == "advisory":
        spec = author_plan.get("spec") if isinstance(author_plan.get("spec"), Mapping) else {}
        done_when, source = normalize_done_when(spec.get("acceptance_claims")), "legacy_author_plan"
    elif isinstance(matched, dict) and matched.get("closed"):
        spec = matched.get("spec") if isinstance(matched.get("spec"), Mapping) else {}
        done_when, source = normalize_done_when(spec.get("acceptance_claims") or matched.get("acceptance_claims")), "legacy_closed_wave"
    elif legacy_v1.get("status") == "closed":
        done_when, source = normalize_done_when(legacy_v1.get("acceptance_claims")), "legacy_closed_wave"
    return {"engaged": engaged_fact, "open_objections": objections, "done_when": done_when, "done_when_source": source,
            "fingerprint": fingerprint, "pending_asks": pending, "pending_wave": pending_wave,
            # a wave reached its reviewers unless the old engine marked it unpaid with nothing to show
            # for it (a free replay, an all-skipped dispatch); roster rows, findings or owed custody prove the send
            "asks_dispatched": sum(1 for w in waves if w.get("paid") is not False or w.get("actors") or w.get("findings")
                                   or w.get("custody_pending"))}


def legacy_facts(root: Any, task_id: str) -> tuple[Dict[str, Any], Dict[str, str]]:
    """``(legacy, error)`` — the ONE loader of the earlier-schema ``plan_review_state`` behind
    ``import_legacy_v2``: the state, the author plan its current attempt names, the projection.
    ``error`` is ``{}`` or names the read that failed (``source`` = ``plan_review_state`` or
    ``author_plan``, plus the message); the projection is then built without that piece. Each
    caller chooses its policy in one line: the gate holds as unreadable, the tool's read says
    so, the claims binder refuses to bind without the bar, the task detail names it."""
    from ouroboros.task_results import load_plan_review_state

    try:
        state = load_plan_review_state(root, task_id)
    except (OSError, TimeoutError, ValueError, StoreError) as exc:
        return import_legacy_v2(None), {"source": "plan_review_state", "error": f"{type(exc).__name__}: {exc}"}
    author_plan, error = None, {}
    if state.get("current_attempt"):
        from ouroboros.tools.plan_review_artifacts import current_author_plan

        try:
            author_plan = current_author_plan(root, task_id, state)
        except (OSError, ValueError) as exc:
            error = {"source": "author_plan", "error": f"{type(exc).__name__}: {exc}"}
    return import_legacy_v2(state, author_plan=author_plan), error


# ------------------------------------------------------------------ read view

def read_view(record: Mapping[str, Any], legacy: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """The free state the mind reads: the current revision, open objections per
    participant (with ``not_adjudicated``), who last saw which revision, the latest
    contributions and the pending asks."""
    latest = latest_revision(record)
    objections = fold_objections(record, legacy)
    by_seat: Dict[str, List[Dict[str, Any]]] = {}
    for objection in objections:
        if objection.get("open"):
            by_seat.setdefault(objection["seat_id"], []).append(objection)
    views = participant_views(record)
    return {
        "revision": copy.deepcopy(latest) if latest else None,
        "revisions": len(record.get("revisions") or []),
        "asks": [{"ask_id": a.get("ask_id"), "ts": a.get("ts"), "revision": a.get("revision"), "message": a.get("message"),
                  "participants": [{**{k: p.get(k) for k in ("seat_id", "model", "owner_marked", "status", "dispatched",
                                                               "not_dispatched_reason", "error", "parse_error",
                                                               "failure_code", "reported_cause", "reset_at", "unresolved")},
                                    "ignored_items": len((p.get("reply") or {}).get("ignored") or []) if isinstance(p.get("reply"), dict) else 0}
                                   for p in a.get("participants") or []],
                  "record_id": a.get("record_id")} for a in record.get("asks") or []],
        "participants": views,
        "open_objections": {seat: [{k: o.get(k) for k in ("id", "text", "ask_id", "revision", "holds", "not_adjudicated", "legacy")}
                                   for o in rows] for seat, rows in by_seat.items()},
        "withdrawn": [{k: o.get(k) for k in ("seat_id", "id", "withdrawn_in", "why")}
                      for o in objections if not o.get("open") and not o.get("unknown_withdrawal")],
        "unknown_withdrawals": [{k: o.get(k) for k in ("seat_id", "id", "ask_id", "why")} for o in objections if o.get("unknown_withdrawal")],
        "pending_asks": [str(a.get("ask_id") or "") for a in pending_asks(record)],
        "attempts": [{k: a.get(k) for k in ("ts", "revision", "kind", "ask", "not_sendable")} for a in record.get("attempts") or []],
    }


# ------------------------------------------------------------------ review history

RECORD_HISTORY_RULE = (
    "An objection stands until the participant that raised it withdraws it: a revision or the author's "
    "answer does not close it and nobody votes. Each reply is its participant's own words from its "
    "retained source; silence is neither agreement nor an objection."
)


def operative_revision(record: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    """The latest revision as the plan family's current subject (``None`` before the first)."""
    latest = latest_revision(record)
    return {"kind": "plan_revision", **copy.deepcopy(latest)} if latest else None


def record_history(drive_root: Any, task_id: str, record: Mapping[str, Any]) -> Dict[str, Any]:
    """The plan record as review history for the context views (the resident review index,
    compaction, a Continue's predecessor view), read-only: one round per ask whose
    participants carry their exact reply read from its retained source (a selectable body),
    one decision row per objection with its standing now, and a gap for every source that
    cannot be read. A reply with no retained source keeps its typed projection inline."""
    from ouroboros.review_history import _gap, _source_text

    gaps: List[Dict[str, Any]] = []
    seen: Dict[str, Any] = {}
    replies: Dict[tuple, Any] = {}
    rounds = []
    for ask in record.get("asks") or []:
        reviewers = []
        for participant in ask.get("participants") or []:
            seat = {key: copy.deepcopy(participant[key]) for key in (
                "seat_id", "model", "owner_marked", "host_added", "status", "error", "parse_error",
                "not_dispatched_reason") if participant.get(key) not in (None, "")}
            source = participant.get("reply_ref")
            if isinstance(source, dict) and source.get("status") == "retained":
                seat["response"] = _source_text(drive_root, task_id, source, seen, gaps)
                replies[(str(ask.get("ask_id") or ""), str(participant.get("seat_id") or ""))] = source
            elif isinstance(participant.get("reply"), dict):
                seat["reply"] = {key: copy.deepcopy(participant["reply"].get(key)) for key in ("prose", "objections", "withdrawn")}
                gaps.append(_gap("reply has no retained source; its typed projection is shown",
                                 plan_ask=ask.get("ask_id"), seat_id=participant.get("seat_id")))
            reviewers.append(seat)
        rounds.append({"plan_ask": ask.get("ask_id"), "revision": ask.get("revision"), "ts": ask.get("ts"),
                       "message": ask.get("message"), "review_record_id": ask.get("record_id"),
                       "attached": copy.deepcopy(ask.get("attached") or []), "reviewers": reviewers})
    decision_rows = []
    for objection in fold_objections(record):
        if objection.get("unknown_withdrawal"):
            continue
        decision_rows.append({
            "decision_kind": "plan_objection", "plan_ask": objection.get("raised_in"),
            "seat_id": objection.get("seat_id"), "objection_id": objection.get("id"), "remark": objection.get("text"),
            "status": {"open": bool(objection.get("open")), "holds": bool(objection.get("holds")),
                       "owner_marked": bool(objection.get("owner_marked")),
                       "not_adjudicated": bool(objection.get("not_adjudicated")),
                       "last_restated_in": objection.get("ask_id"), "withdrawn_in": objection.get("withdrawn_in") or ""},
            "reason": objection.get("why") or "", "history_role": "open" if objection.get("open") else "historical",
            "source": replies.get((str(objection.get("withdrawn_in") or objection.get("ask_id") or ""),
                                   str(objection.get("seat_id") or ""))),
        })
    from ouroboros.review_history_view import canonical_decision_projection

    # A row with a retained source is a bound decision the view addresses (an actor's note,
    # a group of withdrawn ones); without one it stays full.
    return canonical_decision_projection({"status": "source_unavailable" if gaps else "complete",
                                          "rule": RECORD_HISTORY_RULE, "rounds": rounds,
                                          "decision_rows": decision_rows, "gaps": gaps})


# ------------------------------------------------------------------ durable index

def load_plan_record(root: Any, task_id: str) -> Dict[str, Any]:
    """The task's plan record (empty when the task has none); malformed → ``PlanRecordInvalid``."""
    from ouroboros.task_results import validate_task_id

    document = read_document(root, validate_task_id(task_id))
    if document is None:
        return empty_record()
    return validate_record(document.get(PLAN_RECORD_KEY))


def read_reply_source(root: Any, task_id: str, ask_id: str, seat_id: str, *,
                      role: str, part: str, sha256: str) -> bytes:
    """Exact physical task/ask/seat reply; retry aliases never redirect a saved selection."""
    from ouroboros.review_ledger import ReviewSourceChanged, read_source

    record = load_plan_record(root, task_id)
    sources = [p.get("reply_ref") for ask in record["asks"] if ask["ask_id"] == ask_id
               for p in ask["participants"] if p["seat_id"] == seat_id]
    if role != "response" or part or len(sources) != 1 or not isinstance(sources[0], dict):
        raise ReviewSourceChanged("The selected saved plan reply is unavailable.")
    source = sources[0]
    ref = source.get("ref", source)
    if not isinstance(ref, dict) or ref.get("sha256") != sha256 or source.get("status", "retained") != "retained":
        raise ReviewSourceChanged("The selected saved plan reply changed or is unavailable.")
    return read_source(root, task_id, source)


def _size(record: Mapping[str, Any]) -> int:
    return len(json.dumps(record, ensure_ascii=False, default=str).encode("utf-8"))


def _fit(record: Dict[str, Any]) -> Dict[str, Any]:
    """Move the oldest inline prose out to its exact source while the index is over
    its bound; the latest revision and the newest ask stay whole."""
    if _size(record) <= _PLAN_RECORD_MAX_BYTES:
        return record
    for revision in record.get("revisions", [])[:-1]:
        if revision.get("plan_inline", True) and revision.get("text_ref"):
            revision.update(plan="", plan_inline=False)
            if _size(record) <= _PLAN_RECORD_MAX_BYTES:
                return record
    for ask in record.get("asks", [])[:-1]:
        for participant in ask.get("participants") or []:
            reply = participant.get("reply")
            if isinstance(reply, dict) and reply.get("prose_inline", True) and reply.get("prose"):
                reply.update(prose="", prose_inline=False)
                if _size(record) <= _PLAN_RECORD_MAX_BYTES:
                    return record
    return record


def _update_plan_record(root: Any, task_id: str, mutator: Callable[[Dict[str, Any]], Dict[str, Any]]) -> Dict[str, Any]:
    """Strict transactional update of the task's plan record (no fallback write)."""
    from ouroboros.task_results import STATUS_RUNNING, validate_task_id

    path = f"task:{validate_task_id(task_id)}"

    def _merge(existing: Dict[str, Any]) -> Dict[str, Any]:
        require_writable_task_result_schema(existing, path)
        record = validate_record(_fit(mutator(validate_record(existing.get(PLAN_RECORD_KEY)))))
        now = utc_now_iso()
        return stamp_task_result_schema({
            **existing, PLAN_RECORD_KEY: record, "task_id": task_id,
            "status": str(existing.get("status") or STATUS_RUNNING),
            "ts": str(existing.get("ts") or now), "updated_at": now,
        })

    try:
        updated = patch_document(root, task_id, _merge)
    except StoreError as exc:
        raise PlanRecordInvalid(f"PLAN_RECORD_UNAVAILABLE: {exc}") from exc
    return validate_record(updated.get(PLAN_RECORD_KEY))


def retain_revision(root: Any, task_id: str, fields: Mapping[str, Any], *, n: int, ts: str) -> Dict[str, Any]:
    """The exact text of one revision as a task source (write-once by content)."""
    from ouroboros.artifacts import store_actor_source_bytes

    payload = {"kind": "plan_revision", "task_id": str(task_id), "n": int(n), "ts": ts,
               **{key: fields.get(key) for key in ("goal", "plan", "done_when", "affected_paths", "content_sha256", "body", "body_note")}}
    return store_actor_source_bytes(
        root, task_id, category="context_checkpoints", source_id=f"plan-revision-{int(n):04d}",
        data=json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8"),
        extension="json")


def append_revision(root: Any, task_id: str, fields: Mapping[str, Any], *, body: bool, body_note: str = "") -> tuple[Dict[str, Any], Dict[str, Any], bool]:
    """Record revision N+1 with the given CONTENT (``compose_revision`` output) unless it
    is byte-identical to the latest → ``(record, revision, created)``."""
    current = load_plan_record(root, task_id)
    latest = latest_revision(current)
    if latest is not None and str(latest.get("content_sha256") or "") == str(fields.get("content_sha256") or ""):
        return current, latest, False
    n, ts = (int(latest.get("n") or 0) if latest else 0) + 1, utc_now_iso()
    revision = {"n": n, "ts": ts, "goal": str(fields.get("goal") or ""), "plan": str(fields.get("plan") or ""),
                "plan_inline": True, "done_when": [dict(r) for r in fields.get("done_when") or []],
                "affected_paths": [str(p) for p in fields.get("affected_paths") or []],
                "body": bool(body), "body_note": str(body_note or ""),
                "content_sha256": str(fields.get("content_sha256") or content_sha256(fields))}
    revision["text_ref"] = retain_revision(root, task_id, revision, n=n, ts=ts)

    def _append(record: Dict[str, Any]) -> Dict[str, Any]:
        head = latest_revision(record)
        if head is not None and str(head.get("content_sha256") or "") == revision["content_sha256"]:
            return record  # a concurrent identical write landed first
        record["revisions"] = [*record.get("revisions", []), {**revision, "n": (int(head.get("n") or 0) if head else 0) + 1}]
        return record

    record = _update_plan_record(root, task_id, _append)
    stored = latest_revision(record) or revision
    return record, stored, stored.get("content_sha256") == revision["content_sha256"] and stored.get("ts") == ts


def next_ask_id(record: Mapping[str, Any]) -> str:
    used = {str(a.get("ask_id") or "") for a in record.get("asks") or []}
    n = len(used) + 1
    while f"a{n}" in used:
        n += 1
    return f"a{n}"


def append_attempt(root: Any, task_id: str, attempt: Mapping[str, Any]) -> Dict[str, Any]:
    """Record one ask that could reach nobody (typed, $0): the mind asked and every named
    route was unavailable, so no participant was called. Not an ask — it dispatches nobody,
    raises no objection and holds nothing; the record shows it as asked, with no route."""
    def _append(record: Dict[str, Any]) -> Dict[str, Any]:
        record["attempts"] = [*record.get("attempts", []), {"ts": utc_now_iso(), **copy.deepcopy(dict(attempt))}]
        return record

    return _update_plan_record(root, task_id, _append)


def _ask_panel_outcomes(ask: Dict[str, Any]) -> None:
    from ouroboros.review_ledger import panel_not_reached
    from ouroboros.reviewer_slot_config import review_cost_facts

    rows = ask.get("participants") or []
    for row in rows:
        row.update(review_cost_facts(row.get("cost"), route=str(row.get("route") or ""), model=str(row.get("model") or "")))
    if "panel" in ask:
        ask["panel"]["not_reached"] = panel_not_reached(rows)


def append_ask(root: Any, task_id: str, ask: Mapping[str, Any]) -> Dict[str, Any]:
    """Record one dispatched ask (participants with their typed dispatch state)."""
    def _append(record: Dict[str, Any]) -> Dict[str, Any]:
        if any(str(a.get("ask_id") or "") == str(ask.get("ask_id") or "") for a in record.get("asks", [])):
            raise PlanRecordInvalid(f"PLAN_RECORD_INVALID: ask {ask.get('ask_id')!r} is already recorded")
        record["asks"] = [*record.get("asks", []), copy.deepcopy(dict(ask))]
        _ask_panel_outcomes(record["asks"][-1])
        return record

    return _update_plan_record(root, task_id, _append)


def update_participant(root: Any, task_id: str, *, ask_id: str, seat_id: str, facts: Mapping[str, Any]) -> Dict[str, Any]:
    """Merge settlement facts (status, reply, error, cost, refs…) into one participant row.
    A ``pending`` echo never re-opens a seat already settled: the custody hook settles a
    released seat on its own thread while the dispatcher is still writing the barrier's
    rows, so the check happens HERE, under the record's lock, never against a caller's copy."""
    def _merge(record: Dict[str, Any]) -> Dict[str, Any]:
        for ask in record.get("asks", []):
            if str(ask.get("ask_id") or "") != str(ask_id):
                continue
            for participant in ask.get("participants") or []:
                if str(participant.get("seat_id") or "") == str(seat_id):
                    if facts.get("status") == "pending" and str(participant.get("status") or "pending") != "pending":
                        return record
                    participant.update(copy.deepcopy(dict(facts)))
                    _ask_panel_outcomes(ask)
                    return record
        raise PlanRecordInvalid(f"PLAN_RECORD_INVALID: no participant {seat_id!r} in ask {ask_id!r}")

    return _update_plan_record(root, task_id, _merge)


def claim_ask_completion(root: Any, task_id: str, ask_id: str, custody: Mapping[str, str]) -> Optional[Dict[str, Any]]:
    """Claim this ask's one completion frame under the record lock, using ask-time standing.

    An adviser-only consultation still announces its own completion. A council with
    counted participants announces when those finish, retaining every adviser's row.
    """
    claimed = False
    def claim(record: Dict[str, Any]) -> Dict[str, Any]:
        nonlocal claimed
        claimed = False
        ask = next((a for a in record.get("asks", []) if a.get("ask_id") == ask_id), {})
        participants = ask.get("participants") or []
        counted = [p for p in participants if p.get("owner_marked")] or participants
        if counted and not ask.get("completion_announced") and not any(
                p.get("status") == "pending" and not custody.get(str(p.get("seat_id") or "")) for p in counted):
            ask["completion_announced"] = claimed = True
        return record
    record = _update_plan_record(root, task_id, claim)
    return record if claimed else None


def release_ask_completion(root: Any, task_id: str, ask_id: str) -> None:
    """Release an undelivered frame's claim under the record lock for the next callback."""
    def release(record: Dict[str, Any]) -> Dict[str, Any]:
        for ask in record.get("asks", []):
            if ask.get("ask_id") == ask_id:
                ask.pop("completion_announced", None)
                break
        return record
    _update_plan_record(root, task_id, release)


def collection_pending(value: Any) -> bool:
    """Body-only candidate selection, not a plan verdict or a packing obligation.
    Unknown record/participant shapes stay selectable. No sources or liveness are
    read here: the collector validates the record and checks its task's end."""
    if value is None or value == {}:
        return False
    if not isinstance(value, dict) or value.get("schema_version") != PLAN_RECORD_VERSION:
        return True
    asks = value.get("asks")
    if not isinstance(asks, list):
        return True
    for ask in asks:
        if not isinstance(ask, dict) or not isinstance(ask.get("participants"), list):
            return True
        for participant in ask["participants"]:
            if not isinstance(participant, dict):
                return True
            status = participant.get("status")
            if not isinstance(status, str) or status not in PARTICIPANT_STATUSES:
                return True
    return any(pending_participants(value))


def pending_participants(record: Mapping[str, Any]) -> Iterable[tuple[Dict[str, Any], Dict[str, Any]]]:
    """``(ask, participant)`` pairs still pending, oldest first."""
    for ask in record.get("asks") or []:
        for participant in ask.get("participants") or []:
            if participant.get("status") == "pending":
                yield ask, participant
