"""Receipt-backed dialogue writes; provider custody stays with the transport."""

from __future__ import annotations

import hashlib
import json
import threading
from pathlib import Path
from typing import Any

from ouroboros.utils import iter_jsonl_objects, jsonl_chain_handles, utc_now_iso

DELIVERY_VERSION = 1
_FIELDS = frozenset({
    "schema_version", "delivery_id", "part_id", "state", "provider", "account_id",
    "conversation_id", "thread_id", "text", "format", "message", "origin",
})
_STATES = frozenset({"delivered", "accepted", "failed", "uncertain"})


class PresenceDeliveryConflict(ValueError):
    """A retained receipt identity already names different provider facts."""


def delivery_reporting_version(value: Any) -> int:
    if type(value) is not int or value not in (0, DELIVERY_VERSION):
        raise ValueError("delivery_reporting_version must be 0 or 1")
    return value


def _canonical(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def validate_delivery(payload: Any) -> dict[str, Any]:
    """Accept facts only, without interpreting provider text or granting power."""
    if not isinstance(payload, dict) or set(payload) != _FIELDS:
        raise ValueError("invalid presence delivery fields")
    if type(payload["schema_version"]) is not int or payload["schema_version"] != DELIVERY_VERSION:
        raise ValueError("presence delivery schema_version must be 1")
    for key in _FIELDS - {"schema_version", "message", "origin"}:
        if not isinstance(payload[key], str):
            raise ValueError(f"presence delivery {key} must be a string")
    for key in ("delivery_id", "provider", "account_id", "conversation_id"):
        if not payload[key].strip():
            raise ValueError(f"presence delivery {key} is required")
    part = payload["part_id"]
    if part != "status" and not (part and part.isascii() and part.isdecimal()):
        raise ValueError("presence delivery part_id must be an index or status")
    if payload["state"] not in _STATES:
        raise ValueError("presence delivery state must be delivered, accepted, failed or uncertain")
    if not isinstance(payload["message"], dict):
        raise ValueError("presence delivery message must be an object")
    origin = payload["origin"]
    if not isinstance(origin, dict) or set(origin) - {"kind", "task_id", "source_event_id"}:
        raise ValueError("invalid presence delivery origin fields")
    if origin.get("kind") not in {"tool", "automatic"}:
        raise ValueError("presence delivery origin.kind must be tool or automatic")
    if any(not isinstance(value, str) for value in origin.values()):
        raise ValueError("presence delivery origin values must be strings")
    # Freeze caller facts before hashing/writing; arrival time is host-owned.
    return json.loads(_canonical(payload))


def _key(skill: str, payload: dict[str, Any]) -> tuple[str, ...]:
    return (skill, *(payload[key] for key in ("account_id", "delivery_id", "part_id", "state")))


def _payload_from_row(row: dict[str, Any]) -> tuple[str, dict[str, Any]] | None:
    if row.get("type") != "presence_delivery":
        return None
    transport = row.get("transport") or {}
    delivery = transport.get("delivery") or {}
    skill = delivery.get("skill")
    if not isinstance(skill, str) or row.get("source") != f"skill:{skill}":
        raise ValueError("presence delivery history has invalid skill provenance")
    payload = {
        "schema_version": delivery.get("schema_version"),
        **{key: delivery.get(key) for key in ("delivery_id", "part_id", "state")},
        **{key: transport.get(key) for key in (
            "provider", "account_id", "conversation_id", "thread_id", "message", "origin",
        )},
        "text": row.get("text"), "format": row.get("format"),
    }
    return skill, validate_delivery(payload)


def _verified_task(data_dir: Path, skill: str, origin: dict[str, str]) -> tuple[str, dict[str, str]]:
    task_id = origin.get("task_id", "")
    if not task_id:
        return "", {}
    from ouroboros.dialogue_provenance import presence_provenance_from_task
    from ouroboros.task_results import load_task_result

    try:
        task = load_task_result(data_dir, task_id, strict=True) or {}
    except (OSError, ValueError):
        return "", {}
    provenance = presence_provenance_from_task(task)
    if provenance.get("transport_skill") != skill:
        return "", {}
    return task_id, provenance


class PresenceDeliveryRecorder:
    """One Host context's disposable projection over canonical chat history.

    Only this Host receipt writer owns these keys. Other chat writers and
    rotation do not invalidate its incrementally maintained projection. A new
    Host rebuilds once; an ambiguous required write discards the projection.

    ``record`` reports ``history_coverage``: ``indexed`` (the last rebuild plus this
    writer's own writes; concurrent chat writers are not re-scanned) or ``gapped``
    (a retained row was malformed, so ``duplicate=false`` is not proof of absence).
    A physically unreadable archive raises rather than reading as empty history.
    Each append starts on a JSONL record boundary, even after a torn tail.
    """

    def __init__(self, data_dir: Path) -> None:
        self.data_dir = Path(data_dir)
        self._lock = threading.Lock()
        self._index: dict[tuple[str, ...], str] | None = None
        self._history_gapped = False

    def _rebuild(self) -> dict[tuple[str, ...], str]:
        index: dict[tuple[str, ...], str] = {}
        gaps: set[str] = set()
        with jsonl_chain_handles(self.data_dir / "logs" / "chat.jsonl", strict=True) as handles:
            for path, handle in handles:
                for row in iter_jsonl_objects(path, _handle=handle, gap_reasons=gaps):
                    try:
                        parsed = _payload_from_row(row)
                    except (TypeError, AttributeError, ValueError):
                        # A malformed retained receipt is also an unobserved
                        # interval, not proof that its identity never existed.
                        gaps.add("invalid_presence_delivery_row")
                        continue
                    if parsed is None:
                        continue
                    skill, payload = parsed
                    key = _key(skill, payload)
                    digest = hashlib.sha256(_canonical(payload)).hexdigest()
                    if key in index and index[key] != digest:
                        raise OSError("conflicting retained presence delivery receipts")
                    index[key] = digest
        self._history_gapped = bool(gaps)
        return index

    def record(self, skill: str, value: Any) -> dict[str, Any]:
        payload = validate_delivery(value)
        key = _key(skill, payload)
        digest = hashlib.sha256(_canonical(payload)).hexdigest()
        with self._lock:
            if self._index is None:
                self._index = self._rebuild()
            previous = self._index.get(key)
            if previous is not None:
                if previous != digest:
                    raise PresenceDeliveryConflict("presence delivery identity already has different facts")
                return {"ok": True, "recorded": True, "duplicate": True,
                        "history_coverage": "gapped" if self._history_gapped else "indexed"}

            from ouroboros.presence_bindings import conversation_key as presence_conversation_key
            from ouroboros.presence_runner import _stable_numeric_id
            from supervisor.message_bus import log_chat

            task_id, provenance = _verified_task(self.data_dir, skill, payload["origin"])
            conversation_key = presence_conversation_key(*(payload[key] for key in (
                "provider", "account_id", "conversation_id", "thread_id",
            )))
            delivery = {
                "schema_version": DELIVERY_VERSION, "skill": skill,
                **{key: payload[key] for key in ("delivery_id", "part_id", "state")},
                "reported_at": utc_now_iso(),
            }
            if provenance:
                delivery["presence_provenance"] = provenance
            transport = {
                **{key: payload[key] for key in (
                    "provider", "account_id", "conversation_id", "thread_id", "message", "origin",
                )},
                "conversation_key": conversation_key, "delivery": delivery,
            }
            try:
                # log_chat owns the append/rotation lock; never acquire it here.
                log_chat(
                    "out" if payload["state"] in {"delivered", "accepted"} else "system",
                    _stable_numeric_id("presence-conversation", conversation_key), 0,
                    payload["text"], fmt=payload["format"], source=f"skill:{skill}",
                    client_message_id="presence-delivery:" + hashlib.sha256(_canonical({"key": key})).hexdigest(),
                    transport=transport, task_id=task_id, record_type="presence_delivery",
                    drive_root=self.data_dir, require_write=True, ensure_record_boundary=True,
                )
            except Exception:
                # The append may have landed before the failure reached us.
                self._index = None
                self._history_gapped = False
                raise
            self._index[key] = digest
            return {"ok": True, "recorded": True, "duplicate": False,
                    "history_coverage": "gapped" if self._history_gapped else "indexed"}


# --- the reflection's read of these rows: one captured chat-chain window --------------------

# Inline bounds of one reflection's section; the retained record holds every value whole.
_SHOWN_RECEIPTS = 40  # newest receipts rendered inline; older ones are counted by state and binding
_SHOWN_TASK_IDS = 12
_ID_CHARS = 96  # one identity value: a stamp, label, task, skill, provider, account, key, id or address
_START_KEYS = ("started_at", "queued_at")  # what a task result records of when its task began
_RECEIPT_SEMANTICS = (
    "Transport skills report what their provider did with each sent part; the host keeps every\n"
    "report as its own chat-history row. Each state is the reporting skill's account of its provider,\n"
    "never checked with the provider by the host: delivered = the provider reported it posted (not that\n"
    "a person read it); accepted = the provider took it for delivery (email: SMTP acceptance, not inbox\n"
    "arrival); failed = refused or not sent; uncertain = it may or may not have arrived. host_bound =\n"
    "the host itself tied the report to this task (a Presence turn of the same transport); skill_claimed\n"
    "= the skill named this task and the host recorded that claim unchecked. Several reports of one\n"
    "part stay separate facts. delivery_receipt_coverage in the completion observations concerns\n"
    "built-in send tools only. A missing report means none was observed in what was read here, never\n"
    "that a message went undelivered; reports written after the capture stay in chat history.\n"
)


def _receipts_of_segment(chain: Any, index: int, wanted: set[str], gaps: set[str]) -> tuple[list, Any]:
    """Matching receipts of one captured segment and its first row's stamp; a failure is a gap."""
    from ouroboros.chat_chain import format_address, row_address
    from ouroboros.deadline_utils import parse_deadline_ts

    try:
        rows, end = chain.rows(index, chain.segment(index)[0], gaps)
    except Exception:  # this generation is unread; rows of the others still count
        gaps.add("unreadable_source")
        return [], None
    if chain.entries[index][2] and end < chain.segment(index)[1]:
        gaps.add("incomplete_live_line")
    found = []
    for _offset, row in rows:
        if row.get("type") != "presence_delivery":
            continue
        try:
            skill, payload = _payload_from_row(row)
            transport = row["transport"]
        except (TypeError, AttributeError, KeyError, ValueError):
            gaps.add("invalid_presence_delivery_row")  # whose it was is unknown
            continue
        # A nonempty row task decides alone; only an empty one falls back to the skill's claim.
        bound = str(row.get("task_id") or "")
        task_id, binding = (bound, "host_bound") if bound else (payload["origin"].get("task_id", ""), "skill_claimed")
        if task_id in wanted:
            found.append({
                "binding": binding, "task_id": task_id, "skill": skill,
                **{key: payload[key] for key in (
                    "state", "provider", "account_id", "delivery_id", "part_id", "text", "message")},
                "conversation_key": str(transport.get("conversation_key") or ""),
                "origin_kind": payload["origin"]["kind"],
                "reported_at": str((transport.get("delivery") or {}).get("reported_at") or row.get("ts") or ""),
                "address": format_address(row_address(row)), "row": row,
            })
    return found, (parse_deadline_ts(rows[0][1].get("ts")) if rows else None)


def _receipt_lineage(data_dir: Path, task: dict[str, Any]) -> tuple[set[str], list, bool]:
    """``(task ids, recorded starts, walked)`` of the tasks whose receipts are ``task``'s.

    Its own walk of the canonical task results, made only after reflection admission and
    apart from the child evidence that decides admission: this task and its logical root
    (a retry keeps its first attempt), every row naming either as parent or root, and the
    start or queue stamps they persisted beside the input task's own. A failed walk keeps
    the ids and stamps found before it and reports ``walked`` false.
    """
    from ouroboros.task_results import list_task_results

    task_id = str(task.get("id") or "")
    roots = {task_id, str(task.get("root_task_id") or task_id)} - {""}
    ids, starts = set(roots), [task[key] for key in _START_KEYS if task.get(key)]
    try:
        for item in list_task_results(Path(data_dir)):
            if not isinstance(item, dict):
                continue
            item_id = str(item.get("task_id") or item.get("id") or "")
            if roots & {item_id, str(item.get("parent_task_id") or ""), str(item.get("root_task_id") or "")}:
                ids.add(item_id)
                starts += [item[key] for key in _START_KEYS if item.get(key)]
    except Exception:
        return ids, starts, False
    return ids, starts, True


def task_delivery_receipts(data_dir: Path, task: dict[str, Any]) -> dict[str, Any]:
    """Every retained receipt row bound to ``task``'s lineage inside one captured chat-chain window.

    Binding is exact: a nonempty row ``task_id`` (``host_bound``) decides alone, and
    only an empty one falls back to ``transport.origin.task_id`` (``skill_claimed``).
    Every row stays its own fact: no latest-wins or delivered-never-downgrades fold.

    One ``JsonlChainSnapshot`` capture bounds the read; ``captured_at`` is stamped
    right after it, apart from any row stamp, and a row appended later is outside
    it. Segments are read newest first until one whose first row is stamped before
    the earliest parseable recorded start; that whole segment and every archive
    rotated in its same second are included (an unparsable stamp only widens the
    window). Without a start the window stops at the newest nonempty segment,
    even when its rows are malformed or unstamped. Older archives left unread make the
    result nonexhaustive however clean the window was, and a row reordered across
    that boundary is the accepted residual. Gaps keep the rows already found; only
    a capture that fails outright is ``unavailable``.
    """
    from ouroboros.chat_chain import _ROTATED_AT
    from ouroboros.deadline_utils import parse_deadline_ts
    from ouroboros.jsonl_tail import JsonlChainSnapshot

    def rotated_second(index: int) -> str:  # the archive name's stamp text; "" for the live file
        match = _ROTATED_AT.match(Path(chain.entries[index][0]).name)
        return match[1] if match else ""

    task_ids, starts, walked = _receipt_lineage(data_dir, task)
    wanted = {str(task_id) for task_id in task_ids if str(task_id or "").strip()}
    stamps = sorted((parsed, str(raw)) for raw in starts if (parsed := parse_deadline_ts(raw)) is not None)
    anchor = stamps[0] if stamps else None
    evidence: dict[str, Any] = {"task_ids": sorted(wanted), "lineage_unavailable": not walked,
                                "anchor": anchor[1] if anchor else "", "captured_at": "", "segments": 0,
                                "segments_read": 0, "older_unread": 0, "rows": [], "gaps": []}
    gaps: set[str] = set()
    path = Path(data_dir) / "logs" / "chat.jsonl"
    try:
        try:
            chain = JsonlChainSnapshot(path)
        except OSError:
            # One unreadable generation must not hide the readable ones: capture what
            # can be listed and opened, and name the degraded capture.
            chain = JsonlChainSnapshot(path, strict=False)
            gaps.add("unreadable_source")
    except Exception as exc:
        return {**evidence, "read_coverage": "unavailable", "gaps": [type(exc).__name__]}
    evidence["captured_at"] = utc_now_iso()
    count = evidence["segments"] = len(chain.entries)
    found: dict[int, list] = {}
    index = count - 1
    try:
        while index >= 0:
            found[index], first = _receipts_of_segment(chain, index, wanted, gaps)
            if anchor is None and chain.segment(index)[1] > chain.segment(index)[0]:
                break
            if anchor is not None and first is not None and first < anchor[0]:
                break
            index -= 1
        index = max(index, 0)
        second = rotated_second(index) if count else ""
        while second and index > 0 and rotated_second(index - 1) == second:
            index -= 1
            found[index] = _receipts_of_segment(chain, index, wanted, gaps)[0]
    except Exception:  # an unexpected failure keeps every receipt already found
        gaps.add("read_failed")
        index = min(found, default=count)
    evidence.update(rows=[row for key in sorted(found) for row in found[key]], segments_read=len(found),
                    older_unread=index, gaps=sorted(gaps), read_coverage="gapped" if gaps else "complete")
    return evidence


def _clip(value: Any, limit: int = _ID_CHARS) -> str:
    text = " ".join(str(value).split())
    return text if len(text) <= limit else text[:limit] + f"…[+{len(text) - limit} chars]"


_ID_FIELDS = ("reported_at", "binding", "task_id", "skill", "provider", "account_id", "conversation_key",
              "delivery_id", "part_id", "state", "origin_kind", "address")
_LINE = ("- #{n} {reported_at} {binding} task {task_id}: {skill} {provider}/{account_id} {conversation_key} "
         "delivery {delivery_id} part {part_id} {state} (origin {origin_kind}); text \"{text}\"; message {message}; "
         "{address}")


def _tally(rows: list) -> str:
    """Every binding and state counted, sorted: a count, never a fold of one part's reports."""
    tally: dict[str, int] = {}
    for row in rows:
        for key in (row["binding"], row["state"]):
            tally[key] = tally.get(key, 0) + 1
    return ", ".join(f"{key} {tally[key]}" for key in sorted(tally))


def _record_line(context: Any, evidence: dict[str, Any], rows: list) -> tuple[str, str]:
    """The prompt line naming the retained whole selection, and the gap when retention failed.

    Line 1 of the record is this read's coverage with every task id; line ``n + 1`` is
    report ``#n``: its binding, task id and address beside the chat row exactly as read.
    """
    from ouroboros import chat_chain

    head = {key: evidence.get(key) for key in ("captured_at", "task_ids", "lineage_unavailable", "anchor",
                                                "segments", "segments_read", "older_unread", "read_coverage", "gaps")}
    lines = [{"record": "transport_delivery_reports", "reports": len(rows), **head}, *(
        {"n": n, "binding": row["binding"], "task_id": row["task_id"], "address": row["address"],
         "chat_row": row.get("row")} for n, row in enumerate(rows, 1))]
    text = "".join(json.dumps(line, ensure_ascii=False, sort_keys=True) + "\n" for line in lines)
    try:
        ref = chat_chain.retain_memory_source(context, "task_delivery_receipts", text.encode("utf-8"), "jsonl")
    except Exception as exc:  # disclosed beside the reports already found, never in place of them
        return (f"Complete record unavailable ({type(exc).__name__}): the values below are bounded previews and "
                "a report not shown is only counted; every report stays in chat history.\n",
                "complete_record_unavailable")
    return (f"Complete record ({len(text)} chars, optional reading; line 1 is this read's coverage with every task "
            "id, line n+1 is report #n whole as its chat row): read_file "
            + json.dumps(ref["read"]["arguments"], ensure_ascii=False) + "\n", "")


def receipts_prompt_section(evidence: dict[str, Any] | None, context: Any) -> str:
    """Bounded text of ``task_delivery_receipts`` for one reflection prompt.

    Every inline value is clipped with its omission named, and only the newest
    ``_SHOWN_RECEIPTS`` reports are lines; the older ones are counted by binding and
    state, so a failed or uncertain report is never cut silently. The whole selection
    is retained once through ``chat_chain.retain_memory_source`` under ``context`` and
    named with the ``read_file`` arguments the reflection's own reader accepts: a chat
    row address needs a reader it lacks, and rotation moves the row. A retention that
    fails is the gap ``complete_record_unavailable`` and keeps every report shown.
    """
    if not isinstance(evidence, dict):
        return ""
    heading = "## Transport delivery reports (read for this reflection)\n"
    if evidence.get("read_coverage") == "unavailable":
        return (heading + "Unavailable: the chat history could not be captured ("
                + ", ".join(evidence.get("gaps") or ["unknown"]) + "); delivery outcomes stay unknown, not failed.\n\n")
    rows = list(evidence.get("rows") or [])
    record, unretained = _record_line(context, evidence, rows)
    gaps = [*(evidence.get("gaps") or []), *([unretained] if unretained else [])]
    ids = list(evidence.get("task_ids") or [])
    named = ", ".join(_clip(task_id) for task_id in ids[:_SHOWN_TASK_IDS]) + (
        f" and {len(ids) - _SHOWN_TASK_IDS} more" if len(ids) > _SHOWN_TASK_IDS else "")
    lineage = (" (the task-results walk failed: this task, its root and only the ids found before the failure)"
               if evidence.get("lineage_unavailable")
               else " (this task, its root attempts and the descendants found in readable task results)")
    window = f"{evidence['segments_read']} of {evidence['segments']} chat generations read newest first " + (
        f"back to the one holding the earliest recorded start {_clip(evidence['anchor'])}" if evidence.get("anchor")
        else "back to the newest nonempty one: a limited window, no start recorded for these tasks")
    read = (f"Read: chat history captured at {_clip(evidence['captured_at'])}; tasks {named}{lineage}; {window}; "
            f"coverage {'gapped' if gaps else evidence['read_coverage']}" + (f" ({', '.join(gaps)})" if gaps else ""))
    if evidence.get("older_unread"):
        read += (f"; NONEXHAUSTIVE: {evidence['older_unread']} older archived generation(s) not read, "
                 "so an earlier report may exist there")
    if not rows:
        return (heading + "No transport delivery report bound to these tasks was observed in what was read; "
                "that is not evidence of non-delivery.\n" + read + ".\n" + record + "\n")
    older = rows[:-_SHOWN_RECEIPTS] if len(rows) > _SHOWN_RECEIPTS else []
    lines = [_LINE.format(n=n, **{key: _clip(row[key]) for key in _ID_FIELDS}, text=_clip(row["text"], 160),
                          message=_clip(json.dumps(row["message"], ensure_ascii=False, sort_keys=True), 200))
             for n, row in enumerate(rows[len(older):], len(older) + 1)]
    shown = "all shown in chain order:\n"
    if older:
        shown = (f"newest {len(lines)} shown in chain order (#{len(older) + 1}–#{len(rows)}), {len(older)} older "
                 f"not shown (#1–#{len(older)}: {_tally(older)}; "
                 + ("only counted here" if unretained else "each whole in the record") + "):\n")
    return (heading + _RECEIPT_SEMANTICS + read + ".\n" + record
            + f"Reports: {len(rows)} ({_tally(rows)}); " + shown + "\n".join(lines) + "\n\n")
