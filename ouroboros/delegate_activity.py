"""What a delegated executor said and did, kept typed on the existing progress carrier (#1350).

``delegate_progress`` observes a run by polling its detail, whose ``timeline`` rows carry
typed facts (text kind, delta, severity, actor) but no identity of their own. Flattening
them into one string before any consumer saw them made the executor's sentences sit behind
``Bash · tool_result`` labels, dropped severity, and left history with overlapping window
snapshots that no reader could reconcile. This module keeps the facts typed instead:

* **Source identity.** Each record covers an exact run-journal range, ``after_seq``
  (exclusive) to ``source.read_through`` (inclusive), read from the engine's own event
  stream (``gateways.claudexor_run_events``) whose ``seq`` is the same cursor
  ``delegate_wait`` already advances. Every event in the range belongs to exactly one
  part or to the technical seq runs, or is named as a gap inside the range when it did not
  reach this reader whole; no text joins across such a gap, nor across a keyed piece the
  preview left out (``speech_omitted``, ``_left_out``). A joined part names each
  fragment's seq and Unicode code point offset (``cuts``). Within the disclosed cut/run budgets a reader
  can present each ``(run_id, seq)`` once however two records' ranges overlap; beyond
  them it discloses possibly repeated text/counts with the retained source. Equal
  texts or timestamps are never merged, and a range
  is never counted as if every seq in it were an event.
* **Retained originals.** The redacted lines exactly as served, one LF-ended JSONL record
  each, are stored as a task source handle (``artifacts.store_actor_source_bytes``,
  category ``delegated_activity``, in the canonical custody store) and the record names
  that ref; the task-file route serves it by its content address
  (``gateway.task_archive.serve_task_source``).
* **Commit after emission.** The per-run cursor moves only when the caller reports the
  frame emitted. A process that lost it resumes at the last retained range, re-reading
  that range once; a reader's seq identity keeps it shown once.
* **Honest gaps.** A range a bounded read did not reach is named by seq with its reason.
  A failed read does not advance the cursor: the record shows the bounded timeline window
  as provisional and the next read retries the exact range. At the run's end the wait
  drains the tail within its own read allowance, checking owner controls and remaining
  time before each catalog/stream read; an unread tail stays a final gap with its source.
  An engine without the stream keeps the window view.
* **Voice.** The executor's words stay attributed to the executor; the frame keeps
  ``narration=False``. An executor sentence is not evidence that a claim holds.
* **One utterance per engine key.** A message the engine keys (``speech``: an opaque key,
  compared only within its run and actor) is one part per record, unless an event inside the
  range was not read whole (the pieces on either side stay apart); a piece of only whitespace
  is kept for the join and is no line of its own. Its fragments are a draft
  and its complete text replaces them; against a whole draft this process saw stream it is
  ``updated`` (different) or ``carried`` (equal, its beginning shown whole by earlier frame
  texts, so the frame log adds only its new words); unknown claims neither. An abandon statement
  (``speech_abandoned``) marks it, also in the recalled ``latest_message``. Unkeyed text
  keeps the adjacent-delta rule; the typed ``final`` answer stays its own part, never merged.

The record rides ``progress_meta["delegated_activity"]`` (validated by
``subagent_messages.delegated_activity_meta`` at emission, delivery and replay). The frame
text stays a speech-first plain rendering for text-only readers: Telegram, logs, the
model's recent-progress context and older clients.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import threading
import time
from collections import Counter, OrderedDict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

from ouroboros.utils import truncate_within_limit

log = logging.getLogger(__name__)

ACTIVITY_VERSION = 1
SOURCE_CATEGORY = "delegated_activity"
# Per-part preview bounds. The retained source keeps every original byte; these only
# size what one progress row carries, generous enough that ordinary narration is whole.
_TEXT_CHARS = {"message": 6000, "thinking": 2000, "problem": 1200}
_SPEECH_BUDGET = 24_000
_MAX_PARTS = 40
# Identity inside one record: fragment boundaries of joined parts, and runs of technical
# seqs. Beyond these the record says so (``cuts_truncated`` / ``seqs_truncated``).
_MAX_CUTS = 1000
_MAX_SEQ_RUNS = 500
# The carrier's own bound (``subagent_messages._ACTIVITY_MAX_CHARS``) with headroom.
_RECORD_CHARS = 60_000
_TOP_LABELS = 6
_RECENT_TECHNICAL = 24
_DETAIL_CHARS = 160
_RECAP_CHARS = 700
_LINE_MESSAGE_CHARS = 600
_PROGRESS_TEXT_CHARS = 2000
_RETRY_SEC = 0.5
_OMISSION = "\n⚠️ OMISSION NOTE: truncated at "
# The engine's typed error kinds beyond ``payload.error`` / ``tool.status == "error"``:
# mirrors ERROR_EVENT_TYPES in Claudexor control-api/src/run-timeline.ts (80ef3839).
_ERROR_RUN_EVENTS = frozenset({"run.failed", "reviewer.failed", "reviewer.timed_out"})
_MEMO_LIMIT = 256
_SPEECH_OPS = ("fragment", "complete", "abandoned")
_DRAFT_ACTORS = 32


@dataclass
class _RunMemo:
    """What this process committed to the human for one (task, run)."""

    shown_through: Optional[int] = None  # None: unknown here, resolved from retained sources
    latest_message: Dict[str, Any] = field(default_factory=dict)
    pending: Optional[Dict[str, Any]] = None  # a retained view whose emitter did not commit
    # Per actor, its latest keyed text block: (key, digest of the draft committed so far or
    # None once it ended or is unknown, fragment count, code points of it the frame texts
    # showed whole or None). An actor's blocks are sequential, so a new key of an actor this
    # process already saw began here. ``speech_whole``: every event of the run passed this
    # process (it started at seq 0, lost no event, evicted no actor).
    drafts: "OrderedDict[str, Tuple[str, Any, int, Optional[int]]]" = field(default_factory=OrderedDict)
    speech_whole: Optional[bool] = None
    pending_drafts: Optional["OrderedDict[str, Tuple[str, Any, int, Optional[int]]]"] = None


_MEMO: "OrderedDict[Tuple[str, str], _RunMemo]" = OrderedDict()
_MEMO_LOCK = threading.Lock()


def reset_process_memo() -> None:
    """Forget what this process showed and which engines list the stream (tests reuse ids)."""
    from ouroboros.gateways import claudexor_run_events

    from ouroboros import delegate_observe
    from ouroboros.gateways import claudexor_observation

    with _MEMO_LOCK:
        _MEMO.clear()
    with claudexor_run_events._SUPPORT_LOCK:
        claudexor_run_events._SUPPORT.clear()
    with claudexor_observation._SUPPORT_LOCK:
        claudexor_observation._SUPPORT.clear()
    delegate_observe.reset_process_seen()


def _memo(task_id: str, run_id: str) -> _RunMemo:
    with _MEMO_LOCK:
        memo = _MEMO.pop((task_id, run_id), None) or _RunMemo()
        _MEMO[(task_id, run_id)] = memo
        while len(_MEMO) > _MEMO_LIMIT:
            _MEMO.popitem(last=False)
        return memo


def _text(value: Any, limit: int) -> str:
    return truncate_within_limit(value, limit) if isinstance(value, str) else ""


def _words(part: Dict[str, Any]) -> bool:
    """Whether a part shows anything: a draft piece of only whitespace is carried, never a line of its own."""
    return isinstance(part.get("text"), str) and bool(part["text"].strip())


def _actor(harness: Any, attempt: Any) -> str:
    return "/".join(str(v) for v in (harness, attempt) if isinstance(v, str) and v)[:120]


def _safe_run(run_id: str) -> str:
    # ``store_actor_source_bytes`` normalizes a source id this way; the seq suffix is never stripped.
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(run_id)).lstrip("._")


def _source_id(run_id: str, first: int, last: int) -> str:
    return f"{_safe_run(run_id)}-{first}-{last}"[:160]


# -- classification (typed fields only) ------------------------------------------------

def _typed(item: Dict[str, Any], key: Any, op: Any, final: bool) -> Dict[str, Any]:
    """The engine's speech facts on one item: a message's key and op, the typed final
    answer (never keyed), or an abandon statement on any other event."""
    if not (isinstance(key, str) and key and op in _SPEECH_OPS):
        key = op = None
    if item["kind"] == "message" and final:
        item["final"] = True
    elif item["kind"] == "message" and op in ("fragment", "complete"):
        item.update(key=key, op=op)
    elif item["kind"] not in ("message", "thinking") and op == "abandoned":
        item["abandons"] = key
    return item


def _event_item(seq: int, value: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """One served journal event as speech, problem or technical activity. An event this
    reader cannot read whole is marked ``unreadable``: a line that is not a JSON object, or
    a message whose stated speech names no utterance it can use."""
    if value is None:
        return {"kind": "technical", "seq": seq, "actor": "", "label": "unreadable event line", "detail": "",
                "unreadable": True}
    etype = str(value.get("type") or "event")
    payload = value.get("payload") if isinstance(value.get("payload"), dict) else {}
    tool = payload.get("tool") if isinstance(payload.get("tool"), dict) else {}
    actor = _actor(payload.get("harness_id") or payload.get("harness"),
                   payload.get("attempt_id") or payload.get("attemptId"))
    htype = str(payload.get("type") or "") if etype == "harness.event" else ""
    stated = payload.get("speech") if htype else None
    speech = stated if isinstance(stated, dict) else {}
    final = bool(htype) and payload.get("final") is True
    if htype in ("message", "thinking") and isinstance(payload.get("text"), str):
        nested = payload.get("payload") if isinstance(payload.get("payload"), dict) else {}
        item = {"kind": htype, "seq": seq, "actor": actor, "text": payload["text"], "delta": nested.get("delta") is True}
    elif payload.get("error") or tool.get("status") == "error" or etype in _ERROR_RUN_EVENTS:
        text = next((v for v in (payload.get("error"), tool.get("error_summary"), payload.get("text"),
                                 payload.get("message"), payload.get("title")) if isinstance(v, str) and v), etype)
        item = {"kind": "problem", "seq": seq, "actor": actor, "text": text,
                "label": str(tool.get("name") or htype or etype)[:120]}
    else:
        if htype == "tool_call":
            label, detail = str(tool.get("name") or htype), tool.get("target")
        elif htype == "tool_result":
            label, detail = "tool results", " ".join(str(v) for v in (tool.get("name"), tool.get("status")) if v)
        else:
            label = htype or etype
            detail = next((v for v in (payload.get("title"), payload.get("message"), payload.get("summary"))
                           if isinstance(v, str) and v and v != label), "")
        item = {"kind": "technical", "seq": seq, "actor": actor, "label": label[:80], "detail": _text(detail, _DETAIL_CHARS)}
    item = _typed(item, speech.get("key"), speech.get("op"), final)
    if htype == "message" and stated is not None and not (item.get("key") or item.get("final")):
        item["unreadable"] = True
    return item


def _window_item(row: Dict[str, Any]) -> Dict[str, Any]:
    """One bounded timeline-window row (an engine without the stream): no seq exists."""
    actor = _actor(row.get("harnessId"), row.get("attemptId"))
    title = row.get("title")
    speech = (row.get("speechKey"), row.get("speechOp"))
    if row.get("textKind") in ("message", "thinking") and isinstance(title, str):
        return _typed({"kind": row["textKind"], "actor": actor, "text": title, "delta": row.get("textDelta") is True},
                      *speech, row.get("textFinal") is True)
    label = str(title or row.get("type") or "event")
    if row.get("severity") == "error":
        return _typed({"kind": "problem", "actor": actor, "text": label, "label": str(row.get("type") or "")[:120]},
                      *speech, False)
    return _typed({"kind": "technical", "actor": actor, "label": label[:80], "detail": ""}, *speech, False)


def _speech(items: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Split items into ordered speech/problem parts, technical items and abandon statements.

    Unkeyed text: only the engine's delta fact joins it, and only adjacent fragments of one
    stream (kind + actor); complete texts and every other event stay separate parts, as the
    engine's timeline contract states (``textDelta``). A joined part keeps its first and
    last seq, its fragment count and each later fragment's ``[seq, code_point_offset]``.
    A keyed message is one part per (actor, key), placed at its first item: fragments join
    as above wherever they fall, and its complete text waits beside them for ``_settle``.
    The typed final answer never joins anything. No fragment joins across an event this
    reader did not get whole (``_holes``): it opens a new part, of its key too.
    """
    parts: List[Dict[str, Any]] = []
    technical: List[Dict[str, Any]] = []
    abandoned: List[Dict[str, Any]] = []
    first: Dict[Tuple[str, str], Dict[str, Any]] = {}  # a key's first part in this record
    keyed: Dict[Tuple[str, str], Dict[str, Any]] = {}  # the part its next fragment joins
    stream, lost, previous = None, -1, None  # ``lost``: the latest seq not read whole

    def after_loss(part: Dict[str, Any]) -> bool:
        return lost >= part.get("last_seq", lost + 1)

    for item in items:
        kind, seq = item["kind"], item.get("seq")
        if seq is not None:
            if item.get("unreadable"):
                lost = seq
            elif previous is not None and seq > previous + 1:
                lost = seq - 1
            previous = seq
        if item.get("abandons"):
            abandoned.append({"actor": item["actor"], "key": item["abandons"],
                              **({"seq": item["seq"]} if "seq" in item else {})})
        if kind in ("message", "thinking"):
            if item.get("key"):
                scope = (item["actor"], item["key"])
                part = keyed.get(scope)
                if part is None or (item["op"] == "fragment" and after_loss(part)):
                    part = keyed[scope] = {
                        "kind": kind, "actor": item["actor"], "text": "", "speech": {"key": item["key"], "op": "fragment"}}
                    first.setdefault(scope, part)
                    parts.append(part)
                elif item["op"] == "complete" and part is not first[scope]:
                    # A complete text is whole: it takes its utterance's first place; the pieces after a loss go.
                    parts[:] = [kept for kept in parts if kept is first[scope] or "speech" not in kept
                                or (kept["actor"], kept["speech"]["key"]) != scope]
                    part = keyed[scope] = first[scope]
                if "seq" in item:
                    part.setdefault("seq", item["seq"])
                    part["last_seq"] = item["seq"]
                if item["op"] == "complete":
                    part["complete"] = item["text"]
                else:
                    if part.get("fragments") and "seq" in item:
                        part.setdefault("cuts", []).append([item["seq"], len(part["text"])])
                    part["text"] += item["text"]
                    part["fragments"] = part.get("fragments", 0) + 1
                stream = None
                continue
            key = (kind, item["actor"])
            joins = item["delta"] and not item.get("final")
            last = parts[-1] if parts else None
            if joins and last is not None and stream == key and not after_loss(last):
                if "seq" in item:
                    last.setdefault("cuts", []).append([item["seq"], len(last["text"])])
                    last["last_seq"] = item["seq"]
                last["text"] += item["text"]
                last["fragments"] += 1
            else:
                part = {"kind": kind, "actor": item["actor"], "text": item["text"]}
                if "seq" in item:
                    part.update(seq=item["seq"], last_seq=item["seq"])
                if item.get("final"):
                    part["final"] = True
                elif item["delta"]:
                    part.update(delta=True, fragments=1)
                parts.append(part)
            stream = key if joins else None
            continue
        stream = None
        if kind == "problem":
            parts.append({key: item[key] for key in ("kind", "seq", "actor", "label", "text") if key in item})
        else:
            technical.append(item)
    return parts, technical, abandoned


def _settle(parts: List[Dict[str, Any]], abandoned: List[Dict[str, Any]], memo: _RunMemo, coverage: Optional[str]):
    """What each keyed part shows, and the actors' draft state once this record is committed.

    A complete text replaces its draft. Against the whole draft this process saw stream it
    is ``updated`` when they differ; when they are equal and earlier frame texts showed the
    draft's beginning whole (``_displayed``), ``carried`` counts those code points and the
    frame log prints only what follows. The whole draft is known only where the key began in
    view (a new key of an actor seen before, or the run's start) and every event since passed
    here: ``coverage`` "exact" is a gap-free exact range; "broken" lost one (``_holes``),
    "window" passes the cursor on window rows, and None is a provisional window the exact
    read replaces. Only "exact" proves anything, and any other that moves the cursor leaves
    nothing proven for later records. Unknown is never a difference or an equality.
    """
    exact = coverage == "exact"
    drafts = OrderedDict(memo.drafts if exact else ())
    whole = exact and bool(memo.speech_whole)
    for part in parts:
        speech = part.get("speech")
        if speech is None:
            continue
        complete = part.pop("complete", None)
        actor, prior = part["actor"], drafts.get(part["actor"])
        if prior is not None and prior[0] == speech["key"]:
            digest, earlier, shown = (prior[1].copy() if prior[1] is not None else None), prior[2], prior[3]
        else:
            # An earlier key of this actor in the record proves a later key's draft only in an exact range.
            digest, earlier = (hashlib.sha256() if exact and (prior is not None or whole) else None), 0
            shown = 0 if digest is not None else None
        if digest is not None:
            digest.update(part["text"].encode("utf-8", "surrogatepass"))
        count = earlier + part.get("fragments", 0)
        if complete is not None and digest is not None and count:
            if digest.digest() != hashlib.sha256(complete.encode("utf-8", "surrogatepass")).digest():
                part["updated"] = True
            elif earlier and shown and shown == len(complete) - len(part["text"]):
                part["carried"] = shown
        # Until ``_displayed`` says otherwise, this record's frame shows its piece whole.
        drafts[actor] = (speech["key"], None if complete is not None else digest, count,
                         None if complete is not None or shown is None else shown + len(part["text"]))
        drafts.move_to_end(actor)
        if complete is not None:
            part.pop("cuts", None)
            part["text"], speech["op"] = complete, "complete"
        elif part.get("fragments"):
            part["delta"] = True
    if coverage is None:
        return None
    if not exact:
        return OrderedDict(), False
    for mark in abandoned:
        prior = drafts.get(mark["actor"])
        if prior is not None and prior[0] == mark["key"]:
            drafts[mark["actor"]] = (prior[0], None, prior[2], None)
    while len(drafts) > _DRAFT_ACTORS:
        drafts.popitem(last=False)
        whole = False
    return drafts, whole


def _displayed(drafts: "OrderedDict[str, Tuple[str, Any, int, Optional[int]]]", pieces: List[Tuple[str, str]],
               record: Dict[str, Any]) -> None:
    """Keep a draft's displayed prefix only while its frame texts showed every piece whole:
    kept uncut in the record, within its line's bound, in a frame within its own bound.
    Otherwise the frame log prints its complete text from the start (within those bounds)."""
    fits = len(_frame(record)) <= _PROGRESS_TEXT_CHARS
    shown = {(part["actor"], part["speech"]["key"]) for part in record["parts"]
             if fits and (part.get("speech") or {}).get("op") == "fragment" and not part.get("truncated")
             and len(part["text"]) <= _LINE_MESSAGE_CHARS}
    for actor, key in pieces:
        entry = drafts.get(actor)
        if entry is not None and entry[0] == key and (actor, key) not in shown:
            drafts[actor] = (*entry[:3], None)


def _holes(after: int, through: int, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """The seqs of an exact range that did not reach this reader whole, as gaps inside it:
    ``not_served`` (the stream skipped them) or ``unreadable`` (a line or speech fact). Past
    ``_MAX_PARTS`` runs the rest is one ``not_read`` run, which may span read events too."""
    holes: List[Dict[str, Any]] = []

    def add(first: int, last: int, reason: str) -> None:
        if holes and holes[-1]["through_seq"] == first - 1 and holes[-1]["reason"] == reason:
            holes[-1]["through_seq"] = last
        else:
            holes.append({"after_seq": first - 1, "through_seq": last, "reason": reason})

    previous = after
    for item in [*items, {"seq": through + 1}]:
        if item["seq"] > previous + 1:
            add(previous + 1, item["seq"] - 1, "not_served")
        if item.get("unreadable"):
            add(item["seq"], item["seq"], "unreadable")
        previous = item["seq"]
    if len(holes) > _MAX_PARTS:
        holes[_MAX_PARTS - 1:] = [{"after_seq": holes[_MAX_PARTS - 1]["after_seq"],
                                   "through_seq": holes[-1]["through_seq"], "reason": "not_read"}]
    return holes


def _bounded(parts: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Preview bounds per row, the parts kept and those left out: messages and problems claim
    the budget before thinking.

    A truncated part keeps only the fragment cuts inside its preview prefix; the record's
    cut budget is spent in order and a part it ran out on says ``cuts_truncated``.
    """
    budget, chosen = _SPEECH_BUDGET, set()
    order = sorted(range(len(parts)), key=lambda i: parts[i]["kind"] == "thinking")
    for index in order:
        part = parts[index]
        cost = min(len(part["text"]), _TEXT_CHARS[part["kind"]])
        if len(chosen) >= _MAX_PARTS or cost > budget:
            continue
        budget -= cost
        chosen.add(index)
    kept, left, cuts_left = [], [], _MAX_CUTS
    for index, part in enumerate(parts):
        if index not in chosen:
            left.append(part)
            continue
        limit, chars = _TEXT_CHARS[part["kind"]], len(part["text"])
        part["chars"] = chars
        if chars > limit:
            part.update(text=truncate_within_limit(part["text"], limit), truncated=True)
        if "cuts" in part:
            marker = part["text"].rfind(_OMISSION) if part.get("truncated") else -1
            prefix = marker if marker >= 0 else len(part["text"])
            cuts = [cut for cut in part["cuts"] if cut[1] < prefix]
            if len(cuts) > cuts_left:
                cuts, part["cuts_truncated"] = cuts[:cuts_left], True
            cuts_left -= len(cuts)
            part["cuts"] = cuts
        kept.append(part)
    return kept, left


def _left_out(record: Dict[str, Any], part: Dict[str, Any]) -> None:
    """Count a part the preview leaves out. Its seqs stay read, so a keyed piece is also named
    in ``speech_omitted`` (present whenever a message is left out; ``speech_omitted_truncated``
    past its bound): no reader may join an utterance's words across a piece it was not shown."""
    omitted = record.setdefault("omitted", {})
    omitted[part["kind"]] = omitted.get(part["kind"], 0) + 1
    if part["kind"] != "message":
        return
    named = record.setdefault("speech_omitted", [])
    if part.get("speech") and "seq" in part:
        if len(named) < _MAX_PARTS:
            named.append({"actor": part["actor"], "key": part["speech"]["key"], "seq": part["seq"],
                          "last_seq": part["last_seq"]})
        else:
            record["speech_omitted_truncated"] = True


def _seq_runs(seqs: List[int]) -> List[List[int]]:
    runs: List[List[int]] = []
    for seq in seqs:
        if runs and seq == runs[-1][1] + 1:
            runs[-1][1] = seq
        else:
            runs.append([seq, seq])
    return runs


def _technical(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not items:
        return {}
    counts = Counter(item["label"] for item in items)
    top = counts.most_common(_TOP_LABELS)
    recent = [{key: item[key] for key in ("seq", "actor", "label", "detail") if item.get(key) not in (None, "")}
              for item in items[-_RECENT_TECHNICAL:]]
    out: Dict[str, Any] = {"count": len(items), "labels": [[label, n] for label, n in top], "recent": recent}
    if len(counts) > len(top):
        out["other_kinds"] = len(counts) - len(top)
    if len(items) > len(recent):
        out["recent_omitted"] = len(items) - len(recent)
    seqs = [item["seq"] for item in items if "seq" in item]
    if seqs:
        runs = _seq_runs(seqs)
        out.update({"seqs": runs} if len(runs) <= _MAX_SEQ_RUNS else {"seqs_truncated": True})
    return out


def _fit(record: Dict[str, Any]) -> Dict[str, Any]:
    """Keep the record inside the carrier bound, shedding identity detail before words."""
    def size() -> int:
        return len(json.dumps(record, ensure_ascii=False))

    technical = record.get("technical") or {}
    for shed in (lambda: [part.update(cuts=[], cuts_truncated=True) for part in record["parts"] if part.get("cuts")],
                 lambda: technical.pop("seqs", None) is not None and technical.update(seqs_truncated=True),
                 lambda: technical.update(recent=technical.get("recent", [])[-6:])):
        if size() <= _RECORD_CHARS:
            break
        shed()
    # JSON escaping can exceed the carrier even when code-point previews fit.
    # At most _MAX_PARTS removals; the source remains whole and every removed
    # preview is counted, so the binder never silently drops the entire record.
    while size() > _RECORD_CHARS and record["parts"]:
        _left_out(record, record["parts"].pop())
    return record


# -- the observation record --------------------------------------------------------------

def _persist(ctx: Any, task_id: str, run_id: str, events: List[Any]) -> Dict[str, Any]:
    """Retain the served redacted lines as one task source handle; ``{}`` when it failed."""
    try:
        from ouroboros.artifacts import store_actor_source_bytes, task_id_for_artifacts
        from ouroboros.delegate_custody import custody_root

        if task_id_for_artifacts(ctx) != task_id:
            return {}  # a store not bound to the record's own task publishes no ref
        data = "".join(event.data + "\n" for event in events).encode("utf-8")
        ref = store_actor_source_bytes(
            custody_root(ctx), task_id, category=SOURCE_CATEGORY,
            source_id=_source_id(run_id, events[0].seq, events[-1].seq), data=data, extension="jsonl")
        return {key: ref[key] for key in ("kind", "root", "path", "size", "sha256")}
    except Exception:
        log.warning("delegated activity source for %s could not be retained", run_id, exc_info=True)
        return {}


def _retained_sources(ctx: Any, task_id: str, run_id: str) -> List[Tuple[int, int, Dict[str, Any]]]:
    """Address this run's retained ranges; unrelated source bodies are never read."""
    try:
        from ouroboros.artifacts import task_artifact_dir_path
        from ouroboros.delegate_custody import custody_root

        base = task_artifact_dir_path(custody_root(ctx), task_id)
        directory = base / "source_handles" / SOURCE_CATEGORY
        pattern = re.compile(re.escape(_safe_run(run_id)) + r"-(\d+)-(\d+)-([0-9a-f]{64})\.jsonl")
        ranges = []
        with os.scandir(directory) as entries:
            for entry in entries:
                match = pattern.fullmatch(entry.name)
                if match:
                    ranges.append((int(match[1]), int(match[2]), {
                        "kind": "task_source", "root": "artifact_store",
                        "path": f"source_handles/{SOURCE_CATEGORY}/{entry.name}",
                        "size": entry.stat().st_size, "sha256": match[3]}))
        return sorted(ranges, key=lambda item: (item[0], -item[1]))
    except (OSError, ValueError):
        return []


def _retained_range(ctx: Any, task_id: str, run_id: str) -> Optional[Tuple[int, int]]:
    ranges = _retained_sources(ctx, task_id, run_id)
    return max(ranges, key=lambda item: item[1])[:2] if ranges else None


def _record(task_id: str, run_id: str, after: int, through: int, items: List[Dict[str, Any]],
            source: Dict[str, Any], gaps: List[Dict[str, Any]], memo: _RunMemo) -> Dict[str, Any]:
    parts, technical_items, abandoned = _speech(items)
    holes: List[Dict[str, Any]] = []
    if source.get("kind") != "run_events":
        coverage = None if source.get("provisional") else "window"
    else:
        holes = _holes(after, source["read_through"], items)
        coverage = "broken" if holes else "exact"
    if coverage == "exact" and memo.speech_whole is None:
        memo.speech_whole = after == 0
    drafts = _settle(parts, abandoned, memo, coverage)
    pieces = [(part["actor"], part["speech"]["key"]) for part in parts if (part.get("speech") or {}).get("op") == "fragment"]
    gaps = holes + gaps
    # A draft piece that is only whitespace stays: its seq is read, so a reader joining the utterance across
    # records needs it. Other blank parts carry nothing; no reader shows a blank piece as words.
    parts, left = _bounded([part for part in parts if _words(part) or (part.get("speech") or {}).get("op") == "fragment"])
    record: Dict[str, Any] = {"v": ACTIVITY_VERSION, "task_id": task_id, "run_id": run_id,
                              "after_seq": after, "through_seq": through, "source": source, "parts": parts}
    preview: Dict[str, Any] = {}
    for part in left:
        _left_out(preview, part)
    if abandoned:
        record["speech_abandoned"] = abandoned[:_MAX_PARTS]
        if len(abandoned) > _MAX_PARTS:
            preview.setdefault("omitted", {})["speech_abandoned"] = len(abandoned) - _MAX_PARTS
    record.update(preview)
    for key, value in (("technical", _technical(technical_items)), ("gaps", gaps)):
        if value:
            record[key] = value
    if memo.latest_message and not any(part["kind"] == "message" and _words(part) for part in parts):
        # Text-only readers (the Telegram card) keep the run's latest words, marked earlier.
        record["latest_message"] = _recap(memo.latest_message, abandoned)
    record = _fit(record)
    if drafts is not None:
        _displayed(drafts[0], pieces, record)
    memo.pending, memo.pending_drafts = record, drafts
    return memo.pending


def _recap(message: Dict[str, Any], abandoned: List[Dict[str, Any]]) -> Dict[str, Any]:
    """A message as the run's latest words, keeping its engine facts and an abandon stated since."""
    recap = {key: message[key] for key in ("seq", "actor", "final") if key in message}
    speech = message.get("speech")
    if isinstance(speech, dict):
        gone = any(mark["actor"] == message.get("actor") and mark["key"] == speech.get("key") for mark in abandoned)
        recap["speech"] = {"key": speech.get("key"), "op": "abandoned" if gone else speech.get("op")}
    recap["text"] = truncate_within_limit(message["text"], _RECAP_CHARS)
    return recap


def _commit(memo: _RunMemo, record: Dict[str, Any], shown_through: Optional[int]) -> Callable[[], None]:
    def commit() -> None:
        if shown_through is not None:
            memo.shown_through = max(int(memo.shown_through or 0), shown_through)
        messages = [part for part in record["parts"] if part["kind"] == "message" and _words(part)]
        if messages:
            memo.latest_message = _recap(messages[-1], record.get("speech_abandoned") or [])
        elif record.get("latest_message"):
            memo.latest_message = dict(record["latest_message"])
        if memo.pending is record:
            if memo.pending_drafts is not None:
                memo.drafts, memo.speech_whole = memo.pending_drafts
            memo.pending = memo.pending_drafts = None
    return commit


def shown_cursor(ctx: Any, task_id: str, run_id: str, after_seq: Optional[int]) -> int:
    """Where this process's view of the run's journal ends: what it showed, else the last
    retained range, else the journal's beginning — the same resolution ``observations`` makes."""
    memo = _memo(task_id, run_id)
    if memo.shown_through is None:
        retained = _retained_range(ctx, task_id, run_id)
        memo.shown_through = min(int(after_seq or 0), retained[0] - 1) if retained else 0
    return int(memo.shown_through or 0)


def stream_observations(ctx: Any, run_id: str, read: Any, *, after: int):
    """The typed record of one journal read the caller already made (the supervised wait's
    speech read), with the commit its emitter calls; nothing for a read that carried no event."""
    if not read.events:
        return
    task_id = str(getattr(ctx, "task_id", "") or "")
    memo = _memo(task_id, run_id)
    source: Dict[str, Any] = {"kind": "run_events", "read_through": read.through_seq, "events": len(read.events)}
    if read.ended:
        source["ended"] = True
    ref = _persist(ctx, task_id, run_id, read.events)
    source.update({"ref": ref} if ref else {"retained": False})
    record = _record(task_id, run_id, int(after), int(read.through_seq),
                     [_event_item(event.seq, event.value) for event in read.events], source, [], memo)
    yield record, _commit(memo, record, read.through_seq)


def retained_observations(ctx: Any, run_id: str):
    """Replay unshown local sources; readiness never waits for the engine journal."""
    from ouroboros.artifacts import read_actor_source_bytes
    from ouroboros.delegate_custody import custody_root

    task_id = str(getattr(ctx, "task_id", "") or "")
    memo = _memo(task_id, run_id)
    if memo.pending is not None:
        record = memo.pending
        source = record["source"]
        shown = (None if source.get("provisional") else
                 source.get("read_through", record["through_seq"]))
        yield record, _commit(memo, record, shown)
        return
    after = int(memo.shown_through or 0)
    for first, last, ref in _retained_sources(ctx, task_id, run_id):
        if last <= after or first > after + 1:
            continue  # Missing ranges stay unshown, never an invented cursor advance.
        raw = read_actor_source_bytes(custody_root(ctx), task_id, ref)
        # ``_persist`` ends each served line with LF, the one record separator: U+0085,
        # U+2028 and U+2029, where ``str.splitlines`` also cuts, are text in a JSON line.
        text = raw.decode("utf-8")
        events = [json.loads(line) for line in text.removesuffix("\n").split("\n")] if text else []
        if not events or any(not isinstance(row, dict) or not isinstance(row.get("seq"), int)
                             or row.get("run_id") != run_id for row in events):
            continue
        selected = [row for row in events if after < row["seq"] <= last]
        if not selected:
            continue
        record = _record(task_id, run_id, after, last,
                         [_event_item(row["seq"], row) for row in selected],
                         {"kind": "run_events", "read_through": last, "events": len(selected),
                          "ref": ref}, [], memo)
        yield record, _commit(memo, record, last)
        after = int(memo.shown_through or 0)


def _owner_stop(ctx: Any, task_id: str) -> str:
    """The existing native owner and durable Stop/Panic facts; ``""`` when none."""
    try:
        from ouroboros.cancel_intents import cancel_pending
        from ouroboros.delegate_custody import custody_root
        from ouroboros.model_wait import current_model_wait
        from ouroboros.delegate_recovery import owner_restart_in_progress

        owner = current_model_wait()
        if owner is not None and (reason := owner.control_reason()):
            return reason
        root = custody_root(ctx)
        if owner_restart_in_progress(root):
            return "owner_restart"
        return "cancelled" if cancel_pending(root, task_id) else ""
    except Exception:
        log.debug("owner stop facts unreadable during a delegated drain", exc_info=True)
        return "control_unavailable"


def _read_allowance(ctx: Any, task_id: str, until: float, read_sec: float,
                    read_remaining: Optional[Callable[[], float]]) -> Tuple[float, str]:
    """Narrow to existing operation/task clocks, without minting a positive floor."""
    from ouroboros.deadline_utils import dispatch_window_remaining_sec
    from ouroboros.model_wait import current_model_wait, dispatch_deadline_remaining_sec
    from ouroboros.task_pacing import effective_finalization_reserve_sec

    stopped = _owner_stop(ctx, task_id)
    if stopped:
        return 0.0, f"owner_{stopped}"
    remaining = min(float(read_sec), until - time.monotonic())
    if read_remaining is not None:
        remaining = min(remaining, read_remaining())
    if remaining <= 0:
        return 0.0, "read_bound:time"
    metadata = getattr(ctx, "task_metadata", {}) or {}
    owner = current_model_wait()
    bounds = (dispatch_window_remaining_sec(deadline_at=metadata.get("deadline_at"),
                                           reserve_sec=effective_finalization_reserve_sec(ctx)),
              dispatch_deadline_remaining_sec(),
              owner.execution_window_remaining() if owner is not None else None)
    for bound in bounds:
        if bound is not None:
            remaining = min(remaining, bound)
    return (remaining, "") if remaining > 0 else (0.0, "owner_deadline")


def _gap(after: int, through: int, reason: str, final: bool) -> Dict[str, Any]:
    gap: Dict[str, Any] = {"after_seq": after, "through_seq": through, "reason": reason}
    return {**gap, "final": True, "where": "run_journal"} if final else gap


def observations(ctx: Any, gateway: Any, run_id: str, advance: Any, *, after_seq: int, read_sec: float,
                 drain_sec: Optional[float] = None,
                 read_remaining: Optional[Callable[[], float]] = None) -> Iterator[Tuple[Dict[str, Any], Callable[[], None]]]:
    """The typed records for one wait advance, each with the commit its emitter calls.

    Resumes from what this process committed for the run (else the last retained range,
    else the journal's beginning). The wait's ``after_seq`` is a wake cursor, not evidence
    that its earlier events reached this consumer. One bounded read per advance; with
    ``drain_sec`` (the run ended) the
    reads repeat within that allowance until the journal's end or the bound, retrying a
    failed read. A record is yielded before its cursor moves: an emitter that raises leaves
    the range to be read again. Without the stream, the advance's window rows are the view.
    ``read_remaining`` consults the caller's original operation clock before each request;
    in-flight HTTP socket phases retain the transport's bounds, not total-wall preemption.
    """
    from ouroboros.gateways.claudexor import ClaudexorUnavailable
    from ouroboros.gateways.claudexor_run_events import read_run_events, run_events_supported

    task_id = str(getattr(ctx, "task_id", "") or "")
    memo = _memo(task_id, run_id)
    if memo.shown_through is None:
        retained = _retained_range(ctx, task_id, run_id)
        memo.shown_through = min(int(after_seq or 0), retained[0] - 1) if retained else 0
    through = int(advance.seq)
    final = drain_sec is not None
    until = time.monotonic() + max(0.0, float(drain_sec if final else read_sec))
    rows = [_window_item(row) for row in list(getattr(advance, "events", None) or [])]
    window = {"rows": len(rows), **({"rows_omitted": advance.events_omitted}
                                    if getattr(advance, "events_omitted", 0) else {})}
    after = int(memo.shown_through or 0)
    while True:
        if through <= after:
            return
        read, listed = None, True
        allowance, reason = _read_allowance(ctx, task_id, until, read_sec, read_remaining)
        try:
            if allowance > 0:
                listed = run_events_supported(gateway, timeout_sec=allowance)
                allowance, reason = _read_allowance(ctx, task_id, until, read_sec, read_remaining)
                if listed and allowance > 0:
                    read = read_run_events(gateway, run_id, after_seq=after, timeout_sec=allowance,
                                           through_seq=None if final else through)
        except ClaudexorUnavailable as exc:
            reason = f"stream_unavailable:{exc.code}"
        except Exception:
            log.debug("delegated activity read of %s failed", run_id, exc_info=True)
            reason = "stream_read_failed"
        if not listed:
            # No exact source on this engine at all: the bounded window is the whole view
            # and nothing better will come, so the cursor passes the range.
            record = _record(task_id, run_id, after, through, rows,
                             {"kind": "timeline_window", "reason": "stream_not_listed", **window}, [], memo)
            yield record, _commit(memo, record, through)
            return
        allowance, cut = _read_allowance(ctx, task_id, until, read_sec, read_remaining)
        if read is None:
            if final and allowance > 0:
                time.sleep(min(_RETRY_SEC, allowance))
                allowance, cut = _read_allowance(ctx, task_id, until, read_sec, read_remaining)
                if allowance > 0:
                    continue
            # The exact range stays unread and the cursor stays: this row is the window,
            # marked provisional, and a later exact record covering the range supersedes it.
            if not reason:
                reason = cut
            elif cut.startswith("owner_") and cut != reason:
                reason = f"{reason};{cut}"
            record = _record(task_id, run_id, after, through, rows,
                             {"kind": "timeline_window", "reason": reason, "provisional": True, **window},
                             [_gap(after, through, reason, final)], memo)
            yield record, _commit(memo, record, None)
            return
        through = max(through, read.through_seq)
        caught_up = read.through_seq >= through
        if not caught_up and not read.ended and not read.events and final and allowance > 0:
            time.sleep(min(_RETRY_SEC, allowance))
            continue  # a quiet read of an ended run: nothing to show yet, the allowance remains
        more = not caught_up and not read.ended and final and allowance > 0
        gap_reason = "stream_end_before_fence" if read.ended else cut or f"read_bound:{read.stop}"
        gaps = [] if caught_up or more else [_gap(read.through_seq, through, gap_reason, final)]
        source: Dict[str, Any] = {"kind": "run_events", "read_through": read.through_seq, "events": len(read.events)}
        if read.ended:
            source["ended"] = True
        if read.events:
            ref = _persist(ctx, task_id, run_id, read.events)
            source.update({"ref": ref} if ref else {"retained": False})
        record = _record(task_id, run_id, after, through,
                         [_event_item(event.seq, event.value) for event in read.events], source, gaps, memo)
        yield record, _commit(memo, record, read.through_seq)
        if not more:
            return
        # Resumed only after its emitter committed the record: the drain reads on from here.
        after, rows, window = read.through_seq, [], {"rows": 0}


# -- plain renderings (frame text, Telegram card) ----------------------------------------

def _actor_label(actor: str) -> str:
    return actor.replace("/", " · ") if actor else "executor"


def gap_line(gap: Dict[str, Any], run_id: str) -> str:
    """One unread range, as the frame text and the card state it."""
    span = f"seq {int(gap.get('after_seq', 0)) + 1}–{gap.get('through_seq')}"
    if gap.get("reason") in ("not_served", "unreadable", "not_read"):     # inside a read range (``_holes``)
        return (f"not read whole: {span} ({gap.get('reason')}); an utterance around it may lack words — "
                f"the retained source keeps every line served")
    if gap.get("reason") == "stream_end_before_fence":
        return f"not shown: {span} (stream ended before the observed cursor); coverage unresolved on run {run_id}"
    if gap.get("final"):
        return (f"not shown: {span} ({gap.get('reason')}); the wait returned before reading them — "
                f"they remain on run {run_id}'s journal, and another delegate_wait on the run reads them")
    return f"not shown yet: {span} ({gap.get('reason')}); read at the run's next advance or its end"


def _speech_line(part: Dict[str, Any], abandoned: set, chars: int, log: bool) -> str:
    """One message as plain text, its engine facts as a tag. A text-only reader cannot open
    the card's folded answer, so the typed final answer keeps its words here whatever else
    the record or an earlier one said. In the frame ``log`` a complete text whose equal draft
    earlier frames carried prints only the words after it (a card shows the words whole)."""
    actor, text = part.get("actor", ""), part.get("text")
    speech = part.get("speech") if isinstance(part.get("speech"), dict) else {}
    carried = part.get("carried") if log and speech.get("op") == "complete" and not part.get("updated") else None
    if type(carried) is int and carried > 0 and isinstance(text, str):
        cut = text.rfind(_OMISSION) if part.get("truncated") else -1
        rest = text[min(carried, cut if cut >= 0 else len(text)):]
        return (f"{_actor_label(actor)} (continued, complete): {_text(rest, chars)}" if rest
                else f"{_actor_label(actor)} (complete as streamed above)")
    tag = (" (answer)" if part.get("final") else " (updated)" if part.get("updated") else " (abandoned)"
           if speech.get("op") == "fragment" and (actor, speech.get("key")) in abandoned else "")
    return f"{_actor_label(actor)}{tag}: {_text(text, chars)}"


def activity_lines(activity: Dict[str, Any], *, message_chars: int, recap: bool, log: bool = False) -> List[str]:
    """Speech first, then problems, then counts, then gaps — one plain line each."""
    lines: List[str] = []
    parts = [part for part in activity.get("parts") or [] if isinstance(part, dict)]
    abandoned = {(mark.get("actor", ""), mark.get("key")) for mark in activity.get("speech_abandoned") or []
                 if isinstance(mark, dict)}
    for part in parts:
        if part.get("kind") == "message" and _words(part):
            lines.append(_speech_line(part, abandoned, message_chars, log))
    if recap and not lines and isinstance(activity.get("latest_message"), dict):
        earlier = activity["latest_message"]
        speech = earlier.get("speech") if isinstance(earlier.get("speech"), dict) else {}
        facts = ["earlier", *(["abandoned"] if speech.get("op") == "abandoned" else []),
                 *(["answer"] if earlier.get("final") is True else [])]
        lines.append(f"{_actor_label(earlier.get('actor', ''))} ({', '.join(facts)}): {_text(earlier.get('text'), message_chars)}")
    for part in parts:
        if part.get("kind") == "problem":
            label = " ".join(v for v in (_actor_label(part.get("actor", "")), part.get("label", "")) if v)
            lines.append(f"⚠ {label}: {_text(part.get('text'), 300)}")
    thinking = sum(1 for part in parts if part.get("kind") == "thinking")
    if thinking:
        lines.append(f"thinking ×{thinking} (in the task card's details)")
    technical = activity.get("technical") or {}
    if technical.get("count"):
        kinds = " · ".join(f"{label} ×{n}" for label, n in technical.get("labels") or [])
        lines.append(" · ".join(v for v in (f"{technical['count']} technical events", kinds) if v))
    source = activity.get("source") or {}
    if source.get("provisional"):
        cause = "the exact read failed" if str(source.get("reason", "")).startswith("stream_") else "the exact range was not read"
        lines.append(f"bounded timeline window shown meanwhile ({cause})")
    for gap in activity.get("gaps") or []:
        lines.append(gap_line(gap, str(activity.get("run_id") or "")))
    if source.get("rows_omitted"):
        lines.append(f"+{source['rows_omitted']} earlier timeline rows not shown (bounded window; the run's own timeline has them)")
    return lines


def _frame(activity: Dict[str, Any]) -> str:
    """The frame text before its own bound."""
    head = f"🛰 delegated run {activity['run_id']} @seq {activity['through_seq']}"
    lines = activity_lines(activity, message_chars=_LINE_MESSAGE_CHARS, recap=False, log=True)
    ref = (activity.get("source") or {}).get("ref") or {}
    if ref.get("path"):
        lines.append(f"source: {ref.get('root')}:{ref['path']}")
    return "\n".join([head, *lines]) if lines else f"{head}: (new session events)"


def progress_text(activity: Dict[str, Any]) -> str:
    """The frame text: speech-first, bounded, naming the retained source."""
    return truncate_within_limit(_frame(activity), _PROGRESS_TEXT_CHARS)


def card_text(activity: Dict[str, Any], limit: int) -> str:
    """Speech first; reserve the tail for problems, incomplete coverage and counts."""
    parts = [part for part in activity.get("parts") or [] if isinstance(part, dict)]
    speech = "\n".join(activity_lines({**activity, "parts": [p for p in parts if p.get("kind") == "message"],
                                      "technical": {}, "gaps": [], "source": {}}, message_chars=limit, recap=True))
    details = activity_lines({**activity, "parts": [p for p in parts if p.get("kind") != "message"]},
                             message_chars=limit, recap=False)
    omitted = [(kind, count) for kind, count in (activity.get("omitted") or {}).items() if count]
    if omitted:
        details.append("Not in this card: " + ", ".join(f"{count} {kind}" for kind, count in omitted))
    tail = "\n".join(details)
    if len(tail) > limit // 2:
        # Many diagnostics also need a compact summary; no category disappears
        # behind the speech preview. Telegram appends the full task reference.
        details = []
        problems = [p for p in parts if p.get("kind") == "problem"]
        if problems:
            details.append(f"⚠ {len(problems)} problems; first: {_text(problems[0].get('text'), 120)}")
        thinking = sum(p.get("kind") == "thinking" for p in parts)
        if thinking:
            details.append(f"thinking ×{thinking} (in the task card's details)")
        technical = activity.get("technical") or {}
        if technical.get("count"):
            details.append(f"{technical['count']} technical events")
        gaps, source = activity.get("gaps") or [], activity.get("source") or {}
        if gaps:
            first = gaps[0]
            details.append(f"not shown: {len(gaps)} unread ranges (first seq {int(first.get('after_seq', 0)) + 1}–{first.get('through_seq')})")
        if source.get("provisional"):
            details.append("Incomplete source: exact range not read; bounded window shown")
        if source.get("rows_omitted"):
            details.append(f"{source['rows_omitted']} earlier timeline rows not shown")
        if omitted:
            details.append("Not in this card: " + ", ".join(f"{count} {kind}" for kind, count in omitted))
        tail = "\n".join(details)
    preview = truncate_within_limit(speech, max(0, limit - len(tail) - bool(tail)))
    return "\n".join(part for part in (preview, tail) if part)
