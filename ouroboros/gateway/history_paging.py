"""A room's own pages over the existing retained chat/progress JSONL chains.

A page is counted in the room's rows, never in bytes of the shared chain: the
newest page holds the room's newest rows however far back they lie, and each
older page the next older ones (owner decisions 2026-10-05, DESIGN "History
edges"). The chat and progress streams keep separate quotas (``human``,
``progress``), each counted by its own predicate. Archives a Project's lens rules out are skipped unread
(``history_segments``); only a physical read ceiling can end a page early.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
from bisect import bisect_left
from pathlib import Path

from ouroboros.contracts.chat_id_policy import is_a2a_chat_id
from ouroboros.gateway import _helpers
from ouroboros.gateway._helpers import _TAIL_WINDOW_START_BYTES
from ouroboros.gateway.history_segments import segment_summary
from ouroboros.jsonl_tail import JsonlChainSnapshot
from ouroboros.subagent_messages import CARD_ROW_PLACEMENTS, is_task_card_message, subagent_message_meta

_SOURCES = ("chat", "progress")
_READ_BYTES = 64 * 1024
# Physical bound on the bytes one request parses per stream; the cursor resumes
# where a read stopped, so it bounds latency, never what a room can reach.
_READ_CEILING_BYTES = 16 * 1024 * 1024
# The newest-arrival search every recent Project read pays for; a bound reached
# first is carried on by the chain's cursors (``latest_before``/``quiet``).
_ARRIVAL_SEARCH_BYTES = 512 * 1024
_CHAIN_WITNESSES = 16
# A projection that failed part-way: the rows after the failing one are missing
# from the response, so neither its coverage nor its newest arrival is known.
PROJECTION_FAILED = "projection_failed"
# Stored kinds chat.js shows as a bubble of their own whoever sent them: it
# renders them before a child's words are routed to its card, and their
# producers (message_bus ``send_photo`` ...) count every one.
_STANDALONE_KINDS = ("document", "links", "photo", "video", "quiz")


def progress_quota_predicate(row_matches_thread, stored_chat_id):
    def counts(entry):
        if not isinstance(entry, dict) or entry.get("type") in {"review_reference", "task_model_wait"}:
            return False
        if is_a2a_chat_id(entry.get("chat_id", 1)):
            return False
        return bool(
            row_matches_thread(stored_chat_id(entry.get("chat_id"), 1), {"is_progress": True, **entry})
            and str(entry.get("content", entry.get("text", "")))
            and str(entry.get("delegation_role") or "").lower() != "subagent"
            and not entry.get("subagent_event")
        )
    return counts


class HistoryCursorError(ValueError):
    def __init__(self, reason: str, status: int = 409):
        super().__init__(reason)
        self.reason, self.status = reason, status


def room_view_fingerprint(thread_id, project_ids, source_refs, bindings):
    """Fingerprint the existing room lens, without introducing membership state."""
    project = thread_id in project_ids
    relevant = {key: value for key, value in bindings.items()
                if not project or value == thread_id}
    value = [thread_id, project, [] if project else sorted(project_ids), relevant,
             sorted(source_refs, key=lambda row: json.dumps(row, sort_keys=True))]
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def encode_cursor(value):
    return base64.urlsafe_b64encode(json.dumps(value, separators=(",", ":")).encode()).decode().rstrip("=")


def decode_cursor(value, thread_id, view):
    try:
        if not isinstance(value, str) or len(value) > 4096:
            raise ValueError
        decoded = base64.b64decode(value + "=" * (-len(value) % 4), altchars=b"-_", validate=True)
        cursor = json.loads(decoded)
        if not isinstance(cursor, dict) or cursor.get("v") != 1 or cursor.get("kind") not in {"older", "page"}:
            raise ValueError
        for field in ("upper", "before"):
            if set(cursor[field]) != set(_SOURCES):
                raise ValueError
            if any(type(cursor[field][source]) is not int or cursor[field][source] < 0 for source in _SOURCES):
                raise ValueError
        if any(cursor["before"][source] > cursor["upper"][source] for source in _SOURCES):
            raise ValueError
        unfinished = cursor.setdefault("unfinished", [])
        if not isinstance(unfinished, list) or unfinished != sorted(set(unfinished) & set(_SOURCES)):
            raise ValueError
        if type(cursor.setdefault("quiet", False)) is not bool:  # see ``latest_arrival``
            raise ValueError
        if set(cursor["quotas"]) != {"human", "progress"} or any(
            type(value) is not int or value < 0 for value in cursor["quotas"].values()
        ):
            raise ValueError
        if cursor["kind"] == "page":
            if type(cursor.get("recent")) is not bool or set(cursor["lower"]) != set(_SOURCES):
                raise ValueError
            if any(type(cursor["lower"][source]) is not int
                   or not 0 <= cursor["lower"][source] <= cursor["before"][source] for source in _SOURCES):
                raise ValueError
    except (ValueError, TypeError, KeyError):
        raise HistoryCursorError("history_cursor_invalid", 400) from None
    if cursor.get("chat_id") != thread_id or cursor.get("view") != view:
        raise HistoryCursorError("history_view_changed")
    return cursor


class HistorySource(JsonlChainSnapshot):
    """One metadata snapshot, with no handles retained across a read batch."""

    def __init__(self, path: Path, source: str, upper=None, unfinished=False):
        super().__init__(path, upper=upper)
        self.source = source
        # Bytes after the frozen boundary that a writer had not finished when it
        # was frozen (a cursor carries the fact with ``upper``): every read that
        # reaches the boundary discloses them (``incomplete_live_line``), a clean EOF has none.
        self.unfinished_live_line = bool(unfinished)
        if upper is None and self.snapshot["entries"] and self.snapshot["entries"][-1][2]:
            base, end = self.segment(len(self.entries) - 1)
            # A writer owns an unfinished live line. Freeze only complete rows,
            # so completing/rotating that line later cannot alter this page.
            eof = end
            while end > base and self._read(end - 1, end) != b"\n":
                start = max(base, end - _READ_BYTES)
                data = self._read(start, end)
                newline = data.rfind(b"\n")
                end = start + newline + 1 if newline >= 0 else start
            self.upper, self.unfinished_live_line = end, end < eof

    def _boundary_gaps(self, end):
        return {"incomplete_live_line"} if self.unfinished_live_line and end == self.upper else set()

    def _entries(self, start, end, gaps):
        data = self._read(start, end)
        position = start
        stream = self.source
        is_live_end = bool(self.snapshot["entries"] and self.snapshot["entries"][-1][2]
                           and end == self.upper)

        class Lines:
            row_start = row_end = start

            def __iter__(self):
                nonlocal position
                for raw in io.BytesIO(data):
                    self.row_start, self.row_end = position, position + len(raw)
                    position = self.row_end
                    if not raw.endswith(b"\n") and is_live_end:
                        gaps.add("incomplete_live_line")
                        continue
                    yield raw

        lines = Lines()
        rows = []
        for entry in _helpers.iter_jsonl_objects(self.path, _handle=lines, gap_reasons=gaps):
            rows.append({**entry, "history_id": f"{stream}:{lines.row_start}",
                         "history_position": {"source": stream, "offset": lines.row_start},
                         "_history_end": lines.row_end})
        return rows

    def _aligned_start(self, start, end, base):
        if start <= base or self._read(start - 1, start) == b"\n":
            return start
        data = self._read(start, end)
        newline = data.find(b"\n")
        return end if newline < 0 else start + newline + 1

    def _ruled_out(self, index, lens):
        """A closed archive whose summary proves it holds none of the room's rows."""
        path, stat, live = self.snapshot["entries"][index]
        if lens is None or live:
            return False
        summary = segment_summary(path, stat)
        return summary is not None and not lens(summary)

    def recent(self, want, counts, lens=None):
        """The room's newest rows: the live tail, then whole archives newest-first
        until ``want`` counted rows, the chain's start or the read ceiling."""
        entries, before, counted, read = [], self.upper, 0, 0
        gaps = self._boundary_gaps(self.upper)
        for index in reversed(range(len(self.snapshot["entries"]))):
            base, end = self.segment(index)
            if base >= self.upper:
                continue
            live = self.snapshot["entries"][index][2]
            if not live and (counted >= want or read >= _READ_CEILING_BYTES):
                break  # the live tail is always read: its overlays need no quota
            if self._ruled_out(index, lens):
                before = base
                continue
            window = _TAIL_WINDOW_START_BYTES if live else end - base
            while True:
                start = self._aligned_start(max(base, end - window), end, base)
                selected = self._entries(start, end, gaps)
                found = sum(map(counts, selected))
                if start == base or counted + found >= want:
                    break
                window *= 2
            counted, read = counted + found, read + end - start
            entries = selected + entries
            before = start
        return entries, before, gaps

    def older(self, before, want, counts, lens=None, ceiling=None):
        """Consume one backward page of ``want`` counted rows; foreign/invalid bytes
        and ruled-out archives advance the position, the read ceiling ends it early."""
        if want <= 0:
            return [], before, set()
        ceiling = _READ_CEILING_BYTES if ceiling is None else ceiling
        selected, gaps, counted, read = [], self._boundary_gaps(before), 0, 0
        while before > 0 and read < ceiling:
            index = bisect_left(self.snapshot["ends"], before)
            base, _end = self.segment(index)
            if self._ruled_out(index, lens):
                before = base
                continue
            window = _READ_BYTES
            while True:
                start = self._aligned_start(max(base, before - window), before, base)
                if start < before or start == base:
                    break
                window *= 2  # one large JSONL row keeps its existing support
            entries = self._entries(start, before, gaps)
            for entry in reversed(entries):
                selected.append(entry)
                read += before - entry["history_position"]["offset"]
                before = entry["history_position"]["offset"]
                counted += bool(counts(entry))
                if counted >= want or read >= ceiling:
                    return list(reversed(selected)), before, gaps
            read += before - start
            before = start
        return list(reversed(selected)), before, gaps

    def replay(self, lower, before, lens=None):
        """Re-read one frozen page; the archives its first read ruled out stay unread
        (same room view, immutable archives), so a replay costs what that read did."""
        gaps, rows = self._boundary_gaps(before), []
        for index in range(len(self.snapshot["entries"])):
            base, end = self.segment(index)
            start, end = max(base, lower), min(end, before)
            if start < end and not self._ruled_out(index, lens):
                rows.extend(self._entries(start, end, gaps))
        return rows, lower, gaps


def chain_witness(reader):
    """Rolling prefix witnesses through the trailing retained segments below ``upper``.

    Physical byte coordinates survive append and rotation: the live file keeps
    its inode and base when renamed into the archive. Every witness rolls each
    earlier nonempty segment's identity and base, so replacing, removing or
    resizing ANY earlier segment changes it. A span stays comparable while its
    own last witness is still listed by a newer read, i.e. within
    ``_CHAIN_WITNESSES`` rotations; older spans are disclosed as gaps. Metadata
    only: no archive is read to establish it. Empty sources have no prefix.
    """
    witnesses, digest = [], b""
    for index, (_, stat, _) in enumerate(reader.entries):
        base = reader.ends[index - 1] if index else 0
        if base >= reader.upper:
            break
        if stat.st_size:
            digest = hashlib.sha256(digest + f"{stat.st_dev}:{stat.st_ino}@{base}".encode()).digest()
            witnesses.append(digest.hex()[:16])
    return ".".join(witnesses[-_CHAIN_WITNESSES:]) or "empty"


def select_history_page(data_dir, thread_id, view, cursor, quotas, predicates, caps, lens=None):
    """``lens``: the room's archive filter (``history_segments.room_segment_lens``)."""
    continuation = decode_cursor(cursor, thread_id, view) if cursor else None
    if continuation:
        quotas = continuation["quotas"]
        if any(quotas[key] > caps[key] for key in quotas):
            raise HistoryCursorError("history_cursor_invalid", 400)
    recent = not continuation or (continuation["kind"] == "page" and continuation["recent"])
    selections, upper, before, page_ends, chains, unfinished = {}, {}, {}, {}, {}, []
    paths = {"chat": data_dir / "logs" / "chat.jsonl", "progress": data_dir / "logs" / "progress.jsonl"}
    for source, quota in (("chat", "human"), ("progress", "progress")):
        try:
            reader = HistorySource(paths[source], source,
                                   continuation["upper"][source] if continuation else None,
                                   bool(continuation) and source in continuation["unfinished"])
            upper[source] = reader.upper
            chains[source] = chain_witness(reader)
            if reader.unfinished_live_line:
                unfinished.append(source)
            page_ends[source] = continuation["before"][source] if continuation else reader.upper
            if continuation and continuation["kind"] == "page":
                selections[source] = reader.replay(continuation["lower"][source], page_ends[source], lens)
            elif continuation:
                selections[source] = reader.older(page_ends[source], quotas[quota], predicates[source], lens)
            else:
                selections[source] = reader.recent(quotas[quota], predicates[source], lens)
        except OSError:
            if continuation:
                raise
            # The existing recent collector can still show readable rows. An
            # unknown chain prefix cannot establish global offsets or EOF.
            selections[source] = (None, 0, {"source_unavailable"})
        before[source] = selections[source][1]
    return {"v": 1, "chat_id": thread_id, "view": view, "upper": upper, "unfinished": unfinished,
            "quotas": quotas, "recent": recent, "replayed": bool(continuation),
            "quiet": bool(continuation and continuation["quiet"]), "lens": lens,
            "selections": selections, "before": before, "page_ends": page_ends, "chains": chains}


def history_page_coverage(page, stream_gaps):
    """Delivered physical spans, called AFTER quota/lineage deferrals.

    A projected origin, detail overlay or shared row ID proves no scanned span.
    Quota-disabled and unreadable streams are unknown, even when they emit []
    and no continuation. Parse gaps remain part of the coverage evidence; the
    unfinished live line a frozen boundary discloses lies after ``upper``, so
    it is no gap in a span that ends there.
    """
    return {"v": 1, "view": page["view"], "upper": dict(page["upper"]), "spans": {
        source: ({"from": page["before"][source], "to": page["page_ends"][source],
                  "chain": page["chains"][source], "gaps": sorted(stream_gaps[source] - (
                      {"incomplete_live_line"} if source in page["unfinished"] else set()))}
                 if page["selections"][source][0] is not None
                 and page["quotas"]["human" if source == "chat" else source] else None)
        for source in _SOURCES}}


def history_page_tokens(page):
    """The conversation decides whether older history remains: narration (progress)
    keeps paging alongside it but never keeps a press alive on its own, so no press
    lands narration alone. A narration-only read (no human quota) pages narration."""
    if any(selection[0] is None for selection in page["selections"].values()):
        return {"has_more": True, "next_cursor": None, "page_cursor": None,
                "reason_code": "history_source_unavailable"}
    state = {key: page[key] for key in ("v", "chat_id", "view", "upper", "unfinished", "quotas")}
    before = {source: position if page["quotas"]["human" if source == "chat" else source] else 0
              for source, position in page["before"].items()}
    has_more = bool(before["chat"]) if page["quotas"]["human"] else any(before.values())
    return {
        "has_more": has_more,
        "next_cursor": encode_cursor({**state, "kind": "older", "before": before,
                                      "quiet": page.get("quiet_below", False)}) if has_more else None,
        "page_cursor": encode_cursor({**state, "kind": "page", "before": page["page_ends"], "quiet": page["quiet"],
                                      "lower": {source: value[1] for source, value in page["selections"].items()},
                                      "recent": page["recent"]}),
    }


def projected_history_ids(messages):
    """Folded attempts still account for the exact physical rows they contain."""
    ids = set()
    for message in messages:
        if message.get("history_id"):
            ids.add(message["history_id"])
        group = message.get("review_group")
        if isinstance(group, dict):
            ids.update(row["history_id"] for row in group.get("attempts", []) if row.get("history_id"))
    return ids


def deferred_before(source, entries, candidates, messages, before):
    """Do not advance a recent cursor beyond any quota-deferred physical row."""
    if entries is None:
        return before
    deferred = projected_history_ids(candidates) - projected_history_ids(messages)
    references = {(row.get("surface"), row.get("presentation_owner_task_id") or row.get("task_id"))
                  for row in messages if row.get("system_type") == "review_reference"}
    for row in candidates:
        if (row.get("system_type") == "review_reference"
                and (row.get("surface"), row.get("presentation_owner_task_id") or row.get("task_id")) in references):
            # Older invalidations of one already-present review owner are
            # intentionally folded by the existing projection, not quota loss.
            deferred.discard(row.get("history_id"))
    return max([before, *(entry["_history_end"] for entry in entries
                          if entry.get("history_id") in deferred
                          and entry["history_position"]["source"] == source)])


def _card_content(row, kind, child=lambda _row: False):
    """Task-card content: a host placement, or a child's own words (``child``: by the
    lineage its task result recovers). What a child delivers stands alone."""
    return row.get("card_row") in CARD_ROW_PLACEMENTS or (
        kind not in _STANDALONE_KINDS and (is_task_card_message(row) or child(row)))


def _feed_message(row):
    """A standalone conversation row with a physical chat position.

    Progress, a folded review, a task summary and task-card content change a
    card, not the conversation; a skill review is appended beside it and never
    counts as unread."""
    position = row.get("history_position")
    kind = str(row.get("system_type") or "")  # the stored ``type``
    return (isinstance(position, dict) and position.get("source") == "chat"
            and not row.get("is_progress") and not isinstance(row.get("review_group"), dict)
            and kind not in ("task_summary", "skill_review") and not _card_content(row, kind))


def _skipped_row_after(entries, after, upper):
    """Whether the physical selection skipped a row at or after ``after``."""
    position = after
    for start, end in sorted((entry["history_position"]["offset"], entry["_history_end"])
                             for entry in entries if entry["history_position"]["offset"] >= after):
        if start != position:
            return True
        position = end
    return position != upper


def _stored_message(row_matches_thread, stored_chat_id, child):
    """A stored chat row the history projection turns into a standalone message."""
    def message(entry):
        kind = str(entry.get("type") or "")
        return (str(entry.get("direction") or "").lower() in ("out", "system")
                and kind not in ("routing_options", "quiz_answer", "task_summary", "skill_review")
                and not is_a2a_chat_id(entry.get("chat_id", 1))
                and (str(entry.get("text") or "").strip() not in ("", "\u200b") or kind in _STANDALONE_KINDS)
                and row_matches_thread(stored_chat_id(entry.get("chat_id"), 1), entry)
                and not _card_content(entry, kind, child))  # last: it may read the row's task result
    return message


def latest_arrival(rows, page, data_dir, row_matches_thread, stored_chat_id, projected_gaps=frozenset(),
                   task_result=lambda _task_id: {}):
    """``window.latest_message`` of a recent Project read (DESIGN "Project unread dot").

    The window is ordered and tailed by event time, but a Project message counts
    as unread when it ARRIVES: an answer the terminal outbox delivers late keeps
    its original ``ts``, so it can sort above messages that arrived before it
    (``out_of_order``) or below the window's floor, where the first older page
    (``deferred_before``) shows it. Reading the room means reading the standalone
    message that arrived last, named from the persisted rows before the tail and
    annotation rewrite them. A child's words are card content by their stored
    lineage or, on rows written before rows carried it, by the lineage their
    ``task_result`` or this page's progress recovers — the projection's own authority, which shows such a
    row in the child's card; a photo, video, file, link card or question a child
    delivers is shown alone and is a message. ``None``: its arrival is unknown — the chat source
    is unreadable, its projection failed part-way (``projected_gaps`` holds
    ``PROJECTION_FAILED``: any row after the fault may be the newest), a row
    after the newest readable one is unreadable, or the live chat ends in
    a line its writer has not finished (``incomplete_live_line``: this read froze
    before it, so it may be the newest message) — and on a replayed recent page,
    which cannot see what arrived after its frozen boundary.

    Card rows, a child's words and the owner's own messages can fill the recent
    selection's quota. One bounded older read of the room (``_ARRIVAL_SEARCH_BYTES``)
    then names the newest stored message before it — ``out_of_order``, since this
    read cannot place it relative to the bottom — or proves the chat holds none
    (absent); an unreadable row reached first leaves the arrival unknown. Its
    bound reached first leaves it unknown too, but searched: ``latest_before``
    says none lies at or after that offset, and the chain's cursors carry that
    fact (``quiet``: none at or after the cursor's ``before``). An older page of
    a quiet chain names its own newest message — then the newest below the
    chain's frozen ``upper`` — or, holding none, passes the fact on, and the one
    that reaches the start of the chat without a gap proves the chat holds none
    below that ``upper`` (``latest_absent``); any other older page names nothing.
    A legacy child's lineage may first appear on another page. This bounded
    read can then name its final before the client places it in a child card;
    that unavailable standalone node remains unacknowledged, not guessed read.
    """
    quiet = page["quiet"]
    if page["replayed"] and page["recent"]:
        return {"latest_message": None}
    if not (page["recent"] or quiet):
        return {}
    entries, start, gaps = page["selections"]["chat"]
    end = page["page_ends"].get("chat", 0)
    # The client learns these facts before placing any row. A scheduling row's
    # task_id may name the parent; the shared projection names the actual child.
    known_children = {meta["subagent_task_id"] for row in rows if (meta := subagent_message_meta(row))}

    def child(row):
        task_id = str(row.get("task_id") or "")
        return bool(task_id) and (task_id in known_children
                                 or bool(subagent_message_meta(task_result(task_id), task_id=task_id)))

    offset = lambda row: row["history_position"]["offset"]  # noqa: E731
    words = lambda row: _card_content(row, str(row.get("system_type") or ""), child)  # noqa: E731
    feed = [row for row in rows if _feed_message(row)]
    spoken = sorted((row for row in feed if row.get("role") != "user" and (
        str(row.get("text") or "").strip() not in ("", "\u200b") or row.get("msg_type"))), key=offset)
    latest = next((row for row in reversed(spoken) if not words(row)), None)
    if entries is None or PROJECTION_FAILED in projected_gaps or "incomplete_live_line" in gaps or (
            gaps and _skipped_row_after(entries, offset(latest) if latest else start, end)):
        return {"latest_message": None}
    if not page["recent"]:
        page["quiet_below"] = latest is None
        if latest:
            return {"latest_message": {"history_id": latest["history_id"], "out_of_order": True}}
        return {"latest_absent": True} if start == 0 and not gaps else {}
    if latest is None and start > 0:
        message = _stored_message(row_matches_thread, stored_chat_id, child)
        try:
            entries, before, gaps = HistorySource(data_dir / "logs" / "chat.jsonl", "chat", end).older(
                start, 1, message, page.get("lens"), min(_ARRIVAL_SEARCH_BYTES, _READ_CEILING_BYTES))
        except OSError:
            return {"latest_message": None}
        found = next(filter(message, entries), None)  # ``older`` stops at the newest one
        if gaps and _skipped_row_after(entries, offset(found) if found else before, start):
            return {"latest_message": None}
        if not found and before > 0:
            page["quiet_below"] = True
            return {"latest_message": None, "latest_before": before}
        return {"latest_message": {"history_id": found["history_id"], "out_of_order": True}} if found else {}
    if latest is None:
        return {}
    ts = str(latest.get("ts") or "")
    return {"latest_message": {"history_id": latest["history_id"], "out_of_order": any(
        offset(row) < offset(latest) and str(row.get("ts") or "") > ts and not words(row) for row in feed)}}


def replay_evidence_rows(messages, evidence):
    """Carry cross-page evidence only when this page has no equivalent fact."""
    visible = {row.get("history_id") for row in messages if row.get("history_id")}
    answered = {(row.get("task_id"), row["quiz"].get("quiz_id")) for row in messages
                if row.get("msg_type") == "quiz" and row["quiz"].get("state") == "answered"}
    terminals = {row["task_id"]: row["historical_terminal"] for row in messages
                 if row.get("task_id") and row.get("historical_terminal")}
    return [row for row in evidence if row.get("history_id") not in visible
            and not (row.get("system_type") == "quiz_answer"
                     and (row.get("task_id"), row["quiz"].get("quiz_id")) in answered)
            and not (row.get("historical_terminal")
                     and terminals.get(row.get("task_id")) == row["historical_terminal"])]
