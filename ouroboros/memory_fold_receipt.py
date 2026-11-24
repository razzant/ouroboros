"""What a fold returns to its writer, what the Max gate's shortest telling is made of, and a start's flow line.

Everything here is read from captured facts: the writer's captured view (its context plan's snapshot and the
start's receipt) and the write's own result. Nothing reads history, stores anything or calls a model; a fold's
"after" is the writer's one frame with the write applied to it in memory, never a new capture or a start
simulation. Measures only: no threshold, no verdict, no advice; the mind judges what a fold did.
"""
from __future__ import annotations

import dataclasses
from collections import Counter
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

from ouroboros import memory_floor as floor, memory_view as mv, memory_view_conversations as conversations
from ouroboros.memory_form_facts import form_counts, holding_room

# What I read or said with no telling yet, as the ordinary view holds it (``memory_form_facts.holding_room``).
UNTOLD_KINDS = ("untold in this room", "untold in other conversations", "owner words in other rooms")
NO_FRAME = "no comparable frame"
NEXT_START = "comparison at the next comparable start"  # a write over rows the writer's frame did not hold
_ROOM_PIECES = {"owner_words": "my human's words for this work", "late_reviews": "late reviews", "marks": "marks",
                "room": "this room: headings, records, notes and other rows", "conversations": "far conversations as lines"}
_HERE = _ROOM_PIECES["room"]


def telling_parts(view: Any) -> List[List[Any]]:
    """The gate's shortest telling (``memory_floor.telling_level``) by kind, in tokens that sum to its ``telling``
    figure (``tokens_of`` of each block, the floor note in the room block). Read from prices the start already holds,
    never by drawing a text (cost law L7): each story entry's element price in its form at that level (an account's
    own lines), the room block's pieces by key (priced once per key, ``MemoryViewSnapshot.piece``), my human's words
    elsewhere by their element prices and this room's task gists by their line; what no entry or piece names (the
    story's headings and status, joins) stays with its block. Each block's tokens are split over these weights by
    cumulative share. [] where no view was priced."""
    snapshot, level = getattr(view, "snapshot", None), getattr(view, "level", None)
    if snapshot is None or level is None or not snapshot.active:
        return []
    from ouroboros.memory_view_conversation_forms import owner_items

    gone, prices = {step: set(ids) for step, ids in level.addressed}, {}
    for step, ident, whole, short in floor.degradable_elements(snapshot):
        prices[step, ident] = (whole, short)
    story: Dict[str, int] = {}

    def add(kinds: Dict[str, int], kind: str, tokens: int) -> None:
        kinds[kind] = kinds.get(kind, 0) + 4 * max(0, tokens)  # in characters, as the pieces are
    for entry in (e for e in snapshot.story if not e.get("told_by") and e.get("kind") != "legacy"):
        ident, lead = entry["id"], entry["id"] in gone.get("F5l", ())
        whole, short = prices.get(("F5l", ident)) or prices.get(("F5", ident)) or (None, None)
        tokens = floor.drawn_tokens(entry) if whole is None else short if lead else whole
        add(story, "accounts" if entry.get("kind") == "account" else "parts" if entry.get("kind") == "part" else
            "pages standing as leads" if lead else "pages whole (no lead rule)", tokens)
    for step, ident in prices:
        if step == "F3":
            add(story, "old story", prices[step, ident][1 if ident in gone.get("F3", ()) else 0])
    story_chars = mv.story_chars(snapshot, level)
    story["story headings and status"] = max(0, story_chars - sum(story.values()))
    room: Dict[str, int] = {}
    pieces: List[int] = []

    def piece(key: Any, make: Any) -> int:
        chars = snapshot.piece(key, make)
        if chars:
            pieces.append(chars)
            kind = _ROOM_PIECES.get(key[0], "far conversations as lines")
            room[kind] = room.get(kind, 0) + chars
        return chars
    mv._room_parts(snapshot, dict(level.addressed), piece)
    elsewhere = sum(prices.get(("F7", ident), (0, 0))[0] for ident in owner_items(snapshot.conversations)
                    if ident not in gone.get("F7", ()))
    moves = [("my human's words elsewhere", "far conversations as lines", elsewhere)]
    moves += [(kind, _HERE, tokens) for kind, tokens in _room_stages(snapshot, gone, prices).items()]
    for kind, source, tokens in moves:  # this room's piece, split by the stage each of its forms stands at
        moved = min(4 * tokens, room.get(source, 0))
        if moved:
            room[source] -= moved
            room[kind] = room.get(kind, 0) + moved
    room[_HERE] = room.get(_HERE, 0) + 4 * max(0, len(pieces) - 1)  # the blank lines joining the parts
    room["the floor note"] = max(0, mv.room_chars(snapshot, level, floor_note=getattr(view, "note", "")) - sum(room.values()))
    total: Dict[str, int] = {}
    for kinds, chars in ((story, story_chars), (room, view.room_chars)):
        for kind, tokens in _split(kinds, floor.tokens_of(chars)):
            total[kind] = total.get(kind, 0) + tokens
    return [[kind, tokens] for kind, tokens in sorted(total.items(), key=lambda item: (-item[1], item[0])) if tokens]


def _room_stages(snapshot: mv.MemoryViewSnapshot, gone: Mapping[str, Any], prices: Mapping[Any, Any]) -> Dict[str, int]:
    """This room's forms at a level by stage, in tokens from the element prices (whole, or short where taken): finished
    works standing as their lines, my replies, my human's words, works with a gist but no reply here, other task fact
    lines, and task gist lines (priced by their own line)."""
    from ouroboros.memory_forms import stage_of
    from ouroboros.memory_view_work import task_gist_line

    here, out = snapshot.room or {}, {}
    owners = {row["address"] for row in [*(here.get("project_words") or ()), *(here.get("lane1") or ())]
              if row.get("address") and row.get("owner_side", row.get("kind") == "human")}
    for (step, ident), (whole, short) in prices.items():
        taken = ident in gone.get(step, ())
        if step == "F2" and ident.startswith("work:") and ident.endswith(":final"):
            kind, tokens = "finished works standing as lines", short if taken else 0
        elif step == "F2" and not ident.startswith(("work:", "bare:")):
            kind, tokens = "my replies", short if taken else whole
        elif step == "F7" and ident in owners:
            kind, tokens = "my human's words in this room", short if taken else whole
        elif step == "F1":
            only = stage_of(snapshot, step, ident) == "task_only"
            kind = "works with a gist but no reply here" if only else "task fact lines"
            tokens = short if taken else whole
        else:
            continue
        out[kind] = out.get(kind, 0) + tokens
    gists = sum(floor.view_tokens(task_gist_line(task, here["room_id"])) for task in here.get("lane2") or ()
                if task.get("gist") and task["first"] in gone.get("F1", ()))
    if gists:
        out["task gist lines"] = gists
    return out


def _split(chars: Mapping[str, int], tokens: int) -> List[Tuple[str, int]]:
    """``tokens`` over ``chars``'s kinds by cumulative share: the parts sum to ``tokens`` exactly."""
    total, through, used, out = sum(chars.values()), 0, 0, []
    for kind, size in chars.items():
        through += size
        share = tokens * through // total if total else 0
        out.append((kind, share - used))
        used = share
    return out


# --- a fold's receipt -------------------------------------------------------------------------------

# A writer saves one record and names what it changed; it re-reads no whole story around the write (a capture is a
# read of all history, and a store the importer holds would fail a choice already saved): its prices are the writer's
# own captured frame with the write applied in memory. The story as it now stands is memory_read(story=true)'s; a
# selection's full before/after view is memory_read(proposed_selection=...) before it.
_NOT_REFRESHED = ("This task's view is not refreshed; the next start measures its Max room again and rewrites the cached "
                  "story from the first changed retelling.")


def note_read(ctx: Any, node_id: Any) -> None:
    """Remember, for this task's fold receipts, a record it read with ``memory_read(node_id)``: held, not only seen."""
    held = getattr(ctx, "_memory_nodes_read", None)
    if held is None:
        held = set()
        try:
            setattr(ctx, "_memory_nodes_read", held)
        except AttributeError:  # a context that takes no attribute keeps no reads: they are then named unseen
            return
    held.add(str(node_id))


def captured(ctx: Any) -> Tuple[Optional[mv.MemoryViewSnapshot], mv.FloorLevel]:
    """The writer's last captured frame: its plan's snapshot and the level its delivered view showed it at."""
    plan = getattr(ctx, "context_fit_plan", None)
    held = getattr(getattr(plan, "core", None), "memory_view_json", "")
    if not held:
        return None, mv.FULL_VIEW
    snapshot = mv.snapshot_from_json(held)
    mode = getattr(ctx, "active_context_mode", None) or getattr(plan, "initial_mode", "max")
    try:
        return snapshot, floor.kept_level(snapshot, plan.projection(mode).memory_facts)
    except (AttributeError, KeyError, TypeError):
        return snapshot, mv.FULL_VIEW


def measure(snapshot: mv.MemoryViewSnapshot) -> Dict[str, Any]:
    """One frame's two prices: the ordinary untold amount (``UNTOLD_KINDS`` in the ordinary view, remote words at
    whole-text size) and the gate's shortest telling (``telling_level``, memory alone, no floor note), with the
    emergency forms that telling holds and how many pages no part or account holds."""
    told = floor.telling_level(snapshot)
    held = holding_room(snapshot, floor.ordinary_level(snapshot))["tokens"]
    return {"untold": sum(held.get(kind, 0) for kind in UNTOLD_KINDS),
            "telling": floor.tokens_of(mv.story_chars(snapshot, told)) + floor.tokens_of(mv.room_chars(snapshot, told)),
            "emergency": sorted(form_counts(snapshot, told)["emergency"]),
            "unfolded": floor.unfolded_pages(snapshot)["pages"]}


def _sha(address: Any) -> str:
    return str(address or "").rsplit("#", 1)[-1]


def applied(snapshot: mv.MemoryViewSnapshot, *, add: Iterable[Mapping[str, Any]] = (), drop: Iterable[str] = (),
            told_by: Optional[Tuple[str, Iterable[str]]] = None, rows: Iterable[str] = ()) -> mv.MemoryViewSnapshot:
    """The captured frame with one write applied in memory: ``drop`` leaves the story, ``told_by`` (account, ids)
    marks records an account tells, ``add`` joins (a page at the end, a part where its first member stood, an
    account among the accounts), and the open rows whose stored hash ``rows`` names leave this room's lanes and my
    other conversations (a lane-2 line with its first row). A new snapshot object: no price of the old one is kept."""
    drop, covered = {str(i) for i in drop}, {str(sha)[:12] for sha in rows}
    teller, told = (told_by[0], {str(i) for i in told_by[1]}) if told_by else ("", set())
    story = [{**entry, "told_by": teller} if entry["id"] in told and not entry.get("told_by") else entry
             for entry in snapshot.story]
    for entry in (entry for entry in add if entry["id"] not in {e["id"] for e in story}):  # one already shown stays
        place = next((n for n, e in enumerate(story) if e["id"] in drop), None) if entry.get("kind") == "part" else None
        place = sum(e.get("kind") == "account" for e in story) if entry.get("kind") == "account" else place
        story.insert(len(story) if place is None else place, dict(entry))
    story = [entry for entry in story if entry["id"] not in drop]
    room, groups = snapshot.room, snapshot.conversations
    if covered and room is not None:
        keep = lambda items, field: [item for item in items or () if _sha(item.get(field)) not in covered]  # noqa: E731
        room = {**room, "lane1": keep(room.get("lane1"), "address"), "lane2": keep(room.get("lane2"), "first"),
                **({"project_words": keep(room["project_words"], "address")} if room.get("project_words") else {})}
    if covered:
        groups = tuple({**group, "rooms": [{**place, "words": [w for w in place["words"] if _sha(w.get("address")) not in covered]}
                                           for place in group["rooms"]]} for group in groups)
    return dataclasses.replace(snapshot, story=tuple(story), room=room, conversations=groups)


def story_entry(record: Mapping[str, Any], *, label: str = "", span: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """A just-written page, part or account as the story captures it, from the write's own record (its period from
    the rows it covers, when known)."""
    meta = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
    span = dict(span or {})
    entry = {"kind": record.get("kind"), "id": str(record.get("id")), "room_id": str(record.get("room_id")),
             "label": str(meta.get("room_label") or label or record.get("room_id")),
             "period": f"{span.get('start') or '?'} → {span.get('end') or '?'}", "source_period": span,
             "text": str(record.get("current_text") or record.get("text") or ""), "status": str(record.get("status") or ""),
             "stamp": "", "fixes": [], "quotes": list(record.get("quotes") or ())}
    if "lead_rule" in meta:
        entry["lead_rule"] = meta["lead_rule"]
    if record.get("kind") == "account":
        entry.update(selection=str((record.get("selection") or {}).get("id") or ""), sources=list(record.get("sources") or ()))
    return entry


def _order(snapshot: mv.MemoryViewSnapshot) -> List[str]:
    """The story's retellings in the order the story block draws them (``memory_view._story_text``)."""
    shown = [entry for entry in snapshot.story if not entry.get("told_by")]
    return [entry["id"] for group in (("account",), ("legacy",)) for entry in shown if entry.get("kind") in group] + [
        entry["id"] for entry in shown if entry.get("kind") not in ("account", "legacy")]


def first_change(before: mv.MemoryViewSnapshot, after: mv.MemoryViewSnapshot) -> str:
    """Where the cached story changes: the first retelling that differs, counted in the story's drawing order."""
    old, new = _order(before), _order(after)
    for n, (a, b) in enumerate(zip(old, new)):
        if a != b:
            return f"the cached story changes from retelling {n + 1} of {len(new)} on ({b})"
    if len(old) == len(new):
        return "the cached story's retellings stay as they were"
    n = min(len(old), len(new))
    return (f"the cached story keeps its first {n} retelling{'' if n == 1 else 's'} and changes only after them"
            + (f" ({new[n]} joins at the end)" if len(new) > n else ""))


def _forms(snapshot: mv.MemoryViewSnapshot, level: mv.FloorLevel) -> Dict[str, str]:
    """How the writer's delivered view held each source it captured, by record id and by row hash: ``whole`` or
    ``short`` (a lead, an opening, an address, a line or a count)."""
    from ouroboros.memory_view_conversation_forms import local_key
    from ouroboros.memory_view_work import folded_work_rows

    gone = {step: set(ids) for step, ids in level.addressed}
    held: Dict[str, str] = {}
    for entry in snapshot.story:
        short = (entry.get("told_by") or entry["id"] in gone.get("F5", ()) or entry["id"] in gone.get("F5l", ())
                 or entry.get("kind") == "legacy" and gone.get("F3"))
        held[entry["id"]] = "lead" if entry["id"] in gone.get("F5l", ()) and not entry.get("told_by") else (
            "short" if short else "whole")
    room = snapshot.room or {}
    for item in [*room.get("legacy", ()), *room.get("under_parts", ())]:
        held.setdefault(item["id"], "short" if item["id"] in gone.get("F4", ()) else "whole")
    hidden = folded_work_rows(room, gone)[0] if room else set()
    rows = gone.get("F2", set()) | gone.get("F7", set()) | gone.get("F7p", set())
    for row in room.get("lane1", ()):
        address = row["address"]
        short = address in rows or "bare:" + address in rows or address in hidden or bool(gone.get("F7a"))
        held[_sha(address)] = "short" if short else "whole"
    for task in room.get("lane2", ()):
        held[_sha(task["first"])] = "short" if task["first"] in gone.get("F1", ()) else "whole"
    index = conversations.source_index(snapshot)
    for group in snapshot.conversations:
        form = local_key(group, gone, index[group["key"]])[3]
        for place in group["rooms"]:
            for word in place["words"]:
                ident = word.get("id")
                short = form or ident in gone.get("F6w", ()) or "bare:" + str(ident) in gone.get("F6w", ()) or ident in gone.get("F7", ())
                held[_sha(word.get("address"))] = "short" if short else "whole"
    return held


def _held_line(what: str, ids: Iterable[str], forms: Mapping[str, str], read: Iterable[str] = ()) -> str:
    counts = Counter("read" if ident in read else forms.get(ident, "absent") for ident in ids)
    named = (("whole", "whole in my delivered view"), ("read", "read whole with memory_read in this task"),
             ("lead", "only as a lead"), ("short", "only as an opening, a pointer, an address or a count"),
             ("absent", "not in my captured view"))
    return f"{what}: " + "; ".join(f"{counts[key]} {words}" for key, words in named if counts[key])


def _prices(before: Mapping[str, Any], after: Optional[Mapping[str, Any]]) -> str:
    if after is None:
        return f"prices: {NEXT_START} (this write concerns what my captured frame does not hold)"
    emergency = ", ".join(after["emergency"]) or "none"
    return (f"in my one captured frame, before -> after: the ordinary untold amount (this room's open rows, my untold "
            f"words and my human's words in other conversations, remote words at whole-text size) "
            f"{before['untold']:,} -> {after['untold']:,} tokens; the gate's shortest telling (memory alone, no floor "
            f"note) {before['telling']:,} -> {after['telling']:,} tokens; emergency forms in that telling after: "
            f"{emergency}; pages no part or account holds {before['unfolded']} -> {after['unfolded']}")


def fold_receipt(ctx: Any, *, told: str, after: Optional[Dict[str, Any]] = None, sources: Iterable[str] = (),
                 outcome: str = "") -> str:
    """The facts a fold returns: ``told`` (what this telling now covers, counted by the write), how the writer's
    delivered view held the covered rows or the records named (``sources``: an account's or selection's, with
    those this task read whole), the two prices before and after in the writer's one frame (``after``: the
    ``applied`` arguments, None where the write cannot be applied to that frame) and where the cached story
    changes; ``outcome``: a refusal, rejection or supersession, said plainly first."""
    snapshot, level = captured(ctx)
    lines = [outcome] if outcome else []
    lines.append(told)
    if snapshot is None or not snapshot.active:
        lines.append(f"prices: {NO_FRAME}: this task holds no captured memory view")
    else:
        forms, read = _forms(snapshot, level), getattr(ctx, "_memory_nodes_read", set()) or set()
        change = after or {}
        ids = [str(i) for i in sources] or [*change.get("drop", ()), *(str(sha)[:12] for sha in change.get("rows", ()))]
        if sources:
            lines.append(_held_line("its sources as this task held them", ids, forms, read))
        elif ids:
            lines.append(_held_line("what it tells, as my delivered view held it", ids, forms))
        unheld = not sources and any(forms.get(ident, "absent") == "absent" for ident in ids)
        try:
            then = applied(snapshot, **after) if after is not None and not unheld else None
            before = measure(snapshot)
            lines.append(_prices(before, measure(then) if then is not None else None))
            if then is not None:
                lines.append(first_change(snapshot, then))
        except (KeyError, TypeError, ValueError):  # a record this frame cannot draw: no measure, never a guess
            lines.append(f"prices: {NO_FRAME}: the write could not be applied to my captured view")
    lines.append("Originals remain readable by address; memory_read(story=true) measures my story as it stands now. "
                 + _NOT_REFRESHED)
    return "\n".join(lines)


def by_room(counts: Mapping[str, int]) -> str:
    return ", ".join(f"{room} ({n})" for room, n in sorted(counts.items())) or "none"


# --- a start's flow line ----------------------------------------------------------------------------

def flow_line(snapshot: mv.MemoryViewSnapshot, receipt: Mapping[str, Any], task_id: str = "") -> str:
    """Since this room's previous start as this capture records it (its newest lane-2 work other than this task:
    its id and first row's time): open rows that arrived since, retellings whose covered period reaches past it,
    what I read or said that stands untold (with what arrived since), the largest untold rooms with their oldest
    untold date and the last telling there, pages no part or account holds, and why Max was lost in its numbers.
    Facts only; a capture with no earlier start of this room says so."""
    room = snapshot.room or {}
    here = str(room.get("room_id") or "")
    earlier = [t for t in room.get("lane2") or () if t.get("task") and t["task"] != task_id and t.get("ts")]
    since = max(earlier, key=lambda t: t["ts"]) if earlier else None
    untold: Dict[str, List[Any]] = {}  # room -> [label, tokens, rows, oldest ts, rows since]

    def hold(place: str, label: str, rows: Iterable[Mapping[str, Any]]) -> None:
        for row in rows:
            ts = str(row.get("ts") or "")
            item = untold.setdefault(place, [label, 0, 0, ts, 0])
            item[1] += floor.view_tokens(str(row.get("line") or ""))
            item[2] += 1
            item[3] = min(item[3], ts) if item[3] and ts else item[3] or ts
            item[4] += bool(since and ts > since["ts"])
    hold(here, str(room.get("label") or "this room"), [row for row in room.get("lane1") or () if not row.get("gist")])
    index = conversations.source_index(snapshot)
    for group in snapshot.conversations:
        for place in group["rooms"]:
            kinds = index[group["key"]]["rooms"][place["room_id"]]
            hold(str(place["room_id"]), str(place.get("label") or group.get("label")), [*kinds["untold"], *kinds["owner"]])
    tellings = [e for e in snapshot.story if e.get("kind") in ("page", "part", "account")]
    if since is None:
        head = "flow: no comparable capture (no earlier start of this room in this capture)"
    else:
        arrived = Counter(place for place, ts in _open_rows(snapshot, here) if ts > since["ts"])
        reach = [e for e in tellings if str((e.get("source_period") or {}).get("end") or "") > since["ts"]]
        head = (f"flow since this room's previous start, task {since['task']} at {_minute(since['ts'])}: "
                f"{sum(arrived.values())} rows arrived in {len(arrived)} room{'' if len(arrived) == 1 else 's'}; "
                f"{len(reach)} retellings cover time after it")
    total = [sum(item[n] for item in untold.values()) for n in (1, 2, 4)]
    parts = [head, f"untold of what I read or said {total[0]:,} tokens, {total[1]} rows, {len(untold)} rooms"
             + (f", {total[2]} rows after that start" if since else "")]
    for place, (label, tokens, rows, oldest, _new) in sorted(untold.items(), key=lambda kv: (-kv[1][1], kv[0]))[:3]:
        last = max((e for e in tellings if str(e.get("room_id")) == place),
                   key=lambda e: str((e.get("source_period") or {}).get("end") or ""), default=None)
        end = str(((last or {}).get("source_period") or {}).get("end") or "")[:10]
        told = ("no telling of it in my story" if last is None else f"last telling to {end}" if end
                else "last telling's period not recorded") + (f", draft by {last['signer']}" if last and last.get("signer") else "")
        parts.append(f"{label} {tokens:,} tokens, {rows} rows, oldest {oldest[:10] or 'not recorded'}, {told}")
    parts.append(f"{floor.unfolded_pages(snapshot)['pages']} pages no part or account holds")
    fact = receipt.get("max_room")
    if isinstance(fact, Mapping) and fact.get("cause") in ("crowded", "route"):
        parts.append(_books_out(fact))
    return "; ".join(parts)


def _books_out(fact: Mapping[str, Any]) -> str:
    """Why the books are out, and what stays after everything foldable is folded: with no memory at all the rest of
    this start needs ``zero_need`` (``floor_need`` without the tool schemas); where that alone exceeds the window the
    fixed part is the cause. Else the Max room left for memory beside whole books against my shortest telling, and
    the part of that telling only my own page folds (my human's words)."""
    cause, need = floor.lowered_need(fact)
    window = fact["window"]
    if cause != "memory":
        return (f"Max lost, cause {cause}: with no memory this start needs {need:,} of {window:,}, so with all my memory "
                "folded the fixed part alone stays above the window")
    owner = sum(int(tokens) for kind, tokens in fact.get("telling_parts") or () if kind.startswith("my human's words"))
    return (f"Max lost, cause memory: the gate's need {need:,} of {window:,}; with all my memory folded the rest needs "
            f"{fact['zero_need']:,}; the Max room for memory {fact['allowance']:,} against my shortest telling "
            f"{fact.get('telling', 0):,}" + (f", of which my human's words, which only my own page folds, {owner:,}"
                                             if owner else ""))


def _minute(ts: Any) -> str:
    """A captured time to the minute, ``2026-09-07 00:02``: no seconds, fraction or offset."""
    return str(ts or "")[:16].replace("T", " ") or "time not recorded"


def _open_rows(snapshot: mv.MemoryViewSnapshot, here: str) -> List[Tuple[str, str]]:
    room = snapshot.room or {}
    rows = [(here, str(row.get("ts") or "")) for row in room.get("lane1") or ()]
    return rows + [(str(place["room_id"]), str(word.get("ts") or "")) for group in snapshot.conversations
                   for place in group["rooms"] for word in place["words"]]
