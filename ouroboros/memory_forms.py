"""Memory forms and their distance order, independent of a request's budget and fit.

Each element keeps an existing rung and source identity. Stages order those
elements; the clean-start guard walks only ordinary forms, event counts first.
"""
from collections import defaultdict, deque
from itertools import groupby

from ouroboros import memory_view as mv, memory_view_conversations as conversations, memory_view_legacy as legacy
from ouroboros.memory_view_work import works, work_line, task_gist_line, task_line
from ouroboros.memory_form_facts import form_counts as form_counts, holding_room as holding_room

LADDER = ("F1", "F6w", "F6g", "F3", "F2", "F4", "F7p", "F5l", "F5", "F5a", "F7", "F7a", "F8", "F8a", "F6a")
# The owner's order A keeps work_final before remote_last.
STAGES = ("remote_events", "remote_told", "remote_unread", "remote_gists", "task_facts", "work_progress",
          "remote_mine", "remote_people", "remote_all", "work_final", "remote_last", "story_ordinary",
          "story_openings", "room_emergency", "task_only", "protected_replies", "remote_count", "story_count",
          "unread_count", "physical_rows", "room_ranges", "owner")


def _catalogue(snapshot):
    """One capture's forms, source facts and stages; not a render or a persistent snapshot field."""
    room = snapshot.room or {}
    words = {w["id"]: w for g in snapshot.conversations for r in g["rooms"] for w in r["words"]}
    own = {w["address"]: w for w in room.get("lane1", ())}
    tasks = {t["first"]: t for t in room.get("lane2", ())}
    records = {r["id"]: r for r in [*snapshot.story, *room.get("notes", ())]}
    progress, finals, answered = {}, {}, set()  # answered: the works with replies here, by their own key
    for key, rows in works(room):
        progress["work:" + key + ":progress"] = rows
        finals["work:" + key + ":final"] = rows
        answered.add(key)
    parents = {w["id"]: g["key"] for g in snapshot.conversations for r in g["rooms"] for w in r["words"]}
    kinds = {ident: kind for data in conversations.source_index(snapshot).values() for ident, kind in data["kinds"].items()}
    return dict(room=room, words=words, own=own, tasks=tasks, records=records, progress=progress,
                finals=finals, parents=parents, kinds=kinds, answered=answered,
                standing={r["id"] for r in room.get("under_parts", ()) if r.get("standing")})


def stage_of(snapshot, step, ident):
    cached = snapshot.__dict__.get("_form_catalogue")
    if cached is None:
        cached = _catalogue(snapshot)
        object.__setattr__(snapshot, "_form_catalogue", cached)
    if ident.startswith("bare:") and step in ("F2", "F6w", "F7p"):
        return "owner" if step == "F2" and cached["own"].get(ident[5:], {}).get("owner_quoted") else "physical_rows"
    if step == "F1":
        task = cached["tasks"][ident]
        return "task_facts" if not task.get("gist") or task.get("task") in cached["answered"] else "task_only"
    if step == "F2":
        if ident in cached["progress"]:
            return "work_progress"
        if ident in cached["finals"]:
            return "protected_replies" if any(w.get("protected_reply") for w in cached["finals"][ident]) else "work_final"
        own = cached["own"][ident]
        if own.get("owner_quoted"):  # a relay of my human's words yields with my human's own words, last
            return "owner"
        return "protected_replies" if own.get("protected_reply") else "room_emergency"
    if step == "F6w":
        kind = cached["kinds"][ident]
        return ("remote_told" if kind == "told" else "remote_unread" if kind == "unread"
                else "remote_mine" if cached["words"][ident]["kind"] == "ouroboros" else "remote_people")
    if step == "F6g":
        return ("remote_events" if ident.startswith("events:") else "remote_all" if ident.startswith("all:")
                else "remote_last" if ident.startswith("last:") else "remote_gists")
    if step == "F6a":
        return "unread_count" if ident == "unread" else "remote_count"
    if step in ("F5l", "F8"):
        record = cached["records"][ident]
        return "story_ordinary" if "lead_rule" in record and record.get("text", "").strip() else "story_openings"
    if step == "F4" and ident in cached["standing"]:
        return "story_ordinary"
    if step in ("F5a", "F8a"):
        return "story_count"
    if step == "F7a":
        return "room_ranges"
    if step == "F5":
        return "story_openings"
    return "owner" if step == "F7" else "room_emergency"


def ordinary_pairs(snapshot, elements=None):
    cached = snapshot.__dict__.get("_ordinary_forms")
    if elements is not None and cached and cached[0] is elements:
        return cached[1]
    ordinary = {"remote_events", "remote_told", "remote_unread", "remote_gists", "task_facts", "work_progress",
                "work_final", "story_ordinary"}
    listed = list_elements(snapshot) if elements is None else elements
    stages = snapshot.__dict__.get("_form_stages", {})
    pairs = {(step, ident) for step, ident, *_ in listed if stages[step, ident] in ordinary}
    if elements is not None:
        object.__setattr__(snapshot, "_ordinary_forms", (elements, pairs))
    return pairs


def supersedes(step, ident):
    if step in ("F2", "F6w", "F7p") and ident.startswith("bare:"):
        return ((step, ident[5:]),)
    if step == "F5":
        return (("F5l", ident),)
    if step == "F2" and ident.startswith("work:") and ident.endswith(":final"):
        return ((step, ident.removesuffix(":final") + ":progress"),)
    if step == "F6g" and ident.startswith(("all:", "last:")):
        key = ident.split(":", 1)[1]
        return ((step, "events:" + key), (step, key), (step, "all:" + key)) if ident.startswith("last:") else (
            (step, "events:" + key), (step, key))
    if step == "F6g" and not ident.startswith("events:"):
        return ((step, "events:" + ident),)
    return ()


def normalized(selected, replacements=None):
    """Superseded forms do not acquire a second receipt or a different price key."""
    chosen = set(selected)
    relations = replacements.items() if replacements is not None else ((p, supersedes(*p)) for p in selected)
    for pair, lower in relations:
        if pair in selected:
            chosen.difference_update(lower)
    return chosen


class WorkPrices:
    """Marginal savings from a work's own pieces, including the telling it replaces.

    A final line replaces the remaining replies plus the intermediate line, when
    selected. The task's gist moves to the work line only once, on its first fold.
    """
    def __init__(self, room, elements, price):
        self.replies = {ident: (whole, short) for step, ident, whole, short in elements
                        if step == "F2" and ident.startswith("work:")}
        self.tellings = defaultdict(list)
        for task in room.get("lane2", ()):
            if task.get("gist"):
                self.tellings[task["task"]].append((task["first"], price(task_gist_line(task)),
                                                  price(task_line(task)) - price(task_line(task, folded=True))))

    def saving(self, ident, selected):
        whole, short = self.replies[ident]
        progress = ident.removesuffix(":final") + ":progress"
        if ident.endswith(":final") and ("F2", progress) in selected:
            previous_whole, previous_short = self.replies[progress]
            whole -= previous_whole - previous_short
        else:
            key = ident.removeprefix("work:").rsplit(":", 1)[0]
            whole += sum(gist if ("F1", first) in selected else inline
                         for first, gist, inline in self.tellings.get(key, ()))
        return max(0, whole - short)


class Selection:
    """A reversible walk's choices, with visible ids and ordered tuples maintained per rung.

    Taking a coarser form hides its predecessor; returning it reveals that choice
    again. Only the changed rungs need ordering when the next price asks for a level.
    """
    def __init__(self, table, pairs=(), *, old=(), aggregates=None):
        self.table, self.old, self.aggregates = table, old, aggregates or {}
        self.pairs = {(s, i) for s, i in pairs if i in table.positions.get(s, ())}
        self.by_step, self.ordered = defaultdict(set), {}
        self.upper, self.lower = table.replacements
        visible = self.pairs.copy()
        for upper, lower in self.upper.items():
            if upper in self.pairs:
                visible.difference_update(lower)
        for step, ident in visible:
            self.by_step[step].add(ident)
        self.new_count = len(visible.difference(old))

    def __contains__(self, pair):
        return pair in self.pairs

    def _visible(self, pair, shown):
        step, ident = pair
        ids = self.by_step[step]
        if (ident in ids) != shown:
            if shown:
                ids.add(ident)
            else:
                ids.discard(ident)
            self.new_count += (1 if shown else -1) * (pair not in self.old)
            self.ordered.pop(step, None)

    def add(self, pair):
        if pair in self.pairs:
            return
        self.pairs.add(pair)
        for lower in self.upper.get(pair, ()):
            self._visible(lower, False)
        uppers = self.lower.get(pair)
        if not uppers or not any(p in self.pairs for p in uppers):
            # A newly selected pair cannot already be visible. Only predecessors
            # need the conditional update above (some are absent or hidden).
            step, ident = pair
            self.by_step[step].add(ident)
            self.new_count += pair not in self.old
            self.ordered.pop(step, None)

    def discard(self, pair):
        self.pairs.discard(pair)
        self._visible(pair, False)
        for lower in self.upper.get(pair, ()):
            if lower in self.pairs and not any(p in self.pairs for p in self.lower.get(lower, ())):
                self._visible(lower, True)

    def level(self, ladder, by_budget=0, budget=False):
        addressed, new = [], self.new_count
        for step in ladder:
            ids = self.by_step.get(step)
            if not ids:
                continue
            if step in self.aggregates and not any(self.by_step.get(s) for s in self.aggregates[step]):
                new -= sum((step, i) not in self.old for i in ids)
                continue
            if step not in self.ordered:
                whole = self.table.ids[step]
                self.ordered[step] = whole if len(ids) == len(whole) else tuple(sorted(ids, key=self.table.positions[step].__getitem__))
            addressed.append((step, self.ordered[step]))
        return mv.FloorLevel(tuple(addressed), by_budget=by_budget + (new if budget else 0))


def swallowed(snapshot, step, ident, selected):
    """A grouped conversation owns its inner forms until it is returned to full text."""
    uppers = selected.lower.get((step, ident))
    if uppers and any(upper in selected for upper in uppers):
        return True
    if step not in ("F6w", "F6g"):
        return False
    if not (selected.by_step.get("F6g") or selected.by_step.get("F6a")):
        return False
    key = (snapshot.__dict__["_form_catalogue"]["parents"].get(ident.removeprefix("bare:")) if step == "F6w" else
           ident.split(":", 1)[1] if ident.startswith(("all:", "last:", "events:")) else ident)
    return (("F6a", key) in selected or step == "F6w" and any(
        ("F6g", name) in selected for name in (key, "all:" + key, "last:" + key))
        or step == "F6g" and ident.startswith("events:") and ("F6g", key) in selected
        or step == "F6g" and ident == key and ("F6g", "all:" + key) in selected
        or step == "F6g" and ident != "last:" + key and ("F6g", "last:" + key) in selected)


def _unread_return_prices(snapshot, price):
    """The shared unread form hides these local rows in addition to each group's original saving."""
    cached = snapshot.__dict__.get("_unread_return_prices")
    if cached is None:
        from ouroboros.memory_view_conversation_forms import counted_unread_lines
        index = conversations.source_index(snapshot)
        cached = {g["key"]: price("\n".join(counted_unread_lines(g, index[g["key"]])))
                  for g in snapshot.conversations}
        object.__setattr__(snapshot, "_unread_return_prices", cached)
    return cached


def _return_counted_prefix(batch, taken, current, target, failed, level, measure):
    """Locate the fitting nearest prefix after a failed batch, without replaying every conversation.

    The full batch has already failed. Its measured cost supplies the first
    estimate; subsequent probes bisect the remaining prefix lengths. Every
    accepted view is measured, and the immediately older prefix is known not to fit.
    """
    low, high, shown = 0, len(batch), 0
    probe = max(1, min(high - 1, int(high * (target - current) / (failed - current))))
    def move(count):
        nonlocal shown
        if count > shown:
            for pair, _delta in batch[shown:count]:
                taken.discard(pair)
        else:
            for pair, _delta in reversed(batch[count:shown]):
                taken.add(pair)
        shown = count
    while high - low > 1:
        move(probe)
        actual = measure(level(taken))
        if actual <= target:
            low, current = probe, actual
        else:
            high = probe
        probe = (low + high) // 2
    move(low)
    return current


def restore(snapshot, table, taken, added, target, current, level, measure, price):
    """Probe a stage's affordable prefix together; unread count batches search prefix lengths on failure.

    Return order and blocking stay local to a stage. The accumulated price is
    only an estimate: every accepted batch has its complete price checked, and
    a failed batch restores its original choices before trying a shorter prefix.
    """
    if target is None:
        return current
    for _stage, entries in groupby(reversed(added), key=lambda item: table.stages[item[0]]):
        extra = _unread_return_prices(snapshot, price) if _stage == "remote_count" and ("F6a", "unread") in taken else {}
        pending = deque((pair, delta + extra.get(pair[1], 0)) for pair, delta in entries)
        while pending:
            batch, cost = [], 0
            while pending:
                pair, delta = pending[0]
                if swallowed(snapshot, *pair, taken):
                    pending.popleft()
                    continue
                if current + cost + delta > target:
                    break
                pending.popleft()
                taken.discard(pair)
                batch.append((pair, delta))
                cost += delta
            if not batch:
                if not pending:
                    break
                # A local saving can exceed the actual restore price by rounding
                # or shared headers. Probe before giving up on this stage;
                # a failed probe still ends that stage's older prefix.
                pair, _delta = pending.popleft()
                taken.discard(pair)
                actual = measure(level(taken))
                if actual <= target:
                    current = actual
                    continue
                taken.add(pair)
                break
            actual = measure(level(taken))
            if actual <= target:
                current = actual
                continue
            for pair, _delta in reversed(batch):
                taken.add(pair)
            if len(batch) > 1 and _stage == "remote_count" and ("F6a", "unread") in taken and current <= target:
                current = _return_counted_prefix(batch, taken, current, target, actual, level, measure)
            elif len(batch) > 1:
                for pair, delta in batch:
                    if current + delta > target:
                        break
                    taken.discard(pair)
                    actual = measure(level(taken))
                    if actual > target:
                        taken.add(pair)
                        break
                    current = actual
            break
    return current


def list_elements(snapshot):
    if not snapshot.active:
        return []
    room = snapshot.room or {}
    lane1, lane2 = room.get("lane1") or [], room.get("lane2") or []
    mine = sorted((r for r in lane1 if r["kind"] == "ouroboros" and not r.get("gist")), key=lambda r: r["pos"])
    block = conversations.floor_steps(snapshot.conversations, index=conversations.source_index(snapshot))
    work = []
    for key, rows in works(room):
        partial = [r for r in rows[:-1] if not r.get("protected_reply")]
        if partial:
            work.append(("work:" + key + ":progress", "\n".join(r["line"] for r in partial), work_line(partial)))
        work.append(("work:" + key + ":final", "\n".join(r["line"] for r in rows), work_line(rows)))
    story = sorted(snapshot.story, key=lambda r: tuple(r.get("floor_order", (0, 0))))
    work_keys = {key for key, _rows in works(room)}
    steps = {
        "F1": [(r["first"], r["line"], ("" if n else mv._facts_line(room, [r]))
                + (task_gist_line(r) if r.get("gist") and r.get("task") in work_keys else "")) for n, r in enumerate(lane2)],
        **{step: block[step] for step in ("F6w", "F6g", "F6a")},
        "F2": work + [(r["address"], r["line"], mv._row_opening(r, "my reply")) for r in mine]
              + [("bare:" + r["address"], mv._row_opening(r, "my reply"), mv._row_pointer(r, "my reply")) for r in mine],
        "F3": [(rid, "\n".join(map(legacy.retold_record, group)), legacy.room_pointer(group))
               for rid, group in legacy.pointer_rooms(snapshot.story).items()],
        "F4": [(r["id"], mv._retold(r, r.get("standing", False)), mv._retold(r, True))
               for r in [*room.get("legacy", ()), *room.get("under_parts", ())]],
        "F7p": [(prefix + r["address"], mv._row_opening(r, "words") if prefix else r["line"], render(r, "words"))
                 for prefix, render in (("", mv._row_opening), ("bare:", mv._row_pointer)) for r in lane1
                 if r["kind"] != "ouroboros" and not r.get("owner_side", r["kind"] == "human")],
        "F7": [(r["address"], r["line"], r["short"]) for r in room.get("project_words") or () if r.get("address")]
              + block["F7"] + [(r["address"], r["line"], mv._row_pointer(r, "words")) for r in lane1
                              if r.get("owner_side", r["kind"] == "human") and not r.get("immune")]
              + [(ident, row["line"], "") for ident, row in mv.story_quote_rows(snapshot).items()],
        "F5l": [(r["id"], "\n".join(mv._page_lines(r)), mv._page_short(r, lead=True)) for r in story
                 if r.get("kind") in ("page", "part") and r.get("text", "").strip() and not r.get("told_by")],
        "F5": [(r["id"], "\n".join(mv._page_lines(r)), mv._page_short(r)) for r in story
                if r.get("kind") not in ("legacy", "account") and not r.get("told_by")],
        "F7a": [(str(room['room_id']), '', '')] if lane1 else [],
        "F5a": [('story', '', '')] if snapshot.story else [],
        "F8": [(n['id'], '\n'.join(mv._note_lines([n])), mv._note_pointer(n)) for n in room.get('notes', ())],
        "F8a": [(str(room['room_id']), '', '')] if room.get('notes') else [],
    }
    listed = [(step, ident, whole, short) for step in LADDER for ident, whole, short in steps[step]
              if step in ("F1", "F5a", "F7a", "F8", "F8a", "F6a") or len(short) < len(whole)]
    stages = {(step, ident): stage_of(snapshot, step, ident) for step, ident, *_ in listed}
    object.__setattr__(snapshot, "_form_stages", stages)
    order = {stage: n for n, stage in enumerate(STAGES)}
    return sorted(listed, key=lambda e: order[stages[e[0], e[1]]])
