"""A capture that leaves the cold set unread shows exactly what a pass over the whole tail shows (``memory_tail_cold``).

A start reads the stored facts of every chat row after the frontier to find the open ones. The cold set names the rows
no reader of a capture uses while every input of their rooms stands still; any mark that moved makes that start read
the whole tail, as before, and compose the set anew. The oracle: after every change that can move a row's rooms, seals
or use (a new row, a binding, a project, a task changed, removed or admitted late, a restored Reply target, a page
published, accepted, rejected or corrected, a chat or progress text revised, the tail filled again, the index rebuilt;
and in places: a room that becomes one, a delivery of the same words as an earlier sealed row, a place's page accepted or
rejected, my human's account confirmed or withdrawn), the views of the normal path equal, byte for byte, the views of a
whole-tail pass of the same process state: in the
process that lived through the changes (warm), in one without its memos (fresh), and in a real interpreter at the end
of each chain. A warm and a fresh process may still differ from each other where the base code does (a revised Reply
target); that is not this set's to decide. The four changes that refuted the first design are named tests, and every
hot rule and mark has one here that fails without it.
"""
from __future__ import annotations

import dataclasses
import datetime
import hashlib
import json
import os
import pathlib
import random
import sqlite3
import subprocess
import sys

import pytest

from ouroboros import chat_chain as cc, chronicle_store as cs, dialogue_store as ds, memory_inventory as mi
from ouroboros import memory_view as mv, places, projects_registry as pr, room_membership as rm, store, task_store
from ouroboros.chat_references import reference_from_row
from ouroboros.chronicle_store import ChronicleStore
from ouroboros.context_timing import measured_sections, section_spans
from ouroboros.project_dialogue import build_owner_message_ref
from ouroboros.projects_registry import bind_task_to_project, create_project
from ouroboros.task_results import write_task_result
from tests import _memory_inventory_shared as shared

try:  # the oracle lands before the set: until then both paths read the whole tail and the set never acts
    from ouroboros import memory_tail_cold as cold
except ImportError:
    cold = None

LIGHT = {"kind": "helper", "route": "configured-light"}
TASK = {"id": "turn0001", "chat_id": 1}
PLACE = 777
PLACE2 = 778  # a room that becomes a place along the chain
LONG = 779  # a place with more sealed rows than a walk's first page
_START = datetime.datetime(2026, 11, 10, tzinfo=datetime.timezone.utc)
_PROVENANCE = {"binding_id": "p1", "place_id": "p1", "conversation_key": "k", "provider": "slack"}
_TRANSPORT = {"provider": "slack", "account_id": "A", "conversation_id": "C7", "thread_id": "",
              "conversation": {"title": "general"}, "actor": {"display_name": "Ann", "platform_actor_id": "U7"}}
_TRANSPORT2 = {**_TRANSPORT, "conversation_id": "C8", "conversation": {"title": "random"},
               "actor": {"display_name": "Bob", "platform_actor_id": "U8"}}


def _ts(minute: int) -> str:
    return (_START + datetime.timedelta(minutes=minute)).isoformat()


def _writer(task: str) -> dict:
    """A page author whose later terminal rows its page links (``memory_sealing._terminal_writer``)."""
    return {"kind": "mind", "task_id": task, "focus": {"role": "root", "task_id": task}}


def _forget_process_memos() -> None:
    """What a fresh process holds: nothing any module remembered for this one."""
    for memo in (pr._TASK_LINEAGE, pr._ROOM_ANSWERS, pr._THREAD_LENS_CACHE, rm._SAVED_TARGETS, mi._SPAN_ROWS,
                 mi._UNITS, mi._RETELLING_CHARS, mv._STORIES, getattr(pr, "_LINEAGE_SEEN", {})):
        memo.clear()


@pytest.fixture(autouse=True)
def _isolated(monkeypatch, tmp_path):
    monkeypatch.setenv("OUROBOROS_DATA_DIR", str(tmp_path))
    _forget_process_memos()
    yield
    _forget_process_memos()


# --- the oracle -------------------------------------------------------------------------------------------

def _specs(rooms: dict) -> dict:
    """The focuses that read the whole tail: Main, a wake, a place and a Project."""
    return {"main": ("integrator", "1"), "wake": ("consciousness", "1"), "presence": ("presence", str(PLACE)),
            "project": ("integrator", str(rooms["alpha"]))}


def _views(root: pathlib.Path, rooms: dict) -> dict:
    """Each focus's snapshot and both texts, and the open rows with everything they collect, as JSON values."""
    out = {}
    for name, (role, room) in _specs(rooms).items():
        view = mv.capture_memory_view(root, TASK, dataclasses.replace(mv.ROLE_DEFAULTS[role], room_id=room))
        out[name] = [mv.snapshot_json(view), mv.render_story(view), mv.render_room(view)]
    gaps, quiet = {}, {}
    sealed = mi.PlaceRows() if hasattr(mi, "PlaceRows") else {}
    opened = mi.open_rows_by_room(root, gaps=gaps, quiet=quiet, sealed=sealed)
    if not isinstance(sealed, dict):  # step 3a: a place's sealed rows are read for the rooms a reader asks
        sealed = _every_sealed(sealed, {*opened, str(PLACE), str(PLACE2), *(str(chat) for chat in rooms.values())})
    out["open"] = json.loads(json.dumps([opened, {room: sorted(kinds) for room, kinds in gaps.items()}, quiet, sealed]))
    return out


def _every_sealed(sealed, rooms) -> dict:
    """Every sealed row of each asked place room in append order: a walk that goes to its oldest row."""
    walked = {room: list(reversed(list(reversed(place)))) for room, place in sealed.rooms(sorted(rooms)).items()}
    return {room: rows for room, rows in walked.items() if rows}


def _digest(views: dict) -> str:
    return hashlib.sha256(json.dumps(views, sort_keys=True).encode("utf-8")).hexdigest()


def _whole(monkeypatch, root: pathlib.Path, rooms: dict) -> dict:
    """The reference: the same views from a pass over the whole tail; no set read, none composed."""
    with monkeypatch.context() as patch:
        if cold is not None:
            patch.setattr(cold, "gate", lambda _store: None)
        return _views(root, rooms)


def _first_difference(got: dict, want: dict) -> str:
    for name in sorted(set(got) | set(want)):
        a, b = json.dumps(got.get(name), sort_keys=True), json.dumps(want.get(name), sort_keys=True)
        if a != b:
            at = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
            return f"{name} at {at}: normal {a[max(0, at - 160):at + 160]!r} / whole {b[max(0, at - 160):at + 160]!r}"
    return ""


def _same(monkeypatch, root: pathlib.Path, rooms: dict) -> dict:
    """The normal path equals the whole tail in this process (warm), then without its memos (fresh)."""
    warm, reference = _views(root, rooms), _whole(monkeypatch, root, rooms)
    assert warm == reference, "warm: " + _first_difference(warm, reference)
    _forget_process_memos()
    fresh = _views(root, rooms)
    _forget_process_memos()
    reference = _whole(monkeypatch, root, rooms)
    assert fresh == reference, "fresh: " + _first_difference(fresh, reference)
    return reference


def _child(root: pathlib.Path, rooms: dict, *, whole: bool) -> str:
    """The digest a real fresh interpreter computes over the same views: the normal path, or the whole tail."""
    repo = pathlib.Path(__file__).resolve().parents[1]
    code = ("import json, pathlib, sys\n"
            "from tests import test_memory_tail_cold as t\n"
            "if sys.argv[3] == 'whole' and t.cold is not None:\n"
            "    t.cold.gate = lambda _store: None\n"
            "print(t._digest(t._views(pathlib.Path(sys.argv[1]), json.loads(sys.argv[2]))))\n")
    env = {**os.environ, "OUROBOROS_DATA_DIR": str(root), "PYTHONPATH": str(repo), "PYTHONDONTWRITEBYTECODE": "1"}
    done = subprocess.run([sys.executable, "-c", code, str(root), json.dumps(rooms), "whole" if whole else "normal"],
                          capture_output=True, text=True, encoding="utf-8", env=env, cwd=str(root), timeout=600)
    assert done.returncode == 0, done.stderr[-4000:]
    return done.stdout.strip().splitlines()[-1]


def _in_use(root: pathlib.Path) -> bool:
    """Whether a capture of the whole tail stands on the stored cold set (its marks compared in its own read)."""
    with cs.capture(root, tail="all") as derived:
        return bool(derived and derived.get("tail_cold"))


def _work(root: pathlib.Path, rooms: dict, name: str = "main") -> dict:
    """One focus's capture, measured: the counted work of its receipt."""
    role, room = _specs(rooms)[name]
    with measured_sections():
        mv.capture_memory_view(root, TASK, dataclasses.replace(mv.ROLE_DEFAULTS[role], room_id=room))
        return section_spans()["work"]


# --- small worlds -----------------------------------------------------------------------------------------

def _append(root: pathlib.Path, row: dict) -> ds.Receipt:
    return ds.append(root, "chat", row, emit_live=False)


def _fill(root: pathlib.Path) -> None:
    assert mi.fill_tail(root, lambda: False)["state"] == "done"


def _settled(root: pathlib.Path) -> dict:
    rooms = shared.world(root)
    _fill(root)
    return rooms


def _entry(root: pathlib.Path, text: str, nth: int = 0) -> tuple:
    """``(address, row_meta, pos)`` of the ``nth`` row with these words."""
    return [(address, mi.row_meta(row), pos) for address, row, pos in cc.iter_rows(root) if row.get("text") == text][nth]


def _last_of(root: pathlib.Path, **facts) -> tuple:
    """The newest row with these facts (a terminal row has no words to find it by)."""
    return [(address, mi.row_meta(row), pos) for address, row, pos in cc.iter_rows(root)
            if all(row.get(key) == value for key, value in facts.items())][-1]


def _seal(root: pathlib.Path, item: tuple, room, author: dict = shared.MIND) -> str:
    result = ChronicleStore(root).publish_page(room_id=str(room), text="A page.", author=author,
                                               covers={"rows": [item[0]["row_sha256"]], "stream_span": [item[2], item[2]]})
    assert result.ok, result.reason
    return result.record["id"]


def _open_in(root: pathlib.Path, room) -> list:
    return [entry[2] for entry in mi.open_rows_by_room(root).get(str(room), ())]


def _compose(root: pathlib.Path) -> None:
    """A whole pass of the normal path stores the set; the next capture stands on it."""
    mi.open_rows_by_room(root)
    assert _in_use(root), "a whole pass composed no set that stands"


def _texts(views: dict) -> str:
    return json.dumps(views, ensure_ascii=False)


def _cold_positions(root: pathlib.Path) -> set:
    """The positions the stored set leaves unread."""
    with sqlite3.connect(str(ChronicleStore(root).index_path)) as db:
        return {pos for (pos,) in db.execute("SELECT pos FROM tail_cold")}


# --- the chain of changes ---------------------------------------------------------------------------------

class Chain:
    """One installation with every class of row the room rule and the views read, and the changes named above."""

    def __init__(self, root: pathlib.Path, seed: int):
        self.root, self.rng, self.minute = root, random.Random(seed), 0
        self.rooms = shared.world(root)
        self.alpha, self.beta = self.rooms["alpha"], self.rooms["beta"]
        self.projects = ["alpha", "beta"]
        self.tasks = {"free1": 1, "free2": 1, "kid1": 1, "free3": self.alpha, "writer1": 1, "writer2": 1, "q1": PLACE}
        for task, chat in self.tasks.items():
            write_task_result(root, task, "completed", result="done", chat_id=chat)
        store.update_task(root, "kid1", lambda doc: {**doc, "parent_task_id": "free1", "root_task_id": "free1"})
        self.unknown, self.unbound = ["ghost1", "ghost2"], ["free1", "free2", "kid1", "free3", "writer2"]
        self.drafts, self.candidates, self.pages, self.targets, self.owners = [], [], [], [], []
        places.connect_place(root, connector="bridge", provider="slack", account_id="A", place_id="p1", label="Team")
        self.confirmed = False  # whether my human's account (Ann's) is confirmed in that place
        self._seed_targets()
        self._seed_second_room()
        for _ in range(14):
            self.row()
        _fill(root)
        for room, rows in sorted(mi.open_rows_by_room(root).items()):  # most of the history sealed, as on a lived install
            self._publish(room, sorted(rows, key=lambda entry: entry[2])[:-1], "mind")

    def ts(self) -> str:
        self.minute += 1
        return _ts(self.minute)

    def _place_row(self, direction: str, **extra) -> dict:
        return shared.msg(self.ts(), f"place word {self.minute % 3}", chat_id=PLACE, direction=direction,
                          source="presence:slack", transport=_TRANSPORT, presence_provenance=_PROVENANCE, **extra)

    def _seed_targets(self) -> None:
        """Old-style Replies (no ``target_home``) whose targets do not exist yet: the rule reads the dialogue store
        for their rooms (``room_membership._saved_target``); a later step brings a target back."""
        for task in ("bound", "free1"):
            target = shared.msg(self.ts(), f"the answer of {task}", direction="out", type="task_result", task_id=task)
            reference = {key: value for key, value in reference_from_row(target).items() if key != "target_home"}
            _append(self.root, shared.msg(self.ts(), "an old reply", reply_to=reference, client_message_id=f"reply-{task}"))
            self.targets.append(target)

    def _seed_second_room(self) -> None:
        """A room that is no place yet: arrivals over a transport and a word of mine, sealed with the rest below;
        ``become_place`` makes it one, and its earlier rows then say who of ours spoke there and what came again."""
        for n in range(3):
            _append(self.root, shared.msg(self.ts(), f"second word {n % 2}", chat_id=PLACE2, transport=_TRANSPORT2,
                                          client_message_id=f"s{n}"))
        _append(self.root, shared.msg(self.ts(), "my word there", chat_id=PLACE2, direction="out", task_id="free2"))

    def row(self) -> None:
        kind = self.rng.choice(("main_in", "main_out", "alpha_in", "alpha_out", "beta_in", "place_in", "place_out",
                                "transport_in", "quiet", "terminal", "kid", "kid_bare", "ghost", "reply_new",
                                "place_summary", "second_in"))
        ts, n, task = self.ts(), self.minute, self.rng.choice(sorted(self.tasks))
        if kind == "main_in":
            row = shared.msg(ts, f"owner word {n}", client_message_id=f"m{n}")
            self.owners.append(row)
        elif kind == "main_out":
            row = shared.msg(ts, f"my word {n}", direction="out", task_id=task)
        elif kind == "alpha_in":
            row = shared.msg(ts, f"alpha word {n}", chat_id=self.alpha, client_message_id=f"a{n}")
        elif kind == "alpha_out":
            row = shared.msg(ts, f"alpha mine {n}", chat_id=self.alpha, direction="out", task_id=task)
        elif kind == "beta_in":
            row = shared.msg(ts, f"beta word {n % 2}", chat_id=self.beta, client_message_id=f"b{n}")
        elif kind == "place_in":
            row = self._place_row("in", client_message_id=f"ev{n}")
        elif kind == "place_out":
            row = self._place_row("out", task_id=task)
        elif kind == "transport_in":  # an arrival with no place stamp: the room stops being a place while it is last
            row = shared.msg(ts, f"place word {n % 3}", chat_id=PLACE, transport=_TRANSPORT, client_message_id=f"t{n}")
        elif kind == "second_in":  # the same in the second room
            row = shared.msg(ts, f"second word {n % 2}", chat_id=PLACE2, transport=_TRANSPORT2, client_message_id=f"s{n}")
        elif kind == "quiet":
            row = shared.msg(ts, "", direction="out", type="task_summary", task_id=self.rng.choice(("q1", task)),
                             chat_id=self.rng.choice((1, PLACE)), status="completed", wake_silent=True,
                             quiet_effective=True)
        elif kind == "terminal":
            row = shared.msg(ts, "", direction="system", type="task_summary", task_id=self.rng.choice(("writer1", "writer2")),
                             status="completed", result_ref={"reader": "get_task_result", "task_id": task})
        elif kind == "kid":
            row = shared.msg(ts, f"child word {n}", direction="out", task_id="kid1", subagent_task_id="kid1",
                             parent_task_id="free1", root_task_id="free1", delegation_role="subagent")
        elif kind == "kid_bare":  # the same task's row without its lineage fields: the rule reads the store for them
            row = shared.msg(ts, f"child host word {n}", direction="out", task_id="kid1")
        elif kind == "ghost":
            row = shared.msg(ts, f"ghost word {n}", direction="out", task_id=self.rng.choice(("ghost1", "ghost2")))
        elif kind == "reply_new":
            previous = [entry.row for entry in self._entries()][-1]
            row = shared.msg(ts, f"reply word {n}", reply_to=reference_from_row(previous), client_message_id=f"r{n}")
        else:  # the Done row of work a place started: a place row that may stand last in its Project
            row = self._place_row("system", type="task_summary", task_id=self.rng.choice(("free3", "kid1")),
                                  status="completed", result_ref={"reader": "get_task_result", "task_id": task})
        _append(self.root, row)

    def _entries(self) -> list:
        return list(ds.iter_rows(self.root, "chat", lower=0, upper=ds.stream_end(self.root, "chat")))

    # -- the changes ---------------------------------------------------------------------------------------

    def bind(self) -> None:
        if self.unbound:
            task = self.unbound.pop(self.rng.randrange(len(self.unbound)))
            bind_task_to_project(self.root, task, self.rng.choice(self.projects), origin={"absent": "post_hoc_unresolved"})

    def bind_origin(self) -> None:
        """A binding that names an owner message in Main as its Project's origin: that row joins the Project."""
        if self.unbound and self.owners:
            task, owner = self.unbound.pop(self.rng.randrange(len(self.unbound))), self.rng.choice(self.owners)
            ref = build_owner_message_ref(chat_id=1, client_message_id=owner["client_message_id"], ts=owner["ts"],
                                          text=owner["text"])
            bind_task_to_project(self.root, task, self.rng.choice(self.projects), origin={"ref": ref, "text": owner["text"]})

    def new_project(self) -> None:
        name = f"p{self.minute}"
        create_project(self.root, name, name=name.upper())
        self.projects.append(name)
        self.minute += 1

    def relineage(self) -> None:
        task, change = self.rng.choice(sorted(self.tasks)), self.rng.choice(("parent", "root", "chat"))

        def mutate(doc: dict) -> dict:
            if change == "parent":
                doc["parent_task_id"] = self.rng.choice(("free2", "writer1", ""))
            elif change == "root":
                doc["root_task_id"] = self.rng.choice(("free2", "writer2", ""))
            else:
                doc["chat_id"] = self.rng.choice((1, self.alpha, self.beta))
            return doc

        store.update_task(self.root, task, mutate)

    def direct_write(self) -> None:
        """An older body's or the derivation job's write of a lineage column, past ``Txn.store_task``."""
        task = self.rng.choice(sorted(self.tasks))
        with store.open_store(self.root, write=True) as txn:
            txn.conn.execute("UPDATE tasks SET chat_id=? WHERE task_id=?", (str(self.rng.choice((1, self.beta))), task))

    def remove_task(self) -> None:
        if len(self.tasks) > 4:
            task = self.rng.choice(sorted(set(self.tasks) - {"writer1", "writer2"}))
            task_store.delete_document(self.root, task)
            del self.tasks[task]
            self.unbound = [name for name in self.unbound if name != task]

    def admit(self) -> None:
        if self.unknown:
            task = self.unknown.pop(self.rng.randrange(len(self.unknown)))
            write_task_result(self.root, task, "completed", result="done", chat_id=self.alpha)
            self.tasks[task] = self.alpha

    def restore_target(self) -> None:
        if self.targets:
            _append(self.root, self.targets.pop(0))

    def seal(self) -> None:
        opened = mi.open_rows_by_room(self.root)
        rooms = sorted(room for room, rows in opened.items() if rows and room.lstrip("-").isdigit())
        if rooms:
            room = self.rng.choice(rooms)
            rows = sorted(opened[room], key=lambda entry: entry[2])
            self._publish(room, rows[:self.rng.randint(1, max(1, len(rows) - 1))],
                          self.rng.choice(("mind", "writer", "draft", "candidate")))

    def _publish(self, room: str, rows: list, style: str) -> None:
        if not rows or not room.lstrip("-").isdigit():
            return
        covers = {"rows": [entry[0]["row_sha256"] for entry in rows], "stream_span": [rows[0][2], rows[-1][2]]}
        chronicle = ChronicleStore(self.root)
        if style == "mind":
            result = chronicle.publish_page(room_id=room, text="A page.", covers=covers, author=shared.MIND)
        elif style == "writer":
            result = chronicle.publish_page(room_id=room, text="A page.", covers=covers,
                                            author=_writer(self.rng.choice(("writer1", "writer2"))))
        elif style == "draft":
            result = chronicle.publish_page(room_id=room, text="A draft.", covers=covers, author=LIGHT)
        else:
            result = chronicle.publish_page(room_id=room, text="A candidate.", covers=covers, author=LIGHT,
                                            metadata={"candidate": True})
        if result.ok:
            {"draft": self.drafts, "candidate": self.candidates}.get(style, self.pages).append(result.record["id"])

    def reject(self) -> None:
        if self.drafts:
            assert ChronicleStore(self.root).decide(self.drafts.pop(0), False, shared.MIND, "not this").ok

    def accept(self) -> None:
        if self.candidates:  # refused when another page sealed its rows meanwhile: then nothing changes
            result = ChronicleStore(self.root).decide(self.candidates.pop(0), True, shared.MIND, "adopted")
            assert result.ok or result.reason == "already_sealed", result

    def correct(self) -> None:
        if self.pages:
            chronicle, page = ChronicleStore(self.root), self.rng.choice(self.pages)
            result = chronicle.correct(page, "Said better.", shared.MIND)
            if result.reason == "revision_required":  # a later correction names the revision it corrects
                result = chronicle.correct(page, "Said better again.", shared.MIND,
                                           expected_revision=result.current_revision)
            assert result.ok, result

    def revise_chat(self) -> None:
        entries = [entry for entry in self._entries() if entry.row.get("text")]
        ds.revise_text(self.root, "chat", self.rng.choice(entries[-12:]).seq, f"revised {self.minute}")

    def revise_progress(self) -> None:
        receipt = ds.append(self.root, "progress", {"ts": self.ts(), "text": "progress line", "task_id": "free1",
                                                    "is_progress": True}, emit_live=False)
        ds.revise_text(self.root, "progress", receipt.seq, f"revised progress {self.minute}")

    def fill(self) -> None:
        mi.fill_tail(self.root, lambda: False)

    def become_place(self) -> None:
        """A place's word in the second room: its newest row is a place's now, so its earlier rows count."""
        _append(self.root, shared.msg(self.ts(), f"second word {self.minute % 2}", chat_id=PLACE2, source="presence:slack",
                                      transport=_TRANSPORT2, presence_provenance=_PROVENANCE,
                                      client_message_id=f"bp{self.minute}"))

    def place_repeat(self) -> None:
        """A place delivers again the words of its room's earliest arrival (most likely sealed, and left unread)."""
        room = self.rng.choice((PLACE, PLACE2))
        rows = [entry.row for entry in self._entries() if entry.row.get("chat_id") == room
                and entry.row.get("direction") == "in" and entry.row.get("text")]
        if rows:
            _append(self.root, {**rows[0], "ts": self.ts(), "client_message_id": f"again{self.minute}",
                                "source": "presence:slack", "presence_provenance": _PROVENANCE})

    def seal_place(self) -> None:
        """A draft or a candidate page over a place's open rows: a later ``reject`` or ``accept`` decides it."""
        room = self.rng.choice((str(PLACE), str(PLACE2)))
        rows = sorted(mi.open_rows_by_room(self.root).get(room, ()), key=lambda entry: entry[2])
        self._publish(room, rows[:self.rng.randint(1, max(1, len(rows)))], self.rng.choice(("draft", "candidate")))

    def owner_account(self) -> None:
        """My human's account in the place confirmed, or that confirmation withdrawn: whose words its rows are now."""
        action = "revoke_owner_account" if self.confirmed else "confirm_owner_account"
        places.edit_place(self.root, "p1", action, author="owner", platform_actor_id="U7", account_id="A")
        self.confirmed = not self.confirmed

    def rebuild(self) -> None:
        chronicle = ChronicleStore(self.root)
        chronicle.index_path.unlink()
        assert chronicle.ensure_activated()
        mi.fill_tail(self.root, lambda: False)

    STEPS = ("row", "row", "row", "row", "bind", "bind_origin", "new_project", "relineage", "direct_write", "remove_task",
             "admit", "restore_target", "seal", "seal", "seal", "reject", "accept", "correct", "revise_chat",
             "revise_progress", "fill", "fill", "rebuild", "become_place", "place_repeat", "seal_place", "seal_place",
             "owner_account")

    def step(self) -> str:
        name = self.rng.choice(self.STEPS)
        getattr(self, name)()
        return name


# Together the seeds take every change in STEPS, a place page's acceptance (24) and rejection (29); a place's walks read
# its first page of unread rows two at a time there (``memory_inventory._PlaceReads._NEWEST``), so every read past it,
# every earlier copy below it and every return to the dialogue store happens on these short chains too.
@pytest.mark.parametrize("seed,page", [(1, 2), (2, 2), (5, 2), (24, 2), (29, 2), (1, None), (2, None)])
def test_after_every_change_the_normal_path_equals_the_whole_tail_warm_and_fresh(monkeypatch, tmp_path, seed, page):
    if page is not None and hasattr(mi, "_PlaceReads"):
        monkeypatch.setattr(mi._PlaceReads, "_NEWEST", page)
    chain = Chain(tmp_path, seed)
    _same(monkeypatch, tmp_path, chain.rooms)
    acted, trace = 0, []
    for _ in range(16):
        trace.append(chain.step())
        try:
            _same(monkeypatch, tmp_path, chain.rooms)
        except AssertionError as exc:
            raise AssertionError(f"after {trace}: {exc}") from None
        acted += _in_use(tmp_path)
    assert acted, ("the set never stood along this chain, so it proved nothing", trace)
    assert _child(tmp_path, chain.rooms, whole=False) == _child(tmp_path, chain.rooms, whole=True), trace


# --- the four changes that refuted the first design ---------------------------------------------------------

def test_case1_a_restored_reply_target_brings_the_old_reply_into_its_project(monkeypatch, tmp_path):
    rooms = _settled(tmp_path)
    target = shared.msg(_ts(0), "Restored target.", direction="out", type="task_result", task_id="bound")
    reference = {key: value for key, value in reference_from_row(target).items() if key != "target_home"}
    _append(tmp_path, shared.msg(_ts(1), "Old reply.", reply_to=reference, client_message_id="old-reply"))
    _append(tmp_path, shared.msg(_ts(2), "Later word.", client_message_id="later"))
    _fill(tmp_path)
    reply = _entry(tmp_path, "Old reply.")
    _seal(tmp_path, reply, 1)
    _compose(tmp_path)
    assert reply[2] not in _open_in(tmp_path, rooms["alpha"])
    _append(tmp_path, target)  # its target comes back: no mark moves, the Reply joins the target's Project
    assert _in_use(tmp_path)
    assert reply[2] in _open_in(tmp_path, rooms["alpha"])
    _same(monkeypatch, tmp_path, rooms)


def test_case2_a_task_whose_admitted_chat_changes_moves_its_sealed_row_warm_and_fresh(monkeypatch, tmp_path):
    rooms = _settled(tmp_path)
    write_task_result(tmp_path, "lineage", "running", chat_id=1, title="lineage")
    _append(tmp_path, shared.msg(_ts(1), "Lineage row.", direction="out", task_id="lineage"))
    _append(tmp_path, shared.msg(_ts(2), "Later word.", client_message_id="later"))
    _fill(tmp_path)
    row = _entry(tmp_path, "Lineage row.")
    _seal(tmp_path, row, 1)  # Main is its only room
    _compose(tmp_path)
    store.update_task(tmp_path, "lineage", lambda doc: {**doc, "chat_id": rooms["alpha"]})
    assert not _in_use(tmp_path)  # the task store moved the mark in its own write
    assert row[2] in _open_in(tmp_path, rooms["alpha"])  # and this process reads the lineage again, as a fresh one does
    _same(monkeypatch, tmp_path, rooms)


def test_case3_a_room_that_becomes_a_place_reads_its_earlier_sealed_rows(monkeypatch, tmp_path):
    rooms = _settled(tmp_path)
    _append(tmp_path, shared.msg(_ts(1), "Same words.", chat_id=PLACE, client_message_id="early"))
    _append(tmp_path, shared.msg(_ts(2), "Mine there.", chat_id=PLACE, direction="out"))
    _append(tmp_path, shared.msg(_ts(3), "Latest main.", client_message_id="latest"))
    _fill(tmp_path)
    for room in (PLACE, 1):  # sealed in both its rooms (a row of a room that is no place stands in Main too)
        _seal(tmp_path, _entry(tmp_path, "Same words."), room)
    _compose(tmp_path)
    _append(tmp_path, shared.msg(_ts(4), "Same words.", chat_id=PLACE, type="presence_delivery",
                                 presence_provenance={"provider": "slack"}))  # the room's newest row is a place's now
    assert _in_use(tmp_path)
    work = _work(tmp_path, rooms)  # its earlier sealed rows are read by its chat when the block asks, not the whole tail
    assert work.get("tail_whole", 0) == 0 and work.get("rows_asked", 0) > 0, work
    assert work["rows_read"] == work["rows_open"] + work["rows_kept"] + work.get("rows_stale", 0), work  # law L7d
    views = _same(monkeypatch, tmp_path, rooms)
    assert "delivered again" in _texts(views), "the place's later delivery no longer names its earlier sealed copy"


def test_case4_a_rejection_after_the_marks_are_read_reopens_the_row_in_that_same_capture(monkeypatch, tmp_path):
    rooms = _settled(tmp_path)
    _append(tmp_path, shared.msg(_ts(1), "Rejected row.", client_message_id="rejected"))
    _append(tmp_path, shared.msg(_ts(2), "After it.", client_message_id="after"))
    _fill(tmp_path)
    item = _entry(tmp_path, "Rejected row.")
    draft = _seal(tmp_path, item, 1, author=LIGHT)
    _compose(tmp_path)
    assert item[2] not in _open_in(tmp_path, 1)
    real, fired = cold.gate, []

    def racing(store_):  # the rejection lands after the marks are read and before the snapshot's transaction
        marks = real(store_)
        if not fired:
            fired.append(ChronicleStore(tmp_path).decide(draft, False, shared.MIND, "between the marks and the read").ok)
        return marks

    monkeypatch.setattr(cold, "gate", racing)
    with cs.capture(tmp_path, tail="all") as derived:
        assert fired == [True] and not derived.get("tail_cold")  # the snapshot compared the journal in its own read
        assert item[2] in _open_in(tmp_path, 1)
    monkeypatch.setattr(cold, "gate", real)
    _same(monkeypatch, tmp_path, rooms)
    entered, compared, index, stands = [], [], ChronicleStore._index, cold.stands
    monkeypatch.setattr(ChronicleStore, "_index", lambda self: entered.append(1) or index(self))
    monkeypatch.setattr(cold, "stands", lambda db, gate_: compared.append(len(entered)) or stands(db, gate_))
    with cs.capture(tmp_path, tail="all"):
        assert entered == [1] and compared == [1]  # one read transaction: the marks, the seals and the rows


# --- each hot rule ------------------------------------------------------------------------------------------

def _scribe(tmp_path: pathlib.Path) -> tuple:
    """A task whose terminal row is sealed in Main by another page, after a Main row its own page may cover."""
    rooms = _settled(tmp_path)
    write_task_result(tmp_path, "scribe", "completed", result="done", chat_id=1)
    _append(tmp_path, shared.msg(_ts(1), "Before the finish.", client_message_id="before"))
    _append(tmp_path, shared.msg(_ts(2), "", direction="system", type="task_summary", task_id="scribe",
                                 status="completed", result_ref={"reader": "get_task_result", "task_id": "scribe"}))
    _append(tmp_path, shared.msg(_ts(3), "Latest main.", client_message_id="latest"))
    _fill(tmp_path)
    finish = _last_of(tmp_path, type="task_summary", task_id="scribe")
    _seal(tmp_path, finish, 1)
    return rooms, _entry(tmp_path, "Before the finish."), finish


def test_a_page_writers_later_terminal_rows_stay_read(monkeypatch, tmp_path):
    rooms, before, _finish = _scribe(tmp_path)
    _seal(tmp_path, before, 1, author=_writer("scribe"))  # the page links its writer's later finish below it
    _compose(tmp_path)
    views = _same(monkeypatch, tmp_path, rooms)
    assert "Later terminal rows of task scribe" in _texts(views)


def test_a_new_writers_page_reads_the_whole_tail_when_its_finish_was_left_unread(monkeypatch, tmp_path):
    rooms, before, _finish = _scribe(tmp_path)
    _compose(tmp_path)  # the scribe wrote no page yet: its finish is cold
    _seal(tmp_path, before, 1, author=_writer("scribe"))
    assert not _in_use(tmp_path)  # a page of a writer the set left a terminal row of
    views = _same(monkeypatch, tmp_path, rooms)
    assert "Later terminal rows of task scribe" in _texts(views)


def test_a_sealed_quiet_finish_still_considers_the_words_before_it(monkeypatch, tmp_path):
    rooms = _settled(tmp_path)
    write_task_result(tmp_path, "q1", "completed", result="done", chat_id=PLACE)
    place = {"source": "presence:slack", "transport": _TRANSPORT, "presence_provenance": _PROVENANCE}
    _append(tmp_path, shared.msg(_ts(1), "A person asks.", chat_id=PLACE, client_message_id="ask", **place))
    _append(tmp_path, shared.msg(_ts(2), "", chat_id=PLACE, direction="out", type="task_summary", task_id="q1",
                                 status="completed", wake_silent=True, quiet_effective=True))
    _append(tmp_path, shared.msg(_ts(3), "A person adds.", chat_id=PLACE, client_message_id="adds", **place))
    # The room's newest row is no place's: none of its sealed rows is kept read for being a place's.
    _append(tmp_path, shared.msg(_ts(4), "A plain word.", chat_id=PLACE, transport=_TRANSPORT, client_message_id="plain"))
    _fill(tmp_path)
    finish = _last_of(tmp_path, type="task_summary", task_id="q1")
    for room in (PLACE, 1):  # sealed in both its rooms: only the quiet finish rule keeps it read
        _seal(tmp_path, finish, room)
    _compose(tmp_path)
    reference = _same(monkeypatch, tmp_path, rooms)
    ranks = {word["address"]: word["rank"] for group in json.loads(reference["project"][0])["conversations"]
             for room in group["rooms"] if room["room_id"] == str(PLACE) for word in room["words"]}
    asks, adds = (cc.format_address(_entry(tmp_path, text)[0]) for text in ("A person asks.", "A person adds."))
    assert (ranks[asks], ranks[adds]) == (1, 2)  # the quiet finish considered the first, not the second


def test_a_rooms_newest_row_stays_read_so_a_standing_set_needs_no_second_pass(monkeypatch, tmp_path):
    rooms = _settled(tmp_path)
    _append(tmp_path, shared.msg(_ts(1), "A place word.", chat_id=PLACE, type="presence_delivery",
                                 presence_provenance=_PROVENANCE, client_message_id="place"))
    _append(tmp_path, shared.msg(_ts(2), "A plain word.", chat_id=PLACE, transport=_TRANSPORT, client_message_id="plain"))
    _append(tmp_path, shared.msg(_ts(3), "Latest main.", client_message_id="latest"))
    _fill(tmp_path)
    for room in (PLACE, 1):  # the room's newest row, no place's, sealed in both its rooms: only that rule keeps it read
        _seal(tmp_path, _entry(tmp_path, "A plain word."), room)
    _compose(tmp_path)
    work = _work(tmp_path, rooms)
    assert work.get("tail_whole", 0) == 0 and "rows_cold" in work, work
    _same(monkeypatch, tmp_path, rooms)


def test_a_places_sealed_rows_are_read_by_its_room_for_its_repeats(monkeypatch, tmp_path):
    rooms = _settled(tmp_path)
    for minute, text, direction in ((1, "Again words.", "in"), (2, "Again words.", "in"), (3, "Mine there.", "out")):
        _append(tmp_path, shared.msg(_ts(minute), text, chat_id=PLACE, direction=direction, source="presence:slack",
                                     presence_provenance=_PROVENANCE, **({"client_message_id": f"e{minute}"}
                                                                         if direction == "in" else {})))
    _append(tmp_path, shared.msg(_ts(4), "Latest main.", client_message_id="latest"))
    _fill(tmp_path)
    _seal(tmp_path, _entry(tmp_path, "Again words."), PLACE)  # the earlier copy only
    _compose(tmp_path)
    assert _in_use(tmp_path) and _entry(tmp_path, "Again words.")[2] in _cold_positions(tmp_path)
    work = _work(tmp_path, rooms)
    assert work.get("tail_whole", 0) == 0 and work.get("rows_asked", 0) > 0, work
    views = _same(monkeypatch, tmp_path, rooms)
    assert "delivered again" in _texts(views)


def _quiet_place(tmp_path: pathlib.Path, room: int, texts: tuple) -> None:
    place = {"source": "presence:slack", "transport": _TRANSPORT, "presence_provenance": _PROVENANCE}
    for n, words in enumerate(texts):
        _append(tmp_path, shared.msg(_ts(10 + n), words, chat_id=room, client_message_id=f"{room}-{n}", **place))


def test_a_place_no_view_shows_leaves_its_sealed_rows_unread(monkeypatch, tmp_path):
    rooms = _settled(tmp_path)
    _quiet_place(tmp_path, PLACE2, ("Unshown one.", "Unshown two.", "Unshown three."))
    _append(tmp_path, shared.msg(_ts(20), "Latest main.", client_message_id="latest"))
    _fill(tmp_path)
    for words in ("Unshown one.", "Unshown two.", "Unshown three."):  # no open row, note or work: no view shows it
        _seal(tmp_path, _entry(tmp_path, words), PLACE2)
    _compose(tmp_path)
    assert {_entry(tmp_path, words)[2] for words in ("Unshown one.", "Unshown two.")} <= _cold_positions(tmp_path)
    asked, real = [], cold.rows_of
    monkeypatch.setattr(cold, "rows_of", lambda store_, held, chats, *more: asked.append(sorted(chats)) or real(store_, held, chats, *more))
    for name in ("main", "wake", "presence", "project"):
        work = _work(tmp_path, rooms, name)
        assert work.get("tail_whole", 0) == 0 and "rows_cold" in work, (name, work)
    assert all(PLACE2 not in chats for chats in asked), asked
    _same(monkeypatch, tmp_path, rooms)


def test_a_place_row_whose_chat_id_sqlite_would_read_otherwise_is_read_by_its_room(monkeypatch, tmp_path):
    """``int("7_77")`` is 777, as the room rule reads it (``chat_chain._row_chat_id``); ``CAST`` in SQLite says 7."""
    rooms = _settled(tmp_path)
    place = {"source": "presence:slack", "transport": _TRANSPORT, "presence_provenance": _PROVENANCE}
    _append(tmp_path, shared.msg(_ts(1), "Odd key words.", chat_id="7_77", client_message_id="odd", **place))
    _append(tmp_path, shared.msg(_ts(2), "Mine there.", chat_id=PLACE, direction="out", task_id="free1"))
    _append(tmp_path, shared.msg(_ts(3), "Latest main.", client_message_id="latest"))
    _fill(tmp_path)
    for words in ("Odd key words.", "Mine there."):
        for room in (PLACE, 1):  # wherever the rule puts it: left unread only when no room holds it open
            _seal(tmp_path, _entry(tmp_path, words), room)
    _compose(tmp_path)
    assert _entry(tmp_path, "Odd key words.")[2] in _cold_positions(tmp_path)
    _append(tmp_path, shared.msg(_ts(4), "Odd key words.", chat_id=PLACE, client_message_id="again", **place))
    work = _work(tmp_path, rooms)
    assert work.get("tail_whole", 0) == 0 and work.get("rows_asked", 0) >= 1, work  # "Mine there." is the room's newest
    views = _same(monkeypatch, tmp_path, rooms)
    assert "delivered again" in _texts(views), "the earlier copy under the odd key was not read with its room"


def test_a_set_composed_anew_after_a_capture_read_it_answers_that_capture_from_the_dialogue_store(monkeypatch, tmp_path):
    _settled(tmp_path)
    _quiet_place(tmp_path, PLACE, ("Early place.", "Middle place.", "Open place."))
    _append(tmp_path, shared.msg(_ts(20), "Latest main.", client_message_id="latest"))
    _fill(tmp_path)
    for words in ("Early place.", "Middle place."):
        _seal(tmp_path, _entry(tmp_path, words), PLACE)
    _compose(tmp_path)
    with monkeypatch.context() as patch:
        patch.setattr(cold, "gate", lambda _store: None)
        reference = mi.PlaceRows()
        mi.open_rows_by_room(tmp_path, sealed=reference)
        want = _every_sealed(reference, [PLACE])
    assert [entry[2] for entry in want[str(PLACE)]] == [_entry(tmp_path, w)[2] for w in ("Early place.", "Middle place.")]
    with cs.capture(tmp_path, tail="all") as derived:
        assert derived.get("tail_cold")
        sealed = mi.PlaceRows()
        mi.open_rows_by_room(tmp_path, sealed=sealed)
        with sqlite3.connect(str(ChronicleStore(tmp_path).index_path)) as db:  # another process composed it anew
            marks = json.loads(db.execute("SELECT value FROM state WHERE key='tail_cold'").fetchone()[0])
            db.execute("UPDATE state SET value=? WHERE key='tail_cold'", (json.dumps({**marks, "sequence": -1}),))
            db.execute("DELETE FROM tail_cold WHERE chat=?", (PLACE,))  # under its marks the place's rows stay read
        got = _every_sealed(sealed, [PLACE])
    assert json.loads(json.dumps(got)) == json.loads(json.dumps(want))


def test_a_row_of_a_places_chat_whose_work_joined_a_project_is_not_read_as_the_places(monkeypatch, tmp_path):
    """The set keeps a row's chat, never its rooms: a row of a place's chat that the rule moved to a Project (its work
    bound there) is read with that chat and decided again by the rule, so the place never takes it as its own."""
    rooms = _settled(tmp_path)
    place = {"source": "presence:slack", "transport": _TRANSPORT, "presence_provenance": _PROVENANCE}
    write_task_result(tmp_path, "moved", "completed", result="done", chat_id=PLACE)
    _append(tmp_path, shared.msg(_ts(1), "Place asks.", chat_id=PLACE, client_message_id="asks", **place))
    _append(tmp_path, shared.msg(_ts(2), "Moved word.", chat_id=PLACE, direction="out", task_id="moved"))
    _append(tmp_path, shared.msg(_ts(3), "Alpha later.", chat_id=rooms["alpha"], client_message_id="alpha-later"))
    _append(tmp_path, shared.msg(_ts(4), "Place newest.", chat_id=PLACE, client_message_id="newest", **place))
    _append(tmp_path, shared.msg(_ts(5), "Latest main.", client_message_id="latest"))
    _fill(tmp_path)
    moved = _entry(tmp_path, "Moved word.")
    for room in (PLACE, 1):  # where it stood before its work joined a Project: those seals stay
        _seal(tmp_path, moved, room)
    bind_task_to_project(tmp_path, "moved", "alpha", origin={"absent": "post_hoc_unresolved"})
    _seal(tmp_path, moved, rooms["alpha"])  # and in that Project: open nowhere now
    _compose(tmp_path)
    assert moved[2] in _cold_positions(tmp_path)
    work = _work(tmp_path, rooms)
    assert work.get("tail_whole", 0) == 0 and work.get("rows_asked", 0) >= 1, work
    _same(monkeypatch, tmp_path, rooms)


def _long_place(tmp_path: pathlib.Path, mine_at) -> tuple:
    """Forty arrivals of a person in a place of its own (no other row of its chat), all sealed, with a delivery of mine
    at ``mine_at`` (from the oldest; None: none), then an open arrival with the oldest one's words: its room shows, and
    asks for its sealed rows; who of ours spoke there is walked down to that open row."""
    rooms = _settled(tmp_path)
    place = {"source": "presence:slack", "transport": _TRANSPORT, "presence_provenance": _PROVENANCE}
    for n in range(40):
        if n == mine_at:
            _append(tmp_path, shared.msg(_ts(n + 1), "My word there.", chat_id=LONG, direction="out", task_id="free1",
                                         type="presence_delivery", **place))
        else:
            _append(tmp_path, shared.msg(_ts(n + 1), f"Words {n}.", chat_id=LONG, client_message_id=f"w{n}", **place))
    _append(tmp_path, shared.msg(_ts(50), "Words 0." if mine_at != 0 else "Words 1.", chat_id=LONG,
                                 client_message_id="again", **place))
    _fill(tmp_path)
    sealed = [_entry(tmp_path, "My word there." if n == mine_at else f"Words {n}.") for n in range(40)]  # no capture yet
    result = ChronicleStore(tmp_path).publish_page(room_id=str(LONG), text="A page.", author=shared.MIND, covers={
        "rows": [entry[0]["row_sha256"] for entry in sealed], "stream_span": [sealed[0][2], sealed[-1][2]]})
    assert result.ok, result.reason
    _compose(tmp_path)
    return rooms, len(sealed)


def test_a_shown_places_walks_read_its_unread_rows_only_as_far_as_they_go(monkeypatch, tmp_path):
    """My word among its newest rows: who of ours spoke there is known within the first page; the oldest arrival's
    earlier copy is found by its words; the rows between are never read."""
    rooms, sealed = _long_place(tmp_path, mine_at=30)
    work = _work(tmp_path, rooms)
    assert work.get("tail_whole", 0) == 0 and 0 < work.get("rows_asked", 0) < sealed, (work, sealed)
    assert work["rows_read"] == work["rows_open"] + work["rows_kept"] + work.get("rows_stale", 0), work  # law L7d
    views = _same(monkeypatch, tmp_path, rooms)
    assert "delivered again" in _texts(views), "the earlier copy below the first page was not found by its words"


def test_a_walk_past_a_places_first_page_reads_the_rest_of_its_chat(monkeypatch, tmp_path):
    rooms, sealed = _long_place(tmp_path, mine_at=0)  # my word is its oldest row: the walk goes past the first page
    work = _work(tmp_path, rooms)
    assert work.get("tail_whole", 0) == 0 and work.get("rows_asked", 0) >= sealed, (work, sealed)
    _same(monkeypatch, tmp_path, rooms)


def test_a_walk_past_a_places_first_page_reads_every_row_between_not_only_the_copies_it_holds(monkeypatch, tmp_path):
    """My word lies between the first page and the oldest arrival, whose earlier copy of the open word is already read:
    the walk reads every row below the page before it goes on, so it finds my word there."""
    rooms, sealed = _long_place(tmp_path, mine_at=3)
    work = _work(tmp_path, rooms)
    assert work.get("tail_whole", 0) == 0 and work.get("rows_asked", 0) >= sealed, (work, sealed)
    _same(monkeypatch, tmp_path, rooms)


def test_a_place_where_none_of_us_spoke_reads_every_sealed_row(monkeypatch, tmp_path):
    """Only reading all of them shows that none of its sealed rows is ours: its room stays silent."""
    rooms, sealed = _long_place(tmp_path, mine_at=None)
    work = _work(tmp_path, rooms)
    assert work.get("tail_whole", 0) == 0 and work.get("rows_asked", 0) >= sealed, (work, sealed)
    _same(monkeypatch, tmp_path, rooms)


def _project_place(tmp_path: pathlib.Path, rooms: dict) -> None:
    """Project alpha holds the same words twice, the first sealed, and then the Done row of work a place started there,
    its newest row: a Project that is a place, whose rows come from many chats."""
    alpha = rooms["alpha"]
    write_task_result(tmp_path, "placed", "completed", result="done", chat_id=alpha)
    bind_task_to_project(tmp_path, "placed", "alpha", origin={"absent": "post_hoc_unresolved"})
    _append(tmp_path, shared.msg(_ts(1), "Alpha twice.", chat_id=alpha, client_message_id="a-1"))
    _append(tmp_path, shared.msg(_ts(2), "Alpha twice.", chat_id=alpha, client_message_id="a-2"))
    _append(tmp_path, shared.msg(_ts(3), "Latest main.", client_message_id="latest"))


def _done_in_place(tmp_path: pathlib.Path, minute: int) -> None:
    _append(tmp_path, shared.msg(_ts(minute), "", chat_id=PLACE, direction="system", type="task_summary", task_id="placed",
                                 status="completed", source="presence:slack", presence_provenance=_PROVENANCE,
                                 result_ref={"reader": "get_task_result", "task_id": "placed"}))


def test_a_project_whose_newest_row_is_a_places_keeps_its_rows_read(monkeypatch, tmp_path):
    rooms = _settled(tmp_path)
    _project_place(tmp_path, rooms)
    _done_in_place(tmp_path, 4)
    _fill(tmp_path)
    _seal(tmp_path, _entry(tmp_path, "Alpha twice."), rooms["alpha"])
    _compose(tmp_path)
    assert _entry(tmp_path, "Alpha twice.")[2] not in _cold_positions(tmp_path)
    views = _same(monkeypatch, tmp_path, rooms)
    assert "delivered again" in _texts(views)


def test_a_project_that_becomes_a_place_reads_the_whole_tail(monkeypatch, tmp_path):
    rooms = _settled(tmp_path)
    _project_place(tmp_path, rooms)
    _fill(tmp_path)
    _seal(tmp_path, _entry(tmp_path, "Alpha twice."), rooms["alpha"])
    _compose(tmp_path)  # alpha is no place yet: its sealed row is left unread
    assert _entry(tmp_path, "Alpha twice.")[2] in _cold_positions(tmp_path)
    _done_in_place(tmp_path, 4)
    work = _work(tmp_path, rooms)
    assert work.get("tail_whole") == 1, work
    views = _same(monkeypatch, tmp_path, rooms)
    assert "delivered again" in _texts(views)


def test_a_row_whose_task_the_store_does_not_hold_yet_stays_read(monkeypatch, tmp_path):
    rooms = _settled(tmp_path)
    gamma = int(create_project(tmp_path, "gamma", name="Gamma")["chat_id"])
    _append(tmp_path, shared.msg(_ts(1), "Ghost row.", direction="out", task_id="ghost"))
    _append(tmp_path, shared.msg(_ts(2), "Later word.", client_message_id="later"))
    _fill(tmp_path)
    row = _entry(tmp_path, "Ghost row.")
    _seal(tmp_path, row, 1)
    _compose(tmp_path)
    write_task_result(tmp_path, "ghost", "completed", result="done", chat_id=gamma)  # admitted late: no mark moves
    assert _in_use(tmp_path)
    assert row[2] in _open_in(tmp_path, gamma)
    _same(monkeypatch, tmp_path, rooms)


def test_a_row_whose_parent_the_store_does_not_hold_yet_stays_read(monkeypatch, tmp_path):
    rooms = _settled(tmp_path)
    write_task_result(tmp_path, "orphan", "completed", result="done", chat_id=1, parent_task_id="elder",
                      root_task_id="elder")
    _append(tmp_path, shared.msg(_ts(1), "Orphan row.", direction="out", task_id="orphan"))
    _append(tmp_path, shared.msg(_ts(2), "Later word.", client_message_id="later"))
    _fill(tmp_path)
    row = _entry(tmp_path, "Orphan row.")
    _seal(tmp_path, row, 1)
    _compose(tmp_path)
    # its parent is admitted late, under the task bound to alpha: no mark moves, the walk reaches that binding now
    write_task_result(tmp_path, "elder", "completed", result="done", chat_id=1, parent_task_id="bound",
                      root_task_id="bound")
    assert _in_use(tmp_path)
    assert row[2] in _open_in(tmp_path, rooms["alpha"])
    _same(monkeypatch, tmp_path, rooms)


# --- each mark ------------------------------------------------------------------------------------------------

def _sealed_main_row(tmp_path: pathlib.Path, task: str = "late") -> tuple:
    rooms = _settled(tmp_path)
    write_task_result(tmp_path, task, "completed", result="done", chat_id=1)
    _append(tmp_path, shared.msg(_ts(1), "Sealed row.", direction="out", task_id=task))
    _append(tmp_path, shared.msg(_ts(2), "Later word.", client_message_id="later"))
    _fill(tmp_path)
    row = _entry(tmp_path, "Sealed row.")
    _seal(tmp_path, row, 1)
    _compose(tmp_path)
    return rooms, row


def test_a_binding_returns_the_whole_tail_to_reading(monkeypatch, tmp_path):
    rooms, row = _sealed_main_row(tmp_path)
    bind_task_to_project(tmp_path, "late", "beta", origin={"absent": "post_hoc_unresolved"})
    assert not _in_use(tmp_path)
    assert row[2] in _open_in(tmp_path, rooms["beta"])
    _same(monkeypatch, tmp_path, rooms)


def test_a_bound_origin_message_returns_the_whole_tail_to_reading(monkeypatch, tmp_path):
    rooms = _settled(tmp_path)
    owner = shared.msg(_ts(1), "Start beta, please.", client_message_id="origin-b")
    _append(tmp_path, owner)
    _append(tmp_path, shared.msg(_ts(2), "Later word.", client_message_id="later"))
    _fill(tmp_path)
    row = _entry(tmp_path, "Start beta, please.")
    _seal(tmp_path, row, 1)
    _compose(tmp_path)
    ref = build_owner_message_ref(chat_id=1, client_message_id="origin-b", ts=owner["ts"], text=owner["text"])
    bind_task_to_project(tmp_path, "starter", "beta", origin={"ref": ref, "text": owner["text"]})
    assert not _in_use(tmp_path)
    assert row[2] in _open_in(tmp_path, rooms["beta"])
    _same(monkeypatch, tmp_path, rooms)


def test_a_new_project_returns_the_whole_tail_to_reading(monkeypatch, tmp_path):
    rooms, _row = _sealed_main_row(tmp_path)
    create_project(tmp_path, "gamma", name="Gamma")
    assert not _in_use(tmp_path)
    _same(monkeypatch, tmp_path, rooms)


def test_every_write_of_a_tasks_lineage_moves_the_mark_again_whatever_statement_writes_it(monkeypatch, tmp_path):
    """Each change of a task's parent, root, role or admitted chat, or its removal, moves the mark in the store's own
    transaction to a value it never held, again and again, whichever statement writes it: ``Txn.store_task``'s REPLACE,
    an older body's REPLACE, a direct UPDATE (the derivation job), a DELETE. SQLite runs a trigger's statements under
    the conflict policy the firing statement names, so a counter seeded by ``INSERT OR IGNORE`` was replaced by its
    first value under every REPLACE: the mark stood still after the first change of a store's life."""
    from ouroboros.store_hot_facts import lineage_writes

    rooms, _row = _sealed_main_row(tmp_path)
    write_task_result(tmp_path, "late2", "completed", result="done", chat_id=1)  # a new task: nothing to move

    def raw(sql: str, *values) -> None:
        with store.open_store(tmp_path, write=True) as txn:
            txn.conn.execute(sql, values)

    writes = (lambda: store.update_task(tmp_path, "late", lambda doc: {**doc, "chat_id": rooms["alpha"]}),
              lambda: store.update_task(tmp_path, "late2", lambda doc: {**doc, "parent_task_id": "free1"}),
              lambda: raw("INSERT OR REPLACE INTO tasks (task_id, status, parent_task_id, root_task_id) "
                          "VALUES ('late', 'completed', 'free1', 'late')"),  # an older body's, without the chat column
              lambda: raw("UPDATE tasks SET chat_id=? WHERE task_id='late2'", str(rooms["beta"])),
              lambda: raw("UPDATE tasks SET chat_id='1' WHERE task_id='late2'"),
              lambda: store.update_task(tmp_path, "late2", lambda doc: {**doc, "delegation_role": "subagent"}),
              lambda: task_store.delete_document(tmp_path, "late"))
    marks = [lineage_writes(tmp_path)]
    for n, write in enumerate(writes):
        _compose(tmp_path)  # a set that stands on the mark as it is
        write()
        marks.append(lineage_writes(tmp_path))
        assert marks[-1] not in marks[:-1] and not _in_use(tmp_path), (n, marks)
    _same(monkeypatch, tmp_path, rooms)


def _older_body_write(root: pathlib.Path, task: str, chat) -> None:
    """What a body before the chat column writes (``Txn.store_task`` then): its document, and a REPLACE of the typed
    row without that column, so the row stays underived and the room rule reads the chat from the body."""
    with store.open_store(root, write=True) as txn:
        document = {**txn.load(task), "chat_id": chat}
        txn.conn.execute("UPDATE task_bodies SET body=? WHERE task_id=?", (store.encode_document(document), task))
        txn.conn.execute("INSERT OR REPLACE INTO tasks (task_id, status, root_task_id) VALUES (?, ?, ?)",
                         (task, document["status"], task))


def test_an_underived_tasks_rewrite_moves_the_mark_whatever_its_body_says(monkeypatch, tmp_path):
    """A task an older body wrote keeps no typed chat: its room follows the chat its body names
    (``projects_registry._BODY_CHAT``), which no trigger can read. Every rewrite of such a row counts as a lineage
    write, or a second write that moves only the body's chat would leave its sealed row out of its new room."""
    rooms = _settled(tmp_path)
    write_task_result(tmp_path, "older", "running", chat_id=1)
    _older_body_write(tmp_path, "older", 1)  # its typed chat is underived from here on
    _append(tmp_path, shared.msg(_ts(1), "Older body words.", direction="out", task_id="older"))
    _append(tmp_path, shared.msg(_ts(2), "Later word.", client_message_id="later"))
    _fill(tmp_path)
    row = _entry(tmp_path, "Older body words.")
    _seal(tmp_path, row, 1)
    _compose(tmp_path)
    assert row[2] in _cold_positions(tmp_path)
    _older_body_write(tmp_path, "older", rooms["alpha"])  # the same older body moves its work to alpha
    assert not _in_use(tmp_path)
    assert row[2] in _open_in(tmp_path, rooms["alpha"])
    _same(monkeypatch, tmp_path, rooms)


def test_a_write_that_keeps_every_tasks_lineage_keeps_the_set(monkeypatch, tmp_path):
    rooms, _row = _sealed_main_row(tmp_path)
    write_task_result(tmp_path, "late", "failed", result="again")  # the same parent, root, role and chat
    write_task_result(tmp_path, "newcomer", "running", chat_id=1)  # a task admitted: nothing named it before
    assert _in_use(tmp_path)
    _same(monkeypatch, tmp_path, rooms)


def test_a_changed_running_code_reads_the_whole_tail(monkeypatch, tmp_path):
    from ouroboros import transcript_prefix

    rooms, _row = _sealed_main_row(tmp_path)
    monkeypatch.setattr(transcript_prefix, "_running_code", lambda: "another-commit")
    assert not _in_use(tmp_path)
    _same(monkeypatch, tmp_path, rooms)


def test_a_revised_and_refilled_text_reads_the_whole_tail(monkeypatch, tmp_path):
    rooms, row = _sealed_main_row(tmp_path)
    ds.revise_text(tmp_path, "chat", row[0]["hint"]["seq"], "Sealed row, revised.")
    _fill(tmp_path)  # the stored facts of the revised store: complete again, under another source
    assert not _in_use(tmp_path)
    assert row[2] in _open_in(tmp_path, 1)  # its new words are under no seal
    _same(monkeypatch, tmp_path, rooms)


def test_marks_an_older_body_left_beside_a_tail_it_projected_anew_do_not_stand(monkeypatch, tmp_path):
    """This body's refill drops the set's marks with the rows it replaces (``ChronicleStore.extend_tail``); an older
    body's keeps marks it does not know. Beside a tail still filling, or one of another source, they never stand."""
    rooms, row = _sealed_main_row(tmp_path)
    chronicle = ChronicleStore(tmp_path)
    with sqlite3.connect(chronicle.index_path) as db:
        marks = db.execute("SELECT value FROM state WHERE key='tail_cold'").fetchone()[0]

    def older_body_kept_them():
        with sqlite3.connect(chronicle.index_path) as db:
            db.execute("INSERT OR REPLACE INTO state VALUES ('tail_cold', ?)", (marks,))

    state = chronicle.tail_state()  # the same source: its last row no longer answers, the prefix is projected anew
    assert chronicle.extend_tail(state, [], {**state, "anchor": [state["anchor"][0], "0" * 64]})
    monkeypatch.setattr(mi, "_FILL_ROWS", row[2] - 9)  # the refill stops after the chunk that holds the cold row
    assert mi.fill_tail(tmp_path, lambda: True)["state"] == "pending"
    older_body_kept_them()
    assert chronicle.tail_state()["source"] == state["source"] and not _in_use(tmp_path)
    _same(monkeypatch, tmp_path, rooms)
    _fill(tmp_path)
    ds.revise_text(tmp_path, "chat", row[0]["hint"]["seq"], "Sealed row, revised.")  # another source, filled whole
    _fill(tmp_path)
    older_body_kept_them()
    assert not _in_use(tmp_path)
    assert row[2] in _open_in(tmp_path, 1)  # its new words are under no seal
    _same(monkeypatch, tmp_path, rooms)


def test_a_rebuilt_index_holds_no_set_until_a_whole_pass(monkeypatch, tmp_path):
    rooms, _row = _sealed_main_row(tmp_path)
    chronicle = ChronicleStore(tmp_path)
    with sqlite3.connect(chronicle.index_path) as db:
        db.execute("PRAGMA user_version=0")  # another body's schema: the next locked open replays the journal
    assert chronicle.ensure_activated()
    _fill(tmp_path)
    assert not _in_use(tmp_path)
    _same(monkeypatch, tmp_path, rooms)
    assert _in_use(tmp_path)


def test_a_tail_projected_anew_under_the_same_source_holds_no_set(monkeypatch, tmp_path):
    rooms, _row = _sealed_main_row(tmp_path)
    chronicle = ChronicleStore(tmp_path)
    state = chronicle.tail_state()  # its last row no longer answers (rewritten in place): the prefix is projected anew
    assert chronicle.extend_tail(state, [], {**state, "anchor": [state["anchor"][0], "0" * 64]})
    _fill(tmp_path)
    assert chronicle.tail_state()["source"] == state["source"] and not _in_use(tmp_path)
    _same(monkeypatch, tmp_path, rooms)
    assert _in_use(tmp_path)


def test_an_index_an_older_body_built_reads_the_whole_tail_once_then_stands(monkeypatch, tmp_path):
    """After an update the index has no set yet (an older body never made its table): the first capture reads the whole
    tail and says so, its locked write makes the table and the set, and the next capture stands on it."""
    rooms, _row = _sealed_main_row(tmp_path)
    chronicle = ChronicleStore(tmp_path)
    with sqlite3.connect(chronicle.index_path) as db:
        db.execute("DELETE FROM state WHERE key='tail_cold'")
        db.execute("DROP TABLE tail_cold")
    assert not _in_use(tmp_path)  # the lock-free read asks no table it may not have
    first = _work(tmp_path, rooms)
    assert first.get("tail_whole") == 1 and "rows_cold" not in first, first
    assert _in_use(tmp_path)
    _same(monkeypatch, tmp_path, rooms)


def test_a_tail_still_filling_composes_no_set(monkeypatch, tmp_path):
    rooms = shared.world(tmp_path)
    monkeypatch.setattr(mi, "_FILL_ROWS", 3)
    assert mi.fill_tail(tmp_path, lambda: True)["state"] == "pending"
    mi.open_rows_by_room(tmp_path)
    assert not _in_use(tmp_path)
    _fill(tmp_path)
    _same(monkeypatch, tmp_path, rooms)
    assert _in_use(tmp_path)


def test_a_project_that_comes_to_hold_a_tasks_admitted_chat_moves_its_rows_warm_and_fresh(monkeypatch, tmp_path):
    """A Project's room is its id's chat (``project_chat_id``): a task admitted in that chat before the Project existed
    joins it when the Project is made. The registry mark moves; this process also answers that task again, as a fresh
    one does, so the set it composes holds what a fresh process would show."""
    from ouroboros.contracts.chat_id_policy import project_chat_id

    rooms = _settled(tmp_path)
    chat = project_chat_id("gamma")
    write_task_result(tmp_path, "early", "completed", result="done", chat_id=chat)
    _append(tmp_path, shared.msg(_ts(1), "Early row.", direction="out", task_id="early"))
    _append(tmp_path, shared.msg(_ts(2), "Later word.", client_message_id="later"))
    _fill(tmp_path)
    row = _entry(tmp_path, "Early row.")
    _seal(tmp_path, row, 1)
    _compose(tmp_path)
    assert row[2] not in _open_in(tmp_path, chat)
    create_project(tmp_path, "gamma", name="Gamma")
    assert not _in_use(tmp_path)
    assert row[2] in _open_in(tmp_path, chat)
    _same(monkeypatch, tmp_path, rooms)


def test_two_rows_of_one_task_with_different_lineage_fields_get_their_own_answers(monkeypatch, tmp_path):
    """The rule's answer is remembered by its whole input (task, parent, root): the order the rows are judged in,
    which a standing set changes, never changes a row's rooms."""
    rooms = _settled(tmp_path)
    write_task_result(tmp_path, "pa", "completed", result="done", chat_id=1)
    bind_task_to_project(tmp_path, "pa", "alpha", origin={"absent": "post_hoc_unresolved"})
    write_task_result(tmp_path, "twin", "completed", result="done", chat_id=1, parent_task_id="ps", root_task_id="ps")
    _append(tmp_path, shared.msg(_ts(1), "Twin with a parent.", direction="out", task_id="twin", parent_task_id="pa",
                                 root_task_id="pa"))
    _append(tmp_path, shared.msg(_ts(2), "Alpha later.", chat_id=rooms["alpha"], client_message_id="a-later"))
    _append(tmp_path, shared.msg(_ts(3), "Twin bare.", direction="out", task_id="twin"))
    _fill(tmp_path)
    _seal(tmp_path, _entry(tmp_path, "Twin with a parent."), rooms["alpha"])
    _compose(tmp_path)
    bare = _entry(tmp_path, "Twin bare.")
    assert bare[2] in _open_in(tmp_path, 1) and bare[2] not in _open_in(tmp_path, rooms["alpha"])
    _same(monkeypatch, tmp_path, rooms)


def test_a_row_a_page_sealed_after_the_set_was_composed_is_read_and_counted_until_a_whole_pass(tmp_path):
    """A page published since the set (no rejection, no writer of a row it left unread) keeps the set standing; the rows
    it sealed were open when the set was composed, so they are read until the next whole pass, and the receipt says so."""
    rooms = _settled(tmp_path)
    _append(tmp_path, shared.msg(_ts(1), "Open then.", client_message_id="then"))
    _append(tmp_path, shared.msg(_ts(2), "Later word.", client_message_id="later"))
    _fill(tmp_path)
    _compose(tmp_path)
    _seal(tmp_path, _entry(tmp_path, "Open then."), 1)
    work = _work(tmp_path, rooms)
    assert work.get("tail_whole", 0) == 0 and "rows_cold" in work, work
    assert work.get("rows_stale") == 1, work  # read: the set named it read while it was open
    assert work["rows_read"] == work["rows_open"] + work["rows_kept"] + work["rows_stale"], work  # law L7d


def test_a_build_receipt_says_whether_the_set_acted_and_what_it_left_unread(tmp_path):
    rooms, row = _sealed_main_row(tmp_path)
    create_project(tmp_path, "gamma", name="Gamma")
    whole = _work(tmp_path, rooms)
    acting = _work(tmp_path, rooms)
    assert whole.get("tail_whole") == 1 and "rows_cold" not in whole, whole
    assert acting.get("tail_whole", 0) == 0 and acting.get("rows_cold", 0) > 0, acting
    assert acting["rows_read"] + acting["rows_cold"] == whole["rows_read"] + acting.get("rows_asked", 0)
    assert acting["rows_read"] == acting["rows_open"] + acting["rows_kept"], acting  # law L7d: open, kept by a rule, new
    _append(tmp_path, shared.msg(_ts(5), "New and sealed.", client_message_id="new-sealed"))  # sealed before a read
    _append(tmp_path, shared.msg(_ts(6), "Newest and open.", client_message_id="new-open"))  # the room's newest
    _seal(tmp_path, _entry(tmp_path, "New and sealed."), 1)
    newer = _work(tmp_path, rooms)
    assert newer.get("rows_new") == 2 and "rows_cold" in newer, newer
    assert newer["rows_read"] == newer["rows_open"] + newer["rows_kept"], newer  # the sealed one is read as new
    ds.revise_text(tmp_path, "chat", row[0]["hint"]["seq"], "Sealed row, revised.")  # the stored facts wait for the fill
    live = _work(tmp_path, rooms)
    assert live.get("tail_whole") == 1 and "rows_cold" not in live, live  # every row was read live: no set acted
