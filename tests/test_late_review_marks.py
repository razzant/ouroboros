"""Late review feedback kept in view as a host mark of its owning room (#1738), end to end.

A real publisher (a plan council reply that settles after its task ended) persists the full
source, then keeps ONE host mark in the room that owns the work; the helper's own tests publish
the record a pre-fold late acceptance settlement wrote, whose marks stay as history.
The next request of that room carries it through the real memory view and context assembly;
another room's request does not; consciousness reads foreign marks by pointer. Showing, reading, a page seal and a fitted window never
release it; only the integrating mind's ``memory_mark(release_ids, reason)`` does, and a replay
never brings it back, also after the index is lost. An unproven room keeps no mark. No test
calls a model or the network.
"""
from __future__ import annotations

import json
import re
import threading
from types import SimpleNamespace

import pytest

from ouroboros import memory_floor
from ouroboros import memory_view as mv
from ouroboros.artifacts import store_actor_source_bytes
from ouroboros.chronicle_store import ChronicleStore, is_late_review_mark
from ouroboros.late_review_marks import LINE_LIMIT, TEXT_LIMIT, late_review_mark_id, publish_late_review
from ouroboros.tools.chronicle import _chronicle_write, _memory_mark, _memory_read
from ouroboros.tools.registry import ToolContext
from tests import _memory_inventory_shared as shared
from tests._memory_view_context import blocks, room_text, world
from tests._store_shared import seed_result

PRESENCE = {"presence": {"binding_id": "b1", "event": {"provider": "telegram", "conversation_id": "c1"}}}
CHILD = {"delegation_role": "subagent", "parent_task_id": "t1", "root_task_id": "t1"}
NANNY = {**CHILD, "configured_subagent": {"route": {"kind": "agent_session"}}}
WAKE = {"usage_category": "consciousness"}
SECTION = "## Late review feedback (host)"


def _source(root, task_id, text):
    data = json.dumps({"review": text}).encode("utf-8")
    return store_actor_source_bytes(root, task_id, category="context_checkpoints", source_id=f"late-{task_id}",
                                    data=data, extension="json"), data


def _record(root, task_id, chat_id, words, **extra):
    ref, _data = _source(root, task_id, words)
    return {"surface": "task_acceptance", "task_id": task_id, "chat_id": chat_id,
            "publication_id": f"acceptance-late:{task_id}", "settled_at": "2026-10-14T05:00:00+00:00",
            "lines": ["On the delivered version of this answer, reviewers later rejected it.",
                      f"- model/x: rejected it — {words}"],
            "subject": {"panel_id": f"panel-{task_id}", "reviewed_revision": "delivered"},
            "source_ref": ref, **extra}


def _late_marks(root):
    return [mark for mark in ChronicleStore(root).active_marks(None) if is_late_review_mark(mark)]


def _ctx(root, task_id, *, chat=1, meta=None, direct=False):
    return ToolContext(repo_dir=root, drive_root=root, task_id=task_id, current_chat_id=chat,
                       task_metadata=dict(meta or {}), is_direct_chat=direct)


def _release(root, mark_id, *, reason="dealt with it", **ctx):
    return json.loads(_memory_mark(_ctx(root, ctx.pop("task_id", "main1"), **ctx), release_ids=[mark_id],
                                   reason=reason))


# --- the helper: room, identity, text ------------------------------------------------------------

def test_a_publication_is_one_host_mark_of_its_proven_room_and_a_replay_changes_nothing(tmp_path):
    rooms = shared.world(tmp_path)  # Main, Projects alpha and beta, task "bound" bound to alpha
    seed_result(tmp_path, "pres1", {"status": "completed", "chat_id": 555, "metadata": PRESENCE})
    cases = {"main1": (1, "1"), "bound": (1, str(rooms["alpha"])), "beta1": (rooms["beta"], str(rooms["beta"])),
             "pres1": (555, "555")}
    for task_id, (chat, room) in cases.items():
        record = _record(tmp_path, task_id, chat, f"defect in {task_id}")
        first = publish_late_review(tmp_path, record)
        assert (first.status, first.room_id) == ("saved", room), (task_id, first)
        assert first.mark_id == late_review_mark_id("task_acceptance", task_id, f"acceptance-late:{task_id}")
        assert publish_late_review(tmp_path, record) == first._replace(status="unchanged")
    marks = {mark["review"]["task_id"]: mark for mark in _late_marks(tmp_path)}
    assert sorted(marks) == sorted(cases)
    main = marks["main1"]
    assert main["author"] == {"kind": "host", "operation": "late_review", "surface": "task_acceptance",
                              "task_id": "main1"}
    assert main["target_ref"]["task_id"] == "main1" and main["target_ref"]["kind"] == "task_source"
    assert main["text"].startswith("Late result review of task main1, settled 2026-10-14T05:00:00+00:00.")
    assert "Reviewed: panel_id=panel-main1; reviewed_revision=delivered" in main["text"]
    assert "- model/x: rejected it — defect in main1" in main["text"]


@pytest.mark.parametrize("chat,meta", [(777, None), (0, None), (None, None), (-5, None), (777, {"presence": {}})])
def test_an_unproven_room_keeps_no_mark_and_never_defaults_to_main(tmp_path, chat, meta):
    shared.world(tmp_path)
    seed_result(tmp_path, "other1", {"status": "completed", **({"chat_id": chat} if chat is not None else {}),
                                     **({"metadata": meta} if meta else {})})
    before = ChronicleStore(tmp_path).observation_snapshot()[1]
    outcome = publish_late_review(tmp_path, _record(tmp_path, "other1", chat, "a real defect"))
    assert outcome.status == "not_activated" and outcome.room_id == ""
    assert "not proven" in outcome.detail and "source_handles/context_checkpoints/late-other1-" in outcome.detail
    assert ChronicleStore(tmp_path).observation_snapshot()[1] == before, "nothing was written"


def test_other_facts_under_one_publication_conflict_and_bad_records_write_nothing(tmp_path):
    shared.world(tmp_path)
    record = _record(tmp_path, "main1", 1, "first words")
    assert publish_late_review(tmp_path, record).status == "saved"
    changed = publish_late_review(tmp_path, {**record, "lines": ["another verdict"]})
    assert changed.status == "conflict" and "identity collision" in changed.detail
    assert [mark["text"].count("first words") for mark in _late_marks(tmp_path)] == [1]
    ref = record["source_ref"]
    for bad in ({**record, "lines": ["", " "]}, {**record, "source_ref": {**ref, "sha256": ""}},
                {**record, "source_ref": {"kind": "review_ledger_record", "path": "state/x.json", "sha256": "ab"}},
                {**record, "subject": {"nested": {"a": 1}}}, {**record, "publication_id": ""}, ["not", "a", "map"]):
        assert publish_late_review(tmp_path, bad).status == "invalid"
    read = {"tool": "read_file", "arguments": {"root": "runtime_data", "path": "state/x.json"}}
    other = {**record, "publication_id": "change-late:1", "source_read": read,
             "source_ref": {"kind": "review_ledger_record", "path": "state/x.json", "sha256": "ab"}}
    assert publish_late_review(tmp_path, other).status == "saved", "a non-task source names its own reader"
    assert len(_late_marks(tmp_path)) == 2


def test_a_store_that_cannot_serve_is_a_failure_the_writer_keeps_never_an_exception(tmp_path, monkeypatch):
    from ouroboros.store import StoreError

    shared.world(tmp_path)
    record = _record(tmp_path, "main1", 1, "words")
    for error in (TimeoutError("chronicle publication lock unavailable"), StoreError("migrating")):
        def refuse(self, records, *, scan_state=None, _error=error):
            raise _error
        monkeypatch.setattr(ChronicleStore, "publish", refuse)
        outcome = publish_late_review(tmp_path, record)
        assert outcome.status == "failed" and type(error).__name__ in outcome.detail
    monkeypatch.undo()
    assert publish_late_review(tmp_path, record).status == "saved"


def test_the_text_is_the_writers_words_bounded_with_a_disclosed_omission(tmp_path):
    shared.world(tmp_path)
    record = _record(tmp_path, "main1", 1, "x")
    record["lines"] = ["verdict"] + [f"- seat {n}: " + "w" * 5_000 for n in range(3)]
    publish_late_review(tmp_path, record)
    text = _late_marks(tmp_path)[0]["text"]
    assert len(text) <= TEXT_LIMIT and "⚠️ OMISSION NOTE: truncated at" in text
    assert f"truncated at {LINE_LIMIT} chars; original length 5010" in text


# --- the consumer: the next request of the owning room --------------------------------------------

def test_the_next_request_of_the_owning_room_carries_it_and_another_room_never_does(tmp_path):
    env, memory, rooms = world(tmp_path)
    root = memory.drive_root
    seed_result(root, "pres1", {"status": "completed", "chat_id": 555, "metadata": PRESENCE})
    marks = {}
    for task_id, chat in (("main1", 1), ("bound", 1), ("pres1", 555)):  # "bound" belongs to Project alpha
        marks[task_id] = publish_late_review(root, _record(root, task_id, chat, f"MAIN-DEFECT-{task_id}")).mark_id
    main_c = blocks(env, memory, {"id": "next-main", "chat_id": 1})[2]
    alpha_c = blocks(env, memory, {"id": "next-alpha", "chat_id": rooms["alpha"]})[2]
    assert SECTION in main_c and "MAIN-DEFECT-main1" in main_c
    assert f"memory_mark(release_ids=['{marks['main1']}'], reason=…)" in main_c
    assert "full source (sha256 " in main_c and "memory_read(source_ref=" in main_c
    assert "MAIN-DEFECT-bound" not in main_c and "MAIN-DEFECT-pres1" not in main_c
    assert SECTION in alpha_c and "MAIN-DEFECT-bound" in alpha_c and "MAIN-DEFECT-main1" not in alpha_c
    assert "MAIN-DEFECT-bound" in alpha_c.split(SECTION, 1)[1].split("\n## ", 1)[0], "in its own section"
    wake = room_text(root, {"id": "wake1", "chat_id": 1, "metadata": WAKE})
    assert "MAIN-DEFECT-main1" in wake
    assert "MAIN-DEFECT-bound" not in wake and "MAIN-DEFECT-pres1" not in wake
    for task in ("bound", "pres1"):
        lines = [line for line in wake.splitlines() if f"Late review of task {task}:" in line]
        assert len(lines) == 1 and f'memory_read(node_id="{marks[task]}")' in lines[0]
        assert f"MAIN-DEFECT-{task}" in _memory_read(_ctx(root, "wake1"), node_id=marks[task])
    assert len(_late_marks(root)) == 3  # projection/read never releases any mark
    presence = room_text(root, {"id": "pt1", "chat_id": 555, "metadata": PRESENCE})
    assert "MAIN-DEFECT-pres1" in presence and "MAIN-DEFECT-main1" not in presence
    beta = room_text(root, {"id": "tb2", "chat_id": rooms["beta"]})
    assert SECTION not in beta
    child = room_text(root, {"id": "kid7", "chat_id": 1, **CHILD})  # the room's marks, as the child already reads
    assert "MAIN-DEFECT-main1" in child and "MAIN-DEFECT-bound" not in child
    helper_section = child.split(SECTION, 1)[1].split("\n## ", 1)[0]
    assert "the integrating mind releases it" in helper_section and "release_ids=" not in helper_section
    assert "until I release it with my reason" in main_c


def test_showing_reading_sealing_and_a_fitted_window_never_release_it(tmp_path):
    from ouroboros import chat_chain, memory_inventory

    shared.world(tmp_path)
    record = _record(tmp_path, "main1", 1, "KEEP-THIS-CRITIQUE")
    mark_id = publish_late_review(tmp_path, record).mark_id
    task = {"id": "next-main", "chat_id": 1}
    for _ in range(2):  # a first show and a second
        assert "KEEP-THIS-CRITIQUE" in room_text(tmp_path, task)
    ctx = _ctx(tmp_path, "root0001")
    assert "KEEP-THIS-CRITIQUE" in _memory_read(ctx, node_id=mark_id)
    assert mark_id in _memory_read(ctx, room_id="1")
    shown = re.search(r"memory_read\(source_ref=(\{.*?\})\)", room_text(tmp_path, task)).group(1)
    source = _memory_read(ctx, source_ref=json.loads(shown))  # the call exactly as the view prints it
    assert f"sha256 {record['source_ref']['sha256'][:12]}" in source and "KEEP-THIS-CRITIQUE" in source
    assert json.loads(shown) == {key: _late_marks(tmp_path)[0]["target_ref"][key]
                                 for key in ("kind", "root", "path", "size", "sha256", "task_id")}
    rows = memory_inventory.open_room_rows(tmp_path, "1")
    sealed = json.loads(_chronicle_write(ctx, kind="page", text="Main so far, sealed.",
                                         covers={"to": chat_chain.format_address(rows[-1][0])}))
    assert sealed["ok"], sealed
    snapshot = mv.capture_memory_view(tmp_path, task, mv.view_spec_for_task(task, tmp_path))
    level = memory_floor.fit_memory_view(snapshot, {"margin": 1, "physical": 1, "budget": 1})
    assert level.addressed, "the window took everything it may take"
    fitted = mv.render_room(snapshot, level)
    assert "KEEP-THIS-CRITIQUE" in fitted and f"release_ids=['{mark_id}']" in fitted and "memory_read(source_ref=" in fitted
    assert [mark["id"] for mark in _late_marks(tmp_path)] == [mark_id]


# --- release: only the integrating mind, with a reason; never resurrected ---------------------------

def test_only_the_integrating_mind_releases_it_with_a_reason(tmp_path):
    shared.world(tmp_path)
    ids = {task: publish_late_review(tmp_path, _record(tmp_path, task, 1, task)).mark_id
           for task in ("m1", "m2", "m3", "m4")}
    for meta in (CHILD, NANNY):
        refused = _release(tmp_path, ids["m1"], task_id="kid1", meta=meta)
        assert refused["released_ids"] == [] and refused["refused"][0]["id"] == ids["m1"]
        assert refused["refused"][0]["reason"] == "not_integrator"
    assert _release(tmp_path, ids["m1"], reason="")["reason"] == "invalid"
    assert ChronicleStore(tmp_path).release_mark(ids["m1"], {"kind": "host", "operation": "late_review"},
                                                 "the host never decides").reason == "not_integrator"
    assert len(_late_marks(tmp_path)) == 4
    for mark_id, ctx in ((ids["m1"], {"direct": True}), (ids["m2"], {"meta": WAKE}),
                         (ids["m3"], {"chat": 555, "meta": PRESENCE}), (ids["m4"], {})):
        released = _release(tmp_path, mark_id, reason="read the critique; fixed in the follow-up", **ctx)
        assert released["ok"] and released["operation"] == "mark_release", (ctx, released)
    assert _late_marks(tmp_path) == []
    # An ordinary mark keeps its rules: a delegated child still releases its own.
    child_ctx = _ctx(tmp_path, "kid1", meta=CHILD)
    note = json.loads(_chronicle_write(_ctx(tmp_path, "root0001"), kind="note", text="a note"))
    own = json.loads(_memory_mark(child_ctx, text="mine", node_id=note["node_id"]))
    assert json.loads(_memory_mark(child_ctx, release_ids=[own["mark_id"]], reason="done"))["ok"]


def test_a_released_mark_stays_released_across_replay_restart_and_index_loss(tmp_path):
    shared.world(tmp_path)
    record = _record(tmp_path, "main1", 1, "ONCE")
    mark_id = publish_late_review(tmp_path, record).mark_id
    assert _release(tmp_path, mark_id, direct=True)["ok"]
    assert publish_late_review(tmp_path, record).status == "unchanged"
    assert _late_marks(tmp_path) == [] and "ONCE" not in room_text(tmp_path)
    (tmp_path / "memory" / "chronicle" / "index.sqlite3").unlink()  # cold: the journal replays into a new index
    assert publish_late_review(tmp_path, record).status == "unchanged"
    assert _late_marks(tmp_path) == [] and "ONCE" not in room_text(tmp_path)
    kinds = [row["kind"] for row in ChronicleStore(tmp_path).records() if mark_id in (row["id"], row.get("target_id"))]
    assert kinds == ["mark", "mark_release"], "one mark, one release, nothing resurrected"


def test_the_marks_source_reader_survives_retention_packing_its_bytes(tmp_path):
    from ouroboros import evidence_packer
    from ouroboros.artifacts import task_artifact_dir_path

    shared.world(tmp_path)
    seed_result(tmp_path, "main1", {"status": "completed", "chat_id": 1})
    record = _record(tmp_path, "main1", 1, "PACKED-CRITIQUE")
    assert publish_late_review(tmp_path, record).status == "saved"
    loose = task_artifact_dir_path(tmp_path, "main1") / record["source_ref"]["path"]
    report = evidence_packer.run_pass(tmp_path)
    assert not loose.exists(), report  # retention moved the exact bytes into the task's evidence pack
    shown = re.search(r"memory_read\(source_ref=(\{.*?\})\)", room_text(tmp_path)).group(1)
    source = _memory_read(_ctx(tmp_path, "root0001"), source_ref=json.loads(shown))
    assert "PACKED-CRITIQUE" in source and f"sha256 {record['source_ref']['sha256'][:12]}" in source


def test_a_foreign_focus_reads_the_marks_source_by_the_named_reader_never_the_authors_recipe(tmp_path):
    """The producer's ``read_file(root='artifact_store')`` recipe resolves in the author's drive only (the
    26.10 consciousness wake tried it and the digest as a CAS blob); the full mark names the reader every
    focus has, for marks already stored with that recipe too, and the packed bytes read through it."""
    from ouroboros import evidence_packer

    shared.world(tmp_path)
    seed_result(tmp_path, "main1", {"status": "completed", "chat_id": 1})
    record = _record(tmp_path, "main1", 1, "FOREIGN-CRITIQUE")
    assert '"tool": "read_file"' in json.dumps(record["source_ref"])  # the author-relative recipe stays stored
    mark_id = publish_late_review(tmp_path, record).mark_id
    evidence_packer.run_pass(tmp_path)  # no loose file is left at the author-relative path either
    node = _memory_read(_ctx(tmp_path, "wake1", meta=WAKE), node_id=mark_id)
    assert '"tool": "read_file"' not in node, "the author's recipe never reaches another focus"
    shown = json.loads(re.search(r"\nsource: memory_read\(source_ref=(\{.*?\})\)", node).group(1))
    assert shown["task_id"] == "main1" and shown["size"] == record["source_ref"]["size"]
    assert "FOREIGN-CRITIQUE" in _memory_read(_ctx(tmp_path, "wake1", meta=WAKE), source_ref=shown)
    plain = json.loads(_memory_mark(_ctx(tmp_path, "main1"), text="mine", task_id="main1"))
    assert "\nsource: " not in _memory_read(_ctx(tmp_path, "main1"), node_id=plain["mark_id"])  # other targets unchanged


def test_a_retained_source_is_read_without_the_chronicle_and_other_modes_still_wait(tmp_path, monkeypatch):
    from ouroboros.tools import chronicle

    shared.world(tmp_path)
    ref, _data = _source(tmp_path, "main1", "NO-CHRONICLE-NEEDED")

    def pending(_root):
        raise chronicle._NotActivated({"kind": "import_pending"})

    monkeypatch.setattr(chronicle, "_activated", pending)
    assert "NO-CHRONICLE-NEEDED" in _memory_read(_ctx(tmp_path, "wake1"), source_ref={**ref, "task_id": "main1"})
    assert "memory is not active" in _memory_read(_ctx(tmp_path, "wake1"), room_id=1)


def _owed_rows(root):
    from ouroboros import store
    from ouroboros.late_review_marks import OWED_SET

    with store.open_store(root, write=False) as txn:
        return txn.set_rows(OWED_SET)


def _refuse_publication(self, records, *, scan_state=None):
    raise TimeoutError("chronicle publication lock unavailable")


def _sweep(root, monkeypatch):
    """The real off-loop maintenance pass, its unrelated steps idle: owed late reviews are replayed there."""
    import supervisor.workspace_preparation as preparation
    from ouroboros import server_maintenance as maintenance
    from ouroboros.maintenance import child as maintenance_child

    monkeypatch.setattr(maintenance, "DATA_DIR", root)
    monkeypatch.setattr(maintenance, "_run_reconcile_prunes", lambda *_a: None)
    monkeypatch.setattr(maintenance, "_periodic_zombie_reconcile", lambda **_kw: None)
    monkeypatch.setattr(maintenance, "_run_drive_custody_pass", lambda *_a: None)
    monkeypatch.setattr(preparation, "recover_interrupted_preparations", lambda *_a, **_kw: None)
    monkeypatch.setattr(maintenance_child, "tick", lambda *_a, **_kw: None)
    maintenance._run_periodic_reconcile_sweep([0], latch=SimpleNamespace(release=lambda: None))
    from ouroboros.maintenance import run_job
    run_job(root, "ended_plan_asks")


def test_a_failed_publication_stays_owed_and_one_replay_after_restart_places_it_once(tmp_path, monkeypatch):
    from ouroboros import store
    from ouroboros.late_review_marks import OWED_SET, replay_owed_late_reviews

    shared.world(tmp_path)
    seed_result(tmp_path, "main1", {"status": "completed", "chat_id": 1})
    record = _record(tmp_path, "main1", 1, "OWED-CRITIQUE")
    monkeypatch.setattr(ChronicleStore, "publish", _refuse_publication)
    failed = publish_late_review(tmp_path, record)
    assert failed.status == "failed" and list(_owed_rows(tmp_path)) == [failed.mark_id]
    assert replay_owed_late_reviews(tmp_path)["failed"] == 1 and list(_owed_rows(tmp_path)) == [failed.mark_id]
    monkeypatch.undo()  # a restart: nothing in memory, the owed row is the whole duty
    (tmp_path / "memory" / "chronicle" / "index.sqlite3").unlink(missing_ok=True)
    assert replay_owed_late_reviews(tmp_path) == {"settled": 1, "waiting": 0, "retired": 0, "failed": 0}
    assert [mark["id"] for mark in _late_marks(tmp_path)] == [failed.mark_id] and _owed_rows(tmp_path) == {}
    assert "OWED-CRITIQUE" in room_text(tmp_path)
    assert replay_owed_late_reviews(tmp_path)["settled"] == 0, "a discharged duty replays nothing"
    assert _release(tmp_path, failed.mark_id, direct=True)["ok"]
    with store.open_store(tmp_path, write=True) as txn:  # a stale duplicate of the same duty
        txn.put_row(OWED_SET, failed.mark_id, {"record": record})
    assert replay_owed_late_reviews(tmp_path)["settled"] == 1 and _owed_rows(tmp_path) == {}
    assert _late_marks(tmp_path) == [], "a replay never resurrects a released mark"


# --- real publisher: a plan council reply that settles after its task ended (the planning organ) ---------

from ouroboros import plan_record  # noqa: E402
from tests.test_plan_task_tool import CLEAN, _call, _reply, _write, harness  # noqa: E402,F401 — the organ's fixture

PLAN_TASK = "task-1"


def _until(predicate, timeout=20.0):
    import time

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(0.02)
    raise AssertionError("expected state did not arrive")


class _Council:
    """The seats' chat transport under the REAL review substrate: a seat answers its reply only once its gate
    opens, so its producer writes its own retained outcome and settles through custody, as in production."""

    def __init__(self, replies):
        self.replies, self.calls = dict(replies), []
        self.gates = {model: threading.Event() for model in replies}

    def __call__(self):  # LLMClient()
        return self

    def chat(self, **kwargs):
        model = kwargs["model"]
        self.calls.append(model)
        assert self.gates[model].wait(30), f"{model} was never released"
        return {"content": self.replies[model]}, {"prompt_tokens": 10, "completion_tokens": 5, "cost": 0.01,
                                                  "resolved_model": model}

    def release(self, *models):
        for model in models or tuple(self.gates):
            self.gates[model].set()


@pytest.fixture
def council(harness, monkeypatch):  # noqa: F811
    """An ask to the whole council (s1/m-a, s2/m-b, s3/m-c) still pending at the dispatch barrier."""
    rooms = shared.projects(harness.drive)  # Main and two Projects: the room a mark lands in is a proven one
    replies = {"m/a": _reply("The rollback path is missing.", objections=[("o1", "Add the rollback step.")]),
               "m/b": _reply("Looks sound to me."), "m/c": CLEAN}
    transport = _Council(replies)
    monkeypatch.setattr("ouroboros.tools.plan_review_runtime.LLMClient", transport)
    ctx = harness.make_ctx()
    try:
        _write(ctx, ask="council")
        record = plan_record_now(harness)
        assert [p["status"] for p in record["asks"][0]["participants"]] == ["pending"] * 3, record["asks"][0]
        yield SimpleNamespace(h=harness, ctx=ctx, transport=transport, rooms=rooms)
    finally:
        transport.release()


def plan_record_now(h):
    return plan_record.load_plan_record(h.drive, PLAN_TASK)


def _participant(h, seat):
    return next(p for p in plan_record_now(h)["asks"][0]["participants"] if p["seat_id"] == seat)


def _end(h, status="completed"):
    from ouroboros.task_results import write_task_result

    write_task_result(h.drive, PLAN_TASK, status, chat_id=1, result="The author ended.")


def _plan_marks(root):
    return [mark for mark in _late_marks(root) if mark["review"]["surface"] == "plan_review"]


def test_an_ended_authors_late_plan_replies_become_one_mark_each_on_their_own_ledger_reply(council):
    """Two seats settle after their author ended: one mark each, in the task's room, pointing at the
    ledger's own retained reply (never a copy), told as advice. A repeat changes nothing and the plan
    record keeps exactly what the replies said."""
    from ouroboros.artifacts import read_actor_source_bytes
    from ouroboros.tools import plan_ask

    h, transport = council.h, council.transport
    _end(h)
    transport.release("m/a")
    [first] = _until(lambda: _plan_marks(h.drive))
    s1 = _until(lambda: (lambda p: p if p.get("reply_ref") else None)(_participant(h, "s1")))
    assert first["room_id"] == "1" and first["id"] == late_review_mark_id("plan_review", PLAN_TASK, "plan-late:a1:s1")
    assert first["target_ref"] == {**s1["reply_ref"]["ref"], "task_id": s1["reply_ref"]["task_id"]}
    raw = json.loads(read_actor_source_bytes(h.drive, s1["reply_ref"]["task_id"], s1["reply_ref"]["ref"]))
    assert raw["objections"] == [{"id": "o1", "text": "Add the rollback step."}], "the ledger's exact reply"
    text = first["text"]
    assert "Plan council reply from m/a (seat s1) to ask a1 on plan revision 1" in text
    assert "advice for whoever continues this work, not a verdict; it holds nothing." in text
    assert "Prose: The rollback path is missing." in text and "- objection o1: Add the rollback step." in text
    assert "PASS" not in text and "FAIL" not in text
    transport.release("m/b")
    _until(lambda: len(_plan_marks(h.drive)) == 2)
    second = next(mark for mark in _plan_marks(h.drive) if mark["id"] != first["id"])
    assert "(seat s2)" in second["text"] and "objection" not in second["text"]
    before = plan_record_now(h)
    record = plan_ask.collect_pending(None, before, state_root=h.drive, task_id=PLAN_TASK)  # a repeat
    assert len(_plan_marks(h.drive)) == 2 and [m["id"] for m in _plan_marks(h.drive)].count(first["id"]) == 1
    assert [o["id"] for o in plan_record.open_objections(record)] == ["o1"]
    assert transport.calls.count("m/a") == 1 and transport.calls.count("m/b") == 1, "nothing is sent twice"


def _lose_settlement_callbacks(monkeypatch):
    """The process loses each seat's settlement callback after its producer retained its outcome (the crash
    window): the seat stays pending in its ask, which is the whole duty."""
    lost = []
    monkeypatch.setattr("ouroboros.tools.plan_ask.settle_released_reply", lambda *_a, **kw: lost.append(kw["actor"]))
    return lost


def test_a_reply_that_settles_while_its_author_runs_stays_with_the_gate_and_plan_task(council):
    """While the author runs, a reply is the gate's and plan_task()'s: nothing is kept in view, no row is
    owed for later, and the task's end keeps nothing either (organ form of G1's owe-until-end cases)."""
    from ouroboros.late_review_marks import replay_owed_late_reviews

    h, transport = council.h, council.transport
    transport.release("m/a")
    s1 = _until(lambda: (lambda p: p if p.get("reply_ref") else None)(_participant(h, "s1")))
    assert s1["status"] == "answered" and _plan_marks(h.drive) == [] and _owed_rows(h.drive) == {}
    text, _result = _call(council.ctx)  # the author's own free read carries it
    assert "Add the rollback step." in text
    _end(h)
    assert replay_owed_late_reviews(h.drive) == {"settled": 0, "waiting": 0, "retired": 0, "failed": 0}
    assert _plan_marks(h.drive) == [] and _owed_rows(h.drive) == {}


def test_a_late_reply_reaches_its_rooms_next_view_and_consciousness_and_no_other_room(council):
    h, transport = council.h, council.transport
    _end(h)
    transport.release("m/a")
    [mark] = _until(lambda: _plan_marks(h.drive))
    room = room_text(h.drive)
    assert SECTION in room and "Plan council reply from m/a (seat s1)" in room and f"release_ids=['{mark['id']}']" in room
    assert "Plan council reply from m/a" in room_text(h.drive, {"id": "wake1", "chat_id": 1, "metadata": WAKE})
    other = room_text(h.drive, {"id": "alpha-turn", "chat_id": council.rooms["alpha"]})
    assert "Plan council reply" not in other


@pytest.mark.parametrize("end", ["completed", "failed", "cancelled"])
def test_a_reply_whose_settlement_callback_was_lost_reaches_the_room_after_a_normal_or_abnormal_end(
        council, monkeypatch, end):
    """The crash window: the producer retained its outcome but the settlement callback was lost, so the seat
    is still pending in its ask. The task's canonical end, normal or abnormal, lets the real maintenance pass
    collect it at $0 (``plan_ask.collect_pending``) and keep it in the room once; no row was ever owed for it,
    a replay neither duplicates it nor brings back a released mark, and nothing is sent again."""
    h, transport = council.h, council.transport
    with monkeypatch.context() as lossy:
        lost = _lose_settlement_callbacks(lossy)
        transport.release("m/a")
        _until(lambda: lost)
    assert _participant(h, "s1")["status"] == "pending" and _owed_rows(h.drive) == {}
    _sweep(h.drive, monkeypatch)
    assert _plan_marks(h.drive) == [], "a running author's seat stays with its own collector"
    _end(h, end)
    _sweep(h.drive, monkeypatch)
    [mark] = _plan_marks(h.drive)
    assert _participant(h, "s1")["status"] == "answered" and "- objection o1: Add the rollback step." in mark["text"]
    assert _owed_rows(h.drive) == {}, "the ask's pending seat was the whole duty"
    _sweep(h.drive, monkeypatch)
    assert len(_plan_marks(h.drive)) == 1
    assert _release(h.drive, mark["id"], direct=True)["ok"]
    _sweep(h.drive, monkeypatch)
    assert _plan_marks(h.drive) == [] and transport.calls.count("m/a") == 1


def test_a_seat_its_author_collected_before_ending_is_no_late_review(council, monkeypatch):
    """Negative: the running author's own free collection settled s1, so the end keeps nothing for it; s2,
    still running at the end, reaches the room on its own settlement."""
    h, transport = council.h, council.transport
    with monkeypatch.context() as lossy:
        lost = _lose_settlement_callbacks(lossy)
        transport.release("m/a")
        _until(lambda: lost)
    _call(council.ctx)  # plan_task() read: collects the retained outcome at $0 while the author runs
    assert _participant(h, "s1")["status"] == "answered" and _plan_marks(h.drive) == []
    _end(h)
    _sweep(h.drive, monkeypatch)
    assert _plan_marks(h.drive) == []
    transport.release("m/b")
    [mark] = _until(lambda: _plan_marks(h.drive))
    assert mark["review"]["subject"]["seat_id"] == "s2" and transport.calls.count("m/a") == 1


def test_an_unreadable_producer_keeps_the_seat_pending_until_a_later_pass_places_one_mark(council, monkeypatch):
    """A known producer whose retained outcome cannot be read on one pass is still the seat's duty: nothing is
    kept in view, nothing is cleaned and nothing is sent; the next pass places exactly one mark."""
    from ouroboros import observability
    from ouroboros.tools import plan_ask

    h, transport = council.h, council.transport
    with monkeypatch.context() as lossy:
        lost = _lose_settlement_callbacks(lossy)
        transport.release("m/a")
        _until(lambda: lost)
    _end(h, "failed")
    operation = _participant(h, "s1")["operation_id"]
    original = observability.read_call_payload

    def unreadable(root, *, task_id, call_id):
        if call_id == f"{operation}_response":
            raise OSError("producer CAS momentarily unreadable")
        return original(root, task_id=task_id, call_id=call_id)

    monkeypatch.setattr(plan_ask, "dispatch_ask", lambda *a, **k: pytest.fail("no ask is sent"))
    with monkeypatch.context() as fault:
        fault.setattr(observability, "read_call_payload", unreadable)
        _sweep(h.drive, monkeypatch)
    assert _plan_marks(h.drive) == [] and _participant(h, "s1")["status"] == "pending"
    from ouroboros.maintenance_jobs import record
    record(h.drive, "ended_plan_asks", state="pending")  # explicit source repair
    _sweep(h.drive, monkeypatch)
    [mark] = _plan_marks(h.drive)
    assert mark["review"]["subject"]["seat_id"] == "s1"
    _sweep(h.drive, monkeypatch)
    assert len(_plan_marks(h.drive)) == 1 and transport.calls.count("m/a") == 1


def test_while_its_author_runs_the_asks_settled_frame_reaches_its_mailbox_and_nothing_stays_in_view(council):
    """The frame of an ask whose every seat settled while its author ran is that author's wake in its own
    mailbox: it names the counts and points at plan_task(), it is no collection and no verdict, and nothing
    is kept in view for the room."""
    from ouroboros.owner_mailbox import _mailbox_path

    h, transport = council.h, council.transport
    transport.release()
    path = _mailbox_path(h.drive, PLAN_TASK)
    frames = _until(lambda: [json.loads(line)["text"] for line in (path.read_text(encoding="utf-8").splitlines()
                                                                  if path.exists() else []) if "Plan ask a1" in line])
    assert len(frames) == 1 and "every participant settled — 3 answered" in frames[0]
    assert "plan_task() reads the answers." in frames[0] and "PASS" not in frames[0]
    assert _plan_marks(h.drive) == [] and _owed_rows(h.drive) == {}


@pytest.mark.parametrize("raw,lines", [
    (_reply("Split the migration.", objections=[("o2", "Run the backfill first."), ("o3", "Keep the old\nreader.")]),
     ["Prose: Split the migration.", "- objection o2: Run the backfill first.", "- objection o3: Keep the old reader."]),
    (_reply("Looks sound to me."), ["Prose: Looks sound to me."]),
    ("I think this is fine, but no JSON here.",
     ["Its reply could not be read as the reply object; the full text is in its retained source."]),
    ("", None),  # a failed seat: no paid words to keep, so no mark
])
def test_a_late_plan_reply_is_told_in_its_own_terms_never_as_a_verdict(harness, raw, lines):  # noqa: F811
    from ouroboros.review_records import ReviewActorRecord
    from ouroboros.tools import plan_ask

    shared.projects(harness.drive)
    pending = {"status": "pending", "raw_text": "", "operation_state": "pending_dispatch", "late_result_pending": True}
    sub = harness.install({"s1": pending, "s2": pending, "s3": pending})
    ctx = harness.make_ctx()
    _write(ctx, ask="council")
    _end(harness)
    request, slots = sub.calls[0]["request"], {s.slot_id: s for s in sub.calls[0]["slots"]}
    actor = ReviewActorRecord(slot_id="s1", model="m/a", status="ok" if raw else "error", raw_text=raw,
                              error="" if raw else "run ended failed", operation_id="op-s1-1",
                              operation_state="late_settled", usage={"cost": 0.02, "resolved_model": "m/a"})
    plan_ask.settle_released_reply(ctx, request=request, task_id=PLAN_TASK, actor=actor, slot=slots["s1"],
                                   settled_wave={}, roster_size=3)
    marks = _plan_marks(harness.drive)
    if lines is None:
        assert marks == [] and _participant(harness, "s1")["status"] == "failed"
        return
    [mark] = marks
    store_head, reviewed, head, *rest = mark["text"].splitlines()
    assert store_head.startswith(f"Late plan review of task {PLAN_TASK}") and reviewed.startswith("Reviewed: ask_id=a1;")
    assert reviewed.endswith("seat_id=s1") and "plan_revision=1" in reviewed
    assert head.startswith("Plan council reply from m/a (seat s1) to ask a1 on plan revision 1, recorded after its task ended")
    assert "not a verdict; it holds nothing." in head and rest == lines
    assert all(word not in mark["text"] for word in ("PASS", "FAIL", "approved", "rejected"))
