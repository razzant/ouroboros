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
    # Other rooms' late reviews are counted in the wake's waiting facts and listed by room and work on request; the
    # wake draws no line for each of them.
    assert marks["bound"] not in wake and marks["pres1"] not in wake and "Late review of task" not in wake
    listing = _memory_read(_ctx(root, "wake1"), room_id="late_reviews")
    assert listing.startswith("late reviews: 3 in 3 rooms")
    for task in ("bound", "pres1"):
        assert f"result review of task {task}; outcome not recorded; memory_read(node_id='{marks[task]}')" in listing
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


def test_a_view_captured_with_an_old_pointer_entry_draws_no_pointer_and_keeps_this_rooms_words():
    """A view captured before other rooms' late reviews left it holds a pointer entry (no words, ``own_room`` False);
    a held core is re-rendered after an update (``context_fit``), so such an entry draws nothing, while this room's
    late review keeps its words and its release."""
    import dataclasses

    pointer = {"id": "late-review-old", "scope": "room", "room": "Alpha", "text": "", "by": "host",
               "date": "2026-10-17", "target": "source x", "quote": "",
               "late_review": {"task_id": "old1", "source": 'memory_read(node_id="late-review-old")', "sha256": "",
                               "own_room": False}}
    own = {**pointer, "id": "late-review-own", "room": "Main", "text": "OWN-WORDS",
           "late_review": {"task_id": "own1", "source": "memory_read(source_ref={})", "sha256": "abcdef012345"}}
    spec = dataclasses.replace(mv.ROLE_DEFAULTS["consciousness"], room_id="1", other_conversations=False)
    for marks, shown in (((pointer,), False), ((pointer, own), True)):
        text = mv.render_room(mv.MemoryViewSnapshot(spec=spec, store_status={"state": "active"}, frontier={},
                                                    marks=marks))
        assert "late-review-old" not in text and "old1" not in text
        assert (SECTION in text) is shown and ("OWN-WORDS" in text) is shown
        assert ("release_ids=['late-review-own']" in text) is shown


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


@pytest.mark.parametrize("verdict, subject, resolves", [("PASS", "subject-a", True), ("PASS", "subject-b", False),
                                                        ("FAIL", "subject-a", False)])
def test_a_late_pass_rewrites_only_the_review_derived_outcome_of_the_current_row(tmp_path, monkeypatch, verdict, subject,
                                                                                  resolves):
    """The late PASS of an author's pending finish (``resolve_pending_result_review``) sets the review axis and the
    objective ``_objective_axis`` derives from it inside the store's own write over the CURRENT row: an outcome
    write that landed after the resolver's first read is kept, and so is every warning the review never answered
    (``receipt_absent``). The existing late-review event then carries that current outcome, bound to the exact task,
    record and subject, once: durable, and pushed live where a bridge exists. Another verdict or another version
    changes and announces nothing, and a replay announces nothing again."""
    from ouroboros import review_ledger
    from ouroboros.late_review_marks import resolve_pending_result_review
    from ouroboros.outcomes import WARN_RECEIPT_ABSENT, _merge_objective_warning, _objective_axis
    from ouroboros.project_dialogue import outcome_phase
    from ouroboros.task_results import load_task_result, write_task_result

    review = {"status": "awaiting", "source": "result_review", "author_finish": "author_finish_pending",
              "pending_review": {"record_id": "rl-late", "subject_sha": "subject-a"},
              "acceptance_decision": {"status": "finalized_unaccepted", "source": "result_review",
                                      "reason": "author_finish_pending", "enforcement": "advisory"}}
    objective = _objective_axis(review)
    _merge_objective_warning(objective, WARN_RECEIPT_ABSENT)
    write_task_result(tmp_path, "task-1", "completed", outcome_axes={
        "execution": {"status": "ok"}, "review": review, "objective": objective})
    before = load_task_result(tmp_path, "task-1")["outcome_axes"]

    def record_after_a_concurrent_write(_root, record_id):  # another writer lands after the resolver's first read
        axes = load_task_result(tmp_path, "task-1")["outcome_axes"]
        write_task_result(tmp_path, "task-1", "completed", outcome_axes={**axes, "artifacts": {"status": "not_applicable"}})
        return {"record_id": record_id, "subject": {"diff_sha": subject}, "verdict": {"aggregate": verdict},
                "ts": "2026-11-08T22:25:00+00:00"}

    monkeypatch.setattr(review_ledger, "load_record", record_after_a_concurrent_write)
    pushed = []
    monkeypatch.setattr("supervisor.message_bus.try_get_bridge", lambda: SimpleNamespace(push_log=pushed.append))

    def announced():
        path = tmp_path / "logs" / "events.jsonl"
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()] if path.exists() else []
        return [row for row in rows if row.get("type") == "review_late_result"]

    assert resolve_pending_result_review(tmp_path, "task-1", "rl-late") is resolves
    stored = load_task_result(tmp_path, "task-1")["outcome_axes"]
    assert stored["artifacts"] == {"status": "not_applicable"}, "the concurrent write is never lost"
    assert stored["objective"]["warnings"] == [WARN_RECEIPT_ABSENT], "a warning the review never answered stays"
    if not resolves:
        assert (stored["review"], stored["objective"]) == (before["review"], before["objective"])
        assert announced() == [] and pushed == [], "a FAIL or another version announces nothing"
        return
    assert stored["review"]["status"] == "pass" and "pending_review" not in stored["review"]
    assert stored["review"]["pending_finish_resolved"]["objective"] == before["objective"]
    assert {key: value for key, value in stored["objective"].items() if key not in {"warning", "warnings"}} == \
        _objective_axis(stored["review"]) == {"status": "pass", "source": "result_review", "review_status": "pass"}
    [event] = announced()
    assert {key: event[key] for key in ("task_id", "surface", "record_id", "subject_sha", "aggregate")} == {
        "task_id": "task-1", "surface": "result", "record_id": "rl-late", "subject_sha": "subject-a", "aggregate": "PASS"}
    current = event["current_outcome"]
    row = load_task_result(tmp_path, "task-1")
    # The receipt_absent the review never answered stays in the record as a detail; the headline is Done (9 Nov 2026).
    assert current["outcome_phase"] == outcome_phase(row, {}) == "done"
    assert current["outcome_axes"]["objective"]["warnings"] == [WARN_RECEIPT_ABSENT]
    assert current["outcome_axes"]["artifacts"] == {"status": "not_applicable"} and current["revision"] == row["updated_at"]
    assert outcome_phase(current, {}) == current["outcome_phase"], "its card reads the phase a reload computes"
    assert "pending_finish_resolved" not in json.dumps(current) and "late_verdict" not in json.dumps(current), "compact"
    assert [frame["task_id"] for frame in pushed] == ["task-1"] and pushed[0]["current_outcome"] == current
    assert resolve_pending_result_review(tmp_path, "task-1", "rl-late") is False, "resolved once"
    assert len(announced()) == 1 and len(pushed) == 1, "a replay announces nothing again"


def test_a_warning_written_after_the_late_pass_is_in_what_its_card_repaints(tmp_path, monkeypatch):
    """Astra, 10 Nov 2026: a custody warning (``delegated_runs_unreconciled``) that lands after the late PASS resolved
    the row and before the event is published is in the announced outcome, because the event reads the stored row when
    it publishes: the card repaints Done with warnings, as a reload does, never the resolver's earlier Done. Its
    card orders a late outcome by the revision it read (``lateOutcomeIsCurrent``): an older snapshot published later
    never repaints the newer state, and the same frame again still paints."""
    import pathlib
    import subprocess

    from ouroboros import late_review_marks as late, review_ledger
    from ouroboros.outcomes import _objective_axis
    from ouroboros.preflight_node import resolve_node
    from ouroboros.project_dialogue import outcome_phase
    from ouroboros.task_results import load_task_result, write_task_result

    review = {"status": "awaiting", "source": "result_review", "author_finish": "author_finish_pending",
              "pending_review": {"record_id": "rl-late", "subject_sha": "subject-a"},
              "acceptance_decision": {"status": "finalized_unaccepted", "source": "result_review",
                                      "reason": "author_finish_pending", "enforcement": "advisory"}}
    write_task_result(tmp_path, "task-1", "completed", reason_code="final_message", outcome_axes={
        "execution": {"status": "ok", "reason_code": "final_message"}, "review": review,
        "objective": _objective_axis(review)})
    ended_at = load_task_result(tmp_path, "task-1")["updated_at"]
    monkeypatch.setattr(review_ledger, "load_record", lambda _root, record_id: {
        "record_id": record_id, "subject": {"diff_sha": "subject-a"}, "verdict": {"aggregate": "PASS"}, "ts": "t"})
    resolved, announce = [], late.announce_current_outcome

    def custody_lands_first(*args, **kwargs):  # the custody writer lands between the resolution and the event
        resolved.append(load_task_result(tmp_path, "task-1"))
        write_task_result(tmp_path, "task-1", "completed", delegated_runs_unreconciled=["run-x"])
        return announce(*args, **kwargs)

    monkeypatch.setattr(late, "announce_current_outcome", custody_lands_first)
    pushed = []
    monkeypatch.setattr("supervisor.message_bus.try_get_bridge", lambda: SimpleNamespace(push_log=pushed.append))
    assert late.resolve_pending_result_review(tmp_path, "task-1", "rl-late") is True
    stored = load_task_result(tmp_path, "task-1")
    assert stored["outcome_axes"]["review"]["status"] == "pass" and stored["delegated_runs_unreconciled"] == ["run-x"]
    assert outcome_phase(resolved[0], {}) == "done" and resolved[0]["updated_at"] < stored["updated_at"], \
        "the resolver's own copy reads Done and is older than the warning"
    [event] = [row for row in map(json.loads, (tmp_path / "logs" / "events.jsonl").read_text(encoding="utf-8")
                                  .splitlines()) if row.get("type") == "review_late_result"]
    current = event["current_outcome"]
    assert current["outcome_phase"] == outcome_phase(stored, {}) == "warn", "the custody warning stays"
    assert current["delegated_runs_unreconciled"] == ["run-x"] and current["revision"] == stored["updated_at"]
    assert [frame["current_outcome"] for frame in pushed] == [current], "the live frame carries the same outcome"

    stale = {**event, "current_outcome": late.current_outcome(resolved[0])}  # the resolver's copy, published later
    assert stale["current_outcome"]["outcome_phase"] == "done" and stale["ts"] >= current["revision"]
    node = resolve_node()
    assert node, "the card's ordering of a late outcome requires Node"
    script = """import {lateOutcomeIsCurrent, summarizeChatLiveEvent} from './web/modules/log_events.js';
let text = ''; for await (const chunk of process.stdin) text += chunk;
const [current, stale, endedAt] = JSON.parse(text), card = {finished: true, latestSourceTs: endedAt};
const taken = lateOutcomeIsCurrent(card, current), view = summarizeChatLiveEvent(current);
card.latestSourceTs = current.current_outcome.revision;  // the clock chat.js sets when it takes the frame
console.log(JSON.stringify([taken, view.phase, view.headline, lateOutcomeIsCurrent(card, stale),
  lateOutcomeIsCurrent(card, current)]));"""
    checked = subprocess.run([node, "--input-type=module", "-e", script], capture_output=True, text=True,
                             encoding="utf-8", check=True, input=json.dumps([event, stale, ended_at]),
                             cwd=pathlib.Path(__file__).resolve().parents[1])
    assert json.loads(checked.stdout) == [True, "warn", "Done with warnings", False, True], checked.stdout


def test_a_stale_late_outcome_loses_to_a_newer_row_written_inside_its_millisecond(tmp_path, monkeypatch):
    """Astra, 10 Nov 2026 (C1): a stored row's ``updated_at`` carries microseconds and two writes of one task can land
    inside one millisecond. The revisions the real writer and ``current_outcome`` produce keep their order in the card's
    own guard (``lateOutcomeIsCurrent``): the older PASS never clears the newer custody warning, the same revision again
    is a replay, a truly later outcome inside that millisecond paints, and the older warning cannot return over it."""
    import pathlib
    import subprocess

    from ouroboros import late_review_marks as late, task_results
    from ouroboros.outcomes import _objective_axis
    from ouroboros.preflight_node import resolve_node
    from ouroboros.project_dialogue import outcome_phase

    clock = {"now": ""}
    monkeypatch.setattr(task_results, "utc_now_iso", lambda: clock["now"])

    def written(micros, **fields):  # a real store write at an exact moment of one millisecond
        clock["now"] = f"2026-11-10T05:06:26.{micros:06d}+00:00"
        return task_results.write_task_result(tmp_path, "task-1", "completed", **fields)

    review = {"status": "pass", "source": "result_review", "aggregate_signals": ["PASS"]}
    older = written(147169, reason_code="final_message", terminal_origin="model_final", result="The answer", outcome_axes={
        "execution": {"status": "ok", "reason_code": "final_message"}, "review": review,
        "objective": _objective_axis(review)})
    newer = written(147857, delegated_runs_unreconciled=["run-x"])
    later = written(147969, delegated_runs_unreconciled=[])
    assert [outcome_phase(dict(row), {}) for row in (older, newer, later)] == ["done", "warn", "done"]
    assert [row["updated_at"][:23] for row in (older, newer)] == ["2026-11-10T05:06:26.147"] * 2, "one millisecond"
    node = resolve_node()
    assert node, "the card's ordering of a late outcome requires Node"
    script = """import {lateOutcomeIsCurrent} from './web/modules/log_events.js';
let text = ''; for await (const chunk of process.stdin) text += chunk;
const [older, newer, later] = JSON.parse(text).map((current_outcome) => ({ts: '', current_outcome}));
const card = (frame) => ({finished: true, latestSourceTs: frame.current_outcome.revision});
console.log(JSON.stringify([lateOutcomeIsCurrent(card(newer), older), lateOutcomeIsCurrent(card(newer), newer),
  lateOutcomeIsCurrent(card(newer), later), lateOutcomeIsCurrent(card(later), newer)]));"""
    checked = subprocess.run([node, "--input-type=module", "-e", script], capture_output=True, text=True,
                             encoding="utf-8", check=True,
                             input=json.dumps([late.current_outcome(row) for row in (older, newer, later)]),
                             cwd=pathlib.Path(__file__).resolve().parents[1])
    assert json.loads(checked.stdout) == [False, True, True, False], checked.stdout


def test_an_ordinary_ended_task_is_never_resolved_or_announced_by_a_late_pass(tmp_path, monkeypatch):
    """A task that finished without a pending review keeps its stored outcome and its warning whatever late PASS names
    it: nothing is resolved and nothing is announced."""
    from ouroboros import review_ledger
    from ouroboros.late_review_marks import resolve_pending_result_review
    from ouroboros.outcomes import WARN_RECEIPT_ABSENT, _merge_objective_warning
    from ouroboros.task_results import load_task_result, write_task_result

    objective = {"status": "not_evaluated", "source": "none"}
    _merge_objective_warning(objective, WARN_RECEIPT_ABSENT)
    write_task_result(tmp_path, "task-1", "completed", outcome_axes={
        "execution": {"status": "ok"}, "review": {"status": "skipped"}, "objective": objective})
    before = load_task_result(tmp_path, "task-1")["outcome_axes"]
    monkeypatch.setattr(review_ledger, "load_record", lambda _root, record_id: {
        "record_id": record_id, "subject": {"diff_sha": "subject-a"}, "verdict": {"aggregate": "PASS"}})
    assert resolve_pending_result_review(tmp_path, "task-1", "rl-late") is False
    assert load_task_result(tmp_path, "task-1")["outcome_axes"] == before
    assert not (tmp_path / "logs" / "events.jsonl").exists()


@pytest.mark.parametrize("warning", ["", "receipt_absent"])
def test_the_announced_outcome_is_the_parity_row_the_card_and_the_host_both_read(tmp_path, monkeypatch, warning):
    """The compact current outcome the resolver announces is exactly a row of the browser/host parity fixture
    (``web/tests/fixtures/outcome_phase_parity.json``): the card's twin and a reload's host reader read it alike, Done,
    with or without an independent ``receipt_absent`` (a detail)."""
    import pathlib

    from ouroboros import review_ledger
    from ouroboros.late_review_marks import resolve_pending_result_review
    from ouroboros.outcomes import _merge_objective_warning, _objective_axis
    from ouroboros.task_results import write_task_result

    review = {"status": "awaiting", "source": "result_review", "author_finish": "author_finish_pending",
              "pending_review": {"record_id": "rl-late", "subject_sha": "subject-a"},
              "acceptance_decision": {"status": "finalized_unaccepted", "source": "result_review",
                                      "reason": "author_finish_pending", "enforcement": "advisory"}}
    objective = _objective_axis(review)
    if warning:
        _merge_objective_warning(objective, warning)
    write_task_result(tmp_path, "task-1", "completed", reason_code="final_message", outcome_axes={
        "execution": {"status": "ok", "reason_code": "final_message"}, "review": review, "objective": objective})
    monkeypatch.setattr(review_ledger, "load_record", lambda _root, record_id: {
        "record_id": record_id, "subject": {"diff_sha": "subject-a"}, "verdict": {"aggregate": "PASS"}, "ts": "t"})
    assert resolve_pending_result_review(tmp_path, "task-1", "rl-late") is True
    [event] = [json.loads(line) for line in (tmp_path / "logs" / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    fixture = pathlib.Path(__file__).resolve().parents[1] / "web" / "tests" / "fixtures" / "outcome_phase_parity.json"
    rows = [case["record"] for case in json.loads(fixture.read_text(encoding="utf-8"))["cases"] if "late PASS" in case["name"]]
    assert {key: value for key, value in event["current_outcome"].items() if key != "revision"} == rows[1 if warning else 0]


def _js_views(rows):
    """``taskOutcomeSeverity``, ``taskReasonDetail`` and the terminal summary's phase, headline, body and collapsed
    line of each row, as the browser reads them (``log_events.js``)."""
    import pathlib
    import subprocess

    from ouroboros.preflight_node import resolve_node

    node = resolve_node()
    assert node, "the browser twin's reading of a late current outcome requires Node"
    script = """import {taskOutcomeSeverity, taskReasonDetail, taskTerminalSummary, summarizeChatLiveEvent} from './web/modules/log_events.js';
let text = ''; for await (const chunk of process.stdin) text += chunk;
const view = (s) => s && [s.phase, s.headline, s.body, s.activityPreview || ''];
console.log(JSON.stringify(JSON.parse(text).map(([row, late]) => [taskOutcomeSeverity(row), taskReasonDetail(row),
  view(late ? summarizeChatLiveEvent(late) : taskTerminalSummary({...row, outcome_final: true}))])));"""
    # Node writes UTF-8 to a pipe whatever the locale; decoding with the default codec (cp1252 on Windows) breaks "·".
    checked = subprocess.run([node, "--input-type=module", "-e", script], input=json.dumps(rows), capture_output=True,
                             text=True, encoding="utf-8", check=True, cwd=pathlib.Path(__file__).resolve().parents[1])
    return json.loads(checked.stdout)


def test_a_late_current_outcome_reads_like_its_whole_row_in_the_host_and_the_browser():
    """The late projection keeps every fact a finished card's status and cause readers take (``current_outcome``):
    each record of the browser/host parity fixture reads alike whole and projected, through ``outcome_phase`` and
    ``_completion_verdict`` and through their twins ``taskOutcomeSeverity``, ``taskReasonDetail`` and the terminal
    summary; an exhausted delivery keeps its saved-answer pointer."""
    import pathlib

    from ouroboros.late_review_marks import current_outcome
    from ouroboros.outcomes import _merge_objective_warning
    from ouroboros.project_dialogue import _completion_verdict, outcome_phase

    fixture = pathlib.Path(__file__).resolve().parents[1] / "web" / "tests" / "fixtures" / "outcome_phase_parity.json"
    records = [case["record"] for case in json.loads(fixture.read_text(encoding="utf-8"))["cases"]]
    pairs = [(record, current_outcome(record, record.get("terminal_delivery"))) for record in records]
    for whole, projected in pairs:
        assert (outcome_phase(projected, {}), _completion_verdict(projected, {})) == (
            outcome_phase(whole, {}), _completion_verdict(whole, {}))
    # An exhausted delivery reads as its ending row merges it (``terminal_projection._project_row``).
    stored = {"status": "completed", "reason_code": "final_message", "outcome_axes": {"execution": {"status": "ok"}}}
    objective = {}
    _merge_objective_warning(objective, "terminal_delivery_exhausted")
    ending = {**stored, "outcome_axes": {"execution": {"status": "ok"}, "objective": objective},
              "terminal_delivery": {"state": "exhausted", "preserved_path": "/saved/answer.md"}}
    pairs.append((ending, current_outcome(stored, {"state": "exhausted", "delivery_id": "d1",
                                                   "preserved_path": "/saved/answer.md"})))
    views = _js_views([[row, None] for pair in pairs for row in pair])
    assert views[0::2] == views[1::2] and len(views) == 2 * len(pairs) > 300
    assert views[-1][:2] == ["warn", views[-1][1]] and "/saved/answer.md" in views[-1][1]


def test_a_late_pass_keeps_the_saved_candidate_remainder_and_cleanup_its_card_states(tmp_path, monkeypatch):
    """The real seam: a stored outcome whose execution names a saved candidate's uncommitted remainder and its
    cleanup line, resolved by a late PASS, announces a current outcome whose card sentence still states both, exactly
    as the stored row's own readers do; the late event's collapsed line is that cause, never the narration."""
    from ouroboros import review_ledger
    from ouroboros.late_review_marks import resolve_pending_result_review
    from ouroboros.outcomes import _objective_axis
    from ouroboros.project_dialogue import _completion_verdict, outcome_phase
    from ouroboros.task_results import load_task_result, write_task_result

    review = {"status": "awaiting", "source": "result_review", "author_finish": "author_finish_pending",
              "pending_review": {"record_id": "rl-late", "subject_sha": "subject-a"},
              "acceptance_decision": {"status": "finalized_unaccepted", "source": "result_review",
                                      "reason": "author_finish_pending", "enforcement": "advisory"}}
    write_task_result(tmp_path, "task-1", "completed", reason_code="final_message", result="The saved report.",
                      outcome_axes={"execution": {"status": "ok", "reason_code": "final_message",
                                                  "mutation_attribution": {"uncommitted_remainder": {"state": "known", "count": 2}},
                                                  "evolution_cleanup": {"line": "The saved checkout remains for inspection"}},
                                    "review": review, "objective": _objective_axis(review)})
    monkeypatch.setattr(review_ledger, "load_record", lambda _root, record_id: {
        "record_id": record_id, "subject": {"diff_sha": "subject-a"}, "verdict": {"aggregate": "PASS"}, "ts": "t"})
    monkeypatch.setattr("supervisor.message_bus.try_get_bridge", lambda: None)
    assert resolve_pending_result_review(tmp_path, "task-1", "rl-late") is True
    [event] = [json.loads(line) for line in (tmp_path / "logs" / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    row, current = load_task_result(tmp_path, "task-1"), event["current_outcome"]
    sentence = "2 uncommitted files remain in the saved candidate · The saved checkout remains for inspection."
    assert _completion_verdict(current, {}) == _completion_verdict(row, {}) == sentence
    assert outcome_phase(current, {}) == outcome_phase(row, {}) == current["outcome_phase"] == "done"
    assert "The saved report." not in json.dumps(current) and current["_result_text_present"] is True
    [stored, late] = _js_views([[row, None], [current, event]])
    assert stored[:2] == late[:2] == ["done", sentence]
    assert late[2] == ["done", "Done", sentence, sentence], "the card's collapsed line is the cause"


_FAILED_PRE_BRANCH = [{"tool": "submit_skill_to_hub", "is_error": True, "status": "error", "skill_publish_attempt": {
    "ok": False, "skill": "weather", "completed_stage": "local_preflight", "snapshot_hash": "h"}}]
_OVERLAYS = {  # producer: (trace facts, task the publication veto reads, the phase that must stand)
    "publish_failed_pre_branch": ({"tool_calls": _FAILED_PRE_BRANCH}, {"type": "skill_publish", "metadata": {
        "skill_publish_target": {"skill": "weather", "repository": "owner/hub"}}}, "error"),
    "publish_target_missing": ({}, {"type": "skill_publish", "metadata": {}}, "error"),
    "deferred_child": ({"child_result_dispositions": {"deferred_count": 1}}, None, "warn"),
    "delivery_control": ({"delivery_candidate": {"degraded": True}}, None, "warn"),
}


@pytest.mark.parametrize("producer", sorted(_OVERLAYS))
def test_a_late_pass_keeps_what_another_producer_set_over_the_pending_finish(tmp_path, monkeypatch, producer):
    """Reproduced 9 Nov 2026 on the unchanged candidate (an Opus seat's probe): a publication receipt veto, deferred
    child results and delivery control each set the objective over what the pending finish's review gave it, and the
    late PASS of that review erased them (a failed publication read Done). The PASS answers the review alone: the
    review axis passes, the objective keys the pending review derived follow it, and every key another producer set
    stands with its phase and its card sentence, as an independent ``receipt_absent`` already did."""
    from ouroboros import review_ledger
    from ouroboros.late_review_marks import _REVIEW_DERIVED_OBJECTIVE, resolve_pending_result_review
    from ouroboros.outcomes import _objective_axis, derive_loop_outcome
    from ouroboros.project_dialogue import _completion_verdict, outcome_phase
    from ouroboros.skill_publish_result import apply_skill_publish_receipt_veto
    from ouroboros.task_results import load_task_result, write_task_result
    from ouroboros.tools.review_result import _finish_pending

    extra, task, phase = _OVERLAYS[producer]
    trace = {"tool_calls": [], **extra}
    _finish_pending(SimpleNamespace(_completion_selected={"rationale": "The owner needs it now."}, task_metadata={},
                                    task_id="task-1"), trace, "rl-late", SimpleNamespace(diff_sha="subject-a"))
    loop = derive_loop_outcome("Delivered.", {}, trace)
    loop = apply_skill_publish_receipt_veto(loop, task, trace) if task else loop
    write_task_result(tmp_path, "task-1", "completed", reason_code=loop["reason_code"], result="Delivered.",
                      outcome_axes=loop["outcome_axes"])
    before = load_task_result(tmp_path, "task-1")
    stored = before["outcome_axes"]["objective"]
    own = {key: value for key, value in stored.items() if key in _REVIEW_DERIVED_OBJECTIVE
           and value != _objective_axis(before["outcome_axes"]["review"]).get(key)}
    assert own and outcome_phase(before, {}) == phase, (own, stored)
    monkeypatch.setattr(review_ledger, "load_record", lambda _root, record_id: {
        "record_id": record_id, "subject": {"diff_sha": "subject-a"}, "verdict": {"aggregate": "PASS"}, "ts": "t"})
    monkeypatch.setattr("supervisor.message_bus.try_get_bridge", lambda: None)
    assert resolve_pending_result_review(tmp_path, "task-1", "rl-late") is True
    after = load_task_result(tmp_path, "task-1")
    objective = after["outcome_axes"]["objective"]
    assert after["outcome_axes"]["review"]["status"] == "pass" and objective["review_status"] == "pass"
    assert {key: objective.get(key) for key in own} == own, "another producer's objective stands"
    assert objective.get("reason") != "author_finish_pending", "the pending finish's own reason follows its PASS"
    assert {key: objective.get(key) for key in ("receipt_veto", "deferred_count")} == {
        key: stored.get(key) for key in ("receipt_veto", "deferred_count")}
    assert outcome_phase(after, {}) == phase and "before the reviewers answered" not in _completion_verdict(after, {})
    [event] = [json.loads(line) for line in (tmp_path / "logs" / "events.jsonl").read_text(encoding="utf-8").splitlines()
               if '"review_late_result"' in line]
    assert event["current_outcome"]["outcome_phase"] == phase


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
    # The list and the waiting count tell it in the same terms: advice, never an unperformed review with no findings.
    from ouroboros import memory_waiting

    listed = _memory_read(_ctx(harness.drive, "reader"), room_id="late_reviews")
    assert listed.startswith("late reviews: 1 in 1 room; 0 PASS with no findings")
    assert f"plan review of task {PLAN_TASK}; advice, not a verdict; memory_read(node_id='{mark['id']}')" in listed
    counted = memory_waiting.measure([mark], {}, [])["late_reviews"]
    assert (counted["count"], counted["clean_pass"], counted["unrecorded"]) == (1, 0, 0)


# --- outcome facts: what the record said, kept beside the mark's words ----------------------------

def _settled(aggregate, *, critical=0, advisory=0, additional=0, answered=(2, 2), seats=2, state="settled", parts=True):
    """A settled ledger record as ``reduce_verdict`` leaves it: two counted seats, each asked both questions."""
    quorum = {"responded": min(answered), "assigned": seats}
    if parts:
        quorum["parts"] = {name: {"assigned": seats, "responded": got} for name, got in zip(("change", "coupling"), answered)}
    return {"state": state, "verdict": {"aggregate": aggregate, "quorum": quorum,
                                        "critical_findings": [{}] * critical, "advisory_findings": [{}] * advisory,
                                        "additional_findings": [{}] * additional}}


def test_a_marks_outcome_is_a_clean_pass_only_when_its_record_holds_nothing_else(tmp_path):
    """A left wave's mark keeps its record's outcome as facts: a PASS reads clean only with every counted seat
    answered and no finding of any kind; a mark older than these facts answers from its record, never by guess."""
    from ouroboros.late_review_marks import clean_pass, mark_outcome, outcome_facts, outcome_words
    from ouroboros.review_ledger import ledger_dir, record_path

    clean = outcome_facts(_settled("PASS"))
    assert clean == {"verdict": "PASS", "critical": 0, "advisory": 0, "additional": 0, "asked": 4, "answered": 4,
                     "unheard": 0}
    assert clean_pass(clean) and outcome_words(clean) == "PASS, no findings"
    seats_only = outcome_facts(_settled("PASS", parts=False))  # a record that keeps no per-question tally
    assert (seats_only["asked"], seats_only["answered"]) == (2, 2) and clean_pass(seats_only)
    others = {"PASS, 1 advisory": _settled("PASS", advisory=1), "PASS, 2 additional": _settled("PASS", additional=2),
              "PASS, no findings, 1 asked answer missing": _settled("PASS", answered=(2, 1)),
              "FAIL, 2 critical, 1 advisory": _settled("FAIL", critical=2, advisory=1),
              "QUORUM_FAILED, no findings, 4 asked answers missing": _settled("QUORUM_FAILED", answered=(0, 0)),
              "no verdict formed": _settled("PASS", state="pending")}
    for words, record in others.items():
        assert outcome_words(outcome_facts(record)) == words and not clean_pass(outcome_facts(record)), words
    # An adviser is heard outside the count: one that gave no answer leaves something unread, one that answered
    # with nothing to say does not, and a counted seat's missing answer is told as an asked answer, not here.
    def advised(status, additional=True):
        return outcome_facts({**_settled("PASS"), "rows": [{"seat_id": "s1", "status": "responded"},
                                                           {"seat_id": "adv", "status": status, "additional": additional}]})
    assert outcome_words(advised("failed")) == "PASS, no findings, 1 adviser answer missing"
    assert not clean_pass(advised("failed")) and not clean_pass(advised("custody_lost"))
    assert clean_pass(advised("responded")) and clean_pass(advised("failed", additional=False))
    # An adviser asked both questions counts each unreadable answer: a readable change answer beside a coupling
    # matrix it left unanswered or broken (its critical rows kept only as discarded diagnostics) is no clean PASS.
    def asked_of(change, coupling):
        return outcome_facts({**_settled("PASS"), "rows": [{"seat_id": "s1", "status": "responded"}, {
            "seat_id": "adv", "status": "responded", "additional": True, "parts": ["change", "coupling"],
            "answers": {"change": {"status": change}, "coupling": {"status": coupling}}}]})
    half = asked_of("responded", "unanswered")
    assert outcome_words(half) == "PASS, no findings, 1 adviser answer missing" and not clean_pass(half)
    assert outcome_words(asked_of("unanswered", "unanswered")) == "PASS, no findings, 2 adviser answers missing"
    assert clean_pass(asked_of("responded", "responded"))
    assert outcome_facts(None) is None and not clean_pass(None), "no record says nothing, never a verdict"
    earlier = {name: value for name, value in clean.items() if name != "unheard"}  # kept before advisers were counted
    assert not clean_pass(earlier), "an unknown number of unheard advisers is not zero"
    assert outcome_words(earlier) == "PASS, no findings, adviser answers not recorded"
    advice = outcome_facts({**_settled("NOT_PERFORMED"), "surface": "intention"})  # a plan council asks no verdict
    assert advice == {"advice": True} and outcome_words(advice) == "advice, not a verdict" and not clean_pass(advice)
    assert outcome_facts({**_settled("PASS"), "surface": "commit_gate"}) == clean, "a review that asks one keeps it"
    assert not clean_pass({"verdict": "PASS"}), "a PASS that names no asked answer is not known to be whole"
    assert outcome_words(None) == "outcome not recorded"

    shared.world(tmp_path)
    assert publish_late_review(tmp_path, {**_record(tmp_path, "main1", 1, "clean"), "outcome": clean}).status == "saved"
    [mark] = _late_marks(tmp_path)
    assert mark["review"]["outcome"] == clean and mark_outcome(mark) == clean
    record_id = "rl-20261017T000000-aaaaaaaaaaaa"
    older = {**_record(tmp_path, "bound", 1, "older"), "subject": {"record_id": record_id}}
    assert publish_late_review(tmp_path, older).status == "saved"
    [old] = [m for m in _late_marks(tmp_path) if m["review"]["task_id"] == "bound"]
    assert "outcome" not in old["review"] and mark_outcome(old) is None, "no read without the data root"
    assert mark_outcome(old, tmp_path) is None, "an absent record is not a clean one"
    ledger_dir(tmp_path, create=True)
    record_path(tmp_path, record_id).write_text(json.dumps(_settled("PASS", advisory=1)), encoding="utf-8")
    assert outcome_words(mark_outcome(old, tmp_path)) == "PASS, 1 advisory"
    record_path(tmp_path, record_id).write_text("{broken", encoding="utf-8")
    assert mark_outcome(old, tmp_path) is None, "an unreadable record is not a clean one"
    # A mark whose kept outcome does not say how many advisers went unheard answers from its record once the list
    # is asked with the data root, and keeps what it has without the root or without a readable record.
    held_id = "rl-20261017T000000-cccccccccccc"
    held = {**_record(tmp_path, "beta1", 1, "kept earlier"), "outcome": earlier, "subject": {"record_id": held_id}}
    assert publish_late_review(tmp_path, held).status == "saved"
    [kept] = [m for m in _late_marks(tmp_path) if m["review"]["task_id"] == "beta1"]
    assert mark_outcome(kept) == earlier == mark_outcome(kept, tmp_path), "no root or no record: what it has"
    record_path(tmp_path, held_id).write_text(json.dumps({**_settled("PASS"), "rows": [
        {"seat_id": "adv", "status": "failed", "additional": True}]}), encoding="utf-8")
    assert mark_outcome(kept) == earlier, "the waiting line reads no record"
    assert outcome_words(mark_outcome(kept, tmp_path)) == "PASS, no findings, 1 adviser answer missing"
    record_path(tmp_path, held_id).write_text(json.dumps(_settled("PASS")), encoding="utf-8")
    assert clean_pass(mark_outcome(kept, tmp_path)) and mark_outcome(mark, tmp_path) == clean, "whole facts: no read"
    bad = publish_late_review(tmp_path, {**_record(tmp_path, "beta2", 1, "x"), "outcome": {"verdict": ["PASS"]}})
    assert bad.status == "invalid" and "outcome" in bad.detail
    # A wave whose record could not be read publishes no outcome: the list reads its record whenever it can.
    assert publish_late_review(tmp_path, {**_record(tmp_path, "pres1", 1, "gap"), "outcome": None}).status == "saved"
    [gap] = [m for m in _late_marks(tmp_path) if m["review"]["task_id"] == "pres1"]
    assert "outcome" not in gap["review"]


def test_a_late_commit_review_tells_what_is_recorded_of_the_change_it_read(tmp_path):
    """The list says what became of a late commit review's change as it reads now: the commit under this review
    and its tree's relation to the reviewed one, else the pull request a receipt of its task records as merged,
    else its latest receipt with that receipt's outcome, else that neither is recorded. The head counts the PASS
    commit reviews with no commit and no merged receipt; an unread fact is told as unread, never as nothing landed."""
    from ouroboros.late_review_marks import landing_words, mark_landing, outcome_facts
    from ouroboros.review_ledger import ledger_dir, record_path

    shared.world(tmp_path)
    ledger_dir(tmp_path, create=True)
    sha = "ab" * 20

    def late(task, *, surface="commit_gate", verdict="PASS", landings=(), receipts=None):
        record_id = f"rl-20261017T000000-{'abcdef'[len(_late_marks(tmp_path))] * 12}"
        record_path(tmp_path, record_id).write_text(json.dumps({
            **_settled(verdict), "commit_landings": [{"commit_sha": commit, "reviewed_subject": tree}
                                                     for commit, tree in landings]}), encoding="utf-8")
        if receipts is not None:
            seed_result(tmp_path, task, {"status": "completed", "chat_id": 1, "merge_receipts": receipts})
        published = {**_record(tmp_path, task, 1, "late words"), "surface": surface,
                     "outcome": outcome_facts(_settled(verdict)), "subject": {"record_id": record_id}}
        assert publish_late_review(tmp_path, published).status == "saved"
        return next(m for m in _late_marks(tmp_path) if m["review"]["task_id"] == task), record_id

    committed, _ = late("main1", landings=[(sha, "same")])
    changed, _ = late("main2", landings=[(sha, "different")])
    pull = "https://github.com/example/body/pull/3104"
    merged, _ = late("bound", receipts=[{"number": 7, "outcome": {"status": "queued"}},
                                        {"number": 3104, "repo": {"url": pull}, "outcome": {"status": "merged"}}])
    asked, asked_record = late("beta1", receipts=[{"number": 8, "outcome": {"status": "unknown"}}])
    failed, _ = late("pres1", verdict="FAIL", receipts=[])
    nothing, _ = late("pres2", receipts=[])
    assert mark_landing(committed, tmp_path) == {"commit": sha, "tree": "same"}
    assert landing_words(mark_landing(committed, tmp_path)) == "committed as abababababab under this review"
    assert landing_words(mark_landing(changed, tmp_path)) == ("committed as abababababab under this review; its tree "
                                                              "differs from the reviewed one")
    assert landing_words({"commit": sha, "tree": "unknown"}).endswith("; its relation to the reviewed tree is unknown")
    assert mark_landing(merged, tmp_path) == {"pr": pull}
    assert landing_words({"pr": pull}) == f"a receipt of its task records {pull} as merged"
    assert mark_landing(asked, tmp_path) == {"pr": "PR #8", "status": "unknown"}
    assert landing_words(mark_landing(asked, tmp_path)) == "a receipt of its task records PR #8 as unknown"
    assert mark_landing(failed, tmp_path) == {} == mark_landing(nothing, tmp_path)
    assert landing_words({}) == "no commit under this review, no merged receipt of its task"

    listed = _memory_read(_ctx(tmp_path, "reader"), room_id="late_reviews")
    assert listed.splitlines()[0].endswith("; 5 PASS with no findings; 2 PASS commit reviews with no commit under that "
                                           "review and no merged receipt of their task")
    for task, words in (("main1", "PASS, no findings; committed as abababababab under this review"),
                        ("main2", "PASS, no findings; committed as abababababab under this review; its tree differs"),
                        ("bound", f"PASS, no findings; a receipt of its task records {pull} as merged"),
                        ("beta1", "PASS, no findings; a receipt of its task records PR #8 as unknown"),
                        ("pres1", "FAIL, no findings; no commit under this review, no merged receipt of its task"),
                        ("pres2", "PASS, no findings; no commit under this review, no merged receipt of its task")):
        assert f"commit review of task {task}; {words}" in listed, task

    # What cannot be read says so, and every other kind of review says nothing about landing.
    other = {"review": {**asked["review"], "surface": "result"}}
    unknown_task = {"review": {**asked["review"], "task_id": "no-such-task"}}
    assert mark_landing(other, tmp_path) is None and landing_words(None) == ""
    assert mark_landing(unknown_task, tmp_path) == {"unread": True}
    assert landing_words({"unread": True}) == "what landed could not be read"
    record_path(tmp_path, asked_record).write_text("{broken", encoding="utf-8")
    assert mark_landing(asked, tmp_path) == {"unread": True}
    record_path(tmp_path, asked_record).unlink()
    assert mark_landing(asked, tmp_path) == {"unread": True}
    again = _memory_read(_ctx(tmp_path, "reader"), room_id="late_reviews")
    assert "commit review of task beta1; PASS, no findings; what landed could not be read; memory_read(" in again
    assert again.splitlines()[0].endswith("; 1 PASS commit reviews with no commit under that review and no merged receipt "
                                          "of their task")


def test_a_queued_receipt_names_its_later_readback_and_an_unknown_tree_relation_is_said(tmp_path):
    """A queued merge whose pull request reads otherwise since says so in the owner module's words
    (``merge_receipts.queued_earlier_note``); a landing whose tree relation is not known to be the same says it is
    unknown; a receipt told as merged is counted as merged by the list's head as by its words."""
    from ouroboros.late_review_marks import landing_words, mark_landing, outcome_facts
    from ouroboros.memory_waiting import listing
    from ouroboros.merge_receipts import queued_earlier_note
    from ouroboros.review_ledger import ledger_dir, record_path

    shared.world(tmp_path)
    ledger_dir(tmp_path, create=True)
    record_id = "rl-20261017T000000-" + "a" * 12
    record_path(tmp_path, record_id).write_text(json.dumps(_settled("PASS")), encoding="utf-8")
    receipt = {"number": 9, "outcome": {"status": "queued"}, "observed_after": {"state": "CLOSED", "auto_merge": False},
               "effect": {"ok": True}}
    seed_result(tmp_path, "queue1", {"status": "completed", "chat_id": 1, "merge_receipts": [receipt]})
    published = {**_record(tmp_path, "queue1", 1, "late words"), "surface": "commit_gate",
                 "outcome": outcome_facts(_settled("PASS")), "subject": {"record_id": record_id}}
    assert publish_late_review(tmp_path, published).status == "saved"
    mark = next(m for m in _late_marks(tmp_path) if m["review"]["task_id"] == "queue1")
    note = queued_earlier_note(receipt)
    assert note.startswith("earlier; PR now CLOSED")
    assert mark_landing(mark, tmp_path) == {"pr": "PR #9", "status": "queued", "note": note}
    assert landing_words(mark_landing(mark, tmp_path)) == f"a receipt of its task records PR #9 as queued ({note})"
    still_open = {**receipt, "observed_after": {"state": "OPEN", "auto_merge": True}}
    assert queued_earlier_note(still_open) == "", "a queue the readback still shows needs no note"
    assert landing_words({"pr": "PR #9", "status": "queued"}) == "a receipt of its task records PR #9 as queued"
    for tree in (None, "", "weird", "unknown"):
        assert landing_words({"commit": "ab" * 20, "tree": tree}) == (
            "committed as abababababab under this review; its relation to the reviewed tree is unknown"), tree
    assert landing_words({"commit": "ab" * 20, "tree": "same"}) == "committed as abababababab under this review"
    heads = [listing("late_reviews", [mark], {}, landing=lambda _m, became=became: became)[0]
             for became in ({"pr": "PR #9"}, {"pr": "PR #9", "status": "merged"}, {"pr": "PR #9", "status": "queued"})]
    assert heads[0] == heads[1] != heads[2], heads
