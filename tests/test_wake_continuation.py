"""The mind's last authored continuation across a wake it did not end: the real clock, settlement and first request.

The newest attempt's outcome is always shown. Typed terminal facts alone say whether an author ended it: an authored
end stands alone however it ended and however empty its note; a host's end or an unrecorded author keeps the last
authored note beside it as dated understanding, and an attempt no record attributes stays named as a doubt until an
author ends a wake. Its row carries those notes; the live clock also keeps its last computation if persistence fails.
An unreadable row preserves doubt. A restart cannot recover state never saved. No text is classified; no history walk.
"""

from __future__ import annotations

import hashlib
import json
import pathlib

import pytest

from ouroboros import consciousness_wake as wake
from tests import test_consciousness as clock_tests
from tests._wake_summary_shared import seed_wake

REPO = pathlib.Path(__file__).resolve().parents[1]
T0 = 1_800_000_000.0
clock = clock_tests.clock
HOST_WORDS = "⚠️ The provider stopped this wake before any answer."
HOST = {"status": "failed", "result": HOST_WORDS, "terminal_origin": "host_notice"}
UNATTRIBUTED = {"status": "failed", "result": "words of nobody recorded"}
CLEARED = {"status": "completed", "result": "", "terminal_origin": "model_final", "wake_silent": True}


def _raise(exc):
    raise exc


def _launch(clock, number):
    clock.receipt["task_id"] = f"wake{number:04d}"
    clock.clock._next_wake_at = T0 + number * clock_tests.FLOOR
    assert clock.clock.tick(T0 + number * clock_tests.FLOOR) == "launched"
    return clock.launches[-1]


def _previous(launch):
    text = launch["text"].split("- previous wake: ")[1].split("\n- recent facts")[0]
    assert text in launch["task"]["metadata"]["wake_observation"]["projection_text"]  # both deliveries of the input carry it
    return text


def _settle(clock, monkeypatch, number, attempt):
    """The real settlement of one launched attempt; ``settle`` fails the clock's row read or its state write meanwhile."""
    import ouroboros.task_results as results
    from supervisor import state

    attempt, task_id = dict(attempt), f"wake{number:04d}"
    failure = attempt.pop("settle", "")
    document = {"terminal_origin": None, "wake_summary": None, "outcome_axes": {}, "terminal_time": None, **attempt}
    seed_wake(clock.root, task_id=task_id, metadata=clock.launches[-1]["task"]["metadata"], **document)
    with monkeypatch.context() as fault:
        if failure == "read":
            real = results.load_task_result
            fault.setattr(results, "load_task_result", lambda root, tid, **kw: _raise(OSError("store busy"))
                          if tid == task_id and kw.get("strict") else real(root, tid, **kw))
        elif failure == "write":
            fault.setattr(state, "update_state", lambda *a, **kw: _raise(state.StateUnavailable("state lock busy")))
        clock.finish(task_id, document.get("status", "failed") == "completed")


def _settled(clock, monkeypatch, *attempts):
    """A's authored end, then each attempt's real settlement in turn."""
    clock.root.joinpath("repo/prompts/CONSCIOUSNESS.md").write_text(
        REPO.joinpath("prompts/CONSCIOUSNESS.md").read_text(encoding="utf-8"), encoding="utf-8")
    launch = _launch(clock, 1)
    seed_wake(clock.root, task_id="wake0001", result="A's whole note.", wake_summary="A's line.",
              metadata=launch["task"]["metadata"])
    clock.finish("wake0001", True)
    for number, attempt in enumerate(attempts, start=2):
        _launch(clock, number)
        _settle(clock, monkeypatch, number, attempt)


def _chain(clock, monkeypatch, *attempts, restart=False):
    """``_settled``, then the next launch's previous-wake text."""
    from ouroboros.consciousness import BackgroundConsciousness

    _settled(clock, monkeypatch, *attempts)
    if restart:
        clock.clock = BackgroundConsciousness(clock.root, clock.root / "repo", lambda: 7,
                                              now=T0 + (len(attempts) + 1) * clock_tests.FLOOR)
    return _previous(_launch(clock, len(attempts) + 2))


def _wake_after(clock, monkeypatch, restart=False, **attempt):
    return _chain(clock, monkeypatch, attempt, restart=restart)


@pytest.mark.parametrize("restart", [False, True])
def test_a_host_only_failed_wake_keeps_its_outcome_beside_the_last_authored_note(clock, monkeypatch, restart):
    previous = _wake_after(clock, monkeypatch, restart=restart, **HOST)
    assert previous.startswith(f"wake0002, Failed at an unknown time; its line: no line; the host's notice, not a note of mine: «{HOST_WORDS}»")
    assert "its note" not in previous.split("My last note before it:")[0]  # the host's words never read as the mind's
    assert "My last note before it: wake0001, Done " in previous and "its line: «A's line.»; its note: «A's whole note.»" in previous
    assert previous.endswith("— dated, what I understood then")


def test_a_failed_attempt_points_to_its_retained_observation(clock, monkeypatch):
    """Binding retains the source, even when no physical model request ran. It proves no delivery or reading."""
    previous = _wake_after(clock, monkeypatch, **HOST)
    attempt, note = previous.split(". My last note before it: ")
    source = clock.launches[1]["task"]["metadata"]["wake_observation"]["source"]
    path = source["read"]["arguments"]["path"]
    assert attempt.endswith(f"; retained observation: read_file(root='runtime_data', path='{path}', start_line=1)")
    data = clock.root.joinpath(path).read_bytes()
    assert hashlib.sha256(data).hexdigest() == source["sha256"] and json.loads(data.splitlines()[0])["kind"] == "wake_observation"
    assert "retained observation" not in note, "the latest deliberate note remains the continuation"


def test_words_the_host_retained_are_not_taken_for_a_chosen_note(clock, monkeypatch):
    previous = _wake_after(clock, monkeypatch, status="failed", result="half a thought", terminal_origin="host_salvage")
    assert "words the host kept from its unfinished turn, not a note I chose: «half a thought»" in previous
    assert "its note: «A's whole note.»" in previous


@pytest.mark.parametrize("attempt,note", [
    ({"status": "failed", "result": "B failed, and this is why I stop.", "terminal_origin": "model_final"}, "B failed, and this is why I stop."),
    (CLEARED, ""),  # deliberately empty
    ({"status": "completed", "result": "", "wake_summary": "B cleared its plans."}, ""),  # only an accepting exit publishes a line
], ids=["author_nonclean", "author_cleared", "line_without_origin"])
def test_an_authored_end_stands_alone_however_it_ended_and_however_empty_its_note(clock, monkeypatch, attempt, note):
    previous = _wake_after(clock, monkeypatch, **attempt)
    assert previous.endswith(f"its note: «{note}»") and previous.startswith("wake0002, ")
    assert "A's whole note." not in previous and "dated" not in previous, "an authored decision never brings an older note back"


@pytest.mark.parametrize("status,word", [("failed", "Failed"), ("cancelled", "Stopped"), ("interrupted", "Interrupted")])
def test_unknown_authorship_is_no_proof_of_silence_and_names_what_it_may_have_changed(clock, monkeypatch, status, word):
    previous = _wake_after(clock, monkeypatch, status=status, result="words of nobody recorded")
    assert previous.startswith(f"wake0002, {word} at an unknown time; its line: no line; its note, author not recorded: «words of nobody recorded»")
    assert "My last known note before it: wake0001, " in previous and "its note: «A's whole note.»" in previous
    assert previous.endswith("what I understood then; the attempt above may have changed it")


@pytest.mark.parametrize(("host_runs", "restart"), [(1, False), (2, True)])
def test_an_unattributed_attempt_stays_in_doubt_through_later_host_failures(clock, monkeypatch, restart, host_runs):
    previous = _chain(clock, monkeypatch, UNATTRIBUTED, *[HOST] * host_runs, restart=restart)
    latest = f"wake{host_runs + 2:04d}"
    assert previous.startswith(f"{latest}, Failed at an unknown time; its line: no line; the host's notice, not a note of mine: «{HOST_WORDS}»")
    assert (". Before it: wake0002, Failed at an unknown time; its line: no line; "
            "its note, author not recorded: «words of nobody recorded»") in previous
    assert "My last known note before it: wake0001, " in previous and "its note: «A's whole note.»" in previous
    assert previous.endswith("— dated, what I understood then; wake0002 may have changed it")
    assert host_runs == 1 or "wake0003" not in previous, "a host's own end between them adds no doubt and no read"


def test_an_authored_end_settles_the_doubt_and_a_cleared_note_stays_cleared(clock, monkeypatch):
    previous = _chain(clock, monkeypatch, UNATTRIBUTED, UNATTRIBUTED, CLEARED, HOST)
    assert previous.startswith("wake0005, Failed ") and previous.endswith(
        "My last note before it: wake0004, Done at an unknown time; its line: no line; its note: «» — dated, what I understood then")
    assert "wake0002" not in previous and "A's whole note." not in previous and "last known note" not in previous


@pytest.mark.parametrize("unknowns", [1, 2, 7])
@pytest.mark.parametrize("settlement", ["host", "authored"])
def test_resolving_one_attempt_does_not_silently_resolve_an_earlier_span(clock, monkeypatch, unknowns, settlement):
    from ouroboros import task_store

    _settled(clock, monkeypatch, *[UNATTRIBUTED] * unknowns, HOST)
    task_id = f"wake{unknowns + 1:04d}"
    changed = HOST if settlement == "host" else CLEARED
    seed_wake(clock.root, task_id=task_id, metadata=clock.launches[unknowns]["task"]["metadata"],
              wake_summary=None, outcome_axes={}, terminal_time=None, **changed)
    reads, original = [], task_store.read_document
    monkeypatch.setattr(task_store, "read_document", lambda root, tid: (reads.append(tid), original(root, tid))[1])
    carried = {}
    latest = clock.launches[-1]["task"]["metadata"]["wake_continuity"]
    previous = wake.previous_wake_text(clock.root, f"wake{unknowns + 2:04d}", now=T0, carried=carried, **latest)
    assert len(reads) <= 3, "resolution never walks the earlier attempts"
    if settlement == "authored":
        assert "A's whole note." not in previous and "may have changed it" not in previous
        assert carried == {"last_authored": task_id, "unattributed": "", "earlier_unattributed": False}
    elif unknowns == 1:
        assert "A's whole note." in previous and "may have changed it" not in previous
    else:
        assert "A's whole note." in previous and "last known note" in previous
        assert f"earlier unattributed attempts before {task_id} may have changed it" in previous
        assert carried["earlier_unattributed"] is True
        assert carried["unattributed"] == task_id, "the named row retains its predecessors' exact readers"


@pytest.mark.parametrize("failure", ["read", "write"])
def test_a_settlement_that_could_not_read_or_record_never_brings_an_older_note_back(clock, monkeypatch, failure):
    """B ends its plans deliberately while its settlement fails; C reads B; C's host-only end must not revive A."""
    previous = _chain(clock, monkeypatch, {**CLEARED, "settle": failure}, HOST)
    assert previous.startswith("wake0003, Failed ") and previous.endswith(
        "My last note before it: wake0002, Done at an unknown time; its line: no line; its note: «» — dated, what I understood then")
    assert "A's whole note." not in previous


@pytest.mark.parametrize("note_state", ["unreadable", "absent", "none-known"])
def test_earlier_uncertainty_is_visible_even_without_a_readable_note(clock, monkeypatch, note_state):
    from ouroboros import task_store

    _settled(clock, monkeypatch, UNATTRIBUTED, UNATTRIBUTED, HOST)
    seed_wake(clock.root, task_id="wake0003", metadata=clock.launches[2]["task"]["metadata"],
              wake_summary=None, outcome_axes={}, terminal_time=None, **HOST)
    read = task_store.read_document

    def changed(root, tid):
        if tid == "wake0001":
            if note_state == "unreadable":
                raise OSError("unreadable note")
            return None if note_state == "absent" else UNATTRIBUTED
        row = read(root, tid)
        if tid == "wake0004" and note_state == "none-known":
            row["metadata"]["wake_continuity"]["last_authored"] = ""
        return row

    monkeypatch.setattr(task_store, "read_document", changed)
    previous = _previous(_launch(clock, 5))
    assert "wake0004, Failed" in previous and "wake0003, Failed" in previous
    assert "unresolved authorship: earlier unattributed attempts before wake0003" in previous
    assert "A's whole note." not in previous


@pytest.mark.parametrize("note", ["", "B's whole note."])
def test_authored_row_survives_failed_state_then_failed_preparation(clock, monkeypatch, note):
    """Read the named predecessor before fallible observation, without a second settlement writer."""
    from ouroboros import consciousness_wake
    from supervisor import state

    _settled(clock, monkeypatch)
    authored = _launch(clock, 2)
    seed_wake(clock.root, task_id="wake0002", result=note, wake_summary=None,
              terminal_origin="model_final", metadata=authored["task"]["metadata"])
    with monkeypatch.context() as fault:
        fault.setattr(state, "update_state", lambda *args, **kwargs: _raise(state.StateUnavailable("busy")))
        clock.finish("wake0002", True)
    with monkeypatch.context() as prep_fault:
        prep_fault.setattr(consciousness_wake, "observe_wake", lambda *args, **kwargs: _raise(OSError("busy")))
        failed_prepare = _launch(clock, 3)
    seed_wake(clock.root, task_id="wake0003", result=HOST_WORDS, terminal_origin="host_notice",
              metadata=failed_prepare["task"]["metadata"])
    clock.finish("wake0003", False)
    previous = _previous(_launch(clock, 4))
    assert "wake0002" in previous and f"its note: «{note}»" in previous
    assert "wake0001" not in previous, "the authored B row must supersede A even when B's state write failed"


@pytest.mark.parametrize("note", ["", "B's newest note."])
def test_failed_settlement_then_missing_carrier_metadata_does_not_revive_the_older_note(clock, monkeypatch, note):
    """The next carrier's host-only terminal can omit metadata entirely (before-start stop or lost preparation)."""
    from supervisor import state

    _settled(clock, monkeypatch)
    launch = _launch(clock, 2)
    seed_wake(clock.root, task_id="wake0002", result=note, terminal_origin="model_final", wake_summary=None,
              metadata=launch["task"]["metadata"])
    with monkeypatch.context() as fault:
        fault.setattr(state, "update_state", lambda *args, **kwargs: _raise(state.StateUnavailable("busy")))
        clock.finish("wake0002", True)
    _launch(clock, 3)
    seed_wake(clock.root, task_id="wake0003", status="failed", result=HOST_WORDS, terminal_origin="host_notice",
              wake_summary=None, metadata={})
    clock.finish("wake0003", False)
    previous = _previous(_launch(clock, 4))
    assert "wake0002" in previous and f"its note: «{note}»" in previous
    assert "A's whole note." not in previous


def test_a_wake_whose_preparation_failed_hands_the_doubt_on_through_the_clock(clock, monkeypatch):
    """C runs on admitted text alone; earlier continuity is already captured before observation fails."""
    from ouroboros import consciousness_wake

    _settled(clock, monkeypatch, UNATTRIBUTED)
    with monkeypatch.context() as fault:
        fault.setattr(consciousness_wake, "observe_wake", lambda *a, **kw: _raise(OSError("chat store busy")))
        unprepared = _launch(clock, 3)
    assert unprepared["task"]["metadata"]["wake_continuity"]["unattributed"] == "wake0002"
    assert "wake_observation" not in unprepared["task"]["metadata"]
    _settle(clock, monkeypatch, 3, HOST)
    previous = _previous(_launch(clock, 4))
    assert previous.startswith("wake0003, Failed ") and "retained observation" not in previous.split(". Before it: ")[0]
    assert ". Before it: wake0002, " in previous and previous.endswith("; wake0002 may have changed it")


def test_an_upgrade_seeds_the_note_from_the_finished_predecessor_alone(clock, monkeypatch):
    """A body that kept no continuity yet: the exact predecessor, one addressed read, not a history walk."""
    from ouroboros.consciousness import OWN_WAKE_STATE_KEY, BackgroundConsciousness

    clock.root.joinpath("repo/prompts/CONSCIOUSNESS.md").write_text(
        REPO.joinpath("prompts/CONSCIOUSNESS.md").read_text(encoding="utf-8"), encoding="utf-8")
    seed_wake(clock.root, task_id="older1", result="An older body's note.", wake_summary="Older line.")
    seed_wake(clock.root, task_id="old0001", result="The note before the update.", wake_summary="Its line.")
    clock.store[OWN_WAKE_STATE_KEY] = {"task_id": "old0001", "wake_cause": "own_clock", "initiator": "consciousness",
                                       "finished": True}
    clock.clock = BackgroundConsciousness(clock.root, clock.root / "repo", lambda: 7, now=T0)
    first = _previous(_launch(clock, 1))
    assert first.startswith("old0001, Done ") and first.endswith("its note: «The note before the update.»")
    _settle(clock, monkeypatch, 1, HOST)
    previous = _previous(_launch(clock, 2))
    assert previous.startswith("wake0001, Failed ") and "My last note before it: old0001, " in previous
    assert previous.endswith("its note: «The note before the update.» — dated, what I understood then")
    assert "older1" not in previous


def test_an_unreadable_newest_attempt_keeps_the_last_note_in_doubt(clock, monkeypatch):
    from ouroboros import task_store

    seed_wake(clock.root, task_id="authored", result="The note.", wake_summary="Its line.", terminal_origin="model_final")
    previous = wake.previous_wake_text(clock.root, "missing", now=T0, last_authored="authored")
    prefix = "missing, unreadable; get_task_result(task_id='missing') reads it"
    assert previous.startswith(prefix + ". My last known note before it: authored, Done ")
    assert previous.endswith("its note: «The note.» — dated, what I understood then; the attempt above may have changed it")
    assert wake.previous_wake_text(clock.root, "missing", now=T0) == prefix, "the missing row's identity is still known"
    original = task_store.read_document
    monkeypatch.setattr(task_store, "read_document", lambda root, task_id: _raise(OSError("store busy"))
                        if task_id == "newest" else original(root, task_id))
    assert wake.previous_wake_text(clock.root, "newest", now=T0, last_authored="authored") == previous.replace("missing", "newest")


def test_a_doubt_is_read_by_its_current_facts_in_three_addressed_reads_at_most(clock, monkeypatch):
    from ouroboros import task_store

    seed_wake(clock.root, task_id="note-a", result="A.", wake_summary="A line.", terminal_origin="model_final")
    seed_wake(clock.root, task_id="host1", wake_summary=None, outcome_axes={}, terminal_time=None, **HOST)
    seed_wake(clock.root, task_id="doubt", terminal_origin=None, wake_summary=None, outcome_axes={}, terminal_time=None,
              **UNATTRIBUTED)
    seed_wake(clock.root, task_id="now-mine", result="Resumed and ended by me.", terminal_origin="model_final")
    seed_wake(clock.root, task_id="now-host", wake_summary=None, outcome_axes={}, terminal_time=None, **HOST)
    reads, original = [], task_store.read_document
    monkeypatch.setattr(task_store, "read_document", lambda root, task_id: (reads.append(task_id), original(root, task_id))[1])

    def text(unattributed):
        reads.clear()
        return wake.previous_wake_text(clock.root, "host1", now=T0, last_authored="note-a", unattributed=unattributed)

    doubted = text("doubt")
    assert reads == ["host1", "doubt", "note-a"], "the newest attempt, the doubt and the note, by primary key"
    assert ". Before it: doubt, Failed " in doubted and doubted.endswith("; doubt may have changed it")
    mine = text("now-mine")
    assert "My last note before it: now-mine, " in mine and "«A.»" not in mine and "last known note" not in mine
    assert text("now-host").endswith("its note: «A.» — dated, what I understood then") and "now-host" not in text("now-host")
    gone = text("gone")
    assert ". Before it: gone, unreadable. My last known note before it: note-a, " in gone and gone.endswith("; gone may have changed it")
    assert text("host1").endswith("— dated, what I understood then") and "Before it" not in text("host1")


@pytest.mark.parametrize("shape", ["unreadable", "absent", "running"])
def test_an_own_wake_the_clock_starts_over_stays_a_doubt_not_a_silence(clock, monkeypatch, shape):
    """The clock cannot attribute the attempt it replaces; a later host-only failure keeps that doubt in view."""
    import ouroboros.task_results as results
    from tests._store_shared import seed_result

    clock.root.joinpath("repo/prompts/CONSCIOUSNESS.md").write_text(
        REPO.joinpath("prompts/CONSCIOUSNESS.md").read_text(encoding="utf-8"), encoding="utf-8")
    launch = _launch(clock, 1)
    seed_wake(clock.root, task_id="wake0001", result="A's whole note.", wake_summary="A's line.",
              metadata=launch["task"]["metadata"])
    clock.finish("wake0001", True)
    lost = _launch(clock, 2)
    clock.running.pop("wake0002")  # its carrier is gone and no terminal reached the clock
    if shape == "running":
        seed_result(clock.root, "wake0002", {"status": "running", "metadata": lost["task"]["metadata"]})
    with monkeypatch.context() as fault:
        if shape == "unreadable":
            real = results.load_task_result
            fault.setattr(results, "load_task_result", lambda root, tid, **kw: _raise(OSError("store busy"))
                          if tid == "wake0002" else real(root, tid, **kw))
        over = _launch(clock, 3)
    assert _previous(over).endswith("the attempt above may have changed it")
    _settle(clock, monkeypatch, 3, HOST)
    previous = _previous(_launch(clock, 4))
    assert previous.startswith("wake0003, Failed ") and ". Before it: wake0002, " in previous
    assert "My last known note before it: wake0001, " in previous and previous.endswith("; wake0002 may have changed it")


@pytest.mark.parametrize("status", ["completed", "cancelled", "interrupted"])
def test_a_terminal_nobody_reported_still_records_its_author_before_the_clock_starts_over_it(clock, status):
    from ouroboros.consciousness import CONTINUITY_STATE_KEY

    clock.clock._own_wake = {"task_id": "lost1", "wake_cause": "own_clock", "initiator": "consciousness", "finished": False}
    seed_wake(clock.root, task_id="lost1", status=status, result="An authored end nobody reported.")
    clock.clock._settle_unreported_terminal()
    assert clock.store[CONTINUITY_STATE_KEY] == {"last_authored": "lost1", "unattributed": "", "earlier_unattributed": False}
    seed_wake(clock.root, task_id="lost2", status="failed", result=HOST_WORDS, terminal_origin="host_notice", wake_summary=None)
    clock.clock._own_wake = {"task_id": "lost2", "wake_cause": "own_clock", "initiator": "consciousness", "finished": False}
    clock.clock._settle_unreported_terminal()
    assert clock.store[CONTINUITY_STATE_KEY]["last_authored"] == "lost1", "a host-only row never moves the note"


def test_a_run_of_host_only_wakes_still_reaches_the_note_by_two_addressed_reads(clock, monkeypatch):
    from ouroboros import task_store

    row = seed_wake(clock.root, task_id="authored", result="The note.", wake_summary="Its line.", terminal_origin="model_final")
    for task_id in ("host1", "host2"):
        seed_wake(clock.root, task_id=task_id, status="failed", result=HOST_WORDS, terminal_origin="host_notice",
                  wake_summary=None, outcome_axes={}, terminal_time=None)
    reads, original = [], task_store.read_document
    monkeypatch.setattr(task_store, "read_document", lambda root, task_id: (reads.append(task_id), original(root, task_id))[1])
    previous = wake.previous_wake_text(clock.root, "host2", now=T0, last_authored=row["task_id"])
    assert reads == ["host2", "authored"], "no history walk: the newest attempt and the one note, by primary key"
    assert "«The note.»" in previous and "host1" not in previous
    reads.clear()
    assert "dated" not in wake.previous_wake_text(clock.root, "authored", now=T0, last_authored="authored")
    assert reads == ["authored"], "an authored newest wake needs no second read"


def test_a_gap_in_the_last_note_is_named_and_never_hides_the_newest_outcome(clock, monkeypatch):
    from ouroboros import task_store

    seed_wake(clock.root, task_id="host1", status="failed", result=HOST_WORDS, terminal_origin="host_notice",
              wake_summary=None, outcome_axes={}, terminal_time=None)
    record = wake.previous_wake_text(clock.root, "host1", now=T0, last_authored="host1")  # the clock's note is this row
    assert record.startswith(f"host1, Failed at an unknown time; its line: no line; the host's notice, not a note of mine: «{HOST_WORDS}»")
    assert wake.previous_wake_text(clock.root, "host1", now=T0) == record + ". No earlier note of mine is known"
    assert wake.previous_wake_text(clock.root, "host1", now=T0, last_authored="gone") == (
        record + ". My last note before it, gone, is unreadable")
    seed_wake(clock.root, task_id="later-host", status="failed", result="host", terminal_origin="host_notice", wake_summary=None)
    rewritten = wake.previous_wake_text(clock.root, "host1", now=T0, last_authored="later-host")
    assert rewritten.endswith("My last note before it, later-host, no longer reads as a note of mine; "
                              "get_task_result(task_id='later-host') reads it"), "a readable row is never called unreadable"
    assert "«host»" not in rewritten, "a row that no longer reads as authored is not offered as a note"
    original = task_store.read_document

    def broken(root, task_id):
        if task_id == "other":
            raise ValueError("note unavailable")
        return original(root, task_id)

    monkeypatch.setattr(task_store, "read_document", broken)
    assert wake.previous_wake_text(clock.root, "host1", now=T0, last_authored="other") == (
        record + ". My last note before it, other, is unreadable")


def test_the_real_first_request_of_the_next_wake_carries_the_note_and_keeps_room_readers(clock, monkeypatch):
    """What the builder delivers, not a helper's output: the assignment holds the continuation, the system text the room."""
    from ouroboros import dialogue_store as ds
    from ouroboros.agent import Env
    from ouroboros.context import build_llm_messages
    from ouroboros.memory import Memory
    from ouroboros.projects_registry import create_project

    repo = clock.root / "repo"
    (repo / "docs").mkdir(exist_ok=True)
    for name, body in (("prompts/SYSTEM.md", "You are Ouroboros."), ("BIBLE.md", "# Principle 0: Agency"),
                       ("docs/ARCHITECTURE.md", "# Ouroboros v1.2.3 — Architecture"), ("docs/DEVELOPMENT.md", "# DEVELOPMENT.md"),
                       ("README.md", "version-1.2.3"), ("docs/CHECKLISTS.md", "## Change Review Checklist")):
        (repo / name).write_text(body, encoding="utf-8")
    (clock.root / "state").mkdir(exist_ok=True)
    room = create_project(clock.root, "alpha", name="Alpha")
    ds.append(clock.root, "chat", {"chat_id": int(room["chat_id"]), "direction": "in", "ts": "2026-10-01T00:00:00+00:00",
                                   "client_message_id": "alpha1", "text": "The owner's whole words in another room."})
    previous = _chain(clock, monkeypatch, UNATTRIBUTED, HOST)
    env, memory = Env(repo_dir=repo, drive_root=clock.root), Memory(drive_root=clock.root, repo_dir=repo)
    messages, cap = build_llm_messages(env=env, memory=memory, task=clock.launches[-1]["task"])
    system = "\n".join(block["text"] for block in messages[0]["content"])
    assignment = next(message for message in messages if message["role"] == "user")["content"]
    assignment = assignment if isinstance(assignment, str) else "\n".join(part.get("text", "") for part in assignment)
    assert cap["memory_view"]["role"] == "consciousness"
    assert f"- previous wake: {previous}\n" in assignment and HOST_WORDS in assignment and "A's whole note." in assignment
    assert "words of nobody recorded" in assignment and "wake0002 may have changed it" in assignment
    others = system.split("## Other conversations", 1)[1]
    assert "Project Alpha" in others and f"memory_read(room_id='{room['chat_id']}'" in others
    assert "The owner's whole words in another room." not in assignment, "older than the observed window; the room reader retains it"
