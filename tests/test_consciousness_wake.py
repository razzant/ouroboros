"""The wake-up message, its physical-window observation and envelope (Background Consciousness),
and the server's projection of the alarm clock's snapshot into one honest status line."""

from __future__ import annotations

import hashlib
import json
import pathlib
from types import SimpleNamespace

import pytest

from ouroboros import consciousness_wake as wake, dialogue_store as ds
from tests._dialogue import import_all
from tests._store_shared import read_result, result_exists, seed_result
from tests._wake_summary_shared import seed_wake
from tests import test_consciousness as clock_tests

REPO = pathlib.Path(__file__).resolve().parents[1]
T0 = 1_800_000_000.0
clock = clock_tests.clock


@pytest.mark.parametrize("summary", ["Проверил состояние системы.", "Проверил состояние системы. Нашёл 42 строки. Дальше шаг владельца.", ""])
@pytest.mark.parametrize("callback", ["own_clock", "focus", "other-task"])
@pytest.mark.parametrize("title", ["", "Authored title. Keep its detail."])
def test_finished_wake_publishes_its_own_title_once(clock, monkeypatch, summary, callback, title):
    from supervisor import worker_chat_lane

    row = seed_wake(clock.root, title=title, **({"wake_summary": summary, "suggested_name": title or summary} if summary else {}))
    task_id = row["task_id"]
    clock.clock._own_wake = {"task_id": task_id, "wake_cause": "own_clock", "initiator": "consciousness"}
    frames = []
    monkeypatch.setattr(worker_chat_lane, "_broadcast_task_named", frames.append)
    for _ in range(2):
        clock.clock._wake_finished("other" if callback == "other-task" else task_id, True,
            metadata={"wake_cause": callback, "initiator": "consciousness"})
    assert frames == ([{"type": "task_named", "task_id": task_id, "suggested_name": "Проверил состояние системы."}]
                      if summary and callback == "own_clock" and not title else [])


@pytest.mark.parametrize("summary", ["Проверил состояние системы.", "Проверил состояние системы.\n\nНашёл 42 строки. Дальше шаг владельца.", ""])
def test_previous_wake_reads_one_document_and_keeps_the_whole_note(tmp_path, monkeypatch, summary):
    from ouroboros import task_store

    row = seed_wake(tmp_path, **({"wake_summary": summary} if summary else {}))
    row["result"] += "\n\n" + "Своя мысль с {daily_usd}. " * 3000 + "\nLast whole line."
    seed_result(tmp_path, row["task_id"], row)
    reads, original = [], task_store.read_document

    def read(root, task_id):
        reads.append(task_id)
        return original(root, task_id)

    monkeypatch.setattr(task_store, "read_document", read)
    now = wake._parse_iso(row["terminal_time"]["occurred_at"]) + 3600
    previous = wake.previous_wake_text(tmp_path, row["task_id"], now=now)
    assert reads == [row["task_id"]]
    assert previous == (f"{row['task_id']}, Done 1 h 0 min ago; its line: "
        + (f"«{summary}»" if summary else "no line") + f"; its note: «{row['result']}»")
    text = wake.render_wake_message(REPO, drive_root=tmp_path, reason="heartbeat", last_wake_at=now - 3600, now=now,
        level="act", disabled_tools=[], spent_usd=0, daily_usd=20, running=0, max_tasks=2,
        interval=3600, events="nothing new", previous_wake=previous)
    assert f"- previous wake: {previous}\n" in text
    assert row["result"] in text and "{daily_usd}" in text, "authored placeholders are not rendered again"


@pytest.mark.parametrize("status,word", [("completed", "Done"), ("failed", "Failed"),
                                        ("cancelled", "Stopped")])
def test_previous_wake_uses_card_outcome_words_and_never_guesses_completion_time(tmp_path, status, word):
    row = seed_wake(tmp_path, status=status, outcome_axes={}, terminal_time=None)
    previous = wake.previous_wake_text(tmp_path, row["task_id"], now=T0)
    assert previous.startswith(f"{row['task_id']}, {word} at an unknown time; its line: no line;")
    assert previous.endswith(f"its note: «{row['result']}»")


def test_previous_wake_distinguishes_absence_and_unreadability(tmp_path, monkeypatch):
    from ouroboros import task_store
    from ouroboros.store import StoreError

    assert wake.previous_wake_text(tmp_path, "missing", now=T0) == (
        "missing, unreadable; get_task_result(task_id='missing') reads it")
    reads = []

    def broken(root, task_id):
        reads.append(task_id)
        raise StoreError("unreadable task")

    monkeypatch.setattr(task_store, "read_document", broken)
    assert wake.previous_wake_text(tmp_path, "", now=T0) == "none"
    assert reads == []
    assert wake.previous_wake_text(tmp_path, "previous", now=T0) == (
        "previous, unreadable; get_task_result(task_id='previous') reads it")
    assert reads == ["previous"]


@pytest.mark.parametrize("failure", ["read", "projection", "none"])
def test_previous_wake_failure_does_not_block_launch(clock, monkeypatch, caplog, failure):
    from ouroboros import task_store, terminal_time

    row = seed_wake(clock.root, result="The whole previous note.\nLast line.")
    clock.clock._last_wake_task_id = row["task_id"]
    clock.root.joinpath("repo/prompts/CONSCIOUSNESS.md").write_text(
        "- previous wake: {previous_wake}\n", encoding="utf-8")

    def broken(*args, **kwargs):
        raise ValueError("previous wake unavailable")

    if failure == "read":
        monkeypatch.setattr(task_store, "read_document", broken)
    elif failure == "projection":
        monkeypatch.setattr(terminal_time, "terminal_time_fact", broken)
    with caplog.at_level("DEBUG", logger="ouroboros.consciousness_wake"):
        assert clock.clock.tick(clock.clock.next_wake_at) == "launched"
    text = clock.launches[-1]["text"]
    if failure == "none":
        assert row["result"] in text and "unreadable" not in text
    else:
        task_id = row["task_id"]
        assert f"- previous wake: {task_id}, unreadable; get_task_result(task_id='{task_id}') reads it\n" in text
        assert any(row["task_id"] in record.message and record.exc_info for record in caplog.records)


@pytest.mark.parametrize(("failure", "restart"), [("write", False), ("readback", True)])
def test_next_wake_reads_last_admitted_note_even_when_its_input_source_failed(clock, monkeypatch, failure, restart):
    from ouroboros import artifacts, chat_chain
    from ouroboros.consciousness import BackgroundConsciousness, OBSERVATION_STATE_KEY, OWN_WAKE_STATE_KEY

    template = REPO.joinpath("prompts/CONSCIOUSNESS.md").read_text(encoding="utf-8")
    clock.root.joinpath("repo/prompts/CONSCIOUSNESS.md").write_text(template, encoding="utf-8")
    assert clock.clock.tick(T0 + clock_tests.FLOOR) == "launched"
    # A's carrier accepted the boundary; it rides A's row to the clock at A's terminal.
    launched_a = clock.launches[-1]["task"]["metadata"]
    accepted_a = launched_a["wake_observation"]["accepted_boundary"]
    assert OBSERVATION_STATE_KEY not in clock.store
    seed_wake(clock.root, task_id="wake0001", result="A's whole note.", wake_summary="A's line.",
              metadata=launched_a)
    clock.finish("wake0001", True)  # the carrier's terminal settles the wake and lays A's boundary
    assert clock.store[OBSERVATION_STATE_KEY] == accepted_a
    clock.receipt["task_id"] = "wake0002"
    clock.clock._next_wake_at = T0 + 2 * clock_tests.FLOOR
    with monkeypatch.context() as fault:
        if failure == "write":
            def failed_write(*a, **kw):
                raise OSError("input source write failed")
            fault.setattr(chat_chain, "retain_memory_source", failed_write)
        else:
            read = artifacts.read_actor_source_bytes
            fault.setattr(artifacts, "read_actor_source_bytes", lambda root, task, ref:
                          read(root, task, ref) if task == "wake0001" else b"bad readback")
        assert clock.clock.tick(T0 + 2 * clock_tests.FLOOR) == "launched"
    assert clock.store[OBSERVATION_STATE_KEY] == accepted_a
    assert clock.store[OWN_WAKE_STATE_KEY]["task_id"] == "wake0002"
    seed_wake(clock.root, task_id="wake0002", status="interrupted", outcome_axes={}, terminal_time=None,
              result="B's whole note.\n\nStill unfinished.", wake_summary="B's line.")
    seed_wake(clock.root, task_id="newer-foreign", result="Never use the newest task's note.")
    clock.finish("wake0002", True)
    if restart:
        clock.clock = BackgroundConsciousness(clock.root, clock.root / "repo", lambda: 7,
                                              now=T0 + 2 * clock_tests.FLOOR)
        assert clock.clock._last_wake_task_id == "wake0002"
    clock.receipt["task_id"] = "wake0003"
    now = T0 + 3 * clock_tests.FLOOR
    clock.clock._next_wake_at = now
    expected = wake.previous_wake_text(clock.root, "wake0002", now=now)
    assert "Interrupted" in expected and "B's whole note.\n\nStill unfinished." in expected
    assert clock.clock.tick(now) == "launched"
    launch = clock.launches[-1]
    for rendered in (launch["text"], launch["task"]["metadata"]["wake_observation"]["projection_text"]):
        assert f"- previous wake: {expected}\n" in rendered
        assert "A's whole note." not in rendered and "Never use the newest task's note." not in rendered


@pytest.mark.parametrize("has_previous", [True, False])
def test_launch_captures_previous_wake_before_binding_the_new_id(clock, has_previous):
    row = seed_wake(clock.root, wake_summary="Проверил состояние системы.") if has_previous else None
    clock.clock._last_wake_task_id = row["task_id"] if row else ""
    template = REPO.joinpath("prompts/CONSCIOUSNESS.md").read_text(encoding="utf-8")
    clock.root.joinpath("repo/prompts/CONSCIOUSNESS.md").write_text(template, encoding="utf-8")
    now = clock_tests.T0 + clock_tests.FLOOR
    expected = wake.previous_wake_text(clock.root, clock.clock._last_wake_task_id, now=now)
    assert clock.clock.tick(now=now) == "launched"
    launch = clock.launches[-1]
    assert clock.clock._last_wake_task_id == clock.receipt["task_id"]
    assert f"- previous wake: {expected}\n" in launch["text"]
    assert f"- previous wake: {expected}\n" in launch["task"]["metadata"]["wake_observation"]["projection_text"]


_AUTHOR_STOP = {"lifecycle": {"status": "completed"}, "objective": {"status": "fail", "source": "task_completion",
                                                                     "reason": "author_stop"},
                "execution": {"status": "ok", "task_completion": {"action": "stop", "rationale": "It waits on an Allow."}}}


@pytest.mark.parametrize(("summary", "announced", "stopped"), [
    ("Проверил состояние системы: всё спокойно.", True, False), ("", False, False), ("", False, True)])
def test_only_direct_turns_with_a_line_or_an_open_continue_join_the_observation_text(tmp_path, announced, summary,
                                                                                      stopped):
    """A finished turn's exchange is the dialogue itself, so it is counted; an authored line, or an open Continue (a
    stop with work left, which the dialogue does not show), lists it."""
    row = seed_wake(tmp_path, **({"wake_summary": summary} if summary else {}),
                    **({"outcome_axes": _AUTHOR_STOP} if stopped else {}))
    now = wake._parse_iso(row["terminal_time"]["occurred_at"]) + 60
    if announced:
        _chat(tmp_path, _summary(row["task_id"], now - 30))
    observation = wake.observe_wake(tmp_path, boundary=None, since=now - 3600, now=now, reason="heartbeat")
    line = next(line for kind, _, line in observation.events if kind.startswith("direct_turn"))
    assert (line in observation.full_text()) is bool(summary or stopped)
    assert line.endswith(" · Continue open (author_stop)") is stopped
    if summary:
        assert line.endswith(": " + summary)
    else:
        assert row["result"] not in line
    assert row["result"] not in observation.full_text(), "the whole note has its separate previous-wake slot"


@pytest.mark.parametrize("summary", ["Checked the disk: 30 GiB free.", ""])
def test_a_scheduled_runs_sentence_joins_its_terminal_line_and_its_note_stays_out(tmp_path, summary):
    since = T0 - 3600
    seed_result(tmp_path, "sched1", {"task_id": "sched1", "status": "completed", "ts": _iso(since + 5),
        "updated_at": _iso(since + 5), "_schema_version": 1, "description": "Daily check", "metadata": {},
        "result": "Private note.", "wake_silent": True, **({"wake_summary": summary} if summary else {})})
    _chat(tmp_path, _summary("sched1", since + 10))
    observation = _observe(tmp_path)
    line = next(line for kind, _, line in observation.events if kind == "task_terminal" and "sched1" in line)
    assert line.endswith(": Daily check" + (f": {summary}" if summary else ""))
    assert "Private note." not in observation.full_text()


def _iso(ts):
    return wake._iso(ts)


def _result(root, task_id, *, status="completed", ts, cost=1.25, direct=False, quizzes=None, description="",
            project_id=""):
    (root / "task_results").mkdir(exist_ok=True)
    row = {"task_id": task_id, "status": status, "ts": _iso(ts), "updated_at": _iso(ts), "_schema_version": 1,
           "accounted_upper_bound_usd": cost, "description": description or f"do {task_id}",
           "metadata": {}, "_is_direct_chat": direct}
    if project_id:
        row["project_id"] = project_id
    if quizzes:
        row["owner_quiz"] = quizzes
    seed_result(root, task_id, row)


def test_wake_task_metadata_carries_origin_level_and_tree_cap(monkeypatch):
    monkeypatch.setenv("OUROBOROS_CONSCIOUSNESS_AUTONOMY", "act")
    meta = wake.wake_task_metadata("observe", "heartbeat", root_cost_ceiling_usd=3.5)
    assert meta["initiator"] == "consciousness" and meta["usage_category"] == "consciousness"
    assert meta["consciousness_autonomy"] == "observe" and meta["runtime_mode_cap"] == "light"
    assert meta["model_role"] == "consciousness" and meta["wake_reason"] == "heartbeat"
    # P3e: the cap the wake's own root scope binds, never the non-root member ceiling.
    assert "start_task" in meta["disabled_tools"] and meta["root_cost_ceiling_usd"] == 3.5
    assert "root_limit_usd" not in meta
    full = wake.wake_task_metadata("full", "event:x")
    assert full["disabled_tools"] == [] and full["runtime_mode_cap"] == "" and "root_cost_ceiling_usd" not in full
    # A non-positive cap is not stamped: it would read as "no narrowing" downstream and
    # the alarm never launches on an exhausted allowance anyway.
    assert "root_cost_ceiling_usd" not in wake.wake_task_metadata("act", "heartbeat", root_cost_ceiling_usd=0.0)
    assert wake.wake_task_metadata("bogus", "")["consciousness_autonomy"] == "act"  # falls back to the setting
    # Which limit set the ceiling is observed beside it, only beside a stamped ceiling and only in its own words.
    for source in ("allowance", "per_task_cap", "both"):
        stamped = wake.wake_task_metadata("act", "heartbeat", root_cost_ceiling_usd=2.99, root_cost_ceiling_source=source)
        assert stamped["root_cost_ceiling_source"] == source and stamped["root_cost_ceiling_usd"] == 2.99
    assert "root_cost_ceiling_source" not in meta  # no source given: none is guessed
    assert "root_cost_ceiling_source" not in wake.wake_task_metadata(
        "act", "heartbeat", root_cost_ceiling_usd=0.0, root_cost_ceiling_source="allowance")
    assert "root_cost_ceiling_source" not in wake.wake_task_metadata(
        "act", "heartbeat", root_cost_ceiling_usd=1.0, root_cost_ceiling_source="guess")


def _chat(root, *rows, archive: str = ""):
    """Append rows to the live chat log, or write them as one rotated archive segment."""
    path = root / ("archive" if archive else "logs") / (f"chat_{archive}.jsonl" if archive else "chat.jsonl")
    if ds.present(root):
        for row in rows:
            ds.append(root, "chat", row)
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    return path


def _owner(text, ts, *, chat_id=1, cmid="m"):
    return {"ts": _iso(ts), "direction": "in", "chat_id": chat_id, "user_id": 1, "text": text,
            "source": "web", "client_message_id": cmid, "ingress_accepted": True}


def _summary(task_id, ts, *, status="completed", **extra):
    return {"ts": _iso(ts), "direction": "system", "type": "task_summary", "task_id": task_id,
            "status": status, "chat_id": 1, "text": f"{task_id} {status}", **extra}


def _observe(root, *, boundary=None, since=T0 - 3600, reason="heartbeat"):
    return wake.observe_wake(root, boundary=boundary, since=since, now=T0, reason=reason)


def _lines(observation):
    """The listed events (own direct turns are counted, not listed)."""
    return [line for kind, _offset, line in observation.events if kind != "direct_turn"]


def test_every_event_of_the_window_is_observed_without_a_count_cut(tmp_path):
    """No ten-line window, no four-card cap, no 512 KB human tail: all of it, in append order."""
    since = T0 - 3600
    for index in range(15):
        _result(tmp_path, f"t{index:02d}", ts=since + index, description=f"work {index}")
    _result(tmp_path, "asker", ts=since - 500, status="running", quizzes={
        f"c{i}": {"state": "open", "asked_at": _iso(since - 1000 + i), "question": f"q{i}?"} for i in range(7)})
    filler = [{"ts": _iso(since + 1), "direction": "out", "chat_id": 1, "text": "x" * 4000} for _ in range(160)]
    _chat(tmp_path, *filler, _owner("first human line", since + 2, cmid="h1"),
          *[_summary(f"t{index:02d}", since + 10 + index) for index in range(15)],
          *filler, _owner("second human line", since + 40, cmid="h2"))
    assert (tmp_path / "logs" / "chat.jsonl").stat().st_size > 1_000_000
    observation = _observe(tmp_path)
    kinds = [kind for kind, _offset, _line in observation.events]
    assert kinds == ["owner_message"] + ["task_terminal"] * 15 + ["owner_message"]
    lines = _lines(observation)
    assert lines[0].startswith('- owner message in Main, ') and lines[0].endswith('] first human line')
    assert [line.split()[2] for line in lines[1:16]] == [f"t{index:02d}" for index in range(15)]
    assert "- task t14 Done, $1.25" in lines[15] and "work 14" in lines[15]
    assert len(observation.outstanding) == 7  # every answerable card, newest first
    assert [line.split()[3] for line in observation.outstanding] == [f"c{i}" for i in reversed(range(7))]
    text = observation.full_text()
    assert ("2 owner messages (counted, not listed: my view holds the words), 15 task terminals, "
            "7 outstanding cards") in text
    assert "first human line" not in text and "- task t14 Done, $1.25" in text  # my human's words are in my view
    assert "(+" not in text and "more; see" not in text


def test_the_boundary_is_the_accepted_position_not_the_alarm_finish(tmp_path):
    """Rows appended during a wake — even with an older stamp — belong to the next wake."""
    since = T0 - 3600
    _chat(tmp_path, _owner("before the first wake", since + 10, cmid="a"))
    first = _observe(tmp_path)
    assert _lines(first) and first.window["basis"] == "time_bootstrap"
    accepted = first.boundary
    # While that wake runs: an owner line, a child's terminal written late with an OLD stamp.
    _chat(tmp_path, _owner("during the wake", T0 - 5, cmid="b"), _summary("late", since - 900))
    second = _observe(tmp_path, boundary=accepted, since=T0)  # the alarm's finish time is irrelevant now
    assert second.window["basis"] == "accepted_boundary" and second.window["lower"] == accepted["upper"]
    lines = _lines(second)
    assert len(lines) == 2 and "during the wake" in lines[0] and "task late completed" in lines[1]
    assert "before the first wake" not in second.full_text()
    # Nothing new since: an empty window, and the same accepted position again.
    third = _observe(tmp_path, boundary=second.boundary, since=T0)
    assert third.events == () and third.full_text() == "\n- wake cause: the scheduled wake time; events never move it, they are listed as observed"
    assert third.boundary["upper"] == second.boundary["upper"]


def test_the_boundary_survives_rotation_and_a_replaced_chain_is_disclosed(tmp_path):
    since = T0 - 3600
    live = _chat(tmp_path, _owner("old line", since + 5, cmid="a"))
    accepted = _observe(tmp_path).boundary
    # Rotation renames the live generation into the archive, byte for byte.
    (tmp_path / "archive").mkdir(exist_ok=True)
    live.rename(tmp_path / "archive" / "chat_20270115T100000.jsonl")
    _chat(tmp_path, _owner("after rotation", T0 - 20, cmid="b"))
    rotated = _observe(tmp_path, boundary=accepted)
    assert rotated.window["basis"] == "accepted_boundary"
    assert [line.rsplit("] ", 1)[-1] for line in _lines(rotated)] == ["after rotation"]
    # A chain whose segment at the boundary no longer starts with the same line is not trusted.
    (tmp_path / "archive" / "chat_20270115T100000.jsonl").write_text(
        json.dumps(_owner("rewritten history", since + 5, cmid="z")) + "\n", encoding="utf-8")
    replaced = _observe(tmp_path, boundary=accepted, since=T0 - 60)
    assert replaced.window["basis"] == "accepted_boundary"
    assert "accepted_boundary_no_longer_matches_chat_chain" not in replaced.gaps
    assert [line.rsplit("] ", 1)[-1] for line in _lines(replaced)] == ["after rotation"]


# --- canonical transitions: task results by identity, whether or not a chat row announced them ---


def _write(root, task_id, **fields):
    """A task result row exactly as stored (``ts`` is the first write; nothing else is inferred)."""
    (root / "task_results").mkdir(exist_ok=True)
    row = read_result(root, task_id) if result_exists(root, task_id) else {
        "task_id": task_id, "_schema_version": 1, "metadata": {}, "description": f"do {task_id}"}
    row.update(fields)
    seed_result(root, task_id, row)
    return row


def _wake(root, boundary, now):
    return wake.observe_wake(root, boundary=boundary, since=T0 - 3600, now=now, reason="heartbeat")


def test_terminals_no_chat_row_announced_reach_the_next_wake_once(tmp_path):
    """An orphan-swept child and a root whose task_done was lost write no chat row: the task
    results record them, so the wake reports them by identity — once — and an old task whose
    ``updated_at`` merely moved is never re-reported as a completion."""
    _write(tmp_path, "root1", status="running", ts=_iso(T0 - 900), updated_at=_iso(T0 - 900))
    _write(tmp_path, "kid1", status="running", ts=_iso(T0 - 800), updated_at=_iso(T0 - 800),
           parent_task_id="root1", delegation_role="subagent", description="child work")
    _write(tmp_path, "old", status="completed", ts=_iso(T0 - 86400), updated_at=_iso(T0 - 86000))
    first = _wake(tmp_path, None, T0)
    assert first.window["transitions_basis"] == "time_bootstrap"
    state = first.boundary["transitions"]
    assert set(state["inventory"]) == {"root1", "kid1", "old"}  # closed tasks can receive new review facts
    # While/after that wake: the sweep settles the child, the root settles with no task_done
    # row, and enrichment bumps the old task's updated_at. No chat row anywhere.
    _write(tmp_path, "kid1", status="failed", updated_at=_iso(T0 + 100), reason_code="orphaned_running_after_worker_restart")
    _write(tmp_path, "root1", status="completed", updated_at=_iso(T0 + 200))
    _write(tmp_path, "old", updated_at=_iso(T0 + 300), accounted_upper_bound_usd=2.0)
    second = _wake(tmp_path, first.boundary, T0 + 600)
    assert second.window["transitions_basis"] == "accepted_inventory"
    assert [(kind, offset) for kind, offset, _line in second.events] == [("task_terminal", None)] * 2
    lines = _lines(second)
    assert lines[0].startswith("- child task kid1 of root1 Failed, completion time not recorded (last updated 8 min ago)")
    assert lines[1].startswith("- task root1 Done, completion time not recorded (last updated 6 min ago)")
    assert "Recorded in task results with no chat row in this window" in second.full_text()
    assert not any("old" in line.split()[2] for line in lines)
    assert set(second.boundary["transitions"]["inventory"]) == {"root1", "kid1", "old"}
    # Accepted: the next wake finds nothing new, however often the rows are rewritten.
    _write(tmp_path, "root1", updated_at=_iso(T0 + 700))
    assert _wake(tmp_path, second.boundary, T0 + 1200).events == ()


def test_a_task_first_recorded_after_the_scan_is_new_and_a_proven_terminal_keeps_its_own_stamp(tmp_path):
    _write(tmp_path, "seed", status="running", ts=_iso(T0 - 10), updated_at=_iso(T0 - 10))
    first = _wake(tmp_path, None, T0)
    second = _wake(tmp_path, first.boundary, T0 + 600)  # prior_scan_at = T0 from here on
    assert second.events == ()
    # Created, run and settled between two wakes; its first write is in flight during the scan
    # (stamped before it, visible after): the one-scan overlap still finds it.
    _write(tmp_path, "brief", status="completed", ts=_iso(T0 + 590), updated_at=_iso(T0 + 900),
           canonical_terminal_projection_ready={"task_done_ts": _iso(T0 + 880)})
    third = _wake(tmp_path, second.boundary, T0 + 1200)
    assert _lines(third) == ["- task brief Done, 5 min ago: do brief"]  # its task_done stamp, proven
    assert "terminal:brief:" in " ".join(third.boundary["transitions"]["inventory"]["brief"])
    # Reported once: the carried key keeps the overlap from reporting it again.
    assert _wake(tmp_path, third.boundary, T0 + 1800).events == ()


def test_an_expired_card_is_a_transition_and_stays_answerable_in_the_inventory(tmp_path):
    _write(tmp_path, "asker", status="running", ts=_iso(T0 - 900), updated_at=_iso(T0 - 900), owner_quiz={
        "q1": {"quiz_id": "q1", "state": "open", "asked_at": _iso(T0 - 800), "question": "Ship it?"}})
    first = _wake(tmp_path, None, T0)
    assert first.boundary["transitions"]["inventory"] == {"asker": []}
    from ouroboros.owner_quiz import reconcile_terminal

    _write(tmp_path, "asker", status="failed", updated_at=_iso(T0 + 60))
    assert reconcile_terminal(tmp_path, "asker") == ["q1"]  # the sweep's own card leg, no chat row
    second = _wake(tmp_path, first.boundary, T0 + 600)
    kinds = [kind for kind, _offset, _line in second.events]
    assert sorted(kinds) == ["card_state", "task_terminal"]
    card = next(line for kind, _o, line in second.events if kind == "card_state")
    assert card.startswith("- owner card q1 on task asker: Unanswered · task finished")
    assert "Ship it?" in card
    assert len(second.outstanding) == 1 and "q1" in second.outstanding[0]  # still answerable
    assert "1 card closed unanswered" in second.full_text()
    # The owner answers late: that is the next transition, the expiry is not repeated.
    row = read_result(tmp_path, "asker")
    row["owner_quiz"]["q1"].update(state="answered", answered_at=_iso(T0 + 700), comment="yes")
    seed_result(tmp_path, "asker", row)
    third = _wake(tmp_path, second.boundary, T0 + 1200)
    assert [kind for kind, _o, _l in third.events] == ["card_answer"]
    assert "answered in own words: yes" in _lines(third)[0] and third.outstanding == ()


def test_a_transition_takes_the_position_of_the_row_that_announced_it_once(tmp_path):
    _write(tmp_path, "kid", status="running", ts=_iso(T0 - 900), updated_at=_iso(T0 - 900))
    first = _wake(tmp_path, None, T0)
    _write(tmp_path, "kid", status="completed", updated_at=_iso(T0 + 90))
    # A pre-fold acceptance_late_settlement row stays ordinary chat history, never a wake event.
    late_row = {"ts": _iso(T0 + 61), "direction": "system", "type": "acceptance_late_settlement", "task_id": "rv",
                "text": "On the reviewed version…", "card_row": "reviews", "card_row_id": "acceptance-late:rk-9"}
    summary = _summary("kid", T0 + 95)
    _chat(tmp_path, _owner("hello", T0 + 30, cmid="h"), late_row, summary, dict(summary))
    second = _wake(tmp_path, first.boundary, T0 + 600)
    assert [kind for kind, _o, _l in second.events] == ["owner_message", "task_terminal"]
    assert all(offset is not None for _k, offset, _l in second.events)  # each at its announcing row, once
    assert "no chat row in this window" not in second.full_text()
    assert "- task kid Done" in _lines(second)[1]
    assert "On the reviewed version" not in second.full_text()


def test_a_result_first_written_late_under_an_old_stamp_is_found_through_its_row(tmp_path):
    """The compatibility terminal persist may create a row after the scan with the task's
    original (older) ``ts``: its first observed identity counts, at its announcing row."""
    _write(tmp_path, "seed", status="running", ts=_iso(T0 - 10), updated_at=_iso(T0 - 10))
    second = _wake(tmp_path, _wake(tmp_path, None, T0).boundary, T0 + 600)
    _write(tmp_path, "revived", status="completed", ts=_iso(T0 - 86400), updated_at=_iso(T0 + 700))
    _chat(tmp_path, _summary("revived", T0 + 700))
    third = _wake(tmp_path, second.boundary, T0 + 1200)
    assert [kind for kind, offset, _l in third.events if offset is not None] == ["task_terminal"]
    assert "- task revived Done" in _lines(third)[0]


def test_an_unreadable_task_store_keeps_the_accepted_inventory(tmp_path, monkeypatch):
    _write(tmp_path, "kid", status="running", ts=_iso(T0 - 900), updated_at=_iso(T0 - 900))
    first = _wake(tmp_path, None, T0)
    real = wake._task_window
    monkeypatch.setattr(wake, "_task_window", lambda *_args, **_kw: (_ for _ in ()).throw(OSError("busy")))
    _write(tmp_path, "kid", status="completed", updated_at=_iso(T0 + 60))
    blind = _wake(tmp_path, first.boundary, T0 + 600)
    assert blind.boundary["transitions"] == first.boundary["transitions"]  # nothing consumed
    assert "task_results unreadable: OSError" in blind.gaps
    monkeypatch.setattr(wake, "_task_window", real)
    assert "- task kid Done" in _lines(_wake(tmp_path, blind.boundary, T0 + 1200))[0]


def test_unfinished_and_malformed_lines_are_gaps_not_silence(tmp_path):
    since = T0 - 3600
    path = _chat(tmp_path, _owner("complete", since + 1, cmid="a"))
    with path.open("a", encoding="utf-8") as handle:
        handle.write("{not json}\n")
        handle.write(json.dumps(_owner("unfinished", since + 2, cmid="b"))[:-1])
    first = _observe(tmp_path)
    assert [line.rsplit("] ", 1)[-1] for line in _lines(first)] == ["complete"]
    assert any("jsonl_malformed" in gap for gap in first.gaps)
    ds.append(tmp_path, "chat", _owner("unfinished", since + 2, cmid="b"))
    second = _observe(tmp_path, boundary=first.boundary)
    assert [line.rsplit("] ", 1)[-1] for line in _lines(second)] == ["unfinished"]


def test_invalid_only_import_has_an_explicit_gap_and_later_sql_row_keeps_its_old_stamp(tmp_path):
    path = _chat(tmp_path)
    row = _owner("old timestamp, newly complete", T0 - 100, cmid="partial")
    path.write_text(json.dumps(row)[:-1], encoding="utf-8")
    first = wake.observe_wake(tmp_path, boundary=None, since=T0 - 500, now=T0)
    assert not first.events and first.gaps and first.boundary["upper"] > 0
    ds.append(tmp_path, "chat", row)
    second = wake.observe_wake(tmp_path, boundary=first.boundary, since=T0 + 10, now=T0 + 100)
    assert len(second.events) == 1 and second.events[0][2].endswith("newly complete")  # recorded whole, counted
    assert "1 owner message (counted, not listed" in second.full_text()
    assert second.window["basis"] == "accepted_boundary" and not second.gaps
    assert wake.observe_wake(tmp_path, boundary=second.boundary, since=T0 + 110, now=T0 + 200).events == ()


def test_pending_import_does_not_advance_an_accepted_wake_boundary(tmp_path):
    from tests._dialogue import seed

    seed(tmp_path, archives={"chat_20261016T000000.jsonl": [_owner("pending", T0 - 100)]})
    first = _observe(tmp_path)
    assert any("still importing" in gap for gap in first.gaps)
    assert "upper" not in first.boundary
    import_all(tmp_path)
    assert any(line.endswith("] pending") for _kind, _offset, line in _observe(tmp_path, boundary=first.boundary).events)


def test_human_input_is_classified_by_producer_provenance(tmp_path):
    since = T0 - 3600
    presence_chat = 2 ** 41
    block = {"quiz_id": "q9", "state": "answered", "asked_at": _iso(since), "answered_at": _iso(since + 30),
             "options": ["Keep", "Drop"], "answered_index": 0, "question": "Keep the chain?"}
    _chat(tmp_path,
          _owner("from the owner", since + 1, cmid="a"),
          {**_owner("from a correspondent", since + 2), "source": "presence:telegram", "chat_id": presence_chat,
           "transport": {"actor": {"kind": "user"}}},
          {**_owner("I opened this cycle", since + 3), "source": "presence:telegram", "chat_id": presence_chat,
           "transport": {"actor": {"kind": "proactive_initiation"}}},
          {**_owner("machine traffic", since + 4), "chat_id": -5, "source": "skill:a2a"},
          {"ts": _iso(since + 30), "direction": "system", "type": "quiz_answer", "task_id": "asker", "quiz": block},
          {**_owner("Keep", since + 30), "client_message_id": "quiz_late_answer:asker:q9"},
          {"ts": _iso(since + 40), "direction": "out", "chat_id": 1, "text": "my reply"},
          # A pre-fold late acceptance row exactly as its retired producer wrote it (``log_chat``).
          {"ts": _iso(since + 50), "direction": "system", "type": "acceptance_late_settlement", "task_id": "t1",
           "text": "Late review of an earlier revision: 2 findings", "card_row": "reviews",
           "card_row_id": "acceptance-late:rk-1"},
          {"ts": _iso(since + 60), "direction": "system", "type": "task_error", "task_id": "w1",
           "text": "⚠️ Error: RuntimeError: boom", "task_terminal_status": "failed"})
    observation = _observe(tmp_path)
    assert [kind for kind, _o, _l in observation.events] == [
        "owner_message", "correspondent_message", "card_answer", "task_error"]
    lines = _lines(observation)
    assert lines[1].startswith("- message in a place in ") and lines[1].endswith("] from a correspondent")
    assert lines[2].startswith("- owner card q9 on task asker: answered") and "chose option 1: Keep" in lines[2]
    assert "task w1 runner failed" in lines[3]
    text = observation.full_text()
    assert "I opened this cycle" not in text and "machine traffic" not in text and "my reply" not in text
    # A pre-fold acceptance_late_settlement row is ordinary history of its chat, never a wake event.
    assert "Late review of an earlier revision" not in text


@pytest.mark.parametrize("status, origin", [("failed", "host_notice"), ("cancelled", ""),
                                           ("completed", "host_notice")])
def test_failed_inline_presence_is_visible_on_regular_wake_without_reviving_owner_turns(tmp_path, status, origin):
    from ouroboros.presence_runner import _build_task
    from ouroboros.task_results import write_task_result
    from tests.test_presence_runner import _admission, _event

    task = _build_task(_admission(), _event(), drive_root=tmp_path, staged_files=())
    assert task["_is_direct_chat"] is True
    metadata = {**task["metadata"], "presence_outcome": "deferred", "presence_result_text": "",
                "presence_work_ref": "still-running-child"}
    write_task_result(tmp_path, task["id"], status, _is_direct_chat=True, metadata=metadata,
                      terminal_origin=origin, result="Host diagnostic remains available in the task.")
    _result(tmp_path, "owner-turn", ts=T0, direct=True, status="failed")
    _chat(tmp_path, _summary(task["id"], T0 - 60, status=status), _summary("owner-turn", T0 - 30, status="failed"))
    observation = _observe(tmp_path, since=0)
    fact = next(line for line in _lines(observation) if task["id"] in line)
    assert fact.startswith(f"- Presence turn {task['id']} {status}") and "Presence outcome=deferred" in fact
    assert "get_task_result" in fact
    # The owner's own direct turn is counted, not listed: its exchange is the dialogue itself.
    assert not any("owner-turn" in line for line in _lines(observation))
    assert "1 own direct turn ended (counted, not listed" in observation.full_text()
    metadata["initiator"] = "consciousness"  # a wake's own turn is a direct turn, counted not listed
    write_task_result(tmp_path, task["id"], status, _is_direct_chat=True, metadata=metadata, terminal_origin=origin)
    assert not any(task["id"] in line for line in _lines(_observe(tmp_path, since=0)))


def test_answered_cards_of_the_window_render_their_whole_answer(tmp_path):
    since = T0 - 3600
    long_comment = "Keep the old parser.\n\nReason: " + "the migration cost is too high; " * 120 + "END-OF-COMMENT"
    block = {"quiz_id": "qown", "state": "answered", "asked_at": _iso(since + 60), "answered_at": _iso(since + 2400),
             "options": ["A", "B"], "comment": long_comment, "question": "Which parser?"}
    _chat(tmp_path, {"ts": _iso(since + 2400), "direction": "system", "type": "quiz_answer",
                     "task_id": "asker", "quiz": block})
    own = _lines(_observe(tmp_path))[0]
    assert own.startswith("- owner card qown on task asker: answered 20 min ago; asked 59 min ago; "
                          "answered in own words: Keep the old parser.")
    assert long_comment in own and "END-OF-COMMENT; question: Which parser?" in own  # whole, never clipped
    assert not any(word in own for word in ("understood", "confused", "misunderstood"))


def test_answered_card_line_renders_mixed_answers_and_missing_facts_honestly():
    now = T0
    both = wake._answered_card_line("t1", "q1", {
        "answered_at": _iso(now - 120), "asked_at": _iso(now - 240), "options": ["Go", "Stop"],
        "answered_index": 0, "comment": "but only on weekdays", "question": "Deploy?"}, now=now)
    assert both == ("- owner card q1 on task t1: answered 2 min ago; asked 4 min ago; "
                    "chose option 1: Go; with the words: but only on weekdays; question: Deploy?")
    bare = wake._answered_card_line("t1", "q2", {"answered_at": "not a stamp", "answered_index": 5}, now=now)
    assert bare == ("- owner card q2 on task t1: answered at an unknown time; asked at an unknown time; "
                    "chose option 6: label unavailable; question: question text unavailable")
    empty = wake._answered_card_line("t1", "q3", {"answered_at": _iso(now - 60)}, now=now)
    assert "answer text unavailable" in empty

def test_a_project_task_is_an_observed_event_not_a_wake_cause(tmp_path):
    """#1603: a project digest no longer names itself as the wake's cause; the settled
    project task reaches the scheduled wake as an observed transition, the cause line is
    the schedule, and the outstanding cards still follow the events."""
    since = T0 - 3600
    (tmp_path / "state").mkdir()
    (tmp_path / "state" / "projects.json").write_text(json.dumps({
        "projects": [{"id": "p1", "name": "System audit", "chat_id": 42, "lifecycle": "active"}],
    }), encoding="utf-8")
    _result(tmp_path, "finished", ts=since + 10, project_id="p1", description="audit completed")
    _result(tmp_path, "card-task", ts=since - 100, status="running",
            quizzes={"q1": {"state": "expired_terminal", "asked_at": _iso(since - 50),
                              "question": "Should the audit continue?"}})
    observation = _observe(tmp_path, since=since)
    assert observation.trigger == "- wake cause: the scheduled wake time; events never move it, they are listed as observed"
    assert any("finished" in line and "audit completed" in line for line in _lines(observation))
    card = observation.outstanding[0]
    assert card.startswith("- owner card q1 on task card-task: Unanswered · task finished")
    text = observation.full_text()
    assert text.index(observation.trigger) < text.index("Outstanding owner cards (1):") < text.index(card)


def test_trigger_stays_first_when_task_results_are_unreadable(tmp_path, monkeypatch):
    def broken(*_args, **_kw):
        raise OSError("broken task store")

    monkeypatch.setattr(wake, "_task_window", broken)
    observation = _observe(tmp_path, since=T0 - 1)
    assert observation.trigger == "- wake cause: the scheduled wake time; events never move it, they are listed as observed"
    assert observation.gaps == ("task_results unreadable: OSError",)
    assert observation.full_text().split("\n")[1:] == [
        "- wake cause: the scheduled wake time; events never move it, they are listed as observed",
        "Observation coverage: chat log bytes 0–0 (empty chain); task results: unreadable, the accepted "
        "inventory is kept for the next wake; gaps: task_results unreadable: OSError"]


def test_render_substitutes_every_placeholder_and_keeps_every_event(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    (repo / "prompts").mkdir(parents=True)
    (repo / "prompts" / "CONSCIOUSNESS.md").write_text(
        (REPO / "prompts" / "CONSCIOUSNESS.md").read_text(encoding="utf-8"), encoding="utf-8")
    for index in range(15):
        _result(tmp_path, f"t{index:02d}", ts=T0 - 100 + index)
    _chat(tmp_path, *[_summary(f"t{index:02d}", T0 - 100 + index) for index in range(15)])
    observation = _observe(tmp_path, since=T0 - 5400)
    text = wake.render_wake_message(
        repo, drive_root=tmp_path, reason="heartbeat", last_wake_at=T0 - 5400, now=T0,
        level="act", disabled_tools=["toggle_evolution", "request_restart"], spent_usd=4.0, daily_usd=20.0,
        running=1, max_tasks=2, interval=3300, events=observation.full_text())
    for key in wake.PLACEHOLDERS:
        assert "{" + key + "}" not in text, key
    opening = (repo / "prompts/CONSCIOUSNESS.md").read_text(encoding="utf-8").split("\n\n", 1)[0]
    assert text.startswith(opening + "\n\n"), "the actual prompt's complete opening reaches the consumer"
    assert "1 h 30 min ago" in text and "autonomy: Acts — Also acts on its own:" in text
    assert "toggle_evolution, request_restart" in text
    assert "allowance known spend (last 24 h): 4.00 / 20.00 USD" in text and "own tasks: 1/2" in text
    assert "next interval: 3300 s" in text
    assert "previous_wake" in wake.PLACEHOLDERS and "- previous wake: none" in text
    assert "- wake cause: the scheduled wake time; events never move it, they are listed as observed" in text
    assert text.count("- task t") == 15 and "more; see" not in text  # every terminal stays an event
    quiet = wake.render_wake_message(
        repo, drive_root=tmp_path / "empty", reason="heartbeat", last_wake_at=0.0, now=T0, level="full",
        disabled_tools=[], spent_usd=None, daily_usd=0, running=0, max_tasks=0, interval=900,
        events=_observe(tmp_path / "empty", since=T0 + 1).full_text())
    assert "no wake since this process started" in quiet and "wake cause: the scheduled wake time" in quiet
    assert "unavailable tools: none" in quiet and "allowance known spend (last 24 h): unknown / 0.00 USD" in quiet
    assert "Also changes itself: its own code, its settings, connecting new places." in quiet


def test_render_survives_a_missing_template_and_an_unreadable_task_store(tmp_path):
    (tmp_path / "task_results").mkdir()
    (tmp_path / "task_results" / "broken.json").write_text("{not json", encoding="utf-8")
    text = wake.render_wake_message(
        tmp_path / "no-repo", drive_root=tmp_path, reason="heartbeat", last_wake_at=0.0, now=T0, level="act",
        disabled_tools=[], spent_usd=0.0, daily_usd=20.0, running=0, max_tasks=2, interval=3300,
        events=_observe(tmp_path, since=T0 - 1).full_text())
    assert text.startswith("[Wake-up · heartbeat]") and "{" not in text


def test_template_parameters_and_the_explicitly_retained_lookahead():
    template = (REPO / "prompts" / "CONSCIOUSNESS.md").read_text(encoding="utf-8")
    import re

    assert set(re.findall(r"\{([a-z_]+)\}", template)) == set(wake.PLACEHOLDERS)
    # The owner chose to retain this exact tested hint in the first delivery. Other prose's meaning
    # belongs to review; render_substitutes above tests delivery rather than a second copy of that prose.
    assert "If you cannot say from something recent when your human is next expected by someone, look before you finish." in template
    assert "what you heard and did" in (REPO / "prompts/FOCUS.md").read_text(encoding="utf-8")


# --- the server projection --------------------------------------------------------


@pytest.fixture
def describe(monkeypatch):
    import server
    from ouroboros import consciousness_view

    holder = {}
    monkeypatch.setattr(consciousness_view, "time", SimpleNamespace(time=lambda: wake._parse_iso("2027-01-15T12:00:00Z")))
    monkeypatch.setattr(server, "_consciousness", SimpleNamespace(status_snapshot=lambda: dict(holder)))
    return lambda snapshot, enabled=True: (holder.clear(), holder.update(snapshot), server._describe_bg_consciousness_state(enabled))[2]


BASE = {"enabled": True, "level": "act", "next_wake_at": "2027-01-15T12:30:00+00:00",
        "last_wake_at": "", "last_wake_task_id": "", "last_wake_outcome": "", "last_error": "",
        "spent_24h_usd": 1.0, "daily_usd": 20.0, "allowance_resets_at": "", "tasks_running": 0, "max_tasks": 2,
        "live_wake_task_id": ""}


def test_projection_names_every_honest_status(describe):
    import server

    assert describe(BASE, enabled=False)["status"] == "disabled"
    assert describe(BASE, enabled=False)["enabled"] is False  # the caller's flag, not the snapshot's
    sleeping = describe(BASE)
    assert sleeping["status"] == "sleeping" and sleeping["detail"] == f"Sleeping until {server._clock_of(BASE['next_wake_at'])}."
    assert sleeping["next_wake_at"] == BASE["next_wake_at"] and sleeping["enabled"] is True
    # #1603: no early wake exists, so even a stale snapshot key is never rendered as one.
    stale = describe({**BASE, "pending_reason": "task_finished:a:completed"})
    assert "Early wake" not in stale["detail"] and stale["detail"] == sleeping["detail"]
    thinking = describe({**BASE, "live_wake_task_id": "wake0001"})
    assert thinking["status"] == "thinking" and "wake0001" in thinking["detail"]
    first = describe({**BASE, "last_wake_outcome": "skipped:waiting_for_first_conversation"})
    assert first["status"] == "waiting_for_first_conversation"
    exhausted = describe({**BASE, "last_wake_outcome": "skipped:allowance_exhausted", "spent_24h_usd": 21.5,
                          "allowance_resets_at": "2027-01-15T18:00:00+00:00",
                          "next_wake_at": "2027-01-15T18:00:00+00:00"})  # a real skip waits no sooner than its reset
    assert exhausted["status"] == "allowance_exhausted"
    assert "$21.50 of $20.00" in exhausted["detail"] and server._clock_of("2027-01-15T18:00:00+00:00") in exhausted["detail"]
    assert "at least" not in exhausted["detail"] and "degraded" not in exhausted["detail"]
    # PLAN 5.5: an unmetered or quarantined ledger makes the number a floor, and the status says so.
    floor = describe({**BASE, "last_wake_outcome": "skipped:allowance_exhausted", "spent_24h_usd": 21.5,
                      "unknown_unmetered": 2, "integrity_degraded": True})
    assert "at least $21.50 of $20.00" in floor["detail"] and "ledger integrity degraded" in floor["detail"]
    unknown = describe({**BASE, "last_wake_outcome": "skipped:allowance_unknown", "last_error": "OSError: ledger"})
    assert unknown["status"] == "allowance_unknown" and "OSError: ledger" in unknown["detail"]
    rejected = describe({**BASE, "last_wake_outcome": "rejected:budget_exhausted"})
    assert rejected["status"] == "wake_rejected" and "budget_exhausted" in rejected["detail"]
    failed = describe({**BASE, "last_wake_outcome": "failed", "last_error": "wake-up w1 failed in its runner"})
    assert failed["status"] == "wake_failed" and "backing off" in failed["detail"]
    done = describe({**BASE, "last_wake_outcome": "done", "last_wake_at": "2027-01-15T11:35:00+00:00"})
    assert done["status"] == "sleeping"


def test_projection_without_a_constructed_clock_says_why_and_is_never_running(monkeypatch):
    import server

    monkeypatch.setattr(server, "_consciousness", None)
    monkeypatch.setattr(server, "_supervisor_error", None)
    starting = server._describe_bg_consciousness_state(True)
    assert starting["status"] == "starting" and starting["detail"] == "Waits for Ouroboros to finish starting."
    monkeypatch.setattr(server, "_supervisor_error", "Supervisor init failed: boom")
    failed = server._describe_bg_consciousness_state(True)
    assert failed["status"] == "stopped" and failed["detail"].endswith("Restart Ouroboros.")
    assert server._describe_bg_consciousness_state(False)["status"] == "disabled"


def test_render_substitutes_placeholders_in_one_pass(tmp_path):
    """A task title that happens to contain "{daily_usd}" is a fact, not a placeholder."""
    since = T0 - 3600
    repo = tmp_path / "repo"
    (repo / "prompts").mkdir(parents=True)
    (repo / "prompts" / "CONSCIOUSNESS.md").write_text("spent {spent_usd} / {daily_usd}; events: {events}", encoding="utf-8")
    _result(tmp_path, "odd01", ts=since + 10, status="completed", cost=0.1, description="check {daily_usd} later")
    _chat(tmp_path, _summary("odd01", since + 10))
    events = _observe(tmp_path, since=since).full_text()
    text = wake.render_wake_message(repo, drive_root=tmp_path, reason="heartbeat", last_wake_at=since, now=T0, level="act",
                                    disabled_tools=[], spent_usd=1.0, daily_usd=20.0, running=0, max_tasks=2,
                                    interval=900, events=events)
    assert text.startswith("spent 1.00 / 20.00;") and "check {daily_usd} later" in text
    floor = wake.render_wake_message(repo, drive_root=tmp_path, reason="heartbeat", last_wake_at=since, now=T0, level="act",
                                     disabled_tools=[], spent_usd=1.0, daily_usd=20.0, running=0, max_tasks=2,
                                     interval=900, events=events, spent_is_floor=True)
    assert floor.startswith("spent at least 1.00 / 20.00;")


def test_new_result_with_old_timestamps_and_no_chat_row_is_observed_once(tmp_path):
    first = _wake(tmp_path, None, T0)
    _write(tmp_path, "late-write", status="completed", ts=_iso(T0 - 86400), updated_at=_iso(T0 - 80000))
    second = _wake(tmp_path, first.boundary, T0 + 600)
    assert [(kind, offset) for kind, offset, _ in second.events] == [("task_terminal", None)]
    assert "late-write Done" in second.full_text()
    assert _wake(tmp_path, second.boundary, T0 + 1200).events == ()


def test_partial_old_inventory_upgrades_with_explicit_coverage_gap(tmp_path):
    _write(tmp_path, "old", status="completed", ts=_iso(T0 - 86400))
    boundary = _wake(tmp_path, None, T0).boundary
    boundary["transitions"] = {"version": 1, "inventory": {}, "observed": [], "scan_at": _iso(T0)}
    second = _wake(tmp_path, boundary, T0 + 600)
    assert second.window["transitions_basis"] == "partial_inventory_upgrade"
    assert second.gaps and "task old Done" in second.full_text()
    assert _wake(tmp_path, second.boundary, T0 + 1200).events == ()


def test_wake_scan_reads_each_result_and_project_registry_once(tmp_path, monkeypatch):
    from collections import Counter
    from ouroboros import projects_registry as projects, store, task_store

    total = 240
    for index in range(total):
        _write(tmp_path, f"t{index:04d}", status="completed", ts=_iso(T0 - 10), updated_at=_iso(T0 - 10), project_id="p1")
    reads, project_reads = Counter(), []
    real_decode, real_body = task_store._decode, store.Txn.body_text

    def decode(*args):  # every body the scan decodes, by task
        rows = real_decode(*args)
        for row in rows:
            reads[row["task_id"]] += 1
        return rows

    def body_text(self, task_id):
        reads[task_id] += 1
        return real_body(self, task_id)

    monkeypatch.setattr(task_store, "_decode", decode)
    monkeypatch.setattr(store.Txn, "body_text", body_text)
    monkeypatch.setattr(projects, "list_reserved_projects", lambda _root: project_reads.append(True) or [
        {"id": "p1", "name": "Project", "chat_id": 42, "lifecycle": "active"}])
    observed = _wake(tmp_path, None, T0)
    assert len(observed.events) == total
    assert len(reads) == total and set(reads.values()) == {1}
    assert project_reads == [True]


def test_first_wake_uses_late_publication_when_old_readiness_debt_remains(tmp_path):
    _write(tmp_path, "late", status="failed", ts=_iso(T0 - 86400), updated_at=_iso(T0 - 10),
           canonical_terminal_projection_ready={"task_done_ts": _iso(T0 - 86400)},
           canonical_terminal_projection={"written_at": _iso(T0 - 10)},
           terminal_time={"occurred_at": _iso(T0 - 86400), "source": "executor_terminal"})
    first = _wake(tmp_path, None, T0)
    assert len(first.events) == 1 and first.events[0][0] == "task_terminal"
    assert "task late Failed" in first.events[0][2]
    assert _wake(tmp_path, first.boundary, T0 + 600).events == ()


def test_projection_discloses_parked_wake_and_unknown_outcome(describe):
    for outcome in ("paused", "pausing"):
        value = describe({**BASE, "last_wake_outcome": outcome, "tasks_running": 1})
        assert value["status"] == "wake_paused"
        assert outcome in value["detail"] and "returned while" in value["detail"]
        assert value["tasks_running"] == 1
    completed_since = describe({**BASE, "last_wake_outcome": "paused", "tasks_running": 0})
    assert "still occupies" not in completed_since["detail"] and completed_since["tasks_running"] == 0
    unknown = describe({**BASE, "last_wake_outcome": "unknown"})
    assert unknown["status"] == "wake_outcome_unknown" and "unconfirmed" in unknown["detail"]


def test_narrow_task_window_equals_full_scan_and_reads_only_selected_bodies(tmp_path, monkeypatch):
    from ouroboros import task_store, task_results

    for tid, status in (("unchanged", "completed"), ("settling", "running"), ("changed", "completed"),
                        ("chat-addressed", "running"), ("open-card", "running")):
        _write(tmp_path, tid, status=status, ts=_iso(T0 - 600), updated_at=_iso(T0 - 600))
    _write(tmp_path, "open-card", owner_quiz={"q": {"state": "open", "question": "Still open?"}})
    first = _wake(tmp_path, None, T0)
    # Backdated changes and newly inserted results must still be selected.
    _write(tmp_path, "settling", status="completed", updated_at=_iso(T0 - 600))
    _write(tmp_path, "changed", owner_quiz={"closed": {"state": "closed", "question": "Old?",
                                                     "closed_at": _iso(T0 - 600)}})
    _write(tmp_path, "new-old-date", status="failed", ts=_iso(T0 - 900), updated_at=_iso(T0 - 900))
    _chat(tmp_path, _summary("chat-addressed", T0 + 10))
    selected = []
    original_decode = task_store._decode
    def decode(*args):
        rows = original_decode(*args)
        selected.extend(row["task_id"] for row in rows)
        return rows
    monkeypatch.setattr(task_store, "_decode", decode)
    narrow = _wake(tmp_path, first.boundary, T0 + 60)
    assert set(selected) == {"settling", "changed", "new-old-date", "open-card", "chat-addressed"}
    assert len(selected) == 5

    def full(root, _prior, _chat):
        rows = task_results.list_task_results(root, strict=True)
        with task_store.open_store(root, write=False) as txn:
            signatures = {tid: list(stamp) for tid, stamp in txn.write_seqs().items()}
        return rows, signatures, {}
    monkeypatch.setattr(wake, "_task_window", full)
    reference = _wake(tmp_path, first.boundary, T0 + 60)
    assert narrow == reference
    assert narrow.source_bytes() == reference.source_bytes()
    assert narrow.full_text() == reference.full_text()


def test_boundary_without_signatures_reads_once_then_preserves_inventory(tmp_path, monkeypatch):
    from ouroboros import task_store
    _write(tmp_path, "old", status="completed", ts=_iso(T0 - 10), updated_at=_iso(T0 - 10))
    first = _wake(tmp_path, None, T0)
    first.boundary["transitions"].pop("signatures")
    selected = []
    original = task_store._decode
    def decode(*args):
        rows = original(*args)
        selected.extend(row["task_id"] for row in rows)
        return rows
    monkeypatch.setattr(task_store, "_decode", decode)
    second = _wake(tmp_path, first.boundary, T0 + 60)
    assert selected == ["old"] and second.events == ()
    selected.clear()
    third = _wake(tmp_path, second.boundary, T0 + 120)
    assert selected == [] and third.events == ()
    assert third.boundary["transitions"]["inventory"] == first.boundary["transitions"]["inventory"]


# --- the accepted transition chain: a base and exact changes beside the observation ---


def _bind(root, observation, task_id):
    """Bind ``observation`` under the wake ``task_id`` as its carrier does: ``(accepted boundary, task)``."""
    task = {"id": task_id, "metadata": {}}
    return wake.bind_wake_observation(root, task, observation, lambda events: events), task


def _source_header(root, task):
    from ouroboros.artifacts import read_actor_source_bytes

    source = task["metadata"][wake.WAKE_OBSERVATION_KEY]["source"]
    return read_actor_source_bytes(root, source["task_id"], source).splitlines()[0]


def test_the_source_header_stays_small_and_the_next_wake_reads_the_whole_state_from_the_chain(tmp_path, monkeypatch):
    """However many tasks the store holds, the wake source keeps a small header: no inventory, no signatures, its
    boundary naming the chain link. The next wake reads the complete accepted state from that link: a known key is not
    fresh, a new one is, an unchanged task keeps its key without its body being read, and the text is the one the same
    state gives inline."""
    from ouroboros import task_store
    from ouroboros.consciousness_transitions import read_chain

    for index in range(300):
        _write(tmp_path, f"old{index:03d}", status="completed", ts=_iso(T0 - 86400), updated_at=_iso(T0 - 86400))
    _write(tmp_path, "kid", status="running", ts=_iso(T0 - 600), updated_at=_iso(T0 - 600))
    first = _wake(tmp_path, None, T0)
    accepted, task = _bind(tmp_path, first, "wake-a")
    header = _source_header(tmp_path, task)
    assert len(header) < 2048 < len(json.dumps(first.boundary["transitions"]))  # what the header no longer carries
    assert b'"inventory"' not in header and b'"signatures"' not in header
    assert json.loads(header)["boundary"]["transition_chain"] == accepted["transition_chain"]
    assert "transitions" not in accepted
    assert read_chain(tmp_path, accepted["transition_chain"]).state == first.boundary["transitions"]
    _write(tmp_path, "kid", status="completed", updated_at=_iso(T0 + 100))  # a known task's new terminal
    _write(tmp_path, "late", status="completed", ts=_iso(T0 - 86400), updated_at=_iso(T0 - 86400))  # new, old stamps
    decoded, original = [], task_store._decode

    def decode(*args):
        rows = original(*args)
        decoded.extend(row["task_id"] for row in rows)
        return rows

    monkeypatch.setattr(task_store, "_decode", decode)
    second = _wake(tmp_path, accepted, T0 + 600)
    assert second.window["transitions_basis"] == "accepted_inventory" and not second.gaps
    assert sorted(decoded) == ["kid", "late"]  # the 300 unchanged tasks keep their keys unread
    assert sorted(line.split()[2] for line in _lines(second)) == ["kid", "late"]
    assert second.boundary["transitions"]["inventory"]["old000"] == first.boundary["transitions"]["inventory"]["old000"]
    inline = _wake(tmp_path, first.boundary, T0 + 600)
    assert second.events == inline.events and second.full_text() == inline.full_text()


def test_changes_read_back_as_the_newest_state_and_a_base_comes_when_they_reach_the_base(tmp_path):
    """A base, then exact changes (a task added, one changed, one gone as ``null``) read back as the newest whole state.
    The change that would bring the chain's changes, each counted as at least one block, to its base's bytes is written
    as a new base, so reading a chain never costs more than two bases. A window that changes nothing writes no file."""
    from ouroboros import consciousness_transitions as ct

    directory = tmp_path / ct.CHAIN_REL
    inventory = {f"t{index:04d}": [f"terminal:t{index:04d}:completed:{index:016x}"] for index in range(200)}
    state = {"version": ct.TRANSITIONS_VERSION, "scan_at": _iso(T0), "inventory": inventory,
             "signatures": {task: [index, _iso(T0 - index)] for index, task in enumerate(inventory)}}
    chain = ct.read_chain(tmp_path, ct.bind_link(tmp_path, None, state))
    data = (directory / f"{chain.link}.json").read_bytes()
    assert chain.state == state and len(chain.names) == 1 and chain.change_bytes == 0
    assert hashlib.sha256(data).hexdigest() == chain.link and data == json.dumps(
        {**state, "version": 3}, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    files = sorted(directory.iterdir())
    assert ct.bind_link(tmp_path, chain, {**state, "scan_at": _iso(T0 + 60)}) == {"version": 3, "link": chain.link}
    assert sorted(directory.iterdir()) == files  # nothing changed: no file, the same link
    changes = -(-chain.base_bytes // ct.BLOCK_BYTES) - 1  # each counted as one block, together below the base
    assert changes >= 2
    kinds, parent = [], chain.link
    for step in range(1, changes + 2):
        added, gone = f"n{step:04d}", f"t{step:04d}"
        state = {**state, "scan_at": _iso(T0 + step),
                 "inventory": {**{k: v for k, v in state["inventory"].items() if k != gone}, added: []},
                 "signatures": {**{k: v for k, v in state["signatures"].items() if k != gone}, added: [step, _iso(T0)],
                                "t0199": [1000 + step, _iso(T0 + step)]}}
        chain = ct.read_chain(tmp_path, ct.bind_link(tmp_path, chain, state))
        assert chain.state == state and chain.change_bytes < chain.base_bytes
        kinds.append("base" if len(chain.names) == 1 else "change")
        if step == 1:
            record = json.loads((directory / f"{chain.link}.json").read_bytes())
            assert record["parent"] == parent and record["inventory"] == {added: [], gone: None}
            assert record["signatures"] == {added: [step, _iso(T0)], gone: None, "t0199": [1001, _iso(T0 + 1)]}
    assert kinds == ["change"] * changes + ["base"]


def test_a_previous_form_boundary_reads_its_source_and_its_wake_writes_a_base(tmp_path):
    """A boundary an older body accepted names its state in the wake source's header (``transitions.source_ref``):
    the next wake reads it there with the result the same state gives inline, and writes a base of the chain."""
    from ouroboros.chat_chain import retain_memory_source
    from ouroboros.consciousness_transitions import CHAT_BOUNDARY_KEYS, read_chain

    _write(tmp_path, "kid", status="running", ts=_iso(T0 - 600), updated_at=_iso(T0 - 600))
    _write(tmp_path, "old", status="completed", ts=_iso(T0 - 86400), updated_at=_iso(T0 - 86400))
    first = _wake(tmp_path, None, T0)
    header = {"kind": "wake_observation", "captured_at": first.captured_at, "window": first.window,
              "gaps": list(first.gaps), "composition": first.counts(), "trigger": first.trigger, "boundary": first.boundary}
    data = (json.dumps(header, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")
    source = retain_memory_source(SimpleNamespace(drive_root=tmp_path, task_id="wake-old"), "wake_observation", data,
                                  extension="jsonl")
    previous = {**{key: first.boundary[key] for key in (*CHAT_BOUNDARY_KEYS, "memory") if key in first.boundary},
                "transitions": {"version": 2, "source_ref": source}, "task_id": "wake-old",
                "captured_at": first.captured_at}
    _write(tmp_path, "kid", status="completed", updated_at=_iso(T0 + 100))
    upgraded, inline = _wake(tmp_path, previous, T0 + 600), _wake(tmp_path, first.boundary, T0 + 600)
    assert upgraded.window["transitions_basis"] == "accepted_inventory" and not upgraded.gaps
    assert upgraded.events == inline.events and upgraded.full_text() == inline.full_text()
    accepted, _task = _bind(tmp_path, upgraded, "wake-new")
    chain = read_chain(tmp_path, accepted["transition_chain"])
    assert len(chain.names) == 1 and chain.state == upgraded.boundary["transitions"]  # a base of the whole state
    assert _wake(tmp_path, accepted, T0 + 1200).events == ()


@pytest.mark.parametrize("fault", ["missing", "foreign_bytes"])
def test_an_accepted_link_that_cannot_be_read_is_a_disclosed_gap_and_offers_no_boundary(tmp_path, fault):
    """A link that is gone, or holds other bytes than its name, is the unreadable source: the gap is named, no
    boundary is offered and nothing is guessed by time. The same link intact reads and advances."""
    from ouroboros.consciousness_transitions import CHAIN_REL

    _write(tmp_path, "kid", status="running", ts=_iso(T0 - 600), updated_at=_iso(T0 - 600))
    accepted, _task = _bind(tmp_path, _wake(tmp_path, None, T0), "wake-a")
    _write(tmp_path, "kid", status="completed", updated_at=_iso(T0 + 100))
    healthy = _wake(tmp_path, accepted, T0 + 600)
    assert healthy.boundary is not None and not healthy.gaps and "- task kid Done" in _lines(healthy)[0]
    link = tmp_path / CHAIN_REL / f"{accepted['transition_chain']['link']}.json"
    if fault == "missing":
        link.unlink()
    else:
        link.write_bytes(b'{"inventory":{},"signatures":{},"version":3}')
    blind = _wake(tmp_path, accepted, T0 + 600)
    assert blind.boundary is None and blind.window["transitions_basis"] == "unreadable_transition_source"
    assert any(gap.startswith("unreadable_transition_source: ") for gap in blind.gaps)
    assert "accepted inventory source unreadable; boundary unchanged" in blind.full_text()
    assert _bind(tmp_path, blind, "wake-b")[0] is None


@pytest.mark.parametrize("fault", ["write", "readback"])
def test_a_link_that_cannot_be_written_or_read_back_is_a_source_error_and_accepts_nothing(tmp_path, monkeypatch, fault):
    """The link is written before the source; when it cannot be written, or reads back other bytes, the wake keeps its
    full input and accepts no boundary (``source_error``), as when its source cannot be bound."""
    from ouroboros import consciousness_transitions as ct

    _write(tmp_path, "kid", status="running", ts=_iso(T0 - 600), updated_at=_iso(T0 - 600))
    accepted, _task = _bind(tmp_path, _wake(tmp_path, None, T0), "wake-a")
    assert accepted is not None and "transition_chain" in accepted
    _write(tmp_path, "kid", status="completed", updated_at=_iso(T0 + 100))
    real = ct.write_bytes_atomic

    def write(path, data, *, fsync=False):
        if fault == "write":
            raise OSError("disk full")
        real(path, data + b" ", fsync=fsync)  # other bytes than the ones written

    monkeypatch.setattr(ct, "write_bytes_atomic", write)
    refused, task = _bind(tmp_path, _wake(tmp_path, accepted, T0 + 600), "wake-b")
    facts = task["metadata"][wake.WAKE_OBSERVATION_KEY]
    assert refused is None and "source" not in facts
    assert facts["source_error"] == ("OSError: disk full" if fault == "write"
                                     else "ValueError: transition link readback mismatch")


def test_reading_the_accepted_chain_removes_every_other_file_and_an_unreadable_chain_removes_nothing(tmp_path):
    """Once a wake has read its accepted chain it removes what no boundary will name: a link of a wake whose boundary
    was never accepted, the lineage before the newest base, a leftover temporary file; the chain's own links stay. A
    chain that cannot be read removes nothing."""
    from ouroboros import consciousness_transitions as ct

    directory = tmp_path / ct.CHAIN_REL
    _write(tmp_path, "kid", status="running", ts=_iso(T0 - 600), updated_at=_iso(T0 - 600))
    accepted, _task = _bind(tmp_path, _wake(tmp_path, None, T0), "wake-a")
    _write(tmp_path, "kid", status="completed", updated_at=_iso(T0 + 100))
    unaccepted, _task = _bind(tmp_path, _wake(tmp_path, accepted, T0 + 600), "wake-b")  # its terminal never came
    (directory / ".leftover.json.tmp").write_bytes(b"partial")
    kept = {f"{accepted['transition_chain']['link']}.json"}
    stray = {f"{unaccepted['transition_chain']['link']}.json", ".leftover.json.tmp"}
    assert {path.name for path in directory.iterdir()} == kept | stray
    blind = _wake(tmp_path, {**accepted, "transition_chain": {"version": 3, "link": "f" * 64}}, T0 + 600)
    assert blind.boundary is None and {path.name for path in directory.iterdir()} == kept | stray
    again = _wake(tmp_path, accepted, T0 + 600)
    assert not again.gaps and again.boundary is not None and {path.name for path in directory.iterdir()} == kept
    base = ct.bind_link(tmp_path, None, again.boundary["transitions"])  # a newer base, as after a previous-form boundary
    _wake(tmp_path, {**accepted, "transition_chain": base}, T0 + 1200)
    assert {path.name for path in directory.iterdir()} == {f"{base['link']}.json"}


def test_the_export_for_an_older_body_writes_the_previous_form_it_reads_with_the_same_state(tmp_path, monkeypatch):
    """Before a switch to a body without the chain reader, the accepted chain is written back as the wake source a
    previous-form boundary names: the reader of that form (the code the older body runs) gets the state the chain
    assembles, and the boundary names no chain. A previous-form boundary, or none, has nothing to export."""
    from ouroboros import consciousness_transitions as ct
    from ouroboros.consciousness import OBSERVATION_STATE_KEY
    from supervisor import state as runtime_state

    store = {}
    monkeypatch.setattr(runtime_state, "load_state", lambda: dict(store))
    monkeypatch.setattr(runtime_state, "update_state", lambda mutator, **_kw: mutator(store) or dict(store))
    assert ct.export_for_older_body(tmp_path) == {"status": "nothing_to_export"}
    for index in range(60):  # a base above one block, so the next window is a change
        _write(tmp_path, f"old{index:02d}", status="completed", ts=_iso(T0 - 86400), updated_at=_iso(T0 - 86400))
    _write(tmp_path, "kid", status="running", ts=_iso(T0 - 600), updated_at=_iso(T0 - 600))
    accepted, _task = _bind(tmp_path, _wake(tmp_path, None, T0), "wake-a")
    _write(tmp_path, "kid", status="completed", updated_at=_iso(T0 + 100))
    accepted, _task = _bind(tmp_path, _wake(tmp_path, accepted, T0 + 600), "wake-b")
    chain = ct.read_chain(tmp_path, accepted["transition_chain"])
    assert len(chain.names) == 2  # a base and one change
    store[OBSERVATION_STATE_KEY] = accepted
    assert ct.export_for_older_body(tmp_path)["status"] == "exported"
    older = store[OBSERVATION_STATE_KEY]
    assert "transition_chain" not in older and older["transitions"]["version"] == 2
    assert older == {**{key: value for key, value in accepted.items() if key != "transition_chain"},
                     "transitions": {"version": 2, "source_ref": older["transitions"]["source_ref"]}}
    assert ct.read_source_transitions(tmp_path, older) == chain.state
    assert ct.export_for_older_body(tmp_path) == {"status": "nothing_to_export"}
    _write(tmp_path, "late", status="completed", ts=_iso(T0 - 86400), updated_at=_iso(T0 - 86400))
    through_source, through_chain = _wake(tmp_path, older, T0 + 1200), _wake(tmp_path, accepted, T0 + 1200)
    assert through_source.full_text() == through_chain.full_text() and "- task late Done" in _lines(through_source)[0]
