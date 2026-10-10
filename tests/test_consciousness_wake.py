"""The wake-up message, its physical-window observation and envelope (Background Consciousness),
and the server's projection of the alarm clock's snapshot into one honest status line."""

from __future__ import annotations

import json
import pathlib
from types import SimpleNamespace

import pytest

from ouroboros import consciousness_wake as wake

REPO = pathlib.Path(__file__).resolve().parents[1]
T0 = 1_800_000_000.0


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
    (root / "task_results" / f"{task_id}.json").write_text(json.dumps(row), encoding="utf-8")


def test_wake_task_metadata_carries_origin_level_and_tree_cap(monkeypatch):
    monkeypatch.setenv("OUROBOROS_CONSCIOUSNESS_AUTONOMY", "act")
    meta = wake.wake_task_metadata("observe", "heartbeat", root_cost_ceiling_usd=3.5)
    assert meta["initiator"] == "consciousness" and meta["usage_category"] == "consciousness"
    assert meta["consciousness_autonomy"] == "observe" and meta["runtime_mode_cap"] == "light"
    assert meta["model_role"] == "consciousness" and meta["wake_reason"] == "heartbeat"
    # P3e: the cap the wake's own root scope binds, never the non-root member ceiling.
    assert "promote_chat_to_task" in meta["disabled_tools"] and meta["root_cost_ceiling_usd"] == 3.5
    assert "root_limit_usd" not in meta
    full = wake.wake_task_metadata("full", "event:x")
    assert full["disabled_tools"] == [] and full["runtime_mode_cap"] == "" and "root_cost_ceiling_usd" not in full
    # A non-positive cap is not stamped: it would read as "no narrowing" downstream and
    # the alarm never launches on an exhausted allowance anyway.
    assert "root_cost_ceiling_usd" not in wake.wake_task_metadata("act", "heartbeat", root_cost_ceiling_usd=0.0)
    assert wake.wake_task_metadata("bogus", "")["consciousness_autonomy"] == "act"  # falls back to the setting


def _chat(root, *rows, archive: str = ""):
    """Append rows to the live chat log, or write them as one rotated archive segment."""
    path = root / ("archive" if archive else "logs") / (f"chat_{archive}.jsonl" if archive else "chat.jsonl")
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
    assert lines[0].startswith('- owner message in Main, ') and lines[0].endswith('"first human line"')
    assert [line.split()[2] for line in lines[1:16]] == [f"t{index:02d}" for index in range(15)]
    assert "- task t14 completed, $1.25" in lines[15] and "work 14" in lines[15]
    assert len(observation.outstanding) == 7  # every answerable card, newest first
    assert [line.split()[3] for line in observation.outstanding] == [f"c{i}" for i in reversed(range(7))]
    text = observation.full_text()
    assert "2 owner messages, 15 task terminals, 7 outstanding cards" in text
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
    assert third.events == () and third.full_text() == "\n- wake cause: scheduled heartbeat; no event reason is recorded for this wake"
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
    assert [line.rsplit('"', 2)[-2] for line in _lines(rotated)] == ["after rotation"]
    # A chain whose segment at the boundary no longer starts with the same line is not trusted.
    (tmp_path / "archive" / "chat_20270115T100000.jsonl").write_text(
        json.dumps(_owner("rewritten history", since + 5, cmid="z")) + "\n", encoding="utf-8")
    replaced = _observe(tmp_path, boundary=accepted, since=T0 - 60)
    assert replaced.window["basis"] == "time_bootstrap_boundary_mismatch"
    assert "accepted_boundary_no_longer_matches_chat_chain" in replaced.gaps
    assert [line.rsplit('"', 2)[-2] for line in _lines(replaced)] == ["after rotation"]


# --- canonical transitions: task results by identity, whether or not a chat row announced them ---


def _write(root, task_id, **fields):
    """A task result row exactly as stored (``ts`` is the first write; nothing else is inferred)."""
    (root / "task_results").mkdir(exist_ok=True)
    path = root / "task_results" / f"{task_id}.json"
    row = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {
        "task_id": task_id, "_schema_version": 1, "metadata": {}, "description": f"do {task_id}"}
    row.update(fields)
    path.write_text(json.dumps(row), encoding="utf-8")
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
    assert lines[0].startswith("- child task kid1 of root1 failed, completion time not recorded (last updated 8 min ago)")
    assert lines[1].startswith("- task root1 completed, completion time not recorded (last updated 6 min ago)")
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
    assert _lines(third) == ["- task brief completed, 5 min ago: do brief"]  # its task_done stamp, proven
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
    assert card.startswith("- owner card q1 on task asker: Unanswered · the task finished; a late answer is accepted")
    assert "Ship it?" in card
    assert len(second.outstanding) == 1 and "q1" in second.outstanding[0]  # still answerable
    assert "1 card closed unanswered" in second.full_text()
    # The owner answers late: that is the next transition, the expiry is not repeated.
    path = tmp_path / "task_results" / "asker.json"
    row = json.loads(path.read_text(encoding="utf-8"))
    row["owner_quiz"]["q1"].update(state="answered", answered_at=_iso(T0 + 700), comment="yes")
    path.write_text(json.dumps(row), encoding="utf-8")
    third = _wake(tmp_path, second.boundary, T0 + 1200)
    assert [kind for kind, _o, _l in third.events] == ["card_answer"]
    assert "answered in own words: yes" in _lines(third)[0] and third.outstanding == ()


def _late_panel(tmp_path, task_id, *, settled_at=None, pending=False):
    """The acceptance panel exactly as the review writer publishes it (real producer functions)."""
    from ouroboros.acceptance_settlement import late_evidence_fact
    from ouroboros.artifacts import store_actor_source_bytes
    from ouroboros.review_projection import compact_review_projection

    actor = {"slot_id": "s1", "operation_id": "op1", "status": "ok", "raw_text": "FAIL: the tests were never run",
             "semantic_verdict": "FAIL", "operation_state": "in_flight" if pending else "settled"}
    run = {"authority": "host_root", "panel_id": "p1", "aggregate_signal": "FAIL", "actors": [actor],
           "request": {"surface": "task_acceptance", "retry_key": "rk-9", "subject": "answer A", "task_id": task_id}}
    if settled_at:
        fact = late_evidence_fact(run, {"source": "terminal_delivery_registry", "state": "unknown",
                                        "delivery_ids": []}, settled_at=_iso(settled_at))
        run["late_settlement"] = {"note": "On the reviewed version of this answer, reviewers later rejected it.\n"
                                          "- s1: FAIL", **fact}
    ref = store_actor_source_bytes(tmp_path, task_id, category="context_checkpoints", source_id="acceptance",
                                   data=json.dumps(run).encode(), extension="json")
    run['applied_source_ref'] = ref
    return compact_review_projection([run]), ref


def test_a_late_review_is_read_from_its_canonical_fact_with_its_exact_source(tmp_path):
    """No invented chat fields: the settlement, its subject and its exact published source come
    from ``acceptance_settlement.late_acceptance_facts``; the source is readable by read_file."""
    from ouroboros.tools.registry import ToolRegistry
    from tests.test_review_operation_source_closure import read_late_source
    from ouroboros.tools.tool_context import ToolContext

    projection, ref = _late_panel(tmp_path, "rv", pending=True)
    _write(tmp_path, "rv", status="completed", ts=_iso(T0 - 900), updated_at=_iso(T0 - 100),
           review_projection=projection)
    first = _wake(tmp_path, None, T0)
    assert "rv" in first.boundary["transitions"]["inventory"]  # its panel may still settle
    projection, ref = _late_panel(tmp_path, "rv", settled_at=T0 + 60)
    _write(tmp_path, "rv", review_projection=projection)
    second = _wake(tmp_path, first.boundary, T0 + 600)  # the outbox row has not landed yet
    assert [(kind, offset) for kind, offset, _l in second.events] == [("late_review", None)]
    line = _lines(second)[0]
    assert line.startswith("- late review settled for task rv, 9 min ago: On the reviewed version of this answer, "
                           "reviewers later rejected it.; panel p1; signal FAIL; reviewed version unknown; "
                           "emitted answer unknown; 1 reviewer outputs; exact source ")
    assert f"sha256 {ref['sha256']}" in line and "rv" in second.boundary["transitions"]["inventory"]
    read = json.loads(line.split("exact source ", 1)[1].rsplit(" sha256 ", 1)[0])
    reader = ToolContext(repo_dir=REPO, drive_root=tmp_path, task_id="wake0002", task_metadata={})
    registry = ToolRegistry(repo_dir=REPO, drive_root=tmp_path)
    registry.set_context(reader)
    assert '"FAIL: the tests were never run"' in read_late_source(registry, read)
    # The owner-visible row lands afterwards: already accepted, not a second report.
    _chat(tmp_path, {"ts": _iso(T0 + 700), "direction": "system", "type": "acceptance_late_settlement",
                     "task_id": "rv", "text": "On the reviewed version…", "card_row": "reviews",
                     "card_row_id": "acceptance-late:rk-9"})
    assert _wake(tmp_path, second.boundary, T0 + 1200).events == ()


def test_a_transition_takes_the_position_of_the_row_that_announced_it_once(tmp_path):
    projection, _ref = _late_panel(tmp_path, "rv", pending=True)
    _write(tmp_path, "rv", status="completed", ts=_iso(T0 - 900), updated_at=_iso(T0 - 100),
           review_projection=projection)
    _write(tmp_path, "kid", status="running", ts=_iso(T0 - 900), updated_at=_iso(T0 - 900))
    first = _wake(tmp_path, None, T0)
    projection, _ref = _late_panel(tmp_path, "rv", settled_at=T0 + 60)
    _write(tmp_path, "rv", review_projection=projection)
    _write(tmp_path, "kid", status="completed", updated_at=_iso(T0 + 90))
    late_row = {"ts": _iso(T0 + 61), "direction": "system", "type": "acceptance_late_settlement", "task_id": "rv",
                "text": "On the reviewed version…", "card_row": "reviews", "card_row_id": "acceptance-late:rk-9"}
    _chat(tmp_path, _owner("hello", T0 + 30, cmid="h"), late_row, dict(late_row), _summary("kid", T0 + 95))
    second = _wake(tmp_path, first.boundary, T0 + 600)
    assert [kind for kind, _o, _l in second.events] == ["owner_message", "late_review", "task_terminal"]
    assert all(offset is not None for _k, offset, _l in second.events)  # each at its announcing row, once
    assert "no chat row in this window" not in second.full_text()
    assert "- task kid completed" in _lines(second)[2]


def test_a_result_first_written_late_under_an_old_stamp_is_found_through_its_row(tmp_path):
    """The compatibility terminal persist may create a row after the scan with the task's
    original (older) ``ts``: its first observed identity counts, at its announcing row."""
    _write(tmp_path, "seed", status="running", ts=_iso(T0 - 10), updated_at=_iso(T0 - 10))
    second = _wake(tmp_path, _wake(tmp_path, None, T0).boundary, T0 + 600)
    _write(tmp_path, "revived", status="completed", ts=_iso(T0 - 86400), updated_at=_iso(T0 + 700))
    _chat(tmp_path, _summary("revived", T0 + 700))
    third = _wake(tmp_path, second.boundary, T0 + 1200)
    assert [kind for kind, offset, _l in third.events if offset is not None] == ["task_terminal"]
    assert "- task revived completed" in _lines(third)[0]


def test_an_unreadable_task_store_keeps_the_accepted_inventory(tmp_path, monkeypatch):
    import ouroboros.task_results as task_results

    _write(tmp_path, "kid", status="running", ts=_iso(T0 - 900), updated_at=_iso(T0 - 900))
    first = _wake(tmp_path, None, T0)
    real = task_results.list_task_results
    monkeypatch.setattr(task_results, "list_task_results", lambda _root, **_kw: (_ for _ in ()).throw(OSError("busy")))
    _write(tmp_path, "kid", status="completed", updated_at=_iso(T0 + 60))
    blind = _wake(tmp_path, first.boundary, T0 + 600)
    assert blind.boundary["transitions"] == first.boundary["transitions"]  # nothing consumed
    assert "task_results unreadable: OSError" in blind.gaps
    monkeypatch.setattr(task_results, "list_task_results", real)
    assert "- task kid completed" in _lines(_wake(tmp_path, blind.boundary, T0 + 1200))[0]


def test_unfinished_and_malformed_lines_are_gaps_not_silence(tmp_path):
    since = T0 - 3600
    path = _chat(tmp_path, _owner("complete", since + 1, cmid="a"))
    with path.open("a", encoding="utf-8") as handle:
        handle.write("{not json}\n")
        handle.write(json.dumps(_owner("unfinished", since + 2, cmid="b"))[:-1])
    first = _observe(tmp_path)
    assert [line.rsplit('"', 2)[-2] for line in _lines(first)] == ["complete"]
    assert "malformed_jsonl" in first.gaps and "gaps: malformed_jsonl" in first.full_text()
    with path.open("a", encoding="utf-8") as handle:
        handle.write("}\n")  # the writer completes its line after the snapshot
    second = _observe(tmp_path, boundary=first.boundary)
    assert [line.rsplit('"', 2)[-2] for line in _lines(second)] == ["unfinished"]


@pytest.mark.parametrize('rotate', [False, True])
def test_first_partial_line_survives_completion_after_accepted_wake(tmp_path, rotate):
    path = _chat(tmp_path)
    raw = json.dumps(_owner('old timestamp, newly complete', T0 - 100, cmid='partial'))
    path.write_text(raw[:-1], encoding='utf-8')
    first = wake.observe_wake(tmp_path, boundary=None, since=T0 - 500, now=T0)
    assert not first.events and first.boundary['upper'] == 0
    with path.open('a', encoding='utf-8') as handle:
        handle.write('}\n')
    if rotate:
        archive = tmp_path / 'archive'
        archive.mkdir()
        path.rename(archive / 'chat_2027-01-15.jsonl')
        path.write_text('', encoding='utf-8')
    second = wake.observe_wake(tmp_path, boundary=first.boundary, since=T0 + 10, now=T0 + 100)
    assert len(second.events) == 1 and 'newly complete' in second.full_text()
    assert second.window['basis'] == 'accepted_boundary' and not second.gaps
    third = wake.observe_wake(tmp_path, boundary=second.boundary, since=T0 + 110, now=T0 + 200)
    assert third.events == ()


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
          # The durable row as the producer writes it (``log_chat``): its text and card placement only.
          {"ts": _iso(since + 50), "direction": "system", "type": "acceptance_late_settlement", "task_id": "t1",
           "text": "Late review of an earlier revision: 2 findings", "card_row": "reviews",
           "card_row_id": "acceptance-late:rk-1"},
          {"ts": _iso(since + 60), "direction": "system", "type": "task_error", "task_id": "w1",
           "text": "⚠️ Error: RuntimeError: boom", "task_terminal_status": "failed"})
    observation = _observe(tmp_path)
    assert [kind for kind, _o, _l in observation.events] == [
        "owner_message", "correspondent_message", "card_answer", "late_review", "task_error"]
    lines = _lines(observation)
    assert "Presence correspondent message" in lines[1] and "from a correspondent" in lines[1]
    assert lines[2].startswith("- owner card q9 on task asker: answered") and "chose option 1: Keep" in lines[2]
    # No task result records that settlement: the row's own words, never invented subject fields.
    assert lines[3] == "- late review announced for task t1, 59 min ago: Late review of an earlier revision: 2 findings"
    assert "task w1 runner failed" in lines[4]
    text = observation.full_text()
    assert "I opened this cycle" not in text and "machine traffic" not in text and "my reply" not in text


@pytest.mark.parametrize("status, origin", [("failed", "host_notice"), ("cancelled", ""),
                                           ("completed", "host_notice"), ("failed", "model_final")])
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
    own = _lines(_observe(tmp_path, reason="task_finished:asker:completed"))[0]
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

def test_project_digest_pins_human_project_and_related_task_before_cards(tmp_path):
    since = T0 - 3600
    (tmp_path / "state").mkdir()
    (tmp_path / "state" / "projects.json").write_text(json.dumps({
        "projects": [{"id": "p1", "name": "System audit", "chat_id": 42, "lifecycle": "active"}],
    }), encoding="utf-8")
    _result(tmp_path, "finished", ts=since - 10, project_id="p1", description="audit completed")
    _result(tmp_path, "still-running", ts=since + 20, status="running", project_id="p1", description="in progress")
    _result(tmp_path, "direct-finished", ts=since + 30, direct=True, project_id="p1", description="owner chat")
    _result(tmp_path, "card-task", ts=since - 100, status="running",
            quizzes={"q1": {"state": "expired_terminal", "asked_at": _iso(since - 50),
                              "question": "Should the audit continue?"}})
    observation = _observe(tmp_path, since=since, reason="project_digest:p1:finished")
    trigger = observation.trigger
    assert trigger.startswith("- wake cause: project System audit settled task finished (completed), $1.25")
    assert "audit completed" in trigger and "1 h 0 min ago" in trigger
    assert "still-running" not in trigger and "direct-finished" not in trigger
    card = observation.outstanding[0]
    assert card.startswith("- owner card q1 on task card-task: Unanswered · the task finished")
    assert "Should the audit continue?" in card
    text = observation.full_text()
    assert text.index(trigger) < text.index("Outstanding owner cards (1):") < text.index(card)


def test_project_digest_task_id_wins_when_multiple_settled_rows_share_a_project(tmp_path):
    since = T0 - 3600
    (tmp_path / "state").mkdir()
    (tmp_path / "state" / "projects.json").write_text(json.dumps({
        "projects": [{"id": "p1", "name": "System audit", "chat_id": 42, "lifecycle": "active"}],
    }), encoding="utf-8")
    _result(tmp_path, "trigger", ts=since - 10, project_id="p1", description="triggered result")
    _result(tmp_path, "newer", ts=since + 10, project_id="p1", description="newer result")
    trigger = _observe(tmp_path, since=since, reason="project_digest:p1:trigger").trigger
    assert "task trigger" in trigger and "triggered result" in trigger
    assert "latest settled" not in trigger
    assert "task newer" not in trigger


def test_trigger_stays_first_when_task_results_are_unreadable(tmp_path, monkeypatch):
    import ouroboros.task_results as task_results

    def broken(_root, **_kw):
        raise OSError("broken task store")

    monkeypatch.setattr(task_results, "list_task_results", broken)
    observation = _observe(tmp_path, since=T0 - 1, reason="task_finished:t1:completed")
    assert observation.trigger == "- wake cause: task t1 finished (completed)"
    assert observation.gaps == ("task_results unreadable: OSError",)
    assert observation.full_text().split("\n")[1:] == [
        "- wake cause: task t1 finished (completed)",
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
    observation = _observe(tmp_path, since=T0 - 5400, reason="task_finished:t14:completed")
    text = wake.render_wake_message(
        repo, reason="task_finished:t14:completed", last_wake_at=T0 - 5400, now=T0,
        level="act", disabled_tools=["toggle_evolution", "request_restart"], spent_usd=4.0, daily_usd=20.0,
        running=1, max_tasks=2, interval=3300, events=observation.full_text())
    for key in wake.PLACEHOLDERS:
        assert "{" + key + "}" not in text, key
    assert text.startswith("You are Ouroboros. No one has asked for a task")
    assert "1 h 30 min ago" in text and "autonomy: act — everything your runtime mode allows except" in text
    assert "toggle_evolution, request_restart" in text
    assert "allowance known spend (last 24 h): 4.00 / 20.00 USD" in text and "tasks running: 1/2" in text
    assert "next interval: 3300 s" in text
    assert "- wake cause: task t14 finished (completed)" in text
    assert text.count("- task t") == 15 and "more; see" not in text  # the trigger's own terminal stays an event
    quiet = wake.render_wake_message(
        repo, reason="heartbeat", last_wake_at=0.0, now=T0, level="full",
        disabled_tools=[], spent_usd=None, daily_usd=0, running=0, max_tasks=0, interval=900,
        events=_observe(tmp_path / "empty", since=T0 + 1).full_text())
    assert "no wake since this process started" in quiet and "wake cause: scheduled heartbeat" in quiet
    assert "unavailable tools: none" in quiet and "allowance known spend (last 24 h): unknown / 0.00 USD" in quiet
    assert "including evolution" in quiet


def test_render_survives_a_missing_template_and_an_unreadable_task_store(tmp_path):
    (tmp_path / "task_results").mkdir()
    (tmp_path / "task_results" / "broken.json").write_text("{not json", encoding="utf-8")
    text = wake.render_wake_message(
        tmp_path / "no-repo", reason="heartbeat", last_wake_at=0.0, now=T0, level="act",
        disabled_tools=[], spent_usd=0.0, daily_usd=20.0, running=0, max_tasks=2, interval=3300,
        events=_observe(tmp_path, since=T0 - 1).full_text())
    assert text.startswith("[Wake-up · heartbeat]") and "{" not in text


def test_template_names_only_its_placeholders_and_the_wake_hints():
    template = (REPO / "prompts" / "CONSCIOUSNESS.md").read_text(encoding="utf-8")
    import re

    assert set(re.findall(r"\{([a-z_]+)\}", template)) == set(wake.PLACEHOLDERS)
    for hint in ("A pause is a legitimate decision", "Distinguish incremental cash cost",
                 "do not request task acceptance", "wake facts", "recent facts"):
        assert hint in template, hint
    assert "a heartbeat, a task that finished, a project digest" not in template
    assert "up to 10 rounds" not in template and "300 seconds" not in template


# --- the server projection --------------------------------------------------------


@pytest.fixture
def describe(monkeypatch):
    import server

    holder = {}
    monkeypatch.setattr(server, "_consciousness", SimpleNamespace(status_snapshot=lambda: dict(holder)))
    return lambda snapshot, enabled=True: (holder.clear(), holder.update(snapshot), server._describe_bg_consciousness_state(enabled))[2]


BASE = {"enabled": True, "level": "act", "next_wake_at": "2027-01-15T12:30:00+00:00", "pending_reason": "",
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
    pending = describe({**BASE, "pending_reason": "task_finished:a:completed"})
    assert pending["detail"].endswith(" Early wake pending: task_finished:a:completed.")
    thinking = describe({**BASE, "live_wake_task_id": "wake0001"})
    assert thinking["status"] == "thinking" and "wake0001" in thinking["detail"]
    first = describe({**BASE, "last_wake_outcome": "skipped:waiting_for_first_conversation"})
    assert first["status"] == "waiting_for_first_conversation"
    exhausted = describe({**BASE, "last_wake_outcome": "skipped:allowance_exhausted", "spent_24h_usd": 21.5,
                          "allowance_resets_at": "2027-01-15T18:00:00+00:00"})
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


def test_projection_without_a_constructed_clock_is_stopped_not_running(monkeypatch):
    import server

    monkeypatch.setattr(server, "_consciousness", None)
    described = server._describe_bg_consciousness_state(True)
    assert described["status"] == "stopped" and "not constructed" in described["detail"]
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
    text = wake.render_wake_message(repo, reason="heartbeat", last_wake_at=since, now=T0, level="act",
                                    disabled_tools=[], spent_usd=1.0, daily_usd=20.0, running=0, max_tasks=2,
                                    interval=900, events=events)
    assert text.startswith("spent 1.00 / 20.00;") and "check {daily_usd} later" in text
    floor = wake.render_wake_message(repo, reason="heartbeat", last_wake_at=since, now=T0, level="act",
                                     disabled_tools=[], spent_usd=1.0, daily_usd=20.0, running=0, max_tasks=2,
                                     interval=900, events=events, spent_is_floor=True)
    assert floor.startswith("spent at least 1.00 / 20.00;")


def test_new_result_with_old_timestamps_and_no_chat_row_is_observed_once(tmp_path):
    first = _wake(tmp_path, None, T0)
    _write(tmp_path, "late-write", status="completed", ts=_iso(T0 - 86400), updated_at=_iso(T0 - 80000))
    second = _wake(tmp_path, first.boundary, T0 + 600)
    assert [(kind, offset) for kind, offset, _ in second.events] == [("task_terminal", None)]
    assert "late-write completed" in second.full_text()
    assert _wake(tmp_path, second.boundary, T0 + 1200).events == ()


def test_new_and_settled_panel_on_old_closed_task_is_observed_without_chat(tmp_path):
    _write(tmp_path, "old", status="completed", ts=_iso(T0 - 86400), updated_at=_iso(T0 - 80000))
    first = _wake(tmp_path, None, T0)
    projection, _ref = _late_panel(tmp_path, "old", settled_at=T0 - 70000)
    _write(tmp_path, "old", review_projection=projection)
    second = _wake(tmp_path, first.boundary, T0 + 600)
    assert [(kind, offset) for kind, offset, _ in second.events] == [("late_review", None)]
    assert "late review settled for task old" in second.full_text()
    assert _wake(tmp_path, second.boundary, T0 + 1200).events == ()


def test_partial_old_inventory_upgrades_with_explicit_coverage_gap(tmp_path):
    _write(tmp_path, "old", status="completed", ts=_iso(T0 - 86400))
    boundary = _wake(tmp_path, None, T0).boundary
    boundary["transitions"] = {"version": 1, "inventory": {}, "observed": [], "scan_at": _iso(T0)}
    second = _wake(tmp_path, boundary, T0 + 600)
    assert second.window["transitions_basis"] == "partial_inventory_upgrade"
    assert second.gaps and "task old completed" in second.full_text()
    assert _wake(tmp_path, second.boundary, T0 + 1200).events == ()


def test_wake_scan_reads_each_result_and_project_registry_once(tmp_path, monkeypatch):
    from collections import Counter
    from ouroboros import projects_registry as projects, task_results

    total = 240
    for index in range(total):
        _write(tmp_path, f"t{index:04d}", status="completed", ts=_iso(T0 - 10), updated_at=_iso(T0 - 10), project_id="p1")
    reads, project_reads = Counter(), []
    original = task_results.read_json_dict

    def read(path, *args, **kwargs):
        reads[str(path)] += 1
        return original(path, *args, **kwargs)

    monkeypatch.setattr(task_results, "read_json_dict", read)
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
    assert "task late failed" in first.events[0][2]
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
