"""K5 (call-fit consensus v3, D2): the list of running works as one line per work, delivered once.

The Runtime section's shorter form states each running work as one line of every fact of its row; a route that holds
the full rows keeps their bytes. The late roster note after a head whose list is in view says only what that list does
not already carry, and an owner question, stop or hold always again, whole.
"""
import copy
import json
from types import SimpleNamespace

from ouroboros import peer_roster
from ouroboros.context_sections import runtime_host_section
from ouroboros.live_works import WORKS_LINES_HEADING, live_works_present, work_line
from ouroboros.memory_floor import view_tokens
from tests import _live_works_fixture as fixture


def _parse(line):
    """The legend's grammar read back: ``- <task_id>`` then ``key=JSON`` pairs separated by one space."""
    assert line.startswith("- ")
    task_id, rest = line[2:].split(" ", 1)
    pairs, at, decoder = {}, 0, json.JSONDecoder()
    while at < len(rest):
        key = rest[at:rest.index("=", at)]
        value, end = decoder.raw_decode(rest, at + len(key) + 1)
        assert key not in pairs and (end == len(rest) or rest[end] == " ")
        pairs[key], at = value, end + 1
    return task_id, pairs


def _rebuild(line):
    """A ``works`` entry from its line, by the legend alone (the retained source by its digest and size)."""
    task_id, pairs = _parse(line)
    work = {"task_id": task_id, "open_questions": [], "waiting": []}
    names = {"state": "state", "title": "title", "started": "started_at"}
    focus = {}
    for key, value in pairs.items():
        if key in names:
            work[names[key]] = value
        elif key == "chat":
            work["room"] = {"chat_id": value, "project_id": pairs.get("project_id", ""),
                            "project_name": pairs.get("project_name", "")}
        elif key in ("project_id", "project_name"):
            continue
        elif key.startswith("focus"):
            focus[{"focus": "text", "focus_at": "authored_at", "focus_source": "source_ref"}.get(key, key[6:])] = value
        else:
            work[key] = value
    if focus:
        focus.setdefault("author_task_id", task_id)
        if "sha256" in focus:
            focus["source_handle"] = {"kind": "task_source", "root": "artifact_store",
                                      "sha256": focus.pop("sha256"), "size": focus.pop("size")}
        work["focus"] = focus
    return work


def _without_handle_path(work):
    shown = copy.deepcopy(work)
    shown.get("focus", {}).get("source_handle", {}).pop("path", None)
    return shown


def _unknown_tail_row():
    """A waiting root whose run nobody can confirm, whose task result this read could not open (the roster's own
    words for both), as ``work_entry`` lists it."""
    from ouroboros.gateway.work_status import work_tails

    return {"task_id": "w99unk", "title": "A work with an unread record", "state": "running",
            "room": {"chat_id": 1, "project_id": "", "project_name": None}, "started_at": None,
            "open_questions": [], "waiting": [{"kind": "unknown", "since": None, "reason": "model_handoff"}],
            "tails": work_tails([], [{"kind": "run", "state": "unknown", "run_id": "run-" + "r" * 32}]),
            "unavailable": {"task_result": "StoreBusy"}}


def test_every_fact_of_a_row_comes_back_from_its_line(tmp_path):
    """Every kind of row the roster lists (an open owner question, the owner's Pause with its author and words, a
    held Continue, a sleep, a model wait, a project room, a dated focus with its retained source) and an unknown tail
    on an unread record: the legend rebuilds each entry; only the retained source's path leaves, behind its reader."""
    fixture.build(tmp_path)
    present = live_works_present(tmp_path)
    works = present["works"] + [_unknown_tail_row()]
    assert {work["task_id"][3:6] for work in works} >= {"que", "pau", "hel", "sle", "mod", "pro", "unk"}
    for work in works:
        line = work_line(work)
        assert "\n" not in line and _rebuild(line) == _without_handle_path(work)
    paused = next(work for work in works if "stop" in work)
    assert "wait until I answer about the budget" in work_line(paused)  # the owner's words, in his Pause
    asked = next(work for work in works if work["open_questions"])
    assert json.dumps(asked["open_questions"][0]["question"], ensure_ascii=False) in work_line(asked)


def test_the_retained_focus_source_is_read_back_from_what_its_line_names(tmp_path):
    """The one fact the line leaves to a reader: the digest it states selects the retained bytes."""
    from ouroboros.task_finalization import focus_source_projection
    from ouroboros.task_results import load_task_result, write_task_result
    from ouroboros.tools.project_journal import _retain_focus_source
    from ouroboros.utils import utc_now_iso
    from tests._shared import write_queue_document

    source = {"reader": "get_task_result", "task_id": "w00foc"}
    handle = _retain_focus_source(tmp_path, "w00foc", source, "RETAINED_ANSWER of the reader")
    focus = {"text": "Reading the release notes", "author_task_id": "w00foc", "authored_at": "2026-11-26T10:00:00Z",
             "source_ref": source, "source_handle": handle}
    write_task_result(tmp_path, "w00foc", "running", title="Release", chat_id=1, focus=focus)
    write_queue_document(tmp_path / "state/queue_snapshot.json", {"ts": utc_now_iso(), "pending": [], "running": [
        {"id": "w00foc", "attempt": 1, "task": {"id": "w00foc", "chat_id": 1, "focus": focus}}]})
    [work] = live_works_present(tmp_path)["works"]
    named = _rebuild(work_line(work))["focus"]["source_handle"]
    read = focus_source_projection(tmp_path, "w00foc", load_task_result(tmp_path, "w00foc"), 0, named["size"],
                                   sha256=named["sha256"])
    assert "RETAINED_ANSWER of the reader" in json.dumps(read, ensure_ascii=False)
    assert (named["size"], named["sha256"]) == (handle["size"], handle["sha256"])


def test_a_route_that_holds_the_rows_keeps_their_bytes_and_a_tight_one_reads_one_line_per_work(tmp_path):
    """The full text is the producer's object exactly as before this form existed; the shorter form keeps the list's
    other facts in its object and the legend once, and is measured beside it at the install's count."""
    fixture.build(tmp_path)
    present = live_works_present(tmp_path)
    present["incomplete"] = True  # a stale or unreadable source stays said in both forms
    data = {"budget": {"remaining_usd": 10}, "live_works_present": present}
    part = runtime_host_section(copy.deepcopy(data), "\n\n## Task-tree coordination ledger\n\nledger")
    assert part.full_text == str(part) == ("## Runtime context\n\n" + json.dumps(data, ensure_ascii=False, separators=(
        ",", ":")) + "\n\n## Task-tree coordination ledger\n\nledger")
    body = part.operative_view.split("\n\n", 1)[1]
    projected, end = json.JSONDecoder().raw_decode(body)
    shown = projected["live_works_present"]
    assert shown["incomplete"] is True and shown["note"] == present["note"] and shown["works"] != present["works"]
    lines = body[end:].split("\n\n")[1].split("\n")
    assert lines[0].startswith(WORKS_LINES_HEADING) and body.count(WORKS_LINES_HEADING) == 1
    assert [_rebuild(line) for line in lines[1:]] == [_without_handle_path(work) for work in present["works"]]
    assert body.endswith("## Task-tree coordination ledger\n\nledger")
    rows = json.dumps(present["works"], ensure_ascii=False, separators=(",", ":"))
    print(f"\nK5 measure: {len(present['works'])} works; rows {view_tokens(rows)} est. tokens, "
          f"lines with legend {view_tokens(body[end:].split(chr(10) * 2)[1])} est. tokens")
    assert view_tokens("\n".join(lines)) < view_tokens(rows)  # the floor takes only a shorter form


def _head(root, *, shorter=False, exclude="observer"):
    """The head's Runtime section as a context build sends it, in either form."""
    part = runtime_host_section({"live_works_present": live_works_present(root, exclude=exclude)})
    return {"role": "user", "content": [{"type": "text", "text": "## Identity\n\n..."},
                                        {"type": "text", "text": part.operative_view if shorter else part.full_text}]}


def _ctx():
    return SimpleNamespace(task_id="observer", task_metadata={})


def _origin_free(monkeypatch):
    """Rows whose note says nothing beyond the head (no origin, lane wait or conversation lane)."""
    real = peer_roster._compact_root
    monkeypatch.setattr(peer_roster, "_compact_root", lambda *a, **kw: {
        key: value for key, value in real(*a, **kw).items() if key != "origin"})


def test_after_an_unchanged_head_the_note_says_nothing_twice(tmp_path, monkeypatch):
    """Both forms of the head: no note repeats a root it lists as the roster says it now, while an owner question,
    stop or hold is said again whole; with nothing to add for any root no note is appended at all."""
    _origin_free(monkeypatch)
    fixture.build(tmp_path)
    for shorter in (False, True):
        messages = [_head(tmp_path, shorter=shorter)]
        assert peer_roster.maybe_append_roster_note(_ctx(), messages, tmp_path)
        note = messages[-1]["content"]
        assert "beside the list of running works in this context's Runtime context" in note
        for work in live_works_present(tmp_path, exclude="observer")["works"]:
            owner = work["task_id"][3:6] in ("pau", "que", "hel")
            assert (f"- {work['task_id']} ·" in note) is owner, work["task_id"]
            assert (work["focus"]["text"] in note) is owner
    fixture.build(tmp_path, kinds=("focused",) * 5)
    messages = [_head(tmp_path)]
    assert not peer_roster.maybe_append_roster_note(_ctx(), messages, tmp_path) and len(messages) == 1


def test_after_a_change_the_note_delivers_exactly_the_change(tmp_path, monkeypatch):
    from ouroboros.task_results import write_task_result

    _origin_free(monkeypatch)
    fixture.build(tmp_path, kinds=("focused",) * 6)
    messages, ctx = [_head(tmp_path)], _ctx()
    write_task_result(tmp_path, "w03foc", "running", model_waits={"m": {
        "state": "waiting", "task_attempt": 1, "reason": "quota", "role": "main", "started_at": "2026-11-26T08:00:00Z"}})
    assert peer_roster.maybe_append_roster_note(ctx, messages, tmp_path)
    first = messages[-1]["content"]
    assert "- w03foc · " in first and "model access (reason=quota" in first
    assert all(f"w0{n}foc" not in first for n in (0, 1, 2, 4, 5))
    assert not peer_roster.maybe_append_roster_note(ctx, messages, tmp_path)  # nothing new: nothing appended
    write_task_result(tmp_path, "w05foc", "completed")
    assert peer_roster.maybe_append_roster_note(ctx, messages, tmp_path)
    second = messages[-1]["content"]
    assert "Changes since the preceding roster" in second and "- w05foc · " in second and "w03foc" not in second
    # The head leaves view (a renewal rebuilt it): the next note is the complete current snapshot again.
    messages[0] = {"role": "user", "content": "A rebuilt head without the list."}
    write_task_result(tmp_path, "w01foc", "running", title="Renamed work")
    assert peer_roster.maybe_append_roster_note(ctx, messages, tmp_path)
    assert messages[-1]["content"] == peer_roster.render_roster_note(peer_roster.independent_roots(tmp_path),
                                                                     exclude="observer")


def test_a_head_listed_root_keeps_what_only_the_note_says(tmp_path):
    """Origin and the conversation lane are the note's own facts: a carried root's line holds them and nothing else;
    a root the head listed and the roster no longer does is named, never read as finished."""
    from ouroboros.task_results import write_task_result
    from ouroboros.utils import utc_now_iso
    from tests._shared import write_queue_document

    fixture.build(tmp_path, kinds=("focused",) * 3)
    messages = [_head(tmp_path)]
    snapshot = json.loads((tmp_path / "state/queue_snapshot.json").read_text(encoding="utf-8"))
    snapshot["running"][0]["task"]["metadata"] = {"owner_ingress": True}
    snapshot["running"] = snapshot["running"][:2]
    write_queue_document(tmp_path / "state/queue_snapshot.json", {**snapshot, "ts": utc_now_iso()})
    write_task_result(tmp_path, "w02foc", "completed")
    assert peer_roster.maybe_append_roster_note(_ctx(), messages, tmp_path)
    note = messages[-1]["content"]
    [carried] = [line for line in note.splitlines() if line.startswith("- w00foc")]
    assert carried.startswith("- w00foc · origin=") and "owner_ingress" in carried and "Checking" not in carried
    assert "No longer in the displayed roster (not proof of completion): w02foc" in note
    assert "Checking" not in note
