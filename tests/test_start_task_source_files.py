"""Separate work's files follow the physical owner row, and its start result reports what admission recorded.

``start_task`` settles the files of the message the work answers before it sends the start (``_source_files``) and
afterwards tells the model what became of each (``_files_note``). Selection is by physical row: a known row key
decides, a legacy ref needs the one held row it identifies, and equal room, client id, time and words never do. The
per-file sentence reads the token-matched admission (``routing_wait.admitted_files``), calls a file pending while none
exists, and never reports the disk as it stood before the start ran. Everything here is synthetic.
"""

from __future__ import annotations

import types

import pytest

TS = "2026-11-20T10:00:00+00:00"
FIELDS = ("chat_id", "client_message_id", "ts", "text_sha256")


def _file(tmp_path, name, data):
    """A saved upload of the store: ``(its path, the ref the owner door records for it)``."""
    from ouroboros.chat_uploads import store_upload

    return store_upload(data, name, data_dir=tmp_path)


def _row(client_id, files, *, text="Check the totals"):
    """An inbound owner row. The same client id, time and words with other files is another physical row."""
    return {"direction": "in", "chat_id": 1, "ts": TS, "client_message_id": client_id, "text": text,
            "attachments": files}


def _ref(row):
    from ouroboros.owner_relay import owner_row_source

    return owner_row_source(row)[0]


def _ctx(tmp_path, metadata=None, held=()):
    from ouroboros.owner_relay import owner_row_address

    return types.SimpleNamespace(drive_root=tmp_path, task_metadata=dict(metadata or {}),
                                 _held_owner_messages={owner_row_address(row): row for row in held})


def _pair(tmp_path):
    """A and B: one room, client id, time and words, different files: two physical rows."""
    a_path, a_file = _file(tmp_path, "a.csv", b"a,1\n")
    b_path, b_file = _file(tmp_path, "b.csv", b"b,2\n")
    return (a_path, a_file, _row("dup", [a_file])), (b_path, b_file, _row("dup", [b_file]))


# --- a selected row is this turn's own message only by its physical row ---------------------------------------


def test_two_rows_sharing_every_field_are_told_apart_by_their_row_key(tmp_path):
    from ouroboros.tools.control_routing import _is_turn_message

    (_, _, a), (_, _, b) = _pair(tmp_path)
    ref_a, ref_b = _ref(a), _ref(b)
    assert {key: ref_a[key] for key in FIELDS} == {key: ref_b[key] for key in FIELDS}
    assert ref_a["row_sha256"] != ref_b["row_sha256"]
    ctx = _ctx(tmp_path, held=(a, b))
    assert _is_turn_message(ctx, ref_a, ref_a) and _is_turn_message(ctx, ref_b, ref_b)
    assert not _is_turn_message(ctx, ref_a, ref_b) and not _is_turn_message(ctx, ref_b, ref_a)


def test_a_legacy_turn_ref_without_a_row_key_proves_nothing_about_the_selected_row(tmp_path):
    from ouroboros.tools.control_routing import _is_turn_message

    (_, _, a), (_, _, b) = _pair(tmp_path)
    ref_a, ref_b = _ref(a), _ref(b)
    legacy = {key: ref_a[key] for key in FIELDS}  # an older ref: no row key
    both = _ctx(tmp_path, held=(a, b))  # its fields name two held rows: nothing is proven for either
    assert not _is_turn_message(both, legacy, ref_a) and not _is_turn_message(both, legacy, ref_b)
    only_a = _ctx(tmp_path, held=(a,))
    assert not _is_turn_message(only_a, legacy, ref_a), "one held row is not the ingress row key"
    assert not _is_turn_message(_ctx(tmp_path), legacy, ref_a), "nothing held, nothing proven"


def test_another_message_is_never_this_turns_message(tmp_path):
    from ouroboros.tools.control_routing import _is_turn_message

    (_, _, a), _b = _pair(tmp_path)
    _, other_file = _file(tmp_path, "c.csv", b"c,3\n")
    other = _row("owner-other", [other_file], text="A later message")
    assert not _is_turn_message(_ctx(tmp_path, held=(a, other)), _ref(a), _ref(other))
    assert not _is_turn_message(_ctx(tmp_path), "no ref", _ref(a))


def test_a_selection_keeps_the_turns_fresh_files_only_for_its_own_row(tmp_path):
    """B shares A's room, client id, time and words. Selecting B gives the work B's files and names the turn's own
    as not given; selecting A, the message the turn answers, keeps the turn's fresh uploads."""
    from ouroboros.tools.control_routing import _source_files

    (a_path, _a_file, a), (b_path, _b_file, b) = _pair(tmp_path)
    fresh = [{"path": str(a_path), "label": "a.csv"}]
    ctx = _ctx(tmp_path, {"origin_message_ref": _ref(a), "chat_attachment_uploads": fresh}, held=(a, b))
    selected_b = [{"path": str(b_path), "label": "b.csv", "inherited": True}]  # what the selection of B stages
    evt = {"source_ref": _ref(b), "attachment_uploads": list(selected_b)}
    assert _source_files(ctx, evt, selected=True) == (["a.csv"], False)
    assert evt["attachment_uploads"] == selected_b
    evt = {"source_ref": _ref(a), "attachment_uploads": [{"path": str(a_path), "label": "a.csv", "inherited": True}]}
    assert _source_files(ctx, evt, selected=True) == ([], False)
    assert evt["attachment_uploads"] == fresh


def test_an_unproven_legacy_turn_ref_displaces_the_turns_own_files(tmp_path):
    from ouroboros.tools.control_routing import _source_files

    (a_path, _a_file, a), (_, _, b) = _pair(tmp_path)
    legacy = {key: _ref(a)[key] for key in FIELDS}
    fresh = [{"path": str(a_path), "label": "a.csv"}]
    ctx = _ctx(tmp_path, {"origin_message_ref": legacy, "chat_attachment_uploads": fresh}, held=(a, b))
    selected = [{"path": str(a_path), "label": "a.csv", "inherited": True}]
    evt = {"source_ref": _ref(a), "attachment_uploads": list(selected)}
    assert _source_files(ctx, evt, selected=True) == (["a.csv"], False)
    assert evt["attachment_uploads"] == selected


def test_without_a_selection_the_turns_fresh_files_stay_as_sent(tmp_path):
    from ouroboros.tools.control_routing import _source_files

    (a_path, _a_file, a), _b = _pair(tmp_path)
    fresh = [{"path": str(a_path), "label": "a.csv"}]
    ctx = _ctx(tmp_path, {"origin_message_ref": _ref(a), "chat_attachment_uploads": fresh}, held=(a,))
    evt = {"source_ref": _ref(a), "attachment_uploads": list(fresh)}
    assert _source_files(ctx, evt, selected=False) == ([], False)
    assert evt["attachment_uploads"] == fresh


# --- the files of an origin row whose snapshot was not recorded --------------------------------------------------


def test_a_recorded_snapshot_answers_for_its_row_even_when_it_names_no_file(tmp_path):
    from ouroboros.tools.control_routing import _origin_file_refs

    (_, a_file, a), _b = _pair(tmp_path)
    empty = _ctx(tmp_path, {"origin_message_snapshot": {"projectable": True}}, held=(a,))
    assert _origin_file_refs(empty, _ref(a)) == [], "recorded as no files: the held row is not consulted"
    named = _ctx(tmp_path, {"origin_message_snapshot": {"projectable": True, "attachments": [a_file]}})
    assert _origin_file_refs(named, _ref(a)) == [a_file]


@pytest.mark.parametrize("recorded", [None, {}], ids=["absent", "empty-dict"])
def test_a_start_without_a_snapshot_recovers_the_files_of_its_held_origin_row(tmp_path, recorded):
    """No writer keeps an empty snapshot, so ``{}`` is as absent as ``None``: the files are read from the one held
    row the origin ref identifies, never from its twin that shares the room, client id, time and words."""
    from ouroboros.tools.control_routing import _origin_file_refs

    (_, a_file, a), (_, _b_file, b) = _pair(tmp_path)
    metadata = {} if recorded is None else {"origin_message_snapshot": recorded}
    refs = _origin_file_refs(_ctx(tmp_path, metadata, held=(a, b)), _ref(a))
    assert [(ref["name"], ref["upload"]) for ref in refs] == [("a.csv", a_file["upload"])], "the held row of its key"


def test_without_a_held_row_the_one_canonical_row_it_identifies_supplies_the_files(tmp_path, monkeypatch):
    from ouroboros.tools.control_routing import _origin_file_refs

    (_, a_file, a), _b = _pair(tmp_path)
    asked = []
    monkeypatch.setattr("ouroboros.project_dialogue.resolve_owner_message_source",
                        lambda root, ref: asked.append(ref) or a)
    monkeypatch.setattr("ouroboros.owner_relay.quotable_owner_row", lambda row, root=None, **_kw: True)
    refs = _origin_file_refs(_ctx(tmp_path, {}), _ref(a))
    assert [ref["upload"] for ref in refs] == [a_file["upload"]] and asked == [_ref(a)]


@pytest.mark.parametrize("case", ["no_row", "not_quotable", "unreadable"])
def test_an_origin_that_cannot_be_resolved_leaves_its_files_unknown_not_empty(tmp_path, monkeypatch, case):
    from ouroboros.tools.control_routing import _origin_file_refs

    (_, _a_file, a), _b = _pair(tmp_path)

    def resolve(root, ref):
        if case == "unreadable":
            raise OSError("the store is not readable")
        return None if case == "no_row" else a

    monkeypatch.setattr("ouroboros.project_dialogue.resolve_owner_message_source", resolve)
    monkeypatch.setattr("ouroboros.owner_relay.quotable_owner_row", lambda row, root=None, **_kw: case != "not_quotable")
    assert _origin_file_refs(_ctx(tmp_path, {}), _ref(a)) is None


def test_equal_words_in_another_message_never_stand_in_for_the_origin_row(tmp_path, monkeypatch):
    from ouroboros.tools.control_routing import _origin_file_refs

    (_, _a_file, a), _b = _pair(tmp_path)
    _, other_file = _file(tmp_path, "c.csv", b"c,3\n")
    twin = _row("owner-twin", [other_file], text=a["text"])  # the same words, another client id
    monkeypatch.setattr("ouroboros.project_dialogue.resolve_owner_message_source", lambda root, ref: None)
    assert _origin_file_refs(_ctx(tmp_path, {}, held=(twin,)), _ref(a)) is None


def test_work_started_without_a_selection_receives_the_files_of_a_legacy_origin_row(tmp_path):
    from pathlib import Path

    from ouroboros.tools.control_routing import _source_files

    (a_path, _a_file, a), _b = _pair(tmp_path)
    ctx = _ctx(tmp_path, {"origin_message_ref": _ref(a)}, held=(a,))  # no snapshot, no fresh uploads
    evt = {"source_ref": _ref(a), "attachment_uploads": []}
    assert _source_files(ctx, evt, selected=False) == ([], False)
    [spec] = evt["attachment_uploads"]
    assert (spec["label"], spec["inherited"], spec["path_only"]) == ("a.csv", True, True)
    assert Path(spec["path"]).resolve() == a_path.resolve()


def test_an_unresolvable_legacy_origin_starts_without_files_and_says_the_list_is_unknown(tmp_path, monkeypatch):
    from ouroboros.tools.control_routing import _source_files

    (_, _a_file, a), _b = _pair(tmp_path)
    monkeypatch.setattr("ouroboros.project_dialogue.resolve_owner_message_source", lambda root, ref: None)
    evt = {"source_ref": _ref(a), "attachment_uploads": []}
    assert _source_files(_ctx(tmp_path, {"origin_message_ref": _ref(a)}), evt, selected=False) == ([], True)
    assert evt["attachment_uploads"] == []


# --- the sentences of the start's result ----------------------------------------------------------------------------


def _note(tmp_path, specs, receipt, files=([], False)):
    from ouroboros.tools.control_routing import _files_note

    return _files_note(types.SimpleNamespace(drive_root=tmp_path), {"task_id": "t1", "attachment_uploads": specs},
                       receipt, files)


def test_a_start_not_confirmed_yet_calls_its_files_pending_and_never_rewrites_a_name(tmp_path):
    notes = tmp_path / "notes.txt"
    notes.write_text("notes\n", encoding="utf-8")
    specs = [{"path": str(notes), "label": "report staged at start.txt"},
             {"path": "", "label": "elsewhere.txt", "unavailable": "outside_store"},
             {"path": "", "label": "lost.txt"}]
    assert _note(tmp_path, specs, {"status": "unconfirmed", "reason": "confirmation_timeout"}) == (
        " Its source's files: report staged at start.txt (pending with its admission); "
        "elsewhere.txt (not staged: outside_store); lost.txt (not staged: invalid_path).")


def test_a_scheduled_start_with_no_recorded_outcome_for_its_files_says_so(tmp_path):
    assert _note(tmp_path, [{"path": "x", "label": "a.csv"}], {"status": "scheduled"}) == (
        " Its source's files: a.csv (no admission outcome recorded).")


def test_files_beyond_the_listed_rows_are_counted_when_the_complete_manifest_is_unreadable(tmp_path):
    rows = [{"ordinal": index, "status": "staged", "reason": "", "label": f"f{index}.csv"} for index in range(2)]
    specs = [{"path": "x", "label": f"f{index}.csv"} for index in range(5)]
    assert _note(tmp_path, specs, {"status": "scheduled", "attachment_manifest": rows, "attachment_declared": 5}) == (
        " Its source's files: f0.csv (staged at start); f1.csv (staged at start); "
        "3 more are in the new work's own input manifest.")


def test_an_unknown_file_list_and_the_displaced_own_files_each_get_a_sentence_of_their_own(tmp_path):
    from ouroboros.owner_relay import NOT_GIVEN

    assert _note(tmp_path, [], {"status": "scheduled"}, files=(["own.csv", "own2.csv"], True)) == (
        " Its source's file list is unknown (no snapshot of it was recorded and its row cannot be read): "
        "no files were given."
        f" This turn's own files are {NOT_GIVEN} (own.csv; own2.csv): its source is the selected message.")


def test_a_private_start_or_a_place_start_says_nothing_about_files(tmp_path):
    assert _note(tmp_path, [{"path": "x", "label": "a.csv"}], {"status": "scheduled"}, files=None) == ""


# --- the admission receipt carries what it recorded -----------------------------------------------------------


def test_a_token_matched_admission_carries_its_recorded_files_and_a_stale_token_none(tmp_path):
    from ouroboros.routing_wait import wait_for_promotion_admission
    from ouroboros.task_results import STATUS_SCHEDULED, write_task_result

    token = "a" * 32
    write_task_result(tmp_path, "deadbeef", STATUS_SCHEDULED, promotion_admission={"status": "scheduled",
                                                                                   "routing_token": token},
                      attachment_manifest=[
                          {"ordinal": 0, "status": "staged", "reason": "", "label": "a.csv", "abs_path": "/host/a.csv",
                           "relpath": "attachments/a.csv", "size": 3, "sha256": "0" * 64},
                          {"ordinal": 1, "status": "rejected", "reason": "copy_failed", "label": "b.csv"}])
    receipt = wait_for_promotion_admission(tmp_path, "deadbeef", token, timeout_sec=0.0)
    assert receipt["status"] == "scheduled" and receipt["attachment_declared"] == 2
    assert receipt["attachment_manifest"] == [
        {"ordinal": 0, "status": "staged", "reason": "", "label": "a.csv"},
        {"ordinal": 1, "status": "rejected", "reason": "copy_failed", "label": "b.csv"}], "outcomes only, no path"
    stale = wait_for_promotion_admission(tmp_path, "deadbeef", "b" * 32, timeout_sec=0.0)
    assert stale["status"] == "unconfirmed" and "attachment_manifest" not in stale


@pytest.mark.parametrize("stored", ["not a list", 7, {"label": "x"}, [None, "x", 3]])
def test_a_malformed_stored_manifest_yields_no_rows_and_never_raises(stored):
    """It runs inside the admission wait: a bad stored shape reads as no recorded files, never as a failed wait."""
    from ouroboros.routing_wait import admitted_files

    empty = {"attachment_manifest": [], "attachment_declared": 0}
    assert admitted_files({"attachment_manifest": stored}) == empty
    assert admitted_files({"task_contract": {"attachment_manifest": stored, "attachment_manifest_ref": "x"}}) == empty
    assert admitted_files({"task_contract": "x"}) == empty
    assert admitted_files({"task_contract": {"attachment_manifest_ref": {"count": True}}})["attachment_declared"] == 0, (
        "a count that is not an integer declares nothing")


def _declared(count):
    return [{"ordinal": index, "status": "rejected" if index % 7 == 0 else "staged",
             "reason": "copy_failed" if index % 7 == 0 else "", "label": f"file-{index}.txt",
             **({} if index % 7 == 0 else {"root": "artifact_store", "relpath": f"attachments/file-{index}.txt",
                                           "size": 3, "sha256": f"{index:064x}"})} for index in range(count)]


def test_a_start_that_declared_more_files_than_its_preview_is_completed_from_its_descriptor(tmp_path):
    from ouroboros.artifacts import attachment_manifest_projection
    from ouroboros.routing_wait import admitted_files, complete_admitted_files

    task_id = "0123456789abcdef"
    projection = attachment_manifest_projection(tmp_path, task_id, _declared(30))
    assert len(projection["attachment_manifest"]) == 25 and projection["attachment_manifest_ref"]["count"] == 30
    stored = {"attachment_manifest": projection["attachment_manifest"],
              "task_contract": {"attachment_manifest_ref": projection["attachment_manifest_ref"]}}
    receipt = admitted_files(stored)
    assert (len(receipt["attachment_manifest"]), receipt["attachment_declared"]) == (25, 30)
    complete = complete_admitted_files(tmp_path, task_id, receipt)
    assert [row["label"] for row in complete] == [f"file-{index}.txt" for index in range(30)]
    assert [row["status"] for row in complete[:8]] == ["rejected"] + ["staged"] * 6 + ["rejected"]


def test_the_descriptor_is_read_only_when_the_start_declared_more_than_it_lists(tmp_path, monkeypatch):
    from ouroboros import artifacts
    from ouroboros.routing_wait import complete_admitted_files

    reads: list = []
    monkeypatch.setattr(artifacts, "resolve_attachment_manifest", lambda *args, **_kw: reads.append(args) or [])
    rows = [{"ordinal": 0, "status": "staged", "reason": "", "label": "a.csv"}]
    whole = {"attachment_manifest": rows, "attachment_declared": 1, "attachment_manifest_ref": {"count": 1}}
    assert complete_admitted_files(tmp_path, "t1", whole) == rows and reads == []
    assert complete_admitted_files(tmp_path, "t1", {"attachment_manifest": rows, "attachment_declared": 3}) == rows
    textual_count = {"attachment_manifest": rows, "attachment_declared": "3", "attachment_manifest_ref": {"count": 3}}
    assert complete_admitted_files(tmp_path, "t1", textual_count) == rows
    assert reads == [], "no descriptor, or no declared count to compare: nothing to read"


def test_an_unreadable_descriptor_leaves_the_listed_rows(tmp_path, monkeypatch):
    from ouroboros import artifacts
    from ouroboros.routing_wait import complete_admitted_files

    def damaged(*_args, **_kwargs):
        raise ValueError("attachment manifest failed digest verification")

    monkeypatch.setattr(artifacts, "resolve_attachment_manifest", damaged)
    rows = [{"ordinal": 0, "status": "staged", "reason": "", "label": "a.csv"}]
    receipt = {"attachment_manifest": rows, "attachment_declared": 9, "attachment_manifest_ref": {"count": 9}}
    assert complete_admitted_files(tmp_path, "t1", receipt) == rows


# --- the start tool, end to end over the durable receipt store -----------------------------------------------------


def _start(tmp_path, monkeypatch, metadata, manifest=None):
    """One real ``start_task``. ``manifest``: the files the supervisor's admission records, written through the real
    store as it writes them; ``None``: nothing answers, as before a start is confirmed."""
    from ouroboros.task_results import STATUS_SCHEDULED, write_task_result
    from ouroboros.tools import control_events
    from ouroboros.tools.control import _start_task

    monkeypatch.setattr(control_events, "_PROMOTE_CONFIRM_TIMEOUT_SEC", 2.0)
    events: list = []

    def admit(evt):
        events.append(evt)
        write_task_result(tmp_path, evt["task_id"], STATUS_SCHEDULED, attachment_manifest=manifest,
                          promotion_admission={"status": "scheduled", "routing_token": evt["routing_token"]})

    ctx = types.SimpleNamespace(
        pending_events=events, event_queue=types.SimpleNamespace(put_nowait=admit) if manifest is not None else None,
        current_chat_id=1, drive_root=tmp_path, task_metadata=dict(metadata))
    return _start_task(ctx, "Check these inputs", title="Check inputs", workspace="none"), events


def _uploads(tmp_path):
    table, notes = tmp_path / "table.csv", tmp_path / "report staged at start.txt"
    table.write_text("a,1\n", encoding="utf-8")
    notes.write_text("notes\n", encoding="utf-8")
    return {"chat_attachment_uploads": [{"path": str(table), "label": "table.csv"},
                                        {"path": str(notes), "label": "report staged at start.txt"}]}


def test_start_task_tells_the_model_what_admission_recorded_for_each_file(tmp_path, monkeypatch):
    manifest = [{"ordinal": 0, "status": "rejected", "reason": "copy_failed", "label": "table.csv"},
                {"ordinal": 1, "status": "staged", "reason": "", "label": "report staged at start.txt"}]
    out, events = _start(tmp_path, monkeypatch, _uploads(tmp_path), manifest)
    assert out.startswith("OK: task"), out
    assert (" Its source's files: table.csv (not staged: copy_failed); "
            "report staged at start.txt (staged at start).") in out, out
    assert "table.csv (staged at start)" not in out
    assert [spec["label"] for spec in events[0]["attachment_uploads"]] == ["table.csv", "report staged at start.txt"]


def test_start_task_calls_its_files_pending_until_admission_answers(tmp_path, monkeypatch):
    out, _events = _start(tmp_path, monkeypatch, _uploads(tmp_path))
    assert out.startswith("Not confirmed yet:"), out
    assert ("Its source's files: table.csv (pending with its admission); "
            "report staged at start.txt (pending with its admission).") in out, out
    assert "(staged at start)" not in out


# --- fresh and historical files at admission -------------------------------------------------------------------------


def _stage(tmp_path, monkeypatch, uploads):
    import supervisor.workers as workers

    monkeypatch.setattr(workers, "DRIVE_ROOT", tmp_path)
    task: dict = {}
    manifest, refusal = workers._stage_promoted_initial_attachments({"attachment_uploads": uploads}, task, "stagefiles01")
    return manifest, refusal, task


def test_a_turns_fresh_files_stage_what_they_can_and_refuse_the_start_only_when_none_can(tmp_path, monkeypatch):
    good = tmp_path / "good.txt"
    good.write_text("payload", encoding="utf-8")
    gone = {"path": str(tmp_path / "gone.txt"), "label": "gone"}
    manifest, refusal, task = _stage(tmp_path, monkeypatch, [{"path": str(good), "label": "good"}, gone])
    assert refusal is None and [row["status"] for row in manifest] == ["staged", "rejected"]
    assert manifest[1]["reason"] == "source_missing" and "_historical_attachments" not in task
    _manifest, refusal, _task = _stage(tmp_path, monkeypatch, [gone])
    assert refusal["status"] == "needs_manual_target" and refusal["reason"] == "attachment_admission_rejected"


def test_a_held_rows_files_never_refuse_the_start_even_when_every_one_is_missing(tmp_path, monkeypatch):
    good = tmp_path / "good.txt"
    good.write_text("payload", encoding="utf-8")
    gone = {"path": str(tmp_path / "gone.txt"), "label": "gone", "inherited": True}
    manifest, refusal, task = _stage(tmp_path, monkeypatch, [gone])
    assert refusal is None and [(row["status"], row["reason"]) for row in manifest] == [("rejected", "source_missing")]
    assert task["_historical_attachments"] is True
    manifest, refusal, _task = _stage(tmp_path, monkeypatch, [{"path": str(good), "label": "good", "inherited": True}, gone])
    assert refusal is None and [row["status"] for row in manifest] == ["staged", "rejected"]
