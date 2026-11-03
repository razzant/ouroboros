"""The plan record (``ouroboros/plan_record.py``): revisions, the reply parser, the
objection fold, who-saw-what, the gate projection, the acceptance view and the
read-only legacy import. Every rule is pinned in both directions."""
from __future__ import annotations

import json

import pytest

from ouroboros import plan_record as pr
from tests._store_shared import read_result, seed_result

SEAT_A, SEAT_B, CRITIC = "triad_a", "triad_b", "critic_x"


def _participant(seat, *, owner_marked=True, reply=None, status=None, dispatched=True, model="m/x"):
    row = {"seat_id": seat, "model": model, "route": "api_chat", "owner_marked": owner_marked,
           "dispatched": dispatched, "status": status or ("answered" if reply is not None else "pending")}
    if reply is not None:
        row["reply"] = {"prose": "", "objections": [], "withdrawn": [], **reply}
    return row


def _record(*asks, revisions=1):
    record = pr.empty_record()
    for n in range(1, revisions + 1):
        record["revisions"].append({"n": n, "ts": f"2026-10-15T00:0{n}:00+00:00", "goal": "ship", "plan": f"plan v{n}",
                                    "done_when": [{"id": "c1", "claim": f"claim v{n}", "support": "run it"}],
                                    "affected_paths": [], "body": False, "content_sha256": f"{n:064d}"})
    for index, participants in enumerate(asks, start=1):
        record["asks"].append({"ask_id": f"a{index}", "ts": f"2026-10-15T01:0{index}:00+00:00",
                               "revision": min(index, revisions), "message": "", "participants": list(participants)})
    return record


def _gate(record, **over):
    kwargs = {"enforcement": "blocking", "cyber_pro": False, "hurry": False}
    kwargs.update(over)
    return pr.gate_projection(record, **kwargs)


# ---------------------------------------------------------------- revisions

def test_compose_revision_carries_omitted_fields_over_and_normalizes_done_when():
    first = pr.compose_revision(None, goal="ship", plan="do it", done_when=[{"claim": "five slides", "support": "count"}],
                                affected_paths=[" a.py ", ""])
    assert first["affected_paths"] == ["a.py"]
    assert first["done_when"] == [{"id": "claim_1", "claim": "five slides", "support": "count"}]
    second = pr.compose_revision(first, plan="do it better")
    assert second["goal"] == "ship" and second["done_when"] == first["done_when"] and second["affected_paths"] == ["a.py"]
    assert second["content_sha256"] != first["content_sha256"]
    # Identical content composes the identical identity: no revision would be created.
    assert pr.compose_revision(first, plan="do it")["content_sha256"] == first["content_sha256"]


def test_revision_delta_names_reworded_claims_and_diffs_the_prose():
    prev = {"n": 1, "goal": "g", "plan": "a\nb\n", "done_when": [{"id": "c1", "claim": "old", "support": ""}], "affected_paths": ["x"]}
    cur = {"n": 2, "goal": "g", "plan": "a\nc\n", "done_when": [{"id": "c1", "claim": "new", "support": ""}, {"id": "c2", "claim": "more", "support": ""}],
           "affected_paths": []}
    delta = pr.revision_delta(prev, cur)
    assert delta["first"] is False and delta["goal_changed"] is False and delta["plan_changed"] is True
    assert delta["done_when_reworded"][0]["before"]["claim"] == "old" and delta["done_when_reworded"][0]["after"]["claim"] == "new"
    assert [r["id"] for r in delta["done_when_added"]] == ["c2"] and delta["affected_paths_removed"] == ["x"]
    assert any(line.startswith("-b") for line in delta["plan_diff"]) and any(line.startswith("+c") for line in delta["plan_diff"])
    assert pr.revision_delta(None, cur) == {"first": True}


# ---------------------------------------------------------------- reply parser

def test_parse_reply_reads_the_one_object_in_prose_or_fence_and_dedups_ids():
    text = 'Thoughts first.\n```json\n{"prose": "fine", "objections": [{"id": "o1", "text": "a"}, {"id": "o1", "text": "b"}], "withdrawn": []}\n```'
    reply, error = pr.parse_reply(text)
    assert error == "" and reply["prose"] == "fine" and reply["objections"] == [{"id": "o1", "text": "b"}]
    bare, error = pr.parse_reply('{"prose": "ok"}')
    assert error == "" and bare == {"prose": "ok", "objections": [], "withdrawn": []}
    nested, error = pr.parse_reply('I say {"prose": "with {braces} inside \\"quoted\\"", "withdrawn": [{"id": "o2", "why": "done"}]} end')
    assert error == "" and nested["withdrawn"] == [{"id": "o2", "why": "done"}]


@pytest.mark.parametrize("text", ["just prose", "[1, 2]", '{"objections": "nope"}', '{"withdrawn": {"id": "o1", "why": "x"}}',
                                  '{"prose": 3}', '```json\n{"id": "c1", "claim": "a quoted done_when row"}\n```'])
def test_parse_reply_refuses_unreadable_or_malformed_replies_whole(text):
    reply, error = pr.parse_reply(text)
    assert reply is None and error


def test_a_quoted_object_in_a_preamble_is_not_the_reply_and_a_malformed_item_is_named_not_voided():
    """Scope R1 — C1 (Opus) + Fable 2: the reply is the first object carrying prose, objections or
    withdrawn; a preamble quoting a done_when row (the delivery renders them as fenced JSON) or any
    other object is skipped; a malformed ITEM is skipped and named while the valid ones act."""
    preamble = ('Re-reading the bar first:\n```json\n{"id": "c1", "claim": "five slides", "support": "count them"}\n```\n'
                'Now my reply:\n{"prose": "too thin", "objections": [{"id": "o1", "text": "no story"}], "withdrawn": []}')
    reply, error = pr.parse_reply(preamble)
    assert error == "" and reply["prose"] == "too thin" and reply["objections"] == [{"id": "o1", "text": "no story"}]
    braces, error = pr.parse_reply('a {weird} preamble, then {"prose": "still read"} and a tail')
    assert error == "" and braces == {"prose": "still read", "objections": [], "withdrawn": []}
    mixed, error = pr.parse_reply('{"prose": "", "objections": [{"id": "o 2", "text": "spaced id"}, {"text": "no id"}, "junk", '
                                  '{"id": "o3", "text": 7}], "withdrawn": [{"id": "bad id!", "why": "x"}, {"id": "o1", "why": "fixed"}]}')
    assert error == "" and mixed["objections"] == [{"id": "o_2", "text": "spaced id"}] and mixed["withdrawn"] == [{"id": "o1", "why": "fixed"}]
    assert [(i["key"], i["index"], i["id"], i["reason"]) for i in mixed["ignored"]] == [
        ("objections", 1, "", "no usable id"), ("objections", 2, "", "not an object"), ("objections", 3, "o3", "text must be text"),
        ("withdrawn", 0, "bad id!", "no usable id")]
    # Through the fold: the raiser's withdrawal beside a malformed item still releases the hold.
    held = _record([_participant(SEAT_A, reply={"objections": [{"id": "o1", "text": "low"}]})])
    assert _gate(held)["status"] == "held"
    withdrawn, _ = pr.parse_reply('{"withdrawn": [{"id": "o1", "why": "raised"}, {"id": "", "why": "void"}]}')
    released = _record(held["asks"][0]["participants"], [_participant(SEAT_A, reply=withdrawn)])
    assert _gate(released)["status"] == "clear" and pr.read_view(released)["asks"][1]["participants"][0]["ignored_items"] == 1


# ---------------------------------------------------------------- the fold

def test_objection_stands_until_its_raiser_withdraws_it():
    raised = _record([_participant(SEAT_A, reply={"objections": [{"id": "o1", "text": "the bar is too low"}]})])
    [objection] = pr.open_objections(raised)
    assert objection["seat_id"] == SEAT_A and objection["id"] == "o1" and objection["holds"] is True
    assert objection["not_adjudicated"] is False and objection["raised_in"] == "a1"
    withdrawn = _record(raised["asks"][0]["participants"],
                        [_participant(SEAT_A, reply={"withdrawn": [{"id": "o1", "why": "revision 2 raised the bar"}]})], revisions=2)
    assert pr.open_objections(withdrawn) == []
    [closed] = pr.fold_objections(withdrawn)
    assert closed["open"] is False and closed["withdrawn_in"] == "a2" and closed["why"] == "revision 2 raised the bar"


def test_another_seat_cannot_withdraw_it_and_omission_keeps_it_flagged():
    record = _record(
        [_participant(SEAT_A, reply={"objections": [{"id": "o1", "text": "x"}]}), _participant(SEAT_B, reply={"prose": "fine"})],
        [_participant(SEAT_B, reply={"withdrawn": [{"id": "o1", "why": "I disagree with A"}]}),
         _participant(SEAT_A, reply={"prose": "nothing about o1"})],
        revisions=2)
    [open_one] = pr.open_objections(record)
    assert open_one["seat_id"] == SEAT_A and open_one["not_adjudicated"] is True
    foreign = [o for o in pr.fold_objections(record) if o.get("unknown_withdrawal")]
    assert len(foreign) == 1 and foreign[0]["seat_id"] == SEAT_B
    # Restating it clears the flag; the text and the ask it answered move with it.
    restated = _record(*[a["participants"] for a in record["asks"]],
                       [_participant(SEAT_A, reply={"objections": [{"id": "o1", "text": "still: x"}]})], revisions=3)
    [again] = pr.open_objections(restated)
    assert again["not_adjudicated"] is False and again["text"] == "still: x" and again["ask_id"] == "a3" and again["raised_in"] == "a1"


def test_unparseable_reply_changes_nothing_and_mind_added_critics_never_hold():
    record = _record(
        [_participant(SEAT_A, reply={"objections": [{"id": "o1", "text": "x"}]}),
         _participant(CRITIC, owner_marked=False, reply={"objections": [{"id": "k1", "text": "critic says no"}]})],
        [_participant(SEAT_A, status="unparseable")])
    opened = {(o["seat_id"], o["id"]): o for o in pr.open_objections(record)}
    assert opened[(SEAT_A, "o1")]["holds"] is True and opened[(SEAT_A, "o1")]["not_adjudicated"] is False
    assert opened[(CRITIC, "k1")]["holds"] is False and opened[(CRITIC, "k1")]["open"] is True


def test_participant_views_track_last_seen_revision_and_last_answered_ask():
    record = _record(
        [_participant(SEAT_A, reply={"prose": "first"}), _participant(SEAT_B)],
        [_participant(SEAT_A), _participant(SEAT_B, dispatched=False, status="not_dispatched")],
        revisions=2)
    views = pr.participant_views(record)
    assert views[SEAT_A]["last_seen_revision"] == 2 and views[SEAT_A]["last_answered_ask"] == "a1" and views[SEAT_A]["latest_prose"] == "first"
    assert views[SEAT_B]["last_seen_revision"] == 1 and views[SEAT_B]["last_answered_ask"] == "" and views[SEAT_B]["status"] == "not_dispatched"
    assert [a["ask_id"] for a in pr.pending_asks(record)] == ["a1", "a2"]


# ---------------------------------------------------------------- a reading of the request (an ask before any plan)

def _reading(*participants, ask_id="a1"):
    return {"ask_id": ask_id, "ts": "2026-10-15T00:30:00+00:00", "revision": None, "message": "How do you read it?",
            "participants": list(participants)}


def test_a_reading_carries_revision_none_and_every_other_missing_revision_stays_invalid():
    """``revision: None`` (the key present) is the one form of an ask before any plan; a missing key,
    a bool (``True == 1``), zero, an absent number or a string still name a revision the record lacks."""
    reading = {**pr.empty_record(), "asks": [_reading(_participant(SEAT_A))]}
    assert pr.validate_record(reading)["asks"][0]["revision"] is None
    stored = _record(revisions=1)
    assert pr.validate_record({**stored, "asks": [_reading(ask_id="a2")]})["asks"][0]["revision"] is None
    assert pr.validate_record({**stored, "asks": [{"ask_id": "a1", "revision": 1, "participants": []}]})["asks"][0]["revision"] == 1
    for broken in ({"ask_id": "a1", "participants": []}, {"ask_id": "a1", "revision": True, "participants": []},
                   {"ask_id": "a1", "revision": 0, "participants": []}, {"ask_id": "a1", "revision": 2, "participants": []},
                   {"ask_id": "a1", "revision": "1", "participants": []}):
        with pytest.raises(pr.PlanRecordInvalid):
            pr.validate_record({**stored, "asks": [broken]})


def test_a_readings_reply_is_advice_that_moves_no_objection_in_any_mode():
    """A reading's reply raises, restates, withdraws and flags nothing — before a plan or settling
    late beside one — in every mode; the same reply to a stored revision acts as before."""
    objecting = _participant(SEAT_A, reply={"objections": [{"id": "o1", "text": "the owner asked for a story"}]})
    reading = {**pr.empty_record(), "asks": [_reading(objecting)]}
    assert pr.fold_objections(reading) == [] and pr.open_objections(reading) == []
    for mode in ({}, {"enforcement": "advisory"}, {"cyber_pro": True}):
        decision = _gate(reading, **mode)
        assert decision["allow"] is True and decision["holding"] == [] and decision["open_objections"] == []
    assert pr.read_view(reading)["asks"][0]["revision"] is None  # the reading itself stays visible
    # A standing objection on revision 1: a later reading's withdrawal, restatement or silence leaves it as it was.
    standing = _record([objecting])
    for reply in ({"withdrawn": [{"id": "o1", "why": "fine now"}]}, {"objections": [{"id": "o1", "text": "reworded"}]},
                  {"prose": "nothing about o1"}):
        late = {**standing, "asks": [*standing["asks"], _reading(_participant(SEAT_A, reply=reply), ask_id="a2")]}
        [objection] = pr.open_objections(late)
        assert (objection["holds"], objection["ask_id"], objection["text"], objection["not_adjudicated"]) == (
            True, "a1", "the owner asked for a story", False)
        assert _gate(late)["status"] == "held"
    # The other direction: the same objection stated about a stored revision holds, and its raiser's withdrawal there releases it.
    on_plan = _record([objecting])
    assert _gate(on_plan)["status"] == "held" and pr.open_objections(on_plan)[0]["revision"] == 1
    released = _record([objecting], [_participant(SEAT_A, reply={"withdrawn": [{"id": "o1", "why": "fine now"}]})], revisions=2)
    assert _gate(released)["status"] == "clear"


def test_a_reading_leaves_an_imported_legacy_objection_holding():
    legacy = pr.import_legacy_v2(_v2_state(closed=False, blocking_findings=("triad_old",)))
    withdrawal = {"withdrawn": [{"id": "triad_old:f1", "why": "fixed"}]}
    reading = {**pr.empty_record(), "asks": [_reading(_participant("triad_old", reply=withdrawal))]}
    decision = _gate(reading, legacy=legacy)
    assert decision["status"] == "held" and decision["holding"] == ["triad_old:triad_old:f1"]
    assert pr.open_objections(reading, legacy)[0]["not_adjudicated"] is False
    assert _gate(_record([_participant("triad_old", reply=withdrawal)]), legacy=legacy)["status"] == "clear"


# ---------------------------------------------------------------- the gate

def test_blocking_holds_only_on_an_open_objection_of_an_owner_marked_seat():
    held = _record([_participant(SEAT_A, reply={"objections": [{"id": "o1", "text": "x"}]})])
    decision = _gate(held)
    assert decision["status"] == "held" and decision["allow"] is False and decision["holding"] == [f"{SEAT_A}:o1"]
    critic_only = _record([_participant(CRITIC, owner_marked=False, reply={"objections": [{"id": "k1", "text": "x"}]})])
    assert _gate(critic_only)["status"] == "objections_open" and _gate(critic_only)["allow"] is True
    withdrawn = _record(held["asks"][0]["participants"], [_participant(SEAT_A, reply={"withdrawn": [{"id": "o1", "why": "ok"}]})])
    assert _gate(withdrawn)["status"] == "clear" and _gate(withdrawn)["allow"] is True


def test_silence_refusals_and_pending_asks_never_hold():
    silent = _record([_participant(SEAT_A), _participant(SEAT_B, dispatched=False, status="not_dispatched"),
                      _participant(CRITIC, owner_marked=False, status="failed", dispatched=True)])
    decision = _gate(silent)
    assert decision["allow"] is True and decision["status"] == "awaiting" and decision["review_only_awaited"] is True
    assert decision["pending_asks"] == ["a1"]
    nothing_sent = _record([_participant(SEAT_A, dispatched=False, status="not_dispatched")])
    assert _gate(nothing_sent)["status"] == "clear"


def test_advisory_cyber_and_hurry_disclose_instead_of_holding():
    held = _record([_participant(SEAT_A, reply={"objections": [{"id": "o1", "text": "x"}]})])
    advisory = _gate(held, enforcement="advisory")
    assert advisory["allow"] is True and advisory["status"] == "objections_open" and advisory["holding"] == [f"{SEAT_A}:o1"]
    cyber = _gate(held, cyber_pro=True)
    assert cyber["allow"] is True and cyber["decision_authority"] == "cyber_pro"
    hurry = _gate(held, hurry=True)
    assert hurry["allow"] is True and hurry["owner_hurry_local_advisory"] is True and hurry["configured_enforcement"] == "blocking"
    rail = _gate(held, hard_rail="round_limit")
    assert rail["allow"] is True and rail["status"] == "rail_degraded" and rail["reason"] == "round_limit"


def test_unasked_and_unroutable_plans_never_hold_without_an_objection():
    for enforcement in ("blocking", "advisory"):
        for record in (pr.empty_record(), _record(revisions=1),
                       _record([_participant(SEAT_A, dispatched=False, status="not_dispatched")])):
            decision = _gate(record, enforcement=enforcement)
            assert decision["allow"] is True
    reached = _gate(_record([_participant(SEAT_A)]))
    assert reached["status"] == "awaiting" and reached["allow"] is True


def test_a_task_without_a_plan_is_not_required_and_an_unreadable_record_is_engaged():
    assert _gate(pr.empty_record()) == {"required": False, "allow": True, "status": "not_required", "enforcement": "blocking",
                                        "open_objections": [], "holding": [], "pending_asks": [], "revision": 0}
    unreadable = _gate(pr.empty_record(), unreadable=True)
    assert unreadable["required"] is True and unreadable["allow"] is False and unreadable["status"] == "unreadable"
    assert _gate(pr.empty_record(), unreadable=True, enforcement="advisory")["allow"] is False
    assert _gate(pr.empty_record(), unreadable=True, cyber_pro=True)["allow"] is True


# ---------------------------------------------------------------- acceptance

def test_latest_revision_binds_always_and_the_exhibit_carries_every_revisions_text():
    record = _record([_participant(SEAT_A, reply={"objections": [{"id": "o1", "text": "the bar is low"}]})], revisions=2)
    claims, source, exhibit = pr.acceptance_view(record, [])
    assert source == "plan_review" and [c["claim"] for c in claims] == ["claim v2"] and claims[0]["id"] == "c1"
    assert exhibit["binding"] == "latest_revision:2"
    assert [r["done_when"][0]["claim"] for r in exhibit["revisions"]] == ["claim v1", "claim v2"]
    assert exhibit["revisions"][0]["seen_by_asks"] == ["a1"] and exhibit["revisions"][1]["seen_by_asks"] == []
    assert exhibit["open_objections"][0]["id"] == "o1"


def test_ingress_claims_win_and_adopted_promises_stay_visible():
    record = _record(revisions=1)
    ingress = [{"id": "task_1", "claim": "from the owner", "surface": "", "support": "", "priority": "must"}]
    claims, source, exhibit = pr.acceptance_view(record, ingress)
    assert claims == ingress and source == "ingress_contract" and exhibit["binding"] == "ingress_contract"
    assert exhibit["adopted_promises"] == [{"id": "c1", "claim": "claim v1", "support": "run it"}]
    assert pr.acceptance_view(pr.empty_record(), [])[0:2] == ([], "")


# ---------------------------------------------------------------- legacy import

def _v2_state(*, closed, blocking_findings=(), claims=("a", "b"), fingerprint="f" * 64):
    return {"schema_version": 2, "cycles_paid": 1, "current_attempt": {"fingerprint": fingerprint, "status": "open", "reason": ""},
            "waves": [{"request_fingerprint": fingerprint, "closed": closed, "aggregate": "GREEN" if closed else "REVISE_PLAN", "paid": True,
                       "spec": {"acceptance_claims": list(claims)},
                       "findings": [{"finding_id": f"{seat}:f1", "class": "blocking", "slot": seat, "summary": "wrong shape"} for seat in blocking_findings]
                       + [{"finding_id": "triad_z:n1", "class": "note", "slot": "triad_z", "summary": "a note"}]}]}


def test_legacy_open_blocking_findings_become_holding_objections_of_their_seats():
    legacy = pr.import_legacy_v2(_v2_state(closed=False, blocking_findings=("triad_old",)))
    assert legacy["engaged"] is True and legacy["done_when"] == [] and legacy["asks_dispatched"] == 1
    assert legacy["open_objections"] == [{"seat_id": "triad_old", "id": "triad_old:f1", "owner_marked": True, "text": "wrong shape",
                                          "fingerprint": "f" * 64}]
    decision = _gate(pr.empty_record(), legacy=legacy)
    assert decision["status"] == "held" and decision["source"] == "legacy_v2" and decision["open_objections"][0]["legacy"] is True
    assert _gate(pr.empty_record(), legacy=legacy, enforcement="advisory")["allow"] is True
    # The seat that raised it releases it through the new organ; nobody else can.
    released = _record([_participant("triad_old", reply={"withdrawn": [{"id": "triad_old:f1", "why": "fixed"}]})])
    assert _gate(released, legacy=legacy)["status"] == "clear"
    other = _record([_participant(SEAT_B, reply={"withdrawn": [{"id": "triad_old:f1", "why": "I say so"}]})])
    assert _gate(other, legacy=legacy)["status"] == "held"


def test_legacy_closed_wave_and_advisory_author_selection_bind_as_done_when():
    closed = pr.import_legacy_v2(_v2_state(closed=True))
    assert closed["open_objections"] == [] and closed["done_when_source"] == "legacy_closed_wave"
    claims, source, exhibit = pr.acceptance_view(pr.empty_record(), [], legacy=closed)
    assert [c["claim"] for c in claims] == ["a", "b"] and source == "plan_review" and exhibit["binding"] == "legacy_closed_wave"
    author = pr.import_legacy_v2(_v2_state(closed=False), author_plan={
        "spec": {"acceptance_claims": ["selected by the author"]}, "author_disposition": {"action": "finish", "enforcement": "advisory"}})
    assert author["done_when_source"] == "legacy_author_plan" and author["done_when"][0]["claim"] == "selected by the author"
    assert pr.import_legacy_v2({"schema_version": 2, "waves": [], "current_attempt": {}})["engaged"] is False
    assert pr.import_legacy_v2("garbage")["engaged"] is False
    malformed = pr.import_legacy_v2({"schema_version": 2, "current_attempt": {"fingerprint": "x", "status": "open", "reason": ""},
                                     "waves": [{"request_fingerprint": "x", "closed": False, "findings": ["not a dict", {"class": "blocking"}]}]})
    assert malformed["engaged"] is True and malformed["open_objections"] == []



def test_only_an_advisory_author_finish_binds_and_a_blocking_one_waits_for_its_wave():
    """The old organ honoured an author's ``finish`` selection only under Advisory; under
    Blocking the claims bind only once the wave the attempt names had closed."""
    plan = {"spec": {"acceptance_claims": ["selected by the author"]}}
    blocking = pr.import_legacy_v2(_v2_state(closed=False), author_plan={
        **plan, "author_disposition": {"action": "finish", "enforcement": "blocking"}})
    assert blocking["done_when"] == [] and blocking["done_when_source"] == ""
    advisory = pr.import_legacy_v2(_v2_state(closed=False), author_plan={
        **plan, "author_disposition": {"action": "finish", "enforcement": "advisory"}})
    assert advisory["done_when_source"] == "legacy_author_plan"
    stopped = pr.import_legacy_v2(_v2_state(closed=True), author_plan={
        **plan, "author_disposition": {"action": "stop", "enforcement": "advisory"}})
    assert stopped["done_when_source"] == "legacy_closed_wave" and [c["claim"] for c in stopped["done_when"]] == ["a", "b"]


def _v1_state(kind, claims=None):
    fp = "a" * 64
    state = {"schema_version": 1, "current_attempt": {}, "latest_review_fingerprint": "", "waves": []}
    if kind == "absent":
        return state
    state["current_attempt"] = {"fingerprint": fp, "status": "open", "reason": ""}
    state["latest_review_fingerprint"] = fp
    state["waves"] = [{"request_fingerprint": fp, "phase": "reviewed", "review_evidence_status": "integrated",
                       "review": {"aggregate_signal": "GREEN" if kind == "closed" else "REVIEW_REQUIRED", "closed": kind == "closed"},
                       **({"acceptance_claims": claims} if claims else {})}]
    return state


def test_a_v1_record_reads_as_kept_history_and_imports_read_only(tmp_path):
    """A schema-1 record loads wrapped (``legacy_v1``, never migrated); the importer reads it: an
    open v1 wave is engaged and binds nothing, a closed one binds its claims with legacy
    provenance, an absent one is not engaged."""
    from ouroboros.task_results import PLAN_REVIEW_STATE_KEY, load_plan_review_state

    seed_result(tmp_path, "v1", {"task_id": "v1", "status": "completed", PLAN_REVIEW_STATE_KEY: _v1_state("closed", ["old claim"])})
    state = load_plan_review_state(tmp_path, "v1")
    assert state["schema_version"] == 2 and state["legacy_v1"]["schema_version"] == 1 and state["waves"] == []
    closed = pr.import_legacy_v2(state)
    assert closed["engaged"] is True and closed["done_when_source"] == "legacy_closed_wave"
    claims, source, exhibit = pr.acceptance_view(pr.empty_record(), [], legacy=closed)
    assert [c["claim"] for c in claims] == ["old claim"] and source == "plan_review" and exhibit["binding"] == "legacy_closed_wave"
    seed_result(tmp_path, "v1-open", {"task_id": "v1-open", "status": "running", PLAN_REVIEW_STATE_KEY: _v1_state("open", ["unreviewed"])})
    opened = pr.import_legacy_v2(load_plan_review_state(tmp_path, "v1-open"))
    assert opened["engaged"] is True and opened["done_when"] == [] and opened["open_objections"] == []
    assert pr.acceptance_view(pr.empty_record(), [], legacy=opened)[2]["binding"] == "none"
    seed_result(tmp_path, "v1-absent", {"task_id": "v1-absent", "status": "running", PLAN_REVIEW_STATE_KEY: _v1_state("absent")})
    assert pr.import_legacy_v2(load_plan_review_state(tmp_path, "v1-absent"))["engaged"] is False


def test_legacy_in_flight_wave_is_one_pending_ask_that_discloses_and_never_holds():
    """An in-flight v2 wave at upgrade: its unanswered reviewers are ONE pending ask the final
    answer discloses (silence never holds); a blocking finding already landed still holds."""
    from ouroboros.owner_hurry import plan_review_disclosure

    state = _v2_state(closed=False)
    wave = state["waves"][0]
    wave.update(aggregate="DEGRADED", custody_pending=True,
                actors=[{"slot": "s1", "ok": False, "operation_state": "pending_dispatch", "late_result_pending": True}])
    legacy = pr.import_legacy_v2(state)
    assert legacy["pending_asks"] == ["legacy wave " + "f" * 12] and legacy["open_objections"] == []
    decision = _gate(pr.empty_record(), legacy=legacy)
    assert decision["status"] == "awaiting" and decision["allow"] is True and decision["pending_asks"] == legacy["pending_asks"]
    disclosed = plan_review_disclosure(decision)
    assert "legacy wave ffffffffffff" in disclosed and "had not arrived" in disclosed and "silence holds nothing" in disclosed
    # answered and settled: nothing pending, nothing disclosed
    wave.update(custody_pending=False, actors=[{"slot": "s1", "ok": True, "operation_state": "settled", "late_result_pending": False}])
    settled = pr.import_legacy_v2(state)
    assert settled["pending_asks"] == [] and _gate(pr.empty_record(), legacy=settled)["status"] == "clear"
    assert pr.import_legacy_v2(_v2_state(closed=True))["pending_asks"] == []
    # a landed blocking finding holds whether or not the rest of the wave is still owed
    held_state = _v2_state(closed=False, blocking_findings=("triad_old",))
    held_state["waves"][0]["custody_pending"] = True
    held = _gate(pr.empty_record(), legacy=pr.import_legacy_v2(held_state))
    assert held["status"] == "held" and held["pending_asks"] == ["legacy wave " + "f" * 12]

def test_legacy_objections_stand_when_the_author_moved_on_but_claims_do_not_follow():
    """A changed spec at the cap or an author plan moves the attempt's fingerprint to a wave that
    never ran: the latest wave's open blocking findings still hold (a Blocking hold is not dropped
    by deployment), while a closed wave the author left behind binds no claims."""
    moved = _v2_state(closed=False, blocking_findings=("triad_old",))
    moved["current_attempt"] = {"fingerprint": "e" * 64, "status": "cycles_exhausted", "reason": "spent"}
    legacy = pr.import_legacy_v2(moved)
    assert [o["seat_id"] for o in legacy["open_objections"]] == ["triad_old"] and legacy["open_objections"][0]["fingerprint"] == "f" * 64
    assert _gate(pr.empty_record(), legacy=legacy)["status"] == "held"
    left_behind = _v2_state(closed=True)
    left_behind["current_attempt"] = {"fingerprint": "e" * 64, "status": "open", "reason": "author_current_plan"}
    stale = pr.import_legacy_v2(left_behind)
    assert stale["open_objections"] == [] and stale["done_when"] == [] and stale["done_when_source"] == ""
    assert pr.acceptance_view(pr.empty_record(), [], legacy=stale)[2]["binding"] == "none"


# ---------------------------------------------------------------- durable index

def test_durable_record_round_trips_revisions_asks_and_replies(tmp_path):
    root, task = tmp_path, "task-1"
    seed_result(root, task, {"task_id": task, "status": "running"})
    fields = pr.compose_revision(None, goal="ship", plan="first", done_when=["five slides"], affected_paths=[])
    record, revision, created = pr.append_revision(root, task, fields, body=False, body_note="not body")
    assert created is True and revision["n"] == 1 and revision["text_ref"]["kind"] == "task_source"
    again, same, created_again = pr.append_revision(root, task, fields, body=False)
    assert created_again is False and same["n"] == 1 and len(again["revisions"]) == 1
    record = pr.append_ask(root, task, {"ask_id": pr.next_ask_id(record), "ts": "t", "revision": 1, "message": "look",
                                        "participants": [_participant(SEAT_A)]})
    assert record["asks"][0]["ask_id"] == "a1" and record["asks"][0]["participants"][0]["status"] == "pending"
    record = pr.update_participant(root, task, ask_id="a1", seat_id=SEAT_A, facts={
        "status": "answered", "reply": {"prose": "ok", "objections": [{"id": "o1", "text": "x"}], "withdrawn": []}})
    assert pr.open_objections(record)[0]["id"] == "o1"
    assert pr.load_plan_record(root, task) == record
    stored = read_result(root, task)[pr.PLAN_RECORD_KEY]
    assert stored["schema_version"] == 3
    with pytest.raises(pr.PlanRecordInvalid):
        pr.append_ask(root, task, {"ask_id": "a1", "ts": "t", "revision": 1, "participants": []})
    with pytest.raises(pr.PlanRecordInvalid):
        pr.update_participant(root, task, ask_id="a1", seat_id="nobody", facts={"status": "failed"})


def test_absent_record_reads_empty_and_malformed_records_refuse(tmp_path):
    assert pr.load_plan_record(tmp_path, "absent") == pr.empty_record()
    seed_result(tmp_path, "bad", {"task_id": "bad", pr.PLAN_RECORD_KEY: {"schema_version": 3, "revisions": "x", "asks": []}})
    with pytest.raises(pr.PlanRecordInvalid):
        pr.load_plan_record(tmp_path, "bad")
    seed_result(tmp_path, "v2", {"task_id": "v2", "plan_review_state": {"schema_version": 2}})
    assert pr.load_plan_record(tmp_path, "v2") == pr.empty_record()  # a v2 record is not a v3 record
    for broken in ({"schema_version": 3, "revisions": [{"n": 2, "done_when": [], "affected_paths": []}], "asks": []},
                   {"schema_version": 3, "revisions": [], "asks": [{"ask_id": "a1", "revision": 1, "participants": []}]},
                   {"schema_version": 3, "revisions": [{"n": 1, "done_when": [], "affected_paths": []}], "asks": [
                       {"ask_id": "a1", "revision": 1, "participants": [{"seat_id": "s", "status": "weird"}]}]}):
        with pytest.raises(pr.PlanRecordInvalid):
            pr.validate_record(broken)


def test_oversize_index_moves_the_oldest_prose_out_and_keeps_the_latest_whole(tmp_path, monkeypatch):
    root, task = tmp_path, "task-fit"
    seed_result(root, task, {"task_id": task, "status": "running"})
    monkeypatch.setattr(pr, "_PLAN_RECORD_MAX_BYTES", 4_000)
    big = "x" * 3_000
    pr.append_revision(root, task, pr.compose_revision(None, goal="g", plan=big + "1", done_when=[], affected_paths=[]), body=False)
    record, latest, _ = pr.append_revision(root, task, pr.compose_revision(None, goal="g", plan=big + "2", done_when=[], affected_paths=[]), body=False)
    assert record["revisions"][0]["plan_inline"] is False and record["revisions"][0]["plan"] == ""
    assert record["revisions"][0]["text_ref"]["sha256"] and latest["plan"] == big + "2" and latest["plan_inline"] is True
    exact = json.loads((root / "artifacts" / "tasks" / task / record["revisions"][0]["text_ref"]["path"]).read_text(encoding="utf-8")) \
        if (root / "artifacts" / "tasks" / task / record["revisions"][0]["text_ref"]["path"]).exists() else None
    assert exact is None or exact["plan"] == big + "1"


# ---------------------------------------------------------------- review history (context views)

def _answered_record(root, task, *, objections, withdrawn=(), retained=True):
    """A durable record with one revision and one ask: SEAT_A answered (its exact reply retained
    in the ask's ledger sources unless ``retained`` is off), SEAT_B still pending."""
    from ouroboros.review_ledger import retain_text_source

    seed_result(root, task, {"task_id": task, "status": "running"})
    record, _rev, _ = pr.append_revision(root, task, pr.compose_revision(
        None, goal="ship", plan="outline, then draft", done_when=["five slides"], affected_paths=[]), body=False)
    pr.append_ask(root, task, {"ask_id": "a1", "ts": "t", "revision": 1, "message": "look", "record_id": "rec-a1",
                               "participants": [_participant(SEAT_A), _participant(SEAT_B)]})
    raw = json.dumps({"prose": "the bar is too low", "objections": [dict(o) for o in objections],
                      "withdrawn": [dict(w) for w in withdrawn]})
    reply, _ = pr.parse_reply(raw)
    facts = {"status": "answered", "reply": reply}
    if retained:
        facts["reply_ref"] = retain_text_source(root, task, record_id="rec-a1", seat_id=SEAT_A, role="response",
                                                part="intention", text=raw)
    return pr.update_participant(root, task, ask_id="a1", seat_id=SEAT_A, facts=facts), raw


def test_record_history_binds_each_reply_to_its_source_and_rows_carry_the_objection_standing(tmp_path):
    record, raw = _answered_record(tmp_path, "task-h", objections=[{"id": "o1", "text": "no export"}])
    history = pr.record_history(tmp_path, "task-h", record)
    assert history["status"] == "complete" and history["gaps"] == [] and history["rule"] == pr.RECORD_HISTORY_RULE
    (round_,) = history["rounds"]
    assert round_["plan_ask"] == "a1" and round_["revision"] == 1 and round_["review_record_id"] == "rec-a1"
    answered, pending = round_["reviewers"]
    reply_ref = record["asks"][0]["participants"][0]["reply_ref"]
    assert answered["response"] == {"text": raw, "source": reply_ref} and "reply" not in answered
    assert pending["status"] == "pending" and "response" not in pending
    (row,) = history["decision_rows"]
    assert row["decision_kind"] == "plan_objection" and row["seat_id"] == SEAT_A and row["objection_id"] == "o1"
    assert row["remark"] == "no export" and row["history_role"] == "open" and row["source"] == reply_ref
    assert row["status"]["open"] is True and row["status"]["holds"] is True and row["status"]["withdrawn_in"] == ""
    assert pr.operative_revision(record)["kind"] == "plan_revision" and pr.operative_revision(pr.empty_record()) is None
    # The raiser withdrawing it makes the row history, sourced at the withdrawing reply.
    later = pr.append_ask(tmp_path, "task-h", {"ask_id": "a2", "ts": "t2", "revision": 1, "message": "fixed",
                                               "participants": [_participant(SEAT_A, reply={"withdrawn": [{"id": "o1", "why": "added"}]})]})
    (row,) = pr.record_history(tmp_path, "task-h", later)["decision_rows"]
    assert row["history_role"] == "historical" and row["status"]["withdrawn_in"] == "a2" and row["reason"] == "added"


def test_record_history_names_a_reply_without_a_retained_source_and_shows_its_typed_projection(tmp_path):
    record, _raw = _answered_record(tmp_path, "task-g", objections=[{"id": "o1", "text": "no export"}], retained=False)
    history = pr.record_history(tmp_path, "task-g", record)
    answered = history["rounds"][0]["reviewers"][0]
    assert "response" not in answered and answered["reply"]["objections"] == [{"id": "o1", "text": "no export"}]
    assert history["status"] == "source_unavailable" and history["gaps"][0]["source"]["seat_id"] == SEAT_A
    assert history["decision_rows"][0]["source"] is None


def test_an_objection_with_a_retained_source_is_a_bound_decision_notes_and_groups_address(tmp_path):
    """An organ objection row is a source-bound decision of the review-history views: an open one
    takes an actor's note (the view shows the account, the history it came from is untouched) but
    is never grouped; a withdrawn one groups; a row whose reply kept no source stays full."""
    from ouroboros.review_ledger import retain_text_source
    from ouroboros.review_history_view import decision_entries, project_decision_groups, project_decision_notes

    record, _raw = _answered_record(tmp_path, "task-n", objections=[{"id": "o1", "text": "no export"},
                                                                    {"id": "o2", "text": "no title slide"}])
    raw = json.dumps({"prose": "title added", "objections": [], "withdrawn": [{"id": "o2", "why": "added a title"}]})
    reply, _ = pr.parse_reply(raw)
    record = pr.append_ask(tmp_path, "task-n", {"ask_id": "a2", "ts": "t2", "revision": 1, "message": "title?",
                                                "record_id": "rec-a2", "participants": [_participant(SEAT_A)]})
    record = pr.update_participant(tmp_path, "task-n", ask_id="a2", seat_id=SEAT_A, facts={
        "status": "answered", "reply": reply, "reply_ref": retain_text_source(
            tmp_path, "task-n", record_id="rec-a2", seat_id=SEAT_A, role="response", part="intention", text=raw)})
    history = pr.record_history(tmp_path, "task-n", record)
    entries = {e["row"]["objection_id"]: e for e in decision_entries(history)}
    assert all(entries[oid]["bound_decision"] for oid in ("o1", "o2"))
    assert [row["history_role"] for row in history["decision_rows"]] == ["open", "historical"]
    assert all(row["decision_ref"].startswith("decision:") for row in history["decision_rows"])
    before = json.loads(json.dumps(history))
    note = {"bound_decision": entries["o1"]["bound_decision"], "remark": "Export is the open gap.",
            "reason": "The seat still holds it; I answer it next."}
    projected, applied, _unshortened = project_decision_notes(history, [note])
    shown = projected["decision_rows"][entries["o1"]["index"]]
    assert len(applied) == 1 and shown["remark"] == "Export is the open gap." and shown["authored_view"]["authorship"] == "actor"
    assert history == before  # a view: the history it projects is never rewritten
    group = {"bound_decisions": [entries["o2"]["bound_decision"]], "remark": "The title objection, withdrawn.",
             "reason": "Revision 1 added the title; the raiser withdrew it."}
    grouped, groups, gaps = project_decision_groups(history, [group], source_history=history,
                                                    selected_source={"kind": "test", "path": "view.json"})
    assert len(groups) == 1 and not gaps
    assert [row["decision_kind"] for row in grouped["decision_rows"]] == ["plan_objection", "actor_history_group"]
    _, groups, gaps = project_decision_groups(history, [{**group, "bound_decisions": [entries["o1"]["bound_decision"]]}],
                                              source_history=history, selected_source={"kind": "test", "path": "view.json"})
    assert not groups and gaps[0]["reason"] == "current_or_unresolved_decision_stays_explicit"
    # Other direction: no retained source, no binding (the row stays full in every view).
    unbound, _ = _answered_record(tmp_path, "task-u", objections=[{"id": "o1", "text": "no export"}], retained=False)
    (entry,) = decision_entries(pr.record_history(tmp_path, "task-u", unbound))
    assert entry["bound_decision"] is None and entry["reason"] == "unbound_or_legacy_decision"
    assert "decision_ref" not in pr.record_history(tmp_path, "task-u", unbound)["decision_rows"][0]


def test_plan_history_views_read_the_record_and_still_read_a_finished_v2_wave(tmp_path):
    """The context views (``current_plan_history`` → the task authority projection →
    compaction's capture) carry an organ plan, a finished earlier-schema wave, both together,
    and nothing for a task without a plan."""
    from types import SimpleNamespace

    from ouroboros.context import _task_authority_projection
    from ouroboros.review_history_view import (REVIEW_CONTEXT_INDEX_KEY, REVIEW_HISTORY_MESSAGE_KEY,
                                               capture_review_history_messages, current_plan_history)
    from tests import _legacy_plan_shared as legacy

    def views(task):
        runtime = _task_authority_projection(SimpleNamespace(drive_root=tmp_path, repo_dir=None), {"id": task})
        prefix, tail = capture_review_history_messages(runtime, task_id=task, drive_root=tmp_path)
        index = [json.loads(m["content"].partition("\n")[2])["current"] for m in tail if REVIEW_CONTEXT_INDEX_KEY in m]
        bodies = [m["content"] for m in tail if REVIEW_HISTORY_MESSAGE_KEY in m]
        return runtime, prefix, index, bodies

    # an organ plan: the record is the plan family's history, compaction moves it to the resident index
    _answered_record(tmp_path, "organ", objections=[{"id": "o1", "text": "no export"}])
    history, operative = current_plan_history(SimpleNamespace(drive_root=tmp_path, task_id="organ"))
    assert [r["plan_ask"] for r in history["rounds"]] == ["a1"] and operative["n"] == 1 and operative["kind"] == "plan_revision"
    runtime, prefix, index, bodies = views("organ")
    authority = runtime["plan_review_authority"]
    assert set(authority) == {"dispute_history", "operative_subject", "source"} and "plan_record" not in runtime
    assert prefix["plan_review_authority"]["dispute_history"]["representation"] == "resident_review_index"
    assert [row["objection_id"] for row in index[-1]["decision_rows"]] == ["o1"]
    assert index[-1]["operative_subject"]["plan"] == "outline, then draft"
    assert any("the bar is too low" in body for body in bodies)

    # a task with no plan at all: no plan authority and no plan index
    seed_result(tmp_path, "plain", {"task_id": "plain", "status": "running"})
    runtime, _prefix, index, bodies = views("plain")
    assert "plan_review_authority" not in runtime and index == [] and bodies == []

    # a finished v2 record keeps reading from its exact wave artifact
    closed = legacy.with_artifact(tmp_path, "old", legacy.wave(
        1, aggregate="GREEN", closed=True, findings=[{"finding_id": "s1:f", "class": "note", "slot": "s1", "summary": "use charts"}]))
    legacy.seed_plan_review_state(tmp_path, "old", legacy.v2_state(closed), status="completed")
    history, operative = current_plan_history(SimpleNamespace(drive_root=tmp_path, task_id="old"))
    assert [r["aggregate"] for r in history["rounds"]] == ["GREEN"] and operative["spec"]["goal"] == "Ship the deck"
    assert history["decision_rows"][0]["decision_kind"] == "plan_finding"
    runtime, _prefix, index, _bodies = views("old")
    assert runtime["plan_review_authority"]["current_attempt"]["fingerprint"] == closed["request_fingerprint"]
    assert index[-1]["decision_rows"][0]["finding_id"] == "s1:f"

    # an in-flight v2 task that went on with the organ: its waves first, then the record's asks
    _answered_record(tmp_path, "both", objections=[{"id": "o1", "text": "no export"}])
    legacy.seed_plan_review_state(tmp_path, "both", legacy.v2_state(legacy.with_artifact(tmp_path, "both", legacy.wave(1))))
    history, operative = current_plan_history(SimpleNamespace(drive_root=tmp_path, task_id="both"))
    assert [r.get("plan_ask") or r.get("aggregate") for r in history["rounds"]] == ["REVIEW_REQUIRED", "a1"]
    assert [r["decision_kind"] for r in history["decision_rows"]] == ["plan_objection"] and operative["kind"] == "plan_revision"
    assert pr.RECORD_HISTORY_RULE in history["rule"] and len(history["rule"]) > len(pr.RECORD_HISTORY_RULE)
# ---------------------------------------------------------------- Codex audit (2026-10-15), points 2 and 5

def test_unroutable_asks_stay_recorded_without_creating_a_plan_obligation():
    attempt = {"ts": "t", "revision": 1, "kind": "no_route", "ask": "council",
               "not_sendable": {SEAT_A: "health_skip[credential_pool_exhausted]"}}
    asked = {**_record(revisions=1), "attempts": [attempt]}
    decision = _gate(asked)
    assert decision["status"] == "clear" and decision["allow"] is True
    assert decision["asks_unroutable"] == 1 and decision["holding"] == []
    assert _gate(_record(revisions=1))["allow"] is True
    assert _gate({**pr.empty_record(), "attempts": [attempt]})["allow"] is True
    assert pr.unroutable_asks(asked) == 1 and pr.read_view(asked)["attempts"] == [attempt]
    assert pr.validate_record({**pr.empty_record(), "attempts": []})["attempts"] == []
    with pytest.raises(pr.PlanRecordInvalid):
        pr.validate_record({**pr.empty_record(), "attempts": "x"})


def test_a_pending_echo_never_reopens_a_settled_seat_under_the_records_lock(tmp_path):
    """Codex audit, point 5: the custody hook settles a released seat on its own thread while the
    dispatcher still writes the barrier's pending rows from a stale copy — the record itself
    refuses the echo under its lock; a seat still pending is written as before."""
    root, task = tmp_path, "task-race"
    seed_result(root, task, {"task_id": task, "status": "running"})
    pr.append_revision(root, task, pr.compose_revision(None, goal="g", plan="p", done_when=[], affected_paths=[]), body=False)
    pr.append_ask(root, task, {"ask_id": "a1", "ts": "t", "revision": 1, "message": "",
                               "participants": [_participant(SEAT_A), _participant(SEAT_B)]})
    pr.update_participant(root, task, ask_id="a1", seat_id=SEAT_A, facts={
        "status": "answered", "model": "m/x", "cost": 0.01,
        "reply": {"prose": "early", "objections": [{"id": "o1", "text": "no"}], "withdrawn": []}})
    echo = {"status": "pending", "model": "m/x", "operation_state": "pending_dispatch", "late_result_pending": True}
    settled, still = pr.update_participant(root, task, ask_id="a1", seat_id=SEAT_A, facts=echo)["asks"][0]["participants"]
    assert settled["status"] == "answered" and settled["reply"]["objections"][0]["id"] == "o1" and "late_result_pending" not in settled
    assert still["status"] == "pending" and "operation_state" not in still
    still = pr.update_participant(root, task, ask_id="a1", seat_id=SEAT_B, facts=echo)["asks"][0]["participants"][1]
    assert still["operation_state"] == "pending_dispatch" and still["late_result_pending"] is True
    merged = pr.update_participant(root, task, ask_id="a1", seat_id=SEAT_A, facts={"reply_ref": {"status": "empty"}})
    assert merged["asks"][0]["participants"][0]["reply_ref"] == {"status": "empty"}, "status-less facts still merge"
    assert pr.open_objections(merged)[0]["holds"] is True


def test_the_one_legacy_loader_names_the_read_that_failed_and_projects_the_rest(tmp_path, monkeypatch):
    """Scope R1, Fable 3 / D2: one loader behind import_legacy_v2; callers choose their policy."""
    from ouroboros.tools import plan_review_artifacts as artifacts
    from ouroboros.tools.plan_review_artifacts import PlanReviewSourceUnavailable

    seed_result(tmp_path, "t-ok", {"task_id": "t-ok", "status": "running",
                                   "plan_review_state": _v2_state(closed=False, blocking_findings=("s1",))})
    legacy, error = pr.legacy_facts(tmp_path, "t-ok")
    assert error == {} and legacy["engaged"] is True and [o["id"] for o in legacy["open_objections"]] == ["s1:f1"]
    seed_result(tmp_path, "t-bad", {"task_id": "t-bad", "status": "running", "plan_review_state": {"schema_version": 99}})
    legacy, error = pr.legacy_facts(tmp_path, "t-bad")
    assert legacy["engaged"] is False and error["source"] == "plan_review_state" and "PLAN_REVIEW_STATE_INVALID" in error["error"]
    monkeypatch.setattr(artifacts, "current_author_plan",
                        lambda root, tid, st: (_ for _ in ()).throw(PlanReviewSourceUnavailable("PLAN_AUTHOR_SOURCE_UNAVAILABLE: gone")))
    legacy, error = pr.legacy_facts(tmp_path, "t-ok")
    assert error["source"] == "author_plan" and "gone" in error["error"] and [o["id"] for o in legacy["open_objections"]] == ["s1:f1"]
    absent, error = pr.legacy_facts(tmp_path, "t-none")
    assert absent["engaged"] is False and absent["open_objections"] == [] and error == {}
