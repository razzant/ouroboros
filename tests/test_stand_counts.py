"""The stand's counted half: E2, E3, E4, E6 and E8 over the owner's install of 01.11, on WAL and on PERSIST.

tests/_stand.py builds the shape in the booted driver process (no worker, no model; a request is where a turn's count ends). The
reduced shape runs in the default lane; the full one (50 running, 20 paused trees, 61 pending, 122 held runs, 35 schedules, 2,000
markers, 500 rooms, a 125k-token story) is the opt-in ``integration`` lane (``-m integration -o addopts=''``), as the cost contract's
k=100 is. History's axis at this live set is held by tests/test_cost_live_set.py, E2's positive companions by tests/test_cost_contract.py
(B7), the reuse of one story capture by tests/test_memory_story_reuse.py. E1, E5, E7, E9 and E10 need real processes:
tests/system_e2e/test_stand_server.py.

Every law here is an equality of what the meter counts. A breach in today's code is a strict xfail naming the unit that removes it,
from the live-set axis's own gap tables, so the two cannot disagree. Wall-clock figures are reported (into ``OUROBOROS_TEST_EVIDENCE_ROOT``),
never judged.
"""

from __future__ import annotations

import json
import os
import pathlib
from typing import Any, Dict, Iterator

import pytest

from tests._live_set import STAGES
from tests._stand import CHARS_PER_TOKEN, SHAPES
from tests._stand_cache import STAND_MARKS, measuring
from tests._turn_law import readers_of
from tests.test_cost_contract import COUNT_JITTER, ROUND
from tests.test_cost_live_set import (
    ALLOWED_AT_REST,
    BYTE_JITTER,
    GAPS,
    PASSES,
    PHASE,
    RECONCILE_WORK,
    REST_GAPS,
    RESTING,
    _gap,
    _gap_mark,
    judged_pass,
    live_rest_breaches,
    live_set_breaches,
)

pytestmark = [pytest.mark.scale, *STAND_MARKS]

K = 1
# The owner's turn beside others: two read transactions per other live turn, whichever store is asked.
BESIDE_GAP = ("no unit of this sprint: room_reader.is_reading asks task_drive_status and read_fence for every other live turn of the install "
              "(4 more read transactions beside 2 others, 8 beside 4); the rows the others added are payload")
# A pass's flat gap over the whole growth: the union of the axis law's rows for that pass.
FLAT_GAPS = {name: "; ".join(reason for (_axis, gap_name), reason in GAPS.items() if gap_name == name)
             for _axis, name in GAPS}


@pytest.fixture(scope="module", params=[
    pytest.param(("WAL", "reduced"), id="wal-reduced"), pytest.param(("PERSIST", "reduced"), id="persist-reduced"),
    pytest.param(("WAL", "full"), id="wal-full", marks=pytest.mark.integration),
    pytest.param(("PERSIST", "full"), id="persist-full", marks=pytest.mark.integration)])
def stand(request) -> Iterator[Dict[str, Any]]:
    journal, shape = request.param
    with measuring(K, plan="stand", stand=shape, journal=journal) as measured:
        measured["shape_name"], measured["journal"] = shape, journal
        yield measured


def _pass(stand: Dict[str, Any], stage: str, name: str, phase: str = "rest") -> Dict[str, Any]:
    """The live-set axis's own read: these laws compare only the smallest and the largest set, so the passes of the stages
    between answer through the largest one's count (``judged_pass``)."""
    return judged_pass(stand, stage, name, phase)


# ---- E2: the rest law on the live set ------------------------------------------------------------------------------------

@pytest.mark.parametrize("name", [pytest.param(name, marks=_gap(FLAT_GAPS, name)) for name in PASSES])
def test_e2_a_pass_costs_the_same_over_the_smallest_and_the_largest_set(stand, name):
    phase = PHASE.get(("markers", name), "rest")
    little, big = _pass(stand, STAGES[0], name, phase), _pass(stand, STAGES[-1], name, phase)
    breaches = live_set_breaches(little["counts"], big["counts"], 50)
    if phase == "first":
        breaches = {key: value for key, value in breaches.items() if key in RECONCILE_WORK}
    assert not breaches, (name, breaches)


@pytest.mark.parametrize("name", [pytest.param(name, marks=_gap(REST_GAPS, name)) for name in RESTING])
def test_e2_a_pass_over_the_unchanged_largest_set_rests(stand, name):
    counts = {**_pass(stand, STAGES[-1], name)["counts"], **ALLOWED_AT_REST.get(name, {})}
    breaches = {key: value for key, value in live_rest_breaches(counts).items() if key not in ALLOWED_AT_REST.get(name, {})}
    assert not breaches, (name, breaches)


def test_e2_the_set_was_really_there(stand):
    shape = SHAPES[stand["shape_name"]].live
    laid = stand["live_set"]["laid"]
    assert laid["paused"]["fences_in_queue"] == shape.paused and laid["paused"]["states"] == ["paused"], laid["paused"]
    assert (laid["running"]["running"], laid["pending"]["pending"]) == (shape.running, shape.pending), laid
    assert (laid["held"]["held"], laid["held"]["settled"], laid["schedules"]["schedules"]) == (
        shape.held, shape.settled, shape.schedules), laid
    assert laid["markers"]["requested"] == shape.markers, laid["markers"]


# ---- E3: the heartbeat ---------------------------------------------------------------------------------------------------

def test_e3_a_heartbeat_touches_no_store(stand):
    counts = _pass(stand, STAGES[-1], "heartbeat")["counts"]
    assert not {key: counts[key] for key in ("begins", "statements", "connect") if counts.get(key)}, counts


def test_e3_one_stamp_and_one_frame_per_client(stand):
    facts = stand["live_set"]["heartbeat"]
    assert facts["stamp_moved"] and facts["frames_per_client"] == [1, 1], facts


# ---- E4: the queue's publication -------------------------------------------------------------------------------------------

def test_e4_the_owner_s_words_appear_once_in_the_document(stand):
    assert stand["live_set"]["change"]["owner_text_copies"] == 1


def test_e4_a_reader_still_gets_the_whole_row(stand):
    assert stand["live_set"]["change"]["reader_sees_owner_text"]


def test_e4_work_under_the_queue_lock_does_not_follow_the_queue_s_bytes(stand):
    little, big = (stand["live_set"]["change"][stage]["counts"] for stage in (STAGES[0], STAGES[-1]))
    assert not {key: big[key] for key in ("under_q:fsync", "under_q:replace") if big.get(key)}, big
    assert big.get("under_q:json_dumps_chars", 0) - little.get("under_q:json_dumps_chars", 0) <= BYTE_JITTER, (little, big)


def test_e4_the_certificate_is_written_without_a_sync(stand):
    names = stand["live_set"]["change"][STAGES[-1]].get("names", {})
    assert not [name for name in names if name.startswith("fsync") and "queue.alive" in name], names


def test_e4_a_change_lands_and_the_certificate_comes_last(stand):
    change = stand["live_set"]["change"]
    assert change["row_published"] and change["order"][-1] == "queue.alive" and "queue_snapshot.json" in change["order"], change["order"]


def test_e4_a_kill_after_each_step_never_certifies_what_did_not_land(stand):
    """The publication dies after each of its file replacements in turn: a reader's certificate (``queue_stamp``) then names a
    generation only if every document carries that generation: the old fence-free view is never certified by the new one."""
    crashes = stand["live_set"]["crashes"]
    assert len(crashes) >= 3 and not crashes[0]["certifies"], crashes  # it died before its first document landed
    assert crashes[-1]["published"] and crashes[-1]["certifies"] and crashes[-1]["documents_agree"], crashes  # the whole one
    for crash in crashes:
        assert crash["certifies"] is False or crash["documents_agree"], crash


# ---- E6: the owner's wait ------------------------------------------------------------------------------------------------

@_gap_mark(BESIDE_GAP)
def test_e6_the_owner_s_turn_makes_the_same_round_trips_beside_other_turns(stand):
    facts = stand["stand"]
    assert facts["others_held"] == SHAPES[stand["shape_name"]].turns - 1, facts["others_held"]
    alone, beside = facts["T:stand:alone"]["counts"], facts["T:stand:beside"]["counts"]
    growth = {key: beside.get(key, 0) - alone.get(key, 0) for key in ROUND}
    assert not {key: delta for key, delta in growth.items() if delta > COUNT_JITTER}, (growth, readers_of(facts["T:stand:alone"], facts["T:stand:beside"])({"statements": 1}))


def test_e6_the_others_were_live_while_it_was_counted(stand):
    assert stand["stand"]["registry_live"] == SHAPES[stand["shape_name"]].turns - 1, stand["stand"]["registry_live"]


# ---- E8: the horizon and its bytes -----------------------------------------------------------------------------------------

@pytest.mark.parametrize("tag", ["T:stand:first", "T:stand:alone", "T:stand:beside"])
def test_e8_main_s_request_carries_every_room_s_story(stand, tag):
    request = stand["stand"]["requests"][tag]
    assert request["rooms_carried"] == SHAPES[stand["shape_name"]].rooms, request


def test_e8_the_view_s_receipt_counts_the_story_and_every_live_room(stand):
    shape = SHAPES[stand["shape_name"]]
    facts = stand["stand"]
    blocks = facts["memory_view"]["blocks"]
    assert blocks["conversations"] >= shape.rooms, blocks
    assert blocks["story_tokens"] * CHARS_PER_TOKEN >= facts["rooms"]["story_chars"], (blocks, facts["rooms"])


def test_e8_two_foci_over_one_chronicle_read_the_same_bytes(stand):
    shas = {request["story_block_sha"] for request in stand["stand"]["requests"].values()}
    assert len(shas) == 1 and "" not in shas, shas


# ---- the facts --------------------------------------------------------------------------------------------------------------

def test_the_stand_reports_what_it_timed_as_facts(stand):
    """Latencies decide nothing here; they are the receipt a reader compares with the owner's numbers."""
    stages = stand["live_set"]["stages"]
    facts = {"journal": stand["journal"], "shape": stand["shape_name"], "driver_seconds": stand.get("driver_seconds"),
             "marks": stand.get("marks", {}),
             "slowest_passes_sec": sorted(((stages[stage][name]["seconds"], stage, name) for stage in stages for name in stages[stage]),
                                          reverse=True)[:10]}
    print(json.dumps(facts, indent=1))
    target = os.environ.get("OUROBOROS_TEST_EVIDENCE_ROOT", "")  # the launcher keeps this folder past the run
    if target:
        path = pathlib.Path(target) / f"stand-counts-{stand['journal']}-{stand['shape_name']}.json"
        path.write_text(json.dumps(facts, indent=1), encoding="utf-8")
    assert facts["slowest_passes_sec"]
