"""The cost contract's second axis: a recurring pass costs the same whatever the live set holds.

BIBLE P5 and P13, and the class of the 02.11 saturation: the interpreter that answers the owner does work bounded by the
rate of logical changes, never by the live set. tests/test_cost_contract.py holds every boundary flat against unrelated
history on a root with three running tasks and no fence in the queue; on 01.11 every one of those laws held while 28 running
tasks, 20 paused trees and 2,252 reconcile markers took one interpreter's whole core. Here one booted driver process
(tests/_live_set.py) grows the live set an axis at a time, and measures every pass after each step: each duty as the
tick starts it, the reconcile pickup, the delegated body, a heartbeat, ``GET /api/state``, a queue publication, the
assignment tick's hygiene and the timeout pass.

The laws (FINAL_PLAN E2, E3, E4; counts, never a clock):

- FLAT: over an unchanged live set, a pass's counts do not grow when one axis of the set grew (``AXES``: running 1 -> 50,
  paused trees 1 -> 20 with 86 members, pending 1 -> 61, held runs 1 -> 122, schedules 1 -> 35, markers 2 -> 120);
- AT REST: at the largest set, a pass over unchanged inputs writes nothing, reads no task body and no usage row, and
  publishes no file;
- HISTORY: at the largest set, the same pass costs the same at k=1 and k=30;
- the heartbeat is an in-memory stamp and one frame; the queue's owner text appears once in its document and nothing is
  serialized or synced under the queue lock; a pass that rests still works on a real change.

A law a tree cannot meet yet is ``xfail(strict=True)`` naming its measured numbers and the unit that removes it (``GAPS``),
so it stays visible instead of being averaged away, and flips (XPASS) when the unit lands; the row is then deleted.
Every count is read with what the maintenance children behind it reported (``judged_pass``): a child that refused, or a
counted rest whose own child left work pending, makes the read an error by name, never a count that holds (#2465).
"""

from __future__ import annotations

import json
import os
import sqlite3
from typing import Any, Dict, Iterator, List

import pytest

from tests._cost_driver import name_owed, refused
from tests._live_set import STAGES
from tests._stand_cache import STAND_MARKS, measuring
from tests.test_cost_contract import BYTE_JITTER, DUTY_NAMES, contract_breaches, rest_breaches, vm_allowance

pytestmark = [pytest.mark.scale, *STAND_MARKS]

K_SMALL, K_LARGE = 1, 30
AXES = STAGES[1:]
LARGEST = STAGES[-1]
PASSES = (*DUTY_NAMES, "extension-reconcile", "heartbeat", "state", "publication", "assign", "timeouts")
# The growth an axis stands for (the VM steps' logarithmic allowance reads it, as the history law reads k).
GROWTH = {"running": 50, "paused": 20, "pending": 61, "held": 122, "schedules": 35, "markers": 60}
# (axis, pass) and pass -> the breach with its measured numbers and the unit that removes it, a strict xfail until that unit
# lands, then the row goes. Both empty on 02.11: every row the base (1d88f843b) had was removed by the unit it named.
GAPS: Dict[tuple, str] = {
    ("held", "delegated-body"): "The body follows each live held run with its own job threads (one read, one follow, one "
    "collection per run, so a quiet run delays no other): thread starts grow with the live held runs (0 to 36 here, 0 to 8 "
    "on the stand). A follower thread that lives as long as its run would remove this.",
}
REST_GAPS: Dict[str, str] = {}
# (pass -> counts its own work allows at rest). The duty exists to look at the open attempts: one indexed read, zero rows here.
ALLOWED_AT_REST = {"abandoned_usage_reconciliation": {"usage_row_reads": 1}}
HISTORY_GAPS: Dict[str, str] = {}


def _gap_mark(reason: str) -> pytest.MarkDecorator:
    """Only the law's own assertion may fail under its row: a measurement that broke errors by name instead of counting as today's breach."""
    return pytest.mark.xfail(strict=True, raises=AssertionError, reason=reason)


def _gap(table: Dict[Any, str], key: Any) -> tuple:
    return (_gap_mark(table[key]),) if key in table else ()


@pytest.fixture(scope="module")
def live() -> Iterator[Dict[int, Dict[str, Any]]]:
    with measuring(K_SMALL, plan="live_set") as small, measuring(K_LARGE, plan="live_set") as large:
        yield {K_SMALL: small, K_LARGE: large}


def judged_pass(measured: Dict[str, Any], stage: str, name: str, phase: str = "rest", where: str = "") -> Dict[str, Any]:
    """A pass's count at ``stage``, once the maintenance behind it is read (``refused``, tests/_cost_driver.py). A child
    that refused did no work, so no count over it proves anything (#2465), and its failed job need not fall due again:
    the count answers for every run of this pass up to it, at this stage and the earlier ones (``tags``). Pending is
    owed work a later run may do, named, except in the counted rest itself, where every job that ran finished. A run
    that spawned no child has nothing to read. A refusal raises no AssertionError, so a law under a gap row
    (``_gap_mark``) errors by name instead of counting it as today's breach."""
    stages = measured["live_set"]["stages"]
    entry = stages[stage].get(name)
    assert entry is not None, f"{name} was not driven at {stage}{where}"
    own = entry["tags"] if phase == "rest" else entry["tags"][:1]
    behind = [*(tag for earlier in STAGES[:STAGES.index(stage)] for tag in stages[earlier][name]["tags"]), *own]
    reports = measured.get("child_reports", {})
    refusals = [(tag, lines) for tag in behind for report in reports.get(tag, ())
                if (lines := refused(report, rest=phase == "rest" and tag == own[-1]))]
    if refusals:
        raise RuntimeError(f"{own[-1]}{where} was counted over maintenance that did not do its work: {refusals}")
    name_owed(f"{own[-1]}{where}", *reports.get(own[-1], ()))
    return entry[phase]


def _rest(live: Dict[int, Any], k: int, stage: str, name: str, phase: str = "rest") -> Dict[str, Any]:
    return judged_pass(live[k], stage, name, phase, f" at k={k}")


def _growing(before: Dict[str, Any], after: Dict[str, Any]) -> List[tuple]:
    old = before.get("callers", {})
    grown = [(source, amount - old.get(source, 0)) for source, amount in after.get("callers", {}).items()
             if amount > old.get(source, 0)]
    return sorted(grown, key=lambda item: -item[1])[:6]


def live_set_breaches(before: Dict[str, int], after: Dict[str, int], growth: int) -> Dict[str, Any]:
    """``contract_breaches`` with the VM allowance per statement the pass made: over a small table the query planner
    scans by a covering index where over a large one it searches (a read of one id over 350 tasks walks them, over the
    owner's 6,633 it seeks), so one statement's steps follow the table's size up to its plan's switch. That is no read
    per live item, which the statements, the decodes and the bytes hold; a scan per item over a large table still fails."""
    breaches = contract_breaches(before, after, growth)
    steps = breaches.get("vm_steps")
    if steps and after.get("vm_steps", 0) - before.get("vm_steps", 0) <= vm_allowance(growth) * max(1, after.get("statements", 0)):
        del breaches["vm_steps"]
    return breaches


# A pass whose input is consumed by its first run is held on that run: the markers a reload left, then none. Reading each
# marker once is the input's own cost, paid per request made; the reconcile each one causes is what must not follow them.
PHASE = {("markers", "extension-reconcile"): "first"}
RECONCILE_WORK = ("connect", "statements", "begins", "writes", "popen", "thread_start", "lock_acquired")


@pytest.mark.parametrize("axis, name", [pytest.param(axis, name, marks=_gap(GAPS, (axis, name)))
                                        for axis in AXES for name in PASSES])
def test_a_pass_costs_the_same_when_one_axis_of_the_live_set_grew(live, axis, name):
    phase = PHASE.get((axis, name), "rest")
    # A first run's baseline is the first run that met the input: the stage after it has none left to meet.
    before = _rest(live, K_SMALL, STAGES[0] if phase == "first" else STAGES[STAGES.index(axis) - 1], name, phase)
    after = _rest(live, K_SMALL, axis, name, phase)
    breaches = live_set_breaches(before["counts"], after["counts"], GROWTH[axis])
    if phase == "first":
        breaches = {key: value for key, value in breaches.items() if key in RECONCILE_WORK}
    assert not breaches, (axis, name, breaches, _growing(before, after))


def live_rest_breaches(counts: Dict[str, int]) -> Dict[str, int]:
    """The rest law of a duty (``rest_breaches``: no writes, no history or evidence read, no walk of an execution root),
    and the rest of E2: no task body read, no usage row read, no file published."""
    keys = ("task_body_reads", "usage_row_reads", "fsync", "replace")
    return {**rest_breaches(counts), **{key: counts[key] for key in keys if counts.get(key)}}


# The tick's own hygiene and its deadlines are held by the flat law alone: reading money and pending controls once per
# tick is their work; E2 names the duties, the heartbeat, the state and the publication.
RESTING = tuple(name for name in PASSES if name not in ("assign", "timeouts"))


@pytest.mark.parametrize("name", [pytest.param(name, marks=_gap(REST_GAPS, name)) for name in RESTING])
def test_a_pass_over_the_unchanged_largest_set_rests(live, name):
    snapshot = _rest(live, K_SMALL, LARGEST, name)
    breaches = live_rest_breaches({**snapshot["counts"], **ALLOWED_AT_REST.get(name, {})})
    breaches = {key: value for key, value in breaches.items() if key not in ALLOWED_AT_REST.get(name, {})}
    assert not breaches, (name, breaches, sorted(snapshot.get("names", {}).items())[:6],
                          [source for source in snapshot.get("callers", {}) if source.split(" ")[0] in breaches][:6])


@pytest.mark.parametrize("name", [pytest.param(name, marks=_gap(HISTORY_GAPS, name)) for name in PASSES])
def test_a_pass_over_the_largest_set_costs_the_same_at_every_size_of_history(live, name):
    little, big = (_rest(live, k, LARGEST, name) for k in (K_SMALL, K_LARGE))
    breaches = contract_breaches(little["counts"], big["counts"], K_LARGE)
    assert not breaches, (name, breaches, _growing(little, big))


def test_the_live_set_was_really_there(live):
    """The laws above read nothing unless the set exists: the owner's Pause took every tree, and the queue holds their fences."""
    for k in (K_SMALL, K_LARGE):
        laid = live[k]["live_set"]["laid"]
        assert laid["paused"]["fences_in_queue"] == 20 and laid["paused"]["states"] == ["paused"], laid["paused"]
        assert laid["paused"]["members"] + laid["smallest"]["paused"]["members"] == 86, laid["paused"]
        assert laid["running"]["running"] == 50 and laid["pending"]["pending"] == 61, laid
        assert laid["held"] == {**laid["held"], "held": 122, "settled": 113}, laid["held"]
        assert laid["markers"]["requested"] == 120, laid["markers"]
    assert _rest(live, K_SMALL, "markers", "extension-reconcile", "first")["counts"].get("stat", 0) > 0


def test_the_meter_sees_a_sync_a_publication_and_what_the_queue_lock_holds(tmp_path):
    """The instrument of E4: a sync and a replacement by file, and the same work again when the queue lock is held."""
    from supervisor import queue
    from tests._cost_meter import CostMeter, tagged

    def publish(name: str) -> None:
        fd = os.open(tmp_path / f"{name}.tmp", os.O_CREAT | os.O_WRONLY, 0o600)
        os.fsync(fd)
        os.close(fd)
        os.replace(tmp_path / f"{name}.tmp", tmp_path / name)
        json.dumps({"bytes": "x" * 5_000})

    def work() -> None:
        publish("outside")
        with queue._queue_lock:
            publish("inside")

    with CostMeter(tmp_path) as meter:
        tagged("publish", work)
    snapshot = meter.snapshot("publish")
    counts = snapshot["counts"]
    assert (counts["fsync"], counts["replace"], counts["under_q:fsync"], counts["under_q:replace"]) == (2, 2, 1, 1), counts
    assert counts["json_dumps"] == 2 and 5_000 <= counts["under_q:json_dumps_chars"] < counts["json_dumps_chars"], counts
    assert {"fsync outside.tmp", "fsync inside.tmp", "replace outside", "replace inside"} <= set(snapshot["names"]), snapshot["names"]


def test_the_meter_counts_a_task_body_and_a_usage_row_and_names_each_store(tmp_path):
    """The instrument of E2's rest: which store a statement asked, and the two reads the law forbids at rest."""
    from tests._cost_meter import CostMeter, tagged

    def read() -> None:
        for name, table in (("current.sqlite", "task_bodies"), ("usage.sqlite", "attempts"), ("dialog.sqlite", "dialogue"),
                            ("index.sqlite3", "pages")):
            conn = sqlite3.connect(tmp_path / name)
            conn.execute(f"CREATE TABLE IF NOT EXISTS {table} (id INTEGER)")
            conn.execute(f"SELECT id FROM {table}").fetchall()
            conn.close()
        conn = sqlite3.connect(tmp_path / "usage.sqlite")
        conn.execute("CREATE TABLE IF NOT EXISTS meta (id INTEGER)")
        conn.execute("SELECT id FROM meta").fetchall()  # a marker's read is no row of money
        conn.close()

    with CostMeter(tmp_path) as meter:
        tagged("read", read)
    counts = meter.snapshot("read")["counts"]
    assert (counts["task_body_reads"], counts["usage_row_reads"]) == (1, 1), counts
    assert (counts["statements:current"], counts["statements:dialog"], counts["statements:other"]) == (2, 2, 2), counts
    assert counts["statements:usage"] == 4, counts


def test_a_scan_of_the_axis_keeps_the_instrument_honest():
    """Both directions over the comparison itself: one more store read per axis item is a breach, equal counts are not."""
    assert contract_breaches({"begins": 1}, {"begins": 50}, 50) == {"begins": (1, 50)}
    assert not contract_breaches({"begins": 1}, {"begins": 3}, 50)
    assert live_rest_breaches({"writes": 1, "task_body_reads": 2}) == {"writes": 1, "task_body_reads": 2}
    assert live_rest_breaches({"begins": 3, "statements": 9}) == {}


# ---- the heartbeat (E3) --------------------------------------------------------------------------------------------

def test_a_heartbeat_touches_no_store(live):
    counts = _rest(live, K_SMALL, LARGEST, "heartbeat")["counts"]
    assert not {key: counts[key] for key in ("begins", "statements", "connect") if counts.get(key)}, counts


def test_a_heartbeat_stamps_liveness_and_reaches_every_client_once(live):
    """The in-memory stamp moved; one frame for each of the 2 clients, as for none."""
    facts = live[K_SMALL]["live_set"]["heartbeat"]
    assert facts["stamp_moved"], facts
    assert facts["frames_per_client"] == [1, 1], facts


def test_a_heartbeat_costs_the_same_with_clients_as_without(live):
    facts = live[K_SMALL]["live_set"]["heartbeat"]
    assert not contract_breaches(facts["no_clients"], facts["two_clients"], 2), facts


# ---- the queue's publication (E4) -------------------------------------------------------------------------------------

def test_the_published_queue_carries_the_owner_s_words_once(live):
    copies = live[K_SMALL]["live_set"]["change"]["owner_text_copies"]
    assert copies == 1, f"{copies} copies of one message in the document a restart reads"


def test_a_reader_of_the_published_queue_still_sees_the_whole_row(live):
    """However the document stores it, what a reader gets is the row, with its owner text whole."""
    assert live[K_SMALL]["live_set"]["change"]["reader_sees_owner_text"]


def test_a_published_change_holds_nothing_under_the_queue_lock_but_its_capture(live):
    """E4: what an owner's control waits behind does not grow with the queue's bytes, and no file is touched under Q."""
    little = live[K_SMALL]["live_set"]["change"]["smallest"]["counts"]
    big = live[K_SMALL]["live_set"]["change"][LARGEST]["counts"]
    assert not {key: big[key] for key in ("under_q:fsync", "under_q:replace") if big.get(key)}, big
    assert big.get("under_q:json_dumps_chars", 0) - little.get("under_q:json_dumps_chars", 0) <= BYTE_JITTER, (little, big)


def test_a_change_is_published_though_an_unchanged_queue_is_not(live):
    """The rest is not death: one real change lands in the document a reader opens."""
    facts = live[K_SMALL]["live_set"]["change"]
    names = facts[LARGEST].get("names", {})
    assert names.get("replace queue_snapshot.json") == 1, names  # one change, one publication of the document
    assert facts["row_published"], facts["row_published"]


def test_the_liveness_certificate_is_written_without_a_sync(live):
    names = live[K_SMALL]["live_set"]["change"][LARGEST].get("names", {})
    assert not [name for name in names if name.startswith("fsync") and "queue.alive" in name], names


def test_the_certificate_follows_the_documents_it_certifies(live):
    """Invalidate, documents, certify: a reader never finds the new certificate over the old documents."""
    order = live[K_SMALL]["live_set"]["change"]["order"]
    assert order and order[-1] == "queue.alive" and "queue_snapshot.json" in order, order
    assert order.index("queue_snapshot.json") < order.index("queue.alive"), order


# ---- the answer moves when the live set does --------------------------------------------------------------------------

def test_the_state_answer_moves_on_a_real_change(live):
    facts = live[K_SMALL]["live_set"]["state_change"]
    assert facts["after"] == facts["before"] + 1, facts
