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

Where today's code breaks a law the test is ``xfail(strict=True)`` naming its measured numbers and the unit that removes it
(``GAPS``), so this file is green now and flips (XPASS) when the unit lands; the row is then deleted. A row with no
unit is a breach the sprint's plan did not assign: it stays visible here instead of being averaged away.
"""

from __future__ import annotations

from typing import Any, Dict, List

import pytest

from tests._live_set import STAGES
from tests.test_cost_contract import BYTE_JITTER, DUTY_NAMES, contract_breaches, measure, rest_breaches, vm_allowance

pytestmark = pytest.mark.scale

K_SMALL, K_LARGE = 1, 30
AXES = STAGES[1:]
LARGEST = STAGES[-1]
PASSES = (*DUTY_NAMES, "extension-reconcile", "heartbeat", "state", "publication", "assign", "timeouts")
# The growth an axis stands for (the VM steps' logarithmic allowance reads it, as the history law reads k).
GROWTH = {"running": 50, "paused": 20, "pending": 61, "held": 122, "schedules": 35, "markers": 60}
# Today's numbers below: rest counts at k=1, measured 02.11 on the base (1d88f843b). A unit named here removes the row when
# it lands; ``UNASSIGNED`` is a breach of E2 no unit of the sprint owns, kept in view rather than averaged away.
LC, LD, LB, LE = "LC", "LD", "LB", "LE"
UNASSIGNED = "no unit of this sprint"
QUEUE_DOC_PER_ROW = ("task_status.effective_task_result parses the whole queue document once per row it looks at, under "
                     "reconcile_orphaned_running_tasks every 300 s")
QUEUE_DOC_PER_PASS = "_startup_live_task_ids parses the whole queue document once per 600 s pass"

GAPS: Dict[tuple, str] = {
    ("running", "cold-custody"): f"{LC} L2: one budget_pause_row transaction per RUNNING row every 3 s (begins 1 -> 50)",
    ("running", "state"): f"{LE} L6: _task_activity_facts and read_fence per activity row (decodes 145 -> 439, stats 128 -> 520)",
    ("paused", "owner_pause_settlement"): f"{LC} L1: the census of every fence, each member's strict body (begins 14 -> 284)",
    ("paused", "state"): f"{LE} L6: the tree census per paused root (begins 15 -> 53)",
    ("paused", "assign"): f"{LC} L3: _drop_cancelled_pending reads a body per pending row under Q (begins 6 -> 44)",
    ("pending", "state"): f"{LE} L6: one census row per pending task (decodes 914 -> 1,154, stats 824 -> 1,304)",
    ("pending", "assign"): f"{LC} L3: _drop_cancelled_pending reads a body per pending row under Q (begins 44 -> 104)",
    ("held", "state"): f"{LE} L6: 122 held runs decoded to show the open ones (begins 53 -> 71)",
    ("schedules", "schedule-admission"): f"{LD} L9: every row's policy and receipts on every tick (begins 8 -> 49)",
    ("markers", "extension-reconcile"): f"{LD} L8: one reconcile per marker of two skills, each publishing a generation "
                                        "(120 markers: 104 child processes)",
    ("running", "custody-maintenance"): f"{LB} L5 shrinks the bytes, {UNASSIGNED} ends the growth: {QUEUE_DOC_PER_PASS} (0.16 -> 0.21 MB)",
    ("paused", "custody-maintenance"): f"{LB} L5 shrinks the bytes, {UNASSIGNED} ends the growth: {QUEUE_DOC_PER_PASS} (0.21 -> 4.3 MB)",
    ("held", "custody-maintenance"): f"{LE} L6's open-only selector removes part, {UNASSIGNED} the rest: three audits decode "
                                     "every held run (decodes 94 -> 587), and the queue document again",
    ("running", "reconcile-maintenance"): f"{UNASSIGNED}: {QUEUE_DOC_PER_ROW} (begins 28 -> 77, 10 MB read)",
    ("paused", "reconcile-maintenance"): f"{UNASSIGNED}: {QUEUE_DOC_PER_ROW}, and budget_pause_row per member (begins 77 -> 224, 216 MB read)",
    ("pending", "reconcile-maintenance"): f"{UNASSIGNED}: recover_interrupted_preparations lists every open document (decodes 1,031 -> 1,091)",
    ("held", "reconcile-maintenance"): f"{UNASSIGNED}: {QUEUE_DOC_PER_ROW}: the document it parses 240 times grew (216 -> 361 MB read)",
    ("held", "delegated-body"): f"{LE} L6's open-only selector: the body's tick decodes every held run to find the open ones (decodes 5 -> 504)",
}
REST_GAPS: Dict[str, str] = {
    "owner_pause_settlement": f"{LC} L1: 20 settled trees re-derived on every pass (122 task bodies, 20 usage reads)",
    "schedule-admission": f"{LD} L9: 18 task bodies for 35 unchanged rows",
    "heartbeat": f"{LD} L4: a root heartbeat reads custody and usage (two transactions, one usage read)",
    "state": f"{LE} L6: an unchanged revision rebuilds the answer (10 task bodies, 23 usage reads)",
    "publication": f"{LB} L5: queue.alive replaced and synced on every unchanged publication",
    "custody-maintenance": f"{UNASSIGNED}: a 600 s pass over unchanged inputs reads 10 task bodies (delegate_terminal's refresh)",
    "reconcile-maintenance": f"{UNASSIGNED}: a 300 s pass over unchanged inputs reads 52 task bodies (obligations.result_rows, the orphan reconcile)",
}
# (pass -> counts its own work allows at rest). The duty exists to look at the open attempts: one indexed read, zero rows here.
ALLOWED_AT_REST = {"abandoned_usage_reconciliation": {"usage_row_reads": 1}}
HISTORY_GAPS: Dict[str, str] = {}


def _gap(table: Dict[Any, str], key: Any) -> tuple:
    return (pytest.mark.xfail(strict=True, reason=table[key]),) if key in table else ()


def _gap_mark(reason: str) -> pytest.MarkDecorator:
    return pytest.mark.xfail(strict=True, reason=reason)


@pytest.fixture(scope="module")
def live(tmp_path_factory) -> Dict[int, Dict[str, Any]]:
    return {k: measure(tmp_path_factory.mktemp("live"), k, plan="live_set") for k in (K_SMALL, K_LARGE)}


def _rest(live: Dict[int, Any], k: int, stage: str, name: str, phase: str = "rest") -> Dict[str, Any]:
    entry = live[k]["live_set"]["stages"][stage].get(name)
    assert entry is not None, f"{name} was not driven at {stage}, k={k}"
    return entry[phase]


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


# A pass whose input is consumed by its first run is held on that run: the markers a reload left, then none.
PHASE = {("markers", "extension-reconcile"): "first"}


@pytest.mark.parametrize("axis, name", [pytest.param(axis, name, marks=_gap(GAPS, (axis, name)))
                                        for axis in AXES for name in PASSES])
def test_a_pass_costs_the_same_when_one_axis_of_the_live_set_grew(live, axis, name):
    phase = PHASE.get((axis, name), "rest")
    before = _rest(live, K_SMALL, STAGES[STAGES.index(axis) - 1], name, phase)
    after = _rest(live, K_SMALL, axis, name, phase)
    breaches = live_set_breaches(before["counts"], after["counts"], GROWTH[axis])
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
    """The laws above read nothing unless the set exists: the owner's Pause took every tree, the census has them all."""
    for k in (K_SMALL, K_LARGE):
        laid = live[k]["live_set"]["laid"]
        assert laid["paused"]["fences_in_queue"] == 20 and laid["paused"]["states"] == ["paused"], laid["paused"]
        assert laid["running"]["running"] == 50 and laid["pending"]["pending"] == 61, laid
        assert laid["held"] == {**laid["held"], "held": 122, "settled": 113}, laid["held"]
        assert laid["markers"]["requested"] == 120, laid["markers"]
    census = _rest(live, K_SMALL, "paused", "owner_pause_settlement", "first")["counts"]
    assert census.get("begins", 0) >= 20, "the settlement pass met the fences"
    assert _rest(live, K_SMALL, "markers", "extension-reconcile", "first")["counts"].get("stat", 0) > 0


def test_a_scan_of_the_axis_keeps_the_instrument_honest():
    """Both directions over the comparison itself: one more store read per axis item is a breach, equal counts are not."""
    assert contract_breaches({"begins": 1}, {"begins": 50}, 50) == {"begins": (1, 50)}
    assert not contract_breaches({"begins": 1}, {"begins": 3}, 50)
    assert live_rest_breaches({"writes": 1, "task_body_reads": 2}) == {"writes": 1, "task_body_reads": 2}
    assert live_rest_breaches({"begins": 3, "statements": 9}) == {}


# ---- the heartbeat (E3) --------------------------------------------------------------------------------------------

@_gap_mark(f"{LD} L4: a root heartbeat costs two store transactions (custody, then usage) on the drain")
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

@_gap_mark(f"{LB} L5: the owner's message rides every row by value (72 % of the document on 01.11)")
def test_the_published_queue_carries_the_owner_s_words_once(live):
    copies = live[K_SMALL]["live_set"]["change"]["owner_text_copies"]
    assert copies == 1, f"{copies} copies of one message in the document a restart reads"


def test_a_reader_of_the_published_queue_still_sees_the_whole_row(live):
    """However the document stores it, what a reader gets is the row, with its owner text whole."""
    assert live[K_SMALL]["live_set"]["change"]["reader_sees_owner_text"]


@_gap_mark(f"{LB} L5: documents are built, serialized and synced under Q (3 fsyncs, 3.3 MB serialized at the largest set)")
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
    assert any(name.startswith("replace queue_snapshot.json") for name in names), names
    assert facts["row_published"], facts["row_published"]


@_gap_mark(f"{LB} L5: queue.alive is synced like the documents")
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
