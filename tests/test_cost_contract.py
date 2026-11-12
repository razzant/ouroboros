"""The cost contract: boot and interactive boundaries do not grow with unrelated history.

BIBLE P5 «Dialogue must not wait for bookkeeping over unrelated history» and P13 «ordinary
enforcement does not enumerate or reconstruct unrelated completed history». Two data roots
(tests/_scale_root.py) hold the SAME current facts; the second holds 30 times the unrelated
history of the first (settled tasks with 20-200 KB bodies, child drives with their own stores,
frozen legacy result files, archive segments of other rooms newer than the requested room's
rows, closed custody records). Each boundary is driven through its real public entry point in a
fresh interpreter (tests/_cost_driver.py) and its work is counted (tests/_cost_meter.py):
connections, statements, SQLite VM steps, file-system calls, the directory entries they deliver
and bytes read by location, JSON decodes, child processes and threads. The contract is EQUALITY of those counters between the two
roots; the one allowance is ``VM_ALLOWANCE`` SQLite VM steps for a deeper B-tree (logarithmic in
the history, never a share of it). A full scan hidden inside one SELECT grows the VM steps and
fails; reading the requested rooms' own rows is the same work at every size and passes.
The explicit boot compatibility pass may walk child directories and read SQLite headers;
it has a separate bounded allowance, with no SQL, JSON decoding or frozen-history access.

Counting is not timing: this file holds no clock and proves no latency; it makes the next
history-sized reader on these paths fail by name instead of arriving silently.

Where today's code breaks the contract the test is ``xfail(strict=True)`` naming the cause, its
measured numbers and the unit that removes it, so this file is green now and turns red (XPASS)
the moment the fix lands; the marker is then removed. The k=100 variant runs in the opt-in
``integration`` lane (``-m integration -o addopts=''``). ``B6`` is the rest law of the named
maintenance jobs: a second run over unchanged inputs commits nothing and reads no history.

``B7`` is the rest law of every recurring pass of a server process: the reconciled duty
inventory ``server_maintenance.HOST_DUTIES`` (read from production, so a pass added without a row
is a pass this file never measured: tests/test_duty_inventory.py) and the delegated body's tick,
each driven in the booted driver process (tests/_cost_driver.py ``D:<duty>:<pass>``): a second
pass over unchanged inputs writes nothing, opens nothing of the history or the evidence packs and
walks or copies nothing under the delegated execution roots, with every thread, pool work item
and wake the pass dispatched joined into its count; its counts are equal at k=1 and k=30; the pass
after each ordinary activity of a live install (``tests._cost_driver.ACTIVITIES``: a task completes,
a task records one LLM attempt, a new child drive appears) costs the same at both sizes; and a
positive companion per duty (an addressed input change) shows the work still happens. ``B8`` holds
the producer and loop boundaries (``P:*``) flat on their FIRST pass. ``B9`` holds the memory's chat
reader, a whole-stream reader, to a few read transactions per page of rows, never one per row. The scale root carries the
shapes of the 22.10 class in counted numbers (tests/_scale_root.py).

``B10`` is the owner's turn: one message as the supervisor loop handles it and the production direct turn it
starts, counted up to the turn's first model request, where the driver refuses the call and the turn ends. The
history it must not reread (tests/_turn_shapes.py) is the old memory folded by parts, the archived dialogue sealed
by pages and rooms sealed after the frontier, met in the steady state a long-lived install keeps (the tail facts
stored, the OpenRouter catalogue recorded). L1: a project room's turn does the same work at k=1 and k=30; L2:
integrating turns make the same round trips apart from the stored tail's exact page reads and the recent sections'
exact archive backfill, and decode the same rows (their payload is delivered once: Main shows its story);
L3/L4 (k=1): after 40 live rooms and 10 running roots, 650 open rows, or 40 archived knowledge notes and old
rescue snapshots, only the dialogue's page reads may grow, by B9's allowance per page (their statements by the
lease's per transaction); L5: each turn reaches its request and nothing it does reaches the network; L6: the
request carries the newest open row; L7 (the build's own equalities, on a window where both sizes walk Max -> Low):
no piece or whole render of the view and no form of the request is produced twice, whole renders of the view equal the
texts delivered (the walk builds a piece's text once per key to price it and keeps only its length), whole-request
serializations equal the request's forms at every size, a capture reads what it shows and what is new, a chronicle
record is parsed once. A breach in today's code is a ``TURN_GAPS`` row (tests/_turn_law.py): a
strict xfail (non-strict for a host-dependent row) with its numbers, its remover or why it has none, and a ceiling its
growth may not pass.
The first turn's census of the host's processes is the host's, not the history's: tests/test_cost_host.py. The law's second axis,
every recurring pass against the live set rather than history, is tests/test_cost_live_set.py.
"""

from __future__ import annotations

import contextlib
import io
import json
import math
import os
import pathlib
import sqlite3
import sys
from typing import Any, Callable, Dict, Hashable, Iterable, Iterator, List, Tuple

import pytest

from tests._cost_driver import ACTIVITIES, name_owed, refused, schedule_child_job
from tests._cost_meter import EXECUTION, HISTORY, VM_STEP, CostMeter, tagged
from tests._stand_cache import STAND_MARKS, measuring
from tests._turn_law import TURN_GAPS, TurnLawBreach, hold_turn_law, hold_work_equal, readers_of, turn_gap

pytestmark = [pytest.mark.scale, *STAND_MARKS]

K_SMALL = 1
LOCATIONS = ("root", *HISTORY, EXECUTION)
# Counted per location; lock files (``lock``) are left out: how often a lock is probed and retried
# follows the contention between this process's threads at that instant, not the history. Its
# acquisitions (``lock_acquired``) do not, so they count, as do the dialogue rows decoded and every
# refused lookup of a host beyond this machine.
WORK = ("connect", "statements", "begins", "commits", "writes", "json_loads", "json_chars", "popen", "thread_start",
        "lock_acquired", "rows_decoded", "network_refused",
        *(f"{kind}:{where}" for kind in ("connect", "stat", "scandir", "listdir", "dir_entries", "open", "os_open",
                                          "read_bytes")
          for where in (*LOCATIONS, "outside")))


# Two explicit allowances, both absolute and both far below what one unit of history adds at
# k=30 (every history item exists k times, so a reader that touches one per unit adds at least 29):
# - COUNT_JITTER: the boot runs threads concurrently (store import, supervisor, boot coordinator),
#   so one read of a CURRENT fact can happen twice when one thread misses a cache another is filling;
# - BYTE_JITTER: bytes read and characters decoded differ by the digits of a timestamp or a duration
#   written during the boot itself (9.98 s vs 10.02 s); 4 KiB stays below the smallest history unit
#   here (one 20 KB settled body, one ~25 KB archive segment).
COUNT_JITTER = 2
BYTE_JITTER = 4096
BYTE_KEYS = ("json_chars", *(f"read_bytes:{where}" for where in (*LOCATIONS, "outside")))


def vm_allowance(large: int, small: int = K_SMALL) -> int:
    """SQLite VM steps a deeper index may add (logarithmic in the history, never a share of it), plus the
    progress-handler granularity of the statements ``COUNT_JITTER`` lets differ."""
    return 128 * math.ceil(math.log2(large / small)) + COUNT_JITTER * VM_STEP


def _gap(table: Dict[Any, str], name: Hashable) -> tuple:
    """A known breach named in a gap table is driven and measured all the same: a strict xfail that flips
    (XPASS) the moment its cause is gone, so the marker is removed instead of forgotten."""
    return (pytest.mark.xfail(strict=True, raises=AssertionError, reason=table[name]),) if name in table else ()


class KnownBreach(AssertionError):
    """A room gap's breach as measured: only the counters its cause grows, by no more than its row records."""


# A room read's known breach: its cause, and the k=1 -> k=30 growth of each counter it grows on this fixture's
# drives (one causal pair, measured by the exact meter: 2f8ebe37's report). A precondition, an answer that differs,
# any other counter, or more growth fails as a plain AssertionError; no growth at all is a strict XPASS.
ROOM_GAPS: Dict[str, Tuple[str, Dict[str, int]]] = {"R_late": (
    "The tool-call evidence a late task's card replays (tool_call_log.replay_evidence -> Memory.read_task_recent) "
    "reads up to the three newest tools.jsonl archives, newest first, before it reaches the task's own rows; the "
    "tools log stays a rotated file in this change (k=1 -> k=30: one more tools segment, about 39 KB).",
    {"dir_entries:archive": 203, "json_chars": 39044, "json_loads": 150, "read_bytes:archive": 39190})}


# ---- measurement -------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def built_small() -> Dict[str, Any]:
    """The first drive of the stand, over the build itself where this module comes first, as it does in collection order."""
    with measuring(K_SMALL) as counted:
        return counted


@pytest.fixture(scope="module")
def small(built_small) -> Iterator[Dict[str, Any]]:
    with measuring(K_SMALL) as counted:
        yield counted


@pytest.fixture(scope="module", params=[pytest.param(30, id="k30"),
                                        pytest.param(100, id="k100", marks=pytest.mark.integration)])
def measured(request, small) -> Iterator[Dict[Any, Any]]:
    large = request.param
    with measuring(large) as counted:
        yield {K_SMALL: small, large: counted, "large": large}


# ---- the comparison ----------------------------------------------------------------------------

def contract_breaches(small: Dict[str, int], large: Dict[str, int], k_large: int,
                      keys: Iterable[str] = WORK) -> Dict[str, Any]:
    """Every counter that grows from the small root to the large one, and VM steps beyond ``vm_allowance``.
    A count the large root spends less of is background noise in the small root's cold boot, not growth."""
    breaches: Dict[str, Any] = {key: (small.get(key, 0), large.get(key, 0)) for key in keys
                                if large.get(key, 0) - small.get(key, 0) > (
                                    BYTE_JITTER if key in BYTE_KEYS else COUNT_JITTER)}
    allowed = vm_allowance(k_large)
    if large.get("vm_steps", 0) > small.get("vm_steps", 0) + allowed:
        breaches["vm_steps"] = (small.get("vm_steps", 0), large.get("vm_steps", 0), f"allowance {allowed}")
    return breaches


def test_a_restored_stand_measures_what_its_build_measured(built_small, small):
    """``small`` drives what ``_restore`` (tests/_stand_cache.py) put back. Were that a copy, the boot would meet new inodes under
    the recorded file identities and list the history they exist to spare, and the count would say so here. A turn's counts
    (``T:``) differ between any two drives, even of two fresh builds; B10 holds them to its own law."""
    if small["root"] != built_small["root"]:
        pytest.skip("a focused run spread this module over workers: these are two private roots, not one restored")
    tags = [tag for tag, entry in small.items() if isinstance(entry, dict) and "counts" in entry and not tag.startswith("T:")]
    assert {"B1", "B2_ready", "B3", "B5", "R_late"} <= set(tags)
    for tag in tags:
        assert not contract_breaches(built_small[tag]["counts"], small[tag]["counts"], K_SMALL), tag
        assert not contract_breaches(small[tag]["counts"], built_small[tag]["counts"], K_SMALL), tag


def boundary(measured: Dict[Any, Any], tag: str, k: int) -> Dict[str, Any]:
    entry = measured[k].get(tag)
    assert entry is not None, f"boundary {tag} was not reached at k={k}"
    return entry


def child_header_breaches(entry: Dict[str, Any], children: int, root_depth: int) -> Dict[str, Any]:
    """Only directory metadata and bounded headers may grow in the boot compatibility pass.

    Each fixture child has one current store. WAL needs its header twice (mode and tier);
    Windows buffers each 100-byte read. Root resolution and confined directory handles
    also grow with the number/depth of children, never with their stored history; the one
    listing of the layout delivers one entry per child (``dir_entries``: 4 and 91 observed).
    """
    counts = entry["counts"]
    metadata = {"stat", "scandir", "listdir", "dir_entries", "open", "os_open", "read_bytes"}
    breaches = {key: value for key, value in counts.items() if value and (
        key.split(":")[0] not in metadata or key.endswith((":archive", ":task_results")))}
    limits = {"stat": (children + 1) * (2 * root_depth + 32),
              "scandir": 2 * (children + 1), "listdir": 2 * (children + 1), "dir_entries": 2 * (children + 1),
              "open": 6 * children, "os_open": 16 * (children + 1),
              "read_bytes:headless": 2 * io.DEFAULT_BUFFER_SIZE * children}
    breaches.update({key: (counts[key], limit) for key, limit in limits.items() if counts.get(key, 0) > limit})
    if counts.get("read_bytes:headless", 0) < 100 * children:
        breaches["missing_headers"] = (counts.get("read_bytes:headless", 0), children)
    for source, amount in entry.get("callers", {}).items():
        if source.startswith("read_bytes:") and not any(owner in source for owner in (
                "ouroboros.store._read_header", "ouroboros.store._filesystem_type")):
            breaches[source] = amount
    return breaches


def assert_publication_flat(measured: Dict[Any, Any], tag: str) -> None:
    for k in (K_SMALL, measured["large"]):
        entry = boundary(measured, tag, k)["child_headers"]
        assert not (breaches := child_header_breaches(
            entry, measured[k]["facts"]["child_drives"], len(pathlib.Path(measured[k]["root"]).parts))), breaches
    assert_flat(measured, tag)


def assert_flat(measured: Dict[Any, Any], tag: str, gap: Dict[str, int] | None = None) -> None:
    """`tag` does not grow from k=1 to the large root; with a room `gap`, it grows exactly as that gap
    records (KnownBreach), scaled to the large root's history as the gap's k=30 growth is."""
    large = measured["large"]
    answers = [{key: value for key, value in (measured[k].get("responses", {}).get(tag) or {}).items()
                if key in ("status", "echo")} for k in (K_SMALL, large)]
    assert answers[0] == answers[1], f"{tag} answered differently at the two sizes: {answers}"  # a failure is not flat
    big = boundary(measured, tag, large)
    little = boundary(measured, tag, K_SMALL)
    breaches = contract_breaches(little["counts"], big["counts"], large)
    touched = sorted(big.get("paths", {}).items(), key=lambda item: -item[1])[:8]
    growing = sorted(((source, amount - little.get("callers", {}).get(source, 0))
                      for source, amount in big.get("callers", {}).items()
                      if amount > little.get("callers", {}).get(source, 0)), key=lambda item: -item[1])[:8]
    if gap is not None:
        excused = {key: value for key, value in breaches.items()
                   if key in gap and value[1] - value[0] <= gap[key] * (large - K_SMALL) / 29}
        breaches = {key: value for key, value in breaches.items() if key not in excused}
    assert not breaches, (f"{tag} grows with unrelated history (k={K_SMALL} -> k={large}): {breaches}; "
                          f"history touched at k={large}: {touched}; growing readers: {growing}")
    if gap is not None and excused:
        raise KnownBreach(f"{tag} grows as its gap records (k={K_SMALL} -> k={large}): {excused}")


# ---- B1-B5 and the room ------------------------------------------------------------------------

def test_b1_lifespan_before_health_is_flat(measured):
    assert_flat(measured, "B0")
    assert_flat(measured, "B1")
    assert_flat(measured, "B1h")


def test_b1_boot_steps_after_the_yield_are_flat(measured):
    assert_flat(measured, "B1c")


def test_b1_no_child_process_before_health(measured):
    for k in (K_SMALL, measured["large"]):
        assert measured[k]["popen_before_health"] == 0, (k, measured[k]["B1"]["names"])


def test_b2_publication_to_dialogue_is_flat(measured):
    assert_publication_flat(measured, "B2_dialogue")


def test_b2_publication_to_supervisor_ready_is_flat(measured):
    assert_publication_flat(measured, "B2_ready")


def _assert_state_questions(measured, tag):
    """Visible questions stay fixed; k grows delivered history, including the finished author's result."""
    from tests._scale_root import CURRENT

    expected = {"current_questions": sorted([
        {"task_id": CURRENT[key][0], "quiz_id": CURRENT[key][1], "chat_id": 1, "native": True,
         "quiz_state": state} for key, state in (("question", "open"), ("retained_question", "expired_terminal"))
    ], key=lambda row: (row["task_id"], row["quiz_id"])),
        "current_questions_complete": True, "current_questions_gaps": []}
    for k in (K_SMALL, measured["large"]):
        assert measured[k]["responses"][tag]["decisions"] == expected


def test_b3_state_cold_is_flat(measured):
    _assert_state_questions(measured, "B3")
    assert_flat(measured, "B3")


def test_b3_state_warm_is_flat(measured):
    _assert_state_questions(measured, "B3w")
    assert_flat(measured, "B3w")


def test_b4_owner_message_on_a_cold_process_is_flat(measured):
    assert_flat(measured, "B4")
    for k in (K_SMALL, measured["large"]):
        assert boundary(measured, "B4", k)["counts"].get("open:archive", 0) == 0, k


def test_b4_owner_message_on_a_warm_process_is_flat(measured):
    assert_flat(measured, "B4w")


def test_b5_stop_of_a_tree_is_flat(measured):
    assert_flat(measured, "B5")


@pytest.mark.parametrize("tag", [pytest.param(tag, marks=(pytest.mark.xfail(
    strict=True, raises=KnownBreach, reason=ROOM_GAPS[tag][0]),) if tag in ROOM_GAPS else ())
    for tag in ("R_active", "R_late", "R_activew")])
def test_room_history_is_flat(measured, tag):
    for k in (K_SMALL, measured["large"]):  # a page, never the import window's refusal
        assert measured[k]["import"]["state"] == "done" and measured[k]["responses"][tag]["status"] == 200, k
    assert_flat(measured, tag, ROOM_GAPS[tag][1] if tag in ROOM_GAPS else None)


def test_a_room_gap_excuses_only_its_own_counters_within_their_recorded_growth():
    """The refusals of `assert_flat` with a gap, on synthetic counts: only exactly the recorded breach is known."""
    gap = ROOM_GAPS["R_late"][1]

    def measured(grown, answer=200, large=30):
        root = {key: 1000 for key in (*gap, "json_loads", "stat:root")}
        return {"large": large, K_SMALL: {"R_late": {"counts": root}, "responses": {"R_late": {"status": 200}}},
                large: {"R_late": {"counts": {**root, **{key: root[key] + amount for key, amount in grown.items()}}},
                        "responses": {"R_late": {"status": answer}}}}
    with pytest.raises(KnownBreach):
        assert_flat(measured(gap), "R_late", gap)
    with pytest.raises(KnownBreach):  # a smaller history grows proportionally less
        assert_flat(measured({key: amount * 14 // 29 for key, amount in gap.items()}, large=15), "R_late", gap)
    for grown in ({**gap, "stat:root": 29}, {**gap, "json_loads": gap["json_loads"] + 1}):
        with pytest.raises(AssertionError, match="grows with unrelated history") as caught:
            assert_flat(measured(grown), "R_late", gap)
        assert not isinstance(caught.value, KnownBreach), grown
    with pytest.raises(AssertionError, match="answered differently") as caught:
        assert_flat(measured(gap, answer=500), "R_late", gap)
    assert not isinstance(caught.value, KnownBreach)
    assert assert_flat(measured({}), "R_late", gap) is None  # cause gone: the strict xfail XPASSes and fails


# ---- B9: the memory's addressed reads ------------------------------------------------------------

def test_b9_the_memory_reader_looks_rows_up_a_page_at_a_time(measured):
    """The memory's chat reader (``chat_chain.iter_rows``) is a whole-stream reader: its rows grow with the
    stream, so the flat law does not hold it; its lookups must not grow per row. The positions and legacy
    hints of each page of rows take one read each, beside the page itself. On the owner's install (24.10)
    a lookup per row cost two read transactions per open row of 241 rooms on every turn of Main, about
    50,000 reads a minute for a quarter hour after the update."""
    for k in (K_SMALL, measured["large"]):
        rows, begins = measured[k]["B9_rows"], boundary(measured, "B9", k)["counts"].get("begins", 0)
        pages = -(-rows // 500)
        # The stream's own pages (one more when it ends on a page edge), two lookups a page, its end and import check.
        assert rows > 150 * k and begins <= 3 * pages + 3, (k, rows, begins)


# ---- B10: the owner's turn up to its first model request --------------------------------------------

# Round trips: to a store, a lock, a file, a directory, a child process, a thread. A turn may not repeat one per
# unit of history; bytes and decodes are payload, which Main's view shows by design (BIBLE P1: its own story).
ROUND = ("connect", "begins", "statements", "lock_acquired", "open", "os_open", "popen", "thread_start", "scandir",
         "listdir", "dir_entries")
# After live rooms or open rows are added only the dialogue's own page reads may grow: read transactions by B9's
# allowance per page of rows the step added (no divisor), their statements by the statements per transaction the
# turn before made (the lease's, not a constant). A lock, a connection, a file or a child per room, root or row is
# a breach.
ROWS_PER_PAGE = 500  # the memory's chat reader reads rows a page at a time (``chat_chain``)
TAIL_PAGE = 2_000  # the stored tail is read a page of rows at a time (``chronicle_store._TAIL_PAGE``)
ARCHIVES_BACKFILLED = 3  # the recent sections open at most this many newest archives of a log (``jsonl_tail``)
MAIN_TURNS = ("T:main", "T:mainw", "T:main:foreign")
BREADTH_TURNS = ("T:main:broad", "T:main:rows", "T:main:archive:first", "T:main:archive")  # the k=1 root only


def at_request(measured: Dict[Any, Any], tag: str, k: int) -> Dict[str, Any]:
    entry = measured[k].get(f"{tag}@request")
    assert entry is not None, (f"{tag} never reached its first model request at k={k}: "
                               f"{measured[k].get('responses', {}).get(tag)}; "
                               f"model calls {measured[k].get('model_calls')}")
    return entry


def _growth(before: Dict[str, Any], after: Dict[str, Any], keys: Iterable[str]) -> Dict[str, int]:
    return {key: after["counts"].get(key, 0) - before["counts"].get(key, 0) for key in keys}


def _round_trips(measured: Dict[Any, Any], tag: str, k: int) -> Dict[str, Any]:
    """``tag``'s count at its request at size ``k`` apart from two terms of the root, each checked exactly first at the
    sizes pinned above. The stored tail's page reads: one statement per ``TAIL_PAGE`` stored rows plus the empty end
    page. That is the whole-tail read, which grows with the stored tail for every integrating turn; L7d names it as
    debt for Main's first turns only. The recent sections' archive backfill (``Memory.read_task_recent``): a turn's own
    task is new, so each log they read opens its newest archives, as many as the boot's listing holds up to
    ``ARCHIVES_BACKFILLED`` (the small root holds two of the tools log, a larger one dozens). Both sizes are pinned
    here, not read from the product: a smaller page or a larger backfill is more work per unit of history."""
    from ouroboros.chronicle_store import _TAIL_PAGE
    from ouroboros.jsonl_tail import ARCHIVE_BACKFILL_MAX

    assert (_TAIL_PAGE, ARCHIVE_BACKFILL_MAX) == (TAIL_PAGE, ARCHIVES_BACKFILLED), (
        "L2 sets the stored tail's pages and the archive backfill apart at the sizes pinned here; a smaller page or a "
        "larger backfill is more work per unit of history, so a change of either is judged here, not followed",
        _TAIL_PAGE, ARCHIVE_BACKFILL_MAX)
    entry = at_request(measured, tag, k)
    work = entry["view"]["work"]
    stored = work["rows_read"] - work["rows_new"]
    pages = sum(amount for source, amount in entry["callers"].items()
                if source.startswith("statements <- ") and "ouroboros.chronicle_store._tail_rows" in source.split(" <- "))
    assert pages == -(-stored // TAIL_PAGE) + 1, ("stored tail pages", pages, stored, TAIL_PAGE)
    held = measured[k]["turn_shapes"]["steady"]["listings"]["archives"]
    backfill = sum(amount for source, amount in entry["callers"].items() if source.startswith("open:archive <- ")
                   and "ouroboros.memory.read_task_recent" in source.split(" <- "))
    assert backfill == sum(min(ARCHIVES_BACKFILLED, count) for count in held.values()), (
        "archive backfill", backfill, held, {source: amount for source, amount in entry["paths"].items()
                                             if source.startswith("open archive/")})
    counts = {**entry["counts"]}
    counts["statements"], counts["open"] = counts["statements"] - pages, counts["open"] - backfill
    return {**entry, "counts": counts}


@pytest.mark.parametrize("tag", [pytest.param(tag, marks=turn_gap("L2", tag))
                                 for tag in (*MAIN_TURNS, "T:room")])
def test_b10_an_integrating_turn_makes_the_same_round_trips_at_every_size(measured, tag):
    """L2. An integrating turn makes the same round trips and decodes the same dialogue rows at k=1 and k=30:
    Main's (the process's first, the next one, one after activity elsewhere) and a project room's direct turn,
    which integrates as a root task does (BIBLE P1: the whole horizon stays resident there, so its view holds
    every room's story and a line per live room). Its payload is delivered once: an integrating view shows its story. The
    stored tail's pages and the recent sections' archive backfill are checked exactly and apart (``_round_trips``).
    The transactions ``COUNT_JITTER`` lets come and go bring their statements, as in L3; a read per unit of history
    still adds 29 transactions."""
    large = measured["large"]
    little, big = (_round_trips(measured, tag, k) for k in (K_SMALL, large))
    keys = (*ROUND, "rows_decoded")
    per_begin = little["counts"].get("statements", 0) / max(1, little["counts"].get("begins", 0))
    allowed = {**dict.fromkeys(keys, COUNT_JITTER), "statements": math.ceil(per_begin * COUNT_JITTER)}
    hold_turn_law("L2", tag, _growth(little, big, keys), allowed, readers_of(little, big), scale=(large - K_SMALL) / 29)


@pytest.mark.parametrize("law, tag, before, step", [
    pytest.param("L3", "T:main:broad", "T:mainw", "breadth", marks=turn_gap("L3", "T:main:broad")),
    pytest.param("L4", "T:main:rows", "T:main:broad", "rows", marks=turn_gap("L4", "T:main:rows")),
    pytest.param("L3", "T:main:archive", "T:main:rows", "archive", marks=turn_gap("L3", "T:main:archive"))])
def test_b10_main_s_turn_reads_pages_not_rooms_roots_or_rows(measured, law, tag, before, step):
    """L3/L4 (k=1). After 40 live rooms with story and 10 running roots (L3), 650 open rows over those rooms (L4),
    or 40 archived knowledge notes and 40 old rescue snapshots (L3, the turn after the one that met them), only
    the dialogue's own page reads may grow: read transactions by B9's allowance per page of rows the step added
    (one page more when they cross a page edge), statements by as many transactions of the turn before's
    statements per transaction; every other round trip stays within ``COUNT_JITTER``. Payload is delivered once."""
    earlier, later = at_request(measured, before, K_SMALL), at_request(measured, tag, K_SMALL)
    rows = measured[K_SMALL]["turn_shapes"][step]["rows"]
    pages = -(-rows // ROWS_PER_PAGE) + bool(rows)
    begins = COUNT_JITTER + 3 * pages
    per_begin = earlier["counts"].get("statements", 0) / max(1, earlier["counts"].get("begins", 0))
    allowed = {**dict.fromkeys(ROUND, COUNT_JITTER), "begins": begins, "statements": math.ceil(per_begin * begins)}
    hold_turn_law(law, tag, _growth(earlier, later, ROUND), allowed, readers_of(earlier, later))


@pytest.mark.parametrize("tag", [pytest.param(tag, marks=turn_gap("L5", tag))
                                 for tag in (*MAIN_TURNS, "T:room", *BREADTH_TURNS)])
def test_b10_each_turn_reaches_its_request_without_the_network(measured, tag):
    """L5. Every turn reaches its first model request at every size it is driven, and nothing it does before
    that request reaches beyond this machine (the driver refuses and counts every other host's lookup)."""
    sizes = (K_SMALL,) if tag in BREADTH_TURNS else (K_SMALL, measured["large"])
    refused = {k: at_request(measured, tag, k)["counts"].get("network_refused", 0) for k in sizes}
    hosts = {k: [name for name in at_request(measured, tag, k)["names"] if name.startswith("network_refused")]
             for k in sizes}
    hold_turn_law("L5", tag, {"network_refused": max(refused.values())}, {"network_refused": 0}, hosts)


def test_b10_main_s_turn_carries_the_newest_open_row(measured):
    """L6. Counts alone pass a cache that drops data: the request of Main's turn after the rows step carries
    the newest row that step wrote to Main."""
    needle = measured[K_SMALL]["turn_shapes"]["rows"]["main_newest"]
    assert at_request(measured, "T:main:rows", K_SMALL)["carries"] == [needle], needle


# ---- L7: one build does each piece of work once, and delivers what it did ------------------------
# The equalities the build owes on the branch a live install walks (Max -> Low, a window tight for both sizes): what the
# host rendered whole, serialized, read and parsed is what it delivered, and a piece of the view is built once per key
# to be priced. Each stands beside a turn's own receipt
# (``context_memory_view.work``); a breach in today's code is a ``TURN_GAPS`` row with its numbers.

WORK_TURNS = ("T:main", "T:mainw")  # Main's first turn of the process and the next one, at both sizes


def _build_work(measured: Dict[Any, Any], tag: str) -> Dict[int, Dict[str, int]]:
    """Each size's build work of this turn; the laws speak of the Max -> Low branch, so a turn that kept Max is a broken precondition."""
    views = {k: at_request(measured, tag, k)["view"] for k in (K_SMALL, measured["large"])}
    for k, view in views.items():
        assert view["mode_switch"] == {"from": "max", "to": "low"}, (
            f"{tag} k={k}: the build laws are about the live branch Max -> Low; this turn ran {view['mode']} "
            f"(window {view['window_tokens']}, switch {view['mode_switch']})")
    return {k: view["work"] for k, view in views.items()}


def _pairs(works: Dict[int, Dict[str, int]], verb: str, delivered: Callable[[Dict[str, int]], int]) -> Dict[str, Tuple[int, int]]:
    return {f"{verb}(k={k})": (work.get(verb, 0), delivered(work)) for k, work in works.items()}


@pytest.mark.parametrize("tag", [pytest.param(tag, marks=turn_gap("L7a", tag)) for tag in WORK_TURNS])
def test_b10_a_build_never_repeats_a_view_form_or_a_request_form(measured, tag):
    """L7a. No piece of the view is built twice, no form of it is rendered whole twice and no form of the request is
    serialized twice in one build: every ``_again`` is zero. A repeated piece is the class of 222 story renders per
    build, in one equality. A delivered form is built twice by design, once as a piece and once whole (L7b)."""
    works = _build_work(measured, tag)
    hold_work_equal("L7a", tag, {**_pairs(works, "view_renders_again", lambda _work: 0),
                                 **_pairs(works, "view_pieces_again", lambda _work: 0),
                                 **_pairs(works, "request_serializations_again", lambda _work: 0)})


@pytest.mark.parametrize("tag", [pytest.param(tag, marks=turn_gap("L7b", tag)) for tag in WORK_TURNS])
def test_b10_a_build_renders_the_view_it_delivers(measured, tag):
    """L7b. Whole renders of the view equal the texts this start delivers (a story and a room for the projection it
    ships). The walk prices a form by its pieces: each piece's text is built once per key to take its length and is
    dropped (``view_pieces``, L7a), and only the delivered texts are built whole (``view_renders``)."""
    hold_work_equal("L7b", tag, _pairs(_build_work(measured, tag), "view_renders", lambda work: work.get("view_texts_kept", 0)))


@pytest.mark.parametrize("tag", [pytest.param(tag, marks=turn_gap("L7c", tag)) for tag in WORK_TURNS])
def test_b10_a_build_serializes_the_request_once_per_form_at_every_size(measured, tag):
    """L7c. The whole request is serialized once for each of its forms and the number of forms does not grow with the
    history: a form is a frame's shape (mode, schemas, whether memory is present), never a probe of the walk."""
    works = _build_work(measured, tag)
    hold_work_equal("L7c", tag, {**_pairs(works, "request_serializations", lambda work: work.get("request_forms", 0)),
                                 "request_serializations(k=30 vs k=1)": (works[measured["large"]].get("request_serializations", 0),
                                                                         works[K_SMALL].get("request_serializations", 0))})


@pytest.mark.parametrize("tag", [pytest.param(tag, marks=turn_gap("L7d", tag)) for tag in WORK_TURNS])
def test_b10_a_capture_reads_what_it_shows_and_what_is_new(measured, tag):
    """L7d. The capture reads what it shows and what is new: every row of the tail read is open somewhere or arrived
    since the stored facts. The one named debt of this class (``TURN_GAPS``): whole tail reads stay until the exact
    skip of sealed rows is designed."""
    works = _build_work(measured, tag)
    hold_work_equal("L7d", tag, _pairs(works, "rows_read", lambda work: work.get("rows_open", 0) + work.get("rows_new", 0)),
                    scale=(measured["large"] - K_SMALL) / 29)


@pytest.mark.parametrize("tag", [pytest.param(tag, marks=turn_gap("L7e", tag)) for tag in WORK_TURNS])
def test_b10_a_capture_parses_each_record_once(measured, tag):
    """L7e. A chronicle record is parsed once per snapshot: ``records_parsed_again`` is zero."""
    hold_work_equal("L7e", tag, _pairs(_build_work(measured, tag), "records_parsed_again", lambda _work: 0))


def test_the_work_law_fails_a_breach_no_gap_names_and_excuses_only_its_own(monkeypatch):
    """``hold_work_equal`` both ways, on synthetic counts: equal work passes; a breach without a row fails for real
    (a plain AssertionError); a row excuses only the pairs it names, within their recorded work."""
    assert hold_work_equal("Lx", "tag", {"verb(k=1)": (3, 3)}) is None
    with pytest.raises(AssertionError, match="unexcused") as plain:
        hold_work_equal("Lx", "tag", {"verb(k=1)": (4, 3)})
    assert not isinstance(plain.value, TurnLawBreach)
    monkeypatch.setitem(TURN_GAPS, ("Lx", "tag"), ("known", {"verb": 9}))
    with pytest.raises(TurnLawBreach):
        hold_work_equal("Lx", "tag", {"verb(k=1)": (9, 3)})
    for pairs in ({"verb(k=1)": (10, 3)}, {"other(k=1)": (2, 1)}):  # past its ceiling; a pair it does not name
        with pytest.raises(AssertionError, match="unexcused") as caught:
            hold_work_equal("Lx", "tag", pairs)
        assert not isinstance(caught.value, TurnLawBreach)


def test_a_host_dependent_gap_is_a_non_strict_xfail_that_still_fails_other_growth():
    """The L3 broad row's excused growth depends on the host's own sections (a macOS development host: open +3; the Ubuntu
    runner holds the law), so its xfail is not strict; every other row stays strict, and on that row growth it does
    not name, or past its ceiling, still fails for real (a plain AssertionError)."""
    from tests._turn_law import HOST_DEPENDENT_GAPS
    assert HOST_DEPENDENT_GAPS == {("L3", "T:main:broad")}
    for law, tag in TURN_GAPS:
        [mark] = turn_gap(law, tag)
        assert mark.kwargs["raises"] is TurnLawBreach and mark.kwargs["strict"] is ((law, tag) not in HOST_DEPENDENT_GAPS)
    allowed = {"open": 0, "read_bytes": 0}
    with pytest.raises(TurnLawBreach):
        hold_turn_law("L3", "T:main:broad", {"open": 3, "read_bytes": 0}, allowed, [])
    for growth in ({"open": 4, "read_bytes": 0}, {"open": 0, "read_bytes": 1}):  # past its ceiling; a counter not named
        with pytest.raises(AssertionError, match="grew past its gap") as caught:
            hold_turn_law("L3", "T:main:broad", growth, allowed, [])
        assert not isinstance(caught.value, TurnLawBreach)


# ---- B6: the rest law of the named maintenance jobs ---------------------------------------------


@pytest.mark.parametrize("growth", ["other-rooms", "destination"])
def test_message_dating_reads_indexed_candidates_and_only_the_binding_delta(tmp_path, growth):
    """No ANALYZE statistics: unrelated dialogue cannot grow a room probe, and previously admitted
    conversation cannot grow a new binding's dating. Count the real boundary, including its SQL VM work."""
    from ouroboros import dialogue_store as ds, projects_registry as registry, room_messages, store
    from tests._dialogue import import_all, seed

    measured = {}
    for count in (20, 4000):
        root = tmp_path / str(count)
        room = registry.create_project(root, "cost", name="Cost")["chat_id"]
        # Fixture bindings: real saves, but before the dialogue exists, so setup does no dating.
        for index in range(12 if count == 20 or growth == "other-rooms" else 100):
            registry.bind_task_to_project(root, f"old-{index}", "cost", room, origin={"absent": "system"})
        rows = [{"chat_id": room, "ts": "2020-01-01T06:00:00Z", "direction": "in", "text": "Room question"}]
        rows += [{"chat_id": room if growth == "destination" else 1, "ts": "2020-01-01T05:00:00Z",
                  "direction": "out", "task_id": "old-0" if growth == "destination" else "unrelated",
                  "text": f"Old row {index}"} for index in range(count)]
        rows += [{"chat_id": 1, "ts": "2020-01-01T09:00:00Z", "direction": "out",
                  "task_id": "incoming", "text": "Work before binding"}]
        seed(root, archives={"chat_20200101T000000.jsonl": rows})
        import_all(root)
        room_messages.run(root, lambda: False)
        with ds.transaction(root, write=False) as conn:
            assert not conn.execute("SELECT 1 FROM sqlite_master WHERE name='sqlite_stat1'").fetchone()
        store.close_connections_under(root)  # the meter installs callbacks on freshly opened handles
        with CostMeter(root) as meter:
            if growth == "other-rooms":
                assert tagged("newest", lambda: room_messages.newest(root, room)) == "2020-01-01T06:00:00Z"
            tagged("binding", lambda: registry.bind_task_to_project(
                root, "incoming", "cost", room, origin={"absent": "system"}))
        assert room_messages.read(root)[0][room] == "2020-01-01T09:00:00Z"
        tags = ["newest", "binding"] if growth == "other-rooms" else ["binding"]
        measured[count] = {tag: meter.snapshot(tag)["counts"] for tag in tags}
    for tag in measured[20]:
        small, large = measured[20][tag], measured[4000][tag]
        assert small.get("vm_steps", 0) > 0 and large.get("vm_steps", 0) > 0, measured
        assert large.get("vm_steps", 0) <= small.get("vm_steps", 0) + vm_allowance(200), measured
        assert large.get("rows_decoded", 0) <= small.get("rows_decoded", 0), measured
        assert large.get("statements:dialog", 0) <= small.get("statements:dialog", 0) + 2, measured
    print("MESSAGE_DATING_COST " + json.dumps({"growth": growth, "counts": measured}, sort_keys=True))

def second_run(run: Callable[[pathlib.Path], Any], root: pathlib.Path) -> Dict[str, int]:
    """The counters of a job's second run over the inputs its first run left. Each run returns its report, read
    outside the count: a refused run did nothing, so no count of it is a rest (``refused``; #2465)."""
    first = run(root)
    assert not refused(first), ("the first run", refused(first), first)
    with CostMeter(root) as meter:
        second = tagged("rest", lambda: run(root))
    assert not refused(second, rest=True), ("the second run", refused(second, rest=True), second)
    return meter.snapshot("rest")["counts"]


def rest_breaches(counts: Dict[str, int]) -> Dict[str, int]:
    """A second run writes nothing (a read transaction of the store also ends with COMMIT, so the
    written statements are what counts), opens or reads nothing of the history or the evidence packs,
    and walks or copies nothing under the delegated execution roots: there a ``stat`` is a walk and
    the open of a source is a copy (the bytes move in the kernel), so even one ``lexists`` per held
    candidate breaches, by design — the attempt's outcome belongs on the custody row, not on disk.
    Looking whether a history directory is empty is allowed; that it does not grow is the k-equality's to check."""
    keys = ("writes", *(f"{kind}:{where}" for kind in ("open", "os_open", "read_bytes") for where in HISTORY),
            *(f"{kind}:{EXECUTION}" for kind in ("stat", "scandir", "listdir", "open", "os_open", "read_bytes")))
    return {key: counts[key] for key in keys if counts.get(key)}


def _maintenance():
    try:
        from ouroboros import maintenance
    except ImportError:
        return None
    return maintenance


# Every job the maintenance child runs: its own registry.
JOB_NAMES = tuple(getattr(_maintenance(), "JOBS", {}))


def job_runner(name: str) -> Callable[[pathlib.Path], Any]:
    """One named job's run in this process (``maintenance.run_job``), reported as the child's pass reports it."""
    from tests._store_shared import as_install

    maintenance = _maintenance()

    def run(root: pathlib.Path) -> Any:
        with as_install(root):  # the maintenance child runs as the install it maintains
            outcome = maintenance.run_job(root, name, should_stop=lambda: False)
        return {"status": "completed", "jobs": {} if outcome is None else {name: outcome}}
    return run


def pass_runner() -> Callable[[pathlib.Path], Any]:
    """The child's whole pass in this process (``maintenance.run``), as the install it maintains, like a job: from
    any other install every job is refused (``StoreForeign``) and the pass costs almost nothing (#2465)."""
    from tests._store_shared import as_install

    maintenance = _maintenance()
    if maintenance is None or not callable(getattr(maintenance, "run", None)):
        pytest.skip("the maintenance child's pass does not exist on this tree yet")

    def run(root: pathlib.Path) -> Any:
        with as_install(root):
            return maintenance.run(root, should_stop=lambda: False)
    return run


def _rests_at_every_size(measured: Dict[Any, Any], run: Callable[[pathlib.Path], Any], what: str) -> None:
    large = measured["large"]
    rested = {k: second_run(run, pathlib.Path(measured[k]["root"])) for k in (K_SMALL, large)}
    for k, counts in rested.items():
        assert not rest_breaches(counts), (what, k, rest_breaches(counts))
    assert not contract_breaches(rested[K_SMALL], rested[large], large), (what, rested)


def run_after_activity(run: Callable[[pathlib.Path], Any], root: pathlib.Path, what: str, phase: str,
                       k: int) -> Dict[str, int]:
    """The counters of a job's run after it rested and one ordinary activity nobody here owns happened
    (``ACTIVITIES[phase]``, the driver's own shapes, as the install it maintains); reports read as ``second_run``'s."""
    from tests._store_shared import as_install

    for before in (run(root), run(root)):  # it rested
        assert not refused(before), (phase, what, "before the activity", refused(before), before)
    with as_install(root):
        ACTIVITIES[phase](root, f"job-{what}", k)
    with CostMeter(root) as meter:
        after = tagged(phase, lambda: run(root))
    assert not refused(after), (phase, what, refused(after), after)
    name_owed(f"{phase} after {what}", after)  # owed work left pending: not refused, debt the next pass inherits
    return meter.snapshot(phase)["counts"]


def _same_cost_after_activity(measured: Dict[Any, Any], run: Callable[[pathlib.Path], Any], what: str,
                              phase: str) -> None:
    large = measured["large"]
    after = {k: run_after_activity(run, pathlib.Path(measured[k]["root"]), what, phase, k) for k in (K_SMALL, large)}
    assert not contract_breaches(after[K_SMALL], after[large], large), (phase, what, after)


@pytest.mark.parametrize("name", JOB_NAMES)
def test_b6_each_job_rests_on_its_second_run(measured, name):
    _rests_at_every_size(measured, job_runner(name), name)


def test_b6_the_maintenance_pass_rests_on_its_second_run(measured):
    """The whole pass, as a job: its first run does the work a seeded due job owes (the child's own scheduled job,
    ``schedule_child_job``), whose row keeps it done (read outside the count); its second run rests, so it did not."""
    from ouroboros import maintenance_jobs
    from tests._store_shared import as_install

    roots = [pathlib.Path(measured[k]["root"]) for k in (K_SMALL, measured["large"])]
    for root in roots:
        with as_install(root):
            seeded = schedule_child_job(root)
    _rests_at_every_size(measured, pass_runner(), "maintenance.run")  # each run's own report is read there
    for root in roots:
        with as_install(root):
            assert (maintenance_jobs.job(root, seeded) or {}).get("state") == "done", (root, seeded)


# The install is never idle: between two passes one ordinary activity nobody here owns happens
# (``tests._cost_driver.ACTIVITIES``): a task completes (``unrelated``), a task records one LLM attempt
# (``money``), a new child drive appears (``drive``). A job or duty may do the addressed work the activity
# owes it, and its cost is the same at k=1 and k=30; one whose rest keys on an install-wide fact (the store
# revision, the terminal-write counter, the usage ledger's marker, a layout directory's time) redoes
# history-sized work instead. A breach still in the code is named here by (activity, name) with its
# measured numbers and its remover; the marker flips the moment the cause is gone.
ACTIVITY_JOB_GAPS: Dict[Tuple[str, str], str] = {}
ACTIVITY_DUTY_GAPS: Dict[Tuple[str, str], str] = {}


def _activity_params(table: Dict[Tuple[str, str], str], names: Iterable[str]) -> list:
    return [pytest.param(phase, name, marks=_gap(table, (phase, name))) for phase in ACTIVITIES for name in names]


@pytest.mark.parametrize("phase,name", _activity_params(ACTIVITY_JOB_GAPS, JOB_NAMES))
def test_b6_each_job_costs_the_same_after_unrelated_activity(measured, phase, name):
    _same_cost_after_activity(measured, job_runner(name), name, phase)


@pytest.mark.parametrize("phase,name", _activity_params(ACTIVITY_JOB_GAPS, ("maintenance.run",)))
def test_b6_the_maintenance_pass_costs_the_same_after_unrelated_activity(measured, phase, name):
    _same_cost_after_activity(measured, pass_runner(), name, phase)


# ---- B7: the rest law of every recurring pass; B8: the producer and loop boundaries --------------------

def _host_duty_names() -> tuple:
    """The dispatcher's inventory; on an older body without the declaration, the passes its dispatcher
    started (the driver's shim for the negative control on the body that broke the owner's install)."""
    try:
        from ouroboros import server_maintenance
    except ImportError:
        return ()
    from tests._cost_driver import _duty_rows

    return tuple(duty.name for duty in _duty_rows(server_maintenance))


DUTY_NAMES = (*_host_duty_names(), "delegated-body")  # the dispatcher's inventory plus the lifespan's body tick

# Where today's code still breaks the law: the measured cause (k=1 -> k=30, the readers that grow) and
# the change that removes it. A duty or boundary named here is driven and measured all the same; the
# marker turns the known breach into a strict xfail that flips the moment the cause is gone.
REST_LAW_GAPS: Dict[str, str] = {}
BOUNDARIES = ("P:metrics_run", "P:metrics_review", "P:send", "P:allowance")
BOUNDARY_GAPS: Dict[str, str] = {}
# The counter a duty's companion makes appear (work > rest at every size); the body's walk and copy are
# the two execution-root counters the 22.10 control fails on at rest. A copy opens its source through
# either door: ``io.open`` (``shutil.copy2``, the 22.10 body) or ``os.open`` (the exclusive byte copy of
# ``review_checking._retain_files``); the counters of one tuple are summed.
COPY = ("open:execution", "os_open:execution")
COMPANION_COUNTERS = {
    "delegated-body": ("scandir:execution", COPY, "writes"),
    "abandoned_usage_reconciliation": ("writes",),
    "custody-maintenance": ("writes",),
    "reconcile-maintenance": ("writes",),
    "maintenance-child": ("child_spawn",),
    "question_delivery_replay": ("writes",),
    "terminal_delivery_replay": ("open",),
    "cancel_intent_sweep": ("writes",),
    "owner_pause_settlement": ("writes",),
    "cold-custody": ("writes",),
    "schedule-admission": ("writes",),
    "held_continuation_recheck": ("writes",),
    "pause_notice_replay": ("writes",),
    "post_task_synthesis_recovery": ("writes",),
}


def _duty_pass(measured: Dict[Any, Any], name: str, k: int, phase: str) -> Dict[str, Any]:
    entry = measured[k].get("duties", {}).get(name)
    assert entry is not None, f"{name} was not driven at k={k}"
    snapshot = entry.get(phase)
    assert snapshot is not None, f"{name} has no {phase} pass at k={k} (companion: {entry.get('companion')!r})"
    # Each child reports under its pass's tag (``_drain``): refused, that pass measured nothing (#2465); pending is owed
    # work, refused in the counted rest, named elsewhere. The rest answers for its preparation's refusals, not pending.
    reports = measured[k].get("child_reports", {})
    tags = [phase, *(p for p in entry if phase == "rest" and (p == "first" or p.startswith("settle")))]
    children = [(f"D:{name}:{tag}", lines) for tag in tags for report in reports.get(f"D:{name}:{tag}", [])
                if (lines := refused(report, rest=tag == "rest"))]
    assert not children, (name, k, children)
    name_owed(f"D:{name}:{phase} at k={k}", *reports.get(f"D:{name}:{phase}", []))
    return snapshot


def _writes(snapshot: Dict[str, Any]) -> List[str]:
    return [text for text in snapshot.get("statements", {})
            if text.split(None, 1)[0].upper() in ("INSERT", "UPDATE", "DELETE", "REPLACE")][:6]


def _growing(little: Dict[str, Any], big: Dict[str, Any]) -> List[tuple]:
    return sorted(((source, amount - little.get("callers", {}).get(source, 0))
                   for source, amount in big.get("callers", {}).items()
                   if amount > little.get("callers", {}).get(source, 0)), key=lambda item: -item[1])[:8]


def _companion_counter(measured: Dict[Any, Any], name: str, k: int, phase: str, counter: Any) -> int:
    if counter == "child_spawn":  # the child's spawn is recorded under the boundary that requested the look
        _duty_pass(measured, name, k, phase)  # and the child did that pass's jobs
        return measured[k].get("child_spawns", []).count(f"D:{name}:{phase}")
    if isinstance(counter, tuple):  # one kind of work reached through several doors
        return sum(_companion_counter(measured, name, k, phase, door) for door in counter)
    return _duty_pass(measured, name, k, phase)["counts"].get(counter, 0)


@pytest.mark.parametrize("name", [pytest.param(name, marks=_gap(REST_LAW_GAPS, name)) for name in DUTY_NAMES])
def test_b7_each_duty_rests_on_its_second_pass(measured, name):
    """Second pass, inputs unchanged, the duty's own clock advanced: zero writes, zero history or pack reads,
    zero execution-root walks or copies, and the same counts (VM steps included) at k=1 and k=30."""
    large = measured["large"]
    rested = {k: _duty_pass(measured, name, k, "rest") for k in (K_SMALL, large)}
    for k, snapshot in rested.items():
        breaches = rest_breaches(snapshot["counts"])
        assert not breaches, (name, k, breaches, _writes(snapshot), sorted(snapshot.get("paths", {}).items())[:8])
    breaches = contract_breaches(rested[K_SMALL]["counts"], rested[large]["counts"], large)
    assert not breaches, (name, breaches, _growing(rested[K_SMALL], rested[large]))


@pytest.mark.parametrize("phase,name", _activity_params(ACTIVITY_DUTY_GAPS, DUTY_NAMES))
def test_b7_each_duty_costs_the_same_after_unrelated_activity(measured, phase, name):
    """One ordinary activity nobody here owns happened between the previous pass and this one (a task
    completed, a task recorded one LLM attempt, a new child drive appeared): the duty may do the addressed
    work the activity owes it, and its counts are the same at k=1 and k=30."""
    large = measured["large"]
    after = {k: _duty_pass(measured, name, k, phase) for k in (K_SMALL, large)}
    breaches = contract_breaches(after[K_SMALL]["counts"], after[large]["counts"], large)
    assert not breaches, (phase, name, breaches, _growing(after[K_SMALL], after[large]))


@pytest.mark.parametrize("name", sorted(COMPANION_COUNTERS))
def test_b7_each_duty_works_again_when_its_input_changes(measured, name):
    """The rest is not death: after one addressed input change the duty does the work and its counts appear."""
    for k in (K_SMALL, measured["large"]):
        for counter in COMPANION_COUNTERS[name]:
            rest, work = (_companion_counter(measured, name, k, phase, counter) for phase in ("rest", "work"))
            assert work > rest, (name, k, counter, rest, work, measured[k]["duties"][name].get("companion"))


def test_b7_every_duty_has_a_companion():
    assert set(DUTY_NAMES) == set(COMPANION_COUNTERS)


@pytest.mark.parametrize("tag", [pytest.param(tag, marks=_gap(BOUNDARY_GAPS, tag)) for tag in BOUNDARIES])
def test_b8_producer_and_loop_boundaries_are_flat_on_the_first_pass(measured, tag):
    assert_flat(measured, tag)


# ---- the instrument itself, both directions --------------------------------------------------------

def _rooms(root: pathlib.Path, rows: int) -> pathlib.Path:
    db = root / "state" / "current.sqlite"
    db.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE rows (id INTEGER PRIMARY KEY, room INTEGER, body TEXT)")
    conn.execute("CREATE INDEX rows_room ON rows(room)")
    conn.executemany("INSERT INTO rows (room, body) VALUES (?, ?)",
                     [(7 if index < 5 else 1000 + index, "x" * 40) for index in range(rows)])
    conn.commit()
    conn.close()
    return db


@pytest.mark.parametrize("hidden_scan", [False, True])
def test_the_contract_catches_a_scan_hidden_in_one_select(tmp_path, hidden_scan):
    """The same requested rows read through an index stay flat; read through a predicate no index serves,
    the one SELECT walks the whole table and the VM steps grow with it."""
    query = ("SELECT body FROM rows WHERE room + 0 = 7" if hidden_scan else "SELECT body FROM rows WHERE room = 7")
    counted = {}
    for k in (K_SMALL, 30):
        db = _rooms(tmp_path / f"k{k:03d}", 200 * k)
        with CostMeter(tmp_path / f"k{k:03d}") as meter:
            def read():
                conn = sqlite3.connect(db)
                assert len(conn.execute(query).fetchall()) == 5
                conn.close()
            tagged("read", read)
        counted[k] = meter.snapshot("read")["counts"]
    assert counted[K_SMALL]["statements"] == counted[30]["statements"] == 1
    assert bool(contract_breaches(counted[K_SMALL], counted[30], 30)) is hidden_scan, counted


@pytest.mark.parametrize("walks_history", [False, True])
def test_the_contract_catches_a_history_walk(tmp_path, walks_history):
    """A reader that opens only the requested segment is flat; one that walks every segment is not."""
    counted = {}
    for k in (K_SMALL, 30):
        archive = tmp_path / f"k{k:03d}" / "archive"
        archive.mkdir(parents=True)
        for index in range(k):
            (archive / f"chat_{index:04d}.jsonl").write_text('{"chat_id": 9}\n' * 20, encoding="utf-8")
        (archive / "chat_0000.jsonl").write_text('{"chat_id": 7}\n' * 20, encoding="utf-8")

        def read(archive=archive):
            names = sorted(os.listdir(archive)) if walks_history else ["chat_0000.jsonl"]
            for name in names:
                with open(archive / name, encoding="utf-8") as handle:
                    [json.loads(line) for line in handle]

        with CostMeter(tmp_path / f"k{k:03d}") as meter:
            tagged("read", read)
        counted[k] = meter.snapshot("read")["counts"]
    assert bool(contract_breaches(counted[K_SMALL], counted[30], 30)) is walks_history, counted


def _listing(door: str) -> Callable[[pathlib.Path], int]:
    """One reader per door, each ONE call over the directory; ``probe`` examines one entry of it."""
    def scandir(directory):
        with os.scandir(directory) as entries:
            return sum(1 for _entry in entries)

    def listdir(directory):
        return len(os.listdir(directory))

    def iterdir(directory):
        return sum(1 for _entry in pathlib.Path(directory).iterdir())

    def walk(directory):
        return sum(len(files) for _root, _dirs, files in os.walk(directory))

    def probe(directory):  # whether the directory is empty: the same one entry at every size
        with os.scandir(directory) as entries:
            return int(next(entries, None) is not None)
    return {"scandir": scandir, "listdir": listdir, "iterdir": iterdir, "walk": walk, "probe": probe}[door]


@pytest.mark.parametrize("door", ["scandir", "listdir", "iterdir", "walk", "probe"])
def test_the_contract_catches_a_history_sized_listing_made_in_one_call(tmp_path, door):
    """A listing costs the entries it delivers, not the calls it took: ONE ``scandir`` or ``listdir``
    over a history directory (``os.walk`` and ``Path.iterdir`` included) is the same one call at every
    size and grows with the history all the same; looking whether the directory is empty examines one
    entry and stays flat."""
    counted = {}
    for k in (K_SMALL, 30):
        archive = tmp_path / f"k{k:03d}" / "archive"
        archive.mkdir(parents=True)
        for index in range(k):
            (archive / f"chat_{index:04d}.jsonl").write_text("{}\n", encoding="utf-8")
        with CostMeter(tmp_path / f"k{k:03d}") as meter:
            tagged("read", lambda archive=archive: _listing(door)(archive))
        counted[k] = meter.snapshot("read")["counts"]
    calls = [{key: counts.get(key, 0) for key in ("scandir", "listdir")} for counts in counted.values()]
    assert calls[0] == calls[1] and sum(calls[0].values()) == 1, calls  # the call count alone sees nothing
    breaches = contract_breaches(counted[K_SMALL], counted[30], 30)
    assert (set(breaches) == {"dir_entries:archive"}) is (door != "probe"), (door, breaches, counted)
    expected = (1, 30) if door != "probe" else (1, 1)
    assert (counted[K_SMALL]["dir_entries:archive"], counted[30]["dir_entries:archive"]) == expected


@pytest.mark.parametrize(("look_ends", "story", "refusing"), [
    ("inside_the_pass", "done done done", ""),
    ("inside_the_pass", "pending done pending", ""),
    ("inside_the_pass", "raised done done", "first rest"),
    ("during_the_drain", "done pending done", "rest"),
    ("during_the_drain", "failed done done", "first rest"),
])
def test_the_driver_measures_the_child_a_pass_spawned_whenever_its_look_ends(tmp_path, monkeypatch, recwarn, look_ends,
                                                                             story, refusing):
    """A pass whose look spawned the child owns the child's jobs: the driver runs them under the pass's
    tag whether the look's thread ended inside the pass (its spawn recorded before the drain began) or
    ends while the drain joins it. A spawn baseline read after the pass takes the early spawn for an
    old one and measures no child jobs, with no error to show for it. B7 reads what each pass's child (``story``:
    first, rest, unrelated) reported on that pass alone: a refusal fails it and the rest it prepared; pending fails
    the counted rest only, elsewhere it is owed work, named, that a later pass may do."""
    import threading
    import time
    from types import SimpleNamespace

    from ouroboros import maintenance
    from tests._cost_driver import _measured_pass
    from tests._cost_meter import tag_of

    runs, states = [], dict(zip(("D:duty:first", "D:duty:rest", "D:duty:unrelated"), story.split()))

    def run(_root, **_kwargs):
        runs.append(tag_of())
        if states[runs[-1]] == "raised":
            raise RuntimeError("the child's pass died")
        return {"status": "completed", "jobs": {"custody_trash": {"state": states[runs[-1]], "items": 0}}}
    monkeypatch.setattr(maintenance, "run", run)
    monkeypatch.setattr(maintenance, "child", SimpleNamespace(owed=lambda: False))
    dispatcher = SimpleNamespace(_DELIVERY_WAKE=threading.Event(), _tick_maintenance_child=lambda: None)
    result: Dict[str, Any] = {"errors": [], "child_spawns": ["D:earlier:first"], "_started": time.monotonic()}

    def spawn():
        result["child_spawns"].append(tag_of())

    def look():
        if look_ends == "inside_the_pass":
            spawn()
        else:
            threading.Thread(target=lambda: (time.sleep(0.2), spawn()), name="maintenance-child").start()

    with CostMeter(tmp_path) as meter:
        passes = {tag.split(":")[-1]: _measured_pass(meter, tag, look, dispatcher, tmp_path, threading.Event(), result)
                  for tag in states}
    assert result["errors"] == [] and runs == list(states), (look_ends, runs, result["child_spawns"])
    for phase, state in zip(passes, story.split()):
        named, refuses = len(recwarn), phase in refusing.split()
        with pytest.raises(AssertionError, match="custody|Runtime") if refuses else contextlib.nullcontext():
            _duty_pass({K_SMALL: {**result, "duties": {"duty": passes}}}, "duty", K_SMALL, phase)  # as B7 reads it
        assert (len(recwarn) > named) is (state == "pending" and phase != "rest"), (phase, [*map(str, recwarn)])


@pytest.mark.parametrize(("mode", "extra"), [
    ("PERSIST", "none"),
    ("PERSIST", "body"),
    ("PERSIST", "missing_headers"),
    ("WAL", "sql"),
    ("WAL", "legacy_scan"),
])
def test_child_header_allowance_rejects_history_work(tmp_path, monkeypatch, extra, mode):
    from ouroboros import store, store_journal
    from tests.test_store import document, publish

    monkeypatch.setattr(store, "_journal_mode", lambda *_: mode)
    drive = tmp_path / "state/headless_tasks/child/data"
    publish(drive, document("finished", result="x" * 20_000))
    legacy = drive / "task_results"
    legacy.mkdir()
    body = legacy / "finished.json"
    body.write_text(json.dumps(document("finished", result="x" * 20_000)), encoding="utf-8")

    def prepare():
        if extra != "missing_headers":
            store_journal.prepare_children(tmp_path)
        if extra == "sql":
            with store.open_store(drive, write=False) as txn:
                txn.conn.execute("SELECT * FROM task_bodies").fetchall()
        elif extra == "body":
            body.read_bytes()  # even an undecoded history read must fail
        elif extra == "legacy_scan":
            list((tmp_path / "task_results").glob("*.json"))

    try:
        with CostMeter(tmp_path) as meter:
            tagged("headers", prepare)
        breaches = child_header_breaches(meter.snapshot("headers"), 1, len(tmp_path.parts))
        assert bool(breaches) is (extra != "none"), breaches
    finally:
        store.close_connections()


@pytest.mark.parametrize("walks", [None, "copy2", "exclusive"])
def test_the_rest_law_catches_a_walk_or_copy_under_an_execution_root(tmp_path, walks):
    """A pass that reads its own row touches no checkout; one that re-walks a declared evidence tree
    and copies a file shows the walk as ``scandir``/``stat`` and the copy as the source's open under
    ``execution``, through whichever door: ``shutil.copy2`` opens it with ``io.open`` and moves the bytes
    in the kernel (no ``read_bytes`` to count), the exclusive byte copy opens it with ``os.open``."""
    import shutil

    exec_root, data = tmp_path / "exec", tmp_path / "data"
    tree = exec_root / "run" / ".review-evidence"
    tree.mkdir(parents=True)
    data.mkdir()
    (tree / "object").write_bytes(b"x" * 10_000)
    os.chmod(tree / "object", 0o444)
    (data / "row.json").write_text("{}", encoding="utf-8")

    def second_pass():
        json.loads((data / "row.json").read_text(encoding="utf-8"))
        if walks:
            for _current, _dirs, _files in os.walk(exec_root / "run"):
                pass
        if walks == "copy2":
            shutil.copy2(tree / "object", tmp_path / "copy")
        elif walks == "exclusive":
            with os.fdopen(os.open(tree / "object", os.O_RDONLY), "rb") as src, (tmp_path / "copy").open("xb") as dst:
                shutil.copyfileobj(src, dst, 1 << 20)

    with CostMeter(data, execution_root=exec_root) as meter:
        tagged("rest", second_pass)
    counts = meter.snapshot("rest")["counts"]
    assert bool(rest_breaches(counts)) is bool(walks), counts
    if walks:
        door = "open:execution" if walks == "copy2" else "os_open:execution"
        assert counts["scandir:execution"] >= 1 and counts[door] == 1, counts
        assert sum(counts.get(key, 0) for key in COPY) == 1, counts


@pytest.mark.parametrize("by_boundary", [True, False])
def test_the_meter_credits_pool_work_to_the_boundary_that_submitted_it(tmp_path, by_boundary):
    """A pass that submits to a pool owns what the pool does for it; work a background thread submits
    stays the pool worker's own."""
    import threading
    from concurrent.futures import ThreadPoolExecutor

    pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="run-collect")
    try:
        with CostMeter(tmp_path) as meter:
            if by_boundary:
                tagged("pass", lambda: pool.submit(os.stat, tmp_path).result())
            else:
                starter = threading.Thread(target=lambda: pool.submit(os.stat, tmp_path).result(), name="bg-starter")
                starter.start()
                starter.join()
    finally:
        pool.shutdown()
    assert (meter.snapshot("pass")["counts"].get("stat", 0) == 1) is by_boundary
    assert (meter.snapshot("bg:run-collect")["counts"].get("stat", 0) == 1) is not by_boundary


@pytest.mark.parametrize("rests", [False, True])
def test_the_rest_law_catches_a_job_that_redoes_its_work(tmp_path, rests):
    """A job that records its result reads nothing the second time; one that forgets re-reads the archive."""
    archive = tmp_path / "archive"
    archive.mkdir()
    (archive / "chat_0001.jsonl").write_text('{"chat_id": 9}\n' * 20, encoding="utf-8")
    done = tmp_path / "state" / "job.done"
    done.parent.mkdir()

    def job(root: pathlib.Path) -> Dict[str, Any]:  # reported as a job is (``job_runner``)
        if not (rests and done.exists()):
            with open(root / "archive" / "chat_0001.jsonl", encoding="utf-8") as handle:
                [json.loads(line) for line in handle]
            done.write_text("done", encoding="utf-8")
        return {"status": "completed", "jobs": {}}

    assert bool(rest_breaches(second_run(job, tmp_path))) is not rests


@pytest.fixture
def contained_meter():
    """The meter tests below make installs and uninstalls fail on purpose. A regression could leave a
    meter installed; it is put back when the test ends, so the failure stays this test's own instead of
    cascading into every later meter test as "one cost meter at a time"."""
    from tests import _cost_meter

    yield
    if _cost_meter._ACTIVE is not None:
        with contextlib.suppress(Exception):  # the test's own failure is the one to report
            _cost_meter.uninstall(_cost_meter._ACTIVE)


def test_the_meter_installs_where_a_door_is_missing_and_never_records_a_wrapper_as_the_original(
        tmp_path, monkeypatch, contained_meter):
    """``os.pread`` is POSIX-only, so on Windows the meter counts without it. An install that meets a door still
    wrapped refuses and puts back every door it replaced: a wrapper recorded as the original would recurse in
    every later caller of that door, in this process and in every test after it."""
    from tests import _cost_meter

    real_stat, real_open = os.stat, open
    supported = os.supports_dir_fd
    stat_supports_dir_fd = os.stat in supported
    with monkeypatch.context() as patch:
        patch.setattr(json, "loads", _cost_meter._w_json_loads)  # left wrapped; os.stat .. open come first
        with pytest.raises(RuntimeError, match="still the meter's wrapper"):
            CostMeter(tmp_path).__enter__()
        assert os.stat is real_stat and open is real_open
    monkeypatch.delattr(os, "pread", raising=False)
    with CostMeter(tmp_path) as meter:
        assert (os.stat in os.supports_dir_fd) is stat_supports_dir_fd
        tagged("read", lambda: os.stat(tmp_path))
    assert meter.snapshot("read")["counts"]["stat"] == 1 and os.stat is real_stat
    assert os.supports_dir_fd is supported


@pytest.mark.parametrize("takes", [0, 1], ids=["rebinding_refused", "restoring_refused"])
def test_the_meter_puts_every_door_back_past_a_module_that_refuses(tmp_path, monkeypatch, takes, contained_meter):
    """A module holding a door's alias may refuse the wrapper (the install fails) or refuse to give it
    back (the uninstall fails). Either way every other door goes back, the refusal is named once,
    and the next install starts from the real doors."""
    import types

    class OneWay(types.ModuleType):
        """Takes ``takes`` assignments of its ``stat`` alias, then refuses."""

        def __setattr__(self, attr, value):
            if attr == "stat":
                if not self.__dict__["takes"]:
                    raise TypeError("stat is read-only here")
                self.__dict__["takes"] -= 1
            super().__setattr__(attr, value)

    alias = OneWay("tests._cost_meter_one_way")
    alias.__dict__.update(stat=os.stat, takes=takes)
    monkeypatch.setitem(sys.modules, alias.__name__, alias)
    real_stat, real_open = os.stat, open
    if takes:
        meter = CostMeter(tmp_path).__enter__()
        assert alias.stat is not real_stat, "the alias took the wrapper"
        with pytest.raises(RuntimeError, match=r"could not put back tests\._cost_meter_one_way\.stat"):
            meter.__exit__(None, None, None)
    else:
        with pytest.raises(TypeError, match="read-only"):
            CostMeter(tmp_path).__enter__()
    assert os.stat is real_stat and open is real_open, "every other door went back"
    monkeypatch.delitem(sys.modules, alias.__name__)
    with CostMeter(tmp_path) as meter:
        tagged("read", lambda: os.stat(tmp_path))
    assert meter.snapshot("read")["counts"]["stat"] == 1
