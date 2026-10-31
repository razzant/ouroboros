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
Main's turns make the same round trips and decode the same rows (their payload is free: Main shows its story);
L3/L4 (k=1): after 40 live rooms and 10 running roots, 650 open rows, or 40 archived knowledge notes and old
rescue snapshots, only the dialogue's page reads may grow, by B9's allowance per page (their statements by the
lease's per transaction); L5: each turn reaches its request and nothing it does reaches the network; L6: the
request carries the newest open row. A breach in today's code is a ``TURN_GAPS`` row (tests/_turn_law.py): a
strict xfail with its numbers, its remover or why it has none, and a ceiling its growth may not pass.
The first turn's census of the host's processes is the host's, not the history's: tests/test_cost_host.py.
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
from typing import Any, Callable, Dict, Hashable, Iterable, List, Tuple

import pytest

from tests._cost_driver import ACTIVITIES
from tests._cost_meter import EXECUTION, HISTORY, VM_STEP, CostMeter, tagged
from tests._scale_root import worktree_root
from tests._turn_law import hold_turn_law, readers_of, turn_gap

pytestmark = pytest.mark.scale

REPO = pathlib.Path(__file__).resolve().parents[1]
K_SMALL = 1
CHILD_TIMEOUT_SEC = 900  # a transport bound for one child, not a judgment about the boot
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
    return (pytest.mark.xfail(strict=True, reason=table[name]),) if name in table else ()


# ---- measurement -------------------------------------------------------------------------------

def _child(module: str, args: List[str], env: Dict[str, str], log: pathlib.Path) -> None:
    """One fresh interpreter; a timeout or an interrupt stops its whole process tree (tests/group_run.py)."""
    from tests.group_run import run_in_group

    done = run_in_group([sys.executable, "-m", module, *args], timeout=CHILD_TIMEOUT_SEC, cwd=REPO, env=env)
    log.write_text(done.stdout + done.stderr, encoding="utf-8")
    assert done.returncode == 0, f"{module} exited {done.returncode}:\n{(done.stdout + done.stderr)[-6000:]}"


def measure(base: pathlib.Path, k: int) -> Dict[str, Any]:
    """Build the k-root in one interpreter, then boot and query it in another (cold)."""
    home = base / f"k{k:03d}"
    root, repo = home / "data", home / "repo"
    root.mkdir(parents=True)
    repo.mkdir()
    # The delegated copies live under the worktree root the build and the boot both name (tests/_scale_root.py).
    env = {**os.environ, "OUROBOROS_DATA_DIR": str(root), "OUROBOROS_SETTINGS_PATH": str(root / "settings.json"),
           "OUROBOROS_REPO_DIR": str(repo), "OUROBOROS_HOST_SERVICE_PORT": "0",
           "OUROBOROS_SUBAGENT_WORKTREE_ROOT": str(worktree_root(root))}
    facts_path, spec_path, out_path = home / "facts.json", home / "spec.json", home / "measured.json"
    _child("tests._scale_root", [str(root), str(k), str(facts_path)], env, home / "build.log")
    facts = json.loads(facts_path.read_text(encoding="utf-8"))
    spec_path.write_text(json.dumps({"root": str(root), "facts": facts}), encoding="utf-8")
    _child("tests._cost_driver", [str(spec_path), str(out_path)], env, home / "drive.log")
    measured = json.loads(out_path.read_text(encoding="utf-8"))
    assert not measured["errors"], f"k={k}: the driver failed:\n" + "\n".join(measured["errors"])
    measured["root"] = str(root)
    return measured


@pytest.fixture(scope="module")
def small(tmp_path_factory) -> Dict[str, Any]:
    return measure(tmp_path_factory.mktemp("cost"), K_SMALL)


@pytest.fixture(scope="module", params=[pytest.param(30, id="k30"),
                                        pytest.param(100, id="k100", marks=pytest.mark.integration)])
def measured(request, small, tmp_path_factory) -> Dict[int, Dict[str, Any]]:
    large = request.param
    return {K_SMALL: small, large: measure(tmp_path_factory.mktemp("cost"), large), "large": large}


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


def assert_flat(measured: Dict[Any, Any], tag: str) -> None:
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
    assert not breaches, (f"{tag} grows with unrelated history (k={K_SMALL} -> k={large}): {breaches}; "
                          f"history touched at k={large}: {touched}; growing readers: {growing}")


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


@pytest.mark.parametrize("tag", ["R_active", pytest.param("R_late", marks=pytest.mark.xfail(strict=True, reason=(
    "The tool-call evidence a late task's card replays (tool_call_log.replay_evidence -> Memory.read_task_recent) "
    "reads up to the three newest tools.jsonl archives, newest first, before it reaches the task's own rows; the "
    "tools log stays a rotated file in this change (k=1 -> k=30: one more tools segment, about 39 KB)."))),
    "R_activew"])
def test_room_history_is_flat(measured, tag):
    for k in (K_SMALL, measured["large"]):  # a page, never the import window's refusal
        assert measured[k]["import"]["state"] == "done" and measured[k]["responses"][tag]["status"] == 200, k
    assert_flat(measured, tag)


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


@pytest.mark.parametrize("tag", [pytest.param(tag, marks=turn_gap("L2", tag))
                                 for tag in (*MAIN_TURNS, "T:room")])
def test_b10_an_integrating_turn_makes_the_same_round_trips_at_every_size(measured, tag):
    """L2. An integrating turn makes the same round trips and decodes the same dialogue rows at k=1 and k=30:
    Main's (the process's first, the next one, one after activity elsewhere) and a project room's direct turn,
    which integrates as a root task does (BIBLE P1: the whole horizon stays resident there, so its view holds
    every room's story and a line per live room). Its payload is free: an integrating view shows its story."""
    large = measured["large"]
    little, big = at_request(measured, tag, K_SMALL), at_request(measured, tag, large)
    keys = (*ROUND, "rows_decoded")
    hold_turn_law("L2", tag, _growth(little, big, keys), dict.fromkeys(keys, COUNT_JITTER), readers_of(little, big),
                  scale=(large - K_SMALL) / 29)


@pytest.mark.parametrize("law, tag, before, step", [
    pytest.param("L3", "T:main:broad", "T:mainw", "breadth", marks=turn_gap("L3", "T:main:broad")),
    pytest.param("L4", "T:main:rows", "T:main:broad", "rows", marks=turn_gap("L4", "T:main:rows")),
    pytest.param("L3", "T:main:archive", "T:main:rows", "archive", marks=turn_gap("L3", "T:main:archive"))])
def test_b10_main_s_turn_reads_pages_not_rooms_roots_or_rows(measured, law, tag, before, step):
    """L3/L4 (k=1). After 40 live rooms with story and 10 running roots (L3), 650 open rows over those rooms (L4),
    or 40 archived knowledge notes and 40 old rescue snapshots (L3, the turn after the one that met them), only
    the dialogue's own page reads may grow: read transactions by B9's allowance per page of rows the step added
    (one page more when they cross a page edge), statements by as many transactions of the turn before's
    statements per transaction; every other round trip stays within ``COUNT_JITTER``. Payload is free."""
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


# ---- B6: the rest law of the named maintenance jobs ---------------------------------------------

def second_run(run: Callable[[pathlib.Path], Any], root: pathlib.Path) -> Dict[str, int]:
    """The counters of a job's second run over the inputs its first run left."""
    run(root)
    with CostMeter(root) as meter:
        tagged("rest", lambda: run(root))
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
    """One named job's run in this process: ``maintenance.run_job(root, name)``."""
    from tests._store_shared import as_install

    maintenance = _maintenance()

    def run(root: pathlib.Path) -> Any:
        with as_install(root):  # the maintenance child runs as the install it maintains
            return maintenance.run_job(root, name, should_stop=lambda: False)
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
    (``ACTIVITIES[phase]``, the driver's own shapes, as the install it maintains)."""
    from tests._store_shared import as_install

    run(root)
    run(root)
    with as_install(root):
        ACTIVITIES[phase](root, f"job-{what}", k)
    with CostMeter(root) as meter:
        tagged(phase, lambda: run(root))
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
    maintenance = _maintenance()
    if maintenance is None or not callable(getattr(maintenance, "run", None)):
        pytest.skip("the maintenance child's pass does not exist on this tree yet")
    _rests_at_every_size(measured, lambda root: maintenance.run(root, should_stop=lambda: False), "maintenance.run")


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
    maintenance = _maintenance()
    if maintenance is None or not callable(getattr(maintenance, "run", None)):
        pytest.skip("the maintenance child's pass does not exist on this tree yet")
    _same_cost_after_activity(measured, lambda root: maintenance.run(root, should_stop=lambda: False), name, phase)


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
}
COMPANIONS_NOT_BUILT = {
    "pause_notice_replay": "a pause notice obligation needs a parked episode's fence and its notice row",
    "post_task_synthesis_recovery": "a pending root synthesis dispatches a model turn (refused here)",
    "held_continuation_recheck": "a writer-held Continue needs a pending continuation under queue custody",
    "cold-custody": "sleep custody needs a live actor in the queue or the direct registry",
    "owner_pause_settlement": "a requested owner pause stops members through the daemon (absent here)",
    "schedule-admission": "a due schedule admits a task into the queue the held supervisor never serves",
}


def _duty_pass(measured: Dict[Any, Any], name: str, k: int, phase: str) -> Dict[str, Any]:
    entry = measured[k].get("duties", {}).get(name)
    assert entry is not None, f"{name} was not driven at k={k}"
    snapshot = entry.get(phase)
    assert snapshot is not None, f"{name} has no {phase} pass at k={k} (companion: {entry.get('companion')!r})"
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


def test_b7_every_duty_has_a_companion_or_a_named_reason():
    assert set(DUTY_NAMES) == set(COMPANION_COUNTERS) | set(COMPANIONS_NOT_BUILT)
    assert not set(COMPANION_COUNTERS) & set(COMPANIONS_NOT_BUILT)


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


@pytest.mark.parametrize("look_ends", ["inside_the_pass", "during_the_drain"])
def test_the_driver_measures_the_child_a_pass_spawned_whenever_its_look_ends(tmp_path, monkeypatch, look_ends):
    """A pass whose look spawned the child owns the child's jobs: the driver runs them under the pass's
    tag whether the look's thread ended inside the pass (its spawn recorded before the drain began) or
    ends while the drain joins it. A spawn baseline read after the pass takes the early spawn for an
    old one and measures no child jobs, with no error to show for it."""
    import threading
    import time
    from types import SimpleNamespace

    from ouroboros import maintenance
    from tests._cost_driver import _measured_pass
    from tests._cost_meter import tag_of

    runs: List[str] = []
    monkeypatch.setattr(maintenance, "run", lambda _root, **_kwargs: runs.append(tag_of()))
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
        _measured_pass(meter, "D:duty:first", look, dispatcher, tmp_path, threading.Event(), result)
    assert result["errors"] == [] and runs == ["D:duty:first"], (look_ends, runs, result["child_spawns"])


@pytest.mark.parametrize("extra", ["none", "sql", "body", "legacy_scan", "missing_headers"])
@pytest.mark.parametrize("mode", ["PERSIST", "WAL"])
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

    def job(root: pathlib.Path) -> None:
        if rests and done.exists():
            return
        with open(root / "archive" / "chat_0001.jsonl", encoding="utf-8") as handle:
            [json.loads(line) for line in handle]
        done.write_text("done", encoding="utf-8")

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
