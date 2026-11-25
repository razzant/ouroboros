"""The paths that answer the owner read addressed facts, never unrelated history.

ARCHITECTURE §10 invariant 10 and BIBLE P5: a current answer is read from the facts it
names. On 22.10 three readers on the threads that answer did history-sized work: the
``send_message`` handler's receipt capture listed every evidence-pack sidecar (two
supervisor-loop stalls of 125.7 s and 169.0 s), the owner-message intake read every
task's typed row to offer the newest root results (a 19.5 s hold), and the consciousness
tick read its allowance window on the loop (holds up to 23 s).

Each boundary is driven through its real entry point on two roots that hold the same
current facts and k=1 or k=30 units of unrelated history, counted by the cost-contract
meter (``tests/_cost_meter.py``) and compared by the contract's own rule
(``tests/test_cost_contract.py::contract_breaches``: equal work, a logarithmic SQLite
allowance). A warm-up on each root first takes the process-wide first-use costs (imports,
memos), so the counted call is the steady first pass of the boundary itself. Every guard
has a companion that shows the boundary still does its job on both sizes.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import threading
from collections import deque
from types import SimpleNamespace
from typing import Any, Callable, Dict, Tuple

import pytest

from tests._cost_meter import CostMeter, tagged
from tests.test_cost_contract import K_SMALL, contract_breaches

pytestmark = pytest.mark.scale

K_LARGE = 30
TAG = "answer_path"


def _flat(tmp_path, monkeypatch, build: Callable[[Any, int], Any], boundary: Callable[[Any, Any], Callable[[], Any]],
          *, unlisted: Tuple[str, ...] = (), sql: bool = True) -> Dict[int, Dict[str, Any]]:
    """Build each root, let ``boundary(root, facts)`` warm up and return the counted call; compare.

    Cached store connections are closed after the build, so every connection the boundary uses
    is opened under the meter and its SQL is counted (a connection opened before the meter is
    installed carries no trace callback); ``sql=False`` names a call answered from files alone.
    The meter counts a directory listing once, whatever it returns, so a location in ``unlisted``
    must not be listed at all: the names the call examines there are its counted stats and opens,
    compared like every other count."""
    from ouroboros import store

    counted, answers = {}, {}
    for k in (K_SMALL, K_LARGE):
        root = tmp_path / f"k{k:03d}" / "data"
        root.mkdir(parents=True)
        monkeypatch.setenv("OUROBOROS_DATA_DIR", str(root))
        facts = build(root, k)
        store.close_connections()
        with CostMeter(root) as meter:
            call = tagged("warm-up", lambda: boundary(root, facts))
            answers[k] = tagged(TAG, call)
            assert not meter.join_spawned(TAG), "a thread the boundary started is still running"
        counted[k] = meter.snapshot(TAG)
        assert counted[k]["counts"].get("statements") or not sql, "the counted call ran no SQL the meter could see"
        listed = {key: amount for key, amount in counted[k]["counts"].items()
                  if key.split(":")[0] in ("listdir", "scandir") and key.partition(":")[2] in unlisted}
        assert not listed, f"k={k}: the counted call listed {listed}: " + "; ".join(
            source for source in counted[k]["callers"] if source.split(" <- ")[0] in listed)
    breaches = contract_breaches(counted[K_SMALL]["counts"], counted[K_LARGE]["counts"], K_LARGE)
    growing = sorted(((source, amount - counted[K_SMALL]["callers"].get(source, 0))
                      for source, amount in counted[K_LARGE]["callers"].items()
                      if amount > counted[K_SMALL]["callers"].get(source, 0)), key=lambda item: -item[1])[:8]
    assert not breaches, f"grows with unrelated history (k={K_SMALL} -> k={K_LARGE}): {breaches}; growing: {growing}"
    return answers


# ---- send_message: the terminal answer's receipt capture over k-scaled evidence packs ------------

PACKS_PER_K = 3


def _packed_history(root, k):
    """``PACKS_PER_K`` settled tasks per unit, packed by the real packer (sidecars, packs, index)."""
    from ouroboros import evidence_packer
    from tests.test_evidence_packs import _task

    for unit in range(k):
        for index in range(PACKS_PER_K):
            _task(root, f"hist-{unit}-{index}", blobs=2)
    assert evidence_packer.run_pass(root)["faults"] == 0
    return {}


def _sender(root, monkeypatch):
    """The supervisor's send seam with a local bridge; frames are what the owner's chat received."""
    from ouroboros import dialogue_store, utils
    from supervisor import events_chat_delivery, message_bus
    from supervisor.event_observation import ObservedEventQueue
    from tests.test_dialogue_retry import bridge

    # The dialogue in the test's journal: ``test_dialogue_retry.seed`` pins rollback journals for its
    # contention cases, and a pin taken after the packed history's WAL store exists makes the store's
    # next open refuse that store.
    dialogue_store.append(root, "chat", {"ts": "seed", "chat_id": 1, "text": "seed"})
    dialogue_store._plan_first(root)
    frames = bridge(root, monkeypatch)
    queue = ObservedEventQueue()
    monkeypatch.setattr(events_chat_delivery, "_DELIVERED_MESSAGE_IDS", deque(maxlen=256))
    monkeypatch.setattr(events_chat_delivery, "_ACTIVE_DELIVERIES", set())
    monkeypatch.setattr(events_chat_delivery, "_DEFERRED_DELIVERIES", set())
    monkeypatch.setattr("supervisor.workers.get_event_q", lambda **_kw: queue)
    ctx = SimpleNamespace(DRIVE_ROOT=root, RUNNING={}, EVENT_Q=queue, append_jsonl=utils.append_jsonl,
                          send_with_budget=message_bus.send_with_budget)
    return ctx, frames


def _owed_answer(root, task_id):
    from ouroboros.task_results import write_task_result
    from supervisor import terminal_delivery

    write_task_result(root, task_id, "running", chat_id=1, root_task_id=task_id)
    text = f"The answer of {task_id}."
    event = {"type": "send_message", "chat_id": 1, "task_id": task_id, "text": text,
             "delivery_id": terminal_delivery.delivery_id_for(task_id, text)}
    assert terminal_delivery.register_pending_delivery(root, event)
    return event


@pytest.mark.parametrize("journal", ["PERSIST", "WAL"])
@pytest.mark.parametrize("index", ["validated", "stale", "contended", "damaged"])
def test_send_message_receipt_capture_is_flat_across_evidence_packs(tmp_path, monkeypatch, index, journal):
    """The delivering task names its own handle, so the capture reads that task's own sidecars,
    found by their names: the packs directory is never listed (one listing returns every pack's
    names), and the names the capture probes are counted and compared at both sizes.

    ``validated``: a reader of this process opened the index the packer left (the quiet case even
    before the fix: the index answered); ``stale``: the packer wrote the index since (it once cost
    every process a whole-index ``quick_check``); ``contended``: the packer holds the index lock
    past the send's acquisition slice (the 22.10 stalls: every sidecar opened and parsed);
    ``damaged``: the cache awaits a rebuild from every sidecar. The answer still reaches the chat
    and keeps its retained receipt on both sizes.

    ``journal`` is chosen before the first store exists and read back from each store's header:
    the store's own choice (WAL only on a fixed SQLite and a proven local filesystem) would give
    each host just one of the two."""
    from ouroboros import dialogue_store, evidence_packer, evidence_packs, store
    from supervisor import events_chat_delivery

    monkeypatch.setattr(store, "_journal_mode", lambda *_: journal)

    def boundary(root, _facts):
        ctx, frames = _sender(root, monkeypatch)
        journals = {path.name: store.wal_file(path)
                    for path in (store.store_path(root), dialogue_store.path(root),
                                 evidence_packs.packs_dir(root) / evidence_packs.INDEX_NAME)}
        assert journals == dict.fromkeys(journals, journal == "WAL")
        assert events_chat_delivery._handle_send_message(_owed_answer(root, "warm-up"), ctx)
        event = _owed_answer(root, "turn")
        holder = None
        if index == "validated":
            with evidence_packs.open_index(root):
                pass  # a reader of this process opened the bytes the packer left
        elif index == "stale":
            evidence_packer.record_swept(root, "elsewhere", "[]")  # the packer's write of another task's sweep
        elif index == "damaged":
            (evidence_packs.packs_dir(root) / evidence_packs.INDEX_NAME).write_bytes(b"not a database")
        elif index == "contended":
            held, release = threading.Event(), threading.Event()

            def hold():
                with store.name_lock(root, evidence_packs.INDEX_LOCK_NAME, timeout_sec=5):
                    held.set()
                    release.wait(30)

            holder = threading.Thread(target=hold, name="packer-holds-index")
            holder.start()
            assert held.wait(5)

        def send():
            try:
                sent = events_chat_delivery._handle_send_message(event, ctx)
            finally:
                if holder is not None:
                    release.set()
                    holder.join(10)
            registry = json.loads((root / "state" / "terminal_deliveries.json").read_text(encoding="utf-8"))
            receipt = registry["receipts"][event["delivery_id"]]
            return (sent, frames[-1]["delivery_id"] == event["delivery_id"],
                    receipt["source_ref"]["path"].startswith("source_handles/context_checkpoints/"))
        return send

    answers = _flat(tmp_path, monkeypatch, _packed_history, boundary, unlisted=("packs",))
    assert answers[K_SMALL] == answers[K_LARGE] == (True, True, True)


# ---- a packed blob: read by its task and by its sha alone, over k-scaled evidence packs ----------

def _packed_blobs(root, k):
    """The read's own facts at every size: ``wanted`` and ``broken`` packed beside the history of
    ``_packed_history``, then ``broken``'s sidecar damaged (the index the packer left still holds it)."""
    from ouroboros import evidence_packs
    from tests.test_evidence_packs import _task

    refs = {task: _task(root, task, blobs=1)[0][0] for task in ("wanted", "broken")}
    _packed_history(root, k)
    (evidence_packs.packs_dir(root) / "broken.members.json").write_text("{", encoding="utf-8")
    return refs


@pytest.mark.parametrize("journal", ["PERSIST", "WAL"])
@pytest.mark.parametrize("index", ["valid", "busy", "garbage", "absent"])
def test_a_packed_blob_read_is_flat_across_evidence_packs(tmp_path, monkeypatch, index, journal):
    """A packed blob named with its task is read from that task's own sidecars, found by their names;
    named by its sha alone it asks the shared index. Neither lists the packs directory, and no
    counter grows with the packs beside it.

    ``valid``: the index the packer left answers; ``busy``: the packer holds the index lock past the
    read's acquisition slice; ``garbage``: the file is not a database; ``absent``: no index (both
    await the maintenance pass's rebuild, never the reader's). The task's own read answers in every
    state; the sha-only read answers from a valid index and is otherwise ``EvidenceUnavailable``
    (the address stays good), never an absence; a sha no pack holds is ``FileNotFoundError`` once
    the index can say so; a task whose own sidecar is damaged keeps the placement a valid index
    holds, and is as unknown as the index otherwise (never an absence). The reader writes no index."""
    from ouroboros import evidence_packs, store

    monkeypatch.setattr(store, "_journal_mode", lambda *_: journal)
    nothing = {"kind": "json", "encoding": "gzip", "sha256": "0" * 64, "size": 1}

    def outcome(root, ref, **task):
        try:
            return "bytes", hashlib.sha256(evidence_packs.blob_bytes(root, ref, **task)).hexdigest()
        except (OSError, ValueError) as exc:
            return type(exc).__name__, getattr(exc, "reason", None)

    def boundary(root, refs):
        def reads():
            return (outcome(root, refs["wanted"], task_id="wanted"), outcome(root, refs["wanted"]),
                    outcome(root, nothing, task_id="wanted"), outcome(root, refs["broken"], task_id="broken"))

        reads()  # the process-wide first uses (imports, the lock-tier probe) over the valid index
        path, holder = evidence_packs.packs_dir(root) / evidence_packs.INDEX_NAME, None
        if index == "garbage":
            path.write_bytes(b"not a database")
        elif index == "absent":
            path.unlink()
        elif index == "busy":
            held, release = threading.Event(), threading.Event()

            def hold():
                with store.name_lock(root, evidence_packs.INDEX_LOCK_NAME, timeout_sec=5):
                    held.set()
                    release.wait(30)

            holder = threading.Thread(target=hold, name="packer-holds-index")
            holder.start()
            assert held.wait(5)

        def read():
            try:
                with store.acquisition_budget(0.1):
                    return reads()
            finally:
                if holder is not None:
                    release.set()
                    holder.join(10)
        return read

    answers = _flat(tmp_path, monkeypatch, _packed_blobs, boundary, unlisted=("packs",), sql=index == "valid")
    assert answers[K_SMALL] == answers[K_LARGE]
    own, by_sha, missing, broken = answers[K_SMALL]
    unavailable = ("EvidenceUnavailable", "busy" if index == "busy" else "damaged")
    assert own[0] == "bytes"
    assert (by_sha, missing, broken[0] if index == "valid" else broken) == (
        (own, ("FileNotFoundError", None), "bytes") if index == "valid" else (unavailable, unavailable, unavailable))
    for k in (K_SMALL, K_LARGE):  # the reader rebuilt nothing
        path = evidence_packs.packs_dir(tmp_path / f"k{k:03d}" / "data") / evidence_packs.INDEX_NAME
        assert (path.read_bytes() if index == "garbage" else path.exists()) == {
            "garbage": b"not a database", "absent": False}.get(index, True)



# ---- owner-message intake: the routing manifests over k-scaled task history ----------------------

HISTORY_ROOTS_PER_K, HISTORY_CHILDREN_PER_K, ELSEWHERE_ROOTS_PER_K, WAVE_CHILDREN_PER_K = 10, 4, 6, 10


def _stored(txn, template, task_id, ts, **fields):
    txn.store_task({**template, "task_id": task_id, "ts": ts, "updated_at": ts, **fields}, pending_drive=False)


def _wave(txn, template, unit, project=""):
    """``WAVE_CHILDREN_PER_K`` finished subagent children newer than every current row and as many
    dated between the current roots (``2026-09-01T00:MM:00``/``:30``): the newest-first walk meets
    them all before its last root."""
    for index in range(WAVE_CHILDREN_PER_K):
        between = unit * WAVE_CHILDREN_PER_K + index
        for name, ts in ((f"hist-{unit}-wave-{index}", f"2026-09-{3 + unit // 24:02d}T{unit % 24:02d}:{index:02d}:00+00:00"),
                         (f"hist-{unit}-between-{index}", f"2026-09-01T00:{between % 17:02d}:{10 + between // 17:02d}+00:00")):
            _stored(txn, template, name, ts, project_id=project, parent_task_id=f"hist-{unit}-0",
                    root_task_id=f"hist-{unit}-0", delegation_role="subagent")


def _task_history(root, k):
    """The same current results at every size: ``cap + 1`` newest roots and two newer children in
    Main and in the project room (one root past the cap in each). Unrelated history grows with k:
    older Main roots and children, another project's roots, and a finished swarm wave of an older
    Main root (``_wave``), each with a 4 KB result body."""
    from ouroboros import store
    from ouroboros.projects_registry import create_project
    from ouroboros.runtime_limits import get_routing_manifest_result_rows
    from ouroboros.task_results import write_task_result
    from ouroboros.task_store import read_document

    room = create_project(root, "room", name="Room")
    create_project(root, "elsewhere", name="Elsewhere")
    write_task_result(root, "template", "completed", chat_id=1, result="r" * 4000, ts="2023-01-01T00:00:00+00:00")
    template = read_document(root, "template")
    current = get_routing_manifest_result_rows() + 1
    with store.open_store(root, write=True) as txn:
        for scope, project in (("main", ""), ("room", "room")):
            for index in range(current):
                _stored(txn, template, f"cur-{scope}-{index:02d}",
                        f"2026-09-01T00:{index:02d}:{30 if project else 0:02d}+00:00",
                        project_id=project, objective=f"current {scope} work {index}")
            for index in range(2):
                _stored(txn, template, f"cur-{scope}-child-{index}", f"2026-09-02T00:0{index}:00+00:00",
                        project_id=project, parent_task_id=f"cur-{scope}-00", root_task_id=f"cur-{scope}-00",
                        delegation_role="subagent")
    for unit in range(k):
        with store.open_store(root, write=True) as txn:
            day = f"2024-{1 + unit // 28:02d}-{1 + unit % 28:02d}"
            for index in range(HISTORY_ROOTS_PER_K):
                _stored(txn, template, f"hist-{unit}-{index}", f"{day}T01:{index:02d}:00+00:00")
            for index in range(HISTORY_CHILDREN_PER_K):
                _stored(txn, template, f"hist-{unit}-child-{index}", f"{day}T02:{index:02d}:00+00:00",
                        parent_task_id=f"hist-{unit}-0", root_task_id=f"hist-{unit}-0", delegation_role="subagent")
            for index in range(ELSEWHERE_ROOTS_PER_K):
                _stored(txn, template, f"hist-{unit}-elsewhere-{index}", f"{day}T03:{index:02d}:00+00:00",
                        project_id="elsewhere")
            _wave(txn, template, unit)
    return {"room_chat_id": int(room["chat_id"])}


@pytest.mark.parametrize("room", ["main", "project"])
def test_owner_message_intake_reads_the_newest_roots_not_every_task(tmp_path, monkeypatch, room):
    """The supervisor loop routes an owner message: the Main manifest walks the store's root-only
    index until its window is full (no child of the swarm wave is visited), a project room walks
    its own; bodies are decoded for the offered rows alone. Both sizes offer the same newest
    roots, children never among them."""
    from ouroboros import server_owner_routing

    def boundary(root, facts):
        directed, started = [], threading.Event()

        def direct(*_args, task_metadata=None, **_kwargs):
            directed.append(task_metadata)
            started.set()

        ctx = SimpleNamespace(DRIVE_ROOT=root, RUNNING={}, PENDING=[], handle_chat_direct=direct)
        chat_id = 1 if room == "main" else facts["room_chat_id"]

        def message(cmid):
            server_owner_routing._route_owner_message(SimpleNamespace(), ctx, {
                "chat_id": chat_id, "text": "continue the work", "client_message_id": cmid, "task_metadata": None,
                "log_text": "continue the work", "origin_message_ref": {"chat_id": chat_id, "client_message_id": cmid},
                "source": "web"})

        message("warm-up")
        assert started.wait(10), "the warm-up turn never started"

        def route():
            message("counted")
            return len(directed)
        return lambda: (route(), directed)

    answers = _flat(tmp_path, monkeypatch, _task_history, boundary)
    manifests = {}
    for k, (_count, directed) in answers.items():
        metadata = directed[-1]
        manifest = metadata["main_routing_manifest" if room == "main" else "project_routing_manifest"]
        manifests[k] = [row["task_id"] for row in manifest["final_results"]]
    # Main offers every lane's roots (the room's current roots are half a minute newer); a room its own.
    newest = sorted(((f"2026-09-01T00:{index:02d}:{30 if scope == 'room' else 0:02d}", f"cur-{scope}-{index:02d}")
                     for scope in (("main", "room") if room == "main" else ("room",)) for index in range(17)),
                    reverse=True)
    assert manifests[K_SMALL] == manifests[K_LARGE] == [task_id for _ts, task_id in newest[:16]]


@pytest.mark.parametrize("scope", ["main", "project"])
def test_the_newest_roots_are_walked_without_visiting_a_child(tmp_path, monkeypatch, scope):
    """The window walks a root-only index, so ``limit`` bounds the entries read: the scope's own swarm
    waves (children newer than every root, children dated between its roots) are never visited, in
    Main and in a project room. Both sizes return the same newest roots."""
    from ouroboros import store
    from ouroboros.task_results import write_task_result
    from ouroboros.task_store import newest_root_ids, read_document

    project = "room" if scope == "project" else ""

    def build(root, k):
        write_task_result(root, "template", "completed", chat_id=1, result="r" * 4000, ts="2023-01-01T00:00:00+00:00")
        template = read_document(root, "template")
        with store.open_store(root, write=True) as txn:
            for index in range(17):
                _stored(txn, template, f"cur-{index:02d}", f"2026-09-01T00:{index:02d}:00+00:00", project_id=project)
        for unit in range(k):
            with store.open_store(root, write=True) as txn:
                _stored(txn, template, f"hist-{unit}-0", f"2024-01-{1 + unit % 28:02d}T00:00:00+00:00", project_id=project)
                _wave(txn, template, unit, project)
        return {}

    def boundary(root, _facts):
        newest_root_ids(root, 17, project_id=project)
        return lambda: newest_root_ids(root, 17, project_id=project)

    answers = _flat(tmp_path, monkeypatch, build, boundary)
    assert answers[K_SMALL] == answers[K_LARGE] == [f"cur-{index:02d}" for index in reversed(range(17))]


def test_a_store_an_older_body_published_gains_the_root_windows_at_boot(tmp_path, monkeypatch):
    """Schema 8's root-only indexes are DDL of the boot's schema step, committed with the version and
    built once; before that step the window cannot be read (``StoreError``: the manifest says the
    store is unreadable), never answered by walking every row. Afterwards both windows read through
    their indexes and a room's counts stay exact."""
    from ouroboros import store
    from ouroboros.store_import import ensure_current_store
    from ouroboros.task_results import write_task_result
    from ouroboros.task_store import newest_root_ids, project_result_counts

    monkeypatch.setenv("OUROBOROS_DATA_DIR", str(tmp_path))
    with monkeypatch.context() as patch:
        patch.setattr(store, "_SCHEMA", store._SCHEMA.replace("".join(store.ROOT_WINDOWS), ""))
        patch.setattr(store, "SCHEMA_VERSION", 7)
        for index in range(3):
            write_task_result(tmp_path, f"root{index}", "completed", project_id="room", ts=f"2026-09-01T00:0{index}:00Z")
        write_task_result(tmp_path, "child", "completed", project_id="room", parent_task_id="root0",
                          root_task_id="root0", delegation_role="subagent", ts="2026-09-02T00:00:00Z")
    store.close_connections(store.store_path(tmp_path))
    names = "SELECT name FROM sqlite_master WHERE name IN ('tasks_root_ts', 'tasks_project_root_ts')"
    with store.open_store(tmp_path, write=False) as txn:
        assert txn.meta("schema_version") == 7 and not txn.conn.execute(names).fetchall()
    with pytest.raises(store.StoreError, match="no such index"):
        newest_root_ids(tmp_path, 2)
    ensure_current_store(tmp_path)  # the boot's schema step
    with store.open_store(tmp_path, write=False) as txn:
        assert txn.meta("schema_version") == store.SCHEMA_VERSION == 12
        assert sorted(row[0] for row in txn.conn.execute(names)) == ["tasks_project_root_ts", "tasks_root_ts"]
    assert newest_root_ids(tmp_path, 2) == newest_root_ids(tmp_path, 2, project_id="room") == ["root2", "root1"]
    assert project_result_counts(tmp_path, "room") == (3, 1)


def _spy_sidecars(monkeypatch):
    from ouroboros import evidence_packs

    decoded, real = [], evidence_packs.read_sidecar

    def read(path, **kwargs):
        decoded.append(path.name)
        return real(path, **kwargs)

    monkeypatch.setattr(evidence_packs, "read_sidecar", read)
    return decoded


def test_a_task_handle_is_answered_by_its_own_sidecars_only(tmp_path, monkeypatch):
    """Another task's pack never joins the answer: an unreadable unrelated sidecar neither blocks a
    lookup nor turns an absence into an error, a dotted task id is told apart by the sidecar's own
    ``task_id``, and this task's own unreadable sidecar stays an explicit error."""
    from ouroboros import evidence_packer, evidence_packs as packs
    from tests.test_evidence_packs import _task

    monkeypatch.setenv("OUROBOROS_DATA_DIR", str(tmp_path))
    handles = {task: _task(tmp_path, task, blobs=1, handle=f"# focus {task}\n".encode())[1]
               for task in ("a", "a.2", "b")}
    assert evidence_packer.run_pass(tmp_path)["faults"] == 0
    (packs.packs_dir(tmp_path) / "b.members.json").write_text("{", encoding="utf-8")
    decoded = _spy_sidecars(monkeypatch)
    assert packs.handle_bytes(tmp_path, "a", handles["a"]["path"]) == b"# focus a\n"
    assert packs.handle_info(tmp_path, "a", handles["a"]["path"].replace("focus_source", "absent")) is None
    assert packs.packed_handle_names(tmp_path, "a") == [handles["a"]["path"].rsplit("/", 1)[-1]]
    assert packs.handle_bytes(tmp_path, "a.2", handles["a.2"]["path"]) == b"# focus a.2\n"
    assert set(decoded) <= {"a.members.json", "a.2.members.json"}, decoded  # never b's, never another task's
    (packs.packs_dir(tmp_path) / "a.members.json").write_text("{", encoding="utf-8")
    (packs.packs_dir(tmp_path) / packs.INDEX_NAME).unlink()  # no cache that could still vouch for it
    with pytest.raises(packs.SidecarCorrupt, match="evidence_corrupt"):
        packs.handle_bytes(tmp_path, "a", handles["a"]["path"])


def test_a_torn_generation_does_not_hide_the_next_one(tmp_path, monkeypatch):
    """A task's sidecars are probed by name, generation after generation: a pack published without
    its sidecar (a crash between the two) still occupies its generation, the packer moves past it,
    and so does the probe; the first generation holding neither name ends the task."""
    from ouroboros import evidence_packer, evidence_packs as packs
    from ouroboros.artifacts import store_actor_source_bytes, task_artifact_dir_path
    from tests.test_evidence_packs import _task

    monkeypatch.setenv("OUROBOROS_DATA_DIR", str(tmp_path))
    _refs, first = _task(tmp_path, "torn", blobs=1)
    assert evidence_packer.run_pass(tmp_path)["faults"] == 0
    (packs.packs_dir(tmp_path) / packs.pack_name("torn", 2)).write_bytes(b"a pack whose sidecar never landed")
    later = store_actor_source_bytes(tmp_path, "torn", category="context_checkpoints", source_id="focus_source_later",
                                     data=b"# later focus\n", extension="md")
    assert evidence_packer.run_pass(tmp_path)["packed"] == 1
    assert not (task_artifact_dir_path(tmp_path, "torn") / later["path"]).exists()  # only its pack holds it now
    assert list(packs._task_sidecar_names(tmp_path, "torn")) == ["torn.members.json", "torn.3.members.json"]
    assert packs.handle_bytes(tmp_path, "torn", later["path"]) == b"# later focus\n"
    assert packs.packed_handle_names(tmp_path, "torn") == sorted(ref["path"].rsplit("/", 1)[-1] for ref in (first, later))


@pytest.mark.parametrize("index", ["valid", "absent"])
def test_a_valid_index_keeps_a_placement_whose_own_sidecar_was_damaged(tmp_path, monkeypatch, index):
    """The packer's index outlives a damaged sidecar: only then is the existing cache asked (never
    rebuilt). Without a cache the damage stays explicit."""
    from ouroboros import evidence_packer, evidence_packs as packs
    from tests.test_evidence_packs import _task

    monkeypatch.setenv("OUROBOROS_DATA_DIR", str(tmp_path))
    _refs, handle = _task(tmp_path, "kept", blobs=1)
    assert evidence_packer.run_pass(tmp_path)["faults"] == 0
    (packs.packs_dir(tmp_path) / "kept.members.json").write_text("{", encoding="utf-8")
    if index == "absent":
        (packs.packs_dir(tmp_path) / packs.INDEX_NAME).unlink()
        with pytest.raises(packs.SidecarCorrupt):
            packs.handle_bytes(tmp_path, "kept", handle["path"])
        assert not (packs.packs_dir(tmp_path) / packs.INDEX_NAME).exists()  # the read rebuilt nothing
    else:
        assert packs.handle_bytes(tmp_path, "kept", handle["path"]).startswith(b"# focus")
        assert packs.handle_info(tmp_path, "kept", handle["path"]) == {
            "size": handle["size"], "sha256": handle["sha256"], "loose": False}


def test_a_handle_listing_beside_an_unreadable_generation_joins_the_valid_index(tmp_path, monkeypatch):
    """A task's handle listing beside an unreadable generation of its own asks the index the packer left and
    lists both (its readable sidecars may hold a generation the index does not yet): every handle, the damaged
    generation's one read exactly. An index that cannot answer leaves the readable generations' handles; with
    every sidecar readable the index is never asked (``test_a_handle_miss_and_an_ordering_hint_never_wait_...``)."""
    from ouroboros import evidence_packer, evidence_packs as packs, store
    from ouroboros.artifacts import store_actor_source_bytes
    from tests.test_evidence_packs import _task

    monkeypatch.setenv("OUROBOROS_DATA_DIR", str(tmp_path))
    _refs, first = _task(tmp_path, "two", blobs=1)
    assert evidence_packer.run_pass(tmp_path)["faults"] == 0
    later = store_actor_source_bytes(tmp_path, "two", category="context_checkpoints", source_id="focus_source_later",
                                     data=b"# later focus\n", extension="md")
    assert evidence_packer.run_pass(tmp_path)["packed"] == 1  # its second generation
    names = sorted(ref["path"].rsplit("/", 1)[-1] for ref in (first, later))
    assert packs.packed_handle_names(tmp_path, "two") == names
    (packs.packs_dir(tmp_path) / "two.members.json").write_text("{", encoding="utf-8")
    assert packs.packed_handle_names(tmp_path, "two") == names
    assert packs.handle_bytes(tmp_path, "two", first["path"]) == b"# focus\n" + b"y" * 3000
    with store.name_lock(tmp_path, packs.INDEX_LOCK_NAME, timeout_sec=0), store.acquisition_budget(0):
        assert packs.packed_handle_names(tmp_path, "two") == [later["path"].rsplit("/", 1)[-1]]


def test_a_handle_miss_and_an_ordering_hint_never_wait_for_the_index_lock(tmp_path, monkeypatch):
    """With the packer holding the index lock: a handle miss is answered by its task's own readable
    sidecars and never asks the index; ``read_position`` names a blob its owner's sidecars hold the
    same way, and for one they do not hold asks the index without waiting (``EvidenceUnavailable``,
    busy; never ``StoreBusy``); a blob read under a zero acquisition budget waits no more."""
    from ouroboros import evidence_packer, evidence_packs as packs, store
    from tests.test_evidence_packs import _task

    monkeypatch.setenv("OUROBOROS_DATA_DIR", str(tmp_path))
    (ref, *_), handle = _task(tmp_path, "a", blobs=1)
    (foreign, *_), _ = _task(tmp_path, "b", blobs=1)
    assert evidence_packer.run_pass(tmp_path)["faults"] == 0
    held, release, waits, name_lock = threading.Event(), threading.Event(), [], store.name_lock

    def hold():
        with name_lock(tmp_path, packs.INDEX_LOCK_NAME, timeout_sec=5):
            held.set()
            release.wait(30)

    @contextlib.contextmanager
    def observed(root, filename, *, timeout_sec, **kwargs):
        if filename == packs.INDEX_LOCK_NAME:
            waits.append(timeout_sec)
        with name_lock(root, filename, timeout_sec=timeout_sec, **kwargs):
            yield

    holder = threading.Thread(target=hold, name="packer-holds-index")
    holder.start()
    assert held.wait(5)
    monkeypatch.setattr(store, "name_lock", observed)
    try:
        absent = handle["path"].replace("focus_source", "absent")
        assert packs.handle_info(tmp_path, "a", absent) is None
        assert packs.read_position(tmp_path, task_id="a", rel_path=absent) is None
        assert packs.packed_handle_names(tmp_path, "a") == [handle["path"].rsplit("/", 1)[-1]]
        position = packs.read_position(tmp_path, sha256=ref["sha256"], kind="json", task_id="a")
        assert position[0].endswith("a.xz") and waits == []  # its own sidecars answered: the index was never asked
        with pytest.raises(packs.EvidenceUnavailable) as hint:
            packs.read_position(tmp_path, sha256=foreign["sha256"], kind="json", task_id="a")
        with store.acquisition_budget(0), pytest.raises(packs.EvidenceUnavailable) as read:
            packs.blob_bytes(tmp_path, foreign)
        assert (hint.value.reason, read.value.reason, waits) == ("busy", "busy", [0, 0])
    finally:
        release.set()
        holder.join(10)
    assert packs.blob_bytes(tmp_path, foreign) == packs.blob_bytes(tmp_path, foreign, task_id="b")  # free: both answer


def test_a_place_that_cannot_answer_now_leaves_the_read_unknown_before_a_copy_s_damage(tmp_path, monkeypatch):
    """Without bytes, the first place that could not answer now decides the read: a torn loose copy beside a
    sound packed one, read by its sha while the packer holds the index, is ``EvidenceUnavailable`` (a sound copy
    may lie there, and does: the exact bytes once the lock is free); a torn loose copy no pack holds, with the
    index free to say so, stays that copy's ``evidence_corrupt``; a loose copy the file system refuses to open
    (``PermissionError``) is a copy's failure like any other: unknown while the index cannot answer, the packed
    copy's bytes once it can."""
    import gzip
    import pathlib

    from ouroboros import evidence_packs as packs, store
    from tests.test_evidence_packs import _blob

    raw, other = (json.dumps({"result": text}).encode() for text in ("packed", "loose only"))
    ref, alone = _blob(tmp_path, raw), _blob(tmp_path, other)
    packs.write_pack(tmp_path, "owner", 1, [{key: ref[key] for key in ("sha256", "kind", "size")}])
    packs.ensure_index(tmp_path)
    for blob, value in ((ref, raw), (alone, other)):
        with gzip.open(blob["path"], "wb") as handle:  # torn: a sound gzip of a short copy
            handle.write(value[:-3])
    with store.name_lock(tmp_path, packs.INDEX_LOCK_NAME, timeout_sec=0), store.acquisition_budget(0):
        with pytest.raises(packs.EvidenceUnavailable) as unknown:
            packs.blob_bytes(tmp_path, ref)
    assert unknown.value.reason == "busy" and packs.blob_bytes(tmp_path, ref) == raw
    with pytest.raises(packs.EvidenceCorrupt, match="failed size or sha256 verification"):
        packs.blob_bytes(tmp_path, alone)
    real = packs.read_loose_gzip

    def refused(path, *args, **kwargs):  # the file system will not open this loose copy (EACCES; on Windows, a held file)
        if path.name == pathlib.Path(ref["path"]).name:
            raise PermissionError(13, "Permission denied", str(path))
        return real(path, *args, **kwargs)

    monkeypatch.setattr(packs, "read_loose_gzip", refused)
    with store.name_lock(tmp_path, packs.INDEX_LOCK_NAME, timeout_sec=0), store.acquisition_budget(0):
        with pytest.raises(packs.EvidenceUnavailable):  # where a sound copy may lie is unknown now: never the refusal
            packs.blob_bytes(tmp_path, ref)
    assert packs.blob_bytes(tmp_path, ref) == raw  # the index free: the packed copy answers


def test_a_digest_many_packs_hold_is_read_from_one_copy_and_a_bad_copy_yields_the_next(tmp_path, monkeypatch):
    """Equal bytes are packed once per task that references them, so a common digest lies in many packs: a read
    by its sha takes the index's first copy, one row however many packs hold it, and asks for the next copy only
    when that one fails (its pack gone or damaged); every copy bad is that copy's error, as before."""
    import pathlib
    import sqlite3

    from ouroboros import evidence_packs as packs, store
    from tests._packed_read_probe import release_pack_readers
    from tests.test_evidence_packs import _blob

    raw, copies, roots, refs, connect = json.dumps({"result": "common"}).encode(), [], {}, {}, store.connect

    def counted(path, *args, **kwargs):
        conn = connect(path, *args, **kwargs)
        if pathlib.Path(path).name == packs.INDEX_NAME:
            def row(cursor, values):
                copies[-1] += cursor.description[0][0] == "pack"  # a placement the index handed to Python
                return sqlite3.Row(cursor, values)
            conn.row_factory = row
        return conn

    monkeypatch.setattr(store, "connect", counted)
    for k in (K_SMALL, K_LARGE):
        root = roots[k] = tmp_path / f"k{k:03d}"
        ref = refs[k] = _blob(root, raw)
        for n in range(k):
            packs.write_pack(root, f"task-{n:02d}", 1, [{key: ref[key] for key in ("sha256", "kind", "size")}])
        packs.ensure_index(root)
        pathlib.Path(ref["path"]).unlink()  # only the packs hold it now
        copies.append(0)
        assert packs.blob_bytes(root, ref) == raw
    assert copies == [1, 1], copies
    for root in roots.values():  # the reads above hold their packs open: Windows refuses to unlink such a file
        release_pack_readers(root)
    (packs.packs_dir(roots[K_LARGE]) / "task-00.xz").unlink()  # the first copy's pack gone
    (packs.packs_dir(roots[K_LARGE]) / "task-01.xz").write_bytes(b"damaged")  # the second copy's damaged
    assert packs.blob_bytes(roots[K_LARGE], refs[K_LARGE]) == raw  # the third answers
    (packs.packs_dir(roots[K_SMALL]) / "task-00.xz").write_bytes(b"damaged")  # the only copy damaged
    with pytest.raises(packs.EvidenceCorrupt, match="unreadable"):
        packs.blob_bytes(roots[K_SMALL], refs[K_SMALL])
    real = packs._indexed_placements

    def held_past_the_first(*args, first=True):  # the packer takes the lock between two questions
        if not first:
            raise packs.EvidenceUnavailable("evidence_unavailable: held", reason="busy")
        return real(*args, first=first)

    monkeypatch.setattr(packs, "_indexed_placements", held_past_the_first)
    with pytest.raises(packs.EvidenceUnavailable):  # where a sound copy may lie is unknown: never the first's damage
        packs.blob_bytes(roots[K_LARGE], refs[K_LARGE])


def test_a_verdict_on_the_blob_ends_its_copies_and_a_copy_s_wrong_record_yields_the_next(tmp_path, monkeypatch):
    """Verified bytes of a size the ref does not have are the blob's own verdict, alike in every copy: one question
    to the index however many packs hold it; a ref whose own size is over the caller's limit is refused before any
    question. A copy's record is that copy's claim: one whose recorded size is wrong and over the limit yields to
    the next copy, which answers, for a ref with its size and one without (a copy's own failure, as in
    ``test_a_digest_many_packs_hold_is_read_from_one_copy_and_a_bad_copy_yields_the_next``)."""
    import pathlib

    from ouroboros import evidence_packs as packs
    from tests.test_evidence_packs import _blob

    raw, roots, refs, questions, real = json.dumps({"result": "common"}).encode(), {}, {}, [], packs._indexed_placements
    for k in (K_SMALL, K_LARGE):
        roots[k] = tmp_path / f"k{k:03d}"
        refs[k] = _blob(roots[k], raw)
        for n in range(k):
            packs.write_pack(roots[k], f"task-{n:02d}", 1, [{key: refs[k][key] for key in ("sha256", "kind", "size")}])
        packs.ensure_index(roots[k])
        pathlib.Path(refs[k]["path"]).unlink()  # only the packs hold it now

    def asked(*args, **kwargs):
        questions[-1] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(packs, "_indexed_placements", asked)
    outcomes = []
    for k in (K_SMALL, K_LARGE):
        for ref, limit in (({**refs[k], "size": len(raw) + 1}, None), (refs[k], len(raw) - 1)):
            questions.append(0)
            with pytest.raises(ValueError) as verdict:
                packs.blob_bytes(roots[k], ref, max_bytes=limit)
            outcomes.append((questions[-1], type(verdict.value).__name__, str(verdict.value)))
    assert outcomes[:2] == outcomes[2:] and [outcome[:2] for outcome in outcomes[:2]] == [
        (1, "EvidenceCorrupt"), (0, "ValueError")] and "size differs from its ref" in outcomes[0][2], outcomes
    with packs.open_index(roots[K_LARGE], write=True) as conn:  # the first copy's record claims a size over the limit
        conn.execute("UPDATE members SET size = size + 1000 WHERE task_id = 'task-00'")
    for ref in (refs[K_LARGE], {**refs[K_LARGE], "size": None}):
        assert packs.blob_bytes(roots[K_LARGE], ref, max_bytes=len(raw)) == raw


def test_a_task_s_own_sidecars_are_parsed_once_per_file_state_and_only_the_task_read_last_is_kept(tmp_path, monkeypatch):
    """Reassembling one view reads hundreds of its parts from one owner, whose sidecars can hold tens of
    thousands of members: a process keeps the parsed sidecars of the task it read last, every generation of it
    however many, each parsed once per file state and kept once per file; a look-alike's (task ``a.2`` beside
    ``a``) keeps only whose it is; a replaced file is read again (never answered from the old one), a read of
    another task replaces them, and a read naming a task without sidecars leaves them alone."""
    import os
    import pathlib

    from ouroboros import evidence_packer, evidence_packs as packs
    from tests.test_evidence_packs import _blob, _result, _task

    monkeypatch.setenv("OUROBOROS_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(packs, "_KEPT", ("", {}))  # an earlier test may have read a task of this id in its own root
    refs = {task: _task(tmp_path, task, blobs=3)[0] for task in ("a", "a.2", "b")}
    assert evidence_packer.run_pass(tmp_path)["faults"] == 0
    for generation in range(2, 6):  # written terminal again with new evidence: four generations after a.2's name
        refs["a"].append(_blob(tmp_path, json.dumps({"message": {"content": f"answer {generation}"}}).encode()))
        (tmp_path / "observability" / "calls" / "a" / f"llm_{generation}9_response.json").write_text(
            json.dumps({"call_id": f"llm_{generation}9", "full_payload_ref": refs["a"][-1]}), encoding="utf-8")
        _result(tmp_path, "a", "completed")
        assert evidence_packer.run_pass(tmp_path)["packed"] == 1
    names = sorted(packs._task_sidecar_names(tmp_path, "a"))
    sidecar = packs.packs_dir(tmp_path) / "a.members.json"
    decoded = _spy_sidecars(monkeypatch)
    first = [packs.blob_bytes(tmp_path, ref, task_id="a") for ref in refs["a"]]
    for _ in range(20):
        assert [packs.blob_bytes(tmp_path, ref, task_id="a") for ref in refs["a"]] == first
    assert len(names) == 6 and sorted(decoded) == names  # each generation once, whatever the number of reads
    look_alike = [brief for key, (_, brief) in packs._KEPT[1].items() if pathlib.Path(key).name == "a.2.members.json"]
    assert [(brief[0], brief[2], brief[3]) for brief in look_alike] == [("a.2", {}, {})]  # whose it is, no members
    decoded.clear()
    assert packs.blob_bytes(tmp_path, refs["b"][0], task_id="nobody")  # a task without sidecars: nothing parsed
    assert [packs.blob_bytes(tmp_path, ref, task_id="a") for ref in refs["a"]] == first and decoded == []
    assert packs.blob_bytes(tmp_path, refs["b"][0], task_id="b") and decoded == ["b.members.json"]
    assert {pathlib.Path(key).name for key in packs._KEPT[1]} == {"b.members.json"}  # a's are no longer held
    decoded.clear()
    for tick in range(1, 4):  # a's first generation answers its blob, alone; its file in three states, one entry
        os.utime(sidecar, ns=(tick * 10**9, tick * 10**9))
        assert packs.blob_bytes(tmp_path, refs["a"][0], task_id="a") == first[0]
    assert decoded == ["a.members.json"] * 3 and [pathlib.Path(key).name for key in packs._KEPT[1]] == ["a.members.json"]
    (packs.packs_dir(tmp_path) / packs.INDEX_NAME).unlink()  # no cache could vouch for the replaced file
    sidecar.write_text("{", encoding="utf-8")
    decoded.clear()
    with pytest.raises(packs.EvidenceUnavailable):  # its own sidecar unreadable and no cache to say more now
        packs.blob_bytes(tmp_path, refs["a"][0], task_id="a")
    assert decoded[0] == "a.members.json"  # read again (first, generation after generation), never the old parse



def test_a_read_of_a_task_s_first_generation_opens_no_look_alike_sidecar(tmp_path, monkeypatch):
    """A task's sidecars are probed generation after generation by spelling, so for task ``a`` the names of tasks
    ``a.2`` ... (look-alikes) come up too: a blob its first generation holds is answered there, the walk ending at
    that hit, so no look-alike is opened beside 1 or 30 of them; a blob of its second own generation, past the
    look-alikes, is still read."""
    from ouroboros import evidence_packer, evidence_packs as packs
    from tests.test_evidence_packs import _blob, _result, _task

    decoded, opened = _spy_sidecars(monkeypatch), {}
    for k in (K_SMALL, K_LARGE):
        root = tmp_path / f"k{k:03d}"
        monkeypatch.setenv("OUROBOROS_DATA_DIR", str(root))
        first = _task(root, "a", blobs=1)[0][0]
        for n in range(2, k + 2):
            _task(root, f"a.{n}", blobs=1)
        assert evidence_packer.run_pass(root)["faults"] == 0
        later = _blob(root, json.dumps({"message": {"content": "a later answer"}}).encode())
        (root / "observability" / "calls" / "a" / "llm_9_response.json").write_text(
            json.dumps({"call_id": "llm_9", "full_payload_ref": later}), encoding="utf-8")
        _result(root, "a", "completed")
        assert evidence_packer.run_pass(root)["packed"] == 1  # its second own generation, past the look-alikes
        monkeypatch.setattr(packs, "_KEPT", ("", {}))
        decoded.clear()
        assert packs.blob_bytes(root, first, task_id="a")
        opened[k] = list(decoded)
        assert json.loads(packs.blob_bytes(root, later, task_id="a")) == {"message": {"content": "a later answer"}}
    assert opened == {K_SMALL: ["a.members.json"], K_LARGE: ["a.members.json"]}, opened

# ---- the Host Service operation read: one accepted message over k-scaled room, routing and task history

OP_ROWS_PER_K, OP_NOTES_PER_K, OP_TASKS_PER_K = 20, 20, 10
OPERATIONS = {"managed": "op-managed", "direct": "op-direct", "pending": "op-pending"}


def _operation_history(*, damaged: bool = False):
    def build(root, k):
        """One host-accepted message per answer path (a managed task's, a direct turn's final row, none yet),
        each older than every unrelated row. Unrelated history grows with k: answers in the same room (each
        its own task), routing annotations of other messages, saved tasks started by other messages, with
        4 KB results. The maintenance job has run: the store is at rest, ``damaged`` adding a torn first row
        it can never derive and older-writer rows it derived after it."""
        from ouroboros import dialogue_store, store, store_hot_facts
        from ouroboros.project_dialogue import append_chat_annotation, build_owner_message_ref
        from ouroboros.task_results import write_task_result
        from ouroboros.task_store import read_document
        from tests._dialogue import seed
        from tests.test_host_service_api import _seed_token
        from tests.test_host_service_operations import CHAT, SKILL, TOKEN

        def inbound(message_id, second):
            return {"ts": f"2026-10-01T00:00:{second:02d}+00:00", "session_id": "s1", "direction": "in",
                    "chat_id": CHAT, "user_id": 0, "text": f"work {message_id}", "source": f"skill:{SKILL}",
                    "client_message_id": message_id, "transport": {}, "task_id": ""}

        accepted = {path: inbound(message_id, second) for second, (path, message_id) in enumerate(OPERATIONS.items())}
        refs = {path: build_owner_message_ref(chat_id=CHAT, client_message_id=row["client_message_id"], ts=row["ts"],
                                              text=row["text"]) for path, row in accepted.items()}
        final = {"ts": "2026-10-01T00:01:00+00:00", "direction": "out", "chat_id": CHAT, "text": "direct answer",
                 "task_id": "turn-direct", "origin_message_ref": refs["direct"], "task_terminal_status": "completed"}
        unrelated = [{"ts": "2026-10-01T01:00:00+00:00", "direction": "out", "chat_id": CHAT, "text": "an answer",
                      "task_id": f"other-{unit}-{index}"} for unit in range(k) for index in range(OP_ROWS_PER_K)]
        seed(root, live={"chat": [*accepted.values(), final, *unrelated]})
        dialogue_store.ensure(root)  # the production cutover stores the chain, oldest first
        (root / "state").mkdir(parents=True, exist_ok=True)
        (root / "state" / "state.json").write_text(json.dumps({"session_id": "s1"}), encoding="utf-8")
        _seed_token(root, skill=SKILL, token=TOKEN, permissions=["inject_chat"])
        write_task_result(root, "task-managed", "completed", result="managed answer", origin_message_ref=refs["managed"])
        write_task_result(root, "template", "completed", chat_id=CHAT, result="r" * 4000, ts="2023-01-01T00:00:00+00:00")
        template = read_document(root, "template")
        for unit in range(k):
            with store.open_store(root, write=True) as txn:
                for index in range(OP_TASKS_PER_K):
                    other = build_owner_message_ref(chat_id=CHAT, client_message_id=f"other-{unit}-{index}",
                                                    ts="2026-10-01T01:00:00+00:00", text="an answer")
                    _stored(txn, template, f"other-{unit}-{index}", "2026-10-01T01:00:00+00:00", origin_message_ref=other)
            for index in range(OP_NOTES_PER_K):
                assert append_chat_annotation(root, f"other-{unit}-{index}", action="promote_chat_to_task",
                                              target=f"other-{unit}-{index}", status="scheduled")
        if damaged:
            with store.open_store(root, write=True) as txn:
                _stored(txn, template, "0-torn", "2023-01-01T00:00:00+00:00")
                txn.conn.execute("UPDATE task_bodies SET body='{' WHERE task_id='0-torn'")
                txn.conn.execute("UPDATE tasks SET origin=NULL")  # as older writers left every row
        try:
            store_hot_facts.run_hot_facts(root)
        except ValueError:
            assert damaged, "only the torn row may stop the job"
        with store.open_store(root, write=False) as txn:
            assert [row[0] for row in txn.conn.execute("SELECT task_id FROM tasks WHERE origin IS NULL")] == (
                ["0-torn"] if damaged else [])
        return {}
    return build


def _operation_reads(root, monkeypatch):
    """The real ``GET /chat/operations/{ref}`` handler on a real Starlette request, run on this thread's loop."""
    import asyncio

    from starlette.requests import Request

    from ouroboros.gateway import host_service
    from tests.test_host_service_operations import CHAT, TOKEN, _isolate_queue

    _isolate_queue(monkeypatch, root, [])
    app = host_service.create_host_service_app(root, bridge_getter=lambda: None)

    def read(message_id):
        operation = f"{CHAT}:{message_id}"
        request = Request({"type": "http", "method": "GET", "path": f"/chat/operations/{operation}",
                           "headers": [(b"x-skill-token", TOKEN.encode())], "query_string": b"",
                           "path_params": {"operation_ref": operation}, "app": app})
        response = asyncio.run(host_service._api_chat_operation(request))
        return response.status_code, json.loads(response.body)

    return lambda: {path: read(message_id) for path, message_id in OPERATIONS.items()}


@pytest.mark.parametrize("damaged", [False, True])
def test_an_operation_read_costs_the_same_beside_1_or_30_units_of_room_routing_and_task_history(
        tmp_path, monkeypatch, damaged):
    """Each answer path reads its own records: the accepted row by its name, the tasks its typed origin
    names, the dialogue rows indexed under it. Unrelated answers in the same room, routing annotations and
    tasks other messages started are never visited, nor a damaged row the job named once (it is read only
    if its stored text could name the message). Both sizes give the same answers."""
    def boundary(root, _facts):
        reads = _operation_reads(root, monkeypatch)
        reads()
        return reads

    answers = _flat(tmp_path, monkeypatch, _operation_history(damaged=damaged), boundary)
    assert answers[K_SMALL] == answers[K_LARGE]
    shown = {path: (status, body.get("status"), body.get("text"), body.get("task_id"))
             for path, (status, body) in answers[K_SMALL].items()}
    assert shown == {"managed": (200, "completed", "managed answer", "task-managed"),
                     "direct": (200, "completed", "direct answer", None),
                     "pending": (200, "pending", None, None)}


# ---- the consciousness tick: its allowance read over k days of usage history ---------------------

NOW = 1_790_000_000.0  # a fixed clock: the window and the horizon are measured from it
DAY = 86_400.0


def _journal_rows(attempt_id, ts, root_task_id, task_id, category, cost=0.01):
    """One settled attempt in the retired journal format the store's one-time import reads."""
    import datetime as dt

    base = {"attempt_id": attempt_id, "kind": "attempt", "ts": dt.datetime.fromtimestamp(ts, dt.timezone.utc).isoformat(),
            "model": "m", "provider": "p", "category": category, "source": "test", "task_id": task_id,
            "root_task_id": root_task_id, "parent_task_id": ""}
    return [{**base, "state": "reserved", "reservation_upper_bound_usd": "1"}, {**base, "state": "dispatched"},
            {**base, "state": "settled", "cost_usd": str(cost), "cost_final": True, "pricing_known": True}]


def _usage_history(shape):
    def build(root, k):
        """The same last-24-hours window at every size (two own wakes, one started root); each older day
        adds three own wakes and three owner tasks, and ``long_lived`` adds rows of one started root
        that is still at work today."""
        from tests._usage_store_testing import write_journal

        rows = []
        for wake in range(2):
            for index in range(5):
                rows += _journal_rows(f"now-wake{wake}-{index}", NOW - 3600 * (wake + 1) - index,
                                      f"now-wake-{wake}", f"now-wake-{wake}", "consciousness")
        for index in range(5):
            rows += _journal_rows(f"now-started-{index}", NOW - 7200 - index, "now-started", "now-started",
                                  "consciousness_task")
        if shape == "long_lived":
            rows += _journal_rows("long-today", NOW - 1800, "long-root", "long-root", "consciousness_task")
        for day in range(1, k + 1):
            at = NOW - day * DAY - 3 * 3600
            for root_kind, category in (("wake", "consciousness"), ("task", "task")):
                for number in range(3):
                    for index in range(5):
                        rows += _journal_rows(f"d{day}-{root_kind}{number}-{index}", at - 60 * number - index,
                                              f"d{day}-{root_kind}-{number}", f"d{day}-{root_kind}-{number}", category)
            if shape == "long_lived":
                for index in range(5):
                    rows += _journal_rows(f"d{day}-long-{index}", at - index, "long-root", f"long-step-{day}",
                                          "consciousness_task" if index == 0 else "task")
        (root / "state").mkdir(parents=True, exist_ok=True)
        write_journal(root, rows)
        return {}
    return build


@pytest.mark.parametrize("shape", ["separate_trees", "long_lived"])
def test_one_tick_allowance_read_costs_the_same_with_1_or_30_days_of_usage(tmp_path, monkeypatch, shape):
    """The tick reads exactly the addressed window: the consciousness roots of the last 48 hours by
    their category and time, then each root's money rows of the last 24 hours by root and time. Older
    trees are never visited (the quiet case even before the fix); a root still at work today is read
    for its last day, not its whole life. The decision is the same at both sizes."""
    from ouroboros import consciousness_allowance as allowance, usage_store

    monkeypatch.setenv("OUROBOROS_CONSCIOUSNESS_DAILY_USD", "20")
    monkeypatch.setenv("OUROBOROS_SETTINGS_PATH", str(tmp_path / "settings.json"))

    def boundary(root, _facts):
        assert usage_store.read_usage_records(root)  # the one-time import an upgraded install ran at boot
        usage_store.migrate_from_journal(root, initialize_journal=True)  # the lifespan's boot step
        allowance.allowance_window(root, now=NOW)

        def tick_read():
            window = allowance.allowance_window(root, now=NOW)  # exact, as the tick admits a wake
            return {key: window[key] for key in ("status", "settled_usd", "remaining_usd", "roots", "window_rows",
                                                 "resets_at")}
        return tick_read

    try:
        answers = _flat(tmp_path, monkeypatch, _usage_history(shape), boundary)
    finally:
        for k in (K_SMALL, K_LARGE):
            usage_store.forget(tmp_path / f"k{k:03d}" / "data")
    assert answers[K_SMALL] == answers[K_LARGE]
    assert answers[K_SMALL]["status"] == "available" and answers[K_SMALL]["window_rows"] == 15 + (shape == "long_lived")


def test_a_store_an_older_body_created_gains_the_window_index_once_at_boot(tmp_path, monkeypatch):
    """The index is built by the lifespan's boot step, never by the tick: before it the tick still
    answers (through the root index alone), the boot creates it once and later boots only look."""
    from ouroboros import consciousness_allowance as allowance, usage_store

    root = tmp_path / "data"
    monkeypatch.setenv("OUROBOROS_DATA_DIR", str(root))
    monkeypatch.setenv("OUROBOROS_SETTINGS_PATH", str(root / "settings.json"))
    monkeypatch.setenv("OUROBOROS_CONSCIOUSNESS_DAILY_USD", "20")
    _usage_history("long_lived")(root, 3)
    try:
        assert usage_store.read_usage_records(root)
        with usage_store.hold(root, write=True) as txn:
            txn.conn.execute("DROP INDEX attempts_root_time")  # as an older body created the store
        before = allowance.allowance_window(root, now=NOW)
        with usage_store.read(root) as txn:
            assert txn.conn.execute("SELECT 1 FROM sqlite_master WHERE name='attempts_root_time'").fetchone() is None
        assert usage_store.migrate_from_journal(root, initialize_journal=True)["window_index"] == "created"
        assert usage_store.migrate_from_journal(root, initialize_journal=True)["window_index"] == "present"
        with usage_store.read(root) as txn:
            plan = [tuple(row)[-1] for row in txn.conn.execute(
                f"EXPLAIN QUERY PLAN SELECT * FROM attempts WHERE root_task_id = ? AND ts_last_epoch >= ? "
                f"AND {allowance._MONEY_KIND} ORDER BY seq", ("long-root", NOW - DAY))]
        assert any("attempts_root_time" in step for step in plan), plan
        assert allowance.allowance_window(root, now=NOW) == before
    finally:
        usage_store.forget(root)


# ---- the round's source transfer: an applied fold's receipt promoted beside k-scaled packed history -------------

TRANSFER_TASKS = ("warm-up", "turn")


def _transfer_history(root, k):
    """``_packed_history``'s unrelated packed tasks in the canonical root, and for each live task a drive holding an
    applied fold's receipt and the evidence it names, all loose beside the producer: two request views (a skeleton and
    its parts), a response and a tool result, with the call manifests that carry them."""
    import hashlib

    from ouroboros import headless, observability as obs
    from ouroboros.artifacts import store_actor_source_bytes

    _packed_history(root, k)
    facts = {}
    for task in TRANSFER_TASKS:
        child = headless.prepare_task_drive(root, task, "empty")
        settled = [{"role": "user", "content": f"settled turn {index} of {task} " + "x" * 400} for index in range(3)]
        refs = [obs.persist_call(child, task_id=task, call_id=f"llm_{index}_request", call_type="llm_request",
                                 payload={"messages": [*settled, {"role": "user", "content": f"question {index}"}]}
                                 )["manifest_ref"] for index in range(2)]
        for call_id, payload in (("llm_1_response", {"message": {"role": "assistant", "content": "an answer"}}),
                                 ("tool_read_file_0", {"tool": "read_file", "result": "y" * 2000})):
            refs.append(obs.persist_call(child, task_id=task, call_id=call_id, call_type="tool_call",
                                         payload=payload)["manifest_ref"])
        raw = json.dumps({"stage": "candidate_materialization", "published": False,
                          "receipt": {"status": "applied", "refs": refs}}, sort_keys=True).encode("utf-8")
        source = store_actor_source_bytes(child, task, category="context_checkpoints",
                                          source_id=hashlib.sha256(raw).hexdigest(), data=raw, extension="json")
        facts[task] = {"child": child, "source": source, "raw": raw}
    return facts


@pytest.mark.parametrize("index", ["valid", "contended", "damaged"])
def test_a_fold_receipt_transfer_is_flat_across_evidence_packs(tmp_path, monkeypatch, request, index):
    """After an applied fold the round's prologue promotes the receipt and everything it names into the canonical
    root (``promote_task_source_ref``, ``source_retention.retain_tree``) before the next model call. The bytes lie
    loose beside their producer; the canonical root only has to prove or receive them. However its packs catalog
    stands (``valid``; ``contended``: the packer holds its lock past any wait; ``damaged``: the file awaits the
    packing pass's rebuild), the transfer lists no packs directory,
    costs the same beside 1 or 30 units of packed history and never waits for the catalog: a destination it cannot
    prove now is carried and read back from the parent. The receipt and its sources are retained on both sizes and
    nothing is marked unavailable (a pending address would also do; an unavailable one never)."""
    from ouroboros import evidence_packs, observability as obs, store
    from ouroboros.artifacts import read_actor_source_bytes

    def boundary(root, facts):
        patches = pytest.MonkeyPatch()  # the lock spy is this size's alone: undone after the counted call
        request.addfinalizer(patches.undo)

        def promote(task):
            try:
                obs.promote_task_source_ref(facts[task]["child"], root, facts[task]["source"], task_id=task)
                return "retained"
            except obs.ObservabilityPromotionSourceError:
                return "unavailable"
            except OSError:
                return "pending"

        assert promote("warm-up") == "retained"
        holder, waits, thread = None, [], threading.get_ident()
        if index == "contended":
            held, release = threading.Event(), threading.Event()

            def hold():
                with store.name_lock(root, evidence_packs.INDEX_LOCK_NAME, timeout_sec=5):
                    held.set()
                    release.wait(600)

            holder = threading.Thread(target=hold, name="packer-holds-index")
            holder.start()
            assert held.wait(5)
        elif index == "damaged":
            (evidence_packs.packs_dir(root) / evidence_packs.INDEX_NAME).write_bytes(b"not a database")
        real_lock = store.name_lock

        def name_lock(where, name, *args, **kwargs):
            if name == evidence_packs.INDEX_LOCK_NAME and threading.get_ident() == thread:
                waits.append(kwargs.get("timeout_sec"))
            return real_lock(where, name, *args, **kwargs)

        patches.setattr(store, "name_lock", name_lock)

        def transfer():
            try:
                try:
                    outcome = promote("turn")
                finally:
                    if holder is not None:
                        release.set()
                        holder.join(10)
                fact = facts["turn"]
                kept = outcome == "retained" and read_actor_source_bytes(root, "turn", fact["source"]) == fact["raw"]
                return outcome, kept, max(waits, default=0)
            finally:
                patches.undo()
        return transfer

    answers = _flat(tmp_path, monkeypatch, _transfer_history, boundary, unlisted=("packs",))
    assert answers[K_SMALL] == answers[K_LARGE] == ("retained", True, 0), answers
