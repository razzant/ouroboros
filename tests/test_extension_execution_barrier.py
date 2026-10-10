"""F3: the shared-reader / exclusive-writer in-process execution barrier.

Every case here is EVENT-GATED: overlap is proven by both parties standing
inside their scope at the same instant, exclusion by a party that cannot enter
while a gate is held and does enter once it is dropped.  No case asserts a
timing window as evidence.

The contract under test (``research/implementation-decisions.md``, F3):
a lease belongs to one successful ENTER, not to a thread or task; readers
overlap; a writer is exclusive and blocks new readers once it declares intent;
scopes stay NON-REENTRANT; every acquired lease is released exactly once.
"""
from __future__ import annotations

import asyncio
import sys
import threading

import pytest

from ouroboros import extension_isolated_deps as deps
from ouroboros import extension_loader
from tests._extension_loader_shared import (  # noqa: F401  (autouse fixture)
    _clear_loader_state,
    _mark_isolated_deps_installed,
    _prepare_extension,
)

GATE = 5.0  # generous upper bound: every wait below is released by an event


@pytest.fixture(autouse=True)
def _barrier_is_free_before_and_after():
    """No test may start or leave a leaked lease on the module-global barrier."""
    assert _free(), "execution barrier was not free at test start"
    yield
    assert _free(), "execution barrier was not free at test end"


def _free() -> bool:
    barrier = deps._execution_lock
    with barrier.condition:
        return (not barrier.writer_active and barrier.readers == 0
                and barrier.writers_waiting == 0 and not barrier.active_keys)


def _reader(skill_dir, entered: threading.Event, release: threading.Event, sink: list):
    def run():
        try:
            with deps.isolated_site_dirs_scope(skill_dir, enabled=False):
                entered.set()
                release.wait(GATE)
                sink.append("reader")
        except BaseException as exc:  # recorded, never swallowed silently
            sink.append(exc)
    return run


def _writer(skill_dir, entered: threading.Event, release: threading.Event, sink: list):
    def run():
        try:
            with deps.isolated_site_dirs_scope(skill_dir, enabled=True):
                entered.set()
                release.wait(GATE)
                sink.append("writer")
        except BaseException as exc:
            sink.append(exc)
    return run


# --------------------------------------------------------------------------
# Reader overlap: sync, async and mixed
# --------------------------------------------------------------------------

def test_three_sync_readers_stand_in_scope_simultaneously(tmp_path):
    meeting = threading.Barrier(3)
    errors: list = []

    def run(label):
        try:
            with deps.isolated_site_dirs_scope(tmp_path / label, enabled=False):
                meeting.wait(timeout=GATE)  # only passes if all three are inside
        except BaseException as exc:
            errors.append(exc)

    threads = [threading.Thread(target=run, args=(f"r{index}",)) for index in range(3)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=GATE)
    assert not errors, errors
    assert not any(thread.is_alive() for thread in threads)


def test_three_async_readers_stand_in_scope_simultaneously(tmp_path):
    async def main():
        inside = asyncio.Event()
        count = {"n": 0}
        release = asyncio.Event()

        async def scope(label):
            async with deps.async_isolated_site_dirs_scope(tmp_path / label, enabled=False):
                count["n"] += 1
                if count["n"] == 3:
                    inside.set()
                await asyncio.wait_for(release.wait(), GATE)

        tasks = [asyncio.create_task(scope(f"a{index}")) for index in range(3)]
        await asyncio.wait_for(inside.wait(), GATE)
        assert count["n"] == 3
        release.set()
        await asyncio.wait_for(asyncio.gather(*tasks), GATE)

    asyncio.run(main())


def test_sync_and_async_readers_overlap(tmp_path):
    sync_entered = threading.Event()
    sync_release = threading.Event()
    sink: list = []
    thread = threading.Thread(target=_reader(tmp_path / "s", sync_entered, sync_release, sink))

    async def main():
        thread.start()
        assert sync_entered.wait(GATE)
        async with deps.async_isolated_site_dirs_scope(tmp_path / "a", enabled=False):
            # The sync reader is still inside its scope right now.
            assert not sync_release.is_set()
            with deps._execution_lock.condition:
                assert deps._execution_lock.readers == 2

    asyncio.run(main())
    sync_release.set()
    thread.join(timeout=GATE)
    assert sink == ["reader"], sink


# --------------------------------------------------------------------------
# Writer exclusion, both orders, sync / async / mixed
# --------------------------------------------------------------------------

def test_writer_waits_for_a_running_reader_then_excludes_it(tmp_path):
    reader_in, reader_out = threading.Event(), threading.Event()
    writer_in, writer_out = threading.Event(), threading.Event()
    sink: list = []
    reader = threading.Thread(target=_reader(tmp_path / "r", reader_in, reader_out, sink))
    writer = threading.Thread(target=_writer(tmp_path / "w", writer_in, writer_out, sink))

    reader.start()
    assert reader_in.wait(GATE)
    writer.start()
    assert not writer_in.wait(0.2), "writer entered while a reader held the barrier"
    reader_out.set()
    assert writer_in.wait(GATE)
    writer_out.set()
    reader.join(timeout=GATE)
    writer.join(timeout=GATE)
    assert sink == ["reader", "writer"], sink


def test_reader_waits_for_a_running_writer(tmp_path):
    writer_in, writer_out = threading.Event(), threading.Event()
    reader_in, reader_out = threading.Event(), threading.Event()
    sink: list = []
    writer = threading.Thread(target=_writer(tmp_path / "w", writer_in, writer_out, sink))
    reader = threading.Thread(target=_reader(tmp_path / "r", reader_in, reader_out, sink))

    writer.start()
    assert writer_in.wait(GATE)
    reader.start()
    assert not reader_in.wait(0.2), "reader entered while a writer held the barrier"
    writer_out.set()
    assert reader_in.wait(GATE)
    reader_out.set()
    writer.join(timeout=GATE)
    reader.join(timeout=GATE)
    assert sink == ["writer", "reader"], sink


def test_async_writer_waits_for_sync_reader_without_blocking_the_loop(tmp_path):
    """The async acquisition polls; the ASGI loop keeps running other work."""
    reader_in, reader_out = threading.Event(), threading.Event()
    sink: list = []
    reader = threading.Thread(target=_reader(tmp_path / "r", reader_in, reader_out, sink))

    async def main():
        reader.start()
        assert reader_in.wait(GATE)
        pulses = {"n": 0}

        async def pulse():
            while True:
                pulses["n"] += 1
                await asyncio.sleep(0.005)

        heartbeat = asyncio.create_task(pulse())

        async def writer_scope():
            async with deps.async_isolated_site_dirs_scope(tmp_path / "w", enabled=True):
                return pulses["n"]

        task = asyncio.create_task(writer_scope())
        await asyncio.sleep(0.15)
        assert not task.done(), "async writer entered while a sync reader held the barrier"
        before = pulses["n"]
        assert before > 5, f"the event loop stalled while the writer waited: {before}"
        reader_out.set()
        entered_at = await asyncio.wait_for(task, GATE)
        assert entered_at >= before
        heartbeat.cancel()
        with pytest.raises(asyncio.CancelledError):
            await heartbeat

    asyncio.run(main())
    reader.join(timeout=GATE)
    assert sink == ["reader"], sink


def test_async_writer_excludes_a_later_sync_reader(tmp_path):
    reader_in, reader_out = threading.Event(), threading.Event()
    sink: list = []
    reader = threading.Thread(target=_reader(tmp_path / "r", reader_in, reader_out, sink))

    async def main():
        async with deps.async_isolated_site_dirs_scope(tmp_path / "w", enabled=True):
            reader.start()
            assert not reader_in.wait(0.2), "reader entered under an async writer"
        assert reader_in.wait(GATE)

    asyncio.run(main())
    reader_out.set()
    reader.join(timeout=GATE)
    assert sink == ["reader"], sink


# --------------------------------------------------------------------------
# Queued-writer progress: a declared writer is not starved by later readers
# --------------------------------------------------------------------------

def test_queued_writer_runs_before_readers_that_arrive_after_its_intent(tmp_path):
    first_in, first_out = threading.Event(), threading.Event()
    writer_in, writer_out = threading.Event(), threading.Event()
    late_in, late_out = threading.Event(), threading.Event()
    sink: list = []
    first = threading.Thread(target=_reader(tmp_path / "r1", first_in, first_out, sink))
    writer = threading.Thread(target=_writer(tmp_path / "w", writer_in, writer_out, sink))
    late = threading.Thread(target=_reader(tmp_path / "r2", late_in, late_out, sink))

    first.start()
    assert first_in.wait(GATE)
    writer.start()
    # Wait until the writer's intent is visible, then let a new reader arrive.
    deadline = threading.Event()
    for _ in range(int(GATE * 100)):
        with deps._execution_lock.condition:
            if deps._execution_lock.writers_waiting == 1:
                break
        deadline.wait(0.01)
    else:  # pragma: no cover - the writer always registers its intent
        pytest.fail("queued writer never declared intent")
    late.start()
    assert not late_in.wait(0.2), "a later reader jumped a queued writer"
    first_out.set()
    assert writer_in.wait(GATE), "queued writer never ran after the readers drained"
    assert not late_in.is_set()
    writer_out.set()
    assert late_in.wait(GATE)
    late_out.set()
    for thread in (first, writer, late):
        thread.join(timeout=GATE)
    assert sink == ["reader", "writer", "reader"], sink


def test_async_queued_writer_holds_back_later_async_readers(tmp_path):
    async def main():
        first_inside = asyncio.Event()
        first_release = asyncio.Event()
        order: list = []

        async def first_reader():
            async with deps.async_isolated_site_dirs_scope(tmp_path / "r1", enabled=False):
                first_inside.set()
                await asyncio.wait_for(first_release.wait(), GATE)
                order.append("reader-1")

        async def writer():
            async with deps.async_isolated_site_dirs_scope(tmp_path / "w", enabled=True):
                order.append("writer")

        async def late_reader():
            async with deps.async_isolated_site_dirs_scope(tmp_path / "r2", enabled=False):
                order.append("reader-2")

        one = asyncio.create_task(first_reader())
        await asyncio.wait_for(first_inside.wait(), GATE)
        two = asyncio.create_task(writer())
        for _ in range(int(GATE * 100)):  # let the writer register its intent
            with deps._execution_lock.condition:
                if deps._execution_lock.writers_waiting == 1:
                    break
            await asyncio.sleep(0.01)
        three = asyncio.create_task(late_reader())
        await asyncio.sleep(0.1)
        assert not two.done() and not three.done()
        first_release.set()
        await asyncio.wait_for(asyncio.gather(one, two, three), GATE)
        assert order == ["reader-1", "writer", "reader-2"], order

    asyncio.run(main())


# --------------------------------------------------------------------------
# Cancellation
# --------------------------------------------------------------------------

def test_async_writer_cancelled_while_waiting_drops_its_intent(tmp_path):
    """A cancelled waiting writer must not leave readers queued behind a ghost."""
    reader_in, reader_out = threading.Event(), threading.Event()
    sink: list = []
    reader = threading.Thread(target=_reader(tmp_path / "r", reader_in, reader_out, sink))

    async def main():
        reader.start()
        assert reader_in.wait(GATE)

        async def writer_scope():
            async with deps.async_isolated_site_dirs_scope(tmp_path / "w", enabled=True):
                pytest.fail("cancelled writer must never enter")

        task = asyncio.create_task(writer_scope())
        for _ in range(int(GATE * 100)):
            with deps._execution_lock.condition:
                if deps._execution_lock.writers_waiting == 1:
                    break
            await asyncio.sleep(0.01)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        with deps._execution_lock.condition:
            assert deps._execution_lock.writers_waiting == 0
        # A new reader can still enter while the first one holds its lease.
        async with deps.async_isolated_site_dirs_scope(tmp_path / "r2", enabled=False):
            with deps._execution_lock.condition:
                assert deps._execution_lock.readers == 2

    asyncio.run(main())
    reader_out.set()
    reader.join(timeout=GATE)
    assert sink == ["reader"], sink


def test_async_scope_cancelled_inside_its_body_releases_the_lease(tmp_path):
    for writer in (False, True):
        async def main():
            inside = asyncio.Event()

            async def scope():
                async with deps.async_isolated_site_dirs_scope(tmp_path / "x", enabled=writer):
                    inside.set()
                    await asyncio.sleep(GATE)

            task = asyncio.create_task(scope())
            await asyncio.wait_for(inside.wait(), GATE)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

        asyncio.run(main())
        assert _free(), f"cancelled body leaked a lease (writer={writer})"


def test_cancellation_racing_the_grant_never_leaks_a_lease(tmp_path):
    """Cancel repeatedly at the exact moment the poll can succeed."""
    for attempt in range(25):
        async def main():
            holder_in, holder_out = threading.Event(), threading.Event()
            sink: list = []
            holder = threading.Thread(target=_writer(tmp_path / "h", holder_in, holder_out, sink))
            holder.start()
            assert holder_in.wait(GATE)

            entered = {"yes": False}

            async def scope():
                async with deps.async_isolated_site_dirs_scope(tmp_path / "x", enabled=True):
                    entered["yes"] = True

            task = asyncio.create_task(scope())
            await asyncio.sleep(0.02)  # the poll loop is running
            loop = asyncio.get_running_loop()
            # Free the barrier and cancel in the SAME loop iteration window, so
            # the cancellation and the successful try_enter race each other.
            holder_out.set()
            loop.call_soon(task.cancel)
            try:
                await asyncio.wait_for(task, GATE)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass
            holder.join(timeout=GATE)
            assert sink == ["writer"], sink

        asyncio.run(main())
        assert _free(), f"grant/cancel race leaked a lease on attempt {attempt}"


# --------------------------------------------------------------------------
# Exceptions in the body and in cleanup
# --------------------------------------------------------------------------

@pytest.mark.parametrize("writer", [False, True])
def test_body_exception_releases_the_matching_lease(tmp_path, writer, monkeypatch):
    monkeypatch.setattr(deps, "inject_isolated_site_dirs", lambda _dir: [])
    with pytest.raises(RuntimeError, match="body"):
        with deps.isolated_site_dirs_scope(tmp_path, enabled=writer):
            raise RuntimeError("body")
    assert _free()


@pytest.mark.parametrize("writer", [False, True])
def test_async_body_exception_releases_the_matching_lease(tmp_path, writer, monkeypatch):
    monkeypatch.setattr(deps, "inject_isolated_site_dirs", lambda _dir: [])

    async def main():
        async with deps.async_isolated_site_dirs_scope(tmp_path, enabled=writer):
            raise RuntimeError("body")

    with pytest.raises(RuntimeError, match="body"):
        asyncio.run(main())
    assert _free()


@pytest.mark.parametrize("writer", [False, True])
def test_cleanup_exception_still_releases_the_matching_lease(tmp_path, writer, monkeypatch):
    monkeypatch.setattr(deps, "inject_isolated_site_dirs", lambda _dir: ["/nowhere"])

    def fail_cleanup(_site_dirs):
        raise RuntimeError("cleanup failed")

    monkeypatch.setattr(deps, "release_isolated_site_dirs", fail_cleanup)
    with deps.isolated_site_dirs_scope(tmp_path, enabled=writer):
        pass
    assert _free()


def test_release_without_enter_is_refused_in_both_directions():
    barrier = deps._ExecutionBarrier()
    with pytest.raises(RuntimeError, match="writer lease"):
        barrier.release(writer=True)
    with pytest.raises(RuntimeError, match="reader lease"):
        barrier.release(writer=False)
    assert barrier.try_enter(False)
    with pytest.raises(RuntimeError, match="writer lease"):
        barrier.release(writer=True)  # a reader lease is not a writer lease
    barrier.release(writer=False)


# --------------------------------------------------------------------------
# Non-reentrancy — the unchanged, deliberately unsupported topology
# --------------------------------------------------------------------------

def test_scopes_are_not_reentrant_and_the_lease_is_not_thread_owned():
    """A nested enter BLOCKS (it is not re-entered), and any thread may release.

    Run on a private barrier so a deliberately blocked nesting can never leak
    into the module-global one, and unwedged from the main thread — which is
    itself the proof that a lease belongs to an ENTER, not to a thread.
    """
    barrier = deps._ExecutionBarrier()
    nested_entered = threading.Event()
    outer_entered = threading.Event()

    def run():
        barrier.acquire(writer=False)
        outer_entered.set()
        barrier.acquire(writer=True)  # blocks: the thread's own reader lease excludes it
        nested_entered.set()
        barrier.release(writer=True)

    thread = threading.Thread(target=run)
    thread.start()
    assert outer_entered.wait(GATE)
    assert not nested_entered.wait(0.2), "a nested writer re-entered a held reader lease"
    barrier.release(writer=False)  # released by a DIFFERENT thread than the one that entered
    assert nested_entered.wait(GATE)
    thread.join(timeout=GATE)
    assert not thread.is_alive()


def test_non_reentrancy_is_observable_without_blocking():
    barrier = deps._ExecutionBarrier()
    assert barrier.try_enter(False)
    assert not barrier.try_enter(True), "a writer must not join a live reader lease"
    assert barrier.try_enter(False), "independent readers still overlap"
    barrier.release(writer=False)
    barrier.release(writer=False)
    assert barrier.try_enter(True)
    assert not barrier.try_enter(False), "a reader must not join a live writer lease"
    assert not barrier.try_enter(True), "a writer must not join a live writer lease"
    barrier.release(writer=True)


# --------------------------------------------------------------------------
# Scoped worker lifetime: cancelling the awaiting task cannot free the thread
# --------------------------------------------------------------------------

def test_cancelled_to_thread_task_does_not_release_the_worker_lease(tmp_path):
    """The documented sync-in-thread topology: the lease lives in the worker.

    Cancelling the coroutine that awaits ``asyncio.to_thread`` does not unwind
    the worker, so its lease must survive the cancellation and be released only
    when the real body returns.
    """
    release = threading.Event()
    inside = threading.Event()
    finished = threading.Event()

    def worker():
        with deps.isolated_site_dirs_scope(tmp_path / "worker", enabled=False):
            inside.set()
            release.wait(GATE)
        finished.set()

    async def main():
        task = asyncio.create_task(asyncio.to_thread(worker))
        await asyncio.wait_for(asyncio.to_thread(inside.wait, GATE), GATE)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        # The worker body is still inside its scope: a writer cannot enter.
        assert not deps._execution_lock.try_enter(True)
        with deps._execution_lock.condition:
            assert deps._execution_lock.readers == 1

    asyncio.run(main())
    release.set()
    assert finished.wait(GATE)
    assert _free()


# --------------------------------------------------------------------------
# Real imports through the loader, not just the barrier primitive
# --------------------------------------------------------------------------

def test_two_no_deps_extension_loads_overlap_with_real_imports(tmp_path):
    """Two no-deps plugin imports run concurrently — the change F3 actually buys.

    Each plugin body blocks on a shared barrier at import time; a serialized
    lock would make the barrier time out instead of passing.
    """
    meeting = threading.Barrier(2)
    import builtins

    builtins._ouro_1195_meeting = meeting  # module-body rendezvous, removed below
    try:
        prepared = []
        for label in ("overlap_one", "overlap_two"):
            (tmp_path / label).mkdir(parents=True, exist_ok=True)
            prepared.append(_prepare_extension(
                tmp_path / label,
                label,
                (
                    "import builtins\n"
                    "builtins._ouro_1195_meeting.wait(timeout=5)\n"
                    "def register(api):\n"
                    f"    api.register_tool('ping', lambda ctx: '{label}', description='p', schema={{}})\n"
                ),
                permissions=["tool"],
            ))
        errors: list = []

        def load(loaded, drive_root):
            try:
                errors.append(extension_loader.load_extension(
                    loaded, lambda: {}, drive_root=drive_root, _force_in_process=True))
            except BaseException as exc:
                errors.append(exc)

        threads = [
            threading.Thread(target=load, args=(loaded, drive_root))
            for loaded, _repo, drive_root in prepared
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=GATE)
        assert not any(thread.is_alive() for thread in threads), "no-deps loads were serialized"
        assert errors == [None, None], errors
        for label in ("overlap_one", "overlap_two"):
            name = extension_loader.extension_surface_name(label, "ping")
            assert extension_loader.get_tool(name) is not None
            extension_loader.unload_extension(label)
    finally:
        del builtins._ouro_1195_meeting


class _SweepGate:
    """A path-entry finder whose ``invalidate_caches`` runs INSIDE CPython's
    ``PathFinder.invalidate_caches`` sweep, after its snapshot of
    ``sys.path_importer_cache``.  It never claims a module."""

    def __init__(self, on_sweep):
        self._on_sweep = on_sweep

    def find_spec(self, fullname, target=None):
        return None

    def invalidate_caches(self):
        self._on_sweep()


def test_overlapping_loads_never_overlap_their_importer_cache_sweeps(tmp_path, monkeypatch):
    """The macOS CI failure of the overlap case above, made deterministic.

    ``PathFinder.invalidate_caches`` snapshots ``sys.path_importer_cache`` and
    then deletes each ``None`` entry by key; a second sweep that deletes the
    key first makes the earlier one raise ``KeyError`` (CI: ``python310.zip``)
    and the other plugin's rendezvous then breaks.  A gate parked in the cache
    pauses the first load INSIDE its sweep, before a ``None`` entry it still
    has to delete, until the second load has either swept past that entry (the
    race) or waits on the importer-cache lock (the repair).  Event-gated both
    ways; the plugin bodies still meet, so the sweep lock did not serialize
    the imports.
    """
    first_paused, second_moved = threading.Event(), threading.Event()
    waited: list = []

    def before_entry():
        if threading.current_thread().name == "sweep-first" and not first_paused.is_set():
            first_paused.set()
            waited.append(second_moved.wait(GATE))

    def after_entry():
        if threading.current_thread().name == "sweep-second":
            second_moved.set()

    class _SignallingLock:
        """``deps._lock`` that reports the second load waiting behind the first."""

        def __init__(self, inner):
            self._inner = inner

        def __enter__(self):
            if not self._inner.acquire(blocking=False):
                if threading.current_thread().name == "sweep-second":
                    second_moved.set()
                self._inner.acquire()
            return self

        def __exit__(self, *exc_info):
            self._inner.release()

    monkeypatch.setattr(deps, "_lock", _SignallingLock(threading.RLock()))
    keys = [str(tmp_path / name) for name in ("gate-before", "absent.zip", "gate-after")]
    import builtins

    builtins._ouro_sweep_meeting = threading.Barrier(2)
    threads = []
    try:
        prepared = {}
        for label in ("sweep_one", "sweep_two"):
            (tmp_path / label).mkdir()
            prepared[label] = _prepare_extension(
                tmp_path / label,
                label,
                (
                    "import builtins\n"
                    "builtins._ouro_sweep_meeting.wait(timeout=5)\n"
                    "def register(api):\n"
                    f"    api.register_tool('ping', lambda ctx: '{label}', description='p', schema={{}})\n"
                ),
                permissions=["tool"],
            )
        # Appended last, in this order: the paused sweep has already snapshotted
        # the None entry it deletes once the gate returns.
        sys.path_importer_cache[keys[0]] = _SweepGate(before_entry)
        sys.path_importer_cache[keys[1]] = None
        sys.path_importer_cache[keys[2]] = _SweepGate(after_entry)
        results: dict = {}

        def load(label):
            loaded, _repo, drive_root = prepared[label]
            try:
                results[label] = extension_loader.load_extension(
                    loaded, lambda: {}, drive_root=drive_root, _force_in_process=True)
            except BaseException as exc:
                results[label] = exc

        first = threading.Thread(target=load, args=("sweep_one",), name="sweep-first")
        second = threading.Thread(target=load, args=("sweep_two",), name="sweep-second")
        first.start()
        threads.append(first)
        assert first_paused.wait(GATE), "the first load never entered its importer-cache sweep"
        second.start()
        threads.append(second)
        for thread in (first, second):
            thread.join(timeout=GATE * 2)
        assert not first.is_alive() and not second.is_alive(), "extension loads did not settle"
        assert waited == [True], "the second load neither swept nor waited on the importer-cache lock"
        assert results == {"sweep_one": None, "sweep_two": None}, results
        for label in prepared:
            extension_loader.unload_extension(label)
    finally:
        # A failed assertion can leave a load parked on a gate: open both and join
        # every started load before its hooks, and then the patched lock, go away.
        second_moved.set()
        builtins._ouro_sweep_meeting.abort()
        for thread in threads:
            thread.join(timeout=GATE)
        for key in keys:
            sys.path_importer_cache.pop(key, None)
        del builtins._ouro_sweep_meeting


def test_every_runtime_importer_cache_sweep_takes_the_shared_lock():
    """The race is a class: a direct sweep beside the helper would reopen it."""
    import ast
    import inspect
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[1]
    sweeps: dict = {}
    for path in [*root.glob("ouroboros/**/*.py"), *root.glob("supervisor/**/*.py"),
                 root / "server.py", root / "launcher.py"]:
        text = path.read_text(encoding="utf-8")
        if "invalidate_caches" not in text:
            continue
        lines = [
            node.lineno for node in ast.walk(ast.parse(text))
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            and node.func.attr == "invalidate_caches"
            and isinstance(node.func.value, ast.Name) and node.func.value.id == "importlib"
        ]
        if lines:
            sweeps[path.relative_to(root).as_posix()] = lines
    body, start = inspect.getsourcelines(deps.invalidate_import_caches)
    assert list(sweeps) == ["ouroboros/extension_isolated_deps.py"], sweeps
    assert all(start <= line < start + len(body) for line in sweeps[list(sweeps)[0]]), sweeps


def test_a_deps_bearing_load_excludes_a_concurrent_no_deps_load(tmp_path):
    """Real isolated-deps injection is a writer: no no-deps import may overlap it."""
    inside = threading.Event()
    release = threading.Event()
    import builtins

    builtins._ouro_1195_gate = (inside, release)
    threads = []
    try:
        (tmp_path / "owner").mkdir(parents=True, exist_ok=True)
        (tmp_path / "neighbour").mkdir(parents=True, exist_ok=True)
        owner, repo_root, drive_root = _prepare_extension(
            tmp_path / "owner",
            "writer_owner",
            (
                "import builtins\n"
                "import barrier_pkg\n"
                "_inside, _release = builtins._ouro_1195_gate\n"
                "_inside.set()\n"
                "_release.wait(5)\n"
                "def register(api):\n"
                "    api.register_tool('v', lambda ctx: barrier_pkg.VALUE, description='v', schema={})\n"
            ),
            permissions=["tool"],
            extra_frontmatter="dependencies:\n  - barrier_pkg\n",
        )
        site_dir = (
            owner.skill_dir / ".ouroboros_env" / "python" / "lib"
            / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages"
        )
        package = site_dir / "barrier_pkg"
        package.mkdir(parents=True)
        (package / "__init__.py").write_text("VALUE = 'owner'\n", encoding="utf-8")
        _mark_isolated_deps_installed(drive_root, owner)

        neighbour, _repo, neighbour_root = _prepare_extension(
            tmp_path / "neighbour",
            "reader_neighbour",
            (
                "def register(api):\n"
                "    api.register_tool('v', lambda ctx: 'neighbour', description='v', schema={})\n"
            ),
            permissions=["tool"],
        )
        results: dict = {}

        def load_owner():
            try:
                results["owner"] = extension_loader.load_extension(
                    owner, lambda: {}, drive_root=drive_root, _force_in_process=True)
            except BaseException as exc:
                results["owner"] = exc

        def load_neighbour():
            try:
                results["neighbour"] = extension_loader.load_extension(
                    neighbour, lambda: {}, drive_root=neighbour_root, _force_in_process=True)
            except BaseException as exc:
                results["neighbour"] = exc

        writer_thread = threading.Thread(target=load_owner)
        writer_thread.start()
        threads.append(writer_thread)
        assert inside.wait(GATE), "the deps-bearing import never started"
        reader_thread = threading.Thread(target=load_neighbour)
        reader_thread.start()
        threads.append(reader_thread)
        reader_thread.join(timeout=0.3)
        assert reader_thread.is_alive(), "a no-deps load entered during dependency injection"
        release.set()
        writer_thread.join(timeout=GATE)
        reader_thread.join(timeout=GATE)
        if any(thread.is_alive() for thread in threads):
            import faulthandler
            faulthandler.dump_traceback(all_threads=True)
            pytest.fail("extension loads did not settle; all-thread stacks above")
        assert results == {"owner": None, "neighbour": None}, results
        extension_loader.unload_extension("writer_owner")
        extension_loader.unload_extension("reader_neighbour")
    finally:
        release.set()
        for thread in threads:
            thread.join(timeout=GATE)
        del builtins._ouro_1195_gate


# One skill, one handler at a time: the per-skill sequencing a consumer such as
# the bundled Telegram card renderer relies on (read the message id, await the
# send, record it). Cross-skill overlap is the F3 gain; same-skill ordering is
# the contract the old exclusive lock gave every skill author for free.


def test_same_skill_sync_readers_run_one_at_a_time(tmp_path):
    skill_dir = tmp_path / "telegram"
    first_in, first_out, second_in = threading.Event(), threading.Event(), threading.Event()
    sink: list = []
    first = threading.Thread(target=_reader(skill_dir, first_in, first_out, sink))
    second_release = threading.Event()
    second = threading.Thread(target=_reader(skill_dir, second_in, second_release, sink))
    first.start()
    assert first_in.wait(GATE)
    second.start()
    assert not second_in.wait(0.3), "a second handler of the SAME skill entered while the first was live"
    assert not deps._execution_lock.try_enter(False, deps._scope_key(skill_dir))
    assert deps._execution_lock.try_enter(False, deps._scope_key(tmp_path / "other"))
    deps._execution_lock.release(writer=False, key=deps._scope_key(tmp_path / "other"))
    first_out.set()
    assert second_in.wait(GATE), "the second handler never entered after the first left"
    second_release.set()
    first.join(GATE)
    second.join(GATE)
    assert sink == ["reader", "reader"], sink


def test_same_skill_async_handlers_run_one_at_a_time(tmp_path):
    async def main():
        skill_dir = tmp_path / "telegram"
        first_in = asyncio.Event()
        first_out = asyncio.Event()
        order: list = []

        async def handler(label, gate_in, gate_out):
            async with deps.async_isolated_site_dirs_scope(skill_dir, enabled=False):
                order.append(f"{label}:in")
                if gate_in is not None:
                    gate_in.set()
                if gate_out is not None:
                    await gate_out.wait()
                order.append(f"{label}:out")

        one = asyncio.create_task(handler("one", first_in, first_out))
        await first_in.wait()
        two = asyncio.create_task(handler("two", None, None))
        await asyncio.sleep(0.2)
        assert order == ["one:in"], order  # two is waiting, not inside
        first_out.set()
        await asyncio.wait_for(asyncio.gather(one, two), GATE)
        assert order == ["one:in", "one:out", "two:in", "two:out"], order

    asyncio.run(main())


def test_registered_handlers_of_one_skill_serialize_and_of_two_skills_overlap(tmp_path):
    """Through the real PluginAPIImpl wrapper: same skill sequential, other skill concurrent."""
    from ouroboros.extension_plugin_api import PluginAPIImpl, _PluginAPIConfig

    def api_for(label):
        state_dir = tmp_path / "state" / label
        state_dir.mkdir(parents=True, exist_ok=True)
        return PluginAPIImpl(_PluginAPIConfig(
            skill_name=label, permissions=[], env_allowlist=[], state_dir=state_dir,
            settings_reader=lambda: {}, skill_dir=tmp_path / label,
        ))

    api_a = api_for("skill_a")
    api_b = api_for("skill_b")

    a_inside = threading.Event()
    a_release = threading.Event()
    seen: list = []

    def slow_a(*_args):
        seen.append("a:in")
        a_inside.set()
        a_release.wait(GATE)
        seen.append("a:out")

    def quick(label):
        def run(*_args):
            seen.append(f"{label}:in")
            seen.append(f"{label}:out")
        return run

    wrapped_slow_a = api_a._wrap_runtime_handler(slow_a)
    wrapped_quick_a = api_a._wrap_runtime_handler(quick("a2"))
    wrapped_quick_b = api_b._wrap_runtime_handler(quick("b"))

    slow = threading.Thread(target=wrapped_slow_a)
    slow.start()
    assert a_inside.wait(GATE)
    other = threading.Thread(target=wrapped_quick_b)
    other.start()
    other.join(GATE)
    assert not other.is_alive(), "another skill's handler was blocked by skill_a's live handler"
    assert seen[:3] == ["a:in", "b:in", "b:out"], seen
    same = threading.Thread(target=wrapped_quick_a)
    same.start()
    assert not same.join(0.3) and same.is_alive(), "skill_a's second handler overlapped its first"
    a_release.set()
    same.join(GATE)
    slow.join(GATE)
    assert seen == ["a:in", "b:in", "b:out", "a:out", "a2:in", "a2:out"], seen
