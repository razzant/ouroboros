"""The real worker reports entry before the potentially slow extension load."""

import json
import os

import pytest

from tests.test_terminal_file_boundary import worker as worker


@pytest.mark.parametrize("progress_write_fails", [False, True])
def test_entry_progress_precedes_extensions_and_logging_failure_does_not_block_ready(
    worker, monkeypatch, progress_write_fails,
):
    import ouroboros.extension_loader as extensions
    import ouroboros.utils as utils

    worker.task["type"] = "shutdown"
    events = worker.root / "logs" / "events.jsonl"
    real_append = utils.append_jsonl
    attempts, loads = [], []

    def append(path, row, **kwargs):
        if row.get("type") == "worker_starting":
            attempts.append(row)
            if progress_write_fails:
                raise OSError("entry log unavailable")
        return real_append(path, row, **kwargs)

    def load(*_args, **_kwargs):
        loads.append(True)
        assert len(attempts) == 1 and attempts[0]["pid"] == os.getpid()
        assert attempts[0]["worker_id"] == 0 and attempts[0]["phase"] == "entry"
        rows = events.read_text(encoding="utf-8") if events.exists() else ""
        assert '"worker_ready"' not in rows
        assert ('"worker_starting"' in rows) is not progress_write_fails

    monkeypatch.setattr(utils, "append_jsonl", append)
    monkeypatch.setattr(extensions, "reload_all", load)
    worker.run()

    rows = [json.loads(line) for line in events.read_text(encoding="utf-8").splitlines()]
    assert loads == [True] and worker.crashes == [] and worker.calls == []
    assert [row["type"] for row in rows] == (
        ["worker_ready"] if progress_write_fails else ["worker_starting", "worker_ready"]
    )
    assert rows[-1]["pid"] == os.getpid() and rows[-1]["git_sha"] == "test-sha"


def test_ready_row_carries_the_birth_receipt_and_its_window_ends_there(worker, monkeypatch):
    """``worker_ready`` names the birth's steps (wall and thread CPU) and the work they counted (K5's catalog and
    hash counts come from ``reload_all``); the window ends with readiness, so no task's work joins it."""
    import ouroboros.extension_loader as extensions
    from ouroboros import context_timing

    worker.task["type"] = "shutdown"
    monkeypatch.setattr(extensions, "reload_all", lambda *_a, **_kw: context_timing.counted("catalog_reused", 6))
    worker.run()

    ready = json.loads((worker.root / "logs" / "events.jsonl").read_text(encoding="utf-8").splitlines()[-1])
    assert (ready["type"], ready["work"], worker.crashes) == ("worker_ready", {"catalog_reused": 6}, [])
    for spans in (ready["sections_ms"], ready["sections_cpu_ms"]):
        assert set(spans) == {"import", "reload_all", "make_agent", "total"}
    assert context_timing.section_spans() == {}


@pytest.mark.parametrize("warm,fails", [(False, False), (True, False), (True, True)])
def test_conversation_source_preparation_is_before_ready_and_optional(worker, monkeypatch, warm, fails):
    from functools import partial
    from types import SimpleNamespace
    import ouroboros.agent as agent
    from supervisor import worker_process as wp

    worker.task["type"] = "shutdown"
    seen = []
    env = SimpleNamespace(repo_dir=worker.root, drive_root=worker.root)
    monkeypatch.setattr(agent, "make_agent", lambda **_kw: SimpleNamespace(env=env))
    def prepare(current):
        assert current is env
        assert '"worker_ready"' not in (worker.root / "logs" / "events.jsonl").read_text()
        seen.append("warm")
        if fails:
            raise OSError("optional source unavailable")
        return {"architecture": "warm", "story": "inactive"}
    monkeypatch.setattr(wp, "_warm_context_sources", prepare)
    monkeypatch.setattr(wp, "worker_main", partial(wp.worker_main, warm_context_sources=warm))
    worker.run()
    ready = json.loads((worker.root / "logs" / "events.jsonl").read_text().splitlines()[-1])
    assert ready["type"] == "worker_ready" and not worker.crashes
    assert seen == (["warm"] if warm else [])
    assert ("context_sources" in ready["sections_ms"]) == warm
    if warm:
        assert ready["context_sources"] == ({"preparation": "OSError"} if fails else {"architecture": "warm", "story": "inactive"})


def test_source_warming_does_not_activate_memory_and_books_fail_independently(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from ouroboros import reference_books as books
    from ouroboros.chronicle_store import ChronicleStore
    from supervisor.worker_process import _warm_context_sources

    seen = []
    def load(_root, book, _read):
        seen.append(book)
        if book == "architecture":
            raise OSError("missing chapter")
    monkeypatch.setattr(books, "load_reference_book", load)
    monkeypatch.setattr(ChronicleStore, "ensure_activated", lambda *_a, **_kw: pytest.fail("warm activated memory"))
    env = SimpleNamespace(repo_dir=tmp_path, drive_root=tmp_path, repo_path=lambda p: tmp_path / p)
    assert _warm_context_sources(env) == {"architecture": "OSError", "development": "warm", "story": "inactive"}
    assert seen == ["architecture", "development"] and not (tmp_path / "memory").exists()


def _source_consumer_entry(*args, **kwargs):
    """Real spawned worker and context capture, with no extension or model execution."""
    import hashlib
    import pathlib
    import time
    from types import SimpleNamespace
    from ouroboros import agent, context, extension_loader, memory_view, reference_books
    from ouroboros.memory import Memory
    from supervisor import worker_process

    counts = {"parse": 0, "story": 0}
    parse, story = reference_books.parse_markdown_source, memory_view._capture_story
    def parsed(*a, **kw):
        counts["parse"] += 1
        return parse(*a, **kw)
    def captured(*a, **kw):
        counts["story"] += 1
        return story(*a, **kw)
    reference_books._book_source.cache_clear()
    memory_view._STORIES.clear()
    reference_books.parse_markdown_source = parsed
    memory_view._capture_story = captured
    extension_loader.reload_all = lambda *_a, **_kw: None
    worker_process._adopt_published_extensions = lambda *_a: None
    class Consumer:
        def __init__(self, repo_dir, drive_root, **_kw):
            repo, root = pathlib.Path(repo_dir), pathlib.Path(drive_root)
            self.env = SimpleNamespace(repo_dir=repo, drive_root=root, repo_path=lambda p: repo / p,
                                       drive_path=lambda p: root / p)
        def handle_task(self, task):
            before, start, cpu = dict(counts), time.perf_counter(), time.thread_time()
            core = context._capture_context_core(self.env, Memory(drive_root=self.env.drive_root,
                                                  repo_dir=self.env.repo_dir), task, None, None)
            raw = json.dumps([core.architecture_md, core.development_md, core.memory_view_json], ensure_ascii=False).encode()
            args[2].put({"type": "source_consumer", "wall_ms": 1000 * (time.perf_counter() - start),
                         "cpu_ms": 1000 * (time.thread_time() - cpu), "before": before, "after": dict(counts),
                         "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()})
            return []
    agent.make_agent = lambda **kw: Consumer(**kw)
    worker_process.worker_main(*args, **kwargs)


def test_native_spawn_reports_both_readiness_and_equal_context_consumer(tmp_path, monkeypatch):
    """Measure boot cost too: source prewarm can move work, not magically remove it."""
    import multiprocessing
    import pathlib
    import queue
    import time
    from ouroboros.reference_books import BOOK_ENTRYPOINTS, load_reference_book
    from supervisor import worker_process
    from tests import _memory_capture_shape as shape
    from tests._shared import stop_socket_sharer
    from tests.test_book_context_capture import _capture

    env, _core, _task = _capture(tmp_path)
    repo = pathlib.Path(__file__).resolve().parents[1]
    for book in BOOK_ENTRYPOINTS:
        source = load_reference_book(repo, book)
        for member in (source.entrypoint, *source.chapters):
            target = env.repo_path(member.source_path)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(member.raw)
    shape.terminal_links(env.drive_root, 8)
    monkeypatch.setattr(worker_process, "worker_main", _source_consumer_entry)
    ctx, reports = multiprocessing.get_context("spawn"), []
    task = {"id": "source-probe", "chat_id": 1, "type": "task", "text": "Read the full sources", "_is_direct_chat": True}
    for warm, immediate in ((False, False), (True, False), (False, True), (True, True)):
        events = env.drive_root / "logs/events.jsonl"
        offset = events.stat().st_size if events.exists() else 0
        incoming, outgoing = ctx.Queue(), ctx.Queue()
        started = time.perf_counter()
        if immediate:
            incoming.put(task)
        proc = worker_process.spawn_worker_process(ctx, len(reports), incoming, outgoing, env.repo_dir,
                                                   env.drive_root, warm_context_sources=warm)
        try:
            ready, deadline = None, time.monotonic() + 120
            while ready is None and time.monotonic() < deadline:
                assert proc.is_alive(), proc.exitcode
                if events.exists():
                    with events.open("rb") as stream:
                        stream.seek(offset)
                        rows = stream.read().splitlines()
                    for raw in rows:
                        try:
                            row = json.loads(raw)
                        except ValueError:  # append in progress
                            continue
                        if row.get("type") == "worker_ready" and row.get("pid") == proc.pid:
                            ready = row
                if ready is None:
                    time.sleep(.01)
            assert ready is not None
            ready_ms = 1000 * (time.perf_counter() - started)
            ingress = started if immediate else time.perf_counter()
            if not immediate:
                incoming.put(task)
            result = None
            while result is None and time.monotonic() < deadline:
                try:
                    row = outgoing.get(timeout=1)
                except queue.Empty:
                    assert proc.is_alive(), proc.exitcode
                    continue
                if isinstance(row, dict) and row.get("type") == "source_consumer":
                    result = row
            assert result is not None
            reports.append({"warm": warm, "immediate": immediate, "spawn_ready_ms": ready_ms,
                            "ingress_capture_ms": 1000 * (time.perf_counter() - ingress),
                            "birth": {key: ready.get(key) for key in ("sections_ms", "sections_cpu_ms", "context_sources")},
                            **result})
        finally:
            incoming.put(None)
            proc.join(15)
            if proc.is_alive():
                proc.kill()
                proc.join(5)
            worker_process.close_worker_stop_channel(proc)
            for channel in (incoming, outgoing):
                channel.close()
                channel.cancel_join_thread()
            stop_socket_sharer()
        assert proc.exitcode == 0
    assert len({(r["sha256"], r["bytes"]) for r in reports}) == 1
    assert reports[0]["after"]["parse"] > reports[0]["before"]["parse"]
    assert all(r["after"]["parse"] == r["before"]["parse"] and r["before"]["story"] == 1
               and r["after"]["story"] == 1 for r in reports if r["warm"])
    print(json.dumps({"native_spawn_source_profile": reports}, sort_keys=True))
