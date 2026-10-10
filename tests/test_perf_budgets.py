"""Cross-endpoint performance-budget tests (perf/lifecycle sprint SYNTH).

Every budget here is a DETERMINISTIC COUNTER on a real substrate seam —
never a timing/ms assertion. The per-phase suites pin each mechanism in
isolation (rows-memo equivalence, window-doubling reader, SSE chain deltas,
materialize_artifacts call sites); these tests pin the FULL-ENDPOINT budgets
that only exist after the P1+P2 merges:

* /api/state: zero whole-store reads, one usage projection per request handed
  into the evolution snapshot (no recompute inside it).
* /api/chat/history: a bounded byte-tail read of a large progress.jsonl and
  zero artifact materialization / disposition lookups in the annotation step.
* /api/logs/{name}: the same bounded-read budget on a large events log.
* SSE follow (_TaskEventFollower): a tick after the initial replay reads only
  the appended bytes (offsets advance; nothing re-reads from offset 0) and
  performs zero artifact work.
"""

from __future__ import annotations

import asyncio
import json
import logging
import types

from starlette.requests import Request

from ouroboros import usage_accounting as ua
from ouroboros import usage_store


def test_retained_execution_drive_tripwire_counts_both_roots(tmp_path, monkeypatch):
    from ouroboros import context_budget
    from ouroboros.agent_startup_checks import hot_store_growth_notes
    from ouroboros.headless import HEADLESS_TASKS_DIR, TASK_DRIVES_DIR
    from supervisor.state import ISOLATED_BENCHMARK_SENTINEL

    monkeypatch.setattr(context_budget, "RETAINED_EXECUTION_DRIVES_WARN_COUNT", 2)
    env = types.SimpleNamespace(
        drive_root=tmp_path,
        drive_path=lambda rel: tmp_path / rel,
    )
    headless = tmp_path / HEADLESS_TASKS_DIR
    task_drives = tmp_path / TASK_DRIVES_DIR
    (headless / "headless-1").mkdir(parents=True)
    (task_drives / "drive-1").mkdir(parents=True)

    assert hot_store_growth_notes(env) == []

    (task_drives / "drive-2").mkdir()
    notes = hot_store_growth_notes(env)
    assert len(notes) == 1
    assert "retained execution drives" in notes[0]
    assert "total 3 (threshold 2)" in notes[0]

    (tmp_path / ISOLATED_BENCHMARK_SENTINEL).write_text("isolated\n", encoding="utf-8")
    assert hot_store_growth_notes(env) == []


def test_observability_write_failure_warns_once_and_stays_nonfatal(tmp_path, caplog):
    from ouroboros.llm_observability import persist_observed_call

    def fail_write(*args, **kwargs):
        raise OSError("test write failure")

    with caplog.at_level(logging.WARNING, logger="ouroboros.llm_observability"):
        result = persist_observed_call(
            tmp_path,
            payload={"model": "claudexor/test"},
            writer=fail_write,
            task_id="task-1",
        )

    assert result == {}
    records = [
        record for record in caplog.records
        if record.name == "ouroboros.llm_observability"
    ]
    assert len(records) == 1
    assert records[0].levelno == logging.WARNING
    assert records[0].getMessage() == "Failed to persist LLM observability payload"


def _seeded_accounting_root(tmp_path, monkeypatch):
    """Temp drive root with a settled + reserved attempt in the usage ledger."""
    root = tmp_path / "data"
    (root / "state").mkdir(parents=True)
    (root / "logs").mkdir(parents=True)
    (root / "state" / "state.json").write_text(
        json.dumps({"spent_usd": 0.0, "spent_calls": 0}), encoding="utf-8"
    )
    monkeypatch.setenv("OUROBOROS_DATA_DIR", str(root))
    monkeypatch.setenv("OUROBOROS_SETTINGS_PATH", str(root / "settings.json"))
    monkeypatch.setenv("TOTAL_BUDGET", "7.5")
    usage_store.migrate_from_journal(root)  # the lifecycle job, before any request
    settled = ua.reserve_attempt(ua.AttemptRequest(
        model="openai/gpt-5.2", provider="openai", reservation_usd=0.5,
        global_limit_usd=7.5, drive_root=root, task_id="settled",
        root_task_id="root-1", category="task", source="test.perf",
    ))
    ua.mark_dispatched(settled)
    ua.settle_attempt(
        settled, {"prompt_tokens": 100, "completion_tokens": 20},
        cost_usd=0.25, cost_final=True,
    )
    ua.reserve_attempt(ua.AttemptRequest(
        model="openai/gpt-5.2", provider="openai", reservation_usd=1.0,
        global_limit_usd=7.5, drive_root=root, task_id="reserved",
        root_task_id="root-1", category="task", source="test.perf",
    ))
    usage_store.forget(root)  # the first endpoint call opens the store cold
    return root


def _state_request(root):
    return Request({
        "type": "http", "method": "GET", "path": "/api/state", "headers": [],
        "query_string": b"", "scheme": "http", "server": ("test", 80),
        "client": ("test", 1),
        "app": types.SimpleNamespace(
            state=types.SimpleNamespace(drive_root=root, app_start=0.0),
        ),
    })


def _install_artifact_counters(monkeypatch):
    """Count every artifact materialization / disposition-hash seam call."""
    from ouroboros import artifacts, task_status

    counters = {"collect": 0, "copy": 0, "disposition": 0}
    real_collect = artifacts.collect_task_artifact_records
    real_copy = artifacts.copy_file_to_task_artifacts
    real_disposition = task_status._project_child_result_disposition

    def counted_collect(*args, **kwargs):
        counters["collect"] += 1
        return real_collect(*args, **kwargs)

    def counted_copy(*args, **kwargs):
        counters["copy"] += 1
        return real_copy(*args, **kwargs)

    def counted_disposition(*args, **kwargs):
        counters["disposition"] += 1
        return real_disposition(*args, **kwargs)

    monkeypatch.setattr(artifacts, "collect_task_artifact_records", counted_collect)
    monkeypatch.setattr(artifacts, "copy_file_to_task_artifacts", counted_copy)
    monkeypatch.setattr(task_status, "_project_child_result_disposition", counted_disposition)
    return counters


def _install_tail_read_counter(monkeypatch):
    """Record every (path, tail_bytes) the bounded gateway reader requests."""
    from ouroboros.gateway import _helpers

    calls = []
    real_iter = _helpers.iter_jsonl_objects

    def counted_iter(path, *args, **kwargs):
        calls.append((str(path), kwargs.get("tail_bytes")))
        return real_iter(path, *args, **kwargs)

    monkeypatch.setattr(_helpers, "iter_jsonl_objects", counted_iter)
    return calls


def test_api_state_warm_path_replays_ledger_zero_times_and_projects_once(
    tmp_path, monkeypatch,
):
    """Full-endpoint /api/state budget: cold or warm, a call reads ZERO whole
    stores (summary rows only), and its single usage projection is the one the
    (real) evolution snapshot consumes — no second projection anywhere.

    The kwarg-level de-triplication seam is pinned in
    test_gateway_usage_accounting.py; this pins the endpoint-wide budget."""
    from ouroboros.gateway.state import api_state
    from supervisor import queue, state, workers

    root = _seeded_accounting_root(tmp_path, monkeypatch)
    monkeypatch.setattr(state, "TOTAL_BUDGET_LIMIT", 7.5)
    monkeypatch.setattr(state, "DRIVE_ROOT", str(root))
    monkeypatch.setattr(state, "load_state", lambda: {"current_branch": "ouroboros"})
    monkeypatch.setattr(workers, "WORKERS", {})
    monkeypatch.setattr(workers, "PENDING", [])
    monkeypatch.setattr(workers, "RUNNING", {})
    # The REAL get_evolution_status_snapshot runs (that is the point); only its
    # own state/campaign inputs are stubbed to the temp root's trivial values.
    monkeypatch.setattr(queue, "PENDING", [])
    monkeypatch.setattr(queue, "RUNNING", {})
    monkeypatch.setattr(queue, "load_state", lambda: {"current_branch": "ouroboros"})
    monkeypatch.setattr(queue, "_read_evolution_campaign", lambda: {})

    counters = _install_ledger_read_counters(monkeypatch)

    cold = asyncio.run(api_state(_state_request(root)))
    assert cold.status_code == 200
    assert counters["migrations"] == counters["full_reads"] == []

    counters["projections"].clear()
    warm = asyncio.run(api_state(_state_request(root)))
    payload = json.loads(warm.body)

    assert warm.status_code == 200
    assert counters["migrations"] == counters["full_reads"] == []
    # One projection for the whole request; budget_remaining consumed it inside
    # the evolution snapshot instead of recomputing (a second entry here would
    # be the de-triplication regression this test exists to catch).
    assert len(counters["projections"]) == 1
    assert "budget_reserve_usd" in payload["evolution_state"]  # real snapshot ran
    assert payload["spent_usd"] == 0.25  # known (settled) spend; the $1.00 hold rides in accounting
    assert payload["accounting"]["accounted_usd"] == 1.25
    assert payload["accounting"]["authority"] == "physical_attempt_ledger"


def test_chat_history_reads_bounded_progress_tail_with_zero_artifact_work(
    tmp_path, monkeypatch,
):
    """On a progress.jsonl much larger than the 512KB start window (quota
    satisfiable from the tail), /api/chat/history must read a strict byte tail
    — never the whole file — and its terminal-truth annotation must perform
    zero artifact collection/copies and zero disposition-hash lookups."""
    from ouroboros.gateway.history import make_chat_history_endpoint
    from ouroboros import jsonl_tail
    from contextlib import contextmanager
    from ouroboros.task_results import write_task_result

    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "chat.jsonl").write_text(
        json.dumps({"ts": "2026-08-08T00:00:00Z", "direction": "in", "text": "hello"}) + "\n",
        encoding="utf-8",
    )
    with (logs / "progress.jsonl").open("w", encoding="utf-8") as handle:
        for i in range(16000):
            hours, rem = divmod(i, 3600)
            handle.write(json.dumps({
                "ts": f"2026-08-08T{hours:02d}:{rem // 60:02d}:{rem % 60:02d}Z",
                "content": f"telemetry-{i}", "task_id": "t1", "pad": "x" * 120,
            }) + "\n")
    progress_size = (logs / "progress.jsonl").stat().st_size
    assert progress_size > 3_000_000  # much larger than the 512KB start window
    write_task_result(tmp_path, "t1", "completed", result="done", ts="2026-08-08T05:00:00Z")

    # The pager feeds the shared parser an already-bounded borrowed buffer.
    # Count actual source bytes, rather than the old parser's tail_bytes hint.
    reads = []
    chain_handles = jsonl_tail.jsonl_chain_handles

    class CountedHandle:
        def __init__(self, path, handle):
            self.path, self.handle = path, handle

        def __getattr__(self, name):
            return getattr(self.handle, name)

        def read(self, size=-1):
            start = self.handle.tell()
            data = self.handle.read(size)
            reads.append((self.path, start, len(data)))
            return data

    @contextmanager
    def counted_handles(*args, **kwargs):
        with chain_handles(*args, **kwargs) as handles:
            yield [(path, CountedHandle(path, handle)) for path, handle in handles]

    monkeypatch.setattr(jsonl_tail, "jsonl_chain_handles", counted_handles)
    artifact_counters = _install_artifact_counters(monkeypatch)

    endpoint = make_chat_history_endpoint(tmp_path)
    response = asyncio.run(endpoint(types.SimpleNamespace(query_params={})))
    messages = json.loads(response.body)["messages"]

    progress_reads = [(start, size) for path, start, size in reads if path.name == "progress.jsonl"]
    assert progress_reads  # the bounded reader actually served the endpoint
    assert all(start >= progress_size - 512 * 1024 - 1 and size > 0 for start, size in progress_reads)
    # One alignment lookahead and one parsed window, plus newline probes.
    assert sum(size for _start, size in progress_reads) <= 2 * 512 * 1024 + 2 < progress_size
    progress_rows = [row for row in messages if row.get("is_progress")]
    assert len(progress_rows) == 60
    assert progress_rows[-1]["text"] == "telemetry-15999"
    # Annotation ran on the emitted window (terminal truth landed on rows)...
    annotated = [m for m in messages if m.get("task_terminal_status") == "completed"]
    assert annotated
    # ...as a status/cost projection only: zero artifact materialization and
    # zero disposition-hash lookups on the GET path.
    assert artifact_counters == {"collect": 0, "copy": 0, "disposition": 0}


def _install_ledger_read_counters(monkeypatch):
    """Count journal imports, whole-store reads and both live projections."""
    counters: dict = {"migrations": [], "full_reads": [], "projections": [], "breakdowns": []}
    real_migrate = usage_store.migrate_from_journal
    real_full_read = usage_store.read_usage_records
    real_attempts = usage_store.Txn.attempts
    real_projection = ua.usage_projection
    real_breakdown = ua.usage_breakdown

    def counted_migration(target_root):
        counters["migrations"].append(str(target_root))
        return real_migrate(target_root)

    def counted_full_read(target_root):
        counters["full_reads"].append(str(target_root))
        return real_full_read(target_root)

    def counted_attempts(txn, where="1", params=()):
        if str(where).strip() == "1":  # an unaddressed scan of every attempt
            counters["full_reads"].append(str(txn.root))
        return real_attempts(txn, where, params)

    def counted_projection(*args, **kwargs):
        counters["projections"].append(kwargs)
        return real_projection(*args, **kwargs)

    def counted_breakdown(*args, **kwargs):
        counters["breakdowns"].append(kwargs)
        return real_breakdown(*args, **kwargs)

    monkeypatch.setattr(usage_store, "migrate_from_journal", counted_migration)
    monkeypatch.setattr(usage_store, "read_usage_records", counted_full_read)
    monkeypatch.setattr(usage_store.Txn, "attempts", counted_attempts)
    monkeypatch.setattr(ua, "usage_projection", counted_projection)
    monkeypatch.setattr(ua, "usage_breakdown", counted_breakdown)
    return counters


def test_live_root_surfaces_replay_the_ledger_zero_times_when_warm(
    tmp_path, monkeypatch,
):
    """The two surfaces a project chat actually opens on a LIVE root: chat
    history (which projects a non-final root's subtree cost for the newest
    progress row) and GET /api/tasks/{id} (which derives cost_breakdown). The
    sibling budgets above seed a COMPLETED task, so neither live branch runs
    there. Each surface asks its projection exactly ONCE and the store's
    summary rows answer it: ZERO whole-store reads, which is what made an
    oversized ledger cost seconds per open."""
    from ouroboros.gateway.history import make_chat_history_endpoint
    from ouroboros.gateway.tasks import _task_get_response
    from ouroboros.task_results import write_task_result

    root = _seeded_accounting_root(tmp_path, monkeypatch)
    (root / "logs" / "chat.jsonl").write_text(
        json.dumps({"ts": "2026-08-08T00:00:00Z", "direction": "in", "text": "hello"}) + "\n",
        encoding="utf-8",
    )
    with (root / "logs" / "progress.jsonl").open("w", encoding="utf-8") as handle:
        for i in range(5):
            handle.write(json.dumps({
                "ts": f"2026-08-08T00:00:{i:02d}Z", "content": f"step-{i}", "task_id": "root-1",
            }) + "\n")
    # NON-final: `live` in history.py is `status not in FINAL_STATUSES`.
    write_task_result(root, "root-1", "running", result="working", ts="2026-08-08T00:00:00Z")

    history = make_chat_history_endpoint(root)
    request = types.SimpleNamespace(
        path_params={"task_id": "root-1"},
        query_params={},
        app=types.SimpleNamespace(state=types.SimpleNamespace(drive_root=root)),
    )
    counters = _install_ledger_read_counters(monkeypatch)
    asyncio.run(history(request))  # cold: opens the store
    assert counters["migrations"] == counters["full_reads"] == []
    counters["projections"].clear()

    messages = json.loads(asyncio.run(history(request)).body)["messages"]
    live_rows = [m for m in messages if m.get("cost_with_children_partial")]

    assert live_rows  # the live-root branch really ran (not a vacuous budget)
    assert live_rows[-1]["cost_accounting_status"] == "available"
    assert counters["migrations"] == counters["full_reads"] == []
    assert len(counters["projections"]) == 1  # one live-root projection, cached
    assert counters["projections"][0]["root_task_id"] == "root-1"

    counters["projections"].clear()
    payload = json.loads(_task_get_response(request).body)

    assert payload["cost_breakdown"]["authority"] == "physical_attempt_ledger"
    assert counters["migrations"] == counters["full_reads"] == []
    assert len(counters["breakdowns"]) == 1  # one subtree breakdown, cached
    assert counters["breakdowns"][0]["root_task_id"] == "root-1"
    assert counters["projections"] == []  # the detail path reads no projection


def test_logs_tail_reads_bounded_events_tail(tmp_path, monkeypatch):
    """/api/logs/{name} with a satisfiable limit reads a byte tail of a large
    events log, never the whole file."""
    from ouroboros.gateway.logs import api_logs_tail

    logs = tmp_path / "logs"
    logs.mkdir()
    with (logs / "events.jsonl").open("w", encoding="utf-8") as handle:
        for i in range(12000):
            hours, rem = divmod(i, 3600)
            handle.write(json.dumps({
                "ts": f"2026-08-08T{hours:02d}:{rem // 60:02d}:{rem % 60:02d}Z",
                "type": "llm_round", "content": f"event-{i}", "pad": "x" * 120,
            }) + "\n")
    events_size = (logs / "events.jsonl").stat().st_size
    assert events_size > 2_000_000

    tail_calls = _install_tail_read_counter(monkeypatch)

    request = types.SimpleNamespace(
        path_params={"name": "events"},
        query_params={"limit": "100"},
        app=types.SimpleNamespace(state=types.SimpleNamespace(drive_root=tmp_path)),
    )
    response = asyncio.run(api_logs_tail(request))
    entries = json.loads(response.body)["entries"]

    assert len(entries) == 100
    assert entries[-1]["content"] == "event-11999"  # the newest suffix
    events_reads = [c for c in tail_calls if c[0].endswith("events.jsonl")]
    assert events_reads
    assert all(tail is not None for _path, tail in events_reads)  # no full read
    assert sum(tail for _path, tail in events_reads) < events_size


def test_sse_follow_tick_reads_only_appended_bytes_and_zero_artifact_work(
    tmp_path, monkeypatch,
):
    """After the initial SSE replay, a follow tick with N appended rows reads
    exactly the appended bytes: the progress offset resumes where the replay
    stopped (offset > 0), no log is re-read from offset 0, and the tick does
    zero artifact work (the terminal emission's single materializing read
    lives in the stream's emit loop, not in the follower)."""
    from ouroboros.gateway import tasks as gateway_tasks
    from ouroboros.task_results import write_task_result

    data = tmp_path / "data"
    (data / "logs").mkdir(parents=True)
    (data / "state").mkdir(parents=True)
    # newline="\n": Windows text-mode would translate \n -> \r\n on disk,
    # desynchronizing the byte-offset math below from len(line.encode()).
    with (data / "logs" / "progress.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for i in range(50):
            handle.write(json.dumps({
                "ts": f"2026-08-08T00:00:{i:02d}Z", "content": f"step-{i}", "task_id": "t1",
            }) + "\n")
    write_task_result(data, "t1", "running", result="working", ts="2026-08-08T00:00:00Z")
    (data / "state" / "queue_snapshot.json").write_text(
        '{"pending": [], "running": []}', encoding="utf-8"
    )

    follower = gateway_tasks._TaskEventFollower(data, "t1")
    replay = follower.full_merge()
    assert len(replay) >= 50  # initial replay consumed the seeded history

    reads: list = []
    real_read = gateway_tasks._read_live_jsonl_entries

    def counted_read(path, offset):
        entries, new_offset, ino = real_read(path, offset)
        reads.append((str(path), offset, new_offset))
        return entries, new_offset, ino

    monkeypatch.setattr(gateway_tasks, "_read_live_jsonl_entries", counted_read)
    artifact_counters = _install_artifact_counters(monkeypatch)

    appended_bytes = 0
    with (data / "logs" / "progress.jsonl").open("a", encoding="utf-8", newline="\n") as handle:
        for i in range(3):
            line = json.dumps({
                "ts": f"2026-08-08T00:01:{i:02d}Z", "content": f"late-{i}", "task_id": "t1",
            }) + "\n"
            handle.write(line)
            appended_bytes += len(line.encode("utf-8"))

    new_rows, advanced = follower.poll()
    follower.refresh_result()  # the endpoint's post-advance tick refresh

    assert advanced
    assert [row["data"]["content"] for row in new_rows] == ["late-0", "late-1", "late-2"]
    progress_reads = [r for r in reads if r[0].endswith("progress.jsonl")]
    assert len(progress_reads) == 1
    _path, start_offset, end_offset = progress_reads[0]
    assert start_offset > 0  # resumed from the replay's offset, not a re-read
    assert end_offset - start_offset == appended_bytes  # ONLY the appended bytes
    assert all(offset > 0 for _p, offset, _e in reads)  # nothing re-read from 0
    assert artifact_counters == {"collect": 0, "copy": 0, "disposition": 0}


def test_the_chronicle_journal_is_an_enrolled_hot_store_that_warns_only_past_its_threshold(tmp_path, monkeypatch):
    """The memory journal is read on every task context, so it
    is enrolled in the hot-store tripwire (DEVELOPMENT 03 projection-over-replay rule): quiet
    at or below its threshold, one WARNING past it."""
    from ouroboros import context_budget
    from ouroboros.agent_startup_checks import _hot_store_thresholds, hot_store_growth_notes

    assert ("memory/chronicle/records.jsonl", context_budget.CHRONICLE_JOURNAL_WARN_BYTES) in {
        (relative, threshold) for relative, threshold, _remediation in _hot_store_thresholds()}
    monkeypatch.setattr(context_budget, "CHRONICLE_JOURNAL_WARN_BYTES", 10)
    env = types.SimpleNamespace(drive_root=tmp_path, drive_path=lambda rel: tmp_path / rel)
    journal = tmp_path / "memory" / "chronicle" / "records.jsonl"
    journal.parent.mkdir(parents=True)
    journal.write_bytes(b"x" * 10)
    assert hot_store_growth_notes(env) == []
    journal.write_bytes(b"x" * 11)
    [note] = hot_store_growth_notes(env)
    assert "memory/chronicle/records.jsonl" in note and "never delete records" in note


def test_the_review_ledger_index_chain_is_an_enrolled_hot_store(tmp_path, monkeypatch):
    """The ledger's hot index rotates itself; the rotated segments are replayed only by readers
    that walk the chain, so the chain's total size is a tripwire like the chat archive chain."""
    from ouroboros import context_budget
    from ouroboros.agent_startup_checks import hot_store_growth_notes

    monkeypatch.setattr(context_budget, "REVIEW_LEDGER_INDEX_WARN_BYTES", 10)
    env = types.SimpleNamespace(drive_root=tmp_path, drive_path=lambda rel: tmp_path / rel)
    ledger = tmp_path / "state" / "review_ledger"
    ledger.mkdir(parents=True)
    (ledger / "index.jsonl").write_bytes(b"x" * 10)
    (ledger / "rl-not-an-index.json").write_bytes(b"x" * 100)
    assert hot_store_growth_notes(env) == []
    (ledger / "index.20261007T000000.jsonl").write_bytes(b"x")
    [note] = hot_store_growth_notes(env)
    assert "state/review_ledger/index*.jsonl" in note and "never delete the newest" in note
