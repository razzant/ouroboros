"""Fingerprint-keyed render cache inside the ledger rows memo (perf sprint P1).

``usage_projection``/``usage_breakdown`` cache their FINISHED renders inside
``_LedgerRowsMemo.renders``, keyed by every input that shapes the output
(function, root/task filters, resolved limit, integrity bit, include_roots).
These tests pin the cache's safety properties: a warm hit does zero full
replays AND zero re-aggregation; append/rotation invalidate; a non-resumable
crash-tail is never cached; results are deep copies (nested-bucket mutation by
one caller cannot poison another's read); and a concurrent append between the
row read and the publish leaves the stale render uncached (clear-then-publish
race guard). Row-level memo equivalence stays pinned by
tests/test_usage_rows_memo.py; write paths by tests/test_usage_accounting.py.
"""
from __future__ import annotations

import asyncio
import json
import os
import pathlib
import types

import pytest
from starlette.requests import Request

from ouroboros import usage_accounting as ua


@pytest.fixture
def data_root(tmp_path, monkeypatch):
    root = tmp_path / "data"
    monkeypatch.setenv("OUROBOROS_DATA_DIR", str(root))
    monkeypatch.setenv("OUROBOROS_SETTINGS_PATH", str(root / "settings.json"))
    monkeypatch.setenv("TOTAL_BUDGET", "100")
    (root / "state").mkdir(parents=True)
    return root


def _request(data_root, **overrides):
    values = {
        "model": "openai/gpt-5.2",
        "provider": "openai",
        "reservation_usd": 0.05,
        "drive_root": data_root,
        "task_id": "child",
        "root_task_id": "root",
        "source": "test",
    }
    values.update(overrides)
    return ua.AttemptRequest(**values)


def _memo_key(root) -> str:
    return str(pathlib.Path(root).resolve(strict=False))


def _memo(root):
    with ua._ROWS_MEMO_LOCK:
        return ua._ROWS_MEMO.get(_memo_key(root))


def _clear_memo(root) -> None:
    with ua._ROWS_MEMO_LOCK:
        ua._ROWS_MEMO.pop(_memo_key(root), None)


def _seed_settled(data_root, *, cost_usd=0.25, task_id="child", root_task_id="root"):
    reservation = ua.reserve_attempt(
        _request(data_root, task_id=task_id, root_task_id=root_task_id)
    )
    ua.mark_dispatched(reservation)
    ua.settle_attempt(
        reservation,
        {"prompt_tokens": 100, "completion_tokens": 20},
        cost_usd=cost_usd,
        cost_final=True,
    )


_FOREIGN_ROW = {
    "kind": "external_unmetered",
    "attempt_id": "external-foreign-process",
    "state": "settled",
    "model": "", "provider": "external",
    "cost_usd": None, "cost_final": False,
    "reservation_upper_bound_usd": None,
    "prompt_tokens": 3, "completion_tokens": 1,
    "task_id": "foreign", "root_task_id": "foreign",
    "parent_task_id": "", "category": "external", "source": "test",
}


def _install_counters(monkeypatch):
    """Count full ledger replays AND render re-aggregations (every namespace
    that matters: ``ua._read_records_locked`` feeds the memo; ``_summary`` /
    ``_breakdown_bucket`` are the aggregation entry points the render closures
    resolve as module globals, in ``ua`` and in the ``_usage_rows`` leaf where
    ``_projection_from_final`` lives)."""
    from ouroboros import _usage_rows

    counts = {"full": 0, "summary": 0, "bucket": 0}
    real_read = ua._read_records_locked
    real_summary = ua._summary
    real_bucket = ua._breakdown_bucket

    def counting_read(root, *args, **kwargs):
        counts["full"] += 1
        return real_read(root, *args, **kwargs)

    def counting_summary(rows):
        counts["summary"] += 1
        return real_summary(rows)

    def counting_bucket(rows):
        counts["bucket"] += 1
        return real_bucket(rows)

    monkeypatch.setattr(ua, "_read_records_locked", counting_read)
    for module in (ua, _usage_rows):
        monkeypatch.setattr(module, "_summary", counting_summary)
        monkeypatch.setattr(module, "_breakdown_bucket", counting_bucket)
    return counts


def test_warm_hit_serves_cached_render_with_zero_replays_and_zero_recompute(
    data_root, monkeypatch,
):
    _seed_settled(data_root)
    _clear_memo(data_root)

    first_projection = ua.usage_projection(data_root)
    first_breakdown = ua.usage_breakdown(data_root)
    first_root = ua.usage_projection(data_root, root_task_id="root")
    first_task = ua.usage_breakdown(data_root, task_id="child")

    counts = _install_counters(monkeypatch)
    for _ in range(3):
        assert ua.usage_projection(data_root) == first_projection
        assert ua.usage_breakdown(data_root) == first_breakdown
        assert ua.usage_projection(data_root, root_task_id="root") == first_root
        assert ua.usage_breakdown(data_root, task_id="child") == first_task
    assert counts["full"] == 0, "warm hits must not replay the ledger"
    assert counts["summary"] == 0 and counts["bucket"] == 0, (
        "warm hits must serve the cached render, not re-aggregate"
    )


def test_distinct_call_shapes_get_distinct_cache_entries(data_root, monkeypatch):
    _seed_settled(data_root)
    _clear_memo(data_root)

    limited = ua.usage_projection(data_root, global_limit_usd=50.0)
    env_default = ua.usage_projection(data_root)  # TOTAL_BUDGET=100
    assert limited["limit_usd"] == 50.0
    assert env_default["limit_usd"] == 100.0
    # A hot-reloaded env budget resolves BEFORE the key, so it never collides
    # with the previously cached resolution.
    monkeypatch.setenv("TOTAL_BUDGET", "60")
    assert ua.usage_projection(data_root)["limit_usd"] == 60.0


def test_append_invalidates_cached_renders(data_root):
    _seed_settled(data_root)
    warm = ua.usage_projection(data_root)
    assert warm["attempt_counts"] == {"settled": 1}
    assert _memo(data_root).renders, "warm render must be cached"

    ua.release_attempt(ua.reserve_attempt(_request(data_root, task_id="next")))

    after = ua.usage_projection(data_root)
    assert after["attempt_counts"] == {"settled": 1, "released": 1}
    breakdown = ua.usage_breakdown(data_root)
    assert breakdown["attempt_counts"] == {"settled": 1, "released": 1}


def test_rotation_invalidates_cached_renders(data_root):
    _seed_settled(data_root)
    assert ua.usage_projection(data_root)["attempt_counts"] == {"settled": 1}

    ledger = data_root / ua.LEDGER_REL
    replacement = data_root / "state" / "replacement.jsonl"
    rows = [
        {"seq": 1, "ts": "2026-08-08T00:00:00Z", "kind": "attempt", "attempt_id": "n1",
         "state": "reserved", "model": "m", "provider": "openai",
         "reservation_upper_bound_usd": 0.5, "task_id": "t", "root_task_id": "r"},
    ]
    replacement.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    os.replace(replacement, ledger)

    projection = ua.usage_projection(data_root)
    assert projection["attempt_counts"] == {"reserved": 1}
    assert projection["settled_usd"] == 0.0


def test_non_resumable_crash_tail_is_never_cached(data_root, monkeypatch):
    _seed_settled(data_root)
    ledger = data_root / ua.LEDGER_REL
    raw = ledger.read_bytes()
    assert raw.endswith(b"\n")
    ledger.write_bytes(raw[:-1])  # valid JSON final row, no trailing newline

    _clear_memo(data_root)
    first = ua.usage_projection(data_root)
    memo = _memo(data_root)
    assert memo.resume.st_ino == -2
    assert memo.renders == {}, "a non-resumable tail's render must not be cached"

    counts = _install_counters(monkeypatch)
    assert ua.usage_projection(data_root) == first
    assert counts["summary"] > 0, "every crash-tail read must recompute"
    assert _memo(data_root).renders == {}


def test_include_roots_false_omits_by_root_and_keeps_budget_fields(data_root):
    _seed_settled(data_root, task_id="a", root_task_id="root-a")
    _seed_settled(data_root, task_id="b", root_task_id="root-b", cost_usd=0.1)

    full = ua.usage_projection(data_root, global_limit_usd=5.0)
    slim = ua.usage_projection(data_root, global_limit_usd=5.0, include_roots=False)

    assert "by_root" not in slim
    assert set(full["by_root"]) == {"root-a", "root-b"}
    assert slim["limit_usd"] == 5.0
    assert slim["remaining_known_usd"] == full["remaining_known_usd"]
    trimmed = dict(full)
    trimmed.pop("by_root")
    assert slim == trimmed, "include_roots=False must change nothing but by_root"


def test_one_pass_by_root_grouping_matches_row_filtering(data_root):
    for lane in ("a", "b", "c"):
        _seed_settled(
            data_root, task_id=f"task-{lane}", root_task_id=f"root-{lane}",
            cost_usd=0.05,
        )
    projection = ua.usage_projection(data_root)
    assert sorted(projection["by_root"]) == ["root-a", "root-b", "root-c"]
    for lane in ("a", "b", "c"):
        expected = ua.usage_projection(data_root, root_task_id=f"root-{lane}")
        bucket = projection["by_root"][f"root-{lane}"]
        # The per-root projection resolves its own limit from row evidence; the
        # grouped bucket must agree on every shared monetary field.
        for field in (
            "settled_usd", "confirmed_usd", "estimated_usd", "reserved_usd",
            "unresolved_upper_bound_usd", "accounted_usd", "attempt_counts",
        ):
            assert bucket[field] == expected[field]


def test_served_renders_are_deep_copies_nested_bucket_mutation_is_isolated(data_root):
    _seed_settled(data_root)

    first = ua.usage_breakdown(data_root)
    original = first["by_model"]["openai/gpt-5.2"]["settled_usd"]
    first["by_model"]["openai/gpt-5.2"]["settled_usd"] = 999.0
    second = ua.usage_breakdown(data_root)
    assert second["by_model"]["openai/gpt-5.2"]["settled_usd"] == original

    projection = ua.usage_projection(data_root)
    projection["by_root"]["root"]["settled_usd"] = 999.0
    assert ua.usage_projection(data_root)["by_root"]["root"]["settled_usd"] == original


def test_concurrent_append_between_row_read_and_publish_is_not_cached(
    data_root, monkeypatch,
):
    """GPT#1 race: rows are read under the lock, the render is computed outside
    it. If another writer appends (and another reader advances the memo) in
    that window, publishing the stale render would serve pre-append data until
    the next invalidation. The generation guard must return the stale render to
    THIS caller only, without caching it."""
    _seed_settled(data_root)
    _clear_memo(data_root)

    real_summary = ua._summary
    real_read = ua._read_records_locked
    fired = {}

    def racing_summary(rows):
        if not fired:
            fired["x"] = True
            records = real_read(data_root)
            ua._append_rows_locked(data_root, records, [dict(_FOREIGN_ROW)])
            # A concurrent reader advances the memo (generation bump + clear).
            ua._memoized_final_rows(data_root)
        return real_summary(rows)

    from ouroboros import _usage_rows  # the projection render resolves _summary here

    monkeypatch.setattr(_usage_rows, "_summary", racing_summary)
    stale = ua.usage_projection(data_root, include_roots=False)
    monkeypatch.setattr(_usage_rows, "_summary", real_summary)

    # The caller got a consistent snapshot of the rows it read...
    assert stale["attempt_counts"] == {"settled": 1}
    # ...but the stale render was NOT published.
    assert _memo(data_root).renders == {}

    fresh = ua.usage_projection(data_root, include_roots=False)
    assert fresh["attempt_counts"] == {"settled": 2}
    assert _memo(data_root).renders, "the post-append render caches normally"


def test_api_state_uses_slim_projection_and_payload_is_field_identical(
    tmp_path, monkeypatch,
):
    """gateway/state.py passes include_roots=False; /api/state serializes only
    named scalars, so the response must be byte-for-byte identical to one built
    from the full projection."""
    from ouroboros.gateway.state import api_state
    from supervisor import queue, state, workers

    root = tmp_path / "data"
    (root / "state").mkdir(parents=True)
    (root / "logs").mkdir(parents=True)
    (root / "state" / "state.json").write_text(
        json.dumps({"spent_usd": 0.0, "spent_calls": 0}), encoding="utf-8",
    )
    monkeypatch.setenv("OUROBOROS_DATA_DIR", str(root))
    monkeypatch.setenv("OUROBOROS_SETTINGS_PATH", str(root / "settings.json"))
    monkeypatch.setenv("TOTAL_BUDGET", "7.5")
    ua.ensure_legacy_imported(root)
    _seed_settled(root)

    monkeypatch.setattr(state, "TOTAL_BUDGET_LIMIT", 7.5)
    monkeypatch.setattr(state, "DRIVE_ROOT", str(root))
    monkeypatch.setattr(state, "load_state", lambda: {"current_branch": "ouroboros"})
    monkeypatch.setattr(workers, "WORKERS", {})
    monkeypatch.setattr(workers, "PENDING", [])
    monkeypatch.setattr(workers, "RUNNING", {})
    monkeypatch.setattr(queue, "get_evolution_status_snapshot", lambda **_kwargs: {})

    def _state_request():
        return Request({
            "type": "http", "method": "GET", "path": "/api/state", "headers": [],
            "query_string": b"", "scheme": "http", "server": ("test", 80),
            "client": ("test", 1),
            "app": types.SimpleNamespace(
                state=types.SimpleNamespace(drive_root=root, app_start=0.0),
            ),
        })

    seen_kwargs: list = []
    real_projection = ua.usage_projection

    def capturing(*args, **kwargs):
        seen_kwargs.append(dict(kwargs))
        return real_projection(*args, **kwargs)

    monkeypatch.setattr(ua, "usage_projection", capturing)
    slim_response = asyncio.run(api_state(_state_request()))
    assert slim_response.status_code == 200
    assert any(k.get("include_roots") is False for k in seen_kwargs), (
        "/api/state must request the slim projection"
    )

    def forcing_full(*args, **kwargs):
        kwargs.pop("include_roots", None)
        return real_projection(*args, **kwargs)

    monkeypatch.setattr(ua, "usage_projection", forcing_full)
    full_response = asyncio.run(api_state(_state_request()))
    assert full_response.status_code == 200

    slim_payload = json.loads(slim_response.body)
    full_payload = json.loads(full_response.body)
    slim_payload.pop("uptime")
    full_payload.pop("uptime")
    assert slim_payload == full_payload
