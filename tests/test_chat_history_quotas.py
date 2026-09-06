"""Regression tests for chat-history separate quotas (PR-D2, issue #8).

A progress/telemetry burst must never evict the user's real conversation. Subagent
lineage is kept on top of the progress quota so a flood can't drop a RECENT child's
lifecycle events; older lineage survives while its parent is represented in the
window or still alive (#496), so an ABSENT swarm never re-mints a parent card.
"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

from ouroboros.gateway.history import make_chat_history_endpoint
from ouroboros.subagent_messages import SUBAGENT_MESSAGE_FIELDS


def _run_full(tmp_path, params):
    endpoint = make_chat_history_endpoint(tmp_path)
    resp = asyncio.run(endpoint(SimpleNamespace(query_params=params)))
    return json.loads(resp.body.decode("utf-8"))


def _run(tmp_path, params):
    return _run_full(tmp_path, params)["messages"]


def _lineage_row(ts, child, ev):
    return json.dumps({
        "ts": ts, "content": f"child {ev}", "task_id": child,
        "delegation_role": "subagent", "parent_task_id": "root", "subagent_event": ev,
    })


def test_progress_flood_does_not_evict_human_messages(tmp_path):
    logs = tmp_path / "logs"
    logs.mkdir()
    chat_lines = [
        json.dumps({"ts": f"2026-06-05T00:00:0{i}Z",
                    "direction": "in" if i % 2 == 0 else "out", "text": f"human-{i}"})
        for i in range(5)
    ]
    (logs / "chat.jsonl").write_text("\n".join(chat_lines) + "\n", encoding="utf-8")
    prog_lines = [
        json.dumps({"ts": f"2026-06-05T01:00:{i:02d}Z", "content": f"telemetry-{i}", "task_id": "t1"})
        for i in range(50)
    ]
    (logs / "progress.jsonl").write_text("\n".join(prog_lines) + "\n", encoding="utf-8")

    msgs = _run(tmp_path, {"n_human": "3", "n_progress": "2"})
    human = [m["text"] for m in msgs if not m.get("is_progress")]
    progress = [m for m in msgs if m.get("is_progress")]
    assert human == ["human-2", "human-3", "human-4"]   # not evicted by the flood
    assert len(progress) == 2                            # telemetry bounded by n_progress


def test_task_bound_skill_review_refs_fold_before_human_quota(tmp_path):
    logs = tmp_path / "logs"
    logs.mkdir()
    rows = [
        {
            "ts": f"2026-08-24T00:00:0{i}Z",
            "direction": "in" if i % 2 == 0 else "out",
            "chat_id": 1,
            "text": f"human-{i}",
        }
        for i in range(3)
    ]
    rows.extend({
        "ts": f"2026-08-24T00:01:0{i}Z",
        "direction": "system",
        "type": "skill_review",
        "chat_id": 1,
        "text": f"review-{i}",
        "task_id": "child-1",
        "root_task_id": "root-1",
        "origin_task_id": "child-1",
        "origin_root_task_id": "root-1",
        "presentation_owner_task_id": "root-1",
        "group_id": "task:root-1:alpha",
        "skill": "alpha",
        "status": "warnings" if i < 5 else "clean",
        "job_status": "succeeded",
        "terminal_reason": "succeeded",
        "job_id": f"job-{i}",
        "content_hash": f"hash-{i}",
        "review_round": i + 1,
        "snapshot_attempt": 1,
        "snapshot_revised": i > 0,
        "source": "tool",
    } for i in range(6))
    (logs / "chat.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8",
    )
    (logs / "progress.jsonl").write_text("", encoding="utf-8")

    messages = _run(tmp_path, {"n_human": "3", "n_progress": "1"})

    assert [m["text"] for m in messages if m.get("role") != "system"] == [
        "human-0", "human-1", "human-2",
    ]
    review_rows = [m for m in messages if m.get("system_type") == "skill_review"]
    assert len(review_rows) == 1
    row = review_rows[0]
    assert row["task_id"] == "child-1"
    assert row["presentation_owner_task_id"] == "root-1"
    group = row["review_group"]
    assert group == {
        "surface": "skill",
        "id": "task:root-1:alpha",
        "skill": "alpha",
        "presentation_owner_task_id": "root-1",
        "projected_attempt_count": 6,
        "count_is_authoritative": False,
        "attempts": group["attempts"],
    }
    assert [attempt["job_id"] for attempt in group["attempts"]] == [
        f"job-{i}" for i in range(6)
    ]
    review_only = _run(tmp_path, {"n_human": "0", "n_progress": "1"})
    assert [m["system_type"] for m in review_only] == ["skill_review"]
    zero_window = _run_full(tmp_path, {"n_human": "0", "n_progress": "0"})
    assert not [
        row for row in zero_window["messages"]
        if row.get("system_type") == "skill_review"
    ]
    assert zero_window["window"] == {"complete": False, "truncated_by": ["quota"]}


def test_task_bound_skill_review_owners_use_the_progress_window(tmp_path):
    logs = tmp_path / "logs"
    logs.mkdir()
    rows = []
    for owner_index in range(5):
        for skill in ("alpha", "beta"):
            for attempt_index in range(2):
                rows.append({
                    "ts": f"2026-08-24T00:{owner_index:02d}:{attempt_index:02d}Z",
                    "direction": "system",
                    "type": "skill_review",
                    "chat_id": 1,
                    "text": f"{skill}-{owner_index}-{attempt_index}",
                    "task_id": f"child-{owner_index}",
                    "root_task_id": f"owner-{owner_index}",
                    "presentation_owner_task_id": f"owner-{owner_index}",
                    "group_id": f"task:owner-{owner_index}:{skill}",
                    "skill": skill,
                    "status": "clean",
                    "job_id": f"job-{owner_index}-{skill}-{attempt_index}",
                })
    (logs / "chat.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8",
    )
    (logs / "progress.jsonl").write_text("", encoding="utf-8")

    payload = _run_full(tmp_path, {"n_progress": "2"})
    review_rows = [
        row for row in payload["messages"]
        if row.get("system_type") == "skill_review"
    ]
    assert {row["presentation_owner_task_id"] for row in review_rows} == {
        "owner-3", "owner-4",
    }
    assert len(review_rows) == 4
    assert all(len(row["review_group"]["attempts"]) == 2 for row in review_rows)
    assert payload["window"] == {"complete": False, "truncated_by": ["quota"]}

    expanded = _run_full(tmp_path, {"n_progress": "10"})
    expanded_reviews = [
        row for row in expanded["messages"]
        if row.get("system_type") == "skill_review"
    ]
    assert len(expanded_reviews) == 10
    assert {row["presentation_owner_task_id"] for row in expanded_reviews} == {
        f"owner-{index}" for index in range(5)
    }
    assert expanded["window"] == {"complete": True, "truncated_by": []}


def test_legacy_skill_refs_fold_only_with_safe_root_and_skill(tmp_path):
    logs = tmp_path / "logs"
    logs.mkdir()
    rows = [
        {
            "ts": f"2026-08-24T00:00:0{i}Z",
            "direction": "system",
            "type": "skill_review",
            "chat_id": 1,
            "text": f"legacy-{i}",
            "task_id": "child-1",
            "root_task_id": "root-1",
            "skill": "alpha",
            "job_id": f"legacy-job-{i}",
        }
        for i in range(2)
    ]
    rows.append({
        "ts": "2026-08-24T00:00:03Z",
        "direction": "system",
        "type": "skill_review",
        "chat_id": 1,
        "text": "unsafe legacy",
        "task_id": "child-only",
        "skill": "beta",
        "job_id": "unsafe-job",
    })
    (logs / "chat.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8",
    )
    (logs / "progress.jsonl").write_text("", encoding="utf-8")

    messages = _run(tmp_path, {"n_human": "10", "n_progress": "1"})

    groups = [m["review_group"] for m in messages if m.get("review_group")]
    assert len(groups) == 1
    assert groups[0]["id"] == "task:root-1:alpha"
    assert all(attempt["skill"] == "alpha" for attempt in groups[0]["attempts"])
    assert all(
        attempt["group_id"] == "task:root-1:alpha"
        for attempt in groups[0]["attempts"]
    )
    assert all(
        attempt["presentation_owner_task_id"] == "root-1"
        for attempt in groups[0]["attempts"]
    )
    unsafe = next(m for m in messages if m.get("job_id") == "unsafe-job")
    assert "review_group" not in unsafe


def test_skill_review_group_dedupes_only_nonempty_job_ids_latest_wins(tmp_path):
    logs = tmp_path / "logs"
    logs.mkdir()
    base = {
        "direction": "system", "type": "skill_review", "chat_id": 1,
        "task_id": "child-1", "root_task_id": "root-1",
        "presentation_owner_task_id": "root-1",
        "group_id": "task:root-1:alpha", "skill": "alpha",
    }
    rows = [
        {**base, "ts": "2026-08-24T00:00:00Z", "text": "duplicate-old",
         "job_id": "same-job", "status": "warnings"},
        {**base, "ts": "2026-08-24T00:00:01Z", "text": "legacy-one",
         "job_id": "", "status": "warnings"},
        {**base, "ts": "2026-08-24T00:00:02Z", "text": "duplicate-new",
         "job_id": "same-job", "status": "clean"},
        {**base, "ts": "2026-08-24T00:00:03Z", "text": "legacy-two",
         "job_id": "", "status": "clean"},
    ]
    (logs / "chat.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8",
    )
    (logs / "progress.jsonl").write_text("", encoding="utf-8")

    messages = _run(tmp_path, {"n_human": "10", "n_progress": "1"})

    review_rows = [m for m in messages if m.get("system_type") == "skill_review"]
    assert len(review_rows) == 1
    attempts = review_rows[0]["review_group"]["attempts"]
    assert [(a["ts"], a["job_id"], a["status"]) for a in attempts] == [
        ("2026-08-24T00:00:01Z", "", "warnings"),
        ("2026-08-24T00:00:02Z", "same-job", "clean"),
        ("2026-08-24T00:00:03Z", "", "clean"),
    ]
    assert review_rows[0]["review_group"]["projected_attempt_count"] == 3


def test_panel_chat_zero_stays_hidden_and_request_zero_still_means_main(tmp_path):
    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "chat.jsonl").write_text(
        "\n".join([
            json.dumps({
                "ts": "2026-08-24T00:00:00Z", "direction": "system",
                "type": "skill_review", "chat_id": 0, "text": "panel-only",
                "skill": "alpha", "job_id": "panel-job",
            }),
            json.dumps({
                "ts": "2026-08-24T00:00:01Z", "direction": "in",
                "chat_id": 1, "text": "main-row",
            }),
        ]) + "\n",
        encoding="utf-8",
    )
    (logs / "progress.jsonl").write_text("", encoding="utf-8")

    assert [m["text"] for m in _run(tmp_path, {})] == ["main-row"]
    assert [m["text"] for m in _run(tmp_path, {"chat_id": "0"})] == ["main-row"]


def test_panel_zero_exclusion_does_not_override_project_task_binding():
    from ouroboros.gateway.history import _make_thread_filter

    project_filter = _make_thread_filter(42, {42}, [], {"root-1": 42})
    main_filter = _make_thread_filter(1, {42}, [], {"root-1": 42})

    row = {"task_id": "root-1"}
    assert project_filter(0, row) is True
    assert main_filter(0, row) is False


def test_virtual_lifecycle_row_uses_normal_thread_filter(tmp_path):
    import ouroboros.skill_lifecycle_queue as lifecycle_queue

    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "chat.jsonl").write_text("", encoding="utf-8")
    (logs / "progress.jsonl").write_text("", encoding="utf-8")
    panel = lifecycle_queue.LifecycleJob(
        id="panel-job", kind="review", target="alpha", status="running", chat_id=0,
        group_id="manual:alpha",
    )
    task_bound = lifecycle_queue.LifecycleJob(
        id="task-job", kind="review", target="alpha", status="running", chat_id=1,
        group_id="task:root-1:alpha", task_id="child-1", root_task_id="root-1",
        presentation_owner_task_id="root-1",
    )
    previous = lifecycle_queue._active
    try:
        lifecycle_queue._active = panel
        assert not [m for m in _run(tmp_path, {}) if m.get("lifecycle_virtual")]
        lifecycle_queue._active = task_bound
        rows = [m for m in _run(tmp_path, {}) if m.get("lifecycle_virtual")]
        assert len(rows) == 1
        assert rows[0]["chat_id"] == 1
        assert rows[0]["presentation_owner_task_id"] == "root-1"
        assert rows[0]["lifecycle"]["task_id"] == "child-1"
    finally:
        lifecycle_queue._active = previous


def test_recent_lineage_survives_progress_flood(tmp_path):
    """A RECENT child's lifecycle events survive even a tiny progress quota."""
    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "chat.jsonl").write_text("", encoding="utf-8")
    lines = [
        json.dumps({"ts": f"2026-06-05T00:00:{i:02d}Z", "content": f"noise-{i}", "task_id": "root"})
        for i in range(50)
    ]
    # 3 subagent lifecycle events INSIDE the recent window (near the latest noise)
    for i, ev in zip((47, 48, 49), ("scheduled", "update", "completed")):
        lines.append(_lineage_row(f"2026-06-05T00:00:{i:02d}Z", "child1", ev))
    (logs / "progress.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")

    msgs = _run(tmp_path, {"n_progress": "5"})
    lineage = [m for m in msgs if m.get("delegation_role") == "subagent"]
    assert len(lineage) == 3
    assert {m.get("subagent_event") for m in lineage} == {"scheduled", "update", "completed"}


def test_old_lineage_does_not_resurrect_finished_swarm(tmp_path):
    """Lineage OLDER than the window whose parent is ABSENT from it (and not
    alive) is dropped, so a finished swarm never re-mints a stuck parent card."""
    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "chat.jsonl").write_text("", encoding="utf-8")
    lines = []
    # OLD swarm lineage (4 days earlier)
    for i, ev in zip((1, 2, 3), ("scheduled", "running", "completed")):
        lines.append(_lineage_row(f"2026-06-01T00:00:0{i}Z", "oldchild", ev))
    # RECENT telemetry flood
    for i in range(50):
        lines.append(json.dumps({"ts": f"2026-06-05T02:00:{i:02d}Z", "content": f"noise-{i}", "task_id": "other"}))
    (logs / "progress.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")

    msgs = _run(tmp_path, {"n_progress": "5"})
    lineage = [m for m in msgs if m.get("delegation_role") == "subagent"]
    assert lineage == []  # old swarm does NOT resurface


def test_delegation_role_root_respects_progress_quota(tmp_path):
    """delegation_role='root' is NOT subagent lineage and must obey the quota."""
    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "chat.jsonl").write_text("", encoding="utf-8")
    lines = [
        json.dumps({"ts": f"2026-06-05T00:00:{i:02d}Z", "content": f"root-{i}",
                    "task_id": f"root-{i}", "delegation_role": "root"})
        for i in range(50)
    ]
    (logs / "progress.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    progress = [m for m in _run(tmp_path, {"n_progress": "5"}) if m.get("is_progress")]
    assert len(progress) == 5  # 'root' did not bypass the quota


def test_default_quotas_keep_recent_history(tmp_path):
    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "chat.jsonl").write_text(
        json.dumps({"ts": "2026-06-05T00:00:00Z", "direction": "in", "text": "hello"}) + "\n",
        encoding="utf-8",
    )
    (logs / "progress.jsonl").write_text("", encoding="utf-8")
    msgs = _run(tmp_path, {})
    assert any(m.get("text") == "hello" and not m.get("is_progress") for m in msgs)


def test_history_annotates_terminal_from_effective_status(tmp_path, monkeypatch):
    """A SIGKILLed/panic'd task whose raw result is stuck "running" but whose
    EFFECTIVE status is failed (stale-orphan guard) must get task_terminal_status,
    so its card finalizes instead of replaying "Working" forever."""
    import ouroboros.task_status as ts_mod
    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "chat.jsonl").write_text("", encoding="utf-8")
    (logs / "progress.jsonl").write_text(
        json.dumps({"ts": "2026-06-05T00:00:00Z", "content": "step", "task_id": "stuck1"}) + "\n",
        encoding="utf-8",
    )
    # Simulate the orphan guard: effective status resolves to "failed" even though
    # the raw on-disk file is still "running".
    monkeypatch.setattr(
        ts_mod, "load_effective_task_result",
        lambda dr, tid, **kw: {"status": "failed"} if tid == "stuck1" else {},
    )
    msgs = _run(tmp_path, {})
    row = next(m for m in msgs if m.get("task_id") == "stuck1")
    assert row.get("task_terminal_status") == "failed"


def test_history_projects_terminal_review_truth_without_task_summary(tmp_path, monkeypatch):
    """The last retained progress row is the bounded replay anchor when the
    best-effort task summary is absent; earlier progress must not duplicate the
    panel details."""
    import ouroboros.task_status as ts_mod

    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "chat.jsonl").write_text("", encoding="utf-8")
    (logs / "progress.jsonl").write_text(
        "\n".join([
            json.dumps({"ts": "2026-06-05T00:00:00Z", "content": "step one", "task_id": "review1"}),
            json.dumps({"ts": "2026-06-05T00:00:01Z", "content": "step two", "task_id": "review1"}),
        ]) + "\n",
        encoding="utf-8",
    )
    projection = {"panels": [{"panel_id": "terminal-review", "aggregate_signal": "DEGRADED"}]}
    monkeypatch.setattr(
        ts_mod,
        "load_effective_task_result",
        lambda dr, tid, **kw: {
            "status": "completed",
            "reason_code": "acceptance_degraded",
            "outcome_axes": {
                "lifecycle": {"status": "completed"},
                "execution": {"status": "ok"},
                "objective": {"status": "best_effort"},
                "review": {"status": "degraded"},
                "artifacts": {"status": "ready"},
            },
            "review_projection": projection,
        } if tid == "review1" else {},
    )

    rows = [m for m in _run(tmp_path, {}) if m.get("task_id") == "review1"]
    assert len(rows) == 2
    assert all(row.get("task_terminal_status") == "completed" for row in rows)
    assert "review_projection" not in rows[0]
    assert rows[1]["outcome_axes"]["review"]["status"] == "degraded"
    assert rows[1]["reason_code"] == "acceptance_degraded"
    assert rows[1]["review_projection"] == projection


def test_history_projects_terminal_review_truth_only_on_existing_summary(tmp_path, monkeypatch):
    """A retained summary is the one review-truth anchor; progress keeps only
    terminal lifecycle so replay cannot feed the same panel through two rows."""
    import ouroboros.task_status as ts_mod

    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "chat.jsonl").write_text(
        json.dumps({
            "ts": "2026-06-05T00:00:01Z",
            "direction": "system",
            "type": "task_summary",
            "task_id": "review2",
            "chat_id": 1,
            "text": "terminal summary",
            "tool_calls": 1,
            "rounds": 2,
        }) + "\n",
        encoding="utf-8",
    )
    (logs / "progress.jsonl").write_text(
        json.dumps({
            "ts": "2026-06-05T00:00:00Z",
            "content": "last progress",
            "task_id": "review2",
        }) + "\n",
        encoding="utf-8",
    )
    projection = {
        "panels": [{"panel_id": "terminal-review", "aggregate_signal": "DEGRADED"}],
    }
    monkeypatch.setattr(
        ts_mod,
        "load_effective_task_result",
        lambda _dr, tid, **kw: {
            "status": "completed",
            "reason_code": "acceptance_degraded",
            "outcome_axes": {
                "objective": {"status": "best_effort"},
                "review": {"status": "degraded"},
            },
            "review_projection": projection,
        } if tid == "review2" else {},
    )

    rows = [row for row in _run(tmp_path, {}) if row.get("task_id") == "review2"]
    progress = next(row for row in rows if row.get("is_progress"))
    summary = next(row for row in rows if row.get("system_type") == "task_summary")
    assert progress["task_terminal_status"] == "completed"
    assert "review_projection" not in progress
    assert "reason_code" not in progress
    assert summary["review_projection"] == projection
    assert summary["reason_code"] == "acceptance_degraded"
    assert summary["outcome_axes"]["review"]["status"] == "degraded"


# --- perf2 P3: default window constants, truncation metadata, variant A ------


def test_default_request_does_not_open_archives_when_live_tail_sufficient(
    tmp_path, monkeypatch,
):
    """A DEFAULT (no explicit quotas) request whose live files already satisfy
    the module-constant window must never open rotated archive segments."""
    from ouroboros.gateway import _helpers

    logs = tmp_path / "logs"
    logs.mkdir()
    archive = tmp_path / "archive"
    archive.mkdir()
    (archive / "chat_20260605T000000.jsonl").write_text(
        json.dumps({"ts": "2026-06-04T00:00:00Z", "direction": "in", "text": "archived human"}) + "\n",
        encoding="utf-8",
    )
    (archive / "progress_20260605T000000.jsonl").write_text(
        json.dumps({"ts": "2026-06-04T00:00:00Z", "content": "archived step", "task_id": "t-old"}) + "\n",
        encoding="utf-8",
    )
    # Live tails exceed the default quotas (150 human / 60 progress).
    (logs / "chat.jsonl").write_text(
        "\n".join(
            json.dumps({
                "ts": f"2026-06-05T{i // 3600:02d}:{(i % 3600) // 60:02d}:{i % 60:02d}Z",
                "direction": "in" if i % 2 else "out", "text": f"human-{i}",
            })
            for i in range(160)
        ) + "\n",
        encoding="utf-8",
    )
    (logs / "progress.jsonl").write_text(
        "\n".join(
            json.dumps({
                "ts": f"2026-06-05T01:{i // 60:02d}:{i % 60:02d}Z",
                "content": f"telemetry-{i}", "task_id": "t1",
            })
            for i in range(70)
        ) + "\n",
        encoding="utf-8",
    )

    seen = []
    real_iter = _helpers.iter_jsonl_objects

    def counted(path, *args, **kwargs):
        seen.append(str(path))
        return real_iter(path, *args, **kwargs)

    monkeypatch.setattr(_helpers, "iter_jsonl_objects", counted)

    msgs = _run(tmp_path, {})
    assert seen  # the bounded reader served the endpoint
    assert not any("/archive/" in path for path in seen)  # archives untouched
    human = [m for m in msgs if not m.get("is_progress")]
    progress = [m for m in msgs if m.get("is_progress")]
    assert len(human) == 150   # default n_human window
    assert len(progress) == 60  # default n_progress window
    texts = [m["text"] for m in msgs]
    assert "archived human" not in texts
    assert "archived step" not in texts


def test_window_metadata_complete_when_everything_fits(tmp_path):
    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "chat.jsonl").write_text(
        json.dumps({"ts": "2026-06-05T00:00:00Z", "direction": "in", "text": "hello"}) + "\n",
        encoding="utf-8",
    )
    (logs / "progress.jsonl").write_text(
        json.dumps({"ts": "2026-06-05T00:00:01Z", "content": "step", "task_id": "t1"}) + "\n",
        encoding="utf-8",
    )
    window = _run_full(tmp_path, {})["window"]
    assert window == {"complete": True, "truncated_by": []}


def test_window_metadata_reports_quota_truncation(tmp_path):
    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "chat.jsonl").write_text(
        "\n".join(
            json.dumps({"ts": f"2026-06-05T00:00:{i:02d}Z",
                        "direction": "in" if i % 2 else "out", "text": f"human-{i}"})
            for i in range(10)
        ) + "\n",
        encoding="utf-8",
    )
    (logs / "progress.jsonl").write_text("", encoding="utf-8")
    payload = _run_full(tmp_path, {"n_human": "3"})
    assert payload["window"] == {"complete": False, "truncated_by": ["quota"]}
    assert len(payload["messages"]) == 3  # the slice actually applied


def test_review_references_use_the_progress_window_without_consuming_telemetry(tmp_path):
    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "chat.jsonl").write_text("", encoding="utf-8")
    rows = [
        {
            "ts": f"2026-08-28T00:00:0{i}Z",
            "type": "review_reference",
            "surface": "plan_review",
            "task_id": f"owner-{i}",
            "presentation_owner_task_id": f"owner-{i}",
            "review_fingerprint": str(i) * 64,
            "state_revision": f"revision-{i}",
            "content": "",
        }
        for i in range(5)
    ]
    rows.append({
        "ts": "2026-08-28T00:00:05Z",
        "type": "review_reference",
        "surface": "plan_review",
        "task_id": "owner-1",
        "presentation_owner_task_id": "owner-1",
        "review_fingerprint": "a" * 64,
        "state_revision": "revision-1-new",
        "content": "",
    })
    rows.extend({
        "ts": f"2026-08-28T00:01:0{i}Z",
        "task_id": "root",
        "content": f"telemetry-{i}",
    } for i in range(4))
    (logs / "progress.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8",
    )

    payload = _run_full(tmp_path, {"n_progress": "3"})
    references = [
        row for row in payload["messages"]
        if row.get("system_type") == "review_reference"
    ]
    telemetry = [
        row for row in payload["messages"]
        if row.get("is_progress") and row.get("system_type") != "review_reference"
    ]
    assert [row["presentation_owner_task_id"] for row in references] == [
        "owner-3", "owner-4", "owner-1",
    ]
    assert references[-1]["state_revision"] == "revision-1-new"
    assert [row["text"] for row in telemetry] == [
        "telemetry-1", "telemetry-2", "telemetry-3",
    ]
    assert payload["window"] == {"complete": False, "truncated_by": ["quota"]}

    expanded = _run_full(tmp_path, {"n_progress": "10"})
    assert len([
        row for row in expanded["messages"]
        if row.get("system_type") == "review_reference"
    ]) == 5
    assert expanded["window"] == {"complete": True, "truncated_by": []}
    assert not [
        row for row in _run(tmp_path, {"n_progress": "0"})
        if row.get("system_type") == "review_reference"
    ]


def test_window_metadata_reports_quota_when_slice_cuts_only_system_rows(tmp_path):
    """MAJOR review fix: direction:"system" rows (per-task task_summary) do not
    count toward the reader's in/out quota, but the n_human tail slice still
    drops them — the window must honestly report "quota", never complete."""
    logs = tmp_path / "logs"
    logs.mkdir()
    rows = [
        json.dumps({
            "ts": f"2026-06-05T00:00:0{i}Z",
            "direction": "system",
            "type": "task_summary",
            "task_id": f"t{i}",
            "text": f"Task t{i} finished.",
        })
        for i in range(4)
    ]
    # One newest human row; n_human=3 keeps it + the 2 newest system rows and
    # drops ONLY the 2 oldest system rows (in/out quota count stays below 3).
    rows.append(json.dumps({
        "ts": "2026-06-05T00:01:00Z", "direction": "in", "text": "hello",
    }))
    (logs / "chat.jsonl").write_text("\n".join(rows) + "\n", encoding="utf-8")
    (logs / "progress.jsonl").write_text("", encoding="utf-8")

    payload = _run_full(tmp_path, {"n_human": "3"})
    assert payload["window"] == {"complete": False, "truncated_by": ["quota"]}
    human = [m for m in payload["messages"] if not m.get("is_progress")]
    assert len(human) == 3  # the slice really dropped the two oldest system rows


def test_window_metadata_reports_archive_floor(tmp_path):
    """More rotated segments exist than the 3-archive backfill bound and the
    quota is still unmet -> the reader stopped at the archive floor."""
    logs = tmp_path / "logs"
    logs.mkdir()
    archive = tmp_path / "archive"
    archive.mkdir()
    for idx in range(4):  # 4 archives > the 3-newest backfill bound
        (archive / f"progress_2026060{idx + 1}T000000.jsonl").write_text(
            json.dumps({
                "ts": f"2026-06-0{idx + 1}T00:00:00Z",
                "content": f"archived-{idx + 1}", "task_id": "t1",
            }) + "\n",
            encoding="utf-8",
        )
    (logs / "progress.jsonl").write_text(
        json.dumps({"ts": "2026-06-05T01:00:00Z", "content": "live step", "task_id": "t1"}) + "\n",
        encoding="utf-8",
    )
    (logs / "chat.jsonl").write_text("", encoding="utf-8")

    payload = _run_full(tmp_path, {})
    assert payload["window"] == {"complete": False, "truncated_by": ["archive_floor"]}
    texts = [m["text"] for m in payload["messages"]]
    # The 3 newest archives were backfilled; the oldest stayed beyond the floor.
    assert "archived-2" in texts and "archived-4" in texts and "live step" in texts
    assert "archived-1" not in texts


def test_window_metadata_reports_lineage_cap(tmp_path):
    """A swarm fan-out larger than the 300-row lineage cap keeps only the
    newest lineage rows and discloses the cap in the window metadata."""
    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "chat.jsonl").write_text("", encoding="utf-8")
    (logs / "progress.jsonl").write_text(
        "\n".join(
            _lineage_row(f"2026-06-05T{i // 3600:02d}:{(i % 3600) // 60:02d}:{i % 60:02d}Z",
                         "bigswarm", "update")
            for i in range(310)
        ) + "\n",
        encoding="utf-8",
    )
    payload = _run_full(tmp_path, {})
    assert payload["window"]["complete"] is False
    assert payload["window"]["truncated_by"] == ["lineage_cap"]
    lineage = [m for m in payload["messages"] if m.get("delegation_role") == "subagent"]
    assert len(lineage) == 300  # the cap kept only the newest rows


def test_active_silent_child_lineage_survives_recency_floor(tmp_path, monkeypatch):
    """perf2 P3 variant A (owner decision): a QUIET but still-ACTIVE child whose
    lineage rows are all OLDER than the recency floor keeps them — the child's
    card must be reproducible on reload during a live swarm. Each task_results
    read happens ONCE per request (the pre-floor map feeds the annotation)."""
    import ouroboros.task_status as ts_mod

    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "chat.jsonl").write_text("", encoding="utf-8")
    lines = []
    # The quiet child's lifecycle events, all far older than the floor below.
    for i, ev in zip((1, 2, 3), ("scheduled", "running", "update")):
        lines.append(_lineage_row(f"2026-06-05T00:00:0{i}Z", "quietchild", ev))
    # A recent telemetry flood establishes a much newer recency floor.
    for i in range(50):
        lines.append(json.dumps({
            "ts": f"2026-06-05T02:00:{i:02d}Z", "content": f"noise-{i}", "task_id": "root",
        }))
    (logs / "progress.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")

    calls: dict = {}

    def fake_load(_dr, tid, **_kw):
        calls[tid] = calls.get(tid, 0) + 1
        return {"status": "running"} if tid == "quietchild" else {}

    monkeypatch.setattr(ts_mod, "load_effective_task_result", fake_load)

    msgs = _run(tmp_path, {"n_progress": "5"})
    lineage = [m for m in msgs if m.get("delegation_role") == "subagent"]
    assert len(lineage) == 3
    assert all(m.get("task_id") == "quietchild" for m in lineage)
    assert {m.get("subagent_event") for m in lineage} == {"scheduled", "running", "update"}
    # Still active: no terminal annotation may finalize the replayed card.
    assert all("task_terminal_status" not in m for m in lineage)
    # task_results were read once per task for the whole request.
    assert calls.get("quietchild") == 1
    assert calls.get("root") == 1


def test_terminal_silent_child_older_than_floor_stays_dropped(tmp_path):
    """Anti-zombie preserved: a child with a TERMINAL task result, lineage older
    than the floor and a parent ABSENT from the window, is still dropped."""
    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "chat.jsonl").write_text("", encoding="utf-8")
    lines = []
    for i, ev in zip((1, 2, 3), ("scheduled", "running", "completed")):
        lines.append(_lineage_row(f"2026-06-05T00:00:0{i}Z", "donechild", ev))
    for i in range(50):
        lines.append(json.dumps({
            "ts": f"2026-06-05T02:00:{i:02d}Z", "content": f"noise-{i}", "task_id": "other",
        }))
    (logs / "progress.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    results = tmp_path / "task_results"
    results.mkdir()
    (results / "donechild.json").write_text(
        json.dumps({"task_id": "donechild", "status": "completed"}), encoding="utf-8",
    )

    msgs = _run(tmp_path, {"n_progress": "5"})
    lineage = [m for m in msgs if m.get("delegation_role") == "subagent"]
    assert lineage == []  # terminal quiet child does NOT resurface


# --- Stream S: closed lineage window on chat FINALS (floor-symmetric strip) ---


def _child_final_row(ts, task_id, *, lineage=True):
    row = {
        "ts": ts, "direction": "out", "chat_id": 1,
        "text": "child answer", "task_id": task_id,
    }
    if lineage:
        row.update({
            "delegation_role": "subagent", "parent_task_id": "root",
            "root_task_id": "root", "subagent_task_id": task_id,
        })
    return json.dumps(row)


def _fresh_noise_lines(count=5):
    return [
        json.dumps({"ts": f"2026-06-05T02:00:{i:02d}Z", "content": f"noise-{i}",
                    "task_id": "other"})
        for i in range(count)
    ]


def test_stale_child_final_loses_raw_lineage_fields(tmp_path):
    """NEG-1: a child FINAL chat row (raw lineage fields) older than a fresh
    progress floor, with the child settled, loses every subagent lineage field
    on the emitted payload — so the client cannot re-mint an orphaned
    "Working" parent card whose own rows aged out of the window."""
    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "chat.jsonl").write_text(
        _child_final_row("2026-06-01T00:00:00Z", "child") + "\n", encoding="utf-8",
    )
    # Fresh non-lineage telemetry NEWER than the child final -> fresh floor.
    (logs / "progress.jsonl").write_text(
        "\n".join(_fresh_noise_lines()) + "\n", encoding="utf-8",
    )
    results = tmp_path / "task_results"
    results.mkdir()
    (results / "root.json").write_text(
        json.dumps({"task_id": "root", "status": "completed"}), encoding="utf-8",
    )
    (results / "child.json").write_text(
        json.dumps({"task_id": "child", "status": "completed"}), encoding="utf-8",
    )

    row = next(m for m in _run(tmp_path, {}) if m.get("task_id") == "child")
    assert row["text"] == "child answer"  # the row itself survives, de-roled
    assert not any(field in row for field in SUBAGENT_MESSAGE_FIELDS)


def test_stale_legacy_child_final_skips_task_result_revival(tmp_path):
    """NEG-2: a legacy child FINAL (no raw lineage fields in chat.jsonl) older
    than a fresh floor must NOT get lineage re-injected from task_results."""
    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "chat.jsonl").write_text(
        _child_final_row("2026-06-01T00:00:00Z", "child-legacy", lineage=False) + "\n",
        encoding="utf-8",
    )
    (logs / "progress.jsonl").write_text(
        "\n".join(_fresh_noise_lines()) + "\n", encoding="utf-8",
    )
    results = tmp_path / "task_results"
    results.mkdir()
    (results / "root.json").write_text(
        json.dumps({"task_id": "root", "status": "completed"}), encoding="utf-8",
    )
    (results / "child-legacy.json").write_text(json.dumps({
        "task_id": "child-legacy", "status": "completed",
        "delegation_role": "subagent", "parent_task_id": "root",
        "root_task_id": "root", "role": "reviewer",
    }), encoding="utf-8")

    row = next(m for m in _run(tmp_path, {}) if m.get("task_id") == "child-legacy")
    assert not any(field in row for field in SUBAGENT_MESSAGE_FIELDS)


def test_skill_review_group_never_inherits_legacy_subagent_lineage(tmp_path):
    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "chat.jsonl").write_text(json.dumps({
        "ts": "2026-08-24T00:00:00Z", "direction": "system",
        "type": "skill_review", "chat_id": 1, "text": "review",
        "task_id": "child-1", "root_task_id": "root-1",
        "presentation_owner_task_id": "root-1",
        "group_id": "task:root-1:alpha", "skill": "alpha", "job_id": "job-1",
    }) + "\n", encoding="utf-8")
    (logs / "progress.jsonl").write_text("", encoding="utf-8")
    results = tmp_path / "task_results"
    results.mkdir()
    (results / "child-1.json").write_text(json.dumps({
        "task_id": "child-1", "status": "completed",
        "delegation_role": "subagent", "parent_task_id": "root-1",
        "root_task_id": "root-1", "role": "reviewer",
    }), encoding="utf-8")

    row = next(m for m in _run(tmp_path, {}) if m.get("review_group"))
    assert row["task_id"] == "child-1"
    assert row["presentation_owner_task_id"] == "root-1"
    injected_lineage_fields = set(SUBAGENT_MESSAGE_FIELDS) - {"root_task_id"}
    assert not any(field in row for field in injected_lineage_fields)


def test_stale_child_final_keeps_lineage_while_child_active(tmp_path, monkeypatch):
    """POS-2: a child final older than the floor keeps its lineage when the
    child is still ACTIVE (non-terminal effective status + in-window lineage
    progress row makes it active_children-eligible)."""
    import ouroboros.task_status as ts_mod

    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "chat.jsonl").write_text(
        _child_final_row("2026-06-01T00:00:00Z", "child1") + "\n", encoding="utf-8",
    )
    lines = _fresh_noise_lines()
    lines.append(_lineage_row("2026-06-05T02:00:06Z", "child1", "update"))
    (logs / "progress.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    monkeypatch.setattr(
        ts_mod, "load_effective_task_result",
        lambda _dr, tid, **_kw: {"status": "running"} if tid == "child1" else {},
    )

    finals = [m for m in _run(tmp_path, {"n_progress": "5"})
              if m.get("task_id") == "child1" and not m.get("is_progress")]
    assert finals and finals[0]["delegation_role"] == "subagent"
    assert finals[0]["parent_task_id"] == "root"


def test_recent_child_final_keeps_lineage_fields(tmp_path):
    """POS-3: a child final with ts >= floor keeps its lineage fields — fresh
    swarms keep the nested-card UX."""
    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "chat.jsonl").write_text(
        _child_final_row("2026-06-05T03:00:00Z", "child2") + "\n", encoding="utf-8",
    )
    (logs / "progress.jsonl").write_text(
        "\n".join(_fresh_noise_lines()) + "\n", encoding="utf-8",
    )
    results = tmp_path / "task_results"
    results.mkdir()
    (results / "child2.json").write_text(
        json.dumps({"task_id": "child2", "status": "completed"}), encoding="utf-8",
    )

    row = next(m for m in _run(tmp_path, {}) if m.get("task_id") == "child2")
    assert row["delegation_role"] == "subagent"
    assert row["parent_task_id"] == "root"


def test_legacy_limit_governs_human_quota_when_n_human_absent(tmp_path):
    """The legacy `limit` parameter is the n_human DEFAULT, so shipped CLIs that
    always sent it stop getting a placebo; non-positive = absent; explicit n_human wins."""
    logs = tmp_path / "logs"
    logs.mkdir()
    chat_lines = [
        json.dumps({"ts": f"2026-06-05T00:00:0{i}Z",
                    "direction": "in" if i % 2 == 0 else "out", "text": f"human-{i}"})
        for i in range(5)
    ]
    (logs / "chat.jsonl").write_text("\n".join(chat_lines) + "\n", encoding="utf-8")

    legacy_only = _run(tmp_path, {"limit": "2"})
    assert [m["text"] for m in legacy_only] == ["human-3", "human-4"]

    explicit_wins = _run(tmp_path, {"limit": "2", "n_human": "3"})
    assert [m["text"] for m in explicit_wins] == ["human-2", "human-3", "human-4"]
    for non_positive in ({"limit": "0"}, {"limit": "-5"}):   # never an empty conversation
        assert _run(tmp_path, non_positive) == _run(tmp_path, {})
