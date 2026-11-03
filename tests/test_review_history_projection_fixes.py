"""Regression coverage for quiet review references on Chat history rails."""

from __future__ import annotations

from ouroboros import dialogue_store as ds
from ouroboros.chat_chain import format_address, row_address
from ouroboros.gateway.history import (
    _collect_progress_rows,
    _fold_task_bound_skill_reviews,
)


def test_duplicate_lifecycle_pointer_replays_without_a_fake_task(tmp_path):
    progress = tmp_path / "logs" / "progress.jsonl"
    pointer = {
        "job_id": "skill-job-1", "kind": "review", "target": "alpha",
        "status": "running", "chat_id": 17,
        "group_id": "task:root:alpha", "task_id": "child",
        "root_task_id": "root", "origin_task_id": "child",
        "origin_root_task_id": "root", "presentation_owner_task_id": "root",
        "source": "tool",
    }
    ds.append(tmp_path, "progress", {
        "ts": "2026-08-25T00:00:00Z", "type": "send_message",
        "task_id": "", "is_progress": True, "chat_id": 25,
        "content": "already running", "lifecycle_pointer": pointer,
    })

    rows, _quota = _collect_progress_rows(
        progress, tmp_path / "archive", 10, lambda _chat, _entry: True,
    )
    assert len(rows) == 1
    assert rows[0]["task_id"] == ""
    assert rows[0]["lifecycle_pointer"] == pointer


def test_plan_reference_replays_from_progress_without_visible_text(tmp_path):
    progress = tmp_path / "logs" / "progress.jsonl"
    stored = ds.append(tmp_path, "progress", {
        "ts": "2026-08-25T00:00:00Z", "type": "review_reference",
        "surface": "plan_review", "task_id": "task-1",
        "presentation_owner_task_id": "task-1", "review_fingerprint": "a" * 64,
        "state_revision": "b" * 64, "is_progress": True, "chat_id": 1,
        "content": "", "text": "",
    })

    rows, quota = _collect_progress_rows(
        progress, tmp_path / "archive", 10, lambda _chat, _entry: True,
    )
    assert quota == 0
    assert rows == [{
        "row_address": format_address(row_address(stored.row)),
        "text": "", "role": "assistant", "ts": "2026-08-25T00:00:00Z",
        "is_progress": True, "markdown": False, "task_id": "task-1",
        "system_type": "review_reference", "surface": "plan_review",
        "presentation_owner_task_id": "task-1", "review_fingerprint": "a" * 64,
        "state_revision": "b" * 64,
    }]


def test_grouped_skill_attempts_preserve_legacy_text_initiator_and_supersession():
    base = {
        "is_progress": False, "system_type": "skill_review", "skill": "alpha",
        "group_id": "task:root:alpha", "presentation_owner_task_id": "root",
        "root_task_id": "root", "status": "clean",
    }
    folded = _fold_task_bound_skill_reviews([
        {
            **base, "ts": "2026-08-25T00:00:00Z", "job_id": "",
            "task_id": "child-a", "origin_task_id": "child-a",
            "text": "legacy jobless terminal review text",
            "executions": [{"kind": "api", "model": "m1", "cost": 99}],
        },
        {
            **base, "ts": "2026-08-25T00:00:01Z", "job_id": "job-2",
            "task_id": "child-b", "origin_task_id": "child-b",
            "text": "new terminal review text",
            "executions": [{"kind": "harness", "harness_id": "codex", "model": "m2"}],
        },
    ])

    assert len(folded) == 1
    attempts = folded[0]["review_group"]["attempts"]
    assert [attempt["origin_task_id"] for attempt in attempts] == ["child-a", "child-b"]
    assert attempts[0]["text"] == "legacy jobless terminal review text"
    assert [attempt["superseded"] for attempt in attempts] == [True, False]
    assert attempts[0]["executions"] == [{"kind": "api", "model": "m1"}]


def test_grouped_skill_attempts_coerce_malformed_ordinals_without_raising():
    base = {
        "is_progress": False, "system_type": "skill_review", "skill": "alpha",
        "group_id": "task:root:alpha", "presentation_owner_task_id": "root",
    }
    folded = _fold_task_bound_skill_reviews([{
        **base,
        "ts": "2026-08-25T00:00:00Z",
        "job_id": "job-malformed",
        "review_round": "not-a-number",
        "snapshot_attempt": "not-a-number",
    }])

    attempt = folded[0]["review_group"]["attempts"][0]
    assert attempt["review_round"] == 0
    assert attempt["snapshot_attempt"] == 0
