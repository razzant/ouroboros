"""S4 skill-review cards fix (Q15, design b): reference rows + lazy detail route.

Producer stays untouched: since a776639f every ``skill_review`` chat row already
carries the exact-job reference (skill/job_id/content_hash/...). These tests pin
the two READ-side halves added here:

- ``gateway/history.py`` passes the reference fields through on
  ``system_type == "skill_review"`` rows (legacy rows without them unchanged);
- ``GET /api/skills/{skill}/review-history/{job_id}`` serves the server-rendered
  normalized review block for the exact record, with typed 404s (never a 500)
  for missing skill / missing job / malformed history, and raw reviewer text
  kept private (degraded reviewers disclosed by model + status only).
"""
from __future__ import annotations

import json
import pathlib

import pytest
from starlette.applications import Starlette
from starlette.routing import Route
from starlette.testclient import TestClient

from ouroboros.utils import append_jsonl


def _detail_client(tmp_path: pathlib.Path) -> tuple[TestClient, pathlib.Path]:
    from ouroboros.gateway.extensions import api_skill_review_history_detail

    app = Starlette(routes=[
        Route(
            "/api/skills/{skill}/review-history/{job_id}",
            endpoint=api_skill_review_history_detail,
            methods=["GET"],
        ),
    ])
    drive_root = tmp_path / "drive"
    drive_root.mkdir(exist_ok=True)
    app.state.drive_root = drive_root
    from ouroboros.task_results import write_task_result
    write_task_result(drive_root, 'root-alpha', 'running', billing_group={
        'billing_group_id': 'root-alpha', 'billing_group_limit_usd': None,
        'billing_group_limit_source': 'initial_task_admission', 'billing_group_limit_revision': 'fixture'})
    return TestClient(app), drive_root


def _history_path(drive_root: pathlib.Path, skill: str) -> pathlib.Path:
    path = drive_root / "state" / "skills" / skill / "review_history.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _terminal_record(**overrides) -> dict:
    record = {
        "ts": "2026-08-15T00:00:00+00:00",
        "status": "clean",
        "job_status": "succeeded",
        "terminal_reason": "succeeded",
        "content_hash": "abc123def4567890",
        "failure_signature": [],
        "fail_findings": [],
        "job_id": "skill-job-1",
        "group_id": "manual:alpha",
        "review_round": 2,
        "snapshot_attempt": 1,
        "snapshot_revised": False,
        "source": "skills",
        "raw_actor_records": [
            {
                "slot": 0,
                "slot_id": "skill-triad-1",
                "model_id": "openai/gpt-x",
                "status": "responded",
                "raw_text": "RAW-REVIEWER-TEXT-MUST-STAY-PRIVATE",
                "parsed_items": [
                    {
                        "item": "manifest_schema",
                        "verdict": "PASS",
                        "severity": "advisory",
                        "reason": "manifest ok",
                        "model": "openai/gpt-x",
                    },
                    {
                        "item": "secrets_hygiene",
                        "verdict": "FAIL",
                        "severity": "major",
                        "reason": "token appears in plugin.py",
                        "model": "openai/gpt-x",
                    },
                ],
            },
            {
                "slot": 1,
                "slot_id": "skill-triad-2",
                "model_id": "anthropic/claude-y",
                "status": "empty_response",
                "raw_text": "DEGRADED-RAW-BODY-MUST-STAY-PRIVATE",
                "parsed_items": [],
            },
        ],
    }
    record.update(overrides)
    return record


def test_review_history_detail_found_record_renders_private_markdown(tmp_path):
    client, drive_root = _detail_client(tmp_path)
    append_jsonl(_history_path(drive_root, "alpha"), _terminal_record())

    resp = client.get("/api/skills/alpha/review-history/skill-job-1")

    assert resp.status_code == 200
    payload = resp.json()
    assert set(payload) == {"markdown", "status", "content_hash", "job_status"}
    assert payload["status"] == "clean"
    assert payload["content_hash"] == "abc123def4567890"
    assert payload["job_status"] == "succeeded"
    markdown = payload["markdown"]
    # Normalized verdict rows from parsed_items are rendered per reviewer.
    assert "Skill review round 2 — snapshot abc123def456 (attempt 1)" in markdown
    assert "`alpha` — status=clean" in markdown
    assert "### Reviewer: openai/gpt-x" in markdown
    assert "[PASS] manifest_schema: manifest ok" in markdown
    assert "[FAIL major] secrets_hygiene: token appears in plugin.py" in markdown
    # Raw reviewer bodies never leave the history file.
    assert "RAW-REVIEWER-TEXT-MUST-STAY-PRIVATE" not in markdown
    assert "DEGRADED-RAW-BODY-MUST-STAY-PRIVATE" not in markdown
    # Degraded reviewers are disclosed honestly by model + status.
    assert "anthropic/claude-y (empty_response)" in markdown
    assert "withheld from chat" in markdown
    assert (
        "A terminal reviewer slot that never started or refused has no "
        "physical-attempt ledger row; incomplete attempt coverage can therefore be final."
    ) in markdown


def test_review_history_detail_surfaces_failed_lifecycle_with_clean_verdict(tmp_path):
    client, drive_root = _detail_client(tmp_path)
    append_jsonl(_history_path(drive_root, "alpha"), _terminal_record(
        status="clean",
        job_status="failed",
        terminal_reason="dependency install failed",
    ))

    markdown = client.get(
        "/api/skills/alpha/review-history/skill-job-1",
    ).json()["markdown"]

    assert "`alpha` — status=clean" in markdown
    assert "Error: dependency install failed" in markdown


def test_review_history_detail_keeps_legacy_same_model_actors_distinct(tmp_path):
    client, drive_root = _detail_client(tmp_path)
    actors = []
    for item in ("manifest_schema", "secrets_hygiene"):
        actors.append({
            "model_id": "openai/gpt-x",
            "status": "responded",
            "parsed_items": [{
                "item": item,
                "verdict": "PASS",
                "severity": "advisory",
                "reason": "ok",
                "model": "openai/gpt-x",
            }],
        })
    append_jsonl(
        _history_path(drive_root, "alpha"),
        _terminal_record(raw_actor_records=actors),
    )

    markdown = client.get(
        "/api/skills/alpha/review-history/skill-job-1",
    ).json()["markdown"]

    assert (
        "Reviewers: openai/gpt-x [legacy-actor-1], "
        "openai/gpt-x [legacy-actor-2]"
    ) in markdown
    assert "### Reviewer: openai/gpt-x [legacy-actor-1]" in markdown
    assert "### Reviewer: openai/gpt-x [legacy-actor-2]" in markdown
    assert markdown.count("### Reviewer:") == 2


def test_review_history_detail_discloses_legacy_paid_wave_as_unattributable(tmp_path):
    client, drive_root = _detail_client(tmp_path)
    append_jsonl(_history_path(drive_root, "alpha"), _terminal_record(paid=True))

    markdown = client.get(
        "/api/skills/alpha/review-history/skill-job-1",
    ).json()["markdown"]

    assert "paid panel dispatch (counted toward Max Review Cycles by that version)" in markdown
    assert "### Review accounting" not in markdown
    assert "Cost unavailable (a row of an earlier version)" in markdown


def test_review_history_detail_of_a_legacy_paid_row_keeps_its_verdict_without_usage_rows(tmp_path):
    """A row an earlier version wrote with physical-attempt attribution renders its verdict; no usage
    projection is consulted any more (a skill wave's seats are its ledger record's rows)."""
    client, drive_root = _detail_client(tmp_path)
    append_jsonl(_history_path(drive_root, "alpha"), _terminal_record(
        paid=True, wave_id="skill-job-1",
        usage_attribution_schema="physical_attempt_v1",
    ))

    resp = client.get("/api/skills/alpha/review-history/skill-job-1")

    assert resp.status_code == 200
    assert "status=clean" in resp.json()["markdown"]
    assert "Cost unavailable (a row of an earlier version)" in resp.json()["markdown"]


def test_review_history_detail_free_replay_names_zero_physical_dispatch(tmp_path):
    client, drive_root = _detail_client(tmp_path)
    append_jsonl(_history_path(drive_root, "alpha"), _terminal_record(
        paid=True, usage_attribution_schema="physical_attempt_v1",
        replayed_from_ts="2026-08-14T12:00:00Z",
    ))

    markdown = client.get(
        "/api/skills/alpha/review-history/skill-job-1",
    ).json()["markdown"]

    assert "free replay of the 2026-08-14T12:00:00Z verdict" in markdown
    assert "no physical reviewer dispatch for this replay" in markdown
    assert "### Review accounting" not in markdown
    assert "Cost: $0 (free replay)" in markdown


def test_review_history_detail_missing_skill_is_typed_404(tmp_path):
    client, _drive_root = _detail_client(tmp_path)

    resp = client.get("/api/skills/ghost/review-history/skill-job-1")

    assert resp.status_code == 404
    assert resp.json()["error"] == "no review history for skill"


def test_review_history_detail_missing_job_is_typed_404(tmp_path):
    client, drive_root = _detail_client(tmp_path)
    append_jsonl(_history_path(drive_root, "alpha"), _terminal_record())

    resp = client.get("/api/skills/alpha/review-history/skill-job-unknown")

    assert resp.status_code == 404
    assert resp.json()["error"] == "review record not found"


def test_review_history_detail_outside_fixed_tail_is_honestly_unavailable(
    tmp_path, monkeypatch,
):
    from ouroboros import skill_review_history

    client, drive_root = _detail_client(tmp_path)
    path = _history_path(drive_root, "alpha")
    append_jsonl(path, _terminal_record(job_id="skill-job-old"))
    append_jsonl(
        path,
        _terminal_record(job_id="skill-job-new", raw_actor_records=[], padding="x" * 600),
    )
    monkeypatch.setattr(skill_review_history, "_DETAIL_LOOKUP_MAX_BYTES", 256)

    resp = client.get("/api/skills/alpha/review-history/skill-job-old")

    assert resp.status_code == 404
    assert resp.json()["error"] == (
        "review record unavailable outside the bounded history window"
    )


def test_review_history_detail_normalizes_legacy_ordinals_only_with_full_history(tmp_path):
    client, drive_root = _detail_client(tmp_path)
    path = _history_path(drive_root, "alpha")
    for job_id in ("skill-job-first", "skill-job-legacy"):
        record = _terminal_record(job_id=job_id, raw_actor_records=[])
        for key in ("review_round", "snapshot_attempt", "snapshot_revised"):
            record.pop(key)
        append_jsonl(path, record)

    resp = client.get("/api/skills/alpha/review-history/skill-job-legacy")

    assert resp.status_code == 200
    assert "Skill review round 2" in resp.json()["markdown"]
    assert "(attempt 2)" in resp.json()["markdown"]


def test_review_history_detail_refuses_legacy_ordinals_from_truncated_byte_window(
    tmp_path, monkeypatch,
):
    from ouroboros import skill_review_history

    client, drive_root = _detail_client(tmp_path)
    path = _history_path(drive_root, "alpha")
    append_jsonl(path, _terminal_record(job_id="old", raw_actor_records=[], padding="x" * 3000))
    legacy = _terminal_record(job_id="skill-job-legacy", raw_actor_records=[])
    for key in ("review_round", "snapshot_attempt", "snapshot_revised"):
        legacy.pop(key)
    append_jsonl(path, legacy)
    monkeypatch.setattr(skill_review_history, "_DETAIL_LOOKUP_MAX_BYTES", 2048)

    resp = client.get("/api/skills/alpha/review-history/skill-job-legacy")

    assert resp.status_code == 404
    assert resp.json()["error"] == (
        "review record unavailable outside the bounded history window"
    )


def test_review_history_detail_reads_stored_ordinals_from_truncated_windows(
    tmp_path, monkeypatch,
):
    from ouroboros import skill_review_history

    client, drive_root = _detail_client(tmp_path)
    path = _history_path(drive_root, "alpha")
    append_jsonl(path, _terminal_record(job_id="old", raw_actor_records=[], padding="x" * 3000))
    append_jsonl(path, _terminal_record(
        job_id="skill-job-modern", raw_actor_records=[],
        review_round=77, snapshot_attempt=33, snapshot_revised=True,
    ))
    monkeypatch.setattr(skill_review_history, "_DETAIL_LOOKUP_MAX_BYTES", 2048)
    monkeypatch.setattr(skill_review_history, "_DETAIL_LOOKUP_MAX_RECORDS", 1)

    resp = client.get("/api/skills/alpha/review-history/skill-job-modern")

    assert resp.status_code == 200
    assert "Skill review round 77" in resp.json()["markdown"]
    assert "(attempt 33)" in resp.json()["markdown"]


def test_review_history_detail_rejects_malformed_ordinals_from_truncated_window(
    tmp_path, monkeypatch,
):
    from ouroboros import skill_review_history

    client, drive_root = _detail_client(tmp_path)
    path = _history_path(drive_root, "alpha")
    append_jsonl(path, _terminal_record(job_id="old", raw_actor_records=[], padding="x" * 3000))
    append_jsonl(path, _terminal_record(
        job_id="skill-job-malformed", raw_actor_records=[],
        review_round="not-a-number", snapshot_attempt="not-a-number",
    ))
    monkeypatch.setattr(skill_review_history, "_DETAIL_LOOKUP_MAX_BYTES", 2048)
    monkeypatch.setattr(skill_review_history, "_DETAIL_LOOKUP_MAX_RECORDS", 1)

    resp = client.get("/api/skills/alpha/review-history/skill-job-malformed")

    assert resp.status_code == 404
    assert resp.json()["error"] == (
        "review record unavailable outside the bounded history window"
    )


@pytest.mark.parametrize("stored_ordinals", [False, True])
def test_bounded_detail_record_window_never_invents_legacy_ordinals(
    tmp_path, monkeypatch, stored_ordinals,
):
    from ouroboros import skill_review_history

    _client, drive_root = _detail_client(tmp_path)
    path = _history_path(drive_root, "alpha")
    append_jsonl(path, _terminal_record(job_id="old", raw_actor_records=[]))
    target = _terminal_record(job_id="target", raw_actor_records=[])
    if not stored_ordinals:
        target.pop("snapshot_revised")
    append_jsonl(path, target)
    monkeypatch.setattr(skill_review_history, "_DETAIL_LOOKUP_MAX_RECORDS", 1)

    record, status = skill_review_history.find_history_job_bounded(
        drive_root, "alpha", "target",
    )

    if stored_ordinals:
        assert status == "found" and record is not None
    else:
        assert (record, status) == (None, "unavailable")


def test_bounded_detail_refuses_legacy_ordinals_across_malformed_raw_gap(
    tmp_path, monkeypatch,
):
    from ouroboros import skill_review_history

    _client, drive_root = _detail_client(tmp_path)
    path = _history_path(drive_root, "alpha")

    def legacy(job_id):
        record = _terminal_record(job_id=job_id, raw_actor_records=[])
        for key in ("review_round", "snapshot_attempt", "snapshot_revised"):
            record.pop(key)
        return json.dumps(record)

    path.write_text(
        "\n".join([
            legacy("skill-job-first"),
            "{malformed",
            legacy("skill-job-second"),
            legacy("skill-job-target"),
        ]) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(skill_review_history, "_DETAIL_LOOKUP_MAX_RECORDS", 3)

    record, status = skill_review_history.find_history_job_bounded(
        drive_root, "alpha", "skill-job-target",
    )

    assert record is None
    assert status == "unavailable"


def test_bounded_detail_lookup_uses_shared_jsonl_tail_reader(tmp_path, monkeypatch):
    from ouroboros import skill_review_history

    _client, drive_root = _detail_client(tmp_path)
    append_jsonl(
        _history_path(drive_root, "alpha"),
        _terminal_record(job_id="skill-job-modern", raw_actor_records=[]),
    )
    original = skill_review_history.iter_jsonl_objects
    calls = []

    def recording_reader(path, **kwargs):
        calls.append((path, kwargs))
        return original(path, **kwargs)

    monkeypatch.setattr(skill_review_history, "iter_jsonl_objects", recording_reader)

    record, status = skill_review_history.find_history_job_bounded(
        drive_root, "alpha", "skill-job-modern",
    )

    assert status == "found" and record is not None
    assert len(calls) == 1
    assert calls[0][1]["tail_bytes"] == skill_review_history._DETAIL_LOOKUP_MAX_BYTES
    assert calls[0][1]["max_entries"] == (
        skill_review_history._DETAIL_LOOKUP_MAX_RECORDS + 1
    )
    assert isinstance(calls[0][1]["gap_reasons"], set)


def test_bounded_detail_lookup_missing_and_empty_history_are_absent(tmp_path):
    from ouroboros import skill_review_history

    _client, drive_root = _detail_client(tmp_path)
    missing = skill_review_history.find_history_job_bounded(
        drive_root, "alpha", "missing",
    )
    _history_path(drive_root, "alpha").write_text("", encoding="utf-8")
    empty = skill_review_history.find_history_job_bounded(
        drive_root, "alpha", "missing",
    )

    assert missing == (None, "absent")
    assert empty == (None, "absent")


def test_bounded_detail_lookup_stat_permission_error_is_retryable_io_error(tmp_path, monkeypatch):
    from ouroboros import skill_review_history

    _client, drive_root = _detail_client(tmp_path)
    path = _history_path(drive_root, "alpha")
    path.write_text("{}\n", encoding="utf-8")
    original_stat = pathlib.Path.stat

    def guarded_stat(candidate, *args, **kwargs):
        if candidate == path:
            raise PermissionError("denied")
        return original_stat(candidate, *args, **kwargs)

    monkeypatch.setattr(pathlib.Path, "stat", guarded_stat)
    assert skill_review_history.find_history_job_bounded(
        drive_root, "alpha", "missing",
    ) == (None, "io_error")


@pytest.mark.parametrize(
    ("error", "expected"),
    [(FileNotFoundError("gone"), "absent"), (PermissionError("denied"), "io_error")],
)
def test_bounded_detail_lookup_read_errors_are_typed(tmp_path, monkeypatch, error, expected):
    from ouroboros import skill_review_history

    _client, drive_root = _detail_client(tmp_path)
    _history_path(drive_root, "alpha").write_text("{}\n", encoding="utf-8")

    def fail_read(*args, **kwargs):
        raise error

    monkeypatch.setattr(skill_review_history, "iter_jsonl_objects", fail_read)
    assert skill_review_history.find_history_job_bounded(
        drive_root, "alpha", "missing",
    ) == (None, expected)


def test_review_history_detail_io_error_is_retryable_503(tmp_path, monkeypatch):
    from ouroboros import skill_review_history

    client, _drive_root = _detail_client(tmp_path)
    monkeypatch.setattr(
        skill_review_history,
        "find_history_job_bounded",
        lambda *_args, **_kwargs: (None, "io_error"),
    )

    resp = client.get("/api/skills/alpha/review-history/skill-job-1")

    assert resp.status_code == 503
    assert resp.json()["error"] == (
        "review history is temporarily unavailable; retry the detail"
    )


def test_review_history_detail_malformed_lines_degrade_to_typed_404(tmp_path):
    client, drive_root = _detail_client(tmp_path)
    path = _history_path(drive_root, "alpha")
    path.write_text("{not valid json\n", encoding="utf-8")

    resp = client.get("/api/skills/alpha/review-history/skill-job-1")

    assert resp.status_code == 404
    assert "error" in resp.json()

    # A malformed line does NOT poison neighbouring valid records.
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_terminal_record()) + "\n")
    resp2 = client.get("/api/skills/alpha/review-history/skill-job-1")
    assert resp2.status_code == 200
    assert "status=clean" in resp2.json()["markdown"]


def test_review_history_detail_interrupted_record_renders_honest_minimum(tmp_path):
    client, drive_root = _detail_client(tmp_path)
    append_jsonl(
        _history_path(drive_root, "alpha"),
        _terminal_record(
            job_id="skill-job-int",
            status="interrupted",
            job_status="interrupted",
            terminal_reason="owner_process_exited",
            raw_actor_records=[],
        ),
    )

    resp = client.get("/api/skills/alpha/review-history/skill-job-int")

    assert resp.status_code == 200
    payload = resp.json()
    assert payload["status"] == "interrupted"
    assert payload["job_status"] == "interrupted"
    assert "status=interrupted" in payload["markdown"]
    assert "Error: owner_process_exited" in payload["markdown"]
    assert "no parsed findings" in payload["markdown"]


def test_review_history_detail_rejects_non_canonical_skill_names(tmp_path):
    client, _drive_root = _detail_client(tmp_path)

    resp = client.get("/api/skills/../review-history/skill-job-1")

    assert resp.status_code == 404


def test_chat_history_passes_skill_review_reference_fields(tmp_path):
    from ouroboros.gateway.history import _collect_chat_rows

    from ouroboros import dialogue_store as ds

    chat_path = tmp_path / "logs" / "chat.jsonl"
    archive_dir = tmp_path / "archive"
    ds.append(tmp_path, "chat", {
        "ts": "2026-08-15T00:00:00+00:00",
        "direction": "system",
        "type": "skill_review",
        "chat_id": 1,
        "skill": "alpha",
        "status": "clean",
        "content_hash": "abc123def4567890",
        "job_id": "skill-job-1",
        "record_id": "rec-20260815-0001",
        "group_id": "task:root-1:alpha",
        "presentation_owner_task_id": "root-1",
        "root_task_id": "root-1",
        "origin_task_id": "child-1",
        "origin_root_task_id": "root-1",
        "review_round": 2,
        "snapshot_attempt": 1,
        "snapshot_revised": True,
        "job_status": "succeeded",
        "terminal_reason": "clean",
        "source": "skills",
        "format": "markdown",
        "text": "Skill review round 2 — snapshot abc123def456 (attempt 1): `alpha` — status=clean, source=skills",
    })
    ds.append(tmp_path, "chat", {
        "ts": "2026-08-15T00:00:01+00:00",
        "direction": "system",
        "type": "skill_review",
        "chat_id": 1,
        "format": "markdown",
        "text": "legacy full-text review row without reference fields",
    })
    ds.append(tmp_path, "chat", {
        "ts": "2026-08-15T00:00:02+00:00",
        "direction": "out",
        "chat_id": 1,
        "text": "plain assistant row",
    })

    rows, _quota = _collect_chat_rows(
        chat_path, archive_dir, 10, lambda _chat, _entry: True, {},
    )

    assert len(rows) == 3
    reference, legacy, plain = rows
    assert reference["system_type"] == "skill_review"
    assert reference["skill"] == "alpha"
    assert reference["job_id"] == "skill-job-1"
    assert reference["record_id"] == "rec-20260815-0001", "the ledger record the card asks for (a reused review keeps it)"
    assert reference["status"] == "clean"
    assert reference["content_hash"] == "abc123def4567890"
    assert reference["review_round"] == 2
    assert reference["snapshot_attempt"] == 1
    assert reference["snapshot_revised"] is True
    assert reference["group_id"] == "task:root-1:alpha"
    assert reference["presentation_owner_task_id"] == "root-1"
    assert reference["origin_task_id"] == "child-1"
    assert reference["job_status"] == "succeeded"
    assert reference["source"] == "skills"
    # Legacy skill_review rows expose empty reference fields (frontend keeps
    # exactly today's local expansion for them).
    assert legacy["skill"] == "" and legacy["job_id"] == "" and legacy["record_id"] == ""
    # Non-skill-review rows do not grow the new keys at all.
    assert "skill" not in plain and "job_id" not in plain
