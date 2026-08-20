"""The enabled flag and the review state on disk, and the directory that holds them.

Split out of ``tests/test_skill_loader.py`` by theme: the enabled round trip and the corrupt
state it fails closed on, the review-state round trip, the live aggregation of soft findings,
the invalid numeric fields and non-UTF-8 file it refuses, the unknown status clamped to
pending, and the state directory that resists a path escape.
"""

from __future__ import annotations

import json

import pytest

from ouroboros.skill_loader import (
    SkillReviewState,
    load_enabled,
    load_review_state,
    save_enabled,
    save_review_state,
    skill_state_dir,
)


# ---------------------------------------------------------------------------
# State persistence
# ---------------------------------------------------------------------------
def test_enabled_round_trip(tmp_path):
    drive_root = tmp_path / "drive"
    drive_root.mkdir()
    assert load_enabled(drive_root, "x") is False
    save_enabled(drive_root, "x", True)
    assert load_enabled(drive_root, "x") is True
    save_enabled(drive_root, "x", False)
    assert load_enabled(drive_root, "x") is False


@pytest.mark.parametrize("payload,write_bytes", [
    (json.dumps({"enabled": "false"}).encode("utf-8"), None),  # non-boolean value
    (b"{\"enabled\": \xff}", None),                            # non-UTF-8 bytes
])
def test_load_enabled_fails_closed_on_corrupt_state(payload, write_bytes, tmp_path):
    """load_enabled must default to False on any corrupt state file.

    Parametrized in v5.15.x from test_load_enabled_fails_closed_on_non_boolean_payload
    + test_load_enabled_fails_closed_on_non_utf8_state_file."""
    drive_root = tmp_path / "drive"
    drive_root.mkdir()
    raw_path = skill_state_dir(drive_root, "x") / "enabled.json"
    raw_path.write_bytes(payload)
    assert load_enabled(drive_root, "x") is False


def test_review_state_round_trip(tmp_path):
    drive_root = tmp_path / "drive"
    drive_root.mkdir()
    # Default when no file on disk.
    assert load_review_state(drive_root, "x").status == "pending"
    state = SkillReviewState(
        status="pass",
        content_hash="abcd",
        findings=[{"item": "manifest_schema", "verdict": "PASS", "severity": "critical", "reason": "ok"}],
        reviewer_models=["openai/gpt-5.5"],
        timestamp="2026-04-21T00:00:00+00:00",
        prompt_chars=1234,
        cost_usd=0.5,
        raw_actor_records=[{"model_id": "openai/gpt-5.5", "raw_text": "full"}],
    )
    save_review_state(drive_root, "x", state)
    reloaded = load_review_state(drive_root, "x")
    assert reloaded.status == "clean"
    assert reloaded.content_hash == "abcd"
    assert reloaded.reviewer_models == ["openai/gpt-5.5"]
    assert reloaded.prompt_chars == 1234
    assert reloaded.raw_actor_records == [{"model_id": "openai/gpt-5.5", "raw_text": "full"}]

    raw = json.loads((skill_state_dir(drive_root, "x") / "review.json").read_text(encoding="utf-8"))
    assert "status" not in raw


def test_load_review_state_live_aggregates_soft_findings(tmp_path, monkeypatch):
    drive_root = tmp_path / "drive"
    drive_root.mkdir()
    state = SkillReviewState(
        status="advisory",
        content_hash="abcd",
        findings=[{
            "item": "timeout_and_output_discipline",
            "verdict": "FAIL",
            "severity": "advisory",
            "reason": "soft",
        }],
    )
    save_review_state(drive_root, "x", state)

    monkeypatch.setenv("OUROBOROS_REVIEW_ENFORCEMENT", "blocking")
    assert load_review_state(drive_root, "x", skill_type="script").status == "warnings"

    monkeypatch.setenv("OUROBOROS_REVIEW_ENFORCEMENT", "advisory")
    assert load_review_state(drive_root, "x", skill_type="script").status == "warnings"


def test_load_review_state_fails_closed_on_invalid_numeric_fields(tmp_path):
    drive_root = tmp_path / "drive"
    drive_root.mkdir()
    raw_path = skill_state_dir(drive_root, "x") / "review.json"
    raw_path.write_text(
        json.dumps(
            {
                "status": "pass",
                "content_hash": "abcd",
                "prompt_chars": "not-an-int",
                "cost_usd": "not-a-float",
            }
        ),
        encoding="utf-8",
    )
    reloaded = load_review_state(drive_root, "x")
    assert reloaded.status == "clean"
    assert reloaded.prompt_chars == 0
    assert reloaded.cost_usd == 0.0


def test_load_review_state_fails_closed_on_non_utf8_state_file(tmp_path):
    drive_root = tmp_path / "drive"
    drive_root.mkdir()
    raw_path = skill_state_dir(drive_root, "x") / "review.json"
    raw_path.write_bytes(b"{\"status\": \"pass\", \"content_hash\": \xff}")
    reloaded = load_review_state(drive_root, "x")
    assert reloaded.status == "pending"
    assert reloaded.content_hash == ""


def test_review_state_unknown_status_clamped_to_pending(tmp_path):
    drive_root = tmp_path / "drive"
    drive_root.mkdir()
    raw_path = skill_state_dir(drive_root, "x") / "review.json"
    raw_path.write_text(
        json.dumps({"status": "TURBO", "content_hash": "abcd"}),
        encoding="utf-8",
    )
    reloaded = load_review_state(drive_root, "x")
    assert reloaded.status == "pending"
    assert reloaded.content_hash == "abcd"


# ---------------------------------------------------------------------------
# Safety: skill name sanitization
# ---------------------------------------------------------------------------
def test_skill_state_dir_resists_path_escape(tmp_path):
    """A malicious manifest ``name: ../../etc`` cannot escape the state root."""
    drive_root = tmp_path / "drive"
    drive_root.mkdir()
    malicious = "../../etc/passwd"
    state_path = skill_state_dir(drive_root, malicious)
    resolved = state_path.resolve()
    state_root_resolved = (drive_root / "state" / "skills").resolve()
    # The returned path must stay under data/state/skills/.
    assert resolved.is_relative_to(state_root_resolved)
