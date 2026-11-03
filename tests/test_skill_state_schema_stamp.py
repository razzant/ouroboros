"""CPL4-C10 pins: per-skill owner-state files are ABI-2-stamped on write.

review.json / enabled.json / grants.json / review_job.json /
owner_attestation.json all land with
``_schema_version: 1``; readers keep legacy-0 tolerance (an unstamped
pre-upgrade file parses exactly as before and is never rewritten on read).
"""

from __future__ import annotations

import json

from ouroboros.contracts.schema_versions import SCHEMA_VERSION_KEY
from ouroboros.skill_loader import (
    SKILL_OWNER_STATE_SCHEMA_VERSION,
    SkillReviewState,
    load_enabled,
    load_review_state,
    load_skill_grants,
    save_enabled,
    save_review_state,
    save_skill_grants,
    skill_state_dir,
)


def _read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_core_state_writers_stamp_schema_version(tmp_path):
    save_enabled(tmp_path, "s", True)
    save_review_state(tmp_path, "s", SkillReviewState(status="pending"))
    save_skill_grants(tmp_path, "s", ["OPENAI_API_KEY"], content_hash="h1",
                      requested_keys=["OPENAI_API_KEY"])

    state = skill_state_dir(tmp_path, "s")
    for name in ("enabled.json", "review.json", "grants.json"):
        assert _read(state / name)[SCHEMA_VERSION_KEY] == SKILL_OWNER_STATE_SCHEMA_VERSION, name

    # The stamp does not leak into the normalized read projections.
    assert load_enabled(tmp_path, "s") is True
    assert SCHEMA_VERSION_KEY not in load_skill_grants(tmp_path, "s")


def test_legacy_unstamped_files_still_parse(tmp_path):
    state = skill_state_dir(tmp_path, "legacy")
    (state / "enabled.json").write_text(json.dumps({"enabled": True}), encoding="utf-8")
    (state / "review.json").write_text(
        json.dumps({"status": "pending", "content_hash": "h"}), encoding="utf-8",
    )
    (state / "grants.json").write_text(
        json.dumps({"granted_keys": ["K"], "requested_keys": ["K"], "content_hash": "h"}),
        encoding="utf-8",
    )

    assert load_enabled(tmp_path, "legacy") is True
    assert load_review_state(tmp_path, "legacy").status == "pending"
    assert load_skill_grants(tmp_path, "legacy")["granted_keys"] == ["K"]
    # Reads never retrofit the stamp onto the durable file.
    assert SCHEMA_VERSION_KEY not in _read(state / "enabled.json")


def test_review_job_writes_route_through_the_stamped_seam(tmp_path):
    import inspect

    import ouroboros.skill_review_runner as runner

    runner._write_review_job(
        runner.review_job_state_path(tmp_path, "s"), {"status": "running", "job_id": "j1"},
    )
    on_disk = _read(runner.review_job_state_path(tmp_path, "s"))
    assert on_disk[SCHEMA_VERSION_KEY] == SKILL_OWNER_STATE_SCHEMA_VERSION
    assert on_disk["status"] == "running"

    # The heartbeat is a locked merge so it cannot erase a reserved roster.
    runner._patch_review_job(tmp_path, "s", expected_job_id="j1", heartbeat_at="now")
    merged = _read(runner.review_job_state_path(tmp_path, "s"))
    assert merged[SCHEMA_VERSION_KEY] == SKILL_OWNER_STATE_SCHEMA_VERSION
    assert merged["job_id"] == "j1" and merged["heartbeat_at"] == "now"
    runner._patch_review_job(tmp_path, "s", expected_job_id="old", status="failed")
    assert _read(runner.review_job_state_path(tmp_path, "s")) == merged
    # Whole writers keep the stamped seam, the merge stamps before its locked write.
    src = inspect.getsource(runner)
    assert src.count("_write_review_job(") >= 2  # def + the start's whole write (under the lifecycle lock)
    # The heartbeat and the lifecycle's stale/timeout/finish writers are locked, stamped merges,
    # so a seat's pre-POST invocation checkpoint that landed meanwhile survives them.
    for merger in (runner._patch_review_job, runner.mark_stale_review_job_interrupted,
                   runner._mark_review_job_timeout, runner._on_finished):
        merge = inspect.getsource(merger)
        assert "update_json_locked(" in merge and "with_schema_version(" in merge, merger.__name__
    assert "atomic_write_json(review_job_state_path" not in src


def test_owner_attestation_marker_is_stamped(tmp_path):
    import inspect

    import ouroboros.skill_owner_attestation as attestation

    src = inspect.getsource(attestation.run_owner_attestation)
    assert "with_schema_version" in src and "owner_attestation.json" in src
