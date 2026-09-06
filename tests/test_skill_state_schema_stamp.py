"""CPL4-C10 pins: per-skill owner-state files are ABI-2-stamped on write.

review.json / enabled.json / grants.json / review_job.json /
owner_attestation.json / accepted_rebuttals.json all land with
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

    # Every review_job.json write site uses the seam — a direct atomic write
    # would silently un-stamp merge writers.
    src = inspect.getsource(runner)
    assert src.count("_write_review_job(") >= 6  # def + five call sites
    assert "atomic_write_json(review_job_state_path" not in src


def test_accepted_rebuttals_stamped_and_legacy_tolerant(tmp_path):
    from ouroboros.skill_review_cycles import (
        accepted_rebuttals_path,
        load_accepted_rebuttals,
        record_accepted_rebuttal,
    )

    record_accepted_rebuttal(
        tmp_path, "s", item="I1", rebuttal_text="because", content_hash="h1",
    )
    on_disk = _read(accepted_rebuttals_path(tmp_path, "s"))
    assert on_disk[SCHEMA_VERSION_KEY] == SKILL_OWNER_STATE_SCHEMA_VERSION
    assert [entry["item"] for entry in load_accepted_rebuttals(tmp_path, "s")] == ["I1"]

    legacy = accepted_rebuttals_path(tmp_path, "legacy")
    legacy.parent.mkdir(parents=True, exist_ok=True)
    legacy.write_text(json.dumps({"items": [{"item": "L"}]}), encoding="utf-8")
    assert [entry["item"] for entry in load_accepted_rebuttals(tmp_path, "legacy")] == ["L"]


def test_owner_attestation_marker_is_stamped(tmp_path):
    import inspect

    import ouroboros.skill_owner_attestation as attestation

    src = inspect.getsource(attestation.run_owner_attestation)
    assert "with_schema_version" in src and "owner_attestation.json" in src
