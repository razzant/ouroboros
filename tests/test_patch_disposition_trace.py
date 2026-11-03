"""D-trace (owner 4=A): patch apply/reject decisions are attested, not gated.

``integrate_delegated_patch`` applies a child's diff on five mechanical
manifest fields; no review facts exist anywhere on the path (the child is
forbidden from reviewing its own diff, and no host review of the captured
bytes runs). Rather than inventing a review, every verdict lands as a typed
``delegate_run_patch_verdict`` custody row beside its verdict artifact.
Nothing refuses an apply.
"""

import json

from ouroboros import delegate_custody as custody


def test_write_verdict_emits_the_custody_row(tmp_path):
    from types import SimpleNamespace

    from ouroboros.tools.subagent_integration import _write_verdict

    ctx = SimpleNamespace(drive_root=str(tmp_path), task_id="t-parent",
                          task_metadata={}, budget_drive_root=str(tmp_path))
    path = _write_verdict(
        ctx, "run_r9", outcome="applied", reason="ok", files=["a.py"],
        manifest={"sha256": "deadbeef", "diffstat": "1 file"},
        applied=True, conflicts=[], protected=[],
    )
    assert path, "verdict artifact write should succeed in a temp drive"
    rows = [
        row for row in custody._iter_rows(custody.event_log_path(tmp_path))
        if row.get("type") == "delegate_run_patch_verdict"
    ]
    assert len(rows) == 1
    row = rows[0]
    assert row["child_task_id"] == "run_r9"
    assert row["pipeline"] == "delegated"
    assert row["patch_sha256"] == "deadbeef"
    assert row["applied"] is True
    assert row["verdict_artifact_write_failed"] is False


def test_verdict_artifact_content_still_written(tmp_path):
    from types import SimpleNamespace

    from ouroboros.tools.subagent_integration import _write_verdict

    ctx = SimpleNamespace(drive_root=str(tmp_path), task_id="t-parent",
                          task_metadata={}, budget_drive_root=str(tmp_path))
    path = _write_verdict(
        ctx, "child-7", outcome="rejected", reason="conflicts", files=[],
        manifest={}, applied=False, conflicts=["x"], protected=[],
    )
    data = json.loads(open(path).read())
    assert data["outcome"] == "rejected"
    assert data["child_task_id"] == "child-7"
