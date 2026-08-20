"""What ``get_task_result`` hands back, and the verification receipts it carries.

Split out of ``tests/test_task_status_flow.py`` by theme: the completed output a reader
sees, the bounded per-receipt rows, and the child-drive receipts that are published to
the canonical root, refreshed idempotently, and preferred over a stale parent row.
"""

from types import SimpleNamespace


def test_get_task_result_returns_full_completed_output(tmp_path):
    from ouroboros.task_results import STATUS_COMPLETED, write_task_result
    from ouroboros.tools.control import _get_task_result

    full_text = ("hello\n" * 1200) + "TAIL_MARKER"
    write_task_result(
        tmp_path,
        "abc123",
        STATUS_COMPLETED,
        result=full_text,
        cost_usd=1.23,
        trace_summary="trace",
    )

    ctx = SimpleNamespace(drive_root=tmp_path)
    output = _get_task_result(ctx, "abc123")

    assert "TAIL_MARKER" in output
    assert full_text in output
    assert "[SUBTASK_OUTCOME]" in output
    assert '"outcome_axes"' in output
    assert "[BEGIN_SUBTASK_OUTPUT]" in output


def test_get_task_result_carries_bounded_per_receipt_rows(tmp_path):
    """W2: the FULL single-child handoff (get_task_result/wait_task) shows WHICH
    checks passed as bounded identity rows — OUTSTANDING first, then newest, hard
    cap 10, exact omitted count — while the wait_tasks batch projection stays
    counts-compact.

    The bound must not be able to bury the fact the parent's absorption decision
    turns on: a child that failed a check early and then produced ten greens for
    OTHER criteria used to hand up an affirmatively all-green list."""
    import json as _json

    from ouroboros.outcomes import append_verification_receipt
    from ouroboros.task_results import STATUS_COMPLETED, write_task_result
    from ouroboros.tools.control import _get_task_result

    write_task_result(tmp_path, "abc123", STATUS_COMPLETED, result="done", cost_usd=0.1)
    for idx in range(12):
        append_verification_receipt(tmp_path, "abc123", {
            "status": "pass" if idx else "fail",
            "check": f"pytest tests/x{idx}.py",
            "criterion_id": f"claim_{idx}",
        })

    output = _get_task_result(SimpleNamespace(drive_root=tmp_path), "abc123")
    summary = _json.loads(
        output.split("[SUBTASK_OUTCOME]\n", 1)[1].split("\n[/SUBTASK_OUTCOME]", 1)[0]
    )

    rows = summary["verification_receipts"]
    assert len(rows) == 10                                   # hard cap
    assert summary["verification_receipts_omitted"] == 2     # disclosed, exact
    # The still-unreconciled RED is carried FIRST and says why, even though ten
    # newer greens exist — no green of another criterion clears it.
    assert rows[0]["criterion_id"] == "claim_0"
    assert rows[0]["status"] == "fail"
    assert rows[0]["outstanding"] == "unreconciled_failed"
    # ...the rest of the cap is the newest remaining receipts, and only the OLDEST
    # greens are the ones left out.
    assert [row["criterion_id"] for row in rows[1:]] == [
        f"claim_{idx}" for idx in range(11, 2, -1)
    ]
    assert all("outstanding" not in row for row in rows[1:])
    assert "check" in rows[0] and "reconciliation_identity" in rows[0]

    # A red that a LATER green for the same criterion reconciles is not carried:
    # the rule is the shared unreconciled-set SSOT, not "always float failures".
    write_task_result(tmp_path, "closed", STATUS_COMPLETED, result="done", cost_usd=0.1)
    append_verification_receipt(tmp_path, "closed", {
        "status": "fail", "check": "pytest tests/a.py", "criterion_id": "claim_a",
    })
    for idx in range(11):
        append_verification_receipt(tmp_path, "closed", {
            "status": "pass", "check": "pytest tests/a.py", "criterion_id": "claim_a"
            if idx == 0 else f"claim_b{idx}",
        })
    closed = _json.loads(
        _get_task_result(SimpleNamespace(drive_root=tmp_path), "closed")
        .split("[SUBTASK_OUTCOME]\n", 1)[1].split("\n[/SUBTASK_OUTCOME]", 1)[0]
    )
    assert all("outstanding" not in row for row in closed["verification_receipts"])
    assert closed["verification_receipts"][0]["criterion_id"] == "claim_b10"
    # No receipts -> no rows key at all (the wave1 zero-receipt shape stays visible
    # through the ledger counts, not an empty list).
    write_task_result(tmp_path, "noreceipts", STATUS_COMPLETED, result="done")
    bare = _get_task_result(SimpleNamespace(drive_root=tmp_path), "noreceipts")
    assert "verification_receipts_omitted" not in bare


def _receipt_rows_of(output):
    import json as _json

    summary = _json.loads(
        output.split("[SUBTASK_OUTCOME]\n", 1)[1].split("\n[/SUBTASK_OUTCOME]", 1)[0]
    )
    return summary.get("verification_receipts")


def test_child_finalization_publishes_receipts_to_canonical_root(tmp_path):
    """S3 seam (a): every real schedule_subagent child runs memory_mode forked|empty
    on an ISOLATED drive, so verify_and_record writes its receipts under the CHILD
    drive while the parent-side W2 reader resolves them against the canonical root.
    Child finalization (headless.copy_child_task_result) must publish
    verification_receipts.jsonl to the canonical root alongside the artifact rebase
    — WITHOUT any parent read in between (the opportunistic effective-read artifact
    sync must not be the only carrier: it dies with the child drive, which the
    cancel path and the startup prune both delete)."""
    from ouroboros.headless import copy_child_task_result, prepare_task_drive
    from ouroboros.outcomes import append_verification_receipt, read_verification_receipts
    from ouroboros.task_results import STATUS_COMPLETED, STATUS_SCHEDULED, write_task_result
    from ouroboros.tools.control import _get_task_result

    tid = "childsplit"
    child_drive = prepare_task_drive(tmp_path, tid, "forked")
    assert child_drive == tmp_path / "state" / "headless_tasks" / tid / "data"

    # Parent-side scheduled record (the shape schedule_subagent writes); the child
    # self-finalizes and records receipts ONLY on its isolated drive.
    write_task_result(
        tmp_path, tid, STATUS_SCHEDULED,
        drive_root=str(child_drive), child_drive_root=str(child_drive),
    )
    write_task_result(child_drive, tid, STATUS_COMPLETED, result="child split done", cost_usd=0.2)
    append_verification_receipt(child_drive, tid, {
        "status": "fail", "check": "pytest tests/red.py", "criterion_id": "claim_red",
    })
    append_verification_receipt(child_drive, tid, {
        "status": "pass", "check": "pytest tests/green.py", "criterion_id": "claim_green",
    })
    assert read_verification_receipts(tmp_path, tid) == []

    # Finalization copy-back publishes the receipts file to the canonical root
    # (no parent-side read has happened yet — the publish alone must carry them).
    copied = copy_child_task_result(tmp_path, {"id": tid, "drive_root": str(child_drive)})
    assert copied is not None
    canonical = read_verification_receipts(tmp_path, tid)
    assert [r["criterion_id"] for r in canonical] == ["claim_red", "claim_green"]

    # Durability: the receipts survive child-drive pruning (retention GC / the
    # cancel path delete the drive; the canonical copy is the durable record).
    import shutil as _shutil

    _shutil.rmtree(child_drive)
    rows = _receipt_rows_of(_get_task_result(SimpleNamespace(drive_root=tmp_path), tid))
    assert rows is not None and len(rows) == 2
    assert rows[0]["criterion_id"] == "claim_red"
    assert rows[0]["outstanding"] == "unreconciled_failed"


def test_child_receipt_republish_is_idempotent_refresh(tmp_path):
    """S3 seam (a) re-entry: copy_child_task_result runs more than once per child
    (task_done + reaper/cancel re-checks). The publish is a whole-file refresh of
    the append-only child store — newer child receipts land, nothing duplicates."""
    from ouroboros.headless import copy_child_task_result, prepare_task_drive
    from ouroboros.outcomes import append_verification_receipt, read_verification_receipts
    from ouroboros.task_results import STATUS_COMPLETED, write_task_result

    tid = "childagain"
    child_drive = prepare_task_drive(tmp_path, tid, "forked")
    write_task_result(child_drive, tid, STATUS_COMPLETED, result="done")
    append_verification_receipt(child_drive, tid, {
        "status": "fail", "check": "pytest tests/red.py", "criterion_id": "claim_red",
    })
    task = {"id": tid, "drive_root": str(child_drive)}
    copy_child_task_result(tmp_path, task)
    copy_child_task_result(tmp_path, task)  # re-entry: no duplication
    assert [r["criterion_id"] for r in read_verification_receipts(tmp_path, tid)] == ["claim_red"]

    append_verification_receipt(child_drive, tid, {
        "status": "pass", "check": "pytest tests/red.py", "criterion_id": "claim_red",
    })
    copy_child_task_result(tmp_path, task)
    assert [r["criterion_id"] for r in read_verification_receipts(tmp_path, tid)] == [
        "claim_red", "claim_red",
    ]


def test_get_task_result_falls_back_to_child_drive_receipts(tmp_path):
    """S3 seam (b): before ANY canonical copy exists (child still running, or
    self-finalized but the supervisor copy-back / effective-read sync has not
    landed), _get_task_result falls back to the child drive recorded on the
    result, so the W2 rows are never silently absent in the window the parent
    most often absorbs the child in."""
    from ouroboros.headless import prepare_task_drive
    from ouroboros.outcomes import append_verification_receipt, read_verification_receipts
    from ouroboros.task_results import STATUS_SCHEDULED, write_task_result
    from ouroboros.tools.control import _get_task_result

    tid = "childlive"
    child_drive = prepare_task_drive(tmp_path, tid, "forked")
    write_task_result(
        tmp_path, tid, STATUS_SCHEDULED,
        drive_root=str(child_drive), child_drive_root=str(child_drive),
    )
    # The child has recorded receipts but NO result yet (still running): nothing
    # exists canonically and the effective read has no child result to sync from.
    append_verification_receipt(child_drive, tid, {
        "status": "fail", "check": "pytest tests/red.py", "criterion_id": "claim_red",
    })
    assert read_verification_receipts(tmp_path, tid) == []

    rows = _receipt_rows_of(_get_task_result(SimpleNamespace(drive_root=tmp_path), tid))
    assert rows is not None and len(rows) == 1
    assert rows[0]["criterion_id"] == "claim_red"
    assert rows[0]["outstanding"] == "unreconciled_failed"


def test_get_task_result_uses_child_terminal_over_stale_parent(tmp_path):
    from ouroboros.task_results import STATUS_COMPLETED, STATUS_SCHEDULED, write_task_result
    from ouroboros.tools.control import _get_task_result

    child_drive = tmp_path / "state" / "headless_tasks" / "child123" / "data"
    child_drive.mkdir(parents=True)
    write_task_result(
        tmp_path,
        "child123",
        STATUS_SCHEDULED,
        child_drive_root=str(child_drive),
        result="stale parent handoff",
    )
    write_task_result(
        child_drive,
        "child123",
        STATUS_COMPLETED,
        result="child terminal handoff",
        cost_usd=0.42,
        trace_summary="child trace",
    )

    ctx = SimpleNamespace(drive_root=tmp_path)
    output = _get_task_result(ctx, "child123")

    assert "child terminal handoff" in output
    assert "stale parent handoff" not in output
    assert "[SUBTASK_TRACE]" in output
