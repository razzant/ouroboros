"""Receipt-honesty regressions for the contributor review lane."""

import hashlib
import pathlib


def test_packet_path_fields_are_posix_under_their_placeholders_on_every_os(monkeypatch):
    """A Windows run writes the packet a POSIX run writes: a path field under a
    replaced root reads ``$REVIEW_DRIVE/observability/...``, while free text keeps
    its own characters (a transcript's ``\\d`` is not a separator)."""
    from scripts import contributor_review_evidence as evidence

    win = pathlib.PureWindowsPath
    drive, repo, home = win("D:/a/_temp/drive"), win("D:/a/ouroboros"), win("C:/Users/runneradmin")
    replacements = sorted([(str(drive), "$REVIEW_DRIVE"), (str(repo), "$REPO"), (str(home), "$HOME")],
                          key=lambda item: len(item[0]), reverse=True)
    monkeypatch.setattr(evidence, "_NATIVE_SEPARATORS", ("\\", "/"), raising=False)
    packet = evidence.public_projection({
        "prompt_ref": {"manifest_ref": {"path": str(drive / "observability" / "calls" / "review" / "t1_prompt.json")}},
        "review_record": {"path": str(drive / "state" / "r1.json"), "subject": {"root": str(repo)}},
        "review_data_root": str(drive),
        "transcript": f"read {repo / 'README.md'}\nmatch \\d+ then {drive}",
    }, replacements=replacements)
    assert packet == {
        "prompt_ref": {"manifest_ref": {"path": "$REVIEW_DRIVE/observability/calls/review/t1_prompt.json"}},
        "review_record": {"path": "$REVIEW_DRIVE/state/r1.json", "subject": {"root": "$REPO"}},
        "review_data_root": "$REVIEW_DRIVE",
        "transcript": "read $REPO\\README.md\nmatch \\d+ then $REVIEW_DRIVE",
    }

    monkeypatch.setattr(evidence, "_NATIVE_SEPARATORS", ("/",), raising=False)
    assert evidence.public_projection(
        {"path": "/tmp/drive/odd\\name.json", "root": "/tmp/drive"},
        replacements=[("/tmp/drive", "$REVIEW_DRIVE")],
    ) == {"path": "$REVIEW_DRIVE/odd\\name.json", "root": "$REVIEW_DRIVE"}


def test_receipt_mismatch_preserves_the_original_block_cause():
    from scripts.contributor_review_evidence import finalize_contributor_outcome

    exit_code, outcome = finalize_contributor_outcome(
        outcome={
            "status": "blocked",
            "block_reason": "tests_preflight_blocked",
            "message": "Focused preflight failed: test_example.py::test_case",
        },
        exit_code=3,
        mismatches=["response_receipt_absent:triad:slot_1"],
    )

    assert exit_code == 3
    assert outcome["block_reason"] == "execution_receipt_mismatch"
    assert outcome["original_block_reason"] == "tests_preflight_blocked"
    assert outcome["original_message"] == (
        "Focused preflight failed: test_example.py::test_case"
    )


def test_shared_project_receipts_bind_final_retirement_after_all_slots_settle(tmp_path):
    from ouroboros import delegate_custody as custody
    from ouroboros.observability import persist_call
    from scripts.contributor_review_evidence import bind_execution_receipts

    config = {
        "pool_slots": [{
            "slot_id": slot_id,
            "route": {
                "kind": "agent_session",
                "target_id": "codex=gpt-5.6-sol",
                "profile_id": "pinned",
            },
            "effort": "high",
        } for slot_id in ("slot_1", "slot_2")],
    }
    actors = []
    for index, slot_id in enumerate(("slot_1", "slot_2"), start=1):
        run_id = f"run-{index}"
        prompt_ref = persist_call(
            tmp_path, task_id="review", call_id=f"prompt-{index}", call_type="prompt",
            payload={"slot": {
                "slot_id": slot_id, "model": "codex=gpt-5.6-sol",
                "effort": "high", "route": "agent_session",
                "session_target": "codex=gpt-5.6-sol",
                "session_profile": "pinned",
            }},
        )
        transcript = f"review {index} complete"
        response_ref = persist_call(
            tmp_path, task_id="review", call_id=f"response-{index}",
            call_type="response", payload={
                "message": {"session_transcript": transcript},
                "usage": {
                    "provider": "claudexor", "delegated_route": "codex",
                    "resolved_model": "gpt-5.6-sol", "applied_profile": "pinned",
                    "applied_access": "readonly", "custody_durable": True,
                    "delegated_run_id": run_id,
                    "settlement": {
                        "settled": True, "ledger_recorded": True,
                        "project_retired": False,
                    },
                    "verdict_provenance": {
                        "raw_transcript_chars": len(transcript),
                        "raw_transcript_sha256": hashlib.sha256(
                            transcript.encode()
                        ).hexdigest(),
                    },
                },
            },
        )
        actors.append(("pool", {
            "slot_id": slot_id, "status": "responded",
            "prompt_ref": prompt_ref, "response_ref": response_ref,
        }))
        custody.record_started(tmp_path, custody.RunCustody(
            run_id=run_id, task_id="review", project_id="shared-project",
            project_owned=True, ledger_root=str(tmp_path),
        ))

    for index in (1, 2):
        custody.emit(tmp_path, custody.LEDGER_RECORDED, {"run_id": f"run-{index}"})
        custody.emit(tmp_path, custody.SETTLED, {"run_id": f"run-{index}"})
    custody.emit(tmp_path, custody.PROJECT_RETIRED, {"run_id": "run-1"})
    custody._CUSTODY.clear()

    receipts, mismatches, _ = bind_execution_receipts(
        actors=actors, resolved_config=config, drive_root=tmp_path,
    )

    assert mismatches == []
    assert [row["observed"]["settlement"]["project_retired"]
            for row in receipts] == [True, True]
    assert all(
        row["observed"]["settlement"]["bound_at"]
        == "panel_complete_custody_replay"
        for row in receipts
    )
