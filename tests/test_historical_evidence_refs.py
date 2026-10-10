"""Historical preview availability is not complete criterion evidence."""

from pathlib import Path

import pytest

from ouroboros.artifacts import read_actor_source_bytes, task_artifact_dir_path
from ouroboros.review_evidence import (
    acceptance_evidence_ref_vocabulary,
    annotate_criteria_evidence_resolution,
    build_task_acceptance_evidence,
)
from ouroboros.review_substrate import ReviewRunResult, task_acceptance_is_clean
from tests import test_main_authored_context as context_fixtures

main_loop = context_fixtures.main_loop


@pytest.fixture(params=["captured_preview", "never_captured", "source_unavailable"])
def historical_packet(main_loop, request):
    f = main_loop
    if request.param != "never_captured":
        f.run([{"content": "The requested correction is complete."}])
        anchor = f.ctx._historical_author_inputs["anchors"][0]
        assert anchor["status"] == "captured"
        assert anchor["preview_complete"] is False
        raw = read_actor_source_bytes(f.ctx.drive_root, f.ctx.task_id, anchor["source_ref"])
        assert len(raw) > len(anchor["preview"].encode())
        if request.param == "source_unavailable":
            (task_artifact_dir_path(f.ctx.drive_root, f.ctx.task_id)
             / anchor["source_ref"]["path"]).unlink()
    else:
        f.ctx.task_id = "historical-not-captured"
    packet = build_task_acceptance_evidence(
        f.ctx, drive_root=Path(f.ctx.drive_root), task_id=f.ctx.task_id,
        canonical_subject="The requested correction is complete.",
    )
    exhibit = packet["historical_author_inputs"]
    assert packet["__provenance__"]["historical_author_inputs"] == "host_attested"
    assert exhibit["status"] == ("captured" if request.param == "captured_preview" else "unavailable")
    return packet


def _acceptance(packet, refs):
    actor = {
        "slot_id": "s0", "signal": "PASS",
        "parsed": {"verdict": "PASS", "outcome_tier": "solved", "criteria_used": [{
            "criterion": "The requested correction meets its historical purpose.",
            "status": "supported", "evidence_refs": refs,
        }]},
    }
    annotate_criteria_evidence_resolution([actor], packet)
    return ReviewRunResult(
        request={"surface": "task_acceptance", "policy": {"min_successful_slots": 1}},
        actors=[actor], parsed_findings=[], aggregate_signal="PASS",
    )


def test_partial_or_unavailable_history_alone_cannot_make_acceptance_clean(historical_packet):
    result = _acceptance(historical_packet, ["historical_author_inputs"])
    basis = acceptance_evidence_ref_vocabulary(historical_packet)["historical_author_inputs"]
    assert (basis, task_acceptance_is_clean(result)) == ("partial", False)
    actor = result.actors[0]
    assert actor["criteria_refs_unresolved"][0]["refs"] == [
        {"ref": "historical_author_inputs", "resolved_as": "partial"},
    ]
    assert result.aggregate_signal == actor["parsed"]["verdict"] == "PASS"
    assert actor["parsed"]["outcome_tier"] == "solved"


def test_complete_evidence_still_resolves_without_historical_reader_receipt(historical_packet):
    # The existing vocabulary needs one resolving exhibit. Partial history is
    # disclosed, but its presence adds no mandatory reader receipt or veto.
    packet = {**historical_packet, "repo_diff": "Complete correction diff.", "repo_diff_complete": True}
    assert acceptance_evidence_ref_vocabulary(packet)["repo_diff"] == "packet_section"
    result = _acceptance(packet, ["historical_author_inputs", "repo_diff"])
    assert task_acceptance_is_clean(result) is True
    assert result.aggregate_signal == "PASS"
