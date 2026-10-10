"""Cold history through actual commit/change preparation, with only dispatch mocked."""
from pathlib import Path
import json

import pytest

from ouroboros import review_ledger, review_substrate
from ouroboros.review_history import review_dispute_history
from ouroboros.tools.registry import ToolContext
from tests import _contributor_packet_shared as shared
from tests.test_review_change_end_to_end import _brief_text, staged_body as _staged_body

staged_body = _staged_body
TASK = "cold-dispute"
ORIGINAL = "Original recommendation: consider a database. " + "Technical argument. " * 130 + "ORIGINAL_END"
REASON = "Keep bounded files: the owner authorized no service.\n\n" + "Exact author reasoning. " * 160 + "AUTHOR_END"
REBUTTAL = "We retained the file design.\n" + "The replacement is out of scope. " * 110 + "REBUTTAL_END"


def _dispatch(briefs, marker):
    golden = shared.golden_substrate(briefs)

    def run(request, **kwargs):
        result = golden(request, **kwargs)
        for actor in result.actors:
            value = json.loads(actor["raw_text"])
            change = value if isinstance(value, list) else value["change"]
            (change or value["coupling"])[0]["reason"] = marker
            actor["raw_text"] = json.dumps(value, ensure_ascii=False)
        return result

    return run


def _run(ctx, surface, rebuttal):
    if surface == "change":
        from ouroboros.tools.review_change import run_review_change

        result = run_review_change(ctx, root="system_repo", surface="change", subject="index",
                                   goal="Ship the bounded file design", review_rebuttal=rebuttal)
        assert result["state"] == "settled", result
        return result["record_id"]
    from ouroboros.tools.git_review_cycle import _run_non_committing_review_cycle

    result = _run_non_committing_review_cycle(ctx, "fix: update bounded file design", skip_advisory_review=True,
        goal="Ship the bounded file design", review_rebuttal=rebuttal)
    assert result["status"] == "passed", result
    return result["review_record_id"]


@pytest.mark.parametrize("surface", ["change", "commit"])
def test_three_versions_recover_sources_and_author_facts_in_every_delivery(staged_body, tmp_path, monkeypatch, surface):
    from ouroboros import capability_evidence, reviewer_window
    from ouroboros.review_state import load_state, save_state

    monkeypatch.setattr(capability_evidence, "probe", lambda *a, **k: None)
    monkeypatch.setattr(reviewer_window, "reviewer_context_window", lambda *a, **k: 1_000_000)
    monkeypatch.setenv("OUROBOROS_REVIEW_MAX_CYCLES", "6")
    root, drive = Path(staged_body["repo"]), tmp_path / "history-data"
    records, versions, given = [], [], []
    for version in range(3):
        # New context for each round: no _review_history survives in memory.
        ctx = ToolContext(repo_dir=root, drive_root=drive, task_id=TASK)
        (root / "ouroboros/tools/review.py").write_text(f"RULES = 'proposal version {version}'\n", encoding="utf-8")
        shared.git(root, "add", "-A")
        briefs = []
        monkeypatch.setattr(review_substrate, "run_review_request", _dispatch(briefs, ORIGINAL if version != 1 else "Middle-version argument"))
        rid = _run(ctx, surface, REBUTTAL if version == 1 else "")
        records.append(rid)
        record = review_ledger.load_record(drive, rid)
        versions.append(record["subject"]["tree_sha"])
        assert record["brief"]["dispute_input"]["previous_records"] == ([] if version == 0 else [records[-2]])
        assert record["brief"]["dispute_input"]["gaps"] == []
        if version == 0:
            review_ledger.note_author_decision(drive, rid, {"disposition": "partial", "rationale": "First author stance"})
            review_ledger.note_author_decision(drive, rid, {"disposition": "rejected", "rationale": REASON})
        if version == 2:
            given = briefs
    assert len(set(versions)) == 3
    assert {b["slot_id"] for b in given} == {"t1", "t2", "s1"}  # packet, session, native
    indexes = []
    for brief in given:
        text = _brief_text(brief)
        for value in (ORIGINAL, "First author stance", REASON.replace("\n", "\\n"),
                      REBUTTAL.replace("\n", "\\n"), "Middle-version argument", versions[0], versions[1]):
            assert value in text, (brief["slot_id"], value[:60])
        assert "Recorded review dispute index" in text
        indexes.append(json.loads(text.split("### Recorded review dispute index\n\n```json\n", 1)[1].split("\n```", 1)[0]))
    assert all(index == indexes[0] for index in indexes)

    # The newest exact link carries predecessors even without their hot attempt rows.
    state = load_state(drive)
    assert all(a.review_record_id for a in state.attempts)
    state.attempts = [a for a in state.attempts if a.review_record_id == records[-1]]
    save_state(drive, state)
    history = review_dispute_history(drive_root=drive, repo_root=root, task_id=TASK)
    assert history["status"] == "complete", history["gaps"]
    assert [r["review_record_id"] for r in history["rounds"]] == records
    assert history["record_heads"] == [records[-1]]
    original_sources = [s["response"] for r in history["rounds"] for s in r["reviewers"]
                        if "response" in s and ORIGINAL in s["response"].get("text", "")]
    assert original_sources
    for response in original_sources:
        assert review_ledger.read_source(drive, TASK, response["source"]).decode() == response["text"]
    aliases = [s["response"] for r in history["rounds"] for s in r["reviewers"] if "same_content_as" in s.get("response", {})]
    assert aliases  # third version repeated an immutable response, not another embedded prompt
    for alias in aliases:
        assert review_ledger.read_source(drive, TASK, alias["source"]) == review_ledger.read_source(drive, TASK, alias["same_content_as"])
    assert "request_messages" not in json.dumps(history)
    assert REASON in [r["reason"] for r in history["decision_rows"] if r["remark"] == "Explicit author decision"]
    from ouroboros.artifacts import task_artifact_dir_path

    source = original_sources[0]["source"]
    (task_artifact_dir_path(drive, source["task_id"], create=False) / source["ref"]["path"]).unlink()
    missing = review_dispute_history(drive_root=drive, repo_root=root, task_id=TASK)
    assert missing["status"] == "source_unavailable" and missing["gaps"]
    assert len(missing["rounds"]) == 3
    assert REASON in [r["reason"] for r in missing["decision_rows"] if r["remark"] == "Explicit author decision"]


def test_legacy_missing_binding_is_a_gap_and_other_tasks_are_not_swept_in(tmp_path):
    from ouroboros.review_state import AdvisoryReviewState, CommitAttemptRecord, make_repo_key, save_state

    root, drive = tmp_path / "repo", tmp_path / "data"
    root.mkdir()
    state = AdvisoryReviewState()
    state.attempts = [CommitAttemptRecord(ts="2026-10-09T00:00:00Z", commit_message="earlier", status="reviewed",
        task_id=task, repo_key=make_repo_key(root), attempt=1, paid=True,
        critical_findings=[{"reason": reason}], triad_raw_results=[{"raw_text": reason}])
        for task, reason in ((TASK, "Available legacy argument"), ("unrelated-task", "UNRELATED_DO_NOT_INCLUDE"))]
    save_state(drive, state)
    history = review_dispute_history(drive_root=drive, repo_root=root, task_id=TASK)
    text = json.dumps(history)
    assert history["status"] == "source_unavailable"
    assert "review-record binding" in text and "Available legacy argument" in text
    assert "UNRELATED_DO_NOT_INCLUDE" not in text
