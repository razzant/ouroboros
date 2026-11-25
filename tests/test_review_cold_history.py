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


def _run(ctx, surface, rebuttal, *, handed_back=False):
    """The record of one wave; a review-only commit cycle hands a material outcome back for the author's decision."""
    if surface == "change":
        from ouroboros.tools.review_change import run_review_change

        result = run_review_change(ctx, root="system_repo", surface="change", subject="index",
                                   goal="Ship the bounded file design", review_rebuttal=rebuttal)
        assert result["state"] == "settled", result
        return result["record_id"]
    from ouroboros.tools.git_review_cycle import _run_non_committing_review_cycle

    result = _run_non_committing_review_cycle(ctx, "fix: update bounded file design", skip_advisory_review=True,
        goal="Ship the bounded file design", review_rebuttal=rebuttal)
    assert result["status"] == ("reviewed" if handed_back else "passed"), result
    return result["review_reference"]["review_record_id"] if handed_back else result["review_record_id"]


@pytest.mark.parametrize(("surface", "own_folder"), [("change", False), ("commit", False), ("change", True), ("commit", True)])
def test_three_versions_recover_sources_and_author_facts_in_every_delivery(staged_body, tmp_path, monkeypatch, surface,
                                                                           own_folder):
    """``own_folder``: a project-room task has its own data folder. The commit gate writes its attempt rows there,
    ``review_change`` keeps its rows and every record in the install's canonical folder; on both surfaces each
    later round still receives the earlier ones."""
    from ouroboros import capability_evidence, reviewer_window
    from ouroboros.review_state import load_state, save_state

    monkeypatch.setattr(capability_evidence, "probe", lambda *a, **k: None)
    monkeypatch.setattr(reviewer_window, "reviewer_context_window", lambda *a, **k: 1_000_000)
    monkeypatch.setenv("OUROBOROS_REVIEW_MAX_CYCLES", "6")
    root, drive = Path(staged_body["repo"]), tmp_path / "history-data"
    own = drive / "state" / "headless_tasks" / TASK / "data" if own_folder else drive
    records, versions, given = [], [], []
    for version in range(3):
        # New context for each round: no _review_history survives in memory.
        ctx = ToolContext(repo_dir=root, drive_root=own, task_id=TASK, budget_drive_root=str(drive) if own_folder else "",
                          task_metadata={"budget_drive_root": str(drive)} if own_folder else {})
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
    if surface == "commit":
        # The task's own resident index and its note options read the same attempt rows as the next wave's brief.
        from types import SimpleNamespace

        from ouroboros.context import _task_authority_projection
        from ouroboros.review_history_view import review_context_updates, review_note_options

        env = SimpleNamespace(drive_root=own, budget_drive_root=str(drive), repo_dir=root)
        shown = _task_authority_projection(env, {"id": TASK})["commit_review_authority"]
        assert [r["review_record_id"] for r in shown[0]["dispute_history"]["rounds"]] == records
        updates, identities = review_context_updates(ctx, families={"commit": str(root)})
        assert identities["commit"] and all(rid in json.dumps(updates) for rid in records)
        assert review_note_options(ctx)["entries"]

    # The newest exact link carries predecessors even without their hot attempt rows.
    held = own if surface == "commit" else drive  # the folder that holds this surface's attempt rows
    state = load_state(held)
    assert len(state.attempts) >= 3 and all(a.review_record_id for a in state.attempts)
    state.attempts = [a for a in state.attempts if a.review_record_id == records[-1]]
    save_state(held, state)
    history = review_dispute_history(drive_root=drive, attempts_root=own, repo_root=root, task_id=TASK)
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
    missing = review_dispute_history(drive_root=drive, attempts_root=own, repo_root=root, task_id=TASK)
    assert missing["status"] == "source_unavailable" and missing["gaps"]
    assert len(missing["rounds"]) == 3
    assert REASON in [r["reason"] for r in missing["decision_rows"] if r["remark"] == "Explicit author decision"]
    if own_folder:
        # The folder that holds none of this surface's rows cannot be read: the gap names it, the other's rounds stay.
        broken = drive if surface == "commit" else own
        (broken / "state").mkdir(parents=True, exist_ok=True)
        (broken / "state" / "advisory_review.json").write_text("{broken", encoding="utf-8")
        partial = review_dispute_history(drive_root=drive, attempts_root=own, repo_root=root, task_id=TASK)
        assert [r["review_record_id"] for r in partial["rounds"]] == records
        assert [gap["source"].get("folder") for gap in partial["gaps"] if "attempt bindings" in gap["reason"]] == [str(broken)]


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


def _two_waves_one_retry_key(tmp_path):
    """Wave A prepared and bound its input; wave B, the same retry key, a new physical attempt, kept none (its bind
    write failed): the state the issue describes."""
    from ouroboros.review_state import AdvisoryReviewState, CommitAttemptRecord, make_repo_key, save_state

    root = tmp_path / "subject"
    root.mkdir()
    earlier = {"previous_records": ["older-history"], "rebuttal": {"text": "only wave A saw this"}, "gaps": []}
    state = AdvisoryReviewState()
    state.attempts = [
        CommitAttemptRecord(ts=f"2026-10-09T00:0{number}:00Z", commit_message="same intent", status="reviewed",
                            task_id=TASK, repo_key=make_repo_key(root), attempt=number, paid=True,
                            tool_name="commit_reviewed", review_retry_key="same-retry", review_record_id=rid,
                            review_input=inputs)
        for number, rid, inputs in ((1, "wave-a", earlier), (2, "wave-b", {}))]
    save_state(tmp_path, state)
    return root, earlier


def test_new_physical_attempt_never_borrows_an_older_waves_input(tmp_path):
    from ouroboros.review_history import wave_history_input

    root, earlier = _two_waves_one_retry_key(tmp_path)
    facts = {"retry_key": "same-retry", "task_id": TASK, "tool_name": "commit_reviewed"}
    # B's own record, and a record no row names yet (its row not stamped): neither carries A's input; each names the gap.
    for rid, number in (("wave-b", 2), ("new-unbound-wave", 2)):
        missing = wave_history_input(tmp_path, {**facts, "attempt": number}, {"root": str(root)}, rid)
        assert [gap["reason"] for gap in missing["gaps"]] == ["wave preparation has no exact dispute-input binding"]
        assert "rebuttal" not in missing and "older-history" not in missing["previous_records"]
    # A record no row can name, with no attempt number either: the same gap, never the newest row carrying the key.
    unnamed = wave_history_input(tmp_path, facts, {"root": str(root)}, "somebody-else")
    assert unnamed["gaps"] and "rebuttal" not in unnamed


def test_a_free_rejoin_still_reads_the_earlier_waves_own_input(tmp_path):
    from ouroboros.review_history import wave_history_input

    root, earlier = _two_waves_one_retry_key(tmp_path)
    facts = {"retry_key": "same-retry", "task_id": TASK, "tool_name": "commit_reviewed"}
    # The record is wave A's: whichever physical row later collects it, its input is the one A prepared.
    for number in (None, 1, 2):
        assert wave_history_input(tmp_path, {**facts, "attempt": number}, {"root": str(root)}, "wave-a") == earlier


def test_a_second_attempts_failed_bind_records_its_own_prepared_input_and_a_rejoin_keeps_the_first(
        staged_body, tmp_path, monkeypatch):
    """The real ``prepare_history`` and ``settle_commit_review_ledger``: wave A prepared and bound an input; wave B, the
    same retry key, prepared its own but its attempt-row bind write failed. B's record carries B's prepared input with the
    bind gap, never A's and never heads rebuilt after dispatch; a free rejoin of A reads A's input and leaves both."""
    from ouroboros import capability_evidence, reviewer_window
    from ouroboros import review_state as state_mod
    from ouroboros.review_history import prepare_history
    from ouroboros.review_state import CommitAttemptRecord, load_state, make_repo_key, update_state
    from ouroboros.tools import commit_gate
    from ouroboros.tools import git as git_mod

    monkeypatch.setattr(capability_evidence, "probe", lambda *a, **k: None)
    monkeypatch.setattr(reviewer_window, "reviewer_context_window", lambda *a, **k: 1_000_000)
    monkeypatch.setenv("OUROBOROS_REVIEW_MAX_CYCLES", "6")
    root, drive = Path(staged_body["repo"]), tmp_path / "data"
    ctx_a = ToolContext(repo_dir=root, drive_root=drive, task_id=TASK)
    monkeypatch.setattr(review_substrate, "run_review_request", _dispatch([], ORIGINAL))
    rid_a = _run(ctx_a, "commit", "Wave A rebuttal only")
    row_a = next(a for a in load_state(drive).attempts if a.review_record_id == rid_a)
    retry, a_input = row_a.review_retry_key, review_ledger.load_record(drive, rid_a)["brief"]["dispute_input"]
    assert retry and a_input["rebuttal"] and row_a.review_input == a_input

    def fresh_ctx(**state):
        ctx = ToolContext(repo_dir=root, drive_root=drive, task_id=TASK)
        git_mod._reset_commit_review_state(ctx)
        ctx._current_review_retry_key, ctx._last_review_structured = retry, {"retry_key": retry}
        for name, value in state.items():
            setattr(ctx, name, value)
        return ctx

    def settle(ctx):
        facts = {"fingerprint": row_a.pre_review_fingerprint, "binding": {"tree_sha": "t" * 40, "parents": ["p" * 40],
                                                                       "diff_sha256": "d" * 64}}
        return commit_gate.settle_commit_review_ledger(ctx, "same intent", goal="g", scope="s", pre_fingerprint=facts)

    # B: a new physical attempt under A's retry key whose own bind write is the only thing that fails.
    update_state(drive, lambda state: state.record_attempt(CommitAttemptRecord(
        ts="2026-11-18T00:00:00Z", commit_message="same intent", status="reviewing", task_id=TASK,
        repo_key=make_repo_key(root), tool_name="commit_reviewed", attempt=2, paid=True, review_retry_key=retry)))
    ctx_b = fresh_ctx(_current_review_attempt_number=2)
    monkeypatch.setattr(state_mod, "update_state", lambda *a, **k: (_ for _ in ()).throw(OSError("lock held")))
    prepare_history(ctx_b, None, "Wave B rebuttal only")
    monkeypatch.undo()  # the failure belongs to the bind alone; the record write below is ordinary
    prepared = ctx_b._review_dispute_input
    assert [g["reason"] for g in prepared["gaps"]] == ["attempt input binding failed (OSError)"]
    assert prepared["rebuttal"] != a_input["rebuttal"] and prepared["previous_records"] == [rid_a]
    rid_b = settle(ctx_b)
    assert rid_b and rid_b != rid_a
    assert next(a for a in load_state(drive).attempts if a.attempt == 2).review_input == {}  # nothing was bound to B
    assert review_ledger.load_record(drive, rid_b)["brief"]["dispute_input"] == prepared

    # A free rejoin of A: its record, its input; neither record changes.
    ctx_c = fresh_ctx(_review_reconcile_only=True, _current_review_record_id=rid_a, _current_review_attempt_number=1,
                      _pending_review_attempt=row_a)
    prepare_history(ctx_c, None, "")
    assert ctx_c._review_dispute_input == a_input
    assert settle(ctx_c) == rid_a
    assert review_ledger.load_record(drive, rid_a)["brief"]["dispute_input"] == a_input
    assert review_ledger.load_record(drive, rid_b)["brief"]["dispute_input"] == prepared
