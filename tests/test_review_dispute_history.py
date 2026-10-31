"""Both parts of the existing review brief preserve supplied dispute facts."""
from copy import deepcopy
from pathlib import Path

import pytest

from ouroboros.tools.review_helpers import review_history_with_obligations
from tests.test_review_change_end_to_end import staged_body as _staged_body

staged_body = _staged_body


def _rounds():
    return [
        {"attempt": 1, "subject": {"diff_sha": "old-diff"}, "verdict": "FAIL",
         "critical": [{"item": "parser", "reason": "Original objection", "recommendation": "Use a database"}],
         "author_disposition": {"decision": "reject", "rationale": "The bounded file is the accepted scope."},
         "reviewer_outputs": [{"slot_id": "old-seat", "raw_text": "I disagree because atomic writes need a proof."}]},
        {"attempt": 2, "subject": {"diff_sha": "middle-diff"}, "verdict": "PASS",
         "advisory": [{"item": "naming", "reason": "Check the spelling"}]},
        {"attempt": 3, "subject": {"diff_sha": "new-diff"}, "verdict": "FAIL",
         "critical": [{"item": "parser", "reason": "Consider a database again"}],
         "review_rebuttal": "Now multiple writers are required; reconsider the earlier scope."},
    ]


@pytest.mark.parametrize("route,retrieves,structured", [
    pytest.param("api_chat", False, False, id="api_chat-False"),
    pytest.param("api_chat", True, False, id="api_chat-True"),
    pytest.param("agent_session", True, False, id="agent_session-True"),
    pytest.param("api_chat", False, True, id="api_chat-False-structured"),
    pytest.param("api_chat", True, True, id="api_chat-True-structured"),
    pytest.param("agent_session", True, True, id="agent_session-True-structured"),
])
def test_public_packet_and_two_part_brief_keep_all_supplied_arguments(staged_body, tmp_path, monkeypatch, route, retrieves, structured):
    from ouroboros import reviewer_window, capability_evidence
    from ouroboros.tools.registry import ToolContext
    from ouroboros.tools.review_admission import build_two_part_brief
    from ouroboros.tools.review_subject import ReviewSubjectSpec, freeze_subject

    # Every capability answer is local; this test assembles inputs and dispatches nothing.
    monkeypatch.setattr(reviewer_window, "reviewer_context_window", lambda *a, **k: 1_000_000)
    monkeypatch.setattr(capability_evidence, "probe", lambda *a, **k: None)
    root = Path(staged_body["repo"])
    ctx = ToolContext(repo_dir=root, drive_root=tmp_path / "data")
    frozen = freeze_subject(ctx, ReviewSubjectSpec(root_kind="system_repo", root=str(root), kind="index",
                                                 governance_root=str(root), surface="change", layer="body"))
    history = _rounds()
    if structured:
        for wave in history:
            wave["verdict"] = {"aggregate": wave["verdict"], "advisory_findings": [
                {"item": "diagnostic", "reason": "Structured verdict evidence stays visible."}]}
    original_history = deepcopy(history)
    coupling = [{"verdict": "FAIL", "blocked": True, "summary": "The coupling rationale stays visible.",
                 "critical_findings": [{"item": "contract", "reason": "Reader mismatch"}],
                 "author_disposition": {"decision": "reject", "rationale": "The reader is migrated together."}}]
    brief = build_two_part_brief(
        frozen, {"slot_id": "fresh", "model": "fake/reviewer", "route": route, "retrieves": retrieves},
        drive_root=ctx.drive_root, task_id="test-task", review_history=history, coupling_history=coupling,
        owner_words="Owner chose file storage for this scope; that approval excludes a database service.")
    text = brief["system"]
    for value in ("The bounded file is the accepted scope.", "I disagree because atomic writes need a proof.",
                  "Use a database", "Check the spelling", "Now multiple writers are required", "old-diff", "new-diff",
                  "approval excludes a database service"):
        assert value in text
    assert history == original_history
    if structured:
        assert "Structured verdict evidence stays visible." in text
    assert brief["parts"] == (["change", "coupling"] if retrieves else ["change"])
    if retrieves:
        assert "The reader is migrated together." in text
        assert "The coupling rationale stays visible." in text
    assert "An author's rejection is not reviewer agreement" in text


def test_failed_and_open_seat_notes_reach_the_next_round(tmp_path):
    """The ledger keeps a refusal as `failure` and an open seat's note as historical `pending`; history reads both."""
    from ouroboros import review_ledger as ledger
    from ouroboros.review_history import review_dispute_history
    from tests.test_review_ledger import BOTH, _facts, _raw

    refusal = "Engine refusal: measured 940000 characters; limit 900000 characters"
    note = "Timeout at dispatch; the physical review operation is still in flight."
    raws = [_raw("ok", "m1", parts=BOTH, raw_text="Received answer\nEND"),
            _raw("refused", "m2", "error", parts=BOTH, raw_text="", error=refusal),
            _raw("open", "m3", "timeout", parts=BOTH, raw_text=note, operation_id="op", operation_state="in_flight")]
    rid = ledger.write_record(tmp_path, ledger.build_commit_gate_record(_facts(raws), drive_root=tmp_path))["record_id"]

    def seats(labels=None):
        history = review_dispute_history(drive_root=tmp_path, repo_root="/nowhere", task_id="task-1", previous_records=[rid])
        assert not [gap for gap in history["gaps"] if "binding is absent" in gap["reason"]], history["gaps"]
        if labels is not None:
            from ouroboros.review_history_view import review_index_view
            labels.extend(sorted(entry["what"].split(" in record ")[0]
                                 for entry in review_index_view(history, "commit")["addresses"].values() if " of seat " in entry["what"]))
        return {seat["seat_id"]: (seat["response"]["source"]["role"], seat["response"]["text"])
                for seat in history["rounds"][0]["reviewers"]}

    labels = []
    assert seats(labels) == {"ok": ("response", "Received answer\nEND"), "refused": ("failure", refusal), "open": ("pending", note)}
    assert labels == ["answer of seat ok (m1)", "failure note of seat refused (m2)", "pending note of seat open (m3)"]
    ledger.settle_from_collected(tmp_path, rid, [_raw("open", "m3", parts=BOTH, raw_text="Late answer\nEND", operation_id="op")])
    assert seats()["open"] == ("response", "Late answer\nEND"), "a received answer outranks its stale note"
    # Records of the earlier writer named every text `response`, in write order: the newest is the answer.
    stale = ledger.retain_text_source(tmp_path, "task-1", record_id=rid, seat_id="ok", role="response",
                                      part="change,coupling", text=note)
    ledger.revise_record(tmp_path, rid, lambda r: {**r, "rows": [
        {**row, "source_refs": [stale, *row["source_refs"]]} if row["seat_id"] == "ok" else row for row in r["rows"]]})
    assert seats()["ok"] == ("response", "Received answer\nEND")


def test_missing_durable_obligations_are_an_explicit_gap_with_available_history(tmp_path):
    state = tmp_path / "state"
    state.mkdir()
    (state / "advisory_review.json").write_text("{broken", encoding="utf-8")
    text = review_history_with_obligations(_rounds(), drive_root=tmp_path, repo_root=tmp_path / "repo")
    assert "REVIEW_HISTORY_SOURCE_UNAVAILABLE" in text
    assert "The bounded file is the accepted scope." in text
    assert "I disagree because atomic writes need a proof." in text


def test_obligation_reason_is_complete_in_shared_history(tmp_path):
    from ouroboros.review_state import AdvisoryReviewState, ObligationItem, make_repo_key, save_state

    reason = "Exact reason and rejected alternative. " * 150 + "DECISIVE_REASON_END"
    state = AdvisoryReviewState()
    state.open_obligations.append(ObligationItem(
        obligation_id="ob-1", item="contract", severity="critical", reason=reason,
        source_attempt_ts="2026-10-09T00:00:00Z", source_attempt_msg="repair", status="still_open",
        repo_key=make_repo_key(tmp_path)))
    save_state(tmp_path, state)
    text = review_history_with_obligations([], drive_root=tmp_path, repo_root=tmp_path)
    assert reason in text and "OMISSION NOTE" not in text


@pytest.mark.parametrize("route,retrieves", [("api_chat", False), ("agent_session", True)])
@pytest.mark.parametrize("structured", [False, True], ids=["scalar", "structured"])
def test_supplied_source_bound_answers_keep_scalar_or_structured_verdict(staged_body, tmp_path, monkeypatch, route, retrieves, structured):
    import json
    from ouroboros import capability_evidence, reviewer_window, review_ledger, review_history_view
    from ouroboros.review_history import review_dispute_history
    from ouroboros.tools.registry import ToolContext
    from ouroboros.tools.review_admission import build_two_part_brief
    from ouroboros.tools.review_subject import ReviewSubjectSpec, freeze_subject

    monkeypatch.setattr(reviewer_window, "reviewer_context_window", lambda *a, **k: 1_000_000)
    monkeypatch.setattr(capability_evidence, "probe", lambda *a, **k: None)
    root = Path(staged_body["repo"])
    ctx = ToolContext(repo_dir=root, drive_root=tmp_path / "data", task_id="test-task")
    frozen = freeze_subject(ctx, ReviewSubjectSpec(root_kind="system_repo", root=str(root), kind="index",
                                                 governance_root=str(root), surface="change", layer="body"))
    finding = {"item": "parser", "reason": "Exact retained critic argument.", "severity": "critical", "verdict": "FAIL"}
    answer = {"status": "responded", "verdict": "FAIL", "findings": [finding], "critical": 1, "coverage": "n/a"}
    raw = json.dumps(answer)
    source = review_ledger.retain_text_source(ctx.drive_root, ctx.task_id, record_id="prior-record",
        seat_id="old-seat", role="response", part="change", text=raw)
    assert source["status"] == "retained"
    history = [{"review_record_id": "prior-record", "verdict":
        {"aggregate": "FAIL", "critical_findings": [finding]} if structured else "FAIL",
        "reviewers": [{"seat_id": "old-seat", "answers": {"change": answer},
                       "response": {"text": raw, "source": source}}]}]
    original = deepcopy(history)
    brief = build_two_part_brief(frozen,
        {"slot_id": "fresh", "model": "fake/reviewer", "route": route, "retrieves": retrieves},
        drive_root=ctx.drive_root, task_id=ctx.task_id, review_history=history)
    assert "Exact retained critic argument." in brief["system"] and "old-seat" in brief["system"]
    assert brief["parts"] == (["change", "coupling"] if retrieves else ["change"])
    assert history == original and review_ledger.read_source(ctx.drive_root, ctx.task_id, source).decode("utf-8") == raw
    # The reader itself makes this supplied answer a source-bound decision.
    dispute = review_dispute_history(history, drive_root=ctx.drive_root, repo_root=root, task_id=ctx.task_id)
    binding = review_history_view.decision_entries(dispute)[0]["bound_decision"]
    assert binding is not None
    projected, notes, gaps = review_history_view.project_decision_notes(dispute, [
        {"bound_decision": binding, "remark": "Parser concern.", "reason": "Keep its exact retained argument."}])
    assert notes and not gaps
    if structured:
        assert projected["rounds"][0]["verdict"]["aggregate"] == "FAIL"
        assert "authored_view" in projected["rounds"][0]["verdict"]["critical_findings"][0]
    else:
        assert projected["rounds"][0]["verdict"] == "FAIL"
    assert projected["rounds"][0]["reviewers"][0]["answers"]["change"]["critical"] == 1
    assert "authored_view" in projected["rounds"][0]["reviewers"][0]["answers"]["change"]["findings"][0]


# ------------------------------------------------- change since earlier reviews


def _recording_seam(monkeypatch, briefs: list, sends: list, answer=None):
    """The golden physical send under the real custody layer, keeping what each seat was given."""
    import ouroboros.review_substrate as substrate
    from tests import _contributor_packet_shared as shared

    seam = shared.golden_physical_seam(sends, answer=answer)

    def run_slot(self, request, slot, **kwargs):
        briefs.append({"slot_id": slot.slot_id, "messages": [dict(m) for m in request.messages],
                       "session_task": request.session_task})
        return seam(self, request, slot, **kwargs)

    monkeypatch.setattr(substrate.ReviewCoordinator, "_run_slot", run_slot)


def _comparison(ctx, result) -> dict:
    from ouroboros import review_ledger

    return review_ledger.load_record(ctx.drive_root, result["record_id"])["brief"]["dispute_input"].get("comparison")


def _shown_to_every_seat(briefs: list, comparison: dict) -> None:
    from ouroboros.review_history import render_comparison
    from tests.test_review_change_end_to_end import _brief_text

    text = render_comparison(comparison)
    assert text and sorted(brief["slot_id"] for brief in briefs) == ["s1", "t1", "t2"]
    assert all(_brief_text(brief).count(text) == 1 for brief in briefs)  # packet, native and session alike


def test_a_later_review_is_shown_what_changed_since_the_reviews_this_task_bound_on_this_checkout(tmp_path, monkeypatch):
    """Nothing earlier: no comparison at all. The same bytes asked again: the earlier review
    is named as the same subject, not skipped. Then a real change: its paths and an excerpt.
    Each earlier verdict is a label about its own bytes; the whole subject stays reviewed."""
    from ouroboros.tools.review_change import run_review_change
    from tests import _contributor_packet_shared as shared
    from tests.test_review_change_end_to_end import _brief_text, _foreign_project

    ctx, project = _foreign_project(tmp_path, monkeypatch)
    monkeypatch.setenv("OUROBOROS_REVIEW_MAX_CYCLES", "10")
    briefs, sends = [], []
    _recording_seam(monkeypatch, briefs, sends)
    (project / "app.py").write_text("VALUE = 2\n", encoding="utf-8")
    shared.git(project, "add", "app.py")

    first = run_review_change(ctx, subject="index", goal="Bump", scope="app.py")
    assert _comparison(ctx, first) is None
    assert briefs and not any("Change since earlier reviews" in _brief_text(brief) for brief in briefs)

    briefs.clear()
    second = run_review_change(ctx, subject="index", goal="Bump", scope="app.py", author_questions=["Is two right?"])
    comparison = _comparison(ctx, second)
    [anchor] = comparison["anchors"]
    assert (anchor["review_record_id"], anchor["relation"], anchor["named"], anchor["verdict"]) == (
        first["record_id"], "same_subject", False, "PASS")
    _shown_to_every_seat(briefs, comparison)

    briefs.clear()
    (project / "lib.py").write_text("LIB = 'changed'\n", encoding="utf-8")
    shared.git(project, "add", "lib.py")
    third = run_review_change(ctx, subject="index", goal="Bump", scope="app.py and lib.py")
    comparison = _comparison(ctx, third)
    [anchor] = comparison["anchors"]
    assert (anchor["review_record_id"], anchor["relation"], anchor["changed"]) == (
        second["record_id"], "changed", ["M\tlib.py"])
    assert "+LIB = 'changed'" in anchor["excerpt"] and "base_moved" not in anchor
    assert comparison["current"]["tree"] == third["subject"]["tree_sha"]
    _shown_to_every_seat(briefs, comparison)
    assert third["aggregate"] == "PASS"


def test_a_named_review_of_another_copy_is_compared_and_a_foreign_or_unknown_one_is_a_stated_gap(tmp_path, monkeypatch):
    """since_review names a review another task made in a linked copy of this repository:
    the seats see what changed since its subject, never its findings. A review of another
    repository and an id this ledger never wrote are stated gaps, never a comparison."""
    from ouroboros import review_ledger
    from ouroboros.tools.registry import ToolContext
    from ouroboros.tools.review_change import run_review_change
    from tests import _contributor_packet_shared as shared
    from tests.test_review_change_end_to_end import _foreign_project

    ctx, project = _foreign_project(tmp_path, monkeypatch)
    monkeypatch.setenv("OUROBOROS_REVIEW_MAX_CYCLES", "10")
    briefs, sends = [], []
    _recording_seam(monkeypatch, briefs, sends)

    def other_task(root, task):
        return ToolContext(repo_dir=ctx.repo_dir, system_repo_dir=ctx.system_repo_dir, drive_root=ctx.drive_root,
                           workspace_root=root, workspace_mode="external", task_id=task)

    copy = (tmp_path / "work" / "copy").resolve()
    shared.git(project, "worktree", "add", "-q", "--detach", str(copy))
    (copy / "app.py").write_text("VALUE = 3\n", encoding="utf-8")
    shared.git(copy, "add", "app.py")
    in_copy = run_review_change(other_task(copy, "task-copy"), subject="index", goal="Bump in a copy", scope="app.py")
    foreign = (tmp_path / "work" / "foreign").resolve()
    foreign.mkdir()
    shared.git(foreign, "init", "-q")
    (foreign / "x.py").write_text("X = 1\n", encoding="utf-8")
    shared.git(foreign, "add", "-A")
    shared.git(foreign, "commit", "-q", "-m", "base")
    shared.git(foreign, "remote", "add", "origin", "https://example.com/third-party/foreign.git")
    (foreign / "x.py").write_text("X = 2\n", encoding="utf-8")
    shared.git(foreign, "add", "x.py")
    elsewhere = run_review_change(other_task(foreign, "task-foreign"), subject="index", goal="Bump x", scope="x.py")

    (project / "app.py").write_text("VALUE = 3\nEXTRA = 1\n", encoding="utf-8")
    shared.git(project, "add", "app.py")
    briefs.clear()
    named = run_review_change(ctx, subject="index", goal="Bump", scope="app.py", since_review=in_copy["record_id"])
    comparison = _comparison(ctx, named)
    [anchor] = comparison["anchors"]
    assert (anchor["review_record_id"], anchor["named"], anchor["relation"]) == (in_copy["record_id"], True, "changed")
    assert anchor["subject"]["root"] == str(copy) and anchor["changed"] == ["M\tapp.py"]
    assert "+EXTRA = 1" in anchor["excerpt"] and "findings" not in anchor
    assert review_ledger.load_record(ctx.drive_root, named["record_id"])["brief"]["since_review"] == in_copy["record_id"]
    _shown_to_every_seat(briefs, comparison)

    for since, reason in ((elsewhere["record_id"], "another repository"), ("rr-never-written", "no such review record")):
        briefs.clear()
        result = run_review_change(ctx, subject="index", goal="Bump", scope="app.py", since_review=since)
        comparison = _comparison(ctx, result)
        [gap] = [anchor for anchor in comparison["anchors"] if anchor["named"]]
        assert (gap["review_record_id"], gap["relation"]) == (since, "unavailable") and reason in gap["reason"]
        assert result["aggregate"] == "PASS" and result["reused"] is False  # a new question, the whole subject
        _shown_to_every_seat(briefs, comparison)


def test_a_rejoined_pending_wave_keeps_the_comparison_it_was_dispatched_with(tmp_path, monkeypatch):
    """A paid wave is still open when this task's review history moves on (another round
    settles). Rejoining it collects the original operation and keeps the comparison its
    seats were sent, never one rebuilt from today's history heads."""
    from ouroboros.tools.review_change import run_review_change
    from tests import _contributor_packet_shared as shared
    from tests.test_review_change_end_to_end import _foreign_project, _pending_then_answering_seat

    ctx, project = _foreign_project(tmp_path, monkeypatch)
    monkeypatch.setenv("OUROBOROS_REVIEW_MAX_CYCLES", "10")
    briefs, sends = [], []
    _recording_seam(monkeypatch, briefs, sends)
    (project / "app.py").write_text("VALUE = 2\n", encoding="utf-8")
    shared.git(project, "add", "app.py")
    earliest = run_review_change(ctx, subject="index", goal="Bump", scope="app.py")

    answer, _starts = _pending_then_answering_seat("t2", "invocation-comparison")
    _recording_seam(monkeypatch, briefs, sends, answer=answer)
    ask = dict(subject="index", goal="Bump", scope="app.py", author_questions=["The pending round"])
    pending = run_review_change(ctx, **ask)
    assert pending["state"] == "pending", pending
    dispatched = _comparison(ctx, pending)
    assert [anchor["review_record_id"] for anchor in dispatched["anchors"]] == [earliest["record_id"]]

    _recording_seam(monkeypatch, briefs, sends)
    later = run_review_change(ctx, subject="index", goal="Bump", scope="app.py", author_questions=["Another round"])
    assert later["state"] == "settled"
    assert [anchor["review_record_id"] for anchor in _comparison(ctx, later)["anchors"]] == [pending["record_id"]]

    _recording_seam(monkeypatch, briefs, sends, answer=answer)
    paid = len(sends)
    rejoined = run_review_change(ctx, **ask)
    assert (rejoined["record_id"], rejoined["state"]) == (pending["record_id"], "settled"), rejoined
    assert len(sends) == paid + 1 and sends[-1]["reconcile_only"] is True
    assert _comparison(ctx, rejoined) == dispatched


def test_commit_reviewed_compares_with_a_named_review_and_refuses_the_name_where_no_critic_runs(
        staged_body, tmp_path, monkeypatch):
    """The commit gate's own surface: a review another task made of these staged bytes,
    then one more line integrated. The panel is shown the change since that review; an
    author continuation and a paid collection, which send no new brief, refuse the name."""
    import json

    import ouroboros.review_substrate as substrate
    from ouroboros import review_ledger
    from ouroboros.tools import git
    from ouroboros.tools.registry import ToolContext
    from ouroboros.tools.review_change import run_review_change
    from tests import _contributor_packet_shared as shared
    from tests.test_review_change_end_to_end import _brief_text

    repo = Path(staged_body["repo"])
    shared.git(repo, "config", "user.name", "Comparison fixture")
    shared.git(repo, "config", "user.email", "comparison@example.invalid")
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "cyber_pro")
    briefs: list = []
    monkeypatch.setattr(substrate, "run_review_request", shared.golden_substrate(briefs))
    drive = tmp_path / "data"
    earlier = run_review_change(ToolContext(repo_dir=repo, drive_root=drive, task_id="earlier-task"),
                                root="system_repo", subject="index", goal="Ship the proposal", scope="the proposal")
    with (repo / "README.md").open("a", encoding="utf-8") as stream:
        stream.write("Integrated line.\n")
    shared.git(repo, "add", "README.md")
    ctx = ToolContext(repo_dir=repo, drive_root=drive, task_id="commit-task")
    briefs.clear()
    head = shared.git(repo, "rev-parse", "HEAD")

    text = git._commit_reviewed(ctx, "fix: proposal", skip_tests=True, skip_advisory_review=True,
                                goal="Ship the proposal", scope="the proposal", since_review=earlier["record_id"])

    reference = json.JSONDecoder().raw_decode(text[text.index('{"review_reference"'):])[0]["review_reference"]
    record = review_ledger.load_record(drive, reference["review_record_id"])
    assert record["brief"]["since_review"] == earlier["record_id"]
    comparison = record["brief"]["dispute_input"]["comparison"]
    [anchor] = comparison["anchors"]
    assert (anchor["review_record_id"], anchor["named"], anchor["relation"], anchor["changed"]) == (
        earlier["record_id"], True, "changed", ["M\tREADME.md"])
    from ouroboros.review_history import render_comparison
    assert briefs and all(_brief_text(brief).count(render_comparison(comparison)) == 1 for brief in briefs)
    continued = git._commit_reviewed(ctx, "fix: proposal", skip_tests=True, skip_advisory_review=True,
                                     goal="Ship the proposal", scope="the proposal", review_reference=reference,
                                     author_disposition={"disposition": "accepted", "rationale": "Read the panel."},
                                     since_review=earlier["record_id"])
    assert "since_review shapes a new wave's brief" in continued
    collected = git._commit_reviewed(ctx, "fix: proposal", review_reference=reference, since_review=earlier["record_id"])
    assert "cannot change its commit message, brief or panel" in collected
    assert shared.git(repo, "rev-parse", "HEAD") == head  # no commit: Cyber Pro hands every outcome back first


def test_equal_bytes_judged_as_another_subject_are_never_called_the_same_subject(tmp_path, monkeypatch):
    """The subject's identity is more than its tree: the same staged bytes reviewed against
    another base, through another surface or as another kind are named as such, with what
    differs, and never as the same subject (only a new question on that identity is)."""
    from ouroboros.review_history import render_comparison
    from ouroboros.tools.review_change import run_review_change
    from tests import _contributor_packet_shared as shared
    from tests.test_review_change_end_to_end import _foreign_project

    ctx, project = _foreign_project(tmp_path, monkeypatch)
    monkeypatch.setenv("OUROBOROS_REVIEW_MAX_CYCLES", "10")
    briefs, sends = [], []
    _recording_seam(monkeypatch, briefs, sends)
    first_base = shared.git(project, "rev-parse", "HEAD")
    (project / "notes.txt").write_text("an unrelated committed file\n", encoding="utf-8")
    shared.git(project, "add", "notes.txt")
    shared.git(project, "commit", "-q", "-m", "notes")
    (project / "app.py").write_text("VALUE = 2\n", encoding="utf-8")
    shared.git(project, "add", "app.py")
    ask = dict(goal="Bump", scope="app.py")

    at_head = run_review_change(ctx, subject="index", **ask)
    other_base = run_review_change(ctx, subject="index", base=first_base, **ask)
    looked = run_review_change(ctx, subject="index", surface="preflight", reviewers=["t1"], **ask)
    as_worktree = run_review_change(ctx, subject="worktree", **ask)

    expected = [(other_base, at_head, ["base"]), (looked, other_base, ["surface", "base"]),
                (as_worktree, looked, ["kind", "surface"])]
    trees = {result["subject"]["tree_sha"] for result, _earlier, _differs in expected}
    trees.add(at_head["subject"]["tree_sha"])
    assert len(trees) == 1  # one set of bytes throughout
    for result, earlier, differs in expected:
        [anchor] = _comparison(ctx, result)["anchors"]
        assert (anchor["review_record_id"], anchor["relation"], anchor["differs"]) == (
            earlier["record_id"], "same_tree_other_subject", differs)
        text = render_comparison(_comparison(ctx, result))
        assert f"SAME TREE, ANOTHER SUBJECT (differs: {', '.join(differs)})" in text and "SAME SUBJECT" not in text


def test_commit_reviewed_keeps_since_review_in_its_round_and_rejoins_with_the_frozen_comparison(
        staged_body, tmp_path, monkeypatch):
    """Through the real commit tool: a wave dispatched naming review X is still open. The same
    call naming another review Y is another intent and dispatches nothing; repeating the X
    call collects the open wave, and its record keeps the comparison its seats were sent."""
    import json

    from ouroboros import review_ledger
    from ouroboros.tools import git
    from ouroboros.tools.registry import ToolContext
    from ouroboros.tools.review_change import run_review_change
    from tests import _contributor_packet_shared as shared
    from tests.test_review_change_end_to_end import _pending_then_answering_seat

    repo = Path(staged_body["repo"])
    shared.git(repo, "config", "user.name", "Comparison fixture")
    shared.git(repo, "config", "user.email", "comparison@example.invalid")
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "cyber_pro")
    monkeypatch.setenv("OUROBOROS_REVIEW_MAX_CYCLES", "10")
    briefs, sends = [], []
    _recording_seam(monkeypatch, briefs, sends)
    drive = tmp_path / "data"
    named = [run_review_change(ToolContext(repo_dir=repo, drive_root=drive, task_id=f"earlier-{label}"),
                               root="system_repo", subject="index", goal=f"Ship {label}", scope="the proposal")["record_id"]
             for label in ("x", "y")]
    with (repo / "README.md").open("a", encoding="utf-8") as stream:
        stream.write("Integrated line.\n")
    shared.git(repo, "add", "README.md")
    answer, _starts = _pending_then_answering_seat("t2", "invocation-commit-since")
    _recording_seam(monkeypatch, briefs, sends, answer=answer)
    ctx = ToolContext(repo_dir=repo, drive_root=drive, task_id="commit-task")

    def commit(since):
        return git._commit_reviewed(ctx, "fix: proposal", skip_tests=True, skip_advisory_review=True,
                                    goal="Ship the proposal", scope="the proposal", since_review=since)

    first = commit(named[0])
    reference = json.JSONDecoder().raw_decode(first[first.index('{"review_reference"'):])[0]["review_reference"]
    record = review_ledger.load_record(drive, reference["review_record_id"])
    assert record["state"] == "pending" and record["brief"]["since_review"] == named[0], first
    dispatched = record["brief"]["dispute_input"]["comparison"]
    assert [anchor["review_record_id"] for anchor in dispatched["anchors"]] == [named[0]]
    paid = len(sends)

    other = commit(named[1])
    assert "this other intent waits; nothing was dispatched" in other and len(sends) == paid

    again = commit(named[0])
    assert len(sends) == paid + 1 and sends[-1]["reconcile_only"] is True, again
    settled = review_ledger.load_record(drive, reference["review_record_id"])
    assert settled["state"] == "settled" and settled["brief"]["since_review"] == named[0]
    assert settled["brief"]["dispute_input"]["comparison"] == dispatched
