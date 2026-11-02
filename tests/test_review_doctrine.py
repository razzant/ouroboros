"""A helper check is recorded evidence. It is not a review seat and not a PASS."""
import hashlib
import inspect
import json
from pathlib import Path

import pytest

from ouroboros import config, review_ledger as rl
from ouroboros.review_doctrine import (
    FACT_KIND, build_fact, classify_feedback, doctrine_result_lines, enrich_author_commit,
    history_author_fields, parse_subagent_check,
)
from ouroboros.review_history import review_dispute_history
from ouroboros.review_ledger import CouplingOutcome, reduce_verdict
from ouroboros.review_records import validate_author_disposition
from ouroboros.task_results import write_task_result
from ouroboros.tools import git
from tests.test_git_review_preflight_gate import candidate  # noqa: F401
from tests.test_review_ledger import BOTH, CRITICAL, _answer, _facts, _panel, _raw

REPO = Path(__file__).resolve().parents[1]
_CLAIM = {
    "reason": "The panel did not answer, so I checked with a helper.",
    "selection_rationale": "The helper reads this repository.",
    "report_refs": [{"task_id": "helper-1"}],
}


def _mode(monkeypatch, mode, enforcement):
    config.reset_runtime_mode_baseline_for_tests()
    config.initialize_runtime_mode_baseline(mode)
    monkeypatch.setenv("OUROBOROS_REVIEW_ENFORCEMENT", enforcement)
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", mode)
    monkeypatch.setenv("OUROBOROS_REVIEW_MAX_CYCLES", "unlimited")


def _prepare(ctx, monkeypatch):
    from ouroboros.mutation_attribution import capture_mutation_baseline

    ctx.branch_dev = git.run_cmd(["git", "branch", "--show-current"], cwd=ctx.repo_dir).strip()
    git.run_cmd(["git", "reset", "--hard", "HEAD"], cwd=ctx.repo_dir)
    (ctx.repo_dir / "VERSION").write_text("1.0.0\n", encoding="utf-8")
    git.run_cmd(["git", "add", "VERSION"], cwd=ctx.repo_dir)
    git.run_cmd(["git", "commit", "-m", "fixture version"], cwd=ctx.repo_dir)
    write_task_result(ctx.drive_root, ctx.task_id, "running")
    capture_mutation_baseline(
        ctx.drive_root, ctx.task_id, [{"surface_type": "system_repo", "host_root": str(ctx.repo_dir)}])
    (ctx.repo_dir / "change.py").write_text("value = 2\n", encoding="utf-8")
    monkeypatch.setattr(git, "_run_review_preflight_tests", lambda *_a, **_k: None)
    monkeypatch.setattr(git, "_post_commit_result", lambda *_a, **_k: None)
    monkeypatch.setattr(git, "_auto_push", lambda *_a, **_k: "")


def _install_reviewer(ctx, monkeypatch, rows):
    def reviewer(_ctx, message, **kw):
        from ouroboros.review_dispatch import invoke_review_paid_stamp

        invoke_review_paid_stamp(ctx._review_paid_stamp)
        ctx._last_triad_raw_results = rows
        ctx._last_review_critical_findings = []
        ctx._last_scope_raw_result = {}
        return None, CouplingOutcome(verdict="PASS", blocked=False, status="not_performed"), "", []

    monkeypatch.setattr(git, "_run_parallel_review", reviewer)


def _head(ctx):
    return git.run_cmd(["git", "rev-parse", "HEAD"], cwd=ctx.repo_dir).strip()


def _reference(result):
    return json.loads(result.split("\n", 1)[1])["review_reference"]


def _record(ctx, reference):
    return rl.load_record(rl.ledger_root(ctx), reference["review_record_id"])


def test_parse_rejects_an_empty_or_strange_claim_and_keeps_a_real_one():
    parsed = parse_subagent_check(_CLAIM)
    assert parsed["report_refs"] == [{"task_id": "helper-1"}]
    with pytest.raises(ValueError):
        parse_subagent_check({**_CLAIM, "report_refs": []})
    with pytest.raises(ValueError):
        parse_subagent_check({**_CLAIM, "extra": True})


def test_gaps_are_not_verification_and_a_real_digest_is(tmp_path, monkeypatch):
    body = "helper report bytes"
    write_task_result(tmp_path, "helper-1", "completed", result=body, model="helper-model",
                      subagent_id="scout", cost_kind="subscription", subject_hash="candidate-a")
    digest = hashlib.sha256(body.encode("utf-8")).hexdigest()
    bound = build_fact(parse_subagent_check(_CLAIM), subject_hash="candidate-a", review_reference={"surface": "commit"},
                       drive_root=tmp_path, actors=[], observed_at="t0")
    assert bound["reports"][0]["status"] == "bound"
    assert bound["reports"][0]["evidence_sha256"] == digest
    assert "gap" not in bound["reports"][0]
    other = build_fact(
        parse_subagent_check({**_CLAIM, "report_refs": [{"task_id": "helper-1", "subject_hash": "other"}]}),
        subject_hash="candidate-a", review_reference={}, drive_root=tmp_path, actors=[], observed_at="t1")
    assert other["reports"][0]["status"] == "gap" and other["reports"][0]["gap"] == "subject_mismatch"
    assert "evidence_sha256" not in other["reports"][0]
    mismatch = build_fact(
        parse_subagent_check({**_CLAIM, "report_refs": [{"task_id": "helper-1", "report_sha256": "ab" * 32}]}),
        subject_hash="candidate-a", review_reference={}, drive_root=tmp_path, actors=[], observed_at="t2")
    assert mismatch["reports"][0]["gap"] == "digest_mismatch"
    assert mismatch["reports"][0]["host_sha256"] == digest and "evidence_sha256" not in mismatch["reports"][0]
    missing = build_fact(
        parse_subagent_check({**_CLAIM, "report_refs": [{"task_id": "missing-helper"}]}),
        subject_hash="candidate-a", review_reference={}, drive_root=tmp_path, actors=[], observed_at="t3")
    assert missing["reports"][0]["gap"] == "report_unreadable"
    stored = {"disposition": "accepted", "rationale": "r", "subject_hash": "candidate-a", FACT_KIND: bound}
    assert validate_author_disposition(stored, subject_hash="candidate-a")[FACT_KIND]["reports"][0]["evidence_sha256"] == digest
    assert validate_author_disposition(stored, subject_hash="candidate-b") is None
    reloaded = json.loads(json.dumps(bound))
    assert reloaded["reports"][0]["evidence_sha256"] == digest


@pytest.mark.parametrize("top,nested,claim,identity,status", [
    ("", "scout", "scout", "recorded", "bound"),
    ("", "actual", "claimed", "mismatch", "bound"),
    ("top", "nested", "nested", "mismatch", "bound"),
    ("", "", "claimed", "unknown", "gap"),
    ("", "", "", "unknown", "gap"),
])
def test_report_actor_comes_only_from_retained_identity(tmp_path, top, nested, claim, identity, status):
    write_task_result(tmp_path, "helper-1", "completed", result="report", subagent_id=top,
                      configured_subagent={"selected_subagent_id": nested})
    ref = {"task_id": "helper-1", **({"subagent_id": claim} if claim else {})}
    fact = build_fact(parse_subagent_check({**_CLAIM, "report_refs": [ref]}), subject_hash="s",
                      review_reference={}, drive_root=tmp_path, actors=[], observed_at="t")
    report = fact["reports"][0]
    assert report["actor"]["subagent_id"] == (top or nested)
    assert report["ref"] == ref
    assert (report["actor_identity"], report["status"]) == (identity, status)
    stored = {"disposition": "accepted", "rationale": "r", "subject_hash": "s", FACT_KIND: fact}
    assert validate_author_disposition(stored, subject_hash="s")[FACT_KIND] == fact


def test_billing_label_does_not_grant_review_credit(monkeypatch):
    rows = {}

    def fake(_drive, task_id, **_kwargs):
        return rows[task_id]

    monkeypatch.setattr("ouroboros.task_results.load_task_result", fake)
    facts = []
    for label in ("subscription", "measured", "unknown", "priced"):
        rows["helper-1"] = {"task_id": "helper-1", "result": "same report", "model": "m",
                            "cost_kind": label, "subagent_id": "scout"}
        facts.append(build_fact(parse_subagent_check(_CLAIM), subject_hash="s", review_reference={},
                                drive_root="unused", actors=[], observed_at="t"))
    assert {fact["reports"][0]["status"] for fact in facts} == {"bound"}
    assert len({fact["reports"][0]["evidence_sha256"] for fact in facts}) == 1
    assert [fact["reports"][0]["actor"]["billing"] for fact in facts] == [
        "subscription", "measured", "unknown", "unknown"]
    panel = _panel()
    poisoned = [{**row, FACT_KIND: facts[0]} for row in panel]
    assert reduce_verdict(panel)["aggregate"] == reduce_verdict(poisoned)["aggregate"] == "PASS"
    failing = _panel()
    failing[1]["answers"] = {"change": _answer("change", "FAIL", [{"severity": "critical", "item": "x", "verdict": "FAIL"}])}
    poisoned_fail = [{**row, FACT_KIND: facts[0]} for row in failing]
    assert reduce_verdict(failing)["aggregate"] == reduce_verdict(poisoned_fail)["aggregate"] == "FAIL"
    assert FACT_KIND not in inspect.getsource(reduce_verdict)


def test_replay_keeps_the_decision_snapshot_when_later_replies_arrive():
    live = [{"slot_id": "critic", "status": "pending", "operation_state": "in_flight"}]
    author = {"disposition": "accepted", "rationale": "r", "subject_hash": "s", "subagent_check": parse_subagent_check(_CLAIM)}
    first = enrich_author_commit(author, actors=live, prior_decision=None, record_id="rl-1", revision=1,
                                 drive_root=None, cyber=True, observed_at="t0", review_reference={"surface": "commit"})
    assert first["evidence_at_decision"]["feedback_at_decision"] == "none_received"
    assert first[FACT_KIND]["reports"][0]["gap"] == "report_unreadable"
    answered = [{"slot_id": "critic", "status": "responded", "raw_text": "late", "ok": True}]
    second = enrich_author_commit(author, actors=answered, prior_decision=first, record_id="rl-1", revision=2,
                                  drive_root=None, cyber=True, observed_at="t1", review_reference={"surface": "commit"})
    assert second["evidence_at_decision"] == first["evidence_at_decision"]
    assert second[FACT_KIND] == first[FACT_KIND]
    moved = enrich_author_commit({**author, "subject_hash": "other"}, actors=answered, prior_decision=first,
                                 record_id="rl-1", revision=2, drive_root=None, cyber=True, observed_at="t2",
                                 review_reference={"surface": "commit"})
    assert moved["evidence_at_decision"]["subject_hash"] == "other"
    assert moved["evidence_at_decision"]["feedback_at_decision"] == "received"


def test_a_late_finding_still_fails_and_keeps_the_fact(tmp_path):
    pending = [
        _raw("s1", "openai/gpt-5", parts=BOTH, raw_text="two-part answer"),
        _raw("s2", "anthropic/claude-x", "pending", operation_state="in_flight", late_result_pending=True, parts=BOTH),
        _raw("s3", "google/gemini", "pending", operation_state="in_flight", late_result_pending=True, parts=BOTH),
    ]
    record = rl.build_commit_gate_record(_facts(pending, pending=True), drive_root=tmp_path)
    rl.write_record(tmp_path, record)
    fact = build_fact(parse_subagent_check(_CLAIM), subject_hash="fp", review_reference={"surface": "commit"},
                      drive_root=tmp_path, actors=pending, observed_at="t0")
    noted = rl.note_author_decision(tmp_path, record.record_id, {
        "disposition": "accepted", "rationale": "known gap", "subject_hash": "fp", FACT_KIND: fact})
    assert noted["author_decision"][FACT_KIND]["kind"] == FACT_KIND
    failing = _raw("s3", "google/gemini", parts=BOTH, answers={"change": _answer("change", "FAIL", findings=[CRITICAL])})
    settled = rl.settle_from_collected(
        tmp_path, record.record_id, [_raw("s2", "anthropic/claude-x", parts=BOTH), failing])
    assert settled["verdict"]["aggregate"] == "FAIL"
    assert settled["author_decision"][FACT_KIND]["kind"] == FACT_KIND
    assert FACT_KIND not in json.dumps(settled["rows"])
    again = rl.note_commit_landing(tmp_path, record.record_id, {
        "kind": "commit_landing", "landed": True, "subject_hash": "fp",
        "pre_review_fingerprint": "fp", "post_review_fingerprint": "fp"})
    assert again["author_decision"]["landing"]["landed"] is True
    assert again["author_decision"][FACT_KIND]["observed_at"] == "t0"
    assert "evidence_at_decision" not in again["author_decision"]
    quiet = rl.note_commit_landing(tmp_path, record.record_id, again["author_decision"]["landing"])
    assert quiet["revision"] == again["revision"]


def test_history_names_the_fact_only_when_it_was_claimed():
    plain = {"disposition": "accepted", "rationale": "ordinary"}
    claimed = {**plain, FACT_KIND: {"kind": FACT_KIND, "reason": "checked"}}
    assert history_author_fields(plain)["remark"] == "Explicit author decision"
    assert FACT_KIND not in history_author_fields(plain)
    assert history_author_fields(claimed)["remark"] == "Check replaced by subagents"
    assert doctrine_result_lines(claimed) == []
    landed = {**claimed, "landing": {"landed": True}, "evidence_at_decision": {"feedback_at_decision": "none_received"}}
    lines = doctrine_result_lines(landed)
    assert any(line.startswith("review_replaced_by_subagents:") for line in lines)
    assert any(line.startswith("evidence_at_decision: none_received") for line in lines)
    assert doctrine_result_lines({"landing": {"landed": True}})[-1].startswith("landing:")


def test_prompt_and_schema_do_not_depend_on_the_fact(monkeypatch):
    text = (REPO / "prompts" / "SYSTEM.md").read_text(encoding="utf-8")
    # The owner-approved sentence (22.10) names the replacement check in words, never by the fact kind.
    assert "A helper I schedule never counts as a seat." in text
    assert "A helper I schedule is never a review seat." not in text
    assert FACT_KIND not in text
    from ouroboros.subagent_runtime import review_facts_block

    _mode(monkeypatch, "pro", "advisory")
    advisory = review_facts_block()
    config.reset_runtime_mode_baseline_for_tests()
    config.initialize_runtime_mode_baseline("cyber_pro")
    monkeypatch.setenv("OUROBOROS_REVIEW_ENFORCEMENT", "blocking")
    cyber = review_facts_block()
    for block in (advisory, cyber):
        assert FACT_KIND not in block and "subagent_check" not in block

    def schema():
        tool = next(entry for entry in git.get_tools() if entry.name == "commit_reviewed")
        return json.dumps(tool.schema["parameters"]["properties"]["author_disposition"], sort_keys=True)

    advisory_schema = schema()
    assert schema() == advisory_schema
    body = json.loads(advisory_schema)
    assert "subagent_check" in body["properties"] and "subagent_check" not in body["required"]
    config.reset_runtime_mode_baseline_for_tests()


def test_feedback_labels_describe_seats_without_counting_them():
    live = [{"slot_id": "critic", "status": "pending", "operation_state": "in_flight", "error": "still starting"}]
    assert classify_feedback(live) == "none_received"
    errored = [{"slot_id": "critic", "status": "error", "error": "down", "operation_state": "settled"}]
    assert classify_feedback(errored) == "unavailable"
    answered = [{"slot_id": "critic", "status": "responded", "raw_text": "[]"}]
    assert classify_feedback(answered) == "received"
    assert classify_feedback([]) == "unknown"
    assert classify_feedback([{"verdict": "PASS", "status": "responded", "seats": []}]) == "unknown"


def test_blocking_refuses_a_helper_continuation_and_a_clean_pass_still_lands(candidate, monkeypatch):  # noqa: F811
    ctx = candidate
    _mode(monkeypatch, "pro", "blocking")
    _prepare(ctx, monkeypatch)
    before = _head(ctx)
    refused = git._commit_reviewed(
        ctx, "Fix amount", review_reference={"surface": "commit"},
        author_disposition={"disposition": "accepted", "rationale": "The helper checked it.", "subagent_check": _CLAIM})
    assert "requires Advisory" in refused
    assert _head(ctx) == before and FACT_KIND not in refused
    _install_reviewer(ctx, monkeypatch, _panel())
    landed = git._commit_reviewed(ctx, "Fix amount", skip_advisory_review=True)
    assert _head(ctx) != before, landed
    assert FACT_KIND not in landed and "landing:" not in landed


def test_advisory_without_feedback_refuses_and_a_received_error_can_land(candidate, monkeypatch):  # noqa: F811
    ctx = candidate
    _mode(monkeypatch, "pro", "advisory")
    _prepare(ctx, monkeypatch)
    live = [{"slot_id": "critic", "status": "pending", "operation_state": "in_flight"}]
    _install_reviewer(ctx, monkeypatch, live)
    before = _head(ctx)
    first = git._commit_reviewed(ctx, "Fix amount", skip_advisory_review=True)
    assert "returned before commit" in first and _head(ctx) == before
    reference = _reference(first)
    refused = git._commit_reviewed(
        ctx, "Fix amount", review_reference=reference,
        author_disposition={"disposition": "accepted", "rationale": "No reply yet.", "subagent_check": _CLAIM})
    assert "received feedback" in refused or "still running" in refused
    assert _head(ctx) == before
    assert not (_record(ctx, reference) or {}).get("author_decision")


def test_advisory_error_continuation_records_a_claim_and_a_plain_one_does_not(candidate, monkeypatch):  # noqa: F811
    ctx = candidate
    _mode(monkeypatch, "pro", "advisory")
    _prepare(ctx, monkeypatch)
    errored = [{"slot_id": "critic", "status": "error", "error": "provider down", "operation_state": "settled"}]
    _install_reviewer(ctx, monkeypatch, errored)
    before = _head(ctx)
    opened = git._commit_reviewed(ctx, "Fix amount", skip_advisory_review=True)
    assert "returned before commit" in opened and _head(ctx) == before
    opened_ref = _reference(opened)
    write_task_result(ctx.drive_root, "helper-1", "completed", result="checked value = 3\n", model="helper-model",
                      subagent_id="scout", cost_kind="measured")
    claimed = git._commit_reviewed(
        ctx, "Fix amount", review_reference=opened_ref,
        author_disposition={"disposition": "accepted", "rationale": "I read the error and the helper report.",
                            "subagent_check": _CLAIM})
    assert _head(ctx) != before, claimed
    record = _record(ctx, opened_ref)
    assert record["verdict"]["aggregate"] != "PASS"
    assert record["author_decision"][FACT_KIND]["kind"] == FACT_KIND
    assert record["author_decision"][FACT_KIND]["reports"][0]["status"] == "bound"
    assert record["author_decision"]["landing"]["landed"] is True
    assert "evidence_at_decision" not in record["author_decision"]
    assert "review_replaced_by_subagents:" in claimed
    history = review_dispute_history(drive_root=ctx.drive_root, repo_root=ctx.repo_dir, task_id=ctx.task_id)
    assert any(row.get("remark") == "Check replaced by subagents" and FACT_KIND in row
               for row in history["decision_rows"])


def test_advisory_continuation_without_a_claim_lands_without_the_fact(candidate, monkeypatch):  # noqa: F811
    ctx = candidate
    _mode(monkeypatch, "pro", "advisory")
    _prepare(ctx, monkeypatch)
    errored = [{"slot_id": "critic", "status": "error", "error": "provider down", "operation_state": "settled"}]
    _install_reviewer(ctx, monkeypatch, errored)
    before = _head(ctx)
    opened = git._commit_reviewed(ctx, "Fix amount", skip_advisory_review=True)
    plain_ref = _reference(opened)
    plain = git._commit_reviewed(
        ctx, "Fix amount", review_reference=plain_ref,
        author_disposition={"disposition": "accepted", "rationale": "I read the error and continue without a helper."})
    assert _head(ctx) != before, plain
    plain_record = _record(ctx, plain_ref)
    assert FACT_KIND not in plain_record["author_decision"]
    assert plain_record["author_decision"]["landing"]["landed"] is True
    assert FACT_KIND not in plain


def test_cyber_records_the_snapshot_before_any_reply_and_a_failed_git_is_not_a_landing(candidate, monkeypatch):  # noqa: F811
    ctx = candidate
    _mode(monkeypatch, "cyber_pro", "blocking")
    _prepare(ctx, monkeypatch)
    live = [{"slot_id": "critic", "status": "pending", "operation_state": "in_flight"}]
    _install_reviewer(ctx, monkeypatch, live)
    before = _head(ctx)
    first = git._commit_reviewed(ctx, "Fix amount", skip_advisory_review=True)
    assert "returned before commit" in first and _head(ctx) == before, first
    reference = _reference(first)
    write_task_result(ctx.drive_root, "helper-1", "completed", result="checked before any reply", model="helper-model",
                      subagent_id="scout", cost_kind="subscription")
    real = git.run_cmd
    failed = {"done": False}

    def fail_once(cmd, cwd=None, timeout=None):
        if list(cmd[:2]) == ["git", "commit"] and not failed["done"]:
            failed["done"] = True
            raise RuntimeError("commit refused")
        return real(cmd, cwd=cwd, timeout=timeout)

    monkeypatch.setattr(git, "run_cmd", fail_once)
    broken = git._commit_reviewed(
        ctx, "Fix amount", review_reference=reference,
        author_disposition={"disposition": "partial", "rationale": "No reply yet; the helper checked these bytes.",
                            "subagent_check": _CLAIM})
    assert "GIT_ERROR" in broken and _head(ctx) == before, broken
    assert "landing:" not in broken
    record = _record(ctx, reference)
    snapshot = record["author_decision"]["evidence_at_decision"]
    assert snapshot["feedback_at_decision"] == "none_received"
    assert "landing" not in record["author_decision"]
    assert record["author_decision"][FACT_KIND]["reports"][0]["status"] == "bound"
    digest = record["author_decision"][FACT_KIND]["reports"][0]["evidence_sha256"]
    write_task_result(ctx.drive_root, "helper-1", "completed", result="changed after the decision", model="helper-model",
                      subagent_id="scout", cost_kind="subscription")
    from ouroboros.review_state import update_state

    def _mutate(state):
        for row in state.attempts:
            if row.attempt == reference["attempt"] and row.phase == "review_only":
                row.triad_raw_results = [{"slot_id": "critic", "status": "responded", "raw_text": "late", "ok": True}]
        return None

    update_state(ctx.drive_root, _mutate)
    landed = git._commit_reviewed(
        ctx, "Fix amount", review_reference=reference,
        author_disposition={"disposition": "partial", "rationale": "No reply yet; the helper checked these bytes.",
                            "subagent_check": _CLAIM})
    assert _head(ctx) != before, landed
    finished = _record(ctx, reference)
    assert finished["author_decision"]["evidence_at_decision"]["feedback_at_decision"] == "none_received"
    assert finished["author_decision"]["evidence_at_decision"]["observed_at"] == snapshot["observed_at"]
    assert finished["author_decision"][FACT_KIND]["reports"][0]["evidence_sha256"] == digest
    assert finished["author_decision"]["landing"]["landed"] is True
    assert finished["verdict"]["aggregate"] != "PASS"
    assert "evidence_at_decision: none_received" in landed
