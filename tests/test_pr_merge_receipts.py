"""#1333: ``pr_merge`` keeps reviewed, current and merged identities apart.

Drives the real tool handler over a stateful fake ``gh`` (no network) that answers the one synchronous REST
merge request the way ``gh api`` does: the intent row lands before the merge effect, a missing review is loud
but never a lock, queued/auto-merge is not "merged", an unknown outcome is only read back by a plain repeat, a
failed publication never repeats the merge, and the public PR-body block carries no private identifiers while
every unrelated byte of the body survives.
"""

from __future__ import annotations

import json
import subprocess
from types import SimpleNamespace

import pytest

from ouroboros import merge_receipts
from ouroboros.task_results import STATUS_COMPLETED, STATUS_RUNNING, load_task_result, write_task_result
from ouroboros.tools import github
from tests._cancel_intents_shared import DeliveryBus

REAL_GH_RUN = github._gh_run

HEAD = "a" * 40
BASE = "b" * 40
MERGE = "c" * 40
TREE = "d" * 40
URL = "https://github.com/octo/demo/pull/7"
ROUTE = "repos/octo/demo/pulls/7/merge"
MERGED = json.dumps({"sha": MERGE, "merged": True, "message": "Pull Request successfully merged"})
NOT_MERGEABLE = "⚠️ GH_ERROR: gh: Pull Request is not mergeable (HTTP 405)"  # gh api's own stderr shape, exit 1
CHECKLIST = json.dumps([{"item": item, "verdict": "PASS", "severity": "advisory",
                         "reason": f"Checked {item} against the touched module and its tests."}
                        for item in sorted(__import__("ouroboros.tools.scope_review_contract",
                                                      fromlist=["x"]).SCOPE_REQUIRED_ITEMS)])


def sent(call):
    """The one merge request ``pr_merge`` sends: a ``gh api`` PUT of the PR's merge endpoint."""
    return call[:1] == ["api"] and len(call) > 1 and call[1].endswith("/merge") and "PUT" in call


class FakeGh:
    """A PR that merges (or not) the way GitHub answers; records every call."""

    def __init__(self, drive_root, *, merge="ok", body="Summary of the change.\n\nOwner notes stay here."):
        self.drive_root, self.merge, self.calls = drive_root, merge, []
        self.pr = {"number": 7, "state": "OPEN", "url": URL, "title": "demo", "body": body, "comments": [],
                   "headRefOid": HEAD, "baseRefName": "main", "baseRefOid": BASE, "mergeStateStatus": "CLEAN",
                   "isDraft": False, "autoMergeRequest": None, "mergeCommit": None, "mergedAt": None}
        self.fail_view_after_merge = False
        self.fail_edit = False
        self.intent_seen_at_merge = None
        self.merge_inputs, self.merge_payloads = [], []  # each request's commit_message, and its whole body

    def __call__(self, args, ctx, timeout=30, input_data=None, *, repo=github._GENERIC_TRANSPORT):
        self.calls.append(list(args))
        if args[:2] == ["pr", "view"]:
            if self.fail_view_after_merge and any(sent(c) for c in self.calls):
                return github.GhResult(False, "⚠️ GH_ERROR: boom", 1, 502, "exit")
            return github.GhResult(True, json.dumps(self.pr), 0, None, "")
        if sent(args):
            payload = json.loads(input_data)
            self.merge_payloads.append(payload)
            self.merge_inputs.append(payload.get("commit_message"))
            row = load_task_result(self.drive_root, "merge-task") or {}
            self.intent_seen_at_merge = [r.get("state") for r in row.get("merge_receipts") or []]
            if self.merge == "ok":
                self.pr.update(state="MERGED", mergeCommit={"oid": MERGE}, mergedAt="2026-09-26T00:00:00Z")
                return github.GhResult(True, MERGED, 0, None, "")
            if self.merge == "accepted_open":  # GitHub answered merged; its readback does not show it yet
                return github.GhResult(True, MERGED, 0, None, "")
            if self.merge == "timeout_merged":
                self.pr.update(state="MERGED", mergeCommit={"oid": MERGE})
                return github.GhResult(False, "⚠️ GH_TIMEOUT: exceeded 120s.", None, None, "timeout")
            if self.merge == "timeout_open":
                return github.GhResult(False, "⚠️ GH_TIMEOUT: exceeded 120s.", None, None, "timeout")
            if self.merge in ("queued", "timeout_queued"):  # another actor's auto-merge shows; the answer was lost
                self.pr.update(autoMergeRequest={"enabledAt": "now"})
                return github.GhResult(False, "⚠️ GH_TIMEOUT: exceeded 120s.", None, None, "timeout")
            return github.GhResult(False, NOT_MERGEABLE, 1, 405, "exit")
        if args[:2] == ["pr", "edit"]:
            return github.GhResult(False, "GraphQL: The 'login' field requires 'read:org' scope", 1, None, "exit")
        if args[:1] == ["api"] and "PATCH" in args:
            if self.fail_edit:
                return github.GhResult(False, "⚠️ GH_ERROR: forbidden", 1, 403, "exit")
            self.pr["body"] = json.loads(input_data)["body"]
            return github.GhResult(True, "", 0, None, "")
        if args[:1] == ["api"]:
            return github.GhResult(True, json.dumps({"commit": {"tree": {"sha": TREE}},
                                                     "parents": [{"sha": BASE}, {"sha": HEAD}]}), 0, None, "")
        raise AssertionError(f"unexpected gh call {args}")


@pytest.fixture
def world(tmp_path, monkeypatch):
    write_task_result(tmp_path, "merge-task", STATUS_RUNNING, root_task_id="merge-task", delegation_role="root")
    write_task_result(tmp_path, "review-1", STATUS_COMPLETED, root_task_id="merge-task", result="PASS")
    fake = FakeGh(tmp_path)
    monkeypatch.setattr(github, "_gh_run", fake)
    rows = []
    ctx = SimpleNamespace(task_id="merge-task", drive_root=tmp_path, task_metadata={}, repo_dir=tmp_path,
                          current_chat_id=7, pending_events=[],
                          emit_progress_fn=lambda text, **kw: rows.append((text, kw)))
    return SimpleNamespace(ctx=ctx, gh=fake, rows=rows, root=tmp_path)


def _merge(world, **kw):
    args = {"number": 7, "expected_head_sha": HEAD, "method": "squash", **kw}
    return github._pr_merge(world.ctx, **args)


def _receipts(world):
    return (load_task_result(world.root, "merge-task") or {}).get("merge_receipts") or []


def test_a_reviewed_merge_records_intent_first_and_publishes_one_receipt(world):
    out = _merge(world, review_task_ids=["review-1"], reviewed_head_sha=HEAD, reviewed_base_sha=BASE,
                 review_verdict="PASS")
    assert world.gh.intent_seen_at_merge == ["intent_recorded"]  # written before the effect
    # One synchronous REST request to the PR GitHub's readback named, conditioned on the expected head.
    call = next(c for c in world.gh.calls if sent(c))
    assert call == ["api", ROUTE, "--hostname", "github.com", "--method", "PUT", "--input", "-"]
    (payload,) = world.gh.merge_payloads
    assert (payload["sha"], payload["merge_method"]) == (HEAD, "squash") and set(payload) == {
        "sha", "merge_method", "commit_message"}
    assert "Ouroboros-Qualification:" in world.gh.merge_inputs[0]
    assert not any("--auto" in c or "--admin" in c or c[:2] == ["pr", "merge"] for c in world.gh.calls)
    (receipt,) = _receipts(world)
    assert receipt["outcome"] == {**receipt["outcome"], "status": "merged", "merge_sha": MERGE, "merge_tree": TREE,
                                  "merge_parents": [BASE, HEAD], "attribution": "this_call"}
    # A declaration beside a completed named task is the author's statement, not observed coverage.
    assert receipt["coverage"]["status"] == "unknown"
    assert receipt["coverage"]["gaps"] == ["no_host_observed_review_record"]
    assert receipt["review"]["host_observed"][0]["status"] == "completed"
    assert receipt["publication"]["body"] == {"status": "published"}
    assert receipt["publication"]["card"]["status"] == "owed"
    assert world.ctx.pending_events[0]["progress_meta"]["card_row_id"] == f"merge-receipt:{receipt['receipt_id']}"
    assert world.rows == []  # callback return has no delivery authority
    assert out.startswith("⚠️ PR #7 merge: merged")
    body = world.gh.pr["body"]
    assert body.startswith("Summary of the change.\n\nOwner notes stay here.")  # unrelated text kept
    assert "review-1" not in body and str(world.root) not in body and "merge-task" not in body


def test_an_unreviewed_merge_is_loud_and_still_merges(world):
    out = _merge(world)
    (receipt,) = _receipts(world)
    assert receipt["outcome"]["status"] == "merged" and receipt["coverage"]["status"] == "not_declared"
    assert out.startswith("⚠️") and "no_review_declared" in out
    assert "No review was declared" in world.gh.pr["body"]
    # The card's head names that known state, never an unknown coverage (DESIGN principle 7).
    label = merge_receipts.coverage_reading(receipt)[0]
    assert "unknown" not in label and label == merge_receipts._UNDECLARED
    assert merge_receipts.card_row_text(receipt).partition("\n")[0].endswith(f" · {label}")
    assert "\n" not in merge_receipts.card_row_text(receipt)  # the label says it all: no second line repeats it


@pytest.mark.parametrize(("kw", "status", "label"), [
    ({"reviewed_head_sha": "e" * 40}, "changes_after_review", "_EARLIER"),
    ({"reviewed_head_sha": HEAD, "review_scope": "delta"}, "delta_only", "_LATEST_ONLY"),
    ({"review_task_ids": ["missing-review"]}, "unknown", "_UNKNOWN"),
])
def test_coverage_never_claims_more_than_was_declared(world, kw, status, label):
    _merge(world, **kw)
    (receipt,) = _receipts(world)
    assert receipt["coverage"]["status"] == status
    # The card's head names a known coverage, a partial one too; only one nothing resolves reads as unknown.
    said = merge_receipts.coverage_reading(receipt)[0]
    assert ("unknown" in said) is (status == "unknown") and said == getattr(merge_receipts, label)
    assert merge_receipts.card_row_text(receipt).partition("\n")[0].endswith(f" · {said}")
    # A second line only where it says more than the label: why the coverage is unknown, or that it was earlier code.
    assert ("\n" in merge_receipts.card_row_text(receipt)) is (status != "delta_only")


def test_a_moved_head_or_closed_pr_merges_nothing_and_writes_nothing(world):
    world.gh.pr["headRefOid"] = "f" * 40
    assert "head_moved" in _merge(world)
    world.gh.pr.update(headRefOid=HEAD, state="CLOSED")
    assert "pr_not_open" in _merge(world)
    assert not any(sent(c) for c in world.gh.calls) and _receipts(world) == []


@pytest.mark.parametrize(("merge", "outcome"), [
    ("queued", {"status": "queued", "reason": "auto_merge_pending"}),
    # A merge answer the readback does not show yet is neither merged nor a queue: the request never enqueues.
    ("accepted_open", {"status": "unknown", "reason": "merge_answered_pr_still_open"}),
])
def test_queued_is_not_merged(world, merge, outcome):
    world.gh.merge = merge
    out = _merge(world, reviewed_head_sha=HEAD)
    (receipt,) = _receipts(world)
    assert receipt["outcome"] == outcome
    assert "continue_receipt_id" not in out  # neither a queue nor an answered merge may be followed by a new request


QUEUED_CLOSED = "queued earlier; PR now CLOSED, auto-merge off; effect unconfirmed"


def _merges(world):
    return sum(sent(c) for c in world.gh.calls)


@pytest.mark.parametrize(("merge", "reason"), [("queued", "auto_merge_pending")])
def test_a_closed_pr_is_the_latest_readback_beside_the_earlier_queued_request(world, merge, reason):
    """GitHub's newest answer (CLOSED, head and base moved) is the observation and the coverage
    subject; the queued request stays the outcome, observation-only, worded as earlier."""
    world.gh.merge = merge
    _merge(world, reviewed_head_sha=HEAD, reviewed_base_sha=BASE)
    (queued,) = _receipts(world)
    moved_head, moved_base = "e" * 40, "f" * 40
    world.gh.pr.update(state="CLOSED", autoMergeRequest=None, headRefOid=moved_head, baseRefOid=moved_base)
    out = _merge(world, reviewed_head_sha=HEAD, reviewed_base_sha=BASE)
    (receipt,) = _receipts(world)
    assert receipt["receipt_id"] == queued["receipt_id"] and receipt["outcome"] == {"status": "queued", "reason": reason}
    assert receipt["observed_after"] == {**queued["observed_after"], "state": "CLOSED", "auto_merge": False,
                                         "head_sha": moved_head, "base_sha": moved_base}
    assert receipt["coverage"]["status"] == "changes_after_review"
    assert "base_changed_since_review" in receipt["coverage"]["gaps"]
    assert receipt["revision"] == queued["revision"] + 1
    assert out.startswith(f"⚠️ PR #7 merge: {QUEUED_CLOSED}; review coverage changes_after_review")
    assert f"Earlier GitHub readback: {reason}." in out and "\nGitHub readback:" not in out
    assert "sent no new merge request" in out
    body = world.gh.pr["body"]
    assert f"- Outcome: **queued** {QUEUED_CLOSED.removeprefix('queued ')}\n" in body
    assert f"- Observed PR head `{moved_head}`, base `{moved_base}`" in body
    assert world.ctx.pending_events[-1]["text"] == merge_receipts.card_row_text(receipt)
    # Nothing merged: the card's coverage detail names the head this request named, never "what merged", and its
    # head ends on the earlier-version label rather than the sentence.
    head, _, detail = merge_receipts.card_row_text(receipt).partition("\n")
    assert head.endswith(f" · {merge_receipts._EARLIER}") and detail not in head
    assert "this head" in detail and "what merged" not in head + detail
    assert _merges(world) == 1


def test_the_same_closed_readback_again_changes_no_fact_revision_or_delivery(world):
    world.gh.merge = "queued"
    _merge(world)
    world.gh.pr.update(state="CLOSED", autoMergeRequest=None)
    _merge(world)
    (closed,) = _receipts(world)
    patches = sum(c[:1] == ["api"] and "PATCH" in c for c in world.gh.calls)
    assert _merge(world).startswith(f"⚠️ PR #7 merge: {QUEUED_CLOSED};")
    (again,) = _receipts(world)
    assert {k: v for k, v in again.items() if k != "updated_at"} == {k: v for k, v in closed.items() if k != "updated_at"}
    assert sum(c[:1] == ["api"] and "PATCH" in c for c in world.gh.calls) == patches
    assert _merges(world) == 1


def test_an_unconfirmed_request_whose_auto_merge_went_away_reads_as_earlier(world):
    world.gh.merge = "timeout_queued"
    _merge(world)
    assert _receipts(world)[0]["outcome"] == {"status": "queued", "reason": "auto_merge_pending"}
    world.gh.pr.update(autoMergeRequest=None)
    out = _merge(world)
    (receipt,) = _receipts(world)
    assert receipt["outcome"] == {"status": "queued", "reason": "auto_merge_pending"}
    assert (receipt["observed_after"]["state"], receipt["observed_after"]["auto_merge"]) == ("OPEN", False)
    assert out.startswith("⚠️ PR #7 merge: queued earlier; PR now OPEN, auto-merge off; effect unconfirmed;")
    assert "Earlier GitHub readback: auto_merge_pending." in out and "Earlier merge request: failure=timeout" in out
    assert "- Outcome: **queued** earlier; PR now OPEN, auto-merge off; effect unconfirmed\n" in world.gh.pr["body"]
    assert _merges(world) == 1


def test_a_reopened_pr_reads_queued_again_without_a_resend(world):
    world.gh.merge = "queued"
    _merge(world)
    world.gh.pr.update(state="CLOSED", autoMergeRequest=None)
    assert QUEUED_CLOSED in _merge(world)
    world.gh.pr.update(state="OPEN", autoMergeRequest={"enabledAt": "again"})
    out = _merge(world)
    (receipt,) = _receipts(world)
    assert receipt["outcome"] == {"status": "queued", "reason": "auto_merge_pending"}
    assert receipt["observed_after"]["state"] == "OPEN"
    assert out.startswith("⚠️ PR #7 merge: queued; review coverage") and "\nGitHub readback: auto_merge_pending." in out
    assert "- Outcome: **queued**\n" in world.gh.pr["body"]
    assert _merges(world) == 1


def test_an_older_receipt_gh_accepted_into_a_queue_keeps_being_observed_and_cannot_be_continued(world):
    """A receipt the earlier gh transport recorded (exit 0 without a merge answer, PR still open) stays queued: its
    request may still land from the queue, so a plain call reads it back and no decision may follow it."""
    merge_receipts.write_receipt(world.root, "merge-task", {
        "schema": 1, "receipt_id": "legacy", "state": "settled", "repo": {"url": URL}, "number": 7,
        "launch_operation_ids": ["op-old"], "requested": {"method": "squash", "expected_head_sha": HEAD},
        "observed_before": {"state": "OPEN", "head_sha": HEAD, "base_ref": "main"}, "review": {}, "coverage": {},
        "effect": {"ok": True, "exit_code": 0, "http_status": None, "failure": "", "output": "queued"},
        "outcome": {"status": "queued", "reason": "accepted_still_open"}})
    out = _merge(world)
    (receipt,) = _receipts(world)
    assert receipt["outcome"] == {"status": "queued", "reason": "accepted_still_open"} and "sent no new merge" in out
    world.gh.pr.update(state="CLOSED")
    assert "queued earlier; PR now CLOSED, auto-merge off; request accepted, outcome unresolved" in _merge(world)
    world.gh.pr.update(state="OPEN")
    refused = _merge(world, continue_receipt_id="legacy")
    assert "continuation_refused" in refused and "its outcome is queued" in refused
    assert _merges(world) == 0


def test_a_confirmed_merge_keeps_its_readback_against_a_later_closed_answer(world):
    """MERGED is terminal on GitHub, so a later CLOSED answer is a lagging replica."""
    _merge(world, reviewed_head_sha=HEAD, reviewed_base_sha=BASE)
    (merged,) = _receipts(world)
    world.gh.pr.update(state="CLOSED", mergeCommit=None, mergedAt=None)
    out = _merge(world, reviewed_head_sha=HEAD, reviewed_base_sha=BASE)
    (receipt,) = _receipts(world)
    assert all(receipt[key] == merged[key] for key in (
        "outcome", "observed_after", "coverage", "effect", "revision", "publication"))
    assert out.startswith(f"⚠️ PR #7 merge: merged as {MERGE[:12]}") and "already recorded" in out
    assert "PR_MERGE" not in out and "earlier" not in out
    assert f"- Outcome: **merged** — merge commit `{MERGE}`" in world.gh.pr["body"]
    assert _merges(world) == 1


def test_an_unreadable_pr_leaves_a_queued_receipt_untouched(world, monkeypatch):
    world.gh.merge = "queued"
    _merge(world)
    (before,) = _receipts(world)
    original = world.gh
    world.gh.pr.update(state="CLOSED", autoMergeRequest=None)

    def unreadable(args, *a, **kw):
        if args[:2] == ["pr", "view"]:
            return github.GhResult(False, "⚠️ GH_ERROR: HTTP 502", 1, 502, "exit")
        return original(args, *a, **kw)

    monkeypatch.setattr(github, "_gh_run", unreadable)
    out = _merge(world)
    assert "pr_unreadable" in out and f"(receipt_id={before['receipt_id']}, outcome queued)" in out
    assert _receipts(world) == [before] and _merges(world) == 1


def test_an_unknown_outcome_is_read_back_by_plain_repeats_and_continued_only_by_a_decision(world):
    """#2762: an open PR at the same head without a queue does not make a plain repeat send again; only the explicit
    continue_receipt_id does, once, and the earlier request keeps its unknown outcome."""
    world.gh.merge = "timeout_open"
    first = _merge(world)
    (unknown,) = _receipts(world)
    assert "PR_MERGE_UNKNOWN" in first and _merges(world) == 1
    assert f"call pr_merge with continue_receipt_id='{unknown['receipt_id']}'" in first
    world.gh.merge = "ok"
    out = _merge(world)
    assert "sent no new merge request" in out and _merges(world) == 1 and len(_receipts(world)) == 1
    continued = _merge(world, continue_receipt_id=unknown["receipt_id"])
    assert "merge: merged" in continued and f"continues receipt {unknown['receipt_id']}" in continued
    assert _merges(world) == 2
    earlier, successor = sorted(_receipts(world), key=lambda r: r["receipt_id"] != unknown["receipt_id"])
    assert successor["continues"] == unknown["receipt_id"] and "continues" not in earlier
    assert earlier["effect"] == unknown["effect"] and earlier["outcome"] == unknown["outcome"]  # never rewritten
    assert successor["outcome"]["attribution"] == "unproven"  # the earlier request may have landed
    assert "sent no new merge request" in _merge(world) and _merges(world) == 2
    schema = next(e.schema for e in github.get_tools() if e.name == "pr_merge")
    assert "retry_after_unknown" not in schema["parameters"]["properties"]
    assert schema["parameters"]["properties"]["continue_receipt_id"]["type"] == "string"


def test_a_merge_someone_finished_while_the_answer_was_lost_is_unattributed(world):
    world.gh.merge = "timeout_merged"
    _merge(world, reviewed_head_sha=HEAD)
    outcome = _receipts(world)[0]["outcome"]
    assert outcome["status"] == "merged" and outcome["attribution"] == "unproven"
    _merge(world, reviewed_head_sha=HEAD)
    assert _receipts(world)[0]["outcome"]["attribution"] == "unproven"


def test_a_failed_readback_is_unknown(world):
    world.gh.fail_view_after_merge = True
    assert "PR_MERGE_UNKNOWN" in _merge(world)
    assert _receipts(world)[0]["outcome"] == {"status": "unknown", "reason": "readback_failed"}


def test_a_failed_merge_request_shows_its_retained_diagnostic_on_every_call(world, monkeypatch):
    """#1555: the bounded, redacted gh facts the receipt keeps reach the local response on the
    first call and on the observation-only repeat; the outcome stays unknown, never 'rejected'.
    A permission denial is one of the answers that does not prove the request was not applied."""
    original = world.gh
    denial = "⚠️ GH_ERROR: gh: Resource not accessible by personal access token (HTTP 403)"

    def gh(args, *a, **kw):
        if sent(args):
            original.calls.append(list(args))
            return github.GhResult(False, denial, 1, 403, "exit")
        return original(args, *a, **kw)

    monkeypatch.setattr(github, "_gh_run", gh)
    first = _merge(world)
    assert "PR_MERGE_UNKNOWN" in first and f"Merge request: failure=exit, exit_code=1, http_status=403: {denial}" in first
    assert "GitHub readback: pr_state_open_after_unconfirmed_effect." in first
    again = _merge(world)
    assert "had no confirmed outcome" in again and "sent no new merge request" in again
    assert f"Earlier merge request: failure=exit, exit_code=1, http_status=403: {denial}" in again
    assert sum(sent(c) for c in original.calls) == 1


@pytest.mark.parametrize(("merge", "fact"), [
    ("timeout_open", "failure=timeout: ⚠️ GH_TIMEOUT: exceeded 120s."),
    ("refused", f"failure=server_refused, exit_code=1, http_status=405: {NOT_MERGEABLE}"),
])
def test_different_merge_failures_read_differently(world, merge, fact):
    world.gh.merge = merge
    assert f"Merge request: {fact}" in _merge(world)


def test_the_public_pr_body_never_carries_the_local_diagnostic(world):
    world.gh.merge = "timeout_merged"
    assert "Merge request: failure=timeout" in _merge(world, reviewed_head_sha=HEAD)
    assert _receipts(world)[0]["publication"]["body"] == {"status": "published"}
    assert "GH_TIMEOUT" not in world.gh.pr["body"] and "failure=" not in world.gh.pr["body"]


def test_a_repeat_after_a_confirmed_merge_does_not_call_it_unconfirmed(world):
    _merge(world, reviewed_head_sha=HEAD)
    again = _merge(world, reviewed_head_sha=HEAD)
    assert "merge: merged" in again and "sent no new merge request" in again
    assert "already recorded" in again and "had no confirmed outcome" not in again
    assert sum(sent(c) for c in world.gh.calls) == 1


def test_an_unreadable_pr_keeps_the_earlier_request_in_view_by_its_real_status(world, monkeypatch):
    """#1555: this call's no-effect fact must not read as the resolution of the earlier one."""
    original = world.gh

    def unreadable(args, *a, **kw):
        if args[:2] == ["pr", "view"]:
            return github.GhResult(False, "⚠️ GH_ERROR: HTTP 502", 1, 502, "exit")
        return original(args, *a, **kw)

    monkeypatch.setattr(github, "_gh_run", unreadable)
    assert "could not be read; nothing was merged" in _merge(world)  # no earlier request: unchanged
    monkeypatch.setattr(github, "_gh_run", original)
    world.gh.merge = "timeout_open"
    _merge(world)
    (earlier,) = _receipts(world)
    monkeypatch.setattr(github, "_gh_run", unreadable)
    out = _merge(world)
    assert "pr_unreadable" in out and "nothing was merged" not in out and "sent no new merge request" in out
    assert (f"the earlier merge request for {URL} is still unresolved "
            f"(receipt_id={earlier['receipt_id']}, outcome unknown)") in out
    assert _receipts(world) == [earlier]
    assert sum(sent(c) for c in original.calls) == 1

    monkeypatch.setattr(github, "_gh_run", original)
    world.gh.pr.update(state="MERGED", mergeCommit={"oid": MERGE})
    _merge(world)  # observes the merge on the same receipt
    monkeypatch.setattr(github, "_gh_run", unreadable)
    out = _merge(world)
    assert f"a merge of {URL} was already confirmed (receipt_id={earlier['receipt_id']})" in out
    assert "nothing was merged" not in out and "still unresolved" not in out
    assert sum(sent(c) for c in original.calls) == 1


def test_an_intent_without_any_outcome_is_only_read_back(world, monkeypatch):
    """A claim another (or a crashed) call holds has no outcome yet; it is not called unconfirmed."""
    original = world.gh

    def crash_at_effect(args, *a, **kw):
        if sent(args):
            raise RuntimeError("worker died between intent and effect")
        return original(args, *a, **kw)

    monkeypatch.setattr(github, "_gh_run", crash_at_effect)
    with pytest.raises(RuntimeError):
        _merge(world)
    assert [r["state"] for r in _receipts(world)] == ["intent_recorded"]
    monkeypatch.setattr(github, "_gh_run", original)
    out = _merge(world)
    assert "holds its merge intent with no recorded outcome yet" in out and "sent no new merge request" in out
    assert "had no confirmed outcome" not in out
    assert not any(sent(c) for c in original.calls)


def test_an_unwritable_intent_refuses_before_any_effect(world, monkeypatch):
    monkeypatch.setattr(merge_receipts, "write_receipt", lambda *a, **k: (_ for _ in ()).throw(TimeoutError("lock")))
    assert "merge_intent_unwritable" in _merge(world)
    assert not any(sent(c) for c in world.gh.calls)


def test_a_failed_publication_is_a_gap_and_a_retry_never_merges_again(world):
    world.gh.fail_edit = True
    out = _merge(world, reviewed_head_sha=HEAD)
    receipt = _receipts(world)[0]
    assert receipt["outcome"]["status"] == "merged" and receipt["publication"]["body"]["status"] == "gap"
    assert receipt["outcome"]["attribution"] == "this_call"
    assert "not confirmed" in out
    world.gh.fail_edit = False
    out = _merge(world, reviewed_head_sha=HEAD)
    assert "sent no new merge request" in out
    assert sum(1 for c in world.gh.calls if sent(c)) == 1
    assert _receipts(world)[0]["publication"]["body"]["status"] == "published"
    assert _receipts(world)[0]["receipt_id"] == receipt["receipt_id"]
    assert _receipts(world)[0]["outcome"] == receipt["outcome"]


@pytest.mark.parametrize("failure", ["readback_unavailable", "missing_block"])
def test_successful_body_patch_needs_readback_and_recovers_without_second_merge(world, monkeypatch, failure):
    original = world.gh
    description = world.gh.pr["body"]
    patch_seen = False

    def gh(args, *a, **kw):
        nonlocal patch_seen
        if patch_seen and args[:2] == ["pr", "view"] and failure == "readback_unavailable":
            return github.GhResult(False, "readback unavailable", 1, 502, "exit")
        result = original(args, *a, **kw)
        if args[:1] == ["api"] and "PATCH" in args:
            patch_seen = True
            if failure == "missing_block":
                world.gh.pr["body"] = description
        return result

    monkeypatch.setattr(github, "_gh_run", gh)
    assert "not confirmed" in _merge(world)
    (before,) = _receipts(world)
    assert before["outcome"]["status"] == "merged"
    assert before["publication"]["body"] == {"status": "gap", "reason": "readback_missing_block"}
    assert before["publication"]["card"]["status"] == "owed"
    monkeypatch.setattr(github, "_gh_run", original)
    assert "sent no new merge request" in _merge(world)
    (after,) = _receipts(world)
    assert after["publication"]["body"] == {"status": "published"}
    assert all(after[key] == before[key] for key in ("receipt_id", "revision", "outcome", "effect"))
    assert after["publication"]["card"] == before["publication"]["card"]
    assert world.gh.pr["body"] == description + "\n\n" + merge_receipts.public_block(after) + "\n"
    assert sum(sent(c) for c in original.calls) == 1
    assert sum(c[:1] == ["api"] and "PATCH" in c for c in original.calls) == (2 if failure == "missing_block" else 1)
    assert not any(c[:2] == ["pr", "edit"] for c in original.calls)


@pytest.mark.parametrize("url", ["", URL + "junk",
                                 "https://github.com/octo/demo/pull/8"])
def test_a_readback_url_no_merge_request_can_address_sends_and_writes_nothing(world, url):
    """The request goes only to the PR GitHub's readback names; a URL that names no PR #7 leaves no default target."""
    world.gh.pr["url"] = url
    description = world.gh.pr["body"]
    out = _merge(world)
    assert out.startswith("⚠️ PR_MERGE_REFUSED: pr_unreadable") and "nothing was merged" in out
    assert _receipts(world) == [] and world.gh.pr["body"] == description
    assert not any(sent(c) or c[:2] == ["pr", "edit"] or "PATCH" in c for c in world.gh.calls)


def test_the_body_block_replaces_an_older_receipt_and_the_checklist_is_recognized(world):
    world.gh.pr["body"] = f"Intro.\n\n```json\n{CHECKLIST}\n```\n\nFooter."
    _merge(world, reviewed_head_sha=HEAD)
    assert _receipts(world)[0]["review"]["contributor_evidence"] == "present_validated"
    first_body = world.gh.pr["body"]
    updated = merge_receipts.upsert_body(first_body, "<!-- ouroboros:merge-receipt ab -->\nnew\n<!-- /ouroboros:merge-receipt -->")
    assert updated.count("<!-- ouroboros:merge-receipt ") == 1 and "new" in updated and "Footer." in updated and "Intro." in updated
    assert merge_receipts.contributor_evidence("```json\n[{\"item\": \"x\"}]\n```", []) == "present_invalid"
    assert merge_receipts.contributor_evidence("no checklist", []) == "absent"


def test_pr_merge_is_registered_with_an_explicit_safety_policy():
    from ouroboros.safety import POLICY_CHECK, TOOL_POLICY

    names = {entry.name for entry in github.get_tools()}
    assert "pr_merge" in names and TOOL_POLICY["pr_merge"] == POLICY_CHECK
    schema = next(e.schema for e in github.get_tools() if e.name == "pr_merge")
    assert set(schema["parameters"]["required"]) == {"number", "expected_head_sha", "method"}
    assert schema["parameters"]["properties"]["method"]["enum"] == ["merge", "squash", "rebase"]


def test_queue_recovery_observes_the_same_operation_and_recomputes_changed_subject(world):
    world.gh.merge = "queued"
    _merge(world, review_task_ids=["review-1"], reviewed_head_sha=HEAD, reviewed_base_sha=BASE)
    rid = _receipts(world)[0]["receipt_id"]
    _merge(world)
    assert _receipts(world)[0]["outcome"]["status"] == "queued"
    world.gh.pr.update(state="MERGED", headRefOid="e" * 40, mergeCommit={"oid": MERGE})
    _merge(world)
    (receipt,) = _receipts(world)
    assert receipt["receipt_id"] == rid
    assert receipt["outcome"]["attribution"] == "unproven"
    assert receipt["coverage"]["status"] == "changes_after_review"
    assert "merged_head_differs" in receipt["coverage"]["gaps"]
    assert sum(sent(c) for c in world.gh.calls) == 1
    assert merge_receipts.card_row_text(receipt).startswith("⚠️")


def test_base_and_tree_gaps_prevent_green_without_vetoing_merge(world, monkeypatch):
    original = world.gh

    def gh(args, *a, **kw):
        if args[:1] == ["api"] and "/commits/" in args[1]:
            is_merge = args[1].endswith(MERGE)
            return github.GhResult(True, json.dumps({"commit": {"tree": {"sha": TREE if is_merge else "f" * 40}},
                                                      "parents": [{"sha": "e" * 40}]}), 0, None, "")
        return original(args, *a, **kw)

    monkeypatch.setattr(github, "_gh_run", gh)
    _merge(world, review_task_ids=["review-1"], reviewed_head_sha=HEAD, reviewed_base_sha=BASE)
    r = _receipts(world)[0]
    assert r["outcome"]["status"] == "merged"
    assert r["coverage"]["status"] == "unknown"
    assert set(r["coverage"]["gaps"]) == {"base_changed_since_review", "merged_tree_differs_from_reviewed_head",
                                          "no_host_observed_review_record"}


def test_parallel_callers_share_one_atomic_intent(world, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier, Lock

    barrier, lock = Barrier(2), Lock()
    original, views = world.gh, 0

    def gh(args, *a, **kw):
        nonlocal views
        result = original(args, *a, **kw)
        if args[:2] == ["pr", "view"]:
            with lock:
                views += 1
                initial = views <= 2
            if initial:
                barrier.wait(timeout=5)  # both see OPEN before either claims
        return result

    monkeypatch.setattr(github, "_gh_run", gh)
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(lambda _: _merge(world), range(2)))
    assert sum(sent(c) for c in world.gh.calls) == 1
    assert len(_receipts(world)) == 1


@pytest.mark.parametrize(("mode", "status"), [("ok", "merged"), ("queued", "queued"), ("refused", "refused")])
def test_losing_claim_cannot_erase_the_winners_effect_or_settled_outcome(world, monkeypatch, mode, status):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier, Event, local

    world.gh.merge = mode
    initial_reads, loser_claimed, winner_done = Barrier(2), Event(), Event()
    caller = local()
    read, write, gh = merge_receipts.task_receipts, merge_receipts.write_receipt, world.gh
    winner = {}

    def read_together(*args):
        rows = read(*args)
        initial_reads.wait(timeout=5)  # both decide to claim before either writes
        return rows

    def claim_then_delay(*args, **kwargs):
        selected = write(*args, **kwargs)
        if kwargs.get("claim"):
            caller.won = selected["receipt_id"] == args[2]["receipt_id"]
            if not caller.won:
                loser_claimed.set()  # retain the intent snapshot until the winner finishes
                assert winner_done.wait(timeout=5)
        return selected

    def effect(*args, **kwargs):
        if sent(args[0]):
            assert loser_claimed.wait(timeout=5)
        return gh(*args, **kwargs)

    def run():
        try:
            result = _merge(world, review_task_ids=["review-1"], reviewed_head_sha=HEAD, reviewed_base_sha=BASE)
            if caller.won:
                winner.update(_receipts(world)[0])
            return result
        finally:
            if getattr(caller, "won", False):
                winner_done.set()

    monkeypatch.setattr(merge_receipts, "task_receipts", read_together)
    monkeypatch.setattr(merge_receipts, "write_receipt", claim_then_delay)
    monkeypatch.setattr(github, "_gh_run", effect)
    with ThreadPoolExecutor(max_workers=2) as pool:
        outputs = list(pool.map(lambda _: run(), range(2)))
    (stored,) = _receipts(world)
    assert stored["effect"] == winner["effect"]
    assert stored["outcome"] == winner["outcome"]
    assert stored["coverage"] == winner["coverage"]
    # The loser's readback predates the winner's effect: the winner's (auto-merge) snapshot stays.
    assert stored["observed_after"] == winner["observed_after"]
    assert stored["observed_after"]["auto_merge"] is (mode == "queued")
    assert stored["outcome"]["status"] == status and stored["state"] == "settled"
    assert all(f"merge: {status}" in text for text in outputs)
    assert sum(sent(c) for c in world.gh.calls) == 1
    if status != "refused":
        assert f"Outcome: **{status}**" in world.gh.pr["body"]
        assert all(f"merge: {status}" in event["text"] for event in world.ctx.pending_events)


def test_delayed_queue_publication_catches_up_to_the_persisted_merge(world, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event

    queued_edit, merged_published = Event(), Event()
    gh = world.gh
    world.gh.merge = "queued"

    def delayed_edit(args, *a, **kw):
        if args[:1] == ["api"] and "PATCH" in args and "Outcome: **queued**" in kw["input_data"]:
            queued_edit.set()
            assert merged_published.wait(timeout=5)
        return gh(args, *a, **kw)

    monkeypatch.setattr(github, "_gh_run", delayed_edit)
    with ThreadPoolExecutor(max_workers=1) as pool:
        first = pool.submit(_merge, world)
        try:
            assert queued_edit.wait(timeout=5)
            world.gh.pr.update(state="MERGED", mergeCommit={"oid": MERGE})
            assert "merge: merged" in _merge(world)
            settled = _receipts(world)[0]
        finally:
            merged_published.set()
        assert "merge: merged" in first.result(timeout=5)
    (stored,) = _receipts(world)
    assert stored["effect"] == settled["effect"] and stored["outcome"] == settled["outcome"]
    assert stored["outcome"]["attribution"] == "unproven"
    assert merge_receipts.public_block(stored) in world.gh.pr["body"]
    assert stored["publication"]["body"]["status"] == "published"
    assert "merge: merged" in world.ctx.pending_events[-1]["text"]
    assert sum(sent(c) for c in world.gh.calls) == 1


def test_retry_enriches_merge_readback_without_erasing_attribution(world, monkeypatch):
    gh = world.gh

    def unavailable_commit(args, *a, **kw):
        if args[:1] == ["api"] and "/commits/" in args[1]:
            return github.GhResult(False, "unavailable", 1, 502, "exit")
        return gh(args, *a, **kw)

    monkeypatch.setattr(github, "_gh_run", unavailable_commit)
    _merge(world, review_task_ids=["review-1"], reviewed_head_sha=HEAD, reviewed_base_sha=BASE)
    initial = _receipts(world)[0]
    assert initial["coverage"]["status"] == "unknown" and "tree_comparison_unavailable" in initial["coverage"]["gaps"]
    monkeypatch.setattr(github, "_gh_run", gh)
    assert _merge(world).startswith("⚠️ PR #7 merge: merged")  # a declaration never reads green
    (stored,) = _receipts(world)
    assert stored["receipt_id"] == initial["receipt_id"]
    assert stored["outcome"]["attribution"] == "this_call"
    assert stored["outcome"]["merge_tree"] == TREE
    assert stored["coverage"]["status"] == "unknown"
    assert stored["coverage"]["gaps"] == ["no_host_observed_review_record"]  # the tree gap was enriched away
    assert merge_receipts.public_block(stored) in world.gh.pr["body"]
    assert sum(sent(c) for c in world.gh.calls) == 1


def test_unsettled_and_merged_receipts_survive_disposable_history_eviction(world):
    world.gh.merge = "accepted_open"
    _merge(world)
    rid = _receipts(world)[0]["receipt_id"]
    for i in range(merge_receipts._RECEIPTS_CAP + 2):
        merge_receipts.write_receipt(world.root, "merge-task", {
            "receipt_id": f"other-{i}", "repo": {"url": f"other-{i}"}, "state": "settled",
            "outcome": {"status": "refused"}})
    assert any(r["receipt_id"] == rid for r in _receipts(world))
    _merge(world)
    assert sum(sent(c) for c in world.gh.calls) == 1


def test_card_outbox_survives_worker_exit_and_confirms_delivery_separately(world, monkeypatch):
    from supervisor import terminal_delivery as td

    _merge(world)
    receipt = _receipts(world)[0]
    did = receipt["publication"]["card"]["delivery_id"]
    world.ctx.pending_events.clear()  # the worker died before draining its buffer
    monkeypatch.setattr(td, "_REPLAY_MIN_AGE_SEC", 0)
    queue = DeliveryBus()  # replay admits a copy only through the bus's membership
    assert td.replay_pending_deliveries(world.root, event_queue=queue) == [did]
    event = queue.get_nowait()
    assert event["progress_meta"]["card_row_id"] == f"merge-receipt:{receipt['receipt_id']}"
    td.register_delivery(world.root, did)  # supervisor's post-send seam
    _merge(world)
    assert _receipts(world)[0]["publication"]["card"]["status"] == "delivered"
    assert td.pending_deliveries(world.root) == []


def test_body_uses_fresh_publication_view_and_confirms_a_lost_edit_reply(world, monkeypatch):
    original = world.gh
    concurrent_note = "\n\nNote added while merge was running.  \n"

    def gh(args, *a, **kw):
        result = original(args, *a, **kw)
        if sent(args):
            world.gh.pr["body"] += concurrent_note
        if args[:1] == ["api"] and "PATCH" in args:
            return github.GhResult(False, "lost reply", None, None, "timeout")
        return result

    monkeypatch.setattr(github, "_gh_run", gh)
    _merge(world)
    assert concurrent_note in world.gh.pr["body"]
    assert _receipts(world)[0]["publication"]["body"]["status"] == "published"
    _merge(world)
    assert sum(c[:1] == ["api"] and "PATCH" in c for c in world.gh.calls) == 1


@pytest.mark.parametrize(("body_size", "method"), [(80, "merge"), (80, "rebase"), (65536, "squash")])
def test_merge_message_uses_stdin_at_github_body_limit(world, monkeypatch, method, body_size):
    from ouroboros import update_qualification as qualification

    body = ("Owner's `code` and Unicode →\n" * 3000)[:body_size]
    world.gh.pr["body"] = body
    merges = []

    def transport(argv, **kwargs):
        args = argv[1:]
        if sent(args):
            merges.append((args, kwargs.get("input")))
            assert len(subprocess.list2cmdline(argv)) < 32767
        result = world.gh(args, world.ctx, input_data=kwargs.get("input"))
        return subprocess.CompletedProcess(argv, result.exit_code, result.text if result.ok else "",
                                           "" if result.ok else result.text)

    monkeypatch.setattr(github, "_gh_run", REAL_GH_RUN)
    monkeypatch.setattr(github.subprocess, "run", transport)
    monkeypatch.setattr(github, "_gh_env", lambda ctx: {})
    assert "merge: merged" in github._pr_merge(world.ctx, number=7, expected_head_sha=HEAD, method=method)
    args, stdin = merges[0]
    assert args == ["api", ROUTE, "--hostname", "github.com", "--method", "PUT", "--input", "-"]
    payload = json.loads(stdin)
    assert (payload["sha"], payload["merge_method"]) == (HEAD, method)
    if method == "rebase":  # a rebase creates no merge commit to carry the receipt
        assert "commit_message" not in payload
    else:
        message = payload["commit_message"]
        assert message == body.rstrip() + "\n\n" + qualification.commit_summary(_receipts(world)[0]["qualification"])
        # The downstream reader parses the trailer from what GitHub keeps after its automatic title.
        assert qualification.commit_qualification(f"Merge pull request #7 from octo/feature\n\n{message}") == (
            _receipts(world)[0]["qualification"])


# --- #2762: an explicit decision continues the latest attempt ------------------------------------------------


def _by_id(world):
    return {r["receipt_id"]: r for r in _receipts(world)}


def test_decisions_chain_forward_and_repeats_rejoin_without_resend_or_stale_publication(world):
    """Owner choice (9 Nov): A unknown, B decided after A and refused by GitHub, C decided after B and merged. A repeat
    of either decision rejoins the request it created, even a refused one; a plain call observes the latest attempt by
    the ``continues`` relation, never the row an upsert moved last; and the older attempts never publish over C."""
    world.gh.merge = "timeout_open"
    _merge(world)
    (a,) = _receipts(world)
    world.gh.merge = "refused"
    refused = _merge(world, continue_receipt_id=a["receipt_id"])
    b = next(r for r in _receipts(world) if r.get("continues") == a["receipt_id"])
    assert refused.startswith("⚠️ PR_MERGE_REFUSED: ⚠️ PR #7 merge: refused (by GitHub)") and _merges(world) == 2
    # A plain call would only read back while A stays unconfirmed, so the next decision must name B.
    assert f"continue_receipt_id='{b['receipt_id']}'" in refused and f"continue_receipt_id='{a['receipt_id']}'" not in refused
    again = _merge(world, continue_receipt_id=a["receipt_id"])
    assert "This decision already sent its request" in again and f"receipt_id={b['receipt_id']}" in again
    assert f"continue_receipt_id='{b['receipt_id']}'" in again and _merges(world) == 2
    plain = _merge(world)
    assert f"receipt_id={b['receipt_id']}" in plain and "sent no new merge request" in plain and _merges(world) == 2
    world.gh.merge = "ok"
    merged = _merge(world, continue_receipt_id=b["receipt_id"])
    c = next(r for r in _receipts(world) if r.get("continues") == b["receipt_id"])
    assert "merge: merged" in merged and _merges(world) == 3
    assert (c["outcome"]["status"], c["outcome"]["attribution"]) == ("merged", "unproven")
    body, events = world.gh.pr["body"], len(world.ctx.pending_events)
    assert merge_receipts.public_block(c) in body
    old_a = _merge(world, continue_receipt_id=a["receipt_id"])
    assert "This decision already sent its request" in old_a and f"Receipt {c['receipt_id']} continued it since" in old_a
    assert "continue_receipt_id=" not in old_a  # the latest attempt is C: no decision may follow B
    old_b = _merge(world, continue_receipt_id=b["receipt_id"])
    assert "This decision already sent its request" in old_b and f"receipt_id={c['receipt_id']}" in old_b
    assert _merges(world) == 3 and world.gh.pr["body"] == body  # nothing republished over C's block
    rows = [e["progress_meta"]["card_row_id"] for e in world.ctx.pending_events[events:]]
    assert set(rows) <= {f"merge-receipt:{c['receipt_id']}"}  # B was rejoined, never projected over C
    stored = _by_id(world)
    assert stored[a["receipt_id"]]["effect"] == a["effect"] and stored[a["receipt_id"]]["outcome"]["status"] == "unknown"
    assert stored[b["receipt_id"]]["effect"] == b["effect"] and "publication" not in stored[b["receipt_id"]]
    assert merge_receipts.latest_attempt(list(stored.values()))["receipt_id"] == c["receipt_id"]


def test_concurrent_deliveries_of_one_decision_send_once(world, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier, Lock

    world.gh.merge = "timeout_open"
    _merge(world)
    (a,) = _receipts(world)
    world.gh.merge = "ok"
    barrier, lock, original, views = Barrier(2), Lock(), world.gh, [0]

    def gh(args, *a_, **kw):
        result = original(args, *a_, **kw)
        if args[:2] == ["pr", "view"]:
            with lock:
                views[0] += 1
                initial = views[0] <= 2
            if initial:
                barrier.wait(timeout=5)  # both read the PR open at the same head before either claims
        return result

    monkeypatch.setattr(github, "_gh_run", gh)
    with ThreadPoolExecutor(max_workers=2) as pool:
        outputs = list(pool.map(lambda _: _merge(world, continue_receipt_id=a["receipt_id"]), range(2)))
    assert _merges(world) == 2  # the unknown request and exactly one decided successor
    assert len([r for r in _receipts(world) if r.get("continues") == a["receipt_id"]]) == 1
    assert sum("This decision already sent its request" in out for out in outputs) == 1


def test_continuation_links_survive_the_disposable_receipt_cap(world):
    world.gh.merge = "timeout_open"
    _merge(world)
    (first,) = _receipts(world)
    previous = first["receipt_id"]
    world.gh.merge = "refused"
    for _ in range(merge_receipts._RECEIPTS_CAP + 1):
        _merge(world, continue_receipt_id=previous)
        successor = next(r for r in _receipts(world) if r.get("continues") == previous)
        previous = successor["receipt_id"]
    sends = _merges(world)
    _merge(world, continue_receipt_id=first["receipt_id"])
    assert _merges(world) == sends
    assert merge_receipts.latest_attempt(_receipts(world))["receipt_id"] == previous


@pytest.mark.parametrize("change,code", [
    ({"headRefOid": "f" * 40}, "head_moved"),
    ({"baseRefName": "release"}, "base_moved"),
    ({"autoMergeRequest": {"enabledAt": "by another actor"}}, "auto_merge_pending"),
    ({"mergeStateStatus": "BLOCKED"}, "not_mergeable"),
    ({"mergeStateStatus": "BEHIND"}, "not_mergeable"),
    ({"mergeStateStatus": "DIRTY"}, "not_mergeable"),
    ({"state": "CLOSED"}, "pr_not_open"),
], ids=["head", "base-ref", "auto-merge", "blocked", "behind", "dirty", "closed"])
def test_a_continuation_sends_only_after_a_fresh_readback_of_the_same_request(world, change, code):
    world.gh.merge = "timeout_open"
    _merge(world)
    (a,) = _receipts(world)
    world.gh.pr.update(change)
    world.gh.merge = "ok"
    out = _merge(world, continue_receipt_id=a["receipt_id"])
    assert out.startswith(f"⚠️ PR_MERGE_REFUSED: {code}")
    assert _merges(world) == 1 and _receipts(world) == [a]  # a refused decision sends and writes nothing


def test_a_decision_names_this_tasks_own_request_for_the_same_pr_head_and_method(world):
    from ouroboros.task_results import write_task_result

    world.gh.merge = "timeout_open"
    _merge(world)
    (a,) = _receipts(world)
    write_task_result(world.root, "other-task", STATUS_RUNNING, root_task_id="other-task", delegation_role="root")
    merge_receipts.write_receipt(world.root, "other-task", {**a, "receipt_id": "foreign"})
    merge_receipts.write_receipt(world.root, "merge-task", {**a, "receipt_id": "elsewhere",
                                                           "repo": {"url": "https://github.com/octo/other/pull/7"}})
    world.gh.merge = "ok"
    for kw, said in (({"continue_receipt_id": "missing"}, "no merge receipt of this task"),
                     ({"continue_receipt_id": "foreign"}, "no merge receipt of this task"),
                     ({"continue_receipt_id": a["receipt_id"], "method": "merge"}, "repeats its PR number, method and head"),
                     ({"continue_receipt_id": a["receipt_id"], "expected_head_sha": "f" * 40}, "repeats its PR number"),
                     ({"continue_receipt_id": "elsewhere"}, "another pull request")):
        out = _merge(world, **kw)
        assert out.startswith("⚠️ PR_MERGE_REFUSED:") and said in out and "nothing was merged" in out, out
    assert _merges(world) == 1 and not any(r.get("continues") for r in _receipts(world))


@pytest.mark.parametrize("case,why", [
    ("queued", "its outcome is queued"),
    ("accepted_open", "GitHub answered its request as merged"),
    ("running", "its call may still be running"),
])
def test_a_queued_answered_or_running_attempt_cannot_be_continued(world, case, why):
    from ouroboros.task_results import write_task_result

    if case == "running":  # an earlier invocation of this task that has not returned
        write_task_result(world.root, "merge-task", STATUS_RUNNING, root_task_id="merge-task", delegation_role="root",
                          launch_handoffs={"op-live": {"tool": "pr_merge", "task_id": "merge-task", "claimed_at": "t",
                                                       "root_task_id": "merge-task", "state": "claimed",
                                                       "local_owner": {}, "owner_proof": {}}})
        world.gh.merge = "timeout_open"
        _merge(world)
        receipt = _receipts(world)[0]
        merge_receipts.write_receipt(world.root, "merge-task", {**receipt, "launch_operation_ids": ["op-live"]})
    else:
        world.gh.merge = case
        _merge(world)
    (a,) = _receipts(world)
    world.gh.pr.update(autoMergeRequest=None)  # whatever the readback shows now, the attempt itself decides
    world.gh.merge = "ok"
    out = _merge(world, continue_receipt_id=a["receipt_id"])
    assert out.startswith("⚠️ PR_MERGE_REFUSED: continuation_refused") and why in out, out
    assert _merges(world) == 1 and len(_receipts(world)) == 1


@pytest.mark.parametrize("lost", ["crash_before_answer", "malformed_answer"])
def test_a_lost_or_unrecognized_answer_stays_unknown_and_a_decision_may_follow_it(world, monkeypatch, lost):
    original = world.gh

    def first(args, *a, **kw):
        if sent(args):
            if lost == "crash_before_answer":
                raise RuntimeError("worker died after the intent")
            original.calls.append(list(args))
            return github.GhResult(True, json.dumps({"sha": MERGE, "merged": False, "message": "?"}), 0, None, "")
        return original(args, *a, **kw)

    monkeypatch.setattr(github, "_gh_run", first)
    if lost == "crash_before_answer":
        with pytest.raises(RuntimeError):
            _merge(world)
        assert "sent no new merge request" in _merge(world)  # observed, never resent
    else:
        assert "failure=malformed_response" in _merge(world)
    (a,) = _receipts(world)
    assert a["outcome"]["status"] == "unknown" and not merge_receipts.effect_is_definite(a)
    monkeypatch.setattr(github, "_gh_run", original)
    out = _merge(world, continue_receipt_id=a["receipt_id"])
    assert "merge: merged" in out and _merges(world) == 1 + (lost == "malformed_answer")
    assert _by_id(world)[a["receipt_id"]]["outcome"]["status"] == "unknown"


def test_a_merge_queue_requirement_is_refused_with_no_queued_admin_or_async_fallback(world, monkeypatch):
    """The synchronous endpoint does not use merge queues; GitHub's refusal is the answer, never a hidden queue."""
    original = world.gh

    def queue_required(args, *a, **kw):
        if sent(args):
            original.calls.append(list(args))
            original.merge_payloads.append(json.loads(kw["input_data"]))
            return github.GhResult(False, "⚠️ GH_ERROR: gh: Changes must be made through a merge queue (HTTP 405)",
                                   1, 405, "exit")
        return original(args, *a, **kw)

    monkeypatch.setattr(github, "_gh_run", queue_required)
    out = _merge(world)
    assert out.startswith("⚠️ PR_MERGE_REFUSED: ⚠️ PR #7 merge: refused (by GitHub)") and "merge queue" in out
    (receipt,) = _receipts(world)
    assert receipt["effect"]["failure"] == "server_refused" and merge_receipts.effect_is_definite(receipt)
    assert [set(payload) for payload in original.merge_payloads] == [{"sha", "merge_method", "commit_message"}]
    assert [c for c in original.calls if c[:1] == ["api"] and "--method" in c] == [
        ["api", ROUTE, "--hostname", "github.com", "--method", "PUT", "--input", "-"]]
    assert not any("--auto" in c or "--admin" in c or "merge-async" in " ".join(c) or c[:2] == ["pr", "merge"]
                   for c in original.calls)


_FAKE_GH = r'''
import json, os, sys
state_path, log_path = os.environ["FAKE_GH_STATE"], os.environ["FAKE_GH_LOG"]
state = json.load(open(state_path))
args = sys.argv[1:]
with open(log_path, "a") as log:
    log.write(json.dumps(args) + "\n")
def answer(body, code=0, err=""):
    sys.stdout.write(json.dumps(body))
    sys.stderr.write(err)
    sys.exit(code)
if args[:2] == ["pr", "view"]:
    answer(state["pr"])
if args[:1] == ["api"] and "PUT" in args:
    request = json.loads(sys.stdin.read())
    with open(log_path, "a") as log:
        log.write(json.dumps({"stdin": request}) + "\n")
    state["puts"] += 1
    if state["puts"] == 1:  # the answer of the first request is lost behind a gateway error
        json.dump(state, open(state_path, "w"))
        answer({"message": "Server Error"}, 1, "gh: Server Error (HTTP 502)\n")
    state["pr"].update(state="MERGED", mergeCommit={"oid": "c" * 40}, mergedAt="2026-11-09T00:00:00Z")
    json.dump(state, open(state_path, "w"))
    answer({"sha": "c" * 40, "merged": True, "message": "Pull Request successfully merged"})
if args[:1] == ["api"] and "PATCH" in args:
    state["pr"]["body"] = json.loads(sys.stdin.read())["body"]
    json.dump(state, open(state_path, "w"))
    answer({})
if args[:1] == ["api"] and "/commits/" in args[1]:
    answer({"commit": {"tree": {"sha": "d" * 40}}, "parents": [{"sha": "b" * 40}, {"sha": "a" * 40}]})
answer({"message": "unexpected"}, 1, "gh: unexpected call (HTTP 400)\n")
'''


@pytest.mark.skipif(__import__("os").name == "nt", reason="the fake gh is a POSIX shell wrapper")
def test_a_real_gh_process_carries_an_unknown_merge_and_its_decided_continuation(world, monkeypatch, tmp_path):
    """The real transport over a local inert ``gh`` executable: argv, the JSON request on stdin, exit status and
    gh's own ``(HTTP NNN)`` stderr marker cross a real process boundary; no network and no GitHub write."""
    import sys

    bin_dir, state, log = tmp_path / "bin", tmp_path / "gh-state.json", tmp_path / "gh-log.jsonl"
    bin_dir.mkdir()
    (tmp_path / "fake_gh.py").write_text(_FAKE_GH, encoding="utf-8")
    wrapper = bin_dir / "gh"
    wrapper.write_text(f'#!/bin/sh\nexec "{sys.executable}" -I "{tmp_path / "fake_gh.py"}" "$@"\n', encoding="utf-8")
    wrapper.chmod(0o755)
    state.write_text(json.dumps({"pr": world.gh.pr, "puts": 0}), encoding="utf-8")
    monkeypatch.setenv("PATH", f"{bin_dir}{__import__('os').pathsep}/usr/bin:/bin")
    monkeypatch.setenv("FAKE_GH_STATE", str(state))
    monkeypatch.setenv("FAKE_GH_LOG", str(log))
    monkeypatch.setattr(github, "_gh_run", REAL_GH_RUN)
    monkeypatch.setattr(github, "github_token_from_env_or_settings", lambda: "")
    first = _merge(world)
    assert "PR_MERGE_UNKNOWN" in first and "failure=exit, exit_code=1, http_status=502" in first
    (a,) = _receipts(world)
    assert "sent no new merge request" in _merge(world)
    continued = _merge(world, continue_receipt_id=a["receipt_id"])
    assert "merge: merged" in continued, continued
    calls = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
    puts = [c for c in calls if isinstance(c, list) and "PUT" in c]
    assert puts == [["api", ROUTE, "--hostname", "github.com", "--method", "PUT", "--input", "-"]] * 2
    requests = [c["stdin"] for c in calls if isinstance(c, dict)]
    assert [(r["sha"], r["merge_method"]) for r in requests] == [(HEAD, "squash")] * 2
    assert all("Ouroboros-Qualification:" in r["commit_message"] for r in requests)
    stored = _by_id(world)
    successor = next(r for r in stored.values() if r.get("continues") == a["receipt_id"])
    assert (successor["outcome"]["status"], successor["effect"]["merge_sha"]) == ("merged", MERGE)
    assert stored[a["receipt_id"]]["outcome"]["status"] == "unknown"
    assert merge_receipts.public_block(successor) in json.loads(state.read_text(encoding="utf-8"))["pr"]["body"]


def test_a_decision_without_review_arguments_carries_the_review_of_the_request_it_continues(world):
    """The decision repeats the same merge: without review arguments it carries the earlier request's review."""
    checks = [{"name": "full_battery", "status": "passed", "detail": "exact candidate"}]
    world.gh.merge = "timeout_open"
    _merge(world, review_task_ids=["review-1"], reviewed_head_sha=HEAD, reviewed_base_sha=BASE, review_verdict="PASS",
           checks=checks)
    (a,) = _receipts(world)
    world.gh.merge = "ok"
    _merge(world, continue_receipt_id=a["receipt_id"])
    b = next(r for r in _receipts(world) if r.get("continues") == a["receipt_id"])
    assert b["review"]["declared"] == a["review"]["declared"]
    assert [row["task_id"] for row in b["review"]["host_observed"]] == ["review-1"]
    assert b["qualification"]["checks"][0] == {**checks[0], "source": "author_declared"}
    from ouroboros.update_qualification import commit_qualification

    assert commit_qualification(world.gh.merge_inputs[-1]) == b["qualification"]  # the decided request's own trailer
