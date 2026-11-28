"""#1333: ``pr_merge`` keeps reviewed, current and merged identities apart.

Drives the real tool handler over a stateful fake ``gh`` (no network) that answers the one synchronous REST
merge request the way ``gh api`` does: the intent row lands before the merge effect, a missing review is loud
but never a lock, queued/auto-merge is not "merged", an unknown outcome is only read back by a plain repeat, a
failed publication never repeats the merge, and the public PR-body block carries no private identifiers while
every unrelated byte of the body survives.
"""

from __future__ import annotations

import json
import re
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

    def __call__(self, args, ctx, timeout=30, input_data=None, *, repo=github._GENERIC_TRANSPORT, diagnostic_only=False):
        self.calls.append(list(args))
        if (args[:2] in (["pr", "view"], ["api", "graphql"]) and self.fail_view_after_merge
                and any(sent(c) for c in self.calls)):
            return github.GhResult(False, "⚠️ GH_ERROR: boom", 1, 502, "exit")
        if args[:2] == ["pr", "view"]:
            fields = args[args.index("--json") + 1].split(",")
            return github.GhResult(True, json.dumps({k: self.pr[k] for k in fields if k in self.pr}), 0, None, "")
        if args[:2] == ["api", "graphql"]:
            fields = set(re.findall(r"\b[A-Za-z_]\w*\b", json.loads(input_data)["query"]))
            snapshot = {key: value for key, value in self.pr.items() if key in fields}
            return github.GhResult(True, json.dumps({"data": {"repository": {"pullRequest": snapshot}}}), 0, None, "")
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
# What the tool says while a receipt of the PR is observed and of an owed projection, and the read it names for
# status: no answer promises what a later pr_merge call will do, since another call's readback can end any hold.
LOOKUP = "github_read(method='get', kind='pr', target=7, repo='github.com/octo/demo') reads the PR without merging."
# Beside a readback that shows auto-merge on, the read names the fields that show the queue (the concise PR omits it).
QUEUE_LOOKUP = LOOKUP.replace("')", "', detail='full', fields=['state', 'merged_at', 'auto_merge'])")
ONLY_TO_MERGE = ("pr_merge reads back only as long as a receipt of this PR is still observed, then may send a new merge "
                 "request: call it only to merge.")
RETRY = ("a later pr_merge call retries it only if it still reads this receipt back, and is otherwise a new merge "
         "request: call it only to merge.")


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
    # Its last readback showed the queue, so the read the refusal names shows the queue too.
    assert QUEUE_LOOKUP.removesuffix(" reads the PR without merging.") + " reads it without merging" in out
    assert _receipts(world) == [before] and _merges(world) == 1


def test_an_unknown_outcome_is_read_back_by_plain_repeats_and_continued_only_by_a_decision(world):
    """#2762: an open PR at the same head without a queue does not make a plain repeat send again; only the explicit
    continue_receipt_id does, once, and the earlier request keeps its unknown outcome."""
    world.gh.merge = "timeout_open"
    first = _merge(world)
    (unknown,) = _receipts(world)
    assert "PR_MERGE_UNKNOWN" in first and _merges(world) == 1
    # The answer names the decision and the read that cannot merge, and promises nothing of a later plain call.
    assert (f"{ONLY_TO_MERGE} To decide one new request for the same head and method, "
            f"call pr_merge with continue_receipt_id='{unknown['receipt_id']}'. {LOOKUP}") in first
    assert "calls only read" not in first
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
    assert "could not be read (⚠️ GH_ERROR: HTTP 502); nothing was merged" in _merge(world)
    monkeypatch.setattr(github, "_gh_run", original)
    world.gh.merge = "timeout_open"
    _merge(world)
    (earlier,) = _receipts(world)
    monkeypatch.setattr(github, "_gh_run", unreadable)
    out = _merge(world)
    assert "pr_unreadable" in out and "nothing was merged" not in out and "sent no new merge request" in out
    assert "GH_ERROR: HTTP 502" in out
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


def test_snapshot_refusal_scopes_receipts_by_resolved_url(world, monkeypatch):
    gh = world.gh
    other_url = "https://github.com/other/demo/pull/7"
    gh.merge = "timeout_open"
    assert "PR_MERGE_UNKNOWN" in _merge(world)
    (first,) = _receipts(world)
    gh.pr["url"] = other_url

    def unreadable(args, *a, **kw):
        if args[:2] == ["api", "graphql"]:
            return github.GhResult(False, "snapshot unavailable", 1, 502, "exit")
        return gh(args, *a, **kw)

    monkeypatch.setattr(github, "_gh_run", unreadable)
    refused = _merge(world)
    assert "pr_unreadable" in refused and other_url in refused
    assert URL not in refused and first["receipt_id"] not in refused
    assert _receipts(world) == [first] and _merges(world) == 1

    monkeypatch.setattr(github, "_gh_run", gh)
    assert "PR_MERGE_UNKNOWN" in _merge(world)  # B's healthy read can authorize its own independent request
    first, second = _receipts(world)

    def no_target(args, *a, **kw):
        if args[:2] == ["pr", "view"]:
            return github.GhResult(False, "target unavailable", 1, 502, "exit")
        return gh(args, *a, **kw)

    monkeypatch.setattr(github, "_gh_run", no_target)
    refused = _merge(world)
    for receipt in (first, second):
        assert (f"the earlier merge request for {receipt['repo']['url']} is still unresolved "
                f"(receipt_id={receipt['receipt_id']}, outcome unknown)") in refused
    # Reading is offered through github_read, which cannot merge; pr_merge is not called a readback tool.
    reads = [f"github_read(method='get', kind='pr', target=7, repo='github.com/{slug}')" for slug in ("octo/demo", "other/demo")]
    assert f"{reads[0]} or {reads[1]} reads it without merging" in refused
    assert "then may send a new merge request" in refused and "continue_receipt_id" not in refused
    assert _receipts(world) == [first, second] and _merges(world) == 2


def _fail_snapshot(world, monkeypatch):
    gh = world.gh

    def incomplete(args, *a, **kw):
        if args[:2] == ["api", "graphql"]:
            gh.calls.append(args)
            return github.GhResult(False, "snapshot unavailable", 1, 502, "exit")
        return gh(args, *a, **kw)

    monkeypatch.setattr(github, "_gh_run", incomplete)


@pytest.mark.parametrize("mode,status", [("ok", "merged"), ("queued", "queued"), ("timeout_open", "unknown")])
def test_unchanged_partial_readback_keeps_receipt_and_projections(world, monkeypatch, mode, status):
    world.gh.merge = mode
    _merge(world, reviewed_head_sha=HEAD, reviewed_base_sha=BASE)
    (first,) = _receipts(world)
    assert first["outcome"]["status"] == status and first["observed_after"]["base_sha"] == BASE
    before = len(world.gh.calls), world.gh.pr["body"], list(world.ctx.pending_events)

    _fail_snapshot(world, monkeypatch)
    out = _merge(world)
    assert "sent no new merge request" in out and "snapshot unavailable" in out and "retained receipt" in out
    assert _receipts(world) == [first]
    assert all("PATCH" not in args and not sent(args) for args in world.gh.calls[before[0]:])
    assert (world.gh.pr["body"], world.ctx.pending_events) == before[1:]


def test_partial_retarget_does_not_infer_a_fresh_base_but_later_merge_is_visible(world, monkeypatch):
    world.gh.merge = "timeout_open"
    _merge(world, reviewed_head_sha=HEAD, reviewed_base_sha=BASE)
    (first,) = _receipts(world)
    _fail_snapshot(world, monkeypatch)
    new_base = "e" * 40
    world.gh.pr.update(baseRefName="other-base", baseRefOid=new_base)
    assert "does not establish the current PR base SHA" in _merge(world)
    (retargeted,) = _receipts(world)
    assert (retargeted["observed_after"]["base_ref"], retargeted["observed_after"]["base_sha"]) == ("other-base", "")
    assert retargeted["observed_before"] == first["observed_before"]
    assert "base_unavailable" in retargeted["coverage"]["gaps"]

    world.gh.pr.update(state="MERGED", mergeCommit={"oid": MERGE})
    assert "merge: merged" in _merge(world)
    (merged,) = _receipts(world)
    assert merged["receipt_id"] == first["receipt_id"] and merged["outcome"]["merge_sha"] == MERGE
    assert merged["publication"]["body"] == {"status": "published"}
    assert merged["observed_after"]["base_sha"] == ""
    monkeypatch.setattr(github, "_gh_run", world.gh)
    assert "merge: merged" in _merge(world)
    assert _receipts(world)[0]["observed_after"]["base_sha"] == new_base and _merges(world) == 1


def test_partial_retry_repairs_only_the_owed_projection(world, monkeypatch):
    world.gh.fail_edit = True
    _merge(world)
    (first,) = _receipts(world)
    assert first["publication"]["body"]["status"] == "gap"
    before = list(world.ctx.pending_events)
    world.gh.fail_edit = False
    _fail_snapshot(world, monkeypatch)
    assert "sent no new merge request" in _merge(world)
    (again,) = _receipts(world)
    assert again["publication"]["body"] == {"status": "published", "seq": 1}  # the result recorded after the gap
    assert again["publication"]["card"] == first["publication"]["card"] and world.ctx.pending_events == before
    assert all(again[key] == first[key] for key in ("revision", "observed_after", "coverage")) and _merges(world) == 1


@pytest.mark.parametrize("partial", [True, False], ids=["gh-view", "full-snapshot"])
def test_a_read_from_before_a_concurrent_settlement_never_rewrites_it(world, monkeypatch, partial):
    """A delayed call keeps the queued receipt a concurrent call recorded after the delayed
    call's own reads, gh's view when its snapshot fails or a full snapshot read before that call; neither older read
    settles it as refused while auto-merge is on."""
    gh, concurrent = world.gh, {}
    gh.merge = "refused"  # GitHub refuses the concurrent request; another actor's auto-merge shows on its readback

    def enable_auto_merge(args, *a, **kw):
        result = gh(args, *a, **kw)
        if sent(args):
            gh.pr["autoMergeRequest"] = {"enabledAt": "now"}
        return result

    def late_snapshot(args, *a, **kw):
        if args[:2] == ["api", "graphql"] and not concurrent:
            concurrent["snapshot"] = gh(args, *a, **kw)  # read before the concurrent call
            monkeypatch.setattr(github, "_gh_run", enable_auto_merge)
            concurrent["out"] = _merge(world, reviewed_head_sha=HEAD, reviewed_base_sha=BASE)
            (concurrent["receipt"],) = _receipts(world)
            concurrent["body"], concurrent["events"] = gh.pr["body"], list(world.ctx.pending_events)
            monkeypatch.setattr(github, "_gh_run", late_snapshot)
            return github.GhResult(False, "snapshot unavailable", 1, 502, "exit") if partial else concurrent["snapshot"]
        return gh(args, *a, **kw)

    monkeypatch.setattr(github, "_gh_run", late_snapshot)
    out = _merge(world, reviewed_head_sha=HEAD, reviewed_base_sha=BASE)
    assert "merge: queued" in concurrent["out"] and "merge: queued" in out and ("retained receipt" in out) is partial
    assert concurrent["receipt"]["outcome"] == {"status": "queued", "reason": "auto_merge_pending"}
    assert _receipts(world) == [concurrent["receipt"]]
    assert (gh.pr["body"], world.ctx.pending_events) == (concurrent["body"], concurrent["events"])
    assert "Outcome: **queued**" in gh.pr["body"] and _merges(world) == 1


@pytest.mark.parametrize("partial", [True, False], ids=["gh-view", "full-snapshot"])
def test_a_readback_recorded_during_this_calls_reads_keeps_the_receipt_observed(world, monkeypatch, partial):
    """GitHub refused the request and the PR was closed. A delayed call reads it reopened without a queue;
    while it reads, another call reads back another actor's auto-merge and records the queue. The delayed call's read
    follows the revision it read before GitHub, not the one it rereads afterwards, so it never releases the receipt."""
    gh, concurrent = world.gh, {}
    gh.merge = "refused"

    def closed_after_the_request(args, *a, **kw):
        result = gh(args, *a, **kw)
        if sent(args):
            gh.pr["state"] = "CLOSED"
        return result

    monkeypatch.setattr(github, "_gh_run", closed_after_the_request)
    _merge(world)
    assert _receipts(world)[0]["outcome"] == {"status": "unknown", "reason": "pr_state_closed"}
    gh.pr["state"] = "OPEN"

    def late_snapshot(args, *a, **kw):
        if args[:2] == ["api", "graphql"] and not concurrent:
            concurrent["snapshot"] = gh(args, *a, **kw)  # read before the other call's readback
            gh.pr["autoMergeRequest"] = {"enabledAt": "by another actor"}
            monkeypatch.setattr(github, "_gh_run", gh)
            concurrent["out"] = _merge(world)
            (concurrent["receipt"],) = _receipts(world)
            concurrent["body"], concurrent["events"] = gh.pr["body"], list(world.ctx.pending_events)
            monkeypatch.setattr(github, "_gh_run", late_snapshot)
            return github.GhResult(False, "snapshot unavailable", 1, 502, "exit") if partial else concurrent["snapshot"]
        return gh(args, *a, **kw)

    monkeypatch.setattr(github, "_gh_run", late_snapshot)
    out = _merge(world)
    assert concurrent["receipt"]["outcome"] == {"status": "queued", "reason": "auto_merge_pending"}
    assert "merge: queued" in out and "A new pr_merge call" not in out
    assert _receipts(world) == [concurrent["receipt"]]
    assert (gh.pr["body"], world.ctx.pending_events) == (concurrent["body"], concurrent["events"]) and _merges(world) == 1


@pytest.mark.parametrize("partial", [True, False], ids=["gh-view", "full-snapshot"])
def test_a_settlement_recorded_after_the_reread_outranks_this_calls_older_read(world, monkeypatch, partial):
    """Freshness is decided under the receipt lock, not at the reread. Another call settles the receipt queued
    after this call reread it and before this call writes; this call's older read (an OPEN PR without auto-merge,
    from gh's view or a full snapshot) never returns it to refused, and adds no write, edit or card event."""
    gh = world.gh
    gh.merge, gh.fail_view_after_merge = "refused", True  # GitHub refuses; the sender's own readback fails
    _merge(world)
    (first,) = _receipts(world)
    assert first["outcome"] == {"status": "unknown", "reason": "readback_failed"}
    gh.fail_view_after_merge = False
    if partial:
        _fail_snapshot(world, monkeypatch)
    write, fresher = merge_receipts.write_receipt, {}

    def settled_first(drive_root, task_id, receipt, **kw):
        if not fresher and (receipt.get("outcome") or {}).get("status") == "refused":
            fresher["writing"] = True  # this call's older read is about to be written
            gh.pr["autoMergeRequest"] = {"enabledAt": "now"}
            monkeypatch.setattr(github, "_gh_run", gh)
            fresher["out"] = _merge(world)
            (fresher["receipt"],) = _receipts(world)
            fresher["body"], fresher["events"] = gh.pr["body"], list(world.ctx.pending_events)
        return write(drive_root, task_id, receipt, **kw)

    monkeypatch.setattr(merge_receipts, "write_receipt", settled_first)
    out = _merge(world)
    assert "merge: queued" in fresher["out"] and "merge: queued" in out
    assert fresher["receipt"]["outcome"] == {"status": "queued", "reason": "auto_merge_pending"}
    assert _receipts(world) == [fresher["receipt"]]
    assert (gh.pr["body"], world.ctx.pending_events) == (fresher["body"], fresher["events"])
    assert _merges(world) == 1


def test_after_a_refusal_settled_from_gh_view_reading_never_points_at_pr_merge(world, monkeypatch):
    """Once gh's view settles a request GitHub refused, the next plain pr_merge may send, and the
    tool says so; a read that cannot merge is named instead of a pr_merge readback."""
    world.gh.merge, world.gh.fail_view_after_merge = "refused", True
    _merge(world)
    assert _receipts(world)[0]["outcome"] == {"status": "unknown", "reason": "readback_failed"}
    world.gh.fail_view_after_merge = False
    _fail_snapshot(world, monkeypatch)
    out = _merge(world)
    assert _receipts(world)[0]["outcome"] == {"status": "refused"}
    assert "A new pr_merge call for the PR's current head may send one." in out
    assert "github_read(method='get', kind='pr', target=7, repo='github.com/octo/demo') reads the PR without merging" in out
    assert "calls only read it back" not in out and "Read the recorded repository back" not in out
    monkeypatch.setattr(github, "_gh_run", world.gh)
    world.gh.merge = "ok"
    assert "merge: merged" in _merge(world) and _merges(world) == 2  # the request the tool said may follow


def test_gh_view_completes_a_merged_receipt_and_fetches_only_missing_commit_facts(world, monkeypatch):
    """Without GitHub's snapshot a confirmed merge still gains the merge SHA and commit facts it lacked,
    keeping the full snapshot's base and the attribution; commit facts already known are not fetched again."""
    gh = world.gh

    def lagging_merge_commit(args, *a, **kw):
        result = gh(args, *a, **kw)
        if args[:2] == ["api", "graphql"] and any(sent(c) for c in gh.calls):
            envelope = json.loads(result.text)  # the readback right after the merge names no merge commit yet
            envelope["data"]["repository"]["pullRequest"]["mergeCommit"] = None
            return github.GhResult(True, json.dumps(envelope), 0, None, "")
        return result

    monkeypatch.setattr(github, "_gh_run", lagging_merge_commit)
    _merge(world)
    (first,) = _receipts(world)
    assert first["outcome"]["status"] == "merged" and first["outcome"]["merge_sha"] == ""
    assert first["outcome"]["attribution"] == "this_call" and "merge_tree" not in first["outcome"]
    assert first["observed_after"]["base_sha"] == BASE

    _fail_snapshot(world, monkeypatch)
    calls = len(gh.calls)
    assert "merge: merged" in _merge(world)
    (merged,) = _receipts(world)
    assert (merged["outcome"]["merge_sha"], merged["outcome"]["merge_tree"]) == (MERGE, TREE)
    assert merged["outcome"]["merge_parents"] == [BASE, HEAD] and merged["outcome"]["attribution"] == "this_call"
    assert merged["observed_after"] == first["observed_after"]  # the full snapshot's base stays
    assert f"merge commit `{MERGE}`" in gh.pr["body"] and f"base `{BASE}`" in gh.pr["body"]
    assert any(c[:1] == ["api"] and c[1].endswith(f"/commits/{MERGE}") for c in gh.calls[calls:])
    calls, events = len(gh.calls), list(world.ctx.pending_events)
    assert "merge: merged" in _merge(world)
    assert not any(c[:1] == ["api"] and ("/commits/" in c[1] or "PATCH" in c) for c in gh.calls[calls:])
    assert _receipts(world) == [merged] and world.ctx.pending_events == events and _merges(world) == 1


@pytest.mark.parametrize("partial", [True, False], ids=["gh-view", "full-snapshot"])
def test_a_readback_that_cannot_reread_tree_or_body_keeps_the_confirmed_merge(world, monkeypatch, partial):
    """A merged head's tree never changes and a projection confirmed for this revision stays
    confirmed, so a readback (gh's view alone, or GitHub's full snapshot) that cannot reread the head's tree or the
    PR body keeps the confirmed coverage and publication: nothing is written, edited or sent as a new card event."""
    _merge(world, reviewed_head_sha=HEAD, reviewed_base_sha=BASE)
    (first,) = _receipts(world)
    assert first["outcome"]["status"] == "merged" and first["coverage"]["reviewed_head_tree"] == TREE
    assert "tree_comparison_unavailable" not in first["coverage"]["gaps"]
    assert first["publication"]["body"] == {"status": "published"}
    gh, calls, body, events = world.gh, len(world.gh.calls), world.gh.pr["body"], list(world.ctx.pending_events)

    def only_the_technical_view(args, *a, **kw):
        if ((partial and args[:2] == ["api", "graphql"]) or (args[:1] == ["api"] and "/commits/" in args[1])
                or (args[:2] == ["pr", "view"] and "body" in args[args.index("--json") + 1].split(","))):
            gh.calls.append(list(args))
            return github.GhResult(False, "⚠️ GH_ERROR: unavailable (HTTP 502)", 1, 502, "exit")
        return gh(args, *a, **kw)

    monkeypatch.setattr(github, "_gh_run", only_the_technical_view)
    out = _merge(world, reviewed_head_sha=HEAD, reviewed_base_sha=BASE)
    assert "merge: merged" in out and ("retained receipt" in out) is partial and "sent no new merge request" in out
    assert _receipts(world) == [first]
    assert (world.gh.pr["body"], world.ctx.pending_events) == (body, events)
    assert not any("/commits/" in c[1] or "PATCH" in c for c in gh.calls[calls:] if c[:1] == ["api"])


@pytest.mark.parametrize("mode", ["queued", "ok"])
def test_a_senders_partial_readback_keeps_a_concurrent_full_snapshot(world, monkeypatch, mode):
    """The sender's view after its one request (no base SHA) never erases the base, coverage, PR
    block or card that a concurrent plain call recorded from GitHub's full snapshot of the same facts, and the
    sender keeps its own effect and the attribution its merge answer proves."""
    gh, seen = world.gh, {}
    gh.merge = mode

    def observer_during_readback(args, *a, **kw):
        if args[:2] == ["api", "graphql"] and any(sent(c) for c in gh.calls) and not seen:
            monkeypatch.setattr(github, "_gh_run", gh)
            seen["out"] = _merge(world, reviewed_head_sha=HEAD, reviewed_base_sha=BASE)
            (seen["receipt"],) = _receipts(world)
            seen["body"], seen["events"] = gh.pr["body"], list(world.ctx.pending_events)
            monkeypatch.setattr(github, "_gh_run", observer_during_readback)
            return github.GhResult(False, "snapshot unavailable", 1, 502, "exit")
        return gh(args, *a, **kw)

    monkeypatch.setattr(github, "_gh_run", observer_during_readback)
    _merge(world, reviewed_head_sha=HEAD, reviewed_base_sha=BASE)
    (stored,) = _receipts(world)
    observed = seen["receipt"]
    assert "sent no new merge request" in seen["out"] and _merges(world) == 1
    assert observed["observed_after"]["base_sha"] == BASE and stored["observed_after"] == observed["observed_after"]
    assert stored["effect"] == observed["effect"] and "base_unavailable" not in stored["coverage"]["gaps"]
    if mode == "queued":  # the same facts: nothing is written, edited or sent as a new card event
        assert stored["effect"]["failure"] == "timeout" and _receipts(world) == [observed]
        assert (gh.pr["body"], world.ctx.pending_events) == (seen["body"], seen["events"])
    else:  # the sender proves its own merge: one new revision, its base still the full snapshot's
        assert stored["effect"]["ok"] and stored["effect"]["merge_sha"] == MERGE
        assert (stored["outcome"]["attribution"], observed["outcome"]["attribution"]) == ("this_call", "unproven")
        assert stored["revision"] == observed["revision"] + 1
        assert world.ctx.pending_events[:-1] == seen["events"] and len(world.ctx.pending_events) == len(seen["events"]) + 1
        assert f"base `{BASE}`" in gh.pr["body"] and "base `unavailable`" not in gh.pr["body"]


def test_the_senders_readback_after_its_effect_outranks_an_observation_from_before_it(world, monkeypatch):
    """The freshness rule never ranks the sender's own readback, read after its effect row, below an observation
    recorded before that row. Another call saw another actor's auto-merge while the request was in flight; GitHub
    refused the request, and the sender's readback shows that auto-merge off again."""
    gh, during = world.gh, {}
    gh.merge = "refused"

    def queue_seen_during_the_send(args, *a, **kw):
        if sent(args) and not during:
            gh.pr["autoMergeRequest"] = {"enabledAt": "now"}
            monkeypatch.setattr(github, "_gh_run", gh)
            during["out"] = _merge(world)
            (during["receipt"],) = _receipts(world)
            monkeypatch.setattr(github, "_gh_run", queue_seen_during_the_send)
            gh.pr["autoMergeRequest"] = None
        return gh(args, *a, **kw)

    monkeypatch.setattr(github, "_gh_run", queue_seen_during_the_send)
    out = _merge(world)
    assert during["receipt"]["outcome"] == {"status": "queued", "reason": "auto_merge_pending"}
    (stored,) = _receipts(world)
    assert stored["outcome"] == {"status": "refused"} and stored["effect"]["failure"] == "server_refused"
    assert stored["observed_after"]["auto_merge"] is False and "PR_MERGE_REFUSED" in out and _merges(world) == 1


def test_a_view_older_than_the_senders_answer_row_settles_nothing(world, monkeypatch):
    """A view gh read before the sender recorded GitHub's merge answer settles nothing on that receipt; the
    sender's own readback will. Its in-flight row stays exactly as the sender wrote it."""
    intent = {"schema": 1, "receipt_id": "in-flight", "state": "intent_recorded", "number": 7, "repo": {"url": URL},
              "launch_operation_ids": [], "requested": {"method": "squash", "expected_head_sha": HEAD},
              "observed_before": {}, "review": {}, "coverage": {}}
    merge_receipts.write_receipt(world.root, "merge-task", intent)
    gh, answered = world.gh, {}

    def answer_recorded_during_the_snapshot(args, *a, **kw):
        if args[:2] == ["api", "graphql"] and not answered:
            (answered["row"],) = [merge_receipts.write_receipt(world.root, "merge-task", {
                **intent, "state": "effect_attempted", "effect": {"ok": True, "exit_code": 0, "http_status": None,
                                                                  "failure": "", "output": MERGED, "merge_sha": MERGE}})]
            return github.GhResult(False, "snapshot unavailable", 1, 502, "exit")
        return gh(args, *a, **kw)

    monkeypatch.setattr(github, "_gh_run", answer_recorded_during_the_snapshot)
    out = _merge(world)
    assert "no recorded outcome yet" in out and "retained receipt" in out
    assert _receipts(world) == [answered["row"]] and "outcome" not in answered["row"] and _merges(world) == 0


# --- one rule for which readback is current; commit facts and publications join by their own -----------------


@pytest.mark.parametrize("observer", ["read-during-the-send", "claim-loser"])
def test_a_read_from_before_the_effect_row_never_settles_over_the_senders_readback(world, monkeypatch, observer):
    """Another call's GitHub reads predate the sender's effect row, and it writes after that row but
    before the sender's own readback. It read back no request: the receipt stays unsettled, and the sender's readback
    stands (refused once another actor's queue is gone; queued while one shows)."""
    from threading import Event, Thread, current_thread

    gh, write, loser = world.gh, merge_receipts.write_receipt, observer == "claim-loser"
    gh.merge = "refused"
    read, effect_recorded, done, said = Event(), Event(), Event(), {}

    def other_call():
        try:
            said["text"] = _merge(world)
        finally:
            done.set()

    other = Thread(target=other_call, name="observer")

    def transport(args, *a, **kw):
        if sent(args):  # only the sender gets this far
            if not loser:
                gh.pr["autoMergeRequest"] = {"enabledAt": "by another actor"}  # what the other call reads meanwhile
                other.start()
                assert read.wait(5)
                gh.pr["autoMergeRequest"] = None
            result = gh(args, *a, **kw)
            if loser:
                gh.pr["autoMergeRequest"] = {"enabledAt": "by another actor"}  # what the sender reads back
            return result
        result = gh(args, *a, **kw)
        if current_thread() is other and not loser and args[:2] == ["api", "graphql"]:
            read.set()
            assert effect_recorded.wait(5)
        return result

    def ordered(drive_root, task_id, receipt, **kw):
        if current_thread() is other and loser and kw.get("claim"):
            read.set()  # it read the PR open and decided to claim before the sender did
            assert effect_recorded.wait(5)
        stored = write(drive_root, task_id, receipt, **kw)
        if current_thread() is not other and receipt.get("state") == "effect_attempted":
            effect_recorded.set()
            assert done.wait(5)  # the other call writes now, before the sender reads back
        return stored

    monkeypatch.setattr(github, "_gh_run", transport)
    monkeypatch.setattr(merge_receipts, "write_receipt", ordered)
    if loser:
        other.start()
        assert read.wait(5)
    text = _merge(world)
    other.join(5)
    (stored,) = _receipts(world)
    assert "no recorded outcome yet" in said["text"] and "sent no new merge request" in said["text"]
    assert stored["effect"]["failure"] == "server_refused" and stored["state"] == "settled" and _merges(world) == 1
    if loser:
        assert stored["outcome"] == {"status": "queued", "reason": "auto_merge_pending"}
        assert stored["observed_after"]["auto_merge"] is True and "Outcome: **queued**" in gh.pr["body"]
    else:
        assert stored["outcome"] == {"status": "refused"} and stored["observed_after"]["auto_merge"] is False
        assert "PR_MERGE_REFUSED" in text and "merge-receipt" not in gh.pr["body"]


@pytest.mark.parametrize("found", ["gh-view", "full-snapshot", "lost-the-claim"])
def test_reads_from_before_a_request_existed_are_no_readback_of_it(world, monkeypatch, found):
    """A call began with no receipt in view and read the PR; another call then recorded its intent, and its request
    timed out. The first call meets that receipt after its reads (rereading the rows, or losing the claim), so they
    predate the request: it joins the receipt and reports it, settling nothing while the request is in flight. Where
    it had read auto-merge on (GitHub's snapshot, or gh's view when the snapshot fails) and the queue was switched
    off before the intent, both of the sender's own reads show no queue: the outcome stays unknown, nothing shows a
    queue, and the explicit continuation the answer names is not barred."""
    from threading import Event, Thread, current_thread

    gh, write, loser = world.gh, merge_receipts.write_receipt, found == "lost-the-claim"
    gh.merge = "timeout_open"
    if not loser:
        gh.pr["autoMergeRequest"] = {"enabledAt": "by another actor"}
    read, intent, done, said = Event(), Event(), Event(), {}

    def earlier_call():
        try:
            said["text"] = _merge(world)
        finally:
            done.set()

    earlier = Thread(target=earlier_call, name="earlier-reader")

    def transport(args, *a, **kw):
        result = gh(args, *a, **kw)
        if not loser and current_thread() is earlier and args[:2] == ["api", "graphql"] and not read.is_set():
            read.set()  # it has read the queue; the sender's intent is recorded before it looks at the receipts again
            assert intent.wait(5)
            if found == "gh-view":
                return github.GhResult(False, "snapshot unavailable", 1, 502, "exit")
        return result

    def ordered(drive_root, task_id, receipt, **kw):
        if loser and current_thread() is earlier and kw.get("claim"):
            read.set()  # it read the PR open without a queue; the sender claims before it does
            assert intent.wait(5)
        stored = write(drive_root, task_id, receipt, **kw)
        if current_thread() is not earlier and kw.get("claim"):
            intent.set()
            assert done.wait(5)  # the earlier call finishes while the request is in flight
            (said["in_flight"],) = _receipts(world)
        return stored

    monkeypatch.setattr(github, "_gh_run", transport)
    monkeypatch.setattr(merge_receipts, "write_receipt", ordered)
    earlier.start()
    assert read.wait(5)
    gh.pr["autoMergeRequest"] = None
    out = _merge(world)
    earlier.join(5)
    (stored,) = _receipts(world)
    assert said["in_flight"]["state"] == "intent_recorded" and "outcome" not in said["in_flight"]
    assert "no recorded outcome yet" in said["text"] and "sent no new merge request" in said["text"]
    assert "continue_receipt_id=" not in said["text"]  # its request may still be sent: no new decision is offered
    assert stored["outcome"] == {"status": "unknown", "reason": "pr_state_open_after_unconfirmed_effect"}
    assert stored["observed_after"]["auto_merge"] is False and stored["effect"]["failure"] == "timeout"
    assert "merge-receipt" not in gh.pr["body"] and world.ctx.pending_events == []
    assert f"continue_receipt_id='{stored['receipt_id']}'" in out and _merges(world) == 1
    monkeypatch.setattr(merge_receipts, "write_receipt", write)
    gh.merge = "ok"
    assert "merge: merged" in _merge(world, continue_receipt_id=stored["receipt_id"]) and _merges(world) == 2


@pytest.mark.parametrize("change", [{"state": "CLOSED", "autoMergeRequest": None}, {"autoMergeRequest": None}],
                         ids=["closed", "open-without-queue"])
def test_gh_view_of_the_same_facts_keeps_the_readback_beside_an_earlier_queue(world, monkeypatch, change):
    """A full readback no longer shows the queue, which stays the outcome, worded as earlier.
    gh's view of those same facts (no base SHA) keeps that readback's base, its coverage, the revision and both
    projections: nothing is written, edited or sent as a card event."""
    world.gh.merge = "queued"
    _merge(world, reviewed_head_sha=HEAD, reviewed_base_sha=BASE)
    world.gh.pr.update(change)
    assert "queued earlier" in _merge(world)
    (first,) = _receipts(world)
    assert first["outcome"]["status"] == "queued" and first["observed_after"]["base_sha"] == BASE
    assert "base_unavailable" not in first["coverage"]["gaps"]
    calls, body, events = len(world.gh.calls), world.gh.pr["body"], list(world.ctx.pending_events)
    _fail_snapshot(world, monkeypatch)
    out = _merge(world)
    assert "queued earlier" in out and "retained receipt" in out and _receipts(world) == [first]
    assert (world.gh.pr["body"], world.ctx.pending_events) == (body, events)
    assert not any("PATCH" in c or sent(c) for c in world.gh.calls[calls:])


@pytest.mark.parametrize("observed", ["during-the-send", "during-the-readback"])
def test_a_failed_readback_never_replaces_a_complete_one_another_call_recorded(world, monkeypatch, observed):
    """Another call records GitHub's full snapshot while the request is in flight or while the
    sender reads back; both of the sender's own reads then fail. A failed read is no readback: the recorded
    observation and its coverage stay beside the sender's effect."""
    gh, seen = world.gh, {}
    gh.merge = "timeout_open"

    def observe():
        seen["started"] = True
        monkeypatch.setattr(github, "_gh_run", gh)
        _merge(world, reviewed_head_sha=HEAD, reviewed_base_sha=BASE)
        (seen["receipt"],) = _receipts(world)
        monkeypatch.setattr(github, "_gh_run", then_unreadable)
        gh.fail_view_after_merge = True  # GitHub answers neither the snapshot nor gh's view once the request is sent

    def then_unreadable(args, *a, **kw):
        if not seen and (sent(args) if observed == "during-the-send" else
                         args[:2] == ["api", "graphql"] and any(sent(c) for c in gh.calls)):
            observe()
        return gh(args, *a, **kw)

    monkeypatch.setattr(github, "_gh_run", then_unreadable)
    out = _merge(world, reviewed_head_sha=HEAD, reviewed_base_sha=BASE)
    (stored,) = _receipts(world)
    recorded = seen["receipt"]
    assert recorded["outcome"] == {"status": "unknown", "reason": "pr_state_open_after_unconfirmed_effect"}
    assert recorded["observed_after"]["base_sha"] == BASE and "base_unavailable" not in recorded["coverage"]["gaps"]
    assert all(stored[key] == recorded[key] for key in ("outcome", "observed_after", "coverage", "revision"))
    assert stored["effect"]["failure"] == "timeout" and "PR_MERGE_UNKNOWN" in out and _merges(world) == 1


def test_an_older_open_read_never_releases_a_receipt_another_call_read_closed(world, monkeypatch):
    """A release (refused) needs the current readback. GitHub refused the request and the readback failed; a call then
    reads the PR open without a queue, but before it writes another call records the PR closed. The older read
    releases nothing: the receipt stays observed beside the CLOSED readback, and the tool still says so."""
    gh, write, newer = world.gh, merge_receipts.write_receipt, {}
    gh.merge, gh.fail_view_after_merge = "refused", True
    _merge(world)
    gh.fail_view_after_merge = False

    def closed_first(drive_root, task_id, receipt, **kw):
        if not newer and (receipt.get("outcome") or {}).get("status") == "refused":
            newer["started"] = True  # this call's older read is about to be written
            gh.pr["state"] = "CLOSED"
            _merge(world)
            (newer["receipt"],) = _receipts(world)
        return write(drive_root, task_id, receipt, **kw)

    monkeypatch.setattr(merge_receipts, "write_receipt", closed_first)
    out = _merge(world)
    assert newer["receipt"]["outcome"] == {"status": "unknown", "reason": "pr_state_closed"}
    assert _receipts(world) == [newer["receipt"]]
    assert ONLY_TO_MERGE in out and "A new pr_merge call" not in out and _merges(world) == 1


def test_a_failed_read_outranks_no_readback_in_either_order(world):
    """A failed read records no readback. A call that read the effect row before the sender's failed readback was
    settled still records what GitHub showed it, here the release of a refused request; written the other way round,
    the failed read changes nothing."""
    observed = {"state": "OPEN", "head_sha": HEAD, "base_ref": "main", "base_sha": BASE, "merge_state": "CLEAN",
                "is_draft": False, "auto_merge": False}
    unread = {"state": "settled", "observed_after": {}, "outcome": {"status": "unknown", "reason": "readback_failed"}}
    seen = merge_receipts.write_receipt(world.root, "merge-task", {
        "schema": 1, "receipt_id": "r", "state": "effect_attempted", "number": 7, "repo": {"url": URL},
        "launch_operation_ids": [], "requested": {"method": "squash", "expected_head_sha": HEAD},
        "observed_before": {}, "review": {}, "coverage": {},
        "effect": {"ok": False, "exit_code": 1, "http_status": 405, "failure": "server_refused", "output": NOT_MERGEABLE}})
    failed = merge_receipts.write_receipt(world.root, "merge-task", {**seen, **unread})
    assert failed["outcome"] == unread["outcome"] and failed["revision"] == seen["revision"] + 1
    stored = merge_receipts.write_receipt(world.root, "merge-task", {
        **seen, "state": "settled", "observed_after": observed, "outcome": {"status": "refused"}})
    assert (stored["outcome"], stored["observed_after"]) == ({"status": "refused"}, observed)
    assert merge_receipts.write_receipt(world.root, "merge-task", {**stored, **unread}) == stored
    assert _receipts(world) == [stored]


def test_an_older_queue_read_never_replaces_a_newer_closed_readback(world, monkeypatch):
    """A call read the PR open with auto-merge on; before it wrote, another call recorded the PR closed.
    The queue seen stays the outcome, worded as earlier, beside the newer CLOSED readback and its coverage; the
    receipt never shows a live queue on the closed PR."""
    gh, write, newer = world.gh, merge_receipts.write_receipt, {}
    gh.merge = "timeout_open"
    _merge(world, reviewed_head_sha=HEAD, reviewed_base_sha=BASE)
    gh.pr["autoMergeRequest"] = {"enabledAt": "now"}

    def closed_first(drive_root, task_id, receipt, **kw):
        if not newer and (receipt.get("outcome") or {}).get("status") == "queued":
            newer["started"] = True  # this call's older read is about to be written
            gh.pr.update(state="CLOSED", autoMergeRequest=None)
            newer["out"] = _merge(world)
            (newer["receipt"],) = _receipts(world)
        return write(drive_root, task_id, receipt, **kw)

    monkeypatch.setattr(merge_receipts, "write_receipt", closed_first)
    out = _merge(world)
    (stored,) = _receipts(world)
    closed = newer["receipt"]
    assert closed["outcome"] == {"status": "unknown", "reason": "pr_state_closed_after_unconfirmed_effect"}
    assert stored["outcome"] == {"status": "queued", "reason": "auto_merge_pending"}
    assert stored["observed_after"] == closed["observed_after"] and stored["observed_after"]["state"] == "CLOSED"
    assert stored["coverage"] == closed["coverage"] and stored["revision"] == closed["revision"] + 1
    assert out.startswith(f"⚠️ PR #7 merge: {QUEUED_CLOSED};")
    assert f"- Outcome: **queued** {QUEUED_CLOSED.removeprefix('queued ')}\n" in gh.pr["body"] and _merges(world) == 1


@pytest.mark.parametrize("answer", ["ok", "timeout_merged"], ids=["answered", "answer-lost"])
def test_a_senders_readback_through_a_commit_outage_keeps_the_merged_commit_facts(world, monkeypatch, answer):
    """Another call recorded the merge with its commit facts; the sender's own readback then has neither
    GitHub's snapshot nor the commit endpoint. The merge commit's tree and parents, the reviewed head's tree and the
    coverage derived from them stay. An answered request adds the attribution it proves, in one new revision; a
    request whose answer was lost adds nothing, so nothing is written or published again."""
    gh, seen, proven = world.gh, {}, answer == "ok"
    gh.merge = answer

    def outage_after_the_observer(args, *a, **kw):
        if args[:2] == ["api", "graphql"] and any(sent(c) for c in gh.calls) and not seen:
            seen["started"] = True
            monkeypatch.setattr(github, "_gh_run", gh)
            _merge(world, reviewed_head_sha=HEAD, reviewed_base_sha=BASE)
            (seen["receipt"],) = _receipts(world)
            seen["events"] = len(world.ctx.pending_events)
            monkeypatch.setattr(github, "_gh_run", outage_after_the_observer)
        if seen and (args[:2] == ["api", "graphql"] or (args[:1] == ["api"] and "/commits/" in args[1])):
            return github.GhResult(False, "⚠️ GH_ERROR: unavailable (HTTP 502)", 1, 502, "exit")
        return gh(args, *a, **kw)

    monkeypatch.setattr(github, "_gh_run", outage_after_the_observer)
    _merge(world, reviewed_head_sha=HEAD, reviewed_base_sha=BASE)
    (stored,) = _receipts(world)
    recorded = seen["receipt"]
    assert recorded["coverage"]["reviewed_head_tree"] == TREE and recorded["outcome"]["merge_tree"] == TREE
    assert "tree_comparison_unavailable" not in recorded["coverage"]["gaps"]
    assert stored["coverage"] == recorded["coverage"] and stored["observed_after"] == recorded["observed_after"]
    assert stored["outcome"] == {**recorded["outcome"], "attribution": "this_call" if proven else "unproven"}
    assert (stored == recorded) is not proven and stored["revision"] == recorded["revision"] + proven
    assert len(world.ctx.pending_events) == seen["events"] + proven
    assert "tree_comparison_unavailable" not in gh.pr["body"] and _merges(world) == 1


@pytest.mark.parametrize("partial", [True, False], ids=["gh-view", "full-snapshot"])
def test_a_delayed_readback_never_erases_a_head_tree_another_call_recovered(world, monkeypatch, partial):
    """The merged head's tree was unreadable at first. A delayed call, still without it, writes after
    another call read it: the verified tree and the coverage it settles stay, with no new revision or publication."""
    gh, write, recovered = world.gh, merge_receipts.write_receipt, {}

    def no_head_tree(args, *a, **kw):
        if (args[:1] == ["api"] and args[1].endswith(f"/commits/{HEAD}")) or (partial and args[:2] == ["api", "graphql"]
                                                                              and any(sent(c) for c in gh.calls)):
            return github.GhResult(False, "⚠️ GH_ERROR: unavailable (HTTP 502)", 1, 502, "exit")
        return gh(args, *a, **kw)

    def recovered_first(drive_root, task_id, receipt, **kw):
        if recovered.get("armed") and "publication" not in kw:
            recovered["armed"] = False  # the delayed call computed its coverage without the tree and is about to write
            monkeypatch.setattr(github, "_gh_run", gh)
            _merge(world, reviewed_head_sha=HEAD, reviewed_base_sha=BASE)
            (recovered["receipt"],) = _receipts(world)
            recovered["body"], recovered["events"] = gh.pr["body"], list(world.ctx.pending_events)
            monkeypatch.setattr(github, "_gh_run", no_head_tree)
        return write(drive_root, task_id, receipt, **kw)

    monkeypatch.setattr(github, "_gh_run", no_head_tree)
    monkeypatch.setattr(merge_receipts, "write_receipt", recovered_first)
    _merge(world, reviewed_head_sha=HEAD, reviewed_base_sha=BASE)
    assert "tree_comparison_unavailable" in _receipts(world)[0]["coverage"]["gaps"]
    recovered["armed"] = True
    assert "merge: merged" in _merge(world, reviewed_head_sha=HEAD, reviewed_base_sha=BASE)
    read = recovered["receipt"]
    assert read["coverage"]["reviewed_head_tree"] == TREE and "tree_comparison_unavailable" not in read["coverage"]["gaps"]
    assert _receipts(world) == [read]
    assert (gh.pr["body"], world.ctx.pending_events) == (recovered["body"], recovered["events"]) and _merges(world) == 1


def test_a_delayed_failed_publication_never_unconfirms_a_block_another_call_confirmed(world, monkeypatch):
    """Two calls retry the same body gap. One writes and confirms the block; the other's own edit and
    reads then fail. The confirmation stays recorded: a publication is written only over the one its publisher read."""
    gh, confirmed = world.gh, {}
    gh.fail_edit = True
    assert "not confirmed" in _merge(world)
    (owed,) = _receipts(world)
    assert owed["publication"]["body"]["status"] == "gap"
    gh.fail_edit = False

    def another_call_confirms_first(args, *a, **kw):
        if args[:1] == ["api"] and "PATCH" in args and not confirmed:
            confirmed["started"] = True
            monkeypatch.setattr(github, "_gh_run", gh)
            _merge(world)
            (confirmed["receipt"],) = _receipts(world)
            confirmed["body"], confirmed["events"] = gh.pr["body"], list(world.ctx.pending_events)
            monkeypatch.setattr(github, "_gh_run", another_call_confirms_first)
            return github.GhResult(False, "⚠️ GH_ERROR: forbidden", 1, 403, "exit")
        if confirmed and args[:2] == ["pr", "view"] and "body" in args[args.index("--json") + 1].split(","):
            return github.GhResult(False, "⚠️ GH_ERROR: unavailable (HTTP 502)", 1, 502, "exit")
        return gh(args, *a, **kw)

    monkeypatch.setattr(github, "_gh_run", another_call_confirms_first)
    out = _merge(world)
    assert confirmed["receipt"]["publication"]["body"] == {"status": "published", "seq": 1}
    assert _receipts(world) == [confirmed["receipt"]] and "not confirmed" not in out
    assert merge_receipts.public_block(owed) in gh.pr["body"] == confirmed["body"]
    assert world.ctx.pending_events == confirmed["events"] and _merges(world) == 1


def test_a_card_delivered_meanwhile_never_costs_the_block_this_call_read_back(world, monkeypatch):
    """One call writes the owed PR block and reads it back while another records only that the card was delivered.
    The body and the card are confirmed separately: the block stays recorded as published beside the delivery
    without another read of the body, which GitHub no longer answers."""
    from supervisor import terminal_delivery as td

    gh, newer = world.gh, {"body_reads": 0}
    gh.fail_edit = True
    _merge(world)
    (owed,) = _receipts(world)
    assert (owed["publication"]["body"]["status"], owed["publication"]["card"]["status"]) == ("gap", "owed")

    def delivered_meanwhile(args, *a, **kw):
        if args[:1] == ["api"] and "PATCH" in args and "receipt" not in newer:
            td.register_delivery(world.root, owed["publication"]["card"]["delivery_id"])  # the supervisor sent the card
            monkeypatch.setattr(github, "_gh_run", gh)
            _merge(world)  # another call, whose own edit still fails, records the delivery
            monkeypatch.setattr(github, "_gh_run", delivered_meanwhile)
            (newer["receipt"],) = _receipts(world)
            gh.fail_edit = False
        elif "receipt" in newer and args[:2] == ["pr", "view"] and "body" in args[args.index("--json") + 1].split(","):
            newer["body_reads"] += 1
            if newer["body_reads"] > 1:  # GitHub answers this call's readback of its own edit, then no body read
                return github.GhResult(False, "⚠️ GH_ERROR: unavailable (HTTP 502)", 1, 502, "exit")
        return gh(args, *a, **kw)

    monkeypatch.setattr(github, "_gh_run", delivered_meanwhile)
    out = _merge(world)
    (stored,) = _receipts(world)
    delivered = {**owed["publication"]["card"], "status": "delivered"}
    assert newer["receipt"]["publication"] == {"body": owed["publication"]["body"], "card": delivered}
    assert stored["publication"] == {"body": {"status": "published", "seq": 1}, "card": delivered}
    assert "not confirmed" not in out and newer["body_reads"] == 1 and stored["revision"] == owed["revision"]
    assert merge_receipts.public_block(owed) in gh.pr["body"] and _merges(world) == 1


@pytest.mark.parametrize("failed_again", [False, True], ids=["restored", "failed-again-then-restored"])
@pytest.mark.parametrize("readable", [True, False], ids=["body-readable", "body-unreadable"])
def test_a_delayed_failed_repair_never_unconfirms_a_block_another_call_restored(world, monkeypatch, readable,
                                                                                failed_again):
    """An edit removed the confirmed block and one call's repair failed. Before it records that, another call restores
    the block and reads it back (after a third call's repair failed too and was recorded, in the second case). The
    restored block is a later recorded confirmation than the one the failed call read, although both say published:
    the delayed failure is not recorded over it, and no fact revision or card event is added."""
    gh, write, restored = world.gh, merge_receipts.write_receipt, {}
    _merge(world)
    (first,) = _receipts(world)
    assert first["publication"]["body"] == {"status": "published"}
    gh.pr["body"], gh.fail_edit = "Rewritten by the owner.", True

    def unreadable_body(args, *a, **kw):
        if args[:2] == ["pr", "view"] and "body" in args[args.index("--json") + 1].split(","):
            return github.GhResult(False, "⚠️ GH_ERROR: unavailable (HTTP 502)", 1, 502, "exit")
        return gh(args, *a, **kw)

    def restored_first(drive_root, task_id, receipt, **kw):
        if not restored and ((kw.get("publication") or {}).get("body") or {}).get("status") == "gap":
            restored["started"] = True  # this call's failed repair is about to be recorded
            if failed_again:
                assert "not confirmed" in _merge(world)
            gh.fail_edit = False
            assert "not confirmed" not in _merge(world)
            (restored["receipt"],) = _receipts(world)
            restored["body"], restored["events"] = gh.pr["body"], list(world.ctx.pending_events)
            if not readable:
                monkeypatch.setattr(github, "_gh_run", unreadable_body)
        return write(drive_root, task_id, receipt, **kw)

    monkeypatch.setattr(merge_receipts, "write_receipt", restored_first)
    out = _merge(world)
    assert _receipts(world) == [restored["receipt"]] and "not confirmed" not in out
    confirmed = restored["receipt"]["publication"]["body"]
    assert confirmed["status"] == "published" and confirmed != first["publication"]["body"]
    assert restored["receipt"]["revision"] == first["revision"]
    assert merge_receipts.public_block(first) in gh.pr["body"] == restored["body"]
    assert world.ctx.pending_events == restored["events"] and _merges(world) == 1


def _body_read(args):
    return args[:2] == ["pr", "view"] and "body" in args[args.index("--json") + 1].split(",")


UNREADABLE = github.GhResult(False, "⚠️ GH_ERROR: unavailable (HTTP 502)", 1, 502, "exit")


@pytest.mark.parametrize("failed_read", [False, True], ids=["failed-repair", "then-a-failed-read"])
def test_a_restored_block_saved_after_a_failed_repair_stays_confirmed_through_a_read_outage(world, monkeypatch,
                                                                                         failed_read):
    """The reverse order of the case above: both calls read the row that confirmed the block, and the failed repair is
    saved before the restored block (after a third call's body read failed too, in the second case). The restoring call
    read its block back, so its confirmation is recorded over those gaps without another read, which GitHub no longer
    answers; the fact revision, the card and its events stay as they were."""
    gh, state = world.gh, {}
    _merge(world)
    (first,) = _receipts(world)
    events = list(world.ctx.pending_events)
    gh.pr["body"] = "Rewritten by the owner."

    def unreadable_body(args, *a, **kw):
        return UNREADABLE if _body_read(args) else gh(args, *a, **kw)

    def failed_repair_saved_first(args, *a, **kw):
        if args[:1] == ["api"] and "PATCH" in args and not state:
            state["reads"], gh.fail_edit = 0, True  # this call's own repair is about to land
            monkeypatch.setattr(github, "_gh_run", gh)
            assert "not confirmed" in _merge(world)
            if failed_read:
                monkeypatch.setattr(github, "_gh_run", unreadable_body)
                assert "not confirmed" in _merge(world)
            (state["gap"],) = _receipts(world)
            monkeypatch.setattr(github, "_gh_run", failed_repair_saved_first)
            gh.fail_edit = False
        elif state and _body_read(args):
            state["reads"] += 1
            if state["reads"] > 1:  # GitHub answers this call's readback of its own edit, then no body read
                return UNREADABLE
        return gh(args, *a, **kw)

    monkeypatch.setattr(github, "_gh_run", failed_repair_saved_first)
    out = _merge(world)
    assert "not confirmed" not in out
    assert state["gap"]["publication"]["body"] == {"status": "gap", "reason": "edit_failed" if failed_read else "exit",
                                                   "seq": 1}
    (stored,) = _receipts(world)  # saved after the gap, the confirmation is numbered after it
    assert stored["publication"] == {**first["publication"], "body": {"status": "published", "seq": 2}}
    assert stored["revision"] == first["revision"] and world.ctx.pending_events == events
    assert merge_receipts.public_block(first) in gh.pr["body"] and _merges(world) == 1


def test_the_receipt_writer_orders_body_results_by_what_they_confirmed(world):
    """At the writer, a publication made from an older row: a confirmation is saved over a gap of its own number, an
    older failure never over a confirmation, and a gap found after a confirmation saved since keeps its place."""
    _merge(world)
    (first,) = _receipts(world)

    def save(read, body):
        return merge_receipts.write_receipt(world.root, "merge-task", read,
                                            publication={**read["publication"], "body": body})["publication"]["body"]

    gap = merge_receipts.write_receipt(world.root, "merge-task", first, publication={
        **first["publication"], "body": {"status": "gap", "reason": "exit", "seq": 1}})
    assert save(first, {"status": "published", "seq": 1}) == {"status": "published", "seq": 2}
    assert save(first, {"status": "gap", "reason": "timeout", "seq": 1}) == {"status": "published", "seq": 2}
    restored = _receipts(world)[0]["publication"]["body"]  # a gap found from it takes the next number, as published
    found = save(_receipts(world)[0], {"status": "gap", "reason": "exit", "seq": restored["seq"] + 1})
    assert save(gap, {"status": "published", "seq": 2}) == found
    assert _receipts(world)[0]["revision"] == first["revision"]


def test_a_gap_found_after_a_newer_confirmation_outranks_a_delayed_older_one(world, monkeypatch):
    """A genuinely later loss of the block stays recorded. One call restores and reads its block back, but before it
    saves that, an edit removes the block, another call restores and saves it, a further edit removes it, and a third
    call finds the gap and saves it. That gap follows a newer confirmation than the delayed one, so the delayed
    confirmation is not recorded over it, and its caller's next read fails: the receipt keeps the gap."""
    gh, write, state = world.gh, merge_receipts.write_receipt, {}
    _merge(world)
    gh.pr["body"] = "Rewritten by the owner."

    def unreadable_body(args, *a, **kw):
        return UNREADABLE if _body_read(args) else gh(args, *a, **kw)

    def newer_events_first(drive_root, task_id, receipt, **kw):
        if not state and ((kw.get("publication") or {}).get("body") or {}).get("status") == "published":
            state["started"] = True  # this call read its own restored block back and is about to save it
            gh.pr["body"] = "Rewritten again."
            assert "not confirmed" not in _merge(world)
            (state["newer"],) = _receipts(world)
            gh.pr["body"], gh.fail_edit = "Rewritten a third time.", True
            assert "not confirmed" in _merge(world)
            (state["gap"],) = _receipts(world)
            monkeypatch.setattr(github, "_gh_run", unreadable_body)
        return write(drive_root, task_id, receipt, **kw)

    monkeypatch.setattr(merge_receipts, "write_receipt", newer_events_first)
    out = _merge(world)
    assert state["newer"]["publication"]["body"] == {"status": "published", "seq": 1}
    assert state["gap"]["publication"]["body"] == {"status": "gap", "reason": "exit", "seq": 2}
    (stored,) = _receipts(world)
    assert stored["publication"]["body"]["status"] == "gap" and "not confirmed" in out
    assert "merge-receipt" not in gh.pr["body"] and _merges(world) == 1


def test_a_block_an_edit_removed_is_owed_until_a_call_restores_it(world):
    """A confirmed block stays confirmed through a failed re-read, but a body read without it is a verified gap.
    The repair is tried at once; a failed one is recorded as owed in the same revision, and the next call restores
    the block beside the edited text."""
    _merge(world)
    (first,) = _receipts(world)
    world.gh.pr["body"] = "Rewritten by the owner."
    world.gh.fail_edit = True
    assert f"The PR-body receipt block was not confirmed; {RETRY}" in _merge(world)
    (owed,) = _receipts(world)
    assert owed["publication"]["body"] == {"status": "gap", "reason": "exit", "seq": 1}
    assert owed["revision"] == first["revision"]
    world.gh.fail_edit = False
    assert "not confirmed" not in _merge(world)
    (restored,) = _receipts(world)
    assert restored["publication"] == {**first["publication"], "body": {"status": "published", "seq": 2}}
    assert restored["revision"] == first["revision"]
    assert world.gh.pr["body"] == "Rewritten by the owner.\n\n" + merge_receipts.public_block(first) + "\n"
    assert _merges(world) == 1


def test_status_guidance_names_the_read_that_cannot_merge_and_when_a_plain_call_sends(world):
    """GitHub refused the request (405) and its readback failed, so the receipt is observed only until a
    readback shows the PR open without a queue. Another call's readback ends that hold and the next plain call sends,
    which the first answer says instead of promising a read-only repeat."""
    gh = world.gh
    gh.merge, gh.fail_view_after_merge = "refused", True
    first = _merge(world)
    (unknown,) = _receipts(world)
    assert f"{ONLY_TO_MERGE} To decide one new request for the same head and method, call pr_merge with " \
           f"continue_receipt_id='{unknown['receipt_id']}'. {LOOKUP}" in first
    assert "only read it back" not in first and "calls only read" not in first
    gh.fail_view_after_merge = False
    settled = _merge(world)  # another call reads GitHub back: the hold ends
    assert "A new pr_merge call for the PR's current head may send one." in settled and _merges(world) == 1
    gh.merge = "ok"
    assert "merge: merged" in _merge(world) and _merges(world) == 2  # the send the first answer disclosed
    description = next(e.schema for e in github.get_tools() if e.name == "pr_merge")["description"]
    assert ("Call it only to merge, never to check status: github_read(method='get', kind='pr', target=NUMBER, "
            "repo='[HOST/]OWNER/REPO') reads a PR without merging.") in description
    assert "only if that receipt is still observed when the call runs; otherwise it sends" in description
    assert "only reads GitHub back" not in description


@pytest.mark.parametrize("case", ["refused-beside-a-queue", "unconfirmed-beside-a-queue", "merged",
                                  "refused-beside-another-merge"])
def test_an_owed_publication_promises_no_publication_only_repeat(world, monkeypatch, case):
    """A retry of an owed projection is a pr_merge call like any other. Only one that still reads this
    receipt back retries the publication, so the tool promises nothing of a later call and calls none
    publication-only, whatever the outcome; short of a merge it names github_read for status."""
    gh = world.gh
    gh.fail_edit = True
    gh.merge = {"unconfirmed-beside-a-queue": "queued", "merged": "ok"}.get(case, "refused")

    def another_actor(args, *a, **kw):
        result = gh(args, *a, **kw)
        if sent(args) and case == "refused-beside-a-queue":
            gh.pr["autoMergeRequest"] = {"enabledAt": "by another actor"}
        if sent(args) and case == "refused-beside-another-merge":
            gh.pr.update(state="MERGED", mergeCommit={"oid": MERGE}, mergedAt="2026-09-26T00:00:00Z")
        return result

    monkeypatch.setattr(github, "_gh_run", another_actor)
    out = _merge(world)
    (receipt,) = _receipts(world)
    merged = "merge" in case
    assert receipt["outcome"]["status"] == ("merged" if merged else "queued")
    assert receipt["publication"]["body"]["status"] == "gap"
    assert f"⚠️ The PR-body receipt block was not confirmed; {RETRY}" in out
    assert "publication only" not in out and "calls only read" not in out
    assert (f"{ONLY_TO_MERGE} {QUEUE_LOOKUP}" in out) is not merged and LOOKUP not in out


def test_a_chains_hold_is_never_promised_to_last(world, monkeypatch):
    """Both requests of a chain were refused by GitHub. A plain call settles the first as refused while its successor
    is being sent; the successor's readback shows another actor's queue and its PR block is not confirmed. No hold
    here lasts: once that queue is gone one plain call releases the successor too and the next one sends. So the
    successor's answer names github_read for status and calls no later pr_merge read-only or publication-only."""
    gh, write, pending = world.gh, merge_receipts.write_receipt, {}
    gh.merge, gh.fail_view_after_merge = "refused", True
    _merge(world)
    (first,) = _receipts(world)
    gh.fail_view_after_merge, gh.fail_edit = False, True

    def successor_in_flight(args, *a, **kw):
        if sent(args) and "write" in pending:
            pending["stored"] = write(*pending.pop("write"), **pending.pop("kw"))  # the first settles as refused now
            result = gh(args, *a, **kw)
            gh.pr["autoMergeRequest"] = {"enabledAt": "by another actor"}
            return result
        return gh(args, *a, **kw)

    def settles_during_the_send(drive_root, task_id, receipt, **kw):
        if not pending and "publication" not in kw and (receipt.get("outcome") or {}).get("status") == "refused":
            pending.update(write=(drive_root, task_id, receipt), kw=kw)  # this plain call read the PR open, no queue
            pending["out"] = _merge(world, continue_receipt_id=first["receipt_id"])
            return pending["stored"]
        return write(drive_root, task_id, receipt, **kw)

    monkeypatch.setattr(github, "_gh_run", successor_in_flight)
    monkeypatch.setattr(merge_receipts, "write_receipt", settles_during_the_send)
    _merge(world)
    out = pending["out"]
    earlier, successor = sorted(_receipts(world), key=lambda r: r["receipt_id"] != first["receipt_id"])
    assert earlier["outcome"] == {"status": "refused"} and successor["continues"] == first["receipt_id"]
    assert successor["outcome"] == {"status": "queued", "reason": "auto_merge_pending"}
    assert successor["effect"]["failure"] == "server_refused" and successor["publication"]["body"]["status"] == "gap"
    assert f"{ONLY_TO_MERGE} {QUEUE_LOOKUP}" in out and f"The PR-body receipt block was not confirmed; {RETRY}" in out
    assert "publication only" not in out and "calls only read" not in out and _merges(world) == 2
    monkeypatch.setattr(merge_receipts, "write_receipt", write)
    gh.pr["autoMergeRequest"], gh.fail_edit, gh.merge = None, False, "ok"
    assert "A new pr_merge call for the PR's current head may send one." in _merge(world) and _merges(world) == 2
    assert not any(merge_receipts._must_observe(r) for r in _receipts(world))  # both receipts are released
    assert "merge: merged" in _merge(world) and _merges(world) == 3  # a plain call, which no answer called read-only


@pytest.mark.parametrize("held", [True, False], ids=["earlier-request-unresolved", "released"])
def test_a_refused_requests_owed_projection_is_repaired_while_its_receipt_is_read_back(world, monkeypatch, held):
    """GitHub refused a request beside another actor's queue, so the PR body and card show it queued. Once the queue is
    gone a plain call settles it as refused, but the refreshed block cannot be written. Where the request continued an
    earlier one that is still unresolved, the chain stays observed: the answer promises no send, the next plain call
    sends nothing and repairs the block. Where nothing holds the PR any more, the answer says a new call may send."""
    gh = world.gh
    if held:
        gh.merge = "timeout_open"
        _merge(world)
    earlier = [r["receipt_id"] for r in _receipts(world)]
    gh.merge = "refused"

    def another_actors_queue(args, *a, **kw):
        result = gh(args, *a, **kw)
        if sent(args):
            gh.pr["autoMergeRequest"] = {"enabledAt": "by another actor"}
        return result

    monkeypatch.setattr(github, "_gh_run", another_actors_queue)
    _merge(world, **({"continue_receipt_id": earlier[0]} if held else {}))
    monkeypatch.setattr(github, "_gh_run", gh)
    refused = next(r for r in _receipts(world) if r["receipt_id"] not in earlier)
    assert refused["outcome"]["status"] == "queued" and "Outcome: **queued**" in gh.pr["body"]
    sends = _merges(world)
    gh.pr["autoMergeRequest"], gh.fail_edit = None, True
    out = _merge(world)
    settled = _by_id(world)[refused["receipt_id"]]
    assert settled["outcome"] == {"status": "refused"} and settled["publication"]["body"]["status"] == "gap"
    assert "Outcome: **queued**" in gh.pr["body"] and _merges(world) == sends
    if not held:
        assert ("⚠️ The PR-body receipt block was not confirmed; it may still show the earlier outcome, and a new "
                "pr_merge call may send a new merge request.") in out
        assert "would send" not in out and RETRY not in out
        return
    assert f"⚠️ The PR-body receipt block was not confirmed; {RETRY}" in out
    assert "would send" not in out and "may send a new merge request." not in out and ONLY_TO_MERGE in out
    gh.fail_edit = False
    again = _merge(world)
    repaired = _by_id(world)[refused["receipt_id"]]
    assert repaired["publication"]["body"]["status"] == "published" and repaired["revision"] == settled["revision"]
    assert "Outcome: **refused** (by GitHub)" in gh.pr["body"] and "Outcome: **queued**" not in gh.pr["body"]
    assert "not confirmed" not in again and "sent no new merge request" in again and _merges(world) == sends


def test_a_delayed_edit_of_an_earlier_attempt_never_stays_over_its_successors_block(world, monkeypatch):
    """A plain call refreshes a refused request's block. While its edit is in flight a decided successor merges and
    publishes its own block; the delayed edit then lands over it. The earlier attempt no longer owns the PR body, so
    the call that wrote over the successor restores the successor's block before it returns."""
    gh, raced = world.gh, {}
    gh.merge = "refused"

    def another_actors_queue(args, *a, **kw):
        result = gh(args, *a, **kw)
        if sent(args):
            gh.pr["autoMergeRequest"] = {"enabledAt": "by another actor"}
        return result

    monkeypatch.setattr(github, "_gh_run", another_actors_queue)
    _merge(world)
    (refused,) = _receipts(world)
    assert refused["outcome"]["status"] == "queued" and "Outcome: **queued**" in gh.pr["body"]
    gh.pr["autoMergeRequest"], gh.merge = None, "ok"

    def successor_publishes_first(args, *a, **kw):
        if args[:1] == ["api"] and "PATCH" in args and not raced:
            raced["started"] = True  # the refreshed block is on its way: the body it was made from is already stale
            monkeypatch.setattr(github, "_gh_run", gh)
            raced["out"] = _merge(world, continue_receipt_id=refused["receipt_id"])
            raced["body"] = gh.pr["body"]
            monkeypatch.setattr(github, "_gh_run", successor_publishes_first)
        return gh(args, *a, **kw)

    monkeypatch.setattr(github, "_gh_run", successor_publishes_first)
    _merge(world)
    earlier, successor = sorted(_receipts(world), key=lambda r: r["receipt_id"] != refused["receipt_id"])
    assert "merge: merged" in raced["out"] and successor["continues"] == earlier["receipt_id"]
    assert merge_receipts.public_block(successor) in raced["body"] and earlier["outcome"] == {"status": "refused"}
    assert merge_receipts.public_block(successor) in gh.pr["body"] and "Outcome: **refused**" not in gh.pr["body"]
    assert successor["publication"]["body"]["status"] == "published" and _merges(world) == 2


@pytest.mark.parametrize("comments_ok", [True, False], ids=["comments", "comments-unreadable"])
@pytest.mark.parametrize("held", ["settles-meanwhile", "other-repository"])
def test_comment_evidence_is_read_once_this_call_decides_a_new_request(world, monkeypatch, held, comments_ok):
    """An attempt in view when the call starts (one that settles meanwhile, or another repository's
    PR of the same number) defers the comment walk until this call decides a new request; an unreadable walk then
    refuses with gh's words before any request."""
    gh, walks = world.gh, []
    in_flight = {"schema": 1, "receipt_id": "in-flight", "state": "effect_attempted", "number": 7,
                 "repo": {"url": URL if held == "settles-meanwhile" else "https://github.com/other/demo/pull/7"},
                 "launch_operation_ids": [], "requested": {"method": "squash", "expected_head_sha": HEAD},
                 "observed_before": {}, "review": {}, "coverage": {},
                 "effect": {"ok": False, "exit_code": 1, "http_status": 405, "failure": "server_refused", "output": ""}}
    merge_receipts.write_receipt(world.root, "merge-task", in_flight)

    def transport(args, *a, **kw):
        if args[:2] == ["api", "graphql"] and held == "settles-meanwhile":  # GitHub's refusal is read back meanwhile
            merge_receipts.write_receipt(world.root, "merge-task", {**in_flight, "state": "settled",
                                                                    "outcome": {"status": "refused"}})
        fields = args[args.index("--json") + 1].split(",") if args[:2] == ["pr", "view"] else []
        if "comments" in fields:
            walks.append(fields)
            if not comments_ok:
                return github.GhResult(False, "⚠️ GH_ERROR: gh: comments page unavailable (HTTP 502)", 1, 502, "exit")
        return gh(args, *a, **kw)

    monkeypatch.setattr(github, "_gh_run", transport)
    out = _merge(world)
    assert walks == [["url", "comments"]]  # not with the first view: an attempt of PR #7 was in view then
    if comments_ok:
        assert "merge: merged" in out and _merges(world) == 1
        assert next(r for r in _receipts(world) if r["receipt_id"] != "in-flight")["review"]["contributor_evidence"] == "absent"
    else:
        assert f"pr_unreadable — the pull request {URL} could not be read" in out and "comments page unavailable" in out
        assert "nothing was merged" in out and _merges(world) == 0
        assert [r["receipt_id"] for r in _receipts(world)] == ["in-flight"]


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
    # The tool names the read that cannot merge and promises nothing of a later plain call.
    assert ONLY_TO_MERGE in out and LOOKUP in out and "calls only read" not in out
    (settled,) = _receipts(world)
    assert settled["state"] == "settled" and "effect" not in settled  # a readback that followed no effect records none


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
    assert after["publication"]["body"] == {"status": "published", "seq": 1}
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


def test_a_card_the_outbox_gave_up_is_said_and_sent_again_until_it_is_delivered(world):
    """While the outbox owes the card a repeat sends no second event. Once newer owed messages took every slot the
    outbox retired it undelivered: the receipt no longer calls it owed, the tool says so, and each call that reads the
    receipt back sends the same card again until a delivery is recorded."""
    from supervisor import terminal_delivery as td

    _merge(world)
    (first,) = _receipts(world)
    card = first["publication"]["card"]
    (event,) = world.ctx.pending_events
    assert card == {"status": "owed", "delivery_id": event["delivery_id"]}
    assert "not confirmed" not in _merge(world) and _receipts(world) == [first] and world.ctx.pending_events == [event]
    for index in range(td._PENDING_CAP):
        assert td.register_pending_delivery(world.root, {"type": "send_message", "delivery_id": f"newer:{index}",
                                                         "task_id": "merge-task", "chat_id": 7, "text": "newer"})
    registry = td._read_registry(world.root)
    assert card["delivery_id"] in registry["exhausted"] and card["delivery_id"] not in registry["pending"]
    for _ in range(2):  # every call that reads the receipt back retries it
        out = _merge(world)
        (retired,) = _receipts(world)
        assert retired["publication"]["card"] == {**card, "status": "gap", "reason": "outbox_exhausted"}
        assert f"⚠️ The task-card receipt is not confirmed: outbox_exhausted; {RETRY}" in out
        assert retired["revision"] == first["revision"]
        assert retired["publication"]["body"] == first["publication"]["body"]
    again = world.ctx.pending_events[1:]
    assert len(again) == 2 and again == [{**event, "ts": resent["ts"]} for resent in again]
    td.register_delivery(world.root, card["delivery_id"])  # the supervisor's post-send seam: one reached the chat
    assert "not confirmed" not in _merge(world) and len(world.ctx.pending_events) == 3
    assert _receipts(world)[0]["publication"]["card"] == {**card, "status": "delivered"} and _merges(world) == 1


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


@pytest.mark.parametrize("edit", ["removed", "older"])
def test_a_full_repeat_restores_the_block_a_later_edit_dropped_without_a_new_card(world, edit):
    """A later edit of the description dropped the confirmed block or left an older one: a full plain repeat restores
    this receipt's block beside the edited text and records it as a later confirmation of the same revision; it sends
    no second merge and registers no new card."""
    _merge(world)
    (first,) = _receipts(world)
    assert first["publication"]["body"] == {"status": "published"}
    events, patches = list(world.ctx.pending_events), sum(c[:1] == ["api"] and "PATCH" in c for c in world.gh.calls)
    world.gh.pr["body"] = "Rewritten by the owner." + (
        "" if edit == "removed" else "\n\n<!-- ouroboros:merge-receipt ab -->\nolder\n<!-- /ouroboros:merge-receipt -->\n\nFooter.")
    assert "sent no new merge request" in _merge(world)
    body = world.gh.pr["body"]
    assert merge_receipts.public_block(first) in body and "older" not in body
    assert body.startswith("Rewritten by the owner.\n\n<!-- ouroboros:merge-receipt ") and (
        edit == "removed" or body.endswith("<!-- /ouroboros:merge-receipt -->\n\nFooter."))
    assert sum(c[:1] == ["api"] and "PATCH" in c for c in world.gh.calls) == patches + 1
    (restored,) = _receipts(world)
    assert restored["publication"] == {**first["publication"], "body": {"status": "published", "seq": 1}}
    assert {**restored, "publication": first["publication"], "updated_at": first["updated_at"]} == first
    assert world.ctx.pending_events == events and _merges(world) == 1


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
    assert ONLY_TO_MERGE in refused and "calls only read" not in refused  # no promise that A keeps holding
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
    assert sum("This decision already" in out for out in outputs) == 1  # its words follow the effect recorded by then


@pytest.mark.parametrize("snapshot_ok", [True, False], ids=["complete", "unavailable"])
def test_duplicate_decision_rejoins_successor_settled_during_snapshot(world, monkeypatch, snapshot_ok):
    world.gh.merge = "timeout_open"
    _merge(world)
    (first,) = _receipts(world)
    args = {"continue_receipt_id": first["receipt_id"]}
    world.gh.merge = "ok"
    gh, winner = world.gh, []

    def snapshot_race(argv, *a, **kw):
        if argv[:2] == ["api", "graphql"] and not winner:
            winner.append(None)
            winner[0] = _merge(world, **args)
            if not snapshot_ok:
                return github.GhResult(False, "snapshot unavailable", 1, 502, "exit")
        return gh(argv, *a, **kw)

    monkeypatch.setattr(github, "_gh_run", snapshot_race)
    replay = _merge(world, **args)
    assert "merge: merged" in winner[0] and "merge: merged" in replay
    assert "This decision already sent its request" in replay and "sent no new merge request" in replay
    assert _merges(world) == len(_receipts(world)) == 2


def test_a_repeated_decision_before_its_request_is_sent_claims_no_send(world, monkeypatch):
    """A repeat of an explicit decision rejoins its successor while that holds only its intent: the decided call has
    not sent its request yet. The repeat says the decision has this receipt, whose request is not recorded as sent,
    and names no new decision of it; once the request's effect is recorded, a repeat says it was sent."""
    world.gh.merge = "timeout_open"
    _merge(world)
    (a,) = _receipts(world)
    world.gh.merge, gh, repeat = "ok", world.gh, {}

    def repeated_before_the_send(args, *a_, **kw):
        if sent(args) and not repeat:
            monkeypatch.setattr(github, "_gh_run", gh)
            repeat["text"] = _merge(world, continue_receipt_id=a["receipt_id"])
            repeat["row"] = next(r for r in _receipts(world) if r.get("continues") == a["receipt_id"])
            repeat["sends"] = _merges(world)
            monkeypatch.setattr(github, "_gh_run", repeated_before_the_send)
        return gh(args, *a_, **kw)

    monkeypatch.setattr(github, "_gh_run", repeated_before_the_send)
    assert "merge: merged" in _merge(world, continue_receipt_id=a["receipt_id"])
    said, successor = repeat["text"], repeat["row"]
    assert "effect" not in successor and repeat["sends"] == 1 and _merges(world) == 2
    assert f"receipt_id={successor['receipt_id']}" in said
    assert ("This decision already has this receipt, whose merge request is not recorded as sent, so this call only "
            "read GitHub back and sent no new merge request." in said)
    assert "already sent" not in said and "continue_receipt_id=" not in said and "reads the PR without merging" in said
    again = _merge(world, continue_receipt_id=a["receipt_id"])
    assert "This decision already sent its request as this receipt" in again and _merges(world) == 2


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
    answer({k: state["pr"][k] for k in args[args.index("--json")+1].split(",") if k in state["pr"]})
if args[:2] == ["api", "graphql"]:
    envelope = json.loads(sys.stdin.read())
    assert envelope["variables"] == {"owner": "octo", "name": "demo", "number": 7}
    snapshot = {k: v for k, v in state["pr"].items() if k != "comments"}
    answer({"data": {"repository": {"pullRequest": snapshot}}})
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


@pytest.mark.parametrize("status", ["changes_after_review", "unknown"])
def test_the_owners_card_names_a_failed_review_of_earlier_code_and_a_pass_stays_quiet(status):
    """BIBLE P3: continuing over a failed panel leaves a loud, owner-visible trace. When the merged code is not what the
    failed review read (edits after it, or coverage unknown), the owner's card still says that review returned FAIL
    (C11, 25.11: #4276 merged after a FAIL panel and its card said only "earlier version reviewed"): the card head
    carries "· FAIL"; a PASS adds nothing, and a FAIL on exactly what merged keeps its own label without a repeat."""
    def receipt(verdict, coverage):
        return {"number": 7, "outcome": {"status": "merged", "merge_sha": "a" * 40},
                "coverage": coverage, "review": {"record": {"state": "settled", "verdict": {"aggregate": verdict}}}}
    coverage = ({"status": "changes_after_review"} if status == "changes_after_review"
                else {"status": "unknown", "gaps": ["reviewed_head_not_recorded"]})
    label, _sentence = merge_receipts.coverage_reading(receipt("FAIL", coverage))
    assert label.endswith(" · FAIL") and label != merge_receipts._REJECTED
    assert merge_receipts.card_row_text(receipt("FAIL", coverage)).partition("\n")[0].endswith(" · FAIL")
    assert "FAIL" not in merge_receipts.coverage_reading(receipt("PASS", coverage))[0]
    label, _sentence = merge_receipts.coverage_reading(receipt("FAIL", {"status": "covers_head"}))
    assert label == merge_receipts._REJECTED
