"""#1556/#2455/#2762: GitHub's documented refusals of the sent merge are proven no-effects; nothing looser is.

Classification controls run the real ``_gh_run`` and ``_merge_answer`` with only the ``gh`` subprocess
replaced, answering the one REST merge request the way ``gh api`` does. Consumer controls enter at
``ToolRegistry.execute_result("pr_merge")`` and read the receipt, tool custody, Continue and cold sleep. By the
owner's decision a proven refusal ends this attempt's custody even beside an unreadable, closed or foreign-queued
PR, whose readback alone keeps the receipt read-back-only; a later explicit call for an open PR without a queue
sends exactly one new merge. An unknown request keeps its hold and a plain call only reads it back, never resends;
only the owner-chosen explicit continuation (9 Nov) decides one new request, and once any receipt of the task shows
the PR merged no request of it is held. A Resume naming a settled unknown request (9 Nov) goes on without it: the
decision sits beside its unchanged receipt, nothing is sent again, and a later queue or call holds it again.
"""
from __future__ import annotations

import json
import subprocess
from types import SimpleNamespace

import pytest

from ouroboros.task_results import load_task_result
from ouroboros.tools import github
from ouroboros.tools.registry import ToolContext
from tests.test_batch4_producer_custody import _consumers
from tests.test_pr_merge_receipts import HEAD, MERGE, ROUTE, URL, FakeGh, sent


REPAIRED = "e" * 40
SEND = ["api", ROUTE, "--hostname", "github.com", "--method", "PUT", "--input", "-"]
ENDED = "request refused by GitHub"


def answer(status, message):
    """``gh api``'s exit on a refused request: the JSON body on stdout, ``gh: <message> (HTTP NNN)`` on stderr."""
    return (1, json.dumps({"message": message, "documentation_url": "https://docs.github.com/rest"}),
            f"gh: {message} (HTTP {status})\n")


NOT_MERGEABLE = answer(405, "Pull Request is not mergeable")
HEAD_MOVED = answer(409, "Head branch was modified. Review and try the merge again.")
INVALID = answer(422, "Validation Failed")
FORBIDDEN = answer(403, "Resource not accessible by integration")
BAD_GATEWAY = answer(502, "Server Error")
MERGED = (0, json.dumps({"sha": MERGE, "merged": True, "message": "Pull Request successfully merged"}), "")


def _classify(tmp_path, monkeypatch, rc, stdout, stderr):
    ctx = ToolContext(repo_dir=tmp_path, drive_root=tmp_path / "data", task_id="task-fixture")
    seen = []

    def run(argv, **kw):
        seen.append((argv, kw.get("input")))
        return SimpleNamespace(returncode=rc, stdout=stdout, stderr=stderr)

    monkeypatch.setattr(github.subprocess, "run", run)
    monkeypatch.setattr(github, "_gh_env", lambda _ctx: {})
    monkeypatch.setattr(github, "github_token_from_env_or_settings", lambda: "")
    result = github._merge_answer(github._gh_run(SEND, ctx, input_data="{}"))
    assert [argv for argv, _ in seen] == [["gh", *SEND]] and seen[0][1] == "{}"
    return result


@pytest.mark.parametrize("answered", [NOT_MERGEABLE, HEAD_MOVED, INVALID], ids=["405", "409", "422"])
def test_githubs_documented_merge_refusals_are_server_refused(tmp_path, monkeypatch, answered):
    result = _classify(tmp_path, monkeypatch, *answered)

    assert (result.ok, result.exit_code, result.failure) == (False, 1, "server_refused")
    assert result.text == "⚠️ GH_ERROR: " + answered[2].strip()


@pytest.mark.parametrize("answered", [
    answer(401, "Bad credentials"), FORBIDDEN, answer(404, "Not Found"), answer(429, "API rate limit exceeded"),
    BAD_GATEWAY, answer(503, "Service Unavailable"),
    (1, "", "EOF\n"),
    # A status quoted in prose is not gh's own marker, and a refusal status beside another answer proves nothing.
    (1, "", "gh: the merge said HTTP 405 earlier\n"),
    (2, "", "unknown flag: --input\n"),
], ids=["401", "403", "404", "429", "502", "503", "no-status", "prose-status", "usage"])
def test_anything_but_a_documented_refusal_stays_an_unknown_exit(tmp_path, monkeypatch, answered):
    result = _classify(tmp_path, monkeypatch, *answered)

    assert result.ok is False and result.failure == "exit", result


@pytest.mark.parametrize("stdout,merged", [
    (MERGED[1], True),
    (json.dumps({"sha": MERGE.upper(), "merged": True}), True),
    (json.dumps({"sha": MERGE, "merged": False, "message": "Pull Request is not mergeable"}), False),
    (json.dumps({"sha": MERGE, "merged": "true"}), False),
    (json.dumps({"sha": MERGE[:12], "merged": True}), False),
    (json.dumps({"merged": True}), False),
    (json.dumps([MERGE]), False),
    ("<html>Bad gateway</html>", False),
    ("", False),
], ids=["merged", "upper-sha", "merged-false", "merged-string", "short-sha", "no-sha", "list", "html", "empty"])
def test_a_2xx_is_the_requests_merge_only_with_merged_true_and_a_full_sha(tmp_path, monkeypatch, stdout, merged):
    result = _classify(tmp_path, monkeypatch, 0, stdout, "")

    if merged:
        assert (result.ok, result.text, result.failure) == (True, MERGE, "")
    else:
        assert (result.ok, result.failure) == (False, "malformed_response") and "unrecognized merge answer" in result.text


def test_a_merge_timeout_is_never_a_refusal(tmp_path, monkeypatch):
    def run(argv, **_kw):
        raise subprocess.TimeoutExpired(argv, 120)

    monkeypatch.setattr(github.subprocess, "run", run)
    monkeypatch.setattr(github, "_gh_env", lambda _ctx: {})
    ctx = ToolContext(repo_dir=tmp_path, drive_root=tmp_path, task_id="task-fixture")
    assert github._merge_answer(github._gh_run(SEND, ctx, input_data="{}")).failure == "timeout"


# --- the consumer chain -----------------------------------------------------------------


@pytest.fixture
def chain(tmp_path, monkeypatch):
    from tests.test_batch4_producer_custody import _registry

    registry, queue, workers = _registry(tmp_path, monkeypatch)
    fake = FakeGh(tmp_path)
    fake.pr.update(mergeStateStatus="UNKNOWN")  # GitHub has not computed mergeability: the request decides
    state = SimpleNamespace(registry=registry, queue=queue, workers=workers, fake=fake, merges=[], payloads=[], answers=[],
                            root=tmp_path, observe=None, observed=[])
    real_run = subprocess.run

    def run(argv, **kw):
        if argv[:1] != ["gh"]:
            return real_run(argv, **kw)
        args = argv[1:]
        # Repository-bound reads run in the Project; API legs name their repository and host in the route.
        assert "--repo" not in args and (args[:1] == ["api"] or kw["cwd"] == str(registry._ctx.workspace_root))
        if sent(args):
            assert args == SEND  # the PR its readback named, on that host
            state.merges.append(args)
            state.payloads.append(json.loads(kw.get("input")))
            fake.calls.append(list(args))
            answered = state.answers.pop(0)
            if answered == "timeout":
                raise subprocess.TimeoutExpired(argv, 120)
            if answered == "merged":
                fake.pr.update(state="MERGED", mergeCommit={"oid": MERGE}, mergedAt="2026-10-08T12:00:00Z")
                answered = MERGED
            elif answered[0] == "merged_elsewhere":  # refused; another actor merged meanwhile
                fake.pr.update(state="MERGED", mergeCommit={"oid": MERGE}, mergedAt="2026-10-08T12:00:00Z")
                answered = answered[1:]
            elif answered[0] == "queued_elsewhere":  # refused; another actor enabled auto-merge meanwhile
                fake.pr.update(autoMergeRequest={"enabledAt": "by another actor"})
                answered = answered[1:]
            rc, stdout, stderr = answered
            return SimpleNamespace(returncode=rc, stdout=stdout, stderr=stderr)
        if args[:2] == ["pr", "view"] and state.merges and state.observe:
            state.observed.append(state.observe())  # a parallel reader while the handler still reads back
        result = fake(args, registry._ctx, input_data=kw.get("input"))
        return SimpleNamespace(returncode=0 if result.ok else 1, stdout=result.text if result.ok else "",
                               stderr="" if result.ok else result.text)

    monkeypatch.setattr(github.subprocess, "run", run)
    monkeypatch.setattr(github, "_gh_env", lambda _ctx: {})
    monkeypatch.setattr(github, "github_token_from_env_or_settings", lambda: "")
    monkeypatch.setattr(github, "github_cli_configured", lambda: True)
    return state


def _merge(chain, head, method="squash", **kw):
    return chain.registry.execute_result("pr_merge", {"number": 7, "expected_head_sha": head, "method": method, **kw})


def _receipts(chain):
    return (load_task_result(chain.root, "root") or {}).get("merge_receipts") or []


def _held(chain):
    from ouroboros.model_sleep import cold_blockers
    from ouroboros.tool_custody import retained_tool_custody
    from supervisor.continuation_admission import conflicting_writers

    row = load_task_result(chain.root, "root")
    assert not row.get("launch_handoffs"), "the local invocation itself returned"
    merge_hold = any(item["kind"] == "merge_operation" for item in retained_tool_custody(chain.root, "root", row))
    assert bool(cold_blockers(chain.registry._ctx)) is merge_hold
    # Continue's census also names the still-running fixture member; only the merge hold varies.
    assert any(item["kind"] == "merge_operation" for item in conflicting_writers(chain.queue, "root")) is merge_hold
    return merge_hold


def _repair(chain):
    chain.fake.pr.update(headRefOid=REPAIRED, mergeStateStatus="CLEAN")


REFUSALS = pytest.mark.parametrize("answered", [NOT_MERGEABLE, HEAD_MOVED, INVALID], ids=["405", "409", "422"])


@REFUSALS
def test_another_actors_queue_then_close_stays_observed_while_our_refused_request_holds_nothing(chain, answered):
    """Owner decision (#2455): the proven refusal ends this attempt's custody; another actor's auto-merge, enabled while
    the request was answered, and the PR's later close stay the observed outcome, worded as theirs, never resent."""
    from ouroboros.merge_receipts import card_row_text, public_block

    chain.answers.append(("queued_elsewhere", *answered))
    first = _merge(chain, HEAD)
    receipt = _receipts(chain)[0]
    assert (receipt["outcome"]["status"], receipt["effect"]["failure"]) == ("queued", "server_refused")
    assert first.text.startswith(f"⚠️ PR_MERGE_REFUSED: ⚠️ PR #7 merge: queued (PR now OPEN, auto-merge on; {ENDED});")
    assert first.meta["operation_outcome"] == "completed_no_effect" and _held(chain) is False
    assert f"- Outcome: **queued** (PR now OPEN, auto-merge on; {ENDED})" in public_block(receipt)
    chain.fake.pr.update(state="CLOSED", autoMergeRequest=None)
    output = _merge(chain, HEAD)
    (current,) = _receipts(chain)
    for text in (output.text, card_row_text(current), public_block(current)):
        assert "PR now CLOSED" in text and ENDED in text
        assert "effect unconfirmed" not in text
    assert current["effect"] == receipt["effect"] and current["outcome"] == receipt["outcome"]
    assert output.meta["operation_outcome"] == "completed_no_effect"
    assert "changed nothing, so it holds no task custody" in output.text
    assert "Until GitHub shows the PR open without a queue, repeat calls only read it back." in output.text
    assert _held(chain) is False
    _merge(chain, HEAD)
    assert len(chain.merges) == 1


@pytest.mark.parametrize("answered,status,repair", [(NOT_MERGEABLE, 405, True), (INVALID, 422, False)],
                         ids=["405-then-repaired-head", "422-then-same-head"])
def test_githubs_refusal_ends_the_attempt_and_an_explicit_call_merges_once(chain, answered, status, repair):
    """#2455: GitHub refused the sent merge. Nothing is held or resent; the caller's next explicit call for an open PR
    without a queue sends exactly one new merge."""
    chain.answers += [answered, "merged"]
    first = _merge(chain, HEAD)
    assert first.text.startswith("⚠️ PR_MERGE_REFUSED: ⚠️ PR #7 merge: refused (by GitHub);"), first.text
    assert (f"Merge request: failure=server_refused, exit_code=1, http_status={status}: "
            f"⚠️ GH_ERROR: {answered[2].strip()}") in first.text
    assert first.meta["operation_outcome"] == "completed_no_effect"
    (receipt,) = _receipts(chain)
    assert (receipt["outcome"]["status"], receipt["effect"]["failure"]) == ("refused", "server_refused")
    # A first refusal nothing showed yet publishes nothing: no body edit, no card delivery.
    assert "publication" not in receipt and "merge-receipt" not in chain.fake.pr["body"]
    assert not any("PATCH" in call for call in chain.fake.calls)
    assert _held(chain) is False
    head = REPAIRED if repair else HEAD
    if repair:
        _repair(chain)
        assert "head_moved" in _merge(chain, HEAD).text and len(chain.merges) == 1
    else:
        chain.fake.pr.update(mergeStateStatus="CLEAN")  # the base settled; the same head is merged on purpose
    second = _merge(chain, head)
    assert second.meta["operation_outcome"] == "completed", second.text
    assert chain.merges == [SEND, SEND] and [payload["sha"] for payload in chain.payloads] == [HEAD, head]
    assert [(r["outcome"]["status"], r["requested"]["expected_head_sha"]) for r in _receipts(chain)] == [
        ("refused", HEAD), ("merged", head)]
    assert _receipts(chain)[1]["outcome"]["attribution"] == "this_call" and _held(chain) is False


def test_a_github_refusal_with_a_failed_readback_settles_once_github_answers(chain):
    chain.fake.fail_view_after_merge = True
    chain.answers += [NOT_MERGEABLE, "merged"]
    first = _merge(chain, HEAD)
    assert first.text.startswith("⚠️ PR_MERGE_REFUSED: ⚠️ PR #7 merge: unknown (PR unread; request refused by GitHub);")
    assert "GitHub readback: readback_failed." in first.text and first.meta["operation_outcome"] == "completed_no_effect"
    assert _held(chain) is False
    unread = _merge(chain, HEAD)  # still unreadable: refused before any new send, naming the known end
    assert "was refused by GitHub and changed nothing" in unread.text and "still unresolved" not in unread.text
    chain.fake.fail_view_after_merge = False
    _repair(chain)
    observed = _merge(chain, REPAIRED)
    assert "A new pr_merge call for the PR's current head may send one." in observed.text and len(chain.merges) == 1
    assert _receipts(chain)[0]["outcome"]["status"] == "refused"
    assert _merge(chain, REPAIRED).meta["operation_outcome"] == "completed" and len(chain.merges) == 2


def test_a_handler_still_reading_back_keeps_its_own_claim_while_the_refused_merge_holds_nothing(chain):
    """The merge effect's custody ends with GitHub's refusal, but the running handler stays held until it returns:
    a Continue or cold-sleep reader looking in mid-readback never takes over a live invocation."""
    from ouroboros.model_sleep import cold_blockers
    from ouroboros.tool_custody import retained_tool_custody
    from supervisor.continuation_admission import conflicting_writers

    def look():
        row = load_task_result(chain.root, "root")
        return ({item["kind"] for item in retained_tool_custody(chain.root, "root", row)},
                {item["kind"] for item in conflicting_writers(chain.queue, "root")} & {"tool_handoff", "merge_operation"},
                bool(cold_blockers(chain.registry._ctx)), [r["effect"]["failure"] for r in row["merge_receipts"]])

    chain.observe = look
    chain.answers.append(NOT_MERGEABLE)
    assert _merge(chain, HEAD).meta["operation_outcome"] == "completed_no_effect"
    assert chain.observed == [({"tool_handoff"}, {"tool_handoff"}, True, ["server_refused"])]
    assert _held(chain) is False
    _consumers(chain.root, chain.registry, chain.queue, chain.workers, False)


def test_a_pr_already_set_to_auto_merge_waits_and_this_tool_sends_nothing(chain):
    """A queue GitHub already holds is never bypassed: an ordinary first call, or one after a released attempt,
    refuses before any send, cancels nothing and holds nothing."""
    chain.fake.pr.update(autoMergeRequest={"enabledAt": "by another actor"}, mergeStateStatus="CLEAN")
    waiting = _merge(chain, HEAD)
    assert waiting.text.startswith("⚠️ PR_MERGE_REFUSED: auto_merge_pending — GitHub shows auto-merge enabled")
    assert waiting.meta["operation_outcome"] == "completed_no_effect"
    assert not chain.merges and not _receipts(chain) and _held(chain) is False
    chain.fake.pr.update(autoMergeRequest=None)
    chain.answers.append("merged")
    assert _merge(chain, HEAD).meta["operation_outcome"] == "completed" and len(chain.merges) == 1


@pytest.mark.parametrize("merge_state", ["BLOCKED", "BEHIND", "DIRTY"])
def test_a_blocked_behind_or_dirty_pr_gets_no_request_and_no_receipt(chain, merge_state):
    """Like gh without --admin: the endpoint would let an administrator's credential merge past required reviews or
    checks, an out-of-date head or a conflict, so nothing is sent; a state GitHub has not computed still is."""
    chain.fake.pr.update(mergeStateStatus=merge_state)
    refused = _merge(chain, HEAD)
    assert refused.text.startswith(f"⚠️ PR_MERGE_REFUSED: not_mergeable — GitHub reports the pull request {merge_state}")
    assert "only an administrator could make" in refused.text
    assert refused.meta["operation_outcome"] == "completed_no_effect"
    assert not chain.merges and not _receipts(chain) and _held(chain) is False
    chain.fake.pr.update(mergeStateStatus="UNKNOWN")
    chain.answers.append("merged")
    assert _merge(chain, HEAD).meta["operation_outcome"] == "completed" and len(chain.merges) == 1


def test_a_merge_by_someone_else_after_our_refusal_leaves_our_receipt_refused(chain):
    chain.answers.append(NOT_MERGEABLE)
    _merge(chain, HEAD)
    chain.fake.pr.update(state="MERGED", mergeCommit={"oid": MERGE}, mergedAt="2026-10-08T12:00:00Z")
    later = _merge(chain, HEAD)
    assert later.text.startswith("⚠️ PR_MERGE_REFUSED: pr_not_open") and len(chain.merges) == 1
    assert [r["outcome"]["status"] for r in _receipts(chain)] == ["refused"] and _held(chain) is False


def test_an_old_unknown_github_refusal_is_not_reclassified(chain, monkeypatch):
    """A receipt stored as an unknown exit keeps its hold: only a new send is classified."""
    monkeypatch.setattr(github, "_MERGE_REFUSED_STATUSES", frozenset())
    chain.answers.append(NOT_MERGEABLE)
    assert "PR_MERGE_UNKNOWN" in _merge(chain, HEAD).text
    monkeypatch.setattr(github, "_MERGE_REFUSED_STATUSES", frozenset({405, 409, 422}))
    _repair(chain)
    again = _merge(chain, REPAIRED)
    assert "had no confirmed outcome" in again.text and again.meta["operation_outcome"] == "unknown"
    (receipt,) = _receipts(chain)
    assert receipt["effect"]["failure"] == "exit" and "Pull Request is not mergeable" in receipt["effect"]["output"]
    assert receipt["outcome"]["status"] == "unknown" and len(chain.merges) == 1 and _held(chain) is True


@pytest.mark.parametrize("method", ["squash", "merge", "rebase"])
def test_proven_conflict_refusal_frees_custody_and_a_repaired_head_merges_once(chain, method):
    from ouroboros.update_qualification import commit_qualification

    owner_intro = "Summary of the change.\n\nOwner notes keep `literal code` and  spacing.\n\n"
    owner_footer = "\n\nOwner follow-up:\n  Preserve this indentation and punctuation: →."
    chain.fake.pr["body"] = (owner_intro + "<!-- ouroboros:merge-receipt ab -->\nprior receipt\n"
                             "<!-- /ouroboros:merge-receipt -->" + owner_footer)
    chain.answers += [NOT_MERGEABLE, "merged"]
    first = _merge(chain, HEAD, method)
    assert first.text.startswith("⚠️ PR_MERGE_REFUSED"), first.text
    assert first.meta["operation_outcome"] == "completed_no_effect"
    (receipt,) = _receipts(chain)
    assert receipt["outcome"]["status"] == "refused" and receipt["effect"]["failure"] == "server_refused"
    assert receipt["launch_operation_ids"]  # the real registry invocation identity reached the receipt
    assert _held(chain) is False

    _repair(chain)
    stale = _merge(chain, HEAD, method)  # the old head is refused before any send
    assert "head_moved" in stale.text and len(chain.merges) == 1
    second = _merge(chain, REPAIRED, method)
    assert second.meta["operation_outcome"] == "completed", second.text
    assert chain.merges == [SEND, SEND]
    for payload, head, retained in zip(chain.payloads, (HEAD, REPAIRED), _receipts(chain)):
        assert (payload["sha"], payload["merge_method"]) == (head, method)
        if method == "rebase":
            assert "commit_message" not in payload  # rebase has no merge commit to carry a receipt
            continue
        body = payload["commit_message"]
        prose, summary = body.split("\n\nOuroboros qualification receipt", 1)
        assert prose == owner_intro + owner_footer
        qualification = commit_qualification(body)
        assert qualification == retained["qualification"]
        assert {(row["name"], row["status"], row["source"]) for row in qualification["checks"]} == {
            ("full_battery", "unknown", "missing"), ("cost_contract", "unknown", "missing")}
        assert qualification["review"] == {"observed": "unknown", "state": "no_record", "relation": "not_declared",
                                           "declared": "unknown"}
        assert qualification["post_merge"]["status"] == "not_run"
        assert "declarations are not host-observed PASS" in summary
        assert "Post-merge check: not run yet." in summary
    assert chain.fake.pr["body"].startswith(owner_intro) and chain.fake.pr["body"].endswith(owner_footer)
    assert [(r["outcome"]["status"], r["requested"]["expected_head_sha"]) for r in _receipts(chain)] == [
        ("refused", HEAD), ("merged", REPAIRED)]
    assert _receipts(chain)[1]["outcome"]["attribution"] == "this_call"
    assert _held(chain) is False
    _consumers(chain.root, chain.registry, chain.queue, chain.workers, False)


@pytest.mark.parametrize("answered", [BAD_GATEWAY, "timeout", FORBIDDEN, (1, "", "EOF\n")],
                         ids=["502", "timeout", "403", "no-status"])
def test_unknown_refusal_keeps_its_hold_and_is_only_read_back(chain, answered):
    chain.answers.append(answered)
    first = _merge(chain, HEAD)
    assert first.meta["operation_outcome"] == "unknown", first.text
    assert _held(chain) is True

    _repair(chain)
    again = _merge(chain, REPAIRED)
    assert "sent no new merge request" in again.text and len(chain.merges) == 1
    (receipt,) = _receipts(chain)
    assert receipt["requested"]["expected_head_sha"] == HEAD and receipt["outcome"]["status"] == "unknown"
    assert _held(chain) is True

    chain.fake.pr.update(state="MERGED", mergeCommit={"oid": MERGE})
    settled = _merge(chain, REPAIRED)
    assert settled.meta["operation_outcome"] == "completed", settled.text
    (receipt,) = _receipts(chain)
    assert receipt["outcome"]["status"] == "merged" and receipt["outcome"]["attribution"] == "unproven"
    assert len(chain.merges) == 1 and _held(chain) is False


@REFUSALS
@pytest.mark.parametrize("case", ["auto_merge_pending", "readback_failed"])
def test_a_proven_refusal_holds_no_custody_while_its_readback_still_observes(chain, case, answered):
    """Owner decision (#2455), superseding the earlier hold: this attempt's custody ends with its proven refusal even
    while GitHub shows another actor's queue or cannot be read; that observation keeps the receipt read-back-only."""
    if case == "auto_merge_pending":
        chain.answers.append(("queued_elsewhere", *answered))
    else:
        chain.fake.fail_view_after_merge = True
        chain.answers.append(answered)
    first = _merge(chain, HEAD)
    assert first.text.startswith("⚠️ PR_MERGE_REFUSED:") and ENDED in first.text, first.text
    assert first.meta["operation_outcome"] == "completed_no_effect"
    (receipt,) = _receipts(chain)
    assert receipt["effect"]["failure"] == "server_refused"
    assert receipt["outcome"]["status"] == ("queued" if case == "auto_merge_pending" else "unknown")
    assert _held(chain) is False
    _repair(chain)
    again = _merge(chain, REPAIRED)
    assert "sent no new merge request" in again.text and len(chain.merges) == 1
    assert again.meta["operation_outcome"] == "completed_no_effect"
    _consumers(chain.root, chain.registry, chain.queue, chain.workers, False)


@pytest.mark.parametrize("answered", [NOT_MERGEABLE, HEAD_MOVED], ids=["405", "409"])
def test_a_refusal_beside_an_observed_merge_settles_without_claiming_it(chain, answered):
    from ouroboros.merge_receipts import card_row_text, public_block

    chain.answers.append(("merged_elsewhere", *answered))
    result = _merge(chain, HEAD)
    # The PR is merged, by another request: this operation changed nothing and claims no success.
    assert (result.status, result.meta["operation_outcome"]) == ("ok", "completed_no_effect"), result.text
    assert "GitHub shows the PR merged by another request." in result.text
    (receipt,) = _receipts(chain)
    assert receipt["effect"]["failure"] == "server_refused"
    assert receipt["outcome"]["status"] == "merged" and receipt["outcome"]["attribution"] == "unproven"
    for text in (result.text, card_row_text(receipt), public_block(receipt)):
        assert f"merged** (not by this request; {ENDED})" in text or f"merged (not by this request; {ENDED})" in text
    assert card_row_text(receipt).startswith("⚠️ ")
    assert _held(chain) is False and len(chain.merges) == 1


# --- #2762: an explicit decision continues an unconfirmed request -----------------------------------


@pytest.mark.parametrize("unknown", [BAD_GATEWAY, "timeout", FORBIDDEN], ids=["502", "timeout", "403"])
def test_a_decided_continuation_sends_once_and_the_merge_it_shows_releases_the_earlier_hold(chain, unknown):
    """Owner choice (9 Nov): after an unknown answer and a clean readback (open, same head and target, no queue) a plain
    call only reads back, and an explicit continuation sends one new request. The earlier receipt keeps its unknown
    effect and is never rewritten, yet once the PR shows merged neither request holds custody; repeating the
    decision rejoins its request without sending, and the card and PR body show the successor."""
    from ouroboros.merge_receipts import card_row_text, effect_is_definite, public_block

    chain.answers += [unknown, "merged"]
    first = _merge(chain, HEAD)
    (earlier,) = _receipts(chain)
    assert first.meta["operation_outcome"] == "unknown" and _held(chain) is True
    assert f"continue_receipt_id='{earlier['receipt_id']}'" in first.text
    plain = _merge(chain, HEAD)
    assert "sent no new merge request" in plain.text and len(chain.merges) == 1 and _held(chain) is True
    decided = _merge(chain, HEAD, continue_receipt_id=earlier["receipt_id"])
    assert decided.meta["operation_outcome"] == "completed", decided.text
    assert f"continues receipt {earlier['receipt_id']}" in decided.text and chain.merges == [SEND, SEND]
    stored = {r["receipt_id"]: r for r in _receipts(chain)}
    successor = next(r for r in stored.values() if r.get("continues") == earlier["receipt_id"])
    kept = stored[earlier["receipt_id"]]
    assert (kept["effect"], kept["outcome"]["status"]) == (earlier["effect"], "unknown")
    assert (successor["outcome"]["status"], successor["outcome"]["attribution"]) == ("merged", "unproven")
    assert successor["launch_operation_ids"] and successor["launch_operation_ids"] != kept["launch_operation_ids"]
    assert not effect_is_definite(kept) and _held(chain) is False
    rejoined = _merge(chain, HEAD, continue_receipt_id=earlier["receipt_id"])
    assert "This decision already sent its request" in rejoined.text and len(chain.merges) == 2
    assert public_block(successor) in chain.fake.pr["body"] and earlier["receipt_id"] not in chain.fake.pr["body"]
    assert "merge: merged" in card_row_text(successor)
    _consumers(chain.root, chain.registry, chain.queue, chain.workers, False)


def test_a_late_landing_of_the_earlier_request_refuses_the_continuation_and_still_releases_both(chain):
    """The earlier request lands while the decided one is in flight: GitHub refuses the new one (the PR is merged),
    which then shows the merge as not its own, and the merged PR ends the earlier request's hold."""
    chain.answers += [BAD_GATEWAY, ("merged_elsewhere", *NOT_MERGEABLE)]
    _merge(chain, HEAD)
    (earlier,) = _receipts(chain)
    decided = _merge(chain, HEAD, continue_receipt_id=earlier["receipt_id"])
    assert "GitHub shows the PR merged by another request." in decided.text
    assert decided.meta["operation_outcome"] == "completed_no_effect" and len(chain.merges) == 2
    successor = next(r for r in _receipts(chain) if r.get("continues"))
    assert (successor["effect"]["failure"], successor["outcome"]["status"]) == ("server_refused", "merged")
    assert next(r for r in _receipts(chain) if not r.get("continues"))["outcome"]["status"] == "unknown"
    assert _held(chain) is False


# --- 9 Nov (quiz 0ad8b0): a Resume goes on past a settled unknown request without its result -----------------

INCIDENT = (1, "", "Pull Request is still a draft\n")  # the held receipt's shape: an exit gh gave no status for


def _decide(chain, *ids, text="go on without it"):
    """The owner's Resume census of this tree, with its named decision bound as the door binds it."""
    from ouroboros.control_actor import accepting_unknown, owner_door
    from supervisor.continuation_admission import action_writers

    with owner_door("api_task_resume"), accepting_unknown(ids, text):
        return action_writers(chain.queue, "root")


def _custody(chain):
    from ouroboros.tool_custody import retained_tool_custody

    return retained_tool_custody(chain.root, "root", load_task_result(chain.root, "root"))


def _decisions(chain):
    events = chain.root / "logs" / "events.jsonl"
    lines = events.read_text(encoding="utf-8").splitlines() if events.exists() else []
    return [json.loads(line) for line in lines if '"local_custody_unknown_accepted"' in line]


@pytest.mark.parametrize("unknown", [INCIDENT, BAD_GATEWAY, "timeout"], ids=["generic-exit", "502", "timeout"])
def test_a_resume_naming_a_settled_unknown_request_goes_on_and_never_sends_it(chain, unknown):
    """Owner choice (9 Nov): the named decision sits beside the receipt, whose facts, revision and publication stay;
    nothing is sent again, a plain call still only reads it back, and Pause, cold sleep and Continue hold nothing."""
    from ouroboros.merge_receipts import card_row_text, public_block
    from ouroboros.work_stop import resume_outcome

    chain.answers.append(unknown)
    assert _merge(chain, HEAD).meta["operation_outcome"] == "unknown" and _held(chain) is True
    (before,) = _receipts(chain)
    rid = before["receipt_id"]
    blockers = _custody(chain)
    assert blockers == [{"kind": "merge_operation", "task_id": "root", "receipt_id": rid,
                         "reason": "merge_effect_unknown"}]
    refusal = resume_outcome({"ok": False, "error": "owner_pause_effects_unsettled", "blockers": blockers})
    assert refusal["acceptable_unknown"] == [rid]
    assert refusal["message"] == "Resume refused: the outcome of a merge request of task root is unknown"

    _decide(chain)  # a Resume naming nothing accepts nothing
    assert _held(chain) is True and not _decisions(chain)
    _decide(chain, rid, "receipt-elsewhere")
    assert _held(chain) is False
    [event] = _decisions(chain)
    assert event["accepted"] == [rid] and event["unmatched"] == ["receipt-elsewhere"] and not event["refused"]
    (after,) = _receipts(chain)
    decision = after["unknown_accepted"]
    assert {key: value for key, value in after.items() if key != "unknown_accepted"} == before
    assert (decision["actor"], decision["text"], decision["reason"], decision["replay_authorized"]) == (
        "owner", "go on without it", "merge_effect_unknown", False)
    assert "go on without it" not in public_block(after) + card_row_text(after), "the decision stays private"

    _repair(chain)
    again = _merge(chain, REPAIRED)
    assert "sent no new merge request" in again.text and len(chain.merges) == 1
    (kept,) = _receipts(chain)
    assert (kept["unknown_accepted"], kept["effect"], kept["outcome"]["status"]) == (decision, before["effect"], "unknown")
    assert _held(chain) is False
    _consumers(chain.root, chain.registry, chain.queue, chain.workers, False)


def test_a_queue_read_back_after_the_decision_holds_again_and_no_resume_goes_past_it(chain):
    """The decision covers the unknown it was made about: a later readback showing a queue holds the request again
    and naming it is refused by that reason; the PR's merge then frees it. The decision stays recorded."""
    from ouroboros.work_stop import acceptable_unknown

    chain.answers.append(BAD_GATEWAY)
    _merge(chain, HEAD)
    (receipt,) = _receipts(chain)
    _decide(chain, receipt["receipt_id"])
    decision = _receipts(chain)[0]["unknown_accepted"]
    assert _held(chain) is False
    chain.fake.pr.update(autoMergeRequest={"enabledAt": "by another actor"})
    _merge(chain, HEAD)  # a plain call reads it back and sends nothing
    (queued,) = _receipts(chain)
    assert (queued["outcome"]["status"], queued["unknown_accepted"]) == ("queued", decision)
    assert _held(chain) is True and len(chain.merges) == 1
    blockers = _custody(chain)
    assert [b["reason"] for b in blockers] == ["merge_queued"] and acceptable_unknown(blockers) == []
    _decide(chain, receipt["receipt_id"])
    assert _held(chain) is True, "a known queue is never waived"
    assert _decisions(chain)[-1]["refused"] == [{"id": receipt["receipt_id"], "task_id": "root",
                                                 "reason": "merge_queued"}]
    chain.fake.pr.update(state="MERGED", mergeCommit={"oid": MERGE}, autoMergeRequest=None)
    _merge(chain, HEAD)
    assert _receipts(chain)[0]["outcome"]["status"] == "merged" and _held(chain) is False and len(chain.merges) == 1


def test_a_crash_after_githubs_durable_refusal_clears_without_running_the_handler_again(tmp_path):
    """A durably refused merge holds nothing; the ended worker's own claim is retired by the process-end owner
    (positive evidence the process ended), never by re-executing the handler."""
    from ouroboros import merge_receipts
    from ouroboros.task_results import STATUS_RUNNING, write_task_result
    from ouroboros.tool_custody import retained_tool_custody, retire_process_invocations

    owner = {"pid": 999999, "process_birth": "fixture-birth", "task_attempt": 1}
    write_task_result(tmp_path, "root", STATUS_RUNNING, root_task_id="root", delegation_role="root", launch_handoffs={
        "op-1": {"tool": "pr_merge", "task_id": "root", "root_task_id": "root", "claimed_at": "t", "state": "claimed",
                 "local_owner": owner, "owner_proof": {}}})
    merge_receipts.write_receipt(tmp_path, "root", {
        "schema": 1, "receipt_id": "r1", "state": "effect_attempted", "repo": {"url": URL}, "number": 7,
        "launch_operation_ids": ["op-1"], "requested": {"method": "squash", "expected_head_sha": HEAD},
        "observed_before": {}, "review": {}, "coverage": {},
        "effect": {"ok": False, "failure": "server_refused", "exit_code": 1, "http_status": 405,
                   "output": "⚠️ GH_ERROR: " + NOT_MERGEABLE[2].strip()}})
    held = retained_tool_custody(tmp_path, "root", load_task_result(tmp_path, "root"))
    assert [item["kind"] for item in held] == ["tool_handoff"]  # only the worker, which may still run
    assert retire_process_invocations(tmp_path, pid=999999, process_birth="fixture-birth") == ["op-1"]
    assert retained_tool_custody(tmp_path, "root", load_task_result(tmp_path, "root")) == []


@pytest.mark.parametrize("other,released", [
    ({"repo": {"url": URL}, "outcome": {"status": "merged"}}, True),
    ({"repo": {"url": URL}, "outcome": {"status": "merged"}, "effect": {"ok": False, "failure": "server_refused"}}, True),
    ({"repo": {"url": URL}, "outcome": {"status": "queued", "reason": "auto_merge_pending"}}, False),
    ({"repo": {"url": URL}, "outcome": {"status": "refused"}, "effect": {"ok": False, "failure": "server_refused"}},
     False),
    ({"repo": {"url": URL}, "outcome": {"status": "unknown", "reason": "pr_state_closed"}}, False),
    ({"repo": {"url": URL.replace("/pull/7", "/pull/8")}, "outcome": {"status": "merged"}}, False),
    ({"repo": {"url": "https://github.com/octo/other/pull/7"}, "outcome": {"status": "merged"}}, False),
], ids=["merged", "merged-by-another-request", "queued", "refused", "closed", "other-pr", "other-repository"])
def test_only_a_merged_readback_of_the_same_pr_in_the_same_task_ends_an_unconfirmed_hold(tmp_path, other, released):
    """The custody projection reads facts it is given and rewrites nothing; a live call stays its own blocker."""
    from ouroboros import merge_receipts
    from ouroboros.task_results import STATUS_RUNNING, write_task_result
    from ouroboros.tool_custody import retained_tool_custody

    write_task_result(tmp_path, "root", STATUS_RUNNING, root_task_id="root", delegation_role="root", launch_handoffs={
        "op-live": {"tool": "pr_merge", "task_id": "root", "root_task_id": "root", "claimed_at": "t",
                    "state": "claimed", "local_owner": {}, "owner_proof": {}}})
    unknown = {"schema": 1, "receipt_id": "a", "state": "settled", "repo": {"url": URL}, "number": 7,
               "launch_operation_ids": ["op-a"], "requested": {"method": "squash", "expected_head_sha": HEAD},
               "effect": {"ok": False, "failure": "exit", "http_status": 502},
               "outcome": {"status": "unknown", "reason": "pr_state_open_after_unconfirmed_effect"}}
    merge_receipts.write_receipt(tmp_path, "root", unknown)
    merge_receipts.write_receipt(tmp_path, "root", {"schema": 1, "receipt_id": "b", "state": "settled", "number": 7,
                                                    "launch_operation_ids": ["op-b"], **other})
    row = load_task_result(tmp_path, "root")
    kinds = [(item["kind"], item.get("receipt_id")) for item in retained_tool_custody(tmp_path, "root", row)]
    assert ("merge_operation", "a") not in kinds if released else ("merge_operation", "a") in kinds
    assert ("tool_handoff", None) in kinds  # an independent live invocation is never released by a receipt
    stored = next(r for r in row["merge_receipts"] if r["receipt_id"] == "a")
    assert stored["outcome"] == unknown["outcome"] and not merge_receipts.effect_is_definite(stored)
    # Another task's merged receipt for the same PR is not this task's fact.
    write_task_result(tmp_path, "other", STATUS_RUNNING, root_task_id="other", delegation_role="root")
    merge_receipts.write_receipt(tmp_path, "other", {**unknown, "receipt_id": "c", "outcome": {"status": "merged"}})
    if not released:
        row = load_task_result(tmp_path, "root")
        assert ("merge_operation", "a") in [(i["kind"], i.get("receipt_id"))
                                            for i in retained_tool_custody(tmp_path, "root", row)]
    # An empty URL never matches another empty URL.
    write_task_result(tmp_path, "blank", STATUS_RUNNING, root_task_id="blank", delegation_role="root")
    merge_receipts.write_receipt(tmp_path, "blank", {**unknown, "receipt_id": "d", "repo": {"url": ""}})
    merge_receipts.write_receipt(tmp_path, "blank", {**unknown, "receipt_id": "e", "repo": {"url": ""},
                                                     "launch_operation_ids": [], "outcome": {"status": "merged"}})
    assert [i["receipt_id"] for i in retained_tool_custody(tmp_path, "blank", load_task_result(tmp_path, "blank"))
            if i["kind"] == "merge_operation"] == ["d"]


def test_only_a_settled_unknown_request_is_accepted_and_no_observer_writes_or_drops_the_decision(tmp_path):
    """Judged under the receipt lock: a call still claimed, an unsettled request, a queue or GitHub's merged answer
    is refused by its reason, an id of no receipt is unmatched and nothing is recorded twice; a stale observer neither
    drops nor mints a decision, and a new call of the accepted request holds it again while it runs."""
    from ouroboros import merge_receipts
    from ouroboros.local_custody_repair import ACCEPTABLE_UNKNOWN, accept_unknown_custody
    from ouroboros.task_results import STATUS_RUNNING, write_task_result
    from ouroboros.tool_custody import retained_tool_custody

    def custody():
        return retained_tool_custody(tmp_path, "root", load_task_result(tmp_path, "root"))

    def held():
        return {item["receipt_id"]: item["reason"] for item in custody() if item["kind"] == "merge_operation"}

    def stored():
        return {r["receipt_id"]: r for r in load_task_result(tmp_path, "root")["merge_receipts"]}

    assert merge_receipts.UNKNOWN_EFFECT in ACCEPTABLE_UNKNOWN
    write_task_result(tmp_path, "root", STATUS_RUNNING, root_task_id="root", delegation_role="root", launch_handoffs={
        "op-live": {"tool": "pr_merge", "task_id": "root", "root_task_id": "root", "claimed_at": "t",
                    "state": "claimed", "local_owner": {}, "owner_proof": {}}})
    unknown = {"schema": 1, "state": "settled", "repo": {"url": URL}, "number": 7,
               "requested": {"method": "squash", "expected_head_sha": HEAD},
               "effect": {"ok": False, "failure": "exit", "http_status": 502},
               "outcome": {"status": "unknown", "reason": "pr_state_open_after_unconfirmed_effect"}}
    cases = {"unknown": {}, "live": {"launch_operation_ids": ["op-live"]},
             "unsettled": {"state": "effect_attempted", "outcome": {}},
             "queued": {"outcome": {"status": "queued", "reason": "auto_merge_pending"}},
             "answered": {"effect": {"ok": True, "failure": "", "merge_sha": MERGE},
                          "outcome": {"status": "unknown", "reason": "merge_answered_pr_still_open"}}}
    for name, facts in cases.items():
        merge_receipts.write_receipt(tmp_path, "root", {**unknown, "receipt_id": name,
                                                        "launch_operation_ids": [f"op-{name}"], **facts})
    before = stored()
    assert held() == {"unknown": "merge_effect_unknown", "live": "merge_call_unreturned",
                      "unsettled": "merge_unsettled", "queued": "merge_queued", "answered": "merge_answered"}
    by = {"actor": "owner", "text": "go on", "at": "2026-11-09T20:00:00+00:00"}
    outcome = accept_unknown_custody(tmp_path, {"root"}, [*cases, "gone"], root_task_id="root", by=by)
    assert outcome["accepted"] == ["unknown"] and outcome["unmatched"] == ["gone"]
    assert sorted((row["id"], row["reason"]) for row in outcome["refused"]) == [
        ("answered", "merge_answered"), ("live", "merge_call_unreturned"), ("queued", "merge_queued"),
        ("unsettled", "merge_unsettled")]
    after = stored()
    assert after["unknown"].pop("unknown_accepted") == {**by, "reason": "merge_effect_unknown", "replay_authorized": False}
    assert after == before, "no fact, revision or publication of any receipt changed"
    assert "unknown" not in held()
    assert [item["operation_id"] for item in custody() if item["kind"] == "tool_handoff"] == ["op-live"], \
        "a live call stays its own hold"
    again = accept_unknown_custody(tmp_path, {"root"}, ["unknown"], root_task_id="root", by={**by, "text": "again"})
    assert again == {"accepted": [], "refused": [], "unmatched": ["unknown"]}, "nothing is recorded twice"

    # Observers whose snapshot predates the decision, or that carry one of their own, write facts only.
    merge_receipts.write_receipt(tmp_path, "root", {**before["unknown"], "observed_after": {"state": "OPEN"}})
    merge_receipts.write_receipt(tmp_path, "root", {**before["queued"], "unknown_accepted": {"actor": "forged"}})
    assert stored()["unknown"]["unknown_accepted"]["text"] == "go on" and "unknown_accepted" not in stored()["queued"]
    assert "unknown" not in held()
    merge_receipts.write_receipt(tmp_path, "root", {**before["unknown"], "launch_operation_ids": ["op-live"]})
    assert held()["unknown"] == "merge_call_unreturned", "a new call of the request holds it while it runs"


def test_a_refusal_recorded_after_an_observer_published_advances_the_revision(tmp_path):
    """An observer may publish the queued readback before the sender records GitHub's refusal. The refusal changes
    every wording, so it advances the revision, and the stale publication can no longer be recorded over it."""
    from ouroboros import merge_receipts
    from ouroboros.task_results import STATUS_RUNNING, write_task_result

    write_task_result(tmp_path, "root", STATUS_RUNNING, root_task_id="root", delegation_role="root")
    intent = {"schema": 1, "receipt_id": "r1", "state": "intent_recorded", "repo": {"url": URL}, "number": 7,
              "requested": {"method": "squash", "expected_head_sha": HEAD}, "observed_before": {}, "review": {},
              "coverage": {}}
    merge_receipts.write_receipt(tmp_path, "root", intent)
    seen = merge_receipts.write_receipt(tmp_path, "root", {**intent, "state": "settled",
        "outcome": {"status": "queued", "reason": "auto_merge_pending"}, "observed_after": {"state": "OPEN", "auto_merge": True}})
    published = merge_receipts.write_receipt(tmp_path, "root", seen, publication={"body": {"status": "published"}})
    assert published["publication"] and "refused" not in merge_receipts.card_row_text(published)

    superseded = {}
    refused = merge_receipts.write_receipt(tmp_path, "root", {**intent, "state": "effect_attempted", "effect": {
        "ok": False, "failure": "server_refused", "exit_code": 1, "http_status": 405,
        "output": "⚠️ GH_ERROR: " + NOT_MERGEABLE[2].strip()}}, superseded=superseded)
    assert refused["revision"] == published["revision"] + 1 and "publication" not in refused
    assert superseded == {"body": {"status": "published"}}  # the projection the sender's settlement must refresh
    assert refused["outcome"] == seen["outcome"] and refused["observed_after"] == seen["observed_after"]
    assert "(PR now OPEN, auto-merge on; request refused by GitHub)" in merge_receipts.card_row_text(refused)
    stale = merge_receipts.write_receipt(tmp_path, "root", published, publication={"body": {"status": "published"}},
                                         superseded=superseded)
    assert stale["revision"] == refused["revision"] and "publication" not in stale and superseded == {}
    assert merge_receipts.effect_is_definite(stale)
