"""#1556/#2455: gh's own pre-merge refusal, and GitHub's two exact refusals of the sent merge, are proven no-effects;
nothing looser is.

Grammar controls run the real ``_gh_run`` with only the ``gh`` subprocess replaced.
Consumer controls enter at ``ToolRegistry.execute_result("pr_merge")`` and read the
receipt, tool custody, Continue and cold sleep. By the owner's decision a proven refusal
ends this attempt's custody even beside an unreadable, closed or foreign-queued PR, whose
readback alone keeps the receipt read-back-only; a later explicit call for an open PR
without a queue sends exactly one new merge. An unknown request keeps its hold and is
only read back, never resent.
"""
from __future__ import annotations

import subprocess
from types import SimpleNamespace

import pytest

from ouroboros.task_results import load_task_result
from ouroboros.tools import github
from ouroboros.tools.registry import ToolContext
from tests.test_batch4_producer_custody import _consumers
from tests.test_pr_merge_receipts import HEAD, MERGE, URL, FakeGh


REPAIRED = "e" * 40
AUTO = "To have the pull request merged after all the requirements have been met, add the `--auto` flag.\n"
ADMIN = "To use administrator privileges to immediately merge the pull request, add the `--admin` flag.\n"
DIRTY, BLOCKED, BEHIND = ("the merge commit cannot be cleanly created", "the base branch policy prohibits the merge",
                          "the head branch is not up to date with the base branch")
# GitHub's answers to the sent mergePullRequest mutation, as gh prints them (cli/cli#8092).
CONFLICT = "GraphQL: Pull Request has merge conflicts (mergePullRequest)\n"
BASE_RACE = "GraphQL: Base branch was modified. Review and try the merge again. (mergePullRequest)\n"
HEAD_MOVED = "GraphQL: Head branch was modified. Review and try the merge again. (mergePullRequest)\n"
PERMISSION = "GraphQL: Resource not accessible by integration (mergePullRequest)\n"
INTERNAL = "GraphQL: Something went wrong while executing your query. Please include `A1B2:3C4D` when reporting.\n"


def refusal(reason, number=7):
    return f"X Pull request octo/demo#{number} is not mergeable: {reason}.\n"


def hint(action="merge", number=7, remote="origin", base="main", target=""):
    return ("Run the following to resolve the merge conflicts locally:\n"
            f"  gh pr checkout {number} && git fetch {remote} {base} && git {action} {target or f'{remote}/{base}'}\n")


def _classify(tmp_path, monkeypatch, stderr, *, method="squash", repo="", rc=1, stdout="", args=None):
    system = tmp_path / "system"
    system.mkdir(exist_ok=True)
    ctx = ToolContext(repo_dir=system, drive_root=tmp_path / "data", task_id="task-fixture")
    seen = []

    def run(argv, **_kw):
        seen.append(argv)
        return SimpleNamespace(returncode=rc, stdout=stdout, stderr=stderr)

    monkeypatch.setattr(github.subprocess, "run", run)
    monkeypatch.setattr(github, "_gh_env", lambda _ctx: {})
    monkeypatch.setattr(github, "github_token_from_env_or_settings", lambda: "")
    result = github._gh_run(args or ["pr", "merge", "7", f"--{method}", "--match-head-commit", HEAD], ctx, repo=repo)
    assert seen and (seen[0][-2:] == ["--repo", repo]) is bool(repo)
    return result


@pytest.mark.parametrize("method,repo,stderr", [
    ("squash", "", refusal(DIRTY) + AUTO),
    ("merge", "", refusal(DIRTY) + AUTO + hint("merge")),
    ("squash", "", refusal(DIRTY) + AUTO + hint("merge", base="release/2026.10")),
], ids=["dirty", "dirty-merge-hint", "dirty-slash-base"])
def test_gh_pre_merge_refusals_are_pre_effect(tmp_path, monkeypatch, method, repo, stderr):
    result = _classify(tmp_path, monkeypatch, stderr, method=method, repo=repo)

    assert (result.ok, result.exit_code, result.failure) == (False, 1, "pre_effect")
    # The receipt keeps a bounded three-line head; the decision read the whole stderr first.
    assert result.text.count(" | ") <= 2 and "&& git" not in result.text


@pytest.mark.parametrize(("stderr", "method", "repo", "body"), [
    pytest.param(CONFLICT, "squash", "", True, id='conflict-squash--True'),
    pytest.param(CONFLICT, "rebase", "", False, id='conflict-rebase--False'),
    pytest.param(BASE_RACE, "merge", "octo/demo", True, id='base-race-merge-octo/demo-True'),
    pytest.param(BASE_RACE, "rebase", "octo/demo", False, id='base-race-rebase-octo/demo-False'),
])
def test_githubs_exact_refusals_of_the_sent_merge_are_server_refused(tmp_path, monkeypatch, method, repo, body, stderr):
    args = ["pr", "merge", "7", f"--{method}", "--match-head-commit", HEAD] + (["--body-file", "-"] if body else [])
    result = _classify(tmp_path, monkeypatch, stderr, args=args, repo=repo)

    assert (result.ok, result.exit_code, result.failure) == (False, 1, "server_refused")
    assert result.text == "⚠️ GH_ERROR: " + stderr.strip()


@pytest.mark.parametrize("body_file,failure", [("-", "pre_effect"), ("body.txt", "exit")])
def test_stdin_body_retains_exact_pre_effect_refusal(tmp_path, monkeypatch, body_file, failure):
    args = ["pr", "merge", "7", "--merge", "--match-head-commit", HEAD, "--body-file", body_file]
    result = _classify(tmp_path, monkeypatch, refusal(DIRTY) + AUTO, args=args, repo="octo/demo")
    assert result.failure == failure


@pytest.mark.parametrize("stderr,kw", [
    (refusal(DIRTY) + AUTO + ADMIN, {}),
    (refusal(BLOCKED) + AUTO, {}),
    (refusal(BLOCKED) + AUTO + hint() + ADMIN, {}),
    (refusal(DIRTY) + AUTO + hint(), {"repo": "octo/demo"}),
    (refusal(DIRTY) + AUTO + hint(number=8), {}),
    (refusal(DIRTY, number=8) + AUTO, {}),
    (refusal(DIRTY) + AUTO + hint(target="origin/other"), {}),
    (refusal(DIRTY) + AUTO + "HTTP 503: Service Unavailable (https://api.github.com/graphql)\n", {}),
    ("HTTP 503: Service Unavailable (https://api.github.com/graphql)\n" + refusal(DIRTY) + AUTO, {}),
    (refusal(DIRTY), {}),
    # The exact BLOCKED text proves nothing outside the one invocation and exit it belongs to.
    (refusal(BLOCKED, number=8) + AUTO + ADMIN, {}),
    (refusal(BLOCKED) + AUTO + ADMIN, {"args": ["pr", "merge", "7", "--squash", "--match-head-commit", HEAD,
                                                "--admin"]}),
    ("GraphQL: Pull Request is not mergeable (mergePullRequest)\n", {}),
    # Only GitHub's two whole answers to the sent mutation qualify; anything around or beside them stays unknown.
    (CONFLICT + BASE_RACE, {}),
    (CONFLICT + "HTTP 502: Bad Gateway (https://api.github.com/graphql)\n", {}),
    ("gh: " + CONFLICT, {}),
    (CONFLICT.replace(" (mergePullRequest)", ""), {}),
    (HEAD_MOVED, {}),
    (PERMISSION, {}),
    (INTERNAL, {}),
], ids=["dirty-admin", "blocked-no-admin", "blocked-hint", "hint-with-repo", "hint-number", "line-number",
        "hint-target", "status-suffix", "status-prefix",
        "truncated", "blocked-line-number", "extra-flag", "server-refusal", "two-answers", "answer-with-status", "answer-prefix",
        "answer-without-path", "head-moved", "permission", "internal"])
def test_anything_but_the_exact_refusal_stays_an_unknown_exit(tmp_path, monkeypatch, stderr, kw):
    result = _classify(tmp_path, monkeypatch, stderr, **kw)

    assert result.ok is False and result.failure == "exit", result


def test_a_merge_timeout_is_never_pre_effect(tmp_path, monkeypatch):
    def run(argv, **_kw):
        raise subprocess.TimeoutExpired(argv, 120)

    monkeypatch.setattr(github.subprocess, "run", run)
    monkeypatch.setattr(github, "_gh_env", lambda _ctx: {})
    ctx = ToolContext(repo_dir=tmp_path, drive_root=tmp_path, task_id="task-fixture")
    assert github._gh_run(["pr", "merge", "7", "--merge", "--match-head-commit", HEAD], ctx).failure == "timeout"


# --- the consumer chain -----------------------------------------------------------------


@pytest.fixture
def chain(tmp_path, monkeypatch):
    from tests.test_batch4_producer_custody import _registry

    registry, queue, workers = _registry(tmp_path, monkeypatch)
    fake = FakeGh(tmp_path)
    fake.pr.update(mergeStateStatus="DIRTY")
    state = SimpleNamespace(registry=registry, queue=queue, workers=workers, fake=fake, merges=[], merge_inputs=[], answers=[],
                            root=tmp_path, observe=None, observed=[])
    real_run = subprocess.run

    def run(argv, **kw):
        if argv[:1] != ["gh"]:
            return real_run(argv, **kw)
        args = argv[1:]
        assert "--repo" not in args and kw["cwd"] == str(registry._ctx.workspace_root)
        if args[:2] == ["pr", "merge"]:
            state.merges.append(args)
            state.merge_inputs.append(kw.get("input"))
            fake.calls.append(list(args))
            answer = state.answers.pop(0)
            if answer == "timeout":
                raise subprocess.TimeoutExpired(argv, 120)
            if answer == "merged":
                fake.pr.update(state="MERGED", mergeCommit={"oid": MERGE}, mergedAt="2026-10-08T12:00:00Z")
                return SimpleNamespace(returncode=0, stdout="", stderr="")
            if answer == "merged_elsewhere" or answer[:1] == ("merged_elsewhere",):  # refused; another actor merged meanwhile
                fake.pr.update(state="MERGED", mergeCommit={"oid": MERGE}, mergedAt="2026-10-08T12:00:00Z")
                answer = answer[1:] if isinstance(answer, tuple) else (1, "", refusal(DIRTY) + AUTO)
            if answer[:1] == ("queued_elsewhere",):  # refused; another actor enabled auto-merge meanwhile
                fake.pr.update(autoMergeRequest={"enabledAt": "by another actor"})
                answer = answer[1:]
            rc, stdout, stderr = answer
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


def _merge(chain, head, method="squash"):
    return chain.registry.execute_result("pr_merge", {"number": 7, "expected_head_sha": head, "method": method})


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


REFUSALS = pytest.mark.parametrize("stderr,failure,ended", [
    (refusal(DIRTY) + AUTO, "pre_effect", "request refused before effect"),
    (CONFLICT, "server_refused", "request refused by GitHub"),
    (BASE_RACE, "server_refused", "request refused by GitHub"),
], ids=["gh", "conflict", "base-race"])


@REFUSALS
def test_another_actors_queue_then_close_stays_observed_while_our_refused_request_holds_nothing(chain, stderr, failure, ended):
    """Owner decision (#2455): the proven refusal ends this attempt's custody; another actor's auto-merge, enabled while
    the request was answered, and the PR's later close stay the observed outcome, worded as theirs, never resent."""
    from ouroboros.merge_receipts import card_row_text, public_block

    chain.answers.append(("queued_elsewhere", 1, "", stderr))
    first = _merge(chain, HEAD)
    receipt = _receipts(chain)[0]
    assert (receipt["outcome"]["status"], receipt["effect"]["failure"]) == ("queued", failure)
    assert first.text.startswith(f"⚠️ PR_MERGE_REFUSED: ⚠️ PR #7 merge: queued (PR now OPEN, auto-merge on; {ended});")
    assert first.meta["operation_outcome"] == "completed_no_effect" and _held(chain) is False
    assert f"- Outcome: **queued** (PR now OPEN, auto-merge on; {ended})" in public_block(receipt)
    chain.fake.pr.update(state="CLOSED", autoMergeRequest=None)
    output = _merge(chain, HEAD)
    (current,) = _receipts(chain)
    for text in (output.text, card_row_text(current), public_block(current)):
        assert "PR now CLOSED" in text and ended in text
        assert "effect unconfirmed" not in text
    assert current["effect"] == receipt["effect"] and current["outcome"] == receipt["outcome"]
    assert output.meta["operation_outcome"] == "completed_no_effect"
    assert "changed nothing, so it holds no task custody" in output.text
    assert "Until GitHub shows the PR open without a queue, repeat calls only read it back." in output.text
    assert _held(chain) is False
    _merge(chain, HEAD)
    assert len(chain.merges) == 1


@pytest.mark.parametrize("stderr,repair", [(CONFLICT, True), (BASE_RACE, False)],
                         ids=["conflict-then-repaired-head", "base-race-then-same-head"])
def test_githubs_refusal_ends_the_attempt_and_an_explicit_call_merges_once(chain, stderr, repair):
    """#2455: GitHub refused the sent merge (a conflict, or the base moving under it). Nothing is held or resent;
    the caller's next explicit call for an open PR without a queue sends exactly one new merge."""
    chain.answers += [(1, "", stderr), "merged"]
    first = _merge(chain, HEAD)
    assert first.text.startswith("⚠️ PR_MERGE_REFUSED: ⚠️ PR #7 merge: refused (by GitHub);"), first.text
    assert f"Merge request: failure=server_refused, exit_code=1: ⚠️ GH_ERROR: {stderr.strip()}" in first.text
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
    assert [call[:6] for call in chain.merges] == [["pr", "merge", "7", "--squash", "--match-head-commit", sha]
                                                   for sha in (HEAD, head)]
    assert [(r["outcome"]["status"], r["requested"]["expected_head_sha"]) for r in _receipts(chain)] == [
        ("refused", HEAD), ("merged", head)]
    assert _receipts(chain)[1]["outcome"]["attribution"] == "this_call" and _held(chain) is False


def test_a_github_refusal_with_a_failed_readback_settles_once_github_answers(chain):
    chain.fake.fail_view_after_merge = True
    chain.answers += [(1, "", CONFLICT), "merged"]
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
    chain.answers.append((1, "", CONFLICT))
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


def test_a_merge_by_someone_else_after_our_refusal_leaves_our_receipt_refused(chain):
    chain.answers.append((1, "", BASE_RACE))
    _merge(chain, HEAD)
    chain.fake.pr.update(state="MERGED", mergeCommit={"oid": MERGE}, mergedAt="2026-10-08T12:00:00Z")
    later = _merge(chain, HEAD)
    assert later.text.startswith("⚠️ PR_MERGE_REFUSED: pr_not_open") and len(chain.merges) == 1
    assert [r["outcome"]["status"] for r in _receipts(chain)] == ["refused"] and _held(chain) is False


def test_an_old_unknown_github_refusal_is_not_reclassified(chain, monkeypatch):
    """A receipt the earlier classifier stored as an unknown exit keeps its hold: only a new send is classified."""
    classify = github._merge_refusal
    monkeypatch.setattr(github, "_merge_refusal", lambda *_args: "exit")
    chain.answers.append((1, "", CONFLICT))
    assert "PR_MERGE_UNKNOWN" in _merge(chain, HEAD).text
    monkeypatch.setattr(github, "_merge_refusal", classify)
    _repair(chain)
    again = _merge(chain, REPAIRED)
    assert "had no confirmed outcome" in again.text and again.meta["operation_outcome"] == "unknown"
    (receipt,) = _receipts(chain)
    assert receipt["effect"]["failure"] == "exit" and CONFLICT.strip() in receipt["effect"]["output"]
    assert receipt["outcome"]["status"] == "unknown" and len(chain.merges) == 1 and _held(chain) is True


@pytest.mark.parametrize("method,stderr", [
    ("squash", refusal(DIRTY) + AUTO),
    ("merge", refusal(DIRTY) + AUTO + hint("merge")),
    ("rebase", refusal(DIRTY) + AUTO + hint("rebase", base="release/2026.10")),
], ids=["short", "merge-hint", "rebase-hint"])
def test_proven_conflict_refusal_frees_custody_and_a_repaired_head_merges_once(chain, method, stderr):
    from ouroboros.update_qualification import commit_qualification

    owner_intro = "Summary of the change.\n\nOwner notes keep `literal code` and  spacing.\n\n"
    owner_footer = "\n\nOwner follow-up:\n  Preserve this indentation and punctuation: →."
    chain.fake.pr["body"] = (owner_intro + "<!-- ouroboros:merge-receipt ab -->\nprior receipt\n"
                             "<!-- /ouroboros:merge-receipt -->" + owner_footer)
    chain.answers += [(1, "", stderr), "merged"]
    first = _merge(chain, HEAD, method)
    assert first.text.startswith("⚠️ PR_MERGE_REFUSED"), first.text
    assert first.meta["operation_outcome"] == "completed_no_effect"
    (receipt,) = _receipts(chain)
    assert receipt["outcome"]["status"] == "refused" and receipt["effect"]["failure"] == "pre_effect"
    assert receipt["launch_operation_ids"]  # the real registry invocation identity reached the receipt
    assert _held(chain) is False

    _repair(chain)
    stale = _merge(chain, HEAD, method)  # the old head is refused before any send
    assert "head_moved" in stale.text and len(chain.merges) == 1
    second = _merge(chain, REPAIRED, method)
    assert second.meta["operation_outcome"] == "completed", second.text
    assert len(chain.merges) == 2
    for call, body, head, retained in zip(chain.merges, chain.merge_inputs, (HEAD, REPAIRED), _receipts(chain)):
        expected_args = ["pr", "merge", "7", f"--{method}", "--match-head-commit", head]
        if method == "rebase":
            assert call == expected_args  # rebase has no merge commit to carry a receipt
            assert body is None
            continue
        assert call == [*expected_args, "--body-file", "-"]
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


@pytest.mark.parametrize("answer", [
    (1, "", refusal(DIRTY) + AUTO + "HTTP 503: Service Unavailable (https://api.github.com/graphql)\n"),
    "timeout",
    (1, "", HEAD_MOVED),
], ids=["status-suffix", "timeout", "github-head-moved"])
def test_unknown_refusal_keeps_its_hold_and_is_only_read_back(chain, answer):
    chain.answers.append(answer)
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
def test_a_proven_refusal_holds_no_custody_while_its_readback_still_observes(chain, case, stderr, failure, ended):
    """Owner decision (#2455), superseding the earlier hold: this attempt's custody ends with its proven refusal even
    while GitHub shows another actor's queue or cannot be read; that observation keeps the receipt read-back-only."""
    if case == "auto_merge_pending":
        chain.answers.append(("queued_elsewhere", 1, "", stderr))
    else:
        chain.fake.fail_view_after_merge = True
        chain.answers.append((1, "", stderr))
    first = _merge(chain, HEAD)
    assert first.text.startswith("⚠️ PR_MERGE_REFUSED:") and ended in first.text, first.text
    assert first.meta["operation_outcome"] == "completed_no_effect"
    (receipt,) = _receipts(chain)
    assert receipt["effect"]["failure"] == failure
    assert receipt["outcome"]["status"] == ("queued" if case == "auto_merge_pending" else "unknown")
    assert _held(chain) is False
    _repair(chain)
    again = _merge(chain, REPAIRED)
    assert "sent no new merge request" in again.text and len(chain.merges) == 1
    assert again.meta["operation_outcome"] == "completed_no_effect"
    _consumers(chain.root, chain.registry, chain.queue, chain.workers, False)


@pytest.mark.parametrize("answer,ended", [("merged_elsewhere", "request refused before effect"),
                                          (("merged_elsewhere", 1, "", CONFLICT), "request refused by GitHub")],
                         ids=["gh", "github"])
def test_a_refusal_beside_an_observed_merge_settles_without_claiming_it(chain, answer, ended):
    from ouroboros.merge_receipts import card_row_text, public_block

    chain.answers.append(answer)
    result = _merge(chain, HEAD)
    # The PR is merged, by another request: this operation changed nothing and claims no success.
    assert (result.status, result.meta["operation_outcome"]) == ("ok", "completed_no_effect"), result.text
    assert "GitHub shows the PR merged by another request." in result.text
    (receipt,) = _receipts(chain)
    assert receipt["effect"]["failure"] == ("pre_effect" if answer == "merged_elsewhere" else "server_refused")
    assert receipt["outcome"]["status"] == "merged" and receipt["outcome"]["attribution"] == "unproven"
    for text in (result.text, card_row_text(receipt), public_block(receipt)):
        assert f"merged** (not by this request; {ended})" in text or f"merged (not by this request; {ended})" in text
    assert card_row_text(receipt).startswith("⚠️ ")
    assert _held(chain) is False and len(chain.merges) == 1


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
        "ok": False, "failure": "server_refused", "exit_code": 1, "output": "⚠️ GH_ERROR: " + CONFLICT.strip()}},
        superseded=superseded)
    assert refused["revision"] == published["revision"] + 1 and "publication" not in refused
    assert superseded == {"body": {"status": "published"}}  # the projection the sender's settlement must refresh
    assert refused["outcome"] == seen["outcome"] and refused["observed_after"] == seen["observed_after"]
    assert "(PR now OPEN, auto-merge on; request refused by GitHub)" in merge_receipts.card_row_text(refused)
    stale = merge_receipts.write_receipt(tmp_path, "root", published, publication={"body": {"status": "published"}},
                                         superseded=superseded)
    assert stale["revision"] == refused["revision"] and "publication" not in stale and superseded == {}
    assert merge_receipts.effect_is_definite(stale)


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
        "effect": {"ok": False, "failure": "server_refused", "exit_code": 1, "output": "⚠️ GH_ERROR: " + CONFLICT.strip()}})
    held = retained_tool_custody(tmp_path, "root", load_task_result(tmp_path, "root"))
    assert [item["kind"] for item in held] == ["tool_handoff"]  # only the worker, which may still run
    assert retire_process_invocations(tmp_path, pid=999999, process_birth="fixture-birth") == ["op-1"]
    assert retained_tool_custody(tmp_path, "root", load_task_result(tmp_path, "root")) == []
