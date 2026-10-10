"""#1556: gh's own pre-merge conflict refusal is a proven no-effect, nothing looser is.

Grammar controls run the real ``_gh_run`` with only the ``gh`` subprocess replaced.
Consumer controls enter at ``ToolRegistry.execute_result("pr_merge")`` and read the
receipt, tool custody, Continue and cold sleep: a proven refusal leaves no merge hold
and a later explicit call for the repaired head sends exactly one new merge; an
unknown or queued request keeps its hold and is only read back, never resent.
"""
from __future__ import annotations

import subprocess
from types import SimpleNamespace

import pytest

from ouroboros.task_results import load_task_result
from ouroboros.tools import github
from ouroboros.tools.registry import ToolContext
from tests.test_pr_merge_receipts import HEAD, MERGE, FakeGh

pytestmark = pytest.mark.serial

REPAIRED = "e" * 40
AUTO = "To have the pull request merged after all the requirements have been met, add the `--auto` flag.\n"
ADMIN = "To use administrator privileges to immediately merge the pull request, add the `--admin` flag.\n"
DIRTY, BLOCKED, BEHIND = ("the merge commit cannot be cleanly created", "the base branch policy prohibits the merge",
                          "the head branch is not up to date with the base branch")


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
    ("squash", "octo/demo", refusal(DIRTY) + AUTO),
    ("merge", "", refusal(DIRTY) + AUTO + hint("merge")),
    ("squash", "", refusal(DIRTY) + AUTO + hint("merge", base="release/2026.10")),
    ("rebase", "", refusal(DIRTY) + AUTO + hint("rebase", remote="upstream", base="team/main")),
    ("merge", "octo/demo", refusal(BLOCKED) + AUTO + ADMIN),
    ("rebase", "", refusal(BEHIND) + AUTO + ADMIN),
], ids=["dirty", "dirty-repo", "dirty-merge-hint", "dirty-slash-base", "dirty-rebase-hint", "blocked", "behind"])
def test_gh_pre_merge_refusals_are_pre_effect(tmp_path, monkeypatch, method, repo, stderr):
    result = _classify(tmp_path, monkeypatch, stderr, method=method, repo=repo)

    assert (result.ok, result.exit_code, result.failure) == (False, 1, "pre_effect")
    # The receipt keeps a bounded three-line head; the decision read the whole stderr first.
    assert result.text.count(" | ") <= 2 and "&& git" not in result.text


@pytest.mark.parametrize("stderr,kw", [
    (refusal(DIRTY) + AUTO + ADMIN, {}),
    (refusal(BLOCKED) + AUTO, {}),
    (refusal(BLOCKED) + AUTO + hint() + ADMIN, {}),
    (refusal(DIRTY) + AUTO + hint(), {"repo": "octo/demo"}),
    (refusal(DIRTY) + AUTO + hint(number=8), {}),
    (refusal(DIRTY, number=8) + AUTO, {}),
    (refusal(DIRTY) + AUTO + hint("rebase"), {"method": "squash"}),
    (refusal(DIRTY) + AUTO + hint("merge"), {"method": "rebase"}),
    (refusal(DIRTY) + AUTO + hint(target="origin/other"), {}),
    (refusal(DIRTY) + AUTO + "HTTP 503: Service Unavailable (https://api.github.com/graphql)\n", {}),
    ("HTTP 503: Service Unavailable (https://api.github.com/graphql)\n" + refusal(DIRTY) + AUTO, {}),
    (refusal(DIRTY), {}),
    (refusal(DIRTY) + AUTO, {"stdout": "Merged pull request octo/demo#7"}),
    # The exact BLOCKED text proves nothing outside the one invocation and exit it belongs to.
    (refusal(BLOCKED, number=8) + AUTO + ADMIN, {}),
    (refusal(BLOCKED) + AUTO + ADMIN, {"stdout": "Merged pull request octo/demo#7"}),
    (refusal(BLOCKED) + AUTO + ADMIN, {"rc": 2}),
    (refusal(BLOCKED) + AUTO + ADMIN, {"rc": -15}),
    (refusal(BLOCKED) + AUTO + ADMIN, {"args": ["pr", "view", "7"]}),
    (refusal(BLOCKED) + AUTO + ADMIN, {"args": ["pr", "merge", "7", "--squash", "--match-head-commit", HEAD,
                                                "--admin"]}),
    (refusal(BLOCKED) + AUTO + ADMIN, {"args": ["pr", "merge", "7", "--auto", "--match-head-commit", HEAD]}),
    ("GraphQL: Pull Request is not mergeable (mergePullRequest)\n", {}),
], ids=["dirty-admin", "blocked-no-admin", "blocked-hint", "hint-with-repo", "hint-number", "line-number",
        "rebase-hint-for-squash", "merge-hint-for-rebase", "hint-target", "status-suffix", "status-prefix",
        "truncated", "stdout", "blocked-line-number", "blocked-stdout", "exit-2", "signal", "other-command",
        "extra-flag", "auto-flag", "server-refusal"])
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
    state = SimpleNamespace(registry=registry, queue=queue, workers=workers, fake=fake, merges=[], answers=[],
                            root=tmp_path)
    real_run = subprocess.run

    def run(argv, **kw):
        if argv[:1] != ["gh"]:
            return real_run(argv, **kw)
        args = argv[1:]
        assert "--repo" not in args and kw["cwd"] == str(registry._ctx.workspace_root)
        if args[:2] == ["pr", "merge"]:
            state.merges.append(args)
            fake.calls.append(list(args))
            answer = state.answers.pop(0)
            if answer == "timeout":
                raise subprocess.TimeoutExpired(argv, 120)
            if answer == "merged":
                fake.pr.update(state="MERGED", mergeCommit={"oid": MERGE}, mergedAt="2026-10-08T12:00:00Z")
                return SimpleNamespace(returncode=0, stdout="", stderr="")
            if answer == "merged_elsewhere":  # gh refused, another actor merged meanwhile
                fake.pr.update(state="MERGED", mergeCommit={"oid": MERGE}, mergedAt="2026-10-08T12:00:00Z")
                answer = (1, "", refusal(DIRTY) + AUTO)
            rc, stdout, stderr = answer
            return SimpleNamespace(returncode=rc, stdout=stdout, stderr=stderr)
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


@pytest.mark.parametrize("method,stderr", [
    ("squash", refusal(DIRTY) + AUTO),
    ("merge", refusal(DIRTY) + AUTO + hint("merge")),
    ("rebase", refusal(DIRTY) + AUTO + hint("rebase", base="release/2026.10")),
], ids=["short", "merge-hint", "rebase-hint"])
def test_proven_conflict_refusal_frees_custody_and_a_repaired_head_merges_once(chain, method, stderr):
    from tests.test_batch4_producer_custody import _consumers

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
    assert chain.merges == [["pr", "merge", "7", f"--{method}", "--match-head-commit", HEAD],
                            ["pr", "merge", "7", f"--{method}", "--match-head-commit", REPAIRED]]
    assert [(r["outcome"]["status"], r["requested"]["expected_head_sha"]) for r in _receipts(chain)] == [
        ("refused", HEAD), ("merged", REPAIRED)]
    assert _receipts(chain)[1]["outcome"]["attribution"] == "this_call"
    assert _held(chain) is False
    _consumers(chain.root, chain.registry, chain.queue, chain.workers, False)


@pytest.mark.parametrize("answer", [
    (1, "", refusal(DIRTY) + AUTO + "HTTP 503: Service Unavailable (https://api.github.com/graphql)\n"),
    (1, "Merging pull request octo/demo#7", refusal(DIRTY) + AUTO),
    (1, "", refusal(DIRTY) + AUTO + ADMIN),
    "timeout",
], ids=["status-suffix", "stdout", "malformed", "timeout"])
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


@pytest.mark.parametrize("case", ["auto_merge_pending", "readback_failed"])
def test_proven_refusal_without_a_definite_readback_is_not_released(chain, case):
    if case == "auto_merge_pending":
        chain.fake.pr.update(autoMergeRequest={"enabledAt": "2026-10-08T11:00:00Z"})
    else:
        chain.fake.fail_view_after_merge = True
    chain.answers.append((1, "", refusal(DIRTY) + AUTO))
    first = _merge(chain, HEAD)
    assert first.meta["operation_outcome"] == "unknown", first.text
    (receipt,) = _receipts(chain)
    assert receipt["effect"]["failure"] == "pre_effect"
    assert receipt["outcome"]["status"] == ("queued" if case == "auto_merge_pending" else "unknown")
    assert _held(chain) is True
    _repair(chain)
    _merge(chain, REPAIRED)
    assert len(chain.merges) == 1


def test_a_refusal_beside_an_observed_merge_settles_without_claiming_it(chain):
    chain.answers.append("merged_elsewhere")
    result = _merge(chain, HEAD)
    assert result.meta["operation_outcome"] == "completed", result.text
    (receipt,) = _receipts(chain)
    assert receipt["effect"]["failure"] == "pre_effect"
    assert receipt["outcome"]["status"] == "merged" and receipt["outcome"]["attribution"] == "unproven"
    assert _held(chain) is False and len(chain.merges) == 1
