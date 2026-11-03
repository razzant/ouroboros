"""A commit that lands under a review record is typed against the tree that record reviewed (#2242).

The commit gate runs for real over a real repository (only the reviewer answers are stubbed, as in
``test_review_doctrine``): every bound commit adds ``commit_landings`` to its record with its own tree
read from Git, the author's decision names the commit it governed, a failed Git command adds nothing,
and the task's Reviews projection carries the fact without turning any of it into a verdict.
"""
from __future__ import annotations

import re

from ouroboros import review_ledger as rl
from ouroboros.review_doctrine import commit_landing_fact, landed_record, public_commit_landings
from ouroboros.review_projection import compact_review_projection, ledger_record_panel, task_ledger_records
from ouroboros.tools import git
from tests.test_git_review_preflight_gate import candidate  # noqa: F401
from tests.test_review_doctrine import _head, _install_reviewer, _mode, _prepare, _record, _reference
from tests.test_review_ledger import BOTH, CRITICAL, _answer, _raw

PASSING = [_raw("critic", "openai/gpt-5", parts=BOTH)]
REJECTING = [_raw("critic", "openai/gpt-5", parts=BOTH,
                  answers={"change": _answer("change", "FAIL", [CRITICAL]), "coupling": _answer("coupling")})]


def _tree(ctx, rev="HEAD"):
    return git.run_cmd(["git", "rev-parse", f"{rev}^{{tree}}"], cwd=ctx.repo_dir).strip()


def _record_of(ctx, result):
    match = re.search(r"review_record_id: (\S+)", result)
    assert match, result
    return rl.load_record(rl.ledger_root(ctx), match.group(1))


def test_a_blocking_pass_lands_the_exact_reviewed_tree_with_no_author_decision(candidate, monkeypatch):  # noqa: F811
    ctx = candidate
    _mode(monkeypatch, "pro", "blocking")
    _prepare(ctx, monkeypatch)
    _install_reviewer(ctx, monkeypatch, PASSING)
    before = _head(ctx)

    result = git._commit_reviewed(ctx, "Fix amount", skip_advisory_review=True)

    head = _head(ctx)
    assert head != before, result
    record = _record_of(ctx, result)
    assert record["verdict"]["aggregate"] == "PASS" and not record.get("author_decision")  # the field is None
    [landed] = record["commit_landings"]
    assert landed["commit_sha"] == head and landed["tree_sha"] == _tree(ctx) == record["subject"]["tree_sha"]
    assert (landed["parent_sha"], landed["reviewed_subject"], landed["author_subject"]) == (before, "same", "")
    # The task's Reviews read it through the ledger panel: the verdict and the landing stay separate facts.
    records, _omitted = task_ledger_records({"id": ctx.task_id}, ctx.drive_root)
    [panel] = compact_review_projection([], records=records)["panels"]
    assert panel["aggregate_signal"] == "PASS" and panel["commit_landings"] == public_commit_landings(record)
    assert "author_decision" not in panel


def test_an_advisory_continuation_after_an_edit_lands_code_the_review_did_not_see(candidate, monkeypatch):  # noqa: F811
    ctx = candidate
    _mode(monkeypatch, "pro", "advisory")
    _prepare(ctx, monkeypatch)
    _install_reviewer(ctx, monkeypatch, PASSING)
    before = _head(ctx)
    first = git._commit_reviewed(ctx, "Fix amount", skip_advisory_review=True)
    assert "returned before commit" in first and _head(ctx) == before
    reference = _reference(first)
    reviewed_tree = _record(ctx, reference)["subject"]["tree_sha"]
    (ctx.repo_dir / "change.py").write_text("value = 3\n", encoding="utf-8")  # the author's fix after the review

    landed_result = git._commit_reviewed(
        ctx, "Fix amount", review_reference=reference,
        author_disposition={"disposition": "accepted", "rationale": "Applied the reviewer's note after the review."})

    head = _head(ctx)
    assert head != before, landed_result
    record = _record(ctx, reference)
    assert record["verdict"]["aggregate"] == "PASS"  # the critic's verdict stays the verdict of ITS subject
    [landed] = record["commit_landings"]
    assert landed["commit_sha"] == head and landed["tree_sha"] == _tree(ctx) != reviewed_tree
    assert landed["reviewed_subject"] == "different"
    decision = record["author_decision"]
    assert landed["author_subject"] == decision["subject_hash"] and decision["landing"]["landed"] is True
    panel = ledger_record_panel(record)
    assert panel["commit_landings"][0]["reviewed_subject"] == "different"
    assert panel["author_decision"]["rationale"] == "Applied the reviewer's note after the review."


def test_a_rejected_review_continued_on_the_same_bytes_keeps_its_fail_beside_the_landing(candidate, monkeypatch):  # noqa: F811
    ctx = candidate
    _mode(monkeypatch, "pro", "advisory")
    _prepare(ctx, monkeypatch)
    _install_reviewer(ctx, monkeypatch, REJECTING)
    first = git._commit_reviewed(ctx, "Fix amount", skip_advisory_review=True)
    reference = _reference(first)

    git._commit_reviewed(ctx, "Fix amount", review_reference=reference,
                         author_disposition={"disposition": "rejected", "rationale": "The sequence cannot occur."})

    record = _record(ctx, reference)
    assert record["verdict"]["aggregate"] == "FAIL"
    [landed] = record["commit_landings"]
    assert (landed["commit_sha"], landed["reviewed_subject"]) == (_head(ctx), "same")
    assert landed["author_subject"] == record["author_decision"]["subject_hash"]


def test_a_failed_git_command_never_lands_and_its_successful_retry_lands_once(candidate, monkeypatch):  # noqa: F811
    ctx = candidate
    _mode(monkeypatch, "pro", "advisory")
    _prepare(ctx, monkeypatch)
    _install_reviewer(ctx, monkeypatch, PASSING)
    before = _head(ctx)
    reference = _reference(git._commit_reviewed(ctx, "Fix amount", skip_advisory_review=True))
    real, failed = git.run_cmd, {"done": False}

    def fail_once(cmd, cwd=None, timeout=None):
        if list(cmd[:2]) == ["git", "commit"] and not failed["done"]:
            failed["done"] = True
            raise RuntimeError("commit refused")
        return real(cmd, cwd=cwd, timeout=timeout)

    monkeypatch.setattr(git, "run_cmd", fail_once)
    decision = {"disposition": "accepted", "rationale": "Reviewed bytes, unchanged."}
    broken = git._commit_reviewed(ctx, "Fix amount", review_reference=reference, author_disposition=decision)
    assert "GIT_ERROR" in broken and _head(ctx) == before
    assert "commit_landings" not in _record(ctx, reference)

    git._commit_reviewed(ctx, "Fix amount", review_reference=reference, author_disposition=decision)

    record = _record(ctx, reference)
    assert [row["commit_sha"] for row in record["commit_landings"]] == [_head(ctx)]
    again = rl.note_commit_landing(rl.ledger_root(ctx), record["record_id"], record["author_decision"]["landing"],
                                   {"commit_sha": _head(ctx), "tree_sha": _tree(ctx)})
    assert again["revision"] == record["revision"]  # the same landing twice is no new fact


def test_the_landing_fact_types_only_what_trees_prove():
    sha, tree, other = "a" * 40, "b" * 40, "c" * 40
    assert commit_landing_fact({"tree_sha": tree}, {"commit_sha": sha, "tree_sha": tree},
                               recorded_at="t")["reviewed_subject"] == "same"
    assert commit_landing_fact({"tree_sha": other}, {"commit_sha": sha, "tree_sha": tree},
                               recorded_at="t")["reviewed_subject"] == "different"
    for subject, commit in (({}, {"commit_sha": sha, "tree_sha": tree}), ({"tree_sha": tree}, {"commit_sha": sha})):
        assert commit_landing_fact(subject, commit, recorded_at="t")["reviewed_subject"] == "unknown"
    assert commit_landing_fact({"tree_sha": tree}, {"commit_sha": "abc123"}, recorded_at="t") is None
    rows = [{"commit_sha": sha, "reviewed_subject": "guessed"}, {"commit_sha": "short"}, "noise"]
    assert [row["reviewed_subject"] for row in public_commit_landings({"commit_landings": rows})] == ["unknown"]
    assert public_commit_landings({"commit_landings": {"commit_sha": sha}}) == []


def test_each_commit_lands_once_and_the_record_keeps_every_one_while_its_panel_says_what_it_left_out():
    record = {"record_id": "r", "subject": {"tree_sha": "b" * 40}, "verdict": {"aggregate": "PASS"}}
    for n in range(20):
        record = landed_record(record, None, {"commit_sha": f"{n:040x}", "tree_sha": "b" * 40}, recorded_at=str(n))
    assert [row["commit_sha"] for row in record["commit_landings"]] == [f"{n:040x}" for n in range(20)]
    assert landed_record(record, None, {"commit_sha": f"{19:040x}", "tree_sha": "b" * 40}, recorded_at="x") is None
    assert "author_decision" not in record  # an author landing joins only a decision the record holds
    panel = ledger_record_panel(record)
    assert [row["commit_sha"] for row in panel["commit_landings"]] == [f"{n:040x}" for n in range(4, 20)]
    assert panel["commit_landings_omitted"] == 4 and panel["source_ref"]["path"] == "state/review_ledger/r.json"


def test_a_landing_keeps_only_the_decision_it_names():
    sha, tree = "a" * 40, "b" * 40
    decision = {"subject_hash": "s" * 64, "disposition": "rejected", "rationale": "Why this commit went ahead."}
    named = commit_landing_fact({"tree_sha": tree}, {"commit_sha": sha, "tree_sha": tree, "author_subject": "s" * 64},
                                recorded_at="t", decision=decision)
    assert (named["author_disposition"], named["author_reason"]) == ("rejected", "Why this commit went ahead.")
    for subject in ("", "o" * 64):  # no governing decision, or another one than the record holds
        other = commit_landing_fact({"tree_sha": tree}, {"commit_sha": sha, "tree_sha": tree, "author_subject": subject},
                                    recorded_at="t", decision=decision)
        assert (other["author_subject"], other["author_disposition"], other["author_reason"]) == (subject, "", "")


def test_seventeen_real_commits_under_one_record_keep_the_first(candidate, monkeypatch):  # noqa: F811
    """Astra's reproduction (rl-20261029T140644): the 17th continuation used to push the first landing out."""
    ctx = candidate
    _mode(monkeypatch, "pro", "advisory")
    _prepare(ctx, monkeypatch)
    _install_reviewer(ctx, monkeypatch, PASSING)
    reference = _reference(git._commit_reviewed(ctx, "Fix amount", skip_advisory_review=True))
    commits = []
    for index in range(17):
        (ctx.repo_dir / "change.py").write_text(f"value = {index + 2}\n", encoding="utf-8")
        result = git._commit_reviewed(ctx, f"Amount {index}", review_reference=reference,
                                      author_disposition={"disposition": "accepted", "rationale": f"Decision {index}"})
        assert _head(ctx) not in commits, result
        commits.append(_head(ctx))
    record = _record(ctx, reference)
    assert [row["commit_sha"] for row in record["commit_landings"]] == commits
    assert [row["author_reason"] for row in record["commit_landings"]] == [f"Decision {index}" for index in range(17)]
    panel = ledger_record_panel(record)
    assert [row["commit_sha"] for row in panel["commit_landings"]] == commits[1:]
    assert panel["commit_landings_omitted"] == 1
    assert panel["source_ref"]["path"] == f"state/review_ledger/{record['record_id']}.json"


def test_each_landing_keeps_its_own_decision_through_projection_card_and_cold_reload(candidate, monkeypatch, tmp_path):  # noqa: F811
    """Adopt A after a rejection, then commit B (another decision, other code) under the same record: A keeps
    A's reason and attribution in the task detail AND in the history summary row a cold reload reads."""
    import asyncio
    import json
    import os
    import pathlib
    import shutil
    from types import SimpleNamespace

    from ouroboros import body_adoption
    from ouroboros.agent_task_pipeline import _store_task_result
    from ouroboros.gateway.history import make_chat_history_endpoint
    from ouroboros.outcomes import public_task_result
    from ouroboros.post_task_synthesis import _record_task_facts
    from ouroboros.task_results import STATUS_SCHEDULED, write_task_result
    from ouroboros.task_status import load_effective_task_result
    from supervisor import workers
    from tests.group_run import run_in_group

    ctx = candidate
    _mode(monkeypatch, "pro", "advisory")
    _prepare(ctx, monkeypatch)
    _install_reviewer(ctx, monkeypatch, REJECTING)
    reasons = {"a": "The sequence the reviewer describes cannot occur; commit A goes ahead.",
               "b": "Applied the reviewer's naming note after the review."}
    reference = _reference(git._commit_reviewed(ctx, "Fix amount", skip_advisory_review=True))
    git._commit_reviewed(ctx, "Fix amount", review_reference=reference,
                         author_disposition={"disposition": "rejected", "rationale": reasons["a"]})
    commit_a, parent_a = _head(ctx), git.run_cmd(["git", "rev-parse", "HEAD~1"], cwd=ctx.repo_dir).strip()
    (ctx.repo_dir / "change.py").write_text("value = 3\n", encoding="utf-8")
    git._commit_reviewed(ctx, "Follow-up amount", review_reference=reference,
                         author_disposition={"disposition": "accepted", "rationale": reasons["b"]})
    commit_b = _head(ctx)
    record = _record(ctx, reference)
    assert [(row["commit_sha"], row["reviewed_subject"], row["author_disposition"], row["author_reason"])
            for row in record["commit_landings"]] == [(commit_a, "same", "rejected", reasons["a"]),
                                                      (commit_b, "different", "accepted", reasons["b"])]
    assert record["author_decision"]["rationale"] == reasons["b"]  # the record's current decision is B's

    # The task ends: its stored Reviews projection is computed from its ledger records, as at any task end.
    task = {"id": ctx.task_id, "type": "task", "chat_id": 1}
    env = SimpleNamespace(drive_root=ctx.drive_root, repo_dir=ctx.repo_dir)
    write_task_result(ctx.drive_root, ctx.task_id, STATUS_SCHEDULED, chat_id=1, root_task_id=ctx.task_id,
                      result="Task accepted and scheduled.")
    _store_task_result(env, task, "Fixed the amount.", {"rounds": 1, "cost": 0}, {"tool_calls": []})
    _record_task_facts(env, task, {"rounds": 1}, {"tool_calls": []}, pathlib.Path(ctx.drive_root) / "logs")
    # The next boot adopted A: the real publisher keeps the fact on the task result.
    monkeypatch.setattr(workers, "get_event_q", lambda **_kw: SimpleNamespace(put=lambda _event: None))
    body_adoption._publish_on_task(ctx.drive_root, ctx.repo_dir, {
        "task_id": ctx.task_id, "adoption_id": "adoption-a", "outcome": "adopted", "candidate": commit_a,
        "previous": parent_a, "branch": "ouroboros"}, f"Adopted candidate commit {commit_a[:12]} on ouroboros.")

    detail = public_task_result(load_effective_task_result(ctx.drive_root, ctx.task_id))
    response = asyncio.run(make_chat_history_endpoint(ctx.drive_root)(SimpleNamespace(query_params={"chat_id": "1"})))
    [summary] = [row for row in json.loads(response.body)["messages"]
                 if row.get("system_type") == "task_summary" and row.get("task_id") == ctx.task_id][-1:]
    for source in (detail, summary):
        [panel] = [row for row in source["review_projection"]["panels"] if row.get("record_id") == record["record_id"]]
        assert [row["author_reason"] for row in panel["commit_landings"]] == [reasons["a"], reasons["b"]]
        assert source["body_adoptions"][0]["commit_sha"] == commit_a
    sources = tmp_path / "landing-sources.json"
    sources.write_text(json.dumps({"task_id": ctx.task_id, "detail": detail, "summary": summary, "adopted": commit_a,
                                   "adopted_reason": reasons["a"], "later": commit_b, "later_reason": reasons["b"]}),
                       encoding="utf-8")
    node = shutil.which("node")
    assert node, "Node is required for the real Reviews card consumer proof"
    proof = run_in_group([node, "--test", "--test-reporter=tap", "--test-name-pattern=two real commits",
                          "web/tests/review_landing_relation.test.js"], cwd=pathlib.Path(__file__).resolve().parents[1],
                         env={**os.environ, "OURO_TEST_LANDING_SOURCES": str(sources)}, timeout=60)
    assert proof.returncode == 0 and "# pass 1\n" in proof.stdout and "# fail 0\n" in proof.stdout, proof.stdout + proof.stderr
