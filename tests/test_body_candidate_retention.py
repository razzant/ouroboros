"""Quiescent candidate scratch is classified once, with addressed deletion rechecks; one pass's verdicts read the
kept-output debts once, and each removal reads them fresh."""
import json
from pathlib import Path

import pytest

from ouroboros import body_candidate as body, subagent_worktrees as wt, test_environment as environment
from ouroboros.task_results import write_task_result
from ouroboros.tools import kept_service_outputs as kept
from tests._shared import retire_synthetic_retention
from tests.body_candidate_support import isolate, make_ctx, make_serving


@pytest.fixture
def candidate(tmp_path, monkeypatch):
    serving, data = make_serving(tmp_path), isolate(monkeypatch, tmp_path)
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "pro")
    ctx = make_ctx(serving, data, "scratch-owner")
    body.prepare(ctx)
    env = body.process_environment(ctx, ctx.repo_dir)
    write_task_result(data, ctx.task_id, "completed")
    return ctx, data, Path(env["HOME"]).parent


def count_walks(monkeypatch):
    walked, walk = [], environment.os.walk

    def counted(root, **kwargs):
        walked.append(Path(root))
        yield from walk(root, **kwargs)

    monkeypatch.setattr(environment.os, "walk", counted)
    return walked


def large_tree(root):
    for number in range(200):
        leaf = root / "tmp" / f"branch-{number}" / "cache" / "objects"
        leaf.mkdir(parents=True)
        (leaf / "input").write_bytes(b"disposable")
    return leaf


def test_repeated_large_empty_tree_walks_once_then_deletes(candidate, monkeypatch):
    ctx, data, root = candidate
    large_tree(root)
    walks = count_walks(monkeypatch)
    with monkeypatch.context() as hold:
        # Failed unique-work capture keeps the checkout for repeated custody passes.
        hold.setattr(body, "unique_work", lambda row: {"readable": False, "error": "capture unavailable"})
        for _ in range(3):
            assert wt.prune_orphans(retention_days=0) == {"removed": 0, "kept": 1}
    observed = body.find(ctx.task_id, data)["scratch_retention"]["observation"]
    assert observed["retained"] is False and observed["marker"] == ""
    assert walks == [root]
    assert wt.prune_orphans(retention_days=0) == {"removed": 1, "kept": 0}
    assert walks == [root] and not root.exists() and not ctx.repo_dir.exists()


def test_nested_marker_reuses_walk_and_redecides_changed_or_vanished_path(candidate, monkeypatch):
    ctx, data, root = candidate
    first = large_tree(root) / environment.RETENTION_MARKER
    second = root / "tmp" / "branch-0" / environment.RETENTION_MARKER
    first.write_text("synthetic retention\n", encoding="utf-8")
    walks = count_walks(monkeypatch)
    try:
        for _ in range(3):
            assert wt.prune_orphans(retention_days=0) == {"removed": 0, "kept": 1}
        assert walks == [root]
        assert body.find(ctx.task_id, data)["scratch_retention"]["observation"]["marker"] == str(first.relative_to(root))
        first.write_text("changed synthetic retention\n", encoding="utf-8")
        assert wt.prune_orphans(retention_days=0)["removed"] == 0
        assert walks == [root, root]
        second.write_text("another synthetic retention\n", encoding="utf-8")
        first.unlink()
        assert wt.prune_orphans(retention_days=0)["removed"] == 0
        assert body.find(ctx.task_id, data)["scratch_retention"]["observation"]["marker"] == str(second.relative_to(root))
        assert walks == [root] * 3
        second.unlink()
        assert wt.prune_orphans(retention_days=0) == {"removed": 1, "kept": 0}
        assert walks == [root] * 4 and not root.exists()
    finally:
        first.unlink(missing_ok=True)
        second.unlink(missing_ok=True)


@pytest.mark.parametrize("reason", ["owner_not_terminal", "owner_processes_live"])
def test_owner_checks_precede_any_scratch_walk(candidate, monkeypatch, reason):
    ctx, data, root = candidate
    walks = count_walks(monkeypatch)
    monkeypatch.setattr(body, "_recorded_terminal", lambda *args: reason != "owner_not_terminal")
    monkeypatch.setattr(body, "live_lineage_processes", lambda *args: [-1])
    verdict = body.retention_verdict(body.find(ctx.task_id, data), expired=True, data_dir=data)
    assert not verdict["remove"] and verdict["reason"] == reason
    assert not walks and root.exists()


def test_direct_marker_is_addressed_without_walking(candidate, monkeypatch):
    ctx, data, root = candidate
    marker = root / environment.RETENTION_MARKER
    marker.write_text("synthetic direct retention\n", encoding="utf-8")
    walks = count_walks(monkeypatch)
    try:
        for _ in range(3):
            assert wt.prune_orphans(retention_days=0)["removed"] == 0
        assert not walks and body.find(ctx.task_id, data)
    finally:
        marker.unlink()


def test_unreadable_subtree_stays_retained_until_its_witness_changes(candidate, monkeypatch):
    ctx, data, root = candidate
    blocked = root / "tmp" / "unreadable"
    blocked.mkdir()
    walks = []

    def unreadable(target, *, onerror, **kwargs):
        walks.append(target)
        onerror(PermissionError(13, "synthetic unreadable subtree", str(blocked)))
        return iter(())

    with monkeypatch.context() as denied:
        denied.setattr(environment.os, "walk", unreadable)
        for _ in range(3):
            assert wt.prune_orphans(retention_days=0) == {"removed": 0, "kept": 1}
        assert walks == [root]
        assert body.find(ctx.task_id, data)["scratch_retention"]["observation"]["retained"]
    blocked.touch()  # the addressed directory changed after access was restored
    assert wt.prune_orphans(retention_days=0) == {"removed": 1, "kept": 0}
    assert not root.exists()


@pytest.mark.parametrize("declaration", ["missing", "directory", "symlink"])
def test_unclassified_legacy_environment_is_not_declared_or_deleted(candidate, monkeypatch, declaration):
    ctx, data, root = candidate
    marker = root / environment.SCRATCH_MARKER
    marker.unlink()
    if declaration == "directory":
        marker.mkdir()
    elif declaration == "symlink":
        target = root / "unclassified-declaration"
        target.write_text("body-process\n", encoding="utf-8")
        marker.symlink_to(target)
    sentinel = root / "legacy-source"
    sentinel.write_bytes(b"unknown provenance")
    body.process_environment(ctx, ctx.repo_dir)
    assert not marker.exists() if declaration == "missing" else marker.is_symlink() or marker.is_dir()
    walks = count_walks(monkeypatch)
    for _ in range(3):
        assert wt.prune_orphans(retention_days=0) == {"removed": 0, "kept": 1}
    body.discard_scratch(body.find(ctx.task_id, data))
    assert not walks and sentinel.read_bytes() == b"unknown provenance" and ctx.repo_dir.exists()


def test_late_nested_retention_invalidates_empty_observation_without_a_locked_walk(candidate, monkeypatch):
    ctx, data, root = candidate
    nested = root / "tmp" / "already-present" / "preflight"
    nested.mkdir(parents=True)
    walks = count_walks(monkeypatch)
    row = body.find(ctx.task_id, data)
    verdict = body.retention_verdict(row, expired=True, data_dir=data)
    assert verdict["remove"] and walks == [root]
    assert environment.retain_tree(nested, "synthetic late retention")
    try:
        assert not body.verdict_holds(row, verdict, data_dir=data)
        body.discard_scratch(row)
        assert root.exists() and walks == [root]
        for _ in range(2):
            assert wt.prune_orphans(retention_days=0)["removed"] == 0
        assert walks == [root, root]
    finally:
        retire_synthetic_retention(nested, "synthetic late retention")


def test_new_process_invalidates_quiescent_empty_observation(candidate, monkeypatch):
    ctx, data, root = candidate
    row = body.find(ctx.task_id, data)
    verdict = body.retention_verdict(row, expired=True, data_dir=data)
    assert verdict["remove"]
    body.process_environment(ctx, ctx.repo_dir)
    assert not body.verdict_holds(row, verdict, data_dir=data)
    walks = count_walks(monkeypatch)
    assert wt.prune_orphans(retention_days=0) == {"removed": 1, "kept": 0}
    assert walks == [root]


def test_ownership_change_during_classification_cannot_publish_or_delete(candidate, monkeypatch):
    ctx, data, root = candidate
    row = body.find(ctx.task_id, data)
    walk = environment.retention_markers

    def transfer(target):
        found = walk(target)
        body._update_row(row, lambda entry: entry.update(task_id="new-owner"), data)
        return found

    monkeypatch.setattr(environment, "retention_markers", transfer)
    assert wt.prune_orphans(retention_days=0) == {"removed": 0, "kept": 1}
    current = body.find("new-owner", data)
    assert current and "scratch_retention" not in current
    assert root.exists() and ctx.repo_dir.exists()


def finished_candidates(tmp_path, monkeypatch, count):
    serving, data = make_serving(tmp_path), isolate(monkeypatch, tmp_path)
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "pro")
    contexts = [make_ctx(serving, data, f"owner-{number}") for number in range(count)]
    for ctx in contexts:
        body.prepare(ctx)
        write_task_result(data, ctx.task_id, "completed")
    return data, contexts


def owe(data, task, cwd, **fields):
    launch = "launch-" + task.encode("utf-8").hex().ljust(32, "0")[:32]
    path = kept.declaration_path(data, task, launch)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"schema": 1, "launch_id": launch, "task_id": task, "cwd": str(cwd),
                                "outputs": ["report.txt"], **fields}), encoding="utf-8")
    return path


def count_debt_reads(monkeypatch):
    calls, real = [], kept.owed_declarations
    monkeypatch.setattr(kept, "owed_declarations", lambda root: calls.append(root) or real(root))
    return calls


def test_a_pass_of_held_candidates_reads_the_kept_debts_once(tmp_path, monkeypatch):
    """Every verdict shares the first one's read: three candidates held by their debts, however many passes repeat them."""
    data, contexts = finished_candidates(tmp_path, monkeypatch, 3)
    for ctx in contexts:
        owe(data, ctx.task_id, body.find(ctx.task_id, data)["path"])
    calls, reasons = count_debt_reads(monkeypatch), {}
    assert wt.prune_orphans(retention_days=0, reasons=reasons) == {"removed": 0, "kept": 3}
    assert sorted(reasons.values()) == ["kept_service_capture_owed"] * 3 and len(calls) == 1, (reasons, calls)


def test_each_removal_still_reads_the_debts_fresh(tmp_path, monkeypatch):
    """One read for the verdicts, then one per removal at its effect: the removal never trusts the pass's read."""
    data, _contexts = finished_candidates(tmp_path, monkeypatch, 3)
    owe(data, "elsewhere", tmp_path / "unrelated")
    calls = count_debt_reads(monkeypatch)
    assert wt.prune_orphans(retention_days=0) == {"removed": 3, "kept": 0}
    assert len(calls) == 1 + 3, calls


def test_a_debt_declared_after_the_verdicts_keeps_its_candidate(tmp_path, monkeypatch):
    """A kept service declaring outputs in a candidate once the pass has judged it: the effect's fresh read holds it."""
    data, contexts = finished_candidates(tmp_path, monkeypatch, 2)
    rows = {ctx.task_id: body.find(ctx.task_id, data) for ctx in contexts}
    judged, real = [], body.retention_verdict

    def judge(row, **kwargs):
        verdict = real(row, **kwargs)
        judged.append(row)
        if len(judged) == len(contexts):  # every verdict is in, none has been acted on
            owe(data, "late-producer", rows["owner-0"]["path"])
        return verdict

    monkeypatch.setattr(body, "retention_verdict", judge)
    reasons = {}
    assert wt.prune_orphans(retention_days=0, reasons=reasons) == {"removed": 1, "kept": 1}
    assert reasons == {rows["owner-0"]["path"]: "changed"} and Path(rows["owner-0"]["path"]).is_dir(), reasons
    assert not Path(rows["owner-1"]["path"]).exists()


def test_a_damaged_debt_stops_only_a_pass_whose_verdict_reaches_it(tmp_path, monkeypatch):
    """Read when first needed, as before the shared read: a candidate inside retention never asks, an expired one stops
    the pass at its own boundary with nothing deleted."""
    data, contexts = finished_candidates(tmp_path, monkeypatch, 1)
    owe(data, "damaged", tmp_path / "unrelated", outputs="not a list")
    path = body.find(contexts[0].task_id, data)["path"]
    assert wt.prune_orphans(retention_days=30)["removed"] == 0
    with pytest.raises(kept.KeptDeclarationUnreadable):
        wt.prune_orphans(retention_days=0)
    assert Path(path).is_dir() and body.find(contexts[0].task_id, data) is not None
