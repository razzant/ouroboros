"""Explicitly empty Git/skill selectors keep the bounded whole-capture contract."""

import json
import pathlib
from contextlib import contextmanager

import pytest

from ouroboros import delegate_custody as custody
from ouroboros.subagent_worktrees import find_execution_snapshot
from ouroboros.tools.registry import ToolRegistry
from tests.test_delegated_directory import DirectoryEngine, context, entry
from tests.test_delegated_run_isolation import TestCaptureAndIntegrate as _CaptureFixture
from tests.test_delegated_run_isolation import _git, _isolated_entry
from tests.test_delegated_run_isolation_orphans import _captured_flat, _disposer
from tests.test_delegated_skill_payload import _captured


pytestmark = pytest.mark.serial


@pytest.fixture(autouse=True)
def isolated_custody(monkeypatch):
    monkeypatch.setattr(custody, "_CUSTODY", {})
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "advanced")
    monkeypatch.setenv("OUROBOROS_SAFETY_MODE", "off")
    monkeypatch.setenv("OUROBOROS_CONSCIOUSNESS_SAFETY_MODE", "off")


def _call(ctx, run_id, decision="apply", **options):
    registry = ToolRegistry(repo_dir=ctx.repo_dir, drive_root=ctx.drive_root)
    registry.set_context(ctx)
    return registry.execute_result("integrate_delegated_patch", {
        "run_id": run_id, "decision": decision, "reason": "Inspected the complete capture.", **options})


@pytest.mark.parametrize("options", [{}, {"paths": []}])
@pytest.mark.parametrize("decision", ["apply", "reject"])
def test_git_whole_capture_uses_the_normal_disposition(tmp_path, monkeypatch, options, decision):
    from ouroboros.tools.delegate import _capture_terminal_patch

    target, ctx, handle = _CaptureFixture()._provisioned(tmp_path, monkeypatch)
    (pathlib.Path(handle.path) / "tracked.txt").write_text("accepted edit\n", encoding="utf-8")
    (pathlib.Path(handle.path) / "new.txt").write_text("new content\n", encoding="utf-8")
    held = _isolated_entry(ctx, target, handle)
    _capture_terminal_patch(ctx, held)
    before = (target / "tracked.txt").read_bytes()
    (target / "unrelated.txt").write_text("other work\n", encoding="utf-8")
    result = _call(ctx, held.run_id, decision, **options)
    assert result.status == "ok", result.text
    assert held.patch_disposed == ("applied" if decision == "apply" else "rejected")
    assert find_execution_snapshot(handle.snapshot_id) is None
    assert (target / "unrelated.txt").read_text(encoding="utf-8") == "other work\n"
    staged = _git(target, "diff", "--cached", "--name-only").stdout.splitlines()
    assert "unrelated.txt" not in staged
    if decision == "apply":
        assert (target / "tracked.txt").read_text(encoding="utf-8") == "accepted edit\n"
        assert (target / "new.txt").read_text(encoding="utf-8") == "new content\n"
        assert {"tracked.txt", "new.txt"} <= set(staged)
    else:
        assert (target / "tracked.txt").read_bytes() == before
        assert not (target / "new.txt").exists()
    assert ("paths=[] selects the complete captured result" in result.text) == bool(options)
    repeated = _call(ctx, held.run_id, decision, **options)
    assert "ALREADY_DISPOSED" in repeated.text
    assert _git(target, "diff", "--cached", "--name-only").stdout.splitlines() == staged


@pytest.mark.parametrize("paths", [["tracked.txt"], "", False, {}])
def test_git_selection_refusal_has_no_effects(tmp_path, monkeypatch, paths):
    from ouroboros.tools.delegate import _capture_terminal_patch

    target, ctx, handle = _CaptureFixture()._provisioned(tmp_path, monkeypatch)
    (pathlib.Path(handle.path) / "tracked.txt").write_text("child edit\n", encoding="utf-8")
    held = _isolated_entry(ctx, target, handle)
    _capture_terminal_patch(ctx, held)
    before = (target / "tracked.txt").read_bytes()
    staged = _git(target, "diff", "--cached").stdout
    result = _call(ctx, held.run_id, paths=paths)
    assert result.status == "error", result.text
    assert (target / "tracked.txt").read_bytes() == before
    assert _git(target, "diff", "--cached").stdout == staged
    assert not held.patch_disposed and not held.patch_apply_pending
    assert find_execution_snapshot(handle.snapshot_id) is not None


def test_empty_selector_applies_terminal_owner_orphan(tmp_path, monkeypatch):
    target, _, held, _ = _captured_flat(tmp_path, monkeypatch)
    ctx = _disposer(tmp_path, monkeypatch, target, target)
    result = _call(ctx, held.run_id, paths=[])
    assert "Integrated" in result.text and "orphan of terminal task" in result.text
    assert "CHILD-EDIT" in (target / "tracked.txt").read_text(encoding="utf-8")
    rows = list(custody._iter_rows(custody.event_log_path(custody.custody_root(ctx))))
    disposed = [row for row in rows if row["type"] == custody.PATCH_DISPOSED]
    assert len(disposed) == 1 and disposed[0]["disposed_by_task_id"] == ctx.task_id


@pytest.mark.parametrize("drift", [False, True])
def test_empty_skill_selector_preserves_payload_cas_and_review_staleness(tmp_path, monkeypatch, drift):
    ctx, skill, handle, held, _ = _captured(tmp_path, monkeypatch)
    if drift:
        (skill / "notes.txt").write_text("new parent work\n", encoding="utf-8")
    result = _call(ctx, held.run_id, paths=[])
    if drift:
        assert "INTEGRATE_CONFLICT" in result.text
        assert (skill / "notes.txt").read_text(encoding="utf-8") == "new parent work\n"
        assert not (skill / "extra.txt").exists() and not held.patch_disposed
        assert find_execution_snapshot(handle.snapshot_id) is not None
        return
    assert "Integrated" in result.text and "STALE" in result.text, result.text
    assert (skill / "notes.txt").read_text(encoding="utf-8") == "DONE\n"
    assert (skill / "extra.txt").read_bytes() == b"nul\0ok\n"
    assert not (skill / ".git").exists()
    assert held.patch_disposed == "applied" and find_execution_snapshot(handle.snapshot_id) is None


@pytest.mark.parametrize("options,expected", [({}, "applied"), ({"paths": []}, "refused"),
                                               ({"paths": ["out.bin"]}, "partially_applied")])
def test_directory_selector_keeps_its_engine_contract(tmp_path, monkeypatch, options, expected):
    from ouroboros import claudexor_daemon

    ctx, target = context(tmp_path, monkeypatch)
    engine, held = DirectoryEngine(target), entry(ctx, target, "copy")
    custody._CUSTODY[held.run_id] = held
    from ouroboros.delegate_directory import capture_directory_result

    capture_directory_result(ctx.drive_root, held, engine)  # the body's capture, before the decision (#1969)
    requests = []
    apply_run, get_run = engine.apply_run, engine.get_run

    def apply(rid, request, **kwargs):
        requests.append(request)
        return apply_run(rid, request, **kwargs)

    def detail(rid):
        value = get_run(rid)
        if options.get("paths"):
            value["summary"]["result"]["applyState"] = "not_applied"
        return value

    @contextmanager
    def gateway():
        yield engine

    monkeypatch.setattr(engine, "apply_run", apply)
    monkeypatch.setattr(engine, "get_run", detail)
    monkeypatch.setattr(claudexor_daemon, "read_owned_gateway", gateway)
    result = _call(ctx, held.run_id, **options)
    if expected == "refused":
        assert "paths must be a nonempty list" in result.text
        assert not requests and not engine.decisions and not (target / "out.bin").exists()
    else:
        assert json.loads(result.text)["status"] == expected
        assert len(requests) == 1
        assert requests[0].get("paths") == options.get("paths")
        assert (target / "out.bin").read_bytes() == engine.body
    assert bool(held.patch_disposed) == (expected == "applied")
