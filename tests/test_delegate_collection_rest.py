"""Terminal debt rests until an addressed request or a new evidence generation."""
from __future__ import annotations

import errno
import json
import pathlib
import shutil

import pytest

from ouroboros import delegate_collection as body, delegate_custody as custody
from ouroboros import delegate_custody_current as current, usage_accounting
from ouroboros.gateways.claudexor import ClaudexorUnavailable
from ouroboros.tools import delegate
from ouroboros.tools.delegate_terminal_evidence import collect_and_deliver_terminal
from ouroboros.tools.registry import ToolContext
from tests._delegate_capture_child import capture_child  # noqa: F401
from tests.test_delegate_body_collection import started, terminal
from tests.test_delegate_collection_recovery import driven_runtime  # noqa: F401
from tests.test_recovered_capture import prepared


def _prepared(root):
    """An ordinary run's real private copy, its named evidence written, beside ``root``."""
    material = root.with_name(root.name + "-material")
    material.mkdir()
    return prepared(material, root, "ordinary")


@pytest.mark.parametrize("debt", ["money", "output", "gone", "over_cap", "money_and_over_cap", "daemon_down"])
def test_stuck_terminal_debt_does_no_io_after_first_attempt(
        tmp_path, monkeypatch, driven_runtime, debt):  # noqa: F811
    runtime, clock, tick = driven_runtime
    entry = started(tmp_path)
    detail = terminal(text="x" * (256 * 1024) if debt == "over_cap" else "preview")
    if debt != "money":
        detail["primaryOutput"].update(truncated=True, bytes=6 * 1024 * 1024, path="output/primary.md")
    calls, broken = [], [True]

    def money(*args, **kwargs):
        calls.append("ledger")
        if debt in {"money", "money_and_over_cap"} and broken[0]:
            raise OSError(errno.EBUSY, "ledger busy")
        return "session"

    class Engine:
        def get_run(self, rid, **kwargs):
            calls.append("run")
            return detail

        def get_run_artifact(self, rid, path):
            calls.append("artifact")
            if broken[0]:
                status = {"gone": 410, "over_cap": 413, "money_and_over_cap": 413}.get(debt, 503)
                error = ClaudexorUnavailable("artifact_unavailable", "same diagnostic", status_code=status)
                error.problem = {"retryable": True}  # 410/413 remain permanent despite a generic retry hint.
                raise error
            return b"complete answer"

        def close(self):
            pass

    def gateway():
        calls.append("connect")
        if debt == "daemon_down" and broken[0]:
            raise ClaudexorUnavailable("daemon_not_discovered", "descriptor absent")
        return Engine()

    runtime.gateway_factory = gateway
    monkeypatch.setattr(usage_accounting, "record_subscription_session", money)
    body.observe_run(tmp_path, entry, detail)
    tick()
    first = current.run(tmp_path, entry.run_id)
    assert current.collection_pending(first)
    baseline = list(calls)
    reads, read_bytes = [], pathlib.Path.read_bytes

    def counted(path):
        raw = read_bytes(path)
        reads.append(len(raw))
        return raw

    monkeypatch.setattr(pathlib.Path, "read_bytes", counted)
    for moment in range(20):
        clock[0] = (moment + 1) * 3000
        try:
            tick()
        except ClaudexorUnavailable:
            pass  # Count attempted connections as well as successful engine reads.
    assert calls == baseline, "unchanged debt must not connect, observe, fetch or write money per tick"
    assert reads == [], "a resting tick must not reread terminal/output history"
    permanent = debt in {"gone", "over_cap"}
    assert (first.collection_attempt["retry"] == "transient") is not permanent

    # A generation is a fact, and permanent output refusals still need changed input.
    monkeypatch.setattr(runtime, "runtime_id", "next-body-generation")
    tick()
    if permanent:
        assert calls == baseline
    elif debt in {"money", "money_and_over_cap"}:
        assert calls == baseline + ["ledger"], "money-only recovery never opens an engine"
    else:
        assert calls.count("connect") == baseline.count("connect") + 1
        assert calls.count("run") == baseline.count("run") + int(debt != "daemon_down")
    after_generation = list(calls)
    tick()
    assert calls == after_generation

    if permanent:
        ctx = ToolContext(repo_dir=tmp_path, drive_root=tmp_path, task_id=entry.task_id)
        result = collect_and_deliver_terminal(ctx, entry.run_id, detail, body._shape(first), first, Engine())
        assert not result["output_delivery"]["complete"]
        assert calls.count("artifact") == after_generation.count("artifact") + 1

    # An explicit wait can recover even the same terminal's formerly unavailable bytes.
    broken[0] = False
    detail["primaryOutput"].update(bytes=len(b"complete answer"))
    ctx = ToolContext(repo_dir=tmp_path, drive_root=tmp_path, task_id=entry.task_id)
    result = collect_and_deliver_terminal(ctx, entry.run_id, detail, body._shape(first), first, Engine())
    assert result["settlement"]["settled"] and result["output_delivery"]["complete"]
    assert not current.collection_pending(current.run(tmp_path, entry.run_id))
    after_wait = list(calls)
    tick()
    assert calls == after_wait


@pytest.mark.parametrize("terminal_source", [False, True, "arrives_during_read"])
def test_absence_fault_requires_an_unknown_terminal_outcome(
        tmp_path, driven_runtime, terminal_source):  # noqa: F811
    runtime, _, _ = driven_runtime
    entry = started(tmp_path, daemon_store_id="original-store")
    if terminal_source is True:
        body.observe_run(tmp_path, entry, terminal())
        entry = current.run(tmp_path, entry.run_id)

    class Engine:
        store_identity = "unproven-store"

        def get_run(self, rid, **kwargs):
            if terminal_source == "arrives_during_read":
                body.observe_run(tmp_path, entry, terminal())
            raise ClaudexorUnavailable("run_not_found", "no such run", status_code=404)

        def close(self):
            pass

    runtime.gateway_factory = Engine
    try:
        runtime._observe(entry)
    except ClaudexorUnavailable:
        pass
    fresh = current.run(tmp_path, entry.run_id)
    assert bool(fresh.containment_fault) is (terminal_source is False)
    if terminal_source:
        assert fresh.terminal_state == "succeeded" and fresh.terminal_source
    else:
        assert fresh.containment_fault["reason"] == "run_absence_identity_unproven"


@pytest.mark.parametrize("surface", ["delivery", "wait", "supervised_wait"])
def test_explicit_money_waits_use_the_live_capture_generation(
        tmp_path, monkeypatch, driven_runtime, surface):  # noqa: F811
    from ouroboros import delegate_supervision

    runtime, _, tick = driven_runtime
    entry, _, _ = _prepared(tmp_path)
    ctx = ToolContext(repo_dir=tmp_path, drive_root=tmp_path, task_id=entry.task_id)
    copies, ledger = [], []

    def money(*args, **kwargs):
        ledger.append(True)
        raise OSError(errno.EBUSY, "ledger busy")

    copy = shutil.copyfileobj

    def busy(src, dst, *args):
        if ".evidence-" in str(getattr(dst, "name", "")):
            copies.append(True)
            raise OSError(errno.EBUSY, "evidence busy")
        return copy(src, dst, *args)

    monkeypatch.setattr(usage_accounting, "record_subscription_session", money)
    monkeypatch.setattr(shutil, "copyfileobj", busy)
    body.observe_run(tmp_path, entry, terminal(entry.run_id))
    tick()
    tick()
    first = current.run(tmp_path, entry.run_id)
    assert copies == [True] and first.collection_attempt["generation"] == runtime.runtime_id
    home = custody.delegated_capture_dir(tmp_path, entry.task_id, custody.capture_key(entry))
    generations = sorted(home.glob("evidence*"))
    ledger.clear()
    for _ in range(3):
        if surface == "delivery":
            collect_and_deliver_terminal(ctx, entry.run_id, terminal(entry.run_id), body._shape(first), first, None)
        elif surface == "wait":
            delegate._delegate_wait(ctx, entry.run_id)
        else:
            result = json.loads(delegate_supervision.supervised_wait(ctx, entry.run_id).text)
            assert delegate_supervision.acknowledge_pending_wake(ctx, result)
        tick()
        tick()
    assert ledger == [True] * 3, "each explicit wait retries only the owed money"
    assert copies == [True] and sorted(home.glob("evidence*")) == generations
    assert current.run(tmp_path, entry.run_id).collection_attempt == first.collection_attempt
    # Explicit recapture remains a real request in the same body.
    runtime._collect(entry.run_id, recapture=True)
    tick()
    assert copies == [True, True]


@pytest.mark.parametrize("damage", ["missing", "corrupt"])
def test_unreadable_money_source_keeps_retained_delivery(tmp_path, monkeypatch, damage):
    from ouroboros.artifacts import task_artifact_dir_path

    entry = started(tmp_path)
    ctx = ToolContext(repo_dir=tmp_path, drive_root=tmp_path, task_id=entry.task_id)

    def money(*args, **kwargs):
        raise OSError(errno.EBUSY, "ledger busy")

    monkeypatch.setattr(usage_accounting, "record_subscription_session", money)
    body.collect_terminal(tmp_path, entry.run_id, detail=terminal())
    fresh = current.run(tmp_path, entry.run_id)
    source = task_artifact_dir_path(tmp_path, entry.task_id) / fresh.terminal_source["path"]
    raw = source.read_bytes()
    if damage == "missing":
        source.unlink()
    else:
        source.write_bytes(b"invalid source")
    result = json.loads(delegate._delegate_wait(ctx, entry.run_id))
    assert result["primary_output"]["text"] == "complete answer" and result["output_delivery"]["complete"]
    assert not result["settlement"]["settled"]
    assert current.run(tmp_path, entry.run_id).output_source == fresh.output_source
    source.write_bytes(raw)
    monkeypatch.setattr(usage_accounting, "record_subscription_session", lambda *a, **k: "session")
    assert json.loads(delegate._delegate_wait(ctx, entry.run_id))["settlement"]["settled"]


def test_failed_evidence_cleanup_preserves_prepared_patch(tmp_path, monkeypatch, driven_runtime):  # noqa: F811
    runtime, _, tick = driven_runtime
    entry, _, _ = _prepared(tmp_path)
    copy, remove = shutil.copyfileobj, shutil.rmtree
    cleanup = []

    def busy(src, dst, *args):
        if ".evidence-" in str(getattr(dst, "name", "")):
            dst.write(src.read(7))
            raise OSError(errno.EBUSY, "copy interrupted")
        return copy(src, dst, *args)

    def held(path, ignore_errors=False, *args, **kwargs):
        if ".evidence-" in str(path) and pathlib.Path(path).name == "files":
            cleanup.append(ignore_errors)
            if not ignore_errors:
                raise PermissionError(errno.EACCES, "Windows handle prevents cleanup")
            return None
        return remove(path, ignore_errors, *args, **kwargs)

    monkeypatch.setattr(shutil, "copyfileobj", busy)
    monkeypatch.setattr(shutil, "rmtree", held)
    body.observe_run(tmp_path, entry, terminal(entry.run_id))
    tick()
    tick()
    fresh = current.run(tmp_path, entry.run_id)
    assert cleanup and fresh.patch_captured and fresh.named_evidence["status"] == "capture_failed"
    assert fresh.named_evidence["retry"] == "transient" and current.collection_pending(fresh)
    home = custody.delegated_capture_dir(tmp_path, entry.task_id, custody.capture_key(entry))
    assert [p.name for p in (home / fresh.named_evidence["directory"]).iterdir()] == ["summary.json"]
    assert list(home.glob(".evidence-*.pending")), "busy unpublished bytes keep their existing remover"
    monkeypatch.setattr(shutil, "copyfileobj", copy)
    monkeypatch.setattr(shutil, "rmtree", remove)
    runtime._collect(entry.run_id, recapture=True)
    tick()
    fresh = current.run(tmp_path, entry.run_id)
    assert fresh.patch_captured and fresh.named_evidence["status"] == "retained"
    assert not current.collection_pending(fresh)
    assert not list(home.glob(".evidence-*.pending"))
