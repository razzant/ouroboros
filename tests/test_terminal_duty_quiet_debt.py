"""#1569: a missing result keeps its exact projection debt, and a stable failure stays quiet.

Absence is a precondition, not an ownership verdict: the duty stats the canonical
result before the ownership read, retains the exact revision, and projects a body
that lands later without any new ledger write. Diagnostics are one id->reason map
per subject: only a change publishes, only a reason new to the subject logs a
traceback, and an unwritable event log retries without repeating a stack.
"""
import errno
import json
import logging
import os
import pathlib

import pytest

from ouroboros import owner_pause, task_results, usage_store
from ouroboros import terminal_cost_reconciliation as duty
from ouroboros import usage_accounting as usage
from ouroboros.terminal_cost_reconciliation import reconcile_abandoned_usage
from supervisor import events_task_done as done
from tests._usage_store_testing import ledger_rows
from tests.test_terminal_cost_reconciliation import attempt, dirty, env as env, task  # noqa: F401 - fixture

DUTY = "ouroboros.terminal_cost_reconciliation"


@pytest.fixture(autouse=True)
def fresh_process(monkeypatch, caplog):
    monkeypatch.setattr(duty, "_LAST_UNRESOLVED", {})
    caplog.set_level(logging.DEBUG, logger=DUTY)


def child(env, tid="child", **fields):
    return task(env, tid, parent_task_id="root", root_task_id="root", delegation_role="subagent", **fields)


def published(env, subject):
    path = env.root / "logs" / "supervisor.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()] if path.exists() else []
    return [row for row in rows if row.get("type") == "duty_unresolved" and row.get("subject") == subject]


def duty_records(caplog):
    return [record for record in caplog.records if record.name == DUTY]


def finalized(env, tid):
    path = env.root / "logs" / "events.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()] if path.exists() else []
    return [row for row in rows if row.get("type") == "task_cost_finalized" and row.get("task_id") == tid]


def quarantine(env, tid):
    """Move the canonical body aside the way schema admission does."""
    path = task_results.task_result_path(env.root, tid)
    target = path.parent / "quarantine" / path.name
    target.parent.mkdir(exist_ok=True)
    os.replace(path, target)
    return target


@pytest.mark.parametrize("missing", ["absent", "quarantined"])
def test_missing_result_keeps_its_exact_revision_without_fence_traversal(env, monkeypatch, caplog, missing):
    task(env)
    child(env)
    attempt(env, "child", logical="root")
    if missing == "quarantined":
        quarantine(env, "child")
    else:
        task_results.task_result_path(env.root, "child").unlink()
    before = dirty(env)["child"]
    fences, globs, glob = [], [], pathlib.Path.glob
    read_fence = owner_pause.read_fence
    monkeypatch.setattr(owner_pause, "read_fence", lambda root, tid: fences.append(tid) or read_fence(root, tid))
    monkeypatch.setattr(pathlib.Path, "glob", lambda self, pattern: globs.append(pattern) or glob(self, pattern))

    for _ in range(3):
        reconcile_abandoned_usage(env.root)
        assert dirty(env) == {"child": before}  # The root projected; the child keeps its exact revision.

    assert "child" not in fences and not [pattern for pattern in globs if pattern.startswith("child")]
    assert duty_records(caplog) == []  # Expected absence: no traceback, and no per-owner line at any level.
    rows = published(env, duty.COST_PROJECTIONS)
    assert [(row["count"], row["by_basis"], row["task_ids"], row["reasons"]) for row in rows] == [
        (1, {"result_absent": 1}, ["child"], {"child": "result_absent"})]
    # Pause authority is untouched: the quarantined identity still refuses an unknown fence.
    if missing == "quarantined":
        monkeypatch.setattr(owner_pause, "read_fence", read_fence)
        with pytest.raises(ValueError, match="owner_pause_authority_missing"):
            owner_pause.read_fence(env.root, "child")
        from supervisor.task_ownership import TaskOwnershipRead
        with pytest.raises(ValueError, match="owner_pause_authority_missing"):
            TaskOwnershipRead(env.root).load("child")


@pytest.mark.parametrize("restart", [False, True])
@pytest.mark.parametrize("arrival", ["child_copyback", "quarantine_restore"])
def test_a_body_landing_after_the_last_ledger_write_is_projected_without_a_receipt(env, restart, arrival):
    from ouroboros.headless import copy_child_task_result

    task(env)
    child_drive = env.root / "child-drive"
    if arrival == "child_copyback":
        child_drive.mkdir()
        # The child's own body carries its stale local cost; the canonical body does not exist yet.
        task_results.write_task_result(child_drive, "child", "completed", result="child answer",
                                       parent_task_id="root", root_task_id="root", delegation_role="subagent",
                                       accounted_upper_bound_usd=0.01, cost_final=True)
    else:
        child(env)
    attempt(env, "child", logical="root", cost=0.4)
    revision = dirty(env)["child"]
    parked = quarantine(env, "child") if arrival == "quarantine_restore" else None

    reconcile_abandoned_usage(env.root)
    assert dirty(env) == {"child": revision}
    ledger_before = ledger_rows(env.root)

    if arrival == "child_copyback":
        copy_child_task_result(env.root, {"id": "child", "drive_root": str(child_drive)})
        assert task_results.load_task_result(env.root, "child")["accounted_upper_bound_usd"] == 0.01
    else:
        os.replace(parked, task_results.task_result_path(env.root, "child"))
    assert ledger_rows(env.root) == ledger_before and dirty(env) == {"child": revision}  # No new receipt.
    if restart:
        duty._LAST_UNRESOLVED.clear()

    reconcile_abandoned_usage(env.root)
    stored = task_results.load_task_result(env.root, "child")
    assert stored["accounted_upper_bound_usd"] == 0.4 and stored["cost_final"] is True
    assert dirty(env) == {} and len(finalized(env, "child")) == 1
    reconcile_abandoned_usage(env.root)
    assert len(finalized(env, "child")) == 1 and ledger_rows(env.root) == ledger_before
    counts = [row["count"] for row in published(env, duty.COST_PROJECTIONS)]
    # A restarted process starts with nothing published, so the empty set says nothing new.
    assert counts == ([1] if restart else [1, 0])


def test_a_late_price_before_the_body_is_projected_when_the_body_lands(env):
    task(env)
    reservation = attempt(env, "child", logical="root", final=False)
    usage.mark_unresolved(reservation, "response missing")
    reconcile_abandoned_usage(env.root)
    first = dirty(env)["child"]

    usage.settle_attempt(reservation, cost_usd=0.3, cost_final=True)
    reconcile_abandoned_usage(env.root)
    assert dirty(env)["child"] > first  # Still absent: the newer revision is the one retained.
    assert task_results.load_task_result(env.root, "root")["accounted_upper_bound_usd_with_children"] == 0.3

    child(env)
    reconcile_abandoned_usage(env.root)
    stored = task_results.load_task_result(env.root, "child")
    assert stored["accounted_upper_bound_usd"] == 0.3 and stored["cost_final"] is True
    assert dirty(env) == {}


def test_stable_projection_failures_are_quiet_and_every_change_is_published(env, monkeypatch, caplog):
    for tid in ("a", "b", "c"):
        task(env, tid)
        attempt(env, tid)
    failing = {"a": errno.EACCES, "b": errno.ENOSPC}
    real = done._refresh_terminal_task_cost

    def refresh(root, tid, **kwargs):
        if tid in failing:  # The message carries a per-owner path; the reason must not.
            raise OSError(failing[tid], os.strerror(failing[tid]), str(root / tid))
        return real(root, tid, **kwargs)

    monkeypatch.setattr(done, "_refresh_terminal_task_cost", refresh)

    def step(**change):
        failing.update(change)
        for tid in [tid for tid, code in failing.items() if code is None]:
            del failing[tid]
        caplog.clear()
        before = len(published(env, duty.COST_PROJECTIONS))
        reconcile_abandoned_usage(env.root)
        stacks = sorted(record.getMessage().split(" with ")[1].split(" ")[0]
                        for record in duty_records(caplog) if record.exc_info)
        assert all(record.exc_info for record in duty_records(caplog))  # No line without news.
        rows = published(env, duty.COST_PROJECTIONS)[before:]
        return stacks, [row["reasons"] for row in rows]

    assert step() == (["OSError:ENOSPC", "PermissionError:EACCES"],
                      [{"a": "PermissionError:EACCES", "b": "OSError:ENOSPC"}])
    assert step() == ([], [])
    assert step() == ([], [])
    # The same ids, count and reason counts with the reasons swapped is still a change.
    assert step(a=errno.ENOSPC, b=errno.EACCES) == ([], [{"a": "OSError:ENOSPC", "b": "PermissionError:EACCES"}])
    assert step(b=errno.EIO) == (["OSError:EIO"], [{"a": "OSError:ENOSPC", "b": "OSError:EIO"}])
    assert step(a=None) == ([], [{"b": "OSError:EIO"}])  # Projected and acknowledged.
    attempt(env, "a")  # A new receipt re-opens the debt; the old reason recurs as news.
    assert step(a=errno.ENOSPC) == (["OSError:ENOSPC"], [{"a": "OSError:ENOSPC", "b": "OSError:EIO"}])
    assert step(a=None, b=None) == ([], [{}])
    assert dirty(env) == {}


def test_a_skipped_owner_proves_no_recovery(env, monkeypatch, caplog):
    task(env, "a")
    attempt(env, "a")
    monkeypatch.setattr(done, "_refresh_terminal_task_cost",
                        lambda *a, **k: (_ for _ in ()).throw(PermissionError(errno.EACCES, "denied")))
    reconcile_abandoned_usage(env.root)
    assert [row["count"] for row in published(env, duty.COST_PROJECTIONS)] == [1]
    caplog.clear()
    env.live.add("a")  # Not examined this pass: its last reason stands.
    reconcile_abandoned_usage(env.root)
    env.live.clear()
    reconcile_abandoned_usage(env.root)
    assert duty_records(caplog) == [] and [row["count"] for row in published(env, duty.COST_PROJECTIONS)] == [1]


def test_an_unwritable_event_log_retries_publication_without_a_stack_storm(env, monkeypatch, caplog):
    task(env, "a")
    attempt(env, "a")
    monkeypatch.setattr(done, "_refresh_terminal_task_cost",
                        lambda *a, **k: (_ for _ in ()).throw(PermissionError(errno.EACCES, "denied")))
    sink = env.root / "logs" / "supervisor.jsonl"
    sink.unlink(missing_ok=True)
    sink.mkdir(parents=True)  # Any append now fails.
    for _ in range(4):
        reconcile_abandoned_usage(env.root)
    messages = [record.getMessage() for record in duty_records(caplog)]
    assert len(messages) == 2 and all(record.exc_info for record in duty_records(caplog))
    assert "1 unresolved with PermissionError:EACCES" in messages[0]
    assert messages[1] == "Unresolved cost_projection observations could not be published"
    caplog.clear()
    sink.rmdir()
    reconcile_abandoned_usage(env.root)  # The unchanged observation is published once storage works.
    reconcile_abandoned_usage(env.root)
    assert duty_records(caplog) == []
    assert [row["reasons"] for row in published(env, duty.COST_PROJECTIONS)] == [{"a": "PermissionError:EACCES"}]


def test_open_attempts_and_projections_keep_separate_quiet_summaries(env, monkeypatch, caplog):
    from tests.test_usage_abandoned_reconciliation import _attempt

    child(env)
    rows = [_attempt(env, provider="claudexor") for _ in range(3)]

    def recover(*_a, **_kw):
        raise RuntimeError("probe failed")

    monkeypatch.setattr("ouroboros.llm_claudexor.recover_model_attempt", recover)
    for _ in range(3):
        reconcile_abandoned_usage(env.root)
    stacks = [record for record in duty_records(caplog) if record.exc_info]
    assert len(duty_records(caplog)) == len(stacks) == 1
    assert "attempt: 3 unresolved with recovery_failed:RuntimeError" in stacks[0].getMessage()
    attempts = published(env, duty.ATTEMPTS)
    assert [(row["count"], sorted(row["attempt_ids"])) for row in attempts] == [
        (3, sorted(row.attempt_id for row in rows))]
    # The root has no result yet: its projection debt is the other subject's one row.
    assert [row["by_basis"] for row in published(env, duty.COST_PROJECTIONS)] == [{"result_absent": 1}]
    # Live ownership takes the attempts out of this pass; their last reason stands, unannounced.
    caplog.clear()
    env.live.add("child")
    reconcile_abandoned_usage(env.root)
    assert duty_records(caplog) == [] and len(published(env, duty.ATTEMPTS)) == 1


@pytest.mark.parametrize("unproven", ["live_after_probe", "release_race_lost", "terminalize_race_lost"])
def test_an_attempt_outcome_this_pass_did_not_prove_keeps_its_last_reason(env, monkeypatch, caplog, unproven):
    from tests.test_usage_abandoned_reconciliation import _attempt

    child(env)
    native = unproven == "terminalize_race_lost"
    reservation = _attempt(env, provider="openai" if native else "claudexor")
    terminalize, plan = usage.terminalize_abandoned_attempt, []

    def recover(root, row, **_kw):
        step = plan.pop(0)
        if step == "fail":
            raise RuntimeError("probe failed")
        if step == "turns_live":
            env.live.add("child")  # Eligible before the probe, live at the check after it.
        return ("released", {}, None, False) if step == "release" else ("settled", {"prompt_tokens": 3}, 0.4, True)

    def terminalized(*args, **kwargs):
        step = plan.pop(0)
        if step == "fail":
            raise RuntimeError("terminalize failed")
        return "unresolved" if step == "lost" else terminalize(*args, **kwargs)

    monkeypatch.setattr("ouroboros.llm_claudexor.recover_model_attempt", recover)
    monkeypatch.setattr(usage, "terminalize_abandoned_attempt", terminalized)
    if unproven == "release_race_lost":
        monkeypatch.setattr("ouroboros.transport_custody.release_pre_dispatch_attempt", lambda *a, **k: False)
    plan.extend({"live_after_probe": ["fail", "turns_live", "settle"],
                 "release_race_lost": ["fail", "release", "settle"],
                 "terminalize_race_lost": ["fail", "lost", "real"]}[unproven])

    reconcile_abandoned_usage(env.root)
    first = "recovery_failed:RuntimeError"
    assert [row["reasons"] for row in published(env, duty.ATTEMPTS)] == [{reservation.attempt_id: first}]
    caplog.clear()
    reconcile_abandoned_usage(env.root)  # Nothing proven: no recovery row, no line at any level.
    assert duty_records(caplog) == [] and len(published(env, duty.ATTEMPTS)) == 1
    assert duty._LAST_UNRESOLVED[(str(env.root.resolve()), duty.ATTEMPTS)]["observed"] == {
        reservation.attempt_id: first}
    env.live.clear()
    reconcile_abandoned_usage(env.root)  # A real settlement is the proof that clears it.
    assert [row["count"] for row in published(env, duty.ATTEMPTS)] == [1, 0] and not plan


def test_a_failing_gateway_close_is_logged_once_per_change(env, monkeypatch, caplog):
    from types import SimpleNamespace

    from tests.test_usage_abandoned_reconciliation import _attempt

    child(env)
    _attempt(env, provider="claudexor")
    closes = []

    def close():
        error = closes.pop(0)
        if error:
            raise error

    monkeypatch.setattr("ouroboros.claudexor_daemon.read_owned_gateway", lambda: SimpleNamespace(close=close))
    monkeypatch.setattr("ouroboros.llm_claudexor.recover_model_attempt",
                        lambda root, row, gateway_factory: gateway_factory() and None)
    eio, eacces = OSError(errno.EIO, "Input/output error"), PermissionError(errno.EACCES, "denied")
    closes.extend([eio, eio, eio, eacces, eacces, None, eio])
    for _ in range(7):
        reconcile_abandoned_usage(env.root)
    lines = [record.getMessage() for record in duty_records(caplog)]
    assert lines == ["Usage recovery gateway close failed (OSError:EIO)",
                     "Usage recovery gateway close failed (PermissionError:EACCES)",
                     "Usage recovery gateway close failed (OSError:EIO)"]
    assert all(record.exc_info for record in duty_records(caplog)) and not closes


@pytest.mark.parametrize("scale", [(644, 367, 5)])
def test_a_thousand_unprojectable_owners_cost_one_stat_each_and_no_repeated_bytes(env, monkeypatch, caplog, scale):
    """The live install's #1569 shape: 1016 dirty owners, 644 with quarantined results,
    367 with none and 5 present. Present owners are read and projected as before."""
    quarantined, absent, present = scale
    owners = [(f"q{index:04}", 1) for index in range(quarantined)]
    owners += [(f"m{index:04}", 1) for index in range(absent)]
    owners += [(f"p{index}", 1) for index in range(present)]
    box = env.root / "task_results" / "quarantine"
    box.mkdir(parents=True)
    for tid, _ in owners[:quarantined]:
        (box / f"{tid}.json").write_text("{", encoding="utf-8")
    for tid, _ in owners[-present:]:
        task(env, tid)
    monkeypatch.setattr(usage_store.Txn, "dirty_owners", lambda self: list(owners))
    counts = {"stat": 0, "loads": 0, "fence": 0, "glob": 0}  # glob: of the quarantine directory
    stat, load, fence, glob = pathlib.Path.stat, task_results.load_task_result, owner_pause.read_fence, pathlib.Path.glob

    def counted(name, real):
        def call(*a, **k):
            counts[name] += 1
            return real(*a, **k)
        return call

    monkeypatch.setattr(pathlib.Path, "stat", lambda self, *a, **k: (
        counts.__setitem__("stat", counts["stat"] + (self.parent.name == "task_results")) or stat(self, *a, **k)))
    monkeypatch.setattr(task_results, "load_task_result", counted("loads", load))
    monkeypatch.setattr(owner_pause, "read_fence", counted("fence", fence))
    monkeypatch.setattr(pathlib.Path, "glob", lambda self, pattern: (
        counts.__setitem__("glob", counts["glob"] + (self == box)) or glob(self, pattern)))
    summary = usage.usage_projection(env.root)

    reconcile_abandoned_usage(env.root)
    first = dict(counts)
    sink = env.root / "logs" / "supervisor.jsonl"
    size = sink.stat().st_size
    assert first["fence"] == 0 and first["glob"] == 0
    # One presence stat per owner; present owners add their ownership stamps and projection writes.
    assert sum(scale) <= first["stat"] <= sum(scale) + 10 * present
    assert first["loads"] <= 3 * present  # Ownership read plus the result writer's merge reads.
    rows = published(env, duty.COST_PROJECTIONS)
    assert [(row["count"], row["by_basis"], len(row["task_ids"])) for row in rows] == [
        (quarantined + absent, {"result_absent": quarantined + absent}, 50)]
    assert duty_records(caplog) == []

    for name in counts:
        counts[name] = 0
    reconcile_abandoned_usage(env.root)
    assert counts["fence"] == counts["glob"] == 0 and counts["loads"] <= 3 * present
    assert sink.stat().st_size == size and duty_records(caplog) == []
    assert usage.usage_projection(env.root) == summary
