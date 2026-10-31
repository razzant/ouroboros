"""Composed quiet rest and late arrival across the real maintenance owners (an isolated drive, no live data).

One drive holds many settled children with evidence and exact prices beside limited active debt: a
running child, an undeclared duty (an older or newer body's), a live-owned open attempt and a remote
call whose owner Continues while its probe is out. The owners are the real ones: the 20 s pass's
abandoned-usage duty (``server_maintenance._reconcile_abandoned_usage``) and the 300 s pass's maintenance
tick, whose supervised child (``ouroboros.maintenance``) runs its due jobs here in process; its packing job
calls ``evidence_packer.run_pass``. Once settled in, the
repeated passes do nothing: no child spawn, no re-read or re-verified member, no daemon call, no
ledger, result or log write, and every content projection (the parent's child lookup, the results,
money, blob and handle bytes; the payload reader once) is equal, not merely counted. Late work is then
handled by kind: a new child, new evidence on a packed task (beside a duty committed while the pass's
census stalls in a directory probe, which holds nothing), a released owner, and an explicit late receipt
for the ended call whose price the duty no longer searches; an UNKNOWN price is never zero. Packing reads
the terminal status alone: the undeclared duty, the live-owned attempt and the ended call's unknown
price never keep their tasks' evidence loose.
"""
from __future__ import annotations

import json
import pathlib
import threading
from types import SimpleNamespace

import pytest

import ouroboros.maintenance as child_process
from ouroboros import evidence_packer as packer, evidence_packs as packs, obligations
from ouroboros import observability as obs, server_maintenance as maintenance, store, usage_accounting as usage
from ouroboros.artifacts import read_actor_source_bytes, store_actor_source_bytes
from ouroboros.task_results import load_task_result, write_task_result
from ouroboros.task_status import find_child_tasks
from ouroboros.usage_ledger import is_abandoned_settlement
from tests._usage_store_testing import ledger_rows
from tests.test_evidence_packs import REPO, _FakeChild, _blob, _handle_path, _loose

KIDS = [f"kid-{index:02}" for index in range(16)]
DEBT = ("kid-running", "kid-owed", "kid-live", "kid-ended")
DUTY = "future_body_duty"  # declared in neither tuple: an older or newer body's own duty


@pytest.fixture
def env(tmp_path, monkeypatch):
    from ouroboros import claudexor_daemon, llm_claudexor, process_custody
    from supervisor import queue

    root = tmp_path / "data"
    root.mkdir()
    monkeypatch.setenv("OUROBOROS_DATA_DIR", str(root))
    live, calls, reads, reports = set(), [], [], []
    child_run = threading.local()
    monkeypatch.setattr(queue, "task_has_live_ownership", lambda task_id, **_kw: task_id in live)
    monkeypatch.setattr(queue, "DRIVE_ROOT", root)

    class Gateway:  # the engine says the call is over; its owner may Continue while the probe is out
        continue_during_probe = False

        def get_model_operation(self, operation_id, **_kw):
            calls.append(operation_id)
            if Gateway.continue_during_probe:
                live.add("kid-ended")
            return {"state": "cancelled", "dispatch": {"state": "unknown"}}

        def close(self):
            pass

    recover = llm_claudexor.recover_model_attempt
    monkeypatch.setattr(claudexor_daemon, "read_owned_gateway", Gateway)
    monkeypatch.setattr(llm_claudexor, "recover_model_attempt",
                        lambda root, row, **kw: reads.append(row["task_id"]) or recover(root, row, **kw))

    run_pass = packer.run_pass

    def observed_pass(data_root, **kwargs):  # the packing pass inside the child, its report kept
        child_run.active = True
        try:
            reports.append(run_pass(data_root, **kwargs))
        finally:
            child_run.active = False
        return reports[-1]

    def spawn(cmd, **_kwargs):  # the supervised child, in process: its pass over the due jobs
        assert cmd[1:3] == ["-m", "ouroboros.maintenance"]
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(packer, "run_pass", observed_pass)
            report = child_process.run(root)
        return _FakeChild(json.dumps(report).encode())

    monkeypatch.setattr(process_custody, "spawn_supervised", spawn)
    monkeypatch.setattr(process_custody, "current_custody_session_id", lambda: "session")
    packs._READERS.clear()
    yield SimpleNamespace(root=root, live=live, calls=calls, reads=reads, reports=reports, gateway=Gateway,
                          unit=child_process.MaintenanceChild(), refs={}, handles={}, child_run=child_run)
    packs._READERS.clear()


def _child(env, task_id, status="completed", *, evidence=True):
    write_task_result(env.root, task_id, status, result=f"answer of {task_id}", root_task_id="root",
                      parent_task_id="root", delegation_role="subagent")
    if evidence:
        payloads = [json.dumps({"message": {"content": f"{task_id} {n} " + "x" * 300 * n}}).encode() for n in (1, 2)]
        env.refs[task_id] = _evidence(env, task_id, payloads)
        env.handles[task_id] = store_actor_source_bytes(env.root, task_id, category="context_checkpoints",
                                                        source_id=f"focus_{task_id}", extension="md",
                                                        data=f"# focus {task_id}\n".encode() + b"y" * 2000)


def _evidence(env, task_id, payloads, *, first=0):
    """Call manifests referencing gzip CAS blobs; their refs."""
    calls = env.root / "observability" / "calls" / task_id
    calls.mkdir(parents=True, exist_ok=True)
    refs = []
    for index, raw in enumerate(payloads, first):
        refs.append(_blob(env.root, raw))
        (calls / f"llm_{index}_response.json").write_text(
            json.dumps({"call_id": f"llm_{index}", "full_payload_ref": refs[-1]}), encoding="utf-8")
    return refs


def _attempt(env, task_id, *, provider="openai", price=None):
    reservation = usage.reserve_attempt(usage.AttemptRequest(
        model="test-model", provider=provider, drive_root=env.root, task_id=task_id, root_task_id="root",
        parent_task_id="root", reservation_usd=1.25, global_limit_usd=100.0))
    usage.mark_dispatched(reservation)
    if price is not None:
        usage.settle_attempt(reservation, cost_usd=price, cost_final=True)
    return reservation


def _row(env, reservation):
    return {row["attempt_id"]: row for row in ledger_rows(env.root)}[reservation.attempt_id]


def _pass(env):
    """One 20 s abandoned-usage pass, then one 300 s maintenance tick; the child's packing report, or
    None when no packing pass ran (no child at all is proven by the unchanged logs)."""
    spawned = len(env.reports)
    maintenance._reconcile_abandoned_usage(env.root)
    if env.unit.tick(env.root, REPO):
        env.unit._thread.join(60)
        assert not env.unit._thread.is_alive()
    return env.reports[spawned] if len(env.reports) > spawned else None


def _snapshot(env):
    """Every content projection the passes must leave equal: lookup, results, money, evidence bytes."""
    root = env.root
    return {"children": find_child_tasks(root, parent_task_id="root", root_task_id="root"),
            "results": {tid: load_task_result(root, tid) for tid in ("root", *env.refs)},
            "money": usage.usage_projection(root, include_roots=True), "ledger": ledger_rows(root),
            "blobs": {tid: [packs.blob_bytes(root, ref) for ref in refs] for tid, refs in env.refs.items()},
            "handles": {tid: read_actor_source_bytes(root, tid, ref) for tid, ref in env.handles.items()}}


def _loose_paths(env, task_id):
    handle = _handle_path(env.root, task_id, env.handles[task_id])
    return [pathlib.Path(ref["path"]) for ref in env.refs[task_id]] + [handle]


def _settle_in(env):
    """The world, then the passes that settle it in; the ended call's reservation and the loose truth bytes."""
    root = env.root
    write_task_result(root, "root", "completed", result="root answer", root_task_id="root", delegation_role="root",
                      root_phase_checkpoint={"post_task_synthesis": "completed"})
    for kid in KIDS:
        _child(env, kid)
        _attempt(env, kid, price=0.1)
    for kid in ("kid-01", "kid-02"):  # one blob two settled tasks reference
        env.refs[kid] += _evidence(env, kid, [json.dumps({"shared": "kid-01 and kid-02"}).encode()], first=5)
    for kid in DEBT:
        _child(env, kid, "running" if kid == "kid-running" else "completed")
    obligations.add(root, DUTY, "work:kid-owed", {"task_id": "kid-owed"})
    env.live.add("kid-live")
    _attempt(env, "kid-live")
    ended = _attempt(env, "kid-ended", provider="claudexor")
    obs.persist_call(root, task_id="kid-ended", call_id=f"{ended.attempt_id}_model_request",
                     call_type="llm_claudexor_request", payload={}, keep_raw=True,
                     manifest={"invocation_id": ended.attempt_id, "operation_id": "operation"})
    truth = {tid: [packs.blob_bytes(root, ref) for ref in refs] for tid, refs in env.refs.items()}

    # Pass 1. The owner Continues while the ended call's probe is out: the post-I/O fence refuses to
    # settle. Packing moves placements, never content: every projection equals its pre-pack reading.
    env.gateway.continue_during_probe = True
    maintenance._reconcile_abandoned_usage(root)
    assert env.calls == ["operation"] and env.reads == ["kid-ended"]
    assert _row(env, ended)["state"] == "dispatched" and "recovery" not in _row(env, ended)
    unpacked = _snapshot(env)
    assert env.unit.tick(root, REPO) is True
    env.unit._thread.join(60)
    # Every terminal row packs, whatever duty, owner or money state it carries; the running child waits.
    assert env.reports[-1]["packed"] == len(KIDS) + 3 and env.reports[-1]["faults"] == 0
    assert _snapshot(env) == unpacked and unpacked["blobs"] == truth
    assert {tid: [obs.read_blob_ref(root, ref) for ref in refs] for tid, refs in env.refs.items()} == {
        tid: [json.loads(raw) for raw in raws] for tid, raws in truth.items()}  # the payload consumer too
    for kid in (*KIDS, "kid-owed", "kid-live", "kid-ended"):
        assert packs.published_generations(root, kid) and not any(path.exists() for path in _loose_paths(env, kid))
    assert not packs.published_generations(root, "kid-running")
    assert all(path.exists() for path in _loose_paths(env, "kid-running"))

    # Pass 2. The owner is gone: the engine's answer pass 1 retained proves the call ended, offline (no
    # second daemon call). Its price is UNKNOWN: the bound is accounted, never zero, and the row keeps its
    # one late-receipt right. Its evidence was packed in pass 1 (money never held it); the settlement wrote
    # its row, so the packer's census is that one task (the rows written terminal since its position), whose
    # directories hold the stamp pass 1 recorded after its unlinks: no candidate, no packing pass, no
    # generation re-read; the packed children, whose directories only their own unlinks moved, are not looked at.
    env.live.discard("kid-ended")
    env.gateway.continue_during_probe = False
    second = _pass(env)
    assert env.calls == ["operation"] and env.reads == ["kid-ended", "kid-ended"]
    ended_row = _row(env, ended)
    assert is_abandoned_settlement(ended_row) and ended_row["recovery"]["outcome"] == "terminal"
    assert ended_row["cost_usd"] is None and not ended_row.get("late_receipt")
    projected = load_task_result(root, "kid-ended")
    assert projected["accounted_upper_bound_usd"] == 1.25 and projected["cost_final"] is False
    assert len(packs.published_generations(root, "kid-ended")) == 1
    assert second is None and "evidence_packing" not in child_process.due(root)

    # Pass 3: the terminal rows did not move and the last pass unlinked nothing: no child at all.
    assert _pass(env) is None
    return ended, truth


def test_settled_history_rests_with_every_projection_unchanged(env):
    _ended, truth = _settle_in(env)
    settled = _snapshot(env)
    assert settled["blobs"] == truth
    log = env.root / "logs" / "supervisor.jsonl"
    for _ in range(2):
        before = (log.read_bytes(), env.root.joinpath("logs", "events.jsonl").read_bytes())
        assert packer.changed_tasks(env.root) == [] and _pass(env) is None  # no candidate: no child at all
        assert env.calls == ["operation"] and env.reads == ["kid-ended", "kid-ended"]
        assert (log.read_bytes(), env.root.joinpath("logs", "events.jsonl").read_bytes()) == before
        assert _snapshot(env) == settled


def test_late_work_is_handled_by_kind_and_the_history_rests_again(env, monkeypatch):
    ended, truth = _settle_in(env)
    settled, root = _snapshot(env), env.root
    # A new child; new evidence on packed kid-03; kid-live's owner leaves; the explicit late receipt.
    _child(env, "kid-late")
    _attempt(env, "kid-late", price=0.1)
    late = _evidence(env, "kid-03", [json.dumps({"after": "packing"}).encode()], first=9)
    _child(env, "kid-03", evidence=False)  # late evidence rides a row write (``evidence_packer.terminal_revision``)
    env.refs["kid-03"] += late
    truth = {**truth, "kid-late": [packs.blob_bytes(root, ref) for ref in env.refs["kid-late"]],
             "kid-03": truth["kid-03"] + [packs.blob_bytes(root, ref) for ref in late]}
    env.live.discard("kid-live")
    usage.settle_attempt(ended, {}, cost_usd=0.4, cost_final=True)
    # Kid-03's undeclared duty is committed while the pass's census stalls in kid-03's directory
    # probe: the census holds no transaction there, so a 1 s writer commits.
    target, original = root / "observability" / "calls" / "kid-03", pathlib.Path.stat
    entered, release = threading.Event(), threading.Event()

    def stat(self, *args, **kwargs):
        if (self == target and getattr(env.child_run, "active", False) and not entered.is_set()):
            entered.set()
            assert release.wait(10)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(pathlib.Path, "stat", stat)
    maintenance._reconcile_abandoned_usage(root)
    assert env.unit.tick(root, REPO) is True
    try:
        assert entered.wait(10)
        with store.open_store(root, write=True, wait_sec=1.0) as txn:
            txn.put_row(DUTY, "work:kid-03", {"task_id": "kid-03"})
    finally:
        release.set()
        env.unit._thread.join(60)
    monkeypatch.setattr(pathlib.Path, "stat", original)
    arrival = env.reports[-1]
    # No automatic search for the ended call's price; its explicit late receipt settled once.
    assert env.calls == ["operation"] and env.reads == ["kid-ended", "kid-ended"]
    receipt = _row(env, ended)
    assert receipt["settle_reason"] == "late_receipt" and receipt["cost_usd"] == 0.4
    assert load_task_result(root, "kid-ended")["accounted_upper_bound_usd"] == 0.4
    # kid-live's owner left: its attempt closed at UNKNOWN price (the bound, never zero), then it packed.
    [live_row] = [row for row in ledger_rows(root) if row["task_id"] == "kid-live"]
    assert is_abandoned_settlement(live_row) and live_row["cost_usd"] is None
    assert load_task_result(root, "kid-live")["accounted_upper_bound_usd"] == 1.25
    assert packs.published_generations(root, "kid-live") and not any(p.exists() for p in _loose_paths(env, "kid-live"))
    # The new child: the lookup gains exactly its row; the pass packed it.
    children = {row["task_id"]: row for row in find_child_tasks(root, parent_task_id="root", root_task_id="root")}
    before = {row["task_id"]: row for row in settled["children"]}
    assert set(children) - set(before) == {"kid-late"}
    others = [tid for tid in KIDS if tid != "kid-03"]  # kid-03's row carried its late evidence; no other row moved
    assert {tid: children[tid] for tid in others} == {tid: before[tid] for tid in others}
    assert children["kid-03"]["result"] == before["kid-03"]["result"]
    assert packs.published_generations(root, "kid-late") and not _loose(env.refs["kid-late"][0])
    # The duty arrived after the census and before the unlink: it holds nothing, the new member packs.
    assert arrival["faults"] == 0 and not _loose(late[0]) and len(packs.published_generations(root, "kid-03")) == 2
    with store.open_store(root, write=False) as txn:
        assert packs.task_packable(txn, "kid-03")

    # The owners retire their work: kid-03's and kid-owed's duties, kid-running completes.
    obligations.remove(root, DUTY, "work:kid-03")
    obligations.remove(root, DUTY, "work:kid-owed")
    _child(env, "kid-running", evidence=False)
    for _ in range(4):
        _pass(env)
    assert not _loose(late[0]) and all(packs.published_generations(root, tid) for tid in env.refs)
    assert not any(path.exists() for tid in env.refs for path in _loose_paths(env, tid))
    rested = _snapshot(env)
    assert rested["blobs"] == truth
    for _ in range(2):
        assert packer.changed_tasks(root) == [] and _pass(env) is None
        assert _snapshot(env) == rested
