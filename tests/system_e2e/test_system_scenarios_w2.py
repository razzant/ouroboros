"""S6-S10 — Ф4 wave 2 of the deep-integration suite (v7next plan §8).

Three surfaces on the wave-1 skeleton, each asserted through DURABLE artifacts
(never an HTTP 200 alone, never a harness exit code), synchronized by durable-event
polling (``wait_until`` over ``ArtifactOracle`` readers), keyless throughout:

* S6 — SUBAGENT TREE: a scripted parent drives ``schedule_subagent`` →
  ``wait_tasks`` → an exact-hash ``tree_note`` child disposition → final answer,
  with the child on its OWN stub slot (``ReplayModel`` bound by (lineage, slot,
  attempt) — the wire's model id separates the two sides, so no step can be
  consumed by the wrong actor). Asserts honest lineage in the durable rows
  (``parent_task_id``/``root_task_id``/``delegation_role``/depth provenance), the
  fanout receipt in the parent's forked drive, quiescence (the child's terminal
  ``task_done`` precedes the parent's; the parent's terminal is clean, not the
  degraded ``children_unabsorbed`` path), the child result reaching the parent
  verbatim (marker in the parent's durable ``wait_tasks`` tool row), the
  authoritative disposition row on the task-tree ledger, and the root cost rollup
  keys on the parent's terminal event.
* S7 — CANCELLATION (single): a keepalive task is cancelled over the same HTTP
  surface the UI drives; the durable record is the typed ``cancelled`` terminal
  with a non-empty owed answer and a ``cancel_receipt``, the cost plane carries
  honest names only, the durable cancel intent settles and drains, the forensic
  trail is requested→claimed→settled(source=http_single), and no process carrying
  this server's data root survives outside its live process tree.
* S8 — CANCELLATION (cascade): parent schedules a live child, cascade cancel tears
  the subtree down: both durable rows are ``cancelled``, the root intent settles on
  the cascade postcondition, the descendant carries its own
  ``cascade_descendant`` intent naming the root, the subtree snapshot row lists the
  child — and the live process tree holds NO orphans (env-value /proc scan).
* S9 — MANAGED UPDATE (ff core): a REAL isolated install with a LOCAL managed
  repository (the CONFIGURED update source: ``managed_remote_url`` in the managed
  metadata names the local mirror — the fork/air-gap install shape the W2-F2 fix
  honors on every fetch, no runtime code patched) applies a fast-forward update
  over the real HTTP surface
  with DIRTY local work present: the stash-first insurance carries the work, the
  server re-execs onto the target SHA, and boot-finalize is honest (tx marker and
  update intent consumed, ``managed_update_finalized.head`` == target, dirty work
  restored as uncommitted content, durable ``rescue-local-*`` stash pin present).
* S10 — MANAGED UPDATE (rollback contracts): a subprocess driver on a second real
  isolated install proves the typed refusals (absent/null-stamp marker → the
  ``pre_update_sha`` refusal; FUTURE-schema marker → the newer-version refusal with
  the marker bytes left byte-identical, for ``rollback_managed_update`` AND
  ``finalize_managed_update_on_boot``) and the restore contract: a half-applied
  update (target commit + extra dirt + a valid pending tx) rolls back to a tree
  whose every file is BYTE-IDENTICAL to the pre-update snapshot, with the failed
  candidate preserved on ``failed-update-*`` and the durable
  ``managed_update_rolled_back`` receipt written.

Heavy variations disclosed as wave-3 remainder: carrier-conflict / assisted merge /
crash-mid-phase update scenarios, and the paid-lane cancellation E1-E3 analogues.
"""

from __future__ import annotations

import json
import os
import pathlib
import re
import subprocess
import sys
import uuid

import pytest

from tests.system_e2e import harness
from tests.system_e2e.harness import (
    LANE_MOCK,
    MOCK_SLUG,
    REPO_ROOT,
    ArtifactOracle,
    ReplayModel,
    ScriptedStubModel,
    body_text,
    clone_repo,
    keyless_settings,
    pids_with_env_value,
    process_tree_pids,
    require_lane,
    retired_cost_alias_paths,
    scripted_completion,
    start_server,
    submit_running,
    wait_durable_result,
    wait_pid_env_value,
    wait_until,
)

from devtools.benchmarks.common.server_runner import _api

# ===========================================================================
# Default lane: pins for the wave-2 harness surface (no server, no sockets).
# ===========================================================================


def _agent_body(text: str = "keep going") -> dict:
    return {"messages": [{"role": "user", "content": text}],
            "tools": [{"type": "function", "function": {"name": "list_files"}}],
            "model": "mock-model"}


def test_scripted_completion_resolves_callable_steps():
    """A script step may be ``callable(body) -> step``: the dynamic-argument contract
    scenarios use for server-minted ids. Both step shapes must resolve — a tool call
    and an early ``{"final": ...}``."""
    steps = iter([
        lambda body: {"tool": "wait_tasks",
                      "arguments": {"task_ids": [body_text(body)[-8:]]}},
        lambda _body: {"final": "dynamic final"},
    ])

    def _next(_body):
        return next(steps, None)

    kind1, msg1 = scripted_completion(_agent_body("child id: abcd1234"), 1, _next, "done")
    assert kind1 == "agent"
    call = msg1["tool_calls"][0]["function"]
    assert call["name"] == "wait_tasks"
    assert json.loads(call["arguments"])["task_ids"] == ["abcd1234"]
    kind2, msg2 = scripted_completion(_agent_body(), 2, _next, "done")
    assert (kind2, msg2["content"]) == ("final", "dynamic final")


def test_replay_model_callable_steps_and_explicit_model_ids():
    model = ReplayModel(
        {("root", "mock-model", 1): lambda body: {"final": body_text(body)[-4:]}},
        model_ids=["mock-model", "mock-child"],
    )
    # Explicit model_ids override the fixture-derived advertisement — a compound
    # slot_binder scenario would otherwise advertise garbage slot names.
    assert model._model_ids() == ["mock-child", "mock-model"]
    _kind, msg = model._answer({"messages": [{"role": "user", "content": "tail"}],
                                "model": "mock-model"}, 1)
    assert msg["content"] == "tail"
    model.assert_consumed()


@pytest.mark.skipif(sys.platform != "linux", reason="/proc environ scan is Linux-only")
def test_pids_with_env_value_sees_and_loses_a_marked_child():
    """Both directions of the /proc environ oracle, each through its own contract:
    the positive claim waits a bounded window for the just-execed child's environ
    (``wait_pid_env_value``), the no-orphans postcondition keeps its single scan."""
    marker = f"e2e-w2-marker-{uuid.uuid4().hex}"
    child = subprocess.Popen(["sleep", "30"], env={**os.environ, "E2E_W2_MARK": marker})
    try:
        assert wait_pid_env_value(child.pid, marker)
        assert child.pid in pids_with_env_value(marker)
    finally:
        child.kill()
        child.wait(timeout=10)
    assert child.pid not in pids_with_env_value(marker)


@pytest.mark.skipif(sys.platform != "linux", reason="/proc environ scan is Linux-only")
def test_pid_env_wait_rides_out_the_post_exec_empty_environ_window(monkeypatch):
    """The exact CI interleaving (33671108287): ``Popen`` has returned — the exec
    succeeded — but the kernel has not published the new image's env_start/env_end
    yet, so the environ reads EMPTY for a live, correctly marked pid. A single scan
    misses it (and that is what the positive assertion used to be); the bounded
    per-pid wait rides the window out and still returns True."""
    marker = "e2e-w2-window-marker"
    reads = {"n": 0}
    real = harness._read_proc_environ_bytes

    def _windowed(pid):
        if str(pid) != str(os.getpid()):
            return real(pid)
        reads["n"] += 1
        return b"" if reads["n"] <= 3 else f"E2E_W2_MARK={marker}\x00".encode()

    monkeypatch.setattr(harness, "_read_proc_environ_bytes", _windowed)
    assert os.getpid() not in pids_with_env_value(marker)      # the single scan misses it
    assert wait_pid_env_value(os.getpid(), marker, timeout=5)  # the bounded wait does not
    assert reads["n"] > 3


# ===========================================================================
# Shared helpers of the mock-lane scenarios
# ===========================================================================


def _git(args, cwd, check: bool = True) -> str:
    proc = subprocess.run(["git", *args], cwd=str(cwd), check=check,
                          capture_output=True, text=True)
    return (proc.stdout or "").strip()


def _no_stray_processes(server, data_root) -> bool:
    """True when every live pid carrying this server's data root in its environment
    is still INSIDE the server's own process tree (no reparented orphan)."""
    tree = set(process_tree_pids(server.proc.pid))
    return all(pid in tree for pid in pids_with_env_value(str(data_root)))


def _cancel_forensics(oracle: ArtifactOracle, task_id: str) -> list:
    return [row for row in oracle.supervisor_rows("cancel_intent")
            if str(row.get("task_id") or "") == task_id]


def _forensic_events(rows: list) -> list:
    return [str(row.get("event") or "") for row in rows]


def _assert_honest_cancel_terminal(oracle: ArtifactOracle, stored: dict, task_id: str) -> None:
    """The typed cancelled terminal: owed answer accounted for, honest cost names.

    GR2-4/GR4-1: the terminal answer must be DURABLY accounted for before the
    intent settles — for a task WITH chat lineage that is a registered outbox
    delivery (plus the ``cancel_receipt`` block merged into the row); for a
    chatless API-submitted task (this lane) the routing decision itself is
    durable: a typed ``terminal_delivery_handoff`` row with
    ``reason=no_lineage_chat``. Either form satisfies the owed-answer contract;
    a silently dropped answer satisfies neither.
    """
    assert stored.get("status") == "cancelled", stored
    assert str(stored.get("result") or "").strip(), "cancelled terminal owes an answer"

    def _owed_evidence():
        deliveries = oracle.terminal_deliveries()
        known = set(deliveries.get("pending") or {}) | set(deliveries.get("delivered") or [])
        if any(task_id in str(did) for did in known):
            return ("outbox", None)
        handoffs = [row for row in oracle.supervisor_rows("terminal_delivery_handoff")
                    if str(row.get("task_id") or "") == task_id]
        if handoffs:
            return ("handoff", handoffs[-1])
        return None

    owed = wait_until(_owed_evidence, 60)
    assert owed, "the cancelled answer is neither owed in the outbox nor typed as a handoff"
    form, handoff = owed
    if form == "handoff":
        assert handoff.get("settled_status") == "cancelled", handoff
        assert handoff.get("reason") == "no_lineage_chat", handoff
    else:
        receipt = wait_until(
            lambda: (oracle.task_result(task_id).get("cancel_receipt")
                     if isinstance(oracle.task_result(task_id).get("cancel_receipt"), dict)
                     else None),
            60,
        )
        assert isinstance(receipt, dict) and receipt, oracle.task_result(task_id)
        assert receipt.get("settled_status") == "cancelled", receipt
        assert task_id in str(receipt.get("delivery_id") or ""), receipt
    # The $-plane must not lie: honest names present, retired aliases absent at the
    # public top level, and the accounting status is a disclosed enum — an unknown
    # cost is honest, a fabricated $0 is not.
    assert "accounted_upper_bound_usd" in stored, sorted(stored)
    assert str(stored.get("cost_accounting_status") or "") in {"available", "unavailable"}, stored
    top_level_aliases = [p for p in retired_cost_alias_paths(stored) if p.count(".") == 1]
    assert top_level_aliases == [], stored


KEEPALIVE_STEP = {"tool": "list_files", "arguments": {"path": "."}}


# ===========================================================================
# S6 — subagent tree (ReplayModel, child on its own stub slot)
# ===========================================================================

S6_CHILD_SLUG = "openai-compatible::mock-child"
S6_CHILD_MARKER = "CHILD_RESULT_PAYLOAD_e2e_w2"
S6_PARENT_MARKER = "PARENT_FINAL_e2e_w2"

_CHILD_ID_RE = re.compile(r"Subagent request queued ([0-9a-f]{8})")
_CHILD_SHA_RE = re.compile(r"child_result_sha256[\"'=:\s]+([0-9a-f]{64})")


def _s6_slot_binder(body: dict) -> str:
    """Slot = wire model id × tool-bearing shape: the parent loop runs on
    ``mock-model`` with tools, the supervisor's semantic-duplicate probe runs on the
    light slot (same slug) WITHOUT tools, and the child runs on ``mock-child`` — so
    every fixture ordinal is deterministic even while parent and child overlap."""
    return f"{body.get('model') or ''}|{'tools' if body.get('tools') else 'plain'}"


def _s6_wait_step(body: dict) -> dict:
    ids = _CHILD_ID_RE.findall(body_text(body))
    if not ids:
        return {"final": "E2E_SCRIPT_ERROR: no scheduled child id visible in the transcript"}
    return {"tool": "wait_tasks", "arguments": {
        "task_ids": [ids[-1]], "timeout_sec": 240, "mode": "all_terminal"}}


def _s6_dispose_step(body: dict) -> dict:
    text = body_text(body)
    ids = _CHILD_ID_RE.findall(text)
    shas = _CHILD_SHA_RE.findall(text)
    if not ids or not shas:
        return {"final": "E2E_SCRIPT_ERROR: missing child id or exact result hash for the disposition"}
    return {"tool": "tree_note", "arguments": {
        "kind": "decision",
        "text": "Absorbed the scout child's listing into the final answer.",
        "payload": {"type": "child_result_disposition", "child_task_id": ids[-1],
                    "disposition": "integrated", "child_result_sha256": shas[-1]},
    }}


S6_FIXTURE = {
    ("root", "mock-model|tools", 1): {"tool": "schedule_subagent", "arguments": {
        "subagent_id": "mock-child",
        "objective": "List the repository root and report the entries you saw.",
        "expected_output": "A short list of repository root entries.",
    }},
    # The supervisor's admission duplicate-probe (light slot, tool-less).
    ("root", "mock-model|plain", 1): {"final": "No existing task duplicates this request."},
    ("root", "mock-model|tools", 2): _s6_wait_step,
    ("root", "mock-model|tools", 3): _s6_dispose_step,
    ("root", "mock-model|tools", 4): {"final": f"{S6_PARENT_MARKER}: child absorbed; done."},
    ("root", "mock-child|tools", 1): KEEPALIVE_STEP,
    ("root", "mock-child|tools", 2): {"final": f"{S6_CHILD_MARKER}: repository root listed."},
}


@pytest.mark.integration
@pytest.mark.serial
def test_s6_subagent_tree_lineage_quiescence_and_child_result_handoff(
        e2e_clone, tmp_path_factory):
    require_lane(LANE_MOCK)
    root = tmp_path_factory.mktemp("s6")
    model = ReplayModel(S6_FIXTURE, slot_binder=_s6_slot_binder,
                        model_ids=["mock-model", "mock-child"])
    with model:
        settings = keyless_settings(
            model,
            OUROBOROS_SUBAGENTS=json.dumps({"enabled": True, "items": [{
                "subagent_id": "mock-child",
                "recommended_use": "Read-only scout for the system_e2e subagent-tree scenario.",
                "route": {"kind": "api_model", "target_id": S6_CHILD_SLUG},
                "effort": "low",
            }]}),
        )
        server = start_server(e2e_clone, root, settings)
        try:
            parent_id = submit_running(
                server, "Delegate the repository survey to your scout, absorb it, then finish.")
            result = server.wait_task(parent_id, timeout=600)
            assert result.get("status") == "completed", result

            oracle = ArtifactOracle(server.data_root)
            parent_stored = wait_durable_result(oracle, parent_id)
            assert S6_PARENT_MARKER in str(parent_stored.get("result") or ""), parent_stored
            # Clean quiescent finalization — not the degraded forced path.
            assert str(parent_stored.get("reason_code") or "") != "children_unabsorbed", parent_stored

            # Lineage truth in the CHILD's durable row.
            children = oracle.child_task_ids(parent_id)
            assert len(children) == 1, children
            child_id = children[0]
            child_stored = wait_durable_result(oracle, child_id)
            assert child_stored.get("status") == "completed", child_stored
            assert child_stored.get("parent_task_id") == parent_id, child_stored
            assert child_stored.get("root_task_id") == parent_id, child_stored
            assert child_stored.get("delegation_role") == "subagent", child_stored
            assert S6_CHILD_MARKER in str(child_stored.get("result") or ""), child_stored
            provenance = child_stored.get("depth_provenance")
            assert isinstance(provenance, dict) and int(provenance.get("achieved_depth") or 0) == 1, (
                child_stored.get("depth_provenance"))
            snapshot = child_stored.get("configured_subagent")
            assert isinstance(snapshot, dict) and snapshot.get("selected_subagent_id") == "mock-child", snapshot
            contract = child_stored.get("task_contract")
            assert isinstance(contract, dict), child_stored
            lineage = (contract.get("lineage") or {})
            assert lineage.get("parent_task_id") == parent_id, contract

            # The fanout receipt lives in the PARENT's forked drive.
            parent_drive = oracle.task_drive(parent_id)
            assert parent_drive.data_root != oracle.data_root, "parent forked drive missing"
            fanouts = parent_drive.events("swarm_fanout")
            assert fanouts and fanouts[0].get("task_ids") == [child_id], fanouts

            # The child's result reached the parent VERBATIM: the durable wait_tasks
            # tool row in the parent drive carries the child marker.
            wait_rows = [row for row in parent_drive.tools_rows()
                         if "wait_tasks" in str(row.get("tool") or row.get("name") or "")]
            assert wait_rows, "wait_tasks call missing from the parent tools log"
            wait_blob = json.dumps(wait_rows)
            assert child_id in wait_blob, wait_rows
            assert S6_CHILD_MARKER in wait_blob, "child result text never reached the parent"

            # Quiescence: the child's terminal task_done precedes the parent's.
            done_ids = [str(row.get("task_id") or "") for row in oracle.events("task_done")]
            assert child_id in done_ids and parent_id in done_ids, done_ids
            assert done_ids.index(child_id) < done_ids.index(parent_id), done_ids

            # The authoritative exact-hash disposition row on the task-tree ledger.
            board = oracle.tree_blackboard(parent_id)
            dispositions = [row for row in board
                            if isinstance(row.get("payload"), dict)
                            and row["payload"].get("type") == "child_result_disposition"]
            assert dispositions, board
            assert dispositions[-1]["payload"].get("child_task_id") == child_id, dispositions
            assert dispositions[-1]["payload"].get("disposition") == "integrated", dispositions

            # Root cost rollup: the parent's terminal event carries the honest
            # with-children names (values may honestly be unknown; keys must exist).
            parent_done = [row for row in oracle.events("task_done")
                           if str(row.get("task_id") or "") == parent_id][-1]
            assert "accounted_upper_bound_usd_with_children" in parent_done, sorted(parent_done)
            assert retired_cost_alias_paths(parent_done) == [], parent_done
            # Child terminal row: honest cost names at the top level too.
            assert "accounted_upper_bound_usd" in child_stored, sorted(child_stored)
            assert [p for p in retired_cost_alias_paths(child_stored) if p.count(".") == 1] == []

            # Every fixture row was consumed and nothing was missed: the scenario's
            # call pattern is EXACTLY the scripted tree, both sides included.
            assert wait_until(lambda: parent_id not in oracle.running_ids(), 60)
            model.assert_consumed()
        finally:
            server.stop()


# ===========================================================================
# S7 — cancellation: typed terminal, owed answer, honest cost, drained intents
# ===========================================================================


@pytest.mark.integration
@pytest.mark.serial
@pytest.mark.skipif(sys.platform != "linux", reason="/proc no-orphans scan is Linux-only")
def test_s7_cancel_live_task_typed_terminal_owed_answer_and_no_strays(
        e2e_clone, tmp_path_factory):
    require_lane(LANE_MOCK)
    root = tmp_path_factory.mktemp("s7")
    with ScriptedStubModel([dict(KEEPALIVE_STEP) for _ in range(300)],
                           latency_sec=0.15) as stub:
        server = start_server(e2e_clone, root, keyless_settings(stub))
        try:
            oracle = ArtifactOracle(server.data_root)
            task_id = submit_running(server, "Keep listing the repository root until stopped.")
            # The scan oracle actually sees the live tree (guard against a vacuous
            # no-strays pass): the server itself carries the data root in its env.
            assert server.proc.pid in pids_with_env_value(str(server.data_root))

            envelope = server.cancel_task(task_id)
            assert envelope.get("status") == 200, envelope
            assert (envelope.get("body") or {}).get("ok") is True, envelope

            stored = wait_durable_result(oracle, task_id)
            _assert_honest_cancel_terminal(oracle, stored, task_id)

            # The durable intent settled and DRAINED (self-cleaning projection).
            assert wait_until(lambda: task_id not in oracle.cancel_intents(), 60), (
                oracle.cancel_intents())
            # Forensic trail: requested -> claimed -> settled, single-scope HTTP ingress.
            rows = _cancel_forensics(oracle, task_id)
            events = _forensic_events(rows)
            for expected in ("requested", "claimed", "settled"):
                assert expected in events, events
            requested = next(row for row in rows if row.get("event") == "requested")
            assert requested.get("source") == "http_single", requested
            assert requested.get("scope") == "single", requested
            settled = next(row for row in rows if row.get("event") == "settled")
            assert settled.get("outcome") == "cancelled", settled

            # Terminal event (the owed-answer accounting is asserted above).
            assert wait_until(
                lambda: any(str(r.get("task_id") or "") == task_id
                            and str(r.get("status") or "") == "cancelled"
                            for r in oracle.events("task_done")), 60)

            # Queue drained and NO stray process outside the live server tree.
            assert wait_until(lambda: task_id not in oracle.running_ids(), 60)
            assert wait_until(lambda: _no_stray_processes(server, server.data_root), 45), (
                "a process carrying this server's data root survived outside its tree")
        finally:
            server.stop()


# ===========================================================================
# S8 — cancellation cascade: subtree teardown without orphans
# ===========================================================================

S8_SCHEDULE_STEP = {"tool": "schedule_subagent", "arguments": {
    "subagent_id": "mock-scout",
    "objective": "Keep surveying the repository root until told otherwise.",
    "expected_output": "A running commentary of the repository root.",
}}


@pytest.mark.integration
@pytest.mark.serial
@pytest.mark.skipif(sys.platform != "linux", reason="/proc no-orphans scan is Linux-only")
def test_s8_cascade_cancel_tears_down_subtree_without_orphans(
        e2e_clone, tmp_path_factory):
    require_lane(LANE_MOCK)
    root = tmp_path_factory.mktemp("s8")
    script = [dict(S8_SCHEDULE_STEP)] + [dict(KEEPALIVE_STEP) for _ in range(600)]
    with ScriptedStubModel(script, latency_sec=0.15) as stub:
        settings = keyless_settings(
            stub,
            OUROBOROS_SUBAGENTS=json.dumps({"enabled": True, "items": [{
                "subagent_id": "mock-scout",
                "recommended_use": "Read-only scout for the system_e2e cascade scenario.",
                "route": {"kind": "api_model", "target_id": MOCK_SLUG},
                "effort": "low",
            }]}),
        )
        server = start_server(e2e_clone, root, settings)
        try:
            oracle = ArtifactOracle(server.data_root)
            parent_id = submit_running(
                server, "Delegate a survey to your scout and keep working until stopped.")
            # A LIVE subtree: the child exists durably and reaches RUNNING.
            assert wait_until(lambda: oracle.child_task_ids(parent_id), 120), (
                "child task never appeared in task_results")
            child_id = oracle.child_task_ids(parent_id)[0]
            assert wait_until(lambda: child_id in oracle.running_ids(), 180), (
                f"child {child_id} never reached the RUNNING set")

            envelope = server.cancel_task(parent_id, cascade=True)
            assert envelope.get("status") == 200, envelope
            body = envelope.get("body") or {}
            assert body.get("ok") is True and body.get("cascade") is True, envelope

            parent_stored = wait_durable_result(oracle, parent_id)
            child_stored = wait_durable_result(oracle, child_id)
            _assert_honest_cancel_terminal(oracle, parent_stored, parent_id)
            assert child_stored.get("status") == "cancelled", child_stored
            assert child_stored.get("parent_task_id") == parent_id, child_stored

            # Root intent: cascade scope from the ingress, settled ONLY on the
            # subtree postcondition. Descendant: its OWN durable intent naming the root.
            root_rows = _cancel_forensics(oracle, parent_id)
            root_requested = next(r for r in root_rows if r.get("event") == "requested")
            assert root_requested.get("scope") == "cascade", root_requested
            assert root_requested.get("source") == "http_cascade", root_requested
            root_settled = next(r for r in root_rows if r.get("event") == "settled")
            assert "cascade postcondition" in str(root_settled.get("detail") or ""), root_settled
            child_rows = _cancel_forensics(oracle, child_id)
            child_requested = next(r for r in child_rows if r.get("event") == "requested")
            assert child_requested.get("source") == "cascade_descendant", child_requested
            assert child_requested.get("requested_by") == parent_id, child_requested

            # The durable subtree snapshot names the child.
            snapshots = [row for row in oracle.supervisor_rows("task_cancel_subtree_snapshot")
                         if str(row.get("root_task_id") or "") == parent_id]
            assert snapshots and child_id in (snapshots[0].get("descendant_task_ids") or []), snapshots

            # Both intents drained; both terminal events written.
            assert wait_until(
                lambda: parent_id not in oracle.cancel_intents()
                and child_id not in oracle.cancel_intents(), 60), oracle.cancel_intents()
            for tid in (parent_id, child_id):
                assert wait_until(
                    lambda tid=tid: any(str(r.get("task_id") or "") == tid
                                        and str(r.get("status") or "") == "cancelled"
                                        for r in oracle.events("task_done")), 60), tid

            # No orphans: every pid still carrying this data root is in the live tree.
            assert wait_until(lambda: not oracle.running_ids(), 60)
            assert wait_until(lambda: _no_stray_processes(server, server.data_root), 45), (
                "a subtree process survived cascade teardown outside the server tree")
        finally:
            server.stop()


# ===========================================================================
# S9 — managed update: local managed repo, ff apply, dirty-stash insurance,
# honest boot-finalize. The most expensive scenario of the wave — kept single.
# ===========================================================================

OFFICIAL_UPDATE_URL = "https://github.com/razzant/ouroboros"
S9_TRACKED_DIRTY = "CONTRIBUTING.md"
S9_DIRTY_MARK = "e2e-w2: local uncommitted work must survive the managed update\n"
S9_UNTRACKED = "e2e_w2_local_note.txt"
S9_PAYLOAD = "docs/notes/system_e2e_update_payload.md"


def _managed_install(root: pathlib.Path):
    """A REAL isolated managed install: a scenario clone whose ``managed`` remote is
    a LOCAL upstream one ff-commit ahead. The install CONFIGURES its update source
    (``managed_remote_url`` in the managed metadata — the fork/mirror/air-gap
    install shape both bootstraps write), and the runtime honors it on every
    update fetch (W2-F2 fix, owner №4=A). Belt: the hardcoded official URL is
    redirected to a non-existent local path via git ``insteadOf`` config, so a
    REGRESSED repin to the official URL fails loudly in this keyless lane
    instead of silently reaching the network.
    """
    clone = clone_repo(root)
    upstream = pathlib.Path(root) / "upstream"
    subprocess.run(["git", "clone", "--no-hardlinks", "-q", str(clone), str(upstream)],
                   check=True, capture_output=True)
    _git(["checkout", "-q", "ouroboros"], upstream)
    _git(["config", "user.name", "Managed Upstream"], upstream)
    _git(["config", "user.email", "upstream@e2e.invalid"], upstream)
    payload = upstream / S9_PAYLOAD
    payload.parent.mkdir(parents=True, exist_ok=True)
    payload.write_text("# system_e2e S9 managed update payload\n", encoding="utf-8")
    _git(["add", S9_PAYLOAD], upstream)
    _git(["commit", "-q", "-m", "docs: system_e2e S9 managed update payload"], upstream)
    target_sha = _git(["rev-parse", "HEAD"], upstream)

    _git(["remote", "add", "managed", str(upstream)], clone)
    _git(["config",
          f"url.{pathlib.Path(root) / 'nonexistent-official-mirror'}.insteadOf",
          OFFICIAL_UPDATE_URL], clone)
    (clone / ".git" / "ouroboros-managed.json").write_text(json.dumps({
        "managed_remote_name": "managed",
        "managed_remote_url": str(upstream),
        "managed_remote_branch": "ouroboros",
        "managed_local_branch": "ouroboros",
    }), encoding="utf-8")
    return clone, target_sha


@pytest.mark.integration
@pytest.mark.serial
def test_s9_managed_ff_update_applies_with_dirty_stash_and_honest_boot_finalize(
        tmp_path_factory, monkeypatch):
    require_lane(LANE_MOCK)
    root = tmp_path_factory.mktemp("s9")
    clone, target_sha = _managed_install(root)
    base_sha = _git(["rev-parse", "HEAD"], clone)
    assert base_sha != target_sha
    # Belt for the update smoke's dependency sync: the runtime lock is satisfied by
    # this interpreter's environment (verified), so pip must be a no-op — no-index
    # turns any unexpected resolution attempt into a loud failure instead of a
    # network install.
    monkeypatch.setenv("PIP_NO_INDEX", "1")
    with ScriptedStubModel([]) as stub:
        settings = keyless_settings(stub, OUROBOROS_UPDATE_CHANNEL="development")
        server = start_server(clone, root, settings)
        try:
            oracle = ArtifactOracle(server.data_root)
            # Dirty local work AFTER boot: one tracked edit + one untracked file,
            # both untouched by the incoming update commit.
            tracked = clone / S9_TRACKED_DIRTY
            tracked.write_text(tracked.read_text(encoding="utf-8") + S9_DIRTY_MARK,
                               encoding="utf-8")
            (clone / S9_UNTRACKED).write_text("local operator note\n", encoding="utf-8")

            plan = (_api(server.base_url, "POST", "/api/update/preflight", {}, timeout=300)
                    .get("merge_plan") or {})
            assert plan.get("available") is True and plan.get("kind") == "clean", plan
            assert plan.get("base_sha") == base_sha, plan
            assert plan.get("target_sha") == target_sha, plan

            applied = _api(server.base_url, "POST", "/api/update/apply", {
                "strategy": "auto_merge",
                "expected_base_sha": base_sha,
                "expected_target_sha": target_sha,
            }, timeout=600)
            assert applied.get("status") == "ok", applied
            assert applied.get("restarting") is True, applied

            # The server re-execs onto the merged tree. Boot-finalize runs ONLY in
            # the server boot path (server.py schedules finalize_managed_update_on_boot
            # on startup), so the cleared tx marker + the finalize receipt below are
            # durable proof BOTH of the restart and of its honesty. (/api/state's
            # ``sha`` field is empty on source-mode isolated servers — see the
            # wave-2 findings ledger — so runtime identity is not probed there.)
            tx_marker = clone / ".git" / "ouroboros-update-tx.json"
            intent_marker = clone / ".git" / "ouroboros-update-intent.json"
            assert wait_until(lambda: not tx_marker.exists(), 300), "update tx marker never cleared"
            assert server.wait_for_health(300)
            assert not intent_marker.exists(), "update intent marker survived boot-finalize"
            finalized = wait_until(
                lambda: oracle.supervisor_rows("managed_update_finalized") or None, 60)
            assert finalized, "no managed_update_finalized receipt in supervisor.jsonl"
            assert finalized[-1].get("head") == target_sha, finalized[-1]

            # The repo landed exactly on target, on the managed branch, with the
            # update payload present.
            assert _git(["rev-parse", "HEAD"], clone) == target_sha
            assert _git(["rev-parse", "--abbrev-ref", "HEAD"], clone) == "ouroboros"
            assert (clone / S9_PAYLOAD).is_file()

            # Dirty-work insurance (Q1=C): the stash carried the owner's work and
            # boot-finalize restored it as UNCOMMITTED content; the durable
            # rescue-local pin of the stash commit survives as the belt.
            assert wait_until(
                lambda: (clone / S9_TRACKED_DIRTY).read_text(encoding="utf-8").endswith(S9_DIRTY_MARK),
                120), "tracked dirty edit was not restored after the update"
            assert (clone / S9_UNTRACKED).is_file(), "untracked local file was not restored"
            porcelain = _git(["status", "--porcelain"], clone)
            assert S9_TRACKED_DIRTY in porcelain and S9_UNTRACKED in porcelain, porcelain
            pins = _git(["branch", "--list", "rescue-local-*"], clone)
            assert pins.strip(), "durable rescue-local stash pin is missing"

            # W2-F2 (owner №4=A): the CONFIGURED update source survived the whole
            # update cycle — every update fetch repins the managed remote to the
            # install's ``managed_remote_url``, never silently back to the
            # hardcoded official URL (which this install redirects to a
            # non-existent path, so a regression fails loudly, not quietly).
            assert _git(["remote", "get-url", "managed"], clone) == str(
                pathlib.Path(root) / "upstream"), "configured update source was retargeted"
        finally:
            server.stop()


# ===========================================================================
# S10 — managed update rollback contracts on a second real isolated install
# (subprocess driver: the real supervisor code, no live server needed).
# ===========================================================================

S10_DRIVER = r'''
import hashlib, json, os, pathlib, subprocess, sys

clone = pathlib.Path(sys.argv[1])
data = pathlib.Path(sys.argv[2])

from supervisor import git_ops
git_ops.init(clone, data, "")
from supervisor.update_merge import (
    finalize_managed_update_on_boot,
    read_update_tx_strict,
    rollback_managed_update,
    write_update_tx,
)

marker = clone / ".git" / "ouroboros-update-tx.json"
report = {}

def _git(*args):
    proc = subprocess.run(["git", *args], cwd=str(clone), check=True,
                          capture_output=True, text=True)
    return (proc.stdout or "").strip()

def _tree_fingerprint():
    """sha256 of every file under the worktree (excluding .git) — byte truth."""
    digest = hashlib.sha256()
    for path in sorted(clone.rglob("*")):
        rel = path.relative_to(clone)
        if rel.parts and rel.parts[0] == ".git":
            continue
        # __pycache__ is machine-generated bytecode the SERVER SUBPROCESS writes
        # while importing this clone (the isolated server sets no
        # PYTHONPYCACHEPREFIX). It is never repo content, so a .pyc appearing
        # between the baseline and the assertion is not evidence that the
        # refused update touched the worktree — but it DID flip this digest
        # under load, when a late first-import landed after the snapshot.
        if "__pycache__" in rel.parts or path.suffix in {".pyc", ".pyo"}:
            continue
        if path.is_symlink():
            digest.update(b"L" + bytes(rel) + os.readlink(path).encode())
        elif path.is_file():
            digest.update(b"F" + bytes(rel))
            digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()

head0 = _git("rev-parse", "HEAD")

# Phase A: NO marker at all -> typed refusal, nothing moved.
ok, msg = rollback_managed_update("e2e_w2_absent_probe")
report["absent"] = {"ok": ok, "msg": msg, "head_moved": _git("rev-parse", "HEAD") != head0}

# Phase A2: explicit null stamp -> corrupt -> the same typed pre_update_sha refusal,
# marker bytes untouched.
marker.write_text(json.dumps({"_schema_version": None, "pre_update_sha": head0,
                              "phase": "pending_boot_smoke"}), encoding="utf-8")
null_before = marker.read_bytes()
ok, msg = rollback_managed_update("e2e_w2_null_probe")
report["null_stamp"] = {
    "ok": ok, "msg": msg,
    "strict_status": read_update_tx_strict()[0],
    "bytes_intact": marker.read_bytes() == null_before,
    "head_moved": _git("rev-parse", "HEAD") != head0,
}

# Phase B: FUTURE-schema marker -> typed newer-version refusal from BOTH the
# rollback and the boot finalizer; the raw marker stays byte-identical.
marker.write_text(json.dumps({"_schema_version": 2, "pre_update_sha": head0,
                              "phase": "pending_boot_smoke"}), encoding="utf-8")
future_before = marker.read_bytes()
ok, msg = rollback_managed_update("e2e_w2_future_probe")
boot = finalize_managed_update_on_boot(True)
report["future"] = {
    "ok": ok, "msg": msg, "boot": boot,
    "strict_status": read_update_tx_strict()[0],
    "bytes_intact": marker.read_bytes() == future_before,
    "head_moved": _git("rev-parse", "HEAD") != head0,
}
marker.unlink()

# Phase C: a half-applied update rolls back BYTE-FOR-BYTE. Snapshot the clean
# pre-update tree, land a fake update commit, dirty the tree on top, write the
# valid pending tx, roll back, and compare the full worktree fingerprint.
pre_sha = _git("rev-parse", "HEAD")
pre_fingerprint = _tree_fingerprint()
(clone / "BROKEN_UPDATE_CANDIDATE.txt").write_text("broken payload\n", encoding="utf-8")
_git("add", "BROKEN_UPDATE_CANDIDATE.txt")
_git("commit", "-q", "-m", "broken update candidate (system_e2e S10)")
target_sha = _git("rev-parse", "HEAD")
(clone / "junk_untracked.bin").write_bytes(b"junk")
readme = clone / "README.md"
readme.write_text(readme.read_text(encoding="utf-8") + "\ne2e-w2 stray edit\n", encoding="utf-8")
write_update_tx({
    "pre_update_sha": pre_sha, "pre_update_branch": "ouroboros",
    "base_sha": pre_sha, "target_sha": target_sha, "merge_commit": target_sha,
    "phase": "pending_boot_smoke", "pre_restart_smoke": "passed",
    "attempt_id": "e2ew2probe01", "local_work_carrier": "none", "stash_sha": "",
})
ok, msg = rollback_managed_update("e2e_w2_failed_update")
supervisor_log = data / "logs" / "supervisor.jsonl"
rolled_rows = []
if supervisor_log.exists():
    for line in supervisor_log.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if isinstance(row, dict) and row.get("type") == "managed_update_rolled_back":
            rolled_rows.append(row)
report["restore"] = {
    "ok": ok, "msg": msg,
    "head": _git("rev-parse", "HEAD"), "pre_sha": pre_sha,
    "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
    "porcelain": _git("status", "--porcelain"),
    "fingerprint_matches": _tree_fingerprint() == pre_fingerprint,
    "failed_ref": _git("branch", "--list", "failed-update-*"),
    "marker_gone": not marker.exists(),
    "rolled_back_rows": rolled_rows,
}
print(json.dumps(report))
'''


@pytest.mark.integration
@pytest.mark.serial
def test_s10_rollback_typed_refusals_and_byte_for_byte_restore(tmp_path_factory):
    require_lane(LANE_MOCK)
    root = tmp_path_factory.mktemp("s10")
    clone = clone_repo(root)
    data = pathlib.Path(root) / "data"
    (data / "logs").mkdir(parents=True, exist_ok=True)
    driver = pathlib.Path(root) / "s10_driver.py"
    driver.write_text(S10_DRIVER, encoding="utf-8")
    env = {
        **os.environ,
        "OUROBOROS_APP_ROOT": str(root),
        "OUROBOROS_REPO_DIR": str(clone),
        "OUROBOROS_DATA_DIR": str(data),
        "OUROBOROS_SETTINGS_PATH": str(data / "settings.json"),
        "PYTHONPATH": str(REPO_ROOT),
    }
    proc = subprocess.run(
        [sys.executable, str(driver), str(clone), str(data)],
        cwd=str(REPO_ROOT), env=env, capture_output=True, text=True, timeout=300,
    )
    assert proc.returncode == 0, (proc.stdout, proc.stderr)
    report = json.loads(proc.stdout.strip().splitlines()[-1])

    # A: absent marker — the typed pre_update_sha refusal, zero movement.
    absent = report["absent"]
    assert absent["ok"] is False and "no pre_update_sha" in absent["msg"], absent
    assert absent["head_moved"] is False, absent

    # A2: explicit null stamp is CORRUPT (never the legacy unstamped form): same
    # typed refusal, evidence bytes untouched.
    null_stamp = report["null_stamp"]
    assert null_stamp["ok"] is False and "no pre_update_sha" in null_stamp["msg"], null_stamp
    assert null_stamp["strict_status"] == "corrupt", null_stamp
    assert null_stamp["bytes_intact"] is True and null_stamp["head_moved"] is False, null_stamp

    # B: FUTURE marker — the newer-version refusal from rollback AND boot
    # finalize, raw marker byte-identical (left for the owner).
    future = report["future"]
    assert future["ok"] is False and "newer version" in future["msg"], future
    assert future["strict_status"] == "future", future
    assert future["boot"].get("finalized") is False, future
    assert "newer version" in str(future["boot"].get("reason") or ""), future
    assert future["bytes_intact"] is True and future["head_moved"] is False, future

    # C: the failed update rolled back byte-for-byte; candidate preserved;
    # durable receipt written; marker cleared.
    restore = report["restore"]
    assert restore["ok"] is True, restore
    assert restore["head"] == restore["pre_sha"], restore
    assert restore["branch"] == "ouroboros", restore
    assert restore["porcelain"] == "", restore
    assert restore["fingerprint_matches"] is True, "worktree is not byte-identical after rollback"
    assert "failed-update-" in restore["failed_ref"], restore
    assert restore["marker_gone"] is True, restore
    rows = restore["rolled_back_rows"]
    assert rows and rows[-1].get("pre_update_sha") == restore["pre_sha"], rows
