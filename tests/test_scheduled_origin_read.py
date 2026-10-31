"""#1488 (owner 13 Oct, option A): scheduled work reads its DIRECT origin's materials, read-only.

Related and independent follow-ups alike: one hop, the origin's ``task_drive`` and
``artifact_store`` — not its money, deadline, controls, candidate or Git paths, not its own
predecessor, never a write. The authority is the host-written occurrence receipt
(``schedule_admission``) of the actor's root; the root is the queue row's own host-set lineage
(``/api/tasks`` refuses or strips caller lineage, ``/api/schedules`` refuses or ignores caller
provenance), and task metadata an ordinary admission accepted proves nothing. Real producers: the
``schedule_followup`` tool, the schedule table, occurrence admission (``check_scheduled_tasks``), the
real worker start that builds the ToolContext (``agent_start``), the ``schedule_subagent`` tool and
its supervisor admission, a stored pre-upgrade timeout retry (``tests._silence_reaper_support``; no
producer mints one now), both public APIs, the tool registry, ``write_file`` artifact registration,
terminal-file settlement and the drive-retention prune. Missing scratch after retention is an honest
not-found; registered artifacts remain; nothing new is retained.
"""
from __future__ import annotations

import pathlib
import time
from types import SimpleNamespace

import pytest

from ouroboros.artifacts import task_artifact_dir_path
from ouroboros.task_results import load_task_result, write_task_result
from ouroboros.tool_access import lineage_task_ids, scheduled_origin
from ouroboros.tools.followup import _handle_schedule_followup
from ouroboros.tools.registry import ToolContext, ToolRegistry
from supervisor import queue, workers
from tests._execution_clock_seams import agent_start
from tests.test_predecessor_lineage_read import predecessor_envelope
from tests.test_schedule_followup import _ctx
from tests.test_swarm_host_admission import host  # noqa: F401

pytestmark = pytest.mark.serial
ORIGIN, GRAND, STRANGER = "root-1", "pred-0", "other-1"
BINDING = {"billing_group_id": ORIGIN, "billing_group_limit_usd": 10.0,
           "billing_group_limit_source": "initial_task_admission", "billing_group_limit_revision": "rev-1"}


@pytest.fixture
def origin(host, monkeypatch, tmp_path):  # noqa: F811
    ctx = _ctx(tmp_path)  # the origin task's own tool context, on the host's data root
    assert ctx.drive_root == host.root
    ctx.current_chat_id = 0
    ctx.task_metadata["resource_intent"] = {"kind": "system_repo"}
    monkeypatch.setenv("TOTAL_BUDGET", "1000")
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "advanced")
    monkeypatch.setenv("OUROBOROS_SAFETY_MODE", "off")
    monkeypatch.setenv("OUROBOROS_CONSCIOUSNESS_SAFETY_MODE", "off")
    monkeypatch.setattr("ouroboros.config.get_bg_wakeup_min_sec", lambda: 0)
    write_task_result(host.root, ORIGIN, "running", root_task_id=ORIGIN, billing_group=BINDING)
    for task_id, name, text in ((ORIGIN, "report.md", "ORIGIN_DRIVE"), (GRAND, "notes.md", "GRAND_DRIVE"),
                                (STRANGER, "notes.md", "STRANGER_DRIVE")):
        (host.root / "task_drives" / task_id).mkdir(parents=True)
        (host.root / "task_drives" / task_id / name).write_text(text + "\n", encoding="utf-8")
    (task_artifact_dir_path(host.root, ORIGIN, create=True) / "report.txt").write_text("ORIGIN_ARTIFACT\n")
    return ctx


def _fire(host, ctx, relation):  # noqa: F811
    out = _handle_schedule_followup(ctx, relation=relation, run_at="2000-01-01T00:00:00+00:00",
                                    objective="Check this week's data against the report")
    assert out.startswith("FOLLOWUP_SCHEDULED"), out
    queue.check_scheduled_tasks()
    [task] = [row for row in host.pending if (row.get("metadata") or {}).get("schedule_occurrence")]
    return task


def _started_registry(host, task, monkeypatch):  # noqa: F811
    """The ToolContext the real worker start builds for ``task`` (metadata, contract, drives)."""
    with monkeypatch.context() as patched:
        ctx = agent_start(task, host.root, patched).tool_context
    registry = ToolRegistry(repo_dir=ctx.repo_dir, drive_root=ctx.drive_root)
    registry.set_context(ctx)
    return registry, ctx


def _read(registry, root, path):
    return registry.execute("read_file", {"root": root, "path": str(path)})


@pytest.mark.parametrize("relation", ["related", "independent"])
def test_scheduled_work_reads_its_direct_origin_and_nothing_else(host, origin, monkeypatch, relation):  # noqa: F811
    task = _fire(host, origin, relation)
    registry, ctx = _started_registry(host, task, monkeypatch)
    assert lineage_task_ids(ctx) == (task["id"], ORIGIN)
    assert "ORIGIN_DRIVE" in _read(registry, "task_drive", host.root / "task_drives" / ORIGIN / "report.md")
    artifact = task_artifact_dir_path(host.root, ORIGIN) / "report.txt"
    assert "ORIGIN_ARTIFACT" in _read(registry, "artifact_store", artifact)
    for other in (GRAND, STRANGER):
        refused = _read(registry, "task_drive", host.root / "task_drives" / other / "notes.md")
        assert "outside selected root=task_drive" in refused and "_DRIVE" not in refused
    write = registry.execute("write_file", {"root": "task_drive", "path": str(
        host.root / "task_drives" / ORIGIN / "report.md"), "content": "overwrite"})
    assert write.startswith("⚠️") and "outside selected root=task_drive" in write
    assert (host.root / "task_drives" / ORIGIN / "report.md").read_text() == "ORIGIN_DRIVE\n"
    # A read grant only: no predecessor envelope, no candidate/Git adoption, no copied money.
    assert "predecessor_authority" not in ctx.task_contract and "predecessor_authority_source" not in task
    group = (task.get("metadata") or {}).get("billing_group") or {}
    assert group.get("billing_group_id") == (ORIGIN if relation == "related" else task["id"])


def test_a_folderless_helper_start_stages_the_origin_materials_as_copies(host, origin, monkeypatch):  # noqa: F811
    """The disclosed cost of the read grant: a folderless readonly helper gets byte-identical
    copies of every permitted lineage file, the origin's whole drive and artifacts included."""
    import json

    from ouroboros.delegate_readonly_inputs import prepare_folderless_inputs

    task = _fire(host, origin, "independent")
    _registry, ctx = _started_registry(host, task, monkeypatch)
    ctx.task_metadata["resource_intent"] = {"kind": "explicit_none"}  # folderless: no workspace
    staged, _note = prepare_folderless_inputs(ctx, "inv-origin")
    manifest = json.loads((pathlib.Path(staged) / "inputs.json").read_text(encoding="utf-8"))
    copied = {pathlib.Path(row["source"]).name: (pathlib.Path(staged) / row["local"]).read_text()
              for row in manifest}
    assert copied == {"report.md": "ORIGIN_DRIVE\n", "report.txt": "ORIGIN_ARTIFACT\n"}, manifest
    assert all(row["sha256"] == row["source_sha256"] for row in manifest)


def test_folderless_origin_fifo_is_refused_before_open_and_regular_reads_survive(host, origin, monkeypatch):  # noqa: F811
    import json
    import os
    from ouroboros import store
    from ouroboros.delegate_readonly_inputs import prepare_folderless_inputs

    if not hasattr(os, "mkfifo"):
        pytest.skip("real FIFO requires POSIX")
    task = _fire(host, origin, "independent")
    registry, ctx = _started_registry(host, task, monkeypatch)
    ctx.task_metadata["resource_intent"] = {"kind": "explicit_none"}
    fifo = host.root / "task_drives" / ORIGIN / "stream.fifo"
    os.mkfifo(fifo)
    opened, real_open = [], store.open_for_read

    def no_nonregular_open(path):
        opened.append(pathlib.Path(path))
        assert pathlib.Path(path) != fifo, "opening a FIFO before validation can hang the start"
        return real_open(path)

    monkeypatch.setattr(store, "open_for_read", no_nonregular_open)
    try:
        with pytest.raises(OSError, match="not a regular file"):
            prepare_folderless_inputs(ctx, "inv-fifo")
        assert fifo not in opened
        assert "ORIGIN_DRIVE" in _read(registry, "task_drive", fifo.parent / "report.md")
    finally:
        fifo.unlink()
    staged, _note = prepare_folderless_inputs(ctx, "inv-readable")
    manifest = json.loads((pathlib.Path(staged) / "inputs.json").read_text())
    assert {pathlib.Path(row["source"]).name for row in manifest} == {"report.md", "report.txt"}
    assert all(row["sha256"] == row["source_sha256"] for row in manifest)
    assert fifo.parent / "report.md" in opened


def test_an_unreadable_receipt_denies_now_and_the_same_actor_reads_it_again(host, origin, monkeypatch):  # noqa: F811
    """A transient fault is not a permanent absence: nothing is remembered from it."""
    import sys

    from ouroboros import task_results

    task = _fire(host, origin, "independent")
    registry, ctx = _started_registry(host, task, monkeypatch)
    real, faults = task_results.load_task_result, []

    def flaky(root, task_id, *args, **kwargs):  # only the origin receipt's read, only once
        if sys._getframe(1).f_code.co_name == "scheduled_origin" and not faults:
            faults.append(task_id)
            raise OSError("receipt temporarily unreadable")
        return real(root, task_id, *args, **kwargs)

    monkeypatch.setattr(task_results, "load_task_result", flaky)
    report = host.root / "task_drives" / ORIGIN / "report.md"
    refused = _read(registry, "task_drive", report)
    assert faults and "ORIGIN_DRIVE" not in refused and "outside selected root=task_drive" in refused
    assert getattr(ctx, "_scheduled_origin", None) is None, "unknown was not memoized as absence"
    assert "ORIGIN_DRIVE" in _read(registry, "task_drive", report), "the same live actor reads it once readable"
    assert ctx._scheduled_origin == (task["id"], ORIGIN)
    # Recovery widens nothing: still one hop, still read-only.
    assert "GRAND_DRIVE" not in _read(registry, "task_drive", host.root / "task_drives" / GRAND / "notes.md")
    write = registry.execute("write_file", {"root": "task_drive", "path": str(report), "content": "overwrite"})
    assert write.startswith("⚠️") and report.read_text() == "ORIGIN_DRIVE\n"


def test_its_admitted_child_and_its_retry_read_the_same_origin_one_hop(host, origin, monkeypatch):  # noqa: F811
    from ouroboros.tools.control import _schedule_task
    from supervisor import events
    from tests._shared import configure_test_subagent
    from tests._silence_reaper_support import historical_successor

    task = _fire(host, origin, "independent")
    _registry, ctx = _started_registry(host, task, monkeypatch)
    queued = _schedule_task(ctx, subagent_id=configure_test_subagent(monkeypatch), memory_mode="empty",
                            objective="Read the origin report", expected_output="facts")
    assert queued.startswith("Subagent request queued"), queued
    [event] = [e for e in ctx.pending_events if e.get("type") == "schedule_subagent"]
    events._handle_schedule_task(event, host.ctx)
    [child] = [t for t in host.pending if t.get("parent_task_id") == task["id"]]
    assert child["delegation_role"] == "subagent" and child["root_task_id"] == task["id"]
    host.pending.remove(task)
    write_task_result(host.root, task["id"], "running", root_task_id=task["id"])
    historical_successor(dict(task), 1, "retry-1")
    [retry] = [t for t in host.pending if t["id"] == "retry-1"]
    for row in (child, retry):
        registry, started = _started_registry(host, row, monkeypatch)
        assert started.task_metadata["root_task_id"] == task["id"], "lineage the host recorded"
        assert lineage_task_ids(started) == (row["id"], task["id"], ORIGIN), "one hop, no ancestor before it"
        assert "ORIGIN_DRIVE" in _read(registry, "task_drive", host.root / "task_drives" / ORIGIN / "report.md")
        assert "GRAND_DRIVE" not in _read(registry, "task_drive", host.root / "task_drives" / GRAND / "notes.md")


def test_task_metadata_and_a_foreign_receipt_prove_no_origin(host, origin, monkeypatch):  # noqa: F811
    from starlette.applications import Starlette
    from starlette.requests import Request
    from starlette.routing import Route
    from starlette.testclient import TestClient

    from ouroboros.gateway import schedules
    from ouroboros.gateway.tasks import _create_task_from_body

    scheduled = _fire(host, origin, "independent")  # an accepted scheduled root whose receipt names ORIGIN
    request = Request({"type": "http", "app": SimpleNamespace(state=SimpleNamespace(
        drive_root=host.root, repo_dir=workers.REPO_DIR))})
    # Its root is not a caller's to name: refused at the top level, stripped from metadata.
    refused = _create_task_from_body(request, {"task_id": "forged0", "description": "x",
                                               "root_task_id": scheduled["id"]})
    assert refused.status_code == 400
    claimed = _create_task_from_body(request, {"task_id": "claim1", "description": "x", "metadata": {
        "root_task_id": scheduled["id"], "parent_task_id": scheduled["id"]}})
    assert claimed.status_code == 200
    [claim] = [row for row in host.pending if row["id"] == "claim1"]
    assert (claim["root_task_id"], claim["metadata"]["root_task_id"]) == ("claim1", "claim1")
    registry, ctx = _started_registry(host, claim, monkeypatch)
    assert lineage_task_ids(ctx) == ("claim1",)
    assert "ORIGIN_DRIVE" not in _read(registry, "task_drive", host.root / "task_drives" / ORIGIN / "report.md")
    response = _create_task_from_body(request, {"task_id": "forged1", "description": "x", "metadata": {
        "followup_origin": {"task_id": ORIGIN, "root_task_id": ORIGIN}}})
    assert response.status_code == 200
    [forged] = [row for row in host.pending if row["id"] == "forged1"]
    assert forged["metadata"]["followup_origin"]["task_id"] == ORIGIN  # the ingress accepted it
    registry, ctx = _started_registry(host, forged, monkeypatch)
    assert lineage_task_ids(ctx) == ("forged1",)
    assert "ORIGIN_DRIVE" not in _read(registry, "task_drive", host.root / "task_drives" / ORIGIN / "report.md")
    # A receipt naming another task, or not an accepted occurrence, binds nothing either.
    for receipt in ({"status": "accepted", "token": "t", "task": {"id": "elsewhere", "metadata": {
                        "followup_origin": {"task_id": ORIGIN}}}},
                    {"status": "refused", "token": "t", "task": {"id": "foreign1", "metadata": {
                        "followup_origin": {"task_id": ORIGIN}}}}):
        write_task_result(host.root, "foreign1", "scheduled", schedule_admission=receipt)
        probe = ToolContext(repo_dir=origin.repo_dir, drive_root=host.root, task_id="foreign1")
        assert scheduled_origin(probe) == ""
    # The owner's schedule door refuses template provenance and ignores body provenance.
    monkeypatch.setattr(schedules, "request_drive_root", lambda _: host.root)
    once = {"type": "once", "run_at": "2000-01-01T00:00:00Z"}
    with TestClient(Starlette(routes=[Route("/api/schedules", schedules.api_schedules_upsert,
                                            methods=["POST"])])) as client:
        for meta in ({"followup_origin": {"task_id": ORIGIN, "root_task_id": ORIGIN}}, {"origin_task_id": ORIGIN}):
            response = client.post("/api/schedules", json={"id": "t1", "trigger": once, "task": {
                "type": "task", "text": "t", "chat_id": 1, "metadata": meta}})
            assert response.status_code == 400, response.text
        response = client.post("/api/schedules", json={
            "id": "s3", "source": "task_followup", "followup_origin": {"task_id": ORIGIN, "root_task_id": ORIGIN},
            "trigger": once, "task": {"type": "task", "text": "t", "chat_id": 1}})
        assert response.status_code == 200, response.text
    before = {row["id"] for row in host.pending}
    queue.check_scheduled_tasks()
    [owned] = [row for row in host.pending if row["id"] not in before]
    receipt = load_task_result(host.root, owned["id"])["schedule_admission"]
    assert receipt["status"] == "accepted" and not receipt["task"]["metadata"].get("followup_origin")
    registry, ctx = _started_registry(host, owned, monkeypatch)
    assert lineage_task_ids(ctx) == (owned["id"],)
    assert "ORIGIN_DRIVE" not in _read(registry, "task_drive", host.root / "task_drives" / ORIGIN / "report.md")


def test_a_presence_follow_up_keeps_its_ceiling_and_never_inherits_a_second_hop(host, origin, monkeypatch):  # noqa: F811
    from ouroboros.presence_authority import (
        PresenceCapabilityCeiling,
        PresenceResourceGrant,
        PresenceToolGrant,
        presence_ceiling_payload,
    )

    ceiling = PresenceCapabilityCeiling(
        skill_name="community-helper", skill_content_hash="a" * 64, profile_fingerprint="b" * 64,
        state_fingerprint="c" * 64, selection_fingerprint="d" * 64, model_slot="main", inline_max_rounds=10,
        tool_grants=(PresenceToolGrant("read_file"),),
        resource_grants=(PresenceResourceGrant(root="user_files", operations=("read",), path_prefix="shared"),),
        digest="e" * 64)
    origin.task_metadata["presence"] = {"binding_id": "b" * 32, "conversation_key": "telegram:account:chat:0"}
    # The origin is itself a Continue of GRAND: its own predecessor is not the follow-up's.
    origin.task_contract = {"objective": "presence turn", "capability_ceiling": presence_ceiling_payload(ceiling),
                            "predecessor_authority": predecessor_envelope(GRAND)}
    out = _handle_schedule_followup(origin, relation="independent", run_at="2000-01-01T00:00:00+00:00",
                                    objective="Revisit")
    assert out.startswith("FOLLOWUP_SCHEDULED"), out
    record = queue.list_scheduled_tasks(host.root)["tasks"][-1]
    assert sorted(record["task"]["task_contract"]) == ["capability_ceiling"]
    queue.check_scheduled_tasks()
    [task] = [row for row in host.pending if (row.get("metadata") or {}).get("schedule_occurrence")]
    registry, ctx = _started_registry(host, task, monkeypatch)
    assert "predecessor_authority" not in ctx.task_contract
    assert lineage_task_ids(ctx) == (task["id"], ORIGIN)
    # The origin is in its lineage, but the positive ceiling still decides what it may open.
    blocked = _read(registry, "task_drive", host.root / "task_drives" / ORIGIN / "report.md")
    assert "ORIGIN_DRIVE" not in blocked and blocked.startswith("⚠️")


def test_retention_after_the_origin_settles_is_honest(host, origin, monkeypatch):  # noqa: F811
    from ouroboros.artifacts import registered_task_artifact
    from ouroboros.headless import prepare_terminal_task_files, prune_task_drives

    writer = ToolRegistry(repo_dir=origin.repo_dir, drive_root=origin.drive_root)
    writer.set_context(origin)
    deliverable = pathlib.Path.home() / "origin-report.md"
    wrote = writer.execute("write_file", {"root": "user_files", "path": str(deliverable),
                                          "content": "ORIGIN_DELIVERABLE\n"})
    assert wrote.startswith("OK"), wrote
    registered = registered_task_artifact(host.root, ORIGIN, deliverable.name)
    assert registered and registered["kind"] == "user_file", "the origin's write registered it"
    task = _fire(host, origin, "independent")
    registry, _ctx_ = _started_registry(host, task, monkeypatch)
    drive_file = host.root / "task_drives" / ORIGIN / "report.md"
    assert "ORIGIN_DRIVE" in _read(registry, "task_drive", drive_file)
    old = time.time() - 8 * 86400
    write_task_result(host.root, ORIGIN, "completed", result="done",
                      ts=time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime(old)))
    settled = prepare_terminal_task_files(host.root, {"id": ORIGIN, "root_task_id": ORIGIN})
    assert settled["terminal_source_present"] is True and not settled["error"], settled
    report = prune_task_drives(host.root, retention_days=7, now=time.time(), live=lambda _task: False)
    assert ORIGIN in [item["task_id"] for item in report["pruned"]]
    gone = _read(registry, "task_drive", drive_file)
    assert "ORIGIN_DRIVE" not in gone and "outside selected root" not in gone, gone  # not found, not forbidden
    assert registered_task_artifact(host.root, ORIGIN, deliverable.name) == registered
    assert "ORIGIN_DELIVERABLE" in _read(registry, "artifact_store", pathlib.Path(registered["path"])), \
        "a registered artifact survives settlement and the drive prune"
