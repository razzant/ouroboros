"""#1573: a delegated run's outcome and full answer stay discoverable after recovery.

Consumer checks over the real seams, not helper calls alone:

* the terminal payload too large for the tool budget is retained ONCE in the
  ACTUAL staging task's canonical source store and read back through the real
  ``ToolRegistry`` ``read_file`` with the address the delivery advertises, on every
  drive geometry the staging task runs on (canonical, forked top-level, readonly
  and acting children), while the D7 acknowledgement keeps its exact-source,
  coverage and owner/retry rules;
* readers outside that lineage (a forked parent of the nanny, a forked root
  Continue whose old nanny-child staged the source) read the same bytes only
  through ``get_task_result(delegated_output_sha256=...)``, bound to the ref the
  owner's task result publishes, after the child drive is gone — and never earn
  a receipt;
* the owner's own settled run that its current attempt's transcript never took
  is one health line, cleared by the post-append terminal acknowledgement of THAT
  attempt, back after a crash retry, and carried through a duplicate STARTED;
* engine locators ride the settlement and receipts as engine-owned metadata,
  never text; rows settled before these facts existed stay byte-identical.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import shutil
from types import SimpleNamespace

import pytest

from ouroboros import delegate_custody as custody
from ouroboros.artifacts import task_artifact_dir_path
from ouroboros.contracts.task_constraint import TaskConstraint
from ouroboros.tool_capabilities import tool_result_limit
from ouroboros.tools.registry import ToolContext, ToolRegistry
from tests._delegated_transport_shared import (  # noqa: F401  (autouse fixture applies on import)
    _LiveRunStub,
    _owned_gateway_uses_each_test_transport,
)

ROOT = "r5173b7c3c15d4c0b"
NANNY = "c11ae4fd0aa111111"
CONT = "s2f0a1b2c3d4e5f60"
ANSWER = "complete answer line\n" * 6000  # far above the delegate_wait budget


class _Engine(_LiveRunStub):
    """A finished run the way the engine reports it: run folder plus primary-output metadata."""

    def __init__(self, text=ANSWER, state="succeeded", reason="", run_dir=True, primary=True):
        super().__init__(state=state)
        self.text, self.reason, self.run_dir, self.primary = text, reason, run_dir, primary

    def get_run(self, rid, *, timeout_sec=None):
        summary = {"state": self.state, "spendUsd": 0.0, "effectiveAccess": "readonly"}
        if self.run_dir:
            summary["runDir"] = f"/engine/runs/{rid}"
        if self.reason:
            summary["outcomeFacts"] = {"reason": self.reason}
        detail = {"lastSeq": 9, "summary": summary}
        if self.primary:
            detail["primaryOutput"] = {"kind": "answer", "path": "final/answer.md", "text": self.text,
                                       "bytes": len(self.text.encode("utf-8")), "truncated": False}
        return detail


@pytest.fixture
def geo(tmp_path, monkeypatch):
    """Canonical data root, per-task headless drives and a fake home (the lineage-read geometry)."""
    home, repo, canonical = tmp_path / "home", tmp_path / "repo", tmp_path / "data"
    for path in (home, repo, canonical):
        path.mkdir(parents=True)
    monkeypatch.setattr(pathlib.Path, "home", lambda: home)
    monkeypatch.setenv("OUROBOROS_USER_FILES_ROOT", str(home))
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "advanced")
    monkeypatch.setenv("OUROBOROS_SAFETY_MODE", "off")
    monkeypatch.setenv("OUROBOROS_CONSCIOUSNESS_SAFETY_MODE", "off")
    from ouroboros.tools import delegate

    delegate._READ_COVERAGE.clear()
    yield SimpleNamespace(tmp=tmp_path, home=home, repo=repo, canonical=canonical)
    delegate._READ_COVERAGE.clear()


def _drive(geo, task_id):
    path = geo.tmp / "state" / "headless_tasks" / task_id / "data"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _predecessor(task_id):
    return {"kind": "bounded_continuation_envelope", "status": "done", "previous_task_id": "",
            "source": {"kind": "task_result", "task_id": task_id, "human_label": f"task {task_id}",
                       "tool": "get_task_result",
                       "arguments": {"task_id": task_id, "include_authority": True}}}


def _top(geo, task_id, *, forked=False, predecessor=""):
    drive = _drive(geo, task_id) if forked else geo.canonical
    ctx = ToolContext(repo_dir=geo.repo, drive_root=drive, task_id=task_id,
                      **({"memory_mode": "forked", "budget_drive_root": str(geo.canonical)} if forked else {}))
    ctx.task_metadata = ({"memory_mode": "forked", "budget_drive_root": str(geo.canonical)} if forked else {})
    if predecessor:
        ctx.task_contract = {"predecessor_authority": _predecessor(predecessor)}
    return ctx


def _child(geo, task_id, parent, *, acting=False):
    from ouroboros.task_results import load_task_result, write_task_result
    if load_task_result(geo.canonical, parent) is None:  # a child's root exists before it runs
        write_task_result(geo.canonical, parent, "running", root_task_id=parent)
    ctx = ToolContext(repo_dir=geo.repo, drive_root=_drive(geo, task_id), task_id=task_id)
    ctx.budget_drive_root = str(geo.canonical)
    ctx.task_metadata = {"delegation_role": "subagent", "parent_task_id": parent, "root_task_id": parent,
                         "budget_drive_root": str(geo.canonical)}
    if acting:
        worktree = geo.tmp / f"wt-{task_id}"
        worktree.mkdir(exist_ok=True)
        ctx.task_constraint = TaskConstraint(mode="acting_subagent", surface="self_worktree",
                                             write_root=str(worktree))
        ctx.workspace_root, ctx.workspace_mode = str(worktree), "self_worktree"
    else:
        ctx.task_constraint = TaskConstraint(mode="local_readonly_subagent", allow_enable=False)
    return ctx


def _execute(ctx, tool, **arguments):
    registry = ToolRegistry(repo_dir=ctx.repo_dir, drive_root=ctx.drive_root)
    registry.set_context(ctx)
    return registry.execute_result(tool, arguments)


def _stage(geo, ctx, monkeypatch, run_id="run-1", engine=None):
    """The nanny's real terminal path: settle, retain, deliver (``_delegate_wait``)."""
    from ouroboros.gateways import claudexor as gw
    from ouroboros.tools import delegate

    engine = engine or _Engine()
    monkeypatch.setattr(gw, "ClaudexorGateway", lambda *a, **k: engine)
    custody._CUSTODY.clear()
    assert custody.record_started(geo.canonical, custody.RunCustody(
        run_id=run_id, task_id=ctx.task_id, route_id="r", model="m", project_id="p",
        project_owned=False, root_task_id=ctx.task_metadata.get("root_task_id") or ctx.task_id,
        ledger_root=str(geo.canonical)))
    out = json.loads(delegate._delegate_wait(ctx, run_id, wait_sec=1))
    custody._CUSTODY.clear()
    return out


def _read_whole(ctx, read_next):
    """Cover the artifact contiguously through the real registry, like a model would."""
    arguments = {key: read_next[key] for key in ("root", "path")}
    stride = tool_result_limit("read_file") - 5_000
    text = pathlib.Path(read_next["path"]).read_text(encoding="utf-8")
    for number, line in enumerate(text.splitlines(keepends=True), start=1):
        for offset in range(0, max(len(line), 1), stride):
            result = _execute(ctx, "read_file", **arguments, start_line=number, max_lines=1, start_char=offset)
            assert result.status == "ok" and not result.text.startswith("⚠️"), result.text[:300]


def _receipts(geo):
    return [row for row in custody.custody_rows(geo.canonical) if row["type"] == custody.OUTPUT_CONSUMED]


def _publish_terminal_result(geo, task_id, status="cancelled"):
    """The owner's terminal boundary: the real audit producer, then the real recorder."""
    from ouroboros.delegate_terminal import record_terminal_reconciliation, terminal_reconcile_task
    from ouroboros.task_results import load_task_result, write_task_result

    write_task_result(geo.canonical, task_id, status, result="stopped", reason_code="server_shutdown")
    audit = terminal_reconcile_task(geo.canonical, task_id)
    assert audit["audit_status"] == "ok", audit
    record_terminal_reconciliation(geo.canonical, task_id, audit)
    # Output retention may already have published this exact disclosure. Its
    # idempotent recorder need not write twice; the actual published facts must match.
    stored = load_task_result(geo.canonical, task_id)
    assert stored["delegated_runs_unreconciled"] == audit["unreconciled"]
    published = stored["delegate_terminal_reconciliation"]
    for field in ("open_run_ids", "pending_invocation_ids", "undisposed_patch_run_ids",
                  "terminal_runs", "deferred_project_retirements"):
        assert published[field] == audit[field]
    return audit


# --- the staging task reads its own retained answer on every drive -----------------

@pytest.mark.parametrize("geometry", ["canonical_top", "forked_top", "readonly_child", "acting_child"])
def test_the_staging_task_reads_its_retained_output_on_every_drive(geo, monkeypatch, geometry):
    ctx = {
        "canonical_top": lambda: _top(geo, ROOT),
        "forked_top": lambda: _top(geo, CONT, forked=True, predecessor=ROOT),
        "readonly_child": lambda: _child(geo, NANNY, ROOT),
        "acting_child": lambda: _child(geo, NANNY, ROOT, acting=True),
    }[geometry]()
    out = _stage(geo, ctx, monkeypatch)
    delivery = out["output_delivery"]
    read_next, artifact = delivery["read_next"], delivery["artifact"]
    canonical_file = task_artifact_dir_path(geo.canonical, ctx.task_id) / artifact["source"]["path"]
    assert read_next["root"] == "artifact_store" and pathlib.Path(read_next["path"]) == canonical_file
    assert artifact["source"]["task_id"] == ctx.task_id
    assert hashlib.sha256(canonical_file.read_bytes()).hexdigest() == artifact["sha256"]
    assert not (pathlib.Path(ctx.drive_root) / "task_drives" / ctx.task_id / "delegated_runs").exists(), \
        "the terminal payload is no longer staged on the (prunable) execution drive"
    assert "read_file arguments in output_delivery.read_next" in out["result_not_collected"]

    _read_whole(ctx, read_next)
    rows = _receipts(geo)
    assert len(rows) == 1 and rows[0]["reader_task_id"] == ctx.task_id
    assert rows[0]["sha256"] == artifact["sha256"] and rows[0]["artifact"] == artifact["source"]["path"]
    assert custody.settled_unread_outputs(geo.canonical) == []


def test_unrelated_reads_never_touch_custody_and_a_name_alone_is_no_evidence(geo, monkeypatch):
    ctx = _top(geo, ROOT)
    read_next = _stage(geo, ctx, monkeypatch)["output_delivery"]["read_next"]
    replays = []
    real_replay = custody.replay
    monkeypatch.setattr(custody, "replay", lambda *a, **k: replays.append(1) or real_replay(*a, **k))
    (geo.canonical / "notes.txt").write_text("plain\n", encoding="utf-8")
    assert _execute(ctx, "read_file", root="runtime_data", path="notes.txt").status == "ok"
    assert replays == [], "an unrelated runtime read must not replay custody"
    # The SAME bytes under another delegated-run name: a name and equal text are no
    # evidence; only the exact file the custody row recorded earns the receipt.
    copy = pathlib.Path(read_next["path"]).with_name("delegated-run-run-1-copy.json")
    shutil.copyfile(read_next["path"], copy)
    _read_whole(ctx, {"root": "artifact_store", "path": str(copy)})
    assert _receipts(geo) == []
    _read_whole(ctx, read_next)
    assert len(_receipts(geo)) == 1


def test_runtime_data_reads_of_the_exact_source_count_where_that_root_reaches_it(geo, monkeypatch):
    ctx = _top(geo, ROOT)  # the canonical drive IS this task's runtime_data root
    read_next = _stage(geo, ctx, monkeypatch)["output_delivery"]["read_next"]
    _read_whole(ctx, {**read_next, "root": "runtime_data"})
    assert [row["reader_task_id"] for row in _receipts(geo)] == [ROOT]


# --- readers outside the lineage: the task-result-bound exact source --------------

@pytest.mark.parametrize("parent_forked", [False, True])
def test_a_root_continue_reads_its_old_nanny_childs_answer_after_drive_cleanup(geo, monkeypatch, parent_forked):
    nanny = _child(geo, NANNY, ROOT)
    out = _stage(geo, nanny, monkeypatch)
    source = out["output_delivery"]["artifact"]["source"]
    canonical_file = task_artifact_dir_path(geo.canonical, NANNY) / source["path"]
    _publish_terminal_result(geo, NANNY)
    shutil.rmtree(geo.tmp / "state" / "headless_tasks" / NANNY)  # the child drive's GC
    assert canonical_file.exists(), "drive cleanup cannot take the retained answer"

    continuation = _top(geo, CONT, forked=True, predecessor=ROOT)
    parent = _top(geo, ROOT, forked=parent_forked)
    for reader in (continuation, parent):
        refused = _execute(reader, "read_file", root="artifact_store", path=str(canonical_file))
        assert refused.text.startswith("⚠️") and "outside selected root=artifact_store" in refused.text
        # Canonical runtime_data READ parity: an absolute path reaches the original from a
        # forked reader too; a relative one keeps the fork's own (here absent) bytes.
        via_runtime = _execute(reader, "read_file", root="runtime_data", path=str(canonical_file))
        assert via_runtime.status == "ok" and not via_runtime.text.startswith("⚠️"), via_runtime.text[:300]
        relative = str(canonical_file.relative_to(geo.canonical))
        local = _execute(reader, "read_file", root="runtime_data", path=relative)
        assert ("DATA_NOT_YET_CREATED" in local.text) is (pathlib.Path(reader.drive_root) != geo.canonical), local.text[:300]

    listed = _execute(continuation, "get_task_result", task_id=NANNY).text
    assert '"staged_output"' in listed and source["sha256"] in listed, "the receipt names the answer"
    summary = json.loads(listed.split("[SUBTASK_OUTCOME]\n", 1)[1].split("\n[/SUBTASK_OUTCOME]", 1)[0])
    public_receipt = next(row for row in summary["delegated_custody"]["terminal_runs"]
                          if row["run_id"] == "run-1")
    from ouroboros.task_results import load_task_result

    stored = load_task_result(geo.canonical, NANNY)["delegate_terminal_reconciliation"]["terminal_runs"]
    stored_receipt = next(row for row in stored if row["run_id"] == "run-1")
    stored_source = stored_receipt["staged_output"]
    assert public_receipt["staged_output"] == {
        **stored_source, "read": stored_source["source_read"], "direct_read": stored_source["read"]}
    assert public_receipt["staged_output"]["sha256"] == source["sha256"]
    assert public_receipt["staged_output"]["task_id"] == NANNY
    assert '"run_dir": "/engine/runs/run-1"' in listed, "the read tool shows the engine's own locator"
    first_reader = public_receipt["staged_output"]["read"]
    assert first_reader == public_receipt["staged_output"]["source_read"]
    sized = json.loads(_execute(continuation, first_reader["tool"],
                                **first_reader["arguments"]).text)["delegated_output_source"]
    full = canonical_file.read_bytes().decode("utf-8")
    assert sized["reason"] == "source_range_required" and sized["status"] == "available"
    assert sized["complete_chars"] == len(full)
    assert sized["complete_sha256"] == hashlib.sha256(full.encode("utf-8")).hexdigest()
    assert sized["staging_task_id"] == NANNY and sized["run_id"] == "run-1"
    assert "text" not in sized
    pieces, start, step = [], 0, 50_000
    while start < len(full):
        page = json.loads(_execute(continuation, "get_task_result", task_id=NANNY,
                                   delegated_output_sha256=source["sha256"],
                                   source_start_char=start,
                                   source_end_char=min(start + step, len(full))).text)
        pieces.append(page["delegated_output_source"]["text"])
        start += step
    assert "".join(pieces) == full
    assert _receipts(geo) == [], "discovery through the task result never discharges D7"


def _published_nanny_source(geo, monkeypatch, engine=None):
    nanny = _child(geo, NANNY, ROOT)
    out = _stage(geo, nanny, monkeypatch, engine=engine)
    _publish_terminal_result(geo, NANNY)
    return out["output_delivery"]["artifact"], _top(geo, CONT, forked=True, predecessor=ROOT)


def _selector(reader, sha, **ranges):
    result = _execute(reader, "get_task_result", task_id=NANNY, delegated_output_sha256=sha, **ranges)
    return result, json.loads(result.text)["delegated_output_source"]


def test_the_selector_is_an_exact_published_ref_and_every_gap_is_explicit(geo, monkeypatch):
    artifact, reader = _published_nanny_source(geo, monkeypatch)
    canonical_file = task_artifact_dir_path(geo.canonical, NANNY) / artifact["source"]["path"]

    _, unpublished = _selector(reader, "f" * 64)
    assert unpublished["status"] == "unavailable" and unpublished["reason"] == "source_not_published"
    _, invalid = _selector(reader, "not-a-digest")
    assert invalid["reason"] == "source_ref_invalid"
    bad_range, payload = _selector(reader, artifact["sha256"], source_start_char=5, source_end_char=2)
    assert bad_range.code == "TOOL_ARG_ERROR" and payload["reason"] == "source_range_invalid"
    assert "text" not in payload

    raw = canonical_file.read_bytes()
    canonical_file.write_bytes(raw.replace(b"complete", b"COMPLETE", 1))  # same size, other bytes
    _, tampered = _selector(reader, artifact["sha256"], source_start_char=0, source_end_char=10)
    assert tampered["status"] == "unavailable" and tampered["reason"] == "source_identity_mismatch"
    assert "text" not in tampered
    canonical_file.unlink()
    _, missing = _selector(reader, artifact["sha256"], source_start_char=0, source_end_char=10)
    assert missing["status"] == "unavailable" and missing["reason"] == "source_unavailable"


def test_a_retained_engine_preview_is_readable_but_never_reported_whole(geo, monkeypatch):
    from ouroboros.gateways import claudexor as gw

    class _Preview(_Engine):
        def get_run(self, rid, *, timeout_sec=None):
            detail = super().get_run(rid)
            detail["primaryOutput"].update(bytes=10 ** 7, truncated=True)
            return detail

        def get_run_artifact(self, rid, path):
            raise gw.ClaudexorUnavailable("http_404", "no such artifact", status_code=404)

    artifact, reader = _published_nanny_source(geo, monkeypatch, engine=_Preview())
    _, page = _selector(reader, artifact["sha256"], source_start_char=0, source_end_char=20)
    assert page["status"] == "incomplete" and page["source_complete"] is False
    assert page["text"]


def test_existing_read_rules_run_first_and_the_default_read_never_opens_the_source(geo, monkeypatch):
    from ouroboros import artifacts

    artifact, reader = _published_nanny_source(geo, monkeypatch)
    blocked = _execute(reader, "get_task_result", task_id=NANNY, presence_scope="own_binding",
                       delegated_output_sha256=artifact["sha256"])
    assert "PRESENCE_CAPABILITY_BLOCKED" in blocked.text and "delegated_output_source" not in blocked.text

    opened = []
    real = artifacts.read_actor_source_bytes
    monkeypatch.setattr(artifacts, "read_actor_source_bytes",
                        lambda *a, **k: opened.append(a[1]) or real(*a, **k))
    plain = _execute(reader, "get_task_result", task_id=NANNY)
    assert "staged_output" in plain.text and "delegated_output_source" not in plain.text
    assert NANNY not in opened, "the default read must not open or hash retained output bytes"
    known = next(line.split("=", 1)[1] for line in plain.text.splitlines()
                 if line.startswith("child_result_sha256="))
    explicit = _execute(reader, "get_task_result", task_id=NANNY, known_result_sha256=known,
                        delegated_output_sha256=artifact["sha256"])
    assert json.loads(explicit.text)["delegated_output_source"]["complete_chars"] > 0, \
        "an explicit selector is never swallowed by the unchanged-result shortcut"


@pytest.mark.serial  # retry authority reads a fresh queue snapshot (the retry suites' lane)
def test_a_retry_reader_stages_into_its_own_store_and_the_starter_receipt_names_it(tmp_path, monkeypatch):
    from ouroboros.delegate_terminal import _audit_task_custody, record_terminal_reconciliation
    from ouroboros.tools import delegate
    from tests.test_delegate_retry_consumers import (
        TerminalGateway,
        bind_gateway,
        call,
        durable_run,
        retry_context,
    )

    memo = {}  # the retry suites' isolation: a private custody memo, unrestricted modes
    monkeypatch.setattr(custody, "_CUSTODY", memo)
    monkeypatch.setattr(delegate, "_CUSTODY", memo)
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "advanced")
    monkeypatch.setenv("OUROBOROS_SAFETY_MODE", "off")
    monkeypatch.setenv("OUROBOROS_CONSCIOUSNESS_SAFETY_MODE", "off")

    ctx = retry_context(tmp_path, monkeypatch, split=True,
                        fixed_ownership_clock=True)  # retry-b on a forked drive
    held = durable_run(ctx, owner="retry-a")
    bind_gateway(monkeypatch, TerminalGateway(output="V" * 120_000))
    wait = call(ctx, "delegate_wait", run_id=held.run_id)
    artifact = json.loads(wait.text)["output_delivery"]["artifact"]
    data = custody.custody_root(ctx)
    assert artifact["source"]["task_id"] == "retry-b" != held.task_id

    audit = {"task_id": "retry-a", "trigger": "test", "outcomes": [], "unreconciled": [], "audit_status": "ok"}
    _audit_task_custody(data, "retry-a", audit)
    staged = audit["terminal_runs"][0]["staged_output"]
    assert staged["task_id"] == "retry-b" and staged["sha256"] == artifact["sha256"]
    assert staged["read"]["arguments"]["root"] == "artifact_store"
    assert pathlib.Path(staged["read"]["arguments"]["path"]) == pathlib.Path(artifact["path"])
    assert record_terminal_reconciliation(data, "retry-a", audit)

    summary = json.loads(call(ctx, "get_task_result", task_id="retry-a", view="summary").text)
    facts = summary["delegated_custody"]
    runs = [*facts.get("terminal_runs", []), *facts.get("predecessor_terminal_runs", [])]
    public = next(row["staged_output"] for row in runs if row["run_id"] == held.run_id)
    assert public["read"] == public["source_read"]
    assert public["read"]["arguments"]["task_id"] == "retry-a"
    assert public["direct_read"] == staged["read"]

    page = json.loads(call(ctx, "get_task_result", task_id="retry-a", delegated_output_sha256=artifact["sha256"],
                           source_start_char=0, source_end_char=40).text)["delegated_output_source"]
    assert page["staging_task_id"] == "retry-b" and page["status"] == "available"
    assert not [row for row in custody.custody_rows(data) if row["type"] == custody.OUTPUT_CONSUMED]
    # The successor's OWN full read of its staging is the reader-specific D7 receipt.
    _read_whole(ctx, {"root": artifact["root"], "path": artifact["path"]})
    rows = [row for row in custody.custody_rows(data) if row["type"] == custody.OUTPUT_CONSUMED]
    assert [(row["task_id"], row["reader_task_id"]) for row in rows] == [("retry-a", "retry-b")]


def test_a_legacy_task_drive_staging_still_reads_and_acknowledges(geo, monkeypatch):
    """Rows written before retention name ``delegated_runs/<run>.json`` on the owner's drive."""
    ctx = _top(geo, ROOT)
    custody._CUSTODY.clear()
    assert custody.record_started(geo.canonical, custody.RunCustody(
        run_id="run-old", task_id=ROOT, route_id="r", model="m", ledger_root=str(geo.canonical)))
    body = "legacy staged line\n" * 40
    legacy = geo.canonical / "task_drives" / ROOT / "delegated_runs" / "run-old.json"
    legacy.parent.mkdir(parents=True)
    legacy.write_bytes(body.encode("utf-8"))
    sha = hashlib.sha256(body.encode("utf-8")).hexdigest()
    assert custody.emit(geo.canonical, custody.OUTPUT_SPILLED, {
        "run_id": "run-old", "task_id": ROOT, "artifact": "delegated_runs/run-old.json",
        "staged": True, "full_content": True, "sha256": sha})
    assert custody.emit(geo.canonical, custody.SETTLED, {"run_id": "run-old", "task_id": ROOT, "state": "succeeded"})
    custody._CUSTODY.clear()  # a restarted worker: the durable rows are the authority
    entry = custody.replay(geo.canonical)["run-old"]
    assert entry.output_source == {} and entry.engine_locator is None
    from ouroboros.context import build_health_invariants

    env = SimpleNamespace(drive_root=geo.canonical, drive_path=lambda rel="": geo.canonical / rel,
                          repo_path=lambda rel="": geo.repo / rel)
    health = build_health_invariants(env, task_id=ROOT, attempt_key="1")
    assert ("staged at delegated_runs/run-old.json and never read to EOF (owner task "
            f"{ROOT}). Read it with read_file root='task_drive' until") in health
    assert "OWN DELEGATED RUN ENDED" not in health, "a pre-retention settlement proves no missed delivery"
    _read_whole(ctx, {"root": "task_drive", "path": str(legacy)})
    rows = _receipts(geo)
    assert len(rows) == 1 and rows[0]["artifact"] == "delegated_runs/run-old.json" and rows[0]["sha256"] == sha


def test_interaction_spills_keep_their_task_drive_contract(geo):
    from ouroboros.delegate_output import _stage_full_output

    ctx = _top(geo, ROOT)
    spill = _stage_full_output(ctx, "run-1", "questions\n", suffix=".abc.interactions")
    assert spill["root"] == "task_drive" and spill["path"] == "delegated_runs/run-1.abc.interactions.json"
    assert "source" not in spill


def test_a_failed_retention_is_partial_and_unrecoverable_never_a_false_artifact(geo, monkeypatch):
    from ouroboros import artifacts

    def _closed(*_args, **_kwargs):
        raise RuntimeError("publication generation closed")

    monkeypatch.setattr(artifacts, "store_actor_source_bytes", _closed)
    out = _stage(geo, _top(geo, ROOT), monkeypatch)
    delivery = out["output_delivery"]
    assert delivery["artifact"] is None and delivery["read_next"] is None
    assert "could not be retained" in delivery["note"]
    spilled = [row for row in custody.custody_rows(geo.canonical) if row["type"] == custody.OUTPUT_SPILLED]
    assert spilled[-1]["staged"] is False and "source" not in spilled[-1]
    assert custody.replay(geo.canonical)["run-1"].output_source == {}


# --- the owner's own settled run: notice until its terminal wake is appended -------

def _sup_ctx(tmp_path, task_id="root", attempt=1):
    return SimpleNamespace(task_id=task_id, task_attempt=attempt, drive_root=tmp_path,
                           budget_drive_root=str(tmp_path), task_metadata={"root_task_id": task_id})


def _health(tmp_path, task_id="root", attempt_key="1"):
    from ouroboros.context import build_health_invariants

    env = SimpleNamespace(drive_root=tmp_path, drive_path=lambda rel="": tmp_path / rel,
                          repo_path=lambda rel="": tmp_path / "repo" / rel)
    return build_health_invariants(env, task_id=task_id, attempt_key=attempt_key)


def _settle(tmp_path, run_id="run-1", task_id="root", engine=None, **fields):
    entry = custody.RunCustody(run_id=run_id, task_id=task_id, route_id="r", model="m",
                               ledger_root=str(tmp_path), root_task_id=task_id, **fields)
    assert custody.record_started(tmp_path, entry)
    engine = engine or _Engine(text="diagnostic", state="interrupted", reason="crash_interrupted")
    assert custody.settle_run(tmp_path, engine, entry, engine.get_run(run_id))["settled"]
    custody._CUSTODY.clear()
    return engine


def _terminal_wake(ctx, run_id="run-1", status="terminal"):
    from ouroboros.delegate_supervision import supervised_wait

    return supervised_wait(ctx, run_id, wait_once=lambda *_a: json.dumps({
        "status": status, "run_id": run_id, "state": "interrupted", "last_seq": 9}))


def test_an_unseen_own_settled_run_is_one_fact_line_until_this_attempt_takes_its_wake(tmp_path):
    from ouroboros.delegate_supervision import acknowledge_pending_wake

    _settle(tmp_path)
    line = next(item for item in _health(tmp_path).splitlines() if "OWN DELEGATED RUN ENDED" in item)
    assert "run run-1 interrupted (crash_interrupted) at " in line
    assert "run folder /engine/runs/run-1" in line
    assert "primary output answer final/answer.md (10 bytes)" in line and "diagnostic" not in line
    assert "delegate_wait" not in line, "a fact, never a command"
    assert "OWN DELEGATED RUN" not in _health(tmp_path, task_id="other"), "never a foreign run"
    assert "OWN DELEGATED RUN" not in _health(tmp_path, attempt_key=""), "no attempt, no claim"

    ctx = _sup_ctx(tmp_path)
    wake = _terminal_wake(ctx)
    assert "OWN DELEGATED RUN ENDED" in _health(tmp_path), "generated is not appended"
    assert acknowledge_pending_wake(ctx, '{"supervision_wake_id": "another-wake"}') is False
    assert not [row for row in custody.custody_rows(tmp_path) if row["type"] == custody.TERMINAL_ACKNOWLEDGED]
    assert acknowledge_pending_wake(ctx, wake.text)
    acks = [row for row in custody.custody_rows(tmp_path) if row["type"] == custody.TERMINAL_ACKNOWLEDGED]
    assert [(row["run_id"], row["task_id"], row["attempt_key"]) for row in acks] == [("run-1", "root", "1")]
    assert "OWN DELEGATED RUN" not in _health(tmp_path), "appended in THIS attempt: gone at the next build"
    assert "OWN DELEGATED RUN ENDED" in _health(tmp_path, attempt_key="2"), \
        "a crash retry's fresh transcript never took it"

    # A late duplicate STARTED keeps the terminal facts and the acknowledgement.
    custody._CUSTODY.clear()
    assert custody.record_started(tmp_path, custody.RunCustody(
        run_id="run-1", task_id="root", route_id="r", model="m", ledger_root=str(tmp_path)))
    custody._CUSTODY.clear()
    replayed = custody.replay(tmp_path)["run-1"]
    assert (replayed.terminal_state, replayed.terminal_reason, replayed.terminal_ack_attempt) == (
        "interrupted", "crash_interrupted", "1")
    assert replayed.engine_locator["run_dir"] == "/engine/runs/run-1" and replayed.settled_at
    assert "OWN DELEGATED RUN" not in _health(tmp_path)


def test_a_non_terminal_wake_and_a_foreign_ack_row_retire_nothing(tmp_path):
    from ouroboros.delegate_supervision import acknowledge_pending_wake

    _settle(tmp_path)
    ctx = _sup_ctx(tmp_path)
    progress = _terminal_wake(ctx, status="completed")
    assert acknowledge_pending_wake(ctx, progress.text)
    assert not [row for row in custody.custody_rows(tmp_path) if row["type"] == custody.TERMINAL_ACKNOWLEDGED]
    assert custody.emit(tmp_path, custody.TERMINAL_ACKNOWLEDGED, {
        "run_id": "run-1", "task_id": "intruder", "wake_id": "w", "attempt_key": "1"})
    custody._CUSTODY.clear()
    assert "OWN DELEGATED RUN ENDED" in _health(tmp_path)


def test_review_owned_unread_and_pre_retention_runs_add_no_second_line(tmp_path, monkeypatch):
    _settle(tmp_path, run_id="run-review", source="review_substrate", review_slot_id="slot")
    entry = custody.RunCustody(run_id="run-old", task_id="root", route_id="r", ledger_root=str(tmp_path))
    assert custody.record_started(tmp_path, entry)
    assert custody.emit(tmp_path, custody.SETTLED, {"run_id": "run-old", "task_id": "root", "state": "failed"})
    custody._CUSTODY.clear()
    assert "OWN DELEGATED RUN" not in _health(tmp_path)

    ctx = ToolContext(repo_dir=tmp_path, drive_root=tmp_path, task_id="root")
    ctx.task_metadata = {"root_task_id": "root"}
    from ouroboros.gateways import claudexor as gw
    from ouroboros.tools import delegate

    engine = _Engine()
    monkeypatch.setattr(gw, "ClaudexorGateway", lambda *a, **k: engine)
    assert custody.record_started(tmp_path, custody.RunCustody(
        run_id="run-big", task_id="root", route_id="r", model="m", ledger_root=str(tmp_path)))
    out = json.loads(delegate._delegate_wait(ctx, "run-big", wait_sec=1))
    custody._CUSTODY.clear()
    health = _health(tmp_path)
    assert "DELEGATED RESULT NEVER READ — run run-big" in health
    assert out["output_delivery"]["read_next"]["path"] in health
    assert "OWN DELEGATED RUN" not in health, "the unread D7 line already covers it"


def test_unknown_engine_facts_stay_absent_and_the_line_stays_bounded(tmp_path):
    for index in range(12):
        _settle(tmp_path, run_id=f"run-{index:02d}",
                engine=_Engine(state="failed", run_dir=False, primary=False))
    lines = [item for item in _health(tmp_path).splitlines() if "OWN DELEGATED RUN" in item]
    assert len(lines) == 11 and "+2 older settled runs" in lines[-1]
    assert all("engine-owned" not in item for item in lines)
    assert custody.replay(tmp_path)["run-00"].engine_locator == {}


def test_one_custody_read_serves_every_delegated_line(tmp_path, monkeypatch):
    from ouroboros import delegate_custody_current as current

    _settle(tmp_path)
    calls, replays = [], []
    real, real_replay = current.health_runs, custody.replay
    monkeypatch.setattr(current, "health_runs", lambda *a, **k: calls.append(1) or real(*a, **k))
    monkeypatch.setattr(custody, "replay", lambda *a, **k: replays.append(1) or real_replay(*a, **k))
    assert "OWN DELEGATED RUN ENDED" in _health(tmp_path)
    assert calls == [1]
    assert replays == []


def test_the_context_builder_passes_the_actual_attempt(tmp_path, monkeypatch):
    """``build_llm_messages`` reads the attempt the acknowledgement recorded."""
    from ouroboros.context import build_llm_messages
    from ouroboros.delegate_supervision import acknowledge_pending_wake
    from ouroboros.memory import Memory

    monkeypatch.setattr("ouroboros.context.resolve_context_fit_route", lambda *_a, **_kw: (
        {"model": "fixture", "provider": "fixture"},
        SimpleNamespace(route_fp="fixture", status="asserted", stale=False, window_tokens=1_000_000),
    ))

    class FakeEnv:
        repo_dir = tmp_path / "repo"
        drive_root = tmp_path

        def drive_path(self, rel):
            return tmp_path / rel

        def repo_path(self, rel):
            return tmp_path / "repo" / rel

    for rel in ("repo/prompts", "repo/docs", "memory/knowledge", "logs", "state"):
        (tmp_path / rel).mkdir(parents=True, exist_ok=True)
    (tmp_path / "repo" / "prompts" / "SYSTEM.md").write_text("System prompt", encoding="utf-8")
    (tmp_path / "state" / "state.json").write_text('{"spent_usd": 0}', encoding="utf-8")
    _settle(tmp_path)
    ctx = _sup_ctx(tmp_path)
    assert acknowledge_pending_wake(ctx, _terminal_wake(ctx).text)

    def rendered(attempt):
        task = {"id": "root", "type": "task", "text": "continue", "_attempt": attempt}
        messages, _ = build_llm_messages(env=FakeEnv(), memory=Memory(drive_root=tmp_path), task=task)
        return json.dumps(messages, ensure_ascii=False)

    assert "OWN DELEGATED RUN ENDED" not in rendered(1)
    assert "OWN DELEGATED RUN ENDED — run run-1 interrupted" in rendered(2)


# --- receipts: engine locators and the retained source, no churn for old rows ------

def test_receipts_carry_locators_and_old_receipts_stay_byte_identical(geo, monkeypatch):
    from ouroboros.delegate_terminal import refresh_terminal_reconciliation
    from ouroboros.task_results import load_task_result, write_task_result

    nanny = _child(geo, NANNY, ROOT)
    out = _stage(geo, nanny, monkeypatch)
    audit = _publish_terminal_result(geo, NANNY)
    row = audit["terminal_runs"][0]
    assert row["engine_locator"] == {"run_dir": "/engine/runs/run-1", "primary_output": {
        "kind": "answer", "path": "final/answer.md", "bytes": len(ANSWER.encode("utf-8")), "truncated": False}}
    assert "text" not in json.dumps(row["engine_locator"])
    source = out["output_delivery"]["artifact"]["source"]
    assert {key: row["staged_output"][key] for key in source} == source
    assert row["staged_output"]["complete"] is True
    stored = load_task_result(geo.canonical, NANNY)
    assert (stored["status"], stored["reason_code"]) == ("cancelled", "server_shutdown"), \
        "receipts never touch the primary status or reason (owner Q5=A)"

    # A task whose runs settled before these facts existed: the same five keys, no rewrite.
    from ouroboros import obligations, startup_migrations, usage_store
    from ouroboros.task_store import signature
    from supervisor.events_task_done import _terminal_cost_probe

    legacy_task = "legacy0aa111111"
    old_receipt = {"run_id": "run-legacy", "state": "failed", "model": "m", "profile_id": "",
                   "selected_subagent_id": ""}

    def legacy_result(root):
        write_task_result(root, legacy_task, "failed", reason_code="tool_failure",
                          delegate_terminal_reconciliation={
                              "task_id": legacy_task, "trigger": "terminal_boundary", "outcomes": [],
                              "unreconciled": [], "audit_status": "ok", "open_run_ids": [],
                              "pending_invocation_ids": [], "undisposed_patch_run_ids": [],
                              "terminal_runs": [old_receipt], "deferred_project_retirements": []},
                          delegated_runs_unreconciled=[])

    def frozen(root):
        return (signature(root, legacy_task), load_task_result(root, legacy_task),
                (root / "logs" / "events.jsonl").read_bytes())

    assert custody.record_started(geo.canonical, custody.RunCustody(
        run_id="run-legacy", task_id=legacy_task, route_id="r", model="m", ledger_root=str(geo.canonical)))
    assert custody.emit(geo.canonical, custody.SETTLED, {"run_id": "run-legacy", "task_id": legacy_task,
                                                         "state": "failed", "model": "m"})
    custody._CUSTODY.clear()
    # Imported as closed history (an install whose chain predates its store) the run owes nothing:
    # the refresh reads no money and changes no byte.
    history = geo.tmp / "history"
    (history / "logs").mkdir(parents=True)
    shutil.copyfile(geo.canonical / "logs" / "events.jsonl", history / "logs" / "events.jsonl")
    startup_migrations.prepare_startup_state(history)
    assert legacy_task not in obligations.members(history, "delegated_runs")
    legacy_result(history)
    before = frozen(history)
    with monkeypatch.context() as patch:
        for name in ("read", "hold"):
            patch.setattr(usage_store, name, lambda *_a, **_k: pytest.fail("money work without a debt"))
        assert refresh_terminal_reconciliation(history, legacy_task) is False
    assert frozen(history) == before
    # Settled live, the same run owes the task's cost projection once: paying it updates money
    # fields only, never the receipt, and the repeat is a byte-level no-op.
    legacy_result(geo.canonical)
    unpaid = load_task_result(geo.canonical, legacy_task)
    assert refresh_terminal_reconciliation(geo.canonical, legacy_task) is False
    paid = load_task_result(geo.canonical, legacy_task)
    assert paid["delegate_terminal_reconciliation"] == unpaid["delegate_terminal_reconciliation"]
    assert paid["delegate_terminal_reconciliation"]["terminal_runs"] == [old_receipt]
    outcome, money, _ = _terminal_cost_probe(geo.canonical, legacy_task, paid)
    assert outcome == "equal" and legacy_task not in obligations.members(geo.canonical, "delegated_runs")
    assert {key for key in {*unpaid, *paid} if unpaid.get(key) != paid.get(key)} <= {*money, "updated_at"}
    before = frozen(geo.canonical)
    assert refresh_terminal_reconciliation(geo.canonical, legacy_task) is False
    assert frozen(geo.canonical) == before
