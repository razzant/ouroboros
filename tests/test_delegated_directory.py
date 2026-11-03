"""Ordinary-folder dispatch and complete engine result custody without host Git."""
from hashlib import sha256
import json
from pathlib import Path

import httpx
import pytest

from ouroboros import delegate_custody as custody
from ouroboros.delegate_directory import capture_directory_result, integrate_directory_result
from ouroboros.tools import delegate
from ouroboros.tools.registry import ToolContext
from tests._delegated_transport_shared import _HealthStub, _owned_gateway_uses_each_test_transport  # noqa: F401


class DirectoryEngine(_HealthStub):
    def __init__(self, target, strategy="copy", *, lost_apply=False):
        super().__init__(engine_version="99.0.0")
        self.target, self.strategy, self.lost_apply = target, strategy, lost_apply
        self.posts, self.applies, self.decisions = [], [], []
        self.body = b"\x00complete binary result\xff" * 1000
        state = {"kind": "file", "sha256": "sha256:" + sha256(self.body).hexdigest(),
                 "sizeBytes": len(self.body), "mode": 420, "artifactPath": "final/files/output.bin"}
        self.manifest = {"version": 1, "sourceRoot": str(target),
                         "executionRoot": str(target if strategy == "direct" else target.parent / "engine-copy"),
                         "isolation": "live" if strategy == "direct" else "envelope",
                         "scopePaths": ["."], "complete": True,
                         "entries": [{"path": "out.bin", "before": None, "after": state}]}
        self.raw = json.dumps(self.manifest).encode()

    def agent_capabilities(self):
        return {**super().agent_capabilities(), "mutability": {"workspaceKinds": ["git", "directory"]}}

    def find_project_id(self, root):
        assert root == str(self.target)
        return "project-existing"

    def start_run(self, request, *, idempotency_key):
        self.posts.append((request, idempotency_key))
        return {"runId": "directory-run", "runDir": "/engine/run"}

    def get_run(self, rid):
        return {"summary": {"result": {"applyState": "applied" if self.applies else "not_applied"}},
                "workProduct": {"kind": "files", "files": {"manifest": "final/files/manifest.json"},
                                "meta": {"manifest_sha256": "sha256:" + sha256(self.raw).hexdigest(),
                                         "apply_state": "applied" if self.strategy == "direct" else "not_applied"}}}

    def stream_run_artifact(self, rid, path, sink, *, expected=None):
        body = self.raw if path.endswith("manifest.json") else self.body
        facts = {"sha256": sha256(body).hexdigest(), "size": len(body)}
        if expected:
            assert facts["sha256"] == expected["sha256"].removeprefix("sha256:")
            assert expected.get("sizeBytes", len(body)) == len(body)
        sink.write(body)
        return facts

    def apply_run(self, rid, request, *, idempotency_key):
        self.applies.append(idempotency_key)
        (self.target / "out.bin").write_bytes(self.body)
        if self.lost_apply:
            self.lost_apply = False
            raise OSError("response lost after application")
        return {"applied": True, "refused": False, "appliedPaths": ["out.bin"],
                "treeMutated": True, "alreadyApplied": False}

    def decide_run(self, rid, request, *, idempotency_key):
        self.decisions.append((request, idempotency_key))
        return {"accepted": True, "status": "discarded"}


def context(tmp_path, monkeypatch):
    repo, data, target = [tmp_path / name for name in ("system", "data", "documents")]
    for path in (repo, data, target):
        path.mkdir()
    monkeypatch.setenv("OUROBOROS_SUBAGENT_HARNESS", "some-route=weak-model:low")
    return ToolContext(repo_dir=repo, drive_root=data, task_id="parent", workspace_root=target,
                       workspace_mode="external"), target


@pytest.mark.parametrize("strategy", ["direct", "copy"])
@pytest.mark.parametrize("access", ["workspace_write", "full"])
def test_start_uses_normal_writing_mode_without_git_or_fake_snapshot(tmp_path, monkeypatch, strategy, access):
    from ouroboros.gateways import claudexor
    import tests._delegated_transport_shared as shared
    ctx, target = context(tmp_path, monkeypatch)
    engine = DirectoryEngine(target, strategy)
    engine.profiles = ("readonly", "workspace_write", "full")
    snapshot = shared._transport_snapshot
    monkeypatch.setattr(shared, "_transport_snapshot", lambda route: {**snapshot(route), "access": access})
    grants = []
    monkeypatch.setattr(engine, "ensure_full_access", lambda root: grants.append(root), raising=False)
    monkeypatch.setattr(claudexor, "ClaudexorGateway", lambda *a, **k: engine)
    result = json.loads(delegate._delegate_start(ctx, "edit documents", directory_strategy=strategy, scope_paths=["."]).text)
    assert result["status"] == "started", result
    request, key = engine.posts[0]
    assert request["scope"]["root"] == str(target)
    assert request["mode"] == "agent" and request["access"] == access
    assert grants == ([str(target)] if access == "full" else [])
    assert request["execution"]["workspaceKind"] == "directory"
    assert request["execution"]["isolation"] == ("live" if strategy == "direct" else "envelope")
    assert request["execution"]["scopePaths"] == ["."]
    assert result["execution_root"] == (str(target) if strategy == "direct" else None)
    recorded = custody.replay(ctx.drive_root)["directory-run"]
    assert recorded.invocation_id == key and not recorded.snapshot_id and not recorded.baseline_sha
    assert recorded.resource_ref["strategy"] == strategy
    assert not (target / ".git").exists()


@pytest.mark.parametrize("options,expected_scope", [
    ({}, None),
    ({"directory_strategy": "direct"}, None),
    ({"scope_paths": []}, []),
    ({"directory_strategy": "direct", "scope_paths": []}, []),
])
def test_a_write_capable_child_keeps_its_attested_folder_shape(
    tmp_path, monkeypatch, options, expected_scope,
):
    """#882 changed nothing for a child that can actually open the session.

    The read-only repair must not quietly rewrite a write-capable request: an
    explicit `direct` still starts the same live directory session as omitting
    it, and an explicit empty footprint still rides the wire as `scopePaths: []`
    — the parent said "capture nothing", which is a different attested choice
    from saying nothing at all, and only the engine gets to interpret it.
    """
    from ouroboros.gateways import claudexor

    ctx, target = context(tmp_path, monkeypatch)
    engine = DirectoryEngine(target, "direct")
    monkeypatch.setattr(claudexor, "ClaudexorGateway", lambda *a, **k: engine)
    result = json.loads(delegate._delegate_start(ctx, "edit documents", **options).text)
    assert result["status"] == "started", result
    execution = engine.posts[0][0]["execution"]
    assert execution["workspaceKind"] == "directory" and execution["isolation"] == "live"
    assert execution.get("scopePaths") == expected_scope


def _git_workspace_start(tmp_path, monkeypatch, case, **options):
    """Start one write-capable child against a fresh Git workspace."""
    import subprocess

    from ouroboros.gateways import claudexor

    root = tmp_path / case
    root.mkdir()
    monkeypatch.setenv("OUROBOROS_SUBAGENT_WORKTREE_ROOT", str(root / "snaps"))
    ctx, target = context(root, monkeypatch)
    subprocess.run(["git", "init"], cwd=str(target), capture_output=True, check=True)
    engine = DirectoryEngine(target, "direct")
    monkeypatch.setattr(claudexor, "ClaudexorGateway", lambda *a, **k: engine)
    delegate._CUSTODY.clear()
    payload = json.loads(delegate._delegate_start(ctx, "edit documents", **options).text)
    delegate._CUSTODY.clear()
    return payload, engine


def test_a_git_workspace_treats_the_named_default_as_omission_and_still_refuses_real_geometry(
    tmp_path, monkeypatch,
):
    """#882 reaches the sibling refusal site too: a named default is not a request.

    A Git tree keeps its private-snapshot contract, so `copy` or a selected
    footprint is a genuine contradiction for a write-capable child there and stays
    a typed `definitely_unrun` refusal naming that contract. `direct` with nothing
    selected asks for nothing at all — it is the documented spelling of omitting
    both — so it takes the unchanged snapshot path omission takes instead of dying
    at the host's pre-start over a word that changed no behaviour.
    """
    # Each case gets its own folder and its own invocation, so identity fields
    # differ by construction; every OTHER key and value must match, including the
    # key set itself — that is what "took the omitted path" means here.
    per_case = ("root", "execution_root", "snapshot_id", "baseline_sha", "baseline_id",
                "baseline_manifest_read", "run_id", "invocation_id", "authority_target_root",
                "snapshot")  # the receipt's provisioning facts carry wall-clock seconds (#1241)
    compared = lambda payload: {key: ("<per-case identity>" if key in per_case else value)
                                for key, value in payload.items()}
    omitted, omitted_engine = _git_workspace_start(tmp_path, monkeypatch, "omit")
    assert omitted["status"] == "started" and omitted["baseline_id"], omitted
    for index, named_default in enumerate((
        {"directory_strategy": "direct"}, {"scope_paths": []},
        {"directory_strategy": "direct", "scope_paths": []},
    )):
        named, engine = _git_workspace_start(tmp_path, monkeypatch, f"named-{index}", **named_default)
        assert compared(named) == compared(omitted), named_default
        assert len(engine.posts) == len(omitted_engine.posts), named_default
    for index, geometry in enumerate((
        {"directory_strategy": "copy", "scope_paths": ["."]},
        {"scope_paths": ["src"]},
    )):
        refused, engine = _git_workspace_start(tmp_path, monkeypatch, f"geometry-{index}", **geometry)
        assert refused["status"] == "refused", geometry
        assert refused["reason"] == "directory_execution_unavailable"
        assert "Git workspaces keep their snapshot contract" in refused["detail"]
        assert refused["definitely_unrun"] is True
        assert engine.posts == []


def entry(ctx, target, strategy):
    return custody.RunCustody(run_id="directory-run", task_id=ctx.task_id, route_id="some-route",
                             target_root=str(target), settled=True, access="workspace_write",
                             resource_ref={"workspace_kind": "directory", "strategy": strategy, "scopePaths": ["."]})


@pytest.mark.parametrize("strategy", ["direct", "copy"])
def test_capture_reopen_and_disposition_use_complete_engine_bytes(tmp_path, monkeypatch, strategy):
    ctx, target = context(tmp_path, monkeypatch)
    engine = DirectoryEngine(target, strategy)
    held = entry(ctx, target, strategy)
    if strategy == "direct":
        (target / "out.bin").write_bytes(engine.body)
    captured = capture_directory_result(ctx.drive_root, held, engine)
    assert Path(captured["file_outputs"][1]["path"]).read_bytes() == engine.body
    assert capture_directory_result(ctx.drive_root, held, engine) == captured
    response = json.loads(integrate_directory_result(ctx, held, "apply", "accepted", engine))
    assert response["status"] == "applied"
    assert (target / "out.bin").read_bytes() == engine.body
    assert len(engine.applies) == (1 if strategy == "copy" else 0)
    assert not (target / ".git").exists()


def test_discard_is_engine_disposition_not_fake_apply(tmp_path, monkeypatch):
    ctx, target = context(tmp_path, monkeypatch)
    engine, held = DirectoryEngine(target), entry(ctx, target, "copy")
    response = json.loads(integrate_directory_result(ctx, held, "reject", "not selected", engine))
    assert response["status"] == "rejected" and held.patch_disposed == "rejected"
    assert engine.decisions[0][0] == {"action": "discard"}
    assert not engine.applies and not (target / "out.bin").exists()


def test_lost_apply_leaves_existing_intent_pending(tmp_path, monkeypatch):
    ctx, target = context(tmp_path, monkeypatch)
    engine, held = DirectoryEngine(target, lost_apply=True), entry(ctx, target, "copy")
    capture_directory_result(ctx.drive_root, held, engine)  # the body's capture, before the decision (#1969)
    with pytest.raises(OSError, match="response lost"):
        integrate_directory_result(ctx, held, "apply", "accepted", engine)
    assert held.patch_apply_pending and not held.patch_disposed
    assert (target / "out.bin").read_bytes() == engine.body
    assert "APPLY_AMBIGUOUS" in integrate_directory_result(ctx, held, "apply", "", engine)
    assert len(engine.applies) == 1
    response = json.loads(integrate_directory_result(ctx, held, "apply", "", engine, acknowledge_ambiguous=True))
    assert response["status"] == "applied"
    assert engine.applies[0] == engine.applies[1]


def test_selected_apply_keeps_remaining_results_undisposed(tmp_path, monkeypatch):
    ctx, target = context(tmp_path, monkeypatch)
    engine, held = DirectoryEngine(target), entry(ctx, target, "copy")
    capture_directory_result(ctx.drive_root, held, engine)  # the body's capture, before the decision (#1969)
    get_run = engine.get_run

    def partially_delivered(run_id):
        detail = get_run(run_id)
        detail["summary"]["result"]["applyState"] = "not_applied"
        return detail

    monkeypatch.setattr(engine, "get_run", partially_delivered)
    response = json.loads(integrate_directory_result(
        ctx, held, "apply", "selected output", engine, paths=["out.bin"]))
    assert response["status"] == "partially_applied"
    assert response["engine_receipt"]["appliedPaths"] == ["out.bin"]
    assert (target / "out.bin").read_bytes() == engine.body
    assert not held.patch_disposed and not held.patch_apply_pending


def test_registered_integration_tool_accepts_selected_paths():
    from ouroboros.tools.subagent_integration import get_tools
    entry = next(item for item in get_tools() if item.name == "integrate_delegated_patch")
    assert entry.handler.__defaults__ is not None
    assert "paths" in __import__("inspect").signature(entry.handler).parameters


def test_lost_start_replays_original_processing_facts_after_setting_changes(tmp_path, monkeypatch):
    from ouroboros.gateways import claudexor

    ctx, target = context(tmp_path, monkeypatch)
    engine = DirectoryEngine(target)
    capabilities = engine.agent_capabilities
    monkeypatch.setattr(engine, "agent_capabilities", lambda: {
        **capabilities(), "harnesses": [{**row, "processingPreferences": ["fast", "economy"]}
                                       for row in capabilities()["harnesses"]]})
    prepare_actor = delegate.prepare_delegate_start_actor
    preference = "economy"

    def actor(*args, **kwargs):
        captured, refusal = prepare_actor(*args, **kwargs)
        return {**captured, "processing_preference": preference}, refusal

    def start(request, *, idempotency_key):
        engine.posts.append((request, idempotency_key))
        if len(engine.posts) == 1:
            raise claudexor.ClaudexorUnavailable("daemon_unreachable", "response lost")
        return {"runId": "directory-run"}

    monkeypatch.setattr(delegate, "prepare_delegate_start_actor", actor)
    monkeypatch.setattr(engine, "start_run", start)
    monkeypatch.setattr(claudexor, "ClaudexorGateway", lambda: engine)
    lost = json.loads(delegate._delegate_start(ctx, "edit documents", directory_strategy="copy", scope_paths=["."]).text)
    token = lost["pending_invocation_id"]
    original = custody.invocation_record(ctx.drive_root, token)["processing"]
    assert original["requested"] == original["submitted"] == "economy"
    preference = "fast"
    retried = json.loads(delegate._delegate_start(ctx, "edit documents", retry_of=token).text)
    assert retried["status"] == "started" and retried["processing"] == original
    assert engine.posts[0] == engine.posts[1]
    assert engine.posts[1][0]["processingPreference"] == "economy"
    held = custody.replay(ctx.drive_root)["directory-run"]
    assert held.processing_preference == "economy"
    assert held.effort == engine.posts[1][0].get("effort", "")
    from ouroboros.subagent_history import record_session_execution, subagent_last_delegation
    monkeypatch.setattr(custody, "invocation_record", lambda *_a, **_kw: pytest.fail("history scanned an invocation"))
    record_session_execution(ctx.drive_root, held,
        {"summary": {"state": "succeeded", "finishedAt": "2099-01-01T00:00:00Z"}}, {})
    assert subagent_last_delegation(ctx.drive_root)["identity"]["processing_preference"] == "economy"


@pytest.mark.parametrize('access', ['workspace_write', 'full'])
def test_research_folder_uses_copy_retains_evidence_and_never_applies(tmp_path, monkeypatch, access):
    from ouroboros.contracts.task_constraint import TaskConstraint
    from ouroboros.gateways import claudexor
    from ouroboros.tools.delegate_integration import capture_terminal_patch_for_drive
    from ouroboros.tools.subagent_integration_delegated import _delegated_disposition_refusal
    import tests._delegated_transport_shared as shared

    ctx, target = context(tmp_path, monkeypatch)
    ctx.task_constraint = TaskConstraint(mode='local_readonly_subagent')
    engine = DirectoryEngine(target, 'copy')
    engine.profiles = ('readonly', 'workspace_write', 'full')
    monkeypatch.setattr(engine, 'ensure_full_access', lambda root: None, raising=False)
    snapshot = shared._transport_snapshot
    monkeypatch.setattr(shared, '_transport_snapshot', lambda route: {**snapshot(route), 'access': access})
    monkeypatch.setattr(claudexor, 'ClaudexorGateway', lambda *a, **k: engine)
    result = json.loads(delegate._delegate_start(ctx, 'Investigate these documents').text)
    assert result['status'] == 'started', result
    wire, key = engine.posts[0]
    assert wire['access'] == access and wire['execution']['workspaceKind'] == 'directory'
    assert wire['execution']['isolation'] == 'envelope' and wire['execution']['scopePaths'] == ['.']
    held = custody.replay(ctx.drive_root)['directory-run']
    assert held.authority_source == 'investigation' and held.resource_ref['strategy'] == 'copy'
    held.settled = True
    captured = capture_terminal_patch_for_drive(ctx.drive_root, held, gateway=engine)
    assert captured['disposition'] == 'evidence_retained'
    assert not engine.applies and not (target / 'out.bin').exists()
    assert 'INVESTIGATION_EVIDENCE' in _delegated_disposition_refusal(custody.OWNED, held, held.run_id)
    assert engine.decisions[0][0] == {'action': 'discard'}
    assert not custody.undisposed_patches(ctx.drive_root, {held.run_id: held})


def test_research_folder_cannot_select_direct_writes(tmp_path, monkeypatch):
    from ouroboros.contracts.task_constraint import TaskConstraint
    from ouroboros.gateways import claudexor

    ctx, target = context(tmp_path, monkeypatch)
    ctx.task_constraint = TaskConstraint(mode='local_readonly_subagent')
    engine = DirectoryEngine(target, 'copy')
    monkeypatch.setattr(claudexor, 'ClaudexorGateway', lambda *a, **k: engine)
    refused = json.loads(delegate._delegate_start(ctx, 'Investigate', directory_strategy='direct').text)
    assert refused['status'] == 'refused' and refused['definitely_unrun']
    assert refused['reason'] == 'directory_execution_unavailable'
    assert not engine.posts


class _NoProductEngine(DirectoryEngine):
    """A finished run whose engine answer carries no file work product."""

    state = "succeeded"

    def get_run(self, rid):
        return {"summary": {"state": self.state, "result": {"applyState": "not_applied"}}}


@pytest.mark.parametrize("strategy", ["direct", "copy"])
@pytest.mark.parametrize("decision", ["apply", "reject"])
def test_a_finished_run_without_file_work_product_is_finished_with_no_changes(tmp_path, monkeypatch, strategy, decision):
    """VERIFY_SWEEPS S3: such a run stayed collection-pending, re-collected every sweep pass
    (a daemon read, a capture attempt, a money read) because the engine's final answer was
    kept as a retryable capture failure."""
    from ouroboros.delegate_custody_current import collection_pending

    ctx, target = context(tmp_path, monkeypatch)
    engine, held = _NoProductEngine(target, strategy), entry(ctx, target, strategy)
    captured = capture_directory_result(ctx.drive_root, held, engine)
    assert captured["status"] == "ready" and captured["no_changes"] is True and captured["file_outputs"] == []
    assert held.patch_captured and capture_directory_result(ctx.drive_root, held, engine) == captured
    held.terminal_source = {"sha256": "t"}  # every other collection duty done: only the capture was open
    held.ledger_recorded = held.output_complete = True
    held.output_source, held.output_terminal_sha = "final/answer.md", "t"
    assert not collection_pending(held)
    response = json.loads(integrate_directory_result(ctx, held, decision, "nothing to take", engine))
    assert response["status"] == ("applied" if decision == "apply" else "rejected")
    assert held.patch_disposed == response["status"] and not engine.applies
    assert engine.decisions == ([] if strategy == "direct" else [({"action": "discard"}, engine.decisions[0][1])])


@pytest.mark.parametrize("strategy", ["direct", "copy"])
def test_an_unsuccessful_run_without_file_work_product_is_never_reported_as_no_changes(tmp_path, monkeypatch, strategy):
    """Absence after a failed run proves nothing: its capture is closed (no sweep redo), Apply applies
    nothing and keeps the engine copy, and only Reject releases it."""
    ctx, target = context(tmp_path, monkeypatch)
    engine, held = _NoProductEngine(target, strategy), entry(ctx, target, strategy)
    engine.state = "failed"
    captured = capture_directory_result(ctx.drive_root, held, engine)
    assert captured["status"] == "ready" and captured["no_changes"] is False and captured["work_product"] == "absent"
    assert "changes, if any, were not captured" in captured["note"]
    assert ("already in the folder" in captured["note"]) is (strategy == "direct")
    refused = integrate_directory_result(ctx, held, "apply", "take it", engine)
    assert refused.startswith("⚠️ INTEGRATE_DELEGATED_NOTHING_CAPTURED") and not engine.decisions and not engine.applies
    assert not held.patch_disposed
    response = json.loads(integrate_directory_result(ctx, held, "reject", "nothing to take", engine))
    assert response["status"] == "rejected" and held.patch_disposed == "rejected" and not engine.applies
    assert engine.decisions == ([] if strategy == "direct" else [({"action": "discard"}, engine.decisions[0][1])])


UNCHANGED, TWIN = (b"same bytes\n", 0o644), (b"same bytes\n", 0o755)
OLD, NEW = (b"old\x00" * 64, 0o644), (b"new\xff" * 64, 0o644)
# One unchanged file, its bytes again at another path and mode, distinct content, a shared preimage.
FILES = [("a/unchanged.txt", UNCHANGED, UNCHANGED), ("b/twin.sh", None, TWIN),
         ("c/edited.bin", OLD, NEW), ("d/removed.bin", OLD, None)]


class _CutStream(httpx.SyncByteStream):
    def __init__(self, head):
        self.head = head

    def __iter__(self):
        yield self.head
        raise httpx.ReadError("connection closed mid-body")


class ContentEngine(DirectoryEngine):
    """The engine's complete manifest shape with content objects at ``final/files/content/<hex>``.

    Artifact bytes stream through the real gateway reader over an HTTP transport that
    records every request; apply delivers the selected rows' postimages."""

    def __init__(self, target, strategy, files):
        from ouroboros.gateways.claudexor import ClaudexorGateway, DaemonEndpoint

        super().__init__(target, strategy)
        self.objects, self.requests, self.faults, self.delivered = {}, [], {}, set()
        self.manifest["entries"] = [{"path": path, "before": self.state(before, strategy == "copy"),
                                     "after": self.state(after, True)} for path, before, after in files]
        self.raw = json.dumps(self.manifest).encode()
        self.objects["final/files/manifest.json"] = self.raw
        self.wire = ClaudexorGateway(DaemonEndpoint("127.0.0.1", 1, "offline"))
        self.wire._client.close()
        self.wire._client = httpx.Client(base_url="http://127.0.0.1:1", transport=httpx.MockTransport(self.serve))

    def state(self, side, retained):
        if side is None:
            return None
        digest = sha256(side[0]).hexdigest()
        self.objects[f"final/files/content/{digest}"] = side[0]
        return {"kind": "file", "sha256": "sha256:" + digest, "sizeBytes": len(side[0]), "mode": side[1],
                **({"artifactPath": f"final/files/content/{digest}"} if retained else {})}

    def serve(self, request):
        path = request.url.path.split("/artifacts/", 1)[1]
        self.requests.append(path)
        body, fault = self.objects.get(path), self.faults.get(path)
        if body is None or fault == "unavailable":
            return httpx.Response(404, json={"code": "artifact_not_found", "message": "no such artifact"})
        if fault == "corrupt":
            return httpx.Response(200, content=body[:-1] + bytes([body[-1] ^ 1]))
        if fault == "truncated":
            return httpx.Response(200, stream=_CutStream(body[:len(body) // 2]))
        return httpx.Response(200, content=body)

    def stream_run_artifact(self, rid, path, sink, *, expected=None):
        return self.wire.stream_run_artifact(rid, path, sink, expected=expected)

    def get_run(self, rid):
        detail = super().get_run(rid)
        complete = self.delivered == {row["path"] for row in self.manifest["entries"]}
        detail["summary"]["result"]["applyState"] = "applied" if complete else "not_applied"
        return detail

    def apply_run(self, rid, request, *, idempotency_key):
        self.applies.append(idempotency_key)
        chosen = request.get("paths") or [row["path"] for row in self.manifest["entries"]]
        for row in self.manifest["entries"]:
            if row["path"] in chosen:
                destination = self.target / row["path"]
                destination.parent.mkdir(parents=True, exist_ok=True)
                if row["after"]:
                    destination.write_bytes(self.objects[row["after"]["artifactPath"]])
                else:
                    destination.unlink(missing_ok=True)
        self.delivered.update(chosen)
        return {"applied": True, "refused": False, "appliedPaths": chosen, "treeMutated": True}


def _retained(drive, held):
    cap = custody.delegated_capture_dir(drive, held.task_id, held.run_id)
    return cap, {path.name: (path.stat().st_ino, path.stat().st_mode, path.stat().st_mtime_ns, path.read_bytes())
                 for path in cap.iterdir()}


@pytest.mark.parametrize("strategy", ["direct", "copy"])
def test_one_capture_fetches_and_keeps_each_content_identity_once(tmp_path, monkeypatch, strategy):
    from ouroboros.artifacts import ArtifactIdentityError

    ctx, target = context(tmp_path, monkeypatch)
    engine, held = ContentEngine(target, strategy, FILES), entry(ctx, target, strategy)
    captured = capture_directory_result(ctx.drive_root, held, engine)
    cap, _ = _retained(ctx.drive_root, held)
    sides = [(row["path"], side, row[side]) for row in engine.manifest["entries"] for side in ("before", "after")
             if row[side] and "artifactPath" in row[side]]  # a live run's preimages have no engine bytes
    outputs = captured["file_outputs"][1:]
    assert [(row["source_path"], row["side"]) for row in outputs] == [(path, side) for path, side, _ in sides]
    for row, (_, _, state) in zip(outputs, sides):
        assert ("sha256:" + row["sha256"], row["size"]) == (state["sha256"], state["sizeBytes"]) and row["immutable"]
        assert sha256((cap / row["name"]).read_bytes()).hexdigest() == row["sha256"]
    objects = {state["artifactPath"] for _, _, state in sides}
    assert sorted(engine.requests) == sorted(["final/files/manifest.json", *objects])
    kept = {row["name"] for row in outputs}
    assert len(kept) == len(objects) == (3 if strategy == "copy" else 2)
    assert {path.name for path in cap.iterdir()} == {"engine-files-manifest.json", "workspace_patch.json", *kept}
    assert (cap / "engine-files-manifest.json").read_bytes() == engine.raw  # every entry and mode, exactly
    for name in kept:  # one plain file each: no link to another path, no mode from any row
        assert (cap / name).stat().st_nlink == 1 and not (cap / name).stat().st_mode & 0o111
    engine.requests.clear()
    assert capture_directory_result(ctx.drive_root, held, engine) == captured and not engine.requests
    (cap / outputs[0]["name"]).write_bytes(b"changed after capture\n")
    with pytest.raises(ArtifactIdentityError):
        capture_directory_result(ctx.drive_root, held, engine)


def test_partial_apply_and_consumer_copies_leave_shared_content_unchanged(tmp_path, monkeypatch):
    from ouroboros.artifacts import copy_artifact_file

    ctx, target = context(tmp_path, monkeypatch)
    engine, held = ContentEngine(target, "copy", FILES), entry(ctx, target, "copy")
    shared = capture_directory_result(ctx.drive_root, held, engine)["file_outputs"][1]
    cap, before = _retained(ctx.drive_root, held)
    copy = tmp_path / "user" / "twin.sh"
    copy_artifact_file(cap / shared["name"], copy, expected=shared)
    copy.write_bytes(b"the owner's own edit\n")
    partial = json.loads(integrate_directory_result(ctx, held, "apply", "twin only", engine, paths=["b/twin.sh"]))
    assert partial["status"] == "partially_applied" and not held.patch_disposed
    assert (target / "b/twin.sh").read_bytes() == TWIN[0] and not (target / "a/unchanged.txt").exists()
    assert _retained(ctx.drive_root, held)[1] == before
    rest = json.loads(integrate_directory_result(ctx, held, "apply", "the rest", engine))
    assert rest["status"] == "applied" and held.patch_disposed == "applied"
    assert (target / "a/unchanged.txt").read_bytes() == UNCHANGED[0]
    assert (target / "c/edited.bin").read_bytes() == NEW[0] and not (target / "d/removed.bin").exists()
    assert _retained(ctx.drive_root, held)[1] == before


@pytest.mark.parametrize("fault,code", [("corrupt", "artifact_integrity_error"), ("size", "artifact_integrity_error"),
                                        ("unavailable", "artifact_not_found"), ("truncated", "daemon_unreachable")])
def test_failures_of_repeated_content_stay_failures_and_a_new_capture_refetches(tmp_path, monkeypatch, fault, code):
    from ouroboros.gateways.claudexor import ClaudexorUnavailable

    ctx, target = context(tmp_path, monkeypatch)
    engine, held = ContentEngine(target, "copy", FILES), entry(ctx, target, "copy")
    shared = engine.manifest["entries"][0]["after"]["artifactPath"]
    if fault == "size":  # a later row names the same digest with another size: never reused
        engine.manifest["entries"][1]["after"]["sizeBytes"] += 1
        engine.raw = engine.objects["final/files/manifest.json"] = json.dumps(engine.manifest).encode()
    else:
        engine.faults[shared] = fault
    with pytest.raises(ClaudexorUnavailable) as failed:
        capture_directory_result(ctx.drive_root, held, engine)
    assert failed.value.code == code
    cap, _ = _retained(ctx.drive_root, held)
    assert not held.patch_captured and not (cap / "workspace_patch.json").exists()
    assert engine.requests.count(shared) == (2 if fault == "size" else 1)
    assert not [path for path in cap.iterdir() if path.name.startswith(".")]
    engine.faults.clear()
    engine.requests.clear()
    if fault == "size":  # the manifest itself contradicts its bytes: every new capture fails alike
        with pytest.raises(ClaudexorUnavailable, match="differs from its captured manifest"):
            capture_directory_result(ctx.drive_root, held, engine)
        return
    captured = capture_directory_result(ctx.drive_root, held, engine)
    assert captured["status"] == "ready" and held.patch_captured
    assert engine.requests.count(shared) == 1 and len(engine.requests) == 4
