"""Ordinary-folder dispatch and complete engine result custody without host Git."""
from hashlib import sha256
import json
from pathlib import Path

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
def test_start_uses_normal_writing_mode_without_git_or_fake_snapshot(tmp_path, monkeypatch, strategy):
    from ouroboros.gateways import claudexor
    ctx, target = context(tmp_path, monkeypatch)
    engine = DirectoryEngine(target, strategy)
    monkeypatch.setattr(claudexor, "ClaudexorGateway", lambda *a, **k: engine)
    result = json.loads(delegate._delegate_start(ctx, "edit documents", directory_strategy=strategy, scope_paths=["."]).text)
    assert result["status"] == "started", result
    request, key = engine.posts[0]
    assert request["scope"]["root"] == str(target)
    assert request["mode"] == "agent" and request["access"] == "workspace_write"
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
                "baseline_manifest_read", "run_id", "invocation_id", "authority_target_root")
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
