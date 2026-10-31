"""#2479, owner choice A: shared results report what changed, and the parent decides explicitly.

Real consumers end to end: supervisor admission of native external_workspace children,
the nanny's private snapshot/capture/_locked_apply into ONE shared tree, the child's outer
headless capture (every change since its base, siblings' work included), and
integrate_subagent_patch through the ToolRegistry as the immediate parent or as the
owner-Continue successor. A mismatch names each captured path's state and stays
undecided; decision='accept' records the parent's judgment beside that observation,
never as verification. Nothing is re-applied, reverted or staged, and every earlier
verdict stays readable by its bytes.
"""
import hashlib
import json
import os
import pathlib
import shutil
import subprocess
from types import SimpleNamespace

import pytest

from ouroboros.artifacts import copy_file_to_task_artifacts, task_artifact_dir_path
from ouroboros.headless import finalize_task_artifacts, task_artifacts_dir, task_state_dir
from ouroboros.subagent_worktrees import snapshot_commit
from ouroboros.task_custody import _child_store_plan, _collision, _publish, task_custody_lock
from ouroboros.task_results import write_task_result
from ouroboros.task_status import load_effective_task_result
from ouroboros.tools.join_ledger import _child_result_sha256
from ouroboros.tools.patch_verdict import retained_verdicts
from ouroboros.tools.registry import ToolContext, ToolRegistry
from ouroboros.tools.subagent_integration import _integrate_subagent_patch, get_tools
from ouroboros.tools.subagent_integration_delegated import _locked_apply
from ouroboros.workspace_patch_capture import write_workspace_patch_artifacts
from supervisor.events_subagent_admission import _resolve_subagent_constraint
from tests._store_shared import read_result, result_signature, seed_result
from tests.test_continue_successor_integration import continue_from, interrupted_root

# Every Git tree, state root and process is case-local; these fixtures may run in parallel.


def git(root, *args):
    return subprocess.run(["git", *args], cwd=root, capture_output=True, check=True).stdout


def init(root, *, unborn=False, files=None):
    root.mkdir()
    git(root, "init", "-q")
    git(root, "config", "user.name", "Fixture")
    git(root, "config", "user.email", "fixture@example.invalid")
    if unborn:
        return
    for name, data in {"README.md": b"# shared design tree\n", **(files or {})}.items():
        (root / name).write_bytes(data)
    git(root, "add", "-A")
    git(root, "commit", "-qm", "base")


def variant(number, author):
    html = "".join(f"<p>{author} variant {number} line {i}</p>\n" for i in range(40)).encode()
    return {f"variant-{number}/design.html": b"<html>\n" + html + b"</html>\n",
            f"variant-{number}/design.md": f"# variant {number}\nby {author}\n".encode()}


def paths_of(manifest):
    return sorted({*manifest.get("tracked_changed", []), *manifest.get("untracked_included", [])})


def tree_state(root):
    """Files, HEAD, index bytes and the object store's listing: what verification may not touch."""
    files = {p.relative_to(root).as_posix(): p.read_bytes() for p in root.rglob("*")
             if p.is_file() and ".git" not in p.relative_to(root).parts}
    head = subprocess.run(["git", "rev-parse", "-q", "--verify", "HEAD"], cwd=root, capture_output=True).stdout
    index = root / ".git" / "index"
    objects = sorted((p.relative_to(root).as_posix(), p.stat().st_size)
                     for p in (root / ".git" / "objects").rglob("*") if p.is_file())
    return files, head, index.read_bytes() if index.exists() else None, objects


def store_bytes(directory):
    return {p.relative_to(directory).as_posix(): p.read_bytes() for p in directory.rglob("*") if p.is_file()}


def git_dir_state(root):
    """Every entry and byte under the tree's .git, a split index's shared halves included."""
    return {p.relative_to(root / ".git").as_posix(): p.read_bytes() if p.is_file() else None
            for p in (root / ".git").rglob("*")}


class Shared:
    """One shared Git tree, native children writing it, and their parent."""

    def __init__(self, tmp_path, monkeypatch, *, parent="parent", unborn=False, files=None):
        self.tmp, self.parent_id = tmp_path, parent
        self.system, self.tree, self.drive = tmp_path / "system", tmp_path / "shared", tmp_path / "state"
        init(self.system)
        init(self.tree, unborn=unborn, files=files)
        self.drive.mkdir()
        monkeypatch.setenv("OUROBOROS_DATA_DIR", str(self.drive))
        monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "advanced")
        monkeypatch.setenv("OUROBOROS_ALLOW_MUTATIVE_SUBAGENTS", "true")
        monkeypatch.setenv("OUROBOROS_SAFETY_MODE", "off")
        monkeypatch.setenv("OUROBOROS_CONSCIOUSNESS_SAFETY_MODE", "off")
        monkeypatch.setattr("ouroboros.tool_access._user_files_root", lambda: tmp_path)
        # The worker-side queue projection is its own tested seam; no child here is live.
        monkeypatch.setattr("ouroboros.task_status.task_has_live_queue_ownership", lambda _root, _tid: False)
        self.tasks, self.runs = {}, 0

    def ctx(self, task_id=None, **extra):
        return ToolContext(repo_dir=self.system, drive_root=self.drive, task_id=task_id or self.parent_id,
                           workspace_root=self.tree, workspace_mode="external", **extra)

    def admit(self, cid, root=None):
        root = root or self.tree
        constraint, folder, mode, error = _resolve_subagent_constraint(
            SimpleNamespace(REPO_DIR=self.system, DRIVE_ROOT=self.drive), tid=cid,
            requested_constraint={"mode": "acting_subagent", "surface": "external_workspace", "write_root": str(root)},
            workspace_root=str(root), workspace_mode="external", base_sha="", parent_task_id=self.parent_id)
        assert not error, error
        self.tasks[cid] = {"id": cid, "workspace_root": folder, "workspace_mode": mode, "task_constraint": constraint}

    def nanny_apply(self, cid, files):
        """One supervised session run: private copy of the shared files, private capture, in-tree apply."""
        self.runs += 1
        baseline, _tree, _head, _excluded = snapshot_commit(self.tree, message=f"{cid} run {self.runs}")
        private = self.tmp / "private" / f"{cid}-{self.runs}"
        private.parent.mkdir(exist_ok=True)
        git(self.tree, "worktree", "add", "-q", "--detach", str(private), baseline)
        self.write(files, private)
        capture = self.tmp / "captures" / f"{cid}-{self.runs}"
        _, manifest = write_workspace_patch_artifacts(
            private, capture, task={"metadata": {"workspace_preflight": {"git": {"head": baseline}}}})
        assert manifest["status"] == "ready_with_changes", manifest
        outcome = _locked_apply(self.ctx(cid), self.tree, capture / "workspace.patch", paths_of(manifest), baseline)
        assert outcome["proc"] is not None and outcome["proc"].returncode == 0, outcome
        assert not (outcome["drifted"] or outcome["drift_error"] or outcome["staging_failure"]), outcome

    def write(self, files, root=None):
        """A native/API writer (or the owner) writing straight into the tree, unstaged."""
        for rel, data in files.items():
            (root or self.tree).joinpath(rel).parent.mkdir(parents=True, exist_ok=True)
            (root or self.tree).joinpath(rel).write_bytes(data)

    def finish(self, cid, *, register=()):
        """The child's own headless finalization: the outer capture of the SHARED tree."""
        task = self.tasks[cid]
        write_task_result(self.drive, cid, "completed", result=f"{cid} wrote its variant",
                          parent_task_id=self.parent_id, root_task_id=self.parent_id, delegation_role="subagent",
                          workspace_root=task["workspace_root"], task_constraint=task["task_constraint"])
        child = ToolContext(repo_dir=self.system, drive_root=self.drive, task_id=cid,
                            workspace_root=task["workspace_root"], workspace_mode="external")
        for path in register:
            copy_file_to_task_artifacts(child, path)
        finalize_task_artifacts(self.drive, task)
        return json.loads((task_artifact_dir_path(self.drive, cid) / "workspace_patch.json").read_text())

    def integrate(self, cid, ctx=None, **args):
        registry = ToolRegistry(repo_dir=self.system, drive_root=self.drive)
        registry.set_context(ctx or self.ctx())
        return registry.execute_result("integrate_subagent_patch", {"task_id": cid, **args}).text

    def verdict_path(self, cid, decider=None):
        return task_artifact_dir_path(self.drive, decider or self.parent_id) / f"subagent_patch_verdict_{cid}.json"

    def verdict(self, cid, decider=None):
        return json.loads(self.verdict_path(cid, decider).read_text(encoding="utf-8"))

    def states(self, cid, decider=None):
        return {row["path"]: row["state"] for row in self.verdict(cid, decider)["path_findings"]}

    def disposition(self, cid):
        return load_effective_task_result(self.drive, cid).get("child_result_disposition") or ""

    def custody(self, decider=None):
        from ouroboros.delegate_custody_current import marks
        return marks(self.drive, decider or self.parent_id).get("patch_verdicts", [])

    def incident(self):
        """#2479's reported order: C and B land variants, A lands its own, A finishes, B refines."""
        for cid in ("A", "B", "C"):
            self.admit(cid)
        self.nanny_apply("C", variant("02", "C"))
        self.nanny_apply("B", variant("04", "B"))
        self.nanny_apply("A", variant("06", "A"))
        manifest = self.finish("A")
        assert set(variant("06", "A")) < set(paths_of(manifest)), "A's capture holds A's files plus more"
        assert "variant-04/design.md" in paths_of(manifest), "B's earlier file is inside A's capture"
        self.nanny_apply("B", {"variant-04/design.md": b"# variant 04\nrefined by B after A finished\n"})
        return manifest


@pytest.fixture
def shared(tmp_path, monkeypatch):
    return Shared(tmp_path, monkeypatch)


INCIDENT_STATES = {"variant-02/design.html": "unchanged", "variant-02/design.md": "unchanged",
                   "variant-04/design.html": "unchanged", "variant-04/design.md": "changed",
                   "variant-06/design.html": "unchanged", "variant-06/design.md": "unchanged"}


def test_incident_names_each_path_and_the_parent_accepts_explicitly(shared):
    s = shared
    s.incident()
    before, product = tree_state(s.tree), store_bytes(task_artifact_dir_path(s.drive, "A"))
    consumed = _child_result_sha256(load_effective_task_result(s.drive, "A"))

    out = s.integrate("A")
    assert "INTEGRATE_EXTERNAL_WORKSPACE_MISMATCH" in out and "changed 1: variant-04/design.md" in out, out
    assert "unchanged 5: variant-02/design.html" in out and "variant-06/design.md" in out, out
    assert "names no author" in out and "decision='accept'" in out, out
    assert s.states("A") == INCIDENT_STATES
    assert s.verdict("A")["outcome"] == "shared_workspace_mismatch" and s.disposition("A") == ""
    observed = s.verdict_path("A").read_bytes()

    # A judgment needs its reason; without one nothing is recorded or replaced.
    assert "TOOL_ARG_ERROR" in s.integrate("A", decision="accept")
    assert s.verdict_path("A").read_bytes() == observed

    out = s.integrate("A", decision="accept", reason="A's variant-06 is intact; variant-04 is B's to refine")
    assert out.startswith("OK: accepted external_workspace child A") and "not as verification" in out, out
    assert "observed shared_workspace_mismatch" in out and "changed 1: variant-04/design.md" in out, out
    accepted = s.verdict("A")
    assert (accepted["outcome"], accepted["observed_outcome"], accepted["applied"]) == (
        "accepted_shared_workspace", "shared_workspace_mismatch", False)
    assert s.states("A") == INCIDENT_STATES and any("variant-04/design.md" in c for c in accepted["conflicts"])
    assert accepted["reason"] == "A's variant-06 is intact; variant-04 is B's to refine"
    assert s.disposition("A") == "integrated"
    assert [(r["disposition"], r.get("observed_outcome")) for r in s.custody()] == [
        ("shared_workspace_mismatch", None), ("accepted_shared_workspace", "shared_workspace_mismatch")]
    # The observation the decision replaced stays byte-exact beside it, under publication's own name.
    kept = _collision(s.verdict_path("A"), hashlib.sha256(observed).hexdigest())
    assert kept.read_bytes() == observed
    assert sorted(v["outcome"] for v in retained_verdicts(s.drive, "parent", "A")) == [
        "accepted_shared_workspace", "shared_workspace_mismatch"]
    # Verification and acceptance changed no file, index, HEAD, object, captured byte or judged result.
    assert tree_state(s.tree) == before
    assert store_bytes(task_artifact_dir_path(s.drive, "A")) == product
    assert _child_result_sha256(load_effective_task_result(s.drive, "A")) == consumed


def test_owner_continue_successor_accepts_once_and_the_chain_respects_it(tmp_path, monkeypatch):
    s = Shared(tmp_path, monkeypatch, parent="pred")
    interrupted_root(s.drive)
    s.incident()
    successor = continue_from(s.drive)
    before, signature = tree_state(s.tree), result_signature(s.drive, "A")
    consumed = _child_result_sha256(load_effective_task_result(s.drive, "A"))

    out = s.integrate("A", s.ctx(successor))
    assert "INTEGRATE_EXTERNAL_WORKSPACE_MISMATCH" in out and "changed 1: variant-04/design.md" in out, out
    out = s.integrate("A", s.ctx(successor), decision="accept", reason="checked after the restart")
    assert out.startswith("OK: accepted") and "owner-Continue successor of pred" in out, out
    assert "INTEGRATE_DISPOSITION_FAILED" not in out, out
    receipt = s.verdict("A", successor)
    assert (receipt["outcome"], receipt["observed_outcome"]) == (
        "accepted_shared_workspace", "shared_workspace_mismatch")
    assert (receipt["decision_authority"], receipt["original_parent_task_id"]) == ("owner_continue", "pred")
    assert receipt["child_result_sha256"] == consumed and s.states("A", successor) == INCIDENT_STATES
    assert s.disposition("A") == ""  # the predecessor's own tree ledger stays as it was
    assert result_signature(s.drive, "A") == signature  # the author's record is never rewritten
    for again in ("apply", "reject", "accept"):
        out = s.integrate("A", s.ctx(successor), decision=again, reason="once more")
        assert "INTEGRATE_LINEAGE_FORBIDDEN" in out, out
        assert f"{successor} already recorded the decision 'accepted_shared_workspace'" in out, out
    interrupted_root(s.drive, successor)
    second = continue_from(s.drive, successor, "continue-nonce-2")
    out = s.integrate("A", s.ctx(second), decision="accept", reason="override")
    assert "INTEGRATE_LINEAGE_FORBIDDEN" in out and f"{successor} already recorded" in out, out
    assert not s.verdict_path("A", second).exists()
    assert tree_state(s.tree) == before


def test_decision_history_survives_promotion_from_the_deciders_own_drive(tmp_path, monkeypatch):
    s = Shared(tmp_path, monkeypatch, parent="pred")
    interrupted_root(s.drive)
    s.incident()
    successor = continue_from(s.drive)
    own_drive = task_state_dir(s.drive, successor) / "data"
    ctx = s.ctx(successor, budget_drive_root=str(s.drive))
    ctx.drive_root = own_drive
    name = "subagent_patch_verdict_A.json"
    assert "INTEGRATE_EXTERNAL_WORKSPACE_MISMATCH" in _integrate_subagent_patch(ctx, task_id="A")
    observed = (task_artifacts_dir(own_drive, successor) / name).read_bytes()
    with monkeypatch.context() as patched:  # its custody row never lands: the artifact must carry it
        patched.setattr("ouroboros.delegate_custody.emit", lambda *_a, **_k: False)
        out = _integrate_subagent_patch(ctx, task_id="A", decision="accept", reason="custody down")
    assert out.startswith("OK: accepted") and "INTEGRATE_DISPOSITION_FAILED" in out, out
    accepted = (task_artifacts_dir(own_drive, successor) / name).read_bytes()
    staging = tmp_path / "staged"
    staging.mkdir()
    with task_custody_lock(s.drive, successor) as acquired:  # the existing promotion, then the drive goes
        assert acquired
        _publish(s.drive, successor, _child_store_plan(s.drive, own_drive, successor, {}, {}, staging))
        shutil.rmtree(own_drive)
    canonical = task_artifact_dir_path(s.drive, successor)
    kept = _collision(canonical / name, hashlib.sha256(observed).hexdigest()).name
    assert {p.name: p.read_bytes() for p in canonical.glob("subagent_patch_verdict_A*.json")} == {
        name: accepted, kept: observed}
    interrupted_root(s.drive, successor)
    second = continue_from(s.drive, successor, "continue-nonce-2")
    out = s.integrate("A", s.ctx(second), decision="accept", reason="the decision was lost?")
    assert "INTEGRATE_LINEAGE_FORBIDDEN" in out, out
    assert f"{successor} already recorded the decision 'accepted_shared_workspace'" in out, out


@pytest.mark.parametrize("change, state", [("edit", "changed"), ("delete", "removed")])
def test_a_childs_own_change_after_capture_is_named_and_never_absorbed(shared, change, state):
    s = shared
    for cid in ("A", "B"):
        s.admit(cid)
    s.nanny_apply("B", variant("04", "B"))
    s.nanny_apply("A", variant("06", "A"))
    s.finish("A")
    own = s.tree / "variant-06" / "design.md"
    own.write_bytes(b"changed after A finished\n") if change == "edit" else own.unlink()
    out = s.integrate("A")
    assert "INTEGRATE_EXTERNAL_WORKSPACE_MISMATCH" in out and f"{state} 1: variant-06/design.md" in out, out
    assert s.states("A")["variant-06/design.md"] == state
    assert s.states("A")["variant-04/design.md"] == "unchanged"
    assert s.verdict("A")["outcome"] == "shared_workspace_mismatch" and s.disposition("A") == ""


@pytest.mark.parametrize("where", ["inside", "outside"])
def test_same_file_edits_inside_and_outside_the_captured_hunk(tmp_path, monkeypatch, where):
    lines = [f"line {i}\n".encode() for i in range(40)]
    s = Shared(tmp_path, monkeypatch, files={"doc.txt": b"".join(lines)})
    s.admit("A")
    s.write({"doc.txt": b"".join(lines[:2] + [b"line 2 by A\n"] + lines[3:])})
    s.finish("A")
    edited = 4 if where == "inside" else 35  # the captured hunk spans lines 0-5 (A's line 2 plus context)
    current = (s.tree / "doc.txt").read_bytes().splitlines(keepends=True)
    current[edited] = b"edited later\n"
    s.write({"doc.txt": b"".join(current)})
    out = s.integrate("A")
    assert "changed 1: doc.txt" in out and s.states("A") == {"doc.txt": "changed"}, out
    if where == "inside":
        assert "INTEGRATE_EXTERNAL_WORKSPACE_MISMATCH" in out and s.disposition("A") == "", out
    else:  # reverse applicability, as before: it passes and says what it does not prove
        assert "Verified external_workspace child A" in out and "reverse check still passes" in out, out
        assert s.disposition("A") == "integrated"


@pytest.mark.parametrize("after, expected", [
    ("source_recreated", {"keep.txt": "recreated", "kept.txt": "unchanged"}),
    ("destination_edited", {"keep.txt": "unchanged", "kept.txt": "changed"}),
])
def test_a_rename_reports_both_sides(tmp_path, monkeypatch, after, expected):
    text = b"".join(f"kept line {i}\n".encode() for i in range(12))
    s = Shared(tmp_path, monkeypatch, files={"keep.txt": text})
    s.admit("A")
    git(s.tree, "mv", "keep.txt", "kept.txt")
    s.write({"kept.txt": text + b"one more line\n"})
    s.finish("A")
    patch = (task_artifact_dir_path(s.drive, "A") / "workspace.patch").read_bytes()
    assert b"rename from keep.txt" in patch and b"rename to kept.txt" in patch
    s.write({"keep.txt": text} if after == "source_recreated" else {"kept.txt": b"rewritten\n"})
    out = s.integrate("A")
    assert "INTEGRATE_EXTERNAL_WORKSPACE_MISMATCH" in out and s.states("A") == expected, out


@pytest.mark.parametrize("kind", ["binary_patch", "file_reference"])
def test_binary_patches_and_file_references_report_their_state(tmp_path, monkeypatch, kind):
    s = Shared(tmp_path, monkeypatch, files={"frame.bin": b"\x00\x01base frame\xff"})
    if kind == "file_reference":
        monkeypatch.setattr("ouroboros.workspace_patch_capture._PATCH_FILE_REFERENCE_BYTES", 1)
    s.admit("A")
    s.write({"frame.bin": b"\x00\x01A's frame\xff\xfe"})
    manifest = s.finish("A")
    patch = task_artifact_dir_path(s.drive, "A") / "workspace.patch"
    if kind == "file_reference":
        assert [row["path"] for row in manifest["file_output_changes"]] == ["frame.bin"] and not patch.exists()
    else:
        assert b"GIT binary patch" in patch.read_bytes()
    s.write({"frame.bin": b"\x00\x01a sibling's frame\xff"})
    out = s.integrate("A")
    assert "INTEGRATE_EXTERNAL_WORKSPACE_MISMATCH" in out and "changed 1: frame.bin" in out, out
    finding = s.verdict("A")["path_findings"]
    basis = "exact" if kind == "file_reference" else "git"
    assert finding == [{"path": "frame.bin", "state": "changed", "basis": basis}]
    out = s.integrate("A", decision="accept", reason="the sibling's frame supersedes A's")
    assert out.startswith("OK: accepted") and s.verdict("A")["path_findings"] == finding, out
    assert s.disposition("A") == "integrated"


@pytest.mark.parametrize("case", ["unborn_base", "parent_commit"])
def test_unborn_bases_and_parent_commits_keep_their_meaning(tmp_path, monkeypatch, case):
    s = Shared(tmp_path, monkeypatch, unborn=case == "unborn_base")
    s.admit("A")
    s.write(variant("06", "A"))
    manifest = s.finish("A")
    if case == "unborn_base":
        assert manifest["base_head"] == "(unborn)" and manifest["base_is_empty_tree"]
        s.write({"variant-06/design.md": b"# variant 06\nrefined in the unborn tree\n"})
        out = s.integrate("A")
        assert "INTEGRATE_EXTERNAL_WORKSPACE_MISMATCH" in out and "unknown" not in out, out
        assert s.states("A") == {"variant-06/design.html": "unchanged", "variant-06/design.md": "changed"}
        return
    git(s.tree, "add", "-A")
    git(s.tree, "commit", "-qm", "the parent commits A's work")
    out = s.integrate("A")
    assert "Verified external_workspace child A" in out and "unchanged 2:" in out, out
    assert set(s.states("A").values()) == {"unchanged"} and s.disposition("A") == "integrated"


def test_a_capture_without_a_recorded_base_reports_unknown_states_never_inferred(shared):
    s = shared
    s.incident()
    path = task_artifact_dir_path(s.drive, "A") / "workspace_patch.json"
    legacy = json.loads(path.read_text())
    for key in ("base_ref", "base_head"):  # an older capture recorded neither
        legacy.pop(key)
    path.write_text(json.dumps(legacy))
    out = s.integrate("A")
    assert "INTEGRATE_EXTERNAL_WORKSPACE_MISMATCH" in out and "unknown 6:" in out, out
    assert "patch paths unknown: the capture records no Git base object" in out, out
    assert set(s.states("A").values()) == {"unknown"}
    assert s.verdict("A")["path_findings_note"].startswith("the capture records no Git base object")
    out = s.integrate("A", decision="accept", reason="checked the files by hand; the capture predates bases")
    assert out.startswith("OK: accepted") and "unknown 6:" in out, out
    assert set(s.states("A").values()) == {"unknown"} and s.disposition("A") == "integrated"


def test_excluded_and_ignored_files_are_disclosed_not_checked(tmp_path, monkeypatch):
    s = Shared(tmp_path, monkeypatch, files={".gitignore": b"build/\n"})
    s.admit("A")
    s.write({**variant("06", "A"), "build/out.log": b"ignored output\n", ".env": b"TOKEN=fixture\n"})
    manifest = s.finish("A")
    assert [row["path"] for row in manifest["sensitive_blocked"]] == [".env"]
    assert "build/out.log" not in paths_of(manifest)
    s.write({"variant-06/design.md": b"# variant 06\nrefined later\n"})
    for args in ({}, {"decision": "accept", "reason": "the excluded files are not deliverables"}):
        out = s.integrate("A", **args)
        assert "EXCLUDED from this patch by capture policy: .env" in out, out
        assert set(s.states("A")) == set(variant("06", "A")), "only captured paths are compared"
    assert s.disposition("A") == "integrated"


@pytest.mark.parametrize("case, expected", [
    ("missing_reason", "TOOL_ARG_ERROR"),
    ("long_reason", "TOOL_ARG_ERROR"),
    ("self_worktree", "INTEGRATE_ACCEPT_UNSUPPORTED"),
    ("genesis", "INTEGRATE_GENESIS_FORBIDDEN"),
    ("wrong_target", "INTEGRATE_EXTERNAL_WORKSPACE_TARGET_MISMATCH"),
    ("read_denied", "INTEGRATE_TARGET_FORBIDDEN"),
    ("not_a_child", "INTEGRATE_LINEAGE_FORBIDDEN"),
])
def test_accept_refusals_come_before_any_decision(shared, case, expected):
    s = shared
    s.incident()
    args, ctx = {"decision": "accept", "reason": "keep it"}, s.ctx()
    if case == "missing_reason":
        args.pop("reason")
    elif case == "long_reason":
        args["reason"] = "x" * 501
    elif case in ("self_worktree", "genesis"):
        row = read_result(s.drive, "A")
        row["task_constraint"] = {**row["task_constraint"], "surface": case}
        seed_result(s.drive, "A", row)
    elif case == "wrong_target":
        args["target_root"] = str(s.system)
    elif case == "read_denied":
        ctx.task_contract = {"resource_policy": {"protected_artifacts": [{
            "id": "black-box", "paths": [str(s.tree / "variant-06" / "design.md")], "deny": ["read_bytes"]}]}}
    else:
        ctx = s.ctx("stranger")
    before = tree_state(s.tree)
    out = s.integrate("A", ctx, **args)
    assert expected in out, out
    assert s.disposition("A") == "" and tree_state(s.tree) == before
    for decider in (s.parent_id, "stranger"):
        if s.verdict_path("A", decider).exists():
            assert s.verdict("A", decider)["outcome"] != "accepted_shared_workspace"
    assert all(row["disposition"] != "accepted_shared_workspace" for row in s.custody())


@pytest.mark.parametrize("observable", [True, False])
def test_direct_folder_outputs_take_the_same_acceptance(tmp_path, monkeypatch, observable):
    s = Shared(tmp_path, monkeypatch)
    plain = tmp_path / "plain"
    plain.mkdir()
    s.admit("A", plain)
    s.write({"report.txt": b"A's report\n", "notes.txt": b"A's notes\n"}, plain)
    manifest = s.finish("A", register=(plain / "report.txt", plain / "notes.txt"))
    assert manifest["capture_kind"] == "directory_direct"
    s.write({"report.txt": b"a sibling's report\n"}, plain)
    out = s.integrate("A")
    assert "INTEGRATE_DIRECTORY_OUTPUT_MISMATCH" in out and "changed 1: report.txt" in out, out
    assert "unchanged 1: notes.txt" in out and "decision='accept'" in out and s.disposition("A") == "", out
    observed = s.verdict_path("A").read_bytes()
    ctx = s.ctx()
    if not observable:
        ctx.task_contract = {"resource_policy": {"protected_artifacts": [{
            "id": "black-box", "paths": [str(plain / "report.txt")], "deny": ["hash"]}]}}
    out = s.integrate("A", ctx, decision="accept", reason="the sibling's report replaces A's")
    if not observable:
        assert "Acceptance was not recorded" in out and "RESOURCE_POLICY_BLOCKED" in out, out
        assert s.disposition("A") == "" and s.verdict("A")["outcome"] == "direct_output_mismatch"
        return
    assert out.startswith("OK: accepted") and "observed direct_output_mismatch" in out, out
    accepted = s.verdict("A")
    assert (accepted["outcome"], accepted["observed_outcome"]) == (
        "accepted_shared_workspace", "direct_output_mismatch")
    assert s.states("A") == {"notes.txt": "unchanged", "report.txt": "changed"}
    assert s.disposition("A") == "integrated"
    assert _collision(s.verdict_path("A"), hashlib.sha256(observed).hexdigest()).read_bytes() == observed


@pytest.mark.parametrize("subject", ["A", "run_0123abcd"])
def test_every_verdict_rewrite_keeps_the_earlier_bytes_for_both_pipelines(tmp_path, monkeypatch, subject):
    """The one writer serves integrate_subagent_patch and integrate_delegated_patch (run_<rid>) alike."""
    from ouroboros.tools.patch_verdict import write_patch_verdict

    s = Shared(tmp_path, monkeypatch)
    common = dict(reason="", files=["a.txt"], manifest={}, conflicts=[], protected=[])
    path = s.verdict_path(subject)
    assert write_patch_verdict(s.ctx(), subject, outcome="conflict", applied=False, **common) == str(path)
    earlier = path.read_bytes()
    assert write_patch_verdict(s.ctx(), subject, outcome="applied", applied=True, **common) == str(path)
    current = path.read_bytes()
    assert _collision(path, hashlib.sha256(earlier).hexdigest()).read_bytes() == earlier
    assert sorted(v["outcome"] for v in retained_verdicts(s.drive, "parent", subject)) == ["applied", "conflict"]
    # A collision name already holding other bytes refuses the rewrite: the current file stays.
    _collision(path, hashlib.sha256(current).hexdigest()).write_bytes(b"other bytes\n")
    assert write_patch_verdict(s.ctx(), subject, outcome="rejected", applied=False, **common) == ""
    assert path.read_bytes() == current
    rows = [row for row in s.custody() if row["child_task_id"] == subject]
    assert [row["disposition"] for row in rows] == ["conflict", "applied", "rejected"]
    assert rows[-1]["verdict_artifact_write_failed"] is True
    assert {row["pipeline"] for row in rows} == {"delegated" if subject.startswith("run_") else "subagent"}


def test_the_tool_schema_offers_accept_as_judgment_not_verification():
    schema = next(entry.schema for entry in get_tools() if entry.name == "integrate_subagent_patch")
    decision = schema["parameters"]["properties"]["decision"]
    assert decision["enum"] == ["apply", "reject", "accept"]
    assert "accept = external_workspace only" in decision["description"]
    assert "never verification" in schema["description"] and "names no author" in schema["description"]
    assert "required for accept" in schema["parameters"]["properties"]["reason"]["description"]


# Review repairs: each diagnosis is the capture's own, never wider than its authority, never a guess.


@pytest.mark.parametrize("name", ["report.txt", "new.txt"])  # tracked by the target's index, or untracked
def test_each_path_converts_as_the_capture_read_it(tmp_path, monkeypatch, name):
    """A deleted .gitattributes stays in the target's index: the capture's `git diff` read a tracked file through
    it, while `diff --no-index` read an untracked one through none. The diagnosis converts each the same way, so a
    file nobody touched since capture never reads changed; a later edit still does."""
    s = Shared(tmp_path, monkeypatch, files={".gitattributes": b"*.txt text\n", "report.txt": b"base\n"})
    git(s.tree, "config", "core.autocrlf", "false")
    s.admit("A")
    (s.tree / ".gitattributes").unlink()
    s.write({name: b"child report\r\n"})
    s.finish("A")
    out = s.integrate("A")
    assert s.states("A") == {".gitattributes": "unchanged", name: "unchanged"}, out
    assert "patch paths unknown" not in out, out
    if name == "report.txt":  # `git apply` reads no index, so its reverse check alone misses that copy
        assert "INTEGRATE_EXTERNAL_WORKSPACE_MISMATCH" in out and "unchanged 2:" in out, out
    else:
        assert "Verified external_workspace child A" in out and "reverse check still passes" not in out, out
    s.write({name: b"edited after capture\r\n"})
    out = s.integrate("A")
    assert "INTEGRATE_EXTERNAL_WORKSPACE_MISMATCH" in out and s.states("A")[name] == "changed", out


@pytest.mark.parametrize("runtime_mode, refined", [("light", True), ("advanced", False)])  # never Cyber Pro
def test_a_hash_denied_patch_path_is_never_hashed_and_stays_unknown(shared, monkeypatch, runtime_mode, refined):
    """A policy that lets the parent read a captured file but not hash it keeps the reverse check and an explicit
    acceptance; no Git digest of that file is taken, so its own state is unknown while the others stay known."""
    from ouroboros.config import get_runtime_mode

    s = shared
    if refined:
        s.incident()
    else:
        s.admit("A")
        s.nanny_apply("A", variant("06", "A"))
        s.finish("A")
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", runtime_mode)
    assert get_runtime_mode() == runtime_mode
    ctx = s.ctx()
    ctx.task_contract = {"resource_policy": {"protected_artifacts": [{
        "id": "no-hash", "paths": [str(s.tree / "variant-06" / "design.md")], "deny": ["hash"]}]}}
    run, hashing = subprocess.run, []

    def observed(args, *rest, **kwargs):
        if list(args[:1]) == ["git"] and {"update-index", "hash-object"} & set(args):
            hashing.append((" ".join(map(str, args)), bytes(kwargs.get("input") or b"")))
        return run(args, *rest, **kwargs)

    monkeypatch.setattr(subprocess, "run", observed)
    out = s.integrate("A", ctx)
    expected = ({**INCIDENT_STATES, "variant-06/design.md": "unknown"} if refined else
                {"variant-06/design.html": "unchanged", "variant-06/design.md": "unknown"})
    assert s.states("A") == expected, out
    assert "task policy denies hashing 1 patch path(s), so none was hashed" in s.verdict("A")["path_findings_note"]
    assert hashing, "the other paths were staged, so the diagnosis did run"
    assert not any("variant-06/design.md" in args or b"variant-06/design.md" in data for args, data in hashing)
    if not refined:  # the reverse check still decides; an unknown state is no tolerated edit
        assert "Verified external_workspace child A" in out and "reverse check still passes" not in out, out
        return
    assert "INTEGRATE_EXTERNAL_WORKSPACE_MISMATCH" in out and "unknown 1: variant-06/design.md" in out, out
    out = s.integrate("A", ctx, decision="accept", reason="variant-06 is A's and checked by hand; B refines 04")
    assert out.startswith("OK: accepted") and s.states("A") == expected, out
    assert s.disposition("A") == "integrated"
    assert not any("variant-06/design.md" in args or b"variant-06/design.md" in data for args, data in hashing)


def test_scratch_indexes_never_write_a_split_index_into_the_target(tmp_path, monkeypatch):
    s = Shared(tmp_path, monkeypatch)
    git(s.tree, "config", "core.splitIndex", "true")
    git(s.tree, "update-index", "--split-index")
    assert list((s.tree / ".git").glob("sharedindex.*")), "the target's own index is split"
    s.admit("A")
    s.write({**variant("06", "A"), "README.md": b"# shared design tree\nwith A's note\n"})  # untracked and tracked
    s.finish("A")
    s.write({"variant-06/design.md": b"# variant 06\nrefined later\n"})
    before = git_dir_state(s.tree), tree_state(s.tree)
    out = s.integrate("A")
    assert "INTEGRATE_EXTERNAL_WORKSPACE_MISMATCH" in out, out
    assert s.states("A") == {"README.md": "unchanged", "variant-06/design.html": "unchanged",
                             "variant-06/design.md": "changed"}, out
    out = s.integrate("A", decision="accept", reason="the refinement is wanted")
    assert out.startswith("OK: accepted"), out
    assert (git_dir_state(s.tree), tree_state(s.tree)) == before


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="FIFOs are POSIX")
def test_paths_no_git_entry_can_stand_for_are_type_changes_beside_known_paths(tmp_path, monkeypatch):
    s = Shared(tmp_path, monkeypatch, files={"keep.txt": b"keep\n", "edited.txt": b"edit\n", "gone": b"a file\n"})
    s.admit("A")
    s.write({"keep.txt": b"keep, by A\n", "edited.txt": b"edit, by A\n", "dir_now": b"A's file\n",
             "fifo_now": b"A's file\n", "repo_now": b"A's file\n", "linked/file.txt": b"A's linked file\n"})
    (s.tree / "gone").unlink()
    s.finish("A")
    s.write({"edited.txt": b"edited later\n", "gone/inner.txt": b"a directory now\n"})
    for name in ("dir_now", "repo_now"):
        (s.tree / name).unlink()
        (s.tree / name).mkdir()
    git(s.tree / "repo_now", "init", "-q")  # a repository where the capture recorded a file, not a gitlink
    (s.tree / "fifo_now").unlink()
    os.mkfifo(s.tree / "fifo_now")
    (s.tree / "linked").rename(s.tree / "elsewhere")
    (s.tree / "linked").symlink_to("elsewhere")
    before = tree_state(s.tree)
    out = s.integrate("A")
    assert "INTEGRATE_EXTERNAL_WORKSPACE_MISMATCH" in out, out
    assert s.states("A") == {"keep.txt": "unchanged", "edited.txt": "changed", "gone": "recreated",
                             "dir_now": "changed", "fifo_now": "changed", "repo_now": "changed",
                             "linked/file.txt": "changed"}, out
    assert "path_findings_note" not in s.verdict("A") and "patch paths unknown" not in out, out
    assert tree_state(s.tree) == before


def test_a_recreated_path_the_reverse_check_tolerates_is_named_an_edit(tmp_path, monkeypatch):
    s = Shared(tmp_path, monkeypatch, files={"gone": b"a file\n"})
    s.admit("A")
    (s.tree / "gone").unlink()
    s.finish("A")
    s.write({"gone/inner.txt": b"a directory now\n"})
    out = s.integrate("A")
    assert "Verified external_workspace child A" in out and "recreated 1: gone" in out, out
    assert "reverse check still passes" in out and s.states("A") == {"gone": "recreated"}, out


def test_a_failed_stage_leaves_the_other_paths_known_and_says_so(tmp_path, monkeypatch):
    s = Shared(tmp_path, monkeypatch, files={"keep.txt": b"keep\n", "edited.txt": b"edit\n", "base.boom": b"base\n"})
    s.admit("A")
    s.write({"keep.txt": b"keep, by A\n", "edited.txt": b"edit, by A\n", "base.boom": b"A's base\n",
             "frame.boom": b"A's frame\n", "notes.txt": b"A's notes\n", "fresh.txt": b"A's fresh\n"})
    s.finish("A")
    # A required clean filter that fails after capture: Git cannot stage the .boom files now, tracked or
    # untracked, while their batch mates still can.
    s.write({"edited.txt": b"edited later\n", "fresh.txt": b"edited later\n", ".gitattributes": b"*.boom filter=boom\n"})
    git(s.tree, "config", "filter.boom.clean", "false")
    git(s.tree, "config", "filter.boom.required", "true")
    out = s.integrate("A")
    assert s.states("A") == {"keep.txt": "unchanged", "edited.txt": "changed", "base.boom": "unknown",
                             "frame.boom": "unknown", "notes.txt": "unchanged", "fresh.txt": "changed"}, out
    note = s.verdict("A")["path_findings_note"]
    assert "Git could not stage 1 current tracked file(s)" in note and "1 current untracked file(s)" in note, note
    assert "rebuild" not in note, note


@pytest.mark.parametrize("held", ["sparse", "assume_unchanged"])
def test_index_held_paths_read_as_the_capture_read_them(tmp_path, monkeypatch, held):
    """Git, and so the capture, reads a skip-worktree (sparse) or assume-unchanged path from its index entry; so
    does the diagnosis: sparsity is never a removal, an edit Git does not read never a change. The reverse
    check still reads the worktree, and accept stays available."""
    s = Shared(tmp_path, monkeypatch)
    s.write({"in/a.txt": b"visible\n", "out/report.txt": b"base\n"})
    git(s.tree, "add", "-A")
    git(s.tree, "commit", "-qm", "a layout a sparse checkout can narrow")
    s.admit("A")
    s.write({"out/report.txt": b"A's report\n"})
    s.finish("A")
    git(s.tree, "add", "-A")
    git(s.tree, "commit", "-qm", "the parent commits A's work")
    if held == "sparse":
        git(s.tree, "sparse-checkout", "set", "--cone", "in")
        assert not (s.tree / "out" / "report.txt").exists()
    else:
        git(s.tree, "update-index", "--assume-unchanged", "out/report.txt")
        s.write({"out/report.txt": b"edited where Git does not look\n"})
    out = s.integrate("A")
    assert "INTEGRATE_EXTERNAL_WORKSPACE_MISMATCH" in out and s.states("A") == {"out/report.txt": "unchanged"}, out
    out = s.integrate("A", decision="accept", reason="out/report.txt is A's committed work, held by the index")
    assert out.startswith("OK: accepted") and s.states("A") == {"out/report.txt": "unchanged"}, out


@pytest.mark.parametrize("attribute", ["text", "text=auto"])
def test_the_git_basis_names_the_current_representation_not_bytes(tmp_path, monkeypatch, attribute):
    """The owner's attribute change after capture moves how Git represents a file nobody touched. That is no
    regression of the diagnosis: its finding names its basis (the current Git representation under the target's
    attributes and config now, bytes not compared) rather than claiming the bytes changed."""
    s = Shared(tmp_path, monkeypatch)
    git(s.tree, "config", "core.autocrlf", "false")
    s.admit("A")
    s.write({"notes.txt": b"A's notes\r\n"})
    s.finish("A")
    s.write({".gitattributes": f"*.txt {attribute}\n".encode()})  # notes.txt keeps its bytes
    out = s.integrate("A")
    assert s.states("A") == {"notes.txt": "changed"}, out
    assert "current Git representation" in out and "bytes not compared" in out, out
    out = s.integrate("A", decision="accept", reason="only the representation moved; the bytes are A's")
    assert out.startswith("OK: accepted") and "bytes not compared" in out, out


@pytest.mark.skipif(os.name == "nt", reason="POSIX executable bits and symlinks")
@pytest.mark.parametrize("kind", ["regular", "executable", "symlink"])
def test_untracked_modes_follow_no_index_capture_not_the_index_options(tmp_path, monkeypatch, kind):
    s = Shared(tmp_path, monkeypatch)
    git(s.tree, "config", "core.autocrlf", "false")
    git(s.tree, "config", "core.fileMode", "false")
    git(s.tree, "config", "core.symlinks", "false")
    s.admit("A")
    path = s.tree / "new.txt"
    if kind == "symlink":
        path.symlink_to("README.md")
    else:
        s.write({"new.txt": b"A's output\n"})
        path.chmod(0o755 if kind == "executable" else 0o644)
    s.finish("A")
    before = tree_state(s.tree), git_dir_state(s.tree)
    out = s.integrate("A")
    assert s.states("A") == {"new.txt": "unchanged"}, out
    assert (tree_state(s.tree), git_dir_state(s.tree)) == before
    if kind == "symlink":
        path.unlink()
        path.symlink_to("another-target")
    else:
        path.chmod(0o644 if kind == "executable" else 0o755)
    # Match capture's actual no-index reader, not update-index's configurable treatment of modes.
    current = subprocess.run(
        ["git", "diff", "--no-index", "--binary", "--no-ext-diff", "--no-color", "--", os.devnull, "new.txt"],
        cwd=s.tree, capture_output=True, check=False,
    )
    assert current.returncode == 1 and current.stdout != (task_artifact_dir_path(s.drive, "A") / "workspace.patch").read_bytes()
    before = tree_state(s.tree), git_dir_state(s.tree)
    out = s.integrate("A")
    assert s.states("A") == {"new.txt": "changed"}, out
    assert (tree_state(s.tree), git_dir_state(s.tree)) == before


def test_a_kept_verdict_is_written_through_the_sweeps_atomic_temporary(tmp_path, monkeypatch):
    from ouroboros.temp_sweep import is_atomic_temp
    from ouroboros.tools import patch_verdict

    s = Shared(tmp_path, monkeypatch)
    common = dict(reason="", files=["a.txt"], manifest={}, conflicts=[], protected=[])
    assert patch_verdict.write_patch_verdict(s.ctx(), "A", outcome="conflict", applied=False, **common)
    link, temporaries = os.link, []

    def observed(source, destination, *args, **kwargs):
        temporaries.append(pathlib.Path(source).name)
        return link(source, destination, *args, **kwargs)

    monkeypatch.setattr(os, "link", observed)
    assert patch_verdict.write_patch_verdict(s.ctx(), "A", outcome="applied", applied=True, **common)
    assert len(temporaries) == 1 and is_atomic_temp(temporaries[0]), temporaries
    assert not [p.name for p in s.verdict_path("A").parent.iterdir() if ".tmp" in p.name]


def unwritable(*_args, **_kwargs):
    raise OSError("the artifact store refused the write")


@pytest.mark.parametrize("decision, failed", [
    ("accept", ["verdict_artifact_write_failed"]), ("accept", ["custody_row_write_failed"]),
    ("accept", ["verdict_artifact_write_failed", "custody_row_write_failed"]),
    ("apply", ["custody_row_write_failed"]), ("reject", ["verdict_artifact_write_failed"]),
])
def test_a_decision_names_each_failed_write_of_its_own_evidence(shared, monkeypatch, decision, failed):
    s = shared
    s.admit("A")
    s.write(variant("06", "A"))
    s.finish("A")
    if decision == "accept":
        s.write({"variant-06/design.md": b"# variant 06\nedited later\n"})
    if "verdict_artifact_write_failed" in failed:
        monkeypatch.setattr("ouroboros.tools.patch_verdict.atomic_write_json", unwritable)
    if "custody_row_write_failed" in failed:
        monkeypatch.setattr("ouroboros.delegate_custody.emit", lambda *_a, **_k: False)
    out = s.integrate("A", decision=decision, reason="decided after reading the findings")
    assert out.startswith({"accept": "OK: accepted", "apply": "✅ Verified", "reject": "🚫 Rejected"}[decision]), out
    assert "This decision's evidence is incomplete" in out, out
    for fact in ("verdict_artifact_write_failed", "custody_row_write_failed"):
        assert (fact in out) == (fact in failed), out
    assert s.disposition("A") == ("irrelevant" if decision == "reject" else "integrated")


def test_a_successors_acceptance_names_its_unwritten_verdict(tmp_path, monkeypatch):
    s = Shared(tmp_path, monkeypatch, parent="pred")
    interrupted_root(s.drive)
    s.admit("A")
    s.write(variant("06", "A"))
    s.finish("A")
    s.write({"variant-06/design.md": b"# variant 06\nedited later\n"})
    successor = continue_from(s.drive)
    monkeypatch.setattr("ouroboros.tools.patch_verdict.atomic_write_json", unwritable)
    out = s.integrate("A", s.ctx(successor), decision="accept", reason="the later edit is the deliverable")
    assert out.startswith("OK: accepted") and "owner-Continue successor of pred" in out, out
    assert "verdict_artifact_write_failed" in out and "INTEGRATE_DISPOSITION_FAILED" not in out, out


def test_continue_and_the_tool_tell_apply_verify_accept_and_reject_apart():
    from ouroboros.owner_continue import work_order_text

    text = work_order_text("pred", "owner_restart", {})
    assert "can be applied or verified" in text and "accepted as it stands when it is a shared result" in text, text
    assert "or rejected" in text and "can be accepted or rejected" not in text, text
    description = next(entry.schema for entry in get_tools() if entry.name == "integrate_subagent_patch")["description"]
    assert "apply/verify, accept a shared result as it stands, or reject" in description
    assert "may also accept or reject" not in description
