"""Work-domain material flows at their real consumers, beyond the basic tools (``tests/test_work_domain``).

The fake container of ``test_execution_domain`` (its ``sh`` is dash, its group observer real) stands in for the
work domain; the audit hook of ``test_work_domain`` records anything the service process does to a work path by its
own pathname. Covered: a workspace at the mapping's own root; an unlistable subtree read as a hold or a failure,
never as an empty success; hard-linked outputs and supplied member provenance; streamed transfers; an unconfirmed
work-domain end; a foreground command's whole group; staged inputs held by their task; a native skill-payload run
drafted in its work copy; an installed script skill run from its verified work copy with dependencies built there.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import pathlib
import stat
import subprocess
import sys
import time
import zipfile

import pytest

from ouroboros import work_domain
from tests.test_execution_domain import _registry
from tests.test_execution_domain import env as env  # fixture
from tests.test_work_domain import _managed, _work_repo, service_effects_on


def _records(env):
    return sorted(env.root.rglob("workspace_executor_processes/*.json"))


@pytest.mark.parametrize("link", [False, True])
def test_archive_reads_one_work_domain_snapshot_never_a_service_path(env, monkeypatch, link):
    """Real registry and ZIP consumers: regular files use the confined descriptor; links resolve in W. The link
    deliberately points to different, readable S bytes here, so opening it on S would leak the sentinel. The
    mocked W cat is the namespace boundary, not an OS-isolation claim."""
    _managed(env, monkeypatch)
    repo = _work_repo(env.work / "app", {"README.md": "work\n"})
    registry = _registry(env, repo)
    source = io.BytesIO()
    with zipfile.ZipFile(source, "w") as archive:
        archive.writestr("work.txt", "WORK CONTENT")
    payload = source.getvalue()
    target, secret = repo / "input.zip", env.root / "service-only.zip"
    with zipfile.ZipFile(secret, "w") as archive:
        archive.writestr("secret.txt", "SERVICE SENTINEL")
    calls = []
    if link:
        target.symlink_to(secret)
        def work_cat(argv, **kwargs):
            assert argv == ["cat", "--", work_domain.container_path(target)]
            calls.append(argv)
            kwargs["stdout"].write(payload)
            return subprocess.CompletedProcess(argv, 0, b"", b"")
        monkeypatch.setattr(work_domain, "run_in_work", work_cat)
    else:
        target.write_bytes(payload)
    with service_effects_on(secret, env.work, allowed_opens=(str(env.work),)) as seen:
        listing = registry.execute("read_archive", {"path": "input.zip"})
        text = registry.execute("read_archive", {"path": "input.zip", "member": "work.txt"})
    assert not seen, seen
    assert "work.txt" in listing and "WORK CONTENT" in text, (listing, text)
    assert "SERVICE SENTINEL" not in text and "secret.txt" not in listing
    assert bool(calls) == link
    if link:
        def unavailable(argv, **kwargs):
            return subprocess.CompletedProcess(argv, 1, b"", b"not visible in W")
        monkeypatch.setattr(work_domain, "run_in_work", unavailable)
        refused = registry.execute("read_archive", {"path": "input.zip", "member": "secret.txt"})
        assert "SERVICE SENTINEL" not in refused and "⚠️" in refused, refused


@pytest.mark.parametrize("git_repo", [True, False])
def test_structured_queries_read_and_execute_only_through_the_work_domain(env, monkeypatch, git_repo):
    """All query operations keep useful results. The fake W view of a planted symlink differs from S's bytes;
    no service-side open or Git is permitted, including a cached second query and the filesystem fallback."""
    _managed(env, monkeypatch)
    files = {"src/leaf.py": "def ping():\n    return 1\n", "src/caller.py":
             "from .leaf import ping\ndef caller():\n    return ping()\n",
             "src/facade.py": "from src.leaf import ping  # noqa: F401\n",
             "src/consumer.py": "from src.facade import ping\n",
             "ouroboros/domains.toml": '[domains]\nD01 = "Work code"\n[modules]\n"src/leaf.py" = "D01"\n'
             '"src/caller.py" = "D01"\n"src/facade.py" = "D01"\n"src/consumer.py" = "D01"\n'}
    repo = _work_repo(env.work / "app", files)
    if not git_repo:
        (repo / ".git").rename(env.root / "saved-git")
    registry = _registry(env, repo)
    secret = env.root / "service-only.py"
    secret.write_text("def SERVICE_INTERNAL(): pass\n", encoding="utf-8")
    link = repo / "poison.py"
    link.symlink_to(secret)
    original, work_reads = work_domain.read_in_work, []
    def work_view(path, **kwargs):
        if pathlib.Path(path) == link:
            work_reads.append(str(path))
            return b"def work_link(): pass\n"
        return original(path, **kwargs)
    monkeypatch.setattr(work_domain, "read_in_work", work_view)
    queries = [({"op": "digest"}, "leaf.py"), ({"op": "symbols", "path": "src/leaf.py"}, "ping"),
               ({"op": "definition", "query": "ping"}, "leaf.py"),
               ({"op": "references", "query": "ping"}, "caller.py"),
               ({"op": "callers", "query": "ping"}, "caller.py"),
               ({"op": "callees", "query": "caller", "path": "src/caller.py"}, "ping"),
               ({"op": "impact", "query": "src/leaf.py"}, "caller.py"),
               ({"op": "relevant_files", "query": "ping"}, "leaf.py"),
               ({"op": "structural", "query": "FunctionDef", "path": "src"}, "leaf.py"),
               ({"op": "architecture", "query": "owner_of src/leaf.py"}, "D01"),
               ({"op": "architecture", "query": "facade_consumers ping"}, "facade.py"),
               ({"op": "definition", "query": "ping"}, "leaf.py")]
    with service_effects_on(env.work, secret, allowed_opens=(str(env.work),)) as seen:
        answers = [(registry.execute("query_code", args), expected) for args, expected in queries]
    assert not seen, seen
    for answer, expected in answers:
        assert expected in answer and "QUERY_CODE_ERROR" not in answer and "SERVICE_INTERNAL" not in answer, answer
    if git_repo:
        assert work_reads, "the inventory's symlink source was read through W, never S"


@pytest.mark.parametrize("failure", ["head", "enumeration", "outline", "cache", "occurrences", "structural", "carrier"])
def test_code_queries_keep_unsettled_work_distinct_from_missing_source(env, monkeypatch, failure):
    """A started work read with unknown group cleanup cannot become a completed local query or a fallback.
    Its observed bytes and cause reach the registry result; ordinary read failures still keep their old limits."""
    _managed(env, monkeypatch)
    repo = _work_repo(env.work / "app", {"leaf.py": "def ping(): return 1\n",
                                       "ouroboros/domains.toml": '[domains]\nD01="Work"\n[modules]\n"leaf.py"="D01"\n'})
    registry = _registry(env, repo)
    if failure == "cache":
        assert "ping" in registry.execute("query_code", {"op": "symbols", "path": "leaf.py"})
    observed = subprocess.CompletedProcess(["git", "read"], 0, "OBSERVED SOURCE", "OBSERVED DIAGNOSTIC")
    error = work_domain.WorkDomainUnsettled("backend unconfirmed; custody retained", observed)
    original_run, original_read = work_domain.run, work_domain.read_bytes

    def run(cmd, **kwargs):
        if list(cmd[:2]) == ["git", "rev-parse" if failure == "head" else "ls-files"]:
            raise error
        return original_run(cmd, **kwargs)

    def read(path, **kwargs):
        if pathlib.Path(path).name == ("domains.toml" if failure == "carrier" else "leaf.py"):
            raise error
        return original_read(path, **kwargs)

    monkeypatch.setattr(work_domain, "run" if failure in {"head", "enumeration"} else "read_bytes",
                        run if failure in {"head", "enumeration"} else read)
    args = ({"op": "references", "query": "ping"} if failure == "occurrences" else
            {"op": "structural", "query": "FunctionDef"} if failure == "structural" else
            {"op": "architecture", "query": "owner_of leaf.py"} if failure == "carrier" else
            {"op": "symbols", "path": "leaf.py"})
    result = registry.execute_result("query_code", args)
    answer = result.text
    assert "backend unconfirmed" in answer and "OBSERVED SOURCE" in answer and "OBSERVED DIAGNOSTIC" in answer, answer
    assert result.status == "error" and result.meta.get("backend_settled") is False, result
    assert result.meta.get("operation_outcome") != "completed_no_effect", result
    assert error.result is observed


def test_a_workspace_at_the_mapping_root_lists_reads_searches_and_captures(env, monkeypatch):
    """The active workspace is the mapping's own root: its metadata comes from the root the descent holds, so
    listing (flat and tree), reading, search, Git status and capture work there as below it, and the service
    opens nothing there by its own pathname beyond that root."""
    from ouroboros.workspace_patch_capture import write_workspace_patch_artifacts

    _managed(env, monkeypatch)
    repo = _work_repo(env.work, {"src/main.py": "VALUE = 1\n", "README.md": "readme\n"})
    registry = _registry(env, repo)
    (repo / "src" / "main.py").write_text("VALUE = 2\n", encoding="utf-8")
    with service_effects_on(env.work, allowed_opens=(str(env.work),)) as seen:
        flat = registry.execute("list_files", {"root": "active_workspace", "path": "."})
        tree = registry.execute("list_files", {"root": "active_workspace", "path": ".", "depth": 3})
        read = registry.execute("read_file", {"root": "active_workspace", "path": "README.md"})
        found = registry.execute("search_code", {"query": "VALUE", "root": "active_workspace"})
        status = registry.execute("vcs_read", {"action": "status"})
        _records_out, manifest = write_workspace_patch_artifacts(repo, env.root / "capture", task={"metadata": {}})
    assert seen == [], seen
    assert "README.md" in flat and "src/" in flat, flat
    assert "src/main.py" in tree and "readme" in read and "src/main.py:1" in found, (tree, read, found)
    assert "main.py" in status and manifest["status"] == "ready_with_changes", (status, manifest)
    assert b"VALUE = 2" in (env.root / "capture" / "workspace.patch").read_bytes()


def test_an_unlistable_work_subtree_is_a_hold_or_a_failure_never_an_empty_success(env, monkeypatch):
    """A folder the service cannot list is never read as empty: a scratch's retention counts it as a hold, a
    folder output's packaging fails, and a workspace's named evidence asks retention, which records the failure."""
    from ouroboros import session_scratch
    from ouroboros.artifact_directories import copy_directory_to_task_artifacts
    from ouroboros.workspace_file_outputs import _direct_named_evidence

    _managed(env, monkeypatch)
    scratch, out, app = env.work / "obs-scratch", env.work / "out", env.work / "app"
    locked = [scratch / "locked", out / "locked", app / ".review-evidence" / "locked"]
    for folder in locked:
        folder.mkdir(parents=True)
        (folder / "kept.txt").write_text("kept\n", encoding="utf-8")
    (out / "a.txt").write_text("a\n", encoding="utf-8")
    registry = _registry(env, app)
    for folder in locked:
        folder.chmod(0)
    try:
        hold = session_scratch.retention_hold(scratch)
        with pytest.raises(OSError):
            copy_directory_to_task_artifacts(registry._ctx, out)
        retained = _direct_named_evidence(app, {"metadata": {}}, env.root / "retained-evidence", None)
    finally:
        for folder in locked:
            folder.chmod(0o755)
    assert hold is not None and hold["class"] == "producer_retained" and "unreadable" in hold["example"], hold
    assert retained is not None and retained["status"] == "capture_failed", retained


def test_hardlinked_outputs_arrive_and_supplied_members_keep_their_provenance(env, monkeypatch):
    """Two requested files sharing an inode both arrive (the second as the first's bytes again), while a link and a
    special file still refuse the intake; a member a caller supplied from another place is packaged from there."""
    from ouroboros.artifact_directories import copy_directory_to_task_artifacts

    _managed(env, monkeypatch)
    out = env.work / "app" / "out"
    out.mkdir(parents=True)
    (out / "a.bin").write_bytes(b"A" * 4096)
    os.link(out / "a.bin", out / "b.bin")
    (out / "link").symlink_to(out / "a.bin")
    os.mkfifo(out / "pipe")
    digest = hashlib.sha256(b"A" * 4096).hexdigest()
    received = work_domain.export_tree(out, ["a.bin", "b.bin"], env.root / "intake")
    assert {row["path"]: row["sha256"] for row in received} == {"a.bin": digest, "b.bin": digest}
    assert (env.root / "intake" / "b.bin").read_bytes() == b"A" * 4096
    for refused in ("link", "pipe"):
        with pytest.raises(work_domain.WorkDomainRefused):
            work_domain.export_tree(out, [refused], env.root / f"refused-{refused}")
    retained = env.work / "app" / "retained" / "a.txt"
    retained.parent.mkdir()
    retained.write_text("RETAINED\n", encoding="utf-8")
    (out / "a.txt").write_text("SOURCE REL\n", encoding="utf-8")
    registry = _registry(env, env.work / "app")
    records = copy_directory_to_task_artifacts(registry._ctx, out, member_sources=[("a.txt", retained)])
    with zipfile.ZipFile(records[1]["path"]) as packaged:
        assert packaged.namelist() == ["a.txt"] and packaged.read("a.txt") == b"RETAINED\n"


def test_patches_and_file_results_stream_to_the_work_domain(env, monkeypatch):
    """``git apply`` of a service patch and a prepared file result reach the work as open files on stdin: neither is
    read whole into the service's memory."""
    from ouroboros.tools.patch_staging import git_apply
    from ouroboros.workspace_file_outputs import prepare_file_outputs

    _managed(env, monkeypatch)
    repo = _work_repo(env.work / "app", {"a.txt": "one\n"})
    service = env.root / "service"
    service.mkdir()
    patch = service / "change.patch"
    patch.write_text("diff --git a/new.txt b/new.txt\nnew file mode 100644\n--- /dev/null\n+++ b/new.txt\n"
                     "@@ -0,0 +1 @@\n+hello\n", encoding="utf-8")
    payload = bytes(range(256)) * 1024
    archive = service / "outputs.zip"
    with zipfile.ZipFile(archive, "w") as zipped:
        zipped.writestr("big.bin", payload)
    after = {"kind": "file", "size": len(payload), "sha256": hashlib.sha256(payload).hexdigest(), "mode": 0o644}
    rows = [{"path": "big.bin", "before": None, "after": after, "archive_path": str(archive), "member": "big.bin"}]
    whole = pathlib.Path.read_bytes

    def guarded(self):  # the patch, and the prepared result in its temporary folder, are never read whole
        if str(self) == str(patch) or "ouroboros-file-results-" in str(self):
            raise AssertionError(f"read whole into the service: {self}")
        return whole(self)

    monkeypatch.setattr(pathlib.Path, "read_bytes", guarded)
    applied = git_apply(repo, patch, [])
    with prepare_file_outputs(rows, repo) as prepared:
        prepared.apply()
    monkeypatch.setattr(pathlib.Path, "read_bytes", whole)
    assert applied.returncode == 0 and (repo / "new.txt").read_text() == "hello\n", applied.stderr
    assert (repo / "big.bin").read_bytes() == payload


def test_an_unconfirmed_work_domain_end_reaches_its_consumer_and_keeps_its_record(env, monkeypatch):
    """When the container does not confirm a process group ended, a write and a Git read report it
    (``WorkDomainUnsettled``, never ``process_not_started``) with what arrived, and the executor records stay."""
    from ouroboros import execution_domain

    docker = _managed(env, monkeypatch)
    repo = _work_repo(env.work / "app", {"a.txt": "one\n"})
    monkeypatch.setattr(execution_domain.AttachedLaunch, "settle", lambda self: False)
    with pytest.raises(work_domain.WorkDomainUnsettled) as wrote:
        work_domain.write_text(repo / "b.txt", "two\n")
    assert (repo / "b.txt").read_text() == "two\n" and wrote.value.process_not_started is False
    with pytest.raises(work_domain.WorkDomainUnsettled) as ran:
        work_domain.run(["git", "status", "--porcelain"], cwd=str(repo), capture_output=True, text=True)
    assert "b.txt" in ran.value.result.stdout
    assert docker.records()  # kept for Panic and shutdown to settle


def test_a_foreground_command_keeps_custody_of_its_group_until_every_member_ended(env, monkeypatch):
    """``run_command`` in the pinned container: its leader exits while a member it started still runs. Completion
    waits for the bounded group settlement, so the member never runs after custody ends; when the group cannot
    be confirmed ended, the outcome stays unknown and the record stays."""
    from ouroboros import workspace_executor

    _managed(env, monkeypatch)
    registry, later = _registry(env, env.work), env.work / "helper-ran-after-return"
    helper = f"(sleep 3; touch {later}) </dev/null >/dev/null 2>&1 & exit 0"
    text = registry.execute("run_command", {"cmd": ["sh", "-c", helper]})
    assert "⚠️" not in text, text
    assert not _records(env)  # retired only after the whole group ended
    time.sleep(3.5)
    assert not later.exists()  # stopped at settlement, never left running outside custody
    observer = workspace_executor._docker_proc_observer_function
    monkeypatch.setattr(workspace_executor, "_docker_proc_observer_function",
                        lambda: observer() + "; _ouroboros_proc_observe() { printf running; }")
    registry.execute("run_command", {"cmd": ["sh", "-c", "(sleep 2) </dev/null >/dev/null 2>&1 & exit 0"]})
    assert _records(env)  # an unconfirmed group keeps its record for settlement


def test_staged_inputs_are_held_by_their_task_and_retired_with_it(env, monkeypatch):
    """A native run's staged inputs sit in a session scratch its task's drive owns: the custody release retires the
    folder with the task, and a producer's retention marker keeps it as a visible hold instead."""
    from ouroboros import delegate_readonly_inputs, session_scratch
    from ouroboros.test_environment import RETENTION_MARKER

    staging = env.work / "staging"
    _managed(env, monkeypatch, staging_root=str(staging))
    repo = _work_repo(env.work / "app", {"input.txt": "selected input\n", "kept.txt": "kept input\n"})
    registry = _registry(env, repo)
    drive = env.root / "drive" / "task_drives" / "exec-task"
    folders = []
    for invocation, name in (("inv-1", "input.txt"), ("inv-2", "kept.txt")):
        inputs = delegate_readonly_inputs.prepare_readonly_inputs(
            registry._ctx, invocation, [{"root": "active_workspace", "path": name}])
        folder = pathlib.Path(inputs.selected[0]["local"]).parents[3]
        record = json.loads((drive / "session_scratch_owners" / f"inputs-{invocation}" / "scratch.json").read_text())
        assert folder.parent == staging / "scratch" and record["path"] == str(folder), (folder, record)
        folders.append(folder)
    (folders[1] / RETENTION_MARKER).write_text("the producer kept these inputs\n", encoding="utf-8")
    retained = session_scratch.release_session_scratch(drive)
    assert not folders[0].exists() and folders[1].exists()
    assert [row["path"] for row in retained] == [str(folders[1])] and "kept these inputs" in retained[0]["example"]


@pytest.mark.parametrize("administered", [False, True])
@pytest.mark.parametrize("edit_executable", [False, True])
def test_a_native_payload_run_drafts_in_its_work_copy_and_returns_for_the_services_apply(
        env, monkeypatch, administered, edit_executable):
    """The actual start builds the snapshot in W, where the session drafts. Capture returns data; applying to an
    administered live package refuses without losing the draft. An unpinned payload still applies normally."""
    from ouroboros import delegate_custody as custody, execution_domain
    from ouroboros.tools.delegate_integration import payload_content_hash
    from ouroboros.subagent_worktrees import find_execution_snapshot
    from ouroboros.tools.delegate import _capture_terminal_patch
    from ouroboros.tools.subagent_integration import _integrate_delegated_patch
    from tests.test_delegated_skill_payload import _payload_ctx, _seed_skill, _exact_payload_start, _StartStub
    from tests.test_execution_domain import section, work_ref, STORE

    staging = env.work / "staging"
    _managed(env, monkeypatch, staging_root=str(staging))
    ctx = _payload_ctx(env.root, monkeypatch)
    skill = _seed_skill(env.root / "data", name="alpha")
    (skill / "plugin.py").chmod(0o755)
    if administered:
        pin = {"name": "alpha", "path": str(skill), "content_hash": payload_content_hash(skill)}
        env.manage({**section(env), "staging_root": str(staging)},
                   extensions={"when_unmet": "warn", "required": [pin], "trusted": [pin]})
    seen = {}
    class Engine(_StartStub):
        engine_ref, store_identity = work_ref(env), STORE
    monkeypatch.setattr(execution_domain, "native_gateway", lambda: Engine(seen))
    custody._CUSTODY.clear()
    start = json.loads(_exact_payload_start(ctx, "draft a change", root="skill_payload", bucket="external", skill_name="alpha"))
    assert start.get("run_id") == "run-p1", start
    entry = custody._CUSTODY["run-p1"]
    copy = pathlib.Path(find_execution_snapshot(entry.snapshot_id)["path"])
    assert copy.parent == staging / "copies" and (copy / "plugin.py").read_text() == "VALUE = 1\n"
    assert stat.S_IMODE((copy / "plugin.py").stat().st_mode) == 0o755
    assert str(copy) == seen["request"]["scope"]["root"] and entry.engine_ref == work_ref(env)
    assert execution_domain.request_paths_refusal(seen["request"]) == ""
    if edit_executable:
        (copy / "plugin.py").write_text("VALUE = 2\n", encoding="utf-8")
    (copy / "extra.txt").write_text("new\n", encoding="utf-8")
    from ouroboros.delegate_collection import observe_run

    observe_run(ctx.drive_root, entry, {"runId": entry.run_id, "summary": {"state": "succeeded"}})
    with service_effects_on(env.work, allowed_opens=(str(env.work),)) as capturing:
        capture = _capture_terminal_patch(ctx, entry)
    assert capturing == [] and capture["status"] == "ready_with_changes", (capturing, capture)
    manifest = json.loads(pathlib.Path(capture["manifest_artifact"]).read_text(encoding="utf-8"))
    assert set(manifest["tracked_changed"]) == ({"plugin.py", "extra.txt"} if edit_executable else {"extra.txt"}), manifest
    out = _integrate_delegated_patch(ctx, entry.run_id, "apply", "drafted in the work domain")
    if administered:
        assert "MANAGED_BY_DEPLOYMENT" in out and (skill / "plugin.py").read_text() == "VALUE = 1\n", out
        assert copy.exists() and find_execution_snapshot(entry.snapshot_id) is not None
        assert "Rejected" in _integrate_delegated_patch(ctx, entry.run_id, "reject", "keep the captured draft, do not install")
    else:
        assert "✅ Integrated" in out, out
        assert (skill / "plugin.py").read_text() == ("VALUE = 2\n" if edit_executable else "VALUE = 1\n")
        assert (skill / "extra.txt").read_text() == "new\n"
    assert find_execution_snapshot(entry.snapshot_id) is None and not copy.exists()
    assert stat.S_IMODE((skill / "plugin.py").stat().st_mode) == 0o755
    custody._CUSTODY.clear()


def test_light_mode_process_tripwires_observe_git_and_metadata_in_the_work_domain(env, monkeypatch):
    """An allowed light-mode command exercises both snapshots in the real registry. Git configuration remains
    work code; the service never starts Git or follows work pathnames, and the tripwire still notices changes."""
    from ouroboros.tools import registry_guard_process as guard
    from ouroboros.tools.registry import ToolContext, ToolRegistry

    _managed(env, monkeypatch)
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "light")
    repo = _work_repo(env.work / "body-copy", {"README.md": "work\n"})
    (repo / "dirty.txt").write_text("before\n", encoding="utf-8")
    registry = ToolRegistry(repo_dir=repo, drive_root=env.root / "drive")
    registry.set_context(ToolContext(repo_dir=repo, drive_root=env.root / "drive", task_id="light-work"))
    snapshots, original = [], guard._light_repo_snapshot
    def observed(path):
        result = original(path)
        snapshots.append(result)
        return result
    monkeypatch.setattr(guard, "_light_repo_snapshot", observed)
    with service_effects_on(env.work, allowed_opens=(str(env.work),)) as seen:
        answer = registry.execute("run_command", {"cmd": ["printf", "useful work"]})
        refs = guard._git_ref_snapshot(repo)
        before = original(repo)
        work_domain.write_text(repo / "dirty.txt", "changed contents\n")
        after = original(repo)
    assert seen == [], seen
    assert "useful work" in answer and len(snapshots) == 2 and all(snapshots), (answer, snapshots)
    assert refs and refs["head"] and before["digest"] != after["digest"], (refs, before, after)


def test_work_command_scratch_and_non_git_observations_stay_in_the_work_domain(env, monkeypatch):
    """Declared scratch excludes only untracked files; non-Git effect observations never read through a service
    pathname, including a link whose target is visible only to the service."""
    from ouroboros.tools import shell_effects

    _managed(env, monkeypatch)
    repo = _work_repo(env.work / "app", {"tracked.txt": "kept\n"})
    registry = _registry(env, repo)
    loose = env.work / "non-git"
    loose.mkdir()
    (loose / "plain").write_text("before\n", encoding="utf-8")
    (loose / "link").symlink_to(env.root / "service-only")
    with service_effects_on(env.work, allowed_opens=(str(env.work),)) as seen:
        ran = registry.execute("run_command", {"cmd": ["sh", "-c", "printf scratch > scratch.txt"],
                                                "scratch": ["scratch.txt"]})
        blocked = registry.execute("run_command", {"cmd": ["true"], "scratch": ["tracked.txt"]})
        paths = shell_effects._resolve_scratch_abs(["scratch.txt"], repo)
        adopted = shell_effects._scratch_safety_reason(registry._ctx, paths, repo, repo)
        before = shell_effects._shallow_listing(loose)
        fingerprint = shell_effects._tree_fingerprint(loose)
        work_domain.write_text(loose / "plain", "after with more bytes\n")
        after = shell_effects._shallow_listing(loose)
        changed = shell_effects._tree_fingerprint(loose)
    assert not seen, seen
    assert "exit_code=0" in ran and "git-tracked" in blocked and adopted == "", (ran, blocked, adopted)
    assert before != after and fingerprint != changed


def _wheel(folder: pathlib.Path, name: str = "ouroprobe") -> pathlib.Path:
    """A minimal pure-Python wheel: the work-side package index of an offline test."""
    dist = f"{name}-0.1.dist-info"
    files = {f"{name}/__init__.py": b"VALUE = 'from-the-work-venv'\n",
             f"{dist}/METADATA": f"Metadata-Version: 2.1\nName: {name}\nVersion: 0.1\n".encode(),
             f"{dist}/WHEEL": b"Wheel-Version: 1.0\nGenerator: test\nRoot-Is-Purelib: true\nTag: py3-none-any\n"}
    record = "".join(f"{path},sha256={base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b'=').decode()},"
                     f"{len(data)}\n" for path, data in files.items()) + f"{dist}/RECORD,,\n"
    files[f"{dist}/RECORD"] = record.encode()
    wheel = folder / f"{name}-0.1-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w") as zipped:
        for path, data in files.items():
            zipped.writestr(path, data)
    return wheel


def _installed_skill(env, monkeypatch, root: pathlib.Path, name: str, script: str, dependencies=()):
    """A reviewed, enabled script skill in an installed-skills folder OUTSIDE the work mapping."""
    from ouroboros.skill_loader import SkillReviewState, compute_content_hash, save_enabled, save_review_state
    from ouroboros.tools.registry import ToolContext

    skill, drive = root / name, env.root / "drive"
    (skill / "scripts").mkdir(parents=True)
    declared = "dependencies:\n" + "".join(f"  - {item}\n" for item in dependencies) if dependencies else ""
    (skill / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: Probe.\nversion: 0.1.0\ntype: script\nruntime: python3\ntimeout_sec: 60\n"
        f"{declared}scripts:\n  - name: run.py\n    description: Probe.\n---\n# body\n", encoding="utf-8")
    (skill / "scripts" / "run.py").write_text(script, encoding="utf-8")
    (env.root / "repo").mkdir(exist_ok=True)
    drive.mkdir(exist_ok=True)
    save_enabled(drive, name, True)
    save_review_state(drive, name, SkillReviewState(status="pass", content_hash=compute_content_hash(skill)))
    monkeypatch.setenv("OUROBOROS_SKILLS_REPO_PATH", str(root))
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "advanced")
    return ToolContext(repo_dir=env.root / "repo", drive_root=drive), drive


def test_an_installed_script_skill_runs_from_its_verified_work_copy_with_dependencies_built_there(
        env, monkeypatch, tmp_path):
    """An installed skill under the service's data (outside every mapping), managed: the one dependency owner
    copies its reviewed payload into the work domain and builds its packages there (venv and pip in the
    container, from a work-side wheelhouse); ``skill_exec`` runs the script from that copy with that environment.
    A copy the work domain changed — an edited script, a planted file — is replaced before the next run."""
    from ouroboros import skill_dependencies, work_skill_payloads
    from ouroboros.marketplace import isolated_deps
    from ouroboros.skill_loader import find_skill
    from ouroboros.tools import skill_exec

    script = "import os, sys, ouroprobe\nprint(ouroprobe.VALUE, os.environ.get('IN_FAKE_CONTAINER'), sys.prefix)\n"
    installed = tmp_path / "installed-skills"
    ctx, drive = _installed_skill(env, monkeypatch, installed, "probe", script, dependencies=["ouroprobe"])
    staging, state = env.work / "staging", env.work / "skill-state"
    docker = _managed(env, monkeypatch, staging_root=str(staging), skill_state_root=str(state))
    wheels = env.work / "wheelhouse"
    wheels.mkdir()
    _wheel(wheels)
    monkeypatch.setattr(isolated_deps, "_SAFE_ENV_KEYS", {*isolated_deps._SAFE_ENV_KEYS, "PIP_FIND_LINKS", "PIP_NO_INDEX"})
    monkeypatch.setenv("PIP_FIND_LINKS", str(wheels))
    monkeypatch.setenv("PIP_NO_INDEX", "1")
    monkeypatch.setattr(work_skill_payloads, "CONTAINER_PYTHON", sys.executable)  # the fake container's interpreter
    loaded = find_skill(drive, "probe")
    status, detail = skill_dependencies.ensure_skill_dependencies(drive, loaded)
    home = staging / "skills" / "probe" / loaded.content_hash
    assert status == "installed", detail
    assert (home / ".ouroboros_env" / "python").is_dir() and not (installed / "probe" / ".ouroboros_env").exists()
    first = json.loads(skill_exec._handle_skill_exec(ctx, skill="probe", script="scripts/run.py", args=[]))
    assert first["exit_code"] == 0, first
    assert first["stdout"].split() == ["from-the-work-venv", "1", str(home / ".ouroboros_env" / "python")]
    (home / "scripts" / "run.py").write_text("print('TAMPERED')\n", encoding="utf-8")
    (home / "scripts" / "planted.py").write_text("PLANTED = 1\n", encoding="utf-8")
    again = json.loads(skill_exec._handle_skill_exec(ctx, skill="probe", script="scripts/run.py", args=[]))
    assert again["stdout"].split()[0] == "from-the-work-venv" and not (home / "scripts" / "planted.py").exists()
    assert (installed / "probe" / "scripts" / "run.py").read_text() == script  # the installed payload: untouched
    assert not docker.records()
