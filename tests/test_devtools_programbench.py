"""ProgramBench: the cleanroom, the seeded workspace and the submission it ships.

Split verbatim out of ``tests/test_devtools_benchmarks.py`` by theme. This module owns the
task body and its protected-path policy, the cleanroom preflight and container lifecycle,
the idempotent workspace seeding, the checkpoint the poller resumes from, and the blocker
sidecars every failure path leaves behind.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

from devtools.benchmarks.programbench.programbench_adapter import (
    build_instruction,
    build_ouroboros_task_body,
    classify_infra_failure,
    cleanroom_image_ref,
    container_name_for_instance,
    create_submission_tarball,
    prepare_seeded_workspace,
    preflight_cleanroom_container,
    seed_workspace_from_image,
    start_cleanroom_container,
    submit_and_wait,
    terminal_task_status,
    verify_reference_executable_runnable,
)

from tests._devtools_benchmarks_shared import _git_repo
from tests._devtools_benchmarks_shared import _isolate_bench_runs_root as __isolate_bench_runs_root

# Fixtures are requested by name as test parameters, so they are re-bound through a
# module attribute: a direct import of a name that reappears as a parameter is an F811
# redefinition under the CI ruff gate.
_isolate_bench_runs_root = __isolate_bench_runs_root


def test_programbench_task_body_sets_executor_and_protected_policy(tmp_path):
    workspace = tmp_path / "workspace"
    _git_repo(workspace)

    body = build_ouroboros_task_body(
        instruction="solve",
        workspace_host_path=workspace,
        container_name="pb-cleanroom",
        protected_backend_paths=["/workspace/reference_executable"],
    )

    assert body["allowed_resources"] == {"web": False, "network": False, "internet": False}
    assert body["actor_id"] == "programbench"
    assert body["source"] == "programbench"
    assert "actor_id" not in body["metadata"]
    assert body["executor_ref"]["type"] == "docker_exec"
    assert body["executor_ref"]["network"] == "none"
    protected = body["resource_policy"]["protected_artifacts"][0]
    assert protected["role"] == "black_box_reference"
    assert protected["allow"] == ["execute"]
    assert {"read_bytes", "hash", "static_introspection", "dynamic_trace", "debug"} <= set(protected["deny"])
    # House rule: benches measure the single-model Ouroboros harness.
    assert body["disabled_tools"] == ["claude_code_edit", "schedule_subagent"]
    # POST /api/tasks accepts no top-level task_contract field; the pacing block
    # rides in metadata.budget_profile and must already be in the normalized
    # contract shape so build_task_contract() adopts it verbatim.
    assert "task_contract" not in body
    profile = body["metadata"]["budget_profile"]
    assert profile == {
        "cost_hard_stop_pct": 0,
        "improvement_policy": "until_deadline",
        "max_improvement_passes": 6,
        "reserve_finalization_pct": 15,
        "stall_rounds_threshold": 12,
    }
    # Advisory acceptance claims ride the body top-level (gateway-normalized);
    # the wording stays task-general (no benchmark-specific oracle taxonomy).
    claims = body["acceptance_claims"]
    assert len(claims) == 1 and claims[0]["id"] == "behavioral_equivalence"
    assert claims[0]["priority"] == "must"
    from ouroboros.contracts.task_contract import build_task_contract, normalize_budget_profile

    assert normalize_budget_profile(profile) == profile
    assert build_task_contract(body)["budget_profile"] == profile

def test_programbench_git_workspace_does_not_commit_protected_reference(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "reference_executable").write_text("protected-bytes\n", encoding="utf-8")

    build_ouroboros_task_body(
        instruction="solve",
        workspace_host_path=workspace,
        container_name="pb-cleanroom",
        protected_backend_paths=["/workspace/reference_executable"],
    )

    head = subprocess.run(["git", "rev-parse", "--verify", "HEAD"], cwd=workspace, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    show = subprocess.run(["git", "show", "HEAD:reference_executable"], cwd=workspace, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    assert head.returncode != 0
    assert show.returncode != 0

def test_programbench_submission_tarball_excludes_repo_noise(tmp_path):
    workspace = tmp_path / "workspace"
    (workspace / ".git").mkdir(parents=True)
    (workspace / ".git" / "HEAD").write_text("ref\n", encoding="utf-8")
    (workspace / ".ouroboros").mkdir()
    (workspace / ".ouroboros" / "trace.json").write_text("{}\n", encoding="utf-8")
    (workspace / "node_modules" / "pkg").mkdir(parents=True)
    (workspace / "node_modules" / "pkg" / "index.js").write_text("junk\n", encoding="utf-8")
    (workspace / "build").mkdir()
    (workspace / "build" / "out.o").write_text("junk\n", encoding="utf-8")
    (workspace / "dist").mkdir()
    (workspace / "dist" / "bundle.js").write_text("junk\n", encoding="utf-8")
    (workspace / "reference_executable").write_text("protected\n", encoding="utf-8")
    (workspace / "solution.py").write_text("print('ok')\n", encoding="utf-8")

    tar_path = create_submission_tarball(
        workspace,
        tmp_path / "submission.tar.gz",
        protected_paths=["/workspace/reference_executable", "reference_executable"],
    )

    with tarfile.open(tar_path, "r:gz") as tar:
        names = set(tar.getnames())
    assert "solution.py" in names
    assert ".git/HEAD" not in names
    assert ".ouroboros/trace.json" not in names
    assert "node_modules/pkg/index.js" not in names
    assert "build/out.o" not in names
    assert "dist/bundle.js" not in names
    assert "reference_executable" not in names

def test_programbench_submission_excludes_both_root_binaries(tmp_path):
    """Source-submission contract: neither the agent-built ./executable nor the
    reference binary may enter submission.tar.gz — the official eval rebuilds
    via compile.sh, and a shipped binary would mask compile failures. Nested
    files that merely SHARE the name stay in (they are ordinary source tree
    content)."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "executable").write_bytes(b"\x7fELF-agent-built")
    (workspace / "reference_executable").write_bytes(b"\x7fELF-reference")
    (workspace / "compile.sh").write_text("#!/bin/sh\ncc -o executable main.c\n", encoding="utf-8")
    (workspace / "main.c").write_text("int main(void){return 0;}\n", encoding="utf-8")
    (workspace / "tools").mkdir()
    (workspace / "tools" / "executable").write_text("just a source file\n", encoding="utf-8")

    tar_path = create_submission_tarball(workspace, tmp_path / "submission.tar.gz")

    with tarfile.open(tar_path, "r:gz") as tar:
        names = set(tar.getnames())
    assert "compile.sh" in names
    assert "main.c" in names
    assert "tools/executable" in names
    assert "executable" not in names
    assert "reference_executable" not in names

def test_programbench_instance_path_stays_under_run_root(tmp_path):
    from devtools.benchmarks.common.run_roots import safe_join_under

    root = tmp_path / "programbench-run"
    assert safe_join_under(root, "cheat/cheat") == root.resolve(strict=False) / "cheat" / "cheat"
    with pytest.raises(ValueError, match="escapes run root"):
        safe_join_under(root, "../escape")
    with pytest.raises(ValueError, match="escapes run root"):
        safe_join_under(root, "/tmp/escape")

def test_programbench_cleanroom_preflight_requires_task_cleanroom_and_no_network(monkeypatch):
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        return subprocess.CompletedProcess(
            cmd,
            0,
            stdout=json.dumps([
                {
                    "Config": {"Image": "ghcr.io/facebookresearch/programbench/foo:task_cleanroom"},
                    "HostConfig": {"NetworkMode": "none"},
                }
            ]),
            stderr="",
        )

    import devtools.benchmarks.programbench.programbench_adapter as adapter

    monkeypatch.setattr(adapter.subprocess, "run", fake_run)
    assert preflight_cleanroom_container("pb") == {
        "image": "ghcr.io/facebookresearch/programbench/foo:task_cleanroom",
        "network": "none",
    }
    assert calls[0][:2] == ["docker", "inspect"]

def test_programbench_preflight_failure_writes_blocker_sidecars(tmp_path, monkeypatch):
    import devtools.benchmarks.programbench.run_programbench as run_programbench

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    instruction = tmp_path / "instruction.txt"
    instruction.write_text("solve", encoding="utf-8")
    output = tmp_path / "programbench-ledger.jsonl"
    manifest = tmp_path / "programbench-manifest.json"
    monkeypatch.setattr(
        run_programbench,
        "preflight_cleanroom_container",
        lambda _: (_ for _ in ()).throw(RuntimeError("docker missing")),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_programbench.py",
            "--allow-dirty-seed",
            "--workspace",
            str(workspace),
            "--instruction-file",
            str(instruction),
            "--container-name",
            "missing",
            "--instance-id",
            "case1",
            "--ledger-output",
            str(output),
            "--manifest-output",
            str(manifest),
        ],
    )

    with pytest.raises(RuntimeError, match="docker missing"):
        run_programbench.main()
    row = json.loads(output.read_text(encoding="utf-8").splitlines()[0])
    manifest_json = json.loads(manifest.read_text(encoding="utf-8"))
    assert row["status"] == "blocked"
    assert row["reason_code"] == "cleanroom_preflight_failed"
    assert manifest_json["requested_task_ids"] == ["case1"]

def test_programbench_prepare_seeded_workspace_is_idempotent_on_solved_tree(tmp_path):
    """Re-running prepare on an ALREADY-normalized workspace (reference present,
    agent-built ./executable beside it after a solve) must preserve the real
    reference and leave the agent's build product alone — never rename the
    agent binary over the protected reference."""
    from devtools.benchmarks.programbench.programbench_adapter import prepare_seeded_workspace

    root = tmp_path / "ws"
    root.mkdir()
    (root / "reference_executable").write_bytes(b"REAL-REFERENCE")
    (root / "executable").write_bytes(b"AGENT-BUILD")
    layout = prepare_seeded_workspace(root)
    assert (root / "reference_executable").read_bytes() == b"REAL-REFERENCE"
    assert (root / "executable").read_bytes() == b"AGENT-BUILD"
    assert layout["reference_host_path"] == str(root / "reference_executable")

def test_programbench_prepare_only_normalizes_raw_workspace(tmp_path, monkeypatch):
    """run_programbench (prepare-only) must run prepare_seeded_workspace before
    body/submission creation: a raw cleanroom workspace has the REAL reference
    at ./executable — unrenamed it would ship in the tarball while the task
    body points agents at a nonexistent ./reference_executable."""
    import devtools.benchmarks.programbench.run_programbench as run_programbench

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "executable").write_bytes(b"\x7fELF-raw-seeded-reference")
    (workspace / "main.c").write_text("int main(void){return 0;}\n", encoding="utf-8")
    instruction = tmp_path / "instruction.txt"
    instruction.write_text("solve", encoding="utf-8")
    output = tmp_path / "ledger.jsonl"
    manifest = tmp_path / "manifest.json"
    monkeypatch.setattr(run_programbench, "preflight_cleanroom_container",
                        lambda _: {"image": "task_cleanroom", "network": "none"})
    monkeypatch.setattr(sys, "argv", [
        "run_programbench.py", "--allow-dirty-seed", "--workspace", str(workspace),
        "--instruction-file", str(instruction), "--container-name", "pb",
        "--instance-id", "case-prep", "--ledger-output", str(output),
        "--manifest-output", str(manifest),
    ])
    run_programbench.main()

    assert (workspace / "reference_executable").is_file()
    assert not (workspace / "executable").exists()
    with tarfile.open(next(tmp_path.rglob("submission.tar.gz")), "r:gz") as tar:
        names = set(tar.getnames())
    assert "main.c" in names
    assert "reference_executable" not in names
    assert "executable" not in names

def test_programbench_submission_failure_writes_sidecars(tmp_path, monkeypatch):
    import devtools.benchmarks.programbench.run_programbench as run_programbench

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "executable").write_bytes(b"\x7fELF-seeded-reference")
    instruction = tmp_path / "instruction.txt"
    instruction.write_text("solve", encoding="utf-8")
    output = tmp_path / "programbench-ledger.jsonl"
    manifest = tmp_path / "programbench-manifest.json"
    monkeypatch.setattr(run_programbench, "preflight_cleanroom_container", lambda _: {"image": "task_cleanroom", "network": "none"})
    monkeypatch.setattr(
        run_programbench,
        "create_submission_tarball",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("tar failed")),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_programbench.py",
            "--allow-dirty-seed",
            "--workspace",
            str(workspace),
            "--instruction-file",
            str(instruction),
            "--container-name",
            "pb",
            "--instance-id",
            "case2",
            "--ledger-output",
            str(output),
            "--manifest-output",
            str(manifest),
        ],
    )

    with pytest.raises(RuntimeError, match="tar failed"):
        run_programbench.main()
    row = json.loads(output.read_text(encoding="utf-8").splitlines()[0])
    manifest_json = json.loads(manifest.read_text(encoding="utf-8"))
    assert row["status"] == "failed"
    assert row["reason_code"] == "submission_failed"
    assert row["official_eval_status"] == "not_run"
    assert manifest_json["requested_task_ids"] == ["case2"]
    assert manifest_json["extra"]["failure_reason_code"] == "submission_failed"

def test_programbench_official_eval_failure_writes_sidecars(tmp_path, monkeypatch):
    import devtools.benchmarks.programbench.run_programbench as run_programbench

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "executable").write_bytes(b"\x7fELF-seeded-reference")
    instruction = tmp_path / "instruction.txt"
    instruction.write_text("solve", encoding="utf-8")
    output = tmp_path / "programbench-ledger.jsonl"
    manifest = tmp_path / "programbench-manifest.json"
    submission = tmp_path / "submission.tar.gz"
    monkeypatch.setattr(run_programbench, "preflight_cleanroom_container", lambda _: {"image": "task_cleanroom", "network": "none"})
    monkeypatch.setattr(run_programbench, "create_submission_tarball", lambda *_args, **_kwargs: submission)
    monkeypatch.setattr(
        run_programbench,
        "run_official_eval",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("eval failed")),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_programbench.py",
            "--allow-dirty-seed",
            "--workspace",
            str(workspace),
            "--instruction-file",
            str(instruction),
            "--container-name",
            "pb",
            "--instance-id",
            "case3",
            "--ledger-output",
            str(output),
            "--manifest-output",
            str(manifest),
            "--eval",
        ],
    )

    with pytest.raises(RuntimeError, match="eval failed"):
        run_programbench.main()
    row = json.loads(output.read_text(encoding="utf-8").splitlines()[0])
    manifest_json = json.loads(manifest.read_text(encoding="utf-8"))
    assert row["status"] == "failed"
    assert row["reason_code"] == "official_eval_failed"
    assert row["official_eval_status"] == "failed"
    assert manifest_json["requested_task_ids"] == ["case3"]
    assert manifest_json["extra"]["failure_reason_code"] == "official_eval_failed"

def test_programbench_client_poll_error_keeps_container_when_task_live(tmp_path, monkeypatch):
    """A client-side poll failure (timeout OR any transient mid-poll error) after a
    task was submitted must NOT tear down the cleanroom container — the checkpoint
    holds a live task_id and the next run reattaches to it. A failure with NO
    submitted task (creation itself failed) falls to the normal teardown path."""
    import json as _json

    from devtools.benchmarks.programbench import run_programbench_e2e as e2e

    stopped: list[str] = []
    monkeypatch.setattr(e2e, "pull_cleanroom_image", lambda name: {"image": name})
    monkeypatch.setattr(e2e, "seed_workspace_from_image", lambda name, ws: {"seeded": True})
    monkeypatch.setattr(e2e, "start_cleanroom_container",
                        lambda *a, **k: {"preflight": {"ok": True}})
    monkeypatch.setattr(e2e, "stop_cleanroom_container", lambda name: stopped.append(name))
    monkeypatch.setattr(e2e, "build_ouroboros_task_body",
                        lambda **k: {"description": "x", "metadata": {}})

    cfg = e2e.InstanceRunConfig(
        out_root=tmp_path, ouroboros_url="http://127.0.0.1:1", timeout_sec=1.0,
        cpus="1", memory="1g", protected_paths=[], dry_run=False,
        skip_pull=False, redo_existing=False,
    )

    def _fake_submit(reason_exc):
        # Mirror the real submit_and_wait: it writes the checkpoint with a task_id
        # (task submitted) BEFORE polling, then raises on the poll failure.
        def _inner(base_url, body, *, timeout_sec, checkpoint_path):
            Path(checkpoint_path).write_text(
                _json.dumps({"task_id": "tsk-live", "status": "running"}), encoding="utf-8")
            raise reason_exc
        return _inner

    # (a) timeout after submit -> kept alive, timeout reason code
    monkeypatch.setattr(e2e, "submit_and_wait", _fake_submit(TimeoutError("did not finish")))
    row = e2e._process_instance({"instance_id": "inst-a", "image_name": "img-a"}, cfg)
    assert row["status"] == "failed"
    assert row["reason_code"] == "client_poll_timeout_reattachable"
    assert row["details"]["container_left_running"] is True
    assert stopped == []

    # (b) transient NON-timeout error after submit -> ALSO kept alive (r1 #10)
    monkeypatch.setattr(e2e, "submit_and_wait", _fake_submit(RuntimeError("transient 502")))
    row2 = e2e._process_instance({"instance_id": "inst-b", "image_name": "img-b"}, cfg)
    assert row2["status"] == "failed"
    assert row2["reason_code"] == "client_poll_error_reattachable"
    assert stopped == []  # a live task's container must survive a transient poll error

    # (c) failure with NO submitted task (checkpoint never written) -> teardown
    def _creation_failed(*a, **k):
        raise RuntimeError("task creation returned no id")

    monkeypatch.setattr(e2e, "submit_and_wait", _creation_failed)
    row3 = e2e._process_instance({"instance_id": "inst-c", "image_name": "img-c"}, cfg)
    assert row3["status"] == "failed"
    assert row3["reason_code"] == "RuntimeError"
    assert stopped == [e2e.container_name_for_instance("inst-c")]

def test_programbench_resume_skipped_rows_are_successful():
    """A resume-only run (everything already has submission.tar.gz) must exit 0:
    skipped rows are successful prior work for exit-code/failed_count purposes."""
    from devtools.benchmarks.programbench import run_programbench_e2e as e2e

    assert e2e._row_successful({"status": "completed"})
    assert e2e._row_successful({"status": "skipped"})
    assert not e2e._row_successful({"status": "failed"})
    assert not e2e._row_successful({})

def test_programbench_second_run_reattaches_without_cleanroom_reset(tmp_path, monkeypatch):
    """After a client_poll_timeout_reattachable row, the NEXT run must honor the
    live checkpoint: no image pull, no workspace reseed, no container restart
    (start would stop the namesake executor first) — straight to reattach."""
    import json as _json

    from devtools.benchmarks.programbench import run_programbench_e2e as e2e

    def _forbidden(*a, **k):
        raise AssertionError("fresh cleanroom work must not run on the reattach path")

    stopped: list[str] = []
    monkeypatch.setattr(e2e, "pull_cleanroom_image", _forbidden)
    monkeypatch.setattr(e2e, "seed_workspace_from_image", _forbidden)
    monkeypatch.setattr(e2e, "start_cleanroom_container", _forbidden)
    monkeypatch.setattr(e2e, "stop_cleanroom_container", lambda name: stopped.append(name))
    monkeypatch.setattr(e2e, "build_ouroboros_task_body",
                        lambda **k: {"description": "x", "metadata": {}})
    monkeypatch.setattr(e2e, "ouroboros_api_request",
                        lambda *a, **k: {"task_id": "tsk-9", "status": "running"})
    monkeypatch.setattr(e2e, "submit_and_wait",
                        lambda *a, **k: {"task_id": "tsk-9", "status": "completed"})
    monkeypatch.setattr(e2e, "create_submission_tarball",
                        lambda ws, dest, protected_paths: (dest.parent.mkdir(parents=True, exist_ok=True),
                                                           dest.write_bytes(b"x"), dest)[-1])

    cfg = e2e.InstanceRunConfig(
        out_root=tmp_path, ouroboros_url="http://127.0.0.1:1", timeout_sec=1.0,
        cpus="1", memory="1g", protected_paths=[], dry_run=False,
        skip_pull=False, redo_existing=False,
    )
    inst_dir = tmp_path / "inst-a"
    inst_dir.mkdir()
    (inst_dir / e2e.TASK_CHECKPOINT_BASENAME).write_text(
        _json.dumps({"task_id": "tsk-9", "status": "running"}), encoding="utf-8")

    row = e2e._process_instance({"instance_id": "inst-a", "image_name": "img-a"}, cfg)
    assert row["status"] == "completed"
    assert row["details"]["harness"]["reattached_task_id"] == "tsk-9"
    # settled result re-arms normal teardown
    assert stopped == [e2e.container_name_for_instance("inst-a")]

def test_programbench_settled_failed_checkpoint_retries_fresh(tmp_path, monkeypatch):
    """Adversarial review r2 #5: a checkpoint naming a task that already SETTLED
    as FAILED must NOT reattach (that replays the old failure as zero work) — the
    resume must drop the stale checkpoint and re-solve in a fresh cleanroom."""
    import json as _json

    from devtools.benchmarks.programbench import run_programbench_e2e as e2e

    fresh_work: list[str] = []
    monkeypatch.setattr(e2e, "pull_cleanroom_image", lambda img: fresh_work.append("pull") or "sha")
    monkeypatch.setattr(e2e, "seed_workspace_from_image", lambda img, ws: fresh_work.append("seed"))
    monkeypatch.setattr(e2e, "start_cleanroom_container",
                        lambda *a, **k: fresh_work.append("start") or {"container": "c"})
    monkeypatch.setattr(e2e, "stop_cleanroom_container", lambda name: None)
    monkeypatch.setattr(e2e, "build_ouroboros_task_body",
                        lambda **k: {"description": "x", "metadata": {}})
    # The reattach honor-check GET returns a SETTLED-FAILED payload.
    monkeypatch.setattr(e2e, "ouroboros_api_request",
                        lambda *a, **k: {"task_id": "tsk-old", "status": "failed"})
    monkeypatch.setattr(e2e, "submit_and_wait",
                        lambda *a, **k: {"task_id": "tsk-new", "status": "completed"})
    monkeypatch.setattr(e2e, "create_submission_tarball",
                        lambda ws, dest, protected_paths: (dest.parent.mkdir(parents=True, exist_ok=True),
                                                           dest.write_bytes(b"x"), dest)[-1])

    cfg = e2e.InstanceRunConfig(
        out_root=tmp_path, ouroboros_url="http://127.0.0.1:1", timeout_sec=1.0,
        cpus="1", memory="1g", protected_paths=[], dry_run=False,
        skip_pull=False, redo_existing=False,
    )
    inst_dir = tmp_path / "inst-f"
    inst_dir.mkdir()
    checkpoint = inst_dir / e2e.TASK_CHECKPOINT_BASENAME
    checkpoint.write_text(_json.dumps({"task_id": "tsk-old", "status": "running"}), encoding="utf-8")

    row = e2e._process_instance({"instance_id": "inst-f", "image_name": "img-f"}, cfg)
    assert row["details"]["harness"]["reattached_task_id"] == ""  # did NOT reattach
    assert fresh_work == ["pull", "seed", "start"]  # fresh cleanroom ran
    assert row["status"] == "completed"

def test_programbench_build_instruction_renders_instance_fields(tmp_path):
    template = tmp_path / "instruction.md"
    template.write_text("id={{instance_id}} repo={{repository}} lang={{language}} diff={{difficulty}}\n", encoding="utf-8")
    text = build_instruction(
        {
            "instance_id": "foo__bar.abc123",
            "repository": "foo/bar",
            "language": "c",
            "difficulty": "easy",
        },
        template_path=template,
    )
    assert "id=foo__bar.abc123" in text
    assert "repo=foo/bar" in text
    assert "lang=c" in text
    assert "diff=easy" in text

def test_programbench_cleanroom_image_ref_and_container_name():
    assert cleanroom_image_ref("programbench/foo") == "programbench/foo:task_cleanroom_v6"
    assert cleanroom_image_ref("programbench/foo:task_cleanroom_v6") == "programbench/foo:task_cleanroom_v6"
    assert container_name_for_instance("abishekvashok__cmatrix.5c082c6").startswith("ouroboros-pb-")

def test_programbench_seed_workspace_from_image(monkeypatch, tmp_path):
    import devtools.benchmarks.programbench.programbench_adapter as adapter

    workspace = tmp_path / "workspace"
    calls: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        calls.append(list(cmd))
        if cmd[:3] == ["docker", "create", "--platform"]:
            return subprocess.CompletedProcess(cmd, 0, stdout="seed-cid\n", stderr="")
        if cmd[:2] == ["docker", "cp"]:
            workspace.mkdir(parents=True, exist_ok=True)
            (workspace / "executable").write_text("bin\n", encoding="utf-8")
            (workspace / "README.md").write_text("docs\n", encoding="utf-8")
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(adapter.subprocess, "run", fake_run)
    result = seed_workspace_from_image("programbench/demo", workspace)
    assert result["seeded_from"] == "/workspace"
    assert (workspace / "reference_executable").is_file()
    assert not (workspace / "executable").exists()
    if sys.platform != "win32":  # execute bit is a POSIX concept (bench runs in Linux containers)
        assert (workspace / "reference_executable").stat().st_mode & 0o111
    assert "/reference_executable" in (workspace / ".gitignore").read_text(encoding="utf-8")
    assert calls[0][:4] == ["docker", "create", "--platform", "linux/amd64"]
    assert calls[1][:2] == ["docker", "cp"]
    assert ["docker", "rm", "-f", "seed-cid"] in calls

def test_programbench_start_cleanroom_container_invokes_docker_run(monkeypatch, tmp_path):
    import devtools.benchmarks.programbench.programbench_adapter as adapter

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    calls: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        calls.append(list(cmd))
        if cmd[:2] == ["docker", "run"]:
            return subprocess.CompletedProcess(cmd, 0, stdout="running-cid\n", stderr="")
        if cmd[:2] == ["docker", "inspect"]:
            return subprocess.CompletedProcess(
                cmd,
                0,
                stdout=json.dumps([{"Config": {"Image": "programbench/demo:task_cleanroom_v6"}, "HostConfig": {"NetworkMode": "none"}}]),
                stderr="",
            )
        if cmd[:3] == ["docker", "exec", "pb-demo"]:
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(adapter.subprocess, "run", fake_run)
    result = start_cleanroom_container("pb-demo", "programbench/demo", workspace, cpus="2", memory="8g")
    run_cmd = next(cmd for cmd in calls if cmd[:2] == ["docker", "run"])
    assert "--network" in run_cmd and "none" in run_cmd
    assert "-v" in run_cmd
    assert result["container_name"] == "pb-demo"
    assert result["preflight"]["network"] == "none"
    assert result["reference_probe"]["probe_returncode"] == 0

def test_programbench_prepare_seeded_workspace_moves_reference_and_sets_execute_bit(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "executable").write_bytes(b"\x7fELF")

    layout = prepare_seeded_workspace(workspace)

    assert layout["reference_backend_path"] == "/workspace/reference_executable"
    assert (workspace / "reference_executable").is_file()
    assert not (workspace / "executable").exists()
    if sys.platform != "win32":  # execute bit is a POSIX concept (bench runs in Linux containers)
        assert (workspace / "reference_executable").stat().st_mode & 0o111
        assert (workspace / "reference_executable").stat().st_mode & 0o400
    gitignore = (workspace / ".gitignore").read_text(encoding="utf-8")
    assert "/reference_executable" in gitignore
    assert "/executable" in gitignore

def test_programbench_verify_reference_executable_runnable(monkeypatch):
    import devtools.benchmarks.programbench.programbench_adapter as adapter

    calls: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        calls.append(list(cmd))
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(adapter.subprocess, "run", fake_run)
    result = verify_reference_executable_runnable("pb-demo")
    assert result["probe_returncode"] == 0
    assert calls[0][0] == "docker"
    assert calls[0][2] == "pb-demo"
    assert "reference_executable" in calls[0][-1]

def test_programbench_terminal_status_reads_explicit_payload_status():
    assert terminal_task_status({"status": "completed"}) == "completed"
    assert terminal_task_status({"status": "failed"}) == "failed"
    assert terminal_task_status({"status": "running"}) == ""
    # cancel_requested is the cancel-intent latch, not the settled record.
    assert terminal_task_status({"status": "cancel_requested"}) == ""
    assert terminal_task_status({}) == ""
    # A completed task with stale provider noise in reason_code stays completed
    # (the harness must never demote it heuristically) but IS flagged as infra
    # noise for the ledger when the axes say so.
    assert terminal_task_status({"status": "completed", "reason_code": "provider_unavailable"}) == "completed"
    assert classify_infra_failure({"reason_code": "llm_api_error"}) is True
    assert classify_infra_failure({"outcome_axes": {"execution": {"status": "infra_failed"}}}) is True
    assert classify_infra_failure({"status": "failed", "reason_code": "task_not_completed"}) is False

def test_programbench_submit_and_wait_polls_until_terminal(monkeypatch, tmp_path):
    import devtools.benchmarks.programbench.programbench_adapter as adapter

    calls: list[tuple[str, str]] = []

    def fake_api(base_url, method, path, body=None, **kwargs):
        calls.append((method, path))
        if method == "POST":
            return {"task_id": "task-123"}
        if len(calls) == 2:
            return {"task_id": "task-123", "status": "running"}
        return {"task_id": "task-123", "status": "completed", "result": "done"}

    monkeypatch.setattr(adapter, "ouroboros_api_request", fake_api)
    monkeypatch.setattr(adapter.time, "sleep", lambda *_args, **_kwargs: None)
    checkpoint = tmp_path / "checkpoint.json"
    result = submit_and_wait(
        "http://127.0.0.1:8765",
        {"description": "solve"},
        timeout_sec=30,
        poll_interval_sec=0,
        checkpoint_path=checkpoint,
    )
    assert result["status"] == "completed"
    assert calls[0] == ("POST", "/api/tasks")
    assert any(path.endswith("/api/tasks/task-123") for _, path in calls)
    saved = json.loads(checkpoint.read_text(encoding="utf-8"))
    assert saved["task_id"] == "task-123"
    assert saved["status"] == "completed"
    assert saved["task_result"]["result"] == "done"

def test_programbench_submit_and_wait_resumes_from_checkpoint_without_resubmit(monkeypatch, tmp_path):
    import devtools.benchmarks.programbench.programbench_adapter as adapter

    checkpoint = tmp_path / "checkpoint.json"
    checkpoint.write_text(json.dumps({"task_id": "task-999", "status": "running"}), encoding="utf-8")
    calls: list[tuple[str, str]] = []

    def fake_api(base_url, method, path, body=None, **kwargs):
        calls.append((method, path))
        assert method == "GET", "a live checkpoint must re-attach, never re-submit"
        return {"task_id": "task-999", "status": "completed", "result": "done"}

    monkeypatch.setattr(adapter, "ouroboros_api_request", fake_api)
    monkeypatch.setattr(adapter.time, "sleep", lambda *_args, **_kwargs: None)
    result = submit_and_wait(
        "http://127.0.0.1:8765",
        {"description": "solve"},
        timeout_sec=30,
        poll_interval_sec=0,
        checkpoint_path=checkpoint,
    )
    assert result["status"] == "completed"
    assert calls == [("GET", "/api/tasks/task-999")]

def test_programbench_submit_and_wait_stale_checkpoint_falls_back_to_fresh_submit(monkeypatch, tmp_path):
    import devtools.benchmarks.programbench.programbench_adapter as adapter

    checkpoint = tmp_path / "checkpoint.json"
    checkpoint.write_text(json.dumps({"task_id": "task-gone", "status": "running"}), encoding="utf-8")
    calls: list[tuple[str, str]] = []

    def fake_api(base_url, method, path, body=None, **kwargs):
        calls.append((method, path))
        if path.endswith("/api/tasks/task-gone"):
            raise RuntimeError("Ouroboros API GET /api/tasks/task-gone failed (404): task not found")
        if method == "POST":
            return {"task_id": "task-new"}
        return {"task_id": "task-new", "status": "completed"}

    monkeypatch.setattr(adapter, "ouroboros_api_request", fake_api)
    monkeypatch.setattr(adapter.time, "sleep", lambda *_args, **_kwargs: None)
    result = submit_and_wait(
        "http://127.0.0.1:8765",
        {"description": "solve"},
        timeout_sec=30,
        poll_interval_sec=0,
        checkpoint_path=checkpoint,
    )
    assert result["status"] == "completed"
    assert ("POST", "/api/tasks") in calls
    assert json.loads(checkpoint.read_text(encoding="utf-8"))["task_id"] == "task-new"

_PROVIDER_ROUTE_ENV_KEYS = (
    "OPENROUTER_API_KEY",
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "OPENAI_BASE_URL",
    "OPENAI_COMPATIBLE_BASE_URL",
    "CLOUDRU_FOUNDATION_MODELS_API_KEY",
    "GIGACHAT_CREDENTIALS",
    "GIGACHAT_USER",
    "GIGACHAT_PASSWORD",
)

def _scrub_model_route_env(monkeypatch):
    from devtools.benchmarks.common.manifests import MODEL_SLOT_KEYS

    for key in (*_PROVIDER_ROUTE_ENV_KEYS, *MODEL_SLOT_KEYS):
        monkeypatch.delenv(key, raising=False)

def test_programbench_model_preflight_rejects_legacy_ids_on_direct_route(tmp_path, monkeypatch):
    from devtools.benchmarks.programbench.run_programbench_e2e import preflight_model_slots

    _scrub_model_route_env(monkeypatch)
    settings = tmp_path / "settings.json"
    settings.write_text(
        json.dumps({"OPENAI_API_KEY": "test-key", "OUROBOROS_MODEL": "openai/gpt-5.5-mini"}),
        encoding="utf-8",
    )
    with pytest.raises(SystemExit, match="openai::gpt-5.5-mini"):
        preflight_model_slots(settings)

def test_programbench_model_preflight_keeps_openrouter_ids_and_checks_solve_model(tmp_path, monkeypatch):
    from devtools.benchmarks.programbench.run_programbench_e2e import preflight_model_slots

    _scrub_model_route_env(monkeypatch)
    settings = tmp_path / "settings.json"
    settings.write_text(
        json.dumps(
            {
                "OPENROUTER_API_KEY": "test-key",
                "OUROBOROS_MODEL": "openai/gpt-5.5-mini",
                "OUROBOROS_REVIEW_MODELS": "openai/gpt-5.5-mini,openai/gpt-5.5-mini",
            }
        ),
        encoding="utf-8",
    )
    # provider/model is the canonical OpenRouter form: no rewrite, no error.
    slots = preflight_model_slots(settings, solve_model="openai/gpt-5.5-mini")
    assert slots["OUROBOROS_MODEL"] == "openai/gpt-5.5-mini"
    assert slots["OUROBOROS_REVIEW_MODELS"] == "openai/gpt-5.5-mini,openai/gpt-5.5-mini"
    with pytest.raises(SystemExit, match="does not match settings OUROBOROS_MODEL"):
        preflight_model_slots(settings, solve_model="anthropic/claude-sonnet-4.6")

def test_programbench_instruction_states_tree_ships_as_is():
    """v6.74.4: the PB instruction must carry the true submission model (live
    tree, .git dropped, uncommitted edits ship) and the final compile.sh check,
    and must no longer claim a fresh checkout."""
    template = " ".join((
        Path(__file__).resolve().parents[1]
        / "devtools" / "benchmarks" / "programbench" / "instruction_template.md"
    ).read_text(encoding="utf-8").split())
    assert "CURRENT state of your working tree" in template
    assert "uncommitted edits DO ship" in template
    assert "The exporter also excludes" in template
    assert "`.ouroboros/`" in template and "at ANY depth" in template
    assert "run `./compile.sh` one final time" in template
    # The negated truth stays; the old false claim must be gone.
    assert "not from a fresh checkout" in template
    assert "on a fresh checkout" not in template

def test_programbench_submission_tarball_contract(tmp_path):
    """v6.74.4 (codex finding 1): the instruction's submission model must match
    the exporter — uncommitted source ships from the LIVE tree; .git, root
    binaries and build/cache noise do not."""
    import tarfile

    from devtools.benchmarks.programbench.programbench_adapter import (
        create_submission_tarball,
    )

    ws = tmp_path / "ws"
    (ws / ".git").mkdir(parents=True)
    (ws / ".git" / "HEAD").write_text("ref: refs/heads/main\n")
    (ws / "build").mkdir()
    (ws / "build" / "obj.o").write_text("obj")
    (ws / "figlet_clone.c").write_text("int main(void){return 0;}\n")  # uncommitted source
    (ws / ".ouroboros").mkdir()
    (ws / ".ouroboros" / "required.h").write_text("#define X 1\n")
    (ws / "compile.sh").write_text("#!/bin/sh\ncc figlet_clone.c -o executable\n")
    (ws / "executable").write_text("bin")
    (ws / "reference_executable").write_text("refbin")
    (ws / "probe.log").write_text("log")
    out = create_submission_tarball(ws, tmp_path / "sub.tar.gz")
    with tarfile.open(out) as tar:
        names = set(tar.getnames())
    assert "figlet_clone.c" in names and "compile.sh" in names
    assert not any(n == "executable" or n == "reference_executable" for n in names)
    assert not any(n.startswith(".git") or n.startswith("build") for n in names)
    assert not any(n.startswith(".ouroboros") for n in names)
    assert "probe.log" not in names
