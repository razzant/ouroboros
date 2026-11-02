"""Opt-in consumers against an existing, isolated, network-none Docker lab.

Set DOCKER_HOST, DOCKER_OUTPUT_LAB_ROOT and
DOCKER_OUTPUT_LAB_CONTAINER. The lab binds root/workspace to /work,
root/overlay to /work/nested, root/outside to /other. This suite creates only
unique qa-docker-* data; it never creates/stops containers or changes contexts.
DOCKER_OUTPUT_EVIDENCE optionally retains consumer results and actual artifacts.
"""
from __future__ import annotations

import json
import os
import pathlib
import shutil
import uuid
import zipfile
from types import SimpleNamespace

import pytest

from ouroboros.tools.registry import ToolContext, ToolRegistry

pytestmark = [pytest.mark.integration, pytest.mark.serial]


@pytest.fixture
def lab(tmp_path, monkeypatch, request):
    from ouroboros import config

    root_text = os.environ.get("DOCKER_OUTPUT_LAB_ROOT")
    container = os.environ.get("DOCKER_OUTPUT_LAB_CONTAINER")
    if not root_text or not container:
        pytest.skip("requires an explicitly supplied isolated Docker output lab")
    root = pathlib.Path(root_text).resolve()
    q = "qa-docker-" + uuid.uuid4().hex
    cwd, overlay, hidden, outside = (root / p / q for p in ("workspace", "overlay", "workspace/nested", "outside"))
    for p in (cwd, overlay, hidden, outside):
        p.mkdir(parents=True)
    config.reset_runtime_mode_baseline_for_tests()
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "advanced")
    monkeypatch.delenv(config.BOOT_RUNTIME_MODE_ENV_KEY, raising=False)
    monkeypatch.setenv("OUROBOROS_DELIVERABLES_ROOT", str(cwd / "Deliverables"))
    monkeypatch.setattr("ouroboros.safety.check_safety", lambda *a, **k: (True, ""))
    system, drive = tmp_path / "system", tmp_path / "data"
    system.mkdir()
    drive.mkdir()
    ctx = ToolContext(repo_dir=system, system_repo_dir=system, drive_root=drive,
                      workspace_root=root, workspace_mode="external", task_id=q,
                      executor_ref={"type": "docker_exec", "container_name": container, "network": "none",
                                    "path_mappings": [{"host_path": str(root / h), "backend_path": b}
                                                      for h, b in (("workspace", "/work"), ("overlay", "/work/nested"), ("outside", "/other"))]})
    registry = ToolRegistry(repo_dir=system, drive_root=drive)
    registry.set_context(ctx)
    state = SimpleNamespace(root=root, cwd=cwd, q=q, overlay=overlay, hidden=hidden, outside=outside,
                            ctx=ctx, registry=registry, results=[], container=container)
    try:
        yield state
    finally:
        from ouroboros import workspace_executor
        for service in workspace_executor._services_snapshot():
            if service.task_id == q:
                stopped = workspace_executor.stop_service(ctx, service.name)
                assert stopped and not stopped.get("stop_failed"), stopped
        # Retain the public results and registered bytes before safe_test reaps scratch.
        evidence = os.environ.get("DOCKER_OUTPUT_EVIDENCE")
        if evidence:
            dest = pathlib.Path(evidence) / f"{request.node.originalname}-{q}"
            dest.mkdir(parents=True, exist_ok=True)
            (dest / "case.json").write_text(json.dumps({"nodeid": request.node.nodeid, "task_id": q}))
            (dest / "consumer-results.json").write_text(json.dumps(state.results, indent=2, default=str))
            for artifact_root in (drive / "task_results" / "artifacts").glob("*"):
                if artifact_root.is_dir():
                    shutil.copytree(artifact_root, dest / artifact_root.name, dirs_exist_ok=True)
        for p in (cwd, overlay, hidden, outside):
            shutil.rmtree(p)


def call(lab, tool="run_command", **args):
    if tool not in {"stop_service", "service_status"}:
        args.setdefault("cwd", str(lab.cwd))
    result = lab.registry.execute_result(tool, args)
    lab.results.append({"tool": tool, "arguments": args, "code": result.code,
                        "status": result.status, "meta": dict(result.meta), "text": result.text})
    print(json.dumps(lab.results[-1], default=str))
    return result


def artifact(lab, name):
    from ouroboros.artifacts import collect_task_artifact_records

    rows = collect_task_artifact_records(lab.ctx.drive_root, lab.q, require_registered=True)
    row = next(row for row in rows if row["name"] == name)
    return pathlib.Path(row["path"])


@pytest.mark.parametrize("spelling", ["link/../answer.txt", "absolute/../answer.txt", "plain"])
def test_physical_symlinks_with_and_without_parent(lab, spelling):
    target = lab.cwd / "target"
    (target / "deep").mkdir(parents=True)
    (lab.cwd / "answer.txt").write_text("HOST DECOY")
    (lab.cwd / "link").symlink_to("target/deep", target_is_directory=True)
    (lab.cwd / "absolute").symlink_to(f"/work/{lab.q}/target/deep", target_is_directory=True)
    (lab.cwd / "plain").symlink_to(f"/work/{lab.q}/target/answer.txt")
    result = call(lab, cmd=["sh", "-c", f"printf container-bytes > {spelling}"], outputs=[spelling])
    assert result.code == "OK", result.text
    assert artifact(lab, "answer.txt").read_text() == "container-bytes"
    assert (lab.cwd / "answer.txt").read_text() == "HOST DECOY"


@pytest.mark.parametrize("absolute", [False, True])
def test_dangling_link_and_missing_components_via_script(lab, absolute):
    (lab.cwd / "future").symlink_to(f"/work/{lab.q}/new/deep" if absolute else "new/deep")
    result = call(lab, "run_script", interpreter="sh",
                  script="mkdir -p new/deep; printf future-bytes > future/../answer.txt",
                  outputs=["future/../answer.txt"])
    assert result.code == "OK", result.text
    assert artifact(lab, "answer.txt").read_text() == "future-bytes"


def test_probe_preserves_shell_characters_and_newlines_in_components(lab):
    target = lab.cwd / "target ' $ ;\n"
    target.mkdir()
    (lab.cwd / "alias").symlink_to(target.name, target_is_directory=True)
    result = call(lab, cmd=["sh", "-c", "printf exact > alias/answer.txt"], outputs=["alias/answer.txt"])
    assert result.code == "OK", result.text
    assert artifact(lab, "answer.txt").read_text() == "exact"


def test_unreadable_baseline_warns_before_command_without_preventing_effect(lab):
    output = lab.cwd / "answer.txt"
    output.write_text("existing")
    output.chmod(0o000)
    seen = []
    lab.ctx.emit_progress_fn = lambda text: seen.append((text, (lab.cwd / "marker").exists()))
    try:
        result = call(lab, cmd=["sh", "-c", "printf ran > marker; printf preserved-stdout"],
                      outputs=["answer.txt"])
        assert result.code == "ARTIFACT_OUTPUT_ERROR" and result.meta["exit_code"] == 0, result.text
        assert any("baseline unavailable" in text.lower() and not marker for text, marker in seen)
        assert "baseline unavailable before the run" in result.text
        assert "preserved-stdout" in result.text and (lab.cwd / "marker").read_text() == "ran"
        lab.results.append({"baseline_progress": seen})
    finally:
        output.chmod(0o644)


def test_output_still_absent_after_effect_is_an_export_error(lab):
    result = call(lab, cmd=["sh", "-c", "printf once > marker"], outputs=["new/deep/absent.txt"])
    assert result.code == "ARTIFACT_OUTPUT_ERROR" and result.meta["exit_code"] == 0, result.text
    assert "missing output" in result.text and (lab.cwd / "marker").read_text() == "once"


@pytest.mark.parametrize("output,setup,valid", [
    ("missing/../answer.txt", "", False),
    ("answer.txt/", "", False),
    ("created/../answer.txt", "mkdir created; ", True),
    ("answer-dir/", "mkdir answer-dir; cp answer.txt answer-dir/; ", True),
])
def test_original_output_address_must_exist_after_command(lab, output, setup, valid):
    from ouroboros.artifacts import collect_task_artifact_records

    result = call(lab, cmd=["sh", "-c", "printf bytes > answer.txt; " + setup + "printf once > marker; printf preserved-stdout"],
                  outputs=[output])
    assert result.code == ("OK" if valid else "ARTIFACT_OUTPUT_ERROR"), result.text
    assert result.meta["exit_code"] == 0 and "preserved-stdout" in result.text
    assert (lab.cwd / "marker").read_text() == "once"
    rows = collect_task_artifact_records(lab.ctx.drive_root, lab.q, require_registered=True)
    assert bool(rows) is valid


@pytest.mark.parametrize("suffix,location", [("../", "workspace"), ("child/../", "overlay")])
def test_mount_parent_is_container_parent(lab, suffix, location):
    # Existing nested mount's parent is /work, independently of its host Source.
    if location == "overlay":
        (lab.overlay / "child").mkdir()
        output = f"/work/nested/{lab.q}/{suffix}answer.txt"
        expected = lab.overlay / "answer.txt"
    else:
        output = f"/work/nested/{suffix}{lab.q}/answer.txt"
        expected = lab.cwd / "answer.txt"
    result = call(lab, cmd=["sh", "-c", f"printf mount-bytes > {output}"], outputs=[output])
    assert result.code == "OK", result.text
    assert expected.read_text() == artifact(lab, "answer.txt").read_text() == "mount-bytes"


def test_missing_tail_can_return_to_physical_tree(lab):
    (lab.cwd / "target").mkdir()
    (lab.cwd / "jump").symlink_to("target")
    result = call(lab, cmd=["sh", "-c", "mkdir -p missing; printf resumed > missing/../jump/answer.txt"],
                  outputs=["missing/../jump/answer.txt"])
    assert result.code == "OK", result.text
    assert artifact(lab, "answer.txt").read_text() == "resumed"


@pytest.mark.parametrize("case", ["unmapped_mount", "wrong_mapping", "outside_root", "protected", "host_absolute_link"])
def test_invalid_output_refuses_before_marker(lab, case):
    output = "answer.txt"
    if case == "unmapped_mount":
        lab.ctx.executor_ref["path_mappings"] = lab.ctx.executor_ref["path_mappings"][:1]
        output = f"/work/nested/{lab.q}/answer.txt"
    elif case == "wrong_mapping":
        lab.ctx.executor_ref["path_mappings"][0]["host_path"] = str(lab.root)
    elif case == "outside_root":
        lab.ctx.workspace_root = lab.root / "workspace"
        (lab.cwd / "escape").symlink_to(f"/other/{lab.q}", target_is_directory=True)
        output = "escape/answer.txt"
    elif case == "protected":
        output = ".env"
    else:
        (lab.cwd / "host-link").symlink_to(str(lab.cwd / "answer.txt"))
        output = "host-link"
    result = call(lab, cmd=["sh", "-c", "printf ran > marker"], outputs=[output])
    assert result.code == "RESOURCE_POLICY_BLOCKED", result.text
    assert not (lab.cwd / "marker").exists()


def test_unavailable_capability_refuses_before_effect(lab, monkeypatch):
    from ouroboros import workspace_executor_paths as paths

    original = paths._docker_output_probe
    def unavailable(argv):
        if argv[0] == "exec":
            raise paths.OutputResolutionUnavailable("injected readlink capability unavailable")
        return original(argv)
    monkeypatch.setattr(paths, "_docker_output_probe", unavailable)
    result = call(lab, cmd=["sh", "-c", "printf ran > marker"], outputs=["answer.txt"])
    assert result.code == "CAPABILITY_UNAVAILABLE" and result.status == "unavailable", result.text
    assert result.meta["process_not_started"] and not (lab.cwd / "marker").exists()


def test_after_effect_unavailable_preserves_stdout_exit_and_single_effect(lab, monkeypatch):
    from ouroboros import workspace_executor_paths as paths

    original = paths._docker_output_probe
    def unavailable(argv):
        if (lab.cwd / "marker").exists():
            raise paths.OutputResolutionUnavailable("injected unavailable after confirmed execution")
        return original(argv)
    monkeypatch.setattr(paths, "_docker_output_probe", unavailable)
    result = call(lab, cmd=["sh", "-c", "printf once >> marker; printf result > answer.txt; printf preserved-stdout"],
                  outputs=["answer.txt"])
    assert result.code == "ARTIFACT_OUTPUT_ERROR", result.text
    assert result.meta["exit_code"] == 0 and result.meta["operation_outcome"] == "completed"
    assert "preserved-stdout" in result.text and (lab.cwd / "marker").read_text() == "once"
    assert '"container_name":' in result.text and "EXECUTOR_TRACE:" in result.text


def test_post_effect_link_escape_is_refused(lab):
    lab.ctx.workspace_root = lab.root / "workspace"
    result = call(lab, cmd=["sh", "-c", f"printf once > marker; ln -s /other/{lab.q} out; printf external > out/answer.txt"],
                  outputs=["out/answer.txt"])
    assert result.code == "ARTIFACT_OUTPUT_ERROR", result.text
    assert result.meta["exit_code"] == 0 and (lab.cwd / "marker").read_text() == "once"


def test_directory_view_and_baseline_include_overlay_exclude_hidden_decoy(lab):
    (lab.hidden / "decoy.txt").write_text("HIDDEN HOST DECOY")
    (lab.overlay / "visible.txt").write_text("old")
    (lab.overlay / ".env").write_text("skipped credential")
    (lab.overlay / "link").symlink_to("visible.txt")
    output = "/work/nested/.."
    result = call(lab, cmd=["sh", "-c", f"printf overlay-changed > /work/nested/{lab.q}/visible.txt"], outputs=[output])
    assert result.code == "OK", result.text
    from ouroboros.artifacts import collect_task_artifact_records
    rows = collect_task_artifact_records(lab.ctx.drive_root, lab.q, require_registered=True)
    zips = [pathlib.Path(row["path"]) for row in rows if row["name"].endswith(".zip")]
    assert len(zips) == 1
    with zipfile.ZipFile(zips[0]) as z:
        assert z.read(f"nested/{lab.q}/visible.txt") == b"overlay-changed"
        assert f"nested/{lab.q}/decoy.txt" not in z.namelist()
        assert f"nested/{lab.q}/.env" not in z.namelist()
        assert f"nested/{lab.q}/link" not in z.namelist()
    unchanged = call(lab, cmd=["sh", "-c", ":"], outputs=[output])
    assert unchanged.code == "OK" and "unchanged output" in unchanged.text, unchanged.text


@pytest.mark.skipif(os.environ.get("DOCKER_OUTPUT_LAB_TMPFS") != "1",
                    reason="requires lab --tmpfs /work/tmpfs_result/cache:size=1m")
@pytest.mark.parametrize("direct", [False, True], ids=["directory-ancestor", "below-tmpfs"])
def test_hostconfig_tmpfs_refuses_before_command(lab, direct):
    from ouroboros.artifacts import collect_task_artifact_records
    from ouroboros.workspace_executor_paths import _docker_output_probe

    mount = "/work/tmpfs_result/cache"
    facts = json.loads(_docker_output_probe([
        "inspect", "-f", '{"mounts":{{json .Mounts}},"tmpfs":{{json .HostConfig.Tmpfs}}}', lab.container,
    ]))
    lab.results.append({"inspect": facts})
    assert mount in facts["tmpfs"] and not any(m["Destination"] == mount for m in facts["mounts"]), facts
    hidden = lab.root / "workspace/tmpfs_result/cache" / lab.q
    visible = lab.root / "workspace/tmpfs_result" / lab.q
    hidden.mkdir(parents=True)
    visible.mkdir()
    (hidden / "answer.txt").write_text("HIDDEN HOST DECOY")
    backend = f"{mount}/{lab.q}"
    try:
        setup = call(lab, cmd=["sh", "-c", 'mkdir -p "$1"; printf RAM-BYTES > "$1/answer.txt"; cat "$1/answer.txt"',
                               "tmpfs-setup", backend])
        assert setup.code == "OK" and "RAM-BYTES" in setup.text, setup.text
        output = f"{backend}/answer.txt" if direct else "/work/tmpfs_result"
        result = call(lab, cmd=["sh", "-c", f"printf ran > marker; printf visible > /work/tmpfs_result/{lab.q}/visible.txt"],
                      outputs=[output])
        assert result.code == "RESOURCE_POLICY_BLOCKED", result.text
        assert "not a mapped host bind" in result.text
        assert result.meta["operation_outcome"] == "completed_no_effect"
        assert not (lab.cwd / "marker").exists() and not (visible / "visible.txt").exists()
        assert not collect_task_artifact_records(lab.ctx.drive_root, lab.q, require_registered=True)
        assert (hidden / "answer.txt").read_text() == "HIDDEN HOST DECOY"
    finally:
        cleanup = call(lab, cmd=["rm", "-rf", backend])
        assert cleanup.code == "OK", cleanup.text
        shutil.rmtree(hidden)
        shutil.rmtree(visible)


@pytest.mark.parametrize("export_unavailable", [False, True])
def test_service_uses_start_container_and_keeps_confirmed_stop(lab, monkeypatch, export_unavailable):
    from ouroboros import workspace_executor
    from ouroboros import workspace_executor_paths as paths

    (lab.cwd / "absolute").symlink_to(f"/work/{lab.q}/answer.txt")
    started = call(lab, "start_service", name="output-svc", outputs=["absolute"],
                   cmd=["sh", "-c", "printf service-bytes > absolute; printf READY; while :; do sleep 1; done"],
                   readiness={"log_contains": "READY", "timeout_sec": 10})
    assert started.status == "ok", started.text
    records = [r for r in workspace_executor._services_snapshot() if r.task_id == lab.q]
    assert len(records) == 1
    assert records[0].executor.container_name != lab.container  # immutable inspected ID
    assert json.loads(started.text)["executor"]["container_name"] == records[0].executor.container_name
    lab.ctx.executor_ref = None  # stop/export retain the service's start executor
    if export_unavailable:
        def unavailable(argv):
            raise paths.OutputResolutionUnavailable("injected export unavailable after service stop")
        monkeypatch.setattr(paths, "_docker_output_probe", unavailable)
    stopped = call(lab, "stop_service", name="output-svc")
    assert stopped.code == ("ARTIFACT_OUTPUT_ERROR" if export_unavailable else "OK"), stopped.text
    assert '"state": "stopped"' in stopped.text and '"cleanup_dispatched": true' in stopped.text
    assert not [r for r in workspace_executor._services_snapshot() if r.task_id == lab.q]
    if not export_unavailable:
        assert artifact(lab, "answer.txt").read_text() == "service-bytes"


@pytest.mark.parametrize("export_unavailable", [False, True])
def test_automatic_service_stop_finalizes_outputs(lab, monkeypatch, export_unavailable):
    from ouroboros import workspace_executor_paths as paths
    from ouroboros.tools.services import stop_task_services

    started = call(lab, "start_service", name="output-svc", outputs=["answer.txt"],
                   cmd=["sh", "-c", "printf service > answer.txt; printf READY; while :; do sleep 1; done"],
                   readiness={"log_contains": "READY", "timeout_sec": 10})
    assert started.status == "ok", started.text
    lab.ctx.executor_ref = None
    if export_unavailable:
        def unavailable(argv):
            raise paths.OutputResolutionUnavailable("injected export unavailable after automatic stop")
        monkeypatch.setattr(paths, "_docker_output_probe", unavailable)
    results = stop_task_services(lab.ctx)
    assert len(results) == 1 and results[0]["lifecycle"] == "stopped", results
    assert not any(key.startswith("_") for key in results[0])
    assert results[0]["artifact_output_failed"] is export_unavailable
    if export_unavailable:
        assert "injected export unavailable" in results[0]["artifact_outputs"]
    else:
        assert artifact(lab, "answer.txt").read_text() == "service"
    lab.results.append(json.loads(json.dumps({"bulk_custody": results})))
