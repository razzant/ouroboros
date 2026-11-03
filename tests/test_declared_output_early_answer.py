"""#1797: a declared output the resolver refuses is answered before the process starts.

Outputs alone accept a relative ``..`` spelling; its resolved location is judged like
the absolute spelling (owner decisions, 18 Oct 2026). Post-run registration still
re-resolves every output, so a link swapped during the run is refused at the end.
"""
from __future__ import annotations

import hashlib
import json
import os
import pathlib
import re
import subprocess
import sys
import time

import pytest

from ouroboros.tools.registry import ToolContext, ToolRegistry


def _code(text):
    return "\n".join([
        "from pathlib import Path",
        "Path('marker.txt').write_text('ran')",
        *text,
    ])


@pytest.fixture
def project(tmp_path, monkeypatch):
    from ouroboros import config as cfg

    cfg.reset_runtime_mode_baseline_for_tests()
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "advanced")
    monkeypatch.delenv(cfg.BOOT_RUNTIME_MODE_ENV_KEY, raising=False)
    home, system, active, data, outside = (tmp_path / name for name in ("home", "system", "active", "data", "outside"))
    for path in (home, system, active / "site", data, outside / "deep" / "er"):
        path.mkdir(parents=True)
    monkeypatch.setenv("OUROBOROS_USER_FILES_ROOT", str(home))
    monkeypatch.setenv("OUROBOROS_DELIVERABLES_ROOT", str(active / "Deliverables"))
    (active / "Deliverables").mkdir()
    monkeypatch.setattr("ouroboros.safety.check_safety", lambda *a, **k: (True, ""))
    ctx = ToolContext(repo_dir=system, system_repo_dir=system, drive_root=data, workspace_root=active,
                      workspace_mode="external", task_id="early-output")
    registry = ToolRegistry(repo_dir=system, drive_root=data)
    registry.set_context(ctx)
    return registry, active, outside


def _run(registry, tool, cwd, outputs, *lines):
    args = {"cwd": str(cwd), "outputs": outputs}
    if tool == "run_script":
        args.update(script=_code(lines), interpreter=sys.executable)
    else:
        args["cmd"] = [sys.executable, "-c", _code(lines)]
    return registry.execute_result(tool, args)


def _trace_item(tool, args, result):
    from ouroboros.loop_tool_execution import _extract_result_metadata, _is_tool_execution_failure

    is_error = _is_tool_execution_failure(True, result.text, result, fn_name=tool)
    meta = _extract_result_metadata(tool, result.text, is_error, result)
    return {"tool": tool, "args": args, "is_error": is_error, "status": meta.get("status"),
            "artifact_registered": bool(meta.get("artifact_registered")), "result": result.text}


@pytest.mark.parametrize("tool", ["run_command", "run_script"])
def test_a_refused_declaration_stops_the_call_and_the_corrected_one_registers(project, tool):
    from ouroboros._outcome_tool_errors import _classify_tool_errors

    registry, active, _outside = project
    refused = _run(registry, tool, active / "site", ["../../escaped.txt"], "Path('../../escaped.txt').write_text('x')")
    assert refused.code == "RESOURCE_POLICY_BLOCKED", refused.text
    assert refused.meta.get("operation_outcome") == "completed_no_effect"
    assert "output escapes allowed artifact roots: ../../escaped.txt" in refused.text
    assert not (active / "site" / "marker.txt").exists() and not (active.parent / "escaped.txt").exists()

    corrected = _run(registry, tool, active / "site", ["../report.txt"], "Path('../report.txt').write_text('ok')")
    assert corrected.code == "OK", corrected.text
    assert "ARTIFACT_OUTPUTS" in corrected.text and (active / "report.txt").read_text() == "ok"
    trace = {"tool_calls": [_trace_item(tool, {"outputs": ["../../escaped.txt"]}, refused),
                            _trace_item(tool, {"outputs": ["../report.txt"]}, corrected)]}
    classified = _classify_tool_errors(trace)
    assert classified["unresolved"] == [], "a refused-then-corrected declaration is not a tool failure"
    assert [item["status"] for item in classified["policy_denials"] + classified["recovered"]] == ["resource_policy_blocked"]


def test_relative_and_absolute_spellings_of_one_location_get_one_answer(project):
    registry, active, _outside = project
    deliverables = active / "Deliverables"
    for index, outputs in enumerate((["../Deliverables/page.html"], [str(deliverables / "page.html")])):
        result = _run(registry, "run_command", active / "site", outputs,
                      f"Path('../Deliverables/page.html').write_text('<p>{index}</p>')")
        assert result.code == "OK" and "registered output" in result.text, result.text
    (active / "site" / "marker.txt").unlink()
    for outputs in (["../Deliverables/.env"], [str(deliverables / ".env")]):
        result = _run(registry, "run_command", active / "site", outputs)
        assert result.code == "RESOURCE_POLICY_BLOCKED" and "credential-like output .env" in result.text, result.text
    assert not (active / "site" / "marker.txt").exists()


@pytest.mark.skipif(os.name == "nt", reason="backslash is a native filename character only on POSIX")
def test_directory_output_keeps_native_backslash_member(project):
    import zipfile

    from ouroboros.artifacts import collect_task_artifact_records

    registry, active, _ = project
    name = r"part\name.txt"
    result = _run(registry, "run_command", active, ["bundle"],
                  "Path('bundle').mkdir()", f"Path('bundle', {name!r}).write_text('native-name')")
    print(result.text)
    assert result.code == "OK" and result.meta["artifact_registered"], result.text
    rows = collect_task_artifact_records(registry._ctx.drive_root, registry._ctx.task_id, require_registered=True)
    manifest = json.loads(next(pathlib.Path(row["path"]) for row in rows if row["name"].endswith(".manifest.json")).read_text())
    assert [row["path"] for row in manifest["files"]] == [name]
    with zipfile.ZipFile(next(row["path"] for row in rows if row["name"].endswith(".zip"))) as archive:
        assert archive.namelist() == [name] and archive.read(name) == b"native-name"


@pytest.mark.parametrize("names", [["/absolute.txt"], ["nested/../escape.txt"], ["same.txt", "same.txt"]])
def test_directory_archive_refuses_absolute_traversal_and_duplicate_members(project, names):
    from ouroboros import artifacts

    registry, active, _ = project
    source = active / "bundle"
    source.mkdir()
    member = source / "ordinary.txt"
    member.write_text("source-bytes")
    with pytest.raises(OSError, match="invalid or duplicate directory member"):
        artifacts.copy_directory_to_task_artifacts(
            registry._ctx, source, member_sources=[(name, member) for name in names],
        )
    assert not artifacts.collect_task_artifact_records(
        registry._ctx.drive_root, registry._ctx.task_id, require_registered=True,
    )


def _directory_link(link, target):
    try:
        os.symlink(target, link, target_is_directory=True)
    except OSError:
        pytest.skip("directory links cannot be created here")


def test_parent_spelling_keeps_its_physical_meaning_through_a_symlink(project):
    registry, active, outside = project
    site = active / "site"
    _directory_link(site / "alias", outside / "deep" / "er")
    # Collapsed lexically these name site/decoy.txt and Deliverables/x.html. The OS decides what
    # ``alias/..`` means: POSIX resolves the link first (outside the roots), Windows collapses ``..``
    # first. The answer follows the location the OS would write, judged by its own realpath.
    for output, refusal in (("alias/../decoy.txt", "output escapes allowed artifact roots"),
                            ("alias/../../Deliverables/x.html", "Deliverables output escapes its resolved configured root")):
        physical = pathlib.Path(os.path.realpath(site / output))
        result = _run(registry, "run_command", site, [output])
        if physical.is_relative_to(active.resolve()):
            assert result.code != "RESOURCE_POLICY_BLOCKED", (output, result.text)
        else:
            assert result.code == "RESOURCE_POLICY_BLOCKED" and refusal in result.text, (output, result.text)
            assert not (site / "marker.txt").exists()


# #1797 behind an explicit local executor. Its process runs in this host cwd
# (workspace_executor.execute), so a relative output keeps the host meaning it has without
# an executor; Docker filesystem behavior is covered by test_docker_declared_outputs_live.py.
@pytest.fixture
def local_project(project, monkeypatch):
    from ouroboros import workspace_executor

    registry, active, outside = project
    registry._ctx.executor_ref = {"type": "local", "id": "local-fixture",
                                  "workspace_host_path": str(active), "workspace_backend_path": "/workspace/project"}
    launched, real = [], workspace_executor._execute_local
    monkeypatch.setattr(workspace_executor, "_execute_local", lambda executor, cmd, cwd, *a, **k: (
        launched.append((executor.kind, pathlib.Path(cwd))) or real(executor, cmd, cwd, *a, **k)))
    return registry, active, outside, launched


def _registered(registry, text):
    """Artifact-store bytes per registered source, read back from the store itself."""
    store = pathlib.Path(registry._ctx.drive_root) / "task_results" / "artifacts" / registry._ctx.task_id
    found = {}
    for source, name, digest in re.findall(r"registered output (.+?) -> artifact_store:(\S+) sha256=([0-9a-f]{12})", text):
        raw = (store / name).read_bytes()
        assert hashlib.sha256(raw).hexdigest().startswith(digest)
        found[pathlib.Path(source)] = raw
    return found


@pytest.mark.parametrize("tool", ["run_command", "run_script"])
def test_a_local_executor_registers_the_parent_output_its_host_process_wrote(local_project, tool):
    registry, active, _outside, launched = local_project
    escaped = _run(registry, tool, active / "site", ["../../escaped.txt"], "Path('../../escaped.txt').write_text('x')")
    assert escaped.code == "RESOURCE_POLICY_BLOCKED", escaped.text
    assert escaped.meta.get("operation_outcome") == "completed_no_effect"
    assert "output escapes allowed artifact roots: ../../escaped.txt" in escaped.text
    assert launched == [] and not (active / "site" / "marker.txt").exists() and not (active.parent / "escaped.txt").exists()

    result = _run(registry, tool, active / "site", ["../report.txt"], "Path('../report.txt').write_text('local ok')")
    assert result.code == "OK", result.text
    assert launched == [("local", (active / "site").resolve())], "the executor's host process ran the command"
    assert _registered(registry, result.text) == {(active / "report.txt").resolve(): b"local ok"}


def test_a_local_executor_parent_output_keeps_its_physical_meaning_through_a_symlink(local_project):
    registry, active, outside, launched = local_project
    site = active / "site"
    (active / "deep" / "dir").mkdir(parents=True)
    _directory_link(site / "inner", active / "deep" / "dir")
    _directory_link(site / "alias", outside / "deep" / "er")
    (site / "x.txt").write_text("decoy")  # where a lexical collapse of inner/../x.txt would point
    physical = pathlib.Path(os.path.realpath(site / "inner" / ".." / "x.txt"))
    result = _run(registry, "run_command", site, ["inner/../x.txt"], "Path('inner/../x.txt').write_text('physical')")
    assert result.code == "OK", result.text
    assert _registered(registry, result.text) == {physical: b"physical"}
    if physical != (site / "x.txt").resolve():  # POSIX resolves the link before '..'
        assert (site / "x.txt").read_text() == "decoy"
    (site / "marker.txt").unlink()
    if not pathlib.Path(os.path.realpath(site / "alias" / ".." / "decoy.txt")).is_relative_to(active.resolve()):
        escaped = _run(registry, "run_command", site, ["alias/../decoy.txt"])
        assert escaped.code == "RESOURCE_POLICY_BLOCKED" and "escapes allowed artifact roots" in escaped.text, escaped.text
        assert not (site / "marker.txt").exists() and len(launched) == 1


def test_a_local_executor_keeps_the_mapper_for_an_absolute_backend_spelling(local_project):
    registry, active, _outside, launched = local_project
    traversal = _run(registry, "run_command", active / "site", ["/workspace/project/x/../report.txt"])
    assert traversal.code == "RESOURCE_POLICY_BLOCKED" and "not a usable backend path" in traversal.text, traversal.text
    assert launched == []
    mapped = _run(registry, "run_command", active / "site", ["/workspace/project/report.txt"],
                  "Path('../report.txt').write_text('mapped')")
    assert mapped.code == "OK", mapped.text
    assert _registered(registry, mapped.text) == {(active / "report.txt").resolve(): b"mapped"}


def test_absent_and_unchanged_tracked_outputs_are_judged_after_the_run(project, tmp_path):
    registry, _active, _outside = project
    system = registry._ctx.system_repo_dir
    git = ["git", "-c", "user.name=t", "-c", "user.email=t@t", "-C", str(system)]
    subprocess.run(["git", "init", "-q", str(system)], check=True)
    (system / "tracked.txt").write_text("before")
    subprocess.run([*git, "add", "tracked.txt"], check=True)
    subprocess.run([*git, "commit", "-qm", "base"], check=True)
    regenerated = _run(registry, "run_command", system, ["tracked.txt", "fresh.txt"],
                       "Path('tracked.txt').write_text('after')", "Path('fresh.txt').write_text('new')")
    assert regenerated.code == "OK", regenerated.text
    assert regenerated.text.count("registered output") == 2
    subprocess.run([*git, "add", "-A"], check=True)
    subprocess.run([*git, "commit", "-qm", "regenerated"], check=True)  # tracked.txt is clean again
    (system / "marker.txt").unlink()
    untouched = _run(registry, "run_command", system, ["tracked.txt"])
    # The command ran: an unchanged repo file is a post-run fact, never a pre-launch refusal.
    assert (system / "marker.txt").exists()
    assert untouched.code == "ARTIFACT_OUTPUT_ERROR" and "unchanged repo output tracked.txt" in untouched.text


def test_a_link_swapped_during_the_run_is_refused_at_registration(project, tmp_path):
    registry, active, outside = project
    _directory_link(tmp_path / "probe", outside)  # the command below creates the same kind of link
    result = _run(registry, "run_command", active / "site", ["out/report.txt"],
                  "import os, pathlib",
                  f"os.symlink({str(outside)!r}, 'out', target_is_directory=True)",
                  "Path('out/report.txt').write_text('elsewhere')")
    assert (outside / "report.txt").exists(), "the declaration was valid when the command started"
    assert result.code == "ARTIFACT_OUTPUT_ERROR" and "escapes allowed artifact roots" in result.text, result.text


@pytest.mark.parametrize("tool", ["run_command", "run_script", "start_service"])
@pytest.mark.parametrize("absolute", [False, True])
@pytest.mark.parametrize("invalid", ["\x00", "\x01"])
def test_an_unusable_output_spelling_is_a_refusal_not_a_crash(project, tool, absolute, invalid):
    registry, active, _outside = project
    spelling = f"bad{invalid}name.txt"
    output = str(active / "site" / spelling) if absolute else spelling
    if tool == "start_service":
        result = registry.execute_result(tool, {
            "name": "unusable", "cmd": [sys.executable, "-c", _code([])],
            "cwd": str(active / "site"), "outputs": [output], "readiness": {"timeout_sec": 0},
        })
        assert registry.execute("service_status", {"name": "unusable"}) == "⚠️ SERVICE_NOT_FOUND: unusable"
    else:
        result = _run(registry, tool, active / "site", [output])
    assert result.code == "RESOURCE_POLICY_BLOCKED" and "not a usable path" in result.text
    assert result.meta.get("operation_outcome") == "completed_no_effect"
    assert not (active / "site" / "marker.txt").exists()


def test_safe_relpath_keeps_its_rule_and_outputs_reuse_only_its_character_check():
    from ouroboros.utils import normalized_path_text, safe_relpath

    with pytest.raises(ValueError, match="Path traversal"):
        safe_relpath("../x")
    assert safe_relpath("\\a\\b") == "a/b"
    assert normalized_path_text("..\\x") == "../x"
    for bad in ("..\x00", "..\x01"):
        with pytest.raises(ValueError, match="NUL|control character"):
            normalized_path_text(bad)


def test_the_baseline_keeps_a_clean_tracked_output(project):
    from ouroboros.tools.shell import _snapshot_declared_outputs

    registry, _active, _outside = project
    system = registry._ctx.system_repo_dir
    (system / "tracked.txt").write_text("before")
    # Pre-run nothing is changed yet; the baseline is still taken, as the pre-launch view.
    baseline = _snapshot_declared_outputs(registry._ctx, ["tracked.txt"], system, cwd_root="system_repo", changed_paths=set())
    assert list(baseline) == [str((system / "tracked.txt").resolve())] and baseline[str((system / "tracked.txt").resolve())][0]


def test_a_baseline_that_could_not_be_taken_is_said_before_the_command_runs(project, monkeypatch):
    from ouroboros.tools import shell_outputs

    registry, active, _outside = project
    marker = active / "site" / "marker.txt"
    seen = []
    registry._ctx.emit_progress_fn = lambda text: seen.append((text, marker.exists()))
    monkeypatch.setattr(shell_outputs, "_snapshot_declared_outputs", lambda *a, **k: (_ for _ in ()).throw(OSError("store offline")))
    result = _run(registry, "run_command", active / "site", ["page.html"], "Path('page.html').write_text('p')")
    assert seen and seen[0][0].startswith("⚠️ Output baseline unavailable: ") and "store offline" not in seen[0][0]
    assert seen[0][1] is False, "the disclosure precedes the command"
    assert result.code == "OK" and "output baseline unavailable before the run (OSError: store offline)" in result.text


# The real I/O boundary: an existing output the baseline cannot read is not an absent one.
# It becomes readable during the run with unchanged bytes, so only the disclosure says that
# the registered output may predate the run. Owner permission bits do not bind root or Windows.
posix_permissions = pytest.mark.skipif(os.name == "nt" or (hasattr(os, "geteuid") and os.geteuid() == 0),
                                       reason="POSIX permission bits do not restrict this user")


def _locked(path, mode):
    path.chmod(mode)
    return f"os.chmod({str(path)!r}, 0o755 if os.path.isdir({str(path)!r}) else 0o644)"


@posix_permissions
@pytest.mark.parametrize("tool", ["run_command", "run_script"])
@pytest.mark.parametrize("shape", ["unreadable_file", "unreadable_directory", "unsearchable_parent"])
def test_an_existing_output_the_baseline_cannot_read_is_disclosed(project, tool, shape):
    registry, active, _outside = project
    site = active / "site"
    marker = site / "marker.txt"
    seen = []
    registry._ctx.emit_progress_fn = lambda text: seen.append((text, marker.exists()))
    if shape == "unreadable_file":
        output, target = "kept.txt", site / "kept.txt"
        target.write_text("same")
        unlock = _locked(target, 0o000)  # the read fails
    elif shape == "unreadable_directory":
        output, target = "bundle", site / "bundle"
        target.mkdir()
        (target / "page.html").write_text("same")
        unlock = _locked(target, 0o000)  # the walk fails
    else:
        output, target = "locked/kept.txt", site / "locked"
        target.mkdir()
        (target / "kept.txt").write_text("same")
        unlock = _locked(target, 0o600)  # the stat of the output fails
    try:
        result = _run(registry, tool, site, [output], "import os", unlock)
    finally:
        os.chmod(target, 0o755 if target.is_dir() else 0o644)
    warned = [row for row in seen if row[0].startswith("⚠️ Output baseline unavailable: ")]
    assert warned and warned[0][1] is False, ("the disclosure precedes the process", seen)
    assert "PermissionError" not in warned[0][0], "the owner row carries no exception text"
    assert result.code == "OK", result.text
    assert "registered" in result.text and "output baseline unavailable before the run (PermissionError" in result.text


def test_an_absent_output_and_a_readable_one_need_no_disclosure(project):
    registry, active, _outside = project
    (active / "site" / "kept.txt").write_text("same")
    seen = []
    registry._ctx.emit_progress_fn = seen.append
    result = _run(registry, "run_command", active / "site", ["kept.txt", "fresh.txt"], "Path('fresh.txt').write_text('new')")
    assert result.code == "OK" and "output baseline unavailable" not in result.text, result.text
    assert "unchanged output (cosmetic): kept.txt" in result.text and not any("baseline" in row for row in seen)


@posix_permissions
def test_registration_keeps_reading_an_unreadable_output_as_not_fingerprinted(tmp_path):
    from ouroboros.tools.shell_outputs import _fingerprint_output

    locked = tmp_path / "locked.txt"
    locked.write_text("x")
    locked.chmod(0o000)
    try:
        # Post-run readers keep their lenient answer; only the baseline asks for strictness.
        assert _fingerprint_output(locked) == (False, -1, "")
        with pytest.raises(PermissionError):
            _fingerprint_output(locked, strict=True)
        assert _fingerprint_output(tmp_path / "absent.txt", strict=True) == (False, -1, "")
    finally:
        locked.chmod(0o644)


# #1797 on start_service. Services spawn real processes and share
# tools.services._SERVICES, hence serial; the helpers are that suite's own.
from tests.test_services_tool_v2 import (  # noqa: E402
    _force_advanced_runtime, _force_light_runtime, _wait_for_service_log,
)


@pytest.mark.serial
def test_start_refuses_a_refused_output_before_launch_and_exports_a_parent_spelling(tmp_path, monkeypatch):
    _force_light_runtime(monkeypatch)
    monkeypatch.setattr("ouroboros.safety.check_safety", lambda *a, **k: (True, ""))
    repo, drive = tmp_path / "repo", tmp_path / "data"
    repo.mkdir()
    registry = ToolRegistry(repo_dir=repo, drive_root=drive)
    registry._ctx.task_id = "task-early-output"
    task_drive = drive / "task_drives" / "task-early-output"
    (task_drive / "site").mkdir(parents=True)
    body = "from pathlib import Path; Path('marker').touch(); Path('../qualification.html').write_text('<p>ok</p>'); print('READY', flush=True)"
    refused = registry.execute_result("start_service", {
        "name": "early", "cmd": [sys.executable, "-c", body], "cwd": str(task_drive / "site"),
        "outputs": ["../../../outside.html"], "readiness": {"timeout_sec": 1}})
    assert refused.code == "RESOURCE_POLICY_BLOCKED", refused.text
    assert refused.meta.get("operation_outcome") == "completed_no_effect"
    assert not (task_drive / "site" / "marker").exists()
    assert registry.execute("service_status", {"name": "early"}).startswith("⚠️ SERVICE_NOT_FOUND")

    started = json.loads(registry.execute("start_service", {
        "name": "early", "cmd": [sys.executable, "-c", body], "cwd": str(task_drive / "site"),
        "outputs": ["../qualification.html"], "readiness": {"log_contains": "READY", "timeout_sec": 5}}))
    assert "output_baseline_unavailable" not in started
    _wait_for_service_log(drive, "task-early-output", "early.log", lambda b: b"READY" in b)
    stopped = json.loads(registry.execute("stop_service", {"name": "early"}))
    assert stopped["artifact_output_failed"] is False and "ARTIFACT_OUTPUTS" in stopped["artifact_outputs"]
    artifact = drive / "task_results" / "artifacts" / "task-early-output" / "qualification.html"
    assert artifact.read_text(encoding="utf-8") == "<p>ok</p>"


@pytest.mark.serial
def test_a_local_executor_service_refuses_an_escape_before_launch_and_exports_its_parent_output(local_project):
    registry, active, _outside, _launched = local_project
    site = active / "site"
    body = ("from pathlib import Path; Path('marker.txt').touch(); Path('../qualification.html').write_text('<p>local</p>'); "
            "print('READY', flush=True); import time; time.sleep(30)")
    refused = registry.execute_result("start_service", {
        "name": "local-early", "cmd": [sys.executable, "-c", body], "cwd": str(site),
        "outputs": ["../../outside.html"], "readiness": {"timeout_sec": 1}})
    assert refused.code == "RESOURCE_POLICY_BLOCKED", refused.text
    assert refused.meta.get("operation_outcome") == "completed_no_effect"
    assert not (site / "marker.txt").exists() and not (active.parent / "outside.html").exists()
    assert registry.execute("service_status", {"name": "local-early"}).startswith("⚠️ SERVICE_NOT_FOUND")

    started = json.loads(registry.execute("start_service", {
        "name": "local-early", "cmd": [sys.executable, "-c", body], "cwd": str(site),
        "outputs": ["../qualification.html"], "readiness": {"log_contains": "READY", "timeout_sec": 10}}))
    try:
        assert started["state"] == "running" and started["ready"] is True, started
        assert started["executor"]["type"] == "local", started
        assert pathlib.Path(started["host_cwd"]).resolve() == site.resolve(), started
        assert (site / "marker.txt").exists(), "the service ran in this host cwd"
    finally:
        stopped = json.loads(registry.execute("stop_service", {"name": "local-early"}))
    assert stopped["artifact_output_failed"] is False, stopped.get("artifact_outputs")
    assert _registered(registry, stopped["artifact_outputs"]) == {(active / "qualification.html").resolve(): b"<p>local</p>"}


@pytest.mark.serial
def test_a_baseline_that_could_not_be_taken_is_disclosed_at_start(tmp_path, monkeypatch):
    from ouroboros.tools import shell_outputs

    _force_light_runtime(monkeypatch)
    monkeypatch.setattr("ouroboros.safety.check_safety", lambda *a, **k: (True, ""))
    repo, drive = tmp_path / "repo", tmp_path / "data"
    repo.mkdir()
    registry = ToolRegistry(repo_dir=repo, drive_root=drive)
    registry._ctx.task_id = "task-baseline"
    task_drive = drive / "task_drives" / "task-baseline"
    task_drive.mkdir(parents=True)
    monkeypatch.setattr(shell_outputs, "_snapshot_declared_outputs", lambda *a, **k: (_ for _ in ()).throw(OSError("store offline")))
    marker = task_drive / "started.txt"
    seen = []
    registry._ctx.emit_progress_fn = lambda text: seen.append((text, marker.exists()))
    body = ("from pathlib import Path; Path('started.txt').touch(); Path('page.html').write_text('p'); "
            "print('READY', flush=True); import time; time.sleep(30)")
    started = json.loads(registry.execute("start_service", {
        "name": "baseline", "cmd": [sys.executable, "-c", body], "cwd": str(task_drive),
        "outputs": ["page.html"], "readiness": {"log_contains": "READY", "timeout_sec": 5}}))
    try:
        assert started["state"] == "running"
        assert started["output_baseline_unavailable"] == "OSError: store offline"
        warned = [row for row in seen if row[0].startswith("⚠️ Output baseline unavailable: ")]
        assert warned and warned[0][1] is False, "the disclosure precedes the service's first effect"
        assert "store offline" not in warned[0][0]
    finally:
        stopped = json.loads(registry.execute("stop_service", {"name": "baseline"}))
    assert "output baseline was unavailable at start" in stopped["artifact_outputs"]
    assert "registered output" in stopped["artifact_outputs"] and stopped["artifact_output_failed"] is False


@pytest.mark.serial
def test_a_service_output_generated_in_the_system_repo_is_exported_at_stop(tmp_path, monkeypatch):
    from ouroboros.tool_access import build_resolved_resource_binding
    from ouroboros.tools.registry import ToolContext
    from ouroboros.tools.services import _start_service, _stop_service

    _force_advanced_runtime(monkeypatch)
    monkeypatch.setattr("ouroboros.tools.commit_gate._invalidate_advisory", lambda *a, **k: None)
    system, drive = tmp_path / "system", tmp_path / "data"
    system.mkdir()
    drive.mkdir()
    subprocess.run(["git", "init", "-q", str(system)], check=True)
    ctx = ToolContext(repo_dir=system, system_repo_dir=system, drive_root=drive, task_id="system-service")
    binding = build_resolved_resource_binding(ctx, operation="service", process_cwd="system_repo")
    started = json.loads(_start_service(
        ctx, [sys.executable, "-c", "from pathlib import Path; Path('generated.html').write_text('g'); import time; time.sleep(30)"],
        name="generator", cwd="system_repo", outputs=["generated.html"], readiness={"timeout_sec": 0},
        _resolved_binding=binding))
    assert started["cwd_root"] == "system_repo"
    deadline = time.time() + 10
    while not (system / "generated.html").exists() and time.time() < deadline:
        time.sleep(0.05)
    stopped = json.loads(_stop_service(ctx, name="generator"))
    assert stopped["artifact_output_failed"] is False, stopped.get("artifact_outputs")
    assert "registered output" in stopped["artifact_outputs"]


@posix_permissions
@pytest.mark.serial
def test_a_service_output_the_baseline_cannot_read_is_disclosed_at_start_and_stop(tmp_path, monkeypatch):
    _force_light_runtime(monkeypatch)
    monkeypatch.setattr("ouroboros.safety.check_safety", lambda *a, **k: (True, ""))
    repo, drive = tmp_path / "repo", tmp_path / "data"
    repo.mkdir()
    registry = ToolRegistry(repo_dir=repo, drive_root=drive)
    registry._ctx.task_id = "task-locked-baseline"
    task_drive = drive / "task_drives" / "task-locked-baseline"
    task_drive.mkdir(parents=True)
    kept = task_drive / "kept.html"
    kept.write_text("same")
    kept.chmod(0o000)
    marker = task_drive / "started.txt"
    seen = []
    registry._ctx.emit_progress_fn = lambda text: seen.append((text, marker.exists()))
    body = ("import os; from pathlib import Path; Path('started.txt').touch(); os.chmod('kept.html', 0o644); "
            "print('READY', flush=True); import time; time.sleep(30)")
    try:
        started = json.loads(registry.execute("start_service", {
            "name": "locked", "cmd": [sys.executable, "-c", body], "cwd": str(task_drive),
            "outputs": ["kept.html"], "readiness": {"log_contains": "READY", "timeout_sec": 5}}))
        try:
            assert started["state"] == "running"
            assert started["output_baseline_unavailable"].startswith("PermissionError")
            warned = [row for row in seen if row[0].startswith("⚠️ Output baseline unavailable: ")]
            assert warned and warned[0][1] is False, "the disclosure precedes the service's first effect"
        finally:
            stopped = json.loads(registry.execute("stop_service", {"name": "locked"}))
    finally:
        kept.chmod(0o644)
    assert "output baseline was unavailable at start" in stopped["artifact_outputs"]
    assert "registered output" in stopped["artifact_outputs"] and stopped["artifact_output_failed"] is False


@pytest.mark.parametrize("tool", ["run_command", "run_script"])
def test_unexpandable_output_home_refuses_before_effects(project, tool):
    registry, active, _outside = project
    output = "~ouroboros_rv_no_such_user_zz/out.txt"
    try:
        pathlib.Path(output).expanduser()
    except RuntimeError:
        pass
    else:
        pytest.skip("this platform resolves the test's nonexistent home spelling")
    result = _run(registry, tool, active / "site", [output])
    assert result.code == "RESOURCE_POLICY_BLOCKED" and "could not be resolved" in result.text
    assert result.meta.get("operation_outcome") == "completed_no_effect"
    assert not (active / "site" / "marker.txt").exists()


@pytest.mark.parametrize("tool", ["run_command", "run_script"])
@pytest.mark.parametrize("error_type", [RuntimeError, OSError])
def test_late_output_resolution_failure_preserves_completed_command(project, monkeypatch, tool, error_type):
    from ouroboros.tools import shell_outputs

    registry, active, _outside = project
    resolve = shell_outputs._resolve_declared_output

    def resolve_after_work(*args, **kwargs):
        if kwargs.get("judge_changes", True):
            raise error_type("controlled late resolution failure")
        return resolve(*args, **kwargs)

    monkeypatch.setattr(shell_outputs, "_resolve_declared_output", resolve_after_work)
    result = _run(registry, tool, active / "site", ["out.txt"],
                  "Path('out.txt').write_text('done')", "print('COMMAND_COMPLETED')")
    assert result.code == "ARTIFACT_OUTPUT_ERROR", result.text
    assert "COMMAND_COMPLETED" in result.text and "exit_code=0" in result.text
    assert "output could not be resolved" in result.text and "controlled late resolution failure" in result.text
    assert (active / "site" / "marker.txt").exists()
