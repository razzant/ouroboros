from __future__ import annotations

import hashlib
import io
import os
import pathlib
import signal
import subprocess
import sys
import threading
import time
from typing import Any

import pytest

import ouroboros.execd as execd_module
import ouroboros.execd_state as state_module
import ouroboros.workspace_native as native_module
from ouroboros.execd import ExecdProtocolServer, ExecdService
from ouroboros.execd_state import (
    CASBlobStore,
    ExecdError,
    LeaseCustody,
    OperationJournal,
    continuity_host_id,
    initialize_continuity_host_id,
    read_json,
)
from ouroboros.remote_contracts import CONTRACT_SET_VERSION
from ouroboros.remote_protocol import (
    MAX_CONTROL_BYTES,
    MAX_JSON_STRING_BYTES,
    MAX_LEASE_TTL_MS,
    ProtocolError,
    encode_control,
    read_frame,
)
from ouroboros.workspace_diagnostics import ToolExecutionEnvelope
from ouroboros.workspace_native import (
    MANDATORY_REMOTE_NATIVE_OPERATIONS,
    NativeOperationResult,
    execute_native_operation,
)


def _capability_manifest() -> dict[str, Any]:
    return {
        "manifest_sha256": "a" * 64,
        "native_operations": sorted(MANDATORY_REMOTE_NATIVE_OPERATIONS),
    }


def _git_workspace(path: pathlib.Path) -> pathlib.Path:
    path.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=path, check=True)
    subprocess.run(
        ["git", "config", "user.email", "execd-tests@example.invalid"],
        cwd=path,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Execd Tests"],
        cwd=path,
        check=True,
    )
    (path / "README.md").write_text("remote-only\n", encoding="utf-8")
    subprocess.run(["git", "add", "README.md"], cwd=path, check=True)
    subprocess.run(["git", "commit", "-qm", "fixture"], cwd=path, check=True)
    return path


def _service(
    tmp_path: pathlib.Path,
    *,
    generation: str = "generation-a",
    connection_id: str = "connection-a",
    project_id: str = "project-a",
) -> ExecdService:
    workspace = tmp_path / "workspace"
    if not workspace.exists():
        _git_workspace(workspace)
    initialize_continuity_host_id(tmp_path / "state")
    return ExecdService(
        tmp_path / "state",
        workspace,
        connection_id=connection_id,
        project_id=project_id,
        server_generation=generation,
        release_id="test-release",
        artifact_sha256="f" * 64,
        capability_manifest=_capability_manifest(),
    )


def test_continuity_identity_read_is_non_mutating_and_bootstrap_is_explicit(
    tmp_path,
    capsys,
):
    state_root = tmp_path / "state"

    with pytest.raises(ExecdError) as missing:
        continuity_host_id(state_root)
    assert missing.value.code == "host_identity_missing"
    assert not state_root.exists()

    with pytest.raises(ExecdError) as cli_missing:
        execd_module._main([
            "--state-root",
            str(state_root),
            "--print-host-id",
        ])
    assert cli_missing.value.code == "host_identity_missing"
    assert not state_root.exists()

    assert execd_module._main([
        "--state-root",
        str(state_root),
        "--initialize-host-id",
    ]) == 0
    initialized = capsys.readouterr().out.strip()
    assert initialized == continuity_host_id(state_root)

    assert execd_module._main([
        "--state-root",
        str(state_root),
        "--print-host-id",
    ]) == 0
    assert capsys.readouterr().out.strip() == initialized
    assert initialize_continuity_host_id(state_root) == initialized


@pytest.mark.parametrize(
    ("release_id", "artifact_sha256", "code"),
    [
        ("", "f" * 64, "release_identity_invalid"),
        ("../mutable", "f" * 64, "release_identity_invalid"),
        ("release-a", "F" * 64, "artifact_identity_invalid"),
        ("release-a", "f" * 63, "artifact_identity_invalid"),
    ],
)
def test_release_attestation_is_strict(release_id, artifact_sha256, code):
    with pytest.raises(ExecdError) as invalid:
        state_module.release_attestation(release_id, artifact_sha256)
    assert invalid.value.code == code


def test_handshake_attests_exact_release_and_prepared_binding(tmp_path):
    service = _service(tmp_path)
    writer = io.BytesIO()
    server = ExecdProtocolServer(service, io.BytesIO(), writer)

    # `protocol_minor` IS the Home↔execd contract set (`remote_contracts`), and the
    # handshake is now admitted against it, so a synthesized frame has to carry the
    # field a real one always did — `validate_control_message` requires it.
    server._receive_control(
        {"kind": "handshake", "protocol_minor": CONTRACT_SET_VERSION}
    )
    label, response = read_frame(io.BytesIO(writer.getvalue()))

    assert label == "control"
    assert response["kind"] == "handshake_ok"
    assert response["optional"]["artifact"] == {
        "release_id": "test-release",
        "sha256": "f" * 64,
    }
    service.prepare(
        request_id="request-release",
        operation_id="operation-release",
        tool="read_file",
        args={"path": "README.md"},
    )
    binding = service._prepared[
        ("request-release", "operation-release")
    ].prepared
    assert binding["release_id"] == "test-release"
    assert binding["artifact_sha256"] == "f" * 64


def test_execd_admission_supports_pre_2_5_git_facts(tmp_path, monkeypatch):
    workspace = _git_workspace(tmp_path / "workspace")
    real_run = subprocess.run
    observed: list[list[str]] = []

    def legacy_run(command, *args, **kwargs):
        argv = [str(item) for item in command]
        observed.append(argv)
        if argv == ["git", "rev-parse", "--git-common-dir"]:
            return subprocess.CompletedProcess(argv, 0, "--git-common-dir\n", "")
        if argv == ["git", "rev-parse", "--git-path", "index"]:
            return subprocess.CompletedProcess(argv, 0, b"--git-path\nindex\n", b"")
        if argv == [
            "git",
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
            "--ignore-submodules=none",
        ]:
            return subprocess.CompletedProcess(argv, 129, b"", b"unsupported option")
        return real_run(command, *args, **kwargs)

    monkeypatch.setattr(execd_module.subprocess, "run", legacy_run)
    initialize_continuity_host_id(tmp_path / "state")
    service = ExecdService(
        tmp_path / "state",
        workspace,
        connection_id="connection-a",
        project_id="project-a",
        server_generation="generation-a",
        release_id="test-release",
        artifact_sha256="f" * 64,
        capability_manifest=_capability_manifest(),
    )

    assert service.git_facts["common_dir"] == str(workspace / ".git")
    assert service.git_facts["index_present"] is True
    assert all("-C" not in argv for argv in observed)
    assert [
        "git",
        "-c",
        "diff.ignoreSubmodules=none",
        "status",
        "--porcelain",
        "--untracked-files=all",
    ] in observed


def test_continue_revalidates_target_facts_before_journal_or_effect(tmp_path):
    service = _service(tmp_path)
    first = service.workspace_root / "first"
    second = service.workspace_root / "second"
    first.mkdir()
    second.mkdir()
    selected = service.workspace_root / "selected"
    selected.symlink_to(first, target_is_directory=True)
    prepared = service.prepare(
        request_id="request-target",
        operation_id="operation-target",
        tool="run_command",
        args={
            "cmd": [sys.executable, "-c", "open('effect', 'w').write('ran')"],
            "cwd": "selected",
        },
        task_id="task-target",
    )
    selected.unlink()
    selected.symlink_to(second, target_is_directory=True)

    with pytest.raises(ExecdError) as changed:
        service.continue_prepared(
            request_id="request-target",
            operation_id="operation-target",
            prepared_hash=prepared["prepared_hash"],
            prepared_token=prepared["prepared_token"],
        )

    assert changed.value.code == "prepared_target_changed"
    assert changed.value.phase == "authorize"
    assert service.journal.list_records() == []
    assert not (first / "effect").exists()
    assert not (second / "effect").exists()


def test_reviewed_payload_prepare_and_revalidation_receive_staged_blobs(
    tmp_path,
    monkeypatch,
):
    operation = "execute_reviewed_payload"
    payload = b"print('reviewed payload')\n"
    digest = hashlib.sha256(payload).hexdigest()
    content_hash = hashlib.sha256(
        b"main.py\0" + bytes.fromhex(digest)
    ).hexdigest()
    service = _service(tmp_path)
    executed: list[dict[str, bytes]] = []

    def execute(*_args, blobs=None, **_kwargs):
        executed.append(dict(blobs or {}))
        return NativeOperationResult(ToolExecutionEnvelope(text="ok"))

    monkeypatch.setattr(execd_module, "execute_native_operation", execute)
    prepared = service.prepare(
        request_id="request-reviewed",
        operation_id="operation-reviewed",
        tool=operation,
        args={
            "schema_version": 1,
            "kind": "script",
            "payload": {
                "content_hash": content_hash,
                "skill_name": "reviewed",
                "runtime": "python3",
                "files": [{
                    "path": "main.py",
                    "sha256": digest,
                    "size": len(payload),
                    "mode": 0o600,
                }],
            },
            "invocation": {
                "entry": "main.py",
                "argv": [],
                "timeout_sec": 10,
            },
        },
        task_id="task-reviewed",
        blobs={digest: payload},
    )
    result = service.continue_prepared(
        request_id="request-reviewed",
        operation_id="operation-reviewed",
        prepared_hash=prepared["prepared_hash"],
        prepared_token=prepared["prepared_token"],
    )

    assert result["completion"] == "completed"
    assert prepared["native_facts"]["payload_content_hash"] == content_hash
    assert executed == [{digest: payload}]

def _journal(tmp_path: pathlib.Path) -> OperationJournal:
    blobs = CASBlobStore(tmp_path / "blobs")
    return OperationJournal(
        tmp_path / "operations",
        connection_id="connection-a",
        workspace_id="workspace-a",
        spool=CASBlobStore(tmp_path / "spool"),
        blobs=blobs,
    )




def _process_group_exists(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _wait_group_gone(pgid: int, timeout: float = 3.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not _process_group_exists(pgid):
            return True
        time.sleep(0.05)
    return not _process_group_exists(pgid)


def test_new_custody_state_is_durable_before_custodian_spawn(tmp_path, monkeypatch):
    service = _service(tmp_path)
    observed: dict[str, Any] = {}

    class _FakeProcess:
        def poll(self):
            return None

        def terminate(self):
            return None

    def fake_popen(command, **kwargs):
        del kwargs
        state_path = pathlib.Path(command[command.index("--custodian") + 1])
        observed["command"] = list(command)
        observed["state"] = read_json(state_path, required=True)
        return _FakeProcess()

    monkeypatch.setattr(execd_module.subprocess, "Popen", fake_popen)
    process = service._spawn_custodian()

    assert process.poll() is None
    assert observed["state"]["server_generation"] == "generation-a"
    assert observed["state"]["groups"] == []
    # The deadline is on the target's BOOT-ANCHORED MONOTONIC clock, not its wall
    # clock (a wall-clock step must not move the 15s bound), so it is compared
    # against that scale and the stored anchor is what dates it.
    from ouroboros.platform_layer import boot_anchored_monotonic_ms

    anchor, now_ms = boot_anchored_monotonic_ms()
    assert observed["state"]["clock_anchor"] == anchor
    assert observed["state"]["server_expiry_ms"] > now_ms
    assert observed["state"]["custodian_id"]
    assert observed["state"]["custodian_close_requested"] is False
    assert observed["command"][-2:] == [
        "--custodian-id",
        observed["state"]["custodian_id"],
    ]


def test_service_close_durably_stops_exact_custodian(tmp_path, monkeypatch):
    service = _service(tmp_path)

    class _FakeProcess:
        def __init__(self):
            self.terminated = False

        def poll(self):
            return None

        def terminate(self):
            self.terminated = True

    process = _FakeProcess()
    monkeypatch.setattr(
        execd_module.subprocess,
        "Popen",
        lambda *_args, **_kwargs: process,
    )
    service._custodian_process = service._spawn_custodian()
    identity = service._custodian_id

    service.close(kill_owned=True)

    state = service.custody.refresh_snapshot()
    assert state["custodian_id"] == identity
    assert state["custodian_close_requested"] is True
    assert state["server_expiry_ms"] == 0
    assert process.terminated is True
    with pytest.raises(ExecdError) as closing:
        service.renew_lease(10_000)
    assert closing.value.code == "generation_closing"


def _lease_frames(service, **fields) -> list[dict[str, Any]]:
    """Send one lease frame through the real protocol server; return what came back."""

    writer = io.BytesIO()
    server = ExecdProtocolServer(service, io.BytesIO(), writer)
    server._receive_control({"kind": "lease", "seq": 7, **fields})
    stream = io.BytesIO(writer.getvalue())
    frames: list[dict[str, Any]] = []
    while stream.tell() < len(writer.getvalue()):
        label, message = read_frame(stream)
        assert label == "control"
        frames.append(message)
    return frames


def test_a_lease_is_answered_so_the_generation_bound_can_be_observed(tmp_path):
    """Renewal is answered both ways, and the two answers are distinguishable.

    Without an answer the constitutional bound (§3.4: the custodian ends remote work
    within 15s of the last renewal it accepted) has no representation on the wire —
    neither Home nor a test can tell an accepted lease from a dropped frame, so the
    invariant is unobservable and therefore unprovable.
    """

    service = _service(tmp_path, generation="generation-a")

    honored = _lease_frames(
        service,
        server_generation="generation-a",
        lease_id="lease-good",
        ttl_ms=10_000,
        task_id="task-a",
    )

    assert [row["kind"] for row in honored] == ["ack"]
    assert honored[0]["ack_seq"] == 7
    assert honored[0]["optional"]["lease"] == {
        "lease_id": "lease-good",
        "server_generation": "generation-a",
    }
    assert service.custody.refresh_snapshot()["task_expiry_ms"]["task-a"] > 0

    refused = _lease_frames(
        service,
        server_generation="generation-b",
        lease_id="lease-stale",
        ttl_ms=10_000,
        task_id="task-b",
    )

    assert [row["kind"] for row in refused] == ["diagnostic"]
    assert refused[0]["optional"]["lease"] == {"lease_id": "lease-stale"}
    assert refused[0]["diagnostic"]["code"] == "lease_generation_mismatch"
    assert refused[0]["diagnostic"]["phase"] == "authorize"
    assert refused[0]["diagnostic"]["completion"] == "not_started"
    # A foreign generation touched nothing: no task lease was created for it.
    assert "task-b" not in service.custody.refresh_snapshot()["task_expiry_ms"]


def test_a_lease_refusal_names_its_own_reason_not_a_shared_one(tmp_path):
    """`lease_generation_mismatch` is not the code for a closing own generation."""

    service = _service(tmp_path, generation="generation-a")
    service.custody.kill_generation()

    own_but_closing = _lease_frames(
        service,
        server_generation="generation-a",
        lease_id="lease-own",
        ttl_ms=10_000,
    )

    assert own_but_closing[0]["diagnostic"]["code"] == "generation_closing"


def test_frozen_execd_reenters_itself_without_python_module_mode(tmp_path, monkeypatch):
    service = _service(tmp_path)
    commands: list[list[str]] = []

    class _FakeProcess:
        def poll(self):
            return None

    monkeypatch.delenv("OUROBOROS_EXECD_SELF", raising=False)
    monkeypatch.setattr(execd_module.sys, "frozen", True, raising=False)
    monkeypatch.setattr(execd_module.sys, "executable", "/opt/exe/ouroboros-execd")
    monkeypatch.setattr(
        execd_module.subprocess,
        "Popen",
        lambda command, **kwargs: (
            commands.append(list(command)) or _FakeProcess()
        ),
    )

    service._spawn_custodian()

    assert commands
    assert commands[0][0] == "/opt/exe/ouroboros-execd"
    assert commands[0][1] == "--custodian"
    assert "-m" not in commands[0]


def test_configured_execd_self_is_the_bundle_reentry_authority(tmp_path, monkeypatch):
    service = _service(tmp_path)
    commands: list[list[str]] = []

    class _FakeProcess:
        def poll(self):
            return None

    monkeypatch.setenv("OUROBOROS_EXECD_SELF", "/bundle/current/ouroboros-execd")
    monkeypatch.setattr(
        execd_module.subprocess,
        "Popen",
        lambda command, **kwargs: (
            commands.append(list(command)) or _FakeProcess()
        ),
    )

    service._spawn_custodian()

    assert commands[0][:2] == [
        "/bundle/current/ouroboros-execd",
        "--custodian",
    ]


@pytest.mark.serial
def test_custodian_survives_empty_expired_lease_until_explicit_close(tmp_path):
    custody = LeaseCustody(tmp_path / "custody.json", "generation-a")
    identity = custody.claim_custodian()
    custody.renew(ttl_ms=50)
    outcomes: list[int] = []
    thread = threading.Thread(
        target=lambda: outcomes.append(
            state_module.run_custodian(
                custody.state_path,
                "generation-a",
                identity,
            )
        ),
        daemon=True,
    )
    thread.start()

    time.sleep(0.35)
    assert thread.is_alive()
    assert custody.refresh_snapshot()["groups"] == []

    assert custody.request_custodian_close(identity) is True
    thread.join(timeout=1)
    assert not thread.is_alive()
    assert outcomes == [0]


@pytest.mark.serial
def test_replacement_custodian_waits_for_previous_identity_to_close(tmp_path):
    custody = LeaseCustody(tmp_path / "custody.json", "generation-a")
    first = custody.claim_custodian()
    outcomes: list[int] = []
    thread = threading.Thread(
        target=lambda: outcomes.append(
            state_module.run_custodian(
                custody.state_path,
                "generation-a",
                first,
            )
        ),
        daemon=True,
    )
    thread.start()
    with pytest.raises(ExecdError) as active:
        custody.claim_custodian()
    assert active.value.code == "generation_active"
    assert custody.request_custodian_close(first) is True
    second = custody.claim_custodian()

    thread.join(timeout=1)
    assert not thread.is_alive()
    assert outcomes == [0]
    assert custody.refresh_snapshot()["custodian_id"] == second
    assert custody.request_custodian_close(first) is False


def test_custody_refuses_processes_without_required_live_leases(tmp_path, monkeypatch):
    fingerprint = {
        "boot_id": "boot",
        "pid_namespace": "pid:[1]",
        "leader_pid": 12345,
        "pgrp": 12345,
        "session": 12345,
        "start_ticks": 1,
    }
    monkeypatch.setattr(state_module, "_process_fingerprint", lambda _pid: fingerprint)
    custody = LeaseCustody(tmp_path / "custody.json", "generation-a")

    with pytest.raises(ExecdError) as no_generation:
        custody.register(
            pgid=12345,
            task_id="task-a",
            keep_alive=True,
            service_id="service-a",
        )
    assert no_generation.value.code == "server_lease_expired"
    assert custody.snapshot()["groups"] == []

    custody.renew(ttl_ms=10_000)
    with pytest.raises(ExecdError) as no_task:
        custody.register(
            pgid=12345,
            task_id="task-a",
            keep_alive=False,
            service_id="",
        )
    assert no_task.value.code == "task_lease_expired"
    assert custody.snapshot()["groups"] == []

    custody.register(
        pgid=12345,
        task_id="task-a",
        keep_alive=True,
        service_id="service-a",
    )
    assert custody.snapshot()["groups"][0]["service_id"] == "service-a"


def test_failed_group_kill_retains_durable_authority_for_retry(tmp_path, monkeypatch):
    fingerprint = {
        "boot_id": "boot",
        "pid_namespace": "pid:[1]",
        "leader_pid": 12345,
        "pgrp": 12345,
        "session": 12345,
        "start_ticks": 1,
    }
    monkeypatch.setattr(state_module, "_process_fingerprint", lambda _pid: fingerprint)
    custody = LeaseCustody(tmp_path / "custody.json", "generation-a")
    custody.renew(ttl_ms=10_000, task_id="task-a")
    custody.register(
        pgid=12345,
        task_id="task-a",
        keep_alive=False,
        service_id="",
    )
    attempts = 0

    def flaky_group_kill(pgid, *, checked=False):
        nonlocal attempts
        assert pgid == 12345
        assert checked is True
        attempts += 1
        if attempts == 1:
            raise PermissionError("temporarily denied")
        return True

    monkeypatch.setattr(
        state_module,
        "kill_process_group_id",
        flaky_group_kill,
    )

    assert custody.cancel_task("task-a") == 0
    assert [row["pgid"] for row in custody.snapshot()["groups"]] == [12345]
    assert LeaseCustody(
        tmp_path / "custody.json", "generation-a"
    ).snapshot()["groups"][0]["pgid"] == 12345

    assert custody.kill_generation() == 1
    assert custody.snapshot()["groups"] == []


def test_release_and_service_identity_survive_custody_reopen(tmp_path, monkeypatch):
    fingerprint = {
        "boot_id": "boot",
        "pid_namespace": "pid:[1]",
        "leader_pid": 12345,
        "pgrp": 12345,
        "session": 12345,
        "start_ticks": 1,
    }
    monkeypatch.setattr(state_module, "_process_fingerprint", lambda _pid: fingerprint)
    path = tmp_path / "custody.json"
    custody = LeaseCustody(path, "generation-a")
    custody.renew(ttl_ms=10_000)
    custody.register(
        pgid=12345,
        task_id="task-a",
        keep_alive=True,
        service_id="service-a",
    )

    reopened = LeaseCustody(path, "generation-a")
    recovered = reopened.recover_service(service_id="service-a", task_id="task-a")
    assert recovered is not None
    assert recovered["pgid"] == 12345

    with pytest.raises(ExecdError):
        reopened.release(pgid=12345, service_id="service-other")
    reopened.release(pgid=12345, service_id="service-a")
    assert LeaseCustody(path, "generation-a").snapshot()["groups"] == []


def test_custody_fingerprint_mismatch_prunes_without_signalling(tmp_path, monkeypatch):
    recorded = {
        "boot_id": "boot-a",
        "pid_namespace": "pid:[1]",
        "leader_pid": 12345,
        "pgrp": 12345,
        "session": 12345,
        "start_ticks": 10,
    }
    current = {**recorded, "start_ticks": 11}
    monkeypatch.setattr(state_module, "_process_fingerprint", lambda _pid: recorded)
    custody = LeaseCustody(tmp_path / "custody.json", "generation-a")
    custody.renew(ttl_ms=10_000)
    custody.register(
        pgid=12345,
        task_id="task-a",
        keep_alive=True,
        service_id="service-a",
    )
    monkeypatch.setattr(state_module, "_process_fingerprint", lambda _pid: current)
    calls: list[int] = []
    monkeypatch.setattr(
        state_module,
        "kill_process_group_id",
        lambda pgid, *, checked=False: calls.append(pgid) or checked,
    )

    assert custody.kill_generation() == 1
    assert calls == []
    assert custody.refresh_snapshot()["groups"] == []


def test_zero_generation_expiry_retries_retained_keepalive_kill(tmp_path, monkeypatch):
    fingerprint = {
        "boot_id": "boot",
        "pid_namespace": "pid:[1]",
        "leader_pid": 12345,
        "pgrp": 12345,
        "session": 12345,
        "start_ticks": 1,
    }
    monkeypatch.setattr(state_module, "_process_fingerprint", lambda _pid: fingerprint)
    custody = LeaseCustody(tmp_path / "custody.json", "generation-a")
    custody.renew(ttl_ms=10_000)
    custody.register(
        pgid=12345,
        task_id="task-a",
        keep_alive=True,
        service_id="service-a",
    )
    attempts = 0

    def flaky(pgid, *, checked=False):
        nonlocal attempts
        assert checked is True
        attempts += 1
        if attempts == 1:
            raise PermissionError("transient")
        return True

    monkeypatch.setattr(state_module, "kill_process_group_id", flaky)
    assert custody.kill_generation() == 0
    assert custody.refresh_snapshot()["groups"]
    assert custody.expire() == 1
    assert attempts == 2
    assert custody.refresh_snapshot()["groups"] == []


def test_generation_mismatch_cannot_adopt_another_generation_state(tmp_path):
    path = tmp_path / "custody.json"
    LeaseCustody(path, "generation-a")

    with pytest.raises(ExecdError) as mismatch:
        LeaseCustody(path, "generation-b")

    assert mismatch.value.code == "custody_state_mismatch"
    assert read_json(path, required=True)["server_generation"] == "generation-a"


def test_panic_kill_isolated_to_exact_server_generation(tmp_path, monkeypatch):
    def fingerprint(pid):
        return {
            "boot_id": "boot",
            "pid_namespace": "pid:[1]",
            "leader_pid": pid,
            "pgrp": pid,
            "session": pid,
            "start_ticks": pid * 10,
        }

    monkeypatch.setattr(state_module, "_process_fingerprint", fingerprint)
    killed: list[int] = []
    monkeypatch.setattr(
        state_module.os,
        "killpg",
        lambda pgid, sig: (
            killed.append(pgid)
            if sig == signal.SIGKILL
            else pytest.fail(f"unexpected signal: {sig}")
        ),
    )
    first = LeaseCustody(tmp_path / "generation-a.json", "generation-a")
    second = LeaseCustody(tmp_path / "generation-b.json", "generation-b")
    first.renew(ttl_ms=10_000)
    second.renew(ttl_ms=10_000)
    first.register(
        pgid=11111,
        task_id="task-a",
        keep_alive=True,
        service_id="service-a",
    )
    second.register(
        pgid=22222,
        task_id="task-b",
        keep_alive=True,
        service_id="service-b",
    )

    assert first.kill_generation() == 1

    assert killed == [11111]
    assert first.refresh_snapshot()["groups"] == []
    assert [row["pgid"] for row in second.refresh_snapshot()["groups"]] == [
        22222
    ]




























@pytest.mark.serial
def test_native_registration_failure_kills_the_newborn_process_group(tmp_path):
    class _RejectCustody:
        pgid = 0

        def cancelled(self):
            return False

        def register_process(self, *, pgid, **kwargs):
            del kwargs
            self.pgid = pgid
            raise OSError("custody ledger unavailable")

        def release_process(self, **kwargs):
            del kwargs

        def recover_service(self, **kwargs):
            del kwargs
            return None

    control = _RejectCustody()
    result = execute_native_operation(
        tmp_path,
        "run_command",
        {
            "cmd": [sys.executable, "-c", "import time; time.sleep(60)"],
            "cwd": str(tmp_path),
            "timeout_sec": 60,
        },
        control=control,
    )

    assert control.pgid > 0
    assert result.envelope.diagnostic is not None
    assert "custody ledger unavailable" in result.envelope.diagnostic.message
    assert _wait_group_gone(control.pgid)


def test_protocol_eof_and_panic_both_close_owned_groups_without_ack():
    class _Service:
        def __init__(self):
            self.close_calls: list[bool] = []

        def close(self, *, kill_owned=True):
            self.close_calls.append(kill_owned)

    service = _Service()
    reader = __import__("io").BytesIO()
    writer = __import__("io").BytesIO()
    server = ExecdProtocolServer(service, reader, writer)

    server.serve()
    assert service.close_calls == [True]
    assert writer.getvalue() == b""

    service.close_calls.clear()
    server._receive_control(
        {
            "kind": "panic",
            "seq": 0,
            "server_generation": "generation-a",
        }
    )
    assert service.close_calls == [True]
    assert writer.getvalue() == b""


@pytest.mark.serial
def test_protocol_panic_exits_control_loop_without_waiting_for_transport_eof():
    class _Service:
        def __init__(self):
            self.closed = threading.Event()

        def close(self, *, kill_owned=True):
            assert kill_owned is True
            self.closed.set()

    read_fd, write_fd = os.pipe()
    reader = os.fdopen(read_fd, "rb", buffering=0)
    transport_writer = os.fdopen(write_fd, "wb", buffering=0)
    protocol_output = __import__("io").BytesIO()
    service = _Service()
    server = ExecdProtocolServer(service, reader, protocol_output)
    thread = threading.Thread(target=server.serve, daemon=True)
    thread.start()
    try:
        transport_writer.write(
            encode_control(
                {
                    "kind": "panic",
                    "seq": 0,
                    "server_generation": "generation-a",
                }
            )
        )
        transport_writer.flush()

        assert service.closed.wait(timeout=1)
        thread.join(timeout=1)
        assert not thread.is_alive()
        assert protocol_output.getvalue() == b""
    finally:
        transport_writer.close()
        reader.close()
        thread.join(timeout=2)


def test_completed_operation_reconciles_after_new_execd_instance(tmp_path):
    first = _service(tmp_path)
    prepared = first.prepare(
        request_id="request-a",
        operation_id="operation-a",
        tool="read_file",
        args={"path": "README.md"},
        task_id="task-a",
    )
    result = first.continue_prepared(
        request_id="request-a",
        operation_id="operation-a",
        prepared_hash=prepared["prepared_hash"],
        prepared_token=prepared["prepared_token"],
    )
    assert result["completion"] == "completed"

    second = _service(tmp_path)
    reconciled = second.reconcile(
        "request-a",
        "operation-a",
        prepared["prepared_hash"],
    )

    assert reconciled["completion"] == "completed"
    assert reconciled["result"]["prepared_hash"] == prepared["prepared_hash"]


@pytest.mark.serial
def test_keepalive_service_is_recovered_by_new_execd_instance(tmp_path):
    first = _service(tmp_path)
    first.renew_lease(10_000, "task-a")
    prepared = first.prepare(
        request_id="request-start",
        operation_id="operation-start",
        tool="start_service",
        args={
            "name": "worker",
            "cmd": [
                sys.executable,
                "-c",
                (
                    "from pathlib import Path; import time; "
                    "Path('service-output.bin').write_bytes(b'\\x00remote-output'); "
                    "print('ready', flush=True); time.sleep(60)"
                ),
            ],
            "cwd": str(first.workspace_root),
            "keep_alive": True,
            "readiness": {"stdout_contains": "ready", "timeout_sec": 5},
            "outputs": ["service-output.bin"],
        },
        task_id="task-a",
    )
    started = first.continue_prepared(
        request_id="request-start",
        operation_id="operation-start",
        prepared_hash=prepared["prepared_hash"],
        prepared_token=prepared["prepared_token"],
    )
    service_ref = started["envelope"]["trace"]["service_ref"]
    native_module._SERVICES_BY_ID.clear()
    native_module._SERVICES_BY_TASK_NAME.clear()

    second = _service(tmp_path)
    status_prepared = second.prepare(
        request_id="request-status",
        operation_id="operation-status",
        tool="service_status",
        args={"name": "worker", "_service_ref": service_ref},
        task_id="task-a",
    )
    status = second.continue_prepared(
        request_id="request-status",
        operation_id="operation-status",
        prepared_hash=status_prepared["prepared_hash"],
        prepared_token=status_prepared["prepared_token"],
    )

    assert status["envelope"]["trace"]["running"] is True
    observed_ref = status["envelope"]["trace"]["service_ref"]
    assert {
        key: observed_ref[key] for key in ("kind", "service_id", "name")
    } == {
        key: service_ref[key] for key in ("kind", "service_id", "name")
    }
    assert status["envelope"]["trace"]["keep_alive"] is True
    assert status["envelope"]["trace"]["ready"] is True
    assert status["envelope"]["trace"]["outputs"] == ["service-output.bin"]

    stop_prepared = second.prepare(
        request_id="request-stop",
        operation_id="operation-stop",
        tool="stop_service",
        args={"name": "worker", "_service_ref": service_ref},
        task_id="task-a",
    )
    stopped = second.continue_prepared(
        request_id="request-stop",
        operation_id="operation-stop",
        prepared_hash=stop_prepared["prepared_hash"],
        prepared_token=stop_prepared["prepared_token"],
    )
    output = b"\x00remote-output"
    digest = hashlib.sha256(output).hexdigest()
    assert stopped["output_blobs"] == {digest: digest}
    assert second.cas.read(digest, max_bytes=len(output)) == output
    native_module._SERVICES_BY_ID.clear()
    native_module._SERVICES_BY_TASK_NAME.clear()


def test_protocol_ack_marks_task_bound_completed_operation(tmp_path):
    service = _service(tmp_path)
    prepared = service.prepare(
        request_id="request-a",
        operation_id="operation-a",
        tool="read_file",
        args={"path": "README.md"},
        task_id="task-a",
    )
    service.continue_prepared(
        request_id="request-a",
        operation_id="operation-a",
        prepared_hash=prepared["prepared_hash"],
        prepared_token=prepared["prepared_token"],
    )
    server = ExecdProtocolServer(
        service,
        __import__("io").BytesIO(),
        __import__("io").BytesIO(),
    )

    server._receive_control(
        {
            "kind": "ack",
            "seq": 0,
            "request_id": "request-a",
            "operation_id": "operation-a",
            "optional": {"prepared_hash": prepared["prepared_hash"]},
        }
    )

    records = service.journal.list_records()
    assert len(records) == 1
    assert records[0]["acked"] is True


def test_second_blob_manifest_is_rejected_while_upload_is_active(tmp_path):
    service = _service(tmp_path)
    server = ExecdProtocolServer(
        service,
        __import__("io").BytesIO(),
        __import__("io").BytesIO(),
    )
    first = {
        "kind": "blob_manifest",
        "seq": 0,
        "request_id": "request-a",
        "operation_id": "operation-a",
        "blob_id": "blob-a",
        "size": 10,
        "sha256": hashlib.sha256(b"a" * 10).hexdigest(),
    }
    second = {
        "kind": "blob_manifest",
        "seq": 1,
        "request_id": "request-b",
        "operation_id": "operation-b",
        "blob_id": "blob-b",
        "size": 5,
        "sha256": hashlib.sha256(b"b" * 5).hexdigest(),
    }

    server._receive_control(first)
    with pytest.raises(ProtocolError):
        server._receive_control(second)

    assert server._incoming_blob is not None
    assert server._incoming_blob["request_id"] == "request-a"
    assert bytes(server._incoming_blob["data"]) == b""


def test_oversized_native_result_is_spooled_before_result_frame_and_reconciles(
    tmp_path,
):
    service = _service(tmp_path)
    payload = "x" * (MAX_CONTROL_BYTES * 2)
    (service.workspace_root / "huge.txt").write_text(payload, encoding="utf-8")
    prepared = service.prepare(
        request_id="request-huge",
        operation_id="operation-huge",
        tool="read_file",
        args={"path": "huge.txt", "max_lines": 1},
        task_id="task-huge",
    )
    writer = __import__("io").BytesIO()
    server = ExecdProtocolServer(
        service,
        __import__("io").BytesIO(),
        writer,
    )

    server._continue_and_send(
        {
            "kind": "continue",
            "seq": 0,
            "request_id": "request-huge",
            "operation_id": "operation-huge",
            "prepared_hash": prepared["prepared_hash"],
            "optional": {"prepared_token": prepared["prepared_token"]},
        }
    )

    frame = writer.getvalue()
    assert len(frame) <= MAX_CONTROL_BYTES + 5
    label, message = read_frame(__import__("io").BytesIO(frame))
    assert label == "control"
    assert message["kind"] == "result"
    assert message["completion"] == "completed"
    result = message["result"]
    envelope = result["envelope"]
    assert len(envelope["text"].encode("utf-8")) <= MAX_JSON_STRING_BYTES
    references = list(result.get("output_blobs") or {})
    references.extend(
        str(row.get("blob_id") or "")
        for row in envelope.get("artifacts") or []
        if isinstance(row, dict)
    )
    assert any(reference for reference in references)

    reconciled = service.reconcile(
        "request-huge",
        "operation-huge",
        prepared["prepared_hash"],
    )
    assert reconciled["completion"] == "completed"
    assert reconciled["result_unavailable"] is False


def test_custodian_fence_rejects_overlap_and_stale_generation_kill(
    tmp_path,
    monkeypatch,
):
    def fingerprint(pid):
        return {
            "boot_id": "boot",
            "pid_namespace": "pid:[1]",
            "leader_pid": pid,
            "pgrp": pid,
            "session": pid,
            "start_ticks": pid * 10,
        }

    monkeypatch.setattr(state_module, "_process_fingerprint", fingerprint)
    monkeypatch.setattr(
        state_module,
        "kill_process_group_id",
        lambda _pgid, *, checked=False: checked,
    )
    path = tmp_path / "custody.json"
    first = LeaseCustody(path, "generation-a")
    first_id = first.claim_custodian()
    first.renew(ttl_ms=10_000)
    first.register(
        pgid=11111,
        task_id="task-a",
        keep_alive=True,
        service_id="service-a",
    )
    second = LeaseCustody(path, "generation-a")
    with pytest.raises(ExecdError) as active:
        second.claim_custodian()
    assert active.value.code == "generation_active"

    assert first.kill_generation(first_id) == 1
    second_id = second.claim_custodian()
    second.renew(ttl_ms=10_000)
    second.register(
        pgid=22222,
        task_id="task-b",
        keep_alive=True,
        service_id="service-b",
    )

    assert first.kill_generation(first_id) == 0
    snapshot = second.refresh_snapshot()
    assert snapshot["custodian_id"] == second_id
    assert [row["pgid"] for row in snapshot["groups"]] == [22222]



# NOTE (RWS v2 Lane 1, stage A): this file is the donor's tests/test_execd_state.py
# with the tests that exercise NOT-YET-TRANSFERRED stage-B modules deferred to the
# stages that transfer those modules (they return byte-identical with them):
#
#   with remote_ssh.py / scripts/build_execd_bundle.py / scripts/assemble_execd_stage.py:
#     _fake_bundle_stage (helper)
#     test_execd_bundle_builder_is_deterministic_and_dual_arch
#     test_execd_launcher_self_smoke_does_not_mutate_stage
#     test_execd_bundle_builder_rejects_links_and_home_modules
#     test_bootstrap_archive_validator_rejects_duplicate_members
#     test_bootstrap_archive_validator_accepts_empty_and_rejects_digest_mismatch
#     test_musl_platform_fails_before_bundle_upload
#     test_bootstrap_reuses_only_exact_verified_content_addressed_release
#     test_transport_serializes_concurrent_prepare_uploads_without_blocking_cancel
#     test_protocol_ssh_neutralizes_alias_forwarding_but_browser_rejects_it
#     test_ssh_operational_settings_drive_openssh_argv
#
#   with remote_workspace.py (broker) / remote_service_leases.py / remote_finalization.py:
#     test_broker_blocked_session_does_not_block_other_session_or_cancel
#     test_finish_task_forgets_local_lease_even_when_remote_cancel_fails
#     test_close_project_session_does_not_close_sibling_project
#     test_lifecycle_fence_refresh_discards_dead_keepalive
#     test_reconcile_imports_completed_result_then_confirms_ack
#     test_broker_panic_does_not_wait_for_held_state_lock
#     test_result_unavailable_is_fixed_on_home_before_ack
#
# The execd_state/execd/native tests above are byte-identical to the donor's.


# ── the 15s custodian bound lives on a clock nobody can step ──────────────────
#
# The bound is constitutional (BIBLE, Emergency Stop Invariant) and physical: a
# maximum failure-DETECTION time, never a software grace period. It was measured on
# the target's WALL clock — `renew` wrote `time.time()*1000 + ttl` and `expire`
# compared a fresh `time.time()` — so an NTP correction or a manual `date` ON THE
# TARGET fired it early when the clock jumped forward and, worse, held it past the
# ceiling when the clock jumped back. The cross-host half was already right (Home's
# clock never crosses the wire; a TTL travels as a duration); this is the on-target
# half. Deterministic throughout: the clock is injected, nothing sleeps.


def _fake_group(monkeypatch, pgid=4242):
    fingerprint = {
        "boot_id": "boot",
        "pid_namespace": "pid:[1]",
        "leader_pid": pgid,
        "pgrp": pgid,
        "session": pgid,
        "start_ticks": 1,
    }
    monkeypatch.setattr(state_module, "_process_fingerprint", lambda _pid: fingerprint)
    killed: list[int] = []
    monkeypatch.setattr(
        state_module,
        "kill_process_group_id",
        lambda pgid, *, checked=False: killed.append(pgid) or True,
    )
    return pgid, killed


class _InjectedClock:
    """A boot-anchored clock a test can move, independently of the wall clock."""

    def __init__(self, anchor: str = "boot-a", now_ms: int = 1_000_000):
        self.anchor = anchor
        self.now_ms = now_ms

    def __call__(self):
        return self.anchor, self.now_ms


def _custody_on(tmp_path, monkeypatch, clock, name="custody.json"):
    monkeypatch.setattr(state_module, "boot_anchored_monotonic_ms", clock)
    return LeaseCustody(tmp_path / name, "generation-a")


def test_a_wall_clock_step_does_not_move_the_lease_deadline(tmp_path, monkeypatch):
    pgid, killed = _fake_group(monkeypatch)
    clock = _InjectedClock()
    custody = _custody_on(tmp_path, monkeypatch, clock)
    custody.renew(ttl_ms=15_000, task_id="task-a")
    custody.register(pgid=pgid, task_id="task-a", keep_alive=False, service_id="")

    # The wall clock leaps YEARS forward and then back to the epoch. Neither is the
    # clock the deadline is on, so neither may be visible here. The forward step is a
    # genuine regression case: under the wall-clock version it put `now` far past the
    # stored expiry and killed the group on the spot.
    monkeypatch.setattr(state_module.time, "time", lambda: 4_000_000_000.0)
    assert custody.expire() == 0
    assert killed == []
    monkeypatch.setattr(state_module.time, "time", lambda: 1.0)
    assert custody.expire() == 0
    assert killed == []

    # Only the monotonic clock passing the deadline expires it, and it still does so
    # at the fifteen-second bound and not later.
    clock.now_ms += 14_999
    assert custody.expire() == 0
    clock.now_ms += 1
    assert custody.expire() == 1
    assert killed == [pgid]


def test_the_bound_is_at_most_fifteen_seconds_after_the_last_renewal(tmp_path, monkeypatch):
    pgid, killed = _fake_group(monkeypatch)
    clock = _InjectedClock()
    custody = _custody_on(tmp_path, monkeypatch, clock)
    custody.renew(ttl_ms=MAX_LEASE_TTL_MS, task_id="task-a")
    custody.register(pgid=pgid, task_id="task-a", keep_alive=True, service_id="svc")
    # A keep-alive group answers to the GENERATION lease, which is the partition
    # fallback: no renewal for MAX_LEASE_TTL_MS and the custodian completes the kill.
    clock.now_ms += MAX_LEASE_TTL_MS
    assert custody.expire() == 1
    assert killed == [pgid]
    assert MAX_LEASE_TTL_MS == 15_000


def test_a_renewal_extends_from_the_monotonic_now_not_from_the_wall_clock(tmp_path, monkeypatch):
    pgid, killed = _fake_group(monkeypatch)
    clock = _InjectedClock()
    custody = _custody_on(tmp_path, monkeypatch, clock)
    custody.renew(ttl_ms=15_000, task_id="task-a")
    custody.register(pgid=pgid, task_id="task-a", keep_alive=False, service_id="")
    clock.now_ms += 10_000
    custody.renew(ttl_ms=15_000, task_id="task-a")
    clock.now_ms += 14_999
    assert custody.expire() == 0
    clock.now_ms += 1
    assert custody.expire() == 1
    assert killed == [pgid]


def test_a_changed_boot_id_reads_the_lease_as_EXPIRED_not_as_infinite(tmp_path, monkeypatch):
    """The anchor is what makes a stored deadline datable; without it, fail closed.

    A boottime value from a previous boot cannot be compared to this boot's. Believing
    it would turn the 15-second PHYSICAL bound into an unbounded one, which the
    invariant forbids more strongly than it forbids an early kill — so the leases read
    as already expired and the owned groups are killed.
    """
    pgid, killed = _fake_group(monkeypatch)
    clock = _InjectedClock()
    custody = _custody_on(tmp_path, monkeypatch, clock)
    custody.renew(ttl_ms=15_000, task_id="task-a")
    custody.register(pgid=pgid, task_id="task-a", keep_alive=True, service_id="svc")
    assert custody.expire() == 0

    # The host rebooted: same generation file, new boot identity, and a boottime that
    # now reads as EARLIER than the stored deadline — the exact case a bare number
    # would report as "plenty of lease left".
    rebooted = _InjectedClock(anchor="boot-b", now_ms=5_000)
    reopened = _custody_on(tmp_path, monkeypatch, rebooted)
    snapshot = reopened.refresh_snapshot()
    assert snapshot["server_expiry_ms"] == 0
    assert snapshot["task_expiry_ms"] == {}
    assert reopened.expire() == 1
    assert killed == [pgid]


def test_the_deadline_crosses_the_process_boundary_through_the_state_file(tmp_path, monkeypatch):
    """execd writes the deadline; an independent custodian process reads it.

    That is why the scale has to be shared across processes and not per-process: the
    two would otherwise disagree about the same instant, and the watchdog would judge
    execd's deadline against its own epoch.
    """
    pgid, killed = _fake_group(monkeypatch)
    clock = _InjectedClock()
    writer = _custody_on(tmp_path, monkeypatch, clock)
    writer.renew(ttl_ms=15_000, task_id="task-a")
    writer.register(pgid=pgid, task_id="task-a", keep_alive=True, service_id="svc")

    # A SECOND LeaseCustody over the same file stands in for the custodian process.
    reader = _custody_on(tmp_path, monkeypatch, clock)
    assert reader.refresh_snapshot()["server_expiry_ms"] == writer.snapshot()["server_expiry_ms"]
    assert reader.expire() == 0
    clock.now_ms += 15_000
    assert reader.expire() == 1
    assert killed == [pgid]


def test_the_boot_anchored_clock_is_monotonic_and_names_its_boot():
    from ouroboros.platform_layer import boot_anchored_monotonic_ms

    first_anchor, first = boot_anchored_monotonic_ms()
    second_anchor, second = boot_anchored_monotonic_ms()
    assert first_anchor == second_anchor and first_anchor
    assert second >= first
    # Never the wall clock: a boot-relative value is orders of magnitude smaller than
    # a Unix-epoch one, and the two must not be confusable.
    assert second < int(time.time() * 1000) // 2


# ── D8 retention has a PRODUCER on the target ────────────────────────────


def _seal_a_process_log(service, task_id: str) -> pathlib.Path:
    """One sealed spool blob big enough that `seal()` does not expire it inline."""

    from ouroboros.execd_spool import SPOOL_MIN_SEAL_BYTES

    sink = service.process_logs.open_stream(
        task_id=task_id, operation_id="op-1", stream="stdout",
    )
    sink.write(b"y" * (SPOOL_MIN_SEAL_BYTES + 64))
    row = sink.seal()
    assert row is not None, "the premise is a sealed blob, not an inline preview"
    return service.process_logs.sealed_path(row["blob_id"])


def test_a_cancelled_task_gives_its_spool_quota_BACK(tmp_path):
    """The Home-side producer D8 declared and never had.

    The sealed log was written and quotad; the release lived only on the per-operation
    stream sink, which dies with the operation. So a finished task's reservation stayed
    forever and the HOST-wide 8 GiB quota was a one-way ratchet — it fills once and
    then every later remote process on that host refuses to spool, with no single event
    to point at. `cancel` is where the target learns a task is terminal, so it is where
    the quota and the blob are released.
    """

    service = _service(tmp_path)
    blob = _seal_a_process_log(service, "task-1")
    held = service.process_logs.ledger.usage()["host_bytes"]
    assert held > 0 and blob.is_file()

    service.cancel(task_id="task-1", request_id="req-1", operation_id="op-cancel")

    assert service.process_logs.ledger.usage()["host_bytes"] == 0
    assert not blob.exists()


def test_the_custody_TICK_expires_a_log_no_cancel_ever_claimed(tmp_path, monkeypatch):
    """The age backstop, on the signal the target already receives.

    A Home that died mid-task sends no cancel, so nothing declares the task terminal.
    Lease renewal is the recurring signal execd already gets, so the sweep rides it —
    and inside the retention window it must touch NOTHING, because a live task's
    evidence is not garbage.
    """

    from ouroboros import execd_spool

    service = _service(tmp_path)
    blob = _seal_a_process_log(service, "task-1")

    service.renew_lease(15_000, task_id="task-1", server_generation="generation-a")
    assert blob.is_file(), "a fresh log must survive the tick"
    assert service.process_logs.ledger.usage()["host_bytes"] > 0

    monkeypatch.setattr(execd_spool, "SPOOL_RETENTION_TTL_MS", 0)
    service.renew_lease(15_000, task_id="task-1", server_generation="generation-a")

    assert not blob.exists()
    assert service.process_logs.ledger.usage()["host_bytes"] == 0
