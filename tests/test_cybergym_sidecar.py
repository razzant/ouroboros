"""Docker-free contract tests for the CyberGym sidecar helpers."""

from __future__ import annotations

import importlib

import pytest

sidecar = importlib.import_module("devtools.benchmarks.cybergym.cybergym_sidecar")
IMAGE_DIGEST = "sha256:" + "a" * 64


def _plan():
    return sidecar.build_network_plan("camp-7", "task-42", 8080, 18080)


def _host():
    return sidecar.resolve_rootless_docker_host("unix:///run/user/1006/docker.sock")


def _observation(plan, host, *, wildcard=False, workspace_socket=False, mode=None):
    labels_server = sidecar.required_resource_labels(plan, "server")
    labels_workspace = sidecar.required_resource_labels(plan, "workspace")
    network = {
        "NetworkID": "net-123",
        "Aliases": [plan.server_alias],
    }
    workspace_network = {
        "NetworkID": "net-123",
        "Aliases": [plan.workspace_alias],
    }
    host_ip = "0.0.0.0" if wildcard else plan.verifier_bind_host
    server = {
        "Id": "server-123",
        "Name": "/cyber-server",
        "Config": {"Labels": labels_server, "RepoDigests": [f"cyber/server@{IMAGE_DIGEST}"]},
        "State": {"Pid": 101, "Running": True},
        "HostConfig": {"NetworkMode": mode or plan.network_name},
        "NetworkSettings": {
            "Networks": {plan.network_name: network},
            "Ports": {f"{plan.server_container_port}/tcp": [{"HostIp": host_ip, "HostPort": str(plan.verifier_host_port)}]},
        },
        "Mounts": [{"Source": host.socket_path, "Destination": "/var/run/docker.sock"}],
    }
    workspace_mounts = []
    if workspace_socket:
        workspace_mounts.append({"Source": host.socket_path, "Destination": "/var/run/docker.sock"})
    workspace = {
        "Id": "workspace-123",
        "Name": "/cyber-workspace",
        "Config": {"Labels": labels_workspace, "RepoDigests": [f"cyber/worker@{IMAGE_DIGEST}"]},
        "State": {"Pid": 202, "Running": True},
        "HostConfig": {"NetworkMode": plan.network_name},
        "NetworkSettings": {"Networks": {plan.network_name: workspace_network}},
        "Mounts": workspace_mounts,
    }
    return {
        "docker_host": host.value,
        "network": {"Name": plan.network_name, "Id": "net-123", "Internal": False, "Driver": "bridge"},
        "server": server,
        "workspace": workspace,
        "executor_network": "host",
    }


def _connectivity():
    return {
        "agent_to_server": True,
        "verifier_to_private": {"reachable": True},
        "agent_to_public": True,
        "agent_to_verifier": False,
        "agent_socket_visible": False,
    }


def _daemon_info(host):
    return {
        "ID": "daemon-123",
        "ServerVersion": "28.3.3",
        "SecurityOptions": ["name=rootless"],
        "DockerHost": host.value,
        "DockerRootDir": "/mnt/data/cybergym-docker",
    }


def _protected_connectivity():
    return {
        "agent_to_server_protected": {
            "targets": [
                {"reachable": True, "status_code": 401, "mutating": False},
                {"reachable": True, "status_code": 404, "mutating": False},
            ]
        }
    }


def test_rootless_host_is_explicit_and_rootful_or_tcp_is_rejected():
    assert _host().socket_path == "/run/user/1006/docker.sock"
    with pytest.raises(sidecar.SidecarConfigurationError):
        sidecar.resolve_rootless_docker_host(environ={})
    for value in ("unix:///var/run/docker.sock", "tcp://127.0.0.1:2375", "unix:///tmp/default.sock"):
        with pytest.raises(sidecar.SidecarConfigurationError):
            sidecar.resolve_rootless_docker_host(value)
    assert sidecar.resolve_rootless_docker_host("unix:///tmp/owned.sock", allow_custom=True).socket_path == "/tmp/owned.sock"
    for value in ("/var/run/docker.sock", "/run/docker.sock", "/docker.sock"):
        with pytest.raises(sidecar.SidecarConfigurationError):
            sidecar.DockerHostRef(f"unix://{value}", value, allow_custom=True)


def test_network_plan_aliases_and_no_proxy_are_deterministic():
    first, second = _plan(), sidecar.build_network_plan("camp-7", "task-42", 8080, 18080)
    assert first.server_alias == second.server_alias
    assert first.workspace_alias == second.workspace_alias
    assert first.task_id not in first.workspace_alias
    assert first.opaque_agent_id in first.workspace_alias
    workspace_labels = sidecar.required_resource_labels(first, "workspace")
    assert workspace_labels["com.ouroboros.task"] == first.opaque_agent_id
    assert first.task_id not in repr(workspace_labels)
    assert first.network_name == "cybergym-internal"
    assert first.no_proxy == f"{first.server_alias},{first.server_alias}:8080"
    assert sidecar.build_no_proxy(first.server_alias, 8080, existing="localhost") == f"localhost,{first.server_alias},{first.server_alias}:8080"
    with pytest.raises(sidecar.SidecarConfigurationError):
        sidecar.build_no_proxy("*", 8080)
    with pytest.raises(sidecar.SidecarConfigurationError):
        sidecar.build_network_plan("camp-7", "task-42", 8080, 18080, workspace_alias="task-42-agent")


def test_opaque_agent_id_is_stable_and_task_free():
    first = sidecar.make_opaque_agent_id("camp-7", "task-42")
    assert first == sidecar.make_opaque_agent_id("camp-7", "task-42")
    assert first.startswith("agent-") and len(first) == len("agent-") + 24
    assert "task-42" not in first
    short_plan = sidecar.build_network_plan("camp-7", "x", 8080, 18080)
    assert short_plan.task_id not in short_plan.workspace_alias


def test_network_argv_uses_egress_enabled_named_network_and_explicit_daemon():
    argv = sidecar.build_network_create_argv(_host(), _plan())
    assert argv[:7] == ["docker", "--host", _host().value, "network", "create", "--driver", "bridge"]
    assert "--internal" not in argv
    assert argv[-1] == "cybergym-internal"
    assert "--network" not in argv


def test_connectivity_probe_uses_documented_server_route():
    probes = sidecar.build_connectivity_probe_plan(_plan())
    server_probe = next(item for item in probes if item["name"] == "agent_to_server")
    assert server_probe["target"] == f"{_plan().server_url}/docs"
    assert "/health" not in server_probe["target"]


def test_connectivity_probe_checks_unauthenticated_protected_server_routes():
    plan = _plan()
    probe = next(item for item in sidecar.build_connectivity_probe_plan(plan) if item["name"] == "agent_to_server_protected")
    assert probe["targets"] == (f"{plan.server_url}/query-poc", f"{plan.server_url}/submit-fix")
    assert probe["method"] == "POST"
    assert probe["authentication"] == "none"
    assert probe["expected_reachable"] is True
    assert probe["expected_authorized"] is False
    assert probe["expected_mutating"] is False
    assert probe["requires_all"] is True


def test_server_and_workspace_argv_preserve_socket_boundary():
    plan, host = _plan(), _host()
    server = sidecar.SidecarCommandSpec(
        host,
        plan,
        "cyber/server@sha256:" + "a" * 64,
        "cyber-server",
        command=("python", "-m", "cybergym_server"),
    )
    workspace = sidecar.WorkspaceCommandSpec(host, plan, "cyber/worker:pin", "cyber-workspace", "/tmp/cyber-task")
    server_argv, workspace_argv = sidecar.build_sidecar_argv(server), sidecar.build_workspace_argv(workspace)
    assert ["--host", host.value] == server_argv[1:3] == workspace_argv[1:3]
    assert "--network" in server_argv and plan.network_name in server_argv
    assert "--publish" in server_argv and "127.0.0.1:18080:8080/tcp" in server_argv
    assert any(host.socket_path in item for item in server_argv)
    assert "CYBERGYM_API_KEY" in server_argv
    assert "--mount" in workspace_argv and host.socket_path not in workspace_argv
    assert f"CYBERGYM_SERVER_URL={plan.server_url}" in workspace_argv
    assert f"NO_PROXY={plan.no_proxy}" in workspace_argv
    assert f"CYBERGYM_AGENT_ID={plan.opaque_agent_id}" in workspace_argv
    assert "CYBERGYM_TASK_ID" not in " ".join(workspace_argv)
    assert plan.task_id not in " ".join(workspace_argv)
    assert all("real-secret" not in item for item in server_argv + workspace_argv)
    external_binary_server = sidecar.SidecarCommandSpec(
        host,
        plan,
        "cyber/server@sha256:" + "a" * 64,
        "cyber-external-binaries",
        read_only_mounts={"/srv/cybergym-binaries": "/srv/cybergym-binaries"},
    )
    assert "type=bind,src=/srv/cybergym-binaries,dst=/srv/cybergym-binaries,readonly" in sidecar.build_sidecar_argv(external_binary_server)
    runtime_workspace = sidecar.WorkspaceCommandSpec(
        host,
        plan,
        "cyber/worker:pin",
        "cyber-runtime-workspace",
        "/tmp/cyber-task",
        vulnerable_runtime_host_path="/tmp/cyber-vul",
    )
    runtime_argv = sidecar.build_workspace_argv(runtime_workspace)
    assert "type=bind,src=/tmp/cyber-vul,dst=/workspace/.cybergym-runtime,readonly" in runtime_argv
    assert "CYBERGYM_VULNERABLE_RUNTIME=/workspace/.cybergym-runtime" in runtime_argv
    exec_server = sidecar.SidecarCommandSpec(
        host,
        plan,
        "cyber/server@sha256:" + "a" * 64,
        "cyber-server-exec",
        publish_host_port=False,
    )
    exec_argv = sidecar.build_sidecar_argv(exec_server)
    assert "--publish" not in exec_argv
    with pytest.raises(sidecar.SidecarConfigurationError):
        sidecar.WorkspaceCommandSpec(host, plan, "cyber/worker:pin", "cyber-workspace", "/tmp/cyber-task", extra_env={"API_TOKEN": "real-secret"})
    with pytest.raises(sidecar.SidecarConfigurationError):
        sidecar.WorkspaceCommandSpec(host, plan, "cyber/worker:pin", "cyber-workspace", "/tmp/cyber-task", extra_env={"TASK_HINT": "task-42"})
    with pytest.raises(sidecar.SidecarConfigurationError):
        sidecar.WorkspaceCommandSpec(host, plan, "cyber/worker:pin", "cyber-workspace", "/tmp/cyber-task", labels={"hint": "task-42"})
    for key in ("DOCKER_HOST", "CYBERGYM_SERVER_URL", "NO_PROXY", "https_proxy"):
        with pytest.raises(sidecar.SidecarConfigurationError):
            sidecar.WorkspaceCommandSpec(
                host, plan, "cyber/worker:pin", "cyber-workspace", "/tmp/cyber-task", extra_env={key: "override"}
            )
    with pytest.raises(sidecar.SidecarConfigurationError):
        sidecar.SidecarCommandSpec(
            host, plan, "cyber/server:pin", "cyber-server", api_key_env="DOCKER_HOST"
        )
    with pytest.raises(sidecar.SidecarConfigurationError):
        sidecar.SidecarCommandSpec(
            host, plan, "cyber/server:pin", "cyber-server", container_docker_host="tcp://127.0.0.1:2375"
        )
    mounted_socket_server = sidecar.SidecarCommandSpec(
        host, plan, "cyber/server:pin", "cyber-server", container_docker_host="unix:///var/run/docker.sock"
    )
    assert "DOCKER_HOST=unix:///var/run/docker.sock" in sidecar.build_sidecar_argv(mounted_socket_server)
    with pytest.raises(sidecar.SidecarConfigurationError):
        sidecar.WorkspaceCommandSpec(
            host, plan, "cyber/worker:pin", "cyber-workspace", "/tmp/cyber-task",
            container_docker_host="unix:///var/run/docker.sock",
        )


def test_server_data_mount_keeps_one_absolute_root_for_verifier():
    plan, host = _plan(), _host()
    server = sidecar.SidecarCommandSpec(
        host,
        plan,
        "cyber/server:pin",
        "cyber-server",
        data_host_path="/tmp/cyber-server-root",
        data_container_path="/tmp/cyber-server-root",
    )
    argv = sidecar.build_sidecar_argv(server)
    assert "type=bind,src=/tmp/cyber-server-root,dst=/tmp/cyber-server-root" in argv
    with pytest.raises(sidecar.SidecarConfigurationError, match="same absolute"):
        sidecar.SidecarCommandSpec(
            host,
            plan,
            "cyber/server:pin",
            "cyber-server",
            data_host_path="/tmp/cyber-server-root",
        )
    with pytest.raises(sidecar.SidecarConfigurationError, match="same absolute"):
        sidecar.SidecarCommandSpec(
            host,
            plan,
            "cyber/server:pin",
            "cyber-server",
            data_host_path="/tmp/cyber-server-root",
            data_container_path="/cybergym-data",
        )


def test_forbidden_network_modes_and_wildcard_bind_fail_closed():
    with pytest.raises(sidecar.SidecarConfigurationError):
        sidecar.NetworkPlan("camp", "task", 8080, 18080, network_name="bridge")
    with pytest.raises(sidecar.SidecarConfigurationError):
        sidecar.NetworkPlan("camp", "task", 8080, 18080, verifier_bind_host="0.0.0.0")
    with pytest.raises(sidecar.SidecarConfigurationError):
        sidecar.SidecarCommandSpec(_host(), _plan(), "latest", "server")


def test_executor_host_declaration_is_not_docker_host_networking():
    plan, host = _plan(), _host()
    expectation = sidecar.SidecarExpectation(plan, host, "cyber-server", "cyber-workspace", "server-123", "workspace-123", "net-123", host.socket_path, image_digest=IMAGE_DIGEST, server_pid=101, workspace_pid=202)
    report = sidecar.check_sidecar_attestation(
        _observation(plan, host), expectation, api_key={"present": True, "placeholder": False}, connectivity=_connectivity()
    )
    assert report["ok"] is True
    assert report["executor_network_declaration"] == "host"
    assert report["executor_network_is_docker_host"] is False
    with pytest.raises(sidecar.SidecarAttestationError):
        sidecar.attest_sidecar_runtime({**_observation(plan, host), "executor_network": "none"}, expectation, api_key="valid-key", connectivity=_connectivity())


def test_connectivity_requires_all_positive_and_negative_facts():
    result = sidecar.evaluate_connectivity_checks(_connectivity())
    assert result["ok"] is True
    incomplete = dict(_connectivity())
    del incomplete["agent_to_public"]
    assert sidecar.evaluate_connectivity_checks(incomplete)["ok"] is False
    wrong = dict(_connectivity())
    wrong["agent_socket_visible"] = True
    assert "agent_socket_visible" in sidecar.evaluate_connectivity_checks(wrong)["failed"]


def test_protected_route_probe_requires_transport_denial_and_no_mutation():
    result = sidecar.evaluate_connectivity_checks({**_connectivity(), **_protected_connectivity()})
    assert result["ok"] is True
    assert result["checks"]["agent_to_server_protected"]["observed"]["target_count"] == 2
    mutating = {**_connectivity(), **_protected_connectivity()}
    mutating["agent_to_server_protected"]["targets"][1]["mutating"] = True
    failed = sidecar.evaluate_connectivity_checks(mutating)
    assert failed["ok"] is False
    assert "agent_to_server_protected" in failed["failed"]
    incomplete = {**_connectivity(), "agent_to_server_protected": {"reachable": True}}
    assert sidecar.evaluate_connectivity_checks(incomplete)["ok"] is False
    assert sidecar.evaluate_connectivity_checks(
        _connectivity(), require_protected_route_evidence=True
    )["ok"] is False


def test_attestation_rejects_socket_leak_wildcard_and_default_bridge():
    plan, host = _plan(), _host()
    expectation = sidecar.SidecarExpectation(plan, host, "cyber-server", "cyber-workspace", "server-123", "workspace-123", "net-123", host.socket_path, image_digest=IMAGE_DIGEST, server_pid=101, workspace_pid=202)
    for observation in (
        _observation(plan, host, workspace_socket=True),
        _observation(plan, host, wildcard=True),
        _observation(plan, host, mode="bridge"),
    ):
        report = sidecar.check_sidecar_attestation(observation, expectation, api_key="valid-key", connectivity=_connectivity())
        assert report["ok"] is False


def test_attestation_requires_resolved_digest_on_each_container():
    plan, host = _plan(), _host()
    expectation = sidecar.SidecarExpectation(
        plan, host, "cyber-server", "cyber-workspace", "server-123", "workspace-123", "net-123", host.socket_path,
        image_digest=IMAGE_DIGEST,
    )
    observation = _observation(plan, host)
    observation["workspace"]["Config"].pop("RepoDigests")
    report = sidecar.check_sidecar_attestation(observation, expectation, api_key="valid-key", connectivity=_connectivity())
    assert report["ok"] is False
    assert "workspace.image_digest" in report["failed_checks"]


def test_attestation_accepts_distinct_server_and_workspace_digests():
    plan, host = _plan(), _host()
    server_digest = "sha256:" + "a" * 64
    workspace_digest = "sha256:" + "b" * 64
    expectation = sidecar.SidecarExpectation(
        plan,
        host,
        "cyber-server",
        "cyber-workspace",
        "server-123",
        "workspace-123",
        "net-123",
        host.socket_path,
        server_image_digest=server_digest,
        workspace_image_digest=workspace_digest,
        server_pid=101,
        workspace_pid=202,
    )
    observation = _observation(plan, host)
    observation["server"]["Config"]["RepoDigests"] = [f"cyber/server@{server_digest}"]
    observation["workspace"]["Config"]["RepoDigests"] = [f"cyber/worker@{workspace_digest}"]
    report = sidecar.check_sidecar_attestation(
        observation,
        expectation,
        api_key="valid-key",
        connectivity=_connectivity(),
    )
    assert report["ok"] is True
    assert report["server"]["expected_image_digest"] == server_digest
    assert report["workspace"]["expected_image_digest"] == workspace_digest


def test_attestation_supports_internal_exec_private_route_without_publish():
    plan, host = _plan(), _host()
    expectation = sidecar.SidecarExpectation(
        plan,
        host,
        "cyber-server",
        "cyber-workspace",
        "server-123",
        "workspace-123",
        "net-123",
        host.socket_path,
        image_digest=IMAGE_DIGEST,
        server_pid=101,
        workspace_pid=202,
        publish_host_port=False,
    )
    observation = _observation(plan, host)
    observation["server"]["NetworkSettings"]["Ports"] = {"8080/tcp": None}
    report = sidecar.check_sidecar_attestation(
        observation,
        expectation,
        api_key="valid-key",
        connectivity=_connectivity(),
    )
    assert report["ok"] is True
    assert report["published_verifier"]["mode"] == "container_exec"

    observation["server"]["NetworkSettings"]["Ports"]["9999/tcp"] = [
        {"HostIp": "127.0.0.1", "HostPort": "19999"}
    ]
    rejected = sidecar.check_sidecar_attestation(
        observation,
        expectation,
        api_key="valid-key",
        connectivity=_connectivity(),
    )
    assert rejected["ok"] is False
    assert "server.unexpected_publish" in rejected["failed_checks"]


def test_daemon_evidence_is_required_only_for_strict_production_entrypoint():
    plan, host = _plan(), _host()
    expectation = sidecar.SidecarExpectation(
        plan,
        host,
        "cyber-server",
        "cyber-workspace",
        "server-123",
        "workspace-123",
        "net-123",
        host.socket_path,
        image_digest=IMAGE_DIGEST,
        server_pid=101,
        workspace_pid=202,
    )
    pure = sidecar.check_sidecar_attestation(
        _observation(plan, host), expectation, api_key="valid-key", connectivity=_connectivity()
    )
    assert pure["ok"] is True
    strict_missing = sidecar.check_sidecar_attestation(
        _observation(plan, host),
        expectation,
        api_key="valid-key",
        connectivity=_connectivity(),
        require_daemon_evidence=True,
    )
    assert strict_missing["ok"] is False
    assert "docker_daemon_evidence" in strict_missing["failed_checks"]
    observation = _observation(plan, host)
    observation["docker_info"] = _daemon_info(host)
    strict = sidecar.check_sidecar_attestation(
        observation,
        expectation,
        api_key="valid-key",
        connectivity=_connectivity(),
        require_daemon_evidence=True,
    )
    assert strict["ok"] is True
    assert strict["docker_info"]["status"] == "verified"
    assert strict["docker_info"]["daemon_id"] == "daemon-123"
    with pytest.raises(sidecar.SidecarAttestationError) as exc_info:
        sidecar.attest_sidecar_runtime(
            _observation(plan, host),
            expectation,
            api_key="valid-key",
            connectivity=_connectivity(),
            require_daemon_evidence=True,
        )
    assert "docker_daemon_evidence" in exc_info.value.report["failed_checks"]


def test_cleanup_is_exact_and_never_broad():
    plan, host = _plan(), _host()
    expectation = sidecar.SidecarExpectation(plan, host, "cyber-server", "cyber-workspace", "server-123", "workspace-123", "net-123", host.socket_path, image_digest=IMAGE_DIGEST)
    cleanup = sidecar.build_cleanup_plan(expectation)
    commands = sidecar.cleanup_argv(cleanup)
    assert commands[0] == ("docker", "--host", host.value, "rm", "--force", "workspace-123", "server-123")
    assert commands[1][-2:] == ("rm", "net-123")
    assert all("prune" not in item and "*" not in item for command in commands for item in command)
    good = {
        "removed_container_ids": ["workspace-123", "server-123"],
        "network_removed": True,
        "removed_network_id": "net-123",
        "ownership": {"campaign_id": plan.campaign_id, "container_ids": ["workspace-123", "server-123"], "network_id": "net-123"},
    }
    assert sidecar.validate_cleanup_observation(good, cleanup)["ok"] is True
    assert sidecar.validate_cleanup_observation({**good, "removed_container_ids": ["workspace-123", "server-123", "other"]}, cleanup)["ok"] is False
    assert sidecar.validate_cleanup_observation({key: value for key, value in good.items() if key != "ownership"}, cleanup)["ok"] is False
    assert sidecar.validate_cleanup_observation({**good, "removed_network_id": "other-network"}, cleanup)["ok"] is False
    conflicting_owner = {**good, "ownership": {**good["ownership"], "owner_label": "com.ouroboros.campaign=other"}}
    assert sidecar.validate_cleanup_observation(conflicting_owner, cleanup)["ok"] is False


def test_cleanup_requires_resolved_network_id_and_never_falls_back_to_name():
    plan, host = _plan(), _host()
    with pytest.raises(sidecar.SidecarConfigurationError):
        sidecar.CleanupPlan(host, plan.campaign_id, server_container_id="server-123")
    with pytest.raises(sidecar.SidecarConfigurationError):
        sidecar.CleanupPlan(host, plan.campaign_id, network_id="net-123", workspace_container_ids="workspace-123")
    expectation = sidecar.SidecarExpectation(
        plan, host, "cyber-server", "cyber-workspace", "server-123", "workspace-123", None, host.socket_path,
        image_digest=IMAGE_DIGEST,
    )
    with pytest.raises(sidecar.SidecarConfigurationError):
        sidecar.build_cleanup_plan(expectation)


def test_api_key_status_never_returns_secret():
    status = sidecar.api_key_attestation("real-secret-value")
    assert status["present"] is True and status["placeholder"] is False
    assert "real-secret-value" not in repr(status)
    assert sidecar.is_placeholder_api_key("placeholder") is True
    with pytest.raises(sidecar.SidecarConfigurationError):
        sidecar.require_api_key("placeholder")


def test_api_key_status_rejects_upstream_public_default_fingerprint():
    # Only the non-reversible fingerprint is part of this test; the public
    # example key itself must never be copied into source or test artifacts.
    status = sidecar.api_key_attestation(
        {"present": True, "placeholder": False, "fingerprint": "9605ed570966a4e0"}
    )
    assert status["placeholder"] is True
    with pytest.raises(sidecar.SidecarConfigurationError):
        sidecar.require_api_key(status)


def test_production_expectation_requires_resolved_image_digest():
    plan, host = _plan(), _host()
    with pytest.raises(sidecar.SidecarConfigurationError, match="image_digest"):
        sidecar.SidecarExpectation(plan, host, "cyber-server", "cyber-workspace")
    with pytest.raises(sidecar.SidecarConfigurationError, match="image digest"):
        sidecar.SidecarExpectation(plan, host, "cyber-server", "cyber-workspace", image_digest="cyber/server:latest")


def test_process_custody_attests_pid_cwd_and_port_without_spawning():
    custody = sidecar.build_process_custody("server", 101, "server-123", command=("python", "-m", "server"), cwd="/tmp/cyber", port=18080)
    report = sidecar.attest_process_custody({"pid": 101, "container_id": "server-123", "cwd": "/tmp/cyber", "port": 18080}, custody)
    assert report["ok"] is True
    assert sidecar.attest_process_custody({"pid": 102, "container_id": "server-123"}, custody)["ok"] is False


def test_lifecycle_builder_is_pure_and_can_skip_existing_campaign_network():
    plan, host = _plan(), _host()
    server = sidecar.SidecarCommandSpec(host, plan, "cyber/server:pin", "cyber-server")
    workspace = sidecar.WorkspaceCommandSpec(host, plan, "cyber/worker:pin", "cyber-workspace", "/tmp/cyber-task")
    commands = sidecar.build_lifecycle_commands(server, workspace, create_network=False)
    assert len(commands) == 2
    assert all(command[0:3] == ("docker", "--host", host.value) for command in commands)
