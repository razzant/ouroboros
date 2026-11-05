"""Default-lane tests of the S39 driver's plumbing (``test_managed_container``), without Docker.

The relay's container end runs here as a local process against the harness's own stub;
the in-container API client reads a published port file; the Update's wheel installs
offline from bytes; the dependency commit adds exactly one lock line; and the lane names
exactly one Docker daemon, never an ambient one.
"""

from __future__ import annotations

import base64
import json
import os
import subprocess
import sys
import threading
import urllib.request

import pytest

from tests.system_e2e.harness import ScriptedStubModel
from tests.system_e2e.test_managed_container import (
    _API_CLIENT,
    _RELAY_PROGRAM,
    F_REQUIREMENT,
    F_WHEEL,
    _dependency_commit,
    _docker_base,
    _docker_env,
    _free_port,
    _git,
    _marker_wheel,
    _ModelRelay,
    _stub_port,
)
def _post_local(url: str, payload: dict) -> dict:
    request = urllib.request.Request(url, data=json.dumps(payload).encode(), method="POST",
                                     headers={"Content-Type": "application/json"})
    with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(request, timeout=30) as answer:
        return json.loads(answer.read())


@pytest.mark.skipif(sys.platform == "win32", reason="the relay runs inside Linux containers")
def test_the_model_relay_carries_concurrent_streams_to_the_host_stub():
    port = _free_port()
    with ScriptedStubModel([], final_answer="relayed") as stub, \
            _ModelRelay("", _stub_port(stub), command=[sys.executable, "-u", "-c", _RELAY_PROGRAM, str(port)]) as relay:
        url = f"http://127.0.0.1:{port}/v1/chat/completions"
        answers: list = []
        threads = [threading.Thread(target=lambda i=i: answers.append(_post_local(url, {
            "model": "mock-model", "messages": [{"role": "user", "content": f"ping {i}"}]})))
            for i in range(6)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=60)
        assert [answer["choices"][0]["message"]["content"] for answer in answers] == ["relayed"] * 6
        request = urllib.request.Request(url, data=json.dumps({"model": "mock-model", "stream": True, "messages": [
            {"role": "user", "content": "stream"}]}).encode(), method="POST", headers={"Content-Type": "application/json"})
        with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(request, timeout=30) as answer:
            assert answer.read().decode().endswith("data: [DONE]\n\n")
        assert relay.streams == 7 and len(stub.calls) == 7
    assert relay.proc.returncode == 0, "the container end exits on the host's EOF"


def test_the_in_container_api_client_reads_the_published_port(tmp_path):
    with ScriptedStubModel([]) as stub:
        port_file = tmp_path / "server_port"
        port_file.write_text(f"{_stub_port(stub)}\n", encoding="utf-8")

        def ask(method, path, payload=None):
            request = {"port_file": str(port_file), "method": method, "path": path, "payload": payload, "timeout": 30}
            run = subprocess.run([sys.executable, "-c", _API_CLIENT], input=json.dumps(request), capture_output=True,
                                 text=True, timeout=60, check=True)
            answer = json.loads(run.stdout)
            return answer["port"], answer["status"], base64.b64decode(answer["body"])

        port, status, body = ask("GET", "/v1/models")
        assert (port, status) == (_stub_port(stub), 200) and json.loads(body)["data"][0]["id"] == "mock-model"
        assert ask("GET", "/nowhere")[1] == 404
        _port, status, body = ask("POST", "/v1/chat/completions", {"messages": [{"role": "user", "content": "x"}]})
        assert status == 200 and json.loads(body)["choices"]


def test_the_marker_wheel_installs_offline_and_imports(tmp_path):
    wheel = tmp_path / F_WHEEL
    wheel.write_bytes(_marker_wheel("probe-mark"))
    target = tmp_path / "site"
    # A --target install touches only tmp_path; the test boundary's PIP_REQUIRE_VIRTUALENV guards the host.
    install = subprocess.run([sys.executable, "-m", "pip", "install", "--no-index", "--no-deps", "--no-cache-dir",
                              "--disable-pip-version-check", "--target", str(target), str(wheel)],
                             capture_output=True, text=True, timeout=300,
                             env={**os.environ, "PIP_REQUIRE_VIRTUALENV": "false"})
    if install.returncode != 0 and "No module named pip" in install.stderr:
        pytest.skip("pip is not available to this interpreter")
    assert install.returncode == 0, install.stderr[-1500:]
    probe = subprocess.run([sys.executable, "-c", "import ouroboros_s39_marker as marker; print(marker.MARK)"],
                           capture_output=True, text=True, timeout=60, env={**os.environ, "PYTHONPATH": str(target)})
    assert probe.stdout.strip() == "probe-mark", probe.stderr


def test_the_dependency_commit_adds_exactly_the_wheel_requirement(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    _git("-C", str(source), "init", "--quiet", "--initial-branch", "ouroboros")
    (source / "requirements-runtime.lock").write_text("anyio==4.14.2\n", encoding="utf-8")
    _git("-C", str(source), "add", ".")
    _git("-C", str(source), "-c", "user.name=t", "-c", "user.email=t@e.invalid", "commit", "--quiet", "-m", "base")
    work, base, target = _dependency_commit(source, tmp_path)
    assert base == _git("-C", str(source), "rev-parse", "HEAD") == _git("-C", str(work), "rev-parse", f"{target}^")
    assert (work / "requirements-runtime.lock").read_text(encoding="utf-8") == f"anyio==4.14.2\n{F_REQUIREMENT}\n"
    assert _git("-C", str(work), "diff", "--name-only", base, target) == "requirements-runtime.lock"


@pytest.mark.parametrize("env, expected", [
    ({"OUROBOROS_CONTAINER_DOCKER_CONTEXT": "qa"}, ["docker", "--context", "qa"]),
    ({"OUROBOROS_CONTAINER_DOCKER_HOST": "unix:///q/d.sock", "OUROBOROS_CONTAINER_DOCKER_CONFIG": "/q/c"},
     ["docker", "--config", "/q/c", "--host", "unix:///q/d.sock"]),
    ({}, None),
    ({"OUROBOROS_CONTAINER_DOCKER_CONTEXT": "default"}, None),
    ({"OUROBOROS_CONTAINER_DOCKER_CONTEXT": "qa", "OUROBOROS_CONTAINER_DOCKER_HOST": "unix:///q/d.sock"}, None),
])
def test_the_lane_names_exactly_one_docker_daemon(monkeypatch, env, expected):
    for key in ("OUROBOROS_CONTAINER_DOCKER_CONTEXT", "OUROBOROS_CONTAINER_DOCKER_HOST",
                "OUROBOROS_CONTAINER_DOCKER_CONFIG"):
        monkeypatch.delenv(key, raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setenv("DOCKER_HOST", "tcp://ambient.invalid:2375")
    if expected is None:
        with pytest.raises(pytest.skip.Exception):
            _docker_base()
    else:
        assert _docker_base() == expected
    assert "DOCKER_HOST" not in _docker_env(), "an ambient daemon selection never reaches the CLI"
