"""Explicit real-process Docker/OpenSSH lane for remote workspace execution.

Set ``OUROBOROS_RUN_REMOTE_SSH_TESTS=1`` to run it.  The fixture creates an
ephemeral clean SSH account and invokes the real framed execd over OpenSSH
stdio.  It deliberately uses the source-mode execd from this checkout; official
no-Python bundle/bootstrap compatibility is a separate artifact-build gate
(``execd-stage``/``execd-bundle`` in CI) and must not be inferred from here.

The skip is honest by construction: with the variable set, a missing Docker
daemon, ssh client or ssh-keygen FAILS rather than skipping — a green zero on the
release-critical path is worse than a red one.

Named case registry
-------------------
The donor numbered two cases RWS-108 and two RWS-107.  Every case below has a
unique id, and the id is the only handle used in reports:

* **RWS-101** ``source_mode_clean_account_and_continuity`` — a clean account with
  no prior state boots execd and keeps one continuity ``host_id`` across restarts.
* **RWS-102** ``remote_workspace_core_operations_and_no_home_env_leak`` — the core
  operation set runs with real target semantics, and no Home sentinel variable
  reaches the child environment.
* **RWS-103** ``process_output_and_exit_semantics_stay_target_native`` — separate
  stdout/stderr, interleaving, invalid UTF-8, ordinary nonzero and exit 255 as an
  ordinary completed child.
* **RWS-104** ``cancel_preserves_session_and_panic_kills_keepalive`` — cancel kills
  the task tree while the session and an unrelated keep-alive survive; panic then
  kills the keep-alive.
* **RWS-105** ``snapshot_exports_content_addressed_artifacts`` — snapshot export is
  content-addressed and verified.
* **RWS-106** ``snapshot_fingerprint_detects_remote_native_mutation`` — a
  target-native mutation changes the fingerprint.
* **RWS-107** ``openssh_loopback_forward_reaches_remote_service_only`` — a remote
  loopback service is reachable through the forward and is not externally bound.
* **RWS-108** ``interleaved_request_ids_never_cross_result_streams`` — concurrent
  request ids never cross streams.
* **RWS-109** ``broker_bootstrap_admit_read_cancel_and_panic`` — the default broker
  path end to end: admit, read, keep-alive service, reconnect, cancel, panic.
  This is the case whose panic teardown surfaced OPEN-6 (see
  ``tests/test_remote_panic_descriptors.py`` for the deterministic reproduction
  and the fix).
* **RWS-110** ``hardened_forward_rejects_alias_owned_forwards`` — an alias that
  owns its own forwards is refused before spawn.

Panic ledger (plan §3.4), expressible at this level:

* **RWS-111** ``panic_ledger_blackholed_transport_kills_within_the_bound`` — a
  blackholed transport (no clean EOF) still kills non-keepalive work within the
  fixed 15s lost-lease ceiling, on the REMOTE monotonic clock.
* **RWS-112** ``panic_ledger_abrupt_home_death_kills_within_the_bound`` — an abrupt
  Home death (SIGKILL of the local ssh child, no panic frame) reaches the same
  bound through the custodian.
* **RWS-113** ``panic_ledger_local_ssh_child_dies_immediately_on_panic`` — panic
  kills the LOCAL OpenSSH child immediately and does not wait for any ACK.
* **RWS-114** ``panic_ledger_stale_generation_lease_is_refused`` — after a restart,
  a lease renewal from the previous server generation is refused, so a stale
  generation can neither keep its groups alive nor kill the new one's.  The
  refusal and its honored counterpart are both OBSERVABLE ON THE WIRE: execd
  answers every lease with an ``ack`` or a typed ``lease_generation_mismatch``
  diagnostic, which is what makes this ledger entry assertable at all.

# OPEN: the donor's RWS-109 also staged task ATTACHMENTS through
# `admit_workspace(attachment_manifest=..., attachment_blobs=...)`.  Attachment
# staging is Home admission policy and now lives with `workspace_admission`/the
# transfer service, so that portion is transferred with that code; what remains
# here is the session/transport contract.
"""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
import select
import shutil
import socket
import subprocess
import tarfile
import threading
import time
import urllib.request
import uuid
from dataclasses import dataclass
from typing import Any

import pytest

from ouroboros.remote_contracts import CONTRACT_SET_VERSION
from ouroboros.remote_ssh_bootstrap import BUNDLE_MANIFEST_SCHEMA_VERSION
from ouroboros.remote_protocol import (
    MAX_BULK_BYTES,
    PROTOCOL_MAJOR,
    PROTOCOL_MINOR,
    encode_bulk,
    encode_control,
    read_frame,
    session_preamble,
)
from ouroboros.workspace_native import MANDATORY_REMOTE_NATIVE_OPERATIONS

pytestmark = [
    pytest.mark.serial,
    pytest.mark.skipif(
        os.environ.get("OUROBOROS_RUN_REMOTE_SSH_TESTS") != "1",
        reason=(
            "real Docker/OpenSSH lane; set OUROBOROS_RUN_REMOTE_SSH_TESTS=1 "
            "to run explicitly"
        ),
    ),
]

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
_REMOTE_WORKSPACE = "/home/ouroboros/workspace"
_REMOTE_STATE = "/home/ouroboros/.local/state/ouroboros-execd-test"
_CAPABILITY_HASH = "a" * 64


def _run(
    command: list[str],
    *,
    timeout: float = 120,
    check: bool = True,
    text: bool = True,
) -> subprocess.CompletedProcess[Any]:
    return subprocess.run(
        command,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
        check=check,
        text=text,
    )


def _wait_until(predicate, *, timeout: float = 10.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.05)
    return bool(predicate())


@dataclass
class _DockerSSHHost:
    container: str
    image: str
    ssh_config: pathlib.Path
    ssh_wrapper: pathlib.Path
    bundle_dir: pathlib.Path
    test_root: pathlib.Path

    def ssh(
        self,
        *remote_args: str,
        timeout: float = 30,
        check: bool = True,
    ) -> subprocess.CompletedProcess[str]:
        return _run(
            [
                "ssh",
                "-F",
                str(self.ssh_config),
                "ouroboros-real-test",
                *remote_args,
            ],
            timeout=timeout,
            check=check,
        )

    def execd(self, *, generation: str | None = None) -> "_ExecdClient":
        return _ExecdClient(
            self,
            generation=generation or f"generation-{uuid.uuid4().hex}",
        )


@pytest.fixture(scope="module")
def docker_ssh_host(tmp_path_factory):
    if shutil.which("docker") is None:
        pytest.fail("OUROBOROS_RUN_REMOTE_SSH_TESTS=1 but docker is unavailable")
    if shutil.which("ssh") is None or shutil.which("ssh-keygen") is None:
        pytest.fail("real remote lane requires OpenSSH client and ssh-keygen")
    probe = _run(
        ["docker", "info", "--format", "{{.ServerVersion}}"],
        timeout=30,
        check=False,
    )
    if probe.returncode != 0:
        pytest.fail(f"Docker daemon is unavailable: {probe.stderr.strip()}")

    root = tmp_path_factory.mktemp("remote-ssh-real")
    key = root / "id_ed25519"
    _run(
        ["ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-f", str(key)],
        timeout=30,
    )
    public_key = key.with_suffix(".pub").read_text(encoding="utf-8").strip()
    (root / "authorized_key").write_text(public_key + "\n", encoding="utf-8")
    dockerfile = """\
FROM python:3.12-slim-bookworm
RUN apt-get update \\
 && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \\
      ca-certificates git openssh-server procps \\
 && rm -rf /var/lib/apt/lists/*
RUN useradd --create-home --shell /bin/bash ouroboros \\
 && mkdir -p /run/sshd /home/ouroboros/.ssh \\
 && chmod 0700 /home/ouroboros/.ssh
COPY authorized_key /home/ouroboros/.ssh/authorized_keys
RUN chown -R ouroboros:ouroboros /home/ouroboros/.ssh \\
 && chmod 0600 /home/ouroboros/.ssh/authorized_keys \\
 && printf '%s\\n' \\
      'PasswordAuthentication no' \\
      'KbdInteractiveAuthentication no' \\
      'PermitRootLogin no' \\
      'AllowUsers ouroboros' \\
      'AllowTcpForwarding yes' \\
      'X11Forwarding no' \\
      >> /etc/ssh/sshd_config
CMD ["/usr/sbin/sshd", "-D", "-e"]
"""
    (root / "Dockerfile").write_text(dockerfile, encoding="utf-8")
    suffix = uuid.uuid4().hex[:12]
    image = f"ouroboros-execd-real-test:{suffix}"
    container = f"ouroboros-execd-real-test-{suffix}"
    _run(["docker", "build", "-q", "-t", image, str(root)], timeout=600)

    manifest = root / "capability-manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "manifest_sha256": _CAPABILITY_HASH,
                "native_operations": sorted(MANDATORY_REMOTE_NATIVE_OPERATIONS),
            },
            sort_keys=True,
            separators=(",", ":"),
        ),
        encoding="utf-8",
    )
    try:
        _run(
            [
                "docker",
                "run",
                "-d",
                "--name",
                container,
                "-p",
                "127.0.0.1::22",
                "--mount",
                f"type=bind,src={_REPO_ROOT},dst=/opt/ouroboros-src,readonly",
                image,
            ],
            timeout=60,
        )
        _run(
            ["docker", "cp", str(manifest), f"{container}:/opt/execd-manifest.json"],
            timeout=30,
        )
        _run(
            [
                "docker",
                "exec",
                "--user",
                "ouroboros",
                container,
                "bash",
                "-lc",
                (
                    "set -eu; "
                    f"mkdir -p {_REMOTE_WORKSPACE}; "
                    f"cd {_REMOTE_WORKSPACE}; "
                    "git init -q; "
                    "git config user.email real-ssh@example.invalid; "
                    "git config user.name 'Real SSH Test'; "
                    "printf 'remote-only\\n' > README.md; "
                    "git add README.md; "
                    "git commit -qm fixture"
                ),
            ],
            timeout=30,
        )
        port_output = _run(
            ["docker", "port", container, "22/tcp"],
            timeout=30,
        ).stdout.strip()
        port = int(port_output.rsplit(":", 1)[1])
        known_hosts = root / "known_hosts"

        def scan_host() -> bool:
            scanned = _run(
                ["ssh-keyscan", "-p", str(port), "127.0.0.1"],
                timeout=10,
                check=False,
            )
            if scanned.returncode == 0 and scanned.stdout.strip():
                known_hosts.write_text(scanned.stdout, encoding="utf-8")
                return True
            return False

        if not _wait_until(scan_host, timeout=20):
            pytest.fail("ephemeral sshd did not become reachable")
        ssh_config = root / "ssh_config"
        ssh_config.write_text(
            "\n".join(
                [
                    "Host ouroboros-real-test",
                    "  HostName 127.0.0.1",
                    f"  Port {port}",
                    "  User ouroboros",
                    f"  IdentityFile {key}",
                    f"  UserKnownHostsFile {known_hosts}",
                    "  StrictHostKeyChecking yes",
                    "  BatchMode yes",
                    "  IdentitiesOnly yes",
                    "  ForwardAgent no",
                    "  ForwardX11 no",
                    "  PermitLocalCommand no",
                    "  RequestTTY no",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        os.chmod(ssh_config, 0o600)
        ssh_wrapper = root / "ssh-with-test-config"
        ssh_wrapper.write_text(
            (
                "#!/bin/sh\n"
                f"exec {shutil.which('ssh')} -F {ssh_config} \"$@\"\n"
            ),
            encoding="utf-8",
        )
        os.chmod(ssh_wrapper, 0o700)
        bundle_dir = _build_source_mode_bundle(root)
        host = _DockerSSHHost(
            container,
            image,
            ssh_config,
            ssh_wrapper,
            bundle_dir,
            root,
        )
        assert host.ssh("true").returncode == 0
        yield host
    finally:
        _run(["docker", "rm", "-f", container], timeout=30, check=False)
        _run(["docker", "image", "rm", "-f", image], timeout=60, check=False)


def _build_source_mode_bundle(root: pathlib.Path) -> pathlib.Path:
    """Build a tiny test-only launcher; this is not production artifact evidence."""

    bundle_dir = root / "source-mode-bundle"
    stage = root / "source-mode-stage"
    binary = stage / "bin" / "ouroboros-execd"
    binary.parent.mkdir(parents=True)
    binary.write_text(
        """\
#!/bin/sh
if [ "${1:-}" = "--version" ]; then
  printf 'ouroboros-execd source-test\\n'
  exit 0
fi
export PYTHONPATH=/opt/ouroboros-src
exec python3 -m ouroboros.execd "$@"
""",
        encoding="utf-8",
    )
    os.chmod(binary, 0o755)
    checksum_file = stage / "stage-files.sha256"
    checksum_file.write_text(
        f"{hashlib.sha256(binary.read_bytes()).hexdigest()}  "
        "bin/ouroboros-execd\n",
        encoding="utf-8",
    )
    bundle_dir.mkdir()
    archive = bundle_dir / "source-mode.tar.gz"
    with tarfile.open(archive, "w:gz") as handle:
        handle.add(binary, arcname="bin/ouroboros-execd")
        handle.add(checksum_file, arcname="stage-files.sha256")
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    binary_digest = hashlib.sha256(binary.read_bytes()).hexdigest()
    common = {
        "archive": archive.name,
        "sha256": digest,
        "size": archive.stat().st_size,
        "glibc_min": "2.17",
        "files": [
            {
                "path": "bin/ouroboros-execd",
                "size": binary.stat().st_size,
                "sha256": binary_digest,
            },
            {
                "path": "stage-files.sha256",
                "size": checksum_file.stat().st_size,
                "sha256": hashlib.sha256(checksum_file.read_bytes()).hexdigest(),
            },
        ],
    }
    manifest = {
        # A real bundle manifest declares both, and Home now READS both: the schema so a
        # manifest shape this build cannot parse is noticed without unpacking the archive,
        # and the Home↔execd contract set so a stale artifact cannot open a session and
        # have the disagreement surface later inside a tool call. This source-mode bundle
        # runs execd straight from /opt/ouroboros-src — the tree under test — so its
        # contract set is by construction this build's.
        "schema_version": BUNDLE_MANIFEST_SCHEMA_VERSION,
        "contract_set_version": CONTRACT_SET_VERSION,
        "build": f"source-test-{digest[:12]}",
        "assets": {
            "linux-x86_64": {
                **common,
                "loader": "/lib64/ld-linux-x86-64.so.2",
            },
            "linux-aarch64": {
                **common,
                "loader": "/lib/ld-linux-aarch64.so.1",
            },
        },
    }
    (bundle_dir / "manifest.json").write_text(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    return bundle_dir


def _broker_capability_manifest() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "manifest_sha256": _CAPABILITY_HASH,
        "public_schema_sha256": "b" * 64,
        "native_operations": [
            {"name": name} for name in sorted(MANDATORY_REMOTE_NATIVE_OPERATIONS)
        ],
        "native_kernel_modules": ["ouroboros.workspace_native"],
        "native_import_modules": ["ouroboros.workspace_native"],
        "native_import_edges": {},
    }


class _ExecdClient:
    def __init__(self, host: _DockerSSHHost, *, generation: str) -> None:
        self.host = host
        self.generation = generation
        self.nonce = os.urandom(24)
        self._send_seq = 0
        self._receive_seq = 0
        self._stderr = bytearray()
        host.ssh(
            "env",
            "PYTHONPATH=/opt/ouroboros-src",
            "python3",
            "-m",
            "ouroboros.execd",
            "--state-root",
            _REMOTE_STATE,
            "--initialize-host-id",
        )
        command = [
            "ssh",
            "-F",
            str(host.ssh_config),
            "ouroboros-real-test",
            "env",
            "PYTHONPATH=/opt/ouroboros-src",
            "python3",
            "-m",
            "ouroboros.execd",
            "--state-root",
            _REMOTE_STATE,
            "--workspace-root",
            _REMOTE_WORKSPACE,
            "--connection-id",
            "connection-real",
            "--project-id",
            "project-real",
            "--server-generation",
            generation,
            "--release-id",
            "source-test",
            "--artifact-sha256",
            "f" * 64,
            "--capability-manifest",
            "/opt/execd-manifest.json",
            "--session-nonce",
            self.nonce.hex(),
        ]
        self.process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            # The local ssh child gets its OWN process group, and that is load-bearing
            # rather than tidiness. Three panic-ledger cases below reach for "the local
            # ssh child's whole group" with `os.killpg(os.getpgid(client.process.pid))`
            # — and a plain Popen inherits the RUNNER's group, so that call sent SIGKILL
            # to pytest itself. The suite died at RWS-111 with no failure and no report,
            # which reads exactly like a suite that simply ended: RWS-111 through RWS-114
            # — the entire kill arm of the panic ledger — never asserted anything on this
            # lane. With its own session the kill hits what each comment says it hits.
            start_new_session=True,
        )
        assert self.process.stdin is not None
        assert self.process.stdout is not None
        assert self.process.stderr is not None
        self._stderr_thread = threading.Thread(
            target=self._drain_stderr,
            daemon=True,
        )
        self._stderr_thread.start()
        expected = session_preamble(self.nonce)
        observed = self._read_exact(len(expected), timeout=20)
        if observed != expected:
            self.close()
            raise AssertionError(
                f"execd preamble mismatch; stderr={self.stderr_text!r}"
            )
        self._send(
            "handshake",
            nonce=self.nonce.hex(),
            protocol_major=PROTOCOL_MAJOR,
            protocol_minor=PROTOCOL_MINOR,
            capability_hash=_CAPABILITY_HASH,
        )
        self.handshake = self._receive("handshake_ok", timeout=20)

    @property
    def stderr_text(self) -> str:
        return bytes(self._stderr).decode("utf-8", errors="replace")

    def _drain_stderr(self) -> None:
        assert self.process.stderr is not None
        while True:
            chunk = self.process.stderr.read(4096)
            if not chunk:
                return
            remaining = 128_000 - len(self._stderr)
            if remaining > 0:
                self._stderr.extend(chunk[:remaining])

    def _read_exact(self, size: int, *, timeout: float) -> bytes:
        assert self.process.stdout is not None
        chunks = bytearray()
        deadline = time.monotonic() + timeout
        while len(chunks) < size:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise AssertionError(
                    f"timed out reading execd; stderr={self.stderr_text!r}"
                )
            ready, _, _ = select.select(
                [self.process.stdout.fileno()],
                [],
                [],
                remaining,
            )
            if not ready:
                continue
            chunk = os.read(self.process.stdout.fileno(), size - len(chunks))
            if not chunk:
                raise AssertionError(
                    f"execd closed early; stderr={self.stderr_text!r}"
                )
            chunks.extend(chunk)
        return bytes(chunks)

    def _send(self, kind: str, **fields: Any) -> int:
        assert self.process.stdin is not None
        sequence = self._send_seq
        self._send_seq += 1
        self.process.stdin.write(
            encode_control({"kind": kind, "seq": sequence, **fields})
        )
        self.process.stdin.flush()
        return sequence

    def _receive(self, expected: str | set[str], *, timeout: float = 20) -> dict[str, Any]:
        assert self.process.stdout is not None
        wanted = {expected} if isinstance(expected, str) else set(expected)
        ready, _, _ = select.select(
            [self.process.stdout.fileno()],
            [],
            [],
            timeout,
        )
        if not ready:
            raise AssertionError(
                f"timed out waiting for {sorted(wanted)}; stderr={self.stderr_text!r}"
            )
        label, message = read_frame(self.process.stdout)
        assert label == "control"
        assert isinstance(message, dict)
        assert message["seq"] == self._receive_seq
        self._receive_seq += 1
        if message["kind"] not in wanted:
            raise AssertionError(
                f"expected {sorted(wanted)}, got {message}; "
                f"stderr={self.stderr_text!r}"
            )
        return message

    def renew(self, task_id: str = "", ttl_ms: int = 15_000) -> str:
        """Renew a lease and CONSUME its answer; returns the accepted lease_id.

        Every case on this lane renews, so every case now observes that execd honors
        the generation it was started with — a silent wire could not tell an accepted
        lease from a dropped frame.
        """

        lease_id = f"lease-{uuid.uuid4().hex}"
        fields: dict[str, Any] = {
            "server_generation": self.generation,
            "lease_id": lease_id,
            "ttl_ms": ttl_ms,
        }
        if task_id:
            fields["task_id"] = task_id
        sequence = self._send("lease", **fields)
        ack = self._receive("ack")
        assert ack["ack_seq"] == sequence
        assert ack["optional"]["lease"] == {
            "lease_id": lease_id,
            "server_generation": self.generation,
        }
        return lease_id

    def prepare(
        self,
        tool: str,
        args: dict[str, Any],
        *,
        task_id: str,
        request_id: str | None = None,
        operation_id: str | None = None,
    ) -> dict[str, Any]:
        request_id = request_id or f"request-{uuid.uuid4().hex}"
        operation_id = operation_id or f"operation-{uuid.uuid4().hex}"
        self._send(
            "prepare",
            request_id=request_id,
            operation_id=operation_id,
            tool=tool,
            args=args,
            task_id=task_id,
            workspace_id=self.handshake["optional"]["admission"]["workspace_id"],
        )
        return self._receive("prepared")

    def upload_blob(
        self,
        *,
        request_id: str,
        operation_id: str,
        blob_id: str,
        payload: bytes,
    ) -> None:
        digest = hashlib.sha256(payload).hexdigest()
        self._send(
            "blob_manifest",
            request_id=request_id,
            operation_id=operation_id,
            blob_id=blob_id,
            size=len(payload),
            sha256=digest,
        )
        if not payload:
            ack = self._receive("blob_ack")
            assert ack["complete"] is True
            return
        offset = 0
        chunk_seq = 0
        assert self.process.stdin is not None
        while offset < len(payload):
            chunk = payload[offset : offset + MAX_BULK_BYTES]
            self.process.stdin.write(encode_bulk(chunk))
            self.process.stdin.flush()
            offset += len(chunk)
            ack = self._receive("blob_ack")
            assert ack["blob_id"] == blob_id
            assert ack["chunk_seq"] == chunk_seq
            assert ack["complete"] is (offset == len(payload))
            chunk_seq += 1

    def continue_prepared(self, prepared: dict[str, Any]) -> dict[str, Any]:
        self._send(
            "continue",
            request_id=prepared["request_id"],
            operation_id=prepared["operation_id"],
            prepared_hash=prepared["prepared_hash"],
            optional={
                "prepared_token": prepared["prepared"]["prepared_token"],
            },
        )
        message = self._receive({"result", "diagnostic"}, timeout=30)
        if message["kind"] == "diagnostic":
            raise AssertionError(f"remote diagnostic: {message['diagnostic']}")
        ack_sequence = self._send(
            "ack",
            ack_seq=message["seq"],
            request_id=prepared["request_id"],
            operation_id=prepared["operation_id"],
            optional={"prepared_hash": prepared["prepared_hash"]},
        )
        ack = self._receive("ack")
        assert ack["ack_seq"] == ack_sequence
        return message["result"]

    def call(
        self,
        tool: str,
        args: dict[str, Any],
        *,
        task_id: str = "task-real",
    ) -> dict[str, Any]:
        return self.continue_prepared(
            self.prepare(tool, args, task_id=task_id)
        )

    def cancel_task(self, task_id: str) -> None:
        sequence = self._send(
            "cancel",
            request_id=f"cancel-{uuid.uuid4().hex}",
            operation_id=f"cancel-{uuid.uuid4().hex}",
            task_id=task_id,
        )
        ack = self._receive("ack")
        assert ack["ack_seq"] == sequence

    def panic(self) -> None:
        self._send("panic", server_generation=self.generation)
        self.process.wait(timeout=10)

    def close(self) -> None:
        if self.process.stdin is not None and not self.process.stdin.closed:
            self.process.stdin.close()
        try:
            self.process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait(timeout=5)
        self._stderr_thread.join(timeout=2)

    def __enter__(self) -> "_ExecdClient":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        del exc_type, exc, tb
        self.close()


def _envelope(result: dict[str, Any]) -> dict[str, Any]:
    assert result["completion"] == "completed"
    return result["envelope"]


def test_rws_101_source_mode_clean_account_and_continuity(docker_ssh_host):
    assert not pathlib.Path(_REMOTE_WORKSPACE).exists()
    with docker_ssh_host.execd(generation="generation-continuity-a") as first:
        first_id = first.handshake["host_id"]
        assert first.handshake["capability_hash"] == _CAPABILITY_HASH
        assert first.handshake["optional"]["admission"]["canonical_root"] == _REMOTE_WORKSPACE
    with docker_ssh_host.execd(generation="generation-continuity-b") as second:
        assert second.handshake["host_id"] == first_id
        envelope = _envelope(
            second.call("read_file", {"path": "README.md"}, task_id="task-continuity")
        )
        assert "remote-only" in envelope["text"]


def test_rws_102_remote_workspace_core_operations_and_no_home_env_leak(
    docker_ssh_host,
    monkeypatch,
):
    sentinel = f"HOME_SECRET_{uuid.uuid4().hex}"
    monkeypatch.setenv("OUROBOROS_REMOTE_TEST_SECRET", sentinel)
    with docker_ssh_host.execd() as client:
        _envelope(
            client.call(
                "write_file",
                {"path": "core.txt", "content": "one\ntwo\n"},
                task_id="task-core",
            )
        )
        read = _envelope(
            client.call("read_file", {"path": "core.txt"}, task_id="task-core")
        )
        listed = _envelope(
            client.call("list_files", {"path": "."}, task_id="task-core")
        )
        client.renew("task-core")
        process = _envelope(
            client.call(
                "run_command",
                {
                    "cmd": [
                        "python3",
                        "-c",
                        (
                            "import os; "
                            "print(os.environ.get('OUROBOROS_REMOTE_TEST_SECRET','absent'))"
                        ),
                    ],
                    "cwd": _REMOTE_WORKSPACE,
                    "timeout_sec": 10,
                },
                task_id="task-core",
            )
        )
        status = _envelope(
            client.call("vcs_status", {}, task_id="task-core")
        )

    assert "one\ntwo" in read["text"]
    assert "core.txt" in listed["text"]
    assert sentinel not in process["text"]
    assert "absent" in process["text"]
    assert "core.txt" in status["text"]


def test_rws_103_process_output_and_exit_semantics_stay_target_native(
    docker_ssh_host,
):
    with docker_ssh_host.execd() as client:
        client.renew("task-process")
        result = _envelope(
            client.call(
                "run_command",
                {
                    "cmd": [
                        "python3",
                        "-c",
                        (
                            "import os,sys; "
                            "os.write(1,b'out\\xff\\n'+b'x'*350000); "
                            "os.write(2,b'err\\xfe\\n'); "
                            "sys.exit(255)"
                        ),
                    ],
                    "cwd": _REMOTE_WORKSPACE,
                    "timeout_sec": 20,
                },
                task_id="task-process",
            )
        )

    assert result["process"]["returncode"] == 255
    assert "out�" in result["process"]["stdout"]
    assert "err�" in result["process"]["stderr"]
    assert result["artifacts"]


def test_rws_104_cancel_preserves_session_and_panic_kills_keepalive(
    docker_ssh_host,
):
    generation = f"generation-custody-{uuid.uuid4().hex}"
    client = docker_ssh_host.execd(generation=generation)
    try:
        client.renew("task-cancel")
        started = _envelope(
            client.call(
                "start_service",
                {
                    "name": "cancelled",
                    "cmd": [
                        "python3",
                        "-c",
                        "import time; CANCELLED_SENTINEL=1; time.sleep(300)",
                    ],
                    "cwd": _REMOTE_WORKSPACE,
                },
                task_id="task-cancel",
            )
        )
        cancelled_ref = started["trace"]["service_ref"]
        client.cancel_task("task-cancel")
        assert _wait_until(
            lambda: not _envelope(
                client.call(
                    "service_status",
                    {"name": "cancelled", "_service_ref": cancelled_ref},
                    task_id="task-cancel",
                )
            )["trace"]["running"],
            timeout=5,
        )
        assert "remote-only" in _envelope(
            client.call(
                "read_file",
                {"path": "README.md"},
                task_id="task-after-cancel",
            )
        )["text"]

        client.renew("task-keepalive")
        _envelope(
            client.call(
                "start_service",
                {
                    "name": "keepalive",
                    "cmd": [
                        "python3",
                        "-c",
                        "import time; KEEPALIVE_SENTINEL=1; time.sleep(300)",
                    ],
                    "cwd": _REMOTE_WORKSPACE,
                    "keep_alive": True,
                },
                task_id="task-keepalive",
            )
        )
        client.panic()
        assert _wait_until(
            lambda: docker_ssh_host.ssh(
                "pgrep",
                "-f",
                "[K]EEPALIVE_SENTINEL",
                check=False,
            ).returncode
            == 1,
            timeout=5,
        )
    finally:
        client.close()


def test_rws_105_snapshot_exports_content_addressed_artifacts(docker_ssh_host):
    with docker_ssh_host.execd() as client:
        _envelope(
            client.call(
                "write_file",
                {"path": "artifact.txt", "content": "artifact-" + "z" * 200_000},
                task_id="task-artifact",
            )
        )
        result = client.call(
            "snapshot_manifest_and_blob_export",
            {},
            task_id="task-artifact",
        )
        envelope = _envelope(result)
        fingerprint = envelope["trace"]["snapshot"]["fingerprint"]
        patch = (
            b"diff --git a/blob-upload.txt b/blob-upload.txt\n"
            b"new file mode 100644\n"
            b"--- /dev/null\n"
            b"+++ b/blob-upload.txt\n"
            b"@@ -0,0 +1 @@\n"
            b"+uploaded-through-bulk-frame\n"
        )
        uploaded_payload = b"uploaded-through-bulk-frame\n"
        blob_id = hashlib.sha256(patch).hexdigest()
        request_id = f"request-upload-{uuid.uuid4().hex}"
        operation_id = f"operation-upload-{uuid.uuid4().hex}"
        client.upload_blob(
            request_id=request_id,
            operation_id=operation_id,
            blob_id=blob_id,
            payload=patch,
        )
        prepared = client.prepare(
            "guarded_patch_apply",
            {
                "expected_fingerprint": fingerprint,
                "patch_blob_id": blob_id,
                "changes": [
                    {
                        "path": "blob-upload.txt",
                        "before": None,
                        "after": {
                            "path": "blob-upload.txt",
                            "kind": "file",
                            "sha256": hashlib.sha256(uploaded_payload).hexdigest(),
                            "size": len(uploaded_payload),
                            "mode": 0o644,
                        },
                    }
                ],
            },
            task_id="task-artifact",
            request_id=request_id,
            operation_id=operation_id,
        )
        applied = _envelope(client.continue_prepared(prepared))
        uploaded = _envelope(
            client.call(
                "read_file",
                {"path": "blob-upload.txt"},
                task_id="task-artifact",
            )
        )

    assert envelope["trace"]["completion"] in {"complete", "partial"}
    assert envelope["artifacts"]
    for artifact in envelope["artifacts"]:
        assert artifact["blob_id"] == artifact["sha256"]
        assert len(artifact["sha256"]) == 64
        assert result["output_blobs"][artifact["blob_id"]] == artifact["blob_id"]
    assert "guarded remote patch applied" in applied["text"]
    assert "uploaded-through-bulk-frame" in uploaded["text"]


def test_rws_106_snapshot_fingerprint_detects_remote_native_mutation(
    docker_ssh_host,
):
    with docker_ssh_host.execd() as client:
        first = _envelope(
            client.call(
                "snapshot_manifest_and_blob_export",
                {},
                task_id="task-snapshot",
            )
        )
        second = _envelope(
            client.call(
                "snapshot_manifest_and_blob_export",
                {},
                task_id="task-snapshot",
            )
        )
        _envelope(
            client.call(
                "write_file",
                {"path": "snapshot-change.txt", "content": uuid.uuid4().hex},
                task_id="task-snapshot",
            )
        )
        changed = _envelope(
            client.call(
                "snapshot_manifest_and_blob_export",
                {},
                task_id="task-snapshot",
            )
        )

    assert (
        first["trace"]["snapshot"]["fingerprint"]
        == second["trace"]["snapshot"]["fingerprint"]
    )
    assert (
        changed["trace"]["snapshot"]["fingerprint"]
        != first["trace"]["snapshot"]["fingerprint"]
    )


def test_rws_107_openssh_loopback_forward_reaches_remote_service_only(
    docker_ssh_host,
):
    with docker_ssh_host.execd() as client:
        client.renew("task-forward")
        started = _envelope(
            client.call(
                "start_service",
                {
                    "name": "http",
                    "cmd": [
                        "python3",
                        "-m",
                        "http.server",
                        "18080",
                        "--bind",
                        "127.0.0.1",
                    ],
                    "cwd": _REMOTE_WORKSPACE,
                },
                task_id="task-forward",
            )
        )
        service_ref = started["trace"]["service_ref"]
        probe = socket.socket()
        probe.bind(("127.0.0.1", 0))
        local_port = int(probe.getsockname()[1])
        probe.close()
        forward = subprocess.Popen(
            [
                "ssh",
                "-F",
                str(docker_ssh_host.ssh_config),
                "-N",
                "-T",
                "-o",
                "ExitOnForwardFailure=yes",
                "-L",
                f"127.0.0.1:{local_port}:127.0.0.1:18080",
                "ouroboros-real-test",
            ],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        try:
            assert _wait_until(
                lambda: _http_body(local_port) == b"remote-only\n",
                timeout=10,
            )
        finally:
            forward.terminate()
            try:
                forward.wait(timeout=5)
            except subprocess.TimeoutExpired:
                forward.kill()
                forward.wait(timeout=5)
            _envelope(
                client.call(
                    "stop_service",
                    {"name": "http", "_service_ref": service_ref},
                    task_id="task-forward",
                )
            )


def _http_body(port: int) -> bytes | None:
    try:
        with urllib.request.urlopen(
            f"http://127.0.0.1:{port}/README.md",
            timeout=0.5,
        ) as response:
            return response.read()
    except (OSError, ValueError):
        return None


def test_rws_108_interleaved_request_ids_never_cross_result_streams(
    docker_ssh_host,
):
    with docker_ssh_host.execd() as client:
        left = client.prepare(
            "read_file",
            {"path": "README.md"},
            task_id="task-left",
            request_id="request-left",
            operation_id="operation-left",
        )
        right = client.prepare(
            "list_files",
            {"path": "."},
            task_id="task-right",
            request_id="request-right",
            operation_id="operation-right",
        )
        for prepared in (left, right):
            client._send(
                "continue",
                request_id=prepared["request_id"],
                operation_id=prepared["operation_id"],
                prepared_hash=prepared["prepared_hash"],
                optional={
                    "prepared_token": prepared["prepared"]["prepared_token"],
                },
            )
        received: dict[str, dict[str, Any]] = {}
        for _ in range(2):
            message = client._receive({"result", "diagnostic"}, timeout=20)
            assert message["kind"] == "result"
            received[message["request_id"]] = message
        assert set(received) == {"request-left", "request-right"}
        assert "remote-only" in received["request-left"]["result"]["envelope"]["text"]
        assert "README.md" in received["request-right"]["result"]["envelope"]["text"]
        assert received["request-left"]["operation_id"] == "operation-left"
        assert received["request-right"]["operation_id"] == "operation-right"


def test_rws_109_broker_bootstrap_admit_read_cancel_and_panic(
    docker_ssh_host,
):
    from ouroboros.remote_workspace import RemoteSessionBroker

    drive_root = docker_ssh_host.test_root / "broker-drive"
    drive_root.mkdir()
    generation = f"broker-generation-{uuid.uuid4().hex}"
    broker = RemoteSessionBroker(
        drive_root,
        generation,
        _broker_capability_manifest(),
        bundle_dir=docker_ssh_host.bundle_dir,
        ssh_binary=str(docker_ssh_host.ssh_wrapper),
    )
    connection = {
        "id": "connection-real",
        "name": "Real SSH",
        "ssh_alias": "ouroboros-real-test",
        "expected_host_id": "",
    }
    task_id = f"task-broker-{uuid.uuid4().hex}"
    service_name = f"broker-service-{uuid.uuid4().hex[:8]}"
    try:
        broker.start()
        admitted = broker.admit_workspace(
            connection,
            remote_root=_REMOTE_WORKSPACE,
            project_id="project-real",
            task_id=task_id,
        )
        # Session admission returns identity only; the placement descriptor is
        # DERIVED from it. Attachment staging is Home policy and lives elsewhere.
        workspace_ref = admitted["workspace_ref"]
        assert workspace_ref["kind"] == "ssh"
        assert admitted["canonical_root"] == _REMOTE_WORKSPACE
        assert admitted["capability_hash"] == _CAPABILITY_HASH
        assert "attachment_manifest" not in admitted
        prepared = broker.prepare(
            workspace_ref,
            request_id=f"request-read-{uuid.uuid4().hex}",
            operation_id=f"operation-read-{uuid.uuid4().hex}",
            tool="read_file",
            args={"path": "README.md"},
            task_id=task_id,
        )
        read = broker.execute_prepared(
            workspace_ref,
            prepared,
            canonical_args=prepared.execution_args,
            task_id=task_id,
        )
        service_prepared = broker.prepare(
            workspace_ref,
            request_id=f"request-service-{uuid.uuid4().hex}",
            operation_id=f"operation-service-{uuid.uuid4().hex}",
            tool="start_service",
            args={
                "name": service_name,
                "cmd": [
                    "python3",
                    "-c",
                    "import time; BROKER_PANIC_SENTINEL=1; time.sleep(300)",
                ],
                "cwd": _REMOTE_WORKSPACE,
                "keep_alive": True,
            },
            task_id=task_id,
        )
        broker.execute_prepared(
            workspace_ref,
            service_prepared,
            canonical_args=service_prepared.execution_args,
            task_id=task_id,
        )

        assert "remote-only" in read.text
        assert broker.has_active_lease("connection-real") is True
        assert docker_ssh_host.ssh(
            "pgrep",
            "-f",
            "[B]ROKER_PANIC_SENTINEL",
            check=False,
        ).returncode == 0
        reconnected = broker.reconnect_connection(connection, timeout_sec=20)
        assert reconnected["status"] == "ready"
        assert docker_ssh_host.ssh(
            "pgrep",
            "-f",
            "[B]ROKER_PANIC_SENTINEL",
            check=False,
        ).returncode == 0
        assert broker.cancel(
            workspace_ref,
            task_id=task_id,
        )
        # The task lease ended, but its explicit keep-alive service still owns
        # this connection generation until stop/death/panic.
        assert broker.has_active_lease("connection-real") is True
        reconnect_read = broker.prepare(
            workspace_ref,
            request_id=f"request-after-cancel-{uuid.uuid4().hex}",
            operation_id=f"operation-after-cancel-{uuid.uuid4().hex}",
            tool="read_file",
            args={"path": "README.md"},
            task_id="task-after-cancel",
            project_id="project-real",
        )
        assert "remote-only" in broker.execute_prepared(
            workspace_ref,
            reconnect_read,
            canonical_args=reconnect_read.execution_args,
            task_id="task-after-cancel",
        ).text

        broker.panic()
        assert _wait_until(
            lambda: docker_ssh_host.ssh(
                "pgrep",
                "-f",
                "[B]ROKER_PANIC_SENTINEL",
                check=False,
            ).returncode
            == 1,
            timeout=5,
        )
        assert broker.status()["connections"] == []
    finally:
        broker.close(timeout_sec=2)


def test_rws_110_hardened_forward_rejects_alias_owned_forwards(
    docker_ssh_host,
):
    from ouroboros.remote_ssh import validated_ssh_base_command
    from ouroboros.remote_workspace import RemoteWorkspaceError

    hostile_config = docker_ssh_host.test_root / "hostile-forward-config"
    original = docker_ssh_host.ssh_config.read_text(encoding="utf-8")
    hostile_config.write_text(
        original.replace(
            "  RequestTTY no\n",
            (
                "  RequestTTY no\n"
                "  LocalForward 127.0.0.1:39091 127.0.0.1:22\n"
                "  RemoteForward 39092 127.0.0.1:22\n"
                "  DynamicForward 127.0.0.1:39093\n"
            ),
        ),
        encoding="utf-8",
    )
    wrapper = docker_ssh_host.test_root / "ssh-hostile-forward"
    wrapper.write_text(
        (
            "#!/bin/sh\n"
            f"exec {shutil.which('ssh')} -F {hostile_config} \"$@\"\n"
        ),
        encoding="utf-8",
    )
    os.chmod(wrapper, 0o700)

    with pytest.raises(RemoteWorkspaceError) as blocked:
        validated_ssh_base_command(
            "ouroboros-real-test",
            str(wrapper),
            forwarding=True,
        )
    assert blocked.value.code == "unsafe_ssh_forwarding"


# ── panic ledger (plan §3.4) ────────────────────────────────────────────
#
# Every case here proves a BOUND, not a happy path.  The 15 second ceiling is a
# physical failure-detection bound for a true partition, so these tests may only
# assert that death happens WITHIN it — never that it is delayed to it.

_LEASE_CEILING_SEC = 15.0
# Generous headroom over the ceiling: a slow CI host must not turn a satisfied
# bound into a flake, and overshooting the ceiling is still a failure.
_LEASE_OBSERVE_SEC = _LEASE_CEILING_SEC + 10.0


def _sentinel_running(host: _DockerSSHHost, sentinel: str) -> bool:
    return (
        host.ssh("pgrep", "-f", f"[{sentinel[0]}]{sentinel[1:]}", check=False).returncode
        == 0
    )


def _start_sentinel_service(
    client: "_ExecdClient",
    *,
    name: str,
    sentinel: str,
    task_id: str,
    keep_alive: bool = False,
) -> dict[str, Any]:
    return _envelope(
        client.call(
            "start_service",
            {
                "name": name,
                "cmd": [
                    "python3",
                    "-c",
                    f"import time; {sentinel}=1; time.sleep(300)",
                ],
                "cwd": _REMOTE_WORKSPACE,
                **({"keep_alive": True} if keep_alive else {}),
            },
            task_id=task_id,
        )
    )


def test_rws_111_panic_ledger_blackholed_transport_kills_within_the_bound(
    docker_ssh_host,
):
    """No clean EOF, no panic frame — only lease expiry may end this.

    The transport is blackholed by killing the LOCAL ssh child's process group
    without letting it close the channel, which is the closest reachable analogue
    of a partition: the remote side observes silence, not a shutdown.
    """

    sentinel = "BLACKHOLE_SENTINEL"
    client = docker_ssh_host.execd()
    try:
        client.renew("task-blackhole")
        _start_sentinel_service(
            client,
            name="blackholed",
            sentinel=sentinel,
            task_id="task-blackhole",
        )
        assert _sentinel_running(docker_ssh_host, sentinel)

        # SIGKILL the local ssh child's whole group: no protocol frame, no
        # graceful close, and Home stops renewing from this moment on.
        os.killpg(os.getpgid(client.process.pid), 9)

        assert _wait_until(
            lambda: not _sentinel_running(docker_ssh_host, sentinel),
            timeout=_LEASE_OBSERVE_SEC,
        ), "a blackholed transport left remote work alive past the lease ceiling"
    finally:
        docker_ssh_host.ssh(
            "pkill", "-f", f"[{sentinel[0]}]{sentinel[1:]}", check=False
        )
        try:
            client.close()
        except Exception:
            pass


def test_rws_112_panic_ledger_abrupt_home_death_kills_keepalive_within_the_bound(
    docker_ssh_host,
):
    """A keep-alive outlives its TASK, never its server generation.

    Home dies abruptly, so nothing announces the generation's end. The custodian
    must still reclaim the keep-alive group within the same fixed bound.
    """

    sentinel = "HOMEDEATH_SENTINEL"
    client = docker_ssh_host.execd()
    try:
        client.renew("task-home-death")
        _start_sentinel_service(
            client,
            name="home-death-keepalive",
            sentinel=sentinel,
            task_id="task-home-death",
            keep_alive=True,
        )
        client.cancel_task("task-home-death")
        # Explicit keep-alive survives ordinary task finalization.
        assert _sentinel_running(docker_ssh_host, sentinel)

        os.killpg(os.getpgid(client.process.pid), 9)

        assert _wait_until(
            lambda: not _sentinel_running(docker_ssh_host, sentinel),
            timeout=_LEASE_OBSERVE_SEC,
        ), "keep-alive outlived the server generation past the lease ceiling"
    finally:
        docker_ssh_host.ssh(
            "pkill", "-f", f"[{sentinel[0]}]{sentinel[1:]}", check=False
        )
        try:
            client.close()
        except Exception:
            pass


def test_rws_113_panic_ledger_local_ssh_child_dies_immediately_on_panic(
    docker_ssh_host,
):
    """The LOCAL ssh child dies at once and waits for no acknowledgement.

    Measured, not asserted structurally: the local teardown must complete far inside
    the partition ceiling, because that ceiling applies only when delivery is
    impossible. The descriptors it frees must be immediately reusable (OPEN-6), which
    is now ASSERTED here rather than merely demonstrated by opening a probe.

    SCOPE, stated because the name invites a wider reading: what runs here is a signal
    against a live OpenSSH child on a real session — not `OpenSSHExecdTransport.panic`,
    which this lane wires no transport for. Panic's own contract, that it never blocks
    behind a busy send, is driven against the real method in
    `tests/test_remote_panic_descriptors.py::test_panic_does_not_wait_for_a_busy_send`.
    This test used to close on `assert callable(OpenSSHExecdTransport.panic)`, which is
    true of a method whose body is `pass`.
    """

    from ouroboros.remote_ssh import _release_child_streams

    sentinel = "LOCALPANIC_SENTINEL"
    client = docker_ssh_host.execd()
    try:
        client.renew("task-local-panic")
        _start_sentinel_service(
            client,
            name="local-panic",
            sentinel=sentinel,
            task_id="task-local-panic",
        )
        child = client.process

        released = {
            stream.fileno()
            for stream in (child.stdin, child.stdout, child.stderr)
            if stream is not None and not stream.closed
        }
        started = time.monotonic()
        os.killpg(os.getpgid(child.pid), 15)
        child.wait(timeout=5)
        _release_child_streams(child)
        elapsed = time.monotonic() - started

        assert elapsed < 5, "local OpenSSH teardown waited on the remote side"
        assert child.poll() is not None
        # The freed descriptors must be reusable immediately (OPEN-6) — asserted, not
        # merely exercised: opening a probe and closing it again proves nothing about
        # whose descriptors it got.
        probe = subprocess.Popen(
            ["true"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        probe.wait(timeout=5)
        reused = {
            stream.fileno()
            for stream in (probe.stdin, probe.stdout, probe.stderr)
            if stream is not None and not stream.closed
        }
        assert released & reused, "the OS did not reuse a released descriptor"
        probe.stdin.close()
        probe.stdout.close()
        probe.stderr.close()
    finally:
        docker_ssh_host.ssh(
            "pkill", "-f", f"[{sentinel[0]}]{sentinel[1:]}", check=False
        )
        try:
            client.close()
        except Exception:
            pass


def test_rws_114_panic_ledger_stale_generation_lease_is_refused(docker_ssh_host):
    """Two Home generations on one host cannot touch each other's groups.

    A renewal quoting the PREVIOUS generation must be refused, so a restarted
    Home neither inherits nor can be killed by the generation it replaced. The
    refusal is asserted ON THE WIRE rather than inferred: an invariant with no
    representation in the protocol cannot be observed by Home, by this test, or
    by anyone auditing the constitutional bound it protects. Valid renewal is
    the other half of the same contract — every `renew` on this lane now
    consumes and checks its ack — because "refused" only means something if
    "honored" looks different.
    """

    stale_generation = f"generation-stale-{uuid.uuid4().hex}"
    client = docker_ssh_host.execd()
    try:
        client.renew("task-generations")
        # A lease naming a generation this session does not own.
        stale_lease = f"lease-{uuid.uuid4().hex}"
        client._send(
            "lease",
            server_generation=stale_generation,
            lease_id=stale_lease,
            ttl_ms=15_000,
            task_id="task-generations",
        )
        response = client._receive({"ack", "diagnostic"})

        assert response["kind"] == "diagnostic", (
            "a stale generation lease must be refused, not silently accepted"
        )
        assert response["optional"]["lease"] == {"lease_id": stale_lease}
        diagnostic = response["diagnostic"]
        # Typed, and specifically NOT the codes that mean "your own generation,
        # but closing" or "your own generation, but an unusable lease".
        assert diagnostic["code"] == "lease_generation_mismatch"
        assert diagnostic["phase"] == "authorize"
        assert diagnostic["completion"] == "not_started"
        # The session stays usable: one bad lease is not a session kill, and the
        # generation that DOES own this session keeps renewing normally.
        client.renew("task-generations")
        assert "remote-only" in _envelope(
            client.call(
                "read_file", {"path": "README.md"}, task_id="task-generations"
            )
        )["text"]
    finally:
        try:
            client.close()
        except Exception:
            pass
