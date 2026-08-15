"""The OpenSSH execd transport: durability ordering, and the alias it refuses.

Two halves, matching the commit-2 pre-split:

* `OpenSSHExecdTransport` — the donor cases that need a real transport object:
  the durable intent is fsynced BEFORE `continue` and dropped only after `ack`,
  a journal write failure blocks `continue` entirely, a mutation with no closed
  import contract never reaches the wire, and a result that fails verification
  is never ACKed.  These complete the split started in
  `tests/test_remote_pending_operations.py`.
* `remote_ssh_config` — what `ssh -G` must not resolve to.  A fake `ssh` binary
  plays the effective-configuration probe, so hostile aliases are exercised
  without a network or a real sshd.
"""

from __future__ import annotations

import hashlib
import pathlib
from types import MethodType, SimpleNamespace

import pytest

from ouroboros.remote_pending_operations import load_pending_operations
from ouroboros.remote_ssh import OpenSSHExecdTransport
from ouroboros.remote_ssh_config import (
    validated_ssh_base_command,
    validated_ssh_config,
)
from ouroboros.workspace_diagnostics import RemoteWorkspaceError


# ── transport durability ordering ───────────────────────────────────────


def _request(tmp_path):
    return SimpleNamespace(
        connection={"id": "connection-1", "ssh_alias": "build"},
        project_id="project-1",
        workspace_id="workspace-1",
        remote_root="/srv/project",
        drive_root=tmp_path,
        server_generation="generation-1",
        capability_manifest={"manifest_sha256": "capability-1"},
        ssh_binary=None,
    )


class _StubImporter:
    def complete_import(self, *, kind, context, wire_result, envelope, fetched):
        del kind, context, wire_result, fetched
        return dict(envelope)


def _transport(tmp_path, events, *, result=None, blobs=None):
    """A transport wired to fakes for everything below the protocol layer.

    Constructed with `object.__new__` on purpose: `__init__` runs `ssh -G`, and
    what is under test here is the ordering of durable writes against the wire,
    not session startup.
    """

    transport = object.__new__(OpenSSHExecdTransport)
    transport.request = _request(tmp_path)
    transport.home_importer = _StubImporter()
    transport._known_operations = {("request-1", "operation-1"): "a" * 64}
    transport._operation_contexts = {
        ("request-1", "operation-1"): {
            "task_id": "task-1",
            "operation_id": "operation-1",
            "tool": "write_file",
            "validator": None,
            "pending_record": None,
        }
    }
    transport._ensure_session = MethodType(lambda _self: None, transport)
    transport._renew_lease = MethodType(lambda _self, _task_id: None, transport)
    transport._raise_diagnostic = MethodType(lambda _self, _row: None, transport)
    payloads = dict(blobs or {})

    def _fetch(_self, blob_id, max_bytes):
        events.append(f"fetch:{blob_id}")
        payload = payloads[blob_id]
        assert len(payload) <= max_bytes
        return payload

    transport.fetch_blob = MethodType(_fetch, transport)
    sequence = {"value": 0}

    def _send(_self, kind, **_fields):
        sequence["value"] += 1
        events.append(kind)
        if kind == "continue":
            # The whole point of the ordering: by the time CONTINUE is on the
            # wire, the intent is already durable.
            assert len(load_pending_operations(tmp_path)) == 1
        return sequence["value"]

    transport._send = MethodType(_send, transport)
    wire_result = result if result is not None else {
        "completion": "completed",
        "prepared_hash": "a" * 64,
        "envelope": {
            "text": "ok",
            "diagnostic": None,
            "process": None,
            "artifacts": [],
            "trace": {"completion": "complete"},
        },
        "output_blobs": {},
    }

    def _wait(_self, predicate, timeout_sec=None):
        del timeout_sec
        candidates = [
            {
                "kind": "result",
                "seq": 11,
                "request_id": "request-1",
                "operation_id": "operation-1",
                "result": wire_result,
            },
            {
                "kind": "ack",
                "ack_seq": sequence["value"],
                "request_id": "request-1",
                "operation_id": "operation-1",
            },
        ]
        return next(row for row in candidates if predicate(row))

    transport._wait_control = MethodType(_wait, transport)
    return transport


def _execute(transport, **overrides):
    message = {
        "request_id": "request-1",
        "operation_id": "operation-1",
        "prepared_hash": "a" * 64,
        "prepared_token": "secret-prepared-token",
        "task_id": "task-1",
        "_home_import_kind": "task_result_v1",
        "_home_import_context": {},
    }
    message.update(overrides)
    return transport.execute_prepared(message)


def test_pending_record_is_fsynced_before_continue_and_removed_after_ack(tmp_path):
    events: list[str] = []
    transport = _transport(tmp_path, events)

    result = _execute(transport)

    assert result["text"] == "ok"
    assert events.index("continue") < events.index("ack")
    assert load_pending_operations(tmp_path) == []


def test_pending_write_failure_prevents_continue(tmp_path, monkeypatch):
    events: list[str] = []
    transport = _transport(tmp_path, events)
    monkeypatch.setattr(
        "ouroboros.remote_ssh.bind_transport_intent",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("disk full")),
    )

    with pytest.raises(OSError, match="disk full"):
        _execute(transport)

    assert "continue" not in events


def test_callable_only_import_cannot_be_persisted_before_continue(tmp_path):
    events: list[str] = []
    transport = _transport(tmp_path, events)

    with pytest.raises(ValueError, match="durable remote import kind"):
        transport.execute_prepared({
            "request_id": "request-1",
            "operation_id": "operation-1",
            "prepared_hash": "a" * 64,
            "prepared_token": "prepared-1",
            "task_id": "task-1",
            "_home_completion_validator": (
                lambda _wire, envelope, _fetched: dict(envelope)
            ),
        })

    assert "continue" not in events


def test_every_remote_continue_requires_a_closed_home_import_contract(tmp_path):
    events: list[str] = []
    transport = _transport(tmp_path, events)

    with pytest.raises(ValueError):
        transport.execute_prepared({
            "request_id": "request-1",
            "operation_id": "operation-1",
            "prepared_hash": "a" * 64,
            "prepared_token": "prepared-1",
            "task_id": "task-1",
        })

    assert "continue" not in events
    assert "ack" not in events


def test_ack_cleanup_failure_keeps_tracking_but_returns_imported_result(
    tmp_path,
    monkeypatch,
):
    events: list[str] = []
    transport = _transport(tmp_path, events)
    monkeypatch.setattr(
        "ouroboros.remote_ssh.remove_transport_pending",
        lambda _context: False,
    )

    result = _execute(transport)

    assert result["text"] == "ok"
    assert ("request-1", "operation-1") in transport._known_operations
    assert len(load_pending_operations(tmp_path)) == 1


def test_a_prepared_identity_mismatch_never_reaches_the_wire(tmp_path):
    events: list[str] = []
    transport = _transport(tmp_path, events)

    with pytest.raises(RemoteWorkspaceError) as raised:
        _execute(transport, prepared_hash="b" * 64)

    assert raised.value.code == "prepared_identity_mismatch"
    assert raised.value.completion == "not_started"
    assert events == []


def test_blob_integrity_failure_prevents_ack(tmp_path):
    stdout = b"x" * 70_001
    identity = hashlib.sha256(stdout).hexdigest()
    result = {
        "completion": "completed",
        "prepared_hash": "a" * 64,
        "envelope": {
            "text": "preview",
            "diagnostic": None,
            "process": {
                "returncode": 0,
                "stdout": "preview",
                "stderr": "",
                "backend_trace": {},
                "args": [],
            },
            "artifacts": [
                {
                    "name": "stdout.txt",
                    "blob_id": identity,
                    "sha256": identity,
                    "size": len(stdout),
                    "mime": "text/plain",
                    "truncated": False,
                }
            ],
            "trace": {},
        },
        "output_blobs": {identity: identity},
    }
    events: list[str] = []
    transport = _transport(
        tmp_path,
        events,
        result=result,
        blobs={identity: b"z" * len(stdout)},
    )

    with pytest.raises(RemoteWorkspaceError) as raised:
        _execute(transport)

    assert raised.value.code == "remote_result_import_failed"
    assert raised.value.phase == "import"
    # The mutation DID complete remotely; saying otherwise would license a retry.
    assert raised.value.completion == "completed"
    assert "ack" not in events
    assert len(load_pending_operations(tmp_path)) == 1


# ── blob transfer: a per-blob failure stays per-blob ─────────────────────


def test_a_blob_that_stalls_mid_transfer_does_not_take_the_session_with_it(
    tmp_path, monkeypatch
):
    """A timed-out fetch must leave the transport able to fetch the next blob.

    `_download_current` is the latch that says "bulk frames belong to this blob". It
    was raised on the manifest and lowered only on a COMPLETED transfer or a full wire
    reset — never on the timeout in between. So a target that hung after sending the
    manifest left it raised, and the next, unrelated fetch tripped the overlap check
    inside the READER THREAD, where the exception becomes `_reader_error` and every
    later wait on the session answers `ssh_session_disconnected`. The blob's own
    failure is typed `retryable=True`; the session's is not, and it fails every other
    operation riding that transport — including, on the path this is reached from, one
    that had already COMPLETED on the target.
    """

    import threading

    from ouroboros import remote_ssh

    monkeypatch.setattr(remote_ssh, "_SESSION_TIMEOUT_SEC", 0.05)
    payload = b"the second blob's bytes"
    sent: list[str] = []

    transport = object.__new__(OpenSSHExecdTransport)
    transport.request = _request(tmp_path)
    transport._downloads = {}
    transport._download_current = ""
    transport._download_draining = ""
    transport._download_lock = threading.Lock()
    transport._condition = threading.Condition()
    transport._ensure_session = MethodType(lambda _self: None, transport)

    def _send(_self, kind, **fields):
        """The TARGET's half: answer a fetch with a manifest, and maybe the bytes."""
        sent.append(kind)
        if kind != "blob_fetch":
            return len(sent)
        blob_id = str(fields["blob_id"])
        _self._receive_manifest(
            {
                "blob_id": blob_id,
                "request_id": fields["request_id"],
                "size": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
        # `blob-A` is the stall: the manifest arrives and the bytes never do.
        if blob_id != "blob-A":
            _self._receive_bulk(payload)
        return len(sent)

    transport._send = MethodType(_send, transport)

    with pytest.raises(RemoteWorkspaceError) as raised:
        transport.fetch_blob("blob-A", 1024)
    assert raised.value.code == "remote_blob_timeout"
    assert raised.value.retryable is True
    assert transport._download_current == "", (
        "the abandoned blob left its latch raised, so the next manifest overlaps"
    )

    # The target was still streaming when Home walked away. Those chunks are a race
    # Home created, not a desynced wire, so they are dropped rather than raising.
    transport._receive_bulk(b"late chunk for blob-A")

    assert transport.fetch_blob("blob-B", 1024) == payload


# ── effective OpenSSH configuration ─────────────────────────────────────


def _fake_ssh(tmp_path: pathlib.Path, resolved: str, *, overridden: str = "") -> str:
    """A fake `ssh` that answers `-G` differently with and without our options.

    `resolved` is what the OWNER's alias resolves to (the raw probe); `overridden`
    is what survives our fixed `-o` overrides. Distinguishing the two is the
    point of the double probe.
    """

    script = tmp_path / "ssh"
    script.write_text(
        "#!/bin/sh\n"
        'for arg in "$@"; do\n'
        '  if [ "$arg" = "-o" ]; then overridden=1; fi\n'
        "done\n"
        'if [ -n "${overridden:-}" ]; then\n'
        f"  cat <<'EOF'\n{overridden or resolved}\nEOF\n"
        "else\n"
        f"  cat <<'EOF'\n{resolved}\nEOF\n"
        "fi\n",
        encoding="utf-8",
    )
    script.chmod(0o755)
    return str(script)


_CLEAN = "\n".join([
    "host build",
    "requesttty false",
    "remotecommand none",
    "clearallforwardings yes",
    "tunnel none",
])


def test_a_clean_alias_yields_an_argv_ending_in_the_alias(tmp_path):
    command = validated_ssh_base_command(
        "build", _fake_ssh(tmp_path, _CLEAN), forwarding=False
    )

    assert command[-1] == "build"
    assert "-T" in command
    # The channel's own fixed overrides must be present regardless of the alias.
    for option in (
        "BatchMode=yes",
        "ForwardAgent=no",
        "ForwardX11=no",
        "PermitLocalCommand=no",
        "RemoteCommand=none",
        "ClearAllForwardings=yes",
    ):
        assert option in command


@pytest.mark.parametrize(
    "resolved, overridden, code",
    [
        # A RemoteCommand that survives the override would eat the frames.
        (_CLEAN, _CLEAN.replace("remotecommand none", "remotecommand /bin/sh"),
         "unsupported_ssh_client"),
        # A forced TTY cannot carry binary frames at all.
        (_CLEAN, _CLEAN.replace("requesttty false", "requesttty force"),
         "unsupported_ssh_client"),
        # SetEnv is refused outright: `SetEnv=-*` provably does not cancel it.
        (_CLEAN + "\nsetenv OUROBOROS_NETWORK_PASSWORD=x", "",
         "unsafe_ssh_environment"),
        # SendEnv matching a key we retain would forward a Home value.
        (_CLEAN + "\nsendenv PATH", "", "unsafe_ssh_environment"),
        # A wildcard reaches the same retained keys.
        (_CLEAN + "\nsendenv *", "", "unsafe_ssh_environment"),
    ],
)
def test_a_hostile_alias_is_refused_before_spawn(
    tmp_path, resolved, overridden, code
):
    with pytest.raises(RemoteWorkspaceError) as raised:
        validated_ssh_config(
            "build",
            _fake_ssh(tmp_path, resolved, overridden=overridden),
            forwarding=False,
        )

    assert raised.value.code == code
    assert raised.value.phase == "connect"


def test_a_negated_sendenv_pattern_is_inert(tmp_path):
    resolved = _CLEAN + "\nsendenv PATH -PATH"

    command, _config = validated_ssh_config(
        "build", _fake_ssh(tmp_path, resolved), forwarding=False
    )

    assert command[-1] == "build"


def test_alias_forwarding_is_neutralized_and_disclosed_for_the_protocol(tmp_path):
    resolved = _CLEAN + "\nlocalforward 8080 127.0.0.1:8080"

    _command, config = validated_ssh_config(
        "build", _fake_ssh(tmp_path, resolved), forwarding=False
    )

    # Neutralized, not silently dropped — the owner asked for something we
    # disabled and that belongs in the diagnostics they can read.
    assert config["_ouroboros_warning_directives"] == ["localforward"]


def test_alias_forwarding_is_refused_outright_for_a_deliberate_forward(tmp_path):
    resolved = _CLEAN + "\ndynamicforward 1080"

    with pytest.raises(RemoteWorkspaceError) as raised:
        validated_ssh_config(
            "build", _fake_ssh(tmp_path, resolved), forwarding=True
        )

    assert raised.value.code == "unsafe_ssh_forwarding"


def test_forwarding_that_survives_the_override_fails_closed(tmp_path):
    resolved = _CLEAN + "\nremoteforward 9000 127.0.0.1:9000"
    overridden = resolved.replace("clearallforwardings yes", "clearallforwardings no")

    with pytest.raises(RemoteWorkspaceError) as raised:
        validated_ssh_config(
            "build",
            _fake_ssh(tmp_path, resolved, overridden=overridden),
            forwarding=False,
        )

    assert raised.value.code == "unsafe_ssh_forwarding"


def test_an_unresolvable_alias_reports_bounded_stderr(tmp_path):
    script = tmp_path / "ssh"
    script.write_text(
        "#!/bin/sh\necho 'ssh: Could not resolve hostname nope' >&2\nexit 255\n",
        encoding="utf-8",
    )
    script.chmod(0o755)

    with pytest.raises(RemoteWorkspaceError) as raised:
        validated_ssh_config("nope", str(script), forwarding=False)

    assert raised.value.code == "unsupported_ssh_client"
    assert "Could not resolve hostname" in raised.value.details["stderr"]


def test_the_child_environment_is_constructed_not_inherited(monkeypatch):
    from ouroboros.remote_ssh_config import SSH_ENV_KEYS, minimal_ssh_env

    monkeypatch.setenv("PATH", "/usr/bin")
    monkeypatch.setenv("OPENROUTER_API_KEY", "must-not-travel")
    monkeypatch.setenv("OUROBOROS_NETWORK_PASSWORD", "must-not-travel")

    env = minimal_ssh_env()

    assert set(env) <= set(SSH_ENV_KEYS)
    assert "must-not-travel" not in "".join(env.values())


# ── the structural gate ─────────────────────────────────────────────────


def test_the_transport_is_gated_but_never_shipped_in_the_bundle():
    """Both directions of the gate, and the distinction between them.

    `remote_ssh` is a SEED of the import-closure gate (it must not reach a Home
    policy authority) while deliberately NOT being a bundle module — it runs on
    Home. Conflating the two would either ship the transport to the target or
    stop gating it.
    """

    from ouroboros.tool_capabilities import (
        REMOTE_NATIVE_CLOSURE_SEEDS,
        assert_remote_native_import_closure,
    )
    from ouroboros.workspace_native_contract import REMOTE_NATIVE_KERNEL_MODULES

    for module in ("ouroboros.remote_ssh", "ouroboros.remote_ssh_config"):
        assert module in REMOTE_NATIVE_CLOSURE_SEEDS
        assert module not in REMOTE_NATIVE_KERNEL_MODULES

    audit = assert_remote_native_import_closure(pathlib.Path(__file__).parent.parent)
    assert audit["forbidden"] == {}
    # The split halves ride in through the transport and must stay Home-free too.
    for module in (
        "ouroboros.remote_pending_operations",
        "ouroboros.remote_reconciliation",
        "ouroboros.remote_ssh_bootstrap",
    ):
        assert module in audit["modules"]
