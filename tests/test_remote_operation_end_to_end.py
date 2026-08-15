"""One remote operation, end to end over a fake transport (RWS v2 synthesis).

Every lane tested its own half. This drives the WHOLE path — admit → prepare →
authorize over target facts → execute → import → published artifact — because the
defects the rebuild exists to prevent live in the joins, not in the halves:

* **§3.1 structural RPC bound.** ONE bundled prepare per operation. A per-fact RPC
  is not a performance question: it is what made the donor's remote path
  unaffordable and therefore what made mirroring guards tempting. Counted, not
  asserted by inspection.
* **The guards read TARGET facts.** The authorize phase's fact door answers from
  the prepare bundle and refuses anything the bundle does not carry — never a Home
  probe, which would answer about the wrong filesystem AND succeed.
* **D9 — one public identity.** The model-facing text and the published artifact
  record carry no target-native path.
* **The seams meet.** The `ExecutorRef` projection round-trips to the same sealed
  placement descriptor admission wrote, so the broker is addressed by the ref that
  was sealed and not by a second, look-alike one.
"""

from __future__ import annotations

import hashlib
import json
import time

import pytest

from ouroboros import workspace_executor
from ouroboros.execution_facts import RemoteFactsUnavailableError, facts_for_ref
from ouroboros.remote_workspace import RemoteSessionBroker, set_remote_workspace_service
from ouroboros.workspace_native_contract import (
    NATIVE_FACT_GIT_TOPLEVELS,
    NATIVE_FACT_INTERPRETERS,
    NATIVE_FACT_PATH_STATS,
)
from ouroboros.workspace_ref import SshWorkspaceRef

pytestmark = pytest.mark.serial

_ROOT = "/srv/work/app"
_MANIFEST = {
    "schema_version": 1,
    "manifest_sha256": "a" * 64,
    "public_schema_sha256": "c" * 64,
    "native_operations": [{"name": "read_file"}],
    "native_kernel_modules": ["ouroboros.workspace_native"],
    "native_import_modules": ["ouroboros.workspace_native"],
    "native_import_edges": {},
}
_REF = SshWorkspaceRef(connection_id="connection-1", remote_root=_ROOT, workspace_id="workspace-1")
_CONNECTION = {"id": "connection-1", "ssh_alias": "build"}


class _CountingTransport:
    """A transport that COUNTS what Home asked of it.

    The counters are the assertion: a bundled prepare that quietly became several
    round trips would still return correct facts, and only a count can see it.
    """

    def __init__(self, request, *, home_importer=None):
        self.request = request
        self.home_importer = home_importer
        self.calls: list[str] = []
        self.stdout = ("out\n" + "x" * 70_000).encode("utf-8")

    def _record(self, name):
        self.calls.append(name)

    def handshake(self):
        self._record("handshake")
        return {
            "host_id": "host-1",
            "workspace_id": "workspace-1",
            "canonical_root": _ROOT,
            "capability_hash": _MANIFEST["manifest_sha256"],
            "platform": {"system": "linux", "python": "3.11.5"},
        }

    def artifact_identity(self):
        return {}

    def prepare(self, message, blobs):
        del blobs
        self._record("prepare")
        # The block the TARGET fills. The rows are fabricated on purpose: the real
        # producer (`workspace_native_contract.bundle_prepared_facts`) STATS the
        # target's own filesystem, and this root exists on the target, not here.
        # What is under test is the shape and the contract KEY NAMES, which come
        # from the contract module rather than from string literals.
        facts = {
            "workspace_root": _ROOT,
            "resolved_cwd": _ROOT,
            NATIVE_FACT_PATH_STATS: {
                _ROOT: {"canonical": _ROOT, "kind": "dir", "symlink": False, "size": 4096},
            },
            NATIVE_FACT_GIT_TOPLEVELS: {_ROOT: _ROOT},
            NATIVE_FACT_INTERPRETERS: {"python3": f"{_ROOT}/.venv/bin/python"},
        }
        return {
            "request_id": message["request_id"],
            "operation_id": message["operation_id"],
            "tool": message["tool"],
            "prepared_token": "token-1",
            "prepared_hash": "b" * 64,
            "expires_at_ms": int(time.time() * 1000) + 60_000,
            "execution_args": {"cmd": ["python3", "-c", "print()"], "cwd": _ROOT},
            "native_facts": facts,
        }

    def execute_prepared(self, message):
        self._record("execute_prepared")
        digest = hashlib.sha256(self.stdout).hexdigest()
        return {
            "text": "ok",
            "diagnostic": None,
            "process": {
                "returncode": 0,
                "stdout": "preview",
                "stderr": "",
                "backend_trace": {"backend": "ssh_exec", "cwd": _ROOT},
                "args": ["python3", "-c", "print()"],
            },
            "artifacts": [{
                "name": "stdout.txt",
                "blob_id": digest,
                "sha256": digest,
                "size": len(self.stdout),
                "mime": "text/plain",
                "truncated": False,
            }],
            "trace": {"backend": "ssh_exec", "task_id": str(message.get("task_id") or "")},
        }

    def fetch_blob(self, blob_id, max_bytes):
        del max_bytes
        self._record("fetch_blob")
        return self.stdout if blob_id == hashlib.sha256(self.stdout).hexdigest() else b""

    def reconcile(self):
        return []

    def cancel(self, _message):
        return True

    def task_lease(self, _task_id, forget=False):
        del forget
        return False

    def health(self):
        return {"status": "ready", "phase": "ready"}

    def panic(self):
        pass

    def close(self):
        self._record("close")


@pytest.fixture()
def wired(tmp_path, monkeypatch):
    """A live broker with a counting transport, registered as THE service."""

    transports: list[_CountingTransport] = []

    def factory(request, *, home_importer=None):
        transport = _CountingTransport(request, home_importer=home_importer)
        transports.append(transport)
        return transport

    broker = RemoteSessionBroker(
        tmp_path, "generation-1", _MANIFEST, transport_factory=factory
    )
    broker.start()
    set_remote_workspace_service(broker)
    try:
        yield broker, transports, tmp_path
    finally:
        set_remote_workspace_service(None)
        broker.close(timeout_sec=2)


def _admit(broker, task_id="task-1"):
    return broker.admit_workspace(
        _CONNECTION,
        remote_root=_ROOT,
        project_id="project-1",
        workspace_id="workspace-1",
        task_id=task_id,
    )


# ── the whole path ───────────────────────────────────────────────────────────


def test_one_remote_operation_prepares_once_authorizes_on_target_facts_and_publishes(wired):
    broker, transports, drive_root = wired
    admitted = _admit(broker)
    # The placement the broker derived is the one admission would seal.
    assert admitted["workspace_ref"] == _REF.to_payload()

    executor = workspace_executor.ExecutorRef(
        kind="ssh",
        executor_id=_REF.workspace_id,
        network="host",
        mappings=(),
        connection_id=_REF.connection_id,
        remote_root=_REF.remote_root,
    )
    # The projection round-trips: the broker is addressed by the SEALED descriptor.
    assert workspace_executor.ssh_workspace_ref_payload(executor) == _REF.to_payload()

    prepared = workspace_executor.prepare_native_operation(
        executor, "run_command", args={"cmd": ["python3", "-c", "print()"]}, task_id="task-1"
    )
    transport = transports[0]
    # §3.1: exactly ONE prepare, and no per-fact chatter around it.
    assert transport.calls.count("prepare") == 1

    # AUTHORIZE reads the bundle, and only the bundle.
    facts = facts_for_ref(_REF, prepared.native_facts)
    assert facts.canonical_path(_ROOT) == _ROOT
    assert facts.path_fact(_ROOT).is_dir
    assert facts.git_fact(_ROOT).is_worktree_root
    assert facts.interpreter_fact("python3").resolved == f"{_ROOT}/.venv/bin/python"
    # A fact the target did not declare is a typed refusal, NOT a Home probe.
    with pytest.raises(RemoteFactsUnavailableError):
        facts.path_fact("/etc/passwd")
    # Still exactly one prepare: every fact above was a lookup.
    assert transport.calls.count("prepare") == 1

    envelope = workspace_executor.execute_prepared(executor, prepared, task_id="task-1")
    # The facade returns the target's ENVELOPE: its `text` is the operation's
    # model-facing answer (the dispatcher returns it verbatim), and the process
    # evidence rides alongside rather than replacing it.
    assert envelope.text == "ok"
    assert envelope.process.returncode == 0
    assert transport.calls.count("execute_prepared") == 1

    # IMPORT the declared process output through the Home half.
    from ouroboros.remote_reconciliation import prefetch_remote_result_import
    from ouroboros.remote_transfer import RemoteTransferService

    stored = {
        "completion": "completed",
        "prepared_hash": "b" * 64,
        "envelope": transport.execute_prepared({"task_id": "task-1"}),
    }
    envelope, fetched = prefetch_remote_result_import(stored, transport.fetch_blob)
    public = RemoteTransferService().complete_import(
        kind="task_result_v1",
        context={
            "drive_root": str(drive_root),
            "task_id": "t-1",
            "operation_id": prepared.operation_id,
            "connection_id": _REF.connection_id,
            "workspace_id": _REF.workspace_id,
        },
        wire_result=stored,
        envelope=envelope,
        fetched=fetched,
    )

    # D9: the model-facing result names a Home artifact and no target path.
    published = [row for row in public["artifacts"] if row.get("home_ref")]
    assert [row["name"] for row in published] == ["stdout.txt"]
    assert published[0]["home_ref"]["root"] == "artifact_store"
    assert _ROOT not in json.dumps(public["artifacts"])
    assert _ROOT not in public["text"]
    # The published bytes are really on Home, at the hash the record claims.
    found = list(drive_root.rglob(published[0]["home_ref"]["path"]))
    assert found and hashlib.sha256(found[0].read_bytes()).hexdigest() == published[0]["sha256"]


def test_the_bundle_is_gathered_once_even_when_many_facts_are_read(wired):
    """The RPC bound is structural: N facts, still one prepare."""

    broker, transports, _drive = wired
    _admit(broker)
    executor = workspace_executor.ExecutorRef(
        kind="ssh", executor_id="workspace-1", network="host", mappings=(),
        connection_id="connection-1", remote_root=_ROOT,
    )
    prepared = workspace_executor.prepare_native_operation(executor, "run_command", task_id="task-1")
    facts = facts_for_ref(_REF, prepared.native_facts)
    for _ in range(50):
        facts.path_fact(_ROOT)
        facts.git_fact(_ROOT)
        facts.interpreter_fact("python3")
    assert transports[0].calls.count("prepare") == 1
    assert transports[0].calls.count("handshake") == 1


def test_the_fact_block_carries_the_three_declared_keys_and_nothing_implicit(wired):
    """The wire names are a contract: renaming one silently blinds every guard."""

    broker, transports, _drive = wired
    _admit(broker)
    executor = workspace_executor.ExecutorRef(
        kind="ssh", executor_id="workspace-1", network="host", mappings=(),
        connection_id="connection-1", remote_root=_ROOT,
    )
    prepared = workspace_executor.prepare_native_operation(executor, "read_file", task_id="task-1")
    for key in (NATIVE_FACT_PATH_STATS, NATIVE_FACT_GIT_TOPLEVELS, NATIVE_FACT_INTERPRETERS):
        assert key in prepared.native_facts, key


def test_an_accessor_built_before_prepare_refuses_every_fact():
    """The pre-prepare state is a refusal, so no guard can run on absent facts."""

    facts = facts_for_ref(_REF)
    for probe in (
        lambda: facts.canonical_path(_ROOT),
        lambda: facts.path_fact(_ROOT),
        lambda: facts.git_fact(_ROOT),
        lambda: facts.interpreter_fact("python3"),
    ):
        with pytest.raises(RemoteFactsUnavailableError):
            probe()


def test_finishing_the_task_releases_the_lease_and_the_staging(wired):
    from ouroboros.remote_workspace import finish_remote_task
    from ouroboros.workspace_ref import SEALED_WORKSPACE_REF_KEY

    broker, _transports, drive_root = wired
    _admit(broker)
    staging = drive_root / "remote_imports" / "task-1"
    staging.mkdir(parents=True, exist_ok=True)
    assert broker.has_active_lease("connection-1") is True

    assert finish_remote_task(
        {"drive_root": str(drive_root), "metadata": {SEALED_WORKSPACE_REF_KEY: _REF.to_payload()}},
        "task-1",
    )
    assert broker.has_active_lease("connection-1") is False
    assert not staging.exists()
