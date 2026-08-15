"""Transport-side pending-operation journal and reconciliation (RWS v2 §3.2).

Transferred from the donor's `tests/test_remote_pending_operations.py`, split
along the same transport/Home boundary as the modules: the cases here need no
`OpenSSHExecdTransport` and no Home authority, so they pin the journal and the
reconciliation reducer directly.  The donor's transport-driven cases
(fsync-before-CONTINUE ordering, a write failure blocking CONTINUE, the
callable-only refusal, ACK-cleanup failure) land with `remote_ssh` itself; the
donor's Home-recovery cases live with the hook, as noted at the bottom.
"""

from __future__ import annotations

import pathlib
from types import SimpleNamespace

import pytest

from ouroboros.remote_pending_operations import (
    load_pending_operations,
    pending_operation_groups,
    restore_transport_tracking,
    validate_transport_session_identity,
    write_pending_operation,
)
from ouroboros.remote_reconciliation import reconcile_remote_operations
from ouroboros.workspace_diagnostics import RemoteWorkspaceError


def _request(tmp_path, **overrides):
    fields = {
        "connection": {"id": "connection-1"},
        "project_id": "project-1",
        "workspace_id": "workspace-1",
        "remote_root": "/srv/project",
        "drive_root": tmp_path,
    }
    fields.update(overrides)
    return SimpleNamespace(**fields)


class _StubImporter:
    """The injected Home seam: it records calls and returns the envelope."""

    def __init__(self):
        self.calls: list[str] = []

    def complete_import(self, *, kind, context, wire_result, envelope, fetched):
        del context, wire_result, fetched
        self.calls.append(kind)
        return dict(envelope)


def _write_intent(tmp_path, **overrides):
    fields = {
        "task_id": "task-1",
        "request_id": "request-1",
        "operation_id": "operation-1",
        "prepared_hash": "a" * 64,
        "tool": "write_file",
        "import_kind": "task_result_v1",
        "import_context": {},
    }
    fields.update(overrides)
    return write_pending_operation(_request(tmp_path), **fields)


def test_pending_record_contains_no_execution_or_transport_secrets(tmp_path):
    record = _write_intent(tmp_path, tool="run_command")

    stored = {key: value for key, value in record.items() if key != "_path"}
    rendered = str(stored)
    for forbidden in (
        "prepared_token",
        "canonical_args",
        "blobs",
        "ssh_alias",
        "expected_host_id",
        "server_generation",
    ):
        assert forbidden not in rendered


def test_duplicate_operation_identity_with_different_hash_fails_closed(tmp_path):
    for prepared_hash in ("a" * 64, "b" * 64):
        _write_intent(tmp_path, prepared_hash=prepared_hash)

    with pytest.raises(
        RuntimeError,
        match="conflicting pending remote operation identity",
    ):
        restore_transport_tracking(_request(tmp_path))


def test_rewriting_the_same_identity_with_a_different_contract_fails_closed(
    tmp_path,
):
    _write_intent(tmp_path)

    with pytest.raises(
        RuntimeError,
        match="conflicting pending remote operation identity",
    ):
        _write_intent(tmp_path, tool="edit_text")


def test_restart_restores_closed_import_context_from_pending_file(tmp_path):
    record = _write_intent(tmp_path)

    known, contexts = restore_transport_tracking(_request(tmp_path))

    assert pathlib.Path(record["_path"]).name.endswith(".pending.json")
    assert known == {("request-1", "operation-1"): "a" * 64}
    assert contexts[("request-1", "operation-1")]["import_kind"] == "task_result_v1"
    assert contexts[("request-1", "operation-1")]["validator"] is None


def test_unknown_import_kind_is_refused_before_any_durable_write(tmp_path):
    with pytest.raises(ValueError, match="unknown import channels"):
        _write_intent(tmp_path, import_kind="invented_kind_v9")

    assert load_pending_operations(tmp_path) == []


def test_pending_groups_are_scoped_per_project_on_a_shared_connection(tmp_path):
    _write_intent(tmp_path)
    write_pending_operation(
        _request(tmp_path, project_id="project-2"),
        task_id="task-2",
        request_id="request-2",
        operation_id="operation-2",
        prepared_hash="b" * 64,
        tool="write_file",
        import_kind="task_result_v1",
        import_context={},
    )

    groups = pending_operation_groups(tmp_path)

    assert [group["project_id"] for group in groups] == ["project-1", "project-2"]
    assert [len(group["records"]) for group in groups] == [1, 1]


def test_mismatched_host_identity_is_rejected_before_any_pending_query(tmp_path):
    transport = SimpleNamespace(
        request=_request(
            tmp_path,
            connection={"id": "connection-1", "expected_host_id": "trusted-host"},
            capability_manifest={"manifest_sha256": "capability-1"},
        ),
        _handshake={
            "host_id": "other-host",
            "workspace_id": "workspace-1",
            "canonical_root": "/srv/project",
            "capability_hash": "capability-1",
        },
    )

    with pytest.raises(RemoteWorkspaceError) as raised:
        validate_transport_session_identity(transport)

    assert raised.value.code == "host_identity_mismatch"
    assert raised.value.phase == "bootstrap"


def test_matching_session_identity_passes_every_check(tmp_path):
    transport = SimpleNamespace(
        request=_request(
            tmp_path,
            connection={"id": "connection-1", "expected_host_id": "trusted-host"},
            capability_manifest={"manifest_sha256": "capability-1"},
        ),
        _handshake={
            "host_id": "trusted-host",
            "workspace_id": "workspace-1",
            "canonical_root": "/srv/project",
            "capability_hash": "capability-1",
        },
    )

    assert validate_transport_session_identity(transport) is None


def _unavailable_transport(
    tmp_path,
    record,
    *,
    attachment=False,
    lose_ack=False,
    importer=None,
):
    key = ("request-1", "operation-1")
    context = {
        "task_id": "task-1",
        "operation_id": "operation-1",
        "import_kind": "attachment_stage_v1" if attachment else "task_result_v1",
        "import_context": {"expected_manifest": []} if attachment else {},
        "pending_record": record,
    }
    transport = SimpleNamespace(
        request=_request(tmp_path),
        home_importer=importer if importer is not None else _StubImporter(),
        _known_operations={key: "a" * 64},
        _operation_contexts={key: context},
        fetch_blob=lambda *_args: pytest.fail("result_unavailable fetched a blob"),
    )
    sent: list[str] = []

    def send(kind, **_fields):
        sent.append(kind)
        return len(sent)

    def wait(predicate, timeout_sec=None):
        if timeout_sec is not None and lose_ack:
            raise TimeoutError("ACK was lost")
        candidates = [
            {
                "kind": "reconcile_result",
                "seq": 8,
                "request_id": "request-1",
                "operation_id": "operation-1",
                "result": {"completion": "completed", "result_unavailable": True},
            },
            {
                "kind": "ack",
                "ack_seq": 2,
                "request_id": "request-1",
                "operation_id": "operation-1",
            },
        ]
        return next(row for row in candidates if predicate(row))

    transport._send = send
    transport._wait_control = wait
    return transport, sent


@pytest.mark.parametrize("attachment", [False, True])
def test_result_unavailable_preserves_evidence_after_ack(tmp_path, attachment):
    record = _write_intent(
        tmp_path,
        tool="_stage_task_attachments" if attachment else "write_file",
        import_kind="attachment_stage_v1" if attachment else "task_result_v1",
        import_context={"expected_manifest": []} if attachment else {},
    )
    importer = _StubImporter()
    transport, sent = _unavailable_transport(
        tmp_path,
        record,
        attachment=attachment,
        importer=importer,
    )

    rows = reconcile_remote_operations(
        transport,
        ack_timeout_sec=1.0,
        retention_cap=512,
    )

    evidence = tmp_path / rows[0]["evidence_ref"]
    assert rows[0]["imported"] is True
    assert evidence.is_file()
    assert load_pending_operations(tmp_path) == []
    assert sent == ["reconcile", "ack"]
    # An attachment stage validates its own manifest; only a task result goes
    # through the injected Home importer.
    assert importer.calls == ([] if attachment else ["task_result_v1"])


def test_lost_ack_keeps_pending_alongside_terminal_evidence(tmp_path):
    record = _write_intent(tmp_path)
    transport, _sent = _unavailable_transport(tmp_path, record, lose_ack=True)

    rows = reconcile_remote_operations(
        transport,
        ack_timeout_sec=1.0,
        retention_cap=512,
    )

    assert (tmp_path / rows[0]["evidence_ref"]).is_file()
    assert len(load_pending_operations(tmp_path)) == 1


def test_terminal_evidence_retention_never_prunes_another_pending_intent(tmp_path):
    first = _write_intent(tmp_path)
    write_pending_operation(
        _request(tmp_path),
        task_id="task-2",
        request_id="request-2",
        operation_id="operation-2",
        prepared_hash="b" * 64,
        tool="write_file",
        import_kind="task_result_v1",
        import_context={},
    )
    transport, _sent = _unavailable_transport(tmp_path, first)

    reconcile_remote_operations(transport, ack_timeout_sec=1.0, retention_cap=1)

    pending = load_pending_operations(tmp_path)
    assert [row["operation_id"] for row in pending] == ["operation-2"]


def test_missing_home_importer_blocks_the_ack_instead_of_importing_locally(
    tmp_path,
):
    record = _write_intent(tmp_path)
    transport, sent = _unavailable_transport(tmp_path, record)
    transport.home_importer = None

    rows = reconcile_remote_operations(
        transport,
        ack_timeout_sec=1.0,
        retention_cap=512,
    )

    assert rows[0]["imported"] is False
    assert rows[0]["import_error"] == "RuntimeError"
    assert sent == ["reconcile"]
    assert len(load_pending_operations(tmp_path)) == 1


def test_proven_not_started_drops_the_intent_without_importing(tmp_path):
    record = _write_intent(tmp_path)
    key = ("request-1", "operation-1")
    transport = SimpleNamespace(
        request=_request(tmp_path),
        home_importer=_StubImporter(),
        _known_operations={key: "a" * 64},
        _operation_contexts={
            key: {
                "task_id": "task-1",
                "operation_id": "operation-1",
                "import_kind": "task_result_v1",
                "import_context": {},
                "pending_record": record,
            }
        },
    )
    sent: list[str] = []
    transport._send = lambda kind, **_fields: (sent.append(kind), len(sent))[1]
    transport._wait_control = lambda predicate, timeout_sec=None: {
        "kind": "reconcile_result",
        "seq": 3,
        "request_id": "request-1",
        "operation_id": "operation-1",
        "result": {"completion": "not_started"},
    }

    rows = reconcile_remote_operations(
        transport,
        ack_timeout_sec=1.0,
        retention_cap=512,
    )

    assert rows[0]["completion"] == "not_started"
    assert sent == ["reconcile"]
    assert load_pending_operations(tmp_path) == []
    assert transport._known_operations == {}


# The donor's Home-recovery cases
# (`test_broker_startup_reopens_scope_before_reconciliation`,
# `test_retired_recovery_session_is_closed_after_reconciliation`) drive the HOOK,
# `remote_transfer.recover_pending_scopes`, which resolves a pending group against
# the connection store before deciding whether the scope may be reopened at all.
# They live with it: see `tests/test_remote_broker_lifecycle.py`
# (`test_a_pending_scope_whose_connection_is_gone_is_retained_and_reported`,
# `test_an_injected_recovery_hook_owns_reconciliation`). What is pinned HERE is the
# durable journal the hook reads.
