"""`verify_and_record` on a remote workspace (RWS v2 §3.3, R1 remnant).

The tool used to refuse typed on a remote placement, which left a remote task unable to
verify anything. The correct shape splits the tool rather than routing or refusing it:
the check RUNS on the target (a Home run would verify the wrong filesystem) and the
durable receipt is written on HOME (a target-side check with no Home receipt is a
verification whose proof disappears with the session).

`bytes_equal` is compared ON the target on purpose — comparing on Home would transfer
both files in full for a fact that is one boolean plus a bounded divergence window.

The target here is the real `workspace_native` kernel, so the returncode, the byte
comparison and the after-check existence probes are the genuine ones.
"""
from __future__ import annotations

import subprocess

import pytest

from ouroboros.workspace_ref import SEALED_WORKSPACE_REF_KEY


@pytest.fixture
def target_repo(tmp_path):
    root = tmp_path / "srv" / "app"
    root.mkdir(parents=True)
    (root / "golden.txt").write_bytes(b"AAAA")
    (root / "actual.txt").write_bytes(b"AAAA")
    (root / "diverged.txt").write_bytes(b"AAAB")
    subprocess.run(["git", "init", "-q"], cwd=str(root), check=True)
    return root


class _Ctx:
    def __init__(self, tmp_path, target_repo):
        self.task_id = "task-verify"
        self.drive_root = tmp_path / "drive"
        self.repo_dir = tmp_path / "repo"
        self.repo_dir.mkdir(parents=True, exist_ok=True)
        self.task_metadata = {
            SEALED_WORKSPACE_REF_KEY: {
                "kind": "ssh", "connection_id": "conn-1",
                "remote_root": str(target_repo), "workspace_id": "ws-1",
            }
        }


class _Control:
    """The execd-side custody/cancellation seam, minimal and permissive."""

    def cancelled(self):
        return False

    def register_process(self, **kwargs):
        del kwargs

    def release_process(self, **kwargs):
        del kwargs

    def recover_service(self, **kwargs):
        del kwargs
        return None


class _Target:
    """Prepare/execute the REAL `verify_remote_check` native operation."""

    def __init__(self, root):
        self.root = root
        self.operations: list[str] = []

    def prepare(self, ref, *, tool, args, blobs, task_id, **_kw):
        from ouroboros.remote_workspace import PreparedRemoteCall
        from ouroboros.workspace_native import prepare_native_operation

        self.operations.append(f"prepare:{tool}")
        prepared = prepare_native_operation(self.root, tool, dict(args), task_id=task_id)
        self._args = dict(prepared.execution_args)
        return PreparedRemoteCall(
            request_id="r", operation_id=f"op-{tool}", tool=tool,
            prepared_token="tok", prepared_hash="0" * 64, expires_at_ms=1 << 62,
            execution_args=dict(prepared.execution_args),
            native_facts=dict(prepared.native_facts),
        )

    def execute_prepared(self, ref, prepared, *, canonical_args, task_id, **_kw):
        from ouroboros.remote_worker_proxy import envelope_from_dict
        from ouroboros.workspace_native import execute_native_operation

        self.operations.append(f"execute:{prepared.tool}")
        result = execute_native_operation(
            self.root, prepared.tool, dict(canonical_args),
            native_facts=dict(prepared.native_facts),
            control=_Control(),
        )
        envelope = result.envelope
        return envelope_from_dict({
            "text": envelope.text,
            "diagnostic": None,
            "process": (
                {
                    "returncode": envelope.process.returncode,
                    "stdout": envelope.process.stdout,
                    "stderr": envelope.process.stderr,
                    "args": list(envelope.process.args or ()),
                    "backend_trace": dict(envelope.process.backend_trace or {}),
                }
                if envelope.process is not None else None
            ),
            "artifacts": [dict(r) for r in envelope.artifacts or ()],
            "trace": dict(envelope.trace or {}),
        })

    def abort_prepared(self, *_a, **_kw):
        return True


@pytest.fixture
def remote(tmp_path, target_repo, monkeypatch):
    ctx = _Ctx(tmp_path, target_repo)
    target = _Target(target_repo)
    monkeypatch.setattr(
        "ouroboros.workspace_executor._remote_service", lambda executor, phase: target
    )
    return ctx, target, target_repo


def _receipts(ctx) -> list[dict]:
    from ouroboros.outcomes import read_verification_receipts

    rows = read_verification_receipts(ctx.drive_root, ctx.task_id)
    if not rows:
        raise IndexError("no verification receipt was written")
    return rows


def _verify(ctx, **kwargs) -> str:
    from ouroboros.tools.verify import _verify_and_record

    return _verify_and_record(ctx, **kwargs)


# ── the check runs on the target, the receipt lands on Home ──────────────────
def test_a_passing_check_runs_on_the_target_and_records_on_home(remote):
    ctx, target, _root = remote

    message = _verify(
        ctx,
        contract_kind="explicit_command",
        criterion_id="c1",
        check=["sh", "-c", "test -f golden.txt"],
    )

    assert "PASS" in message and "REMOTE workspace" in message
    assert target.operations == ["prepare:verify_remote_check", "execute:verify_remote_check"]
    receipt = _receipts(ctx)[-1]
    assert receipt["status"] == "pass" and receipt["returncode"] == 0
    # The surface is RECORDED, so a remote green and a Home green are never silently
    # treated as the same evidence.
    assert receipt["execution_surface"] == "remote_target"
    assert receipt["check"] == "sh -c 'test -f golden.txt'"


def test_a_failing_check_records_a_fail_rather_than_nothing(remote):
    ctx, _target, _root = remote

    message = _verify(
        ctx,
        contract_kind="explicit_command",
        criterion_id="c2",
        check=["sh", "-c", "test -f absent.txt"],
    )

    assert "FAIL" in message
    receipt = _receipts(ctx)[-1]
    assert receipt["status"] == "fail" and receipt["returncode"] != 0
    assert receipt["execution_surface"] == "remote_target"


# ── bytes_equal is compared on the target ────────────────────────────────────
def test_bytes_equal_is_decided_on_the_target(remote):
    ctx, _target, _root = remote

    message = _verify(
        ctx,
        contract_kind="explicit_command",
        criterion_id="c3",
        check=["sh", "-c", "true"],
        expected_match="bytes_equal",
        artifact_paths=["golden.txt", "actual.txt"],
    )

    assert "PASS" in message and "bytes_equal" in message
    receipt = _receipts(ctx)[-1]
    assert receipt["matched"] is True
    assert "golden.txt == actual.txt" in receipt["summary"]


def test_a_byte_divergence_fails_with_a_bounded_window(remote):
    ctx, _target, _root = remote

    message = _verify(
        ctx,
        contract_kind="explicit_command",
        criterion_id="c4",
        check=["sh", "-c", "true"],
        expected_match="bytes_equal",
        artifact_paths=["golden.txt", "diverged.txt"],
    )

    assert "FAIL" in message
    receipt = _receipts(ctx)[-1]
    assert receipt["matched"] is False
    # The whole files never crossed the wire; a bounded hexdump of the divergence did.
    assert "bytes differ at offset 3" in receipt["summary"]


# ── the after-check artifact probes are the target's ─────────────────────────
def test_the_artifact_probe_after_the_check_is_the_targets(remote):
    ctx, _target, target_repo = remote

    _verify(
        ctx,
        contract_kind="explicit_command",
        criterion_id="c5",
        check=["sh", "-c", "rm -f built.txt && touch built.txt"],
        artifact_paths=["built.txt", "never_built.txt"],
    )

    receipt = _receipts(ctx)[-1]
    lifecycle = {row["path"]: row for row in receipt["artifact_lifecycle"]}
    assert lifecycle["built.txt"]["exists_after"] is True
    assert lifecycle["never_built.txt"]["exists_after"] is False
    assert lifecycle["built.txt"]["check_surface"] == "remote_target"
    assert receipt["artifacts_missing_after"] == ["never_built.txt"]


def test_a_check_that_builds_then_deletes_is_visible(remote):
    ctx, _target, _root = remote

    _verify(
        ctx,
        contract_kind="explicit_command",
        criterion_id="c6",
        check=["sh", "-c", "touch gone.txt && rm gone.txt"],
        artifact_paths=["gone.txt"],
    )

    receipt = _receipts(ctx)[-1]
    assert receipt["status"] == "pass"
    # Flag-only, exactly as on Home: the status is the check's, the absence is disclosed.
    assert receipt["artifacts_missing_after"] == ["gone.txt"]


# ── an absent transport records NOTHING ──────────────────────────────────────
def test_no_transport_records_nothing_rather_than_a_hollow_receipt(tmp_path, target_repo, monkeypatch):
    ctx = _Ctx(tmp_path, target_repo)

    def unavailable(executor, phase):
        from ouroboros.workspace_executor import SshExecutorUnavailableError

        raise SshExecutorUnavailableError("no broker in this process")

    monkeypatch.setattr("ouroboros.workspace_executor._remote_service", unavailable)

    message = _verify(
        ctx, contract_kind="explicit_command", criterion_id="c7", check=["true"],
    )

    assert message.startswith("⚠️ VERIFY_REMOTE_UNAVAILABLE")
    assert "NOTHING was recorded" in message
    with pytest.raises((FileNotFoundError, IndexError, StopIteration, UnboundLocalError)):
        _receipts(ctx)
