"""Home↔execd contract compatibility: admission at the handshake, typed drift.

Three claims, and they are the three halves of one defect (a Home that had added a
single export-policy rule field met a target built without it, and the disagreement
surfaced at PREPARE inside an unrelated tool call as a bare
``ValueError: export policy has unknown fields: ['marker_scoped_suffixes']``):

1. a build pair that does not share the contract set is refused AT ADMISSION — at
   bundle selection, at the session preamble and at execd's first frame — with a
   typed code, both builds and an owner ACTION;
2. EVERY contract in the register answers an unrecognized member with the same typed
   refusal rather than a bare exception, and the strictness is unchanged;
3. the owner surfaces carry it: the connection row says the executor is outdated and
   stops saying so after a bootstrap, and the model-facing prepare text names the
   action instead of a Python class name.

Behavioural throughout: every case calls the real validator or the real projection.
Nothing here reads source, so no gate-boundary paragraph applies.
"""

from __future__ import annotations

import io
import json
import pathlib
import types

import pytest

from ouroboros import execd_state as state_module
from ouroboros.execd import ExecdProtocolServer
from ouroboros.execd_task_files import canonical_attachment_manifest
from ouroboros.export_policy_contract import (
    EXPORT_POLICY_VERSION,
    ExportChannelUnknownError,
    channel_profile,
    normalize_export_policy,
)
from ouroboros.remote_contract_admission import (
    admit_execd_contract_set,
    admit_home_contract_set,
)
from ouroboros.remote_contracts import (
    ACTION_BOOTSTRAP,
    ACTION_REBUILD_BUNDLE,
    CODE_BUNDLE_OUTDATED,
    CODE_EXECD_INCOMPATIBLE,
    CODE_EXECD_OUTDATED,
    CONTRACT_SET_VERSION,
    CONTRACTS,
    ContractDriftError,
    contract_set_compatible,
)
from ouroboros.remote_pending_operations import write_pending_operation
from ouroboros.remote_protocol import (
    PROTOCOL_MAJOR,
    PROTOCOL_MINOR,
    ProtocolError,
    parse_session_preamble,
    session_preamble,
    validate_control_message,
)
from ouroboros.remote_ssh_bootstrap import (
    BUNDLE_MANIFEST_SCHEMA_VERSION,
    select_and_install,
)
from ouroboros.workspace_diagnostics import RemoteWorkspaceError
from ouroboros.workspace_native_contract import admitted_native_operation

REPO = pathlib.Path(__file__).resolve().parent.parent


# ── 1. the carrier ───────────────────────────────────────────────────────────


def test_the_wire_minor_is_the_contract_set_version_and_not_a_second_number():
    """One number, one definition site.

    The whole reason the refusal works against a target installed BEFORE the check
    existed is that the contract set travels in the session preamble and both
    handshake frames of every build ever shipped — because it IS the wire minor. A
    second constant tracking it would be a pair to keep in step, and the pair going
    out of step is the class of defect this exists to close.
    """

    assert PROTOCOL_MINOR == CONTRACT_SET_VERSION
    assert CONTRACT_SET_VERSION >= 1


def test_parsing_tolerance_and_cooperation_admission_are_different_questions():
    """An older peer PARSES and is still refused a session.

    Both predicates read the same number and must disagree on an older peer: the
    tolerant one is what lets Home read the old preamble at all, which is the only
    reason it can say anything exact about it. If admission were the tolerant rule,
    the mismatched pair would be admitted — the original defect.
    """

    from ouroboros.remote_protocol import protocol_compatible

    older = CONTRACT_SET_VERSION - 1
    assert protocol_compatible(PROTOCOL_MAJOR, older) is True
    assert contract_set_compatible(older) is False
    assert contract_set_compatible(CONTRACT_SET_VERSION) is True
    # Not a range, not a coercion: a string, a bool or a float is not a version.
    for bogus in ("1", True, 1.0, None, CONTRACT_SET_VERSION + 1):
        assert contract_set_compatible(bogus) is False


# ── 2. admission at every seam ───────────────────────────────────────────────


def test_home_refuses_an_older_target_from_its_own_preamble_bytes():
    """The earliest seam: refused before Home writes a single frame.

    The preamble is built exactly as a build one contract set behind would emit it,
    and parsed by the shipped parser — so this is the real byte path, not a
    stand-in for it.
    """

    nonce = b"\x11" * 24
    older = session_preamble(
        nonce,
        protocol_major=PROTOCOL_MAJOR,
        protocol_minor=CONTRACT_SET_VERSION - 1,
    )
    consumed, major, announced = parse_session_preamble(older, nonce)
    assert (consumed, major, announced) == (len(older), PROTOCOL_MAJOR, CONTRACT_SET_VERSION - 1)

    with pytest.raises(RemoteWorkspaceError) as refused:
        admit_home_contract_set(
            announced,
            release="6.81.1",
            artifact_sha256="a" * 64,
            connection_id="conn_live",
        )
    error = refused.value
    assert error.code == CODE_EXECD_OUTDATED
    assert error.phase == "bootstrap"
    # The three facts the owner needs, and the action they can actually take.
    assert error.details["peer_contract_set"] == CONTRACT_SET_VERSION - 1
    assert error.details["required_contract_set"] == CONTRACT_SET_VERSION
    assert error.details["peer_build"] == "6.81.1"
    assert error.details["local_build"]
    assert error.details["action"] == ACTION_BOOTSTRAP
    assert error.action == ACTION_BOOTSTRAP
    assert sorted(error.details["contracts"]) == sorted(CONTRACTS)


def test_home_refuses_a_newer_target_with_a_distinguishable_code():
    """Ahead and behind are different diagnoses with the same next step."""

    with pytest.raises(RemoteWorkspaceError) as refused:
        admit_home_contract_set(CONTRACT_SET_VERSION + 1, release="9.9.9")
    assert refused.value.code == CODE_EXECD_INCOMPATIBLE
    assert refused.value.action == ACTION_BOOTSTRAP


def test_a_compatible_pair_is_admitted_silently():
    admit_home_contract_set(CONTRACT_SET_VERSION, release="6.87.6")
    admit_execd_contract_set(CONTRACT_SET_VERSION, release="6.87.6")


def test_execd_refuses_an_older_home_and_says_so_on_stderr(capsys):
    """The direction Home cannot cover, answered on the one channel every build reads.

    A handshake carries no `request_id`, so the serve loop's diagnostic answer cannot
    speak for it, and no control kind an older Home recognizes exists. The transport
    attaches stderr to `details.stderr` of the session error, so the sentence arrives
    even at a Home built before any of this.
    """

    with pytest.raises(state_module.ExecdError) as refused:
        admit_execd_contract_set(CONTRACT_SET_VERSION - 1, release="6.87.6")
    assert refused.value.code == CODE_EXECD_OUTDATED
    assert refused.value.phase == "bootstrap"
    assert refused.value.details["refused_by"] == "execd"

    emitted = json.loads(capsys.readouterr().err.strip().splitlines()[-1])
    assert emitted["execd_refusal"] == CODE_EXECD_OUTDATED
    assert emitted["details"]["action"] == ACTION_BOOTSTRAP
    assert emitted["details"]["required_contract_set"] == CONTRACT_SET_VERSION


def test_a_stale_execd_bundle_is_refused_before_the_target_is_touched(tmp_path):
    """The seam the live failure actually needed.

    The bundle in `assets/execd` is a build artifact that can lag the tree it was
    built from, and the target install faithfully mirrors it — so "the server runs an
    outdated execd" was really "this Home build ships an execd older than its own
    contracts". Bootstrap re-installs the same stale artifact, which is why this
    refusal carries `rebuild_execd_bundle` and not `bootstrap_connection`.

    A manifest with no `contract_set_version` at all is a pre-versioning bundle, read
    as 0 — the honest reading, and the one every already-shipped bundle gets.
    """

    def must_not_run(*args, **kwargs):
        raise AssertionError("a stale bundle must be refused without touching the target")

    for manifest in (
        {"schema_version": BUNDLE_MANIFEST_SCHEMA_VERSION, "build": "6.81.1"},
        {
            "schema_version": BUNDLE_MANIFEST_SCHEMA_VERSION,
            "build": "6.81.1",
            "contract_set_version": CONTRACT_SET_VERSION - 1,
        },
    ):
        bundle_dir = tmp_path / f"bundle-{manifest.get('contract_set_version', 'absent')}"
        bundle_dir.mkdir()
        (bundle_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        request = types.SimpleNamespace(bundle_dir=bundle_dir)
        with pytest.raises(RemoteWorkspaceError) as refused:
            select_and_install(
                request,
                run_remote=must_not_run,
                platform_probe=must_not_run,
                timeout_sec=5.0,
            )
        assert refused.value.code == CODE_BUNDLE_OUTDATED
        assert refused.value.details["action"] == ACTION_REBUILD_BUNDLE
        assert refused.value.action == ACTION_REBUILD_BUNDLE


def test_a_bundle_manifest_schema_this_build_cannot_read_is_refused(tmp_path):
    """`schema_version` was written by the packager and read by nobody."""

    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()
    (bundle_dir / "manifest.json").write_text(
        json.dumps({"schema_version": 99, "build": "6.87.6", "contract_set_version": CONTRACT_SET_VERSION}),
        encoding="utf-8",
    )
    with pytest.raises(RemoteWorkspaceError) as refused:
        select_and_install(
            types.SimpleNamespace(bundle_dir=bundle_dir),
            run_remote=lambda *a, **k: None,
            platform_probe=lambda *a, **k: {},
            timeout_sec=5.0,
        )
    assert refused.value.code == "execd_bundle_invalid"
    assert refused.value.details["manifest_schema_version"] == "99"


def test_the_shipped_bundle_declares_this_builds_contract_set():
    """The artifact beside this tree must be one this tree can talk to.

    Skipped rather than failed when no bundle is present: `assets/execd` is a build
    output and is not in the repository. When it IS there, it is what a bootstrap
    would install, so a mismatch is the live defect sitting in the working tree.
    """

    manifest_path = REPO / "assets" / "execd" / "manifest.json"
    if not manifest_path.is_file():
        pytest.skip("no execd bundle is built in this tree")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest.get("contract_set_version") == CONTRACT_SET_VERSION, (
        f"the execd bundle in assets/execd declares contract set "
        f"{manifest.get('contract_set_version')!r} but this build requires "
        f"{CONTRACT_SET_VERSION}; rebuild it with scripts/build_execd_bundle.py"
    )


# ── 3. every contract answers drift the same way ─────────────────────────────


def _drift(callable_, *args, **kwargs) -> ContractDriftError:
    with pytest.raises(ContractDriftError) as raised:
        callable_(*args, **kwargs)
    return raised.value


def _policy(**overrides):
    document = {
        "version": EXPORT_POLICY_VERSION,
        "channel": "workspace_query",
        "profile": "tree",
    }
    document.update(overrides)
    return document


def test_every_contract_in_the_register_refuses_an_unknown_member_the_same_way():
    """One shape of refusal across the whole register, exercised — not grepped.

    The register is the claim "these contracts are kept in step"; a contract in it
    that answered drift with a bare exception would make the claim untrue for that
    one, which is exactly how the export document behaved before. Each case below
    drives the REAL validator, so the table cannot pass by describing itself.
    """

    cases = {
        # the wire: an unknown control kind, and an unknown field on a known kind
        "wire_protocol": [
            _drift(validate_control_message, {"kind": "not_a_kind", "seq": 0}),
            _drift(
                validate_control_message,
                {"kind": "abort", "seq": 0, "request_id": "r1", "operation_id": "o1", "surprise": 1},
            ),
        ],
        # the export-policy document: unknown field, unknown version, unknown profile
        "export_policy": [
            _drift(normalize_export_policy, _policy(from_a_newer_home=[])),
            _drift(normalize_export_policy, _policy(version=EXPORT_POLICY_VERSION + 1)),
            _drift(normalize_export_policy, _policy(profile="something_else")),
            _drift(channel_profile, "a_channel_nobody_declared"),
        ],
        "native_operations": [_drift(admitted_native_operation, "teleport_file")],
        "attachment_stage": [
            _drift(
                canonical_attachment_manifest,
                [
                    {
                        "attachment_id": "att1",
                        "label": "one",
                        "root": "artifact_store",
                        "relpath": "attachments/one.txt",
                        "mime": "text/plain",
                        "is_image": False,
                        "size": 3,
                        "sha256": "a" * 64,
                        "stage_status": "ready",
                        "redaction_from_a_newer_home": True,
                    }
                ],
            )
        ],
        "import_channel": [
            _drift(
                write_pending_operation,
                types.SimpleNamespace(),
                import_kind="a_channel_nobody_declared",
                import_context={},
                request_id="r1",
                operation_id="o1",
                prepared_hash="a" * 64,
                tool="read_file",
                task_id="t1",
            )
        ],
    }
    for contract, errors in cases.items():
        for error in errors:
            assert error.contract == contract, (contract, error.contract)
            assert error.unknown, error
            assert error.understood, error
            assert error.action == ACTION_BOOTSTRAP
            assert error.contract_set_version == CONTRACT_SET_VERSION
            details = error.details()
            assert details["contract"] == contract
            assert details["unknown"] == list(error.unknown)
            assert details["action"] == ACTION_BOOTSTRAP
            # The message names WHAT was not understood, not merely that something was.
            assert str(error.unknown[0]) in str(error)
    # Every contract exercised above is one the register declares.
    assert set(cases) <= set(CONTRACTS)


def test_wire_drift_is_still_a_protocol_error_so_the_transport_tears_down_as_before():
    """Strictness and routing unchanged; only the diagnosis improved.

    Every transport path treats a `ProtocolError` as "this session cannot continue".
    Routing wire drift through some other type would end the session by a different
    path than every other malformed frame — a behaviour change nobody asked for.
    """

    error = _drift(validate_control_message, {"kind": "not_a_kind", "seq": 0})
    assert isinstance(error, ProtocolError)
    assert isinstance(error, ValueError)


def test_the_channel_registry_refusal_keeps_its_wire_code():
    """`REMOTE_EXPORT_CHANNEL_UNKNOWN` appears in receipts and ledgers.

    Gaining contract/member/action fields must not rename it: the spelling is a wire
    value, and a refusal that improved its own diagnosis by breaking a stored one
    would trade a readable message for an unreadable receipt.
    """

    error = _drift(channel_profile, "a_channel_nobody_declared")
    assert isinstance(error, ExportChannelUnknownError)
    assert error.code == "REMOTE_EXPORT_CHANNEL_UNKNOWN"


def test_every_contract_drift_error_is_still_a_value_error():
    """The base class is load-bearing.

    Boundaries all over the tree guard a normalization with `except ValueError`, and
    the refusal has to keep being caught there. This is why `ContractDriftError`
    subclasses `ValueError` instead of introducing a parallel hierarchy.
    """

    with pytest.raises(ValueError):
        normalize_export_policy(_policy(from_a_newer_home=[]))


def test_the_export_policy_field_from_the_live_failure_is_understood_now():
    """The instance, kept as a regression: `marker_scoped_suffixes` normalizes."""

    document = normalize_export_policy(
        _policy(marker_scoped_suffixes=[".conf", ".yaml"])
    )
    assert document["marker_scoped_suffixes"] == [".conf", ".yaml"]


# ── 4. the diagnostic execd puts on the wire ─────────────────────────────────


def test_contract_drift_reaches_the_wire_as_its_own_code_not_as_ValueError():
    """The projection that produced the owner's unreadable message.

    Three copies of `code=type(exc).__name__` lived in `execd.py`; there is one
    mapping now, and a contract refusal contributes its own code and the
    contract/member/action details. An unexpected exception still reports its class
    name, which for a genuine surprise is the most honest thing available.
    """

    drift = _drift(normalize_export_policy, _policy(from_a_newer_home=[]))
    diagnostic = state_module.exception_diagnostic(
        drift, request_id="r1", operation_id="o1"
    )
    assert diagnostic["code"] == drift.code
    assert diagnostic["code"] != "ValueError"
    assert diagnostic["details"]["contract"] == "export_policy"
    assert diagnostic["details"]["action"] == ACTION_BOOTSTRAP
    assert diagnostic["request_id"] == "r1"

    surprise = state_module.exception_diagnostic(RuntimeError("boom"), request_id="r1")
    assert surprise["code"] == "RuntimeError"

    typed = state_module.exception_diagnostic(
        state_module.ExecdError("prepared_call_stale", "gone", phase="authorize"),
        request_id="r1",
        operation_id="o1",
    )
    assert typed["code"] == "prepared_call_stale"
    assert typed["phase"] == "authorize"


def test_execds_first_frame_refuses_a_mismatched_home_instead_of_answering_it(tmp_path):
    """The handshake is admitted; a mismatched pair never reaches `handshake_ok`."""

    from tests.test_execd_state import _service  # the shipped in-process service fixture

    service = _service(tmp_path)
    writer = io.BytesIO()
    server = ExecdProtocolServer(service, io.BytesIO(), writer)
    with pytest.raises(state_module.ExecdError) as refused:
        server._receive_control(
            {"kind": "handshake", "protocol_minor": CONTRACT_SET_VERSION - 1}
        )
    assert refused.value.code == CODE_EXECD_OUTDATED
    assert writer.getvalue() == b"", "no handshake_ok may be written to a refused pair"


# ── 5. execd's durable journal schema ────────────────────────────────────────


def test_a_journal_record_from_another_schema_is_refused_not_reinterpreted(tmp_path):
    """`JOURNAL_SCHEMA_VERSION` was written into every record and read by nothing.

    So a record laid down by a different execd build was read field by field as if it
    were ours — a completion state, a request hash and a durable ack taken off a
    document whose shape nobody checked. The custody and spool-quota ledgers already
    refused on their own versions; this is the third schema of the same kind.
    """

    record = tmp_path / "record.json"
    record.write_text(
        json.dumps({"_schema_version": 99, "state": "started", "task_id": "t1"}),
        encoding="utf-8",
    )
    with pytest.raises(state_module.ExecdError) as refused:
        state_module.read_json(
            record, schema_version=state_module.JOURNAL_SCHEMA_VERSION
        )
    assert refused.value.code == "durable_state_schema_mismatch"
    assert refused.value.details["expected_schema_version"] == state_module.JOURNAL_SCHEMA_VERSION
    assert refused.value.details["action"] == ACTION_BOOTSTRAP
    # A record of the right schema still reads, and an unversioned read is unchanged.
    record.write_text(
        json.dumps({"_schema_version": state_module.JOURNAL_SCHEMA_VERSION, "state": "started"}),
        encoding="utf-8",
    )
    assert state_module.read_json(
        record, schema_version=state_module.JOURNAL_SCHEMA_VERSION
    )["state"] == "started"
    assert state_module.read_json(record)["state"] == "started"


# ── 6. the owner surfaces ────────────────────────────────────────────────────


def test_the_connection_row_says_the_executor_is_outdated_and_recovers_after_bootstrap(tmp_path):
    """The status a healthy-looking, unusable connection was missing.

    `bootstrapped_at` said a bootstrap had happened; nothing said whether what it
    installed can still talk to this build. So a target running an execd from a
    different contract set read as `ready` in Connections while every remote tool call
    on it failed — which is what the owner actually saw.

    Checked against the CONTRACT SET, not the release id: most Ouroboros releases
    change no shared contract, so a release comparison would flag every connection
    after every upgrade, and that is the noise this carrier exists to avoid.
    """

    from ouroboros.connection_store import (
        add_connection,
        pin_connection_host,
        record_bootstrap,
        retrust_connection,
    )
    from ouroboros.gateway.connections import _record_runtime_health, _runtime_evidence_fields

    path = tmp_path / "state" / "remote_connections.json"
    added = add_connection(name="Build host", ssh_alias="build-host", path=path)
    assert added["bootstrap_contract_set"] == 0

    # Never bootstrapped: incompatible, and NOT reported as outdated — two different
    # sentences, and a surface that conflated them could not say which one it meant.
    fresh = _runtime_evidence_fields(path, added["id"])
    assert fresh["bootstrap_compatible"] is False
    assert fresh["execd_outdated"] is False

    # Bootstrapped by a build that predates the contract-set carrier.
    pin_connection_host(added["id"], "host-a", path=path)
    record_bootstrap(added["id"], build="6.81.1", path=path)
    _record_runtime_health(path, added["id"], {"status": "ready"})
    stale = _runtime_evidence_fields(path, added["id"])
    assert stale["bootstrap_compatible"] is True
    assert stale["status"] == "ready", "the transport is genuinely answering"
    assert stale["execd_outdated"] is True, "…and still cannot serve a remote call"
    assert stale["build"] == "6.81.1"
    assert stale["bootstrap_contract_set"] == 0
    assert stale["required_contract_set"] == CONTRACT_SET_VERSION

    # One Bootstrap with the bundle this build ships, and the status recovers.
    recorded = record_bootstrap(
        added["id"], build="6.87.6", contract_set=CONTRACT_SET_VERSION, path=path
    )
    assert recorded["bootstrap_contract_set"] == CONTRACT_SET_VERSION
    healed = _runtime_evidence_fields(path, added["id"])
    assert healed["execd_outdated"] is False
    assert healed["bootstrap_compatible"] is True
    assert healed["build"] == "6.87.6"

    # Retrust invalidates the whole claim, contract set included: a different host was
    # never proven to carry anything.
    trusted = retrust_connection(added["id"], "host-b", path=path)
    assert trusted["bootstrap_contract_set"] == 0


def test_the_live_projection_carries_the_outdated_fields_to_the_browser():
    """A field the projection drops is a badge the UI can never show."""

    from ouroboros.gateway.connections import _public_live_fields

    projected = _public_live_fields(
        {
            "status": "ready",
            "execd_outdated": True,
            "required_contract_set": CONTRACT_SET_VERSION,
            "bootstrap_contract_set": 0,
            "bootstrap_compatible": True,
        }
    )
    assert projected["execd_outdated"] is True
    assert projected["required_contract_set"] == CONTRACT_SET_VERSION
    assert projected["bootstrap_contract_set"] == 0


def test_the_cli_treats_a_contract_skew_as_unservable_rather_than_retryable():
    """Exit 4, not 2: no number of retries changes a build pair."""

    from ouroboros.cli_connections import _UNSERVABLE_CODES

    assert {
        CODE_EXECD_OUTDATED,
        CODE_EXECD_INCOMPATIBLE,
        CODE_BUNDLE_OUTDATED,
    } <= _UNSERVABLE_CODES


def test_a_typed_transport_refusal_reports_its_action_and_not_a_bare_retry():
    """`RemoteWorkspaceError` had no `action`, so the surface read `retry` for everything.

    Which is the one thing that cannot work here. The action is derived from `details`
    — the slot that already reaches the browser and `--json` — so no raiser has to
    remember a new parameter, and refusals that really are retryable are unchanged.
    """

    skewed = RemoteWorkspaceError(
        CODE_EXECD_OUTDATED,
        "outdated",
        phase="bootstrap",
        details={"action": ACTION_BOOTSTRAP},
    )
    assert skewed.action == ACTION_BOOTSTRAP
    ordinary = RemoteWorkspaceError("ssh_session_disconnected", "gone", phase="stream")
    assert ordinary.action == "retry"


def test_the_model_facing_prepare_refusal_names_the_action_not_a_class_name(
    tmp_path, monkeypatch
):
    """The exact text the owner read, produced by the real dispatch.

    `⚠️ REMOTE_EXECUTION_UNAVAILABLE: ValueError: export policy has unknown fields:
    [...]` was a diagnosis with no next step, given to the one reader who cannot look
    a code up. Two arms produce it and both are driven here through the real
    registry dispatch: a refusal the TARGET raised (a `RemoteWorkspaceError` carrying
    an action) and a contract Home itself could not project (a `ContractDriftError`,
    which must be caught BEFORE the bare `ValueError` arm or it reports as its Python
    class name again — the substitution that made the message unreadable).
    """

    from ouroboros import workspace_executor
    from ouroboros.workspace_ref import SEALED_WORKSPACE_REF_KEY
    from tests.golden_traces import scenarios

    registry, _roots = scenarios._workspace(tmp_path)
    registry._ctx.task_metadata[SEALED_WORKSPACE_REF_KEY] = {
        "kind": "ssh",
        "connection_id": "conn-1",
        "remote_root": "/srv/work/app",
        "workspace_id": "ws-1",
    }

    def _target_refusal(*args, **kwargs):
        raise RemoteWorkspaceError(
            CODE_EXECD_OUTDATED,
            "the peer announces 0, this build requires 1",
            phase="bootstrap",
            details={"action": ACTION_BOOTSTRAP},
        )

    monkeypatch.setattr(workspace_executor, "prepare_native_operation", _target_refusal)
    out = registry.execute("read_file", {"path": "README.md"})
    assert out.startswith("⚠️ REMOTE_EXECUTION_UNAVAILABLE:")
    assert CODE_EXECD_OUTDATED in out
    assert f"[action: {ACTION_BOOTSTRAP}]" in out
    assert "ValueError" not in out

    def _home_drift(*args, **kwargs):
        normalize_export_policy(_policy(from_a_newer_home=[]))

    monkeypatch.setattr(workspace_executor, "prepare_native_operation", _home_drift)
    out = registry.execute("read_file", {"path": "README.md"})
    assert out.startswith("⚠️ REMOTE_EXECUTION_UNAVAILABLE:")
    assert "remote_contract_unknown_member" in out
    assert f"[action: {ACTION_BOOTSTRAP}]" in out
    assert "ValueError:" not in out, "a contract refusal must not wear a class name"

    # A refusal that really IS retryable gains nothing: no noise where none is due.
    def _retryable(*args, **kwargs):
        raise RemoteWorkspaceError("ssh_session_disconnected", "gone", phase="stream")

    monkeypatch.setattr(workspace_executor, "prepare_native_operation", _retryable)
    out = registry.execute("read_file", {"path": "README.md"})
    assert "ssh_session_disconnected" in out
    assert "[action:" not in out
