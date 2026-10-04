"""CyberGym gateway wire-protocol, served-telemetry, and accounting tests.

Split from ``tests/test_cybergym_executor.py`` along the HTTP/gateway seam:
provider probe, submit/verify/private-query wire parsing, served-telemetry
validation and campaign cost accounting. Gateway custody and deadline tests
live in ``test_cybergym_gateway_custody.py``.
Shared fixtures (``_config``, ``dataclasses_replace``) are imported from the
original module; executor-lifecycle tests remain there.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import pathlib

import pytest

from devtools.benchmarks.cybergym import cybergym_executor as executor_module
from devtools.benchmarks.cybergym.cybergym_adapter import (
    CAPABILITY_FINAL_POC_MISSING,
    PROTOCOL_FAIL,
    BudgetLedger,
    TaskSpec,
    run_campaign,
)
from devtools.benchmarks.cybergym.cybergym_executor import (
    CommandResult,
    CyberGymExecutor,
    ExecutorFailure,
    _parse_json_stdout,
    _require_exact_effort,
    _served_telemetry,
    _validate_verify_response,
)
from devtools.benchmarks.cybergym.cybergym_wire import GatewayTransportError
from tests.test_cybergym_executor import _config, dataclasses_replace


def test_provider_probe_checks_exact_model_without_server_search(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter-key")
    captured = {}

    def http(method, url, *, body=None, headers=None, timeout=None):
        if method == "GET" and url.endswith("/models"):
            return {
                "data": [{
                    "id": "deepseek/deepseek-v4-flash-0731",
                    "context_length": 1_310_720,
                    "supported_parameters": ["reasoning", "tools"],
                }]
            }
        if method == "GET" and url.endswith("/key"):
            return {"data": {"limit_remaining": 100}}
        assert method == "POST"
        captured["body"] = body
        captured["headers"] = headers
        return {
            "id": "response-1",
            "model": "deepseek/deepseek-v4-flash-0731",
            "provider": "OpenInference",
            "choices": [{"message": {"content": "OK"}}],
            "usage": {
                "prompt_tokens": 12,
                "completion_tokens": 3,
                "cost": 0.006,
                "cost_estimated": False,
            },
        }

    executor = CyberGymExecutor(
        _config(
            tmp_path,
            provider_probe=True,
            expected_data_sha256="a" * 64,
            expected_binary_sha256="b" * 64,
            http_runner=http,
        )
    )
    executor._probe_provider()  # noqa: SLF001 - provider boundary assertion

    assert captured["body"]["messages"] == [{"role": "user", "content": "Reply with OK."}]
    assert "tools" not in captured["body"]
    from ouroboros.openrouter_attribution import OPENROUTER_APP_HEADERS

    assert {
        key: captured["headers"][key] for key in OPENROUTER_APP_HEADERS
    } == OPENROUTER_APP_HEADERS
    assert executor.provider_observation["observed_model"] == (
        "deepseek/deepseek-v4-flash-0731"
    )


def test_verify_response_requires_success_body_and_designated_poc():
    good = {"message": "All 1 PoCs for this agent_id have been verified", "poc_ids": ["poc-1"]}
    assert _validate_verify_response(good, expected_poc_id="poc-1") == good
    with pytest.raises(ExecutorFailure, match="HTTP 500"):
        _validate_verify_response({"status_code": 500, "body": {"detail": "failed"}})
    with pytest.raises(ExecutorFailure, match="poc_ids"):
        _validate_verify_response({"message": "ok", "poc_ids": []})
    with pytest.raises(ExecutorFailure, match="designated poc_id"):
        _validate_verify_response(good, expected_poc_id="other")


def test_observed_effort_must_be_exactly_high():
    assert _require_exact_effort("high") == "high"
    for value in ("", "High", "max", None):
        with pytest.raises(ExecutorFailure, match="exactly high"):
            _require_exact_effort(value)


def test_served_telemetry_prefers_authoritative_trace_refs_over_requested_fields():
    payload = {
        "model": "requested/not-served",
        "reasoning_effort": "high",
        "trace_refs": {
            "llm_call_refs": [
                {"resolved_model": "deepseek/deepseek-v4-flash-0731", "provider": "provider-a"}
            ]
        },
    }
    observed = _served_telemetry(payload)
    assert observed["observed_model"] == "deepseek/deepseek-v4-flash-0731"
    assert observed["observed_provider"] == "provider-a"
    assert observed["trace_call_count"] == 1
    assert observed["effort_source"] == "runtime_requested_field"


def test_served_telemetry_rejects_incomplete_or_mixed_trace_identity():
    with pytest.raises(ExecutorFailure, match="incomplete served-call"):
        _served_telemetry({"trace_refs": {"llm_call_refs": [{"provider": "provider-a"}]}})
    with pytest.raises(ExecutorFailure, match="mixed served models"):
        _served_telemetry(
            {
                "trace_refs": {
                    "llm_call_refs": [
                        {"resolved_model": "model-a", "provider": "provider-a"},
                        {"resolved_model": "model-b", "provider": "provider-a"},
                    ]
                }
            }
        )


def test_served_telemetry_reads_verified_response_wire_effort(tmp_path):
    drive = tmp_path / "drive"
    calls = drive / "observability" / "calls" / "opaque"
    calls.mkdir(parents=True)
    wire = {
        "requested_effort": "high",
        "applied_effort": "high",
        "attempt_id": "attempt-1",
        "candidate_sha256": "a" * 64,
    }
    blob_raw = json.dumps(
        {"usage": {"request_wire": wire, "response_provider": "backend-a"}},
        sort_keys=True,
    ).encode("utf-8")
    blob_path = drive / "observability" / "blobs" / ("b" * 64 + ".json.gz")
    blob_path.parent.mkdir(parents=True)
    blob_path.write_bytes(gzip.compress(blob_raw))
    blob_ref = {
        "path": str(blob_path),
        "sha256": hashlib.sha256(blob_raw).hexdigest(),
        "size": len(blob_raw),
        "kind": "json",
        "encoding": "gzip",
    }
    manifest_raw = json.dumps(
        {
            "task_id": "opaque",
            "call_id": "llm-1_response",
            "llm_call_id": "llm-1",
            "full_payload_ref": blob_ref,
        },
        sort_keys=True,
    ).encode("utf-8")
    manifest_path = calls / "llm-1_response.json"
    manifest_path.write_bytes(manifest_raw)
    manifest_ref = {
        "path": str(manifest_path),
        "sha256": hashlib.sha256(manifest_raw).hexdigest(),
        "call_id": "llm-1_response",
    }

    observed = _served_telemetry(
        {
            "reasoning_effort": "low",
            "trace_refs": {
                "llm_call_refs": [
                    {
                        "llm_call_id": "llm-1",
                        "resolved_model": "deepseek/deepseek-v4-flash-0731",
                        "provider": "provider-a",
                        "response_ref": manifest_ref,
                    }
                ]
            },
        },
        allowed_roots=(drive,),
    )
    assert observed["observed_effort"] == "high"
    assert observed["observed_provider"] == "backend-a"
    assert observed["observed_provider_attempts"] == ["backend-a"]
    assert observed["provider_distribution"] == {"backend-a": 1}
    assert observed["effort_source"] == "served_response_wire"
    assert observed["response_wire_effort_count"] == 1
    assert observed["response_wire_provider_count"] == 1


def test_served_telemetry_uses_isolate_data_root_for_wire_refs(tmp_path):
    external = (tmp_path / "nvme" / "ouroboros-data").resolve()
    calls = external / "observability" / "calls" / "opaque"
    calls.mkdir(parents=True)
    wire = {
        "requested_effort": "high",
        "applied_effort": "high",
        "attempt_id": "attempt-1",
        "candidate_sha256": "a" * 64,
    }
    blob_raw = json.dumps(
        {"usage": {"request_wire": wire, "response_provider": "backend-a"}},
        sort_keys=True,
    ).encode("utf-8")
    blob_path = external / "observability" / "blobs" / ("b" * 64 + ".json.gz")
    blob_path.parent.mkdir(parents=True)
    blob_path.write_bytes(gzip.compress(blob_raw))
    blob_ref = {
        "path": str(blob_path),
        "sha256": hashlib.sha256(blob_raw).hexdigest(),
        "size": len(blob_raw),
        "kind": "json",
        "encoding": "gzip",
    }
    manifest_raw = json.dumps(
        {
            "task_id": "opaque",
            "call_id": "llm-1_response",
            "llm_call_id": "llm-1",
            "full_payload_ref": blob_ref,
        },
        sort_keys=True,
    ).encode("utf-8")
    manifest_path = calls / "llm-1_response.json"
    manifest_path.write_bytes(manifest_raw)
    payload = {
        "reasoning_effort": "low",
        "trace_refs": {
            "llm_call_refs": [
                {
                    "llm_call_id": "llm-1",
                    "resolved_model": "deepseek/deepseek-v4-flash-0731",
                    "provider": "provider-a",
                    "response_ref": {
                        "path": str(manifest_path),
                        "sha256": hashlib.sha256(manifest_raw).hexdigest(),
                        "call_id": "llm-1_response",
                    },
                }
            ]
        },
    }

    config = _config(tmp_path, isolate_data_root=external)
    executor = CyberGymExecutor(config)
    observed = _served_telemetry(
        payload,
        allowed_roots=executor._telemetry_allowed_roots(),  # noqa: SLF001
    )
    assert observed["effort_source"] == "served_response_wire"
    assert observed["observed_effort"] == "high"
    # Without the external root the same wire evidence is untrusted and the
    # telemetry falls back to the requested field rather than failing closed
    # on a paid-path fact it cannot verify.
    untrusted = _served_telemetry(payload, allowed_roots=(config.run_root,))
    assert untrusted["effort_source"] == "runtime_requested_field"


def test_submit_stdout_parser_accepts_preceding_prose_and_multiline_json():
    parsed = _parse_json_stdout('notice\n{\n  "task_id": "opaque1234",\n  "poc_id": "poc-1"\n}\n')
    assert parsed == {"task_id": "opaque1234", "poc_id": "poc-1"}


def test_private_query_rejects_http_and_body_errors(tmp_path, monkeypatch):
    config = _config(tmp_path)
    monkeypatch.setenv("CYBERGYM_API_KEY", "test-secret-value")
    executor = CyberGymExecutor(
        dataclasses_replace(
            config,
            http_runner=lambda *args, **kwargs: {"status_code": 404, "body": {"detail": "Record not found"}},
        )
    )
    with pytest.raises(ExecutorFailure, match="HTTP 404"):
        executor._private_query("agent-" + "a" * 24, "arvo:1")

    executor = CyberGymExecutor(
        dataclasses_replace(
            config,
            http_runner=lambda *args, **kwargs: {"status_code": 200, "body": {"error": {"message": "bad"}}},
        )
    )
    with pytest.raises(ExecutorFailure, match="error object"):
        executor._private_query("agent-" + "a" * 24, "arvo:1")


def test_private_query_404_with_allow_empty_returns_empty_list(tmp_path, monkeypatch):
    # The pinned upstream /query-poc answers 404 "Record not found" for an
    # agent that has never submitted to this task.  The reuse-check path
    # (allow_empty=True) must read that as the empty list and fall through to
    # ``_submit_final``; only the post-submit query may treat 404 as fatal.
    config = _config(tmp_path)
    monkeypatch.setenv("CYBERGYM_API_KEY", "test-secret-value")
    executor = CyberGymExecutor(
        dataclasses_replace(
            config,
            http_runner=lambda *args, **kwargs: {"status_code": 404, "body": {"detail": "Record not found"}},
        )
    )
    assert executor._private_query("agent-" + "a" * 24, "arvo:1", allow_empty=True) == []

    with pytest.raises(ExecutorFailure, match="HTTP 404"):
        executor._private_query("agent-" + "a" * 24, "arvo:1", allow_empty=False)


def test_private_query_accepts_nested_items_wrapper(tmp_path, monkeypatch):
    config = _config(tmp_path)
    monkeypatch.setenv("CYBERGYM_API_KEY", "test-secret-value")
    record = {"task_id": "arvo:1", "poc_id": "poc-1", "poc_hash": "a" * 64}
    executor = CyberGymExecutor(
        dataclasses_replace(
            config,
            http_runner=lambda *args, **kwargs: {"pocs": {"items": [record]}},
        )
    )
    assert executor._private_query("agent-" + "a" * 24, "arvo:1") == [record]


def test_private_sidecar_transport_failure_is_not_gateway_circuit_class(tmp_path):
    config = _config(tmp_path)
    executor = CyberGymExecutor(
        dataclasses_replace(
            config,
            command_runner=lambda *_args, **_kwargs: CommandResult(
                1, "", "failed"
            ),
        )
    )
    executor.server_id = "a" * 64
    with pytest.raises(ExecutorFailure) as excinfo:
        executor._server_http("POST", "/verify-agent-pocs")
    assert not isinstance(excinfo.value, GatewayTransportError)


def test_submit_response_binds_poc_id_not_nonexistent_hash_and_keeps_exit_code(tmp_path):
    config = _config(tmp_path)
    task_dir = config.run_root / "task"
    task_dir.mkdir()
    (task_dir / "final.poc").write_bytes(b"poc-bytes")
    (task_dir / "submit.sh").write_text("TASK_ID=opaque1234\n", encoding="utf-8")
    executor_name = "workspace"
    executor_id = "c" * 64

    submit_calls = []

    def command(argv, *, cwd=None, env=None, timeout=None):
        submit_calls.append(list(argv))
        return CommandResult(
            0,
            json.dumps(
                {
                    "task_id": "opaque1234",
                    "poc_id": "poc-1",
                    "exit_code": 71,
                    "output": "known",
                    # Upstream does not define a response hash; an incidental
                    # field must not override the local marker binding.
                    "hash": "not-the-poc-hash",
                }
            ),
            "",
        )

    executor = CyberGymExecutor(dataclasses_replace(config, command_runner=command))
    executor._task_containers[executor_name] = executor_id
    response, digest, masked = executor._submit_final(  # noqa: SLF001 - boundary contract assertion
        type("Task", (), {"task_id": "arvo:1"})(), task_dir, "workspace"
    )
    assert response["poc_id"] == "poc-1"
    assert response["exit_code"] == 71
    assert digest == hashlib.sha256(b"poc-bytes").hexdigest()
    assert masked == "opaque1234"
    assert executor_id in submit_calls[0]
    assert executor_name not in submit_calls[0]


def test_delivery_checkpoint_prevents_duplicate_submit_and_verify(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("CYBERGYM_API_KEY", "test-secret-value")
    config = _config(tmp_path, provider_probe=False)
    executor = CyberGymExecutor(config)
    task = TaskSpec("arvo:1", "arvo")
    task_dir = config.run_root / "arvo__1"
    workspace = config.run_root / "workspace"
    task_dir.mkdir(parents=True)
    workspace.mkdir()
    payload = b"poc"
    (workspace / "final.poc").write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    checkpoint = config.run_root / "checkpoint.json"
    checkpoint.write_text(
        json.dumps({"gateway_task_id": "gateway-1", "status": "completed"}),
        encoding="utf-8",
    )
    record = {
        "task_id": task.task_id,
        "agent_id": "opaque1234",
        "poc_id": "poc-1",
        "poc_hash": digest,
        "vul_exit_code": 1,
        "fix_exit_code": 0,
    }
    counts = {"submit": 0, "verify": 0, "query": 0}

    def submit(*_args):
        counts["submit"] += 1
        return {"task_id": "opaque1234", "poc_id": "poc-1"}, digest, "opaque1234"

    def query(*_args, **_kwargs):
        counts["query"] += 1
        return [] if counts["query"] <= 2 else [record]

    def server_http(*_args, **_kwargs):
        counts["verify"] += 1
        return {"message": "verified", "poc_ids": ["poc-1"]}

    monkeypatch.setattr(executor, "_submit_final", submit)
    monkeypatch.setattr(executor, "_private_query", query)
    monkeypatch.setattr(executor, "_server_http", server_http)
    gateway_result = {
        "status": "completed",
        "observed_model": config.model,
        "observed_provider": "backend-a",
        "reasoning_effort": "high",
        "prompt_tokens": 10,
        "completion_tokens": 5,
        "cost_usd": 0.1,
        "cost_final": True,
        "cost_breakdown": {
            "accounted_upper_bound_usd": 0.1,
            "cost_final": True,
        },
        "outcome_axes": {"execution": {"status": "ok"}},
    }
    kwargs = {
        "checkpoint": checkpoint,
        "cleanup_ref": config.run_root / "cleanup.json",
        "alias_ref": config.run_root / "alias.json",
        "attestation_ref": "",
        "sidecar_attestation": {"status": "passed"},
    }
    for _ in range(2):
        outcome = executor._deliver_gateway_result(  # noqa: SLF001
            task,
            task_dir,
            workspace,
            "workspace",
            "agent-" + "a" * 24,
            gateway_result,
            terminal_evidence={},
            **kwargs,
        )
        assert outcome["status"] == "completed"
    assert counts == {"submit": 1, "verify": 1, "query": 4}
    delivery = json.loads(checkpoint.read_text(encoding="utf-8"))["delivery"]
    assert delivery["phase"] == "classified"
    assert delivery["final_poc_sha256"] == digest


def _stub_terminal_task_executor(tmp_path, monkeypatch, gateway_result):
    config = _config(tmp_path, provider_probe=False)
    executor = CyberGymExecutor(config)
    monkeypatch.setattr(executor, "start", lambda: None)
    monkeypatch.setattr(executor, "_generate", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        executor_module,
        "_install_workspace_backend_alias",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(executor, "_workspace", lambda *_args, **_kwargs: "container-a")
    monkeypatch.setattr(
        executor,
        "_task_body",
        lambda task, *_args, **_kwargs: {"task_id": "cybergym-" + task.task_id.replace(":", "-")},
    )
    monkeypatch.setattr(
        executor, "_gateway_wait", lambda *_args, **_kwargs: dict(gateway_result)
    )
    monkeypatch.setattr(
        executor,
        "_cleanup_workspace_container",
        lambda *_args, **_kwargs: {"status": "verified"},
    )
    return config, executor


@pytest.mark.parametrize("change", ["stable_a", "stable_b_before_submit", "changed_back_after_submit"])
def test_delivery_binds_published_bytes_to_submitted_digest(tmp_path, monkeypatch, change):
    gateway_result = {
        "status": "completed", "observed_model": "deepseek/deepseek-v4-flash-0731",
        "observed_provider": "synthetic-backend", "reasoning_effort": "high",
        "prompt_tokens": 10, "completion_tokens": 5, "cost_usd": 0.1, "cost_final": True,
        "cost_breakdown": {"accounted_upper_bound_usd": 0.1, "cost_final": True},
        "outcome_axes": {"execution": {"status": "ok"}},
    }
    config, executor = _stub_terminal_task_executor(tmp_path, monkeypatch, gateway_result)
    original, updated = b"synthetic-original-marker", b"synthetic-later-marker"
    markers, submitted = [], []
    calls = {"gateway": 0, "query": 0}

    def generate(_task, workspace, _agent_id):
        marker = workspace / "final.poc"
        marker.write_bytes(original)
        (workspace / "submit.sh").write_text('TASK_ID="opaque1234"\n', encoding="utf-8")
        markers.append(marker)

    def gateway(*_args, **_kwargs):
        calls["gateway"] += 1
        return dict(gateway_result)

    def command(argv, **_kwargs):
        assert argv[:1] == ["docker"] and "exec" in argv
        # These argv paths are in the Linux container, not the host filesystem.
        # Host Path.resolve() would turn them into drive paths on Windows.
        assert argv[-3:] == ["bash", "/workspace/submit.sh", "/workspace/final.poc"]
        submitted.append(markers[0].read_bytes())
        if change == "changed_back_after_submit":
            markers[0].write_bytes(original)
        return CommandResult(0, json.dumps({"task_id": "opaque1234", "poc_id": "poc-1", "exit_code": 1}), "")

    def query(*_args, **_kwargs):
        calls["query"] += 1
        if calls["query"] == 1:
            if change != "stable_a":
                markers[0].write_bytes(updated)
            return []
        return [{"task_id": "arvo:1", "agent_id": "opaque1234", "poc_id": "poc-1",
                 "poc_hash": hashlib.sha256(submitted[0]).hexdigest(),
                 "vul_exit_code": 1, "fix_exit_code": 0}]

    def external_operation(*_args, **_kwargs):
        pytest.fail("No actual command, HTTP, provider or PoC may execute")

    executor.config = dataclasses_replace(config, command_runner=command, http_runner=external_operation)
    executor._task_containers["container-a"] = "workspace-123"
    monkeypatch.setattr(executor, "_generate", generate)
    monkeypatch.setattr(executor, "_gateway_wait", gateway)
    monkeypatch.setattr(executor, "_private_query", query)
    monkeypatch.setattr(executor, "_ensure_key", lambda: "synthetic-key")
    monkeypatch.setattr(executor, "_server_http", external_operation)
    [row] = run_campaign([TaskSpec("arvo:1", "arvo")], run_root=config.run_root,
                         executor=executor.run_task, estimated_cost_usd=1, budget_cap_usd=10)
    persisted = [json.loads(line) for line in (config.run_root / "result_index.jsonl").read_text(encoding="utf-8").splitlines()]
    assert persisted == [row]
    assert calls["gateway"] == len(submitted) == 1
    expected = original if change == "stable_a" else updated
    assert submitted == [expected]
    digest = hashlib.sha256(expected).hexdigest()
    checkpoint = json.loads(pathlib.Path(row["artifact_refs"]["checkpoint"]).read_text(encoding="utf-8"))
    delivery = checkpoint["delivery"]
    assert delivery["final_poc_sha256"] == digest
    assert delivery["submit"]["poc_id"] == "poc-1"
    destination = pathlib.Path(row["artifact_refs"]["task_dir"]) / "final.poc"
    assert not list(destination.parent.glob(".final-poc-*"))
    if change == "changed_back_after_submit":
        assert row["status"] == "infra_failed"
        assert row["lifecycle"] == "post_gateway_evaluation_failed"
        assert row["error"] == "final PoC changed before copy"
        assert row["final_submission_status"] == "unknown"
        assert row["final_submission_success"] is not True and row["official_success"] is not True
        assert row["prediction_written"] is False and row["official_eval_status"] == "not_run"
        assert row["trial_count"] == 0
        assert not destination.exists()
        assert delivery["phase"] == "submitted" and calls["query"] == 1
    else:
        assert row["status"] == "completed"
        assert row["final_submission_success"] is True and row["official_success"] is True
        assert row["prediction_written"] is True and row["official_eval_status"] == "completed"
        assert row["final_poc_hash"] == digest
        assert destination.read_bytes() == expected
        assert delivery["phase"] == "classified" and calls["query"] > 1
    assert row["cost_usd"] == pytest.approx(0.1) and row["cost_final"] is True
    assert row["prompt_tokens"] == 10 and row["completion_tokens"] == 5
    ledger = BudgetLedger(config.run_root / "claims.jsonl", cap_usd=10)
    assert ledger.projection().settled_usd == pytest.approx(0.1)
    assert ledger.projection().unresolved_upper_bound_usd == 0
    events = [json.loads(line)["event"] for line in ledger.path.read_text(encoding="utf-8").splitlines()]
    assert events.count("claim") == events.count("settle") == 1


def test_regrade_final_poc_uses_official_verifier_without_gateway(tmp_path, monkeypatch):
    config = _config(tmp_path, provider_probe=False)
    executor = CyberGymExecutor(config)
    task = TaskSpec("arvo:1", "arvo")
    source_dir = tmp_path / "historical-task"
    source_dir.mkdir()
    source = source_dir / "final.poc"
    source.write_bytes(b"historical-poc")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    workspace = config.run_root / "regrade-workspace"
    record = {
        "task_id": "arvo:1",
        "agent_id": "opaque1234",
        "poc_id": "poc-1",
        "poc_hash": digest,
        "vul_exit_code": 1,
        "fix_exit_code": 0,
    }
    calls = []

    monkeypatch.setattr(executor, "start", lambda: calls.append("start"))
    monkeypatch.setattr(executor, "_task_network_plan", lambda *_args: object())
    monkeypatch.setattr(executor, "_opaque_workspace_path", lambda *_args: workspace)
    monkeypatch.setattr(executor, "_generate", lambda *_args: calls.append("generate"))
    monkeypatch.setattr(executor, "_workspace", lambda *_args: "workspace-container")

    def submit(_task, workspace_dir, _container):
        assert (workspace_dir / "final.poc").read_bytes() == b"historical-poc"
        calls.append("submit")
        return {"poc_id": "poc-1"}, digest, "opaque1234"

    monkeypatch.setattr(executor, "_submit_final", submit)
    monkeypatch.setattr(executor, "_ensure_key", lambda: "test-key")
    monkeypatch.setattr(
        executor,
        "_server_http",
        lambda *_args, **_kwargs: {"message": "verified", "poc_ids": ["poc-1"]},
    )
    monkeypatch.setattr(executor, "_private_query", lambda *_args: [record])
    monkeypatch.setattr(
        executor,
        "_gateway_wait",
        lambda *_args, **_kwargs: pytest.fail("regrade must not start an Ouroboros task"),
    )

    outcome = executor.regrade_final_poc(
        task,
        source,
        config.run_root / "regrade-task",
        attempt_id="historical-1",
    )

    assert calls == ["start", "generate", "submit"]
    assert outcome["source_final_poc_sha256"] == digest
    assert outcome["classification"]["official_success"] is True


def test_fair_terminal_missing_marker_is_typed_and_settles_cost(tmp_path, monkeypatch):
    gateway_result = {
        "status": "completed",
        "observed_model": "deepseek/deepseek-v4-flash-0731",
        "observed_provider": "backend-a",
        "reasoning_effort": "high",
        "prompt_tokens": 185_217,
        "completion_tokens": 754,
        "cost_usd": 0.019249,
        "cost_final": True,
        "cost_breakdown": {
            "accounted_upper_bound_usd": 0.019249,
            "cost_final": True,
        },
        "outcome_axes": {"execution": {"status": "ok"}},
    }
    config, executor = _stub_terminal_task_executor(
        tmp_path, monkeypatch, gateway_result
    )

    rows = run_campaign(
        ["arvo:47101"],
        run_root=config.run_root,
        executor=executor.run_task,
        estimated_cost_usd=1,
        budget_cap_usd=2,
    )

    assert rows[0]["status"] == "failed"
    assert rows[0]["capability_outcome"] == CAPABILITY_FINAL_POC_MISSING
    assert rows[0]["final_submission_success"] is False
    assert rows[0]["prompt_tokens"] == 185_217
    assert rows[0]["completion_tokens"] == 754
    assert rows[0]["cost_usd"] == pytest.approx(0.019249)
    projection = BudgetLedger(config.run_root / "claims.jsonl", cap_usd=2).projection()
    assert projection.settled_usd == pytest.approx(0.019249)
    assert projection.unresolved_upper_bound_usd == 0


def test_fair_terminal_leftover_dsml_is_protocol_fail(tmp_path, monkeypatch):
    leftover = (
        "<｜DSML｜tool_calls>"
        "<｜DSML｜invoke name=\"run_shell\">broken"
    )
    gateway_result = {
        "status": "completed",
        "content": leftover,
        "observed_model": "deepseek/deepseek-v4-flash-0731",
        "observed_provider": "relace",
        "reasoning_effort": "high",
        "prompt_tokens": 100,
        "completion_tokens": 10,
        "cost_usd": 0.02,
        "cost_final": True,
        "cost_breakdown": {
            "accounted_upper_bound_usd": 0.02,
            "cost_final": True,
        },
        "outcome_axes": {"execution": {"status": "ok"}},
    }
    config, executor = _stub_terminal_task_executor(
        tmp_path, monkeypatch, gateway_result
    )
    rows = run_campaign(
        ["arvo:1"],
        run_root=config.run_root,
        executor=executor.run_task,
        estimated_cost_usd=1,
        budget_cap_usd=2,
    )
    assert rows[0]["status"] == "infra_failed"
    assert rows[0]["lifecycle"] == PROTOCOL_FAIL
    assert rows[0]["infra_reason"] == PROTOCOL_FAIL
    assert rows[0]["capability_outcome"] == ""


def test_fair_terminal_prose_without_markup_is_capability_missing_poc(
    tmp_path, monkeypatch
):
    gateway_result = {
        "status": "completed",
        "content": (
            "I inspected the Baidu parser with 11 tools and wrote a long "
            "analysis. No final.poc was produced."
        ),
        "observed_model": "deepseek/deepseek-v4-flash-0731",
        "observed_provider": "backend-a",
        "reasoning_effort": "high",
        "prompt_tokens": 200,
        "completion_tokens": 80,
        "cost_usd": 0.03,
        "cost_final": True,
        "cost_breakdown": {
            "accounted_upper_bound_usd": 0.03,
            "cost_final": True,
        },
        "outcome_axes": {"execution": {"status": "ok"}},
    }
    config, executor = _stub_terminal_task_executor(
        tmp_path, monkeypatch, gateway_result
    )
    rows = run_campaign(
        ["arvo:1065"],
        run_root=config.run_root,
        executor=executor.run_task,
        estimated_cost_usd=1,
        budget_cap_usd=2,
    )
    assert rows[0]["status"] == "failed"
    assert rows[0]["capability_outcome"] == CAPABILITY_FINAL_POC_MISSING
    assert rows[0]["lifecycle"] != PROTOCOL_FAIL
    assert rows[0]["infra_reason"] == ""


def test_terminal_telemetry_failure_preserves_settled_cost(tmp_path, monkeypatch):
    gateway_result = {
        "status": "completed",
        "observed_model": "deepseek/deepseek-v4-flash-0731",
        "reasoning_effort": "high",
        "prompt_tokens": 100,
        "completion_tokens": 10,
        "cost_usd": 0.25,
        "cost_final": True,
        "cost_breakdown": {
            "accounted_upper_bound_usd": 0.25,
            "cost_final": True,
        },
        "outcome_axes": {"execution": {"status": "ok"}},
    }
    config, executor = _stub_terminal_task_executor(
        tmp_path, monkeypatch, gateway_result
    )

    rows = run_campaign(
        ["arvo:1"],
        run_root=config.run_root,
        executor=executor.run_task,
        estimated_cost_usd=1,
        budget_cap_usd=2,
    )

    assert rows[0]["status"] == "infra_failed"
    assert rows[0]["lifecycle"] == "post_gateway_evaluation_failed"
    assert rows[0]["cost_usd"] == pytest.approx(0.25)
    assert rows[0]["cost_final"] is True and rows[0]["cost_estimated"] is False
    projection = BudgetLedger(config.run_root / "claims.jsonl", cap_usd=2).projection()
    assert projection.settled_usd == pytest.approx(0.25)
    assert projection.unresolved_upper_bound_usd == 0


def test_missing_marker_with_failed_execution_stays_infra(tmp_path, monkeypatch):
    gateway_result = {
        "status": "completed",
        "observed_model": "deepseek/deepseek-v4-flash-0731",
        "observed_provider": "backend-a",
        "reasoning_effort": "high",
        "prompt_tokens": 100,
        "completion_tokens": 10,
        "cost_usd": 0.25,
        "cost_final": True,
        "cost_breakdown": {
            "accounted_upper_bound_usd": 0.25,
            "cost_final": True,
        },
        "outcome_axes": {"execution": {"status": "infra_failed"}},
    }
    config, executor = _stub_terminal_task_executor(
        tmp_path, monkeypatch, gateway_result
    )

    rows = run_campaign(
        ["arvo:1"],
        run_root=config.run_root,
        executor=executor.run_task,
        estimated_cost_usd=1,
        budget_cap_usd=2,
    )

    assert rows[0]["status"] == "infra_failed"
    assert rows[0]["capability_outcome"] == ""
    assert rows[0]["final_submission_success"] is None
    projection = BudgetLedger(config.run_root / "claims.jsonl", cap_usd=2).projection()
    assert projection.settled_usd == pytest.approx(0.25)
    assert projection.unresolved_upper_bound_usd == 0


def test_cleanup_diagnostic_failure_does_not_erase_terminal_cost(
    tmp_path, monkeypatch
):
    gateway_result = {
        "status": "completed",
        "observed_model": "deepseek/deepseek-v4-flash-0731",
        "observed_provider": "backend-a",
        "reasoning_effort": "high",
        "prompt_tokens": 100,
        "completion_tokens": 10,
        "cost_usd": 0.25,
        "cost_final": True,
        "cost_breakdown": {
            "accounted_upper_bound_usd": 0.25,
            "cost_final": True,
        },
        "outcome_axes": {"execution": {"status": "ok"}},
    }
    config, executor = _stub_terminal_task_executor(
        tmp_path, monkeypatch, gateway_result
    )

    def cleanup_failed(*_args, **_kwargs):
        raise ExecutorFailure("cleanup failed")

    original_write_json = executor_module._write_json

    def fail_cleanup_report(path, value):
        if pathlib.Path(path).name == "workspace_cleanup.json":
            raise OSError("cleanup report failed")
        return original_write_json(path, value)

    monkeypatch.setattr(executor, "_cleanup_workspace_container", cleanup_failed)
    monkeypatch.setattr(executor_module, "_write_json", fail_cleanup_report)
    rows = run_campaign(
        ["arvo:1"],
        run_root=config.run_root,
        executor=executor.run_task,
        estimated_cost_usd=1,
        budget_cap_usd=2,
    )

    assert rows[0]["status"] == "failed"
    assert rows[0]["capability_outcome"] == CAPABILITY_FINAL_POC_MISSING
    projection = BudgetLedger(config.run_root / "claims.jsonl", cap_usd=2).projection()
    assert projection.settled_usd == pytest.approx(0.25)
    assert projection.unresolved_upper_bound_usd == 0


def test_pre_gateway_failures_settle_zero_and_do_not_block_next_task(
    tmp_path, monkeypatch
):
    config = _config(tmp_path, provider_probe=False)
    executor = CyberGymExecutor(config)
    monkeypatch.setattr(executor, "start", lambda: None)

    def fail_generation(*_args, **_kwargs):
        raise ExecutorFailure("generation failed")

    monkeypatch.setattr(executor, "_generate", fail_generation)
    rows = run_campaign(
        ["arvo:1", "arvo:2"],
        run_root=config.run_root,
        executor=executor.run_task,
        estimated_cost_usd=1,
        budget_cap_usd=1,
    )

    assert [row["status"] for row in rows] == ["infra_failed", "infra_failed"]
    assert all(row["cost_usd"] == 0 for row in rows)
    assert all(row["cost_status"] == "known_no_dispatch" for row in rows)
    projection = BudgetLedger(config.run_root / "claims.jsonl", cap_usd=1).projection()
    assert projection.settled_usd == 0
    assert projection.unresolved_upper_bound_usd == 0
    assert projection.can_dispatch is True


def test_post_admission_status_error_is_not_reclassified_as_zero_cost(
    tmp_path, monkeypatch
):
    config = _config(tmp_path, provider_probe=False)
    executor = CyberGymExecutor(config)
    monkeypatch.setattr(executor, "start", lambda: None)
    monkeypatch.setattr(executor, "_generate", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        executor_module,
        "_install_workspace_backend_alias",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(executor, "_workspace", lambda *_args, **_kwargs: "container-a")
    monkeypatch.setattr(
        executor,
        "_task_body",
        lambda task, *_args, **_kwargs: {"task_id": "cybergym-" + task.task_id.replace(":", "-")},
    )

    def status_failed(*_args, **_kwargs):
        raise ExecutorFailure("Ouroboros task status returned HTTP 404")

    monkeypatch.setattr(executor, "_gateway_wait", status_failed)
    rows = run_campaign(
        ["arvo:1"],
        run_root=config.run_root,
        executor=executor.run_task,
        estimated_cost_usd=1,
        budget_cap_usd=2,
    )

    assert rows[0]["status"] == "infra_failed"
    assert rows[0]["cost_usd"] is None
    projection = BudgetLedger(config.run_root / "claims.jsonl", cap_usd=2).projection()
    # No terminal gateway frame and no measured spend: the claim settles
    # terminally at its reservation instead of staying unresolved forever.
    assert projection.settled_usd == pytest.approx(1)
    assert projection.unresolved_upper_bound_usd == 0
    assert projection.projected_usd == pytest.approx(1)
    assert projection.can_dispatch is True


def test_explicit_final_with_excluded_vul_exit_and_missing_fix_records_failure():
    """A determinate vul-excluded failure binds without a fix-side code."""
    from devtools.benchmarks.cybergym.cybergym_adapter import build_task_result_row

    digest = "b" * 64
    trial = {
        "trial_id": "final",
        "poc_hash": digest,
        "vul_exit_code": 0,
        "fix_exit_code": None,
        "is_final": True,
    }
    row = build_task_result_row(
        "arvo:3",
        trials=[trial],
        final_trial=trial,
        final_poc_sha256=digest,
        status="completed",
    )
    assert row["status"] == "completed"
    assert row["official_success"] is False
    assert row["final_submission_status"] == "known_failure"
    assert row["final_submission_reason"] == "vul_exit_excluded"


def test_explicit_final_with_missing_vul_exit_still_refused():
    """A missing vulnerable exit keeps the binding refusal."""
    from devtools.benchmarks.cybergym.cybergym_adapter import build_task_result_row

    digest = "c" * 64
    trial = {
        "trial_id": "final",
        "poc_hash": digest,
        "vul_exit_code": None,
        "fix_exit_code": 0,
        "is_final": True,
    }
    with pytest.raises(ValueError, match="must include both raw exit codes"):
        build_task_result_row(
            "arvo:4",
            trials=[trial],
            final_trial=trial,
            final_poc_sha256=digest,
            status="completed",
        )


# r9 (2026-09-04): nine runs that finished on their own, 40-90 min before the
# deadline, with a final message and no marker, were typed infrastructure
# (FinalPocRefused) because the runtime marked execution `degraded` over the
# model's own malformed run_command argument JSON. The fair-completion verdict
# now sees through agent-attributable tool errors and discloses its basis.


def _envelope(execution: dict) -> dict:
    return {"status": "completed", "result": {"outcome_axes": {
        "lifecycle": {"status": "completed"}, "execution": execution,
    }}}


def _tool_error(tool: str, text: str, status: str = "error") -> dict:
    return {"tool": tool, "status": status, "exit_code": None, "signal": None, "result": text}


def test_fair_completion_execution_ok_and_agent_attributable_tool_errors():
    from devtools.benchmarks.cybergym.cybergym_wire import _gateway_fair_completion

    assert _gateway_fair_completion(_envelope({"status": "ok"})) == (True, "execution_ok")
    degraded = _envelope({
        "status": "degraded", "reason_code": "tool_failure",
        "failure": {"kind": "tool", "reason_code": "tool_failure", "tool_errors": [
            _tool_error("run_command", "⚠️ TOOL_ARG_ERROR: Could not parse arguments for 'run_command': Expecting ',' delimiter: line 1 column 257 (char 256)"),
            _tool_error("search_code", "⚠️ TOOL_ARG_ERROR (search_code): invalid arguments for search_code. Accepted parameters: query, path"),
            _tool_error("edit_text", "⚠️ STR_REPLACE_ERROR: old_str not found in sweep.py."),
            _tool_error("list_files", "⚠️ LIST_FILES_ERROR: Directory not found: tmp"),
            _tool_error("run_command", "⚠️ TOOL_TIMEOUT (run_command): command exceeded the per-command timeout of 900s", status="timeout"),
            # full1507 arvo:1461 (2026-09-08): the model hallucinated a `bash`
            # tool after twenty successful run_command calls and finalized
            # without a marker — the model's own defect, fair completion.
            _tool_error("bash", "⚠️ Unknown tool: bash. Available: apply_patch, run_command, ...", status="unknown_tool"),
        ]},
    })
    assert _gateway_fair_completion(degraded) == (True, "agent_attributable_tool_errors")


def test_fair_completion_dead_extension_unknown_tool_text_stays_infra():
    from devtools.benchmarks.cybergym.cybergym_wire import _gateway_fair_completion

    # A dead extension composes the SAME "Unknown tool:" sentence but with the
    # typed status `unavailable`: that is the substrate's answer, not the
    # model's hallucination, so the run keeps the infrastructure verdict.
    degraded = _envelope({
        "status": "degraded", "reason_code": "tool_failure",
        "failure": {"kind": "tool", "reason_code": "tool_failure", "tool_errors": [
            _tool_error("ext_nmap", "⚠️ Unknown tool: ext_nmap. Available: apply_patch, run_command, ...", status="unavailable"),
        ]},
    })
    assert _gateway_fair_completion(degraded) == (False, "degraded_runtime_tool_error")


@pytest.mark.parametrize("execution, basis", [
    ({"status": "failed"}, "execution_failed"),
    ({"status": "infra_failed"}, "execution_infra_failed"),
    ({"status": "best_effort"}, "execution_best_effort"),
    ({}, "execution_missing"),
    ({"status": "degraded", "failure": {"kind": "finalization_control", "reason_code": "delivery_control_degraded"}},
     "degraded_non_tool_failure"),
    ({"status": "degraded", "failure": {"kind": "tool", "reason_code": "tool_failure", "tool_errors": []}},
     "degraded_tool_failure_without_errors"),
    ({"status": "degraded", "failure": {"kind": "tool", "reason_code": "tool_failure", "tool_errors": [
        _tool_error("run_command", "⚠️ TOOL_ARG_ERROR: Could not parse arguments"),
        _tool_error("run_command", "⚠️ TOOL_ERROR (run_command): RuntimeError: Could not determine home directory."),
    ]}}, "degraded_runtime_tool_error"),
    ({"status": "degraded", "failure": {"kind": "tool", "reason_code": "tool_failure", "tool_errors": ["oops"]}},
     "degraded_untyped_tool_error"),
])
def test_fair_completion_refuses_everything_else(execution, basis):
    from devtools.benchmarks.cybergym.cybergym_wire import _gateway_fair_completion

    assert _gateway_fair_completion(_envelope(execution)) == (False, basis)


def test_fair_completion_reads_outermost_execution_axis():
    from devtools.benchmarks.cybergym.cybergym_wire import _gateway_fair_completion

    payload = {"outcome_axes": {"execution": {"status": "ok"}},
               "result": {"outcome_axes": {"execution": {"status": "failed"}}}}
    assert _gateway_fair_completion(payload) == (True, "execution_ok")
    assert _gateway_fair_completion({"result": {"task_result": _envelope({"status": "failed"})}}) == (False, "execution_failed")


def _verified_response_refs(
    run_root: pathlib.Path, disclosure: dict, *, call_id: str = "llm-1"
) -> dict:
    """Write one run-local call manifest + blob the wire reader accepts."""
    blob_raw = json.dumps(
        {"usage": {"request_wire": disclosure, "response_provider": "backend-a"}},
        sort_keys=True,
    ).encode("utf-8")
    blob_path = run_root / "observability" / "blobs" / (f"{'c' * 64}.json.gz")
    blob_path.parent.mkdir(parents=True, exist_ok=True)
    blob_path.write_bytes(gzip.compress(blob_raw))
    manifest_raw = json.dumps(
        {
            "task_id": "opaque",
            "call_id": f"{call_id}_response",
            "llm_call_id": call_id,
            "full_payload_ref": {
                "path": str(blob_path),
                "sha256": hashlib.sha256(blob_raw).hexdigest(),
                "size": len(blob_raw),
                "kind": "json",
                "encoding": "gzip",
            },
        },
        sort_keys=True,
    ).encode("utf-8")
    manifest_path = run_root / "observability" / "calls" / "opaque" / f"{call_id}_response.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_bytes(manifest_raw)
    return {
        "path": str(manifest_path),
        "sha256": hashlib.sha256(manifest_raw).hexdigest(),
        "call_id": f"{call_id}_response",
    }

def test_unsettled_wire_disclosure_reports_effort_but_never_buys_the_cost_gate(
    tmp_path, monkeypatch
):
    """An unsettled attempt may still disclose its effort; cost finality is separate."""
    config = _config(
        tmp_path,
        provider_probe=True,
        expected_data_sha256="a" * 64,
        expected_binary_sha256="b" * 64,
    )
    disclosure = {
        "requested_effort": "high",
        "applied_effort": "high",
        "requested_tool_dialect": "function",
        "applied_tool_dialect": "function",
        "reason_code": "requested_wire_form",
        "source_profile_fingerprint": "d" * 64,
        "accepted_profile_fingerprint": "d" * 64,
        "attempt_id": "attempt-unsettled",
        "candidate_sha256": "e" * 64,
        "ladder_ordinal": 1,
        "applied_actions": [],
        "task_local": False,
    }
    response_ref = _verified_response_refs(config.run_root, disclosure)
    gateway_result = {
        "status": "completed",
        "model": "requested/not-served",
        "prompt_tokens": 12,
        "completion_tokens": 4,
        "cost_usd": 0.07,
        "accounted_upper_bound_usd": 0.07,
        "cost_estimated": False,
        "cost_final": False,
        "trace_refs": {
            "llm_call_refs": [
                {
                    "llm_call_id": "llm-1",
                    "resolved_model": config.model,
                    "provider": "provider-a",
                    "response_ref": response_ref,
                }
            ]
        },
    }

    served = _served_telemetry(gateway_result, allowed_roots=(config.run_root,))
    assert served["observed_effort"] == "high"
    assert served["effort_source"] == "served_response_wire"
    assert served["response_wire_effort_count"] == served["trace_call_count"] == 1
    assert served["response_wire_provider_count"] == 1

    executor = CyberGymExecutor(config)
    monkeypatch.setattr(executor, "start", lambda: None)
    monkeypatch.setattr(executor, "_generate", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        executor_module, "_install_workspace_backend_alias", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(executor, "_workspace", lambda *_args, **_kwargs: "container-a")
    monkeypatch.setattr(executor, "_cleanup_workspace_container", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(executor, "_ensure_key", lambda: "test-key")
    monkeypatch.setattr(
        executor, "_attest_runtime", lambda *_args, **_kwargs: {"status": "ok"}
    )
    monkeypatch.setattr(
        executor,
        "_task_body",
        lambda task, *_args, **_kwargs: {"task_id": "cybergym-" + task.task_id.replace(":", "-")},
    )
    monkeypatch.setattr(executor, "_gateway_wait", lambda *_args, **_kwargs: gateway_result)
    rows = run_campaign(
        ["arvo:1"],
        run_root=config.run_root,
        executor=executor.run_task,
        estimated_cost_usd=1,
        budget_cap_usd=2,
    )

    assert rows[0]["status"] == "infra_failed"
    assert rows[0]["lifecycle"] == "post_gateway_evaluation_failed"
    assert "cost is unknown or estimated" in rows[0]["error"]
