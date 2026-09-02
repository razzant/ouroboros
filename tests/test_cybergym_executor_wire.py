"""CyberGym gateway wire-protocol, served-telemetry, and accounting tests.

Split from ``tests/test_cybergym_executor.py`` along the HTTP/gateway seam:
provider probe, submit/verify/private-query wire parsing, served-telemetry
validation, gateway admission/cancel custody, and campaign cost accounting.
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


def test_unknown_gateway_attempt_blocks_campaign_cleanup(tmp_path):
    config = _config(tmp_path)
    calls = []

    def command(*args, **kwargs):
        calls.append(args)
        raise AssertionError("cleanup must not run while gateway custody is unknown")

    executor = CyberGymExecutor(dataclasses_replace(config, command_runner=command))
    executor.started = True
    executor.server_id = "server-123"
    executor.network_id = "network-123"
    executor._task_containers = {"workspace-agent-aaaaaaaaaaaaaaaaaaaaaaaa": "workspace-123"}
    executor._gateway_attempts = {
        "cybergym-attempt": {
            "gateway_task_id": "cybergym-attempt",
            "status": "admission_unknown",
            "checkpoint": str(config.run_root / "checkpoint.json"),
        }
    }

    report = executor.close()
    assert report["ok"] is False
    assert report["status"] == "custody_pending"
    assert executor.custody_blocked is True
    assert executor.server_id == "server-123"
    assert executor.network_id == "network-123"
    assert calls == []
    assert (config.run_root / "custody_pending.json").is_file()


def test_gateway_admission_transport_error_registers_durable_custody(tmp_path):
    config = _config(tmp_path, provider_probe=False)
    seen = {}

    def failing_http(*args, **kwargs):
        seen.update(kwargs)
        raise ExecutorFailure("HTTP POST transport failed")

    executor = CyberGymExecutor(dataclasses_replace(config, http_runner=failing_http))
    checkpoint = config.run_root / "checkpoint.json"
    body = {"task_id": "cybergym-opaque-attempt", "description": "test"}
    with pytest.raises(ExecutorFailure, match="transport failed"):
        executor._gateway_wait(body, checkpoint)
    assert "cybergym-opaque-attempt" in executor._gateway_attempts
    assert executor._gateway_attempts["cybergym-opaque-attempt"]["status"] == "admission_unknown"
    assert seen["headers"]["Idempotency-Key"].startswith("cybergym-")
    saved = json.loads(checkpoint.read_text(encoding="utf-8"))
    assert saved["custody_required"] is True
    assert saved["status"] == "admission_unknown"


def test_gateway_definitive_admission_rejection_releases_phantom_custody(tmp_path):
    config = _config(tmp_path, provider_probe=False)
    executor = CyberGymExecutor(
        dataclasses_replace(
            config,
            http_runner=lambda *args, **kwargs: {
                "status_code": 400,
                "body": {"detail": "invalid task"},
            },
        )
    )
    checkpoint = config.run_root / "checkpoint.json"
    body = {"task_id": "cybergym-rejected-attempt", "description": "test"}
    with pytest.raises(ExecutorFailure, match="HTTP 400"):
        executor._gateway_wait(body, checkpoint)
    assert executor._gateway_attempts == {}
    saved = json.loads(checkpoint.read_text(encoding="utf-8"))
    assert saved["status"] == "admission_rejected"
    assert saved["custody_required"] is False


def test_gateway_malformed_admission_keeps_unknown_custody(tmp_path):
    config = _config(tmp_path, provider_probe=False)
    executor = CyberGymExecutor(
        dataclasses_replace(config, http_runner=lambda *args, **kwargs: {})
    )
    checkpoint = config.run_root / "checkpoint.json"
    body = {"task_id": "cybergym-malformed-attempt", "description": "test"}
    with pytest.raises(ExecutorFailure, match="no task id"):
        executor._gateway_wait(body, checkpoint)
    assert "cybergym-malformed-attempt" in executor._gateway_attempts
    saved = json.loads(checkpoint.read_text(encoding="utf-8"))
    assert saved["status"] == "admission_unknown_response"
    assert saved["custody_required"] is True


def test_gateway_waits_for_final_cost_after_completed_status(tmp_path):
    config = _config(tmp_path, provider_probe=False, task_timeout_sec=10)
    task_id = "cybergym-cost-pending"
    calls = []
    status_rows = iter(
        (
            {
                "task_id": task_id,
                "status": "completed",
                "result": {"cost_final": False},
            },
            {
                "task_id": task_id,
                "status": "completed",
                "result": {"cost_final": True},
            },
        )
    )

    def http(method, url, **kwargs):
        calls.append(method)
        if method == "POST":
            return {"task_id": task_id, "status": "scheduled"}
        return next(status_rows)

    executor = CyberGymExecutor(
        dataclasses_replace(config, http_runner=http, sleep=lambda _seconds: None)
    )
    result = executor._gateway_wait(
        {"task_id": task_id, "description": "test"},
        config.run_root / "checkpoint.json",
    )

    assert result["result"]["cost_final"] is True
    assert calls == ["POST", "GET", "GET"]


def test_gateway_cost_finality_conflict_keeps_polling(tmp_path):
    config = _config(tmp_path, provider_probe=False, task_timeout_sec=10)
    task_id = "cybergym-cost-conflict"
    calls = []
    status_rows = iter(
        (
            {
                "task_id": task_id,
                "status": "completed",
                "cost_final": True,
                "cost_breakdown": {"cost_final": False},
            },
            {
                "task_id": task_id,
                "status": "completed",
                "cost_final": True,
                "cost_breakdown": {"cost_final": True},
            },
        )
    )

    def http(method, _url, **_kwargs):
        calls.append(method)
        if method == "POST":
            return {"task_id": task_id, "status": "scheduled"}
        return next(status_rows)

    executor = CyberGymExecutor(
        dataclasses_replace(config, http_runner=http, sleep=lambda _seconds: None)
    )
    result = executor._gateway_wait(  # noqa: SLF001 - accounting contract
        {"task_id": task_id, "description": "test"},
        config.run_root / "checkpoint.json",
    )

    assert result["cost_breakdown"]["cost_final"] is True
    assert calls == ["POST", "GET", "GET"]


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
    from ouroboros.tool_call_markup import _DSML_MARK

    leftover = (
        f"<{_DSML_MARK}tool_calls>"
        f"<{_DSML_MARK}invoke name=\"run_shell\">broken"
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
    projection = BudgetLedger(config.run_root / "claims.jsonl", cap_usd=2).projection()
    assert projection.settled_usd == 0
    assert projection.unresolved_upper_bound_usd == pytest.approx(0.25)


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
    assert projection.settled_usd == 0
    assert projection.unresolved_upper_bound_usd == pytest.approx(1)
    assert projection.projected_usd == pytest.approx(1)
    assert projection.can_dispatch is True


def test_cancel_503_recovers_terminal_gateway_payload(tmp_path):
    config = _config(tmp_path, poll_interval_sec=0)
    task_id = "cybergym-cancel-503"
    terminal = {
        "task_id": task_id,
        "status": "failed",
        "cost_usd": 0.060914,
        "accounted_upper_bound_usd": 0.060914,
        "unresolved_upper_bound_usd": 0.020062,
        "cost_final": False,
    }
    calls = []
    responses = iter(
        (
            {"status_code": 503, "body": {"detail": "teardown still live"}},
            {"status_code": 200, "body": terminal},
        )
    )

    def http(method, _url, **_kwargs):
        calls.append(method)
        return next(responses)

    executor = CyberGymExecutor(dataclasses_replace(config, http_runner=http))
    executor._gateway_attempts[task_id] = {  # noqa: SLF001 - custody assertion
        "gateway_task_id": task_id,
        "status": "submitted",
    }
    checkpoint = config.run_root / "checkpoint.json"
    result = executor._cancel_gateway_task(task_id, checkpoint)  # noqa: SLF001

    assert result == terminal
    assert calls == ["POST", "GET"]
    assert task_id not in executor._gateway_attempts
    saved = json.loads(checkpoint.read_text(encoding="utf-8"))
    assert saved["status"] == "failed"
    assert saved["cancel_status_code"] == 503
    assert saved["result"]["accounted_upper_bound_usd"] == pytest.approx(0.060914)


def test_cancel_auth_failure_does_not_fallback_to_get(tmp_path):
    config = _config(tmp_path, poll_interval_sec=0)
    task_id = "cybergym-cancel-auth"
    calls = []

    def http(method, _url, **_kwargs):
        calls.append(method)
        return {"status_code": 401, "body": {"detail": "unauthorized"}}

    executor = CyberGymExecutor(dataclasses_replace(config, http_runner=http))
    executor._gateway_attempts[task_id] = {  # noqa: SLF001 - custody assertion
        "gateway_task_id": task_id,
        "status": "submitted",
    }
    with pytest.raises(ExecutorFailure, match="cancellation request failed"):
        executor._cancel_gateway_task(task_id, config.run_root / "checkpoint.json")  # noqa: SLF001
    assert calls == ["POST"]
    assert task_id in executor._gateway_attempts


def test_cancel_503_with_get_failure_keeps_custody_block(tmp_path):
    config = _config(tmp_path, poll_interval_sec=0)
    task_id = "cybergym-cancel-no-terminal"
    calls = []

    def http(method, _url, **_kwargs):
        calls.append(method)
        if method == "POST":
            return {"status_code": 503, "body": {"detail": "teardown still live"}}
        raise ExecutorFailure("status transport failed")

    executor = CyberGymExecutor(dataclasses_replace(config, http_runner=http))
    executor._gateway_attempts[task_id] = {  # noqa: SLF001 - custody assertion
        "gateway_task_id": task_id,
        "status": "submitted",
    }
    with pytest.raises(ExecutorFailure, match="status transport failed"):
        executor._cancel_gateway_task(task_id, config.run_root / "checkpoint.json")  # noqa: SLF001
    assert calls == ["POST", "GET"]
    assert task_id in executor._gateway_attempts


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
