"""Pure CyberGym protocol and ledger tests; no upstream or Docker dependency."""

from __future__ import annotations

import hashlib
import json
import pathlib

import pytest

from devtools.benchmarks.cybergym.cybergym_adapter import (
    CAPABILITY_FINAL_POC_MISSING,
    DEFAULT_FINAL_POC_PATH,
    DEFAULT_LEVEL,
    OFFICIAL_MODEL,
    BudgetLedger,
    BudgetOverspend,
    BudgetRefused,
    ClaimRefused,
    CyberGymIntegrationUnavailable,
    CyberGymPinRefused,
    FinalPocRefused,
    LedgerError,
    assert_fresh_output_root,
    build_generate_task_argv,
    build_task_result_row,
    classify_official_exit,
    directory_tree_digest,
    final_poc_record,
    final_submission,
    official_pin_skip_reason,
    parse_strict_bool,
    pre_admission_report,
    project_budget,
    run_campaign,
    safe_task_id,
    safe_task_path,
    task_contract_metadata,
    validate_high_effort,
    validate_model_pin,
    validate_positive_finite,
    validate_positive_integral,
    verify_mask_map,
)
from ouroboros.configured_subagents import parse_configured_subagents
from ouroboros.reviewer_slot_config import parse_reviewer_slots


def test_safe_ids_and_argv_are_path_safe(tmp_path):
    assert safe_task_id("arvo:47101") == "arvo:47101"
    assert safe_task_path(tmp_path, "arvo:47101").name == "arvo__47101"
    with pytest.raises(ValueError):
        safe_task_id("../../etc:passwd")
    argv = build_generate_task_argv(
        "oss-fuzz:42535201",
        out_dir=tmp_path / "task",
        data_dir=tmp_path / "data",
        server="http://cybergym-internal:8666",
        mask_map=tmp_path / "mask.json",
        agent_id="lane-1",
        with_flag=True,
    )
    assert argv[:4] == [argv[0], "-m", "cybergym.task.gen_task", "--task-id"]
    assert "--with-flag" in argv
    assert all(isinstance(part, str) for part in argv)


def test_official_pin_skips_arvo_64622_without_calling_executor(tmp_path):
    assert official_pin_skip_reason("arvo:64622") == "broken_symlink_official_pin"
    assert official_pin_skip_reason("arvo:1065") == ""

    called: list[str] = []

    def boom(task, task_dir):
        called.append(task.task_id)
        raise AssertionError("executor must not run a skipped official pin")

    rows = run_campaign(
        ["arvo:64622", "arvo:1"],
        run_root=tmp_path / "pin-skip",
        executor=boom,
        estimated_cost_usd=1,
        budget_cap_usd=5,
    )
    assert called == ["arvo:1"]
    assert rows[0]["task_id"] == "arvo:64622"
    assert rows[0]["status"] == "infra_failed"
    assert rows[0]["lifecycle"] == "broken_symlink_official_pin"
    assert rows[0]["infra_reason"] == "broken_symlink_official_pin"


def test_applied_server_provenance_rewrites_manifest_command(tmp_path):
    from types import SimpleNamespace

    import devtools.benchmarks.cybergym.run_cybergym as launcher

    args = SimpleNamespace(
        server="http://cybergym-internal:8666",
        data_root=tmp_path / "data",
        mask_map=tmp_path / "mask.json",
        difficulty=DEFAULT_LEVEL,
    )
    manifest = {"harness": {"server": args.server}, "official_command": []}
    applied = "http://cybergym-server-campaign:8666"

    launcher._apply_server_provenance(manifest, args, applied)

    assert manifest["harness"] == {
        "server": applied,
        "requested_server": args.server,
        "applied_server": applied,
    }
    assert applied in manifest["official_command"]
    assert args.server not in manifest["official_command"]


def test_pre_admission_is_pure_and_fail_closed(tmp_path):
    report = pre_admission_report(
        task_ids=["arvo:1"],
        output_root=tmp_path / "out",
        repo_dir=tmp_path / "repo",
        source_root=tmp_path / "source",
        data_root=tmp_path / "data",
        settings_path=tmp_path / "settings.json",
        require_settings=True,
        server_url="http://cybergym-internal:8666",
        model="deepseek/deepseek-v4-flash-0731",
    )
    assert report["ok"]
    assert not (tmp_path / "out").exists()
    denied = pre_admission_report(
        task_ids=["arvo:1"],
        output_root=tmp_path / "repo" / "out",
        repo_dir=tmp_path / "repo",
        server_url="http://0.0.0.0:8666",
        model="m",
    )
    assert not denied["ok"]
    assert "output_root_overlaps_repo" in denied["reasons"]
    assert "server_url_wildcard_host" in denied["reasons"]


def test_runtime_paths_are_confined_and_binary_is_nested(tmp_path):
    repo = tmp_path / "repo"
    denied = pre_admission_report(
        task_ids=["arvo:1"],
        output_root=tmp_path / "out",
        repo_dir=repo,
        source_root=tmp_path / "source",
        data_root=tmp_path / "data",
        mask_map=repo / "mask.json",
        server_root=tmp_path / "server",
        binary_dir=tmp_path / "elsewhere",
        require_inputs=True,
        server_url="http://cybergym-internal:8666",
        model=OFFICIAL_MODEL,
    )
    assert not denied["ok"]
    assert "mask_map_overlaps_repo" in denied["reasons"]
    assert "binary_dir_outside_server_root" in denied["reasons"]
    assert not (tmp_path / "out").exists()


def test_fresh_output_root_rejects_nonempty_and_symlink(tmp_path):
    fresh = tmp_path / "fresh"
    assert assert_fresh_output_root(fresh) == fresh
    fresh.mkdir()
    (fresh / "old.json").write_text("{}", encoding="utf-8")
    with pytest.raises(CyberGymPinRefused, match="fresh"):
        assert_fresh_output_root(fresh)
    link = tmp_path / "link"
    link.symlink_to(tmp_path / "missing", target_is_directory=True)
    with pytest.raises(CyberGymPinRefused, match="symlink"):
        assert_fresh_output_root(link)


def test_directory_digest_allows_confined_upstream_symlink_and_rejects_external(tmp_path):
    root = tmp_path / "tree"
    root.mkdir()
    target = root / "libreal.so"
    target.write_bytes(b"binary")
    link = root / "lib.so"
    link.symlink_to("libreal.so")
    first = directory_tree_digest(root)
    assert first["links"] == 1
    assert first["files"] == 1
    assert first["bytes"] == len(b"binary")
    assert directory_tree_digest(root)["sha256"] == first["sha256"]

    link.unlink()
    link.symlink_to("/src/zeek/build/install-root/share/btest/data")
    virtual = directory_tree_digest(root, allowed_virtual_symlink_prefixes=("/src/",))
    assert virtual["links"] == 1

    outside = tmp_path / "outside"
    outside.write_bytes(b"secret")
    link.unlink()
    link.symlink_to(outside)
    with pytest.raises(CyberGymPinRefused, match="external link"):
        directory_tree_digest(root)


def test_issue15_and_final_hash_binding(tmp_path):
    assert classify_official_exit(1, 0)["official_success"] is True
    assert classify_official_exit(71, 0)["official_success"] is False
    assert classify_official_exit(300, None)["official_success"] is False
    assert classify_official_exit(None, 0)["official_success"] is None
    payload = b"poc"
    marker = tmp_path / "final.poc"
    marker.write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    assert final_poc_record(tmp_path).sha256 == digest
    trial = {"trial_id": "final", "poc_id": "p1", "poc_hash": digest, "vul_exit_code": 1, "fix_exit_code": 0}
    projection = final_submission(trial, final_poc_sha256=digest)
    assert projection["final_submission_success"] is True
    assert projection["any_of_success"] is True
    with pytest.raises(FinalPocRefused):
        (tmp_path / "link.poc").symlink_to(marker)
        final_poc_record(tmp_path / "link.poc")


def test_result_row_preserves_final_and_any_of_columns():
    digest = "a" * 64
    row = build_task_result_row(
        "arvo:1",
        trials=[
            {"trial_id": "old", "poc_hash": digest, "vul_exit_code": 1, "fix_exit_code": 0},
            {"trial_id": "final", "poc_hash": "b" * 64, "vul_exit_code": 71, "fix_exit_code": 0},
        ],
        final_trial={"trial_id": "final", "poc_hash": "b" * 64, "vul_exit_code": 71, "fix_exit_code": 0},
        final_poc_sha256="b" * 64,
        status="completed",
    )
    assert row["metric_name"] == "final_submission"
    assert row["final_submission_success"] is False
    assert row["any_of_success"] is True
    assert row["trial_count"] == 2


def test_explicit_final_trial_cannot_rebind_a_stale_record():
    digest = "a" * 64
    trials = [{"trial_id": "final", "poc_id": "p1", "poc_hash": digest, "vul_exit_code": 1, "fix_exit_code": 0}]
    projection = final_submission(
        {"trial_id": "final", "poc_id": "p1", "poc_hash": "b" * 64, "vul_exit_code": 1, "fix_exit_code": 0},
        trials=trials,
    )
    assert projection["final_submission_status"] == "unknown"
    assert projection["final_submission_reason"] == "invalid_final_trial"


def test_budget_claims_are_atomic_and_unresolved_dead_releases_reserve(tmp_path):
    ledger = BudgetLedger(tmp_path / "claims.jsonl", cap_usd=5)
    ledger.claim("arvo:1", 4, attempt_id="a1")
    with pytest.raises(ClaimRefused):
        ledger.claim("arvo:1", 1, attempt_id="a2")
    ledger.mark_unresolved("a1")
    projection = ledger.projection()
    assert projection.can_dispatch is True
    assert projection.reason == "within_cap"
    assert projection.unresolved_upper_bound_usd == pytest.approx(0)
    assert projection.projected_usd == pytest.approx(0)
    assert projection.reserved_usd == pytest.approx(0)
    second = ledger.claim("arvo:2", 1, attempt_id="a2")
    assert second["attempt_id"] == "a2"
    third = ledger.claim("arvo:3", 1, attempt_id="a3")
    assert third["attempt_id"] == "a3"


def test_budget_historical_null_unresolved_does_not_refuse_catalog():
    projection = project_budget(
        [
            {"event": "claim", "task_id": "arvo:1", "attempt_id": "a1", "reserved_usd": 2},
            {"event": "unresolved", "attempt_id": "a1", "upper_bound_usd": None},
        ],
        cap_usd=10,
    )
    assert projection.can_dispatch is True
    assert projection.unresolved_upper_bound_usd == pytest.approx(0)
    assert projection.projected_usd == pytest.approx(0)
    assert "arvo:1" not in projection.active_task_ids


def test_budget_historical_claim_estimate_corpses_do_not_refuse_catalog():
    events = []
    for i in range(163):
        events.append(
            {
                "event": "claim",
                "task_id": f"arvo:{i}",
                "attempt_id": f"a{i}",
                "reserved_usd": 20,
            }
        )
        events.append(
            {"event": "unresolved", "attempt_id": f"a{i}", "upper_bound_usd": 20}
        )
    projection = project_budget(events, cap_usd=3500)
    assert projection.can_dispatch is True
    assert projection.reserved_usd == pytest.approx(0)
    assert projection.projected_usd == pytest.approx(0)
    ledger = BudgetLedger("/tmp/unused-historical-replay", cap_usd=3500)
    # Replay-only: a fresh ledger with the same events would dispatch.
    replayed = project_budget(events + [], cap_usd=3500)
    assert replayed.can_dispatch is True
    del ledger


def test_budget_live_in_flight_huge_reserve_still_blocks(tmp_path):
    ledger = BudgetLedger(tmp_path / "claims.jsonl", cap_usd=5)
    ledger.claim("arvo:1", 4, attempt_id="live")
    projection = ledger.projection()
    assert projection.can_dispatch is True
    assert projection.reserved_usd == pytest.approx(4)
    with pytest.raises(BudgetRefused):
        ledger.claim("arvo:2", 2, attempt_id="next")


def test_budget_unresolved_dead_does_not_block_even_with_huge_written_bound(tmp_path):
    ledger = BudgetLedger(tmp_path / "claims.jsonl", cap_usd=5)
    ledger.claim("arvo:1", 1, attempt_id="a1")
    ledger.mark_unresolved("a1", 100)
    projection = ledger.projection()
    assert projection.can_dispatch is True
    assert projection.unresolved_upper_bound_usd == pytest.approx(0)
    assert projection.projected_usd == pytest.approx(0)
    second = ledger.claim("arvo:2", 1, attempt_id="a2")
    assert second["attempt_id"] == "a2"


def test_budget_projection_replays_terminal_states():
    projection = project_budget(
        [
            {"event": "claim", "task_id": "arvo:1", "attempt_id": "a1", "reserved_usd": 2},
            {"event": "settle", "attempt_id": "a1", "cost_usd": 1.5},
        ],
        cap_usd=3,
    )
    assert projection.projected_usd == 1.5
    assert projection.can_dispatch
    with pytest.raises(LedgerError):
        project_budget([{"event": "settle", "attempt_id": "orphan", "cost_usd": 1}])


def test_budget_settlement_overspend_is_typed_and_stops_dispatch(tmp_path):
    ledger = BudgetLedger(tmp_path / "claims.jsonl", cap_usd=2)
    ledger.claim("arvo:1", 1, attempt_id="a1")
    with pytest.raises(BudgetOverspend):
        ledger.settle("a1", 3)
    projection = ledger.projection()
    assert projection.reason == "budget_cap_exceeded"
    assert projection.can_dispatch is False


def test_exact_model_and_positive_launcher_values_are_strict():
    assert validate_model_pin(OFFICIAL_MODEL) == OFFICIAL_MODEL
    with pytest.raises(ValueError):
        validate_model_pin("deepseek/deepseek-v4-flash")
    assert validate_positive_finite("1.5", field="budget") == 1.5
    for value in (0, -1, float("nan"), float("inf"), True, ""):
        with pytest.raises(ValueError):
            validate_positive_finite(value, field="timeout")
    assert validate_positive_integral("4.0", field="timeout") == 4
    for value in (0, -1, 1.5, float("nan"), True, ""):
        with pytest.raises(ValueError):
            validate_positive_integral(value, field="timeout")
    assert validate_high_effort("HIGH") == "high"
    with pytest.raises(ValueError):
        validate_high_effort("max")


def test_launcher_paid_limits_and_immutable_hash_declarations_are_bounded():
    from types import SimpleNamespace

    from devtools.benchmarks.cybergym.run_cybergym import _validate_launcher_values

    def args(**overrides):
        values = dict(
            model=OFFICIAL_MODEL,
            budget_usd=3500.0,
            timeout_sec=14_400,
            max_rounds=200,
            per_task_cost_usd=20.0,
            workers=32,
            per_task_estimate_usd=1.0,
            dry_run=False,
            allow_dirty_seed=False,
            expected_data_sha256="a" * 64,
            expected_binary_sha256="b" * 64,
            cybergym_python="python3",
            executor="",
            ouroboros_url="",
        )
        values.update(overrides)
        return SimpleNamespace(**values)

    for kwargs, message in (
        ({"budget_usd": 3500.01}, "budget_usd"),
        ({"timeout_sec": 14_401}, "timeout_sec"),
        ({"max_rounds": 0}, "max_rounds"),
        ({"per_task_cost_usd": 0}, "per_task_cost_usd"),
        ({"per_task_cost_usd": 3500.01}, "per_task_cost_usd"),
        ({"workers": 33}, "workers"),
        ({"allow_dirty_seed": True}, "allow-dirty-seed"),
        ({"expected_data_sha256": ""}, "expected-data-sha256"),
        ({"expected_binary_sha256": "not-a-hash"}, "expected-binary-sha256"),
    ):
        with pytest.raises(ValueError, match=message):
            _validate_launcher_values(args(**kwargs))

    normalized = args(
        workers="2", timeout_sec="120.0", max_rounds="1000.0", per_task_cost_usd="20.0"
    )
    _validate_launcher_values(normalized)
    assert normalized.workers == 2
    assert normalized.timeout_sec == 120
    assert normalized.max_rounds == 1000
    assert normalized.per_task_cost_usd == 20.0
    with pytest.raises(ValueError, match="per-task-cost-usd"):
        _validate_launcher_values(args(per_task_cost_usd=None))
    with pytest.raises(ValueError, match="cybergym-python"):
        _validate_launcher_values(args(cybergym_python=""))


def test_paid_executor_observations_are_exact_and_provider_overhead_is_settled(tmp_path):
    from devtools.benchmarks.cybergym.run_cybergym import (
        _record_provider_probe_cost,
        _validate_paid_observations,
    )

    class FakeExecutor:
        provider_observation = {
            "status": "passed",
            "observed_model": OFFICIAL_MODEL,
            "provider": "provider-a",
            "response_id": "resp-1",
            "cost_usd": 0.125,
            "cost_estimated": False,
            "secret": "must-not-be-persisted",
        }
        data_observation = {"sha256": "a" * 64, "files": 1, "bytes": 2}
        binary_observation = {"sha256": "b" * 64, "files": 1, "bytes": 3}

    provider, data, binary, cost = _validate_paid_observations(
        FakeExecutor(),
        None,
        model=OFFICIAL_MODEL,
        expected_data_sha256="a" * 64,
        expected_binary_sha256="b" * 64,
    )
    assert provider["cost_usd"] == 0.125
    assert data["sha256"] == "a" * 64
    assert binary["sha256"] == "b" * 64
    event = _record_provider_probe_cost(tmp_path, 10.0, cost)
    assert event["label"] == "provider_probe"
    ledger = BudgetLedger(tmp_path / "claims.jsonl", cap_usd=10.0)
    assert ledger.projection().settled_usd == pytest.approx(0.125)

    FakeExecutor.provider_observation = {
        **FakeExecutor.provider_observation,
        "cost_estimated": True,
    }
    with pytest.raises(CyberGymIntegrationUnavailable, match="unknown or estimated"):
        _validate_paid_observations(
            FakeExecutor(),
            None,
            model=OFFICIAL_MODEL,
            expected_data_sha256="a" * 64,
            expected_binary_sha256="b" * 64,
        )


def test_reused_input_attestation_binds_exact_paths_and_digests(tmp_path):
    from types import SimpleNamespace

    from devtools.benchmarks.cybergym.run_cybergym import (
        _load_reused_input_observations,
    )

    data_root = tmp_path / "data"
    binary_root = tmp_path / "binary"
    data_root.mkdir()
    binary_root.mkdir()
    data_digest = "a" * 64
    binary_digest = "b" * 64
    manifest = tmp_path / "run_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "created_at_unix": 1.0,
                "extra": {
                    "cybergym_data": {
                        "path": str(data_root),
                        "sha256": data_digest,
                        "expected_sha256": data_digest,
                        "files": 2,
                        "bytes": 3,
                    },
                    "cybergym_binary": {
                        "path": str(binary_root),
                        "sha256": binary_digest,
                        "expected_sha256": binary_digest,
                        "files": 4,
                        "bytes": 5,
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    args = SimpleNamespace(
        reuse_input_attestation=str(manifest),
        data_root=str(data_root),
        binary_dir=str(binary_root),
        expected_data_sha256=data_digest,
        expected_binary_sha256=binary_digest,
    )

    data, binary = _load_reused_input_observations(args)
    assert data is not None and binary is not None
    assert data["attestation_mode"] == "reused_manifest_observation"
    assert binary["attestation_source_sha256"] == hashlib.sha256(
        manifest.read_bytes()
    ).hexdigest()

    args.expected_binary_sha256 = "c" * 64
    with pytest.raises(CyberGymIntegrationUnavailable, match="digest"):
        _load_reused_input_observations(args)


def test_run_campaign_settles_known_actual_on_nonfinal_cost(tmp_path):
    """Known actual cost settles the reserve; the leftover claim estimate is not leftover UB."""

    def callback(_task, task_dir):
        (task_dir / "final.poc").write_bytes(b"poc")
        digest = hashlib.sha256(b"poc").hexdigest()
        return {
            "status": "completed",
            "observed_effort": "high",
            "trials": [
                {
                    "trial_id": "final",
                    "is_final": True,
                    "poc_hash": digest,
                    "vul_exit_code": 1,
                    "fix_exit_code": 0,
                }
            ],
            "cost_usd": 0.5,
            "cost_estimated": False,
            "cost_final": False,
        }

    rows = run_campaign(
        ["arvo:1"],
        run_root=tmp_path / "nonfinal-cost",
        executor=callback,
        estimated_cost_usd=1,
        budget_cap_usd=2,
    )
    assert rows[0]["status"] == "infra_failed"
    assert rows[0]["infra_reason"] == "cost_unverifiable"
    projection = BudgetLedger(tmp_path / "nonfinal-cost" / "claims.jsonl", cap_usd=2).projection()
    assert projection.settled_usd == pytest.approx(0.5)
    assert projection.unresolved_upper_bound_usd == pytest.approx(0)
    assert projection.can_dispatch is True


def test_run_campaign_records_terminal_total_accounted_bound_not_residual(tmp_path):
    """The outer ledger settles the known inner accounted bound, not leftover UB."""

    from devtools.benchmarks.cybergym.cybergym_adapter import _terminal_gateway_accounting

    terminal = {
        "status": "failed",
        "cost_usd": 0.060914,
        "accounted_upper_bound_usd": 0.060914,
        "unresolved_upper_bound_usd": 0.020062,
        "cost_final": False,
    }

    def callback(_task, _task_dir):
        return {
            "status": "infra_failed",
            "lifecycle": "gateway_terminal",
            "infra_reason": "failed",
            "runtime_result": terminal,
        }

    projected = _terminal_gateway_accounting(terminal)
    assert projected["cost_upper_bound_usd"] == pytest.approx(0.060914)
    assert projected["cost_upper_bound_usd"] != pytest.approx(0.020062)
    assert _terminal_gateway_accounting(
        {
            "status": "failed",
            "cost_usd": 0.001,
            "cost_breakdown": {"accounted_upper_bound_usd": 0.060914},
        }
    )["cost_upper_bound_usd"] == pytest.approx(0.060914)
    assert _terminal_gateway_accounting(
        {"status": "failed", "unresolved_upper_bound_usd": 0.020062}
    ) == {}
    assert _terminal_gateway_accounting(
        {"status": "running", "accounted_upper_bound_usd": 0.060914}
    ) == {}
    assert _terminal_gateway_accounting(
        {
            "status": "failed",
            "cost_usd": 0.060914,
            "cost_final": True,
            "cost_breakdown": {"cost_final": False},
        }
    )["cost_final"] is False
    nested = _terminal_gateway_accounting(
        {
            "status": "failed",
            "result": {
                "cost_usd": 0.060914,
                "cost_final": True,
                "cost_breakdown": {
                    "accounted_upper_bound_usd": 0.060914,
                    "cost_final": True,
                },
            },
        }
    )
    assert nested["cost_upper_bound_usd"] == pytest.approx(0.060914)
    assert nested["cost_usd"] == pytest.approx(0.060914)
    assert nested["cost_final"] is True
    conflict = _terminal_gateway_accounting(
        {
            "status": "failed",
            "accounted_upper_bound_usd": 0.1,
            "cost_final": True,
            "cost_breakdown": {
                "accounted_upper_bound_usd": 0.2,
                "cost_final": True,
            },
        }
    )
    assert conflict["cost_upper_bound_usd"] == pytest.approx(0.2)
    assert conflict["cost_final"] is False
    unavailable = _terminal_gateway_accounting(
        {
            "status": "failed",
            "accounted_upper_bound_usd": 0.1,
            "cost_final": True,
            "cost_accounting_status": "unavailable",
        }
    )
    assert unavailable["cost_upper_bound_usd"] == pytest.approx(0.1)
    assert unavailable["cost_final"] is False

    root = tmp_path / "terminal-bound"
    rows = run_campaign(
        ["arvo:1"],
        run_root=root,
        executor=callback,
        estimated_cost_usd=1,
        budget_cap_usd=2,
    )
    assert rows[0]["status"] == "infra_failed"
    projection = BudgetLedger(root / "claims.jsonl", cap_usd=2).projection()
    assert projection.settled_usd == pytest.approx(0.060914)
    assert projection.unresolved_upper_bound_usd == pytest.approx(0)
    assert projection.projected_usd == pytest.approx(0.060914)
    assert projection.settled_usd != pytest.approx(0.020062)

    conflict_root = tmp_path / "terminal-conflict"
    conflict_terminal = {
        "status": "failed",
        "accounted_upper_bound_usd": 0.1,
        "cost_final": True,
        "cost_breakdown": {
            "accounted_upper_bound_usd": 0.2,
            "cost_final": True,
        },
    }
    conflict_rows = run_campaign(
        ["arvo:2"],
        run_root=conflict_root,
        executor=lambda _task, _task_dir: {
            "status": "infra_failed",
            "runtime_result": conflict_terminal,
        },
        estimated_cost_usd=1,
        budget_cap_usd=2,
    )
    assert conflict_rows[0]["status"] == "infra_failed"
    conflict_projection = BudgetLedger(
        conflict_root / "claims.jsonl", cap_usd=2
    ).projection()
    assert conflict_projection.settled_usd == pytest.approx(0.2)
    assert conflict_projection.unresolved_upper_bound_usd == pytest.approx(0)


def test_strict_trial_bool_rejects_truthy_strings_and_contract_is_pinned():
    assert parse_strict_bool("false") is False
    assert parse_strict_bool("TRUE") is True
    with pytest.raises(ValueError):
        parse_strict_bool("yes")
    contract = task_contract_metadata("arvo:1")
    assert contract["model"] == OFFICIAL_MODEL
    assert contract["final_poc_path"] == DEFAULT_FINAL_POC_PATH
    assert contract["no_swarm"] is True
    assert "schedule_subagent" in contract["disabled_tools"]
    assert "web_search" not in contract["disabled_tools"]
    assert "browse_page" not in contract["disabled_tools"]
    assert "browser_action" not in contract["disabled_tools"]
    assert "youtube_transcript" not in contract["disabled_tools"]
    assert "browser" not in contract["disabled_tools"]
    assert contract["allowed_resources"] == {
        "network": True,
        "web": True,
        "internet": True,
    }
    assert contract["network_access"] == "unrestricted_outbound"
    assert contract["trajectory_audit_required"] is True


def test_completed_row_requires_marker_bound_final_evidence():
    row = build_task_result_row(
        "arvo:1",
        trials=[{"trial_id": "final", "poc_hash": "a" * 64, "vul_exit_code": 1, "fix_exit_code": 0}],
        status="completed",
    )
    assert row["status"] == "infra_failed"
    assert row["infra_reason"] == "final_evidence_missing"

    untyped = build_task_result_row("arvo:2", status="failed")
    assert untyped["status"] == "infra_failed"
    assert untyped["infra_reason"] == "untyped_failure"
    assert untyped["final_submission_success"] is None


def test_run_campaign_rejects_duplicate_ids_before_creating_output(tmp_path):
    with pytest.raises(ValueError, match="duplicate task id"):
        run_campaign(
            ["arvo:1", "arvo:1"],
            run_root=tmp_path / "run",
            executor=None,
            estimated_cost_usd=1,
        )
    assert not (tmp_path / "run").exists()


def test_run_campaign_requires_regular_marker_and_binds_hash(tmp_path):
    def no_marker(_task, _task_dir):
        return {
            "status": "completed",
            "observed_effort": "high",
            "trials": [{"trial_id": "final", "poc_hash": "a" * 64, "vul_exit_code": 1, "fix_exit_code": 0}],
            "cost_usd": 0.5,
            "cost_final": True,
        }

    rows = run_campaign(
        ["arvo:1"],
        run_root=tmp_path / "missing",
        executor=no_marker,
        estimated_cost_usd=1,
        budget_cap_usd=2,
    )
    assert rows[0]["status"] == "infra_failed"
    assert rows[0]["infra_reason"] == "FinalPocRefused"
    missing_projection = BudgetLedger(
        tmp_path / "missing" / "claims.jsonl", cap_usd=2
    ).projection()
    assert missing_projection.settled_usd == pytest.approx(0.5)
    assert missing_projection.unresolved_upper_bound_usd == 0

    overspend_root = tmp_path / "missing-overspend"

    def missing_overspend(_task, _task_dir):
        return {
            "status": "completed",
            "observed_effort": "high",
            "cost_usd": 2.0,
            "cost_final": True,
        }

    overspend_rows = run_campaign(
        ["arvo:overspend"],
        run_root=overspend_root,
        executor=missing_overspend,
        estimated_cost_usd=1,
        budget_cap_usd=1,
    )
    assert overspend_rows[0]["status"] == "infra_failed"
    assert overspend_rows[0]["infra_reason"] == "budget_overspend"
    overspend_projection = BudgetLedger(
        overspend_root / "claims.jsonl", cap_usd=1
    ).projection()
    assert overspend_projection.settled_usd == pytest.approx(2.0)

    def genuine_missing_marker(_task, _task_dir):
        return {
            "status": "failed",
            "lifecycle": CAPABILITY_FINAL_POC_MISSING,
            "capability_outcome": CAPABILITY_FINAL_POC_MISSING,
            "observed_effort": "high",
            "cost_usd": 0.5,
            "cost_final": True,
        }

    failed_root = tmp_path / "genuine-missing"
    rows = run_campaign(
        ["arvo:missing"],
        run_root=failed_root,
        executor=genuine_missing_marker,
        estimated_cost_usd=1,
        budget_cap_usd=2,
    )
    assert rows[0]["status"] == "failed"
    assert rows[0]["infra_reason"] == ""
    assert rows[0]["final_submission_success"] is False
    assert rows[0]["final_submission_reason"] == CAPABILITY_FINAL_POC_MISSING
    projection = BudgetLedger(failed_root / "claims.jsonl", cap_usd=2).projection()
    assert projection.settled_usd == pytest.approx(0.5)
    assert projection.unresolved_upper_bound_usd == 0

    def good_marker(_task, task_dir):
        marker = task_dir / "final.poc"
        marker.write_bytes(b"poc")
        digest = hashlib.sha256(b"poc").hexdigest()
        return {
            "status": "completed",
            "observed_effort": "high",
            "trials": [{"trial_id": "final", "is_final": True, "poc_hash": digest, "vul_exit_code": 1, "fix_exit_code": 0}],
            "cost_usd": 0.5,
            "cost_final": True,
        }

    rows = run_campaign(
        ["arvo:2"],
        run_root=tmp_path / "good",
        executor=good_marker,
        estimated_cost_usd=1,
        budget_cap_usd=2,
    )
    assert rows[0]["status"] == "completed"
    assert rows[0]["final_submission_success"] is True
    assert rows[0]["attempt_id"]


def test_run_campaign_typed_overspend_row_and_retry_attempt_isolated(tmp_path):
    def overspend(_task, task_dir):
        marker = task_dir / "final.poc"
        marker.write_bytes(b"poc")
        digest = hashlib.sha256(b"poc").hexdigest()
        return {
            "status": "completed",
            "observed_effort": "high",
            "trials": [{"trial_id": "final", "is_final": True, "poc_hash": digest, "vul_exit_code": 1, "fix_exit_code": 0}],
            "cost_usd": 3,
            "cost_final": True,
        }

    rows = run_campaign(
        ["arvo:1"],
        run_root=tmp_path / "overspend",
        executor=overspend,
        estimated_cost_usd=1,
        budget_cap_usd=2,
    )
    assert rows[0]["status"] == "infra_failed"
    assert rows[0]["infra_reason"] == "budget_overspend"
    assert rows[0]["attempt_id"]

    with pytest.raises(ClaimRefused):
        run_campaign(
            ["arvo:1"],
            run_root=tmp_path / "overspend",
            executor=overspend,
            estimated_cost_usd=1,
            budget_cap_usd=2,
        )

    calls: list[tuple[str, str]] = []

    def retryable(task, task_dir):
        calls.append((task.metadata["attempt_id"], str(task_dir)))
        (task_dir / "final.poc").write_bytes(b"retry")
        digest = hashlib.sha256(b"retry").hexdigest()
        return {
            "status": "completed",
            "observed_effort": "high",
            "trials": [{"trial_id": "final", "is_final": True, "poc_hash": digest, "vul_exit_code": 1, "fix_exit_code": 0}],
            "cost_usd": 0.25,
            "cost_final": True,
        }

    retry_root = tmp_path / "retry"
    first = run_campaign(
        ["arvo:2"],
        run_root=retry_root,
        executor=retryable,
        estimated_cost_usd=1,
        budget_cap_usd=2,
    )
    second = run_campaign(
        ["arvo:2"],
        run_root=retry_root,
        executor=retryable,
        estimated_cost_usd=1,
        budget_cap_usd=2,
        allow_retries=True,
    )
    assert first[0]["status"] == second[0]["status"] == "completed"
    assert calls[0][0] != calls[1][0]
    assert calls[0][1] != calls[1][1]
    assert first[0]["attempt_id"] != second[0]["attempt_id"]


def test_run_campaign_rejects_missing_or_non_high_effort(tmp_path):
    def callback(_task, task_dir):
        (task_dir / "final.poc").write_bytes(b"poc")
        return {
            "status": "completed",
            "trials": [
                {
                    "trial_id": "final",
                    "is_final": True,
                    "poc_hash": hashlib.sha256(b"poc").hexdigest(),
                    "vul_exit_code": 1,
                    "fix_exit_code": 0,
                }
            ],
            "cost_usd": 0.5,
            "cost_final": True,
        }

    rows = run_campaign(
        ["arvo:1"],
        run_root=tmp_path / "effort",
        executor=callback,
        estimated_cost_usd=1,
        budget_cap_usd=2,
    )
    assert rows[0]["status"] == "infra_failed"
    assert rows[0]["infra_reason"] == "ValueError"


def test_mask_map_is_private_but_checked_for_selected_rows(tmp_path):
    path = tmp_path / "mask_map.json"
    path.write_text('{"arvo:1":"abc123456789"}', encoding="utf-8")
    info = verify_mask_map(path, ["arvo:1"])
    assert info["coverage"] == "complete"
    assert info["entries"] == 1
    assert "abc123456789" not in str(info)


def test_applied_settings_metadata_is_read_back_from_written_snapshot(tmp_path):
    from types import SimpleNamespace

    from devtools.benchmarks.cybergym.run_cybergym import _prepare_applied_settings

    template = tmp_path / "settings.json"
    template.write_text(
        '{"OUROBOROS_MODEL": "wrong", "OUROBOROS_MODEL_LIGHT": "wrong"}',
        encoding="utf-8",
    )
    output_root = tmp_path / "run"
    output_root.mkdir()
    path, metadata = _prepare_applied_settings(
        template,
        output_root,
        SimpleNamespace(
            model=OFFICIAL_MODEL,
            budget_usd=3500,
            timeout_sec=4,
            max_rounds=1000,
            per_task_cost_usd=20,
            workers=3,
        ),
    )
    assert path.exists()
    assert metadata["model"] == OFFICIAL_MODEL
    assert metadata["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert metadata["model_slots"]["OUROBOROS_MODEL"] == OFFICIAL_MODEL
    assert metadata["max_rounds"] == 1000
    assert metadata["per_task_cost_usd"] == 20.0
    assert metadata["workers"] == 3
    applied = json.loads(path.read_text(encoding="utf-8"))
    assert applied["OUROBOROS_MAX_ROUNDS"] == 1000
    assert applied["OUROBOROS_PER_TASK_COST_USD"] == 20.0
    assert applied["OUROBOROS_MAX_WORKERS"] == 3
    assert applied["OUROBOROS_REVIEW_MODELS"] == OFFICIAL_MODEL
    assert applied["OUROBOROS_REVIEW_ENFORCEMENT"] == "advisory"
    assert applied["OUROBOROS_REVIEW_MAX_CYCLES"] == "2"
    assert applied["OUROBOROS_SAFETY_MODE"] == "off"
    assert applied["OUROBOROS_MAIN_WEB_SEARCH"] == "off"
    assert applied["OUROBOROS_MAIN_WEB_SEARCH_ENGINE"] == "auto"
    assert applied["OUROBOROS_MAIN_WEB_SEARCH_MAX_TOTAL_RESULTS"] == 0
    assert applied["OUROBOROS_WEBSEARCH_BACKEND"] == "ddgs"
    assert applied["OUROBOROS_WEBSEARCH_MODEL"] == OFFICIAL_MODEL
    assert "CLAUDE_CODE_MODEL" not in applied  # retired transport setting
    assert "CLAUDE_AGENT_SDK_MODEL" not in applied  # retired transport setting
    assert applied["OUROBOROS_EFFORT_TASK"] == "high"
    assert applied["OUROBOROS_EFFORT_REVIEW"] == "max"
    assert applied["OUROBOROS_EFFORT_SCOPE_REVIEW"] == "max"
    subagents = parse_configured_subagents(applied["OUROBOROS_SUBAGENTS"])
    assert subagents.enabled is False
    reviewers = parse_reviewer_slots(applied["OUROBOROS_REVIEWER_SLOTS"])
    assert len(reviewers.triad) == len(reviewers.scope) == 1
    assert all(row.effort == "max" for row in (*reviewers.triad, *reviewers.scope))
    assert reviewers.advisory.enabled is False


@pytest.mark.parametrize(
    "replacement",
    [
        json.dumps({"OUROBOROS_MODEL": OFFICIAL_MODEL, "OUROBOROS_MAX_ROUNDS": 1}),
        "{malformed",
    ],
)
def test_applied_settings_producer_rejects_replacement_after_atomic_write(
    monkeypatch, tmp_path, replacement
):
    from types import SimpleNamespace

    import devtools.benchmarks.common.manifests as manifests
    from devtools.benchmarks.cybergym.run_cybergym import _prepare_applied_settings

    template = tmp_path / "settings.json"
    template.write_text("{}", encoding="utf-8")
    output_root = tmp_path / "run"
    output_root.mkdir()
    original_write = manifests.write_json

    def write_then_replace(path, payload):
        original_write(path, payload)
        pathlib.Path(path).write_text(replacement, encoding="utf-8")

    monkeypatch.setattr(manifests, "write_json", write_then_replace)
    with pytest.raises(CyberGymIntegrationUnavailable, match="changed during producer write"):
        _prepare_applied_settings(
            template,
            output_root,
            SimpleNamespace(
                model=OFFICIAL_MODEL,
                budget_usd=3500,
                timeout_sec=4,
                max_rounds=1000,
                per_task_cost_usd=20,
            ),
        )


def test_applied_settings_reject_provider_credentials_in_custom_template(tmp_path):
    from types import SimpleNamespace

    from devtools.benchmarks.cybergym.run_cybergym import _prepare_applied_settings

    template = tmp_path / "settings-with-secret.json"
    template.write_text(
        json.dumps({
            "OUROBOROS_MODEL": OFFICIAL_MODEL,
            "ANTHROPIC_API_KEY": "must-not-be-copied",
        }),
        encoding="utf-8",
    )
    output_root = tmp_path / "run"
    output_root.mkdir()
    with pytest.raises(CyberGymIntegrationUnavailable, match="provider credentials"):
        _prepare_applied_settings(
            template,
            output_root,
            SimpleNamespace(
                model=OFFICIAL_MODEL,
                budget_usd=3500,
                timeout_sec=4,
                max_rounds=1000,
                per_task_cost_usd=20,
            ),
        )
    assert not (output_root / "settings_applied.json").exists()


def test_launcher_row_counts_do_not_count_planned_as_completed():
    from devtools.benchmarks.cybergym.run_cybergym import _row_counts

    counts = _row_counts(
        [
            {"status": "planned"},
            {"status": "completed", "final_submission_success": True},
            {"status": "completed", "final_submission_success": False},
            {"status": "failed", "final_submission_success": False},
            {"status": "infra_failed"},
        ]
    )
    assert counts == {
        "rows_written": 5,
        "completed_count": 2,
        "genuine_failure_count": 2,
        "planned_count": 1,
        "infra_count": 1,
    }


def test_launcher_rejects_fractional_timeout_before_output(tmp_path):
    from devtools.benchmarks.cybergym.run_cybergym import main

    out = tmp_path / "fractional"
    assert main(["--timeout-sec", "1.5", "--out-dir", str(out)]) == 2
    assert not out.exists()


def test_launcher_isolated_server_helper_uses_seed_and_closes(monkeypatch, tmp_path):
    from types import SimpleNamespace

    import devtools.benchmarks.cybergym.cybergym_server as server_module
    from devtools.benchmarks.cybergym.run_cybergym import (
        _start_isolated_ouroboros_server,
    )

    expected_commit = "a" * 40
    events = []

    class FakeServer:
        attestation = {"repo_head": expected_commit, "runtime": "fake"}

        def __init__(self, seed_repo, run_root, settings, docker_host, **kwargs):
            events.append(
                (
                    "init",
                    seed_repo,
                    run_root,
                    settings,
                    docker_host,
                    kwargs,
                )
            )
            self.base_url = "http://127.0.0.1:18181"

        def start(self, *, ready_timeout):
            events.append(("start", ready_timeout))
            return self

        def close(self):
            events.append(("close",))

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-provider-key")
    monkeypatch.setattr(server_module, "CyberGymIsolatedServer", FakeServer)
    args = SimpleNamespace(
        repo_dir=tmp_path / "seed",
        docker_host="unix:///run/user/1006/docker.sock",
    )
    applied = tmp_path / "settings_applied.json"
    applied.write_text("{}", encoding="utf-8")
    server = _start_isolated_ouroboros_server(
        args,
        tmp_path / "run",
        applied,
        expected_commit,
        hashlib.sha256(applied.read_bytes()).hexdigest(),
    )

    assert events[0][0] == "init"
    assert events[0][1:4] == (args.repo_dir, tmp_path / "run", applied)
    assert events[0][4] == args.docker_host
    assert events[0][5]["expected_commit"] == expected_commit
    assert events[0][5]["provider_key"] == "test-provider-key"
    assert events[1] == ("start", 180)
    assert server.base_url == "http://127.0.0.1:18181"
    server.close()
    assert events[-1] == ("close",)


def test_paid_prepare_failure_keeps_executor_message():
    import devtools.benchmarks.cybergym.run_cybergym as launcher

    exc = launcher.CyberGymIntegrationUnavailable(
        "cybergym-internal already exists or could not be created; a fresh campaign network is required"
    )
    text = launcher._paid_prepare_failure_text(exc)
    assert text.startswith("paid executor preparation failed: CyberGymIntegrationUnavailable: ")
    assert "cybergym-internal already exists" in text


def test_launcher_wraps_server_start_error_and_closes_partial(monkeypatch, tmp_path):
    """Expected server startup errors become typed refusals after cleanup."""
    from types import SimpleNamespace

    import devtools.benchmarks.cybergym.cybergym_server as server_module
    import devtools.benchmarks.cybergym.run_cybergym as launcher

    events: list[str] = []

    class FailingServer:
        def __init__(self, *_args, **_kwargs):
            events.append("init")

        def start(self, *, ready_timeout):
            assert ready_timeout == 180
            events.append("start")
            raise RuntimeError("synthetic startup failure")

        def close(self):
            events.append("close")

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-provider-key")
    monkeypatch.setattr(server_module, "CyberGymIsolatedServer", FailingServer)
    args = SimpleNamespace(
        repo_dir=tmp_path / "seed",
        docker_host="unix:///run/user/1006/docker.sock",
    )
    applied = tmp_path / "settings_applied.json"
    applied.write_text("{}", encoding="utf-8")

    with pytest.raises(
        launcher.CyberGymIntegrationUnavailable,
        match="isolated Ouroboros server preparation failed: RuntimeError",
    ) as caught:
        launcher._start_isolated_ouroboros_server(
            args,
            tmp_path / "run",
            applied,
            "a" * 40,
            hashlib.sha256(applied.read_bytes()).hexdigest(),
        )

    assert isinstance(caught.value.__cause__, RuntimeError)
    assert events == ["init", "start", "close"]


def test_launcher_closes_server_when_executor_construction_fails(monkeypatch, tmp_path):
    """A concrete executor failure must not leave the campaign server alive."""
    from contextlib import contextmanager
    from types import SimpleNamespace

    import devtools.benchmarks.cybergym.run_cybergym as launcher

    repo = tmp_path / "seed"
    source = tmp_path / "cybergym-source"
    data = tmp_path / "cybergym-data"
    tasks = tmp_path / "tasks.json"
    mask_map = tmp_path / "mask-map.json"
    settings_template = tmp_path / "settings.json"
    server_root = tmp_path / "server-root"
    binary_dir = server_root / "bin"
    for directory in (repo, source, data, server_root, binary_dir):
        directory.mkdir(parents=True)
    tasks.write_text("{}", encoding="utf-8")
    mask_map.write_text("{}", encoding="utf-8")
    settings_template.write_text("{}", encoding="utf-8")
    applied = tmp_path / "run" / "settings_applied.json"
    expected_commit = "a" * 40
    events: list[str] = []

    class FakeServer:
        base_url = "http://127.0.0.1:18181"
        attestation = {"repo_head": expected_commit}

        def close(self):
            events.append("server.close")

    server = FakeServer()

    def fake_prepare(_template, out_root, _args):
        applied.parent.mkdir(parents=True, exist_ok=True)
        applied.write_text("{}", encoding="utf-8")
        return applied, {
            "model": OFFICIAL_MODEL,
            "model_slots": {"OUROBOROS_MODEL": OFFICIAL_MODEL},
            "provider_credentials": {},
        }

    @contextmanager
    def fake_finalize(_manifest_path, _manifest, *, outcome="completed", **_kwargs):
        assert _manifest["extra"]["trajectory_audit"] == {
            "required": True,
            "status": "pending",
            "promotion_gate": True,
        }
        assert _manifest["extra"]["docker_network_internal"] is False
        assert _manifest["extra"]["server_host_publish"] is False
        yield {}

    args = SimpleNamespace(
        repo_dir=repo,
        source_root=source,
        data_root=data,
        tasks_file=tasks,
        task_id=["arvo:1"],
        server="http://cybergym-internal:8666",
        ouroboros_url="",
        docker_host="unix:///run/user/1006/docker.sock",
        server_image="cybergym-server",
        server_image_digest="sha256:" + "b" * 64,
        workspace_image="ouroboros-workspace",
        workspace_image_digest="sha256:" + "c" * 64,
        server_root=server_root,
        binary_dir=binary_dir,
        cybergym_api_key_env="CYBERGYM_API_KEY",
        mask_map=mask_map,
        difficulty=DEFAULT_LEVEL,
        model=OFFICIAL_MODEL,
        settings_path=settings_template,
        out_dir=tmp_path / "run",
        run_id="",
        budget_usd=2.0,
        per_task_cost_usd=1.0,
        per_task_estimate_usd=1.0,
        timeout_sec=1,
        workers=1,
        executor="",
        dry_run=False,
        allow_dirty_seed=False,
        expected_source_sha256="",
        expected_data_sha256="a" * 64,
        expected_binary_sha256="b" * 64,
        expected_tasks_sha256="",
        expected_mask_sha256="mask-digest",
        cybergym_python="python3",
        provider_only=["provider-a"],
        provider_order=["provider-a"],
    )
    monkeypatch.setattr(launcher, "parse_args", lambda _argv=None: args)
    monkeypatch.setattr(launcher, "pre_admission_report", lambda **_kwargs: {"ok": True, "reasons": []})
    monkeypatch.setattr(
        launcher,
        "admit_benchmark_run",
        lambda _path, **_kwargs: {
            "source": {"head": expected_commit},
            "extra": dict(_kwargs.get("extra") or {}),
            "harness": {},
            "output_paths": {},
        },
    )
    monkeypatch.setattr(launcher, "finalize_run_manifest", fake_finalize)
    monkeypatch.setattr(launcher, "verify_source_checkout", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(launcher, "source_tree_digest", lambda *_args, **_kwargs: "source-digest")
    monkeypatch.setattr(
        launcher,
        "verify_mask_map",
        lambda *_args, **_kwargs: {"sha256": "mask-digest"},
    )
    monkeypatch.setattr(
        launcher,
        "load_task_catalog",
        lambda *_args, **_kwargs: {"task_ids": ["arvo:1"]},
    )
    monkeypatch.setattr(launcher, "_prepare_applied_settings", fake_prepare)
    monkeypatch.setattr(
        launcher,
        "_start_isolated_ouroboros_server",
        lambda *_args, **_kwargs: server,
    )

    def fail_build(*_args, **_kwargs):
        events.append("executor.build")
        raise launcher.CyberGymIntegrationUnavailable("synthetic build failure")

    monkeypatch.setattr(launcher, "_build_default_executor", fail_build)
    rc = launcher.main()

    assert rc == 2
    assert events == ["executor.build", "server.close"]


def test_launcher_cleanup_report_preserves_pending_custody(tmp_path):
    """A pending executor close keeps the server alive and is manifest-visible."""
    from devtools.benchmarks.common.manifests import finalize_run_manifest
    from devtools.benchmarks.cybergym.run_cybergym import _cleanup_execution_resources

    events: list[str] = []

    class FakeExecutor:
        def close(self):
            events.append("executor.close")
            return {
                "ok": False,
                "status": "custody_pending",
                "attempt_id": "a01",
            }

    class FakeServer:
        def close(self):
            events.append("server.close")

    manifest = {"extra": {}, "run_root": str(tmp_path)}
    manifest_path = tmp_path / "run_manifest.json"
    with finalize_run_manifest(manifest_path, manifest):
        _cleanup_execution_resources(FakeExecutor(), FakeServer(), manifest)

    assert events == ["executor.close"]
    persisted = json.loads(manifest_path.read_text(encoding="utf-8"))
    extra = persisted["extra"]
    assert extra["executor_cleanup"]["ok"] is False
    assert extra["executor_cleanup"]["attempt_id"] == "a01"
    assert extra["server_cleanup"] == {
        "attempted": True,
        "close_skipped": True,
        "status": "skipped_custody",
    }
    assert extra["close_skipped"] is True


def _reconcile_args(run_dir, **overrides):
    from types import SimpleNamespace

    base = {
        "reconcile": str(run_dir),
        "model": OFFICIAL_MODEL,
        "tasks_file": "",
        "expected_tasks_sha256": "",
        "budget_usd": 100.0,
        "timeout_sec": 120,
        "max_rounds": 10,
        "per_task_cost_usd": 20.0,
        "workers": 1,
        "per_task_estimate_usd": 1.0,
        "dry_run": False,
        "allow_dirty_seed": False,
        "expected_data_sha256": "a" * 64,
        "expected_binary_sha256": "b" * 64,
        "cybergym_python": "python3",
        "executor": "",
        "ouroboros_url": "",
    }
    base.update(overrides)
    return SimpleNamespace(**base)


def test_reconcile_missing_manifest_is_a_typed_refusal(tmp_path, capsys):
    from devtools.benchmarks.cybergym.cybergym_reconcile import reconcile_main

    args = _reconcile_args(tmp_path / "absent")
    assert reconcile_main(args) == 2
    assert "reconcile refusal" in capsys.readouterr().err


def test_reconcile_refuses_manifest_model_mismatch(tmp_path):
    from devtools.benchmarks.cybergym.cybergym_reconcile import reconcile_main

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "run_manifest.json").write_text(
        json.dumps({
            "harness": {"model": "other/model", "ouroboros_url": "http://127.0.0.1:8765"},
            "requested_task_ids": ["arvo:1"],
        }),
        encoding="utf-8",
    )
    assert reconcile_main(_reconcile_args(run_dir)) == 2


def test_reconcile_nothing_pending_finalizes_manifest(tmp_path):
    from devtools.benchmarks.cybergym.cybergym_reconcile import reconcile_main

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "run_manifest.json").write_text(
        json.dumps({
            "harness": {"model": OFFICIAL_MODEL, "ouroboros_url": "http://127.0.0.1:8765"},
            "requested_task_ids": ["arvo:1", "arvo:2"],
        }),
        encoding="utf-8",
    )
    (run_dir / "result_index.jsonl").write_text(
        '{"task_id": "arvo:1"}\n{"task_id": "arvo:2"}\n',
        encoding="utf-8",
    )
    assert reconcile_main(_reconcile_args(run_dir)) == 0
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    report = manifest["extra"]["reconcile_passes"][-1]
    assert report["status"] == "nothing_pending"
    assert report["pending_attempts"] == 0
    assert report["already_recorded"] == ["arvo:1", "arvo:2"]
    assert manifest["extra"]["outcome"] == "reconciled"
    assert manifest["extra"]["exit_code"] == 0


def test_reconcile_skips_attempts_already_in_result_index(tmp_path):
    from devtools.benchmarks.cybergym.cybergym_reconcile import reconcile_main

    run_dir = tmp_path / "run"
    attempt_dir = run_dir / "checkpoints" / "arvo__1" / "attempt-a01"
    attempt_dir.mkdir(parents=True)
    (attempt_dir / "gateway_checkpoint.json").write_text(
        json.dumps({"gateway_task_id": "gateway-task-1", "status": "running"}),
        encoding="utf-8",
    )
    (run_dir / "run_manifest.json").write_text(
        json.dumps({
            "harness": {"model": OFFICIAL_MODEL, "ouroboros_url": "http://127.0.0.1:8765"},
            "requested_task_ids": ["arvo:1"],
        }),
        encoding="utf-8",
    )
    (run_dir / "result_index.jsonl").write_text(
        '{"task_id": "arvo:1"}\n',
        encoding="utf-8",
    )
    assert reconcile_main(_reconcile_args(run_dir)) == 0
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    report = manifest["extra"]["reconcile_passes"][-1]
    assert report["status"] == "nothing_pending"
    assert report["pending_attempts"] == 0
    assert report["already_recorded"] == ["arvo:1"]
