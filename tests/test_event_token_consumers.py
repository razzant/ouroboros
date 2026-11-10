"""Event readers distinguish complete totals, partial lower bounds and legacy ambiguity."""

from __future__ import annotations

import copy
import json
from types import SimpleNamespace

import pytest

from ouroboros.event_token_totals import TOKEN_COUNTERS, event_token_totals


def _row(**counts):
    return {"type": "llm_usage", "record_contract_version": 2, **counts}


def test_each_counter_has_independent_unknown_zero_and_partial_evidence():
    evidence = event_token_totals([
        _row(prompt_tokens=20, completion_tokens=None, cached_tokens=0, cache_write_tokens=3),
        _row(prompt_tokens=None, completion_tokens=4, cached_tokens=0),
    ])
    assert evidence["prompt_tokens"] == {
        "total": None, "known_total": 20, "known_count": 1, "unknown_count": 1, "legacy_zero_count": 0,
    }
    assert evidence["completion_tokens"]["known_total"] == 4
    assert evidence["completion_tokens"]["total"] is None
    assert evidence["cached_tokens"]["total"] == 0
    assert evidence["cache_write_tokens"]["known_total"] == 3
    assert evidence["cache_write_tokens"]["unknown_count"] == 1
    assert all(item["total"] is None for item in event_token_totals([]).values())


@pytest.mark.parametrize("version", [None, 1])
def test_historical_zero_is_ambiguous_and_input_rows_are_unchanged(version):
    row = {"prompt_tokens": 0, "completion_tokens": 7, "cached_tokens": 0}
    if version is not None:
        row["record_contract_version"] = version
    original = copy.deepcopy(row)
    evidence = event_token_totals([row])
    assert evidence["prompt_tokens"]["total"] is None
    assert evidence["prompt_tokens"]["legacy_zero_count"] == 1
    assert evidence["completion_tokens"]["total"] == 7
    assert evidence["cached_tokens"]["legacy_zero_count"] == 1
    assert evidence["cache_write_tokens"]["unknown_count"] == 1
    assert row == original


@pytest.mark.parametrize("bad", [None, 1.5, float("nan")])
def test_invalid_values_do_not_become_measured_zero(bad):
    evidence = event_token_totals([_row(**dict.fromkeys(TOKEN_COUNTERS, bad))])
    assert all(item["total"] is None and item["unknown_count"] == 1 for item in evidence.values())


def test_nested_usage_preserves_the_envelope_version():
    evidence = event_token_totals([_row(usage=dict.fromkeys(TOKEN_COUNTERS, 0))])
    assert all(item["total"] == 0 and item["known_count"] == 1 for item in evidence.values())
    evidence = event_token_totals([_row(prompt_tokens=None, usage={"prompt_tokens": 123})])
    assert evidence["prompt_tokens"]["total"] is None
    assert evidence["prompt_tokens"]["known_total"] == 0


def test_editbench_retains_lower_bounds_deduplicates_and_keeps_money_unchanged(tmp_path):
    from devtools.benchmarks.editbench.run_editbench import _mine_metrics, _summarize

    logs = tmp_path / "logs"
    logs.mkdir()
    rows = [
        _row(task_id="t", ledger_attempt_ids=["a"], prompt_tokens=10, completion_tokens=3,
             cached_tokens=0, cache_write_tokens=2, cost=0.25),
        _row(task_id="t", ledger_attempt_ids=["b"], prompt_tokens=None, completion_tokens=0,
             cached_tokens=0, cache_write_tokens=None, cost=None),
    ]
    (logs / "events.jsonl").write_text("".join(json.dumps(row) + "\n" for row in [*rows, rows[0]]))
    metrics = _mine_metrics(tmp_path, "t")
    usage = metrics["usage"]
    assert usage["llm_calls"] == 2 and usage["cost_usd"] == 0.25
    assert usage["prompt_tokens"] is None and usage["completion_tokens"] == 3
    assert usage["cached_tokens"] == 0 and usage["cache_write_tokens"] is None
    assert usage["token_evidence"]["prompt_tokens"]["known_total"] == 10
    assert usage["token_evidence"]["prompt_tokens"]["unknown_count"] == 1
    summary = _summarize([{"config": "full", "pass": True, "files_matched": 1, "wall_sec": 3, **metrics}])
    assert summary["t1_rename/full"]["avg_prompt_tokens"] is None
    assert summary["t1_rename/full"]["avg_completion_tokens"] == 3
    assert summary["t1_rename/full"]["token_evidence"]["prompt_tokens"]["known_total"] == 10
    assert summary["t1_rename/full"]["token_evidence"]["prompt_tokens"]["unknown_count"] == 1


@pytest.mark.parametrize("prompt, expected", [(None, False), (0, False), (8, True)])
def test_live_scenario_requires_positive_evidence_without_inventing_totals(prompt, expected):
    from devtools.e2e_live.scenarios import LaneContext

    rows = [_row(task_id="t", prompt_tokens=prompt), _row(task_id="t", prompt_tokens=None)]
    facts = {}
    ctx = SimpleNamespace(oracle=object(), wait_events=lambda *_: rows,
                          check=lambda name, ok, **kw: facts.update(name=name, ok=ok, **kw))
    LaneContext.check_paid_tokens(ctx, ["t"])
    assert facts["ok"] is expected
    assert facts["prompt_tokens"] is None
    assert facts["completion_tokens"] is None
    assert facts["token_evidence"]["prompt_tokens"]["known_total"] == (prompt or 0)


def test_cowork_audit_preserves_legacy_ambiguity_without_changing_money(tmp_path):
    from devtools.benchmarks.cowork_bench.audit_cowork_bench import audit_task

    logs = tmp_path / "ouroboros"
    logs.mkdir()
    rows = [{"type": "llm_usage", "prompt_tokens": 0, "completion_tokens": 4,
             "cached_tokens": 0, "cost": 0.25}]
    (logs / "events.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    (logs / "tools.jsonl").write_text("")
    (tmp_path / "ouroboros_summary.json").write_text("{}")
    audit = audit_task(tmp_path, {"status": "passed"})
    assert audit["activity"]["prompt_tokens"] is None
    assert audit["activity"]["completion_tokens"] == 4
    assert audit["activity"]["cache_write_tokens"] is None
    assert audit["activity"]["token_evidence"]["prompt_tokens"]["legacy_zero_count"] == 1
    assert audit["cost"]["total_usd"] == 0.25 and audit["cost"]["complete"] is True
