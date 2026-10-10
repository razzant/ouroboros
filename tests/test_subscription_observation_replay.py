"""One physical session remains one charge across late model observations."""

import json
from types import SimpleNamespace

import pytest

from ouroboros import delegate_custody as custody, usage_accounting as usage, usage_store
from tests import fixtures_usage_store as store_fixtures
from tests._usage_store_testing import ledger_rows, write_journal


def _record(root, model, **overrides):
    args = dict(route="selected-route", task_id="task", root_task_id="root", parent_task_id="parent",
                category="review", source="test", review_skill="skill", review_wave_id="wave",
                review_slot_id="slot", spend_usd=2.5)
    args.update(overrides)
    return usage.record_subscription_session("same-run", drive_root=root, model=model, **args)


@pytest.mark.parametrize("before_model,after_model", [("actual", ""), ("", "actual"), ("first", "second")])
def test_later_model_observation_replays_the_original_row_without_repricing(tmp_path, before_model, after_model):
    first = _record(tmp_path, before_model)
    before = ledger_rows(tmp_path)
    assert _record(tmp_path, after_model, spend_usd=99) == first
    assert ledger_rows(tmp_path) == before
    row = next(row for row in before if row.get("attempt_id") == first)
    assert row["model"] == before_model and row["cost_usd"] == 2.5


@pytest.mark.parametrize("field", ["route", "task_id", "root_task_id", "parent_task_id", "category", "source",
                                  "review_skill", "review_wave_id", "review_slot_id"])
def test_late_model_never_waives_remaining_caller_identity(tmp_path, field):
    _record(tmp_path, "first")
    before = ledger_rows(tmp_path)
    with pytest.raises(usage.UsageAccountingError, match="conflicting settled-row identity"):
        _record(tmp_path, "second", **{field: "different-owner-or-route"})
    assert ledger_rows(tmp_path) == before


@pytest.mark.parametrize("field,value", [("kind", "external_unmetered"), ("provider", "other"),
                                        ("subscription_route", "other"), ("session_id_sha256", "0" * 64)])
def test_derived_identity_fields_remain_checked(tmp_path, field, value):
    # A stored row whose derived identity differs while the lookup id is the
    # same: journaled that way (recorded in a scratch root, one field changed)
    # and brought in by this root's one-time import.
    scratch, root = tmp_path / "scratch", tmp_path / "data"
    attempt_id = _record(scratch, "first")
    [row] = ledger_rows(scratch)
    row[field] = value
    write_journal(root, [{key: item for key, item in row.items() if key not in {"revision", "seq"}}])
    before = ledger_rows(root)
    assert [stored["attempt_id"] for stored in before] == [attempt_id]
    with pytest.raises(usage.UsageAccountingError, match="conflicting settled-row identity"):
        _record(root, "second")
    assert ledger_rows(root) == before


def test_external_unmetered_keeps_its_existing_model_identity(tmp_path):
    usage.record_unmetered_external_dispatch("same-external", drive_root=tmp_path, model="first", task_id="task")
    before = ledger_rows(tmp_path)
    with pytest.raises(usage.UsageAccountingError, match="conflicting settled-row identity"):
        usage.record_unmetered_external_dispatch("same-external", drive_root=tmp_path, model="second", task_id="task")
    assert ledger_rows(tmp_path) == before


data_root = store_fixtures.data_root


def test_model_observation_replay_after_export_and_import_keeps_the_recorded_charge(data_root):
    store_fixtures._seed_mixed_ledger(data_root)
    usage_store.export_journal(data_root)  # the downgrade step; the next access imports it again
    before = ledger_rows(data_root)
    usage.record_subscription_session(
        "sess-1", drive_root=data_root, route="claudexor:claude", model="new-observation",
        task_id="t5", root_task_id="root", spend_usd=99,
    )
    assert ledger_rows(data_root) == before


@pytest.mark.parametrize("observed_model", ["actual-model", None])
def test_lost_custody_checkpoint_recovers_without_rewriting_the_old_charge(tmp_path, observed_model):
    drive = tmp_path / "data"
    entry = custody.RunCustody(run_id="run-replay", task_id="t", root_task_id="t",
                               route_id="selected-route", model="requested-model")
    assert custody.record_started(drive, entry)
    usage.record_subscription_session("run-replay", drive_root=drive, route="selected-route",
        model="requested-model", task_id="t", root_task_id="t", spend_usd=2.5)
    resumed = custody.replay(drive)["run-replay"]
    assert not resumed.ledger_recorded and not resumed.settled
    final = tmp_path / "engine" / "final"
    final.mkdir(parents=True)
    (final / "telemetry.yaml").write_text(json.dumps({
        "run_id": "run-replay", "final_attempt_id": "a02", "attempts": [
            {"attempt_id": "a01", "observed_model": "requested-model"},
            {"attempt_id": "a02", "observed_model": observed_model,
             "harness_id": "selected-route", "profile_id": "profile"},
        ]}), encoding="utf-8")
    detail = {"summary": {"runDir": str(final.parent), "state": "succeeded",
                           "model": "requested-model", "spendUsd": 2.5}}
    before = ledger_rows(drive)
    result = custody.settle_run(drive, SimpleNamespace(), resumed, detail)
    assert result["settled"] and result["ledger_recorded"]
    assert ledger_rows(drive) == before
    rows = [json.loads(line) for line in (drive / "logs" / "events.jsonl").read_text().splitlines()]
    event = next(row for row in rows if row.get("type") == custody.SETTLED)
    assert event["model"] == (observed_model or "")
    assert event["observed_attempt"]["attempt_id"] == "a02"
    assert event["cost_usd"] == 2.5
