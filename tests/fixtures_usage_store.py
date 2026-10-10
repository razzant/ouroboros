"""Shared money fixtures over the usage store (``ouroboros/usage_store.py``).

The successors of the retired compaction fixtures: a data root whose
pre-ledger import is already complete, the canonical request/settle helpers,
a realistic mixed history, and ``fold_into_archive`` — the shape an install
upgraded from a compacted journal carries: attempts folded before the store
existed live only in a retained archive segment, and the store holds their
aggregate (imported with its weight).
"""

from __future__ import annotations

import json
from decimal import Decimal

import pytest

from ouroboros import usage_accounting as ua
from ouroboros import usage_store
from ouroboros.usage_journal import IMPORT_REL
from ouroboros.usage_ledger import LEDGER_REL
from tests._usage_store_testing import ledger_rows, write_compacted_journal

ARCHIVE_SEGMENT_REL = "archive/usage_ledger/segment_fixture.jsonl"


@pytest.fixture
def data_root(tmp_path, monkeypatch):
    root = tmp_path / "data"
    monkeypatch.setenv("OUROBOROS_DATA_DIR", str(root))
    monkeypatch.setenv("OUROBOROS_SETTINGS_PATH", str(root / "settings.json"))
    monkeypatch.setenv("TOTAL_BUDGET", "100")
    (root / "state").mkdir(parents=True)
    (root / IMPORT_REL).write_text(json.dumps({"completed": True}), encoding="utf-8")
    yield root
    usage_store.forget(root)


def _request(data_root, **overrides):
    values = {"model": "openai/gpt-5.2", "provider": "openai", "reservation_usd": 1.0,
              "drive_root": data_root, "task_id": "child", "root_task_id": "root", "source": "test"}
    values.update(overrides)
    return ua.AttemptRequest(**values)


def _ledger_rows(data_root):
    """Every attempt's current row (one per attempt), in write order."""
    return ledger_rows(data_root)


def _settle(data_root, *, cost=None, usage=None, cost_final=False, **request_overrides):
    reservation = ua.reserve_attempt(_request(data_root, **request_overrides))
    ua.mark_dispatched(reservation)
    ua.settle_attempt(reservation, usage or {"prompt_tokens": 10, "completion_tokens": 5},
                      cost_usd=cost, cost_final=cost_final)
    return reservation


def _seed_mixed_ledger(data_root):
    """A realistic history: settled (weird floats, unknown costs), unresolved,
    released, in-flight, sessions, external, review-attributed."""
    _settle(data_root, cost=0.123456789012345, cost_final=True)
    _settle(data_root, cost=1.1, task_id="t2", root_task_id="root2", root_limit_usd=50.0)
    _settle(data_root, cost=2.2, task_id="t2", root_task_id="root2", root_limit_usd=40.0)
    _settle(data_root, cost=None, usage={}, model="openai/gpt-5.2-mini")
    reservation = ua.reserve_attempt(_request(data_root, task_id="t3"))
    ua.mark_dispatched(reservation)
    ua.mark_unresolved(reservation, "provider went dark")
    reservation = ua.reserve_attempt(_request(data_root, task_id="t4"))
    ua.release_attempt(reservation, "not_dispatched")
    ua.record_subscription_session("sess-1", drive_root=data_root, route="claudexor:claude", model="fable",
                                   task_id="t5", root_task_id="root", spend_usd=0.5, reset_at="2026-09-02T00:00:00Z")
    ua.record_unmetered_external_dispatch("ext-1", drive_root=data_root, model="ext-model", task_id="t6",
                                          prompt_tokens=7, completion_tokens=3)
    with ua.usage_scope(ua.UsageScope(
        drive_root=data_root, task_id="rv", root_task_id="root",
        review_skill="skill-x", review_wave_id="w1", review_slot_id="s1",
    )):
        _settle(data_root, cost=3.5, cost_final=True)
    # Open attempts: one reserved, one dispatched.
    reserved = ua.reserve_attempt(_request(data_root, task_id="open-r"))
    dispatched = ua.reserve_attempt(_request(data_root, task_id="open-d"))
    ua.mark_dispatched(dispatched)
    return reserved, dispatched


_GROUP_FIELDS = ("state", "model", "provider", "category", "source", "task_id", "root_task_id",
                 "parent_task_id", "billing_group_id")
_TOKEN_FIELDS = ("prompt_tokens", "completion_tokens", "cached_tokens", "cache_write_tokens")


def _group_key(row):
    """The retired compactor's attribution tuple (one aggregate per tuple)."""
    pricing_known = row.get("pricing_known")
    return (*(str(row.get(field) or "") for field in _GROUP_FIELDS),
            "billing_group_limit_usd" in row, row.get("billing_group_limit_usd"),
            row.get("billing_group_limit_source"), row.get("billing_group_limit_revision"),
            str(row.get("prompt_cache_ttl") or ""), row.get("cost_usd") is not None,
            bool(row.get("cost_final")), pricing_known if isinstance(pricing_known, bool) else None,
            row.get("reservation_upper_bound_usd") is not None)


def _aggregate(index, key, rows):
    """One aggregate as the retired compactor wrote it (without carriage)."""
    from ouroboros._usage_rows import _merge_processing_summary, _processing_summary

    (state, model, provider, category, source, task_id, root_task_id, parent_task_id, group_id,
     has_group_limit, group_limit, group_source, group_revision, ttl, cost_known, cost_final,
     pricing_known, bound_known) = key
    weight = sum(max(1, int(row.get("folded_attempt_count") or 1))
                 if row.get("kind") == "usage_baseline_group" else 1 for row in rows)
    row = {"attempt_id": f"fold-fixture-g{index:04d}", "state": state, "model": model, "provider": provider,
           "category": category, "source": source, "task_id": task_id, "root_task_id": root_task_id,
           "parent_task_id": parent_task_id, "review_skill": "", "review_wave_id": "", "review_slot_id": "",
           "folded_attempt_count": weight, "cost_final": cost_final,
           "ts": max(str(item.get("ts") or "") for item in rows) or "2026-01-01T00:00:00+00:00"}
    if ttl:
        row["prompt_cache_ttl"] = ttl
    if group_id:
        row["billing_group_id"] = group_id
    if has_group_limit:
        row.update(billing_group_limit_usd=group_limit, billing_group_limit_source=group_source,
                   billing_group_limit_revision=group_revision)
    if pricing_known is not None:
        row["pricing_known"] = pricing_known

    def total(field):
        return format(sum((Decimal(str(item[field])) for item in rows if item.get(field) is not None),
                          Decimal(0)), "f")

    if cost_known:
        row["cost_usd"] = total("cost_usd")
    if bound_known:
        row["reservation_upper_bound_usd"] = total("reservation_upper_bound_usd")
    for field in _TOKEN_FIELDS:
        values = [int(item[field]) for item in rows if item.get(field) is not None]
        if values:
            row[field] = sum(values)
    limits = [Decimal(str(item["root_limit_usd"])) for item in rows if item.get("root_limit_usd") is not None]
    if limits:
        row["root_limit_usd"] = format(min(limits), "f")
    processing: dict = {}
    for item in rows:
        _merge_processing_summary(processing, _processing_summary([item], decimal_values=True))
    if processing:
        row["processing_summary"] = {name: format(value, "f") if isinstance(value, Decimal) else value
                                     for name, value in processing.items()}
    return row


def fold_into_archive(root, attempt_ids):
    """Rewrite ``root`` as an install upgraded from a compacted journal: the
    named closed attempts' journal rows move to a retained archive segment;
    the journal keeps one aggregate per attribution tuple of them (the retired
    compactor's grouping; earlier aggregates fold into the new block) plus
    every other attempt's chain; the next money access imports it. Returns
    the archived rows."""
    from ouroboros.usage_ledger import is_abandoned_settlement

    attempt_ids = set(attempt_ids)
    usage_store.export_journal(root)  # the store back to a journal of legal chains
    path = root / LEDGER_REL
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    path.unlink()
    archived = [row for row in rows if row.get("attempt_id") in attempt_ids]
    finals = {row["attempt_id"]: row for row in archived}
    assert set(finals) == attempt_ids
    for row in finals.values():
        assert (str(row.get("kind") or "attempt") == "attempt" and row["state"] in {"settled", "released"}
                and not is_abandoned_settlement(row)
                and not any(row.get(key) for key in ("review_skill", "review_wave_id", "review_slot_id"))), row
    folded = [*finals.values(), *(row for row in rows if row.get("kind") == "usage_baseline_group")]
    kept = [{key: value for key, value in row.items() if key != "seq"} for row in rows
            if row.get("attempt_id") not in attempt_ids
            and row.get("kind") not in {"usage_baseline", "usage_baseline_group"}]
    groups: dict = {}
    for row in folded:
        groups.setdefault(_group_key(row), []).append(row)
    segment = root / ARCHIVE_SEGMENT_REL
    segment.parent.mkdir(parents=True, exist_ok=True)
    with segment.open("a", encoding="utf-8") as handle:
        handle.write("".join(json.dumps(row, sort_keys=True) + "\n" for row in archived))
    write_compacted_journal(root, [_aggregate(index, key, groups[key])
                                   for index, key in enumerate(sorted(groups, key=repr), 1)], kept)
    usage_store.forget(root)
    return archived


def foldable_attempt_ids(root):
    """The attempts the retired compactor folded: closed, non-review, not an
    abandoned settlement (which keeps its late-receipt right)."""
    from ouroboros.usage_ledger import is_abandoned_settlement

    return [row["attempt_id"] for row in ledger_rows(root)
            if str(row.get("kind") or "attempt") == "attempt" and row.get("state") in {"settled", "released"}
            and not is_abandoned_settlement(row)
            and not any(row.get(key) for key in ("review_skill", "review_wave_id", "review_slot_id"))
            and not isinstance(row.get("cost_usd"), bool)
            and not isinstance(row.get("reservation_upper_bound_usd"), bool)]
