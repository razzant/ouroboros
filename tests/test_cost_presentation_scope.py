"""#498 — the scoped cost carrier: one amount and the facts that explain it.

The defect this pins: a card could take its number from the task subtree and its
unknown-price markers from the task itself, so a parent that had spent nothing
showed an honest ``$0.00`` over a child whose price nobody knew. The carrier is
built by the producer that summed the rows and states which scope it describes,
so a consumer can never pair the two halves of different questions.

Monetary sums and ``cost_final`` are deliberately unchanged here — the carrier
only adds facts beside them.
"""
from __future__ import annotations

import pytest

from ouroboros._usage_rows import _summary
from ouroboros.cost_projection import (
    COST_OPENNESS_FIELDS,
    COST_SCOPE_OWN,
    COST_SCOPE_ROOT_TREE,
    build_cost_presentation,
    carry_cost_meta,
    cost_projection,
)


def _row(**over):
    return {"kind": "attempt", "state": "settled", "cost_usd": 1.0, "cost_final": True, **over}


class TestPricedEvidence:
    """A zero survives only where a priced or bounded row actually evidenced it."""

    def test_proven_zero_keeps_its_amount(self):
        summary = _summary([_row(cost_usd=0.0)])
        assert summary["priced_rows"] == 1
        carrier = build_cost_presentation(summary, scope=COST_SCOPE_OWN)
        assert carrier == {
            "scope": "own", "tracked_amount": 0.0, "has_unpriced": False,
            "tracked_final": True, "accounting_open": False, "has_rows": True,
        }

    def test_estimated_zero_is_open_not_a_settled_free_result(self):
        summary = _summary([_row(cost_usd=0.0, cost_final=False)])
        carrier = build_cost_presentation(summary, scope=COST_SCOPE_OWN)
        assert carrier["tracked_amount"] == 0.0
        assert carrier["tracked_final"] is False and carrier["accounting_open"] is True

    def test_empty_ledger_is_unknown_not_zero(self):
        carrier = build_cost_presentation(_summary([]), scope=COST_SCOPE_OWN)
        assert carrier["tracked_amount"] is None
        assert carrier["has_rows"] is False and carrier["has_unpriced"] is False

    def test_all_unpriced_without_a_bound_is_unknown(self):
        summary = _summary([_row(cost_usd=None), _row(cost_usd=None)])
        assert summary["priced_rows"] == 0 and summary["unknown_unmetered"] == 2
        carrier = build_cost_presentation(summary, scope=COST_SCOPE_OWN)
        assert carrier["tracked_amount"] is None
        assert carrier["has_rows"] is True and carrier["has_unpriced"] is True

    def test_mixed_zero_keeps_the_tracked_zero_and_discloses_the_unpriced_rows(self):
        # A priced $0 sharing a scope with an unpriced row: the tracked zero is
        # real evidence and survives, with the unknown stated beside it. The
        # count-free heuristic used everywhere else discards exactly this case.
        from ouroboros.cost_projection import honest_accounted_amount

        summary = _summary([_row(cost_usd=0.0), _row(cost_usd=None)])
        assert honest_accounted_amount(summary) is None
        carrier = build_cost_presentation(summary, scope=COST_SCOPE_OWN)
        assert carrier["tracked_amount"] == 0.0 and carrier["has_unpriced"] is True
        assert carrier["tracked_final"] is True
        assert carrier["accounting_open"] is False

    def test_a_retained_bound_is_priced_evidence_and_keeps_the_scope_open(self):
        summary = _summary([{"kind": "attempt", "state": "dispatched",
                             "reservation_upper_bound_usd": 2.0}])
        carrier = build_cost_presentation(summary, scope=COST_SCOPE_OWN)
        assert summary["priced_rows"] == 1
        assert carrier["tracked_amount"] == 2.0
        assert carrier["tracked_final"] is False and carrier["accounting_open"] is True

    def test_an_integrity_gap_is_never_exact(self):
        from ouroboros._usage_rows import _with_integrity

        summary = _with_integrity(_summary([_row()]), True)
        carrier = build_cost_presentation(summary, scope=COST_SCOPE_OWN)
        assert carrier["tracked_amount"] == 1.0
        assert carrier["tracked_final"] is False and carrier["accounting_open"] is True

    def test_unavailable_accounting_carries_no_carrier_at_all(self):
        assert build_cost_presentation(None, scope=COST_SCOPE_OWN) is None


class TestLatePriceAndCompaction:
    def test_a_late_receipt_moves_the_carrier_without_stickiness(self):
        open_summary = _summary([{"kind": "attempt", "state": "dispatched",
                                  "reservation_upper_bound_usd": 2.0}])
        assert build_cost_presentation(open_summary, scope=COST_SCOPE_OWN)["tracked_final"] is False
        settled = _summary([_row(cost_usd=1.4)])
        carrier = build_cost_presentation(settled, scope=COST_SCOPE_OWN)
        assert carrier["tracked_amount"] == 1.4 and carrier["tracked_final"] is True

    def test_a_compacted_group_answers_exactly_like_the_rows_it_folded(self):
        rows = [_row(cost_usd=0.5), _row(cost_usd=0.5), _row(cost_usd=None)]
        folded = [
            {"kind": "usage_baseline_group", "state": "settled", "cost_usd": 1.0,
             "cost_final": True, "folded_attempt_count": 2},
            {"kind": "usage_baseline_group", "state": "settled", "cost_usd": None,
             "folded_attempt_count": 1},
        ]
        raw_summary, folded_summary = _summary(rows), _summary(folded)
        assert raw_summary["priced_rows"] == folded_summary["priced_rows"] == 2
        assert (build_cost_presentation(raw_summary, scope=COST_SCOPE_OWN)
                == build_cost_presentation(folded_summary, scope=COST_SCOPE_OWN))


class TestScopeSelection:
    """The root/child reversal: presence selects the subtree, never a null fallback."""

    def test_a_root_tree_carrier_states_its_own_scope(self):
        carrier = build_cost_presentation(_summary([_row()]), scope=COST_SCOPE_ROOT_TREE)
        assert carrier["scope"] == "root_tree"

    def test_a_null_subtree_amount_never_falls_back_to_the_parents_own_zero(self):
        # The parent itself spent a proven zero; the child's price is unknown.
        own = build_cost_presentation(_summary([_row(cost_usd=0.0)]), scope=COST_SCOPE_OWN)
        subtree = build_cost_presentation(
            _summary([_row(cost_usd=None)]), scope=COST_SCOPE_ROOT_TREE)
        assert own["tracked_amount"] == 0.0
        assert subtree["tracked_amount"] is None and subtree["has_unpriced"] is True
        # A reader that takes the root_tree carrier gets the honest unknown; the
        # two carriers never merge, because each names the scope it describes.
        assert own["scope"] != subtree["scope"]


class TestCarryThrough:
    def test_the_carrier_is_an_openness_marker_and_rides_every_shared_seam(self):
        from ouroboros.task_results import TASK_COST_META_FIELDS

        assert "cost_presentation" in COST_OPENNESS_FIELDS
        assert "cost_presentation" in TASK_COST_META_FIELDS
        carrier = build_cost_presentation(_summary([_row()]), scope=COST_SCOPE_OWN)
        source = {"accounted_upper_bound_usd": 1.0, "cost_final": True,
                  "cost_presentation": carrier}
        assert carry_cost_meta(source)["cost_presentation"] == carrier
        assert cost_projection(source)["cost_presentation"] == carrier
        # Absent stays absent: normalization never invents the carrier.
        assert "cost_presentation" not in carry_cost_meta({"accounted_upper_bound_usd": 1.0})

    def test_monetary_sums_and_finality_are_unchanged_by_the_new_fact(self):
        summary = _summary([_row(cost_usd=0.5), _row(cost_usd=None)])
        assert summary["settled_usd"] == 0.5
        assert summary["accounted_usd"] == 0.5
        assert summary["cost_final"] is False
        assert summary["unknown_unmetered"] == 1

    def test_the_task_result_fields_carry_the_own_scope_carrier(self, tmp_path):
        from supervisor.state import reconstruct_task_cost

        fields = reconstruct_task_cost("nothing-here", fields=True, drive_root=tmp_path)
        assert "cost_presentation" in fields
        if fields.get("cost_accounting_status") == "available":
            # An empty ledger is not a measured zero, even though the sum is 0.0.
            assert fields["cost_presentation"] == {
                "scope": "own", "tracked_amount": None, "has_unpriced": False,
                "tracked_final": False, "accounting_open": False, "has_rows": False,
            }
        else:
            # An unreadable ledger says so through its own status, with no carrier.
            assert fields["cost_presentation"] is None


def test_an_unreadable_count_degrades_to_no_carrier_instead_of_a_wrong_one():
    # A count that cannot be read at all yields NO carrier — never a confident
    # amount over evidence this function could not parse.
    assert build_cost_presentation(
        {**_summary([_row()]), "priced_rows": "x"}, scope=COST_SCOPE_OWN) is None
    # An absent count reads as "nothing priced", which is the conservative answer.
    assert build_cost_presentation(
        {**_summary([_row()]), "priced_rows": None},
        scope=COST_SCOPE_OWN)["tracked_amount"] is None


def test_settled_unknown_does_not_make_known_subtotal_open():
    summary = _summary([_row(cost_usd=1.2), _row(cost_usd=None)])
    carrier = build_cost_presentation(summary, scope=COST_SCOPE_ROOT_TREE)
    assert carrier['tracked_amount'] == 1.2 and carrier['tracked_final'] is True
    assert carrier['accounting_open'] is False and carrier['has_unpriced'] is True
    assert summary['cost_final'] is False  # unchanged monetary contract


def test_zero_bounds_and_estimates_are_positive_nonfinal_row_facts():
    for row in [
        _row(cost_usd=0, cost_final=False),
        _row(cost_usd=None, reservation_upper_bound_usd=0),
        {'state': 'reserved', 'reservation_upper_bound_usd': 0},
        {'state': 'dispatched', 'reservation_upper_bound_usd': 0},
        {'state': 'unresolved', 'reservation_upper_bound_usd': 0},
    ]:
        carrier = build_cost_presentation(_summary([row]), scope=COST_SCOPE_OWN)
        assert carrier['tracked_amount'] == 0
        assert carrier['tracked_final'] is False and carrier['accounting_open'] is True


def test_legacy_summary_without_positive_row_facts_cannot_claim_exactness():
    assert build_cost_presentation({'accounted_usd': 0, 'cost_final': True}, scope=COST_SCOPE_OWN) is None


# Production money fixtures, executed only by the parent's isolated suite.
from tests import fixtures_usage_store as compaction_fixtures

data_root = compaction_fixtures.data_root


def test_an_imported_compacted_journal_preserves_all_presentation_facts(data_root):
    from ouroboros import usage_accounting as usage

    compaction_fixtures._seed_mixed_ledger(data_root)
    compaction_fixtures._settle(data_root, cost=0, cost_final=True, task_id='zero')
    compaction_fixtures._settle(data_root, cost=0, cost_final=False, task_id='estimated-zero')
    before = usage.usage_breakdown(data_root, root_task_id='root')
    compaction_fixtures.fold_into_archive(data_root, compaction_fixtures.foldable_attempt_ids(data_root))
    assert any(row['kind'] == 'usage_baseline_group' for row in compaction_fixtures._ledger_rows(data_root))
    after = usage.usage_breakdown(data_root, root_task_id='root')
    for field in ['priced_rows', 'tracked_nonfinal_rows', 'accounting_open_rows', 'unknown_unmetered',
                  'accounted_usd', 'settled_usd', 'reserved_usd', 'unresolved_upper_bound_usd', 'cost_final']:
        assert after[field] == before[field], field
    assert build_cost_presentation(after, scope=COST_SCOPE_ROOT_TREE) == build_cost_presentation(before, scope=COST_SCOPE_ROOT_TREE)


@pytest.mark.serial
def test_producer_to_wire_consumers_keep_tree_scope_and_late_unknown_price(data_root):
    import asyncio
    import json
    import subprocess
    from pathlib import Path
    from types import SimpleNamespace
    from ouroboros import usage_accounting as usage
    from ouroboros.cost_projection import live_root_cost_projection
    from ouroboros.gateway.history import make_chat_history_endpoint
    from ouroboros.post_task_checkpoint import project_replica_task_result_fields
    from ouroboros.post_task_synthesis import _pre_synthesis_usage_snapshot
    from ouroboros.task_results import write_task_result
    from ouroboros.utils import append_jsonl
    from supervisor.events_task_done import _authoritative_terminal_cost
    from supervisor.state import reconstruct_task_cost

    # Real priced parent zero plus real child's retained unresolved bound.
    compaction_fixtures._settle(data_root, cost=0, cost_final=True, task_id='root')
    child = usage.reserve_attempt(compaction_fixtures._request(data_root, task_id='child', reservation_usd=2.0))
    usage.mark_dispatched(child)
    usage.mark_unresolved(child, 'receipt unavailable')
    usage.terminalize_abandoned_attempt(child, reason='child exited without receipt')
    task = {'id': 'root', 'root_task_id': 'root', 'chat_id': 1}
    terminal = _authoritative_terminal_cost('root', task, {}, {}, data_root)
    heartbeat = live_root_cost_projection('root', task, {}, data_root)
    own = reconstruct_task_cost('root', fields=True, drive_root=data_root)
    assert own['cost_presentation']['tracked_amount'] == 0
    assert terminal['cost_presentation']['scope'] == heartbeat['cost_presentation']['scope'] == 'root_tree'
    assert terminal['cost_presentation']['tracked_amount'] == 2
    assert terminal['cost_presentation']['tracked_final'] is False
    overlay = project_replica_task_result_fields(terminal, own)
    assert 'cost_presentation' not in overlay
    synthesis = _pre_synthesis_usage_snapshot(SimpleNamespace(drive_root=data_root), task, own)
    assert synthesis['cost_presentation'] == terminal['cost_presentation']
    write_task_result(data_root, 'root', 'completed', root_task_id='root', chat_id=1, **terminal)
    append_jsonl(data_root / 'logs/progress.jsonl', {
        'type': 'send_message', 'is_progress': True, 'task_id': 'root', 'chat_id': 1,
        'content': 'Reading source', 'text': 'Reading source', 'ts': '2026-09-20T00:00:00Z', **terminal,
    })
    response = asyncio.run(make_chat_history_endpoint(data_root)(SimpleNamespace(query_params={'chat_id': '1'})))
    rows = json.loads(response.body)['messages']
    history = next(row for row in rows if row.get('task_id') == 'root' and row.get('cost_presentation'))
    assert history['cost_presentation'] == terminal['cost_presentation']
    # A real late receipt replaces the abandoned unknown row. The previous
    # mixed subtotal cannot be sticky-final, even when lifecycle was completed.
    usage.settle_attempt(child, cost_usd=0.35, cost_final=True)
    assert compaction_fixtures._ledger_rows(data_root)[-1]['settle_reason'] == 'late_receipt'
    late = _authoritative_terminal_cost('root', task, {}, {}, data_root)
    assert late['cost_presentation']['has_unpriced'] is False
    assert late['cost_presentation']['tracked_final'] is True
    # Feed production payloads to the real JS consumer, including the child
    # literal/card-meta reader and its sticky reducer. No handcrafted carrier.
    script = """
        import fs from 'node:fs';
        import {taskCostMeta, taskCostProjection, mergeStickyCostMeta, cardMetaKeys} from './web/modules/chat_activity.js';
        const rows = JSON.parse(fs.readFileSync(0, 'utf8'));
        const projections = rows.map((row, i) => taskCostProjection(cardMetaKeys(row), `2026-09-20T00:0${i}:00Z`));
        const sticky = mergeStickyCostMeta(projections[0], projections[3]);
        console.log(JSON.stringify({meta: rows.map(taskCostMeta), sticky: sticky.meta,
            late: mergeStickyCostMeta(sticky, projections[4]).meta}));
    """
    result = subprocess.run(['node', '--input-type=module', '-e', script],
                            cwd=Path(__file__).parents[1], input=json.dumps([terminal, heartbeat, history, own, late]),
                            capture_output=True, text=True, check=True)
    browser = json.loads(result.stdout)
    assert browser['meta'] == [['Tracked: up to $2.00', 'some steps have no price']] * 3 + [['$0.00'], ['$0.35']]
    assert browser['sticky'] == ['Tracked: up to $2.00', 'some steps have no price']
    assert browser['late'] == ['$0.35']


def test_first_terminal_root_frame_preserves_money_on_tree_failure(monkeypatch):
    from ouroboros.cost_projection import with_task_cost_presentation

    original = {'cost_accounting_status': 'available', 'accounted_upper_bound_usd': 0,
                'cost_final': True, 'cost_presentation': {'scope': 'own', 'tracked_amount': 0}}
    monkeypatch.setattr('ouroboros.usage_accounting.usage_breakdown',
                        lambda *_a, **_kw: (_ for _ in ()).throw(OSError('unreadable tree')))
    output = with_task_cost_presentation(original, {'id': 'root', '_skip_post_task_synthesis': True}, '.')
    assert output == {**original, 'cost_presentation': None}


@pytest.mark.parametrize('rollup, swarm, keeps_own', [
    (4.25, {}, False),          # a legacy foreign subtree total: own zero may not stand for it
    (0, {'subagent_count': 2}, False),  # the child fanned out; its own mirror is not the subtree
    (0, {}, True),              # a leaf's pipeline mirror of its own bound carries no second scope
    (None, {}, True),
])
def test_nested_rollup_scope_rule(tmp_path, monkeypatch, rollup, swarm, keeps_own):
    from supervisor.events_task_done import _authoritative_terminal_cost
    own = {'cost_accounting_status': 'available', 'accounted_upper_bound_usd': 0,
           'cost_final': True, 'cost_presentation': {'scope': 'own', 'tracked_amount': 0}}
    monkeypatch.setattr('supervisor.state.reconstruct_task_cost', lambda *_a, **_kw: dict(own))
    result = _authoritative_terminal_cost('child', {'parent_task_id': 'root', 'root_task_id': 'root'},
        {'accounted_upper_bound_usd_with_children': rollup, 'swarm_efficiency': swarm}, {}, tmp_path)
    assert result['accounted_upper_bound_usd_with_children'] == rollup
    assert result['cost_presentation'] == (own['cost_presentation'] if keeps_own else None)
    assert result['accounted_upper_bound_usd'] == 0
