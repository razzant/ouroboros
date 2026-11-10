"""C2 (owner 10=B) + ABI 7.0 (ABI-3): the SSOT cost projection.

`accounted_upper_bound_usd` is the honest name for what `cost_usd` always was.
Since ABI-3 the deprecated alias spellings are READ-ONLY tolerance for stored
legacy records (deprecated wins on a diverged pair) and are never EMITTED:
every write/read seam strips them. Null is null: unknown cost never renders as
$0.00 and finality is never fabricated.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from tests._usage_store_testing import root as root
from ouroboros.cost_projection import (
    cost_display,
    cost_projection,
    honest_accounted_amount,
    with_cost_aliases,
)


@pytest.mark.parametrize("status", ["running", "cancel_requested", "completed"])
def test_current_cost_refreshes_live_totals_without_rewriting_terminal_receipts(root, status):
    from ouroboros import usage_accounting as ua
    from ouroboros.cost_projection import current_cost
    from ouroboros.task_results import write_task_result
    from ouroboros.tools.join_ledger import _child_result_sha256

    write_task_result(root, "parent", "running", root_task_id="parent")
    ua.record_subscription_session(session_id="run-cost", route="test", spend_usd=3.5,
        task_id="child", root_task_id="parent", drive_root=root)
    data = {"task_id": "child", "status": status, "result": "answer", "cost_usd": 0.1,
            "accounted_upper_bound_usd": 0.1, "cost_final": True}
    projected = current_cost(data, root)
    if status == "completed":
        assert projected == data
    else:
        assert projected["accounted_upper_bound_usd"] == 3.5 and projected["cost_final"] is False
        assert "cost_usd" not in projected
    assert data["cost_usd"] == 0.1 and data["cost_final"] is True
    assert _child_result_sha256(projected) == _child_result_sha256(data)


@pytest.mark.parametrize("accounting", ["priced", "empty", "unavailable"])
def test_live_cost_readers_refresh_the_effective_task_and_batch_once(root, monkeypatch, accounting):
    from ouroboros import task_status, usage_accounting as ua
    from ouroboros.task_results import write_task_result
    from ouroboros.tools import control_task_results as results, join_ledger

    write_task_result(root, "parent", "running", root_task_id="parent")
    amount = {"priced": 3.5, "free": 0.0}.get(accounting)
    rows = [{"task_id": tid, "parent_task_id": "parent", "root_task_id": "parent",
             "status": "running", "result": "answer", "accounted_upper_bound_usd": 0.1, "cost_final": True}
            for tid in ("successor", "second", "third")]
    for row in rows if accounting != "empty" else []:
        ua.record_subscription_session(session_id="run-" + row["task_id"], route="test", spend_usd=amount,
            task_id=row["task_id"], root_task_id="parent", drive_root=root)
    ctx = SimpleNamespace(drive_root=root / "mirror", budget_drive_root=root, task_id="parent",
                          task_metadata={"root_task_id": "parent"})
    def effective(drive, tid):
        assert drive == root and tid == "predecessor"
        return dict(rows[0])
    monkeypatch.setattr(results, "load_effective_task_result", effective)
    monkeypatch.setattr(join_ledger, "load_effective_task_result", effective)
    monkeypatch.setattr(task_status, "find_child_tasks", lambda *_a, **_k: rows)
    reads, real_read = [], ua.usage_breakdown
    def counted(drive, **kwargs):
        reads.append((drive, kwargs))
        if accounting == "unavailable":
            raise OSError("ledger unreadable")
        return real_read(drive, **kwargs)
    monkeypatch.setattr(ua, "usage_breakdown", counted)

    authority = json.loads(results._get_task_result(ctx, "predecessor", include_authority=True))["authority"]
    assert authority["accounted_upper_bound_usd"] == 0.1 and authority["cost_final"] is True
    assert reads == []
    summary = json.loads(results._get_task_result(ctx, "predecessor", view="summary"))["cost"]
    assert summary["accounted_upper_bound_usd"] == amount and summary["cost_final"] is False
    assert reads[-1][1]["task_id"] == "successor"
    peek = join_ledger._peek_task(ctx, "predecessor")
    assert (f"cost=${amount:.2f} (upper bound, not final)" if amount is not None else "cost=unknown") in peek
    single = results._unsettled_wait_body(root, "successor", rows[0], "")
    compact = json.loads(single.split("[CHILD_PROJECTION]\n", 1)[1].split("\n[/CHILD_PROJECTION]", 1)[0])
    assert compact["accounted_upper_bound_usd"] == amount and compact["cost_final"] is False

    reads.clear()
    batch = results._batch_wait_payload(ctx, {"tasks": {row["task_id"]: row for row in rows}},
                                       [row["task_id"] for row in rows], {}, [])
    assert len(reads) == 1 and reads[0][1]["root_task_id"] == "parent"
    assert all(row["accounted_upper_bound_usd"] == amount and row["cost_final"] is False
               for row in batch["tasks"].values())
    reads.clear()
    roster = results._children_roster_projection(ctx, root)["children_roster"]
    assert len(reads) == 1 and len(roster) == 3
    assert all(row["accounted_upper_bound_usd"] == amount for row in roster)
    assert all(row["accounted_upper_bound_usd"] == 0.1 and row["cost_final"] for row in rows)


def test_terminal_batches_keep_the_receipt_without_reading_current_accounting(root, monkeypatch):
    from ouroboros import usage_accounting as ua
    from ouroboros.cost_projection import current_costs

    monkeypatch.setattr(ua, "usage_breakdown", lambda *_a, **_k: pytest.fail("terminal batch read accounting"))
    rows = [{"task_id": "done", "status": "completed", "accounted_upper_bound_usd": 0.1, "cost_final": True}]
    assert current_costs(rows, root) == rows


@pytest.mark.parametrize(("lineage_carrier", "same_tree"), [("top_level", True), ("metadata", False)])
def test_wait_cost_batch_reads_each_actual_root_once(root, monkeypatch, same_tree, lineage_carrier):
    from ouroboros import usage_accounting as ua
    from ouroboros.task_results import write_task_result
    from ouroboros.tools import control_task_results as results

    observer = "observer"
    roots = [observer] if same_tree else ["root-a", "root-b"]
    for tid in [observer, *roots]:
        write_task_result(root, tid, "running", root_task_id=tid)
    expected = {}
    for index, owner in enumerate(roots):
        for suffix in ("1", "2"):
            tid = owner + "-child-" + suffix
            lineage = {"root_task_id": owner} if lineage_carrier == "top_level" else {"metadata": {"root_task_id": owner}}
            write_task_result(root, tid, "scheduled", parent_task_id=owner, **lineage)
            expected[tid] = index + 3.5
            ua.record_subscription_session("run-" + tid, drive_root=root, task_id=tid,
                                           root_task_id=owner, route="test", spend_usd=expected[tid])
    reads, real_read = [], ua.usage_breakdown

    def counted(drive, **kwargs):
        reads.append(kwargs)
        return real_read(drive, **kwargs)

    monkeypatch.setattr(ua, "usage_breakdown", counted)
    ctx = SimpleNamespace(drive_root=root, budget_drive_root=root, task_id=observer,
                          task_metadata={"root_task_id": observer})
    waited = json.loads(results._wait_for_tasks(ctx, list(expected), timeout_sec=0))["tasks"]
    assert {tid: row["accounted_upper_bound_usd"] for tid, row in waited.items()} == expected
    assert all(row["cost_final"] is False for row in waited.values())
    assert sorted(row["root_task_id"] for row in reads) == sorted(roots)


def test_batch_unavailable_accounting_warns_once_with_the_original_cause(root, monkeypatch, caplog):
    import logging
    from ouroboros import usage_accounting as ua
    from ouroboros.cost_projection import current_costs

    reads = []

    def unavailable(*args, **kwargs):
        reads.append(kwargs)
        raise OSError("ledger unreadable")

    monkeypatch.setattr(ua, "usage_breakdown", unavailable)
    rows = [{"task_id": f"child-{i}", "root_task_id": "parent", "status": "running",
             "accounted_upper_bound_usd": 0.1, "cost_final": True} for i in range(5)]
    with caplog.at_level(logging.WARNING):
        result = current_costs(rows, root)
    assert len(reads) == 1
    assert all(row["accounted_upper_bound_usd"] is None and row["cost_final"] is False
               and row["cost_accounting_status"] == "unavailable" for row in result)
    warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert len(warnings) == 1 and warnings[0].levelno == logging.WARNING
    assert "ledger unreadable" in warnings[0].getMessage()
    assert "KeyError" not in caplog.text


def test_live_authority_pages_survive_new_usage_rows(root):
    from ouroboros import usage_accounting as ua
    from ouroboros.task_results import write_task_result
    from ouroboros.tools.control_task_results import _get_task_result

    write_task_result(root, "parent", "running", root_task_id="parent")
    write_task_result(root, "child", "scheduled", parent_task_id="parent", root_task_id="parent",
                      task_contract={"objective": "a stable contract"})
    ctx = SimpleNamespace(drive_root=root, budget_drive_root=root, task_id="parent",
                          task_metadata={"root_task_id": "parent"})
    args = {"authority_fields": ["task_contract"], "source_only": True}
    metadata = json.loads(_get_task_result(ctx, "child", **args))
    args["authority_projection_sha256"] = metadata["authority_projection_sha256"]
    first = json.loads(_get_task_result(ctx, "child", **args, source_start_char=0, source_end_char=10))
    ua.record_subscription_session("new-run", drive_root=root, task_id="child", root_task_id="parent",
                                   route="test", spend_usd=3.5)
    last = json.loads(_get_task_result(ctx, "child", **args, source_start_char=10,
                                     source_end_char=metadata["authority_source"]["complete_chars"]))
    assert last.get("reason") != "authority_identity_mismatch", last
    assert first["authority_projection_sha256"] == last["authority_projection_sha256"]
    text = first["authority_source"]["text"] + last["authority_source"]["text"]
    assert json.loads(text) == {"task_contract": {"objective": "a stable contract"}}
    assert json.loads(_get_task_result(ctx, "child", view="summary"))["cost"]["accounted_upper_bound_usd"] == 3.5


@pytest.mark.parametrize("selector", [
    {"call_id": "missing", "manifest_sha256": "a" * 64}, {"include_completion_source": True},
    {"view": "full"},
])
def test_source_and_non_cost_views_do_not_read_live_accounting(root, monkeypatch, selector):
    from ouroboros import usage_accounting as ua
    from ouroboros.task_results import write_task_result
    from ouroboros.tools.control_task_results import _get_task_result

    write_task_result(root, "parent", "running", root_task_id="parent")
    write_task_result(root, "child", "scheduled", parent_task_id="parent", root_task_id="parent")
    ctx = SimpleNamespace(drive_root=root, budget_drive_root=root, task_id="parent",
                          task_metadata={"root_task_id": "parent"})
    reads, real_read = [], ua.usage_breakdown

    def counted(*args, **kwargs):
        reads.append(kwargs)
        return real_read(*args, **kwargs)

    monkeypatch.setattr(ua, "usage_breakdown", counted)
    _get_task_result(ctx, "child", **selector)
    assert reads == []
    for view in ("summary", "evidence"):
        reads.clear()
        _get_task_result(ctx, "child", view=view)
        assert len(reads) == 1


class TestAliases:
    def test_legacy_spelling_resolves_but_is_never_emitted(self):
        out = with_cost_aliases({"cost_usd": 1.5})
        assert out == {"accounted_upper_bound_usd": 1.5}
        out = with_cost_aliases({"accounted_upper_bound_usd": 2.5})
        assert out == {"accounted_upper_bound_usd": 2.5}
        out = with_cost_aliases({"cost_usd_with_children": 3.0})
        assert out == {"accounted_upper_bound_usd_with_children": 3.0}

    def test_deprecated_name_wins_a_diverged_pair(self):
        # Legacy mutators between two seam crossings edit cost_usd; the seam
        # honors that edit (deprecated wins) while emitting the honest name only.
        out = with_cost_aliases({"cost_usd": None, "accounted_upper_bound_usd": 9.0})
        assert out == {"accounted_upper_bound_usd": None}

    def test_unknown_zero_is_not_reintroduced_by_aliasing(self):
        out = with_cost_aliases({
            "cost_usd": 0.0, "unknown_unmetered": 1,
            "cost_final": False,
        })
        assert "cost_usd" not in out
        assert out["accounted_upper_bound_usd"] is None

    def test_aliasing_never_invents_a_field(self):
        assert "accounted_upper_bound_usd" not in with_cost_aliases({"total_rounds": 3})
        assert with_cost_aliases(None) == {}

    def test_explicit_none_stays_none(self):
        out = with_cost_aliases({"cost_usd": None})
        assert out == {"accounted_upper_bound_usd": None}

    def test_idempotent(self):
        once = with_cost_aliases({"cost_usd": 4.0, "cost_final": True})
        assert with_cost_aliases(once) == once


class TestProjection:
    def test_unknown_zero_accounted_subtotal_is_null_but_measured_zero_survives(self):
        assert honest_accounted_amount({
            "accounted_usd": 0.0, "unknown_unmetered": 1,
            "reserved_usd": 0.0, "unresolved_upper_bound_usd": 0.0,
        }) is None
        assert honest_accounted_amount({
            "accounted_usd": 0.0, "unknown_unmetered": 0,
        }) == 0.0
        assert honest_accounted_amount({
            "accounted_usd": 1.25, "unknown_unmetered": 1,
        }) == 1.25

    def test_unknown_cost_is_null_and_never_final(self):
        out = cost_projection({"status": "completed"})
        assert out["accounted_upper_bound_usd"] is None
        assert "cost_usd" not in out
        assert out["cost_known"] is False
        assert out["cost_final"] is False

    def test_unknown_zero_source_cost_projects_as_null(self):
        out = cost_projection({
            "cost_usd": 0.0, "unknown_unmetered": 1,
            "cost_final": False,
        })
        assert out["accounted_upper_bound_usd"] is None
        assert out["cost_known"] is False

    def test_missing_key_default_never_fabricates_zero(self):
        # The exact $0-fabrication class: data.get("cost_usd", 0) at five sites.
        assert cost_projection({})["accounted_upper_bound_usd"] is None
        assert cost_projection(None)["accounted_upper_bound_usd"] is None

    def test_finality_requires_a_known_amount(self):
        assert cost_projection({"cost_final": True})["cost_final"] is False
        assert cost_projection({"cost_final": True, "cost_usd": 0.0})["cost_final"] is True

    def test_legacy_stored_spelling_projects_onto_the_honest_name(self):
        out = cost_projection({"cost_usd": 0.42, "cost_final": True})
        assert out["accounted_upper_bound_usd"] == 0.42
        assert "cost_usd" not in out and out["cost_known"] is True

    def test_with_children_name_projects_only_when_present(self):
        assert "accounted_upper_bound_usd_with_children" not in cost_projection({"cost_usd": 1.0})
        out = cost_projection({"cost_usd_with_children": 2.0})
        assert out["accounted_upper_bound_usd_with_children"] == 2.0
        assert "cost_usd_with_children" not in out

    def test_openness_flags_are_carried_not_dropped(self):
        out = cost_projection({"cost_usd": 1.0, "unknown_unmetered": 2, "non_final_rows": 1,
                               "cost_accounting_status": "available"})
        assert out["unknown_unmetered"] == 2 and out["non_final_rows"] == 1
        assert out["cost_accounting_status"] == "available"

    def test_boolean_is_not_an_amount(self):
        assert cost_projection({"cost_usd": True})["accounted_upper_bound_usd"] is None


class TestDisplay:
    def test_null_never_renders_as_zero_dollars(self):
        text = cost_display({"status": "failed"})
        assert "$0.00" not in text and "unknown" in text

    def test_final_amount_is_plain_and_open_amount_is_labelled(self):
        assert cost_display({"cost_usd": 1.234, "cost_final": True}) == "$1.23"
        assert cost_display({"cost_usd": 1.234}) == "$1.23 (upper bound, not final)"
        assert cost_display({"cost_usd": 0.0, "cost_final": True}) == "$0.00"
        # Under a cent is not a priced zero: four places, as the card and the review records print it.
        assert cost_display({"cost_usd": 0.0004, "cost_final": True}) == "$0.0004"
        # A non-finite amount is unknown, never a crash or $NaN (the web prints it as absent too); a huge one prints.
        for odd in (float("inf"), float("nan")):
            text = cost_display({"cost_usd": odd, "cost_final": True})
            assert text.startswith("unknown") and "$" not in text
        from ouroboros.cost_projection import usd_text
        assert usd_text(float("inf")) == "unknown" and usd_text(1e26).endswith(".00")
        assert usd_text(1e26, 6).startswith("$1000000000000000") and usd_text(1e26, 6).endswith(".000000")
        # Below four places a positive amount still never prints as zero (the card's formatUsdKnown).
        assert cost_display({"cost_usd": 0.00003, "cost_final": True}) == "<$0.0001"

    def test_uncounted_money_names_each_part_and_says_nothing_without_one(self):
        from ouroboros.cost_projection import uncounted_money_text

        both = {"reserved_usd": 2.5063172, "unresolved_upper_bound_usd": 8117.347347}
        assert uncounted_money_text(both) == (
            "$2.51 reserved for calls not yet sent; up to $8117.35 for calls without a known price")
        assert uncounted_money_text({"unresolved_upper_bound_usd": 1.0}) == "up to $1.00 for calls without a known price"
        assert uncounted_money_text({"global_reserved_usd": 3.0}, prefix="global_") == "$3.00 reserved for calls not yet sent"
        assert uncounted_money_text({"reserved_usd": 0.0, "unresolved_upper_bound_usd": None}) == ""
        assert uncounted_money_text(None) == ""
        assert uncounted_money_text({"reserved_usd": 0.004}) == "$0.0040 reserved for calls not yet sent"
        # A tie rounds up, as the card and Telegram round it.
        assert uncounted_money_text({"reserved_usd": 0.125}) == "$0.13 reserved for calls not yet sent"
        # A caller asking for more places keeps them (the ledger text prints six).
        assert uncounted_money_text({"reserved_usd": 0.004213}, decimals=6) == "$0.004213 reserved for calls not yet sent"

    def test_an_unknown_amount_says_why_as_far_as_it_is_known(self):
        unpriced = {"scope": "own", "tracked_amount": None, "has_unpriced": True, "tracked_final": False,
                    "accounting_open": False, "has_rows": True}
        assert cost_display({"cost_presentation": unpriced}) == "unknown (no price recorded for its calls)"
        # A call still in flight may yet get its price: not the subscription words.
        assert cost_display({"cost_presentation": {**unpriced, "accounting_open": True}}) == "unknown (not settled yet)"
        for unreadable in ({"cost_custody": {"status": "unavailable"}}, {"ledger_integrity_degraded": True}):
            assert cost_display({"cost_presentation": unpriced, **unreadable}) == "unknown (accounting unavailable)"
        assert cost_display({"cost_accounting_status": "unavailable"}) == "unknown (accounting unavailable)"
        assert cost_display({"status": "failed"}) == "unknown (accounting unavailable or not settled)"
        # No row in the carrier's scope: a legacy zero beside it is no amount (#2828), as on the card.
        empty = {"scope": "own", "tracked_amount": None, "has_unpriced": False, "tracked_final": False,
                 "accounting_open": False, "has_rows": False}
        legacy_zero = {"accounted_upper_bound_usd": 0.0, "cost_final": True, "cost_presentation": empty}
        assert cost_display(legacy_zero) == "unknown (no usage recorded)"
        assert cost_display({**legacy_zero, "cost_presentation": {**empty, "scope": "root_tree"}}) == "unknown (no usage recorded)"
        assert cost_display({**legacy_zero, "cost_presentation": {**empty, "has_unpriced": True}}) == (
            "unknown (a delegated run closed without a cost record)")
        assert cost_display({**legacy_zero, "cost_presentation": {**empty, "accounting_open": True, "has_unpriced": True}}) == (
            "unknown (not settled yet)")
        # Control: a priced zero keeps its $0.00.
        assert cost_display({**legacy_zero, "cost_presentation": {**empty, "has_rows": True, "tracked_amount": 0.0,
                                                                  "tracked_final": True}}) == "$0.00"


class TestProducers:
    def test_reconstruct_task_cost_fields_carry_the_honest_name_only(self, tmp_path):
        from supervisor.state import reconstruct_task_cost

        fields = reconstruct_task_cost("some-task", fields=True, drive_root=tmp_path)
        assert "accounted_upper_bound_usd" in fields
        assert "cost_usd" not in fields

    def test_subagent_absorption_renders_unknown_not_zero(self):
        from ouroboros.task_status import format_subagent_absorption_message

        text = format_subagent_absorption_message([
            {"task_id": "c1", "status": "completed", "child_status": "completed",
             "result": "done", "role": "researcher", "parent_task_id": "p1"},
        ], parent_task_id="p1")
        assert "$0.0000" not in text
        assert "unknown" in text

    def test_task_cost_breakdown_carries_the_honest_total(self):
        from typing import get_type_hints

        from ouroboros.gateway.contracts import TaskCostBreakdown

        assert "accounted_upper_bound_usd" in get_type_hints(TaskCostBreakdown)

    def test_meta_fields_are_honest_only_and_legacy_resolves_via_carry(self):
        # Ф3.1 fix-round-2 (converted OLD-ABI clause: the list used to keep
        # both spellings as a raw carry): TASK_COST_META_FIELDS names the
        # HONEST set only — a retired alias may never travel forward by key
        # copy. Stored-legacy tolerance lives in carry_cost_meta instead:
        # the pair resolves deprecated-wins and leaves under the honest name.
        from ouroboros.cost_projection import carry_cost_meta
        from ouroboros.task_results import TASK_COST_META_FIELDS

        assert "accounted_upper_bound_usd" in TASK_COST_META_FIELDS
        assert "accounted_upper_bound_usd_with_children" in TASK_COST_META_FIELDS
        assert "cost_usd" not in TASK_COST_META_FIELDS
        assert "cost_usd_with_children" not in TASK_COST_META_FIELDS
        carried = carry_cost_meta({"cost_usd": 1.25, "cost_final": True})
        assert carried["accounted_upper_bound_usd"] == 1.25
        assert "cost_usd" not in carried


class TestOnePrecedence:
    """F7: the read seam and the write seam must pick the SAME winner."""

    def test_read_and_write_agree_on_a_diverged_pair(self):
        diverged = {"cost_usd": 1.0, "accounted_upper_bound_usd": 9.0}
        written = with_cost_aliases(diverged)
        read = cost_projection(diverged)
        assert written["accounted_upper_bound_usd"] == 1.0
        assert read["accounted_upper_bound_usd"] == 1.0

    def test_resolver_is_the_one_answer_for_both(self):
        from ouroboros.cost_projection import COST_ALIAS_PAIRS, resolve_cost_pair

        assert resolve_cost_pair({"cost_usd": 1.0, "accounted_upper_bound_usd": 9.0},
                                 *COST_ALIAS_PAIRS[0]) == (True, 1.0)
        assert resolve_cost_pair({"accounted_upper_bound_usd": 9.0},
                                 *COST_ALIAS_PAIRS[0]) == (True, 9.0)
        assert resolve_cost_pair({}, *COST_ALIAS_PAIRS[0]) == (False, None)

    def test_persisted_result_is_honest_only_and_legacy_records_still_read(self, tmp_path):
        from ouroboros.task_results import load_task_result, write_task_result

        write_task_result(tmp_path, "t1", "completed",
                          **with_cost_aliases({"cost_usd": 2.0, "cost_final": True}))
        stored = load_task_result(tmp_path, "t1")
        assert stored["accounted_upper_bound_usd"] == 2.0
        assert "cost_usd" not in stored
        # A legacy record written by an older release keeps reading through the
        # pair resolver — ABI-3 removed emission, never stored-history reads.
        write_task_result(tmp_path, "t0", "completed", cost_usd=3.5, cost_final=True)
        legacy = load_task_result(tmp_path, "t0")
        assert cost_projection(legacy)["accounted_upper_bound_usd"] == 3.5


class TestOpennessCarry:
    """F12: openness/integrity markers travel with the amount, from ONE list."""

    def test_carry_includes_reserved_unresolved_and_integrity(self):
        from ouroboros.cost_projection import carry_cost_meta

        carried = carry_cost_meta({
            "cost_usd": 3.0, "reserved_usd": 0.5,
            "unresolved_upper_bound_usd": 1.25, "ledger_integrity_degraded": True,
            "cost_final": False, "irrelevant": "x",
        })
        assert carried["accounted_upper_bound_usd"] == 3.0
        assert "cost_usd" not in carried
        assert carried["reserved_usd"] == 0.5
        assert carried["unresolved_upper_bound_usd"] == 1.25
        assert carried["ledger_integrity_degraded"] is True
        assert "irrelevant" not in carried

    def test_durable_meta_fields_derive_from_the_ssot(self):
        from ouroboros.cost_projection import COST_ALIAS_PAIRS, COST_OPENNESS_FIELDS
        from ouroboros.task_results import TASK_COST_META_FIELDS

        # Ф3.1 fix-round-2: derived from the SSOT's HONEST names only (the
        # retired alias spellings are read tolerance, never part of the
        # carry-forward set).
        expected = {new for new, _old in COST_ALIAS_PAIRS}
        expected |= set(COST_OPENNESS_FIELDS)
        assert set(TASK_COST_META_FIELDS) == expected
        assert "ledger_integrity_degraded" in TASK_COST_META_FIELDS

    def test_the_subagent_terminal_frame_covers_every_ssot_field(self):
        # The frame's KEYS must stay literal (a ChatOutbound key set has to be
        # statically checkable — tests/test_contracts.py), so this is the check
        # that keeps the literal honest: a marker added to the SSOT must appear
        # on the frame instead of being silently dropped like the three before it.
        import inspect

        from ouroboros.cost_projection import COST_OPENNESS_FIELDS
        from supervisor.events import _finish_task_done_dispatch

        source = inspect.getsource(_finish_task_done_dispatch)
        for field in (*COST_OPENNESS_FIELDS, "accounted_upper_bound_usd"):
            assert f'"{field}"' in source, f"the terminal subagent frame drops {field}"
        # ABI-3: the retired alias key never appears as an emitted frame key.
        assert '"cost_usd":' not in source

    def test_integrity_marker_is_declared_on_both_contract_mirrors(self):
        from typing import get_type_hints

        from ouroboros.gateway.contracts import ChatOutbound

        assert "ledger_integrity_degraded" in get_type_hints(ChatOutbound, include_extras=True)


class TestRestartRecoverySynthesis:
    """F11: an unknown cost must not become a confident $0.00 on recovery."""

    def test_recovered_usage_keeps_null(self):
        from ouroboros.agent_task_pipeline import _summary_row_cost_fields

        key = "accounted_upper_bound_usd_with_children"
        unknown = {key: cost_projection({"status": "running"})["accounted_upper_bound_usd"]}
        assert unknown[key] is None
        assert _summary_row_cost_fields(unknown) == {key: None}
        # A REAL zero still reads as a zero.
        assert _summary_row_cost_fields({key: 0.0}) == {key: 0.0}


class TestModelVisibleToolSurfaces:
    """External-audit correction lane (base 8827fd2c), item 1: the wait_tasks
    tool DESCRIPTION promised the model a ``cost_usd`` projection key while the
    producer (control_task_results) emits the ABI-3 honest pair
    ``accounted_upper_bound_usd`` + ``cost_final``. Model-visible tool text must
    name the keys the projection actually carries — a description teaching the
    model a removed alias is a lie the model then acts on."""

    def test_wait_tasks_description_names_the_actual_projection_keys(self, tmp_path):
        import pathlib

        from ouroboros.tools.registry import ToolRegistry

        repo_dir = pathlib.Path(__file__).resolve().parents[1]
        registry = ToolRegistry(repo_dir=repo_dir, drive_root=tmp_path)
        by_name = {t["function"]["name"]: t["function"] for t in registry.schemas()}
        desc = by_name["wait_tasks"]["description"]
        assert "cost_usd" not in desc
        assert "accounted_upper_bound_usd" in desc
        assert "cost_final" in desc

    def test_no_builtin_tool_schema_teaches_the_removed_alias(self, tmp_path):
        """Class-wide pin: NO builtin tool description or parameter doc may
        mention the removed ``cost_usd`` spelling (``accounted_upper_bound_usd``
        does not contain it, so honest names pass untouched)."""
        import json as _json
        import pathlib

        from ouroboros.tools.registry import ToolRegistry

        repo_dir = pathlib.Path(__file__).resolve().parents[1]
        registry = ToolRegistry(repo_dir=repo_dir, drive_root=tmp_path)
        offenders = [
            t["function"]["name"] for t in registry.schemas()
            if "cost_usd" in _json.dumps(t)
        ]
        assert offenders == []
