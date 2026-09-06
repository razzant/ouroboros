"""C2 (owner 10=B) + ABI 7.0 (ABI-3): the SSOT cost projection.

`accounted_upper_bound_usd` is the honest name for what `cost_usd` always was.
Since ABI-3 the deprecated alias spellings are READ-ONLY tolerance for stored
legacy records (deprecated wins on a diverged pair) and are never EMITTED:
every write/read seam strips them. Null is null: unknown cost never renders as
$0.00 and finality is never fabricated.
"""

from __future__ import annotations

from ouroboros.cost_projection import (
    cost_display,
    cost_projection,
    honest_accounted_amount,
    with_cost_aliases,
)


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
        from ouroboros.agent_task_pipeline import _synthesis_cost_text

        unknown = {"cost": cost_projection({"status": "running"})["accounted_upper_bound_usd"]}
        assert unknown["cost"] is None
        assert "$0.00" not in _synthesis_cost_text(unknown)
        assert "unknown" in _synthesis_cost_text(unknown)
        # A REAL zero still reads as a zero.
        assert _synthesis_cost_text({"cost": 0.0}) == "$0.00"


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
