"""Tests for ``plan_task`` that are not the planning organ's own contract
(``tests/test_plan_task_tool.py``): the read-only validator and readers of an
earlier-schema (v2) plan review state, the plan record's store discipline, the tool's
public registration and description, the Plan Review Checklist section, the sequential
tool lane, the quorum budget and the typed lane facts of a plan row.
"""

from __future__ import annotations

import json
import pathlib
import unittest
import pytest
from tests._store_shared import result_signature, seed_result, unreadable_result

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _wave(fingerprint: str, *, aggregate: str = "REVIEW_REQUIRED", closed: bool = False) -> dict:
    return {
        "schema_version": 2, "cycle_index": 1, "request_fingerprint": fingerprint,
        "spec": {"goal": "g", "acceptance_claims": []},
        "spec_hash": "b" * 64, "findings": [], "aggregate": aggregate, "closed": closed,
        "dispositions": [], "paid": True,
    }


def _open_state() -> dict:
    """A v2 state with one open paid wave, as a finished task may still hold it."""
    fingerprint = "a" * 64
    return {"schema_version": 2, "series_id": "", "cycles_paid": 1, "need_evidence_seen": [],
            "current_attempt": {"fingerprint": fingerprint, "status": "open", "reason": ""},
            "waves": [_wave(fingerprint)], "waves_omitted": 0}


def _plan_task_entry():
    from ouroboros.tools.plan_review import get_tools

    return next(t for t in get_tools() if t.name == "plan_task")


def test_plan_review_state_validator_rejects_malformed_waves():
    from ouroboros.task_results import _validated_plan_review_state

    def _state(**wave_extra):
        state = _open_state()
        state["waves"][0].update(wave_extra)
        return state

    for bad in (
        {"aggregate": "MAYBE"},
        {"aggregate": "GREEN", "closed": False},
        {"aggregate": "REVISE_PLAN", "closed": True},
        {"closed": "yes"},
        {"request_fingerprint": "nope"},
        {"spec": "not-an-object"},
    ):
        with pytest.raises(ValueError, match="PLAN_REVIEW_STATE_INVALID"):
            _validated_plan_review_state(_state(**bad))
    with pytest.raises(ValueError, match="PLAN_REVIEW_STATE_INVALID"):
        _validated_plan_review_state({**_open_state(), "cycles_paid": -1})
    with pytest.raises(ValueError, match="PLAN_REVIEW_STATE_INVALID"):
        _validated_plan_review_state({"schema_version": 7})
    # A compact (bounded-history) wave needs no spec/findings; a v1 record loads read-only.
    compact = _open_state()
    compact["waves"][0] = {"compact": True, "request_fingerprint": "a" * 64,
                           "aggregate": "REVIEW_REQUIRED", "closed": False}
    assert _validated_plan_review_state(compact)["waves"][0]["compact"] is True
    legacy = _validated_plan_review_state({"schema_version": 1, "current_attempt": {}, "waves": []})
    assert legacy["schema_version"] == 2 and legacy["legacy_v1"]["schema_version"] == 1


def test_a_compact_wave_carries_the_ordered_weaker_fact():
    from ouroboros.tools.plan_review_artifacts import compact_wave

    wave = {"request_fingerprint": "f" * 64, "aggregate": "GREEN", "closed": True, "paid": True,
            "cycle_index": 1, "findings": [], "ordered_weaker": {"s1": {"effort": "low", "owner_effort": "xhigh"}}}
    assert compact_wave(wave)["ordered_weaker"] == {"s1": {"effort": "low", "owner_effort": "xhigh"}}
    plain = dict(wave); plain.pop("ordered_weaker")
    assert "ordered_weaker" not in compact_wave(plain)


def test_unreadable_parent_task_result_fails_closed_without_overwrite(tmp_path, monkeypatch):
    """The store refusing the parent row is a typed unknown for the readers and for the plan
    record's writer alike; none invents an empty state or writes over the row."""
    from ouroboros import plan_record
    from ouroboros.store import StoreError
    from ouroboros.task_results import load_plan_review_state

    seed_result(tmp_path, "parent-corrupt", {"task_id": "parent-corrupt", "status": "running"})
    before = result_signature(tmp_path, "parent-corrupt")
    unreadable = unreadable_result(monkeypatch, "parent-corrupt")

    with pytest.raises(StoreError):
        load_plan_review_state(tmp_path, "parent-corrupt")
    with pytest.raises(StoreError):
        plan_record.load_plan_record(tmp_path, "parent-corrupt")
    with pytest.raises(plan_record.PlanRecordInvalid, match="PLAN_RECORD_UNAVAILABLE"):
        plan_record._update_plan_record(tmp_path, "parent-corrupt", lambda record: record)

    unreadable["on"] = False
    assert result_signature(tmp_path, "parent-corrupt") == before


def test_plan_record_read_during_a_held_writer_sees_one_committed_state_and_writes_nothing(tmp_path):
    """A reader that overlaps an open record transaction gets the committed record before or
    after it, never a half-written one, and the read itself leaves no write behind."""
    from concurrent.futures import ThreadPoolExecutor
    import threading
    from ouroboros import plan_record, task_results

    task_results.write_task_result(tmp_path, "plan", "running", result="running")
    plan_record.append_revision(tmp_path, "plan", plan_record.compose_revision(
        None, goal="g", plan="first", done_when=[], affected_paths=[]), body=False)
    before = result_signature(tmp_path, "plan")
    writing, release = threading.Event(), threading.Event()

    def held_update(record):
        writing.set()
        assert release.wait(5), "test did not release the record writer"
        return {**record, "revisions": [*record["revisions"], {**record["revisions"][0], "n": 2, "plan": "second"}]}

    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="plan-writer") as writers, \
            ThreadPoolExecutor(max_workers=1, thread_name_prefix="plan-reader") as readers:
        writer = writers.submit(plan_record._update_plan_record, tmp_path, "plan", held_update)
        try:
            assert writing.wait(3)
            reader = readers.submit(plan_record.load_plan_record, tmp_path, "plan")
            release.set()  # the reader may be waiting on the store's write lock: let the writer finish
            seen = reader.result(timeout=15)
        finally:
            release.set()
        writer.result(timeout=15)
    assert len(seen["revisions"]) in (1, 2)  # one committed record, whichever side of the writer
    assert len(plan_record.load_plan_record(tmp_path, "plan")["revisions"]) == 2
    after = result_signature(tmp_path, "plan")
    assert after != before and after[0] == before[0] + 1  # exactly the writer's one write


def test_missing_plan_state_read_does_not_create_a_store(tmp_path):
    from ouroboros.task_results import load_plan_review_state

    state = load_plan_review_state(tmp_path, "absent")
    assert state["waves"] == [] and state["cycles_paid"] == 0
    assert not (tmp_path / "task_results").exists()


class TestPlanReviewChecklist(unittest.TestCase):
    def test_checklist_section_exists_and_non_empty(self):
        """Plan Review Checklist section must exist in CHECKLISTS.md (consumed by heading)."""
        from ouroboros.tools.review_helpers import load_checklist_section

        checklist = load_checklist_section("Plan Review Checklist")
        self.assertIsInstance(checklist, str)
        self.assertGreater(len(checklist), 100)
        self.assertTrue(checklist.startswith("## Plan Review Checklist"))


class TestPlanReviewToolRegistration(unittest.TestCase):
    def test_plan_task_schema_is_the_plan_records_form(self):
        tool = _plan_task_entry()
        params = tool.schema["parameters"]["properties"]
        self.assertEqual(set(params), {"goal", "plan", "done_when", "affected_paths", "ask", "message",
                                       "attach", "reviewer_effort", "reason"})
        # Three effects, one tool: nothing is schema-required (the handler reads the effect).
        self.assertEqual(tool.schema["parameters"]["required"], [])

    def test_public_registry_rejects_unknown_plan_task_arguments(self):
        import tempfile

        from ouroboros.tools.registry import ToolRegistry

        with tempfile.TemporaryDirectory() as raw_root:
            root = pathlib.Path(raw_root)
            registry = ToolRegistry(repo_dir=root, drive_root=root)
            registry.override_handler(
                "plan_task",
                lambda _ctx, **_params: "handler-ran",
            )
            result = registry.execute(
                "plan_task",
                {
                    "plan": "Do the work",
                    "goal": "Ship it",
                    "review_dispositon": {},
                },
            )

        self.assertIn("TOOL_ARG_ERROR (plan_task)", result)
        for field in ("done_when", "affected_paths", "ask", "reviewer_effort"):  # the new form is advertised
            self.assertIn(field, result)
        self.assertNotIn("review_disposition", result)  # retired names are hidden, never advertised
        self.assertIn("unsupported argument(s): review_dispositon", result)
        self.assertNotIn("handler-ran", result)

    def test_plan_task_description_is_domain_neutral_and_trigger_free(self):
        desc = _plan_task_entry().schema["description"]
        lower = desc.lower()
        self.assertNotIn("before code", lower)
        self.assertNotIn("planning-scout", lower)
        self.assertNotIn("context level", lower)
        # No numeric trigger and no "say why you skipped" pressure (BIBLE P5).
        self.assertNotIn(">2 files", desc)
        self.assertNotIn("50 lines", desc)
        self.assertNotIn("skip", lower)
        for word in ("research", "deliverable", "action in the world"):
            self.assertIn(word, lower)
        # It promises no host bubble for an open objection.
        self.assertNotIn("the host discloses", desc)

    def test_plan_task_contract_has_no_swarm_knobs(self):
        from ouroboros.config import RETIRED_SETTING_KEYS, SETTINGS_DEFAULTS

        description = _plan_task_entry().schema["description"].lower()
        self.assertNotIn("heartbeat", description)
        self.assertNotIn("swarm", description)
        for key in (
            "OUROBOROS_PLAN_TASK_SWARM_TIMEOUT_SEC",
            "OUROBOROS_PLAN_TASK_SWARM_MAX_WAIT_SEC",
            "OUROBOROS_PLAN_TASK_SWARM_HEARTBEAT_STALE_SEC",
        ):
            self.assertNotIn(key, SETTINGS_DEFAULTS)
            self.assertIn(key, RETIRED_SETTING_KEYS)

    def test_duplicate_plan_calls_use_existing_sequential_tool_lane(self):
        from ouroboros.loop_tool_execution import tool_calls_can_run_parallel

        calls = [
            {"function": {"name": "plan_task", "arguments": "{}"}},
            {"function": {"name": "plan_task", "arguments": "{}"}},
        ]
        self.assertFalse(tool_calls_can_run_parallel(calls))


class TestPlanBudgetFollowsQuorum(unittest.TestCase):
    """The shared-prompt budget is the one a review QUORUM can read, not the global minimum
    (``review_synthesis.quorum_input_token_limit``, still consumed by tools/review.py)."""

    def setUp(self):
        from ouroboros.tools.review_synthesis import quorum_input_token_limit

        self.limit = quorum_input_token_limit
        self.models = ["small/545", "big/a", "big/b"]
        self.limits = {"small/545": 545_000, "big/a": 745_000, "big/b": 745_000}

    def test_reviewer_example_keeps_the_context_two_large_slots_can_read(self):
        self.assertEqual(self.limit(self.models, self.limits), 745_000)

    def test_unavailable_or_uncalibrated_slots_cannot_justify_a_bigger_prompt(self):
        # Only one slot has a cap at assembly time: the other two read 0, so the quorum-th
        # largest cap is 0 and nothing is assembled against a window nobody has.
        self.assertEqual(self.limit(self.models, {"big/a": 745_000}), 0)
        self.assertEqual(self.limit([], {}), 0)

    def test_small_configs_need_every_slot_so_the_budget_is_the_minimum(self):
        # adaptive_quorum: 1 slot -> 1, 2 slots -> both, 3+ -> 2 of N.
        self.assertEqual(self.limit(["a", "b"], {"a": 545_000, "b": 745_000}), 545_000)
        self.assertEqual(self.limit(["a"], {"a": 545_000}), 545_000)


class TestPlanRowTypedFacts(unittest.TestCase):
    """B1: typed lane-failure facts survive the substrate boundary into the plan row
    and its wave-record projection — never re-derived from the error prose."""

    def test_plan_row_carries_typed_failure_facts(self):
        from ouroboros.tools.plan_review_runtime import _plan_row_from_actor, plan_row_typed_facts

        actor = {"slot_id": "s1", "model": "m", "status": "error",
                 "error": "delegated review session run-1 ended failed",
                 "failure_code": "subscription_window_exhausted",
                 "reset_at": "2030-01-01T00:00:00Z", "http_status": 429,
                 "transport_status": "provider_transport_error",
                 "usage": {"capability_delta": [{"reason": "reduced"}]}}
        row = _plan_row_from_actor(actor, None)
        self.assertEqual(row["failure_code"], "subscription_window_exhausted")
        self.assertEqual(row["reset_at"], "2030-01-01T00:00:00Z")
        self.assertEqual(row["http_status"], 429)
        self.assertEqual(row["transport_status"], "provider_transport_error")
        self.assertEqual(row["capability_delta"], [{"reason": "reduced"}])
        self.assertEqual(plan_row_typed_facts(row), {
            "failure_code": "subscription_window_exhausted",
            "reset_at": "2030-01-01T00:00:00Z", "http_status": 429,
            "transport_status": "provider_transport_error", "reported_cause": "", "failure_details": {},
            "capability_delta": [{"reason": "reduced"}],
        })
        # A recorded structured refusal is carried as the same dict, never flattened to prose.
        details = {"code": "native_transcript_cap_exceeded", "measured": 940000, "limit": 900000}
        refused = _plan_row_from_actor({**actor, "failure_details": details}, None)
        self.assertEqual(plan_row_typed_facts(refused)["failure_details"], details)

    def test_a_pre_typed_row_defaults_to_honest_absence(self):
        """Fail-open: an actor from an engine that carries no typed facts (code:null
        world) produces the exact empty defaults — nothing invented, nothing broken."""
        from ouroboros.tools.plan_review_runtime import _plan_row_from_actor, plan_row_typed_facts

        row = _plan_row_from_actor({"slot_id": "s2", "status": "error", "error": "boom"}, None)
        self.assertEqual((row["failure_code"], row["reset_at"], row["http_status"]),
                         ("", "", None))
        self.assertEqual((row["transport_status"], row["capability_delta"]), ("", []))
        facts = plan_row_typed_facts({"slot_id": "s2"})  # e.g. a preflight_oversize row
        self.assertEqual((facts["failure_code"], facts["failure_details"]), ("", {}))
        self.assertEqual(facts["capability_delta"], [])



def test_retired_swarm_keys_are_dropped_on_settings_load(tmp_path, monkeypatch):
    from ouroboros import config

    path = tmp_path / "settings.json"
    path.write_text(json.dumps({
        "OUROBOROS_PLAN_TASK_SWARM_TIMEOUT_SEC": 5,
        "OUROBOROS_PLAN_TASK_SWARM_MAX_WAIT_SEC": 50,
        "OUROBOROS_PLAN_TASK_SWARM_HEARTBEAT_STALE_SEC": 121,
        "OUROBOROS_REVIEW_MAX_CYCLES": "3",
    }), encoding="utf-8")
    monkeypatch.setattr(config, "SETTINGS_PATH", path)
    loaded = config.load_settings()
    for key in ("OUROBOROS_PLAN_TASK_SWARM_TIMEOUT_SEC", "OUROBOROS_PLAN_TASK_SWARM_MAX_WAIT_SEC",
                "OUROBOROS_PLAN_TASK_SWARM_HEARTBEAT_STALE_SEC"):
        assert key not in loaded and key not in config.SETTINGS_DEFAULTS
        assert key in config.RETIRED_SETTING_KEYS
    assert loaded["OUROBOROS_REVIEW_MAX_CYCLES"] == "3"


if __name__ == "__main__":
    unittest.main()
