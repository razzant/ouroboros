"""Shared money stays pooled while existing pacing reports measured facts."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from datetime import datetime, timedelta, timezone
import json
import threading
from types import SimpleNamespace

import pytest

from ouroboros import task_pacing as pacing, usage_accounting as accounting
from tests._usage_store_testing import ledger_rows


@pytest.fixture
def root(tmp_path, monkeypatch):
    monkeypatch.setenv("TOTAL_BUDGET", "100")
    monkeypatch.setenv("OUROBOROS_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("OUROBOROS_SETTINGS_PATH", str(tmp_path / "settings.json"))
    monkeypatch.setattr("ouroboros.config.SETTINGS_PATH", tmp_path / "settings.json")
    monkeypatch.setattr("ouroboros.pricing._fetch_live_rows", lambda *_a, **_kw: pytest.fail("unexpected pricing lookup"))
    return tmp_path


def request(root, **kwargs):
    return accounting.AttemptRequest(**{
        "model": "fixture", "provider": "openai", "drive_root": root,
        "reservation_usd": 1, **kwargs,
    })


def rows(root):
    return ledger_rows(root)


def spend(root, amount, **kwargs):
    """KNOWN spend: what every money limit decides on (#1487)."""
    held = accounting.reserve_attempt(request(root, reservation_usd=amount, **kwargs))
    accounting.mark_dispatched(held)
    accounting.settle_attempt(held, {}, cost_usd=amount, cost_final=True)
    return held


def test_fallback_limit_is_the_value_applied_and_survives_settlement(root, monkeypatch):
    reservation = accounting.reserve_attempt(request(root))
    monkeypatch.setenv("TOTAL_BUDGET", "500")
    accounting.mark_dispatched(reservation)
    accounting.settle_attempt(reservation, {}, cost_usd=0.2, cost_final=True)
    assert [(row["state"], row["revision"]) for row in rows(root)] == [("settled", 3)]
    for row in rows(root):
        assert row["global_limit_usd"] == 100
        assert row["global_limit_source"] == "settings_budget_resolver"
        assert row["global_limit_revision"] is None
        assert row["global_limit_unbounded"] is False


def test_request_override_does_not_inherit_another_limits_provenance(root):
    scope = accounting.UsageScope(drive_root=root, global_limit_usd=80,
        global_limit_source="task_start_budget_resolver", global_limit_revision="known-revision")
    with accounting.usage_scope(scope):
        inherited = accounting.reserve_attempt(request(root))
        overridden = accounting.reserve_attempt(request(root, global_limit_usd=60))
        explicit = accounting.reserve_attempt(request(root, global_limit_usd=70,
            global_limit_source="caller_snapshot", global_limit_revision="caller-revision"))
    by_id = {row["attempt_id"]: row for row in rows(root)}
    assert (by_id[inherited.attempt_id]["global_limit_usd"],
            by_id[inherited.attempt_id]["global_limit_source"],
            by_id[inherited.attempt_id]["global_limit_revision"]) == (80, "task_start_budget_resolver", "known-revision")
    assert by_id[overridden.attempt_id]["global_limit_source"] == "attempt_request"
    assert by_id[overridden.attempt_id]["global_limit_revision"] is None
    assert by_id[explicit.attempt_id]["global_limit_revision"] == "caller-revision"


def _run_task_scoped(root, monkeypatch, scoped):
    from ouroboros.agent import OuroborosAgent

    monkeypatch.setattr("ouroboros.subagent_runtime.apply_task_start_settings_or_disclose", lambda *_: None)
    monkeypatch.setattr("ouroboros.model_wait.task_model_wait_scope", lambda **_: nullcontext())
    host = SimpleNamespace(env=SimpleNamespace(drive_root=root), _emit_live_log=lambda *_: None,
                           _event_queue=None, _handle_task_scoped=scoped)
    return OuroborosAgent.handle_task(host, {"id": "a", "type": "task"})


def test_task_start_captures_no_global_limit_so_the_fence_resolves_the_current_one(root, monkeypatch):
    scope = _run_task_scoped(root, monkeypatch, lambda _: accounting.current_usage_scope())
    assert scope.global_limit_usd is None
    assert (scope.global_limit_source, scope.global_limit_revision) == ("", None)


def test_a_running_task_follows_the_budget_the_owner_saves_mid_run(root, monkeypatch):
    """The worker's environment keeps the budget of the task's first minute; the saved document does not."""
    from ouroboros import config

    settings = root / "settings.json"
    monkeypatch.setattr(config, "SETTINGS_PATH", settings)
    settings.write_text(json.dumps({"TOTAL_BUDGET": 1.5}))
    monkeypatch.setenv("TOTAL_BUDGET", "1.5")

    def scoped(_task):
        spend(root, 1.5)  # known spend reaches the $1.50 the task started under
        with pytest.raises(accounting.BudgetExceeded):
            accounting.reserve_attempt(request(root))
        settings.write_text(json.dumps({"TOTAL_BUDGET": 10}))   # the owner tops the budget up; this process's env is untouched
        raised = accounting.reserve_attempt(request(root))
        settings.write_text(json.dumps({"TOTAL_BUDGET": 1}))    # ...and lowering it bites the same way
        with pytest.raises(accounting.BudgetExceeded):
            accounting.reserve_attempt(request(root))
        return raised

    raised = _run_task_scoped(root, monkeypatch, scoped)
    applied = {row["attempt_id"]: row for row in rows(root)}[raised.attempt_id]
    assert (applied["global_limit_usd"], applied["global_limit_source"], applied["global_limit_revision"]) == (
        10, "settings_budget_resolver", None)


def test_the_wrapup_wallet_observation_follows_the_saved_budget_too(root, monkeypatch):
    """The last-fit rail read the same frozen number: $21 'left' under a limit the owner had already raised."""
    from ouroboros import config
    from ouroboros.loop_budget import _wrapup_global_remaining

    settings = root / "settings.json"
    monkeypatch.setattr(config, "SETTINGS_PATH", settings)
    settings.write_text(json.dumps({"TOTAL_BUDGET": 100}))

    def scoped(_task):
        spend(root, 90)
        before = _wrapup_global_remaining()
        settings.write_text(json.dumps({"TOTAL_BUDGET": 500}))
        return before, _wrapup_global_remaining()

    assert _run_task_scoped(root, monkeypatch, scoped) == (10, 410)


def test_the_saved_document_answers_first_and_the_environment_only_when_it_cannot(root, monkeypatch):
    from ouroboros import config, settings_integrity
    from ouroboros.settings_setup_contract import resolve_total_budget_usd

    settings = root / "settings.json"
    monkeypatch.setattr(config, "SETTINGS_PATH", settings)
    assert resolve_total_budget_usd() == 100            # no document: the environment (fixture) answers
    settings.write_text(json.dumps({"OTHER": 1}))
    assert resolve_total_budget_usd() == 100            # a document without the key is silence, not a decision
    settings.write_text(json.dumps({"TOTAL_BUDGET": 250.5}))
    assert resolve_total_budget_usd() == 250.5
    settings.write_text(json.dumps({"TOTAL_BUDGET": 0}))
    assert resolve_total_budget_usd() is None           # a saved non-positive value stays "no finite budget"

    reads = []
    real_read = settings_integrity.read_settings_json_verified
    monkeypatch.setattr(settings_integrity, "read_settings_json_verified",
                        lambda path: reads.append(path) or real_read(path))
    settings.write_text(json.dumps({"TOTAL_BUDGET": 75}))
    assert [resolve_total_budget_usd() for _ in range(3)] == [75, 75, 75]
    assert len(reads) == 1, "an unchanged file costs one stat, never another parse"


def test_a_refused_or_failed_document_read_is_never_remembered(root, monkeypatch):
    """Under a benchmark pin a changed snapshot RAISES; the money path must answer, and must retry."""
    from ouroboros import config, settings_integrity
    from ouroboros.settings_setup_contract import resolve_total_budget_usd

    settings = root / "settings.json"
    monkeypatch.setattr(config, "SETTINGS_PATH", settings)
    settings.write_text(json.dumps({"TOTAL_BUDGET": 40}))
    real_read = settings_integrity.read_settings_json_verified
    refusing = {"on": True}

    def read(path):
        if refusing["on"]:
            raise settings_integrity.SettingsIntegrityError("settings snapshot changed")
        return real_read(path)

    monkeypatch.setattr(settings_integrity, "read_settings_json_verified", read)
    assert resolve_total_budget_usd() == 100            # the environment: the last verified projection
    refusing["on"] = False
    assert resolve_total_budget_usd() == 40             # same file, same stamp: the failure was not cached


def test_an_unpinned_failed_document_read_answers_none_and_is_never_remembered(root, monkeypatch):
    """Without a pin the verified reader returns None (absent/unreadable/undecodable) instead of raising."""
    from ouroboros import config, settings_integrity
    from ouroboros.settings_setup_contract import resolve_total_budget_usd

    settings = root / "settings.json"
    monkeypatch.setattr(config, "SETTINGS_PATH", settings)
    monkeypatch.delenv("OUROBOROS_SETTINGS_SHA256", raising=False)   # unpinned: the reader answers None, never raises
    settings.write_text(json.dumps({"TOTAL_BUDGET": 40}))
    real_read = settings_integrity.read_settings_json_verified
    failing = {"on": True}

    def read(path):
        return None if failing["on"] else real_read(path)

    monkeypatch.setattr(settings_integrity, "read_settings_json_verified", read)
    assert resolve_total_budget_usd() == 100            # the environment answers the transient failure
    failing["on"] = False
    assert resolve_total_budget_usd() == 40             # same stamp: None was not memoized as "no budget saved"


def test_unbounded_fallback_is_explicit_without_nonfinite_json(root, monkeypatch):
    monkeypatch.setenv("TOTAL_BUDGET", "0")
    accounting.reserve_attempt(request(root))
    row = rows(root)[0]
    assert row["global_limit_usd"] is None and row["global_limit_unbounded"] is True
    assert row["global_limit_source"] == "settings_budget_resolver"
    json.dumps(row, allow_nan=False)


def test_soft_ceilings_reserve_nothing_and_concurrent_sends_still_share_one_pool(root, monkeypatch):
    monkeypatch.setenv("TOTAL_BUDGET", "10")
    assert pacing.resolve_cost_ceiling(10, {}).state == pacing.COST_CEILING_DISABLED  # ordinary: no early stop
    ceilings = [pacing.resolve_cost_ceiling(10, {"cost_hard_stop_pct": 50}) for _ in range(2)]
    assert [ceiling.ceiling_usd for ceiling in ceilings] == [5, 5]
    assert not ledger_rows(root)
    assert all(pacing.cost_ceiling_disclosure(c)["allocation"] == "unreserved_shared_pool" for c in ceilings)
    barrier = threading.Barrier(2)

    def send(task_id):
        barrier.wait()
        try:
            return accounting.reserve_attempt(accounting.AttemptRequest(
                model="fixture", provider="openai", drive_root=root,
                reservation_usd=6, task_id=task_id, root_task_id=task_id))
        except accounting.BudgetExceeded as error:
            return error

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(send, ("a", "b")))
    # Owner Q4-A: both concurrent sends are admitted on $0 known spend — the pool's
    # overshoot is accepted and disclosed as exposure, never prevented by holds.
    assert all(isinstance(result, accounting.AttemptReservation) for result in results)
    assert accounting.usage_projection(root)["accounted_usd"] == 12
    for result in results:
        accounting.mark_dispatched(result)
        accounting.settle_attempt(result, {}, cost_usd=6, cost_final=True)
    assert accounting.usage_projection(root)["settled_usd"] == 12  # a recorded overrun of $2
    with pytest.raises(accounting.BudgetExceeded):
        accounting.reserve_attempt(accounting.AttemptRequest(
            model="fixture", provider="openai", drive_root=root, reservation_usd=0.01,
            task_id="c", root_task_id="c"))


def delivered_tool(name, ident, *, duration=None, error=False):
    return {"fn_name": name, "tool_call_id": ident, "is_error": error,
            "result": "sleep 999; polling; duration_ms=12345", "tool_args": {},
            "args_for_log": {}, "result_meta": {"duration_ms": duration}}


@pytest.mark.parametrize("mode", ["intrinsic", "deadline", "cost"])
def test_real_note_transport_keeps_counts_measured_time_and_open_spend(root, monkeypatch, mode):
    from ouroboros import loop, loop_tool_execution

    now = datetime.now(timezone.utc)
    metadata = {"started_at": (now - timedelta(seconds=700)).isoformat()}
    if mode == "deadline":
        metadata["deadline_at"] = (now + timedelta(seconds=300)).isoformat()
    usage = {"cost": 1.25}
    ctx = SimpleNamespace(task_id="a", drive_root=root, task_metadata=metadata,
                          _accumulated_usage=usage, task_contract={})
    tools = SimpleNamespace(_ctx=ctx)
    trace, messages, checkpoints = {"tool_calls": []}, [], []
    loop_tool_execution.process_tool_results([
        delivered_tool("run_command", "one", duration=60000),
        delivered_tool("run_command", "two"),
        delivered_tool("browser_action", "three", duration=0, error=True),
    ], messages, trace, lambda *_: None, tools=tools)
    monkeypatch.setattr(loop, "_emit_checkpoint_event", lambda _q, _id, _logs, payload: checkpoints.append(payload))
    monkeypatch.setattr(loop, "_loop_tree_accounting", lambda **_: None)
    monkeypatch.setattr(pacing, "get_pacing_interval_sec", lambda: 600)
    def inject():
        if mode == "cost":
            return loop._maybe_inject_cost_budget_milestone(messages, tools,
                budget_remaining_usd=100, accumulated_usage=usage,
                cost_ceiling=pacing.CostCeiling(state="active", ceiling_usd=2))
        return loop._maybe_inject_time_budget_milestone(messages, tools, round_idx=7, accumulated_usage=usage)

    scope = accounting.UsageScope(drive_root=root, task_id="a", root_task_id="a", global_limit_usd=100)
    with accounting.usage_scope(scope):
        own = accounting.reserve_attempt(request(root))
        accounting.mark_dispatched(own)
        accounting.settle_attempt(own, {}, cost_usd=1.25, cost_final=True)
        pending = accounting.reserve_attempt(request(root, reservation_usd=2))
        accounting.mark_dispatched(pending)
        accounting.mark_unresolved(pending, "outcome unknown")
        accounting.record_subscription_session("free", route="fixture", drive_root=root,
            task_id="child", root_task_id="a", spend_usd=0, spend_estimated=False)
        accounting.record_subscription_session("unknown", route="fixture", drive_root=root,
            task_id="child", root_task_id="a", spend_usd=None)
        assert inject()
        facts = checkpoints[-1]["resource_facts"]
        measured = {row["tool"]: row for row in facts["tools"]["by_tool"]}
        assert facts["tools"]["calls"] == 3
        assert measured["run_command"]["calls"] == 2
        assert measured["run_command"]["reported_duration_ms"] == 60000
        assert measured["run_command"]["duration_observations"] == 1
        assert measured["browser_action"]["reported_duration_ms"] == 0
        assert measured["browser_action"]["error_calls"] == 1
        spend = facts["spend"]
        assert spend["own_task"]["accounted_usd"] == 3.25
        assert spend["tree"]["cost_final"] is False
        assert spend["delegated_tree"]["subscription_sessions"] == 2
        assert spend["delegated_tree"]["unknown_unmetered"] == 1
        assert spend["delegated_tree"]["cost_final"] is False
        # Room above KNOWN spend ($1.25); the $2 unresolved bound rides as exposure (#1487).
        assert spend["global"]["remaining_known_usd"] == 98.75
        assert spend["global"]["settled_usd"] == 1.25 and spend["global"]["accounted_usd"] == 3.25
        assert spend["global"]["allocation"] == "unreserved_shared_pool"
        rendered = messages[-1]["content"].split(
            "Observed resource facts (known spend is settled_usd; accounted money adds open holds):\n")[1]
        assert json.loads(rendered) == facts
        monkeypatch.setattr(accounting, "usage_breakdown", lambda *_a, **_kw: pytest.fail("read before next milestone"))
        assert not inject()


def test_unknown_duration_and_unavailable_ledger_do_not_erase_the_note(root, monkeypatch):
    usage = {}
    ctx = SimpleNamespace(_accumulated_usage=usage, task_id="a", drive_root=root)
    for index, duration in enumerate((None, "12", True, -1, float("nan"))):
        pacing.record_tool_activity(ctx, delivered_tool("run_command", str(index), duration=duration))
    monkeypatch.setattr(accounting, "usage_breakdown", lambda *_a, **_kw: (_ for _ in ()).throw(OSError("unreadable")))
    with accounting.usage_scope(accounting.UsageScope(drive_root=root, task_id="a", root_task_id="a")):
        note = pacing.with_resource_facts(pacing.PacingNote("keep working", {"checkpoint_kind": "intrinsic_pacing"}), ctx, usage)
    facts = note.checkpoint["resource_facts"]
    assert note.text.startswith("keep working")
    assert facts["tools"]["by_tool"][0]["duration_observations"] == 0
    assert facts["tools"]["by_tool"][0]["reported_duration_ms"] is None
    assert facts["spend"]["status"] == "unavailable"
    assert "tree" not in facts["spend"]


def test_bounded_tool_view_discloses_omitted_calls_and_stays_independent_of_transcript(root):
    usage = {}
    ctx = SimpleNamespace(_accumulated_usage=usage, task_id="a", drive_root=root)
    for index in range(12):
        pacing.record_tool_activity(ctx, delivered_tool(f"tool_{index:02}", str(index)))
    note = pacing.with_resource_facts(pacing.PacingNote("note", {}), ctx, usage)
    tools = note.checkpoint["resource_facts"]["tools"]
    assert tools["calls"] == 12 and len(tools["by_tool"]) == 10
    assert tools["omitted_calls"] == tools["omitted_tools"] == 2
    assert tools["source_path"] == str(root / "logs" / "tools.jsonl")
    assert "not reconstructed" in tools["coverage"]


def test_review_scope_preserves_limit_provenance(tmp_path, monkeypatch):
    from ouroboros import usage_accounting as ua
    from ouroboros.review_substrate import ReviewCoordinator
    from ouroboros import review_custody
    monkeypatch.setenv('TOTAL_BUDGET', '999')
    class Captured(Exception):
        pass
    observed = {}
    def intercept(**kwargs):
        scoped = kwargs['review_usage_scope']
        with ua.usage_scope(scoped):
            reservation = ua.reserve_attempt(ua.AttemptRequest(model='fixture', provider='openai', reservation_usd=1))
        row = ledger_rows(tmp_path)[-1]
        observed.update(row)
        ua.release_attempt(reservation)
        raise Captured
    monkeypatch.setattr(review_custody, 'run_custodied_review_slots', intercept)
    parent = ua.UsageScope(drive_root=tmp_path, task_id='task', root_task_id='task', global_limit_usd=80,
                          global_limit_source='task_start_budget_resolver', global_limit_revision='revision-42')
    request = SimpleNamespace(reconcile_only=False, surface='plan_review', task_id='task',
                              usage_attribution={}, deadline_at='', task_attempt=None)
    with ua.usage_scope(parent), pytest.raises(Captured):
        ReviewCoordinator(llm=object(), drive_root=tmp_path).run(request, [object()])
    assert observed['global_limit_usd'] == 80
    assert (observed['global_limit_source'], observed['global_limit_revision']) == ('task_start_budget_resolver', 'revision-42')
