"""Owner decisions 1A/2A (#1407/#1409): configured routes before any wait, one recovery for
direct and queued turns, typed unknown eligibility, and model-owned return to the primary.

Real round loop, round dispatcher and physical-attempt ledger; only each model's socket is
scripted, so the ledger proves which physical attempts existed and under which identity.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import httpx
import pytest

import ouroboros.loop as loop_mod
import ouroboros.loop_transport as loop_transport
from ouroboros import owner_mailbox
from ouroboros import usage_accounting as ua
from ouroboros.loop import run_llm_loop
from ouroboros.loop_model_call import _route_candidates, _route_facts_text
from ouroboros.model_slots import MODEL_ACCOUNTS_KEY
from ouroboros.tools.control_runtime import _switch_model
from ouroboros.tools.registry import ToolRegistry
from supervisor.owner_stop import REASON_OWNER_STOPPED_DIRECT_TURN
from tests.test_llm_claudexor import ROUTE, result
from tests.test_llm_claudexor import setup as gateway_fixture
from tests.test_transport_death_retry import _events, _ledger
from tests._usage_store_testing import dispatched_attempts

setup = gateway_fixture
PRIMARY = "primary/model"


def _death():
    return httpx.ReadError("socket died after dispatch")


def _unreadable():
    """An accepted operation that could not be READ: it may still finish."""
    error = RuntimeError("model control link lost")
    error.same_operation_recoverable = True
    return error


class _RouteLLM:
    """Every chat() is a REAL physical attempt; its send follows the script of the model it targets."""

    def __init__(self, root, **scripts):
        self.root, self.sent = root, []
        self.scripts = {model: list(steps) for model, steps in scripts.items()}

    def default_model(self):
        return PRIMARY

    def chat(self, **kwargs):
        model = kwargs["model"]
        self.sent.append((model, [dict(row) for row in kwargs["messages"]]))
        steps = self.scripts.get(model) or []
        step = steps.pop(0) if steps else None

        def send():
            if step is not None:
                error = step()
                if isinstance(error, httpx.HTTPError):
                    raise RuntimeError("Connection error.") from error
                raise error
            return {"content": "ok"}

        request = ua.AttemptRequest(model=model, provider="openrouter", reservation_usd=1.0, drive_root=self.root,
                                    task_id="t-1a", root_task_id="t-1a", source="test.unknown_fallback_first")
        ua.execute_physical_attempt(request, send, extractor=lambda _resp: ({"prompt_tokens": 1, "completion_tokens": 1}, 0.01, True))
        return {"role": "assistant", "content": f"answer from {model}"}, {"prompt_tokens": 1, "completion_tokens": 1}


@pytest.fixture
def data_root(tmp_path, monkeypatch):
    root = tmp_path / "data"
    monkeypatch.setenv("OUROBOROS_DATA_DIR", str(root))
    monkeypatch.setenv("OUROBOROS_SETTINGS_PATH", str(root / "settings.json"))
    monkeypatch.setenv("TOTAL_BUDGET", "100")
    monkeypatch.setenv("OUROBOROS_TASK_REVIEW_MODE", "off")
    monkeypatch.delenv("USE_LOCAL_FALLBACK", raising=False)
    monkeypatch.delenv(MODEL_ACCOUNTS_KEY, raising=False)
    monkeypatch.setattr(loop_mod, "_rebind_context_fit_plan", lambda plan, *_a, **_kw: (plan, "max"))
    from ouroboros import fallback_cooldown
    monkeypatch.setattr(fallback_cooldown, "is_cooling_down", lambda *_a: False)
    (root / "state").mkdir(parents=True)
    return root


def _run(tmp_path, llm, *, direct=False, notes=None):
    registry = ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path)
    registry._ctx.is_direct_chat = direct
    notes = [] if notes is None else notes
    with ua.usage_scope(ua.UsageScope(drive_root=llm.root, task_id="t-1a", root_task_id="t-1a", global_limit_usd=100)):
        text, usage, trace = run_llm_loop(
            messages=[{"role": "user", "content": "go"}], tools=registry, llm=llm, drive_logs=tmp_path,
            emit_progress=lambda text, *, incident=None, **_kw: notes.append(text), incoming_messages=__import__("queue").Queue(),
            task_id="t-1a", drive_root=tmp_path)
    return text, usage, trace, registry


def _notices(messages):
    return [row["content"] for row in messages if "[Transport recovery]" in str(row.get("content"))]


# -- 1A: a configured route answers first, for direct and queued turns alike --------------------


@pytest.mark.parametrize("direct", [False, True])
def test_unknown_outcome_tries_the_configured_route_first_with_a_new_identity(data_root, tmp_path, monkeypatch, direct):
    """No legacy paid primary repeat runs ahead of the fallback (the direct turn's former rail),
    no upstream probe or wait precedes it, and the old attempt keeps its own row and cost."""
    monkeypatch.setenv("OUROBOROS_MODEL_FALLBACKS", "fb/one")
    monkeypatch.setattr(loop_transport, "upstream_transport_reachable",
                        lambda *_a, **_kw: pytest.fail("a configured route answers before any wait"))
    llm = _RouteLLM(data_root, **{PRIMARY: [_death]})
    text, usage, _trace, _registry = _run(tmp_path, llm, direct=direct)

    assert text == "answer from fb/one"
    assert [model for model, _messages in llm.sent] == [PRIMARY, "fb/one"]
    rows = _ledger(data_root)
    assert [(row["state"], row["revision"]) for row in rows] == [("unresolved", 4), ("settled", 3)]
    assert rows[0]["physical_failure"]
    old, new = rows[0]["attempt_id"], rows[1]["attempt_id"]
    assert old != new
    assert ua.usage_projection(data_root)["unresolved_upper_bound_usd"] == 1.0
    notices = _notices(llm.sent[-1][1])
    assert len(notices) == 1 and old in notices[0] and "fb/one" in notices[0]
    assert usage["transport_recovery"]["previous_attempt"]["physical_attempt_id"] == old
    assert _events(tmp_path, "network_wait") == []


def test_repeated_unknowns_move_on_through_the_configured_routes(data_root, tmp_path, monkeypatch):
    """An eligible unknown on a candidate moves on too; each new generation names the attempt it follows."""
    monkeypatch.setenv("OUROBOROS_MODEL_FALLBACKS", "fb/one,fb/two")
    llm = _RouteLLM(data_root, **{PRIMARY: [_death], "fb/one": [_death]})
    text, _usage, _trace, _registry = _run(tmp_path, llm, direct=True)

    assert text == "answer from fb/two"
    assert [model for model, _messages in llm.sent] == [PRIMARY, "fb/one", "fb/two"]
    first, second, _third = (row["attempt_id"] for row in dispatched_attempts(data_root))
    notices = _notices(llm.sent[-1][1])
    assert len(notices) == 2 and first in notices[0] and second in notices[1]


def test_fallback_can_recover_during_an_existing_primary_outage(data_root, tmp_path, monkeypatch):
    monkeypatch.setenv("OUROBOROS_MODEL_FALLBACKS", "fb/one")
    probes, sleeps = [], []
    monkeypatch.setattr(loop_transport, "upstream_transport_reachable",
                        lambda *_a, **kw: probes.append(kw) or {})
    monkeypatch.setattr(loop_transport, "interruptible_wait_sleep",
                        lambda seconds, _wake: sleeps.append(seconds) or False)
    llm = _RouteLLM(data_root, **{PRIMARY: [_death], "fb/one": [_death]})
    text, usage, _trace, _ctx = _run(tmp_path, llm, direct=True)
    assert text == "answer from fb/one"
    assert [model for model, _ in llm.sent] == [PRIMARY, "fb/one", "fb/one"]
    assert len(probes) == len(sleeps) == 1  # the existing episode paces recovery
    assert probes[0]["expected_route"] is None  # the old failed fallback is not the primary probe's identity
    assert usage["transport_recovery"]["old_outcome"] == "unknown"
    assert [row["state"] for row in _ledger(data_root)].count("unresolved") == 2


@pytest.mark.parametrize("primary,ceiling_route_answers", [
    (PRIMARY, False), (PRIMARY, True), ("openai/gpt-5.6-luna", False),
])
def test_a_ceiling_routes_tool_fit_stays_only_where_its_notice_stays(data_root, tmp_path, monkeypatch, primary,
                                                                     ceiling_route_answers):
    """A direct-OpenAI candidate fits the shared resident list and writes its notice into its transcript. A
    cross-family candidate writes into its own copy: if it answers, both are adopted; if it fails, the copy goes
    and the fit with it, so the next route sends the whole list and reads no notice. A same-family candidate
    writes into the shared transcript, so its fit stays with its notice even when it fails."""
    from ouroboros import provider_models

    monkeypatch.setenv("OUROBOROS_MODEL_FALLBACKS", "openai::fb-direct,fb/two")
    monkeypatch.setitem(provider_models.PROVIDER_TOOL_SCHEMA_LIMITS, "openai", 100)
    llm = _RouteLLM(data_root, **{primary: [_death], "openai::fb-direct": [] if ceiling_route_answers else [_death]})
    llm.default_model = lambda: primary
    original, sent = llm.chat, []

    def chat(**kwargs):
        sent.append((kwargs["model"], len(kwargs["tools"] or []), "tool schemas in one request" in str(kwargs["messages"])))
        return original(**kwargs)

    llm.chat = chat
    text, _usage, _trace, registry = _run(tmp_path, llm, direct=True)
    (_primary, whole, _), (ceiling, fitted, told), *rest = sent
    assert ceiling == "openai::fb-direct" and whole > 100 and fitted == 100 and told
    left_out = getattr(registry._ctx, "_route_left_out_tool_names", None) or set()
    if ceiling_route_answers:
        assert text == "answer from openai::fb-direct" and not rest and len(left_out) == whole - 100
    elif primary == PRIMARY:
        assert text == "answer from fb/two" and rest == [("fb/two", whole, False)] and not left_out
    else:
        assert text == "answer from fb/two" and rest == [("fb/two", 100, True)] and len(left_out) == whole - 100


def test_non_unknown_candidate_failure_cannot_buy_a_forced_summary_after_unknown():
    from ouroboros.loop_llm_call import provider_no_call_source

    usage = {"_last_llm_error_kind": "bad_request",
             "_pending_transport_outcome": {"physical_attempt_id": "earlier-unknown"}}
    assert provider_no_call_source(usage, False)[0] == "provider_outcome_unknown_no_resend"
    usage.pop("_pending_transport_outcome")
    assert provider_no_call_source(usage, False)[0] == ""  # normal finalization remains available


def test_without_a_configured_route_a_direct_turn_waits_and_continues_inside_its_bound(data_root, tmp_path, monkeypatch):
    """The existing managed recovery, extended to a direct turn: upstream observation, then a
    NEW attempt with facts-only input; never a paid same-request repeat."""
    monkeypatch.delenv("OUROBOROS_MODEL_FALLBACKS", raising=False)
    observed = []
    monkeypatch.setattr(loop_transport, "upstream_transport_reachable",
                        lambda *_a, **_kw: observed.append(1) or {"kind": "upstream_http", "status_code": 200})
    monkeypatch.setattr(loop_transport, "interruptible_wait_sleep", lambda *_a: False)
    llm = _RouteLLM(data_root, **{PRIMARY: [_death]})
    notes = []
    text, usage, _trace, _registry = _run(tmp_path, llm, direct=True, notes=notes)

    assert text == f"answer from {PRIMARY}" and observed == [1]
    assert [model for model, _messages in llm.sent] == [PRIMARY, PRIMARY]
    assert len(_notices(llm.sent[-1][1])) == 1
    rows = _events(tmp_path, "network_wait")
    assert rows[0]["phase"] == "entered" and rows[-1]["detail"] == "new_attempt_after_unknown_outcome"
    assert all("Stop cancels" not in note for note in notes)  # an interactive episode's wording
    assert usage["transport_recovery"]["old_outcome"] == "unknown"


def test_direct_stop_during_the_unknown_wait_sends_nothing_further(data_root, tmp_path, monkeypatch):
    monkeypatch.delenv("OUROBOROS_MODEL_FALLBACKS", raising=False)
    monkeypatch.setattr(loop_transport, "upstream_transport_reachable", lambda *_a, **_kw: {})

    def stop_arrives(_seconds, _wake):
        owner_mailbox.write_owner_message(tmp_path, REASON_OWNER_STOPPED_DIRECT_TURN, "t-1a",
                                          msg_id="direct-stop", kind=owner_mailbox.KIND_FINALIZE_NOW)
        return True

    monkeypatch.setattr(loop_transport, "interruptible_wait_sleep", stop_arrives)
    llm = _RouteLLM(data_root, **{PRIMARY: [_death]})
    _text, usage, _trace, _registry = _run(tmp_path, llm, direct=True)

    assert [model for model, _messages in llm.sent] == [PRIMARY]
    assert [(row["state"], row["revision"]) for row in _ledger(data_root)] == [("unresolved", 4)]
    assert _ledger(data_root)[0]["physical_failure"]
    assert usage["_last_llm_error_kind"] == "provider_outcome_unknown"


def test_an_unreadable_accepted_operation_is_never_regenerated(data_root, tmp_path, monkeypatch):
    """Disallowed counterpart: the same operation may still finish, so neither a configured
    route nor a continuation runs; the turn ends on the honest no-resend terminal."""
    monkeypatch.setenv("OUROBOROS_MODEL_FALLBACKS", "fb/one")
    monkeypatch.setattr(loop_transport, "upstream_transport_reachable", lambda *_a, **_kw: pytest.fail("no continuation"))
    llm = _RouteLLM(data_root, **{PRIMARY: [_unreadable]})
    _text, usage, trace, _registry = _run(tmp_path, llm, direct=True)

    assert [model for model, _messages in llm.sent] == [PRIMARY]
    assert usage["_pending_transport_outcome"]["same_operation_recoverable"] is True
    assert trace["forced_finalization"]["source"] == "provider_outcome_unknown_no_resend"
    assert _events(tmp_path, "network_wait") == []


def test_an_adopted_fallback_that_fails_retains_the_primary_as_a_route(data_root, tmp_path, monkeypatch):
    """Nothing in Settings repeats Main, yet the primary binding is tried when the acting fallback fails."""
    from tests.test_completion_selection import finish

    monkeypatch.setenv("OUROBOROS_MODEL_FALLBACKS", "fb/one")
    tool_call = {"id": "call-1", "type": "function", "function": {"name": "chat_history", "arguments": "{}"}}
    replies = iter([({"role": "assistant", "content": "", "tool_calls": [tool_call]}, {"prompt_tokens": 1, "completion_tokens": 1})])
    llm = _RouteLLM(data_root, **{PRIMARY: [_death], "fb/one": [None, lambda: RuntimeError("HTTP 400 bad request")]})
    original = llm.chat
    selections = []

    def chat(**kwargs):
        if kwargs["model"] == "fb/one" and llm.scripts["fb/one"] and llm.scripts["fb/one"][0] is None:
            llm.scripts["fb/one"].pop(0)
            llm.sent.append(("fb/one", [dict(row) for row in kwargs["messages"]]))
            return next(replies)
        message, usage = original(**kwargs)
        if kwargs["model"] == PRIMARY and len(llm.sent) > 4:
            # The returned primary answer met the real handover continuation.
            # Select it explicitly instead of repeating unselected prose forever.
            assert kwargs["messages"][-1]["role"] == "user"
            assert any("No completion selection was made" in str(row.get("content"))
                       for row in kwargs["messages"])
            selections.append(PRIMARY)
            assert len(selections) == 1
            return finish(f"answer from {PRIMARY}"), usage
        return message, usage

    llm.chat = chat
    text, _usage, _trace, registry = _run(tmp_path, llm)

    sent = [model for model, _messages in llm.sent]
    # r1: primary unknown -> fb/one answers with a tool call; r2: fb/one refuses -> the primary answers
    # (a later primary round may follow: the host-driven handover's own recovery nudge).
    assert sent[:4] == [PRIMARY, "fb/one", "fb/one", PRIMARY] and set(sent[4:]) <= {PRIMARY}
    assert selections == [PRIMARY] and len(sent) == 5
    assert text == f"answer from {PRIMARY}"
    facts = [row["content"] for row in llm.sent[2][1] if "[ROUTE FACTS]" in str(row.get("content"))]
    assert len(facts) == 1 and PRIMARY in facts[0] and 'switch_model(primary="wait")' in facts[0]
    assert registry._ctx.primary_route == {"model": PRIMARY, "use_local": False, "role": "main"}


# -- binding identity -----------------------------------------------------------------------------


def test_route_candidates_compare_complete_bindings_not_model_strings(monkeypatch):
    main = "claudexor::codex=gpt-test"
    monkeypatch.setenv("OUROBOROS_MODEL_FALLBACKS", main)
    monkeypatch.delenv("USE_LOCAL_FALLBACK", raising=False)
    tool_ctx = SimpleNamespace(primary_route={"model": main, "use_local": False, "role": "main"})
    monkeypatch.setenv(MODEL_ACCOUNTS_KEY, json.dumps({"fallback": ["account-b"]}))
    assert _route_candidates(tool_ctx, main, False, "main", None) == [(main, "fallback:0", False, False)]
    monkeypatch.setenv(MODEL_ACCOUNTS_KEY, json.dumps({"fallback": [""]}))
    assert _route_candidates(tool_ctx, main, False, "main", None) == []  # the same binding is no alternative
    monkeypatch.setenv("OUROBOROS_MODEL_FALLBACKS", "")
    assert _route_candidates(tool_ctx, main, False, "main", None) == []  # an empty chain stays empty
    monkeypatch.setenv("OUROBOROS_MODEL_FALLBACKS", f"{main},other/model")
    monkeypatch.setenv(MODEL_ACCOUNTS_KEY, json.dumps({"fallback": ["account-b", ""]}))
    acting = _route_candidates(tool_ctx, main, False, "fallback:0", None)
    assert acting == [(main, "main", False, True), ("other/model", "fallback:1", False, False)]
    owner_choice = SimpleNamespace(overrides={"main": {"model_account_override": "account-b"}})
    assert _route_candidates(tool_ctx, main, False, "fallback:0", owner_choice) == [
        ("other/model", "fallback:1", False, False)]  # the owner's wait-card choice made them one binding


# -- 2A: the acting model decides whether to return ----------------------------------------------


def test_switch_model_returns_to_the_exact_primary_binding_or_declares_a_wait(tmp_path):
    ctx = SimpleNamespace(primary_route={"model": "consc/model", "use_local": False, "role": "consciousness"},
                          model_wait_context=None, active_model_override=None, active_use_local_override=None,
                          active_effort_override=None)
    assert "primary route consc/model" in _switch_model(ctx, primary="return")
    assert (ctx.active_model_override, ctx.active_use_local_override, ctx.active_role_override) == (
        "consc/model", False, "consciousness")
    assert ctx.route_wait_on_primary is False and ctx.active_effort_override is None  # effort intent untouched
    ctx.model_wait_context = SimpleNamespace(overrides={"consciousness": {"model": "owner/choice", "use_local": True}})
    assert "wait for it" in _switch_model(ctx, primary="wait")
    assert (ctx.active_model_override, ctx.active_use_local_override, ctx.route_wait_on_primary) == (
        "owner/choice", True, True)
    assert "TOOL_ARG_ERROR" in _switch_model(ctx, primary="wait", model="x") or "without model" in _switch_model(
        ctx, primary="wait", model="x")
    assert "primary must be" in _switch_model(ctx, primary="later")
    assert "no recorded primary" in _switch_model(SimpleNamespace(primary_route=None), primary="return")


def test_a_declared_wait_keeps_refusals_and_outages_on_the_primary():
    ctx = SimpleNamespace(task_id="t", exact_model_route=False, route_wait_on_primary=True)
    refusal = {"_last_llm_resource_refusal": "quota"}
    for kind, usage in (("subscription_window_exhausted", refusal), ("transport_unavailable", {}),
                        ("provider_outcome_unknown", {})):
        assert loop_transport.fallback_chain_allowed(ctx, kind, None, usage) is False
    assert loop_transport.fallback_chain_allowed(ctx, "bad_request", None, {}) is True  # ordinary recovery stays
    ctx.route_wait_on_primary = False
    assert loop_transport.fallback_chain_allowed(ctx, "subscription_window_exhausted", None, refusal) is True


def test_primary_return_rebinds_role_and_leaves_auto_unpinned(monkeypatch):
    from ouroboros.loop_model_call import _apply_round_route_overrides

    calls = []
    monkeypatch.setattr(loop_mod, "_rebind_context_fit_plan",
                        lambda plan, *_a, **kwargs: calls.append(kwargs) or (plan, "low"))
    ctx = SimpleNamespace(primary_route={"model": "consc/model", "use_local": False, "role": "consciousness"},
                          model_wait_context=None, active_model_override=None, active_use_local_override=None,
                          active_effort_override=None)
    _switch_model(ctx, primary="return")
    messages = [{"role": "user", "content": "go"}]
    route = _apply_round_route_overrides(ctx, SimpleNamespace(_ctx=ctx), messages, ("fb/one", False, "high"),
                                         "plan", "max", "max", [])
    assert route == ("consc/model", False, "high", "plan", "low")
    assert calls[-1]["model_role"] == "consciousness" and calls[-1]["model_route"] == {}
    assert ctx.active_role_override is None  # one-shot


def test_initial_route_records_a_role_slot_primary(tmp_path):
    ctx = SimpleNamespace(task_model_override="consc/model", task_use_local_override=False,
                          task_metadata={"model_role": "consciousness"}, context_fit_plan=None)
    model, _effort, local, *_rest = loop_mod._initial_round_route(ctx, SimpleNamespace(default_model=lambda: "main/model"), "high")
    assert (model, local) == ("consc/model", False)
    assert ctx.primary_route == {"model": "consc/model", "use_local": False, "role": "consciousness"}


def test_route_facts_name_both_bindings_the_dated_failure_and_the_choices():
    ctx = SimpleNamespace(primary_route={"model": "claudexor::codex=gpt-test", "use_local": False, "role": "main"})
    text = _route_facts_text(ctx, model="fb/one", use_local=False, role="fallback:0",
                             failed_model="claudexor::codex=gpt-test",
                             failure={"failure_code": "subscription_window_exhausted", "ts": "2026-09-30T12:00:00Z",
                                      "reset_at": "2026-09-30T18:00:00Z"}, waiter=None)
    assert text.startswith("[ROUTE FACTS]") and "(role main; account Auto)" in text and "(role fallback:0)" in text
    assert "subscription_window_exhausted at 2026-09-30T12:00:00Z" in text and "2026-09-30T18:00:00Z" in text
    assert 'switch_model(primary="return")' in text and "No catalog check or timer" in text
    assert _route_facts_text(SimpleNamespace(), model="m", use_local=False, role="r", failed_model="p",
                             failure={}, waiter=None) == ""


def test_route_facts_reach_only_the_acting_routes_own_next_round(monkeypatch):
    from ouroboros import loop_model_call

    tool_ctx = SimpleNamespace(_route_facts_pending="[ROUTE FACTS] facts")
    monkeypatch.setattr(loop_mod, "_dispatch_round_model", lambda *_a, **_kw: ({"content": "x"}, 0.0))
    monkeypatch.setattr(loop_model_call, "_append_routing_receipts", lambda _ctx: False)
    monkeypatch.setattr(loop_model_call, "_project_wake_input", lambda *_a, **_kw: False)
    monkeypatch.setattr(loop_mod, "_measure_round_main_fit", lambda *_a, **_kw: None)

    def call(defer):
        messages = [{"role": "user", "content": "go"}, {"role": "assistant", "content": "tool round"}]
        loop_model_call._call_round_model(SimpleNamespace(
            tools=SimpleNamespace(_ctx=tool_ctx), messages=messages, defer_resource_wait=defer,
            attempt_cap=None, accumulated_usage={}, active_context_mode="max",
            active_model="m", active_use_local=False, tool_schemas=[]))
        return messages

    assert call(True)[-1]["content"] == "tool round"  # a candidate's copy never takes the note
    assert call(None)[-1]["content"] == "[ROUTE FACTS] facts"
    assert tool_ctx._route_facts_pending == "" and call(None)[-1]["content"] == "tool round"  # exactly once


# -- Claudexor custody: engine-terminal unknown vs. an operation that could not be read ----------


def test_engine_terminal_unknown_is_eligible_and_a_lost_control_link_is_not(setup):
    from ouroboros import llm_claudexor as transport

    _root, gateway, client = setup
    gateway.results, gateway.dispatch = [result(outcome="unknown"), result()], ["unknown", "response_received"]
    with pytest.raises(transport.ClaudexorModelError) as terminal:
        client.chat([{"role": "user", "content": "hi"}], "claudexor::codex=exact-model")
    assert terminal.value.code == "model_outcome_unknown" and terminal.value.same_operation_recoverable is False
    gateway.pending = gateway.read_error = True
    with pytest.raises(transport.ClaudexorModelError) as lost:
        client.chat([{"role": "user", "content": "hi"}], "claudexor::codex=exact-model", timeout=0.01)
    assert lost.value.code == "model_outcome_unknown" and lost.value.same_operation_recoverable is True


@pytest.mark.parametrize("managed", [True, False])
def test_a_refused_read_of_an_accepted_operation_keeps_its_custody(tmp_path, monkeypatch, managed):
    """A 404 is no proof of non-dispatch: ordinary direct and queued calls rejoin the same operation."""
    from ouroboros import llm_claudexor
    from ouroboros.gateways.claudexor import ClaudexorUnavailable

    inv = llm_claudexor._ModelInvocation({"usage_model": "claudexor::codex=test"}, {}, {"timeout": 1})
    inv.operation_id, inv.invocation_id, inv.task_id, inv.root = "same-op", "same-attempt", "t", tmp_path
    inv.create_attempted = True
    reads = []
    ctx = SimpleNamespace(task_id="t", is_direct_chat=not managed)
    monkeypatch.setattr(llm_claudexor, "current_model_wait",
                        lambda: SimpleNamespace(tool_context=ctx, control_reason=lambda: None))
    now = [0.0]
    monkeypatch.setattr(llm_claudexor, "time", SimpleNamespace(
        monotonic=lambda: now[0], sleep=lambda seconds: now.__setitem__(0, now[0] + seconds)))

    class Gateway:
        def get_model_operation(self, operation, **_kwargs):
            reads.append(operation)
            if len(reads) <= 2:
                raise ClaudexorUnavailable("http_404", "operation not found", status_code=404)
            return {"id": operation, "state": "succeeded", "dispatch": {"state": "response_received", "route": ROUTE},
                    "response": {"state": "ready", "ref": {"sha256": "exact"}}}

        def get_model_result(self, operation, **_kwargs):
            return b'{"outcome":"completed","message":{"content":"same operation answer"}}'

        def create_model_operation(self, *_a, **_kw):
            pytest.fail("a second model operation")

        def close(self):
            pass

    gateway = inv.gateway = Gateway()
    monkeypatch.setattr(llm_claudexor, "read_owned_gateway", lambda: gateway)
    monkeypatch.setattr(inv, "retain", lambda _raw: None)
    assert inv.receive()["message"]["content"] == "same operation answer"
    assert reads == ["same-op"] * 3


def test_a_readable_unknown_after_a_granted_continuation_ends_the_episode(tmp_path):
    """Allowed vs disallowed after a grant: a new eligible unknown keeps the same episode
    (same clock and backoff), while one whose accepted operation is still readable ends it,
    so no further generation can be granted over an operation that may still finish."""
    def granted():
        return loop_transport.TransportWaitEpisode(
            wait_cause="provider_outcome_unknown", started_monotonic=0.0, continuation_granted=True,
            outcome_custody={"physical_attempt_id": "old"}, wait_iterations=1, redials=1)

    kwargs = dict(msg_present=False, error_kind="provider_outcome_unknown", drive_logs=tmp_path, task_id="t",
                  model="m", emit_progress=lambda *_a, **_kw: None)
    eligible = SimpleNamespace(task_id="t", _accumulated_usage={
        "_pending_transport_outcome": {"physical_attempt_id": "new", "outcome": "unknown"}})
    episode = granted()
    assert loop_transport.reconcile_transport_wait(episode, eligible, **kwargs) is episode
    assert not episode.continuation_granted and episode.outcome_custody["physical_attempt_id"] == "new"
    readable = SimpleNamespace(task_id="t", _accumulated_usage={"_pending_transport_outcome": {
        "physical_attempt_id": "new", "outcome": "unknown", "same_operation_recoverable": True}})
    assert loop_transport.reconcile_transport_wait(granted(), readable, **kwargs) is None


@pytest.mark.parametrize("kind", ["provider_transient", "rate_limit"])
def test_chosen_primary_refusal_keeps_existing_episode_bounds_and_unknown_custody(tmp_path, kind):
    ctx = SimpleNamespace(task_id="t", route_wait_on_primary=True, is_direct_chat=True, _accumulated_usage={})
    assert not loop_transport.fallback_chain_allowed(ctx, kind, None, {})
    episode = loop_transport.TransportWaitEpisode(wait_cause="provider_outcome_unknown", started_monotonic=12,
        wait_iterations=3, redials=3, continuation_granted=True, outcome_custody={"physical_attempt_id": "old"})
    kwargs = dict(drive_logs=tmp_path, task_id="t", model="primary", emit_progress=lambda *_a, **_kw: None)
    resumed = loop_transport.reconcile_transport_wait(episode, ctx, msg_present=False, error_kind=kind, **kwargs)
    assert resumed is episode and episode.wait_cause == kind and not episode.continuation_granted
    assert (episode.started_monotonic, episode.wait_iterations, episode.redials) == (12, 3, 3)
    assert episode.outcome_custody == {"physical_attempt_id": "old"}
    assert loop_transport.reconcile_transport_wait(episode, ctx, msg_present=True, error_kind="", **kwargs) is None
    ctx.route_wait_on_primary = False
    assert loop_transport.fallback_chain_allowed(ctx, kind, None, {})
    assert loop_transport.reconcile_transport_wait(None, ctx, msg_present=False, error_kind=kind, **kwargs) is None
    ctx.route_wait_on_primary, ctx.current_task_type = True, "presence"
    assert not loop_transport.primary_refusal_wait(ctx, kind)
    ctx.current_task_type, ctx.exact_model_route = "task", True
    assert not loop_transport.primary_refusal_wait(ctx, kind)
