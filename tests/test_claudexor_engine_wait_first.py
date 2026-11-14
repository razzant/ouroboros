"""The own Claudexor daemon left unanswered before create waits on the work's model (owner wait-first, 13.11).

A typed prepare failure (``llm_claudexor.daemon_not_delivered``: provider claudexor, no operation, a daemon code)
proves only that the install's own control daemon did not answer: nothing was sent or charged, and every Claudexor
route shares that daemon. After its same-route repeats it never opens the configured routes; the round's existing
wait episode redials the same model under its own Stop, deadline and idle rails. A live engine's refusal, another
provider's failure (even when a Claudexor candidate then fails mid-walk) and the unknown-outcome rails keep their
own recovery. The fakes below stand in for routes; no daemon is started or dialed.
"""

from __future__ import annotations

import datetime
import json
import queue
import re
import time
from types import SimpleNamespace

import pytest

import ouroboros.loop as loop_mod
import ouroboros.loop_transport as loop_transport
from ouroboros import model_wait, owner_model, route_choice
from ouroboros import usage_accounting as ua
from ouroboros.llm import LLMClient
from ouroboros.loop import run_llm_loop
from ouroboros.loop_llm_call import TRANSPORT_DEATHS_KEY
from ouroboros.loop_transport import (
    TransportWaitEpisode,
    _provider_terminal_text,
    engine_unanswered,
    fallback_chain_allowed,
    reconcile_transport_wait,
    transport_wait_step,
)
from ouroboros.tools.registry import ToolRegistry
from tests._budget_pause_exact_helpers import _install_queue
from tests._shared import saved_settings
from tests.test_loop_transport_wait import _typed_transport_exc
from tests.test_not_delivered_same_route import _prepare_failure, _receipt

PRIMARY, F1, F2 = "openai::primary", "openai::candidate-one", "openai::candidate-two"


class _Routes:
    """Each model pops its scripted failures (a factory per send), then answers; ``sent`` records every send and
    ``inputs`` its messages. Main is the production client's own reading of the saved slot, live."""

    default_model = LLMClient.default_model

    def __init__(self, root, script):
        self.root, self.sent, self.inputs = root, [], []
        self.script = {model: list(rows) for model, rows in script.items()}

    @model_wait.model_waitable
    def chat(self, **kwargs):
        model = kwargs["model"]
        self.sent.append(model)
        self.inputs.append(json.dumps(kwargs.get("messages"), ensure_ascii=False, default=str))
        rows = self.script.get(model) or []
        if rows:
            failure = rows.pop(0) if len(rows) > 1 else rows[0]  # the last row repeats forever
            if failure is not None:
                raise failure()
        return {"role": "assistant", "content": f"answer from {model}"}, {"prompt_tokens": 1, "completion_tokens": 1}


@pytest.fixture
def env(tmp_path, monkeypatch):
    root = tmp_path / "data"
    (root / "state").mkdir(parents=True)
    monkeypatch.setenv("OUROBOROS_DATA_DIR", str(root))
    monkeypatch.setenv("TOTAL_BUDGET", "100")
    monkeypatch.setenv("OUROBOROS_TASK_REVIEW_MODE", "off")
    monkeypatch.delenv("USE_LOCAL_FALLBACK", raising=False)
    monkeypatch.setattr(loop_mod, "_rebind_context_fit_plan", lambda plan, *_a, **_kw: (plan, "max"))
    monkeypatch.setattr(loop_mod, "_server_web_allowed_by_task", lambda _ctx: False)
    from ouroboros import fallback_cooldown
    monkeypatch.setattr(fallback_cooldown, "is_cooling_down", lambda *_a: False)
    sleeps = {"burst": [], "episode": []}
    monkeypatch.setattr(time, "sleep", lambda sec: sleeps["burst"].append(sec))
    monkeypatch.setattr(loop_transport, "interruptible_wait_sleep",
                        lambda sec, _wake: (sleeps["episode"].append(sec), False)[1])
    return SimpleNamespace(root=root, sleeps=sleeps)


def _run(tmp_path, llm, notes, task_type=""):
    from ouroboros.task_results import write_task_result

    write_task_result(tmp_path, "t-ew", "running")
    registry = ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path)
    task = {"id": "t-ew", "_attempt": 1, "drive_root": str(tmp_path),
            **({"_presence_turn": True} if task_type == "presence" else {})}
    with ua.usage_scope(ua.UsageScope(drive_root=llm.root, task_id="t-ew", root_task_id="t-ew", global_limit_usd=100)), \
            model_wait.task_model_wait_scope(task=task, drive_root=tmp_path, event_queue=None, worker_slot_held=False,
                                             owner_control=lambda: "") as owner:
        registry._ctx.model_wait_context = owner
        owner.tool_context = registry._ctx
        if task_type == "presence":  # the host's own attempt fact for a correspondent's turn (task_carrier)
            registry._ctx.task_metadata = {**(registry._ctx.task_metadata or {}), "_execution_owner": {"kind": "presence"}}
        text, usage, _trace = run_llm_loop(
            messages=[{"role": "user", "content": "go"}], tools=registry, llm=llm, drive_logs=tmp_path,
            emit_progress=lambda text, *, incident=None, **_kw: notes.append(text), incoming_messages=queue.Queue(),
            task_id="t-ew", task_type=task_type, drive_root=tmp_path)
    return text, usage, registry


def _wait_phases(tmp_path):
    rows = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    return [row["phase"] for row in rows if row.get("type") == "network_wait"]


def _recovery_note(notes):
    """The wait's one recovery note: that a response came after the wait, nothing about which engine or model."""
    recovered = [note for note in notes if note.startswith("🌐 Model response received")]
    assert len(recovered) == 1 and re.fullmatch(r"🌐 Model response received after \d+\.\d min — resuming\.",
                                                 recovered[0]), notes
    return recovered[0]


def test_spent_daemon_burst_waits_on_its_model_instead_of_walking(env, tmp_path, monkeypatch):
    """The candidate side of ``test_a_spent_burst_walks_the_configured_route[_prepare_failure]``: that parent
    expectation (a paid configured route answers) is the behaviour this repair removes."""
    saved_settings(monkeypatch, tmp_path, OUROBOROS_MODEL=PRIMARY, OUROBOROS_MODEL_FALLBACKS=f"{F1}, {F2}",
                   OPENAI_API_KEY="key")
    llm, notes = _Routes(env.root, {PRIMARY: [_prepare_failure] * 5 + [None]}), []
    text, _usage, registry = _run(tmp_path, llm, notes)

    assert text == f"answer from {PRIMARY}"
    assert llm.sent == [PRIMARY] * 6  # the same-route burst of three, then one send per free redial
    # The burst's 4 s and 8 s waits wake for owner words like the episode's three redial waits that follow.
    assert env.sleeps["burst"] == [] and env.sleeps["episode"][:2] == [4.0, 8.0] and len(env.sleeps["episode"]) == 5
    assert _wait_phases(tmp_path) == ["entered", "waiting", "waiting", "waiting", "recovered"]
    assert not any("Fallback" in note for note in notes)  # no candidate was announced, prepared or adopted
    assert getattr(registry._ctx, "route_wait_on_primary", False) is False  # no declared wait was invented
    entry = next(note for note in notes if "Claudexor engine is not answering" in note)
    assert entry == ("🌐 The local Claudexor engine is not answering, so this request was not sent ($0 for it). "
                     "The work keeps its model, tries no other route and retries automatically within its existing "
                     "limits. Stop cancels.")
    assert _recovery_note(notes)


@pytest.mark.parametrize("trigger", ["settings", "owner_card"])
def test_an_answer_after_the_owner_moved_the_route_claims_no_recovered_engine(env, tmp_path, monkeypatch, trigger):
    """While the work waits on the silent daemon the owner moves its route: a new Main saved in Settings (the work
    follows Main), or another model chosen on this work's card. The next redial binds it through the round's own
    route handover, and its answer ends the wait. The recovery note says only that a response came: never that the
    local Claudexor engine answered again or that the work resumed on the same model (the earlier sentence said both)."""
    saved_settings(monkeypatch, tmp_path, OUROBOROS_MODEL=PRIMARY, OUROBOROS_MODEL_FALLBACKS=F1, OPENAI_API_KEY="key")
    if trigger == "owner_card":  # the running work the card's durable act names (supervisor queue, task document)
        _queue, _state, workers = _install_queue(tmp_path, monkeypatch)
        workers.RUNNING["t-ew"] = {"attempt": 1, "lead": {"model": PRIMARY, "use_local": False, "basis": "main"},
                                   "task": {"id": "t-ew", "type": "task", "chat_id": 7, "root_task_id": "t-ew",
                                            "drive_root": str(tmp_path), "metadata": {}}}
    moved = []

    def owner_moves_the_route(sec, _wake):  # the burst's two pauses come first, then the episode's paced wait
        env.sleeps["episode"].append(sec)
        if not moved and (tmp_path / "events.jsonl").exists() and "entered" in _wait_phases(tmp_path):
            moved.append(saved_settings(monkeypatch, tmp_path, OUROBOROS_MODEL=F2) if trigger == "settings" else
                         owner_model.request("t-ew", {"request_id": "press", "action": "set", "model": F2,
                                                      "use_local": False}))
        return False

    monkeypatch.setattr(loop_transport, "interruptible_wait_sleep", owner_moves_the_route)
    llm, notes = _Routes(env.root, {PRIMARY: [_prepare_failure]}), []
    text, _usage, registry = _run(tmp_path, llm, notes)

    assert moved and text == f"answer from {F2}"
    assert llm.sent == [PRIMARY] * 3 + [F2]  # F1, the configured route, was never sent: no walk answered
    # The claimed trigger moved the primary, and the mind was told who moved it.
    assert registry._ctx.primary_route["model"] == F2
    if trigger == "settings":
        assert registry._ctx.route_choice == route_choice.following("main")
        assert f"[ROUTE FACTS] Main changed in Settings: {PRIMARY} → {F2}." in llm.inputs[-1]
    else:
        assert registry._ctx.route_choice == route_choice.explicit("owner")
        assert f"[ROUTE FACTS] My human set this work's primary: {F2}." in llm.inputs[-1]
    assert _wait_phases(tmp_path) == ["entered", "waiting", "recovered"]
    assert notes[0].startswith("🌐 The local Claudexor engine is not answering")
    assert _recovery_note(notes)
    assert not any("answered again" in note or "same model" in note for note in notes), notes


def test_an_inline_presence_turn_waits_too_under_its_existing_bound(env, tmp_path, monkeypatch):
    """Presence has no queue rails: its episode carries the existing idle bound from entry, with no new threshold."""
    saved_settings(monkeypatch, tmp_path, OUROBOROS_MODEL=PRIMARY, OUROBOROS_MODEL_FALLBACKS=F1, OPENAI_API_KEY="key")
    entered = []
    real = loop_transport.reconcile_transport_wait

    def spy(*args, **kwargs):
        episode = real(*args, **kwargs)
        if episode is not None and not entered:
            entered.append((episode.interactive, episode.wait_bound_sec, episode.engine_unanswered))
        return episode

    monkeypatch.setattr(loop_mod, "_reconcile_transport_wait", spy)
    llm, notes = _Routes(env.root, {PRIMARY: [_prepare_failure] * 4 + [None]}), []
    text, _usage, _registry = _run(tmp_path, llm, notes, task_type="presence")
    assert text == f"answer from {PRIMARY}" and F1 not in llm.sent
    from ouroboros.config import get_task_idle_timeout_sec
    assert entered == [(True, float(get_task_idle_timeout_sec()), True)]
    assert not any(note.endswith("Stop cancels.") for note in notes)  # interactive notes promise no cancellation


@pytest.mark.parametrize("failure", [
    _receipt,  # the engine answered with its not-delivered receipt: it is alive, the configured routes stay open
    lambda: _receipt(code="credential_pool_exhausted"),  # a live engine's undated pool refusal
    lambda: _receipt(code="daemon_unreachable"),  # an accepted operation is not prepare
    _typed_transport_exc,  # another provider's host connect failure
])
def test_a_live_engine_or_another_providers_failure_still_walks(env, tmp_path, monkeypatch, failure):
    saved_settings(monkeypatch, tmp_path, OUROBOROS_MODEL=PRIMARY, OUROBOROS_MODEL_FALLBACKS=F1, OPENAI_API_KEY="key")
    llm, notes = _Routes(env.root, {PRIMARY: [failure]}), []
    text, _usage, _registry = _run(tmp_path, llm, notes)
    assert text == f"answer from {F1}" and llm.sent[-1] == F1
    assert not any("Claudexor engine" in note for note in notes)


def test_a_claudexor_candidate_failing_mid_walk_never_stops_the_next_walk(env, tmp_path, monkeypatch):
    """The acting primary is another provider: its outage walks; a Claudexor candidate's silent daemon is that
    candidate's row only. The episode it opens words the acting route's outage, and the next redial walks again."""
    saved_settings(monkeypatch, tmp_path, OUROBOROS_MODEL=PRIMARY, OUROBOROS_MODEL_FALLBACKS=f"{F1}, {F2}",
                   OPENAI_API_KEY="key")
    # The walk's last row is the Claudexor candidate's silent daemon; the acting route's own outage opens the wait.
    llm, notes = _Routes(env.root, {PRIMARY: [_typed_transport_exc], F1: [_typed_transport_exc, None],
                                    F2: [_prepare_failure]}), []
    text, _usage, _registry = _run(tmp_path, llm, notes)

    assert text == f"answer from {F1}"
    assert llm.sent == [PRIMARY, F1, F2, PRIMARY, F1]  # the redial's own outage walked again
    assert _wait_phases(tmp_path)[0] == "entered"
    assert not any("Claudexor engine" in note for note in notes)
    assert any(note.startswith("🌐 The request did not reach the provider") for note in notes)


def _ctx(usage, **facts):
    return SimpleNamespace(exact_model_route=False, route_wait_on_primary=False, task_id="t-ew",
                           _accumulated_usage=usage, task_metadata={}, **facts)


def _daemon_row(**overrides):
    return {"not_dispatched_provider": "claudexor", "operation_id": "", "provider_code": "daemon_unreachable",
            "model": PRIMARY, **overrides}


@pytest.mark.parametrize("primary_acting", [True, False])  # False: an adopted Claudexor fallback is acting
def test_the_gate_reads_the_acting_routes_own_failure(primary_acting):
    usage = {"_last_llm_error_kind": "transport_unavailable", "_last_llm_failure": _daemon_row()}
    ctx = _ctx(usage, primary_route={"model": PRIMARY, "role": "main", "use_local": False})
    # An adopted fallback whose daemon is silent neither walks back to the primary nor on: every Claudexor route
    # shares the daemon and a return or a walk would rewrite the work's start elsewhere.
    assert fallback_chain_allowed(ctx, "transport_unavailable", None, usage, primary_acting=primary_acting) is False
    episode = TransportWaitEpisode(started_monotonic=time.monotonic())
    assert fallback_chain_allowed(ctx, "transport_unavailable", episode, usage, primary_acting=primary_acting) is False
    # After a failed walk, the acting route's own row decides, not the last candidate's.
    usage["_acting_route_failure"] = {"not_dispatched_provider": "", "operation_id": "", "provider_code": ""}
    assert engine_unanswered(usage) is False
    assert fallback_chain_allowed(ctx, "transport_unavailable", None, usage, primary_acting=primary_acting) is True


@pytest.mark.parametrize("row", [
    _daemon_row(operation_id="op-1"),  # an accepted operation: its loss is the rejoin/unknown rail's
    _daemon_row(provider_code="transport_not_delivered"),  # the engine's own receipt
    _daemon_row(not_dispatched_provider=""),  # untyped
    _daemon_row(provider_code="model_capacity_unavailable"),
])
def test_only_every_typed_fact_together_is_the_silent_daemon(row):
    usage = {"_last_llm_error_kind": "transport_unavailable", "_last_llm_failure": row}
    assert engine_unanswered(usage) is False


def test_post_create_unknown_and_its_rejoin_keep_their_own_rails():
    """A daemon code on an unknown-outcome row changes nothing: the kind's own rules (a readable operation is only
    rejoined; an eligible one walks from its open episode) decide exactly as without the row."""
    for pending, held in ((None, False), ({"same_operation_recoverable": True}, False), (None, True)):
        verdicts = []
        for row in (_daemon_row(operation_id="op-1"), _daemon_row(), None):
            usage = {"_last_llm_error_kind": "provider_outcome_unknown",
                     **({"_last_llm_failure": row} if row else {}),
                     **({"_pending_transport_outcome": pending} if pending else {}),
                     **({TRANSPORT_DEATHS_KEY: {"round_id": "r", "count": 1}} if held else {})}
            ctx = _ctx(usage)
            assert engine_unanswered(usage) is False
            verdicts.append((fallback_chain_allowed(ctx, "provider_outcome_unknown", None, usage),
                             fallback_chain_allowed(ctx, "provider_outcome_unknown",
                                                    TransportWaitEpisode(started_monotonic=time.monotonic()), usage)))
        assert verdicts[0] == verdicts[1] == verdicts[2]


def _tools(usage, deadline_at=None):
    return SimpleNamespace(_ctx=SimpleNamespace(task_metadata={"deadline_at": deadline_at} if deadline_at else {},
                                                _accumulated_usage=usage, task_attempt=1))


def _step(tmp_path, episode, usage, notes, deadline_at=None, error_kind="transport_unavailable"):
    return transport_wait_step(episode, tools=_tools(usage, deadline_at), error_kind=error_kind, drive_root=None,
                               drive_logs=tmp_path, task_id="t-ew", model=PRIMARY,
                               emit_progress=lambda text, *, incident=None: notes.append(text),
                               incoming_messages=None, owner_msg_seen=None)


def test_periodic_note_words_the_unsent_request_and_the_kept_model(tmp_path, monkeypatch):
    monkeypatch.setattr(loop_transport, "interruptible_wait_sleep", lambda _sec, _wake: False)
    usage, notes = {"_last_llm_error_kind": "transport_unavailable", "_last_llm_failure": _daemon_row()}, []
    episode = TransportWaitEpisode(started_monotonic=time.monotonic() - 180, engine_unanswered=True, redials=4)
    assert _step(tmp_path, episode, usage, notes) is True
    assert notes == ["🌐 Still waiting for the local Claudexor engine — 3 min elapsed, 4 retries; the latest request "
                     "was not sent; the work keeps its model; will resume automatically."]


def test_reconcile_rewords_the_episode_from_each_redials_own_failure(tmp_path):
    """Entered on a silent daemon, a redial that now reaches the engine and fails as a transport outage words the
    next notes as that outage (the gate then opens the walk); the elapsed clock stays the episode's."""
    usage, notes = {"_last_llm_error_kind": "transport_unavailable", "_last_llm_failure": _daemon_row()}, []
    ctx = _ctx(usage)
    episode = reconcile_transport_wait(None, ctx, msg_present=False, error_kind="transport_unavailable",
                                       drive_logs=tmp_path, task_id="t-ew", model=PRIMARY,
                                       emit_progress=lambda text, *, incident=None: notes.append(text))
    assert episode.engine_unanswered is True and "Claudexor engine is not answering" in notes[0]
    usage["_last_llm_failure"] = _daemon_row(provider_code="transport_not_delivered", operation_id="op-1")
    assert reconcile_transport_wait(episode, ctx, msg_present=False, error_kind="transport_unavailable",
                                    drive_logs=tmp_path, task_id="t-ew", model=PRIMARY,
                                    emit_progress=notes.append) is episode
    assert episode.engine_unanswered is False and len(notes) == 1


@pytest.mark.parametrize("later", ["dispatched_unknown", "received_refusal"])
def test_a_later_dispatch_or_refusal_clears_the_engine_words(tmp_path, monkeypatch, later):
    """Entered on a silent daemon, a redial that crosses create and loses its answer (an eligible new attempt) or
    meets a received refusal under a declared primary wait is no unsent request: the episode keeps its clock and
    takes that cause's own words, never "not sent", "the local Claudexor engine" or "keeps its model"."""
    monkeypatch.setattr(loop_transport, "interruptible_wait_sleep", lambda _sec, _wake: False)
    usage, notes = {"_last_llm_error_kind": "transport_unavailable", "_last_llm_failure": _daemon_row()}, []
    ctx = _ctx(usage)
    ctx.route_wait_on_primary = later == "received_refusal"
    say = dict(drive_logs=tmp_path, task_id="t-ew", model=PRIMARY,
               emit_progress=lambda text, *, incident=None: notes.append(text))
    episode = reconcile_transport_wait(None, ctx, msg_present=False, error_kind="transport_unavailable", **say)
    assert episode.engine_unanswered is True
    kind = "provider_outcome_unknown" if later == "dispatched_unknown" else "rate_limit"
    usage.update(_last_llm_error_kind=kind, _last_llm_failure=_daemon_row(operation_id="op-1"),
                 **({"_pending_transport_outcome": {"operation_id": "op-1", "physical_attempt_id": "pa-2"}}
                    if later == "dispatched_unknown" else {}))
    assert reconcile_transport_wait(episode, ctx, msg_present=False, error_kind=kind, **say) is episode
    assert (episode.wait_cause, episode.engine_unanswered) == (kind, False)
    if later == "dispatched_unknown":  # the unknown rail's own custody and words
        assert episode.outcome_custody["operation_id"] == "op-1"
        assert notes[-1].startswith("🌐 Provider connection was lost after dispatch.")
    episode.last_note_monotonic = 0.0
    assert _step(tmp_path, episode, usage, notes, error_kind=kind) is True
    waiting_for = "a provider connection" if later == "dispatched_unknown" else "the primary provider"
    assert notes[-1] == f"🌐 Still waiting for {waiting_for} — 0 min elapsed, 0 redials; will resume automatically."
    episode.interactive, episode.wait_bound_sec, episode.started_monotonic = True, 60.0, time.monotonic() - 120
    assert _step(tmp_path, episode, usage, notes, error_kind=kind) is False
    assert notes[-1] == f"🌐 Stopped waiting for {waiting_for} after 2.0 min — this turn ends as an outage."


def test_a_granted_continuation_met_by_the_silent_daemon_takes_the_engine_words(tmp_path):
    """The new attempt after an unknown outcome released at the silent daemon: the same episode words it as the
    engine's ($0, kept model) while the earlier attempt's custody stays; the walk stays closed."""
    usage, notes = {"_last_llm_error_kind": "transport_unavailable", "_last_llm_failure": _daemon_row()}, []
    ctx = _ctx(usage)
    episode = TransportWaitEpisode(wait_cause="provider_outcome_unknown", started_monotonic=time.monotonic(),
                                   continuation_granted=True, outcome_custody={"operation_id": "op-1"})
    assert reconcile_transport_wait(episode, ctx, msg_present=False, error_kind="transport_unavailable",
                                    drive_logs=tmp_path, task_id="t-ew", model=PRIMARY,
                                    emit_progress=lambda text, *, incident=None: notes.append(text)) is episode
    assert (episode.wait_cause, episode.engine_unanswered, episode.outcome_custody) == (
        "transport_unavailable", True, {"operation_id": "op-1"})
    assert notes == ["🌐 The new attempt could not reach the provider — waiting and redialing automatically "
                     "(that attempt was $0)."]
    assert fallback_chain_allowed(ctx, "transport_unavailable", episode, usage) is False


def test_an_accepted_operation_ends_the_engine_wait_for_its_rejoin(tmp_path):
    """An accepted operation that may still finish is only rejoined: the engine episode ends, no walk dials over it
    and no new wait opens (the post-create rails exactly as without the daemon history)."""
    usage, notes = {"_last_llm_error_kind": "transport_unavailable", "_last_llm_failure": _daemon_row()}, []
    ctx = _ctx(usage)
    say = dict(drive_logs=tmp_path, task_id="t-ew", model=PRIMARY,
               emit_progress=lambda text, *, incident=None: notes.append(text))
    episode = reconcile_transport_wait(None, ctx, msg_present=False, error_kind="transport_unavailable", **say)
    usage.update(_last_llm_error_kind="provider_outcome_unknown", _last_llm_failure=_daemon_row(operation_id="op-1"),
                 _pending_transport_outcome={"operation_id": "op-1", "same_operation_recoverable": True})
    assert reconcile_transport_wait(episode, ctx, msg_present=False, error_kind="provider_outcome_unknown", **say) is None
    assert fallback_chain_allowed(ctx, "provider_outcome_unknown", None, usage) is False
    assert reconcile_transport_wait(None, ctx, msg_present=False, error_kind="provider_outcome_unknown", **say) is None
    rows = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [(row["phase"], row.get("detail", "")) for row in rows if row.get("type") == "network_wait"] == [
        ("entered", ""), ("ended", "error_kind_changed:provider_outcome_unknown")]
    assert len(notes) == 1  # the entry note; a managed task's ended row speaks for itself


def test_an_engine_terminal_after_an_earlier_walk_claims_no_untried_model(env, tmp_path, monkeypatch):
    """A live engine's receipt legally walks; the silent daemon that follows waits on the model. Its terminal names
    the unsent last request and claims nothing about which models were tried or which one the work ended on."""
    saved_settings(monkeypatch, tmp_path, OUROBOROS_MODEL=PRIMARY, OUROBOROS_MODEL_FALLBACKS=F1, OPENAI_API_KEY="key")
    llm, notes = _Routes(env.root, {PRIMARY: [_receipt] * 3 + [_prepare_failure], F1: [_typed_transport_exc]}), []
    # The owner deadline passes with the first silent-daemon redial: the managed wait ends on that rail.
    monkeypatch.setattr(loop_transport, "task_deadline_epoch",
                        lambda _tools: time.time() - 60 if llm.sent.count(PRIMARY) >= 4 else None)
    text, _usage, _registry = _run(tmp_path, llm, notes)
    assert llm.sent == [PRIMARY] * 3 + [F1, PRIMARY]  # the receipt's walk; the silent daemon's redial walks no more
    assert any(note.startswith("⚡ Fallback: ") for note in notes)
    assert text.startswith("⚠️ The local Claudexor engine did not answer: the last request was not sent ($0 for it), "
                           "and the task ended after ")
    assert "no other model" not in text and "selected model" not in text


def test_interactive_idle_bound_ends_the_turn_with_engine_words(tmp_path):
    usage, notes = {"_last_llm_error_kind": "transport_unavailable", "_last_llm_failure": _daemon_row(),
                    "_last_llm_provider_code": "daemon_unreachable"}, []
    episode = TransportWaitEpisode(started_monotonic=time.monotonic() - 960, interactive=True, wait_bound_sec=900.0,
                                   engine_unanswered=True, wait_iterations=3)
    assert _step(tmp_path, episode, usage, notes) is False  # the existing bound: no new threshold
    assert notes == ["🌐 Stopped waiting for the local Claudexor engine after 16.0 min — its last request was not "
                     "sent, and this turn ends without an answer."]
    rows = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    assert rows[-1]["detail"] == "interactive_wait_window_exhausted"
    text = _provider_terminal_text(usage, is_context_overflow=False, is_transport_wait=True, waited_sec=960.0,
                                   interactive=True, is_deadline_exhausted=False)
    assert text == ("⚠️ The local Claudexor engine did not answer: the last request was not sent ($0 for it), and "
                    "this turn ended after 16.0 min of waiting without completing. Last reported code: daemon_unreachable.")


def test_owner_deadline_ends_a_managed_wait_with_engine_words(tmp_path):
    usage, notes = {"_last_llm_error_kind": "transport_unavailable", "_last_llm_failure": _daemon_row()}, []
    past = (datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(minutes=5)).isoformat()
    episode = TransportWaitEpisode(started_monotonic=time.monotonic() - 60, engine_unanswered=True, wait_iterations=1)
    assert _step(tmp_path, episode, usage, notes, deadline_at=past) is False and notes == []  # a managed terminal
    text = _provider_terminal_text(usage, is_context_overflow=False, is_transport_wait=True, waited_sec=0.0,
                                   interactive=False, is_deadline_exhausted=False)
    assert text.startswith("⚠️ The local Claudexor engine did not answer: the last request was not sent ($0 for it), "
                           "and the task ended without completing.")
    assert "Any files written so far are preserved in the workspace." in text


@pytest.mark.parametrize("control", ["stop", "wrap_up"])
def test_stop_during_the_wait_names_the_silent_engine(control):
    from ouroboros.outcomes import REASON_OWNER_REQUESTED_FINALIZATION
    from supervisor.owner_stop import REASON_OWNER_STOPPED_DIRECT_TURN

    reason = REASON_OWNER_STOPPED_DIRECT_TURN if control == "stop" else REASON_OWNER_REQUESTED_FINALIZATION
    usage = {"_last_llm_error_kind": "transport_unavailable", "_last_llm_failure": _daemon_row()}
    text = _provider_terminal_text(usage, is_context_overflow=False, is_transport_wait=True, waited_sec=120.0,
                                   is_deadline_exhausted=False, control_reason=reason, wait_cause="transport_unavailable")
    action = "Stop" if control == "stop" else "Wrap up"
    assert text == (f"⚠️ The owner requested {action} while the local Claudexor engine was not answering. "
                    "The wait ended after 2.0 min; No new summary request was sent. Any files written so far are preserved.")
