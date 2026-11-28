"""One window contract from the runner's current load to the plan a Main turn reuses.

The transport is synthetic (``httpx.MockTransport``): no server, model or socket, so nothing here claims a real
runner. Each scenario keeps ONE plan and ONE usage alive and reads them as a round does (``main_input_anchor``,
then ``measure_main_fit``); none rebuilds the plan from a fresh probe to learn the new fact. The ``_Chats``
scenarios at the end are the exception by design: each owner Chat there is a new conversation with its own plan. What an answer cost
is counted the same whatever its load statement binds: the totals below are the server's own counts.
"""
import json
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from tests._first_contact import ANSWER, _episode, _run
from tests._ollama import TARGET, OllamaHTTP, _compatible_target, chat, completed, install_http as install_http

TOTALS = ("prompt_tokens", "completion_tokens", "cached_tokens", "cache_write_tokens")
BODIES = Path(__file__).parent / "fixtures/size_refusals/bodies"
REFUSAL = json.loads((BODIES / "stand-ollama-0.40.16-native-90k.body.json").read_text())
NUMBERLESS = json.loads((BODIES / "ollama-native.body.json").read_text())
# Synthetic twin of the stand's refusal whose runner states less than the 80k load it went out under. The stand's own
# 80,128 is padding above that load, so a new chat sized by it and one sized by the load alone are the same 80,000; only
# a smaller stated window tells whether the refusal itself reached the next plan.
UNDER_LOAD = {"error": REFUSAL["error"].replace("(80128 tokens)", "(60128 tokens)")}


def _runner(monkeypatch, tmp_path, install_http, server, **windows):
    """Real settings and route resolution for one private Ollama runner that Main and Light share."""
    from ouroboros import config, context_fit as cf

    target = _compatible_target(monkeypatch, TARGET["base_url"])
    model, settings = target["capability_route"]["model"], config.runtime_settings()
    settings.update(OUROBOROS_MODEL_CONTEXT_WINDOWS=dict(windows), OUROBOROS_CONTEXT_MODE="max",
                    OUROBOROS_MODEL_LIGHT=model, USE_LOCAL_LIGHT=False)
    for key, value in {"OUROBOROS_MODEL": model, "OUROBOROS_MODEL_LIGHT": model, "USE_LOCAL_MODEL": "false",
                       "USE_LOCAL_LIGHT": "false", "OUROBOROS_CONTEXT_MODE": "max"}.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr(config, "DATA_DIR", tmp_path / "capability-evidence")
    install_http(server)
    core = cf.ContextCore("SYSTEM", "BIBLE", "", "", "Identity", "Facts", '"Finish"')

    def plan(role="main"):
        return cf.build_context_fit_plan(None, core, {"model": model, "use_local_model": False, "model_role": role},
                                         preferred_mode="max", tool_schemas=[], reasoning_effort="high",
                                         route_resolver=cf.resolve_context_fit_route)

    from ouroboros.llm_ollama import _identity
    return SimpleNamespace(target=target, settings=settings, plan=plan,
                           scope=_identity(dict(target)).removeprefix("ollama:"))


def _frontier(runner):
    from ouroboros import capability_evidence as ce

    return ce._read(ce.canonical_evidence_root()).get("main_windows", {}).get(runner.scope) or {}


def _accepted(runner):
    return [row["accepted"] for row in _frontier(runner).get("observations", []) if row.get("accepted")]


class _Turn:
    """One conversation: one plan and one usage, measured and settled the way ``loop_model_call`` does it."""

    def __init__(self, runner, plan):
        self.runner, self.plan, self.usage, self.round = runner, plan, {"_model_substitutions": []}, 0

    def fit(self):
        from ouroboros.context_anchor import main_input_anchor
        from ouroboros.context_fit import measure_main_fit

        anchor, floor = main_input_anchor(self.usage, self.plan, rewrite_epoch=0)
        return measure_main_fit(self.plan, self.plan.messages_for("max"), [], profile="owner_max", rendered_mode="max",
                                round_id=f"r{self.round}", anchor=anchor, accepted_input_tokens=floor).measurement

    def settle(self, usage):
        """An answered send becomes this conversation's fact, as after ``call_llm_with_retry``."""
        from ouroboros.context_anchor import record_input_anchor
        from ouroboros.llm_pricing import add_usage
        from ouroboros.loop_llm_call import _record_round_cache_facts

        self.round += 1
        add_usage(self.usage, usage)
        _record_round_cache_facts(self.usage, usage, round_idx=self.round, round_id=f"r{self.round}",
                                  model=TARGET["resolved_model"], provider=TARGET["provider"])
        estimate = usage["prompt_tokens"] - 2_000
        self.usage["_context_raw_input_tokens"] = estimate
        record_input_anchor(self.usage, self.plan, round_idx=self.round, estimate_tokens=estimate)
        return usage

    def send(self):
        """This plan's own native send: the load its physical context carries (``llm_routing.configured_window``)."""
        message, usage = chat(target=dict(self.runner.target), window=self.plan.requested_load_window)
        assert message["content"] == "391"
        return self.settle(usage)

    def totals(self):
        return {key: self.usage[key] for key in TOTALS}


@pytest.mark.parametrize("second", [120_000, 80_000], ids=["steady", "reloaded-to-80k"])
def test_a_running_turn_reads_the_runners_current_load_in_its_next_round(tmp_path, monkeypatch, install_http, second):
    """The real loop, Auto. Round one is answered with a tool call while another client reloads the runner at 80k;
    round two measures the SAME plan and must meet the load of that moment, without a probe, a rebind or a read of
    its own. The steady twin sends the same bytes and counts the same usage: capacity never changes what was paid."""
    from ouroboros import capability_evidence as ce, context, context_fit

    proof, main, measured, probes = tmp_path / "repo" / "load-proof.txt", [], [], []

    def answer(body):
        if not body.get("tools"):  # a helper's call is no round of Main
            return completed(message={"role": "assistant", "content": "ok"}, prompt_eval_count=500, eval_count=2)
        main.append(body)
        if len(main) == 1:
            server.window = second  # the runner's load after this answer
            return completed(message={"role": "assistant", "content": "", "tool_calls": [{"function": {
                "name": "read_file", "arguments": {"path": str(proof)}}}]}, prompt_eval_count=40_000, eval_count=7)
        return completed(message={"role": "assistant", "content": ANSWER}, prompt_eval_count=41_000, eval_count=9)

    server = install_http(OllamaHTTP(answer, window=120_000))
    monkeypatch.setattr(ce, "canonical_evidence_root", lambda: tmp_path / "data")  # one store, as an install has
    episode = _episode(tmp_path, monkeypatch, SimpleNamespace(base_url=TARGET["base_url"]))
    proof.write_text("LOAD_CONTRACT_TOOL_PROOF", encoding="utf-8")
    plan = episode.plan
    assert plan.requested_load_window == 0 and plan.window_record.total.tokens == 120_000

    measure, route = context_fit.measure_main_fit, context._context_fit_route

    def measuring(plan, *args, **kwargs):
        disposition = measure(plan, *args, **kwargs)
        measured.append((len(main), plan, disposition.measurement.capacity_total_tokens))
        return disposition

    monkeypatch.setattr(context_fit, "measure_main_fit", measuring)
    monkeypatch.setattr(context, "_context_fit_route", lambda *a, **kw: probes.append(a) or route(*a, **kw))
    reads_before = server.gets.count("/api/ps")
    text, usage, _ = _run(episode)

    assert ANSWER in str(text) and len(main) == 2
    assert not probes  # the plan was never resolved again
    assert all(seen is plan for _, seen, _ in measured) and episode.tools._ctx.context_fit_plan is plan
    assert {capacity for sent, _, capacity in measured if sent == 0} == {120_000}
    assert {capacity for sent, _, capacity in measured if sent == 1} == {second}, measured
    assert server.gets.count("/api/ps") - reads_before == 2 * len(server.posts)  # each send's own two reads, no fit's
    [read] = ce._read(ce.canonical_evidence_root())["main_windows"][plan.window_record.route_fp]["reads"]
    assert read["tokens"] == second
    assert {key: usage[key] for key in TOTALS} == {"prompt_tokens": 81_000, "completion_tokens": 16,
                                                   "cached_tokens": 0, "cache_write_tokens": 0}
    assert usage["_last_round_cache_usage"]["prompt_tokens"] == 41_000
    print("LOAD_CONTRACT_TWO_ROUNDS", {"second": second, "capacities": [(sent, cap) for sent, _, cap in measured],
                                      "posts": len(server.posts), "prompt_tokens": usage["prompt_tokens"]})


def test_one_plan_meets_each_reload_and_a_late_answer_lifts_nothing(tmp_path, monkeypatch, install_http):
    """64k, 32k, 64k on one plan, Auto. The first answer was dispatched under 64k and settles after another sender
    published the 32k reload: it keeps its words, its counts, its density and this conversation's anchor, and is no
    capacity anywhere (the durable store, this turn's accepted floor, the next fit). Each later answer floors only
    the load it was accepted under."""
    from ouroboros import capability_evidence as ce
    from ouroboros.llm_ollama_policy import record_window

    density, witness = [], ce.record_token_density
    monkeypatch.setattr(ce, "record_token_density",
                        lambda *args, **kw: density.append(kw["prompt_tokens"]) or witness(*args, **kw))
    server = OllamaHTTP(window=64_000)
    runner = _runner(monkeypatch, tmp_path, install_http, server)
    turn = _Turn(runner, runner.plan())
    plan = turn.plan
    assert plan.requested_load_window == 0 and turn.fit().window["tokens"] == 64_000
    start_reads = server.gets.count("/api/ps")

    def reloaded_while_answering(_body):
        server.window = 32_000
        record_window(dict(runner.target), 32_000)  # the sender whose request reloaded the runner
        return completed(prompt_eval_count=64_042, eval_count=6)  # 42 tokens of runner padding

    server.reply = reloaded_while_answering
    late = turn.send()
    assert late["ollama_load_statement"] == [runner.scope, 1] and _frontier(runner)["reads"][0]["sequence"] == 2
    assert (late["prompt_tokens"], late["completion_tokens"], late["capacity_input_tokens"]) == (64_042, 6, 64_000)
    fit = turn.fit()
    assert fit.window["tokens"] == 32_000  # the load of this round, not the one the plan was built under
    assert fit.accepted_input_tokens is None and not _accepted(runner)
    assert fit.anchor_input_tokens == 64_042 and set(density) == {64_042}  # its counts still measure the model

    server.reply = completed(prompt_eval_count=20_000, eval_count=5)
    assert turn.send()["ollama_load_statement"] == [runner.scope, 2]
    fit = turn.fit()
    assert (fit.window["tokens"], fit.accepted_input_tokens, _accepted(runner)) == (32_000, 20_000, [20_000])

    server.window, server.reply = 64_000, completed(prompt_eval_count=45_000, eval_count=4)
    assert turn.fit().window["tokens"] == 32_000  # a reload no send has read yet: a fit never asks the runner
    sent = turn.send()
    assert sent["ollama_load_statement"] == [runner.scope, _frontier(runner)["reads"][0]["sequence"]]
    fit = turn.fit()
    assert (fit.window["tokens"], fit.accepted_input_tokens) == (64_000, 45_000)
    assert turn.plan is plan and plan.window_record.total.tokens == 64_000  # one plan, never rebuilt
    assert turn.totals() == {"prompt_tokens": 64_042 + 20_000 + 45_000, "completion_tokens": 15,
                             "cached_tokens": 0, "cache_write_tokens": 0}
    assert server.gets.count("/api/ps") - start_reads == 2 * len(server.posts)  # each send's own two reads


@pytest.mark.parametrize("rival", [32_000, 64_000], ids=["newer-load", "same-load"])
def test_an_auto_read_overtaken_by_a_newer_statement_publishes_nothing(install_http, monkeypatch, rival):
    """Auto: the read before a send and its publication are one version. The runner reloads right after this
    send's ``/api/ps`` answered and another sender publishes that load first: the overtaken read is not written
    over it, and the answer it led to carries no number. An equal statement is no rival."""
    from ouroboros import capability_evidence as ce
    from ouroboros.llm_ollama import _identity
    from ouroboros.llm_ollama_policy import record_window

    server, rivaled, written = OllamaHTTP(completed(prompt_eval_count=20_000), window=64_000), [], []
    scope, save = _identity(dict(TARGET)).removeprefix("ollama:"), ce._save

    def handle(request):
        response = server(request)
        if request.url.path == "/api/ps" and not rivaled:
            rivaled.append(record_window(dict(TARGET), rival))
            server.window = rival
        return response

    def saving(root, data):
        stated = [read["tokens"] for read in data.get("main_windows", {}).get(scope, {}).get("reads", [])]
        if written[-1:] != [stated]:
            written.append(stated)
        return save(root, data)

    install_http(handle)
    monkeypatch.setattr(ce, "_save", saving)
    message, usage = chat(window=0)
    assert message["content"] == "391" and (usage["prompt_tokens"], usage["completion_tokens"]) == (20_000, 4)
    assert rivaled == [1] and written == [[rival]]  # one statement was ever published: the rival's
    [read] = ce._read(ce.canonical_evidence_root())["main_windows"][scope]["reads"]
    assert (read["tokens"], read["sequence"]) == (rival, 1)
    assert usage["ollama_load_statement"] == [scope, 1 if rival == 64_000 else None]
    assert usage["ollama_context"]["loaded"] == rival and usage["capacity_input_tokens"] == 20_000


def _auto_turn_at_80k(monkeypatch, tmp_path, install_http, server):
    """One Auto conversation on a runner loaded at 80k, and the order of the statement its sends go out under."""
    runner = _runner(monkeypatch, tmp_path, install_http, server)
    turn = _Turn(runner, runner.plan())
    assert turn.plan.requested_load_window == 0 and turn.fit().window["tokens"] == 80_000
    return runner, turn, _frontier(runner)["reads"][0]["sequence"]


@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize("later", [120_000, 80_000], ids=["80k-then-120k", "steady-80k"])
def test_an_auto_answer_stays_with_the_load_it_was_sent_under(tmp_path, monkeypatch, install_http, later, asynchronous):
    """Auto at 80k. While the answer is generated another client reloads the runner at 120k, first shown by this
    send's closing ``/api/ps``. The answer keeps its words and every count but stays bound to the 80k statement it
    went out under: its 42 tokens of runner padding are no capacity, and nothing it accepted floors the new load.
    The 120k observation is kept, so the next round of the same plan meets it. Steady, the same answer is current
    and is the 80k load's acceptance (the control)."""
    server = OllamaHTTP(window=80_000)
    runner, turn, dispatched = _auto_turn_at_80k(monkeypatch, tmp_path, install_http, server)

    def reloaded_while_answering(_body):
        server.window = later
        return completed(prompt_eval_count=80_042, eval_count=6)

    server.reply = reloaded_while_answering
    message, usage = chat(asynchronous=asynchronous, target=dict(runner.target), window=0)
    turn.settle(usage)
    assert message["content"] == "391" and (usage["prompt_tokens"], usage["completion_tokens"]) == (80_042, 6)
    assert usage["ollama_load_statement"] == [runner.scope, dispatched]
    assert (usage["ollama_context"]["loaded"], usage["capacity_input_tokens"]) == (80_000, 80_000)
    [read], fit = _frontier(runner)["reads"], turn.fit()
    assert read["tokens"] == fit.window["tokens"] == later and fit.anchor_input_tokens == 80_042
    if later == 80_000:
        assert read["sequence"] == dispatched and fit.accepted_input_tokens == 80_000 and _accepted(runner) == [80_000]
    else:
        assert read["sequence"] > dispatched and fit.accepted_input_tokens is None and not _accepted(runner)
    assert turn.totals() == {"prompt_tokens": 80_042, "completion_tokens": 6, "cached_tokens": 0,
                             "cache_write_tokens": 0}


@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize("later", [120_000, 80_000], ids=["80k-then-120k", "steady-80k"])
def test_an_auto_refusal_stays_with_the_load_it_was_sent_under(tmp_path, monkeypatch, install_http, later, asynchronous):
    """The refusal twin. A 90k request refused by the 80k load it went out under settles after another client
    reloaded the runner at 120k, first shown by its closing read. The refusal keeps the server's own numbers and
    names the 80k statement; that statement is no longer the runner's, so it caps nothing and the next round of
    the same plan meets 120k. Steady, the same refusal is the 80k load's."""
    from ouroboros import context_window as cw
    from ouroboros.size_refusal import stamp_failure

    server = OllamaHTTP(window=80_000, status=400)
    runner, turn, dispatched = _auto_turn_at_80k(monkeypatch, tmp_path, install_http, server)

    def reloaded_while_refusing(_body):
        server.window = later
        return REFUSAL

    server.reply = reloaded_while_refusing
    with pytest.raises(httpx.HTTPStatusError) as refused:
        chat(asynchronous=asynchronous, target=dict(runner.target), window=0)
    assert (refused.value.nominal_context_window, refused.value.ollama_load_statement) == (80_000, (runner.scope, dispatched))
    fact = stamp_failure(turn.usage, refused.value)
    assert (fact.requested, fact.window, fact.token_window, fact.ollama_read_sequence) == (90_068, 80_128, 80_000, dispatched)
    cw.record_refusal(turn.usage, turn.plan)
    refusals = [row["refused"] for row in _frontier(runner).get("observations", []) if row.get("refused")]
    assert _frontier(runner)["reads"][0]["tokens"] == later
    assert refusals == ([80_000] if later == 80_000 else [])
    if later == 120_000:
        assert turn.fit().window["tokens"] == 120_000


def test_a_role_cap_outlives_every_acceptance_on_the_shared_load(tmp_path, monkeypatch, install_http):
    """A 100k input was accepted on the shared 120k load; then the owner caps Main at 80k while Light keeps 120k.
    The load statement does not change, so that acceptance is current evidence of the runner. It is none against
    the role's own cap: Main stays at 80k at its start, in its second round and after Light's larger answer."""
    server = OllamaHTTP(completed(prompt_eval_count=100_000, eval_count=8), window=4096)
    runner = _runner(monkeypatch, tmp_path, install_http, server, main=120_000, light=120_000)
    early = _Turn(runner, runner.plan())
    assert early.plan.requested_load_window == 120_000 and early.fit().window["tokens"] == 120_000
    early.send()
    assert _accepted(runner) == [100_000] and server.posts[-1]["options"]["num_ctx"] == 120_000

    runner.settings["OUROBOROS_MODEL_CONTEXT_WINDOWS"] = {"main": 80_000, "light": 120_000}
    main, light = _Turn(runner, runner.plan()), _Turn(runner, runner.plan("light"))
    assert main.plan.requested_load_window == 120_000 and light.plan.model_role == "light"
    server.reply = completed(prompt_eval_count=70_000, eval_count=3)
    for _round in (1, 2):
        window = main.fit().window
        assert (window["tokens"], window["source"]) == (80_000, "user_setting") and window["admitted"] < 80_000
        main.send()
        assert server.posts[-1]["options"]["num_ctx"] == 120_000  # the runner keeps the shared load
    assert light.fit().window["tokens"] == 120_000
    server.reply = completed(prompt_eval_count=110_000, eval_count=3)
    light.send()
    assert max(_accepted(runner)) == 110_000
    assert main.fit().window["tokens"] == 80_000 and light.fit().window["tokens"] == 120_000
    assert main.totals()["prompt_tokens"] == 140_000 and light.totals()["prompt_tokens"] == 110_000


def test_what_the_old_80k_load_refused_does_not_press_a_new_120k_choice(tmp_path, monkeypatch, install_http):
    """The other direction. Main alone at 80k was refused by size; the owner then chooses 120k. Neither the refusal
    recorded under the old load nor one of its sends that settles after the new choice was published caps the plan
    of the new choice, in its first round or its next."""
    from ouroboros import context_window as cw
    from ouroboros.llm_ollama_policy import record_window
    from ouroboros.size_refusal import stamp_failure

    server = OllamaHTTP(REFUSAL, status=400)
    runner = _runner(monkeypatch, tmp_path, install_http, server, main=80_000)
    old = _Turn(runner, runner.plan())
    assert old.plan.requested_load_window == 80_000 and old.fit().window["tokens"] == 80_000
    with pytest.raises(httpx.HTTPStatusError) as refused:
        chat(target=dict(runner.target), window=80_000)
    stamp_failure(old.usage, refused.value)
    cw.record_refusal(old.usage, old.plan)
    assert [row["refused"] for row in _frontier(runner)["observations"]] == [80_000]

    runner.settings["OUROBOROS_MODEL_CONTEXT_WINDOWS"] = {"main": 120_000}
    new = _Turn(runner, runner.plan())
    assert new.plan.requested_load_window == 120_000 and new.fit().window["tokens"] == 120_000

    def new_choice_published_first(_body):
        record_window(dict(runner.target), 120_000, requested=120_000)
        return REFUSAL

    server.reply = new_choice_published_first
    with pytest.raises(httpx.HTTPStatusError) as late:
        chat(target=dict(runner.target), window=80_000)  # the old plan's send, still in flight
    stamp_failure(old.usage, late.value)
    cw.record_refusal(old.usage, old.plan, refused_input=90_068)
    assert [row.get("refused") for row in _frontier(runner)["observations"]] == [80_000]  # nothing new recorded
    assert _frontier(runner)["reads"][0]["tokens"] == 120_000
    window = new.fit().window
    assert (window["tokens"], window["source"]) == (120_000, "server_metadata")


def test_an_unloaded_or_unread_runner_is_no_capacity(tmp_path, monkeypatch, install_http):
    """Auto. A send whose read finds the model unloaded retires the statement before it goes out, so the plan's
    next round reads the buffer that send's own answer discovered. An answer whose closing read failed states no
    load: it is paid and anchored, and neither published nor accepted as capacity."""
    server = OllamaHTTP(completed(prompt_eval_count=3_000, eval_count=4), window=64_000)
    runner = _runner(monkeypatch, tmp_path, install_http, server)
    turn = _Turn(runner, runner.plan())
    assert turn.fit().window["tokens"] == 64_000

    server.loaded, server.window = False, 4_096  # unloaded; the next request loads the runner's default buffer
    assert turn.send()["ollama_context"]["loaded"] == 4_096
    assert [read["tokens"] for read in _frontier(runner)["reads"]] == [4_096]
    fit = turn.fit()
    assert (fit.window["tokens"], fit.accepted_input_tokens) == (4_096, 3_000)

    reads = []

    def closing_read_fails(request):
        if request.url.path == "/api/ps":
            reads.append(request)
            if len(reads) == 2:
                return httpx.Response(503, json={"error": "busy"})
        return server(request)

    install_http(closing_read_fails)
    server.reply = completed(prompt_eval_count=3_500, eval_count=4)
    unread = turn.send()
    assert unread["ollama_context"] == {"requested": None, "loaded": None, "read": "unavailable"}
    assert (unread["prompt_tokens"], unread["capacity_input_tokens"]) == (3_500, 0)
    fit = turn.fit()
    assert (fit.window["tokens"], fit.accepted_input_tokens, fit.anchor_input_tokens) == (4_096, 3_000, 3_500)
    assert _accepted(runner) == [3_000] and turn.totals()["prompt_tokens"] == 6_500


@pytest.mark.parametrize("extension", [
    {"ollama_load_statement": 7}, {"ollama_load_statement": "runner"}, {"ollama_load_statement": ["scope", 3]},
    {"ollama_load_statement": {"scope": 1}, "ollama_context": {"loaded": 999_999}, "capacity_input_tokens": 1},
], ids=["number", "text", "pair", "mapping"])
def test_a_compatible_servers_same_named_usage_fields_are_no_host_observation(tmp_path, monkeypatch, extension):
    """An ordinary ``/v1`` answer whose usage happens to carry the native adapter's own field names keeps its
    words and counts, loses those fields at the lane's normalization, and settles like any ordinary provider's:
    its acceptance is written, under no load statement."""
    from ouroboros import capability_evidence as ce
    from ouroboros.context_anchor import main_input_anchor, record_input_anchor
    from ouroboros.llm import LLMClient
    from ouroboros.loop_llm_call import _record_round_cache_facts

    monkeypatch.setattr(ce, "canonical_evidence_root", lambda: tmp_path)
    target = {**TARGET, "base_url": "https://gateway.example/v1"}
    reply = {"id": "chatcmpl-1", "model": "mock-model", "choices": [{"index": 0, "finish_reason": "stop",
             "message": {"role": "assistant", "content": "ordinary"}}],
             "usage": {"prompt_tokens": 60_000, "completion_tokens": 3, "total_tokens": 60_003, **extension}}
    message, usage = LLMClient()._normalize_remote_response(reply, target, skip_cost_fetch=True)
    assert message["content"] == "ordinary" and (usage["prompt_tokens"], usage["completion_tokens"]) == (60_000, 3)
    assert not {"ollama_load_statement", "ollama_context", "capacity_input_tokens"} & set(usage)

    plan = SimpleNamespace(provider=TARGET["provider"], model=TARGET["requested_model"], route_fp="gateway-scope",
                           window_record=None, window_witness_bound=True, output_reserve_tokens=8_192, view_kind="")
    accumulated = {"_context_raw_input_tokens": 58_000, "_model_substitutions": []}
    _record_round_cache_facts(accumulated, usage, round_idx=1, round_id="r1", model="mock-model",
                              provider=TARGET["provider"])
    assert "ollama_load_statement" not in accumulated["_last_round_cache_usage"]
    record_input_anchor(accumulated, plan, round_idx=1, estimate_tokens=58_000)
    anchor, floor = main_input_anchor(accumulated, plan, rewrite_epoch=0)
    assert (anchor.input_tokens, floor) == (60_000, 60_000)
    [row] = ce._read(tmp_path)["main_windows"]["gateway-scope"]["observations"]
    assert row["accepted"] == 60_000
    # A lane that passes its provider's usage through whole reaches the round's projection with them: no failure.
    for carried in ({"prompt_tokens": 10, **extension}, {"prompt_tokens": 10, "ollama_load_statement": True}):
        _record_round_cache_facts({}, carried, round_idx=1, round_id="r2", model="m", provider="local")


class _Chats:
    """One installation's owner Chat on one native runner: each ``chat`` is a new direct conversation of the real loop
    (its own plan, the one evidence store), whose first Main send meets ``refuse`` when given; later sends answer."""

    def __init__(self, tmp_path, monkeypatch, install_http, window):
        from ouroboros import capability_evidence as ce, context_fit

        self.tmp_path, self.monkeypatch, self.main, self.fits, self.refuse = tmp_path, monkeypatch, [], [], None
        self.server = install_http(OllamaHTTP(self._answer, window=window))
        monkeypatch.setattr(ce, "canonical_evidence_root", lambda: tmp_path / "data")  # the install's one store
        measure = context_fit.measure_main_fit

        def measuring(plan, *args, **kwargs):
            disposition = measure(plan, *args, **kwargs)
            self.fits.append((len(self.main), disposition.measurement))
            return disposition

        monkeypatch.setattr(context_fit, "measure_main_fit", measuring)

    def _answer(self, body):
        if not body.get("tools"):  # a helper's call is no round of Main
            return completed(message={"role": "assistant", "content": "ok"}, prompt_eval_count=500, eval_count=2)
        self.main.append(body)
        if self.refuse is not None and len(self.main) == 1:
            return httpx.Response(400, json=self.refuse)
        return completed(message={"role": "assistant", "content": ANSWER}, prompt_eval_count=30_000, eval_count=9)

    def chat(self, name, refuse=None):
        self.main.clear(), self.fits.clear()
        self.refuse = refuse
        episode = _episode(self.tmp_path, self.monkeypatch, SimpleNamespace(base_url=TARGET["base_url"]),
                           chat=name, direct=True)
        text, usage, _ = _run(episode)
        assert ANSWER in str(text) and len(self.main) == (2 if refuse else 1), (text, len(self.main))
        return episode.plan, usage

    def first_fit(self):
        """The first round's measurement of this conversation: the window its first send was sized under."""
        return next(measurement for sent, measurement in self.fits if sent == 0)

    def store(self):
        from ouroboros import capability_evidence as ce

        return ce._read(ce.canonical_evidence_root())

    def window(self):
        return self.store().get("main_windows", {})

    def refused_sends(self):
        log = self.tmp_path / "data" / "logs" / "events.jsonl"
        rows = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()] if log.exists() else []
        return [row["refused_send"] for row in rows if "refused_send" in row]


@pytest.mark.parametrize("body,stated,limit", [(REFUSAL, (90_068, 80_128), 80_000), (UNDER_LOAD, (90_068, 60_128), 60_128),
                                               (NUMBERLESS, (0, 0), 80_000)],
                         ids=["stand-numbered", "stated-under-load", "numberless"])
def test_a_new_chat_is_sized_by_what_the_same_80k_load_refused(tmp_path, monkeypatch, install_http, body, stated, limit):
    """The owner's real Chat path on one native runner at 80k (Auto). The first chat's first send is refused; the
    real loop stamps it (``stamp_failure``), records it (``record_refusal``) and recovers in that turn. A new chat on
    the same load is then planned under what the refusal established, with the whole reply reserve charged, and its
    first send is accepted. The refusal is bound to the load statement it went out under. Its Logs row keeps the
    host's count of that send apart from the runner's answer. A numberless refusal is the load's own: no input or
    smaller window is made up for it. No native plan takes the first-send hint of HTTP Chat routes (#3896), which
    would send the refusal down that hint's numberless path and lose the stated window (the under-load twin)."""
    from ouroboros import size_refusal

    chats, stamped = _Chats(tmp_path, monkeypatch, install_http, 80_000), []
    stamp = size_refusal.stamp_failure
    monkeypatch.setattr(size_refusal, "stamp_failure",
                        lambda usage, exc, *a, **kw: stamped.append(stamp(usage, exc, *a, **kw)) or stamped[-1])
    plan, _usage = chats.chat("chat-1", refuse=body)
    assert plan.first_contact == "" and plan.requested_load_window == 0
    [scope] = chats.window()
    [read] = chats.window()[scope]["reads"]
    [fact] = [fact for fact in stamped if fact is not None]
    assert (fact.nominal_window, fact.ollama_scope, fact.ollama_read_sequence) == (80_000, scope, read["sequence"])
    assert (fact.requested, fact.window, fact.token_window) == (*stated, limit)  # the runner's own words, the load apart
    refusals = [row for row in chats.window()[scope]["observations"] if row.get("refused")]
    assert [(row["refused"], row["kind"]) for row in refusals] == [(limit, "total")]
    assert refusals[0]["sequence"] > read["sequence"]  # recorded while the statement it went out under is current
    [sent] = chats.refused_sends()  # #4077: the host's count of that send, the runner's answer, and not the load
    assert sent["host"]["input"] > 0 and sent["provider_counts"] == (
        {"input": stated[0], "input_given": "stated", "window": stated[1]} if stated[0] else {})
    assert "main_first_contact" not in chats.store()  # nothing for another chat's first-send hint

    plan, _usage = chats.chat("chat-2")
    record, first = plan.window_record, chats.first_fit()
    assert plan.first_contact == "" and (record.total.tokens, record.total.source) == (
        limit, "refusal" if limit < 80_000 else "server_metadata")
    assert (first.capacity_total_tokens, first.window["admitted"]) == (limit, limit - 8_192)
    assert first.estimated_input_tokens + first.reply_allowance_tokens <= limit
    print("NEW_CHAT_AFTER_REFUSAL", {"limit": limit, "first_input": first.estimated_input_tokens,
                                     "reply": first.reply_allowance_tokens, "provider": sent["provider_counts"]})


def test_an_80k_refusal_caps_neither_a_120k_reload_nor_a_new_80k_load(tmp_path, monkeypatch, install_http):
    """What an 80k load refused at 60,128 sizes the next chat on that load only. Another client reloads the runner
    at 120k: the next chat's first send reads it, and the chat after is planned at 120k. Then it is reloaded at 80k
    again, a new statement with the same number: the old refusal does not return with it."""
    chats = _Chats(tmp_path, monkeypatch, install_http, 80_000)
    chats.chat("refused-at-80k", refuse=UNDER_LOAD)
    plan, _usage = chats.chat("same-80k")
    assert (plan.window_record.total.tokens, chats.first_fit().capacity_total_tokens) == (60_128, 60_128)

    chats.server.window = 120_000  # another client's reload; this install has not read it yet
    chats.chat("reload-observed")
    [scope] = chats.window()
    assert chats.window()[scope]["reads"][0]["tokens"] == 120_000
    plan, _usage = chats.chat("at-120k")
    first = chats.first_fit()
    assert (first.capacity_total_tokens, first.window["source"]) == (120_000, "server_metadata")

    chats.server.window = 80_000
    chats.chat("reload-to-80k-observed")
    reads = chats.window()[scope]["reads"]
    assert reads[0]["tokens"] == 80_000 and reads[0]["sequence"] > max(
        row["sequence"] for row in chats.window()[scope].get("observations", []) if row.get("refused"))
    plan, _usage = chats.chat("at-80k-again")
    first = chats.first_fit()
    assert (first.capacity_total_tokens, first.window["source"], first.window["admitted"]) == (
        80_000, "server_metadata", 71_808)
