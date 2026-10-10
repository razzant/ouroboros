"""A route that refuses a request carrying images gets one same-round retry without them.

A refusal is an observation about one request (an unsupported image, a size or
count limit, a content filter look the same), never a fact about the route. So a
completed 400/422 or 404/415 of a Main request that physically carried images, or
a 5xx once the ordinary same-request retries used their whole attempt budget,
earns ONE retry in the same round on the same model, route, account and output
reserve, with only the refused images projected as text (Auto: a caption from a
route that did not refuse them, else a marker; Inline: a marker). Success keeps
that memory for the rest of the task, never on disk; failure keeps both errors and
learns nothing. Auth, quota, rate limits, policy, context overflow, deadlines and
unknown outcomes never earn it. A VLM or caption refusal is remembered the same way
without a retry, and a route that refused an image is not offered it again.

Every provider call here is a real physical attempt of its own final candidate
(reservation, sealing, the dispatch predicate, the ledger): only the provider's
answer is scripted.
"""

from __future__ import annotations

import copy
import json
import socket
import sys
from dataclasses import replace

import httpx
import pytest

from ouroboros import capability_evidence, fallback_cooldown, loop, loop_llm_call, usage_accounting as ua
from ouroboros import vision_routing as vr
from ouroboros.llm import LLMClient, ProviderPolicyRefusal
from ouroboros.llm_attempt import _attempt_request, _candidate_before_dispatch
from ouroboros.loop_llm_call import RETRY_ATTEMPTS_SPENT_KEY, RETRY_WALL_EXHAUSTED_KEY, TRANSPORT_DEATHS_KEY
from ouroboros.tools.registry import ToolRegistry

MAIN = "openai::acme-vision-1"
CAPTIONER = "openai::acme-captioner-1"
IMAGE_A = "data:image/png;base64,QUFBQQ=="
IMAGE_B = "data:image/png;base64,QkJCQg=="
DIGEST_A, DIGEST_B = vr._url_digest(IMAGE_A), vr._url_digest(IMAGE_B)
WORDS = "Image input is not supported for this model"


class _ProviderError(Exception):
    """A completed HTTP error answer, shaped like the SDK's status errors."""

    def __init__(self, status: int, message: str = WORDS, code: str = "image_not_supported"):
        super().__init__(f"Error code: {status} - {message}")
        self.status_code = status
        self.body = {"error": {"message": message, "code": code}}


def _died():
    error = RuntimeError("Connection error.")
    error.__cause__ = httpx.ReadError("socket died after dispatch")
    return error


class _Policy(ProviderPolicyRefusal):
    """Fixture refusal: the host policy permits no connection for this call."""


class _Provider:
    """``llm.chat`` whose every call is a real physical attempt of its own final candidate."""

    def __init__(self, root, answer):
        self.root, self.answer = root, answer
        self.sent, self.texts, self.captions = [], [], []

    def default_model(self):
        return MAIN

    def vision_query(self, _prompt, _images, **kwargs):
        self.captions.append(kwargs.get("model"))
        return "a red square", {"cost": 0.0}

    def chat(self, **kwargs):
        target = LLMClient()._resolve_remote_target(kwargs["model"])
        candidate = {"model": target["resolved_model"], "messages": copy.deepcopy(kwargs["messages"]),
                     "max_tokens": kwargs["max_tokens"]}
        request = replace(_attempt_request(target, candidate), drive_root=self.root, task_id="t-img",
                          root_task_id="t-img", reservation_usd=0.01)

        def send():
            self.sent.append(vr.candidate_images(request.candidate_raw_sha256))
            self.texts.append(json.dumps(candidate["messages"]))
            outcome = self.answer(len(self.sent), self.sent[-1])
            if isinstance(outcome, BaseException):
                raise outcome
            return {"usage": {"prompt_tokens": 1, "completion_tokens": 1}}

        ua.execute_physical_attempt(request, send, before_dispatch=_candidate_before_dispatch(candidate, request),
                                    extractor=lambda _r: ({"prompt_tokens": 1, "completion_tokens": 1}, 0.0, True))
        return ({"role": "assistant", "content": "done"},
                {"prompt_tokens": 1, "completion_tokens": 1, "cost": 0.0, "provider": "openai"})


def _refuse_images(error=lambda: _ProviderError(400), only=None):
    """Refuse every send that carries an image (or only ``only``); answer the rest."""
    return lambda _n, images: error() if images and (only is None or only in images) else "ok"


@pytest.fixture
def root(monkeypatch, tmp_path):
    data = tmp_path / "data"
    (data / "state").mkdir(parents=True)
    (data / "logs").mkdir(parents=True)
    for key, value in {
        "OPENAI_API_KEY": "k", "OPENROUTER_API_KEY": "k", "OUROBOROS_DATA_DIR": str(data),
        "OUROBOROS_SETTINGS_PATH": str(data / "settings.json"), "TOTAL_BUDGET": "100",
        "OUROBOROS_IMAGE_INPUT_MODE": "auto", "OUROBOROS_MODEL": MAIN, "OUROBOROS_MODEL_LIGHT": "",
        "OUROBOROS_MODEL_VISION": "", "OUROBOROS_MODEL_FALLBACKS": "",
    }.items():
        monkeypatch.setenv(key, value)
    monkeypatch.delenv("OUROBOROS_MODEL_FALLBACK", raising=False)

    def no_network(*_args, **_kwargs):
        raise OSError("network disabled in the image refusal contract")

    monkeypatch.setattr(socket.socket, "connect", no_network)
    monkeypatch.setattr(socket, "getaddrinfo", no_network)
    monkeypatch.setattr(loop, "_server_web_allowed_by_task", lambda _ctx: False)
    monkeypatch.setattr(loop_llm_call, "_sleep_within_deadline", lambda *_a, **_k: True)
    return data


def _messages(*urls):
    return [{"role": "system", "content": "SYS"},
            {"role": "user", "content": [{"type": "text", "text": "What is on the picture?"},
                                         *({"type": "image_url", "image_url": {"url": url}} for url in urls)]}]


def _round(root, provider, *, messages=None, usage=None, round_idx=1, progress=None, plan=None):
    tools = ToolRegistry(repo_dir=root.parent, drive_root=root)
    tools._ctx.task_id = "t-img"
    return loop._RoundModelCallContext(
        llm=provider, messages=_messages(IMAGE_A) if messages is None else messages,
        tools=tools, context_fit_plan=plan,
        active_model=MAIN, tool_schemas=[], active_effort="medium", max_retries=3,
        drive_logs=root / "logs", task_id="t-img", round_idx=round_idx, event_queue=None,
        accumulated_usage={} if usage is None else usage, task_type="task", active_use_local=False,
        active_context_mode="max", drive_root=root, model_role="main",
        emit_progress=progress.append if progress is not None else None)


def _events(root, kind):
    path = root / "logs" / "events.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []
    return [row for row in rows if row.get("type") == kind or row.get("checkpoint_kind") == kind]


def _ledger(root):
    """Each attempt's current state, in write order (the usage store keeps one row per attempt)."""
    from tests._usage_store_testing import ledger_rows

    return {row["attempt_id"]: row["state"] for row in ledger_rows(root) if row.get("kind", "attempt") == "attempt"}


def _main_plan():
    from tests.test_context_fit_integration import _plan

    return replace(_plan(preferred="max"), model=MAIN, provider="openai")


# ---------------------------------------------------------------------------
# The physical fact: which images a final candidate carried
# ---------------------------------------------------------------------------

def test_a_final_candidate_records_the_images_it_carries_in_every_wire_shape(root):
    client = LLMClient()
    openai_target = client._resolve_remote_target(MAIN)
    with_image = {"model": "acme-vision-1", "messages": _messages(IMAGE_A, IMAGE_B), "max_tokens": 8}
    request = _attempt_request(openai_target, with_image)
    assert vr.candidate_images(request.candidate_raw_sha256) == {DIGEST_A, DIGEST_B}

    # Anthropic splits a data URL into media type and data: the same image, the same key.
    anthropic_target = client._resolve_remote_target("anthropic::acme-claude-1")
    native = client._build_remote_candidate(anthropic_target, _messages(IMAGE_A), "high", 128, "auto", None, None)
    assert json.dumps(native).count("QUFBQQ==") == 1 and IMAGE_A not in json.dumps(native)
    assert vr.candidate_images(_attempt_request(anthropic_target, native).candidate_raw_sha256) == {DIGEST_A}

    text_only = _attempt_request(openai_target, {"model": "acme-vision-1", "messages": _messages(), "max_tokens": 8})
    assert vr.candidate_images(text_only.candidate_raw_sha256) == frozenset()
    assert vr.candidate_images("never-recorded") is None


# ---------------------------------------------------------------------------
# The trigger: a completed structural refusal of a request that carried images
# ---------------------------------------------------------------------------

def _capture(status, *, images=True, state="unresolved"):
    payload = {"model": "m", "messages": _messages(IMAGE_A) if images else _messages(), "max_tokens": 8}
    request = _attempt_request(LLMClient()._resolve_remote_target(MAIN), payload)
    return ua.PhysicalAttemptCapture(
        attempt_id="refused", model=request.model, provider=request.provider, state=state,
        candidate_measurement_kind="canonical_json_v1", candidate_raw_sha256=request.candidate_raw_sha256,
        provider_status_code=status, provider_code="image_not_supported")


SPENT = {RETRY_WALL_EXHAUSTED_KEY: True, RETRY_ATTEMPTS_SPENT_KEY: True}


@pytest.mark.parametrize("kind,status,extra,earns", [
    ("bad_request", 400, {}, True),
    ("bad_request", 422, {}, True),
    ("provider_error", 404, {}, True),
    ("provider_error", 415, {}, True),
    ("provider_transient", 503, SPENT, True),
    ("provider_transient", 503, {RETRY_WALL_EXHAUSTED_KEY: True, RETRY_ATTEMPTS_SPENT_KEY: False}, False),
    ("provider_transient", 503, {}, False),
    ("provider_transient", 429, SPENT, False),
    ("auth_error", 401, {}, False),
    ("auth_error", 403, {}, False),
    ("quota_exhausted", 402, {}, False),
    ("bad_request", 413, {}, False),
    ("provider_error", 400, {}, False),
    ("context_overflow", 400, {}, False),
    ("provider_policy_refusal", None, {}, False),
    ("deadline_exhausted", None, {}, False),
    ("provider_outcome_unknown", None, {}, False),
    ("bad_request", 400, {TRANSPORT_DEATHS_KEY: {"round_id": "r", "count": 1}}, False),
    ("bad_request", 400, {"_pending_transport_outcome": {"outcome": "unknown"}}, False),
], ids=lambda value: str(value))
def test_only_a_completed_structural_refusal_earns_the_retry(root, kind, status, extra, earns):
    usage = {"_last_llm_error_kind": kind, "_last_llm_provider_message": WORDS, **extra}
    facts = vr.image_refusal(usage, _capture(status))
    assert (facts is not None) is earns
    if earns:
        assert facts["status"] == status and facts["message"] == WORDS and facts["digests"] == {DIGEST_A}


def test_the_trigger_needs_images_in_the_physical_candidate_not_the_money_state(root):
    usage = {"_last_llm_error_kind": "bad_request"}
    assert vr.image_refusal(usage, _capture(400, images=False)) is None
    # A completed 400 is "unresolved" in the ledger too; the money state says nothing about the outcome.
    assert vr.image_refusal(usage, _capture(400, state="unresolved")) is not None
    assert vr.image_refusal(usage, _capture(400, state="settled")) is not None


def test_the_retry_wall_tells_a_spent_attempt_budget_from_a_deadline_stop(root, tmp_path, monkeypatch):
    def failing(**_kwargs):
        raise _ProviderError(503, "upstream overloaded", "server_error")

    llm = type("LLM", (), {"chat": staticmethod(failing), "default_model": lambda self: MAIN})()
    spent: dict = {}
    loop_llm_call.call_llm_with_retry(llm, _messages(), MAIN, None, "low", 3, tmp_path, "t", 1, None, spent, "task")
    assert spent[RETRY_WALL_EXHAUSTED_KEY] is True and spent[RETRY_ATTEMPTS_SPENT_KEY] is True

    # One attempt is no series: a 5xx blip on a fallback-chain candidate (attempt_cap=1) keeps its pixels.
    single: dict = {}
    loop_llm_call.call_llm_with_retry(llm, _messages(), MAIN, None, "low", 3, tmp_path, "t", 1, None, single, "task",
                                      attempt_cap=1)
    assert single[RETRY_WALL_EXHAUSTED_KEY] is True and single[RETRY_ATTEMPTS_SPENT_KEY] is False
    assert vr.image_refusal({**single, "_last_llm_error_kind": "provider_transient"}, _capture(503)) is None

    monkeypatch.setattr(loop_llm_call, "_sleep_within_deadline", lambda *_a, **_k: False)
    stopped: dict = {}
    loop_llm_call.call_llm_with_retry(llm, _messages(), MAIN, None, "low", 3, tmp_path, "t", 1, None, stopped, "task")
    assert stopped[RETRY_WALL_EXHAUSTED_KEY] is True and stopped[RETRY_ATTEMPTS_SPENT_KEY] is False
    # The marker never outlives its invocation: a usable answer clears both twins.
    ok = {"_llm_retry_attempts_spent": True}
    llm_ok = type("LLM", (), {"chat": staticmethod(lambda **_kw: ({"role": "assistant", "content": "fine"}, {})),
                              "default_model": lambda self: MAIN})()
    loop_llm_call.call_llm_with_retry(llm_ok, _messages(), MAIN, None, "low", 3, tmp_path, "t", 1, None, ok, "task")
    assert RETRY_ATTEMPTS_SPENT_KEY not in ok and RETRY_WALL_EXHAUSTED_KEY not in ok


# ---------------------------------------------------------------------------
# The retry
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("with_plan", [False, True], ids=["no-fit-plan", "fit-plan"])
def test_refusal_then_text_retry_answers_on_the_same_model_and_remembers_only_that_image(
        root, monkeypatch, with_plan):
    progress: list = []
    provider = _Provider(root, _refuse_images(only=DIGEST_A))
    ctx = _round(root, provider, progress=progress, plan=_main_plan() if with_plan else None)
    if with_plan:
        ctx.active_model = ctx.context_fit_plan.model
    chain, cooled = [], []
    monkeypatch.setattr(loop, "_run_cross_model_fallback_chain", lambda **kw: chain.append(kw) or (None,) * 5)
    monkeypatch.setattr(fallback_cooldown, "mark_cooldown", lambda *a, **kw: cooled.append(a))
    keyed, route_key = [], vr.image_route_key

    def spy(model, role="vision", pin=None):
        keyed.append((sys._getframe(1).f_code.co_name, model, role, pin))
        return route_key(model, role, pin)

    monkeypatch.setattr(vr, "image_route_key", spy)
    msg, _cost, _mode = loop._call_round_model(ctx)

    assert msg == {"role": "assistant", "content": "done"}
    assert provider.sent == [{DIGEST_A}, frozenset()]
    assert provider.captions == []  # every automatic caption candidate is the refusing route
    assert 'HTTP 400, code image_not_supported: \\"Image input is not supported for this model\\"' in provider.texts[1]
    assert sorted(_ledger(root).values()) == ["settled", "unresolved"]
    usage = ctx.accumulated_usage
    assert list(usage[vr.REFUSED_IMAGES_KEY][vr.image_route_key(MAIN)]) == [DIGEST_A]
    # The retry records under the round's own role and account binding (``task_model_binding``), as it sent.
    assert ("retry_refused_image_round", MAIN, "main", None) in keyed
    assert vr._PENDING_REFUSALS_KEY not in usage and "_last_llm_error_kind" not in usage
    [event] = _events(root, "image_refusal_retry")
    assert (event["outcome"], event["status_code"], event["replaced_by"]) == ("answered", 400, ["marker"])
    assert WORDS in progress[0] and "retried with a note" in progress[0]
    # The round answered: no fallback route, no cooldown, and no capability fact on disk.
    loop._recover_failed_round(
        ctx, ctx.tools, msg, None, context_fit_plan=ctx.context_fit_plan, active_context_mode="max",
        emit_progress=progress.append)
    assert chain == [] and cooled == []
    store = root / "state" / "capability_evidence.json"
    assert not store.exists() or not capability_evidence._load(root).get(vr.EVIDENCE_NAMESPACE)

    # The next round does not ask the route for that image again; a new image goes as pixels.
    ctx.messages += [msg, {"role": "user", "content": [{"type": "image_url", "image_url": {"url": IMAGE_B}}]}]
    ctx.round_idx = 2
    msg2, _cost, _mode = loop._call_round_model(ctx)
    assert msg2 is not None and provider.sent[2:] == [{DIGEST_B}]
    assert WORDS in provider.texts[2]


def test_auto_captions_a_refused_image_with_a_route_that_did_not_refuse_it(root, monkeypatch):
    monkeypatch.setenv("OUROBOROS_MODEL_VISION", MAIN)  # the explicit slot is the refusing route
    monkeypatch.setenv("OUROBOROS_MODEL_LIGHT", CAPTIONER)
    provider = _Provider(root, _refuse_images())
    msg, _cost, _mode = loop._call_round_model(_round(root, provider))
    assert msg is not None and provider.sent == [{DIGEST_A}, frozenset()]
    assert provider.captions == [CAPTIONER]
    assert "a caption replaces it] [image caption: a red square]" in provider.texts[1]
    assert _events(root, "image_refusal_retry")[0]["replaced_by"] == ["caption"]


def test_inline_retries_with_a_marker_and_starts_no_caption_call(root, monkeypatch):
    monkeypatch.setenv("OUROBOROS_IMAGE_INPUT_MODE", "inline")
    monkeypatch.setenv("OUROBOROS_MODEL_LIGHT", CAPTIONER)
    progress: list = []
    provider = _Provider(root, _refuse_images(lambda: _ProviderError(404, "No endpoints found that support image input", "404")))
    msg, _cost, _mode = loop._call_round_model(_round(root, provider, progress=progress))
    assert msg is not None and provider.sent == [{DIGEST_A}, frozenset()] and provider.captions == []
    assert "Inline mode keeps this image out of this route" in provider.texts[1]
    assert 'HTTP 404: \\"No endpoints found that support image input\\"' in provider.texts[1]
    assert "No endpoints found that support image input" in progress[0]


def test_a_retry_refused_too_keeps_both_errors_and_learns_nothing(root):
    progress: list = []
    provider = _Provider(root, lambda _n, images: _ProviderError(400) if images
                         else _ProviderError(400, "Unsupported value: temperature", "unsupported_value"))
    ctx = _round(root, provider, progress=progress)
    msg, _cost, _mode = loop._call_round_model(ctx)
    assert msg is None and provider.sent == [{DIGEST_A}, frozenset()]
    assert vr.REFUSED_IMAGES_KEY not in ctx.accumulated_usage
    [event] = _events(root, "image_refusal_retry")
    assert event["outcome"] == "failed" and event["provider_message"] == WORDS
    assert event["retry_error"]["provider_message"] == "Unsupported value: temperature"
    assert event["retry_error"]["status_code"] == 400
    assert [row["provider_message"] for row in _events(root, "llm_api_error")] == [
        WORDS, "Unsupported value: temperature"]
    assert WORDS in progress[0] and "failed too" in progress[0]
    # Nothing was learned: the next round sends the image as pixels again.
    provider.answer = lambda _n, _images: "ok"
    ctx.round_idx = 2
    assert loop._call_round_model(ctx)[0] is not None and provider.sent[2:] == [{DIGEST_A}]


@pytest.mark.parametrize("error", [
    lambda: _ProviderError(401, "invalid api key", "invalid_api_key"),
    lambda: _ProviderError(402, "insufficient credits", "insufficient_credits"),
    lambda: _ProviderError(403, "forbidden", "forbidden"),
    lambda: _ProviderError(429, "slow down", "rate_limited"),
    lambda: _Policy("connection is not permitted"),
    lambda: _ProviderError(400, "maximum context length exceeded", "context_length_exceeded"),
    _died,
], ids=["401", "402", "403", "429", "policy", "context-overflow", "unknown-outcome"])
def test_no_retry_for_auth_quota_rate_policy_overflow_or_unknown_outcome(root, monkeypatch, error):
    provider = _Provider(root, _refuse_images(error))
    ctx = _round(root, provider)
    chain = []
    monkeypatch.setattr(loop, "_run_cross_model_fallback_chain", lambda **kw: chain.append(kw) or (None, MAIN, False, None, "max"))
    msg, _cost, _mode = loop._call_round_model(ctx)
    assert msg is None and provider.sent and all(images == {DIGEST_A} for images in provider.sent)
    assert _events(root, "image_refusal_retry") == [] and vr.REFUSED_IMAGES_KEY not in ctx.accumulated_usage
    if provider.sent and isinstance(error(), _ProviderError) and error().status_code == 401:
        monkeypatch.setenv("OUROBOROS_MODEL_FALLBACKS", CAPTIONER)
        loop._recover_failed_round(ctx, ctx.tools, None, None, context_fit_plan=None, active_context_mode="max",
                                   emit_progress=lambda *_a, **_k: None)
        assert chain, "without the retry the round recovers through the configured routes as before"


def test_a_single_server_error_then_ordinary_retry_keeps_the_pixels(root):
    provider = _Provider(root, lambda n, _images: _ProviderError(500, "temporary", "server_error") if n == 1 else "ok")
    ctx = _round(root, provider)
    assert loop._call_round_model(ctx)[0] is not None
    assert provider.sent == [{DIGEST_A}, {DIGEST_A}]
    assert vr.REFUSED_IMAGES_KEY not in ctx.accumulated_usage and _events(root, "image_refusal_retry") == []


def test_an_exhausted_server_error_series_retries_once_without_that_image(root):
    provider = _Provider(root, _refuse_images(lambda: _ProviderError(500, "image input is not supported", "server_error")))
    ctx = _round(root, provider)
    msg, _cost, _mode = loop._call_round_model(ctx)
    assert msg is not None and len(provider.sent) > 2
    assert provider.sent[:-1] == [{DIGEST_A}] * (len(provider.sent) - 1) and provider.sent[-1] == frozenset()
    assert list(ctx.accumulated_usage[vr.REFUSED_IMAGES_KEY][vr.image_route_key(MAIN)]) == [DIGEST_A]


def test_a_server_error_series_the_deadline_cut_short_earns_no_retry(root, monkeypatch):
    monkeypatch.setattr(loop_llm_call, "_sleep_within_deadline", lambda *_a, **_k: False)
    provider = _Provider(root, _refuse_images(lambda: _ProviderError(500, "image input is not supported", "server_error")))
    ctx = _round(root, provider)
    assert loop._call_round_model(ctx)[0] is None
    assert provider.sent == [{DIGEST_A}] and _events(root, "image_refusal_retry") == []


def test_a_retry_whose_candidate_still_carries_the_image_never_leaves_the_host(root, monkeypatch):
    """The predicate is the authority: a candidate with a refused image is not sent."""
    monkeypatch.setattr(vr, "refused_images", lambda _routing: {})
    provider = _Provider(root, _refuse_images())
    ctx = _round(root, provider)
    msg, _cost, _mode = loop._call_round_model(ctx)
    assert msg is None and provider.sent == [{DIGEST_A}]  # the second candidate was refused before dispatch
    usage = ctx.accumulated_usage
    assert usage["_last_llm_error_kind"] == "bad_request" and usage["_last_llm_status_code"] == 400
    assert sorted(_ledger(root).values()) == ["released", "unresolved"]  # the retry was released, never dispatched
    assert _events(root, "image_refusal_retry")[0]["outcome"] == "not_sent"


def test_a_new_task_sends_the_same_image_as_pixels_again(root):
    provider = _Provider(root, _refuse_images())
    assert loop._call_round_model(_round(root, provider))[0] is not None
    fresh = _Provider(root, lambda _n, _images: "ok")
    assert loop._call_round_model(_round(root, fresh))[0] is not None
    assert fresh.sent == [{DIGEST_A}]


def test_a_fresh_confirmed_yes_is_never_overwritten_by_a_refusal(root, monkeypatch):
    from ouroboros.provider_models import supports_vision
    from tests.test_image_capability_contract import RECORDED_ROWS, SEES, _receive_openrouter_catalog

    _receive_openrouter_catalog(monkeypatch, RECORDED_ROWS)
    before = capability_evidence._load(root).get(vr.EVIDENCE_NAMESPACE)
    assert supports_vision(SEES) is True
    monkeypatch.setenv("OUROBOROS_MODEL", SEES)
    provider = _Provider(root, _refuse_images(only=DIGEST_A))
    ctx = _round(root, provider)
    ctx.active_model = SEES
    assert loop._call_round_model(ctx)[0] is not None and provider.sent == [{DIGEST_A}, frozenset()]
    assert supports_vision(SEES) is True
    assert capability_evidence._load(root).get(vr.EVIDENCE_NAMESPACE) == before
    ctx.messages.append({"role": "user", "content": [{"type": "image_url", "image_url": {"url": IMAGE_B}}]})
    ctx.round_idx = 2
    assert loop._call_round_model(ctx)[0] is not None and provider.sent[2:] == [{DIGEST_B}]


def test_a_refusal_belongs_to_its_account_not_to_every_account_of_the_model(root, monkeypatch):
    """Equal model strings with different accounts are different routes (``model_slots.route_binding``)."""
    from tests.test_llm_claudexor import MODEL as SUBSCRIPTION

    monkeypatch.setenv("OUROBOROS_MODEL_ACCOUNTS", json.dumps({"main": "main-a", "vision": "vision-b"}))
    usage: dict = {vr.REFUSED_IMAGES_KEY: {vr.image_route_key(SUBSCRIPTION, "main"): {DIGEST_A: {"status": 400}}}}
    main_send = vr.VisionRoutingContext(model=SUBSCRIPTION, llm=_Provider(root, None), accumulated_usage=usage)
    assert set(vr.refused_images(main_send)) == {DIGEST_A}, "the refusing account keeps its memory"
    refused_by = vr.refusal_check(usage, [DIGEST_A])
    assert refused_by(SUBSCRIPTION) == "", "the vision account never refused this image: it may still take it"
    on_main_account = vr.VisionRoutingContext(model=SUBSCRIPTION, llm=_Provider(root, None), accumulated_usage=usage,
                                              model_role="vision", model_account_override="main-a")
    assert set(vr.refused_images(on_main_account)) == {DIGEST_A}, "the same account through another role is that route"
    vr.record_image_refusal(usage, SUBSCRIPTION, [DIGEST_A], {"status": 400})  # now the vision account refused it too
    assert "refused this image" in refused_by(SUBSCRIPTION)
    # An API route has no account: a pin or a role never splits its memory.
    assert vr.image_route_key(MAIN, "main", "any-pin") == vr.image_route_key(MAIN)


def test_the_owners_live_account_choice_binds_the_refusal_memory(root, monkeypatch):
    """A task-only switch of the vision route's account (a wait card) is that route, as the send applies it."""
    from types import SimpleNamespace

    from ouroboros import model_wait
    from tests.test_llm_claudexor import MODEL as SUBSCRIPTION

    monkeypatch.setenv("OUROBOROS_MODEL_ACCOUNTS", json.dumps({"vision": "vision-b"}))
    usage: dict = {}
    vr.record_image_refusal(usage, SUBSCRIPTION, [DIGEST_A], {"status": 400})  # the configured account refused it
    assert "refused this image" in vr.refusal_check(usage, [DIGEST_A])(SUBSCRIPTION)
    wait = SimpleNamespace(overrides={"vision": {"model": SUBSCRIPTION, "model_account_override": "vision-c"}})
    monkeypatch.setattr(model_wait, "current_model_wait", lambda: wait)
    assert vr.refusal_check(usage, [DIGEST_A])(SUBSCRIPTION) == "", "the newly chosen account may take the image"
    vr.record_image_refusal(usage, SUBSCRIPTION, [DIGEST_B], {"status": 400})  # refused on the chosen account
    assert "refused this image" in vr.refusal_check(usage, [DIGEST_B])(SUBSCRIPTION)
    monkeypatch.setattr(model_wait, "current_model_wait", lambda: None)
    assert vr.refusal_check(usage, [DIGEST_B])(SUBSCRIPTION) == "", "the configured account never refused it"


def test_a_resent_candidate_stays_in_the_bounded_image_registry(monkeypatch):
    monkeypatch.setattr(vr, "_CANDIDATE_IMAGES", {})
    monkeypatch.setattr(vr, "_CANDIDATE_IMAGES_KEPT", 3)
    payload = {"messages": _messages(IMAGE_A)}
    for key in ("a", "b", "c", "a", "d"):  # "a" is sent again (a 5xx series) before "d" arrives
        vr.note_candidate_images(key, payload)
    assert vr.candidate_images("a") == {DIGEST_A} and vr.candidate_images("b") is None
    assert list(vr._CANDIDATE_IMAGES) == ["c", "a", "d"]


def test_a_fallback_route_never_inherits_another_routes_refusal(root):
    usage: dict = {}
    vr.record_image_refusal(usage, MAIN, [DIGEST_A], {"status": 400})
    routing = vr.VisionRoutingContext(model=CAPTIONER, llm=_Provider(root, None), accumulated_usage=usage)
    assert vr.refused_images(routing) == {}
    assert vr.prepare_messages_for_send(_messages(IMAGE_A), routing=routing)[1]["content"][1]["image_url"]["url"] == IMAGE_A
    same_route = vr.VisionRoutingContext(model=MAIN, llm=_Provider(root, None), accumulated_usage=usage)
    assert IMAGE_A not in json.dumps(vr.prepare_messages_for_send(_messages(IMAGE_A), routing=same_route))


# ---------------------------------------------------------------------------
# VLM and caption calls: no silent retry; the refusing route is not offered the image again
# ---------------------------------------------------------------------------

URL = "https://example.invalid/x.png"


def test_vlm_refusal_is_typed_remembered_and_the_route_is_not_offered_that_image_again(root, monkeypatch):
    from ouroboros.tools import vision
    from tests.test_image_capability_contract import _child_image_refusal, _record_vlm_calls, _vlm_ctx

    usage: dict = {}
    calls = _record_vlm_calls(monkeypatch, error=_child_image_refusal(MAIN))
    first = vision._vlm_query(_vlm_ctx(_accumulated_usage=usage), "Describe.", image_url=URL, model=MAIN)
    assert calls == [MAIN] and "404" in first
    assert list(usage[vr.REFUSED_IMAGES_KEY][vr.image_route_key(MAIN)]) == [vr._url_digest(URL)]

    again = vision._vlm_query(_vlm_ctx(_accumulated_usage=usage), "Describe.", image_url=URL, model=MAIN)
    assert calls == [MAIN], "the explicit model refused this image: it is not called again"
    assert again.startswith("⚠️ VLM_NO_VISION_MODEL") and "refused this image earlier in the task (HTTP 404" in again

    vision._vlm_query(_vlm_ctx(_accumulated_usage=usage), "Describe.", image_url=URL + "?other", model=MAIN)
    assert calls == [MAIN, MAIN], "another image still goes to the same explicit model"

    # The next Main turn on that route does not resend the image either; another image goes as pixels.
    routing = vr.VisionRoutingContext(model=MAIN, llm=_Provider(root, None), accumulated_usage=usage)
    sent = vr.prepare_messages_for_send(_messages(URL, IMAGE_B), routing=routing)
    assert URL not in json.dumps(sent) and IMAGE_B in json.dumps(sent)


def test_a_caption_route_that_refused_an_image_is_not_asked_for_it_again(root, monkeypatch):
    monkeypatch.setenv("OUROBOROS_IMAGE_INPUT_MODE", "caption")
    monkeypatch.setenv("OUROBOROS_MODEL_VISION", CAPTIONER)
    refusal = _ProviderError(400)
    refusal.physical_attempt_capture = ua.PhysicalAttemptCapture(
        attempt_id="caption", model="openai/acme-captioner-1", provider="openai", state="unresolved",
        candidate_measurement_kind="canonical_json_v1", provider_status_code=400, provider_code="image_not_supported")
    asked: list = []

    class _Captioner(_Provider):
        def vision_query(self, _prompt, _images, **kwargs):
            asked.append(kwargs.get("model"))
            raise refusal

    usage: dict = {}
    routing = vr.VisionRoutingContext(model=MAIN, llm=_Captioner(root, None), accumulated_usage=usage)
    first = json.dumps(vr.prepare_messages_for_send(_messages(IMAGE_A), routing=routing))
    assert asked == [CAPTIONER] and "caption unavailable" in first
    second = json.dumps(vr.prepare_messages_for_send(_messages(IMAGE_A), routing=routing))
    assert asked == [CAPTIONER, MAIN], "the next candidate is asked, never the refusing route again"
    assert IMAGE_A not in second


def test_a_nano_retry_sends_under_the_refused_attempts_reply_allowance(tmp_path, monkeypatch):
    """The retry follows the overflow retry's allowance rule: its logical ceiling is the refused
    attempt's sent allowance, and its predicate admits at most that allowance, without the image."""
    from tests.test_loop_compaction import _candidate_request, _ctx, _failed_capture, _fit

    monkeypatch.setenv("OUROBOROS_IMAGE_INPUT_MODE", "auto")
    context = _ctx(tmp_path, preferred="nano", mode="nano")
    vr.note_candidate_images("refused-with-image", {"messages": _messages(IMAGE_A)})
    vr.note_candidate_images("retry-text-only", {"messages": _messages()})
    vr.note_candidate_images("retry-with-image", {"messages": _messages(IMAGE_A)})
    failed = replace(_failed_capture(profile="owner_nano", mode="nano", reserve=20_000),
                     provider_status_code=400, candidate_raw_sha256="refused-with-image")
    checked = []

    def measure(ctx, **_kwargs):
        disposition = _fit(profile="owner_nano", mode="nano")
        loop._remember_main_fit(ctx, disposition)
        return disposition

    def dispatch(ctx, disposition, *, candidate_predicate=None, max_tokens=None, **_kwargs):
        if candidate_predicate is None:
            ctx.accumulated_usage["_last_llm_error_kind"] = "bad_request"
            return None, 0.0
        request = replace(_candidate_request(disposition, size=700, reserve=20_000), candidate_raw_sha256="retry-text-only")
        checked.append((max_tokens, candidate_predicate(request),
                        candidate_predicate(replace(request, max_completion_tokens=8_192)),
                        candidate_predicate(replace(request, max_completion_tokens=20_001)),
                        candidate_predicate(replace(request, candidate_raw_sha256="retry-with-image"))))
        return {"role": "assistant", "content": "seen as text", "tool_calls": []}, 0.0

    monkeypatch.setattr(loop, "_measure_round_main_fit", measure)
    monkeypatch.setattr(loop, "_dispatch_round_model", dispatch)
    monkeypatch.setattr(loop, "last_physical_attempt_capture", lambda: failed)
    monkeypatch.setattr(loop, "_emit_checkpoint_event", lambda *_a, **_kw: None)
    msg, _cost, mode = loop._call_round_model(context)
    assert msg["content"] == "seen as text" and mode == "nano"
    assert checked == [(20_000, True, True, False, False)]


class _Completion:
    def __init__(self, body):
        self.body = body

    def model_dump(self):
        return copy.deepcopy(self.body)


@pytest.mark.parametrize("model", [MAIN, "acme/never-listed-1"], ids=["direct-openai", "openrouter"])
def test_the_real_transport_records_the_sent_images_and_retries_without_them(root, monkeypatch, model):
    """Through the production send path: the finalized wire candidate names its images at the
    binding seam, the provider refuses it, and one text-only retry answers on the same route."""
    from types import SimpleNamespace

    from ouroboros import llm_fallback

    monkeypatch.setenv("OUROBOROS_MODEL", model)
    seen = []

    def create(**payload):
        carried = "QUFBQQ==" in json.dumps(payload.get("messages"))
        seen.append(carried)
        if carried:
            raise _ProviderError(400)
        return _Completion({"id": "c", "object": "chat.completion", "model": model.split("::")[-1], "choices": [
            {"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "seen as text"}}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5}})

    client = LLMClient()
    remote = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    monkeypatch.setattr(client, "_get_remote_client", lambda _target: remote)
    monkeypatch.setattr(llm_fallback, "consume_stream", lambda response, **_kw: response)
    ctx = _round(root, client)
    ctx.active_model = model
    with ua.usage_scope(ua.UsageScope(drive_root=root, task_id="t-img", root_task_id="t-img")):
        msg, _cost, _mode = loop._call_round_model(ctx)
    assert msg is not None and msg.get("content") == "seen as text"
    assert seen[0] is True and seen[-1] is False
    assert [list(images) for images in ctx.accumulated_usage[vr.REFUSED_IMAGES_KEY].values()] == [[DIGEST_A]]
    finals = list(_ledger(root).values())  # request-wire recovery may add refused attempts
    assert finals[-1] == "settled" and set(finals[:-1]) == {"unresolved"}
