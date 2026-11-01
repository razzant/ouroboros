"""OpenRouter's typed parameter-filter refusal, and what the wire-recovery driver does with it.

Under ``provider.require_parameters`` OpenRouter routes only to endpoints that accept EVERY
requested parameter; when none is left it answers HTTP 404 with the typed receipt
``error.metadata.failed_routing_step == "Filter by Parameters"`` and names no field. Found by
the planning trials: every default Opus review seat (plan review and commit review alike) sent
``temperature=0.2`` to a model none of whose eleven endpoints takes ``temperature``, and the
field-less 404 left the driver with nothing to drop, so the seat failed on both builds. The
driver now reads the typed step: ONLY the optional sampling knobs (temperature, top_p, top_k) go;
reasoning, ``response_format`` and tools are never dropped on it, and a refusal that persists
once the knobs are gone propagates as the provider's typed error (nothing learned). A model that
accepts ``temperature`` keeps it, and a refusal of another kind is not touched. Live probe 2026-10-15
(``anthropic/claude-opus-5.5``, max_tokens 16, require_parameters true): with temperature 0.2
→ 404 + this typed step; without → 200, "ok", cost $0.000152."""
from __future__ import annotations

import copy
import json

import pytest

from ouroboros.llm import LLMClient
from ouroboros.request_wire_recovery import (
    plan_nonlearning_optional_retry,
    plan_wire_retry_from_body_error,
    plan_wire_retry_from_exception,
    prepare_wire_payload_for_send,
    request_wire_call_scope,
)
from ouroboros.usage_accounting import UsageScope, usage_scope
from tests.test_request_wire_recovery_phase2b import _Response, _store, _target, _tools
from tests.test_request_wire_recovery_phase2b import evidence_root as _evidence_root

evidence_root = _evidence_root  # noqa: F811 - pytest fixture re-export

ROUTED_OUT = {"error": {
    "code": 404,
    "message": "No endpoints found that can handle the requested parameters. To learn more about provider routing, "
               "visit: https://openrouter.ai/docs/provider-routing",
    "metadata": {"routing_funnel": [{"step": "Initial Endpoints", "endpoint_count": 11},
                                    {"step": "Filter by Regional Surcharge", "endpoint_count": 6}],
                 "failed_routing_step": "Filter by Parameters"},
}}


class _RoutedOut(RuntimeError):
    """The OpenAI SDK's shape of the refusal: ``status_code`` and the JSON ``body``."""

    def __init__(self, body=None, status: int = 404):
        body = copy.deepcopy(body or ROUTED_OUT)
        super().__init__(f"Error code: {status} - {json.dumps(body)}")
        self.status_code = status
        self.body = body


def _other_step():
    body = copy.deepcopy(ROUTED_OUT)
    body["error"]["metadata"]["failed_routing_step"] = "Filter by Regional Surcharge"
    return body


def _review_payload(**extra):
    """What a default review seat sends: the host's sampling hint, the reasoning carrier, tools."""
    payload = {
        "model": "anthropic/claude-opus-5.5",
        "messages": [{"role": "user", "content": "Weigh the plan before the work starts."}],
        "temperature": 0.2, "max_tokens": 64, "tools": _tools(), "tool_choice": "auto",
        "extra_body": {"reasoning": {"effort": "high", "exclude": False}, "provider": {"require_parameters": True}},
    }
    payload.update(extra)
    return payload


def _run(create, payload, tmp_path, label):
    target = _target(model="anthropic/claude-opus-5.5")
    client = LLMClient(api_key="unused")
    with request_wire_call_scope(), usage_scope(UsageScope(drive_root=tmp_path / label, task_id=label)):
        response = client._create_chat_completion_with_retries(create, payload, target)
        return client._normalize_remote_response(response.model_dump(), target, skip_cost_fetch=True)


def test_a_field_less_parameter_filter_404_drops_the_sampling_knob_and_keeps_the_rest(evidence_root, tmp_path):
    sent = []

    def create(**candidate):
        sent.append(copy.deepcopy(candidate))
        if "temperature" in candidate:
            raise _RoutedOut()
        return _Response()

    _msg, usage = _run(create, _review_payload(), tmp_path, "filtered")
    assert len(sent) == 2 and "temperature" in sent[0] and "temperature" not in sent[1]
    # Only the sampling knob left: the reasoning carrier, the tools and the require_parameters pin stay.
    assert sent[1]["extra_body"]["reasoning"] == sent[0]["extra_body"]["reasoning"]
    assert sent[1]["extra_body"]["provider"] == {"require_parameters": True} and sent[1]["tools"] == sent[0]["tools"]
    # Disclosed on the returned usage, and learned for this exact route after the exact success.
    assert "temperature" in json.dumps(usage.get("request_wire") or {})
    assert "temperature" in json.dumps(_store(evidence_root) or {})


def test_sampling_stays_where_endpoints_accept_it_and_other_refusals_are_untouched(evidence_root, tmp_path):
    accepted = []

    def accept(**candidate):
        accepted.append(copy.deepcopy(candidate))
        return _Response()

    _run(accept, _review_payload(), tmp_path, "accepted")
    assert len(accepted) == 1 and accepted[0]["temperature"] == 0.2, "a model that takes the knob keeps it"

    for label, error, payload in (
        ("other-step", _RoutedOut(_other_step()), _review_payload()),            # a 404 of another typed step
        ("rate-limit", _RoutedOut(status=429), _review_payload()),              # not a compatibility refusal
        ("nothing-optional", _RoutedOut(), _review_payload(temperature=None)),  # nothing optional was sent
    ):
        if label == "nothing-optional":
            payload.pop("temperature")
            payload["extra_body"].pop("reasoning")
        sent = []

        def refuse(**candidate):
            sent.append(copy.deepcopy(candidate))
            raise error

        with pytest.raises(_RoutedOut):
            _run(refuse, payload, tmp_path, label)
        assert len(sent) == 1, label  # one send, no blind resend


def test_a_refusal_that_persists_without_the_knobs_propagates_and_never_drops_reasoning_shape_or_tools(evidence_root, tmp_path):
    """Owner decision (Codex audit, point 1): the field-less typed 404 drops ONLY the sampling
    knobs; reasoning, ``response_format`` and tools are never dropped on it. When the refusal
    persists after the knobs are gone the provider's typed error stands — one resend, no
    blind third send, nothing learned for a route that never succeeded."""
    sent = []

    def create(**candidate):
        sent.append(copy.deepcopy(candidate))
        raise _RoutedOut()  # refused with and without the sampling knobs alike

    with pytest.raises(_RoutedOut):
        _run(create, _review_payload(top_p=0.9, response_format={"type": "json_object"}), tmp_path, "persists")
    assert len(sent) == 2, "one sampling-free resend, then the provider's typed error"
    assert "temperature" not in sent[1] and "top_p" not in sent[1]
    assert sent[1]["extra_body"]["reasoning"] == sent[0]["extra_body"]["reasoning"]
    assert sent[1]["response_format"] == sent[0]["response_format"] and sent[1]["tools"] == sent[0]["tools"]
    assert _learned_fields(evidence_root) == [], "a route that never succeeded teaches nothing"
    # The other direction: the knobs alone are learned, and only after the exact success.
    accepted = []

    def accept_without_knobs(**candidate):
        accepted.append(copy.deepcopy(candidate))
        if "temperature" in candidate or "top_p" in candidate:
            raise _RoutedOut()
        return _Response()

    _run(accept_without_knobs, _review_payload(top_p=0.9, response_format={"type": "json_object"}), tmp_path, "knobs-only")
    assert len(accepted) == 2 and accepted[1]["response_format"] == accepted[0]["response_format"]
    assert _learned_fields(evidence_root) == ["temperature", "top_p"]


def _learned_fields(root):
    """Every field the learning store now names in a drop/set action (never the profile's own words)."""
    store = _store(root) or {"profiles": {}}
    return sorted({field for entry in store["profiles"].values() for record in entry.get("records") or []
                   for field in (record["action"].get("fields") or [record["action"].get("field")]) if field})


def test_the_non_learning_path_reads_the_same_typed_step():
    carrierless = {"model": "m", "messages": [{"role": "user", "content": "x"}], "temperature": 0.2, "max_tokens": 8}
    assert plan_nonlearning_optional_retry(carrierless, error=_RoutedOut()) == {
        "model": "m", "messages": [{"role": "user", "content": "x"}], "max_tokens": 8}
    assert plan_nonlearning_optional_retry(carrierless, error=_RoutedOut(_other_step())) is None
    assert plan_nonlearning_optional_retry({"model": "m", "messages": [], "max_tokens": 8}, error=_RoutedOut()) is None


KIMI_REFUSAL = "invalid temperature: only 1 is allowed for this model"


class _KimiRefusal(RuntimeError):
    """Kimi's coding endpoint (k3.1-pro): any temperature other than 1, HTTP 400."""

    def __init__(self, message: str = KIMI_REFUSAL, status: int = 400, param: str = ""):
        super().__init__(message)
        self.status_code = status
        self.body = {"error": {"message": message, "type": "invalid_request_error"}}
        if param:
            self.body["error"]["param"] = param


def _seat_payload(**extra):
    """What the failed review seat sent: the host's temperature hint, the reasoning carrier, tools."""
    payload = {
        "model": "kimi/k3.1-pro",
        "messages": [{"role": "user", "content": "Weigh the plan before the work starts."}],
        "temperature": 0.2, "max_tokens": 64, "tools": _tools(), "tool_choice": "auto",
        "extra_body": {"reasoning": {"effort": "high", "exclude": False}},
    }
    payload.update(extra)
    return payload


def _run_seat(create, payload, tmp_path, label):
    target = _target(provider="openai-compatible", model="kimi/k3.1-pro")
    client = LLMClient(api_key="unused")
    with request_wire_call_scope(), usage_scope(UsageScope(drive_root=tmp_path / label, task_id=label)):
        response = client._create_chat_completion_with_retries(create, payload, target)
        return client._normalize_remote_response(response.model_dump(), target, skip_cost_fetch=True)


def test_a_single_allowed_sampling_value_refusal_drops_the_named_knob(evidence_root, tmp_path):
    """Kimi's k3.1-pro answers the review lane's temperature hint (0.2) with HTTP 400
    "invalid temperature: only 1 is allowed for this model". The VALUE is refused, not
    the field, so dropping the knob is the value fix: one resend without it, the
    reasoning carrier/tools/model untouched, learned for the exact route once that
    resend succeeds — and the next send to the route omits the knob without refusal."""
    sent = []

    def create(**candidate):
        sent.append(copy.deepcopy(candidate))
        if "temperature" in candidate:
            raise _KimiRefusal()
        return _Response()

    _msg, usage = _run_seat(create, _seat_payload(), tmp_path, "kimi")
    assert len(sent) == 2 and sent[0]["temperature"] == 0.2 and "temperature" not in sent[1]
    assert sent[1]["extra_body"]["reasoning"] == sent[0]["extra_body"]["reasoning"]
    assert sent[1]["tools"] == sent[0]["tools"] and sent[1]["model"] == sent[0]["model"]
    assert "temperature" in json.dumps(usage.get("request_wire") or {})
    # The provider pinned the knob's VALUE, so the recorded reason says exactly that.
    assert "provider_prescribed_value" in json.dumps(usage.get("request_wire") or {})
    assert _learned_fields(evidence_root) == ["temperature"]

    fresh = []

    def accept(**candidate):
        fresh.append(copy.deepcopy(candidate))
        return _Response()

    _run_seat(accept, _seat_payload(), tmp_path, "kimi-learned")
    assert len(fresh) == 1 and "temperature" not in fresh[0]


def test_a_single_value_refusal_that_persists_without_the_knob_propagates(evidence_root, tmp_path):
    sent = []

    def create(**candidate):
        sent.append(copy.deepcopy(candidate))
        raise _KimiRefusal()

    with pytest.raises(_KimiRefusal):
        _run_seat(create, _seat_payload(top_p=0.9), tmp_path, "persists")
    assert len(sent) == 2, "one knob-free resend, then the provider's error stands"
    assert "temperature" not in sent[1]
    assert sent[1]["top_p"] == 0.9, "only the NAMED knob goes; its siblings are untouched by this rule"
    assert _learned_fields(evidence_root) == [], "a route that never succeeded teaches nothing"


def test_an_unrelated_400_and_a_range_refusal_keep_todays_one_send(evidence_root, tmp_path):
    for label, error in (
        ("unrelated", _KimiRefusal("model not found")),
        ("range", _KimiRefusal("temperature must be between 0 and 2")),
        ("plain-value", _KimiRefusal("temperature value '0.2' is not supported")),
    ):
        sent = []

        def create(**candidate):
            sent.append(copy.deepcopy(candidate))
            raise error

        with pytest.raises(_KimiRefusal):
            _run_seat(create, _seat_payload(), tmp_path, label)
        assert len(sent) == 1, label
    assert _learned_fields(evidence_root) == []


@pytest.mark.parametrize("body_error", [False, True])
@pytest.mark.parametrize("message,param,dropped", [
    (KIMI_REFUSAL, "", ["temperature"]),
    ("invalid top_p: only 0.5 is allowed", "", ["top_p"]),  # shape A
    ("invalid temperature: only 1 is allowed", "body.temperature", ["temperature"]),  # namespaced validator param: the leaf is our knob
    ("invalid temperature: only 1 is supported", "", ["temperature"]),  # the predicate boundary ends the clause too
    ("temperature can only be 1", "", ["temperature"]),
    ("only 1 is allowed for temperature", "", ["temperature"]),  # shape C: the predicate names the subject itself
    ("temperature can only take 1", "", []),      # outside the three shapes: "take" is not one of them
    ("temperature can only be set to 1", "", []), # outside the three shapes: "be set to" is not "be <value>"
    ("temperature can only be 0 or 1", "", []),   # two allowed values, not one
    ("temperature can only take 0, 1", "", []),    # a list said with "can only take"
    ("temperature can only be 0 and 1", "", []),   # "and" is a continuation, not a clause end
    ("temperature can only be 0 to 2", "", []),    # a range said by continuation
    ("temperature can only be 0 | 1", "", []),     # the pipe list shape
    ("only 0 or 1 is allowed for temperature", "", []),  # a list said with "is allowed"
    ("only 0 and 1 are allowed for temperature", "", []),  # plural predicate, still a list: no single value ends a clause
    ("temperature can only be between 0 and 2", "", []),   # a range said with "can only be" is not a single value
    ("top_p can only take values between 0 and 1", "", []),
    ("invalid temperature and top_p: only 1 is allowed", "", []),  # no shape: the colon follows two knobs, not one named subject
    ("top_p can only be 0.5", "temperature", []),  # the param and the shaped field disagree: no drop
    ("Invalid value: only 1 is allowed for this model", "temperature", []),  # no shape names a sampling subject; the param stays unconfirmed
    ("invalid max_tokens: only 1 is allowed; temperature is optional", "", []),  # the shape's subject is max_tokens, not a sampling knob
    ("invalid max_tokens: only 1 is allowed; temperature is optional", "temperature", []),  # a param the prose mentions confirms nothing
    ("invalid max_tokens: only 1 is allowed, temperature is optional", "", []),  # comma or semicolon alike: no shape takes max_tokens as a sampling subject
    ("invalid max_tokens: only 1 is allowed for temperature-less models", "", []),  # shape C's subject is a hyphenated compound, not the knob
    ("max_tokens is optional; invalid temperature: only 1 is allowed", "", ["temperature"]),  # shape A still names its subject after another clause
    ("max_tokens is optional; invalid temperature: only 1 is allowed", "body.temperature", ["temperature"]),  # the structured param agrees with the shaped field
    (KIMI_REFUSAL.replace("temperature", "response_format"), "", []),   # a non-sampling field named in prose
    ("Invalid value: only 1 is allowed", "response_format", []),        # a non-sampling structured param
    ("Invalid value: only 1 is allowed", "model", []),                   # not an optional field at all
    ("invalid model: only 'kimi/k3.1-pro' is allowed", "", []),          # unrelated field; the prescription is also not a number
    ("temperature must be between 0 and 2", "", []),                     # range, no single-value prescription
    ("invalid value for temperature", "", []),                           # a plain value rejection keeps its carrier
])
def test_the_single_value_rule_drops_only_a_shaped_sampling_knob(evidence_root, body_error, message, param, dropped):
    """The rule is bound to the field an explicit sentence SHAPE names as its own
    subject — "invalid <field>: only <v> is allowed/is supported", "<field> can/may/must
    only be <v>", "only <v> is allowed for <field>" — with the single value anchored to
    the end of its clause, and a structured ``param`` (a namespaced validator's leaf
    included) agreeing with the captured field. The shape set is closed: any other
    wording, a prescription about another field (comma or semicolon joined alike), a
    knob merely mentioned elsewhere, a list or range of allowed values, or two
    different fields shaped at once keeps today's behavior."""
    target = _target(provider="openai-compatible", model="kimi/k3.1-pro")
    source = _seat_payload(top_p=0.9, response_format={"type": "json_object"})
    body = {"status": 400, "message": message, **({"param": param} if param else {})}
    with request_wire_call_scope():
        prepare_wire_payload_for_send(target, source, api_surface="chat.completions")
        retry = (
            plan_wire_retry_from_body_error(body) if body_error
            else plan_wire_retry_from_exception(_KimiRefusal(message, param=param))
        )
    if not dropped:
        assert retry is None
    else:
        assert retry is not None
        assert sorted(set(source) - set(retry)) == dropped


def test_a_message_the_marker_paths_already_recover_keeps_its_old_path(evidence_root, tmp_path):
    """The single-value rule is a pure FALLBACK behind the whole legacy classifier, so a
    refusal the old ladder recovers keeps its old path. An enum-bounded effort
    refusal that also carries the prescription prose takes the old ladder — set the
    effort to the highest lower tier — and the sampling knob is not dropped by the
    newer rule."""
    target = _target(provider="openai-compatible", model="kimi/k3.1-pro")
    source = _seat_payload()  # temperature 0.2, extra_body.reasoning.effort "high"
    body = {"status": 400,
            "message": "reasoning effort must be one of: low, medium; temperature can only be 1"}
    with request_wire_call_scope():
        prepare_wire_payload_for_send(target, source, api_surface="chat.completions")
        retry = plan_wire_retry_from_body_error(body)
    assert retry is not None, "the old enum ladder answers this refusal"
    assert "temperature" in retry, "the single-value rule did not drop the knob"
    assert retry["extra_body"]["reasoning"]["effort"] == "medium"


def test_a_thinking_refusal_with_a_prescription_keeps_the_thinking_path(evidence_root):
    """"model does not support thinking; temperature can only be 1" on a
    reasoning_effort carrier. The thinking-refusal shape is admitted by the legacy
    marker gate — no textual marker names it — and the old path answers by dropping
    the carrier. The single-value rule must not pre-empt it: dropping temperature
    alone would strand the request on the refused carrier and burn the resend on a
    repeat refusal."""
    target = _target(provider="openai-compatible", model="kimi/k3.1-pro")
    source = {  # the standard effort carrier, not the nested one the seat payload uses
        "model": "kimi/k3.1-pro",
        "messages": [{"role": "user", "content": "Weigh the plan before the work starts."}],
        "temperature": 0.2, "max_tokens": 64, "tools": _tools(), "tool_choice": "auto",
        "reasoning_effort": "high",
    }
    body = {"status": 400,
            "message": "model does not support thinking; temperature can only be 1"}
    with request_wire_call_scope():
        prepare_wire_payload_for_send(target, source, api_surface="chat.completions")
        retry = plan_wire_retry_from_body_error(body)
    assert retry is not None, "the old thinking path answers this refusal"
    # The old named-field loop drops BOTH the carrier it implicates via the thinking
    # refusal AND the knob the prose names. The pre-emption bug this test pins: the
    # single-value fallback would drop temperature ALONE and strand the refused
    # carrier for the resend to trip over again — so the carrier must be gone too.
    assert "reasoning_effort" not in retry, "the carrier went with the old thinking recovery"
    assert "temperature" not in retry, "the old loop dropped the named knob, not the fallback alone"
