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
from ouroboros.request_wire_contract import HINT_WITHDRAWAL_REASON
from ouroboros.request_wire_recovery import (
    current_wire_candidate,
    plan_next_wire_retry,
    plan_nonlearning_optional_retry,
    prepare_wire_payload_for_send,
    request_wire_call_scope,
    resolve_temperature_hint,
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


@pytest.mark.parametrize("message,fields_left", [
    ("temperature is unsupported", {"model", "messages", "max_tokens"}),  # the field: its inherited repair stands
    ("unsupported temperature: only 1 is allowed", None),  # a refused value, whatever words surround it
    ("invalid parameter: temperature can only be 0 or 1", None),
    ("unsupported temperature: not only 1 is allowed", None),  # unusable, and still no refusal of the field
    ("unsupported temperature: only 1 or less is allowed", None),
])
def test_the_non_learning_path_keeps_a_refused_value_apart_from_a_refused_field(message, fields_left):
    carrierless = {"model": "m", "messages": [{"role": "user", "content": "x"}], "temperature": 0.2, "max_tokens": 8}
    refused = _RoutedOut({"error": {"message": message}}, status=400)
    repaired = plan_nonlearning_optional_retry(carrierless, error=refused)
    assert (set(repaired) if repaired is not None else None) == fields_left


# PR #2588's case: Kimi's coding endpoint refuses the review lane's temperature hint. The hint keeps
# its origin on the route, so such a refusal may withdraw it and no explicit value. Rows keep the
# contributor's original wording; a comment marks each one whose expectation that policy changed.
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


def _seat_target(*, hint=True):
    """The seat's route; its 0.2 is the host's hint unless the caller stated it explicitly."""
    target = _target(provider="openai-compatible", model="kimi/k3.1-pro")
    assert resolve_temperature_hint(target, None if hint else 0.2, 0.2) == 0.2
    return target


def _run_seat(create, payload, tmp_path, label):
    target = _seat_target()
    client = LLMClient(api_key="unused")
    with request_wire_call_scope(), usage_scope(UsageScope(drive_root=tmp_path / label, task_id=label)):
        response = client._create_chat_completion_with_retries(create, payload, target)
        return client._normalize_remote_response(response.model_dump(), target, skip_cost_fetch=True)


def _plan_seat(*messages, source=None, param="", body_error=True, hint=True):
    """The planner's answer to each refusal in turn, on one call's candidates; and the last one's reasons."""
    target = _seat_target(hint=hint)
    current = source if source is not None else _seat_payload()
    retries = []
    with request_wire_call_scope():
        current = prepare_wire_payload_for_send(target, current, api_surface="chat.completions")
        for message in messages:
            error = ({"status": 400, "message": message, **({"param": param} if param else {})} if body_error
                     else _KimiRefusal(message, param=param))
            current = plan_next_wire_retry(current, error=error, body_error=body_error, target=target)
            retries.append(current)
        return retries, [item.action["reason_code"] for item in current_wire_candidate().applied_actions]


def _plan_carrierless(source, *messages, hint=True, **node):
    """The planner's answer to each refusal in turn, on a send no candidate binds."""
    target = _seat_target(hint=hint)
    current, retries = source, []
    with request_wire_call_scope():
        for message in messages:
            current = prepare_wire_payload_for_send(target, current, api_surface="chat.completions")
            assert current_wire_candidate() is None
            error = {"status": 400, "message": message, **node}
            current = plan_next_wire_retry(current, error=error, body_error=True, target=target)
            retries.append(current)
    return retries


def test_a_single_allowed_value_refusal_withdraws_the_review_hint_and_remembers_the_route(evidence_root, tmp_path):
    """Kimi's k3.1-pro answers the review lane's temperature hint (0.2) with HTTP 400
    "invalid temperature: only 1 is allowed for this model". The VALUE is refused, not
    the field, and the host's own preference gives way: one resend without it, the
    reasoning carrier/tools/model untouched, learned for the exact route once that
    resend succeeds — and the next hint on the route is omitted without refusal."""
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
    # The record names a withdrawn hint: omission succeeded, which proves no numeric default.
    assert HINT_WITHDRAWAL_REASON in json.dumps(usage.get("request_wire") or {})
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
    assert sent[1]["top_p"] == 0.9, "only the hint goes; its siblings are untouched by this rule"
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


@pytest.mark.parametrize(("message", "param", "withdrawn", "body_error"), [
    pytest.param(KIMI_REFUSAL, "", True, True, id='invalid temperature: only 1 is allowed for this model--True-True'),
    ("invalid temperature: only 1 is supported", "", True, True),  # the predicate boundary ends the clause too
    ("temperature can only be 1", "", True, False),
    ("only 1 is allowed for temperature", "", True, True),  # the predicate names the subject itself
    ("max_tokens is optional; invalid temperature: only 1 is allowed", "", True, False),  # names its subject after another clause
    ("temperature can only be set to 1", "", True, True),  # Formerly "not one value, so nothing": the sent hint's own number decides now, and no number here allows 0.2.
    ("temperature can only be 0 or 1", "", True, False),
    ("temperature can only be 0 and 1", "", True, True),
    ("temperature can only be 0 | 1", "", True, False),
    ("only 0 or 1 is allowed for temperature", "", True, True),
    ("only 0 and 1 are allowed for temperature", "", True, False),
    ("Invalid value: only 1 is allowed for this model", "temperature", True, True),  # Formerly "the param stays unconfirmed": a typed parameter owns a statement that opens its own clause...
    ("invalid max_tokens: only 1 is allowed; temperature is optional", "temperature", True, False),  # ...even after another field's colon, a limit of the grammar: the structure outranks that prose, at the bounded cost of one hint-free resend, learned only if it succeeds.
    ("invalid top_p: only 0.5 is allowed", "", False, True),  # Formerly dropped: top_p is no host hint, so it has no origin to withdraw; the provider's error stands.
    ("invalid temperature: only 1 is allowed", "body.temperature", False, False),  # Formerly dropped by the path's leaf: only the exact parameter binds, as "messages.temperature" must not.
    ("max_tokens is optional; invalid temperature: only 1 is allowed", "body.temperature", False, True),
    ("temperature can only take 1", "", False, False),  # Unchanged: nothing here is a complete statement of temperature's values excluding the hint. "take" is outside the closed grammar
    ("temperature can only take 0, 1", "", False, False),
    ("temperature can only be 0 to 2", "", False, True),  # a range holding 0.2
    ("temperature can only be between 0 and 2", "", False, False),
    ("top_p can only take values between 0 and 1", "", False, True),
    ("invalid temperature and top_p: only 1 is allowed", "", False, False),  # the colon follows two fields, not one subject
    ("top_p can only be 0.5", "temperature", False, True),  # that statement has its own subject, whatever the param says
    ("invalid max_tokens: only 1 is allowed; temperature is optional", "", False, False),  # the subject is max_tokens
    ("invalid max_tokens: only 1 is allowed, temperature is optional", "", False, True),  # comma or semicolon alike
    ("invalid max_tokens: only 1 is allowed for temperature-less models", "", False, False),  # a compound is not the field
    ("invalid max_tokens: must be 1 for low-temperature models", "", False, True),
    ("only 1 is allowed for temperature-less models", "", False, False),
    pytest.param(KIMI_REFUSAL.replace("temperature", "response_format"), "", False, True, id='invalid response_format: only 1 is allowed for this model--False-True'),  # a non-sampling field named in prose
    ("Invalid value: only 1 is allowed", "response_format", False, False),  # a non-sampling structured param
    ("Invalid value: only 1 is allowed", "model", False, False),  # not an optional field at all
    ("invalid model: only 'kimi/k3.1-pro' is allowed", "", False, True),  # unrelated field; not a number either
    ("temperature must be between 0 and 2", "", False, False),  # a range holding 0.2
    ("invalid value for temperature", "", False, True),  # names no allowed value
])
def test_the_allowed_values_rule_withdraws_only_a_temperature_hint(evidence_root, body_error, message, param, withdrawn):
    """A refusal withdraws the hint only when a statement bound to the exact field (or by its
    typed parameter) allows temperatures excluding the sent number. A prescription about another
    field, a mention elsewhere, a range holding the hint, or wording outside the grammar keeps
    the provider's error, and the same refusal never removes an explicit value."""
    # The payload carries the sibling fields these refusals name, as the donor's did: none of them goes.
    source = _seat_payload(top_p=0.9, response_format={"type": "json_object"})
    kwargs = {"param": param, "body_error": body_error, "source": source}
    (retry,), reasons = _plan_seat(message, **kwargs)
    if withdrawn:
        assert set(source) - set(retry) == {"temperature"} and reasons == [HINT_WITHDRAWAL_REASON]
    else:
        assert retry is None and reasons == []
    (explicit,), _reasons = _plan_seat(message, hint=False, **kwargs)
    assert explicit is None, "an explicit 0.2 is the caller's choice and this refusal the provider's answer"


def test_a_send_with_no_carrier_withdraws_the_hint_and_never_an_explicit_value(evidence_root, tmp_path):
    """With no reasoning carrier no candidate binds the send, so the withdrawal cannot be learned; it still happens."""
    source = _seat_payload()
    source.pop("extra_body")
    sent = []

    def create(**candidate):
        sent.append(copy.deepcopy(candidate))
        if "temperature" in candidate:
            raise _KimiRefusal()
        return _Response()

    _run_seat(create, source, tmp_path, "carrierless")
    assert len(sent) == 2 and sent[0]["temperature"] == 0.2 and set(source) - set(sent[1]) == {"temperature"}
    assert _learned_fields(evidence_root) == [], "nothing was bound to that success"
    assert _plan_carrierless(source, KIMI_REFUSAL, hint=False) == [None], "an explicit 0.2 carries no mark"
    assert _plan_carrierless(source, KIMI_REFUSAL, type="context_length_exceeded") == [None], "overflow is not its refusal"
    (retry, stop) = _plan_carrierless(source, KIMI_REFUSAL, KIMI_REFUSAL)
    assert set(source) - set(retry) == {"temperature"} and stop is None, "it goes once, then the provider's error stands"


def test_a_message_the_marker_paths_already_recover_keeps_its_old_path(evidence_root, tmp_path):
    """The withdrawal comes after every earlier repair, so a refusal the old ladder recovers
    keeps its old path. An enum-bounded effort refusal that also carries the prescription
    prose takes the old ladder — set the effort to the highest lower tier — and the hint
    is not withdrawn by the newer rule."""
    (retry,), reasons = _plan_seat("reasoning effort must be one of: low, medium; temperature can only be 1")
    assert retry is not None, "the old enum ladder answers this refusal"
    assert "temperature" in retry, "the withdrawal did not pre-empt it"
    assert retry["extra_body"]["reasoning"]["effort"] == "medium" and reasons == ["provider_prescribed_value"]


def test_a_thinking_refusal_with_a_prescription_keeps_the_thinking_path(evidence_root):
    """"model does not support thinking; temperature can only be 1" on a
    reasoning_effort carrier. The thinking-refusal shape is admitted by the legacy
    marker gate — no textual marker names it — and the old path answers by dropping
    the carrier. The withdrawal must not pre-empt it: dropping temperature alone
    would strand the request on the refused carrier and burn the resend on a repeat
    refusal."""
    source = {  # the standard effort carrier, not the nested one the seat payload uses
        "model": "kimi/k3.1-pro",
        "messages": [{"role": "user", "content": "Weigh the plan before the work starts."}],
        "temperature": 0.2, "max_tokens": 64, "tools": _tools(), "tool_choice": "auto",
        "reasoning_effort": "high",
    }
    message = "model does not support thinking; temperature can only be 1"
    (retry, second), reasons = _plan_seat(message, "temperature can only be 1", source=source)
    assert retry is not None, "the old thinking path answers this refusal"
    assert "reasoning_effort" not in retry, "the carrier went with the old thinking recovery"
    # Formerly the old loop dropped the knob with the carrier. "Can only be 1" states values: the
    # field is not refused, so the hint goes only on a refusal of its own, and an explicit value stays.
    assert retry["temperature"] == 0.2
    assert second is not None and "temperature" not in second and "reasoning_effort" not in second
    assert reasons == ["provider_unsupported_field", HINT_WITHDRAWAL_REASON]
    (explicit, stop), _reasons = _plan_seat(message, "temperature can only be 1", source=source, hint=False)
    assert "reasoning_effort" not in explicit and explicit["temperature"] == 0.2 and stop is None
