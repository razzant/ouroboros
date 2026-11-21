"""Accepted round events use the wire observation. Forced finals still use the lane's original projection."""
import json
from types import SimpleNamespace

import pytest

from ouroboros.llm import LLMClient
from ouroboros.loop_llm_call import call_llm_with_retry, forced_response_is_incomplete


CHAT = {"provider": "openrouter", "resolved_model": "m", "usage_model": "m"}
MISSING = object()


@pytest.fixture(autouse=True)
def no_tariff_lookup(monkeypatch):
    monkeypatch.setattr("ouroboros.pricing.estimate_cost_optional", lambda *args, **kwargs: None)


def accepted_round(root, chat, *, local=False):
    root.mkdir(exist_ok=True)
    calls, policy = [], {}

    def send(**kwargs):
        calls.append(1)
        return chat(**kwargs)

    message, _ = call_llm_with_retry(SimpleNamespace(chat=send), [{"role": "user", "content": "Hello."}],
        "local-model" if local else "m", None, "high", 1, root, "finish", 1, None, {},
        use_local=local, response_meta_out=policy)
    assert message["content"] == "Hello." and calls == [1]
    rows = [json.loads(line) for line in (root / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    row, = [row for row in rows if row["type"] == "llm_round"]
    return {k: row[k] for k in ("finish_reason", "finish_reason_origin") if k in row}, policy


@pytest.mark.parametrize("lane", ["chat", "anthropic", "gigachat"])
# Historic normalized values in Chat, Anthropic and GigaChat order.
@pytest.mark.parametrize("finish,expected,normalized", [
    (MISSING, {"finish_reason_origin": "missing"}, (MISSING, MISSING, MISSING)),
    (None, {"finish_reason": None}, (None, MISSING, MISSING)),
    ("stop", {"finish_reason": "stop"}, ("stop", "stop", "stop")),
    ("private_diagnostic", {"finish_reason_origin": "discarded"}, ("private_diagnostic",) * 3),
    (42, {"finish_reason_origin": "discarded"}, (None, "42", MISSING)),
    ("x" * 200, {"finish_reason_origin": "discarded"}, (None, "x" * 200, "x" * 64)),
])
def test_wire_finish_and_lane_policy_reach_their_own_consumers(tmp_path, lane, finish, expected, normalized):
    client = LLMClient(api_key="k")
    supplied = {} if finish is MISSING else {"finish_reason": finish}

    def normalize(**kwargs):
        if lane == "chat":
            choice = {"message": {"role": "assistant", "content": "Hello."}, **supplied}
            # A provider's usage extensions cannot speak for the designated choice.
            return client._normalize_remote_response({"choices": [choice],
                "usage": {"response_finish_reason": "length", "response_finish_reason_origin": "computed"}},
                CHAT, skip_cost_fetch=True)
        if lane == "anthropic":
            body = {"content": [{"type": "text", "text": "Hello."}],
                    **({"stop_reason": finish} if supplied else {})}
            return client._normalize_anthropic_response(body, {"provider": "anthropic", "usage_model": "m"})
        from gigachat.models import Choices, Messages
        # Construct a library response, including an SDK default absent from its supplied fields.
        # Pydantic v2 (the locked gigachat 0.2.3): ``model_construct`` keeps exactly this field set, so an absent
        # finish stays an SDK default (value None, never supplied) beside the supplied None/value pair.
        choice = Choices.model_construct(_fields_set={"message", "index", *supplied},
            message=Messages(role="assistant", content="Hello."), index=0, finish_reason=supplied.get("finish_reason"))
        assert ("finish_reason" in choice.model_fields_set) is bool(supplied)
        return client._normalize_gigachat_response(SimpleNamespace(choices=[choice]), {"provider": "gigachat"})

    fields, policy = accepted_round(tmp_path, normalize)
    assert fields == expected
    value = normalized[("chat", "anthropic", "gigachat").index(lane)]
    assert policy == {"finish_reason_present": value is not MISSING,
                      "finish_reason": None if value is MISSING else value, "tool_call_count": 0}
    assert forced_response_is_incomplete(policy) is (value is None)


@pytest.mark.parametrize("local", [True, False], ids=["local", "chat"])
@pytest.mark.parametrize("presence", ["missing", "default", "null", "stop", "unknown", "extension"])
def test_sdk_defaults_are_not_provider_fields(tmp_path, monkeypatch, local, presence):
    import httpx
    from openai import OpenAI

    choice = {"index": 0, "message": {"role": "assistant", "content": "Hello."}}
    if presence in {"null", "stop", "unknown"}:
        choice["finish_reason"] = "stop" if presence == "stop" else None
    body = {"id": "response", "model": "m", "created": 0, "object": "chat.completion", "choices": [choice]}
    if presence == "extension":
        body["usage"] = {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2,
                         "response_finish_reason": "length", "response_finish_reason_origin": "computed"}
    with OpenAI(api_key="test", base_url="https://model.invalid/v1",
                http_client=httpx.Client(transport=httpx.MockTransport(lambda _: httpx.Response(200, json=body)))) as api:
        sdk = api.chat.completions.create(model="m", messages=[{"role": "user", "content": "Hello."}])
    if presence == "default":
        # Also exercise a supported SDK shape with an explicit field default in its full dump.
        # The OpenAI SDK's own ``construct`` adds every key it is given to the field set, so the default is not
        # passed through it: the parsed choice's real field set is restored on the rebuilt one.
        item = sdk.choices[0]
        rebuilt = type(item).model_construct(**{**item.model_dump(), "message": item.message, "finish_reason": None})
        object.__setattr__(rebuilt, "__pydantic_fields_set__", set(item.model_fields_set))
        sdk.choices[0] = rebuilt
        assert "finish_reason" not in rebuilt.model_fields_set
        assert "finish_reason" in sdk.model_dump()["choices"][0]
    if presence == "unknown":
        sdk = SimpleNamespace(model_dump=sdk.model_dump, choices=sdk.model_dump()["choices"])  # no field-set witness
    dumped_choice = sdk.model_dump()["choices"][0]
    client = LLMClient(api_key="k")
    if local:
        monkeypatch.setattr(client, "_get_local_client", lambda: SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kw: sdk))))
        # Only the local send is synthetic. The lane consumes its ordinary returned SDK object.
        monkeypatch.setattr("ouroboros.llm_local._execute_candidate", lambda request, send, before: send())
        monkeypatch.setattr("ouroboros.local_model.get_manager", lambda: SimpleNamespace(
            serving_context_evidence=lambda: {}, measure_prepared_input=lambda candidate: None))
        chat = client.chat
    else:
        from ouroboros.llm_routing import _reply_dict
        def chat(**kwargs):
            return client._normalize_remote_response(_reply_dict(sdk, CHAT), CHAT, wire_completion=sdk, skip_cost_fetch=True)
    fields, policy = accepted_round(tmp_path, chat, local=local)
    expected = {"stop": {"finish_reason": "stop"}, "null": {"finish_reason": None},
                "unknown": {"finish_reason_origin": "unknown"}}
    assert fields == expected.get(presence, {"finish_reason_origin": "missing"})
    # The local lane historically kept usage extensions. They still cannot provide wire evidence.
    policy_present = presence in {"stop", "extension"} if local else "finish_reason" in dumped_choice
    assert policy["finish_reason_present"] is policy_present
    if local and presence == "extension":
        assert policy["finish_reason"] == "length"
    assert forced_response_is_incomplete(policy) is (policy_present and presence != "stop")


def test_assembled_terminal_and_responses_classification_reach_the_round(tmp_path):
    from ouroboros.llm_stream import consume_stream
    from tests._stream_wire_fixtures import WireResponse, chunk, sse
    from tests._openai_responses_fixtures import _body, _message, _target

    client = LLMClient(api_key="k")
    def stream(**kwargs):
        reply = consume_stream(WireResponse(sse(chunk({"content": "Hello."}), chunk({}, "stop"))))
        return client._normalize_remote_response(reply.model_dump(), CHAT, wire_completion=reply, skip_cost_fetch=True)

    fields, policy = accepted_round(tmp_path / "stream", stream)
    assert fields == {"finish_reason": "stop"} and not forced_response_is_incomplete(policy)
    def responses(**kwargs):
        return client._normalize_remote_response(_body(_message("Hello.")), _target(), skip_cost_fetch=True)

    fields, policy = accepted_round(tmp_path / "responses", responses)
    assert fields == {"finish_reason": "stop", "finish_reason_origin": "computed"}
    assert policy["response_status"] == "completed" and not forced_response_is_incomplete(policy)
