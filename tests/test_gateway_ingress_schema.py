"""ABI 7.0 (ABI-3, owner Q7=A): the executable gateway ABI.

JSON Schema is DERIVED from the contract TypedDicts (never hand-written) and
validation applies on INGRESS only: WS chat/command frames and the typed HTTP
request bodies. Egress and history replay are never validated (the stored
axis: legacy rows keep replaying). GATEWAY_ABI_VERSION carries the ABI version
separately from the product version.
"""

from __future__ import annotations

import pytest

from ouroboros.gateway.schema import (
    GATEWAY_ABI_VERSION,
    json_schema_for,
    validate_ingress,
)

# The contracts' own TypedDict flavour: typing on 3.11+ (3.11 keeps no
# __orig_bases__ on a TypedDict subclass), typing_extensions on 3.10.
from ouroboros.gateway.contracts import NotRequired, Required, TypedDict


class _InheritedBase(TypedDict):
    needed: str
    optional: NotRequired[str]


class _InheritedRequest(_InheritedBase, total=False):
    extra: str
    pinned: Required[int]


class _OptionalBase(TypedDict, total=False):
    optional: str
    pinned: Required[int]


class _TotalRequest(_OptionalBase):
    needed: str
    extra: NotRequired[str]


class TestDerivation:
    def test_optional_base_keys_stay_optional_in_a_total_subclass(self):
        assert json_schema_for(_TotalRequest)["required"] == ["needed", "pinned"]
        assert validate_ingress({"needed": "x", "pinned": 1}, _TotalRequest) == []
        assert validate_ingress({"needed": "x"}, _TotalRequest) == ["pinned is required"]

    def test_chat_inbound_schema_shape(self):
        from ouroboros.gateway.contracts import ChatInbound

        schema = json_schema_for(ChatInbound)
        assert schema["type"] == "object"
        assert schema["additionalProperties"] is True  # open ABI evolution
        assert schema["required"] == ["content", "type"]
        assert schema["properties"]["type"] == {"enum": ["chat"]}
        assert schema["properties"]["content"] == {"type": "string"}
        assert schema["properties"]["chat_id"] == {"type": "integer"}
        assert schema["properties"]["force_plan"] == {"type": "boolean"}

    def test_task_create_request_schema_is_derived_not_hand_written(self):
        from ouroboros.gateway.contracts import TaskCreateRequest

        schema = json_schema_for(TaskCreateRequest)
        assert schema["required"] == ["description"]
        assert schema["properties"]["disabled_tools"] == {
            "type": "array", "items": {"type": "string"}}
        assert schema["properties"]["service_teardown"] == {"enum": ["stop", "keep"]}
        # Nested TypedDict recurses; its Required[...] key is enforced.
        executor = schema["properties"]["executor_ref"]
        assert executor["type"] == "object"
        assert executor["required"] == ["type"]

    def test_requiredness_follows_the_declaring_class_and_qualifiers(self):
        schema = json_schema_for(_InheritedRequest)
        assert schema["required"] == ["needed", "pinned"]
        assert validate_ingress({"needed": "x"}, _InheritedRequest) == ["pinned is required"]
        assert validate_ingress({"pinned": 1}, _InheritedRequest) == ["needed is required"]
        assert validate_ingress({"needed": "x", "pinned": 1}, _InheritedRequest) == []

    def test_optional_none_maps_to_anyof_null(self):
        from ouroboros.gateway.contracts import ChatOutbound

        schema = json_schema_for(ChatOutbound)
        prop = schema["properties"]["accounted_upper_bound_usd"]
        assert {"type": "null"} in prop["anyOf"]

    def test_abi_version_is_the_70_break_and_not_the_product_version(self):
        import pathlib

        assert GATEWAY_ABI_VERSION == "7.0"
        product = (pathlib.Path(__file__).resolve().parents[1] / "VERSION").read_text().strip()
        assert GATEWAY_ABI_VERSION != product


class TestValidator:
    def test_unknown_keys_pass_declared_keys_are_typed(self):
        from ouroboros.gateway.contracts import ChatInbound

        ok = {"type": "chat", "content": "hi", "future_field": {"x": 1}}
        assert validate_ingress(ok, ChatInbound) == []
        bad = {"type": "chat", "content": "hi", "chat_id": "5"}
        errors = validate_ingress(bad, ChatInbound)
        assert errors and "chat_id" in errors[0]

    def test_missing_required_message_keeps_the_historical_form(self):
        from ouroboros.gateway.contracts import TaskCreateRequest

        errors = validate_ingress({"text": "legacy"}, TaskCreateRequest)
        assert errors == ["description is required"]

    def test_bool_is_not_an_integer_or_number(self):
        from ouroboros.gateway.contracts import ChatInbound, TaskCreateRequest

        assert validate_ingress(
            {"type": "chat", "content": "x", "chat_id": True}, ChatInbound)
        assert validate_ingress(
            {"description": "x", "timeout": True}, TaskCreateRequest)
        # An integer IS a JSON number where float is declared.
        assert validate_ingress(
            {"description": "x", "timeout": 5}, TaskCreateRequest) == []

    def test_non_object_payload_is_refused(self):
        from ouroboros.gateway.contracts import CommandInbound

        assert validate_ingress(["command"], CommandInbound) == [
            "payload must be a JSON object"]


class TestHttpIngress:
    def test_tasks_create_reports_inherited_required_field_as_schema_error(self):
        import asyncio
        import json
        from types import SimpleNamespace

        from ouroboros.gateway.tasks import api_tasks_create

        async def body():
            return {"text": "legacy"}

        response = asyncio.run(api_tasks_create(SimpleNamespace(json=body)))
        assert response.status_code == 400
        payload = json.loads(response.body)
        assert payload["schema_errors"] == ["description is required"]

    def test_tasks_create_rejects_wrong_types_before_processing(self):
        import asyncio
        import json as _json
        from types import SimpleNamespace

        from ouroboros.gateway.tasks import api_tasks_create

        async def _json_body():
            return {"description": "x", "disabled_tools": "not-a-list"}

        request = SimpleNamespace(json=_json_body)
        response = asyncio.run(api_tasks_create(request))
        assert response.status_code == 400
        payload = _json.loads(response.body)
        assert "disabled_tools" in payload["error"]
        assert payload["schema_errors"]

    def test_hurry_refuses_a_non_string_request_id_instead_of_coercing(self):
        import asyncio
        import json as _json
        from types import SimpleNamespace

        from ouroboros.gateway.task_hurry import api_task_hurry

        async def _json_body():
            return {"request_id": 5}

        request = SimpleNamespace(
            json=_json_body, path_params={"task_id": "task-x1"})
        response = asyncio.run(api_task_hurry(request))
        assert response.status_code == 400
        payload = _json.loads(response.body)
        assert payload["reason_code"] == "invalid_request_body"
        assert "request_id" in payload["error"]

    def test_update_apply_enforces_the_required_strategy(self):
        import asyncio
        import json as _json
        from types import SimpleNamespace

        from ouroboros.gateway.control import api_update_apply

        async def _json_body():
            return {}

        response = asyncio.run(api_update_apply(SimpleNamespace(json=_json_body)))
        assert response.status_code == 400
        assert "strategy is required" in _json.loads(response.body)["error"]

    def test_provider_test_type_gate(self):
        import asyncio
        import json as _json
        from types import SimpleNamespace

        from ouroboros.gateway.models import api_provider_test

        async def _json_body():
            return {"provider_id": 7}

        response = asyncio.run(api_provider_test(SimpleNamespace(json=_json_body)))
        assert response.status_code == 400
        assert "provider_id" in _json.loads(response.body)["error"]


@pytest.mark.parametrize("msg", [
    {"type": "chat", "content": "hello", "chat_id": 1},
    {"type": "command", "cmd": "status"},
])
def test_ws_ingress_admits_the_shipped_client_shapes(msg):
    from ouroboros.gateway.contracts import ChatInbound, CommandInbound

    td = ChatInbound if msg["type"] == "chat" else CommandInbound
    assert validate_ingress(msg, td) == []


def test_history_replay_is_not_an_ingress_surface():
    """The validator is wired ONLY on inbound seams: gateway/history.py (egress
    replay of stored legacy rows) must never import it — replay rejection is
    exactly what the ABI-3 stored axis forbids."""
    import inspect

    from ouroboros.gateway import history

    source = inspect.getsource(history)
    assert "validate_ingress" not in source
