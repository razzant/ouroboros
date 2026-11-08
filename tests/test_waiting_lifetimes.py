"""Every class waiting for the mind has an actual closure tool in its receiving focus.

This is the cognitive counterpart of test_persistence_lifetimes: an addressed
subject has a closing act and a natural occasion, rather than a host expiry.
"""
from copy import deepcopy
from types import SimpleNamespace

import pytest

from ouroboros.memory_waiting import WAITING_CLASSES
from ouroboros.tools.registry import ToolContext, ToolRegistry


def assert_closure_acts(classes, tools):
    for name, cls in classes.items():
        assert cls["focus"] == "integrating" and cls["moment"] and cls["reader"], name
        schema = tools.get_schema_by_name(cls["action"])
        assert schema, f"{name} has no closure act: {cls['action']}"
        parameter = schema["function"]["parameters"]
        assert cls["parameter"], f"{name} names no act parameter"
        for field in cls["parameter"]:
            assert field in parameter.get("properties", {}), f"{name} has no act parameter: {'.'.join(cls['parameter'])}"
            parameter = parameter["properties"][field]


@pytest.mark.parametrize("metadata,direct", [({}, False), ({}, True),
    ({"usage_category": "consciousness"}, False),
    ({"presence": {"binding_id": "p", "event": {"provider": "telegram", "conversation_id": "c"}}}, False)])
def test_every_waiting_class_has_a_closure_act_in_an_integrating_focus(tmp_path, monkeypatch, metadata, direct):
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "advanced")
    tools = ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path)
    tools.set_context(ToolContext(repo_dir=tmp_path, drive_root=tmp_path, task_id="mind0001", current_chat_id=1,
                                  task_metadata=metadata, is_direct_chat=direct))
    assert_closure_acts(WAITING_CLASSES, tools)
    assert_closure_acts({**WAITING_CLASSES, "a_new_class": WAITING_CLASSES["notes"]}, tools)
    fabricated = {**WAITING_CLASSES, "orphan": {**WAITING_CLASSES["notes"], "action": "no_such_closure_act"}}
    with pytest.raises(AssertionError, match="orphan has no closure act"):
        assert_closure_acts(fabricated, tools)
    for name, cls in WAITING_CLASSES.items():
        schemas = {row["action"]: deepcopy(tools.get_schema_by_name(row["action"])) for row in WAITING_CLASSES.values()}
        parent = schemas[cls["action"]]["function"]["parameters"]
        for field in cls["parameter"][:-1]:
            parent = parent["properties"][field]
        del parent["properties"][cls["parameter"][-1]]
        with pytest.raises(AssertionError, match=f"{name} has no act parameter"):
            assert_closure_acts(WAITING_CLASSES, SimpleNamespace(get_schema_by_name=schemas.get))
