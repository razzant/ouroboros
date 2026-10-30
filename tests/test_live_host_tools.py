"""The call's resident tools follow its fitted mode and explicit discovery."""
import asyncio
import json
from dataclasses import replace

import pytest

from ouroboros.contracts.live_conversation import LiveOptions
from ouroboros.tool_policy import initial_tool_schemas
from tests.test_live_conversation import host_world as host_world


@pytest.mark.parametrize('host_world', [('max', 500000, 'provider')], indirect=True)
def test_lower_mode_keeps_enabled_tools_and_rebinds_discovery(host_world):
    host = host_world.host
    names = lambda schemas: {schema['function']['name'] for schema in schemas}
    async def enable(name):
        await host.tool_batch([{'id': 'enable-' + name, 'type': 'function',
            'function': {'name': 'enable_tools', 'arguments': json.dumps({'tools': name})}}])
    async def wide(session):
        assert session.selected.mode == 'max'
        assert names(json.loads(session.selected.tools_json)) == names(host.tools.schemas()) | host.call_tools.keys()
    async def narrow(session):
        expected = names(initial_tool_schemas(host.tools, context_mode='nano')) | host.call_tools.keys() | {'read_file'}
        assert session.selected.mode == 'nano'
        assert names(json.loads(session.selected.tools_json)) == expected
        # Discovery must now append to the same schema list that the host seals.
        await enable('write_file')
        assert 'write_file' in names(host.schemas)
    async def reopened(session):
        assert {'read_file', 'write_file'} <= names(json.loads(session.selected.tools_json))
    async def run():
        await host.session(wide, LiveOptions())
        # A schema omitted from the earlier resident envelope is explicitly loaded.
        host.schemas[:] = [schema for schema in host.schemas if schema['function']['name'] != 'read_file']
        await enable('read_file')
        assert 'read_file' in host.ctx._actor_loaded_tool_names
        host.model = replace(host.model, facts=replace(host.model.facts,
            conversation_window_tokens=80000, basis='estimator'))
        await host.session(narrow, LiveOptions())
        await host.session(reopened, LiveOptions())
    asyncio.run(run())
