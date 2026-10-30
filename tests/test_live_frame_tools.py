"""The call frame's ordinary verbs are resident before the first connection."""
import asyncio
import json

from tests.test_live_conversation import host_world as host_world, packages as packages


def test_loaded_feature_declares_the_frames_ordinary_tools(packages):
    from ouroboros.live_conversation import loaded_feature
    from skills.live_conversation.driver import CALL_INSTRUCTIONS, INITIAL_TOOLS

    assert loaded_feature('live_conversation:call')['initial_tools'] == INITIAL_TOOLS
    assert set(INITIAL_TOOLS) == {'send_user_message', 'schedule_subagent', 'get_task_result'}
    assert all('`' + name + '`' in CALL_INSTRUCTIONS for name in INITIAL_TOOLS)


def test_first_nano_manifest_includes_frame_and_feature_tools_and_prices_them(host_world, record_property):
    from ouroboros.context_fit import tool_schema_tokens
    from ouroboros.live_conversation import ConversationHost
    from ouroboros.tool_policy import initial_tool_schemas
    from skills.live_conversation.driver import INITIAL_TOOLS, call_tools, session_options

    state = host_world
    state.feature.update(initial_tools=INITIAL_TOOLS, call_tools=call_tools(state.env.drive_root / 'call-state'))
    host = ConversationHost(state.host.turn, state.ctx, list(state.host.messages), state.host.tools,
                            state.task, '', state.host.hub)
    try:
        names = [row['function']['name'] for row in host.schemas]
        assert set(INITIAL_TOOLS) | {'end_call', 'set_call_languages', 'attend'} <= set(names)
        assert len(names) == 9 and len(set(names)) == 9
        base = initial_tool_schemas(host.tools, context_mode='nano')
        before = [*base, *(s for s in host.schemas if s['function']['name'] in host.call_tools)]
        added = tool_schema_tokens(host.schemas) - tool_schema_tokens(before)
        record_property('schema_tokens_before', tool_schema_tokens(before))
        record_property('schema_tokens_after', tool_schema_tokens(host.schemas))
        record_property('schema_tokens_added', added)
        for schema in host.schemas:
            if schema['function']['name'] in INITIAL_TOOLS:
                record_property(schema['function']['name'] + '_tokens', tool_schema_tokens([schema]))
        assert added > 0
        async def inspect(session):
            rows = json.loads(session.selected.messages_json)
            assert all('_host_context_kind' not in row for row in rows)
            [notice] = [r['content'] for r in host.messages if r.get('_host_context_kind') == 'tool_manifest']
            assert f'Loaded now ({len(names)}):' in notice and 'Not loaded does not mean unavailable.' in notice
            assert all(name in notice for name in names)
            assert 'cover most tasks' not in notice and 'core tools' not in notice
            assert set(names) == {s['function']['name'] for s in json.loads(session.selected.tools_json)}
        asyncio.run(host.session(inspect, session_options(host)))
        host._reselect_tools('nano')
        assert [s['function']['name'] for s in host.schemas] == names
        assert len([r for r in host.messages if r.get('_host_context_kind') == 'tool_manifest']) == 1
        async def enable(session):
            await host.tool_batch([{'id': 'load-reader', 'type': 'function', 'function': {
                'name': 'enable_tools', 'arguments': '{"tools":"read_file"}'}}])
        async def reloaded(session):
            rows = json.loads(session.selected.messages_json)
            assert all('_host_context_kind' not in row for row in rows)
            [notice] = [r['content'] for r in host.messages if r.get('_host_context_kind') == 'tool_manifest']
            sent = json.loads(session.selected.tools_json)
            assert len(sent) == len(names) + 1
            assert f'Loaded now ({len(sent)}):' in notice and 'read_file' in notice
        async def reopen():
            await host.session(enable, session_options(host))
            await host.session(reloaded, session_options(host))
        asyncio.run(reopen())
    finally:
        asyncio.run(host.finish())
        host.executor.shutdown(wait=True)
