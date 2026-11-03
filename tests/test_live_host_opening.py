"""Presence's observed event and the host's call opening retain separate authorship."""
import asyncio
import json
from dataclasses import replace

from tests.test_live_conversation import host_world as host_world


def test_presence_selected_input_has_one_labelled_opening_across_connections(host_world):
    from ouroboros.context import build_llm_messages
    from ouroboros.contracts.live_conversation import LiveOptions
    from ouroboros.live_conversation import ConversationHost, LiveConversationTurn
    from ouroboros.memory import Memory

    state = host_world
    spec = replace(state.host.turn.spec, opened_by='skill', presence_binding_id='meeting-binding')
    turn = LiveConversationTurn(state.feature, state.config, spec, publish=lambda _: None)
    task = {**state.task, '_presence_turn': True, 'text': 'The meeting participant joined.',
        'metadata': {'execution_transport': turn.request, 'presence': {
            'binding_id': 'meeting-binding', 'observed_text': 'The meeting participant joined.',
            'event': {'source_event_id': 'meeting-joined', 'provider': 'fixture',
                      'actor': {'display_name': 'Meeting participant'}}}}}
    task.pop('origin_message_ref')
    state.ctx.task_metadata = task['metadata']
    memory = Memory(drive_root=state.env.drive_root, repo_dir=state.env.repo_dir)
    messages, _ = build_llm_messages(state.env, memory, task, ctx=state.ctx,
        tool_schemas=state.host.tools.schemas(), route_resolver=turn.resolve_route)
    assert turn.opening not in json.dumps(messages)
    host = ConversationHost(turn, state.ctx, messages, state.host.tools, task, '', state.host.hub)
    async def consume(session):
        rows = json.loads(session.selected.messages_json)
        openings = [row for row in rows if turn.opening in str(row.get('content'))]
        assert len(openings) == 1
        assert "[Host fact: a live call opened; not the owner's words]" in openings[0]['content']
        assert '_host_context_kind' not in openings[0]
        assert any(row.get('_host_context_kind') == 'call_open' and turn.opening in str(row.get('content'))
                   for row in host.messages)
        assert any('[Observed Presence event]' in str(row.get('content')) and
                   'The meeting participant joined.' in str(row.get('content')) for row in rows)
    async def run():
        try:
            await host.session(consume, LiveOptions())
            await host.session(consume, LiveOptions())
        finally:
            await host.finish()
    asyncio.run(run())
    # The start path leaves wall-clock facts (owner, 18.10 "profile it"): admitted, context ready, session open.
    timing = host.trace['timing']
    assert timing['admitted_at'] <= timing['context_ready_at'] <= timing['session_open_at']
    assert [c['number'] for c in timing['connections']] == [1, 2] and all('open_at' in c for c in timing['connections'])
    wire = [json.loads(line) for line in host.wire_path.read_text(encoding='utf-8').splitlines()]
    assert any(json.loads(row['raw']).get('type') == 'call.timing' for row in wire if row['direction'] == 'host')
