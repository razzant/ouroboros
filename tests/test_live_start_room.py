"""A first full wire leaves a work room; the same work's reconnect pays none."""
from dataclasses import replace
import math

import pytest

from ouroboros.contracts.live_conversation import LiveOptions
from ouroboros.realtime_setup import measure_session_wire, session_candidate
from tests.test_live_conversation import host_world as host_world


@pytest.mark.parametrize('first', [False, True])
def test_late_frame_gets_one_start_room_only_on_the_first_full_candidate(host_world, first):
    from tests.test_memory_view_lever import plan
    host = host_world.host
    source, _ = plan()
    host.plan = host.ctx.context_fit_plan = replace(host.plan, core=source.core, core_sha256=source.core_sha256,
        max_projection=source.max_projection, low_projection=source.low_projection,
        nano_projection=replace(source.max_projection, mode='nano'), output_reserve_tokens=10000)
    host.messages[:] = host.plan.messages_for('nano')
    options = LiveOptions(instructions='Late feature instructions and provider envelope. ' * 150)
    factory = host_world.provider['session']
    _, _, before, _ = measure_session_wire(host, host.plan, host.messages, host.schemas, factory, options)
    initial = before['fit']['measurement']['estimated_input_tokens']
    window = math.ceil((initial + 10000) / .94)
    host.model = replace(host.model, facts=replace(host.model.facts, conversation_window_tokens=window))
    host._session_opened = not first
    previous_system = host.messages[0]
    _, _, after = session_candidate(host, factory, options)
    measured, room = after['fit']['measurement'], after['work_room']
    assert room['first_send'] is first
    assert room['requested_tokens'] == (math.ceil(window / 8) if first else 0)
    assert room['remaining_tokens'] == window - 10000 - measured['estimated_input_tokens']
    assert room['deficit_tokens'] == 0, after
    if first:
        assert measured['estimated_input_tokens'] < initial
        assert room['remaining_tokens'] >= math.ceil(window / 8)
    else:
        assert host.messages[0] == previous_system, 'renewal may not buy another start room by narrowing memory'
        _, actual, _, _ = measure_session_wire(host, host.plan, host.messages, host.schemas, factory, options, _room_fact=False)
        assert measured['estimated_input_tokens'] == actual.prompt_tokens_estimate
        assert room['remaining_tokens'] < math.ceil(window / 8)


def test_start_with_less_room_is_not_a_physical_refusal(host_world):
    from ouroboros.contracts.realtime import RealtimeSetupError
    host = host_world.host
    options = LiveOptions(instructions='Held feature instructions. ' * 1200)
    factory = host_world.provider['session']
    # This fixture is already at its small memory floor; a large immutable
    # feature frame fits physically but cannot leave another eighth empty.
    _, _, baseline, _ = measure_session_wire(host, host.plan, host.messages, host.schemas, factory, options)
    window = baseline['fit']['measurement']['estimated_input_tokens'] + 10000 + 500
    host.model = replace(host.model, facts=replace(host.model.facts, conversation_window_tokens=window))
    _, _, result = session_candidate(host, factory, options)
    room = result['work_room']
    assert 0 <= room['remaining_tokens'] < room['requested_tokens'] and room['deficit_tokens'] > 0, result
    assert any(row.get('_host_context_kind') == 'context_start_room' and
               'Start room is smaller than requested' in row['content'] for row in host.messages)
    assert result['fit']['measurement']['capacity_deficit_tokens'] == 0
    with pytest.raises(RealtimeSetupError, match='context_capacity'):
        session_candidate(host, factory, replace(options, instructions=options.instructions * 2))
