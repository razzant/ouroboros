"""Target notes and independent wakes retain the call branch's continuity boundaries."""
import asyncio
import copy
import json
import queue

import pytest

from tests.test_live_conversation import host_world as host_world
from tests.test_scratchpad_fold import install as install


def test_target_fold_and_authored_notes_survive_seed_and_physical_cut(install):
    from ouroboros.context import build_memory_sections
    from ouroboros.context_compaction import _capsule_metadata
    from ouroboros.context_renewal import physical_window_cut, renewal_seed
    from ouroboros.context_fit import estimate_context_prompt_tokens
    from ouroboros.main_context_authority import late_answer_notes_message, predecessor_notes_message
    from ouroboros.section_floor import HostSection, PROTECTED
    from tests.test_context_physical_cut import conversation
    from tests.test_context_renewal import _noted
    from tests.test_late_answer_asker_notes import _asker_folds_and_ends, ASKER
    from tests.test_scratchpad_fold import _tool, _blocks, scratchpad_block_sha256

    env, memory, registry = install
    for content in ('First scratchpad thought.', 'Second scratchpad thought.'):
        assert _tool(registry, content=content).status == 'ok'
    chosen = [scratchpad_block_sha256(block) for block in _blocks(memory)]
    assert _tool(registry, content='Accepted fold: both thoughts stay connected.', replaces=chosen).status == 'ok'
    sections = build_memory_sections(memory, partition='volatile')
    scratchpad = next(section for section in sections if '## Scratchpad (from' in section)
    assert isinstance(scratchpad, HostSection) and scratchpad.section_id in PROTECTED
    assert 'Accepted fold: both thoughts stay connected.' in scratchpad
    assert 'memory/scratchpad_blocks.json' in scratchpad and 'entry_id=' in scratchpad
    _asker_folds_and_ends(env.drive_root)
    late = late_answer_notes_message({'id': 'late-head', 'metadata': {'late_answer': {
        'task_id': ASKER, 'quiz_id': 'q1'}}}, drive_root=env.drive_root)
    assert late is not None
    # A predecessor row and an actor capsule use their real producers too.
    from tests.test_continuation_context_authority import _pull_source
    predecessor = predecessor_notes_message({'id': 'continued', 'predecessor_authority': {
        'source': _pull_source(ASKER)}}, drive_root=env.drive_root)
    assert predecessor is not None
    capsule = next(row for row in _noted(env.drive_root) if (_capsule_metadata(row)[1] or {}).get('authorship') == 'actor')
    rows = conversation()
    rows[0]['content'] += '\n' + scratchpad
    rows[2:2] = [late, predecessor, capsule]
    original = copy.deepcopy(rows)
    seed, _ = renewal_seed(rows)
    for note in (late, predecessor, capsule):
        assert note in seed
    budget = estimate_context_prompt_tokens(seed) // 2
    kept, receipt = physical_window_cut(seed, drive_root=env.drive_root, task_id='late-head',
                                       measure=estimate_context_prompt_tokens, usable=budget)
    assert receipt and estimate_context_prompt_tokens(kept) <= budget
    assert kept[0] == rows[0] and rows == original
    assert all(note in kept for note in (late, predecessor, capsule))
    assert any(row not in kept for row in seed[5:-2]), 'ordinary old dialogue must still yield'


@pytest.mark.parametrize('transport', ['call', 'text'])
@pytest.mark.parametrize('room', ['main', 'project'])
def test_terminal_relay_reaches_live_room_beside_main_wake(host_world, monkeypatch, transport, room):
    from ouroboros import owner_mailbox as mail
    from ouroboros.consciousness_authority import is_own_clock_wake
    from ouroboros.consciousness_wake import wake_task_metadata
    from ouroboros.loop_round_limits import _drain_incoming_messages
    from ouroboros.task_results import write_task_result
    from supervisor import active_activity, events_chat_delivery as delivery, workers
    from tests._budget_pause_exact_helpers import _loop_ctx
    from tests.test_direct_chat_turn_owner_control import _live_chat_agent

    host = host_world.host
    root, tid = host.ctx.drive_root, host.ctx.task_id
    chat_id = 1 if room == 'main' else host.ctx.current_chat_id
    monkeypatch.setattr(active_activity, '_DIRECT_ACTIVITY_REGISTRY', active_activity.DirectActivityRegistry())
    target = _live_chat_agent(monkeypatch, tid, chat_id=chat_id)
    if transport == 'call':
        target._realtime_turn = host.turn
        host.incoming = queue.Queue()
    wake = _live_chat_agent(monkeypatch, 'parallel-wake', chat_id=1)
    wake._current_task_metadata = wake_task_metadata('act', 'heartbeat')
    write_task_result(root, 'parallel-wake', 'running', chat_id=1, metadata=wake._current_task_metadata)
    assert is_own_clock_wake(workers.direct_chat_turn('parallel-wake')['metadata'])
    _live_chat_agent(monkeypatch, 'foreign-room', chat_id=2)
    _live_chat_agent(monkeypatch, 'ended-room-turn', accepting=False, chat_id=chat_id)
    answer = 'Finished work: its result is available while Main wakes independently.'
    write_task_result(root, 'finished', 'completed', result=answer, chat_id=chat_id, terminal_origin='model_final')
    emitted = {'task_id': 'finished', 'chat_id': chat_id, 'text': answer}
    for _ in range(2):
        assert delivery._register_delivered(type('Context', (), {'DRIVE_ROOT': root})(), 'final:finished:1234567890abcdef', emitted)
    [entry] = mail.drain_owner_entries(root, tid)
    assert entry['provenance'] == 'system' and entry['source_task_id'] == 'finished'
    assert len(mail.drain_owner_entries(root, 'parallel-wake')) == (room == 'main')
    assert mail.drain_owner_entries(root, 'foreign-room') == mail.drain_owner_entries(root, 'ended-room-turn') == []
    if transport == 'call':
        delivered = asyncio.run(host.mailbox())
        assert answer in json.dumps(delivered) and host.take_mail_answer()
        assert asyncio.run(host.mailbox()) == [] and not host.take_mail_answer()
    else:
        owner_ctx, limit = _loop_ctx(root, tid, direct=True)
        limit.messages, limit.owner_msg_seen = [], set()
        _drain_incoming_messages(limit.messages, queue.Queue(), root, tid, queue.Queue(),
                                 limit.owner_msg_seen, owner_ctx=owner_ctx, defer_content_ack=True)
        assert len(limit.messages) == 1 and answer in limit.messages[0]['content']
        assert not owner_ctx._owner_directives
    assert workers.direct_chat_turn('parallel-wake') is not None
