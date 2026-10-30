"""Call captions stay in their room; ordinary replies still reach transports."""
import asyncio

from tests.test_live_conversation import host_world as host_world


def test_live_caption_source_stays_in_room_while_ordinary_reply_mirrors(host_world, monkeypatch):
    from ouroboros.dialogue_store import iter_rows
    from ouroboros.event_bus import CHAT_OUTBOUND
    from skills.telegram.plugin import _target_chat
    from supervisor import message_bus

    host = host_world.host
    frames, mirrored = [], []
    bridge = message_bus.get_bridge()
    monkeypatch.setattr(bridge, '_broadcast_fn', frames.append)
    def outbound(topic, event):
        if topic == CHAT_OUTBOUND:
            assert _target_chat({'TELEGRAM_MIRROR_MODE': 'all', 'TELEGRAM_CHAT_ID': '12345'}, event) == 12345
            mirrored.append(event)
    monkeypatch.setattr(message_bus, 'publish_event', outbound)
    async def run():
        await host.caption('owner', 'Owner call words', 'owner-caption')
        await host.caption('owner', 'Participant call words', 'heard-caption', heard=True)
        await host.caption('ouroboros', 'A spoken answer', 'spoken-caption', completed=True)
    asyncio.run(run())
    assert mirrored == [], 'live_call caption escaped onto the external transport bus'
    assert len(frames) == 3 and all(row['source'] == 'live_call' for row in frames)
    assert [row['role'] for row in frames] == ['user', 'system', 'assistant']
    assert frames[-1]['spoken']['status'] == 'played' and frames[-1]['narration'] is True
    assert frames[1]['system_type'] == 'call_heard'
    saved = [entry.row for entry in iter_rows(host.ctx.drive_root, 'chat')
             if entry.row.get('source') == 'live_call']
    assert len(saved) == 3 and saved[-1]['spoken']['item_id'] == 'spoken-caption'
    # Same words, ordinary typed source: forwarding is still enabled.
    assert bridge.send_message(host.turn.spec.chat_id, 'A spoken answer',
        role='assistant', task_id=host.ctx.task_id, progress_meta={'source': 'chat'})[0]
    assert len(mirrored) == 1 and mirrored[0]['text'] == 'A spoken answer'
    assert mirrored[0]['source'] == 'chat' and mirrored[0]['is_progress'] is False
