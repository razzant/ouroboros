"""Additive host ports keep exact task artifacts and reported Presence authorship."""
import asyncio
import io
import json
from dataclasses import replace
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from ouroboros import call_protocol as cp
from ouroboros.call_peers import LiveCallHub, CallPeerError
from tests.test_live_conversation import host_world as host_world


def synthetic_jpeg():
    image = Image.new('RGB', (1280, 720), '#fafafa')
    draw = ImageDraw.Draw(image)
    for i in range(24):
        draw.text((30, 20 + i * 27), f'Synthetic shared content row {i}: exact reader and independent facts.', fill='#202020')
        draw.rectangle((900, 20 + i * 27, 1220, 34 + i * 27), fill=(i * 9, 60, 140))
    out = io.BytesIO()
    image.save(out, format='JPEG', quality=70)
    return out.getvalue()


def test_retained_frame_is_exact_content_addressed_task_media(host_world):
    host = host_world.host
    data = synthetic_jpeg()
    path = Path(asyncio.run(host.retain_frame(data, 'image/jpeg', source='remote-peer')))
    assert path.read_bytes() == data and path.parent == host.wire_path.parent / 'chat_media'
    assert asyncio.run(host.retain_frame(data, 'image/jpeg')) == str(path)
    assert len(list(path.parent.iterdir())) == 1
    print('SYNTHETIC_FRAME ' + json.dumps({'width':1280, 'height':720, 'quality':0.70, 'bytes':len(data)}))
    with pytest.raises(Exception, match='video_format'):
        asyncio.run(host.retain_frame(data, 'text/plain'))


@pytest.mark.parametrize('label', ['', 'Reported participant'])
def test_host_caption_uses_reported_presence_label_or_meeting_audio(host_world, label):
    from ouroboros.presence_runner import presence_chat_id
    from tests._dialogue import chat_rows
    host = host_world.host
    key = 'synthetic:account:room:thread'
    host.task = {'id':host.ctx.task_id, 'chat_id':presence_chat_id(key), 'metadata':{'presence':{
        'binding_id':'synthetic-binding', 'transport_skill':'synthetic-transport', 'behavior_skill':'synthetic-behavior',
        'event':{'source_event_id':'meeting:join','provider':'synthetic','account_id':'account',
            'conversation_id':'room','thread_id':'thread','conversation_key':key,
            'actor':{'platform_actor_id':'meeting','display_name':'Call title is not a speaker'},
            'conversation':{},'message':{}}}}}
    host.turn.spec = replace(host.turn.spec, presence_binding_id='synthetic-binding',
                             chat_id=host.task['chat_id'], label='Call title is not a speaker')
    kwargs = {'speaker_label':label} if label else {}  # old-style caller exercises the default
    asyncio.run(host.caption('owner', 'Exact participant words', 'heard-1', heard=True, **kwargs))
    row = next(row for row in chat_rows(host.ctx.drive_root) if row.get('client_message_id') == 'meeting:join:heard:heard-1')
    assert row['sender_label'] == (label or 'meeting audio')
    assert row['transport']['actor']['display_name'] == (label or 'meeting audio')


@pytest.mark.parametrize('message', [
    {'type':'share', 'kind':'screen', 'sharing':'yes'},
    {'type':'share', 'kind':'unknown', 'sharing':True},
    {'type':'participant', 'label':{}, 'speaking':True},
    {'type':'participant', 'label':'A', 'speaking':1},
])
def test_share_and_participant_controls_are_validated(message):
    hub = LiveCallHub(cp.CallSpec('share-validation', 1, 'test', 'test'), rates=(24000,24000))
    try:
        hub.join(cp.PeerInfo('owner', cp.PEER_OWNER, True, True, video=True), 24000)
        hub.next_event(0)
        with pytest.raises(CallPeerError, match='invalid_call_control'):
            hub.receive_control('owner', message)
        valid = {'type':'share', 'kind':'screen', 'sharing':True}
        hub.receive_control('owner', valid)
        assert hub.next_event(0).message == valid
    finally:
        hub.close('test')
