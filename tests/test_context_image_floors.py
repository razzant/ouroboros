"""Owner Nano's image floor and call-only same-source projection are distinct."""
import copy
import pytest

from ouroboros.context_images import evict_stale_image_blocks, image_positions
from ouroboros.realtime_setup import rows_for_engine
from tests.test_live_conversation import host_world as host_world


def images(n, source='screen'):
    return [{'role': 'user', 'content': [{'type': 'image_url', 'image_url': {'url': f'data:image/png;base64,{i}'},
              '_caption': f'Exact frame {i}', '_source_path': f'/original/{i}.png',
              **({'_capture_source': source} if source else {})}]} for i in range(n)]


@pytest.mark.parametrize('mode,high,keep', [('nano', 4, 2), ('low', 10, 5), ('max', 10, 5)])
def test_mode_image_floors_are_batched_and_keep_exact_readers(mode, high, keep):
    rows = images(high); before = copy.deepcopy(rows)
    assert not evict_stale_image_blocks(rows, owner_mode=mode) and rows == before
    rows += images(1, 'camera')
    assert evict_stale_image_blocks(rows, owner_mode=mode)
    assert len(image_positions(rows)) == keep
    assert all(f'/original/{i}.png' in rows[i]['content'][0]['text'] for i in range(high+1-keep))


def test_only_fresh_call_projection_collapses_stamped_sources_and_never_changes_canonical():
    rows = [*images(3), *images(2, 'camera'), *images(2, '')]
    before = copy.deepcopy(rows)
    assert len(image_positions(rows_for_engine(rows, vision=True))) == 7
    sent = rows_for_engine(rows, vision=True, fresh=True)
    assert len(image_positions(sent)) == 4  # latest screen, latest camera, two unstamped
    assert '/original/0.png' in sent[0]['content'][0]['text']
    assert rows == before
    assert not image_positions(rows_for_engine(rows, vision=False, fresh=True))
    unreadable = images(2)
    unreadable[0]['content'][0].pop('_source_path')
    assert len(image_positions(rows_for_engine(unreadable, vision=True, fresh=True))) == 2


def test_auto_attach_stamps_producer_and_declared_target_never_the_filename(monkeypatch):
    import json
    from types import SimpleNamespace
    from ouroboros.loop_tool_execution import _maybe_auto_attach_image
    captured = []
    def attach(ctx, path, **kwargs):
        captured.append(kwargs.get('capture_source'))
        return True, 'attached'
    monkeypatch.setattr('ouroboros.tools.vision.attach_local_image_to_context', attach)
    for path, display in [('/first.png', 0), ('/second.png', 0), ('/third.png', 3), ('/fourth.png', None)]:
        payload = {'auto_attach_image': path, **({'display': {'index': display}} if display is not None else {})}
        _maybe_auto_attach_image({'fn_name': 'ext_desktop_screenshot', 'result': json.dumps(payload)},
                                 SimpleNamespace(_ctx=SimpleNamespace(messages=[])))
    assert captured[0] == captured[1] != captured[2] and captured[3] is None
    assert json.loads(captured[0]) == ['ext_desktop_screenshot', 0]


@pytest.mark.parametrize('owner,expected', [('nano', 2), ('max', 5), ('low', 5)])
def test_text_delivery_uses_owner_mode_not_task_local_nano(owner, expected):
    from types import SimpleNamespace
    from ouroboros.loop_messages import _append_or_merge_user_content
    rows = images(4)
    stamp = SimpleNamespace(context_fit_plan=SimpleNamespace(preferred_mode=owner, initial_mode='nano'))
    _append_or_merge_user_content(rows, images(1, 'camera')[0]['content'], stamp=stamp)
    assert len(image_positions(rows)) == expected


@pytest.mark.parametrize('owner,expected', [('nano', 2), ('max', 5)])
def test_call_requested_pixels_use_the_same_owner_image_floor(host_world, owner, expected):
    import asyncio
    from dataclasses import replace
    host = host_world.host
    host.plan = replace(host.plan, preferred_mode=owner)
    async def deliver():
        for row in images(5, source=""):
            await host.observe_context(row)
    asyncio.run(deliver())
    assert len(image_positions(host.messages)) == 5  # canonical call rows never evict mid-connection
    assert len(image_positions(rows_for_engine(host.messages, vision=True, fresh=True, owner_mode=owner))) == expected


def test_owner_image_numbers_share_the_budget_policy_owner():
    from ouroboros.context_budget import (NANO_LIVE_IMAGE_BLOCKS, NANO_LIVE_IMAGE_BLOCKS_HIGH_WATER,
        MAX_LIVE_IMAGE_BLOCKS, LIVE_IMAGE_BLOCKS_HIGH_WATER)
    assert (NANO_LIVE_IMAGE_BLOCKS, NANO_LIVE_IMAGE_BLOCKS_HIGH_WATER) == (2, 4)
    assert (MAX_LIVE_IMAGE_BLOCKS, LIVE_IMAGE_BLOCKS_HIGH_WATER) == (5, 10)
