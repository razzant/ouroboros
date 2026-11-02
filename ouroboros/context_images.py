"""Physical image floors and exact-source placeholders; no content judgment."""
from __future__ import annotations

import copy


def image_placeholder(block, *, reader_available=True):
    caption = str(block.get('_caption') or '').strip()
    source = str(block.get('_source_path') or '').strip()
    return {'type': 'text', '_evicted_image': True, '_caption': caption, '_source_path': source,
            'text': ('[image evicted' + (f': {caption}' if caption else '')
            + ((f'; re-view: view_image path={source}' if reader_available else f'; path={source}')
               if source else '') + ']')}


def render_image_placeholders(rows, *, reader_available=True):
    """Rebind only marked placeholders to the delivering catalog, on a send copy."""
    return [{**row, 'content': [image_placeholder(block, reader_available=reader_available)
             if isinstance(block, dict) and block.get('_evicted_image') else block for block in row['content']]}
            if isinstance(row.get('content'), list) else row for row in rows]


def image_positions(messages):
    return [(mi, bi) for mi, row in enumerate(messages) if isinstance(row.get('content'), list)
            for bi, block in enumerate(row['content'])
            if isinstance(block, dict) and block.get('type') in {'image_url', 'image'}]


def evict_stale_image_blocks(messages, *, incoming=0, owner_mode='max', readable_only=False, reader_available=True):
    """Nano chosen by the owner batches >4 to 2; Max/Low keep their >10 to 5 floor."""
    from ouroboros.context_budget import (MAX_LIVE_IMAGE_BLOCKS, LIVE_IMAGE_BLOCKS_HIGH_WATER,
                                          NANO_LIVE_IMAGE_BLOCKS, NANO_LIVE_IMAGE_BLOCKS_HIGH_WATER)
    keep, high = (NANO_LIVE_IMAGE_BLOCKS, NANO_LIVE_IMAGE_BLOCKS_HIGH_WATER) if owner_mode == 'nano' else (MAX_LIVE_IMAGE_BLOCKS, LIVE_IMAGE_BLOCKS_HIGH_WATER)
    positions = image_positions(messages)
    total = len(positions) + max(0, int(incoming))
    if total <= high:
        return False
    changed = False
    for mi, bi in positions[:total - keep]:
        if readable_only and not messages[mi]['content'][bi].get('_source_path'):
            continue
        messages[mi]['content'][bi] = image_placeholder(messages[mi]['content'][bi], reader_available=reader_available)
        changed = True
    return changed


def fresh_call_images(messages, *, owner_mode='max'):
    """A newer stamped delivery replaces only that source's earlier pixels in this send copy.

    No stamp or no exact reader means no supersession. Canonical bytes stay intact.
    Neither filenames nor pixel similarity define a capture source.
    """
    rows, seen = copy.deepcopy(messages), set()
    evict_stale_image_blocks(rows, owner_mode=owner_mode)
    for mi, bi in reversed(image_positions(rows)):
        block = rows[mi]['content'][bi]
        source = block.get('_capture_source')
        if not source:
            continue
        if source in seen and block.get('_source_path'):
            rows[mi]['content'][bi] = image_placeholder(block)
        seen.add(source)
    return rows
