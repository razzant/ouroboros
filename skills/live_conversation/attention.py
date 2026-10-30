"""The mind chooses when live sources are delivered; a frame never offers a turn."""
from __future__ import annotations

import base64
import json
import math
from time import monotonic
from hashlib import sha256

from ouroboros.contracts.live_conversation import CallTool, Video
from ouroboros.contracts.realtime import RealtimeSetupError
from ouroboros.tool_call_log import current_tool_call_id


FRAME_UNAVAILABLE = 'The shared frame could not be saved; ask what is on the screen.'
FRAME_NOT_SAVED = 'The shared frame could not be saved as a file.'


class Attention:
    def __init__(self, host):
        self.host = host
        self.mode = 'turns'
        self.frames = {}
        self.kinds = {}
        self.seen = {}
        self.looks = {}
        self.revisit_at = None

    def frame(self, event):
        self.frames[event.peer_id] = Video(event.data, event.mime, event.peer_id)

    def forget(self, peer):
        self.frames.pop(peer, None)
        self.kinds.pop(peer, None)
        self.seen.pop(peer, None)

    async def retained(self, frame):
        retain = getattr(self.host, 'retain_frame', None)
        try:
            path = await retain(frame.data, frame.mime, source=frame.source) if retain else ''
        except (OSError, RealtimeSetupError) as exc:
            if isinstance(exc, RealtimeSetupError) and exc.code != 'frame_retention':
                raise
            path = ''
        return path or ''  # a failed save is said in words on every route; a seeing route still gets its pixels

    async def deliver(self, session, *, requested=False):
        """An input/offer boundary, or stream delivery. No response is opened here."""
        if self.mode == 'asked' and not requested:
            return []
        rows = []
        for source, frame in tuple(self.frames.items()):
            digest = sha256(frame.data).hexdigest()
            if self.seen.get(source) == digest:
                continue
            # Input-start, stream and offer tasks may overlap at a socket/store await.
            # Claim this delivery before yielding, without holding a lock across host custody.
            self.seen[source] = digest
            try:
                if self.host.model.vision:
                    await session.send(frame)
                else:
                    path = await self.retained(frame)
                    detail = (f'{path}. This engine is blind; vlm_query with this file_path reads this exact remote frame.'
                              if path else FRAME_UNAVAILABLE)
                    row = {'role': 'user', 'content': '[Host fact; not owner words]\n'
                        f'Latest {self.kinds.get(source, "shared frame")} from peer {source}: {detail}'}
                    await self.host.observe_context(row)
                    rows.append({**row, '_canonical': True})
            except BaseException:
                if self.seen.get(source) == digest:
                    self.seen.pop(source, None)
                raise
        return rows

    async def attend(self, arguments):
        if (not isinstance(arguments, dict) or set(arguments) - {'see', 'look', 'revisit_after_s'}
                or arguments.get('see') not in (None, 'turns', 'stream', 'asked')
                or arguments.get('look') is not None and type(arguments['look']) is not bool):
            raise ValueError('attend takes see (turns, stream, asked or null) and look (boolean or null)')
        revisit = arguments.get("revisit_after_s")
        if revisit is not None and (type(revisit) not in (int, float) or not math.isfinite(revisit) or revisit < 0):
            raise ValueError("revisit_after_s must be finite nonnegative seconds or null")
        before = self.mode
        self.mode = arguments.get('see') or self.mode
        result = {'mode': self.mode, 'changed': self.mode != before}
        if revisit is not None:
            self.revisit_at = monotonic() + revisit
            result['revisit_after_s'] = revisit
            result['revisit'] = ('One request gives one look at the first quiet moment after that time, even if speech '
                                 'or another offer intervenes. Ask again from that look to keep watching.')
        if arguments.get('look'):
            frames, images = [], []
            for source, frame in tuple(self.frames.items()):
                path = await self.retained(frame)
                frames.append({'source': source, 'kind': self.kinds.get(source, ''),
                               **({'file_path': path} if path else
                                  {'error': FRAME_NOT_SAVED if self.host.model.vision else FRAME_UNAVAILABLE})})
                digest = sha256(frame.data).hexdigest()
                if self.host.model.vision and self.seen.get(source) != digest:
                    images.append({'role': 'user', '_call_frame_source': source, '_call_frame_digest': digest, 'content': [
                        {'type': 'text', 'text': f'[attend result; peer {source}; exact frame: {path or "not saved"}]'},
                        {'type': 'image_url', 'image_url': {'url': 'data:' + frame.mime + ';base64,'
                            + base64.b64encode(frame.data).decode()}, '_source_path': path,
                            '_capture_source': 'attend:' + source}]})
                    self.seen[source] = digest
                elif self.host.model.vision:
                    frames[-1]['already_visible'] = True
            result['frames'] = frames
            result['reader'] = 'vlm_query(file_path=...)' if not self.host.model.vision else 'images follow; unchanged frames are already visible'
            # Tool text keeps exact source paths in canonical history. The ordinary
            # ToolOutputs image lane carries their pixels, without changing CallTool's str result.
            self.looks[current_tool_call_id()] = images
        return json.dumps(result, ensure_ascii=False)

    def revisit_due(self):
        return self.revisit_at is not None and monotonic() >= self.revisit_at

    def include_images(self, rows):
        """A refitted look already brings its exact pixels with the pending offer."""
        for row in rows:
            if (row.get("_call_frame_source") and row.get("_call_frame_digest")
                    and isinstance(row.get("content"), list)
                    and any(part.get("type") == "image_url" for part in row["content"])):
                self.seen[row["_call_frame_source"]] = row["_call_frame_digest"]

    def tool_outputs(self, calls, changes):
        """Attach requested images once; a deliberate look continues even on a blind route."""
        images, continue_ids = [], set()
        for call in calls:
            if call['function']['name'] != 'attend':
                continue
            try:
                result = next(row['content'] for row in changes if row.get('tool_call_id') == call['id'])
                looked = 'frames' in json.loads(result)
            except (StopIteration, TypeError, ValueError):
                looked = False
            if looked:
                continue_ids.add(call['id'])
                images.extend(self.looks.pop(call['id'], []))
        return images, tuple(cid for cid in changes.silent if cid not in continue_ids)


def attend_tool():
    async def attend(host, arguments):
        return await host.state['attention'].attend(arguments)

    return CallTool('attend', 'Choose how I watch: turns (latest unseen frame with each turn), stream '
        '(every new frame), asked (only when I look). look=true returns the latest live sources and continues '
        'this turn; unchanged images are already visible. With no change, finish this turn without words. '
        'Null means no change; false does not ask to look. revisit_after_s names seconds until one offer with the current view, '
        'at the next quiet moment. There is no default or repeat. Speech and other offers do not cancel it. '
        'One request gives one look; ask again from that look to keep watching.',
        {'type': 'object', 'properties': {
            'see': {'type': ['string', 'null'], 'enum': ['turns', 'stream', 'asked', None]},
            'look': {'type': ['boolean', 'null']},
            'revisit_after_s': {'type': ['number', 'null'], 'minimum': 0}},
         'required': ['see', 'look', 'revisit_after_s'], 'additionalProperties': False}, attend, silent=True)
