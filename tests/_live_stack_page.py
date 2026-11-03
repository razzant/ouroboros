"""A device-free page socket: real gateway codecs and sample-counted playback.

Mirrors call.js / PlaybackBuffer: 48 kHz, 20 ms capture, 40 ms jitter,
100 ms played receipts (also at drain/flush), monotonic receipts per item.
No device, browser, window or operator state is opened.
"""
import asyncio
import json
from collections import deque
from functools import lru_cache

from ouroboros import call_protocol as cp


@lru_cache
def jpeg(label='synthetic'):
    from tests.test_live_proactive_host import synthetic_jpeg
    data = synthetic_jpeg()
    comment = label.encode()
    return data[:2] + b'\xff\xfe' + (len(comment) + 2).to_bytes(2, 'big') + comment + data[2:]


class Page:
    def __init__(self):
        self.incoming = asyncio.Queue()
        self.closed, self.controls, self.frames = [], [], []
        self.items, self.played, self.dirty = {}, {}, set()
        self.queue = deque()
        self.epoch = self.seq = self.video_seq = 0
        self.queued = self.waited = self.since_report = 0
        self.playing = self.muted = False
        self.send({'type': 'hello', 'protocol': 1, 'role': 'owner', 'mic': True,
                   'speaker': True, 'video': True, 'rate': 48000, 'label': 'Owner'})

    async def accept(self): pass

    def send(self, value):
        if isinstance(value, dict) and value.get('type') == 'mute':
            self.muted = value['muted']
        self.incoming.put_nowait({'type': 'websocket.receive',
            **({'bytes': value} if isinstance(value, bytes) else {'text': json.dumps(value)})})

    async def receive(self):
        return await self.incoming.get()

    async def close(self, **reason):
        self.closed.append(reason)
        self.incoming.put_nowait({'type': 'websocket.disconnect'})

    def acknowledge(self):
        for item in self.dirty:
            self.send({'type': 'played', 'item_id': item, 'played_ms': self.played[item] * 1000 // 48000})
        self.dirty.clear()
        self.since_report = 0

    def flush(self, epoch):
        if not 0 < (epoch - self.epoch) % 65536 < 32768:
            return
        self.acknowledge()
        self.epoch = epoch
        self.queue.clear()
        self.items.clear()
        self.queued = self.waited = 0
        self.playing = False

    async def send_text(self, text):
        msg = json.loads(text)
        self.controls.append(msg)
        if msg['type'] == 'flush':
            self.flush(msg['epoch'])
        if (msg['type'] == 'caption' and msg['speaker'] == 'ouroboros' and not msg['final']
                and msg.get('epoch', self.epoch) == self.epoch):
            self.items[msg.get('stream', 0)] = msg['item_id']

    async def send_bytes(self, data):
        frame = cp.decode_frame(data)
        self.frames.append(frame)
        self.flush(frame.epoch)
        if frame.epoch != self.epoch:
            return
        assert frame.rate == 48000 and frame.stream in self.items
        samples = len(frame.pcm) // 2
        self.queue.append([self.items[frame.stream], samples])
        self.queued += samples

    def capturing(self):
        """The page captures only in its current live state and while unmuted (call.js)."""
        state = next((v['state'] for v in reversed(self.controls) if 'state' in v), '')
        return state == 'live' and not self.muted

    def tick(self, *, capture=True):
        if capture:
            self.send(cp.encode_frame(cp.AudioFrame(cp.FRAME_AUDIO_IN, self.epoch, self.seq, 48000, 0,
                                                   b'\x01\x00' * 960)))
            self.seq += 1
        if not self.queue:
            self.playing = False
            self.waited = 0
            return
        if not self.playing:
            self.waited += 960
            if max(self.queued, self.waited) < 1920:
                return
            self.playing = True
        left = 960
        while left and self.queue:
            item, count = self.queue[0]
            count = min(left, count)
            self.queue[0][1] -= count
            self.queued -= count
            left -= count
            self.since_report += count
            self.played[item] = self.played.get(item, 0) + count
            self.dirty.add(item)
            if not self.queue[0][1]:
                self.queue.popleft()
        if not self.queue or self.since_report >= 4800:
            self.acknowledge()

    def video(self, data):
        self.send(cp.encode_frame(cp.VideoFrame(self.epoch, self.video_seq, 'image/jpeg', data)))
        self.video_seq += 1

    def share(self, sharing, kind='screen'):
        self.send({'type': 'share', 'sharing': sharing, 'kind': kind})

    def end(self):
        self.send({'type': 'bye', 'reason': 'owner_end'})
