"""Local native-wire peer for the real call adapters. No provider or device access.

Uses the shared VoiceWire setup/item acknowledgements and Gemini audio shape;
only the remote provider is scripted. JSON crosses a real loopback websocket.
"""
import asyncio
import base64
import copy
import json
from collections import deque

from tests.test_live_conversation import VoiceWire
from tests.test_gemini_live_boundaries import audio_content


class NativeWire(VoiceWire):
    def __init__(self, engine, number, script):
        super().__init__(engine, None)
        self.number, self.script = number, script
        self.audio = bytearray()
        self.responses = 0
        self.active = ''
        self.spoken = []
        self.outputs = {}
        self.prompt = 2000
        self.audio_tokens = 100
        self.websocket = None
        # Ordinary Gemini (recorded 28.10) sends no interactionStatus and no caption
        # `finished`; extended thinking reports native IDLE. ProviderServer sets the route.
        self.thinking = False

    def start_tool(self):
        pass  # The scenario scripts native VAD, not an in-memory driver event.

    def answer(self):
        action = self.script.popleft() if self.script else ('say', f'Context answered {self.number}-{self.responses + 1}.')
        if action[0] == 'tool':
            self.tool(*action[1:])
        else:
            self.say(action[1], complete=action[0] != 'partial')

    def usage(self):
        if self.engine == 'gemini':
            return {'promptTokenCount': self.prompt, 'responseTokenCount': 12,
                'promptTokensDetails': [{'modality': 'TEXT', 'tokenCount': self.prompt - self.audio_tokens},
                                       {'modality': 'AUDIO', 'tokenCount': self.audio_tokens}],
                'responseTokensDetails': [{'modality': 'AUDIO', 'tokenCount': 12}]}
        return {'input_tokens': self.prompt, 'output_tokens': 12,
                'input_token_details': {'text_tokens': self.prompt - self.audio_tokens, 'audio_tokens': self.audio_tokens,
                                       'image_tokens': 0, 'cached_tokens': 0},
                'output_token_details': {'text_tokens': 0, 'audio_tokens': 12}}

    def begin(self):
        self.responses += 1
        self.active = f'response-{self.number}-{self.responses}'
        if self.engine != 'gemini':
            self.put({'type': 'response.created', 'response': {'id': self.active}})
        return f'item-{self.number}-{self.responses}'

    def complete(self, *, output=(), status='completed'):
        if self.engine == 'gemini':
            self.put({'serverContent': {'turnComplete': True, **({'interactionStatus': 'IDLE'} if self.thinking else {})},
                      'usageMetadata': self.usage()})
        else:
            self.put({'type': 'response.done', 'response': {'id': self.active, 'status': status,
                'usage': self.usage(), 'output': list(output)}, 'usage': {'input_audio_duration': len(self.audio) / 48000}})
        self.active = ''

    def say(self, text, *, complete=True):
        item = self.begin()
        self.spoken.append(text)
        if self.engine == 'gemini':
            content = audio_content(text, finished=complete)
            if not self.thinking:
                del content['serverContent']['outputTranscription']['finished']
            self.put(content)
        else:
            self.put({'type': 'response.output_audio.delta', 'response_id': self.active, 'item_id': item,
                      'delta': base64.b64encode(b'\x01\x00' * (480 if complete else 48000)).decode()})
            self.put({'type': 'response.output_audio_transcript.done' if complete else 'response.output_audio_transcript.delta',
                      'item_id': item, **({'transcript': text} if complete else {'delta': text})})
        if complete:
            self.complete()

    def tool(self, cid, name, arguments):
        self.begin()
        if self.engine == 'gemini':
            self.put({'toolCall': {'functionCalls': [{'id': cid, 'name': name, 'args': arguments}]}})
            self.complete()
        else:
            self.complete(output=[{'type': 'function_call', 'call_id': cid, 'name': name,
                                   'arguments': json.dumps(arguments)}])

    def input_start(self, item):
        if self.engine == 'gemini':
            self.put({'voiceActivity': {'type': 'ACTIVITY_START'}})
            if self.active:
                self.put({'serverContent': {'interrupted': True}})
                self.complete(status='cancelled')
        else:
            self.put({'type': 'input_audio_buffer.speech_started', 'item_id': item})
            if self.active:
                self.put({'type': 'response.output_audio_transcript.done',
                          'item_id': f'item-{self.number}-{self.responses}', 'transcript': self.spoken[-1]})
                self.complete(status='cancelled')  # native interrupt_response cancels before host notification

    def input_end(self, item, text):
        if self.engine == 'gemini':
            self.put({'voiceActivity': {'type': 'ACTIVITY_END'}})
            self.put({'serverContent': {'inputTranscription': {'text': text, **({'finished': True} if self.thinking else {})}}})
            self.answer()
        else:
            self.put({'type': 'input_audio_buffer.speech_stopped', 'item_id': item})
            self.put({'type': 'input_audio_buffer.committed', 'item_id': item})
            self.put({'type': 'conversation.item.input_audio_transcription.completed', 'item_id': item,
                'transcript': text, 'status': 'completed', 'usage': {'type': 'tokens', 'input_tokens': 10, 'output_tokens': 3,
                    'input_token_details': {'text_tokens': 2, 'audio_tokens': 8}}})
            if self.engine == 'grok':
                self.answer()  # xAI VAD owns this response, not an extra host response.create

    async def send(self, raw):
        event = json.loads(raw)
        audio = event.get('audio') if event.get('type') == 'input_audio_buffer.append' else event.get('realtimeInput', {}).get('audio', {}).get('data')
        if audio:
            self.audio.extend(base64.b64decode(audio))
        kind = event.get('type')
        if kind == 'conversation.item.create' and event['item'].get('type') == 'function_call_output':
            self.outputs[event['item']['call_id']] = event['item']['output']
        if kind == 'conversation.item.create' and any(p.get('type') == 'input_image' for p in event['item'].get('content', [])):
            self.sent.append(event)
            echoed = copy.deepcopy(event['item'])
            echoed['content'] = [{'type': 'input_image'} if p['type'] == 'input_image' else p for p in echoed['content']]
            for phase in ('added', 'done'):
                self.put({'type': 'conversation.item.' + phase, 'item': echoed, 'previous_item_id': event.get('previous_item_id')})
        elif kind == 'conversation.item.delete':
            self.sent.append(event)
            self.put({'type': 'conversation.item.deleted', 'item_id': event['item_id']})
        elif kind == 'response.cancel':
            self.sent.append(event)
            if self.active:
                self.complete(status='cancelled')
            else:
                self.put({'type': 'error', 'error': {'code': 'response_cancel_not_active'}})
        elif 'toolResponse' in event:
            self.sent.append(event)
            rows = event['toolResponse']['functionResponses']
            self.outputs.update({r['id']: r['response'] for r in rows})
            if any(r['name'] == 'set_call_languages' for r in rows):
                # Real provider latency exposes a quiet gap after the result was
                # written but before its ordinary answer starts arriving.
                await asyncio.sleep(.1)
            if all(r.get('scheduling') == 'SILENT' or (r['name'] == 'attend'
                   and 'frames' not in json.loads(r['response']['output'])) for r in rows):
                self.put({'serverContent': {'turnComplete': True, **({'interactionStatus': 'IDLE'} if self.thinking else {})}})
            else:
                self.answer()
        elif event.get('clientContent', {}).get('turnComplete'):
            self.sent.append(event)
            self.answer()
        else:
            await super().send(raw)


class ProviderServer:
    def __init__(self, engine):
        self.engine, self.connections, self.script = engine, [], deque()
        self.errors = []
        self.thinking = False

    async def accept(self, websocket):
        from websockets.exceptions import ConnectionClosed
        wire = NativeWire(self.engine, len(self.connections) + 1, self.script)
        wire.websocket, wire.thinking = websocket, self.thinking
        self.connections.append(wire)
        async def outbound():
            while True:
                event = await wire.events.get()
                wire.received.append(event)
                await websocket.send(json.dumps(event))
        writer = asyncio.create_task(outbound())
        try:
            async for raw in websocket:
                await wire.send(raw)
        except ConnectionClosed:
            pass
        except Exception as exc:
            self.errors.append(repr(exc))
            raise
        finally:
            writer.cancel()
            await asyncio.gather(writer, return_exceptions=True)
            wire.closed = True
