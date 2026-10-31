"""Optional private incident replay; normal regression coverage uses synthetic sources.

The evidence location is explicitly supplied by the operator; this test never
opens the install or a stand. No owner words or payloads are printed or copied.
"""
import json
import asyncio
import os
from dataclasses import replace
from pathlib import Path

import pytest

from tests.test_live_conversation import VoiceWire, host_world as host_world, packages as packages


@pytest.mark.parametrize('owner_mode,minimum,maximum,pixels', [('max', 74000, 77000, 5), ('nano', 71000, 72000, 2)])
def test_04dbca78_opens_after_wire_projection_without_a_recovery_pass(host_world, packages, monkeypatch, record_property,
                                                                    owner_mode, minimum, maximum, pixels):
    from ouroboros.contracts.live_conversation import LiveOptions
    from ouroboros.realtime_setup import loaded_provider, session_candidate
    folder = os.environ.get('OURO_CONTEXT_EVIDENCE_DIR')
    if not folder:
        pytest.skip('private incident replay requires the operator-supplied evidence copy')
    payload = json.loads((Path(folder)/'04dbca78-refused-payload.json').read_text())
    manifest = json.loads((Path(folder)/'04dbca78-refused-manifest.json').read_text())
    setup = payload['events'][0]['setup']
    parts = setup['systemInstruction']['parts']
    archive = json.loads(parts[-1]['text'][parts[-1]['text'].index('\n[')+1:])
    images = [p for e in payload['events'][1:] for t in e.get('clientContent', {}).get('turns', [])
              for p in t.get('parts', []) if 'inlineData' in p]
    index = 0
    for row in archive:
        if not isinstance(row.get('content'), list):
            continue
        for n, part in enumerate(row['content']):
            if part.get('text', '').startswith('[Attached image '):
                data = images[index]['inlineData']; index += 1
                row['content'][n] = {'type':'image_url', 'image_url':{'url':
                    'data:' + data['mimeType'] + ';base64,' + data['data']}}
    host = host_world.host
    provider = loaded_provider('gemini_live:gemini')
    host.model = provider['catalog'].models[0]
    host.turn.native_model = payload['model']
    core = [{'type':'text', 'text': part['text']} for part in parts[:-2]]
    projections = {name: replace(getattr(host.plan, name), system_content_json=json.dumps(core))
                   for name in ('max_projection', 'low_projection', 'nano_projection')}
    host.plan = replace(host.plan, preferred_mode=owner_mode,
                        output_reserve_tokens=manifest['fit']['measurement']['reply_allowance_tokens'], **projections)
    host.messages[:] = [host.plan.projection('nano').system_message(), *archive]
    host.schemas[:] = [{'type':'function', 'function': {**{k:v for k,v in tool.items()
                       if k not in {'parametersJsonSchema', 'behavior'}}, 'parameters': tool['parametersJsonSchema']}} for group in setup['tools']
                       for tool in group.get('functionDeclarations', [])]
    host._session_opened = True
    density = manifest['fit']['measurement']['measurement_density']
    monkeypatch.setattr('ouroboros.realtime_setup._session_density', lambda *a: (density, 'fresh_route_usage'))
    wire = VoiceWire('gemini', None)
    def factory(selected, options, **kwargs):
        return provider['session'](selected, options, connect=wire.connect, **kwargs)
    session, request, evidence = session_candidate(host, factory, LiveOptions(instructions=parts[-2]['text']))
    assert minimum < request.prompt_tokens_estimate < maximum, evidence
    assert request.prompt_tokens_estimate + request.working_reserve_tokens <= 92000
    assert not evidence['recovery'], evidence
    assert '_tool_result_record' not in json.dumps(session.wire)
    from ouroboros.context_images import image_positions
    assert len(image_positions(host.messages)) == len(images) == 5  # the call's exact canonical pixels stay
    assert sum('inlineData' in part for event in session.wire['events']
               for turn in event.get('clientContent', {}).get('turns', []) for part in turn.get('parts', [])) == pixels
    async def open_and_close():
        try:
            await session.open()
            assert any('setup' in event for event in wire.sent)
        finally:
            await session.close()
    asyncio.run(open_and_close())
    record_property('before_input', 88184)
    record_property('after_input', request.prompt_tokens_estimate)
    record_property('margin', 92000-request.prompt_tokens_estimate-request.working_reserve_tokens)
    record_property('wire_pixels', pixels)
