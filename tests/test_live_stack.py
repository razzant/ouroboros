"""Owner-visible and retained effects through the real call stack, all native wires."""
import asyncio
import json
from pathlib import Path

import pytest

from tests._extension_loader_shared import _clear_loader_state  # noqa: F401
from tests._live_stack import stack
from tests._live_stack_page import jpeg as synthetic_frame
from tests.test_live_conversation import host_world as host_world


@pytest.mark.parametrize('engine,thinking', [('openai', False), ('grok', False), ('gemini', False), ('gemini', True)])
def test_full_stack_greeting_owner_answer_and_retained_receipts(host_world, monkeypatch, engine, thinking):
    async def scenario():
        async with stack(host_world, monkeypatch, engine, thinking=thinking) as run:
            assert run.page.played, 'greeting must be heard, not only queued or captioned'
            run.provider.script.append(('say', 'Synthetic owner answered.'))
            await run.owner_says('Synthetic owner first words.')
            await run.wait(lambda: any(v.get('text') == 'Synthetic owner answered.' and v.get('final') for v in run.page.controls))
            await run.quiet()
        assert run.result[1]['execution_status'] == 'ok', run.result
        rows = run.transcript()
        assert any(r.get('text') == 'Synthetic owner first words.' for r in rows)
        assert any(r.get('text') == 'Synthetic owner answered.' for r in rows)
        from ouroboros.usage_accounting import read_usage_records
        usage = read_usage_records(run.root)
        assert usage, 'native usage must reach the shared ledger'
        if engine == 'openai':
            assert any(r.get('source') == 'live_transcription_usage' for r in usage), 'input ASR is separately accounted'
        journal = run.journal()
        assert {'send', 'receive', 'host', 'response'} <= {r['direction'] for r in journal}
        times = [r['at'] for r in journal]
        assert times and all(type(t) is float and t > 0 for t in times), 'each record says when it was observed'
        assert not run.provider.errors and not any(r.get('code', 1000) != 1000 for r in run.page.closed)
    asyncio.run(scenario())


@pytest.mark.parametrize('engine,thinking', [('openai', False), ('grok', False), ('gemini', False), ('gemini', True)])
def test_full_stack_languages_barge_in_mail_sharing_attention_and_real_tool(host_world, monkeypatch, engine, thinking):
    async def scenario():
        async with stack(host_world, monkeypatch, engine, thinking=thinking, languages=True) as run:
            assert run.host.state['languages'] == ('en-US',)
            assert any('opening-language' in wire.outputs for wire in run.provider.connections) or engine == 'gemini'
            # Native partial answer, then owner VAD and the page's flush/played acknowledgements.
            run.wire.say('Synthetic interrupted words.', complete=False)
            await run.wait(lambda: bool(run.call.active_responses))
            run.provider.script.append(('say', 'Synthetic barge-in answered.'))
            await run.owner_says('Synthetic first words after interruption.')
            await run.quiet()
            assert any(v['type'] == 'flush' for v in run.page.controls)
            await run.wait(lambda: any('Synthetic first words after interruption.' in str(r) for r in run.transcript()))

            run.provider.script.append(('say', 'Synthetic typed message answered.'))
            run.mail.put({'text': 'Synthetic typed room message.', 'msg_id': 'typed-mid-call'})
            await run.wait(lambda: any(v.get('text') == 'Synthetic typed message answered.' and v.get('final')
                                      for v in run.page.controls))
            await run.quiet()

            for kind, pixels in [('screen', b'synthetic-screen'), ('camera', b'synthetic-camera')]:
                jpeg = synthetic_frame(pixels.decode())
                run.page.video(jpeg)
                run.page.share(True, kind)
                await run.wait(lambda: any(f'started showing the {kind}' in str(r) for r in run.host.messages))
                await run.wait(lambda: any(f.data == jpeg for f in run.call.attention.frames.values()))
                await run.wait(lambda: not run.call.pending_offers and run.call.quiet())
                cid = 'look-' + kind
                run.wire.tool(cid, 'attend', {'see': 'asked', 'look': True, 'revisit_after_s': None})
                await run.wait(lambda: cid in run.wire.outputs)
                await run.quiet()
                result = run.wire.outputs[cid]
                result = json.loads(result['output'] if isinstance(result, dict) else result)
                assert result['frames'] and Path(result['frames'][0]['file_path']).read_bytes() == jpeg
                if engine == 'grok':
                    assert 'input_image' not in str(run.wire.sent) and 'vlm_query' in str(result)
                else:
                    assert any(('video' in v.get('realtimeInput', {}) or any(p.get('type') == 'input_image'
                        for p in v.get('item', {}).get('content', []))) for v in run.wire.sent)
                before = run.wire.responses
                run.wire.tool('silent-' + kind, 'attend', {'see': None, 'look': None, 'revisit_after_s': None})
                await run.wait(lambda: 'silent-' + kind in run.wire.outputs)
                await run.quiet()
                # Thinking has no provider SILENT facility; this fake explicitly chooses no words too.
                assert run.wire.responses == before + 1, 'silent attend cannot manufacture a second response'
                run.page.share(False, kind)
                await run.wait(lambda: any(f'stopped showing the {kind}' in str(r) for r in run.host.messages))
                await run.quiet()

            run.wire.tool('once', 'attend', {'see': None, 'look': None, 'revisit_after_s': .05})
            await run.wait(lambda: any('look you asked' in str(r) for r in run.host.messages))
            await run.quiet()
            assert sum('look you asked' in str(r) for r in run.host.messages) == 1
            assert run.call.attention.revisit_at is None
            before = len(run.provider.connections)
            run.wire.tool('load-reader', 'enable_tools', {'tools': 'read_file'})
            await run.wait(lambda: 'read_file' in {t['function']['name'] for t in run.host.schemas})
            await run.wait(lambda: run.call.connected.is_set() and not run.call.tools_running and not run.call.tool_refit.is_set())
            await run.quiet()
            if engine == 'gemini': assert len(run.provider.connections) > before
            run.wire.tool('real-read', 'read_file', {'path': str(run.state.env.repo_dir / 'BIBLE.md')})
            await run.wait(lambda: 'real-read' in run.wire.outputs)
            await run.quiet()
            assert 'real-read' in run.host.executed
            label = (run.state.env.repo_dir / 'BIBLE.md').read_text().splitlines()[0]
            assert label in str(run.wire.outputs['real-read']), 'the real tool must succeed, not merely return an error'
            run.page.send({'type': 'mute', 'muted': True})
            run.page.send({'type': 'mute', 'muted': False})
        assert run.result[1]['execution_status'] == 'ok', run.result
        facts = [json.loads(r['raw']) for r in run.journal() if r['direction'] == 'host']
        assert len([r for r in facts if r.get('type') == 'call.offer_wait']) == 1
        assert any(r.get('type') == 'call.input_audio' for r in facts)
        assert any('Synthetic barge-in answered.' in str(r) for r in run.transcript())
        from ouroboros.working_checkpoint import checkpoint_path
        saved = json.loads(checkpoint_path(run.root, run.state.ctx.task_id, 1).read_text())
        assert any('started showing the screen' in str(r) for r in saved['messages'])
        assert any('Synthetic typed room message.' in str(r) for r in saved['messages'])
        assert any(r.get('tool_call_id') == 'real-read' for r in saved['messages'])
    asyncio.run(scenario())
