"""The diagnostic and real session seam measure the same retained owner-shaped input."""
from dataclasses import asdict, replace
import json
from types import SimpleNamespace

import pytest

from tests import _memory_scale_fixture as scale
from devtools._context_fit_accounting import cold_density, declared_evidence
from devtools._context_fit_session import measure_session, session_inputs
from devtools.context_fit_report import REPO, measure_core


@pytest.mark.parametrize('window, extra_chars, fits, producers',
    [(92000, 0, True, False), (92000, 24000, True, False), (92000, 100000, False, False)]
    + [(window, extra, fits, True) for window in (85000, 92000, 133554)
       for extra, fits in ((0, True), (400000, False))])
def test_retained_scale_candidate_matches_live_admission(tmp_path, monkeypatch, window, extra_chars, fits, producers):
    import socket
    from ouroboros import context_fit as cf, memory_view as mv
    from ouroboros.call_protocol import CallSpec
    from ouroboros.call_peers import LiveCallHub
    from ouroboros.contracts.live_conversation import LiveConfig
    from ouroboros.contracts.realtime import RealtimeSetupError
    from ouroboros.live_conversation import ConversationHost
    from ouroboros.realtime_setup import session_candidate
    from ouroboros.section_floor import HostSection
    from ouroboros.tools.registry import ToolRegistry
    from ouroboros.tools.tool_context import ToolContext
    from skills.live_conversation.driver import call_tools, session_options

    monkeypatch.setattr(socket.socket, 'connect', lambda *a, **kw: pytest.fail('provider connection'))
    from tests import _nano_context_fixture as nano
    core = nano.core(tmp_path / 'captures', monkeypatch) if producers else scale.core(tmp_path / 'captures')
    schemas = nano.tools() if producers else scale.tools()
    core = replace(core, dynamic_sections=(*core.dynamic_sections,
                    HostSection('additional_held_context', scale.text('Other held facts', extra_chars))))
    transport, model, route, output, factory = session_inputs('gemini_live', window, 10000, tmp_path)
    # A resident view prepared under a wider window is not a fresh start. Renewal
    # retains its forms and pays physical P + R, not a second start work room.
    initial_window = 512000 if producers else 256000
    evidence = declared_evidence(initial_window, 10000,
        asdict(replace(model.facts, conversation_window_tokens=initial_window)), output)
    task = dict(id='parity', chat_id=1, type='task', _is_direct_chat=True)
    with cold_density():
        plan = cf.build_context_fit_plan(SimpleNamespace(drive_root=tmp_path), core, task,
            preferred_mode='max', tool_schemas=schemas, route_resolver=lambda *a, **kw: (route, evidence))
        fresh = measure_core(core, schemas, root=tmp_path, window=window, reserve=10000,
                             route=route, route_facts=asdict(model.facts), response_limit=output)

        def make_host():
            config = LiveConfig(enabled=True, provider='fixture:gemini')
            spec = CallSpec('parity', 1, config.provider, model.id, '')
            turn = SimpleNamespace(config=config, catalog_model=model, route=transport, model=route['model'],
                native_model=model.id, spec=spec, opening='', observation_fingerprint=lambda: '',
                feature={'call_tools': call_tools(tmp_path / 'feature')})
            ctx = ToolContext(repo_dir=REPO, drive_root=tmp_path, task_id='parity',
                              current_chat_id=1, is_direct_chat=True)
            ctx.context_fit_plan = plan
            tools = ToolRegistry(REPO, tmp_path)
            monkeypatch.setattr(tools, 'schemas', lambda: schemas)
            tools.set_context(ctx)
            host = ConversationHost(turn, ctx, plan.messages_for(plan.initial_mode), tools, task, '', LiveCallHub(spec))
            host._session_opened = True  # this parity case explicitly models a renewal
            return host

        server, diagnostic = make_host(), make_host()
        try:
            if fits:
                _, _, expected = session_candidate(server, factory, session_options(server))
            else:
                with pytest.raises(RealtimeSetupError, match='context_capacity'):
                    session_candidate(server, factory, session_options(server))
                expected = server.trace['fit_refusal']
            result = measure_session(diagnostic, factory, session_options(diagnostic))
        finally:
            server.hub.close('test_done')
            diagnostic.hub.close('test_done')
    measured = expected['fit']['measurement']
    assert result['verdict'] == ('fits' if fits else 'context_capacity'), result
    assert result['input_tokens'] == measured['estimated_input_tokens']  # same estimator, tolerance zero
    assert result['reserve'] == measured['response_reserve_tokens']
    assert result['profile'] == measured['profile']
    if not producers:
        assert result['profile'] == 'task_local_nano'
    assert not result['work_room_applies_to_verdict']
    assert not fresh['work_room_applies_to_verdict']  # a soft start target, never a capacity veto
    for diagnostic in (fresh, result):
        arithmetic = diagnostic['arithmetic']
        assert arithmetic['requested_window_tokens'] == window
        assert arithmetic['room_after_reserve_tokens'] == diagnostic['window'] - diagnostic['input_tokens'] - diagnostic['reserve']
        assert arithmetic['room_above_start_tokens'] == diagnostic['margin_tokens'] - diagnostic['start_work_room_tokens']
        story = next(b for b in diagnostic['blocks'] if b['block'] == 'story')
        assert story['forms']['accounts whole'] == 1
    assert result['work_room_levels'] == mv.snapshot_from_json(core.memory_view_json).spec.work_room_levels
    if fits:
        from ouroboros.context_window import resolve_window
        assert result['window_record'] == asdict(resolve_window(server.plan, measured['response_reserve_tokens'],
                                                               root=tmp_path, bound=False))
        assert not result['window_record']['total']['acceptance_can_raise']
    if extra_chars == 24000:
        assert fresh['verdict'] == 'fits' and fresh['work_room_deficit_tokens'] > 0, fresh
        assert fresh['deficit_tokens'] == 0 and result['margin_tokens'] >= 0
        assert fresh['phase'] == 'task_preparation' and result['phase'] == 'session_candidate'
    print('PARITY ' + json.dumps(dict(extra_chars=extra_chars, fresh=fresh, session=result), sort_keys=True))


@pytest.mark.parametrize('provider', ['gemini_live', 'grok_realtime', 'openai_realtime'])
def test_bundled_session_diagnostic_is_offline_and_discloses_its_inputs(tmp_path, monkeypatch, provider):
    import socket
    from devtools._context_fit_session import report_session
    from ouroboros.tools.registry import ToolRegistry

    monkeypatch.setattr(socket.socket, 'connect', lambda *a, **kw: pytest.fail('provider connection'))
    result = report_session(scale.core(tmp_path / 'captures'), ToolRegistry(REPO, tmp_path), root=tmp_path,
                            scratch=tmp_path / 'scratch', window=1000000, reserve=10000, mode='max', provider=provider)
    assert result['verdict'] == 'fits', result
    assert result['phase'] == 'session_candidate'
    assert result['profile'] == 'owner_max'
    assert result['session_provider'] == provider and result['diagnostic_notes']
    assert result['window_record']['total']['acceptance_can_raise'] == (provider == 'openai_realtime')
    assert result['margin_tokens'] == result['window'] - result['reserve'] - result['input_tokens']


def test_snapshot_source_is_guarded_and_scratch_cannot_point_into_it(tmp_path):
    from devtools._context_fit_readonly import readonly_inputs

    source, copied = tmp_path / 'source', tmp_path / 'copy'
    source.mkdir()
    copied.mkdir()
    with readonly_inputs(copied, source_root=source), pytest.raises(PermissionError):
        (source / 'accidental-write').write_text('forbidden')
    with pytest.raises(ValueError, match='outside the measured root'):
        with readonly_inputs(copied, source_root=source, scratch_root=source / 'scratch'):
            pass
    # A nested reader must not remove its parent's source guard on exit.
    with readonly_inputs(source):
        with readonly_inputs(source):
            pass
        with pytest.raises(PermissionError):
            (source / 'still-forbidden').write_text('forbidden')
