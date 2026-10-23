"""The diagnostic and real session seam measure the same retained owner-shaped input."""
from dataclasses import asdict, replace
import json
from types import SimpleNamespace

import pytest

from tests import _memory_scale_fixture as scale
from devtools._context_fit_accounting import cold_density, declared_evidence
from devtools._context_fit_session import measure_session, session_inputs
from devtools.context_fit_report import REPO, measure_core


@pytest.mark.parametrize('extra_chars, fits', [(0, True), (24000, True), (100000, False)])
def test_retained_scale_candidate_matches_live_admission(tmp_path, monkeypatch, extra_chars, fits):
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
    core = scale.core(tmp_path / 'captures')
    core = replace(core, dynamic_sections=(*core.dynamic_sections,
                    HostSection('additional_held_context', scale.text('Other held facts', extra_chars))))
    transport, model, route, output, factory = session_inputs('gemini_live', 92000, 10000, tmp_path)
    # A resident view prepared under a wider window is not a fresh start. Renewal
    # retains its forms and pays physical P + R, not a second start work room.
    evidence = declared_evidence(256000, 10000, asdict(replace(model.facts, conversation_window_tokens=256000)), output)
    task = dict(id='parity', chat_id=1, type='task', _is_direct_chat=True)
    with cold_density():
        plan = cf.build_context_fit_plan(SimpleNamespace(drive_root=tmp_path), core, task,
            preferred_mode='max', tool_schemas=scale.tools(), route_resolver=lambda *a, **kw: (route, evidence))
        fresh = measure_core(core, scale.tools(), root=tmp_path, window=92000, reserve=10000,
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
            monkeypatch.setattr(tools, 'schemas', scale.tools)
            tools.set_context(ctx)
            return ConversationHost(turn, ctx, plan.messages_for(plan.initial_mode), tools, task, '', LiveCallHub(spec))

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
    assert result['profile'] == measured['profile'] == 'task_local_nano'
    assert not result['work_room_applies_to_verdict']
    assert fresh['work_room_applies_to_verdict']
    assert result['work_room_levels'] == mv.snapshot_from_json(core.memory_view_json).spec.work_room_levels
    if fits:
        from ouroboros.context_window import resolve_window
        assert result['window_record'] == asdict(resolve_window(server.plan, measured['response_reserve_tokens'],
                                                               root=tmp_path, bound=False))
        assert not result['window_record']['total']['acceptance_can_raise']
    if extra_chars == 24000:
        assert fresh['verdict'] == 'MinimumViewUnfit' and fresh['work_room_deficit_tokens'] > 0, fresh
        assert fresh['deficit_tokens'] == 0 and result['margin_tokens'] >= 0
        assert fresh['phase'] == 'task_preparation' and result['phase'] == 'session_candidate'
    print('PARITY ' + json.dumps(dict(extra_chars=extra_chars, fresh=fresh, session=result), sort_keys=True))


def test_nano_composition_names_the_reader_not_the_unheld_pattern_body(tmp_path):
    from ouroboros.section_floor import HostSection

    core = scale.core(tmp_path)
    core = replace(core, dynamic_head_sections=(*core.dynamic_head_sections,
                   HostSection('pattern_register', 'Patterns ' * 8000)))
    result = measure_core(core, scale.tools(), root=tmp_path, window=92000, reserve=10000)
    row, = [row for row in result['blocks'] if row['block'] == 'pattern_register']
    assert result['mode'] == 'nano' and row['form'] == 'reader' and row['tokens'] < 100
    assert all(row['tokens'] >= 0 for row in result['blocks']), result
    assert sum(row['tokens'] for row in result['blocks']) == result['input_tokens']


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
