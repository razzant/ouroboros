"""Optional offline call envelope: run the real host seam, never open a session."""
from dataclasses import asdict, replace
import importlib
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

from devtools._context_fit_accounting import cold_density, declared_evidence


def measure_session(host, factory, options, *, observation_fp=None):
    """Measure the caller's actual retained plan, schemas and transcript (including renewals)."""
    from devtools.context_fit_report import measure_core
    from ouroboros.contracts.realtime import RealtimeSetupError
    from ouroboros.realtime_setup import session_candidate
    from ouroboros.context_fit import ContextFitPlan
    from unittest.mock import patch

    candidates, refused = [], False
    reproject = ContextFitPlan.reproject_for_route
    def capture(plan, *args, **kwargs):
        result = reproject(plan, *args, **kwargs)
        candidates.append(result)
        return result
    # Refused candidates are deliberately not published onto the live host. Capture
    # the real ladder's return for diagnostics, without inventing a second projection.
    with patch.object(ContextFitPlan, 'reproject_for_route', capture):
        try:
            _, _, evidence = session_candidate(host, factory, options, observation_fp=observation_fp)
        except RealtimeSetupError as exc:
            if exc.code != 'context_capacity' or 'fit_refusal' not in host.trace:
                raise
            evidence, refused = host.trace['fit_refusal'], True
    candidate = candidates[-1] if refused and candidates else host.plan
    measurement = evidence['fit']['measurement']
    result = measure_core(host.plan.core, host.schemas, root=host.ctx.drive_root,
        window=measurement['capacity_total_tokens'], reserve=measurement['response_reserve_tokens'],
        mode=host.plan.preferred_mode, prepared=candidate, fit=evidence['fit'])
    result.update(phase='session_candidate', diagnostic_notes=[])
    if refused:
        result.update(verdict='context_capacity', ladder_basis='refused candidate; not published to the host',
                      work_room_deficit_tokens=0, work_room_applies_to_verdict=False)
    else:
        work = evidence['work_room']
        result.update(work_room=work, start_work_room_tokens=work['requested_tokens'],
                      work_room_deficit_tokens=work['deficit_tokens'])
        # The arithmetic line tells the same candidate's room as the session evidence (R15-5).
        arithmetic = result['arithmetic']
        arithmetic.update(start_work_room_tokens=work['requested_tokens'],
                          start_room_shortfall_tokens=work['deficit_tokens'])
        if work.get('remaining_tokens') is not None:
            arithmetic['room_above_start_tokens'] = work['remaining_tokens'] - work['requested_tokens']
    return result


def session_inputs(provider, window, reserve, root):
    """A bundled catalog and cached output limit; never infer a provider from a window number."""
    if provider not in {'gemini_live', 'grok_realtime', 'openai_realtime'}:
        raise ValueError('Unsupported offline session provider')
    # Skill entrypoints are packages: OpenAI's live.py imports its plugin facade.
    # Give this offline load its own namespace; do not register or activate a skill.
    name = '_context_fit_provider_' + provider
    module = sys.modules.get(name)
    if module is None:
        directory = Path(__file__).resolve().parents[1] / 'skills' / provider
        spec = importlib.util.spec_from_file_location(name, directory / 'plugin.py',
                                                     submodule_search_locations=[str(directory)])
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    model = module.CATALOG.model('')
    facts = replace(model.facts, conversation_window_tokens=window, working_reserve_tokens=reserve)
    if (window, reserve) != (model.facts.conversation_window_tokens, model.facts.working_reserve_tokens):
        facts = replace(facts, source='diagnostic window/reserve override; ' + facts.source)
    model = replace(model, facts=facts)
    route = dict(provider=module.ROUTE.accounting_provider, model=module.ROUTE.model_prefix + model.id,
                 base_url=module.ROUTE.endpoint, use_local=False, model_role='main', options={})
    from ouroboros.response_limits import resolve_response_limit

    limit = resolve_response_limit(root, provider=route['provider'], model=route['model'],
                                   base_url=route['base_url'], allow_fetch=False)
    if provider == 'openai_realtime':
        session = importlib.import_module(module.__name__ + '.live').OpenAISession
    else:
        session = module.GeminiSession if provider == 'gemini_live' else module.GrokSession

    def no_connect(*args, **kwargs):
        raise RuntimeError('The offline diagnostic cannot connect to a provider')

    def factory(selected, options, *, observe):
        return session(selected, options, connect=no_connect, observe=observe)

    return module.ROUTE, model, route, asdict(limit), factory


def report_session(core, tools, *, root, scratch, window, reserve, mode, provider):
    from devtools.context_fit_report import measure_core, REPO
    from ouroboros.call_peers import LiveCallHub
    from ouroboros.call_protocol import CallSpec
    from ouroboros.context_fit import build_context_fit_plan, MinimumViewUnfit
    from ouroboros.contracts.live_conversation import LiveConfig
    from ouroboros.live_conversation import ConversationHost
    from ouroboros.memory_view import snapshot_from_json
    from ouroboros.tools.tool_context import ToolContext
    from skills.live_conversation.driver import INITIAL_TOOLS, call_tools, session_options

    transport, model, route, output, factory = session_inputs(provider, window, reserve, root)
    evidence = declared_evidence(window, reserve, asdict(model.facts), output)
    task = dict(id=core.section_task_id, chat_id=int(snapshot_from_json(core.memory_view_json).spec.room_id),
                _is_direct_chat=True, type='task')
    with cold_density():
        try:
            plan = build_context_fit_plan(SimpleNamespace(drive_root=root), core, task, preferred_mode=mode,
                tool_schemas=tools.schemas(), route_resolver=lambda *a, **kw: (route, evidence))
        except MinimumViewUnfit:
            result = measure_core(core, tools.schemas(), root=root, window=window, reserve=reserve, mode=mode,
                                  route=route, route_facts=asdict(model.facts), response_limit=output)
            result['diagnostic_notes'].append('Task preparation refused; the session seam was not reached.')
            return result
        config = LiveConfig(enabled=True, provider=provider + ':offline')
        spec = CallSpec('context-fit-report', task['chat_id'], config.provider, model.id, '')
        turn = SimpleNamespace(config=config, catalog_model=model, route=transport, model=route['model'],
            native_model=model.id, spec=spec, opening='', feature={
                'call_tools': call_tools(scratch / 'feature'), 'initial_tools': INITIAL_TOOLS})
        ctx = ToolContext(repo_dir=REPO, drive_root=scratch / 'host', task_id=task['id'],
                          current_chat_id=task['chat_id'], is_direct_chat=True)
        ctx.context_fit_plan = plan
        tools.set_context(ctx)
        hub = LiveCallHub(spec)
        host = ConversationHost(turn, ctx, plan.messages_for(plan.initial_mode), tools, task, '', hub)
        try:
            result = measure_session(host, factory, session_options(host), observation_fp='')
        finally:
            hub.close('diagnostic_done')
    result['diagnostic_notes'] = [
        'Bundled default model and call options; saved feature settings/languages, opening turn and live mailbox are not supplied.',
        'Real host tool selection, call tools, driver instructions and provider wire are included; no session was opened.',
        'Density is fixed at 1.0; credentials and account-scoped acceptance observations are not inferred.']
    result['session_provider'] = provider
    return result
