"""Offline inputs to the ordinary window and physical-admission owners."""
from contextlib import contextmanager
from dataclasses import asdict
from types import SimpleNamespace
from unittest.mock import patch


@contextmanager
def cold_density():
    from ouroboros import capability_evidence, context_fit

    with patch.object(context_fit, '_route_calibration_ratio', lambda *a, **kw: 1.0), patch.object(
            capability_evidence, 'resolve_main_token_density', lambda *a, **kw: (1.0, 'cold_estimate')):
        yield


def declared_evidence(window, reserve, facts=None, response_limit=None):
    from ouroboros.context_window import WindowLimit, WindowRecord

    facts = facts or {'working_reserve_tokens': reserve, 'basis': 'provider', 'acceptance_can_raise': False,
                      'source': 'diagnostic declaration'}
    limit = WindowLimit(window, 'total', facts.get('source', 'diagnostic declaration'),
                        facts.get('observed', ''), facts.get('acceptance_can_raise', False))
    return SimpleNamespace(window_tokens=window, status='confirmed', stale=False, route_fp='',
        source=limit.source, response_limit=response_limit or {}, route_facts=facts,
        window_record=WindowRecord('', total=limit, reserve=reserve, declared=(limit,)))


def physical_fit(plan, messages, schemas, *, root):
    from ouroboros.context_fit import measure_main_fit, _physical_context_for_fit
    from ouroboros.loop_model_call import _main_context_profile
    from ouroboros.llm_attempt import main_context_fit_refusal
    from ouroboros.usage_accounting import AttemptRequest

    fit = measure_main_fit(plan, messages, schemas, profile=_main_context_profile(plan, plan.initial_mode),
                           rendered_mode=plan.initial_mode, round_id='context-fit-report')
    measurement = fit.measurement
    request = AttemptRequest(model=plan.model, provider=plan.provider, drive_root=root,
        prompt_tokens_estimate=measurement.estimated_input_tokens,
        prompt_tokens_bounded_estimate=measurement.raw_input_tokens,
        max_completion_tokens=measurement.reply_allowance_tokens,
        working_reserve_tokens=plan.working_reserve_tokens, physical_context=_physical_context_for_fit(fit))
    return {**asdict(fit), 'refusal': main_context_fit_refusal(request)}
