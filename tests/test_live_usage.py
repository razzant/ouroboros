"""Fixture tariffs and real ledger rows; no public catalog or provider I/O."""
import hashlib
import json

import pytest

from ouroboros import live_usage as live, pricing_catalog as catalog, usage_accounting as ua
from ouroboros.utils import utc_now_iso


def token_usage(**reported):
    return {**{key: 0 for key in live.USAGE_AXES if key != 'connected_seconds'}, **reported}


@pytest.fixture
def quote(tmp_path, monkeypatch):
    monkeypatch.setenv('OUROBOROS_DATA_DIR', str(tmp_path))
    monkeypatch.setenv('TOTAL_BUDGET', '100')
    rows = {'gpt-voice-test': {'litellm_provider': 'openai', 'max_tokens': 32000,
        'input_cost_per_token': 2e-6, 'output_cost_per_token': 8e-6,
        'input_cost_per_audio_token': 10e-6, 'output_cost_per_audio_token': 20e-6,
        'input_cost_per_image_token': 5e-6, 'cache_read_input_token_cost': .2e-6,
        'cache_read_input_audio_token_cost': 1e-6, 'cache_read_input_image_token_cost': .5e-6},
        'gemini/gemini-test': {'litellm_provider': 'gemini', 'input_cost_per_token': 1e-6},
        'xai/grok-test': {'litellm_provider': 'xai', 'input_cost_per_second': .05 / 60}}
    raw = json.dumps(rows).encode()
    def fetch():
        return catalog._decode(dict(url=catalog.CATALOG_URL, content=raw.decode(),
            fetched_at=utc_now_iso(), sha256=hashlib.sha256(raw).hexdigest()))
    monkeypatch.setattr(catalog, '_fetch_catalog', fetch)
    monkeypatch.setattr(catalog, '_cache', {})
    monkeypatch.setattr(catalog, '_retry_after', {})
    monkeypatch.setattr(catalog, '_fetching', set())
    value = live.warm_quote(engine='openai_realtime', model='openai::gpt-voice-test', drive_root=tmp_path)
    monkeypatch.setattr(catalog, '_fetch_catalog', lambda: pytest.fail('network lookup on audio path'))
    return tmp_path, value, rows


def reserve(root, quote):
    return live.reserve_live_response(engine='openai_realtime', model='openai::gpt-voice-test',
        call_id='call-test', response_key='response-test', task_id='task-test',
        rendered_mode='nano', drive_root=root, quote=quote)


def test_exact_quote_retains_entry_digest_date_and_route_identity(quote):
    root, value, rows = quote
    assert value['entry'] == rows['gpt-voice-test']
    assert value['sha256'] == hashlib.sha256(json.dumps(rows).encode()).hexdigest()
    assert value['fetched_at']
    for provider, model, expected in [('gemini_live', 'google::gemini-test', 'gemini/gemini-test'),
            ('grok_realtime', 'grok::grok-test', 'xai/grok-test')]:
        result = catalog.get_catalog_quote(model, provider=provider, drive_root=root, allow_live_fetch=False)
        assert result['catalog_id'] == expected
    assert catalog.get_catalog_quote('openai::gpt-voice-test-new', provider='openai_realtime', drive_root=root,
                                      allow_live_fetch=False) is None
    assert catalog.get_catalog_quote('google::gpt-voice-test', provider='openai_realtime', drive_root=root,
                                      allow_live_fetch=False) is None


def test_modalities_cache_interrupt_and_dedup_keep_original_quote(quote):
    root, value, _ = quote
    attempt = reserve(root, value)
    row, = ua.read_usage_records(root)
    assert row['state'] == 'dispatched' and row['cost_evidence']['tariff_quote'] == value
    usage = token_usage(text_input_tokens=100, audio_input_tokens=200, image_input_tokens=50,
                 cached_input_tokens=55, text_output_tokens=30, audio_output_tokens=60)
    raw = {'input_token_details': {'cached_tokens_details': {'text_tokens': 20, 'audio_tokens': 30, 'image_tokens': 5}}}
    for _ in range(2):
        live.settle_live_response(attempt, usage=usage, raw_usage=raw, status='interrupted')
    row, = ua.read_usage_records(root)
    assert row['cost_usd'] == pytest.approx(.0035615)
    assert row['cost_evidence']['status'] == 'interrupted'
    assert row['cost_evidence']['tariff_quote'] == value
    assert row['cost_final'] is False


def test_missing_modality_price_never_falls_back_to_text(quote, monkeypatch):
    root, value, _ = quote
    value['entry'].pop('output_cost_per_audio_token')
    monkeypatch.setattr(ua, 'estimate_cost_optional', lambda *a, **k: pytest.fail('silently repriced'))
    attempt = reserve(root, value)
    live.settle_live_response(attempt, usage={'text_input_tokens': 100, 'audio_output_tokens': 10},
                             raw_usage={}, status='completed')
    row, = ua.read_usage_records(root)
    assert row['cost_usd'] is None and row['cost_coverage'] == 'partial'
    assert live._price(value, token_usage(text_input_tokens=100), {})[0] == pytest.approx(.0002)
    assert live._price(value, {'text_input_tokens': 100}, {})[0] is None  # absent axes are unknown


def test_connected_seconds_and_unreported_usage(quote):
    root, value, _ = quote
    meter = catalog.get_catalog_quote('xai/grok-test', provider='grok_realtime', drive_root=root, allow_live_fetch=False)
    assert live._price(meter, {'connected_seconds': 60}, {})[0] == pytest.approx(.05)
    assert live._price(None, {'connected_seconds': 60}, {})[0] is None
    assert live._price(value, {}, {})[0] is None
    assert live._price(value, token_usage(), {'input_tokens': 0})[0] == 0


def test_explicit_unbilled_zero_and_additional_thinking_contract(quote):
    _, value, _ = quote
    # The provider declares its impossible axes; the host must not invent them.
    usage = dict(text_input_tokens=100, audio_input_tokens=0, image_input_tokens=0,
        video_input_tokens=0, cached_input_tokens=0, text_output_tokens=10,
        audio_output_tokens=0, thinking_tokens=3)
    assert live._price(value, usage, {})[0] == pytest.approx(.000304)
    # Reasoning already included in text output is counted once by its producer.
    assert live._price(value, {**usage, 'text_output_tokens': 13, 'thinking_tokens': 0}, {})[0] == pytest.approx(.000304)
    for axis in ('video_input_tokens', 'thinking_tokens', 'audio_output_tokens'):
        incomplete = {key: count for key, count in usage.items() if key != axis}
        cost, evidence = live._price(value, incomplete, {})
        assert cost is None and 'unknown_' + axis in evidence['unpriced']


def test_shared_limit_refuses_next_response_but_keeps_late_charge(quote):
    root, value, _ = quote
    with ua.usage_scope(ua.UsageScope(drive_root=root, task_id='task-test', global_limit_usd=.001)):
        attempt = reserve(root, value)
        live.settle_live_response(attempt, usage=token_usage(audio_output_tokens=100), raw_usage={}, status='completed')
        with pytest.raises(ua.BudgetExceeded):
            reserve(root, value)
        live.record_refused_live_response(engine='openai_realtime', model='openai::gpt-voice-test',
            call_id='call-test', response_key='late', task_id='task-test', rendered_mode='nano', drive_root=root,
            reason='budget_exhausted', usage=token_usage(audio_output_tokens=10), raw_usage={}, status='interrupted', quote=value)
    assert len(ua.read_usage_records(root)) == 2
    assert sum(r['cost_usd'] for r in ua.read_usage_records(root)) == pytest.approx(.0022)


@pytest.mark.parametrize('raw', [
    {'input_tokens_details': {'text_tokens': 100, 'audio_tokens': 999}},
    {'usage': {'input_token_details': {'text_tokens': 100, 'image_tokens': 888}}},
    {'usageMetadata': {'promptTokensDetails': [{'modality': 'TEXT', 'tokenCount': 100},
                                             {'modality': 'VIDEO', 'tokenCount': 777}]}},
])
def test_density_witness_requires_matching_explicit_text_modality(raw):
    from ouroboros.live_usage import text_input_witness
    assert text_input_witness({'text_input_tokens': 100}, raw) == 100
    assert text_input_witness({'text_input_tokens': 101}, raw) is None
    assert text_input_witness({'audio_input_tokens': 100}, raw) is None
    assert text_input_witness({'text_input_tokens': 100}, {'input_tokens': 100}) is None
    assert text_input_witness({'text_input_tokens': 100}, {}) is None


@pytest.mark.parametrize('raw,expected', [
    ({'input_tokens': 123, 'output_tokens': 456, 'cached_input_tokens': 100}, 123),
    ({'usage': {'input_tokens': 123}}, 123),
    ({'usageMetadata': {'promptTokenCount': 123}}, 123),
    ({'usage_metadata': {'prompt_token_count': 123}}, 123),
    ({'input_token_details': {'text_tokens': 123}, 'output_tokens': 456}, None),
    ({'connected_seconds': 123}, None),
    ({'input_tokens': True}, None),
    ({'input_tokens': '123'}, None),
    ({'input_tokens': -123}, None),
    ({'input_tokens': 0}, None),
    ({'input_tokens': 123, 'promptTokenCount': 456}, None),
    ({}, None),
])
def test_input_window_witness_uses_only_a_reported_input_total(raw, expected):
    assert live.input_token_witness(raw) == expected
