"""Description cost changes; every original typed binding and required field survives."""
import hashlib
import json

from ouroboros.tools.compact_context import get_tools
from ouroboros.tool_policy import select_tool_schemas
from ouroboros.utils import estimate_tokens


def _types(value):
    if isinstance(value, dict):
        return {k: _types(v) for k, v in value.items() if k != 'description'}
    return [_types(v) for v in value] if isinstance(value, list) else value


def test_schema_cost_and_all_typed_bindings():
    schema = get_tools()[0].schema
    digest = hashlib.sha256(json.dumps(_types(schema), sort_keys=True).encode()).hexdigest()
    assert digest == '79d0df7bfb9bfa55549cd972eb12068166814026e7098dc1e8a03e7fc5eb02c2'
    raw = json.dumps(schema, ensure_ascii=False)
    # 4,271 after the nano pass, plus the two owner-approved phrases the compaction test pins and the four
    # working_note phrases test_main_authored_context pins (future reader, within those sources, unaffected
    # branches, invent no detail), which the nano pass had dropped.
    assert 3800 <= len(raw) <= 4360
    print(json.dumps({'before_chars': 5978, 'after_chars': len(raw), 'before_chars_div_4': 1494.5,
                      'after_chars_div_4': len(raw) / 4, 'before_server_tokens': 1495,
                      'after_server_tokens': estimate_tokens(raw)}, sort_keys=True))
    for name in ('review_notes', 'review_transfers'):
        assert schema['parameters']['properties'][name]['description'].count('.') == 1


def test_compact_tool_stays_in_nano_and_low_max_keep_the_permitted_envelope():
    schemas = [dict(type='function', function=get_tools()[0].schema),
               dict(type='function', function=dict(name='read_file', parameters={'type': 'object'}))]
    for mode in ('max', 'low'):
        assert list(select_tool_schemas(schemas, context_mode=mode).schemas) == schemas
    assert 'compact_context' in {s['function']['name'] for s in select_tool_schemas(schemas, context_mode='nano').schemas}
