"""R28 integration invariants, synthetic memory only; no providers or native state."""
import json

import pytest

from ouroboros import memory_view as mv
from tests._live_world import host_world as host_world
from tests.test_live_conversation import packages as packages


@pytest.mark.parametrize('reader_fails', [False, True])
@pytest.mark.parametrize('provider_id,model_index', [
    ('gemini_live:gemini', 0), ('gemini_live:gemini', 1),
    ('openai_realtime:openai', 0), ('grok_realtime:grok', 0),
])
def test_call_carries_target_waiting_facts_from_real_sources(host_world, packages, provider_id,
                                                            model_index, record_property, monkeypatch, reader_fails):
    from ouroboros import context
    from ouroboros.memory import Memory
    from ouroboros.late_review_marks import publish_late_review
    from ouroboros.tools.chronicle import _chronicle_write
    from ouroboros.context_view_facts import memory_facts
    from ouroboros.memory_waiting import text as waiting_text
    from ouroboros.realtime_setup import loaded_provider, session_candidate
    from ouroboros.contracts.live_conversation import LiveOptions
    from tests.test_late_review_marks import _record

    host = host_world.host
    root = host.ctx.drive_root
    mark = publish_late_review(root, _record(root, 'late-reviewed-work', 1, 'Exact late review.')).mark_id
    note = json.loads(_chronicle_write(host.ctx, kind='note', text='One unsettled thought.'))['node_id']
    assert mark and note
    # Capture through the real initial-context builder after the records exist.
    rows, _ = context.build_llm_messages(host_world.env, Memory(root, host.ctx.repo_dir), host_world.task,
        ctx=host.ctx, tool_schemas=host.tools.schemas(), route_resolver=host.turn.resolve_route)
    host.plan = host.ctx.context_fit_plan
    host.messages[:] = rows
    host._context_changed()
    snapshot = mv.snapshot_from_json(host.plan.core.memory_view_json)
    expected = waiting_text(snapshot.waiting)
    assert snapshot.waiting['late_reviews']['count'] == snapshot.waiting['notes']['count'] == 1
    assert snapshot.waiting['projects']['count'] > 0
    text_facts = memory_facts(host.plan, host.plan.initial_mode, host.messages)
    assert expected and expected in text_facts  # the target's text producer still exposes all three classes

    if reader_fails:
        from ouroboros import artifacts
        store = artifacts.store_actor_source_bytes
        def refuse_reader(*args, **kwargs):
            if kwargs.get('category') == 'context_checkpoints':
                raise OSError('synthetic reader publication failure')
            return store(*args, **kwargs)
        monkeypatch.setattr(artifacts, 'store_actor_source_bytes', refuse_reader)
    provider = loaded_provider(provider_id)
    host.model = provider['catalog'].models[model_index]
    host.turn.native_model = host.model.id
    session, request, evidence = session_candidate(host, provider['session'], LiveOptions())
    delivered = json.loads(session.selected.messages_json)
    def text_content(row):
        body = row.get('content', '')
        return '\n'.join(str(block.get('text', '')) for block in body if isinstance(block, dict)) if isinstance(body, list) else str(body)
    delivered_text = '\n'.join(map(text_content, delivered))
    record_property('waiting_delivery', json.dumps(dict(provider=provider_id, model=host.model.id,
        captured=snapshot.waiting, text_producer=True, call_contains_waiting=expected in delivered_text,
        input_tokens=request.prompt_tokens_estimate, reserve=request.working_reserve_tokens,
        capacity=evidence['fit']['measurement']['capacity_total_tokens'])))
    assert request.prompt_tokens_estimate + request.working_reserve_tokens <= host.model.facts.conversation_window_tokens
    assert expected in delivered_text, 'the call must deliver its already captured waiting census'
    facts = [row for row in delivered if '[CONTEXT_FACTS] Fresh view' in text_content(row)]
    assert len(facts) == 1
    assert ('Exact view before this refresh: read_file' in text_content(facts[0])) is not reader_fails
    if not reader_fails:
        assert text_content(facts[0]).endswith(').'), 'the independent reader stays last'
    # Sealed wire and the admission request include the added facts.
    assert expected in json.dumps(session.wire, ensure_ascii=False)
    from ouroboros.realtime_setup import measure_session_wire
    stripped = [row for row in delivered if row is not facts[0]]
    _, bare, _, _ = measure_session_wire(host, host.plan, stripped, host.schemas,
        provider['session'], LiveOptions(), _room_fact=False)
    assert request.prompt_tokens_estimate > bare.prompt_tokens_estimate
