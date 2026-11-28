"""Ordinary Settings, skill, and worker entrypoints on a fresh synthetic installation.

Only the serving counter is replaced; the route, root capture, policy, and writes
are the production paths. The deterministic token weights are fixture values.
"""
from contextlib import nullcontext
import hashlib
import json
from types import SimpleNamespace

import pytest
from starlette.requests import Request

from ouroboros import config, llm_local, local_model
from ouroboros.chronicle_store import ChronicleStore
from ouroboros.gateway import safety_settings
from ouroboros.gateway.schema import contract_document, contract_typed_dict, validate_ingress
from ouroboros.knowledge import read_knowledge_note, resolve_knowledge_address
from ouroboros.llm import LLMClient
from ouroboros.local_model_server import input_fingerprint
from ouroboros.nano_host_publication import measure_publication_floor
from ouroboros.nano_publication import read_root_sample
from ouroboros.safety_config import OWNER_RULES_KEY
from ouroboros.skill_lifecycle_actions import _toggle
from ouroboros.skill_loader import compute_content_hash, discover_skills, load_enabled
from ouroboros.tools.chronicle import _chronicle_write, _memory_mark
from ouroboros.tools.control_runtime import _update_identity, _update_scratchpad
from ouroboros.tools.knowledge import _knowledge_write
from ouroboros.tools.registry import ToolContext
from ouroboros.utils import atomic_write_json, machine_json
from tests._memory_view_context import world

WEIGHT = 'FIXTURE_WEIGHT_500'
WINDOW = 80_000
PORT = 8767


@pytest.fixture
def installation(tmp_path, monkeypatch):
    env, memory, _ = world(tmp_path)
    root = memory.drive_root
    memory.ensure_files()
    ChronicleStore(root).ensure_activated()
    memory.identity_path().write_text('A synthetic identity.', encoding='utf-8')
    atomic_write_json(memory.scratchpad_blocks_path(), [])
    monkeypatch.setattr(config, 'DATA_DIR', root)
    monkeypatch.setattr(config, 'REPO_DIR', env.repo_dir)
    monkeypatch.setattr(config, 'SETTINGS_PATH', root / 'settings.json')
    settings = {'OUROBOROS_CONTEXT_MODE': 'nano', 'OUROBOROS_CONTEXT_MODE_AUTO_LOW': 'false',
                'OUROBOROS_CONSCIOUSNESS_SAFETY_MODE': 'full', 'USE_LOCAL_MAIN': 'true',
                'LOCAL_MODEL_CONTEXT_LENGTH': WINDOW, 'LOCAL_MODEL_PORT': PORT,
                'LOCAL_MODEL_SOURCE': 'fixture.gguf', OWNER_RULES_KEY: 'Initial synthetic rules.'}
    atomic_write_json(config.SETTINGS_PATH, settings)
    ctx = ToolContext(repo_dir=env.repo_dir, drive_root=root, current_chat_id=1, task_id='abc01234',
                      current_task_type='task', budget_drive_root=str(root))
    live = {'pid': 7, 'active': True, 'supports_count': True, 'externals': [], 'on_count': None, 'b': 36_808}

    def count(payload, *, expected_port):
        assert expected_port == PORT
        if live['on_count']:
            live['on_count']()
        wire = machine_json(payload)
        live['externals'].append(wire)
        if not live['supports_count']:
            return {'supported': False, 'reason': 'unsupported_fixture'}
        count = live['b'] + wire.count(WEIGHT) * 500
        if (extra_weight := live.get('extra_weight')) and extra_weight in wire:
            count += 500
        return {'supported': True, 'input_is_exact': True, 'output_limit_enforced': True, 'reasoning_included_in_limit': True,
                'input_tokens': count, 'context_window': WINDOW, 'raw_context_window': WINDOW + 128,
                'native_input_sha256': input_fingerprint(payload), 'process_id': live['pid'],
                'tokenizer_template_provenance': {'source': 'serving_llama_formatter', 'formatter': 'fixture',
                    'library_version': 'fixture', 'chat_format': 'fixture', 'launched_n_ctx': WINDOW,
                    'rendered_prompt_sha256': hashlib.sha256(wire.encode()).hexdigest()}}

    monkeypatch.setattr(local_model, 'get_manager', lambda **_: SimpleNamespace(
        serving_context_evidence=lambda **_: {'confirmed': live['active'], 'context_window': WINDOW,
                                               'process_id': live['pid'], 'port': PORT}, measure_prepared_input=count))
    monkeypatch.setattr(local_model, 'local_dispatch_port', lambda: PORT)
    monkeypatch.setattr(llm_local, 'measurement_slot', lambda *a, **kw: nullcontext(True))
    monkeypatch.setattr(LLMClient, '_get_local_client', lambda *a: pytest.fail('count-only, no inference'))
    return memory, ctx, settings, live


def _rules(text):
    request = Request({'type': 'http', 'method': 'POST', 'path': '/api/owner/safety-rules', 'headers': []})
    response = safety_settings._api_owner_safety_rules_sync(request, {'rules': text})
    return response.status_code, json.loads(response.body)


def _identity(ctx, memory, text):
    revision = hashlib.sha256(memory.identity_path().read_bytes()).hexdigest()
    return str(_update_identity(ctx, content=text, expected_revision=revision))


def test_worker_enforces_saturated_ordinary_root_and_retains_proposal(installation):
    memory, ctx, _settings, live = installation
    full = f'Composed test identity {WEIGHT} ' + f'{WEIGHT} ' * 39
    assert 'RESOURCE_CONSTRAINT_BLOCKED' not in _identity(ctx, memory, full)
    sample = measure_publication_floor(memory.drive_root)
    assert sample['status'] == 'exact', sample
    assert (sample['memory_tokens'], sample['minimum_work_headroom_at_reply_floor_tokens']) == (20_000, 15_000)
    protection = read_root_sample(memory.drive_root, provider='local')
    assert (protection['status'], protection['protection_scope'], protection['qualified_background_tokens']) == (
        'protecting', 'unchanged_main_background', 36_808)
    assert any('Work starts here.' in wire and 'Initial synthetic rules.' in wire for wire in live['externals'])
    before = memory.load_scratchpad_blocks()
    proposal = f'Synthetic pending block {WEIGHT}.'
    refusal = _update_scratchpad(ctx, proposal)
    assert 'RESOURCE_CONSTRAINT_BLOCKED' in refusal and 'Free room by folding memory' in refusal
    assert memory.load_scratchpad_blocks() == before
    assert proposal in next(memory.drive_root.glob('task_results/artifacts/abc01234/**/*nano-proposed-memory*.json')).read_text()
    assert 'RESOURCE_CONSTRAINT_BLOCKED' not in _identity(ctx, memory, 'A verified shorter identity.')
    assert 'RESOURCE_CONSTRAINT_BLOCKED' not in _update_scratchpad(ctx, proposal)
    assert memory.load_scratchpad_blocks()[-1]['content'] == proposal


def test_writer_compares_complete_world_even_when_root_would_address_it(installation):
    from ouroboros.tools.control_runtime import _update_world

    memory, ctx, _settings, live = installation
    world_file = memory.world_path()
    identity = 'Synthetic fixed identity. ' + (WEIGHT + ' ') * 30
    assert 'RESOURCE_CONSTRAINT_BLOCKED' not in _identity(ctx, memory, identity)

    # A previous startup/tool version left more world text than this profile's
    # reserves can hold. Preserve the existing host-measured block in the fixture.
    marker = 'Synthetic costly WORLD paragraph.'
    with world_file.open('a', encoding='utf-8') as stream:
        stream.write('\n' + marker + '\n' + (WEIGHT + ' ') * 47 + '\n')

    def rewrite(count):
        live['externals'].clear()
        text = marker + '\n' + (WEIGHT + ' ') * count
        return str(_update_world(ctx, content=text, expected_revision=hashlib.sha256(world_file.read_bytes()).hexdigest()))

    before = measure_publication_floor(memory.drive_root)
    assert before['world_form'] == 'whole'
    assert before['without_memory']['input_tokens'] == 60_308
    assert 'RESOURCE_CONSTRAINT_BLOCKED' not in rewrite(30)
    assert any(marker in wire and wire.count(WEIGHT) == 47 for wire in live['externals'])
    assert any(marker in wire and wire.count(WEIGHT) == 30 for wire in live['externals'])
    improved = measure_publication_floor(memory.drive_root)
    assert improved['without_memory']['input_tokens'] == 51_808
    assert improved['whole_memory_tokens'] == before['whole_memory_tokens'] == 15_000
    saved = world_file.read_bytes()
    for count in (32, 80):  # small growth and growth big enough to hide behind a reader
        refusal = rewrite(count)
        assert 'RESOURCE_CONSTRAINT_BLOCKED' in refusal and 'over_capacity' in refusal
        assert any(marker in wire and wire.count(WEIGHT) == count for wire in live['externals'])
        assert world_file.read_bytes() == saved


def test_deferred_local_port_change_counts_only_the_dispatch_binding(installation, monkeypatch):
    from ouroboros import nano_host_publication, platform_layer, server_process

    memory, _ctx, settings, _live = installation
    task = {'chat_id': 1, 'delegation_role': 'root'}
    launch = {'port': PORT, 'pid': 7}
    dispatch = {'port': PORT}
    monkeypatch.setattr(local_model, 'local_dispatch_port', lambda: dispatch['port'])
    monkeypatch.setattr(local_model, 'get_manager', lambda: SimpleNamespace(
        serving_context_evidence=lambda **_: {'confirmed': True, 'context_window': WINDOW,
                                               'process_id': launch['pid'], 'port': launch['port']}))
    monkeypatch.setattr(server_process, 'read_service_bindings', lambda _root: {'local_model': dict(launch)})
    monkeypatch.setattr(platform_layer, 'process_command', lambda pid: (
        f'python -m ouroboros.local_model_server --model /fixture-{pid}.gguf --port {launch["port"]}'
        f' --n_ctx {WINDOW}'))

    def reader():
        return nano_host_publication.main_extension_tool_reader(memory.drive_root, task, 'nano')

    original = nano_host_publication._profile_key()
    assert original.startswith('launch:') and reader()
    new_port = PORT + 1
    atomic_write_json(config.SETTINGS_PATH, {**settings, 'LOCAL_MODEL_PORT': new_port})
    assert nano_host_publication._profile_key() == original and reader()  # saved, not routed yet
    dispatch['port'] = new_port
    pending = nano_host_publication._profile_key()
    assert pending.startswith('settings:') and pending != original and not reader()
    launch.update(port=new_port, pid=8)
    started = nano_host_publication._profile_key()
    assert started.startswith('launch:') and started != original and reader()


def test_writer_samples_web_runtime_once_without_freezing_the_edited_identity(installation, monkeypatch):
    from ouroboros import context

    memory, ctx, _settings, live = installation
    calls = []
    ordinary = context.build_runtime_section

    def changing_activity(*args, **kwargs):
        runtime = ordinary(*args, **kwargs)
        calls.append(kwargs['captured_at'])
        return runtime if len(calls) == 1 else runtime + ' ' + WEIGHT

    monkeypatch.setattr(context, 'build_runtime_section', changing_activity)
    # B is saturated. Re-rendering runtime on the candidate would add 500 to B
    # and refuse this safe edit. The edited identity must still appear on the wire.
    proposal = f'Changed synthetic identity {WEIGHT}.'
    assert 'RESOURCE_CONSTRAINT_BLOCKED' not in _identity(ctx, memory, proposal)
    assert len(calls) == 1
    assert all('owner_client_note' in wire and 'routing_contract' in wire for wire in live['externals'])
    assert any('A synthetic identity.' in wire for wire in live['externals'])
    assert any(proposal in wire for wire in live['externals'])
    assert memory.identity_path().read_text(encoding='utf-8').strip() == proposal


def test_writer_repairs_an_incomplete_saved_identity_only_with_a_measured_fit(installation):
    memory, ctx, _settings, _live = installation
    assert 'RESOURCE_CONSTRAINT_BLOCKED' not in _identity(ctx, memory, 'Counted before the source was damaged.')
    memory.identity_path().write_text('', encoding='utf-8')
    too_large = ('oversize ' + WEIGHT + '\n') * 41
    assert 'RESOURCE_CONSTRAINT_BLOCKED' in _identity(ctx, memory, too_large)
    assert memory.identity_path().read_text(encoding='utf-8') == ''
    assert 'RESOURCE_CONSTRAINT_BLOCKED' not in _identity(ctx, memory, 'An exact fitting repair.')
    assert memory.identity_path().read_text(encoding='utf-8').strip() == 'An exact fitting repair.'


@pytest.mark.parametrize('cold', [False, True])
def test_fresh_main_sender_and_writer_share_the_count_frame(installation, monkeypatch, cold):
    from ouroboros.context import build_context_fit_plan
    from ouroboros.agent import Env
    from ouroboros.loop_llm_call import MAIN_LOOP_MAX_TOKENS
    from ouroboros.nano_host_publication import _as_next_root
    from ouroboros.nano_minimum import measurement_episode
    from ouroboros.send_clock import SendClockPolicy, main_send_scope
    from ouroboros.tool_policy import initial_tool_schemas
    from ouroboros.tools.registry import ToolRegistry
    from ouroboros.usage_accounting import PhysicalAttemptContext, bind_physical_attempt_context

    memory, ctx, settings, live = installation
    root, repo = memory.drive_root, ctx.repo_dir
    instruction = 'An admitted synthetic instruction that must remain in both counts.'
    guide = _install_instruction(root, 'fixture-always-on', instruction)
    from ouroboros.skill_loader import save_enabled
    save_enabled(root, guide.name, True, actor='owner_ui')
    task = {'type': 'task', 'id': 'a17e0001', 'chat_id': 1, 'delegation_role': 'root',
            'budget_drive_root': str(root), 'text': 'A new request using the accepted memory.'}
    tools = ToolRegistry(repo, root)
    tools.set_context(ToolContext(repo, root, current_chat_id=1, current_task_type='task', task_id=task['id']))
    with _as_next_root(settings):
        plan = build_context_fit_plan(Env(repo, root), memory, task, preferred_mode='nano', ctx=tools._ctx,
            tool_schemas=initial_tool_schemas(tools, context_mode='nano'), tool_registry=tools)

    lane = LLMClient()
    sent, captured = [], []

    def create(**wire):
        sent.append(wire)
        return SimpleNamespace(model_dump=lambda: {'choices': [{'message': {'role': 'assistant', 'content': 'fixture'}}]})

    def execute(_request, send, before):
        facts = before(SimpleNamespace(attempt_id='fresh-main', drive_root=root))
        from pathlib import Path
        captured.append(json.loads(Path(facts['path']).read_text(encoding='utf-8'))['nano_minimum'])
        return send()

    monkeypatch.setattr(lane, '_get_local_client', lambda: SimpleNamespace(base_url=f'http://127.0.0.1:{PORT}/v1',
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))))
    monkeypatch.setattr(llm_local, '_execute_candidate', execute)
    monkeypatch.setattr(llm_local, 'limit_physical_send', lambda _, send: send)
    lookahead = lane._local_publication_floor(plan, MAIN_LOOP_MAX_TOKENS)
    rows = plan.messages_for('nano')
    round_idx, usage = 0, {}
    if cold:
        from ouroboros import owner_wait
        from ouroboros.context_budget import LocalContextTooLargeError
        from ouroboros.loop_round_limits import COLD_RETURN_ATTR

        old = tools._ctx
        old.context_fit_plan, old.active_context_mode, old.active_use_local = plan, 'nano', True
        old.active_model, old.active_effort = 'local-model', ''
        old.primary_route = {'model': 'local-model', 'use_local': True, 'role': 'main'}
        original_words = task['text']
        state = json.loads(json.dumps(owner_wait.continuation_state(
            old, rows, {}, {}, 1, lookahead.actual['tools'], set())))
        tools = ToolRegistry(repo, root)
        restored = ToolContext(repo, root, current_chat_id=1, current_task_type='task', task_id=task['id'])
        tools.set_context(restored)
        with _as_next_root(settings):
            restored.context_fit_plan = build_context_fit_plan(Env(repo, root), memory, task, preferred_mode='nano',
                ctx=restored, tool_schemas=initial_tool_schemas(tools, context_mode='nano'), tool_registry=tools)
            owner_wait.restore_continuation_state(tools, state, rows, {}, usage, set())
            plan, mode = owner_wait.rebind_restored_route(tools, state, rows)
        assert getattr(restored, COLD_RETURN_ATTR) and mode == 'nano'
        assert original_words in json.dumps(rows)
        round_idx = 1

    round_context = SimpleNamespace(active_use_local=True, task_type='task', tools=tools,
                                    accumulated_usage=usage, round_idx=round_idx)
    episode = measurement_episode(round_context, plan)
    assert episode == ('cold_resume' if cold else 'cold_start')
    physical = PhysicalAttemptContext('owner_nano', 'nano', 'cold_estimate', 'fixture', 'one', WINDOW, WINDOW,
        False, False, preferred_mode='nano', nano_floor_plan=(plan, episode))
    with bind_physical_attempt_context(physical), main_send_scope(SendClockPolicy()):
        if cold:
            live['b'] = 79_000
            with pytest.raises(LocalContextTooLargeError, match='79[0]00 exact input tokens'):
                lane._chat_local(rows, lookahead.actual['tools'], MAIN_LOOP_MAX_TOKENS, 'auto')
            assert not sent and not captured
            assert all(original_words in wire for wire in live['externals'])
            return
        lane._chat_local(rows, lookahead.actual['tools'], MAIN_LOOP_MAX_TOKENS, 'auto')
    floor, = captured
    assert floor['status'] == 'exact' and floor['room'] == 'main', floor
    assert len(sent) == 1
    assert input_fingerprint(sent[0]) == floor['actual']['native_input_sha256']
    assert all(wire.count(instruction) == 1 for wire in live['externals'])
    assert all('A new request using the accepted memory.' in wire for wire in live['externals'])
    fact = read_root_sample(root, provider='local')
    assert fact['observation'] == 'prepared' and fact['status'] == 'not_claimed', fact

    # Exact ordinary first entry, before any admitted publication. A busy
    # count retains the proposal and cannot silently activate it.
    monkeypatch.setattr(llm_local, 'measurement_slot', lambda *a, **kw: nullcontext(False))
    result = str(_update_scratchpad(ctx, 'A retryable proposal.'))
    assert 'RESOURCE_CONSTRAINT_BLOCKED' in result and 'busy' in result
    assert 'retry when the counter is free' in result and 'Free room' not in result
    assert 'nano-proposed-memory' in result
    assert not memory.load_scratchpad_blocks()
    assert read_root_sample(root, provider='local')['measurement'] == 'exact'


def test_settings_ordinary_writer_keeps_old_text_then_saves_smaller_text(installation):
    memory, _ctx, _settings, live = installation
    too_much = WEIGHT * 41
    status, result = _rules(too_much)
    assert (status, result['code'], result['proposed_rules']) == (409, 'nano_publication_over_capacity', too_much)
    assert result['saved'] is False and 'Make room' in result['error']
    assert not list(memory.drive_root.rglob('*nano-proposed-memory*.json'))
    assert json.loads(config.SETTINGS_PATH.read_text())[OWNER_RULES_KEY] == 'Initial synthetic rules.'
    status, result = _rules('Smaller synthetic rules.')
    assert status == 200, result.get('nano_reason', result)
    assert json.loads(config.SETTINGS_PATH.read_text())[OWNER_RULES_KEY] == 'Smaller synthetic rules.'
    assert any('Smaller synthetic rules.' in wire for wire in live['externals'])
    fact = read_root_sample(memory.drive_root, provider='local')
    assert fact['status'] == 'protecting', fact


@pytest.mark.parametrize(('reason', 'next_step'), [
    ('busy', 'Try again when it is free.'),
    ('unavailable', 'Try again when it is available.'),
    ('stale', 'Check the current rules before retrying.'),
    ('bootstrap_pending', 'Start a chat, then save again.'),
    ('incomplete', 'Check the reported source before saving again.'),
])
def test_settings_nano_refusals_echo_owner_input_without_claiming_growth(installation, monkeypatch, reason, next_step):
    from ouroboros import nano_activation
    from ouroboros.nano_publication import NanoPublicationRefused

    def fail_count(root, text):
        raise NanoPublicationRefused(reason, 'Synthetic check detail.')

    monkeypatch.setattr(nano_activation, 'owner_rules_ticket', fail_count)
    draft = 'Сначала сверить.\nНе отправлять. '
    before = config.SETTINGS_PATH.read_bytes()
    status, result = _rules(draft)
    route = next(r for r in contract_document()['http'] if r['path'] == '/api/owner/safety-rules')
    name = 'OwnerSafetyRulesNanoRefusal'
    assert name in route['responses']['409']['schemas']
    assert not validate_ingress(result, contract_typed_dict(name))
    assert (status, result['saved'], result['code'], result['proposed_rules'], result['nano_reason']) == (
        409, False, f'nano_publication_{reason}', draft, 'Synthetic check detail.')
    assert next_step in result['error'] and 'Make room' not in result['error']
    assert config.SETTINGS_PATH.read_bytes() == before


def test_normal_mark_and_overview_share_the_same_root_limit(installation):
    memory, ctx, _settings, _live = installation
    root = memory.drive_root
    assert 'RESOURCE_CONSTRAINT_BLOCKED' not in _identity(ctx, memory, 'An identity ' + WEIGHT * 39)
    note = json.loads(_chronicle_write(ctx, kind='note', text='An original synthetic fact.'))
    assert note['ok']
    mark = json.loads(_memory_mark(ctx, node_id=note['node_id'], text=f'A counted mark {WEIGHT}', scope='global'))
    assert mark['ok'], mark
    assert measure_publication_floor(root)['memory_tokens'] == 20_000
    blocked_mark = json.loads(_memory_mark(ctx, node_id=note['node_id'], text=f'Refusible mark {WEIGHT}', scope='global'))
    assert blocked_mark['reason'] == 'over_capacity' and 'nano-proposed-memory' in blocked_mark['detail']
    assert _knowledge_write(ctx, 'overview', '# Baseline overview\nUseful facts on the shelf', mode='overwrite')
    address = resolve_knowledge_address(root, 'overview', 'global')
    before = read_knowledge_note(address)
    proposal = f'# Bigger overview\nA readable proposal {WEIGHT}'
    blocked_note = str(_knowledge_write(ctx, 'overview', proposal, mode='overwrite', expected_revision=before.revision))
    assert 'over_capacity' in blocked_note and 'nano-proposed-memory' in blocked_note
    assert read_knowledge_note(address).raw == before.raw
    assert json.loads(_memory_mark(ctx, release_ids=[mark['mark_id']], reason='This size is no longer needed.'))['ok']
    _knowledge_write(ctx, 'overview', proposal, mode='overwrite', expected_revision=before.revision)
    assert proposal in read_knowledge_note(address).text


def test_protection_drops_on_changed_background_and_does_not_hide_root_pressure(installation):
    memory, ctx, _settings, live = installation
    root = memory.drive_root
    full = 'A focused identity ' + WEIGHT * 40
    assert 'RESOURCE_CONSTRAINT_BLOCKED' not in _identity(ctx, memory, full)
    assert read_root_sample(root, provider='local')['status'] == 'protecting'
    original_world = memory.world_path().read_bytes()
    memory.world_path().write_bytes(original_world + b'\nNew mandatory world detail.\n')
    assert read_root_sample(root, provider='local')['status'] == 'not_claimed'
    assert 'RESOURCE_CONSTRAINT_BLOCKED' not in _identity(ctx, memory, full + ' (reworded)')
    assert read_root_sample(root, provider='local')['status'] == 'protecting'
    # Simulate the exact count of a previously untracked, newly mandatory
    # runtime row. Not controlling its arrival never lets a stale proof stand
    # after the next complete observation, even if that proposal is refused.
    live['b'] += 500
    assert 'RESOURCE_CONSTRAINT_BLOCKED' in _update_scratchpad(ctx, f'Pending {WEIGHT}.')
    assert read_root_sample(root, provider='local')['status'] == 'not_claimed'


def test_profile_change_invalidates_inflight_proposal_and_unsupported_route_does_not_count(installation):
    memory, ctx, settings, live = installation
    from ouroboros.settings_integrity import settings_publication_boundary

    def switch():
        live['on_count'] = None
        with settings_publication_boundary(memory.drive_root):
            atomic_write_json(config.SETTINGS_PATH, {**settings, 'USE_LOCAL_MAIN': 'false', 'LOCAL_MODEL_SOURCE': ''})

    live['on_count'] = switch
    before = memory.load_identity()
    assert 'RESOURCE_CONSTRAINT_BLOCKED' in _identity(ctx, memory, 'Candidate for the old profile.')
    assert memory.load_identity() == before
    live['externals'].clear()
    assert 'RESOURCE_CONSTRAINT_BLOCKED' not in _update_scratchpad(ctx, f'Ordinary change on another route {WEIGHT * 45}.')
    assert not live['externals']


def test_never_confirmed_editing_and_previously_confirmed_outage_are_different(installation):
    memory, ctx, _settings, live = installation
    root = memory.drive_root
    live['active'] = False
    live['supports_count'] = False
    # A configured 80k slot is not evidence that this install ever counted.
    assert 'RESOURCE_CONSTRAINT_BLOCKED' not in _identity(ctx, memory, 'Writable before Local starts.')
    assert not live['externals']
    assert read_root_sample(root, provider='local')['status'] == 'not_claimed'
    live['active'] = True
    assert _rules('Writable with a server that does not count.')[0] == 200
    assert live['externals'] and read_root_sample(root, provider='local')['status'] == 'not_claimed'
    live['supports_count'] = True
    assert 'RESOURCE_CONSTRAINT_BLOCKED' not in _identity(ctx, memory, 'Now the formatter is verified.')
    assert read_root_sample(root, provider='local')['status'] == 'protecting'
    live['supports_count'] = False
    refusal = _update_scratchpad(ctx, 'Readable proposal while the confirmed formatter is unavailable.')
    assert 'RESOURCE_CONSTRAINT_BLOCKED' in refusal and 'unsupported_fixture' in refusal
    assert not memory.load_scratchpad_blocks()
    status = read_root_sample(root, provider='local')
    assert (status['status'], status['reason']) == ('not_claimed', 'confirmed_counter_unavailable')
    live['active'] = False
    assert _rules('Do not silently bypass the checked profile.')[1]['code'] == 'nano_publication_unavailable'


def _install_instruction(root, name, text):
    path = root / 'skills' / 'external' / name
    path.mkdir(parents=True)
    (path / 'SKILL.md').write_text(f'---\nname: {name}\ndescription: A synthetic guide\ntype: instruction\n'
                                 f'always_in_context: true\n---\n{text}\n', encoding='utf-8')
    state = root / 'state' / 'skills' / name
    state.mkdir(parents=True)
    atomic_write_json(state / 'review.json', {'status': 'pass', 'content_hash': compute_content_hash(path)})
    return next(skill for skill in discover_skills(root, repo_path=config.REPO_DIR) if skill.name == name)


def _catalog_update(monkeypatch, root, tmp_path, *, trust, name='fixture-guide', suffix='', resident=True):
    from ouroboros import skill_source_install as installer
    from ouroboros.skill_install_record import read_install_record, write_install_record
    from ouroboros.skill_loader import save_enabled
    from ouroboros.skill_source_catalog import tree_release_digest

    old = _install_instruction(root, name, f'Former instruction body.{suffix}')
    if not resident:
        manifest = old.skill_dir / 'SKILL.md'
        manifest.write_text(manifest.read_text().replace('always_in_context: true', f'when_to_use: A trigger{suffix}')
                            .replace(f'Former instruction body.{suffix}', 'Former instruction body.'), encoding='utf-8')
    target = root / 'skills' / 'ouroboroshub' / name
    target.parent.mkdir(parents=True, exist_ok=True)
    old.skill_dir.rename(target)
    row = {'id': 'fixture-git', 'kind': 'git', 'name': 'Fixture source', 'trust': trust}
    write_install_record(root, name, {'source': row, 'basis': 'trusted', 'entry': name,
                                      'release': {'version': '1', 'digest': compute_content_hash(target)}})
    save_enabled(root, name, True, actor='owner_ui')
    release = tmp_path / 'cached-release' / name
    release.mkdir(parents=True)
    prev_find, prev_payload = installer.find_item, installer.entry_payload_dir

    def find_item(drive, item_id, **kw):
        if item_id != f'fixture-git/{name}':
            return prev_find(drive, item_id, **kw)
        return row, {'name': name, 'id': item_id, 'version': '2', 'digest': tree_release_digest(release)}, {}

    def entry_payload_dir(drive, row, snapshot, item_name, **kw):
        return release if item_name == name else prev_payload(drive, row, snapshot, item_name, **kw)

    monkeypatch.setattr(installer, 'find_item', find_item)
    monkeypatch.setattr(installer, 'entry_payload_dir', entry_payload_dir)

    def offer(body, *, trigger=None):
        header = (target / 'SKILL.md').read_text().split('Former instruction body.')[0]
        if trigger is not None:
            header = header.replace(f'when_to_use: A trigger{suffix}', f'when_to_use: {trigger}')
        (release / 'SKILL.md').write_text(header + body + '\n', encoding='utf-8')
        return tree_release_digest(release)

    def update(digest):
        return installer.update_installed(root, name, mode='replace', digest=digest, rows=[row], repo_dir=config.REPO_DIR)

    return SimpleNamespace(target=target, offer=offer, update=update, name=name, record=lambda: read_install_record(root, name))


@pytest.mark.parametrize('trust', ['trusted', 'review'])
def test_instruction_update_checks_candidate_before_replacing_active_bytes(installation, monkeypatch, tmp_path, trust):
    from ouroboros.skill_catalogue import always_in_context_section, discover_context_skills
    from ouroboros.skill_source_install import SkillInstallError

    monkeypatch.setenv('OUROBOROS_RUNTIME_MODE', 'advanced')
    memory, ctx, _settings, live = installation
    root = memory.drive_root
    catalog = _catalog_update(monkeypatch, root, tmp_path, trust=trust)
    live['b'] = 36_308  # one weighted item can fit beside 20k and 15k
    assert 'RESOURCE_CONSTRAINT_BLOCKED' not in _identity(ctx, memory, 'Qualify the baseline background.')
    assert read_root_sample(root, provider='local')['status'] == 'protecting'
    old_record = catalog.record()
    old_bytes = (catalog.target / 'SKILL.md').read_bytes()
    digest = catalog.offer(f'New instruction body. {WEIGHT} {WEIGHT}')
    with pytest.raises(SkillInstallError) as refusal:
        catalog.update(digest)
    assert refusal.value.code == 'nano_publication_over_capacity'
    assert refusal.value.facts['candidate']['digest'] == digest
    assert (catalog.target / 'SKILL.md').read_bytes() == old_bytes
    assert catalog.record() == old_record and load_enabled(root, catalog.name)
    assert read_root_sample(root, provider='local')['status'] == 'protecting'
    assert any('New instruction body.' in row and 'Former instruction body.' not in row for row in live['externals'])
    digest = catalog.offer(f'Accepted instruction body. {WEIGHT}')
    assert catalog.update(digest)['ok']
    assert catalog.record()['release']['digest'] == digest
    active = always_in_context_section(discover_context_skills(root))
    assert ('Accepted instruction body.' in active) is (trust == 'trusted')
    assert read_root_sample(root, provider='local')['status'] == 'not_claimed'


@pytest.mark.parametrize('fits_at_admission', [True, False])
def test_pending_instruction_review_rechecks_before_projecting_its_result(
        installation, monkeypatch, tmp_path, fits_at_admission):
    from ouroboros.skill_catalogue import always_in_context_section, discover_context_skills
    from ouroboros.skill_install_record import settle_pending_update
    from ouroboros.skill_loader import find_skill
    from ouroboros.skill_review import SkillReviewOutcome, _persist_projection

    monkeypatch.setenv('OUROBOROS_RUNTIME_MODE', 'advanced')
    memory, ctx, _settings, live = installation
    root = memory.drive_root
    catalog = _catalog_update(monkeypatch, root, tmp_path, trust='review')
    live['b'] = 36_308
    working = ((catalog.target / 'SKILL.md').read_bytes(), (root / 'state' / 'skills' / catalog.name / 'review.json').read_bytes(),
               always_in_context_section(discover_context_skills(root)))
    digest = catalog.offer(f'Reviewed instruction body. {WEIGHT}')
    assert catalog.update(digest)['needs_review']
    assert not always_in_context_section(discover_context_skills(root))
    live['b'] += 0 if fits_at_admission else 500
    outcome = SkillReviewOutcome(skill_name=catalog.name, status='pass', content_hash=digest, record_id='fixture-review-record')
    _persist_projection(ctx, find_skill(root, catalog.name), root, outcome)
    assert outcome.status == 'pass' and outcome.record_id == 'fixture-review-record'
    before_settle = always_in_context_section(discover_context_skills(root))
    assert ('Reviewed instruction body.' in before_settle) is fits_at_admission
    settled = settle_pending_update(root, catalog.name, expected_digest=digest)
    active = always_in_context_section(discover_context_skills(root))
    if fits_at_admission:
        assert not outcome.error and settled['status'] == 'accepted'
        assert 'Reviewed instruction body.' in active
    else:
        assert 'not activated (over_capacity)' in outcome.error
        assert settled == {'status': 'restored', 'reason': 'nano_publication_over_capacity'}
        rejected = catalog.record()['update_rejected']
        # The review finished: the update is restored for capacity, not as an unfinished review.
        assert (rejected['digest'], rejected['reason']) == (digest, 'nano_publication_over_capacity')
        assert 'Former instruction body.' in active
        assert working == ((catalog.target / 'SKILL.md').read_bytes(),
                           (root / 'state' / 'skills' / catalog.name / 'review.json').read_bytes(), active)


def test_review_auto_enable_uses_the_count_without_changing_the_review(installation):
    from ouroboros.nano_activation import publish_review_auto_enable
    from ouroboros.nano_publication import NanoPublicationRefused
    from ouroboros.skill_catalogue import always_in_context_section, discover_context_skills
    from ouroboros.skill_loader import find_skill

    memory, _ctx, _settings, live = installation
    root = memory.drive_root
    skill = _install_instruction(root, 'auto-guide', f'Automatic instruction body. {WEIGHT}')
    state = root / 'state' / 'skills' / skill.name
    original_review = (state / 'review.json').read_bytes()
    with pytest.raises(NanoPublicationRefused, match='over_capacity'):
        # The background with no instruction already owns all reserved space.
        publish_review_auto_enable(root, skill)
    assert not (state / 'enabled.json').exists()
    assert not always_in_context_section(discover_context_skills(root))
    live['b'] -= 500
    assert publish_review_auto_enable(root, find_skill(root, skill.name))
    assert json.loads((state / 'enabled.json').read_text())['actor'] == 'review_auto_enable'
    assert 'Automatic instruction body.' in always_in_context_section(discover_context_skills(root))
    assert (state / 'review.json').read_bytes() == original_review


def _qualify(ctx, memory):
    assert 'RESOURCE_CONSTRAINT_BLOCKED' not in _identity(ctx, memory, 'Qualify the baseline background.')


def _edit_body(skill_dir, text):
    path = skill_dir / 'SKILL.md'
    header = path.read_text(encoding='utf-8').rsplit('---\n', 1)[0]
    path.write_text(f'{header}---\n{text}\n', encoding='utf-8')
    return compute_content_hash(skill_dir)


def _edited_enabled_instruction(installation, name):
    """On, once admitted, then edited in place: not acting until the new bytes are accepted."""
    from ouroboros.skill_loader import save_enabled

    memory, ctx, _settings, _live = installation
    skill = _install_instruction(memory.drive_root, name, 'Short original body.')
    save_enabled(memory.drive_root, name, True, actor='owner_ui')
    _qualify(ctx, memory)
    digest = _edit_body(skill.skill_dir, f'Corrected instruction body. {WEIGHT}')
    assert always_in_context_section_text(memory.drive_root) == ''
    return skill.skill_dir, digest


def _verdict(root, ctx, name, digest, status, record):
    from ouroboros.skill_loader import find_skill
    from ouroboros.skill_review import SkillReviewOutcome, _persist_projection

    outcome = SkillReviewOutcome(skill_name=name, status=status, content_hash=digest, record_id=record)
    _persist_projection(ctx, find_skill(root, name), root, outcome)
    return outcome


def test_verdict_activating_corrected_files_is_held_and_a_later_one_on_acting_files_is_only_fenced(installation):
    from ouroboros.nano_publication import PUBLICATION_REVISION
    from ouroboros.skill_loader import find_skill

    memory, ctx, _settings, live = installation
    root = memory.drive_root
    _skill_dir, digest = _edited_enabled_instruction(installation, 'edited-guide')
    review = root / 'state' / 'skills' / 'edited-guide' / 'review.json'
    earlier = review.read_bytes()
    outcome = _verdict(root, ctx, 'edited-guide', digest, 'pass', 'fixture-held')
    assert 'not activated (over_capacity)' in outcome.error and 'fixture-held' in outcome.error
    assert review.read_bytes() == earlier and always_in_context_section_text(root) == ''
    live['b'] -= 500
    assert not _verdict(root, ctx, 'edited-guide', digest, 'pass', 'fixture-fits').error
    assert 'Corrected instruction body.' in always_in_context_section_text(root)
    # A critic's verdict about files already acting is saved without a count, behind the fence.
    live['externals'].clear()
    marker = (root / PUBLICATION_REVISION).read_bytes()
    assert not _verdict(root, ctx, 'edited-guide', digest, 'warnings', 'fixture-second').error
    assert not live['externals'] and (root / PUBLICATION_REVISION).read_bytes() != marker
    assert find_skill(root, 'edited-guide').review.record_id == 'fixture-second'


def test_owner_edit_that_does_not_fit_stays_on_disk_without_acceptance(installation):
    from ouroboros.skill_acceptance import record_owner_edit

    memory, _ctx, _settings, live = installation
    root = memory.drive_root
    skill_dir, digest = _edited_enabled_instruction(installation, 'owner-guide')
    assert record_owner_edit(root, skill_dir, expected_digest=digest) is None
    assert not (root / 'state' / 'skills' / 'owner-guide' / 'owner_edit.json').exists()
    # Not an atomic update: the edit tool already replaced the bytes; only their acceptance waits.
    assert 'Corrected instruction body.' in (skill_dir / 'SKILL.md').read_text(encoding='utf-8')
    assert always_in_context_section_text(root) == ''
    live['b'] -= 500
    assert record_owner_edit(root, skill_dir, expected_digest=digest) == digest
    assert 'Corrected instruction body.' in always_in_context_section_text(root)


def test_release_repairing_a_broken_enabled_instruction_is_counted(installation, monkeypatch, tmp_path):
    from ouroboros.skill_source_install import SkillInstallError

    monkeypatch.setenv('OUROBOROS_RUNTIME_MODE', 'advanced')
    memory, ctx, _settings, live = installation
    root = memory.drive_root
    catalog = _catalog_update(monkeypatch, root, tmp_path, trust='trusted')
    digest = catalog.offer(f'Repaired instruction body. {WEIGHT}')
    (catalog.target / 'SKILL.md').write_bytes(b'\xff')  # unreadable: reads as Off, its switch stays saved On
    _qualify(ctx, memory)
    with pytest.raises(SkillInstallError) as refusal:
        catalog.update(digest)
    assert refusal.value.code == 'nano_publication_over_capacity'
    assert (catalog.target / 'SKILL.md').read_bytes() == b'\xff' and load_enabled(root, catalog.name)
    live['b'] -= 500
    assert catalog.update(digest)['ok']
    assert 'Repaired instruction body.' in always_in_context_section_text(root)


def test_reinstalling_a_name_whose_switch_stays_on_is_counted(installation, monkeypatch, tmp_path):
    import shutil

    from ouroboros.skill_install_record import clear_install_record
    from ouroboros.skill_source_install import SkillInstallError

    monkeypatch.setenv('OUROBOROS_RUNTIME_MODE', 'advanced')
    memory, ctx, _settings, live = installation
    root = memory.drive_root
    catalog = _cross_source_replace(monkeypatch, root, tmp_path, trust='trusted')
    digest = catalog.offer(f'Returning instruction body. {WEIGHT}')
    shutil.rmtree(catalog.old_dir)  # removed, with the saved switch kept as uninstall keeps it
    clear_install_record(root, catalog.name)
    _qualify(ctx, memory)
    with pytest.raises(SkillInstallError) as refusal:
        catalog.install(digest)
    assert refusal.value.code == 'nano_publication_over_capacity' and 'was not installed' in str(refusal.value)
    assert not catalog.target.exists() and catalog.record() is None
    live['b'] -= 500
    assert catalog.install(digest)['ok']
    assert 'Returning instruction body.' in always_in_context_section_text(root)


def test_new_off_skill_is_counted_by_its_rendered_line(installation, monkeypatch, tmp_path):
    import shutil

    from ouroboros.skill_install_record import clear_install_record

    monkeypatch.setenv('OUROBOROS_RUNTIME_MODE', 'advanced')
    memory, ctx, _settings, live = installation
    root = memory.drive_root
    catalog = _cross_source_replace(monkeypatch, root, tmp_path, trust='review', enabled=False)
    digest = catalog.offer('Newly offered body.')
    shutil.rmtree(catalog.old_dir)
    clear_install_record(root, catalog.name)
    _qualify(ctx, memory)
    live['externals'].clear()
    assert catalog.install(digest)['needs_review']  # absent to Off: the name joins the Off list
    assert any('Installed, Off' in wire and catalog.name in wire for wire in live['externals'])


def test_pending_release_counts_what_a_fresh_main_reads_now_and_after_admission(installation, monkeypatch, tmp_path):
    monkeypatch.setenv('OUROBOROS_RUNTIME_MODE', 'advanced')
    memory, ctx, _settings, live = installation
    root = memory.drive_root
    catalog = _catalog_update(monkeypatch, root, tmp_path, trust='review')
    live['b'] = 36_308
    digest = catalog.offer('Awaiting instruction body.')
    _qualify(ctx, memory)
    live['externals'].clear()
    assert catalog.update(digest)['needs_review']
    assert any('Enabled, not ready' in wire for wire in live['externals'])
    assert any('Awaiting instruction body.' in wire for wire in live['externals'])


def test_kept_rollback_reserves_an_ordinary_catalogue_row(installation, monkeypatch, tmp_path):
    from ouroboros.skill_install_record import settle_pending_update, write_install_record
    from ouroboros.skill_source_catalog import tree_release_digest

    monkeypatch.setenv('OUROBOROS_RUNTIME_MODE', 'advanced')
    memory, ctx, _settings, _live = installation
    root = memory.drive_root
    catalog = _catalog_update(monkeypatch, root, tmp_path, trust='review')

    def manifest(trigger, body):
        return (f'---\nname: {catalog.name}\ndescription: A synthetic guide\ntype: instruction\n'
                f'when_to_use: {trigger}\n---\n{body}\n')

    # Not always in context: only the card's trigger line carries weight.
    (catalog.target / 'SKILL.md').write_text(manifest(f'Old trigger {WEIGHT}', 'Old body.'), encoding='utf-8')
    old = compute_content_hash(catalog.target)
    write_install_record(root, catalog.name, {**catalog.record(), 'release': {'version': '1', 'digest': old}})
    atomic_write_json(root / 'state' / 'skills' / catalog.name / 'review.json', {'status': 'pass', 'content_hash': old})
    release = tmp_path / 'cached-release' / catalog.name
    (release / 'SKILL.md').write_text(manifest('A light trigger', 'New body.'), encoding='utf-8')
    at_limit = f'Synthetic identity {WEIGHT * 39}'
    assert 'RESOURCE_CONSTRAINT_BLOCKED' not in _identity(ctx, memory, at_limit)
    digest = tree_release_digest(release)
    assert catalog.update(digest)['needs_review']
    blocked = _identity(ctx, memory, at_limit + WEIGHT)
    assert 'RESOURCE_CONSTRAINT_BLOCKED' in blocked and 'over_capacity' in blocked
    assert settle_pending_update(root, catalog.name, expected_digest=digest)['status'] == 'restored'
    assert measure_publication_floor(root)['minimum_work_headroom_at_reply_floor_tokens'] == 15_000


def test_refused_landing_loads_again_what_it_unloaded(installation, monkeypatch, tmp_path):
    from ouroboros.skill_source_install import SkillInstallError

    monkeypatch.setenv('OUROBOROS_RUNTIME_MODE', 'advanced')
    memory, ctx, _settings, live = installation
    root = memory.drive_root
    catalog = _cross_source_replace(monkeypatch, root, tmp_path, trust='trusted')
    live['b'] = 36_308
    digest = catalog.offer('Replacement that goes stale.')
    _qualify(ctx, memory)
    reloaded = []
    from ouroboros.marketplace.install import snapshot_payload_state as real_snapshot

    def changed_switch(drive, name, target, **kwargs):
        from ouroboros.skill_loader import save_enabled

        save_enabled(root, name, False, actor='owner_ui')
        return real_snapshot(drive, name, target, **kwargs)

    monkeypatch.setattr('ouroboros.skill_source_install._unload', lambda *_a: True)  # as a live extension answers
    monkeypatch.setattr('ouroboros.skill_source_install._reload_previous', lambda _d, name: reloaded.append(name))
    monkeypatch.setattr('ouroboros.marketplace.install.snapshot_payload_state', changed_switch)
    with pytest.raises(SkillInstallError) as refusal:
        catalog.install(digest)
    assert refusal.value.code == 'nano_publication_stale' and reloaded == [catalog.name]
    assert not catalog.target.exists() and catalog.old_dir.exists()


def test_switch_refusals_name_their_cause_and_only_capacity_speaks_of_size(installation, monkeypatch):
    memory, ctx, settings, live = installation
    root = memory.drive_root
    skill = _install_instruction(root, 'message-guide', 'A small body.')
    live['b'] = 36_308
    _qualify(ctx, memory)
    monkeypatch.setattr(llm_local, 'measurement_slot', lambda *a, **kw: nullcontext(False))
    busy = _toggle(skill, root, str(config.REPO_DIR), True, 'owner_ui', 'fixture')
    assert busy['code'] == 'nano_publication_busy' and 'does not fit' not in busy['error']
    atomic_write_json(config.SETTINGS_PATH, {**settings, 'OUROBOROS_CONTEXT_MODE': 'max'})

    def unwritable(*_a, **_kw):
        raise OSError('synthetic disk fault')

    monkeypatch.setattr('ouroboros.skill_lifecycle_actions.save_enabled', unwritable)
    failed = _toggle(skill, root, str(config.REPO_DIR), True, 'owner_ui', 'fixture')
    assert (failed['code'], failed['enabled']) == ('switch_io_failed', False)
    assert 'Nano' not in failed['error'] and 'synthetic disk fault' in failed['detail']


def test_forms_of_a_pending_release_are_counted_on_one_base(installation, monkeypatch, tmp_path):
    from ouroboros.nano_publication import NanoPublication, publication_fence

    monkeypatch.setenv('OUROBOROS_RUNTIME_MODE', 'advanced')
    memory, ctx, _settings, live = installation
    root = memory.drive_root
    catalog = _catalog_update(monkeypatch, root, tmp_path, trust='review')
    live['b'] = 36_308
    _qualify(ctx, memory)
    digest = catalog.offer('Awaiting instruction body.')
    tickets, real = [], NanoPublication.prepare

    def recording(self, changes):
        tickets.append(real(self, changes))
        return tickets[-1]

    def another_writer():
        live['on_count'] = None
        with publication_fence(root):  # moves the base between the two forms' counts
            pass

    monkeypatch.setattr(NanoPublication, 'prepare', recording)
    live['on_count'] = another_writer
    assert catalog.update(digest)['needs_review']
    # Not-ready form and admitted form, the base moved, both again: the committed base saw both.
    assert len(tickets) == 4 and tickets[3].witness is not None


@pytest.mark.parametrize('form', ['single', 'batch'])
def test_direct_repair_text_edit_is_refused_before_any_byte_changes(installation, monkeypatch, form):
    from ouroboros.contracts.task_constraint import normalize_task_constraint
    from ouroboros.skill_acceptance import OWNER_EDIT_FILENAME
    from ouroboros.skill_loader import save_enabled
    from ouroboros.skill_repair_admission import load_repair_admission, record_repair_admission
    from ouroboros.tools.registry import ToolRegistry

    memory, ctx, _settings, live = installation
    root = memory.drive_root
    monkeypatch.setenv('OUROBOROS_RUNTIME_MODE', 'advanced')
    monkeypatch.setenv('OUROBOROS_DATA_DIR', str(root))
    monkeypatch.setattr('ouroboros.safety.check_safety', lambda *_a, **_kw: (True, ''))
    monkeypatch.setattr('ouroboros.tools.review_helpers.cyber_review_authority', lambda: False)
    skill = _install_instruction(root, 'repair-guide', f'Heavy repaired body. {WEIGHT}')
    notes = skill.skill_dir / 'notes.md'
    notes.write_text('note\n', encoding='utf-8')  # after its review: On, stale, not acting
    save_enabled(root, skill.name, True, actor='owner_ui')
    digest = compute_content_hash(skill.skill_dir)
    assert always_in_context_section_text(root) == ''
    record_repair_admission(root, skill.name, task_id='t1', owner_request=True, base_content_hash=digest)
    registry = ToolRegistry(repo_dir=ctx.repo_dir, drive_root=root)
    registry._ctx.task_id = 't1'
    registry._ctx.task_constraint = normalize_task_constraint(
        {'mode': 'skill_repair', 'skill_name': skill.name, 'payload_root': 'skills/external/repair-guide'})

    def edit():
        args = {'path': 'notes.md', 'old_str': 'note', 'new_str': 'fixed note'}
        if form == 'batch':
            return registry.execute('edit_batch', {'root': 'skill_payload', 'edits': [args]})
        return registry.execute('edit_text', {'root': 'skill_payload', **args})

    refused = edit()
    assert 'SKILL_REPAIR_NANO_OVER_CAPACITY' in refused and 'Nothing was written' in refused, refused
    assert notes.read_text(encoding='utf-8') == 'note\n'
    assert not (root / 'state' / 'skills' / skill.name / OWNER_EDIT_FILENAME).exists()
    assert load_repair_admission(root, skill.name)['expected_content_hash'] == digest
    assert always_in_context_section_text(root) == ''
    live['b'] -= 500
    accepted = edit()
    assert 'Replaced' in accepted or ' applied 1 edit(s)' in accepted, accepted
    assert (root / 'state' / 'skills' / skill.name / OWNER_EDIT_FILENAME).exists()
    assert 'Heavy repaired body.' in always_in_context_section_text(root)


def _acting_repair(installation, monkeypatch, name):
    """An On skill whose review passed over every file, with a Repair admitted against it: it acts now."""
    from ouroboros.contracts.task_constraint import normalize_task_constraint
    from ouroboros.skill_loader import save_enabled
    from ouroboros.skill_repair_admission import record_repair_admission
    from ouroboros.tools.registry import ToolRegistry

    memory, ctx, _settings, live = installation
    root = memory.drive_root
    monkeypatch.setenv('OUROBOROS_RUNTIME_MODE', 'advanced')
    monkeypatch.setenv('OUROBOROS_DATA_DIR', str(root))
    monkeypatch.setattr('ouroboros.safety.check_safety', lambda *_a, **_kw: (True, ''))
    monkeypatch.setattr('ouroboros.tools.review_helpers.cyber_review_authority', lambda: False)
    live['b'] = 36_308  # an acting body of one weighted item fits
    skill = _install_instruction(root, name, f'Acting body. {WEIGHT}')
    notes = skill.skill_dir / 'notes.md'
    notes.write_text('note\n', encoding='utf-8')
    digest = compute_content_hash(skill.skill_dir)
    atomic_write_json(root / 'state' / 'skills' / name / 'review.json', {'status': 'pass', 'content_hash': digest})
    save_enabled(root, name, True, actor='owner_ui')
    record_repair_admission(root, name, task_id='t1', owner_request=True, base_content_hash=digest)
    registry = ToolRegistry(repo_dir=ctx.repo_dir, drive_root=root)
    registry._ctx.task_id = 't1'
    registry._ctx.task_constraint = normalize_task_constraint(
        {'mode': 'skill_repair', 'skill_name': name, 'payload_root': f'skills/external/{name}'})
    _qualify(ctx, memory)
    assert 'Acting body.' in always_in_context_section_text(root)
    return registry, skill.skill_dir, notes, digest


@pytest.mark.parametrize('form', ['single', 'batch'])
def test_direct_repair_edit_is_refused_when_the_base_moves_before_the_write(installation, monkeypatch, form):
    from ouroboros import nano_activation
    from ouroboros.nano_publication import publication_fence
    from ouroboros.skill_acceptance import OWNER_EDIT_FILENAME
    from ouroboros.skill_repair_admission import load_repair_admission

    memory, _ctx, _settings, live = installation
    root = memory.drive_root
    registry, _skill_dir, notes, digest = _acting_repair(installation, monkeypatch, 'moving-guide')
    # Force the counted path: a note file alone does not change what a fresh Main reads (the stale fixture above
    # is the case where it does); the point here is the interval between the count and the write.
    monkeypatch.setattr(nano_activation, '_reads_the_same', lambda *_a: False)
    state = root / 'state' / 'skills' / 'moving-guide'
    working = tuple((state / name).read_bytes() for name in ('review.json', 'enabled.json'))

    def edit():
        args = {'path': 'notes.md', 'old_str': 'note', 'new_str': 'fixed note'}
        if form == 'batch':
            return registry.execute('edit_batch', {'root': 'skill_payload', 'edits': [args]})
        return registry.execute('edit_text', {'root': 'skill_payload', **args})

    def another_writer():
        live['on_count'] = None
        with publication_fence(root):  # an ordinary writer moves the generation after the count, before the write
            pass

    live['on_count'] = another_writer
    refused = edit()
    assert 'SKILL_REPAIR_NANO_STALE' in refused and 'Nothing was written' in refused, refused
    assert notes.read_text(encoding='utf-8') == 'note\n'
    assert not (state / OWNER_EDIT_FILENAME).exists()
    assert working == tuple((state / name).read_bytes() for name in ('review.json', 'enabled.json'))
    assert load_repair_admission(root, 'moving-guide')['expected_content_hash'] == digest
    assert 'Acting body.' in always_in_context_section_text(root)  # the working edition was never lost
    accepted = edit()  # the base now holds: counted once, written, recorded without a second count
    assert 'Replaced' in accepted or ' applied 1 edit(s)' in accepted, accepted
    assert notes.read_text(encoding='utf-8') == 'fixed note\n'
    assert (state / OWNER_EDIT_FILENAME).exists() and 'Acting body.' in always_in_context_section_text(root)


def test_external_edit_inside_a_held_interval_is_never_accepted(installation, monkeypatch):
    from ouroboros import nano_activation
    from ouroboros.skill_acceptance import OWNER_EDIT_FILENAME, record_owner_edit
    from ouroboros.skill_identity import text_edit_release_hashes
    from ouroboros.skill_loader import find_skill

    memory, _ctx, _settings, _live = installation
    root = memory.drive_root
    _registry, skill_dir, notes, _digest = _acting_repair(installation, monkeypatch, 'external-guide')
    monkeypatch.setattr(nano_activation, '_reads_the_same', lambda *_a: False)
    loaded = find_skill(root, 'external-guide')
    predicted = text_edit_release_hashes(loaded, [(notes, 'note\n', 'fixed note\n')])[1]
    refusal, hold = nano_activation.edit_admission(root, loaded, predicted)
    assert refusal is None and hold is not None
    with hold():
        notes.write_text('another program\n', encoding='utf-8')  # not the predicted bytes
        # The capacity evidence is not a right to act: the digest is verified as before, and fails.
        assert record_owner_edit(root, skill_dir, expected_digest=predicted) is None
    assert not (root / 'state' / 'skills' / 'external-guide' / OWNER_EDIT_FILENAME).exists()
    # Edits by other programs are not followed or undone: the edited files simply stop acting.
    assert always_in_context_section_text(root) == ''


def test_enabling_a_large_instruction_checks_the_non_memory_background(installation):
    memory, ctx, _settings, live = installation
    assert 'RESOURCE_CONSTRAINT_BLOCKED' not in _identity(ctx, memory, 'Near-boundary identity ' + WEIGHT * 40)
    root = memory.drive_root
    large = _install_instruction(root, 'synthetic-guide', 'A bounded skill body ' + WEIGHT)
    result = _toggle(large, root, str(config.REPO_DIR), True, 'owner_ui', 'synthetic fixture')
    assert result['code'] == 'nano_publication_over_capacity', result
    assert not load_enabled(root, large.name)
    assert any('A bounded skill body' in wire and '## Always-in-context skills' in wire for wire in live['externals'])
    assert 'RESOURCE_CONSTRAINT_BLOCKED' not in _identity(ctx, memory, 'Shorter synthetic identity.')
    # Shrinking the current memory does not free space in the background
    # budget: it is still reserved for memory to grow back to its ceiling.
    assert _toggle(large, root, str(config.REPO_DIR), True, 'owner_ui', 'synthetic fixture')['code'] == (
        'nano_publication_over_capacity')
    live['b'] -= 500
    assert _toggle(large, root, str(config.REPO_DIR), True, 'owner_ui', 'synthetic fixture')['ok']
    assert load_enabled(root, large.name)
    live['externals'].clear()
    assert _toggle(large, root, str(config.REPO_DIR), False, 'owner_ui', 'synthetic fixture')['ok']
    assert not load_enabled(root, large.name) and not live['externals']


def _cross_source_replace(monkeypatch, root, tmp_path, *, trust, enabled=True):
    """An always-in-context instruction installed from ClawHub, replaced by a Git release."""
    from ouroboros import skill_source_install as installer
    from ouroboros.skill_install_record import read_install_record, write_install_record
    from ouroboros.skill_loader import save_enabled
    from ouroboros.skill_source_catalog import tree_release_digest

    name = 'fixture-guide'
    old = _install_instruction(root, name, 'Registry instruction body.')
    old_dir = root / 'skills' / 'clawhub' / name
    old_dir.parent.mkdir(parents=True)
    old.skill_dir.rename(old_dir)
    write_install_record(root, name, {'source': {'id': 'clawhub', 'kind': 'registry', 'name': 'ClawHub'},
                                      'basis': 'review', 'entry': name,
                                      'release': {'version': '1', 'digest': compute_content_hash(old_dir)}})
    if enabled:
        save_enabled(root, name, True, actor='owner_ui')
    row = {'id': 'fixture-git', 'kind': 'git', 'name': 'Fixture source', 'trust': trust}
    release = tmp_path / 'cached-release' / name
    release.mkdir(parents=True)
    monkeypatch.setattr('ouroboros.skill_source_install.find_item', lambda *_a, **_kw: (
        row, {'name': name, 'id': f'fixture-git/{name}', 'version': '2', 'digest': tree_release_digest(release)}, {}))
    monkeypatch.setattr('ouroboros.skill_source_install.entry_payload_dir', lambda *_a, **_kw: release)

    def offer(body):
        (release / 'SKILL.md').write_text((old_dir / 'SKILL.md').read_text().split('Registry instruction body.')[0]
                                           + body + '\n', encoding='utf-8')
        return tree_release_digest(release)

    def install(digest):
        return installer.install_item(root, f'fixture-git/{name}', replace=True, rows=[row], repo_dir=config.REPO_DIR)

    return SimpleNamespace(old_dir=old_dir, target=root / 'skills' / 'ouroboroshub' / name, offer=offer,
                           install=install, name=name, record=lambda: read_install_record(root, name))


@pytest.mark.parametrize('trust', ['trusted', 'review'])
def test_replacing_an_enabled_instruction_from_another_source_counts_the_release(
        installation, monkeypatch, tmp_path, trust):
    from ouroboros.skill_source_install import SkillInstallError

    monkeypatch.setenv('OUROBOROS_RUNTIME_MODE', 'advanced')
    memory, ctx, _settings, live = installation
    root = memory.drive_root
    catalog = _cross_source_replace(monkeypatch, root, tmp_path, trust=trust)
    live['b'] = 36_308  # one weighted item fits beside 20k memory and 15k work
    assert 'RESOURCE_CONSTRAINT_BLOCKED' not in _identity(ctx, memory, 'Qualify the baseline background.')
    assert read_root_sample(root, provider='local')['status'] == 'protecting'
    old_record = catalog.record()
    old_bytes = (catalog.old_dir / 'SKILL.md').read_bytes()
    # A candidate that does not fit is refused before the occupant's bytes move.
    digest = catalog.offer(f'Unfittable replacement body. {WEIGHT} {WEIGHT}')
    with pytest.raises(SkillInstallError) as refusal:
        catalog.install(digest)
    assert refusal.value.code == 'nano_publication_over_capacity'
    assert (catalog.old_dir / 'SKILL.md').read_bytes() == old_bytes and not catalog.target.exists()
    assert catalog.record() == old_record and load_enabled(root, catalog.name)
    assert read_root_sample(root, provider='local')['status'] == 'protecting'
    assert any('Unfittable replacement body.' in wire and 'Registry instruction body.' not in wire
               for wire in live['externals'])
    # A fitting candidate is counted, the old directory is replaced by the target, the
    # counted view renders the target bucket, and the landing record names the new source.
    live['externals'].clear()
    digest = catalog.offer(f'Replacement instruction body. {WEIGHT}')
    assert catalog.install(digest)['ok']
    assert 'Replacement instruction body.' in (catalog.target / 'SKILL.md').read_text()
    assert not catalog.old_dir.exists()
    assert catalog.record()['source']['id'] == 'fixture-git'
    assert catalog.record()['release']['digest'] == digest
    active = always_in_context_section_text(root)
    if trust == 'trusted':
        assert 'Replacement instruction body.' in active
        assert "bucket='ouroboroshub'" in active and "bucket='clawhub'" not in active
    else:
        assert active == ""
    assert read_root_sample(root, provider='local')['status'] == 'not_claimed'
    assert any('Replacement instruction body.' in wire for wire in live['externals'])


def test_off_instruction_replaced_while_an_ordinary_enable_takes_the_place_refuses_stale(
        installation, monkeypatch, tmp_path):
    from ouroboros.marketplace.install import snapshot_payload_state as real_snapshot
    from ouroboros.skill_loader import find_skill
    from ouroboros.skill_source_install import SkillInstallError

    monkeypatch.setenv('OUROBOROS_RUNTIME_MODE', 'advanced')
    memory, ctx, _settings, live = installation
    root = memory.drive_root
    catalog = _cross_source_replace(monkeypatch, root, tmp_path, trust='trusted', enabled=False)
    live['b'] = 36_308
    assert 'RESOURCE_CONSTRAINT_BLOCKED' not in _identity(ctx, memory, 'Qualify the baseline background.')

    def racing_snapshot(drive, name, target, **kwargs):
        # An ordinary counted enable of the off occupant commits inside the landing's window.
        live['on_count'] = None
        assert _toggle(find_skill(root, catalog.name), root, str(config.REPO_DIR), True, 'owner_ui',
                       'race fixture')['ok']
        return real_snapshot(drive, name, target, **kwargs)

    monkeypatch.setattr('ouroboros.marketplace.install.snapshot_payload_state', racing_snapshot)
    digest = catalog.offer(f'Raced replacement body. {WEIGHT}')
    with pytest.raises(SkillInstallError) as refusal:
        catalog.install(digest)
    assert refusal.value.code == 'nano_publication_stale'
    assert 'Raced replacement body.' not in (catalog.old_dir / 'SKILL.md').read_text()
    assert not catalog.target.exists() and load_enabled(root, catalog.name)
    assert json.loads((root / 'state' / 'skills' / catalog.name / 'enabled.json').read_text())['actor'] == 'owner_ui'
    assert 'Registry instruction body.' in always_in_context_section_text(root)


def always_in_context_section_text(root):
    from ouroboros.skill_catalogue import always_in_context_section, discover_context_skills

    return always_in_context_section(discover_context_skills(root))


def test_replace_without_admission_lands_without_counting_or_refusal(installation, monkeypatch, tmp_path):
    monkeypatch.setenv('OUROBOROS_RUNTIME_MODE', 'advanced')
    memory, ctx, _settings, live = installation
    root = memory.drive_root
    catalog = _cross_source_replace(monkeypatch, root, tmp_path, trust='trusted')
    live['externals'].clear()
    monkeypatch.setattr('ouroboros.nano_activation.admission_for', lambda *_a, **_kw: None)
    digest = catalog.offer(f'Unmeasured replacement body. {WEIGHT} {WEIGHT}')
    assert catalog.install(digest)['ok']
    assert 'Unmeasured replacement body.' in (catalog.target / 'SKILL.md').read_text()
    assert not catalog.old_dir.exists() and not live['externals']


def test_update_rollback_restores_the_previous_body_through_the_fence(installation, monkeypatch, tmp_path):
    from ouroboros.nano_publication import PUBLICATION_REVISION
    from ouroboros.skill_install_record import settle_pending_update, update_backup_path

    monkeypatch.setenv('OUROBOROS_RUNTIME_MODE', 'advanced')
    memory, ctx, _settings, live = installation
    root = memory.drive_root
    catalog = _catalog_update(monkeypatch, root, tmp_path, trust='review')
    live['b'] = 36_308
    review_before = (root / 'state' / 'skills' / catalog.name / 'review.json').read_bytes()
    marker = root / PUBLICATION_REVISION
    assert 'RESOURCE_CONSTRAINT_BLOCKED' not in _identity(ctx, memory, 'Qualify the baseline background.')
    digest = catalog.offer(f'Reviewed instruction body. {WEIGHT}')
    assert catalog.update(digest)['needs_review']
    marker_before = marker.read_bytes()
    assert (root / 'state' / 'skills' / catalog.name / 'review.json').read_bytes() == review_before
    settled = settle_pending_update(root, catalog.name, expected_digest=digest)
    assert settled['status'] == 'restored'
    # The restore returned the previously counted baseline body and the fence rotated the
    # publication revision, so no in-flight capture can commit on the pre-rollback witness.
    assert 'Former instruction body.' in always_in_context_section_text(root)
    assert marker.read_bytes() != marker_before
    assert (root / 'state' / 'skills' / catalog.name / 'review.json').read_bytes() == review_before
    rejected = catalog.record()['update_rejected']
    assert (rejected['digest'], rejected['version']) == (digest, '2')
    assert load_enabled(root, catalog.name) and not update_backup_path(root, catalog.name).exists()


@pytest.mark.parametrize('startup', [False, True], ids=['ordinary-settle', 'startup-recovery'])
def test_pending_rollback_reserves_room_across_another_memory_publication(installation, monkeypatch, tmp_path, startup):
    from ouroboros.skill_install_record import settle_pending_update, update_backup_path
    from ouroboros.skill_source_install import settle_pending_updates_at_startup

    monkeypatch.setenv('OUROBOROS_RUNTIME_MODE', 'advanced')
    memory, ctx, _settings, live = installation
    root = memory.drive_root
    catalog = _catalog_update(monkeypatch, root, tmp_path, trust='review', suffix=f' {WEIGHT}')
    original = (catalog.target / 'SKILL.md').read_bytes()
    old_review = (root / 'state' / 'skills' / catalog.name / 'review.json').read_bytes()
    at_limit = f'Synthetic identity {WEIGHT * 39}'
    assert 'RESOURCE_CONSTRAINT_BLOCKED' not in _identity(ctx, memory, at_limit)
    assert measure_publication_floor(root)['minimum_work_headroom_at_reply_floor_tokens'] == 15_000
    digest = catalog.offer('Short instruction waiting for independent review.')
    assert catalog.update(digest)['needs_review'] and update_backup_path(root, catalog.name).is_file()
    assert always_in_context_section_text(root) == ''  # neither version acts while pending
    assert 'RESOURCE_CONSTRAINT_BLOCKED' not in _identity(ctx, memory, at_limit + ' reworded')
    live['externals'].clear()
    blocked = _identity(ctx, memory, at_limit + WEIGHT)
    measured_rollback = any('Former instruction body.' in wire for wire in live['externals'])
    saved_before_rollback = memory.load_identity()
    live['externals'].clear()
    live['supports_count'] = False  # a mandatory rollback never needs a fresh counter
    if startup:
        settle_pending_updates_at_startup(root)
    else:
        assert settle_pending_update(root, catalog.name, expected_digest=digest)['status'] == 'restored'
    assert not live['externals'] and not update_backup_path(root, catalog.name).exists()
    assert (catalog.target / 'SKILL.md').read_bytes() == original
    assert (root / 'state' / 'skills' / catalog.name / 'review.json').read_bytes() == old_review
    assert catalog.record()['update_rejected']['digest'] == digest
    assert 'Former instruction body.' in always_in_context_section_text(root)
    live['supports_count'] = True
    headroom = measure_publication_floor(root)['minimum_work_headroom_at_reply_floor_tokens']
    # This assertion comes after the real restore so it fails on the old code
    # with the actually observed shortfall, not just a missed refusal.
    assert headroom == 15_000, (blocked, headroom)
    assert 'RESOURCE_CONSTRAINT_BLOCKED' in blocked and 'over_capacity' in blocked
    assert measured_rollback and saved_before_rollback.strip() == at_limit + ' reworded'


def test_review_settlement_releases_the_pending_rollback_capacity(installation, monkeypatch, tmp_path):
    from ouroboros.skill_install_record import settle_pending_update, update_backup_path
    from ouroboros.skill_loader import find_skill
    from ouroboros.skill_review import SkillReviewOutcome, _persist_projection

    monkeypatch.setenv('OUROBOROS_RUNTIME_MODE', 'advanced')
    memory, ctx, _settings, _live = installation
    root = memory.drive_root
    catalog = _catalog_update(monkeypatch, root, tmp_path, trust='review', suffix=f' {WEIGHT}')
    at_limit = f'Synthetic identity {WEIGHT * 39}'
    assert 'RESOURCE_CONSTRAINT_BLOCKED' not in _identity(ctx, memory, at_limit)
    digest = catalog.offer('Reviewed shorter instruction.')
    assert catalog.update(digest)['needs_review']
    review = SkillReviewOutcome(skill_name=catalog.name, status='pass', content_hash=digest, record_id='fixture-pass')
    _persist_projection(ctx, find_skill(root, catalog.name), root, review)
    assert not review.error and review.status == 'pass' and review.record_id == 'fixture-pass'
    assert update_backup_path(root, catalog.name).exists()
    assert 'Reviewed shorter instruction.' in always_in_context_section_text(root)
    assert 'over_capacity' in _identity(ctx, memory, at_limit + WEIGHT)
    assert settle_pending_update(root, catalog.name, expected_digest=digest)['status'] == 'accepted'
    assert not update_backup_path(root, catalog.name).exists()
    assert 'RESOURCE_CONSTRAINT_BLOCKED' not in _identity(ctx, memory, at_limit + WEIGHT)
    assert find_skill(root, catalog.name).review.record_id == 'fixture-pass'


@pytest.mark.parametrize('qualified', [True, False], ids=['qualified-nano', 'ordinary-profile'])
@pytest.mark.parametrize('resident', [True, False], ids=['resident-instructions', 'catalogue-only'])
def test_only_qualified_competing_replacement_waits_for_settlement(installation, monkeypatch, tmp_path, qualified, resident):
    from ouroboros.skill_install_record import settle_pending_update, update_backup_path
    from ouroboros.skill_loader import find_skill
    from ouroboros.skill_source_install import SkillInstallError

    monkeypatch.setenv('OUROBOROS_RUNTIME_MODE', 'advanced')
    memory, ctx, settings, live = installation
    root = memory.drive_root
    first = _catalog_update(monkeypatch, root, tmp_path, trust='review', name='first-guide', suffix=f' {WEIGHT}',
                            resident=resident)
    second = _catalog_update(monkeypatch, root, tmp_path, trust='review', name='second-guide', suffix=f' {WEIGHT}',
                             resident=resident)
    live['b'] = 35_808
    if qualified:
        assert 'RESOURCE_CONSTRAINT_BLOCKED' not in _identity(ctx, memory, 'Initial profile measurement.')
    else:
        atomic_write_json(config.SETTINGS_PATH, {**settings, 'OUROBOROS_CONTEXT_MODE': 'low'})
    d1, d2 = first.offer('First pending body.'), second.offer('Second pending body.')
    assert first.update(d1)['needs_review']
    if qualified:
        with pytest.raises(SkillInstallError) as exc:
            second.update(d2)
        assert exc.value.code == 'nano_publication_busy'
        assert not update_backup_path(root, second.name).exists()
        assert 'Former instruction body.' in (second.target / 'SKILL.md').read_text()
        # The same update for an inactive skill is not serialized.
        assert _toggle(find_skill(root, second.name), root, str(config.REPO_DIR), False, 'owner_ui', 'fixture')['ok']
    assert second.update(d2)['needs_review']
    if not qualified:
        assert not live['externals']
        # Two rollbacks from before qualification are not an exact bounded
        # witness. Neither startup nor an ordinary restore needs this witness.
        atomic_write_json(config.SETTINGS_PATH, settings)
        assert 'busy' in _identity(ctx, memory, 'Do not claim protection for overlapping rollback states.')
        assert read_root_sample(root, provider='local')['status'] == 'not_claimed'
    assert settle_pending_update(root, first.name, expected_digest=d1)['status'] == 'restored'
    assert settle_pending_update(root, second.name, expected_digest=d2)['status'] == 'restored'
    assert load_enabled(root, second.name) is (not qualified)


def test_mixed_legacy_rollbacks_are_not_proved_by_two_fitting_endpoints(installation, monkeypatch, tmp_path):
    from ouroboros.skill_install_record import settle_pending_update
    from ouroboros.skill_loader import find_skill
    from ouroboros.skill_review import SkillReviewOutcome, _persist_projection

    monkeypatch.setenv('OUROBOROS_RUNTIME_MODE', 'advanced')
    memory, ctx, settings, live = installation
    root = memory.drive_root
    first = _catalog_update(monkeypatch, root, tmp_path, trust='review', name='first-guide', suffix=f' {WEIGHT}',
                            resident=False)
    second = _catalog_update(monkeypatch, root, tmp_path, trust='review', name='second-guide', resident=False)
    live['b'] = 36_308
    atomic_write_json(config.SETTINGS_PATH, {**settings, 'OUROBOROS_CONTEXT_MODE': 'low'})
    first_digest = first.offer('First short candidate.', trigger='First light trigger')
    second_digest = second.offer('Second heavy candidate.', trigger=f'Second heavy trigger {WEIGHT}')
    for catalog, digest in ((first, first_digest), (second, second_digest)):
        assert catalog.update(digest)['needs_review']
        _persist_projection(ctx, find_skill(root, catalog.name), root, SkillReviewOutcome(
            skill_name=catalog.name, status='pass', content_hash=digest, record_id=f'{catalog.name}-pass'))
    at_limit = f'Synthetic identity {WEIGHT * 40}'
    _identity(ctx, memory, at_limit)
    atomic_write_json(config.SETTINGS_PATH, settings)
    assert measure_publication_floor(root)['minimum_work_headroom_at_reply_floor_tokens'] == 15_000
    refused = _identity(ctx, memory, at_limit + ' reworded')
    # The first update can return while the second stays accepted. Restores must not ask a counter.
    live['externals'].clear()
    assert settle_pending_update(root, first.name, expected_digest=first_digest, landing_failed=True)['status'] == 'restored'
    assert settle_pending_update(root, second.name, expected_digest=second_digest)['status'] == 'accepted'
    assert not live['externals']
    assert measure_publication_floor(root)['minimum_work_headroom_at_reply_floor_tokens'] == 14_500
    assert 'busy' in refused
    assert memory.load_identity().strip() == at_limit


def _install_catalogue_skill(root, name, trigger):
    """An instruction that is not always in context: an enable adds only its catalogue row."""
    path = root / 'skills' / 'external' / name
    path.mkdir(parents=True)
    (path / 'SKILL.md').write_text(f'---\nname: {name}\ndescription: A synthetic catalogue guide\ntype: instruction\n'
                                 f'when_to_use: {trigger}\n---\nA synthetic body.\n', encoding='utf-8')
    state = root / 'state' / 'skills' / name
    state.mkdir(parents=True)
    atomic_write_json(state / 'review.json', {'status': 'pass', 'content_hash': compute_content_hash(path)})
    return next(skill for skill in discover_skills(root, repo_path=config.REPO_DIR) if skill.name == name)


def test_enabling_an_ordinary_skill_counts_its_catalogue_row(installation):
    memory, _ctx, settings, live = installation
    root = memory.drive_root
    heavy = _install_catalogue_skill(root, 'catalogue-guide', f'When a synthetic row weighs {WEIGHT}')
    result = _toggle(heavy, root, str(config.REPO_DIR), True, 'owner_ui', 'synthetic fixture')
    assert result['code'] == 'nano_publication_over_capacity', result
    assert not load_enabled(root, heavy.name)
    assert any('When a synthetic row weighs' in wire for wire in live['externals'])
    light = _install_catalogue_skill(root, 'plain-guide', 'When a synthetic row weighs nothing')
    assert _toggle(light, root, str(config.REPO_DIR), True, 'owner_ui', 'synthetic fixture')['ok']
    assert load_enabled(root, light.name)
    # Max neither counts nor waits for the counter: the switch is an ordinary write.
    atomic_write_json(config.SETTINGS_PATH, {**settings, 'OUROBOROS_CONTEXT_MODE': 'max'})
    live['externals'].clear()
    live['active'] = False
    assert _toggle(heavy, root, str(config.REPO_DIR), True, 'owner_ui', 'synthetic fixture')['ok']
    assert load_enabled(root, heavy.name) and not live['externals']


def test_qualified_nano_catalogue_counts_same_floor_after_real_extension_registration(installation, monkeypatch, record_property):
    from ouroboros import context, extension_loader, nano_host_publication, tool_connections
    from ouroboros.agent import Env
    from ouroboros.nano_host_publication import _as_next_root
    from ouroboros.skill_loader import find_skill, save_enabled
    from ouroboros.tools.registry import ToolRegistry
    from ouroboros.tools.tool_discovery import enable_tools, list_available_tools
    from tests._extension_loader_shared import _write_ext_skill

    memory, ctx, settings, live = installation
    root = memory.drive_root
    name = 'nano-floor-fixture'
    tool_name = extension_loader.extension_surface_name(name, 'version')
    payload = _write_ext_skill(root / 'skills' / 'external', name, permissions=['tool'], plugin_body=(
        "def register(api):\n"
        "    api.register_tool('version', lambda ctx: 'synthetic', description='synthetic', schema={})\n"))
    save_enabled(root, name, True)
    state = root / 'state' / 'skills' / name
    atomic_write_json(state / 'review.json', {'status': 'pass', 'content_hash': compute_content_hash(payload)})
    skill = find_skill(root, name, repo_path=str(config.REPO_DIR))
    assert skill is not None and skill.enabled
    registry = ToolRegistry(config.REPO_DIR, root)
    registry.set_context(ctx)
    task = {'type': 'task', 'id': ctx.task_id, 'chat_id': 1, 'delegation_role': 'root', 'budget_drive_root': str(root)}
    live['extra_weight'] = tool_name
    assert 'updated' in _identity(ctx, memory, f'Saturated synthetic identity {WEIGHT * 40}')
    assert 'No tool' in list_available_tools(registry, ctx, [], name)
    before = measure_publication_floor(root)
    assert before['status'] == 'exact', before
    assert before['minimum_work_headroom_at_reply_floor_tokens'] == 15_000

    try:
        assert extension_loader.load_extension(skill, lambda: settings, drive_root=root,
                                               repo_path=str(config.REPO_DIR), _force_in_process=True) is None
        assert tool_name in list_available_tools(registry, ctx, [], name)
        resident = []
        enable_tools(registry, ctx, name, resident, set())
        assert any(row['function']['name'] == tool_name for row in resident)
        with _as_next_root(settings):
            plan = context.build_context_fit_plan(Env(config.REPO_DIR, root),
                                                  memory, task, ctx=ctx, tool_registry=registry,
                                                  preferred_mode='nano')
        section = next(row for row in plan.core.dynamic_sections if row.section_id == 'extension_tool_catalogue')
        assert tool_name in section.full_text and tool_name not in section.operative_view
        assert 'list_available_tools(namespace="<skill name>")' in section.operative_view
        live['externals'].clear()
        after = measure_publication_floor(root)
        assert after['status'] == 'exact', after
        assert after['minimum_work_headroom_at_reply_floor_tokens'] == before['minimum_work_headroom_at_reply_floor_tokens']
        assert any(tool_name not in wire for wire in live['externals'])
        with monkeypatch.context() as control:
            control.setattr(nano_host_publication, 'main_extension_tool_reader', lambda *_args: False)
            legacy = measure_publication_floor(root)
        assert legacy['status'] == 'exact', legacy
        assert legacy['minimum_work_headroom_at_reply_floor_tokens'] == after['minimum_work_headroom_at_reply_floor_tokens'] - 500
        record_property('fixture_work_floor_before_registration', before['minimum_work_headroom_at_reply_floor_tokens'])
        record_property('fixture_work_floor_after_registration', after['minimum_work_headroom_at_reply_floor_tokens'])
        record_property('fixture_work_floor_without_catalogue_step', legacy['minimum_work_headroom_at_reply_floor_tokens'])
        monkeypatch.setattr(tool_connections, 'usable_selectors', lambda _: frozenset({'enable_tools'}))
        (_, nano), section = tool_connections.connection_sections(registry, helper=False, allow_extension_tool_reader=True)
        assert section is None and tool_name in nano
    finally:
        extension_loader.unload_extension(name)


def test_ordinary_catalogue_transition_rechecks_profile_before_write(installation):
    from ouroboros.nano_activation import _visible_transition
    from ouroboros.nano_publication import NanoPublicationRefused
    from ouroboros.skill_loader import save_enabled

    memory, _ctx, settings, live = installation
    root = memory.drive_root
    skill = _install_catalogue_skill(root, 'profile-change-guide', f'New catalogue row {WEIGHT}')
    atomic_write_json(config.SETTINGS_PATH, {**settings, 'OUROBOROS_CONTEXT_MODE': 'max'})
    commit = _visible_transition(root, skill.name, skill, ())
    atomic_write_json(config.SETTINGS_PATH, settings)
    with pytest.raises(NanoPublicationRefused, match='active profile changed'):
        with commit():
            save_enabled(root, skill.name, True)
    assert not load_enabled(root, skill.name)
    assert not live['externals']


def test_stale_replacement_keeps_active_extension_bytes_and_restores_registration(installation, monkeypatch, tmp_path):
    from ouroboros import extension_loader, skill_source_install
    from ouroboros.skill_install_record import write_install_record
    from ouroboros.skill_loader import find_skill
    from ouroboros.skill_source_catalog import tree_release_digest
    from tests._extension_loader_shared import version_plugin

    monkeypatch.setenv('OUROBOROS_RUNTIME_MODE', 'advanced')
    memory, _ctx, settings, _live = installation
    root = memory.drive_root
    catalog = _catalog_update(monkeypatch, root, tmp_path, trust='review', resident=False)
    catalog.offer('Replacement body.')
    staged = tmp_path / 'cached-release' / catalog.name
    for path, version in ((catalog.target, 'old'), (staged, 'new')):
        manifest = path / 'SKILL.md'
        manifest.write_text(manifest.read_text().replace('type: instruction', 'type: extension\nentry: plugin.py\npermissions: [tool]'))
        (path / 'plugin.py').write_text(version_plugin(version))
    current = compute_content_hash(catalog.target)
    old_record = catalog.record()
    write_install_record(root, catalog.name, {**old_record, 'release': {'version': '1', 'digest': current}})
    saved = root / 'state' / 'skills' / catalog.name
    atomic_write_json(saved / 'review.json', {'status': 'pass', 'content_hash': current})
    before = (saved / 'enabled.json').read_bytes()
    tool_name = extension_loader.extension_surface_name(catalog.name, 'version')
    observed = []

    def load_current(*_args):
        active = find_skill(root, catalog.name, repo_path=str(config.REPO_DIR))
        assert extension_loader.load_extension(active, lambda: settings, drive_root=root,
                                               repo_path=str(config.REPO_DIR), _force_in_process=True) is None
        observed.append(extension_loader.get_tool(tool_name) is not None)

    real_unload = skill_source_install._unload

    def enter_nano_after_unload(root, name):
        was_live = real_unload(root, name)
        observed.append(extension_loader.get_tool(tool_name) is not None)
        atomic_write_json(config.SETTINGS_PATH, settings)
        return was_live

    monkeypatch.setattr(skill_source_install, '_unload', enter_nano_after_unload)
    monkeypatch.setattr(extension_loader, 'reconcile_extension', lambda *_a, **_kw: load_current())
    atomic_write_json(config.SETTINGS_PATH, {**settings, 'OUROBOROS_CONTEXT_MODE': 'max'})
    load_current()
    try:
        with pytest.raises(skill_source_install.SkillInstallError) as refused:
            catalog.update(tree_release_digest(staged))
        assert refused.value.code == 'nano_publication_stale'
        assert observed == [True, False, True]
        assert compute_content_hash(catalog.target) == current
        assert (saved / 'enabled.json').read_bytes() == before
    finally:
        extension_loader.unload_extension(catalog.name)


@pytest.mark.parametrize('reason,status', [('busy', 'pending'), ('stale', 'pending'), ('over_capacity', 'installed')])
def test_starter_keeps_only_transient_nano_turn_on_for_retry(installation, monkeypatch, reason, status):
    from ouroboros import skill_lifecycle_actions, skill_source_install, skill_starter
    from ouroboros.skill_install_record import write_install_record

    memory, _ctx, _settings, _live = installation
    root = memory.drive_root
    skill = _install_catalogue_skill(root, 'starter-fixture-guide', 'When exercising the starter retry.')
    write_install_record(root, skill.name, {'basis': 'trusted', 'turn_on_when_ready': True,
                                           'source': {'id': 'builtin'}, 'release': {'digest': skill.content_hash}})
    monkeypatch.setattr(skill_lifecycle_actions, '_toggle', lambda *_a, **_kw: {
        'ok': False, 'code': f'nano_publication_{reason}', 'error': 'The fixture switch stayed off.',
        'detail': 'Fixture reason retained.'})
    result = skill_lifecycle_actions.reconcile_turn_on_when_ready(root, skill.name, repo_path=str(config.REPO_DIR))
    assert result['detail'] == 'Fixture reason retained.' and result['error'] == 'The fixture switch stayed off.'
    monkeypatch.setattr(skill_source_install, 'run_in_lane', lambda _root, _kind, _id, call, **_kw: call())
    monkeypatch.setattr(skill_source_install, 'install_item', lambda *_a, **_kw: {
        'name': skill.name, 'already_installed': True})
    monkeypatch.setattr(skill_source_install, 'accept_and_turn_on', lambda *_a, **_kw: {'turn_on_result': result})
    monkeypatch.setattr(config, 'get_starter_skills_turn_on', lambda: True)
    attempt = skill_starter._install_one(root, 'builtin/starter-fixture-guide', [], config.REPO_DIR)
    assert attempt['status'] == status and attempt['code'] == f'nano_publication_{reason}'
    assert attempt['error'] == result['error'] and not load_enabled(root, skill.name)


def test_pending_unadmitted_on_rollback_retains_its_catalogue_weight(installation, monkeypatch, tmp_path):
    from ouroboros.nano_activation import _pending_rollbacks
    from ouroboros.skill_install_record import settle_pending_update, write_install_record
    from ouroboros.skill_source_catalog import tree_release_digest

    monkeypatch.setenv('OUROBOROS_RUNTIME_MODE', 'advanced')
    memory, ctx, settings, live = installation
    root = memory.drive_root
    catalog = _catalog_update(monkeypatch, root, tmp_path, trust='review', resident=False)
    old = catalog.record()
    path = catalog.target / 'SKILL.md'
    path.write_text(path.read_text().replace('description: A synthetic guide', f'description: Former guide {WEIGHT}'))
    write_install_record(root, catalog.name, {**old, 'basis': 'review'})
    atomic_write_json(root / 'state' / 'skills' / catalog.name / 'review.json',
                      {'status': 'blockers', 'content_hash': compute_content_hash(catalog.target)})
    atomic_write_json(config.SETTINGS_PATH, {**settings, 'OUROBOROS_CONTEXT_MODE': 'max'})
    catalog.offer('Short unadmitted replacement.', trigger='Light trigger')
    staged = tmp_path / 'cached-release' / catalog.name / 'SKILL.md'
    staged.write_text(staged.read_text().replace(f'description: Former guide {WEIGHT}', 'description: Replacement guide'))
    digest = tree_release_digest(staged.parent)
    assert catalog.update(digest)['needs_review']
    atomic_write_json(config.SETTINGS_PATH, settings)
    [old_candidate] = _pending_rollbacks(root)
    assert old_candidate.enabled and not old_candidate.review.gate_for(old_candidate.content_hash)['executable_review']
    assert 'over_capacity' in _identity(ctx, memory, f'Synthetic identity {WEIGHT * 40}')
    assert any(f'Former guide {WEIGHT}' in wire for wire in live['externals'])
    live['externals'].clear()
    assert settle_pending_update(root, catalog.name, expected_digest=digest, landing_failed=True)['status'] == 'restored'
    assert not live['externals'] and load_enabled(root, catalog.name)


def test_first_write_after_onboarding_uses_the_first_start_defaults(installation):
    memory, _ctx, _settings, live = installation
    root = memory.drive_root
    # After onboarding and before the first chat no worker has prepared memory yet. The Settings
    # writer runs in the server, which never calls ``ensure_files`` itself.
    for path in (memory.journal_path(), memory.identity_journal_path()):
        path.unlink()
    status, result = _rules('The first rules after onboarding.')
    assert status == 200, result
    # Counted against the defaults the next start writes; the count itself created no memory file.
    assert not memory.journal_path().exists() and not memory.identity_journal_path().exists()
    assert any('The first rules after onboarding.' in wire for wire in live['externals'])
    fact = read_root_sample(root, provider='local')
    assert fact['status'] == 'protecting', fact


def test_known_absence_is_a_witness_value_and_a_read_error_is_not(installation, monkeypatch):
    from ouroboros import nano_host_publication

    memory, _ctx, _settings, _live = installation
    root = memory.drive_root
    memory.world_path().unlink()  # WORLD generation failed at first start: an ordinary root reads ""
    assert nano_host_publication._profile_witness(root) is not None
    assert read_root_sample(root, provider='local').get('reason') != 'confirmed_counter_unavailable'
    blocked = root / 'memory' / 'not-a-directory'
    blocked.write_text('synthetic', encoding='utf-8')
    monkeypatch.setattr(nano_host_publication, '_active_sources', lambda _root: (blocked / 'journal.jsonl',))
    with pytest.raises(OSError):
        nano_host_publication._profile_witness(root)


def _script_catalog(monkeypatch, root, tmp_path, *, trust, name='fixture-script', initial_bucket='ouroboroshub'):
    import shutil
    from ouroboros import skill_source_install as installer
    from ouroboros.skill_install_record import read_install_record, write_install_record
    from ouroboros.skill_loader import save_enabled
    from ouroboros.skill_source_catalog import tree_release_digest

    def render_manifest(script_name, *, version='1', description='A synthetic script skill.', extra_front=''):
        return (
            f"---\nname: {name}\ndescription: {description}\nversion: '{version}'\n"
            f"type: script\nruntime: python3\nwhen_to_use: Fixture script trigger\n"
            f"scripts:\n  - name: {script_name}\n{extra_front}---\nScript guidance.\n"
        )

    old_dir = root / 'skills' / initial_bucket / name
    (old_dir / 'scripts').mkdir(parents=True)
    (old_dir / 'SKILL.md').write_text(render_manifest('v1_entry.py'), encoding='utf-8')
    (old_dir / 'scripts' / 'v1_entry.py').write_text("print('v1')\n", encoding='utf-8')
    old_digest = compute_content_hash(old_dir)
    source = ({'id': 'clawhub', 'kind': 'registry', 'name': 'ClawHub'}
              if initial_bucket == 'clawhub'
              else {'id': 'fixture-git', 'kind': 'git', 'name': 'Fixture source'})
    write_install_record(root, name, {'source': source, 'basis': trust, 'entry': name,
                                      'release': {'version': '1', 'digest': old_digest}})
    atomic_write_json(root / 'state' / 'skills' / name / 'review.json',
                      {'status': 'pass', 'content_hash': old_digest})
    save_enabled(root, name, True, actor='owner_ui')

    row = {'id': 'fixture-git', 'kind': 'git', 'name': 'Fixture source', 'trust': trust,
           'url': 'https://fixture.invalid/script-skills.git'}
    release = tmp_path / 'cached-release' / name
    release.mkdir(parents=True)
    version_box = {'value': '2'}
    monkeypatch.setattr('ouroboros.skill_source_install.find_item', lambda *_a, **_kw: (
        row, {'name': name, 'id': f'fixture-git/{name}', 'version': version_box['value'],
              'digest': tree_release_digest(release)}, {}))
    monkeypatch.setattr('ouroboros.skill_source_install.entry_payload_dir', lambda *_a, **_kw: release)

    def offer(declared_script, *, present_scripts=None, version='2',
              description='A synthetic script skill.', extra_front=''):
        version_box['value'] = version
        if release.exists():
            shutil.rmtree(release)
        (release / 'scripts').mkdir(parents=True)
        (release / 'SKILL.md').write_text(
            render_manifest(declared_script, version=version, description=description, extra_front=extra_front),
            encoding='utf-8',
        )
        for script_file in (declared_script,) if present_scripts is None else present_scripts:
            (release / 'scripts' / script_file).write_text("print('ok')\n", encoding='utf-8')
        return tree_release_digest(release)

    def update(digest):
        return installer.update_installed(root, name, mode='replace', digest=digest,
                                          rows=[row], repo_dir=config.REPO_DIR)

    def install(digest):
        return installer.install_item(root, f'fixture-git/{name}', replace=True,
                                      rows=[row], repo_dir=config.REPO_DIR)

    return SimpleNamespace(name=name, old_dir=old_dir, target=root / 'skills' / 'ouroboroshub' / name,
                           offer=offer, update=update, install=install,
                           record=lambda: read_install_record(root, name))


def _skill_cards(wire: str, name: str) -> list[list[str]]:
    def strings(item):
        if isinstance(item, str):
            yield item
        elif isinstance(item, dict):
            for value in item.values():
                yield from strings(value)
        elif isinstance(item, list):
            for value in item:
                yield from strings(value)

    cards: list[list[str]] = []
    for text in strings(json.loads(wire)):
        _, sep, rest = text.partition('## Installed Skills (')
        if not sep:
            continue
        lines = rest.split('\n## ', 1)[0].splitlines()
        for idx, line in enumerate(lines):
            if line.startswith(f'- {name} (') or line.startswith(f'- {name}:'):
                card = [line]
                for sub in lines[idx + 1:]:
                    if not sub.startswith('  '):
                        break
                    card.append(sub)
                cards.append(card)
    return cards


@pytest.mark.parametrize(('mode', 'initial_bucket', 'trust'), [
    ('update', 'ouroboroshub', 'trusted'),
    ('update', 'ouroboroshub', 'review'),
    ('replace', 'clawhub', 'trusted'),
])
def test_script_update_with_renamed_entry_reads_candidate_files_and_keeps_destination_bucket(
        installation, monkeypatch, tmp_path, mode, initial_bucket, trust):
    from ouroboros.context_sections import _installed_skills_section
    from ouroboros.skill_catalogue import discover_context_skills
    from ouroboros.skill_install_record import settle_pending_update
    from ouroboros.skill_source_install import SkillInstallError

    monkeypatch.setenv('OUROBOROS_RUNTIME_MODE', 'advanced')
    memory, ctx, _settings, live = installation
    root = memory.drive_root
    catalog = _script_catalog(monkeypatch, root, tmp_path, trust=trust, initial_bucket=initial_bucket)
    live['extra_weight'] = 'entry_unavailable'
    live['b'] = 36_808  # any false entry_unavailable from reading the old folder exceeds 80k
    _qualify(ctx, memory)
    live['externals'].clear()

    digest = catalog.offer('v2_entry.py', description='Updated synthetic script skill.')
    result = catalog.update(digest) if mode == 'update' else catalog.install(digest)
    assert result['ok']
    baseline_address = (f"read_file(root='skill_payload', bucket={initial_bucket!r}, "
                        f"skill_name={catalog.name!r}, path='SKILL.md')")
    expected_address = (f"read_file(root='skill_payload', bucket='ouroboroshub', "
                        f"skill_name={catalog.name!r}, path='SKILL.md')")
    cards = [card for wire in live['externals'] for card in _skill_cards(wire, catalog.name)]
    baseline_header = f'- {catalog.name} (script, v1, clean): A synthetic script skill.'
    updated_header = f'- {catalog.name} (script, v2, clean): Updated synthetic script skill.'
    baseline_cards = [card for card in cards if card[0] == baseline_header]
    updated_cards = [card for card in cards if 'Updated synthetic script skill.' in card[0]]
    assert baseline_cards and updated_cards and len(baseline_cards) + len(updated_cards) == len(cards), cards
    for card in baseline_cards:
        assert card[1:] == [
            '  Trigger: Fixture script trigger',
            f'  Instructions: {baseline_address}',
            '  Tools: skill_exec',
        ]
    admitted_v2 = [card for card in updated_cards if card[0] == updated_header]
    unadmitted_v2 = [card for card in updated_cards if card[0] != updated_header]
    assert admitted_v2 and bool(unadmitted_v2) == (trust == 'review')
    for card in admitted_v2:
        assert card[1:] == [
            '  Trigger: Fixture script trigger',
            f'  Instructions: {expected_address}',
            '  Tools: skill_exec',
        ]
    for card in unadmitted_v2:
        assert card == [
            f'- {catalog.name} (script): Updated synthetic script skill. Status: not ready — review_stale'
        ]
    if trust == 'trusted':
        rendered = _installed_skills_section(SimpleNamespace(drive_root=root), discover_context_skills(root), '')
        assert "bucket='ouroboroshub'" in rendered.full_text and 'Tools: skill_exec' in rendered.full_text
        assert 'entry_unavailable' not in rendered.full_text
        # Independent readiness blockers (e.g. ungranted core secrets) still surface on a valid new entry.
        live['extra_weight'] = 'missing_grants'
        live['externals'].clear()
        grant_digest = catalog.offer('v3_entry.py', version='3', description='Needs a secret grant.',
                                     extra_front='env_from_settings:\n  - OPENAI_API_KEY\n')
        monkeypatch.setenv('OUROBOROS_AUTO_GRANT_REVIEWED_SKILLS', '0')
        with pytest.raises(SkillInstallError) as refusal:
            catalog.update(grant_digest)
        assert refusal.value.code == 'nano_publication_over_capacity'
        assert any('missing_grants' in wire and 'entry_unavailable' not in wire for wire in live['externals'])
    else:
        assert settle_pending_update(root, catalog.name, expected_digest=digest)['status'] == 'restored'


def test_missing_entry_in_candidate_or_rollback_is_counted_from_its_own_version_files(
        installation, monkeypatch, tmp_path):
    from ouroboros.skill_install_record import settle_pending_update, write_install_record
    from ouroboros.skill_source_install import SkillInstallError

    monkeypatch.setenv('OUROBOROS_RUNTIME_MODE', 'advanced')
    memory, ctx, _settings, live = installation
    root = memory.drive_root
    catalog = _script_catalog(monkeypatch, root, tmp_path, trust='trusted')
    live['extra_weight'] = 'entry_unavailable'
    live['b'] = 36_808
    _qualify(ctx, memory)
    live['externals'].clear()

    # 1. Staged candidate omits v1_entry.py even though target still has scripts/v1_entry.py on disk.
    broken = catalog.offer('v1_entry.py', present_scripts=(), description='Broken script release.')
    with pytest.raises(SkillInstallError) as refusal:
        catalog.update(broken)
    assert refusal.value.code == 'nano_publication_over_capacity'
    assert (catalog.target / 'scripts' / 'v1_entry.py').is_file()
    assert any('entry_unavailable: no declared script file exists (scripts/v1_entry.py)' in wire
               for wire in live['externals'])

    # 2. Pending rollback reads backup_dir's own files, neither blocked by nor borrowing target's files.
    review_catalog = _script_catalog(monkeypatch, root, tmp_path, trust='review', name='rollback-script')
    valid_v2 = review_catalog.offer('v2_entry.py', description='Candidate with renamed script.')
    assert review_catalog.update(valid_v2)['needs_review']
    assert not (review_catalog.target / 'scripts' / 'v1_entry.py').exists()
    # Backup still has v1_entry.py, so the rollback branch has no entry_unavailable and keeps bucket='ouroboroshub'.
    live['externals'].clear()
    assert 'RESOURCE_CONSTRAINT_BLOCKED' not in _identity(ctx, memory, f'Synthetic identity {WEIGHT * 39}')
    assert any("bucket='ouroboroshub'" in wire and 'rollback-script' in wire for wire in live['externals'])
    assert all('entry_unavailable' not in wire for wire in live['externals'])
    assert settle_pending_update(root, review_catalog.name, expected_digest=valid_v2)['status'] == 'restored'

    # Conversely, when the kept backup lacks v1_entry.py while the new target supplies it, rollback still prices the gap.
    (review_catalog.target / 'scripts' / 'v1_entry.py').unlink()
    gap_digest = compute_content_hash(review_catalog.target)
    write_install_record(root, review_catalog.name, {**review_catalog.record(),
                                                     'release': {'version': '1', 'digest': gap_digest}})
    atomic_write_json(root / 'state' / 'skills' / review_catalog.name / 'review.json',
                      {'status': 'pass', 'content_hash': gap_digest})
    repaired_v2 = review_catalog.offer('v1_entry.py', description='Repaired script release.')
    assert review_catalog.update(repaired_v2)['needs_review']
    assert not _verdict(root, ctx, review_catalog.name, repaired_v2, 'pass', 'repaired-pass').error
    assert (review_catalog.target / 'scripts' / 'v1_entry.py').is_file()
    live['externals'].clear()
    blocked = _identity(ctx, memory, f'Synthetic identity {WEIGHT * 40}')
    assert 'RESOURCE_CONSTRAINT_BLOCKED' in blocked and 'over_capacity' in blocked
    assert any('entry_unavailable: no declared script file exists (scripts/v1_entry.py)' in wire
               for wire in live['externals'])


def test_off_skill_pending_rollback_is_priced_when_visible_and_skipped_when_unchanged(
        installation, monkeypatch, tmp_path):
    from ouroboros import skill_lifecycle_queue
    from ouroboros.nano_activation import _pending_rollbacks
    from ouroboros.skill_install_record import settle_pending_update, write_install_record
    from ouroboros.skill_loader import find_skill, save_enabled
    from ouroboros.skill_source_catalog import tree_release_digest
    from ouroboros.skill_source_install import SkillInstallError

    monkeypatch.setenv('OUROBOROS_RUNTIME_MODE', 'advanced')
    monkeypatch.setenv('OUROBOROS_SKILL_SOURCES', '[]')
    monkeypatch.setenv('OUROBOROS_SKILLS_REPO_PATH', '')
    memory, ctx, _settings, live = installation
    root = memory.drive_root
    catalog = _catalog_update(monkeypatch, root, tmp_path, trust='review', name='off-guide')
    save_enabled(root, catalog.name, False, actor='owner_ui')
    manifest = catalog.target / 'SKILL.md'
    manifest.write_text(manifest.read_text().replace('description: A synthetic guide',
                                                     f'description: Former off summary {WEIGHT}'), encoding='utf-8')
    old_digest = compute_content_hash(catalog.target)
    write_install_record(root, catalog.name, {**catalog.record(), 'release': {'version': '1', 'digest': old_digest}})
    atomic_write_json(root / 'state' / 'skills' / catalog.name / 'review.json',
                      {'status': 'pass', 'content_hash': old_digest})
    _qualify(ctx, memory)

    # 1. Off update that shortens the visible catalogue description leaves a heavier Off rollback row.
    catalog.offer('Short replacement body.')
    staged = tmp_path / 'cached-release' / catalog.name / 'SKILL.md'
    staged.write_text(staged.read_text().replace(f'description: Former off summary {WEIGHT}',
                                                 'description: Lighter off summary'), encoding='utf-8')
    digest = tree_release_digest(staged.parent)
    assert catalog.update(digest)['needs_review']
    [restored_off] = _pending_rollbacks(root)
    assert not restored_off.enabled
    assert 'RESOURCE_CONSTRAINT_BLOCKED' not in _identity(ctx, memory, f'Synthetic identity {WEIGHT * 39}')
    live['externals'].clear()
    blocked = _identity(ctx, memory, f'Synthetic identity {WEIGHT * 40}')
    assert 'RESOURCE_CONSTRAINT_BLOCKED' in blocked and 'over_capacity' in blocked
    assert any(f'Former off summary {WEIGHT}' in wire and 'Former instruction body.' not in wire
               for wire in live['externals'])

    # Competing review-basis update whose visible row also changes waits behind the pending Off rollback.
    second = _catalog_update(monkeypatch, root, tmp_path / 'second', trust='review', name='second-guide')
    second_digest = second.offer('Competing instruction body.')
    with pytest.raises(SkillInstallError) as busy_refusal:
        second.update(second_digest)
    assert busy_refusal.value.code == 'nano_publication_busy'

    # Enabling a skill inside the lifecycle lane lends the lane lock during counting.
    ready_off = _install_instruction(root, 'lane-toggle-guide', 'Small body.')
    locked_during_count = []
    live['on_count'] = lambda: locked_during_count.append(skill_lifecycle_queue._get_lock().locked())
    try:
        toggled = skill_lifecycle_queue.run_lifecycle_job_blocking(
            kind='enable', target=ready_off.name, source='skills', message='enable', chat_id=0,
            runner=lambda: _toggle(find_skill(root, ready_off.name), root, str(config.REPO_DIR),
                                   True, 'owner_ui', 'fixture'),
            options=skill_lifecycle_queue.LifecycleJobOptions(drive_root=root),
        )
    finally:
        live['on_count'] = None
    assert toggled['ok'] and locked_during_count and not any(locked_during_count)
    assert _toggle(find_skill(root, ready_off.name), root, str(config.REPO_DIR), False, 'owner_ui', 'fixture')['ok']

    assert settle_pending_update(root, catalog.name, expected_digest=digest)['status'] == 'restored'
    assert not load_enabled(root, catalog.name)

    # 2. Positive preserved path: an Off update whose rendered Installed Skills row is unchanged
    # (only body changes while description stays identical) creates no competing rollback debt.
    manifest.write_text(manifest.read_text().replace(f'description: Former off summary {WEIGHT}',
                                                     'description: Stable off summary')
                        .replace('Former instruction body.', f'Hidden off body {WEIGHT}.'), encoding='utf-8')
    stable_old = compute_content_hash(catalog.target)
    write_install_record(root, catalog.name, {**catalog.record(), 'pending_update': None, 'update_rejected': None,
                                              'release': {'version': '1', 'digest': stable_old}})
    atomic_write_json(root / 'state' / 'skills' / catalog.name / 'review.json',
                      {'status': 'pass', 'content_hash': stable_old})
    staged.write_text(manifest.read_text().replace(f'Hidden off body {WEIGHT}.', 'Updated hidden off body.'),
                      encoding='utf-8')
    unchanged_row_digest = tree_release_digest(staged.parent)
    assert catalog.update(unchanged_row_digest)['needs_review']
    assert _pending_rollbacks(root) == []
    assert 'RESOURCE_CONSTRAINT_BLOCKED' not in _identity(ctx, memory, f'Synthetic identity {WEIGHT * 40}')
    assert second.update(second_digest)['needs_review']
    assert settle_pending_update(root, second.name, expected_digest=second_digest)['status'] == 'restored'

    # 3. Enabling that Off skill while its rollback is still pending prices the rollback as On.
    atomic_write_json(root / 'state' / 'skills' / catalog.name / 'review.json',
                      {'status': 'pass', 'content_hash': unchanged_row_digest})
    over_on_rollback = _toggle(find_skill(root, catalog.name), root, '', True, 'owner_ui', 'fixture')
    assert over_on_rollback['code'] == 'nano_publication_over_capacity'
    assert not load_enabled(root, catalog.name)
