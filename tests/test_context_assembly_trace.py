"""One true room piece and architecture book witnessed on a real sealed fixture request."""
from ouroboros import context_attempt_reader, usage_accounting
from ouroboros.physical_attempt_context import PhysicalAttemptContext
from ouroboros.context_fit import ContextCore, _governance_blocks
from ouroboros.context_assembly_trace import layout_witnesses
from ouroboros.llm_messages import STABLE_PREFIX_BLOCKS_KEY, split_leading_system_prefix
from ouroboros.section_floor import HostSection, _system_blocks
from tests._context_reports import _dispatch


def test_assembled_sources_survive_split_and_unverified_placement_is_named(tmp_path, monkeypatch):
    root = tmp_path / 'layout'
    monkeypatch.setenv('OUROBOROS_DATA_DIR', str(root))
    monkeypatch.setenv('OUROBOROS_SETTINGS_PATH', str(root / 'settings.json'))
    monkeypatch.setattr(usage_accounting, 'estimate_cost_optional', lambda *args, **kwargs: 0.01)
    core = ContextCore(base_prompt='Guide', bible_md='Constitution', architecture_md='Map text',
                       development_md='Handbook', semi_stable_text='', dynamic_text='', user_content_json='null',
                       docs_need_development=True, semi_stable_sections=(HostSection('identity', 'Identity text'),),
                       dynamic_head_sections=(HostSection('knowledge_index', 'Knowledge'),),
                       dynamic_sections=(HostSection('recent_tools', 'Tools ran'),))
    governance = _governance_blocks(None, core, mode='max')
    room = 'This room\n\nTelegram elsewhere'
    room_pieces = [{'family': 'room', 'start': 0, 'chars': 9},
                   {'family': 'other_conversation', 'provider': 'telegram', 'start': 11, 'chars': 18}]
    layout = []

    def observe(live):
        layout.extend(layout_witnesses(core, governance, {}, (), live, story_whole=True, room_pieces=room_pieces))

    blocks = _system_blocks(core, governance, 'Story', room, 'max', trace=observe)
    assert blocks == _system_blocks(core, governance, 'Story', room, 'max')
    messages, moved = split_leading_system_prefix([
        {'role': 'system', 'content': blocks, STABLE_PREFIX_BLOCKS_KEY: len(blocks) - 1},
        {'role': 'user', 'content': 'My input'},
    ])
    assert moved == 1
    fit = PhysicalAttemptContext(profile='owner_max', rendered_mode='max', measurement_basis='cold_estimate',
        route_fp='', round_id='1', target_total_tokens=None, capacity_total_tokens=None, context_target_miss=False,
        automatic_pass_used=False, assembly_witnesses=tuple(layout))
    for edited in (False, True):
        current = [dict(message) for message in messages]
        if edited:
            current[-2]['content'] = current[-2]['content'].replace('Tools ran', 'Tools ran again')
        with usage_accounting.bind_physical_attempt_context(fit):
            attempt = _dispatch(root, 'layout-test', payload={'model': 'gpt-x', 'messages': current, 'tools': []})['attempt_id']
        facts, request = context_attempt_reader.read_attempt(root, 'layout-test', attempt, report_facts=True)
        recorded = facts['assembly_source_map']
        assert request is not None
        assert len(recorded) == len(blocks) == 4
        assert {group['family'] for block in recorded for group in block['groups']} >= {
            'system', 'bible', 'development', 'identity', 'story', 'knowledge_index', 'room', 'other_conversation', 'recent_tools',
        }
        assert [block['status'] for block in recorded] == ['located'] * 3 + [
            'position_unverified' if edited else 'located']
        if not edited:
            span = recorded[-1]['prepared_text']
            text = current[-2]['content'][span['start_char']:span['end_char']]
            assert text == blocks[-1]['text']
            # Independently walk the actual carrier; all non-separator text lies in named, non-overlapping groups.
            offset = 0
            for group in recorded[-1]['groups']:
                assert text[offset:group['start']] == ('\n\n' if offset else '')
                offset = group['start'] + group['chars']
            assert offset == len(text)


def test_late_contributions_bind_exact_prepared_forms_without_storing_body(tmp_path):
    import json
    from types import SimpleNamespace

    from ouroboros.context_assembly_trace import bind_contribution_input, note_contribution, verified_contributions
    from ouroboros.context_budget import HOST_CONTEXT_KIND_KEY
    from ouroboros.llm_messages import project_wire_messages
    from ouroboros.model_send_seal import persist_physical_candidate
    from ouroboros.physical_attempt_context import PhysicalAttemptContext
    from ouroboros.tool_result_record import TOOL_RESULT_RECORD_KEY, make_tool_result_record

    owner = {'role': 'user', HOST_CONTEXT_KIND_KEY: 'owner_words', 'content': 'PRIVATE owner input'}
    peer = {'role': 'user', 'content': 'PRIVATE peer text'}
    prior_media = {'role': 'user', 'content': [{'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,PRIVATE'}}]}
    from ouroboros.vision_routing import _rewrite

    projected_media, = _rewrite([prior_media], lambda _image: ('PRIVATE image replacement', 'images_off'))
    result = {'role': 'tool', 'tool_call_id': 'call_1', 'content': 'PRIVATE result'}
    result[TOOL_RESULT_RECORD_KEY] = make_tool_result_record({'tool_call_id': 'call_1', 'invocation_id': 'PRIVATE-id'},
                                                            result['content'], source_ref={'path': 'PRIVATE'})
    tools = [{'type': 'function', 'function': {'name': 'inspect', 'parameters': {'type': 'object'}}}]
    projected = {'messages': [owner, peer, projected_media, result], 'tools': tools}
    slot = SimpleNamespace()
    note_contribution(slot, owner, 'owner_input', form='direct_input')
    note_contribution(slot, peer, 'place_arrival', source='PRIVATE place ID', form='containing_row')
    note_contribution(slot, result, 'result_selection', form='partial_retained')
    fit = PhysicalAttemptContext('owner_low', 'low', 'cold_estimate', '', '', None, None, False, False)
    fit = bind_contribution_input(fit, projected, [owner, peer, prior_media, result], slot)
    wire = {'messages': project_wire_messages(projected['messages']), 'tools': tools}
    same = verified_contributions(wire, fit.contribution_input)
    by_family = {row['family']: row for row in same['entries']}
    assert by_family['owner_input']['path'] == '$.messages[0]'
    assert by_family['place_arrival']['path'] == '$.messages[1]'
    assert by_family['attachment_image']['form'] == 'images_off'
    assert prior_media['content'][0]['type'] == 'image_url' and '_image_projection_fact' not in json.dumps(wire)
    assert by_family['tool_result']['form'] == 'result_view'
    assert by_family['result_selection']['form'] == 'partial_retained'
    assert by_family['tool_schema']['path'] == '$.tools[0]'
    assert 'PRIVATE' not in json.dumps(same)

    rewritten = {'input': [wire['messages'][0], wire['messages'][0],
                           {'type': 'function_call_output', 'call_id': 'call_1', 'output': result['content']}],
                 'tools': [{'type': 'function', 'name': 'inspect', 'parameters': {'type': 'array'}}]}
    changed = {row['family']: row for row in verified_contributions(rewritten, fit.contribution_input)['entries']}
    for family in ('owner_input', 'place_arrival', 'attachment_image', 'tool_schema'):
        assert changed[family]['status'] == 'position_unverified' and 'path' not in changed[family]
    assert changed['tool_result']['path'] == '$.input[2]'
    tool_part = {'type': 'tool_result', 'tool_use_id': 'call_1', 'content': result['content']}
    native = {'messages': [{'role': 'user', 'content': [tool_part, *[{'type': 'text', 'text': 'pad'} for _ in range(31)], tool_part]}]}
    bounded = verified_contributions(native, fit.contribution_input)
    assert bounded['uninspected_physical_items'] == 1
    assert all('path' not in row for row in bounded['entries'])

    persisted = persist_physical_candidate(tmp_path, task_id='task', attempt_id='attempt', candidate=wire,
                                           candidate_facts={'contribution_input': fit.contribution_input})
    manifest = json.loads(open(persisted['manifest_ref']['path'], encoding='utf-8').read())
    assert manifest['late_contributions'] == same and 'contribution_input' not in manifest
    assert 'PRIVATE' not in json.dumps(manifest['late_contributions'])
    assert '_host_context_kind' not in json.dumps(wire)



def test_attachment_limit_records_decisions_without_changing_input(tmp_path, monkeypatch):
    from ouroboros.context import build_user_content
    from ouroboros.artifacts import task_artifact_dir_path

    monkeypatch.setattr('ouroboros.context_budget.MAX_LIVE_IMAGE_BLOCKS', 1)
    location = task_artifact_dir_path(tmp_path, 'report-task', create=True)
    png = __import__('base64').b64decode(
        'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+a1V8AAAAASUVORK5CYII=')
    (location / 'valid.png').write_bytes(png)
    task = {'text': 'Test attachment', 'drive_root': tmp_path, 'id': 'report-task', 'attachment_images': [
        {'is_image': False, 'relpath': 'note.txt'},
        {'is_image': True, 'relpath': 'absent.png'},
        {'is_image': True, 'relpath': 'valid.png'},
        {'is_image': True, 'relpath': 'untouched.png'},
    ]}
    facts = {}
    assert build_user_content(task, attachment_facts=facts) == build_user_content(task)
    assert [entry['decision'] for entry in facts['attachment_items']] == [
        'not_selected_as_image', 'unreadable_source', 'prepared_pixels', 'image_count_limit']
    assert facts['manifest_pixels'] == 1


def test_merged_row_discloses_dropped_contribution_notes():
    from types import SimpleNamespace

    from ouroboros.context_assembly_trace import (bind_contribution_input, move_contributions,
                                                  note_contribution, verified_contributions)

    slot = SimpleNamespace()
    row = {'role': 'user', 'content': 'initial'}
    for i in range(33):
        note_contribution(slot, row, 'task_mail', source=str(i))
    merged = {'role': 'user', 'content': 'initial and later'}
    move_contributions(slot, row, merged)
    note_contribution(slot, merged, 'task_mail', source='later')
    wire = {'messages': [merged]}
    fit = PhysicalAttemptContext('owner_low', 'low', 'cold_estimate', '', '', None, None, False, False)
    bound = bind_contribution_input(fit, wire, [merged], slot)
    facts = verified_contributions(wire, bound.contribution_input)
    assert facts['row_notes_omitted'] == 2
    assert len(facts['entries']) == 32
    assert all(row['path'] == '$.messages[0]' for row in facts['entries'])
