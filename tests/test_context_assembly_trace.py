"""One true room piece and architecture book witnessed on a real sealed fixture request."""
from ouroboros import context_attempt_reader, usage_accounting
from ouroboros.physical_attempt_context import PhysicalAttemptContext
from ouroboros.context_fit import ContextCore, _governance_blocks
from ouroboros.context_assembly_trace import RoomWitness, digest, layout_witnesses
from ouroboros.llm_messages import STABLE_PREFIX_BLOCKS_KEY, split_leading_system_prefix
from ouroboros.section_floor import HostSection, _system_blocks, capture_system_projection, projection_fields
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


def test_projection_witness_is_bound_to_the_selected_blocks_and_declared_append():
    core = ContextCore(base_prompt='Guide', bible_md='Constitution', architecture_md='Map text',
                       development_md='Handbook', semi_stable_text='', dynamic_text='', user_content_json='null',
                       docs_need_development=True, dynamic_sections=(HostSection('recent_tools', 'Tools ran'),))
    governance = _governance_blocks(None, core, mode='max')
    facts = {'floor': {'steps': {}}}
    system, captured = capture_system_projection(core, governance, 'Story', 'Room', 'max', {}, (), True,
                                                  shown=None, facts=facts)
    appended = [*system[:-1], {**system[-1], 'text': system[-1]['text'] + '\n\nSelected first-contact note'}]
    first_contact = {**captured, 'first_contact_preference': True}
    rendered = projection_fields(system, captured)  # every rendered view: a later rebind may select it unrendered
    assert rendered['memory_facts'] == facts and '_assembly_layout' not in rendered['memory_facts']
    assert [row['carrier_sha256'] for row in rendered['assembly_witnesses']] == [digest(block['text']) for block in system]
    assert projection_fields(appended, captured)['assembly_witnesses'] == ()
    measured = projection_fields(appended, first_contact)
    assert measured['memory_facts'] == {**facts, 'first_contact_preference': True}
    tail = measured['assembly_witnesses'][-1]
    assert tail['carrier_sha256'] == digest(appended[-1]['text'])
    assert tail['groups'][-1]['family'] == 'memory_fit_notes'
    changed_book = [{**appended[0], 'text': 'Different stable text'}, *appended[1:]]
    assert projection_fields(changed_book, first_contact)['assembly_witnesses'] == ()


def test_an_optional_section_this_actor_cannot_read_keeps_the_map_and_names_no_source():
    """``memory_floor._address_optional`` shortens an optional attachment for an actor with no reader and records
    ``source_ref: None``. That section is still witnessed as shortened, without a retained address; every other
    section keeps its witness."""
    from ouroboros.memory_floor import _address_optional

    design = HostSection('design_bible', '## Design bible\n\n' + 'Airy rule. ' * 400, operative_view='## Design bible\n\nAiry.')
    core = ContextCore(base_prompt='Guide', bible_md='Constitution', architecture_md='Map text', development_md='Handbook',
                       semi_stable_text='', dynamic_text='', user_content_json='null', docs_need_development=True,
                       dynamic_sections=(HostSection('recent_tools', 'Tools ran'), design))
    receipts, addressed = [], {}
    assert _address_optional(design, receipts, addressed, root='', task_id='') > 0
    assert receipts[0]['source_ref'] is None and 'unavailable to this actor' in addressed['design_bible']
    system, captured = capture_system_projection(core, _governance_blocks(None, core, mode='max'), 'Story', 'Room', 'max',
                                                  addressed, receipts, True, shown=None, facts={'floor': {'steps': {}}})
    witnesses = projection_fields(system, captured)['assembly_witnesses']
    assert [row['carrier_sha256'] for row in witnesses] == [digest(block['text']) for block in system]
    groups = {group['family']: group for row in witnesses for group in row['groups']}
    assert {'system', 'bible', 'development', 'recent_tools', 'story'} <= groups.keys()
    shortened = groups['design_bible']
    assert shortened['form'] == 'shorter' and shortened['reason'] == 'physical_window' and 'retained_source' not in shortened
    text = system[shortened.get('block', -1)]['text'] if 'block' in shortened else system[-1]['text']
    assert text[shortened['start']:shortened['start'] + shortened['chars']] == addressed['design_bible']


def test_a_cold_restore_keeping_low_sends_the_witness_of_its_low_view(tmp_path, monkeypatch):
    """A restored task that ran in Low rebinds this process's fresh Max build to Low (``owner_wait.
    rebind_restored_route``). Where Low already fits, its view is kept, not rendered again; the witness taken when its
    bytes were rendered names each of its blocks, so the send locates all of them with Low's own book forms."""
    import json

    from ouroboros.context_assembly_trace import verified_layout
    from tests._context_fit_fixtures import _built, _rebind

    _core, plan = _built(tmp_path / 'w', monkeypatch)
    fresh = plan(1_000_000, preferred='max', tool_schemas=[])
    messages = fresh.messages_for('max')
    rebound, mode, _event = _rebind(monkeypatch, tmp_path / 'restore', fresh, messages, window=1_000_000, start_mode='low')
    low = rebound.projection('low')
    assert fresh.initial_mode == 'max' and mode == 'low'
    assert low.system_content_json == fresh.projection('low').system_content_json  # kept: no walk, no new render
    blocks = json.loads(low.system_content_json)
    assert [row['carrier_sha256'] for row in low.assembly_witnesses] == [digest(block['text']) for block in blocks]
    layout = verified_layout({'messages': messages}, low.assembly_witnesses)
    assert [block['status'] for block in layout] == ['located'] * len(blocks)
    def forms(rows):
        return {(group['family'], group['form']) for row in rows for group in row['groups']}

    assert ('architecture', 'whole_book') in forms(fresh.projection('max').assembly_witnesses)
    assert ('architecture', 'whole_book') not in forms(layout)  # Low's own render, not the fresh start's Max


def test_late_contributions_bind_exact_prepared_forms_without_storing_body(tmp_path, monkeypatch):
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

    slot = SimpleNamespace()
    monkeypatch.setattr('ouroboros.model_wait.current_model_wait', lambda: SimpleNamespace(tool_context=slot))
    projected_media, = _rewrite([prior_media], lambda _image: ('PRIVATE image replacement', 'images_off'))
    assert projected_media['content'] == [{'type': 'text', 'text': 'PRIVATE image replacement'}]
    result = {'role': 'tool', 'tool_call_id': 'call_1', 'content': 'PRIVATE result'}
    result[TOOL_RESULT_RECORD_KEY] = make_tool_result_record({'tool_call_id': 'call_1', 'invocation_id': 'PRIVATE-id'},
                                                            result['content'], source_ref={'path': 'PRIVATE'})
    tools = [{'type': 'function', 'function': {'name': 'inspect', 'parameters': {'type': 'object'}}}]
    projected = {'messages': [owner, peer, projected_media, result], 'tools': tools}
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

    # An equal but detached text block was not witnessed by the producer.
    detached = {'messages': [{**projected_media, 'content': [dict(projected_media['content'][0])]}]}
    detached_fit = bind_contribution_input(fit, detached, [prior_media], slot)
    assert not verified_contributions(detached, detached_fit.contribution_input)['entries']



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


def test_a_failed_same_model_trial_leaves_the_restored_send_its_mail_source():
    """A configured route with Main's model on another account sends a deep copy of the acting frame
    (``loop_model_call._fallback_rows``) with its notice appended. Its sends read the originals' producer notes through
    bounded pairs, and until the trial settles none retires an original's notes, not even one its longer frame left
    outside its own window: the restored primary send still names the mail's containing row. An answered trial's
    copies take the notes over; a copy that replaces its original at once (an owner's switch) takes them directly."""
    import copy
    from types import SimpleNamespace

    from ouroboros.context_assembly_trace import (bind_contribution_input, carry_contributions, note_contribution,
                                                  settle_copied_contributions, verified_contributions)
    from ouroboros.llm_messages import project_wire_messages
    from ouroboros.loop_model_call import _adopt_fallback_route, _fallback_rows
    from ouroboros.primary_route_observation import tell_failed_attempt

    fit = PhysicalAttemptContext('owner_max', 'max', 'cold_estimate', '', '', None, None, False, False)

    def mail_rows(slot, logical):  # the send seam's reading: each task_mail fact and the row it located
        wire = {'messages': project_wire_messages(logical)}
        bound = bind_contribution_input(fit, wire, logical, slot)
        return [(row.get('path'), row.get('source_id_sha256'), row['form'])
                for row in verified_contributions(wire, bound.contribution_input)['entries'] if row['family'] == 'task_mail']

    def trial(slot, rows):  # a candidate as ``prepare`` makes it: the copy, then its notice
        _plan, copied, _schemas, _mode = _fallback_rows((None, rows, None, 'max', 'model/a'), 'model/a', slot)
        assert copied == rows and not any(new is old for new, old in zip(copied, rows))
        tell_failed_attempt(copied, 'The previous attempt failed.', 'model/a on another account')
        return copied

    slot = SimpleNamespace()
    mail = {'role': 'user', '_host_context_kind': 'owner_words', 'content': '[Message from ancestor task p]\nexact note'}
    rows = [{'role': 'system', 'content': 'view'}, {'role': 'user', 'content': 'go'}, mail]
    note_contribution(slot, mail, 'task_mail', source='mail-1', form='containing_row')
    located = [('$.messages[2]', digest('mail-1'), 'containing_row')]
    failed = trial(slot, rows)
    assert mail_rows(slot, failed) == mail_rows(slot, failed) == located  # each send of the trial names its copy's source
    settle_copied_contributions(slot)  # ``restore_acting_view``: the failed trial's copies go
    assert mail_rows(slot, rows) == located and not slot._context_copies

    edge_slot, edge = SimpleNamespace(), [{'role': 'user', 'content': f'row {i}'} for i in range(256)]
    note_contribution(edge_slot, edge[0], 'task_mail', source='mail-edge', form='containing_row')
    longer = trial(edge_slot, edge)  # its notice leaves the copied mail row outside the trial's 256 rows
    assert mail_rows(edge_slot, longer) == mail_rows(edge_slot, longer) == []
    settle_copied_contributions(edge_slot)
    assert mail_rows(edge_slot, edge) == [('$.messages[0]', digest('mail-edge'), 'containing_row')]

    answered = trial(slot, rows)
    acting = SimpleNamespace(active_model='model/a', active_use_local=False)
    candidate = SimpleNamespace(active_model='model/a', active_use_local=False, messages=answered, tool_schemas=[],
                                context_fit_plan=None, tools=SimpleNamespace(_ctx=slot))
    _adopt_fallback_route(acting, candidate, rows, None, 'max')
    assert rows[2] is answered[2] and mail_rows(slot, rows) == located  # the answered copy carries them now
    assert id(mail) not in slot._context_contributions and not slot._context_copies  # the replaced original is released

    switched = copy.deepcopy(rows)
    carry_contributions(slot, rows, switched, adopt=True)
    assert mail_rows(slot, switched) == located and not slot._context_copies

    many = [{'role': 'user', 'content': f'row {i}'} for i in range(300)]
    for row in many:
        note_contribution(slot, row, 'task_mail', source=row['content'])
    carry_contributions(slot, many, copy.deepcopy(many))
    assert len(slot._context_copies) == 256  # bounded like the notes it reads


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


def test_start_selection_carries_schema_origins_outside_copied_wire_dicts():
    from types import SimpleNamespace

    from ouroboros.context_assembly_trace import bind_contribution_input, note_schema_origin, verified_contributions
    from ouroboros.tool_policy import initial_tool_schemas

    slot = SimpleNamespace()
    names = ('ext_like_builtin', 'notification', 'mcp_reader', 'mcp_unloaded', 'unsourced')
    canonical = [{'type': 'function', 'function': {'name': name}} for name in names]

    def schemas():
        for definition, origin in zip(canonical, ('builtin', 'extension', 'mcp', 'mcp')):
            note_schema_origin(slot, definition, origin)
        return canonical

    registry = SimpleNamespace(_ctx=slot, schemas=schemas, on_demand_schema_names=lambda: {'mcp_unloaded'})
    selected = initial_tool_schemas(registry)
    expected = [*canonical[:3], canonical[4]]
    assert selected == expected
    assert all(new is not old for new, old in zip(selected, expected))
    fit = PhysicalAttemptContext('owner_max', 'max', 'cold_estimate', '', '', None, None, False, False)
    request = {'messages': [], 'tools': selected}
    bound = bind_contribution_input(fit, request, [], slot)
    entries = verified_contributions(request, bound.contribution_input)['entries']
    assert [item['path'] for item in entries] == [f'$.tools[{i}]' for i in range(4)]
    assert [item.get('origin') for item in entries] == ['builtin', 'extension', 'mcp', None]
    # Equal bytes from an old serialized or unaffiliated list do not prove its origin.
    detached = {'messages': [], 'tools': [dict(selected[0])]}
    again = bind_contribution_input(fit, detached, [], slot)
    assert 'origin' not in verified_contributions(detached, again.contribution_input)['entries'][0]


def test_oversize_physical_position_scan_counts_every_unread_item():
    from ouroboros.context_assembly_trace import bind_contribution_input, verified_contributions

    fit = PhysicalAttemptContext('owner_low', 'low', 'cold_estimate', '', '', None, None, False, False)
    row = {'role': 'user', 'content': 'synthetic'}
    logical = [row]
    fit = bind_contribution_input(fit, {'messages': logical}, logical)
    for count, omitted in ((264, 0), (265, 265)):
        facts = verified_contributions({'messages': [row] * count}, fit.contribution_input)
        assert facts['uninspected_physical_items'] == omitted


def test_conversation_observer_uses_same_description_for_every_group_key():
    from types import SimpleNamespace

    keys = ('telegram:one', 'slack:two', 'github:three', 'project:four')
    witness = RoomWitness(SimpleNamespace(conversations=[{'key': k, 'rooms': [{'room_id': '42'}]} for k in keys]))
    pieces = [witness.piece_fact(('conversations', 'group', k, '', (), (), 0)) for k in keys]
    assert all(piece == pieces[0] and 'provider' not in piece for piece in pieces)
    assert pieces[0]['family'] == 'other_conversation' and pieces[0]['room_count'] == 1


def test_failing_room_observation_does_not_replace_rendered_text():
    from types import SimpleNamespace

    witness = RoomWitness(SimpleNamespace())
    lines = ['synthetic']
    assert witness.part(lines, None, [{}]) is lines  # malformed optional kind deliberately raises
    assert witness.observe(('late_reviews',), 'reviews') == 'reviews'
    assert witness.pieces[0]['start'] == 0
    assert witness.observe(('room',), 'first') == 'first'  # malformed optional key deliberately raises
    assert witness.observe(('marks',), 'later') == 'later'
    assert not witness.pieces  # no unwitnessed span is promoted to a memory fit note

    core = ContextCore(base_prompt='Guide', bible_md='Constitution', architecture_md='', development_md='',
                       semi_stable_text='', dynamic_text='', user_content_json='null', docs_need_development=False)
    governance = _governance_blocks(None, core, mode='max')
    room = 'reviews\n\nfirst\n\nlater'
    observed = []

    def trace(live):
        observed.extend(layout_witnesses(core, governance, {}, (), live, story_whole=True, room_pieces=witness.pieces))

    assert _system_blocks(core, governance, '', room, 'max', trace=trace) == _system_blocks(core, governance, '', room, 'max')
    assert observed[-1]['groups'] == [{'family': 'room', 'form': 'rendered', 'start': 0, 'chars': len(room)}]


def test_a_book_map_step_keeps_old_witnesses_off_the_changed_blocks():
    """A map step (``memory_floor.step_book_map``'s yield, taken through ``_address_optional``) changes a selected
    block's text. The stepped projection carries the witnesses of the bytes the producer actually rendered; the
    sealed request viewer re-checks them, so the changed block's old witness names no position in the stepped
    request while every block whose bytes survived stays located."""
    from ouroboros.context_assembly_trace import verified_layout
    from ouroboros.memory_floor import _address_optional

    design = HostSection('design_bible', '## Design bible\n\n' + 'Airy rule. ' * 400,
                          operative_view='## Design bible\n\nAiry.')
    core = ContextCore(base_prompt='Guide', bible_md='Constitution', architecture_md='Map text', development_md='Handbook',
                       semi_stable_text='', dynamic_text='', user_content_json='null', docs_need_development=True,
                       dynamic_sections=(HostSection('recent_tools', 'Tools ran'), design))
    governance = _governance_blocks(None, core, mode='low')
    system, captured = capture_system_projection(core, governance, 'Story', 'Room', 'low', {}, (), True,
                                                  shown=None, facts={'floor': {'steps': {}}})
    witnesses = projection_fields(system, captured)['assembly_witnesses']
    receipts, addressed = [], {}
    assert _address_optional(design, receipts, addressed, root='', task_id='') > 0  # the step's own change
    stepped, _ = capture_system_projection(core, governance, 'Story', 'Room', 'low', addressed, receipts, True,
                                            shown=None, facts={'floor': {'steps': {}}})
    assert stepped[-1]['text'] != system[-1]['text']  # the yield changed the selected head block only
    changed = [i for i in range(len(system)) if stepped[i]['text'] != system[i]['text']]
    assert changed == [len(system) - 1]
    carried = verified_layout({'messages': [{'role': 'system', 'content': stepped}]}, witnesses)
    by_block = {block['block']: block['status'] for block in carried}
    assert by_block[len(system) - 1] == 'position_unverified'  # the old witness names no position in the new text
    assert all(by_block[i] == 'located' for i in range(len(system) - 1))
    fresh = projection_fields(stepped, _)['assembly_witnesses']
    assert [row['carrier_sha256'] for row in fresh] == [digest(block['text']) for block in stepped]


def test_quiz_answer_attachment_observation_avoids_ambient_mail_source(tmp_path, monkeypatch):
    """An ordinary live Reply quiz answer stays a typed owner-words row (``protected_input``, never the
    ``secret_request_outcome`` form) and records genuine attachment_manifest observations on it, without
    forging an owner_input fact, borrowing an earlier mail row's ambient last_owner_delivery source
    address, or logging file bytes."""
    import base64
    import json
    import queue
    from types import SimpleNamespace

    from ouroboros.artifacts import stage_task_attachments
    from ouroboros.context_assembly_trace import bind_contribution_input, verified_contributions
    from ouroboros.llm_messages import project_wire_messages
    from ouroboros.loop_round_limits import _drain_incoming_messages
    from ouroboros.owner_mailbox import KIND_QUIZ_ANSWER, write_owner_message
    from ouroboros.owner_relay import owner_row_address, owner_row_source
    from ouroboros.physical_attempt_context import PhysicalAttemptContext
    from supervisor.message_ingress import log_chat

    root = tmp_path / 'quiz-trace'
    monkeypatch.setenv('OUROBOROS_DATA_DIR', str(root))
    task_id = 'quiz-trace-task'
    png = base64.b64decode(
        'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+a1V8AAAAASUVORK5CYII=')
    img_path = tmp_path / 'answer.png'
    img_path.write_bytes(png)
    doc_path = tmp_path / 'secret-note.txt'
    doc_path.write_text('PRIVATE_FILE_BYTES_NEVER_LOGGED', encoding='utf-8')
    staged = stage_task_attachments(root, task_id, [
        {'path': str(img_path), 'label': 'answer.png'},
        {'path': str(doc_path), 'label': 'secret-note.txt'},
    ])

    mail_row = log_chat('in', 1, 1, 'Ordinary mail first', ts='2026-11-25T00:00:00Z',
                        source='web', client_message_id='mail-1', drive_root=root, require_write=True)
    mail_addr = owner_row_address(mail_row)

    incoming = queue.Queue()
    incoming.put({
        'msg_id': 'direct-mail-1', 'client_message_id': 'mail-1', 'text': 'Ordinary mail first',
        'ts': mail_row['ts'], 'origin_message_ref': owner_row_source(mail_row)[0],
    })
    write_owner_message(
        root, '[Owner quiz answer] See attached reply', task_id, msg_id='quiz-reply-1', kind=KIND_QUIZ_ANSWER,
        quiz_delivery={'task_id': task_id, 'quiz_id': 'q-reply', 'canonical_root': str(root), 'answer_snapshot': {
            'quiz_id': 'q-reply', 'question': 'Any files?', 'options': ['A'],
            'comment': 'See attached reply', 'attachments': staged}},
        attachment_manifest=staged,
    )

    ctx = SimpleNamespace(drive_root=root, budget_drive_root=root, task_id=task_id, current_chat_id=1)
    messages = []
    _drain_incoming_messages(messages, incoming, root, task_id, None, set(), owner_ctx=ctx)
    assert len(messages) == 3  # ordinary mail, ordinary file-reply quiz answer, owner_quote_handles
    assert isinstance(messages[1]['content'], list)

    # The producer binds the trace to the rows as it typed them; ``project_wire_messages`` is only the
    # physical candidate, whose host stamps are stripped before the provider sees them. Binding to that
    # copy would lose the typed row family, so the two roles stay separate here as in the send path.
    wire = {'messages': project_wire_messages(messages)}
    assert '_host_context_kind' not in json.dumps(wire)
    fit = PhysicalAttemptContext('owner_max', 'max', 'cold_estimate', '', '', None, None, False, False)
    bound = bind_contribution_input(fit, {'messages': messages}, messages, ctx)
    entries = verified_contributions(wire, bound.contribution_input)['entries']
    assert 'PRIVATE_FILE_BYTES_NEVER_LOGGED' not in json.dumps(entries)

    by_path = {}
    for entry in entries:
        by_path.setdefault(entry.get('path'), []).append(entry)
    assert any(item['family'] == 'owner_input' and item.get('row_address') == mail_addr for item in by_path['$.messages[0]'])

    reply_notes = {item['family']: item for item in by_path['$.messages[1]']}
    assert 'owner_input' not in reply_notes
    assert reply_notes['protected_input']['form'] == 'typed_row'
    assert reply_notes['attachment_image']['form'] == 'pixels'
    assert reply_notes['attachment_image']['count'] == 1
    assert reply_notes['attachment_manifest']['manifest_items'] == 2
    assert reply_notes['attachment_manifest']['manifest_path_only'] == 1
    assert reply_notes['attachment_manifest']['manifest_pixels'] == 1
    assert reply_notes['attachment_manifest']['attachment_items'] == [
        {'number': 1, 'decision': 'prepared_pixels'},
        {'number': 2, 'decision': 'not_selected_as_image'},
    ]
    assert 'row_address' not in reply_notes['attachment_manifest']
    assert 'source_id_sha256' not in reply_notes['attachment_manifest']
