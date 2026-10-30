#!/usr/bin/env python3
"""Measure an ordinary room projection without starting a task or a provider."""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
import logging
import os
from pathlib import Path
import sqlite3
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def measure_core(core, schemas, *, root, window, reserve, mode='max', prepared=None, fit=None,
                 route=None, route_facts=None, response_limit=None):
    """Observe the existing fitter, including the last frame of a typed refusal."""
    from ouroboros import context_budget, context_fit as cf, memory_floor as mf, memory_view as mv
    from ouroboros.section_floor import section_views
    from ouroboros.tool_policy import select_tool_schemas
    from devtools._context_fit_accounting import declared_evidence, physical_fit, cold_density
    snapshot = mv.snapshot_from_json(core.memory_view_json)
    if not snapshot.active:
        raise ValueError('Memory capture is unavailable; a partial view cannot certify a fit')
    frames, memory = [], []
    system, render = cf._system_blocks, mf.render_view_for_mode

    def capture_system(*args, **kwargs):
        result = system(*args, **kwargs)
        frames.append((kwargs.get('mode') or (args[4] if len(args) > 4 else mode), kwargs.get('addressed', {})))
        return result

    def capture_memory(*args, **kwargs):
        result = render(*args, **kwargs)
        memory.append(result)
        return result

    evidence = declared_evidence(window, reserve, route_facts, response_limit)
    route = route or {'provider': 'diagnostic', 'model': 'diagnostic'}
    failure, plan = None, prepared
    with cold_density(), patch.object(
            cf, '_system_blocks', capture_system), patch.object(mf, 'render_view_for_mode', capture_memory):
        try:
            if plan is None:
                plan = cf.build_context_fit_plan(SimpleNamespace(drive_root=root), core,
                    {'id': core.section_task_id or 'context-fit-report', 'chat_id': int(snapshot.spec.room_id), '_is_direct_chat': True},
                    preferred_mode=mode, tool_schemas=schemas,
                    route_resolver=lambda *a, **kw: (route, evidence))
        except cf.MinimumViewUnfit as exc:
            failure = exc
    selected = plan.initial_mode if plan else frames[-1][0]
    projection = plan.projection(selected) if plan else None
    book_mode = 'low' if selected == 'nano' else selected
    governance = cf._governance_blocks(None, core, mode=book_mode)
    if projection:
        facts = projection.memory_facts
        level = mf.kept_level(snapshot, facts)
        addressed = section_views((*core.semi_stable_sections, *core.dynamic_head_sections, *core.dynamic_sections, *governance[0]),
                                  facts.get('host_sections_addressed', ()))
        sent = schemas if prepared is not None else list(select_tool_schemas(schemas, context_mode=selected).schemas)
        if fit is None:
            with cold_density():
                fit = physical_fit(plan, plan.messages_for(selected), sent, root=root)
        input_tokens = fit['measurement']['estimated_input_tokens']
    else:
        facts = memory[-1][2] if memory else {}
        level = mf.kept_level(snapshot, facts) if memory else mv.FloorLevel(tuple(
            (step, tuple(ident for s, ident, *_ in mf.floor_elements(snapshot) if s == step)) for step in mf.LADDER))
        addressed = frames[-1][1]
        input_tokens = failure.estimated_input_tokens
    blocks = []

    def add(name, text, form='whole', count=None):
        blocks.append(dict(block=name, form=form, tokens=mf.view_tokens(text), chars=len(text),
                           **({'count': count} if count is not None else {})))

    add('SYSTEM', core.base_prompt)
    add('BIBLE', core.bible_md)
    for part in governance[0][2:]:
        add(part.section_id, addressed.get(part.section_id, part.full_text),
            'operative' if part.section_id in addressed else 'orientation' if book_mode == 'low' else 'whole')
    if governance[1]:  # Max without the whole handbook carries its chapter map here
        add('DEVELOPMENT', governance[1], 'whole' if core.docs_need_development else 'orientation')
    for parts, fallback, label in ((core.semi_stable_sections, core.semi_stable_text, 'stable'),
                                   (core.dynamic_head_sections, core.dynamic_head_text, 'knowledge'),
                                   (core.dynamic_sections, core.dynamic_text, 'host')):
        for part in parts:
            add(part.section_id, addressed.get(part.section_id, part.full_text),
                'operative' if part.section_id in addressed else 'whole')
        if not parts and fallback:
            add(label, fallback)
    gone = dict(level.addressed)
    for form, rooms in (
        ('aggregate', [r for r in snapshot.live_rooms if r['room_id'] in {*gone.get('F1b', ()), *gone.get('F6a', ())}]),
        ('directory', [r for r in snapshot.live_rooms if r['room_id'] in gone.get('F6', ()) and r['room_id'] not in gone.get('F6a', ())]),
        ('range', [r for r in snapshot.live_rooms if r['room_id'] in gone.get('F6r', ()) and r['room_id'] not in gone.get('F6', ())]),
        ('whole', [r for r in snapshot.live_rooms if not any(r['room_id'] in gone.get(s, ()) for s in ('F1b', 'F6r', 'F6', 'F6a'))]),
    ):
        add('live rooms', mv._live_text(tuple(rooms), gone) if rooms else '', form, len(rooms))
    room = snapshot.room or {}
    notes = room.get('notes', [])
    short_notes = [note for note in notes if note['id'] in gone.get('F8', ())]
    for form, selected_notes in (('whole', [note for note in notes if note not in short_notes]),
                                 ('aggregate' if gone.get('F8a') else 'line', short_notes)):
        add('this room notes', '\n'.join(mv._room_note_lines(selected_notes, gone)), form, len(selected_notes))
    ids = set(gone.get('F2', ())) | set(gone.get('F7', ())) | set(gone.get('F7p', ()))
    rows = [row for row in room.get('lane1', ()) if row['address'] in ids and not row.get('immune')]
    add('this room addresses', mv._rows_line(room, rows) if rows and gone.get('F7a') else '\n'.join(
        mv._row_pointer(row, 'my reply' if row['kind'] == 'ouroboros' else 'words') for row in rows),
        'aggregate' if gone.get('F7a') else 'pointers', len(rows))
    verbatim = [row for row in room.get('lane1', ()) if row not in rows]
    add('this room verbatim', '\n'.join(row['line'] for row in verbatim), count=len(verbatim))
    facts_short = [row for row in room.get('lane2', ()) if row['first'] in gone.get('F1', ())]
    facts_whole = [row for row in room.get('lane2', ()) if row not in facts_short]
    add('this room task facts', mv._facts_line(room, facts_short) if facts_short else '', 'aggregate', len(facts_short))
    add('this room task facts', '\n'.join(row['line'] for row in facts_whole), 'whole', len(facts_whole))
    pages = room.get('under_parts', ())
    pointers = sum(row['id'] in gone.get('F4', ()) for row in pages)
    add('this room pages', '\n'.join(mv._retold(row, row['id'] in gone.get('F4', ())) for row in pages),
        'pointers' if pointers == len(pages) and pages else 'mixed' if pointers else 'whole', len(pages))
    visible = [entry for entry in snapshot.story if not entry.get('told_by')]
    forms = {}
    for entry in visible:
        form = ('accounts whole' if entry['kind'] == 'account' else
                'aggregate' if entry['id'] in gone.get('F5', ()) and gone.get('F5a') else
                'pointers' if entry['id'] in gone.get('F5', ()) else
                'first sentence' if entry['id'] in gone.get('F5l', ()) else
                'retold pointer' if entry['kind'] == 'legacy' and entry['room_id'] in gone.get('F3', ()) else 'whole')
        forms[form] = forms.get(form, 0) + 1
    add('story', mv.render_story(snapshot, level), '+'.join(forms) or 'whole', len(visible))
    blocks[-1]['forms'] = forms
    add('marks', mv._marks_text(list(snapshot.marks)) if snapshot.marks else '', count=len(snapshot.marks))
    sent = schemas if prepared is not None else list(select_tool_schemas(schemas, context_mode=selected).schemas)
    blocks.append(dict(block='tools', form='selected', tokens=cf.tool_schema_tokens(sent), count=len(sent)))
    blocks.append(dict(block='headings, floor note and envelope', form='framing',
                       tokens=input_tokens - sum(row['tokens'] for row in blocks)))
    actual_reserve = failure.response_reserve_tokens if failure else fit['measurement']['response_reserve_tokens']
    capacity = window if failure else fit['measurement']['capacity_total_tokens']
    margin = capacity - actual_reserve - input_tokens
    profile = fit['measurement']['profile'] if fit else ('owner_' if selected == mode else 'task_local_') + selected
    record = replace(evidence.window_record, reserve=actual_reserve) if failure else cf.resolve_window(
        plan, actual_reserve, root=root, bound=plan.window_witness_bound)
    target = context_budget.context_mode_limits(selected, mode, actual_reserve)[0]
    budget = context_budget.request_context_budget(window_tokens=capacity, output_reserve_tokens=actual_reserve,
        non_memory_tokens=0, target_tokens=target, margin_count=snapshot.spec.work_room_levels,
        boundary_tokens=record.boundary)
    start_room = budget['working_margin_tokens']
    requested_room = start_room if prepared is None else 0  # a renewal asks for no new start room (owner 5A)
    result = dict(mode=selected, requested_mode=mode, density=1.0, window=capacity, reserve=actual_reserve,
        input_tokens=input_tokens, total_tokens=input_tokens + actual_reserve, margin_tokens=margin,
        verdict='MinimumViewUnfit' if failure else 'context_capacity' if fit.get('refusal') else 'fits',
        deficit_tokens=failure.deficit_tokens if failure else max(0, -margin),
        work_room_deficit_tokens=failure.work_room_deficit_tokens if failure else max(0, start_room - margin) if prepared is None else 0,
        ladder=dict(level.steps), ladder_basis='selected' if projection or memory else 'minimum bound before ladder',
        blocks=blocks, profile=profile, work_room_levels=snapshot.spec.work_room_levels,
        start_work_room_tokens=requested_room, work_room_applies_to_verdict=False,
        arithmetic={'requested_window_tokens': window, 'capacity_tokens': capacity, 'boundary_tokens': record.boundary,
                    'input_tokens': input_tokens, 'response_reserve_tokens': actual_reserve,
                    'room_after_reserve_tokens': margin, 'start_work_room_tokens': requested_room,
                    'start_input_allowance_tokens': budget['with_margin_tokens'],
                    'room_above_start_tokens': margin - requested_room,
                    'start_room_shortfall_tokens': max(0, requested_room - margin)},
        window_record=asdict(record), fit=fit,
        phase='prepared_candidate' if prepared is not None else 'task_preparation',
        input_basis='minimum_bound_not_a_candidate' if failure else 'measured_candidate',
        diagnostic_notes=['No live host: feature tools/instructions, dynamic tool notices and provider wire are absent.',
                          'Density is fixed at 1.0; no account-scoped window observations are inferred.'])
    if fit:
        result['density'] = fit['measurement']['measurement_density']
        result['reply_allowance'] = fit['measurement']['reply_allowance_tokens']
    else:
        result['diagnostic_notes'].append(
            'MinimumViewUnfit is a fresh-start refusal, not a session verdict; the input number is a lower bound, not a sent candidate.')
    return result


def report(root, room, window, reserve, mode='max', snapshot=False, session_provider=None):
    from devtools._context_fit_readonly import readonly_inputs, server_alive, snapshot_root
    root = Path(root).resolve(strict=True)
    source_root = root
    if server_alive(root) and not snapshot:
        raise ValueError('The data root has a live server or uncertain admission; use --snapshot')
    with tempfile.TemporaryDirectory(prefix='context-fit-', dir='/tmp') as directory:
        scratch = Path(directory)
        if snapshot:
            copied = scratch / 'data'
            copied.mkdir()
            snapshot_root(root, copied)
            root = copied
        with patch.dict(os.environ, {'OUROBOROS_DATA_DIR': str(root), 'OUROBOROS_SETTINGS_PATH': str(root / 'settings.json')}):
            with readonly_inputs(root, source_root=source_root, scratch_root=scratch / 'observability'):
                from ouroboros import context
                from ouroboros.memory import Memory
                from ouroboros.tools.registry import ToolRegistry
                from ouroboros.project_admission import reserved_project_for_chat
                env = SimpleNamespace(repo_dir=REPO, drive_root=root, repo_path=lambda path: REPO / path)
                task = {'id': 'context-fit-report', 'chat_id': int(room), '_is_direct_chat': True, 'type': 'task', 'text': ''}
                project = reserved_project_for_chat(root, int(room))
                if project:
                    task['project_id'] = project['id']
                core = context._capture_context_core(env, Memory(drive_root=root, repo_dir=REPO), task, None, None)
                core = replace(core, section_root=str(scratch / 'captures'), section_task_id='context-fit-report')
                tools = ToolRegistry(REPO, root)
                if session_provider:
                    from devtools._context_fit_session import report_session
                    result = report_session(core, tools, root=root, scratch=scratch, window=window,
                                            reserve=reserve, mode=mode, provider=session_provider)
                else:
                    result = measure_core(core, tools.schemas(), root=root, window=window, reserve=reserve, mode=mode)
                result['snapshot'] = snapshot
                result['diagnostic_notes'].append('Live process scanning is unavailable; no scan notice enters the measured input.')
                return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--room', required=True, type=int)
    parser.add_argument('--window', type=int, required=True)
    parser.add_argument('--reserve', type=int, required=True)
    parser.add_argument('--mode', choices=('max', 'low', 'nano'), default='max')
    parser.add_argument('--json', action='store_true')
    parser.add_argument('--snapshot', action='store_true')
    parser.add_argument('--session-provider', choices=('gemini_live', 'grok_realtime', 'openai_realtime'),
                        help='also prepare a bundled provider session offline, with default call options (never connect)')
    args = parser.parse_args(argv)
    if args.window <= 0 or args.reserve <= 0:
        parser.error('window and reserve must be positive')
    previous_logging = logging.root.manager.disable
    previous_bytecode = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    logging.disable(logging.CRITICAL)  # Never echo private source text or settings through capture warnings.
    try:
        result = report(args.data, args.room, args.window, args.reserve, args.mode, args.snapshot, args.session_provider)
    except (OSError, ValueError, sqlite3.Error) as exc:
        print(f'Context measurement unavailable ({type(exc).__name__}). Check offline stores or use --snapshot.', file=sys.stderr)
        return 2
    finally:
        logging.disable(previous_logging)
        sys.dont_write_bytecode = previous_bytecode
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print('Block | Form | Tokens | Count')
        for row in result['blocks']:
            print(f"{row['block']} | {row['form']} | {row['tokens']} | {row.get('count', '')}")
        print(f"{result['mode']}: input {result['input_tokens']} + reserve {result['reserve']} / {result['window']}; "
              f"{result['verdict']}, margin {result['margin_tokens']}, work-room deficit {result['work_room_deficit_tokens']}")
        print('Ladder (' + result['ladder_basis'] + '): ' + json.dumps(result['ladder']))
        print(f"Phase: {result['phase']}; profile: {result['profile']}; work-room levels: {result['work_room_levels']}")
        math = result['arithmetic']
        print(f"Room: {math['capacity_tokens']} - input {math['input_tokens']} - reserve {math['response_reserve_tokens']} "
              f"= {math['room_after_reserve_tokens']}; start room {math['start_work_room_tokens']}; "
              f"shortfall {math['start_room_shortfall_tokens']}; start input allowance {math['start_input_allowance_tokens']}")
        for note in result['diagnostic_notes']:
            print('Diagnostic: ' + note)
    return 0 if result['verdict'] == 'fits' else 1


if __name__ == '__main__':
    raise SystemExit(main())
