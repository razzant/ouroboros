#!/usr/bin/env python3
"""Measure one caller's initial context in a room, offline: no task, no provider, no write under the data root.

The measuring task carries its caller kind (`--caller direct`, one chat turn, the default; `root`, a queued
integrating task with two start work rooms; wake, Presence and helper callers are not modeled) and the room's
registered folder; `--session-provider` hands the same task and tool selection to the call host. Tools are
`initial_tool_schemas(mode)` of a standalone registry: built-in tools plus published MCP catalogs, never the live
extension-skill tools of a server, so the set may be smaller or larger than a real task's. The tools summary is the
list the plan measured (Nano's when the plan falls to Nano; a call's own tools counted apart), the requested list only
where they differ. `--reserve` is the route's reply ceiling (a call route's working reserve under `--session-provider`);
each mode applies its own reserve (Nano keeps its 8,192 floor). Figures are chars/4 estimates over the serialized
request at a cold density (`--ratio`, provider tokens per estimator token: 1.0 unless a route's measured figure is
declared), never provider tokens; the assignment text is empty and unmeasured. `--dump FILE` writes the start's text
(`start_text`), outside ROOT; the view's forms are counted by the law of forms (`view_counts`).

Usage: `python devtools/context_fit_report.py --data ROOT --room ID --window TOKENS --reserve TOKENS
[--caller direct|root]`; other options: `--mode max|low|nano`, `--json`, `--ratio R`, `--dump FILE`. `--session-provider
gemini_live|grok_realtime|openai_realtime` (direct caller only) also runs the real conversation host's tool selection
and session-candidate seam with bundled default model/options and cached output limit (call tools, driver
instructions, provider wire), never connecting. Output names missing inputs;
`devtools._context_fit_session.measure_session` measures a retained host. SQLite uses `mode=ro&immutable=1`; ROOT
writes are forbidden, while content-addressed blobs go to scratch so the measured context keeps the server's pointers.
A live or unknown PID/lock, or WAL, needs `--snapshot`: no liveness probe, WAL recovered in a private copy. Changing
inputs, stale indexes and external local skill sources are refused. Live process scanning is unavailable and adds no
text to the measured input; this estimates a fit, not provider acceptance.
"""
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


def start_text(messages):
    """The start's messages as the mind reads them: each one's text, its blocks joined by a newline (the system message
    first, as a saved request's ``system.txt``), the messages by a blank line."""
    def text(content):
        return content if isinstance(content, str) else '\n'.join(
            block.get('text', '') for block in content or () if isinstance(block, dict))
    return '\n\n'.join(text(message.get('content')) for message in messages)


def view_counts(snapshot, level):
    from devtools._context_fit_forms import view_counts as count
    return count(snapshot, level)


def measure_core(core, schemas, *, root, window, reserve, mode='max', prepared=None, fit=None,
                 route=None, route_facts=None, response_limit=None, task=None, ratio=1.0, dump=None):
    """Observe the existing fitter, including the last frame of a typed refusal.

    ``task`` is the task ``core`` was captured for: its caller kind sets the role and the start work room, so the
    plan reads the same one (default: a direct chat turn in the captured room). Without ``route_facts``, ``reserve``
    is a declared working reserve in every mode (a call route's geometry); ``ordinary_route`` declares a Main route.
    ``ratio`` is the route's cold density; ``dump`` a path the start's text is written to (``start_text``)."""
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
    with cold_density(ratio), patch.object(
            cf, '_system_blocks', capture_system), patch.object(mf, 'render_view_for_mode', capture_memory):
        try:
            if plan is None:
                plan = cf.build_context_fit_plan(SimpleNamespace(drive_root=root), core, task or {
                    'id': core.section_task_id or 'context-fit-report', 'chat_id': int(snapshot.spec.room_id),
                    '_is_direct_chat': True}, preferred_mode=mode, tool_schemas=schemas,
                    route_resolver=lambda *a, **kw: (route, evidence))
        except cf.MinimumViewUnfit as exc:
            failure = exc
    selected = plan.initial_mode if plan else frames[-1][0]
    sent = schemas if prepared is not None else list(select_tool_schemas(schemas, context_mode=selected).schemas)
    projection = plan.projection(selected) if plan else None
    book_mode = 'low' if selected == 'nano' else selected
    governance = cf._governance_blocks(None, core, mode=book_mode)
    if projection:
        facts = projection.memory_facts
        level = mf.kept_level(snapshot, facts)
        addressed = section_views((*core.semi_stable_sections, *core.dynamic_head_sections, *core.dynamic_sections, *governance[0]),
                                  facts.get('host_sections_addressed', ()))
        if fit is None:
            with cold_density(ratio):
                fit = physical_fit(plan, plan.messages_for(selected), sent, root=root)
        input_tokens = fit['measurement']['estimated_input_tokens']
    else:
        facts = memory[-1].facts if memory else {}
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
    if governance[1]:  # the whole handbook (its subject's in Max, or the task's own requirement in any mode), else Max's map
        add('DEVELOPMENT', governance[1], 'whole' if core.docs_need_development or core.development_required else 'orientation')
    for parts, fallback, label in ((core.semi_stable_sections, core.semi_stable_text, 'stable'),
                                   (core.dynamic_head_sections, core.dynamic_head_text, 'knowledge'),
                                   (core.dynamic_sections, core.dynamic_text, 'host')):
        for part in parts:
            add(part.section_id, addressed.get(part.section_id, part.full_text),
                'operative' if part.section_id in addressed else 'whole')
        if not parts and fallback:
            add(label, fallback)
    gone = dict(level.addressed)
    from ouroboros import memory_view_conversations as conversations
    counts = set(gone.get('F6a', ()))
    lined = {g['key'] for g in snapshot.conversations if conversations.group_form(g, gone)} | counts
    for form, groups in (
        ('count', [g for g in snapshot.conversations if g['key'] in counts]),
        ('line', [g for g in snapshot.conversations if g['key'] in lined and g['key'] not in counts]),
        ('words by address', [g for g in snapshot.conversations if g['key'] not in lined and any(
            w['id'] in gone.get('F6w', ()) for room in g['rooms'] for w in room['words'])]),
        ('whole', [g for g in snapshot.conversations if g['key'] not in lined and not any(
            w['id'] in gone.get('F6w', ()) for room in g['rooms'] for w in room['words'])]),
    ):
        add('other conversations', '\n\n'.join(conversations.pieces(groups, gone, lambda _key, make: make()))
            if groups else '', form, len(groups))  # each subset's own head and foot
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
    add('connections', core.connections[selected == 'nano'], 'by namespace' if selected == 'nano' else 'mcp servers')
    add('marks', mv._marks_text(list(snapshot.marks)) if snapshot.marks else '', count=len(snapshot.marks))
    blocks.append(dict(block='tools', form='selected', count=len(sent),  # a refusal's bound is plan-side, a candidate's physical
                       tokens=serialized_tool_tokens(plan.messages_for(selected), sent, plan.provider) if fit
                       else cf.tool_schema_tokens(sent)))
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
    result = dict(mode=selected, requested_mode=mode, density=ratio, window=capacity, reserve=actual_reserve,
        input_tokens=input_tokens, total_tokens=input_tokens + actual_reserve, margin_tokens=margin,
        verdict='MinimumViewUnfit' if failure else 'context_capacity' if fit.get('refusal') else 'fits',
        deficit_tokens=failure.deficit_tokens if failure else max(0, -margin),
        work_room_deficit_tokens=failure.work_room_deficit_tokens if failure else max(0, start_room - margin) if prepared is None else 0,
        ladder=dict(level.steps), ladder_basis='selected' if projection or memory else 'minimum bound before ladder',
        view={**view_counts(snapshot, level), 'work_room_tokens': margin, 'requested_work_room_tokens': requested_room},
        blocks=blocks, profile=profile, work_room_levels=snapshot.spec.work_room_levels,
        start_work_room_tokens=requested_room, work_room_applies_to_verdict=False,
        arithmetic={'requested_window_tokens': window, 'capacity_tokens': capacity, 'boundary_tokens': record.boundary,
                    'input_tokens': input_tokens, 'response_reserve_tokens': actual_reserve,
                    'room_after_reserve_tokens': margin, 'start_work_room_tokens': requested_room,
                    'start_input_allowance_tokens': budget['with_margin_tokens'],
                    'room_above_start_tokens': margin - requested_room,
                    'start_room_shortfall_tokens': max(0, requested_room - margin)},
        window_record=asdict(record), fit=fit, tools={'count': len(sent), 'by_surface': surface_counts(sent)},
        phase='prepared_candidate' if prepared is not None else 'task_preparation',
        input_basis='minimum_bound_not_a_candidate' if failure else 'measured_candidate',
        estimator=('estimate_context_prompt_tokens over the serialized messages and tools, chars/4' if fit else
                   'plan-side minimum bound (_request_tokens + tool_schema_tokens), chars/4'),
        diagnostic_notes=['No live host: feature tools/instructions, dynamic tool notices and provider wire are absent.',
                          f'Density is a cold estimate of {ratio}; no account-scoped window observations are inferred.',
                          'Figures are chars/4 estimates, never provider tokens: the input is the serialized request, section rows '
                          'are per-text estimates and the framing row is their signed remainder; the assignment text is empty.',
                          f"Reserve: {reserve} is declared as "
                          f"{'a working reserve' if 'working_reserve_tokens' in evidence.route_facts else 'the reply ceiling'}; "
                          f"the rendered {selected} mode applies {actual_reserve} "
                          "(Main's reply cap, then context_budget.context_mode_limits)."])
    if fit:
        result['density'] = fit['measurement']['measurement_density']
        result['reply_allowance'] = fit['measurement']['reply_allowance_tokens']
    else:
        result['diagnostic_notes'].append(
            'MinimumViewUnfit is a fresh-start refusal, not a session verdict; the input number is a lower bound, not a sent candidate.')
    if dump is not None and plan is None:
        result['diagnostic_notes'].append('No start text is written: the start was refused.')
    elif dump is not None:
        Path(dump).write_text(start_text(plan.messages_for(selected)), encoding='utf-8')
    return result


CALLERS = ('direct', 'root')


def ordinary_route(reply_ceiling):
    """``(route facts, response limit)`` of an ordinary Main route: its reply ceiling is declared and no working
    reserve (only a call provider declares one), so each mode takes its own reserve from the production owner."""
    return ({'basis': 'provider', 'acceptance_can_raise': False, 'source': 'diagnostic declaration'},
            {'max_output_tokens': reply_ceiling, 'status': 'asserted', 'source': 'diagnostic declaration'})


class Unmeasurable(ValueError):
    """A content-free reason this room cannot be measured offline, safe to print."""


def measuring_task(root, room, caller):
    """The facts production reads for this caller kind in this room; the assignment text is empty.

    A project room's registered folder travels as production carries it: the room lens of a direct turn, the
    workspace of a queued root. Folderless rooms can need scratch creation; unavailable attached folders keep
    their address and an availability note in production. This offline report models neither case.
    """
    from ouroboros.project_admission import reserved_project_for_chat
    from ouroboros.workspace_admission import room_chat_lens_dir, room_lens_applies

    if caller not in CALLERS:
        raise ValueError(f'caller is one of {", ".join(CALLERS)}')
    task = {'id': 'context-fit-report', 'chat_id': int(room), 'type': 'task', 'text': '',
            '_is_direct_chat': caller == 'direct'}
    project = reserved_project_for_chat(root, int(room))
    if project:
        task['project_id'] = project['id']
        folder, note = room_chat_lens_dir(root, project['id'])
        if not folder or note:
            raise Unmeasurable('This project room has no usable registered folder; '
                               'this read-only report cannot model its folder context')
        if room_lens_applies(task):
            task['metadata'] = {'_project_room_dir': folder}
        else:
            task.update(workspace_root=folder, workspace_mode='external')
    return task


def tool_context(drive_root, task):
    """The tool context a task of this shape runs under, so credential and project filters select as in production."""
    from ouroboros.tools.tool_context import ToolContext

    workspace = task.get('workspace_root')
    return ToolContext(repo_dir=REPO, drive_root=drive_root, system_repo_dir=REPO, task_id=task['id'],
                       current_chat_id=task['chat_id'], is_direct_chat=task['_is_direct_chat'],
                       project_id=task.get('project_id', ''), task_metadata=dict(task.get('metadata') or {}),
                       workspace_root=Path(workspace) if workspace else None, workspace_mode=task.get('workspace_mode', ''))


def serialized_tool_tokens(messages, tools, provider=''):
    """The tools member of the request on the fit's own physical estimator: the serialization it measures."""
    from ouroboros import context_fit as cf

    return (cf.estimate_context_prompt_tokens(messages, tools, provider=provider)
            - cf.estimate_context_prompt_tokens(messages, None, provider=provider))


def surface_counts(schemas, own=()):
    """Schemas by origin: built-in, extension and MCP namespaces; ``own`` names (a call's own tools) are counted apart."""
    from ouroboros.tool_policy import tool_namespace

    counts = dict.fromkeys(('builtin', 'extension', 'mcp'), 0)
    for schema in schemas:
        name = schema['function']['name']
        space = tool_namespace(name)
        key = 'call' if name in own else 'mcp' if space.startswith('mcp_') else 'extension' if space.startswith('ext_') else 'builtin'
        counts[key] = counts.get(key, 0) + 1
    return counts


def tool_facts(registry, result, *, requested_mode, requested_count):
    """The measured tools (``result['tools']``: the one list ``measure_core`` sent) with where they came from and what
    this process cannot see; no direction of error is implied. The requested-mode list is named only where the plan
    narrowed it."""
    measured, rendered = result['tools'], result['mode']
    if result['phase'] == 'session_candidate':  # the call host chose its own list, from its initial mode
        narrowed = False
        selector = (f'the call host: ouroboros.tool_policy.initial_tool_schemas(context_mode={rendered!r}, '
                    'schema_names=INITIAL_TOOLS) on a standalone ToolRegistry, plus the call tools')
    else:
        narrowed = requested_count != measured['count']
        frame = 'the refusal\'s last frame' if result['verdict'] == 'MinimumViewUnfit' else 'the plan'
        selector = (f'ouroboros.tool_policy.initial_tool_schemas(context_mode={requested_mode!r}) on a standalone ToolRegistry'
                    + (f'; {frame} is {rendered!r}, so the measured list is its select_tool_schemas' if narrowed else ''))
    return {**measured, 'selector': selector,
            **({'requested_mode': requested_mode, 'requested_count': requested_count} if narrowed else {}),
            'on_demand_withheld': len(registry.on_demand_schema_names()),
            'registry_omissions': [{'surface': row.get('surface'), 'reason': row.get('reason'),
                                    'count': len(row.get('tools') or ())} for row in registry.capability_omissions()],
            'verified_complete': False, 'absent': ['live extension-skill tools (a running server loads them)',
                                                   'MCP schemas of servers with no published catalog under the data root']}


def report(root, room, window, reserve, mode='max', snapshot=False, session_provider=None, caller='direct', ratio=1.0,
           dump=None):
    from devtools._context_fit_readonly import readonly_inputs, server_alive, snapshot_root
    from ouroboros.test_environment import default_temp_parent
    root = Path(root).resolve(strict=True)
    source_root = root
    if not snapshot and server_alive(root):  # --snapshot measures a private copy and never probes liveness
        raise ValueError('The data root has a live server or uncertain admission; use --snapshot')
    with tempfile.TemporaryDirectory(prefix='context-fit-', dir=default_temp_parent()) as directory:
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
                from ouroboros.memory_view import view_role
                from ouroboros.tool_policy import initial_tool_schemas
                from ouroboros.tools.registry import ToolRegistry
                env = SimpleNamespace(repo_dir=REPO, drive_root=root, repo_path=lambda path: REPO / path)
                task = measuring_task(root, room, caller)
                tools = ToolRegistry(REPO, root)
                tools.set_context(tool_context(root, task))
                core = context._capture_context_core(env, Memory(drive_root=root, repo_dir=REPO), task, None, None, tools)
                core = replace(core, section_root=str(scratch / 'captures'), section_task_id='context-fit-report')
                schemas = initial_tool_schemas(tools, context_mode=mode)
                if session_provider:
                    if caller != 'direct':
                        raise ValueError('A live call is a direct caller; --session-provider measures no other kind')
                    from devtools._context_fit_session import report_session
                    result = report_session(core, tools, root=root, scratch=scratch, window=window, reserve=reserve,
                                            mode=mode, provider=session_provider, task=task, schemas=schemas)
                else:
                    facts, limit = ordinary_route(reserve)
                    result = measure_core(core, schemas, root=root, window=window, reserve=reserve, mode=mode, task=task,
                                          route_facts=facts, response_limit=limit, ratio=ratio, dump=dump)
                result['caller'] = {'kind': caller, 'role': view_role(task), 'task_fields': sorted(
                    {*task, *(f'metadata.{key}' for key in task.get('metadata', ()))}
                    - {'id', 'chat_id', 'type', 'text', 'metadata'})}
                result['tools'] = tool_facts(tools, result, requested_mode=mode, requested_count=len(schemas))
                result['diagnostic_notes'].append(
                    'Tools come from this process, not a running server: the set can be smaller (extension skills absent) '
                    'or larger (a real task\'s contract, profile or credentials may withhold more).')
                result['snapshot'] = snapshot
                result['diagnostic_notes'].append('Live process scanning is unavailable; no scan notice enters the measured input.')
                return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--room', required=True, type=int)
    parser.add_argument('--window', type=int, required=True)
    parser.add_argument('--reserve', type=int, required=True,
                        help="the route's reply ceiling in tokens (Main's own cap is 65536; a call's working reserve with "
                        "--session-provider); each mode applies its own")
    parser.add_argument('--mode', choices=('max', 'low', 'nano'), default='max')
    parser.add_argument('--caller', choices=CALLERS, default='direct',
                        help='direct: one chat turn (default); root: a queued integrating task, which keeps two start work rooms')
    parser.add_argument('--json', action='store_true')
    parser.add_argument('--snapshot', action='store_true')
    parser.add_argument('--session-provider', choices=('gemini_live', 'grok_realtime', 'openai_realtime'),
                        help='also prepare a bundled provider session offline, with default call options (never connect)')
    parser.add_argument('--ratio', type=float, default=1.0,
                        help="the route's density, provider tokens per estimator token (chars/4), as its usage measured it")
    parser.add_argument('--dump', type=Path, help="write the start's text to this file (outside the data root)")
    args = parser.parse_args(argv)
    if args.window <= 0 or args.reserve <= 0 or args.ratio <= 0:
        parser.error('window, reserve and ratio must be positive')
    if args.session_provider and args.caller != 'direct':
        parser.error('--session-provider measures a live call, which is a direct caller')
    if args.session_provider and (args.dump or args.ratio != 1.0):
        parser.error('--dump and --ratio measure the ordinary start, not a call host')
    if args.dump and args.dump.resolve().is_relative_to(args.data.resolve()):
        parser.error('--dump writes outside the data root')
    previous_logging = logging.root.manager.disable
    previous_bytecode = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    logging.disable(logging.CRITICAL)  # Never echo private source text or settings through capture warnings.
    try:
        result = report(args.data, args.room, args.window, args.reserve, args.mode, args.snapshot, args.session_provider,
                        args.caller, args.ratio, args.dump)
    except Unmeasurable as exc:
        print(f'Context measurement unavailable: {exc}', file=sys.stderr)
        return 2
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
        print('View by the law of forms: ' + json.dumps(result['view'], ensure_ascii=False))
        print(f"Phase: {result['phase']}; profile: {result['profile']}; work-room levels: {result['work_room_levels']}")
        if 'caller' in result:
            tools = result['tools']
            narrowed = (f" in {result['mode']} (the requested {tools['requested_mode']} selection has {tools['requested_count']})"
                        if 'requested_count' in tools else '')
            print(f"Caller: {result['caller']['kind']} (role {result['caller']['role']}); tools: {tools['count']}{narrowed} "
                  f"({', '.join(f'{key} {value}' for key, value in tools['by_surface'].items())})")
        math = result['arithmetic']
        print(f"Room: {math['capacity_tokens']} - input {math['input_tokens']} - reserve {math['response_reserve_tokens']} "
              f"= {math['room_after_reserve_tokens']}; start room {math['start_work_room_tokens']}; "
              f"shortfall {math['start_room_shortfall_tokens']}; start input allowance {math['start_input_allowance_tokens']}")
        for note in result['diagnostic_notes']:
            print('Diagnostic: ' + note)
    return 0 if result['verdict'] == 'fits' else 1


if __name__ == '__main__':
    raise SystemExit(main())
