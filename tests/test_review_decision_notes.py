"""Source-bound actor notes shorten complete review representations, not authority.

The plan-family histories are earlier-schema records the retired plan-review engine left,
recorded from its real flows (``tests/fixtures/legacy_plan_dispute``); the commit/change
family runs its real producers with only reviewer dispatch stubbed.
"""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from ouroboros import review_history_view as view
from ouroboros.artifacts import read_actor_source_bytes
from ouroboros.task_store import read_document
from ouroboros.tools.compact_context import _compact_context, record_context_view
from tests import _legacy_plan_shared as legacy
from tests.test_plan_dispute_history import _state
from tests.test_review_view_integration import _apply, _capture, _index, harness as _harness, staged_body as _staged_body

harness, staged_body = _harness, _staged_body

LONG = ('The existing route covers every requirement exactly. ' * 1000)[:41600]
NOTE = 'The recorded decisions remain; I retain their concise reasons and exact sources.'


def text_paths(value, needle, path=()):
    if isinstance(value, dict):
        return [p for key, item in value.items() for p in text_paths(item, needle, (*path, key))]
    if isinstance(value, list):
        return [p for i, item in enumerate(value) for p in text_paths(item, needle, (*path, i))]
    if isinstance(value, str) and needle in value:
        for raw in (value, value.partition('\n')[2]):
            try:
                parsed = json.loads(raw)
            except (ValueError, TypeError):
                continue
            if not isinstance(parsed, str):
                return text_paths(parsed, needle, (*path, 'json'))
        return [path]
    return []


def notes_for(ctx, *, kind=None, reason='The tested route already covers the requirement.'):
    options = view.review_note_options(ctx)
    return [{'bound_decision': e['bound_decision'], 'remark': 'Keep the exact source route.', 'reason': reason}
            for e in options['entries'] if e['bound_decision'] and (kind is None or e['decision_kind'] == kind)]


def first_plan(harness):
    """The retired engine's one closed round whose single s1 note ('Consider a duplicate cache')
    the author rejected with LONG, as recorded (``first_plan``). Returns the context."""
    legacy.plant_recorded(harness.drive, 'first_plan', workspace=harness.workspace)
    return harness.make_ctx()


def test_long_plan_reason_shortens_in_actual_messages_and_cold_view(harness):
    ctx = first_plan(harness)
    sub = harness.install({})
    before_state = deepcopy(_state(harness))
    history, operative = view.current_plan_history(ctx)
    original = _capture(ctx)
    assert LONG in str(original)
    record_context_view(ctx, original, [])
    inspected = json.loads(_compact_context(ctx, inspect=True))
    assert inspected['review_decisions']['entries']
    notes = notes_for(ctx)
    after, receipt, _ = _apply(ctx, original, note=NOTE, review_notes=notes)
    assert receipt['status'] == 'applied', receipt
    assert receipt['review_notes']['applied'] and not receipt['review_notes']['unshortened']
    assert LONG not in json.dumps(after) and notes[-1]['reason'] in json.dumps(after)
    assert _state(harness) == before_state and sub.calls == []
    assert _index(after)[-1]['operative_subject']['spec'] == operative['spec']
    old = json.loads(read_actor_source_bytes(ctx.drive_root, ctx.task_id, receipt['checkpoint_ref']))
    assert LONG in str(old['messages'])
    assert LONG not in str(_capture(harness.make_ctx()))
    assert view.current_plan_history(ctx)[0]['rounds'][0]['source'] == history['rounds'][0]['source']


def test_absent_and_stale_notes_keep_only_the_unselected_entries_full(harness):
    ctx = first_plan(harness)
    history, operative = view.current_plan_history(ctx)
    entries = view.decision_entries(history)
    assert len(entries) >= 2
    note = notes_for(ctx)[-1]
    projected, applied, missing = view.project_decision_notes(history, [note])
    assert len(applied) == 1 and len(missing) == len(entries) - 1
    assert LONG not in str(projected)
    untouched = [e for e in entries if e['bound_decision'] != note['bound_decision']]
    assert all(projected['decision_rows'][e['index']] == e['row'] for e in untouched)
    stale = deepcopy(note)
    stale['bound_decision']['content_sha256'] = '0' * 64
    full, applied, missing = view.project_decision_notes(history, [stale])
    assert full == history and not applied
    assert any(e['reason'] == 'stale_or_unknown_decision' for e in missing)
    # The invalid entry does not stop generic own-note compaction or source retention.
    after, receipt, _ = _apply(ctx, _capture(ctx), review_notes=[stale])
    assert receipt['status'] == 'applied' and LONG in str(after)
    assert any(e['reason'] == 'stale_or_unknown_decision' for e in receipt['review_notes']['unshortened'])


@pytest.mark.parametrize('surface', ['commit', 'change'])
def test_real_commit_and_change_notes_keep_sources_status_and_next_packet(staged_body, tmp_path, monkeypatch, surface):
    from ouroboros import capability_evidence, reviewer_window, review_substrate, review_ledger
    from ouroboros.review_history import review_dispute_history
    from ouroboros.tools.registry import ToolContext
    from tests.test_review_cold_history import _run, _dispatch, TASK
    from tests.test_review_change_end_to_end import _brief_text
    from tests import _contributor_packet_shared as shared

    monkeypatch.setattr(capability_evidence, 'probe', lambda *a, **k: None)
    monkeypatch.setattr(reviewer_window, 'reviewer_context_window', lambda *a, **k: 1_000_000)
    root, drive = Path(staged_body['repo']), tmp_path / 'history-data'
    briefs = []
    monkeypatch.setattr(review_substrate, 'run_review_request', _dispatch(briefs, 'Recorded critic recommendation.'))
    ctx = ToolContext(repo_dir=root, drive_root=drive, task_id=TASK)
    rid = _run(ctx, surface, '')
    review_ledger.note_author_decision(drive, rid, {'disposition': 'rejected', 'rationale': LONG})
    canonical = deepcopy(review_ledger.load_record(drive, rid))
    notes = notes_for(ctx)
    after, receipt, _ = _apply(ctx, _capture(ctx), review_notes=notes)
    assert receipt['status'] == 'applied', receipt
    assert LONG not in str(after) and notes[-1]['reason'] in str(after)
    assert review_ledger.load_record(drive, rid) == canonical
    original = review_dispute_history(drive_root=drive, repo_root=root, task_id=TASK)
    source = original['rounds'][0]['author_decisions'][-1]['source_ref']
    assert LONG in review_ledger.read_source(drive, TASK, source).decode()
    assert LONG not in str(_capture(ToolContext(repo_dir=root, drive_root=drive, task_id=TASK)))
    (root / 'ouroboros/tools/review.py').write_text("RULES = 'new independent version'\n", encoding='utf-8')
    shared.git(root, 'add', '-A')
    briefs.clear()
    _run(ToolContext(repo_dir=root, drive_root=drive, task_id=TASK), surface, '')
    assert briefs and all(LONG not in _brief_text(b) and notes[-1]['reason'] in _brief_text(b) for b in briefs)
    assert review_ledger.load_record(drive, rid) == canonical


def test_selected_author_reason_and_runtime_authority_mirrors_are_views_only(harness):
    from ouroboros.task_results import plan_review_authority_core
    # first_plan, then the author's recorded selection over that round with the reason below.
    legacy.plant_recorded(harness.drive, 'first_plan_author_selected', workspace=harness.workspace)
    ctx = harness.make_ctx()
    sub = harness.install({})
    reason = ('This selected author reason covers the exact prior concern. ' * 100).strip()
    canonical = deepcopy(_state(harness))
    original = _capture(ctx)
    assert reason in str(original) and LONG in str(original)
    # Real continuation authority copies retain the same immutable source binding.
    prefix = json.loads(original[0]['content'])
    core = plan_review_authority_core(canonical, source_ref={'task_id': ctx.task_id})
    prefix['predecessor_authority'] = {'plan_review_state': core}
    prefix['task_contract'] = {'predecessor_authority': {'plan_review_state': deepcopy(core)}}
    raw_prefix = deepcopy(prefix)
    prefix, _ = view.capture_review_history_messages({**prefix, 'plan_review_authority': {
        **prefix['plan_review_authority'], 'dispute_history': view.current_plan_history(ctx)[0],
        'operative_subject': view.current_plan_history(ctx)[1]}}, task_id=ctx.task_id, drive_root=ctx.drive_root)
    assert prefix != raw_prefix and core == raw_prefix['predecessor_authority']['plan_review_state']
    original[0]['content'] = json.dumps(prefix)
    after, receipt, _ = _apply(ctx, original, review_notes=notes_for(ctx))
    assert receipt['status'] == 'applied', receipt
    assert reason not in str(after) and LONG not in str(after)
    assert _state(harness) == canonical and sub.calls == []
    assert _index(after)[-1]['operative_subject']['spec'] == view.current_plan_history(ctx)[1]['spec']
    assert reason not in str(_capture(harness.make_ctx()))
    selected = json.loads(read_actor_source_bytes(ctx.drive_root, ctx.task_id, canonical['author_history_head']))
    assert selected['author_disposition']['rationale'] == reason


@pytest.mark.parametrize('surface,severity', [('commit', 'advisory'), ('change', 'advisory'), ('change', 'critical')])
def test_structured_critic_answer_shrinks_known_mirrors_but_keeps_typed_ids(staged_body, tmp_path, monkeypatch, surface, severity):
    from ouroboros import capability_evidence, reviewer_window, review_substrate, review_ledger
    from ouroboros.tools.registry import ToolContext
    from tests.test_review_cold_history import _run, _dispatch, TASK
    monkeypatch.setattr(capability_evidence, 'probe', lambda *a, **k: None)
    monkeypatch.setattr(reviewer_window, 'reviewer_context_window', lambda *a, **k: 1_000_000)
    dispatch = _dispatch([], LONG)
    def advisory(request, **kwargs):
        result = dispatch(request, **kwargs)
        for actor in result.actors:
            value = json.loads(actor['raw_text'])
            change = value if isinstance(value, list) else value['change']
            if change:
                change[0].update(verdict='FAIL', severity=severity)
            actor['raw_text'] = json.dumps(value)
        return result
    monkeypatch.setattr(review_substrate, 'run_review_request', advisory)
    ctx = ToolContext(repo_dir=Path(staged_body['repo']), drive_root=tmp_path / 'data', task_id=TASK)
    rid = _run(ctx, surface, '')
    canonical = deepcopy(review_ledger.load_record(ctx.drive_root, rid))
    from ouroboros.review_history import review_dispute_history
    history = review_dispute_history(drive_root=ctx.drive_root, repo_root=ctx.repo_dir, task_id=TASK)
    assert LONG in str(history['decision_rows']) and LONG in str(history['rounds'][0]['reviewers'])
    after, receipt, _ = _apply(ctx, _capture(ctx), review_notes=notes_for(ctx))
    assert receipt['status'] == 'applied'
    assert LONG not in str(after), text_paths(after, LONG)
    assert review_ledger.load_record(ctx.drive_root, rid) == canonical
    assert LONG not in str(_capture(ctx))
    assert canonical['verdict']['aggregate'] == ('FAIL' if severity == 'critical' else 'PASS')
    assert any(LONG in review_ledger.read_source(ctx.drive_root, TASK, seat['response']['source']).decode('utf-8')
               for seat in history['rounds'][0]['reviewers'] if seat.get('response'))
    shown = _index(after)[-1]
    # The record's exact answers and verdict mirrors are behind its one address; each seat's part keeps its
    # typed identity and recorded statuses in its decision row beside the actor's note.
    (entry,) = [e for e in shown['addresses'].values() if e['read'].get('tool') == 'query_data']
    assert entry['read']['arguments']['ids'] == [rid] and 'rounds' not in shown
    assert f"aggregate verdict {history['rounds'][0]['verdict']['aggregate']}" in entry['what']
    for old in history['rounds'][0]['reviewers']:
        for part, answer in old['answers'].items():
            (row,) = [r for r in shown['decision_rows'] if r.get('decision_kind') == 'review_part'
                      and r.get('seat_id') == old['seat_id'] and r.get('part') == part]
            assert row['status'] == {'recorded_verdict': answer['verdict'], 'response_status': answer['status']}
            assert row['authored_view']['authorship'] == 'actor' and LONG not in json.dumps(row)
            assert shown['addresses'][row['source']]['what'].startswith(f"answer of seat {old['seat_id']}")
    record_context_view(ctx, after, [])
    inspected = json.loads(_compact_context(ctx, inspect=True))
    assert all(e['representation'] == 'actor_review_note' for e in inspected['review_decisions']['entries'])


def test_two_decisions_partial_selection_stale_entry_and_new_finding_stay_local(harness):
    # One closed round whose two s1 notes the author rejected with the reasons below, as recorded.
    legacy.plant_recorded(harness.drive, 'two_decisions_rejected', workspace=harness.workspace)
    ctx = harness.make_ctx()
    wave = _state(harness)['waves'][-1]
    reasons = [LONG, 'An independent reason for the second concern. ' * 700]
    canonical = deepcopy(_state(harness))
    notes = notes_for(ctx, kind='plan_finding')
    entries = view.review_note_options(ctx)['entries']
    selected = {json.dumps(e['bound_decision'], sort_keys=True) for e in entries
                if e['finding_id'] == wave['findings'][0]['finding_id']}
    assert len(selected) >= 1
    for note in notes:
        if json.dumps(note['bound_decision'], sort_keys=True) not in selected:
            note['bound_decision']['content_sha256'] = '0' * 64
    after, receipt, _ = _apply(ctx, _capture(ctx), review_notes=notes)
    assert receipt['status'] == 'applied' and len(receipt['review_notes']['applied']) == len(selected)
    assert reasons[0] not in str(after) and reasons[1] in str(after)
    assert _state(harness) == canonical
    assert any(e['reason'] == 'stale_or_unknown_decision' for e in receipt['review_notes']['unshortened'])
    # The next recorded round: one new s1 note, NEW_UNCOVERED_POINT.
    legacy.advance_recorded(ctx.drive_root, 'two_decisions_then_new_point', workspace=harness.workspace)
    cold = _capture(harness.make_ctx())
    assert reasons[0] not in str(cold) and reasons[1] in str(cold) and 'NEW_UNCOVERED_POINT' in str(cold)
    assert any(e['representation'] == 'full' for e in view.review_note_options(ctx)['entries'])


def _replace_hot_wave(root, task_id, wave):
    """Store WAVE in place of the stored wave with its fingerprint and name it the open current
    attempt: the stored state the retired engine's wave writer left for such a replacement
    (compared equal to that writer's result for this test's wave while the writer existed)."""
    state = read_document(root, task_id)['plan_review_state']
    fingerprint = wave['request_fingerprint']
    state = {**state, 'waves': [*[w for w in state['waves'] if w['request_fingerprint'] != fingerprint], wave],
             'current_attempt': {'fingerprint': fingerprint, 'status': 'open', 'reason': ''}}
    legacy.seed_plan_review_state(root, task_id, state)


def test_closure_carried_findings_and_unknown_extension_are_accounted_for(harness):
    from ouroboros.tools import plan_review_artifacts as artifacts
    ctx = first_plan(harness)
    wave = artifacts.authority_wave(ctx.drive_root, ctx.task_id, _state(harness)['waves'][-1])
    wave['closure_notes'] = ['The precise recorded closure rationale. ' * 700]
    wave['actors'][0]['carried_findings'] = deepcopy(wave['findings'])
    wave['reviewer_outputs'][0]['future_review_extension'] = {'reason': 'KEEP_UNKNOWN_EXTENSION_EXACT'}
    wave['wave_artifact'] = legacy.persist_wave(ctx.drive_root, ctx.task_id, wave)
    _replace_hot_wave(ctx.drive_root, ctx.task_id, wave)
    history, _ = view.current_plan_history(ctx)
    canonical = deepcopy(_state(harness))
    notes = notes_for(ctx)
    assert any(e['decision_kind'] == 'plan_closure' for e in view.review_note_options(ctx)['entries'])
    after, receipt, _ = _apply(ctx, _capture(ctx), review_notes=notes)
    assert receipt['status'] == 'applied' and LONG not in str(after)
    assert wave['closure_notes'][0] not in str(after)
    assert any(p['path'][-1] == 'future_review_extension' for p in receipt['review_notes']['preserved_fields'])
    assert _state(harness) == canonical
    # The unknown extension and the carried findings travel by the wave's one address, which resolves exactly.
    shown = _index(after)[-1]
    assert 'KEEP_UNKNOWN_EXTENSION_EXACT' not in str(shown) and 'rounds' not in shown
    identity = {k: wave['wave_artifact'][k] for k in ('kind', 'root', 'path', 'size', 'sha256')}
    entry = next(e for e in shown['addresses'].values() if e.get('source_ref') == identity)
    assert entry['what'].startswith('plan review wave') and entry['read']['arguments']['path'] == identity['path']
    source = json.loads(read_actor_source_bytes(ctx.drive_root, ctx.task_id, entry['source_ref']))
    assert source['reviewer_outputs'][0]['future_review_extension'] == {'reason': 'KEEP_UNKNOWN_EXTENSION_EXACT'}
    old = history['rounds'][-1]['reviewers'][0]['carried_findings'][0]
    assert all(source['actors'][0]['carried_findings'][0][k] == old[k] for k in ('finding_id', 'class', 'breaks'))
    assert source['closure_notes'] == wave['closure_notes']


def test_ordinary_note_discloses_full_decisions_and_lost_selection_restores_them(harness):
    from ouroboros.artifacts import task_artifact_dir_path
    ctx = first_plan(harness)
    sub = harness.install({})
    after, receipt, _ = _apply(ctx, _capture(ctx))
    assert receipt['status'] == 'applied' and LONG in str(after)
    assert not receipt['review_notes']['applied'] and receipt['review_notes']['unshortened']
    short, receipt, _ = _apply(ctx, after, review_notes=notes_for(ctx))
    assert receipt['status'] == 'applied' and LONG not in str(short)
    ref = receipt['selected_review_history_view']['source_ref']
    (task_artifact_dir_path(ctx.drive_root, ctx.task_id, create=False) / ref['path']).unlink()
    cold = _capture(harness.make_ctx())
    assert LONG in str(cold) and 'REVIEW_HISTORY_VIEW_SOURCE_UNAVAILABLE' in str(cold)
    assert sub.calls == []


def test_actual_inspect_pair_remains_in_newer_tail_without_echoing_original_reason(harness):
    from types import SimpleNamespace
    from ouroboros.loop_tool_execution import process_tool_results
    from tests.test_main_authored_context import call
    ctx = first_plan(harness)
    messages = _capture(ctx)
    ctx.messages, ctx.active_context_mode = messages, 'max'
    record_context_view(ctx, messages, [])
    response = _compact_context(ctx, inspect=True)
    inspection = json.loads(response)
    assert LONG not in response
    assert inspection['review_decisions']['entries']
    messages.append(call('compact_context', {'inspect': True}, 'inspect'))
    process_tool_results([{'fn_name': 'compact_context', 'is_error': False, 'tool_call_id': 'inspect',
        'result': response, 'args_for_log': {'inspect': True}, 'trace_ref': {}}], messages,
        {'tool_calls': []}, lambda _: None, SimpleNamespace(_ctx=ctx), tool_schemas=[],
        fit_candidate=lambda *a: {'accepted': True, 'strict_bound_proven': False})
    inspect_pair = deepcopy(messages[-2:])
    assert inspect_pair[0]['tool_calls'][0]['id'] == 'inspect'
    assert inspect_pair[1]['tool_call_id'] == 'inspect'
    notes = [{'bound_decision': e['bound_decision'], 'remark': 'The source route remains.',
              'reason': 'The original concern is already covered.'}
             for e in inspection['review_decisions']['entries'] if e['bound_decision']]
    after, receipt, _ = _apply(ctx, messages, review_notes=notes,
                              expected_view_revision=inspection['view_revision'])
    assert receipt['status'] == 'applied', receipt
    assert all(row in after for row in inspect_pair), 'newer inspector pair was discarded'
    assert LONG not in str(after), text_paths(after, LONG)


@pytest.mark.parametrize('known_renderer', [True, False])
def test_discarded_answer_uses_exact_producer_mirror_unknown_renderer_stays_full(known_renderer):
    import hashlib
    item = {'item': 'D01->D02', 'reason': LONG, 'verdict': 'FAIL', 'severity': 'critical', 'model': 'fixture'}
    answer = {'status': 'unanswered', 'verdict': '', 'critical': 0, 'coverage': 'missing',
              'error': 'matrix invalid', 'findings': [], 'discarded': [item]}
    raw = json.dumps(answer).encode('utf-8')
    source = {'ref': {'kind': 'task_source', 'root': 'artifact_store', 'path': 'source_handles/critic.json',
                      'size': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}}
    mirror = {'severity': 'advisory', 'item': item['item'], 'verdict': 'FAIL', 'model': 'fixture',
              'tag': 'triad' if known_renderer else 'future_renderer',
              'reason': "not counted (coupling answer unanswered: matrix invalid); the seat's critical FAIL said: " + LONG}
    history = {'rounds': [{'review_record_id': 'record', 'verdict': {'aggregate': 'UNANSWERED', 'advisory_findings': [mirror]},
        'reviewers': [{'seat_id': 'seat', 'requested': {'model': 'fixture'}, 'response': {'source': source},
                       'answers': {'coupling': answer}}]}], 'decision_rows': [{
        'decision_kind': 'review_part', 'review_record_id': 'record', 'seat_id': 'seat', 'part': 'coupling',
        'status': {'recorded_verdict': '', 'response_status': 'unanswered'}, 'remark': None,
        'reason': answer, 'source': source}]}
    canonical = deepcopy(history)
    binding = view.decision_entries(history)[0]['bound_decision']
    shown, applied, missing = view.project_decision_notes(history, [
        {'bound_decision': binding, 'remark': 'The matrix is not countable.', 'reason': 'Recorded concerns remain diagnostics.'}])
    assert applied and not missing and history == canonical
    assert shown['rounds'][0]['verdict']['aggregate'] == 'UNANSWERED'
    assert shown['rounds'][0]['reviewers'][0]['answers']['coupling']['critical'] == 0
    if known_renderer:
        assert LONG not in str(shown)
        assert not view.preserved_review_fields(history)
    else:
        assert shown['rounds'][0]['verdict']['advisory_findings'] == [mirror]
        assert any(e['reason'] == 'unbound_verdict_mirror' for e in view.preserved_review_fields(history))


def test_the_organs_objections_take_an_actor_note_and_withdrawn_ones_group_in_the_actual_view(harness):
    """Through the real compaction tool, on a plan the organ recorded: an open objection takes
    the actor's note (the resident index shows the account), a withdrawn one folds into the
    actor's group, an open one is never grouped, and the plan record does not change."""
    from ouroboros import plan_record
    from tests.test_plan_task_tool import _call, _reply, _write

    ctx = harness.make_ctx()
    sub = harness.install({'s1': _reply('Two gaps.', objections=[('o1', 'no export path'), ('o2', 'no title slide')]),
                           's2': _reply('Fine.'), 's3': _reply('Fine.')})
    _write(ctx, ask='council')
    sub.answers['s1'] = _reply('The title is there now.', withdrawn=[('o2', 'revision 2 added a title slide')])
    _call(ctx, plan='Outline first, add a title slide, then draft each slide.', ask=['s1'], message='title added')
    record = deepcopy(plan_record.load_plan_record(harness.drive, ctx.task_id))
    observed = _capture(ctx)
    record_context_view(ctx, observed, [])
    entries = json.loads(_compact_context(ctx, inspect=True))['review_decisions']['entries']
    objections = {e['objection_id']: e for e in entries if e['decision_kind'] == 'plan_objection'}
    assert set(objections) == {'o1', 'o2'} and all(e['bound_decision'] for e in objections.values())
    assert objections['o1']['groupable'] is False and objections['o2']['groupable'] is True
    after, receipt, _ = _apply(ctx, observed, review_notes=[
        {'bound_decision': objections['o1']['bound_decision'], 'remark': 'Export is still open.',
         'reason': 's1 holds it; I answer it in the next revision.'},
        {'bound_decisions': [objections['o2']['bound_decision']], 'remark': 'The withdrawn title objection.',
         'reason': 'Revision 2 added the title slide and s1 withdrew it.'},
        {'bound_decisions': [objections['o1']['bound_decision']], 'remark': 'Do not hide this.',
         'reason': 'An open objection is never grouped.'}])
    assert receipt['status'] == 'applied', receipt
    rows = _index(after)[-1]['decision_rows']
    (shown,) = [r for r in rows if r.get('objection_id') == 'o1']
    assert shown['remark'] == 'Export is still open.' and shown['authored_view']['authorship'] == 'actor'
    assert any(r['decision_kind'] == 'actor_history_group' for r in rows)
    assert not any(r.get('objection_id') == 'o2' for r in rows)
    assert any(g['reason'] == 'current_or_unresolved_decision_stays_explicit' for g in receipt['review_notes']['unshortened'])
    assert plan_record.load_plan_record(harness.drive, ctx.task_id) == record
