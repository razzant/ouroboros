"""Source-bound actor notes shorten complete review representations, not authority."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from ouroboros import review_history_view as view
from ouroboros.artifacts import read_actor_source_bytes
from ouroboros.tools.compact_context import _compact_context, record_context_view
from tests.test_review_view_integration import _apply, _capture, _index, harness as _harness, staged_body as _staged_body
from tests.test_plan_review_engine import CLEAN, _call, _control, _finding, _state

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
    sub = harness.install({'s1': json.dumps([_finding('same', 'note', summary='Consider a duplicate cache')]),
                           's2': CLEAN, 's3': CLEAN})
    ctx = harness.make_ctx()
    assert _control(_call(ctx))['closed']
    first = _state(harness)['waves'][-1]
    from ouroboros.tools.plan_review import _apply_disposition
    _apply_disposition(ctx, {'review_fingerprint': first['request_fingerprint'], 'items': [
        {'finding_id': first['findings'][0]['finding_id'], 'decision': 'reject', 'rationale': LONG}]})
    return ctx, sub


def test_long_plan_reason_shortens_in_actual_messages_cold_view_and_new_packet(harness):
    ctx, sub = first_plan(harness)
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
    assert _state(harness) == before_state and len(sub.calls) == 1
    assert _index(after)[-1]['operative_subject']['spec'] == operative['spec']
    old = json.loads(read_actor_source_bytes(ctx.drive_root, ctx.task_id, receipt['checkpoint_ref']))
    assert LONG in str(old['messages'])
    assert LONG not in str(_capture(harness.make_ctx()))
    assert _control(_call(harness.make_ctx(), plan='Read the final caption against the same contract.'))['closed']
    packet = str(sub.calls[-1]['request'].messages)
    assert LONG not in packet and notes[-1]['reason'] in packet
    assert len(sub.calls) == 2
    assert view.current_plan_history(ctx)[0]['rounds'][0]['source'] == history['rounds'][0]['source']


def test_absent_and_stale_notes_keep_only_the_unselected_entries_full(harness):
    ctx, _ = first_plan(harness)
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


def test_selected_author_reason_and_runtime_authority_mirrors_are_views_only(harness, monkeypatch):
    from ouroboros.task_results import plan_review_authority_core
    from tests.test_plan_dispute_history import DECK_SPEC
    ctx, sub = first_plan(harness)
    harness.state['enforcement'] = 'advisory'
    monkeypatch.setenv('OUROBOROS_REVIEW_ENFORCEMENT', 'advisory')
    critic = _state(harness)['waves'][-1]
    reason = ('This selected author reason covers the exact prior concern. ' * 100).strip()
    selected_spec = {**DECK_SPEC, 'invariants': ['A selected new owner requirement']}
    assert 'Current author plan saved' in _call(ctx, selected_spec, plan='Selected plan complete.',
        review_disposition={'review_fingerprint': critic['request_fingerprint'], 'author_action': 'finish',
                            'author_disposition': {'disposition': 'partial', 'rationale': reason}})
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
    assert _state(harness) == canonical and len(sub.calls) == 1
    assert _index(after)[-1]['operative_subject']['spec'] == view.current_plan_history(ctx)[1]['spec']
    assert reason not in str(_capture(harness.make_ctx()))
    selected = json.loads(read_actor_source_bytes(ctx.drive_root, ctx.task_id, canonical['author_history_head']))
    assert selected['author_disposition']['rationale'] == reason
    _call(harness.make_ctx(), {**DECK_SPEC, 'invariants': ['Next critic view']}, plan='Send the next full plan.')
    assert reason not in str(sub.calls[-1]['request'].messages)
    assert LONG not in str(sub.calls[-1]['request'].messages)


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
    for key in ('aggregate', 'per_row', 'per_question', 'quorum', 'reason'):
        if key in history['rounds'][0]['verdict']:
            assert shown['rounds'][0]['verdict'][key] == history['rounds'][0]['verdict'][key]
    for old, new in zip(history['rounds'][0]['reviewers'], shown['rounds'][0]['reviewers']):
        assert old['seat_id'] == new['seat_id'] and old['status'] == new['status']
        for part, answer in old['answers'].items():
            for key in ('status', 'verdict', 'critical', 'coverage'):
                assert new['answers'][part][key] == answer[key]
            for field in ('items', 'findings', 'discarded'):
                for item, projected in zip(answer.get(field, []), new['answers'][part].get(field, [])):
                    for key in ('item', 'verdict', 'severity', 'obligation_id', 'slot_id'):
                        if key in item:
                            assert projected[key] == item[key]
    record_context_view(ctx, after, [])
    inspected = json.loads(_compact_context(ctx, inspect=True))
    assert all(e['representation'] == 'actor_review_note' for e in inspected['review_decisions']['entries'])


def test_two_decisions_partial_selection_stale_entry_and_new_finding_stay_local(harness):
    sub = harness.install({'s1': json.dumps([_finding('one', 'note'), _finding('two', 'note')]),
                           's2': CLEAN, 's3': CLEAN})
    ctx = harness.make_ctx()
    assert _control(_call(ctx))['closed']
    wave = _state(harness)['waves'][-1]
    reasons = [LONG, 'An independent reason for the second concern. ' * 700]
    from ouroboros.tools.plan_review import _apply_disposition
    _apply_disposition(ctx, {'review_fingerprint': wave['request_fingerprint'], 'items': [
        {'finding_id': f['finding_id'], 'decision': 'reject', 'rationale': r}
        for f, r in zip(wave['findings'], reasons)]})
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
    sub.answers = {'s1': json.dumps([_finding('new', 'note', summary='NEW_UNCOVERED_POINT')])}
    _call(harness.make_ctx(), plan='A new independent version for review.')
    cold = _capture(harness.make_ctx())
    assert reasons[0] not in str(cold) and reasons[1] in str(cold) and 'NEW_UNCOVERED_POINT' in str(cold)
    assert any(e['representation'] == 'full' for e in view.review_note_options(ctx)['entries'])


def test_closure_carried_findings_and_unknown_extension_are_accounted_for(harness):
    from ouroboros.task_results import record_plan_review_wave
    from ouroboros.tools import plan_review_artifacts as artifacts
    ctx, _ = first_plan(harness)
    wave = artifacts.authority_wave(ctx.drive_root, ctx.task_id, _state(harness)['waves'][-1])
    wave['closure_notes'] = ['The precise recorded closure rationale. ' * 700]
    wave['actors'][0]['carried_findings'] = deepcopy(wave['findings'])
    wave['reviewer_outputs'][0]['future_review_extension'] = {'reason': 'KEEP_UNKNOWN_EXTENSION_EXACT'}
    wave['wave_artifact'] = artifacts.persist_wave(ctx.drive_root, ctx.task_id, wave)
    record_plan_review_wave(ctx.drive_root, ctx.task_id, wave)
    history, _ = view.current_plan_history(ctx)
    canonical = deepcopy(_state(harness))
    notes = notes_for(ctx)
    assert any(e['decision_kind'] == 'plan_closure' for e in view.review_note_options(ctx)['entries'])
    after, receipt, _ = _apply(ctx, _capture(ctx), review_notes=notes)
    assert receipt['status'] == 'applied' and LONG not in str(after)
    assert wave['closure_notes'][0] not in str(after) and 'KEEP_UNKNOWN_EXTENSION_EXACT' in str(after)
    assert any(p['path'][-1] == 'future_review_extension' for p in receipt['review_notes']['preserved_fields'])
    assert _state(harness) == canonical
    old = history['rounds'][-1]['reviewers'][0]['carried_findings'][0]
    new = _index(after)[-1]['rounds'][-1]['reviewers'][0]['carried_findings'][0]
    assert all(new[k] == old[k] for k in ('finding_id', 'class', 'breaks'))
    source = json.loads(read_actor_source_bytes(ctx.drive_root, ctx.task_id, wave['wave_artifact']))
    assert source['closure_notes'] == wave['closure_notes']


def test_ordinary_note_discloses_full_decisions_and_lost_selection_restores_them(harness):
    from ouroboros.artifacts import task_artifact_dir_path
    ctx, sub = first_plan(harness)
    after, receipt, _ = _apply(ctx, _capture(ctx))
    assert receipt['status'] == 'applied' and LONG in str(after)
    assert not receipt['review_notes']['applied'] and receipt['review_notes']['unshortened']
    short, receipt, _ = _apply(ctx, after, review_notes=notes_for(ctx))
    assert receipt['status'] == 'applied' and LONG not in str(short)
    ref = receipt['selected_review_history_view']['source_ref']
    (task_artifact_dir_path(ctx.drive_root, ctx.task_id, create=False) / ref['path']).unlink()
    cold = _capture(harness.make_ctx())
    assert LONG in str(cold) and 'REVIEW_HISTORY_VIEW_SOURCE_UNAVAILABLE' in str(cold)
    assert len(sub.calls) == 1


def test_actual_inspect_pair_remains_in_newer_tail_without_echoing_original_reason(harness):
    from types import SimpleNamespace
    from ouroboros.loop_tool_execution import process_tool_results
    from tests.test_main_authored_context import call
    ctx, _ = first_plan(harness)
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
