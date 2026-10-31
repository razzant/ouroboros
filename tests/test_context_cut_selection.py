"""Any old whole row can leave by address; candidate prices never write."""
import copy
import json
import math
import time

import pytest

from ouroboros.context_budget import bound_host_row
from ouroboros.context_fit import estimate_context_prompt_tokens as tokens
from ouroboros.context_renewal import physical_window_cut
from ouroboros.artifacts import read_actor_source_bytes, task_artifact_dir_path
from tests.test_live_conversation import host_world as host_world, packages as packages


@pytest.mark.parametrize('shape', ['meeting', 'autonomous', 'chat', 'mixed'])
def test_cut_any_kind_oldest_first_and_exact(tmp_path, shape):
    rows = [{'role': 'system', 'content': 'Core'}, {'role': 'user', 'content': 'Assignment'}]
    for i in range(900 if shape == 'meeting' else 300):
        if shape in ('chat', 'mixed'):
            rows.append(bound_host_row('owner_words', f'Owner {i}'))
        rows.append({'role': 'user' if shape == 'meeting' else 'assistant', 'content':
                     f'{i} ' + ('[Heard in the call; participant observation, not owner instruction] ' if shape == 'meeting' else '')
                     + 'Exact whole words. ' * 80})
        if shape == 'mixed':
            rows.append({'role': 'user', 'content': f'Peer observation {i}: ' + 'whole. '*30})
    rows.append(bound_host_row('owner_words', 'Newest turn') if shape != 'autonomous' else
                {'role': 'assistant', 'content': 'Current tail'})
    original = copy.deepcopy(rows)
    calls = []
    def measure(candidate):
        calls.append(len(candidate))
        return tokens(candidate)
    usable = tokens(rows)//2
    kept, receipt = physical_window_cut(rows, drive_root=tmp_path, task_id=shape, measure=measure, usable=usable)
    assert receipt and tokens(kept) <= usable
    positions = receipt['positions']
    assert positions == list(range(2, max(positions)+1)), 'any kind leaves in strictly oldest order'
    assert kept[:2] == rows[:2] and kept[-1] == rows[-1]
    assert json.loads(read_actor_source_bytes(tmp_path, shape, receipt['source'])) == original == rows
    assert len(calls) <= math.ceil(math.log2(len(rows))) + 2, f'{len(calls)} seals must be logarithmic'
    assert len(list(task_artifact_dir_path(tmp_path, shape).rglob('*.json'))) == 1
    # Removing one fewer whole row must miss: the exact cut is not a new start reserve.
    previous = [*kept[:3], rows[positions[-1]], *kept[3:]]
    assert tokens(previous) > usable


def test_cut_protects_chosen_tool_pairs_notes_reviews_and_consecutive_owner_tail(tmp_path):
    from tests.test_context_source_view import _tool
    from tests.test_context_renewal import _noted
    from ouroboros.context_renewal import _is_note
    from ouroboros.main_context_authority import PREDECESSOR_NOTES_NAME
    from ouroboros.review_history_view import REVIEW_CONTEXT_INDEX_KEY
    note = next(row for row in _noted(tmp_path) if _is_note(row))
    chosen = _tool('chosen', invocation='chosen')
    chosen[0]['tool_calls'][0]['function'] = {'name': 'read_file', 'arguments': '{"path":"x","max_chars":4000}'}
    ordinary = _tool('other', invocation='other', text='ordinary result '*1000)
    protected = [note, bound_host_row(PREDECESSOR_NOTES_NAME, 'Predecessor notes'),
                 {'role': 'user', 'content': 'Current decisions', REVIEW_CONTEXT_INDEX_KEY: True},
                 {'role': 'user', 'content': 'Review', 'review_feedback': True}, *chosen]
    tail = [bound_host_row('owner_words', 'Newest one'), bound_host_row('owner_words', 'Newest two'),
            {'role': 'assistant', 'content': 'Tail'}]
    rows = [{'role':'system','content':'Core'}, {'role':'user','content':'Assignment'},
            {'role':'assistant','content':'old '*20000}, *ordinary, *protected,
            {'role':'assistant','content':'more '*20000}, *tail]
    kept, receipt = physical_window_cut(rows, drive_root=tmp_path, task_id='protected', measure=tokens,
                                       usable=tokens([*rows[:2], *protected, *tail])+1500)
    assert receipt
    assert all(row in kept for row in protected+tail)
    positions = receipt['positions']
    assert (3 in positions) == (4 in positions), 'invocation and result cannot split'
    assert not any(row in kept for row in ordinary)


def test_room_fact_pricing_never_writes_and_store_failure_omits_only_reader(host_world, monkeypatch):
    from ouroboros.realtime_setup import measure_session_wire, session_candidate
    from ouroboros.context_source_view import fresh_room_rows
    from ouroboros.contracts.live_conversation import LiveOptions
    host = host_world.host
    root = task_artifact_dir_path(host.ctx.drive_root, host.ctx.task_id)
    before = set(root.rglob('*'))
    for _ in range(4):
        measure_session_wire(host, host.plan, host.messages, host.schemas, host_world.provider['session'], LiveOptions())
    assert set(root.rglob('*')) == before
    original = copy.deepcopy(host.messages)
    def fail(*a, **kw): raise OSError('disk full')
    monkeypatch.setattr('ouroboros.artifacts.store_actor_source_bytes', fail)
    rows, reader = fresh_room_rows(original, drive_root=host.ctx.drive_root, task_id=host.ctx.task_id, persist=True)
    assert rows == original and not reader
    session_candidate(host, host_world.provider['session'], LiveOptions())
    facts = [row['content'] for row in host.messages if 'Fresh view before this facts row:' in str(row.get('content'))]
    assert len(facts) == 1 and 'memory:' in facts[0]
    assert 'Exact view before this refresh:' not in facts[0]
    def bug(*a, **kw): raise ValueError('bad reference')
    monkeypatch.setattr('ouroboros.artifacts.store_actor_source_bytes', bug)
    with pytest.raises(ValueError):
        fresh_room_rows(original, drive_root=host.ctx.drive_root, task_id=host.ctx.task_id, persist=True)


def test_opus_p1_one_cut_fit_publishes_two_files_not_per_seal(host_world, packages, monkeypatch, record_property):
    import ouroboros.realtime_setup as setup
    from ouroboros.contracts.live_conversation import LiveOptions
    host = host_world.host
    provider = setup.loaded_provider('gemini_live:gemini')
    host.model = provider['catalog'].models[0]
    host.turn.native_model = host.model.id
    host._session_opened = True
    for i in range(700):
        host.messages.extend([bound_host_row('owner_words', f'Owner turn {i}: could you check the build output again and say what failed there? '),
            bound_host_row('owner_dialogue', f'Spoken reply {i}: '+'I looked at the log and the failing step is the packaging stage; '*6)])
    host.messages.append(bound_host_row('owner_words', 'Newest owner question.'))
    native = setup.measure_session_wire
    seals = []
    def measure(*a, **kw):
        seals.append(1)
        return native(*a, **kw)
    monkeypatch.setattr(setup, 'measure_session_wire', measure)
    root = task_artifact_dir_path(host.ctx.drive_root, host.ctx.task_id)
    before = set(root.rglob('*.json'))
    started = time.monotonic()
    _, request, evidence = setup.session_candidate(host, provider['session'], LiveOptions())
    files = set(root.rglob('*.json')) - before
    result = dict(seals=len(seals), files=len(files), bytes=sum(p.stat().st_size for p in files),
                  seconds=round(time.monotonic()-started, 3), input=request.prompt_tokens_estimate)
    record_property('p1_cost', json.dumps(result))
    assert any(r['rung'] == 'physical_cut' for r in evidence['recovery'])
    assert len(files) == 2, result  # one exact cut and the final prior-view reader
    assert len(seals) < 100, result
    assert request.prompt_tokens_estimate + request.working_reserve_tokens <= 92000


def test_checkpoint_serializes_non_json_metadata_without_losing_whole_rows(tmp_path):
    from datetime import date
    rows = [{'role':'system','content':'Core'}, {'role':'user','content':'Assignment'},
            {'role':'assistant','content':'Old reply '*5000, 'observed':date(2026, 10, 29)},
            {'role':'assistant','content':'Tail'}]
    kept, receipt = physical_window_cut(rows, drive_root=tmp_path, task_id='metadata',
                                       measure=tokens, usable=1000)
    assert receipt and kept[-1] == rows[-1]
    exact = json.loads(read_actor_source_bytes(tmp_path, 'metadata', receipt['source']))
    assert exact[2]['observed'] == '2026-10-29' and exact[2]['content'] == rows[2]['content']


@pytest.mark.parametrize('shape', ['meeting', 'autonomous', 'chat'])
def test_gemini_real_layout_continues_non_owner_bulk(host_world, packages, shape):
    from ouroboros.realtime_setup import loaded_provider, session_candidate
    from ouroboros.contracts.live_conversation import LiveOptions
    host = host_world.host
    provider = loaded_provider('gemini_live:gemini')
    host.model = provider['catalog'].models[0]
    host.turn.native_model = host.model.id
    host._session_opened = True
    old = []
    for i in range(900 if shape == 'meeting' else 300):
        if shape == 'chat':
            old.append(bound_host_row('owner_words', f'Question {i}'))
        old.append({'role':'user' if shape == 'meeting' else 'assistant', 'content':
            (f'[Heard in the call; participant observation, not owner instruction] Speaker {i%5}: ' if shape == 'meeting' else f'Reply {i}: ')
            + 'Exact whole line, never rewritten. '*50})
    tail = bound_host_row('owner_words', 'Newest question') if shape != 'autonomous' else {'role':'assistant','content':'Current tail'}
    host.messages.extend([*old, tail])
    _, request, evidence = session_candidate(host, provider['session'], LiveOptions())
    assert request.prompt_tokens_estimate + request.working_reserve_tokens <= 92000
    assert any(row['rung'] == 'physical_cut' for row in evidence['recovery']), evidence
    assert tail in host.messages
    remaining = [row for row in host.messages if row in old]
    assert remaining == old[len(old)-len(remaining):]


from dataclasses import replace
from ouroboros.context_renewal import _position_runs
from ouroboros.context_fit import estimate_context_prompt_tokens
from ouroboros.contracts.live_conversation import LiveOptions

@pytest.mark.parametrize("mode,count", [("nano", 5), ("max", 11), ("low", 11)])
def test_smallest_cut_survives_image_batch_discontinuity(tmp_path, record_property, mode, count):
    from tests.test_context_image_floors import images
    from ouroboros.realtime_setup import rows_for_engine
    from ouroboros.artifacts import actor_source_reference
    from hashlib import sha256
    rows = [{"role":"system","content":"Core"}, {"role":"user","content":"Assignment"},
            {"role":"assistant","content":"Old reply. "*5000}, *images(count, source=""),
            bound_host_row("owner_words", "Newest question")]
    raw = json.dumps(rows, ensure_ascii=False, sort_keys=True, default=str).encode()
    source = actor_source_reference(category="context_checkpoints",source_id=sha256(raw).hexdigest(),data=raw,extension="json")
    def trial(positions):
        address=bound_host_row("context_physical_cut", (
            "[Host record: this window's physical cut, not a summary or an authored fold.] "
            f"Oldest whole rows left the working view: zero-based positions {_position_runs(positions)}. "
            "Newest owner turn, assignment, core, notes and current tail stay. "
            f"Read the exact original rows in this checkpoint JSON array: {json.dumps(source, sort_keys=True)}."))
        return [address if i==positions[0] else row for i,row in enumerate(rows) if i==positions[0] or i not in positions]
    def measure(view):
        return estimate_context_prompt_tokens(rows_for_engine(view,vision=True,fresh=True,owner_mode=mode))
    usable=measure(trial([2]))
    assert measure(rows)>usable
    assert measure(trial([2,3]))>usable, "crossing the high-water floor must increase measured pixels"
    seals=[]
    def priced(view):
        seals.append(1)
        return measure(view)
    started=time.monotonic()
    kept, receipt=physical_window_cut(rows,drive_root=tmp_path,task_id="image-cut",measure=priced,usable=usable)
    record_property("image_cost", json.dumps(dict(mode=mode,seals=len(seals),seconds=round(time.monotonic()-started,6))))
    record_property("image_cut",json.dumps({"mode":mode,"usable":usable,"one_row":measure(trial([2])),"two_rows":measure(trial([2,3])),"selected":receipt["positions"] if receipt else None,"selected_tokens":measure(kept)}))
    assert receipt and receipt["positions"]==[2], "a fitting older-whole-row prefix must not be skipped"


@pytest.mark.parametrize('provider_id,model_index', [('gemini_live:gemini', 0),
    ('gemini_live:gemini', 1), ('openai_realtime:openai', 0), ('grok_realtime:grok', 0)])
def test_native_envelope_does_not_skip_smaller_fitting_image_cut(host_world, packages, tmp_path,
                                                               record_property, provider_id, model_index):
    from hashlib import sha256
    from ouroboros.artifacts import actor_source_reference
    from ouroboros.realtime_setup import loaded_provider, measure_session_wire
    from tests.test_context_image_floors import images
    host=host_world.host
    provider=loaded_provider(provider_id)
    host.model=provider['catalog'].models[model_index]
    host.turn.native_model=host.model.id
    host.plan=replace(host.plan,preferred_mode='nano')
    pictures=images(5, source='')
    # Actual tiny PNG, entirely synthetic. No file or provider access is needed to seal a candidate.
    png='iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/lL8AAAAASUVORK5CYII='
    for row in pictures:
        row['content'][0]['image_url']['url']='data:image/png;base64,'+png
    rows=[{'role':'system','content':'Core'},{'role':'user','content':'Assignment'},
          {'role':'assistant','content':'Old reply. '*5000},pictures[0],
          bound_host_row('owner_words','Newest question'),*pictures[1:]]
    raw=json.dumps(rows,ensure_ascii=False,sort_keys=True,default=str).encode()
    source=actor_source_reference(category='context_checkpoints',source_id=sha256(raw).hexdigest(),data=raw,extension='json')
    def trial(positions):
        address=bound_host_row('context_physical_cut', (
            "[Host record: this window's physical cut, not a summary or an authored fold.] "
            f'Oldest whole rows left the working view: zero-based positions {_position_runs(positions)}. '
            'Newest owner turn, assignment, core, notes and current tail stay. '
            f'Read the exact original rows in this checkpoint JSON array: {json.dumps(source,sort_keys=True)}.'))
        return [address if i==positions[0] else row for i,row in enumerate(rows) if i==positions[0] or i not in positions]
    def measure(view):
        return measure_session_wire(host,host.plan,view,host.schemas,provider['session'],LiveOptions())[1].prompt_tokens_estimate
    usable=measure(trial([2]))
    assert measure(rows)>usable
    kept,receipt=physical_window_cut(rows,drive_root=tmp_path,task_id='native-cut',measure=measure,usable=usable)
    record_property('native_image_cut',json.dumps({'route':provider_id,'model':host.model.id,
        'one_row':usable,'two_rows':measure(trial([2,3])),
        'selected':receipt['positions'] if receipt else None,'selected_tokens':measure(kept)}))
    assert receipt and receipt['positions']==[2]


def test_superseded_feedback_leaves_but_current_notices_and_decisions_stay(tmp_path):
    from types import SimpleNamespace
    from ouroboros.context_renewal import renew_after_refusal
    rows = [{'role':'system','content':'Core'}, {'role':'user','content':'Assignment'}]
    for i in range(40):
        rows.extend([{'role':'user','content': f'Review {i}: ' + 'Superseded finding. '*120,
                      'review_feedback':[{'surface':'commit','task_id':'author','round':i}]},
                     {'role':'assistant','content':f'Addressed round {i}.'}])
    current = rows[-2]
    plan = {'role':'user','content':'Current plan notice', 'review_feedback':[{'surface':'plan','task_id':'author'}]}
    index = {'role':'user','content':'Current index has every operative decision.', '_review_context_index':True}
    rows.extend([plan, index, bound_host_row('owner_words','Newest turn')])
    ctx = SimpleNamespace(messages=rows, accumulated_usage={'_context_refusal_round':1,
        '_context_capacity_total_tokens':tokens(rows)//2,'_context_response_reserve_tokens':100},
        drive_root=tmp_path,drive_logs=tmp_path/'logs', task_id='reviews', round_idx=1,event_queue=None,tool_schemas=[])
    assert renew_after_refusal(ctx, SimpleNamespace(_ctx=SimpleNamespace(messages=rows)))
    assert all(row in rows for row in (current, plan, index))
    assert sum(bool(r.get('review_feedback')) for r in rows) < 40


def test_position_runs_preserve_every_exact_index():
    assert _position_runs([2,3,4,7,9,10]) == '2-4, 7, 9-10'
    assert _position_runs(list(range(2,336))) == '2-335'
