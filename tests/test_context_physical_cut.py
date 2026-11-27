"""Only physical pressure can move old delivered whole rows to their exact reader."""
import copy
import hashlib
import json
from types import SimpleNamespace

import pytest

from ouroboros.artifacts import read_actor_source_bytes
from ouroboros.context_budget import bound_host_row
from ouroboros.context_fit import estimate_context_prompt_tokens
from ouroboros.context_renewal import physical_window_cut, renew_after_refusal


def conversation():
    return [{'role': 'system', 'content': 'Required core.'}, {'role': 'user', 'content': 'Assignment.'},
            *[bound_host_row(kind, f'{i} Ж e\u0301\r\n' * 2000) for i in range(8)
              for kind in ('owner_words', 'owner_dialogue')],
            bound_host_row('owner_words', 'Newest owner turn.'),
            {'role': 'assistant', 'content': 'Current tail.'}]


def test_physical_cut_keeps_whole_rows_oldest_first_and_exact_checkpoint(tmp_path):
    rows = conversation(); original = copy.deepcopy(rows)
    budget = estimate_context_prompt_tokens(rows) // 2
    kept, receipt = physical_window_cut(rows, drive_root=tmp_path, task_id='cut',
                                       measure=estimate_context_prompt_tokens, usable=budget)
    assert receipt and estimate_context_prompt_tokens(kept) <= budget
    assert receipt['positions'] == list(range(2, receipt['positions'][-1] + 1))
    exact = json.loads(read_actor_source_bytes(tmp_path, 'cut', receipt['source']))
    assert rows == original == exact
    assert kept[:2] == rows[:2] and kept[-2:] == rows[-2:]
    for i, row in enumerate(rows):
        assert (row not in kept) == (i in receipt['positions'])
    assert "window's physical cut" in kept[2]['content']


def test_fitting_view_and_irreducible_tail_never_cut(tmp_path):
    for usable in (estimate_context_prompt_tokens(conversation()) + 1, 1):
        rows = conversation(); before = copy.deepcopy(rows)
        kept, receipt = physical_window_cut(rows, drive_root=tmp_path, task_id='cut',
                                           measure=estimate_context_prompt_tokens, usable=usable)
        assert receipt is None and kept == rows == before


def test_a_fresh_selection_keeps_its_earlier_shared_full_source_during_a_physical_cut(tmp_path):
    from ouroboros.context_budget import HOST_TEXT_SHA_KEY, REPLY_SOURCE_SHA_KEY
    from ouroboros.loop_round_limits import _append_reply_context

    rows = conversation()[:2]
    source = {"address": "row:1@source#0123456789ab", "author": "Ouroboros", "text": "North trail. South gate.",
              "attachments": [{"name": "map.pdf", "mime": "application/pdf", "size": 55}]}
    _append_reply_context(rows, {**source, "excerpt": {"text": "North trail", "start_utf16": 0}}, None)
    full = rows[2]
    rows.append(bound_host_row("owner_words", "First question."))
    rows.extend(conversation()[2:-2])
    _append_reply_context(rows, {**source, "excerpt": {"text": "South gate", "start_utf16": 13}}, None)
    rows.append(bound_host_row("owner_words", "Second question."))
    rows.append({"role": "assistant", "content": "Current tail."})
    assert rows[-3][REPLY_SOURCE_SHA_KEY] == full[HOST_TEXT_SHA_KEY]
    kept, receipt = physical_window_cut(rows, drive_root=tmp_path, task_id="shared-source",
                                       measure=estimate_context_prompt_tokens,
                                       usable=estimate_context_prompt_tokens(rows) // 2)
    assert receipt and full in kept and "map.pdf" in full["content"]
    assert kept[-3:] == rows[-3:] and 2 not in receipt["positions"]
    assert json.loads(read_actor_source_bytes(tmp_path, "shared-source", receipt["source"])) == rows


def test_physical_cut_keeps_fresh_reply_citation_and_never_bisects_an_older_reply(tmp_path):
    old_quote = bound_host_row('reply_citation', 'An old citation.' * 800)
    old_words = bound_host_row('owner_words', 'My old reply.' * 100)
    fresh_quote = bound_host_row('reply_citation', 'A fresh citation.' * 800)
    fresh_words = bound_host_row('owner_words', 'My newest reply.')
    facts = bound_host_row('return_facts', 'Facts before my words.')
    rows = [*conversation()[:2], old_quote, facts, old_words, {'role': 'assistant', 'content': 'Earlier answer.'},
            fresh_quote, facts, fresh_words, {'role': 'assistant', 'content': 'Current tail.'}]
    budget = estimate_context_prompt_tokens(rows) - 500
    kept, receipt = physical_window_cut(rows, drive_root=tmp_path, task_id='cited-cut',
                                       measure=estimate_context_prompt_tokens, usable=budget)
    assert receipt['positions'] == [2, 3, 4, 5]  # the earlier answer first, then the cited pair, never bisected
    assert kept[-4:] == rows[-4:]
    assert json.loads(read_actor_source_bytes(tmp_path, 'cited-cut', receipt['source'])) == rows
    cut, receipt = physical_window_cut(rows, drive_root=tmp_path, task_id='cited-cut',
                                      measure=estimate_context_prompt_tokens, usable=1)
    assert cut == rows and receipt is None


def test_physical_cut_takes_an_older_full_source_alone_and_keeps_the_fresh_source_beside_its_selection(tmp_path):
    """A source is its own host row before its selection: an older one leaves alone (its selection and the words stay a
    pair); the newest turn's source, selection and words are one protected run."""
    old_source = bound_host_row('reply_citation', '[Full source from Ouroboros at row:old]\n' + 'An old source.' * 800)
    old_quote = bound_host_row('reply_citation', 'An old selection.')
    old_words = bound_host_row('owner_words', 'My old reply.' * 100)
    fresh_source = bound_host_row('reply_citation', '[Full source from Ouroboros at row:new]\n' + 'A fresh source.' * 800)
    fresh_quote = bound_host_row('reply_citation', 'A fresh selection.')
    fresh_words = bound_host_row('owner_words', 'My newest reply.')
    rows = [*conversation()[:2], old_source, old_quote, old_words, {'role': 'assistant', 'content': 'Earlier answer.'},
            fresh_source, fresh_quote, fresh_words, {'role': 'assistant', 'content': 'Current tail.'}]
    total, source = estimate_context_prompt_tokens(rows), estimate_context_prompt_tokens([old_source])
    kept, receipt = physical_window_cut(rows, drive_root=tmp_path, task_id='source-cut',
                                       measure=estimate_context_prompt_tokens, usable=total - (source - 400))
    assert receipt['positions'] == [2, 5], 'the earlier answer, then the older source alone'
    assert old_quote in kept and old_words in kept and kept[-4:] == rows[-4:]
    kept, receipt = physical_window_cut(rows, drive_root=tmp_path, task_id='source-cut-more',
                                       measure=estimate_context_prompt_tokens, usable=total - (source + 20))
    assert receipt['positions'] == [2, 3, 4, 5], 'the older selection leaves only with its words'
    assert kept[-4:] == rows[-4:], 'the newest source, selection and words never leave'


def test_text_refusal_uses_same_physical_cut_without_changing_round_or_dispatch(tmp_path):
    rows = conversation()
    ctx = SimpleNamespace(messages=rows, accumulated_usage={'_context_refusal_round': 3,
        '_context_capacity_total_tokens': estimate_context_prompt_tokens(rows) // 2 + 1000,
        '_context_response_reserve_tokens': 1000}, drive_root=tmp_path, drive_logs=tmp_path/'logs',
        task_id='text-cut', round_idx=3, event_queue=None, tool_schemas=[])
    tools = SimpleNamespace(_ctx=SimpleNamespace(messages=rows))
    assert renew_after_refusal(ctx, tools)
    assert ctx.round_idx == 3 and ctx.accumulated_usage['_context_refusal_round'] == 3
    assert "window's physical cut" in rows[2]['content']
    assert rows[-1]['content'] == 'Current tail.'


def test_repeated_growth_keeps_one_cut_record_and_its_exact_predecessor_readable(tmp_path):
    original = conversation()
    budget = estimate_context_prompt_tokens(original) // 2
    first, one = physical_window_cut(original, drive_root=tmp_path, task_id='cut',
                                     measure=estimate_context_prompt_tokens, usable=budget)
    old_address = first[2]
    grown = [*first, *conversation()[2:]]
    second, two = physical_window_cut(grown, drive_root=tmp_path, task_id='cut',
                                      measure=estimate_context_prompt_tokens, usable=budget)
    assert two and estimate_context_prompt_tokens(second) <= budget
    assert sum(r.get('_host_context_kind') == 'context_physical_cut' for r in second) == 1
    exact = json.loads(read_actor_source_bytes(tmp_path, 'cut', two['source']))
    assert exact[two['previous_cuts'][0]] == old_address
    assert json.loads(read_actor_source_bytes(tmp_path, 'cut', one['source'])) == original


def _quiet_owner_conversation():
    """C11's shape: early owner words and outward replies, the newest owner turn, then hours of the mind's tool work,
    prose and roster snapshots (each superseding the one before) before the current exchange."""
    rows = [{'role': 'system', 'content': 'Required core.'}, {'role': 'user', 'content': 'Assignment.'}]
    rows += [bound_host_row(kind, f'early {kind} {i} ' * 400) for i in range(3) for kind in ('owner_words', 'owner_dialogue')]
    rows.append(bound_host_row('owner_words', 'Newest owner turn: go on.'))
    for n in range(6):
        rows.append({'role': 'assistant', 'content': '', 'tool_calls': [{'id': f'c{n}', 'type': 'function',
                     'function': {'name': 'read_file', 'arguments': '{}'}}]})
        rows.append({'role': 'tool', 'tool_call_id': f'c{n}', 'content': f'tool result {n} ' * 600})
        rows.append({'role': 'assistant', 'content': f'My reasoning after result {n}. ' * 120})
        rows.append(bound_host_row('ouroboros_roster_snapshot', f'roster change {n} ' * 200))
    rows.append({'role': 'assistant', 'content': '', 'tool_calls': [{'id': 'now', 'type': 'function',
                 'function': {'name': 'read_file', 'arguments': '{}'}}]})
    rows.append({'role': 'tool', 'tool_call_id': 'now', 'content': 'The result the work is acting on.'})
    return rows


def test_a_long_quiet_owner_keeps_his_words_while_later_host_rows_tool_units_and_prose_leave(tmp_path):
    """Owner 29.10 «1. A» lets the oldest rows leave to an exact copy by address and the work go on; CHECKLISTS item 18
    puts the owner's words last: hours of work after the newest owner turn are not protected for their position.
    Superseded host copies go first (the current snapshot stays: its producer would send it again), then tool units,
    then the mind's prose; the owner's earlier words and outward replies stay while anything else can pay, the newest
    owner turn and the newest exchange always stay, and the address record sits at the first removed position."""
    from ouroboros.context_budget import HOST_CONTEXT_KIND_KEY
    rows = _quiet_owner_conversation(); original = copy.deepcopy(rows)
    owner = {i for i, row in enumerate(rows) if row.get(HOST_CONTEXT_KIND_KEY) in ('owner_words', 'owner_dialogue')}
    snapshots = [i for i, row in enumerate(rows) if row.get(HOST_CONTEXT_KIND_KEY) == 'ouroboros_roster_snapshot']
    budget = estimate_context_prompt_tokens(rows) - estimate_context_prompt_tokens([rows[i] for i in snapshots]) // 2
    kept, receipt = physical_window_cut(rows, drive_root=tmp_path, task_id='quiet',
                                       measure=estimate_context_prompt_tokens, usable=budget)
    assert receipt and estimate_context_prompt_tokens(kept) <= budget
    gone = set(receipt['positions'])
    assert gone and gone <= set(snapshots[:-1]) and rows[snapshots[-1]] in kept
    assert not gone & owner and rows == original
    assert kept[min(gone)][HOST_CONTEXT_KIND_KEY] == 'context_physical_cut' and kept[-2:] == rows[-2:]
    work = estimate_context_prompt_tokens([r for i, r in enumerate(rows) if i > 9 and i < len(rows) - 2
                                           and r.get(HOST_CONTEXT_KIND_KEY) is None])
    kept, receipt = physical_window_cut(rows, drive_root=tmp_path, task_id='quiet',
                                       measure=estimate_context_prompt_tokens,
                                       usable=estimate_context_prompt_tokens(rows) - work)
    gone = set(receipt['positions'])
    assert not gone & owner, 'tool units and prose leave before any of the owner words'
    assert any(rows[i].get('role') == 'tool' for i in gone) and kept[-2:] == rows[-2:]
    assert any(r.get(HOST_CONTEXT_KIND_KEY) == 'owner_words' and 'Newest owner turn' in r['content'] for r in kept)


def test_when_only_the_owners_dialogue_can_pay_its_oldest_rows_leave_and_the_newest_turn_stays(tmp_path):
    """The other direction: with nothing but dialogue to spend, the cut still fits by the oldest owner rows by exact
    address (the 7026c7119 capability), never the newest owner turn or the current exchange."""
    from ouroboros.context_budget import HOST_CONTEXT_KIND_KEY
    rows = conversation()
    kept, receipt = physical_window_cut(rows, drive_root=tmp_path, task_id='dialogue',
                                       measure=estimate_context_prompt_tokens,
                                       usable=estimate_context_prompt_tokens(rows) // 2)
    assert receipt and receipt['positions'][0] == 2
    assert all(rows[i].get(HOST_CONTEXT_KIND_KEY) in ('owner_words', 'owner_dialogue') for i in receipt['positions'])
    assert kept[-2:] == rows[-2:]


_ANTHROPIC = {'provider': 'anthropic', 'base_url': 'https://anthropic.example/v1', 'resolved_model': 'test-model'}
_RESPONSES = {'provider': 'openai', 'api_surface': 'responses', 'base_url': 'https://api.openai.com/v1',
              'resolved_model': 'gpt-test'}


def _native_call(transport, ids, thought):
    """The mind's call of ``ids`` with the native continuation its route's own custody retains for replay."""
    from ouroboros.anthropic_native_custody import retain_native_assistant_content
    from ouroboros.openai_responses_custody import retain_items
    from tests.test_context_view_tool import _own_call

    arguments = {'working_note': 'Measured.'}
    row = _own_call(ids[0], arguments)
    row['tool_calls'] += [{**row['tool_calls'][0], 'id': call_id} for call_id in ids[1:]]
    if transport == 'anthropic':
        return retain_native_assistant_content(row, [
            {'type': 'thinking', 'thinking': thought, 'signature': 'synthetic'},
            *({'type': 'tool_use', 'id': call_id, 'name': 'compact_context', 'input': arguments} for call_id in ids)],
            _ANTHROPIC)
    return retain_items(row, [{'type': 'reasoning', 'id': 'rs_1', 'encrypted_content': thought},
        *({'type': 'function_call', 'call_id': call_id, 'name': 'compact_context', 'arguments': json.dumps(arguments)}
          for call_id in ids)], _RESPONSES, 'completed')


def _answered_after(transport, call, text):
    """The mind's next answered reply, which consumed ``call``'s replay (as the routes' own markers record it)."""
    from ouroboros.anthropic_native_custody import ANTHROPIC_CONSUMED_RECEIPTS_KEY, ANTHROPIC_NATIVE_RECEIPT_KEY
    from ouroboros.openai_responses_custody import RESPONSES_CONSUMED_KEY, RESPONSES_RECEIPT_KEY

    if transport == 'anthropic':
        return {'role': 'assistant', 'content': text,
                ANTHROPIC_CONSUMED_RECEIPTS_KEY: [call[ANTHROPIC_NATIVE_RECEIPT_KEY]['content_sha256']]}
    return {'role': 'assistant', 'content': text, RESPONSES_CONSUMED_KEY: [call[RESPONSES_RECEIPT_KEY]['items_sha256']]}


@pytest.mark.parametrize('transport', ['anthropic', 'responses'])
def test_a_call_leaves_only_with_all_its_results_and_never_while_its_native_replay_is_unconsumed(tmp_path, transport):
    """A call and every result after it are one unit, complete or not, with what arrived between them (an image a
    viewing call attached while its batch ran), whether or not the mind may fold them; an exchange whose native
    reasoning the provider must still replay beside its results stays whole wherever it sits (here my human's words
    arrived while its results waited): a cut fitting by dropping such a call would leave its results alone in the next
    request. A consumed older exchange still leaves, call and results together."""
    from ouroboros.anthropic_native_custody import anthropic_tool_unit_active
    from ouroboros.openai_responses_custody import responses_tool_unit_active
    active = anthropic_tool_unit_active if transport == 'anthropic' else responses_tool_unit_active
    price = estimate_context_prompt_tokens

    def call(ids, path):
        return {'role': 'assistant', 'content': '', 'tool_calls': [{'id': i, 'type': 'function', 'function': {
            'name': 'read_file', 'arguments': json.dumps({'path': path})}} for i in ids]}
    caption = '[image: shot.png; original: /original/shot.png]'
    old = _native_call(transport, ['o1', 'o2'], 'Old reasoning.')
    new = _native_call(transport, ['n1', 'n2'], 'Measured privately. ' * 600)
    rows = [{'role': 'system', 'content': 'Required core.'}, {'role': 'user', 'content': 'Assignment.'},
            call(['p1', 'p2'], 'p' * 16000), {'role': 'tool', 'tool_call_id': 'p1', 'content': 'Only one result came.'},
            call(['v1'], 'v' * 16000), {'role': 'user', 'content': [{'type': 'text', 'text': caption}, {
                'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,iVBORw0KGgo='}, '_caption': caption,
                '_source_path': '/original/shot.png'}]}, {'role': 'tool', 'tool_call_id': 'v1', 'content': 'Attached.'},
            old, *({'role': 'tool', 'tool_call_id': i, 'content': f'old result {i} ' * 300} for i in ('o1', 'o2')),
            _answered_after(transport, old, 'Both old results read.'),
            new, *({'role': 'tool', 'tool_call_id': i, 'content': 'accepted'} for i in ('n1', 'n2')),
            bound_host_row('owner_words', 'Newest owner turn.')]
    assert not active(rows, 7, 9) and active(rows, 11, 13)
    before = copy.deepcopy(rows)
    kept, receipt = physical_window_cut(rows, drive_root=tmp_path, task_id='native', measure=price,
                                       usable=price([*rows[:2], *rows[11:]]))
    assert receipt is None and kept == rows == before, 'only the unconsumed call could pay: no cut'
    kept, receipt = physical_window_cut(rows, drive_root=tmp_path, task_id='native', measure=price,
                                       usable=price(rows) - price([rows[2]]) // 2)
    assert receipt['positions'] == [2, 3], 'the older incomplete call leaves only with the result it has'
    kept, receipt = physical_window_cut(rows, drive_root=tmp_path, task_id='native', measure=price,
                                       usable=price(rows) - price(rows[2:4]) - price([rows[4]]) // 2)
    assert receipt['positions'] == [2, 3, 4, 5, 6], 'the viewing call leaves with its image and its result'
    kept, receipt = physical_window_cut(rows, drive_root=tmp_path, task_id='native', measure=price,
                                       usable=price(rows) - price(rows[2:7]) - price(rows[7:10]) // 2)
    assert receipt['positions'] == list(range(2, 10)) and kept[-4:] == rows[-4:]


@pytest.mark.parametrize('transport', ['anthropic', 'responses'])
def test_the_text_refusal_never_publishes_a_result_without_its_native_call(tmp_path, transport):
    """The ordinary refusal path, where small finished batches make the indexed seed save nothing: the current native
    exchange after my human's newest turn keeps its call beside its result; older batches pay instead."""
    from ouroboros.context_renewal import renewal_view
    price = estimate_context_prompt_tokens
    current = _native_call(transport, ['current'], 'Measured privately. ' * 240)
    result = {'role': 'tool', 'tool_call_id': 'current', 'content': 'OK'}
    for count in (4, 12, 30, 70, 160):
        rows = [{'role': 'system', 'content': 'Core.'}, {'role': 'user', 'content': 'Assignment.'}]
        for i in range(count):
            rows += [{'role': 'assistant', 'content': '', 'tool_calls': [{'id': str(i), 'type': 'function',
                      'function': {'name': 'read_file', 'arguments': '{}'}}]},
                     {'role': 'tool', 'tool_call_id': str(i), 'content': 'OK'}]
        rows += [bound_host_row('owner_words', 'Please verify.'), current, result]
        if renewal_view(rows, observation=None, drive_root=tmp_path, task_id='seed', measure=price)[1] is None:
            break
    else:
        pytest.fail('every indexed seed saved room; the fixture no longer reaches the physical cut')

    def refused(capacity):
        ctx = SimpleNamespace(messages=rows, accumulated_usage={'_context_refusal_round': 1,
            '_context_capacity_total_tokens': capacity, '_context_response_reserve_tokens': 0}, drive_root=tmp_path,
            drive_logs=tmp_path / 'logs', task_id='native-text', round_idx=1, event_queue=None, tool_schemas=[])
        return renew_after_refusal(ctx, SimpleNamespace(_ctx=SimpleNamespace(messages=rows)))

    before = copy.deepcopy(rows)
    assert not refused(600) and rows == before, 'only the current call could pay: no view is published'
    assert refused(price(rows) - price(rows[2:8]))
    assert rows[-2:] == [current, result] and before[2] not in rows


def test_an_arrival_nobody_has_acted_on_stays_until_the_first_act(tmp_path):
    """Before the mind's first act nothing after my human's newest turn has been seen: a child's report delivered
    there stays (the mailbox already gave it up, so no later drain brings it back). Once the mind has acted after it,
    the report is older work like any other and may leave, after the clock that request consumed."""
    from ouroboros.send_clock import CLOCK_ROW_NAME
    price = estimate_context_prompt_tokens
    rows = [{'role': 'system', 'content': 'Core.'}, {'role': 'user', 'content': 'Assignment.'},
            bound_host_row('owner_words', 'Review the measurements the child is sending.'),
            {'role': 'user', 'content': '[Task message from child]: ' + 'Measurement from the child. ' * 1200},
            bound_host_row(CLOCK_ROW_NAME, 'Current time.')]
    before = copy.deepcopy(rows)
    kept, receipt = physical_window_cut(rows, drive_root=tmp_path, task_id='unseen', measure=price,
                                       usable=price(rows) // 2)
    assert receipt is None and kept == rows == before
    ctx = SimpleNamespace(messages=rows, accumulated_usage={'_context_refusal_round': 1,
        '_context_capacity_total_tokens': price(rows) // 2, '_context_response_reserve_tokens': 0}, drive_root=tmp_path,
        drive_logs=tmp_path / 'logs', task_id='unseen', round_idx=1, event_queue=None, tool_schemas=[])
    assert not renew_after_refusal(ctx, SimpleNamespace(_ctx=SimpleNamespace(messages=rows))) and rows == before
    acted = [*rows, {'role': 'assistant', 'content': 'The measurements in the report agree.'}]
    kept, receipt = physical_window_cut(acted, drive_root=tmp_path, task_id='seen', measure=price,
                                       usable=price(acted) // 2)
    assert receipt['positions'] == [3, 4] and kept[-1] == acted[-1]



def test_a_batch_an_outward_message_closed_stays_until_the_mind_answers_after_it(tmp_path):
    """After a batch's results the host appends its review updates and then the record of an outward message (role
    assistant, typed outward dialogue, bound to its text at ``tools/owner_delivery``): no answer to those results, so
    the batch stays with the newest exchange, and a window only it could pay for cuts nothing, also once my human
    answers that message or speaks after a plain batch, until the mind replies after it. Completed speech in a call
    is the same kind without the binding: it is the mind's answer, so the batch it answered pays before my human's
    older instruction, also when a receipt published after that speech closes it, and a call keeps sliding."""
    from ouroboros.context_budget import HOST_CONTEXT_KIND_KEY
    from ouroboros.review_history_view import REVIEW_CONTEXT_INDEX_KEY
    from ouroboros.send_clock import CLOCK_ROW_NAME
    price = estimate_context_prompt_tokens

    def call(*named):
        return {'role': 'assistant', 'content': '', 'tool_calls': [{'id': call_id, 'type': 'function', 'function': {
            'name': name, 'arguments': '{}'}} for call_id, name in named]}

    def spoken(text):
        return {'role': 'assistant', HOST_CONTEXT_KIND_KEY: 'owner_dialogue', 'content': text}
    head = [{'role': 'system', 'content': 'Core.'}, {'role': 'user', 'content': 'Assignment.'}]
    batch = [call(('r', 'read_file'), ('s', 'send_user_message')),
             {'role': 'tool', 'tool_call_id': 'r', 'content': 'Unseen result. ' * 400},
             {'role': 'tool', 'tool_call_id': 's', 'content': 'OK: delivered'}]
    def outward(text):  # as ``tools/owner_delivery`` writes it
        return {**bound_host_row('owner_dialogue', '[Owner-directed dialogue; host delivery facts, not owner '
                                 'instructions]\n{}\n[Exact addressed text]\n' + text), 'role': 'assistant'}
    receipt = outward('Done, see the report.')
    review = {'role': 'user', 'content': 'Current review decisions.', REVIEW_CONTEXT_INDEX_KEY: True}
    clock = bound_host_row(CLOCK_ROW_NAME, 'Current time.')
    owner = bound_host_row('owner_words', 'Newest owner turn.')
    for tail in ([receipt, clock], [review, receipt, clock], [receipt, outward('And the log.'), clock],
                 [receipt, bound_host_row('owner_words', 'Thanks.'), clock],
                 [bound_host_row('owner_words', 'Anything else?'), clock]):
        rows = [*head, owner, *batch, *tail]
        before = copy.deepcopy(rows)
        kept, cut = physical_window_cut(rows, drive_root=tmp_path, task_id='closed', measure=price,
                                       usable=price(rows) - price([batch[1]]) // 2)
        assert cut is None and kept == rows == before, 'the batch nobody answered stays'
    answered = [*head, owner, *batch, receipt, clock, {'role': 'assistant', 'content': 'The report says it is green.'}]
    kept, cut = physical_window_cut(answered, drive_root=tmp_path, task_id='answered', measure=price,
                                   usable=price(answered) - price([batch[1]]) // 2)
    assert {3, 4, 5} <= set(cut['positions']), 'answered, the batch is older work'
    older = bound_host_row('owner_words', 'Keep the staging database. ' * 120)
    answer = [*head, older, spoken('Understood.'), bound_host_row('owner_words', 'What does the log say?'),
              call(('l', 'read_file')), {'role': 'tool', 'tool_call_id': 'l', 'content': 'Log entry. ' * 400},
              spoken('The log says the job finished at noon.')]
    kept, cut = physical_window_cut(answer, drive_root=tmp_path, task_id='spoken', measure=price,
                                   usable=price(answer) - price([answer[6]]) // 2)
    assert cut['positions'] == [5, 6] and older in kept, 'the answered batch pays before the older instruction'
    queued = [*answer, outward('The log says the job finished at noon.')]
    kept, cut = physical_window_cut(queued, drive_root=tmp_path, task_id='queued', measure=price,
                                   usable=price(queued) - price([queued[6]]) // 2)
    assert cut['positions'] == [5, 6] and older in kept, 'a receipt after the spoken answer walks back no further'
    meeting = [*head, bound_host_row('owner_words', 'First question.'), spoken('First answer. ' * 300),
               call(('t', 'read_file')), {'role': 'tool', 'tool_call_id': 't', 'content': 'Looked up. ' * 400},
               spoken('Here is what I found. ' * 50), bound_host_row('owner_words', 'Second question.'),
               spoken('Second answer.')]
    kept, cut = physical_window_cut(meeting, drive_root=tmp_path, task_id='call', measure=price,
                                   usable=price(meeting) - price([meeting[5]]) // 2)
    assert cut['positions'] == [4, 5], 'the older batch slides out of the call first'


def test_host_state_its_producer_would_restate_stays_and_a_superseded_copy_really_pays(tmp_path, monkeypatch):
    """The pre-send producers state their current facts again whenever no visible copy does, so cutting the current
    mark changes or roster chain would free nothing for the next request; a roster snapshot that a later full snapshot
    superseded is not stated again and really pays."""
    from ouroboros import peer_roster
    from ouroboros.mark_anchors import append_mark_changes
    from tests.test_mark_anchors import _chronicle_write, _memory_mark, _running, ctx_for, mark
    from tests.test_peer_roster_changes import _roster
    price = estimate_context_prompt_tokens
    head = [{'role': 'system', 'content': 'Core.'}, {'role': 'user', 'content': 'Assignment.'},
            bound_host_row('owner_words', 'Current owner turn.')]
    peer, reader = ctx_for(tmp_path, 'writer'), _running(tmp_path, 'reader')
    assert json.loads(_chronicle_write(peer, kind='note', text='Journal baseline'))['ok']
    for i in range(9):
        made = mark(peer, f'Consider measured case {i}.', {'tool': 'read_file'}, scope='global')
        assert json.loads(_memory_mark(peer, release_ids=[made['mark_id']], reason=f'Case {i} was measured.'))['ok']
    rows = copy.deepcopy(head)
    assert append_mark_changes(reader, rows)
    rows.append({'role': 'assistant', 'content': 'I have assessed these results.'})
    kept, receipt = physical_window_cut(rows, drive_root=tmp_path, task_id='marks', measure=price,
                                       usable=price(rows) - price([rows[3]]) // 3)
    assert receipt is None and not append_mark_changes(reader, kept)
    state = _roster(32)
    monkeypatch.setattr(peer_roster, 'independent_roots', lambda _root: copy.deepcopy(state))
    observer = SimpleNamespace(task_id='observer', drive_root=tmp_path, task_metadata={})
    rows = copy.deepcopy(head)
    assert peer_roster.maybe_append_roster_note(observer, rows, tmp_path)
    state['roots'][7]['waiting'] = [{'kind': 'owner', 'since': '2026-10-09T13:00:00Z', 'quiz_id': 'q1'}]
    assert peer_roster.maybe_append_roster_note(observer, rows, tmp_path)
    rows.append({'role': 'assistant', 'content': 'Current exchange.'})
    kept, receipt = physical_window_cut(rows, drive_root=tmp_path, task_id='roster', measure=price,
                                       usable=price(rows) - price([rows[3]]) // 3)
    assert receipt is None and not peer_roster.maybe_append_roster_note(observer, kept, tmp_path)
    state['roots'][2]['status'] = 'pending'
    restarted = SimpleNamespace(task_id='observer', drive_root=tmp_path, task_metadata={})  # no process-local chain
    assert peer_roster.maybe_append_roster_note(restarted, rows, tmp_path)  # a full snapshot supersedes the chain
    rows.append({'role': 'assistant', 'content': 'The roster changed; going on.'})
    kept, receipt = physical_window_cut(rows, drive_root=tmp_path, task_id='roster', measure=price,
                                       usable=price(rows) - price([rows[3]]) // 3)
    assert receipt['positions'] == [3] and rows[6] in kept
    assert not peer_roster.maybe_append_roster_note(restarted, kept, tmp_path), 'a superseded copy is not restated'


def test_a_row_its_producer_cannot_vouch_for_goes_with_the_dialogue(tmp_path):
    """Only a row whose producer still vouches for its visible text is machine material: a host row bound to its
    text, an address record with valid metadata, a dispute body bound to its review source. A typed row written
    before that binding, one an older merge appended my human's reply to (``test_context_provenance``), an address
    record without valid metadata and an unbound dispute body wait with the dialogue; no text is read."""
    from ouroboros.context_budget import HOST_CONTEXT_KIND_KEY
    from ouroboros.context_view_facts import CONTEXT_FACTS_NAME
    from ouroboros.review_history_view import REVIEW_HISTORY_MESSAGE_KEY
    price = estimate_context_prompt_tokens
    head = [{'role': 'system', 'content': 'Core.'}, {'role': 'user', 'content': 'Assignment.'}]
    tail = [bound_host_row('owner_words', 'Current owner turn.'), {'role': 'assistant', 'content': 'Current exchange.'}]
    words = 'My correction stands. ' * 1100
    merged = bound_host_row(CONTEXT_FACTS_NAME, 'Round facts: measured.')
    merged['content'] += '\n\n---\n\n[Message from my human]: ' + words
    tool = [{'role': 'assistant', 'content': '', 'tool_calls': [{'id': 'c1', 'type': 'function',
             'function': {'name': 'read_file', 'arguments': '{}'}}]},
            {'role': 'tool', 'tool_call_id': 'c1', 'content': 'Prior output. ' * 3000}]
    for row, spent in ((bound_host_row('tool_manifest', words), [4]), (merged, [2, 3]),
                       ({'role': 'user', HOST_CONTEXT_KIND_KEY: 'tool_manifest', 'content': words}, [2, 3]),
                       ({'role': 'user', 'content': [{'type': 'text', 'text': words, '_context_capsule': {}}]},
                        [2, 3])):
        rows = [*head, *tool, row, *tail]
        kept, receipt = physical_window_cut(rows, drive_root=tmp_path, task_id='vouch', measure=price,
                                           usable=price(rows) - 1500)
        assert receipt['positions'] == spent and (row in kept) is (spent != [4]), row.get(HOST_CONTEXT_KIND_KEY)
    for bound, spent in ((True, [3]), (False, [2])):
        dispute = {'role': 'user', 'content': words, REVIEW_HISTORY_MESSAGE_KEY: {'version': 1, 'task_id': 'task',
            'visible_sha256': hashlib.sha256(words.encode('utf-8')).hexdigest() if bound else '0' * 64,
            'binding': {'kind': 'review_history_body', 'field': 'finding', 'content_sha256': 'b' * 64,
                        'source_ref': {'path': 'reviews/wave.json', 'size': 10, 'sha256': 'a' * 64}}}}
        rows = [*head, bound_host_row('owner_dialogue', 'Earlier outward reply. ' * 1000), dispute, *tail]
        kept, receipt = physical_window_cut(rows, drive_root=tmp_path, task_id='dispute', measure=price,
                                           usable=price(rows) - 1500)
        assert receipt['positions'] == spent, f'bound={bound}'


def test_a_chosen_read_keeps_its_whole_batch_with_an_image_before_its_result_at_the_cut_and_in_the_seed(tmp_path):
    """A viewing call appends its image while its batch runs, so the image row sits between the calls and their
    results (``tools/vision.py``). A range the mind asked for keeps its whole batch at the cut and in the renewal seed;
    a plain read, and a range whose call never started, leave with their batch."""
    from ouroboros.context_renewal import renewal_seed
    from ouroboros.loop_messages import _append_or_merge_user_content
    from ouroboros.tool_result_record import (TOOL_RESULT_RECORD_KEY, make_not_started_tool_result_record,
                                              make_tool_result_record)
    price = estimate_context_prompt_tokens
    body = 'Reading evidence. ' * 700
    ranged = {'path': 'facts.md', 'start_line': 1, 'max_chars': 8000}
    recorded = make_tool_result_record({'invocation_id': 'read-1', 'tool_call_id': 'r', 'execution_id': 'exec',
                                        'round_id': 'round', 'task_attempt': 1}, body)
    for arguments, record, chosen in ((ranged, recorded, True), ({'path': 'facts.md'}, recorded, False),
                                      (ranged, make_not_started_tool_result_record('r', body), False)):
        rows = [{'role': 'system', 'content': 'Required core.'}, {'role': 'user', 'content': 'Assignment.'},
                bound_host_row('owner_words', 'Read the range and look at the screenshot. ' * 40),
                {'role': 'assistant', 'content': '', 'tool_calls': [
                    {'id': 'r', 'type': 'function', 'function': {'name': 'read_file',
                                                                 'arguments': json.dumps(arguments)}},
                    {'id': 'v', 'type': 'function', 'function': {'name': 'view_image',
                                                                 'arguments': '{"path": "shot.png"}'}}]}]
        _append_or_merge_user_content(rows, [{'type': 'text', 'text': '[image: shot.png]'}, {
            'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,iVBORw0KGgo='},
            '_caption': '[image: shot.png]', '_source_path': '/original/shot.png'}])
        result = {'role': 'tool', 'tool_call_id': 'r', 'content': body, TOOL_RESULT_RECORD_KEY: record}
        rows += [result, {'role': 'tool', 'tool_call_id': 'v', 'content': 'Attached.'},
                 {'role': 'assistant', 'content': 'Working on the next check.'}]
        kept, receipt = physical_window_cut(rows, drive_root=tmp_path, task_id='chosen', measure=price,
                                           usable=price(rows) - 50)
        if chosen:
            assert receipt is None and kept == rows, 'the chosen range keeps its batch'
        else:
            assert receipt['positions'] == [3, 4, 5, 6], arguments
        assert (result in renewal_seed(rows)[0]) is chosen



def test_a_call_entry_that_is_no_call_is_no_chosen_read_and_raises_nowhere():
    """A reply's tool-call entry that is not a mapping with a mapping function (a provider may send one) is no chosen
    read: the renewal seed and the cut pass over it as before, and its batch leaves like any other."""
    from ouroboros.context_renewal import _cut_units, renewal_seed
    head = [{'role': 'system', 'content': 'Core.'}, {'role': 'user', 'content': 'Assignment.'},
            bound_host_row('owner_words', 'Turn.')]
    for calls in ([None], ['x'], [{'id': 'a', 'type': 'function', 'function': None}],
                  [{'id': 'a', 'type': 'function', 'function': 'read_file'}]):
        rows = [*head, {'role': 'assistant', 'content': '', 'tool_calls': calls},
                {'role': 'tool', 'tool_call_id': 'a', 'content': 'R'}, {'role': 'assistant', 'content': 'after'}]
        seed, left = renewal_seed(rows)
        assert left['tool_rows'] == 2 and rows[4] not in seed, calls
        assert {3, 4} <= {i for group in _cut_units(rows)[0] for i in group}, calls


def test_a_chosen_read_too_large_for_the_window_leaves_with_its_image_batch_and_a_fitting_one_stays(tmp_path):
    """As at the address rungs, a chosen read stays only while it fits the known usable input alone: one larger than
    the window leaves the renewal seed with its batch and the view is renewed; one that fits stays, and with nothing
    else to spend the refusal keeps the view."""
    from ouroboros.loop_messages import _append_or_merge_user_content
    from ouroboros.tool_result_record import TOOL_RESULT_RECORD_KEY, make_tool_result_record
    price = estimate_context_prompt_tokens
    for size, oversized in ((4000, True), (300, False)):
        body = 'Reading evidence. ' * size
        record = make_tool_result_record({'invocation_id': 'read-1', 'tool_call_id': 'r', 'execution_id': 'exec',
                                          'round_id': 'round', 'task_attempt': 1}, body)
        rows = [{'role': 'system', 'content': 'Required core.'}, {'role': 'user', 'content': 'Assignment.'},
                bound_host_row('owner_words', 'Read the range and look at the screenshot. ' * 40),
                {'role': 'assistant', 'content': '', 'tool_calls': [
                    {'id': 'r', 'type': 'function', 'function': {'name': 'read_file', 'arguments': json.dumps(
                        {'path': 'facts.md', 'start_line': 1, 'max_chars': 80000})}},
                    {'id': 'v', 'type': 'function', 'function': {'name': 'view_image',
                                                                 'arguments': '{"path": "shot.png"}'}}]}]
        _append_or_merge_user_content(rows, [{'type': 'text', 'text': '[image: shot.png]'}, {
            'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,iVBORw0KGgo='},
            '_caption': '[image: shot.png]', '_source_path': '/original/shot.png'}])
        rows += [{'role': 'tool', 'tool_call_id': 'r', 'content': body, TOOL_RESULT_RECORD_KEY: record},
                 {'role': 'tool', 'tool_call_id': 'v', 'content': 'Attached.'}]
        window = price([row for row in rows if row.get('content') != body]) + 1500 if oversized else price(rows) - 200
        ctx = SimpleNamespace(messages=rows, accumulated_usage={'_context_refusal_round': 1,
            '_context_capacity_total_tokens': window, '_context_response_reserve_tokens': 0}, drive_root=tmp_path,
            drive_logs=tmp_path / 'logs', task_id=f'chosen-{size}', round_idx=1, event_queue=None, tool_schemas=[])
        assert renew_after_refusal(ctx, SimpleNamespace(_ctx=SimpleNamespace(messages=rows))) is oversized
        assert any(row.get('content') == body for row in rows) is not oversized


def _recorded(call_id, text):
    from ouroboros.tool_result_record import TOOL_RESULT_RECORD_KEY, make_tool_result_record
    record = make_tool_result_record({'invocation_id': 'inv-' + call_id, 'tool_call_id': call_id,
                                      'execution_id': 'exec', 'round_id': 'round', 'task_attempt': 1}, text)
    return {'role': 'tool', 'tool_call_id': call_id, 'content': text, TOOL_RESULT_RECORD_KEY: record}


def _viewing_batch(owner, calls):
    """Core, assignment, my human's turn, a batch with a viewing call, and the image that call attached (popped)."""
    from ouroboros.loop_messages import _append_or_merge_user_content
    rows = [{'role': 'system', 'content': 'Required core.'}, {'role': 'user', 'content': 'Assignment.'},
            bound_host_row('owner_words', owner), {'role': 'assistant', 'content': '', 'tool_calls': [
                {'id': call_id, 'type': 'function', 'function': {'name': name, 'arguments': json.dumps(arguments)}}
                for call_id, name, arguments in (*calls, ('v', 'view_image', {'path': 'shot.png'}))]}]
    _append_or_merge_user_content(rows, [{'type': 'text', 'text': '[image: shot.png]'}, {
        'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,iVBORw0KGgo='},
        '_caption': '[image: shot.png]', '_source_path': '/original/shot.png'}])
    return rows, rows.pop()


def test_a_small_chosen_read_beside_a_large_plain_read_leaves_with_its_batch_when_the_batch_cannot_fit(tmp_path):
    """A chosen read keeps its whole exchange, so the seed and the cut keep that exchange only while all its rows fit
    the known usable input alone: a small range read beside a large plain read in an image batch leaves with the batch
    and the view is renewed; a batch that fits stays whole and the refusal keeps the view."""
    from ouroboros.context_renewal import _cut_units
    price = estimate_context_prompt_tokens
    for plain, renewed in ((3000, True), (60, False)):
        rows, image = _viewing_batch('Read the range, the log and the screenshot. ' * 40, (
            ('r', 'read_file', {'path': 'facts.md', 'start_line': 1, 'max_chars': 4000}),
            ('p', 'read_file', {'path': 'run.log'})))
        batch = [image, _recorded('r', 'Range line. ' * 60), _recorded('p', 'Log line. ' * plain),
                 _recorded('v', 'Attached.')]
        rows += [*batch, {'role': 'assistant', 'content': 'Working on the next check.'}]
        window = price(rows) - (price([batch[2]]) // 2 if renewed else 200)
        assert ({3, 4, 5, 6, 7} <= {i for group in _cut_units(rows, usable_input=window)[0] for i in group}) is renewed
        ctx = SimpleNamespace(messages=rows, accumulated_usage={'_context_refusal_round': 1,
            '_context_capacity_total_tokens': window, '_context_response_reserve_tokens': 0}, drive_root=tmp_path,
            drive_logs=tmp_path / 'logs', task_id=f'sibling-{plain}', round_idx=1, event_queue=None, tool_schemas=[])
        assert renew_after_refusal(ctx, SimpleNamespace(_ctx=SimpleNamespace(messages=rows))) is renewed
        assert all(row in rows for row in batch) is not renewed


def test_a_released_image_batch_is_named_in_the_renewal_index_and_takes_the_image_it_attached(tmp_path):
    """When the seed lets an image batch go (its chosen range cannot fit the window), the record's index names each of
    its calls with its typed facts, never shown to me without usable-send evidence, and the image its viewing call
    attached before the results leaves with it. An image attached after the results is a delivered row and stays, its
    contiguous batch named shown to me by the same evidence; a batch that fits stays whole."""
    from ouroboros.context_budget import HOST_CONTEXT_KIND_KEY
    from ouroboros.context_renewal import RENEWAL_RECORD_NAME
    from tests._context_source_fixtures import _observation
    price = estimate_context_prompt_tokens
    for place, size, renewed in (('between', 4000, True), ('after', 4000, True), ('between', 300, False)):
        rows, image = _viewing_batch('Read this range and inspect its screenshot.', (
            ('r', 'read_file', {'path': 'facts.md', 'start_line': 1, 'max_chars': 80000}),))
        results = [_recorded('r', 'Reading evidence. ' * size), _recorded('v', 'Image attached.')]
        rows += [*([image] if place == 'between' else []), *results, *([image] if place == 'after' else []),
                 {'role': 'assistant', 'content': 'Now continue.'}]
        ctx = SimpleNamespace(messages=rows, accumulated_usage={'_context_refusal_round': 1,
            '_context_capacity_total_tokens': price([row for row in rows if row is not results[0]]) + 1500,
            '_context_response_reserve_tokens': 0}, drive_root=tmp_path, drive_logs=tmp_path / 'logs',
            task_id=f'index-{place}-{size}', round_idx=1, event_queue=None, tool_schemas=[])
        tools = SimpleNamespace(_ctx=SimpleNamespace(messages=rows, _last_context_observation=_observation(rows)))
        assert renew_after_refusal(ctx, tools) is renewed
        record = next((row for row in rows if row.get(HOST_CONTEXT_KIND_KEY) == RENEWAL_RECORD_NAME), {})
        index = str(record.get('content') or '').partition('Tool units since my newest note')[2]
        if renewed:
            assert all(fact in index for fact in ('read_file', 'view_image', 'invocation inv-r', 'invocation inv-v'))
            assert ('never shown to me' in index) is (place == 'between') and 'shown to me' in index, index
        assert (image in rows) is (place == 'after' or not renewed), 'no image outlives the batch that attached it'
        assert (results[0] in rows) is not renewed
    from ouroboros.context_renewal import units_since_note
    rows, _ = _viewing_batch('Read the page.', (('r', 'read_file', {'path': 'page.html'}),))
    rows += [_recorded('r', '<img src="data:image/png;base64,iVBORw0KGgo=">'), _recorded('v', 'Image attached.')]
    assert 'invocation inv-r' in units_since_note(rows, None), 'inline image data has no unit, yet its batch is named'


def test_an_obsolete_mark_row_pays_before_the_owners_words_and_the_producer_sends_nothing_more(tmp_path):
    """Another work makes a mark and later releases it: the creation row states nothing the release row does not state
    anew, so the mark producer would not send it again (``mark_anchors.superseded_change_rows``); it is a superseded
    host copy and pays before my human's older instruction. The release row and an independent current mark, each
    the only witness of its mark, stay."""
    from ouroboros.mark_anchors import append_mark_changes
    from tests.test_mark_anchors import _chronicle_write, _memory_mark, _running, ctx_for, mark
    price = estimate_context_prompt_tokens
    older = bound_host_row('owner_words', 'Older owner instruction. ' * 700)
    rows = [{'role': 'system', 'content': 'Core.'}, {'role': 'user', 'content': 'The assignment.'}, older,
            {'role': 'assistant', 'content': 'I read that.'}, bound_host_row('owner_words', 'New owner instruction.')]
    writer, reader = ctx_for(tmp_path, 'writer'), _running(tmp_path, 'reader')
    assert json.loads(_chronicle_write(writer, kind='note', text='Journal baseline'))['ok']
    made = mark(writer, 'Measured evidence. ' * 500, {'tool': 'read_file'}, scope='global')
    assert made['ok'] and append_mark_changes(reader, rows)
    creation = rows[-1]
    rows.append({'role': 'assistant', 'content': 'Noted the mark.'})
    assert json.loads(_memory_mark(writer, release_ids=[made['mark_id']], reason='Measured and released.'))['ok']
    assert append_mark_changes(reader, rows)
    release = rows[-1]
    assert mark(writer, 'Another current mark.', {'tool': 'read_file'}, scope='global')['ok']
    assert append_mark_changes(reader, rows)
    independent = rows[-1]
    rows.append({'role': 'assistant', 'content': 'Current exchange.'})
    kept, receipt = physical_window_cut(rows, drive_root=tmp_path, task_id='marks', measure=price,
                                       usable=price(rows) - 1000)
    assert receipt['positions'] == [rows.index(creation)] and older in kept
    assert release in kept and independent in kept and not append_mark_changes(reader, kept)


def test_an_earlier_facts_row_and_a_consumed_clock_pay_before_an_owner_direction(tmp_path):
    """The next request writes its own facts row and stamps its own clock, so samples an earlier request consumed free
    their room for good: they pay before my human's older direction, the mind's prose with them."""
    from hashlib import sha256
    from datetime import datetime, timezone
    from ouroboros.artifacts import actor_source_reference
    from ouroboros.context_renewal import physical_cut_record
    from ouroboros.context_view_facts import CONTEXT_FACTS_NAME
    from ouroboros import send_clock
    price = estimate_context_prompt_tokens
    direction = bound_host_row('owner_words', 'Owner specifies what must be verified. ' * 130)
    clock = send_clock.render_clock_note(send_clock.SendClockPolicy('UTC'),
                                         datetime(2026, 11, 27, 6, tzinfo=timezone.utc))
    rows = [{'role': 'system', 'content': 'Core.'}, {'role': 'user', 'content': 'Assignment.'}, direction,
            {'role': 'assistant', 'content': 'Earlier ordinary prose. ' * 65},
            bound_host_row(CONTEXT_FACTS_NAME, '[CONTEXT_FACTS] round 4: window, input, free room. ' * 12),
            bound_host_row(send_clock.CLOCK_ROW_NAME, clock), bound_host_row('owner_words', 'New steering.'),
            {'role': 'assistant', 'content': 'Checked.'}]
    raw = json.dumps(rows, ensure_ascii=False, sort_keys=True, default=str).encode('utf-8')
    source = actor_source_reference(category='context_checkpoints', source_id=sha256(raw).hexdigest(), data=raw,
                                    extension='json')
    usable = price([*rows[:3], physical_cut_record([3, 4, 5], set(), source), *rows[6:]])
    kept, receipt = physical_window_cut(rows, drive_root=tmp_path, task_id='samples', measure=price, usable=usable)
    assert receipt['positions'] == [3, 4, 5] and direction in kept
