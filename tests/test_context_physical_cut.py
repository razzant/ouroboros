"""Only physical pressure can move old delivered whole rows to their exact reader."""
import copy
import json
from types import SimpleNamespace

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
    assert receipt['positions'] == [2, 3, 4]
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
    assert receipt['positions'] == [2], 'the older source goes alone'
    assert old_quote in kept and old_words in kept and kept[-5:] == rows[-5:]
    kept, receipt = physical_window_cut(rows, drive_root=tmp_path, task_id='source-cut-more',
                                       measure=estimate_context_prompt_tokens, usable=total - (source + 20))
    assert receipt['positions'] == [2, 3, 4], 'the older selection leaves only with its words'
    assert kept[-5:] == rows[-5:], 'the newest source, selection and words never leave'


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
