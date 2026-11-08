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
