"""Sealed story yields before owner words; recent labels and accounts keep their meaning."""
import pytest

from ouroboros import memory_floor as mf, memory_view as mv
from tests import _memory_view_synthetic as syn
from tests.test_memory_view_floor import tokens, frame


def story_snapshot():
    pages = syn.pages(3, 8000)
    for n, page in enumerate(pages):
        page['text'] = f'News from page {n}: ' + 'a' * 200 + '.\nA second sentence. ' + 'details ' * 1000
    account = {**pages[0], 'kind': 'account', 'id': 'resident-account', 'selection': 'selected-account',
               'text': 'My authored account across rooms. ' * 80, 'sources': []}
    owner = {**syn.spoken(0, 'human', 4000), 'owner_side': True, 'carried': True}
    return syn.snapshot(story=[*pages, account], room_facts=syn.room(lane1=[owner]))


def test_pages_step_through_first_sentences_then_oldest_pointers_before_owner_words():
    snap = story_snapshot()
    assert mf.fit_memory_view(snap, frame(tokens(snap))) == mv.FULL_VIEW
    labels = mf.level_with(snap, mv.FULL_VIEW, [f'F5l:{p["id"]}' for p in snap.story[:3]])
    chosen = mf.fit_memory_view(snap, frame(tokens(snap, labels)))
    assert chosen == labels
    text = mv.render_story(snap, chosen)
    assert 'A second sentence.' not in text
    assert all(p['text'].split('\n', 1)[0] in text for p in snap.story[:3])
    # One more shortage takes the oldest page's label, preserving the newest news.
    older = mf.level_with(snap, labels, [f'F5:{snap.story[0]["id"]}'])
    chosen = mf.fit_memory_view(snap, frame(tokens(snap, older)), kept=labels)
    assert dict(chosen.addressed)['F5'] == (snap.story[0]['id'],)
    assert snap.story[2]['text'].split('\n', 1)[0] in mv.render_story(snap, chosen)
    assert 'F7' not in dict(chosen.steps)
    assert snap.room['lane1'][0]['line'] in mv.render_room(snap, chosen)
    from ouroboros import memory_inventory as mi
    facts = mf.view_facts(snap, labels, window_tokens=85000, mode='nano', allowances={})
    shortage = mi.shortage_from_trace({mi.VIEW_TRACE_KEY: facts})
    assert shortage.kind == 'narrative' and shortage.pointer_records == tuple(p['id'] for p in snap.story[:3])


def test_accounts_have_no_floor_rung_even_when_the_minimum_cannot_fit():
    snap = story_snapshot()
    account = snap.story[-1]
    assert not any(ident == account['id'] for _step, ident, *_ in mf.floor_elements(snap))
    with pytest.raises(ValueError, match='protected'):
        mf.level_with(snap, mv.FULL_VIEW, ['F5:' + account['id']])
    level = mf.fit_memory_view(snap, frame(0, 0))
    text = mv.render_story(snap, level)
    assert account['text'] in text
    assert text.index('### My account across rooms') < text.index('### My older pages')
    assert mf.minimal_view_tokens(snap) >= len(account['text']) // 4


def test_pointer_has_one_id_and_label_does_not_cut_a_long_first_sentence():
    snap = story_snapshot()
    page = snap.story[0]
    pointer = mv._page_pointer(page)
    assert pointer.count(page['id']) == 1 and f"memory_read(node_id='{page['id']}')" in pointer
    assert page['text'].split('\n', 1)[0] in mv._page_label(page)
    # A one-sentence page whose label is no smaller goes directly to its pointer.
    page['text'] = 'One short sentence.'
    assert not any(s == 'F5l' and i == page['id'] for s, i, *_ in mf.floor_elements(snap))
