"""Every aggregate still leads to the exact retained rows and story records."""
import re

from ouroboros import memory_view as mv
from ouroboros.memory_floor import drawn_tokens, story_measure
from ouroboros.chronicle_store import ChronicleStore
from ouroboros.tools import chronicle
from tests.test_chronicle_store import page, MIND
from tests.test_chronicle_tools import addr, chat, ctx_for, ts


def test_story_directory_pages_all_ids_and_their_exact_records(tmp_path, monkeypatch):
    store = ChronicleStore(tmp_path)
    store.ensure_activated()
    records = [page(store, f'source-{i}', room='1', at=i, text=f'Exact page {i}.').record for i in range(12)]
    part = store.publish_part(room_id='1', text='Exact part.', member_ids=[r['id'] for r in records[:3]],
                              author=MIND, expected_sequence=store.room_head('1')).record
    monkeypatch.setattr(chronicle, 'tool_result_limit', lambda name: 600)
    ctx, start, chunks = ctx_for(tmp_path), 0, []
    while True:
        text = chronicle._memory_read(ctx, story=True, start=start)
        head, continuation, body = text.split('\n', 2)
        assert head.startswith('My story: 10 pages, parts and accounts; drawn whole ~') and len(text) <= 600
        chunks.append(body)
        if 'complete' in continuation:
            break
        next_start = int(re.search(r'next_start=(\d+)', continuation)[1])
        assert next_start > start and f'memory_read(story=true, start={next_start})' in continuation
        start = next_start
    listing = ''.join(chunks)
    expected, status, refusals = mv.capture_story(store, tmp_path, mv.room_labeler(tmp_path))
    measure = story_measure(expected, status, refusals)
    shown = [item for item in expected if item.get('kind') != 'legacy' and not item.get('told_by')]
    total = sum(drawn_tokens(item) for item in shown)
    # One group per room the records already carry, its drawn weight, kinds and reader; then each record exactly.
    assert listing.split('\n')[0] == f"## Main [room 1]: ~{total:,} tokens; 9 pages, 1 part; memory_read(room_id='1')"
    revisions = {item['id']: item['revision'] for item in store.interpret([store.get(i['id']) for i in shown], sources=False)}
    assert listing.split('\n')[1:] == [f"{mv._page_pointer(item)}; revision {revisions[item['id']]}; "
                                       f"~{drawn_tokens(item):,} tokens" for item in shown]
    assert head == (f"My story: 10 pages, parts and accounts; drawn whole ~{measure['tokens']:,} tokens, "
                    f"{measure['unfolded']['pages']} pages no part or account holds yet ~{measure['unfolded']['tokens']:,}; "
                    "0 records told by accounts, not drawn")
    # The unfolded weight is the start's Max-room measure of the same pages (``memory_floor.unfolded_pages``).
    assert measure['unfolded']['tokens'] == sum(drawn_tokens(item) for item in shown if item['kind'] == 'page')
    for record in [part, *records[3:]]:
        assert record['id'] in listing
        assert record['text'] in chronicle._memory_read(ctx, node_id=record['id'])
    for arguments in ({'rows': True}, {'room_id': '1'}, {'node_id': records[0]['id']}):
        assert 'TOOL_ARG_ERROR' in chronicle._memory_read(ctx, story=True, **arguments)
    # The directory interprets every listed record inside one capture: one index read, never one per record.
    import contextlib

    opened, index = [], ChronicleStore._index

    @contextlib.contextmanager
    def counted(self):
        opened.append(1)
        with index(self) as db:
            yield db

    monkeypatch.setattr(ChronicleStore, '_index', counted)
    assert chronicle._read_story(tmp_path, store, 0, 600) == chronicle._read_story(tmp_path, store, 0, 600)
    assert len(opened) == 2, opened


def test_aggregate_row_range_includes_interleaved_verbatim_rows(tmp_path):
    rows = chat(tmp_path, [{'chat_id': 1, 'direction': 'out' if i % 2 == 0 else 'in',
                           'ts': ts(i), 'text': f'Exact original words {i}.'} for i in range(5)])
    items = [dict(pos=i, address=addr(row), ts=row['ts'], kind='ouroboros') for i, row in enumerate(rows) if i in (0, 4)]
    aggregate = mv._rows_line({'room_id': '1'}, items)
    assert f"from='{addr(rows[0])}', to='{addr(rows[4])}'" in aggregate
    result = chronicle._memory_read(ctx_for(tmp_path), room_id='1', rows=True,
                                   **{'from': addr(rows[0]), 'to': addr(rows[4])})
    for row in rows:
        assert row['text'] in result and addr(row) in result


def test_aggregate_account_keeps_nested_told_records_and_reader(tmp_path):
    from tests.test_common_story import _rooms, _snapshot
    _rooms_map, _alpha, _beta, first, second, _helper = _rooms(tmp_path)
    store = ChronicleStore(tmp_path)
    sources = [first, second]
    for index in range(2):
        account = store.publish_account(room_id='1', text=f'Nested account {index}.',
                                        sources=sources, author=MIND).record['id']
        assert store.select_account(account, replaces=sources, author=MIND, reason='Told here.').ok
        sources = [account]
    snapshot = _snapshot(tmp_path)
    pointers = mv.FloorLevel((('F5', (account,)),))
    aggregate = mv.FloorLevel((*pointers.addressed, ('F5a', ('story',))))
    told = '3 story records told through this account (including nested selections)'
    assert told in mv.render_story(snapshot, pointers)
    shown = mv.render_story(snapshot, aggregate)
    assert told in shown and f"memory_read(node_id='{account}')" in shown
    read = chronicle._memory_read(ctx_for(tmp_path), node_id=account)
    assert 'Nested account 1.' in read and 'replaces 1:' in read
