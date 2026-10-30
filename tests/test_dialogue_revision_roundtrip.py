"""Revised rows retain their canonical identity through storeless-body round trips."""
import pytest

from ouroboros import dialogue_store as ds, dialogue_import as di, dialogue_positions as dp, store
from tests.test_dialogue_store import chat, progress, _legacy_root
from tests._dialogue import chat_rows, seed


@pytest.mark.parametrize('revised', [False, True])
def test_revised_caption_survives_older_body_export_and_return(tmp_path, revised):
    _legacy_root(tmp_path)
    di.cutover(tmp_path)
    first = ds.append(tmp_path, 'chat', chat('', 'Initial caption.', direction='out',
        source='live_call', spoken={'call_id': 'review-call', 'item_id': 'item', 'status': 'played'}))
    following = ds.append(tmp_path, 'chat', chat('after-call', 'The next owner message.'))
    expected = dict(first.row)
    if revised:
        expected['text'] = 'Initial caption. A late transcription correction.'
        ds.revise_text(tmp_path, 'chat', first.seq, expected['text'])
    di.export_for_older_body(tmp_path)
    seed(tmp_path, live={'chat': [chat('older-body', 'Owner message after rollback.')]})
    di.cutover(tmp_path)
    rows = chat_rows(tmp_path)
    selected = [row for row in rows if row.get('source') == 'live_call'
                or row.get('client_message_id') in {'after-call', 'older-body'}]
    assert selected == [expected, following.row, chat('older-body', 'Owner message after rollback.')], [
        (row['text'], row.get('client_message_id')) for row in selected]
    assert ds.accepted_row(tmp_path, 1, 'older-body') is not None


@pytest.mark.parametrize('stream', ['chat', 'progress'])
@pytest.mark.parametrize('index', [0, 1])
@pytest.mark.parametrize('replacement', ['x', 'Длинная правка. ' * 30])
def test_revision_roundtrip_preserves_spans_counts_and_multiple_batches(tmp_path, stream, index, replacement):
    _legacy_root(tmp_path)
    di.cutover(tmp_path)
    make = chat if stream == 'chat' else progress
    receipts = [ds.append(tmp_path, stream, make(f'new-{n}', f'Original row {n}.')) for n in range(3)]
    expected = [receipt.row for receipt in receipts]
    expected[index] = ds.revise_text(tmp_path, stream, receipts[index].seq, replacement)
    di.export_for_older_body(tmp_path)
    receipts.append(ds.append(tmp_path, stream, make('second-batch', 'Another export batch.')))
    expected.append(receipts[-1].row)
    di.export_for_older_body(tmp_path)
    older = make('older-body', 'Written by the older body.')
    seed(tmp_path, live={stream: [older]})
    report = di.cutover(tmp_path)
    di.run_dialogue_import(tmp_path)  # settles segment bases after the delayed archive import
    selected = [row for row in ds.newest(tmp_path, stream, limit=1000) if row.seq >= receipts[0].seq]
    assert [row.row for row in selected] == [*expected, older]
    assert [(row.seq, row.length, row.dispatch) for row in selected[:-1]] == [
        (receipt.seq, receipt.length, receipt.dispatch) for receipt in receipts]
    ordinals = [dp.ordinal_at(tmp_path, stream, row.seq).ordinal for row in selected]
    assert ordinals == list(range(ordinals[0], ordinals[0] + len(selected)))
    if stream == 'chat':
        with ds.transaction(tmp_path, write=False) as conn:
            gen = conn.execute('SELECT gen FROM import_ledger WHERE name=?', (report[stream]['rotated'],)).fetchone()['gen']
        for number, row in enumerate(selected, 1):
            assert ds.resolve_legacy(tmp_path, gen, number).seq == row.seq
            assert dp.generation_row(tmp_path, gen, number - 1).seq == row.seq
    assert di.export_for_older_body(tmp_path)[str(tmp_path)][stream]['rows'] == 0
    # The physical chain and the store now have different ends. A later round trip
    # must still place both the new store row and the older body's row exactly once.
    later = ds.append(tmp_path, stream, make('later', 'After the return.'))
    di.export_for_older_body(tmp_path)
    seed(tmp_path, live={stream: [make('older-again', 'Second downgrade.')]})
    di.cutover(tmp_path)
    assert [row.row for row in ds.newest(tmp_path, stream, limit=1000) if row.seq >= later.seq] == [
        later.row, make('older-again', 'Second downgrade.')]


@pytest.mark.parametrize('resume_export', [False, True])
@pytest.mark.parametrize('stop_import', [False, True])
def test_revision_map_survives_interrupted_export_and_cutover(tmp_path, monkeypatch, resume_export, stop_import):
    di.cutover(tmp_path)
    first = ds.append(tmp_path, 'chat', chat('', 'Original.', direction='out'))
    following = ds.append(tmp_path, 'chat', chat('following', 'Next message.'), dispatch={'text': 'Next message.'})
    revised = ds.revise_text(tmp_path, 'chat', first.seq, 'Corrected text, longer than the allocated span.')

    def died(*_args):
        raise SystemExit('interrupted')

    with monkeypatch.context() as patch:
        patch.setattr(di, 'utc_now_iso', died)
        with pytest.raises(SystemExit):
            di.export_for_older_body(tmp_path)
    store.close_connections()
    if resume_export:
        di.export_for_older_body(tmp_path)
    seed(tmp_path, live={'chat': [chat('older-body', 'Older message.')]})
    if stop_import:
        with monkeypatch.context() as patch:
            patch.setattr(di, 'import_segment', died)
            with pytest.raises(SystemExit):
                di.cutover(tmp_path)
        store.close_connections()
    di.cutover(tmp_path)
    assert chat_rows(tmp_path) == [revised, following.row, chat('older-body', 'Older message.')]
    assert ds.row_at(tmp_path, 'chat', first.seq).length == first.length
    assert ds.accepted_row(tmp_path, 1, 'following').seq == following.seq
    assert ds.dispatch_state(tmp_path, 1, 'following') == 'pending'


def test_torn_revised_export_does_not_leave_mappings_for_unwritten_rows(tmp_path, monkeypatch):
    di.cutover(tmp_path)
    receipts = [ds.append(tmp_path, 'chat', chat(str(n), f'Original {n}.')) for n in range(3)]
    revised = ds.revise_text(tmp_path, 'chat', receipts[0].seq, 'Longer revision. ' * 10)
    with monkeypatch.context() as patch:
        patch.setattr(di, 'utc_now_iso', lambda: (_ for _ in ()).throw(SystemExit('interrupted')))
        with pytest.raises(SystemExit):
            di.export_for_older_body(tmp_path)
    live = tmp_path / 'logs' / 'chat.jsonl'
    live.write_bytes((ds.encode(revised) + '\n').encode('utf-8') + b'{"text"')
    di.cutover(tmp_path)
    assert ds.meta(tmp_path, 'export:chat')['coordinates'] == []
    assert di.export_for_older_body(tmp_path)[str(tmp_path)]['chat']['rows'] == 1
    assert live.read_bytes() == (ds.encode(receipts[2].row) + '\n').encode('utf-8')
    di.cutover(tmp_path)
    assert chat_rows(tmp_path) == [revised, receipts[1].row, receipts[2].row]


@pytest.mark.parametrize('changed', [0, 1])
def test_export_map_does_not_alias_changed_bytes_to_a_canonical_row(tmp_path, changed):
    di.cutover(tmp_path)
    first = ds.append(tmp_path, 'chat', chat('', 'Original.', direction='out'))
    following = ds.append(tmp_path, 'chat', chat('', 'Change me.', direction='out'))
    revised = ds.revise_text(tmp_path, 'chat', first.seq, 'Corrected and extended.')
    di.export_for_older_body(tmp_path)
    live = tmp_path / 'logs' / 'chat.jsonl'
    before, after = (b'Corrected', b'Different') if changed == 0 else (b'Change me.', b'New words.')
    live.write_bytes(live.read_bytes().replace(before, after))
    di.cutover(tmp_path)
    modified = ({**revised, 'text': 'Different and extended.'} if changed == 0
                else {**following.row, 'text': 'New words.'})
    assert chat_rows(tmp_path) == [revised, following.row, modified]


def test_revised_export_keeps_identity_when_rotation_sorts_before_old_archives(tmp_path):
    _legacy_root(tmp_path)
    di.cutover(tmp_path)
    first = ds.append(tmp_path, 'chat', chat('', 'Original.', direction='out'))
    following = ds.append(tmp_path, 'chat', chat('following', 'Next message.'))
    revised = ds.revise_text(tmp_path, 'chat', first.seq, 'Corrected text with a longer span.')
    di.export_for_older_body(tmp_path)
    seed(tmp_path, live={'chat': [chat('older-body', 'Older message.')]})
    (tmp_path / 'logs' / 'chat.jsonl').rename(tmp_path / 'archive' / 'chat_20000101T000000.jsonl')
    di.cutover(tmp_path)
    selected = [row.row for row in ds.newest(tmp_path, 'chat', limit=1000) if row.seq >= first.seq]
    assert selected == [revised, following.row, chat('older-body', 'Older message.')]


@pytest.mark.parametrize('revised', [False, True])
def test_torn_export_missing_only_last_newline(tmp_path, monkeypatch, revised):
    """R8 reviewer regression, including the unchanged-span control."""
    di.cutover(tmp_path)
    first = ds.append(tmp_path, 'chat', chat('', 'First.', direction='out'))
    last = ds.append(tmp_path, 'chat', chat('', 'Last.', direction='out'))
    revised_row = ds.revise_text(tmp_path, 'chat', last.seq, 'Last with a late correction.') if revised else last.row
    payload = b''.join((ds.encode(row) + '\n').encode('utf-8') for row in (first.row, revised_row))
    write = di.os.write
    partial = False

    def cut_after_json(fd, data):
        nonlocal partial
        raw = bytes(data)
        if raw == payload:
            partial = True
            return write(fd, raw[:-1])
        if partial and raw == b'\n':
            raise SystemExit('process stopped between a short write and the final LF')
        return write(fd, data)

    with monkeypatch.context() as scope:
        scope.setattr(di.os, 'write', cut_after_json)
        with pytest.raises(SystemExit):
            di.export_for_older_body(tmp_path)
    live = tmp_path / 'logs/chat.jsonl'
    assert live.read_bytes() == payload[:-1]
    assert ds.meta(tmp_path, 'export_batch:chat') is not None
    di.cutover(tmp_path)
    assert chat_rows(tmp_path) == [first.row, revised_row]
    assert ds.row_at(tmp_path, 'chat', last.seq).length == last.length


@pytest.mark.parametrize('tail_kind', ['different', 'malformed'])
def test_unterminated_export_tail_requires_the_exact_recorded_object(tmp_path, tail_kind):
    di.cutover(tmp_path)
    first = ds.append(tmp_path, 'chat', chat('', 'First.', direction='out'))
    last = ds.append(tmp_path, 'chat', chat('', 'Last.', direction='out'))
    revised = ds.revise_text(tmp_path, 'chat', last.seq, 'Corrected final text.')
    di.export_for_older_body(tmp_path)
    tail = ds.encode(revised).encode('utf-8')
    if tail_kind == 'different':
        tail = tail.replace(b'Corrected', b'Different')  # same byte length, different hash
    else:
        tail = tail[:-1] + b'!'  # same byte length, incomplete JSON
    live = tmp_path / 'logs/chat.jsonl'
    live.write_bytes((ds.encode(first.row) + '\n').encode('utf-8') + tail)
    di.cutover(tmp_path)
    expected = [first.row, revised]
    if tail_kind == 'different':
        expected.append({**revised, 'text': 'Different final text.'})
    assert chat_rows(tmp_path) == expected
    assert ds.range_coverage(tmp_path, 'chat')['gaps'] == int(tail_kind == 'malformed')
