"""Synthetic tests of the real knowledge/chronicle writers and their prospective readers.

No model, endpoint, installation state or hosted task is used.
"""
import json
from contextlib import contextmanager

import pytest

from ouroboros import held_notes, knowledge, knowledge_versions, nano_publication
from ouroboros.chronicle_store import ChronicleStore, capture, prospective_chronicle
from ouroboros.context_sections import build_knowledge_sections
from ouroboros.knowledge import prospective_knowledge
from ouroboros.tools.chronicle import _chronicle_write, _memory_mark
from ouroboros.tools.knowledge import _knowledge_write
from ouroboros.tools.registry import ToolContext


def ctx_for(root):
    return ToolContext(repo_dir=root, drive_root=root, task_id='nano0001', current_chat_id=1)


class Probe:
    """Count instrument for exercising the writers. The production fence still owns publication."""

    def __init__(self, root, observe):
        self.root = root
        self.observe = observe

    @contextmanager
    def ordinary(self):
        yield False

    @contextmanager
    def publication(self, changes):
        if isinstance(changes, nano_publication.MemoryChange):
            changes = (changes,)
        with prospective_knowledge(self.root, changes), prospective_chronicle(self.root, changes):
            after = self.observe(changes)
        if callable(after):
            after()
        with nano_publication.publication_fence(self.root):
            yield


def address(root, topic):
    return knowledge.resolve_knowledge_address(root, topic, 'global')


def snapshot_index(root, topic):
    return knowledge.knowledge_index_views(address(root, topic))


def knowledge_section(root):
    class Env:
        def drive_path(self, *parts):
            return root.joinpath(*parts)

    return str(build_knowledge_sections(Env())[0])


def shelves(root):
    folder = root / 'memory'
    return {p.relative_to(folder): p.read_bytes() for p in folder.rglob('*') if p.is_file() and not p.name.endswith('.lock')}


def test_person_proposal_changes_index_title_summary_and_claim_in_one_write(tmp_path, monkeypatch):
    ctx = ctx_for(tmp_path)
    item = address(tmp_path, 'people/rook')
    seed = knowledge.write_knowledge_note(item, '# Old title\nOld head\n', writer='turn')
    assert seed.ok
    before = shelves(tmp_path)
    text = '---\ntype: person\nsummary: A short witness\nclaims: [room:1]\n---\n# Rook revised\nInhabits a paper tower\n## Further notes\n'
    seen = []

    def observe(changes):
        assert {change.subject for change in changes} == {'knowledge_note', 'knowledge_claims'}
        assert item.path.read_bytes() == seed.current.raw
        assert shelves(tmp_path) == before
        whole, outline = snapshot_index(tmp_path, item.topic)
        for view in (whole, outline):
            assert 'Rook revised' in view and 'A short witness' in view and 'Old title' not in view
        assert knowledge.active_claims(tmp_path) == {'room:1': [item]}
        seen.append((whole, outline))

    monkeypatch.setattr(nano_publication, 'admission_for', lambda *_: Probe(tmp_path, observe))
    result = _knowledge_write(ctx, topic=item.topic, content=text, mode='overwrite', expected_revision=seed.current.revision)
    assert 'Rook revised' in item.path.read_text(), result
    assert seen == [snapshot_index(tmp_path, item.topic)]
    assert knowledge.active_claims(tmp_path) == {'room:1': [item]}


def test_staged_overview_refusal_leaves_active_note_and_index_unchanged(tmp_path):
    item = address(tmp_path, 'overview')
    seed = knowledge.write_knowledge_note(item, '# Baseline\nBefore\n', writer='turn')
    assert seed.ok
    before = shelves(tmp_path)

    def observe(_changes):
        assert 'Staged view' in knowledge_section(tmp_path)
        assert item.path.read_bytes() == seed.current.raw
        raise nano_publication.NanoPublicationRefused('over_capacity', 'Synthetic count exceeded')

    with pytest.raises(nano_publication.NanoPublicationRefused, match='Synthetic count exceeded'):
        knowledge.write_knowledge_note(item, '# Staged view\nGrowth\n', expected_revision=seed.current.revision,
                                       admission=Probe(tmp_path, observe), writer='turn')
    assert shelves(tmp_path) == before
    assert 'Baseline' in knowledge_section(tmp_path)


@pytest.mark.parametrize('admit', [False, True])
def test_new_note_activates_link_in_an_already_held_head(tmp_path, monkeypatch, admit):
    hub = address(tmp_path, 'people/hub')
    target = address(tmp_path, 'fresh note')
    assert knowledge.write_knowledge_note(
        hub, '---\nclaims: [room:1]\n---\n# Hub\n[New](../fresh%20note.md)\n', writer='turn').ok

    def linked():
        heads, gaps = held_notes.linked_heads(knowledge.read_knowledge_note(hub))
        return {topic: held_notes.note_head(note.source) for topic, note in heads.items()}, gaps

    before = linked()
    assert not before[0] and before[1]
    checked = []

    def observe(_changes):
        assert not target.path.is_file()
        checked.append(linked())
        assert checked[-1] == ({target.topic: '\n# New text\nA known value\n'}, [])
        if not admit:
            raise nano_publication.NanoPublicationRefused('over_capacity', 'Synthetic linked head limit')

    monkeypatch.setattr(nano_publication, 'admission_for', lambda *_: Probe(tmp_path, observe))
    text = '# New text\nA known value\n## Later\nThis part is not held.\n'
    result = _knowledge_write(ctx_for(tmp_path), topic=target.topic, scope='global', mode='overwrite', content=text)
    assert len(checked) == 1
    if admit:
        assert target.path.read_text().endswith(text) and linked() == checked[0]
    else:
        assert 'Synthetic linked head limit' in result and not target.path.exists()
        assert linked() == before


def test_claim_write_interruption_does_not_activate_an_unchecked_claim(tmp_path, monkeypatch):
    from ouroboros.knowledge_publication import intermediate_projections

    item = address(tmp_path, 'staged')
    original = knowledge.write_knowledge_note(item, '---\nclaims: [room:1]\n---\n# Before\n', writer='turn')
    assert original.ok
    updated = '---\nclaims: [room:2]\n---\n# After\n'
    compared = []

    def observe(changes):
        assert knowledge.active_claims(tmp_path) == {'room:2': [item]}
        pending = tuple(intermediate_projections(changes))
        assert len(pending) == 1
        with prospective_knowledge(tmp_path, pending[0]):
            compared.append(knowledge.active_claims(tmp_path))

    def interrupt(*_args, **_kwargs):
        raise OSError('Synthetic claims publication interruption')

    with monkeypatch.context() as patch:
        patch.setattr(knowledge_versions.ClaimWrite, 'publish', interrupt)
        result = knowledge.write_knowledge_note(item, updated, expected_revision=original.current.revision,
                                                writer='turn', admission=Probe(tmp_path, observe))
    assert not result.ok and result.reason == 'publication_incomplete'
    assert item.path.read_text() == result.current.text and '# After\n' in result.current.text
    assert compared == [knowledge.active_claims(tmp_path)] == [{}]
    assert knowledge.write_knowledge_note(item, updated, expected_revision=result.current.revision,
                                           writer='turn', admission=Probe(tmp_path, lambda _: None)).ok
    assert knowledge.active_claims(tmp_path) == {'room:2': [item]}


def test_archive_restores_same_source_and_checks_active_navigation(tmp_path):
    item = address(tmp_path, 'people/rook')
    seed = knowledge.write_knowledge_note(item, '---\ntype: person\nsummary: Rook witness\nclaims: [room:1]\n---\n# Source\n',
                                         writer='turn')
    assert seed.ok
    views = []

    def observe(_changes):
        note = knowledge.read_knowledge_note(item)
        views.append((note.state, 'Rook witness' in snapshot_index(tmp_path, item.topic)[1],
                      'room:1' in knowledge.active_claims(tmp_path)))

    probe = Probe(tmp_path, observe)
    archived = knowledge.write_knowledge_note(item, '', mode='archive', reason='Synthetic reduction',
                                             expected_revision=seed.current.revision, admission=probe, writer='turn')
    assert archived.ok and archived.current.state == 'archived' and '# Source' in archived.current.text
    restored = knowledge.write_knowledge_note(item, '', mode='restore', expected_revision=archived.current.revision,
                                             admission=probe, writer='turn')
    assert restored.ok and restored.current.metadata == seed.current.metadata
    assert knowledge.body_chars(restored.current) == knowledge.body_chars(seed.current)
    assert restored.current.text.endswith('# Source\n')
    assert views == [('archived', False, False), ('active', True, True)]


def test_knowledge_rechecks_revision_without_holding_the_old_lock(tmp_path):
    item = address(tmp_path, 'test')
    seed = knowledge.write_knowledge_note(item, '# Baseline')

    def competing_write():
        assert knowledge.write_knowledge_note(item, '# Winner', expected_revision=seed.current.revision).ok
    with pytest.raises(nano_publication.NanoPublicationRefused, match='Knowledge shelf changed'):
        knowledge.write_knowledge_note(item, '# Stale', expected_revision=seed.current.revision,
                                       admission=Probe(tmp_path, lambda _: competing_write))
    assert knowledge.read_knowledge_note(item).text.endswith('# Winner')


def test_mark_projection_and_rollback_via_real_tool(tmp_path, monkeypatch):
    ctx = ctx_for(tmp_path)
    store = ChronicleStore(tmp_path)
    note = json.loads(_chronicle_write(ctx, kind='note', text='original Synthetic Fact'))
    assert note['ok']
    origin = note['node_id']
    observed = []

    def observe(changes):
        txn = json.loads(changes[0].after)
        check = txn['records'][0]
        assert store.log_path.stat().st_size == int(changes[0].before)
        with capture(tmp_path, tail='none'):
            marks = store.active_marks('1')
            observed.append((check['kind'], [(m['text'], m['visibility']) for m in marks]))
            assert store.get(origin)['text'] == 'original Synthetic Fact'
        if check['kind'] == 'mark' and check['text'] == 'Too long':
            raise nano_publication.NanoPublicationRefused('over_capacity', 'Synthetic mark limit')

    monkeypatch.setattr(nano_publication, 'admission_for', lambda *_: Probe(tmp_path, observe))
    created = json.loads(_memory_mark(ctx, node_id=origin, quote='Synthetic Fact', text='Retain', scope='global'))
    assert created['ok']
    mark_id = created['mark_id']
    assert json.loads(_memory_mark(ctx, mark_id=mark_id, visibility='meaning', reason='Synthetic reduction'))['ok']
    raw = store.log_path.read_bytes()
    rejected = json.loads(_memory_mark(ctx, node_id=origin, text='Too long'))
    assert rejected['reason'] == 'over_capacity'
    assert store.log_path.read_bytes() == raw
    assert json.loads(_memory_mark(ctx, release_ids=[mark_id], reason='Synthetic release'))['ok']
    assert not store.active_marks('1')
    assert store.get(origin)['text'] == 'original Synthetic Fact'
    assert store.get(mark_id)['quote'] == 'Synthetic Fact'
    assert observed == [('mark', [('Retain', 'full')]), ('mark_view', [('Retain', 'meaning')]),
                        ('mark', [('Retain', 'meaning'), ('Too long', 'full')]), ('mark_release', [])]


def test_chronicle_competing_append_and_idempotent_repeat(tmp_path):
    store = ChronicleStore(tmp_path)
    author = knowledge.focus_signature(ctx_for(tmp_path))
    note = {'kind': 'note', 'room_id': '1', 'id': 'source', 'author': author, 'task_id': author['task_id'],
            'text': 'Original'}
    assert store.publish([note]).ok
    mark = {'kind': 'mark', 'id': 'mark', 'room_id': '1', 'author': author, 'scope': 'global', 'text': 'Keep',
            'target_ref': {'kind': 'chronicle', 'id': 'source'}, 'visibility': 'full'}

    def competing_write():
        assert store.publish([{**note, 'id': 'competing'}]).ok
    candidate = ChronicleStore(tmp_path, admission=Probe(tmp_path, lambda _: competing_write))
    with pytest.raises(nano_publication.NanoPublicationRefused, match='Chronicle changed'):
        candidate.publish([mark])
    assert store.get('mark') is None and store.get('competing')
    assert store.publish([mark]).ok
    raw = store.log_path.read_bytes()
    assert candidate.publish([mark]).reason == 'unchanged'
    assert store.log_path.read_bytes() == raw


def test_shared_broker_refuses_growth_and_retains_transaction_not_journal(tmp_path, monkeypatch):
    from types import SimpleNamespace

    ctx = ctx_for(tmp_path)
    source = json.loads(_chronicle_write(ctx, kind='note', text='Original'))['node_id']
    store = ChronicleStore(tmp_path)
    original_journal = store.log_path.read_bytes()
    retained = []
    stamps = dict(status='exact', context_window=80_000, process_id='synthetic', raw_context_window=80_000,
                  formatter='one', caller_max_tokens=65_536, reply_floor_tokens=8192,
                  minimum_work_headroom_at_reply_floor_tokens=15_000, without_memory={'input_tokens': 36_808})

    def floor(memory):
        return SimpleNamespace(count=lambda: {**stamps, 'memory_tokens': memory})

    def candidate(changes):
        assert store.log_path.read_bytes() == original_journal
        with prospective_chronicle(tmp_path, changes), capture(tmp_path, tail='none'):
            (mark,) = store.active_marks(1)
            return floor(19_995), floor(19_995 + len(mark['text']))

    def retain(changes):
        retained.extend(changes)
        return 'synthetic proposal address'

    gate = nano_publication.NanoPublication(tmp_path, lambda: 'fixed', candidate, retain)
    monkeypatch.setattr(nano_publication, 'admission_for', lambda *_: gate)
    refusal = json.loads(_memory_mark(ctx, node_id=source, text='123456'))
    assert refusal['reason'] == 'over_capacity' and 'synthetic proposal address' in refusal['detail']
    assert store.log_path.read_bytes() == original_journal
    assert len(retained) == 1 and retained[0].path is None
    assert json.loads(retained[0].after)['records'][0]['text'] == '123456'
    assert json.loads(_memory_mark(ctx, node_id=source, text='12345'))['ok']  # exactly at the synthetic limit
    assert len(store.active_marks(1)) == 1


def test_project_note_uses_no_installation_admission(tmp_path, monkeypatch):
    item = knowledge.resolve_knowledge_address(tmp_path, 'project-note', 'project:alpha')

    def must_not_admit(*_args):
        raise AssertionError('the isolated project shelf is not a global memory source')

    monkeypatch.setattr(nano_publication, 'admission_for', must_not_admit)
    monkeypatch.setattr(nano_publication, 'publication_fence', must_not_admit)
    assert knowledge.write_knowledge_note(item, '# Project\n', writer='turn').ok
    assert item.path.read_text().endswith('# Project\n')


def test_ordinary_scratch_appends_never_depend_on_a_nano_count(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    from ouroboros.memory import Memory

    memory = Memory(drive_root=tmp_path)

    class Ordinary(Probe):
        @contextmanager
        def ordinary(self):
            with nano_publication.publication_fence(self.root):
                yield True

    def must_not_count(_changes):
        raise AssertionError('an ordinary write must never leave its original write transaction to count')

    def append(index):
        return memory.append_scratchpad_block(f'Synthetic block {index}', nano_admission=gate)

    gate = Ordinary(tmp_path, must_not_count)
    with ThreadPoolExecutor(max_workers=2) as pool:
        written = list(pool.map(append, (1, 2)))
    assert sorted(row['content'] for row in memory.load_scratchpad_blocks()) == sorted(row['content'] for row in written)


def test_factory_note_grows_through_checked_shelf_writer(tmp_path, monkeypatch):
    from ouroboros import nano_host_publication, own_time_seed

    item = address(tmp_path, own_time_seed.TOPIC)
    checked = []
    refuse = True

    def observe(_changes):
        assert not item.path.exists()
        assert knowledge.active_claims(tmp_path) == {'wake': [item]}
        checked.append(knowledge.read_knowledge_note(item).text)
        if refuse:
            raise nano_publication.NanoPublicationRefused('over_capacity', 'Synthetic factory note limit')

    monkeypatch.setattr(nano_host_publication, 'host_admission', lambda *_: Probe(tmp_path, observe))
    with pytest.raises(nano_publication.NanoPublicationRefused, match='Synthetic factory note limit'):
        own_time_seed._write(item, 'isolated-start')
    assert not item.path.exists() and not knowledge.active_claims(tmp_path)
    refuse = False
    assert own_time_seed._write(item, 'isolated-start') == (own_time_seed.WRITTEN, '')
    assert checked == [item.path.read_text()] * 2
    assert knowledge.active_claims(tmp_path) == {'wake': [item]}
