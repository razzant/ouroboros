"""Deferred #1379 custody still preserves exact ACKed owner words for Continue."""
import json

import pytest

from ouroboros import headless, observability, task_custody, owner_mailbox
from ouroboros.task_results import load_task_result
from tests._budget_pause_exact_helpers import _install_queue
from tests.test_owner_continue import _interrupted, _owner_mail, NONCE


@pytest.mark.parametrize('fault', ['none', 'readback', 'source_gap', 'unrecorded_mode'])
def test_deferred_call_inventory_and_owner_words_survive_child_retirement(tmp_path, monkeypatch, fault):
    """The row records the mode its producer prepared, as current producers do; ``unrecorded_mode`` is a historical
    row that names its retired drive without one: unknown, refused before any claim, never guessed."""
    from supervisor.continuation_admission import admit_continuation

    parent, tid = tmp_path / 'parent', 'pred-1'
    _, _, workers = _install_queue(parent, monkeypatch)
    child = headless.prepare_task_drive(parent, tid, 'empty')
    _interrupted(child, **({} if fault == 'unrecorded_mode' else {'memory_mode': 'empty'}))
    _owner_mail(child)
    trace = observability.persist_call(child, task_id=tid, call_id='physical', call_type='tool_call',
                                       payload={'bytes': 'exact physical result'})
    headless.copy_child_task_result(parent, {'id': tid, 'drive_root': str(child)})
    assert task_custody.settle_child_drive(parent, tid, child, live=lambda _: False)['status'] == 'retained'
    headless.retry_child_task_refs(parent, child, tid)
    owner_mailbox.write_owner_message(child, 'Final owner correction', tid, msg_id='after-promotion')
    owner_mailbox.acknowledge_transcript_entry(child, tid, {'msg_id': 'after-promotion'})
    if fault == 'source_gap':
        path = owner_mailbox._mailbox_path(child, tid)
        path.write_text(path.read_text() + '{torn owner row\n')
        outcome = task_custody.settle_child_drive(parent, tid, child, live=lambda _: False)
        assert outcome['status'] == 'retained' and child.exists()
        resumed = admit_continuation(tid, action_nonce=NONCE)
        assert not resumed['ok'] and resumed['error'] == 'owner_source_missing' and not workers.PENDING
        assert not load_task_result(parent, tid).get('continued_by')
        return
    if fault == 'readback':
        original = task_custody._write_custody_fields
        def lose_owner(root, task_id, observed, fields_for, **kw):
            return original(root, task_id, observed, lambda row: {
                k: v for k, v in fields_for(row).items() if k != 'owner_mailbox'}, **kw)
        with monkeypatch.context() as broken:
            broken.setattr(task_custody, '_write_custody_fields', lose_owner)
            outcome = task_custody.settle_child_drive(parent, tid, child, live=lambda _: False)
        assert outcome['status'] == 'retained' and child.exists()
    outcome = task_custody.settle_child_drive(parent, tid, child, live=lambda _: False)
    assert outcome['status'] == 'removed', outcome
    assert not child.exists()
    stored = load_task_result(parent, tid)
    words = [json.loads(raw)['text'] for raw in stored['owner_mailbox']['rows']]
    assert words == ['Also add the charts', "Use last week's numbers", 'Final owner correction']
    manifest = observability.read_call_manifest_ref(parent, trace['manifest_ref'], task_id=tid)
    assert observability.read_blob_ref(parent, manifest['full_payload_ref']) == {'bytes': 'exact physical result'}
    resumed = admit_continuation(tid, action_nonce=NONCE)
    if fault == 'unrecorded_mode':
        assert stored.get('headless_child_drive_root') and not stored.get('memory_mode')
        assert resumed == {'ok': False, 'error': 'continuation_drive_mode_unknown'} and not workers.PENDING
        assert not load_task_result(parent, tid).get('continued_by')
        return
    assert resumed['ok'], resumed
    [successor] = workers.PENDING
    assert successor['memory_mode'] == 'empty' and successor['drive_root'] != str(child)
    assert 'It recorded no memory export.' in successor['text'] and str(child) not in successor['text']
    assert successor['metadata']['billing_group']['billing_group_id'] == tid
    assert successor['deadline_at'].startswith('2099')
    assert [r['content'] for r in successor['metadata']['owner_corpus']] == [
        'Write the Friday report', 'Also add the charts', "Use last week's numbers", 'Final owner correction']
