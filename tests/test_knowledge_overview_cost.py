"""Overview cost belongs to its stored body revision, never a live request counter."""
import json
from types import SimpleNamespace

from ouroboros import knowledge as store
from ouroboros.context_sections import build_knowledge_sections
from ouroboros.tools.knowledge import _knowledge_write
from ouroboros.tools.registry import ToolContext


def test_overview_receipt_prices_body_before_after_and_next_capture(tmp_path):
    env = SimpleNamespace(drive_root=tmp_path, drive_path=lambda p: tmp_path / p)
    ctx = ToolContext(repo_dir=tmp_path, drive_root=tmp_path, task_id='overview-cost')
    address = store.resolve_knowledge_address(tmp_path, 'overview', 'global')
    old = store.write_knowledge_note(address, '---\ntype: note\nsummary: Metadata stays outside the body count.\n---\n'
                                    '# Overview\n\n' + 'Дословное понимание. ' * 4500 + 'x').current
    before = build_knowledge_sections(env)[0]
    reply = _knowledge_write(ctx, 'overview', content='# Overview\n\nA smaller authored map.', mode='overwrite',
                             expected_revision=old.revision)
    assert reply.startswith('✅')
    receipt = json.loads(reply.split('\n', 1)[1])
    current = store.read_knowledge_note(address)
    a, b = store.body_chars(old), store.body_chars(current)
    assert a != len(old.text) and b < a
    assert receipt['knowledge_delta']['old_body_chars'] == a
    assert receipt['knowledge_delta']['new_body_chars'] == b
    residency = receipt['knowledge_residency']
    assert f'Body {a} → {b} chars' in residency
    assert a > 89000 and a % 4 != 0
    assert f'estimate {a / 4:.2f} → {b / 4:.2f} tokens' in residency
    assert f'delta {(b - a) / 4:+.2f} tokens' in residency
    assert 'every integrating context (Main, tasks, wakes and Presence)' in residency
    assert 'next context build' in residency and 'earlier capture' in residency
    after = build_knowledge_sections(env)[0]
    assert store.overview_residency(current) in after and current.source.text_at(current.source.body_span) in after
    assert after == build_knowledge_sections(env)[0]  # no clock/route numbers
    assert store.overview_residency(old) in before and old.source.text_at(old.source.body_span) in before
    assert after.operative_view is None
    # A stale revision neither changes the page nor issues a successful delta.
    refused = _knowledge_write(ctx, 'overview', content='Stale overwrite', mode='overwrite',
                               expected_revision=old.revision)
    assert 'revision_conflict' in refused and 'knowledge_residency' not in refused
    assert store.read_knowledge_note(address).revision == current.revision
    unchanged = _knowledge_write(ctx, 'overview', content=current.text, mode='overwrite',
                                 expected_revision=current.revision)
    assert 'delta +0.00 tokens' in unchanged


def test_ordinary_notes_remain_on_demand_and_overview_creation_prices_zero(tmp_path):
    ctx = ToolContext(repo_dir=tmp_path, drive_root=tmp_path, task_id='overview-create')
    ordinary = _knowledge_write(ctx, 'work/report', content='# Report\nDetails.', mode='overwrite')
    assert 'read on demand' in ordinary and 'next context build' not in ordinary
    created = _knowledge_write(ctx, 'overview', content='# Overview\nNew understanding.', mode='overwrite')
    assert 'Body 0 → ' in created and 'estimate 0.00 → ' in created


def test_the_resident_overview_carries_its_revision_after_its_body(tmp_path):
    """The revision covers the whole note, its summary too, so it follows the body: a summary-only edit of the
    overview moves only the section's last line, never a byte in front of the overview body (the cached prefix)."""
    env = SimpleNamespace(drive_root=tmp_path, drive_path=lambda p: tmp_path / p)
    address = store.resolve_knowledge_address(tmp_path, 'overview', 'global')
    body = '# Overview\n\nWhat I understand.\n'
    first = store.write_knowledge_note(address, '---\nsummary: Before.\n---\n' + body).current
    section = build_knowledge_sections(env)[0]
    assert section.endswith(f"{body}\nRevision of this overview: {first.revision}.\n")
    assert build_knowledge_sections(env)[0] == section  # nothing else moves it
    second = store.write_knowledge_note(address, '', 'edit', first.revision, summary='After.').current
    moved = build_knowledge_sections(env)[0]
    through_body = section.index(body) + len(body)
    assert moved[:through_body] == section[:through_body]  # every byte up to the body's end is unchanged
    assert moved.endswith(f"Revision of this overview: {second.revision}.\n") and first.revision not in moved
