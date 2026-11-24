"""The book-yield gate keeps a delegated helper's Low view out of the map yields.

Low yields a map only for an integrating focus, so a re-send never shortens a delegated helper's optional DESIGN
map on its unread Low start. Nano keeps its optional map: an unread Nano start yields it even for a helper. A view
a reply already read keeps its chosen forms, and a view without the book yields nothing.
"""
from types import SimpleNamespace

import ouroboros.loop_round_limits as limits


def _ctx(mode, role, sections=('design_bible',)):
    plan = SimpleNamespace(
        core=SimpleNamespace(required_book_tokens=lambda m: 4000 if m == mode else 0,
                             dynamic_head_sections=tuple(SimpleNamespace(section_id=s) for s in sections)),
        projection=lambda m: SimpleNamespace(memory_facts={'role': role, 'host_sections_addressed': ()}),
    )
    plan.low_projection = plan.projection('low') if mode == 'low' else None
    plan.nano_projection = plan.projection('nano') if mode == 'nano' else None
    return SimpleNamespace(active_context_mode=mode, context_fit_plan=plan,
                           tools=SimpleNamespace(_ctx=SimpleNamespace()), accumulated_usage={})


def test_a_helpers_low_view_takes_no_map_yields(monkeypatch):
    monkeypatch.setattr(limits, 'start_unread', lambda *a, **k: True)
    monkeypatch.setattr('ouroboros.context_fit._integrating', lambda core, role: role is None)
    assert limits.book_yield_due(_ctx('low', 'child')) is False  # a helper's unread Low start: none, as reviewed
    assert limits.book_yield_due(_ctx('low', None)) is True  # an integrating Low start keeps every yield
    assert limits.book_yield_due(_ctx('nano', 'child')) is True  # Nano takes its optional map alone


def test_a_read_view_or_view_without_the_book_never_yields(monkeypatch):
    monkeypatch.setattr('ouroboros.context_fit._integrating', lambda core, role: role is None)
    read = _ctx('low', None)
    monkeypatch.setattr(limits, 'start_unread', lambda *a, **k: False)
    assert limits.book_yield_due(read) is False  # a reply read its input: the view keeps its chosen forms
    monkeypatch.setattr(limits, 'start_unread', lambda *a, **k: True)
    bare = _ctx('nano', 'child', sections=())
    assert limits.book_yield_due(bare) is False  # no optional section left to take: nothing due
