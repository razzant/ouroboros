"""No count, size or age cap removes working memory (#918; replaces the ibl-2b09abdadd25 caps).

The owner chose blocks that stay until the integrating mind accepts a fold (29 Oct 2026:
«А — блоки, свёртку принимаю я»). Two former layers are gone, each pinned by its inverse here:

1. **Source-side caps** (``Memory.append_scratchpad_block``). The 10-block count cap and the
   60 000-character content cap evicted oldest blocks (``block_evicted``). Now every append
   stays; ``memory.scratchpad_block_sha256`` names a block for ``fold_scratchpad_blocks``.
2. **Consumer-side degradation** (``context._render_scratchpad_for_context``). Above
   ``SCRATCHPAD_SECTION_BUDGET_CHARS`` it dropped the oldest whole blocks with a gap marker.
   Now ``context._scratchpad_for_context`` renders every canonical block through the writer's
   own ``render_scratchpad_markdown`` (tier-0 core); the budget only logs. A window too small
   for the core is ``memory_floor``'s measured fact, not a section cut. A flat pre-block
   scratchpad (no ``scratchpad_blocks.json``) is still returned raw.
"""
from __future__ import annotations

import json
import logging
import re

import pytest

from ouroboros.context import _scratchpad_for_context, build_memory_sections
from ouroboros.context_budget import SCRATCHPAD_SECTION_BUDGET_CHARS
from ouroboros.memory import Memory, render_scratchpad_markdown, scratchpad_block_sha256

# ---------- helpers ----------------------------------------------------------


def _mem(tmp_path):
    drive = tmp_path / "data"
    (drive / "memory").mkdir(parents=True)
    (drive / "logs").mkdir(parents=True)
    mem = Memory(drive_root=drive)
    mem.ensure_files()
    return mem


def _blocks(mem):
    bp = mem.scratchpad_blocks_path()
    return json.loads(bp.read_text(encoding="utf-8")) if bp.exists() else []


def _journal_types(mem):
    jp = mem.journal_path()
    if not jp.exists():
        return []
    return [json.loads(line).get("type") for line in jp.read_text(encoding="utf-8").splitlines() if line.strip()]


def _oversized_labelled(tmp_path, n_blocks=4, each_chars=25_000):
    """Blocks whose rendered section exceeds the section budget; a unique leading label
    per block makes the render order observable by offset."""
    mem = _mem(tmp_path)
    blocks = [
        {"ts": f"2026-09-01T12:0{i}:00+00:00", "source": "forced", "content": f"BLOCK{i}-" + "x" * (each_chars - 8)}
        for i in range(n_blocks)
    ]
    mem.mutate_scratchpad_blocks(lambda _rows: blocks)
    return mem, blocks


def _body(mem):
    return _scratchpad_for_context(mem)[1]


# ---------- 1. no source-side cap ----------------------------------------------


def test_two_large_blocks_above_the_former_content_cap_both_stay(tmp_path):
    """2 x 40_000 = 80_000 characters: the former 60_000 cap evicted the older block."""
    mem = _mem(tmp_path)
    mem.append_scratchpad_block("a" * 40_000, source="task")
    mem.append_scratchpad_block("b" * 40_000, source="task")
    assert [block["content"][0] for block in _blocks(mem)] == ["a", "b"]
    assert "block_evicted" not in _journal_types(mem)


def test_five_blocks_above_the_former_content_cap_all_stay(tmp_path):
    mem = _mem(tmp_path)
    for i in range(5):
        mem.append_scratchpad_block(("c" * 20_000) + f"-{i}", source="task")
    kept = _blocks(mem)
    assert [block["content"][-2:] for block in kept] == [f"-{i}" for i in range(5)]
    assert sum(len(block["content"]) for block in kept) > 100_000


def test_twelve_blocks_above_both_former_caps_all_stay(tmp_path):
    """12 x 10_000: the former single pass evicted down to both caps; now nothing leaves."""
    mem = _mem(tmp_path)
    for i in range(12):
        mem.append_scratchpad_block(("d" * 10_000) + f"-{i}", source="task")
    kept = _blocks(mem)
    assert len(kept) == 12 and kept[0]["content"].endswith("-0") and kept[-1]["content"].endswith("-11")
    assert _journal_types(mem) == ["block_appended"] * 12


def test_small_blocks_append_plainly(tmp_path):
    mem = _mem(tmp_path)
    for i in range(3):
        mem.append_scratchpad_block(f"small block {i}", source="task")
    assert len(_blocks(mem)) == 3
    assert "block_evicted" not in _journal_types(mem)


# ---------- 2. the context section is always whole ----------------------------------------


def test_a_fitting_scratchpad_is_the_writers_file(tmp_path):
    mem = _mem(tmp_path)
    mem.append_scratchpad_block("a normal block", source="task")
    assert _scratchpad_for_context(mem) == ("memory/scratchpad_blocks.json", mem.load_scratchpad())


def test_a_section_above_the_budget_renders_every_block_whole(tmp_path):
    mem, blocks = _oversized_labelled(tmp_path, n_blocks=4, each_chars=25_000)
    raw = mem.load_scratchpad()
    assert len(raw) > SCRATCHPAD_SECTION_BUDGET_CHARS
    body = _body(mem)
    assert body == raw
    assert all(block["content"] in body for block in blocks)
    assert "budget gap" not in body and "omitted" not in body


def test_a_flat_pre_block_scratchpad_is_returned_raw(tmp_path):
    mem = _mem(tmp_path)
    mem.scratchpad_path().write_text("## Scratchpad (legacy flat)\n\nSome legacy note here.\n", encoding="utf-8")
    assert _scratchpad_for_context(mem) == ("memory/scratchpad.md", mem.load_scratchpad())


def test_a_single_block_above_the_budget_renders_whole(tmp_path):
    mem = _mem(tmp_path)
    mem.append_scratchpad_block("z" * 200_000, source="task")
    body = _body(mem)
    assert "z" * 200_000 in body and "budget gap" not in body


def test_every_rendered_block_is_whole_and_ends_at_its_separator(tmp_path):
    mem, blocks = _oversized_labelled(tmp_path, n_blocks=3, each_chars=30_000)
    body = _body(mem)
    matches = list(re.finditer(r"### \[\d{4}-\d{2}-\d{2}T\d{2}:\d{2} — forced\] sha256 [0-9a-f]{64}\n(BLOCK\d-x+)", body))
    assert len(matches) == 3, "every block renders, none is cut"
    for match in matches:
        assert match.group(1) in [block["content"] for block in blocks]
        assert "---" in body[match.end():match.end() + 10]


def test_the_section_budget_only_logs(tmp_path, caplog):
    mem, blocks = _oversized_labelled(tmp_path)
    with caplog.at_level(logging.WARNING, logger="ouroboros.context"):
        [section] = [sec for sec in build_memory_sections(mem, partition="volatile") if sec.startswith("## Scratchpad")]
    assert "Context section 'scratchpad' exceeds budget" in caplog.text
    assert all(block["content"] in section for block in blocks)


# ---------- section header -----------------------------------------------------------------


def _scratchpad_header(mem):
    sections = build_memory_sections(mem, partition="volatile")
    scratchpad = [sec for sec in sections if sec.startswith("## Scratchpad (from")]
    assert len(scratchpad) == 1
    return scratchpad[0].split("\n", 1)[0]


def test_an_oversized_section_header_still_says_already_loaded(tmp_path):
    """Nothing is omitted, so nothing needs a re-read: the former PARTIAL header is gone."""
    mem, _blocks_written = _oversized_labelled(tmp_path)
    header = _scratchpad_header(mem)
    assert "PARTIAL" not in header and "already loaded" in header


def test_the_header_names_the_canonical_blocks(tmp_path):
    mem = _mem(tmp_path)
    mem.append_scratchpad_block("a normal block", source="task")
    assert _scratchpad_header(mem) == (
        "## Scratchpad (from `memory/scratchpad_blocks.json` — already loaded; do not "
        "re-read via read_file(root='runtime_data', path='memory/scratchpad.md'))"
    )


# ---------- render order -------------------------------------------------------------------


def test_the_section_renders_newest_first_like_the_file(tmp_path):
    mem, _ = _oversized_labelled(tmp_path, n_blocks=4, each_chars=25_000)
    body = _body(mem)
    labels = [f"BLOCK{i}-" for i in range(4)]
    assert sorted(labels, key=body.index) == ["BLOCK3-", "BLOCK2-", "BLOCK1-", "BLOCK0-"]
    first_heading = body.index("### [")
    assert body[first_heading:].startswith("### [2026-09-01T12:03 — forced]")


def test_the_section_is_the_writers_rendering_of_every_block(tmp_path):
    mem, blocks = _oversized_labelled(tmp_path, n_blocks=4, each_chars=25_000)
    assert _body(mem) == render_scratchpad_markdown(blocks, journal_pointer=mem.journal_path().exists(),
                                                    journal_reader=mem.scratchpad_journal_reader())
    assert [scratchpad_block_sha256(block) for block in reversed(blocks)] == re.findall(
        r"sha256 ([0-9a-f]{64})", _body(mem))


# ---------- 3. the physical window: whole, or a typed refusal — never a cut -----------------


@pytest.mark.parametrize("window, outcome", [(400_000, "whole"), (40_000, "refused")])
def test_the_route_window_holds_the_whole_scratchpad_or_refuses_typed_never_cuts(tmp_path, monkeypatch, window, outcome):
    """No section cut stands between a large scratchpad and the window. The scratchpad is tier-0 core
    measured in the floor need: a route that holds it gets every block whole; a route whose window
    cannot hold the core refuses the start with the typed ``MinimumViewUnfit`` (measured numbers,
    no request built) rather than dropping a block. Folding is the mind's (``update_scratchpad``
    ``replaces``); this physical limit is the owner-accepted remainder, not a hidden cap."""
    from ouroboros import capability_evidence as ce
    from ouroboros import context_fit as cf
    from ouroboros import memory_view as mv
    from tests import _memory_view_synthetic as syn

    mem = _mem(tmp_path)
    blocks = [{"ts": f"2026-09-01T12:0{i}:00+00:00", "source": "forced", "content": f"BLOCK{i}-" + "s" * 60_000}
              for i in range(5)]
    mem.mutate_scratchpad_blocks(lambda _rows: blocks)
    [section] = [sec for sec in build_memory_sections(mem, partition="volatile") if sec.startswith("## Scratchpad")]
    snapshot = syn.snapshot(role="consciousness", story=syn.pages(1, 100), room_facts=syn.room(lane2=3))
    core = cf.ContextCore("Required core", "Constitution", "b" * 1_000, "", "Identity", section, '"Assignment"',
                          False, memory_view_json=mv.snapshot_json(snapshot))
    monkeypatch.setattr(cf, "_route_calibration_ratio", lambda *_a, **_kw: 1.0)

    def start():
        return cf.build_context_fit_plan(
            None, core, {"id": "room01", "usage_category": "consciousness"}, preferred_mode="max",
            route_resolver=lambda *_a, **_kw: ({"provider": "test", "model": "test"},
                                               ce.CapabilityEvidence(window, "confirmed", "metadata", "room-route")))

    if outcome == "refused":
        with pytest.raises(cf.MinimumViewUnfit, match="cannot fit"):
            start()
        return
    plan = start()
    shown = plan.projection(plan.initial_mode).system_content_json
    assert plan.initial_mode == "max"
    assert all(block["content"] in shown for block in blocks) and "budget gap" not in shown
