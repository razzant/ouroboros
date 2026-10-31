"""Motion in the stylesheets (docs/DESIGN.md "Motion" and §10 "Moving between places"): every loop that runs by
itself stops under reduced motion. A census: a looping animation added without its reduced-motion stop fails here."""
from __future__ import annotations

import re
from pathlib import Path

WEB = Path(__file__).resolve().parents[1] / "web"
REDUCED = re.compile(r"@media\s*\(prefers-reduced-motion:\s*reduce\)\s*\{")


def _blocks(css: str) -> tuple[str, list[str]]:
    """The sheet without its reduced-motion blocks, and those blocks' bodies."""
    outside, inside, at = [], [], 0
    for match in REDUCED.finditer(css):
        if match.start() < at:
            continue
        depth, i = 1, match.end()
        while depth and i < len(css):
            depth += {"{": 1, "}": -1}.get(css[i], 0)
            i += 1
        outside.append(css[at:match.start()])
        inside.append(css[match.end():i - 1])
        at = i
    outside.append(css[at:])
    return "".join(outside), inside


def _rules(css: str):
    for match in re.finditer(r"([^{}]+)\{([^{}]*)\}", css):
        yield [" ".join(part.split()) for part in match.group(1).split(",")], match.group(2)


def test_every_loop_stops_under_reduced_motion():
    loops, stops = [], set()
    for sheet in sorted(WEB.glob("*.css")):
        css = re.sub(r"/\*.*?\*/", "", sheet.read_text(encoding="utf-8"), flags=re.S)
        outside, reduced = _blocks(css)
        for selectors, body in _rules(outside):
            if re.search(r"animation:[^;]*\binfinite\b", body):
                loops += [(sheet.name, selector) for selector in selectors]
        for block in reduced:
            for selectors, body in _rules(block):
                if re.search(r"animation(-name)?:\s*none", body):
                    stops.update(selectors)
    assert loops, "the census found no loop at all: its parser no longer reads the sheets"
    assert ("ui.css", '.status-mark[data-shape="moving"] > i') in loops
    missing = [loop for loop in loops if loop[1] not in stops]
    assert not missing, f"loops that keep running under reduced motion: {missing}"
