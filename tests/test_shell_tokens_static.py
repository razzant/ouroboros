"""The shell's stacking order is a token scale in ui.css and the shell's hover rules live under
`@media (hover: hover)` (docs/DESIGN.md "Shell", plan P9): a sticky hover tint on a phone after a tap was the
live card's defect, and twenty magic z-index numbers were the shell's."""
from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
Z_TOKENS = ("--z-page-header", "--z-sidebar", "--z-room", "--z-resize-handle", "--z-project-scrim", "--z-project-panel",
            "--z-project-layer", "--z-drawer-scrim", "--z-popover")
SHELL_RULES = {
    "web/style.css": ("#primary-sidebar {", ".resize-handle {", ".chat-page-header {",
                      'html[data-layout="stack"] .nav-drawer-backdrop {', 'html[data-layout="stack"] #primary-sidebar {',
                      'html[data-layout="stack"] #content {',
                      ".project-panel {", ".project-panel-backdrop {", 'html[data-layout="stack"] .project-panel {'),
    "web/ui.css": (".ui-popup {",),
}
GATED_HOVERS = (".nav-row:hover", ".nav-project-kebab:hover", ".nav-projects-add:hover", ".resize-handle:hover",
                ".chat-header-btn:hover", ".chat-header-btn.danger:hover",
                ".chat-budget-pill:hover", ".nav-drawer-close):hover", ".chat-budget-pill.ui-chrome-control:hover",
                ".chat-project-row:hover")


def _read(rel: str) -> str:
    return (REPO / rel).read_text(encoding="utf-8")


def _blocks(css: str, head: str):
    """Bodies of every `@media <head>` block, balanced."""
    for match in re.finditer(r"@media\s*" + re.escape(head) + r"\s*\{", css):
        depth, pos = 1, match.end()
        while depth and pos < len(css):
            depth += {"{": 1, "}": -1}.get(css[pos], 0)
            pos += 1
        yield css[match.end():pos - 1]


def test_shell_rules_read_the_z_tokens_and_the_scale_stays_ordered():
    ui = _read("web/ui.css")
    values = {name: int(re.search(rf"{re.escape(name)}: (\d+);", ui).group(1)) for name in Z_TOKENS}
    assert list(values.values()) == sorted(values.values()), values
    assert values["--z-project-scrim"] < values["--z-project-panel"]
    # On a phone the drawer lies under the room it pushes aside (a Project open over Main is that room), and the clear
    # scrim that brings the room back lies over all of them.
    assert values["--z-sidebar"] < values["--z-room"] < values["--z-project-layer"] < values["--z-drawer-scrim"], values
    for rel, selectors in SHELL_RULES.items():
        css = _read(rel)
        for selector in selectors:
            body = css.split(selector, 1)[1].split("}", 1)[0]
            z = re.search(r"z-index:\s*([^;]+);", body)
            assert z and z.group(1).startswith("var(--z-"), f"{rel} {selector} {z.group(1) if z else 'has no z-index'}"


def test_shell_hover_rules_are_gated_on_a_hovering_pointer():
    css = _read("web/style.css")
    gated = "\n".join(_blocks(css, "(hover: hover)"))
    outside = css
    for body in _blocks(css, "(hover: hover)"):
        outside = outside.replace(body, "")
    for selector in GATED_HOVERS:
        if selector not in css:
            continue
        assert selector in gated, f"{selector} has no gated recipe"
        assert selector not in outside, f"{selector} also paints on a touch screen (outside @media (hover: hover))"
    assert "transform: scale(1.02)" not in css, "a hover never moves (DESIGN 'Colour answers touch')"
