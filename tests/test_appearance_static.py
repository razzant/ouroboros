"""Static guards for the appearance pass: the facts a browser cannot show.

Three of the four claims here are invisible to the Playwright suite:

* ``private_mode=False`` decides whether the DESKTOP shell keeps localStorage
  across a full quit-and-reopen. The browser tests drive a browser, and a server
  restart never reproduced the loss — only closing the app did. This source guard verifies the flag, not
  actual retention: the rebuilt desktop launcher still needs a cold-launch test.
* a colour baked into a ``data:`` URI cannot be themed, because a CSS custom
  property cannot be interpolated into the URI string. The fix is to make the
  whole image a token, or to cut the shape with a black SVG mask out of a token
  background; either is a structural fact about the stylesheet.
* the appearance choice is client-local. ``web/modules/settings.js`` collects
  ``input[id^="s-"]`` into the ``/api/settings`` payload, so the guarantee
  "Appearance never writes runtime settings" rests on the panel owning no such
  input. The browser suite asserts the consequence; this asserts the cause.

Pattern follows ``tests/test_web_dialogs_static.py``: read the sources, assert
the structural fact, no browser needed.
"""

from __future__ import annotations

import ast
import pathlib
import re

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
WEB = REPO_ROOT / "web"

# Every first-party window must explicitly request persistent storage.
WEBVIEW_HOSTS = ("launcher.py", "ouroboros/launcher_onboarding.py")


def _read(rel: str) -> str:
    return (REPO_ROOT / rel).read_text(encoding="utf-8")


def _assert_persistent_webview_starts(source: str, rel: str) -> int:
    calls = [node for node in ast.walk(ast.parse(source))
             if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
             and isinstance(node.func.value, ast.Name)
             and node.func.value.id == "webview" and node.func.attr == "start"]
    for call in calls:
        assert any(keyword.arg == "private_mode" and isinstance(keyword.value, ast.Constant)
                   and keyword.value.value is False for keyword in call.keywords), (
            f"{rel}:{call.lineno}: webview.start() must explicitly pass private_mode=False; "
            "a private WebView discards the appearance choice when the window closes"
        )
    return len(calls)


def test_every_first_party_window_uses_a_persistent_profile():
    calls = sum(_assert_persistent_webview_starts(_read(rel), rel) for rel in WEBVIEW_HOSTS)
    # The count is asserted so that a NEW window added without the flag fails
    # here rather than silently using the library default.
    assert calls == 5, f"expected 5 first-party webview.start() calls, found {calls}"


def test_persistent_profile_guard_reads_multiline_calls_and_ignores_non_calls():
    source = '''
# webview.start()
example = "webview.start(private_mode=True)"
other.start(private_mode=True)
webview.start(
    debug=False,
    private_mode=False,
    menu=build_menu(),
)
'''
    assert _assert_persistent_webview_starts(source, "fixture.py") == 1


@pytest.mark.parametrize("argument", ["", "private_mode=True", "private_mode=0"])
def test_persistent_profile_guard_rejects_missing_or_non_false_argument(argument):
    with pytest.raises(AssertionError, match="must explicitly pass private_mode=False"):
        _assert_persistent_webview_starts(f"webview.start(\n    {argument}\n)", "fixture.py")


_MASK_DECLARATION = re.compile(r"^\s*(?:-webkit-)?mask(?:-image)?\s*:")
_SVG_INK = re.compile(r"(?:fill|stroke|stop-color|flood-color|color)\s*[=:]\s*['\"]?([^'\";)\s]+)", re.I)
# A mask only cuts the shape out of the element's own token paint (`background: var(--…)`); its geometry
# is drawn black, and any other ink would be a tone baked into the mask (a luminance mask reads it).
_GEOMETRY_INKS = {"none", "black", "%23000", "%23000000"}


def _baked_svg_lines(css: str) -> list[str]:
    """Inline SVG lines that fix a colour: neither the `--select-arrow` token nor black mask geometry."""
    return [line for line in css.splitlines() if "data:image/svg+xml" in line and "--select-arrow:" not in line
            and not (_MASK_DECLARATION.match(line) and {ink.lower() for ink in _SVG_INK.findall(line)} <= _GEOMETRY_INKS)]


def test_the_select_arrow_is_a_theme_token_not_a_baked_colour():
    ui = _read("web/ui.css")
    dark, light = ui.split(':root[data-theme="light"]', 1)
    assert "--select-arrow:" in dark and "--select-arrow:" in light
    # Two DIFFERENT inks: the pale arrow on a white field was the reported defect.
    dark_ink = re.search(r"--select-arrow:.*?stroke='(%23[0-9a-fA-F]{6})'", dark).group(1)
    light_ink = re.search(r"--select-arrow:.*?stroke='(%23[0-9a-fA-F]{6})'", light).group(1)
    assert dark_ink.lower() != light_ink.lower()
    for rel in ("web/ui.css", "web/settings.css", "web/style.css", "web/onboarding.css"):
        baked = _baked_svg_lines(_read(rel))
        assert not baked, f"{rel}: inline SVG carries a fixed colour: {baked}"


_SVG = "url(\"data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg'%3E%3Cpath d='M5 7.5 10 12.5' fill='none' stroke='{}'/%3E%3C/svg%3E\")"


@pytest.mark.parametrize(("line", "baked"), [
    (f"    --select-arrow: {_SVG.format('%23efeff2')};", False),
    (f"    mask-image: {_SVG.format('black')};", False),
    (f"    -webkit-mask-image: {_SVG.format('black')};", False),
    (f"    background-image: {_SVG.format('black')};", True),
    (f"    mask-image: {_SVG.format('black').replace('/%3E', ' style=%22fill:%23888%22/%3E')};", True),
])
def test_inline_svg_guard_separates_token_paint_and_mask_geometry_from_baked_colour(line, baked):
    assert bool(_baked_svg_lines(line)) is baked


def _rules(css: str) -> list[tuple[str, str]]:
    """``(selector, body)`` of every innermost rule, comments removed (an @media wrapper is not a selector)."""
    css = re.sub(r"/\*.*?\*/", "", css, flags=re.S)
    return [(" ".join(selector.split()), body) for selector, body in re.findall(r"([^{}]+)\{([^{}]*)\}", css)]
def test_status_text_uses_foreground_roles_not_raw_hues():
    # Raw hues remain valid for fills/borders, never small status text.
    for rel in ('web/style.css', 'web/settings.css'):
        assert not re.search(r'(?:^|[;{])\s*color:\s*var\(--(?:amber|red|green|blue)\)',
                             _read(rel), re.M), rel


def test_surface_sheets_carry_no_near_white_ink():
    """Light is only as complete as the last hard-coded near-white literal.

    `rgba(250, 250, 250, .1)` is a border on the Dark shell and nothing at all on the Light one;
    near-white text is simply unreadable there. Such ink goes through the palette channels
    (`rgba(var(--neutral-rgb), a)`) or a foreground role; only `web/ui.css`, the token sheet,
    may name a literal, because it declares both palettes side by side.
    """
    near_white = re.compile(r"rgba\(\s*(2[0-5]\d)\s*,\s*(2[0-5]\d)\s*,\s*(2[0-5]\d)\s*,")
    leftovers = [
        f"{sheet.relative_to(REPO_ROOT)}:{number}: {line.strip()}"
        for sheet in sorted(WEB.glob("*.css")) if sheet.name != "ui.css"
        for number, line in enumerate(sheet.read_text(encoding="utf-8").splitlines(), 1)
        if near_white.search(line)
    ]
    assert not leftovers, "near-white literals outside the token sheet:\n" + "\n".join(leftovers)


# ---- Screens before the interface (#1111): what paints before, or outside, the page that reads the choice ----

def _constant(rel: str, name: str) -> str:
    """A module-level string constant, read without importing its module (a skill's sidecar among them)."""
    for node in ast.parse(_read(rel)).body:
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") == name:
            return node.value.value
    raise LookupError(f"{rel}: {name}")


def _palette_values(css: str, selector: str) -> dict:
    """``--name: value`` of one rule as the stylesheet writes it, ``var(--neutral-rgb)`` and spacing normalised."""
    body = "".join(found for sel, found in _rules(css) if sel == selector)
    values = dict(re.findall(r"(--[\w-]+)\s*:\s*([^;]+)", body))
    neutral = values.get("--neutral-rgb", "")
    clean = lambda value: re.sub(r"(?<![\d])\.(\d)", r"0.\1", re.sub(r"\s+", "", value.replace("var(--neutral-rgb)", neutral)))  # noqa: E731
    return {name: clean(value) for name, value in values.items()}


def test_screens_before_the_interface_restate_exactly_the_palette():
    """The macOS window's first frame and the Telegram connecting page paint before ui.css can load, in code
    that cannot read it; they restate the palette's own room and ink, both themes."""
    from ouroboros.launcher_window_chrome import ROOM

    ui = _read("web/ui.css")
    dark, light = _palette_values(ui, ":root"), _palette_values(ui, ':root[data-theme="light"]')
    assert ROOM == {"light": light["--bg-primary"], "dark": dark["--bg-primary"]}
    page = _constant("skills/telegram/scripts/sidecar.py", "_BOOTSTRAP_HTML")
    css = re.search(r"<style>(.*?)</style>", page, re.S).group(1)
    for theme, selector, palette in (("dark", "html", dark), ("light", 'html[data-theme="light"]', light)):
        restated = _palette_values(css, selector)
        assert restated["--bg-primary"] == palette["--bg-primary"], theme
        assert restated["--text-secondary"] == palette["--text-secondary"], theme


def test_a_host_charts_default_series_are_the_approved_tokens_of_each_theme():
    """#474, owner 27.10: four series in one order for both themes, deeper in Light. theme_palette.js keeps
    the dark values as the colours of a config built without a document and of a page without the tokens."""
    ui = _read("web/ui.css")
    names = [f"--chart-series-{n}" for n in range(1, 5)]
    dark, light = _palette_values(ui, ":root"), _palette_values(ui, ':root[data-theme="light"]')
    assert [light[name] for name in names] == ["#c0284c", "#2c68d0", "#1e9c76", "#bd6a00"]
    assert [dark[name] for name in names] == ["#ee6780", "#60a5fa", "#3fdca3", "#fbbf24"]
    fallback = re.search(r"CHART_SERIES = Object\.freeze\(\[([^\]]*)\]\)", _read("web/modules/theme_palette.js"))
    assert re.findall(r"'(#[0-9a-f]{6})'", fallback.group(1)) == [dark[name] for name in names]


def test_windows_before_the_server_wear_the_interface_not_a_fixed_palette():
    """Every launcher window whose page is a string is built by ``_window_page`` (theme.js and ui.css inlined)
    and opens in the room colour of macOS (``paint_room``); no colour is written into those pages."""
    tree = ast.parse(_read("launcher.py"))
    pages = [keyword.value for node in ast.walk(tree) if isinstance(node, ast.Call)
             and getattr(node.func, "attr", "") == "create_window" for keyword in node.keywords if keyword.arg == "html"]
    assert len(pages) == 3, "already running, Git required, startup failed"
    assert all(isinstance(page, ast.Call) and getattr(page.func, "id", "") == "_window_page" for page in pages)
    painted = [node for node in ast.walk(tree) if isinstance(node, ast.Call) and getattr(node.func, "id", "") == "paint_room"]
    assert len(painted) == len(pages)
    source = _read("launcher.py")
    window_css = _constant("launcher.py", "_WINDOW_CSS")
    assert not re.search(r"#[0-9a-fA-F]{3,6}\b", window_css)
    assert "background:#0f0f11" not in source and "color:#efeff2" not in source
