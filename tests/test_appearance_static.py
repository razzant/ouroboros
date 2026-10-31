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
    (f"    background-image: {_SVG.format('%23888')};", True),
    (f"    -webkit-mask-image: {_SVG.format('%23888')};", True),
    (f"    mask-image: {_SVG.format('black').replace('/%3E', ' style=%22fill:%23888%22/%3E')};", True),
])
def test_inline_svg_guard_separates_token_paint_and_mask_geometry_from_baked_colour(line, baked):
    assert bool(_baked_svg_lines(line)) is baked


def _rules(css: str) -> list[tuple[str, str]]:
    """``(selector, body)`` of every innermost rule, comments removed (an @media wrapper is not a selector)."""
    css = re.sub(r"/\*.*?\*/", "", css, flags=re.S)
    return [(" ".join(selector.split()), body) for selector, body in re.findall(r"([^{}]+)\{([^{}]*)\}", css)]


THEME_BLOCKS = (":root", ':root[data-theme="light"]')
DARK_VOICE = ':root:not([data-theme="light"])'
# The roles the dark palette sets apart from the light theme's: each has its own light value, equal to the light
# theme's before the dark voices were added (docs/DESIGN.md seed table). `--green` among them: the light theme read
# the dark seed's green until the dark one changed.
LIGHT_RESTATED = (
    "--voice-owner", "--voice-ouroboros", "--voice-ouroboros-card", "--system-sender", "--amber",
    "--green", "--status-ok-fg", "--status-warn-fg", "--status-error-fg", "--accent-light", "--focus-accent-border",
    "--assistant-sender", "--panic-ink", "--tone-chosen-answer", "--tone-chosen-answer-ink",
    "--tone-chosen-answer-detail", "--tone-chosen-answer-shadow", "--chosen-answer-cue-display", "--tone-recommended",
    "--tone-recorded-answer", "--tone-recorded-mark", "--quiz-held-opacity", "--message-pending-opacity",
)


def test_the_dark_voice_materials_stay_out_of_the_light_theme():
    """A voice material (ui.css, after the light block) restates palette roles on a voice's own objects. Only the dark
    theme has one. Code blocks deliberately reset their quiet hover to the room's neutral material in both themes;
    every other role override stays in its theme block."""
    rules = _rules(_read("web/ui.css"))
    declared = lambda body: set(re.findall(r"(--[\w-]+)\s*:", body))  # noqa: E731
    materials = {selector: declared(body) for selector, body in rules if selector.startswith(DARK_VOICE)}
    assert set(materials) == {f"{DARK_VOICE} :is(.chat-bubble.assistant, .chat-live-card)", f"{DARK_VOICE} .chat-bubble.user"}, materials
    voice_roles = set().union(*materials.values())
    neutral_code = [body.strip() for selector, body in rules if selector == ".md-code-block"]
    assert neutral_code == ["--tone-quiet-hover: var(--bg-hover);"], neutral_code
    leaks = [(selector, sorted(declared(body) & voice_roles)) for selector, body in rules
             if selector not in (*THEME_BLOCKS, ".md-code-block")
             and not selector.startswith(DARK_VOICE) and declared(body) & voice_roles]
    assert not leaks, f"voice material roles restated outside the dark voice materials: {leaks}"
    light = "".join(body for selector, body in rules if selector == ':root[data-theme="light"]')
    missing = [name for name in LIGHT_RESTATED if not re.search(rf"(?<![\w-]){re.escape(name)}\s*:", light)]
    assert not missing, f"these roles differ in the dark seed and need their own light value: {missing}"


def test_every_colour_the_design_seed_table_names_is_a_value_of_the_palette():
    """The seed table (docs/DESIGN.md) and web/ui.css say the same colours: a value changed on one side only fails."""
    design = _read("docs/DESIGN.md")
    table = design.split("**Seed (light | dark)**", 1)[1].split("\n\n", 2)[1]
    named = {value.lower() for value in re.findall(r"#[0-9a-fA-F]{6}\b", table)}
    ui = {value.lower() for value in re.findall(r"#[0-9a-fA-F]{6}\b", re.sub(r"/\*.*?\*/", "", _read("web/ui.css"), flags=re.S))}
    assert len(named) > 15, "the seed table was not found"
    assert not named - ui, f"the seed table names colours web/ui.css does not have: {sorted(named - ui)}"


def test_selected_advisory_reads_like_every_other_setting_value():
    """A setting's value is never amber: the selected Advisory takes the shared selected look of a chosen option,
    the themed soft crimson pair, so the old ~2:1 amber wash on the light surface cannot come back."""
    css = _read("web/settings.css")
    assert not re.search(r'advisory"\][^{]*\{', css), "Advisory has a look of its own again"
    selected = re.search(r':root\[data-theme\] \.settings-effort-btn\.active,?[^{]*\{([^}]*)\}', css)
    assert selected, "the shared selected-value rule is gone"
    assert "var(--tone-crimson-soft)" in selected.group(1) and "var(--tone-crimson-soft-ink)" in selected.group(1)
    assert ".wizard-choice.advisory" not in _read("web/onboarding.css")


def test_appearance_is_a_named_destination_that_never_reaches_the_server():
    ui = _read("web/modules/settings_ui.js")
    assert re.search(r"value:\s*'appearance'", ui), "Appearance needs its own tab"
    panel = re.search(
        r'data-settings-panel="appearance"(.*?)</section>', ui, re.S).group(1)
    assert "data-theme-control" in panel
    # An id starting with s- would be swept into the /api/settings payload.
    assert not re.search(r'\bid="s-[\w-]*"', panel.replace('id="s-appearance-theme-label"', '')), \
        "a settings-collected input would POST the client-local appearance choice"
    # Appearance now owns a second client-local block (notifications), so the
    # guard is stated as its own intent instead of "no control at all": every
    # control here must lack the ONE handle the collector reads.
    for control in re.finditer(r"<(?:input|select)\b[^>]*>", panel):
        assert not re.search(r'\bid="s-', control.group(0)), (
            f"{control.group(0)} would be swept into the /api/settings payload"
        )
    collector = _read("web/modules/settings.js")
    assert 'input[id^="s-"]' in collector, "the collector moved; re-check this guard"


def test_the_onboarding_wizard_is_no_longer_pinned_to_the_viewport():
    # body{min-height:100vh;padding:16px} plus a shell pinned to
    # calc(100vh - 32px) exactly fills the window, so ANY control above #root
    # overflowed by its own height.
    css = _read("web/onboarding.css")
    assert "calc(100vh - 32px)" not in css
    assert ".onboarding-appearance" in css
    template = _read("web/onboarding_template.html")
    assert "data-theme-control" in template
    assert "data-theme-toggle" not in template


def test_status_text_uses_foreground_roles_not_raw_hues():
    # Raw hues remain valid for fills/borders, never small status text.
    for rel in ('web/style.css', 'web/settings.css'):
        assert not re.search(r'(?:^|[;{])\s*color:\s*var\(--(?:amber|red|green|blue)\)',
                             _read(rel), re.M), rel


def test_chat_palette_redraw_is_local_viewport_work():
    chat = _read('web/modules/chat.js')
    mount = chat.split('function enhanceMountedMarkdown(root)', 1)[1].split('const {', 1)[0]
    assert 'onThemeDomWrite: foldedWrite(withStableViewport)' in mount
    assert 'const foldedWrite = writer => (mutate, options) => writer(() => {' in mount
    assert mount.index('const changed = mutate();') < mount.index('refreshProjectAnswerFold(') < mount.index('return changed;')
    assert 'refreshProjectAnswerFold(root.closest?.(\'.project-answer\'))' in mount


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


def test_an_old_desktop_app_is_named_where_its_choices_are_lost():
    """In-app updates replace the core, never the frozen launcher that decides WebView storage, so
    the Appearance panel carries the line ``desktop_shell.js`` fills for an app that erases it."""
    panel = re.search(
        r'data-settings-panel="appearance"(.*?)</section>', _read("web/modules/settings_ui.js"), re.S).group(1)
    assert "data-shell-storage-status" in panel
    settings = _read("web/modules/settings.js")
    assert "mountDesktopShell(page" in settings and "disposeShell()" in settings
    shell = _read("web/modules/desktop_shell.js")
    assert "PERSISTENT_STORAGE_SINCE = Object.freeze([7, 2, 0])" in shell, (
        "7.2.0 is the first release whose launcher passes private_mode=False (git tag --contains e30957143)")


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


def test_the_telegram_connecting_page_reads_the_one_saved_choice_before_it_paints():
    """No second store: the same key theme.js writes. The script runs in <head> after Telegram's (whose
    colorScheme is the System source there), never deferred, so the first paint already has the palette;
    the page's CSP still admits no inline script."""
    key = re.search(r"const KEY = '([^']+)';", _read("web/theme.js")).group(1)
    script = _constant("skills/telegram/scripts/sidecar.py", "_BOOTSTRAP_JS")
    assert f"localStorage.getItem('{key}')" in script
    assert script.index("colorScheme") < script.index("prefers-color-scheme")  # Telegram's before the device's
    assert "DOMContentLoaded" in script  # the launch waits for the document it reports into
    page = _constant("skills/telegram/scripts/sidecar.py", "_BOOTSTRAP_HTML")
    head = page.split("</head>", 1)[0]
    sdk = head.index('<script src="https://telegram.org/js/telegram-web-app.js"></script>')
    own = head.index('<script src="/__telegram/bootstrap.js"></script>')
    assert sdk < own and "defer" not in head and "async" not in head
    assert re.findall(r"<script[^>]*>([^<]*)</script>", page) == ["", ""], "an inline script would break its CSP"


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
