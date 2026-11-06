"""SM1 palette delivery through both production documents in both themes; no model or runtime process.

Reuse the subscription UI fixture's real static server, templates, bootstrap and mock APIs.
The restart/commit are wiring inputs here, not a claimed paid self-modification result.

The stand's probe opens in Playwright's Light and this lane's pages in Dark (``tests/conftest.py``), so the
oracle is run from both: the old script's edit, which changed only the dark seed, passed here and failed the
real rehearsal (SM1_a1 of 2026-11-03: light ``--accent`` computed ``#c93545``).
"""
from __future__ import annotations

import colorsys
import json
import os
import pathlib
import re
import types

import pytest

from devtools.e2e_live import scenarios
from devtools.e2e_live.ui_probe import UIProbe
from tests import test_subscription_setup_browser as setup_browser

subscription_ui = setup_browser.subscription_ui
pytestmark = [pytest.mark.ui_browser]

TARGET = scenarios.SM1_NEW_ACCENT
DARK_ROOT, LIGHT_ROOT = ":root {", ':root[data-theme="light"] {'
ACCENT_ONLY = {"--accent": TARGET}
WHITE = "rgb(255, 255, 255)"
# What real controls wear in every theme, through the tokens a new accent rewrites (DESIGN "The tone primitive").
CONTROL_ROLES = {"primary action": "--tone-crimson", "primary ink": "--tone-crimson-ink", "current step": "--tone-crimson-soft",
                 "current step index": "--tone-crimson-soft-ink", "chosen place": "--tone-crimson", "chosen ink": "--tone-crimson-ink"}
CONTROL_FILLS = ("primary action", "current step", "chosen place")


def _seed() -> str:
    """The shared sheet as an accent change starts from it. A candidate where SM1 already landed is put back on a red,
    so this file proves the same thing when that candidate's own qualification runs it."""
    css = (setup_browser.WEB / "ui.css").read_text(encoding="utf-8")
    return css if scenarios.accent_value(css) != TARGET else scenarios.css_with_accent(css, "#c93545")


def _with_roles(css: str, roles: dict[str, str], opening: str = DARK_ROOT) -> str:
    """The sheet with these roles of one theme's root block set by name, whatever their values are."""
    start = css.index(opening)
    end = css.index("\n}", start)
    block = css[start:end]
    for name, value in roles.items():
        block, found = re.subn(rf"(?m)^(\s*{re.escape(name)}\s*:\s*)[^;]+;", rf"\g<1>{value};", block, count=1)
        assert found, f"{opening} has no {name}"
    return css[:start] + block + css[end:]


def _landed(fault: str, seed: str) -> str:
    """The sheet a landing served. ``dark-only`` is byte for byte what the old script wrote in the real rehearsal."""
    if fault == "dark-only":
        return _with_roles(seed, ACCENT_ONLY)
    if fault == "light-only":
        return _with_roles(seed, ACCENT_ONLY, LIGHT_ROOT)
    if fault == "roles-left":
        return _with_roles(_with_roles(seed, ACCENT_ONLY), ACCENT_ONLY, LIGHT_ROOT)
    return scenarios.css_with_accent(seed, TARGET)


def _on_accent_hue(paint: str) -> bool:
    """A computed ``rgb()``/``rgba()`` paint on the accent's hue, whatever shade and alpha its theme gives it."""
    channels = [float(value) / 255 for value in re.findall(r"[\d.]+", paint)[:3]]
    hue = colorsys.rgb_to_hls(*channels)[0]
    target = colorsys.rgb_to_hls(*(channel / 255 for channel in bytes.fromhex(TARGET[1:])))[0]
    return abs(hue - target) < 0.01


def _computed_paint(token: str) -> str:
    """A declared color as ``getComputedStyle`` serializes it."""
    if token.startswith("#"):
        return "rgb({}, {}, {})".format(*bytes.fromhex(token[1:]))
    red, green, blue, alpha = re.findall(r"[\d.]+", token)
    return f"rgba({red}, {green}, {blue}, {float(alpha):g})"


def _real_controls(probe: UIProbe, page, theme: str) -> dict[str, str]:
    """What a person sees of the accent on real controls, in one theme of both documents."""
    probe.color_scheme(theme)
    probe.goto("/onboarding", ready_selector=".wizard-shell")
    paints = {"primary action": probe.computed_property("#next-btn", "background-color"),
              "primary ink": probe.computed_property("#next-btn", "color"),
              "current step": probe.computed_property(".wizard-step.active", "background-color"),
              "current step index": probe.computed_property(".wizard-step.active .wizard-step-index", "color")}
    probe.goto("/")
    # The chosen row takes its tone through a transition: read the color it settles on.
    page.wait_for_function("() => document.querySelector('.nav-row-main.active')?.getAnimations().length === 0")
    return {**paints, "chosen place": probe.computed_property(".nav-row-main.active", "background-color"),
            "chosen ink": probe.computed_property(".nav-row-main.active", "color")}


# A theme's own landing runs from both openings. Each other fault opens where reading one theme would have missed it.
@pytest.mark.parametrize("fault,opens_in", [
    ("none", "light"), ("none", "dark"), ("dark-only", "light"), ("dark-only", "dark"), ("light-only", "light"), ("light-only", "dark"),
    ("roles-left", "dark"), ("wizard-missing-link", "light"), ("wizard-focus-dark", "light"), ("wizard-focus-light", "dark")])
def test_every_theme_of_both_documents_decides_whatever_the_browser_opens_in(subscription_ui, tmp_path, fault, opens_in):
    ui = subscription_ui
    page = ui["new_page"](color_scheme=opens_in)
    ui["errors"].watch(page)   # the oracle navigates this page: its own cut-off reads are not failures
    seed = _seed()
    shared = _landed(fault, seed)
    tokens, seed_tokens = scenarios.css_theme_tokens(shared), scenarios.css_theme_tokens(seed)
    page.route("**/static/ui.css", lambda route: route.fulfill(content_type="text/css", body=shared))
    if fault == "wizard-missing-link":
        html = (setup_browser.WEB / "onboarding_template.html").read_text(encoding="utf-8")
        link = '<link rel="stylesheet" href="/onboarding/static/ui.css">'
        assert html.count(link) == 1
        bootstrap = (setup_browser.WEB / "tests/fixtures/onboarding_bootstrap.json").read_text(encoding="utf-8")
        html = html.replace(link, "").replace("__ONBOARDING_BOOTSTRAP__", bootstrap)
        page.route("**/onboarding", lambda route: route.fulfill(content_type="text/html", body=html))
    elif fault.startswith("wizard-focus-"):
        # A page-local shadow of one shared role: specificity keeps each to its own theme, so reading one theme misses it.
        opening = DARK_ROOT if fault.endswith("dark") else LIGHT_ROOT
        local = (setup_browser.WEB / "onboarding.css").read_text(encoding="utf-8") + f"\n{opening} --focus-accent-border: #00ff00; }}"
        page.route("**/static/onboarding.css", lambda route: route.fulfill(content_type="text/css", body=local))
    probe = UIProbe(ui["url"])
    probe.page = page  # the existing fixture owns this browser's launch and cleanup
    server = types.SimpleNamespace(base_url=ui["url"])
    shots = pathlib.Path(os.environ.get("OUROBOROS_UI_EVIDENCE_DIR") or tmp_path) / f"sm1-palette-{fault}-opens-{opens_in}"
    ctx = scenarios.LaneContext(server=server, clone=tmp_path, data_root=tmp_path, oracle=None, harness=None,
                                ui_resolver=lambda _: (probe, ""), ui_reason="", shots=shots,
                                log=lambda _: None, task_timeout=1, restart=lambda: server)
    ctx.check("commit_landed", True)  # synthetic wiring input, never model/review evidence
    names = scenarios.SM1_REQUIRED_PALETTE | scenarios.sm1_palette_tokens(seed).keys()
    scenarios.check_sm1_rendered_palette(ctx, names, served=True, seed_css=seed)

    verdict = json.dumps({key: ctx.facts[key] for key in ("palette_problems", "palette_computed")}, indent=1)
    wrong_sheet = fault in {"dark-only", "light-only", "roles-left"}
    assert ctx.checks["ui_app_palette"] is (not wrong_sheet), verdict
    assert ctx.checks["ui_onboarding_palette"] is (not wrong_sheet and fault != "wizard-missing-link"), verdict
    assert ctx.checks["ui_computed_style"] is (fault == "none"), verdict
    assert not ctx.ui_reason and len(ctx.screenshots) == 4
    computed, problems = ctx.facts["palette_computed"], ctx.facts["palette_problems"]
    if wrong_sheet:
        for document in ("app", "onboarding"):
            for theme in scenarios.SM1_THEMES:
                accent_wrong = {"dark-only": "light", "light-only": "dark"}.get(fault) == theme
                assert (f"--accent is {seed_tokens[theme]['--accent']}" in problems[document][theme]) is accent_wrong, verdict
                assert "--accent-10 keeps its seed value" in problems[document][theme], verdict
    if fault == "dark-only":   # the real rehearsal's own reading, now a verdict on the theme that decided it
        assert computed["app"]["light"]["--accent"] == "#c93545" == computed["onboarding"]["light"]["--accent"]
        assert computed["app"]["dark"]["--accent"] == TARGET == computed["onboarding"]["dark"]["--accent"]
    if fault.startswith("wizard-focus-"):
        shadowed = "dark" if fault.endswith("dark") else "light"
        for theme in scenarios.SM1_THEMES:
            assert (computed["onboarding"][theme]["--focus-accent-border"] == "#00ff00") is (theme == shadowed), verdict
            assert (computed["app"][theme] == computed["onboarding"][theme]) is (theme != shadowed), verdict
    if fault == "none":
        for document in ("app", "onboarding"):
            for theme in scenarios.SM1_THEMES:
                assert computed[document][theme] == {name: tokens[theme][name] for name in sorted(names)}, "the source's reading is the browser's"
        for theme in scenarios.SM1_THEMES:
            paints = _real_controls(probe, page, theme)
            assert page.evaluate("() => document.documentElement.dataset.theme") == theme
            assert paints == {name: _computed_paint(tokens[theme][role]) for name, role in CONTROL_ROLES.items()}, (theme, paints)
            assert all(_on_accent_hue(paint) or paint == WHITE for paint in paints.values()), (theme, paints)
            assert all(_on_accent_hue(paints[name]) for name in CONTROL_FILLS), (theme, paints)
    if fault == "roles-left":   # the accent token alone never reached the controls: reading the token would not have shown it
        for theme in scenarios.SM1_THEMES:
            paints = _real_controls(probe, page, theme)
            assert not any(_on_accent_hue(paints[name]) for name in CONTROL_FILLS), (theme, paints)
    ctx.close_ui()
