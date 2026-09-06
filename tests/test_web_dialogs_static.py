"""Static class ban: no native browser dialogs in ``web/modules``.

pywebview's WKWebView implements no ``window.prompt`` (it answers ``null``
silently — a dead control on the macOS desktop app), and native
``confirm()``/``alert()`` render OS-modal chrome outside the design system.
The whole class is therefore banned in favor of the in-house
``openConfirmDialog`` (input mode replaces prompt, ``alert: true`` replaces
alert) — owner decision Б2-2, v6.90.3 full-class migration.

Pattern follows ``tests/test_mcp_ui_static.py``: read the sources and assert
the structural fact — no browser needed, so quick CI enforces the ban
automatically.
"""

from __future__ import annotations

import pathlib
import re


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
WEB_MODULES = REPO_ROOT / "web" / "modules"

# A real native-dialog CALL:
#   * ``window.prompt(`` / ``window . confirm(`` / ``window.alert(``, or
#   * bare ``prompt(`` / ``confirm(`` / ``alert(`` NOT preceded by an
#     identifier character or ``.``.
# Deliberate non-matches:
#   * ``openConfirmDialog(``      — the name continues past ``confirm``;
#   * ``promptProfileName(`` etc. — same;
#   * ``foo.confirm(``            — a method on our own object (``.`` excluded);
#   * ``alert: true``             — an option, not a call.
NATIVE_DIALOG_CALL = re.compile(
    r"window\s*\.\s*(?:prompt|confirm|alert)\s*\("
    r"|(?<![\w$.])(?:prompt|confirm|alert)\s*\("
)

def _code_lines(source: str) -> list[tuple[int, str]]:
    """Source lines in code position: full-line ``//``/``*`` comment lines are
    skipped, nothing else is stripped.

    Deliberately NO block-comment blanking: a ``/*...*/`` DOTALL sweep cannot
    tell a comment opener from the same two characters inside a string, and
    ``accept="*/*"`` in chat.js opened a pseudo comment that blanked 403 lines
    of real code — a hole in the only automated enforcement of this class.
    Trailing ``//`` comments are likewise not stripped (a ``//`` inside a URL
    string would truncate real code). Both choices over-approximate in the
    safe direction: a call smuggled into a comment is reported, which for a
    class ban is a nuisance, while a blind region is a defect."""
    lines: list[tuple[int, str]] = []
    for lineno, line in enumerate(source.splitlines(), 1):
        stripped = line.strip()
        if stripped.startswith("//") or stripped.startswith("*"):
            continue
        lines.append((lineno, line))
    return lines


def test_no_native_dialog_calls_in_web_modules() -> None:
    sources = sorted(WEB_MODULES.glob("**/*.js"))
    assert sources, f"no JS modules found under {WEB_MODULES}"
    violations: list[str] = []
    for path in sources:
        for lineno, line in _code_lines(path.read_text(encoding="utf-8")):
            if NATIVE_DIALOG_CALL.search(line):
                violations.append(
                    f"{path.relative_to(REPO_ROOT)}:{lineno}: {line.strip()}"
                )
    assert not violations, (
        "Native browser dialogs are banned in web/modules: window.prompt is a "
        "silent no-op under pywebview (macOS desktop), and confirm()/alert() "
        "bypass the design system. Use openConfirmDialog from "
        "web/modules/confirm_dialog.js (input mode for prompt, alert:true for "
        "alert).\n" + "\n".join(violations)
    )


def test_confirm_dialog_offers_the_alert_mode() -> None:
    """The replacement the ban points at must actually exist: the in-house
    dialog exposes the alert option the migrated alert() sites rely on."""
    source = (WEB_MODULES / "confirm_dialog.js").read_text(encoding="utf-8")
    assert "alert = false" in source
    assert "openConfirmDialog" in source


def test_confirm_dialog_backdrop_stacks_above_every_overlay() -> None:
    """Dialogs must render OVER the reconnect shade — not under it, visible
    through the tint but unclickable (issue #146). Static pin of the layer
    order; the cross-element stack is decided by z-index alone because both
    are body-level siblings. The dialog backdrop also carries
    .marketplace-modal-backdrop, whose z-index 90 reaches it through a
    :where() selector of specificity 0, so .confirm-dialog-backdrop wins that
    inherited default on specificity and not merely on source order. (The
    former .update-dialog-overlay was retired by the Updates-tab redesign; the
    reconnect shade is the one full-screen overlay left at the 1000 layer.)"""
    css = (REPO_ROOT / "web" / "style.css").read_text(encoding="utf-8")

    def effective_z(selector_re: str) -> int:
        # Cascade-honest: for equal-specificity selector blocks the LAST
        # declaration wins, so the pin reads the final value, not the max.
        found = []
        for block in re.finditer(selector_re + r"[^{]*{[^}]*}", css):
            m = re.search(r"z-index:\s*(\d+)", block.group(0))
            if m:
                found.append(int(m.group(1)))
        assert found, f"no z-index found for {selector_re}"
        return found[-1]

    confirm = effective_z(r"\.confirm-dialog-backdrop")
    assert confirm > effective_z(r"#reconnect-overlay")
