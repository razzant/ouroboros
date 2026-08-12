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
    # `web/app.js` is the LARGEST client file and it is not under `web/modules`, so
    # it sat outside the only automated enforcement this class ban has. Nothing in
    # it violates the ban today; the hole is the finding, not a breach (I15).
    sources = sorted(WEB_MODULES.glob("**/*.js")) + [REPO_ROOT / "web" / "app.js"]
    assert sources, f"no JS modules found under {WEB_MODULES}"
    assert (REPO_ROOT / "web" / "app.js").is_file(), "web/app.js moved; the scan lost its largest file"
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
