"""The page's notices are held by one shared blocking-surface seam (docs/DESIGN.md §4, owner 26.11 «A»).

toast.js reads ui_helpers `modalSurfaces` and watches the attributes that make or conceal a blocker. Every module that
draws a backdrop or scrim over the page declares `aria-modal="true"` on the surface itself, so a page notice's × is
never left under a press that would cancel the question or close the sheet. The known portals are kept as scope
evidence that the scan still covers the surfaces the seam was chosen against.
"""
from __future__ import annotations

from pathlib import Path

MODULES = Path(__file__).resolve().parents[1] / "web" / "modules"
# What a module draws over the page. drawer_drag.js only names the backdrop, to leave a drag gesture alone.
OVERLAYS = ("ui-modal-backdrop", "model-sheet-scrim", "onboarding-overlay")
EXEMPT = {"drawer_drag.js"}
# The portals the seam was chosen against: the confirmation, the skills dialogs, Project create and folder, the MCP import,
# the peer introduction, the secret form, the phone's Model sheet and the setup overlay.
CROSS_CHECKED = ("confirm_dialog.js", "skill_dialogs.js", "project_create.js", "project_folder.js", "mcp_import_dialog.js",
                 "peer_intro.js", "secret_hitl.js", "model_sheet.js", "onboarding_overlay.js")


def _declares_modal(text: str) -> bool:
    return ('aria-modal="true"' in text or "setAttribute('aria-modal', 'true')" in text
            or 'setAttribute("aria-modal", "true")' in text)


def test_the_shared_seam_and_every_backdrop_or_scrim_declare_a_blocking_surface():
    toast = (MODULES / "toast.js").read_text(encoding="utf-8")
    assert "modalSurfaces" in toast and "'hidden'" in toast, "toast.js must watch the shared seam and hidden blockers"
    texts = {path.name: path.read_text(encoding="utf-8") for path in sorted(MODULES.glob("*.js"))}
    drawers = {name: _declares_modal(text) for name, text in texts.items()
               if name not in EXEMPT and any(overlay in text for overlay in OVERLAYS)}
    silent = [name for name, said in drawers.items() if not said]
    assert not silent, (f"{silent} draw a backdrop or scrim over the page without `aria-modal=\"true\"`: toast.js would "
                        "leave a page notice's × under it, where a press lands on the backdrop (ui_helpers `modalSurfaces`)")
    cross = {name: _declares_modal(texts[name]) for name in CROSS_CHECKED}
    assert all(cross.values()), ("the portals the seam was cross-checked against must still declare it", cross)
