"""First-run onboarding as the desktop launcher presents it.

Extracted from ``launcher.py`` (which stays the process/window orchestrator)
because the wizard is no longer a detached pre-server document: the gateway
starts first and this module only opens a window onto its ``/onboarding`` page.
"""

from __future__ import annotations

import logging

from ouroboros.config import (
    apply_settings_to_env as _apply_settings_to_env,
    load_settings,
)
from ouroboros.server_runtime import apply_runtime_provider_defaults, has_startup_ready_provider

# The launcher's own logger: these lines belong in launcher.log next to the
# startup sequence they are part of.
log = logging.getLogger("launcher")


def prepare_first_run_settings() -> tuple[dict, bool]:
    """Normalize provider defaults; answer whether first-run onboarding is due.

    Runs BEFORE the managed server starts, because the answer decides what the
    launcher shows once it is healthy. The onboarding SURFACE is not rendered
    here: the live server serves it (``present_first_run_onboarding``).
    """
    settings, _provider_defaults_changed, _provider_default_keys = apply_runtime_provider_defaults(load_settings())
    # The normalization is APPLIED, not persisted. Startup is a read, and a read that
    # rewrites the file it read is how a normalization becomes an owner decision: the
    # fresh-install case already had to be carved out of this save (it would create
    # settings.json before the owner's own onboarding write and lose safety-light
    # authorship and the install-time agent presets), which is the same objection in a
    # narrower dress. Nothing is dropped, because nothing here was the only place the
    # normalization happens: every reader re-derives it (`/api/settings`, `/onboarding`,
    # the onboarding host, the plan-review script), and the completion save persists it.
    _apply_settings_to_env(settings)
    return settings, not has_startup_ready_provider(settings)


def present_first_run_onboarding(settings: dict, port: int, *, headless: bool = False) -> dict:
    """Show first-run onboarding served by the ALREADY-RUNNING managed gateway.

    D-8: one wizard on every host. The setup window loads the same live
    ``/onboarding`` page a browser owner sees and can reach ``/api/*`` while the
    owner is still setting up — which is what makes connecting an agent
    subscription during first-run possible at all. A gateway without a
    supervisor is a supported runtime state (ARCHITECTURE §2), so nothing new
    is started to make this work.

    Returns ``{"saved": bool, "restart_required": bool}`` — reported BY the page
    through the lifecycle bridge, not written here: this module persists
    nothing. Closing the window without saving stays non-fatal: startup
    continues and the main window's blocking overlay still offers the same
    wizard.

    ``settings`` is the launcher's already-normalized snapshot. It is no longer
    read (the completion endpoint reloads and validates for itself) and is kept
    only so the launcher's call site stays untouched.
    """
    outcome = {"saved": False, "restart_required": False}

    if headless:
        # Browser mode (#56): no GUI backend for a setup window, so the
        # ESTABLISHED web-onboarding flow (/api/onboarding probe + blocking
        # overlay + hot supervisor start after save) is the first-run surface —
        # the same path Docker/browser installs already use.
        log.info(
            "First-run setup window skipped: no GUI backend; onboarding is "
            "served in the browser."
        )
        print(
            "No GUI backend (GTK/QT) for the setup window; onboarding is "
            "served in the browser at the URL below.",
            flush=True,
        )
        return outcome

    import webview

    class OnboardingHostApi:
        """Window-lifecycle bridge for the desktop setup window.

        NOT a settings authority, and no longer even capable of being one. The
        page completes through ``POST /api/onboarding/complete`` exactly as a
        browser owner does, and that endpoint authors the fresh-install
        ``light`` safety coverage on its OWN server-side freshness proof — the
        one reason a desktop-only save path ever existed. A bridge method that
        can write ``settings.json`` while nothing calls it is not dead code but
        a live authority nobody audits, so it is gone: window lifecycle only.
        """

        def onboarding_finished(self, result: dict | None = None) -> str:
            payload = result if isinstance(result, dict) else {}
            # Absent/malformed payload means the wizard told us NOTHING, and on
            # the flag that says "the owner's settings were saved" the honest
            # default is no (BIBLE P1). Unreachable today — the wizard only
            # calls this with the completion envelope — but the default must
            # not be the one that invents a save.
            if payload.get("ok", False):
                outcome["saved"] = True
            if payload.get("restart_required"):
                outcome["restart_required"] = True
            for window in webview.windows:
                window.destroy()
            return "ok"

    webview.create_window(
        "Ouroboros — Setup",
        url=f"http://127.0.0.1:{port}/onboarding",
        js_api=OnboardingHostApi(),
        width=980,
        height=780,
        min_size=(840, 640),
    )
    webview.start()
    return outcome


__all__ = ["prepare_first_run_settings", "present_first_run_onboarding"]
