"""Structural guards for notifications: the facts a browser cannot show.

Three claims here are invisible to a Playwright run:

* **Replay safety is structural.** Nothing about past notifications is stored,
  so "a reload does not re-notify" rests entirely on the notifier being reached
  from live ``onWs`` handlers and from nowhere else. A browser test can show
  that one reload was quiet; only the source can show WHY every reload is.
* **The preferences never reach the server.** ``web/modules/settings.js``
  collects ``s-``-prefixed fields into the ``/api/settings`` payload, so
  "notification choices stay client-local" rests on the Appearance block owning
  no such field. The browser suite asserts the consequence; this asserts the
  cause.
* **One canonical policy.** DESIGN §9 is the single owner-facing statement and
  DEVELOPMENT points at it instead of restating it — a documentation invariant,
  not a rendered behaviour.

Pattern follows ``tests/test_appearance_static.py``: read the sources, assert
the structural fact, no browser needed.
"""

from __future__ import annotations

import pathlib
import re

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]


def _read(rel: str) -> str:
    return (REPO_ROOT / rel).read_text(encoding="utf-8")


def _appearance_panel(source: str) -> str:
    start = source.index('data-settings-panel="appearance"')
    end = source.index("</section>", start)
    return source[start:end]


def test_notification_controls_are_client_local():
    panel = _appearance_panel(_read("web/modules/settings_ui.js"))
    assert "data-notify-settings" in panel, "the notification block lives under Appearance"
    block = panel[panel.index("data-notify-settings"):]
    prefs = set(re.findall(r'data-notify-pref="([a-z_]+)"', block))
    assert prefs == {
        "enabled", "needs_answer", "task_done", "important", "notice", "main_reply", "sound", "show_text",
    }, prefs
    assert "data-notify-test" in block, "a test notification is the honest way to see the surface"
    # The cause of "never posted to the server": no settings field at all.
    for attribute in ('id="s-', 'name="s-'):
        assert attribute not in block, (
            f"{attribute} inside the notification block would enter the /api/settings payload"
        )


def test_settings_mounts_the_notifier_like_the_theme():
    source = _read("web/modules/settings.js")
    assert "getNotifier().mountSettings(page)" in source
    # A client-local control must not mark the SERVER draft dirty, or leaving
    # Settings would ask to discard changes that were never going to be posted.
    assert "data-notify-settings" in source
    assert "page.addEventListener('change', onServerSettingEdited)" in source
    # Client-local surfaces are mounted after the panel is injected, next to the
    # appearance choice they share their storage discipline with.
    assert source.index("ouroTheme?.mount()") < source.index("mountSettings(page)")


def test_the_subscription_outlives_every_room():
    """The defect this guard exists for: a chat instance dies with its room.

    Closing a Project panel calls destroyProjectInstance -> inst.destroy(),
    which disposes that instance's ws subscriptions. A notifier wired inside a
    chat instance is therefore silent in exactly the case notifications are for
    — the owner left, the room is closed. So the subscription must be taken once
    at client level, on the shared socket, and chat.js must not be the wiring
    point at all.
    """
    app = _read("web/app.js")
    assert "getNotifier().attach({" in app
    attach = app[app.index("getNotifier().attach({"):]
    attach = attach[:attach.index("});") + 3]
    assert "ws," in attach, "the shared socket, not a room, carries the subscription"
    assert "ownerVisibleChat" in attach, "machine traffic must not notify"
    assert "projectChatIds" in attach, "a closed Project's frames must still be eligible"

    chat = _read("web/modules/chat.js")
    assert "notifications.js" not in chat, (
        "chat.js must not wire notifications: its handlers are per room and die with it"
    )
    assert "getNotifier" not in chat

    # Only live socket frames reach the notifier; the module takes no history
    # reader, so replay safety is structural rather than a stored ledger.
    module = _read("web/modules/notifications.js")
    for event in ("'chat'", "'quiz'", "'log'"):
        assert f"on({event}" in module, f"the {event} frame is a live source"
    assert "applyHistoryMessages" not in module and "chat/history" not in module


def test_policy_has_one_canonical_home():
    design = _read("docs/DESIGN.md")
    assert "## 9. Notifications" in design
    assert "ONE canonical statement" in design
    development = _read("docs/development/03-module-size-and-complexity.md")
    assert "notifications ring for live events only" in development
    assert "`docs/DESIGN.md` §9" in development, "DEVELOPMENT points at the policy, never restates it"
    architecture = _read("docs/architecture/03-web-ui-pages-and-buttons.md")
    assert "web/modules/notifications.js" in architecture


def test_importance_uses_the_existing_discriminator():
    source = _read("web/modules/notifications.js")
    assert "proactive_message" in source
    # No second model call and no new host field: the whole decision is local.
    for forbidden in ("fetch(", "apiFetch", "apiClient"):
        assert forbidden not in source, f"{forbidden} would make a notification decision remote"
