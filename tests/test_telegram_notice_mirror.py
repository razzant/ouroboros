"""The Telegram skill's mirror of host owner notifications (``owner.notification``).

Opt-in like its two periodic lanes: no pinned chat or ``TELEGRAM_NOTIFY_NOTICES``
off means silence; on, one plain line names the source and carries the sentence.
"""

from __future__ import annotations

import asyncio
import importlib.util
import pathlib
import sys
import types

REPO = pathlib.Path(__file__).resolve().parents[1]


def _load_notifier():
    """Import ``skills/telegram/lib/telegram_notifier.py`` as its package module."""
    pkg_root = REPO / "skills" / "telegram"
    if "skills" not in sys.modules:
        skills_pkg = types.ModuleType("skills")
        skills_pkg.__path__ = [str(REPO / "skills")]
        sys.modules["skills"] = skills_pkg
    for name, path in (("skills.telegram", pkg_root), ("skills.telegram.lib", pkg_root / "lib")):
        if name not in sys.modules:
            module = types.ModuleType(name)
            module.__path__ = [str(path)]
            sys.modules[name] = module
    spec = importlib.util.spec_from_file_location(
        "skills.telegram.lib.telegram_notifier", pkg_root / "lib" / "telegram_notifier.py",
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class _Api:
    def __init__(self, settings):
        self.settings = settings
        self.logged = []

    def get_settings(self, keys):
        return {"TELEGRAM_BOT_TOKEN": "t"}

    def log(self, level, message):
        self.logged.append((level, message))


def _run(handler, event):
    asyncio.run(handler(event))


def test_notice_is_mirrored_only_with_a_pinned_chat_and_the_toggle_on(monkeypatch):
    notifier = _load_notifier()
    sent = []

    async def fake_push(api, chat_id, text, *, trust_env=False):
        sent.append((chat_id, text))
        return "sent", None

    monkeypatch.setattr(notifier, "_push_notification", fake_push)
    event = {"topic": "owner.notification", "text": "Meeting in 15 min", "source": "skill:calendar"}

    off = _Api({"TELEGRAM_CHAT_ID": "42", "TELEGRAM_NOTIFY_NOTICES": "off"})
    monkeypatch.setattr(notifier, "_load_settings", lambda api: api.settings)
    _run(notifier._make_notice(off), event)
    assert sent == [], "off by default: a pinned chat alone is not consent"

    unpinned = _Api({"TELEGRAM_NOTIFY_NOTICES": "on"})
    _run(notifier._make_notice(unpinned), event)
    assert sent == [], "no pinned chat, no delivery — never a transport-origin fallback"

    on = _Api({"TELEGRAM_CHAT_ID": "42", "TELEGRAM_NOTIFY_NOTICES": "on"})
    _run(notifier._make_notice(on), event)
    assert sent == [(42, "🔔 calendar: Meeting in 15 min")]
    _run(notifier._make_notice(on), {**event, "source": "task_followup", "text": "Call mother"})
    assert sent[-1] == (42, "🔔 Ouroboros: Call mother")
    _run(notifier._make_notice(on), {**event, "text": "   "})
    assert len(sent) == 2, "an empty sentence is not a push"


def test_notice_handler_never_raises_into_the_bus(monkeypatch):
    notifier = _load_notifier()

    async def broken_push(api, chat_id, text, *, trust_env=False):
        raise RuntimeError("network down")

    monkeypatch.setattr(notifier, "_push_notification", broken_push)
    monkeypatch.setattr(notifier, "_load_settings", lambda api: api.settings)
    api = _Api({"TELEGRAM_CHAT_ID": "42", "TELEGRAM_NOTIFY_NOTICES": "on"})
    _run(notifier._make_notice(api), {"text": "hi", "source": "skill:x"})
    assert api.logged and api.logged[-1][0] == "error"


def test_manifest_and_settings_form_declare_the_notice_lane():
    manifest = (REPO / "skills" / "telegram" / "SKILL.md").read_text(encoding="utf-8")
    assert "owner.notification" in manifest.split("---", 2)[1]
    plugin = (REPO / "skills" / "telegram" / "plugin.py").read_text(encoding="utf-8")
    assert 'api.subscribe_event("owner.notification"' in plugin
    assert '"TELEGRAM_NOTIFY_NOTICES"' in plugin.split("_SETTINGS_FORM_KEYS", 1)[1].split(")", 1)[0]
