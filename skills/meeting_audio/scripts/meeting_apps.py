"""Which application's audio is "the meeting": names, matching and typed refusals.

Capturing only the meeting application is what keeps Ouroboros's own whispered
hints (played by the owner's surface) out of the meeting audio. The owner or the
mind may name the application (``app``); when nobody does, the default is every
running application from the short table below of programs that host calls,
because those identifiers are external facts the platforms report (bundle ids on
macOS, executable names on Windows, client names on Linux), not a judgement about
what is being said. A browser is never a default: the Ouroboros page itself may
be playing in it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

# (owner-facing name, is a browser, identifiers as the platforms report them, lowercase)
KNOWN_APPS: Tuple[Tuple[str, bool, Tuple[str, ...]], ...] = (
    ("Zoom", False, ("us.zoom.xos", "zoom.us", "zoom.exe", "zoom", "zoom.real", "zoom voiceengine")),
    ("Microsoft Teams", False, ("com.microsoft.teams2", "com.microsoft.teams", "ms-teams.exe",
                                "teams.exe", "teams-for-linux", "microsoft teams")),
    ("Webex", False, ("cisco-systems.spark", "com.webex.meetingmanager", "webex.exe",
                      "ciscocollabhost.exe", "atmgr.exe", "webex")),
    ("Slack", False, ("com.tinyspeck.slackmacgap", "slack.exe", "slack")),
    ("Discord", False, ("com.hnc.discord", "discord.exe", "discord")),
    ("FaceTime", False, ("com.apple.facetime", "facetime")),
    ("Skype", False, ("com.skype.skype", "skype.exe", "skypeforlinux", "skype")),
    ("Telegram", False, ("ru.keepcoder.telegram", "org.telegram.desktop", "telegram.exe",
                         "telegram-desktop", "telegram")),
    ("Google Chrome", True, ("com.google.chrome", "chrome.exe", "google-chrome", "chrome",
                             "google chrome")),
    ("Chromium", True, ("org.chromium.chromium", "chromium", "chromium-browser")),
    ("Microsoft Edge", True, ("com.microsoft.edgemac", "msedge.exe", "microsoft-edge", "msedge",
                              "microsoft edge")),
    ("Firefox", True, ("org.mozilla.firefox", "firefox.exe", "firefox")),
    ("Safari", True, ("com.apple.safari", "safari")),
    ("Arc", True, ("company.thebrowser.browser", "arc.exe", "arc")),
    ("Brave", True, ("com.brave.browser", "brave.exe", "brave", "brave-browser")),
    ("Opera", True, ("com.operasoftware.opera", "opera.exe", "opera")),
    ("Vivaldi", True, ("com.vivaldi.vivaldi", "vivaldi.exe", "vivaldi")),
)
# How many candidate names a refusal lists before naming the rest as a count.
MAX_LISTED_CANDIDATES = 12


@dataclass(frozen=True)
class RunningApp:
    """One running program as the platform reports it.

    ``name`` is what the owner sees, ``ident`` the platform identifier (bundle id,
    executable or client binary), ``pid`` its process (the tree root on Windows),
    ``handle`` a backend-specific address such as a PipeWire node id.
    """

    name: str
    ident: str = ""
    pid: int = 0
    handle: str = ""


@dataclass(frozen=True)
class Selection:
    """The applications one capture listens to."""

    apps: Tuple[RunningApp, ...]
    label: str
    browser: bool
    requested: str = ""


class TargetRefused(Exception):
    """No capture target: a typed reason, plain words and the names that could be chosen."""

    def __init__(self, code: str, message: str, candidates: Sequence[str] = ()):
        super().__init__(message)
        self.code = code
        self.message = message
        self.candidates = list(candidates)


def known_app(app: RunningApp) -> Optional[Tuple[str, bool]]:
    keys = {app.ident.strip().lower(), app.name.strip().lower()}
    for name, browser, identifiers in KNOWN_APPS:
        if keys & set(identifiers):
            return name, browser
    return None


def display_name(app: RunningApp) -> str:
    known = known_app(app)
    return known[0] if known else (app.name or app.ident)


def listed(names: Iterable[str]) -> str:
    """Join names for a sentence, naming what is left out as a count (never silent)."""
    unique = list(dict.fromkeys(name for name in names if name))
    shown = unique[:MAX_LISTED_CANDIDATES]
    text = ", ".join(shown)
    if len(unique) > len(shown):
        text += f" and {len(unique) - len(shown)} more"
    return text


def _label(apps: Sequence[RunningApp]) -> str:
    names = list(dict.fromkeys(display_name(app) for app in apps))
    if len(names) <= 2:
        return " and ".join(names)
    return ", ".join(names[:-1]) + f" and {names[-1]}"


def _group(apps: Sequence[RunningApp]) -> List[str]:
    return list(dict.fromkeys(display_name(app) for app in apps))


def choose(running: Sequence[RunningApp], requested: str = "") -> Selection:
    """Select the meeting application(s) among ``running``.

    A named ``requested`` application matches a running program's name, platform
    identifier or owner-facing name, exactly first and then as a part of it;
    several different programs matching a part is a refusal that lists them.
    """
    wanted = (requested or "").strip()
    known_running = [app for app in running if known_app(app) is not None]
    browsers = [display_name(app) for app in known_running if known_app(app)[1]]
    if wanted:
        needle = wanted.lower()

        def keys(app: RunningApp) -> set:
            return {app.name.lower(), app.ident.lower(), display_name(app).lower()} - {""}

        matches = [app for app in running if needle in keys(app)]
        if not matches:
            matches = [app for app in running if any(needle in key for key in keys(app))]
        if not matches:
            choices = _group(known_running) or _group(running)
            raise TargetRefused(
                "MEETING_APP_NOT_FOUND",
                f"No running application matches '{wanted}'."
                + (f" Running now: {listed(choices)}." if choices else ""),
                choices,
            )
        groups = _group(matches)
        if len(groups) > 1:
            raise TargetRefused(
                "MEETING_APP_AMBIGUOUS",
                f"'{wanted}' matches several applications: {listed(groups)}. Name one of them.",
                groups,
            )
        return Selection(tuple(matches), _label(matches),
                         any((known_app(app) or ("", False))[1] for app in matches), wanted)
    meetings = [app for app in known_running if not known_app(app)[1]]
    if not meetings:
        hint = (f" If the meeting runs in a browser, name it with app (running: {listed(browsers)})."
                if browsers else " Name the application that plays the meeting with app.")
        raise TargetRefused("MEETING_APP_NOT_FOUND", "No meeting application is running." + hint,
                            list(dict.fromkeys(browsers)))
    return Selection(tuple(meetings), _label(meetings), False, "")


def same_browser_note(selection: Selection) -> str:
    """The disclosure a browser-hosted meeting needs: tabs of one browser cannot be separated."""
    if not selection.browser:
        return ""
    return (f"The meeting audio comes from {selection.label}. If the Ouroboros page is open in "
            f"{selection.label} too, Ouroboros's own whispered hints are in this meeting audio; "
            "speech in Ouroboros's own voice here is not a meeting participant.")
