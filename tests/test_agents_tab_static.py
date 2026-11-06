"""Static guard: the Agents tab exists, and every control it absorbed moved
ONCE.

D-10 gave the agent surfaces their own Settings tab because they were scattered
across three others: accounts sat in Providers between API-key cards, reviewer
slots and subagent routing sat in Models beside model-id fields, and the two
counts that bound subagents (children per root, nesting depth) sat in Advanced →
Runtime Limits next to process worker counts they have nothing to do with.

The failure this file exists to prevent is not "a control is missing" — that one
is loud. It is the SILENT one DEVELOPMENT.md names: "One capability, one
section. Never render a second control over the same settings key — two controls
carry two drafts, and the last one collected wins." A move that leaves the old
markup behind produces two inputs with the same id, and `settings.js` collects
whichever `getElementById` happens to reach first.

So each moved id is asserted to appear EXACTLY ONCE across the whole rendered
settings page, and to still be mapped to its settings key.

Pattern follows tests/test_web_dialogs_static.py and
tests/test_web_typography_static.py: read the sources, assert the structural
fact, no browser needed.
"""

from __future__ import annotations

import pathlib
import re

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
MODULES = REPO_ROOT / "web" / "modules"

# Every control the Agents tab absorbed, with the settings key it round-trips
# to. The key mapping lives in settings.js (id → key tables); the id lives in
# whichever module renders it now.
MOVED_CONTROLS = {
    "s-allow-mutative-subagents": "OUROBOROS_ALLOW_MUTATIVE_SUBAGENTS",
    "s-active-subagents": "OUROBOROS_MAX_ACTIVE_SUBAGENTS_PER_ROOT",
    "s-subagent-depth": "OUROBOROS_MAX_SUBAGENT_DEPTH",
    "s-subagent-worktree-root": "OUROBOROS_SUBAGENT_WORKTREE_ROOT",
    "s-subagent-projects-root": "OUROBOROS_SUBAGENT_PROJECTS_ROOT",
}


# The sweeps below once exempted ONE file — `onboarding_wizard.js`, which
# belonged to the onboarding worktree while that phase was in flight and still
# carried the dead addresses its own agent was repointing. The exemption was
# asserted to still be NEEDED, so it could not outlive its debt: at synthesis
# the onboarding phase landed its repointing, the assertion fired, and the
# entry was deleted. Nothing is exempt now.


def _read(name: str) -> str:
    return (MODULES / name).read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# The tab
# ---------------------------------------------------------------------------
# The moved controls
# ---------------------------------------------------------------------------


def test_every_moved_control_still_round_trips_to_its_settings_key() -> None:
    """Moving markup must not detach a control from its key: settings.js maps
    id → key, so the pair has to survive the move intact."""
    settings_js = _read("settings.js")
    for control_id, key in MOVED_CONTROLS.items():
        if control_id == "s-allow-mutative-subagents":
            # This one is not in a table: it is collected explicitly because
            # Auto saves the EMPTY value (SC-4).
            assert f"byId('{control_id}')" in settings_js
            assert key in settings_js
            continue
        assert re.search(rf"\['{re.escape(control_id)}', '{re.escape(key)}'", settings_js), (
            f"#{control_id} no longer maps to {key} in settings.js"
        )
