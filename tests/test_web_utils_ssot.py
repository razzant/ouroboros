"""SSOT contract for shared web helpers (v5.8.3-rc.5 dedup pass).

``safeExternalHrefAttr``, ``renderMarkdownSafe``, ``boundedText``, and
``fetchJson`` previously had local copies in ``marketplace.js``,
``skills.js``, ``widgets.js``, and ``ouroboroshub.js``. They are now
owned by ``web/modules/utils.js`` for content helpers and
``web/modules/api_client.js`` for the gateway JSON-fetch helper, so the
URL/markdown/API error contracts cannot drift between modules.

These checks are static-text guards: they pin the helpers' presence in
``utils.js`` and the absence of duplicate function definitions in the
consumer modules. Any reintroduction of a local copy would be caught here
before the marketplace/skills/widgets surface diverged on a security
boundary (e.g. a publisher-supplied ``javascript:`` href slipping through
because one module's local helper was missing the protocol allowlist).
"""

from __future__ import annotations

import re

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
WEB_MODULES = REPO_ROOT / "web" / "modules"


def _read(name: str) -> str:
    return (WEB_MODULES / name).read_text(encoding="utf-8")


def test_skills_does_not_redeclare_shared_helpers():
    src = _read("skills.js")
    renderer = _read("skill_card_renderer.js")
    api_client = _read("api_client.js")
    assert "boundedText" in src
    assert "safeExternalHrefAttr as safeExternalUrl" in renderer
    assert "function boundedText(" not in src
    assert "function safeExternalUrl(" not in src + renderer
    # The host names the removal (listing `removal`); the card renders it and derives none itself (inverted pin).
    assert "skill.removal === 'delete'" in renderer and "skill.removal === 'uninstall'" in renderer
    assert "startsWith('skills/external/')" not in renderer and "if (!isBuiltIn(skill))" not in renderer
    assert "skills-delete-local" in renderer
    assert "apiClient.deleteSkill(name, payloadRoot || skill.payload_root, skill.content_hash)" in src
    assert "/api/skills/${encodeURIComponent(skill)}/delete" in api_client
    assert "payload_root: payloadRoot" in api_client
    assert "expected_content_hash: expectedContentHash" in api_client


def test_accent_tokens_have_concrete_rgba_values():
    """The v5.8.3-rc.5 ``--accent-*`` numbered fade tokens added to
    ``web/style.css`` :root must each map to a concrete
    ``rgba(201, 53, 69, X)`` value — never to ``var(--accent-XX)`` (a
    self-reference forms an invalid CSS cycle and the entire crimson
    accent system silently fails to apply at computed-value time).

    Triad reviewers (gpt-5.5, gemini-3.5-flash, claude-opus-4.6) caught
    this exact regression in the first dry-run of v5.8.3-rc.5: a
    file-wide sed swap had also rewritten the :root definitions
    themselves, leaving every ``--accent-04: var(--accent-04);``-style
    cycle. This guard pins the fix and prevents the same regression
    class from returning silently.
    """
    src = (REPO_ROOT / "web" / "ui.css").read_text(encoding="utf-8")
    root_match = re.search(r":root\s*\{([^}]+)\}", src, re.S)
    assert root_match, ":root block not found in web/ui.css"
    root_body = root_match.group(1)

    # The ladder is read from the file rather than snapshotted here: rungs are
    # added and removed as the accent's call sites move onto tokens, and a
    # hardcoded list turns that into an unrelated test failure while silently
    # exempting any rung added after it was written.
    ladder = re.findall(r"(--accent-(?:dim|\d{2}))\s*:\s*([^;]+);", root_body)
    assert len(ladder) >= 8, f"the accent alpha ladder looks truncated: {ladder}"
    for name, raw_value in ladder:
        value = raw_value.strip()
        # The value must be a literal rgba(...) anchored on the crimson
        # accent triple; never a ``var(<itself>)`` cycle, never a ``var()``
        # to a different accent token (that pattern works in CSS but
        # would silently break this token if its referent regresses).
        assert "var(" not in value, (
            f"{name} must not reference another CSS variable (cycle / "
            f"silent-drift risk); got: {value!r}"
        )
        assert "rgba(201, 53, 69," in value, (
            f"{name} must be defined on the crimson accent triple "
            f"rgba(201, 53, 69, X); got: {value!r}"
        )


def test_status_read_provenance_has_one_reader_across_surfaces():
    """``claudexor_status_store.facetReadState`` is the ONE place that decides
    what an empty status collection MEANS (ok / not_read / failed / transport /
    indeterminate), and no consumer may re-parse the ``reads`` block or the
    daemon state on its own.

    The predecessor of this test (``test_onboarding_account_truth_mirrors_the_
    panel_rules``) compared the wizard's IIFE mirrors of these rules against the
    accounts panel clause by clause — the wizard could not import back then, so
    a restated copy was the least evil and the pin guarded its drift. The wizard
    is a linked ES module now and the mirrors are deleted; the pin flips from
    "the copies must agree" to "there must be no copies". The behaviour of the
    single reader itself (an unread facet claims nothing; ``unreachable`` is
    never worded as "not running"; the first render says "Reading…", not a
    verdict) is pinned in web/tests/claudexor_status_store.test.js.
    """
    store = _read("claudexor_status_store.js")
    assert "export function facetReadState(" in store, (
        "the store lost the single facet reader; every consumer would grow "
        "its own definition of what an empty collection means"
    )
    assert "export function accountRows(" in store, (
        "the store lost the single connected-rows projection"
    )
    # #244: a native-only machine is an engine on the retired account model.
    # It reads as that typed state, decided once beside the projection, never
    # as "0 accounts connected" and never as pseudo-rows re-read off the old
    # client's carrier.
    rows_body = store.split("export function accountRows(", 1)[1].split("\nexport ", 1)[0]
    rows_code = "\n".join(line for line in rows_body.splitlines() if not line.strip().startswith("//"))
    assert "harnessAccounts" not in rows_code, "accountRows reads the retired native carrier again"
    assert "export function accountModelUnsupported(" in store, (
        "the store lost the one retired-account-model decision; a pre-3.6 engine "
        "would read as 0 accounts connected again"
    )
    # «Не вини демона, который отвечал»: `unreachable` is the endpoint's global
    # "the answer did not complete" — a RUNNING daemon refusing one fanned-out
    # read also lands there, so treating it as stopped would print "the daemon
    # is not running" about a daemon that answered.
    stopped = re.search(r"export const DAEMON_STATES_STOPPED = \[([^\]]*)\]", store)
    assert stopped, "the store no longer declares which daemon states mean stopped"
    assert "unreachable" not in stopped.group(1), (
        "`unreachable` classified as stopped: a refused read would be announced "
        "as 'the daemon is not running' about a daemon that answered"
    )

    # No consumer redeclares the reader or re-derives a facet from the payload.
    consumers = (
        "onboarding_wizard.js",
        "onboarding_agents_step.js",
        "harness_accounts.js",
        "subagents_settings.js",
    )
    for name in consumers:
        src = _read(name)
        assert "function facetReadState(" not in src, (
            f"{name} redeclares facetReadState instead of using the store's reader"
        )
        assert "function accountRows(" not in src, (
            f"{name} redeclares accountRows — a second definition of 'connected'"
        )

    # The wizard specifically: its old IIFE mirrors must stay dead, and its only
    # view of account state is the Agents step riding the shared store.
    wizard = _read("onboarding_wizard.js")
    for mirror in ("claudexorAccountsRead", "countConnectedAccounts"):
        assert mirror not in wizard, (
            f"the wizard's dead IIFE mirror {mirror} came back; the store is "
            "the one reader of /api/claudexor/status"
        )
    assert "/api/claudexor/status" not in wizard, (
        "the wizard fetches the status endpoint directly instead of going "
        "through the shared store"
    )
    assert "from './onboarding_agents_step.js';" in wizard

    # The Agents step consumes provenance through the store's own vocabulary:
    # the shared sentence (unavailableNote) and the accounts-facet licence
    # (accountsKnown) — never a homegrown re-reading of the payload.
    step = _read("onboarding_agents_step.js")
    assert "from './claudexor_status_store.js';" in step
    assert "accountRows" in step
    assert "unavailableNote(" in step, (
        "the Agents step stopped using the store's shared unavailable sentence"
    )
    assert "accountsKnown" in step, (
        "the Agents step no longer gates its claims on the accounts read facet"
    )

    def code_only(src: str) -> str:
        """Drop // and block-comment prose — a rule that lives only in prose is
        not a rule, and a pin that reads prose fails on a comment that merely
        NAMES the daemon while explaining why the code never touches it."""
        return "\n".join(
            line for line in src.splitlines()
            if not line.strip().startswith(("//", "*", "/*"))
        )

    assert "daemon" not in code_only(step), (
        "the Agents step reads the daemon state directly; the store's facets "
        "are the only licensed view of it"
    )
