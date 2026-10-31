"""Gateway envelopes of the skill sources, catalog and install lifecycle (``gateway/skill_sources.py``).

Field-identical JSDoc twins live in ``web/modules/skill_source_contract_types.js``.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

try:  # Python 3.11+
    from typing import NotRequired, Required, TypedDict  # type: ignore[attr-defined]
except ImportError:  # pragma: no cover - CI supports Python 3.10.
    from typing_extensions import NotRequired, Required, TypedDict  # type: ignore[assignment]


class SkillSourceRow(TypedDict, total=False):
    """One source with its read state. ``kind``: builtin | hub | git | local | registry; ``trust``:
    trusted (installs without review) | review (reviews each skill); ``part`` names the half of a hub
    one row shows (official | community). ``state``: ready | reading | error | search. ``managed``: a
    build's settings supplied the row (shown as set by the organization; it stays editable)."""

    id: Required[str]
    name: Required[str]
    kind: Required[str]
    trust: Required[str]
    confirmed: Required[bool]  # false: the first-start confirmation of a preselected trust is pending
    url: str
    ref: str
    commit: str
    path: str
    part: str
    state: str
    last_read_at: str
    error: str
    refreshing: bool
    managed: bool


class SkillSourcesResponse(TypedDict):
    """``GET /api/skill-sources``. ``materialized`` is false while the list is derived from the older
    settings; ``problem`` explains a saved list that does not validate (the derived one is in use)."""

    sources: List[SkillSourceRow]
    materialized: bool
    problem: str


class SkillSourceProbeRequest(TypedDict, total=False):
    """``POST /api/skill-sources/probe``: a repository address or a folder on this computer."""

    url: str
    path: str


class SkillSourceProbeResponse(TypedDict, total=False):
    """What the address holds: ``kind`` hub (a ``hub.json`` repository), git (a plain skills repository)
    or local (a folder); ``found`` names its skills; ``owner`` is the hub's maintainer, else the account
    the address names (absent for a folder). ``error`` explains an unreadable address;
    ``warning: not_encrypted`` discloses an accepted plain-HTTP address."""

    kind: str
    suggested_name: str
    owner: str
    found: List[str]
    warning: str  # not_encrypted
    error: str


class SkillSourceAddRequest(TypedDict, total=False):
    """``POST /api/skill-sources``: an address (``url``) or a folder (``path``), a name and a trust."""

    url: str
    path: str
    name: Required[str]
    trust: Required[str]


class SkillSourceChangeRequest(TypedDict, total=False):
    """``POST /api/skill-sources/{source_id}``: rename, change trust (existing installs keep their
    basis) or confirm a preselected trust."""

    name: str
    trust: str
    confirmed: bool


class SkillSourceChangeResponse(TypedDict, total=False):
    """A saved source change: the whole list as it is now, plus the touched row."""

    ok: Required[bool]
    sources: List[SkillSourceRow]
    source: SkillSourceRow
    error: str
    code: str


class MySkillsRequest(TypedDict):
    """``POST /api/skill-sources/my-skills``: an empty object, an explicit owner action."""


class MySkillsResponse(TypedDict, total=False):
    """Created or connected the owner's repository. Failures use code no_token, cannot_create or failed."""

    ok: Required[bool]
    created: bool
    repository: str
    private: bool
    source: SkillSourceRow
    code: str
    error: str


class SkillCatalogItem(TypedDict, total=False):
    """One skill a source offers. ``id`` is ``<source_id>/<name>``; ``digest`` names the exact release;
    ``signals`` null means no data (never zero); ``installed`` null means not installed, else
    ``{state: installed, version, digest, modified, update_available}`` or ``{state: name_taken,
    source_id, source_name}``. ``updated_at`` is when the release last changed in its source (a git
    source's newest commit touching it, a registry's update time); absent when unknown.
    Official/recommended require exact endorsement through an Official hub row or the bundled source."""

    id: Required[str]
    source_id: Required[str]
    source_name: Required[str]
    trust: str
    name: Required[str]
    description: str
    category: List[str]
    version: str
    digest: str
    official: bool
    recommended: bool
    owner: str
    updated_at: str
    repo_url: str
    author_repo: str
    signals: Optional[Dict[str, Any]]  # {votes, comments, discussion_url, author_repo_stars}
    withdrawn: Optional[Dict[str, Any]]  # {reason, at}
    type: str
    needs_keys: List[str]
    permissions: List[str]
    install_specs: Any
    installed: Optional[Dict[str, Any]]
    registry_slug: str  # registry hits only
    registry_stats: Dict[str, Any]  # registry hits only


class SkillCatalogResponse(TypedDict, total=False):
    """``GET /api/skill-catalog``: every connected source's skills from the local cache (a stale or
    ``refresh=1`` read starts a background refresh), the category list and each source's read state.
    ``starter`` is the first-start installer's marker (``state`` upgrade: show the recommended shelf)."""

    items: Required[List[SkillCatalogItem]]
    categories: Required[List[Dict[str, str]]]  # [{id, label}]
    sources: Required[List[SkillSourceRow]]
    starter: Dict[str, Any]


class SkillCatalogSearchResponse(TypedDict, total=False):
    """``GET /api/skill-catalog/search``: matching items, registry (ClawHub) hits appended; a registry
    that could not answer sets ``registry_error`` and keeps the local matches."""

    items: Required[List[SkillCatalogItem]]
    registry_error: str


class SkillCatalogItemDetailResponse(TypedDict, total=False):
    """``GET /api/skill-catalog/item/{source_id}/{name}``: the detail sheet. ``uses`` lists what the skill
    can use ({kind: permission|key, id, label}); ``about`` is its SKILL.md text (``about_error`` when the
    files could not be read)."""

    item: Required[SkillCatalogItem]
    about: str
    about_error: str
    uses: List[Dict[str, str]]
    versions: List[Dict[str, Any]]
    links: Dict[str, str]


class SkillCatalogInstallRequest(TypedDict):
    """``POST /api/skill-catalog/install``. ``replace`` confirms replacing another skill of the same
    name; ``turn_on_when_ready`` (default true) turns the skill on once it is ready."""

    id: str
    turn_on_when_ready: NotRequired[bool]
    replace: NotRequired[bool]


class SkillCatalogInstallResponse(TypedDict, total=False):
    """An accepted install job (the lifecycle lane runs it; watch ``GET /api/skills/lifecycle-queue``),
    or a typed refusal (``code``: name_taken, withdrawn, update_instead, not_found, ...)."""

    ok: Required[bool]
    job_id: str
    name: str
    basis: str  # trusted | review
    review_job_id: str
    already_installed: bool
    code: str
    error: str
    occupant: Dict[str, Any]  # name_taken: {name, source, locations}


class SkillPublishDestinationChoice(TypedDict):
    """Where the owner publishes a skill: a source id (``hub`` = the hub) and ``mode`` origin (where it
    came from), entry (that source; the hub lists it as an entry) or new_entry (a new name in the hub)."""

    source_id: str
    mode: str


class SkillPublishPreflightRequest(TypedDict, total=False):
    """``POST /api/skills/{skill}/publish-preflight``: an optional destination; without one the skill's
    origin, else the hub. The answer's ``repository`` is the destination the publish task binds."""

    destination: SkillPublishDestinationChoice


class SkillUpdateRequest(TypedDict, total=False):
    """``POST /api/skills/{skill}/update``: ``mode`` replace | keep | merge, ``digest`` the candidate
    the dialog showed, ``local_digest`` the local revision it showed (a change refuses).
    ``skip_review`` is the owner's Update without review, only for replace."""

    mode: Required[str]
    digest: str
    local_digest: str
    skip_review: bool


class SkillUpdateResponse(TypedDict, total=False):
    """``replace`` answers an accepted job (``job_id``); ``keep`` the skipped release; ``merge`` the queued
    task (``base_available`` false: two-way only). A refusal carries ``code``."""

    ok: Required[bool]
    name: str
    mode: str
    job_id: str
    skipped: Dict[str, str]
    base_available: bool
    task: str
    candidate: Dict[str, str]
    local_digest: str
    code: str
    error: str


class SkillUninstallResponse(TypedDict, total=False):
    """``POST /api/skills/{skill}/uninstall``: one uninstall for every source; grants and the publication
    receipt survive."""

    ok: Required[bool]
    name: str
    source: str
    code: str
    error: str


class StarterSkillsResponse(TypedDict):
    """``GET /api/onboarding/starter-skills``: built-in and Official skills grouped by category (the ticked
    ones are the starter list), the two hub rows whose trust the owner confirms, and source states."""

    groups: List[Dict[str, Any]]  # [{category, label, items: [{id, name, description, official, builtin, recommended, needs_keys}]}]
    hub_rows: List[Dict[str, str]]  # [{id, name, trust, part}]
    sources: List[SkillSourceRow]


__all__ = [
    "MySkillsRequest", "MySkillsResponse",
    "SkillCatalogInstallRequest", "SkillCatalogInstallResponse", "SkillCatalogItem", "SkillCatalogItemDetailResponse",
    "SkillCatalogResponse", "SkillCatalogSearchResponse", "SkillPublishDestinationChoice",
    "SkillPublishPreflightRequest", "SkillSourceAddRequest", "SkillSourceChangeRequest",
    "SkillSourceChangeResponse", "SkillSourceProbeRequest", "SkillSourceProbeResponse", "SkillSourceRow",
    "SkillSourcesResponse", "SkillUninstallResponse", "SkillUpdateRequest", "SkillUpdateResponse",
    "StarterSkillsResponse",
]
