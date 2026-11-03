"""Interface-language gateway envelopes (``/api/ui/i18n*``): their own module beside ``contracts.py``.

Descriptive TypedDicts, the twin of ``web/modules/ui_i18n_types.js``; the memory file they
describe is ``ouroboros/i18n_memory.py``'s schema 1."""

from __future__ import annotations

from typing import Any

try:  # Python 3.11+
    from typing import Literal, NotRequired, Required, TypedDict  # type: ignore[attr-defined]
except ImportError:  # pragma: no cover - CI supports Python 3.10.
    from typing_extensions import Literal, NotRequired, Required, TypedDict  # type: ignore[assignment]


class UiI18nProfile(TypedDict):
    """A language's profile inside its translation memory (ouroboros/i18n_memory.py)."""

    label: str  # display name the owner typed or the generator chose
    instruction: str  # free-text description for the generator (an invented language's brief)
    direction: Literal["ltr", "rtl"]
    lexicon: NotRequired[str]  # generator-written vocabulary/rules for rare or invented languages (file and export only)
    lexicon_chars: NotRequired[int]  # the GET carries the lexicon's size, not its text


class UiI18nEntry(TypedDict):
    """One translation: `text`, or plural `forms` keyed by CLDR category."""

    text: NotRequired[str]
    forms: NotRequired[dict[str, str]]
    provenance: Literal["generated", "owner", "imported"]
    source_hash: NotRequired[str]
    model: NotRequired[str]
    at: NotRequired[str]
    attempt_id: NotRequired[str]
    pack: NotRequired[str]
    pack_version: NotRequired[str]
    context: NotRequired[str]
    source: NotRequired[str]  # the English a code entry translated (the browser's reword check)


class UiI18nStats(TypedDict):
    entries: int
    generated: int
    owner: int
    imported: int
    stale: int | None  # None = the current English of code keys was not available
    pending: int  # queued misses awaiting the generator
    refused: int  # keys the generator gave up on (cleared by Regenerate)


class UiI18nLanguageSummary(TypedDict):
    """One language present on disk, for the Settings select."""

    language: str
    label: str
    entries: int
    pending: int
    malformed: bool


class UiI18nGenerator(TypedDict):
    """The translation generator's state in this server process (ouroboros/ui_translation.py)."""

    state: Literal["idle", "running", "no_model", "failed"]
    error: str  # the last failure, secret-free; "" when the state is not failed/no_model
    updated_at: str
    applied: int  # entries written since the process started
    language: str  # the language the worker last ran for ("" before any run)
    in_flight: int  # keys of the batch at the model right now (counted into stats.pending)


class UiI18nResponse(TypedDict):
    """GET /api/ui/i18n and the body of a successful language POST."""

    ok: NotRequired[bool]
    language: str  # BCP-47 tag; "" = not chosen (English source renders)
    chosen: bool
    english: bool  # not chosen or chosen English: entries are empty by construction
    revision: int
    profile: UiI18nProfile | None
    plural_select: dict[str, Any] | None  # {"map": {"0": "other", ...}, "period": int|None}, written by the browser
    plural_categories: list[str] | None
    entries: dict[str, UiI18nEntry]
    stats: UiI18nStats
    updated_at: str
    memory_error: str  # nonempty when the stored file is malformed (English fallback in effect)
    languages: list[UiI18nLanguageSummary]
    generator: UiI18nGenerator


class UiI18nLanguageRequest(TypedDict):
    """POST /api/ui/i18n/language."""

    language: str  # BCP-47 tag ("" = not chosen, "en" = chosen English)
    label: NotRequired[str]
    profile: NotRequired[dict[str, Any]]
    plural_select: NotRequired[dict[str, Any]]  # Intl.PluralRules select(n) for 0..100 (+ period)
    plural_categories: NotRequired[list[str]]


class UiI18nMissingRequest(TypedDict):
    """POST /api/ui/i18n/missing: strings the renderer could not translate."""

    language: str  # must equal the install's current language (409 otherwise)
    items: list[dict[str, Any]]  # {key, context?}; shape-filtered, at most 200 per call


class UiI18nMissingResponse(TypedDict):
    accepted: int
    dropped: int
    pending: int


class UiI18nMemoryDocument(TypedDict, total=False):
    """A translation memory file (schema 1): what GET /api/ui/i18n/export downloads and POST
    /api/ui/i18n/import accepts. Import needs a non-English ``language``; ``schema``, when sent,
    must be 1. Its ``entries`` replace the whole imported layer: owner entries stay and shadow it,
    generated entries yield, imported entries absent from the new file are dropped. A body
    without ``entries`` is read as the entries map itself. Unknown top-level keys are kept."""

    schema: int
    language: Required[str]
    revision: int
    created_at: str
    updated_at: str
    profile: UiI18nProfile
    plural_select: dict[str, Any] | None
    plural_categories: list[str] | None
    glossary_hash: str
    entries: dict[str, UiI18nEntry]
    shadow: dict[str, UiI18nEntry]  # imported entries an owner entry hides
    refused: dict[str, dict[str, Any]]  # keys the generator gave up on
    pack: str  # stamped on every imported entry
    pack_version: str


class UiI18nImportResult(TypedDict):
    added: int
    replaced: int
    shadowed: int  # imported entries kept behind an owner entry
    dropped: int  # earlier imported entries absent from this file


class UiI18nImportResponse(TypedDict):
    """POST /api/ui/i18n/import: the memory's new revision. Every import bumps it."""

    ok: bool
    language: str
    revision: int
    result: UiI18nImportResult
    stats: UiI18nStats


class UiI18nRegenerateRequest(TypedDict, total=False):
    """POST /api/ui/i18n/regenerate: drop generated entries and refusals, then generate again.
    An absent or empty ``language`` means the install's current language."""

    language: str


class UiI18nRegenerateResponse(TypedDict):
    ok: bool
    language: str
    revision: int
    removed: int  # generated entries dropped; progress shows in GET /api/ui/i18n stats.pending
    stats: UiI18nStats


class UiLanguageChangedOutbound(TypedDict):
    """WS frame: the install's interface language was chosen (any client or transport); re-read GET /api/ui/i18n."""

    type: Literal["ui_language_changed"]
    language: str  # the chosen BCP-47 tag ("" = not chosen)


class UiI18nUpdatedOutbound(TypedDict):
    """WS frame: the translation memory of ``language`` reached ``revision`` (generated, imported, regenerated or an owner pin)."""

    type: Literal["ui_i18n_updated"]
    language: str
    revision: int
    applied: NotRequired[int]  # entries the generator wrote in this batch; absent for import, regenerate and a pin


__all__ = [
    "UiI18nProfile", "UiI18nEntry", "UiI18nStats", "UiI18nLanguageSummary", "UiI18nGenerator",
    "UiI18nResponse", "UiI18nLanguageRequest", "UiI18nMissingRequest", "UiI18nMissingResponse",
    "UiI18nMemoryDocument", "UiI18nImportResult", "UiI18nImportResponse",
    "UiI18nRegenerateRequest", "UiI18nRegenerateResponse",
    "UiLanguageChangedOutbound", "UiI18nUpdatedOutbound",
]
