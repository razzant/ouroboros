"""The bridge speaks the install's interface language.

Every line this skill composes itself (menu titles, buttons, card hints, notifications,
health words) is authored in English here and translated by the install's translation
memory (``ouroboros/i18n_memory.py``) under the one install-wide language,
``OUROBOROS_UI_LANGUAGE`` — the same choice the web UI and the desktop window follow.
The skill ships no dictionary: a table registers its English rows with the translation
generator (``ouroboros/ui_translation.py``) under ``code:tg.<table>.<key>``, and the
light model fills the memory when a language is chosen.

Two seams:

* ``Index`` / ``Texts`` — a table of English rows read by key in a language:
  ``TEXTS[lang]["btn_back"]``. The existing call sites keep their shape; only the source
  of truth moved from an ``en``/``ru`` pair to one English table plus the memory.
* ``language()`` — the install's tag (``""`` reads as English) and ``label(tag)``.

A host row reaches Telegram as one line from ``ouroboros.host_notice.notice_line``: a
notice's phrase and its step, and the technical pause (a host-authored reason with no
quoted user content), each part translated before they are joined. A host sentence a
panel relays without a code (a work's reason, ``telegram_health``) is translated by its
text (``host_sentence``). Other relays can include ids and the author's own words.

The bridge's former private ``TELEGRAM_LANGUAGE`` setting is migrated once
(``migration_target``): an install that chose Russian in the bridge before the install-wide
language existed chooses ``ru`` for the whole installation.
"""

from __future__ import annotations

import logging
import pathlib
from typing import Any, Callable, Dict, Iterator, Mapping, Optional

from ouroboros import i18n_memory as memory
from ouroboros.ui_language import is_english, normalize_language_tag

log = logging.getLogger(__name__)

CATALOG_PREFIX = "tg"
_DRIVE_ROOT: Optional[pathlib.Path] = None
_REGISTERED: Dict[str, Dict[str, str]] = {}
# Keys already reported as missing in this process, per language: a row the memory lacks is
# reported once, not on every render (the pending file dedupes too; this spares the disk).
_REPORTED: Dict[str, set] = {}


def configure(drive_root: Optional[pathlib.Path]) -> None:
    """The runtime data root the memory lives under (``register`` sets it from the api;
    tests point it at their own root). ``None`` falls back to the configured data dir."""
    global _DRIVE_ROOT
    _DRIVE_ROOT = pathlib.Path(drive_root) if drive_root is not None else None


def drive_root() -> pathlib.Path:
    if _DRIVE_ROOT is not None:
        return _DRIVE_ROOT
    from ouroboros.config import DATA_DIR

    return pathlib.Path(DATA_DIR)


def language() -> str:
    """The install's interface language; ``""`` when none is chosen (English renders)."""
    return memory.current_language()


def english(tag: Optional[str]) -> bool:
    return not tag or is_english(tag)


def label(tag: Optional[str]) -> str:
    """A language's display name: its memory profile label, else the tag, else English."""
    normalized = normalize_language_tag(tag)
    if not normalized or is_english(normalized):
        return "English"
    doc = memory.cached_memory(drive_root(), normalized)
    return str(((doc or {}).get("profile") or {}).get("label") or normalized)


def known_languages() -> list:
    """Languages with a memory on disk, for the ``/language`` keyboard."""
    return memory.list_languages(drive_root())


class Texts(Mapping[str, str]):
    """One table's rows in one language: ``texts["key"]`` is the memory's translation of
    ``code:tg.<table>.<key>`` or the English row. Templates keep their ``{placeholders}``
    (the generator validates that), so ``texts["x"].format(...)`` works in every language."""

    def __init__(self, table: str, rows: Dict[str, str], tag: str) -> None:
        self._table = table
        self._rows = rows
        self._tag = normalize_language_tag(tag) or ""

    def __getitem__(self, key: str) -> str:
        source = self._rows[key]
        if english(self._tag):
            return source
        code = f"{memory.CODE_PREFIX}{CATALOG_PREFIX}.{self._table}.{key}"
        found = memory.tr(code, self._tag, None, drive_root=drive_root())
        if isinstance(found, str) and found:
            return found
        # English for now; the miss goes to the generator exactly as a browser miss would, so a
        # table registered after the language was chosen (a skill enabled later, a new line) is
        # translated without another language event.
        _report_miss(code, self._tag, {"source": source, "table": f"{CATALOG_PREFIX}.{self._table}", "role": "telegram"})
        return source

    def format(self, key: str, **params: Any) -> str:
        """``self[key].format(**params)`` that never breaks the bridge: a generated template
        whose placeholders do not fit the call renders the English row instead."""
        template = self[key]
        try:
            return template.format(**params)
        except Exception:  # noqa: BLE001 — any formatting failure of a translated template falls back to English
            log.warning("telegram i18n: template %s.%s does not take its parameters; English used", self._table, key)
            return self._rows[key].format(**params)

    def get(self, key: str, default: Any = None) -> Any:  # type: ignore[override]
        return self[key] if key in self._rows else default

    def __iter__(self) -> Iterator[str]:
        return iter(self._rows)

    def __len__(self) -> int:
        return len(self._rows)

    @property
    def tag(self) -> str:
        return self._tag


class Index:
    """``INDEX[lang]`` → ``Texts`` for that language. Registers the English table with the
    translation generator once, so a chosen language gets these rows in its first batch."""

    def __init__(self, table: str, rows: Dict[str, str], note: str) -> None:
        self.table = table
        self.rows = dict(rows)
        self.note = note
        _REGISTERED[table] = self.rows
        try:
            from ouroboros.ui_translation import register_catalog

            register_catalog(f"{CATALOG_PREFIX}.{table}", lambda rows=self.rows: rows, note)
        except Exception:  # the generator is optional at import time; the English stays
            log.debug("telegram i18n: catalog registration skipped for %s", table, exc_info=True)

    def __getitem__(self, lang: Optional[str]) -> Texts:
        return Texts(self.table, self.rows, lang or "")

    @property
    def en(self) -> Dict[str, str]:
        return self.rows


def host_sentence(source: str, tag: str) -> str:
    if english(tag):
        return source
    found = memory.tr(source, tag, drive_root=drive_root())
    if found is None:
        _report_miss(source, tag, {"source": source, "role": "telegram"})
    return found or source


def _report_miss(key: str, tag: str, context: Dict[str, Any]) -> None:
    seen = _REPORTED.setdefault(tag, set())
    if key in seen:
        return
    seen.add(key)
    try:
        root = drive_root()
        if memory.record_missing(root, tag, [{"key": key, "context": context}])["accepted"]:
            _wake_generator(root, tag)
    except Exception:
        log.debug("telegram i18n: miss not recorded", exc_info=True)


def _wake_generator(root: pathlib.Path, tag: str) -> None:
    try:
        from ouroboros.ui_translation import on_language_event

        on_language_event("missing", root, tag)
    except Exception:
        log.debug("telegram i18n: generator wake skipped", exc_info=True)


def migration_target(settings: Mapping[str, Any]) -> Optional[str]:
    """The install language the bridge's retired ``TELEGRAM_LANGUAGE`` implies, once.

    ``ru`` chosen in the bridge before the install-wide language existed means the owner
    wants Russian; an install that has since chosen any language keeps its choice. ``None``
    when nothing is to be done (already migrated, English, or a language already chosen)."""
    if settings.get("TELEGRAM_LANGUAGE_MIGRATED"):
        return None
    legacy = normalize_language_tag(str(settings.get("TELEGRAM_LANGUAGE") or "").strip().lower())
    if not legacy or is_english(legacy) or language():
        return None
    return legacy


LanguagePost = Callable[[Dict[str, Any]], Any]

__all__ = [
    "CATALOG_PREFIX", "Index", "Texts", "configure", "drive_root", "language", "english", "label",
    "known_languages", "migration_target",
]
