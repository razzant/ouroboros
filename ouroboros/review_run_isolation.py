"""Isolated review run: the data-root switch and the run identities it sets.

The contributor review lane (``scripts/run_external_review.py --contributor``)
makes its review drive the process's WHOLE data root, so every default writer
(the usage store and its one-time journal import, capability evidence, reviewer
markers, locks) stays off the host. Host-rooted reads stay deliberate: the
pinned settings document, the wrapper's keys-file fallback and the attached
engine's marker/descriptor/token; that engine journals its own runs in its home.
Its two run identities are read by the runtime: ``REVIEW_RUN_CAP_ENV``, the
explicit USD cap that is the isolated ledger's whole global limit
(``settings_setup_contract.resolve_total_budget_usd``), and ``ATTACH_HOME_ENV``,
the host engine home reached attach-only (``claudexor_daemon``). Stdlib-only:
it runs before ``ouroboros.config`` freezes ``DATA_DIR`` at import.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import pathlib
import sys
import tempfile
from typing import Any, Dict, Optional

from ouroboros.settings_integrity import SETTINGS_INTEGRITY_ENV

REVIEW_RUN_CAP_ENV = "OUROBOROS_REVIEW_RUN_CAP_USD"
ATTACH_HOME_ENV = "OUROBOROS_CLAUDEXOR_ATTACH_HOME"
ISOLATION_RECORD = "contributor-review-isolation.json"
# Under the integrity pin the pinned document IS the review panel: an inherited
# projection of these keys (possibly older than the document) never stands in
# for it — a key the document names is its value, one it leaves unset is the
# product default, as on the host. A default panel's model and provider inputs
# come from the document too, as a host task derives them
# (``run_external_review._pinned_default_panel_view``); every other key keeps "an
# explicit environment value wins" (``run_external_review._load_settings_into_env``).
# The retired reviewer comma-lists (``settings_defaults.RETIRED_COMMA_LIST_SETTING_KEYS``)
# are not panel keys: the settings read seam drops them from a document and
# their environment spellings are only a projection derived from a panel, so
# under the pin neither the document nor an inherited environment supplies one.
PINNED_PANEL_KEYS = frozenset({
    "OUROBOROS_REVIEWER_SLOTS", "OUROBOROS_SUBAGENTS", "OUROBOROS_EFFORT_REVIEW",
    "OUROBOROS_EFFORT_SCOPE_REVIEW",
})


def parse_run_cap(raw: Any) -> float:
    """A positive finite USD amount, else ``ValueError``."""
    try:
        cap = float(str(raw or "").strip())
    except ValueError:
        cap = math.nan
    if not math.isfinite(cap) or cap <= 0:
        raise ValueError("--run-cap-usd must be a positive finite USD amount")
    return cap


def run_cap_from_env() -> Optional[float]:
    """The launcher's run cap; ``None`` when unset, a zero allowance when unreadable."""
    raw = str(os.environ.get(REVIEW_RUN_CAP_ENV, "") or "").strip()
    if not raw:
        return None
    try:
        return parse_run_cap(raw)
    except ValueError:
        return 0.0


def retire_comma_lists(document: Dict[str, Any]) -> Dict[str, Any]:
    """The pinned document without a retired comma-list, none left in the environment either."""
    from ouroboros.settings_defaults import RETIRED_COMMA_LIST_SETTING_KEYS  # not at import: this leaf loads first

    for key in RETIRED_COMMA_LIST_SETTING_KEYS:
        os.environ.pop(key, None)
    return {key: value for key, value in document.items() if key not in RETIRED_COMMA_LIST_SETTING_KEYS}


def attach_home() -> Optional[pathlib.Path]:
    """The host engine home selected for attach-only use, or ``None``."""
    raw = str(os.environ.get(ATTACH_HOME_ENV, "") or "").strip()
    return pathlib.Path(raw) if raw else None


def _within(path: pathlib.Path, root: pathlib.Path) -> bool:
    """``path`` is ``root`` or below it under any spelling ``resolve`` keeps: case-folded (a
    case-insensitive volume), or the same directory reached another way (a firmlink)."""
    folded = [part.casefold() for part in root.parts]
    for level in (path, *path.parents):
        if [part.casefold() for part in level.parts] == folded:
            return True
        try:
            if os.path.samefile(level, root):
                return True
        except OSError:  # not there (yet): its existing ancestors decide
            continue
    return False


def isolate_review_data(*, host_data: pathlib.Path, drive_root: str, run_cap: str,
                        attach_host_engine: bool) -> Dict[str, Any]:
    """Make the drive this process's data root; return the facts (and record them once).

    The host settings stay in place under the integrity pin (verified on every
    read, refused to every writer, never copied). A drive reused for a
    continuation keeps its ledger and must keep its recorded cap.
    """
    if "ouroboros.config" in sys.modules:
        raise RuntimeError("ouroboros.config was imported before the review data root was isolated")
    cap = parse_run_cap(run_cap)
    host = pathlib.Path(host_data).expanduser().resolve(strict=False)
    if not drive_root:  # a default drive is allocated only below a temporary root outside the host
        temp_root = pathlib.Path(tempfile.gettempdir()).resolve()
        if _within(temp_root, host):
            raise RuntimeError(f"the temporary directory {temp_root} is inside the host data root {host}; "
                               "pass a --drive-root outside it")
        drive_root = tempfile.mkdtemp(prefix="ouroboros-external-review-", dir=temp_root)
    drive = pathlib.Path(drive_root).expanduser().resolve(strict=False)
    if _within(drive, host) or _within(host, drive):
        raise RuntimeError(f"the review drive {drive} overlaps the host data root {host}")
    drive.mkdir(parents=True, exist_ok=True)
    record_path = drive / ISOLATION_RECORD
    record = json.loads(record_path.read_text(encoding="utf-8")) if record_path.is_file() else {}
    if not record and any(drive.iterdir()):  # another data root's ledger would not start empty
        raise RuntimeError(f"the review drive {drive} is not empty and no contributor review opened it; "
                           "pass a new or empty --drive-root")
    if record and float(record.get("run_cap_usd") or 0) != cap:
        raise RuntimeError(f"this review drive was opened with --run-cap-usd {record.get('run_cap_usd')}; "
                           "a continuation keeps that cap and the spend already recorded against it")
    settings = pathlib.Path(
        os.environ.get("OUROBOROS_SETTINGS_PATH", "") or (host / "settings.json")
    ).expanduser().resolve(strict=False)
    try:
        settings_sha = hashlib.sha256(settings.read_bytes()).hexdigest()
    except FileNotFoundError:
        if attach_host_engine:  # an engine host without settings is a mistaken root, never the default panel
            raise RuntimeError(f"--attach-host-engine found no host settings at {settings}; "
                               "point OUROBOROS_DATA_DIR at the host data root") from None
        settings, settings_sha = drive / "settings.json", ""  # product defaults; writes stay here
    selected = {"OUROBOROS_DATA_DIR": str(drive), "OUROBOROS_SETTINGS_PATH": str(settings),
                SETTINGS_INTEGRITY_ENV: settings_sha, REVIEW_RUN_CAP_ENV: repr(cap),
                ATTACH_HOME_ENV: str(host / "claudexor") if attach_host_engine else ""}
    for key, value in selected.items():  # an inherited pin or attach is not this run's choice
        if value:
            os.environ[key] = value
        else:
            os.environ.pop(key, None)
    facts = {"host_data_root": str(host), "review_data_root": str(drive),
             "settings_path": str(settings), "settings_sha256": settings_sha or None,
             "run_cap_usd": cap, "attached_engine_home": selected[ATTACH_HOME_ENV] or None}
    if not record:
        record_path.write_text(json.dumps(facts, indent=2) + "\n", encoding="utf-8")
    return facts
