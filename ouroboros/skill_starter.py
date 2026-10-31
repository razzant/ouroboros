"""Starter skills: what a new install offers at first start, and the one-time installer.

The starter list is ``OUROBOROS_STARTER_SKILLS`` (catalog ids; an enterprise build ships its own). Its
default is every built-in (for this host) and Official hub skill the maintainer marked recommended.
Onboarding also offers configured ids from any source in their categories. An explicit [] selects none;
only absent/blank means defaults. Completing it installs once in the background; the build's
OUROBOROS_STARTER_SKILLS_TURN_ON (default true) decides first enablement. An install
that never shows onboarding (headless, Colab, a preset) runs the same installer at its first start with
the default list. ``state/skill_starter.json`` records the outcome so the installer never runs twice:
built-ins work offline, and a hub skill that could not be fetched yet is retried at a later start. An
existing install that upgrades gets only the bundled default-on built-ins, once (``skill_builtin``);
Browse shows the recommended shelf instead. The marker's ``defaults_offered`` names the default-on
built-ins an install was already offered, by its starter (ticked or not) or at an update.
"""

from __future__ import annotations

import json
import logging
import pathlib
import threading
import time
from typing import Any, Dict, List, Optional

from ouroboros.utils import atomic_write_json, read_json_dict, utc_now_iso

log = logging.getLogger(__name__)

STARTER_SETTING = "OUROBOROS_STARTER_SKILLS"
MARKER_FILENAME = "skill_starter.json"
_RETRY_CODES = frozenset({"fetch_failed", "source_unread"})
_LOCK = threading.Lock()
_THREADS_LOCK = threading.Lock()
_THREADS: set[threading.Thread] = set()


def marker_path(drive_root: pathlib.Path) -> pathlib.Path:
    return pathlib.Path(drive_root) / "state" / MARKER_FILENAME


def read_marker(drive_root: pathlib.Path) -> Dict[str, Any]:
    return read_json_dict(marker_path(drive_root)) or {}


def _write_marker(drive_root: pathlib.Path, **fields: Any) -> Dict[str, Any]:
    marker = {**read_marker(drive_root), **fields, "schema_version": 1, "updated_at": utc_now_iso()}
    atomic_write_json(marker_path(drive_root), marker)
    return marker


def _offer(drive_root: pathlib.Path, rows: List[Dict[str, Any]], repo_dir: Optional[pathlib.Path],
           configured: Optional[set[str]] = None) -> List[Dict[str, Any]]:
    """Built-in/Official items and every configured starter, from available source data."""
    from ouroboros.skill_source_catalog import items_for, snapshot_for

    out: List[Dict[str, Any]] = []
    configured = configured or set()
    for row in rows:
        usual = row["kind"] == "builtin" or (row["kind"] == "hub" and row.get("part") == "official")
        if usual or any(item.startswith(row["id"] + "/") for item in configured):
            snapshot, _status = snapshot_for(drive_root, row, repo_dir=repo_dir)
            out.extend(item for item in items_for(row, snapshot)
                       if (usual and (row["kind"] == "builtin" or item["official"])) or item["id"] in configured)
    found = {item["id"] for item in out}
    for item_id in sorted(configured - found):
        source_id, _, name = item_id.partition("/")
        out.append({"id": item_id, "source_id": source_id, "name": name, "category": [],
                    "description": "Source details are not available yet."})
    return out


def default_starter_ids(drive_root: pathlib.Path, rows: Optional[List[Dict[str, Any]]] = None, *,
                        repo_dir: Optional[pathlib.Path] = None) -> List[str]:
    from ouroboros.skill_sources import load_sources

    rows = rows if rows is not None else load_sources()
    return [item["id"] for item in _offer(drive_root, rows, repo_dir) if item.get("recommended")]


def starter_ids(drive_root: pathlib.Path, settings: Optional[Dict[str, Any]] = None,
                rows: Optional[List[Dict[str, Any]]] = None, *, repo_dir: Optional[pathlib.Path] = None) -> List[str]:
    """Configured ids, including an empty list; only absent or blank selects the defaults."""
    if settings is None:
        from ouroboros.config import load_settings

        settings = load_settings()
    raw = settings.get(STARTER_SETTING, "")
    if raw != "":
        try:
            value = json.loads(raw) if isinstance(raw, str) else raw
            if isinstance(value, list) and all(isinstance(item, str) for item in value):
                return list(dict.fromkeys(value))
        except ValueError:
            pass
        log.warning("%s is not a JSON list of catalog ids; no starter skills selected", STARTER_SETTING)
        return []
    return default_starter_ids(drive_root, rows, repo_dir=repo_dir)


def starter_groups(drive_root: pathlib.Path, *, rows: Optional[List[Dict[str, Any]]] = None,
                   repo_dir: Optional[pathlib.Path] = None) -> Dict[str, Any]:
    """``GET /api/onboarding/starter-skills``: the offer grouped by category, ticks = starter list."""
    from ouroboros.skill_source_catalog import snapshot_for
    from ouroboros.skill_sources import load_sources

    rows = rows if rows is not None else load_sources()
    ticked = set(starter_ids(drive_root, rows=rows, repo_dir=repo_dir))
    labels: Dict[str, str] = {}
    order: List[str] = []
    statuses = []
    for row in rows:
        snapshot, status = snapshot_for(drive_root, row, repo_dir=repo_dir)
        for category in (snapshot or {}).get("categories") or []:
            if category["id"] not in labels:
                labels[category["id"]] = category["label"]
                order.append(category["id"])
        if row["kind"] in {"builtin", "hub"} or any(i.startswith(row["id"] + "/") for i in ticked):
            statuses.append({**row, **status})
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for item in _offer(drive_root, rows, repo_dir, ticked):
        category = (item.get("category") or ["other"])[0]
        groups.setdefault(category, []).append({
            "id": item["id"], "name": item["name"], "description": item.get("description", ""),
            "official": bool(item.get("official")), "builtin": item["source_id"] == "builtin",
            "recommended": item["id"] in ticked, "needs_keys": list(item.get("needs_keys") or []),
        })
    builtin_first = [cat for cat, items in groups.items() if any(i["builtin"] for i in items)]
    ranked = sorted(groups, key=lambda cat: (cat not in builtin_first,
                                             order.index(cat) if cat in order else len(order), cat))
    return {
        "groups": [{"category": cat, "label": labels.get(cat, cat.capitalize()),
                    "items": sorted(groups[cat], key=lambda i: (not i["builtin"], i["name"]))} for cat in ranked],
        "hub_rows": [{"id": row["id"], "name": row["name"], "trust": row["trust"], "part": row.get("part", "")}
                     for row in rows if row["kind"] == "hub" and row.get("part")],
        "sources": statuses,
    }


def record_defaults_offered(drive_root: pathlib.Path, names: List[str]) -> None:
    """Default-on built-ins this install was offered: no later start offers them again. Never raises."""
    try:
        done = set(read_marker(drive_root).get("defaults_offered") or [])
        if names and not set(names) <= done:
            _write_marker(drive_root, defaults_offered=sorted(done | set(names)))
    except Exception:
        log.warning("Starter skill default offer could not be recorded", exc_info=True)


def _record_presented_defaults(drive_root: pathlib.Path, repo_dir: Optional[pathlib.Path]) -> None:
    """A starter choice presented the default-on built-ins (ticked by default); an untick stands."""
    from ouroboros.skill_builtin import bundled_skills_dir, default_on_names

    record_defaults_offered(drive_root, default_on_names(
        pathlib.Path(repo_dir) / "skills" if repo_dir is not None else bundled_skills_dir()))


def _install_one(drive_root: pathlib.Path, item_id: str, rows: List[Dict[str, Any]],
                 repo_dir: Optional[pathlib.Path]) -> Dict[str, Any]:
    from ouroboros.config import get_starter_skills_turn_on
    from ouroboros.skill_source_install import SkillInstallError, install_item, run_in_lane, start_review

    try:
        result = run_in_lane(drive_root, "install", item_id, lambda: install_item(
            drive_root, item_id, actor="owner", reason="starter skills",
            turn_on_when_ready=get_starter_skills_turn_on(),
            rows=rows, repo_dir=repo_dir), message=f"Installing starter skill {item_id}")
    except SkillInstallError as exc:
        status = "pending" if exc.code in _RETRY_CODES else "skipped" if exc.code == "name_taken" else "failed"
        return {"status": status, "code": exc.code, "error": str(exc)[:300]}
    except Exception as exc:
        return {"status": "pending", "code": "error", "error": f"{type(exc).__name__}: {exc}"[:300]}
    if result.get("needs_review"):
        start_review(drive_root, result["name"], repo_dir=repo_dir)
    return {"status": "installed", "name": result.get("name", ""), "basis": result.get("basis", "")}


def run_starter(drive_root: pathlib.Path, ids: Optional[List[str]], *, origin: str,
                repo_dir: Optional[pathlib.Path] = None) -> Dict[str, Any]:
    """Install ``ids`` once (blocking; callers run it in the background). Returns the marker.

    ``None`` means the configured starter list, resolved after the git-hosted sources were read: while
    a source cannot be read its recommended skills are unknown, so the run stays pending and a later
    start resolves the list again (installed items are kept).
    """
    from ouroboros.config import load_settings
    from ouroboros.skill_source_catalog import read_snapshot, refresh_source
    from ouroboros.skill_sources import load_sources, source_by_id

    with _LOCK:
        rows = load_sources()
        settings = load_settings()
        defaults = ids is None and settings.get(STARTER_SETTING, "") == ""
        builtin_rows = [row for row in rows if row["kind"] == "builtin"]
        chosen = list(ids) if ids is not None else starter_ids(drive_root, settings, builtin_rows, repo_dir=repo_dir)
        marker = _write_marker(drive_root, state="running", origin=origin, requested=chosen,
                               resolve_default=ids is None)
        _record_presented_defaults(drive_root, repo_dir)
        results = dict(marker.get("results") or {})
        def install(item_id: str) -> None:
            if (results.get(item_id) or {}).get("status") not in {"installed", "skipped", "failed"}:
                results[item_id] = _install_one(drive_root, item_id, rows, repo_dir)
                _write_marker(drive_root, results=results)

        for item_id in chosen:
            if any(item_id.startswith(row["id"] + "/") for row in builtin_rows):
                install(item_id)
        fetched: Dict[str, bool] = {}
        for row in rows:
            if row["kind"] in {"hub", "git"} and (defaults or any(i.startswith(row["id"] + "/") for i in chosen)):
                refresh_source(drive_root, row)
                fetched[row["id"]] = read_snapshot(drive_root, row) is not None
        unread = [row["id"] for row in rows if fetched.get(row["id"]) is False]
        if defaults:
            chosen = starter_ids(drive_root, settings, rows, repo_dir=repo_dir)
        _write_marker(drive_root, requested=chosen)
        for item_id in chosen:
            if (results.get(item_id) or {}).get("status") in {"installed", "skipped", "failed"}:
                continue
            row = source_by_id(rows, item_id.partition("/")[0])
            if row is not None and row["id"] in unread:
                results[item_id] = {"status": "pending", "code": "source_unread",
                                    "error": f"{row['name']} could not be read yet"}
                continue
            install(item_id)
        pending = [item for item, value in results.items() if value.get("status") == "pending"]
        waiting = bool(pending) or (defaults and bool(unread))
        return _write_marker(drive_root, state="pending" if waiting else "done", results=results)


def schedule_starter(drive_root: pathlib.Path, ids: Optional[List[str]], *, origin: str,
                     repo_dir: Optional[pathlib.Path] = None) -> threading.Thread:
    def _run() -> None:
        try:
            run_starter(drive_root, ids, origin=origin, repo_dir=repo_dir)
        except Exception:
            log.warning("Starter skill installation failed", exc_info=True)

    thread = threading.Thread(target=_run, daemon=True, name="starter-skills")
    with _THREADS_LOCK:
        _THREADS.difference_update([old for old in _THREADS if not old.is_alive()])
        _THREADS.add(thread)
        try:
            thread.start()
        except Exception:
            _THREADS.discard(thread)
            raise
    return thread


def join_starters(timeout: float = 60.0) -> bool:
    """Join scheduled starters within one deadline; return whether this owner is idle."""
    deadline = time.monotonic() + timeout
    with _THREADS_LOCK:
        threads = list(_THREADS)
    for thread in threads:
        thread.join(max(0.0, deadline - time.monotonic()))
    with _THREADS_LOCK:
        _THREADS.difference_update([thread for thread in _THREADS if not thread.is_alive()])
        return not _THREADS


def _existing_install(drive_root: pathlib.Path, settings: Dict[str, Any]) -> bool:
    """Evidence this data root ran before starter skills existed (an upgrade)."""
    root = pathlib.Path(drive_root)
    if str(settings.get("OUROBOROS_ONBOARDING_COMPLETED_AT") or "").strip():
        return True
    if (root / "skills" / "native" / ".bootstrap-seed-complete").exists():
        return True
    state_skills = root / "state" / "skills"
    return state_skills.is_dir() and any(state_skills.iterdir())


def classify_first_start(drive_root: pathlib.Path, settings: Dict[str, Any]) -> str:
    """Record whether this data root is new or an upgrade; call before the built-in startup work. Never raises."""
    try:
        state = str(read_marker(drive_root).get("state") or "")
        if state:
            return state
        state = "upgrade" if _existing_install(drive_root, settings) else "fresh"
        _write_marker(drive_root, state=state)
        return state
    except Exception:
        log.warning("Starter skill first-start classification failed", exc_info=True)
        return "error"


def at_startup(drive_root: pathlib.Path, settings: Dict[str, Any], *, provider_ready: bool,
               repo_dir: Optional[pathlib.Path] = None) -> str:
    """Startup work of skill sources after the built-in startup work: convert sidecar-era install records, then run
    the starter installer for a new install that skips onboarding, or retry what could not be fetched.
    Never raises."""
    try:
        from ouroboros.skill_install_record import convert_legacy_records
        from ouroboros.skill_source_install import settle_pending_updates_at_startup

        convert_legacy_records(drive_root)
        threading.Thread(target=settle_pending_updates_at_startup, args=(drive_root,),
                         daemon=True, name="skill-update-settle").start()
        marker = read_marker(drive_root)
        state = str(marker.get("state") or "")
        onboarded = bool(str(settings.get("OUROBOROS_ONBOARDING_COMPLETED_AT") or "").strip())
        if state == "fresh" and provider_ready and not onboarded:
            schedule_starter(drive_root, None, origin="first_start", repo_dir=repo_dir)
            return "scheduled"
        if state in {"pending", "running"}:
            retry = None if marker.get("resolve_default") else list(marker.get("requested") or [])
            schedule_starter(drive_root, retry, origin=str(marker.get("origin") or "retry"), repo_dir=repo_dir)
            return "retrying"
        return state
    except Exception:
        log.warning("Starter skill startup work failed", exc_info=True)
        return "error"


def skills_startup(drive_root: pathlib.Path, settings: Dict[str, Any], *, provider_ready: bool) -> str:
    """The skills part of server startup: classify the data root (new install or upgrade) before the
    built-in source's startup work (identity conversion, re-seeding changed built-ins), then
    ``at_startup``."""
    from ouroboros.launcher_bootstrap import ensure_data_skills_seeded

    classify_first_start(drive_root, settings)
    ensure_data_skills_seeded()
    return at_startup(drive_root, settings, provider_ready=provider_ready)


def after_onboarding(drive_root: pathlib.Path, ids: Optional[List[str]], *,
                     repo_dir: Optional[pathlib.Path] = None) -> str:
    """Run the installer for the owner's ticks once onboarding completed (``None`` = the default list)."""
    state = str(read_marker(drive_root).get("state") or "")
    if state in {"done", "running", "pending", "upgrade"}:
        return state
    if ids is not None and not ids:
        _write_marker(drive_root, state="done", origin="onboarding", requested=[], results={})
        _record_presented_defaults(drive_root, repo_dir)
        return "done"
    schedule_starter(drive_root, list(ids) if ids is not None else None, origin="onboarding", repo_dir=repo_dir)
    return "scheduled"


def onboarding_skill_choices(body: Dict[str, Any], settings: Dict[str, Any]) -> Dict[str, str]:
    """Settings the wizard's starter and hub-trust choices add to the one completion write.

    ``skill_source_trust`` ({hub row id: trusted|review}) sets and confirms those rows, which writes the
    source list; ``starter_skills`` is checked here and used after the write. A malformed choice raises
    ``ValueError`` so the completion saves nothing.
    """
    starter = body.get("starter_skills")
    if starter is not None and (not isinstance(starter, list) or not all(isinstance(i, str) for i in starter)):
        raise ValueError("starter_skills must be a list of catalog ids.")
    trust = body.get("skill_source_trust")
    if trust in (None, {}):
        return {}
    if not isinstance(trust, dict) or any(value not in ("trusted", "review") for value in trust.values()):
        raise ValueError("skill_source_trust must map source ids to trusted or review.")
    from ouroboros.skill_sources import SOURCES_SETTING, load_sources, with_changed_source

    rows = load_sources(dict(settings))
    for source_id, value in trust.items():
        rows = with_changed_source(rows, str(source_id), trust=value, confirmed=True)
    return {SOURCES_SETTING: json.dumps(rows, ensure_ascii=False, separators=(",", ":"))}


__all__ = [
    "MARKER_FILENAME", "STARTER_SETTING", "after_onboarding", "at_startup", "classify_first_start", "default_starter_ids",
    "join_starters", "onboarding_skill_choices", "read_marker",
    "run_starter", "schedule_starter", "skills_startup", "starter_groups", "starter_ids",
]
