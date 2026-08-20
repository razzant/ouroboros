"""The one-time legacy usage-telemetry import (v7 L-C2 split).

The resumable import of pre-ledger usage telemetry (``llm_usage`` events and
the ``state.json`` cost baseline) into the append-only attempt ledger: source
snapshot and read-only archive, deduplicated candidate rows, the metadata/delta
reconciliation against the state baseline, and the completed-import watermark.
Extracted from usage_accounting.py; usage_accounting re-exports every name, so
historical imports and monkeypatch targets keep working."""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
from typing import Any, Dict, Optional, Tuple

from ouroboros.usage_ledger import (
    UsageAccountingError,
    _append_rows_locked,
    _drive_root,
    _named_lock,
    _number,
    _write_bytes_atomic_fsync,
)
from ouroboros.utils import append_jsonl, atomic_write_json, utc_now_iso

IMPORT_REL = pathlib.Path("state/usage_import_watermark.json")


def _usage():
    """The parent usage_accounting module, read at call time.

    The accounting members stay monkeypatch-addressable at their historical
    ``ouroboros.usage_accounting`` bindings (tests rebind them there), so this
    leaf resolves every cross-reference through the module at each call instead
    of freezing whatever object a from-import saw at import time.
    """
    from ouroboros import usage_accounting

    return usage_accounting


def _legacy_snapshot(root: pathlib.Path) -> Tuple[list[Dict[str, Any]], Dict[str, Any], Dict[str, str]]:
    events_path = root / "logs" / "events.jsonl"
    state_path = root / "state" / "state.json"
    settings_path = pathlib.Path(os.environ.get("OUROBOROS_SETTINGS_PATH") or root / "settings.json")
    sources = {"events.jsonl": events_path, "state.json": state_path}
    snapshots: Dict[str, bytes] = {}
    for name, path in sources.items():
        try:
            snapshots[name] = path.read_bytes()
        except FileNotFoundError:
            continue
        except OSError as exc:
            raise UsageAccountingError(f"cannot snapshot legacy usage source {path}: {exc}") from exc
    hashes = {name: hashlib.sha256(snapshots[name]).hexdigest() if name in snapshots else "" for name in sources}
    # Settings are owner-secret state: prove non-mutation by hash, but never copy
    # their contents into the usage archive.
    try:
        hashes["settings.json"] = hashlib.sha256(settings_path.read_bytes()).hexdigest()
    except FileNotFoundError:
        hashes["settings.json"] = ""
    except OSError as exc:
        raise UsageAccountingError(f"cannot hash settings file {settings_path}: {exc}") from exc
    rows: list[Dict[str, Any]] = []
    try:
        event_text = snapshots.get("events.jsonl", b"").decode("utf-8")
        for line_no, line in enumerate(event_text.splitlines(), 1):
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict) and value.get("type") == "llm_usage":
                rows.append({**value, "_legacy_line": line_no})
    except UnicodeDecodeError:
        pass
    try:
        state = json.loads(snapshots.get("state.json", b"{}").decode("utf-8"))
        if not isinstance(state, dict):
            state = {}
    except (UnicodeDecodeError, json.JSONDecodeError):
        state = {}

    combined = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode("utf-8")).hexdigest()[:16]
    archive = root / "archive" / "usage_import" / combined
    archive.mkdir(parents=True, exist_ok=True)
    for name, payload in snapshots.items():
        target = archive / name
        if target.exists():
            if target.read_bytes() != payload:
                raise UsageAccountingError(f"legacy usage archive mismatch: {target}")
        else:
            _write_bytes_atomic_fsync(target, payload)
            try:
                target.chmod(0o400)
            except OSError:
                pass
    atomic_write_json(archive / "sha256.json", hashes, trailing_newline=True, fsync=True)
    return rows, state, hashes


def ensure_legacy_imported(
    drive_root: Optional[pathlib.Path] = None,
) -> Dict[str, Any]:
    """One resumable import of legacy usage telemetry and the state cost delta."""
    root = _drive_root(drive_root)
    completed = _completed_import_watermark(root)
    if completed is not None:
        return completed
    # Separate from the hot budget lock: source snapshot/archive may do I/O,
    # while concurrent startup importers still serialize on one generation.
    with _named_lock(root, "usage_import.lock", timeout_sec=60.0, stale_sec=600.0):
        return _ensure_legacy_imported_locked(root)


def _completed_import_watermark(root: pathlib.Path) -> Optional[Dict[str, Any]]:
    try:
        value = json.loads((root / IMPORT_REL).read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) and value.get("completed") else None


def _ensure_legacy_imported_locked(
    root: pathlib.Path,
) -> Dict[str, Any]:
    watermark = root / IMPORT_REL
    existing = _completed_import_watermark(root)
    if existing is not None:
        return existing

    legacy_rows, state, hashes = _usage()._legacy_snapshot(root)
    baseline_source = "state.json"
    candidates: list[Dict[str, Any]] = []
    seen_fingerprints: set[str] = set()
    imported_cost = 0.0
    usage_count = 0
    for event in legacy_rows:
        line_no = int(event.pop("_legacy_line", 0) or 0)
        legacy_usage = event.get("usage") if isinstance(event.get("usage"), dict) else {}
        fingerprint = hashlib.sha256(
            json.dumps(event, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
        ).hexdigest()
        if fingerprint in seen_fingerprints:
            continue
        seen_fingerprints.add(fingerprint)
        task_id = str(event.get("task_id") or "")
        root_task_id = str(event.get("root_task_id") or task_id)
        raw_cost = event.get("cost")
        if raw_cost is None:
            raw_cost = legacy_usage.get("cost", legacy_usage.get("total_cost"))
        cost = _number(raw_cost)

        def legacy_int(field: str, *aliases: str) -> int:
            for candidate in (field, *aliases):
                value = event.get(candidate)
                if value in (None, ""):
                    value = legacy_usage.get(candidate)
                try:
                    return max(0, int(float(value or 0)))
                except (TypeError, ValueError):
                    continue
            return 0

        prompt = legacy_int("prompt_tokens", "input_tokens")
        completion = legacy_int("completion_tokens", "output_tokens")
        provider = str(event.get("provider") or event.get("api_key_type") or "unknown")
        if cost == 0 and (prompt or completion) and provider != "local":
            cost = None  # legacy zero may mean unknown pricing, never "free"
        usage_count += 1
        if cost is not None:
            imported_cost += cost
        candidates.append(
            {
                "kind": "legacy_usage",
                "attempt_id": f"legacy-{fingerprint[:24]}",
                "state": "settled",
                "model": str(event.get("model") or ""),
                "provider": provider,
                "cost_usd": cost,
                "cost_final": bool(cost is not None and not event.get("cost_estimated")),
                "reservation_upper_bound_usd": None,
                "prompt_tokens": prompt,
                "completion_tokens": completion,
                "cached_tokens": legacy_int("cached_tokens", "cache_read_input_tokens"),
                "cache_write_tokens": legacy_int("cache_write_tokens", "cache_creation_input_tokens"),
                "prompt_cache_ttl": str(
                    event.get("prompt_cache_ttl") or legacy_usage.get("prompt_cache_ttl") or ""
                ),
                "task_id": task_id,
                "root_task_id": root_task_id,
                "parent_task_id": str(event.get("parent_task_id") or ""),
                "category": str(event.get("category") or "legacy"),
                "source": "legacy_llm_usage",
                "legacy_line": line_no,
            }
        )
    legacy_calls = max(0, int(state.get("spent_calls") or state.get("calls") or 0))
    metadata_count = max(0, legacy_calls - usage_count)
    if metadata_count:
        identity = hashlib.sha256(
            f"legacy-metadata:{metadata_count}:{hashes.get('state.json', '')}".encode()
        ).hexdigest()
        candidates.append(
            {
                "kind": "legacy_metadata",
                "attempt_id": f"legacy-{identity[:24]}",
                "state": "unresolved",
                "model": "",
                "provider": "legacy",
                "reservation_upper_bound_usd": None,
                "ambiguous_call_count": metadata_count,
                "task_id": "",
                "root_task_id": "",
                "parent_task_id": "",
                "category": "legacy",
                "source": "legacy_state_call_delta",
            }
        )
    state_spent = _number(state.get("spent_usd")) or 0.0
    delta = round(max(0.0, state_spent - imported_cost), 6)
    if delta:
        identity = hashlib.sha256(f"legacy-delta:{delta:.6f}:{hashes.get('state.json', '')}".encode()).hexdigest()
        candidates.append(
            {
                "kind": "legacy_delta",
                "attempt_id": f"legacy-{identity[:24]}",
                "state": "settled",
                "model": "",
                "provider": "legacy",
                "cost_usd": delta,
                "cost_final": False,
                "reservation_upper_bound_usd": None,
                "task_id": "",
                "root_task_id": "",
                "parent_task_id": "",
                "category": "legacy",
                "source": "legacy_state_delta",
            }
        )

    with _usage()._locked(root):
        current_watermark = _completed_import_watermark(root)
        if current_watermark is not None:
            return current_watermark
        records = _usage()._read_records_locked(root)
        existing_ids = {str(row.get("attempt_id") or "") for row in records}
        missing = [row for row in candidates if row["attempt_id"] not in existing_ids]
        _append_rows_locked(root, records, missing)
        result = {
            "completed": True,
            "completed_at": utc_now_iso(),
            "source_sha256": hashes,
            "legacy_baseline_source": baseline_source,
            "legacy_baseline_spent_usd": state_spent,
            "legacy_baseline_spent_calls": legacy_calls,
            "legacy_usage_count": usage_count,
            "legacy_metadata_count": metadata_count,
            "legacy_delta_usd": delta,
            # The legacy schema has no trustworthy typed test/operator bit.
            # Never invent exclusions from names, task ids, or source strings.
            "quarantined_test_operator_rows": 0,
            "test_operator_quarantine_policy": "typed_evidence_only_no_inference",
            "events_exceed_state_calls": max(0, usage_count - legacy_calls),
            "events_exceed_state_usd": round(max(0.0, imported_cost - state_spent), 6),
            "rows_appended": len(missing),
        }
        atomic_write_json(watermark, result, trailing_newline=True, fsync=True)
    append_jsonl(root / "logs" / "events.jsonl", {"type": "usage_import_completed", **result})
    return result
