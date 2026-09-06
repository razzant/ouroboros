"""Shared dependency-spec resolution for skill payloads."""

from __future__ import annotations

import pathlib
from typing import Any, Dict, List

from ouroboros.marketplace.install_specs import normalize_install_specs
from ouroboros.utils import read_json_dict


def _coerce_dependency_specs(raw: Any) -> Any:
    if raw in (None, "", [], {}):
        return []
    if isinstance(raw, list):
        # Bare string lists mean Python packages by convention. Object lists
        # are already OpenClaw/Ouroboros install specs.
        if all(isinstance(item, str) for item in raw):
            return [{"kind": "pip", "package": item} for item in raw]
        return raw
    if isinstance(raw, dict):
        out: List[Dict[str, Any]] = []
        for key, kind in (
            ("python", "pip"),
            ("pip", "pip"),
            ("npm", "npm"),
            ("node", "npm"),
        ):
            value = raw.get(key)
            if value in (None, "", [], {}):
                continue
            items = value if isinstance(value, list) else [value]
            for item in items:
                if isinstance(item, dict):
                    spec = dict(item)
                    spec.setdefault("kind", kind)
                    out.append(spec)
                else:
                    out.append({"kind": kind, "package": str(item)})
        if out:
            return out
        if raw.get("kind"):
            return raw
    return raw


def normalize_declared_dependency_specs(raw: Any) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[str]]:
    return normalize_install_specs(_coerce_dependency_specs(raw))


def _manifest_declared_raw(manifest: Any) -> Any:
    extras = dict(getattr(manifest, "raw_extra", {}) or {})
    raw = extras.get("install_specs")
    if raw in (None, "", [], {}):
        raw = extras.get("install")
    if raw in (None, "", [], {}):
        raw = extras.get("dependencies")
    return raw


def _manifest_install_specs(manifest: Any) -> List[Dict[str, Any]]:
    auto, _manual, _warnings = normalize_declared_dependency_specs(_manifest_declared_raw(manifest))
    return auto


def manual_install_specs_for_skill(loaded: Any) -> tuple[List[Dict[str, Any]], List[str]]:
    """Return ``(manual_specs, warnings)`` declared in the skill's manifest.

    G3 (capinv-447): the auto-spec resolver used to DROP the manual portion, so
    a skill whose only dependencies were manual reported an empty dependency
    list and READY. Manual specs only ever come from the reviewed manifest
    declaration (provenance/sidecar records carry the auto list alone).
    """
    _auto, manual, warnings = normalize_declared_dependency_specs(
        _manifest_declared_raw(getattr(loaded, "manifest", None))
    )
    return manual, warnings


def _payload_sidecar_specs(skill_dir: pathlib.Path) -> List[Dict[str, Any]]:
    for filename in (".ouroboroshub.json", ".clawhub.json"):
        path = pathlib.Path(skill_dir) / filename
        record = read_json_dict(path)
        if not record:
            continue
        auto = list((record.get("install_specs") or {}).get("auto") or [])
        if auto:
            return auto
    return []


def payload_declared_install_specs(loaded: Any) -> List[Dict[str, Any]]:
    """Auto specs declared by HASH-COVERED payload carriers only (6.2=A).

    The review content hash covers the payload sidecars and the manifest, but
    NOT the state-plane provenance record ``auto_install_specs_for_skill``
    prefers for ClawHub payloads. This projection is the declarative
    dependency fingerprint: a new declared name changes the payload bytes and
    therefore forces re-review, while the installed ``.ouroboros_env`` bytes
    stay outside the hash by design.
    """
    sidecar = _payload_sidecar_specs(pathlib.Path(loaded.skill_dir))
    if sidecar:
        return sidecar
    return _manifest_install_specs(getattr(loaded, "manifest", None))


def declared_dependency_names(specs: Any) -> frozenset[str]:
    """Canonical ``kind:package`` name set of a declared spec list."""
    return frozenset(
        f"{str(item.get('kind') or '').strip().lower()}:{str(item.get('package') or '').strip()}"
        for item in (specs or [])
        if isinstance(item, dict)
    )


def auto_install_specs_for_skill(drive_root: pathlib.Path, loaded: Any) -> List[Dict[str, Any]]:
    """Return normalized auto-install specs declared for ``loaded``.

    ClawHub provenance remains authoritative for ClawHub-sourced skills only:
    the state-plane ``clawhub.json`` survives payload transitions, so reading
    it source-blind would let a stale record dictate dependencies for a skill
    that now lives in another bucket. Other sources declare dependencies in
    their reviewed manifest or, for official catalog installs, in a payload
    sidecar.
    """

    if str(getattr(loaded, "source", "") or "") == "clawhub":
        try:
            from ouroboros.marketplace.provenance import read_provenance

            prov = read_provenance(drive_root, loaded.name) or {}
            auto = list((prov.get("install_specs") or {}).get("auto") or [])
            if auto:
                return auto
        except Exception:
            pass

    sidecar = _payload_sidecar_specs(pathlib.Path(loaded.skill_dir))
    if sidecar:
        return sidecar

    return _manifest_install_specs(getattr(loaded, "manifest", None))
