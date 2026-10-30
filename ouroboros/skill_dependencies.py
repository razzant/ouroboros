"""Shared dependency-spec resolution and installed readiness for skill payloads."""

from __future__ import annotations

import logging
import pathlib
from typing import Any, Dict, List

from ouroboros.marketplace.install_specs import normalize_install_specs
from ouroboros.utils import read_json_dict

log = logging.getLogger(__name__)


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


def skill_deps_not_ready(drive_root: pathlib.Path, loaded: Any, *, strict: bool = False) -> tuple[str, str]:
    """``(status, reason)`` of unready dependencies; ``("", "")`` when ready.

    A probe failure reads as ready for execution readers; ``strict`` re-raises it so a
    caller selecting startup package work cannot mistake it for ready.
    """
    try:
        from ouroboros.marketplace.install_specs import install_specs_hash as _specs_hash
        from ouroboros.marketplace.isolated_deps import read_deps_state

        auto_specs = auto_install_specs_for_skill(drive_root, loaded)
        if not auto_specs:
            return "", ""
        deps_state = read_deps_state(drive_root, loaded.name, loaded.skill_dir)
        deps_status = str(deps_state.get("status") or "pending")
        if deps_status != "installed":
            return deps_status, "status"
        if str(deps_state.get("specs_hash") or "") != _specs_hash(auto_specs):
            return deps_status, "fingerprint"
        return "", ""
    except Exception:
        if strict:
            raise
        log.debug("skill deps readiness probe failed", exc_info=True)
        return "", ""


def unusable_python_env(drive_root: pathlib.Path, loaded: Any) -> str:
    """The binding (``moved``, ``absent``, ...) when ``loaded`` declares Python packages that apply on this host
    and its env python cannot run them on this interpreter, else ``""``: such a skill never runs on another
    interpreter (``skill_exec`` and the process-tool resolver refuse); a skill whose Python packages are all
    for other platforms, or that has none, is unaffected (the installer skips the former, ``spec_applies_here``)."""
    from ouroboros.marketplace.isolated_deps import python_env_binding, spec_applies_here

    if not any(str(spec.get("kind") or "").strip().lower() in {"pip", "pipx", "uv"} and spec_applies_here(spec)
               for spec in auto_install_specs_for_skill(drive_root, loaded) if isinstance(spec, dict)):
        return ""
    binding = python_env_binding(loaded.skill_dir)
    return "" if binding == "current" else binding


# One owner sentence per typed dependency code (the toast and the card prefix the skill name).
_DEPENDENCY_SENTENCES = {
    "deps_missing": "Its packages are not installed.",
    "deps_failed": "Its packages did not install.",
    "deps_stale": "Its packages are out of date.",
    "deps_declaration_desync": "Its package list does not match its files.",
    "dependencies_failed": "Its packages did not install.",
}


def dependency_sentence(code: str) -> str:
    """The owner's sentence for a dependency code: shared by the enable refusal, extension
    liveness and the extension loader so one state has one wording everywhere."""
    return _DEPENDENCY_SENTENCES.get(str(code or ""), _DEPENDENCY_SENTENCES["deps_missing"])


def ensure_skill_dependencies(drive_root: pathlib.Path, loaded: Any, *, progress: Any = None) -> tuple[str, str]:
    """Establish the declared packages of ``loaded`` whenever a runnable skill is requested.

    The one dependency owner: whatever admits the exact bytes (``gate_for(content_hash)``)
    admits the declarations they carry, so enable, review end, install acceptance, author
    finish and the startup pass all call this and nothing else. ``(status, detail)``:
    ``not_required`` (no auto specs), ``installed`` (already recorded for these specs, or
    installed now), ``not_admitted`` (the gate does not admit the bytes: nothing written),
    ``failed`` (the installer's text in ``detail``; ``deps.json`` keeps it, the verdict is
    untouched, the next call retries). An environment left on another host's Python
    (``python_env`` of the reader) is re-pointed without pip when only its base moved and
    rebuilt by the installer for another major.minor. Idempotent; never takes the lifecycle lane.
    """
    try:
        from ouroboros.marketplace.install_specs import install_specs_hash
        from ouroboros.marketplace.isolated_deps import install_isolated_dependencies, read_deps_state, rebind_python_env

        auto_specs = auto_install_specs_for_skill(drive_root, loaded)
        if not auto_specs:
            return "not_required", ""

        def recorded(state: Dict[str, Any]) -> bool:
            return str(state.get("status") or "") == "installed" and state.get("specs_hash") == install_specs_hash(auto_specs)

        state = read_deps_state(drive_root, loaded.name, loaded.skill_dir)
        if recorded(state):
            if read_deps_state(drive_root, loaded.name).get("status") != "installed":
                # A runtime record corroborated the live environment; finish its interrupted publication.
                from ouroboros.marketplace.isolated_deps import DEPS_STATE_FILENAME
                from ouroboros.skill_loader import skill_state_dir
                from ouroboros.utils import atomic_write_json
                atomic_write_json(skill_state_dir(drive_root, loaded.name) / DEPS_STATE_FILENAME, state, trailing_newline=True)
            return "installed", ""
        if not loaded.review.gate_for(loaded.content_hash)["executable_review"]:
            return "not_admitted", ""
        if state.get("python_env") == "moved":
            rebind_python_env(loaded.skill_dir)
            if recorded(read_deps_state(drive_root, loaded.name, loaded.skill_dir)):
                return "installed", ""
        if progress is not None:
            progress("Installing packages…")
        install_isolated_dependencies(drive_root, loaded.name, loaded.skill_dir, auto_specs)
        return "installed", ""
    except Exception as exc:
        log.debug("skill dependency step failed for %s", getattr(loaded, "name", "?"), exc_info=True)
        return "failed", f"{type(exc).__name__}: {exc}"
