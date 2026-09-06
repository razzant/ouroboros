"""Secret-loading helpers that never print credential values."""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
from typing import Any, Iterable


SECRET_KEYS = (
    "OPENROUTER_API_KEY",
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "MINIMAX_API_KEY",
    "DEEPSEEK_API_KEY",
    "GITHUB_TOKEN",
)


def settings_path(default_home: pathlib.Path | None = None) -> pathlib.Path:
    home = default_home or pathlib.Path(__file__).resolve().parents[4]
    return pathlib.Path(os.environ.get("OUROBOROS_SETTINGS_PATH") or home / "data" / "settings.json")


def load_secret_env(path: pathlib.Path | None = None) -> dict[str, str]:
    values: dict[str, str] = {}
    for key in SECRET_KEYS:
        value = os.environ.get(key)
        if value:
            values[key] = value
    settings_file = path or settings_path()
    try:
        loaded = json.loads(settings_file.read_text(encoding="utf-8"))
    except Exception:
        loaded = {}
    if isinstance(loaded, dict):
        for key in SECRET_KEYS:
            value = loaded.get(key)
            if value and key not in values:
                values[key] = str(value)
    return values


def redacted_env_summary(env: dict[str, str], keys: Iterable[str] | None = None) -> dict[str, bool]:
    return {key: bool(env.get(key)) for key in (SECRET_KEYS if keys is None else keys)}


def credential_fingerprint(value: Any) -> str:
    """Stable, non-reversible identity for a credential value — NEVER the value itself.

    A truncated SHA-256 over a high-entropy API key is not brute-forceable, and it is what
    lets an auditor answer the only question that matters across two runs: was this the SAME
    key?  Empty/absent values fingerprint to the empty string, not to the hash of ""."""
    text = str(value or "")
    if not text:
        return ""
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def isolated_credential_grants(
    cfg: dict,
    *,
    include_claude_sdk_defaults: bool = True,
) -> dict:
    """Describe, by FINGERPRINT and never by value, which provider credentials a benchmark
    settings mapping actually carries — and which its declared model slots called for.

    ``planned_keys`` is the derivation (``ouroboros.provider_models.provider_credential_plan``,
    which reads the same prefix->provider registry ``llm._resolve_remote_target`` routes on);
    ``granted`` is the truth about the file. The two are reported separately on purpose: a
    caller may hand an explicit credential override the slots did not ask for, and an auditor
    must SEE that rather than infer it. Prevention without evidence is half a fix."""
    from ouroboros.provider_models import ALL_PROVIDER_CREDENTIAL_KEYS, provider_credential_plan

    settings = cfg or {}
    plan = provider_credential_plan(
        settings,
        include_claude_sdk_defaults=include_claude_sdk_defaults,
    )
    present = {
        key: settings.get(key)
        for key in sorted(ALL_PROVIDER_CREDENTIAL_KEYS)
        if str(settings.get(key) or "").strip()
    }
    return {
        "schema": "ouroboros.benchmark.provider_credentials.v1",
        "declared_model_slots": plan["declared_model_slots"],
        "providers": plan["providers"],
        "planned_keys": plan["planned_keys"],
        "fail_open": plan["fail_open"],
        "granted": credential_disclosure(present, sorted(present)),
    }


def credential_disclosure(env: dict[str, Any], keys: Iterable[str] | None = None) -> dict[str, dict[str, Any]]:
    """Fingerprint-level extension of ``redacted_env_summary``: ``{key: {present, fingerprint}}``.

    Same mechanism, one rung more auditable — a bare ``True`` cannot distinguish "the run
    reached the declared bucket" from "the run reached some other key that happened to be in
    the live settings file"."""
    names = sorted(env) if keys is None else list(keys)
    return {
        str(key): {
            "present": bool(str(env.get(key) or "").strip()),
            "fingerprint": credential_fingerprint(env.get(key)),
        }
        for key in names
    }
