"""Validate the engine's same-spawn Cursor catalog evidence without resolving a model.

``attempts[].observed_model_catalog_match`` is absent/null or contains the
strings ``label``, ``id``, ``source: account_catalog`` and ``profile_id``.
``final_attempt_facts`` first binds the containing telemetry to a held run and
one unique final attempt. This additive match belongs to that attempt's first
native model observation; a later profile rotation cannot lend it authority.
No request, summary, alias table or post-run catalog lookup supplies a match.
"""
from __future__ import annotations

from typing import Any


def validated_catalog_match(value: Any, attempt: Any) -> dict | None:
    """A catalog ID supplements the raw ``model`` in a normalized observed attempt.

    The engine alone proves uniqueness in its account catalog. Validate its
    explicit source and exact label/profile binding, retaining absence for old
    engines or malformed evidence. Reused when reading a persisted receipt.
    """
    if not isinstance(value, dict) or not isinstance(attempt, dict):
        return None
    if any(not isinstance(value.get(key), str) or not value[key].strip()
           for key in ("label", "id", "source", "profile_id")):
        return None
    if (value["source"] != "account_catalog" or attempt.get("harness_id") != "cursor"
            or value["label"] != attempt.get("model")
            or value["profile_id"] != attempt.get("profile_id")):
        return None
    return {key: value[key] for key in ("label", "id", "source", "profile_id")}
