"""The engine daemon runs on a Node this installation keeps, not on the app's own copy.

The daemon outlives the app that started it and spawns its children with ``process.execPath`` (browser MCP
launcher, delegation belt, harness CLI, setup runners, its own respawn). A Node inside the app bundle is gone
once the app is moved, replaced or deleted, or its AppImage mount ends, and every such spawn then fails until the
daemon restarts. ``claudexor_runtime`` therefore ranks an exact managed Node, complete or Node-only, before the
bundled one. When none exists, its foreground install owner (``_ensure_node`` under the install lock, never the
read-only resolver) calls :func:`seed_node` once: the bundle's exact Node bytes are copied into a new immutable
``state/cx/node`` generation beside the downloaded ones, with no link back to the app. The record states what the
copy is (``source: bundle_seed``, its sha256, the app version) and never the ``archive_*`` fields that only a
verified download earns, so a seed serves the daemon and never the npm toolchain. If the copy fails, the bundled
Node stays in use and the warning names it. Retention keeps Node generations like the downloaded ones
(docs/PERSISTENCE.md).
"""
from __future__ import annotations

import hashlib
import logging
import os
import pathlib
import re
import shutil
import uuid
from typing import Any, Mapping

log = logging.getLogger(__name__)

SEED_SOURCE = "bundle_seed"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def bundled_node(version: str) -> str:
    """The first bundled Node that runs as exactly ``version``, or ``""``."""
    from ouroboros.platform_layer import bundled_resource_bases, embedded_node_candidates, probe_node_version

    for base in bundled_resource_bases():
        for candidate in embedded_node_candidates(base):
            try:
                if candidate.is_file() and probe_node_version(str(candidate)) == version:
                    return str(candidate)
            except (OSError, RuntimeError):
                continue
    return ""


def is_seed_record(raw: Mapping[str, Any], version: str, platform_key: str) -> bool:
    """A Node-only seed record for this version and platform; it never carries an ``archive_*`` field."""
    return (raw.get("schema_version") == 1 and raw.get("source") == SEED_SOURCE
            and raw.get("version") == version and raw.get("platform") == platform_key
            and bool(_SHA256.fullmatch(str(raw.get("binary_sha256") or "")))
            and not any(str(key).startswith("archive_") for key in raw))


def _sha256(path: "str | pathlib.Path") -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def seed_node(legacy: pathlib.Path, version: str, platform_key: str, record_name: str) -> str:
    """Copy the exact bundled Node into a generation beside ``legacy``, its record named ``record_name`` (the
    runtime's own reader's file), and return the copy.

    Returns the bundled path when that copy fails or this platform has no managed Node layout, and
    ``""`` when the bundle has no exact Node (the caller downloads one)."""
    source = bundled_node(version)
    if not source or not platform_key:
        return source
    from ouroboros.platform_layer import embedded_node_candidates, probe_node_version
    from ouroboros.utils import atomic_write_json, replace_atomic

    nonce = uuid.uuid4().hex[:8]
    staging = legacy.parent / f".tmp-{os.getpid()}-{nonce}"
    try:
        node = embedded_node_candidates(staging)[0]
        node.parent.mkdir(parents=True)
        shutil.copyfile(source, node)  # the bytes themselves: a link would die with the app
        node.chmod(0o755)
        digest = _sha256(node)
        if digest != _sha256(source) or probe_node_version(str(node)) != version:
            raise OSError(f"the copy does not run as the bundled Node {version}")
        atomic_write_json(staging / record_name, {
            "schema_version": 1, "version": version, "platform": platform_key, "source": SEED_SOURCE,
            "binary_sha256": digest, "app_version": os.environ.get("OUROBOROS_APP_VERSION", "").strip(),
        }, trailing_newline=True, fsync=True)
        root = legacy.parent / f"{legacy.name}.g-{digest[:12]}-{nonce}"
        replace_atomic(staging, root)
        return str(embedded_node_candidates(root)[0])
    except Exception as exc:  # a full disk or a vanished bundle: the bundled Node keeps serving, and says so
        log.warning("Managed Node seed failed; the engine daemon runs on the bundled Node %s: %s: %s",
                    source, type(exc).__name__, exc)
        return source
    finally:
        shutil.rmtree(staging, ignore_errors=True)
