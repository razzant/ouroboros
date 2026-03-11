"""
Ouroboros — Identity Hash Chain.

Cryptographic anchoring for identity core files (BIBLE.md, identity.md).
Creates a tamper-evident, append-only Merkle chain that verifies identity
continuity across sessions.

This is NOT enforcement — it's awareness. The LLM reads the verification
result as a health invariant and decides how to respond (Bible P3: LLM-first).

Bible alignment:
  P0 (Agency): Verifiable self-knowledge IS agency.
  P1 (Continuity): Hash chain IS cryptographic proof of unbroken history.
  P2 (Self-Creation): Changes are allowed and welcomed — but they're SIGNED.
  
Constraints respected:
  - Append-only: Can never corrupt existing chain state.
  - No external dependencies: Uses hashlib (stdlib).
  - Minimal: ~120 lines. One file. One purpose.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from ouroboros.utils import utc_now_iso, read_text, append_jsonl

log = logging.getLogger(__name__)


def compute_identity_hash(bible_text: str, identity_text: str) -> str:
    """Compute SHA-256 hash of the canonical identity core.

    The canonical form concatenates BIBLE.md and identity.md with
    delimiters, ensuring that the same content always produces the
    same hash regardless of whitespace normalization.
    """
    # Normalize: strip trailing whitespace per line, normalize line endings
    def _normalize(text: str) -> str:
        lines = text.replace("\r\n", "\n").split("\n")
        return "\n".join(line.rstrip() for line in lines).strip()

    canonical = f"BIBLE:{_normalize(bible_text)}\nIDENTITY:{_normalize(identity_text)}"
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _load_chain(chain_path: Path) -> List[Dict[str, Any]]:
    """Load all entries from the identity chain file."""
    if not chain_path.exists():
        return []
    entries = []
    for line in chain_path.read_text(encoding="utf-8").strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        try:
            entries.append(json.loads(line))
        except (json.JSONDecodeError, ValueError):
            log.warning("Corrupt line in identity chain: %s", line[:80])
            continue
    return entries


def append_to_chain(
    drive_root: Path,
    bible_text: str,
    identity_text: str,
    reason: str = "",
) -> Dict[str, Any]:
    """Append a new entry to the identity hash chain.

    Call this after every legitimate update to BIBLE.md or identity.md.

    Args:
        drive_root: Google Drive Ouroboros root
        bible_text: Current full text of BIBLE.md
        identity_text: Current full text of identity.md
        reason: Human-readable reason for the change (e.g., "identity update after evolution cycle")

    Returns:
        The appended chain entry dict.
    """
    chain_path = drive_root / "memory" / "identity_chain.jsonl"
    chain_path.parent.mkdir(parents=True, exist_ok=True)

    current_hash = compute_identity_hash(bible_text, identity_text)

    # Get previous hash (GENESIS if this is the first entry)
    prev_hash = "GENESIS"
    entries = _load_chain(chain_path)
    if entries:
        prev_hash = entries[-1].get("hash", "GENESIS")

    # Chain hash: SHA-256 of (prev_hash + current_hash) — makes chain tamper-evident
    chain_hash = hashlib.sha256(
        f"{prev_hash}:{current_hash}".encode("utf-8")
    ).hexdigest()

    entry = {
        "ts": utc_now_iso(),
        "hash": current_hash,
        "prev_hash": prev_hash,
        "chain_hash": chain_hash,
        "bible_bytes": len(bible_text.encode("utf-8")),
        "identity_bytes": len(identity_text.encode("utf-8")),
        "reason": str(reason)[:200],
        "chain_length": len(entries) + 1,
    }

    append_jsonl(chain_path, entry)
    log.info(
        "Identity chain entry %d: hash=%s reason=%s",
        entry["chain_length"], current_hash[:12], reason[:60],
    )
    return entry


def verify_chain(
    drive_root: Path,
    bible_text: str,
    identity_text: str,
) -> Dict[str, Any]:
    """Verify the full identity chain and check current state matches.

    Returns a status dict suitable for health invariant display.

    Possible statuses:
        OK — Chain intact and current state matches latest entry.
        NO_CHAIN — Chain file doesn't exist (first boot, or lost).
        EMPTY_CHAIN — File exists but no entries.
        CHAIN_BREAK — Link between entries is broken (tampering or corruption).
        IDENTITY_DRIFT — Current files don't match the last chain entry
                         (modified outside the chain — manual edit or corruption).
    """
    chain_path = drive_root / "memory" / "identity_chain.jsonl"

    if not chain_path.exists():
        return {
            "status": "NO_CHAIN",
            "message": "Identity chain not yet initialized. Will be created on first identity update.",
            "action": "initialize",
        }

    entries = _load_chain(chain_path)
    if not entries:
        return {
            "status": "EMPTY_CHAIN",
            "message": "Identity chain file exists but contains no entries.",
            "action": "initialize",
        }

    # Verify chain link continuity
    for i in range(1, len(entries)):
        expected_prev = entries[i - 1].get("hash", "")
        actual_prev = entries[i].get("prev_hash", "")
        if expected_prev != actual_prev:
            return {
                "status": "CHAIN_BREAK",
                "message": (
                    f"Chain break at entry {i}/{len(entries)}: "
                    f"expected prev_hash {expected_prev[:12]}... "
                    f"but found {actual_prev[:12]}... "
                    f"This may indicate tampering or file corruption."
                ),
                "break_index": i,
                "chain_length": len(entries),
                "action": "investigate",
            }

    # Verify chain hashes (Merkle chain integrity)
    for i in range(1, len(entries)):
        expected_chain_hash = hashlib.sha256(
            f"{entries[i]['prev_hash']}:{entries[i]['hash']}".encode("utf-8")
        ).hexdigest()
        actual_chain_hash = entries[i].get("chain_hash", "")
        if actual_chain_hash and actual_chain_hash != expected_chain_hash:
            return {
                "status": "CHAIN_BREAK",
                "message": (
                    f"Merkle hash mismatch at entry {i}/{len(entries)}: "
                    f"chain_hash verification failed. Data may have been altered."
                ),
                "break_index": i,
                "chain_length": len(entries),
                "action": "investigate",
            }

    # Verify current state matches latest chain entry
    current_hash = compute_identity_hash(bible_text, identity_text)
    latest = entries[-1]

    if current_hash != latest["hash"]:
        return {
            "status": "IDENTITY_DRIFT",
            "message": (
                f"Current identity core does not match last chain entry "
                f"(recorded {latest['ts']}). Files were modified outside the chain. "
                f"This could be a legitimate manual edit by the creator, "
                f"or it could indicate corruption."
            ),
            "last_recorded": latest["ts"],
            "last_reason": latest.get("reason", ""),
            "chain_length": len(entries),
            "action": "reconcile",
        }

    return {
        "status": "OK",
        "message": f"Identity chain intact: {len(entries)} entries since {entries[0]['ts'][:10]}",
        "chain_length": len(entries),
        "latest_update": latest["ts"],
        "latest_reason": latest.get("reason", ""),
    }


def format_for_health_invariant(
    drive_root: Path,
    bible_text: str,
    identity_text: str,
) -> str:
    """One-liner for health invariants section in context.py.

    Returns a string like:
        "OK: identity chain intact (42 entries)"
        "CRITICAL: IDENTITY DRIFT — files modified outside chain since 2026-02-20"
        "CRITICAL: CHAIN BREAK at entry 15/42 — possible tampering"
    """
    try:
        result = verify_chain(drive_root, bible_text, identity_text)
        status = result["status"]

        if status == "OK":
            return f"OK: identity chain intact ({result['chain_length']} entries)"
        elif status == "NO_CHAIN":
            return "INFO: identity chain not yet initialized"
        elif status == "EMPTY_CHAIN":
            return "INFO: identity chain empty — awaiting first update"
        elif status == "CHAIN_BREAK":
            return f"CRITICAL: IDENTITY CHAIN BREAK — {result['message']}"
        elif status == "IDENTITY_DRIFT":
            return f"CRITICAL: IDENTITY DRIFT — {result['message']}"
        else:
            return f"WARNING: unknown identity chain status: {status}"
    except Exception as e:
        log.debug("Identity chain verification failed", exc_info=True)
        return f"WARNING: identity chain verification error: {e}"
