"""Fresh, non-executable conflict metadata over canonical skill locations.

Only the selected execution subject needs payload identity, review and grants.
Peers need manifest conflicts and enablement; hashing their payloads contributes
nothing to that decision. There is no stat/TTL cache and no partial LoadedSkill.

Every observation reads each peer manifest's bytes again. Only a successful parse
of byte-identical text is reused, as the two facts a peer keeps (``conflicts`` and
the declared ``os``): an equal-size edit with a restored mtime is new text, an
error is parsed again next time, and enablement, location, source class and the
host platform are evaluated per observation. ``PEER_PARSE_REUSE_ENTRIES`` bounds
memory, not meaning: past it a peer is parsed again. A new process starts empty.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from ouroboros.contracts.skill_manifest import SkillManifest, SkillManifestError, parse_skill_manifest_text
from ouroboros.skill_loader import (
    _ManifestUnreadable, _classify_skill_source, _manifest_text_for_dir, _native_platform_visible,
    _SkillLocationCandidate, _skill_location_inventory, load_enabled,
)

PEER_PARSE_REUSE_ENTRIES = 128


@dataclass(frozen=True)
class SkillPeer:
    name: str
    location: str
    skill_dir: Path
    enabled: bool = False
    conflicts: tuple[str, ...] = ()
    identity_collision: bool = False
    manifest_error: str = ""


@dataclass(frozen=True)
class _PeerManifestFacts:
    """What a peer keeps from its manifest text; never the mutable ``SkillManifest``."""
    conflicts: tuple[str, ...]
    os: str
    # The manifest's own rule over ``os``, so the host platform is read at each call.
    applies_to_platform = SkillManifest.applies_to_platform


@lru_cache(maxsize=PEER_PARSE_REUSE_ENTRIES)
def _peer_manifest_facts(text: str) -> _PeerManifestFacts:
    """Parse once per exact text; a raised error is never stored."""
    manifest = parse_skill_manifest_text(text)
    return _PeerManifestFacts(conflicts=tuple(manifest.conflicts or ()), os=manifest.os)


def _manifest_peer(drive_root: Path, candidate: _SkillLocationCandidate, enabled: bool | None = None) -> SkillPeer | None:
    """One unique location from fresh manifest bytes; ``None`` where discovery omits it."""
    identity = dict(name=candidate.name, location=candidate.location, skill_dir=candidate.skill_dir)
    try:
        read = _manifest_text_for_dir(candidate.skill_dir)
        if read is None:
            return None
        facts = _peer_manifest_facts(read[0])
    except (_ManifestUnreadable, SkillManifestError) as exc:
        return SkillPeer(**identity, manifest_error=type(exc).__name__)
    if candidate.location == "native" and not _native_platform_visible(candidate.skill_dir, facts) and _classify_skill_source(
            candidate.skill_dir, location=candidate.location, drive_root=drive_root) == "native":
        return None  # discovery hides a launcher seed for another platform
    return SkillPeer(**identity, enabled=load_enabled(drive_root, candidate.name) if enabled is None else enabled,
                     conflicts=facts.conflicts)


def discover_skill_peers(
    drive_root: Path, *, repo_path: str | None = None,
    locations: tuple[_SkillLocationCandidate, ...] | None = None,
) -> tuple[SkillPeer, ...]:
    """Match discover_skills' peer semantics, without opening payload files.

    Collisions and malformed/unreadable manifests are disabled placeholders.
    A valid manifest retains its enablement and conflicts even if its payload
    would be unreadable: that is exactly how load_skill projects such a peer.
    Locations may be shared within an observation; manifest/state reads stay fresh.
    Missing manifests are omitted by the ordinary location inventory.
    """
    inventory = locations if locations is not None else _skill_location_inventory(drive_root, repo_path=repo_path)
    counts = Counter(candidate.name for candidate in inventory)
    peers = []
    for candidate in inventory:
        if counts[candidate.name] > 1:
            peers.append(SkillPeer(name=candidate.name, location=candidate.location,
                                   skill_dir=candidate.skill_dir, identity_collision=True))
        elif (peer := _manifest_peer(drive_root, candidate)) is not None:
            peers.append(peer)
    return tuple(peers)


def discover_conflict_peers(
    drive_root: Path, subject: str, *, repo_path: str | None = None,
    locations: tuple[_SkillLocationCandidate, ...] | None = None,
) -> tuple[SkillPeer, ...]:
    """The peers able to conflict with canonical ``subject``, read fresh.

    ``skill_conflict_status`` reaches the verdict it reaches over
    ``discover_skill_peers``: there, collision and malformed placeholders and
    OFF peers are inert. Collisions are counted over the whole inventory before
    the subject is left out; enablement is read before any manifest, so an OFF
    peer costs one state read and an ON peer keeps its fresh manifest read.
    """
    inventory = locations if locations is not None else _skill_location_inventory(drive_root, repo_path=repo_path)
    counts = Counter(candidate.name for candidate in inventory)
    peers = []
    for candidate in inventory:
        if counts[candidate.name] > 1 or candidate.name == subject or not load_enabled(drive_root, candidate.name):
            continue
        peer = _manifest_peer(drive_root, candidate, enabled=True)
        if peer is not None and not peer.manifest_error:
            peers.append(peer)
    return tuple(peers)
