"""Fresh, non-executable conflict metadata over canonical skill locations.

Only the selected execution subject needs payload identity, review and grants.
Peers need manifest conflicts and enablement; hashing their payloads contributes
nothing to that decision, except for a deployment's required name at its required
path, whose enablement depends on its bytes (``deployment_extensions.peer_enabled``,
the loader's reused read). There is no stat/TTL cache and no partial LoadedSkill.

Every observation reads each peer manifest's bytes again. A successful parse
leaves the two facts a peer keeps (``conflicts`` and the declared ``os``), held
for that location with the exact text they came from and reused only while a
fresh read returns that text: an equal-size edit with a restored mtime is new
text, parsed at once in place of the old facts, and an error is parsed again
next time, never held. Enablement, location, source class and the host
platform are evaluated per observation. Each pass over an install's inventory
keeps facts only for the locations it lists, so a removed, renamed or
disconnected package leaves nothing behind and what is held is the install
itself, of any size (#1217: a fixed count of reused texts was outrun by an
install with more enabled peers, which every observation then parsed again).
Facts are held per data root and repository spelling for the few installs a
process observes (``HELD_INSTALLS``); a new process starts empty.
"""

from __future__ import annotations

import threading
from collections import Counter, OrderedDict
from dataclasses import dataclass
from pathlib import Path

from ouroboros.contracts.skill_manifest import SkillManifest, SkillManifestError, parse_skill_manifest_text
from ouroboros.deployment_extensions import peer_enabled
from ouroboros.skill_loader import (
    _ManifestUnreadable, _classify_skill_source, _manifest_text_for_dir, _native_platform_visible,
    _SkillLocationCandidate, _skill_location_inventory, load_enabled,
)

HELD_INSTALLS = 8  # a process observes one install (tests: many); past it the least recent one parses again


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


_HeldFacts = dict[Path, tuple[str, _PeerManifestFacts]]  # location -> (exact manifest text, its facts)
_HELD_FACTS: "OrderedDict[tuple[str, str | None], _HeldFacts]" = OrderedDict()  # by (data root, repo_path)
_HELD_LOCK = threading.Lock()


def _peer_manifest_facts(text: str) -> _PeerManifestFacts:
    manifest = parse_skill_manifest_text(text)
    return _PeerManifestFacts(conflicts=tuple(manifest.conflicts or ()), os=manifest.os)


def _held_facts(drive_root: Path, repo_path: str | None,
                inventory: tuple[_SkillLocationCandidate, ...]) -> _HeldFacts:
    """This install's held facts, released for every location ``inventory`` no longer lists."""
    key = (str(drive_root), repo_path)
    with _HELD_LOCK:
        previous = _HELD_FACTS.pop(key, {})
        held = {}
        for candidate in inventory:
            entry = previous.get(candidate.skill_dir)
            if entry is not None:
                held[candidate.skill_dir] = entry
        _HELD_FACTS[key] = held
        while len(_HELD_FACTS) > HELD_INSTALLS:
            _HELD_FACTS.popitem(last=False)
    return held


def _manifest_peer(drive_root: Path, candidate: _SkillLocationCandidate, held: _HeldFacts,
                   enabled: bool | None = None) -> SkillPeer | None:
    """One unique location from fresh manifest bytes; ``None`` where discovery omits it."""
    identity = dict(name=candidate.name, location=candidate.location, skill_dir=candidate.skill_dir)
    entry = held.pop(candidate.skill_dir, None)  # kept again only for a successful read and parse
    try:
        read = _manifest_text_for_dir(candidate.skill_dir)
        if read is None:
            return None
        text = read[0]
        facts = entry[1] if entry is not None and entry[0] == text else _peer_manifest_facts(text)
    except (_ManifestUnreadable, SkillManifestError) as exc:
        return SkillPeer(**identity, manifest_error=type(exc).__name__)
    held[candidate.skill_dir] = (text, facts)
    if candidate.location == "native" and not _native_platform_visible(candidate.skill_dir, facts) and _classify_skill_source(
            candidate.skill_dir, location=candidate.location, drive_root=drive_root) == "native":
        return None  # discovery hides a launcher seed for another platform
    if enabled is None:  # a deployment's required package reads as on (``deployment_extensions``)
        enabled = peer_enabled(drive_root, candidate, load_enabled(drive_root, candidate.name))
    return SkillPeer(**identity, enabled=enabled, conflicts=facts.conflicts)


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
    held = _held_facts(drive_root, repo_path, inventory)
    counts = Counter(candidate.name for candidate in inventory)
    peers = []
    for candidate in inventory:
        if counts[candidate.name] > 1:
            peers.append(SkillPeer(name=candidate.name, location=candidate.location,
                                   skill_dir=candidate.skill_dir, identity_collision=True))
        elif (peer := _manifest_peer(drive_root, candidate, held)) is not None:
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
    held = _held_facts(drive_root, repo_path, inventory)
    counts = Counter(candidate.name for candidate in inventory)
    peers = []
    for candidate in inventory:
        if counts[candidate.name] > 1 or candidate.name == subject or not peer_enabled(
                drive_root, candidate, load_enabled(drive_root, candidate.name)):
            continue
        peer = _manifest_peer(drive_root, candidate, held, enabled=True)
        if peer is not None and not peer.manifest_error:
            peers.append(peer)
    return tuple(peers)
