"""Exact reuse of installed-skill discovery: one inventory and one load per content identity (cost law L13).

Every context build, ``list_skills`` call and Host Service request discovered the skills again: the walk of the
skills roots (``skill_loader._skill_location_inventory``) and every package's manifest, payload hash and owner
state (``skill_loader._load_candidate``), though installs, updates, reviews, enablement and edits are rare. Both
answers are kept here, per process, with the identity of exactly what they read, and handed out again only while
that identity is unchanged:

- the inventory: keyed by the data root, the roots the settings name and the selected manifestless name; its
  identity is the change key of each root and of every entry within the two levels the walk lists
  (``_walk_skill_packages``), and the same of each connected folder's standard skill folders, read one level
  deeper (``skill_sources.STANDARD_SKILL_FOLDERS``), so an install, removal, rename, or a manifest added or
  removed changes it;
- one package: keyed by the data root, its location candidate and the manifestless choice (no load fact depends
  on the runtime mode: the gate reads the mode when asked); its identity is the change key of every directory
  and file of its payload tree (the cache names the hash skips are skipped, the manifest-declared entries the hash
  adds are added) and of the owner-state files a load reads (``STATE_FILES``): an update, review, enablement,
  attestation, owner edit or an external edit of any payload byte changes it;
- one release digest (``skill_identity.payload_digest``): keyed by the directory and the arguments that shape the
  walk, with the payload tree's identity alone, so a load that hashes the bytes discovery just hashed (an
  extension's load, its Host Service token) takes that digest instead of walking and hashing again.

A change key is ``skill_identity._change_key``'s (device, inode, size, mtime, ctime) on a local POSIX filesystem,
the rule the payload digests already trust: a write, truncation, ``utime``, rename or entry added to a directory
moves a change time the caller cannot set back. The identity is taken after the read it covers, and kept only when
no change time lies within ``_RACY_NS`` of the read's start, so a write during the read or in its coarse clock tick
is read again next time. On Windows or a filesystem the store does not call local nothing is kept (``keeps``): discovery
reads everything every time, as before, and a load hashes again what discovery just hashed (``skill_hash_again``, a
repeat of that platform). Each caller gets its own skill, review and manifest objects, since callers assign
their fields (``enabled``, ``review``, ``source``); the values inside are shared and no caller mutates them in place,
as no reader of a parsed view does (a deep copy would cost more than the reviews' JSON it saves parsing). Nothing is
written, and no process shares another's answers.
"""
from __future__ import annotations

import copy
import os
import pathlib
import time
from collections import OrderedDict
from typing import Any, Callable, Dict, Iterable, Optional, Tuple

from ouroboros.skill_acceptance import OWNER_EDIT_FILENAME
from ouroboros.skill_identity import _RACY_NS, _change_key
from ouroboros.skill_install_record import INSTALL_RECORD_FILENAME

# The owner-state files of ``state/skills/<name>/`` one load reads (``load_enabled``, ``load_review_state`` with its
# acceptance facts, ``is_self_authored_skill_dir``, ``_classify_skill_source``); the rest of that directory (health,
# review jobs and history, call logs) changes often and is never read by a load.
STATE_FILES = ("enabled.json", "review.json", "owner_attestation.json", INSTALL_RECORD_FILENAME, OWNER_EDIT_FILENAME,
               "self_authored.json")
_MISSING = "missing"
_INVENTORIES: "OrderedDict[Tuple[Any, ...], Tuple[Dict[str, Any], Any]]" = OrderedDict()
_SKILLS: "OrderedDict[Tuple[Any, ...], Tuple[Dict[str, Any], Any]]" = OrderedDict()
_DIGESTS: "OrderedDict[Tuple[Any, ...], Tuple[Dict[str, Any], str]]" = OrderedDict()
_KEPT = 4096  # entries of each memo: a memory bound (an install holds tens to hundreds of packages), not a policy


def _key(path: Any) -> Any:
    """The change key of ``path`` (links followed), ``"missing"`` when absent, ``None`` when it cannot be trusted."""
    try:
        return _change_key(pathlib.Path(path))
    except (FileNotFoundError, NotADirectoryError):
        return _MISSING
    except OSError:
        return None


def keeps(path: Any) -> bool:
    """Whether a read under ``path`` may be kept: its change key is trusted (a local filesystem, never Windows)."""
    return _key(path) is not None


def _tree(root: Any, *, skip: Iterable[str] = (), depth: Optional[int] = None, links: bool = False) -> Dict[str, Any]:
    """The change keys of ``root`` and of every entry below it, ``depth`` levels deep (all, when None), outside the
    ``skip`` names; a symbolic link to a directory is descended only with ``links`` (the inventory's ``is_dir``)."""
    keys, pending = {str(root): _key(root)}, [(str(root), 0)]
    while pending:
        directory, level = pending.pop()
        try:
            with os.scandir(directory) as found:
                entries = [entry for entry in found if entry.name not in skip]
        except OSError:  # an unlistable directory is read by no walk; its own key stands for it
            continue
        for entry in entries:
            keys[entry.path] = _key(entry.path)
            try:
                deeper = entry.is_dir(follow_symlinks=links)
            except OSError:
                deeper = False
            if deeper and (depth is None or level + 1 < depth):
                pending.append((entry.path, level + 1))
    return keys


def _trusted(keys: Dict[str, Any], started: int) -> bool:
    """Every key is known, and none changed within ``_RACY_NS`` of the read that started at ``started``."""
    return all(key == _MISSING or key is not None and key[1][2] < started - _RACY_NS for key in keys.values())


def _own(loaded: Any) -> Any:
    """A caller's own skill, review and manifest objects over the shared values inside them (None stays None)."""
    if loaded is None:
        return None
    own = copy.copy(loaded)
    own.review, own.manifest = copy.copy(loaded.review), copy.copy(loaded.manifest)
    return own


def _kept(memo: "OrderedDict[Tuple[Any, ...], Any]", key: Tuple[Any, ...], held: Any) -> None:
    memo.pop(key, None)
    memo[key] = held
    if len(memo) > _KEPT:
        memo.popitem(last=False)


def reused_inventory(key: Tuple[Any, ...], roots: Iterable[Any], walk: Callable[[], Any]) -> Any:
    """``walk()``, the location inventory of ``roots``, or the last one of ``key`` while its directories hold."""
    roots = [pathlib.Path(root) for root in roots]
    if not all(keeps(root) for root in roots):  # Windows or a filesystem not called local: read, keep nothing
        return walk()

    def identity() -> Dict[str, Any]:
        keys: Dict[str, Any] = {}
        for root in roots:
            keys.update(_tree(root, depth=2, links=True))
        return keys

    held = _INVENTORIES.get(key)
    if held is not None and identity() == held[0]:
        return held[1]  # a tuple of frozen candidates: shared as it is
    started = time.time_ns()
    inventory = walk()
    keys = identity()
    if _trusted(keys, started):
        _kept(_INVENTORIES, key, (keys, inventory))
    return inventory


def declared_entries(entry: Any, scripts: Any) -> Tuple[str, ...]:
    """The manifest-declared entry and script names the release digest re-adds even under a skipped cache name."""
    return tuple(filter(None, (str(entry or ""), *(str(script.get("name") or "") for script in scripts or ()
                                                   if isinstance(script, dict)))))


def _payload_identity(skill_dir: Any, declared: Iterable[str]) -> Dict[str, Any]:
    """The change keys of the payload tree the release digest walks, and of its declared entries."""
    from ouroboros.skill_loader import _SKILL_DIR_CACHE_NAMES

    keys = _tree(skill_dir, skip=_SKILL_DIR_CACHE_NAMES)
    for rel in declared:
        for path in (pathlib.Path(skill_dir) / rel, pathlib.Path(skill_dir) / "scripts" / rel):
            keys[str(path)] = _key(path)
    return keys


def reused_digest(key: Tuple[Any, ...], skill_dir: Any, declared: Tuple[str, ...], compute: Callable[[], str]) -> str:
    """``compute()``, the release digest of ``skill_dir``, or the last one of ``key`` while its payload tree holds.
    Counted in the measured window (``context_timing``): ``skill_hash`` per digest computed, by ``key``,
    ``skill_hash_reused`` per digest taken; where nothing is kept a second digest of ``key`` is ``skill_hash_again``."""
    from ouroboros.context_timing import counted

    local = keeps(skill_dir)  # Windows or a filesystem not called local: compute, keep nothing
    held = _DIGESTS.get(key) if local else None
    if held is not None and _payload_identity(skill_dir, declared) == held[0]:
        counted("skill_hash_reused")
        return held[1]
    counted("skill_hash", key=key)
    started = time.time_ns()
    digest = compute()
    if local and _trusted(keys := _payload_identity(skill_dir, declared), started):
        _kept(_DIGESTS, key, (keys, digest))
    return digest


def reused_skill(key: Tuple[Any, ...], skill_dir: Any, state_dir: Any, load: Callable[[], Any]) -> Any:
    """A copy of ``load()``, the loaded package at ``skill_dir`` (or None), or of the last one of ``key`` while
    its payload tree, the owner-state files in ``state_dir`` and the manifest-declared entries hold."""
    if not keeps(skill_dir):  # Windows or a filesystem not called local: read, keep nothing
        return load()

    def identity(loaded: Any) -> Dict[str, Any]:
        manifest = getattr(loaded, "manifest", None)
        keys = _payload_identity(skill_dir, declared_entries(getattr(manifest, "entry", ""),
                                                            getattr(manifest, "scripts", None)))
        keys.update((str(pathlib.Path(state_dir) / name), _key(pathlib.Path(state_dir) / name)) for name in STATE_FILES)
        return keys

    held = _SKILLS.get(key)
    if held is not None and identity(held[1]) == held[0]:
        return _own(held[1])
    started = time.time_ns()
    loaded = load()
    keys = identity(loaded)
    if _trusted(keys, started):
        _kept(_SKILLS, key, (keys, _own(loaded)))
    return loaded
