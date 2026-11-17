"""The accepted wake's transition state, kept beside its observation as a chain of immutable files.

A wake accepts, with its chat position and memory sequence, every observed task's transition identities and write
stamps (``consciousness_wake._transitions_since``, ``_task_window``). That state lives under
``state/consciousness_transitions/`` as a base and exact changes to it, each file named by the sha256 of its bytes
(``sort_keys``, no spaces, ``ensure_ascii=False``): a base is ``{"version": 3, "scan_at", "inventory", "signatures"}``;
a change carries the same fields with ``"parent"`` (the link it extends) and only the tasks whose entries differ from
its parent's state, ``null`` for a task the state no longer holds. The accepted boundary names the newest link
(``transition_chain``); the wake's observation source keeps its header and events. A window that changes nothing
writes no link. A change is written as a new base once the chain's changes, each counted as at least one filesystem
block, would reach its base's bytes, so reading a chain never costs more than two bases in bytes or files: storage
mechanics, nothing the mind sees depends on it. Every read verifies each file against its name; a failure is the
caller's unreadable source, never a guess by time.

A boundary of the previous form names the state in its wake source's header (``transitions.source_ref``); the next
wake reads it there and writes a base. ``export_for_older_body`` writes that form back before a switch to a body
without this module (``store_switch``).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import pathlib
import re
from dataclasses import dataclass
from typing import Any, Dict, FrozenSet, Optional, Tuple

from ouroboros.store import fsync_directory
from ouroboros.utils import write_bytes_atomic

log = logging.getLogger(__name__)

CHAIN_REL = pathlib.Path("state") / "consciousness_transitions"
CHAIN_VERSION = 3  # the files' format and the boundary's ``transition_chain`` reference
TRANSITIONS_VERSION = 2  # the state as a wake compares it: complete per-task inventory, closed tasks included
CHAT_BOUNDARY_KEYS = ("version", "upper", "segment_start", "segment_sha256")  # the accepted chat position
BLOCK_BYTES = 4096  # the least one file occupies on disk
_FIELDS = ("inventory", "signatures")
_NAME = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True)
class TransitionChain:
    """An accepted chain as read: its newest link, the state it assembles and what reading it costs."""

    link: str
    state: Dict[str, Any]  # ``{"version": TRANSITIONS_VERSION, "scan_at", "inventory", "signatures"}``
    base_bytes: int
    change_bytes: int  # its changes, each counted as at least one block
    names: FrozenSet[str]  # its files


def _encode(record: Dict[str, Any]) -> bytes:
    return json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _check_inventory(inventory: Any) -> None:
    """The inventory a wake compares with: task id -> its transition keys."""
    if not isinstance(inventory, dict) or any(not isinstance(keys, list) or not all(isinstance(key, str) for key in keys)
                                              for keys in inventory.values()):
        raise ValueError("transition inventory has an unexpected shape")


def read_chain(drive_root: Any, reference: Any) -> TransitionChain:
    """The state a ``transition_chain`` reference assembles, from its base forward; each file must hash to its name
    and hold the format's fields, anything else raises."""
    directory, links = pathlib.Path(drive_root) / CHAIN_REL, []
    name = reference.get("link") if isinstance(reference, dict) and reference.get("version") == CHAIN_VERSION else None
    while True:
        if not isinstance(name, str) or not _NAME.fullmatch(name):
            raise ValueError("transition link is not named by a sha256")
        data = (directory / f"{name}.json").read_bytes()
        record = json.loads(data.decode("utf-8")) if hashlib.sha256(data).hexdigest() == name else None
        if not isinstance(record, dict) or record.get("version") != CHAIN_VERSION or not all(
                isinstance(record.get(key), dict) for key in _FIELDS):
            raise ValueError(f"transition link {name[:12]} does not match its name or format")
        links.append((name, record, len(data)))
        if "parent" not in record:
            break
        name = record["parent"]
    *changes, (_name, base, base_bytes) = links
    state = {"version": TRANSITIONS_VERSION, "scan_at": links[0][1].get("scan_at"),
             **{key: dict(base[key]) for key in _FIELDS}}
    for _name, record, _size in reversed(changes):
        for key in _FIELDS:
            for task_id, value in record[key].items():
                if value is None:
                    state[key].pop(task_id, None)
                else:
                    state[key][task_id] = value
    _check_inventory(state["inventory"])
    return TransitionChain(link=links[0][0], state=state, base_bytes=base_bytes,
                           change_bytes=sum(max(size, BLOCK_BYTES) for _name, _record, size in changes),
                           names=frozenset(f"{name}.json" for name, _record, _size in links))


def read_source_transitions(drive_root: Any, accepted: Dict[str, Any]) -> Dict[str, Any]:
    """The state a previous-form boundary names in its wake source's header; the header must match the boundary."""
    from ouroboros.artifacts import read_actor_source_bytes

    ref = accepted["transitions"]["source_ref"]
    header = json.loads(read_actor_source_bytes(pathlib.Path(drive_root), ref["task_id"], ref).splitlines()[0])
    stored = header["boundary"]
    state = stored["transitions"]
    if (header.get("kind") != "wake_observation" or header.get("captured_at") != accepted.get("captured_at")
            or any(stored.get(key) != accepted.get(key) for key in CHAT_BOUNDARY_KEYS)
            or stored.get("memory") != accepted.get("memory") or state.get("version") != TRANSITIONS_VERSION):
        raise ValueError("wake source does not match accepted boundary")
    _check_inventory(state.get("inventory"))
    return state


def prune(drive_root: Any, chain: TransitionChain) -> None:
    """Remove every file of the chain directory outside ``chain``: links no boundary accepted, links before its base,
    a leftover temporary file. No one reads them again: the clock starts one own wake at a time (``live_wake``) and
    accepts a wake's boundary only while that wake is its current one (``consciousness._settle_wake``), so the chain a
    wake reads is the newest any boundary will name until its own link. A failure is logged and leaves the wake alone."""
    try:
        with os.scandir(pathlib.Path(drive_root) / CHAIN_REL) as entries:
            stray = [entry.path for entry in entries if entry.name not in chain.names]
    except OSError:
        log.debug("consciousness: transition files not listed", exc_info=True)
        return
    for path in stray:
        try:
            os.unlink(path)
        except OSError:
            log.debug("consciousness: transition file %s not removed", path, exc_info=True)


def accepted_transitions(drive_root: Any, accepted: Dict[str, Any]) -> Tuple[Any, Optional[TransitionChain]]:
    """``(state, chain)`` an accepted boundary carries: the chain it names (then every other file of its directory is
    removed: ``prune``), the wake source a previous-form boundary names (no chain), else its inline state. A named
    state that cannot be read raises, and nothing is removed."""
    if "transition_chain" in accepted:
        chain = read_chain(drive_root, accepted["transition_chain"])
        prune(drive_root, chain)
        return chain.state, chain
    prior = accepted.get("transitions")
    if isinstance(prior, dict) and "source_ref" in prior:
        return read_source_transitions(drive_root, accepted), None
    return prior, None


def bind_link(drive_root: Any, chain: Optional[TransitionChain], state: Dict[str, Any]) -> Dict[str, Any]:
    """The ``transition_chain`` reference a wake that observed ``state`` accepts: ``chain``'s newest link when neither
    its inventory nor its signatures differ, else a change to that link, or a base when there is no chain or the
    chain's changes would reach its base's bytes. A link is written as a synced temporary file replaced into its name,
    the directory synced, then read back byte for byte; any failure raises."""
    current = {key: state.get(key) or {} for key in _FIELDS}
    head, data = {"version": CHAIN_VERSION, "scan_at": state.get("scan_at")}, b""
    if chain is not None:
        change = {key: {task_id: current[key].get(task_id) for task_id in current[key].keys() | chain.state[key].keys()
                        if current[key].get(task_id) != chain.state[key].get(task_id)} for key in _FIELDS}
        if not any(change.values()):
            return {"version": CHAIN_VERSION, "link": chain.link}
        delta = _encode({**head, **change, "parent": chain.link})
        if chain.change_bytes + max(len(delta), BLOCK_BYTES) < chain.base_bytes:
            data = delta
    data = data or _encode({**head, **current})
    name = hashlib.sha256(data).hexdigest()
    path = pathlib.Path(drive_root) / CHAIN_REL / f"{name}.json"
    write_bytes_atomic(path, data, fsync=True)
    fsync_directory(path.parent)
    if path.read_bytes() != data:
        raise ValueError("transition link readback mismatch")
    return {"version": CHAIN_VERSION, "link": name}


def export_for_older_body(drive_root: Any) -> Dict[str, Any]:
    """Before a switch to a body without this module: the accepted chain's state written as the wake source a
    previous-form boundary names (one header line, under the accepted wake), then the boundary pointed at it in place
    of its chain, so the older body reads the same state. A boundary of the previous form, or none, has nothing to
    export. Any failure raises with the boundary unchanged; the caller decides what that costs the switch."""
    from types import SimpleNamespace

    from ouroboros.artifacts import read_actor_source_bytes
    from ouroboros.chat_chain import retain_memory_source
    from ouroboros.consciousness import OBSERVATION_STATE_KEY
    from supervisor import state as runtime_state

    root, accepted = pathlib.Path(drive_root), runtime_state.load_state().get(OBSERVATION_STATE_KEY)
    if not isinstance(accepted, dict) or "transition_chain" not in accepted:
        return {"status": "nothing_to_export"}
    chain = read_chain(root, accepted["transition_chain"])
    kept = {key: value for key, value in accepted.items() if key != "transition_chain"}
    boundary = {key: kept[key] for key in (*CHAT_BOUNDARY_KEYS, "memory") if key in kept}
    header = {"kind": "wake_observation", "captured_at": kept.get("captured_at"), "window": {}, "gaps": [],
              "composition": {}, "trigger": "", "boundary": {**boundary, "transitions": chain.state}}
    data = (json.dumps(header, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")
    task_id = str(kept["task_id"])
    source = retain_memory_source(SimpleNamespace(drive_root=root, task_id=task_id), "wake_observation", data,
                                  extension="jsonl")
    if read_actor_source_bytes(root, task_id, source) != data:
        raise ValueError("exported wake source readback mismatch")
    older = {**kept, "transitions": {"version": TRANSITIONS_VERSION, "source_ref": source}}

    def point(state: Dict[str, Any]) -> None:
        if state.get(OBSERVATION_STATE_KEY) != accepted:
            raise ValueError("the accepted wake boundary changed during the export")
        state[OBSERVATION_STATE_KEY] = older

    runtime_state.update_state(point)
    return {"status": "exported", "link": chain.link, "tasks": len(chain.state["inventory"]), "source": source["path"]}
