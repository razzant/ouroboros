"""The accepted wake's transition state, kept beside its observation as a chain of immutable files.

A wake accepts, with its chat position and memory sequence, every observed task's transition identities and write
stamps (``consciousness_wake._transitions_since``, ``_task_window``). That state lives under
``state/consciousness_transitions/`` as a base and exact changes to it, each file named by the sha256 of its bytes
(``sort_keys``, no spaces, ``ensure_ascii=False``): a base is ``{"version": 3, "scan_at", "inventory", "signatures"}``;
a change carries the same fields with ``"parent"`` (the link it extends) and only the tasks whose entries differ from
its parent's state, ``null`` for a task the state no longer holds. The accepted boundary names the newest link
(``transition_chain``); the wake's observation source keeps its header and events. Every read verifies each file
against its name; a failure is the caller's unreadable source, never a guess by time.

A boundary of the previous form names the state in its wake source's header (``transitions.source_ref``); the next
wake reads it there and writes a base.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import re
from dataclasses import dataclass
from typing import Any, Dict, FrozenSet, Optional, Tuple

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


def accepted_transitions(drive_root: Any, accepted: Dict[str, Any]) -> Tuple[Any, Optional[TransitionChain]]:
    """``(state, chain)`` an accepted boundary carries: the chain it names, the wake source a previous-form boundary
    names (no chain), else its inline state. A named state that cannot be read raises."""
    if "transition_chain" in accepted:
        chain = read_chain(drive_root, accepted["transition_chain"])
        return chain.state, chain
    prior = accepted.get("transitions")
    if isinstance(prior, dict) and "source_ref" in prior:
        return read_source_transitions(drive_root, accepted), None
    return prior, None
