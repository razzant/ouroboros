"""The turn law's gap table and the function that holds a law to it (tests/test_cost_contract.py B10,
tests/test_cost_host.py).

A breach in today's code is a ``TURN_GAPS`` row: its cause with today's numbers and its remover, or why it has
none, and the counters it excuses, each up to today's growth. The law's test is then a strict xfail (``turn_gap``)
that flips (XPASS) the moment its cause is gone, and the row is deleted; a counter the row does not name, or growth
past a ceiling, fails for real, so the marker cannot hide a new regression. A row whose growth depends on the host the
stand runs on (``HOST_DEPENDENT_GAPS``) is a non-strict xfail: a host where the law holds passes it.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, List, Tuple

import pytest


class TurnLawBreach(AssertionError):
    """A turn law broken where a ``TURN_GAPS`` row excuses it. Any other failure is a real one."""


# (law, tag) -> (the cause: today's numbers, its readers and its remover, or why it has none; the counters it
# excuses, each with the growth it may not pass meanwhile: measured at k=30 for the history laws, scaled by the
# units of history added). Measured 26.10 on the integrated tree in its steady state (tests/_turn_shapes.py).
TURN_GAPS: Dict[Tuple[str, str], Tuple[str, Dict[str, int]]] = {
    ("L3", "T:main:archive"): (
        "40 archived knowledge notes and 40 old rescue snapshots (k=1, the turn after the one that met them): "
        "the knowledge shelf is listed for the index the mind is shown, which counts the archived notes "
        "(knowledge.inventory_knowledge: dir_entries +80, scandir +2), and the health block lists the rescue "
        "folder (context_health._collect_log_analysis_checks: dir_entries +40, listdir +1); no note body is "
        "read again (knowledge._navigation_row). NEVER: both listings are what the turn shows (the shelf's "
        "index; the rescue scan, kept by decision); the row stays so that any other work per item fails.",
        {"dir_entries": 122, "scandir": 4, "listdir": 3}),
    ("L3", "T:main:broad"): (
        "40 more live rooms with story and 10 running roots in a 118,000-token window whose tool schemas take "
        "72,260: the start keeps Low and its reply reserve of 32,000 by moving host sections behind their readers "
        "(section_floor), as many as this host's own sections need. On a macOS development host it moves four "
        "(runtime, review pool, drive state, world), each a new request part written once (observability."
        "_write_parts: 10 new parts for the 7 of the turn before, open +3); on the Ubuntu CI runner (run "
        "43119287886) the law holds within its allowance. NEVER: the count is the host sections the start could "
        "not hold whole, not rooms, roots or rows; the row stays so that any other work per item fails.",
        {"open": 3}),
    ("L7d", "T:main"): (
        "A capture reads every tail row to find the open ones: 6,630 rows for 31 shown or new at k=30 (250 for 31 at "
        "k=1; live 74,963 for 13,834). NEVER in this PR (owner's decision 1A): an exact skip of sealed rows needs its own "
        "design (membership stored with its target facts, consensus T1). The row stays so the debt is visible and "
        "anything beyond this read fails.", {"rows_read": 7000}),
    ("L7d", "T:mainw"): (
        "The same whole-tail read as T:main's L7d row (6,631 rows for 32 at k=30); the same owner decision.",
        {"rows_read": 7000}),
    ("host", "census"): (
        "check_stray_server_processes spawns one ps per listed process: 2 children for one, 25 for 24 (26.10); a "
        "process's first owner turn on the owner's Mac paid 373 to 632. Remover: razzant/ouroboros#2025 (one ps "
        "per 256 pids), owned by the voice sprint (owner decision 22.10, 3A).",
        {"popen": 25}),
}


# Rows whose excused growth depends on the host's own sections, not on this code: where the law holds, the test passes.
HOST_DEPENDENT_GAPS = frozenset({("L3", "T:main:broad")})


def turn_gap(law: str, tag: str) -> tuple:
    gap = TURN_GAPS.get((law, tag))
    return (pytest.mark.xfail(strict=(law, tag) not in HOST_DEPENDENT_GAPS, raises=TurnLawBreach,
                              reason=gap[0]),) if gap else ()


def hold_turn_law(law: str, tag: str, growth: Dict[str, int], allowed: Dict[str, int], readers: Any, *,
                  scale: float = 1.0) -> None:
    """Each counter's growth within its ``allowed`` growth; a ``TURN_GAPS`` row excuses the counters it names,
    each up to its ceiling (times ``scale``, the units of history added over the 29 they were measured on), and
    nothing else. ``readers`` names the readers: a value, or a function of the breaching counters."""
    breaches = {key: delta for key, delta in growth.items() if delta > allowed[key]}
    named = readers(breaches) if callable(readers) else readers
    ceilings = {key: math.ceil(ceiling * scale) for key, ceiling in TURN_GAPS.get((law, tag), ("", {}))[1].items()}
    unexcused = {key: (delta, ceilings.get(key)) for key, delta in breaches.items() if delta > ceilings.get(key, -1)}
    if ceilings:
        assert not unexcused, f"{law} {tag} grew past its gap (growth, ceiling): {unexcused}; readers: {named}"
    if breaches:
        raise TurnLawBreach(f"{law} {tag} grew {breaches}; readers: {named}")


def hold_work_equal(law: str, tag: str, pairs: Dict[str, Tuple[int, int]], *, scale: float = 1.0) -> None:
    """Each pair ``(done, delivered)`` is equal: an equality, no ceiling. ``name(...)`` in a pair's name is its
    measurement; a ``TURN_GAPS`` row excuses a pair by its bare name, only while ``done`` stays within that row's
    recorded work (times ``scale``, as in ``hold_turn_law``); anything else fails for real."""
    breaches = {name: pair for name, pair in pairs.items() if pair[0] != pair[1]}
    if not breaches:
        return
    ceilings = {name: math.ceil(ceiling * scale) for name, ceiling in TURN_GAPS.get((law, tag), ("", {}))[1].items()}
    unexcused = {name: pair for name, pair in breaches.items()
                 if pair[0] > ceilings.get(name.split("(", 1)[0], -1)}
    assert not unexcused, f"{law} {tag}: done != delivered, unexcused: {unexcused} (its gap: {ceilings})"
    raise TurnLawBreach(f"{law} {tag}: done != delivered: {breaches}")


def readers_of(before: Dict[str, Any], after: Dict[str, Any]) -> Callable[[Dict[str, int]], List[tuple]]:
    """The readers that grew the breaching counters most (a read transaction is named by its statements)."""
    def named(breaches: Dict[str, int]) -> List[tuple]:
        kinds = {key.split(":")[0] for key in breaches}
        kinds |= {"statements"} if {"begins", "commits"} & kinds else set()
        old = before.get("callers", {})
        grown = [(source, amount - old.get(source, 0)) for source, amount in after.get("callers", {}).items()
                 if source.split(" ", 1)[0].split(":")[0] in kinds and amount > old.get(source, 0)]
        return sorted(grown, key=lambda item: -item[1])[:8]
    return named
