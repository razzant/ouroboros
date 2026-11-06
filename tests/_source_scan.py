"""Parse a source tree without the cyclic collector walking every tree parsed so far.

A scan that holds its trees in a worker-sized heap spends most of its time in the collector: the
whole runtime tree took 13 CPU-s with it running and 2.6 with it paused around each parse, and
under load such scans are where the 300 s bound fires. AST nodes are acyclic, so a pause leaks
nothing. Trees stay the caller's: no cache, so no scan can read another's stale result.
"""
from __future__ import annotations

import ast
import gc
from typing import Any


def parse(source: Any, *args: Any, **kwargs: Any) -> ast.AST:
    """``ast.parse`` with the collector paused, then back as it was found."""
    collecting = gc.isenabled()
    gc.disable()
    try:
        return ast.parse(source, *args, **kwargs)
    finally:
        if collecting:
            gc.enable()
