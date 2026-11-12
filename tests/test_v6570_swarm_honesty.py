"""v6.57.0 (Phase 1) — swarm/outcome honesty + settings regression tests.

Covers: find_child_tasks scope=direct (no false grandchild absorption), the
policy_denials outcome bucket, verify_and_record refused_out_of_scope, the
recursive cost_usd_with_children rollup, the subagent profile summary, the
protected-artifact glob carve-out, and the EFFORT_SCALE SSOT.
"""
from __future__ import annotations



from ouroboros.config import (
    clamp_effort_to,
)


# --- EFFORT_SCALE SSOT (1.8) -------------------------------------------------


def test_clamp_effort_to_ceiling():
    assert clamp_effort_to("xhigh", "high") == "high"   # clamped down
    assert clamp_effort_to("low", "high") == "low"      # already under ceiling
    assert clamp_effort_to("max", "max") == "max"       # equal
    assert clamp_effort_to("high", "") == "high"        # unknown ceiling: pass-through


# --- 1.5 subagent profile summary -------------------------------------------
