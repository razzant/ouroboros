"""The measured context carried from Main fit to a settled physical attempt.

This data record is independent of the usage ledger and of context assembly.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional


@dataclass(frozen=True)
class PhysicalAttemptContext:
    profile: Literal["owner_max", "owner_low", "owner_nano", "task_local_low", "task_local_nano"]
    rendered_mode: Literal["max", "low", "nano"]
    measurement_basis: Literal["fresh_route_usage", "fresh_model_usage", "cold_estimate"]
    route_fp: str
    round_id: str
    target_total_tokens: Optional[int]
    capacity_total_tokens: Optional[int]
    context_target_miss: bool
    automatic_pass_used: bool
    # Main's calibration (real tokens per estimated token) of the measurement above;
    # the send finalizer sizes the reply with it. None on rows written before it existed.
    measurement_density: Optional[float] = None
    view_kind: str = ""  # the captured memory view; old producers stay unclassified
    response_reserve_tokens: int = 0
    window_witness_bound: bool = False
    window_observation_fp: str = ""  # Main account scope, separate from capability/density route_fp
