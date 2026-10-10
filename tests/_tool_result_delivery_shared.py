"""A measured Main fit callback for tool-result delivery tests.

Mirrors the fields ``loop_model_call._measure_main_context_view`` returns for
one sealed candidate and measures the SAME way (``estimate_context_prompt_tokens``
over the complete candidate, calibrated by one density). ``accepted`` is always
``True`` on purpose, as in production: a delivery that read it as fit would be
wrong, and the tests prove it is derived from the bounds instead.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


def measured_fit(*, window: Optional[int] = None, target: Optional[int] = None,
                 reserve: int = 0, density: float = 1.0, calls: Optional[List[Dict[str, Any]]] = None):
    """``fit_candidate(messages, schemas)`` bound to a route window / owner target."""
    from ouroboros.context_fit import estimate_context_prompt_tokens

    def fit(messages: list, schemas: list) -> Dict[str, Any]:
        import math

        raw = int(estimate_context_prompt_tokens(messages, schemas))
        measurement = {
            "estimated_input_tokens": int(math.ceil(raw * density)),
            "raw_input_tokens": raw,
            "response_reserve_tokens": int(reserve),
            "target_total_tokens": target,
            "capacity_total_tokens": window,
            "measurement_basis": "fresh_route_usage" if window else "cold_estimate",
            "measurement_density": float(density),
            "accepted": True,
            "strict_bound_proven": False,
        }
        if calls is not None:
            calls.append(measurement)
        return measurement

    return fit
