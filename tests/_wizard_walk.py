"""Walks of the setup wizard that follow its own step order (the browser tests' shared helper).

The served bootstrap's ``stepOrder`` is the one list of steps the page renders: a root whose first state is still to
come offers Consciousness between Choose access and Review limits, a root that already has a state does not. A walk
reads that list instead of counting, so the same test is right on both shapes.
"""
from __future__ import annotations

_STEP_ORDER = "() => (window.__OURO_ONBOARDING_BOOTSTRAP__ || {}).stepOrder || []"


def step_order(page) -> list:
    """The step ids this page renders, in order."""
    return list(page.evaluate(_STEP_ORDER))


def walk(page, start: str, end: str) -> None:
    """Continue (or Back, when ``end`` comes first) from step ``start`` to step ``end``, one click per step between."""
    order = step_order(page)
    clicks = order.index(end) - order.index(start)
    for _ in range(abs(clicks)):
        page.click("#next-btn" if clicks > 0 else "#back-btn")


def past_consciousness(page) -> None:
    """Review & start lands on Consciousness where the page offers it; its Continue goes to Summary."""
    if "consciousness" in step_order(page):
        page.wait_for_selector("[data-consciousness-step]")
        page.click("#next-btn")


WITH_STATE = ["accounts", "models", "review_mode", "budget", "summary"]
FIRST_STATE = ["accounts", "models", "review_mode", "consciousness", "budget", "summary"]


def rail_shows_its_steps(page) -> bool:
    """The rail draws one pill per served step, and the order is one of the wizard's two shapes."""
    order = step_order(page)
    return order in (WITH_STATE, FIRST_STATE) and page.locator(".wizard-step").count() == len(order)
