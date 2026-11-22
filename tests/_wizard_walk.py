"""Walks of the setup wizard that follow its own step order (the browser tests' shared helper).

The served bootstrap's ``stepOrder`` is the one list of steps the page renders. Two steps are optional: Consciousness
(between Choose access and Skills) while a first state is still to come, and Skills (before Review limits) while this
data root's starter list never ran. A walk reads that list instead of counting, so the same test is right on every
shape.
"""
from __future__ import annotations

from itertools import combinations

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


QUICK_STEPS = {"consciousness": "[data-consciousness-step]", "skills": '[data-starter-skills="step"]'}


def past_quick_steps(page) -> None:
    """Review & start lands on Consciousness, then Skills, where the page offers them; the last Continue goes to Summary."""
    for step, selector in QUICK_STEPS.items():
        if step in step_order(page):
            page.wait_for_selector(selector)
            page.click("#next-btn")


def untick_starter_skills(page) -> None:
    """From Summary: back to the Skills step through its row, Untick all, and on to Summary again, so an isolated root
    installs nothing from the public hub. Where the step is not offered nothing would install, and nothing is done."""
    if "skills" in step_order(page):
        page.locator("[data-starter-change]").click()
        page.locator('[data-starter-all="0"]').click()
        walk(page, "skills", "summary")
        page.wait_for_selector(".summary-card")


FIRST_RUN = ["accounts", "models", "review_mode", "consciousness", "skills", "budget", "summary"]
OPTIONAL = ("consciousness", "skills")
SHAPES = [[step for step in FIRST_RUN if step not in dropped]
          for count in range(len(OPTIONAL) + 1) for dropped in combinations(OPTIONAL, count)]


def rail_shows_its_steps(page) -> bool:
    """The rail draws one pill per served step, and the order is one of the wizard's shapes."""
    order = step_order(page)
    return order in SHAPES and page.locator(".wizard-step").count() == len(order)
