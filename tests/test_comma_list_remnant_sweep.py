"""Retired reviewer route environment keys cannot override the catalog pool."""


def test_retired_route_envs_are_ignored(monkeypatch):
    """Retired-envs-are-ignored pin: a stale environment exporting the retired
    per-row route spellings changes NOTHING — a pool of api rows stays api_chat."""
    from ouroboros.review_execution import ReviewRouteKind
    from ouroboros.reviewer_slot_config import review_pool_slots
    from tests.review_pool_rosters import set_review_pool

    set_review_pool(monkeypatch, ["m1", "m2", "m3"])
    monkeypatch.setenv("OUROBOROS_REVIEW_ROUTES", "agent_session,agent_session,agent_session")
    assert all(row.route is ReviewRouteKind.API_CHAT
               for row in review_pool_slots(role_hint="commit review"))
