from __future__ import annotations

import asyncio
import json
import types

import pytest
from starlette.requests import Request

from ouroboros import usage_accounting as ua


def _data_root(tmp_path, monkeypatch):
    root = tmp_path / "data"
    (root / "state").mkdir(parents=True)
    (root / "logs").mkdir(parents=True)
    (root / "state" / "state.json").write_text(
        json.dumps({"spent_usd": 0.0, "spent_calls": 0}),
        encoding="utf-8",
    )
    monkeypatch.setenv("OUROBOROS_DATA_DIR", str(root))
    monkeypatch.setenv("OUROBOROS_SETTINGS_PATH", str(root / "settings.json"))
    monkeypatch.setenv("TOTAL_BUDGET", "7.5")
    ua.ensure_legacy_imported(root)
    return root


def _attempt(root, *, reservation_usd, task_id, category="task"):
    return ua.reserve_attempt(ua.AttemptRequest(
        model="openai/gpt-5.2",
        provider="openai",
        reservation_usd=reservation_usd,
        global_limit_usd=7.5,
        drive_root=root,
        task_id=task_id,
        root_task_id="root-1",
        category=category,
        source="test.gateway",
    ))


def _seed_accounting(root):
    settled = _attempt(root, reservation_usd=0.5, task_id="settled", category="review")
    ua.mark_dispatched(settled)
    ua.settle_attempt(
        settled,
        {
            "prompt_tokens": 100,
            "completion_tokens": 20,
            "cached_tokens": 40,
            "cache_write_tokens": 5,
            "prompt_cache_ttl": "default",
        },
        cost_usd=0.25,
        cost_final=True,
    )
    _attempt(root, reservation_usd=1.0, task_id="reserved")
    unresolved = _attempt(root, reservation_usd=0.5, task_id="unresolved")
    ua.mark_dispatched(unresolved)
    ua.mark_unresolved(unresolved, "provider outcome unknown")
    ua.record_unmetered_external_dispatch(
        "external-skill-call",
        drive_root=root,
        provider="external-skill",
        task_id="external",
        root_task_id="root-1",
        category="skill",
        source="test.external",
    )


def test_cost_breakdown_uses_ledger_not_later_compatibility_events(tmp_path, monkeypatch):
    from ouroboros.gateway.history import make_cost_breakdown_endpoint
    from supervisor import state as supervisor_state

    root = _data_root(tmp_path, monkeypatch)
    _seed_accounting(root)
    monkeypatch.setattr(supervisor_state, "TOTAL_BUDGET_LIMIT", 7.5)
    with (root / "logs" / "events.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({
            "type": "llm_usage",
            "model": "fabricated/compatibility-only",
            "provider": "openrouter",
            "cost": 99.0,
            "prompt_tokens": 999,
        }) + "\n")

    response = asyncio.run(make_cost_breakdown_endpoint(root)(None))
    payload = json.loads(response.body)

    assert response.status_code == 200
    assert payload["total_cost"] == 0.25
    assert payload["total_calls"] == 3
    assert payload["total_prompt_tokens"] == 100
    assert "fabricated/compatibility-only" not in payload["by_model"]
    assert payload["by_model"]["openai/gpt-5.2"]["cost"] == 0.25
    assert payload["by_task_category"]["review"]["calls"] == 1
    external = payload["by_api_key"]["external-skill"]
    assert external["cost"] == 0.0
    assert external["unknown_unmetered"] == 1
    assert external["cost_final"] is False
    assert payload["accounting"] == {
        "available": True,
        "settled_usd": 0.25,
        "confirmed_usd": 0.25,
        "estimated_usd": 0.0,
        "reserved_usd": 1.0,
        "unresolved_upper_bound_usd": 0.5,
        "accounted_usd": 1.75,
        "unknown_unmetered": 1,
        "cost_final": False,
        # The disclosed cause of `cost_final: False`, and broader than
        # `unknown_unmetered`: three of the four seeded rows are still open --
        # the reserved attempt, the unresolved one, and the external unmetered
        # dispatch whose price is unknown. Only the settled $0.25 row is final.
        "non_final_rows": 3,
        "attempt_counts": {"reserved": 1, "settled": 2, "unresolved": 1},
        "authority": "physical_attempt_ledger",
        "limit_usd": 7.5,
        "remaining_known_usd": 5.75,
    }


@pytest.mark.parametrize("raw", [None, "not-a-number"])
def test_cost_breakdown_uses_resolved_default_for_absent_or_invalid_budget(
    tmp_path, monkeypatch, raw,
):
    from ouroboros.config import SETTINGS_DEFAULTS
    from ouroboros.gateway.history import make_cost_breakdown_endpoint
    from supervisor import state as supervisor_state

    root = _data_root(tmp_path, monkeypatch)
    if raw is None:
        monkeypatch.delenv("TOTAL_BUDGET", raising=False)
    else:
        monkeypatch.setenv("TOTAL_BUDGET", raw)
    monkeypatch.setattr(supervisor_state, "TOTAL_BUDGET_LIMIT", 0.0)

    response = asyncio.run(make_cost_breakdown_endpoint(root)(None))
    payload = json.loads(response.body)

    expected = float(SETTINGS_DEFAULTS["TOTAL_BUDGET"])
    assert payload["accounting"]["limit_usd"] == expected
    assert payload["accounting"]["remaining_known_usd"] == expected


def test_api_state_money_and_call_count_are_ledger_projections(tmp_path, monkeypatch):
    from ouroboros.gateway.state import api_state
    from supervisor import queue, state, workers

    root = _data_root(tmp_path, monkeypatch)
    _seed_accounting(root)
    # Compatibility state may lag or be corrupt without becoming monetary authority.
    (root / "state" / "state.json").write_text(
        json.dumps({"spent_usd": 99.0, "spent_calls": 999, "current_branch": "ouroboros"}),
        encoding="utf-8",
    )
    monkeypatch.setattr(state, "TOTAL_BUDGET_LIMIT", 7.5)
    monkeypatch.setattr(state, "load_state", lambda: {
        "spent_usd": 99.0,
        "spent_calls": 999,
        "current_branch": "ouroboros",
    })
    monkeypatch.setattr(workers, "WORKERS", {})
    monkeypatch.setattr(workers, "PENDING", [])
    monkeypatch.setattr(workers, "RUNNING", {})
    monkeypatch.setattr(queue, "get_evolution_status_snapshot", lambda **_kwargs: {})
    app = types.SimpleNamespace(state=types.SimpleNamespace(
        drive_root=root,
        app_start=0.0,
    ))
    request = Request({
        "type": "http",
        "method": "GET",
        "path": "/api/state",
        "headers": [],
        "query_string": b"",
        "scheme": "http",
        "server": ("test", 80),
        "client": ("test", 1),
        "app": app,
    })

    response = asyncio.run(api_state(request))
    payload = json.loads(response.body)

    assert response.status_code == 200
    assert payload["spent_usd"] == 1.75
    assert payload["spent_calls"] == 3
    assert payload["budget_limit"] == 7.5
    assert payload["accounting"]["authority"] == "physical_attempt_ledger"
    assert payload["accounting"]["accounted_usd"] == 1.75
    assert payload["accounting"]["unknown_unmetered"] == 1
    assert payload["accounting"]["remaining_known_usd"] == 5.75


def test_cost_breakdown_fails_loudly_when_authoritative_history_is_corrupt(tmp_path, monkeypatch):
    from ouroboros.gateway.history import make_cost_breakdown_endpoint

    root = _data_root(tmp_path, monkeypatch)
    (root / ua.LEDGER_REL).write_bytes(b"not-json\n{}\n")
    (root / "logs" / "events.jsonl").write_text(
        json.dumps({"type": "llm_usage", "model": "fake", "cost": 99.0}) + "\n",
        encoding="utf-8",
    )

    response = asyncio.run(make_cost_breakdown_endpoint(root)(None))
    payload = json.loads(response.body)

    assert response.status_code == 503
    assert "error" in payload
    assert payload["accounting"] == {
        "available": False,
        "authority": "physical_attempt_ledger",
        "cost_final": False,
        "error_code": "ledger_unavailable",
    }
    assert "total_cost" not in payload


def test_api_state_passes_projection_only_when_roots_match_and_computation_succeeded(
    tmp_path, monkeypatch,
):
    """De-triplication seam pin: the snapshot receives this request's projection
    exactly when the request root IS the supervisor root and accounting
    computed; a foreign root or a failed computation passes NOTHING, so the
    snapshot's own strict attempt keeps owning the paused-evolution disclosure."""
    from ouroboros.gateway.state import api_state
    from supervisor import queue, state, workers

    root = _data_root(tmp_path, monkeypatch)
    _seed_accounting(root)
    monkeypatch.setattr(state, "TOTAL_BUDGET_LIMIT", 7.5)
    monkeypatch.setattr(state, "load_state", lambda: {"current_branch": "ouroboros"})
    monkeypatch.setattr(workers, "WORKERS", {})
    monkeypatch.setattr(workers, "PENDING", [])
    monkeypatch.setattr(workers, "RUNNING", {})
    captured: list = []
    monkeypatch.setattr(
        queue, "get_evolution_status_snapshot",
        lambda **kwargs: captured.append(kwargs) or {},
    )

    def _request():
        return Request({
            "type": "http", "method": "GET", "path": "/api/state", "headers": [],
            "query_string": b"", "scheme": "http", "server": ("test", 80),
            "client": ("test", 1),
            "app": types.SimpleNamespace(
                state=types.SimpleNamespace(drive_root=root, app_start=0.0),
            ),
        })

    monkeypatch.setattr(state, "DRIVE_ROOT", str(root))
    assert asyncio.run(api_state(_request())).status_code == 200
    assert list(captured[-1]) == ["budget_projection"]
    assert captured[-1]["budget_projection"]["limit_usd"] == 7.5

    monkeypatch.setattr(state, "DRIVE_ROOT", str(tmp_path / "other-root"))
    assert asyncio.run(api_state(_request())).status_code == 200
    assert captured[-1] == {}

    monkeypatch.setattr(state, "DRIVE_ROOT", str(root))
    (root / ua.LEDGER_REL).write_text("not-json\n{}\n", encoding="utf-8")
    assert asyncio.run(api_state(_request())).status_code == 200
    assert captured[-1] == {}


def test_api_state_marks_accounting_unavailable_without_legacy_zero(tmp_path, monkeypatch):
    from ouroboros.gateway.state import api_state
    from supervisor import queue, state, workers

    root = _data_root(tmp_path, monkeypatch)
    (root / ua.LEDGER_REL).write_text("not-json\n{}\n", encoding="utf-8")
    monkeypatch.setattr(state, "TOTAL_BUDGET_LIMIT", 7.5)
    monkeypatch.setattr(state, "load_state", lambda: {
        "spent_usd": 99.0, "spent_calls": 999, "current_branch": "ouroboros",
    })
    monkeypatch.setattr(workers, "WORKERS", {})
    monkeypatch.setattr(workers, "PENDING", [])
    monkeypatch.setattr(workers, "RUNNING", {})
    monkeypatch.setattr(queue, "get_evolution_status_snapshot", lambda **_kwargs: {})
    request = Request({
        "type": "http", "method": "GET", "path": "/api/state", "headers": [],
        "query_string": b"", "scheme": "http", "server": ("test", 80),
        "client": ("test", 1),
        "app": types.SimpleNamespace(state=types.SimpleNamespace(drive_root=root, app_start=0.0)),
    })

    response = asyncio.run(api_state(request))
    payload = json.loads(response.body)
    assert response.status_code == 200
    assert payload["spent_usd"] is None
    assert payload["spent_calls"] is None
    assert payload["budget_pct"] is None
    assert payload["accounting"]["available"] is False
    assert payload["accounting"]["accounted_usd"] is None
    assert payload["accounting"]["remaining_known_usd"] is None
    assert payload["accounting"]["error_code"] == "ledger_unavailable"


# --- v6.91 root task-detail cost_breakdown view ------------------------------


def _settled_attempt(root, *, task_id, cost, category="task"):
    attempt = _attempt(root, reservation_usd=cost, task_id=task_id, category=category)
    ua.mark_dispatched(attempt)
    ua.settle_attempt(
        attempt,
        {"prompt_tokens": 10, "completion_tokens": 5},
        cost_usd=cost,
        cost_final=True,
    )
    return attempt


def _seed_tree_accounting(root):
    """Root own spend + child spend + one disclosed-free and one undisclosed
    delegated session — the submarine 'where did the money go' shape."""
    _settled_attempt(root, task_id="root-1", cost=0.30)
    _settled_attempt(root, task_id="child-1", cost=0.50)
    ua.record_subscription_session(
        "session-free", drive_root=root, route="claudexor/claude",
        task_id="child-1", root_task_id="root-1", spend_usd=0.0,
    )
    ua.record_subscription_session(
        "session-undisclosed", drive_root=root, route="claudexor/codex",
        task_id="child-1", root_task_id="root-1", spend_usd=None,
    )


def test_usage_breakdown_delegated_axis_is_a_filter_not_a_new_sum(tmp_path, monkeypatch):
    root = _data_root(tmp_path, monkeypatch)
    _seed_tree_accounting(root)

    breakdown = ua.usage_breakdown(root, root_task_id="root-1")
    delegated = breakdown["delegated"]
    # The delegated bucket is the SAME subscription rows already counted in the
    # top-level summary (sessions axis), filtered by execution kind.
    assert delegated["subscription_sessions"] == 2
    assert breakdown["subscription_sessions"] == 2
    assert delegated["settled_usd"] == 0.0  # disclosed-free session
    assert delegated["unknown_unmetered"] == 1  # undisclosed spend, never $0
    assert delegated["physical_calls"] == 0  # sessions are not provider sends


def test_task_detail_cost_breakdown_view_for_root(tmp_path, monkeypatch):
    from ouroboros.gateway.tasks import _task_cost_breakdown_view

    root = _data_root(tmp_path, monkeypatch)
    _seed_tree_accounting(root)

    view = _task_cost_breakdown_view(root, {"task_id": "root-1", "root_task_id": "root-1"})
    assert view is not None
    assert view["own_usd"] == 0.30
    assert view["children_usd"] == 0.50  # subtree minus own — the by-hand subtraction, formalized
    assert view["delegated_disclosed_usd"] == 0.0
    assert view["subscription_sessions"] == 2
    assert view["unknown_unmetered"] == 1
    assert view["cost_final"] is False  # the undisclosed session keeps finality honest
    assert view["authority"] == "physical_attempt_ledger"


def test_task_detail_cost_breakdown_view_only_for_roots_and_fails_soft(tmp_path, monkeypatch):
    from ouroboros.gateway.tasks import _task_cost_breakdown_view

    root = _data_root(tmp_path, monkeypatch)
    _seed_tree_accounting(root)
    # Non-root: subtree math is not ledger-attributable mid-tree — omitted.
    assert _task_cost_breakdown_view(root, {"task_id": "child-1", "root_task_id": "root-1"}) is None
    # Unreadable ledger: absent view, never a confident $0 object.
    (root / ua.LEDGER_REL).write_text("not-json\n{}\n", encoding="utf-8")
    view = _task_cost_breakdown_view(root, {"task_id": "root-1", "root_task_id": "root-1"})
    assert view is None or view.get("cost_final") is False


def test_task_detail_cost_breakdown_view_absent_when_nothing_was_accounted(tmp_path, monkeypatch):
    """A READABLE but empty/legacy-only ledger must not publish a confident $0.

    `_summary()` always returns a float for `accounted_usd`, so availability
    decided on the dollar sum reports `own 0 / children 0 / cost_final true`
    for a root whose real money was never ledger-attributed — the v6.64.1
    "unknown rendered as a confident $0" class. Availability is decided on the
    attributable ROW COUNTS instead."""
    from ouroboros.gateway.tasks import _task_cost_breakdown_view

    root = _data_root(tmp_path, monkeypatch)
    empty = ua.usage_breakdown(root, root_task_id="root-1")
    # The trap: the ledger answers 0.0/final for a subtree it never measured.
    assert empty["accounted_usd"] == 0.0 and empty["cost_final"] is True
    assert _task_cost_breakdown_view(
        root, {"task_id": "root-1", "root_task_id": "root-1", "cost_usd": 42.75},
    ) is None

    # Legacy-imported money is real but carries no root_task_id, so the
    # root-filtered subtree is empty while the ledger holds dollars.
    legacy_root = tmp_path / "legacy"
    (legacy_root / "state").mkdir(parents=True)
    (legacy_root / "logs").mkdir(parents=True)
    (legacy_root / "state" / "state.json").write_text(
        json.dumps({"spent_usd": 42.75, "spent_calls": 12}), encoding="utf-8",
    )
    ua.ensure_legacy_imported(legacy_root)
    assert ua.usage_breakdown(legacy_root)["accounted_usd"] > 0.0
    assert _task_cost_breakdown_view(
        legacy_root, {"task_id": "root-1", "root_task_id": "root-1"},
    ) is None

    # One priced row makes the SAME subtree genuinely measured: the view returns
    # and a root that spent nothing itself keeps its honest measured zero.
    _settled_attempt(root, task_id="child-1", cost=0.25)
    view = _task_cost_breakdown_view(root, {"task_id": "root-1", "root_task_id": "root-1"})
    assert view is not None
    assert view["own_usd"] == 0.0 and view["children_usd"] == 0.25


def test_task_detail_unknown_zero_subtree_is_omitted_not_reported_free(tmp_path, monkeypatch):
    from ouroboros.gateway.tasks import _task_cost_breakdown_view

    root = _data_root(tmp_path, monkeypatch)
    ua.record_unmetered_external_dispatch(
        "detail-opaque", drive_root=root, provider="external-skill",
        task_id="root-1", root_task_id="root-1",
    )
    assert _task_cost_breakdown_view(
        root, {"task_id": "root-1", "root_task_id": "root-1"},
    ) is None


def test_task_detail_cost_breakdown_view_discloses_unattributed_money(tmp_path, monkeypatch):
    """Subtree money that no task id claims gets its own axis, not the children's."""
    from ouroboros.gateway.tasks import _task_cost_breakdown_view

    root = _data_root(tmp_path, monkeypatch)
    _settled_attempt(root, task_id="root-1", cost=0.30)
    _settled_attempt(root, task_id="", cost=0.20)
    view = _task_cost_breakdown_view(root, {"task_id": "root-1", "root_task_id": "root-1"})
    assert view is not None
    assert view["own_usd"] == 0.30
    assert view["unattributed_usd"] == 0.20
    # Not silently folded into the children's share; the three axes still sum.
    assert view["children_usd"] == 0.0
    assert round(view["own_usd"] + view["children_usd"] + view["unattributed_usd"], 6) == 0.50
