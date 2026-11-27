"""Observe Development's exact A -> B offer and open it using only the existing desktop UI."""

from __future__ import annotations

from typing import Any, Callable

import git_traces as gt
import prepare_build as pb
from windows_host import Unread

PHASE = "update_check"
PILL = {"automation_id": "update-pill"}
CHECK = {"automation_id": "nav-version-check", "name_prefix": "Check for updates"}
WARNING_CODES = frozenset({
    "remote_config_error", "fetch_error", "branch_error", "head_error", "status_error",
    "managed_update_requires_branch", "official_status_requires_check", "missing_remote",
    "target_ref_error", "divergence_parse_error", "divergence_error",
})


def offered_b(body: dict[str, Any], a: str, b: str, *, after: str = "") -> bool:
    return (gt.completed_check(body, target=b, development=True, after=after) is not None
            and body.get("current_sha") == body.get("running_sha") == a
            and body.get("latest_sha") == b
            and body.get("available") is True and body.get("safe_to_apply") is True)


def status_facts(body: dict[str, Any] | None, a: str, b: str) -> dict[str, Any]:
    """No values from arbitrary strings, URLs, exception detail, or revisions outside the fixture."""
    if body is None:
        return {"read": False}
    channel, warnings = body.get("update_channel"), body.get("warnings")
    return {
        "read": True,
        "channel": channel if channel in ("stable", "qa", "development") else "unknown",
        "available": body.get("available") is True,
        "safe_to_apply": body.get("safe_to_apply") is True,
        "current_matches_a": body.get("current_sha") == a,
        "running_matches_a": body.get("running_sha") == a,
        "checked_matches_b": body.get("checked_target_sha") == b,
        "latest_matches_b": body.get("latest_sha") == b,
        "check_time_readable": gt.when(body.get("checked_at")) is not None,
        "from_cache": body.get("from_cache") is True,
        "warnings": sorted({code for warning in warnings if isinstance(warning, str)
                            and (code := warning.partition(":")[0]) in WARNING_CODES}) if isinstance(warnings, list) else [],
    }


def open_development_offer(run: Any, poll_until: Callable[..., Any]) -> None:
    """Pick one UI door. A passive/preflight observation cannot create a manual-check mark."""
    driver, a, b = run.driver, run.candidate, run.b_sha
    hwnd, facts = driver.main_hwnd, run.result["facts"]

    def status(key: str) -> dict[str, Any]:
        body = run.api("/api/update/status")
        facts[key] = status_facts(body, a, b)
        return body or {}

    def nodes(query: dict[str, Any]) -> list[dict[str, Any]]:
        return driver.find_nodes(hwnd, phase=PHASE, **query)

    status("startup_update_status")  # diagnostic only; boot's Stable check may never settle
    # Dashboard opens on Logs: positive lookups can read from a cut tree there.
    tab = {"automation_id": "dashboard-tab-updates", "allow_truncated": True}
    dev = {"role": "button", "name_prefix": "Development", "require_visible": False, "allow_truncated": True}
    driver.act_or_click(hwnd, phase=PHASE, label="Dashboard", query={"role": "button", "name": "Dashboard"},
                        post_condition=lambda: bool(nodes(tab)))
    driver.act_or_click(hwnd, phase=PHASE, label="Dashboard > Updates", query=tab, verbs=("select", "invoke"),
                        post_condition=lambda: bool(nodes(dev)))
    driver.act_or_click(hwnd, phase=PHASE, label="Which updates > Development", query=dev, verbs=("toggle", "invoke"),
                        post_condition=lambda: (pb.read_json_retry(run.data / "settings.json") or {}).get("OUROBOROS_UPDATE_CHANNEL") == "development")
    run.check("development_channel_saved", True, phase=PHASE)
    run.ensure_main_chat_open(phase=PHASE, click_sidebar=True)

    def door() -> str | None:
        body = status("development_status_before_action")
        if body.get("update_channel") != "development" or body.get("current_sha") != a or body.get("running_sha") != a:
            return None
        pills = nodes(PILL)
        if pills or offered_b(body, a, b):
            return "already_offered" if len(pills) == 1 and offered_b(body, a, b) else None
        return "plain_check" if len(nodes(CHECK)) == 1 else None

    route = poll_until(45.0, door, interval=0.5)
    if route is None:
        raise Unread("development_update_control_unobserved")
    facts["development_offer_route"] = route
    if route == "plain_check":
        clicked = pb._utc_now()
        driver.act_or_click(hwnd, phase=PHASE, label="Sidebar Check for updates", query=CHECK,
                            select={"automation_id": CHECK["automation_id"]}, single_press_strict=True)
        run.result["marks"]["plain_check_clicked"] = clicked
        if not poll_until(120.0, lambda: offered_b(status("development_status_after_check"), a, b, after=clicked)
                          and len(nodes(PILL)) == 1, interval=1.0):
            raise Unread("plain_check_of_b_unobserved")
        run.check("plain_update_check_used_without_describe", True, phase=PHASE)
    else:
        run.mark("development_offer_observed")
    driver.act_or_click(hwnd, phase=PHASE, label="Sidebar update pill", query=PILL, select=PILL,
                        post_condition=lambda: bool(nodes({"automation_id": "btn-update-primary"})))
