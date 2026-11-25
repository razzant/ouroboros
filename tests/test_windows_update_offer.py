"""Qualification reads the real Update offer; these tests only supply inert status and driver objects."""

from __future__ import annotations

import pathlib
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "devtools" / "qualification" / "windows_update"))

import git_traces as gt  # noqa: E402
import update_offer as uo  # noqa: E402

A, B, C = (char * 40 for char in "abc")
CLICK = "2026-11-25T05:00:03Z"
CHECKED = "2026-11-25T05:00:04Z"
OFFER = {"update_channel": "development", "current_sha": A, "running_sha": A,
         "checked_at": CHECKED, "checked_target_sha": B, "latest_sha": B,
         "available": True, "safe_to_apply": True, "check_ok": None}
NO_OFFER = {key: OFFER[key] for key in ("update_channel", "current_sha", "running_sha")}


@pytest.mark.parametrize("patch", [
    {"update_channel": "stable"}, {"checked_target_sha": None}, {"latest_sha": C},
    {"running_sha": B}, {"current_sha": C}, {"available": 1}, {"safe_to_apply": False},
    {"checked_at": CLICK}, {"checked_at": "2026-11-25T05:00:04"},
])
def test_only_fresh_exact_development_offer_can_follow_plain_check(patch):
    assert uo.offered_b(OFFER, A, B, after=CLICK)
    assert not uo.offered_b({**OFFER, **patch}, A, B, after=CLICK)


def test_stable_is_exact_and_diagnostic_does_not_export_arbitrary_strings():
    stable = {"update_channel": "stable", "checked_target_sha": A, "checked_at": CHECKED, "check_ok": None}
    assert gt.completed_check(stable, target=A, development=False)
    for channel in (None, "qa", ""):
        assert gt.completed_check({**stable, "update_channel": channel}, target=A, development=False) is None
    packet = uo.status_facts({**stable, "update_channel": "unexpected", "remote": "private",
                              "warnings": ["fetch_error:private payload", "private warning"]}, A, B)
    assert packet["channel"] == "unknown" and packet["warnings"] == ["fetch_error"]
    assert "private" not in repr(packet)


class Driver:
    main_hwnd = 1

    def __init__(self, *, pill):
        self.pill, self.labels = pill, []

    def find_nodes(self, hwnd, **query):
        key = query.get("automation_id")
        return [{}] if {"update-pill": self.pill, "nav-version-check": not self.pill}.get(key, True) else []

    def act_or_click(self, hwnd, *, label, post_condition=None, **_):
        self.labels.append(label)
        if label == "Sidebar Check for updates":
            self.pill = True
        if post_condition is not None:
            assert post_condition()


def fake_run(monkeypatch, *, already, before=None, checked=OFFER):
    driver = Driver(pill=already)
    if before is None:
        before = OFFER if already else NO_OFFER
    reads = iter([None, before, checked])  # startup GET unavailable, then normal UI reads
    checked_names, marks, facts = [], {}, {}
    monkeypatch.setattr(uo.pb, "read_json_retry", lambda _: {"OUROBOROS_UPDATE_CHANNEL": "development"})
    monkeypatch.setattr(uo.pb, "_utc_now", lambda: CLICK)
    run = SimpleNamespace(driver=driver, api=lambda _: next(reads), candidate=A, b_sha=B, data=pathlib.Path("synthetic"),
                          result={"marks": marks, "facts": facts}, mark=lambda key: marks.update({key: CLICK}),
                          check=lambda key, _ok, **_: checked_names.append(key), ensure_main_chat_open=lambda **_: None)
    return run, checked_names


def once(_timeout, read, **_):
    return read()


@pytest.mark.parametrize("already", [False, True])
def test_update_enters_one_ui_door_even_without_observed_stable(monkeypatch, already):
    run, checks = fake_run(monkeypatch, already=already)
    uo.open_development_offer(run, once)
    assert run.result["facts"]["development_offer_route"] == ("already_offered" if already else "plain_check")
    assert ("plain_check_clicked" in run.result["marks"]) is not already
    assert ("development_offer_observed" in run.result["marks"]) is already
    assert ("plain_update_check_used_without_describe" in checks) is not already
    assert run.driver.labels[-1] == "Sidebar update pill"
    assert run.driver.labels.count("Sidebar Check for updates") == int(not already)


def test_existing_pill_does_not_authorize_wrong_target_or_force_manual_check(monkeypatch):
    run, _checks = fake_run(monkeypatch, already=True, before={**OFFER, "latest_sha": C})
    with pytest.raises(uo.Unread, match="development_update_control_unobserved"):
        uo.open_development_offer(run, once)
    assert not any(label.startswith("Sidebar ") for label in run.driver.labels)


def test_plain_button_alone_does_not_prove_fresh_check_or_authorize_update(monkeypatch):
    run, checks = fake_run(monkeypatch, already=False, checked={**OFFER, "checked_at": CLICK})
    with pytest.raises(uo.Unread, match="plain_check_of_b_unobserved"):
        uo.open_development_offer(run, once)
    assert run.driver.labels.count("Sidebar Check for updates") == 1
    assert "Sidebar update pill" not in run.driver.labels
    assert "plain_update_check_used_without_describe" not in checks


@pytest.mark.parametrize("route,phase", [("development_offer_observed", "preflight"), ("plain_check_clicked", "check")])
def test_only_the_selected_fetch_window_counts(route, phase):
    update_to, update_now = "2026-11-25T05:00:05Z", "2026-11-25T05:00:10Z"
    marks = {route: CLICK, "update_to_clicked": update_to, "update_now_clicked": update_now}
    bounds = gt.bounds(marks, {})
    assert bounds[phase] == ((update_to, update_now) if phase == "preflight" else (CLICK, update_to))
    assert {"check", "preflight"}.intersection(bounds) == {phase}

    def requirements(instant):
        instant = gt.when(instant)
        traces = {"sessions": {"fetch": {"start": instant, "cmd": "fetch", "argv": ["git", "fetch"],
                                          "ends": [(0, instant)], "signal": False}}}
        windows = {p: gt.window(traces, gt.when(start), gt.when(end)) for p, (start, end) in bounds.items()}
        return gt.phase_requirements(windows)

    early, late = CHECKED, "2026-11-25T05:00:07Z"
    assert requirements(early)[phase] is (phase == "check")
    assert requirements(late)[phase] is (phase == "preflight")
