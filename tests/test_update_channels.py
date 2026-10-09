from __future__ import annotations

import pathlib
import subprocess
import sys


REPO = pathlib.Path(__file__).resolve().parents[1]


def test_update_channel_mapping_is_closed_and_defaults_to_stable():
    from ouroboros import config
    from ouroboros import update_channels

    assert config.SETTINGS_DEFAULTS["OUROBOROS_UPDATE_CHANNEL"] == "stable"
    assert update_channels.UPDATE_CHANNEL_BRANCHES == {
        "stable": "main",
        "qa": "ouroboros-stable",
        "development": "ouroboros",
    }
    assert update_channels.get_update_branch({"OUROBOROS_UPDATE_CHANNEL": "stable"}) == "main"
    assert update_channels.get_update_branch({"OUROBOROS_UPDATE_CHANNEL": "qa"}) == "ouroboros-stable"
    assert update_channels.get_update_branch({"OUROBOROS_UPDATE_CHANNEL": "development"}) == "ouroboros"
    assert update_channels.get_update_channel({"OUROBOROS_UPDATE_CHANNEL": " NIGHTLY "}) == "stable"


def test_git_timeout_settings_share_defaults_clamps_and_env_projection(monkeypatch):
    from ouroboros import config
    from ouroboros import update_channels

    getters = {
        "OUROBOROS_MANAGED_UPDATE_FETCH_TIMEOUT_SEC": (
            update_channels.get_managed_update_fetch_timeout_sec
        ),
        "OUROBOROS_RESCUE_GIT_TIMEOUT_SEC": update_channels.get_rescue_git_timeout_sec,
    }
    for name, getter in getters.items():
        assert config.SETTINGS_DEFAULTS[name] == 300
        assert name in config.settings_env_keys()
        monkeypatch.delenv(name, raising=False)
        assert getter() == 300
        monkeypatch.setenv(name, "invalid")
        assert getter() == 300
        monkeypatch.setenv(name, "1")
        assert getter() == 30
        monkeypatch.setenv(name, "9999")
        assert getter() == 1800


def test_update_channel_setting_is_exposed_and_round_trips_in_ui():
    settings_js = (REPO / "web" / "modules" / "settings.js").read_text(encoding="utf-8")
    settings_ui = (REPO / "web" / "modules" / "settings_ui.js").read_text(encoding="utf-8")
    settings_css = (REPO / "web" / "settings.css").read_text(encoding="utf-8")

    assert "['s-update-channel', 'OUROBOROS_UPDATE_CHANNEL', 'stable']" in settings_js
    assert 'id="s-update-channel"' in settings_ui
    assert "{ value: 'stable', label: 'Stable' }" in settings_ui
    assert "{ value: 'qa', label: 'QA' }" in settings_ui
    assert "{ value: 'development', label: 'Development' }" in settings_ui
    # Three choices take three equal columns from the shared count-derived grid,
    # not from a per-control override (tests/test_segmented_layout_browser.py).
    assert "[data-update-channel-group]" not in settings_css
    assert "--segment-columns: min(var(--segment-count, 4), 4)" in settings_css


def test_update_source_can_be_imported_first_in_a_clean_process():
    result = subprocess.run(
        [sys.executable, "-c", "import supervisor.update_source"],
        cwd=REPO,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_stable_target_is_latest_release_present_in_main_and_qa():
    from supervisor.update_source import resolve_managed_update_target

    def capture(cmd):
        if cmd[:3] == ["git", "rev-parse", "--verify"]:
            ref = cmd[3]
            values = {
                "managed/main^{commit}": "main-sha",
                "managed/ouroboros-stable^{commit}": "qa-sha",
                "refs/ouroboros-managed/tags/v6.88.0^{commit}": "sha-6880",
                "refs/ouroboros-managed/tags/v6.87.5^{commit}": "sha-6875",
            }
            return (0, values[ref], "") if ref in values else (1, "", "missing")
        if cmd == [
            "git",
            "for-each-ref",
            "--format=%(refname:strip=3)",
            "refs/ouroboros-managed/tags/v*",
        ]:
            return 0, "v6.87.5\nv6.88.0-rc.1\nv6.88.0", ""
        if cmd[:3] == ["git", "merge-base", "--is-ancestor"]:
            sha, branch = cmd[3], cmd[4]
            if sha == "sha-6880" and branch == "managed/ouroboros-stable":
                return 1, "", ""
            return 0, "", ""
        raise AssertionError(cmd)

    target_ref, target_sha, error = resolve_managed_update_target(
        "managed",
        "main",
        "managed/main",
        update_channel="stable",
        capture=capture,
    )

    assert error == ""
    assert target_ref == "refs/ouroboros-managed/tags/v6.87.5"
    assert target_sha == "sha-6875"


def test_non_stable_target_follows_selected_branch_tip():
    from supervisor.update_source import resolve_managed_update_target

    seen = []

    def capture(cmd):
        seen.append(cmd)
        return 0, "qa-tip", ""

    target_ref, target_sha, error = resolve_managed_update_target(
        "managed",
        "ouroboros-stable",
        "managed/ouroboros-stable",
        update_channel="qa",
        capture=capture,
    )

    assert (target_ref, target_sha, error) == (
        "managed/ouroboros-stable",
        "qa-tip",
        "",
    )
    assert seen == [["git", "rev-parse", "--verify", "managed/ouroboros-stable^{commit}"]]
