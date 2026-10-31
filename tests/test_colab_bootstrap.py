from __future__ import annotations

import ast
import json
import os
import pathlib
import subprocess
from types import SimpleNamespace

import pytest


def _stage_constitution(repo):
    (repo / "BIBLE.md").write_text("constitution\n", encoding="utf-8")
    subprocess.run(["git", "add", "BIBLE.md"], cwd=repo, check=True, capture_output=True)


def _publish_release(repo, tag):
    subprocess.run(
        ["git", "branch", "-f", "ouroboros-stable", "HEAD"],
        cwd=repo,
        check=True,
        capture_output=True,
    )
    subprocess.run(["git", "tag", tag, "HEAD"], cwd=repo, check=True, capture_output=True)


def _load_quickstart_stable_selector():
    source = pathlib.Path(__file__).resolve().parents[1].joinpath(
        "notebooks", "colab_quickstart.py"
    ).read_text(encoding="utf-8")
    tree = ast.parse(source)
    wanted = {
        "_BOOTSTRAP_MANAGED_TAG_NAMESPACE",
        "_BOOTSTRAP_RELEASE_TAG_RE",
    }
    body = [
        node
        for node in tree.body
        if (
            isinstance(node, (ast.Import, ast.ImportFrom))
            and any(alias.name in {"re", "subprocess"} for alias in node.names)
        )
        or (
            isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id in wanted for target in node.targets)
        )
        or (
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name in {"_bootstrap_capture", "_bootstrap_stable_ref"}
        )
    ]
    namespace = {}
    exec(compile(ast.Module(body=body, type_ignores=[]), "colab-selector", "exec"), namespace)
    return namespace["_bootstrap_stable_ref"]


def test_build_colab_settings_defaults_auto_grant_and_runtime():
    from ouroboros.colab_bootstrap import build_colab_settings, masked_secret_status
    settings = build_colab_settings({"OPENROUTER_API_KEY": "or-key", "TELEGRAM_BOT_TOKEN": "tg-token", "GITHUB_TOKEN": "gh-token"}, github_repo="anton/ouroboros", total_budget=25, runtime_mode="pro", max_workers=2)
    assert settings["GITHUB_REPO"] == "anton/ouroboros"
    assert settings["OUROBOROS_AUTO_GRANT_REVIEWED_SKILLS"] == "true"
    assert settings["OUROBOROS_UPDATE_CHANNEL"] == "stable"
    assert masked_secret_status(settings)["TELEGRAM_BOT_TOKEN"] is True


def test_build_colab_settings_uses_shipped_budget_defaults():
    from ouroboros.colab_bootstrap import build_colab_settings
    from ouroboros.config import SETTINGS_DEFAULTS

    settings = build_colab_settings({"OPENROUTER_API_KEY": "or-key"})

    assert settings["TOTAL_BUDGET"] == SETTINGS_DEFAULTS["TOTAL_BUDGET"] == 200.0
    assert (
        settings["OUROBOROS_PER_TASK_COST_USD"]
        == SETTINGS_DEFAULTS["OUROBOROS_PER_TASK_COST_USD"]
        == 50.0
    )


def test_build_colab_settings_merges_existing_owner_choices():
    # A Colab re-run must preserve prior owner choices not set by the launch knobs
    # (pinned chat, tweaked model) and drop private sentinel keys.
    from ouroboros.colab_bootstrap import build_colab_settings
    existing = {
        "TELEGRAM_CHAT_ID": "12345",
        "OUROBOROS_MODEL": "custom/model",
        "OUROBOROS_UPDATE_CHANNEL": "qa",
        "_settings_file_exists": True,
    }
    out = build_colab_settings({"OPENROUTER_API_KEY": "k"}, existing=existing)
    assert out["TELEGRAM_CHAT_ID"] == "12345"
    assert out["OUROBOROS_MODEL"] == "custom/model"
    assert out["OUROBOROS_UPDATE_CHANNEL"] == "qa"
    assert "_settings_file_exists" not in out
    assert out["OPENROUTER_API_KEY"] == "k"


def test_build_colab_settings_accepts_vision_model_override():
    from ouroboros.colab_bootstrap import build_colab_settings

    out = build_colab_settings(
        {"OPENROUTER_API_KEY": "k"},
        models={"OUROBOROS_MODEL_VISION": "google/gemini-2.5-pro"},
    )
    assert out["OUROBOROS_MODEL_VISION"] == "google/gemini-2.5-pro"


# --- the three launch knobs: explicit launch value > what Drive saved > the first-launch triple ----------------

KNOBS = ("TOTAL_BUDGET", "OUROBOROS_RUNTIME_MODE", "OUROBOROS_MAX_WORKERS")
FIRST_LAUNCH = (200.0, "advanced", 1)
SAVED_ON_DRIVE = {"TOTAL_BUDGET": 999.0, "OUROBOROS_RUNTIME_MODE": "pro", "OUROBOROS_MAX_WORKERS": 8,
                  "TELEGRAM_CHAT_ID": "12345", "OUROBOROS_MODEL": "custom/model"}


def _knobs(settings):
    return tuple(settings[key] for key in KNOBS)


def _notebook_launch(monkeypatch, *, environment, drive):
    """The notebook's own launch statements (``secrets`` through ``settings = build_colab_settings(...)``),
    parsed from ``notebooks/colab_quickstart.py`` and run unchanged against the real builder: the cell's
    environment handling is part of the contract, not only the builder behind it."""
    from ouroboros.colab_bootstrap import build_colab_settings

    source = pathlib.Path(__file__).resolve().parents[1].joinpath("notebooks", "colab_quickstart.py").read_text(encoding="utf-8")
    body = ast.parse(source).body

    def assigns(node, name):
        return isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets)

    start = next(i for i, node in enumerate(body) if assigns(node, "secrets"))
    end = next(i for i, node in enumerate(body) if assigns(node, "settings"))
    launch = compile(ast.Module(body=body[start:end + 1], type_ignores=[]), "colab_quickstart.py", "exec")
    for name in KNOBS:
        monkeypatch.delenv(name, raising=False)
    for name, value in environment.items():
        monkeypatch.setenv(name, value)
    # What the cell hands the builder before its launch statements (it always names the channel).
    scope = {"os": os, "collect_colab_secrets": lambda: {"OPENROUTER_API_KEY": "k"},
             "build_colab_settings": build_colab_settings,
             "_settings_seed": {**drive, "OUROBOROS_UPDATE_CHANNEL": "stable"}, "_existing_settings": dict(drive)}
    exec(launch, scope)  # noqa: S102 -- the repo's own notebook source
    return scope["settings"]


@pytest.mark.parametrize(("environment", "expected"), [
    ({}, (999.0, "pro", 8)),  # nothing explicit: every saved choice survives
    ({"TOTAL_BUDGET": "75"}, (75.0, "pro", 8)),  # one explicit knob moves one knob
    ({"OUROBOROS_RUNTIME_MODE": "cyber_pro"}, (999.0, "cyber_pro", 8)),
    ({"OUROBOROS_MAX_WORKERS": "4"}, (999.0, "pro", 4)),
    ({"TOTAL_BUDGET": "75", "OUROBOROS_RUNTIME_MODE": "light", "OUROBOROS_MAX_WORKERS": "4"}, (75.0, "light", 4)),
    # An explicit value equal to the first-launch default is still a choice, and beats what Drive saved.
    ({"TOTAL_BUDGET": "200", "OUROBOROS_RUNTIME_MODE": "advanced", "OUROBOROS_MAX_WORKERS": "1"}, FIRST_LAUNCH),
    ({"TOTAL_BUDGET": "200"}, (200.0, "pro", 8)),
])
def test_notebook_launch_keeps_saved_knobs_unless_one_is_explicit(monkeypatch, environment, expected):
    out = _notebook_launch(monkeypatch, environment=environment, drive=SAVED_ON_DRIVE)
    assert _knobs(out) == expected
    # Everything else the Drive document holds survives, and the launch's own host rule is untouched.
    assert out["TELEGRAM_CHAT_ID"] == "12345" and out["OUROBOROS_MODEL"] == "custom/model"
    assert out["OUROBOROS_SERVER_HOST"] == "127.0.0.1"


@pytest.mark.parametrize(("drive", "expected"), [
    ({}, FIRST_LAUNCH),  # a first launch: no Drive document at all
    ({"TELEGRAM_CHAT_ID": "12345"}, FIRST_LAUNCH),  # a document that holds none of the three
    ({"TOTAL_BUDGET": 999.0}, (999.0, "advanced", 1)),  # a document holding one keeps it, the rest are first-launch
    ({"OUROBOROS_MAX_WORKERS": 8}, (200.0, "advanced", 8)),
])
def test_notebook_first_launch_takes_the_triple_only_for_what_drive_does_not_hold(monkeypatch, drive, expected):
    assert _knobs(_notebook_launch(monkeypatch, environment={}, drive=drive)) == expected


def test_a_repeat_launch_through_drive_keeps_the_saved_triple_and_an_app_edit_between_launches(monkeypatch, tmp_path):
    """The Drive document the first launch wrote is what the second reads; a choice made in the app between
    the two launches (Drive changes, the kernel's environment does not) is the one that survives the repeat."""
    from unittest import mock

    from ouroboros.colab_bootstrap import export_colab_env, write_colab_settings

    data = tmp_path / "data"
    with mock.patch.dict(os.environ):  # export_colab_env updates the process environment
        first = _notebook_launch(monkeypatch, environment={}, drive=SAVED_ON_DRIVE)
        path = write_colab_settings(data, first)
        exported = export_colab_env(tmp_path / "repo", data, path)
        assert not set(exported) & set(KNOBS), "the exporter never turns a built knob into a launch variable"
        assert not set(KNOBS) & set(os.environ), "so the kernel's environment still holds no knob"
        saved = json.loads(path.read_text(encoding="utf-8"))
        assert _knobs(saved) == (999.0, "pro", 8)
        repeat = _notebook_launch(monkeypatch, environment={}, drive=saved)
        assert _knobs(repeat) == (999.0, "pro", 8)
        # The owner changes all three in the app; the next launch of the same kernel keeps THEIR values.
        edited = {**saved, "TOTAL_BUDGET": 123.0, "OUROBOROS_RUNTIME_MODE": "light", "OUROBOROS_MAX_WORKERS": 6}
        assert _knobs(_notebook_launch(monkeypatch, environment={}, drive=edited)) == (123.0, "light", 6)


@pytest.mark.parametrize(("environment", "error"), [
    ({"TOTAL_BUDGET": ""}, ValueError), ({"TOTAL_BUDGET": "lots"}, ValueError),
    ({"OUROBOROS_MAX_WORKERS": ""}, ValueError), ({"OUROBOROS_MAX_WORKERS": "many"}, ValueError),
])
def test_a_set_but_blank_or_malformed_launch_variable_still_fails_as_it_did(monkeypatch, environment, error):
    """Unset is "not given"; a set variable is a choice and goes through the same ``float``/``int`` as before."""
    with pytest.raises(error):
        _notebook_launch(monkeypatch, environment=environment, drive=SAVED_ON_DRIVE)


def test_an_explicit_runtime_mode_is_stored_as_given_and_read_through_the_existing_normalization(monkeypatch):
    from ouroboros.config import normalize_settings_raw

    out = _notebook_launch(monkeypatch, environment={"OUROBOROS_RUNTIME_MODE": "bogus"}, drive=SAVED_ON_DRIVE)
    assert out["OUROBOROS_RUNTIME_MODE"] == "bogus", "never silently replaced by the saved 'pro', never treated as unset"
    assert normalize_settings_raw(dict(out))["OUROBOROS_RUNTIME_MODE"] == "advanced", "the reader's closed enum decides"


def test_build_colab_settings_direct_callers_follow_the_same_precedence():
    from ouroboros.colab_bootstrap import build_colab_settings

    assert _knobs(build_colab_settings({}, existing=SAVED_ON_DRIVE)) == (999.0, "pro", 8)
    assert _knobs(build_colab_settings({})) == FIRST_LAUNCH
    assert _knobs(build_colab_settings({}, existing=SAVED_ON_DRIVE, total_budget=25, runtime_mode="cyber_pro", max_workers=3)) == (25.0, "cyber_pro", 3)
    # An explicit zero-like value is still explicit (not "unset").
    assert build_colab_settings({}, existing=SAVED_ON_DRIVE, total_budget=0.0)["TOTAL_BUDGET"] == 0.0
    # Credentials keep their rule: a blank collected secret never wipes the saved one.
    out = build_colab_settings({"OPENROUTER_API_KEY": ""}, existing={**SAVED_ON_DRIVE, "OPENROUTER_API_KEY": "saved-key"})
    assert out["OPENROUTER_API_KEY"] == "saved-key"


def test_build_colab_settings_normalizes_explicit_update_channel():
    from ouroboros.colab_bootstrap import build_colab_settings

    development = build_colab_settings(
        {"OPENROUTER_API_KEY": "k"},
        existing={"OUROBOROS_UPDATE_CHANNEL": " DEVELOPMENT "},
    )
    invalid = build_colab_settings(
        {"OPENROUTER_API_KEY": "k"},
        existing={"OUROBOROS_UPDATE_CHANNEL": "unknown"},
    )
    assert development["OUROBOROS_UPDATE_CHANNEL"] == "development"
    assert invalid["OUROBOROS_UPDATE_CHANNEL"] == "stable"

def test_quickstart_waits_on_the_server_only_while_its_process_lives():
    """The boot wait has no clock: the notebook hands it the server process's liveness, so a server
    that exits before it answers ends the wait with that exit instead of retrying forever."""
    import ast

    source = pathlib.Path(__file__).resolve().parents[1].joinpath("notebooks", "colab_quickstart.py").read_text(encoding="utf-8")
    [call] = [node for node in ast.walk(ast.parse(source)) if isinstance(node, ast.Call)
              and getattr(node.func, "id", "") == "ensure_native_telegram_live"]
    alive = {keyword.arg: keyword.value for keyword in call.keywords}["alive"]
    scope = {}
    predicate = eval(compile(ast.Expression(alive), "colab_quickstart.py", "eval"), scope)  # noqa: S307 -- the repo's own source
    for returncode, lives in ((None, True), (0, False), (1, False)):
        scope["server"] = type("Process", (), {"poll": staticmethod(lambda code=returncode: code)})
        assert predicate() is lives, (returncode, lives)


def test_quickstart_uses_clone_or_update_repo_helper():
    source = pathlib.Path(__file__).resolve().parents[1].joinpath("notebooks", "colab_quickstart.py").read_text(encoding="utf-8")
    assert "clone_or_update_repo" in source
    assert '"stable": "main"' in source
    assert '"qa": "ouroboros-stable"' in source
    assert '"development": "ouroboros"' in source
    bootstrap_pos = source.index("_bootstrap_channel = str(")
    initial_clone_pos = source.index('"clone", "--no-checkout"')
    stable_resolve_pos = source.index("_initial_source_ref = _bootstrap_stable_ref(REPO_DIR)")
    initial_check_pos = source.index("if not _bootstrap_ref_has_constitution(REPO_DIR, _initial_source_ref)")
    detached_pos = source.index('"checkout", "--detach", _initial_source_ref')
    legacy_check_pos = source.index('if not _bootstrap_ref_has_constitution(REPO_DIR, "FETCH_HEAD")')
    legacy_merge_pos = source.index('["git", "merge", "--ff-only", "FETCH_HEAD"]')
    import_pos = source.index("from ouroboros.colab_bootstrap import")
    channel_pos = source.index("_requested_channel = normalize_update_channel")
    clone_pos = source.index("clone_or_update_repo(\n    REPO_DIR", channel_pos)
    pip_pos = source.index('"pip", "install"', clone_pos)
    assert bootstrap_pos < initial_clone_pos < stable_resolve_pos < initial_check_pos < detached_pos < import_pos
    assert legacy_check_pos < legacy_merge_pos < import_pos < channel_pos < clone_pos < pip_pos
    assert "source_branch=_source_branch" in source
    assert 'local_branch="ouroboros"' in source
    assert 'not (REPO_DIR / "ouroboros" / "update_channels.py").is_file()' in source
    assert '["fetch", "https://github.com/razzant/ouroboros.git", "ouroboros"]' in source
    assert "if not isinstance(_existing_settings, dict):" in source


def test_quickstart_stable_selector_ignores_untagged_main_tip(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "t@example.com"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "T"], cwd=repo, check=True)
    (repo / "BIBLE.md").write_text("constitution\n", encoding="utf-8")
    subprocess.run(["git", "add", "BIBLE.md"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "release"], cwd=repo, check=True)
    release_sha = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, check=True, capture_output=True, text=True
    ).stdout.strip()
    subprocess.run(
        ["git", "update-ref", "refs/remotes/origin/ouroboros-stable", release_sha],
        cwd=repo,
        check=True,
    )
    subprocess.run(
        ["git", "update-ref", "refs/ouroboros-managed/tags/v1.2.3", release_sha],
        cwd=repo,
        check=True,
    )
    (repo / "unreleased.txt").write_text("main tip\n", encoding="utf-8")
    subprocess.run(["git", "add", "unreleased.txt"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "unreleased main"], cwd=repo, check=True)
    subprocess.run(
        ["git", "update-ref", "refs/remotes/origin/main", "HEAD"], cwd=repo, check=True
    )

    selected = _load_quickstart_stable_selector()(repo)

    assert selected == "refs/ouroboros-managed/tags/v1.2.3"
    selected_sha = subprocess.run(
        ["git", "rev-parse", f"{selected}^{{commit}}"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert selected_sha == release_sha


@pytest.mark.parametrize(
    ("entry", "expected"),
    [
        ("100644 blob deadbeef 12\tBIBLE.md\n", True),
        ("100755 blob deadbeef 12\tBIBLE.md\n", True),
        ("100644 blob deadbeef 0\tBIBLE.md\n", False),
        ("120000 blob deadbeef 12\tBIBLE.md\n", False),
        ("", False),
    ],
)
def test_official_ref_constitution_predicate(monkeypatch, tmp_path, entry, expected):
    import supervisor.update_source as update_source

    monkeypatch.setattr(
        update_source.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(returncode=0, stdout=entry),
    )

    assert update_source.official_ref_has_constitution(
        "managed/main", repo_dir=tmp_path
    ) is expected


def test_clone_or_update_repo_rejects_target_without_constitution_before_checkout(tmp_path):
    from ouroboros.colab_bootstrap import clone_or_update_repo

    upstream = tmp_path / "upstream"
    upstream.mkdir()
    subprocess.run(["git", "init"], cwd=upstream, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "t@example.com"], cwd=upstream, check=True)
    subprocess.run(["git", "config", "user.name", "T"], cwd=upstream, check=True)
    subprocess.run(["git", "checkout", "-b", "main"], cwd=upstream, check=True, capture_output=True)
    (upstream / "marker.txt").write_text("unsafe\n", encoding="utf-8")
    subprocess.run(["git", "add", "marker.txt"], cwd=upstream, check=True)
    subprocess.run(["git", "commit", "-m", "missing constitution"], cwd=upstream, check=True, capture_output=True)
    _publish_release(upstream, "v0.0.1")

    checkout = tmp_path / "checkout"
    with pytest.raises(RuntimeError, match="lacks a non-empty regular BIBLE.md"):
        clone_or_update_repo(checkout, source_url=str(upstream))

    assert not (checkout / "marker.txt").exists()


def test_colab_network_timeout_kills_process_tree(monkeypatch):
    import ouroboros.colab_bootstrap as bootstrap
    import ouroboros.platform_layer as platform_layer

    killed = []

    class HungProcess:
        returncode = 1

        def __init__(self):
            self.calls = 0

        def communicate(self, timeout=None):
            self.calls += 1
            if self.calls == 1:
                raise subprocess.TimeoutExpired(["git", "clone"], timeout)
            return "", ""

    proc = HungProcess()
    monkeypatch.setattr(bootstrap.subprocess, "Popen", lambda *_a, **_k: proc)
    monkeypatch.setattr(bootstrap, "get_managed_update_fetch_timeout_sec", lambda: 0.01)
    monkeypatch.setattr(platform_layer, "kill_process_tree", lambda value: killed.append(value))

    with pytest.raises(RuntimeError, match="timed out"):
        bootstrap._run_colab_git_network(["clone", "upstream", "checkout"])

    assert killed == [proc]


def test_personal_origin_restore_uses_bounded_network_helper(tmp_path, monkeypatch):
    import ouroboros.colab_bootstrap as bootstrap

    calls = []
    monkeypatch.setattr(
        bootstrap,
        "_run_colab_git_network",
        lambda args, cwd=None: calls.append((args, cwd)),
    )

    class Result:
        returncode = 0

    monkeypatch.setattr(bootstrap.subprocess, "run", lambda *_a, **_k: Result())

    assert bootstrap._ff_to_origin_if_ahead(tmp_path, "ouroboros") is True
    assert calls == [(["fetch", "origin", "ouroboros"], tmp_path)]

def test_clone_or_update_repo_fast_forwards_existing_checkout(tmp_path, monkeypatch):
    from ouroboros.colab_bootstrap import clone_or_update_repo
    upstream = tmp_path / "upstream"; upstream.mkdir()
    subprocess.run(["git", "init"], cwd=upstream, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "t@example.com"], cwd=upstream, check=True)
    subprocess.run(["git", "config", "user.name", "T"], cwd=upstream, check=True)
    subprocess.run(["git", "checkout", "-b", "main"], cwd=upstream, check=True, capture_output=True)
    _stage_constitution(upstream)
    (upstream / "marker.txt").write_text("v1\n", encoding="utf-8")
    subprocess.run(["git", "add", "marker.txt"], cwd=upstream, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "v1"], cwd=upstream, check=True, capture_output=True)
    _publish_release(upstream, "v1.0.0")
    checkout = tmp_path / "checkout"
    clone_or_update_repo(
        checkout,
        source_url=str(upstream),
        source_branch="main",
        local_branch="ouroboros",
    )
    fallback_sha = subprocess.run(
        ["git", "rev-parse", "ouroboros-stable"],
        cwd=checkout,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    (upstream / "marker.txt").write_text("v2\n", encoding="utf-8")
    subprocess.run(["git", "add", "marker.txt"], cwd=upstream, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "v2"], cwd=upstream, check=True, capture_output=True)
    subprocess.run(
        ["git", "branch", "-f", "ouroboros-stable", "HEAD"],
        cwd=upstream,
        check=True,
        capture_output=True,
    )
    subprocess.run(["git", "tag", "v1.0.1", "HEAD"], cwd=upstream, check=True, capture_output=True)
    clone_or_update_repo(
        checkout,
        source_url=str(upstream),
        source_branch="main",
        local_branch="ouroboros",
    )
    assert (checkout / "marker.txt").read_text(encoding="utf-8") == "v2\n"
    assert subprocess.run(
        ["git", "rev-parse", "ouroboros-stable"],
        cwd=checkout,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip() == fallback_sha
    assert subprocess.run(
        ["git", "rev-parse", "ouroboros"],
        cwd=checkout,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip() != fallback_sha
    branch = subprocess.run(
        ["git", "branch", "--show-current"],
        cwd=checkout,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert branch == "ouroboros"
    assert subprocess.run(
        ["git", "config", "user.name"],
        cwd=checkout,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip() == "Ouroboros"
    assert subprocess.run(
        ["git", "config", "user.email"],
        cwd=checkout,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip() == "ouroboros@local.mac"
    meta = json.loads((checkout / ".git" / "ouroboros-managed.json").read_text(encoding="utf-8"))
    assert meta["source_branch"] == "main"
    assert meta["managed_local_branch"] == "ouroboros"
    assert meta["colab_source_mode"] is True
    import ouroboros.update_channels as update_channels
    import supervisor.git_ops as git_ops

    monkeypatch.setattr(git_ops, "REPO_DIR", checkout)
    monkeypatch.setattr(update_channels, "get_update_branch", lambda settings=None: "ouroboros-stable")
    assert git_ops._managed_update_target() == (
        "managed",
        "ouroboros-stable",
        "managed/ouroboros-stable",
    )


def test_clone_or_update_repo_keeps_local_branch_across_all_channels(tmp_path):
    from ouroboros.colab_bootstrap import clone_or_update_repo

    upstream = tmp_path / "upstream"
    upstream.mkdir()
    subprocess.run(["git", "init"], cwd=upstream, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "t@example.com"], cwd=upstream, check=True)
    subprocess.run(["git", "config", "user.name", "T"], cwd=upstream, check=True)
    subprocess.run(["git", "checkout", "-b", "main"], cwd=upstream, check=True, capture_output=True)
    _stage_constitution(upstream)
    (upstream / "channel.txt").write_text("stable\n", encoding="utf-8")
    subprocess.run(["git", "add", "channel.txt"], cwd=upstream, check=True)
    subprocess.run(["git", "commit", "-m", "stable"], cwd=upstream, check=True, capture_output=True)
    _publish_release(upstream, "v1.0.0")
    for source_branch, marker in (("ouroboros-stable", "qa\n"), ("ouroboros", "development\n")):
        subprocess.run(["git", "checkout", "-B", source_branch, "main"], cwd=upstream, check=True, capture_output=True)
        (upstream / "channel.txt").write_text(marker, encoding="utf-8")
        subprocess.run(["git", "add", "channel.txt"], cwd=upstream, check=True)
        subprocess.run(["git", "commit", "-m", source_branch], cwd=upstream, check=True, capture_output=True)

    for source_branch, marker in (
        ("main", "stable\n"),
        ("ouroboros-stable", "qa\n"),
        ("ouroboros", "development\n"),
    ):
        checkout = tmp_path / f"checkout-{source_branch}"
        clone_or_update_repo(
            checkout,
            source_url=str(upstream),
            source_branch=source_branch,
            local_branch="ouroboros",
        )
        assert (checkout / "channel.txt").read_text(encoding="utf-8") == marker
        current = subprocess.run(
            ["git", "branch", "--show-current"],
            cwd=checkout,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        assert current == "ouroboros"
        fallback_marker = subprocess.run(
            ["git", "show", "ouroboros-stable:channel.txt"],
            cwd=checkout,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        assert fallback_marker == "stable\n"

    # Match the notebook's two-stage QA bootstrap: importing from a detached
    # selected channel must not consume the local stable recovery branch name.
    checkout = tmp_path / "checkout-qa-bootstrap"
    subprocess.run(
        ["git", "clone", "--no-checkout", str(upstream), str(checkout)],
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "checkout", "--detach", "origin/ouroboros-stable"],
        cwd=checkout,
        check=True,
        capture_output=True,
    )
    clone_or_update_repo(
        checkout,
        source_url=str(upstream),
        source_branch="ouroboros-stable",
        local_branch="ouroboros",
    )
    stable = subprocess.run(
        ["git", "rev-parse", "ouroboros-stable"],
        cwd=checkout,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    main = subprocess.run(
        ["git", "rev-parse", "managed/main"],
        cwd=checkout,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert stable == main


def test_clone_or_update_repo_does_not_silently_use_stale_ref_after_fetch_failure(tmp_path):
    from ouroboros.colab_bootstrap import clone_or_update_repo

    upstream = tmp_path / "upstream"
    upstream.mkdir()
    subprocess.run(["git", "init"], cwd=upstream, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "t@example.com"], cwd=upstream, check=True)
    subprocess.run(["git", "config", "user.name", "T"], cwd=upstream, check=True)
    subprocess.run(["git", "checkout", "-b", "main"], cwd=upstream, check=True, capture_output=True)
    _stage_constitution(upstream)
    (upstream / "marker.txt").write_text("fresh\n", encoding="utf-8")
    subprocess.run(["git", "add", "marker.txt"], cwd=upstream, check=True)
    subprocess.run(["git", "commit", "-m", "fresh"], cwd=upstream, check=True, capture_output=True)
    _publish_release(upstream, "v1.0.0")
    checkout = tmp_path / "checkout"
    clone_or_update_repo(checkout, source_url=str(upstream))

    with pytest.raises(RuntimeError, match="official update fetch failed"):
        clone_or_update_repo(
            checkout,
            source_url=str(tmp_path / "missing-upstream"),
            source_branch="main",
        )
    assert (checkout / "marker.txt").read_text(encoding="utf-8") == "fresh\n"


def test_clone_or_update_repo_preserves_local_ahead_when_switching_channel(tmp_path):
    from ouroboros.colab_bootstrap import clone_or_update_repo

    upstream = tmp_path / "upstream"
    upstream.mkdir()
    subprocess.run(["git", "init"], cwd=upstream, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "t@example.com"], cwd=upstream, check=True)
    subprocess.run(["git", "config", "user.name", "T"], cwd=upstream, check=True)
    subprocess.run(["git", "checkout", "-b", "main"], cwd=upstream, check=True, capture_output=True)
    _stage_constitution(upstream)
    (upstream / "channel.txt").write_text("stable\n", encoding="utf-8")
    subprocess.run(["git", "add", "channel.txt"], cwd=upstream, check=True)
    subprocess.run(["git", "commit", "-m", "stable"], cwd=upstream, check=True, capture_output=True)
    _publish_release(upstream, "v1.0.0")
    subprocess.run(["git", "checkout", "-b", "ouroboros"], cwd=upstream, check=True, capture_output=True)
    (upstream / "channel.txt").write_text("development\n", encoding="utf-8")
    subprocess.run(["git", "add", "channel.txt"], cwd=upstream, check=True)
    subprocess.run(["git", "commit", "-m", "development"], cwd=upstream, check=True, capture_output=True)

    checkout = tmp_path / "checkout"
    clone_or_update_repo(
        checkout,
        source_url=str(upstream),
        source_branch="ouroboros",
    )
    clone_or_update_repo(
        checkout,
        source_url=str(upstream),
        source_branch="main",
    )

    assert (checkout / "channel.txt").read_text(encoding="utf-8") == "development\n"
    current = subprocess.run(
        ["git", "branch", "--show-current"],
        cwd=checkout,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert current == "ouroboros"

def test_get_colab_secret_optional_returns_empty_without_prompt(monkeypatch):
    from ouroboros.colab_bootstrap import get_colab_secret
    monkeypatch.delenv("OUROBOROS_TEST_ABSENT_KEY", raising=False)
    # required=False must never block on getpass when the secret is absent.
    assert get_colab_secret("OUROBOROS_TEST_ABSENT_KEY", required=False) == ""

def _native_telegram_index(*, missing=None, conflict=None):
    return {
        "skills": [{
            "name": "telegram",
            "source": "native",
            "review_profile": "native_seed",
            "executable_review": True,
            "review_stale": False,
            "conflict": conflict,
            "grants": {
                "missing_keys": list(missing or []),
                "missing_permissions": [],
            },
        }]
    }


def test_ensure_native_telegram_grants_enables_and_sets_poc_modes(monkeypatch):
    from ouroboros.colab_bootstrap import ensure_native_telegram_live
    calls = []
    index_calls = 0
    monkeypatch.setattr("ouroboros.colab_bootstrap.time.sleep", lambda _seconds: None)
    def fake_request(method, path, body=None, timeout=None):
        nonlocal index_calls
        calls.append((method, path, body, timeout))
        if path == "/api/health":
            return 200, {"ok": True}
        if path == "/api/extensions":
            index_calls += 1
            payload = _native_telegram_index(missing=["TELEGRAM_BOT_TOKEN"])
            payload["skills"][0]["grants"]["missing_permissions"] = ["subscribe_event:chat.outbound"]
            if index_calls == 1:
                payload["skills"][0]["executable_review"] = False
                payload["skills"][0]["review_stale"] = True
            return 200, payload
        if path.endswith("/grants"):
            assert index_calls == 2
        return 200, {"ok": True}
    status = ensure_native_telegram_live(
        settings={
            "TELEGRAM_BOT_TOKEN": "x",
            "OUROBOROS_AUTO_GRANT_REVIEWED_SKILLS": "true",
        },
        request=fake_request,
        timeout=5,
    )
    assert status["ok"] is True and status["settings_ok"] is True
    assert status["steps"] == ["ready", "discovered", "granted", "enabled", "settings_saved"]
    triples = [(m, p, b) for (m, p, b, t) in calls]
    assert ("POST", "/api/skills/telegram/grants", {"items": ["TELEGRAM_BOT_TOKEN", "subscribe_event:chat.outbound"]}) in triples
    assert ("POST", "/api/skills/telegram/toggle", {"enabled": True}) in triples
    assert (
        "POST",
        "/api/extensions/telegram/settings/save",
        {
            "TELEGRAM_COMMAND_MODE": "full_access",
            "TELEGRAM_MIRROR_MODE": "all",
            "TELEGRAM_MINIAPP_ENABLED": "on",
        },
    ) in triples
    assert all("marketplace" not in path and not path.endswith("/review") for _, path, _ in triples)
    assert all("chat.document" not in str(body) for _, _, body in triples)


def test_ensure_native_telegram_respects_disabled_auto_grant():
    from ouroboros.colab_bootstrap import ensure_native_telegram_live
    calls = []
    def fake_request(method, path, body=None, timeout=None):
        calls.append((method, path, body))
        if path == "/api/health":
            return 200, {}
        if path == "/api/extensions":
            return 200, _native_telegram_index(missing=["TELEGRAM_BOT_TOKEN"])
        return 200, {}
    status = ensure_native_telegram_live(
        settings={
            "TELEGRAM_BOT_TOKEN": "x",
            "OUROBOROS_AUTO_GRANT_REVIEWED_SKILLS": "false",
        },
        request=fake_request,
        timeout=5,
    )
    assert status["ok"] is False
    assert "automatic grants are disabled" in status["error"]
    assert all(not path.endswith(("/grants", "/toggle")) for _, path, _ in calls)


def test_ensure_native_telegram_settings_failure_is_not_silent():
    from ouroboros.colab_bootstrap import ensure_native_telegram_live
    def fake_request(method, path, body=None, timeout=None):
        if path == "/api/health":
            return 200, {}
        if path == "/api/extensions":
            return 200, _native_telegram_index()
        if path.endswith("/settings/save"):
            return 404, {"error": "route not found"}
        return 200, {}
    status = ensure_native_telegram_live(settings={"TELEGRAM_BOT_TOKEN": "x"}, request=fake_request, timeout=5)
    assert status["ok"] is True
    assert status.get("settings_ok") is False
    assert status.get("warning")
    assert "settings_saved" not in status["steps"]


def test_ensure_native_telegram_reports_conflict_without_disabling_peer():
    from ouroboros.colab_bootstrap import ensure_native_telegram_live
    calls = []
    def fake_request(method, path, body=None, timeout=None):
        calls.append((method, path, body))
        if path == "/api/health":
            return 200, {}
        if path == "/api/extensions":
            return 200, _native_telegram_index(conflict={
                "code": "skill_conflict",
                "skills": ["telegram-bridge"],
                "omitted": 0,
            })
        return 200, {}
    status = ensure_native_telegram_live(
        settings={"TELEGRAM_BOT_TOKEN": "x"},
        request=fake_request,
        timeout=5,
    )
    assert status["ok"] is False
    assert "telegram-bridge" in status["error"]
    assert all(not path.endswith("/toggle") for _, path, _ in calls)


def test_ensure_native_telegram_reports_server_not_ready():
    from ouroboros.colab_bootstrap import ensure_native_telegram_live
    status = ensure_native_telegram_live(request=lambda *a, **k: (503, {}), timeout=0.2)
    assert status["ok"] is False and "ready" in status["error"]


def test_ensure_native_telegram_stops_on_enable_error():
    from ouroboros.colab_bootstrap import ensure_native_telegram_live
    def fake_request(method, path, body=None, timeout=None):
        if path == "/api/health":
            return 200, {}
        if path == "/api/extensions":
            return 200, _native_telegram_index()
        if path.endswith("/toggle"):
            return 409, {"error": "cannot enable until requested key and permission grants are approved"}
        return 200, {}
    status = ensure_native_telegram_live(settings={"TELEGRAM_BOT_TOKEN": "x"}, request=fake_request, timeout=5)
    assert status["ok"] is False and "enable failed" in status["error"]


def test_ensure_native_telegram_waits_for_the_dialogue_not_for_a_clock(monkeypatch):
    """The wait follows the server's own phases: a serving core whose dialogue is not open yet is
    waited for (however long), the first dialogue_ready answer proceeds, and each phase is printed
    once. An older core without ``boot`` counts at its first 200."""
    from ouroboros.colab_bootstrap import ensure_native_telegram_live, wait_for_dialogue

    monkeypatch.setattr("ouroboros.colab_bootstrap.time.sleep", lambda _seconds: None)
    health_calls, printed = [], []
    phases = iter([(503, {})] * 3 + [(200, {"boot": {"phase": "serving", "dialogue_ready": False, "tasks_ready": False}})] * 200
                  + [(200, {"boot": {"phase": "serving", "dialogue_ready": True, "tasks_ready": False}})])

    def fake_request(method, path, body=None, timeout=None):
        if path == "/api/health":
            health_calls.append(path)
            return next(phases)
        if path == "/api/extensions":
            return 200, _native_telegram_index()
        return 200, {}

    status = ensure_native_telegram_live(settings={"TELEGRAM_BOT_TOKEN": "x"}, request=fake_request)
    assert status["ok"] is True and status["steps"][0] == "ready"
    assert len(health_calls) == 204, "every not-ready answer was waited through; nothing timed out"

    phases = iter([(200, {"boot": {"phase": "serving", "dialogue_ready": False, "tasks_ready": False}})] * 2
                  + [(200, {"boot": {"phase": "serving", "dialogue_ready": True, "tasks_ready": False}})])
    assert wait_for_dialogue(fake_request, report=printed.append)["ok"] is True
    assert printed == ["Ouroboros boot: serving (dialogue_ready=False, tasks_ready=False)",
                       "Ouroboros boot: serving (dialogue_ready=True, tasks_ready=False)"], "each phase once"
    assert wait_for_dialogue(lambda *a, **k: (200, {}))["ok"] is True, "an older core: its first answer is readiness"
    providerless = lambda *a, **k: (200, {"boot": {"phase": "ready", "dialogue_ready": False, "tasks_ready": True}})  # noqa: E731
    printed.clear()
    assert wait_for_dialogue(providerless, report=printed.append)["ok"] is True, "a providerless boot ends at ready"
    assert printed[-1].startswith("Ouroboros is ready, but its chat is not accepting messages yet")


def test_ensure_native_telegram_ends_on_a_failed_boot_or_a_dead_server(monkeypatch):
    from ouroboros.colab_bootstrap import ensure_native_telegram_live, wait_for_dialogue

    monkeypatch.setattr("ouroboros.colab_bootstrap.time.sleep", lambda _seconds: None)
    failed = lambda *a, **k: (200, {"boot": {"phase": "failed", "dialogue_ready": False, "tasks_ready": False,  # noqa: E731
                                              "error": "current-facts store not published: failed"}})
    status = ensure_native_telegram_live(settings={"TELEGRAM_BOT_TOKEN": "x"}, request=failed)
    assert status["ok"] is False and "store not published" in status["error"]

    lives = iter([True, True, False])
    status = ensure_native_telegram_live(settings={"TELEGRAM_BOT_TOKEN": "x"}, request=lambda *a, **k: (503, {}),
                                         alive=lambda: next(lives))
    assert status["ok"] is False and "exited" in status["error"]
    assert wait_for_dialogue(lambda *a, **k: (503, {}), timeout=0.0)["error"] == "server did not become ready", \
        "a caller's own bound still ends the wait"
