from __future__ import annotations

import json
import hashlib
import os
import subprocess
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.contributor_review_evidence import finalize_contributor_outcome
from scripts.run_external_review import (
    _RELEASE_MACHINERY_PATHS,
    _apply_contributor_landing_obligations,
    _apply_contributor_review_env,
    _assert_contributor_review_config,
    _classify_exit,
    _configured_openrouter_models,
    _contributor_snapshot,
    _contributor_execution_receipts,
    _contributor_result,
    _create_isolated_checkout,
    _freeze_contributor_slots,
    _openrouter_key_health,
    _openrouter_pool,
    _remove_isolated_checkout,
    _require_contributor_budget,
    _run_on_trusted_base,
    _prepare_review_configuration,
    _resolved_review_config,
    _review_evidence_and_cost,
    _select_healthy_openrouter_key,
    _write_contributor_packet,
)


# Stands in for the review script in the seeded repo's BASE commit, so the
# base-side run is really executed and reports which tree it ran from.
_BASE_SIDE_PROBE = """import json, os, pathlib, subprocess, sys

out = os.environ.get("REVIEW_PROBE_OUT", "")
if out:
    here = pathlib.Path(__file__).resolve().parents[1]
    pathlib.Path(out).write_text(json.dumps({
        "argv": sys.argv[1:],
        "machinery_sha": subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=str(here),
            capture_output=True, text=True,
        ).stdout.strip(),
        "machinery_root": str(here),
        "cwd": os.getcwd(),
        "data_dir": os.environ.get("OUROBOROS_DATA_DIR", ""),
    }), encoding="utf-8")
raise SystemExit(1)
"""


def _probe_path(monkeypatch, tmp_path: Path) -> Path:
    probe = tmp_path / "base-side-run.json"
    monkeypatch.setenv("REVIEW_PROBE_OUT", str(probe))
    return probe


@pytest.mark.parametrize(
    "changed_path",
    [
        # A proposal that touches nothing review-related...
        "a.txt",
        # ...and proposals that rewrite the review machinery itself take the
        # SAME path. That identity IS the contract: the lane never asks what a
        # diff contains before deciding whose review code runs.
        "ouroboros/review_substrate.py",
        "scripts/run_external_review.py",
    ],
)
def test_contributor_review_always_runs_on_the_trusted_base(
    tmp_path, monkeypatch, changed_path
):
    """Owner decision (2026-08-19): review always runs on the old version.

    The proposal is still the reviewed subject — the base-side run is handed the
    same base/head commits — but the machinery executing the review is the
    target base's own, whatever the proposal touches, and the base-side exit
    code is the review's exit code. The base script really runs here: it reports
    the tree it was loaded from.
    """
    repo = _init_contributor_repo(tmp_path, monkeypatch)
    probe = _probe_path(monkeypatch, tmp_path)
    path = repo / changed_path
    path.write_text("# proposal\n", encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-m", f"proposal touches {changed_path}")
    base_sha = _git(repo, "rev-parse", "base").strip()
    head_sha = _git(repo, "rev-parse", "HEAD").strip()

    exit_code = _run_on_trusted_base(SimpleNamespace(
        base_ref="base", head_ref="HEAD", commit_message="PR title",
        goal="goal", scope="scope", output="", drive_root="",
    ))

    assert exit_code == 1  # the base-side verdict is this review's verdict
    ran = json.loads(probe.read_text(encoding="utf-8"))
    assert ran["machinery_sha"] == base_sha != head_sha
    assert Path(ran["machinery_root"]) != repo
    assert ran["cwd"] == ran["machinery_root"]
    assert ran["data_dir"]
    assert "--contributor" in ran["argv"]
    # Commits, not refs: a moving ref cannot re-point the trusted run.
    options = _forwarded_options(ran["argv"])
    assert options["base-ref"] == base_sha
    assert options["head-ref"] == head_sha
    assert ran["argv"][-2:] == ["--", "PR title"]
    # The trusted worktree is temporary: it is removed once the review returns.
    assert not Path(ran["machinery_root"]).exists()


def _forwarded_options(argv: list[str]) -> dict[str, str]:
    return dict(
        item[2:].split("=", 1) for item in argv if item.startswith("--") and "=" in item
    )


def test_the_handoff_forwards_artifact_paths_the_child_can_still_reach(
    tmp_path, monkeypatch
):
    """Relative artifact paths must not resolve inside the temporary checkout.

    The child runs with cwd set to the materialized base worktree, which is
    deleted when the review returns; a verbatim relative --output/--drive-root
    would put the operator's results there and lose them. They are absolutized
    against the INVOKING cwd instead. Options travel in equals form so a value
    starting with "-" reaches the child as a value, not as a broken flag.
    """
    _init_contributor_repo(tmp_path, monkeypatch)
    probe = _probe_path(monkeypatch, tmp_path)
    monkeypatch.chdir(tmp_path)

    _run_on_trusted_base(SimpleNamespace(
        base_ref="base", head_ref="HEAD", commit_message="-title-like-a-flag",
        goal="--goal-like-a-flag", scope="-s", output="artifacts/run",
        drive_root="~/drive",
    ))

    ran = json.loads(probe.read_text(encoding="utf-8"))
    options = _forwarded_options(ran["argv"])
    assert options["output"] == str(tmp_path / "artifacts" / "run")
    # A quoted "~/..." keeps its home meaning: the parent expands it exactly
    # the way the in-place lane's own resolution would have.
    assert options["drive-root"] == os.path.abspath(os.path.expanduser("~/drive"))
    for key in ("output", "drive-root"):
        assert not Path(options[key]).is_relative_to(Path(ran["machinery_root"]))
    # Values that look like flags survive as values.
    assert options["goal"] == "--goal-like-a-flag"
    assert options["scope"] == "-s"
    assert ran["argv"][-2:] == ["--", "-title-like-a-flag"]


def test_contributor_review_invoked_from_the_target_base_runs_in_place(
    tmp_path, monkeypatch
):
    """No re-run when the executing tree already IS the target base."""
    repo = _init_contributor_repo(tmp_path, monkeypatch)
    probe = _probe_path(monkeypatch, tmp_path)
    head_sha = _git(repo, "rev-parse", "HEAD").strip()
    _git(repo, "checkout", "--detach", "base")

    assert _run_on_trusted_base(SimpleNamespace(
        base_ref="base", head_ref=head_sha, commit_message="",
        goal="", scope="", output="", drive_root="",
    )) is None
    assert not probe.exists()


def test_the_real_wrapper_hands_off_before_it_reviews_anything(tmp_path, monkeypatch):
    """End-to-end pin of the main() wiring, not just the helper.

    The REAL script is invoked as a process from a checkout that is not the
    target base, exactly as a contributor runs it. Reaching any review work
    without handing off first would leave the base-side probe unexecuted, so
    deleting the main() hook fails here even while the helper stays perfect.
    """
    repo = _init_contributor_repo(tmp_path, monkeypatch)
    probe = _probe_path(monkeypatch, tmp_path)
    wrapper = Path(__file__).resolve().parent.parent / "scripts" / "run_external_review.py"
    # The base commit keeps the probe; the proposal carries the real wrapper.
    (repo / "scripts" / "run_external_review.py").write_text(
        wrapper.read_text(encoding="utf-8"), encoding="utf-8"
    )
    _git(repo, "add", "-A")
    _git(repo, "commit", "-m", "proposal adopts the real wrapper")
    base_sha = _git(repo, "rev-parse", "base").strip()

    proc = subprocess.run(
        [
            sys.executable, str(repo / "scripts" / "run_external_review.py"),
            "--contributor", "--base-ref=base", "--head-ref=HEAD", "--", "PR title",
        ],
        cwd=str(repo), capture_output=True, text=True, timeout=300,
        env={**os.environ, "REVIEW_PROBE_OUT": str(probe)},
    )

    assert probe.exists(), f"the base-side run never happened: {proc.stderr[-2000:]}"
    ran = json.loads(probe.read_text(encoding="utf-8"))
    assert ran["machinery_sha"] == base_sha
    assert proc.returncode == 1  # the probe's exit code, passed through


def test_contributor_review_refuses_a_dirty_authoring_worktree(tmp_path, monkeypatch):
    """The uncommitted half of a proposal must not silently drop out.

    The base-side run sees a freshly materialized (always clean) worktree, so
    this is read in the authoring worktree before the re-run leaves it.
    """
    repo = _init_contributor_repo(tmp_path, monkeypatch)
    probe = _probe_path(monkeypatch, tmp_path)
    (repo / "uncommitted.txt").write_text("work in progress\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="not clean"):
        _run_on_trusted_base(SimpleNamespace(
            base_ref="base", head_ref="HEAD", commit_message="",
            goal="", scope="", output="", drive_root="",
        ))
    assert not probe.exists()


def test_contributor_result_is_decided_by_the_exit_code_alone():
    """The retired D31 classifier is not a gate anywhere in the outcome path.

    A proposal rewriting the review machinery gets the same result vocabulary as
    any other, because nothing but the review's exit code reaches this decision.
    """
    assert _contributor_result(0) == "READY_FOR_INTEGRATION"
    assert _contributor_result(1) == "BLOCKED"
    assert _contributor_result(3) == "INCOMPLETE"


def test_external_review_script_delegates_verdict_to_production_gate():
    source = Path("scripts/run_external_review.py").read_text(encoding="utf-8")
    assert "v6.10.0" not in source
    assert "Google Colab" not in source
    assert "_run_non_committing_review_cycle" in source
    assert "adaptive_quorum" not in source
    assert "aggregate_review_verdict" not in source
    # The default operator lane still runs the REAL advisory. Contributor mode
    # explicitly skips it while reusing the production triad+scope cycle.
    assert "operator_binding" not in source
    assert "_handle_advisory_pre_review" in source
    assert "skip_advisory_review=args.contributor" in source
    assert "_CONTRIBUTOR_PROFILE = \"external_pr_readiness\"" in source


def test_external_review_script_defaults_to_pro_mode():
    source = Path("scripts/run_external_review.py").read_text(encoding="utf-8")
    assert 'setdefault("OUROBOROS_RUNTIME_MODE", "pro")' in source


def test_external_review_advisory_warning_uses_safe_canonical_reason(monkeypatch):
    import scripts.run_external_review as module
    from ouroboros.tools import claude_advisory_review as advisory

    monkeypatch.setattr(
        advisory,
        "advisory_gate_unavailability_reason",
        lambda: "agent_session_route_unavailable",
    )
    warning = module._advisory_unavailability_warning()
    assert "agent_session_route_unavailable" in warning
    assert advisory.ADVISORY_REVIEW_CHOICE_GUIDANCE in warning
    assert "ANTHROPIC_API_KEY" not in warning

    secret_error = "secret-setting-value-must-not-leak"

    def _malformed():
        raise ValueError(secret_error)

    monkeypatch.setattr(advisory, "advisory_gate_unavailability_reason", _malformed)
    warning = module._advisory_unavailability_warning()
    assert "invalid_advisory_configuration" in warning
    assert secret_error not in warning


def test_external_review_checks_advisory_after_settings_load_without_key_heuristic():
    import inspect
    import scripts.run_external_review as module

    main_source = inspect.getsource(module.main)
    prepare_source = inspect.getsource(module._prepare_review_configuration)
    assert main_source.index("_prepare_review_configuration(args)") < main_source.index(
        "_advisory_unavailability_warning()"
    )
    assert "_load_settings_into_env()" in prepare_source
    assert 'if not os.environ.get("ANTHROPIC_API_KEY"' not in main_source


def test_external_review_script_resolves_models_and_efforts(monkeypatch):
    for key in (
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "CLOUDRU_FOUNDATION_MODELS_API_KEY",
        "GIGACHAT_CREDENTIALS",
        "GIGACHAT_USER",
        "GIGACHAT_PASSWORD",
        "OPENAI_BASE_URL",
        "OPENAI_COMPATIBLE_BASE_URL",
        "OUROBOROS_MODEL",
        "OUROBOROS_MODEL_LIGHT",
    ):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.delenv("OUROBOROS_REVIEWER_SLOTS", raising=False)
    monkeypatch.setenv("OUROBOROS_REVIEW_MODELS", "anthropic/claude-opus-4.8,google/gemini-3.5-flash,openai/gpt-5.5")
    monkeypatch.setenv("OUROBOROS_SCOPE_REVIEW_MODELS", "openai/gpt-5.5")
    monkeypatch.setenv("OUROBOROS_EFFORT_REVIEW", "high")
    monkeypatch.setenv("OUROBOROS_EFFORT_SCOPE_REVIEW", "high")
    monkeypatch.setenv("OUROBOROS_REVIEW_ENFORCEMENT", "blocking")
    monkeypatch.setenv("OUROBOROS_CONTEXT_MODE", "max")

    config = _resolved_review_config()

    assert config["triad_models"] == [
        "anthropic/claude-opus-4.8",
        "google/gemini-3.5-flash",
        "openai/gpt-5.5",
    ]
    assert config["triad_efforts"] == ["high", "high", "high"]
    assert config["scope_models"] == ["openai/gpt-5.5"]
    assert config["scope_efforts"] == ["high"]
    assert all(row["route"]["kind"] == "api_chat" for row in config["triad_slots"])
    assert config["review_enforcement"] == "blocking"
    # v6.80.0: the scope-review floor key is gone; the operator line pins the context
    # mode instead, because that is now what decides scope-review applicability.
    assert config["context_mode"] == "max"


def _write_target_config(repo: Path) -> None:
    package = repo / "ouroboros"
    package.mkdir(exist_ok=True)
    (package / "config.py").write_text(
        "SETTINGS_DEFAULTS = {\n"
        "    'OUROBOROS_REVIEW_MODELS': 'anthropic/fable,openai/sol,google/flash',\n"
        "    'OUROBOROS_SCOPE_REVIEW_MODELS': 'anthropic/fable',\n"
        "    'OUROBOROS_EFFORT_REVIEW': 'high',\n"
        "    'OUROBOROS_EFFORT_SCOPE_REVIEW': 'high',\n"
        "}\n",
        encoding="utf-8",
    )


def _init_contributor_repo(tmp_path: Path, monkeypatch) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test")
    _git(repo, "config", "core.autocrlf", "false")
    # Same ignores as the real repository: importing from a checkout writes
    # bytecode into it, which would otherwise read as an unclean worktree.
    (repo / ".gitignore").write_text("__pycache__/\n*.pyc\n", encoding="utf-8")
    (repo / "scripts").mkdir()
    (repo / "scripts" / "run_external_review.py").write_text(
        _BASE_SIDE_PROBE, encoding="utf-8"
    )
    _write_target_config(repo)
    # A real (not namespace) package, so a wrapper executed out of this repo
    # imports its module-level dependency from here and not from whatever
    # ouroboros the host interpreter happens to have installed.
    (repo / "ouroboros" / "__init__.py").write_text("", encoding="utf-8")
    (repo / "ouroboros" / "runtime_mode_policy.py").write_text(
        "GIT_OPS_FAMILY_PATHS = frozenset()\n", encoding="utf-8"
    )
    (repo / "ouroboros" / "review_substrate.py").write_text(
        "# trusted review substrate\n", encoding="utf-8"
    )
    (repo / "ouroboros" / "utils.py").write_text(
        "# trusted review utilities\n", encoding="utf-8"
    )
    (repo / "ouroboros" / "tools").mkdir()
    (repo / "ouroboros" / "tools" / "registry.py").write_text(
        "# trusted review context\n", encoding="utf-8"
    )
    (repo / "pyproject.toml").write_text(
        '[project]\nname = "test-project"\nversion = "1.2.3"\n',
        encoding="utf-8",
    )
    (repo / "uv.lock").write_text(
        'version = 1\n\n[[package]]\nname = "ouroboros"\nversion = "1.2.3"\n'
        'source = { editable = "." }\n',
        encoding="utf-8",
    )
    (repo / "web" / "modules").mkdir(parents=True)
    (repo / "web" / "package.json").write_text(
        '{"name":"ouroboros-ui","version":"1.2.3"}\n', encoding="utf-8"
    )
    (repo / "web" / "modules" / "api_types.js").write_text(
        'export const GATEWAY_CONTRACT_VERSION = "1.2.3";\n', encoding="utf-8"
    )
    (repo / "README.md").write_text(
        "[![Version 1.2.3](https://example.test/version.svg)](#)\n\n"
        "[download-macos-arm64]: https://example.test/v1.2.3/Ouroboros-1.2.3.dmg\n\n"
        "## Version History\n\n| 1.2.3 | Current |\n",
        encoding="utf-8",
    )
    (repo / "docs").mkdir()
    (repo / "docs" / "ARCHITECTURE.md").write_text(
        "# Ouroboros v1.2.3\n", encoding="utf-8"
    )
    for root in (repo / "site" / "install", repo / "docs" / "install"):
        root.mkdir(parents=True)
        (root / "index.html").write_text(
            '<a href="https://example.test/v1.2.3/Ouroboros-1.2.3.dmg" '
            'data-release-download="macos-arm64">Download</a>\n'
            '<a data-release-download="macos-arm64" '
            'href="https://example.test/v1.2.3/Ouroboros-1.2.3.dmg">'
            'Quick start</a>\n',
            encoding="utf-8",
        )
    (repo / "VERSION").write_text("1.2.3\n", encoding="utf-8")
    (repo / "a.txt").write_text("base\n", encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-m", "base")
    _git(repo, "branch", "base")
    (repo / "a.txt").write_text("proposal\n", encoding="utf-8")
    _git(repo, "add", "a.txt")
    _git(repo, "commit", "-m", "proposal")

    import scripts.run_external_review as module

    monkeypatch.setattr(module, "REPO", repo)
    return repo


def test_contributor_policy_preserves_configured_routes(monkeypatch):
    payload = {
        "triad": [
            {"slot_id": "session", "route": {
                "kind": "agent_session", "target_id": "codex=gpt-5.6-sol",
                "profile_id": "account-a"}, "effort": "high"},
            {"slot_id": "direct", "route": {
                "kind": "api_chat", "target_id": "anthropic::claude-fable-5"},
                "effort": "xhigh"},
        ],
        "scope": [{"slot_id": "scope", "route": {
            "kind": "api_chat", "target_id": "openai/gpt-5.6-sol"},
            "effort": "high"}],
    }
    raw = json.dumps(payload)
    monkeypatch.setenv("OUROBOROS_REVIEWER_SLOTS", raw)
    for key in ("OUROBOROS_REVIEW_ENFORCEMENT", "OUROBOROS_CONTEXT_MODE",
                "OUROBOROS_OBSERVABILITY_KEEP_RAW", "OUROBOROS_PRE_PUSH_TESTS",
                "OUROBOROS_PREFLIGHT_DIFF_AWARE"):
        monkeypatch.setenv(key, "")

    _apply_contributor_review_env()
    config = _resolved_review_config(profile="external_pr_readiness")

    assert os.environ["OUROBOROS_REVIEWER_SLOTS"] == raw
    assert os.environ["OUROBOROS_REVIEW_ENFORCEMENT"] == "blocking"
    assert os.environ["OUROBOROS_OBSERVABILITY_KEEP_RAW"] == "0"
    assert os.environ["OUROBOROS_PRE_PUSH_TESTS"] == "1"
    assert os.environ["OUROBOROS_PREFLIGHT_DIFF_AWARE"] == "false"
    assert [row["route"]["kind"] for row in config["triad_slots"]] == [
        "agent_session", "api_chat",
    ]
    assert config["triad_slots"][0]["route"]["profile_id"] == "account-a"
    assert _configured_openrouter_models(config) == ["openai/gpt-5.6-sol"]
    assert _configured_openrouter_models({
        "triad_slots": [{"route": {
            "kind": "api_chat", "target_id": "openrouter::openai/gpt-5.6-sol",
        }}],
    }) == ["openai/gpt-5.6-sol"]
    _assert_contributor_review_config(config)
    frozen = _freeze_contributor_slots(config)
    assert frozen["execution_slot_config_source"] == "frozen_structured"
    assert len(frozen["slot_plan_sha256"]) == 64
    assert json.loads(os.environ["OUROBOROS_REVIEWER_SLOTS"])["triad"][0][
        "slot_id"
    ] == "session"


def test_agent_session_only_preflight_needs_no_api_budget_or_key(monkeypatch):
    import scripts.run_external_review as module

    config = {
        "triad_slots": [{"slot_id": "t1", "route": {
            "kind": "agent_session", "target_id": "codex=gpt-5.6-sol"},
            "effort": "high"}],
        "scope_slots": [{"slot_id": "s1", "route": {
            "kind": "agent_session", "target_id": "cursor=claude-fable-5"},
            "effort": "high"}],
        "review_enforcement": "blocking", "context_mode": "max",
    }
    monkeypatch.delenv("TOTAL_BUDGET", raising=False)
    monkeypatch.setattr(module, "_load_settings_into_env", lambda: None)
    monkeypatch.setattr(module, "_contributor_snapshot", lambda *_args: {"base_sha": "a" * 40})
    monkeypatch.setattr(module, "_apply_contributor_review_env", lambda: None)
    monkeypatch.setattr(module, "_resolved_review_config", lambda **_kwargs: config)
    monkeypatch.setattr(module, "_freeze_contributor_slots", lambda value: value)
    monkeypatch.setattr(
        module, "_select_healthy_openrouter_key",
        lambda **_kwargs: pytest.fail("agent-only review must not probe OpenRouter"),
    )

    snapshot, base, resolved = _prepare_review_configuration(SimpleNamespace(
        contributor=True, base_ref="", head_ref="HEAD",
    ))

    assert snapshot and base == "a" * 40
    assert resolved == config


def test_contributor_budget_must_be_explicit_positive_and_finite(monkeypatch):
    monkeypatch.delenv("TOTAL_BUDGET", raising=False)
    with pytest.raises(RuntimeError, match="TOTAL_BUDGET is required"):
        _require_contributor_budget()
    for invalid in ("0", "-1", "inf", "not-a-number"):
        monkeypatch.setenv("TOTAL_BUDGET", invalid)
        with pytest.raises(RuntimeError, match="positive finite"):
            _require_contributor_budget()
    monkeypatch.setenv("TOTAL_BUDGET", "125.50")
    assert _require_contributor_budget() == 125.5


def test_contributor_snapshot_binds_clean_base_head_and_tree(tmp_path, monkeypatch):
    repo = _init_contributor_repo(tmp_path, monkeypatch)

    snapshot = _contributor_snapshot("base", "HEAD")

    assert snapshot["base_sha"] == snapshot["merge_base_sha"]
    assert snapshot["target_version"] == "1.2.3"
    assert snapshot["head_tree_sha"] == _git(repo, "rev-parse", "HEAD^{tree}").strip()
    assert snapshot["changed_paths"] == ["a.txt"]
    assert snapshot["diff_sha256"]
    # The retired trust-boundary classification leaves no snapshot residue.
    assert not {"review_substrate_changed", "review_substrate_matches_base"} & set(
        snapshot
    )

    (repo / "dirty.txt").write_text("not committed\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="not clean"):
        _contributor_snapshot("base", "HEAD")


@pytest.mark.parametrize(
    ("relative_path", "before", "after", "carrier"),
    [
        ("VERSION", "1.2.3", "1.2.4", "VERSION"),
        ("pyproject.toml", 'version = "1.2.3"', 'version = "1.2.4"',
         "pyproject.project.version"),
        ("uv.lock", 'version = "1.2.3"', 'version = "1.2.4"',
         "uv.editable_root.version"),
        ("web/package.json", '"version":"1.2.3"', '"version":"1.2.4"',
         "web.package.version"),
        ("web/modules/api_types.js", 'VERSION = "1.2.3"', 'VERSION = "1.2.4"',
         "gateway.contract.version"),
        ("README.md", "Version 1.2.3", "Version 1.2.4",
         "readme.badge.version"),
        ("README.md", "| 1.2.3 | Current |", "| 1.2.4 | Current |",
         "readme.latest_history_row"),
        ("README.md", "v1.2.3/Ouroboros-1.2.3.dmg",
         "v1.2.4/Ouroboros-1.2.4.dmg", "readme.download.macos-arm64.0"),
        ("site/install/index.html", "v1.2.3/Ouroboros-1.2.3.dmg",
         "v1.2.4/Ouroboros-1.2.4.dmg", "site.install.download.macos-arm64.0"),
        ("docs/install/index.html", "v1.2.3/Ouroboros-1.2.3.dmg",
         "v1.2.4/Ouroboros-1.2.4.dmg", "docs.install.download.macos-arm64.0"),
        ("docs/ARCHITECTURE.md", "Ouroboros v1.2.3", "Ouroboros v1.2.4",
         "architecture.header.version"),
    ],
)
def test_contributor_snapshot_rejects_every_version_carrier(
    tmp_path, monkeypatch, relative_path, before, after, carrier
):
    repo = _init_contributor_repo(tmp_path, monkeypatch)
    path = repo / relative_path
    path.write_text(path.read_text(encoding="utf-8").replace(before, after), encoding="utf-8")
    _git(repo, "add", relative_path)
    _git(repo, "commit", "-m", "bad contributor bump")

    with pytest.raises(RuntimeError, match=carrier):
        _contributor_snapshot("base", "HEAD")


def test_contributor_snapshot_checks_each_duplicate_installer_link(
    tmp_path, monkeypatch
):
    repo = _init_contributor_repo(tmp_path, monkeypatch)
    path = repo / "site" / "install" / "index.html"
    current = "https://example.test/v1.2.3/Ouroboros-1.2.3.dmg"
    stale = "https://example.test/v1.2.4/Ouroboros-1.2.4.dmg"
    text = path.read_text(encoding="utf-8")
    path.write_text(text.replace(current, stale, 1), encoding="utf-8")
    _git(repo, "add", str(path.relative_to(repo)))
    _git(repo, "commit", "-m", "change one duplicate installer link")

    with pytest.raises(RuntimeError, match="site.install.download.macos-arm64.0"):
        _contributor_snapshot("base", "HEAD")


@pytest.mark.parametrize(
    "relative_path",
    ["Ouroboros.spec", "ouroboros/tool_module_inventory.py"],
)
def test_contributor_snapshot_flags_frozen_inventory_release_machinery(
    tmp_path, monkeypatch, relative_path
):
    assert relative_path in _RELEASE_MACHINERY_PATHS
    repo = _init_contributor_repo(tmp_path, monkeypatch)
    path = repo / relative_path
    path.write_text("# proposal changes release machinery\n", encoding="utf-8")
    _git(repo, "add", str(path.relative_to(repo)))
    _git(repo, "commit", "-m", "change release machinery")

    snapshot = _contributor_snapshot("base", "HEAD")

    assert snapshot["release_metadata_or_machinery_changed"] is True
    assert snapshot["release_sensitive_changes"]["machinery_paths"] == [relative_path]


def test_contributor_snapshot_rejects_release_carrier_changes_without_version_file(
    tmp_path, monkeypatch
):
    repo = _init_contributor_repo(tmp_path, monkeypatch)
    path = repo / "pyproject.toml"
    path.write_text(
        '[project]\nname = "test-project"\nversion = "1.2.4"\n',
        encoding="utf-8",
    )
    _git(repo, "add", "pyproject.toml")
    _git(repo, "commit", "-m", "change package carrier only")

    with pytest.raises(
        RuntimeError,
        match=r"must not change release-version carriers \(pyproject\.project\.version\)",
    ):
        _contributor_snapshot("base", "HEAD")

def test_contributor_landing_obligations_are_exact_typed_items_only():
    version_only = {
        "status": "blocked",
        "block_reason": "critical_findings",
        "combined_findings": [
            {"item": "version_bump", "severity": "critical"},
            {"item": "changelog_and_badge", "severity": "critical"},
        ],
    }
    deferred = _apply_contributor_landing_obligations(version_only)
    assert deferred["status"] == "passed"
    assert {item["item"] for item in deferred["landing_obligations"]} == {
        "version_bump",
        "changelog_and_badge",
    }

    real_defect = {
        **version_only,
        "combined_findings": [
            *version_only["combined_findings"],
            {"item": "self_consistency", "severity": "critical"},
        ],
    }
    assert _apply_contributor_landing_obligations(real_defect) == real_defect
    scope_failure = {
        **version_only,
        "block_reason": "scope_blocked",
    }
    assert _apply_contributor_landing_obligations(scope_failure) == scope_failure
    assert _apply_contributor_landing_obligations(
        version_only,
        release_sensitive=True,
    ) == version_only


def test_contributor_packet_is_redacted_and_shareable(tmp_path):
    output = tmp_path / "packet"
    output.mkdir()
    local_root = "/Users/example/private/repo"
    packet = _write_contributor_packet(
        output_dir=output,
        snapshot={
            "base_sha": "a" * 40,
            "head_sha": "b" * 40,
            # A proposal rewriting the review script is packeted like any other.
            "changed_paths": ["scripts/run_external_review.py"],
        },
        resolved_config={"triad_models": ["anthropic/fable"]},
        outcome={"status": "passed", "path": local_root, "api_key": "test-secret-value"},
        exit_code=0,
        evidence_refs=[],
        cost_report={"reported_actor_cost_usd": 1.0},
        elapsed_sec=1.5,
        triad_raw=[{"authorization": "Bearer secret-token-value", "path": local_root}],
        scope_raw={"status": "responded"},
        execution_receipts=[{
            "surface": "triad", "slot_id": "slot_1",
            "observed": {"route_kind": "agent_session"},
            "model_verification": "observed_display_label",
        }],
        execution_mismatches=[],
        session_transcripts=[{
            "surface": "triad", "slot_id": "slot_1", "sha256": "a" * 64,
            "chars": 18, "transcript": "transcript EOF_MARK",
        }],
        degraded_reasons=["reviewer-3=parse_failure (quorum still met)"],
        replacements=[(local_root, "$REPO")],
    )

    evidence_text = (output / "review-evidence.json").read_text(encoding="utf-8")
    full_text = (output / "full-output.txt").read_text(encoding="utf-8")
    public_evidence = json.loads(evidence_text)
    assert "test-secret-value" not in evidence_text
    assert "secret-token-value" not in full_text
    assert local_root not in evidence_text + full_text
    assert "$REPO" in evidence_text + full_text
    assert "production_triad_quorum_plus_authoritative_scope" in evidence_text
    assert '"execution_receipts_consistent": true' in evidence_text
    # Evidence records the unconditional contract, never a per-proposal verdict
    # about whose review code ran.
    assert public_evidence["result"] == "READY_FOR_INTEGRATION"
    assert public_evidence["trust"]["review_machinery"] == "target_base_unconditional"
    assert "owner decision 2026-08-19" in public_evidence["trust"]["note"]
    assert "rerun" not in evidence_text
    assert "triad:slot_1:observed_model_is_display_label" in evidence_text
    assert "quorum still met" in evidence_text
    assert "transcript EOF_MARK" in full_text
    transcript_meta = public_evidence["review_execution"]["session_transcript_artifacts"][0]
    assert transcript_meta["chars"] == len("transcript EOF_MARK")
    assert transcript_meta["sha256"] == hashlib.sha256(
        b"transcript EOF_MARK"
    ).hexdigest()
    with zipfile.ZipFile(packet) as archive:
        assert set(archive.namelist()) == {
            "review-evidence.json",
            "outcome.json",
            "full-output.txt",
        }


def _complete_ctx():
    triad = [
        {
            "slot_id": f"slot_{idx}",
            "model_id": f"reviewer-{idx}",
            "status": "responded",
            "tokens_in": 100,
            "cost_usd": 0.01,
            "prompt_ref": {"manifest_ref": f"prompt-{idx}"},
            "response_ref": {"manifest_ref": f"response-{idx}"},
        }
        for idx in range(1, 4)
    ]
    scope_actor = {
        "slot_id": "scope_slot_1",
        "model_id": "scope-reviewer",
        "status": "responded",
        "tokens_in": 200,
        "cost_usd": 0.0,
        "prompt_ref": {"manifest_ref": "scope-prompt"},
        "response_ref": {"manifest_ref": "scope-response"},
    }
    return SimpleNamespace(
        _last_triad_raw_results=triad,
        _last_scope_raw_result={"raw_results": [scope_actor]},
    )


def test_external_review_cost_report_never_turns_unknown_into_zero():
    evidence, report = _review_evidence_and_cost(_complete_ctx())
    assert len(evidence) == 4
    assert report["reported_actor_cost_usd"] == 0.03
    assert report["unreported_or_unknown_cost_slots"] == ["scope_slot_1"]
    assert "not treated as $0" in report["note"]


def test_exit_classification_separates_infra_from_genuine_blocks():
    assert _classify_exit({"status": "passed"}) == 0
    assert _classify_exit({"status": "blocked", "block_reason": "critical_findings"}) == 1
    # A scope CRITICAL with concrete findings is a genuine reviewer verdict...
    assert _classify_exit({
        "status": "blocked",
        "block_reason": "scope_blocked",
        "combined_findings": [{"severity": "CRITICAL", "text": "real defect"}],
    }) == 1
    # ...while a findings-less scope block is fail-closed infrastructure.
    assert _classify_exit({"status": "blocked", "block_reason": "scope_blocked"}) == 3
    for infra_reason in (
        "tests_preflight_blocked",
        "core_protection_blocked",
        "no_advisory",
        "review_quorum",
        "fingerprint_unavailable",
        "",
    ):
        assert _classify_exit({"status": "blocked", "block_reason": infra_reason}) == 3, infra_reason


def test_contributor_outcome_fails_closed_on_receipt_drift_only():
    exit_code, outcome = finalize_contributor_outcome(
        outcome={"status": "passed"}, exit_code=0,
        mismatches=["provider_mismatch:triad:t1"],
    )
    assert exit_code == 3
    assert outcome["block_reason"] == "execution_receipt_mismatch"

    # Nothing about the proposal's contents downgrades a clean run any more: the
    # machinery that produced it was the target base's either way.
    assert finalize_contributor_outcome(
        outcome={"status": "passed"}, exit_code=0, mismatches=[],
    ) == (0, {"status": "passed"})


def test_openrouter_pool_orders_hope_keys_last(monkeypatch, tmp_path):
    keys = tmp_path / "file1.txt"
    keys.write_text(
        "hope_new_key_openrouter: sk-or-hope-000\n"
        "openrouter_kuznetsov3: sk-or-kuz-111\n"
        "backup_hope_openrouter: sk-or-hope-bak-444\n"
        "openai: sk-oa-222\n"
        "anton_openrouter_main: sk-or-anton-333\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("OUROBOROS_KEYS_FILE", str(keys))
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    pool = _openrouter_pool()

    names = [name for name, _ in pool]
    # Any hope-bucket key sinks to the tail, prefix or not.
    assert names == [
        "openrouter_kuznetsov3",
        "anton_openrouter_main",
        "hope_new_key_openrouter",
        "backup_hope_openrouter",
    ]


def test_contributor_openrouter_preflight_fails_closed(monkeypatch):
    import scripts.run_external_review as module

    monkeypatch.setattr(module, "_openrouter_pool", lambda: [])
    with pytest.raises(RuntimeError, match="no OpenRouter key"):
        _select_healthy_openrouter_key(required=True)

    monkeypatch.setattr(module, "_openrouter_pool", lambda: [("key", "secret")])
    monkeypatch.setattr(
        module,
        "_openrouter_key_health",
        lambda _token, **_kwargs: (False, "model_probe_http_403"),
    )
    with pytest.raises(RuntimeError, match="no healthy OpenRouter key"):
        _select_healthy_openrouter_key(required=True)


def _persist_review_prompt(tmp_path, *, call_id: str, slot: dict):
    from ouroboros.observability import persist_call

    return persist_call(
        tmp_path, task_id="review", call_id=call_id, call_type="prompt",
        payload={"request": {"surface": "review"}, "slot": slot},
    )


def _persist_review_response(
    tmp_path, *, call_id: str, usage: dict, transcript: str = "",
):
    from ouroboros.observability import persist_call

    message = {"content": "[]"}
    if transcript:
        message["session_transcript"] = transcript
        usage = {**usage, "verdict_provenance": {
            "raw_transcript_chars": len(transcript),
            "raw_transcript_sha256": hashlib.sha256(
                transcript.encode("utf-8", "replace")
            ).hexdigest(),
        }}
    return persist_call(
        tmp_path, task_id="review", call_id=call_id, call_type="response",
        payload={"message": message, "usage": usage},
    )


def test_contributor_receipts_bind_session_and_api_execution(tmp_path):
    config = {
        "triad_slots": [{"slot_id": "t1", "route": {
            "kind": "agent_session", "target_id": "codex=gpt-5.6-sol",
            "profile_id": "pinned"}, "effort": "high"}],
        "scope_slots": [{"slot_id": "s1", "route": {
            "kind": "api_chat", "target_id": "openai/gpt-5.6-sol"},
            "effort": "xhigh"}],
        "review_enforcement": "blocking",
        "context_mode": "max",
    }
    _assert_contributor_review_config(config)
    session_prompt = _persist_review_prompt(tmp_path, call_id="session_prompt", slot={
        "slot_id": "t1", "model": "codex=gpt-5.6-sol", "effort": "high",
        "route": "agent_session", "session_target": "codex=gpt-5.6-sol",
        "session_profile": "pinned",
    })
    transcript = "full session transcript\nEOF_SENTINEL"
    session_ref = _persist_review_response(
        tmp_path, call_id="session", transcript=transcript, usage={
            "provider": "claudexor", "delegated_route": "codex",
            "resolved_model": "gpt-5.6-sol", "applied_profile": "pinned",
            "applied_access": "readonly", "delegated_run_id": "run-1",
            "custody_durable": True, "output_conformance": "passed",
            "verdict_method": "schema", "settlement": {
                "settled": True, "ledger_recorded": True,
                "project_retired": True,
            }},
    )
    api_prompt = _persist_review_prompt(tmp_path, call_id="api_prompt", slot={
        "slot_id": "s1", "model": "openai/gpt-5.6-sol", "effort": "xhigh",
        "route": "api_chat", "session_target": "", "session_profile": "",
    })
    api_ref = _persist_review_response(
        tmp_path, call_id="api", usage={
            "provider": "openrouter", "resolved_model": "openai/gpt-5.6-sol"},
    )
    ctx = SimpleNamespace(
        _last_triad_raw_results=[{
            "slot_id": "t1", "model_id": "gpt-5.6-sol", "status": "responded",
            "prompt_ref": session_prompt, "response_ref": session_ref,
        }],
        _last_scope_raw_result={"raw_results": [{
            "slot_id": "s1", "model_id": "openai/gpt-5.6-sol",
            "status": "responded", "prompt_ref": api_prompt, "response_ref": api_ref,
        }]},
    )

    receipts, mismatches, transcripts = _contributor_execution_receipts(
        ctx, config, tmp_path
    )

    assert mismatches == []
    assert receipts[0]["observed"] == {
        "route_kind": "agent_session", "provider": "claudexor",
        "harness": "codex", "model": "gpt-5.6-sol",
        "profile_id": "pinned", "access": "readonly", "effort": None,
        "delegated_run_id": "run-1", "custody_durable": True,
        "settlement": {
            "settled": True, "ledger_recorded": True,
            "project_retired": True,
        },
        "output_conformance": "passed", "verdict_method": "schema",
    }
    assert receipts[0]["dispatched"]["effort"] == "high"
    assert receipts[0]["model_verification"] == "exact"
    assert transcripts[0]["transcript"].endswith("EOF_SENTINEL")
    drifted = json.loads(json.dumps(config))
    drifted["triad_slots"][0]["route"]["target_id"] = "cursor=gpt-5.6-sol"
    _, mismatches, _ = _contributor_execution_receipts(ctx, drifted, tmp_path)
    assert any(item.startswith("harness_mismatch:triad:t1") for item in mismatches)


def test_contributor_receipts_fail_closed_on_blob_provider_model_and_status_drift(tmp_path):
    config = {
        "triad_slots": [{"slot_id": "t1", "route": {
            "kind": "api_chat", "target_id": "anthropic::claude-fable-5"},
            "effort": "high"}],
        "scope_slots": [{"slot_id": "s1", "route": {
            "kind": "agent_session", "target_id": "codex=gpt-5.6-sol"},
            "effort": "high"}],
        "review_enforcement": "blocking", "context_mode": "max",
    }
    api_prompt = _persist_review_prompt(tmp_path, call_id="api_prompt_bad", slot={
        "slot_id": "t1", "model": "anthropic::claude-fable-5", "effort": "high",
        "route": "api_chat", "session_target": "", "session_profile": "",
    })
    api_response = _persist_review_response(
        tmp_path, call_id="api_bad", usage={
            "provider": "openrouter", "resolved_model": "openai/gpt-5.5"},
    )
    session_prompt = _persist_review_prompt(tmp_path, call_id="session_prompt_bad", slot={
        "slot_id": "s1", "model": "codex=gpt-5.6-sol", "effort": "high",
        "route": "agent_session", "session_target": "codex=gpt-5.6-sol",
        "session_profile": "",
    })
    session_response = _persist_review_response(
        tmp_path, call_id="session_bad", transcript="raw\nEOF", usage={
            "provider": "claudexor", "delegated_route": "codex",
            "resolved_model": "GPT-5.6 Terra 300K High",
            "applied_profile": "auto-profile",
            "applied_access": "readonly", "custody_durable": True,
            "capability_delta": [{"reason": "session_ran_off_pinned_route"}],
        },
    )
    ctx = SimpleNamespace(
        _last_triad_raw_results=[{
            "slot_id": "t1", "status": "parse_failure",
            "prompt_ref": api_prompt, "response_ref": api_response,
        }],
        _last_scope_raw_result={"raw_results": [{
            "slot_id": "s1", "status": "responded",
            "prompt_ref": session_prompt, "response_ref": session_response,
        }]},
    )

    _, mismatches, _ = _contributor_execution_receipts(ctx, config, tmp_path)

    assert any(item.startswith("provider_mismatch:triad:t1") for item in mismatches)
    assert any(item.startswith("model_mismatch:triad:t1") for item in mismatches)
    assert any(item.startswith("model_identity_unverified:scope:s1")
               for item in mismatches)
    assert "delegated_run_id_absent:scope:s1" in mismatches
    assert any(item.startswith("session_settlement_unproven:scope:s1")
               for item in mismatches)
    assert "capability_delta:scope:s1:session_ran_off_pinned_route" in mismatches

    missing_response = json.loads(json.dumps(ctx._last_triad_raw_results[0]))
    missing_response["response_ref"] = {}
    missing_response_ctx = SimpleNamespace(
        _last_triad_raw_results=[missing_response],
        _last_scope_raw_result=ctx._last_scope_raw_result,
    )
    _, missing_response_mismatches, _ = _contributor_execution_receipts(
        missing_response_ctx, config, tmp_path
    )
    assert "response_receipt_absent:triad:t1" in missing_response_mismatches

    tampered = json.loads(json.dumps(ctx._last_triad_raw_results[0]))
    tampered["response_ref"]["redacted_projection_ref"]["sha256"] = "0" * 64
    tampered_ctx = SimpleNamespace(
        _last_triad_raw_results=[tampered],
        _last_scope_raw_result=ctx._last_scope_raw_result,
    )
    _, tampered_mismatches, _ = _contributor_execution_receipts(
        tampered_ctx, config, tmp_path
    )
    assert any(item.startswith("unreadable_response_receipt:triad:t1")
               for item in tampered_mismatches)


def test_contributor_receipts_require_settlement_but_keep_advisory_delta(tmp_path):
    config = {
        "triad_slots": [{"slot_id": "t1", "route": {
            "kind": "agent_session", "target_id": "codex=gpt-5.6-sol"},
            "effort": "high"}],
        "scope_slots": [], "review_enforcement": "blocking", "context_mode": "max",
    }
    prompt_ref = _persist_review_prompt(tmp_path, call_id="session_prompt_terminal", slot={
        "slot_id": "t1", "model": "codex=gpt-5.6-sol", "effort": "high",
        "route": "agent_session", "session_target": "codex=gpt-5.6-sol",
        "session_profile": "",
    })
    response_ref = _persist_review_response(
        tmp_path, call_id="session_terminal", transcript="raw\nEOF", usage={
            "provider": "claudexor", "delegated_route": "codex",
            "resolved_model": "gpt-5.6-sol", "applied_profile": "auto-profile",
            "applied_access": "readonly", "custody_durable": True,
            "delegated_run_id": "run-1", "settlement": {
                "settled": True, "ledger_recorded": False,
                "project_retired": True,
            },
            "capability_delta": [{"reason": "schema_unavailable_on_effective_route"}],
        },
    )
    ctx = SimpleNamespace(
        _last_triad_raw_results=[{
            "slot_id": "t1", "status": "responded",
            "prompt_ref": prompt_ref, "response_ref": response_ref,
        }],
        _last_scope_raw_result={},
    )

    _, mismatches, _ = _contributor_execution_receipts(ctx, config, tmp_path)

    assert "session_settlement_unproven:triad:t1:ledger_recorded" in mismatches
    assert not any(item.startswith("capability_delta:") for item in mismatches)


def test_contributor_receipts_accept_present_usage_less_error_payload(tmp_path):
    from ouroboros.observability import persist_call
    config = {
        "triad_slots": [{"slot_id": "t1", "route": {
            "kind": "api_chat", "target_id": "openai/gpt-5.6-sol"},
            "effort": "high"}],
        "scope_slots": [], "review_enforcement": "blocking", "context_mode": "max",
    }
    prompt_ref = _persist_review_prompt(tmp_path, call_id="error_prompt", slot={
        "slot_id": "t1", "model": "openai/gpt-5.6-sol", "effort": "high",
        "route": "api_chat", "session_target": "", "session_profile": "",
    })
    response_ref = persist_call(
        tmp_path, task_id="review", call_id="error_response", call_type="response",
        payload={"error": "Timeout after 300s"},
    )
    ctx = SimpleNamespace(
        _last_triad_raw_results=[{
            "slot_id": "t1", "status": "error",
            "prompt_ref": prompt_ref, "response_ref": response_ref,
        }],
        _last_scope_raw_result={},
    )
    receipts, mismatches, _ = _contributor_execution_receipts(ctx, config, tmp_path)
    assert mismatches == []
    assert receipts[0]["observed"]["route_kind"] is None


def test_normal_key_probe_stays_single_model_but_contributor_probes_all(monkeypatch):
    import scripts.run_external_review as module

    calls: list[str] = []
    monkeypatch.setattr(module, "_review_probe_models", lambda: ["one", "two", "three"])
    monkeypatch.setattr(
        module,
        "_probe_model_for_key",
        lambda _token, model: (calls.append(model) is None, f"ok:{model}"),
    )
    class Response:
        status_code = 200

        @staticmethod
        def json():
            return {"data": {"limit": None}}

    import httpx

    monkeypatch.setattr(httpx, "get", lambda *_args, **_kwargs: Response())

    assert _openrouter_key_health("secret")[0] is True
    assert calls == ["one"]
    calls.clear()
    assert _openrouter_key_health("secret", probe_all_models=True)[0] is True
    assert calls == ["one", "two", "three"]


def _git(repo: Path, *args: str) -> str:
    proc = subprocess.run(
        ["git", *args], cwd=str(repo), capture_output=True, text=True, check=True,
    )
    return proc.stdout


def test_isolated_checkout_freezes_the_reviewed_tree(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test")
    # Windows runners default to autocrlf=true, which rewrites checked-out
    # files to CRLF and breaks LF patch application in the detached worktree.
    _git(repo, "config", "core.autocrlf", "false")
    (repo / "a.txt").write_text("base\n", encoding="utf-8")
    _git(repo, "add", "a.txt")
    _git(repo, "commit", "-m", "base")
    (repo / "a.txt").write_text("staged change\n", encoding="utf-8")
    _git(repo, "add", "a.txt")
    staged_patch = _git(repo, "diff", "--cached", "--binary")

    import scripts.run_external_review as module

    monkeypatch.setattr(module, "REPO", repo)
    checkout_root, checkout = _create_isolated_checkout(staged_patch)
    try:
        # The frozen checkout carries the staged content in both index and tree.
        assert (checkout / "a.txt").read_text(encoding="utf-8") == "staged change\n"
        assert "a.txt" in _git(checkout, "diff", "--cached", "--name-only")
        # A later edit in the primary worktree does not leak into the checkout.
        (repo / "a.txt").write_text("post-review drift\n", encoding="utf-8")
        assert (checkout / "a.txt").read_text(encoding="utf-8") == "staged change\n"
    finally:
        _remove_isolated_checkout(checkout_root, checkout)
    assert not checkout.exists()


def test_reviewed_tree_comparison_is_untracked_safe(tmp_path):
    """A NEW staged file must not read as drift after the cycle's reset HEAD.

    The production cycle ends with ``git reset HEAD`` in the checkout, turning
    newly added files untracked; only a homogeneous re-staged comparison
    (git add -A + git diff --cached) matches the operator's staged patch.
    """
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test")
    # Windows runners default to autocrlf=true, which rewrites checked-out
    # files to CRLF and breaks LF patch application in the detached worktree.
    _git(repo, "config", "core.autocrlf", "false")
    (repo / "base.txt").write_text("base\n", encoding="utf-8")
    _git(repo, "add", "base.txt")
    _git(repo, "commit", "-m", "base")
    (repo / "brand_new.py").write_text("print('new module')\n", encoding="utf-8")
    _git(repo, "add", "brand_new.py")
    staged_patch = _git(repo, "diff", "--cached", "--binary")

    # Simulate the post-cycle state: staged patch applied, then reset HEAD.
    _git(repo, "reset", "HEAD")
    naive = _git(repo, "diff", "HEAD", "--binary")
    assert naive.strip() != staged_patch.strip()  # the trap: untracked lost

    _git(repo, "add", "-A")
    homogeneous = _git(repo, "diff", "--cached", "--binary")
    assert homogeneous.strip() == staged_patch.strip()
