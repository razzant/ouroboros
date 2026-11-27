"""Synthetic corpus for the carrier-aware update engine (spec §1.9-10, owner
batch №8 answer 6=A; mandatory v7next return, owner answers 5.12-5.14=A).

The mandatory matrix: a carrier-span conflict resolves by policy (official
side wins INSIDE the span only) and NEVER routes the update to the assisted
lane; a non-carrier conflict in the same file REMAINS a conflict; a malformed
anchor and a duplicate anchor each degrade to the ordinary assisted path;
rollback, crash and dirty-tree cases; and the honest frame that the FIRST
pre-v7 upgrade is driven by the OLD updater — documented and pinned, never
simulated. All merge corpus cases are steady-state (a 7.0.0 tree updating to
an official 7.0.1) because that is the population the ratified policy targets.
Re-proven against the update-flow redesign: the planner runs the mandatory Q8
projection + postcondition after span resolution, and the live materializer
resolves spans BEFORE the M0 baseline pin (owner decision Ф-2=A)."""

import pathlib
import subprocess

import supervisor.git_ops as git_ops
import supervisor.update_candidate as update_candidate
import supervisor.update_carriers as update_carriers
import supervisor.update_merge as update_merge
import supervisor.update_merge_plan as update_merge_plan
from ouroboros.tools.release_sync import (
    VERSION_CARRIER_SPANS,
    carrier_spans_for,
    check_worktree_version_sync,
    locate_carrier_span,
    published_download_version,
    sync_release_metadata,
    version_carrier_desyncs,
)
from tests import test_update_merge_assisted as tua
from tests.test_release_sync import _download_page, _download_references, _make_repo

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]

CARRIER_FILES = (
    "VERSION",
    "pyproject.toml",
    "web/package.json",
    "web/modules/api_types.js",
    "README.md",
    "docs/ARCHITECTURE.md",
    "uv.lock",
)


def _git(repo, *args):
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True)


def _history_rows(*versions):
    return "".join(f"| {v} | 2026-08-18 | release {v}. |\n" for v in versions)


def _download_refs(version):
    """The direct-download reference block of the tip README carrier shape.

    All proof ids are present with canonical URLs: the Q8 projection's
    postcondition (version_carrier_desyncs) checks every RELEASE_ASSET_TEMPLATES
    member once a README opts into the projection, and the readme_download_refs
    span requires the block to anchor."""
    from ouroboros.tools.release_sync import (
        RELEASE_ASSET_TEMPLATES,
        release_asset_download_url,
    )

    return "".join(
        f"[download-{proof_id}]: {release_asset_download_url(proof_id, version)}\n"
        for proof_id in RELEASE_ASSET_TEMPLATES
    )


def _write_carriers(repo, version, *, history=("7.0.0", "6.104.0"), intro="Intro line.\n"):
    (repo / "VERSION").write_text(f"{version}\n", newline="\n")
    (repo / "pyproject.toml").write_text(
        '[project]\nname = "ouroboros"\n'
        f'version = "{version}"\n'
        'description = "self-modifying agent"\n', newline="\n"
    )
    (repo / "web").mkdir(exist_ok=True)
    (repo / "web" / "package.json").write_text(
        '{\n  "name": "ouroboros-web",\n'
        f'  "version": "{version}",\n'
        '  "private": true\n}\n', newline="\n"
    )
    (repo / "web" / "modules").mkdir(exist_ok=True)
    (repo / "web" / "modules" / "api_types.js").write_text(
        f"export const GATEWAY_CONTRACT_VERSION = '{version}';\n"
        "export const OTHER = 1;\n", newline="\n"
    )
    (repo / "README.md").write_text(
        "# Ouroboros\n\n"
        f"[![Version {version}](https://img.shields.io/badge/version-{version}-green.svg)](VERSION)\n\n"
        f"{intro}\n"
        "## Version History\n\n"
        "| Version | Date | Description |\n"
        "|---------|------|-------------|\n"
        + _history_rows(*history)
        + "\n"
        + _download_refs(version), newline="\n"
    )
    (repo / "docs").mkdir(exist_ok=True)
    # encoding pinned: the header carries an em dash, and Windows' locale codec
    # (cp1252) would otherwise commit non-utf-8 bytes that the carrier-span
    # resolver cannot decode — the file then stays a conflict instead of clean.
    (repo / "docs" / "ARCHITECTURE.md").write_text(
        f"# Ouroboros v{version} — Architecture & Reference\n\nArchitecture body.\n",
        encoding="utf-8", newline="\n"
    )
    (repo / "uv.lock").write_text(
        'version = 1\n\n[[package]]\nname = "ouroboros"\n'
        f'version = "{version}"\nsource = {{ editable = "." }}\n\n'
        '[[package]]\nname = "httpx"\nversion = "0.27.0"\n', newline="\n"
    )


def _init_carrier_repo(tmp_path):
    """A synthetic 7.0.0 tree carrying all 7 release carriers plus code."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "t@example.com")
    _git(repo, "config", "user.name", "t")
    _git(repo, "config", "commit.gpgsign", "false")
    (repo / "BIBLE.md").write_text("constitution\n", newline="\n")
    (repo / "a.txt").write_text("base\n", newline="\n")
    _write_carriers(repo, "7.0.0")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "v7.0.0 baseline")
    head = _git(repo, "symbolic-ref", "--short", "HEAD").stdout.strip()
    return repo, head


def _official_bump(repo, head, version="7.0.1", *, extra=None):
    """The official target: every carrier bumped, history row prepended."""
    _git(repo, "checkout", "-q", "-b", "remote-sim")
    _write_carriers(repo, version, history=(version, "7.0.0", "6.104.0"))
    if extra:
        extra(repo)
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", f"official {version}")
    _git(repo, "checkout", "-q", head)


def _local_bump(repo, version="7.1.0", *, intro="Intro line.\n", extra=None):
    """A committed local self-modification bumping the same carrier spans."""
    _write_carriers(repo, version, history=(version, "7.0.0", "6.104.0"), intro=intro)
    if extra:
        extra(repo)
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", f"local {version}")


def _point_at(monkeypatch, tmp_path, repo, head):
    monkeypatch.setattr(git_ops, "REPO_DIR", repo)
    monkeypatch.setattr(git_ops, "BRANCH_DEV", head)
    monkeypatch.setattr(git_ops, "DRIVE_ROOT", tmp_path / "data")
    monkeypatch.setattr(git_ops, "_git_dir", lambda: repo / ".git")
    monkeypatch.setattr(
        git_ops, "_managed_update_target", lambda branch=None: ("", "", "remote-sim")
    )
    monkeypatch.setattr(
        git_ops,
        "_resolve_managed_update_target",
        lambda *_args: (
            "remote-sim",
            _git(repo, "rev-parse", "remote-sim").stdout.strip(),
            "",
        ),
    )
    (tmp_path / "data" / "logs").mkdir(parents=True, exist_ok=True)


# --- matrix case 1: a carrier-span conflict resolves by policy ---------------


def test_planner_resolves_carrier_span_conflict_to_the_official_side(tmp_path, monkeypatch):
    """Steady-state 7.0.0 -> 7.0.1: local bumped its carriers to 7.1.0, the
    official target to 7.0.1 — every carrier file conflicts inside its spans
    only, so the plan is CLEAN (a conflicted carrier never routes the update
    to the assisted lane), the built merge adopts the official spans, and the
    local non-span README edit survives (never whole-file theirs)."""
    repo, head = _init_carrier_repo(tmp_path)
    _official_bump(repo, head, "7.0.1")
    _local_bump(repo, "7.1.0", intro="Locally rewritten intro.\n")
    _point_at(monkeypatch, tmp_path, repo, head)

    plan = update_merge.plan_managed_update_merge(fetch=False, build=True)

    assert plan["kind"] == "clean", plan
    assert plan["auto_mergeable"] is True
    assert plan["recommended_strategy"] == "auto_merge"
    assert sorted(plan["carrier_resolved_paths"]) == sorted(CARRIER_FILES)
    assert plan["merge_commit"], plan
    ok, message = update_merge.apply_managed_merge_update(head, plan["merge_commit"])
    assert ok, message
    assert (repo / "VERSION").read_text() == "7.0.1\n"
    assert 'version = "7.0.1"' in (repo / "pyproject.toml").read_text()
    assert '"version": "7.0.1"' in (repo / "web" / "package.json").read_text()
    assert "GATEWAY_CONTRACT_VERSION = '7.0.1'" in (
        repo / "web" / "modules" / "api_types.js"
    ).read_text()
    readme = (repo / "README.md").read_text()
    assert "version-7.0.1-green" in readme
    assert "| 7.0.1 |" in readme and "| 7.1.0 |" not in readme
    assert "# Ouroboros v7.0.1" in (repo / "docs" / "ARCHITECTURE.md").read_text()
    # Never whole-file theirs: the local NON-span edit survived the update.
    assert "Locally rewritten intro." in readme
    # A real 2-parent merge commit landed (reviewed base first, official second).
    parents = _git(repo, "rev-list", "--parents", "-n", "1", "HEAD").stdout.split()
    assert len(parents) == 3


# --- matrix case 2: a non-carrier conflict in the same file stays a conflict -


def test_non_carrier_conflict_in_the_same_file_stays_a_conflict(tmp_path, monkeypatch):
    repo, head = _init_carrier_repo(tmp_path)

    def official_description(r):
        text = (r / "pyproject.toml").read_text()
        (r / "pyproject.toml").write_text(
            text.replace('description = "self-modifying agent"',
                         'description = "official rewrite"'), newline="\n"
        )

    def local_description(r):
        text = (r / "pyproject.toml").read_text()
        (r / "pyproject.toml").write_text(
            text.replace('description = "self-modifying agent"',
                         'description = "local rewrite"'), newline="\n"
        )

    _official_bump(repo, head, "7.0.1", extra=official_description)
    _local_bump(repo, "7.1.0", extra=local_description)
    _point_at(monkeypatch, tmp_path, repo, head)

    plan = update_merge.plan_managed_update_merge(fetch=False)

    assert plan["kind"] == "conflicting", plan
    assert "pyproject.toml" in plan["code_conflict_paths"]
    assert "pyproject.toml" not in plan["carrier_resolved_paths"]
    # The other carrier files, conflicted only inside their spans, DID resolve.
    assert "VERSION" in plan["carrier_resolved_paths"]
    assert plan["recommended_strategy"] == "assisted"


# --- matrix cases 3 + 4: malformed / duplicate anchors degrade to assisted ---


def test_malformed_anchor_degrades_to_assisted(tmp_path, monkeypatch):
    repo, head = _init_carrier_repo(tmp_path)
    _official_bump(repo, head, "7.0.1")
    _local_bump(repo, "7.1.0")
    (repo / "VERSION").write_text("not-a-version\n", newline="\n")  # anchor destroyed locally
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "malformed local VERSION")
    _point_at(monkeypatch, tmp_path, repo, head)

    plan = update_merge.plan_managed_update_merge(fetch=False)

    assert plan["kind"] == "conflicting", plan
    assert "VERSION" in plan["code_conflict_paths"]
    assert "VERSION" not in plan["carrier_resolved_paths"]


def test_duplicate_anchor_degrades_to_assisted(tmp_path, monkeypatch):
    repo, head = _init_carrier_repo(tmp_path)
    _official_bump(repo, head, "7.0.1")
    _local_bump(repo, "7.1.0")
    text = (repo / "pyproject.toml").read_text()
    (repo / "pyproject.toml").write_text(text + 'version = "9.9.9"\n', newline="\n")  # second anchor
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "duplicate local version anchor")
    _point_at(monkeypatch, tmp_path, repo, head)

    plan = update_merge.plan_managed_update_merge(fetch=False)

    assert plan["kind"] == "conflicting", plan
    assert "pyproject.toml" in plan["code_conflict_paths"]
    assert "pyproject.toml" not in plan["carrier_resolved_paths"]


# --- insertion point 2: the base re-merge, before write-tree -----------------


def test_base_re_merge_resolves_carrier_conflicts_before_write_tree(tmp_path, monkeypatch):
    """Dirty-tree case of the corpus: committed local carrier bumps PLUS dirty
    uncommitted work force the Q1=C base re-merge, whose carrier conflicts the
    engine resolves BEFORE write-tree; the dirty file never enters the built
    commit."""
    repo, head = _init_carrier_repo(tmp_path)
    _official_bump(repo, head, "7.0.1")
    _local_bump(repo, "7.1.0")
    (repo / "dirty.txt").write_text("uncommitted owner work\n", newline="\n")
    _point_at(monkeypatch, tmp_path, repo, head)

    plan = update_merge.plan_managed_update_merge(fetch=False, build=True)

    assert plan["kind"] == "clean", plan
    assert plan["local_dirty_count"] >= 1
    assert plan["merge_commit"], plan
    tree = _git(repo, "ls-tree", "-r", "--name-only", plan["merge_commit"]).stdout
    assert "dirty.txt" not in tree  # Q1=C: dirty work never enters history
    shown = _git(repo, "show", f"{plan['merge_commit']}:VERSION").stdout
    assert shown.strip() == "7.0.1"
    parents = _git(
        repo, "rev-list", "--parents", "-n", "1", plan["merge_commit"]
    ).stdout.split()
    assert parents[1:] == [plan["base_sha"], plan["target_sha"]]


# --- insertion point 3: the live assisted materializer -----------------------


def test_live_materializer_resolves_carrier_conflicts_for_the_assisted_lane(tmp_path, monkeypatch):
    """A real (non-carrier) code conflict routes the update to the assisted
    lane; the live materializer still resolves the version-carrier spans —
    BEFORE the M0 baseline is pinned (Ф-2=A) — so the resolver task only
    faces the real conflict."""
    repo, head = _init_carrier_repo(tmp_path)

    def official_code(r):
        (r / "a.txt").write_text("official code change\n", newline="\n")

    def local_code(r):
        (r / "a.txt").write_text("local code change\n", newline="\n")

    _official_bump(repo, head, "7.0.1", extra=official_code)
    _local_bump(repo, "7.1.0", extra=local_code)
    _point_at(monkeypatch, tmp_path, repo, head)
    plan = update_merge.plan_managed_update_merge(fetch=False)
    assert plan["kind"] == "conflicting", plan
    assert "a.txt" in plan["code_conflict_paths"]

    ok, message, m0_tree = update_merge.materialize_assisted_merge_live(
        head, plan["local_snapshot"], plan["target_sha"], plan["base_sha"]
    )

    assert ok, message
    assert m0_tree, message  # the mechanical baseline pinned AFTER span policy
    assert update_merge._merge_head_sha() == plan["target_sha"]
    unmerged = _git(repo, "diff", "--name-only", "--diff-filter=U").stdout.split()
    assert "a.txt" in unmerged  # the real conflict stays for the resolver
    for path in CARRIER_FILES:
        assert path not in unmerged, path
    assert (repo / "VERSION").read_text() == "7.0.1\n"
    assert "<<<<<<<" in (repo / "a.txt").read_text()
    # M0 carries the resolved carrier spans: the pinned baseline tree names the
    # official VERSION blob, not a conflicted one.
    m0_version = _git(repo, "cat-file", "-p", f"{m0_tree}:VERSION").stdout
    assert m0_version == "7.0.1\n"


# --- rollback case -----------------------------------------------------------


def test_rollback_restores_pre_update_sha_after_carrier_resolved_apply(tmp_path, monkeypatch):
    repo, head = _init_carrier_repo(tmp_path)
    _official_bump(repo, head, "7.0.1")
    _local_bump(repo, "7.1.0")
    _point_at(monkeypatch, tmp_path, repo, head)
    pre = _git(repo, "rev-parse", "HEAD").stdout.strip()
    plan = update_merge.plan_managed_update_merge(fetch=False, build=True)
    assert plan["kind"] == "clean" and plan["merge_commit"], plan
    ok, message = update_merge.apply_managed_merge_update(head, plan["merge_commit"])
    assert ok, message
    update_merge.write_update_tx({
        "phase": "pending_boot_smoke", "pre_update_sha": pre,
        "pre_update_branch": head, "target_sha": plan["target_sha"],
        "merge_commit": plan["merge_commit"],
    })
    gate_calls = []
    import supervisor.workers as workers

    monkeypatch.setattr(
        workers, "close_repo_writer_admission",
        lambda reason: gate_calls.append(("close", reason)),
    )
    monkeypatch.setattr(
        workers, "open_repo_writer_admission_after_update_abort",
        lambda expected_reason="": gate_calls.append(("open", expected_reason)) or True,
    )

    ok, message = update_merge.rollback_managed_update("carrier_test_rollback")

    assert ok, message
    assert _git(repo, "rev-parse", "HEAD").stdout.strip() == pre
    assert (repo / "VERSION").read_text() == "7.1.0\n"  # the local tree is back
    assert update_merge.read_update_tx_strict()[0] == "absent"


# --- crash cases -------------------------------------------------------------


def test_resolver_crash_degrades_the_plan_never_the_live_tree(tmp_path, monkeypatch):
    """A per-file resolver crash degrades that file to the assisted path; a
    crash of the whole resolver is swallowed by the planner's best-effort
    envelope. Neither touches the live worktree or leaks a temp worktree."""
    repo, head = _init_carrier_repo(tmp_path)
    _official_bump(repo, head, "7.0.1")
    _local_bump(repo, "7.1.0")
    _point_at(monkeypatch, tmp_path, repo, head)

    def boom(*_args, **_kwargs):
        raise RuntimeError("carrier resolver crashed")

    monkeypatch.setattr(update_carriers, "resolve_carrier_conflict_file", boom)
    plan = update_merge.plan_managed_update_merge(fetch=False)
    assert plan["kind"] == "conflicting", plan  # per-file degrade, no crash
    assert plan["carrier_resolved_paths"] == []
    assert not _git(repo, "status", "--porcelain").stdout.strip()

    monkeypatch.setattr(update_merge_plan, "resolve_carrier_conflicts", boom)
    plan2 = update_merge.plan_managed_update_merge(fetch=False)
    assert plan2["kind"] == "unknown", plan2  # planner envelope, still no crash
    assert not _git(repo, "status", "--porcelain").stdout.strip()
    worktrees = _git(repo, "worktree", "list").stdout.strip().splitlines()
    assert len(worktrees) == 1  # no leaked temp worktree


# --- SSOT + wiring pins ------------------------------------------------------


def test_every_descriptor_matches_the_live_repo_exactly_once():
    """The span SSOT must describe the REAL carriers of this checkout: every
    descriptor anchors exactly once in the live file it names."""
    for span in VERSION_CARRIER_SPANS:
        text = (REPO_ROOT / span.path).read_text(encoding="utf-8")
        status, location = locate_carrier_span(text, span)
        assert status == "ok" and location is not None, (span.carrier_id, status)
    readme_spans = carrier_spans_for("README.md")
    assert {span.carrier_id for span in readme_spans} == {
        "readme_badge", "readme_history", "readme_download_refs",
        "readme_windows_badge", "readme_windows_links", "readme_windows_quickstart",
    }
    # The span set is cut from THIS tree's carrier inventory (everything
    # sync_release_metadata writes / version_carrier_desyncs checks): the 7
    # ratified carriers + README-history, README direct-download refs & Windows
    # projection spans, uv.lock root-package mirror, web/package-lock.json header,
    # and 8 release-download anchors on each of the two public install pages.
    assert len(VERSION_CARRIER_SPANS) == 29
    assert {s.carrier_id for s in VERSION_CARRIER_SPANS} >= {
        "uv_lock_root_package", "web_package_lock_version", "readme_download_refs",
        "site_install_download_macos-arm64_button",
        "site_install_download_macos-arm64_step",
        "docs_install_download_linux-x86_64",
    }
    install_paths = {s.path for s in VERSION_CARRIER_SPANS if "install" in s.carrier_id}
    assert install_paths == {"site/install/index.html", "docs/install/index.html"}


def test_one_shared_resolver_serves_all_three_insertion_points():
    """The ratified wiring (spec §1.9-10): ONE shared resolver, called at the
    planner merge, the base re-merge and the live materializer — all with the
    official-side preference — and nowhere else in the update machinery.

    Honest frame, documented rather than simulated: the FIRST pre-v7 -> 7.0.0
    upgrade is driven by the OLD updater, whose code (the pre-redesign
    supervisor/update_merge.py) never called this engine; the parent facade
    still contains no resolver call, so the engine governs steady state only
    (7.0.0 -> 7.0.1 and beyond). The boot-recovery M0 backfill window
    (update_merge._recover_assisted_on_boot) deliberately re-runs only the Q8
    projection, not span resolution — upstream recovery semantics are the
    floor, and a carrier that stayed conflicted through that crash window
    degrades to the assisted lane (fail-safe, never fail-wrong)."""
    leaf_source = (REPO_ROOT / "supervisor" / "update_merge_plan.py").read_text(
        encoding="utf-8"
    )
    calls = leaf_source.count("resolve_carrier_conflicts(")
    assert calls == 3, calls
    assert leaf_source.count('prefer="theirs"') == 3
    parent_source = (REPO_ROOT / "supervisor" / "update_merge.py").read_text(
        encoding="utf-8"
    )
    assert "resolve_carrier_conflicts" not in parent_source


def test_resolver_rejects_an_unknown_preference(tmp_path):
    try:
        update_carriers.resolve_carrier_conflicts(str(tmp_path), [], prefer="mine")
    except ValueError as exc:
        assert "mine" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("unknown preference must be a typed refusal")


# --- carrier-only change predicate (the commit-review pack cut) ---------------


def _uv_lock(version):
    return (
        'version = 1\n\n[[package]]\nname = "ouroboros"\n'
        f'version = "{version}"\nsource = {{ editable = "." }}\n\n'
        '[[package]]\nname = "httpx"\nversion = "0.27.0"\n'
    )


def test_span_substitution_primitive_is_owned_by_the_carrier_ssot():
    """ONE span-substitution primitive: the update resolver calls the SSOT's
    ``substitute_carrier_spans`` (release_sync) and keeps no private copy, so
    the commit-review pack cut and the managed-update resolver can never drift
    on what "inside the span" means."""
    from ouroboros.tools.release_sync import substitute_carrier_spans

    assert not hasattr(update_carriers, "_substitute_spans")
    source = (REPO_ROOT / "supervisor" / "update_carriers.py").read_text(encoding="utf-8")
    assert "substitute_carrier_spans(" in source and "def substitute_carrier_spans" not in source
    spans = carrier_spans_for("uv.lock")
    before, after = _uv_lock("7.0.0"), _uv_lock("7.0.1")
    assert substitute_carrier_spans(after, spans, before) == (before, "")
    # A missing anchor on either side is the typed degradation, never a guess.
    assert substitute_carrier_spans("", spans, before)[0] is None
    assert substitute_carrier_spans(after, spans, "")[1].endswith(":preferred_side")


def test_carrier_only_change_matrix(tmp_path):
    """True only for a declared carrier whose whole change sits inside its
    declared version spans; every other shape keeps the file's full text."""
    from ouroboros.tools.release_sync import carrier_only_change

    old, new = tmp_path / "old", tmp_path / "new"
    old.mkdir()
    new.mkdir()
    _write_carriers(old, "7.0.0")
    _write_carriers(new, "7.0.1", history=("7.0.1", "7.0.0", "6.104.0"))
    for rel in CARRIER_FILES:
        before = (old / rel).read_text(encoding="utf-8")
        after = (new / rel).read_text(encoding="utf-8")
        assert before != after, rel
        assert carrier_only_change(before, after, rel) is True, rel
    # A dependency edit OUTSIDE uv.lock's root-package span keeps the full text.
    lock_before = _uv_lock("7.0.0")
    assert carrier_only_change(lock_before, _uv_lock("7.0.1").replace("0.27.0", "0.28.0"), "uv.lock") is False
    # Prose edits beside the carrier spans, a new file, a deleted file, a
    # malformed (duplicate) anchor and a non-carrier path all answer False.
    readme_before = (old / "README.md").read_text(encoding="utf-8")
    readme_after = (new / "README.md").read_text(encoding="utf-8").replace("Intro line.", "Rewritten intro.")
    assert carrier_only_change(readme_before, readme_after, "README.md") is False
    assert carrier_only_change("", _uv_lock("7.0.1"), "uv.lock") is False
    assert carrier_only_change(lock_before, "", "uv.lock") is False
    assert carrier_only_change(lock_before, _uv_lock("7.0.1") + _uv_lock("7.0.1"), "uv.lock") is False
    assert carrier_only_change("base\n", "changed\n", "a.txt") is False
    # An unchanged carrier is trivially span-only (nothing outside the spans moved).
    assert carrier_only_change(lock_before, lock_before, "uv.lock") is True


def test_carrier_postcondition_names_a_web_package_lock_the_sync_cannot_fix(tmp_path, monkeypatch):
    """MAJOR-1 (rc.15 review): the Q8 projection's postcondition promises "a
    carrier the sync cannot fix is a typed failure, never a silent success" —
    for web/package-lock.json too. A packages[""] root entry the token sync
    does not recognise (no "name" key ahead of its version) keeps the fork
    version; the projection must report the desync naming the lockfile."""
    repo, head = tua._init_repo(tmp_path)
    (repo / "VERSION").write_text("1.0.0\n", newline="\n")
    (repo / "web").mkdir()
    (repo / "web" / "package.json").write_text('{\n  "version": "1.0.0"\n}\n', newline="\n")
    (repo / "web" / "package-lock.json").write_text(
        '{\n  "name": "ouroboros-web",\n  "version": "1.0.0",\n  "lockfileVersion": 3,\n'
        '  "packages": {\n    "": {\n      "version": "1.0.0"\n    }\n  }\n}\n', newline="\n"
    )
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "carrier base")
    _git(repo, "checkout", "-q", "-b", "remote-sim")
    (repo / "VERSION").write_text("2.0.0\n", newline="\n")
    (repo / "official.txt").write_text("official\n", newline="\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "official release")
    target = _git(repo, "rev-parse", "HEAD").stdout.strip()
    _git(repo, "checkout", "-q", head)
    (repo / "VERSION").write_text("1.5.0\n", newline="\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "fork release")
    _point_at(monkeypatch, tmp_path, repo, head)
    _git(repo, "merge", "--no-commit", "--no-ff", target)

    ok, note, error = update_candidate.project_version_carriers(target)

    assert not ok, (note, error)
    assert error.startswith("carriers still desynced after token sync: ")
    assert 'web/package-lock.json (expected both root "version" entries = "2.0.0")' in error
    # The recognised root entry WAS token-synced; only the unrecognised one stayed behind.
    lock_text = (repo / "web" / "package-lock.json").read_text()
    assert '"name": "ouroboros-web",\n  "version": "2.0.0"' in lock_text
    assert '"": {\n      "version": "1.0.0"' in lock_text


# --- a managed update onto a development marker (BIBLE P9) ------------------------


def _marker_update_scene(tmp_path, monkeypatch):
    """Base at the published 7.9.0 with every carrier; ``remote-sim`` opens the marker; HEAD stays the fork."""
    repo, head = tua._init_repo(tmp_path)
    # The repository's LF text policy, as in every real checkout: what this scene's writers produce (CRLF in
    # Windows text mode, the release sync included) reaches the index stages the resolver reads as LF.
    (repo / ".gitattributes").write_bytes((REPO_ROOT / ".gitattributes").read_bytes())
    _make_repo(repo, "7.9.0")
    sync_release_metadata(str(repo))
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "published 7.9.0")
    _git(repo, "checkout", "-q", "-b", "remote-sim")
    (repo / "VERSION").write_text("7.10.0-alpha.0\n", newline="\n")
    sync_release_metadata(str(repo))
    (repo / "official.txt").write_text("official\n", newline="\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "official marker")
    target = _git(repo, "rev-parse", "HEAD").stdout.strip()
    _git(repo, "checkout", "-q", head)
    _point_at(monkeypatch, tmp_path, repo, head)
    return repo, target


def _links_in_tree(repo):
    return tuple(published_download_version((repo / rel).read_text(encoding="utf-8"))
                 for rel in ("README.md", "site/install/index.html", "docs/install/index.html"))


def test_a_marker_target_brings_its_published_links_over_a_forks_own_release(tmp_path, monkeypatch):
    repo, target = _marker_update_scene(tmp_path, monkeypatch)
    (repo / "VERSION").write_text("7.9.1\n", newline="\n")
    sync_release_metadata(str(repo))  # the fork's own numbered release moved every link to unpublished v7.9.1
    assert _links_in_tree(repo) == ("7.9.1", "7.9.1", "7.9.1")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "fork release")
    _git(repo, "merge", "--no-commit", "--no-ff", target)  # README, VERSION and friends conflict; the pages do not

    conflicted = _git(repo, "diff", "--name-only", "--diff-filter=U").stdout.split()
    assert "README.md" in conflicted and "site/install/index.html" not in conflicted
    assert update_carriers.resolve_carrier_conflicts(str(repo), conflicted, prefer="theirs")["kept"] == {}
    ok, note, error = update_candidate.project_version_carriers(target)

    assert ok, (note, error)
    assert (repo / "VERSION").read_text().strip() == "7.10.0-alpha.0"
    assert _links_in_tree(repo) == ("7.9.0", "7.9.0", "7.9.0")
    assert check_worktree_version_sync(repo) == ""
    assert _git(repo, "diff", "--name-only", target, "--", "README.md", "site/install/index.html",
                "docs/install/index.html").stdout == ""
    assert _git(repo, "diff", "--cached", "--name-only", "--diff-filter=U").stdout == ""


def test_the_next_projection_heals_the_dead_marker_links_an_older_body_merged(tmp_path, monkeypatch):
    """An installed body without the marker rule projects a marker's VERSION onto every link and commits the merge.

    It cannot be corrected in flight: the running body owns that projection. This tree carries it as
    it was committed, and the next update's projection writes back the target's published links; no
    tag or remote read decides what was published.
    """
    repo, target = _marker_update_scene(tmp_path, monkeypatch)
    _git(repo, "merge", "-q", "--no-ff", "-m", "the older body's merge", target)
    dead = "7.10.0-alpha.0"
    readme = (repo / "README.md").read_text(encoding="utf-8")
    (repo / "README.md").write_text(readme.replace(_download_references("7.9.0"), _download_references(dead)),
                                    newline="\n", encoding="utf-8")
    for rel in ("site/install/index.html", "docs/install/index.html"):
        (repo / rel).write_text(_download_page(dead), newline="\n", encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "--amend", "--no-edit")
    assert _links_in_tree(repo) == (dead, dead, dead)
    # Until then the new body reports the state rather than inheriting it silently.
    assert "direct-download links" in check_worktree_version_sync(repo)

    _git(repo, "checkout", "-q", "remote-sim")
    (repo / "landing.txt").write_text("a neutral landing\n", newline="\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "a neutral landing")
    next_target = _git(repo, "rev-parse", "HEAD").stdout.strip()
    _git(repo, "checkout", "-q", "-")
    _git(repo, "merge", "--no-commit", "--no-ff", next_target)
    assert _links_in_tree(repo) == (dead, dead, dead)  # Git alone keeps the fork's side of an untouched span

    ok, note, error = update_candidate.project_version_carriers(next_target)

    assert ok, (note, error)
    assert _links_in_tree(repo) == ("7.9.0", "7.9.0", "7.9.0")
    assert check_worktree_version_sync(repo) == ""
    assert version_carrier_desyncs(
        "7.10.0-alpha.0", download_readme_text=(repo / "README.md").read_text(encoding="utf-8")) == []
    assert _git(repo, "diff", "--cached", "--name-only", next_target, "--", "README.md", "site/install/index.html",
                "docs/install/index.html").stdout == ""


def _diverged_from_a_target_with(repo, rel, text):
    """The official tip rewrites *rel*; the fork diverges and starts merging that tip."""
    _git(repo, "checkout", "-q", "remote-sim")
    (repo / rel).write_text(text, newline="\n", encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", f"official {rel}")
    target = _git(repo, "rev-parse", "HEAD").stdout.strip()
    _git(repo, "checkout", "-q", "-")
    (repo / "local.txt").write_text("fork\n", newline="\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "fork work")
    _git(repo, "merge", "--no-commit", "--no-ff", target)
    return target


def test_a_target_whose_readme_names_no_one_release_fails_the_projection_rather_than_guess(tmp_path, monkeypatch):
    repo, official = _marker_update_scene(tmp_path, monkeypatch)
    readme = _git(repo, "show", f"{official}:README.md").stdout
    mixed = readme.replace("releases/download/v7.9.0/Ouroboros-7.9.0.dmg", "releases/download/v7.8.0/Ouroboros-7.8.0.dmg")
    assert mixed != readme
    target = _diverged_from_a_target_with(repo, "README.md", mixed)

    ok, _note, error = update_candidate.project_version_carriers(target)

    assert not ok and error.startswith("carriers still desynced after token sync: ")
    assert "expected one published plain release below 7.10.0-alpha.0" in error


def test_a_target_page_behind_its_readme_is_projected_to_the_release_the_readme_names(tmp_path, monkeypatch):
    repo, _target = _marker_update_scene(tmp_path, monkeypatch)
    target = _diverged_from_a_target_with(repo, "site/install/index.html", _download_page("7.8.0"))
    assert _links_in_tree(repo) == ("7.9.0", "7.8.0", "7.9.0")

    ok, note, error = update_candidate.project_version_carriers(target)

    assert ok, (note, error)
    assert _links_in_tree(repo) == ("7.9.0", "7.9.0", "7.9.0")


def test_the_marker_target_conflict_inventory_keeps_the_official_side_inside_the_spans(tmp_path, monkeypatch):
    """Release ancestry in the other direction: a fork already on the plain release meets the next marker."""
    repo, target = _marker_update_scene(tmp_path, monkeypatch)
    (repo / "VERSION").write_text("7.10.0\n", newline="\n")
    readme = (repo / "README.md").read_text(encoding="utf-8")
    (repo / "README.md").write_text(readme.replace("| 7.9.0 |", "| 7.10.0 | 2026-11-03 | Fork row. |\n| 7.9.0 |", 1),
                                    newline="\n", encoding="utf-8")
    sync_release_metadata(str(repo))
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "fork cut the release number locally")
    _git(repo, "merge", "--no-commit", "--no-ff", target)

    conflicted = _git(repo, "diff", "--name-only", "--diff-filter=U").stdout.split()
    assert update_carriers.resolve_carrier_conflicts(str(repo), conflicted, prefer="theirs")["kept"] == {}
    ok, note, error = update_candidate.project_version_carriers(target)

    assert ok, (note, error)
    assert _links_in_tree(repo) == ("7.9.0", "7.9.0", "7.9.0")
    assert "| 7.10.0-alpha.0 |" not in (repo / "README.md").read_text(encoding="utf-8")


def _seed_real_windows_repo(tmp_path, version: str, history_row: str, *, with_setup: bool):
    repo, head = tua._init_repo(tmp_path)
    (repo / ".gitattributes").write_bytes((REPO_ROOT / ".gitattributes").read_bytes())
    for rel in (
        "VERSION", "pyproject.toml", "uv.lock", "web/package.json", "web/package-lock.json",
        "web/modules/api_types.js", "README.md", "docs/ARCHITECTURE.md",
        "site/install/index.html", "docs/install/index.html",
    ):
        dest = repo / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        text = (REPO_ROOT / rel).read_text(encoding="utf-8")
        if rel == "README.md":
            start, end = locate_carrier_span(text, carrier_spans_for("README.md")[1])[1]
            text = f"{text[:start]}{history_row}\n{text[end:]}"
        dest.write_text(text, encoding="utf-8", newline="\n")
    (repo / "build_windows.ps1").write_text("# zip build\n", encoding="utf-8", newline="\n")
    if with_setup:
        (repo / "scripts").mkdir(exist_ok=True)
        (repo / "scripts" / "pack_windows_setup.ps1").write_text("# setup build\n", encoding="utf-8", newline="\n")
    (repo / "VERSION").write_text(f"{version}\n", encoding="utf-8", newline="\n")
    sync_release_metadata(str(repo))
    return repo, head


def test_real_install_pages_and_readme_cross_windows_installer_boundary_cleanly(tmp_path, monkeypatch):
    """Managed updates cross ZIP-only <-> setup + ZIP cleanly in both directions,
    preserve local edits inside the Windows card and quick start, and honor an
    intervening ZIP-only hotfix above 7.12.1 from the target README."""
    from ouroboros.tools.release_sync import (
        CARRIER_SPAN_PATHS,
        WINDOWS_SETUP_PROOF_ID,
        carrier_only_change,
        release_asset_download_url,
        release_metadata_findings,
        resolve_release_windows_installer,
    )

    repo, head = _seed_real_windows_repo(
        tmp_path, "7.12.1", "| 7.12.1 | 2026-11-26 | Published ZIP-only release. |", with_setup=False,
    )
    assert check_worktree_version_sync(repo) == ""
    assert release_metadata_findings(
        {p: (repo / p).read_text(encoding="utf-8") for p in CARRIER_SPAN_PATHS if (repo / p).exists()},
        windows_installer=resolve_release_windows_installer(repo, "7.12.1"),
    ) == []
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "published 7.12.1 ZIP-only")
    _git(repo, "tag", "v7.12.1")
    base_sha = _git(repo, "rev-parse", "HEAD").stdout.strip()
    site_7121 = (repo / "site/install/index.html").read_text(encoding="utf-8")
    readme_7121 = (repo / "README.md").read_text(encoding="utf-8")

    # Official cuts 7.12.2 (adds signed per-user Windows installer beside portable ZIP).
    _git(repo, "checkout", "-q", "-b", "remote-sim")
    (repo / "scripts").mkdir(exist_ok=True)
    (repo / "scripts" / "pack_windows_setup.ps1").write_text("# setup build\n", encoding="utf-8", newline="\n")
    (repo / "VERSION").write_text("7.12.2\n", encoding="utf-8", newline="\n")
    (repo / "README.md").write_text(
        readme_7121.replace("| 7.12.1 |", "| 7.12.2 | 2026-11-27 | Windows installer. |\n| 7.12.1 |", 1),
        encoding="utf-8", newline="\n",
    )
    sync_release_metadata(str(repo))
    site_7122 = (repo / "site/install/index.html").read_text(encoding="utf-8")
    readme_7122 = (repo / "README.md").read_text(encoding="utf-8")
    assert carrier_only_change(site_7121, site_7122, "site/install/index.html") is True
    assert carrier_only_change(readme_7121, readme_7122, "README.md") is True
    for span in (*carrier_spans_for("site/install/index.html"), *carrier_spans_for("README.md")):
        target_text = site_7122 if span.path == "site/install/index.html" else readme_7122
        assert locate_carrier_span(target_text, span)[0] == "ok", span.carrier_id
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "official 7.12.2 setup+zip")

    # Fork adds a local note INSIDE <article id="windows"> and INSIDE ### Windows quick start.
    _git(repo, "checkout", "-q", head)
    (repo / "VERSION").write_text("7.12.2-alpha.0\n", encoding="utf-8", newline="\n")
    fork_site = site_7121.replace(
        "      </article>\n      <article id=\"linux\">",
        '        <p class="local-note">FORK_WINDOWS_CARD_NOTE</p>\n      </article>\n      <article id="linux">',
        1,
    )
    assert carrier_only_change(site_7121, fork_site, "site/install/index.html") is False
    (repo / "site/install/index.html").write_text(fork_site, encoding="utf-8", newline="\n")
    (repo / "README.md").write_text(
        readme_7121.replace("### Linux quick start", "\nFORK_WINDOWS_QUICKSTART_NOTE\n\n### Linux quick start", 1),
        encoding="utf-8", newline="\n",
    )
    sync_release_metadata(str(repo))
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "fork marker + local Windows card & quick-start notes")
    _point_at(monkeypatch, tmp_path, repo, head)

    plan = update_merge.plan_managed_update_merge(fetch=False, build=True)
    assert plan["kind"] == "clean", plan
    ok, message = update_merge.apply_managed_merge_update(head, plan["merge_commit"])
    assert ok, message
    merged_site = (repo / "site/install/index.html").read_text(encoding="utf-8")
    merged_readme = (repo / "README.md").read_text(encoding="utf-8")
    assert "FORK_WINDOWS_CARD_NOTE" in merged_site
    assert "FORK_WINDOWS_QUICKSTART_NOTE" in merged_readme
    assert release_asset_download_url(WINDOWS_SETUP_PROOF_ID, "7.12.2") in merged_site
    assert "Run the signed per-user installer" in merged_site
    assert "Download for Windows (.exe)" in merged_readme
    assert check_worktree_version_sync(repo) == ""

    # Reverse direction onto an intervening ZIP-only hotfix 7.12.2 under marker 7.13.0-alpha.0:
    # target_readme has NO windows-setup-x64 even though 7.12.2 > 7.12.1.
    _git(repo, "checkout", "-q", "-b", "official-zip-hotfix-marker", base_sha)
    (repo / "VERSION").write_text("7.13.0-alpha.0\n", encoding="utf-8", newline="\n")
    (repo / "README.md").write_text(
        readme_7121.replace("| 7.12.1 |", "| 7.12.2 | 2026-11-27 | ZIP-only hotfix. |\n| 7.12.1 |", 1),
        encoding="utf-8", newline="\n",
    )
    sync_release_metadata(str(repo), published_version="7.12.2", windows_installer=False)
    (repo / "official-note.txt").write_text("hotfix cycle\n", encoding="utf-8", newline="\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "official marker pointing to ZIP-only 7.12.2 hotfix")
    hotfix_marker_target = _git(repo, "rev-parse", "HEAD").stdout.strip()

    _git(repo, "checkout", "-q", head)
    _git(repo, "merge", "--no-commit", "--no-ff", hotfix_marker_target)
    conflicted = _git(repo, "diff", "--name-only", "--diff-filter=U").stdout.split()
    assert update_carriers.resolve_carrier_conflicts(str(repo), conflicted, prefer="theirs")["kept"] == {}
    ok, note, error = update_candidate.project_version_carriers(hotfix_marker_target)
    assert ok, (note, error)
    assert _links_in_tree(repo) == ("7.12.2", "7.12.2", "7.12.2")
    final_readme = (repo / "README.md").read_text(encoding="utf-8")
    final_site = (repo / "site/install/index.html").read_text(encoding="utf-8")
    assert WINDOWS_SETUP_PROOF_ID not in final_readme
    assert WINDOWS_SETUP_PROOF_ID not in final_site
    assert "Extract the ZIP to a permanent folder" in final_site
    assert "installer" not in final_site.split('<article id="windows">', 1)[1].split("</article>", 1)[0].lower()
    final_qs = final_readme.split("### Windows quick start", 1)[1].split("### Linux quick start", 1)[0]
    assert "(.exe)" not in final_qs and "installer" not in final_qs.lower()
    assert "FORK_WINDOWS_CARD_NOTE" in final_site
    assert "FORK_WINDOWS_QUICKSTART_NOTE" in final_readme
    assert check_worktree_version_sync(repo) == ""


def test_managed_update_preserves_custom_captions_and_uses_numbered_target_build_inventory(tmp_path, monkeypatch):
    """Custom captions (including case-only variants) stay outside the replaceable span,
    and project_version_carriers resolves a numbered target from its build inventory rather
    than trusting a wrongly ZIP-only README."""
    from ouroboros.tools.release_sync import WINDOWS_SETUP_PROOF_ID, WINDOWS_ZIP_PROOF_ID, carrier_only_change, release_asset_download_url

    repo, head = _seed_real_windows_repo(
        tmp_path, "7.12.2", "| 7.12.2 | 2026-11-27 | Windows installer. |", with_setup=True,
    )
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "published 7.12.2 setup+zip")
    base_site = (repo / "site/install/index.html").read_text(encoding="utf-8")
    readme_7122 = (repo / "README.md").read_text(encoding="utf-8")

    # Official: marker 7.12.3-alpha.0 + numbered 7.12.3 with Setup build but wrongly ZIP-only README.
    _git(repo, "checkout", "-q", "-b", "remote-sim")
    (repo / "VERSION").write_text("7.12.3-alpha.0\n", encoding="utf-8", newline="\n")
    sync_release_metadata(str(repo))
    (repo / "marker-note.txt").write_text("marker\n", encoding="utf-8", newline="\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "official 7.12.3-alpha.0 marker")
    marker_target = _git(repo, "rev-parse", "HEAD").stdout.strip()

    (repo / "VERSION").write_text("7.12.3\n", encoding="utf-8", newline="\n")
    (repo / "README.md").write_text(
        readme_7122.replace("| 7.12.2 |", "| 7.12.3 | 2026-11-28 | Next cut. |\n| 7.12.2 |", 1),
        encoding="utf-8", newline="\n",
    )
    sync_release_metadata(str(repo), windows_installer=False)  # broken ZIP-only README on Setup build
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "official 7.12.3 setup build with wrong zip-only readme")
    broken_numbered_target = _git(repo, "rev-parse", "HEAD").stdout.strip()

    # Fork customizes the Windows installer caption (case-only variant) on site/install/index.html.
    _git(repo, "checkout", "-q", head)
    custom_caption = "DOWNLOAD FOR WINDOWS (.exe)"
    fork_site = base_site.replace(">Download for Windows (.exe)</a>", f">{custom_caption}</a>", 1)
    assert carrier_only_change(base_site, fork_site, "site/install/index.html") is False
    (repo / "site/install/index.html").write_text(fork_site, encoding="utf-8", newline="\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "fork case-only custom Windows button caption")
    _point_at(monkeypatch, tmp_path, repo, head)

    # 1. Conflicting release degrades cleanly instead of overwriting the custom caption.
    _git(repo, "merge", "--no-commit", "--no-ff", broken_numbered_target)
    conflicted = _git(repo, "diff", "--name-only", "--diff-filter=U").stdout.split()
    resolved = update_carriers.resolve_carrier_conflicts(str(repo), conflicted, prefer="theirs")
    assert resolved["kept"]["site/install/index.html"].startswith("malformed_anchor:site_install_download_windows-x64")
    assert custom_caption in (repo / "site/install/index.html").read_text(encoding="utf-8")
    _git(repo, "merge", "--abort")

    # 2. Non-conflicting marker adoption updates hrefs while preserving the case-only caption.
    _git(repo, "merge", "--no-commit", "--no-ff", marker_target)
    conflicted_m = _git(repo, "diff", "--name-only", "--diff-filter=U").stdout.split()
    assert update_carriers.resolve_carrier_conflicts(str(repo), conflicted_m, prefer="theirs")["kept"] == {}
    ok, note, error = update_candidate.project_version_carriers(marker_target)
    assert ok, (note, error)
    adopted_site = (repo / "site/install/index.html").read_text(encoding="utf-8")
    assert f">{custom_caption}</a>" in adopted_site
    assert release_asset_download_url(WINDOWS_SETUP_PROOF_ID, "7.12.2") in adopted_site
    assert release_asset_download_url(WINDOWS_ZIP_PROOF_ID, "7.12.2") in adopted_site
    _git(repo, "merge", "--abort")

    # 3. project_version_carriers on a numbered Setup-build target with wrong ZIP-only README
    # uses target build inventory (Setup + ZIP), NOT the target's wrong README.
    (repo / "site/install/index.html").write_text(base_site, encoding="utf-8", newline="\n")
    _git(repo, "add", "site/install/index.html")
    _git(repo, "commit", "-q", "-m", "restore canonical caption")
    _git(repo, "merge", "--no-commit", "--no-ff", broken_numbered_target)
    conflicted_n = _git(repo, "diff", "--name-only", "--diff-filter=U").stdout.split()
    assert update_carriers.resolve_carrier_conflicts(str(repo), conflicted_n, prefer="theirs")["kept"] == {}
    ok, note, error = update_candidate.project_version_carriers(broken_numbered_target)
    assert ok, (note, error)
    assert release_asset_download_url(WINDOWS_SETUP_PROOF_ID, "7.12.3") in (repo / "README.md").read_text(encoding="utf-8")
    assert release_asset_download_url(WINDOWS_SETUP_PROOF_ID, "7.12.3") in (repo / "site/install/index.html").read_text(encoding="utf-8")
    assert check_worktree_version_sync(repo) == ""
