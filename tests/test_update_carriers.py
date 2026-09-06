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
    locate_carrier_span,
)
from tests import test_update_merge_assisted as tua

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
    (repo / "VERSION").write_text(f"{version}\n")
    (repo / "pyproject.toml").write_text(
        '[project]\nname = "ouroboros"\n'
        f'version = "{version}"\n'
        'description = "self-modifying agent"\n'
    )
    (repo / "web").mkdir(exist_ok=True)
    (repo / "web" / "package.json").write_text(
        '{\n  "name": "ouroboros-web",\n'
        f'  "version": "{version}",\n'
        '  "private": true\n}\n'
    )
    (repo / "web" / "modules").mkdir(exist_ok=True)
    (repo / "web" / "modules" / "api_types.js").write_text(
        f"export const GATEWAY_CONTRACT_VERSION = '{version}';\n"
        "export const OTHER = 1;\n"
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
        + _download_refs(version)
    )
    (repo / "docs").mkdir(exist_ok=True)
    # encoding pinned: the header carries an em dash, and Windows' locale codec
    # (cp1252) would otherwise commit non-utf-8 bytes that the carrier-span
    # resolver cannot decode — the file then stays a conflict instead of clean.
    (repo / "docs" / "ARCHITECTURE.md").write_text(
        f"# Ouroboros v{version} — Architecture & Reference\n\nArchitecture body.\n",
        encoding="utf-8",
    )
    (repo / "uv.lock").write_text(
        'version = 1\n\n[[package]]\nname = "ouroboros"\n'
        f'version = "{version}"\nsource = {{ editable = "." }}\n\n'
        '[[package]]\nname = "httpx"\nversion = "0.27.0"\n'
    )


def _init_carrier_repo(tmp_path):
    """A synthetic 7.0.0 tree carrying all 7 release carriers plus code."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "t@example.com")
    _git(repo, "config", "user.name", "t")
    _git(repo, "config", "commit.gpgsign", "false")
    (repo / "BIBLE.md").write_text("constitution\n")
    (repo / "a.txt").write_text("base\n")
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
                         'description = "official rewrite"')
        )

    def local_description(r):
        text = (r / "pyproject.toml").read_text()
        (r / "pyproject.toml").write_text(
            text.replace('description = "self-modifying agent"',
                         'description = "local rewrite"')
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
    (repo / "VERSION").write_text("not-a-version\n")  # anchor destroyed locally
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
    (repo / "pyproject.toml").write_text(text + 'version = "9.9.9"\n')  # second anchor
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
    (repo / "dirty.txt").write_text("uncommitted owner work\n")
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
        (r / "a.txt").write_text("official code change\n")

    def local_code(r):
        (r / "a.txt").write_text("local code change\n")

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
        workers, "open_repo_writer_admission",
        lambda expected_reason="": gate_calls.append(("open", expected_reason)),
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
    }
    # The span set is cut from THIS tree's carrier inventory (everything
    # sync_release_metadata writes / version_carrier_desyncs checks): the 7
    # ratified carriers + README-history, the README direct-download reference
    # block, the uv.lock root-package mirror, the web/package-lock.json header
    # (both root version entries in one span), and 8 release-download anchors on
    # each of the two public install pages (macos-arm64 appears twice there).
    assert len(VERSION_CARRIER_SPANS) == 26
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
    engine_doc = update_carriers.__doc__ or ""
    assert "OLD updater" in engine_doc and "steady state" in engine_doc


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
    (repo / "VERSION").write_text("1.0.0\n")
    (repo / "web").mkdir()
    (repo / "web" / "package.json").write_text('{\n  "version": "1.0.0"\n}\n')
    (repo / "web" / "package-lock.json").write_text(
        '{\n  "name": "ouroboros-web",\n  "version": "1.0.0",\n  "lockfileVersion": 3,\n'
        '  "packages": {\n    "": {\n      "version": "1.0.0"\n    }\n  }\n}\n'
    )
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "carrier base")
    _git(repo, "checkout", "-q", "-b", "remote-sim")
    (repo / "VERSION").write_text("2.0.0\n")
    (repo / "official.txt").write_text("official\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "official release")
    target = _git(repo, "rev-parse", "HEAD").stdout.strip()
    _git(repo, "checkout", "-q", head)
    (repo / "VERSION").write_text("1.5.0\n")
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
