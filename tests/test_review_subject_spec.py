"""The review subject as an object (ARCHITECTURE §6 "Subject operation").

``ReviewSubjectSpec`` → ``freeze_subject`` / ``isolated_checkout`` →
``run_parallel_review(subject=...)``, and the three identities derived from the
frozen subject: the settled-record reuse key (a), the rebuttal round (b) and the
custody retry key (c). The system repo's ``index`` subject must be byte-identical
to today's commit gate: same api pack, same session task, same scope brief, same
seats, same aggregate.
"""

from __future__ import annotations

import hashlib
import pathlib
import subprocess
from types import SimpleNamespace

import pytest

from ouroboros import review_ledger as rl
from ouroboros.tools.parallel_review import _prepare_scope_rows as _REAL_PREPARE_SCOPE_ROWS
from ouroboros.tools.review_binary_context import capture_staged_diff
from ouroboros.tools.review_subject import (
    FrozenSubject,
    ReviewSubjectSpec,
    assigned_seats,
    freeze_subject,
    isolated_checkout,
    reuse_or_none,
    review_retry_key,
    review_reuse_key,
    review_round_sha,
)


def _git(repo, *args, **kw):
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, check=True, **kw)


def _out(repo, *args):
    return _git(repo, *args).stdout.strip()


def _repo(path, *, files):
    path.mkdir(parents=True)
    _git(path, "init", "-q")
    _git(path, "config", "user.email", "t@example.com")
    _git(path, "config", "user.name", "t")
    _git(path, "config", "commit.gpgsign", "false")
    _git(path, "config", "core.autocrlf", "false")
    for name, text in files.items():
        (path / name).write_bytes(text.encode("utf-8"))
    _git(path, "add", "-A")
    _git(path, "commit", "-q", "-m", "base")
    return path


def _system_repo(tmp_path):
    """The installed body stand-in: HEAD plus one staged hunk (the gate's subject)."""
    repo = _repo(tmp_path / "system", files={"x.txt": "x\n"})
    (repo / "x.txt").write_bytes(b"y\n")
    _git(repo, "add", "-A")
    return repo


def _ctx(system_repo, tmp_path, **extra):
    data = tmp_path / "data"
    (data / "logs").mkdir(parents=True, exist_ok=True)
    ns = SimpleNamespace(
        repo_dir=str(system_repo), drive_root=str(data), task_id="t-subject", task_metadata=None,
        _review_history=[], _review_advisory=[], _review_iteration_count=0,
        drive_logs=lambda: data / "logs",
    )
    for key, value in extra.items():
        setattr(ns, key, value)
    return ns


def _index_spec(repo, **over):
    spec = {"root_kind": "system_repo", "root": str(repo), "kind": "index", "surface": "commit_gate", "layer": "body"}
    spec.update(over)
    return ReviewSubjectSpec(**spec)


# ---------------------------------------------------------------------------
# freeze_subject: the system index is the gate's binding, byte for byte
# ---------------------------------------------------------------------------


def test_system_index_subject_is_the_gate_binding(tmp_path):
    from ouroboros.tools.git_review_cycle import _fingerprint_staged_diff

    repo = _system_repo(tmp_path)
    ctx = _ctx(repo, tmp_path)
    frozen = freeze_subject(ctx, _index_spec(repo))
    binding = _fingerprint_staged_diff(pathlib.Path(repo))["binding"]

    assert isinstance(frozen, FrozenSubject) and frozen.is_system_index
    # One subject, one identity: the frozen digest IS the gate's binding digest.
    assert frozen.diff_sha == binding["diff_sha256"]
    assert frozen.tree_sha == binding["tree_sha"] == _out(repo, "write-tree")
    assert frozen.parent_sha == binding["parents"][0] == _out(repo, "rev-parse", "HEAD")
    assert frozen.diff_text == capture_staged_diff(pathlib.Path(repo))
    assert "+y" in frozen.diff_text and "-x" in frozen.diff_text
    assert frozen.name_status == (("M", "x.txt"),) and frozen.managed is None and frozen.checkout == ""
    assert frozen.spec.governance_root == str(pathlib.Path(repo).resolve()) and frozen.review_root == str(repo)
    assert frozen.record_subject() == {
        "root_kind": "system_repo", "root": str(repo), "kind": "index", "base": frozen.parent_sha, "head": "",
        "tree_sha": frozen.tree_sha, "diff_sha": frozen.diff_sha, "checkout": "",
    }
    # The -U0 fit rung re-renders the same subject, never a different capture.
    assert frozen.render_prompt_diff(unified=0) == capture_staged_diff(pathlib.Path(repo), unified=0)
    assert frozen.staged_tree == frozen.tree_sha and frozen.m0_tree == frozen.parent_sha


def test_spec_validation_fails_closed(tmp_path):
    repo = _system_repo(tmp_path)
    ctx = _ctx(repo, tmp_path)
    with pytest.raises(ValueError, match="kind"):
        freeze_subject(ctx, _index_spec(repo, kind="staged"))
    with pytest.raises(ValueError, match="root_kind"):
        freeze_subject(ctx, _index_spec(repo, root_kind="elsewhere"))
    with pytest.raises(ValueError, match="isolated_checkout"):
        freeze_subject(ctx, _index_spec(repo, kind="base..head", base="HEAD", head="HEAD"))


# ---------------------------------------------------------------------------
# Golden: the wave on the frozen system index == today's gate wave
# ---------------------------------------------------------------------------


def _row_plan(routes):
    from ouroboros.review_execution import ReviewRouteKind

    kinds = {"api": ReviewRouteKind.API_CHAT, "session": ReviewRouteKind.AGENT_SESSION}
    return {
        "models": [f"m/{i}-{r}" for i, r in enumerate(routes)], "routes": [kinds[r] for r in routes],
        "efforts": ["" for _ in routes], "session_targets": ["" for _ in routes],
        "session_profiles": ["" for _ in routes], "use_local": [False for _ in routes],
        "slot_ids": [f"slot_{i}" for i, _ in enumerate(routes)],
    }


def _run_wave(monkeypatch, ctx, subject=None):
    """The REAL assembly of both triad deliveries and every scope brief; only the
    paid dispatch seams are patched. Returns what each reviewer would be GIVEN."""
    from ouroboros.tools import parallel_review as pr
    from ouroboros.tools import review as review_mod
    from ouroboros.tools.scope_review import ScopeReviewResult
    import ouroboros.reviewer_slot_config as slot_cfg

    given = {"scope_briefs": []}
    monkeypatch.setattr(slot_cfg, "commit_triad_delivery", lambda: _row_plan(["api", "session", "session"]))
    monkeypatch.setattr(review_mod, "calibrated_input_token_limit", lambda *a, **k: 2_000_000)

    def fake_dispatch(_ctx, _msg, prepared):
        given["prompt"], given["session_task"] = prepared["prompt"], prepared["session_task"]
        given["models"], given["target_repo"] = list(prepared["models"]), str(prepared["target_repo"])
        return None

    monkeypatch.setattr(review_mod, "_dispatch_unified_review", fake_dispatch)

    def capture_rows(*a, **k):
        rows = _REAL_PREPARE_SCOPE_ROWS(*a, **k)
        for row in rows:
            assert row["final"] is None, row["final"]
            given["scope_briefs"].append(row["prepared"]["session_task"])
        return rows

    monkeypatch.setattr(pr, "_prepare_scope_rows", capture_rows)
    monkeypatch.setattr(pr, "run_scope_review", lambda _ctx, _msg, **kw: ScopeReviewResult(
        blocked=False, status="responded", model_id=kw["prepared"]["scope_model_id"]))
    review_err, scope_result, reason, advisory = pr.run_parallel_review(
        ctx, "golden subject commit", goal="the goal", scope="the scope", subject=subject)
    given.update(review_err=review_err, scope_status=scope_result.status, reason=reason, advisory=advisory,
                 structured=dict(ctx._last_review_structured), retry_key=ctx._last_review_structured["retry_key"])
    return given


def test_frozen_system_index_wave_is_byte_identical_to_the_gate(tmp_path, monkeypatch):
    repo = _system_repo(tmp_path)
    today = _run_wave(monkeypatch, _ctx(repo, tmp_path))
    ctx = _ctx(repo, tmp_path)
    frozen = freeze_subject(ctx, _index_spec(repo))
    parametric = _run_wave(monkeypatch, ctx, subject=frozen)

    assert "+y" in today["prompt"] and today["session_task"] and today["scope_briefs"]
    for key in ("prompt", "session_task", "scope_briefs", "models", "target_repo",
                "review_err", "scope_status", "reason", "advisory"):
        assert parametric[key] == today[key], key
    volatile = {"started_ts", "retry_key", "subject", "layer"}
    assert {k: v for k, v in parametric["structured"].items() if k not in volatile} == \
        {k: v for k, v in today["structured"].items() if k not in volatile}
    assert parametric["structured"]["subject"] == frozen.record_subject() and parametric["structured"]["layer"] == "body"
    assert "subject" not in today["structured"]
    # Identity (c): the subject's retry key was set on the context BEFORE dispatch.
    assert parametric["retry_key"] == review_retry_key(frozen) == ctx._current_review_retry_key
    # The gate's own key (set by its free-cycle gate) is kept when present.
    kept = _ctx(repo, tmp_path, _current_review_retry_key="gate-key")
    assert _run_wave(monkeypatch, kept, subject=freeze_subject(kept, _index_spec(repo)))["retry_key"] == "gate-key"


def test_frozen_foreign_base_head_wave_runs_the_core_layer_on_every_delivery(tmp_path, monkeypatch):
    """The mirror of the golden case: a foreign ``base..head`` subject under the
    core layer. All three deliveries (api packet, session task, scope brief) are
    assembled by the SAME code as the gate's wave, with the layer threaded through
    every builder: the universal checklist and the subject's own navigation are
    delivered; the body's constitution, standing disclosures, body-layer section
    and reference books are not."""
    from ouroboros.tools.review_multi_model import _CONSTITUTIONAL_PREAMBLE, triad_api_messages

    system = _system_repo(tmp_path)
    foreign, base, head = _foreign_repo(tmp_path, docs={"README.md": "# Foreign\n\n## Usage\n\nRun it.\n"})
    ctx = _ctx(system, tmp_path)
    spec = ReviewSubjectSpec(root_kind="active_workspace", root=str(foreign), kind="base..head", base=base, head=head,
                             surface="change", layer="core", body_fact="false", body_how="foreign_remote")
    with isolated_checkout(ctx, spec) as frozen:
        given = _run_wave(monkeypatch, ctx, subject=frozen)
        checkout = frozen.checkout

    assert given["review_err"] is None and given["scope_status"] == "responded"
    assert given["target_repo"] == checkout and given["structured"]["layer"] == "core"
    deliveries = {"prompt": given["prompt"], "session_task": given["session_task"], **{
        f"scope_brief_{i}": brief for i, brief in enumerate(given["scope_briefs"])}}
    assert len(given["scope_briefs"]) == 1
    for name, text in deliveries.items():
        # The packet and the brief carry the frozen diff; the retrieving session reads the checkout.
        assert name == "session_task" or ("+two" in text and "-one" in text), name
        # The universal rule set and the subject's own navigation are delivered …
        assert "## Change Review Checklist" in text or "## Intent / Scope Review Checklist" in text, name
        assert "## Governance navigation (core layer)" in text and "### Subject documents" in text, name
        assert "README.md" in text and "Usage" in text and checkout in text, name
        # … the body's governance is not: no constitution text, no standing
        # disclosures, no body-layer section or items, no book navigation.
        assert "BIBLE.md (Full Text)" not in text and "P1 Continuity" not in text, name
        assert "CHECKLISTS_ARCHIVE" not in text and "## Ouroboros Body Layer" not in text, name
        assert "| 10 |" not in text and "version_bump" not in text, name
        assert "docs/ARCHITECTURE.md" not in text and "(navigation map)" not in text.replace("## README.md (navigation map)", ""), name
        assert "Its Constitution is BIBLE.md" not in text, name
    assert "## Change Review Checklist" in deliveries["prompt"] and "## Change Review Checklist" in deliveries["session_task"]
    # The api head of a core-layer row carries no constitutional preamble either.
    messages, bible_text = triad_api_messages(given["prompt"], 0, "turn", layer="core")
    system_text = "".join(block.get("text", "") if isinstance(block, dict) else str(block)
                          for block in ([messages[0]["content"]] if isinstance(messages[0]["content"], str)
                                        else messages[0]["content"]))
    assert bible_text == "" and _CONSTITUTIONAL_PREAMBLE not in system_text and "BIBLE" not in system_text
    # The same subject under the body layer (treat_as_body) carries the body's rules.
    body_spec = ReviewSubjectSpec(**{**spec.__dict__, "layer": "body", "body_fact": "true", "body_how": "treat_as_body"})
    with isolated_checkout(_ctx(system, tmp_path), body_spec) as body_frozen:
        body = _run_wave(monkeypatch, _ctx(system, tmp_path), subject=body_frozen)
    assert "## Ouroboros Body Layer" in body["prompt"] and "Its Constitution is BIBLE.md" in body["prompt"]
    assert "## Governance navigation (core layer)" not in body["prompt"]


# ---------------------------------------------------------------------------
# base..head in an isolated checkout under the data root; worktree of a live tree
# ---------------------------------------------------------------------------


def _foreign_repo(tmp_path, *, docs=None):
    repo = _repo(tmp_path / "foreign", files={"a.txt": "one\n", "keep.txt": "k\n"})
    base = _out(repo, "rev-parse", "HEAD")
    (repo / "a.txt").write_text("two\n", encoding="utf-8")
    (repo / "new.bin").write_bytes(b"\x00\x01\x02\xff")
    for name, text in (docs or {}).items():
        (repo / name).write_text(text, encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "proposal")
    return repo, base, _out(repo, "rev-parse", "HEAD")


def test_an_index_or_worktree_subject_is_read_against_its_base_when_one_is_named(tmp_path):
    """``review_change(subject=index|worktree, base=<rev>)`` promises a tree judged
    against ``base``; the frozen parent, diff, patch and name-status follow it, and
    the record's ``base`` is that commit — not HEAD. Without ``base`` the parent is
    HEAD and the system index stays the gate's own subject."""
    system = _system_repo(tmp_path)
    foreign, base, head = _foreign_repo(tmp_path)
    (foreign / "keep.txt").write_text("staged later\n", encoding="utf-8")
    _git(foreign, "add", "keep.txt")
    ctx = _ctx(system, tmp_path)

    plain = freeze_subject(ctx, ReviewSubjectSpec(root_kind="active_workspace", root=str(foreign), kind="index", surface="change"))
    assert plain.parent_sha == head and plain.spec.base == head and plain.record_subject()["base"] == head
    assert plain.name_status == (("M", "keep.txt"),) and "+staged later" in plain.diff_text and "+two" not in plain.diff_text

    against = freeze_subject(ctx, ReviewSubjectSpec(root_kind="active_workspace", root=str(foreign), kind="index",
                                                    base="HEAD~1", surface="change"))
    assert against.parent_sha == base and against.spec.base == base and against.record_subject()["base"] == base
    assert against.tree_sha == plain.tree_sha == _out(foreign, "write-tree")
    assert against.name_status == (("M", "a.txt"), ("M", "keep.txt"), ("A", "new.bin"))
    assert "+two" in against.diff_text and "+staged later" in against.diff_text and against.diff_sha != plain.diff_sha
    with isolated_checkout(ctx, ReviewSubjectSpec(root_kind="active_workspace", root=str(foreign), kind="index",
                                                  base=base, surface="change")) as checked:
        assert checked.parent_sha == base and _out(checked.checkout, "rev-parse", "HEAD") == base
        assert _out(checked.checkout, "write-tree") == against.tree_sha
        assert (pathlib.Path(checked.checkout) / "keep.txt").read_text(encoding="utf-8") == "staged later\n"

    live = freeze_subject(ctx, ReviewSubjectSpec(root_kind="active_workspace", root=str(foreign), kind="worktree",
                                                 base=base, surface="change"))
    assert live.parent_sha == base and live.spec.base == base and live.name_status == against.name_status
    # The system index named against HEAD is the gate's subject; against another
    # commit it is an ordinary tree delta (the gate never reviews that).
    assert freeze_subject(ctx, _index_spec(system)).is_system_index
    _git(system, "commit", "-q", "-m", "landed")
    (system / "x.txt").write_text("z\n", encoding="utf-8")
    _git(system, "add", "-A")
    older = freeze_subject(ctx, _index_spec(system, base="HEAD~1"))
    assert not older.is_system_index and older.managed is None and older.parent_sha == _out(system, "rev-parse", "HEAD~1")
    assert "-x" in older.diff_text and "+z" in older.diff_text
    with pytest.raises(ValueError):
        freeze_subject(ctx, _index_spec(system, base="not-a-revision"))


def test_a_frozen_index_rerenders_its_own_trees_at_u0_never_the_live_index(tmp_path):
    """The -U0 fit rung of a frozen index subject is parent→tree of the FROZEN
    subject. The live root's index may move after the freeze (the author stages
    more); a recapture of ``--cached`` there would review bytes the record never
    bound. Only the gate's own subject keeps the gate's live capture."""
    system = _system_repo(tmp_path)
    foreign, base, _head = _foreign_repo(tmp_path)
    (foreign / "keep.txt").write_text("staged first\n", encoding="utf-8")
    _git(foreign, "add", "keep.txt")
    ctx = _ctx(system, tmp_path)
    at_head = freeze_subject(ctx, ReviewSubjectSpec(root_kind="active_workspace", root=str(foreign), kind="index", surface="change"))
    against = freeze_subject(ctx, ReviewSubjectSpec(root_kind="active_workspace", root=str(foreign), kind="index",
                                                    base=base, surface="change"))
    assert not at_head.is_system_index and not against.is_system_index
    expected = {frozen: _out(foreign, "diff", "--no-ext-diff", "--no-textconv", "--no-color", "--unified=0",
                             frozen.parent_sha, frozen.tree_sha) for frozen in (at_head, against)}

    # The author stages more after the freeze: the live index is now a different tree.
    (foreign / "keep.txt").write_text("staged later\n", encoding="utf-8")
    (foreign / "extra.txt").write_text("extra\n", encoding="utf-8")
    _git(foreign, "add", "-A")
    assert _out(foreign, "write-tree") != at_head.tree_sha
    for frozen in (at_head, against):
        rendered = frozen.render_prompt_diff(unified=0)
        assert rendered.strip() == expected[frozen].strip()
        assert "+staged first" in rendered and "staged later" not in rendered and "extra" not in rendered
    assert "+two" in against.render_prompt_diff(unified=0) and "+two" not in at_head.render_prompt_diff(unified=0)

    # The gate's subject is the one live capture: the body's own staged index against HEAD.
    gate = freeze_subject(ctx, _index_spec(system))
    assert gate.is_system_index and gate.render_prompt_diff(unified=0).strip() == _out(
        system, "diff", "--cached", "--no-ext-diff", "--no-textconv", "--no-color", "--unified=0")


def test_base_head_subject_reads_an_isolated_checkout_in_the_data_root(tmp_path):
    system = _system_repo(tmp_path)
    foreign, base, head = _foreign_repo(tmp_path)
    (foreign / "a.txt").write_text("drift after the proposal\n", encoding="utf-8")  # live edits must not leak
    ctx = _ctx(system, tmp_path)
    spec = ReviewSubjectSpec(root_kind="active_workspace", root=str(foreign), kind="base..head", base="HEAD~1",
                             head="HEAD", surface="change")
    checkouts_root = (tmp_path / "data").resolve() / "state" / "review_checkouts"

    with isolated_checkout(ctx, spec) as frozen:
        checkout = pathlib.Path(frozen.checkout)
        assert checkout.is_dir() and checkout.parent.parent == checkouts_root and checkout.name == "repo"
        assert frozen.review_root == str(checkout) and not frozen.is_system_index
        assert (frozen.spec.base, frozen.spec.head) == (base, head)  # resolved to exact shas
        assert frozen.parent_sha == base and frozen.tree_sha == _out(foreign, "rev-parse", f"{head}^{{tree}}")
        assert _out(checkout, "write-tree") == frozen.tree_sha and _out(checkout, "rev-parse", "HEAD") == base
        assert (checkout / "a.txt").read_text(encoding="utf-8") == "two\n"  # the head tree, not the live one
        assert (checkout / "new.bin").read_bytes() == b"\x00\x01\x02\xff"
        assert "+two" in frozen.diff_text and "-one" in frozen.diff_text and "Binary files" in frozen.diff_text
        assert frozen.name_status == (("M", "a.txt"), ("A", "new.bin"))
        patch = subprocess.run(["git", "-C", str(foreign), "diff", "--binary", "--no-ext-diff", "--no-textconv", base, head],
                               capture_output=True, check=True).stdout
        assert frozen.diff_sha == hashlib.sha256(patch.decode("utf-8", "replace").strip().encode("utf-8")).hexdigest()
        # Governance is ALWAYS the installed body, never the foreign root.
        assert frozen.spec.governance_root == str(pathlib.Path(system).resolve())
        assert frozen.record_subject()["checkout"] == str(checkout) and frozen.record_subject()["head"] == head
        assert "@@ -1 +1 @@" in frozen.render_prompt_diff(unified=0)
        assert len(_out(foreign, "worktree", "list").splitlines()) == 2
    assert not checkout.exists() and not checkout.parent.exists()
    assert len(_out(foreign, "worktree", "list").splitlines()) == 1
    assert (foreign / "a.txt").read_text(encoding="utf-8") == "drift after the proposal\n"  # the live tree is untouched


def test_base_head_requires_an_ancestor_base(tmp_path):
    system = _system_repo(tmp_path)
    foreign, _base, _head = _foreign_repo(tmp_path)
    trunk = _out(foreign, "rev-parse", "--abbrev-ref", "HEAD")
    _git(foreign, "checkout", "-q", "-b", "side", "HEAD~1")
    (foreign / "side.txt").write_text("s\n", encoding="utf-8")
    _git(foreign, "add", "-A")
    _git(foreign, "commit", "-q", "-m", "side")
    ctx = _ctx(system, tmp_path)
    spec = ReviewSubjectSpec(root_kind="active_workspace", root=str(foreign), kind="base..head", base=trunk, head="side")
    with pytest.raises(ValueError, match="not an ancestor"):
        with isolated_checkout(ctx, spec):
            pytest.fail("a non-ancestor base must not produce a subject")
    assert not ((tmp_path / "data").resolve() / "state" / "review_checkouts").exists() or not any(
        ((tmp_path / "data").resolve() / "state" / "review_checkouts").iterdir())


def test_worktree_subject_freezes_the_live_tree(tmp_path):
    system = _system_repo(tmp_path)
    foreign, _base, head = _foreign_repo(tmp_path)
    (foreign / "a.txt").write_text("three\n", encoding="utf-8")  # unstaged edit
    (foreign / "untracked.txt").write_text("u\n", encoding="utf-8")  # untracked file
    ctx = _ctx(system, tmp_path)
    frozen = freeze_subject(ctx, ReviewSubjectSpec(root_kind="active_workspace", root=str(foreign), kind="worktree"))

    assert frozen.parent_sha == head and frozen.tree_sha != _out(foreign, "rev-parse", "HEAD^{tree}")
    assert frozen.name_status == (("M", "a.txt"), ("A", "untracked.txt"))
    assert "+three" in frozen.diff_text and "+u" in frozen.diff_text and frozen.checkout == ""
    assert frozen.review_root == str(foreign) and not frozen.is_system_index and frozen.managed is None
    assert _out(foreign, "diff", "--cached", "--name-only") == ""  # the repo's own index is untouched
    assert "@@ -1 +1 @@" in frozen.render_prompt_diff(unified=0)
    again = freeze_subject(ctx, ReviewSubjectSpec(root_kind="active_workspace", root=str(foreign), kind="worktree"))
    assert (again.diff_sha, again.tree_sha) == (frozen.diff_sha, frozen.tree_sha)
    # The same head with a different base is a DIFFERENT subject.
    with isolated_checkout(ctx, ReviewSubjectSpec(root_kind="active_workspace", root=str(foreign), kind="base..head",
                                                  base="HEAD~1", head="HEAD")) as ranged:
        assert ranged.tree_sha == _out(foreign, "rev-parse", "HEAD^{tree}") and ranged.diff_sha != frozen.diff_sha
        assert review_retry_key(ranged) != review_retry_key(frozen)


# ---------------------------------------------------------------------------
# Identities: (a) reuse key, (b) rebuttal round + shared ceiling, (c) retry key
# ---------------------------------------------------------------------------


def _seat_rows(status="responded"):
    return [{"slot_id": f"s{i}", "model_id": m, "status": status, "raw_text": "[]", "parsed_items": [], "usd": 0.01}
            for i, m in enumerate(("openai/gpt-5", "anthropic/claude-x", "google/gemini"), 1)]


def _record_facts(frozen, *, reuse_key, pending=False, task_id="task-1"):
    rows = _seat_rows("pending" if pending else "responded")
    return {
        "task_id": task_id, "root_task_id": task_id, "repo_dir": frozen.spec.root, "governance_root": frozen.spec.governance_root,
        "goal": "g", "scope": "s", "enforcement": "blocking", "enforcement_blocks": True, "blocked": False,
        "block_reason": "", "triad_raw": rows, "scope_raw": {}, "pending": pending,
        "structured": {"triad_prompt": "T", "scope_brief": "", "started_ts": "2026-10-07T00:00:00+00:00",
                       "triad_rows": [{"slot_id": r["slot_id"], "model": r["model_id"], "route": "api_chat"} for r in rows],
                       "scope_rows": [], "retry_key": review_retry_key(frozen), "subject": frozen.record_subject(),
                       "layer": frozen.spec.layer},
        "reuse_key": reuse_key, "review_contract_fingerprint": "cf", "review_wave_id": "wave-1",
    }


def _keys(frozen, **over):
    args = {"rules_sha": "rules", "layer": frozen.spec.layer, "assigned": assigned_seats(["s1", "s2", "s3"], []),
            "enforcement": "blocking", "contract_fp": "cf"}
    args.update(over)
    return review_reuse_key(frozen, **args)


def test_reuse_key_returns_the_settled_record_without_a_wave(tmp_path):
    repo = _system_repo(tmp_path)
    ctx = _ctx(repo, tmp_path)
    drive = pathlib.Path(ctx.drive_root)
    frozen = freeze_subject(ctx, _index_spec(repo))
    key = _keys(frozen)
    assert len(key) == 64 and key == _keys(frozen)  # stable
    assert reuse_or_none(drive, key) is None and rl.find_reusable(drive, "") is None

    record = rl.build_wave_record(_record_facts(frozen, reuse_key=key), surface="commit_gate", drive_root=drive)
    assert record.fingerprints["reuse_key"] == key and record.fingerprints["retry_key"] == review_retry_key(frozen)
    assert record.subject == {**frozen.record_subject(), "candidate_branch": ""}
    assert record.brief["checklist"]["layer"] == "body"
    rl.write_record(drive, record)
    reused = reuse_or_none(drive, key)
    assert reused is not None and reused["reused"] is True and reused["usd"] == 0.0
    assert reused["record_id"] == record.record_id and reused["record"]["verdict"]["aggregate"] == "PASS"
    # Another surface, another composition, another layer, another enforcement or
    # another rules source is another key — a new wave, not a reuse.
    other = freeze_subject(ctx, _index_spec(repo, surface="change"))
    for variant in (
        _keys(other), _keys(frozen, assigned=assigned_seats(["s1", "s2"], ["scope"])), _keys(frozen, layer="core"),
        _keys(frozen, enforcement="advisory"), _keys(frozen, rules_sha="other-rules"), _keys(frozen, contract_fp="cf2"),
    ):
        assert variant != key and reuse_or_none(drive, variant) is None
    # The composition hashes as rows, however it is spelled.
    assert _keys(frozen, assigned=[("s3", "change"), ("s1", "change"), ("s2", "change")]) == key
    assert _keys(frozen, assigned=[{"seat_id": s, "parts": ["change"]} for s in ("s1", "s2", "s3")]) == key


def test_pending_or_refused_records_are_never_reused(tmp_path):
    repo = _system_repo(tmp_path)
    ctx = _ctx(repo, tmp_path)
    drive = pathlib.Path(ctx.drive_root)
    frozen = freeze_subject(ctx, _index_spec(repo))
    key = _keys(frozen)
    pending = rl.build_wave_record(_record_facts(frozen, reuse_key=key, pending=True), surface="commit_gate", drive_root=drive)
    rl.write_record(drive, pending)
    assert pending.state == "pending" and reuse_or_none(drive, key) is None
    facts = _record_facts(frozen, reuse_key=key)
    facts.update(dispatch_refusal={"kind": "review_cycles_exhausted", "message": "no"}, triad_raw=[],
                 structured={**facts["structured"], "triad_rows": []})
    refused = rl.build_wave_record(facts, surface="commit_gate", drive_root=drive)
    rl.write_record(drive, refused)
    assert refused.verdict["aggregate"] == "NOT_DISPATCHED" and reuse_or_none(drive, key) is None
    # Once the wave settles with a verdict, the key reuses it.
    settled = rl.build_wave_record(_record_facts(frozen, reuse_key=key), surface="commit_gate", drive_root=drive)
    rl.write_record(drive, settled)
    assert reuse_or_none(drive, key)["record_id"] == settled.record_id


def test_rebuttal_is_a_new_round_and_the_ceiling_stays_common_per_root(tmp_path, monkeypatch):
    """Identity (b): a new ``review_rebuttal`` (its content sha) is one more paid
    round of the SAME subject — the reuse key changes, the identical-diff refusal
    does not fire — while the paid-cycle ceiling counts it and stays common to the
    root task across subjects: a second subject under an exhausted root is refused."""
    from ouroboros.review_state import CommitAttemptRecord, _utc_now, make_repo_key, update_state
    from ouroboros.tools.commit_gate import check_identical_verdict_refusal, check_review_cycles_ceiling

    monkeypatch.delenv("OUROBOROS_REVIEW_MAX_CYCLES", raising=False)
    repo = _system_repo(tmp_path)
    ctx = _ctx(repo, tmp_path)
    drive = pathlib.Path(ctx.drive_root)
    frozen = freeze_subject(ctx, _index_spec(repo))
    first_rebuttal = hashlib.sha256(b"new evidence").hexdigest()
    rebutted_round = review_round_sha(frozen, rebuttal_sha=first_rebuttal)
    key, rebutted = _keys(frozen), _keys(frozen, round_sha=rebutted_round)
    assert rebutted != key and _keys(frozen, round_sha=rebutted_round) == rebutted  # repeating it reuses the round

    repo_key = make_repo_key(pathlib.Path(repo))

    def _verdict_block(state, *, attempt, rebuttal=""):
        state.attempts.append(CommitAttemptRecord(
            ts=_utc_now(), commit_message="msg", status="blocked", block_reason="critical_findings",
            block_class="verdict", repo_key=repo_key, tool_name="commit_reviewed", task_id="root-1", attempt=attempt,
            phase="blocking_review", paid=True, pre_review_fingerprint=frozen.diff_sha, root_task_id="root-1",
            rebuttal_sha256=rebuttal, review_contract_fingerprint="cf"))

    update_state(drive, lambda state: _verdict_block(state, attempt=1))
    # The identical subject without a rebuttal is refused for free ...
    assert "IDENTICAL_DIFF_REFUSED" in check_identical_verdict_refusal(ctx, frozen.diff_sha, contract_fingerprint="cf")
    # ... a NEW rebuttal buys exactly one paid round (the refusal cache does not fire) ...
    assert check_identical_verdict_refusal(ctx, frozen.diff_sha, rebuttal_sha256=first_rebuttal, contract_fingerprint="cf") == ""
    assert check_review_cycles_ceiling(ctx, root_task_id="root-1") is None  # 1 of 2 paid
    update_state(drive, lambda state: _verdict_block(state, attempt=2, rebuttal=first_rebuttal))
    # ... and the SPENT rebuttal is refused again, while the round counted toward the ceiling.
    assert "repeated rebuttal" in check_identical_verdict_refusal(
        ctx, frozen.diff_sha, rebuttal_sha256=first_rebuttal, contract_fingerprint="cf")
    exhausted = check_review_cycles_ceiling(ctx, root_task_id="root-1")
    assert exhausted is not None and exhausted["cycles_paid"] == 2 and "REVIEW_CYCLES_EXHAUSTED" in exhausted["message"]
    # A second subject under the same root (new bytes, new key) does NOT reset the ceiling.
    (repo / "x.txt").write_text("z\n", encoding="utf-8")
    _git(repo, "add", "-A")
    second = freeze_subject(ctx, _index_spec(repo))
    assert second.diff_sha != frozen.diff_sha and _keys(second) != key
    assert check_identical_verdict_refusal(ctx, second.diff_sha, contract_fingerprint="cf") == ""  # not identical
    assert check_review_cycles_ceiling(ctx, root_task_id="root-1") is not None  # still exhausted
    assert check_review_cycles_ceiling(ctx, root_task_id="root-2") is None  # another root, its own ceiling


def test_retry_key_is_stable_per_subject_and_distinct_across_subjects(tmp_path):
    repo = _system_repo(tmp_path)
    ctx = _ctx(repo, tmp_path)
    frozen = freeze_subject(ctx, _index_spec(repo))
    from ouroboros.review_state import make_repo_key

    expected = f"review:{make_repo_key(pathlib.Path(repo))}:index:{frozen.diff_sha}:commit_gate"
    assert review_retry_key(frozen) == expected == review_retry_key(freeze_subject(ctx, _index_spec(repo)))
    # Another surface of the same bytes is another physical review; so are other bytes.
    assert review_retry_key(freeze_subject(ctx, _index_spec(repo, surface="change"))) != expected
    # The logical round rides the key (identity b → c): a round is a new physical
    # operation of the same bytes, a retry of the SAME round the same one.
    round_sha = review_round_sha(frozen, rebuttal_sha="r1", questions=["q?"], goal="g", scope="s")
    keyed = review_retry_key(frozen, round_sha=round_sha)
    assert keyed == f"{expected}:{round_sha[:16]}" == review_retry_key(freeze_subject(ctx, _index_spec(repo)), round_sha=round_sha)
    assert keyed != review_retry_key(frozen, round_sha=review_round_sha(frozen, rebuttal_sha="r2", questions=["q?"], goal="g", scope="s"))
    (repo / "x.txt").write_text("z\n", encoding="utf-8")
    _git(repo, "add", "-A")
    assert review_retry_key(freeze_subject(ctx, _index_spec(repo))) != expected


def test_the_logical_round_is_the_whole_request_in_both_identities(tmp_path):
    """Identity (b) is what the author asked THIS time — rebuttal, questions, goal,
    scope — and the resolved revisions the record names. Each enters the reuse key
    (a changed brief never reuses an old answer; two commits with one tree are two
    rounds) and the custody retry key (a new round never replays the previous
    round's answers out of custody); the identical request keeps both keys."""
    system = _system_repo(tmp_path)
    foreign, base, head = _foreign_repo(tmp_path)
    ctx = _ctx(system, tmp_path)
    _git(foreign, "commit", "-q", "--allow-empty", "-m", "empty")  # one more revision, the same tree
    moved = _out(foreign, "rev-parse", "HEAD")
    (foreign / "keep.txt").write_text("k2\n", encoding="utf-8")
    _git(foreign, "add", "-A")
    frozen = freeze_subject(ctx, ReviewSubjectSpec(root_kind="active_workspace", root=str(foreign), kind="index", surface="change"))
    asked = {"rebuttal_sha": "", "questions": ["Is it bounded?"], "goal": "Bound it", "scope": "keep.txt"}
    round_sha = review_round_sha(frozen, **asked)
    assert round_sha == review_round_sha(frozen, **asked) == rl.round_sha_of(**asked, base=frozen.parent_sha, head="")
    reuse, retry = _keys(frozen, round_sha=round_sha), review_retry_key(frozen, round_sha=round_sha)
    for changed in ({"rebuttal_sha": "r"}, {"questions": []}, {"questions": ["Is it bounded?", "Fast?"]},
                    {"goal": "Unbound it"}, {"scope": ""}):
        other = review_round_sha(frozen, **{**asked, **changed})
        assert other != round_sha, changed
        assert _keys(frozen, round_sha=other) != reuse and review_retry_key(frozen, round_sha=other) != retry, changed

    # Two revision pairs with ONE tree (the empty commit on top) are two rounds.
    with isolated_checkout(ctx, ReviewSubjectSpec(root_kind="active_workspace", root=str(foreign), kind="base..head",
                                                  base=base, head=head, surface="change")) as first, \
            isolated_checkout(ctx, ReviewSubjectSpec(root_kind="active_workspace", root=str(foreign), kind="base..head",
                                                     base=base, head=moved, surface="change")) as second:
        assert first.tree_sha == second.tree_sha and first.diff_sha == second.diff_sha
        assert review_round_sha(first, **asked) != review_round_sha(second, **asked)
        assert _keys(first, round_sha=review_round_sha(first, **asked)) != _keys(second, round_sha=review_round_sha(second, **asked))
