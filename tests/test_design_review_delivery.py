"""DESIGN navigation follows body identity and actual review delivery."""

from __future__ import annotations

import subprocess
from hashlib import sha256

import pytest

# Initialize review's module-level root aliases before registry tests patch
# review_helpers.REPO_ROOT; a lazy registry import would retain the fake root.
from ouroboros.tools import review as _review  # noqa: F401
from ouroboros.tools.governance_context import governance_context
from tests._review_substrate_shared import user_text as _user_text
from tests.test_governance_context import repo as _governance_repo
from tests.test_plan_task_tool import CLEAN, _record, _slots, _write
from tests.test_plan_task_tool import harness as _harness

governance_repo = _governance_repo
harness = _harness


DESIGN = "# Design\n\n## Interaction meaning\n\nOriginal interaction rule.\n\n## Appearance\n\nOriginal appearance rule.\n"
DESIGN_BIBLE = '## Design bible: "Airy dopamine"\n\n' + ''.join(
    f'{i}. **Principle {i}.** Governing rule.\n   More of this principle.\n' for i in range(1, 8))


def _design(root, text=DESIGN):
    path = root / "docs" / "DESIGN.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


@pytest.mark.parametrize("delegated", [False, True], ids=["native", "external-session"])
def test_deep_review_separates_installed_governance_from_candidate_sources(
    governance_repo, tmp_path, monkeypatch, delegated,
):
    from ouroboros import deep_self_review, review_execution
    from ouroboros.tools import review_helpers
    from ouroboros.tools.scope_required_sources import source_text_identity
    from tests.test_deep_review_slot import (
        _FakeSessionExecutor, _REPORT, _ScriptedLLM, _native_row, _session_row, _tool_call,
    )

    monkeypatch.setenv("OPENROUTER_API_KEY", "test")
    monkeypatch.setattr(review_helpers, "REPO_ROOT", governance_repo)
    source = _design(governance_repo)
    candidate = tmp_path / "candidate"
    _design(candidate, "## Proposed replacement\n")
    proposed_bible = "# Proposed constitution\nCandidate-only proposal.\n"
    (candidate / "BIBLE.md").write_bytes(proposed_bible.encode("utf-8"))
    required = [{"root": "active_workspace", "path": "BIBLE.md",
                 **source_text_identity(proposed_bible.encode())}]
    llm = _ScriptedLLM([
        {"tool_calls": [_tool_call("read_file", {"root": "system_repo", "path": "BIBLE.md"}, "c1"),
                        _tool_call("read_file", {"path": "BIBLE.md"}, "c2")]},
        {"content": _REPORT},
    ])
    if delegated:
        monkeypatch.setattr(review_execution, "_review_route_executor", _FakeSessionExecutor)
        monkeypatch.setattr(deep_self_review, "_session_route_reason", lambda row: "")
    report, usage = deep_self_review.run_deep_self_review(
        candidate, tmp_path / "drive", llm, lambda _: None,
        slot=_session_row() if delegated else _native_row(), required_sources=required)
    if delegated:
        request = _FakeSessionExecutor.instances[-1].assignment.request
        brief = request.session_task
        assert request.session_root == str(candidate)
        assert request.policy["native_required_sources"] == required
    else:
        brief = _user_text(next(m["content"] for m in llm.calls[0]["messages"] if m["role"] == "user"))
        installed, subject = usage["native_tool_receipts"]
        assert installed["opened_root"] == "system_repo"
        assert subject["opened_root"] == "active_workspace"
        assert subject["source_revision"] == required[0]["source_revision"]
        assert "BIBLE.md read in full" in report
    assert (governance_repo / "BIBLE.md").read_text() in brief
    assert proposed_bible not in brief and "Proposed replacement" not in brief
    assert str(source) in brief and "Interaction meaning" in brief

@pytest.mark.parametrize(("touched", "delivery", "surface"), [
    ("ouroboros/event.py", "packet", "triad"),
    ("ouroboros/event.py", "packet", "deep_self_review"),
    ("web/modules/chat.js", "retrieving", "scope"),
])
def test_noninline_design_keeps_map_and_direct_source(governance_repo, surface, delivery, touched):
    path = _design(governance_repo)
    context = governance_context(governance_repo, surface=surface, delivery=delivery,
                                 touched_paths=[touched], usable_window_tokens=0)

    assert "Interaction meaning" in context.navigation
    assert str(path) in context.navigation
    assert "Original interaction rule." not in context.navigation + context.selected_inline
    assert "availability is not evidence of reading or application" in context.navigation
    assert "docs/DESIGN.md" not in context.inline_whole_documents
    row = next(row for row in context.manifest if row["path"] == "docs/DESIGN.md")
    assert row["disposition"] == "navigation" and row["chars"] == len(DESIGN)
    if touched.startswith("web/"):
        assert "inline share exhausted" in row["reason"]
    if delivery == "packet":
        assert "no repository tools" in context.navigation and "NOT delivered" in context.navigation


def test_inline_design_is_content_without_duplicate_map(governance_repo):
    _design(governance_repo)
    context = governance_context(governance_repo, surface="triad", touched_paths=["web/ui.js"],
                                 usable_window_tokens=1_000_000)
    assert context.inline_whole_documents["docs/DESIGN.md"] == DESIGN
    assert "Original interaction rule." in context.selected_inline
    assert "Interaction meaning" not in context.navigation


@pytest.mark.parametrize("window,inline", [(0, False), (1_000_000, True)])
@pytest.mark.parametrize("changed", [False, True])
@pytest.mark.parametrize("delivery", ["packet", "retrieving"])
def test_python_word_evidence_uses_only_design_bible(governance_repo, window, inline, changed, delivery):
    from ouroboros.review_words import compare_sources

    _design(governance_repo, DESIGN_BIBLE + '\n**Seed**\n\nFull design recipes.\n' + DESIGN)
    before = 'TEXT = "Read the result"'
    after = 'TEXT = "Read the updated result"' if changed else before + "\n# comment only\n"
    words = compare_sources([("ouroboros/notice.py", before, after)])
    context = governance_context(governance_repo, surface="triad", touched_paths=["ouroboros/notice.py"],
                                 usable_window_tokens=window, review_words=words, delivery=delivery)
    assert "docs/DESIGN.md" not in context.inline_whole_documents
    assert (DESIGN_BIBLE.strip() in context.selected_inline) is (inline and changed)
    assert "Full design recipes." not in context.selected_inline
    assert "Interaction meaning" in context.navigation
    if changed:
        row = next(row for row in context.manifest if row["path"] == "docs/DESIGN.md#Design bible")
        assert row["disposition"] == ("inline" if inline else "navigation")
        assert "excerpt" in row["reason"]
        assert row["chars"] == len(DESIGN_BIBLE.strip())


def test_bible_excerpt_follows_the_numbered_list_not_a_fixed_count(governance_repo):
    from ouroboros.review_words import compare_sources

    eighth = "8. **Principle 8.** A later rule.\n   Its continuation.\n"
    _design(governance_repo, DESIGN_BIBLE + eighth + "\n**Seed**\n\nFull design recipes.\n" + DESIGN)
    words = compare_sources([("ouroboros/notice.py", 'TEXT = "Read the result"', 'TEXT = "Read the updated result"')])
    context = governance_context(governance_repo, surface="triad", touched_paths=["ouroboros/notice.py"],
                                 usable_window_tokens=1_000_000, review_words=words)
    assert "Its continuation." in context.selected_inline and "principles 1–8" in context.selected_inline
    assert "Full design recipes." not in context.selected_inline


def test_missing_bible_excerpt_never_falls_back_to_whole_design(governance_repo):
    from ouroboros.review_words import compare_sources

    _design(governance_repo)
    words = compare_sources([("notice.py", '', 'LABEL = "Resume"')])
    context = governance_context(governance_repo, surface="triad", touched_paths=["notice.py"],
                                 usable_window_tokens=1_000_000, review_words=words)
    assert "Original interaction rule." not in context.selected_inline
    assert "Interaction meaning" in context.navigation
    assert any("excerpt unavailable" in row["reason"] for row in context.manifest)


@pytest.mark.parametrize("layer", ["body", "core"])
def test_final_packet_preserves_design_delivery_and_omission_facts(governance_repo, layer):
    from ouroboros.tools.review_admission import assemble_packet_prompt

    path = _design(governance_repo)
    context = governance_context(governance_repo, surface="triad", delivery="packet", layer=layer,
                                 touched_paths=["web/ui.js"], usable_window_tokens=0)
    prompt, stable_end = assemble_packet_prompt(
        layer=layer, checklist_section="Review the change.", governance=context,
        goal_section="Check signals", scope_section="Change", files_section="No DESIGN body attached",
        diff_text="diff --git a/web/ui.js b/web/ui.js\n", changed_files="web/ui.js")
    assert ("Interaction meaning" in prompt) == (layer == "body")
    assert (str(path) in prompt) == (layer == "body")
    assert "Original interaction rule." not in prompt
    if layer == "body":
        assert "Interaction meaning" not in prompt[:stable_end]
        assert "were NOT delivered" in prompt[stable_end:]


@pytest.mark.parametrize("delivery", ["packet", "retrieving"])
def test_missing_design_is_named_without_claiming_content(governance_repo, delivery):
    path = governance_repo / "docs" / "DESIGN.md"
    path.unlink()
    context = governance_context(governance_repo, surface="triad", delivery=delivery,
                                 touched_paths=["web/ui.js"], usable_window_tokens=1_000_000)
    assert str(path) in context.navigation and "source unavailable" in context.navigation
    assert "No DESIGN content or navigation delivered" in context.navigation
    assert "docs/DESIGN.md" not in context.inline_whole_documents
    row = next(row for row in context.manifest if row["path"] == "docs/DESIGN.md")
    assert row["disposition"] == "navigation" and row["chars"] == 0
    assert "not found" in row["reason"]


def test_foreign_review_indexes_own_design_without_ouroboros_rules(governance_repo, tmp_path):
    from ouroboros.review_words import compare_sources

    _design(governance_repo)
    foreign = tmp_path / "foreign"
    _design(foreign, "# Foreign product\n\n## Foreign colors\n\nIts own rule.\n")
    context = governance_context(governance_repo, surface="triad", layer="core", subject_root=foreign,
                                 touched_paths=["web/ui.js"], usable_window_tokens=1_000_000,
                                 review_words=compare_sources([("web/ui.js", "", "const label = 'Resume';")]))
    assert "Foreign colors" in context.navigation
    assert "Interaction meaning" not in context.navigation + context.selected_inline
    assert str(governance_repo / "docs/DESIGN.md") not in context.navigation
    assert next(row for row in context.manifest if row["path"] == "docs/DESIGN.md")["disposition"] == "not_applicable"


@pytest.mark.parametrize("delegated", [False, True], ids=["native", "external-session"])
def test_code_and_coupling_brief_retains_installed_design_source(governance_repo, tmp_path, monkeypatch, delegated):
    from ouroboros.tools import review_brief_coupling as coupling

    source = _design(governance_repo)
    candidate = tmp_path / "candidate"
    _design(candidate, "# Candidate proposal\n\n## Candidate replacement\n\nProposed rule.\n")
    monkeypatch.setattr(coupling, "first_send_bound", lambda brief: 1_000_000)
    text, manifest = coupling.build_retrieving_brief(candidate, coupling.BriefInputs(
        governance_repo_dir=governance_repo, delegated=delegated, diff_text="diff --git a/event.py b/event.py\n",
        touched_paths=("event.py",), required_sources=[], touched_manifest=[],
        checklist_section="Review the change.", parts=("change", "coupling")))

    assert "Part 1" in text and "Part 2" in text
    assert "Interaction meaning" in text and str(source) in text
    assert "Candidate replacement" not in text
    assert next(row for row in manifest["governance_manifest"]
                if row["path"] == "docs/DESIGN.md")["disposition"] == "navigation"


@pytest.mark.parametrize("root", [None, "system_repo"])
def test_native_registry_reads_installed_design_outside_home(tmp_path, monkeypatch, root):
    from ouroboros.review_native_episode import inspection_registry
    from ouroboros.tools import review_helpers
    from ouroboros.tools.core_file_tools import delivered_source_prefix

    # Docker topology: /app and /candidate are siblings of HOME=/root.
    installed, candidate, home = (tmp_path / name for name in ("app", "candidate", "root"))
    home.mkdir()
    source = _design(installed)
    proposed = _design(candidate, "# Proposed\n\n## Candidate replacement\n\nProposed rule.\n")
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr(review_helpers, "REPO_ROOT", installed)
    context = governance_context(installed, surface="triad", delivery="retrieving",
                                 touched_paths=["event.py"], usable_window_tokens=0)
    assert str(source) in context.navigation and "Interaction meaning — lines 3-6" in context.navigation

    registry, reader, _ = inspection_registry(str(candidate), tmp_path / "data")
    arguments = {"path": str(source), "start_line": 3, "max_lines": 4}
    if root:
        arguments["root"] = root
    result = registry.execute_result("read_file", arguments)
    assert result.status == "ok" and "Original interaction rule." in result.text, result.text
    assert "Proposed rule." not in result.text and "Original appearance rule." not in result.text
    assert reader.last_read_view["target"] == str(source)
    assert reader.last_read_view["source_revision"] == sha256(source.read_bytes()).hexdigest()
    assert reader.last_read_view["opened_root"] == "system_repo"
    assert delivered_source_prefix(reader.last_read_view, result.text, len(result.text)) == "".join(
        DESIGN.splitlines(keepends=True)[2:6])
    result = registry.execute_result("read_file", {"root": "system_repo", "path": "docs/DESIGN.md"})
    assert "Original interaction rule." in result.text and "Proposed rule." not in result.text

    for arguments in ({"path": "docs/DESIGN.md"},
                      {"path": str(proposed)},
                      {"root": "active_workspace", "path": "docs/DESIGN.md"}):
        result = registry.execute_result("read_file", arguments)
        assert result.status == "ok" and "Proposed rule." in result.text, result.text
        assert "Original interaction rule." not in result.text
        assert reader.last_read_view["target"] == str(proposed)
        assert reader.last_read_view["source_revision"] == sha256(proposed.read_bytes()).hexdigest()
        assert reader.last_read_view["opened_root"] == "active_workspace"

    # Explicit roots never silently substitute another tree; off-home unrelated
    # files and mutation remain outside this inspection registry's authority.
    unrelated = _design(tmp_path / "unrelated", "## Unrelated\nMust not be delivered.\n")
    for arguments in ({"root": "active_workspace", "path": str(source)},
                      {"root": "system_repo", "path": str(proposed)},
                      {"path": str(unrelated)}, {"root": "user_files", "path": str(unrelated)}):
        result = registry.execute_result("read_file", arguments)
        assert result.status != "ok", result.text
        assert "Must not be delivered." not in result.text
    result = registry.execute_result("write_file", {"root": "system_repo", "path": str(source), "content": "changed"})
    assert result.status != "ok" and source.read_text() == DESIGN

    source.unlink()
    result = registry.execute_result("read_file", {"path": str(source)})
    assert "NOT_FOUND" in result.text and "Proposed rule." not in result.text


# The planning organ (``plan_task``): which subjects are Ouroboros's own body is ONE fact,
# ``plan_ask.plan_is_body``, recorded on the revision. A body plan's stable prompt
# (``plan_ask.stable_prompt``) carries BIBLE.md, ARCHITECTURE.md and the installed DESIGN
# navigation map to every participant; a non-body plan carries none of them (owner 4.3=A).
# The retired engine kept a separate "constitutional" gate and sent a body copy or declared
# own-body research only the map; by the organ's design (body plan = the installed body, a
# recognized body copy, or declared own-body research) both are body plans with the whole pack.

_PACKET_SENTENCE = "This participant has no repository tools"
_SESSION_SENTENCE = "Read relevant original DESIGN sections with your own tools"


def _plan_ask(harness, ctx, *, affected_paths, attach=None):
    """One ``plan_task`` write+ask to an API seat and a session seat, through the organ:
    ``(revision, api system prompt, api user content, session task)`` as each seat received it."""
    harness.state["slots"] = _slots(("api1", "m/a"), ("sess1", "cursor=grok", "session"))
    substrate = harness.install({"api1": CLEAN, "sess1": CLEAN})
    text, result = _write(ctx, affected_paths=affected_paths, ask="council", **({"attach": attach} if attach else {}))
    assert result.status == "ok", text
    assert len(substrate.calls) == 1 and [s.slot_id for s in substrate.calls[0]["slots"]] == ["api1", "sess1"]
    request = substrate.calls[0]["request"]
    api = request.slot_messages["api1"]
    revision = _record(harness, ctx.task_id)["revisions"][-1]
    return revision, _user_text(api[0]["content"]), _user_text(api[-1]["content"]), request.slot_session_tasks["sess1"]


def test_plan_api_and_session_receive_same_map_without_rule_content(harness, monkeypatch):
    from ouroboros.review_native_episode import inspection_registry
    from ouroboros.tools import review_helpers

    source = _design(harness.system)
    _design(harness.workspace, "## Proposed DESIGN\nCandidate rule.\n")
    monkeypatch.setattr(review_helpers, "REPO_ROOT", harness.system)
    revision, api, api_user, session = _plan_ask(
        harness, harness.make_ctx(), affected_paths=[str(harness.system / "ouroboros/loop.py")])
    assert revision["body"] is True
    for text in (api, session):
        assert "Interaction meaning" in text and str(source) in text
        assert "Original interaction rule." not in text and "Candidate rule." not in text
        assert "availability is not evidence of reading or application" in text
        assert "DESIGN.md source unavailable" not in text
    assert "Interaction meaning" not in api_user  # the map rides the stable part, not the plan's content
    # Each delivery class is told how it reaches the original text. (The retired engine also named its
    # reviewers' `need_evidence` request with a `::lines=A-B` locator; the organ's reply object has no such
    # channel: a packet participant needs the excerpts attached.)
    assert _PACKET_SENTENCE in api and _PACKET_SENTENCE not in session
    assert _SESSION_SENTENCE in session and _SESSION_SENTENCE not in api
    registry, reader, _ = inspection_registry(str(harness.workspace), harness.drive)
    result = registry.execute_result("read_file", {"path": str(source), "start_line": 3, "max_lines": 4})
    assert "Original interaction rule." in result.text and "Candidate rule." not in result.text
    assert reader.last_read_view["target"] == str(source)
    result = registry.execute_result("read_file", {"path": "docs/DESIGN.md"})
    assert "Candidate rule." in result.text and "Original interaction rule." not in result.text


def _git_origin(harness, subject):
    """The workspace becomes a git checkout whose origin is the system repo (candidate) or foreign."""
    if (harness.workspace / ".git").exists():
        return
    subprocess.run(["git", "init", "-q"], cwd=harness.workspace, check=True, capture_output=True)
    origin = str(harness.system) if subject == "candidate" else "https://example.invalid/foreign.git"
    subprocess.run(["git", "remote", "add", "origin", origin], cwd=harness.workspace,
                   check=True, capture_output=True)


@pytest.mark.parametrize("subject, affected", [
    ("candidate", "event.py"), ("candidate", "."), ("folderless", ""),
])
def test_plan_navigation_and_pack_follow_body_identity(harness, subject, affected):
    source = _design(harness.system)
    _design(harness.workspace, "# Workspace\n\n## Workspace design\n\nWorkspace rule.\n")
    if subject in {"candidate", "foreign"}:
        _git_origin(harness, subject)
    ctx = harness.make_ctx(active_workspace=subject != "folderless")
    revision, api, _user, session = _plan_ask(harness, ctx, affected_paths=[affected] if affected else [])

    # The retired engine asserted `constitutional is False` with the map present for a candidate copy;
    # in the organ the candidate copy is a body plan (a recognized copy of the body at the active root).
    body = subject == "candidate"
    assert revision["body"] is body
    for text in (api, session):
        assert ("Interaction meaning" in text) is body
        assert (str(source) in text) is body
        assert "Workspace design" not in text  # a workspace's own DESIGN never stands in for the body's
    assert ("## BIBLE.md (constitution" in api) is body
    assert ("## Governance pack (this plan touches Ouroboros's own body)" in session) is body


def _research_context(harness, subject, task_id="task-1"):
    ctx = harness.make_ctx(active_workspace=subject not in {"main", "folderless"}, task_id=task_id)
    if subject == "folderless":
        ctx.project_id = "folderless-research"
    elif subject in {"candidate", "foreign"}:
        _git_origin(harness, subject)
    elif subject in {"body_copy", "foreign_copy"}:
        ctx.workspace_mode = "self_worktree"
        ctx.task_metadata["workspace_copy"] = {
            "execution_root": str(harness.workspace), "source_root": str(harness.system),
            "source_is_system_repo": subject == "body_copy",
        }
    return ctx


@pytest.mark.parametrize(("subject", "flag", "expected", "flag_location"), [
    ("main", True, True, "contract"),
    ("candidate", True, True, "metadata"),
    ("body_copy", True, True, "contract"),
    ("folderless", True, True, "metadata"),
    ("foreign", True, False, "contract"),
    ("unknown", True, False, "metadata"),
    ("foreign_copy", True, False, "contract"),
    ("main", None, False, "metadata"),
    ("main", False, False, "contract"),
    ("candidate", None, False, "metadata"),
    ("folderless", None, False, "contract"),
    ("unknown", None, False, "metadata"),
])
def test_empty_path_plan_is_a_body_plan_only_for_a_declared_body_subject(
    harness, subject, flag, expected, flag_location,
):
    source = _design(harness.system)
    _design(harness.workspace, "# Workspace\n\n## Workspace design\n\nWorkspace rule.\n")
    ctx = _research_context(harness, subject)
    if flag is not None:
        declaration = ctx.task_contract if flag_location == "contract" else ctx.task_metadata
        declaration["context_requires_self_body_docs"] = flag
    # Neither prose nor a DESIGN lookalike supplies body identity or subject.
    ctx.task_contract["objective"] = "Research and brainstorm Ouroboros DESIGN"
    revision, api, _user, session = _plan_ask(harness, ctx, affected_paths=[])

    # The retired engine kept `constitutional is False` here and delivered the map alone; in the organ
    # declared own-body research with a recognized body source IS a body plan, whole pack included.
    assert revision["body"] is expected
    for text in (api, session):
        assert ("Interaction meaning" in text) is expected
        assert (str(source) in text) is expected
        assert "Workspace design" not in text and "Original interaction rule." not in text
        assert ("availability is not evidence of reading or application" in text) is expected
    assert (_PACKET_SENTENCE in api) is expected
    assert (_SESSION_SENTENCE in session) is expected
    assert ("## BIBLE.md (constitution" in api) is expected


@pytest.mark.parametrize("subject", ["main", "body_copy", "folderless"])
def test_declared_body_research_flag_cannot_override_actual_foreign_file_target(harness, subject):
    source = _design(harness.system)
    # The declaration makes this subject's plan a body plan while it names no file ...
    declared = _research_context(harness, subject, task_id="task-declared")
    declared.task_contract["context_requires_self_body_docs"] = True
    revision, api, _user, session = _plan_ask(harness, declared, affected_paths=[])
    assert revision["body"] is True and "Interaction meaning" in api and str(source) in session
    # ... and never turns an actual foreign file target into Ouroboros's body.
    ctx = _research_context(harness, subject)
    ctx.task_contract["context_requires_self_body_docs"] = True
    foreign_target = harness.drive / "foreign-product" / "event.py"
    revision, api, _user, session = _plan_ask(harness, ctx, affected_paths=[str(foreign_target)])

    assert revision["body"] is False
    for text in (api, session):
        assert "Interaction meaning" not in text and str(source) not in text
    assert "## BIBLE.md (constitution" not in api


def test_plan_missing_design_is_explicit_without_new_refusal(harness):
    revision, api, _user, session = _plan_ask(
        harness, harness.make_ctx(), affected_paths=[str(harness.system / "ouroboros/loop.py")])
    assert revision["body"] is True  # the body plan was asked (``_plan_ask`` checks the dispatch), not refused
    for text in (api, session):
        assert "DESIGN.md source unavailable" in text
        assert str(harness.system / "docs/DESIGN.md") in text
        assert "No DESIGN content or navigation delivered" in text and "Interaction meaning" not in text


def test_plan_selected_design_excerpt_is_content_in_evidence(harness):
    source = _design(harness.system)
    locator = f"{source}::lines=3-5"
    _revision, api_system, api_user, session = _plan_ask(
        harness, harness.make_ctx(), affected_paths=[str(harness.system / "ouroboros/loop.py")], attach=[locator])
    assert "Original interaction rule." not in api_system
    assert "Original interaction rule." in api_user and locator in api_user
    assert "Original appearance rule." not in api_user and "Original appearance rule." not in session
    assert session.index("## ATTACHED EVIDENCE") < session.index("Original interaction rule.")
    assert [item["locator"] for item in _record(harness)["asks"][0]["attached"]] == [locator]
