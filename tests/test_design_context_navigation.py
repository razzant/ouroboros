"""DESIGN reaches real author/API projections: its design bible whole, the rest by headings, with source identity."""

import json
from types import SimpleNamespace

import pytest

from ouroboros import context
from ouroboros.context_layout import task_design_navigation, task_design_source
from tests.test_book_context_capture import _plan
from tests.test_body_candidate import _registry, body as _body
from tests.test_doc_context import _make_env_and_memory, foreign_tree

body = _body

# Lines 2-14 are the bible: a fenced heading inside it ends nothing, its seed table rides with its principles.
BIBLE = ('## Design bible: "Airy dopamine"\nBIBLE_RULE_DELIVERED\n```md\n## Fake\n```\n### Principles\n'
         "1. **The room is plain.** PRINCIPLE_DELIVERED\n\n**Seed (light | dark)**\n\n| Role | Light |\n| --- | --- |\n"
         "| Room | SEED_DELIVERED |\n")
DESIGN = "# Design\n" + BIBLE + "## Attention\nBODY_RULE_NOT_DELIVERED\n### Signals\nDETAIL_NOT_DELIVERED\n"


def _in(fragment, text):
    """``fragment`` as a JSON dump of the request carries it."""
    return json.dumps(fragment, ensure_ascii=False)[1:-1] in text


@pytest.mark.parametrize("role", ["main", "api", "declared_nanny"])
def test_design_bible_reaches_all_author_modes_and_keeps_common_prefix(tmp_path, role):
    env, memory = _make_env_and_memory(tmp_path)
    path = env.repo_dir / "docs/DESIGN.md"
    path.write_text(DESIGN)
    task = {"id": "design-author", "type": "task", "text": "Investigate signals"}
    if role != "main":
        task.update(delegation_role="subagent", configured_subagent={"route": {
            "kind": "agent_session" if role == "declared_nanny" else "api_model"}})
    if role.startswith("declared"):
        task["task_contract"] = {"input_sources": "declared", "context_requires_self_body_docs": False}
    core = context._capture_context_core(env, memory, task, None, None)
    plan = _plan(env, core, task, "max")
    captured = {}
    for mode in ("max", "low", "nano"):
        messages = plan.messages_for(mode)
        text = json.dumps(messages, ensure_ascii=False)
        captured[mode] = text
        assert _in(BIBLE.strip(), text)  # the whole section verbatim: principles, seed table and the fenced line
        assert _in('- Design bible: "Airy dopamine" (whole above) — lines 2-14', text)
        assert "- Attention — lines 15-18" in text and "  - Signals — lines 17-18" in text
        assert "- Principles —" not in text and "- Fake" not in text  # inside the bible, inside a fence
        assert _in(str(path), text)
        assert "BODY_RULE_NOT_DELIVERED" not in text and "DETAIL_NOT_DELIVERED" not in text
        assert "Inlined above: its design bible whole (lines 2-14)" in text and "Navigation only" not in text
        assert "from its installed body" not in text  # the body's own book needs no word on whose interface it is
        assert "availability is not evidence of reading or application" in text.lower()
        # A'/D/B remain reusable between task roots; source addresses ride C.
        blocks = plan.projection(mode).system_message()["content"]
        assert all("BIBLE_RULE_DELIVERED" not in block["text"] and "DESIGN.md (navigation map)" not in block["text"]
                   for block in blocks if block.get("cache_control"))
    assert "from its installed body" not in next(part for part in core.dynamic_head_sections
                                                 if part.section_id == "design_bible").operative_view
    path.write_text("## Later source revision\n")
    assert {mode: json.dumps(plan.messages_for(mode), ensure_ascii=False) for mode in captured} == captured


@pytest.mark.parametrize("owner_mode", ["max", "low", "nano"])
def test_the_design_bible_rides_every_owner_mode(tmp_path, monkeypatch, owner_mode):
    """The selected horizon captures the same exact excerpt in every mode, with a map for physical pressure."""
    env, memory = _make_env_and_memory(tmp_path)
    (env.repo_dir / "docs/DESIGN.md").write_text(DESIGN)
    monkeypatch.setattr(context, "get_context_mode", lambda: owner_mode)
    core = context._capture_context_core(env, memory, {"id": "owner-mode", "text": "Plan"}, None, None)
    design = next(part for part in core.dynamic_head_sections if "DESIGN.md (navigation map)" in part)
    assert design.section_id == "design_bible" and design.full_text.startswith(BIBLE.strip())
    assert "BIBLE_RULE_DELIVERED" not in design.operative_view and "- Attention — lines 15-18" in design.operative_view
    assert "Navigation only: DESIGN content is not inlined" in design.operative_view


def test_the_author_view_keeps_its_bytes_while_the_file_does(tmp_path, monkeypatch):
    """No time, counter or service state shapes the view: two captures at different instants carry the same bytes,
    so the block it rides keeps them through a return; a changed file changes them."""
    env, memory = _make_env_and_memory(tmp_path)
    path = env.repo_dir / "docs/DESIGN.md"
    path.write_text(DESIGN)

    def captured(instant):
        monkeypatch.setattr(context, "utc_now_iso", lambda: instant)
        core = context._capture_context_core(env, memory, {"id": "stable", "text": "Plan"}, None, None)
        return next(part for part in core.dynamic_head_sections if part.section_id == "design_bible")

    first, later = captured("2026-11-21T10:00:00+00:00"), captured("2026-11-24T18:30:00+00:00")
    assert (first.full_text, first.operative_view) == (later.full_text, later.operative_view)
    assert "2026-11-2" not in first.full_text
    path.write_text(DESIGN.replace("SEED_DELIVERED", "SEED_REVISED"))
    assert "SEED_REVISED" in captured("2026-11-24T18:30:00+00:00").full_text


def _ctx(env, **kwargs):
    return SimpleNamespace(repo_dir=env.repo_dir, system_repo_dir=env.repo_dir,
                           drive_root=env.drive_root, **kwargs)


def test_selected_copy_bytes_win_and_default_false_cannot_hide_body(tmp_path):
    env, memory = _make_env_and_memory(tmp_path)
    (env.repo_dir / "docs/DESIGN.md").write_text("## Installed heading\n")
    copy = tmp_path / "selected-copy"
    (copy / "docs").mkdir(parents=True)
    (copy / "docs/DESIGN.md").write_text(DESIGN)
    task = {"id": "body-copy", "text": "Plan", "workspace_root": str(copy),
            "workspace_mode": "self_worktree", "task_contract": {"context_requires_self_body_docs": False},
            "metadata": {"workspace_copy": {"execution_root": str(copy), "source_is_system_repo": True}}}
    core = context._capture_context_core(env, memory, task, None, None)
    text = json.dumps(_plan(env, core, task, "max").messages_for("max"), ensure_ascii=False)
    assert json.dumps(str(copy / "docs/DESIGN.md"), ensure_ascii=False)[1:-1] in text and "- Attention" in text
    assert _in(BIBLE.strip(), text)  # the copy's own bible: the contract's default false hides nothing
    assert "Installed heading" not in text


# Who carries the installed bible (item 97, council C13 R2): the tree's identity, and a TRUE declaration beside it.
# The contract writes false for silence (``task_contract``), so false or absent narrows nothing; proven-foreign work
# without a true declaration carries none, and a foreign tree's own DESIGN never stands in for the body's.
@pytest.mark.parametrize(("kind", "declared", "delivered"), [
    ("proven_foreign", None, False), ("proven_foreign", False, False), ("proven_foreign", True, True),
    ("foreign_copy", True, True), ("foreign_room", None, False), ("foreign_room", True, True),
    ("unknown_folder", None, True), ("folderless", None, True), ("folderless", False, True),
])
def test_the_bible_follows_body_identity_and_a_true_declaration(tmp_path, kind, declared, delivered):
    env, _memory = _make_env_and_memory(tmp_path)
    (env.repo_dir / "docs/DESIGN.md").write_text(DESIGN)
    foreign = tmp_path / "another-project"
    if kind == "unknown_folder":  # no Git, no remote: its body is unknown, not proven foreign
        foreign.mkdir()
    elif kind != "folderless":
        foreign_tree(foreign)
    if kind != "folderless":
        (foreign / "docs").mkdir()
        (foreign / "docs/DESIGN.md").write_text("## Foreign lookalike\nFOREIGN_RULE\n")
    ctx = {"proven_foreign": dict(workspace_root=foreign, workspace_mode="external", task_metadata={}),
           "unknown_folder": dict(workspace_root=foreign, workspace_mode="external", task_metadata={}),
           "foreign_copy": dict(workspace_root=foreign, workspace_mode="self_worktree", task_metadata={
               "workspace_copy": {"execution_root": str(foreign), "source_is_system_repo": False}}),
           "foreign_room": dict(is_direct_chat=True, project_id="room", task_metadata={"_project_room_dir": str(foreign)}),
           "folderless": dict(project_id="project-without-folder", task_metadata={})}[kind]
    task = {} if declared is None else {"task_contract": {"context_requires_self_body_docs": declared}}
    view = task_design_navigation(_ctx(env, **ctx), task)
    assert ("BIBLE_RULE_DELIVERED" in view) is delivered
    # A tree not established as the body reads the installed book under a first line saying whose rules these are,
    # in the whole form and in the map a short room leaves (RVD1, Fable N2).
    assert view.startswith("These are the interface rules of Ouroboros itself, from its installed body") is delivered
    assert getattr(view, "operative_view", "").startswith(
        "These are the interface rules of Ouroboros itself, from its installed body") is delivered
    assert (str(env.repo_dir / "docs/DESIGN.md") in view) is delivered
    assert "FOREIGN_RULE" not in view and "Foreign lookalike" not in view


@pytest.mark.parametrize("kind", ["folderless", "proven_foreign"])
def test_a_folderless_room_request_carries_the_bible_a_foreign_one_does_not(tmp_path, kind):
    env, memory = _make_env_and_memory(tmp_path)
    (env.repo_dir / "docs/DESIGN.md").write_text(DESIGN)
    task = {"id": "room-work", "text": "Plan", "task_contract": {"context_requires_self_body_docs": False}}
    if kind == "folderless":
        task["project_id"] = "project-without-folder"
    else:
        task.update(workspace_root=str(foreign_tree(tmp_path / "another-project")), workspace_mode="external")
    core = context._capture_context_core(env, memory, task, None, None)
    text = json.dumps(_plan(env, core, task, "max").messages_for("max"), ensure_ascii=False)
    assert _in(BIBLE.strip(), text) is (kind == "folderless")
    assert _in(str(env.repo_dir / "docs/DESIGN.md"), text) is (kind == "folderless")


def test_missing_design_is_a_visible_source_gap_in_actual_author_input(tmp_path):
    env, memory = _make_env_and_memory(tmp_path)
    task = {"id": "missing-design", "text": "Plan"}
    core = context._capture_context_core(env, memory, task, None, None)
    text = json.dumps(_plan(env, core, task, "max").messages_for("max"))
    assert "DESIGN.md source unavailable" in text
    assert json.dumps(str(env.repo_dir / "docs/DESIGN.md"))[1:-1] in text
    assert "No DESIGN content or navigation delivered" in text


def test_explicit_folderless_self_work_points_to_system_source(tmp_path):
    env, _memory = _make_env_and_memory(tmp_path)
    (env.repo_dir / "docs/DESIGN.md").write_text(DESIGN)
    ctx = _ctx(env, project_id="folderless", task_metadata={})
    task = {"context_requires_self_body_docs": True}
    assert task_design_source(ctx, task) == env.repo_dir
    view = task_design_navigation(ctx, task)
    assert "- Attention" in view
    # Declared folderless work reads the installed book; it says whose rules these are (RVD1, Astra 4).
    assert view.startswith("These are the interface rules of Ouroboros itself, from its installed body")
    assert view.operative_view.startswith("These are the interface rules of Ouroboros itself")


def test_explicit_flag_brings_the_installed_bible_without_making_a_room_folder_the_body(tmp_path):
    """A mockup of my own screen from a room with someone else's folder (item 97, council C13 R2): the declaration
    brings the installed bible, never the folder's lookalike, and the folder stays foreign for plans and sessions."""
    env, _memory = _make_env_and_memory(tmp_path)
    (env.repo_dir / "docs/DESIGN.md").write_text(DESIGN)
    foreign = foreign_tree(tmp_path / "room-folder")
    (foreign / "docs").mkdir(parents=True)
    (foreign / "docs/DESIGN.md").write_text("## Foreign rules\n")
    ctx = _ctx(env, is_direct_chat=True, project_id="room",
               task_metadata={"_project_room_dir": str(foreign)})
    declared = {"context_requires_self_body_docs": True}
    view = task_design_navigation(ctx, declared)
    assert view.startswith("These are the interface rules of Ouroboros itself, from its installed body")
    assert BIBLE.strip() in view and str(env.repo_dir / "docs/DESIGN.md") in view and "Foreign rules" not in view
    assert task_design_source(ctx, declared) is None  # body identity, which plans and sessions read, is unchanged


def test_folderless_prepared_candidate_survives_restore_in_author_and_session_delivery(body):
    from pathlib import Path

    from ouroboros import body_candidate
    from ouroboros.agent import Env
    from ouroboros.memory import Memory
    from ouroboros.subagent_work_order import session_design_navigation
    from ouroboros.tools.tool_resolution import active_repo_dir_for
    from tests.body_candidate_support import make_ctx

    serving, data = body
    ctx = make_ctx(serving, data, "folderless-design", project_id="folderless")
    scratch = active_repo_dir_for(ctx)
    registry = _registry(serving, data, ctx)
    assert "is new" in registry.execute("prepare_self_change", {})
    candidate = Path(body_candidate.descriptor(ctx)["path"])
    source = candidate / "docs/DESIGN.md"
    source.parent.mkdir()
    source.write_text(DESIGN)
    env, memory = Env(repo_dir=serving, drive_root=data), Memory(drive_root=data, repo_dir=serving)
    task = {"id": ctx.task_id, "text": "Plan", "context_requires_self_body_docs": False}

    def delivered(bound):
        assert active_repo_dir_for(bound) == scratch  # preparing a body never reclassifies scratch
        assert task_design_source(bound, task) == candidate
        core = context._capture_context_core(env, memory, task, None, bound)
        for mode in ("max", "low", "nano"):
            text = json.dumps(_plan(env, core, task, mode).messages_for(mode))
            assert json.dumps(str(source))[1:-1] in text and "- Attention" in text
            assert "BODY_RULE_NOT_DELIVERED" not in text
        text = session_design_navigation(bound, target_root=str(scratch), source_root=str(scratch))
        assert str(source) in text and "- Attention" in text
        foreign = data.parent / "foreign"
        foreign.mkdir(exist_ok=True)
        assert session_design_navigation(bound, target_root=str(foreign), source_root=str(foreign)) == ""
        for foreign_kind in ("workspace", "room"):
            other = make_ctx(serving, data, ctx.task_id, project_id="folderless", is_direct_chat=True)
            body_candidate.restore(other)
            if foreign_kind == "workspace":
                other.workspace_root, other.workspace_mode = foreign, "external"
            else:
                other.task_metadata["_project_room_dir"] = str(foreign)
            assert active_repo_dir_for(other) == foreign
            assert task_design_source(other, task) is None

    delivered(ctx)
    restored = make_ctx(serving, data, ctx.task_id, project_id="folderless")
    assert body_candidate.restore(restored)["path"] == str(candidate)
    delivered(restored)
    assert "is resumed" in _registry(serving, data, restored).execute(
        "prepare_self_change", {"resume": ctx.task_id})
    delivered(restored)


@pytest.mark.parametrize("own_body", [True, False], ids=["recognized-fork", "foreign-repository"])
def test_isolated_copy_uses_body_identity_not_only_shared_git_directory(tmp_path, own_body):
    from ouroboros.subagent_work_order import session_design_navigation
    from ouroboros.workspace_copies import source_is_system_repo
    from tests.test_review_body_fact import FORK_URL, FOREIGN_URL, _git, _repo

    system = _repo(tmp_path / "system", remote=FORK_URL)
    source = _repo(tmp_path / "source", remote=FORK_URL if own_body else FOREIGN_URL)
    execution = tmp_path / "execution"
    _git(tmp_path, "clone", "-q", str(source), str(execution))
    (execution / "docs").mkdir()
    (execution / "docs/DESIGN.md").write_text(DESIGN, encoding="utf-8")
    # Admission's flag asks only whether the repositories share a Git directory.
    shared_git = source_is_system_repo(source, system)
    assert shared_git is False
    binding = {"source_root": str(source), "execution_root": str(execution),
               "source_is_system_repo": shared_git}
    ctx = SimpleNamespace(repo_dir=system, system_repo_dir=system,
                          workspace_root=execution, workspace_mode="self_worktree",
                          task_metadata={"workspace_copy": binding}, drive_root=tmp_path / "data")
    expected = execution if own_body else None
    assert task_design_source(ctx, {}) == expected
    nav = session_design_navigation(ctx, target_root=str(execution), source_root=str(execution))
    if own_body:
        assert str(execution / "docs/DESIGN.md") in nav and "- Attention" in nav
    else:
        assert nav == ""


BIG_BIBLE = BIBLE + "\n" + "A principle of the room and its objects, told for both themes. " * 240 + "\n"


def _fitting(tmp_path, monkeypatch, *, section_root=None, required_words=0, others=True, authored=False):
    """Exercise the context assembly/first request with pinned route facts and no provider send."""
    from ouroboros import context_fit as fit, memory_view
    from ouroboros.reference_books import body_map, compose_book, load_reference_book
    from ouroboros.section_floor import HostSection, capture_sections
    from tests import _memory_view_synthetic as syn

    monkeypatch.setattr(fit, "_route_calibration_ratio", lambda *_a, **_kw: 1.0)
    monkeypatch.setattr(fit, "main_output_reserve_tokens", lambda **_kw: 8_192)
    monkeypatch.setattr("ouroboros.capability_evidence.resolve_main_token_density",
                        lambda *_a, **_kw: (1.0, "cold_estimate"))
    repo = tmp_path / "repo"

    def chapter(title, parts):
        return f"# {title}\n\nHow it holds.\n\n" + "".join(
            f"## Part {i}\n\n" + "A sentence of the mechanism and its why. " * 120 + "\n\n" for i in range(parts))

    for path, text in {"docs/ARCHITECTURE.md": "# Body\n\nThe whole body map.\n\n## Chapters\n\n- [Runtime](architecture/runtime.md)\n",
                       "docs/architecture/runtime.md": chapter("Runtime", 30),
                       "docs/DEVELOPMENT.md": "# Engineering\n\nHow I change.\n\n## Chapters\n\n- [Change](development/change.md)\n",
                       "docs/development/change.md": chapter("Change", 20 + required_words),
                       "docs/DESIGN.md": "# Design\n" + BIG_BIBLE + "## Attention\nBODY_RULE_NOT_DELIVERED\n"}.items():
        (repo / path).parent.mkdir(parents=True, exist_ok=True)
        (repo / path).write_text(text, encoding="utf-8")
    books = tuple(load_reference_book(repo, book_id, lambda p: (repo / p).read_bytes())
                  for book_id in ("architecture", "development"))
    design = task_design_navigation(SimpleNamespace(repo_dir=repo, system_repo_dir=repo, drive_root=tmp_path / "data",
                                                    task_metadata={}), {})
    assert design.section_id == "design_bible"
    index = HostSection("knowledge_index", "## Knowledge index\n\n" + "- a durable lesson told whole. " * 900,
                        "## Knowledge index\n\n- every note by address.")
    world = HostSection("world", "## My World\n\n" + "a fact of the machine. " * 900, "## My World\n\nIts headings.")
    services = [HostSection(name, f"## {name}\n\n" + "an observed fact. " * 400, f"## {name}\n\nIts operative state.")
                for name in ("runtime", "review_pool", "installed_skills")] if others else []
    owner = syn.spoken(2, "human", 4_000)
    owner["line"] += "\nOWNER_UNTOLD_WHOLE"
    story = syn.pages(2, 12_000)
    if authored:
        story = [{**page, "lead_rule": 2, "text": f"{page['id']} MY_OWN_RECAP.\n\n{page['text']} SEALED_PAGE_TAIL"}
                 for page in story]
    snapshot = syn.snapshot(story=story, room_facts=syn.room("1", lane1=[syn.spoken(1, "ouroboros", 8_000), owner]),
        conversations=[] if authored else [syn.conversation(i, words=10, word_chars=600) for i in range(30)])
    head, dynamic = [index, design], [*services, world]
    # The resident Map its books render, with its parts and addresses (a lowered frame stands it at those parts). The
    # handbook rides only as the task's own requirement (``required_words``).
    view, parts = body_map(*books)
    core = fit.ContextCore(
        base_prompt="SYSTEM WHOLE", bible_md="BIBLE WHOLE", body_map=view.text,
        body_map_parts=parts, development_md=compose_book(books[1]) if required_words else "",
        development_required=bool(required_words), reference_books=books, semi_stable_text="IDENTITY WHOLE",
        dynamic_text="\n\n".join(dynamic), dynamic_head_text="\n\n".join(head),
        user_content_json=json.dumps("WORDS THAT STARTED THE WORK"),
        dynamic_sections=capture_sections(dynamic, prefix="dynamic"),
        dynamic_head_sections=capture_sections(head, prefix="head"),
        section_root=str(tmp_path) if section_root is None else section_root, section_task_id="task-1",
        memory_view_json=memory_view.snapshot_json(snapshot))

    def fitted(window, preferred, known=True, *, core=core):
        evidence = SimpleNamespace(route_fp="route", status="confirmed" if known else "unprobeable", stale=False,
                                   window_tokens=window, source="synthetic")
        plan = fit.build_context_fit_plan(SimpleNamespace(drive_root=tmp_path), core, {"id": "task-1"},
            preferred_mode=preferred, route_resolver=lambda *_a, **_kw: ({"provider": "test", "model": "synthetic"},
                                                                         evidence))
        view = plan.projection(plan.initial_mode)
        return plan, view, *_shown(view)

    fitted.core = core  # a test asking the route question itself builds from the same capture
    return design, fitted


def _shown(view):
    """A projection's host rows, memory steps and system text."""
    rows = [row["section_id"] for row in view.memory_facts.get("host_sections_addressed", ())]
    text = "\n\n".join(block["text"] for block in view.system_message()["content"])
    return rows, (view.memory_facts.get("floor") or {}).get("steps") or {}, text


@pytest.mark.parametrize("preferred", ["max", "low", "nano"])
def test_a_short_room_keeps_the_whole_bible_beside_authored_tellings(tmp_path, monkeypatch, preferred):
    import math

    from ouroboros import context_budget

    design, fitted = _fitting(tmp_path, monkeypatch, section_root="", authored=True)
    plan, view, _, _, roomy = fitted(2_000_000, preferred)
    reserve = context_budget.context_mode_limits(preferred, preferred, plan.output_reserve_tokens)[1]
    levels = view.memory_facts["spec"]["work_room_levels"]
    fraction = 1 - levels / context_budget.RECLAIM_LOW_WATER_DIVISOR
    # The pressure comes from the same complete first-request shape in each mode, minus part of an authored page.
    window = math.ceil((view.estimated_tokens + reserve - 2_000) / fraction)
    plan, view, rows, steps, tight = fitted(window, preferred)
    assert plan.initial_mode == preferred
    assert not rows and steps.get("F5l") and design.full_text in tight
    assert "SEALED_PAGE_TAIL" in roomy and tight.count("SEALED_PAGE_TAIL") < roomy.count("SEALED_PAGE_TAIL")
    assert "MY_OWN_RECAP." in tight and "memory_read(node_id='page-" in tight
    assert "OWNER_UNTOLD_WHOLE" in tight
    assert all(s in tight for s in ("SYSTEM WHOLE", "BIBLE WHOLE", "IDENTITY WHOLE"))
    assert "WORDS THAT STARTED THE WORK" in json.dumps(plan.messages_for(preferred))


def test_a_nano_budget_takes_the_bible_to_its_map_before_it_cuts_what_i_have_not_told(tmp_path, monkeypatch):
    """Nano's own number governs all my memory, below my tellings too: the end of a reply of mine (F2), a page cut to its
    opening (F5). In a wide window the whole bible therefore stays at a first start where my retold forms alone hold it
    in the start's own frame; where its weight is what makes the walk take such a form its map is the budget's
    (``start_room``); where the walk takes the same forms beside the map, the bible is not their cause and stays."""
    from ouroboros import context_budget, memory_floor as floor

    design, fitted = _fitting(tmp_path, monkeypatch, authored=True, others=False)

    def start(target):
        monkeypatch.setattr(context_budget, "OWNER_NANO_TARGET_TOKENS", target)
        plan, view, rows, steps, text = fitted(2_000_000, "nano")
        return design.full_text in text, set(steps), view.memory_facts, text

    low, high = 16_000, 60_000  # the smallest number at which a Nano start keeps the whole bible on my tellings alone
    while high - low > 1:
        middle = (low + high) // 2
        whole, steps, *_ = start(middle)
        low, high = (low, middle) if whole and steps == {"F5l"} else (middle, high)
    whole, steps, facts, _ = start(high)
    assert whole and steps == {"F5l"} and facts["floor"]["by_budget"] == sum(facts["floor"]["steps"].values())
    whole, steps, facts, text = start(high - 1)  # one number below, only the budget's cuts of tellings, the map the budget's
    assert not whole and steps == {"F5l"} and facts["floor"]["by_budget"] == sum(facts["floor"]["steps"].values())
    assert {row["section_id"]: row.get("start_room") for row in facts["host_sections_addressed"]}["design_bible"] == "nano"
    assert "the owner's Nano mode budget, not for the physical window" in text and design.operative_view in text
    with monkeypatch.context() as patch:  # what the map spares: the same number, the bible kept whole by force
        patch.setattr(floor, "untold_forms", lambda *_a: frozenset())
        forced, steps, *_ = start(high - 1)
    assert forced and "F2" in steps
    whole, steps, *_ = start(low := 16_000)  # a number the walk takes F2 and F5 beside the map too: not the bible's doing
    with monkeypatch.context() as patch:
        patch.setattr(floor, "optional_excerpt_fits", lambda *_a, **_kw: False)
        mapped = start(low)
    assert whole and {"F2", "F5"} <= steps and not mapped[0] and mapped[1] == steps


@pytest.mark.parametrize("window", [120_000, 200_000])
@pytest.mark.parametrize("shape, needs_map", [((4, 3, 60_000), False), ((40, 10, 1_500), True)],
                         ids=["same-cuts", "extra-untold-cuts"])
def test_nano_prices_design_after_the_cuts_the_map_needs_anyway(tmp_path, monkeypatch, window, shape, needs_map):
    """In both rooms the ordinary tellings overstate the physical need. Only one needs the map, for the target."""
    from dataclasses import replace

    from ouroboros import context_budget, context_fit as fit, memory_floor as floor, memory_view as mv
    from ouroboros.section_floor import HostSection, predicted_section, relocated_text
    from tests import _memory_view_synthetic as syn

    monkeypatch.setattr(context_budget, "OWNER_NANO_TARGET_TOKENS", 85_000)
    design, fitted = _fitting(tmp_path, monkeypatch, authored=True, others=False)
    rooms, words, chars = shape
    conversations = [syn.conversation(i, words=words, word_chars=chars) for i in range(rooms)]
    for conversation in conversations:
        for n, word in enumerate(conversation["rooms"][0]["words"]):
            word.update(rank=1, kind="ouroboros" if n < words // 2 else "human")  # considered, not retold
    snapshot = replace(mv.snapshot_from_json(fitted.core.memory_view_json), conversations=tuple(conversations),
                       owner_words="OPERATIVE_OWNER_WORDS")
    core = replace(fitted.core, memory_view_json=mv.snapshot_json(snapshot))
    map_text = relocated_text(design, predicted_section(design)["reader"], start_room="nano")

    def start(part):
        head = tuple(part if section.section_id == design.section_id else section for section in core.dynamic_head_sections)
        plan, view, _, _, text = fitted(window, "nano", core=replace(core, dynamic_head_sections=head,
                                                                     dynamic_head_text="\n\n".join(head)))
        _, reserve = context_budget.context_mode_limits("nano", "nano", plan.output_reserve_tokens)
        bound = floor.floor_allowances(window_tokens=window, output_reserve_tokens=reserve, non_memory_tokens=0,
                                       work_room_levels=snapshot.spec.work_room_levels)["window"]
        assert plan.initial_mode == "nano" and view.estimated_tokens <= bound
        assert "OWNER_UNTOLD_WHOLE" in text and "OPERATIVE_OWNER_WORDS" in text
        messages = plan.messages_for("nano")
        assert "WORDS THAT STARTED THE WORK" in json.dumps(messages)
        if part.full_text == map_text:
            # Same legal memory and same books, assignment and reserve: only replace the map text with the excerpt.
            block, = [block for block in messages[0]["content"] if map_text in block["text"]]
            block["text"] = block["text"].replace(map_text, design.full_text, 1)
            assert fit.estimate_context_prompt_tokens(messages, [], provider="test") <= bound
        return view.memory_facts, text, floor.untold_forms(snapshot, floor.kept_level(snapshot, view.memory_facts))

    # The two controls send the actual content of each form on the same frame, with the same receipt cost as the budget map.
    _, _, mapped_cuts = start(HostSection(design.section_id, map_text))
    _, _, whole_cuts = start(HostSection(design.section_id, design.full_text))
    assert mapped_cuts and bool(whole_cuts - mapped_cuts) is needs_map
    facts, text, chosen_cuts = start(design)
    assert chosen_cuts == (mapped_cuts if needs_map else whole_cuts)
    design_steps = [r.get("start_room") for r in facts.get("host_sections_addressed", ()) if r["section_id"] == design.section_id]
    assert design_steps == (["nano"] if needs_map else [])
    assert (design.full_text in text) is not needs_map
    assert (map_text in text) is needs_map


def test_the_guard_asks_again_where_only_the_window_took_the_bible_to_its_map_under_a_nano_budget(tmp_path, monkeypatch):
    """Nano's number 85,000 and the handbook directly required: the smallest window that starts, above that number,
    holds it only beside the bible's map, taken for the window. The early answer that sized it is asked again at the
    usual point and the larger window walked: one repeat, the handbook and every word whole. The same window walks nothing
    again, and a map the budget alone took asks nothing."""
    from ouroboros import context_budget, context_fit as fit

    design, fitted = _fitting(tmp_path, monkeypatch, authored=True, others=False, required_words=30)
    monkeypatch.setattr(fit, "_context_route", lambda _task: ({"provider": "claudexor", "model": "fixture"}, {}))
    walked, walk = [], fit._view_projections
    monkeypatch.setattr(fit, "_view_projections", lambda *a, **kw: walked.append(kw["window_tokens"]) or walk(*a, **kw))

    def build(core, *windows):
        asked, walked[:] = [], []

        def question(_task, **_kw):  # the early window first, the usual point's next
            asked.append(windows[min(len(asked), len(windows) - 1)])
            return ({"provider": "claudexor", "model": "fixture", "options": {"account_fingerprint": "fixture-fp"}},
                    SimpleNamespace(route_fp="route", status="confirmed", stale=False, window_tokens=asked[-1],
                                    source="synthetic"))
        plan = fit.build_context_fit_plan(SimpleNamespace(drive_root=tmp_path), core, {"id": "late", "chat_id": 1},
                                          preferred_mode="nano", route_resolver=fit.RouteQuestion({"id": "late"}, question))
        return plan, asked, plan.projection(plan.initial_mode)

    def starts(window):
        try:
            return fitted(window, "nano")
        except fit.MinimumViewUnfit:
            return None

    low, high = context_budget.OWNER_NANO_TARGET_TOKENS, 400_000
    while high - low > 1:
        middle = (low + high) // 2
        low, high = (low, middle) if starts(middle) else (middle, high)
    early, late = high, 2 * high
    _, _, rows, _, text = starts(early)
    assert rows == ["design_bible"] and design.operative_view in text  # the window's map, above the budget's number
    plan, asked, view = build(fitted.core, early, late)
    rows, steps, text = _shown(view)
    assert asked == [early, late] and walked == [early, late] and plan.window_tokens == late
    assert not rows and design.full_text in text and text.count("A sentence of the mechanism and its why.") == 50 * 120
    assert "OWNER_UNTOLD_WHOLE" in text and "WORDS THAT STARTED THE WORK" in json.dumps(plan.messages_for("nano"))
    plan, asked, view = build(fitted.core, early, early)  # the same window: asked once more, nothing walked again
    assert asked == [early, early] and walked == [early] and _shown(view)[0] == ["design_bible"]
    # The budget alone, the window wide: it takes host forms and the bible's map for itself (``start_room``), asks nothing.
    monkeypatch.setattr(context_budget, "OWNER_NANO_TARGET_TOKENS", 30_000)
    mine, other = _fitting(tmp_path / "budget", monkeypatch, authored=True, others=True, required_words=30)
    plan, asked, view = build(other.core, 300_000)
    held = view.memory_facts["host_sections_addressed"]
    assert asked == [300_000] and walked == [300_000] and mine.operative_view in _shown(view)[2]
    assert {row["section_id"] for row in held} >= {"runtime", "design_bible"} and {row.get("start_room") for row in held} == {"nano"}


def test_without_a_measured_window_only_a_refusal_past_tellings_takes_the_bible(tmp_path, monkeypatch):
    """A refusal need not give a number for the window; the owner's target alone is not a refusal."""
    from ouroboros import context_budget

    design, fitted = _fitting(tmp_path, monkeypatch, authored=True)
    plan, *_ = fitted(0, "max", known=False)
    assert not _shown(plan.projection("max"))[0]  # nothing asks: the bible stays whole
    asked = {}
    for refused in (1_000, 7_500):
        refit = plan.reproject_for_route(window_tokens=0, known_window=False, ratio=1.0, output_reserve=8_192,
                                         tool_schemas=[], start_mode="max", refused_tokens=refused)
        asked[refused] = _shown(refit.projection(refit.initial_mode))
    rows, steps, text = asked[1_000]
    assert not rows and steps.get("F5l") and design.full_text in text
    rows, steps, text = asked[7_500]
    assert rows == ["design_bible"] and steps.get("F5l") and "OWNER_UNTOLD_WHOLE" in text
    assert "omitted for the physical window" in text and design.operative_view in text
    monkeypatch.setattr(context_budget, "OWNER_NANO_TARGET_TOKENS", 19_000)
    assert design.full_text in fitted(0, "nano", known=False)[-1]


def test_running_max_keeps_the_body_map_at_both_sides_of_the_design_boundary(tmp_path, monkeypatch, record_property):
    """At a measured wall, compare both Max frames at the same room goal before the Map steps to its parts."""
    import copy
    from dataclasses import replace
    from hashlib import sha256

    from ouroboros import context_budget, context_fit as fit, memory_floor as floor, memory_view as mv
    from ouroboros.artifacts import read_actor_source_bytes
    from ouroboros.reference_books import compose_book
    from ouroboros.section_floor import predicted_section, relocated_text
    from tests._context_fit_fixtures import _rebind
    from tests._memory_view_synthetic import pages

    design, fitted = _fitting(tmp_path, monkeypatch, others=False, authored=True)
    memory = mv.snapshot_from_json(fitted.core.memory_view_json)
    story = tuple({**page, "lead_rule": 2, "text": f"{page['id']} MY_OWN_RECAP.\n\n{page['text']} SEALED_PAGE_TAIL"}
                  for page in pages(6, 12_000))
    memory = replace(memory, story=story, owner_words="Governing owner words: keep the lamp blue.")
    # Only the authored pages or DESIGN can pay before the Map steps. Stimulus: a Map as large as the whole Architecture
    # composition (whole books once weighed on Max so) whose parts and addresses its books render.
    core = replace(fitted.core, dynamic_head_sections=(design,), dynamic_sections=(),
                   body_map=compose_book(fitted.core.reference_books[0]), memory_view_json=mv.snapshot_json(memory))
    initial, *_ = fitted(2_000_000, "max", core=core)
    sent = initial.messages_for("max") + [{"role": "assistant", "content": "Exact earlier reply."},
                                           {"role": "user", "content": "Exact latest words 🧪."}]
    reserve = initial.output_reserve_tokens
    level = floor.telling_level(memory)
    gov = fit._governance_blocks(None, core, mode="max")
    bound = fit.estimate_context_prompt_tokens(sent, []) + reserve
    divisor = context_budget.RECLAIM_LOW_WATER_DIVISOR

    def threshold(addressed):
        window = bound
        while True:  # the number printed in the floor note is part of the charge
            note = floor.floor_note(level, window_tokens=window, mode="max", snapshot=memory)
            blocks = fit._system_blocks(core, gov, mv.render_story(memory, level),
                                        mv.render_room(memory, level, floor_note=note), "max", addressed, story_whole=False)
            charge = fit.estimate_context_prompt_tokens([{"role": "system", "content": blocks}, *sent[1:]], []) + reserve
            # Smallest W where W - ceil(W / divisor) still covers the input and the full reply.
            needed = (charge * divisor + divisor - 2) // (divisor - 1)
            if needed == window:
                return needed
            assert needed < window
            window = needed

    whole = threshold({})
    pointer = threshold({"design_bible": relocated_text(design, predicted_section(design)["reader"])})
    assert pointer < whole < bound
    assert sum(block["text"].count("SEALED_PAGE_TAIL") for block in sent[0]["content"]) == len(story)
    for case, window in (("wide", bound + 100), ("whole", whole), ("pointer", whole - 1),
                         ("pointer_edge", pointer), ("low", pointer - 1)):
        messages = copy.deepcopy(sent)
        rebound, mode, event = _rebind(monkeypatch, tmp_path / case, initial, messages, window=window,
                                        holds_reply=True, start_mode="max")
        facts = rebound.projection(mode).memory_facts
        text = "\n".join(block["text"] for block in messages[0]["content"])
        assert rebound.preferred_mode == "max" and mode == event["effective_mode"]
        assert messages[1:] == sent[1:]
        assert "OWNER_UNTOLD_WHOLE" in text and memory.owner_words in text
        assert "SYSTEM WHOLE" in text and "BIBLE WHOLE" in text and "IDENTITY WHOLE" in text
        assert event["view_kept"] is (case == "wide")
        whole_design = design.full_text in text
        addressed = [row for row in facts.get("host_sections_addressed", ()) if row["section_id"] == "design_bible"]
        if not whole_design:
            row, = addressed
            assert str(tmp_path / "repo/docs/DESIGN.md") in text and design.operative_view in text and row["reader"] in text
            assert row["source_ref"]["sha256"] == sha256(design.full_text.encode()).hexdigest()
            assert read_actor_source_bytes(core.section_root, core.section_task_id, row["source_ref"]) == design.full_text.encode()
        else:
            assert not addressed
        free = window - reserve - event["projected_prompt_tokens"]
        eighth = (window + divisor - 1) // divisor
        whole_books = core.body_map in text  # the Map whole; else it stands at its parts and addresses
        if not whole_books:
            assert core.body_map_parts in text
        recaps, tails = text.count("MY_OWN_RECAP."), text.count("SEALED_PAGE_TAIL")
        record_property(case, json.dumps(dict(window=window, reserve=reserve, free=free, required=0 if case == "wide" else eighth,
            input=event["projected_prompt_tokens"], original_input=bound - reserve, whole_threshold=whole, pointer_threshold=pointer,
            preferred=rebound.preferred_mode, effective=mode, books="whole" if whole_books else "map_parts", design=whole_design,
            recaps=recaps, whole_page_tails=tails, owner_and_tail_exact=True,
            host_steps=[r["section_id"] for r in facts.get("host_sections_addressed", ())],
            request_sha256=sha256(json.dumps(messages, ensure_ascii=False, sort_keys=True).encode()).hexdigest())))
        if case == "wide":
            assert 0 <= free < eighth and messages == sent
        else:
            assert free >= eighth, (case, free, eighth)
        assert not (set(facts["floor"]["steps"]) - {"F5l"})
        if case == "low":
            assert mode == "low" and not whole_books and whole_design
            assert facts["floor"]["mode_switch"] == {"from": "max", "to": "low"} and not facts["floor"]["by_budget"]
        else:
            assert mode == "max" and whole_books and whole_design is (case in {"wide", "whole"})
            if case == "whole":
                assert recaps == len(story) and tails == 0


def test_an_actor_without_a_reader_gets_the_whole_bible_or_its_name_unavailable(tmp_path, monkeypatch):
    """Only an accessible source may be named when the excerpt yields."""
    design, fitted = _fitting(tmp_path, monkeypatch, section_root="")
    plan, *_ = fitted(2_000_000, "max")
    assert design.full_text in _shown(plan.projection("max"))[2]  # with room, the whole bible
    total = plan.projection("max").estimated_tokens
    window, seen = int(total * 1.6), None
    while seen is None or not seen[0]:
        seen, window = fitted(window, "max")[2:], int(window * 0.94)
    rows, _steps, text = seen
    assert rows == ["design_bible"]  # the other host forms stay whole for this actor
    assert '## Design bible: "Airy dopamine"\n\n[Omitted for ' in text and "unavailable to this actor." in text
    assert BIG_BIBLE.strip() not in text and "DESIGN.md (navigation map)" not in text
    assert str(tmp_path / "repo/docs/DESIGN.md") not in text and "artifact_store" not in text
    assert "Use this exact source path" not in text


def test_a_required_handbook_start_takes_the_bible_map_before_it_refuses(tmp_path, monkeypatch):
    """A task that directly requires the whole handbook near its route's limit (RV1, Astra 1): where the handbook fits
    only beside the bible's map, the start takes the map and keeps the whole handbook; below that an integrating Low
    start yields the Body Map at its parts (owner 23.11.2026, 1; one Map since 25.11.2026); a window that cannot hold it even
    so is refused by name, as before, and with room the bible stays whole. A Map whose parts are no shorter than its
    text never steps. (Stimulus: a Map as large as the whole Architecture composition, so its step saves.)"""
    from dataclasses import replace

    from ouroboros import context_fit as fit, memory_floor as floor
    from ouroboros.reference_books import compose_book

    design, fitted = _fitting(tmp_path, monkeypatch, required_words=30, others=False)
    small = fitted.core
    large = replace(small, body_map=compose_book(small.reference_books[0]))
    refusals = {}

    def starts(window, core=large):
        try:
            return fitted(window, "low", core=core)
        except fit.MinimumViewUnfit as unfit:
            refusals[window] = str(unfit)
            return None

    def smallest(core):  # the smallest start: refused below, started at or above
        low, high = 1_000, 2_000_000
        while high - low > 1:
            middle = (low + high) // 2
            low, high = (low, middle) if starts(middle, core) else (middle, high)
        return low, high

    # The real Map of these books is shorter whole than at its parts beside a receipt: it never steps.
    assert starts(smallest(small)[1], small)[2] == ["design_bible"]

    roomy = starts(2_000_000)
    assert roomy and not roomy[2] and len([b for b in roomy[1].system_message()["content"]
                                           if b["text"].startswith("## DEVELOPMENT.md\n")]) == 1
    low, high = smallest(large)
    plan, view, rows, steps, text = starts(high)
    assert rows == ["design_bible", *floor.BOOK_MAPS] and design.operative_view in text and BIG_BIBLE.strip() not in text
    assert any(block["text"].startswith("## DEVELOPMENT.md\n") for block in view.system_message()["content"])
    assert starts(low) is None and "directly required handbook" in refusals[low]  # named, as before
    # No larger window refuses: the one admission prices the handbook's note before it maps the bible (RVD1, Fable N1).
    def whole(window):
        shown = starts(window)
        return bool(shown) and "design_bible" not in shown[2]

    below, whole_from = high, 2_000_000
    while whole_from - below > 1:
        middle = (below + whole_from) // 2
        below, whole_from = (below, middle) if whole(middle) else (middle, whole_from)
    for window in range(whole_from - 64, whole_from):  # where the check without the note let the refusal through
        assert starts(window), (window, refusals.get(window))
    assert starts(whole_from - 1)[2] == ["design_bible"]  # the bible's map alone, where it is enough
    # The bible's map is what made the room: with its retention failing it stays whole and the same window refuses.
    monkeypatch.setattr(floor, "retain_section", lambda *_a, **_kw: (_ for _ in ()).throw(OSError("unavailable")))
    assert starts(high) is None and "directly required handbook" in refusals[high]


def test_the_installed_book_never_comes_from_a_candidate_the_context_is_bound_to(tmp_path):
    """An installed-book fallback (a true declaration from a foreign tree) reads the serving checkout, not an unadopted
    candidate the context is bound to (RV1, Astra 7); a recognized body still reads its own book."""
    from ouroboros.context_layout import author_design_source

    serving, candidate = tmp_path / "serving", tmp_path / "candidate"
    for root, marker in ((serving, "SERVING_RULE"), (candidate, "CANDIDATE_RULE")):
        (root / "docs").mkdir(parents=True)
        (root / "docs/DESIGN.md").write_text(DESIGN.replace("BIBLE_RULE_DELIVERED", marker))
    foreign = foreign_tree(tmp_path / "foreign")
    ctx = SimpleNamespace(repo_dir=candidate, system_repo_dir=candidate, serving_repo_dir=serving,
                          drive_root=tmp_path / "data", workspace_root=foreign, workspace_mode="external", task_metadata={})
    declared = {"context_requires_self_body_docs": True}
    assert author_design_source(ctx, declared) == (serving, True)
    view = task_design_navigation(ctx, declared)
    assert "SERVING_RULE" in view and "CANDIDATE_RULE" not in view
    assert view.startswith("These are the interface rules of Ouroboros itself, from its installed body")


def test_the_bible_is_the_heading_titled_so_not_a_lookalike(tmp_path):
    """Only a heading titled exactly "Design bible" (a subtitle after ": ") starts the section (RV1, Carbon 1.3); the
    slice reads the same whatever line endings the file was written with (RV1, Astra 4)."""
    from ouroboros.context_layout import design_bible_section, design_bible_view

    lookalikes = "# Design\n## Design bible history\nOLD_NOTES\n## Design bible-archive\nARCHIVED\n"
    text = lookalikes + BIBLE + "## Attention\nBODY_RULE_NOT_DELIVERED\n"
    assert design_bible_section(text) == BIBLE.strip()
    assert design_bible_section(lookalikes) is None
    crlf = text.replace("\n", "\r\n")
    assert design_bible_section(crlf) == BIBLE.strip()
    assert design_bible_view(crlf, source="`x`") == design_bible_view(text, source="`x`")
