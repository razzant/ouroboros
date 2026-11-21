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
    path.write_text("## Later source revision\n")
    assert {mode: json.dumps(plan.messages_for(mode), ensure_ascii=False) for mode in captured} == captured


@pytest.mark.parametrize("owner_mode", ["max", "low", "nano"])
def test_the_design_bible_rides_every_owner_mode(tmp_path, monkeypatch, owner_mode):
    """The owner's horizon choice for Low and Nano is open (item 97, council C13 R4): A, here, captures the bible
    whole in every mode, its map the window's form. B would set ``context_layout.DESIGN_BIBLE_MODES`` to ``{"max"}``
    and this test would expect Low and Nano to capture the map alone (a plain section, no ``design_bible`` form)."""
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
    # A tree not established as the body reads the installed book under a first line saying whose rules these are.
    assert view.startswith("These are the interface rules of Ouroboros itself, from its installed body") is delivered
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
    assert "- Attention" in task_design_navigation(ctx, task)


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


def _fitting(tmp_path, monkeypatch, *, section_root=None, required_words=0, others=True):
    """A real capture shape with real chaptered books, the author's real view of a large bible, ordinary host forms
    that follow it (runtime, review pool, skills), WORLD, a knowledge index and memory whose other conversations
    fold to my shorter telling long before anything else must go: a class that held the bible while memory folds
    (``TELLING_SPARED``, ``WINDOW_ONLY``) would show memory folded beside a whole bible. Returns ``(design, fitted)``;
    ``fitted(window, preferred, known=True)`` -> ``(plan, view, host rows, memory steps, text)``."""
    from ouroboros import context_fit as fit, memory_view
    from ouroboros.reference_books import compose_book, load_reference_book
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
    snapshot = syn.snapshot(story=syn.pages(2, 12_000), room_facts=syn.room("1", lane1=[
        syn.spoken(1, "ouroboros", 8_000), syn.spoken(2, "human", 4_000)]),
        conversations=[syn.conversation(i, words=10, word_chars=600) for i in range(30)])
    head, dynamic = [index, design], [*services, world]
    core = fit.ContextCore(
        base_prompt="SYSTEM WHOLE", bible_md="BIBLE WHOLE", architecture_md=compose_book(books[0]),
        development_md=compose_book(books[1]), docs_need_development=not required_words,
        development_required=bool(required_words), reference_books=books, semi_stable_text="IDENTITY WHOLE",
        dynamic_text="\n\n".join(dynamic), dynamic_head_text="\n\n".join(head),
        user_content_json=json.dumps("WORDS THAT STARTED THE WORK"),
        dynamic_sections=capture_sections(dynamic, prefix="dynamic"),
        dynamic_head_sections=capture_sections(head, prefix="head"),
        section_root=str(tmp_path) if section_root is None else section_root, section_task_id="task-1",
        memory_view_json=memory_view.snapshot_json(snapshot))

    def fitted(window, preferred, known=True):
        evidence = SimpleNamespace(route_fp="route", status="confirmed" if known else "unprobeable", stale=False,
                                   window_tokens=window, source="synthetic")
        plan = fit.build_context_fit_plan(SimpleNamespace(drive_root=tmp_path), core, {"id": "task-1"},
            preferred_mode=preferred, route_resolver=lambda *_a, **_kw: ({"provider": "test", "model": "synthetic"},
                                                                         evidence))
        view = plan.projection(plan.initial_mode)
        return plan, view, *_shown(view)

    return design, fitted


def _shown(view):
    """A projection's host rows, memory steps and system text."""
    rows = [row["section_id"] for row in view.memory_facts.get("host_sections_addressed", ())]
    text = "\n\n".join(block["text"] for block in view.system_message()["content"])
    return rows, (view.memory_facts.get("floor") or {}).get("steps") or {}, text


def test_a_short_room_takes_the_bible_to_its_map_before_memory_or_any_book(tmp_path, monkeypatch):
    """Whenever the start's room or the mode's budget runs short, the bible is the first host form to take its
    navigation (item 97, council C13 R3 and RV1): before the other host forms, before my memory takes a shorter
    telling, before a book map steps, before WORLD or the knowledge index outline, and before Max gives up its whole
    books; the owner set WORLD and the return facts alone above memory (15.11.2026 4A, 18.11.2026 1A). Every frame
    shown at a smaller window holds this; the first window that asks anything takes the bible alone, and a Max the
    window lowered judged its books beside the bible's map."""
    from ouroboros import memory_floor as floor

    design, fitted = _fitting(tmp_path, monkeypatch)
    for preferred in ("max", "low"):
        plan, view, *_ = fitted(2_000_000, preferred)
        total = plan.projection(preferred).estimated_tokens
        whole = floor.view_tokens(plan.core.dynamic_head_text) + floor.view_tokens(plan.core.dynamic_text)
        saving = floor.view_tokens(design.full_text) - floor.view_tokens(design.operative_view)
        window, stages = int(total * 1.6), set()
        while window > total // 4:
            plan, view, rows, steps, text = fitted(window, preferred)
            if "design_bible" in rows:
                assert rows[0] == "design_bible", (preferred, window, rows)  # the first host form to yield
                assert design.operative_view in text and BIG_BIBLE.strip() not in text  # its map stays
            else:  # whole: no other host form, memory step, book map, WORLD or index outline taken before it
                assert not steps and not rows and design.full_text in text, (preferred, window, rows, steps)
            if preferred == "max" and plan.initial_mode != "max":  # the window took Max's whole books ...
                parts = dict(view.memory_facts["max_room"]["parts"])  # ... judged beside the bible's map
                assert parts["knowledge, scratchpad and runtime facts"] < whole - saving // 2, (window, parts)
            stages.add("whole" if not rows else "bible alone" if rows == ["design_bible"] and not steps
                       else "memory or books after it")
            stages.update(["books lowered"] if preferred == "max" and plan.initial_mode != "max" else [])
            window = int(window * (0.9 if stages - {"whole"} else 0.99))  # finely until the first yield
        assert {"whole", "bible alone", "memory or books after it"} <= stages, (preferred, stages)
        assert preferred != "max" or "books lowered" in stages


def test_without_a_measured_window_the_bible_still_pays_before_memory(tmp_path, monkeypatch):
    """No numeric window (a refusal on an unmeasured route, an owner's Nano budget): no host form has the window's
    evidence to yield to, yet the optional bible still pays a requested saving or the mode's budget before memory
    does (RV1, Astra 2); the other host forms, WORLD and the index keep their whole forms there."""
    from ouroboros import context_budget

    design, fitted = _fitting(tmp_path, monkeypatch)
    plan, *_ = fitted(0, "max", known=False)
    assert not _shown(plan.projection("max"))[0]  # nothing asks: the bible stays whole
    asked = {}
    for refused in (1_000, 30_000):
        refit = plan.reproject_for_route(window_tokens=0, known_window=False, ratio=1.0, output_reserve=8_192,
                                         tool_schemas=[], start_mode="max", refused_tokens=refused)
        asked[refused] = _shown(refit.projection(refit.initial_mode))
    rows, steps, text = asked[1_000]
    assert rows == ["design_bible"] and not steps  # a small saving: the bible alone pays it
    assert "omitted for the physical window" in text and design.operative_view in text
    rows, steps, text = asked[30_000]
    assert rows == ["design_bible"] and steps  # a larger one: memory pays only what the bible could not
    # An owner's Nano budget on an unmeasured route: lowered from above, the first budget that binds takes the bible alone.
    request = fitted(0, "nano", known=False)[0].projection("nano").estimated_tokens
    target, seen = request * 2, None
    while seen is None or not seen[0]:
        monkeypatch.setattr(context_budget, "OWNER_NANO_TARGET_TOKENS", target)
        seen, target = _shown(fitted(0, "nano", known=False)[1]), int(target * 0.99)
    rows, steps, text = seen
    assert rows == ["design_bible"] and not steps and "the owner's Nano mode budget" in text


def test_an_actor_without_a_reader_gets_the_whole_bible_or_its_name_unavailable(tmp_path, monkeypatch):
    """A Presence turn whose ceiling has no read_file keeps host forms whole: a pointer would name a source it cannot
    open (BIBLE P1). The optional bible rides whole while it fits; short of room it still yields before memory, and
    only its heading stays, named unavailable to this actor: no map, path or instruction to read (RV1, RVD1 Astra 3)."""
    design, fitted = _fitting(tmp_path, monkeypatch, section_root="")
    plan, *_ = fitted(2_000_000, "max")
    assert design.full_text in _shown(plan.projection("max"))[2]  # with room, the whole bible
    total = plan.projection("max").estimated_tokens
    window, seen = int(total * 1.6), None
    while seen is None or not seen[0]:
        assert seen is None or not seen[1], seen  # memory never shortens while the bible is whole
        seen, window = fitted(window, "max")[2:], int(window * 0.99)
    rows, steps, text = seen
    assert rows == ["design_bible"] and not steps  # the other host forms stay whole for this actor
    assert '## Design bible: "Airy dopamine"\n\n[Omitted for ' in text and "unavailable to this actor." in text
    assert BIG_BIBLE.strip() not in text and "DESIGN.md (navigation map)" not in text
    assert str(tmp_path / "repo/docs/DESIGN.md") not in text and "artifact_store" not in text
    assert "Use this exact source path" not in text


def test_a_required_handbook_start_takes_the_bible_map_before_it_refuses(tmp_path, monkeypatch):
    """A task that directly requires the whole handbook near its route's limit (RV1, Astra 1): where the handbook fits
    only beside the bible's map, the start takes the map and keeps the whole handbook; a window that cannot hold it
    even so is refused by name, as before, and with room the bible stays whole."""
    from ouroboros import context_fit as fit, memory_floor as floor

    design, fitted = _fitting(tmp_path, monkeypatch, required_words=30, others=False)

    refusals = {}

    def starts(window):
        try:
            return fitted(window, "low")
        except fit.MinimumViewUnfit as unfit:
            refusals[window] = str(unfit)
            return None

    roomy = starts(2_000_000)
    assert roomy and not roomy[2] and len([b for b in roomy[1].system_message()["content"]
                                           if b["text"].startswith("## DEVELOPMENT.md\n")]) == 1
    low, high = 1_000, 2_000_000  # the smallest start: refused below, started at or above
    while high - low > 1:
        middle = (low + high) // 2
        low, high = (low, middle) if starts(middle) else (middle, high)
    plan, view, rows, steps, text = starts(high)
    assert rows == ["design_bible"] and design.operative_view in text and BIG_BIBLE.strip() not in text
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
