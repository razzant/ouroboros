"""Producer-owned views keep navigation and operative authority under physical pressure."""
def test_the_body_map_keeps_every_part_and_address_and_retains_the_whole_map(tmp_path):
    from pathlib import Path

    from ouroboros import context_layout, reference_books
    from ouroboros.reference_books import compose_book, load_reference_book
    from ouroboros.section_floor import HostSection

    repo = Path(__file__).resolve().parents[1]
    books = tuple(load_reference_book(repo, book_id) for book_id in ("architecture", "development"))
    view, parts = reference_books.body_map(*books)
    section = context_layout.body_map_section(view.text, parts, steppable=True)
    assert isinstance(section, HostSection) and section.section_id == "body_map"
    assert (section.full_text, section.operative_view) == (view.text, parts)
    assert view.text.startswith("## Body map\n") and parts.startswith("## Body map (parts and addresses)")
    for chapter in (*books[0].chapters, *books[1].chapters):  # every manual of both books keeps its address
        assert f"`{chapter.source_path}`" in parts
    assert "`docs/DESIGN.md`" in parts and "`docs/PERSISTENCE.md`" in parts  # the Reference Atlas addresses
    assert len(parts) < len(view.text) / 4
    assert "## ARCHITECTURE.md" not in view.text and " lines " not in parts  # no whole book, no line ranges
    compact, development = context_layout.reference_doc_sections(
        context_mode="nano", body_map=view.text, body_map_parts=parts, development_text=compose_book(books[1]))
    assert [getattr(part, "section_id", None) for part in compact] == ["body_map"] and development == ""
    full, _dev = context_layout.reference_doc_sections(context_mode="max", body_map=view.text, body_map_parts=parts)
    assert len(full) == 1 and (not isinstance(full[0], HostSection) or full[0].operative_view is None)
    assert full[0] == view.text  # Max keeps the Map whole: a plain part with nothing shorter to take


def test_world_and_subagents_keep_names_and_state(tmp_path, monkeypatch):
    from ouroboros import context
    from tests.test_cache_optimization import _make_env_and_memory

    profile = "\n\n## System\n\nmacOS 26, arm64, 128 GB.\n\n## Available Tools\n\ngit, node, python.\n\n### Shell\n\nzsh.\n"
    world = context._world_section(profile)
    import hashlib

    # The file exactly as stored (its edges too), then the revision of those bytes.
    revision = hashlib.sha256(profile.encode("utf-8")).hexdigest()
    assert world.section_id == "world" and world.full_text.endswith(
        "\n\n" + profile + f"\nRevision of this map: {revision}.\n")
    # Its heading map addresses the file's own lines (the leading blank lines included).
    assert "- System — lines 3-6" in world.operative_view and "  - Shell — lines 11-13" in world.operative_view
    assert "macOS 26" not in world.operative_view and "path='memory/WORLD.md'" in world.operative_view

    env, memory = _make_env_and_memory(tmp_path)
    catalog = {"rows": [{"subagent_id": f"helper-{i}", "route_class": "API model", "effort": "auto",
                         "review_eligible": i == 0,
                         **({"recommended_use": "The owner's words about this row. " * 20} if i < 2 else {})}
                        for i in range(3)]}
    monkeypatch.setattr("ouroboros.subagent_runtime.current_model_visible_subagent_catalog", lambda: catalog)
    core = context._capture_context_core(env, memory, {"id": "t1", "type": "task", "text": "hi"}, None, None)
    subagents = next(part for part in core.semi_stable_sections if part.section_id == "subagents")
    assert "- helper-0: API model, effort auto, reviews" in subagents.operative_view
    assert "- helper-2: API model, effort auto" in subagents.operative_view
    assert "owner's words" in subagents.full_text
    for row in catalog["rows"][:2]:
        assert row["recommended_use"] in subagents.operative_view
    assert subagents.operative_view.count("recommended_use:") == 2
    assert subagents.operative_view.endswith("- helper-2: API model, effort auto")
    assert core.semi_stable_text == "\n\n".join(core.semi_stable_sections)
    assert [part.section_id for part in core.semi_stable_sections if part.section_id in ("identity", "subagents")] \
        == ["subagents", "identity"]



def test_runtime_view_counts_schedules_candidates_and_the_letter_and_keeps_authority_exact():
    import copy
    import json

    from ouroboros.context_sections import runtime_host_section

    held = [{"id": f"s{i}", "name": "Daily check " * 10, "status": "held", "followup_hold": True} for i in range(17)]
    data = {"runtime_mode": "advanced", "git_head": "abc123",
            "task_contract": {"objective": "Exact owner words", "grants": ["read every source"]},
            "capabilities": {"filesystem": {"writable_roots": ["task_drive", "user_files"], "profile": "full"},
                             "mutative_subagent_surfaces": ["external_workspace"]},
            "scheduled_tasks": {"active": [{"id": "a"}], "held": held, "waiting": [], "omitted_count": 4},
            "body_candidates": [{"id": f"c{i}", "path": "/candidates/" + "deep/" * 10 + str(i)} for i in range(6)],
            "official_update": {"status": "update_available", "letter": {"state": "ready", "relation": "pending",
                                                                       "text": "What changes in this update. " * 40}}}
    before = copy.deepcopy(data)
    section = runtime_host_section(data)
    assert section.full_text == "## Runtime context\n\n" + json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    view = json.loads(section.operative_view.split("\n\n", 1)[1])
    schedules = view["scheduled_tasks"]
    assert (schedules["held"]["count"], schedules["active"]["count"], schedules["waiting"]["count"]) == (17, 1, 0)
    assert schedules["omitted_count"] == 4 and view["body_candidates"]["count"] == 6
    letter = view["official_update"]["letter"]
    assert letter["text"] == {"chars": len(data["official_update"]["letter"]["text"]),
                              "details": "retained full runtime section"}
    assert (letter["state"], letter["relation"], view["official_update"]["status"]) == ("ready", "pending",
                                                                                         "update_available")
    # Authority stays exact: the contract, the filesystem affordances, the surfaces, the scalars.
    for key in ("runtime_mode", "git_head", "task_contract"):
        assert view[key] == data[key]
    assert view["capabilities"] == data["capabilities"]
    assert "Daily check" not in section.operative_view and "Daily check" in section.full_text
    assert len(section.operative_view) < len(section.full_text) / 3 and data == before


def test_runtime_view_counts_the_routing_manifests_history_and_keeps_their_omissions():
    import json

    from ouroboros.context_sections import runtime_host_section

    finals = [{"task_id": f"r{i}", "final_answer": "What this root delivered. " * 40} for i in range(16)]
    dialogue = [{"direction": "in", "text": "The owner's earlier words. " * 20} for _ in range(20)]
    data = {"main_routing_manifest": {"projects": [], "root_tasks": [], "final_results": finals,
                                      "recent_canonical_dialogue": dialogue,
                                      "omissions": {"final_results": 3, "dialogue_gaps": []}},
            "project_routing_manifest": {"final_results": finals[:5], "active_roots": [{"task_id": "live"}],
                                         "omissions": {"active_roots": 0}}}
    section = runtime_host_section(data)
    view = json.loads(section.operative_view.split("\n\n", 1)[1])
    main, project = view["main_routing_manifest"], view["project_routing_manifest"]
    assert (main["final_results"]["count"], main["recent_canonical_dialogue"]["count"]) == (16, 20)
    assert (project["final_results"]["count"], project["active_roots"]["count"]) == (5, 1)
    # What was left out stays named exactly; the whole lists stay in the retained full section.
    assert main["omissions"] == data["main_routing_manifest"]["omissions"]
    assert "What this root delivered" not in section.operative_view
    assert "What this root delivered" in section.full_text and "The owner's earlier words" in section.full_text
    assert len(section.operative_view) < len(section.full_text) / 10
