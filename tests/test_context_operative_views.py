"""Host producers' operative views: the actionable facts that stay inline when a window is physically small.

When the window is physically small, host service blocks shrink to their actionable facts inline
plus a pointer to the retained full text; tier zero stays whole. Each producer builds its own view
from the facts it holds, never by parsing its rendered prose. The sizes here follow the composition
of a mature install whose whole call context measured about 184k tokens.
"""
from __future__ import annotations

import types

import ouroboros.context_health as context_health
from ouroboros.delegate_custody import RunCustody


def _env(tmp_path):
    env = types.SimpleNamespace()
    env.drive_root = tmp_path
    env.repo_path = lambda *parts: tmp_path / "repo-none" / "/".join(parts)
    env.drive_path = lambda *parts: tmp_path / "/".join(parts)
    return env


def _obligations(monkeypatch, *, available=119, unread=53, patches=35, faults=2):
    """Delegated-run obligations in a mature install's measured composition (~111 KB of lines)."""
    import ouroboros.delegate_custody as custody

    runs = []
    for i in range(available + unread):
        run = RunCustody(run_id=f"{i:08x}-{i * 7:07x}", task_id=f"{i % 87:08x}")
        run.output_artifact = f"delegated/{run.task_id}/outputs/{'nested-segment/' * 4}{run.run_id}/final.json"
        run.terminal_source = {"kind": "engine"} if i < available else {}
        runs.append(run)
    held = []
    for i in range(patches):
        run = RunCustody(run_id=f"{i:08x}-p{i:06x}", task_id=f"{i % 31:08x}")
        run.patch_captured = bool(i % 2)
        run.target_root = f"/Users/someone/projects/{'deep/' * 8}target-{i}"
        held.append(run)
    open_faults = [{"run_id": f"live-{i:011x}", "task_id": f"{i:08x}", "reason": "stop_unverified",
                    "ts": "2026-10-16T10:00:00Z"} for i in range(faults)]
    monkeypatch.setattr(custody, "settled_unread_outputs", lambda *_a, **_k: runs)
    monkeypatch.setattr(custody, "undisposed_patches", lambda *_a, **_k: held)
    monkeypatch.setattr(custody, "open_containment_faults", lambda *_a, **_k: open_faults)
    return runs, held, open_faults


def test_health_view_keeps_one_short_line_per_obligation_at_the_owners_scale(tmp_path, monkeypatch):
    runs, held, faults = _obligations(monkeypatch)
    other = ["OK: passing check " + "with recorded details " * 20 for _ in range(12)]
    other += ["CRITICAL: VERSION DESYNC — VERSION=1.2.3, README=1.2.2", "WARNING: COST ACCOUNTING UNAVAILABLE — "
              "budget drift check skipped", "INFO: cache hit rate moderate (40%)"]
    monkeypatch.setattr(context_health, "_collect_log_analysis_checks", lambda _env, found: found.extend(other))
    section = context_health.build_health_invariants(_env(tmp_path), task_id="task-reader")
    full, view = section.full_text, section.operative_view
    assert len(full) > 90_000  # the full section keeps every ownership rule and address
    assert len(view) < len(full) / 4  # one short line per obligation under its kind, not its prose
    for run in [*runs, *held]:
        assert f"\n  - run {run.run_id}, owner task {run.task_id}" in view
    assert "- INFO: DELEGATED RESULT AVAILABLE — 119: its owner task reads each retained output.\n" in view
    assert "- WARNING: DELEGATED RESULT NEVER READ — 53: its owner task reads each to EOF.\n" in view
    assert "- WARNING: DELEGATED PATCH AWAITS DISPOSITION — 35: its owner task decides each while live" in view
    for fault in faults:
        assert (f"- CRITICAL: DELEGATED RUN MAY STILL BE LIVE — run {fault['run_id']} (stop_unverified), "
                f"owner task {fault['task_id']}, since {fault['ts']}\n") in view
    # Every other non-passing fact stays exact, unknown state included; passing checks are counted.
    for line in other[-3:]:
        assert f"- {line}\n" in view
    passing = sum(line.startswith("- OK:") for line in full.splitlines())
    assert passing >= 12 and other[0] not in view and view.endswith(f"- {passing} passing checks; details retained.")
    # The full section is unchanged: every exact address and rule is still there to read.
    for run in runs:
        assert run.output_artifact in full and run.output_artifact not in view
    assert "apply requires the caller's active Git root" in full and "apply requires" not in view


def test_health_view_names_the_reader_s_own_verb(tmp_path, monkeypatch):
    runs, held, _faults = _obligations(monkeypatch, available=0, unread=1, patches=1, faults=0)
    monkeypatch.setattr(context_health, "_collect_log_analysis_checks", lambda _env, found: None)
    own = context_health.build_health_invariants(_env(tmp_path), task_id=runs[0].task_id).operative_view
    assert f"\n  - run {runs[0].run_id}, owner task {runs[0].task_id}: this task reads it to EOF\n" in own
    assert (f"\n  - run {held[0].run_id}, owner task {held[0].task_id}, snapshot only, no patch captured: "
            "this task applies or rejects it\n") in own
    foreign = context_health.build_health_invariants(_env(tmp_path), task_id="task-reader").operative_view
    assert f"\n  - run {runs[0].run_id}, owner task {runs[0].task_id}\n" in foreign
    assert "this task" not in foreign


def test_book_maps_keep_their_chapters_and_paths_and_retain_the_map(tmp_path):
    from pathlib import Path

    from ouroboros import context_layout
    from ouroboros.reference_books import load_reference_book
    from ouroboros.section_floor import HostSection

    repo = Path(__file__).resolve().parents[1]
    book = load_reference_book(repo, "architecture")
    section = context_layout.book_map_section(book, "architecture_map")
    assert isinstance(section, HostSection) and section.full_text == context_layout.book_navigation(book)
    for chapter in book.chapters:
        assert f"`{chapter.source_path}`" in section.operative_view
    assert len(section.operative_view) < len(section.full_text) / 4
    assert "(navigation map)" not in section.operative_view  # the per-chapter heading index stays retained
    books = (book, load_reference_book(repo, "development"))
    parts, development = context_layout.reference_doc_sections(None, context_mode="nano", include_development=True,
                                                               books=books)
    assert [getattr(part, "section_id", None) for part in parts][:2] == ["architecture_map", "development_map"]
    full, _dev = context_layout.reference_doc_sections(None, context_mode="max", include_development=False,
                                                        books=books)
    assert not isinstance(full[0], HostSection) or full[0].operative_view is None  # Max keeps the book resident


def test_world_subagents_and_reflections_keep_names_and_state(tmp_path, monkeypatch):
    from ouroboros import context
    from tests.test_cache_optimization import _make_env_and_memory

    profile = "\n\n## System\n\nmacOS 26, arm64, 128 GB.\n\n## Available Tools\n\ngit, node, python.\n\n### Shell\n\nzsh.\n"
    world = context._world_section(profile)
    assert world.section_id == "world" and world.full_text.endswith("\n\n" + profile.strip())
    # Its heading map addresses the file's own lines (the leading blank lines included).
    assert "- System — lines 3-6" in world.operative_view and "  - Shell — lines 11-13" in world.operative_view
    assert "macOS 26" not in world.operative_view and "path='memory/WORLD.md'" in world.operative_view

    env, memory = _make_env_and_memory(tmp_path)
    catalog = {"rows": [{"subagent_id": f"helper-{i}", "route_class": "API model", "effort": "auto",
                         "review_eligible": i == 0, "recommended_use": "The owner's words about this row. " * 20}
                        for i in range(3)]}
    monkeypatch.setattr("ouroboros.subagent_runtime.current_model_visible_subagent_catalog", lambda: catalog)
    core = context._capture_context_core(env, memory, {"id": "t1", "type": "task", "text": "hi"}, None, None)
    subagents = next(part for part in core.semi_stable_sections if part.section_id == "subagents")
    assert "- helper-0: API model, effort auto, reviews" in subagents.operative_view
    assert "- helper-2: API model, effort auto" in subagents.operative_view
    assert "owner's words" in subagents.full_text and "owner's words" not in subagents.operative_view
    assert core.semi_stable_text == "\n\n".join(core.semi_stable_sections)
    assert [part.section_id for part in core.semi_stable_sections if part.section_id in ("identity", "subagents")] \
        == ["subagents", "identity"]

    rows = [{"ts": "2026-10-16T10:00:00Z", "type": "late_review_settlement", "task_type": "task", "task_id": f"t{i}",
             "reflection": "What happened and what I learned. " * 30} for i in range(2)]
    section = context._reflections_section("execution_reflections", "## Execution reflections", rows,
                                           "the whole text", "Historical automatic reflections: 4 row(s) ...")
    assert section.full_text == "## Execution reflections\n\nthe whole text"
    assert section.operative_view == ("## Execution reflections\n\n- 2026-10-16T10:00 | task | t0\n"
                                      "- 2026-10-16T10:00 | task | t1\nHistorical automatic reflections: 4 row(s) ...")


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
    assert section.full_text == "## Runtime context\n\n" + json.dumps(data, ensure_ascii=False, indent=2)
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
