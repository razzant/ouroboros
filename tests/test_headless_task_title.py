"""Every headless run gets an owner-facing name (owner decision 6A/4A).

A task created from the CLI or the task API carried no name at all, so its card
fell back to the status phrase and rendered "Done with warnings" twice — once as
the title, once as the chip. Naming it costs no model call: an explicit
``--title`` is authorship and fills both name slots like a promoted chat turn,
and without one the request's own first line is derived for display.
"""

from types import SimpleNamespace

import pytest
from starlette.applications import Starlette
from starlette.routing import Route
from starlette.testclient import TestClient

from ouroboros.gateway.tasks import _admission_names, api_tasks_create
from ouroboros.projects_registry import PROJECT_NAME_MAX
from ouroboros.task_results import load_task_result


def test_an_explicit_title_is_authorship_and_fills_both_slots():
    assert _admission_names({"title": "Audit the report"}, "ignored") == (
        "Audit the report", "Audit the report",
    )


def test_a_derived_name_is_display_only_and_leaves_title_empty():
    """`title` means "someone named this".

    It sits FIRST in the presentation cascade, so a truncated prompt written
    there would outrank a real name coined later. The derivation fills only the
    slot the card, history replay and the Project lifecycle row actually read.
    """
    title, suggested = _admission_names({}, "Fix the failing test\n\nmore detail")
    assert title == ""
    assert suggested == "Fix the failing test"


def test_markdown_is_stripped_before_the_first_line_is_taken():
    # The live incident: the Main lifecycle row read "Project › # Задача: ...".
    title, suggested = _admission_names(
        {}, "# Задача: жёсткая многомодельная прожарка отчёта\n\nтело запроса",
    )
    assert title == ""
    assert suggested == "Задача: жёсткая многомодельная прожарка отчёта"


def test_a_long_first_line_is_capped_like_a_project_name():
    _title, suggested = _admission_names({}, "https://example.com/" + "x" * 400)
    assert len(suggested) == PROJECT_NAME_MAX
    assert suggested.endswith("…")


def test_an_empty_request_never_invents_a_name():
    assert _admission_names({}, "") == ("", "")


@pytest.fixture()
def admission(tmp_path, monkeypatch):
    from supervisor import workers

    repo = tmp_path / "repo"
    repo.mkdir()
    data = tmp_path / "data"
    (data / "memory").mkdir(parents=True)
    (data / "memory" / "identity.md").write_text("seed identity", encoding="utf-8")
    monkeypatch.setattr(workers, "WORKERS", {0: SimpleNamespace()})
    monkeypatch.setattr(workers, "_WORKER_POOL_DISABLED_REASON", "")
    captured, broadcasts = [], []
    monkeypatch.setattr("supervisor.queue.enqueue_task", lambda task: captured.append(dict(task)) or task)
    monkeypatch.setattr("supervisor.queue.persist_queue_snapshot", lambda reason="": True)
    monkeypatch.setattr("ouroboros.workspace_admission.bootstrap_process_path", lambda: [])
    monkeypatch.setattr(
        "supervisor.message_bus.try_get_bridge",
        lambda: SimpleNamespace(broadcast=lambda payload: broadcasts.append(payload)),
    )
    app = Starlette(routes=[Route("/api/tasks", endpoint=api_tasks_create, methods=["POST"])])
    app.state.drive_root = data
    app.state.repo_dir = repo
    return TestClient(app), data, captured, broadcasts


def test_the_admitted_task_and_its_durable_record_carry_the_name(admission):
    client, data, captured, _broadcasts = admission
    response = client.post("/api/tasks", json={"description": "audit the report", "title": "Nightly audit"})
    assert response.status_code == 200, response.text
    assert captured[0]["title"] == "Nightly audit"
    assert captured[0]["suggested_name"] == "Nightly audit"
    stored = load_task_result(data, response.json()["task_id"]) or {}
    assert stored.get("title") == "Nightly audit"
    assert stored.get("suggested_name") == "Nightly audit"


def test_the_card_learns_the_name_from_the_first_frame(admission):
    """Without this the card paints as its status phrase until a history replay.

    The frame is WS-only — never a chat.jsonl row — and the client buffers a
    name that arrives before the card exists, so ordering does not matter.
    """
    client, _data, _captured, broadcasts = admission
    response = client.post("/api/tasks", json={"description": "Rebuild the index\nand verify"})
    assert response.status_code == 200
    assert broadcasts == [{
        "type": "task_named",
        "task_id": response.json()["task_id"],
        "suggested_name": "Rebuild the index",
    }]


def test_a_title_hidden_in_metadata_is_refused_like_a_project_id(admission):
    client, _data, _captured, _broadcasts = admission
    response = client.post(
        "/api/tasks", json={"description": "x", "metadata": {"title": "sneaky"}},
    )
    assert response.status_code == 400
    assert "title must be a top-level field, not metadata" in response.text
    # The pre-existing refusal keeps its exact wording.
    other = client.post("/api/tasks", json={"description": "x", "metadata": {"project_id": "p"}})
    assert other.status_code == 400
    assert "project_id must be a top-level field, not metadata" in other.text


def test_a_derived_name_is_not_reported_as_model_coined():
    """Turn-into-project reuses the name slot; it must not claim authorship.

    Two producers fill `suggested_name`: the proactive namer coins one with a
    model, and headless admission derives one from the request's first line.
    The conversion cannot tell them apart, so its naming reason names the SLOT
    it read rather than a coiner that may not exist.
    """
    import pathlib

    source = (pathlib.Path(__file__).resolve().parents[1] / "ouroboros/gateway/projects.py").read_text(
        encoding="utf-8",
    )
    assert '"proactive_namer"' not in source
    assert '"preset_suggested_name"' in source
    # An explicit caller title still reports itself as exactly that.
    assert '"explicit_task_title"' in source
