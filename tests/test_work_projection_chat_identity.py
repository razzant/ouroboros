"""A chat id is a VALUE in the work projection: 0 is the hidden partition, a real room; only a missing id is absent.

An answered task's upkeep that no census row names (a direct turn after its answer) is placed by the chat its stored
result names and by the Project a durable binding moved it to. Through the real census, stores and projection, composed
as ``/api/state`` composes them (``tests/_work_projection_shared.state_fields``): the hidden partition keeps its room
beside Main and a Project, a result that names no chat stays unplaced (never turned into the hidden partition's 0), and
a binding places the work by the chat it names, or not at all.
"""
from __future__ import annotations

import pytest

from ouroboros import store
from ouroboros.contracts.chat_id_policy import HIDDEN_CHAT_ID, WEB_UI_CHAT_ID
from ouroboros.post_task_checkpoint import release_post_task_synthesis, reserve_post_task_synthesis
from ouroboros.task_results import write_task_result
from tests._budget_pause_exact_helpers import _install_queue
from tests._work_projection_shared import state_fields

PROJECT_CHAT = 4242  # a Project's room (project chat ids start at chat_id_policy.PROJECT_CHAT_ID_MIN)


@pytest.fixture
def root(tmp_path, monkeypatch):
    _install_queue(tmp_path, monkeypatch)
    yield tmp_path
    store.close_connections_under(tmp_path)


def _upkeep(root, bindings=None, **stored):
    """The ``/api/state`` work fields while one answered task's upkeep still runs and no census row names it."""
    write_task_result(root, "upkeep", "completed", **stored)
    held = reserve_post_task_synthesis(root, "upkeep")
    assert held is not None
    try:
        fields = state_fields(root, bindings)
    finally:
        release_post_task_synthesis(held)
    assert fields["active_chat_activities"] == []  # the projection's own upkeep path, not a census row's
    return fields


@pytest.mark.parametrize("stored, room, key", [
    pytest.param({"chat_id": WEB_UI_CHAT_ID}, WEB_UI_CHAT_ID, "1", id="main"),
    pytest.param({"chat_id": PROJECT_CHAT, "project_id": "p-x"}, PROJECT_CHAT, str(PROJECT_CHAT), id="project"),
    # Negative control: absence is None, not the hidden partition's 0. The page files an unplaced work under the
    # hidden room's key (``String(work.room || 0)``, web/modules/work_status.js) and so does the server.
    pytest.param({}, None, "0", id="names-no-chat"),
])
def test_an_upkeep_stays_in_the_room_its_result_names(root, stored, room, key):
    fields = _upkeep(root, **stored)
    [work] = fields["works"]
    assert work["room"] == room
    assert work["project_id"] == stored.get("project_id", "")
    assert list(fields["rooms"]) == [key]


@pytest.mark.parametrize("binding, room, project", [
    # The registry binds Projects (chat ids from 1000); a binding that names the hidden partition is still a binding.
    pytest.param({"project_id": "", "chat_id": HIDDEN_CHAT_ID}, HIDDEN_CHAT_ID, "", id="hidden-partition"),
    pytest.param({"project_id": "p-x", "chat_id": PROJECT_CHAT}, PROJECT_CHAT, "p-x", id="project"),
    # Negative controls: a binding that names no chat places nothing, and an origin-bound one re-homes nothing.
    pytest.param({"project_id": "p-x"}, WEB_UI_CHAT_ID, "", id="names-no-chat"),
    pytest.param({"project_id": "p-x", "chat_id": PROJECT_CHAT, "origin_bound": True}, WEB_UI_CHAT_ID, "",
                 id="origin-bound"),
])
def test_a_durable_binding_places_the_upkeep_by_the_chat_it_names(root, binding, room, project):
    [work] = _upkeep(root, {"upkeep": binding}, chat_id=WEB_UI_CHAT_ID)["works"]
    assert (work["room"], work["project_id"]) == (room, project)
