"""Host consumer checks for Claudexor's delegated thread workspace contract."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from ouroboros.review_execution import ReviewRouteUnavailable
from ouroboros.review_thread_continuity import (
    ensure_review_thread,
    require_delegated_review_threads,
    start_review_thread_turn,
)


class Custody:
    @staticmethod
    def idempotency_key(*parts):
        return repr(parts)


@pytest.fixture
def roots(tmp_path):
    author, checking = tmp_path / "author", tmp_path / "checking"
    author.mkdir()
    checking.mkdir()
    return str(author), str(checking)


def _ensure(gateway, roots, *, thread_id="", **changes):
    author, checking = roots
    values = {
        "route": SimpleNamespace(route_id="claude", profile_id="profile-a"),
        "root": author, "workspace_root": checking,
        "surface": "plan_review", "slot_id": "slot-a", "task_id": "task-1",
        "access": "full",
    }
    values.update(changes)
    return ensure_review_thread(gateway, Custody(), thread_id, **values)


@pytest.mark.parametrize("access", ["full", "workspace_write"])
def test_thread_creation_binds_author_identity_and_host_owned_workspace_separately(access, roots):
    author, checking = roots
    calls = []

    class Gateway:
        engine_version = "3.23.0"

        def create_thread(self, request, *, idempotency_key):
            calls.append((request, idempotency_key))
            return {"id": "thread-1"}

    assert _ensure(Gateway(), roots, access=access) == "thread-1"
    request, key = calls[0]
    assert request == {
        "title": "Ouroboros plan_review · slot-a",
        "scope": {"kind": "project", "root": author},
        "mode": "agent", "workspace": "delegated", "workspaceRoot": checking,
        "authPreference": "subscription", "primaryHarness": "claude",
        "eligibleHarnesses": ["claude"], "access": access,
        "credentialProfileId": "profile-a",
    }
    assert repr(author) in key and repr(checking) in key


@pytest.mark.parametrize("workspace_root", ["", "relative-copy", "{author}", "{author}/../author"])
def test_writable_thread_never_falls_back_to_author_root(workspace_root, roots):
    candidate = workspace_root.format(author=roots[0])
    # Author aliases are absolute on every platform, so the identity check refuses them.
    assert Path(candidate).is_absolute() == workspace_root.startswith("{author}")
    with pytest.raises(ReviewRouteUnavailable) as error:
        _ensure(SimpleNamespace(engine_version="3.23.0"), roots, workspace_root=candidate)
    assert error.value.code == "review_thread_workspace_invalid"


def test_readonly_thread_keeps_legacy_wire_shape_below_writable_floor(roots):
    received = []

    class Gateway:
        engine_version = "3.22.0"

        def create_thread(self, request, *, idempotency_key):
            received.append(request)
            return {"id": "readonly-thread"}

    assert _ensure(Gateway(), roots, access="readonly", workspace_root="") == "readonly-thread"
    assert received[0]["scope"] == {"kind": "project", "root": roots[0]}
    assert received[0]["mode"] == "ask"
    assert received[0]["workspace"] == "in_place"
    assert "workspaceRoot" not in received[0]


@pytest.mark.parametrize("response", [{}, {"id": ""}, {"id": "  "}, None])
def test_missing_thread_identity_is_typed_refusal(response, roots):
    class Gateway:
        engine_version = "3.23.0"

        def create_thread(self, request, *, idempotency_key):
            return response

    with pytest.raises(ReviewRouteUnavailable) as error:
        _ensure(Gateway(), roots)
    assert error.value.code == "review_thread_id_missing"


@pytest.mark.parametrize("version", ["", "3.22.99", "unknown", "3.23.0-beta", "3.23garbage.0"])
def test_writable_thread_requires_supported_negotiated_release(version):
    with pytest.raises(ReviewRouteUnavailable) as error:
        require_delegated_review_threads(SimpleNamespace(engine_version=version))
    assert error.value.code == "review_checking_thread_unsupported"


@pytest.mark.parametrize("version", ["3.23.0", "3.23.1", "4.0.0"])
def test_writable_thread_uses_engine_version_without_invented_capability_flag(version):
    require_delegated_review_threads(SimpleNamespace(engine_version=version))


@pytest.mark.parametrize("geometry", [
    {"isolation": "live"},
    {"workspaceKind": "directory", "scopePaths": [], "isolation": "live"},
    {"workspaceKind": "directory", "scopePaths": ["fixture"], "isolation": "live"},
])
def test_thread_turn_keeps_geometry_and_profile_but_not_thread_owned_addresses(geometry):
    received = []

    class Gateway:
        def start_thread_turn(self, thread_id, request, *, idempotency_key):
            received.append((thread_id, request, idempotency_key))
            return {"threadId": thread_id, "turnId": "turn-2", "runId": "run-2"}

    original = {
        "prompt": "Continue the same review", "mode": "agent", "access": "full",
        "scope": {"kind": "project", "root": "/author"},
        "execution": {"delegated": True, "workspaceRoot": "/copy", **geometry},
        "_use_thread": True, "_thread_id": "thread-1",
        "credentialProfileId": None,
    }
    result = start_review_thread_turn(
        Gateway(), "thread-1", original, idempotency_key="turn-2-key",
    )

    assert received == [("thread-1", {
        "prompt": "Continue the same review", "mode": "agent", "access": "full",
        "credentialProfileId": None, "execution": geometry,
    }, "turn-2-key")]
    assert result["runId"] == "run-2"
    assert original["execution"]["workspaceRoot"] == "/copy"
