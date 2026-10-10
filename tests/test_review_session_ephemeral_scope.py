"""A review reads a tree the review stack minted without registering it.

An isolated review checkout or retained input view under the data root (or a
registry checkout) is one wave's tree: without a sticky thread its session declares
``scope.ephemeral`` and makes no project request. A sticky plan-review thread and any
other root (the system repository, an owner's project) keep their registration.
"""

from __future__ import annotations

import pytest

from tests._review_session_route_shared import _owned_gateway_uses_each_test_transport as __owned_gateway_uses_each_test_transport
from tests._review_session_route_shared import fake_route as __fake_route

# Fixtures are requested by name as test parameters, so they are re-bound through a
# module attribute: a direct import of a name that reappears as a parameter is an F811
# redefinition under the CI ruff gate.
_owned_gateway_uses_each_test_transport = __owned_gateway_uses_each_test_transport
fake_route = __fake_route

from tests._review_session_route_shared import _run_session_directly  # noqa: E402


_ROOTS = {
    "review_checkout": ("state", "review_checkouts", "tok", "repo"),
    "review_inputs": ("artifacts", "t-plan", "source_handles", "review_inputs", "request-1"),
    # An owner's project that happens to live under the data root is not a minted tree.
    "owner_under_data": ("workspace",),
}


@pytest.mark.parametrize("where", ["review_checkout", "review_inputs", "owner_under_data", "elsewhere"])
def test_a_minted_review_tree_starts_ephemeral_without_a_project_request(fake_route, tmp_path, where):
    root = tmp_path.joinpath(*_ROOTS[where]) if where in _ROOTS else "/tmp/fake-repo"
    if where in _ROOTS:
        root.mkdir(parents=True)
    _run_session_directly(tmp_path, root=str(root))
    gateway = fake_route.instances[-1]
    [request] = gateway.start_requests
    if where in ("review_checkout", "review_inputs"):
        assert request["scope"] == {"kind": "project", "root": str(root), "ephemeral": True}
        assert gateway.project_lookups == [] and gateway.registrations == []
    else:  # not a tree the host minted: the registration flow is unchanged
        assert request["scope"] == {"kind": "project", "root": str(root)}
        assert gateway.project_lookups == [str(root)]


def test_a_sticky_review_thread_keeps_its_registration(fake_route, tmp_path, monkeypatch):
    from ouroboros import observability, review_execution

    root = tmp_path / "state" / "review_checkouts" / "tok" / "repo"
    root.mkdir(parents=True)
    threads = []

    def create_thread(self, request, **_kwargs):
        threads.append(request)
        return {"id": "retained-thread"}

    def fail_blob(*_args, **_kwargs):  # stop right after registration and thread creation
        raise OSError("request blob cannot be persisted")

    monkeypatch.setattr(fake_route, "create_thread", create_thread, raising=False)
    monkeypatch.setattr(observability, "write_blob", fail_blob)
    invocation = review_execution.SessionInvocation(
        task_id="t-plan", surface="plan_review", slot_id="plan_slot_1", timeout_sec=30, use_thread=True)
    with pytest.raises(review_execution.ReviewRouteUnavailable):
        review_execution.run_delegated_review_session(
            prompt="review", root=str(root), custody_drive=tmp_path, invocation=invocation)
    gateway = fake_route.instances[-1]
    assert gateway.project_lookups == [str(root)], "the thread needs the registration it is bound to"
    assert threads and threads[0]["scope"] == {"kind": "project", "root": str(root)}
