"""Configured review rights through the current pool and physical session consumer.

The gateway is synthetic; custody, frozen inputs and copy capture are real.
"""
from __future__ import annotations

import dataclasses

import pytest

from ouroboros import delegate_custody as custody
from ouroboros import reviewer_slot_config as slots
from ouroboros.review_checking import wave_checking_source
from ouroboros.review_execution import AgentSessionReviewExecutor, ReviewAssignment
from ouroboros.review_records import ReviewRequest
from tests.review_pool_rosters import pool_roster, pool_seat, set_review_pool
from tests.test_review_session_rights import (  # noqa: F401
    FakeGateway, _owned_gateway_uses_each_test_transport, fake_route, repo, writable_engine,
)


def _catalog(monkeypatch, access):
    set_review_pool(monkeypatch, pool_roster(
        pool_seat('session', 'fake-review=fake-small', kind='agent_session', access=access),
        pool_seat('native', 'm/native', delivery='native'),
        pool_seat('packet', 'm/packet', delivery='packet'),
    ))


@pytest.mark.parametrize('access', ['readonly', 'workspace_write', 'full'])
def test_all_pool_consumers_keep_rights_and_the_delivery_enum(monkeypatch, access):
    from ouroboros.tools.plan_review_runtime import plan_review_slots

    _catalog(monkeypatch, access)
    session, native, packet = slots.review_pool_slots()
    assert (session.session_access, native.session_access, packet.session_access) == (access, '', '')
    assert [row.native_retrieval for row in (session, native, packet)] == [False, True, False]
    row_plan = slots.commit_triad_delivery()  # shared commit and skill reader
    assert row_plan['session_accesses'] == [access, '', '']
    assert row_plan['retrieves'] == [True, True, False]
    assert plan_review_slots()[0].session_access == access
    assert slots.catalog_review_row(None, 'session').access == access  # selected preflight/system row
    with slots.composed_review_pool([slots.PoolSeat(session, ('change', 'coupling'))]):
        assert slots.commit_triad_delivery()['session_accesses'] == [access]


@pytest.mark.parametrize('surface', ['multi_model_review', 'review_change', 'preflight', 'skill_review'])
@pytest.mark.parametrize('access', ['readonly', 'full'])
def test_pool_rights_reach_actual_session_request_and_immutable_custody(
        monkeypatch, repo, tmp_path, writable_engine, surface, access):  # noqa: F811
    _catalog(monkeypatch, access)
    slot = dataclasses.replace(slots.review_pool_slots()[0], timeout_sec=60)
    drive = tmp_path / 'drive'
    request = ReviewRequest(surface=surface, goal='Check the frozen input', task_id='pool-rights',
                            session_root=str(repo), session_task='Read the input and report evidence')
    request.checking_source = wave_checking_source(repo, [slot], data_root=drive)
    executor = AgentSessionReviewExecutor(ReviewAssignment(
        request=request, slot=slot, call_id='pool-operation', call_type=surface, custody_root=drive))
    executor._run_session()
    gateway = FakeGateway.instances[-1]
    [body] = gateway.start_requests
    assert body['access'] == access
    record = custody.invocation_record(drive, gateway.start_keys[0])
    assert record['request'] == body
    assert body['mode'] == ('ask' if access == 'readonly' else 'agent')
    if access == 'full':
        assert record['authority_source'] == 'investigation'
        assert body['execution']['workspaceRoot'] != str(repo)
        assert body['execution']['delegated'] is True
    else:
        assert 'execution' not in body and not request.checking_source
