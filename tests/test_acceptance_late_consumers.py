"""A review wave keeps the root cap its usage scope was bound with.

Late acceptance through the owner tool is gone with the task-acceptance machine.
The review substrate it relied on still sends under the caller's bound scope: a
cap the scope names (even "unlimited", which is None with a source) is the cap of
every reviewer send, never replaced by the current per-task setting.
"""
import json

import pytest


@pytest.mark.parametrize('amount,explicit,expected', [(None, 'original_admission', None), (2.0, 'original_admission', 2.0), (None, '', 9.0)])
def test_review_substrate_preserves_bound_original_unlimited_cap(tmp_path, monkeypatch, amount, explicit, expected):
    from ouroboros.review_substrate import ReviewRequest, ReviewSlot, run_review_request
    from ouroboros.usage_accounting import UsageScope, current_usage_scope, usage_scope

    monkeypatch.setenv('OUROBOROS_PER_TASK_COST_USD', '9')
    captured = []

    class Observed:
        def chat(self, **kwargs):
            captured.append(current_usage_scope())
            return {'content': json.dumps({'verdict': 'PASS', 'findings': [], 'summary': 'ok'})}, {}

    with usage_scope(UsageScope(drive_root=tmp_path, task_id='late', root_task_id='original',
                               root_limit_usd=amount, root_limit_source=explicit)):
        run_review_request(ReviewRequest(surface='result', goal='review', task_id='late'),
                           slots=[ReviewSlot(slot_id='slot', model='test/model')], drive_root=tmp_path, llm=Observed())
    assert captured[0].root_task_id == 'original'
    assert captured[0].root_limit_usd == expected and captured[0].root_limit_source == explicit
