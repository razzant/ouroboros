"""A retained interrupted root for tests that admit an owner Continue."""

NONCE = "press-0001-abcdef"


def _running_predecessor(tmp_path, task_id):
    from ouroboros.task_results import STATUS_RUNNING, write_task_result
    from ouroboros.usage_admission import original_group_limit

    prior = original_group_limit(tmp_path, task_id)
    initial = {} if prior["source"] == "ledger_first_row" else {"billing_group": {
        "billing_group_id": task_id, "billing_group_limit_usd": 20.0,
        "billing_group_limit_source": "initial_task_admission", "billing_group_limit_revision": "admission-1"}}
    write_task_result(tmp_path, task_id, STATUS_RUNNING, chat_id=7, project_id="", **initial,
                      origin_message_text="Write the Friday report",
                      origin_message_ref={"chat_id": 7, "client_message_id": "m-1"},
                      root_task_id=task_id, title="Friday report", deadline_at="2099-01-01T00:00:00+00:00")


def _interrupted(tmp_path, task_id="pred-1", *, origin=None, reason_code="", **extra):
    from ouroboros.task_results import STATUS_CANCELLED, write_task_result

    _running_predecessor(tmp_path, task_id)
    fields = {"cancel_origin": origin if origin is not None else {"source": "snapshot_restore",
                                                                   "reason": "server_shutdown"}}
    if reason_code:
        fields = {"reason_code": reason_code}
    write_task_result(tmp_path, task_id, STATUS_CANCELLED if not reason_code else "failed",
                      result="interrupted", **fields, **extra)
