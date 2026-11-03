"""Unavailable checks and mandatory evidence overflow cannot become a speech grant."""
import pytest

from tests._safety_fixtures import supervisor as supervisor_fixture
supervisor = supervisor_fixture


@pytest.mark.parametrize("mode", ["advanced", "cyber_pro"])
@pytest.mark.parametrize("required", ["explicit", "consciousness"])
def test_no_checker_hold_and_no_retry_grant(supervisor, monkeypatch, mode, required):
    h = supervisor
    monkeypatch.setattr(h.module, "get_runtime_mode", lambda: mode)
    monkeypatch.setattr(h.module, "_resolve_safety_routing", lambda: (False, False, "no backend"))
    if required == "explicit":
        h.values["OUROBOROS_MODEL_SAFETY"] = "explicit/model"
    else:
        h.ctx.task_metadata["work_kind"] = "consciousness"
    for _ in range(2):
        ok, msg = h.module.check_safety("place_reply", {"text": "hello"}, ctx=h.ctx)
        assert not ok and "SAFETY_UNAVAILABLE" in msg
    assert not h.module._PRIOR_VERDICTS
    assert h.module.check_safety("write_remote", {}, ctx=h.ctx)[0] == (mode == "cyber_pro")
    # Restored checker can approve the exact same speech; it did not consume an earlier flag.
    monkeypatch.setattr(h.module, "_resolve_safety_routing", lambda: (False, False, None))
    assert h.module.check_safety("place_reply", {"text": "hello"}, ctx=h.ctx) == (True, "")
    assert len(h.calls) == 1


def test_legacy_owner_backend_absence_still_warns(supervisor, monkeypatch):
    h = supervisor
    monkeypatch.setattr(h.module, "_resolve_safety_routing", lambda: (False, False, "no backend"))
    assert h.module.check_safety("write_remote", {}, ctx=h.ctx)[0]
    h.values["OUROBOROS_MODEL_SAFETY"] = "required"
    assert not h.module.check_safety("write_remote", {}, ctx=h.ctx)[0]


@pytest.mark.parametrize("source", ["subject", "rules", "directives", "prior"])
def test_mandatory_bundle_never_truncated(supervisor, monkeypatch, source):
    h = supervisor
    monkeypatch.setattr(h.module, "_SAFETY_BUNDLE_BYTE_BUDGET", 2000)
    args = {"text": "small"}
    if source == "subject":
        args["text"] = "tail" * 1000
    elif source == "rules":
        h.values["OUROBOROS_SAFETY_OWNER_RULES"] = "rules" * 1000
    elif source == "directives":
        h.ctx._owner_directives = [{"role": "user", "content": "exact" * 1000}]
    else:
        h.module._PRIOR_VERDICTS["work"] = {"x": {"reason": "earlier" * 1000}}
    for _ in range(2):
        ok, msg = h.module.check_safety("place_reply", args, ctx=h.ctx)
        assert not ok and "TOO_LARGE" in msg
    assert not h.calls
    h.module._PRIOR_VERDICTS.clear()
    h.values["OUROBOROS_SAFETY_OWNER_RULES"] = ""
    h.ctx._owner_directives = []
    assert h.module.check_safety("place_reply", {"text": "fits"}, ctx=h.ctx)[0]
    assert len(h.calls) == 1
