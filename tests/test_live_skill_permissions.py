"""Voice permissions use the ordinary skill permission selector."""


def test_voice_permissions_are_requested_only_when_declared():
    from ouroboros.skill_loader import requested_skill_permissions

    assert requested_skill_permissions(["call_peer", "live_conversation", "call_peer"]) == [
        "live_conversation", "call_peer"]
    assert requested_skill_permissions([]) == []
