from ouroboros.contracts.plugin_api import VALID_EXTENSION_PERMISSIONS


def test_privileged_extension_permissions_have_skill_review_items() -> None:
    text = open("docs/CHECKLISTS.md", encoding="utf-8").read()

    required_mentions = {
        "inject_chat": "inject_chat_minimization",
        "notify_owner": "inject_chat_minimization",
        "presence": "`presence` permission",
        "subscribe_event": "event_subscription_minimization",
        "companion_process": "companion_process_safety",
        "supervised_task": "companion_process_safety",
    }
    for permission, item in required_mentions.items():
        assert permission in VALID_EXTENSION_PERMISSIONS
        assert item in text
