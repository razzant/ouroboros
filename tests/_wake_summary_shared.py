"""A sanitized stored wake, planted through the current store for word/quiet tests."""
import json
from pathlib import Path

from tests._store_shared import seed_result


def seed_wake(root, *, task_id="214a31b6", **fields):
    document = json.loads((Path(__file__).parent / "fixtures" / "wake_document_214a31b6.json").read_text(encoding="utf-8"))
    document.update(task_id=task_id, root_task_id=task_id, budget_drive_root=str(root))
    document["execution_owner"].update(task_id=task_id, data_root=str(root))
    document.update(fields)
    return seed_result(root, task_id, document)
