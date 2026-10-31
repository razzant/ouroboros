"""Pre-v2 loader from 761a992b, bound to its frozen lifecycle writer.

Frozen here so downgrade-writer compatibility tests do not require a Git checkout.
"""

import copy
import pathlib

from ouroboros.swarm_rollup import _accumulate, fold_event_suffix


def load_swarm_rollup(drive_root, task_id):
    """Catch up from the last complete line, including a failed producer update.

    Legacy rows replay once. Rotation preserves chain offsets; missing or torn
    history refuses a partial rollup. A concurrent publisher cannot move the
    cursor backwards, and finalization never invents a missing task row.
    """
    from ouroboros.task_results import load_task_result
    from tests.fixtures.swarm_761a_writer import write_task_result

    root = pathlib.Path(drive_root)
    row = load_task_result(root, task_id, strict=True) or {}
    prior = row.get("swarm_fanout_rollup")
    facts = copy.deepcopy(prior) if isinstance(prior, dict) and prior.get("source_root") == str(root.resolve()) else {}
    children, lanes = set(facts.get("child_ids", [])), list(facts.get("lanes_requested", []))
    through = fold_event_suffix(root, facts.get("through", 0),
                                lambda event: _accumulate(facts, children, lanes, event, task_id))
    facts.update(source_root=str(root.resolve()), through=through, child_ids=sorted(children), lanes_requested=lanes)
    if row and facts != prior:
        def publish(current, fields):
            if current.get("swarm_fanout_rollup") != prior:
                return None
            return {**fields, "status": current["status"], "swarm_fanout_rollup": facts}
        write_task_result(root, task_id, row["status"], _field_projector=publish, strict_existing_dict=True)
    return facts
