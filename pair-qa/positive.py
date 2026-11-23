"""CI-only subprocess of the paired registry fixture; never discovers a daemon.

Project registration, inspection and release use the real authenticated gateway.
The delegated-run transport is deliberately not exercised: only B reaches a
fixture-only START receiver, which exposes its terminal receipt later.
"""

import json
import os
import re
import sys
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import quote

A, B = "pair-a", "pair-b"
INV_A, INV_B = "pair-inv-a", "pair-inv-b"
ROUTE = "pair-fixture-only"


def check(condition):
    if not condition:
        raise AssertionError


class FixtureStartReceiver:
    """An in-process stand-in for START only; no HTTP, command, or provider."""

    def __init__(self):
        self.submissions = []

    def submit_without_ack(self, invocation_id, request):
        check(invocation_id == INV_B and request["primaryHarness"] == ROUTE)
        self.submissions.append(invocation_id)
        # This fixture accepts B but gives the host no acknowledgement yet.
        return None

    def completed(self, invocation_id, store_id):
        check(self.submissions == [invocation_id] == [INV_B])
        # The fixture recorded this exact submission. Its simulated run is now terminal.
        return {"runId": "pair-fixture-run-b", "storeId": store_id, "state": "succeeded"}


def main():
    stage = "fixture_input"
    try:
        check(sys.platform == "linux" and os.environ.get("PAIR_QA_JOB") == "github-hosted")
        fixture = json.loads(sys.stdin.readline())
        port, root, token = (fixture[k] for k in ("port", "root", "token"))
        check(type(port) is int and 0 < port < 65536)
        check(isinstance(token, str) and re.fullmatch(r"[a-f0-9]{64}", token))
        home = Path.home()
        path = Path(root)
        check(home.parent == Path(os.environ["RUNNER_TEMP"]).resolve() and home.name.startswith("cx-pair-"))
        check(path == home / "project + &" and path == path.resolve(strict=True))
        marker = path / "sentinel.txt"
        check(not marker.is_symlink() and marker.read_bytes() == b"fixture\n")
        descriptor = home / "fixture-generation"
        check(not descriptor.is_symlink() and descriptor.read_bytes() == b"fixture generation\n")
        drive = home / "host-custody"
        drive.mkdir()
        # The child process receives no app/engine/account settings from the parent.
        os.environ["OUROBOROS_DATA_DIR"] = str(drive)
        os.environ["OUROBOROS_SETTINGS_PATH"] = str(drive / "settings.json")

        stage = "host_imports"
        from ouroboros import delegate_custody as custody, delegate_custody_current as current
        from ouroboros import delegate_registration_policy as policy
        from ouroboros.delegate_custody_reconcile import recovered_custody
        from ouroboros.delegate_start_claims import claimed_start_request
        from ouroboros.gateways.claudexor import ClaudexorGateway, DaemonEndpoint
        from ouroboros.gateways.claudexor_maintenance import bound_registration_supported
        from ouroboros.gateways.claudexor_metadata import descriptor_stamp
        from ouroboros.tools.delegate import _retire_orphaned_registration
        from tests._store_shared import seed_result

        endpoint = DaemonEndpoint("127.0.0.1", port, token, str(descriptor), descriptor_stamp(descriptor))
        with ClaudexorGateway(endpoint=endpoint) as gateway:
            project_route = ""
            release_enabled = False
            wire_rejected = []
            releases = []

            def fixture_requests_only(request):
                method, url = request.method, request.url
                route = url.raw_path.split(b"?", 1)[0].decode("ascii")
                origin = (url.scheme, url.host, url.port) == ("http", "127.0.0.1", port)
                control = (method, route) in {
                    ("POST", "/v2/handshake"), ("GET", "/v2/operations"), ("GET", "/v2/projects"),
                }
                register = ((method, route) == ("POST", "/v2/projects")
                            and json.loads(request.content) == {"root": root})
                inspect = bool(project_route and (method, route) == ("GET", project_route + "/registration"))
                release = bool(project_route and (method, route) == ("POST", project_route + "/release"))
                if not origin or not (control or register or inspect or (release and release_enabled)):
                    wire_rejected.append(True)  # latch even if custody catches the exception
                    raise AssertionError
                if release:
                    releases.append(True)

            gateway._client.event_hooks["request"].append(fixture_requests_only)
            stage = "operation_catalog"
            store_id = gateway.handshake()["storeId"]
            check(bool(store_id) and gateway.store_identity == store_id and bound_registration_supported(gateway))
            start = FixtureStartReceiver()

            def claim(task_id, invocation_id):
                seed_result(drive, task_id, {"status": "running", "root_task_id": task_id})
                request = {"scope": {"kind": "project", "root": root}, "primaryHarness": ROUTE,
                           "access": "readonly", "mode": "ask", "prompt": "synthetic fixture"}
                with ExitStack() as preparation:
                    pid, owner, persistent, ephemeral, resolution = policy.resolve_for_start(
                        drive, gateway, root, "", "readonly", task_id=task_id, invocation_id=invocation_id,
                        preparation=preparation,
                    )
                    check(bool(pid) and not persistent and not ephemeral)
                    check(policy.registration_proof(resolution) == {
                        "registration_store_id": store_id, "registration_root": root,
                    })
                    # A cleanup pass between registration and the durable start must also be harmless.
                    custody.retire_settled_registrations(drive, gateway, live_task_ids={A, B})
                    check(not wire_rejected and not releases)
                    landed, refusal = claimed_start_request(
                        drive, gateway=gateway, serving_store_id=store_id, registration_resolution=resolution,
                        task_id=task_id, root_task_id=task_id, run_id="", invocation_id=invocation_id,
                        idempotency_key=invocation_id, project_id=pid, project_owned=bool(owner),
                        project_persistent=False, route=ROUTE, authority_source="readonly", request=request,
                    )
                check(landed and not refusal)
                return pid, owner, resolution, request

            stage = "register_creator_and_adopter"
            pid, owner_a, resolution_a, _ = claim(A, INV_A)
            check(owner_a == pid)
            project_route = "/v2/projects/" + quote(pid, safe="")
            adopted, owner_b, resolution_b, request_b = claim(B, INV_B)
            check(adopted == pid and not owner_b)
            identity = {"projectId": pid, "storeId": store_id, "root": root}

            def close_start(task, invocation_id, owned_id, resolution, *, definite, reason):
                return _retire_orphaned_registration(
                    SimpleNamespace(drive_root=drive, task_id=task), gateway, owned_id,
                    definite_refusal=definite, reason=reason, resolution=resolution, invocation_id=invocation_id,
                )

            stage = "host_closes_unsent_a_and_keeps_pending_b"
            check(start.submit_without_ack(INV_B, request_b) is None)
            unknown = close_start(B, INV_B, owner_b, resolution_b, definite=False, reason="daemon_unreachable")
            unsent = close_start(A, INV_A, owner_a, resolution_a, definite=True, reason="owner_pause")
            check(not unknown["project_retired"] and not unknown.get("project_id"))
            check(not unsent["project_retired"] and unsent["project_retention_reason"] == policy.REGISTRATION_IN_USE)
            check(start.submissions == [INV_B])
            check(custody.invocation_record(drive, INV_A)["state"] == "failed_definite")
            failed = [event for event in custody.custody_rows(drive) if event["type"] == custody.START_FAILED]
            check([(event["invocation_id"], event["definite"]) for event in failed] == [(INV_B, False), (INV_A, True)])
            [duty] = current.owed_registrations(drive)
            check((duty["task_id"], duty["invocation_id"], duty["project_id"]) == (A, INV_A, pid))
            custody.retire_settled_registrations(drive, gateway, live_task_ids={A, B})
            pending = custody.invocation_record(drive, INV_B)
            check((pending["state"], pending["project_id"], pending["request"]) == ("pending", pid, request_b))
            check(not wire_rejected and not releases and current.owed_registrations(drive) == [duty])
            stage = "pending_registration_remains_present"
            check(gateway.bound_registration(pid, store_id, root) == {**identity, "state": "present"})

            stage = "host_binds_late_fixture_terminal"
            late = start.completed(INV_B, store_id)
            entry, shape = recovered_custody(
                drive, gateway, {**pending, "invocation_id": INV_B}, late["runId"], daemon_store_id=late["storeId"],
            )
            check(custody.bind_started(drive, entry, shape=shape, exclusive=True).durable)
            check(custody.invocation_record(drive, INV_B)["run_id"] == late["runId"])
            check(custody.emit(drive, custody.SETTLED, {
                "run_id": late["runId"], "task_id": B, "route": ROUTE, "state": late["state"],
            }))
            settled = current.run(drive, late["runId"])
            check(settled.settled and settled.terminal_state == late["state"])
            # Fixture owner census: no activity or Continue remains after these two tasks finish.
            # As in the host home tests, seed only their result facts, not a task/process terminal hook.
            for task in (A, B):
                row = seed_result(drive, task, {"status": "completed", "root_task_id": task})
                check(row["status"] == "completed")

            stage = "last_use_releases_exactly_once"
            release_enabled = True
            custody.retire_settled_registrations(drive, gateway, live_task_ids=set())
            check(not wire_rejected and len(releases) == 1 and not current.owed_registrations(drive))
            check(gateway.bound_registration(pid, store_id, root) == {**identity, "state": "absent"})
            stage = "repeat_sweep_has_no_new_mutation"
            custody.retire_settled_registrations(drive, gateway, live_task_ids=set())
            check(not wire_rejected and len(releases) == 1 and not current.owed_registrations(drive))
            events = [event for event in custody.custody_rows(drive) if event["type"] == custody.PROJECT_RETIRED]
            [retired] = events
            check(retired["project_id"] == pid and retired["removal_outcome"] == "removed")
            check(marker.read_bytes() == b"fixture\n")
    except Exception as exc:
        # Never emit an exception message, request, response, path, or token.
        print(f"paired gateway failed at {stage} ({type(exc).__name__})", file=sys.stderr)
        return 1
    print("paired custody: unsent A; B pending -> fixture terminal; release POST=1; repeat=0")
    return 0


if __name__ == "__main__":
    sys.exit(main())
