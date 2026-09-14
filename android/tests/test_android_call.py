"""Exercise the actual CLI transport without Android, credentials or runtime data."""
import json
from pathlib import Path
import runpy
import socket
import tempfile
import threading
import unittest

import pytest

pytestmark = pytest.mark.serial


CALL = runpy.run_path(str(Path(__file__).parents[1] / "bootstrap" / "android-call"))["call"]


class AndroidCallTest(unittest.TestCase):
    def exchange(self, request, make_response):
        observed, errors = [], []
        with tempfile.TemporaryDirectory(prefix="obo-rpc-") as directory:
            address = str(Path(directory) / "socket")
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as listener:
                listener.bind(address)
                listener.listen(1)
                listener.settimeout(2)

                def serve():
                    try:
                        with listener.accept()[0] as connection:
                            connection.settimeout(2)
                            with connection.makefile("rb") as stream:
                                received = json.loads(stream.readline())
                            observed.append(received)
                            response = make_response(received)
                            for start in range(0, len(response), 3):
                                connection.sendall(response[start:start + 3])
                    except Exception as error:
                        errors.append(error)

                worker = threading.Thread(target=serve)
                worker.start()
                result = CALL(request, address, 2)
                worker.join(3)
                self.assertFalse(worker.is_alive())
                self.assertEqual(errors, [])
        self.assertEqual(len(observed), 1)
        return result, observed[0]

    def test_fragmented_unicode_and_typed_payload_are_lossless(self):
        request = {"method": "content.call", "params": {"arg": "Строка\n第二行",
                   "extras": {"id": {"type": "long", "value": "9223372036854775807"}}}}
        result, received = self.exchange(request, lambda p: (json.dumps(
            {"id": p["id"], "ok": True, "result": p["params"]}, ensure_ascii=False) + "\n").encode())
        self.assertEqual(result["result"], request["params"])
        self.assertEqual(received["params"], request["params"])
        self.assertNotIn("id", request)

    def test_lost_response_is_unknown_and_does_not_repeat_mutation(self):
        result, _ = self.exchange({"method": "content.insert"}, lambda p: b'{"ok":true')
        self.assertFalse(result["ok"])
        self.assertEqual(result["error"]["outcome"], "unknown")
        self.assertFalse(result["error"]["retry_automatically"])

    def test_unavailable_socket_has_not_dispatched(self):
        with tempfile.TemporaryDirectory(prefix="obo-rpc-") as directory:
            result = CALL({"method": "capabilities"}, str(Path(directory) / "absent"), 1)
        self.assertEqual(result["error"]["outcome"], "not_dispatched")

    def test_mismatched_response_cannot_confirm_an_operation(self):
        result, _ = self.exchange({"method": "content.insert"},
                                  lambda p: b'{"id":"someone-else","ok":true}\n')
        self.assertFalse(result["ok"])
        self.assertEqual(result["error"]["outcome"], "unknown")

    def test_permission_error_before_parsing_preserves_real_refusal(self):
        failure = {"id": None, "ok": False, "error": {"type": "java.lang.SecurityException",
                   "message": "Root or host UID required", "outcome": "not_dispatched"}}
        result, _ = self.exchange({"method": "capabilities"}, lambda p: (json.dumps(failure) + "\n").encode())
        self.assertEqual(result, failure)


if __name__ == "__main__":
    unittest.main()
