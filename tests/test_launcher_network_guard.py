"""Falsify the independent HTTP/socket boundary without ever opening a connection."""
import http.client
import socket
import urllib.request

import pytest

from tests.launcher_network_guard import deny_real_network


@pytest.mark.parametrize("operation", ["urllib", "http", "socket", "connect_ex"])
def test_real_connect_is_refused_below_the_urlopen_mock(operation):
    with deny_real_network() as attempts:
        with pytest.raises((AssertionError, urllib.error.URLError), match="real HTTP/socket"):
            if operation == "urllib":
                urllib.request.build_opener(urllib.request.ProxyHandler({})).open("http://127.0.0.1:1/api/command")
            elif operation == "http":
                http.client.HTTPConnection("127.0.0.1", 1).request("POST", "/api/command", b'{"cmd":"/panic"}')
            else:
                with socket.socket() as sock:
                    getattr(sock, operation if operation == "connect_ex" else "connect")(("127.0.0.1", 1))
        assert len(attempts) == 1, "the refusal is independently recorded even if a caller catches it"
