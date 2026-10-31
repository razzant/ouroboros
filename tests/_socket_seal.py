"""Seal test network calls without breaking asyncio's Windows self-pipe."""
import socket
import threading


def block_network_except_socketpair(monkeypatch, refusal):
    """Only socketpair's own numeric-loopback connect passes, in its calling thread."""
    real_socketpair, real_connect, pairing = socket.socketpair, socket.socket.connect, threading.local()

    def own_socketpair(*args, **kwargs):
        outer, pairing.active = getattr(pairing, "active", False), True
        try:
            return real_socketpair(*args, **kwargs)
        finally:
            pairing.active = outer

    def sealed_connect(sock, *args, **kwargs):
        if (getattr(pairing, "active", False) and not kwargs and len(args) == 1
                and isinstance(args[0], tuple) and args[0][:1] in {("127.0.0.1",), ("::1",)}):
            return real_connect(sock, *args)
        return refusal(sock, *args, **kwargs)

    monkeypatch.setattr(socket, "socketpair", own_socketpair)
    monkeypatch.setattr(socket.socket, "connect", sealed_connect)
    monkeypatch.setattr(socket.socket, "connect_ex", refusal)
