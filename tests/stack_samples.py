"""Periodic Python stack samples for the nested pytest that preflight diagnostics observe.

The fixture used to arm ``faulthandler.dump_traceback_later``, whose C watchdog thread walks
every thread without holding the GIL. In a bounded stress child on bundled CPython 3.10.19
with four busy threads, that watchdog crashed the process with SIGSEGV in each run, while dumps
taken with the GIL held crashed none within the same bounds (#1943). The native cause of the
diagnostic test's earlier crashes is not known.
Each sample here is taken by an ordinary daemon thread calling ``faulthandler.dump_traceback``
with the GIL held, so no Python thread runs or exits under the walk. The limit is native: a
hang that keeps the GIL (code that never releases it) stops the samples, and nothing samples it.

Stdlib only: tests/conftest.py copies this file beside the gate's worker probe, because the
nested pytest cannot import this checkout.
"""
import faulthandler
import threading


def start(path, interval):
    """Append every thread's stack to ``path`` every ``interval`` seconds; return the stop."""
    stream = open(path, "a", encoding="utf-8")
    stopping = threading.Event()

    def sample():
        taken = 0
        while not stopping.wait(interval):
            taken += 1
            # Its own header: a sample is no pytest-timeout or faulthandler timeout report.
            stream.write(f"Stack sample {taken} (every {interval:g}s):\n")
            stream.flush()
            faulthandler.dump_traceback(file=stream, all_threads=True)

    sampler = threading.Thread(target=sample, name="preflight-stack-sampler", daemon=True)
    sampler.start()

    def stop(timeout=5.0):
        """Stop and close once the sampler has ended; a sampler that outlives ``timeout`` keeps the file."""
        stopping.set()
        sampler.join(timeout)
        if sampler.is_alive():
            return False
        stream.close()
        return True

    return stop
