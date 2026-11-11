"""A spent journal-reader allowance never opens a stream."""

from ouroboros.gateways.claudexor_run_events import read_run_events


def test_zero_reader_allowance_opens_no_stream():
    class Gateway:
        def open_run_events(self, *_args, **_kwargs):
            raise AssertionError("a spent allowance opened the journal")

    read = read_run_events(Gateway(), "run-1", after_seq=0, through_seq=None, timeout_sec=0)
    assert read.through_seq == 0 and read.stop == "time" and not read.events
