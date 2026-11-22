"""One finger on a Chromium page through CDP, shared by the phone browser tests (not a test module)."""


def touch(page, x, y, hold_ms, move=None):
    cdp = page.context.new_cdp_session(page)
    try:
        cdp.send('Input.dispatchTouchEvent', {'type': 'touchStart', 'touchPoints': [{'x': x, 'y': y, 'id': 1}]})
        if move:
            for step in range(1, 7):
                cdp.send('Input.dispatchTouchEvent', {'type': 'touchMove', 'touchPoints': [
                    {'x': x + move[0] * step / 6, 'y': y + move[1] * step / 6, 'id': 1}]})
                page.wait_for_timeout(16)
        page.wait_for_timeout(hold_ms)
        cdp.send('Input.dispatchTouchEvent', {'type': 'touchEnd', 'touchPoints': []})
    finally:
        cdp.detach()
