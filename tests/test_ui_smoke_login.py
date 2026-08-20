"""The harness login card and the faces it may show.

Split verbatim out of ``tests/test_ui_smoke_playwright.py`` by theme. This module owns
the explicit recovery, reconcile, detach and retry actions, the dismissal that may not
drop a live job or freeze the card, the stale GET that may not overwrite a terminal
face, and the page-hide that detaches without a lifecycle request.

Every test here launches a real browser and is marked ``ui_browser``, so the default
local run deselects the whole module.
"""

from __future__ import annotations


import pytest


from tests._ui_smoke_shared import direct_server_with_data as _direct_server_with_data

# Fixtures are requested by name as test parameters, so they are re-bound through a
# module attribute: a direct import of a name that reappears as a parameter is an F811
# redefinition under the CI ruff gate.
direct_server_with_data = _direct_server_with_data


@pytest.mark.ui_browser
def test_ui_smoke_login_recovery_reconcile_detach_and_retry_are_explicit(direct_server_with_data):
    """Recovery lifecycle."""
    pytest.importorskip("playwright.sync_api", reason="Playwright is not installed")
    from playwright.sync_api import Error as PlaywrightError
    from playwright.sync_api import sync_playwright

    url = direct_server_with_data["url"]
    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch()
            try:
                page = browser.new_page()
                posts: list[str] = []
                deletes: list[str] = []
                reconciles: list[str] = []

                def handle_create(route):
                    posts.append(route.request.url)
                    if len(posts) == 1:
                        job = '{"state": "running"}'
                    elif len(posts) == 2:
                        job = ('{"state": "failed", "outcome": '
                               '{"reason": "termination_unconfirmed"}}')
                    else:
                        job = '{"state": "running"}'
                    route.fulfill(
                        status=200,
                        content_type="application/json",
                        body='{"job_id":"job-recovery","job":' + job + '}',
                    )

                def handle_job(route):
                    if route.request.url.endswith("/reconcile"):
                        reconciles.append(route.request.url)
                        if len(reconciles) == 1:
                            route.fulfill(status=409, content_type="application/json", body=(
                                '{"error":"still present","code":"setup_termination_unconfirmed",'
                                '"required_actions":["retry_setup_reconciliation"]}'
                            ))
                        else:
                            route.fulfill(status=200, content_type="application/json", body=(
                                '{"job":{"state":"failed","outcome":'
                                '{"reason":"termination_unconfirmed"},'
                                '"terminationReconciliation":{"status":"empty"}}}'
                            ))
                    else:
                        deletes.append(route.request.url)
                        route.fulfill(status=200, content_type="application/json", body=(
                            '{"job":{"state":"failed","outcome":'
                            '{"reason":"termination_unconfirmed"}}}'
                        ))

                page.route("**/api/claudexor/login", handle_create)
                page.route("**/api/claudexor/login/*", handle_job)
                page.route("**/api/claudexor/login/*/reconcile", handle_job)
                page.goto(url, wait_until="domcontentloaded", timeout=30_000)

                setup_result = page.evaluate(
                    """
                    async () => {
                        const host = document.getElementById('harness-login-card');
                        if (!host) return 'NO-HOST';
                        const m = await import('/static/modules/harness_accounts.js');
                        const wait = async (sel) => {
                            for (let i = 0; i < 100; i++) {
                                const b = host.querySelector(sel);
                                if (b && !b.disabled) return b;
                                await new Promise((r) => setTimeout(r, 20));
                            }
                        };
                        const p1 = m.startLogin('codex', 'race-a');
                        const p2 = m.startLogin('codex', 'race-a');
                        await Promise.all([p1, p2]);
                        host.querySelector('[data-login-dismiss]')?.click();
                        (await wait('[data-login-reconcile]'))?.click();
                        return 'RECONCILE-CLICKED';
                    }
                    """
                )
                assert setup_result == "RECONCILE-CLICKED"
                # Deterministic settle wait (a fixed sleep was flaky under load:
                # the card could still say "Checking…"). The reconcile round-trip
                # is settled only when the card re-renders the retained-custody
                # recovery face: the outcome detail note exists (it is absent
                # before the click) and "Check again" is enabled again.
                page.wait_for_function(
                    "() => { const host = document.getElementById('harness-login-card');"
                    " const btn = host?.querySelector('[data-login-reconcile]');"
                    " return Boolean(host?.querySelector('[data-login-detail]'))"
                    " && Boolean(btn) && !btn.disabled; }",
                    timeout=30_000,
                )
                recovery_html = page.evaluate(
                    "() => document.getElementById('harness-login-card').innerHTML"
                )
                assert len(posts) == 1
                assert len(deletes) == 1
                assert len(reconciles) == 1
                assert "Check again" in recovery_html and "job-recovery" in reconciles[0]

                before_detach = (len(posts), len(deletes), len(reconciles))
                detached_html = page.evaluate(
                    """async () => {
                        const h = document.getElementById('harness-login-card');
                        h.querySelector('[data-login-dismiss]')?.click();
                        await new Promise((r) => setTimeout(r, 50));
                        return h.innerHTML;
                    }"""
                )
                assert detached_html == ""
                assert (len(posts), len(deletes), len(reconciles)) == before_detach

                final_html = page.evaluate(
                    """async () => {
                        const h = document.getElementById('harness-login-card');
                        const m = await import('/static/modules/harness_accounts.js');
                        await m.startLogin('codex', 'race-a');
                        h.querySelector('[data-login-reconcile]')?.click();
                        for (let i = 0; i < 100 && !h.querySelector('[data-login-retry]'); i++)
                            await new Promise((r) => setTimeout(r, 20));
                        h.querySelector('[data-login-retry]')?.click();
                        await new Promise((r) => setTimeout(r, 100));
                        return h.innerHTML;
                    }"""
                )
                assert len(posts) == 3
                assert len(deletes) == 1
                assert len(reconciles) == 2 and all("job-recovery" in u for u in reconciles)
                assert "Starting" in final_html or "sign-in" in final_html
            finally:
                browser.close()
    except PlaywrightError as exc:
        if "Executable doesn't exist" in str(exc) or "playwright install" in str(exc).lower():
            pytest.skip(str(exc))
        raise

@pytest.mark.ui_browser
def test_ui_smoke_dismiss_overlapping_start_cannot_drop_a_live_job(direct_server_with_data):
    """Queued start follows slow Dismiss."""
    import time as _time

    pytest.importorskip("playwright.sync_api", reason="Playwright is not installed")
    from playwright.sync_api import Error as PlaywrightError
    from playwright.sync_api import sync_playwright

    url = direct_server_with_data["url"]
    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch()
            try:
                page = browser.new_page()
                events: list = []

                def handle_create(route):
                    events.append("post")
                    route.fulfill(
                        status=200,
                        content_type="application/json",
                        body='{"job_id": "job-ov-%d", "job": {"state": "running"},'
                             ' "attach_command": ""}' % len(events),
                    )

                def handle_cancel(route):
                    events.append("delete-open")
                    _time.sleep(0.35)
                    events.append("delete-done")
                    route.fulfill(status=200, content_type="application/json",
                                  body='{"job":{"state":"cancelled"}}')

                page.route("**/api/claudexor/login", handle_create)
                page.route("**/api/claudexor/login/*", handle_cancel)
                page.goto(url, wait_until="domcontentloaded", timeout=30_000)

                result = page.evaluate(
                    """
                    async () => {
                        const host = document.getElementById('harness-login-card');
                        if (!host) return { error: 'NO-HOST' };
                        const m = await import('/static/modules/harness_accounts.js');
                        await m.startLogin('codex', 'ov-a');
                        host.querySelector('[data-login-dismiss]')?.click();
                        await m.startLogin('codex', 'ov-b');
                        await new Promise((r) => setTimeout(r, 600));
                        const cardAfterQueuedStart = host.innerHTML.length > 0;
                        await m.startLogin('codex', 'ov-c');
                        return {
                            cardAfterQueuedStart,
                            finalHasCard: host.innerHTML.length > 0,
                        };
                    }
                    """
                )
                assert result.get("error") is None
                assert result["cardAfterQueuedStart"] is True
                assert result["finalHasCard"] is True
                posts = events.count("post")
                deletes = events.count("delete-open")
                assert posts == 3
                assert deletes == 2
                first_delete_done = events.index("delete-done")
                second_post = [i for i, e in enumerate(events) if e == "post"][1]
                assert first_delete_done < second_post
            finally:
                browser.close()
    except PlaywrightError as exc:
        if "Executable doesn't exist" in str(exc) or "playwright install" in str(exc).lower():
            pytest.skip(str(exc))
        raise

@pytest.mark.ui_browser
@pytest.mark.parametrize("face", ["recovery", "reconciled", "unavailable"])
def test_ui_smoke_stale_get_cannot_overwrite_login_terminal_faces(
    direct_server_with_data, face,
):
    """Stale GET cannot repaint custody."""
    pytest.importorskip("playwright.sync_api", reason="Playwright is not installed")
    from playwright.sync_api import Error as PlaywrightError
    from playwright.sync_api import sync_playwright

    url = direct_server_with_data["url"]
    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch()
            try:
                page = browser.new_page()
                creates: list[str] = []
                deletes: list[str] = []
                reconciles: list[str] = []

                def handle_create(route):
                    creates.append(route.request.url)
                    route.fulfill(
                        status=200,
                        content_type="application/json",
                        body='{"job_id": "job-stale", "job": {"state": "running"},'
                             ' "attach_command": ""}',
                    )

                def handle_job(route):
                    if route.request.method == "DELETE":
                        deletes.append(route.request.url)
                        if face == "unavailable":
                            route.fulfill(status=404, content_type="application/json", body="{}")
                        else:
                            route.fulfill(status=200, content_type="application/json", body=(
                                '{"job":{"state":"failed","outcome":'
                                '{"reason":"termination_unconfirmed"}}}'
                            ))
                        return
                    route.fulfill(status=200, content_type="application/json",
                                  body='{"job": {"state": "running"}}')

                def handle_reconcile(route):
                    reconciles.append(route.request.url)
                    route.fulfill(status=200, content_type="application/json", body=(
                        '{"job":{"state":"failed","outcome":'
                        '{"reason":"termination_unconfirmed"},'
                        '"terminationReconciliation":{"status":"empty"}}}'
                    ))

                page.route("**/api/claudexor/login", handle_create)
                page.route("**/api/claudexor/login/*", handle_job)
                page.route("**/api/claudexor/login/*/reconcile", handle_reconcile)
                page.goto(url, wait_until="domcontentloaded", timeout=30_000)

                result = page.evaluate(
                    """
                    async (face) => {
                        const realFetch = window.fetch.bind(window);
                        let releaseStale;
                        const stale = new Promise((resolve) => { releaseStale = resolve; });
                        let gets = 0;
                        window.fetch = (input, init = {}) => {
                            const url = String(input?.url || input);
                            const method = String(init.method || input?.method || 'GET').toUpperCase();
                            if (method === 'GET' && url.includes('/api/claudexor/login/job-stale')) {
                                gets += 1;
                                return stale;
                            }
                            return realFetch(input, init);
                        };
                        const host = document.getElementById('harness-login-card');
                        const m = await import('/static/modules/harness_accounts.js');
                        await m.startLogin('codex', 'stale-' + face);
                        await new Promise((r) => setTimeout(r, 3200));
                        host.querySelector('[data-login-dismiss]')?.click();
                        await new Promise((r) => setTimeout(r, 100));
                        if (face === 'reconciled') {
                            host.querySelector('[data-login-reconcile]')?.click();
                            await new Promise((r) => setTimeout(r, 100));
                        }
                        const before = host.innerHTML;
                        releaseStale(new Response('{"job":{"state":"running"}}', {
                            status: 200, headers: { 'Content-Type': 'application/json' },
                        }));
                        await new Promise((r) => setTimeout(r, 3400));
                        return { before, after: host.innerHTML, gets };
                    }
                    """, face,
                )
                assert result["gets"] == 1
                assert result["before"] == result["after"]
                marker = {
                    "recovery": "could not prove",
                    "reconciled": "no longer blocking",
                    "unavailable": "no longer available",
                }[face]
                assert marker in result["after"]
                assert len(creates) == 1 and len(deletes) == 1
                assert len(reconciles) == (1 if face == "reconciled" else 0)
            finally:
                browser.close()
    except PlaywrightError as exc:
        if "Executable doesn't exist" in str(exc) or "playwright install" in str(exc).lower():
            pytest.skip(str(exc))
        raise

@pytest.mark.ui_browser
def test_ui_smoke_window_pagehide_detaches_login_without_lifecycle_http(direct_server_with_data):
    """Window pagehide detaches locally."""
    pytest.importorskip("playwright.sync_api", reason="Playwright is not installed")
    from playwright.sync_api import Error as PlaywrightError
    from playwright.sync_api import sync_playwright

    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch()
            try:
                page = browser.new_page()
                page.goto(direct_server_with_data["url"], wait_until="domcontentloaded")
                result = page.evaluate(
                    """
                    async () => {
                        const {createAgentsStep} = await import('/static/modules/onboarding_agents_step.js');
                        let connect, release;
                        const pending = new Promise((r) => { release = r; });
                        const calls = {create: 0, delete: 0, reconcile: 0, get: 0};
                        const button = {getAttribute: () => 'claude',
                            addEventListener: (_t, fn) => { connect = fn; }};
                        const host = {innerHTML: '', querySelector: () => null};
                        const list = {innerHTML: '', querySelectorAll: () => [button]};
                        const other = {textContent: '', hidden: false, dataset: {}};
                        const doc = {defaultView: window, getElementById: (id) =>
                            id === 'agents-login-host' ? host
                                : id === 'agents-family-list' ? list : other};
                        const store = {
                            accountsKnown: false, snapshot: null, subscribe: () => () => {},
                            refresh: () => {}, unavailableNote: () => null,
                        };
                        const fetchImpl = async (input, init={}) => {
                            const url=String(input), method=init.method || 'GET';
                            if (url === '/api/claudexor/login' && method === 'POST') {
                                calls.create++; return pending;
                            }
                            if (url.endsWith('/reconcile')) calls.reconcile++;
                            else if (method === 'DELETE') calls.delete++; else calls.get++;
                            return new Response('{"job":{"state":"running"}}', {
                                status: 200, headers: {'Content-Type':'application/json'}});
                        };
                        const step = createAgentsStep({doc, store, fetchImpl});
                        step.mount(); connect(); await Promise.resolve();
                        window.dispatchEvent(new PageTransitionEvent('pagehide',{persisted:true}));
                        const cached=host.innerHTML, before={...calls};
                        window.dispatchEvent(new PageTransitionEvent('pagehide',{persisted:false}));
                        const immediate=host.innerHTML;
                        connect(); await Promise.resolve();
                        release(new Response('{"job_id":"late","job":{"state":"running"}}',
                            {status:200,headers:{'Content-Type':'application/json'}}));
                        await new Promise((r) => setTimeout(r, 50));
                        return {cached, immediate, final:host.innerHTML, before, after:calls};
                    }
                    """
                )
                assert result["cached"]
                assert result["immediate"] == result["final"] == ""
                assert result["before"] == result["after"] == dict(
                    create=1, delete=0, reconcile=0, get=0)
            finally:
                browser.close()
    except PlaywrightError as exc:
        if "Executable doesn't exist" in str(exc) or "playwright install" in str(exc).lower():
            pytest.skip(str(exc))
        raise

@pytest.mark.ui_browser
def test_ui_smoke_dismiss_overlapping_settle_never_freezes_the_card(direct_server_with_data):
    """Terminal GET wins over slow Dismiss."""
    pytest.importorskip("playwright.sync_api", reason="Playwright is not installed")
    from playwright.sync_api import Error as PlaywrightError
    from playwright.sync_api import sync_playwright

    url = direct_server_with_data["url"]
    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch()
            try:
                page = browser.new_page()
                creates: list[str] = []
                gets: list[str] = []
                reconciles: list[str] = []

                def handle_create(route):
                    creates.append(route.request.url)
                    route.fulfill(status=200, content_type="application/json",
                                  body='{"job_id": "job-os-1", "job": {"state": "running"},'
                                       ' "attach_command": ""}')

                def handle_job(route):
                    gets.append(route.request.url)
                    route.fulfill(status=200, content_type="application/json",
                                  body='{"job": {"state": "succeeded"}}')

                def handle_reconcile(route):
                    reconciles.append(route.request.url)
                    route.fulfill(status=500, content_type="application/json", body="{}")

                page.route("**/api/claudexor/login", handle_create)
                page.route("**/api/claudexor/login/*", handle_job)
                page.route("**/api/claudexor/login/*/reconcile", handle_reconcile)
                page.goto(url, wait_until="domcontentloaded", timeout=30_000)

                result = page.evaluate(
                    """
                    async () => {
                        const realFetch = window.fetch.bind(window);
                        let deletes = 0;
                        window.fetch = (input, init = {}) => {
                            const url = String(input && input.url ? input.url : input);
                            const method = String((init && init.method)
                                || (input && input.method) || 'GET').toUpperCase();
                            if (method === 'DELETE' && url.includes('/api/claudexor/login/')) {
                                deletes += 1;
                                return new Promise((resolve) => setTimeout(() => resolve(
                                    new Response('{"error": "daemon busy"}',
                                        { status: 503,
                                          headers: { 'Content-Type': 'application/json' } })
                                ), 4000));
                            }
                            return realFetch(input, init);
                        };
                        const host = document.getElementById('harness-login-card');
                        const m = await import('/static/modules/harness_accounts.js');
                        await m.startLogin('codex', 'os-a');
                        host.querySelector('[data-login-dismiss]')?.click();
                        await new Promise((r) => setTimeout(r, 5200));
                        return { html: host.innerHTML, deletes,
                            cardCount: host.querySelectorAll('[data-login-card]').length,
                            verdict: host.querySelector('[data-login-verdict]')?.textContent.trim() };
                    }
                    """
                )
                assert result["cardCount"] == 1
                assert result["verdict"] == "Connected."
                assert "Could not cancel" not in result["html"]
                assert (len(creates), result["deletes"], len(gets), len(reconciles)) == (1, 1, 1, 0)
            finally:
                browser.close()
    except PlaywrightError as exc:
        if "Executable doesn't exist" in str(exc) or "playwright install" in str(exc).lower():
            pytest.skip(str(exc))
        raise
