"""Temporary, fixture-only frames; copy only into the detached UI worktree."""
import os
from pathlib import Path
from urllib.parse import urlparse

import pytest
from playwright.sync_api import expect

from tests.test_design_skill_assets_browser import assert_nothing_fetched, browser, open_page, sample_page, wait_shown
from tests.test_subscription_setup_browser import subscription_ui as subscription_ui


def capture(locator, name):
    target = Path(os.environ['OUROBOROS_UI_EVIDENCE_DIR'])
    target.mkdir(parents=True, exist_ok=True)
    png = locator.screenshot(animations='disabled')
    assert 0 < len(png) <= 1024 * 1024
    (target / name).write_bytes(png)


@pytest.mark.ui_browser
@pytest.mark.parametrize('width,height', [(1280, 900), (390, 844)], ids=['desktop', 'phone'])
def test_costs_compares_zero_with_no_hourly_rate(subscription_ui, width, height):
    ui = subscription_ui
    page = ui['new_page'](viewport={'width': width, 'height': height}, has_touch=width < 600,
                          locale='en-US', timezone_id='UTC', color_scheme='light', service_workers='block')
    page.set_default_timeout(15000)
    origin = urlparse(ui['url']).netloc
    page.route('**/*', lambda route: route.fallback() if urlparse(route.request.url).netloc == origin else route.abort())
    page.add_init_script("localStorage.setItem('ouroboros.theme', 'light')")
    models = [{'model': f'fixture/hour-{name}', 'calls': 1, 'cost': 0.5, 'write_price_lifetime': '1h',
               'write_price_in_reads': ratio, 'tokens': {'cache_write': 10, 'cache_read': 20}, 'by_class': split}
              for name, ratio, split in [('zero', 0, {'cache_read': 0.5}), ('unlisted', None, None)]]
    page.route('**/api/cost-breakdown?period=24h', lambda route: route.fulfill(json={'period': {
        'available': True, 'period': '24h', 'models': models, 'cache_writes': []}}))
    page.goto(ui['url'] + '/#dashboard', wait_until='domcontentloaded')
    page.locator('[data-dashboard-tab="costs"]').click()
    card = page.locator('#costs-period')
    expect(card.locator('.costs-period-model')).to_have_count(2)
    capture(card, f'hourly-{width}x{height}.png')
    zero = card.locator('.costs-period-model', has_text='fixture/hour-zero')
    missing = card.locator('.costs-period-model', has_text='fixture/hour-unlisted')
    zero_words = 'One token written for 1 hour costs 0 read tokens at this model’s base price.'
    missing_words = 'The base price of a write for 1 hour cannot be expressed in read tokens.'
    expect(zero).to_contain_text(zero_words)
    expect(zero).not_to_contain_text(missing_words)
    expect(missing).to_contain_text(missing_words)
    expect(missing).not_to_contain_text(zero_words)
    expect(missing.locator('.costs-period-sum')).to_contain_text('$0.50')
    assert page.evaluate('document.documentElement.scrollWidth <= window.innerWidth')
    assert ui['posts'] == []


@pytest.mark.browser
@pytest.mark.parametrize('engine', ['chromium', 'webkit'])
@pytest.mark.parametrize('width', [900, 390])
def test_comparison_template_frame(engine, width, tmp_path):
    # Existing helper instantiates the shipped skill asset with synthetic srcdoc scenes.
    with browser(engine) as instance:
        page = open_page(instance, sample_page(tmp_path).as_uri(), width=width)
        wait_shown(page, 'now', 'draft')
        page.locator('[data-scene="long"]').click()
        wait_shown(page, 'now', 'long')
        page.keyboard.press('3')
        wait_shown(page, 'b', 'long')
        capture(page.locator('body'), f'compare-{engine}-{width}.png')
        assert_nothing_fetched(page)
