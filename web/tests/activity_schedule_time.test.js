import assert from 'node:assert/strict';
import test from 'node:test';

// The DOM-free helper; the rendered Activity page is exercised by the browser lane.
globalThis.document = globalThis.document || { createElement: () => ({}) };
const { scheduleInstantHtml } = await import('../modules/activity.js');

function expectedTime(iso, zone, includeYear = false) {
    // Native Intl supplies locale grammar, not the instant, zone or format contract.
    const fields = { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' };
    if (includeYear) fields.year = 'numeric';
    const instant = new Date(iso);
    const local = new Intl.DateTimeFormat([], { ...fields, timeZone: zone, timeZoneName: 'short' }).format(instant);
    const utc = new Intl.DateTimeFormat([], { ...fields, timeZone: 'UTC' }).format(instant);
    const escape = text => text.replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;').replaceAll('"', '&quot;').replaceAll("'", '&#39;');
    return `<time datetime="${iso}" title="${iso}">${escape(local)} (${escape(utc)} UTC)</time>`;
}

test('a stored UTC instant renders in the viewer zone with the exact UTC beside it', () => {
    assert.equal(
        scheduleInstantHtml('2027-01-15T09:00:00+00:00', { timeZone: 'Asia/Tokyo' }),
        expectedTime('2027-01-15T09:00:00Z', 'Asia/Tokyo'),
    );
    assert.equal(
        scheduleInstantHtml('2027-07-15T09:00:00Z', { timeZone: 'America/New_York' }),
        expectedTime('2027-07-15T09:00:00Z', 'America/New_York'),
    );
});

test('an unparseable or absent value is shown raw, never guessed', () => {
    assert.equal(scheduleInstantHtml('tomorrow <9am>'), 'tomorrow &lt;9am&gt;');
    assert.equal(scheduleInstantHtml(''), '');
    assert.equal(scheduleInstantHtml(undefined), '');
});

test('a hard deadline includes its year in both viewer and UTC text', () => {
    // January 1 in UTC is still the preceding local year in New York.
    for (const timeZone of ['Asia/Tokyo', 'America/New_York']) {
        assert.equal(
            scheduleInstantHtml('2099-01-01T00:00:00Z', { timeZone, includeYear: true }),
            expectedTime('2099-01-01T00:00:00Z', timeZone, true),
        );
    }
});
