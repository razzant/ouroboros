import assert from 'node:assert/strict';
import test from 'node:test';

// The DOM-free helper; the rendered Activity page is exercised by the browser lane.
globalThis.document = globalThis.document || { createElement: () => ({}) };
const { scheduleInstantHtml } = await import('../modules/activity.js');

function expectedTime(iso, zone, includeYear = false) {
    // Native Intl supplies the date's and the time's grammar; the instant, its zone and "<date> at <time>" (Storage's
    // sentence shape) are the contract.
    const instant = new Date(iso);
    const date = instant.toLocaleDateString('en-GB', { day: 'numeric', month: 'short', ...(includeYear ? { year: 'numeric' } : {}), timeZone: zone });
    const time = instant.toLocaleTimeString('en-GB', { hour: '2-digit', minute: '2-digit', timeZone: zone });
    return `<time datetime="${iso}" title="${iso}">${date} at ${time}</time>`;
}

test('a stored UTC instant renders in the one zone it is given, the exact UTC instant kept in its attributes', () => {
    assert.equal(
        scheduleInstantHtml('2027-01-15T09:00:00+00:00', { timeZone: 'Asia/Tokyo' }),
        expectedTime('2027-01-15T09:00:00Z', 'Asia/Tokyo'),
    );
    assert.equal(scheduleInstantHtml('2027-01-15T09:00:00Z', { timeZone: 'Asia/Tokyo' }),
        '<time datetime="2027-01-15T09:00:00Z" title="2027-01-15T09:00:00Z">15 Jan at 18:00</time>');
    assert.equal(
        scheduleInstantHtml('2027-07-15T09:00:00Z', { timeZone: 'America/New_York' }),
        expectedTime('2027-07-15T09:00:00Z', 'America/New_York'),
    );
    assert.doesNotMatch(scheduleInstantHtml('2027-07-15T09:00:00Z', { timeZone: 'America/New_York' }), /UTC|GMT|EDT/,
        'one zone, never a second one beside it');
});

test('an unparseable or absent value is shown raw, never guessed', () => {
    assert.equal(scheduleInstantHtml('tomorrow <9am>'), 'tomorrow &lt;9am&gt;');
    assert.equal(scheduleInstantHtml(''), '');
    assert.equal(scheduleInstantHtml(undefined), '');
});

test('a hard deadline includes its year in its zone', () => {
    // January 1 in UTC is still the preceding local year in New York.
    for (const timeZone of ['Asia/Tokyo', 'America/New_York']) {
        assert.equal(
            scheduleInstantHtml('2099-01-01T00:00:00Z', { timeZone, includeYear: true }),
            expectedTime('2099-01-01T00:00:00Z', timeZone, true),
        );
    }
    assert.match(scheduleInstantHtml('2099-01-01T00:00:00Z', { timeZone: 'America/New_York', includeYear: true }), />31 Dec 2098 at 19:00</);
});
