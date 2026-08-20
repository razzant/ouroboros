// Shared fixtures for the harness-accounts test family.
//
// `harness_accounts.test.js` grew past the module size gate and was split into
// four sibling `*.test.js` files (v7 stream W). `fakeResponse` is the only
// fixture used by more than one of them, so it lives here; every other helper
// moved with the single section that uses it. The function text below is
// byte-identical to its text in the pre-split file — the `export` is a separate
// statement so the declaration itself is unchanged.

function fakeResponse(status, body) {
    return { ok: status >= 200 && status < 300, status, json: async () => body };
}

export { fakeResponse };
