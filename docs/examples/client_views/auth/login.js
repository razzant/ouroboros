const next = new URLSearchParams(location.search).get('next') || '/';
document.querySelector('#next').value = next;
document.querySelector('#recovery').href = '/auth/login?builtin=1&next=' + encodeURIComponent(next);

document.querySelector('#login').addEventListener('submit', async event => {
    event.preventDefault();
    const button = event.currentTarget.querySelector('button');
    button.disabled = true;
    try {
        const response = await fetch('/auth/login', {
            method: 'POST', headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({password: document.querySelector('#password').value, next}),
        });
        const receipt = await response.json();
        if (!response.ok || receipt.ok !== true) {
            document.querySelector('#error').textContent = receipt.error || `Sign-in returned HTTP ${response.status}`;
            return;
        }
        // Use the core's sanitized destination, never the query string directly.
        location.assign(receipt.next);
    } catch (error) { document.querySelector('#error').textContent = error.message; }
    finally { button.disabled = false; }
});
