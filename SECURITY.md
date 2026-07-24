# Security Policy

## Supported Versions

Security fixes are maintained for the latest published release and the current `ouroboros` branch. Older tags preserve project history but do not receive backported fixes.

## Report a Vulnerability Privately

Do not open a public issue or discussion for a suspected vulnerability. Send a private message to [Anton Razzhigaev on Telegram](https://t.me/abstractDL) with a concise description and a safe way to continue the conversation.

Please include:

- the affected version, commit, or branch;
- the platform and installation method;
- the security boundary you believe can be crossed;
- minimal reproduction steps and the observed impact;
- any mitigation you have already tested.

Do not send API keys, access tokens, personal data, or live exploit payloads in the first message. Redact secrets and coordinate a safer transfer method if additional evidence is necessary.

## Relevant Security Boundaries

Reports are especially useful when they involve:

- credential exposure or unintended secret persistence;
- escape from filesystem, runtime-mode, task, or extension boundaries;
- unauthorized file, network, process, browser, model, or tool access;
- bypass of review, grant, provenance, update, or release-integrity checks;
- dependency, marketplace, companion-process, or other supply-chain risks.

Ordinary bugs, support questions, and product ideas belong in the repository's issue forms or Discussions. Security reports will be assessed against the current architecture and coordinated privately until disclosure is safe.
