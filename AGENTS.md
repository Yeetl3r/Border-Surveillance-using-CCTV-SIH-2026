# Agent Directives for Jules

## Constraints & Permissions
1. Never push commits directly to remote branches or publish PRs without explicit confirmation.
2. For all requested scans or features, output a complete plan followed by standard Git unified diffs (`git diff` format).
3. Always wait for human confirmation before writing to disk or creating isolated branches.

## Code Quality & Security Standards
- Prioritize OWASP Top 10 vulnerabilities, insecure dependency usage, and secret leakage.
- Suggest architectural and performance improvements when touching existing endpoints or modules.
- Ensure all suggested code includes unit tests matching the project's testing framework.
