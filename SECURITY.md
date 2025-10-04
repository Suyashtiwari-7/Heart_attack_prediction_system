# Security Summary

- Passwords: use bcrypt (via passlib).
- Transport: serve API over HTTPS (TLS). For local dev, use reverse proxy with TLS.
- Database: enable Oracle TDE and use secure user accounts. Use parameterized queries.
- Input Validation: use Pydantic models to validate inputs (see app/routes.py).
- Logging: redact sensitive fields before logging (do NOT log full PII).
