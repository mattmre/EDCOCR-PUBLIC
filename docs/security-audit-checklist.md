# Security Audit Checklist — EDCOCR

> **Last updated:** 2026-03-16
> **Scope:** Full-stack security review of EDCOCR pipeline, API, coordinator, and deployment infrastructure.

## Summary

| Severity | Count | Status |
|----------|-------|--------|
| Critical | 2 | Open |
| High | 6 | Open |
| Medium | 7 | Open |
| Passed | 30+ | ✅ |

**Critical items require immediate remediation before production deployment.**

---

## 1. Authentication & Authorization

- [x] API key transport via header (not query params)
- [x] Timing-safe key comparison (`secrets.compare_digest`)
- [x] Multi-tenant keys hashed (SHA-256)
- [x] Key generation uses `secrets.token_urlsafe`
- [x] Key revocation and expiration
- [x] RBAC with 3 roles (viewer, operator, admin)
- [x] OAuth2/OIDC JWT validation
- [ ] Legacy `OCR_API_KEY` rotation mechanism
- [ ] Brute-force protection / account lockout
- [ ] `ALLOW_UNAUTHENTICATED` should grant viewer not admin

### Notes

The `ALLOW_UNAUTHENTICATED` flag currently grants full access when set.  In
production this flag should either be removed or restricted so that
unauthenticated requests receive the **viewer** role only (read-only access).

Legacy `OCR_API_KEY` rotation is not automated.  A key rotation mechanism
(graceful dual-key window) should be implemented before multi-tenant
deployments go live.

---

## 2. Input Validation

- [x] Path traversal prevention (`path_safety.py`)
- [x] Upload size limits (`MAX_UPLOAD_SIZE_MB`)
- [x] Job/Tenant ID regex validation
- [x] Pydantic model validation
- [x] Magic bytes verification
- [x] Windows device path rejection
- [ ] Request body size limit middleware
- [ ] Upload filename sanitization audit

### Notes

FastAPI does not impose a global request body size limit by default.  A
middleware should be added to reject requests larger than a configurable
maximum (e.g., 100 MB) to prevent memory exhaustion attacks.

Uploaded filenames should be sanitized to strip path separators, null bytes,
and Unicode homoglyphs before use in any filesystem operations.

---

## 3. SSRF Protection

- [x] URL scheme enforcement
- [x] Private IP blocking
- [x] Redirect blocking
- [x] TOCTOU re-validation
- [ ] DNS rebinding mitigation (IP pinning)

### Notes

The current SSRF guard validates the resolved IP at request time but does not
pin the IP for the lifetime of the connection.  A DNS rebinding attack could
cause the initial check to pass against a public IP, then rebind to a private
IP during the actual request.  Pin the resolved IP address when opening the
connection.

---

## 4. Secrets Management

- [x] Pluggable credential backends (Vault, AWS SM)
- [x] Placeholder detection
- [**CRITICAL**] `coordinator/.env` committed with real secrets
- [ ] Webhook secrets encrypted at rest
- [ ] PII entity values encrypted at rest

### Notes

**CRITICAL:** The `coordinator/.env` file has been committed to version control
with real secrets (database passwords, Django secret key, MinIO credentials).
Immediate remediation steps:

1. Rotate all secrets present in the committed file.
2. Remove `coordinator/.env` from version control (`git rm --cached`).
3. Add `coordinator/.env` to `.gitignore`.
4. Provide `coordinator/.env.example` with placeholder values.

Webhook secrets and PII entity values should be encrypted at rest using the
credential manager's encryption backend rather than stored as plaintext in the
database.

---

## 5. Injection Prevention

- [x] SQLAlchemy ORM (no raw SQL with user input)
- [x] Django ORM for coordinator
- [x] JSON-only Celery serialization
- [x] HTML sanitization for table output

### Notes

No raw SQL queries with string formatting were found.  All database access uses
parameterized queries via ORM.  Celery is configured with `serializer='json'`
which prevents pickle deserialization attacks.

---

## 6. Transport Security

- [x] Webhook HTTPS enforcement
- [ ] TLS on API server
- [ ] TLS on inter-service communication
- [ ] HSTS headers

### Notes

The API server (`uvicorn`) runs without TLS by default.  In production, TLS
should be terminated at a reverse proxy (nginx, traefik) or enabled directly
on uvicorn with `--ssl-keyfile` / `--ssl-certfile`.

Inter-service communication (API ↔ Celery broker, API ↔ MinIO) should use TLS
in production deployments.  HSTS headers should be set by the reverse proxy.

---

## 7. Container Security

- [x] Minimal base images (`python:3.x-slim`)
- [x] Build cache cleanup
- [x] Multi-stage builds
- [x] Docker healthcheck
- [**CRITICAL**] Non-root container user
- [ ] Read-only filesystem where possible

### Notes

**CRITICAL:** The Dockerfile does not include a `USER` directive.  The
container runs as root by default, which increases the blast radius of any
container escape vulnerability.

Remediation:

```dockerfile
RUN useradd -r -s /bin/false ocr
USER ocr
```

Add a non-root user and switch to it before the `CMD` / `ENTRYPOINT`.  Ensure
volume mounts have appropriate ownership.

Where possible, run containers with `--read-only` and mount only specific
writable paths (`/tmp`, output directories).

---

## 8. Rate Limiting

- [x] Per-endpoint rate limits (slowapi)
- [x] Queue capacity limits
- [x] Per-tenant quotas
- [ ] Auth-attempt rate limiting
- [ ] `X-Forwarded-For` trust configuration

### Notes

Authentication endpoints do not have dedicated rate limits to prevent
brute-force attacks.  Add per-IP rate limiting on failed authentication
attempts (e.g., 5 failures per minute with exponential backoff).

The `X-Forwarded-For` header should only be trusted from known proxy IPs.
Without this configuration, attackers can spoof their source IP to bypass
per-IP rate limits.

---

## 9. Logging & Audit

- [x] Hash-chained audit trail
- [x] No raw API keys in logs
- [x] Structured audit fields
- [x] Audit chain verification

### Notes

The audit logging implementation is solid.  API keys are masked in all log
output.  The hash-chained audit trail provides tamper evidence.  No issues
found.

---

## 10. Dependencies

- [ ] Dependency vulnerability scanning (`pip-audit`)
- [ ] `python-multipart` upgrade to >=0.0.12
- [ ] `sqlalchemy` pin update to latest 2.0.x

### Notes

No automated dependency vulnerability scanning is configured in CI.  Add
`pip-audit` to the CI pipeline to catch known CVEs in transitive dependencies.

`python-multipart` versions prior to 0.0.12 have known vulnerabilities
(content-type parsing DoS).  Upgrade to the latest version.

`sqlalchemy` should be pinned to the latest 2.0.x release to pick up security
patches.

---

## 11. CORS

- [ ] CORSMiddleware configuration (if browser access needed)

### Notes

No CORS middleware is configured.  If browser-based clients will access the
API directly (e.g., admin dashboard), add `CORSMiddleware` with an explicit
origin allowlist.  Do not use `allow_origins=["*"]` in production.

If the API is accessed only by server-side clients, CORS configuration is not
required.

---

## Remediation Priority

| Priority | Item | Section |
|----------|------|---------|
| P0 | Remove `coordinator/.env` from VCS, rotate secrets | §4 |
| P0 | Add non-root `USER` to Dockerfiles | §7 |
| P1 | Add request body size limit middleware | §2 |
| P1 | TLS on API server | §6 |
| P1 | Auth-attempt rate limiting | §8 |
| P1 | DNS rebinding mitigation | §3 |
| P2 | `ALLOW_UNAUTHENTICATED` → viewer role | §1 |
| P2 | Legacy key rotation mechanism | §1 |
| P2 | Dependency vulnerability scanning | §10 |
| P2 | `python-multipart` upgrade | §10 |
| P3 | CORS configuration | §11 |
| P3 | Filename sanitization audit | §2 |
| P3 | Webhook/PII encryption at rest | §4 |
| P3 | Read-only container filesystem | §7 |
| P3 | HSTS headers | §6 |
