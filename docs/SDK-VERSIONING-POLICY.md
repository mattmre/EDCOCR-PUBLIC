# SDK Versioning Policy

This document defines the versioning strategy, compatibility guarantees, deprecation
lifecycle, and release process for the EDCOCR SDK clients (Python and TypeScript).

---

## 1. Versioning Strategy

### Single Source of Truth

All SDK versions are locked to the core product version defined in `version.py`.
The script `scripts/check_version_consistency.py` enforces that every version
source file contains the same value.

### Version Format

SDKs follow **Semantic Versioning 2.0** (`MAJOR.MINOR.PATCH`), matching the
product version exactly:

| Component | Meaning |
|-----------|---------|
| MAJOR | Breaking API changes (endpoint removal, response schema change) |
| MINOR | New features, new endpoints, new optional parameters |
| PATCH | Bug fixes, documentation, internal improvements |

### Version Sources

The following files must always agree:

| File | Field |
|------|-------|
| `version.py` | `__version__` |
| `sdk/python/pyproject.toml` | `project.version` |
| `sdk/python/src/edcocr_sdk/__init__.py` | `__version__` |
| `sdk/python/src/edcocr_sdk/client.py` | `SDK_VERSION` |
| `sdk/typescript/package.json` | `version` |
| `sdk/typescript/src/client.ts` | `SDK_VERSION` |
| `helm/ocr-local/Chart.yaml` | `appVersion` |

---

## 2. Compatibility Matrix

| SDK Version | Min API Version | Max API Version | Python Versions | Node.js Versions | Status |
|-------------|-----------------|-----------------|-----------------|------------------|--------|
| 1.2.x | 1.0.0 | 1.2.x | 3.10+ | 18+ | **Active** |
| 1.1.x | 1.0.0 | 1.1.x | 3.10+ | 18+ | Maintenance |
| 1.0.x | 1.0.0 | 1.0.x | 3.10+ | 18+ | End-of-life |

**Active**: Full support -- new features, bug fixes, security patches.
**Maintenance**: Security fixes only. No new features.
**End-of-life**: No support. Users should upgrade.

### API Version Header

Every API response includes an `X-API-Version` header (produced by
`api/versioning.py:get_version_header`). SDKs should check this header to
detect version drift between client and server.

---

## 3. API Stability Tiers

The API surface is organized into three stability tiers, defined in
`api/versioning.py`:

| Tier | Guarantee | SDK Coverage |
|------|-----------|-------------|
| **Stable** | Full backward compatibility within MAJOR version | Fully typed in both SDKs |
| **Beta** | May change in any MINOR release with notice | Available but marked beta |
| **Experimental** | May change or be removed at any time | Not included in SDKs |

SDKs expose only Stable and Beta endpoints. Experimental endpoints are
accessible only through raw HTTP calls.

---

## 4. Deprecation Lifecycle

### Timeline

1. **Deprecation notice**: Announced at least 1 MINOR version before removal.
2. **Runtime warning**: Deprecated APIs emit warnings when called.
3. **Removal**: Only happens on MAJOR version bumps.

### Python SDK Deprecation

Deprecated methods use `warnings.warn` with `DeprecationWarning`:

```python
import warnings

def old_method(self):
    warnings.warn(
        "old_method is deprecated; use new_method instead. "
        "Will be removed in 2.0.0.",
        DeprecationWarning,
        stacklevel=2)
    return self.new_method
```

### TypeScript SDK Deprecation

Deprecated methods use JSDoc `@deprecated` tags and `console.warn`:

```typescript
/**
 * @deprecated Use {@link newMethod} instead. Will be removed in 2.0.0.
 */
oldMethod: Promise<Result> {
  console.warn('oldMethod is deprecated; use newMethod instead.');
  return this.newMethod;
}
```

### Deprecation Tracking

The `api/versioning.py` `EndpointRecord` dataclass includes `deprecated` and
`deprecated_in` fields. When an endpoint is deprecated, set:

```python
EndpointRecord(
    "GET", "/api/v1/old-endpoint", "old_endpoint",
    StabilityTier.STABLE,
    deprecated=True,
    deprecated_in="1.3.0")
```

---

## 5. Breaking Change Policy

### What Counts as Breaking

| Change | Breaking? |
|--------|-----------|
| Removing an endpoint | Yes |
| Changing a response field type | Yes |
| Removing a response field | Yes |
| Removing a required parameter | Yes |
| Changing a URL path | Yes |
| Adding auth to a previously unauthenticated endpoint | Yes |
| Adding a new endpoint | No |
| Adding an optional parameter | No |
| Adding a new response field | No |
| Adding a new enum value | No |
| Relaxing a constraint (e.g., longer max length) | No |

### Compatibility Checking

Use `api/versioning.py:check_backward_compatibility` to programmatically
verify that a proposed API surface change is backward compatible with the
previous version's stable endpoints.

### Migration Guides

When a MAJOR version introduces breaking changes, a migration guide must be
published at `docs/sdk-migration-MAJOR.md` (e.g., `docs/sdk-migration-2.md`).

---

## 6. SDK Release Process

### Pre-Release Checklist

1. All version sources updated (enforced by `scripts/check_version_consistency.py`).
2. `scripts/validate_sdk_policy.py` passes with exit code 0.
3. CI is green on the release branch.
4. CHANGELOG updated with SDK-relevant changes.

### Publishing Workflow

Both SDKs are published via `.github/workflows/sdk-publish.yml`:

| SDK | Tag Pattern | Registry | Auth Secret |
|-----|------------|----------|-------------|
| Python (`edcocr-sdk`) | `sdk-python-v*` | PyPI | `PYPI_TOKEN` |
| TypeScript (`@edcocr/sdk`) | `sdk-ts-v*` | npm | `NPM_TOKEN` |

### Release Steps

1. Bump version in `version.py` and all version sources.
2. Merge to `main` with passing CI.
3. Tag the release commit:
   ```bash
   git tag sdk-python-v1.2.0
   git tag sdk-ts-v1.2.0
   git push origin sdk-python-v1.2.0 sdk-ts-v1.2.0
   ```
4. The `sdk-publish.yml` workflow triggers automatically:
   - Runs `check_version_consistency.py` (version-check job).
   - Builds, tests, and publishes each SDK to its registry.
5. Verify published packages:
   ```bash
   pip install edcocr-sdk==1.2.0
   npm info @edcocr/sdk@1.2.0
   ```

### Manual / Dry-Run Publish

Use the `workflow_dispatch` trigger for testing without a tag:
```
gh workflow run sdk-publish.yml
```

---

## 7. SDK Support Lifecycle

| Release Relationship | Support Level |
|---------------------|--------------|
| Current MAJOR.MINOR | Full support (features, fixes, security) |
| Previous MINOR | Security fixes only |
| Two or more MINOR versions behind | End-of-life, no support |

### End-of-Life Process

1. Announce EOL in the CHANGELOG and SDK README.
2. Final security patch release if needed.
3. Mark the version row as "End-of-life" in the compatibility matrix above.
4. PyPI/npm packages remain available but receive no updates.

---

## 8. Validation

Run the SDK policy validation script to verify compliance:

```bash
# Text output
python scripts/validate_sdk_policy.py

# JSON output
python scripts/validate_sdk_policy.py --json

# Write markdown report
python scripts/validate_sdk_policy.py --report docs/reports/sdk-policy-validation.md
```

The script checks:
- Version alignment across all sources
- Python SDK metadata (pyproject.toml fields)
- TypeScript SDK metadata (package.json fields)
- API stability contract presence (api/versioning.py)
- Deprecation marker consistency
- Policy document existence
- CI publish workflow presence

---

## References

- `version.py` -- Single source of truth for product version
- `api/versioning.py` -- API stability tiers, endpoint registry, compatibility checking
- `scripts/check_version_consistency.py` -- Version alignment enforcement
- `scripts/validate_sdk_policy.py` -- SDK policy compliance validation
- `.github/workflows/sdk-publish.yml` -- CI/CD publish workflow
- `sdk/python/` -- Python SDK package
- `sdk/typescript/` -- TypeScript SDK package

---

*Last Updated: 2026-05-20 | Applies to: v1.2.0+*
