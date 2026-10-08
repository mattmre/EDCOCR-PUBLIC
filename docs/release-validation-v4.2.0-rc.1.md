# v4.2.0-rc.1 local validation

Candidate preparation date: 2026-10-08. Nothing in this record certifies an
external production deployment or indicates that registry artifacts have been
published. Hosted CI is a convenience; acceptance below ran locally.

## Executed checks

| Surface | Observed result | Limit |
|---|---|---|
| Public CPU SOLO + API images | Clean build; installed Python SDK submitted a two-page fixture, retrieved searchable PDF and text, and verified a five-event custody chain | Linux ARM64, English fixture, one host |
| API lifecycle | Unauthenticated job request returned 401; completed job and artifact hashes survived restart; watched input directory remained empty | Owned API key and local filesystem |
| Public coordinator image | PostgreSQL 16 upgrade preserves jobs; empty migration reversal/reapply works; populated retirement reversal is refused; tenant/control-plane RLS checks pass | Owned single PostgreSQL instance |
| Object cleanup | Real MinIO late upload after an empty purge sweep is removed by recovery; neighboring prefix survives | Current objects only; no AWS or retained-version erasure claim |
| Coordinator regression | 998 passed, 6 skipped, 23 subtests passed | Exported coordinator suite; skips remain disclosed |
| Frontend | 399 tests passed; lint, type checking and production build passed | Simulated browser tests; no customer browser deployment |
| TypeScript SDK | 32 tests passed; compile and tarball packaging passed | Public package identity and prerelease version |
| Python SDK | 82 package tests passed; wheel/sdist build and metadata validation passed; original CLI/MCP names and environment variables retained | Wheel runtime 3.9+; source build 3.10+ |
| Independent review | Export import/COPY closure, installed SDK compatibility, SOLO rendering, prerelease tagging and actionlint passed | Fresh reviewer on shared host/model family; no production certification |

## Regression accounting

The first broad public run had 13,290 passes, 72 failures, 12 setup errors and 33
skips. It was not green. Failures involved public prerelease assertions,
workflow configuration, and historical documentation contracts. Repairs were
replayed through affected modules: 626 passed with one remaining version
assertion, followed by 35 version tests passing after that correction. The
CLI/MCP and workflow compatibility subset separately passed 31 tests. A final
combined rerun of all affected configuration, documentation, version and SDK
modules passed 642 tests.

Twenty-two checks for private quarterly strategy and migration-research
headings were removed from the public export with their non-public documents.
Runtime and configuration tests remain. This disposition is not presented as
a second full-suite pass. General warnings include upstream deprecations,
optional tooling skips, and simulated-browser navigation messages.

## Dependency disposition

The shipped coordinator Python inventory reports no known advisories. The
frontend production dependency audit and TypeScript SDK audit report zero.
Frontend development dependencies still have advisories and are not included
in the production dependency claim.

The CPU OCR image reports the imgaug 0.4.0 advisory CVE-2026-31235 (also
PYSEC-2026-356). Its reviewed build hardener removes the affected background
augmentation implementation; invoking it in the shipped image raises the
expected disabled-capability error. It is mitigated, not scanner-clean.

## Unqualified targets

Real AWS role/IAM and bucket policy, physical multi-host/GPU execution,
client/server mutual TLS identity, real NFS durability and permissions, and
coordinated replicated database/object restoration still need their actual
targets. Source review and these owned fixtures validate approach and bounded
behavior only. The standalone API and coordinator retain separate job stores.
