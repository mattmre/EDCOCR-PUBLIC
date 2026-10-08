# v4.2.0-rc.1 scope and limits

This candidate updates the existing v4.1.0 public release line. It targets a
single-host standalone API and the separate coordinator API with scalable OCR
workers. These APIs retain independent job stores. No production, physical-farm,
GPU, cloud IAM, enforced mesh, real NFS, or replicated recovery certification is
implied by source review or a local test pass.

## Changes requiring operator attention

- A persistent process lock prevents concurrent standalone API owners. Drain
  active work before replacement; timed-out background writers retain ownership.
- CPU SOLO builds have an explicit overlay and API base-image dependency.
  API uploads and batch inputs use different host directories.
- Native-PDF fast processing requires completeness checks; incomplete content
  falls back to OCR while retaining source evidence. Latin lines remain
  searchable on pages whose script-specific font lacks Latin glyphs.
- IAM mode uses refreshable SDK credentials on the control plane. Workers use
  scoped presigned access and exact hostname allowlists; explicit static and
  ambient credential exposure is rejected in incompatible configurations.
- Unsupported federation mutual TLS is refused during Helm rendering.
- Backup validation preserves PostgreSQL TLS/session parameters and custom
  schemas. NFS configuration uses the actual mounted path.
- Tenant purge and retention preserve staged cleanup reservations. Deleted S3
  jobs retain independent prefix cleanup records for late uploads. This is
  eventual deletion of current objects, not immediate erasure or deletion of
  historical versions and locked copies. Beat and credentialed coordinator
  workers must keep running; monitor overdue/error retirement records.
- Migration 0032 retains that cleanup authority across Job deletion and refuses
  schema rollback while records exist. It cannot reconstruct authority already
  lost by an older release. Retired job UUIDs must never be reused.
- The TypeScript SDK requires Node.js 20.19+ (20.x) or 22.12+. Python SDK
  console commands `edcocr` and `edcocr-mcp`, and `EDCOCR_API_URL` /
  `EDCOCR_API_KEY`, remain available. SDK wheels support Python 3.9+;
  source/sdist builds require Python 3.10+ for the reviewed build backend.
- Frontend and TypeScript SDK dependencies were updated. The OCR dependency
  inventory still reports CVE-2026-31235 in imgaug 0.4.0; the build disables its
  unused affected BackgroundAugmenter and refuses unreviewed upstream bytes.
  Treat it as mitigated, not scanner-clean. Synchronous inference remains tested.

## Acceptance scope

Release acceptance includes clean CPU builds, real two-page OCR with PDF/text
and custody verification, API restart, coordinator broker recovery, storage
restore/retry, delayed-upload cleanup, PostgreSQL migration/rollback and tenant
isolation, SDK packages, and local regression results. The final candidate's
checksums and validation record must identify the exact public source used.
Historical design guides describe optional surfaces; they are not acceptance
results for a target or evidence that every optional feature is complete.

Review AWS role/bucket policy, exact GPU/driver combinations, distinct physical
hosts, enforced client/server certificate identity, NFS durability/permissions,
and coordinated replicated database/object restore against their intended use.
Source review and owned local fixtures cannot establish those target properties.

## Publication

The public history and SDK identities are preserved. The candidate must undergo
sanitization review, local acceptance, and owner sign-off before publication.
Registry packages and container images require separate publication checks;
source availability alone does not mean those artifacts are published.

The [local validation record](release-validation-v4.2.0-rc.1.md) distinguishes
executed checks, repaired regression failures, skips and target limitations.
