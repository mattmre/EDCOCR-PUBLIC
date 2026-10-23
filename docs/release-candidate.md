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
  These geometric checks do not certify glyph-to-Unicode semantic identity.
- IAM mode uses boto3's supported provider chain, including ambient environment
  credentials, IMDS, ECS, IRSA, role and SSO profiles. Workers retain scoped
  presigned access; downloads bind the configured bucket and exact artifact key.
- Helm rejects bundled broker/Redis/database client-only TLS without compatible
  listeners; explicit external TLS endpoints retain their supported configuration.
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
- The TypeScript SDK retains Node.js 20.19+ (20.x) or 22.12+. Python SDK
  commands `edcocr` / `edcocr-mcp` and `EDCOCR_API_URL` / `EDCOCR_API_KEY`
  remain canonical. Python3.9 installs/builds select Pydantic2.13.5 and
  Hatchling1.27; Python3.10+ selects the refreshed dependency branches.
- Reviewed PDF/API/auth/ORM/coordinator/frontend/tooling dependencies were
  refreshed without migrating the qualified Paddle/NumPy/OpenCV/protobuf bundle.
  The imgaug0.4.0 inventory advisory remains upstream-unfixed; its unused unsafe
  background API is disabled. This is mitigation, not a scanner-clean inventory.
- Litigation hold preserves raster bytes and reservations without allowing held
  cleanup rows to starve active recovery. Retirement errors are isolated.
- Jobs applies saved Display preferences to its first request and polling cadence.
  These preferences govern Jobs only; Dashboard, Batches and Review explicitly
  retain their independent behavior. Jobs does not skip polling on hidden tabs.
- The refreshed frontend runtime launches Next directly with Node26.10.0.
  Global npm/npx are removed before copying the donor rootfs into a scratch
  stage, so their bytes are absent from shipping layers. The retained Node
  donor/build cache still contain them; this is not OS/Node advisory clearance.
- Release preparation supports ASCII SemVer prerelease/build identifiers and
  updates `ocr_local/config/version.py`, not the root forwarding shim. Existing
  candidate changelog headings are preserved. Writes are atomic per-file only.
- Root CPython3.10 has a Linux ARM64 fasttext source-build/inference/HTTP and real
  CPU OCR canary. This does not qualify every language, fallback/tooling surface,
  processing-provider parent image or optional translation model. Legacy UIE remains
  separate and unqualified; its requirements are not installed in default images.
- Standalone API and coordinator purges remain independent. Local fences and
  retirement recovery are not global SQL/object/NFS/Redis/broker saga atomicity;
  cross-plane identity/lifecycle/completion authority and intended-target proof
  require separate engineering/operator acceptance.

## Acceptance scope

Release acceptance includes clean CPU builds, real two-page OCR with PDF/text
and custody verification, API restart, coordinator broker recovery, storage
restore/retry, delayed-upload cleanup, PostgreSQL migration/rollback and tenant
isolation, SDK packages, and local regression results. The final candidate's
checksums and validation record must identify the exact public source used.
Historical design guides describe optional surfaces; they are not acceptance
results for a target or evidence that every optional feature is complete.

October8 acceptance records remain historical: they do not qualify the
October9 dependency/source refresh. New source/image/package identities and
runtime records must bind the refreshed candidate before publication.

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
