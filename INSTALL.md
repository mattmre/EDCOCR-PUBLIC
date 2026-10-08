# Installing v4.2.0-rc.1

Start with the [CPU quickstart](README.md#cpu-quickstart). Use Docker Compose
2.24.4 or newer. Shipping Python images use **Python 3.12 and Debian 12**;
older Python instructions in historical guides do not qualify this candidate.
Plan roughly 16–32 GB RAM and at least 50 GB free build space for local CPU
acceptance; actual workload capacity must be measured. The default model set
can take substantial time and disk space to download.

## Single-host API

Run one API process against each persistent API database. The process lock
rejects a second owner. Keep database, source staging, and output artifacts
on persistent volumes. On Linux, the containers run as UID/GID 1000; provision
writable directories before starting. The API binds only to loopback by default.
Configure authentication and trusted HTTPS ingress for remote clients.

The CPU overlay selects the CPU Dockerfile and removes NVIDIA reservations.
`SOLO_ENABLE_GPU=false` alone has no effect. `SOLO_IMAGE` can select a custom
worker tag; Compose builds that service before using it as the API base.
`SOLO_API_BASE_IMAGE` is an advanced override requiring the same runtime
contract, including `/usr/bin/tini` and matching source.

The batch process scans `ocr_source`, completes its batch, and can be restarted
for another scan. API jobs run through their own child pipelines and stage
uploads in `ocr_api_source`. These folders must remain separate.

`SOLO_MAX_CONCURRENT_JOBS` defaults to 1. The API's CPU worker, extractor,
compressor, and queue defaults are deliberately bounded; the `SOLO_API_*`
variables adjust them. CPU batch defaults are one OCR worker and two extractors.
`SOLO_MODEL_PRELOAD_LANGUAGES=en` can build an English-only CPU acceptance image;
that image does not qualify omitted languages.

## Distributed workers and optional targets

The coordinator deployment lives in `coordinator/`; Kubernetes templates live
in `helm/ocr-local`. Submit distributed work to the **coordinator API**. Scaling
workers does not move standalone API jobs into the coordinator or unify their
browser views. Build images locally with `scripts/with_runtime_source.sh` and
the corresponding `coordinator` or `worker` profile. Set image repositories and
immutable digests explicitly for your registry.

GPU images require qualified CUDA/Paddle wheels and target hardware. The stock
GPU image is for qualified amd64 deployments; an ARM64 GPU host needs a separately
qualified image. Server TLS, service-mesh annotation, and federation mutual TLS
are different capabilities. `federation.mtls.enabled=true` is rejected because
the chart does not implement end-to-end mutual authentication for that mode.

S3 defaults to explicit static authentication. `S3_AUTH_MODE=iam` uses the SDK
credential provider chain. In Helm it requires a distinct operator-managed
control-plane service account, presigned worker access, and exact allowed
presigned hosts. The chart does not provision cloud IAM or bucket policies.
Real NFS requires the intended shared POSIX export and UID permissions. See
[candidate scope](docs/release-candidate.md) for qualification limits.

## Upgrade and rollback

1. Stop admission at the client/ingress boundary. Drain jobs and background
   operations. Global API health drain fields must show no writers; tenant-scoped
   health is insufficient for whole-process replacement.
2. Save the current image digests and take a recoverable backup of database,
   source, output, and object storage at a consistent boundary. Test restoration.
3. Stop the previous API owner. Build the reviewed source with its fingerprint,
   apply coordinator migrations, and start the replacement. Verify health,
   existing jobs/artifacts, and a real new document before reopening admission.
4. If rollback is necessary, retain additive schema and cleanup records. Restore
   the matching database/storage backup when an older binary is incompatible.
   Do not drop migration 0032 while retired S3 jobs need cleanup: reversal refuses
   those records, and older binaries do not service their cleanup queue.

A SQL-only backup does not prove coordinated recovery with object storage.
PostgreSQL TLS verification and custom schema handling are supported by the
backup validation tool; replicated recovery still needs its actual target.

## SDKs

The public package names remain `edcocr-sdk` and `@edcocr/sdk`. Python imports
remain `edcocr_sdk.EDCOCRClient` and `AsyncEDCOCRClient`. Build/install the candidate
from `sdk/python` or `sdk/typescript`; package-registry publication is a separate
release action. The Python wheel uses the normalized prerelease version
`4.2.0rc1`; runtime/JavaScript version strings use `4.2.0-rc.1`.

Python SDK wheels support Python 3.9+. Source/sdist SDK installation needs
Python 3.10+ for the reviewed build backend; the application uses Python 3.12.
