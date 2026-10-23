# v4.2.0-rc.1 local validation

## 2026-10-09 refresh — unpublished

The current candidate is an exported working tree, not a newly committed public
release. Public HEAD remains `5bc0e81493c6294b47d2f7f24ea92a769a85b8c1`.
The release packet binds the complete source archive, installed packages and
image identities separately. No push, tag, merge, registry publication or
deployment was performed; owner approval remains required.

| Surface | Observed refreshed result | Limit |
|---|---|---|
| Baked CPU OCR | Two searchable pages, 351261-byte PDF, 894 text characters and five custody events | Linux ARM64, English fixture, non-root runtime; no CUDA or all-language claim |
| Coordinator lab | Real OCR, broker-outage durable recovery, idempotency/conflict, tenant isolation, bundle/PDF/custody, separate MinIO restore and four retention failure boundaries/retry passed | One host with PostgreSQL/RabbitMQ/MinIO; not actual AWS or replicated production restore |
| Baked public API | Completed two-page job, PDF/text/custody retrieval, HTTP401 denial and identical artifact hashes after restart | Owned nonproduction API and local filesystem |
| Public coordinator jobs regression | 913 passed, six skipped | Explicit `coordinator/jobs/tests` selection: 919 collected; not a full coordinator comparison with the historical 998 run; S3-target skips remain |
| Public API/forensic regression | 487 passed | Affected modules, not the full public suite |
| Public Helm/dependency regression | 160 passed | Five Helm modules plus dependency module; not live TLS/certificate qualification |
| Public Python SDK | 142 passed; wheel/sdist installed on Python3.9.25 ARM64; canonical CLI and MCP exercised against the baked public API | Pydantic2.13.5/Hatchling1.27 markers preserve Python3.9 source builds |
| Public TypeScript SDK | 31 passed; type check and packaged status/pagination/PDF/text/auth-denial exercised | One incidental version-shape assertion removed; runtime API unchanged |
| Changed Python lint | Ruff0.16.10 passed on 21 files | Pinned pre-commit Ruff0.8.6 was not executed |
| Frontend refresh | Private refreshed source:399 tests, type/build, real production Jobs browser/network proof | Not represented as a fresh full public frontend or customer deployment pass |
| Historical review | Scoped review findings repaired; numeric reviewer resumed with implementer summaries, conservatively capped at70 | Not an independent whole-archive approval; later message sanitization separately runtime-qualified |

Root image: `sha256:5317908dcfb280d02abc15305ab9711c294e4fa56dafff6926f3cc44dc57057b`,
source profile `810a67426058e18e3ee0fc2ea0a87a08a9c75f24b8ed50d676c1c9dc61194664`.
Coordinator image: `sha256:ed1c829fb254828f3f67af099d10187351f3c3e71ae266b55557589dd295d032`,
source profile `7e4b53ead1f7ccb80bf75ab4fb30c082cf298b44fb624f95905bd1394b23b006`.
API image: `sha256:e283351c81b65afc072b5d7026532aa644f50bfdacd15225ab01148a96ae7a38`.
Image revision labels name historical public HEAD; runtime-source hashes bind
the refreshed bytes. They must not be presented as a clean commit of that HEAD.
Root/API were rebuilt after narrowly scoped public diagnostic-message sanitization.
Fresh baked CPU OCR and baked API retrieval, installed SDK/CLI, restart and
separate-container bind-mount persistence passed; wrong-key HTTP401 verified.
Installed Python inventories exactly match the earlier reviewed images.
Coordinator runtime profile is unchanged; older reviewed root/API evidence remains historical.

The actual coordinator Python inventory has no known advisory in the recorded
scan. Root/API retain one upstream-unfixed imgaug advisory duplicated by the
scanner; the baked non-root runtime denies the vulnerable background constructor
and preserves synchronous identity augmentation. This is mitigation, not a
scanner-clean claim. Debian SBOMs are inventories, not OS-advisory clearance.
The earlier dependency-runner scan skipped `en-core-web-sm3.8.0`; it does not
qualify model packages or imply that model is installed in the final images.

Incidental numeric-version assertions were deleted, not re-pinned to the
candidate. Native geometric coverage does not certify glyph/Unicode semantic
identity or a hard MuPDF watchdog. Optional processing overlays still require
operator images, selectors and endpoints; candidate example tags are not
publication or production qualification. Added collectors deliberately retain
isolated-lab safety allowlists; they are not generic destructive-operation tools.

## Additional local qualification — still unpublished

These runs supplement, not replace, historical receipts. Source remains a dirty
working tree; full-source/package/image bindings and owner approval are separate.

| Surface | Newly observed result | Limit |
|---|---|---|
| Current-image coordinator | PG/RabbitMQ/MinIO OCR, recovery, retirement faults and object restore passed on root5317908d/coordinatored1c829f | Single-host lab; no AWS/multihost/replicated-production claim |
| Current-root SFTP | Paramiko5 fetch/hash and wrong-host-key rejection passed | Size rejection is after transfer, not an on-wire cap |
| CPython3.10 | Linux ARM64 fasttext source-build, inference/import/HTTP and production CPU OCR passed | Not all-language/GPU/system-fallback or optimization-tool qualification |
| Full-native and original mixed PDF | Full native completed; original mixed generic coverage0.413793 rerouted before native publication and produced both phrases in final TXT/searchable PDF | Eager Celery/SQLite/local NFS; quota and broker-concurrency transitions have separate coverage |
| Current coordinator suites | Public1080 passed/6skipped/25subtests; private1290 passed/8skipped/41subtests | Two forwarding-only tests removed; reduced node set, not equal-node historical comparison |
| First integrated root attempts | Public1failed/13073passed/33skipped; private29failed/16352passed/37skipped | Public optional UIE source inventory repaired without installing it; private27 Git-context nodes subsequently passed with read-only metadata. FZ source-only assertions removed under test policy, not certified passing; raw failures retained |
| Corrected integrated root suites | Public13069 passed/33skipped/86warnings; private16373 passed/37skipped/89warnings;22subtests each; both exit0 | Frozen corrected source with read-only private Git metadata; reduced node sets. Exact378 public removed IDs and26 private first-to-corrected removals disclosed; first failures retained. Subsequent docs/inventory-only changes require explicit input-equivalence binding |
| Installer constraints |18 cases passed per tree: actual specifier admit/deny, held bundle/security boundaries, legacy UIE platform/Python gating | Installer contract data, not string spelling; actual pip dry-run on Python3.12 excludes legacy PaddleNLP; dependency versions unchanged |
| Frontend source and production surface | Private/public399 tests and typecheck; shipping Node26 image5b720cec browser/API preference/pagination/stale/auth/error behavior observed | Tests run on hostNode22; literal2-row contraction is component scope, browser used10; hidden-tab simulation is not OS backgrounding |
| Current installed public packages | Python3.9 SDK/CLI/MCP and Node26 TypeScript tarball exercised against a fresh actual API CPU job; PDF/TXT/custody,401, restart/cold recreation passed; two builds byte-identical | Same-host persistence, no whole-coordinator/GPU claim; TS has getStatus and no custody method; MCP forwarding verified in API audit |
| Current retained SDK regression | Python packaged SDK/CLI/MCP90 passed on qualified CPython3.12 runner; locked TypeScript31 passed plus typecheck on actual Node26 | Installed Python3.9 consumer proof is separate. Failed Python harness setup/empty-directory attempts and a corrected owned-staging cleanup failure remain disclosed; no removed node counted passing |
| Fresh current Python audit | Root/API held imgaug advisory persists; coordinator Python inventory has no findings | No model/OS/universal scanner-clearance claim |
| Frontend dependency audit/runtime transition | Public direct-Node image3066dc94 bindsc9878f4d; actual startup/nonroot/shutdown, saved preferences,60020ms poll/reset30007ms, offsets and401/403 verified; fresh installed26/nondev55 manifest queries zero findings | Shipping layers exclude npm; retained donor/build cache still contain it. Full compiled .next and Node binary equal5b; stale/contraction/error paths inherited by explicit byte equivalence, not freshly replayed. Private synchronized recipe profile692897a0 is measured, not a private-image run; no OS/Node/native-library scan |
| Release helper | Canonical version owner, ASCII SemVer/prerelease, existing-heading preservation and actual owned-copy mutation verified | Real metadata unchanged; atomic writes per-file, not a multi-file rollback transaction |
| Current system OCR fallback | Real Tesseract5.3/eng preserved `CLEAR TEXT SAMPLE` in searchable PDF/TXT; default cascade succeeded after an identified pre-predictor Paddle-unavailable stimulus; both-unavailable case retained exact raster pixels as ImageOnly | One-page/one-process English fixture; injected Paddle availability fault, not a live predictor crash or full scheduler/assembler/EasyOCR/GPU proof |
| Current PDF optimizer | Real Ghostscript10.00 `/prepress` produced APPLIED only after forensic validation; mixed fixture19037→18065 bytes, mode0640 preserved; oversized visual-budget/invalid inputs preserved original hashes | Linux ARM64 ext4; APPLIED may grow files. No Windows/power-loss/all-quality/glyph-semantic or hard MuPDF time-containment claim |

The reduced regression node set is disclosed in the operator-held packet:
378 exact public removed IDs/312 families;26 exact private first-to-corrected
removals, not a full historical-private comparison. Earlier355-ID scope remains
preserved separately. Deleted tests are not passing results.
Advisory disagreement over source/wiring-test deletion remains explicit;
no legacy finalizer or offline probe is relabeled complete. Meaningful installer
security/held-bundle guards remain. Optional translation model inference is not
established by mocked transport, registry metadata or model-absence tests.
Processing-provider qualification still requires a qualified digest-bound parent;
root531 has no registry RepoDigest, and its config ID is not a manifest digest.

Exact root skip reasons remain operator-held: Windows-specific behavior,
optional Django/LayoutLM paths without root-runner setup, absent external
translation process/historical prewarm artifacts, and a known signature
heuristic false-positive calibration (100% versus85% limit). They are not
passing qualifications. The alternate qualified-image CPU ceiling is separate
from the skipped private default `ocr-gpu-local:canary` probe.

Standalone API/coordinator purges remain independent. Their local fences,
custody and cleanup retries do not provide a global SQL/object/NFS/Redis/broker
saga. Cross-plane identity/lifecycle/completion policy and intended storage
canaries require approved engineering/operator authority. All seven external
qualification debts, retained-resource disposition, eligible independent review,
exact-byte owner approval and separate release-operation authorization remain
unresolved; official score stays70. Nothing here is release clearance.

## Historical 2026-10-08 record

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
