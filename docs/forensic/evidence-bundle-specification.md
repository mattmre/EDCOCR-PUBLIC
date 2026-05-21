# Evidence Bundle Specification -- Certified Translation Deliverable

**Status**: P1 deliverable (E-X-005, tracked under  + )
**Authority**: Panel 3 , Forensic / eDiscovery Expert Panel
**Jurisdictional scope**: U.S. federal courts (FRE 902(14) / FRE 901 / FRCP 26(g)); state and foreign analogues apply by reference
**Owner**: Forensic-AI Boundary Guardian (rotating chair) + Trial Counsel
**Version**: 1.0

---

## Purpose

The **evidence bundle** is the canonical courtroom deliverable for certified translations produced by the pipeline. It is the single `.tar.gz` archive a party hands to opposing counsel, a tribunal, or a neutral in response to a production request or a judicial order.

The bundle is designed to satisfy **FRE 902(14)** self-authentication for electronic records, to support **FRE 901(b)(9)** process-authentication, and to meet the proponent's burden under the **2024 FRE 702 amendments** for expert translation testimony. It ships with a human-readable README and a verification procedure that opposing counsel can execute with standard open-source tools.

The bundle is also the "Courtroom Evidence Pack" surfaced by the one-click export button in  Screen 15 (UI).

## Non-goals

- The bundle is **not** a sworn translation deliverable. Where a jurisdiction requires a sworn translation, the bundle may serve as input to that process but does not substitute for it.
- The bundle is **not** a SaaS product artifact. It is an on-premises / customer-tenant deliverable built from custody on demand.
- The bundle is **not** a legal-privilege container. Privileged material inside the bundle remains privileged; 502(d) protective order protections must be arranged by counsel.

---

## Bundle format

- Single file: deterministic `.tar.gz`.
  - Entries sorted lexicographically.
  - Fixed mtime for all entries: the `timestamp` field of the triggering `JOB_CLOSED` custody event (ISO-8601 UTC).
  - No uid/gid leakage (all entries: uid=0, gid=0, uname=root, gname=root).
  - Deflate compression with fixed `mtime=0` gzip header field (set via `gzip -n`).
  - Two independent operators rebuilding from the same custody chain on two different machines must produce byte-identical archives (SHA-256 equal).
- File naming: `<document_id>.evidence.<YYYY-MM-DDTHH-MM-SSZ>.tar.gz`.
- Top-level `MANIFEST.json` enumerates every entry in the bundle with its SHA-256; `MANIFEST.json.sig` is a detached cosign-style signature over `MANIFEST.json`.

### Bundle trigger

The bundle is built by invoking:

```
scripts/build_evidence_bundle.py --doc <document_id> --out <path>
```

The same operation is reachable from the UI's **"Courtroom Evidence Pack"** button in  Screen 15. The UI export is a thin wrapper around the script; the script is the single source of truth. The script:

1. Validates the document's custody chain end-to-end (`verify_custody_file`).
2. Downloads or locates the RFC3161 TSA responses countersigning the chain tips.
3. Collects the SLSA attestations and SBOM for every model referenced in `TRANSLATION_APPLIED` events within the chain.
4. Writes the canonical directory layout.
5. Computes SHA-256 of every file and emits `MANIFEST.json`.
6. Signs `MANIFEST.json` with the release-automation identity key (detached signature to `MANIFEST.json.sig`).
7. Emits the deterministic tarball.

---

## Bundle layout

```
<document_id>.evidence.<ts>.tar.gz
├── MANIFEST.json                       # Enumerates all entries + SHA-256s + bundle-level hash
├── MANIFEST.json.sig                   # Detached signature over MANIFEST.json (release automation identity)
├── README.md                           # Human-readable "how to verify this bundle"
├── original/
│   ├── source.pdf                      # Original evidence file (hash matches JOB_STARTED.source_sha256)
│   ├── source.sha256                   # Text file: the SHA-256 of source.pdf
│   └── metadata.json                   # Intake metadata: tenant_id, privilege_flag, source_mime, ingest timestamp
├── ocr/
│   ├── <doc>.ocr.pdf                   # Authoritative searchable PDF with embedded OCR text layer
│   ├── <doc>.ocr.txt                   # Authoritative OCR plain text
│   ├── <doc>.structure.json            # Layout/table structure JSON if Document Intelligence ran
│   └── ocr.manifest.json               # Per-page confidence, engine versions, DPI escalation record
├── language/
│   ├── <doc>.language.json             #  per-span language sidecar (redacted per A-EN-02)
│   └── language.manifest.json          # Detector version, detector_model_sha256, tokenizer_sha256
├── translation/
│   ├── <doc>.translation.json          #  translation sidecar (the certified artifact if certified=true)
│   ├── <doc>.translation.json.sig      # Detached signature over sidecar SHA-256 by the reviewer (PIV/CAC)
│   ├── translation.manifest.json       # All engines attempted, COMETKiwi scores, rejections summary
│   └── glossary/
│       └── <tenant>.<glossary_id>.json.sig    # Optional signed glossary used during decoding
├── custody/
│   ├── <doc>.custody.jsonl             # Complete hash-chained event log for the job
│   ├── custody.verify.txt              # Output of `verify_custody_file <doc>.custody.jsonl`
│   └── tsa/
│       ├── anchor-<n>.tsr              # RFC3161 TimeStampResp for chain tip at anchor point n
│       └── anchor-<n>.metadata.json    # Anchor metadata: chain tip hash, TSA URL, anchor reason
├── attestations/
│   ├── slsa/
│   │   ├── ctranslate2-opus-mt-en-fr.intoto.jsonl
│   │   ├── ctranslate2-nllb-200-1.3b.intoto.jsonl
│   │   ├── vllm-open-weights-llm-awq-int4.intoto.jsonl
│   │   └── ...                         # One file per model referenced in TRANSLATION_APPLIED
│   ├── sbom/
│   │   └── container-image.spdx.json   # SPDX SBOM for the runtime image at job time
│   └── policy/
│       └── <tenant_policy_hash>.json   # Frozen policy payload (engine allowlist, residency, retention)
└── reviewer/
    ├── certification.json              # reviewer_id, auth_method, decision, timestamps, sidecar hashes
    ├── certification.sig               # PIV/CAC/OIDC-MFA-bound signature over certification.json
    └── reviewer-certs/                 # X.509 certificate chain for reviewer's signing certificate
        ├── reviewer-leaf.crt
        └── reviewer-chain.pem
```

Notes:

- The cloud-engine case (managed cloud LLM-MT provider under BAA): the `attestations/slsa/` directory carries the provider's public model card and any provider-signed attestation where available. Where no SLSA attestation is obtainable, a `cloud-llm.provenance-gap.json` note file documents the "cloud-trusted, provenance unverifiable" posture and references the applicable playbook step.
- The air-gap case: the `attestations/policy/` directory may carry the frozen offline CA public key that signed in-house plugins (see `PLUGIN_REGISTERED.signer_fingerprint`).
- For documents where translation was never performed (OCR-only evidence), the `translation/` and `reviewer/` directories are absent and the README is templated accordingly.

---

## `manifest.json` schema

`MANIFEST.json` is a JSON document with a stable shape. Its own SHA-256 is **not** included in `MANIFEST.json` (for obvious reasons); instead, `MANIFEST.json.sig` is the detached signature binding.

```
{
  "bundle_spec_version": "1.0",
  "bundle_id": "<document_id>.evidence.<ISO-8601-UTC>",
  "document_id": "<document_id>",
  "tenant_id": "<tenant_id>",
  "bundle_built_at": "<ISO-8601-UTC>",
  "bundle_builder": {
    "tool": "scripts/build_evidence_bundle.py",
    "tool_version": "<semver>",
    "pipeline_version": "<ocr_local.version.__version__>"
  },
  "signing_identity": {
    "key_id": "<pinned release-automation key id>",
    "fingerprint_sha256": "<hex>",
    "issuer": "<subject/issuer DN or Sigstore identity>"
  },
  "entries": [
    {
      "path": "MANIFEST.json.sig",
      "sha256": "<hex>",
      "size": <bytes>
    },
    {
      "path": "README.md",
      "sha256": "<hex>",
      "size": <bytes>
    },
    {
      "path": "original/source.pdf",
      "sha256": "<hex>",
      "size": <bytes>,
      "notes": "matches JOB_STARTED.source_sha256"
    },
    ...
  ],
  "bundle_sha256": "<hex>",
  "custody_chain_tip_sha256": "<hex>",
  "tsa_anchors": [
    {
      "anchor_index": 1,
      "path": "custody/tsa/anchor-1.tsr",
      "chain_tip_sha256": "<hex>",
      "tsa_url": "<url>",
      "anchor_reason": "document_close"
    }
  ],
  "certified": <true|false>,
  "reviewer_auth_method": "<piv_cac|oidc_mfa|hardware_token|none>",
  "disclosure": {
    "engine_classes_used": ["nmt", "llm_local"],
    "cloud_providers_used": [],
    "retention_classes": [],
    "licenses_touched": ["Apache-2.0"],
    "privilege_hard_blocks": 0
  }
}
```

Field rules:

- `bundle_spec_version` is this document's version. Incremented only on breaking layout / field changes.
- `bundle_sha256` is the SHA-256 of the assembled tarball content **before** the manifest itself is written (chicken-and-egg: this is approximated via a two-pass build where the final manifest records the hash of the inner-tar body and the outer signature binds it).
- `entries[]` is lexicographically sorted by `path`.
- `disclosure.cloud_providers_used` is the set of cloud engine classes invoked for **any** translation in this job, even if not the certified output. Populated from the union of `TRANSLATION_APPLIED.engine_type=llm_cloud` events in the custody chain.
- `disclosure.privilege_hard_blocks` is the count of `TRANSLATION_REJECTED` events with `reason_code=privilege_hard_block`. Useful for discovery meet-and-confer.
- `reviewer_auth_method` is copied from the latest `TRANSLATION_REVIEWED` event for the document; `"none"` if no certification occurred.

---

## Verification procedure (documented in README.md)

Opposing counsel or a neutral can verify the bundle end-to-end with openly available tools (`openssl`, `cosign` or compatible, `jq`, `sha256sum`, and the project's `verify_custody_file`).

1. **Validate bundle signature against pinned CA / signing identity**:
   - Extract the tarball to a working directory.
   - Run `cosign verify-blob --key <pinned release key>.pub --signature MANIFEST.json.sig MANIFEST.json` (or the equivalent `openssl dgst -sha256 -verify ... -signature MANIFEST.json.sig MANIFEST.json`).
2. **Re-compute SHA-256 of every file; compare to `MANIFEST.json.entries[]`**:
   - `sha256sum <each-path>` and cross-reference with `MANIFEST.json`.
3. **Verify custody chain**:
   - `verify_custody_file custody/<doc>.custody.jsonl` -- confirms each event's hash chains to the prior event and that no event has been modified.
4. **Verify RFC3161 anchor responses** against chain tips:
   - For each anchor: `openssl ts -verify -data <chain_tip_binary> -in custody/tsa/anchor-<n>.tsr -CAfile <tsa-ca-bundle>`. The `chain_tip_binary` is the canonical serialization of the event whose SHA-256 is recorded in the anchor's `metadata.json`.
5. **Verify reviewer signature on translation sidecar**:
   - Construct the reviewer's full certificate chain from `reviewer/reviewer-certs/`.
   - `openssl dgst -sha256 -verify <reviewer-leaf.crt pubkey> -signature translation/<doc>.translation.json.sig translation/<doc>.translation.json`.
   - Confirm the reviewer cert chain's root is accepted (federal PKI for PIV/CAC, enterprise IdP root for OIDC-MFA, or the offline root for air-gap hardware tokens).
6. **Confirm `TRANSLATION_REVIEWED` event in custody** matches the reviewer signature:
   - Locate the `TRANSLATION_REVIEWED` event with `decision=approved` and `reviewer_auth_method in {piv_cac, oidc_mfa, hardware_token}`.
   - Confirm `new_sidecar_sha256` equals the SHA-256 of `translation/<doc>.translation.json`.
   - Confirm `reviewer_id` matches the subject of the reviewer's leaf certificate.
7. **Confirm `source_ocr_sha256` in `TRANSLATION_APPLIED`** matches the SHA-256 of `ocr/<doc>.ocr.txt`:
   - `sha256sum ocr/<doc>.ocr.txt` and cross-reference with the `source_ocr_sha256` field of every `TRANSLATION_APPLIED` event in custody.
8. **Spot-check `TRANSLATION_REJECTED` events for unexpected denials**:
   - `jq 'select(.event_type == "TRANSLATION_REJECTED")' custody/<doc>.custody.jsonl` -- confirm that reject reasons are consistent with the matter's posture (e.g., `privilege_hard_block` is expected for privileged documents).
9. **Inspect `attestations/slsa/`** for provenance of each model used:
   - For each `TRANSLATION_APPLIED.weights_sha256`, locate the corresponding `.intoto.jsonl` and confirm the subject digest matches.
10. **Read `README.md`** -- the human-readable summary is the proponent's disclosure statement. It documents any `CUSTODY_TSA_WARNING` events, any cloud engine usage, any NC-licensed model touch, and the matter-specific reviewer authority.

A `custody/custody.verify.txt` file is included with the pre-computed output of `verify_custody_file` for convenience; it is redundant with step 3 above and is provided only for quick inspection.

---

## FRE 902(14) self-authentication posture

Under Federal Rule of Evidence 902(14), a record generated by an electronic process or system that produces an accurate result may be self-authenticated by a written certification of a qualified person describing the process. The evidence bundle's combination of:

- `MANIFEST.json.sig` signed by a pinned release-automation identity key, and
- the reviewer's X.509 certificate chain (PIV/CAC or equivalent), and
- RFC3161 TSA responses countersigning chain tips, and
- the documented pipeline attestations (SLSA + SBOM),

meets this standard when accompanied by the custodian-of-records declaration template shipped alongside this specification. The declaration template lives at `docs/forensic/declaration-translation-custodian.md` (to be authored as a companion deliverable); it recites:

- That the witness is the custodian of records for the OCR/translation pipeline.
- That the bundle was produced by `scripts/build_evidence_bundle.py` from the custody chain of the document in question.
- That the custody chain was verified end-to-end before the bundle was produced.
- That the pipeline process is the same process documented in this specification and in the LLM-MT Admissibility Playbook.
- That the custodian has personal knowledge of the pipeline's ordinary operation, or is testifying from business records sufficient to establish it.

For expert-witness translation testimony (distinct from custodian testimony), the expert declaration template at `docs/forensic/declaration-translation-expert.md` (to be authored) addresses the Daubert factors as laid out in the LLM-MT Admissibility Playbook Step 10.

---

## Determinism requirement

The evidence bundle **must be deterministically reproducible**. Two independent operators running `scripts/build_evidence_bundle.py` with access to the same custody chain and the same attestation/TSA/model archives must produce byte-identical `.tar.gz` files.

Determinism is enforced by:

- Sorted tar entries.
- Fixed per-entry mtime (doc-close timestamp).
- Fixed uid/gid (0/0) and user/group names (root/root).
- Gzip with `-n` (no name, no mtime in gzip header).
- Stable JSON serialization (sorted keys, UTF-8, no trailing whitespace, `\n` line endings) for all JSON files in the bundle.

A CI test verifies that two independent bundle builds of a golden-corpus document produce identical SHA-256s. Any non-determinism regression blocks release.

---

## Threats addressed by the bundle

The bundle is designed to defeat or answer the following adversary actions (cross-referenced to Panel 3's threat-model matrix):

- **T4 -- Custody chain modified post-hoc**: MANIFEST signature + RFC3161 anchors detect tampering; chain verification via `verify_custody_file` localizes the break to a specific event.
- **T6 -- Model binary swapped**: SLSA attestation verification against `weights_sha256` recorded in custody catches binary substitution.
- **T9 -- TSA outage during document close**: `CUSTODY_TSA_WARNING` events are disclosed in README; retroactive anchoring runbook applies.
- **T13 -- Reviewer cert revoked post-signing**: archived reviewer cert chain in `reviewer/reviewer-certs/` allows historical signature validation under the certificate valid at signing time.
- **T15 -- Legal hold / GDPR erasure conflict**: the bundle itself is a retained artifact independent of the source custody file; once produced, it is governed by legal-hold rules, not erasure obligations.

---

## UI trigger --  Screen 15 "Courtroom Evidence Pack"

The  management UI exposes the bundle export as a one-click action on Screen 15 (Audit). The button is enabled only when the document's custody chain satisfies the bundle precondition gates:

- Custody chain verified (`verify_custody_file` passes).
- RFC3161 anchor coverage at minimum (document-close anchor present OR the document has no `TRANSLATION_REVIEWED` event yet).
- If any `TRANSLATION_REVIEWED` event exists, reviewer cert chain is retrievable.
- If any `TRANSLATION_APPLIED` event exists, corresponding SLSA attestations are retrievable.

When any precondition is unmet, the UI displays which gate failed and links to the operator runbook for resolution. The button never silently degrades -- clicking it when a gate has failed is an error, not a best-effort export.

The UI also exposes a **dry-run / preview** mode that produces the manifest without emitting the tarball, so operators and reviewers can inspect the contents before committing to a bundle export. The preview output is **not signed** and **not valid** as evidence; it is a diagnostic view only.

---

## Governance

- This specification is reviewed at minimum **annually** by the forensic panel or whenever a material regulatory / case-law / layout change is required. Changes are versioned via `bundle_spec_version`.
- `bundle_spec_version=1.0` is the initial release ( +  P1 deliverable under E-X-005).
- Backward compatibility: bundles built under `1.0` must remain verifiable under all future specification versions. Verification tooling retains historical readers.
- Any change that would alter `bundle_sha256` for the same input is a breaking change and requires a major-version bump.

---

## Related documents

- `docs/forensic/custody-schema-v1.json` -- authoritative custody event schema (10 events)
- `docs/forensic/llm-mt-admissibility-playbook.md` -- legal posture for defending LLM-MT output
- `docs/architecture/forensic-ai-boundary-contract.md` -- forensic-core vs. AI-adjacent boundary contract
- `docs/planning/2026-04-24-translation-swarm/panels/panel-03-forensic-ediscovery.md` -- panel of record
- `docs/planning/2026-04-24-translation-swarm/04-plan-D-management-ui-enrichments.md` --  UI design, Screen 15 Audit
- `scripts/build_evidence_bundle.py` -- canonical bundle builder (to be authored under  +  )

---

*End of Evidence Bundle Specification v1.0. Panel-of-record: .*
