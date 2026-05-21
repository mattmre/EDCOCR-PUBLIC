# LLM-MT Admissibility Playbook

**Status**: P1 deliverable (E-X-005, tracked under )
**Authority**: Panel 3 , Forensic / eDiscovery Expert Panel
**Jurisdictional scope**: U.S. federal courts (FRE / FRCP); state analogues apply by reference
**Owner**: Forensic-AI Boundary Guardian (rotating chair) + Trial Counsel
**Version**: 1.0

---

## Purpose

This playbook is the project's formal legal posture for defending LLM-based machine translation (LLM-MT) output in litigation. It ships with the project documentation, is referenced by  (translation pipeline), and is binding on any PR that introduces or modifies a translation engine, a reviewer workflow, or an evidence-bundle deliverable.

The playbook answers, in order, ten questions a federal court is likely to ask under FRE 702, FRE 901, FRE 902(14), FRE 1001-1008, FRCP 26(g), FRCP 34, FRCP 37(e), FRE 502(d), and the 2024 amendments to FRE 702.

## Scope and operating assumptions

- Machine translation output covered by this playbook is produced by the pipeline defined in `docs/planning/2026-04-24-translation-swarm/02-plan-B-translation-pipeline.md`.
- The authoritative OCR text layer is produced by CTC-only OCR (PaddleOCR + Tesseract fallback). There is **no generative text in the OCR layer**; this is the forensic-AI boundary contract's central invariant.
- "Certified translation" in this document is a **pipeline artifact** (`certified: true` on the translation sidecar after TRANSLATION_REVIEWED). It is not synonymous with a sworn translation by a court-certified translator; the playbook explicitly distinguishes the two (see Step 4).
- All custody events referenced are defined in `docs/forensic/custody-schema-v1.json`.

---

## Step 1 -- Establish CTC-OCR baseline (no generative text in OCR)

The first admissibility anchor is that the **authoritative text layer is not AI-generated**.

- OCR is performed by PaddleOCR 2.9.1 (CTC-only decoding) with Tesseract as a fallback. Neither engine is a large language model; neither engine performs sampling over a vocabulary distribution to produce fluent text.
- Under FRE 702, the expert custodian testifies: *"The text layer was produced by a Connectionist Temporal Classification model that maps pixel features to character sequences without a generative component. The model does not hallucinate text; a character not present in the image cannot appear in the output."*
- Expert exhibit: the forensic-AI boundary contract (`docs/architecture/forensic-ai-boundary-contract.md`) and the CI validator (`scripts/validate_feature_boundary.py`) demonstrate that this invariant is enforced at every merge.

**Witness preparation**: the expert must be prepared to distinguish CTC OCR from generative LLM-based OCR systems and to explain why the project categorically rejects generative text in the OCR layer.

---

## Step 2 -- Engine provenance (SLSA attestation, model SHA-256)

For every translation offered as evidence, the proponent must be able to identify the exact engine that produced it.

- Every bundled translation model (CTranslate2-converted OPUS-MT, NLLB-200, MADLAD-400; AWQ-quantized open-weights LLM-MT variants) ships with:
  - **SLSA v1.0 provenance attestation** (`attestations/slsa/*.intoto.jsonl`) describing the build pipeline.
  - **In-toto attestation chain** for the conversion pipeline (raw HuggingFace weights -> CTranslate2 binary; AWQ quantization steps for open-weights LLM-MT variants).
  - **SBOM** (`attestations/sbom/container-image.spdx.json`) for the runtime image.
  - **`weights_sha256`** and **`tokenizer_sha256`** recorded in every `TRANSLATION_APPLIED` custody event.
- Managed cloud engines **cannot** provide SLSA attestations for their model weights. This is a **documented admissibility gap**. Cloud-engine output is treated in discovery as "cloud-trusted, provenance unverifiable" and carries a stronger reviewer-certification requirement (see Step 5) before being offered as evidence.
- Consumer-grade cloud LLM endpoints are removed from production tenant routing. Approved managed cloud LLM-MT paths under BAA (where PHI applies), residency pinning, and a zero-retention contractual rider are the only permitted cloud LLM-MT paths.

**Expert exhibit**: the attestation bundle inside the evidence bundle's `attestations/` directory, cross-referenced with the `weights_sha256` from `TRANSLATION_APPLIED`.

---

## Step 3 -- Custody chain completeness (all 10 event types present, TSA anchored)

A custody chain is **complete** when:

1. The chain begins with a `JOB_STARTED` event anchoring `source_sha256` of the original document.
2. Every translation attempt produces either `TRANSLATION_APPLIED` or `TRANSLATION_REJECTED` (never a silent drop).
3. Every policy decision produces `ENGINE_POLICY_ENFORCED`.
4. Every certification decision produces `TRANSLATION_REVIEWED`.
5. Every plugin load produces `PLUGIN_REGISTERED` or `PLUGIN_REJECTED`.
6. The chain ends with a `JOB_CLOSED` event recording the final chain tip.
7. RFC3161 TSA responses (`.tsr`) countersign the chain tip at **document close**, every **TRANSLATION_REVIEWED**, and every **100 events or 1 hour** (whichever first). Anchor failures degrade to `CUSTODY_TSA_WARNING` warnings and are disclosed in the evidence bundle README.

The 10 event types are defined in `docs/forensic/custody-schema-v1.json`. Any translation offered as evidence without a chain complete in these dimensions is rejected by the pipeline's own certification gate; it cannot reach `certified: true`.

**Expert exhibit**: the output of `verify_custody_file` over the custody JSONL plus the `.tsr` files under `custody/tsa/`.

---

## Step 4 -- `certified: false` semantics (working translation vs. sworn translation)

This is the most frequently misunderstood admissibility point. The project explicitly carries **two** concepts:

- **`certified: false`** is the default on every translation sidecar. It means: *"This is a pipeline-produced working translation. It is not evidence. It is a triage artifact."*
- **`certified: true`** is set only after a `TRANSLATION_REVIEWED` event with `decision=approved` and an acceptable reviewer auth method (see Step 5). It means: *"A qualified human has reviewed the output and certified it for evidentiary use within this matter's scope."*

Neither `certified: false` nor `certified: true` is synonymous with a **court-certified sworn translation** by a translator on a court's approved list. Where a jurisdiction or matter requires a sworn translation, `certified: true` is a necessary but not sufficient precondition; the sworn-translator process is a separate overlay on top of the pipeline certification.

Customer-facing documentation, marketing copy, and the evidence bundle README must all carry this distinction. Conflation of the two is itself a Daubert vulnerability -- an opposing expert will exploit any drift in the project's terminology.

**Customer-facing wording (evidence bundle README template)**:

> "The translation in this bundle carries the `certified: true` flag, which reflects that a qualified human reviewer reviewed the machine translation and certified it within the scope of this matter using a strong authentication method. This is distinct from a 'sworn translation' prepared by a court-certified translator; where the court or opposing counsel requires a sworn translation, this bundle may serve as input to that process but does not substitute for it."

---

## Step 5 -- Reviewer authentication (PIV/CAC / OIDC-MFA / hardware token required for certified flip)

`certified: false -> true` is gated on:

- `decision == "approved"` AND
- `reviewer_auth_method in {piv_cac, oidc_mfa, hardware_token}`.

A `username_password` reviewer is recorded in `TRANSLATION_REVIEWED` but **does not** flip the certified flag. This is a hard-coded rule in the pipeline, mirrored in the CI validator, and is the closure for RED-03 in Panel 3. The rationale is FRE 702 expert witness testimony: when the custodian is asked *"Can you tie this certification to a specific individual whose identity you verified?"*, the answer must be yes, and it must be backed by a strong-auth method appropriate to the jurisdiction (PIV/CAC in U.S. federal / DoD, OIDC-MFA in commercial, hardware token (YubiKey 5-FIPS or equivalent) in air-gap deployments).

The reviewer's X.509 certificate chain is archived in the evidence bundle at `reviewer/reviewer-certs/` so that historical signature validation survives reviewer cert rotation.

---

## Step 6 -- Privilege-lock enforcement (TRANSLATION_REJECTED events logged, never silent drop)

Privilege-flagged documents **never** reach cloud engines. The pipeline enforces this at the policy layer; attempted routing to a cloud engine produces a `TRANSLATION_REJECTED` event with `reason_code=privilege_hard_block` and is blocked before the engine sees any content.

- This is the **privilege log's technical evidence**: every blocked attempt is recorded, so opposing counsel and the court can verify that privileged material was isolated to local engines.
- `TRANSLATION_REJECTED` with `reason_code=policy_denied` covers non-privilege denials (license, residency, COMETKiwi below threshold, operator revocation).
- A 502(d) protective order should be sought early in any matter that uses the translation pipeline, to contain clawback exposure if a local-engine translation of privileged material is later challenged.

**Expert exhibit**: the full list of `TRANSLATION_REJECTED` events from the custody chain, demonstrating completeness of the privilege perimeter.

---

## Step 7 -- Determinism caveat (GPU non-determinism; CPU path is bit-deterministic reference)

The `deterministic: true` field in `TRANSLATION_APPLIED` is **qualified**:

- **CPU path (CTranslate2 on CPU)**: bit-deterministic across platforms given identical (seed, beam_size, temperature=0). This is the reference deterministic path.
- **GPU path (CTranslate2 / vLLM on GPU)**: bit-deterministic on the **same GPU class + same CUDA driver** only. Kernel non-determinism across GPU generations and driver versions is a published CUDA limitation; see the CUDA Toolkit Documentation on floating-point determinism.
- The project's position is that deterministic reproducibility is a property of the CPU path; the GPU path is "deterministic within envelope," and the envelope is documented in the evidence bundle README.

For any translation offered as evidence where reproducibility may be challenged, the pipeline operator should re-run the translation on the CPU reference path and archive both sidecars inside the evidence bundle. Two byte-identical sidecars from CPU path = airtight reproducibility; GPU-only sidecars = defensible within the documented envelope but weaker against a determined cross-examination.

**Expert exhibit**: the regression test in `tests/translation/test_determinism.py` (to be authored under B-EN-09) asserting byte-identical output on two runs, plus the documented GPU envelope.

---

## Step 8 -- RFC3161 timestamp anchor (contemporaneity defense)

The custody chain's integrity is **local-clock-trusted** unless anchored to an external trusted time source. RFC3161 Time-Stamp Protocol (TSP) responses are the project's anchor mechanism.

- TSA anchors are taken at **document close**, every **TRANSLATION_REVIEWED**, and every **100 events or 1 hour** (whichever first).
- A TSA response is a `.tsr` file (DER-encoded TimeStampResp) that countersigns the chain tip's SHA-256 with a trusted TSA's signature.
- Default TSA is a configurable external provider (e.g., DigiCert, Sectigo, FreeTSA); enterprise deployments substitute an internal TSA; air-gap deployments substitute an offline internal TSA whose root is pinned in the image.
- TSA outage does **not** block the pipeline; it fires `CUSTODY_TSA_WARNING` and continues. The runbook describes retroactive anchoring when the TSA is reachable again. Warnings are disclosed in the evidence bundle README.

Under FRE 901(b)(9), the proponent may show authenticity by evidence describing a process or system used to produce a result and showing that the process or system produces an accurate result. RFC3161 anchoring + custody chain + MANIFEST signature is that process.

Verification by opposing counsel: `openssl ts -verify -data <chain_tip> -in anchor-<n>.tsr -CAfile <tsa-ca-bundle>`.

---

## Step 9 -- Evidence bundle export (tar.gz, deterministic, signed manifest per FRE 902(14))

The single courtroom deliverable is the **evidence bundle** defined in `docs/forensic/evidence-bundle-specification.md`.

- Format: deterministic `.tar.gz` (sorted entries, fixed mtime from doc-close timestamp, no uid/gid leakage). Two independent operators rebuilding from the same custody must produce byte-identical bundles.
- File name: `<document_id>.evidence.<YYYY-MM-DDTHH-MM-SSZ>.tar.gz`.
- Contents: original source file, authoritative OCR artifacts, language sidecars, translation sidecars (including rejections manifest), full custody JSONL, RFC3161 anchor responses, SLSA/in-toto attestations, SBOM, reviewer certification + cert chain, and a human-readable README with the verification procedure and the disclosure text for opposing counsel.
- Self-authentication under FRE 902(14): the combination of `MANIFEST.json.sig` (detached cosign-style signature), reviewer X.509 chain, RFC3161 TSA responses, and the documented pipeline attestations meets the "written certification... of a process known to produce a reliable result" standard.

A custodian-of-records declaration template (`docs/forensic/declaration-translation-custodian.md`, to be authored) and an expert-witness declaration template (`docs/forensic/declaration-translation-expert.md`, to be authored) ship alongside the playbook.

---

## Step 10 -- Expert witness qualification (Daubert factors)

For a translation offered as evidence with an expert declaration, the expert witness (typically the technical custodian or a contracted ML-MT specialist) should be prepared to address each Daubert factor on the record.

| Daubert factor | Evidence |
|---|---|
| **Testability** | Engine is reproducible. `seed`, `beam_size`, `temperature`, `weights_sha256`, `tokenizer_sha256`, and `source_ocr_sha256` are recorded in `TRANSLATION_APPLIED`; anyone with the same weights can replay. CPU reference path is bit-deterministic. |
| **Peer review / publication** | OPUS-MT (Tiedemann et al., 2020), NLLB-200 (NLLB Team, 2022), and MADLAD-400 (Kudugunta et al., 2023) are published research models with peer-reviewed papers. Managed cloud LLM-MT engines, where used, must have public safety / eval documentation from their provider. |
| **Known error rate** | Model-level BLEU / chrF / COMET on held-out benchmarks is disclosed; per-matter calibration on tenant-specific held-out sets is recommended when certified output is the basis for expert testimony. COMETKiwi is a **triage gate**, not an accuracy claim. |
| **Standards / controls** | Deterministic decoding, PIV/CAC / OIDC-MFA reviewer authentication, hash-chained custody, RFC3161 anchoring, SLSA v1.0 attestations, Sedona Principles compliance on production. |
| **General acceptance** | NMT is generally accepted (OPUS-MT and NLLB-200 are in widespread production use across industry and academia). LLM-MT is emerging (2023+); the project's position is that **uncertified** LLM-MT is a translation aid, not evidence, and **certified** LLM-MT with human sign-off under this playbook is the evidence. |

---

## Daubert Challenge Response Matrix

Anticipated cross-examination questions and the evidence-bundle artifacts that answer them.

| # | Likely challenge | Answer / artifact |
|---|---|---|
| 1 | "Who reviewed this translation?" | `TRANSLATION_REVIEWED.reviewer_id` + `reviewer/certification.json` + `reviewer/reviewer-certs/*.crt`. |
| 2 | "How was that reviewer authenticated?" | `TRANSLATION_REVIEWED.reviewer_auth_method` (must be in {piv_cac, oidc_mfa, hardware_token}) + reviewer cert chain + optional PIV/CAC authority letter. |
| 3 | "Could the model produce a different output on a different day?" | `deterministic=true` + `seed` + `beam_size` + `temperature` + `weights_sha256` in `TRANSLATION_APPLIED`; reproducibility test artifact; CPU reference path. |
| 4 | "What is the error rate?" | Model BLEU / chrF / COMET on benchmark corpora (cited in README) AND note that certified output reflects human judgment layered on top, not raw model accuracy. |
| 5 | "Did any cloud provider retain this data?" | `TRANSLATION_APPLIED.provider_retention_class` + `provider_region`; privilege-flagged documents never left the local environment (`TRANSLATION_REJECTED.reason_code=privilege_hard_block` events prove the perimeter). |
| 6 | "Did the glossary bias the translation?" | `GLOSSARY_APPLIED.glossary_sha256` + glossary file (producible) + `GLOSSARY_APPLIED.constrained_decoding` (soft hint vs. hard constraint). |
| 7 | "How do you know the custody chain was not tampered with?" | Output of `verify_custody_file` + RFC3161 `.tsr` responses + detached `MANIFEST.json.sig`. Any tampering breaks the hash chain at the tampered event. |
| 8 | "Could the reviewer have signed without seeing the translation?" | Reviewer UI workflow requires rendering the sidecar before accepting the signature; signature is over `sidecar_sha256` which is content-bound; UI evidence is captured in `reviewer/certification.json.ui_trace` (when instrumented). |
| 9 | "What if the model was updated after your translation?" | `weights_sha256` at the time of `TRANSLATION_APPLIED` is preserved; the evidence bundle carries the SLSA attestation for that specific version; a reproducibility archive indexed by `weights_sha256` retains deprecated weights for re-verification. |
| 10 | "What is a COMETKiwi score and why does it matter?" | COMETKiwi is a reference-free quality triage score; it is **not** a calibrated accuracy rate. It gates the pipeline from emitting obviously-low-quality output but does not substitute for human certification. The certified output is the evidence; COMETKiwi is the pre-check. |
| 11 | "Was this model trained on the opposing party's data?" | Public training data is disclosed in the model cards (HuggingFace / research papers). For proprietary tenant-specific fine-tunes (future feature), a separate attestation and disclosure is required. |
| 12 | "Can opposing counsel reproduce this translation independently?" | Yes -- the evidence bundle ships the model SHA-256, tokenizer SHA-256, decoding parameters, and source OCR bytes. Any peer with the same model binaries can re-run. For cloud engines, re-running is subject to the provider's current model version (reproducibility is model-version-scoped). |
| 13 | "Did the pipeline auto-flip certification at any point?" | No -- the pipeline has no auto-flip path. `certified: false -> true` requires a `TRANSLATION_REVIEWED` event with a human-attributable, strong-auth reviewer. |
| 14 | "Is COMETKiwi 0.70 a recognized threshold?" | It is a project-chosen quality gate, not a standardized one. The threshold is configurable per tenant and documented in `tenant_policy_hash`. The pipeline position is that any threshold is a triage signal, not an admissibility claim. |
| 15 | "What if a plugin altered the translation?" | `PLUGIN_REGISTERED` events document every plugin loaded at job time; `boundary_side=forensic_core` plugins require signed manifest; any unsigned plugin fires `PLUGIN_REJECTED` and is quarantined. |

---

## Recent case law guardrails (2023-2025)

- ***Mata v. Avianca***, 22-CV-1461 (S.D.N.Y. Jun. 22, 2023) -- Rule 11 sanctions for submitting generative-AI output (fabricated case citations) without verification. **Our certified-translation workflow with PIV/CAC reviewer authentication is the verification.**
- ***Park v. Kim***, 91 F.4th 610 (2d Cir. 2024) -- confirmed that counsel cannot delegate verification of AI-produced content to the AI itself. Reinforces the human-reviewer requirement.
- ***United States v. Cohen***, 23-CR-371 (S.D.N.Y. 2024) -- hallucinated case citations in a sentencing letter. Distinguished from translation: translation is not legal reasoning, and COMETKiwi is a pre-human triage check; certified output still requires human sign-off.
- ***In re StockX Customer Data Sec. Breach Litig.***, No. 19-cv-12441 (E.D. Mich. 2024) -- predictive coding admissibility extended by analogy to AI-assisted review under Sedona Principle 6. Our position: translation is a pre-review step; the **evidence** is the certified output, not the raw MT.
- ***Walters v. major LLM provider***, 23-cv-3122 (N.D. Ga. 2024) -- product-liability framing of hallucination. Our mitigation: disclose model limitations in README, document the human-certification liability backstop.
- **2024 FRE 702 amendments** (effective December 2023) -- raised the proponent's burden: the court must find by a preponderance that the expert's opinion reflects a reliable application of methodology to the facts. This playbook is the methodology record; the evidence bundle is the application record.

---

## Pre-production checklist (run before offering any translation as evidence)

1. Verify the evidence bundle self-authentication path end-to-end: build, sign, rebuild from custody, re-verify.
2. Run the translation twice on identical inputs with identical seeds; confirm byte-identical sidecars on the CPU reference path (or identical on the same GPU class envelope).
3. Confirm the reviewer workflow with the specific reviewer who will testify. Verify their certificate chain and authentication method.
4. Document the translator chain for any tenant-specific glossary; verify `GLOSSARY_APPLIED` events carry the correct `glossary_sha256`.
5. Run `verify_custody_file` and `openssl ts -verify` against the bundle's `.tsr` files.
6. Confirm no `CUSTODY_TSA_WARNING` events appear in the custody chain; if any do, document the retroactive-anchor status.
7. Review all `TRANSLATION_REJECTED` events for the job -- confirm none are unexpected or suggest a privilege-log gap.
8. Confirm `source_ocr_sha256` in `TRANSLATION_APPLIED` matches the SHA-256 of the OCR text layer in the bundle.
9. Confirm no cloud engine was used for any privilege-flagged document.
10. Prepare the expert-witness declaration citing the Daubert response matrix and the bundle artifacts.

---

## Governance

- This playbook is reviewed at minimum **annually** by the forensic panel (rotating chair) or whenever a material regulatory, case-law, or engine-class change occurs (e.g., a new cloud LLM-MT provider, a new model-governance regulation, a new FRE amendment).
- Updates to this playbook are squash-merge PRs with `docs(forensic)` prefix; any change is tracked in the document's changelog.
- The playbook's canonical location is `docs/forensic/llm-mt-admissibility-playbook.md`. All customer-facing marketing and sales materials that reference admissibility must link to this path.

---

## Related documents

- `docs/forensic/custody-schema-v1.json` -- authoritative custody event schema
- `docs/forensic/evidence-bundle-specification.md` -- courtroom evidence bundle format
- `docs/architecture/forensic-ai-boundary-contract.md` -- forensic-core vs. AI-adjacent boundary contract
- `docs/planning/2026-04-24-translation-swarm/02-plan-B-translation-pipeline.md` --  translation pipeline design
- `docs/planning/2026-04-24-translation-swarm/panels/panel-03-forensic-ediscovery.md` -- panel of record

---

*End of LLM-MT Admissibility Playbook v1.0. Panel-of-record: .*
