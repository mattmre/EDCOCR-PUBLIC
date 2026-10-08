"""Pure custody-chain authority selection for API job surfaces."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from ocr_local.features.custody import (
    CustodyChain,
    validate_custody_document_id,
)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class CustodyBytesCandidate:
    """One descriptor-bound custody artifact and its already-read bytes."""

    relative_path: str
    raw: bytes
    device: int | None = None
    inode: int | None = None
    size_bytes: int | None = None
    nlink: int | None = None


@dataclass(frozen=True)
class SelectedCustodyChain:
    """The sole chain authorized for the requested persisted job identity."""

    relative_path: str
    document_id: str
    chain: CustodyChain
    chain_head: str
    verification_message: str
    device: int | None
    inode: int | None
    size_bytes: int | None


class CustodyAuthorityError(ValueError):
    """Fail-closed custody selection error with a stable API-facing code."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class _InspectedCandidate:
    source: CustodyBytesCandidate
    document_id: str
    chain: CustodyChain
    verification_message: str
    declared_hashes: frozenset[str]
    job_match: bool
    hash_match: bool


def _declared_source_hashes(chain: CustodyChain) -> tuple[frozenset[str], bool]:
    declared: set[str] = set()
    malformed = False
    for event in chain.events:
        if str(event.get("event_type") or "").lower() != "file_ingested":
            continue
        data = event.get("data") or {}
        if not isinstance(data, dict):
            continue
        fingerprint = data.get("source_fingerprint") or {}
        if not isinstance(fingerprint, dict):
            fingerprint = {}
        for value in (
            data.get("file_hash"),
            data.get("source_hash"),
            fingerprint.get("content_sha256"),
        ):
            normalized = str(value or "").strip().lower()
            if not normalized:
                continue
            if _SHA256_RE.fullmatch(normalized):
                declared.add(normalized)
            else:
                malformed = True
    return frozenset(declared), malformed


def _jsonl_envelope_count(raw: bytes) -> int:
    """Count newline-delimited envelopes before chain verification.

    Malformed or unverifiable payloads still consume the request-wide event
    ceiling so an attacker cannot force unbounded parse/verify work.
    """
    if not raw:
        return 0
    count = raw.count(b"\n")
    if not raw.endswith(b"\n"):
        count += 1
    return count


def select_job_custody_chain(
    *,
    job_id: str,
    source_hash: str,
    candidates: Sequence[CustodyBytesCandidate],
    max_bytes: int,
    max_events: int,
    allow_unattested_single: bool = False,
    accept_job_id_hash_conflict: bool = False,
    allow_legacy_job_without_hash: bool = True,
    max_candidates: int | None = None,
    max_total_bytes: int | None = None,
    max_total_events: int | None = None,
) -> SelectedCustodyChain:
    """Select exactly one verified filename-safe chain for a persisted job.

    No filesystem access occurs here. Callers must supply bytes read through
    their already-authorized descriptors. The two compatibility switches are
    limited to the mutation path for historical redetect chains; evidence
    export uses the strict defaults.
    """
    source_hash_s = str(source_hash or "").strip().lower()
    source_hash_valid = bool(_SHA256_RE.fullmatch(source_hash_s))
    if max_candidates is not None and len(candidates) > max_candidates:
        raise CustodyAuthorityError(
            "custody_chain_bounds_exceeded",
            "Custody candidate count exceeds the configured bound.",
        )
    if max_total_bytes is not None and sum(len(item.raw) for item in candidates) > (
        max_total_bytes
    ):
        raise CustodyAuthorityError(
            "custody_chain_bounds_exceeded",
            "Custody candidate bytes exceed the configured bound.",
        )
    inspected: list[_InspectedCandidate] = []
    invalid_count = 0
    total_events = 0

    for candidate in candidates:
        filename = Path(candidate.relative_path).name
        if not filename.endswith(".custody.jsonl"):
            invalid_count += 1
            continue
        if candidate.nlink is not None and candidate.nlink != 1:
            invalid_count += 1
            continue
        envelopes = _jsonl_envelope_count(candidate.raw)
        total_events += envelopes
        if max_total_events is not None and total_events > max_total_events:
            raise CustodyAuthorityError(
                "custody_chain_bounds_exceeded",
                "Custody event count exceeds the configured bound.",
            )
        try:
            chain = CustodyChain.load_from_bytes(
                candidate.raw,
                max_bytes=max_bytes,
                max_events=max_events,
            )
            valid, message = chain.verify_chain()
            if not valid or not chain.events or not chain.document_id:
                raise ValueError("custody chain is empty or unverifiable")
            document_id = validate_custody_document_id(chain.document_id)
            if filename != f"{document_id}.custody.jsonl":
                raise ValueError("custody filename does not match document id")
        except CustodyAuthorityError:
            raise
        except (TypeError, ValueError):
            invalid_count += 1
            continue

        hashes, malformed_hash = _declared_source_hashes(chain)
        if malformed_hash:
            invalid_count += 1
            continue
        inspected.append(
            _InspectedCandidate(
                source=candidate,
                document_id=document_id,
                chain=chain,
                verification_message=message,
                declared_hashes=hashes,
                job_match=document_id == str(job_id),
                hash_match=source_hash_valid and source_hash_s in hashes,
            )
        )

    if invalid_count:
        raise CustodyAuthorityError(
            "custody_chain_invalid",
            "One or more custody artifacts are malformed or unverifiable.",
        )

    authoritative: list[_InspectedCandidate] = []
    for candidate in inspected:
        hashes_match_exactly = source_hash_valid and candidate.declared_hashes == {
            source_hash_s
        }
        legacy_job_match = (
            allow_legacy_job_without_hash
            and candidate.job_match
            and not candidate.declared_hashes
        )
        compatible_job_match = candidate.job_match and (
            hashes_match_exactly or accept_job_id_hash_conflict
        )
        if (
            legacy_job_match
            or compatible_job_match
            or (not candidate.job_match and hashes_match_exactly)
        ):
            authoritative.append(candidate)

    conflicting_job_authority = any(
        candidate.job_match
        and bool(candidate.declared_hashes)
        and candidate.declared_hashes != {source_hash_s}
        for candidate in inspected
    )

    if len(authoritative) > 1 or (conflicting_job_authority and authoritative):
        raise CustodyAuthorityError(
            "custody_chain_ambiguous",
            "Conflicting canonical custody chains match this job.",
        )
    if len(authoritative) == 1:
        selected = authoritative[0]
    elif allow_unattested_single and len(inspected) == 1:
        selected = inspected[0]
    elif inspected:
        raise CustodyAuthorityError(
            "custody_chain_foreign",
            "No verified custody chain attests the requested job identity.",
        )
    else:
        raise CustodyAuthorityError(
            "custody_chain_missing",
            "No verified custody chain exists for this job.",
        )

    return SelectedCustodyChain(
        relative_path=selected.source.relative_path,
        document_id=selected.document_id,
        chain=selected.chain,
        chain_head=str(selected.chain.events[-1].get("hash") or "n/a"),
        verification_message=selected.verification_message,
        device=selected.source.device,
        inode=selected.source.inode,
        size_bytes=selected.source.size_bytes,
    )
