"""Local RFC3161/TSA scaffolding for certified translation custody tests."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol
from urllib.parse import urlparse


class TSAError(RuntimeError):
    """Base class for timestamp-authority failures."""


class TSATimestampUnavailable(TSAError):
    """Raised when a TSA cannot produce a timestamp response."""


class TSAClient(Protocol):
    def timestamp(self, request_sha256: str) -> bytes:
        """Return an RFC3161 timestamp response token for a request digest."""


@dataclass(frozen=True)
class TSAAnchor:
    clock_source: str
    rfc3161_tsr_sha256: str
    tsa_request_sha256: str
    tsa_response_path: str


class FakeRFC3161TSAClient:
    """Deterministic local TSA client for tests and offline validation."""

    def __init__(self, *, responder_id: str = "local-fake-tsa") -> None:
        self.responder_id = responder_id

    def timestamp(self, request_sha256: str) -> bytes:
        if len(request_sha256) != 64:
            raise TSATimestampUnavailable("request_sha256 must be a SHA-256 hex digest")
        payload = {
            "format": "fake-rfc3161-tsr-v1",
            "request_sha256": request_sha256,
            "responder_id": self.responder_id,
        }
        return json.dumps(payload, sort_keys=True).encode("utf-8")


class OutageTSAClient:
    """TSA client that always fails; used to prove certified-flip refusal."""

    def timestamp(self, request_sha256: str) -> bytes:
        raise TSATimestampUnavailable("timestamp authority unavailable")


def _der_length(length: int) -> bytes:
    if length < 0x80:
        return bytes([length])
    encoded = length.to_bytes((length.bit_length() + 7) // 8, "big")
    return bytes([0x80 | len(encoded)]) + encoded


def _der(tag: int, value: bytes) -> bytes:
    return bytes([tag]) + _der_length(len(value)) + value


def _der_sequence(*values: bytes) -> bytes:
    return _der(0x30, b"".join(values))


def _der_integer(value: int) -> bytes:
    if value == 0:
        raw = b"\x00"
    else:
        raw = value.to_bytes((value.bit_length() + 7) // 8, "big")
        if raw[0] & 0x80:
            raw = b"\x00" + raw
    return _der(0x02, raw)


def _der_boolean(value: bool) -> bytes:
    return _der(0x01, b"\xff" if value else b"\x00")


def _der_oid(oid: str) -> bytes:
    parts = [int(part) for part in oid.split(".")]
    if len(parts) < 2:
        raise ValueError("OID must have at least two arcs")
    encoded = bytes([40 * parts[0] + parts[1]])
    for part in parts[2:]:
        if part == 0:
            encoded += b"\x00"
            continue
        chunks = []
        while part:
            chunks.append(part & 0x7F)
            part >>= 7
        for index, chunk in enumerate(reversed(chunks)):
            encoded += bytes([chunk | (0x80 if index < len(chunks) - 1 else 0)])
    return _der(0x06, encoded)


def build_rfc3161_timestamp_query(request_sha256: str, *, cert_req: bool = True) -> bytes:
    """Build a minimal RFC3161 TimeStampReq DER payload for a SHA-256 digest."""

    if len(request_sha256) != 64:
        raise TSATimestampUnavailable("request_sha256 must be a SHA-256 hex digest")
    try:
        digest = bytes.fromhex(request_sha256)
    except ValueError as exc:
        raise TSATimestampUnavailable("request_sha256 must be hex encoded") from exc

    sha256_algorithm_identifier = _der_sequence(_der_oid("2.16.840.1.101.3.4.2.1"))
    message_imprint = _der_sequence(sha256_algorithm_identifier, _der(0x04, digest))
    return _der_sequence(_der_integer(1), message_imprint, _der_boolean(cert_req))


def _read_der_length(data: bytes, offset: int) -> tuple[int, int]:
    if offset >= len(data):
        raise ValueError("truncated DER length")
    first = data[offset]
    offset += 1
    if first < 0x80:
        return first, offset
    count = first & 0x7F
    if count == 0 or offset + count > len(data):
        raise ValueError("invalid DER length")
    return int.from_bytes(data[offset : offset + count], "big"), offset + count


def _extract_rfc3161_status(tsr: bytes) -> int:
    """Extract PKIStatus from a TimeStampResp enough to reject failures."""

    offset = 0
    if not tsr or tsr[offset] != 0x30:
        raise ValueError("TSA response is not a DER SEQUENCE")
    _, offset = _read_der_length(tsr, offset + 1)
    if offset >= len(tsr) or tsr[offset] != 0x30:
        raise ValueError("TSA response missing status sequence")
    _, offset = _read_der_length(tsr, offset + 1)
    if offset >= len(tsr) or tsr[offset] != 0x02:
        raise ValueError("TSA response missing PKIStatus integer")
    length, offset = _read_der_length(tsr, offset + 1)
    if offset + length > len(tsr):
        raise ValueError("truncated PKIStatus integer")
    return int.from_bytes(tsr[offset : offset + length], "big")


class HTTPRFC3161TSAClient:
    """Minimal HTTP RFC3161 TSA client for low-volume dev/live evidence."""

    def __init__(self, url: str, *, timeout: float = 15.0) -> None:
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("TSA URL must be an http(s) URL")
        self.url = url
        self.timeout = timeout

    def timestamp(self, request_sha256: str) -> bytes:
        try:
            import requests

            response = requests.post(
                self.url,
                data=build_rfc3161_timestamp_query(request_sha256),
                headers={
                    "Content-Type": "application/timestamp-query",
                    "Accept": "application/timestamp-reply",
                },
                timeout=self.timeout,
            )
            response.raise_for_status()
            tsr = response.content
            status = _extract_rfc3161_status(tsr)
        except Exception as exc:
            raise TSATimestampUnavailable(f"timestamp authority request failed: {exc}") from exc
        if status not in {0, 1}:
            raise TSATimestampUnavailable(f"timestamp authority rejected request: status={status}")
        return tsr


def canonical_request_sha256(payload: dict) -> str:
    """Hash the certification payload that will be timestamped."""

    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def anchor_certification_payload(
    payload: dict,
    *,
    tsa_client: TSAClient,
    tsa_store_dir: str | Path,
) -> TSAAnchor:
    """Timestamp a certification payload and persist the TSR bytes locally."""

    request_sha = canonical_request_sha256(payload)
    tsr = tsa_client.timestamp(request_sha)
    tsr_sha = hashlib.sha256(tsr).hexdigest()
    store = Path(tsa_store_dir)
    store.mkdir(parents=True, exist_ok=True)
    tsr_path = store / f"{tsr_sha}.tsr"
    tsr_path.write_bytes(tsr)
    return TSAAnchor(
        clock_source="rfc3161_tsa",
        rfc3161_tsr_sha256=tsr_sha,
        tsa_request_sha256=request_sha,
        tsa_response_path=str(tsr_path),
    )
