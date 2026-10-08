"""Cross-cluster custody-chain ingest endpoint -- Plan C Phase 1, item C6.

A peer cluster posts a signed custody event here when it hands a job
off, receives a job, or rebalances a job to/from this cluster.  The
endpoint:

* Authenticates the caller via a shared bearer token
  (``OCR_FEDERATION_CUSTODY_AUTH_TOKEN``).
* Verifies the event's HMAC-SHA256 signature against
  ``OCR_FEDERATION_CUSTODY_HMAC_KEY`` (constant-time).
* Validates the schema via Pydantic (rejects unknown event types,
  empty fields, etc.).
* Persists the event into a SQLite store with a
  ``UNIQUE (job_id, signature)`` constraint so retries are
  idempotent.

The router is feature-gated -- it is **only mounted** in
``api.main.create_app`` when ``OCR_FEDERATION_CUSTODY_ENABLED`` is
truthy.  ``api.auth`` separately exempts the path from the global
API-key middleware so peer clusters can authenticate with the bearer
token alone.
"""

from __future__ import annotations

import hmac
import logging
import os
import sqlite3
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Header, HTTPException, Request, status
from pydantic import BaseModel, Field, field_validator, model_validator

from api.limits import get_default_rate, limiter
from coordinator.federation.custody import (
    VALID_EVENT_TYPES,
    verify_signature,
)

router = APIRouter(prefix="/api/v1/federation", tags=["federation"])

_LOG = logging.getLogger("api.federation_custody")


# ---------------------------------------------------------------------------
# State management
# ---------------------------------------------------------------------------
_SCHEMA_LOCK = threading.Lock()
_SCHEMA_INITIALIZED_FOR: str | None = None

# SQLite DDL.  ``UNIQUE (job_id, signature)`` is what makes idempotent
# replay possible: if the same peer retries the same signed payload,
# the INSERT raises IntegrityError and we return ``status="duplicate"``.
_SCHEMA = """
CREATE TABLE IF NOT EXISTS cross_cluster_custody_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id TEXT NOT NULL,
    tenant_id TEXT NOT NULL,
    source_cluster TEXT NOT NULL,
    target_cluster TEXT NOT NULL,
    parent_event_hash TEXT NOT NULL,
    event_type TEXT NOT NULL,
    timestamp_utc TEXT NOT NULL,
    dispatch_reason TEXT NOT NULL,
    signature TEXT NOT NULL,
    received_at REAL NOT NULL DEFAULT (strftime('%s','now')),
    UNIQUE (job_id, signature)
);
CREATE INDEX IF NOT EXISTS ix_ccce_job_id
    ON cross_cluster_custody_events (job_id);
CREATE INDEX IF NOT EXISTS ix_ccce_event_type
    ON cross_cluster_custody_events (event_type);
"""


def _db_path() -> str:
    """Resolve the configured SQLite path (defaults to ocr_output/...)."""
    return os.environ.get(
        "OCR_FEDERATION_CUSTODY_DB_PATH",
        "ocr_output/federation_custody.db",
    )


def _ensure_schema() -> str:
    """Create the schema if it has not been created for this DB path yet."""
    global _SCHEMA_INITIALIZED_FOR
    path = _db_path()
    if _SCHEMA_INITIALIZED_FOR == path:
        return path
    with _SCHEMA_LOCK:
        if _SCHEMA_INITIALIZED_FOR == path:
            return path
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(path)
        try:
            conn.executescript(_SCHEMA)
            columns = {
                row[1]
                for row in conn.execute(
                    "PRAGMA table_info(cross_cluster_custody_events)"
                )
            }
            if "tenant_id" not in columns:
                conn.execute(
                    "ALTER TABLE cross_cluster_custody_events "
                    "ADD COLUMN tenant_id TEXT NOT NULL DEFAULT ''"
                )
            conn.commit()
        finally:
            conn.close()
        _SCHEMA_INITIALIZED_FOR = path
    return path


def _reset_state_for_tests() -> None:
    """Clear the schema-init guard so the next call re-initialises."""
    global _SCHEMA_INITIALIZED_FOR
    with _SCHEMA_LOCK:
        _SCHEMA_INITIALIZED_FOR = None


# ---------------------------------------------------------------------------
# Pydantic schema
# ---------------------------------------------------------------------------
class CustodyIngestRequest(BaseModel):
    """Schema for the cross-cluster custody ingest payload."""

    job_id: str = Field(min_length=1, max_length=128)
    tenant_id: str = Field(default="", max_length=128)
    source_cluster: str = Field(min_length=1, max_length=64)
    target_cluster: str = Field(min_length=1, max_length=64)
    parent_event_hash: str = Field(default="", max_length=128)
    event_type: str = Field(min_length=1, max_length=32)
    timestamp_utc: str = Field(min_length=1, max_length=64)
    dispatch_reason: str = Field(min_length=1, max_length=128)
    signature: str = Field(min_length=1, max_length=256)

    @field_validator("event_type")
    @classmethod
    def _check_event_type(cls, v: str) -> str:
        if v not in VALID_EVENT_TYPES:
            raise ValueError(f"unknown event_type: {v!r}")
        return v

    @field_validator("timestamp_utc")
    @classmethod
    def _check_timestamp(cls, v: str) -> str:
        """Reject unparseable or timezone-naive custody timestamps."""
        candidate = v.strip().replace("Z", "+00:00")
        try:
            parsed = datetime.fromisoformat(candidate)
        except ValueError as exc:
            raise ValueError("timestamp_utc must be an ISO-8601 timestamp") from exc
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            raise ValueError("timestamp_utc must include an explicit timezone")
        return v.strip()

    @model_validator(mode="after")
    def _check_parent_hash(self) -> "CustodyIngestRequest":
        """Keep production handoff chains anchored to canonical SHA-256 hashes."""
        if self.event_type == "JOB_REBALANCED":
            if self.parent_event_hash:
                raise ValueError("JOB_REBALANCED parent_event_hash must be empty")
            return self
        deployment = os.environ.get("DEPLOYMENT_ENV", "development").strip().lower()
        if deployment in {"production", "prod"} and not self.tenant_id.strip():
            raise ValueError("production custody handoffs require tenant_id")
        if deployment in {"production", "prod"}:
            import re

            if not re.fullmatch(r"[0-9a-fA-F]{64}", self.parent_event_hash):
                raise ValueError(
                    "production custody handoffs require a 64-character SHA-256 parent hash"
                )
        return self

    def to_payload(self) -> dict[str, Any]:
        """Return the payload as a plain ``dict`` (signature included)."""
        return {
            "job_id": self.job_id,
            "tenant_id": self.tenant_id,
            "source_cluster": self.source_cluster,
            "target_cluster": self.target_cluster,
            "parent_event_hash": self.parent_event_hash,
            "event_type": self.event_type,
            "timestamp_utc": self.timestamp_utc,
            "dispatch_reason": self.dispatch_reason,
            "signature": self.signature,
        }


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------
def _persist_event(req: CustodyIngestRequest) -> bool:
    """Insert ``req`` into SQLite; return ``True`` if newly persisted, ``False`` if duplicate."""
    path = _ensure_schema()
    conn = sqlite3.connect(path)
    try:
        try:
            conn.execute(
                "INSERT INTO cross_cluster_custody_events ("
                "job_id, tenant_id, source_cluster, target_cluster, parent_event_hash, "
                "event_type, timestamp_utc, dispatch_reason, signature"
                ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    req.job_id,
                    req.tenant_id,
                    req.source_cluster,
                    req.target_cluster,
                    req.parent_event_hash,
                    req.event_type,
                    req.timestamp_utc,
                    req.dispatch_reason,
                    req.signature,
                ),
            )
            conn.commit()
            return True
        except sqlite3.IntegrityError:
            # Duplicate (job_id, signature) -- idempotent replay.
            return False
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------
def _expected_bearer_token() -> str:
    return os.environ.get("OCR_FEDERATION_CUSTODY_AUTH_TOKEN", "").strip()


def _federation_insecure_allowed() -> bool:
    """S3-A13: never allow INSECURE under production DEPLOYMENT_ENV."""
    insecure = os.environ.get(
        "OCR_FEDERATION_CUSTODY_INSECURE", ""
    ).strip().lower() in ("1", "true", "yes", "on")
    if not insecure:
        return False
    dep = os.environ.get("DEPLOYMENT_ENV", "development").strip().lower()
    if dep in ("production", "prod"):
        return False
    return True


def _check_bearer(authorization: str | None) -> None:
    expected = _expected_bearer_token()
    if not expected:
        # Insecure mode: allow unauthenticated calls only when explicit
        # opt-in env var is set AND not production.  Otherwise fail closed.
        if _federation_insecure_allowed():
            return
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Federation custody auth token is not configured",
        )
    if not authorization:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing Authorization header",
        )
    parts = authorization.split(" ", 1)
    if len(parts) != 2 or parts[0].lower() != "bearer":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authorization header must be 'Bearer <token>'",
        )
    provided = parts[1].strip()
    try:
        matches = bool(provided) and hmac.compare_digest(
            provided.encode("utf-8"), expected.encode("utf-8")
        )
    except (UnicodeEncodeError, TypeError):
        matches = False
    if not matches:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid bearer token",
        )


def _check_tenant_authority(tenant_id: str) -> None:
    """Bind the fleet credential to an explicit configured tenant set."""
    allowed = {
        value.strip()
        for value in os.environ.get(
            "OCR_FEDERATION_CUSTODY_TENANT_ALLOWLIST", ""
        ).split(",")
        if value.strip()
    }
    deployment = os.environ.get("DEPLOYMENT_ENV", "development").strip().lower()
    if deployment in {"production", "prod"} and not allowed:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Federation custody tenant allowlist is not configured",
        )
    if allowed and tenant_id not in allowed:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Federation credential is not authorized for this tenant",
        )


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------
@router.post("/custody/ingest")
@limiter.limit(get_default_rate())
async def ingest_custody_event(
    body: CustodyIngestRequest,
    request: Request,
    authorization: str | None = Header(default=None),
) -> dict[str, Any]:
    """Accept a signed cross-cluster custody event from a peer cluster."""
    _check_bearer(authorization)
    _check_tenant_authority(body.tenant_id)

    hmac_key = os.environ.get("OCR_FEDERATION_CUSTODY_HMAC_KEY", "").strip()
    insecure = _federation_insecure_allowed()
    if not hmac_key and not insecure:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="HMAC key is not configured",
        )

    if not verify_signature(body.to_payload(), hmac_key, insecure=insecure):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Signature verification failed",
        )

    persisted_new = _persist_event(body)
    return {
        "status": "accepted" if persisted_new else "duplicate",
        "job_id": body.job_id,
        "tenant_id": body.tenant_id,
        "event_type": body.event_type,
    }


__all__ = ["router"]
