"""Forensic chain-of-custody logging with tamper-evident hash chains.

Each processing event is cryptographically linked to the previous event
via SHA-256 hashing, creating a verifiable audit trail.

Output: JSONL files (one JSON object per line) for append-only safety.
Location: ocr_output/custody/<document_hash>.custody.jsonl
"""

import datetime
import hashlib
import json
import logging
import os
import threading
import time
from typing import Any, Optional

__all__ = [
    "CustodyChain",
    "EVENT_TYPES",
    "MAX_CUSTODY_RETRIES",
    "CUSTODY_RETRY_DELAYS",
    "compute_file_hash",
    "verify_custody_file",
]

logger = logging.getLogger(__name__)

# Retry configuration for custody chain disk writes
MAX_CUSTODY_RETRIES = 3
CUSTODY_RETRY_DELAYS = [0.1, 0.5, 2.0]  # seconds


class CustodyChain:
    """Hash-chained event log for a single document's processing lifecycle.
    
    Each event contains:
    - event_type: Processing stage identifier
    - data: Stage-specific metadata
    - timestamp: ISO 8601 with millisecond precision
    - prev_hash: SHA-256 hash of previous event (None for genesis)
    - hash: SHA-256 hash of this event (computed from all fields)
    """
    
    def __init__(self, document_id: str, source_path: str, custody_dir: str = ""):
        """Initialize a new custody chain for a document.
        
        Args:
            document_id: Unique document identifier (e.g., SHA-256 hash prefix)
            source_path: Original source file path
            custody_dir: Directory for .custody.jsonl files
        """
        self.document_id = document_id
        self.source_path = source_path
        self.events: list[dict] = []
        self._prev_hash: Optional[str] = None
        self._custody_dir = custody_dir
        self._filepath: Optional[str] = None
        self._lock = threading.Lock()
        self.integrity_compromised = False

        if custody_dir:
            os.makedirs(custody_dir, exist_ok=True)
            safe_id = os.path.basename(document_id)
            self._filepath = os.path.join(custody_dir, f"{safe_id}.custody.jsonl")
    
    def append_event(self, event_type: str, data: dict[str, Any] | None = None) -> dict:
        """Append a hash-chained event to the custody log.

        Args:
            event_type: One of the defined event types (see EVENT_TYPES)
            data: Stage-specific metadata dictionary

        Returns:
            The complete event dict with computed hash
        """
        with self._lock:
            event = {
                "document_id": self.document_id,
                "event_type": event_type,
                "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="milliseconds"),
                "data": data or {},
                "prev_hash": self._prev_hash,
            }

            # Compute hash of event (excluding the hash field itself)
            event_bytes = json.dumps(event, sort_keys=True, default=str).encode("utf-8")
            event_hash = hashlib.sha256(event_bytes).hexdigest()
            event["hash"] = event_hash

            # Write to disk BEFORE advancing in-memory hash.
            # fsync ensures bytes reach stable storage before we update
            # in-memory state, preventing hash chain divergence on power loss.
            if self._filepath:
                write_succeeded = False
                for attempt in range(1, MAX_CUSTODY_RETRIES + 1):
                    try:
                        with open(self._filepath, "a", encoding="utf-8") as f:
                            f.write(json.dumps(event, default=str) + "\n")
                            f.flush()
                            os.fsync(f.fileno())
                        write_succeeded = True
                        break
                    except OSError as exc:
                        delay = CUSTODY_RETRY_DELAYS[attempt - 1]
                        if attempt < MAX_CUSTODY_RETRIES:
                            logger.warning(
                                "Custody write attempt %d failed: %s, retrying in %ss",
                                attempt, exc, delay,
                            )
                            time.sleep(delay)
                        else:
                            logger.critical(
                                "Custody chain write FAILED after %d attempts for %s "
                                "-- forensic integrity compromised",
                                MAX_CUSTODY_RETRIES,
                                self.document_id,
                            )
                            self.integrity_compromised = True

                if not write_succeeded:
                    # Append to in-memory events but do NOT advance the hash,
                    # so the next event will still chain from the last persisted hash.
                    self.events.append(event)
                    return event

            self.events.append(event)
            self._prev_hash = event_hash

            return event

    def log_event(self, event_type: str, data: dict[str, Any] | None = None) -> dict:
        """Alias for :meth:`append_event` -- preferred name for new callers.

        Plan B M1-PR3 introduced the ``log_event`` name to align translation
        custody adapters with the language-detection / translation panel
        playbook.  ``append_event`` remains for backwards compatibility.
        """

        return self.append_event(event_type, data)

    def verify_chain(self) -> tuple[bool, str]:
        """Verify the integrity of the entire chain.
        
        Returns:
            (is_valid, message) tuple
        """
        if not self.events:
            return True, "Empty chain"
        
        prev_hash = None
        for i, event in enumerate(self.events):
            # Check prev_hash linkage
            if event["prev_hash"] != prev_hash:
                return False, f"Broken chain at event {i}: expected prev_hash={prev_hash}, got {event['prev_hash']}"
            
            # Recompute hash
            event_copy = {k: v for k, v in event.items() if k != "hash"}
            event_bytes = json.dumps(event_copy, sort_keys=True, default=str).encode("utf-8")
            computed_hash = hashlib.sha256(event_bytes).hexdigest()
            
            if computed_hash != event["hash"]:
                return False, f"Tampered event {i}: computed hash {computed_hash} != stored {event['hash']}"
            
            prev_hash = event["hash"]
        
        return True, f"Chain verified: {len(self.events)} events"
    
    @classmethod
    def load_from_file(cls, filepath: str) -> "CustodyChain":
        """Load a custody chain from a JSONL file.
        
        Args:
            filepath: Path to .custody.jsonl file
            
        Returns:
            CustodyChain instance with loaded events
        """
        chain = cls(document_id="", source_path="")
        
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    event = json.loads(line)
                    chain.events.append(event)
                    chain._prev_hash = event.get("hash")
                    if not chain.document_id:
                        chain.document_id = event.get("document_id", "")
                        chain.source_path = event.get("data", {}).get("source_path", "")
        
        chain._filepath = filepath
        return chain
    
    def get_summary(self) -> dict[str, Any]:
        """Get a summary of the custody chain for metadata/reporting."""
        return {
            "document_id": self.document_id,
            "total_events": len(self.events),
            "event_types": list({e["event_type"] for e in self.events}),
            "first_event": self.events[0]["timestamp"] if self.events else None,
            "last_event": self.events[-1]["timestamp"] if self.events else None,
            "chain_hash": self._prev_hash,
        }


# Defined event types for the processing pipeline
EVENT_TYPES = {
    "file_ingested": "Source file accepted for processing",
    "page_extracted": "Page image extracted from document",
    "ocr_primary": "Primary OCR engine (PaddleOCR) processed page",
    "ocr_fallback": "Fallback OCR engine (Tesseract) processed page",
    "ocr_image_only": "OCR failed, page preserved as image",
    "language_detected": "Language detection completed",
    "language_reprocess": "Page re-processed with detected language model",
    "docintel_analysis": "Document Intelligence analysis completed",
    "assembly_complete": "Final document assembled from pages",
    "compression_complete": "PDF compression/optimization completed",
    "dpi_escalation": "Page re-extracted at higher DPI due to low OCR confidence",
    "processing_failed": "Processing stage failed",
    "temp_files_purged": "Orphaned temporary files purged from storage",
    "pii_purged": "PII/PHI entity records purged on demand",
    "tenant_purged": "All data for a tenant purged on demand",
    "output_cleaned": "Output files cleaned per retention policy",
    "audit_logs_rotated": "Audit log records archived and rotated",
    # Plan B M1-PR3 -- translation pipeline events
    "TRANSLATION_APPLIED": "Translation engine applied to OCR spans",
    "TRANSLATION_REJECTED": "Translation rejected by tenant policy or privilege guard",
    "QUALITY_BELOW_THRESHOLD": "Translation quality score below configured threshold",
    "TRANSLATION_FALLBACK": "Translation fell back from one engine to another",
    "TRANSLATION_REVIEWED": "Human reviewer approved/rejected a translation (forensic gate)",
    "GLOSSARY_APPLIED": "Glossary term overrides applied to translation output",
    "TRANSLATION_SKIPPED": "Translation pipeline skipped for this document/span",
    "CUSTODY_TSA_WARNING": "Timestamp authority outage or warning recorded for custody anchoring",
    "PLUGIN_REGISTERED": "Enrichment plugin admitted to the Plan D plugin bus",
    "PLUGIN_REJECTED": "Enrichment plugin rejected by Plan D plugin-bus admission",
}


def compute_file_hash(filepath: str) -> str:
    """Compute SHA-256 hash of a file."""
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def verify_custody_file(filepath: str) -> tuple[bool, str]:
    """Convenience function to verify a custody JSONL file.
    
    Args:
        filepath: Path to .custody.jsonl file
        
    Returns:
        (is_valid, message) tuple
    """
    try:
        chain = CustodyChain.load_from_file(filepath)
        return chain.verify_chain()
    except (json.JSONDecodeError, OSError) as exc:
        return False, f"Failed to load custody file: {exc}"
