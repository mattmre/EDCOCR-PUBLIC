"""Failure tracking models and CSV storage with backward compatibility."""

import csv
import logging
import re
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path

logger = logging.getLogger(__name__)


class FailureType(Enum):
    """Classification of failure types."""

    OCR_TEXT = "ocr_text"
    CRITICAL = "critical"
    EXTRACT_FAILED = "extract_failed"
    UNKNOWN = "unknown"


class FailureStatus(Enum):
    """Status of failure record."""

    FAILED = "failed"
    PENDING_RETRY = "pending_retry"
    RESOLVED = "resolved"
    RETRY_EXHAUSTED = "retry_exhausted"
    NOT_RETRIABLE = "not_retriable"


@dataclass
class FailureRecord:
    """Complete failure record with retry tracking."""

    timestamp: str
    source_path: str
    page_num: int
    error: str
    failure_type: str = field(default="unknown")
    status: str = field(default="failed")
    retry_count: int = field(default=0)
    dpi_used: int = field(default=300)
    last_retry_timestamp: str = field(default="")
    resolution_method: str = field(default="")

    def to_dict(self) -> dict:
        """Convert to dictionary for CSV writing."""
        return asdict(self)


class FailureStore:
    """CSV-based failure storage with backward compatibility."""

    # Legacy format: 4 columns
    LEGACY_HEADERS = ["Timestamp", "SourcePath", "PageNum", "Error"]

    # Enhanced format: 10 columns
    ENHANCED_HEADERS = [
        "Timestamp",
        "SourcePath",
        "PageNum",
        "Error",
        "FailureType",
        "Status",
        "RetryCount",
        "DPIUsed",
        "LastRetryTimestamp",
        "ResolutionMethod",
    ]

    def __init__(self, csv_path: str | Path):
        """Initialize failure store.

        Args:
            csv_path: Path to failures CSV file
        """
        self.csv_path = Path(csv_path)

    def read_failures(self) -> list[FailureRecord]:
        """Read failure records from CSV with backward compatibility.

        Supports both legacy 4-column and enhanced 10-column formats.

        Returns:
            List of FailureRecord objects
        """
        if not self.csv_path.exists():
            return []

        records = []
        with open(self.csv_path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            headers = reader.fieldnames or []

            # Detect format by column count
            is_legacy = len(headers) == 4

            for row in reader:
                try:
                    if is_legacy:
                        # Legacy format: auto-classify and set defaults
                        record = FailureRecord(
                            timestamp=row["Timestamp"],
                            source_path=row["SourcePath"],
                            page_num=int(row["PageNum"]),
                            error=row["Error"],
                            failure_type=self.classify_failure(row["Error"]).value,
                            status=FailureStatus.FAILED.value,
                            retry_count=0,
                            dpi_used=300,
                            last_retry_timestamp="",
                            resolution_method="",
                        )
                    else:
                        # Enhanced format: read all fields
                        record = FailureRecord(
                            timestamp=row["Timestamp"],
                            source_path=row["SourcePath"],
                            page_num=int(row["PageNum"]),
                            error=row["Error"],
                            failure_type=row.get("FailureType", "unknown"),
                            status=row.get("Status", "failed"),
                            retry_count=int(row.get("RetryCount", 0)),
                            dpi_used=int(row.get("DPIUsed", 300)),
                            last_retry_timestamp=row.get("LastRetryTimestamp", ""),
                            resolution_method=row.get("ResolutionMethod", ""),
                        )
                    records.append(record)
                except (ValueError, KeyError) as e:
                    logger.warning("Skipping malformed row in CSV: %s. Error: %s", row, e)

        return records

    def write_failures(self, records: list[FailureRecord]) -> None:
        """Write failure records to CSV in enhanced format.

        Args:
            records: List of FailureRecord objects
        """
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)

        with open(self.csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.ENHANCED_HEADERS)
            writer.writeheader()

            for record in records:
                row = {
                    "Timestamp": record.timestamp,
                    "SourcePath": record.source_path,
                    "PageNum": str(record.page_num),
                    "Error": record.error,
                    "FailureType": record.failure_type,
                    "Status": record.status,
                    "RetryCount": str(record.retry_count),
                    "DPIUsed": str(record.dpi_used),
                    "LastRetryTimestamp": record.last_retry_timestamp,
                    "ResolutionMethod": record.resolution_method,
                }
                writer.writerow(row)

    def classify_failure(self, error: str) -> FailureType:
        """Classify failure type based on error message.

        Args:
            error: Error message string

        Returns:
            FailureType enum value
        """
        error_lower = error.lower()

        # OCR text extraction failures
        ocr_patterns = [
            r"no text extracted",
            r"ocr.*empty",
            r"text extraction.*failed",
            r"blank.*page",
            r"insufficient.*text",
        ]
        for pattern in ocr_patterns:
            if re.search(pattern, error_lower):
                return FailureType.OCR_TEXT

        # Critical/system failures
        critical_patterns = [
            r"out of memory",
            r"cuda.*error",
            r"gpu.*error",
            r"system.*error",
            r"permission denied",
        ]
        for pattern in critical_patterns:
            if re.search(pattern, error_lower):
                return FailureType.CRITICAL

        # Extract/render failures
        extract_patterns = [
            r"extract.*failed",
            r"pdf.*corrupt",
            r"cannot.*render",
            r"invalid.*pdf",
            r"decrypt.*failed",
        ]
        for pattern in extract_patterns:
            if re.search(pattern, error_lower):
                return FailureType.EXTRACT_FAILED

        return FailureType.UNKNOWN

    def get_retriable_failures(
        self,
        all_failures: list[FailureRecord] | None = None,
        max_retries: int = 2,
    ) -> list[FailureRecord]:
        """Get failures eligible for retry.

        Args:
            all_failures: Pre-loaded list of FailureRecord objects to filter.
                If None, reads from CSV (backward-compatible).
            max_retries: Maximum retry attempts allowed

        Returns:
            List of retriable FailureRecord objects
        """
        if all_failures is None:
            all_failures = self.read_failures()
        retriable = []

        for record in all_failures:
            # Skip if already resolved or exhausted
            if record.status in (
                FailureStatus.RESOLVED.value,
                FailureStatus.RETRY_EXHAUSTED.value,
                FailureStatus.NOT_RETRIABLE.value,
            ):
                continue

            # Skip if max retries reached
            if record.retry_count >= max_retries:
                continue

            # Skip EXTRACT_FAILED and CRITICAL - not retriable with DPI escalation
            if record.failure_type in (
                FailureType.EXTRACT_FAILED.value,
                FailureType.CRITICAL.value,
            ):
                continue

            retriable.append(record)

        return retriable
