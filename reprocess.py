"""CLI entry point for failure reprocessing pipeline."""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

from reprocess.failures import FailureRecord, FailureStatus, FailureStore
from reprocess.ocr_core import OCREngine
from reprocess.renderer import RenderError, get_next_dpi, render_page
from reprocess.reporter import ReportGenerator, ReprocessResult

logger = logging.getLogger(__name__)


def setup_logging(verbose: bool = False) -> None:
    """Configure logging.

    Args:
        verbose: Enable debug logging
    """
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def reprocess_failure(
    record: FailureRecord,
    ocr_engine: OCREngine,
    dry_run: bool = False,
) -> ReprocessResult:
    """Reprocess a single failure record.

    Args:
        record: FailureRecord to reprocess
        ocr_engine: OCREngine instance
        dry_run: If True, simulate without actual processing

    Returns:
        ReprocessResult with outcome
    """
    logger.info(f"Reprocessing {record.source_path} page {record.page_num}")

    # Get next DPI
    next_dpi = get_next_dpi(record.dpi_used)
    if next_dpi is None:
        logger.warning(f"No higher DPI available (current: {record.dpi_used})")
        return ReprocessResult(
            source_path=record.source_path,
            page_num=record.page_num,
            original_error=record.error,
            original_dpi=record.dpi_used,
            retry_dpi=record.dpi_used,
            success=False,
            resolution_method="",
            new_error="No higher DPI available",
        )

    if dry_run:
        logger.info(f"[DRY RUN] Would retry at {next_dpi} DPI")
        return ReprocessResult(
            source_path=record.source_path,
            page_num=record.page_num,
            original_error=record.error,
            original_dpi=record.dpi_used,
            retry_dpi=next_dpi,
            success=False,
            resolution_method="dry_run",
            new_error="",
        )

    # Render at higher DPI
    try:
        image = render_page(record.source_path, record.page_num, next_dpi)
    except RenderError as e:
        logger.error(f"Render failed: {e}")
        return ReprocessResult(
            source_path=record.source_path,
            page_num=record.page_num,
            original_error=record.error,
            original_dpi=record.dpi_used,
            retry_dpi=next_dpi,
            success=False,
            resolution_method="",
            new_error=str(e),
        )

    # Run OCR
    try:
        text, method = ocr_engine.run_ocr(image)
        if text and text.strip():
            logger.info(f"Success with {method} at {next_dpi} DPI")
            return ReprocessResult(
                source_path=record.source_path,
                page_num=record.page_num,
                original_error=record.error,
                original_dpi=record.dpi_used,
                retry_dpi=next_dpi,
                success=True,
                resolution_method=method,
                new_error="",
            )
        else:
            logger.warning(f"OCR returned empty text with {method}")
            return ReprocessResult(
                source_path=record.source_path,
                page_num=record.page_num,
                original_error=record.error,
                original_dpi=record.dpi_used,
                retry_dpi=next_dpi,
                success=False,
                resolution_method=method,
                new_error="No text extracted at higher DPI",
            )
    except Exception as e:
        logger.error(f"OCR failed: {e}")
        return ReprocessResult(
            source_path=record.source_path,
            page_num=record.page_num,
            original_error=record.error,
            original_dpi=record.dpi_used,
            retry_dpi=next_dpi,
            success=False,
            resolution_method="",
            new_error=str(e),
        )


def update_failure_record(
    record: FailureRecord,
    result: ReprocessResult,
) -> FailureRecord:
    """Update failure record based on reprocessing result.

    Args:
        record: Original FailureRecord
        result: ReprocessResult from reprocessing

    Returns:
        Updated FailureRecord
    """
    record.retry_count += 1
    record.last_retry_timestamp = datetime.now().isoformat()
    record.dpi_used = result.retry_dpi

    if result.success:
        record.status = FailureStatus.RESOLVED.value
        record.resolution_method = result.resolution_method
    else:
        if result.new_error:
            record.error = result.new_error
        # Status will be updated in main() if max retries reached

    return record


def main() -> int:
    """Main CLI entry point.

    Returns:
        Exit code (0 for success)
    """
    parser = argparse.ArgumentParser(
        description="Reprocess OCR failures with DPI escalation"
    )
    parser.add_argument(
        "--failures",
        default="/app/ocr_output/failures.csv",
        help="Path to failures CSV (default: /app/ocr_output/failures.csv)",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=2,
        help="Maximum retry attempts per failure (default: 2)",
    )
    parser.add_argument(
        "--output-report",
        help="Path for markdown report (default: failures_report.md in same dir as CSV)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Simulate processing without making changes",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable debug logging",
    )

    args = parser.parse_args()
    setup_logging(args.verbose)

    # Determine report path
    if args.output_report:
        report_path = Path(args.output_report)
    else:
        csv_path = Path(args.failures)
        report_path = csv_path.parent / "failures_report.md"

    logger.info("Starting failure reprocessing pipeline")
    logger.info(f"Failures CSV: {args.failures}")
    logger.info(f"Max retries: {args.max_retries}")
    logger.info(f"Report output: {report_path}")
    if args.dry_run:
        logger.info("DRY RUN MODE - no changes will be made")

    # Load failures
    store = FailureStore(args.failures)
    all_failures = store.read_failures()
    logger.info(f"Total failures in CSV: {len(all_failures)}")

    # Filter retriable (reuse already-loaded failures)
    retriable = store.get_retriable_failures(all_failures=all_failures, max_retries=args.max_retries)
    logger.info(f"Retriable failures: {len(retriable)}")

    if not retriable:
        logger.info("No retriable failures found")
        # Generate empty report
        reporter = ReportGenerator(report_path)
        reporter.save_report()
        return 0

    # Initialize OCR engine
    ocr_engine = OCREngine(use_paddle=True, lang="en")

    # Process each failure
    reporter = ReportGenerator(report_path)
    updated_records = []

    for record in retriable:
        result = reprocess_failure(record, ocr_engine, dry_run=args.dry_run)
        reporter.add_result(result)

        if not args.dry_run:
            # Update record
            updated_record = update_failure_record(record, result)

            # Mark as exhausted if max retries reached
            if updated_record.retry_count >= args.max_retries and not result.success:
                updated_record.status = FailureStatus.RETRY_EXHAUSTED.value

            updated_records.append(updated_record)

    # Update CSV with changes
    if not args.dry_run and updated_records:
        # Merge updated records back into full list
        updated_map = {
            (r.source_path, r.page_num): r for r in updated_records
        }
        final_records = []
        for record in all_failures:
            key = (record.source_path, record.page_num)
            if key in updated_map:
                final_records.append(updated_map[key])
            else:
                final_records.append(record)

        store.write_failures(final_records)
        logger.info(f"Updated {len(updated_records)} records in CSV")

    # Generate report
    reporter.save_report()
    logger.info(f"Report saved to {report_path}")

    # Summary
    resolved_count = sum(1 for r in reporter.results if r.success)
    logger.info(f"Resolved: {resolved_count}/{len(retriable)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
