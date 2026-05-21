"""Reprocessing report generation."""

from dataclasses import dataclass
from pathlib import Path


@dataclass
class ReprocessResult:
    """Result of a single reprocessing attempt."""

    source_path: str
    page_num: int
    original_error: str
    original_dpi: int
    retry_dpi: int
    success: bool
    resolution_method: str
    new_error: str = ""


class ReportGenerator:
    """Generate markdown reports for reprocessing runs."""

    def __init__(self, output_path: str | Path):
        """Initialize report generator.

        Args:
            output_path: Path where report will be saved
        """
        self.output_path = Path(output_path)
        self.results: list[ReprocessResult] = []

    def add_result(self, result: ReprocessResult) -> None:
        """Add a reprocessing result.

        Args:
            result: ReprocessResult to add
        """
        self.results.append(result)

    def generate_report(self) -> str:
        """Generate markdown report from accumulated results.

        Returns:
            Markdown-formatted report string
        """
        if not self.results:
            return self._generate_empty_report()

        # Count successes/failures
        resolved = [r for r in self.results if r.success]
        exhausted = [r for r in self.results if not r.success]

        lines = [
            "# Reprocessing Report",
            "",
            "## Summary",
            "",
            f"- **Total Processed**: {len(self.results)}",
            f"- **Resolved**: {len(resolved)}",
            f"- **Retry Exhausted**: {len(exhausted)}",
            "",
        ]

        # Resolved failures table
        if resolved:
            lines.extend([
                "## Resolved Failures",
                "",
                "| Source | Page | Original DPI | Retry DPI | Method |",
                "|--------|------|--------------|-----------|--------|",
            ])
            for r in resolved:
                source_name = Path(r.source_path).name
                lines.append(
                    f"| {source_name} | {r.page_num} | {r.original_dpi} | {r.retry_dpi} | {r.resolution_method} |"
                )
            lines.append("")

        # Retry exhausted table
        if exhausted:
            lines.extend([
                "## Retry Exhausted",
                "",
                "| Source | Page | Original DPI | Retry DPI | Error |",
                "|--------|------|--------------|-----------|-------|",
            ])
            for r in exhausted:
                source_name = Path(r.source_path).name
                error_preview = r.new_error[:50] + "..." if len(r.new_error) > 50 else r.new_error
                lines.append(
                    f"| {source_name} | {r.page_num} | {r.original_dpi} | {r.retry_dpi} | {error_preview} |"
                )
            lines.append("")

        return "\n".join(lines)

    def save_report(self) -> None:
        """Write report to disk."""
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        report_content = self.generate_report()
        self.output_path.write_text(report_content, encoding="utf-8")

    def _generate_empty_report(self) -> str:
        """Generate report when no results."""
        return "\n".join([
            "# Reprocessing Report",
            "",
            "## Summary",
            "",
            "- **Total Processed**: 0",
            "- **Resolved**: 0",
            "- **Retry Exhausted**: 0",
            "",
            "No failures were processed.",
            "",
        ])
