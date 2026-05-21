"""PDF page rendering with DPI escalation."""

from pathlib import Path
from typing import Optional

try:
    from pdf2image import convert_from_path
except ImportError:
    convert_from_path = None


class RenderError(Exception):
    """Exception raised during PDF rendering."""

    pass


# DPI escalation schedule
DPI_SCHEDULE = [300, 450, 600]


def get_next_dpi(current_dpi: int) -> Optional[int]:
    """Get next DPI step from escalation schedule.

    Args:
        current_dpi: Current DPI value

    Returns:
        Next DPI value or None if at maximum
    """
    try:
        current_index = DPI_SCHEDULE.index(current_dpi)
        if current_index < len(DPI_SCHEDULE) - 1:
            return DPI_SCHEDULE[current_index + 1]
    except ValueError:
        pass

    return None


def render_page(pdf_path: str | Path, page_num: int, dpi: int):
    """Render a single PDF page at specified DPI.

    Args:
        pdf_path: Path to PDF file
        page_num: Page number (1-indexed)
        dpi: DPI resolution for rendering

    Returns:
        PIL Image object

    Raises:
        RenderError: If rendering fails
    """
    if convert_from_path is None:
        raise RenderError("pdf2image not available")

    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise RenderError(f"PDF file not found: {pdf_path}")

    try:
        images = convert_from_path(
            pdf_path,
            dpi=dpi,
            first_page=page_num,
            last_page=page_num,
        )

        if not images:
            raise RenderError(f"No image rendered for page {page_num}")

        return images[0]

    except RenderError:
        raise
    except Exception as e:
        raise RenderError(f"Failed to render page {page_num} at {dpi} DPI: {e}") from e
