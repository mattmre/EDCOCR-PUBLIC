"""Shared utilities for transform operations.

Provides helpers for page spec parsing, path validation, file operations,
and safe atomic writes that are reused across PDF and image transform operations.
"""

import logging
import os
import tempfile
from pathlib import Path
from typing import Union

logger = logging.getLogger(__name__)


# --- Path and Extension Helpers ---


def normalize_extension(ext: str) -> str:
    """Normalize file extension to lowercase with leading dot.
    
    Args:
        ext: File extension (with or without leading dot)
        
    Returns:
        Normalized extension (e.g., ".pdf", ".png")
    """
    ext = ext.strip().lower()
    if not ext.startswith("."):
        ext = "." + ext
    return ext


def get_file_extension(path: Union[str, Path]) -> str:
    """Get normalized file extension from path.
    
    Args:
        path: File path
        
    Returns:
        Normalized extension (e.g., ".pdf")
    """
    return normalize_extension(Path(path).suffix)


def validate_file_exists(path: Union[str, Path], param_name: str = "input_path") -> None:
    """Validate that a file exists and is readable.
    
    Args:
        path: File path to validate
        param_name: Parameter name for error messages
        
    Raises:
        ValueError: If file doesn't exist or isn't readable
    """
    path_obj = Path(path)
    if not path_obj.exists():
        raise ValueError(f"{param_name} does not exist: {path}")
    if not path_obj.is_file():
        raise ValueError(f"{param_name} is not a file: {path}")
    if not os.access(path, os.R_OK):
        raise ValueError(f"{param_name} is not readable: {path}")


def validate_output_path(path: Union[str, Path]) -> None:
    """Validate that output path can be written.
    
    Args:
        path: Output path to validate
        
    Raises:
        ValueError: If output path is invalid or parent doesn't exist
    """
    path_obj = Path(path)
    parent = path_obj.parent
    
    if not parent.exists():
        raise ValueError(f"Output directory does not exist: {parent}")
    if path_obj.exists() and not os.access(path, os.W_OK):
        raise ValueError(f"Output path exists but is not writable: {path}")


# --- Page Specification Parsing ---


def parse_page_spec(spec: Union[str, list, int, None], total_pages: int) -> list[int]:
    """Parse flexible page specification into 0-indexed page list.
    
    Supports:
    - None or "all": all pages
    - Single integer: [n]
    - List of integers: [1, 3, 5]
    - String ranges: "1-3", "1,3,5", "1-3,7-9"
    - Negative indices: -1 for last page
    - String "first"/"last": first/last page
    
    Args:
        spec: Page specification
        total_pages: Total number of pages in document
        
    Returns:
        Sorted list of 0-indexed page numbers
        
    Raises:
        ValueError: If spec is invalid or references out-of-range pages
    """
    if total_pages <= 0:
        raise ValueError(f"total_pages must be positive, got {total_pages}")
    
    # Handle None or "all"
    if spec is None or (isinstance(spec, str) and spec.lower() == "all"):
        return list(range(total_pages))
    
    # Handle single integer
    if isinstance(spec, int):
        page_idx = spec if spec >= 0 else total_pages + spec
        if page_idx < 0 or page_idx >= total_pages:
            raise ValueError(f"Page {spec} out of range [0, {total_pages-1}]")
        return [page_idx]
    
    # Handle list of integers
    if isinstance(spec, list):
        pages = []
        for item in spec:
            if not isinstance(item, int):
                raise ValueError(f"Page list must contain only integers, got {type(item)}")
            page_idx = item if item >= 0 else total_pages + item
            if page_idx < 0 or page_idx >= total_pages:
                raise ValueError(f"Page {item} out of range [0, {total_pages-1}]")
            pages.append(page_idx)
        return sorted(set(pages))
    
    # Handle string specifications
    if isinstance(spec, str):
        spec = spec.strip().lower()
        
        # Handle special keywords
        if spec == "first":
            return [0]
        if spec == "last":
            return [total_pages - 1]
        
        # Parse range specification (e.g., "1-3,5,7-9")
        pages = []
        for part in spec.split(","):
            part = part.strip()
            if "-" in part:
                # Range: "1-3" or "0-5"
                try:
                    start_str, end_str = part.split("-", 1)
                    start = int(start_str.strip())
                    end = int(end_str.strip())
                    
                    # Handle negative indices
                    if start < 0:
                        start = total_pages + start
                    if end < 0:
                        end = total_pages + end
                    
                    if start < 0 or end < 0 or start >= total_pages or end >= total_pages:
                        raise ValueError(f"Range {part} out of bounds [0, {total_pages-1}]")
                    if start > end:
                        raise ValueError(f"Invalid range {part}: start > end")
                    
                    pages.extend(range(start, end + 1))
                except ValueError as e:
                    if "invalid literal" in str(e):
                        raise ValueError(f"Invalid range format: {part}")
                    raise
            else:
                # Single page
                try:
                    page = int(part)
                    page_idx = page if page >= 0 else total_pages + page
                    if page_idx < 0 or page_idx >= total_pages:
                        raise ValueError(f"Page {page} out of range [0, {total_pages-1}]")
                    pages.append(page_idx)
                except ValueError:
                    raise ValueError(f"Invalid page specification: {part}")
        
        return sorted(set(pages))
    
    raise ValueError(f"Unsupported page spec type: {type(spec)}")


# --- Safe File Writing ---


def atomic_write(content_writer, output_path: Union[str, Path], temp_dir: Union[str, Path, None] = None) -> None:
    """Write file atomically using temp file + rename.
    
    Args:
        content_writer: Callable that accepts output path and writes content
        output_path: Final output path
        temp_dir: Optional temp directory (defaults to same dir as output)
        
    Raises:
        Exception: Re-raises any exceptions from content_writer after cleanup
    """
    output_path = Path(output_path)
    temp_dir = Path(temp_dir) if temp_dir else output_path.parent
    
    # Create temp file in same directory as output for atomic rename
    fd, temp_path = tempfile.mkstemp(
        suffix=output_path.suffix,
        dir=temp_dir,
        prefix=".tmp_"
    )
    os.close(fd)  # Close the file descriptor, we'll write via content_writer
    
    temp_path_obj = Path(temp_path)
    try:
        # Write content via callback
        content_writer(temp_path)
        
        # Atomic rename (on same filesystem)
        temp_path_obj.replace(output_path)
        logger.debug(f"Atomic write complete: {output_path}")
        
    except Exception as exc:
        # Clean up temp file on failure
        if temp_path_obj.exists():
            try:
                temp_path_obj.unlink()
            except Exception as cleanup_exc:
                logger.warning(f"Failed to clean up temp file {temp_path}: {cleanup_exc}")
        raise exc


def safe_write_bytes(data: bytes, output_path: Union[str, Path]) -> None:
    """Write bytes to file atomically.
    
    Args:
        data: Bytes to write
        output_path: Output file path
    """
    def writer(path):
        with open(path, "wb") as f:
            f.write(data)
    
    atomic_write(writer, output_path)
