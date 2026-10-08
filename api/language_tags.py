"""Small, dependency-free validation helpers for BCP-47-like language tags."""

from __future__ import annotations

import re

_LANGUAGE_TAG_RE = re.compile(r"^[A-Za-z]{2,3}(?:-[A-Za-z0-9]{2,8})*$")


def canonicalize_language_tag(value: str, *, max_length: int = 16) -> str:
    """Validate and canonically case a bounded language tag.

    This intentionally covers the language/script/region/variant forms used
    by the translation APIs without importing a large locale database.  It
    accepts private or deployment-specific language codes such as ``qq``
    while rejecting whitespace, underscores, separators, and oversized
    subtags.  Duplicate tags can then be detected by comparing the returned
    value.
    """
    if not isinstance(value, str):
        raise ValueError("language tag must be a string")
    tag = value.strip()
    if not tag or len(tag) > max_length or not _LANGUAGE_TAG_RE.fullmatch(tag):
        raise ValueError("language tag must be a valid bounded BCP-47 value")

    parts = tag.split("-")
    canonical = [parts[0].lower()]
    for part in parts[1:]:
        if len(part) == 4 and part.isalpha():
            canonical.append(part.title())
        elif (len(part) == 2 and part.isalpha()) or (len(part) == 3 and part.isdigit()):
            canonical.append(part.upper())
        else:
            canonical.append(part.lower())
    return "-".join(canonical)


def canonicalize_language_tags(values: list[str], *, max_items: int = 32) -> list[str]:
    """Validate, canonicalize, and de-duplicate language tags in order."""
    if len(values) > max_items:
        raise ValueError(f"at most {max_items} language tags are supported")
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        canonical = canonicalize_language_tag(value)
        if canonical in seen:
            raise ValueError("language tags must be unique")
        seen.add(canonical)
        result.append(canonical)
    if not result:
        raise ValueError("at least one language tag is required")
    return result
