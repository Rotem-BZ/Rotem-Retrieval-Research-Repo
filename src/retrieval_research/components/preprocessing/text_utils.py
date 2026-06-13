"""Shared text preprocessing helpers."""

from __future__ import annotations

import re


def apply_text_transforms(
    text: str,
    *,
    prefix: str,
    suffix: str,
    strip: bool,
    lowercase: bool,
    replace_regexes: dict[str, str],
) -> str:
    transformed = text
    for pattern, replacement in replace_regexes.items():
        transformed = re.sub(pattern, replacement, transformed)
    if strip:
        transformed = transformed.strip()
    if lowercase:
        transformed = transformed.lower()
    return f"{prefix}{transformed}{suffix}"
