from __future__ import annotations

import re


class TextCleaner:
    """
    Cleans raw extracted text before chunking.

    This mirrors the behaviour of utils/text_cleaner.py in the repo so that
    the chunking module can stand alone — the extractors already clean the
    text, but calling clean() here acts as a safety net for any leftover
    noise (e.g. page markers, null bytes, excessive whitespace).
    """

    # Page-marker patterns inserted by PDFExtractor ("--- Page N ---")
    _PAGE_MARKER = re.compile(r"-{3,}\s*Page\s+\d+\s*-{3,}", re.IGNORECASE)

    # Sequences of three or more newlines compressed to two
    _MULTI_NEWLINE = re.compile(r"\n{3,}")

    # Horizontal whitespace (spaces/tabs) collapsed to a single space
    _MULTI_SPACE = re.compile(r"[ \t]+")

    # Null bytes or other non-printable control characters (keep \n \t)
    _CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

    def clean(self, text: str) -> str:
        """
        Apply all cleaning steps and return the cleaned string.

        """
        if not text:
            return ""

        # Step 1 — control characters
        text = self._CONTROL_CHARS.sub(" ", text)

        # Step 2 — PDF page markers ("--- Page 3 ---")
        text = self._PAGE_MARKER.sub("\n", text)

        # Step 3 — collapse triple+ newlines
        text = self._MULTI_NEWLINE.sub("\n\n", text)

        # Step 4 — collapse spaces/tabs
        text = self._MULTI_SPACE.sub(" ", text)

        # Step 5 — per-line strip
        lines = [line.strip() for line in text.split("\n")]
        text = "\n".join(lines)

        # Step 6 — full strip
        return text.strip()