"""Small helpers for Markdown tables and report files."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path


def table(header: Sequence[str], rows: Sequence[Sequence[object]]) -> str:
    """Render a GitHub-flavoured Markdown table."""
    widths = [max([len(str(h)), *(len(str(r[i])) for r in rows)]) for i, h in enumerate(header)]

    def line(cells: Sequence[object]) -> str:
        return "| " + " | ".join(str(c).ljust(w) for c, w in zip(cells, widths, strict=True)) + " |"

    sep = "|" + "|".join("-" * (w + 2) for w in widths) + "|"
    return "\n".join([line(header), sep, *(line(r) for r in rows)])


def write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def status(ok: bool) -> str:
    return "PASS" if ok else "FAIL"
