"""Download and print the character grid published in a Google Doc."""

from __future__ import annotations

import re
import sys
from typing import Sequence

import requests
from bs4 import BeautifulSoup


DEFAULT_URL = (
    "https://docs.google.com/document/d/e/"
    "2PACX-1vSvM5gDlNvt7npYHhp_XfsJvuntUhq184By5xO_pA4b_gCWeXb6dM6ZxwN8rE6S4ghUsCj2VKR21oEP/pub"
)


def _header_columns(cells: list[str]) -> tuple[int, int, int]:
    """Return the x, character, and y column indices from the table header."""
    labels = [re.sub(r"[^a-z]", "", value.lower()) for value in cells]
    try:
        x_index = next(i for i, label in enumerate(labels) if label in {"x", "xcoordinate"})
        char_index = next(i for i, label in enumerate(labels) if label in {"character", "char"})
        y_index = next(i for i, label in enumerate(labels) if label in {"y", "ycoordinate"})
    except StopIteration as exc:
        raise ValueError("table header must identify x-coordinate, Character, and y-coordinate") from exc
    if len(cells) != 3 or len({x_index, char_index, y_index}) != 3:
        raise ValueError("table header must contain exactly the x-coordinate, Character, and y-coordinate columns")
    return x_index, char_index, y_index


def decode_message(url: str) -> None:
    """Fetch the published document, validate points, and print its grid."""
    response = requests.get(
        url,
        timeout=20,
        headers={"User-Agent": "Mozilla/5.0 (compatible; decode-message/1.0)"},
    )
    response.raise_for_status()
    if not response.content or not response.text.strip():
        raise ValueError("the document response is empty")

    soup = BeautifulSoup(response.text, "html.parser")
    table = soup.find("table")
    if table is None:
        raise ValueError("the document contains no table")
    rows = table.find_all("tr")
    if not rows:
        raise ValueError("the document table contains no rows")

    header_cells = [cell.get_text(strip=True) for cell in rows[0].find_all("td")]
    x_col, char_col, y_col = _header_columns(header_cells)
    points: dict[tuple[int, int], str] = {}
    skipped = 0
    parsed = 0

    for row_number, row in enumerate(rows[1:], start=2):
        cells = [cell.get_text(strip=True) for cell in row.find_all("td")]
        reason = ""
        if len(cells) != 3:
            reason = f"expected exactly 3 cells, found {len(cells)}"
        else:
            x_text, character, y_text = cells[x_col], cells[char_col], cells[y_col]
            if not re.fullmatch(r"\d+", x_text):
                reason = "x-coordinate is not a non-negative integer"
            elif not re.fullmatch(r"\d+", y_text):
                reason = "y-coordinate is not a non-negative integer"
            elif len(character) != 1:
                reason = "character must be exactly one character"
            else:
                parsed += 1
                coordinate = (int(x_text), int(y_text))
                if coordinate in points:
                    print(
                        f"Warning: row {row_number}: duplicate coordinate {coordinate}; keeping last value",
                        file=sys.stderr,
                    )
                points[coordinate] = character
        if reason:
            skipped += 1
            print(f"Warning: row {row_number}: {reason}", file=sys.stderr)

    if not points:
        raise ValueError("the document table contains no valid points")

    width = max(x for x, _ in points) + 1
    height = max(y for _, y in points) + 1
    grid = [[" " for _ in range(width)] for _ in range(height)]
    for (x, y), character in points.items():
        grid[y][x] = character

    for line in reversed(grid):
        print("".join(line).rstrip())
    print(f"Data rows parsed: {parsed}; skipped: {skipped}; grid: {width}x{height}", file=sys.stderr)


def main(argv: Sequence[str]) -> int:
    # The published picture may use Unicode block characters on Windows consoles.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    url = argv[1] if len(argv) > 1 else DEFAULT_URL
    try:
        decode_message(url)
    except requests.exceptions.HTTPError as exc:
        status = exc.response.status_code if exc.response is not None else "unknown"
        print(f"Error: HTTP request failed with status {status}", file=sys.stderr)
        return 1
    except requests.exceptions.Timeout:
        print("Error: request timed out", file=sys.stderr)
        return 1
    except requests.exceptions.ConnectionError:
        print("Error: could not connect to the document (network may be blocked)", file=sys.stderr)
        return 1
    except requests.exceptions.RequestException as exc:
        print(f"Error: request failed: {exc}", file=sys.stderr)
        return 1
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
