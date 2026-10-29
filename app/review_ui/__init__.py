"""Minimal reviewer pages — PRD P1-1, PLAN.md WS-G + WS-I item 1.

Two static pages and the assets they share, under :data:`STATIC_DIR`:

``test_page.html``
    The independent HTML test page (WS-I item 1).  Upload a document to
    ``POST /v1/documents``, then read the returned envelope as something other
    than a wall of JSON: bounce banner, per-page quality, extracted fields with
    their provenance, and — when the browser still holds the uploaded image —
    the evidence boxes drawn over the page.

``review.html``
    The keyboard-first review fragment (P1-1): one flagged field at a time,
    beside the crop of the page it came from, with accept/correct keys.

Both are plain HTML + vanilla JavaScript.  **No build step, no framework, no
CDN, no font, no icon, no analytics** — production is air-gapped (PLAN.md
"Air gap"), so every byte a page loads is a file in this directory.
``tests/unit/test_review_ui_static.py`` parses the pages and fails the build if
an absolute URL ever appears in one.

They work two ways with no code change: served by FastAPI (WS-F mounts
:data:`STATIC_DIR`), or opened straight from disk over ``file://`` with the API
base typed into the page.  The ``file://`` route needs the API to allow the
``null`` origin in CORS; when the fetch fails the page says so rather than
showing an empty result.

This module does NOT handle:
  * mounting, routing or serving — importing FastAPI here would make a
    static-asset lookup depend on the web framework; WS-F owns ``app/api``;
  * templating.  The pages are static; the envelope is fetched at runtime;
  * writing corrections back.  The published API contract has no review
    endpoint, so the review page **exports a decisions JSON file** and does not
    invent a route to POST it to;
  * rasterising PDFs.  A browser cannot without a library, and no library may
    be fetched, so PDF uploads render their envelope with the page-image panel
    disabled and a message saying why;
  * authentication of any kind.  These pages assume they are reached through
    whatever the deployment puts in front of the API.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

__all__ = [
    "REVIEW_PAGE_FILENAME",
    "REVIEW_UI_DIR",
    "SHARED_ASSET_FILENAMES",
    "STATIC_DIR",
    "TEST_PAGE_FILENAME",
    "review_page_path",
    "static_assets",
    "static_path",
    "test_page_path",
]

#: This package's directory, resolved from ``__file__`` so the CWD never
#: matters (PLAN.md working agreements).
REVIEW_UI_DIR: Final[Path] = Path(__file__).resolve().parent
#: The directory a web server mounts.
STATIC_DIR: Final[Path] = REVIEW_UI_DIR / "static"

TEST_PAGE_FILENAME: Final[str] = "test_page.html"
REVIEW_PAGE_FILENAME: Final[str] = "review.html"
#: Loaded by both pages; the reason they are two pages and not two copies.
SHARED_ASSET_FILENAMES: Final[tuple[str, ...]] = ("review_ui.css", "review_ui.js")


def static_path(filename: str) -> Path:
    """Absolute path of one static asset.

    Args:
        filename: a bare filename inside :data:`STATIC_DIR`.

    Raises:
        ValueError: when ``filename`` contains a path separator or ``..``.
            This function exists partly so that a future request handler has a
            resolver that cannot be walked out of the directory.
        FileNotFoundError: when the asset is missing — a packaging bug, and one
            that must be loud rather than serve a 404 nobody investigates.
    """
    if not filename or "/" in filename or "\\" in filename or filename in (".", ".."):
        raise ValueError(
            f"{filename!r} is not a bare filename inside the review-UI static "
            "directory; this resolver never traverses"
        )
    candidate = STATIC_DIR / filename
    if not candidate.is_file():
        raise FileNotFoundError(
            f"review-UI asset {filename!r} is missing from {STATIC_DIR}; the "
            "package was installed without its data files"
        )
    return candidate


def test_page_path() -> Path:
    """Path of the standalone API test page (WS-I item 1)."""
    return static_path(TEST_PAGE_FILENAME)


def review_page_path() -> Path:
    """Path of the keyboard-first review page (PRD P1-1)."""
    return static_path(REVIEW_PAGE_FILENAME)


def static_assets() -> tuple[Path, ...]:
    """Every file a deployment must ship for the pages to work offline."""
    names = (TEST_PAGE_FILENAME, REVIEW_PAGE_FILENAME, *SHARED_ASSET_FILENAMES)
    return tuple(static_path(name) for name in names)
