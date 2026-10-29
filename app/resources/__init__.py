"""Committed, non-model data files the stages read at runtime.

These are *reference tables*, not weights: they are small, text, licence-clean
and belong in the repository, which is exactly what makes them different from
everything under ``models/``.  An air-gapped node therefore needs no extra step
to get them.

Current contents:

``ifsc_master_sample.json``
    A **synthetic** IFSC master covering the five banks the committed corpus
    uses (OPEN-QUESTIONS Q4, option b).  It declares ``"authoritative": false``,
    which changes how :mod:`app.stages.s5_validators` treats a code it cannot
    find — see :class:`~app.stages.s5_validators.IFSCMaster`.  Dropping the real
    RBI list in is a path change, not a code change.

``bank_aliases.json``
    Display names and native-script aliases per bank code, used by the
    ``bank_name`` fuzzy matcher.  Separate from the IFSC file because the real
    RBI master carries branch rows and one bank spelling, not the Devanagari
    renderings an OCR'd complaint actually contains.

This module does NOT handle:
  * loading or parsing — the stage that uses a file owns its loader and its
    schema, so a malformed drop-in fails inside a typed loader rather than at
    import time;
  * downloading or refreshing anything.  Nothing here touches the network;
  * model weights of any kind (``models/``, gitignored).
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

__all__ = [
    "BANK_ALIASES_FILENAME",
    "IFSC_MASTER_SAMPLE_FILENAME",
    "RESOURCES_DIR",
    "bank_aliases_path",
    "ifsc_master_sample_path",
    "resource_path",
]

#: Directory this package lives in, resolved from ``__file__`` so the working
#: directory never matters (PLAN.md working agreements).
RESOURCES_DIR: Final[Path] = Path(__file__).resolve().parent

IFSC_MASTER_SAMPLE_FILENAME: Final[str] = "ifsc_master_sample.json"
BANK_ALIASES_FILENAME: Final[str] = "bank_aliases.json"


def resource_path(filename: str) -> Path:
    """Absolute path of a committed resource.

    Raises:
        FileNotFoundError: when the file is absent — a packaging bug, and one
            that must be loud rather than degrade into an empty table.
    """
    candidate = RESOURCES_DIR / filename
    if not candidate.is_file():
        raise FileNotFoundError(
            f"resource {filename!r} is missing from {RESOURCES_DIR}; the "
            "package was installed without its data files"
        )
    return candidate


def ifsc_master_sample_path() -> Path:
    """Path of the committed synthetic IFSC master (OPEN-QUESTIONS Q4)."""
    return resource_path(IFSC_MASTER_SAMPLE_FILENAME)


def bank_aliases_path() -> Path:
    """Path of the committed bank display-name/alias table."""
    return resource_path(BANK_ALIASES_FILENAME)
