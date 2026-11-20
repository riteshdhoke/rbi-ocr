"""Route handwritten raster pages directly to the VLM before Tier-1 OCR."""

from __future__ import annotations

from app.providers.base import ModelProvider, PageImage
from app.stages.s3_classify import estimate_handwriting_ratio
from app.stages.s4b_vlm import Tier2Result, escalate_page


def extract_handwritten_page(
    image: PageImage, *, provider: ModelProvider, min_ratio: float
) -> Tier2Result | None:
    """Use pixel-only detection and transcribe selected pages using vLLM.

    A result means the VLM owns recognition for this page, even if inference
    failed or returned no lines. The caller records that outcome.
    """
    if estimate_handwriting_ratio(image) < min_ratio:
        return None
    return escalate_page(image, (), provider=provider)
