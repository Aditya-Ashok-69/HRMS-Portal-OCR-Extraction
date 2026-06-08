# extractors/pan.py
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from extract_id import (
    extract_from_pan_image,
    extract_from_pan_pdf,
)


def extract_pan(file_path: str, sub_type: str = "image") -> dict:
    """
    sub_type options:
      "image" → PAN card image
      "pdf"   → PAN PDF
    """
    if sub_type == "pdf":
        return extract_from_pan_pdf(file_path)
    else:
        return extract_from_pan_image(file_path)