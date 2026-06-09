"""
document_processor.py
Central router — maps (file_path, doc_type) to the right extractor.

Supported doc_type values:
    "aadhaar"   → auto-detects image vs PDF
    "pan"       → auto-detects image vs PDF
    "payslip"   → auto-detects PDF vs image
"""

from pathlib import Path

from extract_id import (
    extract_from_aadhaar_image,
    extract_from_aadhaar_pdf,
    extract_from_pan_image,
    extract_from_pan_pdf,
    extract_from_payslip
)


_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
_PDF_EXT    = ".pdf"


def _is_image(path: str) -> bool:
    return Path(path).suffix.lower() in _IMAGE_EXTS


def _is_pdf(path: str) -> bool:
    return Path(path).suffix.lower() == _PDF_EXT


def process_document(file_path: str, doc_type: str) -> dict:
    """
    Route the file to the correct extractor based on doc_type and file extension.

    Args:
        file_path:  Absolute path to the uploaded file.
        doc_type:   One of "aadhaar", "pan", "payslip".

    Returns:
        A dict with extracted fields.  Always includes a "doc_type" key.
        On routing/unsupported-type errors an "error" key is added.
    """
    ext = Path(file_path).suffix.lower()

    try:
        if doc_type == "aadhaar":
            if _is_pdf(file_path):
                return extract_from_aadhaar_pdf(file_path)
            elif _is_image(file_path):
                # extract_from_aadhaar_image handles:
                #   - single front image
                #   - combined front+back (wide aspect ratio)
                #   - vertical combined (tall aspect ratio)
                return extract_from_aadhaar_image(file_path)
            else:
                return _unsupported(doc_type, ext)

        elif doc_type == "pan":
            if _is_pdf(file_path):
                return extract_from_pan_pdf(file_path)
            elif _is_image(file_path):
                return extract_from_pan_image(file_path)
            else:
                return _unsupported(doc_type, ext)

        elif doc_type == "payslip":
            # payslip_extractor handles both PDF and image internally
            return extract_from_payslip(file_path)

        else:
            return {
                "error": f"Unknown doc_type: '{doc_type}'",
                "doc_type": doc_type,
            }

    except Exception as exc:
        return {
            "error": str(exc),
            "doc_type": doc_type,
        }


def _unsupported(doc_type: str, ext: str) -> dict:
    return {
        "error": f"Unsupported file extension '{ext}' for doc_type '{doc_type}'.",
        "doc_type": doc_type,
    }