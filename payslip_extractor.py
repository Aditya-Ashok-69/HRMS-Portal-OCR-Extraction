import re
from pathlib import Path
from PIL import Image
import numpy as np
import pypdfium2 as pdfium

# Import shared OCR helpers from extract_id
from extract_id import paddle_ocr_image, preprocess_for_ocr

# UAN: 12-digit number starting with 1xx (EPFO format)
UAN_REGEX = re.compile(r"\b(1[0-9]{11})\b")

# Secondary: sometimes printed as "UAN No" or "UAN:" followed by the number
UAN_LABEL_REGEX = re.compile(
    r"UAN\s*(?:No\.?|Number|#)?\s*[:\-]?\s*([0-9]{12})",
    re.IGNORECASE,
)


def _extract_text_from_pdf(pdf_path: str) -> str:
    """Try native text extraction first; fall back to OCR per page."""
    pdf = pdfium.PdfDocument(pdf_path)
    native_text = ""
    try:
        for page in pdf:
            textpage = page.get_textpage()
            native_text += textpage.get_text_range() + "\n"
    finally:
        pdf.close()

    if len(native_text.strip()) > 50:
        return native_text

    # Native text too sparse — render pages and run OCR
    pdf = pdfium.PdfDocument(pdf_path)
    all_text = []
    try:
        for page in pdf:
            pil_image = page.render(scale=3).to_pil()
            processed = preprocess_for_ocr(pil_image)
            text = paddle_ocr_image(processed)
            all_text.append(text)
    finally:
        pdf.close()
    return "\n".join(all_text)


def _extract_text_from_image(image_path: str) -> str:
    img = Image.open(image_path).convert("RGB")
    processed = preprocess_for_ocr(img)
    return paddle_ocr_image(processed)


def _find_uan(text: str) -> str | None:
    # Try labelled match first (most reliable)
    m = UAN_LABEL_REGEX.search(text)
    if m:
        candidate = m.group(1)
        if len(candidate) == 12:
            return candidate

    # Fall back: any 12-digit sequence starting with 1
    matches = UAN_REGEX.findall(text)
    if matches:
        return matches[0]

    return None


def extract_from_payslip(file_path: str) -> dict:
    """
    Extract UAN number from a payslip PDF or image.

    Returns:
        {"uan": <str or None>, "doc_type": "payslip"}
    """
    ext = Path(file_path).suffix.lower()

    if ext == ".pdf":
        text = _extract_text_from_pdf(file_path)
    elif ext in {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}:
        text = _extract_text_from_image(file_path)
    else:
        return {"uan": None, "doc_type": "payslip", "error": f"Unsupported file type: {ext}"}

    uan = _find_uan(text)

    return {
        "uan": uan,
        "doc_type": "payslip",
    }