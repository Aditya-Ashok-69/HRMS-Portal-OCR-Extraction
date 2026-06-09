"""
Basic Aadhaar OCR using RapidOCR
Supports: JPG, PNG images and PDF files
"""

import sys
from pathlib import Path

import cv2
import numpy as np
from rapidocr_onnxruntime import RapidOCR

# For PDF support
try:
    import fitz  # PyMuPDF
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False
    print("[Warning] PyMuPDF not installed. PDF support disabled.")
    print("  Install with: pip install pymupdf")


def extract_from_image(image_path: str, engine: RapidOCR) -> list[dict]:
    """Extract text from a single image file."""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")

    result, _ = engine(img)

    if not result:
        return []

    # result format: [[box_coords, text, confidence], ...]
    extracted = []
    for item in result:
        box, text, confidence = item
        extracted.append({
            "text": text,
            "confidence": round(float(confidence), 3),
            "box": box,
        })
    return extracted


def extract_from_pdf(pdf_path: str, engine: RapidOCR, dpi: int = 200) -> dict[int, list[dict]]:
    """Extract text from all pages of a PDF."""
    if not PDF_SUPPORT:
        raise RuntimeError("PyMuPDF is required for PDF support. Run: pip install pymupdf")

    doc = fitz.open(pdf_path)
    pages_result = {}

    for page_num in range(len(doc)):
        page = doc[page_num]
        # Render page to image
        mat = fitz.Matrix(dpi / 72, dpi / 72)  # 72 is default PDF DPI
        pix = page.get_pixmap(matrix=mat)
        img_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)

        # Convert RGBA → BGR if needed
        if pix.n == 4:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2BGR)
        elif pix.n == 3:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

        result, _ = engine(img_array)
        extracted = []
        if result:
            for item in result:
                box, text, confidence = item
                extracted.append({
                    "text": text,
                    "confidence": round(float(confidence), 3),
                    "box": box,
                })
        pages_result[page_num + 1] = extracted  # 1-indexed pages

    doc.close()
    return pages_result


def print_results(results: list[dict] | dict, source: str):
    """Pretty-print OCR results."""
    print(f"\n{'='*50}")
    print(f"Source: {source}")
    print('='*50)

    if isinstance(results, list):
        # Single image
        if not results:
            print("No text detected.")
            return
        for item in results:
            print(f"[{item['confidence']:.2f}] {item['text']}")
        all_text = " ".join(i["text"] for i in results)
        print(f"\n--- Full Text ---\n{all_text}")

    elif isinstance(results, dict):
        # PDF pages
        for page_num, items in results.items():
            print(f"\n--- Page {page_num} ---")
            if not items:
                print("  No text detected.")
                continue
            for item in items:
                print(f"  [{item['confidence']:.2f}] {item['text']}")


def process_file(file_path: str):
    """Auto-detect file type and run OCR."""
    path = Path(file_path)
    if not path.exists():
        print(f"Error: File not found: {file_path}")
        sys.exit(1)

    engine = RapidOCR()
    ext = path.suffix.lower()

    if ext in (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"):
        results = extract_from_image(file_path, engine)
        print_results(results, path.name)

    elif ext == ".pdf":
        results = extract_from_pdf(file_path, engine)
        print_results(results, path.name)

    else:
        print(f"Unsupported file type: {ext}")
        sys.exit(1)


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python aadhaar_ocr.py <image_or_pdf_path>")
        print("  e.g. python aadhaar_ocr.py aadhaar.jpg")
        print("  e.g. python aadhaar_ocr.py aadhaar.pdf")
        sys.exit(1)

    process_file(sys.argv[1])