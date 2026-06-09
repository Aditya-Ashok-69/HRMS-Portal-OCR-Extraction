"""
Aadhaar OCR + Structured JSON Extraction
Supports: JPG, PNG images and PDF files
Extraction: regex-based, no external API needed
"""

import json
import re
import sys
from pathlib import Path

import cv2
import numpy as np
from rapidocr_onnxruntime import RapidOCR

# ── PDF support (optional) ─────────────────────────────────────────────────────
try:
    import fitz  # PyMuPDF
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False
    print("[Warning] PyMuPDF not installed. PDF support disabled.")
    print("  Install with: pip install pymupdf")


# ══════════════════════════════════════════════════════════════════════════════
# OCR helpers
# ══════════════════════════════════════════════════════════════════════════════

def _ocr_image_array(img_array: np.ndarray, engine: RapidOCR) -> list[dict]:
    """Run OCR on a numpy BGR image array. Returns list of {text, confidence, box}."""
    result, _ = engine(img_array)
    if not result:
        return []
    extracted = []
    for item in result:
        box, text, confidence = item
        extracted.append({
            "text": text,
            "confidence": round(float(confidence), 3),
            "box": box,
        })
    return extracted


def extract_from_image(image_path: str, engine: RapidOCR) -> list[dict]:
    """Extract OCR blocks from a single image file."""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")
    return _ocr_image_array(img, engine)


def extract_from_pdf(pdf_path: str, engine: RapidOCR, dpi: int = 200) -> dict[int, list[dict]]:
    """Extract OCR blocks from every page of a PDF. Returns {page_number: [blocks]}."""
    if not PDF_SUPPORT:
        raise RuntimeError("PyMuPDF required. Install: pip install pymupdf")
    doc = fitz.open(pdf_path)
    pages = {}
    for page_num in range(len(doc)):
        page = doc[page_num]
        mat = fitz.Matrix(dpi / 72, dpi / 72)
        pix = page.get_pixmap(matrix=mat)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
        if pix.n == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        elif pix.n == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        pages[page_num + 1] = _ocr_image_array(img, engine)
    doc.close()
    return pages


# ══════════════════════════════════════════════════════════════════════════════
# Regex-based structured extraction
# ══════════════════════════════════════════════════════════════════════════════

# Noise tokens that are never part of a name
_NOISE_TOKENS = {
    'GOVERNMENTOFINDIA', 'GOVERNMENT', 'OFINDIA',
    'UNIQUEIDENTIFICATIONAUTHORITYOFINDIA', 'UNIQUEIDENTIFICATION', 'ORITYOFHINDIA',
    'ADHAAR', 'AADHAAR', 'ADDRESS', 'ADDRESS:', 'HRRR', 'HTTRTH', 'TATH',
    'INDIA', 'DOWNLOADDATE', 'AADHAAR-AAMADMIKADHIKAR',
}
_NOISE_RE     = re.compile(
    r'^(HRRR|HTTRTH|TATH|GOVERNMENT.*|UNIQUE.*|ADHAAR|AADHAAR.*|ORITYOFHINDIA)$', re.I
)

# Field patterns
_AADHAAR_RE   = re.compile(r'\b(\d{4}\s?\d{4}\s?\d{4})\b')
_DOB_RE       = re.compile(r'DOB[：:]\s*(\d{2}[/\-.]\d{2}[/\-.]\d{4})', re.IGNORECASE)
_GENDER_RE    = re.compile(r'\b(MALE|FEMALE)\b', re.IGNORECASE)
_MOBILE_RE    = re.compile(r'(?:MobileNo\.?|Mobile\s*No\.?|Mobile:?)\s*(\d{10})', re.IGNORECASE)
_VID_RE       = re.compile(r'VID[：:]\s*(\d{16})', re.IGNORECASE)
_DOWNLOAD_RE  = re.compile(r'DownloadDate\s*(\d{2}[/\-.]\d{2}[/\-.]\d{4})', re.IGNORECASE)
_PINCODE_RE   = re.compile(r'\b(\d{6})\b')
_RELATION_RE  = re.compile(r'\b([SDWCsdwc][/\\](?:[Oo]|0)\.?)([A-Za-z ]+?)(?=[,.\s]|$)')
_NAME_TOKEN_RE = re.compile(r'^[A-Z][a-zA-Z]{1,29}$')  # CamelCase or all-caps word


def _clean_address(raw: str) -> str:
    """Strip noise from the address string."""

    raw = re.sub(r'[^\x00-\x7F\u0900-\u097F\s,.\-/]+', '', raw)   # drop foreign garbage
    raw = re.sub(r'\b\d{12}\b', '', raw)                          # leaked aadhaar number
    raw = re.sub(r'(Aadhaar[-–].*|help@.*|www\..*)', '', raw, flags=re.IGNORECASE)

    # Remove D/O, S/O, W/O, C/O and the following name at the start
    raw = re.sub(
        r'^(?:D/O|S/O|W/O|C/O)\s*[A-Za-z\s.]+?(?=H\.?No\.?|House\s*No|Flat\s*No|\d)',
        '',
        raw,
        flags=re.IGNORECASE
    )

    return re.sub(r'\s{2,}', ' ', raw).strip(' ,')


def _extract_name(tokens: list[str], skip: set[str]) -> str | None:
    """
    Heuristic name extraction: first run of 1-3 consecutive tokens that
    look like a proper name, ignoring noise and already-captured values.
    """
    run = []
    for tok in tokens:
        clean = tok.strip('.,:/\\')
        if clean.upper() in _NOISE_TOKENS or _NOISE_RE.match(clean):
            run = []
            continue
        if clean in skip:
            run = []
            continue
        if _NAME_TOKEN_RE.match(clean):
            run.append(clean)
            if len(run) == 3:
                break
        else:
            if run:
                break
    return " ".join(run) if run else None


def parse_to_json(tokens: list[str]) -> dict:
    """
    Convert a flat list of OCR tokens into a structured Aadhaar dict.

    Keys returned (when found):
      name, dob, gender, aadhaar_number, vid,
      mobile, download_date,
      relation, relative_name,
      address, pincode
    """
    full = " ".join(tokens)
    result: dict = {}
    skip: set[str] = set()  # tokens already claimed by a specific field

    # ── Aadhaar number ──────────────────────────────────────────
    m = _AADHAAR_RE.search(full)
    if m:
        result["aadhaar_number"] = re.sub(r'\s', '', m.group(1))

    # ── VID ─────────────────────────────────────────────────────
    m = _VID_RE.search(full)
    if m:
        result["vid"] = m.group(1)

    # ── Date of birth ────────────────────────────────────────────
    m = _DOB_RE.search(full)
    if m:
        result["dob"] = m.group(1)

    # ── Gender ───────────────────────────────────────────────────
    m = _GENDER_RE.search(full)
    if m:
        result["gender"] = m.group(1).capitalize()

    # ── Mobile number ────────────────────────────────────────────
    m = _MOBILE_RE.search(full)
    if m:
        result["mobile"] = m.group(1)

    # ── Download date (e-Aadhaar) ────────────────────────────────
    m = _DOWNLOAD_RE.search(full)
    if m:
        result["download_date"] = m.group(1)

    # ── Relation + relative name (S/o, D/o, W/o, C/o) ───────────
    m = _RELATION_RE.search(full)
    if m:
        rel_map = {'s': "Son of", 'd': "Daughter of", 'w': "Wife of", 'c': "Care of"}
        result["relation"] = rel_map.get(m.group(1)[0].lower(), m.group(1))
        rel_name = m.group(2).strip().title()
        result["relative_name"] = rel_name
        skip.update(rel_name.split())

    # ── Address ──────────────────────────────────────────────────
    addr_m = re.search(r'Address[：:]?\s*(.+)', full, re.IGNORECASE)
    if addr_m:
        result["address"] = _clean_address(addr_m.group(1))
        pins = _PINCODE_RE.findall(result["address"])
        if pins:
            result["pincode"] = pins[-1]

    # ── Name (last — uses skip set to avoid collisions) ──────────
    name = _extract_name(tokens, skip)
    if name:
        result["name"] = name

    return result


# ══════════════════════════════════════════════════════════════════════════════
# Main pipeline
# ══════════════════════════════════════════════════════════════════════════════

def process_file(file_path: str) -> dict:
    """
    Full pipeline: OCR → token list → structured JSON dict.
    For PDFs, merges all pages into one token list before parsing.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    engine = RapidOCR()
    ext = path.suffix.lower()

    if ext in (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"):
        blocks = extract_from_image(file_path, engine)
        tokens = [b["text"] for b in blocks]

    elif ext == ".pdf":
        pages = extract_from_pdf(file_path, engine)
        tokens = [b["text"] for page_blocks in pages.values() for b in page_blocks]

    else:
        raise ValueError(f"Unsupported file type: {ext}")

    structured = parse_to_json(tokens)
    structured["_source"] = path.name
    return structured


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python aadhaar_ocr.py <image_or_pdf>")
        sys.exit(1)

    output = process_file(sys.argv[1])
    print(json.dumps(output, indent=2, ensure_ascii=False))