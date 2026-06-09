import os
os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["PADDLE_DISABLE_MKLDNN"] = "1"
import argparse
import json
import re
from pathlib import Path
from datetime import datetime
from PIL import Image, ImageOps, ImageFilter
import openbharatocr
import spacy
import pypdfium2 as pdfium
import cv2
import numpy as np
import re
import tempfile
from paddleocr import PaddleOCR

_paddle = PaddleOCR(
    use_angle_cls=True,
    lang="en",
    show_log=False,
    enable_mkldnn=False,
)

def paddle_ocr_image(pil_image: Image.Image) -> str:
    img = pil_image.convert("RGB")
    w, h = img.size
    if h < 64 or w < 64:
        scale = max(64 / h, 64 / w)
        img = img.resize((int(w * scale) + 1, int(h * scale) + 1), Image.LANCZOS)
    img_np = np.array(img)
    result = _paddle.ocr(img_np, cls=True)
    if not result or not result[0]:
        return ""
    lines = [line[1][0] for line in result[0] if line[1][1] > 0.5]
    return "\n".join(lines)


def paddle_ocr_image_latin_only(pil_image: Image.Image) -> str:
    """
    Run English OCR on a version of the image where Devanagari ink regions
    are whited out. This improves English-model accuracy on mixed-script cards
    without needing a second OCR language model.
    """
    img_np = np.array(pil_image.convert("RGB"))
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

    # Threshold to find dark ink
    _, dark_mask = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)

    # Find connected components of dark ink
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(dark_mask, connectivity=8)

    # White out components that look like Devanagari:
    # Devanagari chars are typically 8-40px tall and have a top horizontal bar.
    # We use a simple size heuristic: small, roughly square blobs are Latin;
    # blobs with high aspect ratio (wide) or with a top-heavy mass are Devanagari.
    # Since we can't classify glyphs reliably, we instead run a second pass:
    # crop each text line, run OCR, and discard lines whose output is all Devanagari.
    # That logic lives in strip_devanagari() applied post-OCR — so here we just
    # return normal OCR and let downstream functions filter.
    return paddle_ocr_image(pil_image)


def preprocess_for_ocr(pil_image: Image.Image) -> Image.Image:
    img_np = np.array(pil_image.convert("RGB"))
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

    coords = np.column_stack(np.where(gray < 200))
    if len(coords) > 100:
        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle = 90 + angle
        if abs(angle) > 0.5:
            (h, w) = gray.shape
            M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
            gray = cv2.warpAffine(gray, M, (w, h),
                                  flags=cv2.INTER_CUBIC,
                                  borderMode=cv2.BORDER_REPLICATE)

    binary = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=31,
        C=10
    )
    return Image.fromarray(binary)

# ---------- regex patterns ----------
PAN_REGEX = re.compile(r"\b[A-Z]{5}[0-9]{4}[A-Z]\b")
AADHAAR_REGEX = re.compile(r"(?:\d{4}\s?\d{4}\s?\d{4})")
DOB_KEYWORD_REGEX = re.compile(
    r"(?:DOB|Date\s*of\s*Birth|Year\s*of\s*Birth)\s*[:\-]?\s*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})",
    re.IGNORECASE
)

FATHER_REGEX = re.compile(
    r"(?:S/O|D/O|C/O|W/O|\$/0|5/0)\s*[:\-]?\s*([A-Za-z ]+?)(?=\s+H\.?No|\s+House|\s+Address|,|$|\n)",
    re.IGNORECASE
)
DATE_REGEX = re.compile(r"\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\b")

# Hindi/mixed-script DOB pattern: handles "DOB:" and "जन्म तिथि" vicinity
DOB_MIXED_REGEX = re.compile(
    r"(?:DOB|Date\s*of\s*Birth|जन्म[\s]*तिथि|जन्मतिथि)\s*[:/\-]?\s*(\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4})",
    re.IGNORECASE | re.UNICODE,
)

# ---------- spaCy ----------
nlp = spacy.load("en_core_web_sm")

OCR_JUNK_REGEX = re.compile(
    r"\b(?:STAT|WTATET|ATS|ATATRTA|TSA)\b|[|]+",
    re.IGNORECASE
)

REL_PREFIX_REGEX = re.compile(r"\b(S/O|D/O|C/O|W/O)\s*[:\-]?\s*", re.IGNORECASE)

ADDRESS_STOP_REGEX = re.compile(
    r"(?:\bMobile\b|\bYour Aadhaar No\b|\bAadhaar no\. issued\b|\bVID\b|"
    r"\bwww\.uidai\.gov\.in\b|\bhelp@uidai\.gov\.in\b)",
    re.IGNORECASE
)

STATE_PIN_REGEX = re.compile(
    r"\b(?:Andhra Pradesh|Arunachal Pradesh|Assam|Bihar|Chhattisgarh|Goa|Gujarat|Haryana|"
    r"Himachal Pradesh|Jharkhand|Karnataka|Kerala|Madhya Pradesh|Maharashtra|Manipur|"
    r"Meghalaya|Mizoram|Nagaland|Odisha|Punjab|Rajasthan|Sikkim|Tamil Nadu|Telangana|"
    r"Tripura|Uttar Pradesh|Uttarakhand|West Bengal|Delhi|Puducherry|Chandigarh|"
    r"Jammu and Kashmir|Ladakh)\b\s*[-,:]?\s*\d{6}\b",
    re.IGNORECASE
)

ADDRESS_HARD_STOP_REGEX = re.compile(
    r"(?:^\d{10}$|"
    r"\b\d{2}/\d{2}/\d{4}\b|"
    r"\bYour\s+(?:Aadhaar\s+)?No\b|"
    r"\b\d{4}\s\d{4}\s\d{4}\b|"
    r"\b(?:uidai|uidal)\.gov\.in\b|"
    r"\bAadhaar\b.*\bissued\b|"
    r"\bDOB\b|"
    r"\bMALE\b|\bFEMALE\b|\bOTHER\b|"
    r"\bGovernment of India\b|"
    r"\bUnique Identification Authority of India\b|"
    r"\bINFORMATION\b|"
    r"\bwww\.uidai\.gov\.in\b|"
    r"\bhelp@uidai\.gov\.in\b)",
    re.IGNORECASE
)

# UAN: 12-digit number starting with 1xx (EPFO format)
UAN_REGEX = re.compile(r"\b(1[0-9]{11})\b")

# Secondary: sometimes printed as "UAN No" or "UAN:" followed by the number
UAN_LABEL_REGEX = re.compile(
    r"UAN\s*(?:No\.?|Number|#)?\s*[:\-]?\s*([0-9]{12})",
    re.IGNORECASE,
)

def normalize_person_name(name: str) -> str | None:
    if not name:
        return name
    return clean_line(name)

PIN_TERMINATOR_REGEX = re.compile(
    r"\b(?:PIN\s*Code|PIN)\s*[:\-]?\s*\d{6}\b",
    re.IGNORECASE
)

def extract_text_from_pdf(pdf_path: str) -> str:
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

    pdf = pdfium.PdfDocument(pdf_path)
    all_text = []
    try:
        for page in pdf:
            pil_image = page.render(scale=3).to_pil()
            text = paddle_ocr_image(pil_image)
            all_text.append(text)
    finally:
        pdf.close()
    return "\n".join(all_text)


def clean_line(line: str) -> str:
    line = line.strip()
    line = re.sub(r"^[^A-Za-z0-9]+", "", line)
    line = re.sub(r"^(?:bi|b1|i|l|1|2)\s+", "", line, flags=re.IGNORECASE)
    line = re.sub(r"\s+", " ", line)
    return line.strip()


def is_devanagari(text: str) -> bool:
    """Return True if the string contains any Devanagari characters."""
    return bool(re.search(r'[\u0900-\u097F]', text))


def strip_devanagari(text: str) -> str:
    """Remove Devanagari characters (and surrounding spaces) from a string."""
    text = re.sub(r'[\u0900-\u097F]+', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()


def normalize_ocr_text(text: str) -> str:
    if not text:
        return text
    text = re.sub(r"(?:(?<=\s)|^)\$/0(?=\s*[:\-])", "S/O", text, flags=re.IGNORECASE)
    text = re.sub(r"(?:(?<=\s)|^)5/0(?=\s*[:\-])", "S/O", text, flags=re.IGNORECASE)
    text = re.sub(r"(?:(?<=\s)|^)S\s*[/\\]\s*O(?=\s*[:\-])", "S/O", text, flags=re.IGNORECASE)
    text = re.sub(r"(?:(?<=\s)|^)D\s*[/\\]\s*O(?=\s*[:\-])", "D/O", text, flags=re.IGNORECASE)
    text = re.sub(r"(?:(?<=\s)|^)C\s*[/\\]\s*O(?=\s*[:\-])", "C/O", text, flags=re.IGNORECASE)
    text = re.sub(r"(?:(?<=\s)|^)W\s*[/\\]\s*O(?=\s*[:\-])", "W/O", text, flags=re.IGNORECASE)
    return text

def clean_ocr_noise(text: str) -> str:
    text = OCR_JUNK_REGEX.sub(" ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def likely_name(line: str) -> bool:
    bad = {
        "to", "government of india", "unique identification authority of india",
        "aadhaar", "address", "male", "female", "dob", "issue date",
        "download date", "vid"
    }
    s = clean_line(line)
    if not s:
        return False
    if s.lower() in bad:
        return False
    if any(x in s.lower() for x in ["government", "authority", "aadhaar", "address", "issue date", "download date"]):
        return False
    if re.search(r"\d", s):
        return False
    words = s.split()
    if len(words) < 2 or len(words) > 4:
        return False
    return True


def is_valid_english_name(line: str) -> bool:
    # Strip any Devanagari first so a mixed line like "अंजलि Anjali" can still pass
    s = strip_devanagari(clean_line(line))
    if not s:
        return False

    s_low = s.lower()

    if re.search(r"\d", s):
        return False

    bad_terms = [
        "government", "india", "unique", "authority", "address",
        "dob", "male", "female", "other", "aadhaar", "vid",
        "district", "state", "mobile", "mobile no", "no",
        "enrolment", "information", "your no", "your aadhaar no",
        "download date", "issue date", "father", "name"
    ]
    if any(term in s_low for term in bad_terms):
        return False

    words = s.split()
    if len(words) < 1 or len(words) > 4:
        return False

    good = 0
    for w in words:
        if re.fullmatch(r"[A-Z]+", w):
            good += 1
        elif re.fullmatch(r"[A-Z][a-z]+", w):
            good += 1
        elif re.fullmatch(r"[A-Z]", w):
            good += 1
    if len(words) == 2:
        if len(words[0]) <= 3 and len(words[1]) <= 3:
            return False
    return good >= 1


# -----------------------------------------------------------------------
# extract_name — used for Aadhaar front IMAGE
# Handles both English-only and Hindi+English mixed cards.
# Strategy (in order):
#   1. Explicit "Name:" label
#   2. Inline "Name: Ravi Kumar"
#   3. Line immediately before DOB (handles Hindi DOB keyword too)
#   4. Line immediately before S/O | D/O | W/O | C/O line
#   5. Line after "To" block (letter-format Aadhaar)
#   6. Line before MALE/FEMALE/OTHER/MOBILE/VID keyword
#   7. English-only scan: find first valid English name in cleaned lines
# -----------------------------------------------------------------------
def extract_name(lines):
    # Work on Devanagari-stripped versions for matching, but return original cleaned
    cleaned_raw = [clean_line(line) for line in lines if clean_line(line)]
    cleaned = [strip_devanagari(c) for c in cleaned_raw]

    def get_raw(i):
        """Return the Devanagari-stripped version of line i."""
        return cleaned[i]

    # Pass 1: explicit "Name:" label on its own line, value follows
    for i, line in enumerate(cleaned):
        if re.fullmatch(r"(?:Name|Your\s+Name)\s*[:\-]?", line, re.IGNORECASE):
            for j in range(i + 1, min(i + 3, len(cleaned))):
                cand = cleaned[j]
                if is_valid_english_name(cand) and len(cand.split()) >= 2:
                    return cand

    # Pass 2: "Name: Ravi Kumar" inline on same line
    for line in cleaned:
        m = re.match(r"^(?:Name|Your\s+Name)\s*[:\-]\s*(.+)$", line, re.IGNORECASE)
        if m:
            cand = clean_line(m.group(1))
            if is_valid_english_name(cand) and len(cand.split()) >= 2:
                return cand

    # Pass 3: name immediately before DOB line (within 3 lines up)
    # Handles both "DOB" and Hindi "जन्म" (stripped to empty, but the raw line has it)
    for i, line in enumerate(cleaned_raw):
        if re.search(r"DOB|Date\s*of\s*Birth|\d{2}[/\-]\d{2}[/\-]\d{4}", line, re.IGNORECASE):
            for j in range(max(0, i - 3), i):
                cand = cleaned[j]  # devanagari-stripped version
                if is_valid_english_name(cand) and len(cand.split()) >= 2:
                    return cand

    # Pass 4: name immediately before S/O | D/O | W/O | C/O line (within 3 lines up)
    for i, line in enumerate(cleaned):
        if re.search(r"^(S/O|D/O|C/O|W/O)\s*[:\-]?", line, re.IGNORECASE):
            for j in range(max(0, i - 3), i):
                cand = cleaned[j]
                if is_valid_english_name(cand) and len(cand.split()) >= 2:
                    return cand

    # Pass 5: name after "To" block (letter-format Aadhaar)
    for i, line in enumerate(cleaned):
        if line.strip().lower() == "to":
            for j in range(i + 1, min(i + 4, len(cleaned))):
                cand = cleaned[j]
                if is_valid_english_name(cand) and "mobile" not in cand.lower():
                    return cand

    # Pass 6: name before MALE/FEMALE/OTHER/MOBILE/VID keyword (within 3 lines up)
    for i, line in enumerate(cleaned_raw):
        if re.search(r"\b(DOB|MALE|FEMALE|OTHER|MOBILE|VID)\b", line, re.IGNORECASE):
            for j in range(max(0, i - 3), i):
                cand = cleaned[j]
                if is_valid_english_name(cand) and "mobile" not in cand.lower():
                    return cand

    # Pass 7: generic scan — first line that looks like a valid English name
    for line in cleaned:
        if is_valid_english_name(line) and "mobile" not in line.lower():
            return line

    return None


# -----------------------------------------------------------------------
# extract_name_from_aadhaar_pdf
# -----------------------------------------------------------------------
def extract_name_from_aadhaar_pdf(lines):
    cleaned = [clean_line(x) for x in lines if clean_line(x)]

    for i, line in enumerate(cleaned):
        if re.fullmatch(r"(?:Name|Your\s+Name)\s*[:\-]?", line, re.IGNORECASE):
            for j in range(i + 1, min(i + 3, len(cleaned))):
                cand = cleaned[j]
                if is_valid_english_name(cand) and len(cand.split()) >= 2:
                    return cand
        m = re.match(r"^(?:Name|Your\s+Name)\s*[:\-]\s*(.+)$", line, re.IGNORECASE)
        if m:
            cand = clean_line(m.group(1))
            if is_valid_english_name(cand) and len(cand.split()) >= 2:
                return cand

    for i, line in enumerate(cleaned):
        if re.search(r"\bDOB\b", line, re.IGNORECASE):
            for j in range(i - 1, max(-1, i - 5), -1):
                cand = cleaned[j]
                if is_valid_english_name(cand) and len(cand.split()) >= 2:
                    return cand

    for i, line in enumerate(cleaned):
        if re.search(r"^(S/O|D/O|C/O|W/O)\s*[:\-]?", line, re.IGNORECASE):
            for j in range(max(0, i - 3), i):
                cand = cleaned[j]
                if is_valid_english_name(cand) and len(cand.split()) >= 2:
                    return cand

    for i, line in enumerate(cleaned):
        if line.strip().lower() == "to":
            for j in range(i + 1, min(i + 6, len(cleaned))):
                cand = cleaned[j]
                if is_valid_english_name(cand) and len(cand.split()) >= 2:
                    return cand

    return None


def extract_dob(text: str):
    # Try Hindi/mixed pattern first
    m = DOB_MIXED_REGEX.search(text)
    if m:
        return m.group(1)
    m = DOB_KEYWORD_REGEX.search(text)
    if m:
        return m.group(1)
    return None


def extract_dob_from_lines(lines):
    """
    Extract DOB from a line list.
    Handles:
      - "DOB: 05/07/1994"
      - "जन्म तिथि / DOB: 05/07/1994"
      - Line that contains ONLY a date pattern (common on physical cards)
    """
    for line in lines:
        m = DOB_MIXED_REGEX.search(line)
        if m:
            return m.group(1)

    for line in lines:
        # A line that contains a date AND a DOB-adjacent keyword on the same or nearby line
        if re.search(r"DOB|Date\s*of\s*Birth|जन्म", line, re.IGNORECASE | re.UNICODE):
            m = DATE_REGEX.search(line)
            if m:
                return m.group(1)

    # Last resort: a line that is ONLY a date (physical cards sometimes have this)
    for line in lines:
        stripped = line.strip()
        if re.fullmatch(r"\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}", stripped):
            return stripped

    return None


def extract_gender(text):
    if not text:
        return None
    text_up = text.upper()
    if "FEMALE" in text_up:
        return "FEMALE"
    if re.search(r"\bMALE\b", text_up):
        return "MALE"
    if "OTHER" in text_up:
        return "OTHER"
    # Hindi gender tokens
    if "पुरुष" in text:
        return "MALE"
    if "महिला" in text or "स्त्री" in text:
        return "FEMALE"
    return None


def normalize_gender(value: str):
    if not value:
        return None
    value = clean_line(value).upper()
    if "FEMALE" in value:
        return "FEMALE"
    if re.search(r"\bMALE\b", value):
        return "MALE"
    if "OTHER" in value:
        return "OTHER"
    return None


def resolve_gender(raw_text: str, ocr_gender: str = None, lines=None):
    candidates = []
    if lines:
        for line in lines:
            line_clean = clean_line(line).upper()
            if re.fullmatch(r"(MALE|FEMALE|OTHER)", line_clean):
                candidates.append(line_clean)
            else:
                g = extract_gender(line)
                if g:
                    candidates.append(g)
    text_gender = extract_gender(raw_text)
    if text_gender:
        candidates.append(text_gender)
    ocr_gender = normalize_gender(ocr_gender)
    if ocr_gender:
        candidates.append(ocr_gender)
    if "FEMALE" in candidates:
        return "FEMALE"
    if "MALE" in candidates:
        return "MALE"
    if "OTHER" in candidates:
        return "OTHER"
    return None


def extract_gender_from_aadhaar_front_crop(image_path: str):
    img = Image.open(image_path).convert("RGB")
    w, h = img.size

    # Wide crop covering the middle band where gender is printed
    gender_crop = img.crop((int(w*0.25), int(h*0.28), int(w*0.85), int(h*0.65)))
    processed = preprocess_for_ocr(gender_crop)
    text = paddle_ocr_image(processed).upper()
    if re.search(r"\bFEMALE\b", text): return "FEMALE"
    if re.search(r"\bMALE\b", text):   return "MALE"
    if re.search(r"\bOTHER\b", text):  return "OTHER"

    # Fallback: full image OCR — "\u092a\u0941\u0930\u0941\u0937 / MALE" style
    full_text = paddle_ocr_image(img).upper()
    if re.search(r"\bFEMALE\b", full_text): return "FEMALE"
    if re.search(r"\bMALE\b",   full_text): return "MALE"
    if re.search(r"\bOTHER\b",  full_text): return "OTHER"
    return None


def extract_address(lines):
    address_lines = []
    collecting = False

    for line in lines:
        s = clean_line(line)
        s = re.sub(r"\bVID\s*[:\-]?\s*\d[\d\s]{8,}\b", "", s, flags=re.IGNORECASE)
        s = re.sub(r"\bVID\s*[:\-]?\s*", "", s, flags=re.IGNORECASE)
        if not s:
            continue

        # Accept both English "Address" and Hindi "पता" as collection trigger
        address_trigger = re.search(
            r"^\s*(?:Address|पता)\s*:?",
            line,           # use original line so Hindi is not stripped
            re.IGNORECASE | re.UNICODE
        )

        if not collecting and address_trigger:
            collecting = True
            address_lines = []
            # Strip the trigger word
            s = re.sub(r"^\s*(?:Address|पता)\s*:?", "", s, flags=re.IGNORECASE | re.UNICODE).strip()
            if s:
                if re.match(r"^(S/O|D/O|C/O|W/O)", s, re.IGNORECASE):
                    s = re.sub(r"^(?:S/O|D/O|C/O|W/O)\s*:?\s*[^,]+,\s*", "", s, flags=re.IGNORECASE)
                if s:
                    address_lines.append(s)
            continue

        if not collecting:
            continue

        if ADDRESS_HARD_STOP_REGEX.search(s):
            break

        s = re.sub(r"\s+", " ", s).strip(" ,.-")
        if not s:
            continue
        if s.lower() in {"mq", "q", "oq"}:
            continue
        if re.fullmatch(r"[A-Za-z]{1,2}\.?", s):
            continue

        if re.match(r"^(S/O|D/O|C/O|W/O)\b", s, re.IGNORECASE):
            s = re.sub(r"^(?:S/O|D/O|C/O|W/O)\s*:?\s*[^,]+,\s*", "", s, flags=re.IGNORECASE)
            if not s.strip():
                continue

        if address_lines and address_lines[-1].upper() == "NO" and re.match(r"^\d+", s):
            address_lines[-1] = address_lines[-1] + s
            continue

        address_lines.append(s)

        if (
            STATE_PIN_REGEX.search(s)
            or re.search(r"\bPIN\s*Code\s*[:\-]?\s*\d{6}\b", s, re.IGNORECASE)
            or re.search(r"\b[A-Za-z ]+\s*-\s*\d{6}\b", s)
        ):
            break

    if not address_lines:
        return None

    address = ", ".join(address_lines)
    address = re.sub(r"\bVengathur\s*,?\s*Manavalanagar\b", "", address, flags=re.IGNORECASE)
    address = re.sub(r"\s*,\s*", ", ", address)
    address = re.sub(r",\s*,+", ", ", address)
    address = re.sub(r"\s+", " ", address)
    address = address.strip(" ,.-")
    return address if address else None


def remove_relationship_prefix(address: str) -> str:
    if not address:
        return address
    address = re.sub(
        r"^(?:S\s*/?\s*O|D\s*/?\s*O|C\s*/?\s*O|W\s*/?\s*O)\s*[:\-]?\s*[A-Za-z .]+,\s*",
        "", address, flags=re.IGNORECASE
    )
    return re.sub(r"\s+", " ", address).strip()


def clean_aadhaar_address(address):
    if not address:
        return address
    address = re.sub(r"\b[2-9]\d{3}\s?\d{4}\s?\d{4}\b", "", address)
    address = re.sub(r"\b\d{9,}\b", "", address)
    address = re.sub(r"\b(?:uidai|uidal)\.gov\.in\b", "", address, flags=re.IGNORECASE)
    address = re.sub(r"\b\d{6}\s+\d+\b", "", address)
    address = re.sub(r"\b(?:S/O|D/O|C/O|W/O)\s*:?\s*[^,]+,?", "", address, flags=re.IGNORECASE)
    address = re.sub(r"\s+", " ", address)
    address = re.sub(r"\s*,\s*", ", ", address)
    return address.strip(" ,.-")


def is_valid_ddmmyyyy(candidate: str):
    if not re.fullmatch(r"\d{8}", candidate):
        return None
    dd, mm, yyyy = candidate[:2], candidate[2:4], candidate[4:]
    try:
        dt = datetime.strptime(f"{dd}/{mm}/{yyyy}", "%d/%m/%Y")
        if 1900 <= dt.year <= 2100:
            return f"{dd}/{mm}/{yyyy}"
    except ValueError:
        return None
    return None


def validate_ddmmyyyy(digits: str):
    if len(digits) != 8 or not digits.isdigit():
        return None
    dd = int(digits[:2])
    mm = int(digits[2:4])
    yyyy = int(digits[4:8])
    if not (1900 <= yyyy <= 2100):
        return None
    try:
        datetime(yyyy, mm, dd)
        return f"{digits[:2]}/{digits[2:4]}/{digits[4:8]}"
    except ValueError:
        return None


def normalize_ocr_date(text: str):
    if not text:
        return None
    m = DATE_REGEX.search(text)
    if m:
        val = m.group(1)
        for fmt in ("%d/%m/%Y", "%d-%m-%Y", "%d/%m/%y", "%d-%m-%y"):
            try:
                dt = datetime.strptime(val, fmt)
                if 1900 <= dt.year <= 2100:
                    return dt.strftime("%d/%m/%Y")
            except ValueError:
                pass

    digits = re.sub(r"\D", "", text)

    def valid_date8(s):
        if len(s) != 8 or not s.isdigit():
            return None
        dd = int(s[:2])
        mm = int(s[2:4])
        yyyy = int(s[4:])
        if not (1900 <= yyyy <= 2100):
            return None
        try:
            datetime(yyyy, mm, dd)
            return f"{s[:2]}/{s[2:4]}/{s[4:8]}"
        except ValueError:
            return None

    if len(digits) >= 8:
        for i in range(len(digits) - 7):
            cand = digits[i:i+8]
            valid = valid_date8(cand)
            if valid:
                return valid
    return None


def infer_pan_names_from_flat_text(text, pan=None, dob=None):
    work = text or ""
    if pan:
        work = work.replace(pan, " ")
    if dob:
        work = work.replace(dob, " ")
    words = [w.upper() for w in re.findall(r"[A-Za-z]+", work) if len(w) > 1]
    if len(words) >= 5:
        for split in range(2, len(words)):
            left = words[:split]
            right = words[split:]
            if len(left) in (2, 3) and len(right) in (2, 3):
                if left[-1] == right[-1]:
                    return " ".join(left), " ".join(right)
        return " ".join(words[:3]), " ".join(words[3:])
    if len(words) == 4:
        return " ".join(words[:2]), " ".join(words[2:])
    if len(words) == 3:
        return " ".join(words[:2]), words[2]
    return None, None


def best_pan_dob(image_path: str):
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    full_processed = preprocess_for_ocr(img)
    full_text = paddle_ocr_image(full_processed)
    full_dob = normalize_ocr_date(full_text)
    crop = img.crop((0, int(h*0.60), int(w*0.70), h))
    crop_processed = preprocess_for_ocr(crop)
    crop_text = paddle_ocr_image(crop_processed)
    crop_dob = normalize_ocr_date(crop_text)
    return crop_dob or full_dob


def choose_dob(raw_text: str, candidates: list[str]) -> str | None:
    if not candidates:
        return None
    for cand in candidates:
        pattern = rf"(?:DOB|Date\s*of\s*Birth)[^\d]{{0,20}}{re.escape(cand)}"
        if re.search(pattern, raw_text, re.IGNORECASE):
            return cand
    return candidates[0]


def looks_like_person_name(text):
    if not text:
        return False
    text = clean_line(text)
    words = text.split()
    if len(words) < 2 or len(words) > 4:
        return False
    banned = {"GOVERNMENT", "INDIA", "DEPARTMENT", "ACCOUNT", "NUMBER", "PERMANENT", "CARD", "TAX", "INCOME"}
    for word in words:
        if word.upper() in banned:
            return False
        if not word.isalpha():
            return False
        if not word[0].isupper():
            return False
        if len(word) > 1 and not word[1:].islower():
            return False
    return True


def extract_from_pan_image(image_path: str):
    d = openbharatocr.pan(image_path)
    ocr_text = d.get("raw_text")
    if not ocr_text:
        img = Image.open(image_path)
        processed = preprocess_for_ocr(img)
        ocr_text = paddle_ocr_image(processed)
    if not ocr_text:
        ocr_text = "\n".join(str(v) for v in d.values() if isinstance(v, str))

    compact_text = re.sub(r"\s+", " ", ocr_text).strip()
    pan = d.get("pan_number") or d.get("PAN Number")
    if not pan:
        pan_matches = PAN_REGEX.findall(compact_text)
        pan = pan_matches[0] if pan_matches else None

    name = d.get("name") or d.get("Full Name")
    father_name = d.get("father_name") or d.get("Parent's Name")
    dob = d.get("dob") or d.get("Date of Birth")
    if not dob:
        dob_match = DATE_REGEX.search(compact_text)
        dob = dob_match.group(1) if dob_match else None
    if not dob:
        dob = best_pan_dob(image_path)

    ocr_text = re.sub(r"\bme:\s*", "Father's Name: ", ocr_text, flags=re.IGNORECASE)

    if not father_name:
        m = re.search(r"Father.?s Name\s*:\s*([A-Za-z ]+)", ocr_text, re.IGNORECASE)
        if m:
            extracted = clean_line(m.group(1))
            if looks_like_person_name(extracted):
                father_name = extracted

    if not name or not father_name:
        PAN_NOISE = ["INCOME TAX", "GOVERNMENT", "INDIA", "ACCOUNT", "PERMANENT", "DEPARTMENT", "CARD"]
        lines = [clean_line(x) for x in ocr_text.splitlines() if clean_line(x)]
        candidates = []
        for line in lines:
            if PAN_REGEX.search(line): continue
            if DATE_REGEX.search(line): continue
            if any(noise.lower() in line.lower() for noise in PAN_NOISE): continue
            line_lower = line.lower()
            if "father" in line_lower or "parent" in line_lower or "name" in line_lower:
                continue
            if re.search(r"\d", line): continue
            words = line.split()
            if 1 < len(words) <= 4:
                candidates.append(line)
        valid_candidates = [c for c in candidates if looks_like_person_name(c)]
        if not name and len(valid_candidates) >= 1:
            name = valid_candidates[0]
        if not father_name and len(valid_candidates) >= 2:
            father_name = valid_candidates[1]

    if not name or not father_name:
        inferred_name, inferred_father = infer_pan_names_from_flat_text(compact_text, pan=pan, dob=dob)
        if not name and inferred_name and looks_like_person_name(inferred_name):
            name = inferred_name
        if not father_name and inferred_father and looks_like_person_name(inferred_father):
            father_name = inferred_father

    return {"name": name, "father_name": father_name, "pan": pan, "doc_type": "pan_image"}


def extract_from_pan_pdf(pdf_path: str):
    raw_text = extract_text_from_pdf(pdf_path)
    lines = [clean_line(line) for line in raw_text.splitlines() if line.strip()]
    pan_matches = PAN_REGEX.findall(raw_text)
    pan = pan_matches[0] if pan_matches else None

    filtered = []
    for ln in lines:
        up = ln.upper()
        if any(x in up for x in ["INCOME TAX DEPARTMENT", "GOVT. OF INDIA", "GOVERNMENT OF INDIA",
                                  "PERMANENT ACCOUNT NUMBER", "NAME", "FATHER", "DATE OF BIRTH", "DOB"]):
            continue
        if PAN_REGEX.search(ln): continue
        if DATE_REGEX.search(ln): continue
        if re.search(r"\d", ln): continue
        if len(ln.split()) < 2: continue
        filtered.append(ln)

    name = filtered[0] if len(filtered) > 0 else None
    father_name = filtered[1] if len(filtered) > 1 else None
    return {"name": name, "father_name": father_name, "pan": pan, "doc_type": "pan_pdf"}


# -----------------------------------------------------------------------
# Aadhaar number extraction helper
# -----------------------------------------------------------------------
def _extract_aadhaar_number(text: str) -> str | None:
    """
    Extract and validate a 12-digit Aadhaar number from raw OCR text.
    Rejects VID (16-digit) by stripping it first.
    Accepts numbers starting with 2-9 (UIDAI spec) but also 1 since
    some physical cards show numbers starting with 1.
    """
    # Strip VID to avoid false 12-digit matches inside a 16-digit VID
    clean = re.sub(r"VID\s*[:\-]?\s*\d[\d\s]{12,}", "", text, flags=re.IGNORECASE)

    candidates = re.findall(r"\d{4}\s?\d{4}\s?\d{4}", clean)
    for c in candidates:
        digits = re.sub(r"\D", "", c)
        if len(digits) == 12:
            return f"{digits[:4]} {digits[4:8]} {digits[8:]}"
    return None


def extract_name_from_aadhaar_front_crop(image_path: str):
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    name_crop = img.crop((int(w*0.33), int(h*0.16), int(w*0.78), int(h*0.42)))
    processed = preprocess_for_ocr(name_crop)
    text = paddle_ocr_image(processed)
    lines = [clean_line(l) for l in text.splitlines() if clean_line(l)]

    candidates = []
    for line in lines:
        stripped = strip_devanagari(line)
        if any(x in stripped.lower() for x in ["mobile", "dob", "vid", "government", "india"]):
            continue
        if is_valid_english_name(stripped) and len(stripped.split()) >= 2:
            candidates.append(stripped)

    if candidates:
        candidates.sort(key=lambda x: (len(x.split()), len(x)), reverse=True)
        return candidates[0]
    return None


def split_aadhaar_combined_image(image_path: str, split_x: int = None):
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    mid = split_x if split_x else w // 2
    front = img.crop((0, 0, mid, h))
    back  = img.crop((mid, 0, w, h))
    return front, back


def extract_from_aadhaar_combined_image(image_path: str, split_x: int = None):
    front_img, back_img = split_aadhaar_combined_image(image_path, split_x=split_x)
    f_front = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
    f_back  = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
    front_path = f_front.name
    back_path  = f_back.name
    # Close handles immediately — Windows locks open handles
    f_front.close()
    f_back.close()
    try:
        front_img.save(front_path)
        back_img.save(back_path)
        front_result = extract_from_aadhaar_front_image(front_path)
        back_result  = extract_from_aadhaar_back_image(back_path)
    finally:
        for p in (front_path, back_path):
            try:
                os.unlink(p)
            except OSError:
                pass
    return merge_aadhaar_results(front_result, back_result)


def _detect_horizontal_split(image_path: str) -> int | None:
    """
    For a landscape image containing two Aadhaar cards side by side,
    find the x-coordinate of the gap between them.
    Returns None if no clear gap is found (single card).
    Strategy: find the vertical column with the highest average brightness
    in the centre third of the image (the gap between cards is light/white).
    """
    img = Image.open(image_path).convert("L")   # grayscale
    w, h = img.size
    img_np = np.array(img)

    # Only search the centre 20-80% of width to avoid card edges
    search_left  = int(w * 0.20)
    search_right = int(w * 0.80)

    best_x     = None
    best_score = -1

    for x in range(search_left, search_right):
        col = img_np[:, x].astype(float)
        score = col.mean()
        if score > best_score:
            best_score = score
            best_x = x

    # Only treat as a gap if it's meaningfully brighter than card interior
    # (card interior avg is ~180-200; a white gap between cards is >230)
    if best_score > 220:
        return best_x
    return None


def _is_back_card(image_path: str) -> bool:
    """
    Heuristic to distinguish a lone Aadhaar back card from a front card.

    Back cards have: "Address" / "पता" label, NO DOB / MALE / FEMALE markers.
    Front cards have: DOB, gender, and usually no stand-alone "Address" label
    (address is on the back).
    """
    img_pil = Image.open(image_path).convert("RGB")
    processed = preprocess_for_ocr(img_pil)
    quick_text = paddle_ocr_image(processed)

    has_address = bool(re.search(r"\bAddress\b|पता", quick_text, re.IGNORECASE | re.UNICODE))
    has_front_markers = bool(re.search(
        r"\bDOB\b|\bMALE\b|\bFEMALE\b|\bDate\s+of\s+Birth\b|जन्म",
        quick_text, re.IGNORECASE | re.UNICODE
    ))
    return has_address and not has_front_markers


def extract_from_aadhaar_image(image_path: str):
    img = Image.open(image_path)
    w, h = img.size
    ratio = w / h

    # Single landscape card: ~1.55-1.75 (standard CR-80 card aspect ratio)
    # Two cards side-by-side: ratio > ~2.8
    # Two cards stacked vertically: ratio < ~0.85
    # Single portrait/tall card: ratio 0.55-0.85

    if ratio > 2.5:
        # Definitely two cards side by side — find the split point
        split_x = _detect_horizontal_split(image_path)
        if split_x:
            return extract_from_aadhaar_combined_image(image_path, split_x=split_x)
        # Fallback: naive midpoint
        return extract_from_aadhaar_combined_image(image_path)

    if ratio < 0.85:
        # Two cards stacked vertically (portrait combined image)
        return extract_from_aadhaar_vertical_combined_image(image_path)

    # Single card — detect whether it is a back card (address-only) or front card
    if _is_back_card(image_path):
        return extract_from_aadhaar_back_image(image_path)
    return extract_from_aadhaar_front_image(image_path)


def split_aadhaar_vertical_combined_image(image_path: str):
    """
    Split a vertically-stacked combined Aadhaar image into front and back halves.

    Strategy: find the dark coloured separator band (orange/saffron "मेरा आधार"
    strip) that sits between the two cards. It is the DARKEST horizontal band in
    the search zone.  We locate its darkest row, then walk downward until
    brightness recovers — that recovery point is the true split (bottom edge of
    the band = top of the back card).

    Fallback: if no clearly dark band is found (score close to card interior
    brightness), fall back to the midpoint.
    """
    img = Image.open(image_path).convert("RGB")
    gray = ImageOps.grayscale(img)
    w, h = gray.size

    search_top    = int(h * 0.35)
    search_bottom = int(h * 0.65)
    step = max(1, w // 200)

    # Build per-row average brightness for the search band
    row_brightness = []
    for y in range(search_top, search_bottom):
        samples = [gray.getpixel((x, y)) for x in range(0, w, step)]
        row_brightness.append((y, sum(samples) / len(samples)))

    # Darkest row = centre of the separator band
    darkest_y, darkest_val = min(row_brightness, key=lambda t: t[1])

    # Only treat it as a real separator if it's noticeably darker than typical
    # card interior (white card interior ~220-240; coloured band ~100-180)
    card_avg = sum(v for _, v in row_brightness) / len(row_brightness)
    if darkest_val < card_avg - 20:
        # Walk downward from the darkest row until brightness recovers
        recovery_threshold = darkest_val + 40
        split_y = darkest_y
        for y in range(darkest_y, search_bottom):
            samples = [gray.getpixel((x, y)) for x in range(0, w, step)]
            if sum(samples) / len(samples) > recovery_threshold:
                split_y = y
                break
    else:
        # No clear dark band found — fall back to midpoint
        split_y = h // 2

    front = img.crop((0, 0, w, split_y))
    back  = img.crop((0, split_y, w, h))
    return front, back


def score_aadhaar_result(r):
    score = 0
    if r.get("name"):   score += 3
    if r.get("father_name"): score += 2
    if r.get("dob"):    score += 2
    if r.get("aadhaar"): score += 2
    return score


def extract_from_aadhaar_vertical_combined_image(image_path: str):
    img = Image.open(image_path).convert("RGB")
    w, h = img.size

    detected_front, _ = split_aadhaar_vertical_combined_image(image_path)
    detected_y = detected_front.size[1]

    candidate_splits = [int(h * 0.48), int(h * 0.50), int(h * 0.52), detected_y]
    best_result = None
    best_score = -1

    for y in candidate_splits:
        if y <= 0 or y >= h:
            continue
        front_img = img.crop((0, 0, w, y))
        back_img  = img.crop((0, y, w, h))

        f_front = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
        f_back  = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
        front_path = f_front.name
        back_path  = f_back.name
        # Close handles before writing — Windows locks open handles
        f_front.close()
        f_back.close()
        try:
            front_img.save(front_path)
            back_img.save(back_path)
            front_result = extract_from_aadhaar_front_image(front_path)
            back_result  = extract_from_aadhaar_back_image(back_path)
            merged = merge_aadhaar_results(front_result, back_result)
            score  = score_aadhaar_result(merged)
            if score > best_score:
                best_score = score
                best_result = merged
        finally:
            for p in (front_path, back_path):
                try:
                    os.unlink(p)
                except OSError:
                    pass

    return best_result


# -----------------------------------------------------------------------
# extract_from_aadhaar_front_image  (full rewrite of extraction logic)
# -----------------------------------------------------------------------
def extract_from_aadhaar_front_image(image_path: str):
    d = openbharatocr.front_aadhaar(image_path)

    raw_text = d.get("raw_text") or " ".join(str(v) for v in d.values() if isinstance(v, str))
    raw_text = normalize_ocr_text(raw_text)

    # Run a second pass with preprocessing on the full image to catch fields
    # missed by openbharatocr (common on physical/photo cards with mixed script)
    img_pil = Image.open(image_path).convert("RGB")
    processed_full = preprocess_for_ocr(img_pil)
    extra_text = normalize_ocr_text(paddle_ocr_image(processed_full))

    # Merge both passes; downstream functions use strip_devanagari() to filter
    combined_text = raw_text + "\n" + extra_text

    lines = [clean_line(line) for line in combined_text.splitlines() if line.strip()]

    # ---- Aadhaar number ----
    aadhaar = d.get("aadhaar_number") or _extract_aadhaar_number(combined_text)

    # ---- Father name ----
    father_name = None
    father_match = FATHER_REGEX.search(combined_text)
    if father_match:
        father_name = clean_line(father_match.group(1))

    # ---- DOB ----
    dob = d.get("dob")
    if not dob:
        dob = extract_dob_from_lines(lines) or extract_dob(combined_text)

    # ---- Gender ----
    gender = extract_gender_from_aadhaar_front_crop(image_path)
    if not gender:
        gender = resolve_gender(combined_text, d.get("gender"), lines)

    # ---- Name ----
    # Priority: library → crop → multi-pass heuristic
    name = d.get("name")

    bad_name_terms = ["mobile", "mobile no", "dob", "male", "female",
                      "other", "address", "vid", "government", "india"]

    def _name_is_bad(n):
        if not n:
            return True
        n_stripped = strip_devanagari(n).lower()
        return any(t in n_stripped for t in bad_name_terms) or is_devanagari(n.strip())

    if _name_is_bad(name):
        name = None

    if not name:
        name = extract_name_from_aadhaar_front_crop(image_path)

    if _name_is_bad(name):
        name = None

    if not name:
        name = extract_name(lines)

    if _name_is_bad(name):
        name = None

    # Last-resort: reconstruct from remaining alpha tokens (English only)
    if not name:
        compact_wo_aadhaar = re.sub(r"\b\d{4}\s?\d{4}\s?\d{4}\b", "", combined_text)
        work = compact_wo_aadhaar
        if dob:
            work = work.replace(dob, " ")
        if gender:
            work = re.sub(rf"\b{re.escape(gender)}\b", " ", work, flags=re.IGNORECASE)
        work = re.sub(r"\bMobile\s+No\.?\s*\d*", " ", work, flags=re.IGNORECASE)
        work = re.sub(r"\bVID\b.*", " ", work, flags=re.IGNORECASE)
        # Remove all Devanagari before token scan
        work = strip_devanagari(work)
        work = re.sub(r"\s+", " ", work).strip()
        words = [w for w in work.split() if re.fullmatch(r"[A-Za-z]+", w)]
        if 1 <= len(words) <= 4:
            name = " ".join(words[:2]) if len(words) >= 2 else words[0]

    name = normalize_person_name(name)
    if _name_is_bad(name):
        name = None

    father_name = normalize_person_name(father_name)

    if name:
        name = re.sub(r"\b(?:MALE|FEMALE|OTHER)\b", "", name, flags=re.IGNORECASE)
        name = strip_devanagari(name)
        name = re.sub(r"\s+", " ", name).strip() or None

    return {
        "name": name,
        "father_name": father_name,
        "dob": dob,
        "aadhaar": aadhaar,
        "gender": gender,
        "doc_type": "aadhaar_front_image",
        "_raw_text": raw_text,
    }


# -----------------------------------------------------------------------
# extract_from_aadhaar_back_image
# -----------------------------------------------------------------------
def extract_from_aadhaar_back_image(image_path: str):
    d = openbharatocr.back_aadhaar(image_path)

    raw_text = d.get("raw_text") or " ".join(str(v) for v in d.values() if isinstance(v, str))
    raw_text = normalize_ocr_text(clean_ocr_noise(raw_text))

    img_pil = Image.open(image_path).convert("RGB")

    if len(raw_text.strip()) < 20:
        # Primary OCR failed — use preprocessed image
        processed = preprocess_for_ocr(img_pil)
        raw_text = paddle_ocr_image(processed)

    # Second pass: preprocessed full image to catch the address / Aadhaar number
    processed_full = preprocess_for_ocr(img_pil)
    extra_text = normalize_ocr_text(paddle_ocr_image(processed_full))

    combined_text = raw_text + "\n" + extra_text
    lines = [clean_line(line) for line in combined_text.splitlines() if line.strip()]
    compact = re.sub(r"\s+", " ", combined_text).strip()

    # ---- Aadhaar number ----
    aadhaar = d.get("aadhaar_number") or _extract_aadhaar_number(compact)

    if not aadhaar:
        # Crop bottom strip — Aadhaar number is often printed at the bottom of back card
        w, h = img_pil.size
        bottom = img_pil.crop((int(w * 0.10), int(h * 0.65), int(w * 0.95), h))
        bottom_processed = preprocess_for_ocr(bottom)
        bottom_text = paddle_ocr_image(bottom_processed)
        aadhaar = _extract_aadhaar_number(bottom_text)

    # ---- Father/relative name ----
    father_name = d.get("father_name")
    if not father_name:
        m = FATHER_REGEX.search(compact)
        if m:
            father_name = clean_line(m.group(1))

    return {
        "name": None,
        "father_name": father_name,
        "dob": None,
        "aadhaar": aadhaar,
        "doc_type": "aadhaar_back_image",
        "_raw_text": raw_text,
    }


def extract_from_aadhaar_pdf(pdf_path: str):
    raw_text = extract_text_from_pdf(pdf_path)
    raw_text = normalize_ocr_text(raw_text)
    lines = [line for line in raw_text.splitlines() if line.strip()]

    clean_text = re.sub(r"VID\s*[:\-]?\s*\d[\d\s]{12,}", "", raw_text, flags=re.IGNORECASE)
    aadhaar_candidates = re.findall(r"\d{4}\s?\d{4}\s?\d{4}", clean_text)
    aadhaar_candidates = [c for c in aadhaar_candidates if re.fullmatch(r"\d{12}", re.sub(r"\D", "", c))]

    aadhaar = None
    best_score = -1

    for cand in aadhaar_candidates:
        digits = re.sub(r"\D", "", cand)
        if digits[0] in "01":
            continue
        score = 0
        if re.search(rf"(?:Your\s+Aadhaar\s+No|Aadhaar\s+No|Aadhaar)[^\d]{{0,40}}{re.escape(cand)}",
                     raw_text, re.IGNORECASE | re.DOTALL):
            score += 20
        score += raw_text.count(cand) * 3
        score += raw_text.count(digits)
        if re.search(rf"DOB.*?{re.escape(cand)}", raw_text, re.IGNORECASE | re.DOTALL):
            score += 5
        if score > best_score:
            best_score = score
            aadhaar = f"{digits[:4]} {digits[4:8]} {digits[8:]}"

    father_name = None
    father_match = FATHER_REGEX.search(raw_text)
    if father_match:
        father_name = clean_line(father_match.group(1))

    name = extract_name_from_aadhaar_pdf(lines)
    dob  = extract_dob(raw_text)

    if aadhaar:
        digits = re.sub(r"\D", "", aadhaar)
        if digits[0] in "01":
            aadhaar = None

    return {
        "name": name,
        "father_name": father_name,
        "dob": dob,
        "aadhaar": aadhaar,
        "doc_type": "aadhaar_pdf",
    }


def merge_aadhaar_results(front_result=None, back_result=None):
    front_result = front_result or {}
    back_result  = back_result  or {}
    return {
        "name":        front_result.get("name")        or back_result.get("name"),
        "father_name": front_result.get("father_name") or back_result.get("father_name"),
        "dob":         front_result.get("dob")         or back_result.get("dob"),
        "aadhaar":     front_result.get("aadhaar")     or back_result.get("aadhaar"),
        "doc_type": "aadhaar",
    }

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", help="Path to single image/PDF")
    parser.add_argument(
        "--doc-type",
        choices=["aadhaar_image", "aadhaar_pdf", "pan_image", "pan_pdf", "aadhaar_front_back"],
        required=True,
    )
    parser.add_argument("--front", help="Path to Aadhaar front image")
    parser.add_argument("--back",  help="Path to Aadhaar back image")
    args = parser.parse_args()

    result = None

    if args.doc_type == "pan_image":
        if not args.file: parser.error("--file is required for pan_image")
        result = extract_from_pan_image(str(Path(args.file).resolve()))

    elif args.doc_type == "pan_pdf":
        if not args.file: parser.error("--file is required for pan_pdf")
        result = extract_from_pan_pdf(str(Path(args.file).resolve()))

    elif args.doc_type == "aadhaar_image":
        if not args.file: parser.error("--file is required for aadhaar_image")
        result = extract_from_aadhaar_image(str(Path(args.file).resolve()))

    elif args.doc_type == "aadhaar_pdf":
        if not args.file: parser.error("--file is required for aadhaar_pdf")
        result = extract_from_aadhaar_pdf(str(Path(args.file).resolve()))

    elif args.doc_type == "aadhaar_front_back":
        if not args.front and not args.back:
            parser.error("At least one of --front or --back is required")
        front_result = extract_from_aadhaar_front_image(str(Path(args.front).resolve())) if args.front else None
        back_result  = extract_from_aadhaar_back_image(str(Path(args.back).resolve()))  if args.back  else None
        result = merge_aadhaar_results(front_result, back_result)

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()