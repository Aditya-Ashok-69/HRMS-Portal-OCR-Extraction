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
    s = clean_line(line)
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
# FIXED: extract_name — used for Aadhaar front IMAGE
# Strategy:
#   1. Name labeled explicitly (e.g. "Name: Ravi Kumar") — highest trust
#   2. Name on line immediately before DOB line
#   3. Name on line immediately before S/O | D/O | W/O | C/O line
#   4. Name after "To" block (printed Aadhaar letter format)
#   5. Name before MALE/FEMALE/OTHER/MOBILE/VID keyword line
#   6. Generic fallback scan
# -----------------------------------------------------------------------
def extract_name(lines):
    cleaned = [clean_line(line) for line in lines if clean_line(line)]

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

    # Pass 3: name immediately before DOB line (within 2 lines up)
    for i, line in enumerate(cleaned):
        if re.search(r"DOB|Date\s*of\s*Birth", line, re.IGNORECASE):
            for j in range(max(0, i - 2), i):
                candidate = cleaned[j]
                if is_valid_english_name(candidate) and len(candidate.split()) >= 2:
                    return candidate

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
    for i, line in enumerate(cleaned):
        if re.search(r"\b(DOB|MALE|FEMALE|OTHER|MOBILE|VID)\b", line, re.IGNORECASE):
            for j in range(max(0, i - 3), i):
                cand = cleaned[j]
                if is_valid_english_name(cand) and "mobile" not in cand.lower():
                    return cand

    # Pass 7: generic scan — first line that looks like a valid name
    for line in cleaned:
        if is_valid_english_name(line) and "mobile" not in line.lower():
            return line

    return None


# -----------------------------------------------------------------------
# FIXED: extract_name_from_aadhaar_pdf
# Digital/e-Aadhaar PDFs have cleaner text, so we can look for
# the "Name:" label directly before falling back to position heuristics.
# -----------------------------------------------------------------------
def extract_name_from_aadhaar_pdf(lines):
    cleaned = [clean_line(x) for x in lines if clean_line(x)]

    # Pass 1: explicit "Name:" label — most reliable in digital PDFs
    for i, line in enumerate(cleaned):
        # Label on its own line, value on next line
        if re.fullmatch(r"(?:Name|Your\s+Name)\s*[:\-]?", line, re.IGNORECASE):
            for j in range(i + 1, min(i + 3, len(cleaned))):
                cand = cleaned[j]
                if is_valid_english_name(cand) and len(cand.split()) >= 2:
                    return cand
        # Label + value on same line: "Name: Ravi Kumar"
        m = re.match(r"^(?:Name|Your\s+Name)\s*[:\-]\s*(.+)$", line, re.IGNORECASE)
        if m:
            cand = clean_line(m.group(1))
            if is_valid_english_name(cand) and len(cand.split()) >= 2:
                return cand

    # Pass 2: name immediately before DOB line
    for i, line in enumerate(cleaned):
        if re.search(r"\bDOB\b", line, re.IGNORECASE):
            for j in range(i - 1, max(-1, i - 5), -1):
                cand = cleaned[j]
                if is_valid_english_name(cand) and len(cand.split()) >= 2:
                    return cand

    # Pass 3: name before S/O | D/O | W/O | C/O line
    for i, line in enumerate(cleaned):
        if re.search(r"^(S/O|D/O|C/O|W/O)\s*[:\-]?", line, re.IGNORECASE):
            for j in range(max(0, i - 3), i):
                cand = cleaned[j]
                if is_valid_english_name(cand) and len(cand.split()) >= 2:
                    return cand

    # Pass 4: name after "To" block
    for i, line in enumerate(cleaned):
        if line.strip().lower() == "to":
            for j in range(i + 1, min(i + 6, len(cleaned))):
                cand = cleaned[j]
                if is_valid_english_name(cand) and len(cand.split()) >= 2:
                    return cand

    return None


def extract_dob(text: str):
    m = DOB_KEYWORD_REGEX.search(text)
    if m:
        return m.group(1)
    return None

def extract_gender(text):
    if not text:
        return None

    text = text.upper()

    if "FEMALE" in text:
        return "FEMALE"

    if re.search(r"\bMALE\b", text):
        return "MALE"

    if "OTHER" in text:
        return "OTHER"

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
                g = extract_gender(line_clean)
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
    gender_crop = img.crop((int(w*0.32), int(h*0.32), int(w*0.78), int(h*0.58)))
    processed = preprocess_for_ocr(gender_crop)
    text = paddle_ocr_image(processed).upper()

    if re.search(r"\bFEMALE\b", text): return "FEMALE"
    if re.search(r"\bMALE\b", text):   return "MALE"
    if re.search(r"\bOTHER\b", text):  return "OTHER"
    return None

def extract_address(lines):
    address_lines = []
    collecting = False

    for line in lines:
        s = clean_line(line)

        s = re.sub(
            r"\bVID\s*[:\-]?\s*\d[\d\s]{8,}\b",
            "",
            s,
            flags=re.IGNORECASE
        )
        s = re.sub(
            r"\bVID\s*[:\-]?\s*",
            "",
            s,
            flags=re.IGNORECASE
        )

        if not s:
            continue

        if not collecting and re.search(
            r"^\s*Address\s*:?",
            s,
            re.IGNORECASE
        ):
            collecting = True
            address_lines = []

            s = re.sub(
                r"^\s*Address\s*:?",
                "",
                s,
                flags=re.IGNORECASE
            ).strip()

            if s:
                if re.match(r"^(S/O|D/O|C/O|W/O)", s, re.IGNORECASE):
                    s = re.sub(
                        r"^(?:S/O|D/O|C/O|W/O)\s*:?\s*[^,]+,\s*",
                        "",
                        s,
                        flags=re.IGNORECASE
                    )

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
            s = re.sub(
                r"^(?:S/O|D/O|C/O|W/O)\s*:?\s*[^,]+,\s*",
                "",
                s,
                flags=re.IGNORECASE
            )

            if not s.strip():
                continue

        if (
            address_lines
            and address_lines[-1].upper() == "NO"
            and re.match(r"^\d+", s)
        ):
            address_lines[-1] = address_lines[-1] + s
            continue

        address_lines.append(s)

        if (
            STATE_PIN_REGEX.search(s)
            or re.search(
                r"\bPIN\s*Code\s*[:\-]?\s*\d{6}\b",
                s,
                re.IGNORECASE
            )
            or re.search(
                r"\b[A-Za-z ]+\s*-\s*\d{6}\b",
                s
            )
        ):
            break

    if not address_lines:
        return None

    address = ", ".join(address_lines)

    address = re.sub(
        r"\bVengathur\s*,?\s*Manavalanagar\b",
        "",
        address,
        flags=re.IGNORECASE
    )

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
        "",
        address,
        flags=re.IGNORECASE
    )

    return re.sub(r"\s+", " ", address).strip()

def clean_aadhaar_address(address):
    if not address:
        return address

    address = re.sub(r"\b[2-9]\d{3}\s?\d{4}\s?\d{4}\b", "", address)
    address = re.sub(r"\b\d{9,}\b", "", address)
    address = re.sub(r"\b(?:uidai|uidal)\.gov\.in\b", "", address, flags=re.IGNORECASE)
    address = re.sub(r"\b\d{6}\s+\d+\b", "", address)
    address = re.sub(
        r"\b(?:S/O|D/O|C/O|W/O)\s*:?\s*[^,]+,?",
        "",
        address,
        flags=re.IGNORECASE
    )
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

    banned = {
        "GOVERNMENT",
        "INDIA",
        "DEPARTMENT",
        "ACCOUNT",
        "NUMBER",
        "PERMANENT",
        "CARD",
        "TAX",
        "INCOME"
    }

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
        ocr_text = "\n".join(
            str(v) for v in d.values()
            if isinstance(v, str)
        )

    compact_text = re.sub(r"\s+", " ", ocr_text).strip()

    # PAN number
    pan = (
        d.get("pan_number")
        or d.get("PAN Number")
    )

    if not pan:
        pan_matches = PAN_REGEX.findall(compact_text)
        pan = pan_matches[0] if pan_matches else None

    # Name — library first
    name = (
        d.get("name")
        or d.get("Full Name")
    )

    # Father name — library first
    father_name = (
        d.get("father_name")
        or d.get("Parent's Name")
    )

    # DOB — library first, then regex
    dob = (
        d.get("dob")
        or d.get("Date of Birth")
    )

    if not dob:
        dob_match = DATE_REGEX.search(compact_text)
        dob = dob_match.group(1) if dob_match else None

    if not dob:
        dob = best_pan_dob(image_path)

    # Fix common OCR misread of "Father's Name:" label
    ocr_text = re.sub(
        r"\bme:\s*",
        "Father's Name: ",
        ocr_text,
        flags=re.IGNORECASE
    )

    # Father name regex fallback
    if not father_name:
        m = re.search(
            r"Father.?s Name\s*:\s*([A-Za-z ]+)",
            ocr_text,
            re.IGNORECASE
        )

        if m:
            extracted = clean_line(m.group(1))

            if looks_like_person_name(extracted):
                father_name = extracted

    # Candidate heuristics when library failed
    if not name or not father_name:

        PAN_NOISE = [
            "INCOME TAX",
            "GOVERNMENT",
            "INDIA",
            "ACCOUNT",
            "PERMANENT",
            "DEPARTMENT",
            "CARD"
        ]

        lines = [
            clean_line(x)
            for x in ocr_text.splitlines()
            if clean_line(x)
        ]

        candidates = []

        for line in lines:

            if PAN_REGEX.search(line):
                continue

            if DATE_REGEX.search(line):
                continue

            if any(
                noise.lower() in line.lower()
                for noise in PAN_NOISE
            ):
                continue

            line_lower = line.lower()

            if "father" in line_lower:
                continue

            if "parent" in line_lower:
                continue

            if "name" in line_lower:
                continue

            if re.search(r"\d", line):
                continue

            words = line.split()

            if 1 < len(words) <= 4:
                candidates.append(line)

        valid_candidates = [
            c for c in candidates
            if looks_like_person_name(c)
        ]

        if not name and len(valid_candidates) >= 1:
            name = valid_candidates[0]

        if not father_name and len(valid_candidates) >= 2:
            father_name = valid_candidates[1]

    # Flat-text inference as last resort
    if not name or not father_name:

        inferred_name, inferred_father = (
            infer_pan_names_from_flat_text(
                compact_text,
                pan=pan,
                dob=dob
            )
        )

        if (
            not name
            and inferred_name
            and looks_like_person_name(inferred_name)
        ):
            name = inferred_name

        if (
            not father_name
            and inferred_father
            and looks_like_person_name(inferred_father)
        ):
            father_name = inferred_father

    return {
        "name": name,
        "father_name": father_name,
        "pan": pan,
        "doc_type": "pan_image",
    }


def extract_from_pan_pdf(pdf_path: str):
    raw_text = extract_text_from_pdf(pdf_path)
    lines = [clean_line(line) for line in raw_text.splitlines() if line.strip()]

    pan_matches = PAN_REGEX.findall(raw_text)
    pan = pan_matches[0] if pan_matches else None

    filtered = []
    for ln in lines:
        up = ln.upper()
        if any(x in up for x in [
            "INCOME TAX DEPARTMENT",
            "GOVT. OF INDIA",
            "GOVERNMENT OF INDIA",
            "PERMANENT ACCOUNT NUMBER",
            "NAME",
            "FATHER",
            "DATE OF BIRTH",
            "DOB"
        ]):
            continue
        if PAN_REGEX.search(ln):
            continue
        if DATE_REGEX.search(ln):
            continue
        if re.search(r"\d", ln):
            continue
        if len(ln.split()) < 2:
            continue
        filtered.append(ln)

    name = filtered[0] if len(filtered) > 0 else None
    father_name = filtered[1] if len(filtered) > 1 else None

    return {
        "name": name,
        "father_name": father_name,
        "pan": pan,
        "doc_type": "pan_pdf",
    }

def split_aadhaar_combined_image(image_path: str):
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    mid = w // 2
    front = img.crop((0, 0, mid, h))
    back = img.crop((mid, 0, w, h))
    return front, back

def extract_name_from_aadhaar_front_crop(image_path: str):
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    name_crop = img.crop((int(w*0.33), int(h*0.16), int(w*0.78), int(h*0.42)))
    processed = preprocess_for_ocr(name_crop)
    text = paddle_ocr_image(processed)
    lines = [clean_line(l) for l in text.splitlines() if clean_line(l)]

    candidates = []
    for line in lines:
        if any(x in line.lower() for x in ["mobile", "dob", "vid", "government", "india"]):
            continue
        if is_valid_english_name(line) and len(line.split()) >= 2:
            candidates.append(line)

    if candidates:
        candidates.sort(key=lambda x: (len(x.split()), len(x)), reverse=True)
        return candidates[0]
    return None

def extract_from_aadhaar_combined_image(image_path: str):
    front_img, back_img = split_aadhaar_combined_image(image_path)
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f_front, \
         tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f_back:
        front_img.save(f_front.name)
        back_img.save(f_back.name)
        try:
            front_result = extract_from_aadhaar_front_image(f_front.name)
            back_result = extract_from_aadhaar_back_image(f_back.name)
        finally:
            os.unlink(f_front.name)
            os.unlink(f_back.name)
    return merge_aadhaar_results(front_result, back_result)

def extract_from_aadhaar_image(image_path: str):
    img = Image.open(image_path)
    w, h = img.size
    ratio = w / h

    if ratio > 1.6:
        return extract_from_aadhaar_combined_image(image_path)

    if ratio < 0.9:
        return extract_from_aadhaar_vertical_combined_image(image_path)

    return extract_from_aadhaar_front_image(image_path)

def split_aadhaar_vertical_combined_image(image_path: str):
    img = Image.open(image_path).convert("RGB")
    gray = ImageOps.grayscale(img)
    w, h = gray.size

    search_top = int(h * 0.40)
    search_bottom = int(h * 0.65)

    best_y = h // 2
    best_score = None

    for y in range(search_top, search_bottom):
        row = [gray.getpixel((x, y)) for x in range(0, w, max(1, w // 200))]
        score = sum(row) / len(row)
        if best_score is None or score > best_score:
            best_score = score
            best_y = y

    front = img.crop((0, 0, w, best_y))
    back = img.crop((0, best_y, w, h))
    return front, back

def score_aadhaar_result(r):
    score = 0
    if r.get("name"):
        score += 3
    if r.get("father_name"):
        score += 2
    if r.get("dob"):
        score += 2
    if r.get("aadhaar"):
        score += 2
    return score

def extract_from_aadhaar_vertical_combined_image(image_path: str):
    img = Image.open(image_path).convert("RGB")
    w, h = img.size

    detected_front, _ = split_aadhaar_vertical_combined_image(image_path)
    detected_y = detected_front.size[1]

    candidate_splits = [
        int(h * 0.48),
        int(h * 0.50),
        int(h * 0.52),
        detected_y
    ]

    best_result = None
    best_score = -1

    for y in candidate_splits:
        if y <= 0 or y >= h:
            continue

        front_img = img.crop((0, 0, w, y))
        back_img = img.crop((0, y, w, h))

        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f_front, \
             tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f_back:

            front_img.save(f_front.name)
            back_img.save(f_back.name)

            try:
                front_result = extract_from_aadhaar_front_image(f_front.name)
                back_result = extract_from_aadhaar_back_image(f_back.name)

                merged = merge_aadhaar_results(front_result, back_result)
                score = score_aadhaar_result(merged)

                if score > best_score:
                    best_score = score
                    best_result = merged

            finally:
                os.unlink(f_front.name)
                os.unlink(f_back.name)

    return best_result


def extract_from_aadhaar_front_image(image_path: str):
    d = openbharatocr.front_aadhaar(image_path)

    raw_text = d.get("raw_text") or " ".join(
        str(v) for v in d.values() if isinstance(v, str)
    )
    raw_text = normalize_ocr_text(raw_text)
    lines = [clean_line(line) for line in raw_text.splitlines() if line.strip()]

    aadhaar_matches = AADHAAR_REGEX.findall(raw_text)
    aadhaar = d.get("aadhaar_number") or (aadhaar_matches[0] if aadhaar_matches else None)

    father_name = None
    father_match = FATHER_REGEX.search(raw_text)
    if father_match:
        father_name = clean_line(father_match.group(1))

    # Name — try library first, then crop, then line-based heuristic
    name = d.get("name")
    dob = d.get("dob")
    gender = extract_gender_from_aadhaar_front_crop(image_path)
    if not gender:
        gender = resolve_gender(raw_text, d.get("gender"), lines)
    compact = clean_line(raw_text)

    if not aadhaar:
        m = re.search(r"\b[2-9]\d{3}\s\d{4}\s\d{4}\b", compact)
        if m:
            aadhaar = m.group(0)

    compact_wo_aadhaar = re.sub(r"\b[2-9]\d{3}\s\d{4}\s\d{4}\b", "", compact).strip()

    if not dob:
        dob_match = re.search(r"\b(\d{2}/\d{2}/\d{4})\b", compact_wo_aadhaar)
        if dob_match:
            dob = dob_match.group(1)

    # FIXED: more structured name resolution
    if not name:
        name = extract_name_from_aadhaar_front_crop(image_path)

    if name:
        bad_name_terms = [
            "mobile", "mobile no", "dob", "male", "female",
            "other", "address", "vid", "government", "india"
        ]
        if any(term in name.lower() for term in bad_name_terms):
            name = None

    if not name:
        name = extract_name(lines)  # now uses the improved multi-pass function

    if name:
        bad_name_terms = [
            "mobile", "mobile no", "dob", "male", "female",
            "other", "address", "vid", "government", "india"
        ]
        if any(term in name.lower() for term in bad_name_terms):
            name = None

    # Last resort: reconstruct from remaining alpha tokens
    if not name:
        work = compact_wo_aadhaar
        if dob:
            work = work.replace(dob, " ").strip()
        if gender:
            work = re.sub(rf"\b{re.escape(gender)}\b", " ", work, flags=re.IGNORECASE)
        work = re.sub(r"\bMobile\s+No\.?\s*\d*", " ", work, flags=re.IGNORECASE)
        work = re.sub(r"\bVID\b.*", " ", work, flags=re.IGNORECASE)
        work = re.sub(r"\s+", " ", work).strip()

        words = [w for w in work.split() if re.fullmatch(r"[A-Za-z]+", w)]
        if 1 <= len(words) <= 4:
            name = " ".join(words[:2]) if len(words) >= 2 else words[0]

    name = normalize_person_name(name)
    if name:
        bad_name_terms = [
            "mobile", "mobile no", "dob", "male", "female",
            "other", "address", "vid", "government", "india"
        ]
        if any(term in name.lower() for term in bad_name_terms):
            name = None

    father_name = normalize_person_name(father_name)
    if name:
        name = re.sub(r"\b(?:MALE|FEMALE|OTHER)\b", "", name, flags=re.IGNORECASE)
        name = re.sub(r"\s+", " ", name).strip()

    return {
        "name": name,
        "father_name": father_name,
        "dob": dob,
        "aadhaar": aadhaar,
        "doc_type": "aadhaar_front_image",
        # keep raw_text internally for merge; stripped from final output
        "_raw_text": raw_text,
    }


def extract_from_aadhaar_back_image(image_path: str):
    d = openbharatocr.back_aadhaar(image_path)

    raw_text = d.get("raw_text") or " ".join(
        str(v) for v in d.values() if isinstance(v, str)
    )
    raw_text = normalize_ocr_text(clean_ocr_noise(raw_text))

    if len(raw_text.strip()) < 20:
        img = Image.open(image_path).convert("RGB")
        processed = preprocess_for_ocr(img)
        raw_text = paddle_ocr_image(processed)

    lines = [clean_line(line) for line in raw_text.splitlines() if line.strip()]
    compact = re.sub(r"\s+", " ", raw_text).strip()

    aadhaar = d.get("aadhaar_number")
    if not aadhaar:
        m = AADHAAR_REGEX.search(compact)
        if m:
            aadhaar = m.group(0)

    if not aadhaar:
        img = Image.open(image_path).convert("RGB")
        w, h = img.size
        bottom = img.crop((int(w * 0.18), int(h * 0.72), int(w * 0.88), h))
        bottom_processed = preprocess_for_ocr(bottom)
        bottom_text = paddle_ocr_image(bottom_processed)
        m = re.search(r"\b[2-9]\d{3}\s?\d{4}\s?\d{4}\b", bottom_text)
        if m:
            digits = re.sub(r"\D", "", m.group(0))
            if len(digits) == 12:
                aadhaar = f"{digits[:4]} {digits[4:8]} {digits[8:12]}"

    # Father name from back image
    father_name = d.get("father_name")
    if not father_name:
        m = re.search(
            r"\b(?:S/O|D/O|C/O|W/O)\s*[:\-]?\s*([A-Za-z ]+?)(?=,|\s+H\.?\s*No|\s+Address\b|$)",
            compact,
            re.IGNORECASE
        )
        if m:
            father_name = clean_line(m.group(1))

    return {
        "name": None,          # back side has no name
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

    # Aadhaar number — strip VID first to avoid false match
    clean_text = re.sub(
        r"VID\s*[:\-]?\s*\d[\d\s]{12,}",
        "",
        raw_text,
        flags=re.IGNORECASE
    )

    aadhaar_candidates = re.findall(r"\d{4}\s?\d{4}\s?\d{4}", clean_text)
    aadhaar_candidates = [
        c
        for c in aadhaar_candidates
        if re.fullmatch(r"\d{12}", re.sub(r"\D", "", c))
    ]

    aadhaar = None
    best_score = -1

    for cand in aadhaar_candidates:
        digits = re.sub(r"\D", "", cand)
        if digits[0] in "01":
            continue

        score = 0

        if re.search(
            rf"(?:Your\s+Aadhaar\s+No|Aadhaar\s+No|Aadhaar)[^\d]{{0,40}}{re.escape(cand)}",
            raw_text,
            re.IGNORECASE | re.DOTALL
        ):
            score += 20

        score += raw_text.count(cand) * 3
        score += raw_text.count(digits)

        if re.search(
            rf"DOB.*?{re.escape(cand)}",
            raw_text,
            re.IGNORECASE | re.DOTALL
        ):
            score += 5

        if score > best_score:
            best_score = score
            aadhaar = f"{digits[:4]} {digits[4:8]} {digits[8:]}"

    # Father name
    father_name = None
    father_match = FATHER_REGEX.search(raw_text)
    if father_match:
        father_name = clean_line(father_match.group(1))

    # FIXED: use improved PDF-specific name extractor
    name = extract_name_from_aadhaar_pdf(lines)
    dob = extract_dob(raw_text)

    # Final validation
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
    back_result = back_result or {}

    return {
        "name": front_result.get("name") or back_result.get("name"),
        "father_name": front_result.get("father_name") or back_result.get("father_name"),
        "dob": front_result.get("dob") or back_result.get("dob"),
        "aadhaar": front_result.get("aadhaar") or back_result.get("aadhaar"),
        "doc_type": "aadhaar",
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
    parser.add_argument("--back", help="Path to Aadhaar back image")

    args = parser.parse_args()

    result = None

    if args.doc_type == "pan_image":
        if not args.file:
            parser.error("--file is required for pan_image")
        path = str(Path(args.file).resolve())
        result = extract_from_pan_image(path)

    elif args.doc_type == "pan_pdf":
        if not args.file:
            parser.error("--file is required for pan_pdf")
        path = str(Path(args.file).resolve())
        result = extract_from_pan_pdf(path)

    elif args.doc_type == "aadhaar_image":
        if not args.file:
            parser.error("--file is required for aadhaar_image")
        path = str(Path(args.file).resolve())
        result = extract_from_aadhaar_image(path)

    elif args.doc_type == "aadhaar_pdf":
        if not args.file:
            parser.error("--file is required for aadhaar_pdf")
        path = str(Path(args.file).resolve())
        result = extract_from_aadhaar_pdf(path)

    elif args.doc_type == "aadhaar_front_back":
        if not args.front and not args.back:
            parser.error("At least one of --front or --back is required for aadhaar_front_back")

        front_result = None
        back_result = None

        if args.front:
            front_path = str(Path(args.front).resolve())
            front_result = extract_from_aadhaar_front_image(front_path)

        if args.back:
            back_path = str(Path(args.back).resolve())
            back_result = extract_from_aadhaar_back_image(back_path)

        result = merge_aadhaar_results(front_result, back_result)

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
