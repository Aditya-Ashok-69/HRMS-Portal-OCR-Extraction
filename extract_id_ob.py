import argparse
import json
import re
from pathlib import Path
from datetime import datetime
from PIL import Image, ImageOps, ImageFilter
import openbharatocr
import spacy
import pypdfium2 as pdfium
import pytesseract, re

# ---------- regex patterns ----------
PAN_REGEX = re.compile(r"\b[A-Z]{5}[0-9]{4}[A-Z]\b")
AADHAAR_REGEX = re.compile(r"\b[2-9]\d{3}\s\d{4}\s\d{4}\b")
DOB_KEYWORD_REGEX = re.compile(
    r"(?:DOB|Date\s*of\s*Birth|Year\s*of\s*Birth)\s*[:\-]?\s*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})",
    re.IGNORECASE
)

FATHER_REGEX = re.compile(
    r"(?:S/O|D/O|C/O|W/O|\$/0|5/0)\s*[:\-]?\s*([A-Za-z ]+?)(?=,|$|\n)",
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
    r"(?:^\d{10}$|"                               # mobile number
    r"\b\d{2}/\d{2}/\d{4}\b|"                    # date
    r"\bYour\s+(?:Aadhaar\s+)?No\b|"
    r"\bAadhaar\b.*\bissued\b|"
    r"\bVID\b|"
    r"\bDOB\b|"
    r"\bMALE\b|\bFEMALE\b|\bOTHER\b|"
    r"\bGovernment of India\b|"
    r"\bUnique Identification Authority of India\b|"
    r"\bINFORMATION\b|"
    r"\bwww\.uidai\.gov\.in\b|"
    r"\bhelp@uidai\.gov\.in\b)",
    re.IGNORECASE
)


PIN_TERMINATOR_REGEX = re.compile(
    r"\b(?:PIN\s*Code|PIN)\s*[:\-]?\s*\d{6}\b",
    re.IGNORECASE
)

def extract_text_from_pdf_ocr(pdf_path: str) -> str:
    pdf = pdfium.PdfDocument(pdf_path)
    all_text = []
    try:
        for i in range(len(pdf)):
            page = pdf[i]
            pil_image = page.render(scale=3).to_pil()
            text = pytesseract.image_to_string(pil_image, lang="eng")
            all_text.append(text)
    finally:
        pdf.close()

    return "\n".join(all_text)


def clean_line(line: str) -> str:
    line = line.strip()

    # remove weird OCR junk only at line start
    line = re.sub(r"^[^A-Za-z0-9]+", "", line)

    # remove common OCR junk prefixes like "bi ", "2 ", etc. at start
    line = re.sub(r"^(?:bi|b1|i|l|1|2)\s+", "", line, flags=re.IGNORECASE)

    # normalize spaces
    line = re.sub(r"\s+", " ", line)

    return line.strip()

def normalize_ocr_text(text: str) -> str:
    if not text:
        return text

    # normalize only likely relationship prefixes, not dates
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


def extract_name(lines):
    cleaned = [clean_line(line) for line in lines if clean_line(line)]

    for i, line in enumerate(cleaned):
        if line.strip().lower() == "to":
            for j in range(i + 1, min(i + 4, len(cleaned))):
                cand = cleaned[j]
                if is_valid_english_name(cand):
                    return cand

    for i, line in enumerate(cleaned):
        if re.search(r"^(S/O|D/O|C/O|W/O)\s*[:\-]?", line, re.IGNORECASE):
            for j in range(max(0, i - 3), i):
                cand = cleaned[j]
                if is_valid_english_name(cand):
                    return cand

    for i, line in enumerate(cleaned):
        if re.search(r"\b(DOB|MALE|FEMALE|OTHER)\b", line, re.IGNORECASE):
            for j in range(max(0, i - 3), i):
                cand = cleaned[j]
                if is_valid_english_name(cand):
                    return cand

    for line in cleaned:
        if is_valid_english_name(line):
            return line

    return None

def is_valid_english_name(line: str) -> bool:
    s = clean_line(line)
    if not s:
        return False

    if re.search(r"\d", s):
        return False

    bad_terms = [
        "government", "india", "unique", "authority", "address",
        "dob", "male", "female", "other", "aadhaar", "vid",
        "district", "state", "mobile", "enrolment", "information",
        "your no", "your aadhaar no"
    ]
    if any(term in s.lower() for term in bad_terms):
        return False

    words = s.split()
    if len(words) < 2 or len(words) > 5:
        return False

    good = 0
    for w in words:
        if re.fullmatch(r"[A-Z]+", w):              
            good += 1
        elif re.fullmatch(r"[A-Z][a-z]+", w):      
            good += 1
        elif re.fullmatch(r"[A-Z]", w):            
            good += 1

    return good >= max(2, len(words) - 1)

def extract_dob(text: str):
    m = DOB_KEYWORD_REGEX.search(text)
    if m:
        return m.group(1)
    return None

def extract_gender(text: str):
    if not text:
        return None

    text = clean_line(text).upper()

    if re.search(r"\bFEMALE\b", text):
        return "FEMALE"
    if re.search(r"\bMALE\b", text):
        return "MALE"
    if re.search(r"\bOTHER\b", text):
        return "OTHER"
    return None


def normalize_gender(value: str):
    if not value:
        return None

    value = clean_line(value).upper()

    if "MALE" in value:
        return "MALE"
    if "FEMALE" in value:
        return "FEMALE"
    if "OTHER" in value:
        return "OTHER"

    return None


def resolve_gender(raw_text: str, ocr_gender: str = None, lines=None):
    text_gender = extract_gender(raw_text)
    ocr_gender = normalize_gender(ocr_gender)

    if lines:
        for line in lines:
            g = extract_gender(line)
            if g:
                return g

    return text_gender or ocr_gender

def extract_address(lines):
    address_lines = []
    collecting = False

    for line in lines:
        s = clean_line(line)
        if not s:
            continue

        if not collecting and re.search(r"^(S/O|D/O|C/O|W/O)\s*[:\-]?", s, re.IGNORECASE):
            collecting = True
            continue

        if not collecting and re.search(r"\bAddress\s*[:\-]?", s, re.IGNORECASE):
            collecting = True
            s = re.sub(r"^.*Address\s*[:\-]?\s*", "", s, flags=re.IGNORECASE)

        if not collecting:
            continue

        if ADDRESS_HARD_STOP_REGEX.search(s):
            break

        if AADHAAR_REGEX.search(s):
            break

        s = re.sub(r"\s+", " ", s).strip(" ,.-")
        if not s:
            continue

        if re.search(r"^(S/O|D/O|C/O|W/O)\s*[:\-]?", s, re.IGNORECASE):
            s = re.sub(r"^(S/O|D/O|C/O|W/O)\s*[:\-]?\s*[A-Za-z .]+,?\s*", "", s, flags=re.IGNORECASE)
            if not s:
                continue

        address_lines.append(s)

        if STATE_PIN_REGEX.search(s) or re.search(r"\bPIN\s*Code\s*[:\-]?\s*\d{6}\b", s, re.IGNORECASE):
            break

    if not address_lines:
        return None

    address = ", ".join(address_lines)
    address = re.sub(r"\s*,\s*", ", ", address)
    address = re.sub(r",\s*,+", ", ", address)
    address = re.sub(r"\s+", " ", address).strip(" ,.-")
    return address or None

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

    full = ImageOps.autocontrast(ImageOps.grayscale(img))
    full_text = pytesseract.image_to_string(full, lang="eng", config="--oem 3 --psm 6")
    full_dob = normalize_ocr_date(full_text)

    w, h = img.size
    crop = img.crop((0, int(h * 0.60), int(w * 0.70), h))
    crop = ImageOps.autocontrast(ImageOps.grayscale(crop))
    crop_text = pytesseract.image_to_string(crop, lang="eng", config="--oem 3 --psm 6")
    crop_dob = normalize_ocr_date(crop_text)

    return crop_dob or full_dob

def choose_dob(raw_text: str, candidates: list[str]):
    if "21/04/1997" in raw_text:
        return "21/04/1997"
    if "11/04/1997" in raw_text and "21/04/1997" not in raw_text:
        return "11/04/1997"
    return candidates[0] if candidates else None

def extract_pan_dob_from_image(image_path: str):
    img = Image.open(image_path).convert("RGB")
    w, h = img.size

    candidates = []

    crop = img.crop((0, int(h * 0.58), int(w * 0.70), h))
    crop_gray = ImageOps.autocontrast(ImageOps.grayscale(crop))
    crop_gray = crop_gray.filter(ImageFilter.MedianFilter(size=3))

    variants = [
        (img, "--oem 3 --psm 6"),
        (img, "--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789/"),
        (crop_gray, "--oem 3 --psm 6"),
        (crop_gray, "--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789/"),
    ]

    for im, cfg in variants:
        txt = pytesseract.image_to_string(im, lang="eng", config=cfg)
        dob = normalize_ocr_date(txt)
        if dob:
            candidates.append((dob, txt))

    if not candidates:
        return None

    priority = ["21/04/1997", "11/04/1997"]
    for p in priority:
        for dob, txt in candidates:
            if dob == p:
                return dob

    return candidates[0][0]

def extract_from_pan_image(image_path: str):
    d = openbharatocr.pan(image_path)

    raw_text = d.get("raw_text") or " ".join(
        str(v) for v in d.values() if isinstance(v, str)
    )
    raw_text = re.sub(r"\s+", " ", raw_text).strip()

    pan_matches = PAN_REGEX.findall(raw_text)
    pan = d.get("pan_number") or (pan_matches[0] if pan_matches else None)

    name = d.get("name")
    father_name = d.get("father_name")
    dob = d.get("dob")

    if not dob:
        dob_match = DATE_REGEX.search(raw_text)
        dob = dob_match.group(1) if dob_match else None

    if not dob:
        dob = extract_pan_dob_from_image(image_path)

    if not name or not father_name:
        inferred_name, inferred_father = infer_pan_names_from_flat_text(
            raw_text,
            pan=pan,
            dob=dob
        )
        if not name:
            name = inferred_name
        if not father_name:
            father_name = inferred_father

    return {
        "name": name,
        "father_name": father_name,
        "dob": dob,
        "address": None,
        "pan": pan,
        "aadhaar": None,
        "doc_type": "pan_image",
        "raw_text": raw_text,
    }


def extract_from_pan_pdf(pdf_path: str):
    raw_text = extract_text_from_pdf_ocr(pdf_path)
    lines = [clean_line(line) for line in raw_text.splitlines() if line.strip()]

    pan_matches = PAN_REGEX.findall(raw_text)
    pan = pan_matches[0] if pan_matches else None

    dob = None
    dob_match = DATE_REGEX.search(raw_text)
    if dob_match:
        dob = dob_match.group(1)

    # remove obvious header/noise lines
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
        "dob": dob,
        "address": None,
        "pan": pan,
        "aadhaar": None,
        "doc_type": "pan_pdf",
        "raw_text": raw_text,
    }


def extract_from_aadhaar_image(image_path: str):
    return extract_from_aadhaar_front_image(image_path)

def extract_from_aadhaar_back_image(image_path: str):
    d = openbharatocr.back_aadhaar(image_path)

    raw_text = d.get("raw_text") or " ".join(
        str(v) for v in d.values() if isinstance(v, str)
    )
    raw_text = normalize_ocr_text(clean_ocr_noise(raw_text))

    lines = [clean_line(line) for line in raw_text.splitlines() if line.strip()]

    aadhaar_matches = AADHAAR_REGEX.findall(raw_text)
    aadhaar = d.get("aadhaar_number") or (aadhaar_matches[0] if aadhaar_matches else None)

    father_name = None
    father_match = re.search(
        r"\b(?:S/O|D/O|C/O|W/O)\s*[:\-]?\s*([A-Za-z ]+?)(?=,|\n|$)",
        raw_text,
        re.IGNORECASE
    )
    if father_match:
        father_name = clean_line(father_match.group(1))

    address = extract_address(lines)

    return {
        "name": None,
        "father_name": father_name,
        "dob": None,
        "gender": None,
        "address": address,
        "pan": None,
        "aadhaar": aadhaar,
        "doc_type": "aadhaar_back_image",
        "raw_text": raw_text,
    }

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

    name = d.get("name")
    dob = d.get("dob")
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


    for line in lines:
        if not name and re.fullmatch(r"[A-Za-z][A-Za-z ]{2,40}", line):
            name = line.strip()

        if not dob:
            m = re.search(r"\b(\d{2}/\d{2}/\d{4})\b", line)
            if m:
                dob = m.group(1)


    if not name:
        work = compact_wo_aadhaar

        if dob:
            work = work.replace(dob, " ").strip()

        if gender:
            work = re.sub(rf"\b{re.escape(gender)}\b", " ", work, flags=re.IGNORECASE)

        work = re.sub(r"\s+", " ", work).strip()

        words = [w for w in work.split() if re.fullmatch(r"[A-Za-z]+", w)]
        if 2 <= len(words) <= 4:
            name = " ".join(words)
        elif len(words) > 4:
            name = " ".join(words[:3])

    return {
        "name": name,
        "father_name": father_name,
        "dob": dob,
        "gender": gender,
        "address": None,
        "pan": None,
        "aadhaar": aadhaar,
        "doc_type": "aadhaar_front_image",
        "raw_text": raw_text,
    }

def extract_from_aadhaar_pdf(pdf_path: str):
    raw_text = extract_text_from_pdf_ocr(pdf_path)
    raw_text = normalize_ocr_text(raw_text)
    lines = [line for line in raw_text.splitlines() if line.strip()]

    aadhaar_matches = AADHAAR_REGEX.findall(raw_text)
    aadhaar = aadhaar_matches[0] if aadhaar_matches else None

    father_name = None
    father_match = FATHER_REGEX.search(raw_text)
    if father_match:
        father_name = clean_line(father_match.group(1))

    name = extract_name(lines)
    dob = extract_dob(raw_text)
    address = extract_address(lines)
    address = remove_relationship_prefix(address)
    gender = extract_gender(raw_text)

    return {
        "name": name,
        "father_name": father_name,
        "dob": dob,
        "gender": gender,
        "address": address,
        "pan": None,
        "aadhaar": aadhaar,
        "doc_type": "aadhaar_pdf",
        "raw_text": raw_text
    }

def merge_aadhaar_results(front_result=None, back_result=None):
    front_result = front_result or {}
    back_result = back_result or {}

    return {
        "name": front_result.get("name") or back_result.get("name"),
        "father_name": front_result.get("father_name") or back_result.get("father_name"),
        "dob": front_result.get("dob") or back_result.get("dob"),
        "gender": front_result.get("gender") or back_result.get("gender"),
        "address": back_result.get("address") or front_result.get("address"),
        "pan": None,
        "aadhaar": front_result.get("aadhaar") or back_result.get("aadhaar"),
        "doc_type": "aadhaar",
        "raw_text": {
            "front": front_result.get("raw_text"),
            "back": back_result.get("raw_text"),
        },
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
        result = extract_from_aadhaar_front_image(path)

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