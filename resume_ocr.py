"""
Resume text extraction + structured JSON parsing.

Supports:
  - Digital PDFs  → pdfplumber (best quality, preserves layout)
  - Scanned PDFs  → RapidOCR via PyMuPDF page rendering
  - Images (JPG/PNG/etc.) → RapidOCR

Output JSON keys:
  name, title, email, phone, linkedin, github, location,
  summary, experience, education, skills,
  certifications, projects, languages,
  hobbies/interests (if present)
"""

import json
import re
import sys
from pathlib import Path

import cv2
import numpy as np
from rapidocr_onnxruntime import RapidOCR

try:
    import pdfplumber
    PDFPLUMBER_OK = True
except ImportError:
    PDFPLUMBER_OK = False

try:
    import fitz
    PYMUPDF_OK = True
except ImportError:
    PYMUPDF_OK = False


# ══════════════════════════════════════════════════════════════════════════════
# Text acquisition
# ══════════════════════════════════════════════════════════════════════════════

def _ocr_array(img: np.ndarray, engine: RapidOCR) -> str:
    result, _ = engine(img)
    if not result:
        return ""
    return "\n".join(item[1] for item in result)


def text_from_image(path: str, engine: RapidOCR) -> str:
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Cannot read image: {path}")
    return _ocr_array(img, engine)


def text_from_pdf(path: str, engine: RapidOCR, ocr_dpi: int = 200) -> str:
    """
    Try pdfplumber first (digital PDFs).
    If the extracted text is too short, fall back to RapidOCR on rendered pages.
    """
    text = ""

    if PDFPLUMBER_OK:
        with pdfplumber.open(path) as pdf:
            pages_text = []
            for page in pdf.pages:
                t = page.extract_text(x_tolerance=2, y_tolerance=2) or ""
                pages_text.append(t)
            text = "\n".join(pages_text).strip()

    # Fallback: scanned PDF → render pages → OCR
    if len(text) < 100:
        if not PYMUPDF_OK:
            raise RuntimeError("PyMuPDF required for scanned PDFs: pip install pymupdf")
        doc = fitz.open(path)
        parts = []
        for page in doc:
            mat = fitz.Matrix(ocr_dpi / 72, ocr_dpi / 72)
            pix = page.get_pixmap(matrix=mat)
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
            if pix.n == 4:
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
            elif pix.n == 3:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            parts.append(_ocr_array(img, engine))
        doc.close()
        text = "\n".join(parts)

    return text


# ══════════════════════════════════════════════════════════════════════════════
# Regex patterns
# ══════════════════════════════════════════════════════════════════════════════

_EMAIL_RE    = re.compile(r'\b[\w.+-]+@[\w-]+\.[a-zA-Z]{2,}\b')
_PHONE_RE    = re.compile(r'(?:\+91[-\s]?|0)?[6-9]\d{9}\b')
_LINKEDIN_RE = re.compile(r'linkedin\.com/in/[\w-]+', re.IGNORECASE)
_GITHUB_RE   = re.compile(r'github\.com/[\w-]+', re.IGNORECASE)
_URL_RE      = re.compile(r'https?://[\w./\-?=%&]+|(?:www|portfolio)\.[\w./\-]+', re.IGNORECASE)

_DATE_RANGE_RE = re.compile(
    r'(?:'
      r'(?:(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|'
      r'Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)'
      r'\.?\s+)?\d{4}'
      r'\s*[-–—to]+\s*'
      r'(?:'
        r'(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|'
        r'Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)'
        r'\.?\s+\d{4}|Present|\d{4}'
      r')'
    r')',
    re.IGNORECASE
)

_GRADE_RE = re.compile(r'(?:CGPA|GPA|Grade)[:\s]+(\d+(?:\.\d+)?)|(\d{2,3})\s*%', re.IGNORECASE)

_SECTION_RE = re.compile(
    r'^\s*(?P<heading>'
    r'SUMMARY|OBJECTIVE|PROFILE|ABOUT(?: ME)?'
    r'|(?:WORK\s+|PROFESSIONAL\s+)?EXPERIENCE|EMPLOYMENT(?: HISTORY)?'
    r'|EDUCATION(?:AL BACKGROUND)?|ACADEMIC(?:S| BACKGROUND)?|QUALIFICATION'
    r'|(?:TECHNICAL\s+|KEY\s+|CORE\s+)?SKILLS?|(?:CORE\s+)?COMPETENC(?:Y|IES)'
    r'|PROJECTS?(?: EXPERIENCE)?'
    r'|CERTIFICATIONS?|ACHIEVEMENTS?|AWARDS?(?: & ACHIEVEMENTS?)?'
    r'|LANGUAGES?'
    r'|HOBBIES?(?: & INTERESTS?)?|INTERESTS?'
    r'|PUBLICATIONS?|VOLUNTEERING?|EXTRA.CURRICULAR'
    r')\s*:?\s*$',
    re.IGNORECASE | re.MULTILINE
)

# canonical key names for section headings
_SECTION_KEYS = {
    'OBJECTIVE': 'summary', 'PROFILE': 'summary', 'ABOUT': 'summary', 'ABOUT ME': 'summary',
    'WORK EXPERIENCE': 'experience', 'PROFESSIONAL EXPERIENCE': 'experience',
    'EMPLOYMENT': 'experience', 'EMPLOYMENT HISTORY': 'experience',
    'EDUCATIONAL BACKGROUND': 'education', 'ACADEMICS': 'education',
    'ACADEMIC BACKGROUND': 'education', 'QUALIFICATION': 'education',
    'TECHNICAL SKILLS': 'skills', 'KEY SKILLS': 'skills', 'CORE SKILLS': 'skills',
    'CORE COMPETENCIES': 'skills', 'COMPETENCY': 'skills', 'COMPETENCIES': 'skills',
    'CERTIFICATION': 'certifications', 'ACHIEVEMENTS': 'certifications',
    'AWARDS': 'certifications', 'AWARDS & ACHIEVEMENTS': 'certifications',
    'PROJECT EXPERIENCE': 'projects',
    'LANGUAGE': 'languages',
    'HOBBIES & INTERESTS': 'hobbies', 'INTERESTS': 'hobbies',
}


# ══════════════════════════════════════════════════════════════════════════════
# Section parsers
# ══════════════════════════════════════════════════════════════════════════════

def _split_sections(text: str) -> dict[str, str]:
    matches = list(_SECTION_RE.finditer(text))
    sections: dict[str, str] = {}
    if matches:
        sections["__header__"] = text[:matches[0].start()].strip()
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        heading = m.group("heading").upper().strip()
        sections[heading] = text[m.end():end].strip()
    return sections


def _parse_header(text: str) -> dict:
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    info: dict = {}

    m = _EMAIL_RE.search(text);    info["email"]    = m.group() if m else None
    m = _PHONE_RE.search(text);    info["phone"]    = m.group() if m else None
    m = _LINKEDIN_RE.search(text); info["linkedin"] = ("https://" + m.group()) if m else None
    m = _GITHUB_RE.search(text);   info["github"]   = ("https://" + m.group()) if m else None

    # name: first line that is purely alphabetic words (≤5), no contact info
    for line in lines:
        if (not _EMAIL_RE.search(line) and not _PHONE_RE.search(line)
                and not _URL_RE.search(line)
                and re.match(r'^[A-Za-z .]+$', line)
                and 1 < len(line.split()) <= 5):
            info["name"] = line
            break

    # title: next clean alphabetic line after name
    name_seen = False
    for line in lines:
        if line == info.get("name"):
            name_seen = True
            continue
        if name_seen and re.match(r'^[A-Za-z .,()/&+-]+$', line) and len(line) > 4:
            info["title"] = line
            break

    # location: "City, State" or "City, State, Country"
    loc = re.search(r'[A-Za-z ]+,\s*[A-Za-z ]+(?:,\s*India)?', text)
    if loc and len(loc.group().strip()) < 60:
        info["location"] = loc.group().strip()

    # drop None values
    return {k: v for k, v in info.items() if v}


def _parse_experience(text: str) -> list[dict]:
    entries = []
    for block in re.split(r'\n{2,}', text.strip()):
        lines = [l.strip() for l in block.splitlines() if l.strip()]
        if not lines:
            continue
        entry: dict = {}
        parts = re.split(r'\s+[-|@]\s+', lines[0], maxsplit=1)
        if len(parts) == 2:
            entry["title"] = parts[0].strip()
            co_parts = parts[1].split(',', 1)
            entry["company"] = co_parts[0].strip()
            if len(co_parts) > 1:
                entry["location"] = co_parts[1].strip()
        else:
            entry["title"] = lines[0]
        dr = _DATE_RANGE_RE.search(block)
        if dr:
            entry["duration"] = dr.group().strip()
        bullets = [l.lstrip('-•*▪► ').strip() for l in lines if l[:1] in '-•*▪►']
        if bullets:
            entry["responsibilities"] = bullets
        entries.append(entry)
    return entries


def _parse_education(text: str) -> list[dict]:
    entries = []
    for block in re.split(r'\n{2,}', text.strip()):
        lines = [l.strip() for l in block.splitlines() if l.strip()]
        if not lines:
            continue
        entry: dict = {}
        parts = re.split(r'\s+[-|@]\s+', lines[0], maxsplit=1)
        if len(parts) == 2:
            entry["degree"] = parts[0].strip()
            entry["institution"] = parts[1].strip()
        else:
            entry["degree"] = lines[0]
        dr = _DATE_RANGE_RE.search(block)
        if dr:
            entry["duration"] = dr.group().strip()
        gm = _GRADE_RE.search(block)
        if gm:
            entry["grade"] = gm.group().strip()
        entries.append(entry)
    return entries


def _parse_skills(text: str) -> list[str]:
    flat = re.sub(r'[|\n•\-*▪►]', ',', text)
    tokens = [t.strip() for t in flat.split(',')]
    return [t for t in tokens if 2 <= len(t) <= 40 and re.search(r'[A-Za-z]', t)]


def _parse_certifications(text: str) -> list[str]:
    lines = [l.strip().lstrip('-•*▪► ') for l in text.splitlines() if l.strip()]
    return [l for l in lines if l]


def _parse_projects(text: str) -> list[dict]:
    entries = []
    for block in re.split(r'\n{2,}', text.strip()):
        lines = [l.strip() for l in block.splitlines() if l.strip()]
        if not lines:
            continue
        entry: dict = {"name": lines[0]}
        desc_lines = [l for l in lines[1:] if not re.match(r'(?:Tech|Stack|Tools?)[:\s]', l, re.I)]
        if desc_lines:
            entry["description"] = " ".join(desc_lines)
        tech_m = re.search(r'(?:Tech(?:nologies)?|Stack|Tools?)[:\s]+(.+)', block, re.IGNORECASE)
        if tech_m:
            entry["technologies"] = [t.strip() for t in re.split(r'[,|]', tech_m.group(1)) if t.strip()]
        entries.append(entry)
    return entries


def _parse_languages(text: str) -> list[dict]:
    entries = []
    for chunk in re.split(r',\s*|\n', text):
        chunk = chunk.strip().lstrip('-•*▪► ')
        if not chunk:
            continue
        m = re.match(r'([A-Za-z]+)\s*\(([^)]+)\)', chunk)
        if m:
            entries.append({"language": m.group(1), "proficiency": m.group(2)})
        elif re.match(r'^[A-Za-z ]{2,30}$', chunk):
            entries.append({"language": chunk})
    return entries


def _parse_plain_list(text: str) -> list[str]:
    return [l.strip().lstrip('-•*▪► ') for l in text.splitlines() if l.strip()]


_SECTION_PARSERS: dict = {
    'SUMMARY':    lambda t: t.strip(),
    'EXPERIENCE': _parse_experience,
    'EDUCATION':  _parse_education,
    'SKILLS':     _parse_skills,
    'CERTIFICATIONS': _parse_certifications,
    'PROJECTS':   _parse_projects,
    'LANGUAGES':  _parse_languages,
    'HOBBIES':    _parse_plain_list,
    'PUBLICATIONS': _parse_plain_list,
    'VOLUNTEERING': _parse_plain_list,
    'EXTRA-CURRICULAR': _parse_plain_list,
}


# ══════════════════════════════════════════════════════════════════════════════
# Master parser
# ══════════════════════════════════════════════════════════════════════════════

def parse_resume(text: str) -> dict:
    """Convert raw resume text into a structured dict."""
    sections = _split_sections(text)
    result: dict = {}

    result.update(_parse_header(sections.pop("__header__", "")))

    for raw_heading, content in sections.items():
        canonical_key = _SECTION_KEYS.get(raw_heading, raw_heading.lower())
        parser_key = _SECTION_KEYS.get(raw_heading, raw_heading)
        parser = _SECTION_PARSERS.get(parser_key) or _SECTION_PARSERS.get(canonical_key.upper())
        if parser:
            result[canonical_key] = parser(content)
        else:
            result[canonical_key] = content.strip()

    return result


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

def process_file(file_path: str) -> dict:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    engine = RapidOCR()
    ext = path.suffix.lower()

    if ext in (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"):
        text = text_from_image(file_path, engine)
    elif ext == ".pdf":
        text = text_from_pdf(file_path, engine)
    else:
        raise ValueError(f"Unsupported format: {ext}")

    result = parse_resume(text)
    result["_source"] = path.name
    return result


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python resume_ocr.py <resume.pdf|resume.jpg>")
        sys.exit(1)
    output = process_file(sys.argv[1])
    print(json.dumps(output, indent=2, ensure_ascii=False))