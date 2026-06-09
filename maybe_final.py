"""
Document Field Extractor using RapidOCR
========================================
Supports: Aadhaar, PAN, Resume, Payslip
Formats:  PDF, PNG, JPG, JPEG, TIFF, BMP, WEBP, DOCX

Install:
    pip install rapidocr-onnxruntime rapidocr-pdf pdf2image pillow
    pip install python-docx pdfplumber opencv-python-headless numpy

Usage:
    result = extract_fields("aadhaar.pdf")
    result = extract_fields("resume.docx")
    result = extract_fields("payslip.jpg")
"""

import re
import os
import sys
import json
from pathlib import Path

# ── RapidOCR ──────────────────────────────────────────────────────────────────
try:
    from rapidocr_onnxruntime import RapidOCR
except ImportError:
    print("Install: pip install rapidocr-onnxruntime")
    sys.exit(1)

# ── PDF handling ───────────────────────────────────────────────────────────────
try:
    import pdfplumber          # for digital/text PDFs (no OCR needed)
except ImportError:
    pdfplumber = None

try:
    from pdf2image import convert_from_path   # for scanned PDFs
except ImportError:
    convert_from_path = None

# ── DOCX handling ──────────────────────────────────────────────────────────────
try:
    from docx import Document as DocxDocument
except ImportError:
    DocxDocument = None

import numpy as np
from PIL import Image

# ── Initialise RapidOCR once (loads ~50MB ONNX models) ────────────────────────
ocr = RapidOCR()


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 1 — FILE → RAW TEXT
# ══════════════════════════════════════════════════════════════════════════════

def _ocr_image(img) -> str:
    """Run RapidOCR on a PIL Image or numpy array, return joined text."""
    if isinstance(img, Image.Image):
        img = np.array(img.convert("RGB"))
    result, _ = ocr(img)
    if result is None:
        return ""
    # result is list of [bbox, text, score]
    lines = [item[1] for item in result if item[1].strip()]
    return "\n".join(lines)


def _extract_text_from_pdf(filepath: str) -> str:
    """
    Try pdfplumber first (instant, perfect for digital PDFs like e-Aadhaar).
    Fall back to pdf2image + RapidOCR for scanned/image PDFs.
    """
    text = ""

    # --- Digital PDF path ---
    if pdfplumber:
        with pdfplumber.open(filepath) as pdf:
            pages_text = []
            for page in pdf.pages:
                t = page.extract_text()
                if t:
                    pages_text.append(t)
            text = "\n".join(pages_text).strip()

    # If meaningful text found digitally, use it
    if len(text) > 50:
        return text

    # --- Scanned PDF fallback via RapidOCR ---
    if convert_from_path is None:
        raise ImportError("pip install pdf2image  (and poppler-utils on Linux)")

    images = convert_from_path(filepath, dpi=250)
    pages_text = [_ocr_image(img) for img in images]
    return "\n".join(pages_text)


def _extract_text_from_image(filepath: str) -> str:
    img = Image.open(filepath).convert("RGB")
    return _ocr_image(img)


def _extract_text_from_docx(filepath: str) -> str:
    if DocxDocument is None:
        raise ImportError("pip install python-docx")
    doc = DocxDocument(filepath)
    return "\n".join(p.text for p in doc.paragraphs if p.text.strip())


def get_raw_text(filepath: str) -> str:
    """Dispatch to the right extractor based on file extension."""
    ext = Path(filepath).suffix.lower()
    if ext == ".pdf":
        return _extract_text_from_pdf(filepath)
    elif ext in {".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".webp"}:
        return _extract_text_from_image(filepath)
    elif ext == ".docx":
        return _extract_text_from_docx(filepath)
    else:
        raise ValueError(f"Unsupported file type: {ext}")


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 2 — DOCUMENT TYPE DETECTION
# ══════════════════════════════════════════════════════════════════════════════

def detect_doc_type(text: str) -> str:
    """
    Heuristic detection of document type from raw OCR text.
    Returns: 'aadhaar' | 'pan' | 'resume' | 'payslip' | 'unknown'
    """
    t = text.lower()

    aadhaar_signals = ["aadhaar", "आधार", "uid", "unique identification",
                       "uidai", "government of india", "enrolment"]
    pan_signals     = ["permanent account number", "income tax department",
                       "pan", "govt. of india", "आयकर विभाग"]
    payslip_signals = ["uan", "universal account number", "epf", "pf no",
                       "provident fund", "payslip", "salary slip",
                       "net pay", "gross salary", "basic pay", "deductions"]
    resume_signals  = ["objective", "summary", "experience", "education",
                       "skills", "projects", "work history", "employment",
                       "linkedin", "github", "curriculum vitae", "resume",
                       "cgpa", "b.tech", "m.tech", "mba", "internship"]

    scores = {
        "aadhaar": sum(1 for s in aadhaar_signals if s in t),
        "pan":     sum(1 for s in pan_signals if s in t),
        "payslip": sum(1 for s in payslip_signals if s in t),
        "resume":  sum(1 for s in resume_signals if s in t),
    }
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "unknown"


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 3 — FIELD EXTRACTORS
# ══════════════════════════════════════════════════════════════════════════════

# ── helpers ───────────────────────────────────────────────────────────────────

def _after_label(text: str, *labels) -> str:
    """Return the value on the same line after any of the given labels."""
    for label in labels:
        pattern = re.compile(
            rf'{re.escape(label)}\s*[:\-]?\s*(.+)',
            re.IGNORECASE
        )
        m = pattern.search(text)
        if m:
            val = m.group(1).strip().split("\n")[0].strip()
            if val:
                return val
    return ""


def _next_nonempty_line(text: str, *labels) -> str:
    """Return the first non-empty line AFTER a line that contains any label."""
    lines = [l.strip() for l in text.splitlines()]
    for i, line in enumerate(lines):
        if any(lbl.lower() in line.lower() for lbl in labels):
            for j in range(i + 1, min(i + 4, len(lines))):
                if lines[j]:
                    return lines[j]
    return ""


def _clean_name(s: str) -> str:
    """Remove noise tokens commonly mixed in with names from OCR."""
    noise = ["male", "female", "dob", "date", "year", "महिला", "पुरुष",
             "help", "www", "uidai", "gov", "in"]
    parts = [w for w in s.split() if w.lower() not in noise and len(w) > 1]
    return " ".join(parts[:5])   # cap at 5 words


# ── Aadhaar ───────────────────────────────────────────────────────────────────

def extract_aadhaar(text: str) -> dict:
    """
    Fields: name, father_name, dob, aadhaar_number
    """
    result = {
        "document_type": "aadhaar",
        "name": "",
        "father_name": "",
        "dob": "",
        "aadhaar_number": "",
    }

    lines = [l.strip() for l in text.splitlines() if l.strip()]

    # ── Aadhaar number: 12-digit in groups of 4 ──────────────────────────────
    # Also handles masked (XXXX XXXX 1234) and plain 12-digit
    aadhaar_pat = re.search(
        r'\b([X\d]{4}[\s\-]?[X\d]{4}[\s\-]?\d{4})\b',
        text, re.IGNORECASE
    )
    if aadhaar_pat:
        result["aadhaar_number"] = re.sub(r'\s', ' ', aadhaar_pat.group(1)).strip()

    # ── DOB ──────────────────────────────────────────────────────────────────
    dob_pat = re.search(
        r'\b(\d{2}[\/\-]\d{2}[\/\-]\d{4})\b',
        text
    )
    if not dob_pat:
        dob_pat = re.search(r'(\d{4})',text)   # year fallback for DOB:YYYY format
        if dob_pat and _after_label(text, "dob", "date of birth", "जन्म"):
            result["dob"] = _after_label(text, "dob", "date of birth", "जन्म")
    if dob_pat:
        result["dob"] = result["dob"] or dob_pat.group(1)

    # ── Name & Father's name ─────────────────────────────────────────────────
    # Strategy: on Aadhaar the name appears before DOB/gender line
    # Father/husband name often follows on the next line
    for i, line in enumerate(lines):
        line_lower = line.lower()

        # DOB line found — name is usually 1–3 lines above
        if re.search(r'\d{2}[\/\-]\d{2}[\/\-]\d{4}', line) or "date of birth" in line_lower:
            # Walk backwards to find name
            for back in range(1, 4):
                candidate = lines[i - back] if i - back >= 0 else ""
                # Skip lines that are clearly not names
                if candidate and not re.search(r'\d{6,}', candidate):
                    if not any(x in candidate.lower() for x in
                               ["government", "india", "uidai", "authority"]):
                        result["name"] = _clean_name(candidate)
                        # Father's name one line above name
                        if i - back - 1 >= 0:
                            prev = lines[i - back - 1]
                            if prev and not re.search(r'\d{6,}', prev):
                                result["father_name"] = _clean_name(prev)
                        break

        # Explicit label matching
        if "father" in line_lower or "s/o" in line_lower or "d/o" in line_lower \
                or "पिता" in line or "husband" in line_lower:
            val = _after_label(line, "father", "s/o", "d/o", "husband", "पिता")
            if not val and i + 1 < len(lines):
                val = lines[i + 1]
            if val:
                result["father_name"] = _clean_name(val)

    # If name still empty, try inline label
    if not result["name"]:
        result["name"] = _clean_name(_after_label(text, "name", "नाम"))

    return result


# ── PAN ───────────────────────────────────────────────────────────────────────

def extract_pan(text: str) -> dict:
    result = {
        "document_type": "pan",
        "pan_number": "",
        "name": "",
        "father_name": "",
        "dob": "",
    }

    lines = [l.strip() for l in text.splitlines() if l.strip()]

    # ── PAN number ────────────────────────────────────────────────────────────
    pan_pat = re.search(r'\b([A-Z]{5}[0-9]{4}[A-Z])\b', text)
    if pan_pat:
        result["pan_number"] = pan_pat.group(1)

    # ── DOB ───────────────────────────────────────────────────────────────────
    dob_pat = re.search(r'\b(\d{2}[\/\-]\d{2}[\/\-]\d{4})\b', text)
    if dob_pat:
        result["dob"] = dob_pat.group(1)

    # ── Shared helpers ────────────────────────────────────────────────────────
    NOISE = {
        "income tax department", "govt. of india", "govt of india",
        "government of india", "incometaxdepartment", "govtofindia",
        "permanent account number", "permanent account number card",
        "permanent account", "signature", "/signature", "hrarr", "hrrr",
    }

    def is_name_line(s: str) -> bool:
        sl = s.lower().strip()
        if not sl or len(sl) < 2: return False
        if any(n in sl for n in NOISE): return False
        if re.search(r'\d{2}[\/\-]\d{2}[\/\-]\d{4}', s): return False  # DOB line
        if re.match(r'^[A-Z]{5}\d{4}[A-Z]$', s.strip()): return False  # PAN number
        if re.match(r'^\d+$', s.strip()): return False                  # pure digits
        if s.strip().startswith('/'): return False                       # label like /Name
        if re.match(r'^[a-z]{2,4}[A-Z]', s.strip()): return False      # fused Hindi+English label
        return True

    # ── Layout detection ──────────────────────────────────────────────────────
    # Layout B (bilingual card): has lines like "/Name", "/Father'sName",
    #   or fused OCR labels like "fuaIFather'sName" — value is on the NEXT line.
    # Layout A (classic card): name and father appear directly above the DOB line.

    has_label_markers = any(
        l.strip().startswith('/')
        or re.match(r'^[a-z]{2,4}[A-Z]', l.strip())
        for l in lines
    )

    if has_label_markers:
        # ── Layout B: label → value pairs ─────────────────────────────────
        father_re = re.compile(
            r"(?:/\s*)?(?:[a-z\u0900-\u097F]*\s*)?(?:father|parent)",
            re.IGNORECASE
        )
        name_re = re.compile(
            r"(?:/\s*)?(?:[a-z\u0900-\u097F]*\s*)?name",
            re.IGNORECASE
        )
        for i, line in enumerate(lines):
            nxt = lines[i + 1].strip() if i + 1 < len(lines) else ""
            if father_re.search(line) and is_name_line(nxt):
                result["father_name"] = nxt.title()
            elif name_re.search(line) and not father_re.search(line) \
                    and is_name_line(nxt) and not result["name"]:
                result["name"] = nxt.title()
    else:
        # ── Layout A: walk backwards from DOB line ─────────────────────────
        dob_idx = next(
            (i for i, l in enumerate(lines)
             if re.search(r'\b\d{2}[\/\-]\d{2}[\/\-]\d{4}\b', l)), None
        )
        if dob_idx is not None:
            candidates = []
            for i in range(dob_idx - 1, max(dob_idx - 5, -1), -1):
                if is_name_line(lines[i]):
                    candidates.insert(0, lines[i])
                else:
                    break
            if len(candidates) >= 2:
                result["name"]        = candidates[-2].title()
                result["father_name"] = candidates[-1].title()
            elif len(candidates) == 1:
                result["name"] = candidates[0].title()

    return result

# ── Payslip ───────────────────────────────────────────────────────────────────

def extract_payslip(text: str) -> dict:
    """
    Fields: uan_number
    """
    result = {
        "document_type": "payslip",
        "uan_number": "",
    }

    # UAN is 12 digits
    uan_pat = re.search(
        r'(?:uan|universal\s+account\s+number)[^\d]*(\d{12})',
        text, re.IGNORECASE
    )
    if not uan_pat:
        # Sometimes appears as a standalone 12-digit number near "UAN" label
        uan_pat = re.search(r'\bUAN\b[^\n]*\n?[^\d]*(\d{12})', text, re.IGNORECASE)
    if not uan_pat:
        # Last resort: any standalone 12-digit number
        uan_pat = re.search(r'\b(\d{12})\b', text)

    if uan_pat:
        result["uan_number"] = uan_pat.group(1)

    return result


# ── Resume ────────────────────────────────────────────────────────────────────

# IT keywords for experience classification
IT_KEYWORDS = {
    "python", "java", "javascript", "typescript", "c++", "c#", "golang", "ruby",
    "php", "swift", "kotlin", "rust", "scala", "r", "matlab",
    "react", "angular", "vue", "node", "django", "flask", "spring", "fastapi",
    "sql", "mysql", "postgresql", "mongodb", "redis", "elasticsearch",
    "aws", "azure", "gcp", "docker", "kubernetes", "terraform", "jenkins",
    "git", "linux", "devops", "ci/cd", "microservices", "api", "rest",
    "machine learning", "deep learning", "ai", "nlp", "data science",
    "software engineer", "developer", "programmer", "sde", "swe",
    "backend", "frontend", "full stack", "fullstack", "cloud",
    "data analyst", "data engineer", "ml engineer", "devops engineer",
    "network", "cybersecurity", "it support", "system admin", "database",
    "html", "css", "android", "ios", "mobile", "embedded", "iot",
    "tableau", "power bi", "spark", "hadoop", "kafka", "airflow",
}


def _years_from_duration(duration_str: str) -> float:
    """Convert '2 years 3 months' or '2019-2022' or 'Jan 2020 - Dec 2022' → years."""
    duration_str = duration_str.lower()

    # Pattern: explicit "X years Y months"
    yrs  = re.search(r'(\d+)\s*year', duration_str)
    mons = re.search(r'(\d+)\s*month', duration_str)
    if yrs or mons:
        return (int(yrs.group(1)) if yrs else 0) + \
               (int(mons.group(1)) / 12 if mons else 0)

    # Pattern: YYYY–YYYY or YYYY–Present
    range_pat = re.search(
        r'(20\d{2}|19\d{2})\s*[\-–to]+\s*(20\d{2}|19\d{2}|present|current|now)',
        duration_str
    )
    if range_pat:
        start = int(range_pat.group(1))
        end_str = range_pat.group(2)
        import datetime
        end = datetime.datetime.now().year if end_str in ("present", "current", "now") \
              else int(end_str)
        return max(0.0, float(end - start))

    return 0.0


def _is_it_role(text_block: str) -> bool:
    t = text_block.lower()
    return any(kw in t for kw in IT_KEYWORDS)


def extract_resume(text: str) -> dict:
    """
    Fields: name, email, phone, experience_jobs (list), total_it_experience_years
    """
    result = {
        "document_type": "resume",
        "name": "",
        "email": "",
        "phone": "",
        "experience_jobs": [],
        "total_it_experience_years": 0.0,
    }

    lines = [l.strip() for l in text.splitlines() if l.strip()]

    # ── Email ─────────────────────────────────────────────────────────────────
    email_pat = re.search(r'[\w.\-+]+@[\w.\-]+\.[a-zA-Z]{2,}', text)
    if email_pat:
        result["email"] = email_pat.group(0).lower()

    # ── Phone ─────────────────────────────────────────────────────────────────
    phone_pat = re.search(
        r'(?:\+91[\s\-]?)?[6-9]\d{9}|'      # Indian mobile
        r'\+?[\d][\d\s\-\(\)]{8,14}[\d]',   # international
        text
    )
    if phone_pat:
        result["phone"] = re.sub(r'[\s\-\(\)]', '', phone_pat.group(0))

    # ── Name (usually the very first non-empty line of a resume) ─────────────
    # Skip lines that look like labels, emails, phones, or addresses
    for line in lines[:6]:
        if not re.search(r'[@\d]', line) and len(line.split()) <= 6 and len(line) > 3:
            if not any(x in line.lower() for x in
                       ["resume", "curriculum", "vitae", "cv", "objective",
                        "summary", "contact", "linkedin", "github"]):
                result["name"] = line.strip()
                break

    # ── Experience section parsing ────────────────────────────────────────────
    # Find where "Experience" / "Work History" / "Employment" section starts
    exp_section_start = -1
    exp_section_end   = len(lines)

    section_headers = re.compile(
        r'^(work\s+experience|professional\s+experience|employment|'
        r'experience|work\s+history|career\s+history)',
        re.IGNORECASE
    )
    end_headers = re.compile(
        r'^(education|skills|projects|certifications|achievements|'
        r'awards|languages|interests|hobbies|references|publications)',
        re.IGNORECASE
    )

    for i, line in enumerate(lines):
        if section_headers.match(line) and exp_section_start == -1:
            exp_section_start = i + 1
        elif end_headers.match(line) and exp_section_start != -1:
            exp_section_end = i
            break

    exp_lines = lines[exp_section_start:exp_section_end] if exp_section_start != -1 else lines

    # ── Parse individual job entries ──────────────────────────────────────────
    # Heuristic: a job title line is often followed by company + date range
    date_range_pat = re.compile(
        r'((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[\w\s,]*'
        r'\d{4}\s*[\-–to]+\s*'
        r'(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[\w\s,]*\d{4}'
        r'|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[\w\s,]*\d{4}'
        r'\s*[\-–to]+\s*(?:present|current|now)'
        r'|\d{4}\s*[\-–to]+\s*(?:\d{4}|present|current|now))',
        re.IGNORECASE
    )

    jobs = []
    i = 0
    while i < len(exp_lines):
        line = exp_lines[i]
        date_match = date_range_pat.search(line)

        if date_match:
            duration_str = date_match.group(1)
            # Title is usually on this line (before date) or previous line
            title_line = line[:date_match.start()].strip() or \
                         (exp_lines[i - 1] if i > 0 else "")
            # Company might be on the next line
            company = exp_lines[i + 1].strip() if i + 1 < len(exp_lines) else ""

            # Collect description lines (bullet points, skills mentioned)
            desc_lines = []
            j = i + 2
            while j < len(exp_lines) and not date_range_pat.search(exp_lines[j]):
                if not end_headers.match(exp_lines[j]):
                    desc_lines.append(exp_lines[j])
                j += 1

            description = " ".join(desc_lines)
            years = _years_from_duration(duration_str)
            is_it  = _is_it_role(title_line + " " + description)

            jobs.append({
                "title":       title_line.strip(),
                "company":     company,
                "duration":    duration_str.strip(),
                "years":       round(years, 1),
                "is_it_role":  is_it,
            })
            i = j
        else:
            i += 1

    result["experience_jobs"] = jobs

    # ── Total IT experience ───────────────────────────────────────────────────
    it_years = sum(job["years"] for job in jobs if job["is_it_role"])
    result["total_it_experience_years"] = round(it_years, 1)

    # Fallback: grep for "X years of experience" anywhere in text
    if it_years == 0:
        exp_mention = re.search(
            r'(\d+(?:\.\d+)?)\+?\s*years?\s+(?:of\s+)?(?:it\s+)?experience',
            text, re.IGNORECASE
        )
        if exp_mention:
            result["total_it_experience_years"] = float(exp_mention.group(1))

    return result


# ══════════════════════════════════════════════════════════════════════════════
#  PUBLIC API
# ══════════════════════════════════════════════════════════════════════════════

def extract_fields(filepath: str, doc_type: str = "auto") -> dict:
    """
    Main entry point.

    Args:
        filepath : path to Aadhaar / PAN / Resume / Payslip file
        doc_type : 'auto' (default) | 'aadhaar' | 'pan' | 'resume' | 'payslip'

    Returns:
        dict with extracted fields + metadata
    """
    if not os.path.exists(filepath):
        return {"error": f"File not found: {filepath}"}

    try:
        raw_text = get_raw_text(filepath)
    except Exception as e:
        return {"error": f"Text extraction failed: {e}"}

    if not raw_text.strip():
        return {"error": "No text could be extracted from the file"}

    if doc_type == "auto":
        doc_type = detect_doc_type(raw_text)

    extractor_map = {
        "aadhaar": extract_aadhaar,
        "pan":     extract_pan,
        "payslip": extract_payslip,
        "resume":  extract_resume,
    }

    if doc_type not in extractor_map:
        return {
            "document_type": "unknown",
            "raw_text_preview": raw_text[:500],
            "warning": "Could not identify document type. Pass doc_type manually."
        }

    fields = extractor_map[doc_type](raw_text)
    fields["_raw_text_preview"] = raw_text[:300] + "..." if len(raw_text) > 300 else raw_text
    return fields


# ══════════════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python document_extractor.py <file_path> [doc_type]")
        print("       doc_type: aadhaar | pan | resume | payslip | auto (default)")
        sys.exit(1)

    file_path = sys.argv[1]
    forced_type = sys.argv[2] if len(sys.argv) > 2 else "auto"

    print(f"\n{'='*60}")
    print(f"  File     : {file_path}")
    print(f"  Doc Type : {forced_type}")
    print(f"{'='*60}\n")

    output = extract_fields(file_path, forced_type)
    print(json.dumps(output, indent=2, ensure_ascii=False))