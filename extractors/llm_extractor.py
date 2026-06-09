# extractors/llm_extractor.py
import requests
import json
import re
from datetime import datetime
from dateutil.relativedelta import relativedelta  # pip install python-dateutil

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "phi4-mini"

# -----------------------------------------------------------------------
# UAN regex — 12-digit number preceded by "UAN" label.
# Handles: "UAN: 123456789012", "UAN No 123456789012", "UAN\t123456789012"
# -----------------------------------------------------------------------
UAN_REGEX = re.compile(
    r"\bUAN(?:\s*(?:No\.?|Number|#))?\s*[:\-]?\s*(\d{12})\b",
    re.IGNORECASE
)

# -----------------------------------------------------------------------
# Schemas — only fields we actually need per doc type
# -----------------------------------------------------------------------
SCHEMAS = {
    "resume": {
        "name": "string - full name of the candidate",
        "email": "string",
        "phone": "string",
        "skills": "list of strings - extract ALL technical skills from ALL subsections including Technical, Tools, Technologies, Frameworks, Databases. Each item 1-4 words. Exclude metrics, methodologies, and soft skills.",
        "education": [{
            "degree": "string - degree name",
            "institution": "string - university or college name",
            "year": "string - graduation year only e.g. 2020 or June 2020, not the full date range"}],
        "experience": [{"company": "string", "role": "string", "duration": "string"}],
        "total_experience_years": "number or null - IT/software experience ONLY"
    },
    "payslip": {
        "uan": "string - Universal Account Number, exactly 12 digits"
    },
    "experience_letter": {
        "employee_name": "string",
        "employer_name": "string",
        "designation": "string",
        "joining_date": "string",
        "relieving_date": "string",
        "reason_for_leaving": "string or null"
    },
    "degree_certificate": {
        "candidate_name": "string",
        "degree": "string",
        "institution": "string",
        "year_of_passing": "string",
        "specialization": "string or null"
    }
}

CHAR_LIMITS = {
    "resume": 5000,
    "payslip": 3000,
    "experience_letter": 2000,
    "degree_certificate": 2000,
}


# -----------------------------------------------------------------------
# UAN extraction — regex first, LLM only as fallback
# -----------------------------------------------------------------------
def extract_uan_from_text(text: str) -> str | None:
    """Try regex first. Returns 12-digit UAN string or None."""
    m = UAN_REGEX.search(text)
    if m:
        return m.group(1)
    return None


def extract_uan_with_llm(raw_text: str) -> str | None:
    """LLM fallback when regex found nothing."""
    prompt = f"""Extract the UAN (Universal Account Number) from this payslip.
UAN is always exactly 12 digits and is labeled 'UAN' or 'UAN No'.

Return ONLY a JSON object: {{"uan": "123456789012"}} or {{"uan": null}} if not found.
No explanation, no markdown.

PAYSLIP TEXT:
{raw_text[:2000]}"""

    try:
        response = requests.post(
            OLLAMA_URL,
            json={"model": MODEL, "prompt": prompt, "stream": False},
            timeout=60
        )
        response.raise_for_status()
        parsed = _parse_response(response.json()["response"])
        val = parsed.get("uan")
        # Validate it's actually 12 digits
        if val and re.fullmatch(r"\d{12}", str(val).strip()):
            return str(val).strip()
        return None
    except Exception:
        return None


# -----------------------------------------------------------------------
# IT experience calculation
# -----------------------------------------------------------------------

# Keywords that identify IT / software roles
_IT_ROLE_KEYWORDS = re.compile(
    r"\b("
    r"software|developer|engineer|programmer|devops|data\s*scientist|"
    r"data\s*analyst|machine\s*learning|ml\s*engineer|ai\s*engineer|"
    r"backend|frontend|full[\s-]?stack|cloud|sre|qa|quality\s*assurance|"
    r"test\s*engineer|mobile\s*developer|android|ios\s*developer|"
    r"database\s*admin|dba|system\s*admin|network\s*engineer|"
    r"it\s*support|technical\s*support|solutions\s*architect|"
    r"scrum|agile|product\s*manager|tech\s*lead|engineering\s*manager|"
    r"3d\s*artist|3d\s*generalist|3d\s*specialist|3d\s*lead|3d\s*consultant|"
    r"3d\s*technical|technical\s*specialist|technical\s*lead|"
    r"unity|unreal|vr|ar|xr|mixed\s*reality|augmented\s*reality|"
    r"visuali[sz]ation|animator|animation|rendering|technical\s*artist|"
    r"ui\s*ux|ux\s*designer|ui\s*designer|graphic\s*designer|"
    r"game\s*developer|game\s*designer|level\s*designer|"
    r"it\s*specialist|it\s*consultant|it\s*manager|it\s*analyst|"
    r"systems\s*engineer|integration\s*engineer|automation\s*engineer|"
    r"embedded\s*engineer|firmware|hardware\s*engineer|"
    r"security\s*engineer|cybersecurity|penetration\s*tester|"
    r"data\s*engineer|etl|bi\s*developer|business\s*intelligence|"
    r"salesforce|sap\s*consultant|erp|crm\s*developer|"
    r"scrum\s*master|agile\s*coach|release\s*engineer"
    r")\b",
    re.IGNORECASE
)

# Keywords that clearly indicate NON-IT roles
_NON_IT_ROLE_KEYWORDS = re.compile(
    r"\b("
    r"civil\s*engineer|structural|construction|site\s*engineer|"
    r"teacher|professor|lecturer|doctor|nurse|accountant|"
    r"sales\s*executive|marketing|retail|store\s*manager|"
    r"driver|operator|technician|electrician|plumber|mechanic"
    r")\b",
    re.IGNORECASE
)


def _parse_duration_to_months(duration: str) -> int | None:
    """
    Parse a duration string like:
      "Jan 2021 - Mar 2023"
      "2020 - Present"
      "June 2019 – Current"
      "2 years 3 months"
    Returns total months or None if unparseable.
    """
    if not duration:
        return None

    # LLM sometimes returns duration as a list e.g. ["Jan 2021", "Mar 2023"]
    # or as an int/float — normalise to string before any processing
    if isinstance(duration, list):
        duration = " - ".join(str(x) for x in duration if x)
    elif not isinstance(duration, str):
        duration = str(duration)

    duration = duration.strip()

    # "X years Y months" / "X years" / "Y months"
    m = re.match(
        r"(\d+)\s*years?\s*(?:(\d+)\s*months?)?",
        duration,
        re.IGNORECASE
    )
    if m:
        y = int(m.group(1))
        mo = int(m.group(2)) if m.group(2) else 0
        return y * 12 + mo

    # Date range patterns
    month_map = {
        "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
        "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12
    }

    # Regex for "Mon YYYY – Mon YYYY/Present"
    # Also handles "Mon-YYYY" (hyphen between month and year)
    range_pat = re.compile(
        r"((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?)[\s-]*(\d{4})"
        r"\s*[-–]\s*"
        r"((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?[\s-]*\d{4}|Present|Current|Now)",
        re.IGNORECASE
    )
    rm = range_pat.search(duration)
    if rm:
        start_mon = month_map.get(rm.group(1)[:3].lower(), 1)
        start_year = int(rm.group(2))
        end_str = rm.group(3).strip().lower()
        if end_str in ("present", "current", "now"):
            end_dt = datetime.today()
        else:
            ep = re.match(
                r"(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?[\s-]*(\d{4})",
                end_str,
                re.IGNORECASE
            )
            if ep:
                end_dt = datetime(int(ep.group(2)), month_map.get(ep.group(1)[:3].lower(), 1), 1)
            else:
                return None
        start_dt = datetime(start_year, start_mon, 1)
        diff = relativedelta(end_dt, start_dt)
        return diff.years * 12 + diff.months

    # "YYYY – YYYY" or "YYYY – Present"
    year_pat = re.compile(
        r"(\d{4})\s*[-–]\s*(\d{4}|Present|Current)",
        re.IGNORECASE
    )
    ym = year_pat.search(duration)
    if ym:
        start_y = int(ym.group(1))
        end_str = ym.group(2).strip().lower()
        end_y = datetime.today().year if end_str in ("present", "current") else int(end_str)
        return max(0, (end_y - start_y) * 12)

    # "to till date", "to date", "till date", "to present" etc.
    present_pat = re.compile(
        r"((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?[\s-]*\d{4}|\d{4})"
        r"\s+(?:to\s+)?(?:till\s+)?(?:date|present|now|current)",
        re.IGNORECASE
    )
    pm = present_pat.search(duration)
    if pm:
        start_str = pm.group(1).strip()
        ep2 = re.match(
            r"((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*)[\s-]*(\d{4})",
            start_str, re.IGNORECASE
        )
        if ep2:
            month_map2 = {
                "jan":1,"feb":2,"mar":3,"apr":4,"may":5,"jun":6,
                "jul":7,"aug":8,"sep":9,"oct":10,"nov":11,"dec":12
            }
            sm = month_map2.get(ep2.group(1)[:3].lower(), 1)
            sy = int(ep2.group(2))
            start_dt2 = datetime(sy, sm, 1)
            diff2 = relativedelta(datetime.today(), start_dt2)
            return diff2.years * 12 + diff2.months
        else:
            yr_m = re.search(r"\d{4}", start_str)
            if yr_m:
                return max(0, (datetime.today().year - int(yr_m.group(0))) * 12)

    return None


def _calculate_it_experience_from_entries(experience: list[dict]) -> float | None:
    """
    Given a list of experience dicts {company, role, duration},
    sum months for roles that look like IT roles.
    """
    total_months = 0
    found_any = False

    for entry in experience:
        if not isinstance(entry, dict):
            continue
        role = entry.get("role") or ""
        company = entry.get("company") or ""
        duration = entry.get("duration") or ""

        combined = f"{role} {company}"

        # Skip if explicitly non-IT
        if _NON_IT_ROLE_KEYWORDS.search(combined):
            continue

        # Only count if role looks like IT
        if not _IT_ROLE_KEYWORDS.search(combined):
            # Ambiguous — if company name gives no signal, skip conservatively
            continue

        months = _parse_duration_to_months(duration)
        if months is not None and months > 0:
            total_months += months
            found_any = True

    if not found_any:
        return None

    # Round to 1 decimal (e.g. 14 months → 1.2 years)
    return round(total_months / 12, 1)


def _calculate_it_experience_with_company_fallback(experience: list) -> float | None:
    """
    When keyword matching finds no IT roles but entries with durations exist,
    pass them to the LLM with company context to decide.
    """
    if not experience:
        return None
    fallback_text = "\n".join(
        f"{e.get('role', '')} at {e.get('company', '')} ({e.get('duration', '')})"
        for e in experience if e.get("duration")
    )
    if not fallback_text.strip():
        return None
    return _calculate_it_experience_with_llm(fallback_text)


def _calculate_it_experience_with_llm(exp_text: str) -> float | None:
    """
    LLM fallback for IT experience calculation when regex parsing fails.
    Only called when structured entries couldn't be parsed from duration strings.
    """
    if not exp_text or not exp_text.strip():
        return None

    prompt = f"""You are a resume parser.

Given the EXPERIENCE section below, calculate the total years the candidate has worked
in IT / software / technology roles ONLY.

Rules:
- Count: software developer, data scientist, ML engineer, backend/frontend engineer,
  DevOps, cloud, QA, IT support, data analyst, and similar tech roles.
- Do NOT count: civil engineering, construction, teaching non-tech subjects, sales,
  marketing, or any non-IT domain.
- If duration is given as date ranges (e.g. "Jan 2022 - Mar 2024"), calculate the difference.
- If duration is stated explicitly (e.g. "2 years 3 months"), use that.
- Return ONLY: {{"it_experience_years": 3.5}} or {{"it_experience_years": null}} if none found.
- No explanation, no markdown.

EXPERIENCE SECTION:
{exp_text[:2000]}"""

    try:
        response = requests.post(
            OLLAMA_URL,
            json={"model": MODEL, "prompt": prompt, "stream": False},
            timeout=60
        )
        response.raise_for_status()
        parsed = _parse_response(response.json()["response"])
        val = parsed.get("it_experience_years")
        return float(val) if val is not None else None
    except Exception:
        return None


# -----------------------------------------------------------------------
# Resume helpers (unchanged from original)
# -----------------------------------------------------------------------
def preprocess_resume_text(text: str) -> str:
    text = re.sub(r'[●•◆▪■◦‣⁃]', '\n- ', text)
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'([A-Za-z]):', r'\1: ', text)
    text = re.sub(r',([^\s])', r', \1', text)
    return text.strip()


def _get_section(text: str, start_patterns, end_patterns):
    lines = text.splitlines()
    start_idx = None

    for i, line in enumerate(lines):
        if any(re.search(p, line, re.IGNORECASE) for p in start_patterns):
            start_idx = i + 1
            break

    if start_idx is None:
        return ""

    end_idx = len(lines)
    for j in range(start_idx, len(lines)):
        if any(re.search(p, lines[j], re.IGNORECASE) for p in end_patterns):
            end_idx = j
            break

    return "\n".join(lines[start_idx:end_idx]).strip()

def extract_resume_sections(text: str) -> dict:
    text = preprocess_resume_text(text)

    section_pattern = re.compile(
        r'\n(SKILLS?|TECHNICAL SKILLS?|EDUCATION|EXPERIENCE|WORK EXPERIENCE|'
        r'EMPLOYMENT|PROJECTS?|CERTIFICATIONS?|SUMMARY|OBJECTIVE|SUMMARY OF SKILLS?)\s*\n',
        re.IGNORECASE
    )

    parts = section_pattern.split(text)
    sections = {}
    if len(parts) > 1:
        for i in range(1, len(parts) - 1, 2):
            sections[parts[i].strip().upper()] = parts[i + 1].strip()

    header_text = text[:1000]
    result = {}
    result.update(_extract_contact(header_text))

    skills_text = _slice_between_headings(
        text,
        start_names=(r"SUMMARY OF SKILLS", r"SKILLS", r"TECHNICAL SKILLS"),
        end_names=(r"CONTACT", r"EDUCATION", r"EXPERIENCE", r"WORK EXPERIENCE",
                   r"OBJECTIVE", r"ROLES AND RESPONSIBILITIES", r"AWARDS",
                   r"CERTIFICATES", r"HOBBIES", r"INTEREST AND STRENGTHS",
                   r"TECH STACK", r"CERTIFICATIONS")
    )
    result["skills"] = _extract_skills(skills_text) if skills_text else []

    edu_text = sections.get("EDUCATION") or ""
    edu_list = _extract_education(edu_text) if edu_text else []

    exp_text = _slice_between_headings(
        text,
        start_names=(r"EXPERIENCE", r"WORK EXPERIENCE", r"PROFESSIONAL EXPERIENCE",
                     r"EMPLOYMENT HISTORY", r"EMPLOYMENT", r"CAREER HISTORY",
                     r"ROLES AND RESPONSIBILITIES"),
        end_names=(r"SKILLS", r"SUMMARY OF SKILLS", r"EDUCATION", r"AWARDS",
                   r"CERTIFICATES", r"HOBBIES")
    )
    exp_list = _extract_experience(exp_text) if exp_text else []
    if not exp_list and exp_text:
        exp_list = _extract_experience_blocks(exp_text)

    if not edu_list:
        edu_list = _extract_education(text)

    if not exp_list:
        exp_list = _extract_experience(text)

    result["education"] = edu_list or []
    result["experience"] = exp_list or []

    # FIXED: IT-only experience calculation — try structured parsing first
    it_years = _calculate_it_experience_from_entries(result["experience"])
    if it_years is None and result["experience"]:
        it_years = _calculate_it_experience_with_company_fallback(result["experience"])
    if it_years is None and exp_text:
        # Fallback to LLM only when structured parsing yielded nothing
        it_years = _calculate_it_experience_with_llm(exp_text)
    result["total_experience_years"] = it_years

    result["doc_type"] = "resume"
    return _normalize_resume_result(result)


def _fallback_name_from_email_context(text: str, email: str | None) -> str | None:
    if not email:
        return None

    lines = [l.strip() for l in text.splitlines() if l.strip()]
    idx = None
    for i, line in enumerate(lines):
        if email in line:
            idx = i
            break
    if idx is None:
        return None

    for offset in range(1, 4):
        j = idx - offset
        if j < 0:
            break
        candidate = lines[j]
        if re.search(r'(?:resume|curriculum vitae|cv)', candidate, re.IGNORECASE):
            continue
        if any(x in candidate.lower() for x in ['@', 'http', 'www.']):
            continue
        if len(re.findall(r'[A-Za-z]', candidate)) >= 4 and len(candidate) <= 70:
            return candidate

    return None


def _extract_contact(text: str) -> dict:
    email_match = re.search(r'[\w.+-]+@[\w-]+\.[a-z]{2,}', text, re.IGNORECASE)
    email = email_match.group(0) if email_match else None

    phone_match = re.search(
        r'(?:\+?91[-.\s]?)?(?:\(?0\)?[-.\s]?)?[6-9]\d{9}'
        r'|'
        r'\b[6-9]\d{2}[-.\s]\d{3}[-.\s]\d{4}\b',
        text
    )
    phone = phone_match.group(0) if phone_match else None

    lines = [l.strip() for l in text.splitlines() if l.strip()]

    heading_words = {
        "PROFILE", "CONTACT", "SUMMARY", "SUMMARY OF SKILLS",
        "SKILLS", "TECHNICAL SKILLS", "STRENGTH", "HOBBIES",
        "EDUCATION", "EXPERIENCE", "WORK EXPERIENCE",
        "AWARDS", "PROJECTS", "CERTIFICATIONS", "OBJECTIVE",
        "KEY SKILLS", "PROFESSIONAL SUMMARY", "ROLES AND RESPONSIBILITIES",
    }

    # Section headings that should never be treated as names
    _NEVER_A_NAME = {
        "PROFILE", "CONTACT", "SUMMARY", "SKILLS", "TECHNICAL SKILLS",
        "STRENGTH", "HOBBIES", "EDUCATION", "EXPERIENCE", "WORK EXPERIENCE",
        "AWARDS", "PROJECTS", "CERTIFICATIONS", "OBJECTIVE", "KEY SKILLS",
        "PROFESSIONAL SUMMARY", "ROLES AND RESPONSIBILITIES",
        "SUMMARY OF SKILLS",
    }

    def is_heading(line: str) -> bool:
        upper = line.upper()
        return upper in heading_words or upper.rstrip(':') in heading_words

    def is_strong_name(line: str) -> bool:
        if line.strip().upper() in _NEVER_A_NAME:
            return False
        if any(ch.isdigit() for ch in line):
            return False
        low = line.lower()
        if any(x in low for x in (
            "@", "http", "www.", "linkedin",
            "specialist", "consultant", "manager", "lead",
            "analyst", "developer", "engineer",
        )):
            return False
        words = [w for w in line.split() if w]
        if not (2 <= len(words) <= 4):
            return False
        return True

    name = None

    # Pass 1: lines before the first section heading (standard resume layout)
    header_lines = []
    for line in lines:
        if is_heading(line):
            break
        header_lines.append(line)
    for candidate in header_lines:
        candidate = candidate.strip()
        if is_strong_name(candidate):
            name = candidate
            break

    # Pass 2: scan ALL lines for a name-shaped line.
    # Needed for PDFs where name appears AFTER section headings in text order
    # (e.g. side-panel layout where left column is extracted after right column).
    # A "name-shaped" line: 1-4 words, all purely alphabetic words, not a heading,
    # not a job title blurb, reasonably short.
    if not name:
        def _looks_like_name(line: str) -> bool:
            s = line.strip()
            if s.upper() in _NEVER_A_NAME:
                return False
            if any(ch.isdigit() for ch in s):
                return False
            low = s.lower()
            if any(x in low for x in (
                "@", "http", "www.", "linkedin",
                "specialist", "consultant", "manager", "lead",
                "analyst", "developer", "engineer",
                "driven", "professional", "experience", "result",
                "dedicated", "motivated", "passionate", "objective",
            )):
                return False
            words = [w for w in s.split() if w]
            if not (1 <= len(words) <= 4):
                return False
            # All words must be purely alphabetic (names don't have symbols)
            if not all(re.fullmatch(r"[A-Za-z]+", w) for w in words):
                return False
            # At least one word must be >= 3 chars (avoids initials-only lines)
            if not any(len(w) >= 3 for w in words):
                return False
            # Prefer Title Case or ALL CAPS names, not random mixed case
            title_or_caps = all(
                w[0].isupper() for w in words
            )
            return title_or_caps

        for line in lines:
            if _looks_like_name(line):
                name = line.strip()
                break
    if name:
        name = re.sub(
            r'\(\s*\d+\+?\s*Years?\s*\)',
            '',
            name,
            flags=re.IGNORECASE
        ).strip()
    return {
        "name": name,
        "email": email,
        "phone": phone,
    }


def _is_heading(line: str) -> bool:
    s = line.strip()
    if not s:
        return False
    if len(s) > 45:
        return False
    if re.search(r'[.]{2,}', s):
        return False
    words = s.split()
    return (
        s.isupper()
        or (len(words) <= 4 and all(w[0].isupper() for w in words if w[0].isalpha()))
    )


def _slice_between_headings(text: str, start_names: tuple, end_names: tuple) -> str:
    lines = text.splitlines()
    start = None
    for i, line in enumerate(lines):
        stripped = line.strip().rstrip(":")
        if any(re.search(rf"^{p}$", stripped, re.IGNORECASE) for p in start_names):
            start = i + 1
            break
    if start is None:
        return ""

    end = len(lines)
    for j in range(start, len(lines)):
        stripped_j = lines[j].strip().rstrip(":")
        if _is_heading(lines[j]) and not any(re.search(rf"^{p}$", stripped_j, re.IGNORECASE) for p in start_names):
            if any(re.search(rf"^{p}$", stripped_j, re.IGNORECASE) for p in end_names):
                end = j
                break
    return "\n".join(lines[start:end]).strip()


def _extract_skills(skills_text: str) -> list:
    text = re.sub(r'^[A-Za-z ]+:\s*', '', skills_text, flags=re.MULTILINE)
    text = (
        text.replace('', '\n')
            .replace('•', '\n')
            .replace('●', '\n')
            .replace('▪', '\n')
    )
    SKILL_NORMALIZATION = {
        "python development": "Python",
        "aws services": "AWS",
        "devops practices": "DevOps",
        "security & compliance": "Security Compliance",
        "monitoring and logging": "Monitoring and Logging",
        "database integration": "Database Integration",
        "api integration": "API Integration",
        "serverless architecture": "Serverless Architecture",
        "aws deployment": "AWS Deployment",
        "version control": "Version Control"
    }

    items = re.split(r'[,\n;]+', text)
    skills = []

    NON_TECHNICAL = {
        "research",
        "professional teaching",
        "public speaking",
        "communication",
        "leadership",
        "team management",
        "problem solving",
    }
    reject_words = (
        "education", "experience", "work experience", "contact", "profile",
        "objective", "awards", "certificates", "hobbies", "strengths",
        "roles and responsibilities"
    )
    TECH_WORDS = {
        "architecture", "integration", "database", "serverless", "control",
        "security", "deployment", "monitoring", "logging", "api", "aws",
        "python", "devops", "framework", "testing", "automation", "cloud",
        "microservices", "docker", "kubernetes"
    }

    for idx, item in enumerate(items):
        s = item.strip(" -•●▪\t")
        if not s:
            continue
        low = s.lower()
        if low in NON_TECHNICAL:
            continue
        if low in SKILL_NORMALIZATION:
            s = SKILL_NORMALIZATION[low]
        if any(p in low for p in ("development program",)):
            continue
        if re.fullmatch(r"[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3}", s):
            words = {w.lower() for w in s.split()}
            if idx < 3 and not words.intersection(TECH_WORDS):
                continue
        if s.upper() in {"SKILLS", "TECHNICAL SKILLS", "EDUCATION", "EXPERIENCE", "WORK EXPERIENCE"}:
            continue
        s = re.sub(
            r'^(experience in|proficient in|working with)\s+',
            '',
            s,
            flags=re.IGNORECASE
        )
        if any(w in low for w in reject_words):
            continue
        if len(s.split()) > 5:
            continue
        if len(s) > 60:
            continue
        if s.endswith('.') or s.endswith(',') or s.endswith(':'):
            continue
        skills.append(s)

    return list(dict.fromkeys(skills))


def _extract_education(edu_text: str) -> list:
    entries = []
    lines = [l.strip() for l in edu_text.splitlines() if l.strip()]
    i = 0
    while i < len(lines):
        line = lines[i]

        year_matches = re.findall(r'\b((?:19|20)\d{2})\b', line)
        if year_matches:
            year = year_matches[-1]

            degree = None
            institution = None

            if re.search(r'[A-Za-z]', line.replace(year, '')):
                degree_raw = re.sub(
                    r'\s+(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|'
                    r'Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)?\s*(?:19|20)\d{2}.*$',
                    '',
                    line,
                    flags=re.IGNORECASE
                ).strip()
                degree = re.sub(r':\s*', ' in ', degree_raw).strip()

            if i > 0 and "|" in lines[i - 1]:
                prev = lines[i - 1]
                parts = [p.strip() for p in prev.split('|') if p.strip()]
                if len(parts) == 2:
                    institution = parts[0]
                    degree = parts[1]
                elif len(parts) == 1 and institution is None:
                    degree = parts[0]

            if institution is None and i + 1 < len(lines):
                next_line = lines[i + 1]
                if not re.search(r'\b((?:19|20)\d{2})\b', next_line):
                    institution = re.sub(r',\s*[A-Za-z\s]+$', '', next_line).strip()

            combined = f"{degree or ''} {institution or ''}".lower()

            if re.search(r'\b(technologies|solutions|private limited|pvt|ltd|inc|consultant|freelancer|hcl|mindtree)\b', combined):
                i += 1
                continue
            if "certification" in combined.lower():
                i += 1
                continue
            entries.append({
                "degree": degree or None,
                "institution": institution or None,
                "year": year
            })

        i += 1

    return entries


def _looks_like_education(company, role) -> bool:
    combined = f"{company or ''} {role or ''}".lower()

    keywords = (
        "university", "college", "school", "institute", "certification",
        "certificate", "course", "b.e", "b.tech", "m.e", "m.tech",
        "b.sc", "m.sc", "mba", "diploma", "bca", "mca"
    )

    score = sum(1 for k in keywords if k in combined)
    return score >= 2


def _extract_experience(exp_text: str) -> list:
    entries = []
    lines = [l.strip() for l in exp_text.splitlines() if l.strip()]

    date_pattern = re.compile(
        r'((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)'
        r'[a-z]*\.?[\s-]+\d{4}|\d{4})'
        r'\s*[-–]\s*'
        r'((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)'
        r'[a-z]*\.?[\s-]+\d{4}|\d{4}|Present|present|Current|current)',
        re.IGNORECASE
    )

    for i, line in enumerate(lines):
        if line.startswith(('-', '•', '●', '')):
            continue

        date_match = date_pattern.search(line)

        if not date_match and i + 1 < len(lines):
            next_line = lines[i + 1]
            next_date_match = date_pattern.search(next_line)

            if next_date_match and "|" in line:
                parts = [p.strip() for p in line.split("|") if p.strip()]

                company = parts[0] if len(parts) > 0 else None
                role = parts[1] if len(parts) > 1 else None

                entries.append({
                    "company": company,
                    "role": role,
                    "duration": next_date_match.group(0).strip()
                })
            continue
        multi_matches = re.findall(
            r'([A-Za-z &]+?)\s*\|\s*([A-Za-z ,/&-]+?)\s*\|\s*'
            r'((?:19|20)\d{2}.*?(?:Present|Current|(?:19|20)\d{2}))',
            line,
            re.IGNORECASE
        )
        if not date_match:
            continue

        duration = date_match.group(0).strip()
        prefix = line[:date_match.start()].strip().rstrip(",|-–").strip()

        company = prefix or None
        role = None

        if "|" in (prefix or ""):
            parts = [p.strip() for p in prefix.split("|") if p.strip()]
            if len(parts) >= 2:
                company, role = parts[0], parts[1]

        elif i > 0:
            prev = lines[i - 1].strip()
            if "|" in prev:
                parts = [p.strip() for p in prev.split("|") if p.strip()]
                if len(parts) >= 2:
                    company = parts[0]
                    role = parts[1]
            elif (
                prev
                and not date_pattern.search(prev)
                and not prev.startswith(('-', '•', '●', ''))
            ):
                role = prev

        company = re.sub(r'\(.*$', '', company or '').strip(" ,-–")

        if _looks_like_education(company, role):
            continue
        if len(multi_matches) > 1:
            for company, role, duration in multi_matches:
                entries.append({
                    "company": company.strip(),
                    "role": role.strip(),
                    "duration": duration.strip()
                })
            continue
        entries.append({
            "company": company or None,
            "role": role,
            "duration": duration
        })

    return _clean_experience_list(entries)


def _extract_experience_blocks(exp_text: str) -> list:
    """
    Block-based experience parser for multi-line resume formats where company,
    role, and dates appear on separate lines rather than pipe-separated on one line.
    Used as fallback when _extract_experience returns an empty list.
    """
    date_pat = re.compile(
        r'((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s*\d{4}|\d{4})'
        r'\s*[-–to]+\s*'
        r'((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s*\d{4}'
        r'|\d{4}|Present|present|Current|current|Till\s*Date|till\s*date)',
        re.IGNORECASE
    )
    blocks = re.split(r'\n{2,}', exp_text.strip())
    entries = []
    for block in blocks:
        lines = [l.strip() for l in block.splitlines() if l.strip()]
        if not lines:
            continue
        duration = None
        for line in lines:
            m = date_pat.search(line)
            if m:
                duration = m.group(0).strip()
                break
        if not duration:
            continue
        non_date_lines = [l for l in lines if not date_pat.search(l)
                          and not l.startswith(('-', '•', '●', '\uf0b7'))]
        company = non_date_lines[0] if non_date_lines else None
        role = non_date_lines[1] if len(non_date_lines) > 1 else None
        if company and _IT_ROLE_KEYWORDS.search(company) and role:
            company, role = role, company
        entries.append({"company": company, "role": role, "duration": duration})
    return _clean_experience_list(entries)


def preprocess_payslip_text(text: str, page: int = -1) -> str:
    text = text.replace('■', '').replace('▪', '')
    text = text.replace('\r\n', '\n').replace('\r', '\n')

    chunks = re.split(
        r'(?<=does not require a signature\.)\s*',
        text,
        flags=re.IGNORECASE
    )
    chunks = [c.strip() for c in chunks if c.strip() and len(c.strip()) > 100]

    if len(chunks) > 1:
        text = chunks[page]
    known_labels = [
        "Employee Name", "Employee ID", "Designation", "Department",
        "Pay Period", "Bank", "Account No", "IFSC Code",
        "PF Account No", "UAN No", "Universal Account Number",
        "Gross Pay", "Net Pay", "Basic Pay", "Basic Salary",
        "EPF Wages", "Date of Joining", "PAN", "Location"
    ]
    pattern = r'(?<!\n)(' + '|'.join(re.escape(l) for l in known_labels) + r')(?=\s)'
    text = re.sub(pattern, r'\n\1', text)
    return text


def _normalize_resume_result(result: dict) -> dict:
    for key in ("skills", "education", "experience"):
        if result.get(key) is None:
            result[key] = []

    if isinstance(result.get("education"), list):
        result["education"] = _normalize_education_list(result["education"])

    if isinstance(result.get("experience"), list):
        result["experience"] = _clean_experience_list(result["experience"])

    if isinstance(result.get("skills"), list):
        cleaned = []
        for s in result["skills"]:
            if not s:
                continue
            s = str(s).strip(" -•●\t")
            if not s:
                continue
            cleaned.append(s)
        result["skills"] = list(dict.fromkeys(cleaned))
    if result.get("phone") is not None:
        result["phone"] = str(result["phone"])
    return result


def _normalize_education_list(edu_list: list) -> list:
    normalized = []
    degree_keywords = (
        "b.e", "b.tech", "bsc", "b.sc", "bca", "m.e", "m.tech", "msc", "m.sc",
        "mba", "bachelor", "master", "diploma", "certificate", "certification",
        "phd", "doctorate"
    )
    company_keywords = (
        "technologies", "technology", "solutions", "private limited", "pvt", "ltd",
        "inc", "llc", "consultant", "freelancer", "global", "mindtree", "hcl"
    )

    for e in edu_list:
        if not isinstance(e, dict):
            continue

        degree = (e.get("degree") or "").strip()
        institution = (e.get("institution") or "").strip()
        year = e.get("year")

        combined = f"{degree} {institution}"

        if not year:
            years = re.findall(r'\b((?:19|20)\d{2})\b', combined)
            if years:
                year = max(years)

        if degree and institution:
            deg_low = degree.lower()
            inst_low = institution.lower()

            degree_looks_like_school = not any(k in deg_low for k in degree_keywords) and (
                "university" in deg_low or "college" in deg_low or "school" in deg_low or "institute" in deg_low
            )
            inst_looks_like_degree = any(k in inst_low for k in degree_keywords)

            if degree_looks_like_school and inst_looks_like_degree:
                degree, institution = institution, degree

        combined_low = combined.lower()
        if any(k in combined_low for k in company_keywords) and not any(k in combined_low for k in degree_keywords):
            continue

        if degree or institution:
            normalized.append({
                "degree": degree or None,
                "institution": institution or None,
                "year": year or None
            })

    return normalized


def _clean_experience_list(exp_list: list) -> list:
    cleaned = []

    for entry in exp_list:
        if not isinstance(entry, dict):
            continue

        def _to_str(val) -> str:
            if val is None:
                return ""
            if isinstance(val, list):
                return " ".join(str(x) for x in val if x).strip()
            return str(val).strip()

        company  = _to_str(entry.get("company"))
        role     = _to_str(entry.get("role"))
        duration = _to_str(entry.get("duration"))

        if role and role.upper().startswith("WORK EXPERIENCE"):
            role = ""

        if company == "(":
            company = ""

        if "|" in (company or ""):
            parts = [p.strip() for p in company.split("|") if p.strip()]
            if len(parts) >= 2:
                company = parts[0]
                role = role or parts[1]

        if company:
            date_match = re.search(
                r'((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s*[- ]?\s*(?:19|20)\d{2}|\b(?:19|20)\d{2}\b).*'
                r'((?:Present|Current|\b(?:19|20)\d{2}\b))?',
                company,
                re.IGNORECASE
            )
            if date_match:
                years = re.findall(r'\b((?:19|20)\d{2})\b', company)
                if years and not duration:
                    duration = " – ".join([years[0], years[-1]]) if len(years) > 1 else years[0]
                company = re.sub(r'\(.*?(?:19|20)\d{2}.*?\)', '', company).strip(" ,-–")

        if not (company or role or duration):
            continue

        cleaned.append({
            "company": company or None,
            "role": role or None,
            "duration": duration or None
        })

    deduped = []
    seen = set()
    for e in cleaned:
        key = (
            (e.get("company") or "").lower(),
            (e.get("role") or "").lower(),
            (e.get("duration") or "").lower(),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(e)

    return deduped


# -----------------------------------------------------------------------
# Main LLM call (used only for resume fields LLM needs to fill in)
# -----------------------------------------------------------------------
def _build_resume_prompt_text(raw_text: str, max_chars: int = 5000) -> str:
    """
    Build LLM prompt text for resumes by combining the header (name/contact)
    with the experience section, instead of blindly truncating from the start.
    Experience sections in long resumes are often cut off by a naive [:5000] slice.
    """
    header = raw_text[:1500]
    exp_section = _slice_between_headings(
        raw_text,
        start_names=(r"EXPERIENCE", r"WORK EXPERIENCE", r"PROFESSIONAL EXPERIENCE",
                     r"EMPLOYMENT HISTORY", r"EMPLOYMENT", r"CAREER HISTORY",
                     r"ROLES AND RESPONSIBILITIES"),
        end_names=(r"EDUCATION", r"SKILLS", r"SUMMARY OF SKILLS", r"AWARDS",
                   r"PROJECTS", r"CERTIFICATIONS", r"HOBBIES")
    )
    remainder_budget = max_chars - len(header)
    return header + "\n\n" + exp_section[:remainder_budget]


def _call_llm(raw_text: str, doc_type: str, schema: dict, strict: bool = False) -> dict:
    char_limit = CHAR_LIMITS.get(doc_type, 3000)
    if doc_type == "resume":
        text_chunk = _build_resume_prompt_text(raw_text, max_chars=char_limit)
    else:
        text_chunk = raw_text[:char_limit]

    if strict:
        prompt = f"""Extract fields from this {doc_type}.
Return ONLY a JSON object. Start your response with {{ and end with }}.
No explanation, no markdown, no extra text.

Fields: {json.dumps(schema)}

TEXT:
{text_chunk}"""
    else:
        prompt = f"""You are a document extraction system.

Extract fields from the following {doc_type} document.
Return ONLY a valid JSON object with these fields:
{json.dumps(schema, indent=2)}

Rules:
- Use null for any field not found
- Do not include any text outside the JSON object
- Numbers should be actual numbers, not strings

DOCUMENT TEXT:
{text_chunk}"""

    try:
        response = requests.post(
            OLLAMA_URL,
            json={"model": MODEL, "prompt": prompt, "stream": False},
            timeout=60
        )
        response.raise_for_status()
        raw_response = response.json()["response"]
        return _parse_response(raw_response)

    except requests.exceptions.Timeout:
        return {"error": "llm_timeout"}
    except requests.exceptions.ConnectionError:
        return {"error": "ollama_not_running"}
    except Exception as e:
        return {"error": f"llm_error: {str(e)}"}


def _call_llm_for_edu_exp(raw_text: str) -> dict:
    mini_schema = {
        "education": SCHEMAS["resume"]["education"],
        "experience": SCHEMAS["resume"]["experience"]
    }
    char_limit = CHAR_LIMITS.get("resume", 5000)
    text_chunk = raw_text[:char_limit]

    prompt = f"""You are a resume extraction system.
Extract ONLY education and experience from this resume text.

Return ONLY a valid JSON object with exactly these fields:
{json.dumps(mini_schema, indent=2)}

Rules:
- Use [] if not found
- Do not include any fields other than education and experience
- No explanation, no markdown, only JSON

RESUME TEXT:
{text_chunk}"""

    try:
        response = requests.post(
            OLLAMA_URL,
            json={"model": MODEL, "prompt": prompt, "stream": False},
            timeout=60
        )
        response.raise_for_status()
        raw_response = response.json()["response"]
        parsed = _parse_response(raw_response)

        if not isinstance(parsed, dict):
            return {"education": [], "experience": []}

        if parsed.get("education") is None:
            parsed["education"] = []
        if parsed.get("experience") is None:
            parsed["experience"] = []

        parsed["education"] = _normalize_education_list(parsed["education"])
        parsed["experience"] = _clean_experience_list(parsed["experience"])
        return parsed

    except Exception:
        return {"education": [], "experience": []}


def _parse_response(raw: str) -> dict:
    cleaned_raw = re.sub(r'(\d),(\d{3})\b', r'\1\2', raw)
    try:
        return json.loads(cleaned_raw.strip())
    except json.JSONDecodeError:
        pass

    match = re.search(r'\{.*\}', cleaned_raw, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    cleaned = re.sub(r"```(?:json)?", "", cleaned_raw).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    return {"error": "parse_failed", "raw": raw}


# -----------------------------------------------------------------------
# Main entry point
# -----------------------------------------------------------------------
def extract_with_llm(raw_text: str, doc_type: str, **kwargs) -> dict:
    schema = SCHEMAS.get(doc_type)
    if not schema:
        return {
            "doc_type": doc_type,
            "error": f"No schema defined for doc_type: {doc_type}"
        }

    # --------------------------------------------------
    # PAYSLIP: regex-first, LLM only as fallback
    # --------------------------------------------------
    if doc_type == "payslip":
        page = kwargs.get("page", -1)
        preprocessed = preprocess_payslip_text(raw_text, page=page)

        uan = extract_uan_from_text(preprocessed)
        if not uan:
            uan = extract_uan_from_text(raw_text)  # try unprocessed too
        if not uan:
            uan = extract_uan_with_llm(preprocessed)  # LLM fallback

        return {
            "uan": uan,
            "doc_type": "payslip",
        }

    # --------------------------------------------------
    # RESUME: regex extraction first
    # --------------------------------------------------
    if doc_type == "resume":
        parsed = extract_resume_sections(raw_text)

        complete_regex = (
            parsed.get("name")
            and parsed.get("email")
            and parsed.get("phone")
            and parsed.get("skills")
            and parsed.get("education")
            and parsed.get("experience")
        )
        if complete_regex:
            parsed["doc_type"] = "resume"
            # Ensure IT experience is always calculated from structured data, never skipped
            if parsed.get("total_experience_years") is None and parsed.get("experience"):
                exp_text_c = _slice_between_headings(
                    raw_text,
                    start_names=(r"EXPERIENCE", r"WORK EXPERIENCE", r"PROFESSIONAL EXPERIENCE",
                                 r"EMPLOYMENT HISTORY", r"EMPLOYMENT", r"CAREER HISTORY",
                                 r"ROLES AND RESPONSIBILITIES", r"Professional Experience"),
                    end_names=(r"SKILLS", r"SUMMARY OF SKILLS", r"EDUCATION", r"AWARDS",
                               r"CERTIFICATES", r"HOBBIES")
                )
                it_c = _calculate_it_experience_from_entries(parsed["experience"])
                if it_c is None and parsed["experience"]:
                    it_c = _calculate_it_experience_with_company_fallback(parsed["experience"])
                if it_c is None and exp_text_c:
                    it_c = _calculate_it_experience_with_llm(exp_text_c)
                parsed["total_experience_years"] = it_c
            result_c = _normalize_resume_result(parsed)
            return {
                "name": result_c.get("name"),
                "email": result_c.get("email"),
                "phone": result_c.get("phone"),
                "total_experience_years": result_c.get("total_experience_years"),
                "doc_type": "resume",
            }

        # LLM to fill gaps in non-complete resume extraction
        result = _call_llm(raw_text, doc_type, schema)

        if isinstance(result, dict) and result.get("error") == "parse_failed":
            result = _call_llm(raw_text, doc_type, schema, strict=True)

        if not isinstance(result, dict):
            return {"doc_type": doc_type, "error": "llm_returned_non_dict"}

        result["doc_type"] = doc_type

        # Merge reliable regex fields over LLM fields
        if parsed:
            if parsed.get("education"):
                result["education"] = parsed["education"]
            if parsed.get("experience"):
                result["experience"] = parsed["experience"]
            if not result.get("name"):
                result["name"] = parsed.get("name")
            if not result.get("email"):
                result["email"] = parsed.get("email")
            if not result.get("phone"):
                result["phone"] = parsed.get("phone")
            if not result.get("skills"):
                result["skills"] = parsed.get("skills", [])

        # FIXED: recalculate IT experience on final merged experience list
        exp_text = _slice_between_headings(
            raw_text,
            start_names=(r"EXPERIENCE", r"WORK EXPERIENCE", r"PROFESSIONAL EXPERIENCE",
                         r"EMPLOYMENT HISTORY", r"EMPLOYMENT", r"CAREER HISTORY",
                         r"ROLES AND RESPONSIBILITIES", r"Professional Experience"),
            end_names=(r"SKILLS", r"SUMMARY OF SKILLS", r"EDUCATION", r"AWARDS",
                       r"CERTIFICATES", r"HOBBIES")
        )
        # Always prefer regex-parsed experience for IT year calculation
        exp_entries = result.get("experience") or []
        it_years = _calculate_it_experience_from_entries(exp_entries)
        if it_years is None and exp_entries:
            it_years = _calculate_it_experience_with_company_fallback(exp_entries)
        if it_years is None and exp_text:
            it_years = _calculate_it_experience_with_llm(exp_text)
        result["total_experience_years"] = it_years

        result = _normalize_resume_result(result)
        if result.get("phone") is not None:
            result["phone"] = str(result["phone"])

        # Recovery pass for long resumes with missing edu/exp
        edu_empty = not result.get("education")
        exp_empty = not result.get("experience")

        if edu_empty or exp_empty:
            char_limit = CHAR_LIMITS.get("resume", 5000)
            tail_candidates = [
                raw_text[char_limit:],
                raw_text[-4000:] if len(raw_text) > 4000 else ""
            ]

            for tail_text in tail_candidates:
                if not tail_text.strip():
                    continue

                tail_res = _call_llm_for_edu_exp(tail_text)

                if edu_empty and tail_res.get("education"):
                    result["education"] = tail_res["education"]
                    edu_empty = False

                if exp_empty and tail_res.get("experience"):
                    result["experience"] = tail_res["experience"]
                    exp_empty = False

                if not edu_empty and not exp_empty:
                    break

        result = _normalize_resume_result(result)

        # Keep only the fields callers need
        return {
            "name": result.get("name"),
            "email": result.get("email"),
            "phone": result.get("phone"),
            "total_experience_years": result.get("total_experience_years"),
            "doc_type": "resume",
        }

    # --------------------------------------------------
    # Other doc types (experience_letter, degree_certificate)
    # --------------------------------------------------
    result = _call_llm(raw_text, doc_type, schema)

    if isinstance(result, dict) and result.get("error") == "parse_failed":
        result = _call_llm(raw_text, doc_type, schema, strict=True)

    if not isinstance(result, dict):
        return {"doc_type": doc_type, "error": "llm_returned_non_dict"}

    result["doc_type"] = doc_type
    return result