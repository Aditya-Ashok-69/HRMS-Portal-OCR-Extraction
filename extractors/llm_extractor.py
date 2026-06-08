# extractors/llm_extractor.py
import requests
import json
import re


OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "phi4-mini"


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
        "total_experience_years": "number or null"
    },
    "payslip": {
        "employee_name": "string - name of the employee",
        "employee_id": "string - employee ID or staff number",
        "employer_name": "string - company/employer name. Usually appears at the very top of the payslip. Examples: Infosys Ltd, TCS, Cognizant Technology Solutions, HCL Technologies. Do NOT return employee name.",
        "pay_period": "string - extract FULL pay period including month and year exactly as shown. Examples: 'March 2025', 'Apr 2024', '01-Mar-2025 to 31-Mar-2025'. Never return month only.",
        "gross_pay": "number - total earnings before deductions",
        "net_pay": "number - take-home pay after deductions",
        "basic_pay": "number - basic salary component",
        "pan": "string or null",
        "uan": "string - Universal Account Number"
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


DEGREE_KEYWORDS = (
    "b.e", "be ", "b.tech", "btech", "m.e", "me ", "m.tech", "mtech",
    "b.sc", "bsc", "m.sc", "msc", "mba", "bca", "mca", "phd",
    "bachelor", "master", "diploma", "degree", "nursing", "psychology",
    "computer science", "engineering"
)

EDU_REJECT_WORDS = (
    "experience", "work experience", "professional experience", "project",
    "projects", "publication", "publications", "journal", "conference",
    "paper presentation", "responsibilities", "declaration", "personal profile",
    "dob", "date of birth", "nationality", "language", "languages", "hobbies",
    "skills", "technical skills", "certification", "certifications"
)

EXP_REJECT_WORDS = (
    "education", "languages", "hobbies", "certifications", "declaration",
    "journal", "publication", "publications", "dissertation", "paper presentation",
    "project completed", "participated in various events", "personal profile"
)

ROLE_WORDS = (
    "engineer", "developer", "analyst", "consultant", "intern", "associate",
    "manager", "lead", "professor", "instructor", "designer", "tester",
    "administrator", "recruiter", "executive", "programmer"
)

COMPANY_HINTS = (
    "ltd", "limited", "pvt", "private", "solutions", "technologies",
    "systems", "services", "infotech", "college", "university", "hospital",
    "company", "corp", "inc", "llp"
)


def preprocess_resume_text(text: str) -> str:
    """
    Normalize resume text before sending to LLM:
    - Replace bullet characters with newline + dash
    - Normalize whitespace
    - Keep structure readable
    """
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
        end_names=(r"CONTACT", r"EDUCATION", r"EXPERIENCE", r"WORK EXPERIENCE", r"OBJECTIVE", r"ROLES AND RESPONSIBILITIES", r"AWARDS", r"CERTIFICATES", r"HOBBIES", r"INTEREST AND STRENGTHS", r"TECH STACK", r"CERTIFICATIONS")
    )
    result["skills"] = _extract_skills(skills_text) if skills_text else []

    edu_text = sections.get("EDUCATION") or ""
    edu_list = _extract_education(edu_text) if edu_text else []

    exp_text = _slice_between_headings(
        text,
        start_names=(r"EXPERIENCE", r"WORK EXPERIENCE", r"PROFESSIONAL EXPERIENCE", r"ROLES AND RESPONSIBILITIES"),
        end_names=(r"SKILLS", r"SUMMARY OF SKILLS", r"EDUCATION", r"AWARDS", r"CERTIFICATES", r"HOBBIES")
    )
    exp_list = _extract_experience(exp_text) if exp_text else []

    if not edu_list:
        edu_list = _extract_education(text)

    if not exp_list:
        exp_list = _extract_experience(text)

    result["education"] = edu_list or []
    result["experience"] = exp_list or []
    exp_match = re.search(
        r'(\d+)\+?\s*Years',
        text,
        re.IGNORECASE
    )
    result["total_experience_years"] = (
        int(exp_match.group(1))
        if exp_match
        else None
    )
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

    phone_match = re.search(r'(?:\+91[\s-]?)?[6-9]\d{9}', text)
    phone = phone_match.group(0) if phone_match else None

    lines = [l.strip() for l in text.splitlines() if l.strip()]

    heading_words = {
        "PROFILE", "CONTACT", "SUMMARY", "SUMMARY OF SKILLS",
        "SKILLS", "TECHNICAL SKILLS", "STRENGTH", "HOBBIES",
        "EDUCATION", "EXPERIENCE", "WORK EXPERIENCE"
    }

    def is_heading(line: str) -> bool:
        upper = line.upper()
        return upper in heading_words or upper.rstrip(':') in heading_words

    def is_strong_name(line: str) -> bool:
        if any(ch.isdigit() for ch in line):
            return False
        low = line.lower()
        if any(x in low for x in (
            "@", "http", "www.", "linkedin",
            "developer", "engineer", "analyst",
            "consultant", "manager", "lead"
        )):
            return False
        words = [w for w in line.split() if w]
        if not (2 <= len(words) <= 4):
            return False
        return True

    name = None

    header_lines = []
    for line in lines:
        if is_heading(line):
            break
        header_lines.append(line)
    if header_lines:
        candidate = header_lines[0].strip()
        if is_strong_name(candidate):
            name = candidate

    if not name:
        for line in lines:
            low = line.lower()
            if any(x in low for x in ['@', 'http', 'www.', 'linkedin']):
                continue
            if re.search(r'(?:\+?\d[\d\s-]{6,})', line):
                continue
            if len(re.findall(r'[A-Za-z]', line)) >= 4 and len(line) <= 70:
                name = line
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
        if any(re.fullmatch(p, line.strip(), re.IGNORECASE) for p in start_names):
            start = i + 1
            break
    if start is None:
        return ""

    end = len(lines)
    for j in range(start, len(lines)):
        if _is_heading(lines[j]) and not any(re.fullmatch(p, lines[j].strip(), re.IGNORECASE) for p in start_names):
            if any(re.fullmatch(p, lines[j].strip(), re.IGNORECASE) for p in end_names):
                end = j
                break
    return "\n".join(lines[start:end]).strip()


def _extract_skills(skills_text: str) -> list:
    text = re.sub(r'^[A-Za-z ]+:\s*', '', skills_text, flags=re.MULTILINE)
    text = (
        text.replace('', '\n')
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

    stop_phrases = (
        "ability to",
        "experience in",
        "experience leading",
        "good exposure",
        "excellent",
        "familiarity",
        "working with",
        "deep understanding",
        "proficient in",
        "developing",
        "having professional"
    )
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
        "architecture",
        "integration",
        "database",
        "serverless",
        "control",
        "security",
        "deployment",
        "monitoring",
        "logging",
        "api",
        "aws",
        "python",
        "devops",
        "framework",
        "testing",
        "automation",
        "cloud",
        "microservices",
        "docker",
        "kubernetes"
    }

    for idx, item in enumerate(items):
        s = item.strip(" -•●▪\t")
        if not s:
            continue
        low = s.lower()
        if low in NON_TECHNICAL:
            continue
        if any(p in low for p in stop_phrases):
            continue
        if low in SKILL_NORMALIZATION:
            s = SKILL_NORMALIZATION[low]
        reject_phrases = (
            "development program",
        )
        if any(p in low for p in reject_phrases):
            continue
        if re.fullmatch(r"[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3}", s):
            words = {w.lower() for w in s.split()}
            if idx < 3 and not words.intersection(TECH_WORDS):
                continue
        if s.upper() in {
            "SKILLS",
            "TECHNICAL SKILLS",
            "EDUCATION",
            "EXPERIENCE",
            "WORK EXPERIENCE"
        }:
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


def _looks_like_education_line(line: str) -> bool:
    low = line.lower().strip()
    if not low:
        return False
    if any(x in low for x in EDU_REJECT_WORDS):
        return False
    return any(k in low for k in DEGREE_KEYWORDS)


def _extract_education(edu_text: str) -> list:
    entries = []
    lines = [l.strip() for l in edu_text.splitlines() if l.strip()]
    i = 0
    while i < len(lines):
        line = lines[i]

        year_matches = re.findall(r'\b((?:19|20)\d{2})\b', line)
        if year_matches and _looks_like_education_line(line):
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
                    if not any(x in next_line.lower() for x in EDU_REJECT_WORDS):
                        institution = re.sub(r',\s*[A-Za-z\s]+$', '', next_line).strip()

            combined = f"{degree or ''} {institution or ''}".lower()

            if any(x in combined for x in EDU_REJECT_WORDS):
                i += 1
                continue
            if re.search(r'\b(technologies|solutions|private limited|pvt|ltd|inc|consultant|freelancer|hcl|mindtree)\b', combined):
                i += 1
                continue
            if "certification" in combined:
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
        "university",
        "college",
        "school",
        "institute",
        "certification",
        "certificate",
        "course",
        "b.e",
        "b.tech",
        "m.e",
        "m.tech",
        "b.sc",
        "m.sc",
        "mba",
        "diploma",
        "bca",
        "mca"
    )

    score = sum(1 for k in keywords if k in combined)
    return score >= 2


def _looks_like_company_or_role(text: str) -> bool:
    low = (text or "").lower().strip()
    if not low:
        return False
    return any(w in low for w in ROLE_WORDS) or any(h in low for h in COMPANY_HINTS)


def _looks_like_sentence_fragment(text: str) -> bool:
    s = (text or "").strip()
    if not s:
        return False
    low = s.lower()
    if len(s.split()) > 10:
        return True
    if any(low.startswith(x) for x in (
        "worked on", "involved in", "responsible for", "participated in",
        "effectiveness of", "designed and", "developed and", "collaborated with",
        "identified and", "conducted", "supported", "prepared"
    )):
        return True
    return False


def _extract_experience(exp_text: str) -> list:
    entries = []
    lines = [l.strip() for l in exp_text.splitlines() if l.strip()]

    date_pattern = re.compile(
        r'((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{4}|\d{4})'
        r'\s*[-–]\s*'
        r'((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{4}|\d{4}|Present|present|Current|current)',
        re.IGNORECASE
    )

    for i, line in enumerate(lines):
        if line.startswith(('-', '•', '●', '')):
            continue
        if any(x in line.lower() for x in EXP_REJECT_WORDS):
            continue

        date_match = date_pattern.search(line)
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
                and not prev.startswith(('-', '•', '●', ''))
                and not any(x in prev.lower() for x in EXP_REJECT_WORDS)
            ):
                role = prev

        company = re.sub(r'\(.*$', '', company or '').strip(" ,-–")

        if _looks_like_education(company, role):
            continue

        if len(multi_matches) > 1:
            for company, role, duration in multi_matches:
                if not (_looks_like_company_or_role(company) or _looks_like_company_or_role(role)):
                    continue
                entries.append({
                    "company": company.strip(),
                    "role": role.strip(),
                    "duration": duration.strip()
                })
            continue

        if not (_looks_like_company_or_role(company) or _looks_like_company_or_role(role)):
            continue

        entries.append({
            "company": company or None,
            "role": role,
            "duration": duration
        })

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
        "b.e", "be ", "b.tech", "btech", "m.e", "me ", "m.tech", "mtech",
        "b.sc", "bsc", "m.sc", "msc", "mba", "bca", "mca", "phd",
        "bachelor", "master", "diploma", "degree", "nursing", "psychology",
        "computer science", "engineering"
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

        company = (entry.get("company") or "").strip()
        role = (entry.get("role") or "").strip()
        duration = (entry.get("duration") or "").strip()

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

        combined_low = f"{company} {role}".lower()

        if any(x in combined_low for x in EXP_REJECT_WORDS):
            continue

        if _looks_like_sentence_fragment(company) and not _looks_like_company_or_role(company):
            continue

        if _looks_like_sentence_fragment(role) and not _looks_like_company_or_role(role):
            continue

        if company and role and company.lower() == role.lower():
            role = None

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


def extract_with_llm(raw_text: str, doc_type: str, **kwargs) -> dict:
    schema = SCHEMAS.get(doc_type)
    if not schema:
        return {
            "doc_type": doc_type,
            "error": f"No schema defined for doc_type: {doc_type}"
        }

    parsed = None

    # --------------------------------------------------
    # Resume: try regex extraction first
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
            return _normalize_resume_result(parsed)

    # --------------------------------------------------
    # Payslip preprocessing
    # --------------------------------------------------
    if doc_type == "payslip":
        page = kwargs.get("page", -1)
        raw_text = preprocess_payslip_text(raw_text, page=page)

    # --------------------------------------------------
    # Main LLM extraction
    # --------------------------------------------------
    result = _call_llm(raw_text, doc_type, schema)

    if isinstance(result, dict) and result.get("error") == "parse_failed":
        result = _call_llm(raw_text, doc_type, schema, strict=True)

    if not isinstance(result, dict):
        return {
            "doc_type": doc_type,
            "error": "llm_returned_non_dict"
        }

    result["doc_type"] = doc_type
    if doc_type == "payslip":

        if not result.get("pay_period"):
            m = re.search(
                r'(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|'
                r'Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|'
                r'Nov(?:ember)?|Dec(?:ember)?)\s+((?:19|20)\d{2})',
                raw_text,
                re.IGNORECASE
            )
            if m:
                result["pay_period"] = f"{m.group(1)} {m.group(2)}"

        if not result.get("employer_name"):
            lines = [
                l.strip()
                for l in raw_text.splitlines()
                if l.strip()
            ]

            for line in lines[:15]:
                if len(line) < 5:
                    continue

                if any(x in line.lower() for x in [
                    "employee",
                    "salary",
                    "payslip",
                    "pay slip",
                    "gross pay",
                    "net pay",
                    "basic pay",
                    "uan",
                    "pan"
                ]):
                    continue

                if re.search(r'[A-Za-z]', line):
                    result["employer_name"] = line
                    break

    # --------------------------------------------------
    # Resume-specific post processing
    # --------------------------------------------------
    if doc_type == "resume":

        # Merge reliable regex header fields
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

        result = _normalize_resume_result(result)
        if result.get("phone") is not None:
            result["phone"] = str(result["phone"])

        edu_empty = not result.get("education")
        exp_empty = not result.get("experience")

        # --------------------------------------------------
        # Recovery pass for long resumes
        # --------------------------------------------------
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

    return result


CHAR_LIMITS = {
    "resume": 5000,
    "payslip": 3000,
    "experience_letter": 2000,
    "degree_certificate": 2000,
}


def _call_llm(raw_text: str, doc_type: str, schema: dict, strict: bool = False) -> dict:
    char_limit = CHAR_LIMITS.get(doc_type, 3000)
    text_chunk = raw_text[:char_limit]

    extra_rules = ""

    if doc_type == "payslip":
        extra_rules = """
IMPORTANT:
- employer_name is the company issuing the payslip.
- employee_name is the employee receiving the payslip.
- pay_period must include BOTH month and year.
- If the document contains 'March 2025', return exactly 'March 2025'.
- Do NOT return only 'March'.
- employer_name is usually displayed at the top of the payslip.
- Do NOT confuse employer_name with employee_name.
"""

    if strict:
        prompt = f"""Extract fields from this {doc_type}.
Return ONLY a JSON object. Start your response with {{ and end with }}.
No explanation, no markdown, no extra text.

{extra_rules}

Fields: {json.dumps(schema)}

TEXT:
{text_chunk}"""
    else:
        prompt = f"""You are a document extraction system.

{extra_rules}

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