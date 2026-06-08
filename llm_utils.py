import requests
import json


def call_phi(prompt, model="phi4-mini"):

    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "stream": False
        },
        timeout=120
    )

    response.raise_for_status()

    return response.json()["response"]

def classify_document(raw_text):

    prompt = f"""
You are a document classifier.

Possible document types:

aadhaar_card
pan_card
resume_cv
payslip
other

Return ONLY valid JSON.

Example:
{{"doc_type":"resume_cv"}}

Do not explain.
Do not add markdown.
Do not add extra text.

OCR Text:

{raw_text[:4000]}
"""

    result = call_phi(prompt)

    start = result.find("{")
    end = result.rfind("}") + 1

    data = json.loads(result[start:end])

    if "doc_type" in data:
        data["doc_type"] = data["doc_type"].lower()

    return data

def extract_resume_fields(raw_text):

    prompt = f"""
Extract information from this resume.

Return ONLY valid JSON.

Schema:

{{
  "doc_type":"resume_cv",
  "name": null,
  "email": null,
  "phone": null,
  "skills": [],
  "education": []
}}

Use null if missing.

OCR Text:

{raw_text[:6000]}
"""

    result = call_phi(prompt)

    start = result.find("{")
    end = result.rfind("}") + 1

    return json.loads(result[start:end])

import json

def extract_resume_fields(raw_text: str) -> dict:

    prompt = f"""
You are a resume parser.

Extract information from the resume.

Return ONLY valid JSON.

Schema:

{{
    "doc_type": "resume_cv",
    "name": null,
    "email": null,
    "phone": null,
    "skills": [],
    "education": []
}}

Rules:
- Return ONLY JSON.
- No explanations.
- No markdown.
- No extra text.
- Use null if missing.
- skills must be an array.
- education must be an array.

Resume Text:

{raw_text[:6000]}
"""

    result = call_phi(prompt)

    try:
        start = result.find("{")
        end = result.rfind("}") + 1

        if start == -1 or end == 0:
            raise ValueError("No JSON found")

        data = json.loads(result[start:end])

        return data

    except Exception as e:
        return {
            "doc_type": "resume_cv",
            "error": str(e),
            "raw_response": result
        }
    
def extract_payslip_fields(raw_text: str) -> dict:

    prompt = f"""
You are a payslip parser.

Extract information from the payslip.

Return ONLY valid JSON.

Schema:

{{
    "doc_type": "payslip",
    "employee_name": null,
    "employee_id": null,
    "employer_name": null,
    "pay_period": null,
    "gross_pay": null,
    "net_pay": null,
    "pan": null,
    "uan": null
}}

Rules:
- Return ONLY JSON.
- No explanations.
- No markdown.
- No extra text.
- Use null if missing.

Payslip Text:

{raw_text[:6000]}
"""

    result = call_phi(prompt)

    try:
        start = result.find("{")
        end = result.rfind("}") + 1

        if start == -1 or end == 0:
            raise ValueError("No JSON found")

        data = json.loads(result[start:end])

        return data

    except Exception as e:
        return {
            "doc_type": "payslip",
            "error": str(e),
            "raw_response": result
        }