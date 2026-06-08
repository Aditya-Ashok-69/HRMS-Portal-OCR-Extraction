# debug_resume.py (temporary)
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from document_processor import _get_text
from extractors.llm_extractor import extract_resume_sections

file_path = "samples/resume.pdf"


raw_text = _get_text(file_path)
print("FILE:", file_path)
print("=== RAW TEXT (first 4000) ===")
print(raw_text[:4000])

parsed = extract_resume_sections(raw_text)
print("=== PARSED (regex) ===", parsed)