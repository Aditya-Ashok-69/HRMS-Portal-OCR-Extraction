from llm_utils import classify_document

sample_text = """
CURRICULUM VITAE

Name: Aditya
Skills: Python
Education: BE Civil Engineering
"""

print(classify_document(sample_text))

sample_text = """
Government of India

AADHAAR

Name: Aditya
DOB: 10/10/2004
Male
1234 5678 9123
"""

print(classify_document(sample_text))
