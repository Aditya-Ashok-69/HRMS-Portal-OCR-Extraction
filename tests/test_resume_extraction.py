from llm_utils import extract_resume_fields

sample_resume = """
ADITYA ASHOK

Email: aditya@gmail.com
Phone: 9876543210

Education:
BE Civil Engineering

Skills:
Python
SQL
Machine Learning
"""

result = extract_resume_fields(sample_resume)

print(result)