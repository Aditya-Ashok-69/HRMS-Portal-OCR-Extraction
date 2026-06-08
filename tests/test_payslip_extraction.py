from llm_utils import extract_payslip_fields

sample_payslip = """
ABC Technologies Pvt Ltd

Employee Name: Aditya Ashok
Employee ID: EMP12345

Pay Period: June 2026

Gross Salary: 75000
Net Salary: 68000

PAN: ABCDE1234F
UAN: 123456789012
"""

result = extract_payslip_fields(sample_payslip)

print(result)