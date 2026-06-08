from document_processor import process_document
import json

results = {
    "aadhaar": process_document("samples/aadhaar_front.jpg", doc_type="aadhaar", sub_type="front"),
    "pan":     process_document("samples/pan.jpg", doc_type="pan"),
    "resume":  process_document("samples/resume.pdf", doc_type="resume"),
    "payslip": process_document("samples/payslip.pdf", doc_type="payslip", page=-1),
}

for doc, result in results.items():
    print(f"{doc.upper()}:", json.dumps(result, indent=2, ensure_ascii=False))