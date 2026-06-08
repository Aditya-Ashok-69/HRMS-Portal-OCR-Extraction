# save_output.py
import json
from pathlib import Path
from document_processor import process_document


def extract_and_save(file_path: str, doc_type: str, output_path: str = None, **kwargs):
    result = process_document(file_path, doc_type=doc_type, **kwargs)

    if output_path is None:
        stem = Path(file_path).stem
        output_path = f"output_{stem}_{doc_type}.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"Saved → {output_path}")
    return result


# Run directly: python save_output.py
if __name__ == "__main__":
    examples = [
        ("samples/r2.pdf", "resume", {}),
        ("samples/r1.docx", "resume", {}),
    ]

    for file_path, doc_type, kwargs in examples:
        
        if Path(file_path).exists():
            extract_and_save(file_path, doc_type, **kwargs)
        else:
            print(f"Skipped (not found): {file_path}")