# extractors/aadhaar.py
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))  # root on path

from extract_id import (
    extract_from_aadhaar_pdf,
    extract_from_aadhaar_front_image,
    extract_from_aadhaar_back_image,
    merge_aadhaar_results,
)


def extract_aadhaar(file_path: str, sub_type: str = "image") -> dict:
    """
    sub_type options:
      "image"       → front image (most common single upload)
      "pdf"         → aadhaar PDF
      "front"       → explicit front image
      "back"        → back image only
      "front_back"  → expects dict {"front": path, "back": path}
    """
    if sub_type == "pdf":
        return extract_from_aadhaar_pdf(file_path)

    elif sub_type in ("image", "front"):
        return extract_from_aadhaar_front_image(file_path)

    elif sub_type == "back":
        return extract_from_aadhaar_back_image(file_path)

    elif sub_type == "front_back":
        # file_path is expected to be a dict here: {"front": ..., "back": ...}
        paths = file_path if isinstance(file_path, dict) else {}
        front_result = extract_from_aadhaar_front_image(paths["front"]) if paths.get("front") else None
        back_result  = extract_from_aadhaar_back_image(paths["back"])  if paths.get("back")  else None
        return merge_aadhaar_results(front_result, back_result)

    else:
        return {"error": f"Unknown aadhaar sub_type: {sub_type}"}