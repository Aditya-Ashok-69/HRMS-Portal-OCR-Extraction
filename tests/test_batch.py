import subprocess, json, sys
from pathlib import Path

# Define your test cases: (doc_type, file_or_front, back_optional, expected_fields)
TEST_CASES = [
    {
        "label": "Aadhaar front image - clear",
        "args": ["--doc-type", "aadhaar_image", "--file", "samples/aadhaar_front_clear.jpg"],
        "expect": {"name": True, "dob": True, "aadhaar": True}
    },
    {
        "label": "Aadhaar front image - blurry",
        "args": ["--doc-type", "aadhaar_image", "--file", "samples/aadhaar_front_blurry.jpg"],
        "expect": {"aadhaar": True}  # name/dob might fail on bad image, aadhaar must not
    },
    {
        "label": "Aadhaar front+back",
        "args": ["--doc-type", "aadhaar_front_back", "--front", "samples/front.jpg", "--back", "samples/back.jpg"],
        "expect": {"name": True, "address": True, "aadhaar": True}
    },
    {
        "label": "PAN image",
        "args": ["--doc-type", "pan_image", "--file", "samples/pan.jpg"],
        "expect": {"name": True, "dob": True, "pan": True}
    },
    {
        "label": "e-Aadhaar PDF",
        "args": ["--doc-type", "aadhaar_pdf", "--file", "samples/eaadhaar.pdf"],
        "expect": {"name": True, "dob": True, "aadhaar": True, "address": True}
    },
]

passed = 0
failed = 0

for case in TEST_CASES:
    result = subprocess.run(
        [sys.executable, "extract_id.py"] + case["args"],
        capture_output=True, text=True
    )
    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError:
        print(f"FAIL [{case['label']}] — script crashed or no JSON output")
        print("  STDERR:", result.stderr[:200])
        failed += 1
        continue

    missing = [k for k, required in case["expect"].items() if required and not data.get(k)]
    low_conf = data.get("_meta", {}).get("low_confidence_fields", [])

    if missing:
        print(f"FAIL [{case['label']}] — missing fields: {missing}")
        failed += 1
    else:
        flag = f" ⚠ low_confidence: {low_conf}" if low_conf else ""
        print(f"PASS [{case['label']}]{flag}")
        passed += 1

print(f"\n{passed} passed, {failed} failed out of {len(TEST_CASES)} tests")