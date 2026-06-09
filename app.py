import streamlit as st
import tempfile
import os
import json

from document_processor import process_document
from extract_id import (
    extract_from_aadhaar_front_image,
    extract_from_aadhaar_back_image,
    extract_from_aadhaar_image,
    merge_aadhaar_results,
)

st.set_page_config(
    page_title="HRMS Document Auto Fill",
    page_icon="📄",
    layout="wide",
)

# --------------------------------------------------
# CONFIG
# --------------------------------------------------

DOCUMENT_TYPES = {
    "Aadhaar": {
        "value": "aadhaar",
        "extensions": ["pdf", "jpg", "jpeg", "png"],
    },
    "PAN": {
        "value": "pan",
        "extensions": ["jpg", "jpeg", "png", "pdf"],
    },
    "Payslip": {
        "value": "payslip",
        "extensions": ["pdf", "jpg", "jpeg", "png"],
    },
}

FIELD_CONFIG = {
    "aadhaar": [
        ("name",        "Name"),
        ("father_name", "Father's Name"),
        ("dob",         "Date of Birth"),
        ("aadhaar",     "Aadhaar Number"),
    ],
    "pan": [
        ("pan",         "PAN Number"),
        ("name",        "Name"),
        ("father_name", "Father's Name"),
    ],
    "payslip": [
        ("uan", "UAN Number"),
    ],
}


# --------------------------------------------------
# HELPERS
# --------------------------------------------------

def save_uploaded_file(uploaded_file) -> str:
    """
    Write an uploaded file to a temp path and return the path.
    The file handle is closed before returning so Windows doesn't
    lock it when downstream OCR libraries try to open it.
    """
    suffix = "." + uploaded_file.name.rsplit(".", 1)[-1]
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        tmp.write(uploaded_file.getbuffer())
    finally:
        tmp.close()   # <-- close handle immediately; safe on all platforms
    return tmp.name


def safe_remove(path: str):
    try:
        if path and os.path.exists(path):
            os.remove(path)
    except OSError:
        pass


def render_result_form(result: dict, doc_type: str, render_key: int) -> dict:
    fields = FIELD_CONFIG.get(doc_type, [])
    if not fields:
        st.warning(f"No field config defined for doc_type '{doc_type}'.")
        return {}

    st.subheader("Extracted Information")
    edited = {}
    for key, label in fields:
        raw = result.get(key)
        if raw is None:
            display = ""
        elif isinstance(raw, float) and raw == int(raw):
            display = str(int(raw))
        else:
            display = str(raw)
        edited[key] = st.text_input(label, value=display, key=f"field_{key}_{render_key}")
    return edited


# --------------------------------------------------
# SESSION STATE INIT
# --------------------------------------------------

for _k, _v in [
    ("result",          None),
    ("result_doc_type", ""),
    ("render_key",      0),
    ("last_file_id",    None),
]:
    if _k not in st.session_state:
        st.session_state[_k] = _v


# --------------------------------------------------
# UI
# --------------------------------------------------

st.title("📄 HRMS Document Auto Fill")
st.caption("Upload employee documents and auto-populate profile information")

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Document Upload")

    document_display = st.selectbox("Select Document Type", list(DOCUMENT_TYPES.keys()))
    selected_doc     = DOCUMENT_TYPES[document_display]

    # ---- Aadhaar: choose upload mode ----
    if selected_doc["value"] == "aadhaar":
        aadhaar_mode = st.radio(
            "Upload mode",
            ["Single image / PDF (auto-detect)", "Separate front & back images"],
            horizontal=True,
        )
    else:
        aadhaar_mode = None

    # ---- File uploaders ----
    uploaded_file       = None
    uploaded_front      = None
    uploaded_back       = None

    if selected_doc["value"] == "aadhaar" and aadhaar_mode == "Separate front & back images":
        uploaded_front = st.file_uploader(
            "Aadhaar Front",
            type=["jpg", "jpeg", "png"],
            key="uploader_front",
        )
        uploaded_back = st.file_uploader(
            "Aadhaar Back",
            type=["jpg", "jpeg", "png"],
            key="uploader_back",
        )
        # Use a composite file_id based on both uploads
        _fid_front = (uploaded_front.name, uploaded_front.size) if uploaded_front else None
        _fid_back  = (uploaded_back.name,  uploaded_back.size)  if uploaded_back  else None
        file_id    = (_fid_front, _fid_back)
    else:
        uploaded_file = st.file_uploader(
            "Upload File",
            type=selected_doc["extensions"],
            key="uploader_single",
        )
        file_id = (uploaded_file.name, uploaded_file.size) if uploaded_file else None

    # Clear stale result when a different file is uploaded
    if file_id and file_id != st.session_state["last_file_id"]:
        st.session_state["result"]       = None
        st.session_state["last_file_id"] = file_id

    extract_btn = st.button("Extract Information", use_container_width=True)

# ---- Extraction logic ----
with col2:
    if extract_btn:
        temp_paths = []
        result = None

        try:
            # ---- Aadhaar front+back mode ----
            if selected_doc["value"] == "aadhaar" and aadhaar_mode == "Separate front & back images":
                if not uploaded_front and not uploaded_back:
                    st.error("Please upload at least one image (front or back).")
                    st.stop()

                with st.spinner("Extracting information…"):
                    front_result = None
                    back_result  = None

                    if uploaded_front:
                        fp = save_uploaded_file(uploaded_front)
                        temp_paths.append(fp)
                        front_result = extract_from_aadhaar_front_image(fp)

                    if uploaded_back:
                        bp = save_uploaded_file(uploaded_back)
                        temp_paths.append(bp)
                        back_result = extract_from_aadhaar_back_image(bp)

                    result = merge_aadhaar_results(front_result, back_result)
                    result["doc_type"] = "aadhaar"

            # ---- Single file mode (all doc types) ----
            else:
                if uploaded_file is None:
                    st.error("Please upload a file.")
                    st.stop()

                with st.spinner("Extracting information…"):
                    fp = save_uploaded_file(uploaded_file)
                    temp_paths.append(fp)
                    result = process_document(fp, selected_doc["value"])

            if result and result.get("error"):
                st.error(f"Extraction error: {result['error']}")
                st.stop()

            st.success("Document processed successfully.")
            st.session_state["result"]          = result
            st.session_state["result_doc_type"] = selected_doc["value"]
            st.session_state["render_key"]      += 1

        except Exception as e:
            st.exception(e)

        finally:
            for p in temp_paths:
                safe_remove(p)

    # ---- Results display ----
    if st.session_state["result"] is not None:
        result     = st.session_state["result"]
        doc_type   = st.session_state["result_doc_type"]
        render_key = st.session_state["render_key"]

        edited_data = render_result_form(result, doc_type, render_key)

        st.divider()

        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("Save Details", use_container_width=True):
                st.success("Details saved successfully.")
        with col_b:
            json_str = json.dumps(edited_data, indent=2, ensure_ascii=False)
            st.download_button(
                label="Download JSON",
                data=json_str,
                file_name="extracted_data.json",
                mime="application/json",
                use_container_width=True,
            )

        with st.expander("View Raw JSON"):
            st.json(edited_data)