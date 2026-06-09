import streamlit as st
import tempfile
import os
import json

from document_processor import process_document

st.set_page_config(
    page_title="HRMS Document Auto Fill",
    page_icon="📄",
    layout="wide"
)

# --------------------------------------------------
# CONFIG
# --------------------------------------------------

DOCUMENT_TYPES = {
    "Aadhaar": {
        "value": "aadhaar",
        "extensions": ["pdf"],
    },
    "PAN": {
        "value": "pan",
        "extensions": ["jpg", "jpeg", "png"],
    },
    "Resume": {
        "value": "resume",
        "extensions": ["pdf", "docx", "pptx"],
    },
    "Payslip": {
        "value": "payslip",
        "extensions": ["pdf"],
    },
}

# Only the fields we need — in display order
FIELD_CONFIG = {
    "aadhaar": [
        ("name",        "Name"),
        ("father_name", "Father's Name"),
        ("dob",         "Date of Birth"),
        ("aadhaar",     "Aadhaar Number"),
    ],
    "pan": [
        ("name",        "Name"),
        ("father_name", "Father's Name"),
        ("pan",         "PAN Number"),
    ],
    "resume": [
        ("name",                   "Full Name"),
        ("email",                  "Email"),
        ("phone",                  "Phone Number"),
        ("total_experience_years", "IT Experience (Years)"),
    ],
    "payslip": [
        ("uan", "UAN Number"),
    ],
}


# --------------------------------------------------
# HELPERS
# --------------------------------------------------

def save_uploaded_file(uploaded_file) -> str:
    suffix = "." + uploaded_file.name.rsplit(".", 1)[-1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getbuffer())
        return tmp.name


def render_result_form(result: dict, doc_type: str, render_key: int) -> dict:
    """Render only the scoped fields for this doc type.

    render_key increments on every new extraction so Streamlit treats
    the text_input widgets as brand-new and respects value= instead of
    returning stale widget state from the previous extraction.
    """
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
            display = str(int(raw))   # 2.0 → "2", keeps 1.5 as "1.5"
        else:
            display = str(raw)

        # Unique key per (field, extraction run) — forces widget re-creation
        edited[key] = st.text_input(label, value=display, key=f"field_{key}_{render_key}")

    return edited


# --------------------------------------------------
# UI
# --------------------------------------------------

# Initialise session state keys we depend on
if "result" not in st.session_state:
    st.session_state["result"] = None
if "result_doc_type" not in st.session_state:
    st.session_state["result_doc_type"] = ""
if "render_key" not in st.session_state:
    st.session_state["render_key"] = 0
if "last_file_id" not in st.session_state:
    st.session_state["last_file_id"] = None

st.title("📄 HRMS Document Auto Fill")
st.caption("Upload employee documents and auto-populate profile information")

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Document Upload")

    document_display = st.selectbox("Select Document Type", list(DOCUMENT_TYPES.keys()))
    selected_doc = DOCUMENT_TYPES[document_display]

    uploaded_file = st.file_uploader(
        "Upload File",
        type=selected_doc["extensions"],
    )

    # Clear stale result when user uploads a different file
    if uploaded_file is not None:
        file_id = (uploaded_file.name, uploaded_file.size)
        if file_id != st.session_state["last_file_id"]:
            st.session_state["result"] = None
            st.session_state["last_file_id"] = file_id

    extract_btn = st.button("Extract Information", use_container_width=True)

with col2:

    if extract_btn:
        if uploaded_file is None:
            st.error("Please upload a file.")
            st.stop()

        temp_path = None
        try:
            temp_path = save_uploaded_file(uploaded_file)

            with st.spinner("Extracting information..."):
                result = process_document(temp_path, selected_doc["value"])

            if result.get("error"):
                st.error(f"Extraction error: {result['error']}")
                st.stop()

            st.success("Document processed successfully.")

            # Store new result and bump render_key so all widgets are re-created
            st.session_state["result"] = result
            st.session_state["result_doc_type"] = selected_doc["value"]
            st.session_state["render_key"] += 1

        except Exception as e:
            st.exception(e)
        finally:
            if temp_path and os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except Exception:
                    pass

    if st.session_state["result"] is not None:
        result      = st.session_state["result"]
        doc_type    = st.session_state["result_doc_type"]
        render_key  = st.session_state["render_key"]

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