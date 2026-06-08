import streamlit as st
import tempfile
import os
import json
import pandas as pd

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
        "extensions": ["pdf"]
    },
    "PAN": {
        "value": "pan",
        "extensions": ["jpg", "jpeg", "png"]
    },
    "Resume": {
        "value": "resume",
        "extensions": ["pdf", "docx", "pptx"]
    },
    "Payslip": {
        "value": "payslip",
        "extensions": ["pdf"]
    }
}


# --------------------------------------------------
# HELPERS
# --------------------------------------------------

def save_uploaded_file(uploaded_file):
    suffix = "." + uploaded_file.name.split(".")[-1]

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getbuffer())
        return tmp.name


def format_label(key):
    return key.replace("_", " ").title()


def render_result_form(result):
    """
    Render extracted fields dynamically.
    """

    edited_data = {}

    st.subheader("Extracted Information")

    for key, value in result.items():

        if key == "doc_type":
            continue

        label = format_label(key)

        # None values
        if value is None:
            edited_data[key] = st.text_input(label, value="")
            continue

        # Simple fields
        if isinstance(value, (str, int, float)):
            edited_data[key] = st.text_input(label, value=str(value))
            continue

        # List of strings
        if isinstance(value, list) and (
            len(value) == 0 or isinstance(value[0], str)
        ):
            text = "\n".join(value)
            edited_text = st.text_area(
                label,
                value=text,
                height=120
            )

            edited_data[key] = [
                x.strip()
                for x in edited_text.split("\n")
                if x.strip()
            ]
            continue

        # List of dictionaries
        if isinstance(value, list) and len(value) > 0 and isinstance(value[0], dict):

            st.markdown(f"### {label}")

            df = pd.DataFrame(value)

            edited_df = st.data_editor(
                df,
                use_container_width=True,
                num_rows="dynamic",
                key=f"table_{key}"
            )

            edited_data[key] = edited_df.to_dict(orient="records")
            continue

        # Dictionary
        if isinstance(value, dict):

            st.markdown(f"### {label}")

            edited_sub = {}

            for sub_key, sub_val in value.items():
                edited_sub[sub_key] = st.text_input(
                    format_label(sub_key),
                    value=str(sub_val) if sub_val else "",
                    key=f"{key}_{sub_key}"
                )

            edited_data[key] = edited_sub
            continue

        edited_data[key] = value

    return edited_data


# --------------------------------------------------
# UI
# --------------------------------------------------

st.title("📄 HRMS Document Auto Fill")
st.caption("Upload employee documents and auto-populate profile information")

col1, col2 = st.columns([1, 2])

with col1:

    st.subheader("Document Upload")

    document_display = st.selectbox(
        "Select Document Type",
        list(DOCUMENT_TYPES.keys())
    )

    selected_doc = DOCUMENT_TYPES[document_display]

    allowed_ext = selected_doc["extensions"]

    uploaded_file = st.file_uploader(
        "Upload File",
        type=allowed_ext
    )

    extract_btn = st.button(
        "Extract Information",
        use_container_width=True
    )

with col2:

    if extract_btn:

        if uploaded_file is None:
            st.error("Please upload a file.")
            st.stop()

        temp_file_path = None

        try:

            temp_file_path = save_uploaded_file(uploaded_file)

            with st.spinner("Extracting information..."):

                result = process_document(
                    temp_file_path,
                    selected_doc["value"]
                )

            if result.get("error"):
                st.error(result["error"])
                st.stop()

            st.success("Document processed successfully")

            st.session_state["result"] = result

        except Exception as e:
            st.exception(e)

        finally:
            if temp_file_path and os.path.exists(temp_file_path):
                try:
                    os.remove(temp_file_path)
                except:
                    pass

    if "result" in st.session_state:

        result = st.session_state["result"]

        edited_data = render_result_form(result)

        st.divider()

        col_a, col_b = st.columns(2)

        with col_a:

            if st.button(
                "Save Details",
                use_container_width=True
            ):
                st.success("Details validated successfully.")

        with col_b:

            json_str = json.dumps(
                edited_data,
                indent=2,
                ensure_ascii=False
            )

            st.download_button(
                label="Download JSON",
                data=json_str,
                file_name="extracted_data.json",
                mime="application/json",
                use_container_width=True
            )

        with st.expander("View Raw JSON"):

            st.json(edited_data)