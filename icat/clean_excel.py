import streamlit as st
import pandas as pd
from io import BytesIO

st.set_page_config(page_title="Student File Cleaner", layout="centered")

st.title("📂 Student File Cleaner")

st.write("Upload a **CSV** or **Excel** file, and this tool will clean it so it’s ready for upload to FastAPI/MySQL.")

# Upload file
uploaded_file = st.file_uploader("Choose a file", type=["csv", "xlsx"])

if uploaded_file:
    try:
        # Read file
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        st.subheader("🔍 Raw Data Preview")
        st.dataframe(df.head())

        # ✅ Expected columns for FastAPI
        expected_columns = [
            "Application_Number", "Family_Name", "First_Name", "Middle_Name", "Sex", "Strand", "Course",
            "General_Ability", "Verbal_Aptitude", "Numerical_Aptitude", "Spatial_Aptitude",
            "Perceptual_Aptitude", "Manual_Dexterity", "date_taken"
        ]

        # Normalize headers (strip spaces, case-insensitive match)
        df.columns = [c.strip().replace(" ", "_") for c in df.columns]

        # Force rename to expected casing
        rename_map = {c.lower(): c for c in expected_columns}
        df.columns = [rename_map.get(c.lower(), c) for c in df.columns]

        # Ensure all expected columns exist
        for col in expected_columns:
            if col not in df.columns:
                df[col] = None

        # Ensure date_taken column
        if "date_taken" not in df.columns:
            df["date_taken"] = pd.Timestamp.now().strftime("%Y-%m-%d")

        # Clean empty values → None
        df = df.where(pd.notnull(df), None)
        df = df.applymap(lambda x: None if (isinstance(x, str) and x.strip() == "") else x)

        st.subheader("✅ Cleaned Data Preview")
        st.dataframe(df.head())

        # Convert to CSV/Excel for download
        def to_csv(df):
            return df.to_csv(index=False).encode("utf-8")

        def to_excel(df):
            output = BytesIO()
            with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
                df.to_excel(writer, index=False, sheet_name="Cleaned")
            return output.getvalue()

        # Download buttons
        st.download_button(
            label="⬇️ Download Cleaned CSV",
            data=to_csv(df),
            file_name="cleaned_students.csv",
            mime="text/csv"
        )

        st.download_button(
            label="⬇️ Download Cleaned Excel",
            data=to_excel(df),
            file_name="cleaned_students.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

    except Exception as e:
        st.error(f"❌ Error processing file: {e}")
