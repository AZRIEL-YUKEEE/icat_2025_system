import streamlit as st
import pandas as pd
import mysql.connector
from io import BytesIO

st.set_page_config(page_title="Student Uploader", layout="centered")

st.title("📤 Student Data Uploader")

# --- Database connection ---
def get_connection():
    return mysql.connector.connect(
        host="127.0.0.1",
        user="root",          # change if needed
        password="",          # change if needed
        database="icat_db"    # change to your DB
    )

# --- Save records to database ---
def save_students(records):
    conn = get_connection()
    cursor = conn.cursor()
    sql = """INSERT INTO students 
             (application_number, family_name, first_name, middle_name, sex, strand, course,
              general_ability, verbal_aptitude, numerical_aptitude, spatial_aptitude,
              perceptual_aptitude, manual_dexterity, date_taken)
             VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)"""
    try:
        cursor.executemany(sql, records)
        conn.commit()
    except Exception as e:
        conn.rollback()
        st.error(f"❌ Database Error: {e}")
        return False
    finally:
        cursor.close()
        conn.close()
    return True

# --- File uploader ---
uploaded_file = st.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx", "xls"])

if uploaded_file:
    try:
        filename = uploaded_file.name.lower()

        # ✅ Read file
        if filename.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        st.subheader("📑 Raw Data Preview")
        st.dataframe(df.head())

        # ✅ Ensure date_taken exists
        if "date_taken" not in df.columns:
            df["date_taken"] = pd.Timestamp.now().strftime("%Y-%m-%d")

        # ✅ Replace NaN and empty values with None (→ NULL in MySQL)
        df = df.where(pd.notnull(df), None)   # NaN → None
        df = df.applymap(lambda x: None if (isinstance(x, str) and str(x).strip().lower() in ["", "nan", "none"]) else x)

        # ✅ Convert DataFrame to list of tuples with NaN → None
        records = []
        for _, row in df.iterrows():
            records.append((
                None if pd.isna(row.get("Application Number")) else row.get("Application Number"),
                None if pd.isna(row.get("Family_Name")) else row.get("Family_Name"),
                None if pd.isna(row.get("First_Name")) else row.get("First_Name"),
                None if pd.isna(row.get("Middle_Name")) else row.get("Middle_Name"),
                None if pd.isna(row.get("Sex")) else row.get("Sex"),
                None if pd.isna(row.get("Strand")) else row.get("Strand"),
                None if pd.isna(row.get("Course")) else row.get("Course"),
                None if pd.isna(row.get("General_Ability")) else row.get("General_Ability"),
                None if pd.isna(row.get("Verbal_Aptitude")) else row.get("Verbal_Aptitude"),
                None if pd.isna(row.get("Numerical_Aptitude")) else row.get("Numerical_Aptitude"),
                None if pd.isna(row.get("Spatial_Aptitude")) else row.get("Spatial_Aptitude"),
                None if pd.isna(row.get("Perceptual_Aptitude")) else row.get("Perceptual_Aptitude"),
                None if pd.isna(row.get("Manual_Dexterity")) else row.get("Manual_Dexterity"),
                None if pd.isna(row.get("date_taken")) else row.get("date_taken"),
            ))

        st.subheader("✅ Cleaned Data Preview")
        st.dataframe(df.head())

        # --- Upload button ---
        if st.button("🚀 Upload to Database"):
            if save_students(records):
                st.success(f"✅ Saved {len(records)} records successfully!")
            else:
                st.error("❌ Upload failed. Check database connection and table structure.")

    except Exception as e:
        st.error(f"❌ Error processing file: {e}")
