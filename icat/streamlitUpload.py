import streamlit as st
import pandas as pd
import mysql.connector
from io import BytesIO
from datetime import datetime


PASSWORD = "icat2025password"  # <-- change this!

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    st.title("🔐 Secure Access")
    password = st.text_input("Enter password:", type="password")
    if st.button("Login"):
        if password == PASSWORD:
            st.session_state.authenticated = True
            st.success("✅ Access granted")
        else:
            st.error("❌ Wrong password")
    st.stop()


# Database connection
def get_connection():
    return mysql.connector.connect(
        host="127.0.0.1",
        user="root",          # change if needed
        password="",          # change if needed
        database="icat_db"    # change to your DB
    )

# Save to database
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
        st.error(f"❌ DB Error: {str(e)}")
    finally:
        cursor.close()
        conn.close()

# Streamlit UI
st.title("📤 Student Data Uploader")
st.write("Upload a **CSV or Excel** file to insert student records into MySQL")

uploaded_file = st.file_uploader("Choose a file", type=["csv", "xlsx", "xls"])

if uploaded_file is not None:
    try:
        # ✅ Read file into pandas
        if uploaded_file.name.lower().endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        st.write("✅ File Loaded Successfully:")
        st.dataframe(df.head())

        # ✅ Ensure date_taken column exists
        if "date_taken" not in df.columns:
            df["date_taken"] = datetime.now().strftime("%Y-%m-%d")

        # ✅ Replace NaN/empty with None
        df = df.where(pd.notnull(df), None)
        df = df.applymap(
            lambda x: None if (
                x is None or
                (isinstance(x, str) and x.strip().lower() in ["", "nan", "none"])
            ) else x
        )

        # ✅ Convert to list of tuples
        records = []
        for _, row in df.iterrows():
            records.append((
                row.get("Application Number"),
                row.get("Family_Name"),
                row.get("First_Name"),
                row.get("Middle_Name"),
                row.get("Sex"),
                row.get("Strand"),
                row.get("Course"),
                row.get("General_Ability"),
                row.get("Verbal_Aptitude"),
                row.get("Numerical_Aptitude"),
                row.get("Spatial_Aptitude"),
                row.get("Perceptual_Aptitude"),
                row.get("Manual_Dexterity"),
                row.get("date_taken"),
            ))

        if st.button("🚀 Upload to Database"):
            save_students(records)
            st.success(f"✅ Successfully saved {len(records)} records to the database!")

    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
