import streamlit as st
import pandas as pd
import mysql.connector
from io import BytesIO
import warnings

warnings.filterwarnings('ignore')


# ----------------- DATABASE CONNECTION -----------------
def get_connection():
    return mysql.connector.connect(
        host="127.0.0.1",
        user="root",
        password="",
        database="icat_db"
    )


# ----------------- ENHANCED SAVE STUDENT RECORDS -----------------
def save_students(records):
    """Save multiple student records with proper error handling"""
    conn = get_connection()
    cursor = conn.cursor()

    sql = """INSERT INTO students 
             (application_number, family_name, first_name, middle_name, sex, strand, course,
              general_ability, verbal_aptitude, numerical_aptitude, spatial_aptitude,
              perceptual_aptitude, manual_dexterity, date_taken)
             VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"""

    try:
        cursor.executemany(sql, records)
        conn.commit()
        return True
    except Exception as e:
        conn.rollback()
        st.error(f"Database error: {str(e)}")
        return False
    finally:
        cursor.close()
        conn.close()


# ----------------- ENHANCED DATA VALIDATION -----------------
def clean_dataframe(df):
    """Clean and validate the uploaded dataframe"""
    expected_columns = {
        'Application Number': 'application_number',
        'Family_Name': 'family_name',
        'First_Name': 'first_name',
        'Middle_Name': 'middle_name',
        'Sex': 'sex',
        'Strand': 'strand',
        'Course': 'course',
        'General_Ability': 'general_ability',
        'Verbal_Aptitude': 'verbal_aptitude',
        'Numerical_Aptitude': 'numerical_aptitude',
        'Spatial_Aptitude': 'spatial_aptitude',
        'Perceptual_Aptitude': 'perceptual_aptitude',
        'Manual_Dexterity': 'manual_dexterity'
    }

    df = df.rename(columns={k: v for k, v in expected_columns.items() if k in df.columns})

    text_columns = ['application_number', 'family_name', 'first_name', 'middle_name', 'sex', 'strand', 'course']
    numeric_columns = ['general_ability', 'verbal_aptitude', 'numerical_aptitude',
                       'spatial_aptitude', 'perceptual_aptitude', 'manual_dexterity']

    for col in text_columns:
        if col in df.columns:
            df[col] = df[col].astype(str).replace(['nan', 'NaN', 'None', ''], '').fillna('')

    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    if 'date_taken' not in df.columns:
        df['date_taken'] = pd.Timestamp.now().strftime('%Y-%m-%d')
    else:
        df['date_taken'] = pd.to_datetime(df['date_taken'], errors='coerce').dt.strftime('%Y-%m-%d').fillna(
            pd.Timestamp.now().strftime('%Y-%m-%d'))

    return df


# ----------------- STREAMLIT UI -----------------
st.set_page_config(page_title="ICAT Student Data Upload", layout="wide")
st.title("📂 ICAT Student Data Uploader")

st.subheader("Upload Student Data (CSV or Excel)")

uploaded_file = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx", "xls"])

if uploaded_file:
    try:
        if uploaded_file.name.endswith(".csv"):
            df_upload = pd.read_csv(uploaded_file)
        else:
            df_upload = pd.read_excel(uploaded_file)

        df_upload = clean_dataframe(df_upload)

        # Preview
        st.write("### 📋 Preview of Uploaded Data (After Cleaning)")
        st.dataframe(df_upload.head(10))

        # Data summary
        st.write("### 📊 Data Summary")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Records", len(df_upload))
        with col2:
            st.metric("Columns", len(df_upload.columns))
        with col3:
            missing_values = df_upload.isnull().sum().sum()
            st.metric("Missing Values", missing_values)

        # Save options
        st.write("### 💾 Save Options")
        col1, col2 = st.columns(2)

        with col1:
            if st.button("🚀 Save All Data to Database", type="primary", use_container_width=True):
                records = []
                for _, row in df_upload.iterrows():
                    record = tuple(row[col] if col in df_upload.columns else '' for col in [
                        'application_number', 'family_name', 'first_name', 'middle_name',
                        'sex', 'strand', 'course', 'general_ability', 'verbal_aptitude',
                        'numerical_aptitude', 'spatial_aptitude', 'perceptual_aptitude',
                        'manual_dexterity', 'date_taken'
                    ])
                    records.append(record)

                if save_students(records):
                    st.success(f"✅ Successfully saved {len(records)} records to database!")
                else:
                    st.error("❌ Failed to save records to database")

        with col2:
            if st.button("📤 Save Sample (First 10 Records)", use_container_width=True):
                sample_df = df_upload.head(10)
                records = []
                for _, row in sample_df.iterrows():
                    record = tuple(row[col] if col in sample_df.columns else '' for col in [
                        'application_number', 'family_name', 'first_name', 'middle_name',
                        'sex', 'strand', 'course', 'general_ability', 'verbal_aptitude',
                        'numerical_aptitude', 'spatial_aptitude', 'perceptual_aptitude',
                        'manual_dexterity', 'date_taken'
                    ])
                    records.append(record)

                if save_students(records):
                    st.success(f"✅ Successfully saved 10 sample records to database!")
                else:
                    st.error("❌ Failed to save sample records")

    except Exception as e:
        st.error(f"❌ Error processing file: {str(e)}")
