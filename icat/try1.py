import streamlit as st
import pandas as pd
import mysql.connector
import matplotlib.pyplot as plt
import seaborn as sns

# ----------------- DATABASE CONNECTION -----------------
def get_connection():
    return mysql.connector.connect(
        host="127.0.0.1",
        user="root",  # change if needed
        password="",  # change if needed
        database="icat_db"
    )

# ----------------- SAVE STUDENT RECORD -----------------
def save_students(records):
    conn = get_connection()
    cur = conn.cursor()
    sql = """INSERT INTO students 
             (application_number, family_name, first_name, middle_name, sex, strand, course,
              general_ability, verbal_aptitude, numerical_aptitude, spatial_aptitude,
              perceptual_aptitude, manual_dexterity, date_taken)
             VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)"""
    cur.executemany(sql, records)  # bulk insert
    conn.commit()
    cur.close()
    conn.close()

# ----------------- LOAD DATA FOR PLOTTING -----------------
def load_data():
    conn = get_connection()
    query = "SELECT * FROM students"
    df = pd.read_sql(query, conn)
    conn.close()

    # Normalize sex
    df["sex"] = df["sex"].str.lower().str.strip()
    sex_map = {"male": 0, "female": 1}
    df["sex_code"] = df["sex"].map(sex_map)

    # Clean course names (remove suffixes like "- Filipino")
    df["course_clean"] = df["course"].str.replace(r' - .*', '', regex=True).str.strip()

    # Course mapping (exactly as you provided)
    course_map = {
        "Bachelor of Secondary Education": 0,
        "Bachelor of Elementary Education": 1,
        "Bachelor of Physical Education": 2,
        "Bachelor of Science in Business Administration": 3,
        "Bachelor of Science in Mathematics": 4,
        "Bachelor of Arts in English Language": 5,
        "Bachelor of Arts in Psychology": 6,
        "Bachelor of Arts in Social Science": 7,
        "Bachelor of Science in Entrepreneurship": 8,
        "Bachelor of Science in Information Technology": 9
    }

    df["course_code"] = df["course_clean"].map(course_map)

    # Define aptitude columns
    aptitude_cols = [
        "general_ability", "verbal_aptitude", "numerical_aptitude",
        "spatial_aptitude", "perceptual_aptitude", "manual_dexterity"
    ]

    # Compute overall performance
    df["overall_performance"] = df[aptitude_cols].mean(axis=1)

    return df

# ----------------- GENDER COMPARISON PLOTS -----------------
def plot_gender_comparison(df, score, title):
    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        x=score, y="overall_performance",
        hue="sex_code", data=df,
        palette={0: "blue", 1: "orange"}, s=70
    )
    plt.title(title)
    plt.xlabel(score.replace("_", " ").title())
    plt.ylabel("Overall Performance")
    plt.legend(title="Gender", labels=["Male (0)", "Female (1)"])
    st.pyplot(plt)
    plt.clf()

# ----------------- COURSE SCATTER PLOT (ALL COURSES TOGETHER) -----------------
def plot_course_comparison_all(df, score, title):
    plt.figure(figsize=(12, 8))

    # Plot all students, color by course_code (0–9)
    sns.scatterplot(
        x=score, y="overall_performance",
        hue="course_code", data=df,
        palette="tab10", s=60, alpha=0.7
    )

    # Custom legend labels
    course_labels = [
        "0 BSEd", "1 BEEd", "2 BPEd", "3 BSBA", "4 BSMath",
        "5 AB English", "6 AB Psychology", "7 AB Social Science",
        "8 BS Entrepreneurship", "9 BSIT"
    ]

    # Get handles and labels from plot
    handles, _ = plt.gca().get_legend_handles_labels()
    plt.legend(handles=handles[:10], labels=course_labels,
               title="Course", bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)

    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel(score.replace("_", " ").title(), fontsize=12)
    plt.ylabel("Overall Performance", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    st.pyplot(plt)
    plt.clf()

# ----------------- STREAMLIT UI -----------------
st.set_page_config(page_title="Upload Files", layout="wide")
st.title("📂 ICAT Student Data Uploader")

st.subheader("Upload Student Data (CSV or Excel)")
uploaded_file = st.file_uploader("Choose a file", type=["csv", "xlsx"])

if uploaded_file:
    # Load the file
    if uploaded_file.name.endswith(".csv"):
        df_upload = pd.read_csv(uploaded_file)
    else:
        df_upload = pd.read_excel(uploaded_file)

    # ✅ Ensure date_taken column exists
    if "date_taken" not in df_upload.columns:
        df_upload["date_taken"] = "2025-01-01"  # default date
    else:
        df_upload["date_taken"] = pd.to_datetime(
            df_upload["date_taken"], errors="coerce"
        ).dt.strftime("%Y-%m-%d").fillna("2025-01-01")

    # ✅ Replace NaN with None (for text/date fields) and 0 (for scores)
    score_columns = [
        "General_Ability", "Verbal_Aptitude", "Numerical_Aptitude",
        "Spatial_Aptitude", "Perceptual_Aptitude", "Manual_Dexterity"
    ]

    for col in score_columns:
        if col in df_upload.columns:
            df_upload[col] = df_upload[col].fillna(0)  # missing score = 0

    df_upload = df_upload.where(pd.notnull(df_upload), None)  # all others → NULL

    st.write("✅ Preview of Uploaded Data")
    st.dataframe(df_upload.head())

    if st.button("Save Uploaded Data to Database"):
        try:
            # Build records for executemany
            records = []
            for _, row in df_upload.iterrows():
                records.append((
                    row['Application Number'], row['Family_Name'], row['First_Name'],
                    row['Middle_Name'], row['Sex'], row['Strand'], row['Course'],
                    row['General_Ability'], row['Verbal_Aptitude'], row['Numerical_Aptitude'],
                    row['Spatial_Aptitude'], row['Perceptual_Aptitude'], row['Manual_Dexterity'],
                    row['date_taken']
                ))
            save_students(records)
            st.success("🎉 All uploaded data saved successfully to the database!")
        except Exception as e:
            st.error(f"❌ Error saving data: {e}")

# ----------------- AFTER UPLOAD: SHOW PLOTS -----------------
else:
    # Only show plots when no file is uploaded (or after upload, but we'll do it always)
    st.header("📊 ICAT Analysis Dashboard")

    # Load data for plotting
    df = load_data()
    if df.empty:
        st.warning("No student records found in the database.")
        st.stop()

    st.success(f"✅ Data loaded: {df.shape[0]} students")

    # Aptitude scores
    scores = [
        "general_ability", "verbal_aptitude", "manual_dexterity",
        "numerical_aptitude", "spatial_aptitude", "perceptual_aptitude"
    ]

    # Add overall performance
    df["overall_performance"] = df[scores].mean(axis=1)

    # ---------- GENDER COMPARISONS (UNCHANGED) ----------
    st.header("🔹 Gender Comparison (0 = Male, 1 = Female)")
    for score in scores:
        plot_gender_comparison(df, score, f"{score.replace('_', ' ').title()} by Gender")

    # ---------- COURSE COMPARISON PLOTS (NEW) ----------
    st.header("🔹 Course Comparison: All 10 Courses (0–9)")
    st.markdown("Each plot shows all students from all 10 courses on one aptitude score vs Overall Performance.")

    for score in scores:
        plot_course_comparison_all(df, score, f"{score.replace('_', ' ').title()} vs Overall Performance")