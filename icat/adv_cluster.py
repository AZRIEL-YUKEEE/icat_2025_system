
import streamlit as st
import pandas as pd
import mysql.connector
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

# ----------------- DATABASE CONNECTION -----------------
def get_connection():
    return mysql.connector.connect(
        host="127.0.0.1",
        user="root",
        password="",
        database="icat_db"
    )

# ----------------- SAVE STUDENTS -----------------
def save_students(records):
    conn = get_connection()
    cur = conn.cursor()
    sql = """INSERT INTO students 
             (application_number, family_name, first_name, middle_name, sex, strand, course,
              general_ability, verbal_aptitude, numerical_aptitude, spatial_aptitude,
              perceptual_aptitude, manual_dexterity, date_taken)
             VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)"""
    cur.executemany(sql, records)
    conn.commit()
    cur.close()
    conn.close()

# ----------------- LOAD DATA (ENHANCED) -----------------
def load_data():
    conn = get_connection()
    query = "SELECT * FROM students"
    df = pd.read_sql(query, conn)
    conn.close()

    # Normalize sex
    df["sex"] = df["sex"].str.lower().str.strip()
    sex_map = {"male": 0, "female": 1}
    df["sex_code"] = df["sex"].map(sex_map)

    # Normalize course names
    df["course_clean"] = df["course"].fillna("").str.upper().str.strip()

    # Handle missing/blank → Others
    df.loc[df["course_clean"] == "", "course_clean"] = "OTHERS"

    # Course mapping
    course_map = {
        "BSED": 0, "BSED-ENGLISH": 0, "BSED-FILIPINO": 0,
        "BSED-MATH": 0, "BSED-SCIENCE": 0, "BSED-SOCIAL": 0,
        "BSED/BAEL": 0, "EDUC": 0, "EDUC-": 0,
        "BACHELOR OF SECONDARY EDUCATION": 0,
        "BEED": 1, "BACHELOR OF ELEMENTARY EDUCATION": 1,
        "BPED": 2, "BACHELOR OF PHYSICAL EDUCATION": 2,
        "BSBA": 3, "BSBA-FM": 3, "BSBA-MARKETING": 3,
        "BACHELOR OF SCIENCE IN BUSINESS ADMINISTRATION": 3,
        "BS MATH": 4, "BACHELOR OF SCIENCE IN MATHEMATICS": 4,
        "BAEL": 5, "BACHELOR OF ARTS IN ENGLISH LANGUAGE": 5,
        "BAP": 6, "PSYCHOLOGY": 6, "BACHELOR OF ARTS IN PSYCHOLOGY": 6,
        "BASS": 7, "BACHELOR OF ARTS IN SOCIAL SCIENCE": 7,
        "BS ENTREP": 8, "BACHELOR OF SCIENCE IN ENTREPRENEURSHIP": 8,
        "BSIT": 9, "BACHELOR OF SCIENCE IN INFORMATION TECHNOLOGY": 9,
    }

    df["course_code"] = df["course_clean"].map(course_map).fillna(-1)

    # Aptitude scores
    aptitude_cols = [
        "general_ability", "verbal_aptitude", "numerical_aptitude",
        "spatial_aptitude", "perceptual_aptitude", "manual_dexterity"
    ]

    # Overall performance (mean of all aptitudes)
    df["overall_performance"] = df[aptitude_cols].mean(axis=1)

    return df

# ----------------- PLOTS -----------------
def plot_gender_comparison(df, score, title):
    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        x=score, y="overall_performance",
        hue="sex_code", data=df,
        palette={0: "blue", 1: "orange"}, s=70
    )

    X = df[[score, "overall_performance"]].dropna()
    if len(X) >= 2:
        kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
        kmeans.fit(X)
        centroids = kmeans.cluster_centers_
        plt.scatter(
            centroids[:, 0], centroids[:, 1],
            c="red", marker="X", s=200, label="Centroids"
        )

    plt.title(title)
    plt.xlabel(score.replace("_", " ").title())
    plt.ylabel("Overall Performance")
    plt.legend(title="Gender / Clusters")
    st.pyplot(plt)
    plt.clf()

def plot_course_comparison_all(df, score, title):
    plt.figure(figsize=(12, 8))
    sns.scatterplot(
        x=score, y="perceptual_aptitude",
        hue="course_code", data=df,
        palette="tab10", s=60, alpha=0.7
    )

    X = df[[score, "perceptual_aptitude"]].dropna()
    if len(X) >= 2:
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        kmeans.fit(X)
        centroids = kmeans.cluster_centers_
        plt.scatter(
            centroids[:, 0], centroids[:, 1],
            c="red", marker="X", s=200, label="Centroids"
        )

    course_labels = [
        "0 BSEd", "1 BEEd", "2 BPEd", "3 BSBA", "4 BSMath",
        "5 AB English", "6 AB Psychology", "7 AB Social Science",
        "8 BS Entrepreneurship", "9 BSIT", "-1 Others"
    ]

    handles, _ = plt.gca().get_legend_handles_labels()
    plt.legend(handles=handles, labels=course_labels,
               title="Course", bbox_to_anchor=(1.05, 1),
               loc='upper left', fontsize=9)

    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel(score.replace("_", " ").title(), fontsize=12)
    plt.ylabel("Perceptual Aptitude", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    st.pyplot(plt)
    plt.clf()

# ----------------- STREAMLIT APP -----------------
st.set_page_config(page_title="ICAT Clustering", layout="wide")
st.title("📊 ICAT Student Clustering Dashboard")

# Upload section
st.subheader("Upload Student Data (CSV or Excel)")
uploaded_file = st.file_uploader("Choose a file", type=["csv", "xlsx"])

if uploaded_file:
    if uploaded_file.name.endswith(".csv"):
        df_upload = pd.read_csv(uploaded_file)
    else:
        df_upload = pd.read_excel(uploaded_file)

    if "date_taken" not in df_upload.columns:
        df_upload["date_taken"] = "2025-01-01"
    else:
        df_upload["date_taken"] = pd.to_datetime(
            df_upload["date_taken"], errors="coerce"
        ).dt.strftime("%Y-%m-%d").fillna("2025-01-01")

    score_columns = [
        "General_Ability", "Verbal_Aptitude", "Numerical_Aptitude",
        "Spatial_Aptitude", "Perceptual_Aptitude", "Manual_Dexterity"
    ]

    for col in score_columns:
        if col in df_upload.columns:
            df_upload[col] = df_upload[col].fillna(0)

    df_upload = df_upload.where(pd.notnull(df_upload), None)

    st.write("✅ Preview of Uploaded Data")
    st.dataframe(df_upload.head())

    if st.button("Save Uploaded Data to Database"):
        try:
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

# Analysis Dashboard
st.header("📊 ICAT Analysis Dashboard")
df = load_data()

if df.empty:
    st.warning("No student records found in the database.")
    st.stop()

st.success(f"✅ Data loaded: {df.shape[0]} students")

scores = [
    "general_ability", "verbal_aptitude", "manual_dexterity",
    "numerical_aptitude", "spatial_aptitude", "perceptual_aptitude"
]

# Gender comparison
st.header("🔹 Gender Comparison (0 = Male, 1 = Female)")
for score in scores:
    plot_gender_comparison(df, score, f"{score.replace('_', ' ').title()} by Gender with Centroids")

# Course comparison
st.header("🔹 Course Comparison (Scatter with Centroids)")
st.markdown("Each plot shows all courses (0–9 plus Others) on one aptitude score vs Perceptual Aptitude.")

for score in scores:
    plot_course_comparison_all(df, score, f"{score.replace('_', ' ').title()} by Courses with Centroids")
