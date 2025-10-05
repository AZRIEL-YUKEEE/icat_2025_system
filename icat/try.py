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
        user="root",  # change if needed
        password="",  # change if needed
        database="icat_db"
    )

# ----------------- LOAD DATA -----------------
def load_data():
    conn = get_connection()
    query = "SELECT * FROM students"
    df = pd.read_sql(query, conn)
    conn.close()

    # Normalize sex
    df["sex"] = df["sex"].str.lower().str.strip()
    sex_map = {"male": 0, "female": 1}
    df["sex_code"] = df["sex"].map(sex_map)

    # Clean course names
    df["course_clean"] = df["course"].str.upper().str.strip()

    # Expanded course mapping
    course_map = {
        "BSED": 0, "BSED-ENGLISH": 0, "BSED-FILIPINO": 0, "BSED-MATH": 0,
        "BSED-SCIENCE": 0, "BSED-SOCIAL": 0, "BSED/BAEL": 0, "EDUC": 0, "EDUC-": 0,
        "BEED": 1,
        "BPED": 2,
        "BSBA": 3, "BSBA-FM": 3, "BSBA-MARKETING": 3,
        "BS MATH": 4, "BS MATHEMATICS": 4,
        "BAEL": 5, "BA ENGLISH LANGUAGE": 5,
        "BAP": 6, "PSYCHOLOGY": 6, "BA PSYCHOLOGY": 6,
        "BASS": 7, "BA SOCIAL SCIENCE": 7,
        "BS ENTREP": 8, "BS ENTREPRENEURSHIP": 8,
        "BSIT": 9, "BS INFORMATION TECHNOLOGY": 9
    }
    df["course_code"] = df["course_clean"].map(course_map).fillna(-1).astype(int)

    # Define aptitude columns
    aptitude_cols = [
        "general_ability", "verbal_aptitude", "numerical_aptitude",
        "spatial_aptitude", "perceptual_aptitude", "manual_dexterity"
    ]

    # Compute overall performance
    df["overall_performance"] = df[aptitude_cols].mean(axis=1)

    return df

# ----------------- GENDER COMPARISON (SCATTER) -----------------
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

# ----------------- COURSE COMPARISON (SCATTER + KMEANS) -----------------
def plot_course_scatter(df, score, title, n_clusters=3):
    plt.figure(figsize=(12, 8))

    X = df[[score, "overall_performance"]].dropna()

    # Apply k-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df["cluster"] = kmeans.fit_predict(X)
    centroids = kmeans.cluster_centers_

    sns.scatterplot(
        x=score, y="overall_performance",
        hue="course_code", data=df,
        palette="tab10", s=60, alpha=0.7
    )

    # Plot centroids
    plt.scatter(centroids[:, 0], centroids[:, 1],
                c="red", marker="X", s=200, label="Centroids")

    course_labels = {
        -1: "Unknown",
         0: "BSEd / Educ",
         1: "BEEd",
         2: "BPEd",
         3: "BSBA",
         4: "BS Math",
         5: "BA English",
         6: "BA Psychology",
         7: "BA Social Science",
         8: "BS Entrep",
         9: "BSIT"
    }

    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend(
        handles=handles,
        labels=[course_labels.get(int(c), c) for c in df["course_code"].unique()],
        title="Course",
        bbox_to_anchor=(1.05, 1),
        loc='upper left'
    )

    plt.title(title, fontsize=14, fontweight="bold")
    plt.xlabel(score.replace("_", " ").title(), fontsize=12)
    plt.ylabel("Overall Performance", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.6)
    st.pyplot(plt)
    plt.clf()

# ----------------- MAIN APP -----------------
st.set_page_config(page_title="ICAT Analysis Dashboard", layout="wide")
st.title("📊 ICAT Student Analysis Dashboard")

df = load_data()
if df.empty:
    st.warning("No student records found in the database.")
    st.stop()

st.success(f"✅ Data loaded: {df.shape[0]} students")

# GENDER COMPARISON
st.header("🔹 Gender Comparison (0 = Male, 1 = Female)")
scores = [
    "general_ability", "verbal_aptitude", "manual_dexterity",
    "numerical_aptitude", "spatial_aptitude", "perceptual_aptitude"
]
for score in scores:
    plot_gender_comparison(df, score, f"{score.replace('_', ' ').title()} by Gender")

# COURSE COMPARISON (WITH CLUSTERS)
st.header("🔹 Course Scatter Plots (-1 to 9 with Clusters)")
for score in scores:
    plot_course_scatter(df, score, f"{score.replace('_', ' ').title()} vs Overall Performance", n_clusters=3)
