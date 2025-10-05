import streamlit as st
import pandas as pd
import mysql.connector
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np


# ----------------- DATABASE CONNECTION -----------------
def get_connection():
    return mysql.connector.connect(
        host="localhost",
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
    return df


# ----------------- STREAMLIT APP -----------------
st.set_page_config(page_title="ICAT Student Aptitude Clustering", layout="wide")
st.title("📊 ICAT Student Aptitude Clustering")

# Load data
try:
    df = load_data()
except Exception as e:
    st.error(f"❌ Failed to load data: {e}")
    st.stop()

if df.empty:
    st.warning("⚠️ No student records found.")
    st.stop()

# ----------------- COURSE CLEANING -----------------
df["course_clean"] = df["course"].astype(str).str.upper().str.strip()

course_map = {
    "BSIT": 9, "BACHELOR OF SCIENCE IN INFORMATION TECHNOLOGY": 9,
    "BAEL": 5, "BACHELOR OF ARTS IN ENGLISH LANGUAGE": 5,
    "BAP": 6, "PSYCHOLOGY": 6, "BACHELOR OF ARTS IN PSYCHOLOGY": 6,
    "BASS": 7, "BACHELOR OF ARTS IN SOCIAL SCIENCE": 7,
    "BEED": 1, "BACHELOR OF ELEMENTARY EDUCATION": 1,
    "BPA": 2, "BPED": 2, "BACHELOR OF PHYSICAL EDUCATION": 2,
    "BS ENTREP": 8, "BACHELOR OF SCIENCE IN ENTREPRENEURSHIP": 8,
    "BSBA": 3, "BSBA-FM": 3, "BSBA-MARKETING": 3, "BACHELOR OF SCIENCE IN BUSINESS ADMINISTRATION": 3,
    "BSED-ENGLISH": 0, "BSED-FILIPINO": 0, "BSED-MATH": 0,
    "BSED-SCIENCE": 0, "BSED-SOCIAL": 0, "BSED/BAEL": 0,
    "BSED": 0, "EDUC": 0, "EDUC-": 0, "BACHELOR OF SECONDARY EDUCATION": 0,
}
df["course_code"] = df["course_clean"].map(course_map).fillna(-1).astype(int)

# Normalize gender — keep NaN for filtering
df["sex"] = df["sex"].astype(str).str.lower().str.strip()
sex_map = {"male": 0, "female": 1}
df["sex_code"] = df["sex"].map(sex_map)  # unmapped → NaN

# ----------------- FEATURES -----------------
features = [
    "general_ability", "verbal_aptitude", "numerical_aptitude",
    "spatial_aptitude", "perceptual_aptitude", "manual_dexterity"
]

# Compute overall performance
df["overall_performance"] = df[features].mean(axis=1)

# Drop rows with missing aptitude data
df_valid = df.dropna(subset=features + ["overall_performance"]).copy()

if df_valid.empty:
    st.error("❌ No valid aptitude data found.")
    st.stop()

# ----------------- COURSE LABELS & COLORS -----------------
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

# Assign unique colors using tab20 (11 distinct colors)
tab20_colors = plt.cm.tab20(np.linspace(0, 1, 20))
selected_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 15]
unique_colors = [tab20_colors[i] for i in selected_indices]
course_codes_sorted = sorted(course_labels.keys())
course_color_map = {code: unique_colors[i] for i, code in enumerate(course_codes_sorted)}


# ----------------- PLOTTING FUNCTIONS -----------------

def plot_gender_scatter_with_centroids(score_col):
    data = df_valid[[score_col, "overall_performance", "sex_code"]].copy()
    data = data[data["sex_code"].isin([0, 1])].dropna()

    if data.empty:
        st.warning(f"No valid gender data for {score_col}")
        return

    X = data[[score_col, "overall_performance"]].values
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    kmeans.fit(X)
    centroids = kmeans.cluster_centers_

    color_map = {0: "steelblue", 1: "indianred"}
    colors = data["sex_code"].map(color_map)

    plt.figure(figsize=(8, 6))
    plt.scatter(
        data[score_col],
        data["overall_performance"],
        c=colors,
        alpha=0.8,
        s=60,
        edgecolor='k',
        linewidth=0.3
    )
    plt.scatter(
        centroids[:, 0], centroids[:, 1],
        c="red", marker="X", s=250,
        edgecolor='white', linewidth=1.5
    )

    plt.title(f"{score_col.replace('_', ' ').title()} vs Overall Performance (by Gender)", fontweight='bold')
    plt.xlabel(score_col.replace("_", " ").title())
    plt.ylabel("Overall Performance")
    plt.ylim(0, 100)
    plt.yticks(np.arange(0, 101, 10))
    plt.grid(True, linestyle='--', alpha=0.5)

    # ✅ FIXED LEGEND: use handles + labels
    plt.legend(
        handles=[
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='steelblue', markersize=8),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='indianred', markersize=8),
            plt.Line2D([0], [0], marker='X', color='w', markerfacecolor='red', markersize=10)
        ],
        labels=['Male', 'Female', 'K-Means Centroids']
    )
    st.pyplot(plt)
    plt.clf()


def plot_course_scatter_with_centroids(score_col):
    data = df_valid[[score_col, "overall_performance", "course_code"]].copy()
    data = data.dropna(subset=[score_col, "overall_performance", "course_code"])

    if data.empty:
        st.warning(f"No valid course data for {score_col}")
        return

    X = data[[score_col, "overall_performance"]].values
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    kmeans.fit(X)
    centroids = kmeans.cluster_centers_

    plt.figure(figsize=(11, 6))
    for code in course_codes_sorted:
        subset = data[data["course_code"] == code]
        if not subset.empty:
            plt.scatter(
                subset[score_col],
                subset["overall_performance"],
                color=course_color_map[code],
                label=course_labels[code],
                alpha=0.8,
                s=60,
                edgecolor='k',
                linewidth=0.3
            )

    plt.scatter(
        centroids[:, 0], centroids[:, 1],
        c="red", marker="X", s=250,
        edgecolor='white', linewidth=1.5
    )

    plt.title(f"{score_col.replace('_', ' ').title()} vs Overall Performance (by Course)", fontweight='bold')
    plt.xlabel(score_col.replace("_", " ").title())
    plt.ylabel("Overall Performance")
    plt.ylim(0, 100)
    plt.yticks(np.arange(0, 101, 10))
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(title="Course", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    st.pyplot(plt)
    plt.clf()


# ----------------- GENERATE PLOTS -----------------
scores = [
    "general_ability", "verbal_aptitude", "numerical_aptitude",
    "spatial_aptitude", "perceptual_aptitude", "manual_dexterity"
]

st.header("📊 Aptitude Scores by Gender (Color = Gender, with K-Means Centroids)")
for score in scores:
    plot_gender_scatter_with_centroids(score)

st.header("📊 Aptitude Scores by Course (Color = Course, with K-Means Centroids)")
for score in scores:
    plot_course_scatter_with_centroids(score)