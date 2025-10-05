import streamlit as st
import pandas as pd
import mysql.connector
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
import numpy as np


# ----------------- DATABASE CONNECTION -----------------
def get_connection():
    return mysql.connector.connect(
        host="127.0.0.1",
        user="root",
        password="",
        database="icat_db"
    )


# ----------------- LOAD & PREPARE DATA -----------------
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
    df["course_clean"] = df["course"].str.replace(r' - .*', '', regex=True).str.strip()

    # Course mapping
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
    df["course_code"] = df["course_clean"].map(course_map).fillna(-1).astype(int)

    # Aptitude columns
    aptitude_cols = [
        "general_ability", "verbal_aptitude", "numerical_aptitude",
        "spatial_aptitude", "perceptual_aptitude", "manual_dexterity"
    ]

    # Compute overall performance
    df["overall_performance"] = df[aptitude_cols].mean(axis=1)

    return df


# ----------------- PLOT WITH K-MEANS (GENDER) -----------------
def plot_gender_kmeans(df, score_col):
    # Prepare data
    data = df[[score_col, "overall_performance", "sex_code"]].dropna()
    X = data[[score_col, "overall_performance"]].values

    # KMeans
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X)
    centroids = kmeans.cluster_centers_

    # Plot
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(
        data[score_col], data["overall_performance"],
        c=clusters, cmap="viridis", alpha=0.7, s=60
    )
    plt.scatter(centroids[:, 0], centroids[:, 1], c="red", marker="X", s=200, label="Centroids")

    plt.title(f"{score_col.replace('_', ' ').title()} vs Overall Performance (by Gender)", fontweight='bold')
    plt.xlabel(score_col.replace("_", " ").title())
    plt.ylabel("Overall Performance")
    plt.ylim(0, 100)
    plt.yticks(np.arange(0, 101, 10))
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()

    # Add gender color reference (optional)
    for sex_val, color, label in [(0, 'blue', 'Male'), (1, 'orange', 'Female')]:
        subset = data[data["sex_code"] == sex_val]
        if not subset.empty:
            plt.scatter([], [], color='gray', label=f"{label} (Cluster colors vary)")

    st.pyplot(plt)
    plt.clf()


# ----------------- PLOT WITH K-MEANS (COURSE) -----------------
def plot_course_kmeans(df, score_col):
    # Prepare data
    data = df[[score_col, "overall_performance", "course_code"]].dropna()
    X = data[[score_col, "overall_performance"]].values

    # KMeans
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X)
    centroids = kmeans.cluster_centers_

    # Plot
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(
        data[score_col], data["overall_performance"],
        c=clusters, cmap="tab10", alpha=0.7, s=60
    )
    plt.scatter(centroids[:, 0], centroids[:, 1], c="red", marker="X", s=200, label="Centroids")

    plt.title(f"{score_col.replace('_', ' ').title()} vs Overall Performance (by Course)", fontweight='bold')
    plt.xlabel(score_col.replace("_", " ").title())
    plt.ylabel("Overall Performance")
    plt.ylim(0, 100)
    plt.yticks(np.arange(0, 101, 10))
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()

    # Course labels for reference
    course_labels = {
        -1: "Unknown",
        0: "BSEd", 1: "BEEd", 2: "BPEd", 3: "BSBA", 4: "BSMath",
        5: "AB English", 6: "AB Psych", 7: "AB SocSci",
        8: "BS Entrep", 9: "BSIT"
    }
    # Show course mapping in caption (since color = cluster, not course)
    st.caption(
        "Note: Colors represent K-Means clusters (not courses). Course info used only for context in data selection.")

    st.pyplot(plt)
    plt.clf()


# ----------------- STREAMLIT APP -----------------
st.set_page_config(page_title="ICAT Analysis Dashboard", layout="wide")
st.title("📊 ICAT Student Aptitude Analysis (K-Means Clustering)")

df = load_data()
if df.empty:
    st.warning("No student records found in the database.")
    st.stop()

st.success(f"✅ Loaded {df.shape[0]} student records")

# Define aptitude scores
scores = [
    "general_ability", "verbal_aptitude", "numerical_aptitude",
    "spatial_aptitude", "perceptual_aptitude", "manual_dexterity"
]

# --- Gender-Based Plots ---
st.header("🔹 Gender-Based K-Means Clustering (Aptitude vs Overall Performance)")
for score in scores:
    plot_gender_kmeans(df, score)

# --- Course-Based Plots ---
st.header("🔹 Course-Based K-Means Clustering (Aptitude vs Overall Performance)")
for score in scores:
    plot_course_kmeans(df, score)