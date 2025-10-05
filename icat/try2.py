import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np

# ----------------- PAGE SETUP -----------------
st.set_page_config(page_title="ICAT Paired Analysis", layout="wide")
st.title("📊 ICAT Student Aptitude: Paired Cluster vs Domain Views")

# ----------------- AUTHENTICATION -----------------
PASSWORD = "icat2025password"

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

# ----------------- FILE UPLOAD -----------------
st.sidebar.header("📂 Upload Student Data Files")
uploaded_files = st.sidebar.file_uploader(
    "Upload one or more CSV or Excel files",
    type=["csv", "xlsx"],
    accept_multiple_files=True
)

if not uploaded_files:
    st.info("👆 Please upload at least one file to start.")
    st.stop()

# ----------------- ANALYSIS LOOP -----------------
for uploaded_file in uploaded_files:
    st.divider()
    st.header(f"📄 File: {uploaded_file.name}")

    # ----------------- LOAD FILE -----------------
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
    except Exception as e:
        st.error(f"❌ Error reading {uploaded_file.name}: {e}")
        continue

    if df.empty:
        st.warning(f"⚠️ {uploaded_file.name} is empty.")
        continue

    # ----------------- DATA CLEANING -----------------
    if "sex" not in df.columns or "course" not in df.columns:
        st.error(f"❌ {uploaded_file.name} missing columns: 'sex' or 'course'")
        continue

    aptitude_cols = [
        "general_ability", "verbal_aptitude", "numerical_aptitude",
        "spatial_aptitude", "perceptual_aptitude", "manual_dexterity"
    ]
    missing_cols = [col for col in aptitude_cols if col not in df.columns]
    if missing_cols:
        st.error(f"❌ {uploaded_file.name} missing aptitude columns: {', '.join(missing_cols)}")
        continue

    df["sex"] = df["sex"].astype(str).str.lower().str.strip()
    sex_map = {"male": 0, "female": 1}
    df["sex_code"] = df["sex"].map(sex_map)

    df["course_clean"] = df["course"].astype(str).str.upper().str.strip()
    course_map = {
        "BSED": 0, "BSED-ENGLISH": 0, "BSED-FILIPINO": 0, "BSED-MATH": 0,
        "BSED-SCIENCE": 0, "BSED-SOCIAL": 0, "BSED/BAEL": 0,
        "EDUC": 0, "EDUC-": 0, "BACHELOR OF SECONDARY EDUCATION": 0,
        "BEED": 1, "BACHELOR OF ELEMENTARY EDUCATION": 1,
        "BPED": 2, "BPA": 2, "BACHELOR OF PHYSICAL EDUCATION": 2,
        "BSBA": 3, "BSBA-FM": 3, "BSBA-MARKETING": 3,
        "BACHELOR OF SCIENCE IN BUSINESS ADMINISTRATION": 3,
        "BS MATH": 4, "BS MATHEMATICS": 4,
        "BACHELOR OF SCIENCE IN MATHEMATICS": 4,
        "BAEL": 5, "BACHELOR OF ARTS IN ENGLISH LANGUAGE": 5,
        "BAP": 6, "PSYCHOLOGY": 6, "BACHELOR OF ARTS IN PSYCHOLOGY": 6,
        "BASS": 7, "BACHELOR OF ARTS IN SOCIAL SCIENCE": 7,
        "BS ENTREP": 8, "BS ENTREPRENEURSHIP": 8,
        "BACHELOR OF SCIENCE IN ENTREPRENEURSHIP": 8,
        "BSIT": 9, "BACHELOR OF SCIENCE IN INFORMATION TECHNOLOGY": 9,
    }
    df["course_code"] = df["course_clean"].map(course_map).fillna(-1).astype(int)
    df["overall_performance"] = df[aptitude_cols].mean(axis=1)

    df_valid = df.dropna(subset=aptitude_cols + ["overall_performance"]).copy()
    if df_valid.empty:
        st.error(f"❌ No valid aptitude data found in {uploaded_file.name}.")
        continue

    # ----------------- COLOR CONFIG -----------------
    course_labels = {-1: "Unknown", 0: "BSEd / Educ", 1: "BEEd", 2: "BPEd", 3: "BSBA",
                     4: "BS Math", 5: "BA English", 6: "BA Psychology", 7: "BA Social Science",
                     8: "BS Entrep", 9: "BSIT"}

    tab20_colors = plt.cm.tab20(np.linspace(0, 1, 20))
    selected_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 15]
    unique_colors = [tab20_colors[i] for i in selected_indices]
    course_codes_sorted = sorted(course_labels.keys())
    course_color_map = {code: unique_colors[i] for i, code in enumerate(course_codes_sorted)}

    # ----------------- PLOTTING FUNCTIONS -----------------
    def plot_cluster_view(data, score_col, title_suffix=""):
        data_plot = data[[score_col, "overall_performance"]].dropna()
        if data_plot.empty:
            st.warning(f"No data for {score_col} (Cluster View)")
            return
        X = data_plot.values
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(X)
        centroids = kmeans.cluster_centers_

        plt.figure(figsize=(9, 5.5))
        plt.scatter(data_plot[score_col], data_plot["overall_performance"],
                    c=clusters, cmap="viridis", alpha=0.7, s=60)
        plt.scatter(centroids[:, 0], centroids[:, 1], c="red", marker="X", s=250, label="Centroids")
        plt.title(f"{score_col.replace('_', ' ').title()} vs Overall Performance\n(Cluster View {title_suffix})", fontweight='bold')
        plt.xlabel(score_col.replace("_", " ").title())
        plt.ylabel("Overall Performance")
        plt.ylim(0, 100)
        plt.yticks(np.arange(0, 101, 10))
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.legend()
        st.pyplot(plt)
        plt.clf()

    def plot_gender_view(score_col):
        data = df_valid[[score_col, "overall_performance", "sex_code"]].dropna()
        if data.empty:
            st.warning(f"No valid gender data for {score_col}")
            return
        X = data[[score_col, "overall_performance"]].values
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10).fit(X)
        centroids = kmeans.cluster_centers_

        color_map = {0: "steelblue", 1: "indianred"}
        colors = data["sex_code"].map(color_map)

        plt.figure(figsize=(9, 5.5))
        plt.scatter(data[score_col], data["overall_performance"], c=colors, alpha=0.8, s=60, edgecolor='k', linewidth=0.3)
        plt.scatter(centroids[:, 0], centroids[:, 1], c="red", marker="X", s=250, edgecolor='white', linewidth=1.5)
        plt.title(f"{score_col.replace('_', ' ').title()} vs Overall Performance (by Gender)", fontweight='bold')
        plt.xlabel(score_col.replace("_", " ").title())
        plt.ylabel("Overall Performance")
        plt.ylim(0, 100)
        plt.yticks(np.arange(0, 101, 10))
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.legend(
            handles=[
                plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='steelblue', markersize=8),
                plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='indianred', markersize=8),
                plt.Line2D([0], [0], marker='X', color='w', markerfacecolor='red', markersize=10)
            ],
            labels=['Male', 'Female', 'Centroids']
        )
        st.pyplot(plt)
        plt.clf()

    def plot_course_view(score_col):
        data = df_valid[[score_col, "overall_performance", "course_code"]].dropna()
        if data.empty:
            st.warning(f"No valid course data for {score_col}")
            return
        X = data[[score_col, "overall_performance"]].values
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10).fit(X)
        centroids = kmeans.cluster_centers_

        plt.figure(figsize=(10, 6))
        for code in course_codes_sorted:
            subset = data[data["course_code"] == code]
            if not subset.empty:
                plt.scatter(
                    subset[score_col], subset["overall_performance"],
                    color=course_color_map[code],
                    label=course_labels[code],
                    alpha=0.8, s=60, edgecolor='k', linewidth=0.3
                )
        plt.scatter(centroids[:, 0], centroids[:, 1], c="red", marker="X", s=250, edgecolor='white', linewidth=1.5)
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

    # ----------------- DISPLAY CLUSTER GRAPHS -----------------
    st.markdown("### 🔍 For each aptitude, compare: **(Top)** data-driven clusters vs **(Bottom)** real-world groups.")
    for i, col in enumerate(aptitude_cols, 1):
        st.markdown(f"#### 📌 {i}. {col.replace('_', ' ').title()}")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Cluster View (Gender Context)**")
            plot_cluster_view(df_valid[df_valid["sex_code"].isin([0, 1])], col, "(Gender Context)")
        with c2:
            st.markdown("**Domain View (Gender)**")
            plot_gender_view(col)
        st.markdown("**Cluster View (Course Context)**")
        plot_cluster_view(df_valid, col, "(Course Context)")
        st.markdown("**Domain View (Course)**")
        plot_course_view(col)
        st.markdown("---")

    st.success(f"✅ Finished analyzing {uploaded_file.name}")

st.caption("💡 Tip: Each file is processed independently. Check how K-Means centroids (red X) align with gender or course clusters.")
